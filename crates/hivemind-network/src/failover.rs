use crate::pipeline::Pipeline;
use crate::NetworkService;
use hivemind_core::{HivemindError, NodeId, Result, Tensor};
use std::time::Duration;
use tracing::warn;

/// Per-hop failure detection policy used by the sending node.
#[derive(Debug, Clone)]
pub struct FailoverPolicy {
    /// How long to wait for a hop's activation response before declaring
    /// the primary failed. Kept tight: a dropped node should cost the user
    /// one delayed token, not a stalled session.
    pub hop_timeout: Duration,
    /// Retries against the same primary before promoting a standby, to
    /// absorb transient blips without burning a warm standby.
    pub retries_before_promote: u32,
}

impl Default for FailoverPolicy {
    fn default() -> Self {
        Self {
            hop_timeout: Duration::from_millis(1000),
            retries_before_promote: 1,
        }
    }
}

/// Record of a completed local failover, reported to the orchestrator
/// (`RoutingService.ReportFailover`) so it can refill the standby pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Promotion {
    pub failed: NodeId,
    pub promoted: NodeId,
    pub position: usize,
}

/// Promotes the first standby of the slot `failed` is primary for.
///
/// The promoted node becomes the slot's primary and is removed from every
/// slot's standby pool (it can no longer back other slots). Returns an error
/// when the node has no slot or the slot's standby pool is empty — the
/// caller should fall back to full pipeline re-assembly.
pub fn promote_standby(pipeline: &mut Pipeline, failed: NodeId) -> Result<Promotion> {
    let slot = pipeline
        .slots
        .iter_mut()
        .find(|s| s.node_id == failed)
        .ok_or_else(|| {
            HivemindError::Pipeline(format!("node {failed} is not a primary in this pipeline"))
        })?;

    if slot.standbys.is_empty() {
        return Err(HivemindError::Pipeline(format!(
            "no standby available for slot {} ({})",
            slot.position, slot.layer_range
        )));
    }

    let promoted = slot.standbys.remove(0);
    slot.node_id = promoted;
    let position = slot.position;

    for s in &mut pipeline.slots {
        s.standbys.retain(|&n| n != promoted);
    }

    Ok(Promotion { failed, promoted, position })
}

/// Outcome of a hop that survived via failover.
#[derive(Debug)]
pub struct HopResult {
    pub output: Tensor,
    /// Promotions performed while completing this hop (empty on the happy path).
    pub promotions: Vec<Promotion>,
}

/// Forwards an activation to the primary of `position`, promoting standbys on
/// timeout or error until the hop succeeds or the slot runs out of standbys.
///
/// This is the upstream node's hot-path failover: detection and rerouting are
/// local decisions, with no orchestrator round-trip. Any promotions performed
/// are returned so the caller can report them via `ReportFailover`.
pub async fn forward_with_failover<N: NetworkService + ?Sized>(
    net: &N,
    pipeline: &mut Pipeline,
    position: usize,
    tensor: &Tensor,
    policy: &FailoverPolicy,
) -> Result<HopResult> {
    let pipeline_id = pipeline.id;
    let mut promotions = Vec::new();

    loop {
        let target = pipeline
            .slots
            .get(position)
            .ok_or_else(|| {
                HivemindError::Pipeline(format!("pipeline has no slot at position {position}"))
            })?
            .node_id;

        for attempt in 0..=policy.retries_before_promote {
            let fwd = net.forward_activations(target, pipeline_id, tensor.clone());
            match tokio::time::timeout(policy.hop_timeout, fwd).await {
                Ok(Ok(output)) => return Ok(HopResult { output, promotions }),
                Ok(Err(e)) => {
                    warn!(%target, attempt, error = %e, "hop failed");
                }
                Err(_) => {
                    warn!(%target, attempt, timeout_ms = policy.hop_timeout.as_millis() as u64, "hop timed out");
                }
            }
        }

        match promote_standby(pipeline, target) {
            Ok(p) => {
                warn!(failed = %p.failed, promoted = %p.promoted, position = p.position, "promoted standby");
                promotions.push(p);
            }
            Err(e) => {
                return Err(HivemindError::Pipeline(format!(
                    "hop {position} failed and no standby remains ({e}); pipeline must be re-assembled"
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{assemble_pipeline, PipelineSpec};
    use crate::peer::PeerInfo;
    use async_trait::async_trait;
    use hivemind_core::{LayerRange, PipelineId};
    use std::collections::HashSet;
    use std::sync::Mutex;
    use uuid::Uuid;

    fn peer(start: u32, end: u32, latency: u32) -> PeerInfo {
        PeerInfo {
            node_id: Uuid::new_v4(),
            peer_id: None,
            addrs: vec![],
            layer_range: LayerRange::new(start, end),
            latency_ms: Some(latency),
            reputation: 80,
            survival: 1.0,
        }
    }

    fn two_slot_pipeline_with_standbys(peers: &[PeerInfo]) -> Pipeline {
        let spec = PipelineSpec::new(80, 500);
        assemble_pipeline(&spec, peers, Uuid::new_v4()).unwrap()
    }

    /// Mock network where a configurable set of node IDs is unreachable.
    struct FlakyNetwork {
        dead: Mutex<HashSet<NodeId>>,
        calls: Mutex<Vec<NodeId>>,
    }

    impl FlakyNetwork {
        fn new(dead: impl IntoIterator<Item = NodeId>) -> Self {
            Self {
                dead: Mutex::new(dead.into_iter().collect()),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl NetworkService for FlakyNetwork {
        async fn discover_peers(&self, _range: LayerRange) -> Result<Vec<PeerInfo>> {
            Ok(vec![])
        }

        async fn assemble_pipeline(&self, _requester: NodeId, _total: u32) -> Result<Pipeline> {
            unimplemented!()
        }

        async fn forward_activations(
            &self,
            destination: NodeId,
            _pipeline_id: PipelineId,
            tensor: Tensor,
        ) -> Result<Tensor> {
            self.calls.lock().unwrap().push(destination);
            if self.dead.lock().unwrap().contains(&destination) {
                Err(HivemindError::Network("connection refused".into()))
            } else {
                Ok(Tensor::zeros(tensor.shape))
            }
        }

        async fn report_health(&self, _node_id: NodeId) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn happy_path_no_promotions() {
        let peers = vec![peer(0, 40, 10), peer(40, 80, 15), peer(0, 40, 20)];
        let mut p = two_slot_pipeline_with_standbys(&peers);
        let net = FlakyNetwork::new([]);
        let result = forward_with_failover(
            &net,
            &mut p,
            0,
            &Tensor::zeros(vec![1, 4]),
            &FailoverPolicy::default(),
        )
        .await
        .unwrap();
        assert!(result.promotions.is_empty());
    }

    #[tokio::test]
    async fn dead_primary_promotes_standby_and_succeeds() {
        let peers = vec![peer(0, 40, 10), peer(40, 80, 15), peer(0, 40, 20)];
        let mut p = two_slot_pipeline_with_standbys(&peers);
        let primary = p.slots[0].node_id;
        let standby = p.slots[0].standbys[0];

        let net = FlakyNetwork::new([primary]);
        let result = forward_with_failover(
            &net,
            &mut p,
            0,
            &Tensor::zeros(vec![1, 4]),
            &FailoverPolicy::default(),
        )
        .await
        .unwrap();

        assert_eq!(result.promotions.len(), 1);
        assert_eq!(result.promotions[0].failed, primary);
        assert_eq!(result.promotions[0].promoted, standby);
        assert_eq!(p.slots[0].node_id, standby);
        // Retried the primary once before promoting (default policy).
        assert_eq!(net.calls.lock().unwrap().iter().filter(|&&n| n == primary).count(), 2);
    }

    #[tokio::test]
    async fn exhausted_standbys_is_an_error() {
        let peers = vec![peer(0, 40, 10), peer(40, 80, 15), peer(0, 40, 20)];
        let mut p = two_slot_pipeline_with_standbys(&peers);
        let primary = p.slots[0].node_id;
        let standby = p.slots[0].standbys[0];

        let net = FlakyNetwork::new([primary, standby]);
        let err = forward_with_failover(
            &net,
            &mut p,
            0,
            &Tensor::zeros(vec![1, 4]),
            &FailoverPolicy::default(),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("re-assembled"));
    }

    #[test]
    fn promoted_node_removed_from_all_standby_pools() {
        // One node backs both slots; promoting it into slot 0 must remove it
        // from slot 1's pool too.
        use hivemind_core::PipelineSlot;
        let backup = Uuid::new_v4();
        let mut p = Pipeline {
            id: Uuid::new_v4(),
            slots: vec![
                PipelineSlot {
                    node_id: Uuid::new_v4(),
                    layer_range: LayerRange::new(0, 40),
                    position: 0,
                    standbys: vec![backup],
                },
                PipelineSlot {
                    node_id: Uuid::new_v4(),
                    layer_range: LayerRange::new(40, 80),
                    position: 1,
                    standbys: vec![backup],
                },
            ],
            requester: Uuid::new_v4(),
        };

        let failed = p.slots[0].node_id;
        let promo = promote_standby(&mut p, failed).unwrap();
        assert_eq!(promo.promoted, backup);
        assert_eq!(p.slots[0].node_id, backup);
        assert!(p.slots[1].standbys.is_empty());
    }
}
