use hivemind_core::{NodeId, PipelineId, PipelineSlot};
use crate::peer::PeerInfo;
use uuid::Uuid;

/// Latency assumed for a peer we have never measured, in milliseconds.
const UNMEASURED_LATENCY_MS: u32 = 50;

/// Parameters for assembling a pipeline.
#[derive(Debug, Clone)]
pub struct PipelineSpec {
    pub total_layers: u32,
    /// Maximum acceptable end-to-end latency in milliseconds.
    pub max_latency_ms: u32,
    /// Warm standbys to assign per slot (0 disables redundancy).
    pub standbys_per_slot: usize,
}

impl PipelineSpec {
    pub fn new(total_layers: u32, max_latency_ms: u32) -> Self {
        Self {
            total_layers,
            max_latency_ms,
            standbys_per_slot: 2,
        }
    }
}

/// An assembled, ordered chain of pipeline slots covering all model layers.
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub id: PipelineId,
    pub slots: Vec<PipelineSlot>,
    pub requester: NodeId,
}

impl Pipeline {
    /// Returns true when every layer `[0, total_layers)` is covered with no gaps.
    pub fn is_complete(&self, total_layers: u32) -> bool {
        let mut covered: Vec<bool> = vec![false; total_layers as usize];
        for slot in &self.slots {
            for l in slot.layer_range.start..slot.layer_range.end {
                if (l as usize) < covered.len() {
                    covered[l as usize] = true;
                }
            }
        }
        covered.iter().all(|&c| c)
    }

    /// The slot a given node is the primary for, if any.
    pub fn slot_of(&self, node_id: NodeId) -> Option<&PipelineSlot> {
        self.slots.iter().find(|s| s.node_id == node_id)
    }
}

/// Assembles an ordered pipeline from a candidate peer list.
///
/// Greedy interval covering: walk from layer 0, and at each uncovered layer
/// pick the candidate that reaches furthest (fewest hops), breaking ties by
/// [`placement_score`] — which prefers peers likely to outlive the session
/// over marginally faster ones. Each chosen slot is then assigned up to
/// `spec.standbys_per_slot` warm standbys: non-primary peers whose loaded
/// range fully covers the slot, in score order.
pub fn assemble_pipeline(
    spec: &PipelineSpec,
    candidates: &[PeerInfo],
    requester: NodeId,
) -> hivemind_core::Result<Pipeline> {
    if candidates.is_empty() {
        return Err(hivemind_core::HivemindError::Pipeline(
            "no candidate peers available".into(),
        ));
    }

    let mut slots: Vec<PipelineSlot> = Vec::new();
    let mut primaries: Vec<NodeId> = Vec::new();
    let mut cursor: u32 = 0;
    let mut estimated_latency_ms: u32 = 0;

    while cursor < spec.total_layers {
        let chosen = candidates
            .iter()
            .filter(|p| {
                p.layer_range.contains_layer(cursor) && !primaries.contains(&p.node_id)
            })
            .max_by(|a, b| {
                a.layer_range
                    .end
                    .cmp(&b.layer_range.end)
                    .then_with(|| {
                        placement_score(a)
                            .partial_cmp(&placement_score(b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .ok_or_else(|| {
                hivemind_core::HivemindError::Pipeline(format!(
                    "no available peer serves layer {cursor}"
                ))
            })?;

        let slot_end = chosen.layer_range.end.min(spec.total_layers);
        slots.push(PipelineSlot {
            node_id: chosen.node_id,
            layer_range: hivemind_core::LayerRange::new(cursor, slot_end),
            position: slots.len(),
            standbys: Vec::new(),
        });
        primaries.push(chosen.node_id);
        estimated_latency_ms = estimated_latency_ms.saturating_add(latency_of(chosen));
        cursor = slot_end;
    }

    if estimated_latency_ms > spec.max_latency_ms {
        return Err(hivemind_core::HivemindError::Pipeline(format!(
            "best pipeline latency {estimated_latency_ms}ms exceeds budget {}ms",
            spec.max_latency_ms
        )));
    }

    // Assign warm standbys: peers that fully cover a slot's range and are not
    // primaries anywhere in this pipeline. The same peer may back several
    // slots — it is only promoted into one.
    for slot in &mut slots {
        let mut backups: Vec<&PeerInfo> = candidates
            .iter()
            .filter(|p| {
                !primaries.contains(&p.node_id)
                    && p.layer_range.start <= slot.layer_range.start
                    && p.layer_range.end >= slot.layer_range.end
            })
            .collect();
        backups.sort_by(|a, b| {
            placement_score(b)
                .partial_cmp(&placement_score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        slot.standbys = backups
            .into_iter()
            .take(spec.standbys_per_slot)
            .map(|p| p.node_id)
            .collect();
    }

    Ok(Pipeline {
        id: Uuid::new_v4(),
        slots,
        requester,
    })
}

fn latency_of(peer: &PeerInfo) -> u32 {
    peer.latency_ms.unwrap_or(UNMEASURED_LATENCY_MS)
}

/// Desirability of a peer for a slot, combining survival, latency, and
/// reputation. Survival is squared so a node likely to vanish mid-session
/// loses to a stable one even at noticeably higher latency: a dropped node
/// costs a failover and a KV rebuild, a slow one costs milliseconds.
fn placement_score(peer: &PeerInfo) -> f64 {
    let survival = (peer.survival as f64).clamp(0.0, 1.0);
    let reputation_weight = 0.5 + peer.reputation as f64 / 200.0;
    survival * survival * reputation_weight / (latency_of(peer) as f64 + 10.0)
}

/// Checks whether a set of peers can collectively cover all `total_layers`.
pub fn can_cover_layers(peers: &[PeerInfo], total_layers: u32) -> bool {
    let mut covered: Vec<bool> = vec![false; total_layers as usize];
    for p in peers {
        for l in p.layer_range.start..p.layer_range.end.min(total_layers) {
            covered[l as usize] = true;
        }
    }
    covered.iter().all(|&c| c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hivemind_core::LayerRange;

    fn peer(start: u32, end: u32, latency: u32, reputation: u8) -> PeerInfo {
        PeerInfo {
            node_id: Uuid::new_v4(),
            peer_id: None,
            addrs: vec![],
            layer_range: LayerRange::new(start, end),
            latency_ms: Some(latency),
            reputation,
            survival: 1.0,
        }
    }

    #[test]
    fn assembles_minimal_chain_and_covers_all_layers() {
        let peers = vec![
            peer(0, 40, 10, 80),
            peer(0, 20, 5, 90), // shorter reach — should lose to 0..40
            peer(40, 80, 15, 85),
        ];
        let spec = PipelineSpec::new(80, 500);
        let p = assemble_pipeline(&spec, &peers, Uuid::new_v4()).unwrap();
        assert!(p.is_complete(80));
        assert_eq!(p.slots.len(), 2);
        assert_eq!(p.slots[0].node_id, peers[0].node_id);
        assert_eq!(p.slots[1].node_id, peers[2].node_id);
    }

    #[test]
    fn assigns_standbys_that_fully_cover_each_slot() {
        let peers = vec![
            peer(0, 40, 10, 80),
            peer(40, 80, 15, 85),
            peer(0, 40, 20, 70),  // standby for slot 0
            peer(30, 80, 25, 75), // standby for slot 1 only (doesn't cover 0..40)
        ];
        let spec = PipelineSpec::new(80, 500);
        let p = assemble_pipeline(&spec, &peers, Uuid::new_v4()).unwrap();
        assert_eq!(p.slots[0].standbys, vec![peers[2].node_id]);
        assert_eq!(p.slots[1].standbys, vec![peers[3].node_id]);
    }

    #[test]
    fn standbys_capped_and_preference_ordered() {
        let peers = vec![
            peer(0, 80, 10, 80),
            peer(0, 80, 30, 90),
            peer(0, 80, 20, 90),
            peer(0, 80, 40, 50),
        ];
        let spec = PipelineSpec {
            total_layers: 80,
            max_latency_ms: 500,
            standbys_per_slot: 2,
        };
        let p = assemble_pipeline(&spec, &peers, Uuid::new_v4()).unwrap();
        assert_eq!(p.slots.len(), 1);
        // Lowest-latency standbys first, capped at 2.
        assert_eq!(p.slots[0].standbys, vec![peers[2].node_id, peers[1].node_id]);
    }

    #[test]
    fn stable_node_beats_marginally_faster_flaky_one() {
        let mut flaky = peer(0, 80, 10, 90);
        flaky.survival = 0.4; // laptop that joined three minutes ago
        let stable = peer(0, 80, 35, 90); // desktop, online all day
        let peers = vec![flaky, stable.clone()];
        let spec = PipelineSpec::new(80, 500);
        let p = assemble_pipeline(&spec, &peers, Uuid::new_v4()).unwrap();
        assert_eq!(p.slots[0].node_id, stable.node_id);
    }

    #[test]
    fn errors_on_coverage_gap() {
        let peers = vec![peer(0, 30, 10, 80), peer(50, 80, 15, 85)];
        let spec = PipelineSpec::new(80, 500);
        let err = assemble_pipeline(&spec, &peers, Uuid::new_v4()).unwrap_err();
        assert!(err.to_string().contains("layer 30"));
    }

    #[test]
    fn errors_when_latency_budget_exceeded() {
        let peers = vec![peer(0, 40, 300, 80), peer(40, 80, 300, 85)];
        let spec = PipelineSpec::new(80, 500);
        assert!(assemble_pipeline(&spec, &peers, Uuid::new_v4()).is_err());
    }
}
