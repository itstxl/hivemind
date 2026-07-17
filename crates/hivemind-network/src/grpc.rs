//! Client-driven pipeline session over real gRPC.
//!
//! The requester assembles a pipeline via the orchestrator, then drives each
//! generation step hop by hop: token ids go to the first shard (which
//! embeds), boundary activations flow through the middle shards, and the
//! last shard returns logits for the client to sample. Client-driven routing
//! makes the client the "upstream node" for every hop, so failover and
//! KV-replay recovery are entirely client-local:
//!
//! - A hop that times out or refuses (dead, draining, overloaded) gets its
//!   warm standby promoted and the orchestrator notified (`ReportFailover`).
//! - A hop that answers `FAILED_PRECONDITION` (kv gap — a fresh standby with
//!   cold KV) gets its full input history replayed in one prefill: the
//!   client records every row it has ever sent to each hop, in f32, so a
//!   rebuilt shard continues the sequence bit-exactly.

use crate::failover::{promote_standby, FailoverPolicy};
use crate::pipeline::Pipeline;
use hivemind_core::{HivemindError, NodeId, Result, Tensor};
use hivemind_proto::activations::activation_service_client::ActivationServiceClient;
use hivemind_proto::activations::ActivationRequest;
use hivemind_proto::routing::routing_service_client::RoutingServiceClient;
use hivemind_proto::routing::{AssembleRequest, FailoverReport, PipelineAssignment};
use std::collections::HashMap;
use tonic::transport::Channel;
use tonic::Code;
use tracing::{info, warn};
use uuid::Uuid;

/// Input rows previously sent to one hop, kept verbatim for replay.
#[derive(Default, Clone)]
struct RowHistory {
    width: usize,
    data: Vec<f32>,
}

impl RowHistory {
    fn extend_rows(&mut self, t: &Tensor) {
        if self.width == 0 {
            self.width = t.shape.last().copied().unwrap_or(0);
        }
        self.data.extend_from_slice(&t.data);
    }

    fn as_tensor(&self) -> Option<Tensor> {
        if self.width == 0 || self.data.is_empty() {
            return None;
        }
        Some(Tensor::new(
            self.data.clone(),
            vec![self.data.len() / self.width, self.width],
        ))
    }
}

/// A live generation session against an assembled pipeline.
pub struct PipelineSession {
    routing: RoutingServiceClient<Channel>,
    pipeline: Pipeline,
    addrs: HashMap<NodeId, String>,
    tokens: Vec<u32>,
    /// Sequence positions fully processed by every hop.
    committed: u32,
    /// Per-hop input history for replay; index matches slot position.
    /// Position 0 needs none — the first shard re-embeds from token ids.
    histories: Vec<RowHistory>,
    policy: FailoverPolicy,
}

impl PipelineSession {
    /// Assembles a pipeline through the orchestrator and opens a session.
    pub async fn connect(
        orchestrator_url: &str,
        model_name: &str,
        prompt_tokens: Vec<u32>,
    ) -> Result<Self> {
        if prompt_tokens.is_empty() {
            return Err(HivemindError::Pipeline("empty prompt".into()));
        }
        let mut routing = RoutingServiceClient::connect(orchestrator_url.to_string())
            .await
            .map_err(|e| HivemindError::Network(format!("orchestrator connect: {e}")))?;

        let requester = Uuid::new_v4();
        let resp = routing
            .assemble(AssembleRequest {
                requester_id: requester.to_string(),
                model_name: model_name.to_string(),
                max_latency_ms: 10_000,
            })
            .await
            .map_err(|e| HivemindError::Pipeline(format!("assemble: {e}")))?
            .into_inner();
        let assignment = resp
            .assignment
            .ok_or_else(|| HivemindError::Pipeline("empty assignment".into()))?;

        let (pipeline, addrs) = Self::import_assignment(&assignment, requester)?;
        info!(
            pipeline = %pipeline.id,
            hops = pipeline.slots.len(),
            "pipeline assembled"
        );
        let n_slots = pipeline.slots.len();
        Ok(Self {
            routing,
            pipeline,
            addrs,
            tokens: prompt_tokens,
            committed: 0,
            histories: vec![RowHistory::default(); n_slots],
            policy: FailoverPolicy::default(),
        })
    }

    fn import_assignment(
        a: &PipelineAssignment,
        requester: NodeId,
    ) -> Result<(Pipeline, HashMap<NodeId, String>)> {
        let mut addrs = HashMap::new();
        let mut slots = Vec::new();
        let parse = |raw: &str| -> Result<NodeId> {
            raw.parse()
                .map_err(|_| HivemindError::Pipeline(format!("bad node id {raw}")))
        };
        let mut sorted = a.slots.clone();
        sorted.sort_by_key(|s| s.position);
        for s in &sorted {
            let node_id = parse(&s.node_id)?;
            addrs.insert(node_id, s.peer_addr.clone());
            let mut standbys = Vec::new();
            for sb in &s.standbys {
                let id = parse(&sb.node_id)?;
                addrs.insert(id, sb.peer_addr.clone());
                standbys.push(id);
            }
            slots.push(hivemind_core::PipelineSlot {
                node_id,
                layer_range: hivemind_core::LayerRange::new(s.layer_start, s.layer_end),
                position: s.position as usize,
                standbys,
            });
        }
        let id = a
            .pipeline_id
            .parse()
            .map_err(|_| HivemindError::Pipeline("bad pipeline id".into()))?;
        Ok((Pipeline { id, slots, requester }, addrs))
    }

    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Appends a sampled token; it is processed on the next `step`.
    pub fn push_token(&mut self, token: u32) {
        self.tokens.push(token);
    }

    /// Runs all uncommitted positions through every hop and returns the
    /// final shard's logits (`[n_new, vocab]`). Survives node failures via
    /// standby promotion + history replay; errors only when a hop has no
    /// standbys left.
    pub async fn step(&mut self) -> Result<Tensor> {
        let seq_len = self.tokens.len() as u32;
        if seq_len <= self.committed {
            return Err(HivemindError::Pipeline("no new tokens to process".into()));
        }
        let n_new = (seq_len - self.committed) as usize;

        let mut boundary: Option<Tensor> = None;
        let mut logits: Option<Tensor> = None;
        for pos in 0..self.pipeline.slots.len() {
            let out = self.forward_hop(pos, boundary.as_ref(), seq_len).await?;
            // Record what this hop consumed, for future replays.
            if pos > 0 {
                if let Some(rows) = &boundary {
                    self.histories[pos].extend_rows(rows);
                }
            }
            let is_last = pos + 1 == self.pipeline.slots.len();
            if is_last {
                logits = Some(out);
            } else {
                boundary = Some(out);
            }
        }
        self.committed = seq_len;
        logits.ok_or_else(|| {
            HivemindError::Pipeline("pipeline produced no logits — last shard must own the final layer".into())
        })
        .inspect(|l| {
            debug_assert_eq!(l.shape.first().copied().unwrap_or(0), n_new);
        })
    }

    /// Sends the current rows to the hop's primary, falling back to history
    /// replay on kv gaps and standby promotion on transport failures.
    async fn forward_hop(
        &mut self,
        pos: usize,
        current: Option<&Tensor>,
        seq_len: u32,
    ) -> Result<Tensor> {
        let n_new = (seq_len - self.committed) as usize;
        let mut retries_left = self.policy.retries_before_promote;
        loop {
            let target = self.pipeline.slots[pos].node_id;
            let outcome = self.try_forward(target, current, seq_len).await;
            let outcome = match outcome {
                // Cold KV on the serving node (fresh standby or restart):
                // replay everything we ever sent this hop plus the current
                // rows, then keep only the rows the chain still needs.
                Err(HopError::KvGap) if pos > 0 => {
                    let replay = self.build_replay(pos, current)?;
                    warn!(hop = pos, node = %target, rows = replay.shape[0], "replaying history to cold shard");
                    self.try_forward(target, Some(&replay), seq_len).await
                }
                other => other,
            };
            match outcome {
                Ok(t) => return Ok(last_rows(t, n_new)),
                Err(HopError::Fatal(e)) => return Err(e),
                Err(HopError::KvGap) => {
                    // Position 0 embeds from tokens and can always catch up;
                    // a gap here (or a gap right after replay) is a logic bug.
                    return Err(HivemindError::Pipeline(format!(
                        "hop {pos} reported kv gap that replay could not fix"
                    )));
                }
                Err(HopError::Unreachable(e)) => {
                    if retries_left > 0 {
                        retries_left -= 1;
                        continue;
                    }
                    let promo = promote_standby(&mut self.pipeline, target).map_err(|_| {
                        HivemindError::Pipeline(format!(
                            "hop {pos} down ({e}) and no standby remains — session lost"
                        ))
                    })?;
                    warn!(hop = pos, failed = %promo.failed, promoted = %promo.promoted, "promoted standby");
                    self.report_failover(&promo, pos).await;
                    retries_left = self.policy.retries_before_promote;
                }
            }
        }
    }

    async fn try_forward(
        &self,
        target: NodeId,
        input: Option<&Tensor>,
        seq_len: u32,
    ) -> std::result::Result<Tensor, HopError> {
        let addr = self
            .addrs
            .get(&target)
            .cloned()
            .filter(|a| !a.is_empty())
            .ok_or_else(|| HopError::Unreachable(format!("no address for node {target}")))?;

        let attempt = async {
            let mut client = ActivationServiceClient::connect(addr.clone())
                .await
                .map_err(|e| HopError::Unreachable(format!("connect {addr}: {e}")))?;
            let req = ActivationRequest {
                pipeline_id: self.pipeline.id.to_string(),
                sender_position: 0,
                input: input.map(Into::into),
                token_ids: self.tokens.clone(),
                seq_len,
            };
            let resp = client.forward(req).await.map_err(|status| match status.code() {
                Code::FailedPrecondition => HopError::KvGap,
                Code::InvalidArgument | Code::Internal => HopError::Fatal(
                    HivemindError::Pipeline(format!("shard rejected request: {status}")),
                ),
                // Unavailable, ResourceExhausted (draining), DeadlineExceeded…
                _ => HopError::Unreachable(status.to_string()),
            })?;
            let out = resp
                .into_inner()
                .output
                .ok_or_else(|| HopError::Fatal(HivemindError::Network("empty output tensor".into())))?;
            Ok(Tensor::from(&out))
        };
        match tokio::time::timeout(self.policy.hop_timeout, attempt).await {
            Ok(r) => r,
            Err(_) => Err(HopError::Unreachable(format!(
                "hop timed out after {:?}",
                self.policy.hop_timeout
            ))),
        }
    }

    /// Full input matrix for a hop: everything committed plus the in-flight
    /// rows. Feeding this to a cold shard rebuilds its KV bit-exactly.
    fn build_replay(&self, pos: usize, current: Option<&Tensor>) -> Result<Tensor> {
        let mut h = self.histories[pos].clone();
        if let Some(c) = current {
            h.extend_rows(c);
        }
        h.as_tensor().ok_or_else(|| {
            HivemindError::Pipeline(format!("no history to replay for hop {pos}"))
        })
    }

    async fn report_failover(&mut self, promo: &crate::failover::Promotion, pos: usize) {
        let report = FailoverReport {
            pipeline_id: self.pipeline.id.to_string(),
            failed_node_id: promo.failed.to_string(),
            promoted_node_id: promo.promoted.to_string(),
            position: pos as u32,
        };
        match self.routing.report_failover(report).await {
            Ok(resp) => {
                for sb in resp.into_inner().new_standbys {
                    if let Ok(id) = sb.node_id.parse::<NodeId>() {
                        let slot = &mut self.pipeline.slots[pos];
                        if id != slot.node_id && !slot.standbys.contains(&id) {
                            self.addrs.insert(id, sb.peer_addr.clone());
                            slot.standbys.push(id);
                        }
                    }
                }
            }
            Err(e) => warn!(error = %e, "failover report failed (continuing)"),
        }
    }
}

enum HopError {
    /// Node unreachable / refusing — promote a standby.
    Unreachable(String),
    /// Node alive but KV cache cold — replay history.
    KvGap,
    /// Request malformed or shard broken — do not retry.
    Fatal(HivemindError),
}

fn last_rows(t: Tensor, n: usize) -> Tensor {
    let width = t.shape.last().copied().unwrap_or(0);
    let rows = t.shape.first().copied().unwrap_or(0);
    if width == 0 || rows <= n {
        return t;
    }
    let keep = &t.data[(rows - n) * width..];
    Tensor::new(keep.to_vec(), vec![n, width])
}

/// Greedy argmax over the last row of a `[n, vocab]` logits tensor.
pub fn sample_greedy(logits: &Tensor) -> Result<u32> {
    let vocab = *logits
        .shape
        .last()
        .ok_or_else(|| HivemindError::Inference("empty logits".into()))?;
    let last = logits
        .data
        .chunks(vocab)
        .last()
        .ok_or_else(|| HivemindError::Inference("no logit rows".into()))?;
    Ok(last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0))
}
