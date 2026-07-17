//! Real tonic implementation of `ActivationService` — the hot path every
//! shard node serves.

#![allow(clippy::result_large_err)] // tonic handlers return Status by value

use crate::scheduler::ResourceScheduler;
use hivemind_core::{MicroToken, PipelineId, Tensor};
use hivemind_ledger::Wallet;
use hivemind_proto::activations::activation_service_server::ActivationService;
use hivemind_proto::activations::{
    ActivationRequest, ActivationResponse, ReplayChunk, ReplayRequest,
};
use hivemind_shard::{ActivationCheckpointStore, ForwardRequest, InferenceEngine};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{debug, warn};
use uuid::Uuid;

type ReplayStream = Pin<Box<dyn tokio_stream::Stream<Item = Result<ReplayChunk, Status>> + Send>>;

/// Shard node request handler: runs the local engine's layer range over
/// incoming activations, checkpoints outputs, and earns tokens per layer.
pub struct ShardService {
    pub engine: Arc<dyn InferenceEngine>,
    pub scheduler: Arc<ResourceScheduler>,
    pub wallet: Arc<Wallet>,
    pub checkpoints: Arc<Mutex<ActivationCheckpointStore>>,
}

impl ShardService {
    pub fn new(
        engine: Arc<dyn InferenceEngine>,
        scheduler: Arc<ResourceScheduler>,
        wallet: Arc<Wallet>,
    ) -> Self {
        Self {
            engine,
            scheduler,
            wallet,
            checkpoints: Arc::new(Mutex::new(ActivationCheckpointStore::default())),
        }
    }

    fn parse_pipeline_id(raw: &str) -> Result<PipelineId, Status> {
        raw.parse::<Uuid>()
            .map_err(|_| Status::invalid_argument(format!("bad pipeline_id: {raw}")))
    }

    fn run_forward(&self, req: &ActivationRequest) -> Result<ActivationResponse, Status> {
        let pipeline_id = Self::parse_pipeline_id(&req.pipeline_id)?;
        let seq_len = req.seq_len;
        let kv_len = self.engine.kv_len(pipeline_id);

        let input_tensor: Option<Tensor> = req
            .input
            .as_ref()
            .filter(|t| !t.data.is_empty())
            .map(Tensor::from);

        // Establish where these rows start in the sequence and demand replay
        // when our KV cache doesn't line up (fresh standby / restarted node).
        let start_pos = match &input_tensor {
            Some(t) => {
                let n_new = t.shape.first().copied().unwrap_or(0) as u32;
                let expected_start = seq_len.saturating_sub(n_new);
                if kv_len != expected_start {
                    return Err(Status::failed_precondition(format!(
                        "kv gap: have {kv_len} cached tokens, request rows start at {expected_start} — replay required"
                    )));
                }
                expected_start
            }
            None => {
                // First shard embeds from token ids; it can always catch up
                // from its own KV position.
                if kv_len >= seq_len {
                    return Err(Status::invalid_argument(format!(
                        "nothing to do: kv_len {kv_len} >= seq_len {seq_len}"
                    )));
                }
                kv_len
            }
        };

        let started = Instant::now();
        let out = self
            .engine
            .forward(ForwardRequest {
                pipeline_id,
                token_ids: &req.token_ids,
                start_pos,
                inputs: input_tensor.as_ref(),
            })
            .map_err(|e| Status::internal(format!("forward pass failed: {e}")))?;
        let compute_us = started.elapsed().as_micros() as u64;

        // Checkpoint each output row so a promoted standby downstream can be
        // rebuilt from us (ReplayBoundary). Logits are terminal — nothing
        // downstream consumes them, so there is nothing to checkpoint.
        if !out.is_logits {
            let d = *out.tensor.shape.last().unwrap_or(&0);
            if d > 0 {
                let mut store = self.checkpoints.lock().unwrap();
                for (i, row) in out.tensor.data.chunks(d).enumerate() {
                    let t = Tensor::new(row.to_vec(), vec![1, d]);
                    if let Err(e) = store.record(pipeline_id, start_pos + i as u32, &t) {
                        warn!(%pipeline_id, error = %e, "checkpoint record failed");
                        break;
                    }
                }
            }
        }

        let layers = self.engine.layer_range().len() as u64;
        let n_new = (seq_len - start_pos) as u64;
        self.wallet.earn(MicroToken(layers * n_new));

        Ok(ActivationResponse {
            pipeline_id: req.pipeline_id.clone(),
            output: Some((&out.tensor).into()),
            compute_us,
        })
    }
}

#[tonic::async_trait]
impl ActivationService for ShardService {
    async fn forward(
        &self,
        request: Request<ActivationRequest>,
    ) -> Result<Response<ActivationResponse>, Status> {
        let _guard = self
            .scheduler
            .acquire_pipeline()
            .ok_or_else(|| Status::resource_exhausted("node is draining or at capacity"))?;
        let req = request.into_inner();
        debug!(pipeline = %req.pipeline_id, seq_len = req.seq_len, "forward");
        self.run_forward(&req).map(Response::new)
    }

    type ForwardStreamStream = ReplayForwardStream;

    async fn forward_stream(
        &self,
        _request: Request<ActivationRequest>,
    ) -> Result<Response<Self::ForwardStreamStream>, Status> {
        // Client-driven routing samples client-side; per-hop streaming is a
        // latency optimization for the node-to-node forwarding mode.
        Err(Status::unimplemented(
            "streaming forward not yet supported; use Forward per step",
        ))
    }

    type ReplayBoundaryStream = ReplayStream;

    async fn replay_boundary(
        &self,
        request: Request<ReplayRequest>,
    ) -> Result<Response<Self::ReplayBoundaryStream>, Status> {
        let req = request.into_inner();
        let pipeline_id = Self::parse_pipeline_id(&req.pipeline_id)?;
        let tensors = self
            .checkpoints
            .lock()
            .unwrap()
            .replay(pipeline_id, req.from_token)
            .ok_or_else(|| {
                Status::not_found(format!(
                    "no checkpoints for pipeline {pipeline_id}; fall back to full prefill"
                ))
            })?;

        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let raw_id = req.pipeline_id.clone();
        let from = req.from_token;
        tokio::spawn(async move {
            for (i, t) in tensors.iter().enumerate() {
                let chunk = ReplayChunk {
                    pipeline_id: raw_id.clone(),
                    token_index: from + i as u32,
                    activation: Some(t.into()),
                };
                if tx.send(Ok(chunk)).await.is_err() {
                    break;
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// Placeholder stream type for the unimplemented `ForwardStream`.
pub type ReplayForwardStream =
    Pin<Box<dyn tokio_stream::Stream<Item = Result<ActivationResponse, Status>> + Send>>;
