//! Boundary-activation checkpointing for surgical KV-cache recovery.
//!
//! Every pipeline node keeps the *output* activations it has produced for
//! each token of an active sequence. When the node downstream of us dies and
//! a warm standby is promoted, the standby has the right weights loaded but a
//! cold KV cache. Instead of re-running the whole prompt through the whole
//! chain, the upstream node replays its cached boundary activations
//! (`ActivationService.ReplayBoundary`), and the standby runs one batched
//! prefill over *just its own layer range* — sub-second recovery instead of a
//! full-pipeline replay.
//!
//! Checkpoints are quantized to int8 (4x smaller than the f32 wire format;
//! boundary activations tolerate this easily since they are re-quantized by
//! the receiving node's own compute anyway). A 4K-token sequence on a 72B
//! model costs ~32 MB per pipeline at int8.

use hivemind_core::{HivemindError, PipelineId, Result, Tensor};
use std::collections::HashMap;

/// Default memory budget for cached activations: 256 MB.
pub const DEFAULT_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// A boundary activation quantized to symmetric int8 with a per-tensor scale.
#[derive(Debug, Clone)]
struct QuantizedActivation {
    scale: f32,
    shape: Vec<usize>,
    data: Vec<i8>,
}

impl QuantizedActivation {
    fn quantize(t: &Tensor) -> Self {
        let max_abs = t.data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let data = t
            .data
            .iter()
            .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        Self { scale, shape: t.shape.clone(), data }
    }

    fn dequantize(&self) -> Tensor {
        Tensor::new(
            self.data.iter().map(|&q| q as f32 * self.scale).collect(),
            self.shape.clone(),
        )
    }

    fn nbytes(&self) -> usize {
        self.data.len() + 8 * self.shape.len() + std::mem::size_of::<f32>()
    }
}

/// Per-sequence checkpoint history, one activation per generated token.
struct SequenceCheckpoints {
    /// Indexed by token position; always contiguous from token 0.
    activations: Vec<QuantizedActivation>,
    bytes: usize,
    /// Logical clock of the last touch, for LRU eviction.
    last_touched: u64,
}

/// Bounded store of boundary activations for the sequences this node is
/// currently serving. When the budget is exceeded, the least-recently-active
/// *other* sequence is dropped whole — a partial history is useless for
/// replay, and the sequence being written must never evict itself.
pub struct ActivationCheckpointStore {
    sequences: HashMap<PipelineId, SequenceCheckpoints>,
    max_bytes: usize,
    total_bytes: usize,
    clock: u64,
}

impl ActivationCheckpointStore {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            sequences: HashMap::new(),
            max_bytes,
            total_bytes: 0,
            clock: 0,
        }
    }

    /// Records the output boundary activation for `token_index` of a sequence.
    ///
    /// Token indices must be recorded in order with no gaps; re-recording the
    /// most recent index is allowed (idempotent retry after a hop failure).
    pub fn record(
        &mut self,
        pipeline_id: PipelineId,
        token_index: u32,
        activation: &Tensor,
    ) -> Result<()> {
        self.clock += 1;
        let clock = self.clock;
        let seq = self
            .sequences
            .entry(pipeline_id)
            .or_insert_with(|| SequenceCheckpoints {
                activations: Vec::new(),
                bytes: 0,
                last_touched: clock,
            });

        let next = seq.activations.len() as u32;
        let q = QuantizedActivation::quantize(activation);
        let q_bytes = q.nbytes();

        if token_index == next {
            seq.activations.push(q);
            seq.bytes += q_bytes;
            self.total_bytes += q_bytes;
        } else if token_index + 1 == next {
            // Idempotent overwrite of the latest checkpoint.
            let old = &mut seq.activations[token_index as usize];
            seq.bytes = seq.bytes - old.nbytes() + q_bytes;
            self.total_bytes = self.total_bytes - old.nbytes() + q_bytes;
            *old = q;
        } else {
            return Err(HivemindError::Pipeline(format!(
                "checkpoint gap for pipeline {pipeline_id}: got token {token_index}, expected {next}"
            )));
        }
        seq.last_touched = clock;

        self.evict_over_budget(pipeline_id);
        Ok(())
    }

    /// Returns dequantized activations for tokens `[from_token, len)`, or
    /// `None` when the sequence is unknown (finished or evicted).
    pub fn replay(&self, pipeline_id: PipelineId, from_token: u32) -> Option<Vec<Tensor>> {
        let seq = self.sequences.get(&pipeline_id)?;
        Some(
            seq.activations
                .get(from_token as usize..)
                .unwrap_or(&[])
                .iter()
                .map(QuantizedActivation::dequantize)
                .collect(),
        )
    }

    /// Number of checkpointed tokens for a sequence.
    pub fn token_count(&self, pipeline_id: PipelineId) -> usize {
        self.sequences
            .get(&pipeline_id)
            .map_or(0, |s| s.activations.len())
    }

    /// Drops a completed sequence's checkpoints.
    pub fn finish(&mut self, pipeline_id: PipelineId) {
        if let Some(seq) = self.sequences.remove(&pipeline_id) {
            self.total_bytes -= seq.bytes;
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    fn evict_over_budget(&mut self, exempt: PipelineId) {
        while self.total_bytes > self.max_bytes {
            let victim = self
                .sequences
                .iter()
                .filter(|(id, _)| **id != exempt)
                .min_by_key(|(_, s)| s.last_touched)
                .map(|(id, _)| *id);
            match victim {
                Some(id) => self.finish(id),
                // Only the active sequence remains; let it exceed the budget
                // rather than destroy its own replay history.
                None => break,
            }
        }
    }
}

impl Default for ActivationCheckpointStore {
    fn default() -> Self {
        Self::new(DEFAULT_BUDGET_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn tensor(vals: &[f32]) -> Tensor {
        Tensor::new(vals.to_vec(), vec![1, vals.len()])
    }

    #[test]
    fn quantization_roundtrip_is_close() {
        let t = tensor(&[0.5, -1.25, 3.75, 0.0, -127.0]);
        let q = QuantizedActivation::quantize(&t);
        let back = q.dequantize();
        for (a, b) in t.data.iter().zip(back.data.iter()) {
            assert!((a - b).abs() <= q.scale, "value {a} decoded as {b}");
        }
    }

    #[test]
    fn replay_returns_suffix_from_token() {
        let mut store = ActivationCheckpointStore::default();
        let pid = Uuid::new_v4();
        for i in 0..5u32 {
            store.record(pid, i, &tensor(&[i as f32])).unwrap();
        }
        let replayed = store.replay(pid, 3).unwrap();
        assert_eq!(replayed.len(), 2);
        assert_eq!(replayed[0].data[0], 3.0);
        assert_eq!(replayed[1].data[0], 4.0);
    }

    #[test]
    fn rejects_gaps_but_allows_idempotent_rewrite() {
        let mut store = ActivationCheckpointStore::default();
        let pid = Uuid::new_v4();
        store.record(pid, 0, &tensor(&[1.0])).unwrap();
        store.record(pid, 1, &tensor(&[2.0])).unwrap();
        // Retry of the latest token is fine.
        store.record(pid, 1, &tensor(&[2.5])).unwrap();
        assert_eq!(store.token_count(pid), 2);
        // A gap is not.
        assert!(store.record(pid, 5, &tensor(&[9.0])).is_err());
    }

    #[test]
    fn finish_frees_memory() {
        let mut store = ActivationCheckpointStore::default();
        let pid = Uuid::new_v4();
        store.record(pid, 0, &tensor(&[1.0, 2.0, 3.0])).unwrap();
        assert!(store.total_bytes() > 0);
        store.finish(pid);
        assert_eq!(store.total_bytes(), 0);
        assert!(store.replay(pid, 0).is_none());
    }

    #[test]
    fn evicts_lru_sequence_but_never_the_active_one() {
        // Budget fits roughly one sequence's worth of data.
        let mut store = ActivationCheckpointStore::new(600);
        let old = Uuid::new_v4();
        let active = Uuid::new_v4();
        for i in 0..4u32 {
            store.record(old, i, &tensor(&[1.0; 32])).unwrap();
        }
        for i in 0..8u32 {
            store.record(active, i, &tensor(&[1.0; 32])).unwrap();
        }
        assert!(store.replay(old, 0).is_none(), "LRU sequence evicted");
        assert_eq!(store.token_count(active), 8, "active sequence kept whole");
    }
}
