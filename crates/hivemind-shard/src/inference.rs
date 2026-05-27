use hivemind_core::{Result, HivemindError, Tensor};
use crate::loader::LoadedShard;

/// Runs a forward pass through the local shard layers.
///
/// Accepts `input` activations from the previous pipeline node (or the
/// embedding layer for the first node), processes them through the shard's
/// transformer layers, and returns output activations for the next node.
///
/// TODO: implement via llama.cpp `llama_decode()`.
/// The input/output tensors are `[seq_len, hidden_dim]` in bfloat16 on the
/// wire (downcast to f32 here for now).
pub fn forward_pass(input: Tensor, _shard: &LoadedShard) -> Result<Tensor> {
    // TODO: implement
    // 1. Copy `input.data` into a llama.cpp batch
    // 2. Call llama_decode() which runs only the layers owned by this shard
    // 3. Extract the output hidden states from the KV cache / logits buffer
    // 4. Return as Tensor

    // Sanity-check: output shape matches input shape (hidden states pass through)
    if input.shape.len() < 2 {
        return Err(HivemindError::Inference(format!(
            "expected at least 2D tensor, got {}D",
            input.shape.len()
        )));
    }

    Err(HivemindError::Inference(
        "llama.cpp forward pass not yet implemented".into(),
    ))
}

/// Runs token sampling on the final node's output logits to produce a token ID.
///
/// TODO: implement greedy / top-p / top-k sampling via llama.cpp sampler API.
pub fn sample_token(logits: &Tensor, _temperature: f32, _top_p: f32) -> Result<u32> {
    if logits.data.is_empty() {
        return Err(HivemindError::Inference("empty logits tensor".into()));
    }
    // TODO: replace with proper sampling
    Err(HivemindError::Inference("sampling not yet implemented".into()))
}
