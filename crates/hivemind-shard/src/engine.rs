//! Inference engines: the trait every shard backend implements, plus the
//! built-in reference transformer.
//!
//! The reference engine is a real decoder-only transformer — real matmuls,
//! real multi-head attention, real KV cache, genuine layer splitting — with
//! weights derived deterministically from the model name. Every node
//! materializes identical weights locally, so a pipeline split across nodes
//! must produce *bit-identical* output to single-node execution. That makes
//! the distributed system testable end-to-end: any routing, failover, or
//! replay bug shows up as divergent tokens. Production models (llama.cpp
//! GGUF shards) implement the same trait.

use hivemind_core::{HivemindError, LayerRange, PipelineId, Result, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;

/// One forward call against a shard: process `inputs` (or embed from
/// `token_ids` when this shard owns layer 0) for the positions starting at
/// `start_pos`, appending to the sequence's KV cache.
pub struct ForwardRequest<'a> {
    pub pipeline_id: PipelineId,
    /// Full token sequence so far. Only consulted by the shard that owns
    /// layer 0 (for embedding); other shards work purely on activations.
    pub token_ids: &'a [u32],
    /// Sequence position of the first new input row. Must equal the shard's
    /// cached KV length for this pipeline — a mismatch means the KV cache is
    /// cold (fresh standby) and the caller must replay history first.
    pub start_pos: u32,
    /// `[n_new, d_model]` boundary activations from the previous shard, or
    /// `None` when this shard embeds from `token_ids`.
    pub inputs: Option<&'a Tensor>,
}

#[derive(Debug)]
pub struct ForwardOutput {
    /// `[n_new, d_model]` boundary activations, or `[n_new, vocab]` logits
    /// when this shard owns the final layer.
    pub tensor: Tensor,
    pub is_logits: bool,
}

/// A loaded model shard that can serve its layer range.
pub trait InferenceEngine: Send + Sync {
    fn layer_range(&self) -> LayerRange;

    /// KV-cache length (tokens processed) for a pipeline; 0 when unknown.
    fn kv_len(&self, pipeline_id: PipelineId) -> u32;

    fn forward(&self, req: ForwardRequest<'_>) -> Result<ForwardOutput>;

    /// Frees per-sequence state once a pipeline completes.
    fn drop_session(&self, pipeline_id: PipelineId);
}

/// Hyperparameters of the reference model. Small enough to run on anything;
/// the point is real distributed semantics, not model quality.
#[derive(Debug, Clone)]
pub struct RefConfig {
    pub n_layers: u32,
    pub d_model: usize,
    pub n_heads: usize,
    pub vocab: usize,
    pub max_seq: usize,
    pub seed: u64,
}

impl RefConfig {
    /// Byte-level vocab: token ids are UTF-8 bytes.
    pub const BYTE_VOCAB: usize = 256;

    pub fn for_model(name: &str, n_layers: u32) -> Self {
        Self {
            n_layers,
            d_model: 64,
            n_heads: 4,
            vocab: Self::BYTE_VOCAB,
            max_seq: 512,
            seed: fnv1a(name.as_bytes()),
        }
    }

    fn d_ff(&self) -> usize {
        4 * self.d_model
    }

    fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// splitmix64: deterministic weight generation from (seed, tag) with no
/// stored weight files. Values are uniform in [-scale, scale].
fn weights(seed: u64, tag: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed ^ tag.wrapping_mul(0x9e3779b97f4a7c15);
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            // Map to [-scale, scale)
            ((z >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

struct LayerWeights {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>, // d_model -> d_ff
    w2: Vec<f32>, // d_ff -> d_model
}

/// Per-pipeline attention cache: `k[layer][token]` and `v[layer][token]`,
/// each a `d_model`-length vector, for this shard's layers only.
struct KvCache {
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
}

impl KvCache {
    fn new(n_layers: usize) -> Self {
        Self {
            k: vec![Vec::new(); n_layers],
            v: vec![Vec::new(); n_layers],
        }
    }

    fn len(&self) -> u32 {
        self.k.first().map_or(0, |l| l.len() as u32)
    }
}

/// The reference engine, holding materialized weights for one layer range.
pub struct RefEngine {
    cfg: RefConfig,
    range: LayerRange,
    layers: Vec<LayerWeights>,
    /// Token + positional embeddings; present only when `range.start == 0`.
    embed: Option<(Vec<f32>, Vec<f32>)>,
    /// Unembedding matrix; present only when `range.end == n_layers`.
    unembed: Option<Vec<f32>>,
    sessions: Mutex<HashMap<PipelineId, KvCache>>,
}

impl RefEngine {
    pub fn new(cfg: RefConfig, range: LayerRange) -> Result<Self> {
        if range.end > cfg.n_layers || range.is_empty() {
            return Err(HivemindError::Shard(format!(
                "layer range {range} invalid for {}-layer model",
                cfg.n_layers
            )));
        }
        let d = cfg.d_model;
        let scale = 1.0 / (d as f32).sqrt();
        let layers = (range.start..range.end)
            .map(|l| {
                let tag = |slot: u64| (l as u64) << 8 | slot;
                LayerWeights {
                    wq: weights(cfg.seed, tag(1), d * d, scale),
                    wk: weights(cfg.seed, tag(2), d * d, scale),
                    wv: weights(cfg.seed, tag(3), d * d, scale),
                    wo: weights(cfg.seed, tag(4), d * d, scale),
                    w1: weights(cfg.seed, tag(5), d * cfg.d_ff(), scale),
                    w2: weights(cfg.seed, tag(6), cfg.d_ff() * d, scale),
                }
            })
            .collect();
        let embed = (range.start == 0).then(|| {
            (
                weights(cfg.seed, 0xE0_0000, cfg.vocab * d, 1.0),
                weights(cfg.seed, 0xE0_0001, cfg.max_seq * d, 0.1),
            )
        });
        let unembed =
            (range.end == cfg.n_layers).then(|| weights(cfg.seed, 0xE0_0002, d * cfg.vocab, scale));
        Ok(Self {
            cfg,
            range,
            layers,
            embed,
            unembed,
            sessions: Mutex::new(HashMap::new()),
        })
    }

    /// Runs one token's residual vector through one layer, extending that
    /// layer's KV cache. Attention is causal by construction: the query
    /// attends to every cached position plus itself.
    fn layer_forward(&self, li: usize, x: &mut [f32], kv: &mut KvCache) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let w = &self.layers[li];

        // Attention block (pre-LN).
        let h = layer_norm(x);
        let q = matvec(&w.wq, &h, d, d);
        let k = matvec(&w.wk, &h, d, d);
        let v = matvec(&w.wv, &h, d, d);
        kv.k[li].push(k);
        kv.v[li].push(v);

        let hd = cfg.head_dim();
        let inv_sqrt = 1.0 / (hd as f32).sqrt();
        let t_len = kv.k[li].len();
        let mut attn = vec![0.0f32; d];
        for head in 0..cfg.n_heads {
            let o = head * hd;
            // Scores against every cached position (including the current one).
            let mut scores: Vec<f32> = (0..t_len)
                .map(|t| {
                    let kt = &kv.k[li][t][o..o + hd];
                    q[o..o + hd].iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * inv_sqrt
                })
                .collect();
            softmax(&mut scores);
            for (t, s) in scores.iter().enumerate() {
                let vt = &kv.v[li][t][o..o + hd];
                for (ai, vi) in attn[o..o + hd].iter_mut().zip(vt) {
                    *ai += s * vi;
                }
            }
        }
        let attn_out = matvec(&w.wo, &attn, d, d);
        for (xi, ai) in x.iter_mut().zip(&attn_out) {
            *xi += ai;
        }

        // MLP block (pre-LN).
        let h2 = layer_norm(x);
        let mut ff = matvec(&w.w1, &h2, d, cfg.d_ff());
        for f in &mut ff {
            *f = gelu(*f);
        }
        let mlp_out = matvec(&w.w2, &ff, cfg.d_ff(), d);
        for (xi, mi) in x.iter_mut().zip(&mlp_out) {
            *xi += mi;
        }
    }

    fn embed_token(&self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let (tok_emb, pos_emb) = self.embed.as_ref().ok_or_else(|| {
            HivemindError::Inference("shard does not own the embedding layer".into())
        })?;
        let d = self.cfg.d_model;
        if token as usize >= self.cfg.vocab {
            return Err(HivemindError::Inference(format!("token {token} out of vocab")));
        }
        if pos >= self.cfg.max_seq {
            return Err(HivemindError::Inference(format!(
                "position {pos} exceeds max_seq {}",
                self.cfg.max_seq
            )));
        }
        let t = &tok_emb[token as usize * d..(token as usize + 1) * d];
        let p = &pos_emb[pos * d..(pos + 1) * d];
        Ok(t.iter().zip(p).map(|(a, b)| a + b).collect())
    }
}

impl InferenceEngine for RefEngine {
    fn layer_range(&self) -> LayerRange {
        self.range
    }

    fn kv_len(&self, pipeline_id: PipelineId) -> u32 {
        self.sessions
            .lock()
            .unwrap()
            .get(&pipeline_id)
            .map_or(0, KvCache::len)
    }

    fn forward(&self, req: ForwardRequest<'_>) -> Result<ForwardOutput> {
        let d = self.cfg.d_model;
        let mut sessions = self.sessions.lock().unwrap();
        let kv = sessions
            .entry(req.pipeline_id)
            .or_insert_with(|| KvCache::new(self.layers.len()));

        if kv.len() != req.start_pos {
            return Err(HivemindError::Inference(format!(
                "kv gap: shard has {} cached tokens but request starts at {} — replay required",
                kv.len(),
                req.start_pos
            )));
        }

        // Assemble the new input rows: embed locally or take upstream activations.
        let rows: Vec<Vec<f32>> = match req.inputs {
            Some(t) => {
                if t.shape.len() != 2 || t.shape[1] != d {
                    return Err(HivemindError::Inference(format!(
                        "expected [n, {d}] activations, got shape {:?}",
                        t.shape
                    )));
                }
                t.data.chunks(d).map(|c| c.to_vec()).collect()
            }
            None => {
                let new_tokens = req
                    .token_ids
                    .get(req.start_pos as usize..)
                    .unwrap_or(&[]);
                if new_tokens.is_empty() {
                    return Err(HivemindError::Inference("no new tokens to embed".into()));
                }
                new_tokens
                    .iter()
                    .enumerate()
                    .map(|(i, &tok)| self.embed_token(tok, req.start_pos as usize + i))
                    .collect::<Result<_>>()?
            }
        };

        // Process tokens strictly one at a time so batched prefill and
        // incremental decode are bit-identical — replay recovery depends on it.
        let mut out = Vec::with_capacity(rows.len() * d);
        let n_rows = rows.len();
        for mut x in rows {
            for li in 0..self.layers.len() {
                self.layer_forward(li, &mut x, kv);
            }
            out.extend_from_slice(&x);
        }

        if let Some(wu) = &self.unembed {
            let vocab = self.cfg.vocab;
            let mut logits = Vec::with_capacity(n_rows * vocab);
            for row in out.chunks(d) {
                let h = layer_norm(row);
                logits.extend(matvec(wu, &h, d, vocab));
            }
            Ok(ForwardOutput {
                tensor: Tensor::new(logits, vec![n_rows, vocab]),
                is_logits: true,
            })
        } else {
            Ok(ForwardOutput {
                tensor: Tensor::new(out, vec![n_rows, d]),
                is_logits: false,
            })
        }
    }

    fn drop_session(&self, pipeline_id: PipelineId) {
        self.sessions.lock().unwrap().remove(&pipeline_id);
    }
}

/// Greedy sampling: argmax over the final row of a `[n, vocab]` logits tensor.
/// Deterministic, which the end-to-end identity tests rely on.
pub fn sample_greedy(logits: &Tensor) -> Result<u32> {
    let vocab = *logits.shape.last().ok_or_else(|| {
        HivemindError::Inference("empty logits tensor".into())
    })?;
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

fn matvec(w: &[f32], x: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), d_in * d_out);
    debug_assert_eq!(x.len(), d_in);
    (0..d_out)
        .map(|o| {
            let row = &w[o * d_in..(o + 1) * d_in];
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn layer_norm(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter().map(|v| (v - mean) * inv).collect()
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn cfg() -> RefConfig {
        RefConfig::for_model("hivemind-ref-test", 6)
    }

    /// Runs a full generation on a single engine holding all layers.
    fn generate_single(prompt: &[u32], n_gen: usize) -> Vec<u32> {
        let engine = RefEngine::new(cfg(), LayerRange::new(0, 6)).unwrap();
        let pid = Uuid::new_v4();
        let mut tokens: Vec<u32> = prompt.to_vec();
        let mut start = 0u32;
        for _ in 0..n_gen {
            let out = engine
                .forward(ForwardRequest {
                    pipeline_id: pid,
                    token_ids: &tokens,
                    start_pos: start,
                    inputs: None,
                })
                .unwrap();
            assert!(out.is_logits);
            start = tokens.len() as u32;
            tokens.push(sample_greedy(&out.tensor).unwrap());
        }
        tokens[prompt.len()..].to_vec()
    }

    /// Runs the same generation through a chain of split engines, passing
    /// boundary activations exactly as the network will.
    fn generate_split(prompt: &[u32], n_gen: usize, splits: &[(u32, u32)]) -> Vec<u32> {
        let engines: Vec<RefEngine> = splits
            .iter()
            .map(|&(s, e)| RefEngine::new(cfg(), LayerRange::new(s, e)).unwrap())
            .collect();
        let pid = Uuid::new_v4();
        let mut tokens: Vec<u32> = prompt.to_vec();
        let mut start = 0u32;
        for _ in 0..n_gen {
            let mut boundary: Option<Tensor> = None;
            let mut logits: Option<Tensor> = None;
            for engine in &engines {
                let out = engine
                    .forward(ForwardRequest {
                        pipeline_id: pid,
                        token_ids: &tokens,
                        start_pos: start,
                        inputs: boundary.as_ref(),
                    })
                    .unwrap();
                if out.is_logits {
                    logits = Some(out.tensor);
                } else {
                    boundary = Some(out.tensor);
                }
            }
            start = tokens.len() as u32;
            tokens.push(sample_greedy(&logits.unwrap()).unwrap());
        }
        tokens[prompt.len()..].to_vec()
    }

    fn prompt() -> Vec<u32> {
        "fn main() {".bytes().map(u32::from).collect()
    }

    #[test]
    fn deterministic_across_runs() {
        assert_eq!(generate_single(&prompt(), 8), generate_single(&prompt(), 8));
    }

    #[test]
    fn split_pipeline_matches_single_node_exactly() {
        let single = generate_single(&prompt(), 8);
        for splits in [
            vec![(0u32, 3u32), (3, 6)],
            vec![(0, 2), (2, 4), (4, 6)],
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        ] {
            assert_eq!(
                generate_split(&prompt(), 8, &splits),
                single,
                "split {splits:?} diverged from single-node output"
            );
        }
    }

    #[test]
    fn cold_kv_cache_demands_replay() {
        let engine = RefEngine::new(cfg(), LayerRange::new(2, 4)).unwrap();
        let d = 64;
        let step = Tensor::new(vec![0.1; d], vec![1, d]);
        let err = engine
            .forward(ForwardRequest {
                pipeline_id: Uuid::new_v4(),
                token_ids: &[],
                start_pos: 5,
                inputs: Some(&step),
            })
            .unwrap_err();
        assert!(err.to_string().contains("replay required"));
    }

    #[test]
    fn replayed_standby_continues_bit_exactly() {
        // Generate 4 tokens on chain A|B, then replace B with a cold B'
        // rebuilt from replayed boundary activations and check the next
        // tokens match the uninterrupted run.
        let full = generate_single(&prompt(), 8);

        let a = RefEngine::new(cfg(), LayerRange::new(0, 3)).unwrap();
        let b = RefEngine::new(cfg(), LayerRange::new(3, 6)).unwrap();
        let pid = Uuid::new_v4();
        let mut tokens = prompt();
        let mut start = 0u32;
        // History of boundary activations fed into B (what the client records).
        let mut b_history: Vec<f32> = Vec::new();
        let mut generated = Vec::new();

        for _ in 0..4 {
            let mid = a
                .forward(ForwardRequest { pipeline_id: pid, token_ids: &tokens, start_pos: start, inputs: None })
                .unwrap();
            b_history.extend_from_slice(&mid.tensor.data);
            let out = b
                .forward(ForwardRequest { pipeline_id: pid, token_ids: &tokens, start_pos: start, inputs: Some(&mid.tensor) })
                .unwrap();
            start = tokens.len() as u32;
            let tok = sample_greedy(&out.tensor).unwrap();
            tokens.push(tok);
            generated.push(tok);
        }

        // B "dies"; B' rebuilds its KV from the recorded history in one shot.
        let b2 = RefEngine::new(cfg(), LayerRange::new(3, 6)).unwrap();
        let d = 64;
        let n_hist = b_history.len() / d;
        let replay = Tensor::new(b_history.clone(), vec![n_hist, d]);
        b2.forward(ForwardRequest { pipeline_id: pid, token_ids: &tokens, start_pos: 0, inputs: Some(&replay) })
            .unwrap();
        assert_eq!(b2.kv_len(pid), start, "replay must rebuild the full KV history");

        for _ in 0..4 {
            let mid = a
                .forward(ForwardRequest { pipeline_id: pid, token_ids: &tokens, start_pos: start, inputs: None })
                .unwrap();
            let out = b2
                .forward(ForwardRequest { pipeline_id: pid, token_ids: &tokens, start_pos: start, inputs: Some(&mid.tensor) })
                .unwrap();
            start = tokens.len() as u32;
            let tok = sample_greedy(&out.tensor).unwrap();
            tokens.push(tok);
            generated.push(tok);
        }

        assert_eq!(generated, full, "failover must not change the generation");
    }
}
