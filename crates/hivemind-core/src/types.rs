use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a network node.
pub type NodeId = Uuid;

/// Unique identifier for a pipeline execution.
pub type PipelineId = Uuid;

/// Index identifying a shard (contiguous layer range) assignment.
pub type ShardId = u32;

/// Non-transferable utility token represented in micro-units (1 token = 1_000_000 micro).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub struct MicroToken(pub u64);

impl MicroToken {
    pub const ZERO: Self = Self(0);
    pub const ONE_TOKEN: Self = Self(1_000_000);

    pub fn from_tokens(t: u64) -> Self {
        Self(t.saturating_mul(1_000_000))
    }

    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    pub fn as_tokens_f64(self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }
}

impl std::fmt::Display for MicroToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4} HMT", self.as_tokens_f64())
    }
}

/// A contiguous range of transformer layers `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerRange {
    pub start: u32,
    pub end: u32,
}

impl LayerRange {
    pub fn new(start: u32, end: u32) -> Self {
        debug_assert!(start < end, "LayerRange: start must be < end");
        Self { start, end }
    }

    pub fn len(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    pub fn contains_layer(&self, layer: u32) -> bool {
        layer >= self.start && layer < self.end
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

impl std::fmt::Display for LayerRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}..{})", self.start, self.end)
    }
}

/// Hardware capabilities detected on the local machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// NVIDIA GPU model name, if detected.
    pub gpu_model: Option<String>,
    /// Total VRAM in megabytes.
    pub vram_mb: Option<u64>,
    /// Total system RAM in megabytes.
    pub ram_mb: u64,
    /// CUDA compute capability `(major, minor)`.
    pub compute_capability: Option<(u32, u32)>,
}

impl HardwareProfile {
    pub fn has_gpu(&self) -> bool {
        self.gpu_model.is_some()
    }

    /// Effective memory budget for model shards given a fractional GPU allocation.
    pub fn shard_budget_mb(&self, gpu_fraction: f64) -> u64 {
        if let Some(vram) = self.vram_mb {
            (vram as f64 * gpu_fraction.clamp(0.0, 1.0)) as u64
        } else {
            // CPU-only: cap at half of RAM to leave headroom
            self.ram_mb / 2
        }
    }
}

/// A flat tensor passed between pipeline nodes as activations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// Row-major flat data.
    pub data: Vec<f32>,
    /// Dimension sizes.
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Constructs an all-zero tensor of the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self { data: vec![0.0; n], shape }
    }
}

/// Describes a single node's assignment within an assembled pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSlot {
    pub node_id: NodeId,
    pub layer_range: LayerRange,
    /// Zero-based position in the pipeline chain.
    pub position: usize,
}

/// Final output of a completed inference request.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub pipeline_id: PipelineId,
    /// Generated token IDs.
    pub tokens: Vec<u32>,
    /// Wall-clock latency in milliseconds.
    pub latency_ms: u64,
}
