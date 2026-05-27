use hivemind_core::{HardwareProfile, LayerRange, Result, HivemindError};
use std::path::PathBuf;
use tracing::info;

/// Parameters for downloading a model shard.
#[derive(Debug, Clone)]
pub struct ShardDownloadConfig {
    pub model_name: String,
    pub quantization: String,
    pub layer_range: LayerRange,
    pub output_dir: PathBuf,
}

/// A model shard that has been loaded into memory and is ready for inference.
#[derive(Debug)]
pub struct LoadedShard {
    pub model_name: String,
    pub layer_range: LayerRange,
    /// Opaque handle to the underlying llama.cpp context.
    /// Will be `Box<llama_cpp::Model>` once the binding is wired up.
    _handle: (),
}

impl LoadedShard {
    pub fn layer_range(&self) -> LayerRange {
        self.layer_range
    }
}

/// Downloads the GGUF shard file for the given layer range from the model registry.
///
/// TODO: implement actual HTTP download from the Hivemind model registry.
/// The shard filename encodes the layer range, e.g.
/// `qwen2.5-coder-72b-q4_k_m-layers-0-8.gguf`.
pub async fn download_shard(
    cfg: &ShardDownloadConfig,
    _profile: &HardwareProfile,
) -> Result<PathBuf> {
    let filename = format!(
        "{}-{}-layers-{}-{}.gguf",
        cfg.model_name, cfg.quantization, cfg.layer_range.start, cfg.layer_range.end
    );
    let dest = cfg.output_dir.join(&filename);

    if dest.exists() {
        info!(path = %dest.display(), "shard already cached, skipping download");
        return Ok(dest);
    }

    // TODO: implement download
    // 1. Resolve shard URL from registry: GET https://registry.hivemind.dev/v1/shards/{model}/{layers}
    // 2. Stream download with progress reporting
    // 3. Verify SHA-256 checksum
    // 4. Write to dest path
    Err(HivemindError::Shard(format!(
        "download not yet implemented — shard file expected at: {}",
        dest.display()
    )))
}

/// Loads a previously downloaded GGUF shard file into memory using llama.cpp.
///
/// TODO: implement with llama-cpp-rs bindings.
/// The loaded context will be pinned to the GPU layers specified in `profile`.
pub fn load_shard(path: &std::path::Path, layer_range: LayerRange, _profile: &HardwareProfile) -> Result<LoadedShard> {
    // TODO: implement
    // 1. Open GGUF file with llama_model_load()
    // 2. Configure GPU offloading based on profile.gpu_allocation
    // 3. Pin the context to the specified layer_range
    // 4. Return the opaque handle
    Err(HivemindError::Shard(format!(
        "llama.cpp binding not yet wired up — cannot load shard at {} for layers {}",
        path.display(),
        layer_range,
    )))
}
