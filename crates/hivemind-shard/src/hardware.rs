use hivemind_core::{HardwareProfile, Result, HivemindError};
use tracing::{info, warn};

/// Detects the hardware capabilities of the current machine.
///
/// Always succeeds; falls back to CPU-only if GPU detection fails at runtime.
pub fn detect() -> Result<HardwareProfile> {
    let ram_mb = detect_ram_mb();
    info!(ram_mb, "system RAM detected");

    match detect_nvidia_gpu() {
        Ok(Some((model, vram_mb, cc))) => {
            info!(gpu_model = %model, vram_mb, compute_major = cc.0, compute_minor = cc.1,
                  "NVIDIA GPU detected");
            Ok(HardwareProfile {
                gpu_model: Some(model),
                vram_mb: Some(vram_mb),
                ram_mb,
                compute_capability: Some(cc),
            })
        }
        Ok(None) => {
            info!("no NVIDIA GPU detected — CPU-only node");
            Ok(HardwareProfile { gpu_model: None, vram_mb: None, ram_mb, compute_capability: None })
        }
        Err(e) => {
            warn!("GPU detection error ({}), falling back to CPU-only", e);
            Ok(HardwareProfile { gpu_model: None, vram_mb: None, ram_mb, compute_capability: None })
        }
    }
}

fn detect_ram_mb() -> u64 {
    use sysinfo::System;
    let sys = System::new_all();
    // total_memory() returns bytes in sysinfo >=0.30
    sys.total_memory() / 1_048_576
}

/// Returns `(model_name, vram_mb, (compute_major, compute_minor))` for the
/// first NVIDIA GPU via NVML, or `None` if no GPU is available.
fn detect_nvidia_gpu() -> Result<Option<(String, u64, (u32, u32))>> {
    use nvml_wrapper::Nvml;

    let nvml = match Nvml::init() {
        Ok(n) => n,
        // NVML not available — no NVIDIA driver or not an NVIDIA system
        Err(_) => return Ok(None),
    };

    if nvml.device_count().unwrap_or(0) == 0 {
        return Ok(None);
    }

    let device = nvml
        .device_by_index(0)
        .map_err(|e| HivemindError::Hardware(format!("open device 0: {e}")))?;

    let name = device
        .name()
        .map_err(|e| HivemindError::Hardware(format!("device name: {e}")))?;

    let mem = device
        .memory_info()
        .map_err(|e| HivemindError::Hardware(format!("memory info: {e}")))?;

    let vram_mb = mem.total / 1_048_576;

    let cc = device
        .cuda_compute_capability()
        .map_err(|e| HivemindError::Hardware(format!("compute capability: {e}")))?;

    // NVML returns i32; compute capability is always positive
    let cc_tuple = (cc.major.unsigned_abs(), cc.minor.unsigned_abs());

    Ok(Some((name, vram_mb, cc_tuple)))
}
