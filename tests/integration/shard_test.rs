// Integration tests for hardware detection and shard loading.

use hivemind_core::LayerRange;
use hivemind_shard::hardware::detect;

#[test]
fn hardware_detection_succeeds() {
    // Should always succeed — falls back to CPU-only on non-NVIDIA systems.
    let profile = detect().expect("hardware detection must not fail");
    assert!(profile.ram_mb > 0, "RAM must be detected");
    // GPU detection is optional
    println!("Detected: {profile:#?}");
}

#[test]
fn hardware_profile_budget_cpu_only() {
    use hivemind_core::HardwareProfile;
    let profile = HardwareProfile {
        gpu_model: None,
        vram_mb: None,
        ram_mb: 16_384,
        compute_capability: None,
    };
    let budget = profile.shard_budget_mb(0.8);
    assert_eq!(budget, 8_192, "CPU-only budget should be RAM / 2");
}

#[test]
fn hardware_profile_budget_gpu() {
    use hivemind_core::HardwareProfile;
    let profile = HardwareProfile {
        gpu_model: Some("NVIDIA RTX 4090".into()),
        vram_mb: Some(24_576),
        ram_mb: 65_536,
        compute_capability: Some((8, 9)),
    };
    let budget = profile.shard_budget_mb(0.8);
    assert_eq!(budget, 19_660, "GPU budget should be VRAM * gpu_allocation");
}

#[test]
fn layer_range_coverage() {
    let r = LayerRange::new(0, 80);
    assert_eq!(r.len(), 80);
    assert!(!r.is_empty());
    assert!(r.contains_layer(0));
    assert!(r.contains_layer(79));
    assert!(!r.contains_layer(80));
}

#[test]
fn layer_range_overlap() {
    let a = LayerRange::new(0, 40);
    let b = LayerRange::new(20, 60);
    let c = LayerRange::new(40, 80);
    assert!(a.overlaps(&b));
    assert!(!a.overlaps(&c));
}
