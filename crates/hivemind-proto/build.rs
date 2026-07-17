fn main() -> Result<(), Box<dyn std::error::Error>> {
    // No system protoc required — use the vendored binary.
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);

    tonic_build::configure().compile_protos(
        &[
            "../../proto/activations.proto",
            "../../proto/routing.proto",
            "../../proto/discovery.proto",
            "../../proto/tokens.proto",
        ],
        &["../../proto"],
    )?;
    Ok(())
}
