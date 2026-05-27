use anyhow::{Context, Result};
use clap::Args;
use hivemind_core::Config;
use hivemind_shard::detect_hardware;
use uuid::Uuid;

#[derive(Args)]
pub struct InitArgs {
    /// Override the node name (default: auto-generated from hostname)
    #[arg(long)]
    pub name: Option<String>,

    /// Skip downloading the model shard (useful for testing)
    #[arg(long)]
    pub no_download: bool,
}

pub async fn run(args: &InitArgs) -> Result<()> {
    println!("  Hivemind — initializing node\n");

    // Step 1: hardware detection
    let spinner = crate::ui::spinner::Spinner::new("Detecting hardware...");
    let profile = detect_hardware().context("hardware detection failed")?;
    spinner.finish("Hardware detected");

    if let Some(ref gpu) = profile.gpu_model {
        println!("  GPU   : {gpu}");
        if let Some(vram) = profile.vram_mb {
            println!("  VRAM  : {} MB", vram);
        }
        if let Some(cc) = profile.compute_capability {
            println!("  CUDA  : {}.{}", cc.0, cc.1);
        }
    } else {
        println!("  GPU   : none detected (CPU-only node)");
    }
    println!("  RAM   : {} MB", profile.ram_mb);

    let budget_mb = profile.shard_budget_mb(0.8);
    println!("  Shard budget: {} MB\n", budget_mb);

    // Step 2: generate node identity
    let node_id = Uuid::new_v4().to_string();
    let node_name = args
        .name
        .clone()
        .unwrap_or_else(|| generate_node_name(&node_id));

    println!("  Node ID   : {node_id}");
    println!("  Node name : {node_name}\n");

    // Step 3: write config
    let config = Config::generate(&node_id, &node_name);
    config.save().context("failed to write config")?;

    let config_path = Config::default_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "~/.hivemind/config.toml".into());

    println!("  Config written to: {config_path}\n");

    // Step 4: shard download
    if args.no_download {
        println!("  Skipping shard download (--no-download).");
    } else {
        println!("  Shard download not yet implemented.");
        println!("  TODO: hivemind will fetch layers for {budget_mb} MB budget automatically.");
    }

    println!("\n  Run `hivemind chat` to start coding.");
    Ok(())
}

/// Generates a human-readable node name like `node-amsterdam-a1b2`.
fn generate_node_name(node_id: &str) -> String {
    const CITIES: &[&str] = &[
        "amsterdam", "berlin", "boston", "cairo", "dallas", "dublin",
        "houston", "lagos", "lima", "london", "madrid", "miami",
        "milan", "montreal", "mumbai", "nairobi", "osaka", "paris",
        "prague", "rotterdam", "seoul", "singapore", "stockholm",
        "sydney", "tokyo", "toronto", "vienna", "warsaw", "zurich",
    ];
    let suffix = &node_id[..4];
    let city_idx = node_id
        .bytes()
        .fold(0usize, |acc, b| acc.wrapping_add(b as usize))
        % CITIES.len();
    format!("node-{}-{}", CITIES[city_idx], suffix)
}
