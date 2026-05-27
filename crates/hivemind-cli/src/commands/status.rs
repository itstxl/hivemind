use anyhow::Result;
use clap::Args;
use hivemind_core::{Config, HardwareProfile};
use hivemind_ledger::Wallet;
use hivemind_shard::detect_hardware;

#[derive(Args)]
pub struct StatusArgs {
    /// Output raw JSON instead of a formatted dashboard
    #[arg(long)]
    pub json: bool,
}

pub async fn run(args: &StatusArgs) -> Result<()> {
    let config = Config::load().unwrap_or_else(|_| Config::generate("local", "local-node"));
    let wallet = Wallet::default(); // TODO: load persisted wallet
    let profile = detect_hardware().unwrap_or(HardwareProfile {
        gpu_model: None,
        vram_mb: None,
        ram_mb: 0,
        compute_capability: None,
    });

    if args.json {
        print_json(&config, &wallet, &profile);
    } else {
        print_dashboard(&config, &wallet, &profile);
    }

    Ok(())
}

fn print_dashboard(
    config: &Config,
    wallet: &Wallet,
    profile: &HardwareProfile,
) {
    println!("  Hivemind Node Status\n");
    println!("  Node");
    println!("    ID    : {}", config.node.id);
    println!("    Name  : {}", config.node.name);
    println!("    Port  : {}", config.network.listen_port);

    println!("\n  Hardware");
    if let Some(ref gpu) = profile.gpu_model {
        println!("    GPU   : {gpu}");
        if let Some(vram) = profile.vram_mb {
            println!("    VRAM  : {} MB", vram);
        }
    } else {
        println!("    GPU   : none (CPU-only)");
    }
    println!("    RAM   : {} MB", profile.ram_mb);

    println!("\n  Tokens");
    println!("    Balance : {}", wallet.balance());

    println!("\n  Network");
    println!("    Model   : {}", config.model.name);
    println!("    Quant   : {}", config.model.quantization);
    println!("    Peers   : — (not yet connected)");
    println!("    Pipelines served today: —");
}

fn print_json(
    config: &Config,
    wallet: &Wallet,
    profile: &HardwareProfile,
) {
    // Minimal JSON without pulling in serde_json
    println!("{{");
    println!("  \"node_id\": \"{}\",", config.node.id);
    println!("  \"node_name\": \"{}\",", config.node.name);
    println!("  \"balance_micro\": {},", wallet.balance().0);
    println!("  \"gpu\": {},", profile.gpu_model.as_deref().map(|g| format!("\"{g}\"")).unwrap_or_else(|| "null".into()));
    println!("  \"ram_mb\": {}", profile.ram_mb);
    println!("}}");
}
