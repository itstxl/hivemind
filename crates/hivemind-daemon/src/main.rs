mod scheduler;
mod server;

use anyhow::Context;
use hivemind_core::{Config, LayerRange};
use hivemind_ledger::Wallet;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_env("HIVEMIND_LOG")
                .add_directive("hivemind=info".parse()?),
        )
        .init();

    let config = Config::load().context("failed to load config — run `hivemind init` first")?;

    info!(node_id = %config.node.id, node_name = %config.node.name, "hivemind daemon starting");

    // TODO: determine layer assignment from network orchestrator based on hardware profile
    // For now, assume the node serves a fixed shard assigned during `hivemind init`
    let layer_range = LayerRange::new(0, 8); // placeholder

    let wallet = Wallet::default();
    let state = server::ServerState::new(config.clone(), wallet, layer_range);

    let addr = format!("0.0.0.0:{}", config.network.listen_port)
        .parse()
        .context("invalid listen address")?;

    let _scheduler = scheduler::ResourceScheduler::new(config.network.max_concurrent_pipelines);

    server::serve(addr, state).await
}
