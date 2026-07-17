use anyhow::Context;
use hivemind_core::{Config, LayerRange};
use hivemind_daemon::node::{spawn_orchestrator, spawn_worker, WorkerConfig};
use std::time::Duration;
use tracing::info;

/// Mode and topology come from environment variables so the same binary can
/// run every role:
///
///   HIVEMIND_MODE=orchestrator | worker   (default: worker)
///   HIVEMIND_ORCHESTRATOR_URL=http://host:port   (workers)
///   HIVEMIND_LAYERS=start..end                   (workers, e.g. 0..4)
///   HIVEMIND_TOTAL_LAYERS=8
///   HIVEMIND_BIND=127.0.0.1:0
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_env("HIVEMIND_LOG")
                .add_directive("hivemind=info".parse()?),
        )
        .init();

    let config = Config::load().context("failed to load config — run `hivemind init` first")?;
    let mode = std::env::var("HIVEMIND_MODE").unwrap_or_else(|_| "worker".into());
    let total_layers: u32 = std::env::var("HIVEMIND_TOTAL_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let bind = std::env::var("HIVEMIND_BIND")
        .unwrap_or_else(|_| format!("0.0.0.0:{}", config.network.listen_port));

    match mode.as_str() {
        "orchestrator" => {
            let handle = spawn_orchestrator(&bind, &config.model.name, total_layers).await?;
            info!(url = %handle.url, model = %config.model.name, "orchestrator running");
            wait_for_shutdown_signal().await?;
            handle.shutdown();
        }
        _ => {
            let orchestrator_url = std::env::var("HIVEMIND_ORCHESTRATOR_URL")
                .context("HIVEMIND_ORCHESTRATOR_URL is required in worker mode")?;
            let layers = std::env::var("HIVEMIND_LAYERS").unwrap_or_else(|_| "0..8".into());
            let (start, end) = layers
                .split_once("..")
                .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
                .context("HIVEMIND_LAYERS must look like 0..4")?;

            let handle = spawn_worker(WorkerConfig {
                model_name: config.model.name.clone(),
                total_layers,
                layer_range: LayerRange::new(start, end),
                orchestrator_url,
                bind,
                heartbeat_every: Duration::from_secs(30),
            })
            .await?;
            info!(node = %handle.node_id, url = %handle.url, "worker running");

            wait_for_shutdown_signal().await?;
            // Most departures are lid-closes and shutdowns, not crashes:
            // drain in-flight work and tell the orchestrator we're going.
            info!("shutdown signal — draining before departure");
            handle.drain_and_depart(Duration::from_secs(30)).await;
            info!("departed cleanly");
        }
    }
    Ok(())
}

/// Resolves on ctrl-c everywhere, plus SIGTERM on unix (systemd stop, OS
/// shutdown).
async fn wait_for_shutdown_signal() -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            r = tokio::signal::ctrl_c() => r?,
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}
