use anyhow::Result;
use clap::Args;
use hivemind_core::Config;

#[derive(Args)]
pub struct ChatArgs {
    /// Model to use (overrides config)
    #[arg(long)]
    pub model: Option<String>,

    /// Temperature for sampling (0.0–2.0)
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,
}

pub async fn run(args: &ChatArgs) -> Result<()> {
    let config = Config::load().unwrap_or_else(|_| {
        eprintln!("No config found — run `hivemind init` first.");
        Config::generate("local", "local-node")
    });

    let model_name = args
        .model
        .as_deref()
        .unwrap_or(&config.model.name)
        .to_string();

    // Run the TUI in a blocking task so the async runtime stays free
    let temperature = args.temperature;
    tokio::task::spawn_blocking(move || {
        crate::ui::chat_tui::run_blocking(&model_name, temperature)
    })
    .await??;

    Ok(())
}
