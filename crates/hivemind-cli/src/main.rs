mod commands;
mod ui;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "hivemind",
    about = "Decentralized LLM inference — contribute compute, earn tokens, code with AI",
    version
)]
struct Cli {
    /// Log verbosity: error, warn, info, debug, trace
    #[arg(long, default_value = "warn", env = "HIVEMIND_LOG")]
    log: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect hardware, download your shard, and join the network
    Init(commands::init::InitArgs),
    /// Start an interactive coding-assistant chat session
    Chat(commands::chat::ChatArgs),
    /// Show node stats, token balance, and network health
    Status(commands::status::StatusArgs),
    /// Single-shot completion — reads prompt from stdin or argument
    Complete(commands::complete::CompleteArgs),
    /// Get or set configuration values
    Config(commands::config::ConfigArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log)),
        )
        .with_target(false)
        .without_time()
        .init();

    match &cli.command {
        Commands::Init(args)     => commands::init::run(args).await,
        Commands::Chat(args)     => commands::chat::run(args).await,
        Commands::Status(args)   => commands::status::run(args).await,
        Commands::Complete(args) => commands::complete::run(args).await,
        Commands::Config(args)   => commands::config::run(args).await,
    }
}
