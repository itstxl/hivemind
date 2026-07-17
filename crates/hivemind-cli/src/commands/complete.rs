use anyhow::{Context, Result};
use clap::Args;
use hivemind_core::Config;
use hivemind_network::grpc::{sample_greedy, PipelineSession};
use std::io::{self, Read, Write};

#[derive(Args)]
pub struct CompleteArgs {
    /// Prompt text. If omitted, reads from stdin.
    pub prompt: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "512")]
    pub max_tokens: u32,

    /// Temperature for sampling (currently greedy; kept for compatibility)
    #[arg(long, default_value = "0.2")]
    pub temperature: f32,
}

pub async fn run(args: &CompleteArgs) -> Result<()> {
    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    if prompt.trim().is_empty() {
        anyhow::bail!("prompt is empty — pass text as an argument or pipe it to stdin");
    }

    let config = Config::load().unwrap_or_else(|_| Config::generate("local", "local-node"));
    let orchestrator_url = std::env::var("HIVEMIND_ORCHESTRATOR_URL")
        .ok()
        .or_else(|| config.network.orchestrator_url.clone())
        .context(
            "no orchestrator configured — set HIVEMIND_ORCHESTRATOR_URL or \
             network.orchestrator_url in ~/.hivemind/config.toml",
        )?;

    // Reference model is byte-level: UTF-8 bytes are the token ids.
    let tokens: Vec<u32> = prompt.bytes().map(u32::from).collect();
    let mut session = PipelineSession::connect(&orchestrator_url, &config.model.name, tokens)
        .await
        .with_context(|| format!("failed to open pipeline via {orchestrator_url}"))?;

    eprintln!(
        "[hivemind] pipeline of {} hops assembled — generating up to {} tokens",
        session.pipeline().slots.len(),
        args.max_tokens
    );

    let mut stdout = io::stdout();
    for _ in 0..args.max_tokens {
        let logits = session.step().await?;
        let tok = sample_greedy(&logits)?;
        session.push_token(tok);
        stdout.write_all(&[tok as u8])?;
        stdout.flush()?;
    }
    stdout.write_all(b"\n")?;

    Ok(())
}
