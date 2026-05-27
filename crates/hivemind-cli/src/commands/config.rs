use anyhow::{bail, Result};
use clap::{Args, Subcommand};
use hivemind_core::Config;

#[derive(Args)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub action: ConfigAction,
}

#[derive(Subcommand)]
pub enum ConfigAction {
    /// Print the current configuration
    Show,
    /// Get a single value by dot-path (e.g. hardware.gpu_allocation)
    Get {
        key: String,
    },
    /// Set a value by dot-path (e.g. hardware.gpu_allocation 0.5)
    Set {
        key: String,
        value: String,
    },
    /// Print the path to the config file
    Path,
}

pub async fn run(args: &ConfigArgs) -> Result<()> {
    match &args.action {
        ConfigAction::Show => {
            let config = Config::load().unwrap_or_else(|_| Config::generate("local", "local-node"));
            let raw = toml::to_string_pretty(&config)?;
            print!("{raw}");
        }

        ConfigAction::Get { key } => {
            let config = Config::load().unwrap_or_else(|_| Config::generate("local", "local-node"));
            let value = get_config_value(&config, key)?;
            println!("{value}");
        }

        ConfigAction::Set { key, value } => {
            let mut config = Config::load().unwrap_or_else(|_| Config::generate("local", "local-node"));
            set_config_value(&mut config, key, value)?;
            config.save()?;
            println!("Set {key} = {value}");
        }

        ConfigAction::Path => {
            let path = Config::default_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(unknown)".into());
            println!("{path}");
        }
    }
    Ok(())
}

fn get_config_value(config: &Config, key: &str) -> Result<String> {
    match key {
        "node.id"                        => Ok(config.node.id.clone()),
        "node.name"                      => Ok(config.node.name.clone()),
        "hardware.gpu_allocation"        => Ok(config.hardware.gpu_allocation.to_string()),
        "hardware.max_cpu_threads"       => Ok(config.hardware.max_cpu_threads.to_string()),
        "hardware.max_bandwidth_mbps"    => Ok(config.hardware.max_bandwidth_mbps.to_string()),
        "model.name"                     => Ok(config.model.name.clone()),
        "model.quantization"             => Ok(config.model.quantization.clone()),
        "model.shard_path"               => Ok(config.model.shard_path.display().to_string()),
        "network.listen_port"            => Ok(config.network.listen_port.to_string()),
        "network.max_concurrent_pipelines" => Ok(config.network.max_concurrent_pipelines.to_string()),
        "tokens.auto_earn"               => Ok(config.tokens.auto_earn.to_string()),
        _ => bail!("unknown config key `{key}` — run `hivemind config show` to see all keys"),
    }
}

fn set_config_value(config: &mut Config, key: &str, value: &str) -> Result<()> {
    match key {
        "node.name"                   => config.node.name = value.to_string(),
        "hardware.gpu_allocation"     => config.hardware.gpu_allocation = value.parse()?,
        "hardware.max_cpu_threads"    => config.hardware.max_cpu_threads = value.parse()?,
        "hardware.max_bandwidth_mbps" => config.hardware.max_bandwidth_mbps = value.parse()?,
        "model.name"                  => config.model.name = value.to_string(),
        "model.quantization"          => config.model.quantization = value.to_string(),
        "network.listen_port"         => config.network.listen_port = value.parse()?,
        "network.max_concurrent_pipelines" => config.network.max_concurrent_pipelines = value.parse()?,
        "tokens.auto_earn"            => config.tokens.auto_earn = value.parse()?,
        "node.id" => bail!("`node.id` is read-only"),
        _ => bail!("unknown config key `{key}`"),
    }
    Ok(())
}
