use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level configuration, loaded from `~/.hivemind/config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub node: NodeConfig,
    pub hardware: HardwareConfig,
    pub model: ModelConfig,
    pub network: NetworkConfig,
    pub tokens: TokensConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Fraction of VRAM to donate to the network (0.0–1.0).
    pub gpu_allocation: f64,
    pub max_cpu_threads: u32,
    pub max_bandwidth_mbps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub quantization: String,
    pub shard_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub bootstrap_nodes: Vec<String>,
    pub listen_port: u16,
    pub max_concurrent_pipelines: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokensConfig {
    pub auto_earn: bool,
}

impl Config {
    /// Returns `~/.hivemind/config.toml`.
    pub fn default_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".hivemind").join("config.toml"))
    }

    pub fn load() -> crate::error::Result<Self> {
        let path = Self::default_path().ok_or_else(|| {
            crate::error::HivemindError::Config("cannot determine home directory".into())
        })?;
        Self::load_from(&path)
    }

    pub fn load_from(path: &Path) -> crate::error::Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| {
            crate::error::HivemindError::Config(format!("read {}: {e}", path.display()))
        })?;
        toml::from_str(&raw).map_err(|e| {
            crate::error::HivemindError::Config(format!("parse config: {e}"))
        })
    }

    pub fn save(&self) -> crate::error::Result<()> {
        let path = Self::default_path().ok_or_else(|| {
            crate::error::HivemindError::Config("cannot determine home directory".into())
        })?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self).map_err(|e| {
            crate::error::HivemindError::Config(format!("serialize config: {e}"))
        })?;
        std::fs::write(&path, contents)?;
        Ok(())
    }

    /// Generates a sensible default configuration for a new node.
    pub fn generate(node_id: &str, node_name: &str) -> Self {
        let shard_path = dirs::home_dir()
            .unwrap_or_default()
            .join(".hivemind")
            .join("shards");
        Self {
            node: NodeConfig {
                id: node_id.to_string(),
                name: node_name.to_string(),
            },
            hardware: HardwareConfig {
                gpu_allocation: 0.8,
                max_cpu_threads: 4,
                max_bandwidth_mbps: 100,
            },
            model: ModelConfig {
                name: "qwen2.5-coder-72b".to_string(),
                quantization: "q4_k_m".to_string(),
                shard_path,
            },
            network: NetworkConfig {
                bootstrap_nodes: vec![
                    "/ip4/seed1.hivemind.dev/tcp/4001".to_string(),
                    "/ip4/seed2.hivemind.dev/tcp/4001".to_string(),
                ],
                listen_port: 4001,
                max_concurrent_pipelines: 3,
            },
            tokens: TokensConfig { auto_earn: true },
        }
    }
}
