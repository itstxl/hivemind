use thiserror::Error;

pub type Result<T> = std::result::Result<T, HivemindError>;

#[derive(Debug, Error)]
pub enum HivemindError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("hardware detection failed: {0}")]
    Hardware(String),

    #[error("shard error: {0}")]
    Shard(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("ledger error: {0}")]
    Ledger(String),

    #[error("pipeline error: {0}")]
    Pipeline(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(String),
}
