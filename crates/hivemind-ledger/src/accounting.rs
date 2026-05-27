use hivemind_core::MicroToken;
use std::time::Duration;

/// A record of compute work done on behalf of the network.
#[derive(Debug, Clone)]
pub struct ComputeContribution {
    /// Number of transformer layers processed.
    pub layers_processed: u32,
    /// Number of tokens in the processed sequence.
    pub sequence_length: u32,
    /// Wall-clock time taken.
    pub duration: Duration,
}

impl ComputeContribution {
    /// Computes the token reward for this contribution.
    ///
    /// Reward formula: `layers * seq_len * base_rate_per_layer_token`
    /// Base rate: 1 micro-token per (layer × token).
    ///
    /// TODO: tune against real inference cost benchmarks and network supply.
    pub fn earned_tokens(&self) -> MicroToken {
        let work_units = self.layers_processed as u64 * self.sequence_length as u64;
        MicroToken(work_units)
    }
}

/// Computes the inference cost for a consumer request.
///
/// Cost formula: `total_layers * seq_len * consumer_rate`
/// Consumer rate is 2× the earn rate (network takes a 50% cut).
///
/// TODO: implement dynamic pricing based on network congestion.
pub fn inference_cost(total_layers: u32, sequence_length: u32) -> MicroToken {
    let work_units = total_layers as u64 * sequence_length as u64;
    MicroToken(work_units * 2)
}

/// Reward rate in micro-tokens per (layer × token).
pub const EARN_RATE: u64 = 1;

/// Consumer cost rate in micro-tokens per (layer × token).
pub const SPEND_RATE: u64 = 2;
