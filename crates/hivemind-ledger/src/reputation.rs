use hivemind_core::NodeId;
use std::collections::HashMap;

/// Reputation score in the range `[0, 100]`.
///
/// Higher scores unlock priority placement in pipeline assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReputationScore(pub u8);

impl ReputationScore {
    pub const MIN: Self = Self(0);
    pub const MAX: Self = Self(100);
    pub const DEFAULT: Self = Self(50);

    /// Adjusts the score for a successful pipeline contribution.
    pub fn record_success(self) -> Self {
        Self(self.0.saturating_add(1).min(100))
    }

    /// Adjusts the score for a failed or timed-out contribution
    /// (transient: overloaded, slow link).
    pub fn record_failure(self) -> Self {
        Self(self.0.saturating_sub(5))
    }

    /// Adjusts the score for vanishing mid-pipeline without announcing
    /// departure. Costs far more than a transient failure: hard drops are
    /// what force failovers and KV-cache rebuilds on other people's sessions.
    pub fn record_hard_drop(self) -> Self {
        Self(self.0.saturating_sub(15))
    }

    /// A departure announced via `DiscoveryService.Depart` with in-flight
    /// work drained or handed off. Free — shutting down politely is the
    /// behaviour the network wants to be the rational default.
    pub fn record_graceful_exit(self) -> Self {
        self
    }
}

/// Local reputation ledger: tracks scores for all known peers.
pub struct ReputationLedger {
    scores: HashMap<NodeId, ReputationScore>,
}

impl ReputationLedger {
    pub fn new() -> Self {
        Self { scores: HashMap::new() }
    }

    pub fn score_of(&self, node_id: &NodeId) -> ReputationScore {
        *self.scores.get(node_id).unwrap_or(&ReputationScore::DEFAULT)
    }

    pub fn record_success(&mut self, node_id: NodeId) {
        let s = self.score_of(&node_id).record_success();
        self.scores.insert(node_id, s);
    }

    pub fn record_failure(&mut self, node_id: NodeId) {
        let s = self.score_of(&node_id).record_failure();
        self.scores.insert(node_id, s);
    }

    pub fn record_hard_drop(&mut self, node_id: NodeId) {
        let s = self.score_of(&node_id).record_hard_drop();
        self.scores.insert(node_id, s);
    }

    pub fn record_graceful_exit(&mut self, node_id: NodeId) {
        let s = self.score_of(&node_id).record_graceful_exit();
        self.scores.insert(node_id, s);
    }
}

impl Default for ReputationLedger {
    fn default() -> Self {
        Self::new()
    }
}
