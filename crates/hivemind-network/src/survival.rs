//! Session-history-based node survival estimation.
//!
//! `healthy_nodes()` answers "who is up *now*"; the scheduler actually needs
//! "who will still be up in twenty minutes". This module tracks each node's
//! past session lengths and estimates the probability it survives a given
//! horizon, conditioned on how long its current session has already lasted.
//! A desktop that has been online six hours a day all week is a better home
//! for a long conversation than a laptop that joined three minutes ago —
//! even if the laptop's latency is lower right now.

use hivemind_core::NodeId;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Completed sessions remembered per node.
const MAX_HISTORY: usize = 32;

/// Assumed session half-life for nodes with no usable history: a new node is
/// modelled as 50% likely to survive each additional 30 minutes.
const NEW_NODE_HALF_LIFE: Duration = Duration::from_secs(30 * 60);

#[derive(Default)]
struct NodeSessions {
    /// Durations of completed sessions, oldest first.
    completed: VecDeque<Duration>,
    /// When the current session began; `None` while offline.
    current_start: Option<Instant>,
}

/// Tracks join/leave events and estimates conditional survival probability.
#[derive(Default)]
pub struct SessionTracker {
    nodes: HashMap<NodeId, NodeSessions>,
}

impl SessionTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a node coming online.
    pub fn record_join(&mut self, node_id: NodeId, now: Instant) {
        self.nodes.entry(node_id).or_default().current_start = Some(now);
    }

    /// Records a node going offline (gracefully or not).
    pub fn record_leave(&mut self, node_id: NodeId, now: Instant) {
        let sessions = self.nodes.entry(node_id).or_default();
        if let Some(start) = sessions.current_start.take() {
            sessions.completed.push_back(now.duration_since(start));
            while sessions.completed.len() > MAX_HISTORY {
                sessions.completed.pop_front();
            }
        }
    }

    /// Estimates the probability that `node_id` is still online `horizon`
    /// from `now`, conditioned on its current session age.
    ///
    /// Empirical: of the past sessions that lasted at least as long as the
    /// current one has, what fraction went on to last `horizon` longer?
    /// Falls back to an exponential prior when history can't answer
    /// (brand-new node, or one that has outlived everything on record).
    /// Returns 0.0 for nodes that are offline.
    pub fn survival_probability(
        &self,
        node_id: NodeId,
        now: Instant,
        horizon: Duration,
    ) -> f64 {
        let Some(sessions) = self.nodes.get(&node_id) else {
            return 0.0;
        };
        let Some(start) = sessions.current_start else {
            return 0.0;
        };
        let age = now.duration_since(start);

        let reached_age = sessions.completed.iter().filter(|d| **d >= age).count();
        if reached_age == 0 {
            return exponential_prior(horizon);
        }
        let survived = sessions
            .completed
            .iter()
            .filter(|d| **d >= age + horizon)
            .count();
        survived as f64 / reached_age as f64
    }

    /// Joint survival probability of a whole pipeline: every node must live.
    pub fn pipeline_survival(
        &self,
        nodes: impl IntoIterator<Item = NodeId>,
        now: Instant,
        horizon: Duration,
    ) -> f64 {
        nodes
            .into_iter()
            .map(|n| self.survival_probability(n, now, horizon))
            .product()
    }

    /// Stamps each peer's `survival` estimate before pipeline assembly, so
    /// the greedy assembler prefers nodes likely to outlive the session.
    pub fn annotate_peers(
        &self,
        peers: &mut [crate::peer::PeerInfo],
        now: Instant,
        horizon: Duration,
    ) {
        for p in peers {
            p.survival = self.survival_probability(p.node_id, now, horizon) as f32;
        }
    }
}

fn exponential_prior(horizon: Duration) -> f64 {
    0.5f64.powf(horizon.as_secs_f64() / NEW_NODE_HALF_LIFE.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    const MIN: Duration = Duration::from_secs(60);

    #[test]
    fn offline_and_unknown_nodes_have_zero_survival() {
        let mut t = SessionTracker::new();
        let now = Instant::now();
        assert_eq!(t.survival_probability(Uuid::new_v4(), now, 10 * MIN), 0.0);

        let n = Uuid::new_v4();
        t.record_join(n, now);
        t.record_leave(n, now + 5 * MIN);
        assert_eq!(t.survival_probability(n, now + 6 * MIN, 10 * MIN), 0.0);
    }

    #[test]
    fn new_node_gets_exponential_prior() {
        let mut t = SessionTracker::new();
        let now = Instant::now();
        let n = Uuid::new_v4();
        t.record_join(n, now);
        let p = t.survival_probability(n, now, Duration::from_secs(30 * 60));
        assert!((p - 0.5).abs() < 1e-9, "one half-life ⇒ 0.5, got {p}");
    }

    #[test]
    fn history_conditions_on_current_session_age() {
        let mut t = SessionTracker::new();
        let base = Instant::now();
        let n = Uuid::new_v4();
        // Sessions: 10, 20, 40, 60 minutes.
        for (i, mins) in [10u64, 20, 40, 60].into_iter().enumerate() {
            let start = base + (i as u32 * 100) * MIN;
            t.record_join(n, start);
            t.record_leave(n, start + mins as u32 * MIN);
        }
        let start = base + 1000 * MIN;
        t.record_join(n, start);

        // 15 min into the session, only the 20/40/60-min sessions are
        // comparable; 2 of 3 lasted ≥ 15+20 = 35 min.
        let p = t.survival_probability(n, start + 15 * MIN, 20 * MIN);
        assert!((p - 2.0 / 3.0).abs() < 1e-9, "got {p}");

        // A brand-new session: all 4 reached age 0; 2 of 4 lasted ≥ 30 min.
        let p0 = t.survival_probability(n, start, 30 * MIN);
        assert!((p0 - 0.5).abs() < 1e-9, "got {p0}");
    }

    #[test]
    fn node_outliving_all_history_falls_back_to_prior() {
        let mut t = SessionTracker::new();
        let base = Instant::now();
        let n = Uuid::new_v4();
        t.record_join(n, base);
        t.record_leave(n, base + 10 * MIN);
        t.record_join(n, base + 100 * MIN);
        // 50 minutes in — longer than any completed session.
        let p = t.survival_probability(n, base + 150 * MIN, Duration::from_secs(30 * 60));
        assert!((p - 0.5).abs() < 1e-9, "exponential prior expected, got {p}");
    }

    #[test]
    fn pipeline_survival_is_joint_product() {
        let mut t = SessionTracker::new();
        let now = Instant::now();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        t.record_join(a, now);
        t.record_join(b, now);
        let horizon = Duration::from_secs(30 * 60);
        let joint = t.pipeline_survival([a, b], now, horizon);
        assert!((joint - 0.25).abs() < 1e-9, "0.5 * 0.5 expected, got {joint}");
    }
}
