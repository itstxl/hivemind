use hivemind_core::NodeId;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Point-in-time health snapshot for a remote node.
#[derive(Debug, Clone)]
pub struct NodeHealth {
    pub online: bool,
    pub latency_ms: Option<u32>,
    pub uptime_fraction: f32,
    pub last_seen: Instant,
}

impl NodeHealth {
    pub fn is_stale(&self, threshold: Duration) -> bool {
        self.last_seen.elapsed() > threshold
    }
}

/// Tracks health of known peers and exposes a simple query interface.
pub struct HealthTracker {
    records: HashMap<NodeId, NodeHealth>,
    stale_threshold: Duration,
}

impl HealthTracker {
    pub fn new(stale_threshold: Duration) -> Self {
        Self {
            records: HashMap::new(),
            stale_threshold,
        }
    }

    /// Records an observed latency for a peer, marking it online.
    pub fn record_latency(&mut self, node_id: NodeId, latency_ms: u32) {
        let entry = self.records.entry(node_id).or_insert(NodeHealth {
            online: false,
            latency_ms: None,
            uptime_fraction: 0.0,
            last_seen: Instant::now(),
        });
        entry.online = true;
        entry.latency_ms = Some(latency_ms);
        entry.last_seen = Instant::now();
    }

    /// Marks a peer as unreachable.
    pub fn record_failure(&mut self, node_id: NodeId) {
        let entry = self.records.entry(node_id).or_insert(NodeHealth {
            online: false,
            latency_ms: None,
            uptime_fraction: 0.0,
            last_seen: Instant::now(),
        });
        entry.online = false;
        entry.last_seen = Instant::now();
    }

    pub fn get(&self, node_id: &NodeId) -> Option<&NodeHealth> {
        self.records.get(node_id)
    }

    /// Returns all nodes currently considered healthy (online and not stale).
    pub fn healthy_nodes(&self) -> Vec<NodeId> {
        self.records
            .iter()
            .filter(|(_, h)| h.online && !h.is_stale(self.stale_threshold))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Sends a ping to a node and records the result.
    ///
    /// TODO: implement with libp2p ping protocol.
    pub async fn ping(&mut self, _node_id: NodeId) -> hivemind_core::Result<u32> {
        Err(hivemind_core::HivemindError::Network(
            "ping not yet implemented".into(),
        ))
    }
}
