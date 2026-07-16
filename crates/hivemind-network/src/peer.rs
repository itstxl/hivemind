use hivemind_core::{LayerRange, NodeId};
use libp2p::{Multiaddr, PeerId};
use serde::{Deserialize, Serialize};

/// Known information about a remote peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: NodeId,
    /// libp2p peer identity derived from the node's keypair.
    #[serde(skip)]
    pub peer_id: Option<PeerId>,
    /// Reachable addresses for this peer.
    pub addrs: Vec<String>,
    /// Transformer layers this peer is serving.
    pub layer_range: LayerRange,
    /// Last measured round-trip latency in milliseconds.
    pub latency_ms: Option<u32>,
    /// Peer's self-reported reputation score (0–100).
    pub reputation: u8,
    /// Estimated probability this peer stays online for the placement
    /// horizon, stamped locally from its session history before assembly
    /// (see `survival::SessionTracker::annotate_peers`). Defaults to 1.0
    /// (no history-based penalty).
    #[serde(default = "default_survival")]
    pub survival: f32,
}

fn default_survival() -> f32 {
    1.0
}

impl PeerInfo {
    pub fn multiaddrs(&self) -> Vec<Multiaddr> {
        self.addrs
            .iter()
            .filter_map(|a| a.parse().ok())
            .collect()
    }
}

/// Peer discovery backed by Kademlia DHT.
///
/// TODO: implement with `libp2p::kad::Behaviour`.
pub struct KademliaPeerDiscovery {
    // TODO: libp2p::Swarm<ComposedBehaviour>
}

impl KademliaPeerDiscovery {
    /// Creates a new peer discovery instance and connects to bootstrap nodes.
    ///
    /// TODO: implement Swarm setup with Kademlia + TCP + Noise + Yamux.
    pub async fn new(_bootstrap_nodes: &[String]) -> hivemind_core::Result<Self> {
        Err(hivemind_core::HivemindError::Network(
            "Kademlia DHT not yet implemented".into(),
        ))
    }

    /// Queries the DHT for peers serving the given layer range.
    ///
    /// TODO: encode layer range in the DHT record key and walk closest peers.
    pub async fn find_peers_for_layers(
        &mut self,
        _range: LayerRange,
    ) -> hivemind_core::Result<Vec<PeerInfo>> {
        Err(hivemind_core::HivemindError::Network(
            "DHT peer lookup not yet implemented".into(),
        ))
    }
}
