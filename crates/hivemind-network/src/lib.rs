pub mod coverage;
pub mod failover;
pub mod grpc;
pub mod health;
pub mod peer;
pub mod pipeline;
pub mod survival;
pub mod transport;

use async_trait::async_trait;
use hivemind_core::{LayerRange, NodeId, PipelineId, Result, Tensor};
use peer::PeerInfo;
use pipeline::Pipeline;

/// Core network interface used by both the daemon and CLI.
///
/// Implemented by [`MockNetwork`] for development and by the real libp2p
/// backend once the DHT and transport layers are wired up.
#[async_trait]
pub trait NetworkService: Send + Sync {
    /// Queries the DHT for peers that can serve the given layer range.
    async fn discover_peers(&self, range: LayerRange) -> Result<Vec<PeerInfo>>;

    /// Assembles an ordered pipeline covering all model layers.
    async fn assemble_pipeline(
        &self,
        requester: NodeId,
        total_layers: u32,
    ) -> Result<Pipeline>;

    /// Forwards an activation tensor to the next pipeline node and awaits
    /// the output tensor.
    async fn forward_activations(
        &self,
        destination: NodeId,
        pipeline_id: PipelineId,
        tensor: Tensor,
    ) -> Result<Tensor>;

    /// Reports this node's health metrics to the network.
    async fn report_health(&self, node_id: NodeId) -> Result<()>;
}

/// In-process mock that returns plausible fake data for CLI development.
pub struct MockNetwork {
    pub local_node_id: NodeId,
}

impl MockNetwork {
    pub fn new(local_node_id: NodeId) -> Self {
        Self { local_node_id }
    }
}

#[async_trait]
impl NetworkService for MockNetwork {
    async fn discover_peers(&self, range: LayerRange) -> Result<Vec<PeerInfo>> {
        use uuid::Uuid;
        // Three fake primaries evenly covering the range, plus a warm standby
        // for each third so assembled pipelines exercise failover paths.
        let chunk = range.len() / 3;
        let mut port = 4002;
        let mut peers = Vec::new();
        for (i, (start, end)) in [
            (range.start, range.start + chunk),
            (range.start + chunk, range.start + 2 * chunk),
            (range.start + 2 * chunk, range.end),
        ]
        .into_iter()
        .enumerate()
        {
            for replica in 0..2u32 {
                peers.push(PeerInfo {
                    node_id: Uuid::new_v4(),
                    peer_id: None,
                    addrs: vec![format!("/ip4/127.0.0.1/tcp/{port}")],
                    layer_range: LayerRange::new(start, end),
                    latency_ms: Some(12 + 5 * i as u32 + 10 * replica),
                    reputation: 90,
                    survival: 0.95 - 0.1 * replica as f32,
                });
                port += 1;
            }
        }
        Ok(peers)
    }

    async fn assemble_pipeline(
        &self,
        requester: NodeId,
        total_layers: u32,
    ) -> Result<Pipeline> {
        let peers = self.discover_peers(LayerRange::new(0, total_layers)).await?;
        let spec = pipeline::PipelineSpec::new(total_layers, 500);
        pipeline::assemble_pipeline(&spec, &peers, requester)
    }

    async fn forward_activations(
        &self,
        _destination: NodeId,
        _pipeline_id: PipelineId,
        input: Tensor,
    ) -> Result<Tensor> {
        // Echo back a zero tensor of the same shape (mock inference output)
        Ok(Tensor::zeros(input.shape.clone()))
    }

    async fn report_health(&self, _node_id: NodeId) -> Result<()> {
        Ok(())
    }
}
