// Integration tests for the full inference pipeline across mock nodes.
//
// To run: add `hivemind-network` and `hivemind-core` as dev-dependencies
// in a test crate, then `cargo test --test pipeline_test`.

use hivemind_core::{LayerRange, MicroToken, Tensor};
use hivemind_network::{MockNetwork, NetworkService};
use uuid::Uuid;

#[tokio::test]
async fn mock_network_discovers_peers() {
    let net = MockNetwork::new(Uuid::new_v4());
    let peers = net.discover_peers(LayerRange::new(0, 80)).await.unwrap();
    assert!(!peers.is_empty(), "mock network should return peers");
    // Peers should collectively cover layer range
    assert!(hivemind_network::pipeline::can_cover_layers(&peers, 80));
}

#[tokio::test]
async fn mock_network_assembles_pipeline() {
    let local = Uuid::new_v4();
    let net = MockNetwork::new(local);
    let pipeline = net.assemble_pipeline(local, 80).await.unwrap();
    assert!(pipeline.is_complete(80), "assembled pipeline should cover all layers");
}

#[tokio::test]
async fn mock_network_forwards_activations() {
    let local = Uuid::new_v4();
    let net = MockNetwork::new(local);
    let input = Tensor::zeros(vec![1, 4096]);
    let output = net
        .forward_activations(Uuid::new_v4(), Uuid::new_v4(), input.clone())
        .await
        .unwrap();
    assert_eq!(output.shape, input.shape, "mock should echo the shape");
}

#[tokio::test]
async fn tensor_serialization_roundtrip() {
    use hivemind_network::transport::{deserialize_tensor, serialize_tensor};

    let original = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let bytes = serialize_tensor(&original);
    let decoded = deserialize_tensor(&bytes).unwrap();

    assert_eq!(decoded.shape, original.shape);
    assert_eq!(decoded.data, original.data);
}
