//! Generated gRPC/protobuf types for the Hivemind wire protocol.
//!
//! Source of truth is the `.proto` files in `proto/`; this crate exposes the
//! tonic-generated clients, servers, and messages for the rest of the
//! workspace.

pub mod activations {
    tonic::include_proto!("hivemind.activations");
}

pub mod routing {
    tonic::include_proto!("hivemind.routing");
}

pub mod discovery {
    tonic::include_proto!("hivemind.discovery");
}

pub mod tokens {
    tonic::include_proto!("hivemind.tokens");
}

impl From<&hivemind_core::Tensor> for activations::Tensor {
    fn from(t: &hivemind_core::Tensor) -> Self {
        Self {
            data: t.data.clone(),
            shape: t.shape.iter().map(|&d| d as u32).collect(),
        }
    }
}

impl From<&activations::Tensor> for hivemind_core::Tensor {
    fn from(t: &activations::Tensor) -> Self {
        hivemind_core::Tensor::new(
            t.data.clone(),
            t.shape.iter().map(|&d| d as usize).collect(),
        )
    }
}
