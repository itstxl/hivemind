use hivemind_core::{LayerRange, NodeId, PipelineId, PipelineSlot};
use crate::peer::PeerInfo;
use uuid::Uuid;

/// Parameters for assembling a pipeline.
#[derive(Debug, Clone)]
pub struct PipelineSpec {
    pub total_layers: u32,
    /// Maximum acceptable end-to-end latency in milliseconds.
    pub max_latency_ms: u32,
}

/// An assembled, ordered chain of pipeline slots covering all model layers.
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub id: PipelineId,
    pub slots: Vec<PipelineSlot>,
    pub requester: NodeId,
}

impl Pipeline {
    /// Returns true when every layer `[0, total_layers)` is covered with no gaps.
    pub fn is_complete(&self, total_layers: u32) -> bool {
        let mut covered: Vec<bool> = vec![false; total_layers as usize];
        for slot in &self.slots {
            for l in slot.layer_range.start..slot.layer_range.end {
                if (l as usize) < covered.len() {
                    covered[l as usize] = true;
                }
            }
        }
        covered.iter().all(|&c| c)
    }
}

/// Assembles an ordered pipeline from a candidate peer list.
///
/// TODO: implement greedy layer-covering algorithm:
/// 1. Sort peers by latency.
/// 2. Greedily assign layers from 0..total_layers, preferring low-latency peers.
/// 3. Fall back to splitting large layer ranges across multiple peers if needed.
pub fn assemble_pipeline(
    spec: &PipelineSpec,
    candidates: &[PeerInfo],
    requester: NodeId,
) -> hivemind_core::Result<Pipeline> {
    if candidates.is_empty() {
        return Err(hivemind_core::HivemindError::Pipeline(
            "no candidate peers available".into(),
        ));
    }

    // TODO: real covering algorithm
    // For now, build a placeholder pipeline from whatever candidates are available.
    let slots: Vec<PipelineSlot> = candidates
        .iter()
        .enumerate()
        .map(|(i, p)| PipelineSlot {
            node_id: p.node_id,
            layer_range: p.layer_range,
            position: i,
        })
        .collect();

    Ok(Pipeline {
        id: Uuid::new_v4(),
        slots,
        requester,
    })
}

/// Checks whether a set of peers can collectively cover all `total_layers`.
pub fn can_cover_layers(peers: &[PeerInfo], total_layers: u32) -> bool {
    let mut covered: Vec<bool> = vec![false; total_layers as usize];
    for p in peers {
        for l in p.layer_range.start..p.layer_range.end.min(total_layers) {
            covered[l as usize] = true;
        }
    }
    covered.iter().all(|&c| c)
}
