//! Network-wide layer-coverage accounting and prefetch planning.
//!
//! Warm-standby failover only works if standbys exist, so the orchestrator
//! treats layer redundancy as a global concern rather than a per-pipeline
//! one: it watches how many live nodes serve each layer, and when a range
//! falls below the redundancy target it directs idle nodes with spare memory
//! to pre-fetch those weights *before* anyone fails. Weight downloads are the
//! slow, catastrophic part of recovery — they should never happen on the
//! critical path. Directives ride back on heartbeat responses
//! (`DiscoveryService.Heartbeat`).

use crate::peer::PeerInfo;
use hivemind_core::{LayerRange, NodeId};

/// Per-layer replica counts across the live peer set.
#[derive(Debug, Clone)]
pub struct LayerCoverage {
    counts: Vec<u32>,
}

impl LayerCoverage {
    /// Computes how many of `peers` serve each layer of `[0, total_layers)`.
    pub fn from_peers(peers: &[PeerInfo], total_layers: u32) -> Self {
        let mut counts = vec![0u32; total_layers as usize];
        for p in peers {
            for l in p.layer_range.start..p.layer_range.end.min(total_layers) {
                counts[l as usize] += 1;
            }
        }
        Self { counts }
    }

    /// The lowest replica count of any layer — the network's weakest link.
    pub fn min_replication(&self) -> u32 {
        self.counts.iter().copied().min().unwrap_or(0)
    }

    /// Contiguous ranges where fewer than `target` nodes serve the layers.
    pub fn under_replicated(&self, target: u32) -> Vec<LayerRange> {
        let mut ranges = Vec::new();
        let mut run_start: Option<u32> = None;
        for (i, &c) in self.counts.iter().enumerate() {
            if c < target {
                run_start.get_or_insert(i as u32);
            } else if let Some(start) = run_start.take() {
                ranges.push(LayerRange::new(start, i as u32));
            }
        }
        if let Some(start) = run_start {
            ranges.push(LayerRange::new(start, self.counts.len() as u32));
        }
        ranges
    }

    /// Total shortfall against `target` across all layers.
    pub fn deficit(&self, target: u32) -> u64 {
        self.counts
            .iter()
            .map(|&c| target.saturating_sub(c) as u64)
            .sum()
    }
}

/// An idle node with spare memory available for pre-fetching weights.
#[derive(Debug, Clone)]
pub struct PrefetchCandidate {
    pub node_id: NodeId,
    /// How many additional layers this node's spare budget can hold.
    pub max_layers: u32,
}

/// Instruction for one node to pre-load a contiguous layer range.
/// Delivered in `HeartbeatResponse.prefetch`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchDirective {
    pub node_id: NodeId,
    pub layer_range: LayerRange,
}

/// Plans prefetch assignments that raise every layer toward `target`
/// replicas.
///
/// Greedy: each candidate (largest spare budget first) takes the contiguous
/// window, at most `max_layers` wide, with the greatest remaining shortfall.
/// Candidates that can no longer help are skipped, so quiet periods produce
/// no directives.
pub fn plan_prefetch(
    coverage: &LayerCoverage,
    target: u32,
    candidates: &[PrefetchCandidate],
) -> Vec<PrefetchDirective> {
    let mut counts = coverage.counts.clone();
    let total_layers = counts.len();
    let mut directives = Vec::new();
    if total_layers == 0 {
        return directives;
    }

    let mut ordered: Vec<&PrefetchCandidate> = candidates.iter().collect();
    ordered.sort_by_key(|c| std::cmp::Reverse(c.max_layers));

    for cand in ordered {
        let width = (cand.max_layers as usize).min(total_layers);
        if width == 0 {
            continue;
        }

        // Sliding-window sum of remaining deficit; pick the neediest window.
        let deficit_at =
            |i: usize| -> u64 { target.saturating_sub(counts[i]) as u64 };
        let mut window: u64 = (0..width).map(deficit_at).sum();
        let mut best = (window, 0usize);
        for start in 1..=(total_layers - width) {
            window = window - deficit_at(start - 1) + deficit_at(start + width - 1);
            if window > best.0 {
                best = (window, start);
            }
        }

        if best.0 == 0 {
            continue; // nothing left for this candidate to improve
        }
        let range = LayerRange::new(best.1 as u32, (best.1 + width) as u32);
        for l in range.start..range.end {
            counts[l as usize] += 1;
        }
        directives.push(PrefetchDirective { node_id: cand.node_id, layer_range: range });
    }

    directives
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn peer(start: u32, end: u32) -> PeerInfo {
        PeerInfo {
            node_id: Uuid::new_v4(),
            peer_id: None,
            addrs: vec![],
            layer_range: LayerRange::new(start, end),
            latency_ms: Some(10),
            reputation: 80,
            survival: 1.0,
        }
    }

    #[test]
    fn detects_under_replicated_ranges() {
        // Layers 0..40 double-covered, 40..80 single-covered.
        let peers = vec![peer(0, 40), peer(0, 40), peer(40, 80)];
        let cov = LayerCoverage::from_peers(&peers, 80);
        assert_eq!(cov.min_replication(), 1);
        assert_eq!(cov.under_replicated(2), vec![LayerRange::new(40, 80)]);
        assert!(cov.under_replicated(1).is_empty());
    }

    #[test]
    fn plans_prefetch_into_weakest_window() {
        let peers = vec![peer(0, 40), peer(0, 40), peer(40, 80)];
        let cov = LayerCoverage::from_peers(&peers, 80);
        let cand = PrefetchCandidate { node_id: Uuid::new_v4(), max_layers: 40 };
        let plan = plan_prefetch(&cov, 2, std::slice::from_ref(&cand));
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].node_id, cand.node_id);
        assert_eq!(plan[0].layer_range, LayerRange::new(40, 80));
    }

    #[test]
    fn no_directives_when_target_met() {
        let peers = vec![peer(0, 80), peer(0, 80)];
        let cov = LayerCoverage::from_peers(&peers, 80);
        let cand = PrefetchCandidate { node_id: Uuid::new_v4(), max_layers: 40 };
        assert!(plan_prefetch(&cov, 2, &[cand]).is_empty());
    }

    #[test]
    fn successive_candidates_spread_across_deficit() {
        // Nothing served at all; two candidates should cover different halves.
        let cov = LayerCoverage::from_peers(&[], 80);
        let a = PrefetchCandidate { node_id: Uuid::new_v4(), max_layers: 40 };
        let b = PrefetchCandidate { node_id: Uuid::new_v4(), max_layers: 40 };
        let plan = plan_prefetch(&cov, 1, &[a, b]);
        assert_eq!(plan.len(), 2);
        assert!(!plan[0].layer_range.overlaps(&plan[1].layer_range));
        assert_eq!(
            plan[0].layer_range.len() + plan[1].layer_range.len(),
            80
        );
    }

    #[test]
    fn small_candidate_takes_partial_window() {
        let peers = vec![peer(0, 60), peer(0, 60), peer(60, 80)];
        let cov = LayerCoverage::from_peers(&peers, 80);
        let cand = PrefetchCandidate { node_id: Uuid::new_v4(), max_layers: 10 };
        let plan = plan_prefetch(&cov, 2, &[cand]);
        assert_eq!(plan.len(), 1);
        let r = plan[0].layer_range;
        assert_eq!(r.len(), 10);
        assert!(r.start >= 60, "window must sit inside the deficit, got {r}");
    }
}
