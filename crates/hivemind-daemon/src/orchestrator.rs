//! Orchestrator mode: `DiscoveryService` + `RoutingService` backed by a live
//! node registry, session-history survival tracking, and the coverage-driven
//! prefetch planner. Initially run by bootstrap nodes; the registry is plain
//! in-process state so it can later be replaced by DHT-backed discovery
//! without touching the service surface.

#![allow(clippy::result_large_err)] // tonic handlers return Status by value

use hivemind_core::{LayerRange, NodeId};
use hivemind_ledger::ReputationLedger;
use hivemind_network::coverage::{plan_prefetch, LayerCoverage, PrefetchCandidate};
use hivemind_network::peer::PeerInfo;
use hivemind_network::pipeline::{assemble_pipeline, Pipeline, PipelineSpec};
use hivemind_network::survival::SessionTracker;
use hivemind_proto::discovery::discovery_service_server::DiscoveryService;
use hivemind_proto::discovery::{
    AnnounceRequest, AnnounceResponse, DepartRequest, DepartResponse, FindPeersRequest,
    FindPeersResponse, HeartbeatRequest, HeartbeatResponse, NodeCapabilities, PrefetchDirective,
};
use hivemind_proto::routing::routing_service_server::RoutingService;
use hivemind_proto::routing::{
    AssembleRequest, AssembleResponse, FailoverReport, FailoverResponse, PipelineAssignment,
    PipelineHealthRequest, PipelineHealthResponse, PipelineSlot as ProtoSlot, Standby,
};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tonic::{Request, Response, Status};
use tracing::{info, warn};
use uuid::Uuid;

/// Nodes silent for longer than this are treated as departed.
const STALE_AFTER: Duration = Duration::from_secs(90);
/// Placement horizon: how far ahead survival estimates look.
const PLACEMENT_HORIZON: Duration = Duration::from_secs(20 * 60);
/// Warm standbys the network tries to keep per layer.
const REDUNDANCY_TARGET: u32 = 2;

struct NodeEntry {
    caps: NodeCapabilities,
    last_seen: Instant,
    latency_ms: Option<u32>,
}

struct Registry {
    nodes: HashMap<NodeId, NodeEntry>,
    tracker: SessionTracker,
    reputation: ReputationLedger,
    /// Assignments handed out, kept so failover reports can be resolved to a
    /// layer range when refilling standby pools.
    pipelines: HashMap<Uuid, Pipeline>,
}

/// Cheaply cloneable: Discovery and Routing service registrations share the
/// same registry.
#[derive(Clone)]
pub struct Orchestrator {
    model_name: String,
    total_layers: u32,
    registry: std::sync::Arc<Mutex<Registry>>,
}

impl Orchestrator {
    pub fn new(model_name: impl Into<String>, total_layers: u32) -> Self {
        Self {
            model_name: model_name.into(),
            total_layers,
            registry: std::sync::Arc::new(Mutex::new(Registry {
                nodes: HashMap::new(),
                tracker: SessionTracker::new(),
                reputation: ReputationLedger::new(),
                pipelines: HashMap::new(),
            })),
        }
    }

    fn parse_node_id(raw: &str) -> Result<NodeId, Status> {
        raw.parse::<Uuid>()
            .map_err(|_| Status::invalid_argument(format!("bad node_id: {raw}")))
    }

    /// Live peers as the assembler sees them: latency, reputation, and
    /// survival stamped from the registry's trackers.
    fn live_peers(reg: &Registry, now: Instant) -> Vec<PeerInfo> {
        reg.nodes
            .iter()
            .filter(|(_, e)| now.duration_since(e.last_seen) < STALE_AFTER)
            .map(|(id, e)| PeerInfo {
                node_id: *id,
                peer_id: None,
                addrs: e.caps.addrs.clone(),
                layer_range: LayerRange::new(e.caps.layer_start, e.caps.layer_end),
                latency_ms: e.latency_ms,
                reputation: reg.reputation.score_of(id).0,
                survival: reg.tracker.survival_probability(*id, now, PLACEMENT_HORIZON) as f32,
            })
            .collect()
    }

    fn addr_of(reg: &Registry, node_id: &NodeId) -> String {
        reg.nodes
            .get(node_id)
            .and_then(|e| e.caps.addrs.first().cloned())
            .unwrap_or_default()
    }

    fn to_proto_assignment(reg: &Registry, pipeline: &Pipeline, total_layers: u32) -> PipelineAssignment {
        let slots = pipeline
            .slots
            .iter()
            .map(|s| ProtoSlot {
                node_id: s.node_id.to_string(),
                peer_addr: Self::addr_of(reg, &s.node_id),
                layer_start: s.layer_range.start,
                layer_end: s.layer_range.end,
                position: s.position as u32,
                latency_ms: reg
                    .nodes
                    .get(&s.node_id)
                    .and_then(|e| e.latency_ms)
                    .unwrap_or(0),
                standbys: s
                    .standbys
                    .iter()
                    .map(|n| Standby {
                        node_id: n.to_string(),
                        peer_addr: Self::addr_of(reg, n),
                        latency_ms: reg.nodes.get(n).and_then(|e| e.latency_ms).unwrap_or(0),
                    })
                    .collect(),
            })
            .collect();
        PipelineAssignment {
            pipeline_id: pipeline.id.to_string(),
            slots,
            total_layers,
        }
    }
}

#[tonic::async_trait]
impl DiscoveryService for Orchestrator {
    async fn announce(
        &self,
        request: Request<AnnounceRequest>,
    ) -> Result<Response<AnnounceResponse>, Status> {
        let caps = request
            .into_inner()
            .capabilities
            .ok_or_else(|| Status::invalid_argument("missing capabilities"))?;
        let node_id = Self::parse_node_id(&caps.node_id)?;
        if caps.layer_start >= caps.layer_end {
            return Err(Status::invalid_argument("empty layer range"));
        }
        if caps.model_name != self.model_name {
            return Ok(Response::new(AnnounceResponse {
                accepted: false,
                message: format!("this network serves {}", self.model_name),
            }));
        }
        let now = Instant::now();
        let mut reg = self.registry.lock().unwrap();
        let fresh = !reg.nodes.contains_key(&node_id);
        if fresh {
            reg.tracker.record_join(node_id, now);
        }
        info!(node = %node_id, layers = ?(caps.layer_start..caps.layer_end), fresh, "node announced");
        reg.nodes.insert(
            node_id,
            NodeEntry { caps, last_seen: now, latency_ms: None },
        );
        Ok(Response::new(AnnounceResponse {
            accepted: true,
            message: "welcome".into(),
        }))
    }

    async fn find_peers(
        &self,
        request: Request<FindPeersRequest>,
    ) -> Result<Response<FindPeersResponse>, Status> {
        let req = request.into_inner();
        let now = Instant::now();
        let reg = self.registry.lock().unwrap();
        let wanted = LayerRange::new(req.layer_start, req.layer_end.max(req.layer_start + 1));
        let mut peers: Vec<NodeCapabilities> = reg
            .nodes
            .values()
            .filter(|e| now.duration_since(e.last_seen) < STALE_AFTER)
            .filter(|e| {
                LayerRange::new(e.caps.layer_start, e.caps.layer_end).overlaps(&wanted)
                    && (req.model_name.is_empty() || e.caps.model_name == req.model_name)
            })
            .map(|e| e.caps.clone())
            .collect();
        if req.limit > 0 {
            peers.truncate(req.limit as usize);
        }
        Ok(Response::new(FindPeersResponse { peers }))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();
        let node_id = Self::parse_node_id(&req.node_id)?;
        let now = Instant::now();
        let mut reg = self.registry.lock().unwrap();
        let Some(entry) = reg.nodes.get_mut(&node_id) else {
            // Unknown node (registry restart, expiry): make it re-announce.
            return Ok(Response::new(HeartbeatResponse { alive: false, prefetch: vec![] }));
        };
        entry.last_seen = now;
        entry.latency_ms = Some(req.latency_ms);

        // Coverage-driven prefetch: if this node has spare budget and some
        // layer range is under-replicated, send it shopping.
        let mut prefetch = Vec::new();
        if req.spare_layers > 0 {
            let peers = Self::live_peers(&reg, now);
            let cov = LayerCoverage::from_peers(&peers, self.total_layers);
            let cand = PrefetchCandidate { node_id, max_layers: req.spare_layers };
            for d in plan_prefetch(&cov, REDUNDANCY_TARGET, std::slice::from_ref(&cand)) {
                prefetch.push(PrefetchDirective {
                    layer_start: d.layer_range.start,
                    layer_end: d.layer_range.end,
                    model_name: self.model_name.clone(),
                    quantization: String::new(),
                });
            }
        }
        Ok(Response::new(HeartbeatResponse { alive: true, prefetch }))
    }

    async fn depart(
        &self,
        request: Request<DepartRequest>,
    ) -> Result<Response<DepartResponse>, Status> {
        let req = request.into_inner();
        let node_id = Self::parse_node_id(&req.node_id)?;
        let now = Instant::now();
        let mut reg = self.registry.lock().unwrap();
        reg.tracker.record_leave(node_id, now);
        reg.reputation.record_graceful_exit(node_id);
        reg.nodes.remove(&node_id);
        info!(node = %node_id, reason = req.reason, "graceful departure");
        Ok(Response::new(DepartResponse { acknowledged: true }))
    }
}

#[tonic::async_trait]
impl RoutingService for Orchestrator {
    async fn assemble(
        &self,
        request: Request<AssembleRequest>,
    ) -> Result<Response<AssembleResponse>, Status> {
        let req = request.into_inner();
        let requester = Self::parse_node_id(&req.requester_id)?;
        let now = Instant::now();
        let mut reg = self.registry.lock().unwrap();
        let peers = Self::live_peers(&reg, now);
        let spec = PipelineSpec {
            total_layers: self.total_layers,
            max_latency_ms: if req.max_latency_ms == 0 { 10_000 } else { req.max_latency_ms },
            standbys_per_slot: REDUNDANCY_TARGET as usize,
        };
        let pipeline = assemble_pipeline(&spec, &peers, requester)
            .map_err(|e| Status::unavailable(format!("cannot assemble pipeline: {e}")))?;
        let assignment = Self::to_proto_assignment(&reg, &pipeline, self.total_layers);
        reg.pipelines.insert(pipeline.id, pipeline);
        Ok(Response::new(AssembleResponse { assignment: Some(assignment) }))
    }

    async fn check_pipeline_health(
        &self,
        request: Request<PipelineHealthRequest>,
    ) -> Result<Response<PipelineHealthResponse>, Status> {
        let req = request.into_inner();
        let node_id = Self::parse_node_id(&req.node_id)?;
        let reg = self.registry.lock().unwrap();
        let alive = reg
            .nodes
            .get(&node_id)
            .is_some_and(|e| e.last_seen.elapsed() < STALE_AFTER);
        Ok(Response::new(PipelineHealthResponse { alive, uptime_ms: 0 }))
    }

    async fn report_failover(
        &self,
        request: Request<FailoverReport>,
    ) -> Result<Response<FailoverResponse>, Status> {
        let req = request.into_inner();
        let failed = Self::parse_node_id(&req.failed_node_id)?;
        let promoted = Self::parse_node_id(&req.promoted_node_id)?;
        let pipeline_id = req
            .pipeline_id
            .parse::<Uuid>()
            .map_err(|_| Status::invalid_argument("bad pipeline_id"))?;
        let now = Instant::now();
        let mut reg = self.registry.lock().unwrap();

        // The failed node vanished mid-pipeline: hard-drop penalty, remove it
        // from the live set so it stops being placed.
        reg.reputation.record_hard_drop(failed);
        reg.tracker.record_leave(failed, now);
        reg.nodes.remove(&failed);
        warn!(node = %failed, pipeline = %pipeline_id, "hard drop reported");

        // Track the promotion so future reports see current primaries, then
        // refill the slot's standby pool with other live nodes that fully
        // cover its layer range.
        if let Some(p) = reg.pipelines.get_mut(&pipeline_id) {
            if let Some(slot) = p.slots.get_mut(req.position as usize) {
                slot.node_id = promoted;
                slot.standbys.retain(|&n| n != promoted);
            }
        }
        let Some(pipeline) = reg.pipelines.get(&pipeline_id).cloned() else {
            return Ok(Response::new(FailoverResponse { new_standbys: vec![] }));
        };
        let Some(slot) = pipeline.slots.get(req.position as usize) else {
            return Ok(Response::new(FailoverResponse { new_standbys: vec![] }));
        };
        let primaries: Vec<NodeId> = pipeline.slots.iter().map(|s| s.node_id).collect();
        let new_standbys = Self::live_peers(&reg, now)
            .into_iter()
            .filter(|p| {
                !primaries.contains(&p.node_id)
                    && p.node_id != failed
                    && p.layer_range.start <= slot.layer_range.start
                    && p.layer_range.end >= slot.layer_range.end
            })
            .map(|p| Standby {
                node_id: p.node_id.to_string(),
                peer_addr: p.addrs.first().cloned().unwrap_or_default(),
                latency_ms: p.latency_ms.unwrap_or(0),
            })
            .collect();
        Ok(Response::new(FailoverResponse { new_standbys }))
    }
}
