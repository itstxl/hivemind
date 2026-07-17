//! Node runtime: spawn orchestrators and worker shards as real gRPC servers.
//! Used by `main` for standalone processes and by integration tests to build
//! whole networks in-process.

use crate::orchestrator::Orchestrator;
use crate::scheduler::ResourceScheduler;
use crate::services::ShardService;
use hivemind_core::{LayerRange, NodeId, Result};
use hivemind_ledger::Wallet;
use hivemind_proto::activations::activation_service_server::ActivationServiceServer;
use hivemind_proto::discovery::discovery_service_client::DiscoveryServiceClient;
use hivemind_proto::discovery::discovery_service_server::DiscoveryServiceServer;
use hivemind_proto::discovery::{AnnounceRequest, DepartReason, DepartRequest, NodeCapabilities};
use hivemind_proto::routing::routing_service_server::RoutingServiceServer;
use hivemind_shard::{InferenceEngine, RefConfig, RefEngine};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tracing::{info, warn};
use uuid::Uuid;

/// A running orchestrator server.
pub struct OrchestratorHandle {
    pub url: String,
    task: JoinHandle<()>,
}

impl OrchestratorHandle {
    pub fn shutdown(self) {
        self.task.abort();
    }
}

/// Starts an orchestrator serving Discovery + Routing on `bind` (use port 0
/// for an ephemeral port). Returns once the server is accepting connections.
pub async fn spawn_orchestrator(
    bind: &str,
    model_name: &str,
    total_layers: u32,
) -> Result<OrchestratorHandle> {
    let listener = TcpListener::bind(bind)
        .await
        .map_err(|e| hivemind_core::HivemindError::Network(format!("bind {bind}: {e}")))?;
    let addr = listener.local_addr().map_err(hivemind_core::HivemindError::Io)?;
    let orch = Orchestrator::new(model_name, total_layers);

    let task = tokio::spawn(async move {
        let r = tonic::transport::Server::builder()
            .add_service(DiscoveryServiceServer::new(orch.clone()))
            .add_service(RoutingServiceServer::new(orch))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await;
        if let Err(e) = r {
            warn!(error = %e, "orchestrator server exited");
        }
    });
    let url = format!("http://{addr}");
    info!(%url, "orchestrator up");
    Ok(OrchestratorHandle { url, task })
}

/// Configuration for one worker shard node.
pub struct WorkerConfig {
    pub model_name: String,
    pub total_layers: u32,
    pub layer_range: LayerRange,
    pub orchestrator_url: String,
    /// Bind address, e.g. `127.0.0.1:0`.
    pub bind: String,
    pub heartbeat_every: Duration,
}

/// A running worker shard.
pub struct WorkerHandle {
    pub node_id: NodeId,
    pub url: String,
    pub scheduler: Arc<ResourceScheduler>,
    pub wallet: Arc<Wallet>,
    orchestrator_url: String,
    server: JoinHandle<()>,
    heartbeat: JoinHandle<()>,
}

impl WorkerHandle {
    /// Simulates a hard drop: the process vanishes mid-everything, exactly
    /// like a laptop lid slamming shut. No departure announcement.
    pub fn kill(self) {
        self.server.abort();
        self.heartbeat.abort();
    }

    /// Graceful exit: stop accepting work, drain in-flight pipelines,
    /// announce departure, then stop serving.
    pub async fn drain_and_depart(self, drain_window: Duration) {
        self.scheduler.begin_drain();
        self.scheduler.wait_drained(drain_window).await;
        if let Ok(mut c) = DiscoveryServiceClient::connect(self.orchestrator_url.clone()).await {
            let _ = c
                .depart(DepartRequest {
                    node_id: self.node_id.to_string(),
                    reason: DepartReason::DepartShutdown as i32,
                    deadline_ms: 0,
                    active_pipeline_ids: vec![],
                })
                .await;
        }
        self.server.abort();
        self.heartbeat.abort();
    }
}

/// Starts a worker serving its layer range of the reference model, announces
/// it to the orchestrator, and begins heartbeating.
pub async fn spawn_worker(cfg: WorkerConfig) -> Result<WorkerHandle> {
    let engine: Arc<dyn InferenceEngine> = Arc::new(RefEngine::new(
        RefConfig::for_model(&cfg.model_name, cfg.total_layers),
        cfg.layer_range,
    )?);
    spawn_worker_with_engine(cfg, engine).await
}

/// Same as [`spawn_worker`] but with a caller-provided engine (tests use
/// this to inject faulty or instrumented shards).
pub async fn spawn_worker_with_engine(
    cfg: WorkerConfig,
    engine: Arc<dyn InferenceEngine>,
) -> Result<WorkerHandle> {
    let node_id = Uuid::new_v4();
    let scheduler = Arc::new(ResourceScheduler::new(8));
    let wallet = Arc::new(Wallet::default());

    let listener = TcpListener::bind(&cfg.bind)
        .await
        .map_err(|e| hivemind_core::HivemindError::Network(format!("bind {}: {e}", cfg.bind)))?;
    let addr = listener.local_addr().map_err(hivemind_core::HivemindError::Io)?;
    let url = format!("http://{addr}");

    let service = ShardService::new(engine, Arc::clone(&scheduler), Arc::clone(&wallet));
    let server = tokio::spawn(async move {
        let r = tonic::transport::Server::builder()
            .add_service(ActivationServiceServer::new(service))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await;
        if let Err(e) = r {
            warn!(error = %e, "shard server exited");
        }
    });

    // Announce to the orchestrator.
    let mut disc = DiscoveryServiceClient::connect(cfg.orchestrator_url.clone())
        .await
        .map_err(|e| hivemind_core::HivemindError::Network(format!("orchestrator connect: {e}")))?;
    let caps = NodeCapabilities {
        node_id: node_id.to_string(),
        peer_id: String::new(),
        addrs: vec![url.clone()],
        layer_start: cfg.layer_range.start,
        layer_end: cfg.layer_range.end,
        model_name: cfg.model_name.clone(),
        quantization: String::new(),
        vram_mb: 0,
        ram_mb: 0,
        compute_major: 0,
        compute_minor: 0,
        max_concurrent: 8,
        bandwidth_mbps: 0,
    };
    let resp = disc
        .announce(AnnounceRequest { capabilities: Some(caps) })
        .await
        .map_err(|e| hivemind_core::HivemindError::Network(format!("announce: {e}")))?
        .into_inner();
    if !resp.accepted {
        server.abort();
        return Err(hivemind_core::HivemindError::Network(format!(
            "orchestrator rejected node: {}",
            resp.message
        )));
    }
    info!(node = %node_id, %url, layers = %cfg.layer_range, "worker announced");

    // Heartbeat loop. Prefetch directives are logged for now; acting on them
    // (loading extra layers and re-announcing) is the next increment.
    let hb_id = node_id;
    let hb_url = cfg.orchestrator_url.clone();
    let every = cfg.heartbeat_every;
    let heartbeat = tokio::spawn(async move {
        let mut interval = tokio::time::interval(every);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            let Ok(mut c) = DiscoveryServiceClient::connect(hb_url.clone()).await else {
                continue;
            };
            let req = hivemind_proto::discovery::HeartbeatRequest {
                node_id: hb_id.to_string(),
                latency_ms: 1,
                spare_layers: 0,
            };
            match c.heartbeat(req).await {
                Ok(resp) => {
                    for p in resp.into_inner().prefetch {
                        info!(node = %hb_id, layers = ?(p.layer_start..p.layer_end), "prefetch directive received");
                    }
                }
                Err(e) => warn!(node = %hb_id, error = %e, "heartbeat failed"),
            }
        }
    });

    Ok(WorkerHandle {
        node_id,
        url,
        scheduler,
        wallet,
        orchestrator_url: cfg.orchestrator_url,
        server,
        heartbeat,
    })
}
