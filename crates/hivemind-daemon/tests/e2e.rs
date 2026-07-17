//! End-to-end tests: a real orchestrator and real shard workers on real
//! localhost gRPC, driven by the real client session.
//!
//! The reference model is deterministic, so these tests can demand the
//! strongest possible property: distributed output — even across a node
//! kill mid-generation — must be *bit-identical* to single-node inference.

use hivemind_core::LayerRange;
use hivemind_daemon::node::{spawn_orchestrator, spawn_worker, WorkerConfig, WorkerHandle};
use hivemind_network::grpc::{sample_greedy, PipelineSession};
use hivemind_shard::{engine, ForwardRequest, InferenceEngine, RefConfig, RefEngine};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

const MODEL: &str = "hivemind-ref";
const TOTAL_LAYERS: u32 = 6;
const N_GENERATE: usize = 8;

fn prompt_tokens() -> Vec<u32> {
    "fn hive(".bytes().map(u32::from).collect()
}

/// Ground truth: the whole model on one engine, no network.
fn single_node_reference(n_gen: usize) -> Vec<u32> {
    let engine = RefEngine::new(
        RefConfig::for_model(MODEL, TOTAL_LAYERS),
        LayerRange::new(0, TOTAL_LAYERS),
    )
    .unwrap();
    let pid = Uuid::new_v4();
    let mut tokens = prompt_tokens();
    let mut start = 0u32;
    let mut out = Vec::new();
    for _ in 0..n_gen {
        let logits = engine
            .forward(ForwardRequest {
                pipeline_id: pid,
                token_ids: &tokens,
                start_pos: start,
                inputs: None,
            })
            .unwrap();
        start = tokens.len() as u32;
        let tok = engine::sample_greedy(&logits.tensor).unwrap();
        tokens.push(tok);
        out.push(tok);
    }
    out
}

/// Spins up an orchestrator plus one primary and one standby worker per
/// layer-range third. Returns (orchestrator_url, node_id -> handle).
async fn spawn_network() -> (String, HashMap<Uuid, WorkerHandle>) {
    let orch = spawn_orchestrator("127.0.0.1:0", MODEL, TOTAL_LAYERS)
        .await
        .expect("orchestrator");
    let url = orch.url.clone();
    // Leak the handle so the server lives for the whole test.
    std::mem::forget(orch);

    let mut workers = HashMap::new();
    for (start, end) in [(0u32, 2u32), (2, 4), (4, 6)] {
        for _replica in 0..2 {
            let w = spawn_worker(WorkerConfig {
                model_name: MODEL.into(),
                total_layers: TOTAL_LAYERS,
                layer_range: LayerRange::new(start, end),
                orchestrator_url: url.clone(),
                bind: "127.0.0.1:0".into(),
                heartbeat_every: Duration::from_secs(5),
            })
            .await
            .expect("worker");
            workers.insert(w.node_id, w);
        }
    }
    (url, workers)
}

async fn generate(session: &mut PipelineSession, n: usize) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..n {
        let logits = session.step().await.expect("step");
        let tok = sample_greedy(&logits).expect("sample");
        session.push_token(tok);
        out.push(tok);
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn distributed_generation_matches_single_node() {
    let (url, workers) = spawn_network().await;
    let mut session = PipelineSession::connect(&url, MODEL, prompt_tokens())
        .await
        .expect("connect");
    assert_eq!(session.pipeline().slots.len(), 3, "three hops expected");
    for slot in &session.pipeline().slots {
        assert!(!slot.standbys.is_empty(), "every slot should have a warm standby");
    }

    let generated = generate(&mut session, N_GENERATE).await;
    assert_eq!(
        generated,
        single_node_reference(N_GENERATE),
        "distributed output must be bit-identical to single-node"
    );

    // Serving nodes earned tokens for their layers.
    let earned: u64 = session
        .pipeline()
        .slots
        .iter()
        .filter_map(|s| workers.get(&s.node_id))
        .map(|w| w.wallet.balance().0)
        .sum();
    assert!(earned > 0, "pipeline nodes should have earned micro-tokens");
}

#[tokio::test(flavor = "multi_thread")]
async fn node_kill_mid_generation_is_survived_bit_exactly() {
    let (url, mut workers) = spawn_network().await;
    let mut session = PipelineSession::connect(&url, MODEL, prompt_tokens())
        .await
        .expect("connect");

    let first_half = generate(&mut session, N_GENERATE / 2).await;

    // Hard-kill the middle hop's primary — no drain, no announcement,
    // exactly like a laptop lid slamming shut.
    let victim = session.pipeline().slots[1].node_id;
    workers.remove(&victim).expect("victim handle").kill();
    tokio::time::sleep(Duration::from_millis(100)).await;

    let second_half = generate(&mut session, N_GENERATE - N_GENERATE / 2).await;

    let replacement = session.pipeline().slots[1].node_id;
    assert_ne!(replacement, victim, "standby must have been promoted");

    let mut all = first_half;
    all.extend(second_half);
    assert_eq!(
        all,
        single_node_reference(N_GENERATE),
        "generation across a node kill must be bit-identical to an uninterrupted run"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn draining_node_is_routed_around() {
    let (url, mut workers) = spawn_network().await;
    let mut session = PipelineSession::connect(&url, MODEL, prompt_tokens())
        .await
        .expect("connect");

    let first = generate(&mut session, 2).await;

    // Graceful departure of the last hop's primary: drain, announce, stop.
    let victim = session.pipeline().slots[2].node_id;
    workers
        .remove(&victim)
        .expect("victim handle")
        .drain_and_depart(Duration::from_secs(2))
        .await;

    let rest = generate(&mut session, N_GENERATE - 2).await;
    assert_ne!(session.pipeline().slots[2].node_id, victim);

    let mut all = first;
    all.extend(rest);
    assert_eq!(all, single_node_reference(N_GENERATE));
}
