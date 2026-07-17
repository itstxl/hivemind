# hivemind

### Premium AI without data centers. Powered by everyone.

The world doesn't need more data centers. We need to stop pretending it does.

Right now, billions of GPUs and CPUs sit mostly idle — in laptops, gaming rigs, and workstations — while corporations are drilling wells, diverting rivers, and burning coal to build the infrastructure to run AI for us. They're not doing it out of generosity. They do it to sit between you and the model, to harvest your data, to charge you a subscription, and to make themselves impossible to live without.

Here's the thing: we already have enough compute. Collectively, we always have. It's just fragmented, uncoordinated, and going to waste. The device in your pocket, the gaming rig under your desk, the MacBook on your kitchen table — together they dwarf what any one company could build. We just never had a way to use it together.

Hivemind is that way.

Every person who installs Hivemind contributes a tiny slice of their device to a shared inference network. A 72B parameter model — the same quality you'd pay $20/month for elsewhere — is split across thousands of ordinary machines. Your laptop runs a few transformer layers. Your neighbour's GPU runs a few more. When you send a prompt, it flows across that chain in milliseconds and comes back as a response. No data center. No corporation in the middle. No water pulled from a drought-stricken aquifer to cool a server rack you'll never see.

You give a little compute. You get a premium AI coding assistant for free. Everyone does. That's the whole deal.

This isn't a cost-saving hack. It's a different answer to the question of who gets to run AI, and who gets to benefit from it.

## Quick start

There is no public network yet, but the whole system runs today on your own
machine — a real orchestrator, real shard nodes, and real distributed
inference over gRPC:

```sh
git clone <this repo> && cd hivemind && cargo build

# 1. Start the orchestrator
HIVEMIND_MODE=orchestrator HIVEMIND_BIND=127.0.0.1:7001 \
  HIVEMIND_TOTAL_LAYERS=6 target/debug/hivemind-daemon

# 2. Start a worker per layer range (repeat with 0..2, 2..4, 4..6)
HIVEMIND_MODE=worker HIVEMIND_LAYERS=0..2 HIVEMIND_TOTAL_LAYERS=6 \
  HIVEMIND_ORCHESTRATOR_URL=http://127.0.0.1:7001 \
  HIVEMIND_BIND=127.0.0.1:0 target/debug/hivemind-daemon

# 3. Generate across the pipeline
HIVEMIND_ORCHESTRATOR_URL=http://127.0.0.1:7001 \
  target/debug/hivemind complete "fn main() {"
```

Today this serves a small built-in reference model (untrained, so the output
is deterministic noise) — its purpose is proving the distributed machinery,
not writing your code yet. Production models plug in behind the same
`InferenceEngine` trait; see [Status](#status).

## How it works

```
Prompt: "write a binary search in Rust"

  User ──► Node A (layers 0–8) ──► Node B (layers 8–24) ──► Node C (layers 24–80)
              RTX 3060                 M2 MacBook                  RTX 4090
              12 GB VRAM               16 GB RAM                   24 GB VRAM

  Each node runs its transformer layers on the incoming activations
  and passes the result to the next node over an encrypted QUIC stream.

  Last node samples a token ──► streams it back to the user ──► repeat.
```

The orchestrator (initially run by bootstrap nodes, eventually fully decentralized) handles pipeline assembly: given a request, it finds a chain of available nodes that collectively cover all 80 layers, optimizing for minimum end-to-end latency. (The current implementation routes hops through the requesting client, Petals-style; direct node-to-node forwarding is planned.)

## Surviving churn

The network is made of laptops that close and desktops that reboot, so a node
vanishing must never kill a session. Four mechanisms make churn survivable —
and they work today: in the end-to-end tests, a node hard-killed
mid-generation changes nothing but a moment's latency, and the output stays
**bit-identical** to an uninterrupted run.

- **Warm standbys.** Every pipeline slot is assembled with backup nodes that
  already hold its weights. When a hop stops answering, the standby is
  promoted locally — no orchestrator round-trip — and the orchestrator
  refills the pool afterwards.
- **Activation replay.** A promoted standby has cold attention state. The
  client records every boundary activation it has sent and replays the hop's
  history in one prefill, rebuilding the standby's KV cache exactly — no
  full-pipeline rerun.
- **Graceful drain.** Most departures are lid-closes, not crashes. On
  SIGTERM the daemon stops accepting work, finishes in-flight sequences, and
  announces its departure so standbys are promoted before anything fails.
  Announced exits are free; vanishing mid-pipeline costs heavy reputation.
- **Survival-aware placement.** The orchestrator tracks each node's session
  history and estimates how likely it is to still be online in twenty
  minutes. A stable desktop beats a marginally faster laptop that joined
  three minutes ago, and idle nodes are directed to pre-load
  under-replicated layers before failures happen.

## Why a network can do what no individual can

The best consumer GPU money can buy — an RTX 4090 with 24 GB of VRAM — can run a 70B model at Q4 quantization, and it's slow. A 405B model is simply impossible. You'd need roughly $15,000 worth of GPUs, or a cloud instance at $30+ per hour. The best open source models are, in practice, inaccessible to individuals.

A Hivemind network with 1,000 active nodes each contributing 8–12 GB of VRAM has **8–12 terabytes of aggregate VRAM.** That's enough to run Llama 405B unquantized. Multiple models simultaneously. And eventually, future open source models with 1T+ parameters that no single organisation outside Google or Microsoft could afford to serve.

This is the thing no centralized provider can replicate. A company has to buy all that hardware, power it, cool it, and depreciate it. Hivemind gets it for free — from people who already own it and weren't using it anyway.

The scaling argument also inverts in a way that matters: **the larger the model, the more valuable the network becomes.** A 7B model? Anyone can run that on a laptop — Hivemind offers little advantage. A 405B model? Nobody can run that alone. Hivemind is the only way an individual accesses it without paying per-token to a corporation. A future 1T parameter model? Hivemind may be the only way anyone runs it outside of three companies on earth.

This isn't a cheaper alternative to OpenAI. It's access to something OpenAI itself can't offer you: a frontier model running entirely on hardware you and your peers already own, with no company in the loop.

## Hardware requirements

| Tier | Hardware | Role |
|------|----------|------|
| Minimum | 8 GB RAM, any CPU | CPU-only node, serves ~2 layers |
| Recommended | Any NVIDIA GPU, 6 GB+ VRAM | GPU node, serves 8–16 layers |
| Ideal | RTX 3090 / 4090, 24 GB VRAM | Full quarter of the model |

## Architecture

```
hivemind/
├── crates/
│   ├── hivemind-core      # shared types, config, error handling
│   ├── hivemind-proto     # tonic-generated gRPC clients/servers
│   ├── hivemind-shard     # InferenceEngine trait, reference transformer,
│   │                      #   activation checkpoints (GGUF/llama.cpp planned)
│   ├── hivemind-network   # pipeline assembly, failover, survival model,
│   │                      #   coverage planner, client session driver
│   ├── hivemind-ledger    # token accounting and reputation
│   ├── hivemind-daemon    # shard server + orchestrator (one binary, two roles)
│   └── hivemind-cli       # user-facing CLI with ratatui chat TUI
└── proto/
    ├── activations.proto  # tensor passing + KV replay between nodes
    ├── routing.proto      # pipeline assembly and failover reporting
    ├── discovery.proto    # announce, heartbeat, graceful departure, prefetch
    └── tokens.proto       # token ledger operations
```

**Key dependencies:** `tonic`/`prost` (gRPC), `tokio` (async runtime), `libp2p` (Kademlia + QUIC, planned), `llama-cpp-rs` (inference, planned), `ratatui` (TUI).

## Status

Early development — but the distributed core is real and tested. What works
today, verified by end-to-end tests against live localhost servers:

- Orchestrator + shard workers as real gRPC processes; announce, heartbeat,
  pipeline assembly, and generation across multiple nodes.
- Distributed output bit-identical to single-node inference.
- Hard node kills and graceful drains mid-generation survived bit-exactly
  via standby promotion and activation replay.
- Per-layer token earning on serving nodes.

What is not real yet: the model is an untrained reference transformer
(llama.cpp/GGUF backend is the next big step), discovery is registry-based
rather than DHT, transport is HTTP/2 rather than QUIC, prefetch directives
are logged but not yet acted on, wallets are in-memory, and none of the
verification/privacy work (untrusted nodes returning garbage, activations
being readable in transit) has begun. Treat it as a working prototype of the
network layer, not something to expose to the internet.

## Token economics

- **Earn:** Every transformer layer you process earns 1 micro-token per layer per sequence token.
- **Spend:** Inference costs 2 micro-tokens per layer per sequence token (network takes 50%).
- **Non-transferable:** Tokens are utility tokens only — no trading, no speculation.
- **Reputation:** Nodes with higher uptime and lower latency get priority in pipeline assembly.

## Commands

```sh
hivemind init                    # one-time setup
hivemind chat                    # interactive coding assistant
hivemind complete "explain X"    # single-shot, pipeable
hivemind status                  # node stats and token balance
hivemind config show             # view full config
hivemind config set hardware.gpu_allocation 0.6
```

## Contributing

Hivemind is in early development. The network protocol and token economics are not final.

**Good first issues:**
- Implement a llama.cpp/GGUF backend for the `InferenceEngine` trait in `hivemind-shard/src/engine.rs`
- Act on prefetch directives: load the extra layers and re-announce (`hivemind-daemon/src/node.rs`)
- Implement Kademlia peer discovery in `hivemind-network/src/peer.rs`
- Add wallet persistence in `hivemind-ledger/src/wallet.rs`
- Wire the chat TUI to `PipelineSession` and stream tokens as they arrive

Run `cargo build` and `cargo clippy` before submitting a PR.

## License

Business Source License 1.1 — source available, **free for non-commercial use**.

Commercial use (running a paid inference service, embedding in a commercial product) requires a license. Contact the maintainers. The license converts to Apache 2.0 four years after each version's release date.
