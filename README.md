# Hivemind

**Your GPU is idle 90% of the day. What if it was powering AI for everyone?**

Hivemind is a decentralized LLM inference network where every user is also a node. Download the CLI, it detects your hardware, downloads a model shard, and you simultaneously contribute compute to the network *and* use the network as a coding assistant. Think BitTorrent for LLM inference, with Claude Code as the interface.

## The core idea

A 72B parameter model (Qwen2.5-Coder-72B) is split across many consumer devices using pipeline parallelism. Each node holds a few transformer layers. When someone sends a prompt, it flows through a chain of nodes — each processing their layers and passing activations to the next. Users earn non-transferable utility tokens for contributing compute and spend them on inference.

## Quick start

```sh
cargo install hivemind-cli
hivemind init      # detect hardware, download your shard, join the network
hivemind chat      # start coding
```

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

The orchestrator (initially run by bootstrap nodes, eventually fully decentralized) handles pipeline assembly: given a request, it finds a chain of available nodes that collectively cover all 80 layers, optimizing for minimum end-to-end latency.

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
│   ├── hivemind-shard     # model loading (GGUF/llama.cpp) and inference
│   ├── hivemind-network   # P2P layer: Kademlia DHT, QUIC transport
│   ├── hivemind-ledger    # token accounting and reputation
│   ├── hivemind-daemon    # background node process (gRPC shard server)
│   └── hivemind-cli       # user-facing CLI with ratatui chat TUI
└── proto/
    ├── activations.proto  # tensor passing between nodes
    ├── routing.proto      # pipeline assembly
    ├── discovery.proto    # peer discovery and health
    └── tokens.proto       # token ledger operations
```

**Key dependencies:** `libp2p` (Kademlia + QUIC), `tonic`/`prost` (gRPC), `llama-cpp-rs` (inference), `ratatui` (TUI), `tokio` (async runtime).

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
- Wire up `llama-cpp-rs` in `hivemind-shard/src/inference.rs`
- Implement Kademlia peer discovery in `hivemind-network/src/peer.rs`
- Add wallet persistence in `hivemind-ledger/src/wallet.rs`
- Stream tokens in the chat TUI as they arrive

Run `cargo build` and `cargo clippy` before submitting a PR.

## License

Business Source License 1.1 — source available, **free for non-commercial use**.

Commercial use (running a paid inference service, embedding in a commercial product) requires a license. Contact the maintainers. The license converts to Apache 2.0 four years after each version's release date.
