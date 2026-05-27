# Hivemind

### Premium AI without data centers. Powered by everyone.

The world doesn't need more data centers. We need to stop pretending it does.

Right now, billions of GPUs and CPUs sit mostly idle — in laptops, gaming rigs, and workstations — while corporations are drilling wells, diverting rivers, and burning coal to build the infrastructure to run AI for us. They're not doing it out of generosity. They do it to sit between you and the model, to harvest your data, to charge you a subscription, and to make themselves impossible to live without.

Here's the thing: we already have enough compute. Collectively, we always have. It's just fragmented, uncoordinated, and going to waste. The device in your pocket, the gaming rig under your desk, the MacBook on your kitchen table — together they dwarf what any one company could build. We just never had a way to use it together.

Hivemind is that way.

Every person who installs Hivemind contributes a tiny slice of their device to a shared inference network. A 72B parameter model — the same quality you'd pay $20/month for elsewhere — is split across thousands of ordinary machines. Your laptop runs a few transformer layers. Your neighbour's GPU runs a few more. When you send a prompt, it flows across that chain in milliseconds and comes back as a response. No data center. No corporation in the middle. No water pulled from a drought-stricken aquifer to cool a server rack you'll never see.

You give a little compute. You get a premium AI coding assistant for free. Everyone does. That's the whole deal.

This isn't a cost-saving hack. It's a different answer to the question of who gets to run AI, and who gets to benefit from it.

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
