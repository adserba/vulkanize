# Vulkanize

AMD-first GGUF inference runtime built on Vulkan compute shaders. Not a llama.cpp wrapper — an independent implementation that may reference llama.cpp for benchmark comparison only.

## Principles

- **AMD GPU first** — target AMD Radeon with RADV/Mesa Vulkan drivers
- **Vulkan compute first** — all tensor execution via Vulkan compute shaders
- **GGUF model loading** — native GGUF parser; no dependency on external loaders
- **No CUDA** — NVIDIA not a target platform
- **No ROCm** — do not pull in HIP/ROCm toolchains or libraries
- **No llama.cpp runtime** — never `#include` or `cargo add` llama.cpp; reference only
- **CPU for control flow only** — parsing, tokenization, sampling, testing, debugging may run on CPU
- **Shared runtime** — CLI and server mode use the same `vulkanize-runtime` crate

## Commands

```bash
vulkanize inspect model.gguf       # show model metadata and tensor info
vulkanize generate model.gguf      # run inference (CLI mode)
vulkanize serve --host 127.0.0.1 --port 8000  # OpenAI-compatible API server
```

## Stack

- Language: Rust
- GPU API: Vulkan (compute shaders, SPIR-V)
- Model format: GGUF v3/v4
- Shaders: written in GLSL/HLSL, compiled to SPIR-V via `glslc` or similar

## Crate structure

| Crate | Purpose |
|---|---|
| `crates/gguf` | GGUF file parsing, metadata extraction, tensor descriptor loading |
| `crates/vulkan-backend` | Vulkan device/queue setup, buffer management, compute pipeline, shader loading |
| `crates/runtime` | Inference orchestration: model init, forward pass scheduling, KV cache, sampling loop |
| `crates/api` | OpenAI-compatible HTTP API types and handlers (used by server mode) |
| `crates/cli` | Binary entrypoint; subcommands delegate to runtime |

## Build and run

```bash
cargo build --release
cargo test
cargo bench
```

## License

TBD
