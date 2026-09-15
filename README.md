# Vulkanize

Vulkanize is an early-stage AMD-first GGUF inference runtime built in Rust on Vulkan compute shaders. It is an independent implementation, not a llama.cpp wrapper. llama.cpp may be used later only as a correctness and benchmark reference.

The project currently implements GGUF v3 parsing, Vulkan compute infrastructure, and GPU smoke tests for basic compute dispatch and synthetic F32 embedding lookup. It does not yet generate text or expose a working HTTP server.

## Implemented Now

- Native GGUF v3 parser for headers, metadata, tensor descriptors, architecture metadata extraction, tensor lookup helpers, and raw tensor byte reads.
- `vulkanize inspect model.gguf` prints model metadata and tensor information.
- Vulkan backend using `ash`: instance/device/queue setup, AMD-preferred device selection, command pool, buffers, staging upload, readback, shader module loading, descriptor sets, push constants, compute pipelines, dispatch, fences, and explicit transfer/compute barriers.
- `vulkanize vulkan-info` prints selected GPU, queue family, command pool, device extensions, and discovered physical devices.
- GLSL compute shaders for no-op dispatch and synthetic F32 embedding lookup.
- CPU reference helpers for F32 embedding lookup, F16 byte conversion, and tolerance-based float comparison.
- Unit tests for the GGUF parser, Vulkan backend support types, and runtime reference helpers.
- Ignored GPU integration tests for no-op compute dispatch and synthetic F32 embedding lookup.

## Current Limitations

- `vulkanize generate` is a stub and does not run inference.
- `vulkanize serve` is a stub and does not start an HTTP server.
- Full transformer forward pass, KV cache management, tokenizer integration, sampling, and text generation are planned but not implemented.
- OpenAI-compatible API types, handlers, streaming, and server lifecycle are planned but not implemented.
- Real-model GPU embedding orchestration is not implemented yet; the current embedding GPU test uses synthetic F32 data.
- Benchmarks are planned but no benchmark suite exists yet.

## Commands

Implemented:

```bash
cargo run --release -- inspect model.gguf
cargo run --release -- vulkan-info
```

Currently stubbed:

```bash
cargo run --release -- generate model.gguf "prompt"
cargo run --release -- serve --host 127.0.0.1 --port 8000
```

## Roadmap

The near-term roadmap is to wire the existing GGUF tensor helpers, Vulkan buffers, and embedding shader into the runtime crate so a real GGUF model's F32 embedding tensor can be uploaded, dispatched, read back, and compared against the CPU reference.

Later phases cover transformer block kernels, full forward pass, sampling/text generation, server mode, quantized kernels, and benchmarking. See `docs/roadmap.md` and `docs/current-status.md` for the detailed phase breakdown.

## Repository Structure

| Path | Current role |
|---|---|
| `crates/gguf` | Functional GGUF v3 parser and tensor metadata helpers |
| `crates/vulkan-backend` | Functional Vulkan compute backend primitives |
| `crates/runtime` | Partial runtime crate; currently CPU reference embedding helpers |
| `crates/api` | Stub crate for planned OpenAI-compatible API layer |
| `crates/cli` | CLI entrypoint; `inspect` and `vulkan-info` work, `generate` and `serve` are stubs |
| `shaders` | Tracked GLSL shader sources; generated `.spv` files are ignored |
| `docs` | Design notes, current status, and roadmap |
| `configs` | Future deployment examples; server mode is not usable yet |

## Prerequisites

- Rust stable toolchain with Cargo.
- Vulkan loader and a Vulkan-capable GPU/driver.
- Development and GPU validation are currently performed primarily on Linux with AMD RADV/Mesa.
- `glslangValidator` is required to compile GLSL shaders into `.spv` files for GPU integration tests.
- On Debian/Ubuntu, `glslangValidator` is usually provided by `glslang-tools`; Vulkan runtime tooling is commonly provided by packages such as `vulkan-tools` and the appropriate Mesa/driver packages.

## Build

```bash
cargo build --release
```

This builds the current crates; it does not build a complete inference engine.

## Shader Compilation

Generated SPIR-V files are intentionally ignored by Git. Compile the tracked shader sources before running ignored GPU integration tests:

```bash
./scripts/compile-shaders.sh
```

This writes:

- `shaders/no_op.spv`
- `shaders/embedding_lookup.spv`

## Tests

Run CPU/unit tests:

```bash
cargo test
```

Run ignored GPU integration tests after compiling shaders:

```bash
cargo test --test smoke_no_op -- --ignored
cargo test --test embedding_lookup -- --ignored
```

The ignored GPU tests require a working Vulkan driver and compatible GPU. They are not expected to run on every development machine or CI worker by default.

## License

Vulkanize is dual-licensed under the MIT License or Apache License 2.0. See
[`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE).
