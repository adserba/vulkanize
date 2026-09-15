# Design — Vulkanize

## Vision

Independent GGUF inference runtime targeting AMD GPUs via Vulkan compute. Not a wrapper around llama.cpp; llama.cpp is only a reference implementation and benchmark baseline.

## Core principle

Vulkanize is not only meant to run GGUF models — it is meant to run them efficiently on AMD GPUs through Vulkan. Every architectural decision, kernel design choice, and runtime control should serve that goal. Running a model correctly is the baseline; running it fast on AMD hardware is the objective.

## Architecture overview

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  crates/cli  │────▶│  crates/runtime  │────▶│vulkan-backend     │
│  (entrypoint)│     │  (orchestration) │     │  (GPU abstractions)│
└──────────────┘     └──────────────────┘     └───────────────────┘
                              │                        ▲
                              │                        │
                              ▼                        │
                       ┌──────────────────┐            │
                       │  crates/gguf     │────────────┘
                       │  (model parsing) │   loads tensor
                       └──────────────────┘   descriptors
                              │
                              ▼
                       ┌──────────────────┐
                       │  crates/api      │
                       │  (HTTP handlers) │
                       └──────────────────┘
```

## Crate contracts

### `vulkanize-gguf`

- Parse GGUF header: magic, version, tensor count, string table
- Expose `GgufMetadata` (key-value pairs; architecture, block count, dimensions)
- Expose `TensorDescriptor` (name, n_dims, shape, dtype, offset) — **implemented**
- Expose `GgufTensorType` (36 GGML tensor data types) — **implemented**
- Expose `parse_gguf_full()` returning `(GgufHeader, GgufMetadata, GgufTensors)` — **implemented**
- Planned: memory-map the file; do NOT copy weights into CPU memory — pass file handle + offsets to runtime
- **Must not** execute any tensor computation or depend on Vulkan

### `vulkanize-vulkan-backend`

- Device/physical device selection with AMD vendor preference (vendor ID 0x1002)
- Queue family discovery: must have a compute queue
- Buffer allocation with correct memory type flags (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` for weights, `HOST_VISIBLE | HOST_COHERENT` for staging)
- Compute pipeline management: shader module loading from `.spv`, pipeline layout, descriptor sets
- Command buffer recording and submission
- **Must not** know about GGUF format or inference algorithms

### `vulkanize-runtime`

Current status: partial. The crate currently contains CPU reference helpers for embedding lookup validation. The responsibilities below are planned architecture for the full runtime.

- Owns the inference lifecycle: model load → warm-up → forward pass → sampling loop
- Manages KV cache allocation, position tracking, context window management
- Schedules compute dispatches via vulkan-backend
- Implements sampling strategies (greedy, top-k, top-p/ nucleus, temperature) on CPU
- Tokenizer interface (deferred: will integrate a UTF-8/BPE tokenizer crate)
- **Must not** contain Vulkan calls or HTTP server code

### `vulkanize-api`

Current status: stub. The API types and handlers below are planned architecture for server mode.

- OpenAI-compatible JSON types: `ChatCompletionRequest`, `ChatCompletionResponse`, `StreamEvent`
- Request handlers that call into runtime
- **Must not** contain Vulkan or GGUF code directly

### `vulkanize` (cli)

Current status: `inspect` and `vulkan-info` are implemented. `generate` and `serve` are command stubs.

- Subcommands: `inspect`, `generate`, `serve`
- Planned: delegate all inference work to runtime
- Planned: delegate API serving to api crate
- Handles signal trapping, progress reporting, stderr logging

## Planned data flow — forward pass

The full forward pass below is not implemented yet. It describes the target runtime architecture.

1. Runtime receives token IDs from tokenizer/sampler
2. Embedding lookup: dispatch compute shader with token indices → hidden state buffer
3. For each transformer block:
   - RMSNorm → QKV projection → attention (with KV cache write) → FFN → residual add
4. Final RMSNorm → LM head → logits buffer
5. Runtime reads back logits for sampling (or uses async copy to staging buffer)

## Design decisions

| Decision | Rationale |
|---|---|
| Vulkan over ROCm | Single portable API; current development and validation use Linux with RADV/Mesa; other drivers are future targets |
| SPIR-V precompiled | Avoids runtime shader compilation overhead and complexity |
| Memory-map GGUF | Zero-copy weight loading; GPU reads directly via buffer views where possible |
| No CUDA path | Scope discipline; AMD is the target, cross-platform GPU adds maintenance burden |
| Shared runtime for CLI + server | Single source of truth for inference logic |

## Deferred design decisions

- Quantization support: will we implement dequant kernels in shaders, or require FP16/FP32 GGUF initially?
- Flash attention vs standard attention: depends on shader complexity budget
- Multi-GPU / tensor parallelism: out of scope for initial phases
- Async I/O model for server: `tokio` is the likely choice
