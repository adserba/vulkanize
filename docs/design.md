# Design — Vulkanize

## Vision

Independent GGUF inference runtime targeting AMD GPUs via Vulkan compute. Not a wrapper around llama.cpp; llama.cpp is only a reference implementation and benchmark baseline.

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
- Expose `ModelMetadata` (architecture, block count, dimensions, data type)
- Expose `TensorDescriptor` (name, shape, stride, offset, dtype)
- Memory-map the file; do NOT copy weights into CPU memory — pass file handle + offsets to runtime
- **Must not** execute any tensor computation or depend on Vulkan

### `vulkanize-vulkan-backend`

- Device/physical device selection with AMD vendor preference (vendor ID 0x1002)
- Queue family discovery: must have a compute queue
- Buffer allocation with correct memory type flags (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` for weights, `HOST_VISIBLE | HOST_COHERENT` for staging)
- Compute pipeline management: shader module loading from `.spv`, pipeline layout, descriptor sets
- Command buffer recording and submission
- **Must not** know about GGUF format or inference algorithms

### `vulkanize-runtime`

- Owns the inference lifecycle: model load → warm-up → forward pass → sampling loop
- Manages KV cache allocation, position tracking, context window management
- Schedules compute dispatches via vulkan-backend
- Implements sampling strategies (greedy, top-k, top-p/ nucleus, temperature) on CPU
- Tokenizer interface (deferred: will integrate a UTF-8/BPE tokenizer crate)
- **Must not** contain Vulkan calls or HTTP server code

### `vulkanize-api`

- OpenAI-compatible JSON types: `ChatCompletionRequest`, `ChatCompletionResponse`, `StreamEvent`
- Request handlers that call into runtime
- **Must not** contain Vulkan or GGUF code directly

### `vulkanize` (cli)

- Subcommands: `inspect`, `generate`, `serve`
- Delegates all inference work to runtime
- Delegates API serving to api crate
- Handles signal trapping, progress reporting, stderr logging

## Data flow — forward pass

1. Runtime receives token IDs from tokenizer/sampler
2. Embedding lookup: dispatch compute shader with token indices → hidden state buffer
3. For each transformer block:
   - RMSNorm → QKV projection → attention (with KV cache write) → FFN → residual add
4. Final RMSNorm → LM head → logits buffer
5. Runtime reads back logits for sampling (or uses async copy to staging buffer)

## Design decisions

| Decision | Rationale |
|---|---|
| Vulkan over ROCm | Single portable API; works on Linux with RADV/Mesa and Windows with AMDVLK |
| SPIR-V precompiled | Avoids runtime shader compilation overhead and complexity |
| Memory-map GGUF | Zero-copy weight loading; GPU reads directly via buffer views where possible |
| No CUDA path | Scope discipline; AMD is the target, cross-platform GPU adds maintenance burden |
| Shared runtime for CLI + server | Single source of truth for inference logic |

## Deferred design decisions

- Quantization support: will we implement dequant kernels in shaders, or require FP16/FP32 GGUF initially?
- Flash attention vs standard attention: depends on shader complexity budget
- Multi-GPU / tensor parallelism: out of scope for initial phases
- Async I/O model for server: `tokio` is the likely choice
