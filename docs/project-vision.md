# Project Vision — Vulkanize

## What Vulkanize is

Vulkanize is an AMD-first GGUF inference runtime built natively on Vulkan compute shaders. It reads GGUF model files, executes the full transformer forward pass on the GPU, and produces token outputs — without depending on llama.cpp, CUDA, or ROCm.

## Core principle

Vulkanize is not only meant to run GGUF models — it is meant to run them efficiently on AMD GPUs through Vulkan. Running a model correctly is the baseline; running it fast on AMD hardware is the objective.

## Key characteristics

**AMD-first.** Every optimization, kernel choice, and architectural decision targets AMD RDNA GPUs first. The Vulkan abstraction makes the code portable, but the tuning is AMD-specific.

**Vulkan-native.** Not a wrapper around another GPU runtime. Vulkan compute shaders are the sole execution path for tensor math. The `ash` crate provides thin bindings to the Vulkan C API — no higher-level abstraction that hides synchronization or memory management.

**GGUF-focused.** GGUF is the model format. The parser is a first-class component, not an afterthought. Vulkanize understands GGUF tensor layouts, quantization types, and architecture metadata.

**Performance-oriented.** The project exists to deliver fast inference on consumer AMD GPUs. Compatibility is necessary but not sufficient. Kernel fusion, memory layout, wavefront alignment, and matrix-core utilization are first-class concerns.

**Independent implementation.** Vulkanize is NOT a llama.cpp wrapper. llama.cpp serves only as a reference implementation for numerical correctness and as a benchmark baseline. Vulkanize has its own parser, its own kernels, its own runtime.

## Runtime controls (design requirement)

Future inference and server design must expose granular runtime controls comparable in spirit to mature inference runtimes such as llama.cpp. This is a design requirement, not an optional feature:

- context size
- batch size and microbatch size
- GPU device selection
- KV cache configuration and prompt cache controls
- sampling parameters (temperature, top-p, top-k, min-p, repeat penalty, presence penalty, seed, max tokens)
- memory limits
- concurrency controls
- shader/kernel selection
- pipeline cache options
- OpenAI-compatible API flags

The point is not to copy llama.cpp's flag names — it is to ensure that a power user or server operator has the same level of control over how Vulkanize behaves.

## Future performance goals

- **MTP / multi-token prediction** — models that predict multiple tokens per forward pass
- **Speculative decoding** — draft-model + verify-model pipeline for faster generation
- **KV-cache controls** — reusable KV cache across requests, paged KV cache if useful
- **Batching and microbatching** — prefill batching, continuous batching for server mode
- **AMD/RDNA optimized kernels** — wavefront-aligned workgroups, memory-coalesced access patterns, RDNA3/RDNA4-specific paths
- **Cooperative matrix / matrix-core support** — use AMD matrix cores where the Vulkan device exposes them
- **Benchmarking against llama.cpp** — automated throughput and correctness comparison on same model/hardware

## What Vulkanize is not

- Not a CUDA project
- Not a ROCm/HIP project
- Not a llama.cpp fork, wrapper, or binding
- Not a general-purpose GPU compute framework
- Not a multi-GPU tensor-parallel system (not in scope for initial phases)

## Success criteria

Vulkanize succeeds when it runs GGUF models on AMD GPUs faster than competing runtimes, with a codebase that is understandable, maintainable, and tuned specifically for AMD hardware.
