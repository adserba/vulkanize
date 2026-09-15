# Project Vision — Vulkanize

## What Vulkanize is intended to become

Vulkanize is an AMD-first GGUF inference runtime being built natively on Vulkan compute shaders. The target is to read GGUF model files, execute the full transformer forward pass on the GPU, and produce token outputs without depending on llama.cpp, CUDA, or ROCm.

Current implementation status: Vulkanize can parse GGUF v3 model structure, inspect model metadata, create and use Vulkan compute resources, and validate synthetic GPU embedding lookup on real hardware. Full model inference, text generation, server mode, KV cache management, tokenizer integration, and sampling are planned but not implemented yet.

## Core principle

Vulkanize is not only meant to run GGUF models — it is meant to run them efficiently on AMD GPUs through Vulkan. Running a model correctly is the baseline; running it fast on AMD hardware is the objective.

## Key characteristics

**AMD-first.** Every optimization, kernel choice, and architectural decision targets AMD RDNA GPUs first. The Vulkan abstraction makes the code portable, but the tuning is AMD-specific.

**Vulkan-native.** Not a wrapper around another GPU runtime. Vulkan compute shaders are the sole execution path for tensor math. The `ash` crate provides thin bindings to the Vulkan C API — no higher-level abstraction that hides synchronization or memory management.

**GGUF-focused.** GGUF is the model format. The parser is a first-class component, not an afterthought. Vulkanize currently supports GGUF v3 parsing and exposes tensor layouts, quantization type identifiers, and architecture metadata needed by later runtime phases.

**Performance-oriented.** The project exists to deliver fast inference on consumer AMD GPUs. Compatibility is necessary but not sufficient. Kernel fusion, memory layout, wavefront alignment, and matrix-core utilization are first-class design concerns for later inference kernels.

**Independent implementation.** Vulkanize is NOT a llama.cpp wrapper. llama.cpp serves only as a reference implementation for numerical correctness and as a future benchmark baseline. Vulkanize has its own parser and Vulkan kernels; the full runtime is still being built.

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
