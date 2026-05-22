# Roadmap — Vulkanize

## Phase 0: Foundation

- [x] Repository structure, workspace Cargo.toml
- [x] Crate stubs with correct dependency graph
- [x] Design documentation
- [x] `.gitignore` and initial commit conventions
- [ ] Shader build pipeline skeleton

## Phase 1: GGUF parser

- [x] Parse GGUF header (magic, version, counts)
- [x] Read string table, type definitions
- [x] Enumerate tensor descriptors with correct offsets
- [x] `GgufTensorType` enum (36 GGML types)
- [x] `TensorDescriptor` struct (name, n_dims, shape, dtype, offset)
- [x] `parse_gguf_full()` returning header + metadata + tensors
- [x] `vulkanize inspect` prints tensor summary
- [x] 106 unit tests with synthetic GGUF byte arrays
- [x] Extract architecture metadata (block count, dimensions, types)
- [ ] Integration test against a small GGUF file (e.g., `Qwen2.5-0.5B-Q4_0`)

## Phase 2: Vulkan backend skeleton

### Phase 2.1: Context skeleton (done)

- [x] Vulkan loader/entry initialization
- [x] Instance creation with validation layers (debug) / without (release)
- [x] Physical device enumeration and info collection
- [x] `PhysicalDeviceInfo` and `QueueFamilyInfo` types
- [x] Physical device selection — prefer AMD vendor ID 0x1002
- [x] Logical device creation with compute queue
- [x] `VulkanContext` owning instance + device + queues
- [x] `vulkanize vulkan-info` CLI command
- [x] 22 unit tests for formatting helpers and error types

### Phase 2.2: Buffers and pipelines (next)

- [ ] Buffer allocation helpers (device-local, host-visible staging)
- [ ] Compute pipeline from `.spv` module
- [ ] Command buffer record → submit → wait cycle
- [ ] Smoke test: dispatch a no-op kernel and verify completion

## Phase 3: First kernel — embedding lookup

- [ ] Write `embedding_lookup.comp.glsl` shader
- [ ] Compile to SPIR-V, embed or ship as `.spv`
- [ ] Runtime: memory-map GGUF weights, create device buffers
- [ ] Dispatch embedding lookup for a single token
- [ ] Read back and verify against CPU reference

## Phase 4: Transformer block

- [ ] RMSNorm kernel
- [ ] QKV projection + RoPE rotation kernel
- [ ] Attention with KV cache write
- [ ] FFN (SiLU-gated) kernel
- [ ] Single-block forward pass verification

## Phase 5: Full forward pass

- [ ] Chain all blocks through runtime scheduler
- [ ] LM head kernel + logits readback
- [ ] End-to-end: single token → logits, verify against llama.cpp output

## Phase 6: Sampling and generation loop

- [ ] CPU sampler: greedy, temperature, top-k, top-p
- [ ] KV cache growth across tokens
- [ ] `vulkanize generate` command produces coherent text
- [ ] Benchmark vs llama.cpp on same model/hardware

## Phase 7: Server mode

- [ ] `tokio` + `axum` / `warp` HTTP server
- [ ] OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
- [ ] Streaming SSE responses
- [ ] Model loading/unloading lifecycle
- [ ] `systemd` service unit example

## Phase 8: Performance

- [ ] Quantized kernel support (Q4_0, Q8_0 dequant in shaders)
- [ ] Batch inference for prefill
- [ ] Kernel fusion opportunities (RMSNorm + projection, etc.)
- [ ] Vulkan timeline semaphores for multi-batch pipelining
- [ ] Benchmark suite with automated llama.cpp comparison

## Future design requirements

The items below are future design requirements, not current implementation tasks. They define the capabilities Vulkanize must support as the project matures.

### Granular runtime controls

Future inference and server design must expose runtime controls comparable in spirit to mature inference runtimes such as llama.cpp, but designed around AMD/Vulkan rather than CUDA/ROCm.

Required controls:

- context size
- batch size
- microbatch size
- GPU device selection
- KV cache configuration
- prompt cache controls
- sampling parameters (temperature, top-p, top-k, min-p, repeat penalty, presence penalty, seed, max tokens)
- memory limits
- concurrency controls
- shader/kernel selection
- pipeline cache options
- OpenAI-compatible API flags

### Performance roadmap

Future performance work items:

- MTP / multi-token prediction
- speculative decoding
- reusable KV cache
- paged KV cache if useful
- AMD/RDNA-specific optimized kernels
- cooperative matrix / matrix-core paths where available
- benchmarking against llama.cpp
