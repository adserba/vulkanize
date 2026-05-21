# Roadmap — Vulkanize

## Phase 0: Foundation (current)

- [x] Repository structure, workspace Cargo.toml
- [x] Crate stubs with correct dependency graph
- [x] Design documentation
- [ ] `.gitignore` and initial commit conventions
- [ ] Shader build pipeline skeleton

## Phase 1: GGUF parser

- [x] Parse GGUF header (magic, version, counts)
- [x] Read string table, type definitions
- [ ] Enumerate tensor descriptors with correct offsets
- [ ] Extract architecture metadata (block count, dimensions, types)
- [ ] Integration test against a small GGUF file (e.g., `Qwen2.5-0.5B-Q4_0`)

## Phase 2: Vulkan backend skeleton

- [ ] Instance creation with validation layers (debug) / without (release)
- [ ] Physical device selection — prefer AMD vendor ID 0x1002
- [ ] Device creation with compute queue
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

## Future server and inference controls

Vulkanize should eventually expose granular runtime controls similar in spirit to mature inference servers, but designed around AMD/Vulkan rather than CUDA/ROCm.

Planned runtime flags:

- model path
- host / port
- context size
- batch size
- microbatch size
- AMD GPU device selection
- KV cache configuration
- sampling settings:
  - temperature
  - top-p
  - top-k
  - min-p
  - repeat penalty
  - presence penalty
  - seed
  - max output tokens
- memory limits
- shader/kernel selection
- pipeline cache options
- prompt cache controls
- reusable KV cache controls
- server concurrency limits
- OpenAI-compatible API mode

Planned performance features:

- speculative decoding
- MTP / multi-token prediction support
- prompt cache
- reusable KV cache
- paged KV cache if useful
- AMD-specific shader variants
- RDNA3/RDNA4-specific kernel paths
- cooperative matrix / matrix core support where available
