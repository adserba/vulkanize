# Next Steps — Vulkanize

> Immediate engineering milestones: Phase 3 — runtime orchestration for embedding lookup.

## AI workflow: reconstructing project state

Before making changes, a new session should:

1. Read `docs/current-status.md` — know what is built and what is not
2. Read `docs/roadmap.md` — find the current phase and what is next
3. Read `docs/phase-3-plan.md` — detailed Phase 3 plan and results
4. Read `docs/design.md` — review crate contracts and architecture
5. Run `cargo build --release && cargo test` — confirm the baseline is green

Then work on the next roadmap item. After completing it, update `docs/roadmap.md`, `docs/current-status.md`, and `docs/phase-3-plan.md`.

**Session discipline:** One OpenCode session = one narrow milestone. Do not attempt multiple phases in a single session. Use `docs/` as the primary source of truth — avoid re-reading large `lib.rs` files unless the implementation requires exact API signatures.

## Phase 3: First kernel — embedding lookup

Phase 2.3 and Phase 3.0 are complete. The backend can allocate buffers, upload data, load shaders, create pipelines with descriptor sets and push constants, dispatch compute work with explicit memory barriers, and read back results. The smoke test proves end-to-end GPU execution. Phase 3.2 validates the embedding lookup shader with synthetic F32 data on real hardware.

### What is done (Phase 3.0 + 3.2)

- Embedding lookup shader: batched F32, 3-buffer descriptor layout, push constants, validated on RADV
- Backend: multi-buffer descriptors, push constants, explicit transfer/compute barriers
- CPU reference: `embedding_lookup_f32()`, `compare_f32()` in `runtime::embedding`
- GPU smoke test: synthetic F32 embeddings, GPU output matches CPU reference within 1e-5

### What remains

#### Next: Runtime embedding orchestration

The `runtime` crate needs to orchestrate the full embedding lookup pipeline:

1. **GGUF tensor upload integration** — use `gguf::read_tensor_bytes()` + `VulkanContext::upload_to_device_local()` to load a real model's embedding tensor to GPU
2. **Dispatch orchestration** — allocate output buffer, build descriptor set, push constants, dispatch, readback
3. **CPU vs GPU validation path** — run CPU reference on same tensor data, compare with `compare_f32()`

#### Implementation order for remaining Phase 3

1. **Runtime: GGUF tensor upload** — `ModelLoader` or equivalent that parses GGUF, finds embedding tensor, uploads to GPU
2. **Runtime: embedding dispatch** — wires up backend APIs (descriptor sets, push constants, barriers, dispatch) into a single `embedding_lookup()` call
3. **Runtime: validation** — CPU reference comparison on real GGUF tensor data
4. **Integration test** — end-to-end with small real GGUF model (F32)

#### Synthetic tests before real models

- The synthetic smoke test in `vulkan-backend/tests/embedding_lookup.rs` validates the GPU path in isolation
- Before running against a real GGUF model, add a runtime-level synthetic test that exercises the same orchestration code path with synthetic data
- Only then integrate with `gguf::read_tensor_bytes()` for real-model validation

#### Dtype strategy

- **F32 first** — simplest shader, easiest validation, establishes the pipeline (DONE)
- **F16 next** — requires `float16.spv` capability check + type conversion in shader or pre-conversion on upload
- **Quantized embeddings deferred** — Q4_0/Q8_0 dequantization adds significant shader complexity; tackle after F32/F16 kernels for the full transformer block are working

### What not to do yet

- Do not implement quantized embedding dequantization
- Do not implement the full forward pass before embedding lookup produces correct results with a real model
- Do not add async dispatch or pipeline caching
- Do not implement the tokenizer — use hardcoded token IDs for testing

## Risks and challenges

- **Vulkan shader complexity.** Compute shaders for attention and FFN are substantially more complex than CPU code. Expect iteration on workgroup sizing, shared memory usage, and memory access patterns.
- **AMD driver differences.** RADV (Mesa) and AMDVLK may behave differently. Test on the target driver early.
- **RADV-specific Vulkan pitfalls** (discovered Phase 3.2):
  - NULL buffer handles in `vkCmdPipelineBarrier` cause GPUVM faults — always pass actual buffer handles
  - `DescriptorBufferInfo` holds raw `vk::Buffer` handles — ensure `VulkanBuffer` RAII wrappers outlive descriptor set updates
- **Quantization kernels.** Dequantizing Q4_0/Q8_0 in shaders adds significant complexity. Start with FP16/FP32 models.
- **Memory pressure.** Large models need efficient weight loading. Memory-mapping + single-upload is the plan, but VRAM-constrained systems will need attention.
- **Numerical correctness.** Floating-point differences between CPU and GPU are expected. Need tolerance-based comparison, not exact equality.
- **SPIR-V toolchain.** `glslangValidator` must be available on the build machine. Consider CI implications.
- **ash API surface.** The `ash` crate is thin — all Vulkan boilerplate (descriptor sets, pipeline layouts, synchronization) is manual. This is intentional but increases implementation surface.

## What not to do yet

- Do not implement the full forward pass before embedding lookup produces correct results
- Do not implement server mode before `vulkanize generate` produces text
- Do not optimize kernels before they produce correct results
- Do not add multi-GPU or tensor parallelism before single-GPU is working
- Do not write benchmarks before numerical correctness is verified

## Future housekeeping

- **Modularize `vulkan-backend/lib.rs`** — currently a single large file. After Phase 3 is complete, split into modules: `buffer.rs`, `command.rs`, `descriptor.rs`, `pipeline.rs`, `shader.rs`, `context.rs`, `error.rs`. Do this as a standalone refactoring commit, not mixed with feature work.
