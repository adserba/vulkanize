# Next Steps — Vulkanize

> Immediate engineering milestones: Phase 3 — first real kernel (embedding lookup).

## AI workflow: reconstructing project state

Before making changes, a new session should:

1. Read `docs/project-vision.md` — understand the goal and constraints
2. Read `docs/current-status.md` — know what is built and what is not
3. Read `docs/roadmap.md` — find the current phase and what is next
4. Read `docs/phase-3-plan.md` — detailed Phase 3 implementation plan
5. Read `docs/design.md` — review crate contracts and architecture
6. Scan `crates/*/src/lib.rs` and `crates/cli/src/main.rs` — verify code matches documentation
7. Run `cargo build --release && cargo test` — confirm the baseline is green

Then work on the next roadmap item. After completing it, update `docs/roadmap.md` and `docs/current-status.md`.

## Phase 3: First kernel — embedding lookup

Phase 2.3 is complete. The backend can allocate buffers, upload data, load shaders, create pipelines with descriptor sets, dispatch compute work, and read back results. The smoke test proves end-to-end GPU execution.

Phase 3 implements the first real inference kernel: embedding lookup.

### What embedding lookup does

Given a token ID and the model's embedding table (shape: `[vocab_size, hidden_dim]`), the kernel reads one row from the table and writes it to the output buffer. This is the first operation in the transformer forward pass.

### Implementation order

1. **Shader** — Write `embedding_lookup.comp.glsl`
2. **Backend extensions** — Multi-buffer descriptors, push constants
3. **Runtime** — GGUF tensor identification, upload, dispatch orchestration
4. **CPU reference** — Simple CPU embedding lookup for validation
5. **Integration test** — End-to-end with real GGUF model

### Dtype strategy

- **F32 first** — simplest shader, easiest validation, establishes the pipeline
- **F16 next** — requires `float16.spv` capability check + type conversion in shader or pre-conversion on upload
- **Quantized embeddings deferred** — Q4_0/Q8_0 dequantization adds significant shader complexity; tackle after F32/F16 kernels for the full transformer block are working

### Backend APIs needed

The current single-buffer descriptor convenience (`create_descriptor_set_layout_with_buffer`, `create_descriptor_set_with_buffer`) is insufficient. Phase 3 needs:

- **Multi-binding descriptor set layout** — at least 2 bindings: one for the weight buffer (read-only), one for the output buffer (read-write)
- **Uniform buffer or push constants** — to pass shape parameters (`vocab_size`, `hidden_dim`, `token_id`) to the shader
- **Push constants preferred** — simpler than uniform buffers for small per-dispatch config; no extra buffer allocation needed

### Shader binding layout (proposed)

```glsl
layout(set = 0, binding = 0, std430) readonly buffer EmbeddingTable {
    float weights[];  // [vocab_size][hidden_dim], row-major
} embedding_table;

layout(set = 0, binding = 1, std430) buffer OutputBuffer {
    float output[];   // [hidden_dim]
} output_buf;

layout(push_constant) uniform PushConstants {
    uint vocab_size;
    uint hidden_dim;
    uint token_id;
} params;
```

### CPU reference path

Implement a trivial CPU embedding lookup in `runtime` crate:

```rust
fn embedding_lookup_cpu(weights: &[f32], vocab_size: usize, hidden_dim: usize, token_id: usize) -> Vec<f32> {
    let start = token_id * hidden_dim;
    weights[start..start + hidden_dim].to_vec()
}
```

Use this to validate GPU output. Compare with tolerance (1e-5 for F32).

### GGUF tensor identification

The embedding tensor is identified by name pattern:
- `token_embd.weight` — standard llama-family naming
- Shape: `[hidden_dim, vocab_size]` (note: GGUF stores column-major, so dimensions are `[rows, cols]` = `[hidden_dim, vocab_size]`)

The GGUF parser already extracts this via `extract_model_arch()`. The runtime needs to:
1. Find the tensor by name in the parsed tensor list
2. Read its data from the GGUF file at the recorded offset
3. Upload to a device-local buffer

### Validation approach

- **GPU vs CPU**: dispatch kernel, read back result, compare with CPU reference using tolerance-based comparison (not exact equality — FP differences are expected)
- **Tolerance**: 1e-5 for F32, consider 1e-2 for F16
- **Test model**: use a small F32 GGUF for initial testing (e.g., manually converted small model)

### What not to do yet

- Do not implement quantized embedding dequantization
- Do not implement batch embedding lookup (single token first)
- Do not implement the full forward pass before embedding lookup is verified
- Do not add async dispatch or pipeline caching
- Do not implement the tokenizer — use hardcoded token IDs for testing

## Risks and challenges

- **Vulkan shader complexity.** Compute shaders for attention and FFN are substantially more complex than CPU code. Expect iteration on workgroup sizing, shared memory usage, and memory access patterns.
- **AMD driver differences.** RADV (Mesa) and AMDVLK may behave differently. Test on the target driver early.
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
