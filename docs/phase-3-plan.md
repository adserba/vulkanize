# Phase 3 Plan — Embedding Lookup Kernel

> Condensed implementation plan for the first real compute kernel.
> Reference: `docs/current-status.md`, `docs/next-steps.md`, `docs/design.md`

## Goal

Dispatch a compute shader that performs embedding table lookup for a single token, read back the result, and verify it matches a CPU reference computation.

## Prerequisites (Phase 2.3 done)

- Vulkan context, device, queue, command pool
- Buffer allocation (device-local + host-visible)
- Staging upload (`upload_to_device_local`)
- Readback (`readback_buffer_data`)
- Shader module loading from `.spv`
- Compute pipeline with descriptor set layouts
- Descriptor set layout/pool/set management
- Command buffer recording + dispatch
- Fence-based synchronization
- Smoke test proving end-to-end GPU execution

## Commit breakdown

### Commit 1: embedding_lookup shader (F32)

**Files:** `shaders/embedding_lookup.comp.glsl`, `shaders/embedding_lookup.spv`

```glsl
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer EmbeddingTable {
    float weights[];
} embedding_table;

layout(set = 0, binding = 1, std430) buffer OutputBuffer {
    float output[];
} output_buf;

layout(push_constant) uniform PushConstants {
    uint hidden_dim;
    uint token_id;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid < params.hidden_dim) {
        uint idx = params.token_id * params.hidden_dim + gid;
        output_buf.output[gid] = embedding_table.weights[idx];
    }
}
```

**Notes:**
- 64-wide workgroup for AMD wavefront alignment
- GGUF stores column-major: `[hidden_dim, vocab_size]` → row `token_id` starts at `token_id * hidden_dim`
- No barrier needed: each thread writes to a unique output index
- Compile with `glslangValidator -V embedding_lookup.comp.glsl -o embedding_lookup.spv`

### Commit 2: backend — push constant support

**Files:** `crates/vulkan-backend/src/lib.rs`

Add to `ComputePipeline`:
- `new_with_push_constants(device, shader, entry_point, set_layouts, push_constant_range)` — creates pipeline layout with push constant range
- Extend `VulkanContext::create_compute_pipeline_with_push_constants(...)`

Add to `CommandBuffer`:
- `push_constants(layout, set, offset, data: &[u8])` — records `vkCmdPushConstants`

Add to `VulkanError`:
- `PushConstantValidation(String)` — if needed

**API design:**
```rust
pub fn new_with_push_constants(
    device: &ash::Device,
    shader_module: &ShaderModule,
    entry_point: &str,
    set_layouts: &[vk::DescriptorSetLayout],
    push_constant_range: vk::PushConstantRange,
) -> Result<Self, VulkanError>
```

### Commit 3: backend — multi-buffer descriptor sets

**Files:** `crates/vulkan-backend/src/lib.rs`

Add to `VulkanContext`:
- `create_descriptor_set_layout_with_buffers(bindings: &[(binding: u32, desc_type: vk::DescriptorType, flags: vk::ShaderStageFlags)])` — generic multi-binding layout
- `create_descriptor_set_with_buffers(layout, writes: &[DescriptorWrite])` — generic multi-buffer descriptor set

Or, more ergonomically:
- `create_descriptor_set_layout_multiple(bindings: &[DescriptorSetLayoutBinding])` — takes pre-built `vk::DescriptorSetLayoutBinding` array
- `update_descriptor_set(descriptor_set, writes: &[WriteDescriptorSet])` — convenience wrapper

**Minimum needed for embedding lookup:**
- Layout with 2 bindings: binding 0 (STORAGE_BUFFER, readonly), binding 1 (STORAGE_BUFFER, readwrite)
- Descriptor set updated with 2 buffer infos

**Note:** `vk::DescriptorType::UNIFORM_TEXEL_BUFFER` and `vk::DescriptorType::STORAGE_TEXEL_BUFFER` are NOT needed here — `STORAGE_BUFFER` with `readonly` in GLSL is sufficient for the weight buffer.

### Commit 4: runtime — GGUF tensor upload

**Files:** `crates/runtime/src/lib.rs`

```rust
pub struct ModelLoader {
    ctx: VulkanContext,
    gguf_header: GgufHeader,
    gguf_metadata: GgufMetadata,
    gguf_tensors: Vec<TensorDescriptor>,
    file: std::fs::File,
}

impl ModelLoader {
    pub fn new(ctx: VulkanContext, path: &std::path::Path) -> Result<Self, Error> {
        // Parse GGUF, open file
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.gguf_tensors.iter().find(|t| t.name == name)
    }

    pub fn upload_tensor(&self, tensor: &TensorDescriptor) -> Result<VulkanBuffer, Error> {
        // Read tensor data from file at offset
        // Create device-local buffer
        // Upload via ctx.upload_to_device_local
    }
}
```

**Notes:**
- Read tensor data with `std::io::Seek` + `std::io::Read` to avoid loading entire file
- For F32 tensors: element size = 4 bytes
- For F16 tensors: element size = 2 bytes (will need `half` crate for f16→f32 conversion, or handle in shader)
- GGUF tensor data is stored in a specific element order — verify against `docs/gguf-format-notes.md`

### Commit 5: runtime — embedding dispatch

**Files:** `crates/runtime/src/lib.rs`

```rust
pub fn embedding_lookup(
    ctx: &VulkanContext,
    weight_buffer: &VulkanBuffer,
    token_id: u32,
    hidden_dim: u32,
) -> Result<Vec<f32>, Error> {
    // Load shader
    // Create descriptor layout (2 bindings)
    // Create pipeline with push constants
    // Create output buffer (hidden_dim * 4 bytes)
    // Create descriptor set with weight + output bindings
    // Record: begin → bind pipeline → push constants → bind descriptors → dispatch → end → submit
    // Read back output buffer
    // Convert bytes to f32 slice
}
```

**Dispatch sizing:**
- `group_count_x = (hidden_dim + 63) / 64`
- `group_count_y = 1`
- `group_count_z = 1`

**Push constant layout:**
```rust
#[repr(C)]
struct EmbeddingPushConstants {
    hidden_dim: u32,
    token_id: u32,
}
```

### Commit 6: runtime — CPU reference + validation

**Files:** `crates/runtime/src/lib.rs`

```rust
pub fn embedding_lookup_cpu(weights: &[f32], hidden_dim: usize, token_id: usize) -> Vec<f32> {
    let start = token_id * hidden_dim;
    weights[start..start + hidden_dim].to_vec()
}

pub fn assert_float_slices_equal(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(diff <= tolerance, "mismatch at index {}: got {}, expected {}, diff {}", i, a, e, diff);
    }
}
```

### Commit 7: integration test

**Implemented test location:** `crates/vulkan-backend/tests/embedding_lookup.rs`

The original plan placed this test under `crates/runtime/tests/`; the
implemented GPU integration test belongs to the Vulkan backend test suite.

```rust
#[test]
#[ignore]
fn test_embedding_lookup_f32() {
    // Create VulkanContext
    // Load small F32 GGUF model
    // Find embedding tensor
    // Upload embedding tensor
    // Dispatch GPU embedding lookup for token_id=0
    // Compute CPU reference
    // Compare with tolerance 1e-5
}
```

**Test model:** Need a small F32 GGUF. Options:
- Convert a small Q4_0 model to F32 using llama.cpp `llama-quantize --type f32`
- Use a tiny test model (e.g., 1-layer, small vocab) for fast iteration

## Explicit deferrals

| Item | Reason | Phase |
|---|---|---|
| Quantized embedding dequant (Q4_0, Q8_0) | Shader complexity; establish F32 pipeline first | Phase 8 |
| F16 embedding lookup | Requires float16 Vulkan capability check | Phase 3.5 (after F32 works) |
| Batch embedding lookup (multiple tokens) | Single-token validates the pipeline first | Phase 4 |
| Memory-mapped GGUF loading | `std::fs::File` + seek is sufficient for now | Phase 5 |
| Tokenizer integration | Use hardcoded token IDs for testing | Phase 6 |
| Async dispatch / pipeline cache | Synchronous path validates correctness first | Phase 8 |

## Shader compilation

```bash
./scripts/compile-shaders.sh
```

Verify `glslangValidator` is installed. If not: `sudo apt install glslang-tools` (Ubuntu/Debian) or `sudo dnf install glslang` (Fedora).

## Validation checklist

- [x] Shader compiles to SPIR-V without warnings
- [x] Pipeline creates with push constants + 3 descriptor bindings (weights, token IDs, output)
- [x] Descriptor set updates with all buffer bindings
- [x] Push constants are set correctly before dispatch
- [x] Dispatch completes without Vulkan errors
- [x] Readback returns correct number of f32 values
- [x] GPU output matches CPU reference within tolerance
- [x] Integration test passes with `--ignored`
- [x] `cargo test` still passes (312 unit tests at the time of this historical plan)
- [x] `cargo clippy` is clean

## Architectural notes

- **Memory barriers:** The smoke test works without explicit barriers on RADV, but production code needs a `vkCmdPipelineBarrier` between transfer (upload) and compute (dispatch) stages. Consider adding this in Commit 2 or 3.
- **Descriptor set layout binding flags:** `VK_DESCRIPTOR_BINDING_STORAGE_BUFFER_UPDATE_AFTER_BIND_BIT` is not needed for static bindings. Keep it simple.
- **Push constant size limit:** Vulkan guarantees at least 128 bytes of push constant space. Our 8-byte struct is well within limits.
- **GGUF column-major:** GGUF stores tensors in column-major order. For the embedding table `[hidden_dim, vocab_size]`, element `(row, col)` is at offset `row + col * hidden_dim`. This means token `col`'s embedding starts at `col * hidden_dim` and spans `hidden_dim` elements — which is what the shader computes.

## Phase 3.2 results

### Smoke test: PASSED

- **Hardware:** AMD Radeon AI PRO R9700 (RADV driver)
- **Test:** `crates/vulkan-backend/tests/embedding_lookup.rs::smoke_embedding_lookup_f32`
- **Configuration:** vocab=4, dim=64, batch=1, token_id=2, 4096-byte buffers
- **Result:** GPU output matches CPU reference (`embedding_lookup_f32`) within 1e-5 tolerance across all 64 elements

### Validated command sequence

```
upload_to_device_local(weights)
upload_to_device_local(token_ids)
allocate_command_buffer → begin
  → record_barrier_transfer_to_compute(weight_buf, token_buf)
  → bind_compute_pipeline
  → push_constants(vocab_size, embedding_dim, batch_size)
  → bind_descriptor_sets(3 bindings)
  → dispatch(1, 1, 1)
  → record_barrier_compute_to_transfer(output_buf)
end → submit_and_wait
wait_idle → readback_buffer_data(output_buf)
```

### Validated descriptor layout

| Binding | Type | Buffer | Range |
|---|---|---|---|
| 0 | STORAGE_BUFFER (readonly) | weights | total_weights bytes |
| 1 | STORAGE_BUFFER (readonly) | token IDs | 4 bytes |
| 2 | STORAGE_BUFFER (readwrite) | output | embedding_dim * 4 bytes |

### Validated push constant layout (12 bytes)

| Offset | Field | Type |
|---|---|---|
| 0 | vocab_size | u32 |
| 4 | embedding_dim | u32 |
| 8 | batch_size | u32 |

### Bugs discovered and fixed

1. **DescriptorBufferInfo lifetime bug**: `vk::DescriptorBufferInfo` holds a `vk::Buffer` handle, not a reference. If the `VulkanBuffer` RAII wrapper is dropped before the descriptor set update, the handle dangles. **Fix:** ensure `VulkanBuffer` instances outlive the descriptor set update call — hold references through the submit scope.

2. **NULL buffer barrier → GPUVM fault**: Passing `vk::NULL_HANDLE` as the buffer in `vkCmdPipelineBarrier` caused a GPUVM page fault on RADV. **Fix:** always pass the actual buffer handle in barrier structs. Never use NULL handles in `vk::BufferMemoryBarrier2` on RADV.

### Lessons learned

- **Buffer alignment matters:** 4096-byte buffers avoid GPU memory controller alignment issues on RADV. Smaller buffers may work but are less portable.
- **Explicit barriers are mandatory:** The original smoke test worked without barriers on RADV by accident. Production kernels must use explicit `transfer→compute` and `compute→transfer` barriers.
- **Synthetic data first:** Using synthetic F32 embeddings (sequential values 0..N) makes debugging trivial — you can visually inspect the output array for correctness without needing a real model.
- **Shader evolved from plan:** The original plan had a single-token shader with 2 bindings. The implemented version is batched with 3 bindings (token IDs as a buffer, not push constant). This is more flexible and supports future batch prefill.
- **Push constants vs buffer for token IDs:** Token IDs are passed via a storage buffer (binding 1) rather than push constants, allowing variable-length token sequences without push constant size limits.

### What remains (Phase 3.3+)

- Runtime orchestration: `ModelLoader` with GGUF tensor upload
- Real-model integration test: end-to-end embedding lookup with F32 GGUF
- F16 embedding variant (Phase 3.5)
- Modularization of `vulkan-backend/lib.rs` (post-Phase 3 housekeeping)
