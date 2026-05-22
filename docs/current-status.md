# Current Status — Vulkanize

> Last updated: GGUF parser real-file compatibility fix complete. Phase 3 runtime orchestration next.

## Build and test status

- `cargo build --release` — clean, 0 warnings
- `cargo test` — 316 unit tests pass (158 gguf, 137 vulkan-backend, 21 runtime)
- `cargo test --test smoke_no_op -- --ignored` — passes (end-to-end GPU dispatch with explicit barriers)
- `cargo test --test embedding_lookup -- --ignored` — passes (GPU embedding lookup, synthetic F32, validated on AMD Radeon AI PRO R9700 / RADV)
- `cargo clippy --all-targets --all-features -- -D warnings` — clean
- 2 integration tests — both ignored by default, pass with `--ignored`

## Completed milestones

### Phase 0: Foundation

- Workspace with 5 crates and correct dependency graph
- Design documentation in `docs/`
- `.gitignore`, commit conventions

### Phase 1: GGUF parser

- Full GGUF v3 parser: header, metadata, tensor descriptors
- `GgufTensorType` enum covering 36 GGML types (F32 through TQ2_0)
- `TensorDescriptor` with name, dims, shape, dtype, offset, `element_count()`
- `GgufMetadata` with `extract_model_arch()` — extracts architecture, block count, context length, embedding/FFN dimensions, head counts, GQA, RoPE base, file type
- `parse_gguf_full_from_path()` — reads file, returns `(GgufHeader, GgufMetadata, GgufTensors)`
- Metadata arrays are tolerated and skipped safely by element type/length, including tokenizer token/merge arrays
- Tensor descriptor offsets are tracked as GGUF-relative offsets plus aligned tensor data-section start; `read_tensor_bytes()` resolves absolute reads as `data_start + offset`
- Safer integer extraction for u32-like architecture metadata rejects negative and out-of-range values
- 158 unit tests with synthetic GGUF byte arrays
- `vulkanize inspect model.gguf` prints model architecture, metadata, tensor list

### Phase 2.1: Vulkan context skeleton

- Vulkan loader/entry initialization via `ash`
- Instance creation with optional validation layers (debug builds)
- Physical device enumeration and info collection
- `PhysicalDeviceInfo` and `QueueFamilyInfo` types
- Device selection with AMD vendor ID 0x1002 preference (scoring: AMD bonus, discrete GPU bonus, compute queue count, driver version tie-break)
- `vulkanize vulkan-info` CLI command

### Phase 2.2: Logical device, compute queue, command pool

- `VulkanDevice` — wraps `ash::Device` + `compute_queue: vk::Queue`
- `QueueFamilySelection` — selected compute queue family index, count, flags
- `CommandResources` — command pool handle
- `VulkanContext` composes `VulkanDevice` + `QueueFamilySelection` + `CommandResources`
- Clean Drop order: command pool → device → instance
- 27 unit tests (formatting helpers, type construction, error display)

### Phase 2.3.1: Buffer allocation layer

- `MemoryTypeSelector` — wraps `vkGetPhysicalDeviceMemoryProperties`, provides `find_memory_type(bits, flags)` helper
- `VulkanBuffer` — RAII wrapper owning `VkBuffer` + `VkDeviceMemory` with correct Drop cleanup (unmap → free memory → destroy buffer)
- Buffer creation helpers on `VulkanContext`:
  - `create_device_local_buffer(size, usage)` — DEVICE_LOCAL memory for weights/KV cache
  - `create_host_visible_buffer(size)` — HOST_VISIBLE | HOST_COHERENT, auto-mapped, for staging uploads
  - `create_buffer(size, usage, memory_flags)` — generic creation with arbitrary flags
- Mapping support on `VulkanBuffer`:
  - `map()` / `unmap()` — explicit control with double-map protection
  - `write_data(&[u8])` — convenience: map → memcpy → unmap
  - `write_at(offset, &[u8])` — offset-aware writes for future suballocation
- Host-visible buffers are mapped immediately at creation time
- Error types: `BufferCreation`, `MemoryAllocation`, `MemoryBinding`, `NoSuitableMemoryType`, `MemoryMapping`
- 18 new unit tests: memory type selection logic (8), error display (5), buffer struct properties (5)
- `MemoryTypeSelector` cached in `VulkanContext` (constant per device, looked up once)

### Phase 2.3.2: Staging upload path

- `Fence` — RAII wrapper for `VkFence` with `create()`, `create_signaled()`, `wait()`, `reset()`, and correct `Drop` cleanup
- `CommandBuffer` — RAII wrapper for `VkCommandBuffer` with:
  - `begin_one_time_submit()` / `begin()` — start recording with appropriate flags
  - `end()` — finalize recording
  - `record_copy_buffer(src, dst, size)` — record `vkCmdCopyBuffer` with usage flag validation
  - `submit_and_wait(queue, queue_family_index)` — submit, wait on fence, reset for reuse
  - Automatic free-to-pool on `Drop`
- `VulkanContext` command buffer management:
  - `allocate_command_buffer()` — allocate primary command buffer from pool
  - `execute_immediate(closure)` — allocate, record via closure, submit, wait, cleanup
- Transfer/upload operations on `VulkanContext`:
  - `copy_buffer(src, dst, size)` — full synchronous buffer-to-buffer copy with size validation
  - `upload_to_device_local(dst, data)` — end-to-end CPU→GPU upload: staging buffer → map/write → copy → cleanup
- Validation:
  - `TRANSFER_SRC` / `TRANSFER_DST` usage flag checks on buffer copy
  - `DEVICE_LOCAL` memory property check on upload target
  - Size bounds checks (data size vs buffer size, copy size vs src/dst size)
  - Empty data fast-path (no-op)
- Error types: `CommandBufferAllocation`, `FenceCreation`, `FenceWait`, `TransferValidation(String)`
- 22 new unit tests: Fence/CommandBuffer struct construction (5), error display (4), transfer validation logic (8), upload validation logic (5)
- Synchronization model: fence-based, synchronous, single-queue. Designed for future async transfer queue extension.

### Phase 2.3.3: Shader module loading

- `ShaderModule` — RAII wrapper owning `VkShaderModule` with correct `Drop` cleanup (destroy shader module)
- SPIR-V loading:
  - `ShaderModule::from_spv_bytes(device, &[u8])` — create from raw bytes with alignment validation
  - `load_spv_file(&Path)` — standalone file I/O helper, returns `Vec<u8>`
  - `VulkanContext::create_shader_module_from_spv_bytes(&[u8])` — context-integrated creation
  - `VulkanContext::create_shader_module_from_spv_file(&Path)` — file → module in one call
- Pre-validation:
  - Empty byte slice rejected with `InvalidSpvBytes` error
  - Non-word-aligned byte length (not multiple of 4) rejected with descriptive error
  - Vulkan API failures wrapped as `ShaderModuleCreation(vk::Result)`
- Error types: `ShaderModuleCreation`, `InvalidSpvBytes(String)`, `SpvLoadingError(String)`
- `word_count()` accessor on `ShaderModule` for SPIR-V word count
- `shaders/` directory created for precompiled `.spv` binaries
- 13 new unit tests: struct construction (5), word count / byte-to-word logic (4), error display (4)
- No runtime shader compilation — `.spv` files are prebuilt externally

### Phase 2.3.4: Compute pipeline creation

- `ComputePipeline` — RAII wrapper owning `VkPipeline` + `VkPipelineLayout` with correct `Drop` cleanup order (destroy pipeline first, then layout)
- Pipeline creation from `ShaderModule` + entry point name:
  - `ComputePipeline::new(device, shader_module, entry_point)` — direct creation (no descriptor sets)
  - `ComputePipeline::new_with_layouts(device, shader_module, entry_point, set_layouts)` — with descriptor set layouts
  - `VulkanContext::create_compute_pipeline(shader_module, entry_point)` — context-integrated
  - `VulkanContext::create_compute_pipeline_with_layouts(shader_module, entry_point, set_layouts)` — context-integrated with layouts
- Entry point validation:
  - CString conversion failure caught as `InvalidEntryPoint` error
  - Null bytes in entry point name rejected before Vulkan API call
- Error types: `PipelineLayoutCreation(vk::Result)`, `ComputePipelineCreation(vk::Result)`, `InvalidEntryPoint(String)`
- Accessors: `handle()` returns `vk::Pipeline`, `layout()` returns `vk::PipelineLayout`
- 14 new unit tests: struct construction (5), CString entry point validation (5), error display (3), drop safety (1)

### Phase 2.3.5: Descriptor sets and compute dispatch

- `DescriptorSetLayout` — RAII wrapper owning `VkDescriptorSetLayout`
- `DescriptorPool` — RAII wrapper owning `VkDescriptorPool` with `allocate()` and `free_all()`
- `DescriptorSet` — RAII wrapper owning `VkDescriptorSet`, auto-frees back to pool on drop
- Descriptor creation helpers on `VulkanContext`:
  - `create_descriptor_set_layout_with_buffer(binding)` — single storage buffer binding layout
  - `create_descriptor_set_with_buffer(layout, binding, buffer, range)` — full pool + set + update in one call
- Command buffer recording extensions:
  - `CommandBuffer::bind_compute_pipeline(pipeline)` — `vkCmdBindPipeline(COMPUTE)`
  - `CommandBuffer::bind_descriptor_sets(layout, set_offset, descriptor_sets)` — `vkCmdBindDescriptorSets`
  - `CommandBuffer::dispatch(group_x, group_y, group_z)` — `vkCmdDispatch`
- `VulkanContext::readback_buffer_data(src, size)` — device-local → host-visible staging copy → CPU readback
- `VulkanContext::wait_idle()` — convenience wrapper for `vkDeviceWaitIdle`
- Error types: `DescriptorSetLayoutCreation`, `DescriptorPoolCreation`, `DescriptorSetAllocation`, `CommandBufferNotRecording`
- `shaders/no_op.comp.glsl` + `shaders/no_op.spv` — trivial compute shader writing `0x12345678` to storage buffer

### Phase 2.3.6: Smoke test

- `crates/vulkan-backend/tests/smoke_no_op.rs` — end-to-end integration test:
  1. Create VulkanContext
  2. Create device-local buffer (4096 bytes, STORAGE_BUFFER + TRANSFER_DST + TRANSFER_SRC)
  3. Upload initial data (0xDEADBEEF) via `upload_to_device_local`
  4. Readback and verify upload correctness
  5. Load `no_op.spv` shader module
  6. Create descriptor set layout + pipeline with layouts + descriptor set
  7. Record command buffer: begin → bind pipeline → bind descriptors → dispatch(1,1,1) → end → submit_and_wait
  8. Wait idle, readback result, verify shader wrote 0x12345678
- Test is `#[ignore]` by default (requires GPU). Run with `cargo test --test smoke_no_op -- --ignored`
- Validates: buffer allocation, upload, readback, shader loading, descriptor sets, pipeline creation, dispatch, synchronization
- Updated to use explicit memory barriers: `record_barrier_transfer_to_compute` before dispatch, `record_barrier_compute_to_transfer` after dispatch

### Phase 3.1: GGUF tensor helpers + CPU reference

- `GgufTensorType::type_block_size()` — returns bytes per block for all known GGML types (F32=4, F16=2, Q4_0=64, Q5_K=176, etc.), `None` for `Q4_1_F16` (uncertain block size)
- `GgufTensorType::block_size()` — returns elements per block (1 for non-quantized, 32/64/128 for quantized types)
- `GgufTensorType::tensor_byte_size(element_count)` — computes total byte size from element count with overflow/alignment guards
- `GgufTensors::find_tensor(name)` — exact name lookup, returns `Option<&TensorDescriptor>`
- `GgufTensors::find_token_embedding(&ModelArch)` — finds token embedding tensor with:
  - Exact `token_embd.weight` name preference
  - Fallback `_embd.weight` suffix pattern matching with ambiguity detection
  - Architecture validation: n_dims == 2, one dimension matches `embedding_length`
  - Returns `(TensorDescriptor, vocab_size)` or `TensorLookupError`
- `TensorLookupError` — error enum with `NotFound`, `Ambiguous`, `InvalidShape`, `DimensionMismatch`, `UnsupportedType`, `Overflow` variants, implements `Display` + `Error`
- `read_tensor_bytes(file, desc)` — seeks to tensor offset, computes byte size, reads raw bytes with `read_exact`
- `runtime::embedding` module — CPU reference implementations for correctness validation:
  - `embedding_lookup_f32(weights, hidden_dim, token_id)` — column-major embedding table lookup
  - `f16_to_f32_bytes(bytes)` — IEEE 754 F16 (little-endian) to F32 conversion, handles normal/denormal/zero/inf/NaN
  - `compare_f32(actual, expected, tolerance)` — element-wise comparison with tolerance, returns detailed mismatch info
- 48 new unit tests: type_block_size (15), block_size (4), tensor_byte_size (7), find_tensor (2), find_token_embedding (8), TensorLookupError display (4), read_tensor_bytes (5), embedding lookup (3), F16 conversion (11), compare_f32 (5)
- GGUF crate stays parsing/I/O only — no computation or dequantization

### Phase 3.0: Backend preparation (multi-buffer descriptors, push constants, barriers)

- `shaders/embedding_lookup.comp.glsl` + `shaders/embedding_lookup.spv` — batched embedding lookup shader:
  - 3 storage buffer bindings: embedding weights (float, binding 0), token IDs (uint, binding 1), output hidden state (float, binding 2)
  - Push constants: vocab_size, embedding_dim, batch_size
  - local_size_x = 64 for AMD wavefront alignment
  - Guards: out-of-range global ID, invalid token ID (writes 0.0)
- `DescriptorBinding` — config struct for individual descriptor bindings (binding number, type, stage flags)
- `VulkanContext::create_descriptor_set_layout_with_buffers(&[DescriptorBinding])` — multi-binding layout creation with consecutive-binding validation
- `VulkanContext::create_descriptor_set_with_buffers(layout, &[(binding, &buffer, range)])` — multi-buffer descriptor set creation (pool + allocate + update)
- Existing single-buffer helpers (`create_descriptor_set_layout_with_buffer`, `create_descriptor_set_with_buffer`) preserved for backward compatibility
- `ComputePipeline::new_with_push_constants(device, shader, entry, layouts, push_constant_range)` — pipeline with push constant range in layout
- `VulkanContext::create_compute_pipeline_with_push_constants(...)` — context-integrated push constant pipeline creation
- `CommandBuffer::push_constants(layout, offset, data)` — records `vkCmdPushConstants` with word-alignment validation
- `CommandBuffer::record_barrier_transfer_to_compute(&[&VulkanBuffer])` — explicit pipeline barrier: TRANSFER_WRITE → SHADER_READ
- `CommandBuffer::record_barrier_compute_to_transfer(&[&VulkanBuffer])` — explicit pipeline barrier: SHADER_WRITE → TRANSFER_READ
- `VulkanError::PushConstantValidation(String)` — new error variant for push constant and descriptor binding validation
- 30 new unit tests: descriptor binding construction/validation (9), push constant range construction (4), push constant data alignment (4), error display (2), single-buffer regression (3), barrier model (8)
- Smoke test updated with explicit barriers, passes on RADV

### Phase 3.2: GPU embedding lookup smoke test

- `crates/vulkan-backend/tests/embedding_lookup.rs` — end-to-end GPU embedding lookup integration test:
  1. Create VulkanContext
  2. Build synthetic F32 embedding table (vocab=4, dim=64, values 0.0..255.0)
  3. Create 3 device-local buffers (weights, token IDs, output), 4096 bytes each
  4. Upload weights + token IDs via `upload_to_device_local`
  5. Load `embedding_lookup.spv` shader
  6. Create 3-binding descriptor layout + descriptor set
  7. Create pipeline with 12-byte push constant range (vocab_size, embedding_dim, batch_size)
  8. Record command buffer: begin → transfer→compute barrier → bind pipeline → push constants → bind descriptors → dispatch(1,1,1) → compute→transfer barrier → end → submit_and_wait
  9. Wait idle, readback output, convert to f32
  10. Compare GPU output vs CPU reference (`embedding_lookup_f32`) with 1e-5 tolerance
- Validated on AMD Radeon AI PRO R9700 (RADV driver)
- Confirms: push constants, multi-buffer descriptors, explicit transfer/compute barriers, shader dispatch, readback, numerical correctness for synthetic F32 embeddings
- `#[ignore]` by default. Run with `cargo test --test embedding_lookup -- --ignored`

#### Vulkan bugs discovered during Phase 3.2

- **DescriptorBufferInfo lifetime bug**: `vk::DescriptorBufferInfo` holds a `vk::Buffer` handle, not a reference. If the `VulkanBuffer` RAII wrapper is dropped before the descriptor set update, the handle becomes dangling. Fix: ensure `VulkanBuffer` instances outlive the descriptor set update call (hold references through the `execute_immediate` / submit scope).
- **NULL buffer in pipeline barrier → GPUVM fault**: Passing `vk::NULL_HANDLE` as the buffer in `vkCmdPipelineBarrier`'s `vk::ImageMemoryBarrier2` / `vk::BufferMemoryBarrier2` caused a GPUVM page fault on RADV. Fix: always pass the actual buffer handle in the barrier, even when the barrier only needs stage/sync flags. Never use NULL handles in barrier structs on RADV.

## Current crate responsibilities

| Crate | State | What it does |
|---|---|---|
| `vulkanize-gguf` | **Functional** | Parses GGUF v3 files, exposes typed scalar metadata and tensor descriptors. Metadata arrays are consumed safely without materializing tokenizer payloads. Tensor offsets are resolved relative to the aligned data section. Helper methods: `type_block_size()`, `block_size()`, `tensor_byte_size()`. Lookup: `find_tensor()`, `find_token_embedding()`. I/O: `read_tensor_bytes()`. Error: `TensorLookupError`. |
| `vulkanize-vulkan-backend` | **Functional** | Full Vulkan init through compute dispatch. `VulkanContext` owns instance/device/queue/command pool/memory selector. `VulkanBuffer` provides RAII buffer+memory management. `CommandBuffer` and `Fence` provide command recording, submission, and sync. `ShaderModule` loads SPIR-V. `ComputePipeline` manages pipeline+layout with push constant support. `DescriptorSetLayout`/`DescriptorPool`/`DescriptorSet` manage descriptor lifecycle with multi-buffer layouts. `DescriptorBinding` configures individual bindings. `upload_to_device_local()` and `readback_buffer_data()` provide full CPU↔GPU data movement. `CommandBuffer::dispatch()` executes compute shaders. `CommandBuffer::push_constants()` pushes per-dispatch data. `CommandBuffer::record_barrier_transfer_to_compute()` and `record_barrier_compute_to_transfer()` provide explicit memory synchronization. |
| `vulkanize-runtime` | **Partial** | `embedding` module with CPU reference: `embedding_lookup_f32()`, `f16_to_f32_bytes()`, `compare_f32()` for correctness validation |
| `vulkanize-api` | **Stub** | `pub fn init() {}` — placeholder |
| `vulkanize` (cli) | **Partial** | `inspect` works, `vulkan-info` works. `generate` and `serve` print "not yet implemented". |

## Implemented CLI commands

```
vulkanize inspect model.gguf       # fully working — prints arch, metadata, tensors
vulkanize vulkan-info              # fully working — prints GPU info, device, queue, command pool
vulkanize generate model.gguf [p]  # stub — exits with "not yet implemented"
vulkanize serve                     # stub — exits with "not yet implemented"
```

## Current backend capabilities

| Capability | Status |
|---|---|
| Instance/device/queue creation | Done |
| Buffer allocation (device-local + host-visible) | Done |
| Buffer mapping and data writes | Done |
| Staging upload (CPU → GPU) | Done |
| Readback (GPU → CPU) | Done |
| Shader module loading from .spv | Done |
| Compute pipeline creation | Done |
| Compute pipeline with push constants | Done |
| Descriptor set layout/pool/set | Done |
| Multi-buffer descriptor set layouts | Done |
| Multi-buffer descriptor set creation | Done |
| Push constants (pipeline + command buffer) | Done |
| Command buffer recording + submission | Done |
| Compute dispatch | Done |
| Fence-based synchronization | Done |
| Explicit memory barriers (transfer↔compute) | Done |
| Embedding lookup shader (F32) | Done (shader + SPV + GPU smoke test) |
| Embedding lookup GPU smoke test (synthetic F32) | Done (validated on RADV) |
| GGUF tensor byte size calculation | Done |
| GGUF token embedding lookup | Done |
| GGUF raw tensor byte reading | Done |
| CPU reference embedding lookup (F32) | Done |
| CPU reference F16→F32 conversion | Done |
| CPU reference float comparison | Done |
| Uniform buffer descriptors | Not yet |

## Remaining limitations

- No async dispatch — all operations are synchronous with fence wait
- No pipeline cache
- No runtime orchestration for embedding lookup (shader + backend APIs + GGUF helpers + CPU reference + synthetic GPU test exist, runtime integration is next)
- No integration test for embedding lookup with real GGUF model (synthetic test passes; real-model test requires GGUF tensor upload in runtime)
- Embedding shader is F32 only — F16 and quantized variants not yet implemented
- `read_tensor_bytes()` requires `std::fs::File` — memory-mapped loading not yet implemented

## OpenCode workflow assumptions

- Sessions are driven by `AGENTS.md` instructions and `docs/`
- Each session starts by reading `docs/roadmap.md` and `docs/design.md`
- Rust code changes are validated with `cargo build --release` and `cargo test`
- No CUDA, HIP, or ROCm dependencies are ever added
- No llama.cpp vendoring or runtime dependency
- Shader changes and Rust crate changes are committed separately unless tightly coupled
- `docs/` files are the source of truth for architecture decisions
- Future sessions should read `docs/current-status.md` first to reconstruct state
