# Current Status — Vulkanize

> Last updated: Phase 3 backend preparation complete. Phase 3 runtime integration next.

## Build and test status

- `cargo build --release` — clean, 0 warnings
- `cargo test` — 243 unit tests pass (106 gguf, 137 vulkan-backend)
- `cargo test --test smoke_no_op -- --ignored` — 1 integration test passes (end-to-end GPU dispatch with explicit barriers)
- `cargo clippy --all-targets --all-features -- -D warnings` — clean
- 1 integration test (`smoke_no_op`) — ignored by default, passes with `--ignored`

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
- Array-value skipping in metadata region (for tensor offset calculation)
- 106 unit tests with synthetic GGUF byte arrays
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

## Current crate responsibilities

| Crate | State | What it does |
|---|---|---|
| `vulkanize-gguf` | **Functional** | Parses GGUF v3 files, exposes typed metadata and tensor descriptors |
| `vulkanize-vulkan-backend` | **Functional** | Full Vulkan init through compute dispatch. `VulkanContext` owns instance/device/queue/command pool/memory selector. `VulkanBuffer` provides RAII buffer+memory management. `CommandBuffer` and `Fence` provide command recording, submission, and sync. `ShaderModule` loads SPIR-V. `ComputePipeline` manages pipeline+layout with push constant support. `DescriptorSetLayout`/`DescriptorPool`/`DescriptorSet` manage descriptor lifecycle with multi-buffer layouts. `DescriptorBinding` configures individual bindings. `upload_to_device_local()` and `readback_buffer_data()` provide full CPU↔GPU data movement. `CommandBuffer::dispatch()` executes compute shaders. `CommandBuffer::push_constants()` pushes per-dispatch data. `CommandBuffer::record_barrier_transfer_to_compute()` and `record_barrier_compute_to_transfer()` provide explicit memory synchronization. |
| `vulkanize-runtime` | **Stub** | `pub fn init() {}` — placeholder |
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
| Embedding lookup shader (F32) | Done (shader + SPV only) |
| Uniform buffer descriptors | Not yet |

## Remaining limitations

- No async dispatch — all operations are synchronous with fence wait
- No pipeline cache
- No tensor/weight loading from GGUF files (upload path exists but not integrated with gguf crate)
- No runtime orchestration for embedding lookup (shader + backend APIs exist, runtime integration is next)
- No CPU reference computation for validation
- No integration test for embedding lookup with real GGUF model
- Embedding shader is F32 only — F16 and quantized variants not yet implemented

## OpenCode workflow assumptions

- Sessions are driven by `AGENTS.md` instructions and `docs/`
- Each session starts by reading `docs/roadmap.md` and `docs/design.md`
- Rust code changes are validated with `cargo build --release` and `cargo test`
- No CUDA, HIP, or ROCm dependencies are ever added
- No llama.cpp vendoring or runtime dependency
- Shader changes and Rust crate changes are committed separately unless tightly coupled
- `docs/` files are the source of truth for architecture decisions
- Future sessions should read `docs/current-status.md` first to reconstruct state
