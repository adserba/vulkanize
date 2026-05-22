# Current Status — Vulkanize

> Last updated: Phase 2.3.1 complete.

## Build and test status

- `cargo build --release` — clean, 0 warnings
- `cargo test` — 151 tests pass (106 gguf, 45 vulkan-backend)
- No integration tests or benchmarks yet

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

## Current crate responsibilities

| Crate | State | What it does |
|---|---|---|
| `vulkanize-gguf` | **Functional** | Parses GGUF v3 files, exposes typed metadata and tensor descriptors |
| `vulkanize-vulkan-backend` | **Partial** | Full Vulkan init through buffer allocation. `VulkanContext` owns instance/device/queue/command pool/memory selector. `VulkanBuffer` provides RAII buffer+memory management with mapping support. No pipelines, dispatches, or command recording yet. |
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

## What is NOT yet implemented

- Staging upload path (vkMapMemory → memcpy → vkCmdCopyBuffer → unmap + free)
- Compute pipeline creation from `.spv` modules
- Command buffer recording, submission, or synchronization
- Any shader code or SPIR-V binaries
- Weight loading or memory mapping
- Forward pass, sampling, or generation
- HTTP server / OpenAI API

## OpenCode workflow assumptions

- Sessions are driven by `AGENTS.md` instructions and `docs/`
- Each session starts by reading `docs/roadmap.md` and `docs/design.md`
- Rust code changes are validated with `cargo build --release` and `cargo test`
- No CUDA, HIP, or ROCm dependencies are ever added
- No llama.cpp vendoring or runtime dependency
- Shader changes and Rust crate changes are committed separately unless tightly coupled
- `docs/` files are the source of truth for architecture decisions
- Future sessions should read `docs/current-status.md` first to reconstruct state
