# AMD Vulkan Backend Notes

## Target platform

- **GPU**: AMD Radeon RX 7000 / 6000 series (RDNA3 / RDNA2)
- **Driver**: RADV (Mesa open-source, Linux primary); AMDVLK is a possible future alternative
- **OS**: Linux (primary); Windows is a future target

Development and GPU validation have primarily been performed with RADV/Mesa on Linux. AMDVLK and Windows are not equally validated.

## Device selection

```c
// AMD vendor ID
#define AMD_VENDOR_ID 0x1002

// Prefer AMD devices when multiple GPUs present
VkPhysicalDeviceProperties props;
vkGetPhysicalDeviceProperties(phys_dev, &props);
if (props.vendorID == AMD_VENDOR_ID) {
    // Prefer this device
}
```

Selection priority:
1. AMD vendor ID match
2. Highest compute queue count
3. Largest local heap size
4. Validation layer support in debug builds

## Queue families

Vulkanize needs at minimum:
- **Compute queue** (`VK_QUEUE_COMPUTE_BIT`) — mandatory
- **Graphics queue** — NOT needed (compute-only workload)

Query with `vkGetPhysicalDeviceQueueFamilyProperties`. A single queue family supporting `VK_QUEUE_COMPUTE_BIT` is sufficient.

## Memory types

AMD GPUs expose these key memory types:

| Flag | Use case |
|---|---|
| `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` | Model weights, KV cache — fast GPU access |
| `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT \| HOST_COHERENT_BIT` | Staging buffers for readback (logits, sampling) |
| `VK_MEMORY_PROPERTY_HOST_CACHED_BIT` | Optional; avoid for performance-critical paths |

Use `vkGetPhysicalDeviceMemoryProperties` to find the right `memoryTypeIndex` for each allocation. Cache the lookup — it's constant per device.

### Memory type selection helper

```rust
fn find_memory_type(
    memory_type_bits: u32,
    properties: VkMemoryPropertyFlags,
) -> Option<u32> {
    (0..memory_properties.memoryTypeCount).find(|&i| {
        (memory_type_bits & (1 << i)) != 0
            && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties
    })
}
```

## Buffer strategy

### Model weights

- Allocate device-local buffers for each tensor (or one large buffer with offsets)
- Initialize via staging buffer: host-visible upload → `vkCmdCopyBuffer` → device-local
- Once uploaded, staging buffers can be freed
- Consider memory mapping the GGUF file and using it as the source for a single bulk upload

### KV cache

- Device-local buffer, pre-allocated for max context length
- Key and value matrices: `[max_seq_len][num_kv_heads][head_dim]` per layer
- Grown incrementally; no need to reallocate per token

### Logits readback

- Staging buffer with `HOST_VISIBLE | HOST_COHERENT`
- Async copy from device-local logits → staging → CPU reads for sampling
- Map once at startup, keep mapped (coherent means no explicit flush needed)

## Compute pipeline

```rust
// Simplified flow
let shader_module = vkCreateShaderModule(&spv_bytes);
let entry_point = "main";
let pipeline_layout = create_pipeline_layout(&descriptor_set_layouts);
let pipeline = vkCreateComputePipelines(pipeline_layout, shader_module, entry_point);
```

### Descriptor sets

- **Set 0**: Uniform buffers and storage buffers for model weights
- **Set 1**: Push constants or small uniform buffers for per-dispatch config (shape dims, token index)
- Bind once per kernel type; update bindings only when switching kernels

## Synchronization

- Use `VkSemaphore` for stage synchronization between submissions
- Timeline semaphores (`VK_KHR_timeline_semaphore`) for fine-grained multi-batch ordering
- Fences (`VkFence`) for CPU-side completion wait
- **Avoid** `vkDeviceWaitIdle` in the hot path — only use for cleanup/shutdown

## Validation layers

Enable in debug builds:
```
VK_LAYER_KHRONOS_validation
```

Set via environment or `VkDebugUtilsMessengerEXT`. Disable in release builds for performance.

## SPIR-V compilation

- Shaders written in GLSL, compiled to SPIR-V before GPU integration tests
- Toolchain: `glslangValidator` from the glslang package
- Generated `.spv` files live in `shaders/` alongside source `.comp.glsl` after compilation
- `scripts/compile-shaders.sh` handles current shader compilation
- Shader sources and the compile script are committed to git; generated `.spv` binaries are ignored

## Performance considerations on AMD

- **Wavefront size**: RDNA uses 64-wide wavefronts. Align workgroup sizes to 64 or 128 for occupancy.
- **Local memory (SGPR/VGPR)**: Minimize use of `shared`/`block` memory unless doing matrix multiply tiling.
- **Memory coalescing**: Ensure thread IDs map to sequential memory accesses for load/store throughput.
- **Avoid divergent branches**: AMD GPUs handle divergence poorly; prefer predication or uniform control flow.

## Rust Vulkan bindings

Recommended crate: `ash` (low-level, direct mapping to Vulkan C API) or `vulkano` (higher-level RAII). Prefer `ash` for:
- Full control over allocation patterns
- No hidden synchronization overhead
- Direct access to AMD-specific extensions if needed

Alternative: `raw-window-handle` is NOT needed — this is a headless compute application with no surfaces or swapchains.

## Debugging

- `VK_INSTANCE_LAYERS`: set to `VK_LAYER_KHRONOS_validation` for validation output
- RenderDoc can capture Vulkan compute frames for shader inspection
- `vulkaninfo` (from `vulkan-tools`) shows device capabilities and memory types
