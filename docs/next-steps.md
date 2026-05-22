# Next Steps — Vulkanize

> Immediate engineering milestones after Phase 2.3.2.

## AI workflow: reconstructing project state

Before making changes, a new session should:

1. Read `docs/project-vision.md` — understand the goal and constraints
2. Read `docs/current-status.md` — know what is built and what is not
3. Read `docs/roadmap.md` — find the current phase and what is next
4. Read `docs/design.md` — review crate contracts and architecture
5. Scan `crates/*/src/lib.rs` and `crates/cli/src/main.rs` — verify code matches documentation
6. Run `cargo build --release && cargo test` — confirm the baseline is green

Then work on the next roadmap item. After completing it, update `docs/roadmap.md` and `docs/current-status.md`.

## Phase 2.3: Buffers and pipelines (next)

This phase adds the GPU resource layer: buffers for data, pipelines for shaders, and the command submission cycle.

### 2.3.1 Buffer allocation

- Implement buffer allocation helpers in `vulkan-backend`:
  - Device-local buffers (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`) for weights and KV cache
  - Host-visible staging buffers (`HOST_VISIBLE | HOST_COHERENT`) for uploads and readback
- Memory type selection using `vkGetPhysicalDeviceMemoryProperties`
- Add `BufferInfo` type: handle, size, memory type, mapped pointer (for staging)
- Add `Drop` implementations that free memory and destroy buffers
- Tests: none requiring GPU; test memory type selection logic with mock data if feasible

### 2.3.2 Staging upload path (DONE)

- `Fence` — RAII fence wrapper with create/wait/reset
- `CommandBuffer` — RAII command buffer with begin/end/record/submit_and_wait
- `VulkanContext::allocate_command_buffer()` — primary command buffer allocation
- `VulkanContext::execute_immediate()` — record + submit + wait + cleanup in one call
- `VulkanContext::copy_buffer()` — synchronous buffer-to-buffer copy with validation
- `VulkanContext::upload_to_device_local()` — end-to-end CPU→GPU upload via staging buffer
- Transfer validation: TRANSFER_SRC/DST usage flags, DEVICE_LOCAL target check, size bounds
- 22 unit tests for struct construction, error display, and validation logic

### 2.3.3 Shader module loading

- Add `.spv` binary loading from `shaders/` directory
- Create `vkShaderModule` from byte slice
- Add `ShaderModule` wrapper type with `Drop` to destroy the module
- No shader compilation at runtime — `.spv` files are prebuilt

### 2.3.4 Compute pipeline creation

- Build compute pipeline from shader module: `vkPipelineLayout` → `vkComputePipeline`
- Add `ComputePipeline` wrapper type
- Pipeline layout starts simple: no descriptor sets yet, just the shader module and entry point
- Will need descriptor set layouts once kernels require buffer bindings

### 2.3.5 Command buffer cycle

- Allocate secondary or primary command buffer from `CommandResources`
- Record: `beginCommandBuffer` → `bindPipeline` → `dispatch` → `endCommandBuffer`
- Submit via `vkQueueSubmit` on the compute queue
- Wait via fence or `vkQueueWaitIdle`
- Add `CommandBuffer` wrapper with proper lifecycle

### 2.3.6 No-op kernel smoke test

- Write a trivial compute shader: `no_op.comp.glsl` — writes a single value to a storage buffer
- Compile to SPIR-V with `glslangValidator`
- End-to-end: allocate buffer → upload input → dispatch no-op kernel → read back → verify output matches expected value
- This is the first end-to-end GPU execution test

## Phase 3: First kernel — embedding lookup

After Phase 2.3, the first real kernel is the embedding lookup:

1. Write `embedding_lookup.comp.glsl` — given token indices and embedding table, output hidden state vectors
2. Memory-map GGUF file, upload embedding tensor to device-local buffer
3. Dispatch kernel for single token
4. Read back result, compare against CPU reference computation
5. Verify numerical correctness against llama.cpp output for same model/token

## Phase 4: Transformer block

Build the kernel chain for one transformer block:

1. RMSNorm kernel
2. QKV projection + RoPE rotation kernel
3. Attention kernel with KV cache write
4. FFN (SiLU-gated) kernel
5. Wire them together in runtime for single-block forward pass

## Risks and challenges

- **Vulkan shader complexity.** Compute shaders for attention and FFN are substantially more complex than CPU code. Expect iteration on workgroup sizing, shared memory usage, and memory access patterns.
- **AMD driver differences.** RADV (Mesa) and AMDVLK may behave differently. Test on the target driver early.
- **Quantization kernels.** Dequantizing Q4_0/Q8_0 in shaders adds significant complexity. May need to start with FP16/FP32 models.
- **Memory pressure.** Large models need efficient weight loading. Memory-mapping + single-upload is the plan, but VRAM-constrained systems will need attention.
- **Numerical correctness.** Floating-point differences between CPU and GPU are expected. Need tolerance-based comparison, not exact equality.
- **SPIR-V toolchain.** `glslangValidator` must be available on the build machine. Consider CI implications.
- **ash API surface.** The `ash` crate is thin — all Vulkan boilerplate (descriptor sets, pipeline layouts, synchronization) is manual. This is intentional but increases implementation surface.

## What not to do yet

- Do not implement the full forward pass before the no-op kernel works
- Do not implement server mode before `vulkanize generate` produces text
- Do not optimize kernels before they produce correct results
- Do not add multi-GPU or tensor parallelism before single-GPU is working
- Do not write benchmarks before numerical correctness is verified
