# AMD Vulkan GGUF Inference — Skill

## Purpose

Specialized guidance for implementing AMD-targeted Vulkan compute kernels for transformer inference on GGUF models.

## When to use this skill

- Writing or debugging Vulkan compute shaders for transformer operations
- Optimizing kernel performance on AMD RDNA GPUs
- Designing memory layouts for GPU tensor storage
- Debugging Vulkan pipeline issues specific to AMD drivers

## Key references

- `docs/amd-vulkan-backend.md` — device selection, memory types, queue families
- `docs/design.md` — crate contracts, data flow
- `crates/vulkan-backend/` — Vulkan initialization and pipeline management
- `shaders/` — GLSL compute shader sources and compiled `.spv` files

## AMD-specific constraints

1. **Wavefront size is 64**: workgroup sizes should be multiples of 64 (64, 128, 256)
2. **No shared memory tiling unless necessary**: RDNA local memory is limited; prefer register-heavy kernels
3. **Memory coalescing matters**: global memory access patterns must be contiguous across wavefront
4. **Divergence penalty**: avoid `if/else` on non-uniform predicates; use select() or predication

## Kernel development workflow

1. Write shader in `shaders/name.comp.glsl`
2. Compile to SPIR-V: `glslangValidator -V shaders/name.comp.glsl -o shaders/name.spv`
3. Load `.spv` in vulkan-backend, create compute pipeline
4. Dispatch from runtime with correct workgroup dimensions
5. Validate output against CPU reference computation

## Common transformer kernels (in development order)

1. **Embedding lookup** — index → hidden state vector
2. **RMSNorm** — layer normalization variant
3. **QKV projection + RoPE** — linear transform + rotary position embedding
4. **Attention** — scaled dot-product with KV cache write
5. **FFN** — SiLU-gated feed-forward network (gate × up → down)
6. **LM head** — final projection to vocabulary logits

## Debugging checklist

- [ ] Validation layers enabled (`VK_LAYER_KHRONOS_validation`)
- [ ] Workgroup size is multiple of 64
- [ ] Descriptor set bindings match pipeline layout
- [ ] Memory barriers between copy and compute passes
- [ ] Staging buffer has `HOST_COHERENT` for readback
- [ ] Pipeline barrier with correct src/dst stage masks
