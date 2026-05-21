# Vulkan Kernel Development — Skill

## Purpose

Guidance for writing, compiling, and debugging Vulkan compute shaders in GLSL.

## When to use this skill

- Writing new compute shaders for transformer operations
- Debugging shader compilation or dispatch errors
- Optimizing kernel performance (occupancy, memory access patterns)
- Setting up the shader build pipeline

## Shader file convention

- Source: `shaders/name.comp.glsl`
- Binary: `shaders/name.spv`
- Both committed to git

## Compile command

```bash
glslangValidator -V shaders/name.comp.glsl -o shaders/name.spv
```

Requires `glslang-tools` package (Debian/Ubuntu) or `glslang` (Arch).

## Shader template

```glsl
#version 450 core

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer InputBuf {
    float input[];
};

layout(set = 0, binding = 1) buffer OutputBuf {
    float output[];
};

layout(push_constant) uniform PushConstants {
    uint dim;
    uint batch;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.dim * pc.batch) return;
    // ... kernel logic
}
```

## Dispatch calculation

```rust
let total_invocations = dim * batch;
let workgroup_size = 64u32;
let num_workgroups = (total_invocations + workgroup_size - 1) / workgroup_size;
vkCmdDispatch(cmd_buf, num_workgroups, 1, 1);
```

## Descriptor set layout

- **Set 0**: All storage buffers (weights, activations, KV cache)
- **Push constants**: Per-dispatch configuration (dimensions, indices, scales)
- Keep binding numbers stable; document in shader source comments

## Performance tips

- Use `shared` memory sparingly — prefer registers and coalesced global access
- Avoid division in hot loops; precompute reciprocals on CPU or push constant
- Use `gl_WorkGroupID` × `local_size_x` + `gl_LocalInvocationID` for explicit indexing when `gl_GlobalInvocationID` is insufficient
- For matrix multiply: tile into shared memory, but benchmark vs naively vectorized loads first

## Testing approach

1. Implement kernel with known dimensions (e.g., 64×64)
2. Feed known input values (sequential integers or all-ones)
3. Read back output and compare against CPU reference in a unit test
4. Only scale to real model dimensions after small-case verification passes
