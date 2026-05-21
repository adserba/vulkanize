# Inference Runtime Design — Skill

## Purpose

Guidance for implementing the inference orchestration layer in `crates/runtime`.

## When to use this skill

- Implementing the forward pass scheduler
- Managing KV cache allocation and growth
- Integrating tokenizer and sampler with GPU compute
- Debugging end-to-end generation quality

## Key references

- `docs/design.md` — crate contracts, data flow diagram
- `crates/runtime/src/lib.rs` — runtime implementation
- `crates/vulkan-backend/` — GPU dispatch interface
- `crates/gguf/` — model metadata and tensor descriptors

## Runtime responsibilities

1. **Model loading**: parse GGUF → create GPU buffers for each tensor → upload weights
2. **Inference loop**: token IDs → embedding → transformer blocks → logits → sample → repeat
3. **KV cache management**: allocate, track position, handle context overflow
4. **Sampling**: greedy / temperature / top-k / top-p on CPU (readback logits from GPU)

## Forward pass data flow

```
token_ids[N]
  → embedding_lookup → hidden[N][d_model]
  → for each block i:
      rms_norm(hidden) → normalized
      qkv_proj(normalized) → Q, K, V
      apply_rope(Q, K, positions) → Q_rot, K_rot
      attention(Q_rot, K_rot, V, kv_cache[i]) → attn_out
      ffn_gate(norm(hidden)) → gate, up
      ffn_down(gate * up) → ffn_out
      residual = hidden + ffn_out
  → final_rms_norm(residual)
  → lm_head → logits[N][vocab_size]
  → readback to CPU for sampling
```

## KV cache layout

Per-layer:
```
keys[layer][seq_pos][kv_heads][head_dim]
values[layer][seq_pos][kv_heads][head_dim]
```

- Pre-allocate for max context length from GGUF metadata
- Increment `seq_pos` each generation step
- No eviction initially; future: sliding window or PagedAttention-style management

## Sampling interface

The sampler runs on CPU after logits are read back:

```rust
trait Sampler {
    fn sample(&self, logits: &[f32]) -> u32;  // returns token ID
}
```

Implementations:
- `GreedySampler` — argmax
- `TemperatureSampler` — softmax( logits / temp ), categorical sample
- `TopKSampler` — mask all but top-k, then temperature
- `TopPSampler` (nucleus) — cumulative probability threshold

## Concurrency model

- Single-threaded inference loop initially
- GPU operations are submitted and waited synchronously
- Future: async dispatch with timeline semaphores for overlapping compute and I/O

## Testing strategy

- Compare logits output against llama.cpp on same input tokens and model
- Use a small model (Qwen2.5-0.5B) for fast iteration
- Fix random seeds in all sampling tests for determinism
