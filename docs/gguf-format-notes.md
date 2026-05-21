# GGUF Format Notes

## Overview

GGUF (GPT-Generated Unified Format) is the successor to GGML. It stores model weights and metadata in a single file with a self-describing header. Vulkanize parses GGUF directly — no external library.

## File layout

```
┌─────────────────┐
│  Header          │
│  - magic: "GGUF" │
│  - version: u32  │
│  - tensor count  │
│  - kv count      │
├─────────────────┤
│  Key-Value pairs │  (kv count entries)
│  - key (str)     │
│  - type (u32)    │
│  - value         │
├─────────────────┤
│  Tensor headers  │  (tensor count entries)
│  - name (str)    │
│  - n_dims (u32)  │
│  - shape (dims)  │
│  - stride (dims) │
│  - type (u32)    │
│  - offset (u64)  │
├─────────────────┤
│  Padding         │  (align to 32 bytes)
├─────────────────┤
│  Tensor data     │  (raw weight bytes)
└─────────────────┘
```

## Version awareness

| Version | Notes |
|---|---|
| 2 | Legacy; most models now use v3+ |
| 3 | Current standard; u64 tensor offsets, stride in headers |
| 4 | Future-compatible; watch for new fields |

Parse version from header and handle differences gracefully.

## Data types (GGML type enum)

Key types we must support:

| Type ID | Name | Bytes/element | Notes |
|---|---|---|---|
| 0 | F32 | 4 | Full precision, easy to start with |
| 1 | F16 | 2 | Common in GGUF, needs dequant or native FP16 shader support |
| 2 | Q4_0 | block | 32 weights/block, 64 bytes total |
| 7 | Q8_0 | block | 32 weights/block, simple scale |
| 10 | Q5_0 | block | 32 weights/block, 5-bit + aux bits |

Quantized types store data in blocks. Each block has:
- A scaling factor (and possibly min for asymmetric)
- Compressed weight bytes (4-bit nibbles for Q4/Q5)
- Auxiliary bytes for Q5 (5th bit per weight)

Shader kernels must dequantize inline or load pre-dequantized buffers.

## Architecture metadata keys

Keys are architecture-specific. Common patterns:

| Key | Meaning |
|---|---|
| `general.architecture` | e.g., "llama", "gemma2", "qwen2" |
| `general.name` | Human-readable model name |
| `{arch}.block_count` | Number of transformer layers |
| `{arch}.context_length` | Max sequence length |
| `{arch}.embedding_length` | Hidden dimension size |
| `{arch}.attention.head_count` | GQA/MHA heads |
| `{arch}.attention.head_count_kv` | KV heads (GQA) |
| `{arch}.feed_forward_length` | FFN intermediate dim |
| `{arch}.rope.freq_base` | RoPE base frequency |

## Tensor naming patterns

Llama-family models use a consistent pattern:

```
token_embd.weight                       — embedding table
blk.{i}.attn_norm.weight                — pre-attention norm
blk.{i}.ffn_norm.weight                 — pre-FFN norm
blk.{i}.feed_forward.{gate,down,up}.weight  — FFN projections
blk.{i}.attn.{q,k,v,wo}.weight          — attention projections
output_norm.weight                      — final layer norm
output.weight                           — LM head (may be tied to token_embd)
```

Not all models have `output.weight` — some tie weights to the embedding matrix.

## Parsing strategy for Vulkanize

1. Memory-map the file with `std::fs::File` + `memmap2`
2. Read header 4 fields from offset 0
3. Walk KV pairs, build a `HashMap<String, GGUFValue>`
4. Walk tensor headers, build `Vec<TensorDescriptor>`
5. Tensor data is accessed via file offset + mmap pointer — never copy to CPU heap
6. Pass tensor descriptors + mmap handle to runtime for GPU buffer creation

## Gotchas

- **Padding**: tensor data region starts at a 32-byte aligned offset after all headers. The header's last byte position + pad → tensor data start.
- **Stride vs shape**: stride is in elements, not bytes. Compute byte size as `stride * type_size`.
- **Offset is from file start**: tensor `offset` field is absolute file offset, not relative to data region.
- **String table**: GGUF strings are UTF-8 with a u32 length prefix (no null terminator).
- **Big-endian on v2**: GGUF v2 used big-endian for some fields; v3+ is little-endian always.

## Reference implementations

- `llama.cpp/gguf.h` — authoritative C header
- `llama.cpp/gguf-reader.cpp` — reference parser
- These are for understanding format details only; do NOT vendor this code.
