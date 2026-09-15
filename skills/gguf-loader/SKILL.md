# GGUF Loader — Skill

## Purpose

Guidance for implementing the GGUF file parser in `crates/gguf`.

## When to use this skill

- Implementing or debugging GGUF header parsing
- Reading tensor descriptors and metadata
- Handling quantized weight formats (Q4_0, Q5_0, Q8_0, etc.)
- Memory-mapping the file for zero-copy GPU access

## Key references

- `docs/gguf-format-notes.md` — format specification, gotchas, type table
- `crates/gguf/src/lib.rs` — parser implementation
- `llama.cpp/gguf.h` (external) — reference header for field names and types

## Parsing checklist

- [ ] Read magic bytes: must be `GGUF` (0x46554747)
- [ ] Read version field; currently accept GGUF v3 only (little-endian)
- [ ] Walk KV pairs into a typed map
- [ ] Walk tensor headers into a descriptor list
- [ ] Validate 32-byte alignment before tensor data region
- [ ] Memory-map the file for GPU buffer initialization (planned; current parser uses file I/O)

## Type handling

For quantized types, the parser must expose:
- Block size (e.g., 32 for Q4_0)
- Block byte size (e.g., 64 for Q4_0)
- Per-tensor total bytes = `(elements / block_size) * block_byte_size`

The parser does NOT dequantize — that happens in GPU shaders.

## Common pitfalls

- Tensor `offset` is relative to the aligned tensor-data section; resolve the file position as `data_start + offset`
- `stride` is in elements, convert to bytes: `stride[i] * type_element_size`
- GGUF string lengths use a u64 length prefix, with no null terminator
- Not all models have an `output.weight` tensor — check for weight tying
