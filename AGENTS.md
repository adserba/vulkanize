# AGENTS.md — Vulkanize

## Quick start

```bash
cargo build --release    # build all crates
cargo test               # run all tests
cargo bench              # run benchmarks (requires --bench flag on individual bins)
```

## Project identity

- AMD-first GGUF inference runtime on Vulkan compute shaders
- NOT a llama.cpp wrapper — independent implementation; llama.cpp is reference and benchmark only
- No CUDA, no ROCm/HIP, no llama.cpp runtime dependency
- CPU is allowed for control flow, parsing, tokenization, sampling, testing, debugging
- All tensor math must execute on GPU via Vulkan compute

## Crate boundaries

| Crate | Responsibility | Must NOT |
|---|---|---|
| `crates/gguf` | Parse GGUF files, expose metadata + tensor descriptors | Execute any computation |
| `crates/vulkan-backend` | Vulkan init, buffers, pipelines, shader compilation/SPV loading | Know about GGUF format or inference algorithms |
| `crates/runtime` | Orchestrate forward pass, KV cache management, sampling loop | Contain GPU driver code or HTTP server code |
| `crates/api` | OpenAI-compatible API types and request/response handlers | Contain Vulkan or GGUF code directly |
| `crates/cli` | Binary entrypoint; argument parsing; dispatch to runtime | Duplicate runtime logic |

## CLI commands (design targets)

```
vulkanize inspect model.gguf              # dump model metadata
vulkanize generate model.gguf [prompt]    # CLI inference
vulkanize serve --host 127.0.0.1 --port 8000  # OpenAI API server
```

`cli` and `serve` must share the same `runtime` crate — no duplicated inference paths.

## Shaders

- Live in `shaders/`, written in GLSL, compiled to SPIR-V
- Compiled at build time or loaded as precompiled `.spv` binaries
- No runtime shader compilation (avoid `shaderc` dependency if possible; use pre-built SPV)

## Testing strategy

- Unit tests inline with crate code (`#[cfg(test)]`)
- Integration tests in `tests/` against real GGUF models
- Benchmarks in `benches/` — compare output and throughput vs llama.cpp baseline
- No flaky容忍: all tests must be deterministic or use fixed seeds

## Commit discipline

- Small, reviewable commits. One logical change per commit.
- Never mix shader changes with Rust crate changes unless they are tightly coupled.
- Reference docs/ in commit messages when a decision is documented there.

## Docs to check before working

| Doc | When to read |
|---|---|
| `docs/current-status.md` | First — what is built, what is not |
| `docs/project-vision.md` | Project goals, constraints, non-goals |
| `docs/design.md` | Architectural decisions, crate contracts |
| `docs/roadmap.md` | Phase order, what's in scope next |
| `docs/next-steps.md` | Immediate milestones, risks, workflow |
| `docs/phase-3-plan.md` | Phase 3 results, bugs, validated patterns |
| `docs/gguf-format-notes.md` | GGUF parsing details, tensor layout quirks |
| `docs/amd-vulkan-backend.md` | Vulkan device selection, queue families, memory types on AMD |
| `docs/server-mode.md` | OpenAI API compatibility requirements |

## Session workflow

- **One session = one narrow milestone.** Do not attempt multiple phases or unrelated tasks in a single session.
- **Use `docs/` as the primary source of truth.** Avoid re-reading large `lib.rs` files unless the implementation requires exact API signatures.
- **Update docs after meaningful milestones.** When a test passes or a feature lands, update `docs/current-status.md`, `docs/roadmap.md`, and the relevant phase plan before committing.
- **Commit after each completed milestone.** Small, reviewable commits. Reference `docs/` in commit messages when a decision is documented there.

## Things to never do

- Add CUDA, HIP, or ROCm dependencies
- Vendor llama.cpp or pull it in as a crate
- Put GPU compute logic in `cli` or `api` crates
- Skip the `vulkan-backend` abstraction and call Vulkan from `runtime` directly
- Generate documentation files other than what's already in `docs/`
