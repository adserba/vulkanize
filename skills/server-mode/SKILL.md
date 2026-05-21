# Server Mode — Skill

## Purpose

Guidance for implementing the OpenAI-compatible HTTP API server in `crates/api` and `crates/cli`.

## When to use this skill

- Implementing `/v1/chat/completions` or `/v1/completions` endpoints
- Adding streaming SSE responses
- Integrating the HTTP server with vulkanize-runtime
- Designing request/response types

## Key references

- `docs/server-mode.md` — API specification, endpoint list, JSON schemas
- `crates/api/src/lib.rs` — API types and handlers
- `crates/cli/src/main.rs` — `serve` subcommand entrypoint
- OpenAI API docs: https://platform.openai.com/docs/api-reference

## Implementation steps (in order)

1. Define request/response types matching OpenAI schema
2. Implement non-streaming `/v1/completions` first (simpler response path)
3. Wire up `serve` subcommand to start HTTP server and load model
4. Add streaming SSE for `/v1/chat/completions`
5. Add `GET /v1/models` health check

## Framework choice

- **Axum** (recommended): ergonomic, tokio-native, typed routing
- Alternative: warp — similar async model, slightly more combinator-heavy

## Concurrency constraints

- **Single request at a time** initially — simpler KV cache management
- Block the endpoint when a generation is in progress
- Future: per-request KV cache isolation with queue-based scheduling

## Model lifecycle

```
serve --model path.gguf
  → runtime::Model::load(path)   // one-time, at startup
  → serve HTTP requests          // reuse loaded model
  → SIGINT/SIGTERM               → graceful shutdown, unmap GGUF
```

Do NOT reload the model per request. Load once, serve many.

## Streaming (SSE) details

- Response header: `Content-Type: text/event-stream`
- Each token produces one SSE `data:` line
- Final line: `data: [DONE]`
- Flush after each write to maintain real-time streaming
- Handle client disconnect gracefully (drop the response stream)

## Error handling

- Return HTTP 503 if model is not loaded or GPU unavailable
- Return HTTP 400 for malformed requests
- Log errors to stderr with `RUST_LOG` level control
