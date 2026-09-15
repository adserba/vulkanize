# Server Mode — Planned OpenAI-Compatible API

> Server mode is planned architecture. `vulkanize serve` is currently a CLI stub and does not start an HTTP server.

## Overview

`vulkanize serve` is intended to expose an HTTP server compatible with the OpenAI API, allowing OpenAI clients to use Vulkanize as a local inference backend once text generation and server mode are implemented.

## Planned endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat completion (streaming and non-streaming) |
| `POST` | `/v1/completions` | Legacy completion endpoint |

## Planned request/response types

### ChatCompletionRequest

```json
{
  "model": "local",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 256,
  "stream": false
}
```

### ChatCompletionResponse (non-streaming)

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "local",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello!"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 2,
    "total_tokens": 7
  }
}
```

### Streaming response (SSE)

Each chunk is an SSE event:
```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"He"}}]}
```

Final chunk:
```
data: {"choices":[{"finish_reason":"stop"}]}
data: [DONE]
```

## Architecture

The planned server is a thin HTTP layer over `vulkanize-runtime`:

```
HTTP request → api crate validates + deserializes → runtime runs inference →
api crate serializes response → HTTP response / SSE stream
```

### Concurrency model

- Single-model, single-request-at-a-time initially (simplifies KV cache management)
- Future: queue-based request handling with context isolation per conversation
- Tokio async runtime for I/O; Vulkan operations are synchronous from the CPU perspective

## Implementation plan

Current status: `crates/api` is a placeholder crate and `vulkanize serve` exits with `not yet implemented`.

1. **Crate**: `vulkanize-api` provides types and handlers
2. **Framework**: `axum` (preferred) or `warp` — both work well with tokio
3. **Model lifecycle**: load model at startup, hold in memory for server lifetime
4. **Health check**: `GET /health` returns OK once model is loaded

## Deferred features

- [ ] Multi-model hot-swapping
- [ ] Request queuing with priority
- [ ] Token usage tracking per request
- [ ] System prompt configuration via API
- [ ] Authentication/API key support
- [ ] WebSocket alternative for streaming
