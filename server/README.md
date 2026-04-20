# gemma-metal-bridge

Python FastAPI bridge that exposes the Swift Metal engine (`libgemma_metal.dylib`)
as an OpenAI-compatible HTTP server. Every browser-side call is a plain HTTP
endpoint — curl-testable in isolation.

## architecture

```
  browser (fetch + SSE)  ─┐
                          ├──► FastAPI bridge (port 8000) ──► libgemma_metal.dylib ──► Metal GPU
  curl / HTTPie          ─┘       (1 pump thread, asyncio handlers)
```

The Swift dylib is the **only** Swift anyone has to care about — it's a thin
C-ABI (see `ffi.swift`) wrapping `LmEngine`. Python drives.

## build

```bash
# Build the Swift dylib (once).
cd ..                              # repo root
make libgemma_metal.dylib

# Sync the Python venv + run the bridge.
cd server
uv sync
GGUF_PATH=/path/to/model.gguf uv run uvicorn bridge:app --port 8000
```

First call takes ~10 s (weight load + first-CB compile). Subsequent requests
are instant.

## endpoints

Every web-app call maps to one of these. Each has a curl recipe.

### `GET /health`

Liveness + runtime stats.

```bash
curl http://localhost:8000/health
# {"status":"ready","model":"gemma-4-a4b-q4km","active_sessions":0,"pump_running":true}
```

### `GET /v1/models`

OpenAI-shaped model list.

```bash
curl http://localhost:8000/v1/models
```

### `POST /v1/chat/completions`

Non-streaming:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-a4b-q4km",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "max_tokens": 24
  }'
```

Streaming (SSE):

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a haiku."}],"max_tokens":30,"stream":true}'
# data: {"id":"chatcmpl-…","object":"chat.completion.chunk",...}
# data: …
# data: [DONE]
```

Two concurrent requests ride the B=4 batched scheduler:

```bash
(curl -sS -X POST http://localhost:8000/v1/chat/completions \
   -H 'Content-Type: application/json' \
   -d '{"messages":[{"role":"user","content":"Count to 5."}],"max_tokens":30}' &
 curl -sS -X POST http://localhost:8000/v1/chat/completions \
   -H 'Content-Type: application/json' \
   -d '{"messages":[{"role":"user","content":"List colors."}],"max_tokens":30}' &
 wait)
```

### `POST /v1/completions`

Raw-prompt legacy endpoint (skips the chat template render).

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"<|turn>user\nHi<turn|>\n<|turn>model\n","max_tokens":16}'
```

### `GET /`

The side-by-side streaming demo. Two textareas, two response panes, one GPU.

```bash
open http://localhost:8000/    # macOS
```

## python FFI directly (no HTTP)

For scripts that want to drive the engine without going through FastAPI:

```bash
GGUF_PATH=/path/to/model.gguf uv run python scripts/smoke.py
```

`scripts/smoke.py` opens two sessions with different prompts, pumps the engine,
prints both responses. Useful for testing the dylib without HTTP involvement.

## env vars

| variable | default | purpose |
|---|---|---|
| `GEMMA_DYLIB` | auto-discover | path to `libgemma_metal.dylib` |
| `GGUF_PATH` | hard-coded | path to the GGUF weights file |
| `GEMMA_MODEL_NAME` | `gemma-4-a4b-q4km` | model id returned by `/v1/models` |
| `GEMMA_BRIDGE_HOST` | `0.0.0.0` | bind address |
| `GEMMA_BRIDGE_PORT` | `8000` | listen port |
| `GEMMA_BRIDGE_LOG` | `info` | uvicorn log level |
