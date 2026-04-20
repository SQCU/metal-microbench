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

### multimodal (image_url)

Pass OpenAI-style image content to `/v1/chat/completions`:

```bash
python3 -c "
import base64, json, sys
img = base64.b64encode(open('path/to/image.png','rb').read()).decode()
print(json.dumps({
  'messages': [{
    'role': 'user',
    'content': [
      {'type': 'text', 'text': 'Describe this image in one sentence.'},
      {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{img}'}}
    ]
  }],
  'max_tokens': 32
}))" > /tmp/body.json
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/body.json
```

The bridge decodes the data URI, writes a tempfile, runs the vision tower
via `gemma_submit_image_path`, brackets the soft tokens with BOI/EOI
(255999/258882) markers, and the session generates as usual. Vision adds
~7 s TTFT on M5 (one-time per request; no caching yet).

Multimodal status is reported in `/health`:

```json
{"status":"ready", "multimodal":true, "vision_cache":{"entries":0,"hits":0,"misses":0,"bytes":0}}
```

### vision soft-tokens cache

Repeat submissions of the same image byte-for-byte skip the vision tower
entirely. The cache is keyed by SHA-256 of the raw image bytes and stores
the already-padded (280-row fp32) soft-tokens MTLBuffer; LRU-evicts at
64 entries (~200 MB).

Inspect:

```bash
curl http://localhost:8000/v1/cache/stats
# {"entries":1,"hits":7,"misses":1,"bytes":3153920,"hit_rate":0.875}
```

Flush:

```bash
curl -X POST http://localhost:8000/v1/cache/clear
# {"evicted":1}
```

Pre-populate (skip the first-request TTFT hit for an image you know
you'll reference soon):

```bash
curl -X POST http://localhost:8000/v1/images/prewarm \
  -H 'Content-Type: application/json' \
  -d '{"image_url": {"url": "data:image/png;base64,..."}}'
# {"soft_tokens":280,"cache_key":"<sha256>","elapsed_ms":7073,"stats":{...}}
```

Measured win on M5 (amongus frame):
- cold chat (miss): **13.2 s**
- warm chat (hit):  **5.7 s**  — same image, second request

### `GET /`

The side-by-side streaming demo. Two textareas, two response panes, a file
picker per pane for optional image attachment, one GPU.

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
