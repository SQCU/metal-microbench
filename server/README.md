# gemma-metal-bridge

Python FastAPI bridge that exposes the Swift Metal engine (`libgemma_metal.dylib`)
as an OpenAI-compatible HTTP server. Every browser-side call is a plain HTTP
endpoint — curl-testable in isolation.

## architecture

```
  browser (fetch + SSE)  ─┐
                          ├──► FastAPI bridge (port 8001) ──► libgemma_metal.dylib ──► Metal GPU
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

# Canonical, config-driven launch (reads server/config.toml; default port 8001):
uv run python serve.py

# Or the env-override path:
GEMMA_PORT=8002 uv run python serve.py

# Direct uvicorn (legacy form; bypasses config-toml resolution):
GGUF_PATH=/path/to/model.gguf uv run uvicorn bridge:app --port 8001
```

First call takes ~10 s (weight load + first-CB compile). Subsequent requests
are instant.

## endpoints

Every web-app call maps to one of these. Each has a curl recipe.

### `GET /health`

Liveness + runtime stats.

```bash
curl http://localhost:8001/health
# {"status":"ready","model":"gemma-4-a4b","active_sessions":0,"pump_running":true}
```

### `GET /v1/models`

OpenAI-shaped model list.

```bash
curl http://localhost:8001/v1/models
```

### `POST /v1/chat/completions`

Non-streaming:

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-a4b",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "max_tokens": 24
  }'
```

Streaming (SSE):

```bash
curl -N -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a haiku."}],"max_tokens":30,"stream":true}'
# data: {"id":"chatcmpl-…","object":"chat.completion.chunk",...}
# data: …
# data: [DONE]
```

Two concurrent requests ride the B=4 batched scheduler:

```bash
(curl -sS -X POST http://localhost:8001/v1/chat/completions \
   -H 'Content-Type: application/json' \
   -d '{"messages":[{"role":"user","content":"Count to 5."}],"max_tokens":30}' &
 curl -sS -X POST http://localhost:8001/v1/chat/completions \
   -H 'Content-Type: application/json' \
   -d '{"messages":[{"role":"user","content":"List colors."}],"max_tokens":30}' &
 wait)
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
curl -sS -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/body.json
```

The bridge decodes the data URI and runs the vision tower via
`gemma_submit_image_bytes` (image bytes pass straight through — no
filesystem round-trip), brackets the soft tokens with BOI/EOI
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
curl http://localhost:8001/v1/cache/stats
# {"entries":1,"hits":7,"misses":1,"bytes":3153920,"hit_rate":0.875}
```

Flush:

```bash
curl -X POST http://localhost:8001/v1/cache/clear
# {"evicted":1}
```

Pre-populate (skip the first-request TTFT hit for an image you know
you'll reference soon):

```bash
curl -X POST http://localhost:8001/v1/images/prewarm \
  -H 'Content-Type: application/json' \
  -d '{"image_url": {"url": "data:image/png;base64,..."}}'
# {"soft_tokens":280,"cache_key":"<sha256>","elapsed_ms":7073,"stats":{...}}
```

Measured win on M5 (amongus frame):
- cold chat (miss): **13.2 s**
- warm chat (hit):  **5.7 s**  — same image, second request

### `GET /`  — the tetraplex demo

Four streams, one GPU. The web page at the root is a live demonstration of
everything the engine does: multi-slot batched decode, prefix cache sharing,
multimodal input, per-session KV tenancy.

Layout:

```
  ┌───────────────────────────── header ─────────────────────────────┐
  │ [ launch all 4 ]  fires every pane at once                        │
  ├──────────────────────┬───────────────────────────────────────────┤
  │ stream A             │ stream B                                   │
  │  textarea + optional │                                            │
  │  image upload        │                                            │
  │  ask button          │                                            │
  │  (streaming output)  │                                            │
  │  [kv pages strip]    │   ← colored cells: green=private,          │
  ├──────────────────────┼───────────────────────────────────────────┤      orange=shared with 1 other,
  │ stream C             │ stream D                                   │       red=shared with ≥2 others
  │                      │                                            │
  ├──────────────────────┴───────────────────────────────────────────┤
  │  bandwidth chart (4 lines + aggregate, 30-second rolling window)  │
  │  A: tok/s · B: tok/s · C: tok/s · D: tok/s · Σ: total             │
  └───────────────────────────────────────────────────────────────────┘
```

**What to look for during a recording:**

1. *Cold-start burst*: hit **launch all 4** with four different short prompts. All
   four bandwidth lines sit at zero for ~0.5 s (multi-slot prefill), then the
   aggregate rockets to ~120 tok/s while each individual stream settles around
   30 tok/s. The Σ counter stays pinned near peak until the first stream finishes.

2. *Prefix sharing*: give A and C the same system prompt (e.g. *"You are a concise
   assistant."*). After A's prefill completes, the content-hash cache has its pages
   indexed — C's submit probes, finds the hit, and adopts A's filled pages read-only.
   A's kv-strip and C's kv-strip now both have orange cells at the same `phys` ids.
   Hover a cell for the exact sharing tuple.

3. *Multimodal*: attach an image to any pane, ask something about it. Vision tower
   runs once on first submission (~7 s TTFT on M5); subsequent identical images hit
   the SHA-256 soft-tokens cache and skip vision entirely (~5 s saved per repeat).

4. *Introspection moment*: attach a screenshot of the running demo to one pane and
   ask "what is happening in this image?". Gemma-4 will describe your dashboard
   (the four streams, the telemetry fields, the fact that two panes are duplicates
   if you set them that way) — a small but surprisingly sharp example of the
   model reading its own interface.

**Curl-equivalence**: every button in the demo hits an endpoint documented above.
`launch all 4` is four parallel `POST /v1/chat/completions` with `stream:true`. The
kv tenancy strip polls `GET /v1/kv/snapshot` every 300 ms. The bandwidth chart is
pure client-side (counts SSE delta arrivals over a 500 ms sliding window).

```bash
open http://localhost:8001/    # macOS
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
| `GEMMA_MODEL_NAME` | `gemma-4-a4b` | model id returned by `/v1/models` |
| `GEMMA_HOST` | `0.0.0.0` (config.toml [server].host) | bind address |
| `GEMMA_PORT` | `8001` (config.toml [server].port) | listen port |
| `GEMMA_LOG_LEVEL` | `warning` | uvicorn log level |
| `BRIDGE_URL` | (resolved from config) | client-side override (clients only) |
| `QUANT_BRIDGE_URL` | (legacy alias) | quant_search harnesses |
| `GEMMA_BRIDGE` | (legacy alias) | scringlo / older harnesses |

**Config-driven defaults** live in `server/config.toml`. Env vars
override; client-side scripts read the resolved URL via
`server/bridge_config.py` so changing one line in config.toml flips
every harness in lockstep.

## driving the engine without the HTTP bridge

The Swift dylib also speaks a direct ctypes ABI (`gemma_ffi.py`) — no
FastAPI / asyncio involvement. The probe scripts and offline tests
use this path:

```bash
# stdlib-only probes (no third-party deps; either python form works,
# but uv-managed is canonical for consistency with the rest of the project):
cd server
uv run python test_batch_ffi.py             # smoke test
uv run python probe_oversubscription.py     # M:K throughput sweep
uv run python probe_sustained_eval.py       # 5-min sustained eval shape

# The probes import gemma_ffi (a thin ctypes wrapper around the dylib),
# which expects libgemma_metal.dylib to be discoverable — built once via
# `make libgemma_metal.dylib` from the repo root.
```
