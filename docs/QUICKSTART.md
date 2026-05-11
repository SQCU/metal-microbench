# Quickstart

Apple Silicon (M-series) Mac required; tested on M5 Max with 128 GB unified memory. Xcode command-line tools give you the Swift compiler; [uv](https://docs.astral.sh/uv/) manages the Python bridge's dependencies.

```bash
# 1. Build the Metal inference shared library (kernels.swift + engine + FFI)
make libgemma_metal.dylib

# 2. Install bridge deps (uv creates a venv automatically)
cd server && uv sync

# 3. Fetch the Gemma-4 weights (GGUF + bf16 safetensors) into ./models/
uv run python scripts/fetch-weights.py

# 4. Launch — reads server/config.toml, runs uvicorn on port 8001 by default
uv run python serve.py
```

Open <http://localhost:8001> for the client index; pick one of the five demo clients or drive the REST API directly.

## Configuration

All paths and server settings live in **`server/config.toml`**. Edit it to point at different model files, change the port, or override the HuggingFace repos that `fetch-weights.py` pulls from:

```toml
[model]
gguf_path = "models/gemma-4/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
safetensors_path = "models/gemma-4-bf16/model-00001-of-00002.safetensors"

[server]
host = "0.0.0.0"
port = 8001
log_level = "warning"

[fetch]
gguf_repo = "unsloth/gemma-3-27b-it-GGUF"
gguf_filename = "gemma-3-27b-it-UD-Q4_K_M.gguf"
safetensors_repo = "google/gemma-3-27b-it"
# ...
```

Env-var overrides take precedence: `GEMMA_GGUF`, `GEMMA_SAFETENSORS`, `GEMMA_HOST`, `GEMMA_PORT`, `GEMMA_LOG_LEVEL`, `GEMMA_MODEL_NAME`.

Gemma is a gated HuggingFace repo — run `huggingface-cli login` before `fetch-weights.py` if you haven't already accepted its license.

---

## Clients

The bridge exposes one OpenAI-compatible REST surface
(`/v1/chat/completions`, `/v1/models`, `/v1/tokenize`, `/health`) plus
one engine-telemetry endpoint (`/v1/engine/state`) for clients that
visualize engine internals. Two kinds of clients consume it:

**SillyTavern fork** (chat-completion-shaped product testing) — we
develop against and test through a fork of
[SillyTavern](https://github.com/SillyTavern/SillyTavern). See
`tools/st-debug/` for the debug harness, settings seed, and Playwright
suite that drives it. For anything ST already does well (chat,
multi-turn, swipes, persona+toolcards, multimodal upload), drive ST
rather than reimplementing it — re-rendering ST inside a test page is
equivalent to maintaining a parallel ST fork.

**Bespoke engine-internals visualizers** (`server/static/`) for
behavior ST does not and will not render. Mounted at `/static/`; the
root `/` redirects to `clients.html` (the index).

| client | what it shows | engine surface it exercises |
| --- | --- | --- |
| `tetraplex` (`index.html`) | four synchronized streams, bandwidth timeseries, KV-cache tenancy strip | batched AR decode at B=4 + content-hash prefix cache |
| `labeler` | drag a folder of PNGs, watch 4 live-decoding ribbons drain a queue with vision-cache stats | vision cache + shared-schema prefix + multi-image queue |
| `loom` | tree-of-thoughts with SVG-woven branches forked from a shared root | content-hash prefix cache + fork-from-node |
| `steering` | preview UI: 4 parallel chat lanes + live KV tenancy. Per-token cvec heatmap deferred until the engine extends `StreamUpdateOut` with activation deltas. | batched AR decode (placeholder for future cvec write/read pipeline) |

All LLM work flows through `/v1/chat/completions`. Per-completion
telemetry (token counts, KV cache hits/misses, vision cache hits)
comes from the OAI `usage` block. Engine-wide instantaneous state
(global page tenancy, vision cache size, active-stream registry)
comes from `/v1/engine/state`, polled at 1–2 Hz — see
`docs/engine_telemetry_endpoint.md` for the payload shape rationale
and cost characteristics.

---

## REST API surface

OpenAI-compatible for the chat/completion endpoints; one engine
telemetry endpoint (`/v1/engine/state`) covers the global state slice
the `usage` block can't carry.

| method | path | purpose |
| --- | --- | --- |
| `GET` | `/health` | engine readiness, model name, aggregate counters |
| `POST` | `/v1/chat/completions` | OpenAI-compat chat; accepts `image_url` content items; `usage.cache_hits`, `usage.cache_misses`, `usage.vision_cache_hits` extension fields surface per-completion telemetry |
| `POST` | `/v1/tokenize` | stateless tokenizer |
| `GET` | `/v1/models` | OpenAI-shape model list |
| `GET` | `/v1/engine/state` | JSON snapshot of KV-cache page tenancy, vision cache, active-stream registry. Polling-friendly. See `docs/engine_telemetry_endpoint.md`. |
| `GET` | `/static/*` | visualizers + demo assets (the index is `/static/clients.html`; `/` redirects there) |

---

## Key numbers on M5 Max

| workload | throughput | notes |
| --- | --- | --- |
| LM AR decode, 4 concurrent text streams | **3.6× speedup** vs single stream | weight-load-amortized via paged attention at B=4 |
| Vision tower (2520 patches, fp32 intermediates) | **880 ms/image** | 16×16 MMA tiles + fused FFN, bf16 trajectory preserved |
| Vision GEMM effective throughput | **1.66 TFLOPS** | ~55% of realistic fp32-accum ceiling on this hardware |
| 8-image labeling at 4-concurrent | **30 s** end-to-end | down from 53 s naïve; scheduler coalesces by work-type |

---

## Repo map

```
.                             top-level Metal + Swift engine
├── kernels.swift             all MSL kernels (attention, GEMM, MoE, vision)
├── lm_engine.swift           multi-session scheduler + tick() loop
├── vision_tower.swift        27-layer vision encoder (batched + async)
├── vision_residency.swift    3-state memory pressure handling
├── paged_attention.swift     page table + content-hash prefix cache
├── ffi.swift                 C-ABI surface for the Python bridge
│
├── server/
│   ├── bridge.py             FastAPI bridge, OpenAI-compat surface
│   ├── gemma_ffi.py          ctypes wrapper over libgemma_metal.dylib
│   └── static/               engine-internals visualizers (tetraplex,
│                             labeler, loom, steering) — see Clients
│                             section; consume the bridge as web demos
│                             distinct from chat-completion clients
│
├── tools/st-debug/           SillyTavern-fork harness (chat-completion
│                             client) + bootstrap.sh, run.sh,
│                             Playwright tests/
│
├── test_data/                reference tensors + test frames
└── recordings/               captured demo videos
```
