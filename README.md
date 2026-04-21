# gemma-metal

A from-scratch Metal inference engine for **Gemma-4-A4B** on Apple Silicon, with the scheduling and demo clients that make its primitives legible.

---

**What this repo demonstrates, in three bullets:**

1. **Kernel layer** — hand-written MSL kernels (paged attention, flex attention with per-block mask bitmaps, batched MMA GEMM, fused FFN, 2D-RoPE, vision tower) driving a 27-layer vision encoder + 30-layer MoE LM at batched AR throughput on M5 Max.
2. **Scheduler layer** — content-hash prefix cache across sessions, work-type coalescing so staggered HTTP arrivals still batch, vision↔LM async pipelining across dedicated command queues, and a slot-parallel multi-session runtime that scales to ~3.6× at four concurrent text streams.
3. **Client layer** — three peer web demos (tetraplex, labeler, loom) consuming one OpenAI-compatible REST surface, each one exercising a different slice of what the primitives unlock in practice.

---

## Headline demo — the multi-slot soft-prefill fix

Four concurrent multimodal queries sharing an image across four sessions. Same hardware, same model weights, same kernels; the only thing that changed between these two recordings is that the scheduler learned to fire soft-prefill on ≥ 2 ready slots instead of waiting for all-or-nothing.

<table>
<tr>
<th>Before — serialized soft-prefill</th>
<th>After — multi-slot batch (50% faster)</th>
</tr>
<tr>
<td><video src="recordings/tetraplex-20260420-115156.mp4" controls width="420"></video></td>
<td><video src="recordings/tetraplex-20260420-122258.mp4" controls width="420"></video></td>
</tr>
<tr>
<td align="center"><code>17.3s</code> end-to-end</td>
<td align="center"><code>8.7s</code> end-to-end</td>
</tr>
</table>

---

## Three demo clients, same API

All three live under the bridge's `/v1/*` OpenAI-compat surface. Server privileges nothing; each client is plain HTML + fetch.

| path | client | what it exercises |
| --- | --- | --- |
| [`/tetraplex`](http://localhost:8000/tetraplex) | four synchronized streams with bandwidth chart + per-session KV tenancy strip | batched AR decode at B=4 |
| [`/labeler`](http://localhost:8000/labeler) | drop a folder of PNGs, get JSONL labels back; live ribbons drain a queue | vision cache + shared-schema prefix + multi-slot soft prefill |
| [`/loom`](http://localhost:8000/loom) | tree-of-thoughts with SVG-woven branches forked from a shared root | content-hash prefix cache + fork-from-node |

---

## Quick start

```bash
# 1. Build the Metal inference shared library
make libgemma_metal.dylib

# 2. Install bridge deps (uv manages a venv automatically)
cd server && uv sync

# 3. Point at your Gemma-4 safetensors + GGUF and run
GEMMA_SAFETENSORS=/path/to/model-00001-of-00002.safetensors \
uv run uvicorn bridge:app --host 0.0.0.0 --port 8000

# 4. Open http://localhost:8000 — pick a client
```

Prerequisites: Apple Silicon (M-series), Xcode command-line tools for the Swift compiler, and `uv` for Python deps.

---

## Key numbers on M5 Max

| workload | throughput | notes |
| --- | --- | --- |
| LM AR decode, 4 concurrent text streams | **3.6× speedup** vs single stream | weight-load-amortized via paged attention at B=4 |
| Vision tower (2520 patches, fp32 intermediates) | **880 ms/image** | 16×16 MMA tiles + fused FFN, bf16 trajectory preserved |
| Vision GEMM effective TFLOPS | **1.66 TFLOPS** | ~55% of realistic fp32-accum ceiling on this hardware |
| 8-image labeling at 4-concurrent | **30 s** end-to-end | down from 53 s naïve; scheduler coalesces by work-type |

---

## What's in the tree

```
.                                   top-level Metal + Swift engine
├── kernels.swift                   all MSL kernels (attention, GEMM, MoE, vision)
├── lm_engine.swift                 multi-session scheduler + tick() loop
├── vision_tower.swift              27-layer vision encoder (batched + async)
├── vision_residency.swift          3-state memory pressure handling
├── paged_attention.swift           page table + content-hash prefix cache
├── ffi.swift                       C-ABI surface for the Python bridge
│
├── server/
│   ├── bridge.py                   FastAPI bridge, OpenAI-compat + demo endpoints
│   ├── gemma_ffi.py                ctypes wrapper over libgemma_metal.dylib
│   └── static/                     the three demo clients
│       ├── clients.html            the directory page served at /
│       ├── index.html              tetraplex
│       ├── labeler.html            image labeler
│       └── loom.html               tree-of-thoughts
│
├── test_data/                      reference tensors + test frames
└── recordings/                     captured demo videos
```

---

## API surface, in case you want to write a fourth client

| method | path | purpose |
| --- | --- | --- |
| `GET` | `/health` | engine readiness, model name, cache stats |
| `POST` | `/v1/chat/completions` | OpenAI-compat chat; supports `image_url` and `softs` content items |
| `POST` | `/v1/completions` | raw-text continuation (no chat template) |
| `POST` | `/v1/images/prewarm` | run vision tower + cache softs ahead of time |
| `POST` | `/v1/media/extract` | run vision, return softs as base64 for client-side persistence |
| `GET` | `/v1/kv/snapshot` | per-session page ownership + refcount |
| `GET` | `/v1/cache/stats` | vision cache hit rate + bytes |

The [`/v1/media/extract`](server/bridge.py) + `softs` content-type pair is the "take your conversation anywhere" mechanic — the client persists opaque base64 soft-token blobs and replays them across sessions, so the server stays stateless-per-request and the vision tower never runs twice for the same image bytes.

---

## Hardware + model

Tested on a **128 GB M5 Max**. Weights are bf16 safetensors for the vision tower (mmap'd, on-demand hydrated + OS-evictable under memory pressure) and Q4_K_M GGUF for the LM. One 128 GB unified-memory box runs the whole stack — vision + LM + KV caches for many sessions + bridge — without any tiering gymnastics.
