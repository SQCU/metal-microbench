# Quickstart

Apple Silicon (M-series) Mac required; tested on M5 Max with 128 GB unified memory. Xcode command-line tools give you the Swift compiler; [uv](https://docs.astral.sh/uv/) manages the Python bridge's dependencies.

```bash
# 1. Build the Metal inference shared library (kernels.swift + engine + FFI)
make libgemma_metal.dylib

# 2. Install bridge deps (uv creates a venv automatically)
cd server && uv sync

# 3. Point the bridge at your weights and run it
GEMMA_SAFETENSORS=/path/to/model-00001-of-00002.safetensors \
GEMMA_GGUF=/path/to/model.q4_k_m.gguf \
uv run uvicorn bridge:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000> for the client index; pick one of the three demo clients or drive the REST API directly.

---

## Demo clients

| path | client | what it exercises |
| --- | --- | --- |
| `/tetraplex` | four synchronized chat streams, bandwidth chart + KV tenancy strip | batched AR decode at B=4 |
| `/labeler` | drop a folder of PNGs, get JSONL labels back; 4 live-decoding ribbons drain a queue | vision cache + shared-schema prefix + multi-slot soft prefill |
| `/loom` | tree-of-thoughts with SVG-woven branches forked from a shared root | content-hash prefix cache + fork-from-node |

---

## REST API surface

OpenAI-compatible for the chat/completion endpoints; the extras (`/v1/images/prewarm`, `/v1/media/extract`, `/v1/kv/snapshot`) expose the engine's internals for building custom clients.

| method | path | purpose |
| --- | --- | --- |
| `GET` | `/health` | engine readiness, model name, cache stats |
| `POST` | `/v1/chat/completions` | OpenAI-compat chat; accepts `image_url` and `softs` content items |
| `POST` | `/v1/completions` | raw-text continuation (no chat template) |
| `POST` | `/v1/images/prewarm` | run vision tower + cache softs ahead of time |
| `POST` | `/v1/media/extract` | run vision, return softs as base64 for client-side persistence |
| `GET` | `/v1/kv/snapshot` | per-session page ownership + refcount |
| `GET` | `/v1/cache/stats` | vision cache hit rate + bytes |
| `POST` | `/v1/cache/clear` | flush the vision cache |

The `/v1/media/extract` + `{"type":"softs"}` content item pair lets a client persist opaque base64 soft-token blobs locally and replay them across sessions — server stays stateless-per-request, vision tower never runs twice for the same image bytes.

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
│   ├── bridge.py             FastAPI bridge, OpenAI-compat + demo endpoints
│   ├── gemma_ffi.py          ctypes wrapper over libgemma_metal.dylib
│   └── static/               the three demo clients (tetraplex, labeler, loom)
│
├── test_data/                reference tensors + test frames
└── recordings/               captured demo videos
```
