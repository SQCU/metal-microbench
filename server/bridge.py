"""FastAPI bridge: OpenAI-compatible HTTP → Swift Metal engine via gemma_ffi.

Endpoints (every browser-side call is curl-testable):
  GET  /health                          — liveness + engine stats
  GET  /v1/models                       — list (static, one entry)
  POST /v1/chat/completions             — OpenAI-compatible chat (stream or not)
  POST /v1/completions                  — legacy non-chat completions
  GET  /                                — serves the two-pane web demo
  GET  /static/*                        — static web assets

Concurrency model:
  - One background pump thread loops gemma_tick() + drains every live session
    into per-session thread-safe queues. Async request handlers await chunks
    from their session's queue; this is the only cross-thread channel.
  - All Swift calls go through gemma_ffi (which serializes via Swift's own
    NSRecursiveLock), so Python-side locking isn't strictly needed, but we
    keep request handlers cheap and let the pump do the heavy work.

Run:
  uv run uvicorn bridge:app --host 0.0.0.0 --port 8000 --log-level info
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import gemma_ffi as g


# Paths can come from two places, checked in this order:
#   1. GEMMA_GGUF / GEMMA_SAFETENSORS env vars (what serve.py sets after
#      resolving config.toml; also the escape hatch for one-off overrides)
#   2. Legacy GGUF_PATH / VISION_ST env vars for anyone running the bridge
#      manually via the old command-line invocation.
# If none of the above are set, the bridge will error at startup with a
# pointer to serve.py + fetch-weights.py.
GGUF_PATH = (
    os.environ.get("GEMMA_GGUF")
    or os.environ.get("GGUF_PATH")
    or ""
)
VISION_SAFETENSORS = (
    os.environ.get("GEMMA_SAFETENSORS")
    or os.environ.get("VISION_ST")
    or ""
)
MODEL_NAME = os.environ.get("GEMMA_MODEL_NAME", "gemma-4-a4b-q4km")

# Active sessions on the bridge side: sid → per-session state.
@dataclass
class SessionState:
    sid: int
    queue: "queue.Queue[list[int] | None]" = field(default_factory=queue.Queue)  # None = EOF
    done: bool = False

_sessions: dict[int, SessionState] = {}
_sessions_lock = threading.Lock()
_pump_running = False
_pump_thread: threading.Thread | None = None


def _pump_loop() -> None:
    """Background thread: advance the engine and drain outputs into queues.

    Sleeps briefly when the engine has no work to avoid busy-spinning. Any
    poll() that returns tokens is fanned out to the session's queue; EOF
    is signalled with a None sentinel once state==.done.
    """
    idle_sleep = 0.002
    while _pump_running:
        did_work = False
        if g.has_work():
            g.tick()
            did_work = True
        # Drain every live session we know about, even if tick emitted zero
        # new tokens — some sessions may have pending output from earlier ticks.
        with _sessions_lock:
            snapshot = list(_sessions.items())
        for sid, state in snapshot:
            if state.done:
                continue
            tokens = g.poll(sid, 64)
            if tokens:
                state.queue.put(tokens)
                did_work = True
            # Check terminal state AFTER drain so we don't miss the last tokens.
            st = g.session_state(sid)
            if st == g.STATE_DONE:
                # One more drain to be safe, then EOF.
                leftover = g.poll(sid, 64)
                if leftover:
                    state.queue.put(leftover)
                state.queue.put(None)
                state.done = True
        if not did_work:
            time.sleep(idle_sleep)
        else:
            # Yield briefly so other FFI callers (kv snapshot pollers,
            # /health, any HTTP handler that calls into gemma_ffi) get a
            # chance to acquire ffiLock between our tight-loop reacquires.
            # Without this, the pump's back-to-back FFI calls starve
            # snapshot threads indefinitely: NSLock isn't strictly fair,
            # and Python's GIL + thread scheduling compounds the effect.
            # Empirically: ~30% throughput under 2Hz snapshot polling → 0%
            # after adding this yield.
            time.sleep(0.0005)  # 0.5 ms — one tick loses <2% but
                                # snapshot handlers drop from ~120 ms to
                                # one-tick latency.


app = FastAPI(title="Gemma Metal Bridge", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global _pump_running, _pump_thread
    if not GGUF_PATH or not Path(GGUF_PATH).exists():
        raise RuntimeError(
            f"GGUF weights not found at {GGUF_PATH!r}. "
            "Either run `uv run --with huggingface_hub python server/scripts/fetch-weights.py` "
            "to download, point server/config.toml's gguf_path at an existing file, "
            "or launch with GEMMA_GGUF=/path/to/.gguf."
        )
    print(f"[bridge] loading {GGUF_PATH}")
    t0 = time.time()
    g.init(GGUF_PATH)
    print(f"[bridge] LM ready in {time.time() - t0:.2f}s (bos={g.bos_id()}, eos={g.eos_id()})")

    # Vision weights are optional — only loaded if the safetensors file
    # exists at the configured path. Without them, /v1/chat/completions
    # with image_url content items will 400.
    if VISION_SAFETENSORS and Path(VISION_SAFETENSORS).exists():
        t1 = time.time()
        try:
            g.vision_init(VISION_SAFETENSORS)
            print(f"[bridge] vision ready in {time.time() - t1:.2f}s")
        except Exception as e:
            print(f"[bridge] vision init failed: {e}")
    else:
        where = VISION_SAFETENSORS or "(not configured)"
        print(f"[bridge] vision safetensors not at {where} — image_url disabled")

    _pump_running = True
    _pump_thread = threading.Thread(target=_pump_loop, name="gemma-pump", daemon=True)
    _pump_thread.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    global _pump_running
    _pump_running = False


# --- Chat template + submit orchestration ---

GEMMA_TURN_OPEN = "<|turn>"
GEMMA_TURN_CLOSE = "<turn|>"


def _decode_image_url(url: str) -> bytes:
    """Accepts data:image/...;base64,... URIs and http(s) URLs.
    Returns raw PNG/JPEG/... bytes suitable for writing to a temp file."""
    if url.startswith("data:"):
        # data:image/png;base64,XXXX
        try:
            header, payload = url.split(",", 1)
        except ValueError:
            raise HTTPException(400, "malformed data URL")
        if ";base64" in header:
            return base64.b64decode(payload)
        # percent-encoded text (rare for images, but spec-compliant).
        return urllib.parse.unquote(payload).encode("latin-1")
    if url.startswith("http://") or url.startswith("https://"):
        with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310 (trusted dev demo)
            return r.read()
    raise HTTPException(400, f"unsupported image_url scheme: {url[:32]!r}")


def _render_user_content_to_chunks(content) -> list[tuple[str, Any]]:
    """Break an OpenAI message's `content` into an ordered list of chunks
    to submit, interleaving text and images as the user sent them.

    Returns: [('text', str), ('image_bytes', bytes), ...]
    """
    if isinstance(content, str):
        return [('text', content)]
    if isinstance(content, list):
        out: list[tuple[str, Any]] = []
        for part in content:
            t = part.get("type")
            if t == "text":
                out.append(('text', part.get("text", "")))
            elif t == "image_url":
                url = part.get("image_url", {})
                if isinstance(url, dict):
                    url = url.get("url", "")
                out.append(('image_bytes', _decode_image_url(url)))
            elif t == "softs":
                # Client is replaying soft tokens it received from a
                # previous /v1/media/extract response. Skips the vision
                # tower entirely — the server is pure function of its
                # inputs. {"type":"softs", "softs_b64":..., "n_tokens":N,
                # "is_fp32": true}
                out.append(('softs', {
                    "bytes": base64.b64decode(part.get("softs_b64", "")),
                    "n_tokens": int(part.get("n_tokens", 0)),
                    "is_fp32": bool(part.get("is_fp32", True)),
                }))
            # Unknown types silently dropped; could also 400 here.
        return out
    return [('text', str(content))]


def submit_messages(sid: int, messages: list[dict]) -> None:
    """Build the session's priming queue from an OpenAI messages list.

    The critical correctness property: text spans must be tokenized as
    long contiguous strings. BPE tokenization is NOT splitting-safe —
    tokenize('<|turn>user\\n') + tokenize('Count to 5.') does not equal
    tokenize('<|turn>user\\nCount to 5.'). So we accumulate text into a
    pending buffer and only flush on image boundaries (or end-of-input).
    """
    pending_text: list[str] = []
    first_submit = True

    def flush_text() -> None:
        nonlocal first_submit
        if not pending_text:
            return
        combined = "".join(pending_text)
        pending_text.clear()
        toks = g.tokenize(combined, add_bos=first_submit)
        if toks:
            g.submit(sid, toks)
            first_submit = False

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        pending_text.append(f"{GEMMA_TURN_OPEN}{role}\n")
        for kind, payload in _render_user_content_to_chunks(content):
            if kind == 'text':
                pending_text.append(payload)
            elif kind == 'image_bytes':
                # Flush everything pending as one tokenize+submit so the chat
                # template text tokenizes correctly, THEN run vision and
                # submit BOI/softs/EOI, THEN start a new pending buffer.
                flush_text()
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    f.write(payload)
                    tmp = f.name
                try:
                    n = g.submit_image_path(sid, tmp)
                    first_submit = False
                    if n <= 0:
                        raise HTTPException(500, "vision tower returned no soft tokens")
                finally:
                    try: os.unlink(tmp)
                    except OSError: pass
            elif kind == 'softs':
                # Client-provided softs from a prior extract call. Same
                # flush-text-first invariant as image_bytes.
                flush_text()
                n = g.submit_softs(sid, payload["bytes"],
                                   payload["n_tokens"], payload["is_fp32"])
                first_submit = False
                if n <= 0:
                    raise HTTPException(500, "submit_softs returned no soft tokens")
        pending_text.append(f"{GEMMA_TURN_CLOSE}\n")
    pending_text.append(f"{GEMMA_TURN_OPEN}model\n")
    flush_text()


# --- Endpoints ---

@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ready" if g.is_ready() else "loading",
        "model": MODEL_NAME,
        "multimodal": g.vision_is_ready(),
        "active_sessions": g.active_session_count(),
        "pump_running": _pump_running,
        "vision_cache": g.vision_cache_stats() if g.vision_is_ready() else None,
        "vision_residency": {
            "state": g.vision_residency_state() if g.vision_is_ready() else "unbound",
            "bytes": g.vision_residency_bytes() if g.vision_is_ready() else 0,
        },
    })


@app.post("/v1/vision/allow_evict")
def vision_allow_evict() -> JSONResponse:
    """Flip vision working buffers to .volatile so macOS can reclaim them
    under pressure. Next vision request either finds the pages still
    resident (fast re-pin) or rehydrates from the mmap (~160 ms)."""
    g.vision_allow_evict()
    return JSONResponse({
        "state": g.vision_residency_state(),
        "bytes": g.vision_residency_bytes(),
    })


@app.post("/v1/vision/force_drop")
def vision_force_drop() -> JSONResponse:
    """Drop vision working buffers immediately (simulate .critical pressure).
    Session KV pages stay pinned. Next vision request will fully rehydrate."""
    g.vision_force_drop()
    return JSONResponse({
        "state": g.vision_residency_state(),
        "bytes": g.vision_residency_bytes(),
    })


_STATE_NAMES = {0: "idle", 1: "priming", 2: "generating", 3: "paused", 4: "done"}


@app.get("/v1/kv/snapshot")
def kv_snapshot(detail: bool = False) -> JSONResponse:
    """Per-session page ownership stats. Clients poll this during generation
    to render live tenancy — pages with refcount > 1 are shared across
    sessions (prefix-cache hit or explicit adoptKvFrom).

    Default shape (used by the demo UIs) is aggregate-only — page_count
    and shared_count per session. This is O(sessions) FFI calls under one
    ffiLock each.

    Pass ?detail=1 to get the per-page (phys, refcount, shared_with)
    list. That mode is O(sessions × pages) FFI calls and serializes
    against the pump's g.tick() via ffiLock — polling it at 2 Hz during
    active decoding collapses AR throughput ~4×. Use sparingly; the live
    UI pollers should stay on the default aggregate path.
    """
    sessions = []
    if detail:
        # Per-page enumeration (O(sessions + pages) FFI calls). Pollers
        # should NOT use this — it serializes against gemma_tick().
        sids = g.active_session_ids()
        for sid in sids:
            snap = g.session_snapshot(sid)
            if not snap:
                continue
            page_entries = []
            for phys in snap["pages"]:
                owners = g.page_owners(int(phys))
                page_entries.append({
                    "phys": int(phys),
                    "refcount": len(owners),
                    "shared_with": [o for o in owners if o != snap["sid"]],
                })
            sessions.append({
                "sid": snap["sid"],
                "position": snap["position"],
                "state": _STATE_NAMES.get(snap["state"], "?"),
                "page_count": len(page_entries),
                "shared_count": sum(1 for p in page_entries if p["refcount"] > 1),
                "pages": page_entries,
            })
    else:
        # Bulk aggregate path — ONE FFI call returns every session's
        # (sid, position, state, page_count, shared_count) under a
        # single ffiLock acquisition. At 2 Hz polling this takes up to
        # ~30 ms of tick-contention per second (one tick-length), down
        # from 150+ ms in the naive per-page-owners path that starved
        # AR decode ~3-4×.
        for row in g.kv_snapshot_summary():
            sessions.append({
                "sid": row["sid"],
                "position": row["position"],
                "state": _STATE_NAMES.get(row["state"], "?"),
                "page_count": row["page_count"],
                "shared_count": row["shared_count"],
            })
    return JSONResponse({
        "sessions": sessions,
        "page_size_tokens": 16,      # PAGE_SLIDE — constant for the demo
        "total_pages": 8192,         # pool capacity
    })


@app.get("/v1/cache/stats")
def cache_stats() -> JSONResponse:
    """Inspect the vision-tower soft-tokens cache."""
    stats = g.vision_cache_stats() if g.vision_is_ready() else {
        "entries": 0, "hits": 0, "misses": 0, "bytes": 0,
    }
    total = stats["hits"] + stats["misses"]
    stats["hit_rate"] = (stats["hits"] / total) if total > 0 else 0.0
    return JSONResponse(stats)


@app.post("/v1/cache/clear")
def cache_clear() -> JSONResponse:
    """Flush the vision cache. Returns the number of entries evicted."""
    n = g.vision_cache_clear() if g.vision_is_ready() else 0
    return JSONResponse({"evicted": n})


@app.post("/v1/images/prewarm")
async def prewarm_image(req: Request) -> JSONResponse:
    """Accept the same image_url shape as chat content items and populate
    the cache. First call pays the vision-tower cost; subsequent chat calls
    using the same image bytes skip vision entirely.

    Body: {"image_url": {"url": "data:image/png;base64,..."}}
          or {"image_url": "data:..."}
          or {"url": "..."}
    """
    body = await req.json()
    url = body.get("image_url") or body.get("url") or ""
    if isinstance(url, dict):
        url = url.get("url", "")
    if not url:
        raise HTTPException(400, "image_url is required")
    if not g.vision_is_ready():
        raise HTTPException(503, "vision not initialized")

    png = _decode_image_url(url)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png)
        tmp = f.name
    try:
        t0 = time.time()
        n = g.vision_prewarm_path(tmp)
        dt = time.time() - t0
    finally:
        try: os.unlink(tmp)
        except OSError: pass

    return JSONResponse({
        "soft_tokens": n,
        "cache_key": g.vision_last_cache_key(),
        "elapsed_ms": int(dt * 1000),
        "stats": g.vision_cache_stats(),
    })


@app.post("/v1/control/screen")
async def control_screen(req: Request) -> JSONResponse:
    """Screen every layer for separation strength + per-pair coherence
    on a pair of prose sets, then register a cvec at the best layer(s).
    Body: {"id": "anti-elara",
           "positive": [...],
           "negative": [...],
           "chat_template": true,
           "top_k": 3,             # register at top-K layers (id-0, id-1, ...)
           "register_best": true}  # register at the single best layer under `id`

    Returns per-layer stats so the UI can plot signal vs layer and pick
    intervention points. Metric is a composite:
        signal   = ||mean(positive) - mean(negative)||
        coherence = mean over pairs i of cos(diff_i, mean_diff)
        score     = signal * max(0, coherence)
    High score = directions consistently point the same way AND
    separation is large — the best layer for steering this feature."""
    import numpy as np
    body = await req.json()
    cvec_id = str(body.get("id", "")).strip()
    positives = [str(x) for x in (body.get("positive") or []) if str(x).strip()]
    negatives = [str(x) for x in (body.get("negative") or []) if str(x).strip()]
    chat_template = bool(body.get("chat_template", True))
    register_best = bool(body.get("register_best", True))
    top_k = int(body.get("top_k", 1))
    if not cvec_id or not positives or not negatives:
        raise HTTPException(400, "need id, positive (≥1), negative (≥1)")

    HIDDEN = 2816
    NUM_LAYERS = 30

    def capture_stack(examples: list[str]) -> np.ndarray:
        # Returns [n_examples, NUM_LAYERS, HIDDEN] fp32.
        rows = []
        for ex in examples:
            blob = g.capture_all_layer_residuals_for_prompt(ex, chat_template=chat_template)
            arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
            rows.append(arr.reshape(NUM_LAYERS, HIDDEN))
        return np.stack(rows)  # [N, L, D]

    pos = capture_stack(positives)   # [P, L, D]
    neg = capture_stack(negatives)   # [N, L, D]

    # Per-layer stats. For "coherence" we pair each positive with each
    # negative cross-product style — mean signal is well-defined over
    # (unpaired) sets. If the caller happens to pass same-count lists,
    # coherence computes over element-wise pairs too; here we default
    # to the set-mean difference.
    layer_stats = []
    for L in range(NUM_LAYERS):
        pL = pos[:, L, :]           # [P, D]
        nL = neg[:, L, :]           # [N, D]
        mean_diff = pL.mean(axis=0) - nL.mean(axis=0)       # [D]
        signal = float(np.linalg.norm(mean_diff))
        # Coherence: for each individual "positive - negative_mean" and
        # "positive_mean - negative_i" projection onto mean_diff, how
        # consistently aligned are they? Measured via cosine.
        if signal > 1e-9:
            u = mean_diff / signal
            pos_coh = float(np.mean((pL - nL.mean(0)) @ u / (np.linalg.norm(pL - nL.mean(0), axis=1) + 1e-9)))
            neg_coh = float(np.mean((pL.mean(0) - nL) @ u / (np.linalg.norm(pL.mean(0) - nL, axis=1) + 1e-9)))
            coherence = (pos_coh + neg_coh) / 2
        else:
            coherence = 0.0
        score = signal * max(coherence, 0.0)
        layer_stats.append({
            "layer": L,
            "signal": signal,
            "coherence": coherence,
            "score": score,
        })

    # Sort by score, pick top-K for registration (under `id` for the
    # single best, and "`id`-L{n}" for runners-up if top_k > 1).
    ranked = sorted(layer_stats, key=lambda s: s["score"], reverse=True)
    if register_best and ranked:
        best_L = ranked[0]["layer"]
        direction = pos[:, best_L, :].mean(0) - neg[:, best_L, :].mean(0)
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        g.control_register_fp16(cvec_id, direction.astype(np.float16).tobytes())
    registered = [{"id": cvec_id, "layer": ranked[0]["layer"]}] if (register_best and ranked) else []
    for i in range(1, min(top_k, len(ranked))):
        L = ranked[i]["layer"]
        subid = f"{cvec_id}-L{L}"
        direction = pos[:, L, :].mean(0) - neg[:, L, :].mean(0)
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        g.control_register_fp16(subid, direction.astype(np.float16).tobytes())
        registered.append({"id": subid, "layer": L})

    return JSONResponse({
        "registered": registered,
        "best_layer": ranked[0]["layer"] if ranked else None,
        "best_score": ranked[0]["score"] if ranked else None,
        "layer_stats": layer_stats,
        "ranked_layers": [{"layer": s["layer"], "score": s["score"]} for s in ranked[:5]],
    })


@app.post("/v1/control/construct")
async def control_construct(req: Request) -> JSONResponse:
    """Build a control vector from contrastive prose pairs. Body:
      {"id":            "joy-direction",
       "layer":         15,
       "positive":      ["joyful example 1", "another cheerful example", ...],
       "negative":      ["sad example 1",    "another downcast example",  ...],
       "chat_template": true}   // default true — wrap each example in a user/model turn

    For each example, the engine runs a forward pass and captures the
    last-priming-tick residual at the given layer. The final cvec is
    normalized(mean(positive) - mean(negative)). Registered under `id`
    in the cvec registry — use it as an effector or detector cvec_id
    in /v1/chat/completions. Returns the pre-normalization norm + counts
    as a rough "signal strength" indicator (very small norm = classes
    are indistinguishable at this layer)."""
    import struct, math
    body = await req.json()
    cvec_id = str(body.get("id", "")).strip()
    layer = int(body.get("layer", 15))
    positives = [str(x) for x in (body.get("positive") or []) if str(x).strip()]
    negatives = [str(x) for x in (body.get("negative") or []) if str(x).strip()]
    chat_template = bool(body.get("chat_template", True))
    if not cvec_id or not positives or not negatives:
        raise HTTPException(400, "need id, positive (≥1), negative (≥1)")

    HIDDEN = 2816
    def accumulate(examples: list[str]) -> list[float]:
        acc = [0.0] * HIDDEN
        for ex in examples:
            raw = g.capture_residual_for_prompt(ex, layer, chat_template=chat_template)
            # raw is HIDDEN little-endian float16 halves. Use struct to
            # decode — Python's '<e' is IEEE-754 binary16.
            vals = struct.unpack(f"<{HIDDEN}e", raw)
            for i in range(HIDDEN):
                acc[i] += vals[i]
        return [x / len(examples) for x in acc]

    pos_mean = accumulate(positives)
    neg_mean = accumulate(negatives)
    direction = [p - n for p, n in zip(pos_mean, neg_mean)]
    raw_norm = math.sqrt(sum(x * x for x in direction))
    if raw_norm < 1e-9:
        raise HTTPException(400, "positive and negative means are identical — "
                                 "examples are indistinguishable at this layer")
    direction = [x / raw_norm for x in direction]
    # Pack to fp16 and register.
    bytes_out = struct.pack(f"<{HIDDEN}e", *direction)
    try:
        g.control_register_fp16(cvec_id, bytes_out)
    except Exception as e:
        raise HTTPException(400, str(e))
    return JSONResponse({
        "id": cvec_id,
        "layer": layer,
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "hidden": HIDDEN,
        "raw_norm": raw_norm,
        "note": ("constructed from mean(positive)-mean(negative), normalized. "
                 "raw_norm indicates separation strength — higher is a more "
                 "distinct direction in representation space"),
    })


@app.post("/v1/tokenize")
async def tokenize_endpoint(req: Request) -> JSONResponse:
    """Diagnostic: tokenize a raw string, return [{id, text}] pairs so
    callers can verify the Gemma-4 chat template special-tokens
    (<|turn>, <turn|>, <|channel>, <channel|>) map to single special-
    token IDs rather than being split into surface-char subtokens.
    Body: {"text": "...", "add_bos": false}"""
    body = await req.json()
    text = str(body.get("text", ""))
    add_bos = bool(body.get("add_bos", False))
    toks = g.tokenize(text, add_bos=add_bos)
    out = [{"id": int(t), "text": g.detokenize([int(t)])} for t in toks]
    return JSONResponse({"tokens": out, "count": len(toks)})


@app.post("/v1/detokenize")
async def detokenize_endpoint(req: Request) -> JSONResponse:
    """Diagnostic: given a list of token IDs, return the string each
    one detokenizes to. Lets callers probe what ID 106 (eos) or ID 2
    (bos) actually look like as strings."""
    body = await req.json()
    ids = body.get("ids") or []
    out = [{"id": int(t), "text": g.detokenize([int(t)])} for t in ids]
    return JSONResponse({"tokens": out})


@app.post("/v1/raw_generate")
async def raw_generate(req: Request) -> JSONResponse:
    """Diagnostic: submit a raw token-ID list (bypassing chat-template
    construction) and greedily generate up to max_tokens new tokens.
    Lets callers A/B test whether constructing a prompt with actual
    special-token IDs (105 = <|turn>, 106 = <turn|>, etc.) vs their
    surface-string forms ('<|turn>' → 4 subtoken IDs) produces
    materially different generation.
    Body: {"token_ids": [2, 105, ...], "max_tokens": 64}"""
    body = await req.json()
    token_ids = [int(x) for x in body.get("token_ids", [])]
    max_tokens = int(body.get("max_tokens", 64))
    if not token_ids:
        raise HTTPException(400, "token_ids required")
    sid = g.open_session(max_new_tokens=max_tokens)
    try:
        g.submit(sid, token_ids)
        out_ids: list[int] = []
        deadline = time.time() + 60
        while time.time() < deadline:
            out_ids.extend(g.poll(sid, 64))
            if g.session_state(sid) == g.STATE_DONE:
                out_ids.extend(g.poll(sid, 64))
                break
            await asyncio.sleep(0.02)
        text = g.detokenize(out_ids)
        return JSONResponse({
            "prompt_tokens": len(token_ids),
            "output_tokens": len(out_ids),
            "output_token_ids": out_ids,
            "output_text": text,
        })
    finally:
        _close_session(sid)


@app.post("/v1/perplexity")
async def perplexity(req: Request) -> JSONResponse:
    """Teacher-force a completion under an optional set of intervention
    controls and return per-token logprobs + top-K alternatives + mean
    perplexity. The use case is diagnosing steering pathologies: a
    steered completion that degenerates into token-repetition will
    show collapsed entropy (top-1 logprob near 0, all alternatives
    vanishingly small), while a genuinely shifted-but-coherent
    completion will show a normal entropy profile with a moved mean.

    Implementation: open a session, attach controls if any, submit the
    prompt (prefill), then walk the completion token-by-token via AR
    teacher-force: read slot logits → compute log-softmax → record the
    actual token's logprob + top-K alternatives → set_next_input(
    completion_token) → tick. One AR step per completion token. With
    B=4 batching, 4 such calls can run in parallel without slowdown.

    Body:
      {"prompt":      "...",
       "completion":  "...",         // text to teacher-force
       "controls":    [...],         // optional, same shape as chat/completions
       "chat_template": true,        // default true: wrap prompt in
                                     // <|turn>user\n...<turn|>\n<|turn>model\n
                                     // and prefix completion with thought
                                     // markers so teacher-forced sequence
                                     // matches what /v1/chat/completions
                                     // actually emits. Without this, scores
                                     // are computed on a completely
                                     // mis-contextualized sequence and come
                                     // out hugely negative.
       "temperature": 0.0,
       "top_k":       5}             // per-position top-K alternatives

    Returns:
      {"prompt_tokens": N,
       "completion_tokens": M,
       "mean_logprob": float,
       "perplexity": float,
       "min_logprob": float,
       "collapsed_positions": int,   // count of positions where top-1 prob > 0.95
       "tokens": [{"id", "text", "logprob", "top_alternatives": [...]}]}
    """
    import numpy as np
    body = await req.json()
    prompt = str(body.get("prompt", "")).strip()
    completion = str(body.get("completion", "")).strip()
    controls = body.get("controls", []) or []
    temperature = float(body.get("temperature", 0.0))
    top_k = int(body.get("top_k", 5))
    chat_template = bool(body.get("chat_template", True))
    if not prompt or not completion:
        raise HTTPException(400, "prompt and completion are both required")

    # When chat_template=True (default), wrap the prompt as a user turn
    # and prefix the completion with the model-turn + thought-channel
    # markers that /v1/chat/completions emits. The teacher-forced
    # sequence then looks exactly like a realistic chat, and logprobs
    # reflect the model's actual in-context predictions. Without this
    # the model is being asked to predict raw prose tokens after raw
    # instruction tokens — a sequence it has never been trained on,
    # producing catastrophic logprobs (≈ -30 on the first token).
    if chat_template:
        prompt_wrapped = f"{GEMMA_TURN_OPEN}user\n{prompt}{GEMMA_TURN_CLOSE}\n{GEMMA_TURN_OPEN}model\n<|channel>thought\n<channel|>"
        completion_wrapped = completion
    else:
        prompt_wrapped = prompt
        completion_wrapped = completion

    prompt_toks = g.tokenize(prompt_wrapped, add_bos=True)
    completion_toks = g.tokenize(completion_wrapped, add_bos=False)
    if not completion_toks:
        raise HTTPException(400, "completion tokenized to 0 tokens")

    # Flow (no teacher-forcing APIs needed; we just re-submit tokens as
    # 1-token prefill tiles, using each post-prefill logits buffer):
    #   1. submit(prompt)                  → logits predict completion[0]
    #   2. for i in 1..N: submit([c[i-1]]) → logits predict completion[i]
    # Between submits, drain waits for state to settle back to .generating.
    # Cost: N 1-token prefill tiles ≈ N * 40ms. Batchable at B=4 if the
    # UI fires multiple perplexity calls concurrently.
    #
    # The session's own post-prefill "sampled first token" is a phantom:
    # it goes into outputQueue but never K/V-writes, because no AR tick
    # consumes it before our next submit flips state back to .priming.
    # maxNewTokens is set wide enough to never hit the generation cap.
    max_tokens = len(completion_toks) * 2 + 16
    sid = g.open_session(max_new_tokens=max_tokens)
    try:
        if temperature > 0:
            g.session_set_temperature(sid, temperature)
        if controls:
            _attach_controls(sid, controls)

        def score_slot_logits(expected_tok: int) -> tuple[float, list[dict], float]:
            """Read slot logits, compute log-softmax of `expected_tok`,
            top-K alts (excluding the expected token), and top-1 prob
            (for collapse detection)."""
            raw = g.session_get_slot_logits(sid)
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            m = arr.max()
            arr = arr - m
            lse = float(np.log(np.exp(arr).sum()))
            logprobs = arr - lse
            lp = float(logprobs[int(expected_tok)])
            # argpartition for top (k+1) so we can skip the actual
            # token without losing a slot.
            kp = min(top_k + 1, len(logprobs))
            idx = np.argpartition(-logprobs, kp - 1)[:kp]
            idx = idx[np.argsort(-logprobs[idx])]
            alts = []
            for alt_id in idx:
                if int(alt_id) == int(expected_tok): continue
                alts.append({
                    "id": int(alt_id),
                    "text": g.detokenize([int(alt_id)]),
                    "logprob": float(logprobs[int(alt_id)]),
                })
                if len(alts) >= top_k: break
            top1_prob = float(np.exp(float(logprobs[int(idx[0])])))
            return lp, alts, top1_prob

        def wait_position(target: int, timeout_s: float = 30.0):
            """Block until session position reaches `target`. The pump
            ticks the session once per AR step (~30 ms), advancing
            position by 1 per tick on single-token priming chunks; by
            qLen=8 per tile on multi-token prefill. Polling position
            directly sidesteps the state-machine race where a session
            briefly passes through .generating between ticks."""
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                pos = g.session_position(sid)
                if pos >= target:
                    return
                time.sleep(0.002)
            raise HTTPException(500,
                f"session position {g.session_position(sid)} didn't reach "
                f"target {target} within {timeout_s}s")

        # Submit prompt, wait for prefill → position == len(prompt).
        g.submit(sid, prompt_toks)
        target_pos = len(prompt_toks)
        wait_position(target_pos)
        # Immediately pause: prevents the pump from firing an unwanted
        # AR tick on the now-.generating session before we can read
        # logits and queue the next teacher-forced chunk. Without this,
        # the pump's ~30 ms tick cadence reliably interleaves between
        # our "read logits" and "submit next token", shifting positions
        # and corrupting the scored sequence.
        g.session_pause(sid)

        per_token = []
        sum_lp = 0.0
        min_lp = 0.0
        collapsed = 0
        for i, tok in enumerate(completion_toks):
            # Slot logits now predict the token at position target_pos
            # (= completion[i]). Read + score while paused.
            lp, alts, top1 = score_slot_logits(int(tok))
            sum_lp += lp
            if lp < min_lp: min_lp = lp
            if top1 > 0.95: collapsed += 1
            per_token.append({
                "id": int(tok),
                "text": g.detokenize([int(tok)]),
                "logprob": lp,
                "top_alternatives": alts,
            })
            # Teacher-force the next token: queue it, resume, wait for
            # position to advance, then re-pause. The pump fires exactly
            # one AR step (the 1-token chunk below the min=2 fast-prefill
            # threshold falls to step()) which writes K/V at target_pos
            # and new logits at target_pos+1.
            if i + 1 < len(completion_toks):
                g.submit(sid, [int(tok)])
                target_pos += 1
                g.session_resume(sid)
                wait_position(target_pos)
                g.session_pause(sid)

        mean_lp = sum_lp / len(completion_toks)
        return JSONResponse({
            "prompt_tokens": len(prompt_toks),
            "completion_tokens": len(completion_toks),
            "mean_logprob": mean_lp,
            "perplexity": float(np.exp(-mean_lp)),
            "min_logprob": min_lp,
            "collapsed_positions": collapsed,
            "tokens": per_token,
        })
    finally:
        _close_session(sid)


@app.post("/v1/control/construct_pca")
async def control_construct_pca(req: Request) -> JSONResponse:
    """Fit a control-vector *set* across the whole model. For each layer
    L, compute deltas = positive_residual_L - mean(negative_residual_L)
    and run SVD on the [P, HIDDEN] matrix. Each (layer, component) pair
    gets its own eigenvalue. Flatten into one global list, sort by
    eigenvalue descending, keep the top entries until cumulative ≥
    top_p × total_variance.

    The output set is a partition of the model's pos-vs-neg variation
    across layers AND components — a richer story than "steer at layer
    N", one that doesn't suggest residual streams are layer-local.
    Typical shape: a handful of (L, 0) PC1s plus a few high-signal
    (L, 1) PC2s dominate the first ~80% of variance; everything else is
    noise.

    Body:
      {"id_prefix":              "joy",      // cvecs registered as "joy-L12-C0", ...
       "positive":               [...],
       "negative":               [...],
       "top_p":                  0.80,       // cumulative-variance threshold (default 0.80)
       "max_components_per_layer": 4,        // hard cap (default 4; full rank is min(P,H))
       "chat_template":          true}

    Returns:
      {"components": [{layer, component, eigenvalue, fraction,
                       cumulative, cvec_id, scale}, ...],
       "total_variance", "captured_variance", "captured_fraction",
       "top_p", "all_eigenvalues": [[ev per component] per layer]}

    `scale` is eigenvalue / max_eigenvalue — apply these as the
    peakMagnitude of each attached ActiveControl to preserve the
    relative importance structure PCA gave us. A single user-facing
    intensity multiplier on top of these scales is all you should need.
    """
    import numpy as np
    body = await req.json()
    id_prefix = str(body.get("id_prefix", "")).strip()
    positives = [str(x) for x in (body.get("positive") or []) if str(x).strip()]
    negatives = [str(x) for x in (body.get("negative") or []) if str(x).strip()]
    top_p = float(body.get("top_p", 0.80))
    max_components_per_layer = int(body.get("max_components_per_layer", 4))
    chat_template = bool(body.get("chat_template", True))
    # capture_mode controls what "position in context" the residual is
    # sampled from:
    #   "user_end"  (default) — example wraps as user prose, residual
    #                captured at the end of the user turn. Represents
    #                "the model is ABOUT TO respond to this user text."
    #   "model_gen" — example wraps as the model having generated the
    #                prose inside its own turn, residual captured at
    #                end-of-model-content. Represents "the model IS
    #                generating this style of prose right now."
    # The "model_gen" mode closes the context mismatch between where
    # the cvec was fit (user-prose endpoint) and where it's applied
    # (mid-model-generation), which often manifests as degenerate token
    # repetition in small-intensity steered generations.
    capture_mode = str(body.get("capture_mode", "user_end")).strip()
    if capture_mode not in ("user_end", "model_gen", "rollout", "shared_source"):
        raise HTTPException(400,
            "capture_mode must be 'user_end', 'model_gen', 'rollout', or 'shared_source'")
    # rollout/shared_source-mode specific knobs.
    # K = positions per example.
    rollout_depth = int(body.get("rollout_depth", 16))
    rollout_prompt = str(body.get("rollout_prompt",
        "Continue the prose in the same style.")).strip()
    if capture_mode in ("rollout", "shared_source") and rollout_depth < 2:
        raise HTTPException(400,
            f"rollout_depth must be ≥ 2 for capture_mode={capture_mode}")
    if not id_prefix or not positives or not negatives:
        raise HTTPException(400, "need id_prefix, positive (≥1), negative (≥1)")
    if not (0.0 < top_p <= 1.0):
        raise HTTPException(400, "top_p must be in (0, 1]")
    if max_components_per_layer < 1:
        raise HTTPException(400, "max_components_per_layer must be ≥ 1")

    HIDDEN = 2816
    NUM_LAYERS = 30

    def wrap_for_capture(example: str) -> tuple[str, bool]:
        """Return (text, use_engine_chat_template).
        For model_gen mode we build the full template manually (with a
        generic "write a paragraph:" user turn) and pass the entire
        string through with chat_template=False so the engine doesn't
        double-wrap. Captured residual is at end of the model's prose."""
        if capture_mode == "model_gen":
            wrapped = (f"{GEMMA_TURN_OPEN}user\nwrite a short paragraph of prose.{GEMMA_TURN_CLOSE}\n"
                       f"{GEMMA_TURN_OPEN}model\n<|channel>thought\n<channel|>{example}")
            return wrapped, False
        return example, chat_template

    def teacher_force_residuals(continuation: str) -> np.ndarray:
        """Teacher-force `continuation` tokens through the engine,
        rooted in a SHARED anchor prompt `rollout_prompt`, capturing
        all-layer residuals at each of K teacher-forced positions.
        Returns [K, NUM_LAYERS, HIDDEN] fp32.

        Why this matters vs rollout mode: rollout lets the model
        generate its own K tokens past the seed, which means if the
        model drifts off-class (e.g. a "joyful" seed doesn't actually
        elicit joyful continuation, or the sampler lands on a neutral
        token-repetition attractor early), the captured residuals are
        contaminated by whatever the model chose rather than staying
        in the target class. Teacher-forcing through a KNOWN in-class
        continuation guarantees residuals come from the right
        distribution position-by-position.

        Also: all examples share the same S_0 (rollout_prompt + turn
        markers), so per-layer deltas between pos and neg are
        trajectory-comparable from the same starting state — no
        prompt-specific noise in the fitted direction.
        """
        anchor_wrapped = (f"{GEMMA_TURN_OPEN}user\n{rollout_prompt}{GEMMA_TURN_CLOSE}\n"
                           f"{GEMMA_TURN_OPEN}model\n<|channel>thought\n<channel|>")
        anchor_toks = g.tokenize(anchor_wrapped, add_bos=True)
        cont_toks = g.tokenize(continuation, add_bos=False)[:rollout_depth]
        K = len(cont_toks)
        if K < 2:
            # Example tokenizes to too little; skip by returning a
            # zero-weight placeholder (caller should filter).
            return np.zeros((0, NUM_LAYERS, HIDDEN), dtype=np.float32)

        sid = g.open_session(max_new_tokens=K + 4)
        try:
            # Submit anchor, drain prefill → position = len(anchor).
            # After prefill the slot's residual buffer has state from
            # the last anchor position — that's S_0's endpoint.
            g.submit(sid, anchor_toks)
            target_pos = len(anchor_toks)
            deadline = time.time() + 20
            while time.time() < deadline:
                if g.session_position(sid) >= target_pos: break
                time.sleep(0.002)
            g.session_pause(sid)

            rows = np.zeros((K, NUM_LAYERS, HIDDEN), dtype=np.float32)
            # Row 0: residual at S_0's endpoint (before any
            # continuation token consumed). Same for all examples of
            # the same class since anchor is shared — this anchors
            # the trajectory.
            rows[0] = np.frombuffer(g.get_all_layer_residuals(),
                                    dtype=np.float16).astype(np.float32).reshape(NUM_LAYERS, HIDDEN)

            # For each continuation token i in 0..K-2, submit as a
            # 1-token chunk which the engine's AR step consumes as
            # teacher-forced input. Position advances by 1 per
            # submit; we capture the residual written at that new
            # position. Row i+1 = residual having just consumed
            # cont_toks[i].
            for i in range(K - 1):
                g.submit(sid, [int(cont_toks[i])])
                target_pos += 1
                g.session_resume(sid)
                d2 = time.time() + 10
                while time.time() < d2:
                    if g.session_position(sid) >= target_pos: break
                    time.sleep(0.002)
                g.session_pause(sid)
                rows[i + 1] = np.frombuffer(g.get_all_layer_residuals(),
                                             dtype=np.float16).astype(np.float32).reshape(NUM_LAYERS, HIDDEN)
            return rows
        finally:
            _close_session(sid)

    def rollout_residuals(seed: str) -> np.ndarray:
        """Run rollout_depth AR steps on a seed and capture all-layer
        residuals at each step. Returns [rollout_depth, NUM_LAYERS,
        HIDDEN] fp32. Seed is framed as the start of the model's own
        prose response to a generic continue-the-prose user prompt,
        so each captured residual reflects the model's generation-time
        state (conditioned on the seed's tonal class), not its
        planning-time state. Uses pause/resume to prevent the pump
        from racing past the position we're about to capture."""
        wrapped = (f"{GEMMA_TURN_OPEN}user\n{rollout_prompt}{GEMMA_TURN_CLOSE}\n"
                   f"{GEMMA_TURN_OPEN}model\n<|channel>thought\n<channel|>{seed}")
        prompt_toks = g.tokenize(wrapped, add_bos=True)
        sid = g.open_session(max_new_tokens=rollout_depth + 4)
        try:
            g.submit(sid, prompt_toks)
            target_pos = len(prompt_toks)
            # Poll position until prefill done + post-prefill sample happened.
            deadline = time.time() + 30
            while time.time() < deadline:
                if g.session_position(sid) >= target_pos: break
                time.sleep(0.002)
            g.session_pause(sid)
            # Capture the residual at the end of prefill (first rollout step).
            rows = np.zeros((rollout_depth, NUM_LAYERS, HIDDEN), dtype=np.float32)
            rows[0] = np.frombuffer(g.get_all_layer_residuals(),
                                    dtype=np.float16).astype(np.float32).reshape(NUM_LAYERS, HIDDEN)
            # Tick K-1 more times; capture each post-tick residual.
            for k in range(1, rollout_depth):
                g.session_resume(sid)
                target_pos += 1
                d2 = time.time() + 10
                while time.time() < d2:
                    if g.session_position(sid) >= target_pos: break
                    time.sleep(0.002)
                g.session_pause(sid)
                rows[k] = np.frombuffer(g.get_all_layer_residuals(),
                                         dtype=np.float16).astype(np.float32).reshape(NUM_LAYERS, HIDDEN)
            return rows
        finally:
            _close_session(sid)

    def capture_stack(examples: list[str]) -> np.ndarray:
        if capture_mode in ("rollout", "shared_source"):
            # Both modes need the all-layer capture enabled for the
            # duration. rollout_residuals samples model-generated
            # continuation; teacher_force_residuals teacher-forces a
            # known continuation. Flatten [n_examples, K, L, D] →
            # [n_examples*K, L, D].
            fn = teacher_force_residuals if capture_mode == "shared_source" else rollout_residuals
            g.set_capture_all_layers(True)
            try:
                chunks = [fn(ex) for ex in examples]
                chunks = [c for c in chunks if c.shape[0] > 0]
                if not chunks:
                    raise HTTPException(400,
                        "every example tokenized to < 2 tokens — add longer prose")
                stacked = np.concatenate(chunks, axis=0)
            finally:
                g.set_capture_all_layers(False)
            return stacked
        # user_end / model_gen single-position paths unchanged.
        rows = []
        for ex in examples:
            text, use_chat = wrap_for_capture(ex)
            blob = g.capture_all_layer_residuals_for_prompt(text, chat_template=use_chat)
            arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
            rows.append(arr.reshape(NUM_LAYERS, HIDDEN))
        return np.stack(rows)

    pos = capture_stack(positives)   # [P, L, D]
    neg = capture_stack(negatives)   # [N, L, D]
    P = pos.shape[0]

    # Per-layer SVD. deltas_L[i] = pos_L[i] - mean(neg_L); we do NOT
    # mean-center deltas because PC1 should align with the class-mean
    # direction (the primary steering direction), and mean-centering
    # would project that out. PC2+ then capture variation *around* PC1
    # — fine-grained sub-directions within the "positive" class.
    max_k = min(max_components_per_layer, P)
    all_components = []   # (layer, k, eigenvalue, unit_direction)
    per_layer_eigenvalues: list[list[float]] = []
    for L in range(NUM_LAYERS):
        deltas = pos[:, L, :] - neg[:, L, :].mean(axis=0, keepdims=True)  # [P, D]
        # SVD: deltas = U @ diag(s) @ Vt; Vt[k] is the kth principal
        # direction; s[k]^2 is the corresponding eigenvalue (variance).
        try:
            _, s, Vt = np.linalg.svd(deltas, full_matrices=False)
        except np.linalg.LinAlgError:
            per_layer_eigenvalues.append([0.0] * max_k)
            continue
        eigs = (s ** 2).tolist()
        per_layer_eigenvalues.append(eigs[:max_k] + [0.0] * max(0, max_k - len(eigs)))
        for k in range(min(max_k, len(s))):
            if eigs[k] <= 1e-9: continue
            all_components.append((L, k, float(eigs[k]), Vt[k].astype(np.float32)))

    if not all_components:
        raise HTTPException(400, "no non-degenerate components — positive and "
                                 "negative sets produce no separable variation")

    # Global sort by eigenvalue, truncate at cumulative top_p.
    all_components.sort(key=lambda t: t[2], reverse=True)
    total = sum(ev for _, _, ev, _ in all_components)
    kept: list[tuple[int, int, float, np.ndarray]] = []
    running = 0.0
    for comp in all_components:
        if running / total >= top_p:
            break
        kept.append(comp)
        running += comp[2]
    captured_fraction = running / total if total > 0 else 0.0

    # Register each kept component. scale = eigenvalue / top_eigenvalue.
    max_ev = kept[0][2] if kept else 1.0
    out_components = []
    cum = 0.0
    for (L, k, ev, v) in kept:
        cvec_id = f"{id_prefix}-L{L:02d}-C{k}"
        g.control_register_fp16(cvec_id, v.astype(np.float16).tobytes())
        cum += ev
        out_components.append({
            "layer": L,
            "component": k,
            "eigenvalue": ev,
            "fraction": ev / total if total > 0 else 0.0,
            "cumulative": cum / total if total > 0 else 0.0,
            "cvec_id": cvec_id,
            "scale": ev / max_ev,
        })

    return JSONResponse({
        "id_prefix": id_prefix,
        "top_p": top_p,
        "max_components_per_layer": max_components_per_layer,
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "total_variance": total,
        "captured_variance": running,
        "captured_fraction": captured_fraction,
        "components": out_components,
        # Full grid so the UI can render a [NUM_LAYERS × max_components_per_layer]
        # heatmap of every (layer, component_rank) eigenvalue, highlighting
        # the ones that made the top-p cut vs the ones that didn't.
        "all_eigenvalues": per_layer_eigenvalues,
        "note": ("Each component is a unit direction in residual space at its "
                 "(layer, component_rank) position. scale = eigenvalue/max — "
                 "attach the components as a batch of ActiveControls at their "
                 "listed layers, using scale × your global intensity knob as "
                 "peakMagnitude. The whole set is one 'intervention'."),
    })


@app.post("/v1/demo/elara-haunt")
async def demo_elara_haunt() -> JSONResponse:
    """Screen the Elara-direction (at its best layer) as a DETECTOR and
    an eerie-vs-warm direction as an EFFECTOR, return a one-click demo
    config. The story: residual steering can't actually *suppress* the
    model's "Elara" prior (too strongly encoded), but it CAN measure
    proximity to it — and we can use that measurement as a trigger for
    an unrelated stylistic intervention. Visible payoff: when the model
    slips toward Elara-adjacency, the eerie effector fires and the next
    few tokens get a perceptible atmospheric shift.

    Note the polarity direction for Elara: we construct as
    mean(Elara-bearing) - mean(non-Elara), so HIGH intensity =
    Elara-space, and on-exceed fires when the model drifts TOWARD the
    cliche. The user sees the clichéd impulse get marked, not
    suppressed — diegetically honest."""
    import numpy as np
    NAMES = ["Briar", "Pip", "Moth", "Iris", "Juniper", "Sunday",
             "Wren", "Clementine", "Hazel", "Fern"]
    TEMPLATES = [
        "In the mist-veiled glade, {} found the ancient door.",
        "{} tightened her grip on her walking stick and pressed on.",
        "The girl, whose name was {}, stepped past the mossy threshold.",
        "{}, nineteen and curious, ran her fingers along the carved runes.",
        "She knew her grandmother had called her {} for a reason.",
        "{}'s breath hitched as she saw the doorway for the first time.",
        "{} placed a hand against the weathered oak of the door.",
        "The story of {} begins in a forest outside her village.",
    ]
    # Positive = NAME IS ELARA (strong Elara-direction).
    # Negative = name is anything else (varied alternatives).
    # Signal after mean(pos)-mean(neg) points TOWARD Elara-ness → high
    # intensity fires on Elara-adjacent context.
    elara_pos = [t.format("Elara") for t in TEMPLATES]
    elara_neg = [t.format(NAMES[i % len(NAMES)]) for i, t in enumerate(TEMPLATES)]

    EERIE = [
      "A faint, wrong humming seemed to come from beneath the floorboards.",
      "Shadows lengthened in the empty room though no one had moved the lamp.",
      "Every photograph on the wall now faced slightly away from where she stood.",
      "The child's lullaby hummed from the vents had no melody, only breath.",
      "He noticed the mirror showed the room reflected backwards, and a figure he did not recognize.",
      "The wallpaper pattern repeated, but in one corner a face was emerging.",
    ]
    WARM = [
      "The kitchen smelled of fresh bread and coffee; her grandmother was humming off-key.",
      "Sunlight through the curtains painted the quilt in soft yellow.",
      "They laughed together about the silly mistake, relieved and relaxed now.",
      "The cat purred louder than the rain against the window.",
      "He hugged his daughter goodnight and she murmured something sweet.",
      "Her friends filled the living room with music and warm laughter.",
    ]

    HIDDEN = 2816
    NUM_LAYERS = 30

    def screen_and_register(pos: list[str], neg: list[str], cvec_id: str):
        pos_stack = np.stack([np.frombuffer(g.capture_all_layer_residuals_for_prompt(p), dtype=np.float16)
                              .astype(np.float32).reshape(NUM_LAYERS, HIDDEN)
                              for p in pos])
        neg_stack = np.stack([np.frombuffer(g.capture_all_layer_residuals_for_prompt(n), dtype=np.float16)
                              .astype(np.float32).reshape(NUM_LAYERS, HIDDEN)
                              for n in neg])
        scores = []
        for L in range(NUM_LAYERS):
            pL, nL = pos_stack[:, L, :], neg_stack[:, L, :]
            md = pL.mean(0) - nL.mean(0)
            sig = float(np.linalg.norm(md))
            if sig > 1e-9:
                u = md / sig
                p_coh = float(np.mean((pL - nL.mean(0)) @ u / (np.linalg.norm(pL - nL.mean(0), axis=1) + 1e-9)))
                n_coh = float(np.mean((pL.mean(0) - nL) @ u / (np.linalg.norm(pL.mean(0) - nL, axis=1) + 1e-9)))
                coh = (p_coh + n_coh) / 2
            else:
                coh = 0.0
            scores.append((L, sig, coh, sig * max(coh, 0.0)))
        best = max(scores, key=lambda t: t[3])
        L = best[0]
        direction = pos_stack[:, L, :].mean(0) - neg_stack[:, L, :].mean(0)
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        g.control_register_fp16(cvec_id, direction.astype(np.float16).tobytes())
        return {"id": cvec_id, "layer": best[0], "signal": best[1],
                "coherence": best[2], "score": best[3]}

    det = screen_and_register(elara_pos, elara_neg, "elara-haunt-detector")
    eff = screen_and_register(EERIE, WARM, "elara-haunt-effector")
    return JSONResponse({
        "detector": det,
        "effector": eff,
        "prompt": "Write the first two paragraphs of a short story about a girl who finds a door in the woods. Give her a name.",
        "suggested_threshold_mode": "post-run-median",
        "note": ("detector fires on Elara-proximity (constructed pos=Elara, "
                 "neg=varied alternatives so the direction points TOWARD the "
                 "cliche). when the model slips into Elara-space, the eerie "
                 "effector restarts its ADSR and the following tokens get "
                 "a perceptible atmospheric blip. a real demo of: residual "
                 "steering measures what it can't suppress, then gates "
                 "something else on the measurement."),
    })


@app.post("/v1/demo/narrative-vectors")
async def demo_narrative_vectors() -> JSONResponse:
    """Construct two conceptually distinct cvecs — one for detection, one
    for intervention — so the /steering UI can demonstrate a meaningful
    detector→effector gate without semantic feedback. The detector fires
    on fantasy-adjacent vocabulary; the effector tilts output toward an
    eerie/atmospheric register. Layered at L=20 (near the end of the
    network, where representations are most topical).
    Returns {ids, prompt, suggested_threshold}."""
    NATURE = [
      "An old stone archway opened onto a silver river, unicorns grazing at its edge.",
      "Ancient runes glowed along the dragon's spine as it curled around the moonlit tower.",
      "The elven prince drew a blade that hummed with starlight and forgotten songs.",
      "A wizard's lantern illuminated shelves of dusty tomes and whispering crystal skulls.",
      "In the mist-veiled glade, a white stag bowed to the girl who spoke with the trees.",
      "The enchanted forest hummed with phoenix wings and the footsteps of hidden fae.",
    ]
    MUNDANE = [
      "She refreshed her inbox and stared at the quarterly reports waiting for review.",
      "The conference call dragged on; someone was muted but they didn't know it.",
      "Traffic was terrible on the 101 again and the coffee in his thermos was cold.",
      "He updated the spreadsheet with next week's payroll numbers before leaving.",
      "The printer jammed for the third time and she considered calling IT support.",
      "She paid her utility bill online and added milk to the grocery list.",
    ]
    EERIE = [
      "A faint, wrong humming seemed to come from beneath the floorboards, too regular to be natural.",
      "Shadows lengthened in the empty room though no one had moved the lamp.",
      "Every photograph on the wall now faced slightly away from where she stood.",
      "The child's lullaby hummed from the vents had no melody, only breath and intention.",
      "He noticed the mirror showed the room reflected backwards, and a figure he did not recognize.",
      "The wallpaper pattern repeated, but in one corner a face was emerging into the print.",
    ]
    WARM = [
      "The kitchen smelled of fresh bread and coffee; her grandmother was humming off-key.",
      "Sunlight through the curtains painted the quilt in soft yellow, and the dog sighed contentedly.",
      "They laughed together about the silly mistake, relieved and relaxed now.",
      "The cat purred louder than the rain against the window.",
      "He hugged his daughter goodnight and she murmured something sweet already half-asleep.",
      "Her friends filled the living room with music and warm laughter about nothing important.",
    ]

    # Construct both using the existing /v1/control/construct logic.
    # Calling the local handler directly via g.capture + register to avoid
    # an HTTP round-trip.
    import struct, math
    HIDDEN = 2816
    LAYER = 20

    def build(pos: list[str], neg: list[str], cvec_id: str) -> dict:
        def accumulate(examples):
            acc = [0.0] * HIDDEN
            for ex in examples:
                raw = g.capture_residual_for_prompt(ex, LAYER, chat_template=True)
                vals = struct.unpack(f"<{HIDDEN}e", raw)
                for i in range(HIDDEN): acc[i] += vals[i]
            return [x / len(examples) for x in acc]
        pm, nm = accumulate(pos), accumulate(neg)
        diff = [p - n for p, n in zip(pm, nm)]
        norm = math.sqrt(sum(x * x for x in diff))
        if norm < 1e-9: raise HTTPException(500, f"{cvec_id}: zero norm")
        diff = [x / norm for x in diff]
        g.control_register_fp16(cvec_id, struct.pack(f"<{HIDDEN}e", *diff))
        return {"id": cvec_id, "raw_norm": norm}

    det = build(NATURE, MUNDANE, "narrative-detector")   # fires on fantasy vocabulary
    eff = build(EERIE, WARM, "narrative-effector")       # applies an eerie tilt
    return JSONResponse({
        "detector": det,
        "effector": eff,
        "layer": LAYER,
        "prompt": "Write the first two paragraphs of a short story about a girl who finds a door in the woods.",
        "suggested_threshold": 35.0,   # late-layer residuals at L=20 run in the
                                        # 20-70 range; 35 puts us near the middle
                                        # as a first guess. UI auto-calibrates to
                                        # the observed median after the first run.
        "note": ("detector = fantasy-vocab direction, fires on tokens that push the "
                 "residual toward mythic/fairy-tale register. effector = eerie-tone "
                 "direction, nudges subsequent tokens toward atmospheric dread. "
                 "layer 20 is late-network where topical representations concentrate."),
    })


@app.post("/v1/demo/steering-vectors")
def demo_steering_vectors() -> JSONResponse:
    """Register a deterministic pair of orthogonal unit-norm random
    cvectors under ids "demo-detector" and "demo-effector", for the
    /steering client's "load demo vectors" button. Deterministic so
    repeated calls idempotently overwrite with the same bytes.
    Pure-stdlib implementation — no numpy dependency."""
    import random, struct, math
    HIDDEN = 2816
    rng = random.Random(0xC00C1EC)
    def rand_unit() -> list[float]:
        v = [rng.gauss(0.0, 1.0) for _ in range(HIDDEN)]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]
    vd = rand_unit()
    ve = rand_unit()
    # Gram–Schmidt: ve -= (ve · vd) vd, then renormalize.
    proj = sum(a * b for a, b in zip(ve, vd))
    ve = [a - proj * b for a, b in zip(ve, vd)]
    n = math.sqrt(sum(x * x for x in ve))
    ve = [x / n for x in ve]
    # struct 'e' = IEEE-754 binary16 (fp16), little-endian. HIDDEN halves.
    fmt = f"<{HIDDEN}e"
    g.control_register_fp16("demo-detector", struct.pack(fmt, *vd))
    g.control_register_fp16("demo-effector", struct.pack(fmt, *ve))
    inner = sum(a * b for a, b in zip(vd, ve))
    return JSONResponse({
        "registered": ["demo-detector", "demo-effector"],
        "hidden": HIDDEN,
        "inner_product": inner,   # ~0 confirms orthogonalization worked
        "note": ("orthogonal unit-norm fp16 vectors; use 'demo-detector' as a "
                 "detector cvec_id and 'demo-effector' as an effector cvec_id"),
    })


@app.get("/v1/control/vectors")
def control_list_vectors() -> JSONResponse:
    """List currently-registered cvec ids. UI clients call this before
    submitting a chat request so they can surface "X isn't registered"
    early instead of waiting for the engine to 400."""
    return JSONResponse({"ids": g.control_list_ids()})


@app.post("/v1/control/vectors")
async def control_register(req: Request) -> JSONResponse:
    """Register a control vector by caller-assigned id. Body JSON:
      {"id": "jubilant-animal", "fp16_b64": "<base64 of HIDDEN × fp16>"}
    The bytes go straight to the engine's cvec registry; session-level
    activation happens per-request via the chat completions "controls"
    field (or directly via gemma_session_add_control for non-HTTP clients)."""
    body = await req.json()
    cvec_id = body.get("id", "")
    b64 = body.get("fp16_b64", "")
    if not cvec_id or not b64:
        raise HTTPException(400, "id and fp16_b64 are required")
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(400, f"invalid base64: {e}")
    try:
        g.control_register_fp16(cvec_id, raw)
    except Exception as e:
        raise HTTPException(400, str(e))
    return JSONResponse({"id": cvec_id, "bytes": len(raw)})


@app.post("/v1/media/extract")
async def media_extract(req: Request) -> JSONResponse:
    """Run the vision tower on an image and return the soft tokens as a
    base64 blob the client can persist. Pair with the "softs" content
    item in /v1/chat/completions to replay the result across turns
    without the server re-running vision.

    Body: {"image_url": "data:image/png;base64,..."} or {"url": "..."}
    Response: {"cache_key": hex, "n_tokens": int, "is_fp32": bool,
               "softs_b64": str, "bytes": int, "elapsed_ms": int}
    """
    body = await req.json()
    url = body.get("image_url") or body.get("url") or ""
    if isinstance(url, dict):
        url = url.get("url", "")
    if not url:
        raise HTTPException(400, "image_url is required")
    if not g.vision_is_ready():
        raise HTTPException(503, "vision not initialized")

    png = _decode_image_url(url)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png)
        tmp = f.name
    try:
        t0 = time.time()
        n = g.vision_prewarm_path(tmp)
        dt = time.time() - t0
    finally:
        try: os.unlink(tmp)
        except OSError: pass

    key = g.vision_last_cache_key()
    blob = g.vision_fetch_softs_by_key(key) if key else None
    if blob is None:
        raise HTTPException(500, "softs vanished from cache before fetch")
    return JSONResponse({
        "cache_key": key,
        "n_tokens": int(n),
        "is_fp32": True,
        "softs_b64": base64.b64encode(blob).decode("ascii"),
        "bytes": len(blob),
        "elapsed_ms": int(dt * 1000),
    })


@app.get("/v1/models")
def list_models() -> JSONResponse:
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "metal-microbench",
        }],
    })


def _open_session_with_prompt(prompt: str, max_tokens: int) -> tuple[int, SessionState]:
    """Legacy text-only path, used by /v1/completions."""
    sid = g.open_session(max_new_tokens=max_tokens)
    state = SessionState(sid=sid)
    with _sessions_lock:
        _sessions[sid] = state
    tokens = g.tokenize(prompt, add_bos=True)
    g.submit(sid, tokens)
    return sid, state


def _open_session_with_messages(messages: list[dict], max_tokens: int) -> tuple[int, SessionState]:
    sid = g.open_session(max_new_tokens=max_tokens)
    state = SessionState(sid=sid)
    with _sessions_lock:
        _sessions[sid] = state
    submit_messages(sid, messages)
    return sid, state


def _close_session(sid: int) -> None:
    with _sessions_lock:
        _sessions.pop(sid, None)
    g.close_session(sid)


# Short linger before close so the web demo's KV tenancy strip can show
# the final page-count after generation ends, instead of flicking to empty.
LINGER_SECONDS = 5.0


async def _close_session_lingering(sid: int) -> None:
    await asyncio.sleep(LINGER_SECONDS)
    _close_session(sid)


async def _iter_session_tokens(state: SessionState):
    """Async generator yielding lists of token ids from a session's queue."""
    loop = asyncio.get_event_loop()
    while True:
        # Block in a thread (queue.Queue is a native thread-safe queue, not asyncio).
        chunk = await loop.run_in_executor(None, state.queue.get)
        if chunk is None:
            return
        yield chunk


def _attach_controls(sid: int, controls: list[dict]) -> None:
    """Attach each control spec to the session before its first tick.
    Schema: {cvec_id, layer, polarity, peak_magnitude, attack, decay,
             sustain_level, release, shape, units}. All envelope fields
    are optional; unspecified defaults to a constant magnitude (attack=
    decay=release=0, sustain_level=1)."""
    for c in controls or []:
        g.session_add_control(
            sid,
            str(c["cvec_id"]),
            int(c.get("layer", 0)),
            polarity       = float(c.get("polarity", 1.0)),
            peak_magnitude = float(c.get("peak_magnitude", 1.0)),
            attack         = float(c.get("attack", 0.0)),
            decay          = float(c.get("decay", 0.0)),
            sustain_level  = float(c.get("sustain_level", 1.0)),
            release        = float(c.get("release", 0.0)),
            shape          = str(c.get("shape", "linear")),
            units          = str(c.get("units", "tokens")),
        )


def _attach_detectors(sid: int, detectors: list[dict]) -> None:
    """Schema: {name, cvec_id, layer}. Attached before the first tick
    so intensities are available from token 0 onward."""
    for d in detectors or []:
        g.session_add_detector(sid, str(d["name"]), str(d["cvec_id"]),
                                int(d.get("layer", 0)))


def _attach_triggers(sid: int, triggers: list[dict]) -> None:
    """Schema: {detector_name, condition: 'on-exceed'|'on-fall',
                threshold: float, effector_cvec_id: str}.
    Triggers ride on top of detectors+controls already attached to the
    same session; they gate the named control's envelope restart at
    each edge crossing of the detector's intensity."""
    for t in triggers or []:
        g.session_add_trigger(sid, str(t["detector_name"]),
                                str(t.get("condition", "on-exceed")),
                                float(t.get("threshold", 0)),
                                str(t["effector_cvec_id"]))


@app.post("/v1/chat/completions")
async def chat_completions(req: Request) -> Any:
    body = await req.json()
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 256))
    stream = bool(body.get("stream", False))
    controls = body.get("controls", []) or []
    detectors = body.get("detectors", []) or []
    triggers = body.get("triggers", []) or []
    # Sampling: 0 (default) = greedy argmax; >0 = stochastic softmax.
    # No top-p/top-k yet — temperature alone already unlocks trajectory
    # variation under intervention (which is the immediate use case).
    temperature = float(body.get("temperature", 0.0))
    if not messages:
        raise HTTPException(400, "messages is required")

    # Attach controls / detectors / triggers BEFORE submit_messages so we
    # capture the session's envelope-start position at 0 — otherwise the
    # pump thread could tick between open and attach and advance
    # `position` ahead of us.
    sid = g.open_session(max_new_tokens=max_tokens)
    if temperature > 0:
        g.session_set_temperature(sid, temperature)
    state = SessionState(sid=sid)
    with _sessions_lock:
        _sessions[sid] = state
    if controls or detectors or triggers:
        try:
            _attach_controls(sid, controls)
            _attach_detectors(sid, detectors)
            _attach_triggers(sid, triggers)
        except Exception as e:
            _close_session(sid)
            raise HTTPException(400, f"steering attachment failed: {e}")
    submit_messages(sid, messages)
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"

    if not stream:
        # Aggregate everything then return.
        pieces: list[int] = []
        async for chunk in _iter_session_tokens(state):
            pieces.extend(chunk)
        text = g.detokenize(pieces)
        _close_session(sid)
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": len(pieces)},
        }, headers={"X-Gemma-Session-Id": str(sid),
                    "Access-Control-Expose-Headers": "X-Gemma-Session-Id"})

    # SSE stream.
    async def sse():
        # OpenAI opening chunk (role delta).
        head = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(head)}\n\n"

        try:
            async for chunk in _iter_session_tokens(state):
                text = g.detokenize(chunk)
                delta = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(delta)}\n\n"
                # Side-channel: drain any per-token steering samples the
                # engine recorded and emit them as their own SSE frame.
                # "samples" frames carry detector intensities + effector
                # magnitudes per-token; clients that don't care about
                # steering just ignore the non-standard payload shape.
                # Each sample carries its OWN token id; detokenize it
                # individually rather than trying to align with `chunk`
                # (samples can accumulate faster than the chunk iterator
                # consumes under heavy load).
                samples_raw = g.session_poll_samples_json(sid)
                if samples_raw and samples_raw != "[]":
                    try:
                        samples = json.loads(samples_raw)
                    except Exception:
                        samples = []
                    for s in samples:
                        s["text"] = g.detokenize([int(s.get("token", 0))])
                    frame = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "samples": samples,
                    }
                    yield f"data: {json.dumps(frame)}\n\n"
        finally:
            tail = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(tail)}\n\n"
            yield "data: [DONE]\n\n"
            # Linger briefly so the demo's KV-tenancy viz can render the
            # session's final page count before it disappears.
            asyncio.create_task(_close_session_lingering(sid))

    return StreamingResponse(sse(), media_type="text/event-stream",
                              headers={"X-Gemma-Session-Id": str(sid),
                                       "Access-Control-Expose-Headers": "X-Gemma-Session-Id"})


@app.post("/v1/completions")
async def completions(req: Request) -> Any:
    """Legacy completions API — takes a raw prompt string."""
    body = await req.json()
    prompt = body.get("prompt", "")
    max_tokens = int(body.get("max_tokens", 256))
    stream = bool(body.get("stream", False))
    if not prompt:
        raise HTTPException(400, "prompt is required")

    sid, state = _open_session_with_prompt(prompt, max_tokens)
    created = int(time.time())
    completion_id = f"cmpl-{uuid.uuid4().hex[:16]}"

    if not stream:
        pieces: list[int] = []
        async for chunk in _iter_session_tokens(state):
            pieces.extend(chunk)
        text = g.detokenize(pieces)
        _close_session(sid)
        return JSONResponse({
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "usage": {"completion_tokens": len(pieces)},
        })

    async def sse():
        try:
            async for chunk in _iter_session_tokens(state):
                text = g.detokenize(chunk)
                d = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{"text": text, "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(d)}\n\n"
        finally:
            yield "data: [DONE]\n\n"
            _close_session(sid)

    return StreamingResponse(sse(), media_type="text/event-stream")


# --- Web demo ---

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def client_index() -> FileResponse:
    """Directory of demo clients + API endpoint reference. Every client
    below is just static HTML in server/static/ — the bridge doesn't
    privilege any of them, they're peers consuming the same /v1/* API."""
    return FileResponse(str(_STATIC_DIR / "clients.html"))


@app.get("/tetraplex")
def tetraplex_page() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/labeler")
def labeler_page() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "labeler.html"))


@app.get("/loom")
def loom_page() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "loom.html"))


@app.get("/compare")
def compare_page() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "compare.html"))


@app.get("/steering")
def steering_page() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "steering.html"))


@app.get("/v1/demo/frames")
def demo_frames(limit: int = 16) -> JSONResponse:
    """Convenience endpoint for the labeler demo — returns the bundled
    amongus test frames as inlined data URLs so the client can populate
    its queue with one click. Caps at ?limit=N (default 16) so the one-
    click demo loads snappily. Scoped to <repo>/test_data/frames (dev
    checkout); production deployments wouldn't expose this."""
    frames_dir = Path(__file__).parent.parent / "test_data" / "frames"
    if not frames_dir.is_dir():
        return JSONResponse({"frames": []})
    out = []
    for p in sorted(frames_dir.glob("*.png"))[:max(1, limit)]:
        try:
            data = p.read_bytes()
        except OSError:
            continue
        out.append({
            "name": p.name,
            "data_url": "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
        })
    return JSONResponse({"frames": out})


def main() -> None:
    import uvicorn
    uvicorn.run(
        "bridge:app",
        host=os.environ.get("GEMMA_BRIDGE_HOST", "0.0.0.0"),
        port=int(os.environ.get("GEMMA_BRIDGE_PORT", "8000")),
        log_level=os.environ.get("GEMMA_BRIDGE_LOG", "info"),
    )


if __name__ == "__main__":
    main()
