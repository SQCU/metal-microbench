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


GGUF_PATH = os.environ.get(
    "GGUF_PATH",
    "/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
)
VISION_SAFETENSORS = os.environ.get(
    "VISION_ST",
    "/Users/mdot/models/gemma-4-a4b-bf16/model-00001-of-00002.safetensors",
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
    print(f"[bridge] loading {GGUF_PATH}")
    t0 = time.time()
    g.init(GGUF_PATH)
    print(f"[bridge] LM ready in {time.time() - t0:.2f}s (bos={g.bos_id()}, eos={g.eos_id()})")

    # Vision weights are optional — only loaded if the safetensors file
    # exists at the configured path. Without them, /v1/chat/completions
    # with image_url content items will 400.
    if Path(VISION_SAFETENSORS).exists():
        t1 = time.time()
        try:
            g.vision_init(VISION_SAFETENSORS)
            print(f"[bridge] vision ready in {time.time() - t1:.2f}s")
        except Exception as e:
            print(f"[bridge] vision init failed: {e}")
    else:
        print(f"[bridge] vision safetensors not at {VISION_SAFETENSORS} — image_url disabled")

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
def kv_snapshot() -> JSONResponse:
    """Per-session page ownership + per-page refcount. Clients poll this
    during generation to render a live tenancy strip — pages with refcount>1
    are shared across sessions (prefix-cache hit or explicit adoptKvFrom).
    """
    sids = g.active_session_ids()
    sessions = []
    for sid in sids:
        snap = g.session_snapshot(sid)
        if not snap:
            continue
        # Annotate each page with (refcount, other_owner_sids).
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
    if not messages:
        raise HTTPException(400, "messages is required")

    # Attach controls / detectors / triggers BEFORE submit_messages so we
    # capture the session's envelope-start position at 0 — otherwise the
    # pump thread could tick between open and attach and advance
    # `position` ahead of us.
    sid = g.open_session(max_new_tokens=max_tokens)
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
