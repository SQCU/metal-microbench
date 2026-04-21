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


@app.post("/v1/chat/completions")
async def chat_completions(req: Request) -> Any:
    body = await req.json()
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 256))
    stream = bool(body.get("stream", False))
    if not messages:
        raise HTTPException(400, "messages is required")

    sid, state = _open_session_with_messages(messages, max_tokens)
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
def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/labeler")
def labeler_page() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "labeler.html"))


@app.get("/loom")
def loom_page() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "loom.html"))


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
