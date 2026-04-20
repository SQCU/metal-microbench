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
        })

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
            _close_session(sid)

    return StreamingResponse(sse(), media_type="text/event-stream")


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
