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
import json
import os
import queue
import threading
import time
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
    print(f"[bridge] ready in {time.time() - t0:.2f}s (bos={g.bos_id()}, eos={g.eos_id()})")
    _pump_running = True
    _pump_thread = threading.Thread(target=_pump_loop, name="gemma-pump", daemon=True)
    _pump_thread.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    global _pump_running
    _pump_running = False


# --- Chat template ---

GEMMA_TURN_OPEN = "<|turn>"
GEMMA_TURN_CLOSE = "<turn|>"


def render_messages(messages: list[dict]) -> str:
    """messages → Gemma chat-format string, ending with the model open-turn.

    OpenAI format: [{role: user/assistant/system, content: str}, ...]
    Gemma format:  <|turn>role\\nCONTENT<turn|>\\n<|turn>role\\n...
    Always emit a trailing '<|turn>model\\n' to prime generation.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            # OpenAI vision-style content: list of {type, text|image_url}.
            # For now collapse to text only.
            content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        parts.append(f"{GEMMA_TURN_OPEN}{role}\n{content}{GEMMA_TURN_CLOSE}\n")
    parts.append(f"{GEMMA_TURN_OPEN}model\n")
    return "".join(parts)


# --- Endpoints ---

@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ready" if g.is_ready() else "loading",
        "model": MODEL_NAME,
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
    sid = g.open_session(max_new_tokens=max_tokens)
    state = SessionState(sid=sid)
    with _sessions_lock:
        _sessions[sid] = state
    tokens = g.tokenize(prompt, add_bos=True)
    g.submit(sid, tokens)
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

    prompt = render_messages(messages)
    sid, state = _open_session_with_prompt(prompt, max_tokens)
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
