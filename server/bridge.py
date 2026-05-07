"""FastAPI bridge: OpenAI-compatible HTTP → Swift Metal engine.

Architecture (notes/specs/batch_ffi_abi.md, notes/spec_vs_inductions.md):
  - Bridge is a dumb HTTP queue + dispatch + stream + billing layer.
  - All FFI calls go through ONE coordinator coroutine. Concurrent
    HTTP handlers enqueue StreamSpec; the coordinator drains the
    queue and calls g.submit() with everything pending — that's what
    makes the backend's in-batch shared-prefix detection actually fire
    on the curriculum's concurrent rollouts.
  - There are no per-session FFI calls. There is no engine-state lock
    (the coordinator IS the serialization point). There is no `Session`
    object on the Python side; the bridge tracks streams by id.

Endpoints:
  GET  /health                 — liveness + engine stats
  GET  /v1/models              — list (one entry)
  POST /v1/chat/completions    — OpenAI-compatible chat (stream or not)

Research endpoints (control vectors, structured grammars, layer dumps,
etc.) live as parameters on the unified interface — they are fields
on StreamSpec / SamplingParams, not separate FFI calls. The bridge
will surface them through /v1/chat/completions body fields once their
backend support lands. See notes/specs/batch_ffi_abi.md "Open
questions" for the schema.

Run:
  uv run uvicorn bridge:app --host 0.0.0.0 --port 8000 --log-level info
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import gemma_ffi as g

from chat_template import (
    TextChunk, ImageChunk, render_chat,
    tokenize_with_specials,
)
# 2026-05-07: stripped ConversationCache + warm-path adoption per the
# 'NO REMOTE LOCKS / chat completions stateless' principle. The bridge
# no longer remembers any state across requests — every chat-completion
# render runs canonically from the supplied `messages` list.
# Previous module: conversation_state.py (kept for now; dead code).
from conversation_state import StoredSegment
# NOTE on tool calls: the bridge does NOT parse tool-call markers out
# of model output. The model emits whatever bytes it emits (including
# `<|tool_call>...<tool_call|>` markers when tools are present in the
# request); the response content carries those bytes verbatim. Clients
# that want OAI-shape `tool_calls` parse them themselves — that work
# is application-level interpretation, not tensor service work, and
# putting it on the bridge's response-latency path means every chat
# completion paid synchronous CPU for parsing the model's text against
# a regex even when no tool was ever requested. The toolcards runner
# already has its own dispatch logic; if a chat client wants OAI tool
# shape it should run a tiny proxy that does the regex extraction
# client-side.


# ----------------------------------------------------------------------
# Config.
# ----------------------------------------------------------------------
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
MODEL_NAME = os.environ.get("GEMMA_MODEL_NAME", "gemma-4-a4b")


# ----------------------------------------------------------------------
# Coordinator state.
#
# 2026-05-07: ONE dedicated native threading.Thread owns the entire
# gemma_poll driver loop for the lifetime of the process. HTTP handlers
# call gemma_submit DIRECTLY (it's thread-safe — Swift-side
# gIntakeCond.lock+append+signal+unlock, microsecond fast path).
# Updates fan from the driver thread to per-stream asyncio.Queues via
# loop.call_soon_threadsafe(rq.put_nowait, u).
#
# Why this replaces the old "two asyncio tasks each call to_thread" pattern:
#   - The old pattern paid an asyncio→threadpool→cgo round-trip per
#     submit AND per poll cycle (~5-10 ms each). At ~12 AR ticks/sec
#     that was ~120-240 ms/sec of pure host-side overhead between
#     productive CBs.
#   - The old pattern used TWO threadpool workers concurrently calling
#     into the FFI (submit + poll). Even with internal Swift-side
#     locking that was correct, it doubled cgo entry overhead.
#   - sched_sim_bridge's P8 sensitivity sweep showed the dominant
#     unmodeled cost in the 57-vs-77-tok/s gap is exactly threadpool
#     dispatch + per-update asyncio scheduling. Native owner thread
#     predicted 80+ tok/s.
#
# Wakeup story: when the engine is idle, gemma_poll cond_waits on
# gIntakeCond inside Swift. HTTP handlers calling gemma_submit signal
# that cond, waking the driver thread within microseconds — so cold-
# start latency is signal latency, not the driver's poll deadline.
# ----------------------------------------------------------------------
import threading

_response_qs: dict[int, "asyncio.Queue[g.StreamUpdate]"] = {}
_next_stream_id_lock: "asyncio.Lock | None" = None
_next_stream_id = 1
_engine_thread: "threading.Thread | None" = None
_engine_thread_stop = threading.Event()
_engine_loop: "asyncio.AbstractEventLoop | None" = None
_TOOL_CALL_CLOSE_TOKENS: list[int] = []  # set at startup
_TURN_END_TOKENS: list[int] = []         # set at startup; chat-template end-of-turn

# Engine-driver poll deadline. 1000 ms means gemma_poll's drive loop
# runs many CBs per native-thread iteration without ever returning to
# Python; cgo-entry overhead is amortized to ~1/sec. The deadline only
# bounds how long gemma_poll cond_waits when the engine is TRULY idle
# (intake empty, no resident sessions decoding); active work runs back-
# to-back regardless of the deadline.
_DRIVER_POLL_DEADLINE_MS = 1000

# 2026-05-07: deleted _conv_cache. The bridge is now stateless across
# chat-completion requests — no warm-path adoption, no
# message-hash-keyed memory, no LRU. Engine-side content-hash KV page
# adoption (page_manager.contentIndex) still benefits multi-turn chat
# at the engine layer where it's a passive accelerator (no client-side
# entanglement); the bridge no longer constructs request-shape state
# across requests.


async def _next_stream_id_alloc() -> int:
    global _next_stream_id
    async with _next_stream_id_lock:  # type: ignore[arg-type]
        sid = _next_stream_id
        _next_stream_id += 1
        return sid


def _push_update_threadsafe(loop: asyncio.AbstractEventLoop,
                              update: "g.StreamUpdate") -> None:
    """Schedule `response_q.put_nowait(update)` on the asyncio loop from
    the native driver thread. asyncio.Queue.put_nowait is itself
    thread-safe against the get-side; call_soon_threadsafe handles the
    cross-thread loop wakeup. No await chain, no to_thread."""
    rq = _response_qs.get(update.stream_id)
    if rq is None:
        return
    try:
        loop.call_soon_threadsafe(rq.put_nowait, update)
    except RuntimeError:
        # Loop closed during shutdown — drop the update silently.
        pass


def _engine_driver_thread(loop: asyncio.AbstractEventLoop) -> None:
    """Single-owner FFI driver thread. Lives for the process lifetime.
    Tight-loops gemma_poll (long deadline; engine self-yields to drive
    existing work without burning the deadline). Each productive CB
    inside gemma_poll fans its updates to per-stream asyncio.Queues
    via loop.call_soon_threadsafe.

    HTTP handlers call gemma_submit on their own thread (asyncio loop
    thread); the Swift-side gIntakeCond.signal wakes this thread out
    of cond_wait if engine was idle, so cold-start latency is
    cond-signal speed (microseconds), not the deadline.
    """
    print("[engine_driver] thread started "
          f"(poll_deadline={_DRIVER_POLL_DEADLINE_MS}ms)", flush=True)
    iters = 0
    while not _engine_thread_stop.is_set():
        try:
            updates = g.poll(_DRIVER_POLL_DEADLINE_MS)
            iters += 1
            if updates:
                for u in updates:
                    _push_update_threadsafe(loop, u)
        except Exception as e:
            print(f"[engine_driver] error iter={iters}: {e}", flush=True)
            _engine_thread_stop.wait(timeout=0.05)
    print(f"[engine_driver] exiting after {iters} iterations", flush=True)


# ----------------------------------------------------------------------
# OpenAI sampling-param parsing. Translates the OpenAI-shaped body
# fields into a g.SamplingParams.
# ----------------------------------------------------------------------
def _parse_sampling(body: dict, max_tokens: int) -> g.SamplingParams:
    temperature = float(body.get("temperature", 0.0))
    top_p = float(body.get("top_p", 1.0))
    top_k = int(body.get("top_k", 0))
    seed = body.get("seed")
    seed_int = int(seed) if seed is not None else 0
    rep_pen = float(body.get("repetition_penalty",
                              body.get("frequency_penalty", 1.0)))
    stop = body.get("stop")
    stop_tokens: list[int] = []
    if isinstance(stop, list):
        # OpenAI's `stop` is text strings; we'd need to tokenize each
        # to convert. For now, leave empty; the engine's actual stop
        # signal for chat-completions is configured via
        # `sampling.stop_sequences` in the chat_completions handler
        # below (the engine consumes stop_sequences but ignores
        # stop_tokens — see ffi_batch.swift:277 vs lm_engine.swift's
        # AR loop).
        pass
    capture_logits = bool(body.get("logprobs", False))
    top_logprobs = int(body.get("top_logprobs", 0))
    # OpenAI sends logit_bias as {"<token_id_str>": float}; coerce keys.
    raw_lb = body.get("logit_bias") or {}
    logit_bias = {int(k): float(v) for k, v in raw_lb.items()}
    return g.SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=rep_pen,
        max_new_tokens=max_tokens,
        seed=seed_int,
        eos_token_id=g.eos_id(),
        stop_tokens=stop_tokens,
        top_logprobs=top_logprobs,
        logit_bias=logit_bias,
    ), capture_logits


def _build_stream_spec(stream_id: int,
                       messages: list[dict],
                       sampling: g.SamplingParams,
                       capture_logits: bool,
                       tools: list | None = None) -> tuple[g.StreamSpec, list[StoredSegment]]:
    """OpenAI messages → (StreamSpec, submitted_text_tokens).

    Two paths:
      * Warm-conversation: messages[:-1] hashes to a known prior turn's
        prefix tokens. Build submission as
        `prior_prefix_tokens + tokenize(render_user_turn_delta(messages[-1]))`,
        bypassing canonical re-render of historical content.
      * Cold: canonical `render_chat()` + `tokenize_with_specials()`,
        same as before.

    The returned `submitted_text_tokens` is the flat sequence of token
    IDs across all `kind=0` (text) segments — image / softs segments
    contribute their own tokens at the engine level, which we don't
    attempt to track here. Conversation-state recording therefore runs
    only for purely text turns; mixed-modality turns fall through to
    canonical for now (`record_state=False`).
    """
    # 2026-05-07: bridge is stateless across chat completions per the
    # 'NO REMOTE LOCKS / no entanglement' principle. Each request renders
    # canonically from `messages` — no warm-path adoption from a stored
    # prior conversation-state. The engine's content-hash KV page cache
    # at the page_manager layer still produces multi-turn KV reuse for
    # bit-identical prefix bytes (passive accelerator, no client-side
    # state); but the bridge itself constructs no cross-request state.
    try:
        delta_chunks = render_chat(
            messages, add_generation_prompt=True, tools=tools)
    except ValueError as e:
        # image_url that isn't a data: URI is a client-shape error,
        # not a server fault.
        if "image_url" in str(e):
            raise HTTPException(400, str(e)) from None
        raise

    delta_segments = _chunks_to_segments(delta_chunks, add_bos=True)
    spec_segments = [
        g.Segment(kind=s.kind, tokens=list(s.tokens), image_bytes=s.image_bytes)
        for s in delta_segments
    ]
    return g.StreamSpec(
        stream_id=stream_id, action=0,
        flags=0x01 if capture_logits else 0,
        segments=spec_segments, sampling=sampling,
    ), delta_segments


def _chunks_to_segments(chunks, *, add_bos: bool) -> list[StoredSegment]:
    """Convert TextChunk/ImageChunk list into StoredSegments.
    Consecutive TextChunks coalesce so the engine sees a stable
    boundary structure independent of how the chunk source split text.
    """
    segments: list[StoredSegment] = []
    pending: list[int] = []
    did_bos = not add_bos

    def flush() -> None:
        nonlocal did_bos
        if pending:
            segments.append(StoredSegment(kind=0, tokens=list(pending)))
            pending.clear()
            did_bos = True

    for ch in chunks:
        if isinstance(ch, TextChunk):
            toks = tokenize_with_specials(
                ch.text, tokenize_fn=g.tokenize, add_bos=(not did_bos))
            if toks:
                pending.extend(toks)
                did_bos = True
        elif isinstance(ch, ImageChunk):
            flush()
            segments.append(StoredSegment(kind=1, image_bytes=ch.data))
    flush()
    return segments


# 2026-05-07: deleted _warm_path_eligible. Bridge no longer carries
# any cross-request conversation state to inherit from.


# ----------------------------------------------------------------------
# FastAPI app + lifecycle.
# ----------------------------------------------------------------------
app = FastAPI(title="Gemma Metal Bridge", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    global _next_stream_id_lock

    if not GGUF_PATH or not Path(GGUF_PATH).exists():
        raise RuntimeError(
            f"GGUF weights not found at {GGUF_PATH!r}. Set GEMMA_GGUF=/path/to/.gguf "
            f"or run server/scripts/fetch-weights.py to download.")

    print(f"[bridge] loading {GGUF_PATH}", flush=True)
    t0 = time.time()
    rc = g.init(GGUF_PATH)
    if rc != 0:
        raise RuntimeError(f"gemma_init returned {rc}")
    print(f"[bridge] LM ready in {time.time() - t0:.2f}s "
          f"(bos={g.bos_id()}, eos={g.eos_id()})", flush=True)

    if VISION_SAFETENSORS and Path(VISION_SAFETENSORS).exists():
        t1 = time.time()
        rc = g.vision_init(VISION_SAFETENSORS)
        if rc == 0:
            print(f"[bridge] vision ready in {time.time() - t1:.2f}s", flush=True)
        else:
            print(f"[bridge] vision_init returned {rc}", flush=True)
    else:
        print(f"[bridge] vision disabled (no safetensors path)", flush=True)

    # Pre-tokenize Gemma-4's tool-call close marker. Used as a stop
    # sequence on every chat-completion that ships tools[]: as soon as
    # the model emits <tool_call|> the engine self-terminates with
    # done_reason=1 (eos-equivalent), no wasted decode of the
    # downstream <|tool_response> spam.
    global _TOOL_CALL_CLOSE_TOKENS
    _TOOL_CALL_CLOSE_TOKENS = list(g.tokenize("<tool_call|>", add_bos=False))
    print(f"[bridge] tool_call close tokens: {_TOOL_CALL_CLOSE_TOKENS}", flush=True)

    # Gemma-4's chat-template end-of-turn special token. The chat
    # template emits `<end_of_turn>` (special-token id 106 in the
    # Gemma vocabulary, which renders as the literal text `<turn|>`
    # when decoded) to delimit every turn. The bridge MUST treat 106
    # as a stop signal — without it, the model continues past its
    # natural turn boundary and emits auxiliary scaffolding (a
    # `thought` channel, secondary turns, `<channel|>` blocks) that
    # leaks into the response content.
    #
    # GGUF metadata's `eos_token_id` field is unreliable for this:
    # the Q4_K_M GGUF reports eos=106 (correct, == <end_of_turn>),
    # but the fp16 GGUF reports eos=1 (bare <eos>, NOT the chat-
    # template end-of-turn). Hardcoding token 106 here makes the
    # bridge respect the chat template's actual turn boundary
    # regardless of which token the GGUF metadata happens to claim
    # as "EOS". Token id is stable across all Gemma 1/2/3/4 vocabs.
    #
    # Note: tokenizing the literal STRING "<turn|>" via g.tokenize
    # does NOT yield 106 — it yields the BPE breakdown of those 8
    # ASCII chars as ordinary tokens. We want the special token id
    # the model actually emits, which is 106.
    global _TURN_END_TOKENS
    _TURN_END_TOKENS = [106]
    print(f"[bridge] chat-template end-of-turn tokens: {_TURN_END_TOKENS} "
          f"(<end_of_turn>)", flush=True)

    _next_stream_id_lock = asyncio.Lock()

    # 2026-05-07: spawn the native FFI driver thread. This replaces
    # the old _submit_pump + _poll_pump asyncio tasks. HTTP handlers
    # call g.submit() directly (thread-safe via gIntakeCond); the
    # driver thread tight-loops g.poll() and fans updates via
    # loop.call_soon_threadsafe. No to_thread overhead on the hot path.
    global _engine_thread, _engine_loop
    _engine_loop = asyncio.get_running_loop()
    _engine_thread_stop.clear()
    _engine_thread = threading.Thread(
        target=_engine_driver_thread, args=(_engine_loop,),
        name="gemma-engine-driver", daemon=True)
    _engine_thread.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _engine_thread
    _engine_thread_stop.set()
    if _engine_thread is not None:
        # Driver thread loops until stop flag is set; gemma_poll's
        # built-in deadline (_DRIVER_POLL_DEADLINE_MS, default 1s)
        # bounds the wait if it's currently inside a cond_wait.
        _engine_thread.join(timeout=2.0)
        if _engine_thread.is_alive():
            print("[bridge] engine_driver did not exit within 2s "
                  "of shutdown signal; continuing shutdown anyway",
                  flush=True)
        _engine_thread = None
    g.shutdown()


# ----------------------------------------------------------------------
# Endpoints.
# ----------------------------------------------------------------------
@app.get("/health")
def health() -> JSONResponse:
    s = g.status()
    return JSONResponse({
        "status": "ready",
        "model": MODEL_NAME,
        "multimodal": g.vision_is_ready(),
        "active_streams": s.active_streams,
        "cached_pages": s.cached_pages,
        "free_pages": s.free_pages,
        "vision_cache_entries": s.vision_cache_entries,
        "vision_cache_hits": s.vision_cache_hits,
        "total_steps": s.total_steps,
        "total_tokens_emitted": s.total_tokens_emitted,
        "capabilities": {
            "max_q_len": g.max_q_len(),
        },
    })


@app.post("/v1/tokenize")
async def tokenize_endpoint(req: Request) -> JSONResponse:
    """Stateless tokenizer call. Returns token IDs for `input` text plus
    optional BOS prepend.

    `g.tokenize` is synchronous CPU work calling into the Swift FFI; we
    run it in a thread so high-volume callers (perplexity-stride scans,
    HellaSwag pre-tokenization, etc.) don't pin the asyncio event loop
    and starve concurrent chat-completion / teacher-forced traffic.
    """
    body = await req.json()
    text = body.get("input", "")
    add_bos = bool(body.get("add_bos", False))
    tokens = await asyncio.to_thread(
        lambda: list(g.tokenize(str(text), add_bos=add_bos)))
    return JSONResponse({"tokens": tokens, "n_tokens": len(tokens)})


def _models_payload() -> dict:
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "metal-microbench",
        }],
    }


@app.get("/v1/models")
def list_models_v1() -> JSONResponse:
    return JSONResponse(_models_payload())


# Compatibility alias: clients that set their base URL to the bridge
# host without `/v1` end up requesting `/models` directly. Mirroring it
# avoids a confusing 404 during initial config.
@app.get("/models")
def list_models_compat() -> JSONResponse:
    return JSONResponse(_models_payload())


# Compatibility alias: same as /v1/chat/completions for clients whose
# base URL omits the `/v1` prefix. Stacked decorators register both paths
# on the same handler.
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(req: Request) -> Any:
    body = await req.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(400, "messages is required")
    max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 256)))
    stream = bool(body.get("stream", False))

    # Research-feature body fields that don't yet have unified-FFI
    # equivalents (detectors / triggers / prefix_softs / prefix_kv).
    # Each is planned as a `StreamSpec` extension; until then, 501.
    for k in ("detectors", "triggers", "prefix_softs", "prefix_kv"):
        if body.get(k):
            raise HTTPException(
                501, f"'{k}' not yet wired through the unified FFI; "
                     f"see notes/specs/batch_ffi_abi.md")
    # `controls` (control vectors) is now SamplingParams-shape via
    # gemma_register_resource + StreamSpec.control_vectors. The
    # curriculum doesn't send `controls`, but if a research client does
    # we'd translate body["controls"] here. For now: 501 with pointer
    # to the unified shape — research callers should use the new
    # ABI directly once the bridge plumbing lands.
    if body.get("controls"):
        raise HTTPException(
            501, "'controls' is now StreamSpec.control_vectors via the "
                 "unified ABI. Bridge translation TBD; see notes/specs/"
                 "batch_ffi_abi.md")

    sampling, capture_logits = _parse_sampling(body, max_tokens)
    # structured_cot: bool → default labels, list → custom labels.
    sc = body.get("structured_cot")
    if sc is True:
        sampling.cot_labels = ["GOAL", "APPROACH", "EDGE"]
    elif isinstance(sc, list) and sc:
        sampling.cot_labels = [str(x).strip() for x in sc if str(x).strip()]
    # OpenAI tool calling: tools[] → injected via chat template's native
    # <|tool>declaration:...<tool|> blocks. tool_choice is currently
    # advisory; the model decides. (Forced tool selection / "required" /
    # named-function modes are not yet wired through.)
    tools = body.get("tools") if isinstance(body.get("tools"), list) else None
    # Engine-side stop_sequences are matched against the recently-
    # emitted token tail; when any sequence matches, done_reason=1
    # fires and the AR loop terminates. We always include the
    # chat-template end-of-turn ([106] = <end_of_turn>) so the model
    # stops at its natural turn boundary instead of bleeding into
    # auxiliary scaffolding (thought channel, secondary turns)
    # that the bridge would otherwise expose as response content.
    # When tools are present we also include the tool_call_close
    # multi-token sequence so the engine self-terminates on
    # <tool_call|> rather than burning 4096 tokens waiting for an
    # injected <|tool_response>.
    sampling.stop_sequences = [list(_TURN_END_TOKENS)]
    if tools and _TOOL_CALL_CLOSE_TOKENS:
        sampling.stop_sequences.append(list(_TOOL_CALL_CLOSE_TOKENS))
    print(f"[bridge] chat_completions: tools={len(tools) if tools else 0}, "
          f"tool_choice={body.get('tool_choice')!r}, "
          f"messages={len(messages)}, stream={stream}, "
          f"stop_seqs={len(sampling.stop_sequences)}", flush=True)
    stream_id = await _next_stream_id_alloc()
    spec, _ = _build_stream_spec(
        stream_id, messages, sampling, capture_logits, tools=tools)
    response_q: asyncio.Queue = asyncio.Queue()
    _response_qs[stream_id] = response_q

    # 2026-05-07: removed conversation-state recording. Bridge is
    # stateless across requests; the engine's page-cache is the only
    # cross-request memory and operates without client-side coupling.

    # Direct gemma_submit. Thread-safe via Swift-side gIntakeCond
    # (briefly takes that lock, appends to gIntakeQueue, signals,
    # unlocks — microseconds). The signal wakes the engine_driver
    # thread if it's currently in gemma_poll's cond_wait branch
    # (engine-idle), so cold-start submission latency is signal
    # speed, not the driver's poll deadline.
    rc = g.submit([spec])
    if rc != 0:
        _response_qs.pop(stream_id, None)
        raise HTTPException(500, f"engine submit failed: rc={rc}")

    if not stream:
        # Aggregate path: handler awaits the entire response, so handler-
        # level finally is correct.
        clean_close_aggregate = False
        try:
            all_tokens: list[int] = []
            usage: dict[str, int] = {}
            done_reason = 0
            collected_logprobs: list[g.TokenLogprob] = []
            while True:
                u: g.StreamUpdate = await response_q.get()
                all_tokens.extend(u.new_tokens)
                if u.logprobs:
                    collected_logprobs.extend(u.logprobs)
                # The engine now self-terminates on the <tool_call|>
                # token sequence (see SamplingParams.stop_sequences,
                # populated above when tools are present). No bridge-
                # side break-early needed: state==2 fires naturally with
                # done_reason=1.
                if u.state == 2:
                    usage = {
                        "prompt_tokens": u.prompt_tokens_seen,
                        "completion_tokens": u.completion_tokens_emitted,
                        "total_tokens": u.prompt_tokens_seen + u.completion_tokens_emitted,
                        "cache_hits": u.cache_hits,
                        "cache_misses": u.cache_misses,
                        "vision_cache_hits": u.vision_cache_hits,
                    }
                    done_reason = u.done_reason
                    break
            text = g.detokenize(all_tokens)
            print(f"[bridge] usage: prompt_tokens={usage.get('prompt_tokens')}, "
                  f"completion_tokens={usage.get('completion_tokens')}, "
                  f"cache_hits={usage.get('cache_hits')}, "
                  f"cache_misses={usage.get('cache_misses')}, "
                  f"vision_cache_hits={usage.get('vision_cache_hits')}, "
                  f"done_reason={done_reason}", flush=True)
            finish = "stop" if done_reason == 1 else (
                "length" if done_reason == 2 else "stop")
            # Tool-call extraction is NOT a bridge concern. The model
            # may have emitted `<|tool_call>...<tool_call|>` bytes if
            # tools were in the request; those bytes are part of the
            # response text verbatim. Clients parse them client-side.
            message: dict[str, Any] = {"role": "assistant", "content": text}
            # 2026-05-07: no conversation-state recording. Stateless
            # bridge per the NO REMOTE LOCKS principle.
            choice: dict[str, Any] = {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
            if capture_logits:
                choice["logprobs"] = {
                    "content": [{
                        "token": g.detokenize([lp.token]),
                        "logprob": lp.sampled_logprob,
                        "top_logprobs": [
                            {"token": g.detokenize([t]), "logprob": p}
                            for t, p in lp.top_logprobs
                        ],
                    } for lp in collected_logprobs]
                }
            response = JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [choice],
                "usage": usage,
            })
            clean_close_aggregate = True
            return response
        finally:
            # 2026-05-07: cancel-on-disconnect. If we did not reach
            # state==2 (done_reason set), the client almost certainly
            # gave up on the request — TCP reset / asyncio cancel /
            # exception. Without enqueueing a cancel here, the engine
            # would keep running this stream's prefill+AR until natural
            # EOS, holding a slot away from other clients. The principle
            # is 'a dead client must NOT delay work for live ones'.
            if not clean_close_aggregate:
                cancel_spec = g.StreamSpec(stream_id=stream_id, action=2)
                try:
                    g.submit([cancel_spec])
                except Exception:
                    pass
            _response_qs.pop(stream_id, None)

    # SSE streaming. Cleanup MUST be inside the generator's own finally;
    # the handler returns immediately after constructing StreamingResponse,
    # before the generator iterates. A handler-level finally would pop
    # the response_q while the generator is still trying to read from it,
    # causing every coordinator update to be silently dropped and the
    # generator to block on response_q.get() forever.
    async def gen():
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        # Stream content deltas as they arrive. Tool-call markers in
        # the model's output are NOT extracted or reshaped here — they
        # ride through as part of the content text. Clients that want
        # OAI tool-call shape parse them themselves. The bridge does
        # not pay synchronous CPU on the response-streaming hot path
        # for application-level interpretation.
        all_tokens: list[int] = []
        last_update: g.StreamUpdate | None = None
        clean_close = False
        try:
            while True:
                u: g.StreamUpdate = await response_q.get()
                last_update = u
                if u.new_tokens:
                    all_tokens.extend(u.new_tokens)
                    delta_text = g.detokenize(u.new_tokens)
                    yield ("data: " + json.dumps({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta_text},
                            "finish_reason": None,
                        }],
                    }) + "\n\n")
                if u.state == 2:
                    finish = "stop" if u.done_reason == 1 else (
                        "length" if u.done_reason == 2 else "stop")
                    # 2026-05-07: no conversation-state recording.
                    yield ("data: " + json.dumps({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish,
                        }],
                        "usage": {
                            "prompt_tokens": u.prompt_tokens_seen,
                            "completion_tokens": u.completion_tokens_emitted,
                            "total_tokens": u.prompt_tokens_seen + u.completion_tokens_emitted,
                            "cache_hits": u.cache_hits,
                            "cache_misses": u.cache_misses,
                            "vision_cache_hits": u.vision_cache_hits,
                        },
                    }) + "\n\n")
                    yield "data: [DONE]\n\n"
                    print(f"[bridge] usage (SSE): "
                          f"prompt_tokens={u.prompt_tokens_seen}, "
                          f"completion_tokens={u.completion_tokens_emitted}, "
                          f"cache_hits={u.cache_hits}, "
                          f"cache_misses={u.cache_misses}, "
                          f"vision_cache_hits={u.vision_cache_hits}, "
                          f"done_reason={u.done_reason}", flush=True)
                    clean_close = True
                    break
        finally:
            # Log usage regardless of how we exited the loop. Clean
            # close (state==2) reaches the dedicated print earlier;
            # disconnect / cancellation lands here with `last_update`
            # holding the most recent StreamUpdate. This prevents the
            # silent-cache-hits-on-cancel observability gap that bit
            # us when Zed cancels SSE on tool-mode prose.
            if not clean_close and last_update is not None:
                u = last_update
                print(f"[bridge] usage (DISCONNECT): "
                      f"prompt_tokens={u.prompt_tokens_seen}, "
                      f"completion_tokens={u.completion_tokens_emitted}, "
                      f"cache_hits={u.cache_hits}, "
                      f"cache_misses={u.cache_misses}, "
                      f"vision_cache_hits={u.vision_cache_hits}, "
                      f"state={u.state}, "
                      f"done_reason={u.done_reason}", flush=True)
            # 2026-05-07: cancel-on-disconnect (SSE path). If the
            # client gave up before we hit state==2, free the engine
            # slot now rather than leaving it pinned until natural EOS.
            if not clean_close:
                cancel_spec = g.StreamSpec(stream_id=stream_id, action=2)
                try:
                    g.submit([cancel_spec])
                except Exception:
                    pass
            _response_qs.pop(stream_id, None)
    return StreamingResponse(gen(), media_type="text/event-stream")


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
