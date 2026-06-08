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

Run (canonical, config-driven):
  ./server/serve.py
  # (reads server/config.toml; default port 8001)

Or via the env-override path (priority: BRIDGE_URL > GEMMA_PORT > config):
  GEMMA_PORT=8002 ./server/serve.py

Clients should import bridge_config (server/bridge_config.py) rather
than hardcoding 127.0.0.1:8001 — that module reads the same config
file and exposes BRIDGE_URL / chat_completions_url() / etc.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import gemma_ffi as g
import gemma_tool_call_parser as gtc

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
# Gemma-4 has both sliding-window and full-attention layers. The slide
# layers can address MAX_PAGES_PER_SLOT * PAGE_SLIDE = 131072 tokens, but
# the full-attention block table is PAGE_FULL-granular: 8192 * 8 = 65536.
# Advertise the binding public context limit so clients do not submit a
# prompt the engine cannot index safely.
CONTEXT_LENGTH = 65536

# Batch shape the engine decodes through, advertised so CLIENTS STOP GUESSING
# IT. The kernel runs KERNEL_BATCH streams per decode step (bootstrap.swift:431
# `let B = 8`, "K=B=8 kernel-batch positions") behind an admission cap of
# MAX_SESSIONS logical sessions (M=64). The kernel pays its full B-wide dispatch
# cost regardless of occupancy, so a client running fewer than KERNEL_BATCH
# concurrent streams wastes the kernel. These MUST mirror bootstrap.swift; they
# are surfaced in /health so tools/batch_scaler.py can read the width from the
# engine instead of every experiment re-deriving it. (FFI accessors would fully
# single-source this; until then this constant is the declared mirror.)
KERNEL_BATCH = int(os.environ.get("GEMMA_KERNEL_BATCH", "8"))
MAX_SESSIONS = int(os.environ.get("GEMMA_MAX_SESSIONS", "64"))


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
# WHAT THIS ACTUALLY GIVES (measured, not predicted):
#   - Single FFI owner. The old two-asyncio-tasks pattern violated the
#     "single coroutine/thread calls into the dylib at a time" contract
#     documented in gemma_ffi.py: submit_pump and poll_pump could each
#     get a different threadpool worker and concurrently enter the FFI.
#     Even with Swift-side locking correctness, that doubled cgo entry
#     overhead. Native single owner is architecturally clean.
#   - Direct g.submit from handler. Cold-start submission latency is
#     gIntakeCond signal speed (microseconds) instead of waiting for
#     the next poll deadline. No measurable latency win in the bench
#     (cold start is rare), but real for low-arrival-rate workloads.
#
# WHAT THIS DID NOT GIVE (an honestly disclaimed past prediction):
# An earlier sched_sim_bridge sensitivity sweep predicted +30-40%
# aggregate throughput from this refactor. The actual measured AR-tick
# rate was 10.60 ticks/sec — identical to the prior asyncio.to_thread +
# 100ms-deadline-with-work-conserving-poll pattern. The sim's modeling
# of asyncio.to_thread overhead was off; the work-conserving poll fix
# from earlier in the session had already amortized that cost inside
# one poll call. The architectural cleanup matters for correctness;
# the throughput needle stays put at ~91% kernel saturation per CB.
# ----------------------------------------------------------------------
import threading

# 2026-05-23 (append-log refactor): the bridge now keeps a per-stream
# append-only log of EVERY StreamUpdate the engine emits. This is the
# substrate for true "replay from offset" reconnect semantics — matching
# real OpenAI/Anthropic behavior under packet loss, where a client whose
# socket drops mid-stream can reconnect and replay the bytes it never
# saw, not just attach to the tail of an ongoing producer.
#
# The asyncio.Queue is kept around purely as a NOTIFICATION mechanism:
# the engine driver thread does `loop.call_soon_threadsafe(q.put_nowait,
# None)` to wake any consumer awaiting new appends. The payload lives in
# `_stream_logs[sid]`; consumers read `_stream_logs[sid][offset:]` and
# advance their cursor.
#
# Memory budget (operator's "buffers are cheap, tokens are tiny"
# principle): a StreamUpdate is ~64-128 bytes. At max_tokens=2048,
# 2048 updates × ~96 bytes = ~200 KB per stream. 100 concurrent
# sessions = ~20 MB. Negligible vs modern process budgets.
_stream_logs: dict[int, list["g.StreamUpdate"]] = {}
_response_qs: dict[int, "asyncio.Queue[None]"] = {}
# Per-stream count of updates dropped at the _push_update_threadsafe
# high-water gate. Used for log-cadence-throttled WARN. Cleared on
# stream cleanup (pop from _response_qs).
_drop_counts: dict[int, int] = {}

# Per-stream rolling samples of (monotonic_ts_sec, cumulative_completion_tokens).
# Used by _compute_stream_tok_per_sec to publish live tok/s telemetry on
# /v1/engine/state (per active stream) and /health (aggregate). Window
# is a fixed-capacity deque — we use the last ~_RATE_WINDOW_SAMPLES
# StreamUpdate observations and compute tok/s = Δtokens / Δt over the
# span. At canonical ~30 tok/s that's ~1.6 s of history (sample-per-token
# is the dominant cadence), enough to smooth single-tick jitter without
# lagging perceived rate changes. Cleared in the same sites that pop
# _response_qs / _drop_counts so per-stream state lifecycles stay coupled.
_RATE_WINDOW_SAMPLES = 50
_stream_rate_samples: dict[int, "deque[tuple[float, int]]"] = {}
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
_DRIVER_POLL_DEADLINE_MS = 50  # 2026-05-23: bisected from 1000 — long deadline starved aggregate-mode handlers (0.16 tok/sec). Short deadline restores cooperative cycling.

# 2026-05-23 (operator directive): the response-queue high-water no
# longer triggers an engine cancel. It now ONLY decides when the bridge
# starts silently dropping updates (per `_push_update_threadsafe`). The
# engine continues running unaware of bridge queue depth — sampling
# termination is engine-owned (EOS, max_tokens, stop sequences).
#
# Sizing rationale at 65536: each StreamUpdate is ~64-128 bytes; 65k
# updates is ~8MB headroom per stream — trivial against modern process
# memory budgets. This gives any conceivable consumer-stall scenario
# (a hung downstream HTTP write, an SSE client backpressuring at ~0
# tok/sec) tens of minutes of full-rate buffering before drops kick in.
# At canonical 30 tok/s that's ~36 minutes of total stall budget. The
# previous 256 was a "presumed-dead" threshold tied to the cancel-on-
# overflow logic; once cancel-on-overflow is gone the small cap had
# no defensive value.
#
# Legacy P5 narrative (kept for historians, see git blame for the
# previous 256 era): the original 2026-05-18 design wired three layers
# of dead-consumer defense: bridge HIGH_WATER cancel (P5), bridge
# is_disconnected() poll (P1), engine forward-progress deadline (P3).
# Only P1 + P3 remain. The engine no longer cares whether the bridge
# is consuming — it self-terminates on sampling guards regardless.
_RESPONSE_Q_HIGH_WATER = 65536
_DISCONNECT_POLL_INTERVAL_S = 0.25

# B5 — Bounded TRANSPORT replay buffer for completed streams. This is
# NOT "engine warmth" or session retention — it is purely the tail of
# the transport double-buffer: once the engine emits state==2 and no wire
# consumer is attached, the stream's append-log moves into this
# OrderedDict (newest at the END) so a reconnect can still replay its
# bytes (GET /v1/streams/{id}/sse?since=N). The engine session itself is
# already gone (it closed itself on natural termination); all that
# remains here is the recorded byte stream, held only so a flaky client
# can resume. It stays addressable until pushed out by NEWER completed
# streams crossing a resource budget. NO WALLCLOCK TIMERS: eviction is
# bounded by RESOURCES (bytes + count), not by elapsed time — easier to
# reason about under load (no "what time is it" coupling, no timer firing
# while busy doing something else).
#
# Two bounds, both checked on every new addition:
#   - BRIDGE_MAX_RETAINED_BYTES (default 64MB) — bytes-budget across
#     all completed-idle logs (active streams are NOT in this LRU and
#     do not count). Primary bound — "expectation over buffer size".
#   - BRIDGE_MAX_COMPLETED_STREAMS (default 256) — count cap as a
#     safety net for pathological tiny-but-numerous streams.
#
# Reconnect transitions a stream OUT of the LRU (becomes "active again")
# while a consumer is attached; on consumer-disconnect it goes back in,
# moved to the END (most-recently-used).
from collections import OrderedDict

_BRIDGE_MAX_RETAINED_BYTES = int(
    os.environ.get("BRIDGE_MAX_RETAINED_BYTES", str(64 * 1024 * 1024)))
_BRIDGE_MAX_COMPLETED_STREAMS = int(
    os.environ.get("BRIDGE_MAX_COMPLETED_STREAMS", "256"))
# Estimate of a StreamUpdate's heap footprint. Used for bytes-budget
# accounting without paying sys.getsizeof per entry. StreamUpdate is a
# small dataclass (~6 ints + 3 small lists); 96 B is a sane average.
_STREAM_UPDATE_EST_BYTES = 96

# Completed-idle stream log LRU. Insertion order = oldest-first.
# Eviction pops from the front (oldest). On reconnect: pop the entry
# (becomes active again); on consumer-disconnect after state==2, add
# back at the END (newest).
_completed_stream_lru: "OrderedDict[int, None]" = OrderedDict()

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
    """Append `update` to `_stream_logs[sid]` and wake any consumer
    awaiting new appends via the per-stream notification queue.

    2026-05-23 append-log refactor: the update is now stored in the
    per-stream append-only log; the asyncio.Queue carries only a
    `None` sentinel as a wakeup. Consumers track their own offset
    cursor into `_stream_logs[sid]` and re-read the log slice on each
    wakeup. This preserves all updates so a reconnect can replay from
    any prior offset — matching real OpenAI/Anthropic semantics where
    a flaky-network client can drop and resume mid-stream without
    losing tokens that the producer already emitted.

    The high-water gate (now applied against log length, not queue
    qsize) still exists as a defensive ceiling, but at modern budgets
    (~96 B/update × 65536 = ~6 MB per pathologically stalled stream)
    it should essentially never fire in normal operation. When it does
    fire, the engine keeps producing; the bridge silently drops the
    overflow appends and logs at WARN cadence.
    """
    sid = update.stream_id
    log = _stream_logs.get(sid)
    if log is None:
        # Stream was already GC'd / never registered. Silently drop;
        # mirrors the old `rq is None` early-return path.
        return
    # Record rolling-rate sample BEFORE the high-water drop check so
    # we still observe rate evolution even on a stalled consumer
    # (the engine is producing; tok/s metric should reflect engine
    # throughput, not consumer behavior).
    samples = _stream_rate_samples.get(sid)
    if samples is None:
        samples = deque(maxlen=_RATE_WINDOW_SAMPLES)
        _stream_rate_samples[sid] = samples
    samples.append((time.monotonic(), int(update.completion_tokens_emitted)))
    if len(log) > _RESPONSE_Q_HIGH_WATER:
        n = _drop_counts.get(sid, 0) + 1
        _drop_counts[sid] = n
        if n == 1 or (n % 64) == 0:
            print(f"[bridge] stream_log over high-water on sid={sid} "
                  f"(len={len(log)} > {_RESPONSE_Q_HIGH_WATER}); "
                  f"dropping update (dropped={n} so far). Engine continues "
                  f"— no cancel submitted.", flush=True)
        return
    # list.append is atomic under the GIL — safe from the driver thread.
    log.append(update)
    # Notify any awaiting consumer. The queue is bounded but the
    # sentinel is tiny; if it's full, the consumer will pick up the
    # append on its next read anyway (notifications are coalesced).
    rq = _response_qs.get(sid)
    if rq is None:
        return
    try:
        loop.call_soon_threadsafe(_notify_consumer, rq)
    except RuntimeError:
        # Loop closed during shutdown — drop the wakeup silently.
        pass


def _notify_consumer(rq: "asyncio.Queue[None]") -> None:
    """Put a single notification sentinel onto the queue, ignoring
    QueueFull (the consumer will see the append on its next read regardless).
    Runs on the asyncio loop thread via call_soon_threadsafe."""
    try:
        rq.put_nowait(None)
    except asyncio.QueueFull:
        pass


def _compute_stream_tok_per_sec(stream_id: int) -> float:
    """Return rolling tok/s estimate for `stream_id` over the last
    ~_RATE_WINDOW_SAMPLES StreamUpdate observations.

    Returns 0.0 when:
      - the stream has no samples (not started, or already GC'd), or
      - fewer than 2 samples have been recorded (need two points to
        define a rate), or
      - the elapsed window is degenerate (clamped to 0.001 s floor).

    Computed as (latest_tokens - oldest_tokens) / (latest_ts - oldest_ts).
    Snapshot-style read of the deque endpoints; deque indexing is O(1)
    and the dict lookup is atomic under the GIL — no lock needed.
    """
    samples = _stream_rate_samples.get(stream_id)
    if samples is None or len(samples) < 2:
        return 0.0
    oldest_ts, oldest_tokens = samples[0]
    latest_ts, latest_tokens = samples[-1]
    dt = max(0.001, latest_ts - oldest_ts)
    dtok = latest_tokens - oldest_tokens
    if dtok <= 0:
        return 0.0
    return float(dtok) / dt


def _compute_aggregate_tok_per_sec() -> float:
    """Sum of `_compute_stream_tok_per_sec` across all tracked streams.
    Safe to call concurrently with the engine driver thread; iterates
    a snapshot of dict keys to avoid 'dict changed during iteration'.
    """
    total = 0.0
    for sid in list(_stream_rate_samples.keys()):
        total += _compute_stream_tok_per_sec(sid)
    return total


async def _wait_until_disconnected(request: Request) -> None:
    """Block until the client TCP-disconnects. Cancellable.

    `Request.is_disconnected()` reads the ASGI receive channel for an
    http.disconnect message. It's a cheap non-blocking check — we poll
    it on a short cadence rather than awaiting a single receive event
    so that handler-level cancellation propagates cleanly through
    asyncio.wait. Returns silently on disconnect; exceptions
    (uvicorn going away mid-poll) propagate so the caller can decide.
    """
    while True:
        if await request.is_disconnected():
            return
        await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)


# SSE forward-progress contract (2026-06): a streaming response must NEVER be
# silent. While the engine prefills (no tokens emitted yet), the SSE consumer
# emits a keep-alive every _SSE_HEARTBEAT_S so the client knows the stream is
# alive (not hung), announces the stream immediately (role delta + prefilling
# comment), and surfaces a GENUINE engine hang as a fast clean error after
# _SSE_FORWARD_PROGRESS_DEADLINE_S of zero engine updates — instead of leaving
# the wire silent until an opaque client socket timeout (~30s) papered over by
# downstream band-aids. Opt-in via heartbeat_s so the aggregate/reconnect
# callers stay byte-identical.
_SSE_HEARTBEAT_S = 2.0
_SSE_FORWARD_PROGRESS_DEADLINE_S = 90.0


async def _consume_engine_stream(stream_id: int, request: Request,
                                   from_offset: int = 0,
                                   retain_on_clean_close: bool = True,
                                   heartbeat_s=None):
    """Single shared coroutine that owns engine-stream lifecycle.

    Yields `(offset, StreamUpdate)` tuples in order, starting at
    `from_offset` (default 0 = full replay of the per-stream append-log).

    2026-05-23 append-log refactor: the engine driver thread appends
    every StreamUpdate to `_stream_logs[stream_id]`; this consumer
    advances a cursor over that list and yields each entry exactly
    once. The notification queue (`_response_qs[stream_id]`) signals
    new appends — empty/coalesced sentinels are fine because the
    cursor re-reads `_stream_logs[stream_id][cursor:]` on every wakeup.

    Guarantees:
      - On reach of an update with state==2 at the cursor, yields it
        then returns ("clean close" — the engine signaled natural
        termination).
      - On client disconnect (within _DISCONNECT_POLL_INTERVAL_S),
        returns without further yields. Disconnects DO NOT cancel
        the engine session (operator directive 2026-05-23). The log
        keeps appending; subsequent reconnect via
        `GET /v1/streams/{id}/sse?since={N}` can replay from any prior
        offset.
      - If retain_on_clean_close is true, the log is preserved after
        natural completion via the completed-idle LRU. Non-streaming
        aggregate calls pass false because there is no useful reconnect
        surface for their stream id; retaining those logs only accumulates
        bridge-side resources.
      - The log is preserved past the disconnect via the completed-idle
        LRU (`_completed_stream_lru`) — bounded by bytes + count, not
        wallclock. See `_mark_stream_completed_for_eviction` and
        `_evict_completed_streams_to_bounds`.

    Used by BOTH the aggregate and SSE branches of chat_completions,
    plus the reconnect endpoint. The branches differ only in what
    they do with each yielded (offset, update). This function owns
    the cross-cutting concerns: disconnect detection, log-cursor
    advancement, background-drain spawn on disconnect.

    Caller pattern:
        async for offset, u in _consume_engine_stream(stream_id, req):
            # render u as the caller's response format, embedding offset
            if u.state == 2:
                ...  # render final state, then loop will end
        # if we get here without ever seeing u.state == 2, the
        # client disconnected; the engine session continues running
        # and the log keeps accruing for any reconnect.
    """
    response_q = _response_qs[stream_id]
    log = _stream_logs[stream_id]
    cursor = int(from_offset)
    last_yielded_update: g.StreamUpdate | None = None
    clean_close = False
    # 2026-05-23 PERF FIX: hoist disc_task OUTSIDE the loop so it lives
    # for the duration of the stream, not per-iteration. Disc fires once
    # on client disconnect; we cancel it in the finally block.
    disc_task = asyncio.create_task(_wait_until_disconnected(request))
    try:
        while True:
            # Fast path: disconnect already fired between iterations.
            if disc_task.done():
                try: disc_task.result()
                except Exception: pass
                break
            # Drain everything currently in the log past our cursor.
            # list slicing is GIL-atomic; new appends may race but
            # we'll catch them on the next wakeup.
            while cursor < len(log):
                u = log[cursor]
                cursor += 1
                last_yielded_update = u
                # Set clean_close BEFORE yield so a caller that breaks
                # after seeing state==2 still triggers clean-close
                # accounting in the finally block.
                if u.state == 2:
                    clean_close = True
                yield (cursor - 1, u)
                if clean_close:
                    break
            if clean_close:
                break
            # Cursor caught up to log tail; wait for a notification.
            get_task = asyncio.create_task(response_q.get())
            try:
                if heartbeat_s is None:
                    # Aggregate / reconnect callers: wait indefinitely; the
                    # cursor re-reads the log on the next append. Unchanged.
                    done, _pending = await asyncio.wait(
                        {get_task, disc_task},
                        return_when=asyncio.FIRST_COMPLETED)
                else:
                    # SSE forward-progress: bounded waits so the wire is NEVER
                    # silent. Each interval with no engine update yields a
                    # heartbeat sentinel (cursor, None) — the caller emits an SSE
                    # keep-alive and owns the hang deadline. The pending get_task
                    # survives across timeouts (asyncio.wait does not cancel it).
                    while True:
                        done, _pending = await asyncio.wait(
                            {get_task, disc_task}, timeout=heartbeat_s,
                            return_when=asyncio.FIRST_COMPLETED)
                        if done:
                            break
                        yield (cursor, None)   # heartbeat; wire stays alive
            except asyncio.CancelledError:
                get_task.cancel(); disc_task.cancel()
                raise
            if disc_task in done:
                get_task.cancel()
                try: disc_task.result()
                except Exception: pass
                break
            # get_task won; loop back to drain freshly-appended updates.
            # The sentinel value (None) we just received is discarded —
            # only `_stream_logs[stream_id]` carries payload now.
    finally:
        # Always cancel the per-stream disconnect watcher we own. Idempotent.
        try: disc_task.cancel()
        except Exception: pass
        # B5 — BRIDGE = TRANSPORT DOUBLE-BUFFER. The TCP socket is PURELY
        # a transport for the append-log: connect/disconnect/reconnect/
        # streaming/replay are reads over `_stream_logs[stream_id]`. The
        # bridge NEVER submits action=2/closeSession and NEVER retains or
        # frees engine resources based on a socket signal. A disconnected-
        # but-running generation keeps writing to the log; a subsequent
        # reconnect (GET /v1/streams/{id}/sse?since=N) replays it in full.
        #
        # Resource pressure is handled by ADMISSION CONTROL, not socket
        # teardown: the engine refuses new sessions when its residency cap
        # is hit (surfaced as HTTP 503 admission backpressure), and the
        # engine's own forward-progress reaper (lm_engine
        # expireStalledSessions) closes a genuinely-wedged session. A
        # client that is truly DONE with a running generation must say so
        # explicitly via POST /v1/streams/{id}/cancel — the only bridge
        # path that submits action=2.
        #
        # APPEND-LOG: the per-stream log captures everything the engine
        # emits regardless of consumer presence. On clean_close we add the
        # log to the bounded transport replay buffer so a reconnect can
        # still replay it. On disconnect with the engine still running, the
        # background drain task just watches for state==2 and then moves
        # the log into that buffer; it does NOT cancel anything.
        if clean_close:
            # Natural completion: caller observed state==2. Move into
            # the completed-idle LRU so a reconnect can still replay
            # the full sequence — the LRU evictor (count + bytes
            # bounds, no wallclock) decides when to free.
            if retain_on_clean_close:
                _mark_stream_completed_for_eviction(stream_id)
            else:
                _release_stream_log(stream_id)
        else:
            if last_yielded_update is not None:
                u = last_yielded_update
                print(f"[bridge] usage (DISCONNECT — session continues): "
                      f"stream_id={stream_id}, "
                      f"cursor={cursor}, log_len={len(log)}, "
                      f"prompt_tokens={u.prompt_tokens_seen}, "
                      f"completion_tokens={u.completion_tokens_emitted}, "
                      f"cache_hits={u.cache_hits}, "
                      f"cache_misses={u.cache_misses}, "
                      f"vision_cache_hits={u.vision_cache_hits}, "
                      f"state={u.state}, "
                      f"done_reason={u.done_reason}", flush=True)
            else:
                print(f"[bridge] disconnect before first yielded update "
                      f"(stream_id={stream_id}, cursor={cursor}, "
                      f"log_len={len(log)}); session continues — log "
                      f"keeps appending for any reconnect",
                      flush=True)
            # Disconnect is a transport event only — the engine session is
            # never cancelled or freed here. The log keeps appending; spawn
            # (or refresh) the background-drain monitor so the log moves
            # into the bounded transport replay buffer once the engine
            # emits state==2 naturally.
            _spawn_background_drain(stream_id)
        # Release the single-consumer slot if we held it (the initial
        # POST does not claim it; the reconnect handler does).
        _active_consumer_token.pop(stream_id, None)


# Background-drain tasks for streams whose HTTP consumer disconnected.
# Keyed by stream_id so the spawn helper can avoid double-registration.
# Tasks self-remove on exit.
#
# 2026-05-23 APPEND-LOG REFACTOR: the drain task no longer "preserves
# the queue" or "hands off the queue to a reconnect consumer" — the
# per-stream append-log captures everything regardless of consumer
# presence. The drain task's ONLY remaining job is to watch for
# state==2 and trigger the retention-window timer. If a reconnect
# attaches before retention fires, the timer just keeps running until
# disconnect — multi-reader semantics over an append-log don't conflict
# with retention.
_background_drains: dict[int, "asyncio.Task[None]"] = {}

# 2026-05-23 (reconnect endpoint): single-consumer-at-a-time gate.
# The value is an opaque token identifying the currently-attached
# wire consumer. We enforce this because emitting the same SSE
# bytes to two simultaneous HTTP consumers is wasteful (and confusing
# in logs); the log itself is multi-reader-safe, but we keep "one
# wire reader at a time" as the public contract. A reconnect request
# that finds the slot occupied returns 409.
_active_consumer_token: dict[int, str] = {}


def _retained_log_bytes() -> int:
    """Estimate total bytes held by completed-idle logs in the LRU.
    Active streams (consumer attached or engine still running) are NOT
    in `_completed_stream_lru` and don't count toward this budget.
    """
    total = 0
    for sid in _completed_stream_lru.keys():
        log = _stream_logs.get(sid)
        if log is not None:
            total += len(log) * _STREAM_UPDATE_EST_BYTES
    return total


def _evict_completed_streams_to_bounds() -> None:
    """Evict oldest completed-idle streams from `_completed_stream_lru`
    (and free their per-stream state) until both bounds are satisfied:
      - len(_completed_stream_lru) <= _BRIDGE_MAX_COMPLETED_STREAMS
      - _retained_log_bytes() <= _BRIDGE_MAX_RETAINED_BYTES

    Called on every transition INTO the LRU (i.e., every
    `_mark_stream_completed_for_eviction` call). NOT periodic — no
    wallclock involvement. Eviction happens exactly when new completed
    streams push the budget over.
    """
    while _completed_stream_lru:
        too_many = len(_completed_stream_lru) > _BRIDGE_MAX_COMPLETED_STREAMS
        too_big = _retained_log_bytes() > _BRIDGE_MAX_RETAINED_BYTES
        if not (too_many or too_big):
            return
        # Pop oldest (front of OrderedDict).
        sid, _ = _completed_stream_lru.popitem(last=False)
        log = _stream_logs.get(sid)
        log_len = len(log) if log is not None else 0
        reason = "count-bound" if too_many else "bytes-bound"
        print(f"[bridge] LRU evict: freeing sid={sid} "
              f"(log_len={log_len}, lru_size_post_evict={len(_completed_stream_lru)}, "
              f"retained_bytes_post_evict={_retained_log_bytes()}, "
              f"reason={reason})", flush=True)
        _release_stream_log(sid)


def _mark_stream_completed_for_eviction(stream_id: int) -> None:
    """Move `stream_id` into the completed-idle LRU and run the
    eviction policy. Called when:
      - the wire consumer observes state==2 and disconnects, OR
      - the background drain observes state==2 with no consumer attached

    Idempotent: if already in the LRU, move to the END (most-recently
    completed). If the stream is currently active (consumer attached),
    DON'T add — wait until disconnect.
    """
    if stream_id in _active_consumer_token:
        # Currently being read; stays "active". On consumer-disconnect
        # the disconnect handler will call us again.
        return
    if stream_id not in _stream_logs:
        # Log already freed (LRU eviction, or an explicit
        # POST /v1/streams/{id}/cancel); nothing to track.
        return
    # Move-to-end if already present; otherwise append.
    _completed_stream_lru.pop(stream_id, None)
    _completed_stream_lru[stream_id] = None
    _evict_completed_streams_to_bounds()


def _release_stream_log(stream_id: int) -> None:
    """Pop ALL per-stream transport state. Called by the LRU evictor
    (bounded transport replay buffer overflow) and by an explicit
    POST /v1/streams/{id}/cancel. NEVER called on a bare socket
    disconnect — disconnect is a transport event only."""
    _stream_logs.pop(stream_id, None)
    _response_qs.pop(stream_id, None)
    _drop_counts.pop(stream_id, None)
    _stream_rate_samples.pop(stream_id, None)
    _completed_stream_lru.pop(stream_id, None)
    _active_consumer_token.pop(stream_id, None)


# ----------------------------------------------------------------------
# ADMISSION-PRESSURE-CANCEL (2026-06, KV-retention/connection-decoupling).
#
# The contract: under page pressure with NO free/evictable pages, the
# engine refuses a new session (admission backpressure). Rather than
# bouncing the caller with a bare 503 while zombie generations (running
# but with no wire consumer — a disconnected client whose engine session
# the B5 transport-double-buffer kept alive) squat KV pages, the bridge
# SHEDS the lowest-value such generation: a WHOLE generation killed +
# freed via the existing /cancel action=2 path (kill+free, NEVER pause).
#
# Why kill, never pause: re-prefill is NOT bit-exact (lm_engine qLen/tile
# reduction-order dependent), so pausing-and-reprefilling a live generation
# would change its output bytes. Killing is bit-exact-safe — the partial
# output already produced stays addressable in the transport append-log,
# so a reconnect (GET /v1/streams/{id}/sse?since=N) still replays every
# byte the engine emitted before the kill, and the client can resume from
# there via continue_final_message.
#
# VALUE / ELIGIBILITY (bridge-side proxy for the engine's decayed-citation
# value function): the bridge cannot see per-page citation scores, but it
# CAN see which generations have an active wire consumer. The eligible set
# is generations that are:
#   - LIVE in the engine (engine_state reports them with pages_owned > 0
#     and state != "done"), AND
#   - have NO active wire consumer (stream_id not in _active_consumer_token)
#       — i.e. the socket dropped and B5 kept the engine session alive.
# These are exactly the "no-active-consumer" generations the spec targets.
# A generation with a live wire consumer is NEVER shed (a watching client's
# working set is protected). Among the eligible set we pick the LOWEST
# value: the engine-reported decayed-citation score if it surfaces one
# (`value` / `decayed_citation` field on the active_streams entry), else
# the least-recently-active generation (oldest bridge rate-sample
# timestamp), with most-pages-owned as the final tiebreak (free the most
# under one kill). expireStalledSessions (engine forward-progress KILL)
# remains the independent backstop for genuinely-wedged sessions.
# ----------------------------------------------------------------------
def _shed_lowest_value_generation() -> int | None:
    """Kill+free the single lowest-value no-active-consumer generation.

    Returns the stream_id that was shed, or None if there was no eligible
    victim (every live generation has an attached wire consumer, or the
    engine reports no live generations). Synchronous: reads engine_state,
    submits action=2, releases the bridge log. Safe to call from a handler
    coroutine (engine_state + submit are thread-safe FFI calls).
    """
    try:
        snap = g.engine_state()
    except Exception as e:
        print(f"[bridge] admission-pressure-cancel: engine_state() failed "
              f"({e}); cannot pick a victim", flush=True)
        return None
    live = snap.get("active_streams") or []
    candidates: list[tuple[float, int, int]] = []  # (value_key, -pages, sid)
    now = time.monotonic()
    for entry in live:
        try:
            sid = int(entry.get("stream_id"))
        except (TypeError, ValueError):
            continue
        state = entry.get("state")
        pages = int(entry.get("pages_owned", 0) or 0)
        # Only consider generations actually holding pages and not done.
        if state == "done" or pages <= 0:
            continue
        # ELIGIBILITY: no active wire consumer = sheddable. A watched
        # generation's working set is protected (never shed live-watched
        # pages; over-subscription of watched work is the engine's
        # admission-cap problem, not ours to kill).
        if sid in _active_consumer_token:
            continue
        # VALUE: prefer an engine-reported decayed-citation score if the
        # engine surfaces one; lower = shed first. Fall back to recency
        # (least-recently-active bridge rate sample), then to a 0 floor.
        val = entry.get("decayed_citation",
                        entry.get("value"))
        if val is not None:
            try:
                value_key = float(val)
            except (TypeError, ValueError):
                value_key = _last_activity_age(sid, now)
        else:
            value_key = _last_activity_age(sid, now)
        candidates.append((value_key, -pages, sid))
    if not candidates:
        return None
    # MIN value (engine path) / for recency-age fallback the key is an
    # AGE (older = larger), so the engine-value path picks min-citation
    # while the recency fallback would want max-age. Normalize: if every
    # candidate used the recency fallback (no engine value present), pick
    # the OLDEST (max age). Detect by whether any engine value was seen.
    any_engine_value = any(
        (e.get("decayed_citation", e.get("value")) is not None)
        for e in live
        if str(e.get("stream_id")) and _safe_int(e.get("stream_id")) is not None
    )
    if any_engine_value:
        # Lowest decayed-citation first; -pages tiebreak frees the most.
        victim = min(candidates, key=lambda c: (c[0], c[1]))[2]
    else:
        # Recency fallback: oldest (largest age) first; -pages tiebreak.
        victim = max(candidates, key=lambda c: (c[0], -c[1]))[2]
    print(f"[bridge] admission-pressure-cancel: shedding lowest-value "
          f"no-consumer generation sid={victim} "
          f"(candidates={[(c[2], round(c[0], 3), -c[1]) for c in candidates]}); "
          f"kill+free via action=2 (partial output remains in transport log)",
          flush=True)
    try:
        g.submit([g.StreamSpec(stream_id=victim, action=2)])
    except Exception as e:
        print(f"[bridge] admission-pressure-cancel: action=2 submit for "
              f"sid={victim} failed ({e})", flush=True)
        return None
    _release_stream_log(victim)
    return victim


def _safe_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _last_activity_age(sid: int, now: float) -> float:
    """Age (seconds) since the bridge last observed an update for `sid`.
    Larger = staler. Streams with no samples are treated as maximally
    stale (float('inf')) so a never-progressing zombie sheds first."""
    samples = _stream_rate_samples.get(sid)
    if not samples:
        return float("inf")
    return now - samples[-1][0]


# Bound on how many shed-and-retry rounds a single admission-pressure
# spike triggers before the caller is told to back off (503). Each round
# sheds exactly one no-active-consumer generation, so the worst case is
# this many whole generations killed to admit one new request. Small by
# design — if pressure persists past a few sheds the right answer is to
# tell the caller to retry later, not to kill the whole pool.
_ADMISSION_SHED_MAX_ROUNDS = int(
    os.environ.get("BRIDGE_ADMISSION_SHED_MAX_ROUNDS", "3"))
# How long to wait for the engine's first update on a freshly-submitted
# stream when probing for an admission-backpressure terminal. The engine
# emits the synthetic backpressure terminal on the very next poll tick,
# so this only needs to cover one driver poll deadline plus slack. If no
# update arrives in this window the request was admitted (the engine is
# busy prefilling) and we hand off to the normal consumer untouched.
_ADMISSION_PROBE_TIMEOUT_S = float(
    os.environ.get("BRIDGE_ADMISSION_PROBE_TIMEOUT_S", "0.5"))


def _head_is_admission_backpressure(stream_id: int) -> bool:
    """True iff the FIRST logged update for `stream_id` is a terminal
    admission-backpressure rejection. Peek-only — does NOT advance any
    consumer cursor (the downstream consume loop still starts at offset 0
    on the non-backpressure path)."""
    log = _stream_logs.get(stream_id)
    if not log:
        return False
    head = log[0]
    return (head.state == 2 and head.done_reason == 3
            and (head.err_msg or "").startswith("admission backpressure"))


def _head_is_context_too_large(stream_id: int) -> bool:
    """True iff the FIRST logged update for `stream_id` is a terminal
    context-too-large rejection (PERMANENT — the request can never be
    served because its worst-case k_len exceeds the slot block-table
    capacity or the KV pool capacity). Peek-only, mirrors
    _head_is_admission_backpressure. These are NEVER retried — the
    _resubmit_after_shed retry loop keys only off the backpressure prefix,
    so a context-too-large terminal falls through to the normal consumer
    and surfaces as HTTP 413."""
    log = _stream_logs.get(stream_id)
    if not log:
        return False
    head = log[0]
    return (head.state == 2 and head.done_reason == 3
            and (head.err_msg or "").startswith("context too large"))


async def _resubmit_after_shed_if_backpressure(
        stream_id: int, spec: "g.StreamSpec",
        response_q: "asyncio.Queue") -> int:
    """Probe a freshly-submitted stream for an admission-backpressure
    terminal; if found, shed the lowest-value no-active-consumer
    generation and re-submit, up to _ADMISSION_SHED_MAX_ROUNDS times.

    Returns the stream_id the downstream consumer should iterate. On the
    happy path (request admitted, or no shed needed) this is the original
    stream_id, untouched, with its log still at offset 0 for the consumer.

    Each retry round, on a confirmed backpressure terminal:
      1. release the rejected stream's bridge log (it carries only the
         synthetic terminal — nothing a client wants to replay),
      2. shed one zombie generation (kill+free via action=2),
      3. allocate a FRESH stream_id + log + queue and re-submit the same
         spec, then probe again.

    If shedding finds no eligible victim, or we exhaust the round budget,
    the most-recent (still-backpressured) stream_id is returned so the
    normal consumer surfaces the backpressure terminal as HTTP 503 / SSE
    error — preserving the existing client-visible contract.
    """
    cur_sid = stream_id
    cur_q = response_q
    for _round in range(_ADMISSION_SHED_MAX_ROUNDS):
        # Wait briefly for the first update (or none if admitted+busy).
        try:
            await asyncio.wait_for(cur_q.get(), timeout=_ADMISSION_PROBE_TIMEOUT_S)
        except asyncio.TimeoutError:
            # No update yet → admitted; engine is prefilling. Hand off.
            return cur_sid
        if not _head_is_admission_backpressure(cur_sid):
            # First update is real progress (or a non-backpressure
            # terminal). Hand off to the normal consumer; its cursor
            # starts at offset 0 and re-reads the log including this
            # update (we only drained the notification sentinel, not the
            # payload, which lives in _stream_logs).
            return cur_sid
        # Confirmed backpressure. Shed a zombie and re-submit.
        victim = _shed_lowest_value_generation()
        if victim is None:
            print(f"[bridge] admission-pressure-cancel: no sheddable "
                  f"no-consumer generation for backpressured sid={cur_sid}; "
                  f"surfacing backpressure to caller", flush=True)
            return cur_sid
        # Drop the rejected stream's log (only the synthetic terminal).
        _release_stream_log(cur_sid)
        # Fresh stream for the retry.
        cur_sid = await _next_stream_id_alloc()
        _stream_logs[cur_sid] = []
        cur_q = asyncio.Queue(maxsize=64)
        _response_qs[cur_sid] = cur_q
        retry_spec = g.StreamSpec(
            stream_id=cur_sid, action=spec.action, flags=spec.flags,
            segments=spec.segments, sampling=spec.sampling,
            tokens=spec.tokens, control_vectors=spec.control_vectors)
        print(f"[bridge] admission-pressure-cancel: re-submitting as fresh "
              f"sid={cur_sid} after shedding sid={victim} "
              f"(round {_round + 1}/{_ADMISSION_SHED_MAX_ROUNDS})", flush=True)
        try:
            rc = g.submit([retry_spec])
        except Exception as e:
            _release_stream_log(cur_sid)
            print(f"[bridge] admission-pressure-cancel: retry submit failed "
                  f"({e})", flush=True)
            return cur_sid
        if rc != 0:
            print(f"[bridge] admission-pressure-cancel: retry submit rc={rc}",
                  flush=True)
            return cur_sid
    # Round budget exhausted; return the last (backpressured) stream so the
    # caller surfaces 503/SSE-error per the existing contract.
    return cur_sid


async def _drain_until_engine_done(stream_id: int) -> None:
    """Background monitor for streams whose wire consumer disconnected.

    2026-05-23 APPEND-LOG REFACTOR: this is now a CURSOR-BASED
    monitor over `_stream_logs[stream_id]`, not a queue-drainer.
    It walks the log forward looking for an update with state==2,
    then starts the retention timer. It does NOT consume the log
    (the log is multi-reader-safe and a reconnect can still replay
    from offset 0).

    If a reconnect attaches mid-monitor, the monitor exits cleanly
    (no work left — the wire consumer will observe state==2 itself
    and trigger retention). The reconnect detection is via
    `_active_consumer_token`.
    """
    log = _stream_logs.get(stream_id)
    if log is None:
        return
    rq = _response_qs.get(stream_id)
    if rq is None:
        return
    cursor = 0
    try:
        while True:
            # Drain everything currently in the log past our cursor.
            while cursor < len(log):
                u = log[cursor]
                cursor += 1
                if u.state == 2:
                    print(f"[bridge] background drain (sid={stream_id}): "
                          f"observed natural state==2 after scanning "
                          f"{cursor} log entries; "
                          f"prompt_tokens={u.prompt_tokens_seen}, "
                          f"completion_tokens={u.completion_tokens_emitted}, "
                          f"cache_hits={u.cache_hits}, "
                          f"cache_misses={u.cache_misses}, "
                          f"done_reason={u.done_reason}; "
                          f"moving sid into completed-idle LRU "
                          f"(eviction by bytes+count bounds, no wallclock)",
                          flush=True)
                    _mark_stream_completed_for_eviction(stream_id)
                    return
            # Check for reconnect-takeover before waiting.
            if stream_id in _active_consumer_token:
                print(f"[bridge] background drain (sid={stream_id}): "
                      f"reconnect consumer attached "
                      f"(token={_active_consumer_token[stream_id]!r}); "
                      f"monitor handing off, wire consumer will "
                      f"trigger retention on state==2", flush=True)
                return
            # Wait for a notification (new append or wakeup).
            try:
                await asyncio.wait_for(rq.get(),
                                         timeout=_DISCONNECT_POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                # Periodic wake to recheck reconnect-takeover even
                # without a fresh append. Cheap.
                pass
    except asyncio.CancelledError:
        print(f"[bridge] background drain (sid={stream_id}) cancelled "
              f"after scanning {cursor} log entries", flush=True)
        raise
    except Exception as e:
        print(f"[bridge] background drain (sid={stream_id}) "
              f"errored after {cursor} log entries: {e}", flush=True)
    finally:
        _background_drains.pop(stream_id, None)


def _spawn_background_drain(stream_id: int) -> None:
    """Start a background drain task for `stream_id` if not already
    running. Idempotent; safe to call from `_consume_engine_stream`'s
    finally block (which runs on the asyncio loop thread)."""
    if stream_id in _background_drains:
        return
    if stream_id not in _stream_logs:
        return
    try:
        task = asyncio.create_task(_drain_until_engine_done(stream_id))
        _background_drains[stream_id] = task
    except RuntimeError as e:
        # No running loop — shouldn't happen in handler context.
        print(f"[bridge] _spawn_background_drain (sid={stream_id}) "
              f"failed: {e}; releasing log immediately", flush=True)
        _release_stream_log(stream_id)


# 2026-05-24: _stream_log_retention_sweep removed. Wallclock-driven
# eviction was operator-disliked ("BUFFERS. more BUFFERS. but no time
# coupling"). Replaced by `_evict_completed_streams_to_bounds`, which
# runs synchronously inside `_mark_stream_completed_for_eviction` (i.e.
# only at the moment a new stream enters the completed-idle LRU). No
# background task, no timers, no asyncio.sleep.


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
    # API-boundary clamp: temperature=0.0 is forbidden. Per design
    # principle (no greedy/argmax in eval-instrument paths because
    # it eliminates the stochastic regime we actually deploy under),
    # we clamp incoming temperatures up to a small floor. Callers
    # who omit temperature get the OpenAI default of 1.0 instead of
    # the previous 0.0. Documented at docs/dataflow_pipeline_spec.md.
    temperature = float(body.get("temperature", 1.0))
    if temperature < 0.01:
        temperature = 0.01
    top_p = float(body.get("top_p", 1.0))
    top_k = int(body.get("top_k", 0))
    seed = body.get("seed")
    seed_int = int(seed) if seed is not None else 0
    # SillyTavern/OpenAI-compatible clients commonly use seed=-1 as a
    # "random/default seed" sentinel. The native sampler expects a
    # non-negative seed; letting -1 cross the FFI boundary produces an
    # immediate 500 before any token work happens. Treat negative values
    # as omitted/default.
    if seed_int < 0:
        seed_int = 0
    rep_pen = float(body.get("repetition_penalty",
                              body.get("frequency_penalty", 1.0)))
    # OpenAI's `stop` is a list of text strings (or a single string).
    # The engine takes `stop_sequences` as lists of token IDs — we
    # tokenize each stop string and append to sampling.stop_sequences
    # in the caller (chat_completions handler, where we have access
    # to the tokenizer-aware path). _parse_sampling stays tokenizer-
    # free; pass the raw strings through `body['stop']` and let the
    # caller resolve them.
    # `stop_tokens` (a different SamplingParams field) is documented
    # as ignored by the engine in ffi_batch.swift:277 — we don't try
    # to populate it.
    stop_tokens: list[int] = []
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


# ────────────────────────────────────────────────────────────────────────
# Tool-call extraction (2026-05-07).
#
# Gemma-4's chat template (format_argument macro) emits string arguments
# as `<|"|>STRING<|"|>` (the atomic id-52 token wrapping the content),
# but in practice the model improvises a SECOND format when the string
# contains literal `"` characters (e.g., the SVG markup `width="100"`):
# it switches to a raw-string form `<|<DELIM>STRING<DELIM>|>` where
# <DELIM> is typically a backtick. Both forms are observed in the wild;
# the bridge accepts either.
#
# A tool call from the model looks like:
#
#     <|tool_call>call:NAME{key1:value1,key2:value2,...}<tool_call|>
#
# This extractor:
#   1. finds every `<|tool_call>...<tool_call|>` block in the response,
#   2. parses the function name + argument body,
#   3. converts it to OpenAI tool_calls[] shape:
#         {"id": "call_<uuid>", "type": "function",
#          "function": {"name": "<name>", "arguments": "<json-string>"}}
#   4. returns (cleaned_content_with_markers_stripped, tool_calls_list).
#
# When tools are NOT in the request, this is a no-op. When tools ARE in
# the request and no markers are found (model declined to call),
# tool_calls_list is None and content is unchanged.
# Channel-marker scaffolding (`<|channel>thought\n<channel|>` etc.) and
# tool_call argument double-bracketing used to be patched up here. Both
# are now fixed at the chat-template level: the no-thinking epilogue is
# omitted (chat_template.jinja#add_generation_prompt block) and tool_call
# arguments-as-string are parsed via the `from_json` Jinja filter
# (chat_template.py:_safe_from_json + chat_template.jinja#tool_calls
# rendering), so the model sees and emits a single canonical format.


# The OUTER block bounds (`<|tool_call>` ... `<tool_call|>`) are atomic
# tokenizer-vocab tokens; they can't appear nested inside themselves so
# a simple find/skip walk over the two literal boundaries is sufficient
# (and equivalent to the prior non-greedy regex). The BODY between is
# parsed by gemma_tool_call_parser.parse_tool_call_body — a proper
# recursive-descent parser whose grammar mirrors chat_template.jinja's
# format_argument macro one-for-one. The previous three-regex-pass +
# json.loads approach was structurally inadequate (couldn't handle
# nested braces in atomic-quoted strings, bareword keys inside string
# content, etc.) and was the proximate cause of "some tool calls
# rendered, others didn't" — content-dependent parse failures left the
# raw block in `content` so ST rendered the marker text as markdown
# instead of getting a structured `tool_calls` chunk.
_TOOL_CALL_OPEN = '<|tool_call>'
_TOOL_CALL_CLOSE = '<tool_call|>'

# Gemma-4 chat-template turn delimiters in textual form. Gemma-4 (unlike
# Gemma-3) uses the `<|name>` / `<name|>` open/close convention for turn
# boundaries — same shape as tool_call markers, different role. These are
# atomic special tokens in the vocab (ids 105 and 106 respectively) but
# detokenize() renders them as their literal surface strings. They should
# never appear in user-facing content: the bridge already stops generation
# on token 106, but token 106's TEXT can land in the output buffer before
# the stop check fires, and the model can also spontaneously emit a turn
# marker mid-stream (a sampling artefact analogous to the tool_call
# spontaneous emission case).
_TURN_OPEN_TEXT  = '<|turn>'
_TURN_CLOSE_TEXT = '<turn|>'


def _strip_turn_markers(text: str) -> str:
    """Strip Gemma-4 turn-delimiter surface strings from user-facing
    content. Handles three cases:

      1. Full marker emitted as the special token (id 105 or 106): the
         vocab renders it as the literal string `<|turn>` / `<turn|>`,
         which we replace verbatim.
      2. Full marker emitted as ordinary byte-level tokens (the model
         hallucinates the marker as literal ASCII rather than as the
         special token — observed when the prompt contains other angle-
         bracket tags like </likert> that cue the model into a closing-
         tag mode). Caught by the same substring replace.
      3. PARTIAL marker emitted as ordinary tokens, then truncated by
         max_tokens or EOS mid-marker: text ends with e.g. `<tur` or
         `<turn` (a prefix of `<turn|>`). Caught by the trailing-prefix
         scan below. We only strip prefixes of length ≥ 2 so we don't
         eat legitimate trailing `<` characters in normal prose.
    """
    if _TURN_OPEN_TEXT in text or _TURN_CLOSE_TEXT in text:
        text = text.replace(_TURN_OPEN_TEXT, "").replace(_TURN_CLOSE_TEXT, "")
    # Trailing-prefix strip: walk longest → shortest, stop at first match.
    for marker in (_TURN_CLOSE_TEXT, _TURN_OPEN_TEXT):
        for plen in range(len(marker) - 1, 1, -1):
            if text.endswith(marker[:plen]):
                return text[:-plen]
    return text


class _StreamTurnMarkerStripper:
    """Stateful filter that suppresses `<|turn>` and `<turn|>` markers
    from streamed content deltas.

    Both markers are atomic special tokens (ids 105, 106) — detokenize()
    yields each as a single full string, so within a single delta they
    appear in full or not at all. We still keep a small tail buffer of
    (max_marker_len - 1) chars in case a future detokenize change ever
    splits a marker across deltas. Drop the literal substring; do NOT
    drop surrounding text (these are point markers, not span openers).
    """

    __slots__ = ("tail",)

    def __init__(self):
        self.tail = ""

    def feed(self, chunk: str) -> str:
        buf = self.tail + chunk
        # Strip both markers from the buf as far as we can confidently.
        # Hold back the last (marker_len - 1) chars in case a marker is
        # being assembled across delta boundaries.
        hold = max(len(_TURN_OPEN_TEXT), len(_TURN_CLOSE_TEXT)) - 1
        safe_end = max(0, len(buf) - hold)
        safe_part = buf[:safe_end].replace(_TURN_OPEN_TEXT, "").replace(_TURN_CLOSE_TEXT, "")
        self.tail = buf[safe_end:]
        return safe_part

    def flush(self) -> str:
        """Emit the held tail at end-of-stream, stripped of any whole
        markers that happen to be entirely within it."""
        t = self.tail.replace(_TURN_OPEN_TEXT, "").replace(_TURN_CLOSE_TEXT, "")
        self.tail = ""
        return t


class _StreamMarkerStripper:
    """Stateful filter that suppresses `<|tool_call>...<tool_call|>` spans
    from streamed content deltas.

    The marker tokens are atomic in the tokenizer, so the most common
    case is that `<|tool_call>` arrives whole inside a single delta_text.
    But because detokenize() coalesces multiple new_tokens per stream
    update, and because the surrounding text may end inside a marker
    span, we keep state across calls:

      - `in_marker` is True iff we have seen an open marker and have
        not yet seen its matching close. While in this state, ALL
        incoming text is suppressed.
      - `tail` is up to (max_marker_len - 1) characters held back from
        the previous yield, in case the delta boundary fell inside a
        marker literal (e.g. delta ends with `<|tool_ca` and the next
        delta starts with `ll>...`). This way we never yield a partial
        marker prefix.

    Note: this is a stripper, not a parser. It does NOT attempt to
    materialise tool_calls out of the suppressed spans during streaming.
    Bridge's existing end-of-stream `_extract_tool_calls(full_text, ...)`
    call is the authoritative parser; this just stops the literal marker
    bytes from leaking into the client's visible prose mid-stream.
    """

    __slots__ = ("in_marker", "tail")

    def __init__(self):
        self.in_marker = False
        self.tail = ""

    def feed(self, chunk: str) -> str:
        """Consume `chunk`, return the portion safe to emit now."""
        buf = self.tail + chunk
        out: list[str] = []
        i = 0
        n = len(buf)
        m_open = _TOOL_CALL_OPEN
        m_close = _TOOL_CALL_CLOSE
        # Max bytes we might need to hold back so we don't accidentally
        # emit a partial open or close marker. -1 because if we have a
        # FULL marker we should consume it now, not hold it back.
        hold = max(len(m_open), len(m_close)) - 1
        while i < n:
            if not self.in_marker:
                idx = buf.find(m_open, i)
                if idx == -1:
                    # No more opens visible. Emit everything except a
                    # trailing hold-window that might be a marker prefix.
                    safe_end = max(i, n - hold)
                    out.append(buf[i:safe_end])
                    i = safe_end
                    break
                out.append(buf[i:idx])
                self.in_marker = True
                i = idx + len(m_open)
            else:
                idx = buf.find(m_close, i)
                if idx == -1:
                    # In-marker tail extends beyond this chunk; hold the
                    # last `hold` bytes in case they're part of a close
                    # marker prefix split across deltas. Drop the rest.
                    i = max(i, n - hold)
                    break
                self.in_marker = False
                i = idx + len(m_close)
        # Whatever's left at [i:] becomes the new tail.
        self.tail = buf[i:]
        return ''.join(out)

    def flush(self) -> str:
        """Emit any held-back tail at end-of-stream.

        At stream end, a leftover tail is either:
          - some out-of-marker text that happened to look like a marker
            prefix (we should emit it verbatim), or
          - in-marker bytes with an unclosed open (we should drop them).
        """
        t = self.tail
        self.tail = ""
        if self.in_marker:
            self.in_marker = False
            return ""
        return t


def _iter_tool_call_blocks(content: str):
    """Yield (start, end, body) for each `<|tool_call>...<tool_call|>`
    block found in `content`. The OUTER markers are atomic tokenizer
    tokens that do not nest, so this find/skip walk is equivalent to
    `re.compile(r'<\\|tool_call>(.*?)<tool_call\\|>', re.DOTALL).finditer`.
    """
    pos = 0
    while True:
        s = content.find(_TOOL_CALL_OPEN, pos)
        if s < 0:
            return
        body_start = s + len(_TOOL_CALL_OPEN)
        e = content.find(_TOOL_CALL_CLOSE, body_start)
        if e < 0:
            # Unclosed opener — bail; matches prior regex behaviour
            # (the non-greedy `.*?` couldn't anchor a close either).
            return
        yield s, e + len(_TOOL_CALL_CLOSE), content[body_start:e]
        pos = e + len(_TOOL_CALL_CLOSE)


def _extract_tool_calls(content: str,
                         had_tools: bool) -> tuple[str, list[dict] | None]:
    """Extract tool calls from a model response into OpenAI shape.

    Returns (cleaned_content, tool_calls_or_None). When a parse fails
    on a particular block, the block stays in cleaned_content as-is
    so the client at least sees the bytes (debugging affordance) and
    the parse error logs to stderr for triage.

    Marker stripping is unconditional: even when had_tools is False
    (the caller did not register any tool grammar this turn), the
    model can still spontaneously emit `<|tool_call>...<tool_call|>`
    spans. Those are sampling drift / hallucination artefacts. We
    strip them from the content so they don't leak into the user-
    facing prose, but we do NOT emit them as structured tool_calls
    (because there are no tools to dispatch against and the parse
    would be meaningless). The diagnostic residue pathway above is
    only reached on the had_tools=True branch.
    """
    if _TOOL_CALL_OPEN not in content:
        return content, None
    if not had_tools:
        # Strip the spans verbatim; drop any inner text. The model
        # got into tool-call-token territory without being asked to,
        # so the inner content is not a real tool invocation — it's
        # the assistant freelancing markup that has no consumer.
        cleaned_parts: list[str] = []
        last_end = 0
        for block_start, block_end, _body in _iter_tool_call_blocks(content):
            cleaned_parts.append(content[last_end:block_start])
            last_end = block_end
        cleaned_parts.append(content[last_end:])
        return ''.join(cleaned_parts).strip(), None

    tool_calls: list[dict] = []
    cleaned_parts: list[str] = []
    last_end = 0
    for block_start, block_end, body in _iter_tool_call_blocks(content):
        cleaned_parts.append(content[last_end:block_start])
        body = body.strip()
        try:
            name, args = gtc.parse_tool_call_body(body)
        except gtc.ToolCallParseError as e:
            # HARD INVARIANT: every tool-call attempt that reached the
            # bridge MUST leave a permanent diagnostic residue in the
            # response — what was submitted, what happened, where the
            # parser failed. No silent drops.
            #
            # Implementation: emit a synthetic tool_calls entry whose
            # name (__tool_call_parse_error__) is not a registered
            # tool. ST's ToolManager dispatches, the dispatch returns
            # an error string ("No tool with the name X has been
            # registered."), and ST records the full record —
            # function.name + function.arguments (which carry the
            # parse error message AND the raw body verbatim) + the
            # dispatch error — in extra.tool_invocations. ST's
            # tool-invocations-collapsible renders all of this as the
            # user-visible diagnostic.
            #
            # Position info: gtc.parse_tool_call_body's exception
            # messages already include "at pos N" markers from the
            # recursive-descent parser. The full str(e) carries the
            # location of the syntactic failure.
            #
            # NOTE on framing: tools shouldn't have failure modes by
            # design (require args that can't be reliably produced).
            # The diagnostic residue here is the safety net for cases
            # where the tool design failed to anticipate an input
            # shape — it surfaces "the integration broke" instead of
            # silently producing nothing. The fix for a recurring
            # parse failure is to redesign the tool's input grammar,
            # not to lean harder on this diagnostic.
            err_str = str(e)
            print(f"[bridge] tool_call parse failed: {err_str}; "
                  f"emitting synthetic diagnostic tool_call. "
                  f"body length={len(body)}, body={body!r}")
            tool_calls.append({
                "id": f"call_parse_error_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": "__tool_call_parse_error__",
                    "arguments": json.dumps({
                        "bridge_parse_error": err_str,
                        "raw_tool_call_body": body,
                        "raw_body_length": len(body),
                        "block_start_offset_in_content": block_start,
                        "block_end_offset_in_content": block_end,
                        "diagnostic": (
                            "The tool_call body emitted between "
                            "<|tool_call> and <tool_call|> failed to "
                            "parse against the gemma tool_call body "
                            "grammar (see server/gemma_tool_call_parser.py). "
                            "This is a tool-design or grammar issue. "
                            "The raw body is preserved verbatim above "
                            "for inspection. Action: look at the body "
                            "structure relative to the parser's grammar "
                            "and decide whether to fix the parser, "
                            "relax the grammar, or redesign the tool's "
                            "input shape so the failure case can't "
                            "arise."),
                    }, ensure_ascii=False),
                },
            })
            cleaned_parts.append(content[block_start:block_end])
            last_end = block_end
            continue
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
        last_end = block_end
    cleaned_parts.append(content[last_end:])
    cleaned = ''.join(cleaned_parts).strip()
    return cleaned, (tool_calls if tool_calls else None)


def _build_stream_spec(stream_id: int,
                       messages: list[dict],
                       sampling: g.SamplingParams,
                       capture_logits: bool,
                       tools: list | None = None,
                       enable_thinking: bool = False,
                       capture_cvec_activations: bool = False,
                       control_vectors: list | None = None,
                       continue_final_message: bool = False) -> tuple[g.StreamSpec, list[StoredSegment]]:
    """OpenAI messages → (StreamSpec, delta_segments).

    Bridge is stateless across chat completions: each request renders
    canonically through render_chat() + chunks_to_segments. There is
    no warm-path adoption from a prior conversation-state — multi-turn
    KV reuse comes from the engine's content-hash KV page cache
    (passive accelerator at the page_manager layer, not bridge state).

    Removed 2026-05-07 per the 'NO REMOTE LOCKS / no entanglement'
    principle: the bridge previously built a warm-conversation path
    that hashed messages[:-1] to a known prior turn's prefix tokens.
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
            messages,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
            tools=tools,
            enable_thinking=enable_thinking)
    except ValueError as e:
        # image_url that isn't a data: URI is a client-shape error,
        # not a server fault. continue_final_message also raises
        # ValueError when the last message isn't role='assistant';
        # surface that as a 400 too.
        msg = str(e)
        if ("image_url" in msg
                or "continue_final_message" in msg):
            raise HTTPException(400, msg) from None
        raise

    delta_segments = _chunks_to_segments(delta_chunks, add_bos=True)
    # 2026-05-23 DEBUG: log a hash of the FIRST N tokens of the rendered
    # prompt so we can see if successive ST requests have the same
    # prefix (which they should — tools + system are universal).
    import hashlib as _hl
    _all_tokens = []
    for s in delta_segments:
        if s.kind == 0:
            _all_tokens.extend(s.tokens)
    _pfx512 = _all_tokens[:512]
    _h = _hl.sha1(repr(_pfx512).encode()).hexdigest()[:12]
    print(f"[bridge DEBUG] prompt prefix hash (first 512 toks): {_h} "
          f"| total_tokens={len(_all_tokens)} | first 20 toks: {_all_tokens[:20]}",
          flush=True)
    spec_segments = [
        g.Segment(kind=s.kind, tokens=list(s.tokens), image_bytes=s.image_bytes)
        for s in delta_segments
    ]
    flags = 0
    if capture_logits:
        flags |= 0x01
    if enable_thinking:
        flags |= 0x02
    if capture_cvec_activations:
        flags |= 0x04
    return g.StreamSpec(
        stream_id=stream_id, action=0,
        flags=flags,
        segments=spec_segments, sampling=sampling,
        control_vectors=control_vectors or [],
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

    # Gemma-4's tool-call close marker. Used as a stop sequence on every
    # chat-completion that ships tools[]: as soon as the model emits
    # <tool_call|> the engine self-terminates with done_reason=1
    # (eos-equivalent), no wasted decode of the downstream
    # <|tool_response> hallucinations.
    #
    # 2026-05-07 (regression fix): MUST be hardcoded to the atomic vocab
    # ID 49, not produced via g.tokenize("<tool_call|>"). Same pitfall as
    # _TURN_END_TOKENS=[106] documented below: `g.tokenize` BPE-splits
    # the literal angle-bracket string into multiple tokens, but the
    # *model* emits a single atomic token id 49 (the special-token entry
    # in tokenizer.json). A multi-token BPE stop-sequence never matches
    # the model's single-token emission, so the engine NEVER STOPS at the
    # marker — generation continues past the close marker into hallucinated
    # post-call content (often more tool calls, or a fake <|tool_response>
    # block). When max_tokens was 256 this was masked because truncation
    # cut the trailing garbage early enough; the 256→4096 default bump on
    # 2026-05-07 unmasked it and broke client-side tool-call parsers.
    #
    # Atomic vocab IDs (from /Users/mdot/models/gemma-4-a4b-bf16/tokenizer.json
    # added_tokens, all special=true):
    #     <|tool>          : 46     <tool|>          : 47
    #     <|tool_call>     : 48     <tool_call|>     : 49
    #     <|tool_response> : 50     <tool_response|> : 51
    global _TOOL_CALL_CLOSE_TOKENS
    _TOOL_CALL_CLOSE_TOKENS = [49]
    print(f"[bridge] tool_call close tokens: {_TOOL_CALL_CLOSE_TOKENS} "
          f"(<tool_call|> atomic vocab id)", flush=True)

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
          f"(<turn|> — Gemma-4 turn-close marker; renders as the literal "
          f"7-char string and is stripped from user-facing content)", flush=True)

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

    # 2026-05-24: retention sweep removed. Eviction is now
    # resource-bounded (BRIDGE_MAX_RETAINED_BYTES + MAX_COMPLETED_STREAMS),
    # not wallclock-bounded. `_evict_completed_streams_to_bounds` runs
    # synchronously inside `_mark_stream_completed_for_eviction` — no
    # background task needed.


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
@app.get("/v1/engine/state")
def engine_state() -> JSONResponse:
    """Snapshot of engine-internal state for the static visualizers.

    Data NOT scoped to a single completion: KV-cache page tenancy
    (which phys pages are live, what their refcounts are, which pages
    are content-cached), the vision cache aggregate hit/miss counters,
    and the active-stream registry (state, position, pages_owned).
    Per-completion telemetry (cache_hits, vision_cache_hits, token
    counts) is on the OAI usage block — clients should read it from
    there. See docs/engine_telemetry_endpoint.md for the principle.

    Polling-friendly: cheap to invoke at 1-2 Hz from a browser. The
    underlying FFI call walks active streams + live pages under the
    same gEngineLock that serializes the AR loop; payload is < 256 KB
    in the worst realistic case (8192-page pool, 8 active streams).

    Bridge enrichment (2026-05-23): each entry in `active_streams[]` is
    augmented with a `tok_per_sec` field (rolling rate over the last
    ~_RATE_WINDOW_SAMPLES StreamUpdate observations on the bridge side
    — see _compute_stream_tok_per_sec). 0.0 for streams that haven't
    emitted at least two updates yet (pure-prefill, just-submitted, or
    GC'd). `tokens_emitted_so_far` is added as a stable alias for
    `completion_tokens_emitted` so consumers that don't know the
    Swift-side field name can still derive their own rates.
    """
    snapshot = g.engine_state()
    streams = snapshot.get("active_streams") or []
    for entry in streams:
        try:
            sid = int(entry.get("stream_id"))
        except (TypeError, ValueError):
            entry["tok_per_sec"] = 0.0
            continue
        entry["tok_per_sec"] = _compute_stream_tok_per_sec(sid)
        if "tokens_emitted_so_far" not in entry:
            entry["tokens_emitted_so_far"] = entry.get(
                "completion_tokens_emitted", 0)
    return JSONResponse(snapshot)


@app.get("/health")
def health() -> JSONResponse:
    s = g.status()
    # KV-cache page telemetry (2026-06, KV-retention/connection-decoupling
    # acceptance). test_kv_no_connection_pinning + test_kv_eviction_guarantees
    # read these directly off /health:
    #   - total_pages : pool capacity (grows under G2 dynamic-pool demand);
    #       the eviction tests derive pinned = total - free - cached.
    #   - free_pages / cached_pages : pool occupancy split.
    #   - resident_sessions : the residency gauge (sessions holding KV pages).
    #       T2 of the pinning leak detector asserts this stays BOUNDED, not
    #       climbing toward MAX_RESIDENT_SESSIONS with the connection count.
    #       Aliased as resident_count for the test's field-name tolerance.
    #   - cache_hits : engine-wide monotonic prefix-adoption tally
    #       (cross-session; the per-request usage block carries the
    #       per-completion figure the G1/G3/G4/G5 assertions key on).
    return JSONResponse({
        "status": "ready",
        "model": MODEL_NAME,
        "multimodal": g.vision_is_ready(),
        "active_streams": s.active_streams,
        "total_pages": s.total_pages,
        # G2 dynamic-pool growth observable (2026-06): total_pages is the
        # CONSTANT budget cap (so the pinned derivation total-free-cached
        # stays correct); committed_pages is the high-water of pages the pool
        # has actually exposed, which RISES under demand. G2 asserts growth on
        # committed_pages. pool_capacity_pages == total_pages, surfaced for
        # explicitness.
        "committed_pages": s.committed_pages,
        "pool_capacity_pages": s.pool_capacity_pages,
        "cached_pages": s.cached_pages,
        "free_pages": s.free_pages,
        # pinned = refcount>0 pages = total - free - cached. Surfaced
        # pre-computed so the leak detector doesn't have to re-derive it
        # (it still does, for field-name tolerance, but this is canonical).
        "pinned_pages": max(0, s.total_pages - s.free_pages - s.cached_pages),
        "resident_sessions": s.resident_sessions,
        "resident_count": s.resident_sessions,
        "cache_hits": s.cache_hits,
        # Tier-1 cold-KV SSD store telemetry (2026-06). used_slots <= max_slots
        # is the in-tier-eviction bound invariant (soak: used == max at cap);
        # demote/reload counts surface the cold-cache value (low32).
        "kv_ssd_tier": {
            "used_slots": s.ssd_used_slots,
            "max_slots": s.ssd_max_slots,
            "demote_count": s.ssd_demote_count,
            "reload_count": s.ssd_reload_count,
        },
        "vision_cache_entries": s.vision_cache_entries,
        "vision_cache_hits": s.vision_cache_hits,
        "total_steps": s.total_steps,
        "total_tokens_emitted": s.total_tokens_emitted,
        # 2026-05-23: live bridge-side aggregate throughput across all
        # currently-tracked streams. Computed from the same rolling
        # _stream_rate_samples deque that backs /v1/engine/state's
        # per-stream tok_per_sec — sum is exact across streams; each
        # stream's rate is its own deque window so a freshly-attached
        # stream contributes 0.0 until it has two samples. Cheap: O(n)
        # over active streams.
        "aggregate_tok_per_sec": _compute_aggregate_tok_per_sec(),
        "active_stream_count": s.active_streams,
        "bridge_stream_logs": {
            "tracked": len(_stream_logs),
            "retained_completed": len(_completed_stream_lru),
            "retained_bytes_est": _retained_log_bytes(),
            "background_drains": len(_background_drains),
        },
        "capabilities": {
            # Renamed from max_q_len (2026-05-24): this is the GPU prefill
            # CHUNK size (256 tokens per kernel dispatch), NOT the context
            # window.  The old name caused clients that scraped /health to
            # cap their context budget at 256.
            "prefill_chunk_size": g.max_q_len(),
            "context_length": CONTEXT_LENGTH,
            # Declared batch shape so clients saturate the kernel instead of
            # guessing a worker count. See tools/batch_scaler.py.
            "kernel_batch": KERNEL_BATCH,
            "max_sessions": MAX_SESSIONS,
        },
    })


@app.post("/v1/resources/register")
async def register_resource_endpoint(req: Request) -> JSONResponse:
    """Register a binary resource (e.g. a control vector) inside the
    bridge process's libgemma registry. Required because the registry is
    per-process: a client that calls `gemma_register_resource` from its
    OWN Python process puts the bytes in its own dylib copy, which the
    bridge never sees. Body:
        {"kind": "cvec", "id": "<id>", "data_b64": "<base64>"}
    `kind` is currently "cvec" (a fp16 HIDDEN-length direction). `data_b64`
    is the raw bytes base64-encoded. Returns {"rc": int} from the FFI;
    rc != 0 indicates a size-mismatch or other registration failure
    (the bridge prints the reason to stderr).
    """
    body = await req.json()
    kind = str(body.get("kind", ""))
    rid = str(body.get("id", ""))
    data_b64 = body.get("data_b64")
    if not kind or not rid or not data_b64:
        raise HTTPException(400, "kind, id, data_b64 are required")
    try:
        data = base64.b64decode(data_b64)
    except Exception as e:
        raise HTTPException(400, f"data_b64 decode failed: {e}") from None
    rc = await asyncio.to_thread(
        lambda: g.register_resource(kind, rid, data))
    if rc != 0:
        raise HTTPException(
            400,
            f"register_resource({kind!r}, {rid!r}) returned rc={rc} "
            f"(usually size mismatch — see bridge stderr for details)")
    return JSONResponse({"rc": rc, "kind": kind, "id": rid, "bytes": len(data)})


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
    # context_length must be present so that clients (e.g. SillyTavern) do not
    # fall through to their hard-coded legacy default (max_4k = 4095).  The
    # true per-session budget is bounded by full-attention pages:
    # MAX_PAGES_PER_SLOT(8192) × PAGE_FULL(8) = 65 536 tokens. Slide
    # layers can address 131 072, but full-attention is the binding
    # block-table limit.
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "metal-microbench",
            "context_length": CONTEXT_LENGTH,
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
    # max_tokens propagation: caller's body wins, with a conservative
    # default when omitted. 2026-05-18 RCA P6: the previous 4096 default
    # (set 2026-05-07 when there was no consumer-liveness backstop) meant
    # a single misbehaving generation could pin a slot for many minutes.
    # With the bounded-queue + disconnect-poll + engine-side liveness
    # deadline triple now in place, the value matters less for safety,
    # but a high default is still poor hygiene — callers should declare
    # their intent. 1024 is plenty for any "give me a sentence" call
    # (the median chat turn is a few hundred tokens) and forces real
    # long-form callers to be explicit. A WARN log fires when omitted
    # so the failure mode "I forgot to set max_tokens and got truncated"
    # is observable rather than silent.
    if body.get("max_tokens") is None and body.get("max_completion_tokens") is None:
        print(f"[bridge] WARN: max_tokens omitted by caller; "
              f"defaulting to 1024. Callers should set max_tokens "
              f"explicitly to make their intent visible.", flush=True)
    max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 1024)))
    stream = bool(body.get("stream", False))

    # Research-feature body fields that don't yet have unified-FFI
    # equivalents (detectors / triggers / prefix_softs / prefix_kv).
    # Each is planned as a `StreamSpec` extension; until then, 501.
    for k in ("detectors", "triggers", "prefix_softs", "prefix_kv"):
        if body.get(k):
            raise HTTPException(
                501, f"'{k}' not yet wired through the unified FFI; "
                     f"see notes/specs/batch_ffi_abi.md")
    # `controls` (control vectors): translate body["controls"] (list of
    # OAI-shape dicts) into StreamSpec.control_vectors (list of
    # CVApplication). The cvec_id must reference a CV previously
    # uploaded via gemma_register_resource(kind='cvec', ...). Each entry
    # mirrors the CVApplication dataclass — see server/gemma_ffi.py.
    request_controls: list[g.CVApplication] = []
    raw_controls = body.get("controls")
    if raw_controls:
        if not isinstance(raw_controls, list):
            raise HTTPException(400, "`controls` must be a list of dicts")
        for c in raw_controls:
            if not isinstance(c, dict) or not c.get("cvec_id"):
                raise HTTPException(400,
                    "each `controls` entry needs at least a string cvec_id")
            request_controls.append(g.CVApplication(
                cvec_id=str(c["cvec_id"]),
                layer=int(c.get("layer", 0)),
                polarity=float(c.get("polarity", 1.0)),
                peak_magnitude=float(c.get("peak_magnitude", 1.0)),
                attack=float(c.get("attack", 0.0)),
                decay=float(c.get("decay", 0.0)),
                sustain_level=float(c.get("sustain_level", 1.0)),
                release=float(c.get("release", 0.0)),
                shape=int(c.get("shape", 0)),
                units=int(c.get("units", 0)),
                mode=int(c.get("mode", 0)),
                target=float(c["target"]) if c.get("target") is not None else float("nan"),
                transport_scale=float(c.get("transport_scale", 0.0)),
                transport_offset=float(c.get("transport_offset", 0.0)),
            ))

    sampling, capture_logits = _parse_sampling(body, max_tokens)
    # structured_cot: bool → default labels, list → custom labels.
    sc = body.get("structured_cot")
    if sc is True:
        sampling.cot_labels = ["GOAL", "APPROACH", "EDGE"]
    elif isinstance(sc, list) and sc:
        sampling.cot_labels = [str(x).strip() for x in sc if str(x).strip()]

    # OpenAI `response_format` (JSON mode). Two forms supported:
    #   {"type": "json_object"}            — generic "JSON only"
    #   {"type": "json_schema",
    #    "json_schema": {"schema": {...},
    #                     "name": "...",
    #                     "strict": bool}}  — schema-described JSON
    #
    # Implementation: prompt-based (prepend a strong instruction to
    # the messages list). The engine doesn't have grammar-constrained
    # decoding for general JSON yet; structured_cot's grammar is
    # think-block-only. For Gemma-4 26B-A4B the prompt-based approach
    # produces well-formed JSON in practice, but it is NOT a hard
    # guarantee — clients that need bit-strict output should validate
    # post-hoc and retry on parse failures.
    #
    # The rider is added as the first message so it persists across
    # multi-turn rendering. We don't merge into an existing system
    # message because that would silently rewrite caller-supplied
    # system text, and chat-template renderers handle multi-system-
    # message lists fine.
    rf = body.get("response_format")
    if isinstance(rf, dict):
        rf_type = rf.get("type")
        rider: str | None = None
        if rf_type == "json_object":
            rider = (
                "You MUST respond with a single valid JSON object. "
                "Output ONLY the JSON — no prose, no markdown fences, "
                "no preamble or postamble. Begin with `{` and end with `}`."
            )
        elif rf_type == "json_schema":
            schema_obj = rf.get("json_schema") or {}
            schema = schema_obj.get("schema", {})
            name = schema_obj.get("name") or "response"
            try:
                schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                schema_text = "(schema not JSON-serializable)"
            rider = (
                f"You MUST respond with a single valid JSON object "
                f"named '{name}' matching this schema:\n\n{schema_text}\n\n"
                f"Output ONLY the JSON — no prose, no markdown fences. "
                f"Every required field must be present."
            )
        if rider:
            # Prepend as a system message; preserve any caller-supplied
            # system messages by stacking after the rider.
            messages = [{"role": "system", "content": rider}] + list(messages)
            print(f"[bridge] response_format={rf_type!r}: rider injected "
                  f"({len(rider)} chars); messages now {len(messages)}")
    # OpenAI tool calling: tools[] → injected via chat template's native
    # <|tool>declaration:...<tool|> blocks.
    tools = body.get("tools") if isinstance(body.get("tools"), list) else None
    # tool_choice modes: only "auto" and "none" (and the implicit absent
    # case, which is auto) are wired through. "required" and named-
    # function modes need grammar-constrained decoding which the engine
    # doesn't expose yet — return 501 rather than silently letting the
    # model decide on its own (a client passing "required" expecting
    # forced tool emission would otherwise get a misleading no-tool
    # response).
    tc = body.get("tool_choice")
    if tc is None or tc == "auto":
        pass  # default behavior, model decides
    elif tc == "none":
        # Drop tools entirely — model produces a normal text reply.
        tools = None
    elif tc == "required" or (isinstance(tc, dict) and tc.get("type") == "function"):
        raise HTTPException(
            501,
            f"tool_choice={tc!r} not implemented; only 'auto' and 'none' "
            f"are supported. 'required' and named-function modes need "
            f"grammar-constrained decoding which the engine doesn't "
            f"expose yet.",
        )
    else:
        raise HTTPException(400, f"unrecognized tool_choice={tc!r}")
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
    # Wire user-supplied OpenAI-shape `stop` strings into the engine's
    # stop_sequences. Each string is tokenized and added as one
    # sequence; multi-token sequences are matched against the
    # recently-emitted token tail.
    user_stop = body.get("stop")
    if isinstance(user_stop, str):
        user_stop = [user_stop]
    if isinstance(user_stop, list):
        for s in user_stop:
            if not isinstance(s, str) or not s:
                continue
            try:
                toks = g.tokenize(s)
                if toks:
                    sampling.stop_sequences.append(list(toks))
            except Exception as e:
                print(f"[bridge] could not tokenize stop string {s!r}: {e}")
    # PYTHONUNBUFFERED=1 (set by the lifecycle launcher) means every
    # write hits stdout immediately; the explicit `flush=True` is
    # gratuitous lock acquisition on each per-request print.
    # `reasoning_effort` (OpenAI o1-shape): any non-null/non-"none"
    # value enables the model's thinking-channel. The flag flows
    # through render_chat (template emits <|think|> prelude) and
    # through the FFI flags bit so the engine routes channel-block
    # tokens to its thinkingQueue instead of dropping them. The
    # bridge then surfaces drained thinking content as
    # `reasoning_content` in the response.
    re_raw = body.get("reasoning_effort")
    enable_thinking = bool(re_raw) and (str(re_raw).lower() != "none")
    # `capture_cvec_activations` (vendor extension, not in OAI spec):
    # opt-in flag that asks the engine to report per-(token, ActiveControl)
    # magnitude + layer records back via `delta.cvec_activations` in the
    # SSE stream. Used by `server/static/steering.html` for the per-token
    # cvec heatmap. When unset, the engine's instrumentation path is
    # dormant — apply kernels still run identically, the per-step records
    # just don't get queued. Off-path bit-identity verified by
    # tools/cvec_validation/baseline_logprobs.py.
    capture_cvec = bool(body.get("capture_cvec_activations", False))
    # `continue_final_message` (OpenAI/vLLM-compat extension): resume
    # generation INSIDE the existing final assistant message instead of
    # opening a fresh turn. The last message in `messages` must be
    # role='assistant'; the bridge renders the chat history with the
    # trailing `<turn|>\n` closer stripped and add_generation_prompt
    # off, so the very next sampled token continues the partial
    # assistant text. This is the canonical client-side resume path
    # paired with the engine's content-hash KV cache: a client whose
    # SSE socket dropped mid-stream simply resubmits with the partial
    # tokens it received appended to the assistant message and
    # continue_final_message=true; the cache makes the re-prefill
    # nearly free up to the resume point.
    continue_final_message = bool(body.get("continue_final_message", False))
    if continue_final_message:
        # 400-class validation: enforce the precondition before
        # touching the engine. render_chat raises ValueError too but
        # surfacing it from here gives a cleaner error for plain
        # config mistakes (e.g. forgetting to append the partial).
        if not messages or messages[-1].get("role") != "assistant":
            raise HTTPException(
                400,
                "continue_final_message=true requires messages[-1].role"
                "='assistant'; pass the partial content the client "
                "already received as the final assistant message and "
                "the engine will continue from where the previous "
                "request was cut off (content-cache makes the "
                "re-prefill cheap).")
    print(f"[bridge] chat_completions: tools={len(tools) if tools else 0}, "
          f"tool_choice={body.get('tool_choice')!r}, "
          f"messages={len(messages)}, stream={stream}, "
          f"stop_seqs={len(sampling.stop_sequences)}, "
          f"reasoning_effort={re_raw!r}, capture_cvec={capture_cvec}, "
          f"continue_final_message={continue_final_message}", flush=True)
    # 2026-05-23 DEBUG (st-cache investigation): dump big-prompt requests
    # to disk so we can replay them via curl deterministically.
    import os as _os, json as _json
    if _os.environ.get("BRIDGE_DUMP_REQUESTS") == "1" and len(messages) >= 2:
        try:
            _dump_path = f"/tmp/bridge_request_{int(__import__('time').time()*1000)}.json"
            with open(_dump_path, "w") as _f:
                _json.dump(body, _f, indent=2, default=str)
            print(f"[bridge DEBUG] dumped request to {_dump_path}", flush=True)
        except Exception as _e:
            print(f"[bridge DEBUG] dump failed: {_e}", flush=True)
    stream_id = await _next_stream_id_alloc()
    # 2026-05-28: optional exact-token recording. When the client sets
    # `return_token_ids: true`, the non-streaming response carries the raw
    # emitted completion token ids AND the assembled prompt token layout
    # (per-segment text token ids — incl. BOS/turn markers since they come
    # from the applied chat template — plus image soft-token span counts).
    # Lets harnesses log token-in/token-out indices (incl. image soft tokens)
    # without re-tokenizing or scraping the bridge log. See
    # tools/svg_elicit/recorder.py.
    return_token_ids = bool(body.get("return_token_ids", False))
    spec, delta_segments = _build_stream_spec(
        stream_id, messages, sampling, capture_logits, tools=tools,
        enable_thinking=enable_thinking,
        capture_cvec_activations=capture_cvec,
        control_vectors=request_controls,
        continue_final_message=continue_final_message)
    # 2026-05-23 APPEND-LOG REFACTOR: install both the per-stream log
    # (the payload substrate) and the notification queue (sentinel-only).
    # The log accepts every StreamUpdate from the engine driver thread;
    # consumers walk an offset cursor over the log. The queue's `put_nowait(None)`
    # wakes any awaiting consumer. Queue maxsize is small (64) because
    # notifications are coalesced — multiple appends between consumer
    # reads just result in (at most) one wakeup, which is fine.
    _stream_logs[stream_id] = []
    response_q: asyncio.Queue = asyncio.Queue(maxsize=64)
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
    try:
        rc = g.submit([spec])
    except ValueError as e:
        _release_stream_log(stream_id)
        raise HTTPException(400, f"invalid engine request: {e}") from e
    if rc != 0:
        _release_stream_log(stream_id)
        raise HTTPException(500, f"engine submit failed: rc={rc}")
    # 2026-05-23 DEBUG (st-cache investigation): trace lifecycle so we can
    # see if SSE consumers ever start iterating after submit.
    print(f"[bridge DEBUG] submitted stream_id={stream_id} rc={rc} stream={stream}",
          flush=True)

    # ADMISSION-PRESSURE-CANCEL (2026-06): the engine refuses a new session
    # under page pressure by emitting a synthetic terminal update
    # (done_reason=3, errMsg "admission backpressure: ..."). When that
    # happens AND there is a sheddable no-active-consumer generation
    # squatting KV pages, the bridge kills+frees the lowest-value such
    # generation (whole, via action=2 — never pause) and RE-SUBMITS this
    # request once. This converts a transient pressure spike into a clean
    # retry instead of a bounced 503 while a disconnected zombie holds
    # pages. The number of relief rounds is bounded so we never spin.
    stream_id = await _resubmit_after_shed_if_backpressure(
        stream_id, spec, response_q)

    if not stream:
        # Aggregate (non-streaming) path. Iterates the shared consumer
        # coroutine — all cross-cutting concerns (disconnect detection,
        # opcode-2 cancel on non-clean exit, response_q cleanup, usage
        # logging) live inside _consume_engine_stream. This branch
        # owns ONLY response-shape construction.
        all_tokens: list[int] = []
        all_thinking_tokens: list[int] = []
        collected_logprobs: list[g.TokenLogprob] = []
        terminal: g.StreamUpdate | None = None
        # The aggregate branch ignores offsets — it just accumulates.
        async for _offset, u in _consume_engine_stream(
                stream_id, req, retain_on_clean_close=False):
            all_tokens.extend(u.new_tokens)
            all_thinking_tokens.extend(u.new_thinking_tokens)
            if u.logprobs:
                collected_logprobs.extend(u.logprobs)
            # The engine self-terminates on stop_sequences (chat-template
            # end-of-turn, plus <tool_call|> when tools are present).
            # state==2 is the natural completion signal; consumer keeps
            # iterating and exits the loop on the next pass.
            if u.state == 2:
                terminal = u
        if terminal is None:
            # Consumer exited without ever observing state==2 → client
            # disconnected mid-stream. The shared coroutine already
            # submitted opcode-2 cancel + logged usage; nothing else
            # to do here. 499 is the conventional "client closed
            # request" status (nginx-originated, widely understood).
            # uvicorn drops the response if the socket is gone.
            raise HTTPException(499, "client closed request "
                                      "before engine completed; "
                                      "engine session cancelled")
        # Terminal update observed. Map done_reason and build response.
        usage = {
            "prompt_tokens": terminal.prompt_tokens_seen,
            "completion_tokens": terminal.completion_tokens_emitted,
            "total_tokens": terminal.prompt_tokens_seen + terminal.completion_tokens_emitted,
            "cache_hits": terminal.cache_hits,
            "cache_misses": terminal.cache_misses,
            "vision_cache_hits": terminal.vision_cache_hits,
        }
        done_reason = terminal.done_reason
        err_msg = terminal.err_msg or ""
        # done_reason=3 is the engine-side error code. Two flavors:
        #   - Admission backpressure (engine refused new session
        #     because pool was below the free-page floor or the
        #     residency cap was hit). errMsg starts with "admission
        #     backpressure:". Surface as HTTP 503 + Retry-After so
        #     clients back off and retry. This is the right status
        #     for week-long ops where many clients open and close
        #     sessions on their own cadence.
        #   - Other engine failures (vision tower returned 0 soft
        #     tokens, consumer-abandonment, etc.). Surface as HTTP 500.
        # In both cases, errMsg carries the human-readable cause.
        if done_reason == 3:
            print(f"[bridge] stream errored: done_reason=3 err_msg={err_msg!r}")
            # PERMANENT (-> 413): the request can never be served (slot
            # block-table cap or KV pool capacity exceeded). NO retry, NO
            # Retry-After — the client must shrink prompt + max_tokens.
            # Checked before backpressure (prefixes are disjoint; order is
            # only for clarity).
            if err_msg.startswith("context too large"):
                raise HTTPException(413, f"context too large: {err_msg}")
            # TRANSIENT (-> 503): fits-but-not-now (pool floor / residency
            # cap / fit-check / mid-prefill pool exhaustion). Retryable.
            if err_msg.startswith("admission backpressure"):
                raise HTTPException(
                    503,
                    f"engine admission backpressure: {err_msg}",
                    headers={"Retry-After": "2"})
            # Generic real engine error (vision 0 soft tokens, consumer
            # abandonment, etc.) -> 500.
            raise HTTPException(500, f"upstream stream errored: {err_msg or 'unspecified engine failure'}")
        text = g.detokenize(all_tokens)
        # Strip Gemma-4 turn-delimiter surface strings (`<|turn>`,
        # `<turn|>`) from the final text. These are atomic special
        # tokens whose text form leaks into the output when the
        # model emits them — typically a single `<turn|>` at the
        # very end (immediately before the stop-token detection
        # fires), but also possible mid-stream as a sampling artefact.
        text = _strip_turn_markers(text)
        print(f"[bridge] usage: prompt_tokens={usage.get('prompt_tokens')}, "
              f"completion_tokens={usage.get('completion_tokens')}, "
              f"cache_hits={usage.get('cache_hits')}, "
              f"cache_misses={usage.get('cache_misses')}, "
              f"vision_cache_hits={usage.get('vision_cache_hits')}, "
              f"done_reason={done_reason}")
        finish = "stop" if done_reason == 1 else (
            "length" if done_reason == 2 else "stop")
        # 2026-05-07: server-side tool-call extraction. When tools
        # were in the request and the model emitted
        # `<|tool_call>...<tool_call|>` markers, parse them out of
        # `text` and surface as OpenAI-shape `message.tool_calls[]`.
        # Markers are stripped from `content` so clients see a clean
        # response; finish_reason flips to "tool_calls" per OAI spec.
        # Models that emit malformed blocks have their bytes left
        # verbatim (parse-failure fallback) so debugging stays
        # affordant.
        cleaned_text, extracted_tool_calls = _extract_tool_calls(
            text, had_tools=tools is not None)
        message: dict[str, Any] = {"role": "assistant",
                                    "content": cleaned_text}
        if all_thinking_tokens:
            # The engine routes every token between CHANNEL_OPEN_ID
            # and CHANNEL_CLOSE_ID into thinkingQueue verbatim,
            # which includes the channel-name preamble that gemma's
            # tokenizer-grammar puts there ("thought\n" terminated
            # by the first newline). The OAI o1-shape
            # `reasoning_content` field is supposed to carry the
            # user-visible thinking body, not the format-internal
            # channel header — so we strip the preamble here at the
            # bridge layer. This is api-side translation work, not
            # client-side rendering work: clients never need to
            # know that gemma calls its thinking channel "thought".
            raw = g.detokenize(all_thinking_tokens)
            nl = raw.find("\n")
            message["reasoning_content"] = raw[nl + 1:] if nl >= 0 else raw
        if extracted_tool_calls:
            message["tool_calls"] = extracted_tool_calls
            finish = "tool_calls"
            print(f"[bridge] extracted {len(extracted_tool_calls)} "
                  f"tool_call(s); names="
                  f"{[tc['function']['name'] for tc in extracted_tool_calls]}")
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
                    "id": lp.token,  # raw token id (was dropped — only the str was returned)
                    "token": g.detokenize([lp.token]),
                    "logprob": lp.sampled_logprob,
                    "top_logprobs": [
                        {"id": t, "token": g.detokenize([t]), "logprob": p}
                        for t, p in lp.top_logprobs
                    ],
                } for lp in collected_logprobs]
            }
        if return_token_ids:
            # Exact token-out: raw emitted ids (and thinking ids, if any).
            choice["token_ids"] = list(all_tokens)
            if all_thinking_tokens:
                choice["thinking_token_ids"] = list(all_thinking_tokens)
            # Exact token-in layout: per-segment text token ids (BOS/turn
            # markers included — they come from render_chat) interleaved with
            # image segments. Image soft-tokens are continuous vision
            # embeddings (no discrete ids); their count = prompt_tokens_total
            # - total text tokens, reported here so a recorder can place the
            # exact [start,end) soft-token spans.
            _text_ids = [t for s in delta_segments if s.kind == 0 for t in s.tokens]
            _n_img = sum(1 for s in delta_segments if s.kind != 0)
            choice["prompt_token_layout"] = {
                "segments": [
                    ({"kind": "text", "n": len(s.tokens), "token_ids": list(s.tokens)}
                     if s.kind == 0 else {"kind": "image"})
                    for s in delta_segments
                ],
                "text_token_ids": _text_ids,
                "n_text_tokens": len(_text_ids),
                "n_images": _n_img,
                "prompt_tokens_total": usage["prompt_tokens"],
                "image_soft_tokens_total": usage["prompt_tokens"] - len(_text_ids),
            }
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [choice],
                "usage": usage,
            },
            headers={"X-Stream-Id": str(stream_id)},
        )

    # SSE streaming. The shared _consume_engine_stream coroutine owns
    # all the cross-cutting concerns: client-disconnect detection,
    # opcode-2 cancel on non-clean exit, response_q cleanup, DISCONNECT
    # usage logging. The gen() body below owns ONLY the SSE-frame
    # rendering — symmetric with the aggregate path which owns ONLY
    # the JSON-response construction. Before this refactor the two
    # paths had separately-implemented (and divergent) lifecycle code.
    async def gen():
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        # Tokenwise live streaming. When tools are in the request, the
        # model may emit `<|tool_call>...<tool_call|>` markers as part
        # of the generated text; the markers stream visibly to the
        # client just like any other token. Tool-call extraction runs
        # ONCE at end-of-generation against the accumulated text, and
        # if any markers parsed cleanly we emit a `tool_calls` delta
        # followed by `finish_reason="tool_calls"`. Clients that
        # understand OpenAI tool_calls (SillyTavern's tool-calling
        # extractor, openai's own SDK) consume the structured shape;
        # clients that don't will see the raw markers as content.
        had_tools = tools is not None
        all_tokens: list[int] = []
        # Per-generator state for stripping the channel-name preamble
        # from streamed reasoning_content. Gemma's thinking channel
        # opens with `<|channel>NAME\n...` and the engine routes every
        # token between the sentinel pair to thinkingQueue verbatim,
        # including the NAME + newline. We strip everything up to and
        # including the first newline; flag flips to True once we've
        # seen it. After that, deltas pass through unchanged. The
        # newline may not be in the first delta — we accumulate
        # pending bytes until it shows up, then emit only the body.
        thinking_header_consumed = False
        thinking_header_buffer = ""
        # Per-generator state for stripping `<|tool_call>...<tool_call|>`
        # spans from streamed content deltas. The model can emit these
        # markers spontaneously (sampling drift) even when no tools were
        # registered, and the raw marker text leaks into client-visible
        # prose if we don't filter mid-stream. End-of-stream extraction
        # (the had_tools branch below) still runs against the full
        # accumulated text and supersedes this filter for the structured
        # tool_calls payload. This is purely a content-cleanliness gate.
        marker_stripper = _StreamMarkerStripper()
        # Same role, different markers: `<|turn>` / `<turn|>` are atomic
        # special tokens whose text leaks into deltas when the model
        # emits them. Run BEFORE the tool-call stripper because the
        # tool-call stripper assumes its input is free of unrelated
        # special-token surface bytes.
        turn_stripper = _StreamTurnMarkerStripper()
        # Wire-format helper for the append-log offset.
        # 2026-05-23: prefix each SSE event with a `: offset=N` comment
        # line so a reconnecting client can checkpoint its progress
        # WITHOUT polluting the OpenAI-shape data payload. SSE comments
        # (any line starting with `:`) are stripped by EventSource
        # parsers and ignored by jq / shell tooling that filters on
        # `data:` prefix — perfectly backwards compatible.
        def _sse(offset: int, data_obj: dict | str) -> str:
            if isinstance(data_obj, str):
                # Pre-formatted (e.g. "[DONE]" sentinel).
                return f": offset={offset}\ndata: {data_obj}\n\n"
            return f": offset={offset}\ndata: {json.dumps(data_obj)}\n\n"
        current_offset = 0
        # Forward-progress contract: announce the stream IMMEDIATELY so it is
        # never silent while the engine prefills. The role delta is the standard
        # OpenAI "assistant is responding" first chunk (the admitted/prefilling
        # signal); the prefilling comment is an SSE keep-alive. Heartbeats then
        # flow every _SSE_HEARTBEAT_S until the first token; a genuine hang is
        # surfaced as a clean error after _SSE_FORWARD_PROGRESS_DEADLINE_S.
        _role_chunk = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(_role_chunk)}\n\n"
        yield ": prefilling\n\n"
        _heartbeats = 0
        async for current_offset, u in _consume_engine_stream(
                stream_id, req, heartbeat_s=_SSE_HEARTBEAT_S):
            if u is None:
                # Heartbeat: engine produced nothing this interval (prefill /
                # queue / hang). Keep the wire alive; surface a genuine hang fast.
                _heartbeats += 1
                if _heartbeats * _SSE_HEARTBEAT_S >= _SSE_FORWARD_PROGRESS_DEADLINE_S:
                    print(f"[bridge] (SSE) forward-progress timeout: no engine "
                          f"update for {_SSE_FORWARD_PROGRESS_DEADLINE_S}s on "
                          f"stream {stream_id}; surfacing engine_stall")
                    yield _sse(current_offset, {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": ""},
                            "finish_reason": "error",
                        }],
                        "error": {
                            "message": f"engine produced no output for {int(_SSE_FORWARD_PROGRESS_DEADLINE_S)}s",
                            "type": "engine_stall",
                        },
                    })
                    yield _sse(current_offset, "[DONE]")
                    return
                yield ": heartbeat\n\n"
                continue
            _heartbeats = 0
            if u.new_thinking_tokens:
                raw_delta = g.detokenize(u.new_thinking_tokens)
                if not thinking_header_consumed:
                    thinking_header_buffer += raw_delta
                    nl = thinking_header_buffer.find("\n")
                    if nl < 0:
                        # Still in the header bytes; nothing to emit yet.
                        thinking_delta = ""
                    else:
                        thinking_delta = thinking_header_buffer[nl + 1:]
                        thinking_header_consumed = True
                        thinking_header_buffer = ""
                else:
                    thinking_delta = raw_delta
                if thinking_delta:
                    yield _sse(current_offset, {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {"reasoning_content": thinking_delta},
                            "finish_reason": None,
                        }],
                    })
            if u.new_cvec_activations:
                # Vendor extension: per-(ActiveControl, AR-step)
                # records. Each tuple is (token_position, layer,
                # magnitude). See server/static/steering.html for
                # the consumer; off-path bit-identity guard is at
                # tools/cvec_validation/baseline_logprobs.py.
                yield _sse(current_offset, {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {"cvec_activations": [
                            {"token_position": tp, "layer": ly,
                             "magnitude": mg}
                            for (tp, ly, mg) in u.new_cvec_activations
                        ]},
                        "finish_reason": None,
                    }],
                })
            if u.new_tokens:
                all_tokens.extend(u.new_tokens)
                delta_text = g.detokenize(u.new_tokens)
                if delta_text:
                    # Filter `<|tool_call>...<tool_call|>` spans out
                    # of the visible content stream. Markers are
                    # processed structurally at end-of-stream below.
                    # Pre-filter `<|turn>` / `<turn|>` first — those
                    # are point-strips, never wrap tool_call spans.
                    delta_text = turn_stripper.feed(delta_text)
                    filtered = marker_stripper.feed(delta_text)
                    if filtered:
                        yield _sse(current_offset, {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": filtered},
                                "finish_reason": None,
                            }],
                        })
            if u.state == 2:
                # Engine-side error path: done_reason=3 + err_msg. SSE
                # has no clean "error after partial response" shape, so
                # we yield a final delta carrying an OpenAI-shape error
                # event then return — the client sees
                # finish_reason="error" with the message.
                if u.done_reason == 3:
                    err_msg = u.err_msg or "unspecified engine failure"
                    # Distinguish three terminal classes (SSE has already
                    # sent 200 headers, so 413/503 semantics are surfaced
                    # via the OpenAI error type rather than HTTP status):
                    #   context_too_large  -> permanent (413), no retry
                    #   admission_backpressure -> transient (503), retry
                    #   engine_error -> generic (500)
                    if err_msg.startswith("context too large"):
                        err_type = "context_too_large"
                    elif err_msg.startswith("admission backpressure"):
                        err_type = "admission_backpressure"
                    else:
                        err_type = "engine_error"
                    print(f"[bridge] (SSE) stream errored: done_reason=3 type={err_type} err_msg={err_msg!r}")
                    yield _sse(current_offset, {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": ""},
                            "finish_reason": "error",
                        }],
                        "error": {"message": err_msg, "type": err_type},
                    })
                    yield _sse(current_offset, "[DONE]")
                    return
                finish = "stop" if u.done_reason == 1 else (
                    "length" if u.done_reason == 2 else "stop")
                # Flush any held-back tail from the marker strippers
                # before final extraction. Each holds up to
                # (max_marker_len-1) bytes that *might* have been a
                # marker prefix; at stream end we know they aren't.
                # Run turn-stripper first (point markers, can be inside
                # any text), then tool-call stripper (span markers).
                turn_tail = turn_stripper.flush()
                if turn_tail:
                    final_tail = marker_stripper.feed(turn_tail) + marker_stripper.flush()
                else:
                    final_tail = marker_stripper.flush()
                if final_tail:
                    yield _sse(current_offset, {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": final_tail},
                            "finish_reason": None,
                        }],
                    })
                # End-of-generation tool-call extraction. The marker
                # spans were stripped mid-stream; if any parsed cleanly
                # here, emit a structured tool_calls delta and flip
                # finish to "tool_calls" per OAI spec.
                if had_tools:
                    full_text = g.detokenize(all_tokens)
                    # Strip turn markers from the aggregated text used
                    # for tool-call extraction so a stray `<turn|>`
                    # doesn't break the tool_call body parser.
                    full_text = _strip_turn_markers(full_text)
                    _, extracted = _extract_tool_calls(
                        full_text, had_tools=True)
                    if extracted:
                        finish = "tool_calls"
                        print(f"[bridge] (SSE) extracted "
                              f"{len(extracted)} tool_call(s); names="
                              f"{[tc['function']['name'] for tc in extracted]}")
                        tc_deltas = [{
                            "index": i,
                            "id": tc["id"],
                            "type": tc["type"],
                            "function": tc["function"],
                        } for i, tc in enumerate(extracted)]
                        yield _sse(current_offset, {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {"tool_calls": tc_deltas},
                                "finish_reason": None,
                            }],
                        })
                yield _sse(current_offset, {
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
                })
                yield _sse(current_offset, "[DONE]")
                print(f"[bridge] usage (SSE): "
                      f"prompt_tokens={u.prompt_tokens_seen}, "
                      f"completion_tokens={u.completion_tokens_emitted}, "
                      f"cache_hits={u.cache_hits}, "
                      f"cache_misses={u.cache_misses}, "
                      f"vision_cache_hits={u.vision_cache_hits}, "
                      f"done_reason={u.done_reason}")
                return
        # If the async-for exits without hitting state==2, the client
        # disconnected. The engine session keeps running; the append-log
        # keeps growing. A reconnect via `GET /v1/streams/{id}/sse?since=N`
        # can replay from any prior offset within the retention window.
    # X-Stream-Id surfaces the engine-side stream id to the client so
    # it can reconnect to `GET /v1/streams/{id}/sse` if the SSE socket
    # drops mid-generation. See `stream_reconnect_sse` below. We pick
    # the header path over an in-band first SSE chunk to keep the chat-
    # completions stream payload OpenAI-compatible byte-for-byte; the
    # header is invisible to clients that don't know to read it.
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"X-Stream-Id": str(stream_id)})


# ----------------------------------------------------------------------
# Reconnect endpoint.
#
# 2026-05-23 APPEND-LOG REFACTOR: the reconnect endpoint is now a thin
# wrapper that delegates to `_consume_engine_stream(stream_id, req,
# from_offset=since)`. The per-stream append-log preserves every
# StreamUpdate the engine has ever emitted for this stream (within
# the retention window); a reconnect with `?since=N` replays from log
# index N forward, then awaits future appends. This matches real
# OpenAI/Anthropic packet-loss semantics: a client whose socket dropped
# mid-stream can pick up EXACTLY where it left off, including bytes
# its original consumer already drained but never rendered.
#
# Wire format: each SSE event is prefixed with `: offset=N\n` (SSE
# comment line) so the client can update its cursor without parsing
# the OAI-shape JSON. The original POST stream uses the same envelope.
#
# Stream-id surfacing: clients read the `X-Stream-Id` response header
# from the initial POST.
#
# Single-wire-consumer-at-a-time: a 409 is returned if another wire
# reader is already attached. The append-log is multi-reader-safe in
# principle, but emitting the same SSE bytes to two simultaneous
# downstream sockets is wasteful and confusing in logs; keeping the
# "one wire reader at a time" contract sidesteps that.
#
# Retention: after the engine emits state==2 with no consumer attached,
# the log moves into a completed-idle LRU (`_completed_stream_lru`).
# The LRU is bounded by BRIDGE_MAX_RETAINED_BYTES (default 64MB) and
# BRIDGE_MAX_COMPLETED_STREAMS (default 256) — no wallclock, no
# scheduled sweep. A reconnect for a stream in the LRU pops it out
# (becomes active again for the duration of the consumer attach); on
# consumer disconnect it goes back to the END of the LRU (most
# recently used). Eviction happens only when NEW completed streams
# push the LRU over budget; the oldest tail entry gets freed.
# ----------------------------------------------------------------------
@app.post("/v1/streams/{stream_id}/cancel")
async def stream_cancel(stream_id: int) -> JSONResponse:
    """Explicitly RELEASE an in-flight engine session.

    The bridge treats the TCP socket as pure transport (B5): a session
    continues after the HTTP client disconnects so a flaky/reconnecting
    client doesn't lose its generation — disconnect NEVER cancels engine
    work. The corollary is that a client which is genuinely DONE — or a
    test harness that wants the engine to stop NOW rather than run to
    max_tokens with no consumer — must say so explicitly instead of
    relying on socket teardown. This endpoint is that one explicit signal:
    the ONLY bridge path that submits action=2/closeSession.

    Submits the engine cancel opcode (action=2 → closeSession → frees the slot
    + KV pages within one poll tick) and drops the bridge-side stream log. The
    stream_id is the value surfaced in the `X-Stream-Id` response header of the
    original /v1/chat/completions call. Idempotent: cancelling an unknown or
    already-finished stream is a 200 no-op, so clients can fire-and-forget.
    """
    try:
        g.submit([g.StreamSpec(stream_id=stream_id, action=2)])
    except Exception as e:
        raise HTTPException(500, f"cancel submit failed: {e}") from e
    _release_stream_log(stream_id)
    return JSONResponse({"stream_id": stream_id, "cancelled": True})


@app.get("/v1/streams/{stream_id}/sse")
async def stream_reconnect_sse(stream_id: int, req: Request,
                                  since: int = 0):
    """Attach as the (new) wire consumer of an in-flight engine session,
    replaying from log offset `since` (default 0 = full replay).

    Returns 404 if the stream id is unknown (never existed, or already
    GC'd past the retention window after natural termination).
    Returns 409 if another wire consumer is currently attached.
    Returns 200 + SSE on success; emits OpenAI-shape
    chat.completion.chunk frames identical in shape to the initial
    POST stream, each prefixed with a `: offset=N\\n` SSE comment line
    that the client can read to update its replay cursor.
    """
    if stream_id not in _stream_logs:
        raise HTTPException(404, f"stream_id={stream_id} not found "
                                   f"(never existed or already GC'd past "
                                   f"retention window)")
    if stream_id in _active_consumer_token:
        raise HTTPException(409,
            f"stream_id={stream_id} already has an attached wire "
            f"consumer (token={_active_consumer_token[stream_id]!r}); "
            f"single-reader invariant")
    if since < 0:
        raise HTTPException(400, f"since={since} must be >= 0")

    # Claim the wire consumer slot. _consume_engine_stream's finally
    # block will release it (we also clear it explicitly here on
    # exit-paths to keep semantics tidy).
    token = uuid.uuid4().hex[:12]
    _active_consumer_token[stream_id] = token
    # Pop from the completed-idle LRU: this stream is "active again"
    # for the duration of the reconnect. On consumer-exit, if
    # state==2 has been observed, `_mark_stream_completed_for_eviction`
    # re-adds the entry at the END (most-recently-used position).
    _completed_stream_lru.pop(stream_id, None)

    log_len = len(_stream_logs.get(stream_id, []))
    print(f"[bridge] reconnect attached (sid={stream_id}, "
          f"token={token}, since={since}, log_len={log_len}, "
          f"replay_count={max(0, log_len - since)})", flush=True)

    async def gen():
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        all_tokens: list[int] = []
        # Mid-stream marker stripping. A reconnect that picks up
        # mid-stream from offset N may see marker bytes split across
        # the offset boundary; the strippers re-establish invariants
        # from this attach point forward.
        marker_stripper = _StreamMarkerStripper()
        turn_stripper = _StreamTurnMarkerStripper()
        # Reuse the same wire-format helper as the initial POST stream.
        def _sse(offset: int, data_obj: dict | str) -> str:
            if isinstance(data_obj, str):
                return f": offset={offset}\ndata: {data_obj}\n\n"
            return f": offset={offset}\ndata: {json.dumps(data_obj)}\n\n"
        current_offset = since
        try:
            async for current_offset, u in _consume_engine_stream(
                    stream_id, req, from_offset=since):
                if u.new_thinking_tokens:
                    raw_delta = g.detokenize(u.new_thinking_tokens)
                    if raw_delta:
                        yield _sse(current_offset, {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {"reasoning_content": raw_delta},
                                "finish_reason": None,
                            }],
                        })
                if u.new_tokens:
                    all_tokens.extend(u.new_tokens)
                    delta_text = g.detokenize(u.new_tokens)
                    if delta_text:
                        delta_text = turn_stripper.feed(delta_text)
                        filtered = marker_stripper.feed(delta_text)
                        if filtered:
                            yield _sse(current_offset, {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_NAME,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": filtered},
                                    "finish_reason": None,
                                }],
                            })
                if u.state == 2:
                    if u.done_reason == 3:
                        err_msg = u.err_msg or "unspecified engine failure"
                        # Same three-class split as the main SSE path:
                        # context_too_large (413), admission_backpressure
                        # (503), engine_error (500) — surfaced via the
                        # OpenAI error type since SSE headers are sent.
                        if err_msg.startswith("context too large"):
                            err_type = "context_too_large"
                        elif err_msg.startswith("admission backpressure"):
                            err_type = "admission_backpressure"
                        else:
                            err_type = "engine_error"
                        yield _sse(current_offset, {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": ""},
                                "finish_reason": "error",
                            }],
                            "error": {"message": err_msg, "type": err_type},
                        })
                        yield _sse(current_offset, "[DONE]")
                        return
                    finish = "stop" if u.done_reason == 1 else (
                        "length" if u.done_reason == 2 else "stop")
                    turn_tail = turn_stripper.flush()
                    if turn_tail:
                        final_tail = marker_stripper.feed(turn_tail) + marker_stripper.flush()
                    else:
                        final_tail = marker_stripper.flush()
                    if final_tail:
                        yield _sse(current_offset, {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": final_tail},
                                "finish_reason": None,
                            }],
                        })
                    yield _sse(current_offset, {
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
                    })
                    yield _sse(current_offset, "[DONE]")
                    print(f"[bridge] usage (RECONNECT SSE sid={stream_id}): "
                          f"replay_from={since}, final_offset={current_offset}, "
                          f"prompt_tokens={u.prompt_tokens_seen}, "
                          f"completion_tokens={u.completion_tokens_emitted}, "
                          f"cache_hits={u.cache_hits}, "
                          f"cache_misses={u.cache_misses}, "
                          f"vision_cache_hits={u.vision_cache_hits}, "
                          f"done_reason={u.done_reason}", flush=True)
                    return
        finally:
            # _consume_engine_stream's finally has already cleared
            # _active_consumer_token[stream_id]; defensive double-pop
            # is a no-op.
            _active_consumer_token.pop(stream_id, None)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"X-Stream-Id": str(stream_id)})


# ----------------------------------------------------------------------
# Static visualizers (server/static/). These are bespoke engine-internals
# dashboards (tetraplex / labeler / loom / steering) that consume the
# same OAI-compatible chat-completions endpoint as any other client but
# additionally poll /v1/engine/state for the engine-state slice the OAI
# response doesn't carry (KV page tenancy, vision cache, active streams).
# Mounted at /static so the bridge's API namespace (/health, /v1/*) is
# unchanged; the root path redirects to the visualizer index.
# ----------------------------------------------------------------------
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)),
              name="static")

    @app.get("/")
    def _root_redirect() -> RedirectResponse:
        # 302 to the visualizer index. Keeping the bridge's namespace
        # clean (no FileResponse routes for individual pages) — every
        # visualizer is reachable at /static/<name>.html.
        return RedirectResponse(url="/static/clients.html", status_code=302)


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
