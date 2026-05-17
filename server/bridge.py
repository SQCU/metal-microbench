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
                       control_vectors: list | None = None) -> tuple[g.StreamSpec, list[StoredSegment]]:
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
            messages, add_generation_prompt=True, tools=tools,
            enable_thinking=enable_thinking)
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
    """
    return JSONResponse(g.engine_state())


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
    # Default bumped 256 → 4096 on 2026-05-07. The 256 default came from
    # before the engine could sustain long contexts; Gemma-4 rates 128k and
    # the engine kernel-side handles 64k+ full-cache cleanly. Clients that
    # want shorter responses pass max_tokens explicitly.
    max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 4096)))
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
    print(f"[bridge] chat_completions: tools={len(tools) if tools else 0}, "
          f"tool_choice={body.get('tool_choice')!r}, "
          f"messages={len(messages)}, stream={stream}, "
          f"stop_seqs={len(sampling.stop_sequences)}, "
          f"reasoning_effort={re_raw!r}, capture_cvec={capture_cvec}")
    stream_id = await _next_stream_id_alloc()
    spec, _ = _build_stream_spec(
        stream_id, messages, sampling, capture_logits, tools=tools,
        enable_thinking=enable_thinking,
        capture_cvec_activations=capture_cvec,
        control_vectors=request_controls)
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
            all_thinking_tokens: list[int] = []
            usage: dict[str, int] = {}
            done_reason = 0
            collected_logprobs: list[g.TokenLogprob] = []
            while True:
                u: g.StreamUpdate = await response_q.get()
                all_tokens.extend(u.new_tokens)
                all_thinking_tokens.extend(u.new_thinking_tokens)
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
                    err_msg = u.err_msg or ""
                    break
            # done_reason=3 is the engine-side error code. Two flavors:
            #   - Admission backpressure (engine refused new session
            #     because pool was below the free-page floor or the
            #     residency cap was hit). errMsg starts with "admission
            #     backpressure:". Surface as HTTP 503 + Retry-After so
            #     clients back off and retry. This is the right status
            #     for week-long ops where many clients open and close
            #     sessions on their own cadence.
            #   - Other engine failures (vision tower returned 0 soft
            #     tokens, etc.). Surface as HTTP 500.
            # In both cases, errMsg carries the human-readable cause.
            if done_reason == 3:
                print(f"[bridge] stream errored: done_reason=3 err_msg={err_msg!r}")
                if err_msg.startswith("admission backpressure"):
                    raise HTTPException(
                        503,
                        f"engine admission backpressure: {err_msg}",
                        headers={"Retry-After": "2"})
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
        last_update: g.StreamUpdate | None = None
        clean_close = False
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
        try:
            while True:
                u: g.StreamUpdate = await response_q.get()
                last_update = u
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
                        yield ("data: " + json.dumps({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {"reasoning_content": thinking_delta},
                                "finish_reason": None,
                            }],
                        }) + "\n\n")
                if u.new_cvec_activations:
                    # Vendor extension: per-(ActiveControl, AR-step)
                    # records. Each tuple is (token_position, layer,
                    # magnitude). See server/static/steering.html for
                    # the consumer; off-path bit-identity guard is at
                    # tools/cvec_validation/baseline_logprobs.py.
                    yield ("data: " + json.dumps({
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
                    }) + "\n\n")
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
                            yield ("data: " + json.dumps({
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_NAME,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": filtered},
                                    "finish_reason": None,
                                }],
                            }) + "\n\n")
                if u.state == 2:
                    # Engine-side error path: done_reason=3 + err_msg.
                    # Same handling as non-streaming aggregate path above.
                    # SSE has no clean "error after partial response"
                    # shape, so we yield a final delta carrying an
                    # OpenAI-shape error event then break — the client
                    # sees finish_reason="error" with the message.
                    if u.done_reason == 3:
                        err_msg = u.err_msg or "unspecified engine failure"
                        # Distinguish admission backpressure (transient,
                        # client should retry) from genuine engine
                        # failure (likely persistent). Both yield as
                        # SSE error events but with different error
                        # types so clients can react differently.
                        is_backpressure = err_msg.startswith("admission backpressure")
                        err_type = ("admission_backpressure"
                                    if is_backpressure else "engine_error")
                        print(f"[bridge] (SSE) stream errored: done_reason=3 type={err_type} err_msg={err_msg!r}")
                        yield ("data: " + json.dumps({
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
                        }) + "\n\n")
                        yield "data: [DONE]\n\n"
                        clean_close = True
                        break
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
                        yield ("data: " + json.dumps({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": final_tail},
                                "finish_reason": None,
                            }],
                        }) + "\n\n")
                    # End-of-generation tool-call extraction. The marker
                    # spans were stripped mid-stream; if any parsed
                    # cleanly here, emit a structured tool_calls delta
                    # and flip finish to "tool_calls" per OAI spec.
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
                            yield ("data: " + json.dumps({
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_NAME,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"tool_calls": tc_deltas},
                                    "finish_reason": None,
                                }],
                            }) + "\n\n")
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
                          f"done_reason={u.done_reason}")
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
                      f"done_reason={u.done_reason}")
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
