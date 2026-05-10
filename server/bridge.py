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
import json
import re
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
# ────────────────────────────────────────────────────────────────────────
# Per-turn scaffolding bytes the model emits but clients shouldn't see.
# 2026-05-07: surfaced after a SillyTavern-fork user reported
# `<|channel|>thought\n` appearing at the start of responses.
#
# The chat template's `add_generation_prompt=True, enable_thinking=False`
# epilogue is `<|turn>model\n<|channel>thought\n<channel|>` (the empty-
# thought no-thinking marker). The model is supposed to start its
# response AFTER `<channel|>` closes — and does — but in some contexts
# (notably multi-turn flows after a prior tool_call) it ECHOES the
# `<|channel>thought\n<channel|>` structure as the first tokens of its
# output, presumably as a learned formatting habit. The bytes ride
# through detokenize verbatim and bleed into client UIs.
#
# Likewise, `<turn|>` (id 106) is the engine's natural-stop marker
# (`_TURN_END_TOKENS`). It's an EOS signal, not response content;
# clients don't need it.
#
# Conservative strip rules:
#   - Leading `<|channel>BODY<channel|>` ONLY when BODY is short (<= 40
#     chars between markers) — i.e., the empty/short echo. Real
#     thinking content (longer body) is preserved so callers using
#     enable_thinking can still surface it.
#   - Trailing `<turn|>` regardless.
_LEADING_CHANNEL_RE = re.compile(
    r'^\s*<\|channel>[^<]{0,40}<channel\|>\s*',
    re.DOTALL,
)
_TRAILING_TURN_RE = re.compile(r'\s*<turn\|>\s*$', re.DOTALL)


def _strip_response_scaffolding(text: str) -> str:
    """Drop per-turn scaffolding bytes the client shouldn't see.

    See _LEADING_CHANNEL_RE / _TRAILING_TURN_RE comments for what's
    stripped and why. Body content between markers is preserved.
    """
    text = _LEADING_CHANNEL_RE.sub('', text)
    text = _TRAILING_TURN_RE.sub('', text)
    return text


# Match ANY `<|channel>BODY<channel|>` block, body up to ~200 chars.
# More permissive than _LEADING_CHANNEL_RE because we're cleaning
# arbitrary historical contamination — could be anywhere in the message,
# not just at the start.
_ANY_CHANNEL_BLOCK_RE = re.compile(
    r'<\|channel>[^<]{0,200}<channel\|>',
    re.DOTALL,
)


def _clean_history_message(msg: dict) -> dict:
    """Normalize a historical message before rendering through the chat
    template. Two fixes applied:

    1. CHANNEL-MARKER CONTAMINATION (assistant content):
       Strip `<|channel>...<channel|>` blocks from assistant content
       that a previous turn's response leaked into client-saved history.

    2. TOOL-CALL ARGUMENTS DOUBLE-BRACKETING (assistant tool_calls):
       OpenAI's standard tool_calls shape has `arguments` as a JSON-
       encoded STRING (e.g., `'{"query":"..."}'`). The chat template
       (chat_template.jinja:244-255) emits its OWN `{` and `}` brackets
       around this — producing `{{"query":"..."}}` when concatenated
       with the string's existing brackets. The model is trained on
       this doubled-bracket format and reproduces it in NEW tool_call
       output. The bridge's server-side tool-call extractor then fails
       to parse the doubled braces as JSON.

       Fix: parse the JSON string into a dict so the template takes
       the `is mapping` branch (lines 245-251) which formats correctly
       without doubled brackets. As a side effect, the model sees
       single-bracket format in input AND emits single-bracket format
       in output — consistency restored.

    Tool messages and user messages pass through (channel markers in
    user content are likely intentional examples, not bleed).
    """
    role = msg.get("role")
    if role == "assistant":
        out = dict(msg)
        # 1. Channel-marker strip on string content.
        content = msg.get("content")
        if isinstance(content, str):
            cleaned = _ANY_CHANNEL_BLOCK_RE.sub('', content)
            cleaned = _TRAILING_TURN_RE.sub('', cleaned)
            if cleaned != content:
                out["content"] = cleaned.strip()
        # 2. tool_calls arguments string → dict.
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            new_tcs = []
            for tc in tcs:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            parsed = json.loads(args)
                            new_fn = dict(fn)
                            new_fn["arguments"] = parsed
                            new_tc = dict(tc)
                            new_tc["function"] = new_fn
                            new_tcs.append(new_tc)
                            continue
                        except (TypeError, json.JSONDecodeError):
                            pass
                new_tcs.append(tc)
            out["tool_calls"] = new_tcs
        return out
    return msg


_TOOL_CALL_BLOCK_RE = re.compile(
    r'<\|tool_call>(.*?)<tool_call\|>',
    re.DOTALL,
)
# <|"|>STRING<|"|> — training format for atomic-quoted strings
_ATOMIC_QUOTE_RE = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)
# <|D...D|> — model's improvised raw-string form. D is any single char
# that isn't itself `>` or `<` (the surrounding angle brackets) or `|`
# (which would re-enter the marker syntax). Backtick is what we've
# observed; the regex accepts any single non-restricted char.
_RAW_QUOTE_RE = re.compile(r'<\|([^<>|])(.*?)\1\|>', re.DOTALL)
# Unquoted bareword keys in pseudo-JSON: `{key:` or `,key:`.
_BAREWORD_KEY_RE = re.compile(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:')


def _parse_tool_call_args(arg_block: str) -> dict | None:
    """Convert the model's argument-DSL inside `{...}` to a Python dict.

    Strategy: rewrite the DSL to standard JSON, then json.loads. Returns
    None if the result isn't parseable as JSON — caller decides whether
    to fall back (e.g., return raw bytes as the argument value).
    """
    # 1. <|"|>X<|"|>  →  "X" (with proper JSON-escape of X)
    rewritten = _ATOMIC_QUOTE_RE.sub(
        lambda m: json.dumps(m.group(1)), arg_block)
    # 2. <|D...D|>  →  "..." (raw-string form, content can have anything)
    rewritten = _RAW_QUOTE_RE.sub(
        lambda m: json.dumps(m.group(2)), rewritten)
    # 3. Bareword keys → quoted keys
    rewritten = _BAREWORD_KEY_RE.sub(r'\1"\2":', rewritten)
    # 4. Try to parse as JSON
    try:
        return json.loads(rewritten)
    except json.JSONDecodeError:
        return None


def _extract_tool_calls(content: str,
                         had_tools: bool) -> tuple[str, list[dict] | None]:
    """Extract tool calls from a model response into OpenAI shape.

    Returns (cleaned_content, tool_calls_or_None). When a parse fails
    on a particular block, the block stays in cleaned_content as-is
    so the client at least sees the bytes (debugging affordance).
    """
    if not had_tools or '<|tool_call>' not in content:
        return content, None

    tool_calls: list[dict] = []
    cleaned_parts: list[str] = []
    last_end = 0
    for m in _TOOL_CALL_BLOCK_RE.finditer(content):
        cleaned_parts.append(content[last_end:m.start()])
        body = m.group(1).strip()
        # Body shape: `call:NAME{ARGS}` (NAME is plain identifier;
        # ARGS may contain nested braces, so split at the FIRST `{`).
        prefix, _, rest = body.partition('{')
        prefix = prefix.strip()
        if not (prefix.startswith('call:') and rest):
            # Malformed: leave the block unmodified.
            cleaned_parts.append(m.group(0))
            last_end = m.end()
            continue
        name = prefix[len('call:'):].strip()
        # rest is "...args...}" — strip the trailing `}` to get the args body.
        if not rest.endswith('}'):
            cleaned_parts.append(m.group(0))
            last_end = m.end()
            continue
        args_body = '{' + rest  # restore opening brace; close already there
        parsed = _parse_tool_call_args(args_body)
        if parsed is None:
            # Couldn't convert to JSON — keep raw block visible to
            # the client and skip the tool_calls[] entry.
            cleaned_parts.append(m.group(0))
            last_end = m.end()
            continue
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(parsed, ensure_ascii=False),
            },
        })
        last_end = m.end()
    cleaned_parts.append(content[last_end:])
    cleaned = ''.join(cleaned_parts).strip()
    return cleaned, (tool_calls if tool_calls else None)


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

    # Clean prior assistant `content` before rendering: strip any
    # `<|channel>...<channel|>` blocks that a previous turn's response
    # may have leaked into the saved history. This was a real failure
    # mode — the model's first-tokens-per-turn echo of the no-thinking
    # epilogue would land in client conversation state, then on the
    # NEXT turn that literal text gets atomic-id-emitted on the input
    # path (per chat_template.py:SPECIAL_TOKENS) and the model sees a
    # mid-message empty thought-channel block in its own past output.
    # Stripping on input is belt to the suspenders of stripping on
    # output (_strip_response_scaffolding), since old contaminated
    # history persists even after the output-side fix.
    messages = [_clean_history_message(m) for m in messages]
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
    # PYTHONUNBUFFERED=1 (set by the lifecycle launcher) means every
    # write hits stdout immediately; the explicit `flush=True` is
    # gratuitous lock acquisition on each per-request print.
    print(f"[bridge] chat_completions: tools={len(tools) if tools else 0}, "
          f"tool_choice={body.get('tool_choice')!r}, "
          f"messages={len(messages)}, stream={stream}, "
          f"stop_seqs={len(sampling.stop_sequences)}")
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
                  f"done_reason={done_reason}")
            finish = "stop" if done_reason == 1 else (
                "length" if done_reason == 2 else "stop")
            # The engine emits done_reason=1 even when generation hit
            # the requested max_tokens budget (rather than emitting EOS
            # naturally), so done_reason alone undercounts "length"
            # truncations. Detect the budget-hit case explicitly: when
            # completion_tokens equals the requested max_tokens AND the
            # engine claimed natural stop, the truncation is actually
            # length-driven. Without this, automated retry / continuation
            # logic that branches on finish_reason gets the wrong signal.
            if (finish == "stop"
                    and isinstance(usage.get("completion_tokens"), int)
                    and usage["completion_tokens"] >= max_tokens):
                finish = "length"
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
            cleaned_text = _strip_response_scaffolding(cleaned_text)
            message: dict[str, Any] = {"role": "assistant",
                                        "content": cleaned_text}
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
        # 2026-05-07: when tools are present in the request we BUFFER
        # the response content rather than streaming it tokenwise. This
        # is so we can parse `<|tool_call>...<tool_call|>` blocks out of
        # the complete response and emit OpenAI-shape `tool_calls[]`
        # deltas at the end (matching how every other OAI-compat server
        # streams tool calls). Tool-call responses are typically short
        # (engine stops at <tool_call|>), so the latency cost of
        # buffering is bounded and the shape correctness is worth it.
        # Non-tool requests stream tokenwise as before.
        had_tools = tools is not None
        all_tokens: list[int] = []
        last_update: g.StreamUpdate | None = None
        clean_close = False
        try:
            while True:
                u: g.StreamUpdate = await response_q.get()
                last_update = u
                if u.new_tokens:
                    all_tokens.extend(u.new_tokens)
                    if not had_tools:
                        # Tokenwise content streaming (the common path).
                        # Strip per-turn scaffolding (trailing <turn|>,
                        # leading channel echoes) per-delta. Found by
                        # tools/st-debug api_probe: pre-fix, the final
                        # delta contained "<turn|>" because token id 106
                        # detokenizes to the literal string. Most deltas
                        # are unaffected; the strip is a no-op when no
                        # marker is present.
                        delta_text = _strip_response_scaffolding(
                            g.detokenize(u.new_tokens))
                        if delta_text:
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
                    # Same length-truncation override as the non-streaming
                    # path: engine reports done_reason=1 even when the
                    # max_tokens budget was the actual cause, so detect
                    # that here from completion_tokens_emitted.
                    if (finish == "stop"
                            and u.completion_tokens_emitted >= max_tokens):
                        finish = "length"
                    # When tools were present, drain accumulated content
                    # through the tool-call extractor and emit the right
                    # shape: either a final content delta (no tool calls
                    # found) or a tool_calls delta + finish_reason=tool_calls.
                    if had_tools:
                        full_text = g.detokenize(all_tokens)
                        cleaned_text, extracted = _extract_tool_calls(
                            full_text, had_tools=True)
                        cleaned_text = _strip_response_scaffolding(cleaned_text)
                        if extracted:
                            finish = "tool_calls"
                            print(f"[bridge] (SSE) extracted "
                                  f"{len(extracted)} tool_call(s); names="
                                  f"{[tc['function']['name'] for tc in extracted]}")
                            # Surface any non-tool prose first (rare, but
                            # the model occasionally adds prose before/
                            # after the tool call).
                            if cleaned_text:
                                yield ("data: " + json.dumps({
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": MODEL_NAME,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": cleaned_text},
                                        "finish_reason": None,
                                    }],
                                }) + "\n\n")
                            # Tool-calls delta. OpenAI's streaming shape
                            # uses `index` per tool call within the choice.
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
                        else:
                            # No tool calls extracted — emit the full
                            # buffered content as one delta so the
                            # client at least gets the bytes.
                            yield ("data: " + json.dumps({
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_NAME,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": cleaned_text},
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
