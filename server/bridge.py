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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import gemma_ffi as g

from chat_template import (
    TextChunk, ImageChunk, SoftsChunk, render_chat,
    tokenize_with_specials,
)
from tool_call_parser import extract_tool_calls


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
MODEL_NAME = os.environ.get("GEMMA_MODEL_NAME", "gemma-4-a4b-q4km")


# ----------------------------------------------------------------------
# Coordinator state.
#
# A single asyncio task owns ALL FFI calls. HTTP handlers enqueue
# StreamSpec into _submit_q and read updates from per-stream queues.
# Because only the coordinator calls g.submit / g.poll, no lock is
# needed — the coroutine awaits each FFI call before issuing the next.
# ----------------------------------------------------------------------
_submit_q: "asyncio.Queue[g.StreamSpec]" = None  # type: ignore[assignment]
_response_qs: dict[int, "asyncio.Queue[g.StreamUpdate]"] = {}
_next_stream_id_lock: "asyncio.Lock | None" = None
_next_stream_id = 1
_coord_task: "asyncio.Task | None" = None
_TOOL_CALL_CLOSE_TOKENS: list[int] = []  # set at startup


async def _next_stream_id_alloc() -> int:
    global _next_stream_id
    async with _next_stream_id_lock:  # type: ignore[arg-type]
        sid = _next_stream_id
        _next_stream_id += 1
        return sid


async def _coordinator() -> None:
    """Single coroutine that owns all FFI calls.

    Drain the submission queue (non-blocking burst), submit anything
    pending as ONE batch (so backend in-batch shared-prefix detection
    can fire), then poll for updates. When idle, sleep briefly.
    """
    while True:
        try:
            new_specs: list[g.StreamSpec] = []
            try:
                while True:
                    new_specs.append(_submit_q.get_nowait())
            except asyncio.QueueEmpty:
                pass

            if new_specs:
                rc = await asyncio.to_thread(g.submit, new_specs)
                if rc != 0:
                    print(f"[coord] submit returned {rc}", flush=True)

            updates = await asyncio.to_thread(g.poll, 50)
            for u in updates:
                rq = _response_qs.get(u.stream_id)
                if rq is not None:
                    await rq.put(u)

            if not new_specs and not updates:
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[coord] error: {e}", flush=True)
            await asyncio.sleep(0.1)


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
        # to convert. For now, leave empty if any are non-empty strings;
        # bridge does post-detokenize string matching client-side.
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
                       tools: list | None = None) -> g.StreamSpec:
    """OpenAI messages → StreamSpec via the model's own chat template.

    `tools` (OpenAI function-calling tool schemas) is forwarded to the
    template, which emits Gemma-4's native <|tool>declaration:...<tool|>
    blocks in the system turn so the model knows what's available.
    """
    chunks = render_chat(messages, add_generation_prompt=True, tools=tools)
    segments: list[g.Segment] = []
    pending: list[int] = []
    did_bos = False

    def flush() -> None:
        nonlocal did_bos
        if pending:
            segments.append(g.Segment(kind=0, tokens=list(pending)))
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
            segments.append(g.Segment(kind=1, image_bytes=ch.data))
        elif isinstance(ch, SoftsChunk):
            raise HTTPException(
                400, "client-replayed soft tokens not yet supported on /v1/chat/completions")
    flush()

    return g.StreamSpec(
        stream_id=stream_id,
        action=0,
        flags=0x01 if capture_logits else 0,
        segments=segments,
        sampling=sampling,
    )


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
    global _submit_q, _next_stream_id_lock, _coord_task

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

    _submit_q = asyncio.Queue()
    _next_stream_id_lock = asyncio.Lock()
    _coord_task = asyncio.create_task(_coordinator())


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _coord_task
    if _coord_task is not None:
        _coord_task.cancel()
        try:
            await _coord_task
        except asyncio.CancelledError:
            pass
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
    })


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
    # When tools are present, ask the engine to self-terminate on the
    # tool-call close marker so we don't burn 4096 tokens of
    # <|tool_response> waiting for an injected response. Engine-side
    # stop_sequences match against the recently-emitted tail; once the
    # 5-token <tool_call|> sequence appears, done_reason=1 fires.
    if tools and _TOOL_CALL_CLOSE_TOKENS:
        sampling.stop_sequences = [list(_TOOL_CALL_CLOSE_TOKENS)]
    print(f"[bridge] chat_completions: tools={len(tools) if tools else 0}, "
          f"tool_choice={body.get('tool_choice')!r}, "
          f"messages={len(messages)}, stream={stream}, "
          f"stop_seqs={len(sampling.stop_sequences)}", flush=True)
    stream_id = await _next_stream_id_alloc()
    spec = _build_stream_spec(stream_id, messages, sampling, capture_logits, tools=tools)
    response_q: asyncio.Queue = asyncio.Queue()
    _response_qs[stream_id] = response_q

    await _submit_q.put(spec)

    if not stream:
        # Aggregate path: handler awaits the entire response, so handler-
        # level finally is correct.
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
            # Detect Gemma's native tool-call output. If found, lift
            # them into the OAI-shape `tool_calls` field on the message
            # and strip from `content`. finish_reason becomes
            # "tool_calls" per the OpenAI spec so the client knows to
            # invoke the tool rather than treat the response as final.
            tool_calls, residual = extract_tool_calls(text)
            if tool_calls:
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": residual or None,
                    "tool_calls": tool_calls,
                }
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": text}
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
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [choice],
                "usage": usage,
            })
        finally:
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
        # Buffer all tokens; once we see Gemma's tool-call open marker
        # we suppress further content chunks until done, then emit the
        # parsed tool_calls in one final delta. Until the marker shows
        # up we stream content normally so plain replies have low TTFT.
        all_tokens: list[int] = []
        suppress_content = False
        # Rolling text window for tool-call open-marker detection. The
        # marker can span delta-token boundaries, so we keep a small tail
        # window and search there. O(1) per token vs the previous
        # O(n) detokenize-history-per-token cost (which dominated SSE
        # latency on long generations).
        _open_window = ""
        _OPEN_MARKER = '<|tool_call>'
        _OPEN_WINDOW_LEN = len(_OPEN_MARKER) * 2 + 32
        try:
            while True:
                u: g.StreamUpdate = await response_q.get()
                if u.new_tokens:
                    all_tokens.extend(u.new_tokens)
                    delta_text = g.detokenize(u.new_tokens)
                    if not suppress_content:
                        # Slide the window with the new delta and search
                        # for the open marker. O(window_len) per token.
                        _open_window = (_open_window + delta_text)[-_OPEN_WINDOW_LEN:]
                        if _OPEN_MARKER in _open_window:
                            suppress_content = True
                        else:
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
                    # No bridge-side cancel needed — when tools are
                    # present we passed the <tool_call|> token sequence
                    # in SamplingParams.stop_sequences, so the engine
                    # self-terminates with state==2 / done_reason=1 as
                    # soon as it emits the close marker.
                if u.state == 2:
                    finish = "stop" if u.done_reason == 1 else (
                        "length" if u.done_reason == 2 else "stop")
                    full_text = g.detokenize(all_tokens)
                    tool_calls, residual = extract_tool_calls(full_text)
                    if tool_calls:
                        finish = "tool_calls"
                        # Emit one chunk with the full tool_calls array.
                        # Each entry carries `index`, the streaming-shape
                        # ST/OAI-clients accumulate by.
                        delta_tc = []
                        for i, tc in enumerate(tool_calls):
                            delta_tc.append({
                                "index": i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            })
                        delta = {"tool_calls": delta_tc}
                        if residual:
                            delta["content"] = residual
                        yield ("data: " + json.dumps({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": delta,
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
                    break
        finally:
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
