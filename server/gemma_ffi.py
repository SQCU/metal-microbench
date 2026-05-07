"""ctypes wrapper around libgemma_metal.dylib.

The interface has exactly five functional / two utility / two admin
entries. There is no per-session shape — a single submission is a
batch of size one and goes through the same call as 16 simultaneous
submissions. Sampler-side features (grammar, top_p, top_k, logit_bias)
are fields on `SamplingParams`. Control-vector applications (when we
add them) will be a field on `StreamSpec`. Neither becomes a new FFI
function.

Single-threaded contract: only one Python coroutine/thread calls into
the dylib at a time. The bridge enforces this by funnelling all FFI
calls through one coordinator coroutine. There is no lock here because
none is needed; the contract holds upstream.

ABI source of truth: notes/specs/batch_ffi_abi.md.
"""
from __future__ import annotations

import ctypes as C
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------------------
# Library load.
# ----------------------------------------------------------------------
def _find_dylib() -> Path:
    env = os.environ.get("GEMMA_DYLIB")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    for c in (here.parent / "libgemma_metal.dylib",
              Path.cwd() / "libgemma_metal.dylib"):
        if c.exists():
            return c.resolve()
    raise FileNotFoundError("libgemma_metal.dylib not found")


_lib = C.CDLL(str(_find_dylib()))


# ----------------------------------------------------------------------
# C signatures. The Swift @_cdecl exports are gemma_submit + gemma_poll
# (the runtime functional pair) plus gemma_init / gemma_vision_init /
# gemma_status / gemma_shutdown / gemma_bos_id / gemma_eos_id /
# gemma_tokenize / gemma_detokenize.
# ----------------------------------------------------------------------
_lib.gemma_init.argtypes = [C.c_char_p]
_lib.gemma_init.restype = C.c_int32

_lib.gemma_vision_init.argtypes = [C.c_char_p]
_lib.gemma_vision_init.restype = C.c_int32

_lib.gemma_vision_is_ready.argtypes = []
_lib.gemma_vision_is_ready.restype = C.c_int32

_lib.gemma_bos_id.argtypes = []
_lib.gemma_bos_id.restype = C.c_uint32

_lib.gemma_eos_id.argtypes = []
_lib.gemma_eos_id.restype = C.c_uint32

_lib.gemma_tokenize.argtypes = [
    C.c_char_p, C.c_int32, C.c_int32,
    C.POINTER(C.c_uint32), C.c_int32,
]
_lib.gemma_tokenize.restype = C.c_int32

_lib.gemma_detokenize.argtypes = [
    C.POINTER(C.c_uint32), C.c_int32,
    C.c_char_p, C.c_int32,
]
_lib.gemma_detokenize.restype = C.c_int32

_lib.gemma_submit.argtypes = [C.POINTER(C.c_uint8), C.c_int32]
_lib.gemma_submit.restype = C.c_int32

_lib.gemma_poll.argtypes = [C.c_int32, C.POINTER(C.c_uint8), C.c_int32]
_lib.gemma_poll.restype = C.c_int32

_lib.gemma_status.argtypes = [C.POINTER(C.c_uint8), C.c_int32]
_lib.gemma_status.restype = C.c_int32

_lib.gemma_shutdown.argtypes = []
_lib.gemma_shutdown.restype = C.c_int32

_lib.gemma_register_resource.argtypes = [
    C.c_char_p, C.c_char_p, C.POINTER(C.c_uint8), C.c_int32,
]
_lib.gemma_register_resource.restype = C.c_int32

_lib.gemma_max_q_len.argtypes = []
_lib.gemma_max_q_len.restype = C.c_int32


# ----------------------------------------------------------------------
# Stream / sampling / segment dataclasses (Python-side shape).
# ----------------------------------------------------------------------
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 64
    seed: int = 0
    eos_token_id: int = -1            # -1 = use model default
    stop_tokens: list[int] = field(default_factory=list)
    top_logprobs: int = 0
    # Sampler-side: per-token additive bias. Empty dict = no bias.
    logit_bias: dict[int, float] = field(default_factory=dict)
    # Sampler-side: drop tokens whose softmax prob < min_p × max. 0 = off.
    min_p: float = 0.0
    # Sampler-side: structured-cot grammar role labels. Empty = no grammar.
    # Common values: ["GOAL", "APPROACH", "EDGE"] for the default cot.
    cot_labels: list[str] = field(default_factory=list)
    # Multi-token stop sequences. After each emitted token, the engine
    # checks whether the recent emitted tail equals any sequence here;
    # if so, the stream finishes with done_reason=1 (eos-equivalent).
    # Used for tool-call early-stop: tokenize "<tool_call|>" once and
    # pass it here so the engine self-terminates on tool-call close
    # without the bridge having to detect-and-cancel from outside.
    stop_sequences: list[list[int]] = field(default_factory=list)


@dataclass
class CVApplication:
    """One control-vector application on a stream.

    `cvec_id` references a CV uploaded via register_resource(kind='cvec', ...).
    The bridge does NOT register CVs per-request; uploads happen once at
    startup and CVs are referenced by id forever after."""
    cvec_id: str
    layer: int = 0
    polarity: float = 1.0
    peak_magnitude: float = 1.0
    attack: float = 0.0
    decay: float = 0.0
    sustain_level: float = 1.0
    release: float = 0.0
    shape: int = 0           # 0=linear, 1=expIn, 2=expOut, 3=cubic
    units: int = 0           # 0=tokens, 1=turns
    mode: int = 0            # 0=additive, 1=project, 2=transport
    target: float = float("nan")  # NaN = envelope-as-target
    transport_scale: float = 0.0
    transport_offset: float = 0.0


@dataclass
class Segment:
    """A single segment of a stream's input. Tokens or image_bytes."""
    kind: int                          # 0=tokens, 1=image_bytes
    tokens: list[int] = field(default_factory=list)
    image_bytes: bytes = b""


@dataclass
class StreamSpec:
    stream_id: int
    action: int                        # 0=start, 1=continue, 2=cancel, 3=touch
    flags: int = 0                     # bit 0 = capture_logits
    segments: list[Segment] = field(default_factory=list)
    sampling: SamplingParams = field(default_factory=SamplingParams)
    # Convenience: pass tokens=[...] in lieu of segments=[Segment(...)].
    tokens: list[int] = field(default_factory=list)
    # Forward-pass-side: control vectors. Empty list = no-op.
    control_vectors: list[CVApplication] = field(default_factory=list)


@dataclass
class TokenLogprob:
    token: int
    sampled_logprob: float
    top_logprobs: list[tuple[int, float]]


@dataclass
class StreamUpdate:
    stream_id: int
    state: int                         # 0=priming, 1=generating, 2=done, 3=error
    done_reason: int                   # 0=n/a, 1=eos, 2=max_tokens, 3=cancelled, 4=error
    new_tokens: list[int]
    err_msg: str
    prompt_tokens_seen: int
    completion_tokens_emitted: int
    cache_hits: int
    cache_misses: int
    vision_cache_hits: int
    logprobs: list[TokenLogprob] = field(default_factory=list)


@dataclass
class ServerStats:
    total_pages: int
    free_pages: int
    cached_pages: int
    active_streams: int
    generating_streams: int
    priming_streams: int
    total_steps: int
    total_tokens_emitted: int
    vision_cache_entries: int
    vision_cache_hits: int


# ----------------------------------------------------------------------
# Wire encoding / decoding (hand-rolled little-endian binary, per ABI).
# ----------------------------------------------------------------------
_MAGIC_REQ = 0x424D4547   # 'GEMB'
_MAGIC_RESP = 0x52454D47  # 'GEMR'


def _encode_request(streams: list[StreamSpec]) -> bytes:
    # Normalize convenience .tokens into a single tokens-segment.
    norm: list[StreamSpec] = []
    for s in streams:
        if s.segments:
            norm.append(s)
        elif s.tokens and s.action != 2:
            norm.append(StreamSpec(
                stream_id=s.stream_id, action=s.action, flags=s.flags,
                segments=[Segment(kind=0, tokens=list(s.tokens))],
                sampling=s.sampling))
        else:
            norm.append(s)
    streams = norm

    streams_base = 16
    streams_bytes = 104 * len(streams)
    heap_base = streams_base + streams_bytes
    heap = bytearray()
    seg_offsets: list[int] = []
    stop_offsets: list[int] = []
    seg_counts: list[int] = []
    lb_offsets: list[int] = []
    lb_counts: list[int] = []
    cot_offsets: list[int] = []
    cot_counts: list[int] = []
    cv_offsets: list[int] = []
    cv_counts: list[int] = []
    ssq_offsets: list[int] = []
    ssq_counts: list[int] = []

    for s in streams:
        if s.segments and s.action != 2:
            seg_count = len(s.segments)
            seg_arr_off = heap_base + len(heap)
            seg_struct_off = len(heap)
            heap += b"\x00" * (16 * seg_count)
            for idx, seg in enumerate(s.segments):
                if seg.kind == 0:
                    data_off = heap_base + len(heap)
                    for t in seg.tokens:
                        heap += struct.pack("<I", t)
                    count = len(seg.tokens)
                elif seg.kind == 1:
                    data_off = heap_base + len(heap)
                    heap += seg.image_bytes
                    count = len(seg.image_bytes)
                else:
                    raise ValueError(f"unknown segment kind {seg.kind}")
                heap[seg_struct_off + idx*16 : seg_struct_off + (idx+1)*16] = struct.pack(
                    "<BBBBII4x", seg.kind, 0, 0, 0, count, data_off)
            seg_offsets.append(seg_arr_off)
            seg_counts.append(seg_count)
        else:
            seg_offsets.append(0)
            seg_counts.append(0)

        if s.sampling.stop_tokens:
            stop_off = heap_base + len(heap)
            for st in s.sampling.stop_tokens:
                heap += struct.pack("<I", st)
            stop_offsets.append(stop_off)
        else:
            stop_offsets.append(0)

        if s.sampling.logit_bias:
            lb_off = heap_base + len(heap)
            for tok, bias in s.sampling.logit_bias.items():
                heap += struct.pack("<If", int(tok), float(bias))
            lb_offsets.append(lb_off)
            lb_counts.append(len(s.sampling.logit_bias))
        else:
            lb_offsets.append(0)
            lb_counts.append(0)

        if s.sampling.cot_labels:
            cot_off = heap_base + len(heap)
            for label in s.sampling.cot_labels:
                lb = label.encode("utf-8")
                heap += struct.pack("<I", len(lb))
                heap += lb
            cot_offsets.append(cot_off)
            cot_counts.append(len(s.sampling.cot_labels))
        else:
            cot_offsets.append(0)
            cot_counts.append(0)

        # Multi-token stop_sequences: each sequence laid out as
        # [u32 length][u32 tok0][u32 tok1]...[u32 tokN-1] back-to-back.
        # The per-stream count is the number of sequences; the offset
        # points to the first length word.
        if s.sampling.stop_sequences:
            ssq_arr_off = heap_base + len(heap)
            for seq in s.sampling.stop_sequences:
                heap += struct.pack("<I", len(seq))
                for tok in seq:
                    heap += struct.pack("<I", int(tok))
            ssq_offsets.append(ssq_arr_off)
            ssq_counts.append(len(s.sampling.stop_sequences))
        else:
            ssq_offsets.append(0)
            ssq_counts.append(0)

        if s.control_vectors:
            # Reserve fixed-size CV blocks first; backfill cvec_id_offset
            # after writing the id strings to the heap.
            cv_arr_off = heap_base + len(heap)
            cv_struct_off = len(heap)
            heap += b"\x00" * (64 * len(s.control_vectors))
            for k, cv in enumerate(s.control_vectors):
                id_bytes = cv.cvec_id.encode("utf-8")
                id_off = heap_base + len(heap)
                heap += id_bytes
                cv_struct = struct.pack(
                    "<IIifffffffBBBxfff12x",
                    id_off, len(id_bytes),
                    int(cv.layer),
                    float(cv.polarity), float(cv.peak_magnitude),
                    float(cv.attack), float(cv.decay),
                    float(cv.sustain_level), float(cv.release),
                    int(cv.shape) & 0xff,
                    int(cv.units) & 0xff,
                    int(cv.mode)  & 0xff,
                    float(cv.target),
                    float(cv.transport_scale),
                    float(cv.transport_offset),
                )
                assert len(cv_struct) == 64, f"CVApplication must be 64 bytes, got {len(cv_struct)}"
                heap[cv_struct_off + k * 64:cv_struct_off + (k + 1) * 64] = cv_struct
            cv_offsets.append(cv_arr_off)
            cv_counts.append(len(s.control_vectors))
        else:
            cv_offsets.append(0)
            cv_counts.append(0)

    out = bytearray()
    # Wire format v4 — adds stop_sequences in what were the 8 reserved
    # bytes following cot_offset (formerly "8x"). Version bumped from 3
    # so older Swift parsers that only read v3 fail loudly instead of
    # silently mis-reading the new fields as zeros.
    out += struct.pack("<IIII", _MAGIC_REQ, 4, len(streams), heap_base)
    for i, s in enumerate(streams):
        out += struct.pack(
            "<QBBHIII"           # 24 bytes (header)
            "ffIfIQiII"          # 40 bytes (sampling: T,topP,topK,repPen,maxN,seed,eos,stopCount,stopOff)
            "III"                # 12 bytes (sampling: topLogprobs, lbCount, lbOffset)
            "fIIII"              # 20 bytes (sampling: minP, cotCount, cotOff, ssqCount, ssqOff)
            "II",                # 8 bytes (StreamSpec extension: cv_count, cvs_offset)
            s.stream_id, s.action, s.flags, 0,
            seg_counts[i], seg_offsets[i], 0,
            s.sampling.temperature, s.sampling.top_p,
            s.sampling.top_k, s.sampling.repetition_penalty,
            s.sampling.max_new_tokens, s.sampling.seed,
            s.sampling.eos_token_id, len(s.sampling.stop_tokens),
            stop_offsets[i],
            s.sampling.top_logprobs, lb_counts[i], lb_offsets[i],
            s.sampling.min_p, cot_counts[i], cot_offsets[i],
            ssq_counts[i], ssq_offsets[i],
            cv_counts[i], cv_offsets[i],
        )
    out += heap
    return bytes(out)


def _decode_response(buf: bytes) -> list[StreamUpdate]:
    if len(buf) < 16:
        return []
    magic, version, count, heap_off = struct.unpack_from("<IIII", buf, 0)
    if magic != _MAGIC_RESP:
        raise ValueError(f"bad response magic 0x{magic:08x}")
    if version != 1:
        raise ValueError(f"bad response version {version}")
    out: list[StreamUpdate] = []
    for i in range(count):
        base = 16 + i * 64
        (sid, state, done_reason, _r,
         nt_count, nt_off, err_count, err_off,
         pts, cte, ch, cm, vch,
         lp_bytes, lp_off,
         _r1, _r2) = struct.unpack_from("<QBBHIIIIIIIIIIIII", buf, base)
        toks = list(struct.unpack_from(f"<{nt_count}I", buf, nt_off)) if nt_count else []
        msg = bytes(buf[err_off:err_off + err_count]).decode("utf-8", errors="replace") if err_count else ""
        lps: list[TokenLogprob] = []
        cur, end = lp_off, lp_off + lp_bytes
        while cur < end:
            tok, slp, top = struct.unpack_from("<IfI", buf, cur); cur += 12
            top_pairs: list[tuple[int, float]] = []
            for _ in range(top):
                tid, lpv = struct.unpack_from("<If", buf, cur); cur += 8
                top_pairs.append((tid, lpv))
            lps.append(TokenLogprob(token=tok, sampled_logprob=slp, top_logprobs=top_pairs))
        out.append(StreamUpdate(
            stream_id=sid, state=state, done_reason=done_reason,
            new_tokens=toks, err_msg=msg,
            prompt_tokens_seen=pts, completion_tokens_emitted=cte,
            cache_hits=ch, cache_misses=cm, vision_cache_hits=vch,
            logprobs=lps,
        ))
    return out


# ----------------------------------------------------------------------
# High-level entrypoints. No locks — single-threaded contract enforced
# upstream by the bridge's coordinator coroutine.
# ----------------------------------------------------------------------
def init(gguf_path: str, vision_safetensors_path: Optional[str] = None) -> int:
    rc = _lib.gemma_init(gguf_path.encode("utf-8"))
    if rc != 0:
        return rc
    if vision_safetensors_path:
        return _lib.gemma_vision_init(vision_safetensors_path.encode("utf-8"))
    return 0


def vision_init(safetensors_path: str) -> int:
    return _lib.gemma_vision_init(safetensors_path.encode("utf-8"))


def vision_is_ready() -> bool:
    return _lib.gemma_vision_is_ready() != 0


def shutdown() -> int:
    return _lib.gemma_shutdown()


def register_resource(kind: str, id: str, data: bytes) -> int:
    """Upload a named resource (control vector today). Replaces the legacy
    per-feature register entries. kind='cvec' expects HIDDEN×2 fp16 bytes."""
    arr = (C.c_uint8 * len(data)).from_buffer_copy(data)
    return _lib.gemma_register_resource(
        kind.encode("utf-8"), id.encode("utf-8"), arr, len(data))


def bos_id() -> int:
    return _lib.gemma_bos_id()


def eos_id() -> int:
    return _lib.gemma_eos_id()


def tokenize(text: str, add_bos: bool = False) -> list[int]:
    """Tokenize a string. Returns a list of token IDs."""
    s = text.encode("utf-8")
    cap = max(64, len(s) * 2)
    out = (C.c_uint32 * cap)()
    n = _lib.gemma_tokenize(s, len(s), 1 if add_bos else 0, out, cap)
    if n < 0:
        raise RuntimeError(f"gemma_tokenize returned {n}")
    if n > cap:
        out = (C.c_uint32 * n)()
        n = _lib.gemma_tokenize(s, len(s), 1 if add_bos else 0, out, n)
    return list(out[:n])


def detokenize(tokens: list[int]) -> str:
    n = len(tokens)
    if n == 0:
        return ""
    arr = (C.c_uint32 * n)(*tokens)
    cap = max(256, n * 4)
    buf = C.create_string_buffer(cap)
    written = _lib.gemma_detokenize(arr, n, buf, cap)
    if written < 0:
        raise RuntimeError(f"gemma_detokenize returned {written}")
    if written > cap:
        cap = written + 1
        buf = C.create_string_buffer(cap)
        written = _lib.gemma_detokenize(arr, n, buf, cap)
    return buf.value.decode("utf-8", errors="replace")


def submit(streams: list[StreamSpec]) -> int:
    """Submit a batch of stream actions. Non-blocking. Returns 0 on success.

    A "batch" of size 1 goes through this exact call. There is no
    separate single-stream entrypoint."""
    buf = _encode_request(streams)
    arr = (C.c_uint8 * len(buf)).from_buffer_copy(buf)
    return _lib.gemma_submit(arr, len(buf))


# Module-level reusable poll buffer. Starts at 64 KB which covers
# single-stream single-token updates (the common AR streaming case);
# doubles on -ENOSPC. Avoids the previous 4 MB allocation per poll
# call (~184 calls per turn × 4 MB zero-init = 736 MB of churn).
_POLL_BUF_CAPACITY: int = 64 * 1024
_POLL_BUF = (C.c_uint8 * _POLL_BUF_CAPACITY)()
_ENOSPC = -28


def poll(timeout_ms: int = 100) -> list[StreamUpdate]:
    """Drive the engine forward and report progress. Returns a list of
    StreamUpdate (possibly empty on timeout)."""
    global _POLL_BUF, _POLL_BUF_CAPACITY
    while True:
        n = _lib.gemma_poll(timeout_ms, _POLL_BUF, _POLL_BUF_CAPACITY)
        if n == _ENOSPC:
            _POLL_BUF_CAPACITY *= 2
            _POLL_BUF = (C.c_uint8 * _POLL_BUF_CAPACITY)()
            continue
        if n < 0:
            raise RuntimeError(f"gemma_poll returned {n}")
        if n == 0:
            return []
        return _decode_response(bytes(_POLL_BUF[:n]))


def max_q_len() -> int:
    """Engine-side cap on a single teacher-forced eval call (in tokens).
    Longer corpora are stride-windowed by the caller."""
    return _lib.gemma_max_q_len()


def status() -> ServerStats:
    out = (C.c_uint8 * 64)()
    n = _lib.gemma_status(out, 64)
    if n != 64:
        raise RuntimeError(f"gemma_status returned {n}")
    raw = bytes(out[:n])
    (tp, fp, cp, act, gen, prim, ts, tok, ve, _pad, vh) = struct.unpack_from(
        "<IIIIIIQQIIQ", raw, 0)
    return ServerStats(
        total_pages=tp, free_pages=fp, cached_pages=cp,
        active_streams=act, generating_streams=gen, priming_streams=prim,
        total_steps=ts, total_tokens_emitted=tok,
        vision_cache_entries=ve, vision_cache_hits=vh,
    )
