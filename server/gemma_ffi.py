"""ctypes wrapper around libgemma_metal.dylib.

The Swift side is single-threaded-from-its-POV — every @_cdecl entry takes
an NSRecursiveLock. This wrapper just binds the C symbols and marshals
Python <-> C types. Concurrency on the Python side is handled by the
bridge (one pump thread + async request handlers).

Session state enum — keep in sync with ffi.swift's gemma_session_state:
  0 = idle, 1 = priming, 2 = generating, 3 = paused, 4 = done
"""
from __future__ import annotations

import ctypes as C
import os
import threading
from pathlib import Path


STATE_IDLE = 0
STATE_PRIMING = 1
STATE_GENERATING = 2
STATE_PAUSED = 3
STATE_DONE = 4


def _find_dylib() -> Path:
    # Search order: $GEMMA_DYLIB, sibling ../libgemma_metal.dylib, cwd.
    env = os.environ.get("GEMMA_DYLIB")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "libgemma_metal.dylib",
        Path.cwd() / "libgemma_metal.dylib",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        "libgemma_metal.dylib not found. Build it with `make libgemma_metal.dylib` "
        "in the repo root, or set GEMMA_DYLIB=<path>."
    )


_lib_path = _find_dylib()
_lib = C.CDLL(str(_lib_path))

# --- Signatures ---

_lib.gemma_init.argtypes = [C.c_char_p]
_lib.gemma_init.restype = C.c_int32

_lib.gemma_is_ready.argtypes = []
_lib.gemma_is_ready.restype = C.c_int32

_lib.gemma_open_session.argtypes = [C.c_int32]
_lib.gemma_open_session.restype = C.c_int32

_lib.gemma_close_session.argtypes = [C.c_int32]
_lib.gemma_close_session.restype = C.c_int32

_lib.gemma_pause_session.argtypes = [C.c_int32]
_lib.gemma_pause_session.restype = C.c_int32

_lib.gemma_submit.argtypes = [C.c_int32, C.POINTER(C.c_uint32), C.c_int32]
_lib.gemma_submit.restype = C.c_int32

_lib.gemma_append.argtypes = [C.c_int32, C.POINTER(C.c_uint32), C.c_int32]
_lib.gemma_append.restype = C.c_int32

_lib.gemma_tick.argtypes = []
_lib.gemma_tick.restype = C.c_int32

_lib.gemma_has_work.argtypes = []
_lib.gemma_has_work.restype = C.c_int32

_lib.gemma_poll.argtypes = [C.c_int32, C.POINTER(C.c_uint32), C.c_int32]
_lib.gemma_poll.restype = C.c_int32

_lib.gemma_session_state.argtypes = [C.c_int32]
_lib.gemma_session_state.restype = C.c_int32

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

_lib.gemma_bos_id.argtypes = []
_lib.gemma_bos_id.restype = C.c_uint32

_lib.gemma_eos_id.argtypes = []
_lib.gemma_eos_id.restype = C.c_uint32

_lib.gemma_active_session_count.argtypes = []
_lib.gemma_active_session_count.restype = C.c_int32

_lib.gemma_vision_init.argtypes = [C.c_char_p]
_lib.gemma_vision_init.restype = C.c_int32

_lib.gemma_vision_is_ready.argtypes = []
_lib.gemma_vision_is_ready.restype = C.c_int32

_lib.gemma_submit_image_path.argtypes = [C.c_int32, C.c_char_p]
_lib.gemma_submit_image_path.restype = C.c_int32

_lib.gemma_vision_residency_state.argtypes = []
_lib.gemma_vision_residency_state.restype = C.c_int32

_lib.gemma_vision_residency_bytes.argtypes = []
_lib.gemma_vision_residency_bytes.restype = C.c_uint64

_lib.gemma_vision_allow_evict.argtypes = []
_lib.gemma_vision_allow_evict.restype = C.c_int32

_lib.gemma_vision_force_drop.argtypes = []
_lib.gemma_vision_force_drop.restype = C.c_int32

_lib.gemma_vision_prewarm_path.argtypes = [C.c_char_p]
_lib.gemma_vision_prewarm_path.restype = C.c_int32

_lib.gemma_vision_last_cache_key.argtypes = [C.c_char_p, C.c_int32]
_lib.gemma_vision_last_cache_key.restype = C.c_int32

_lib.gemma_vision_fetch_softs_by_key.argtypes = [C.c_char_p, C.POINTER(C.c_uint8), C.c_int32]
_lib.gemma_vision_fetch_softs_by_key.restype = C.c_int32

_lib.gemma_submit_softs.argtypes = [C.c_int32, C.POINTER(C.c_uint8), C.c_int32, C.c_int32, C.c_int32]
_lib.gemma_submit_softs.restype = C.c_int32

# Control-vector API (Phase B).
_lib.gemma_control_register_fp16.argtypes = [C.c_char_p, C.POINTER(C.c_uint8), C.c_int32]
_lib.gemma_control_register_fp16.restype = C.c_int32

_lib.gemma_session_add_control.argtypes = [
    C.c_int32, C.c_char_p, C.c_int32,
    C.c_float, C.c_float,       # polarity, peakMagnitude
    C.c_float, C.c_float, C.c_float, C.c_float,  # attack, decay, sustainLevel, release
    C.c_int32, C.c_int32,       # shape (0-3), units (0=tokens, 1=turns)
]
_lib.gemma_session_add_control.restype = C.c_int32

_lib.gemma_session_clear_controls.argtypes = [C.c_int32]
_lib.gemma_session_clear_controls.restype = C.c_int32

_lib.gemma_session_release_control.argtypes = [C.c_int32, C.c_char_p]
_lib.gemma_session_release_control.restype = C.c_int32

_lib.gemma_vision_cache_entries.argtypes = []
_lib.gemma_vision_cache_entries.restype = C.c_int32

_lib.gemma_vision_cache_hits.argtypes = []
_lib.gemma_vision_cache_hits.restype = C.c_uint64

_lib.gemma_vision_cache_misses.argtypes = []
_lib.gemma_vision_cache_misses.restype = C.c_uint64

_lib.gemma_vision_cache_bytes.argtypes = []
_lib.gemma_vision_cache_bytes.restype = C.c_uint64

_lib.gemma_vision_cache_clear.argtypes = []
_lib.gemma_vision_cache_clear.restype = C.c_int32

_lib.gemma_active_session_ids.argtypes = [C.POINTER(C.c_int32), C.c_int32]
_lib.gemma_active_session_ids.restype = C.c_int32

_lib.gemma_session_snapshot.argtypes = [
    C.c_int32,                       # sid
    C.POINTER(C.c_int32),            # out_position
    C.POINTER(C.c_int32),            # out_state
    C.POINTER(C.c_uint32),           # out_pages
    C.c_int32,                       # max_pages
]
_lib.gemma_session_snapshot.restype = C.c_int32

_lib.gemma_page_refcount.argtypes = [C.c_int32]
_lib.gemma_page_refcount.restype = C.c_int32

_lib.gemma_page_owners.argtypes = [C.c_int32, C.POINTER(C.c_int32), C.c_int32]
_lib.gemma_page_owners.restype = C.c_int32

_lib.gemma_session_counts.argtypes = [
    C.c_int32,                       # sid
    C.POINTER(C.c_int32),            # out_page_count
    C.POINTER(C.c_int32),            # out_shared_count
]
_lib.gemma_session_counts.restype = C.c_int32

_lib.gemma_kv_snapshot_summary.argtypes = [
    C.POINTER(C.c_int32),            # out_buf [N, 5]
    C.c_int32,                       # max_sessions
]
_lib.gemma_kv_snapshot_summary.restype = C.c_int32


# --- Public Python API ---

# Single init guard (so re-imports / reloads don't double-initialize).
_init_lock = threading.Lock()
_inited = False


def init(gguf_path: str) -> None:
    global _inited
    with _init_lock:
        if _inited:
            return
        rc = _lib.gemma_init(gguf_path.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"gemma_init failed (rc={rc})")
        _inited = True


def is_ready() -> bool:
    return _lib.gemma_is_ready() == 1


def open_session(max_new_tokens: int = 256) -> int:
    sid = _lib.gemma_open_session(int(max_new_tokens))
    if sid < 0:
        raise RuntimeError("gemma_open_session failed (engine not inited or at session cap)")
    return sid


def close_session(sid: int) -> None:
    _lib.gemma_close_session(int(sid))


def pause_session(sid: int) -> None:
    _lib.gemma_pause_session(int(sid))


def submit(sid: int, tokens: list[int]) -> None:
    if not tokens:
        return
    arr = (C.c_uint32 * len(tokens))(*tokens)
    rc = _lib.gemma_submit(int(sid), arr, len(tokens))
    if rc != 0:
        raise RuntimeError(f"gemma_submit failed (rc={rc})")


def append(sid: int, tokens: list[int]) -> None:
    if not tokens:
        return
    arr = (C.c_uint32 * len(tokens))(*tokens)
    rc = _lib.gemma_append(int(sid), arr, len(tokens))
    if rc != 0:
        raise RuntimeError(f"gemma_append failed (rc={rc})")


def tick() -> int:
    return _lib.gemma_tick()


def has_work() -> bool:
    return _lib.gemma_has_work() == 1


def poll(sid: int, max_tokens: int = 64) -> list[int]:
    buf = (C.c_uint32 * max_tokens)()
    n = _lib.gemma_poll(int(sid), buf, max_tokens)
    if n < 0:
        return []
    return [buf[i] for i in range(n)]


def session_state(sid: int) -> int:
    return _lib.gemma_session_state(int(sid))


def tokenize(text: str, add_bos: bool = False) -> list[int]:
    b = text.encode("utf-8")
    # Query size first.
    n_needed = _lib.gemma_tokenize(b, len(b), 1 if add_bos else 0, None, 0)
    if n_needed < 0:
        raise RuntimeError("gemma_tokenize failed")
    buf = (C.c_uint32 * max(n_needed, 1))()
    n = _lib.gemma_tokenize(b, len(b), 1 if add_bos else 0, buf, n_needed)
    return [buf[i] for i in range(n)]


def detokenize(tokens: list[int]) -> str:
    if not tokens:
        return ""
    arr = (C.c_uint32 * len(tokens))(*tokens)
    # Query byte count first. Start with generous 16 bytes/tok upper bound.
    cap = max(16, len(tokens) * 16)
    buf = C.create_string_buffer(cap)
    n = _lib.gemma_detokenize(arr, len(tokens), buf, cap)
    if n < 0:
        raise RuntimeError("gemma_detokenize failed")
    if n >= cap:
        # rare: token expanded past our upper bound; retry with a bigger buf.
        buf = C.create_string_buffer(n + 16)
        n = _lib.gemma_detokenize(arr, len(tokens), buf, len(buf))
    return buf.raw[:n].decode("utf-8", errors="replace")


def bos_id() -> int:
    return _lib.gemma_bos_id()


def eos_id() -> int:
    return _lib.gemma_eos_id()


def active_session_count() -> int:
    return _lib.gemma_active_session_count()


# --- Vision ---

_vision_init_lock = threading.Lock()
_vision_inited = False


def vision_init(safetensors_path: str) -> None:
    """Load vision weights once. Required before submit_image_path."""
    global _vision_inited
    with _vision_init_lock:
        if _vision_inited:
            return
        rc = _lib.gemma_vision_init(safetensors_path.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"gemma_vision_init failed (rc={rc})")
        _vision_inited = True


def vision_is_ready() -> bool:
    return _lib.gemma_vision_is_ready() == 1


_RESIDENCY_NAMES = {-1: "unbound", 0: "unloaded", 1: "volatile", 2: "pinned"}


def vision_residency_state() -> str:
    s = _lib.gemma_vision_residency_state()
    return _RESIDENCY_NAMES.get(s, f"unknown({s})")


def vision_residency_bytes() -> int:
    return int(_lib.gemma_vision_residency_bytes())


def vision_allow_evict() -> None:
    _lib.gemma_vision_allow_evict()


def vision_force_drop() -> None:
    _lib.gemma_vision_force_drop()


def submit_image_path(sid: int, png_path: str) -> int:
    """Preprocess PNG at path + run vision tower + submit BOI/softs/EOI
    chunks to the session. Cache-aware: SHA-256 of file bytes keys a soft-tokens
    buffer; repeat submissions of the same image skip the vision tower.
    Returns soft-token count submitted, or raises."""
    n = _lib.gemma_submit_image_path(int(sid), png_path.encode("utf-8"))
    if n < 0:
        raise RuntimeError(f"gemma_submit_image_path failed (session={sid}, path={png_path})")
    return n


def vision_prewarm_path(png_path: str) -> int:
    """Populate the cache for an image without attaching to any session."""
    n = _lib.gemma_vision_prewarm_path(png_path.encode("utf-8"))
    if n < 0:
        raise RuntimeError(f"gemma_vision_prewarm_path failed (path={png_path})")
    return n


def vision_last_cache_key() -> str:
    """Hex SHA-256 of the most recent image submitted / prewarmed."""
    buf = C.create_string_buffer(65)
    n = _lib.gemma_vision_last_cache_key(buf, len(buf))
    if n <= 0:
        return ""
    return buf.raw[:n].decode("ascii", errors="replace")


def vision_fetch_softs_by_key(hex_key: str) -> bytes | None:
    """Copy soft tokens out of the cache for a given SHA-256 hex key.
    Returns None on cache miss. Lets clients round-trip softs across
    turns: run vision once, persist the blob locally, re-submit it
    on later turns via submit_softs() without re-running the tower."""
    if len(hex_key) != 64:
        return None
    key = hex_key.encode("ascii")
    need = _lib.gemma_vision_fetch_softs_by_key(key, None, 0)
    if need <= 0:
        return None
    buf = (C.c_uint8 * need)()
    n = _lib.gemma_vision_fetch_softs_by_key(key, buf, need)
    if n <= 0:
        return None
    return bytes(buf[:n])


def submit_softs(sid: int, softs: bytes, n_tokens: int, is_fp32: bool = True) -> int:
    """Submit pre-computed soft tokens to a session. `softs` is the raw
    byte blob previously returned by vision_fetch_softs_by_key (or an
    equivalent client-side cache). Brackets with BOI/EOI server-side;
    no vision tower runs. Returns n_tokens on success."""
    if not softs or n_tokens <= 0:
        raise ValueError("softs bytes and n_tokens required")
    buf = (C.c_uint8 * len(softs)).from_buffer_copy(softs)
    r = _lib.gemma_submit_softs(int(sid), buf, len(softs), int(n_tokens), 1 if is_fp32 else 0)
    if r < 0:
        raise RuntimeError(f"gemma_submit_softs failed (session={sid}, bytes={len(softs)}, n_tokens={n_tokens}, fp32={is_fp32})")
    return r


def vision_cache_stats() -> dict:
    return {
        "entries": _lib.gemma_vision_cache_entries(),
        "hits": _lib.gemma_vision_cache_hits(),
        "misses": _lib.gemma_vision_cache_misses(),
        "bytes": _lib.gemma_vision_cache_bytes(),
    }


def vision_cache_clear() -> int:
    return _lib.gemma_vision_cache_clear()


# --- KV snapshot (for the tenancy viz in the web demo) ---

def active_session_ids(max_n: int = 64) -> list[int]:
    buf = (C.c_int32 * max_n)()
    n = _lib.gemma_active_session_ids(buf, max_n)
    return [buf[i] for i in range(n)] if n > 0 else []


def session_snapshot(sid: int, max_pages: int = 1024) -> dict | None:
    """Return {'sid', 'position', 'state', 'pages': [phys_ids]} or None if missing."""
    pos = C.c_int32(0)
    state = C.c_int32(0)
    pages = (C.c_uint32 * max_pages)()
    n = _lib.gemma_session_snapshot(int(sid), C.byref(pos), C.byref(state), pages, max_pages)
    if n < 0:
        return None
    return {
        "sid": int(sid),
        "position": pos.value,
        "state": state.value,  # 0..4, see STATE_* constants
        "pages": [pages[i] for i in range(n)],
    }


def page_refcount(phys: int) -> int:
    return _lib.gemma_page_refcount(int(phys))


def page_owners(phys: int, max_n: int = 32) -> list[int]:
    buf = (C.c_int32 * max_n)()
    n = _lib.gemma_page_owners(int(phys), buf, max_n)
    return [buf[i] for i in range(n)] if n > 0 else []


def session_counts(sid: int) -> tuple[int, int]:
    """Return (page_count, shared_count) in one FFI call."""
    pc = C.c_int32(0); sc = C.c_int32(0)
    rc = _lib.gemma_session_counts(int(sid), C.byref(pc), C.byref(sc))
    if rc < 0:
        return (0, 0)
    return (int(pc.value), int(sc.value))


# Bulk snapshot. Returns a list of dicts, one per active session, without
# making any other FFI calls. Prefer this for 2Hz UI pollers.
def kv_snapshot_summary(max_sessions: int = 64) -> list[dict]:
    buf = (C.c_int32 * (max_sessions * 5))()
    n = _lib.gemma_kv_snapshot_summary(buf, max_sessions)
    out = []
    for i in range(n):
        base = i * 5
        out.append({
            "sid": int(buf[base + 0]),
            "position": int(buf[base + 1]),
            "state": int(buf[base + 2]),
            "page_count": int(buf[base + 3]),
            "shared_count": int(buf[base + 4]),
        })
    return out


# --- Control-vector API (Phase B) ---

_SHAPES = {"linear": 0, "exp-in": 1, "exp-out": 2, "cubic": 3}
_UNITS = {"tokens": 0, "turns": 1}

def control_register_fp16(cvec_id: str, fp16_bytes: bytes) -> None:
    """Register a cvec by string id. bytes must be HIDDEN*2 (5632 at HIDDEN=2816)."""
    if not cvec_id:
        raise ValueError("cvec_id must be non-empty")
    buf = (C.c_uint8 * len(fp16_bytes)).from_buffer_copy(fp16_bytes)
    r = _lib.gemma_control_register_fp16(cvec_id.encode("utf-8"), buf, len(fp16_bytes))
    if r != 0:
        raise RuntimeError(f"gemma_control_register_fp16 failed for id={cvec_id!r}")

def session_add_control(sid: int, cvec_id: str, layer: int,
                          polarity: float = 1.0,
                          peak_magnitude: float = 1.0,
                          attack: float = 0.0, decay: float = 0.0,
                          sustain_level: float = 1.0, release: float = 0.0,
                          shape: str = "linear", units: str = "tokens") -> None:
    r = _lib.gemma_session_add_control(
        int(sid), cvec_id.encode("utf-8"), int(layer),
        float(polarity), float(peak_magnitude),
        float(attack), float(decay), float(sustain_level), float(release),
        int(_SHAPES.get(shape, 0)), int(_UNITS.get(units, 0)),
    )
    if r != 0:
        raise RuntimeError(f"gemma_session_add_control failed (sid={sid}, cvec={cvec_id})")

def session_clear_controls(sid: int) -> None:
    _lib.gemma_session_clear_controls(int(sid))

def session_release_control(sid: int, cvec_id: str) -> None:
    _lib.gemma_session_release_control(int(sid), cvec_id.encode("utf-8"))


# --- Phase C-Read: detectors + triggers ---

_lib.gemma_session_add_detector.argtypes = [C.c_int32, C.c_char_p, C.c_char_p, C.c_int32]
_lib.gemma_session_add_detector.restype = C.c_int32

_lib.gemma_session_add_trigger.argtypes = [C.c_int32, C.c_char_p, C.c_int32, C.c_float, C.c_char_p]
_lib.gemma_session_add_trigger.restype = C.c_int32

_lib.gemma_session_clear_detectors.argtypes = [C.c_int32]
_lib.gemma_session_clear_detectors.restype = C.c_int32

_lib.gemma_session_read_intensity.argtypes = [C.c_int32, C.c_char_p]
_lib.gemma_session_read_intensity.restype = C.c_float

_TRIGGER_CONDS = {"on-exceed": 0, "on-fall": 1}

def session_add_detector(sid: int, name: str, cvec_id: str, layer: int) -> None:
    r = _lib.gemma_session_add_detector(int(sid), name.encode("utf-8"),
                                         cvec_id.encode("utf-8"), int(layer))
    if r != 0:
        raise RuntimeError(f"gemma_session_add_detector failed (sid={sid}, name={name}, cvec={cvec_id})")

def session_add_trigger(sid: int, detector_name: str, condition: str,
                          threshold: float, effector_cvec_id: str) -> None:
    cond = _TRIGGER_CONDS.get(condition)
    if cond is None:
        raise ValueError(f"condition must be one of {list(_TRIGGER_CONDS)}")
    r = _lib.gemma_session_add_trigger(int(sid), detector_name.encode("utf-8"),
                                         cond, float(threshold),
                                         effector_cvec_id.encode("utf-8"))
    if r != 0:
        raise RuntimeError(f"gemma_session_add_trigger failed")

def session_clear_detectors(sid: int) -> None:
    _lib.gemma_session_clear_detectors(int(sid))

def session_read_intensity(sid: int, name: str) -> float:
    return float(_lib.gemma_session_read_intensity(int(sid), name.encode("utf-8")))


_lib.gemma_session_poll_samples_json.argtypes = [C.c_int32, C.c_char_p, C.c_int32]
_lib.gemma_session_poll_samples_json.restype = C.c_int32

def session_poll_samples_json(sid: int) -> str:
    """Drain the session's per-token sample queue as a JSON array string.
    Returns "[]" when empty. Each array element is:
      {"token": int, "position": int,
       "detectors": {name: intensity, ...},
       "effectors": {cvec_id: magnitude, ...}}"""
    # Size the buffer conservatively: 256 bytes per sample + envelope.
    needed = _lib.gemma_session_poll_samples_json(int(sid), None, 0)
    if needed <= 0:
        return "[]"
    buf = C.create_string_buffer(max(needed, 64))
    n = _lib.gemma_session_poll_samples_json(int(sid), buf, len(buf))
    if n <= 0:
        return "[]"
    return buf.raw[:n].decode("utf-8", errors="replace")


# --- Residual capture (for /v1/control/construct) ---

_lib.gemma_set_capture_layer.argtypes = [C.c_int32]
_lib.gemma_set_capture_layer.restype = C.c_int32

_lib.gemma_get_captured_residual.argtypes = [C.POINTER(C.c_uint8), C.c_int32]
_lib.gemma_get_captured_residual.restype = C.c_int32

def set_capture_layer(layer: int) -> None:
    _lib.gemma_set_capture_layer(int(layer))

def get_captured_residual() -> bytes:
    """Copy the engine's most-recently-captured residual (HIDDEN × fp16)
    into a caller-visible buffer. Use while capture is active to grab
    the residual from the tick just completed."""
    HIDDEN_BYTES = 2816 * 2   # HIDDEN × fp16
    buf = (C.c_uint8 * HIDDEN_BYTES)()
    n = _lib.gemma_get_captured_residual(buf, HIDDEN_BYTES)
    return bytes(buf[:n]) if n > 0 else b""

def capture_residual_for_prompt(prompt: str, layer: int,
                                  chat_template: bool = True,
                                  timeout_s: float = 20.0) -> bytes:
    """Run `prompt` through the engine, return the layer-L residual at
    the last priming token as fp16 bytes. Opens and closes a throwaway
    session; pump thread (if any) drives the ticking.

    chat_template=True wraps the prompt in Gemma's user/model turn
    template so the model sees it as a real conversational turn rather
    than raw context."""
    import time
    set_capture_layer(int(layer))
    try:
        sid = open_session(max_new_tokens=1)
        try:
            if chat_template:
                wrapped = f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
            else:
                wrapped = prompt
            toks = tokenize(wrapped, add_bos=True)
            submit(sid, toks)
            # Wait until the engine finishes priming — at that moment,
            # gResidualCaptureBuf holds the last-priming-tick residual
            # at layer L. (The same tick samples the first generated
            # token; we discard it.)
            t0 = time.time()
            while session_state(sid) in (STATE_IDLE, STATE_PRIMING):
                if time.time() - t0 > timeout_s:
                    raise RuntimeError(
                        f"capture timed out after {timeout_s}s (state={session_state(sid)})")
                time.sleep(0.002)
            return get_captured_residual()
        finally:
            close_session(sid)
    finally:
        set_capture_layer(-1)


_lib.gemma_control_list_ids.argtypes = [C.c_char_p, C.c_int32]
_lib.gemma_control_list_ids.restype = C.c_int32

def control_list_ids() -> list[str]:
    """Return currently-registered cvec ids, sorted."""
    need = _lib.gemma_control_list_ids(None, 0)
    if need <= 0:
        return []
    buf = C.create_string_buffer(need + 8)
    n = _lib.gemma_control_list_ids(buf, len(buf))
    if n <= 0:
        return []
    return [s for s in buf.raw[:n].decode("utf-8", errors="replace").split(",") if s]


# --- All-layer residual capture ---

_lib.gemma_set_capture_all_layers.argtypes = [C.c_int32]
_lib.gemma_set_capture_all_layers.restype = C.c_int32

_lib.gemma_get_all_layer_residuals.argtypes = [C.POINTER(C.c_uint8), C.c_int32]
_lib.gemma_get_all_layer_residuals.restype = C.c_int32

def set_capture_all_layers(enabled: bool) -> None:
    _lib.gemma_set_capture_all_layers(1 if enabled else 0)

def get_all_layer_residuals() -> bytes:
    """Return NUM_LAYERS × HIDDEN × fp16 = 30*2816*2 = 168960 bytes.
    Layer L starts at offset L * HIDDEN * 2."""
    need = _lib.gemma_get_all_layer_residuals(None, 0)
    if need <= 0:
        return b""
    buf = (C.c_uint8 * need)()
    n = _lib.gemma_get_all_layer_residuals(buf, need)
    return bytes(buf[:n]) if n > 0 else b""

def capture_all_layer_residuals_for_prompt(prompt: str, chat_template: bool = True,
                                             timeout_s: float = 20.0) -> bytes:
    """Run prompt through engine once, return all 30 layers' last-
    priming-token residuals concatenated as one fp16 blob. Same pattern
    as capture_residual_for_prompt but captures every layer in one pass."""
    import time
    set_capture_all_layers(True)
    try:
        sid = open_session(max_new_tokens=1)
        try:
            if chat_template:
                wrapped = f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
            else:
                wrapped = prompt
            toks = tokenize(wrapped, add_bos=True)
            submit(sid, toks)
            t0 = time.time()
            while session_state(sid) in (STATE_IDLE, STATE_PRIMING):
                if time.time() - t0 > timeout_s:
                    raise RuntimeError(f"capture timeout ({timeout_s}s)")
                time.sleep(0.002)
            return get_all_layer_residuals()
        finally:
            close_session(sid)
    finally:
        set_capture_all_layers(False)
