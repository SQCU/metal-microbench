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


def submit_image_path(sid: int, png_path: str) -> int:
    """Preprocess PNG at path + run vision tower + submit BOI/softs/EOI
    chunks to the session. Returns soft-token count submitted, or raises."""
    n = _lib.gemma_submit_image_path(int(sid), png_path.encode("utf-8"))
    if n < 0:
        raise RuntimeError(f"gemma_submit_image_path failed (session={sid}, path={png_path})")
    return n
