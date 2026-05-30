#!/usr/bin/env python3
"""Model-native faithfulness metric: Gemma-4 vision-encoder embedding distance.

The position-rigid pixel metrics (MSE/SSIM) are the wrong yardstick for SVG
reconstruction — they penalize "correct structure, slightly misplaced". This
module scores faithfulness in the representation the LM itself reasons over: the
vision tower's pooled soft-token embedding. Cosine distance in that space moved
monotonically toward the target over refinement in the spec_validate probe
(corr(round, dist) down to -0.89), with far more dynamic range than MSE/SSIM.

Loads libgemma_metal.dylib in THIS process and inits ONLY the vision tower
(gemma_vision_init) — it does NOT touch the LM server on :8001, so it is safe to
run offline alongside a live bridge. FFI sequence (ffi.swift):
  gemma_vision_prewarm_bytes(png_bytes) -> runs the tower, caches softs
  gemma_vision_fetch_softs_by_key(sha256_hex) -> copies [280 x 2816] fp32 out
We mean-pool the non-zero rows -> one (2816,) embedding per image.

Usage:
    from vision_metric import embed, cosine_distance, score_trajectory
    e_t = embed("target.png"); e_r = embed(render_pil_image)
    d = cosine_distance(e_t, e_r)                       # 0 = identical, larger = worse
    traj = score_trajectory("target.png", ["r00.png", "r01.png", ...])
"""
from __future__ import annotations

import ctypes as C
import hashlib
import io
import os
import threading
from pathlib import Path
from typing import Union

import numpy as np

DYLIB = os.environ.get("GEMMA_DYLIB", "/Users/mdot/metal-microbench/libgemma_metal.dylib")
SAFETENSORS = os.environ.get(
    "GEMMA_SAFETENSORS",
    "/Users/mdot/models/gemma-4-a4b-bf16/model-00001-of-00002.safetensors")
HIDDEN = 2816
TARGET_SOFT = 280

_lib = None
_init_lock = threading.Lock()
_inited = False

ImageSrc = Union[str, Path, bytes, "object"]  # path / png-bytes / PIL.Image


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    lib = C.CDLL(DYLIB)
    lib.gemma_vision_init.argtypes = [C.c_char_p]
    lib.gemma_vision_init.restype = C.c_int32
    lib.gemma_vision_is_ready.argtypes = []
    lib.gemma_vision_is_ready.restype = C.c_int32
    lib.gemma_vision_prewarm_bytes.argtypes = [C.POINTER(C.c_uint8), C.c_int32]
    lib.gemma_vision_prewarm_bytes.restype = C.c_int32
    lib.gemma_vision_fetch_softs_by_key.argtypes = [C.c_char_p, C.POINTER(C.c_uint8), C.c_int32]
    lib.gemma_vision_fetch_softs_by_key.restype = C.c_int32
    _lib = lib
    return lib


def init(safetensors: str | None = None) -> bool:
    """Init the vision tower once (idempotent). Returns True if ready."""
    global _inited
    with _init_lock:
        if _inited:
            return True
        lib = _load_lib()
        rc = lib.gemma_vision_init((safetensors or SAFETENSORS).encode())
        _inited = (rc == 0 and lib.gemma_vision_is_ready() == 1)
        if not _inited:
            raise RuntimeError(f"gemma_vision_init failed rc={rc} ready={lib.gemma_vision_is_ready()}")
        return _inited


def _png_bytes(src: ImageSrc) -> bytes:
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    if isinstance(src, (str, Path)):
        return Path(src).read_bytes()
    # assume PIL.Image
    buf = io.BytesIO()
    src.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def embed(src: ImageSrc) -> np.ndarray:
    """Return the (2816,) mean-pooled vision-encoder embedding for an image
    (path / PNG bytes / PIL.Image). Raises on tower failure."""
    init()
    lib = _load_lib()
    data = _png_bytes(src)
    arr = (C.c_uint8 * len(data)).from_buffer_copy(data)
    n_soft = lib.gemma_vision_prewarm_bytes(arr, len(data))
    if n_soft < 0:
        raise RuntimeError(f"vision prewarm failed ({n_soft})")
    key = hashlib.sha256(data).hexdigest()
    need = lib.gemma_vision_fetch_softs_by_key(key.encode(), None, 0)
    if need < 0:
        raise RuntimeError(f"vision fetch sizing failed ({need})")
    out = (C.c_uint8 * need)()
    got = lib.gemma_vision_fetch_softs_by_key(key.encode(), out, need)
    if got <= 0:
        raise RuntimeError(f"vision fetch failed ({got})")
    softs = np.frombuffer(bytes(out[:got]), dtype=np.float32).reshape(TARGET_SOFT, HIDDEN)
    real = int((np.linalg.norm(softs, axis=1) > 1e-6).sum())
    return (softs[:real] if real > 0 else softs).mean(axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity. 0 = identical direction, larger = less faithful."""
    na = a / (np.linalg.norm(a) + 1e-12)
    nb = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(na, nb))


def score_trajectory(target: ImageSrc, renders: list[ImageSrc]) -> list[float]:
    """Cosine-distance-to-target for each render (same order as input)."""
    et = embed(target)
    return [cosine_distance(et, embed(r)) for r in renders]


if __name__ == "__main__":
    import sys, glob
    init()
    print(f"vision tower ready (dylib={DYLIB})")
    if len(sys.argv) >= 3:
        t, *rs = sys.argv[1:]
        for r, d in zip(rs, score_trajectory(t, rs)):
            print(f"  {d:.5f}  {r}")
    else:
        print("usage: vision_metric.py <target.png> <render0.png> [render1.png ...]")
