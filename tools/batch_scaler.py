#!/usr/bin/env python3
"""Single source of truth for engine batch saturation — stop reinventing the tensor core.

The Gemma engine decodes through a FIXED-WIDTH kernel: B=8 streams per step
(``bootstrap.swift:431`` — "K=B=8 kernel-batch positions"), behind an admission
cap of M=64 logical sessions. The kernel pays its full B-wide dispatch cost
*regardless* of how many of those 8 lanes carry real work, so running FEWER than
B concurrent streams is strictly wasteful — you buy the 8-wide kernel and use 2
lanes of it. Every experiment that hardcoded ``--workers 6`` /
``ThreadPoolExecutor(max_workers=4)`` / an unbounded ``Promise.all`` was
independently re-deriving (and usually mis-guessing) a number the hardware
already fixes. That is the class of bug this module exists to delete.

The number must come from ONE place: the engine *declares* its kernel batch
width in ``GET /health`` (``capabilities.kernel_batch`` and ``.max_sessions``);
clients query it ONCE (cached) and saturate. No probing, no per-experiment
sweeps to "discover" the optimal batch — the tensor core's optimal shape is
known, you read it. If the engine is too old to declare the field, we fall back
to an env override and then the compiled-in default, so a caller NEVER ends up
running below the kernel width by accident.

The only lines a client needs
-----------------------------
Drop-in for ``ThreadPoolExecutor(max_workers=GUESS).map(fn, jobs)``::

    from batch_scaler import saturated_map
    for r in saturated_map(run_one, jobs):    # runs at the engine's kernel width
        ...

When you want the executor itself (``.submit`` / ``as_completed``)::

    from batch_scaler import SaturatingPool
    with SaturatingPool(n_items=len(jobs)) as pool:
        futs = [pool.submit(run_one, j) for j in jobs]

When you just want the width (e.g. to size something else)::

    from batch_scaler import target_workers
    w = target_workers(n_items=len(jobs))

Design notes
------------
* ``fill`` (default 1) multiplies the kernel width before clamping to M. The
  default saturates the kernel exactly (B in flight). A modest overschedule
  (``fill=2``) keeps the kernel full despite decode-length variance — when one
  stream finishes early another is already admitted — at the cost of more KV
  pressure. Above M the engine just queues, so we never exceed M.
* We clamp DOWN to ``n_items`` when given: there is no point spawning 8 workers
  for 3 jobs. We never clamp below 1.
* Resolution is cached per base URL; the first resolution prints a one-line
  banner so it is visible in logs which width (and from which source) was used.
* Stdlib only (urllib/json/os/threading/concurrent.futures) so any client in any
  directory can ``import batch_scaler`` with no new dependency.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Compiled-in fallbacks. These MUST mirror the engine: bootstrap.swift:431
# `let B = 8` (kernel-batch width) and the M=64 logical-session admission cap.
# They are only used when /health does not declare the value and no env override
# is set — i.e. against an engine too old to advertise its own shape.
_DEFAULT_KERNEL_BATCH = 8
_DEFAULT_MAX_SESSIONS = 64

_HEALTH_TIMEOUT_S = 3.0

# Cache: base_url -> {"kernel_batch": int, "max_sessions": int, "source": str}
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
_banner_shown: set[str] = set()


def default_base() -> str:
    """The engine base URL clients talk to (env GEMMA_BASE, else localhost:8001)."""
    return os.environ.get("GEMMA_BASE", "http://127.0.0.1:8001").rstrip("/")


def _probe_health(base: str) -> dict:
    """GET {base}/health and pull the declared batch shape. Returns a partial
    dict with whatever of {kernel_batch, max_sessions} the engine advertised.
    Never raises — a dead/old engine just yields {}."""
    try:
        with urllib.request.urlopen(base + "/health", timeout=_HEALTH_TIMEOUT_S) as r:
            doc = json.loads(r.read().decode())
    except Exception:
        return {}
    caps = doc.get("capabilities") or {}
    out: dict = {}
    # Accept either capabilities.* (preferred) or a top-level mirror, and a few
    # name spellings, so the client tolerates engine-side naming drift.
    for key, names in (
        ("kernel_batch", ("kernel_batch", "max_batch", "batch_width")),
        ("max_sessions", ("max_sessions", "max_streams", "session_cap")),
    ):
        for src in (caps, doc):
            for n in names:
                v = src.get(n)
                if isinstance(v, int) and v > 0:
                    out[key] = v
                    break
            if key in out:
                break
    return out


def _resolve(base: Optional[str]) -> dict:
    """Resolve (and cache) the batch shape for ``base``. Precedence per field:
    env override > /health declaration > compiled-in default."""
    base = (base or default_base()).rstrip("/")
    with _cache_lock:
        cached = _cache.get(base)
    if cached is not None:
        return cached

    health = _probe_health(base)

    def pick(env_key: str, health_key: str, default: int) -> tuple[int, str]:
        env = os.environ.get(env_key)
        if env is not None:
            try:
                v = int(env)
                if v > 0:
                    return v, "env"
            except ValueError:
                pass
        if health_key in health:
            return health[health_key], "health"
        return default, "default"

    kb, kb_src = pick("GEMMA_KERNEL_BATCH", "kernel_batch", _DEFAULT_KERNEL_BATCH)
    ms, ms_src = pick("GEMMA_MAX_SESSIONS", "max_sessions", _DEFAULT_MAX_SESSIONS)
    # max_sessions can never be below the kernel width — that would make the cap
    # itself underfill the kernel.
    ms = max(ms, kb)
    resolved = {"kernel_batch": kb, "max_sessions": ms,
                "source": kb_src, "session_source": ms_src, "base": base}
    with _cache_lock:
        _cache[base] = resolved
    return resolved


def _banner(base: str, info: dict) -> None:
    if base in _banner_shown:
        return
    _banner_shown.add(base)
    print(
        f"[batch_scaler] saturating at kernel_batch={info['kernel_batch']} "
        f"(max_sessions={info['max_sessions']}, source={info['source']}) "
        f"base={base}",
        file=sys.stderr, flush=True,
    )


def kernel_batch(base: Optional[str] = None) -> int:
    """The engine's kernel batch width B — the number of concurrent streams you
    must run to saturate one decode step. Running below this wastes the kernel."""
    return _resolve(base)["kernel_batch"]


def max_sessions(base: Optional[str] = None) -> int:
    """The engine's admission cap M — the most logical sessions it will admit
    before queueing. Never schedule more than this concurrently."""
    return _resolve(base)["max_sessions"]


def target_workers(n_items: Optional[int] = None, base: Optional[str] = None,
                   fill: int = 1) -> int:
    """How many workers to run to saturate the engine for ``n_items`` of work.

    = clamp( kernel_batch * fill , 1 , max_sessions ), then clamp down to
    n_items if fewer items than that exist. This is the ONLY place a client
    should get a concurrency number from."""
    info = _resolve(base)
    _banner(info["base"], info)
    w = info["kernel_batch"] * max(1, int(fill))
    w = min(w, info["max_sessions"])
    if n_items is not None:
        w = min(w, max(1, int(n_items)))
    return max(1, w)


def saturated_map(fn: Callable[[T], R], items: Iterable[T], *,
                  base: Optional[str] = None, fill: int = 1,
                  ordered: bool = True) -> Iterator[R]:
    """Run ``fn`` over ``items`` at the engine's kernel width and yield results.

    Drop-in replacement for ``ThreadPoolExecutor(max_workers=GUESS).map(fn,
    items)`` — except the width is the engine's, not a guess. ``ordered=True``
    yields in input order (like ``.map``); ``ordered=False`` yields as each
    completes (better for live progress when order does not matter). Exceptions
    in ``fn`` propagate on iteration, same as ``.map`` — wrap ``fn`` in your own
    try/except if you want per-item error capture (the existing drivers do)."""
    items = list(items)
    if not items:
        return
    w = target_workers(len(items), base, fill)
    with ThreadPoolExecutor(max_workers=w) as ex:
        if ordered:
            yield from ex.map(fn, items)
        else:
            from concurrent.futures import as_completed
            futs = [ex.submit(fn, it) for it in items]
            for f in as_completed(futs):
                yield f.result()


class SaturatingPool:
    """Context manager yielding a ThreadPoolExecutor sized to the engine's kernel
    width. For clients that want ``.submit`` / ``as_completed`` rather than the
    one-shot ``saturated_map``::

        with SaturatingPool(n_items=len(jobs)) as pool:
            futs = [pool.submit(run, j) for j in jobs]
            for f in as_completed(futs):
                ...
    """

    def __init__(self, n_items: Optional[int] = None, base: Optional[str] = None,
                 fill: int = 1):
        self.workers = target_workers(n_items, base, fill)
        self._ex: Optional[ThreadPoolExecutor] = None

    def __enter__(self) -> ThreadPoolExecutor:
        self._ex = ThreadPoolExecutor(max_workers=self.workers)
        return self._ex

    def __exit__(self, *exc) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None


if __name__ == "__main__":
    # Quick self-check / introspection: print what this engine declares.
    info = _resolve(None)
    print(json.dumps(info, indent=2))
    print(f"target_workers(100 items) = {target_workers(100)}")
    print(f"target_workers(3 items)   = {target_workers(3)}")
    print(f"target_workers(100, fill=2) = {target_workers(100, fill=2)}")
