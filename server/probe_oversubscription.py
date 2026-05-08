#!/usr/bin/env python3
"""Measure the throughput payoff of M:K permutation + atomic-construction.

Submits a wave of N concurrent streams (each with a distinct ~32-token
prompt + 64 generated tokens) at increasing oversubscription levels:
  M = 1, 2, 4, 8 (saturated B), 16, 32, 64 (full residency cap).

For each M, reports:
  - aggregate tok/s         (M × per-stream tok/s)
  - per-stream tok/s        (geometric mean across streams)
  - wall time
  - approximate slot utilization (totalSlotTicks / (totalSteps × B))
    via gemma_engine_scheduler_stats

The win we expect: aggregate tok/s should keep climbing as M grows past
B=8 (because the per-CB picker now has more residents to choose from
each step → fewer wasted slot positions per CB), then plateau when
slot utilization saturates.

Pre-refactor baseline (recorded in handover): ~40% slot utilization,
flat tok/s past M=8. Post-refactor target: ≥80% slot utilization,
visible aggregate-tok/s gain through M=32 at minimum.
"""
from __future__ import annotations
import ctypes as C
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemma_ffi as g  # noqa: E402

GGUF = os.environ.get(
    "GGUF_PATH",
    "/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
)

# Bind the unwrapped scheduler-stats FFI directly (Python wrapper doesn't
# expose it). Returns 0 on success, fills the out-pointers with current
# engine totals.
_lib = g._lib
_lib.gemma_engine_scheduler_stats.argtypes = [
    C.POINTER(C.c_int32),   # B
    C.POINTER(C.c_int32),   # resident_count
    C.POINTER(C.c_int64),   # total_steps
    C.POINTER(C.c_int64),   # total_tokens
    C.POINTER(C.c_int64),   # total_slot_ticks
    C.POINTER(C.c_double),  # last_step_ms
]
_lib.gemma_engine_scheduler_stats.restype = C.c_int32


def scheduler_stats() -> dict:
    B = C.c_int32(0)
    resident = C.c_int32(0)
    steps = C.c_int64(0)
    toks = C.c_int64(0)
    slot_ticks = C.c_int64(0)
    last_ms = C.c_double(0)
    rc = _lib.gemma_engine_scheduler_stats(
        C.byref(B), C.byref(resident), C.byref(steps),
        C.byref(toks), C.byref(slot_ticks), C.byref(last_ms))
    if rc != 0:
        return {}
    return {
        "B": B.value,
        "resident": resident.value,
        "total_steps": steps.value,
        "total_tokens": toks.value,
        "total_slot_ticks": slot_ticks.value,
        "last_step_ms": last_ms.value,
    }


def run_wave(n_streams: int, completion_tokens: int = 64,
             prompt_tokens: int = 32, deadline_s: float = 180.0) -> dict:
    """Submit N concurrent streams, drive to completion, return metrics."""
    sampling = g.SamplingParams(
        temperature=0.0,
        max_new_tokens=completion_tokens,
        eos_token_id=999_999,  # impossible token → never stops early
    )
    base_sid = 100_000 + n_streams * 1000   # unique per wave
    specs = []
    for i in range(n_streams):
        sid = base_sid + i
        # Distinct prompts per stream (no in-batch dedup — measures the
        # raw scheduling efficiency, not cache reuse).
        prompt = [2] + [(50 + sid * 11 + j * 7) % 32000
                         for j in range(prompt_tokens - 1)]
        specs.append(g.StreamSpec(
            stream_id=sid, action=0, tokens=prompt, sampling=sampling))

    pending = set(s.stream_id for s in specs)
    finals: dict[int, g.StreamUpdate] = {}
    completion_counts: dict[int, int] = {sid: 0 for sid in pending}

    pre_stats = scheduler_stats()
    t0 = time.time()
    rc = g.submit(specs)
    if rc != 0:
        return {"error": f"submit rc={rc}"}

    deadline = t0 + deadline_s
    while pending and time.time() < deadline:
        for u in g.poll(timeout_ms=200):
            completion_counts[u.stream_id] = (
                completion_counts.get(u.stream_id, 0)
                + len(u.new_tokens))
            if u.state == 2:
                finals[u.stream_id] = u
                pending.discard(u.stream_id)
    t1 = time.time()
    post_stats = scheduler_stats()

    if pending:
        return {"error": f"timeout: {len(pending)}/{n_streams} pending"}

    wall = t1 - t0
    total_completion = sum(completion_counts.values())
    per_stream_rates = [completion_counts[sid] / wall for sid in completion_counts]

    # Slot utilization: how many of the B kernel positions actually had
    # work each step. 100% = picker found B busy residents every CB.
    d_steps = post_stats["total_steps"] - pre_stats["total_steps"]
    d_slot_ticks = (post_stats["total_slot_ticks"]
                    - pre_stats["total_slot_ticks"])
    B = post_stats["B"]
    slot_util = (d_slot_ticks / (d_steps * B)) if (d_steps > 0 and B > 0) else 0.0

    return {
        "n_streams": n_streams,
        "wall_s": wall,
        "total_completion_tokens": total_completion,
        "aggregate_tok_per_s": total_completion / wall,
        "per_stream_tok_per_s_mean": statistics.fmean(per_stream_rates),
        "per_stream_tok_per_s_min": min(per_stream_rates),
        "per_stream_tok_per_s_max": max(per_stream_rates),
        "engine_steps": d_steps,
        "engine_slot_ticks": d_slot_ticks,
        "slot_utilization_pct": slot_util * 100.0,
        "B": B,
    }


def main() -> int:
    print("=== M:K oversubscription throughput probe ===")
    rc = g.init(GGUF)
    if rc != 0:
        print(f"  init failed: rc={rc}")
        return 1

    # Warmup: small wave, discard stats
    print("  warmup (M=4, 32 toks)...", flush=True)
    _ = run_wave(n_streams=4, completion_tokens=32)

    levels = [1, 2, 4, 8, 16, 32, 64]
    print()
    print(f"  {'M':>3} | {'wall':>6} | {'agg t/s':>8} | "
          f"{'per-stream t/s':>15} | {'slot util':>9} | {'steps':>6}")
    print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*8}-+-{'-'*15}-+-{'-'*9}-+-{'-'*6}")
    results = []
    for M in levels:
        r = run_wave(n_streams=M, completion_tokens=64)
        if "error" in r:
            print(f"  M={M:3d} ERROR: {r['error']}")
            continue
        results.append(r)
        per_stream = (
            f"{r['per_stream_tok_per_s_min']:5.1f}-"
            f"{r['per_stream_tok_per_s_max']:5.1f}")
        print(
            f"  {r['n_streams']:>3} | {r['wall_s']:>6.2f} | "
            f"{r['aggregate_tok_per_s']:>8.1f} | "
            f"{per_stream:>15} | "
            f"{r['slot_utilization_pct']:>8.1f}% | "
            f"{r['engine_steps']:>6}"
        )

    # Speedup table vs. M=1 baseline
    if results:
        baseline = results[0]["aggregate_tok_per_s"]
        print()
        print(f"  Aggregate-tok/s speedup vs M=1 baseline ({baseline:.1f} t/s):")
        for r in results:
            sp = r["aggregate_tok_per_s"] / baseline
            print(f"    M={r['n_streams']:3d}: {sp:5.2f}x  "
                  f"(slot_util={r['slot_utilization_pct']:5.1f}%)")

    g.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
