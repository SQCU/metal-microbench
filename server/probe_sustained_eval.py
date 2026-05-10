#!/usr/bin/env python3
"""Sustained-throughput eval-shape probe.

Mimics what an eval harness actually does to the engine: a continuous
rolling stream of requests where new ones get submitted as old ones
complete, keeping M concurrent in-flight at all times for a wall-time
budget. Realistic prompt + completion sizes (256 + 128 tokens by default).

Reports across the run:
  - total tokens generated
  - aggregate completion-tok/s
  - aggregate prefill-tok/s
  - requests-per-second
  - per-request wall latency: p50, p90, p99
  - engine slot-tick utilization (totalSlotTicks / (totalSteps × B))

Compared to bench_b_sweep.py (one-shot wave) and probe_oversubscription.py
(submit-and-drain at varying M), this exercises the SUBMISSION QUEUE +
ADMISSION + COMPLETION-TRIGGERED RESUBMIT loop that an eval harness
runs into. If polynomial-in-task-count slowdowns ever sneak back in
(bridge state accumulation, KV pool fragmentation, etc.), this probe
will catch them — wall-time-per-request should be flat across the run.

Usage:
  python3 server/probe_sustained_eval.py
  WALL_BUDGET_S=300 M_INFLIGHT=8 PROMPT_TOK=256 COMPL_TOK=128 \\
      python3 server/probe_sustained_eval.py
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
WALL_BUDGET_S = float(os.environ.get("WALL_BUDGET_S", "300"))   # 5 min
M_INFLIGHT = int(os.environ.get("M_INFLIGHT", "8"))
PROMPT_TOK = int(os.environ.get("PROMPT_TOK", "256"))
COMPL_TOK = int(os.environ.get("COMPL_TOK", "128"))
WINDOW_S = float(os.environ.get("WINDOW_S", "30"))   # rolling stats window

# Bind scheduler-stats FFI directly (Python wrapper doesn't expose it).
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
    B = C.c_int32(0); resident = C.c_int32(0)
    steps = C.c_int64(0); toks = C.c_int64(0)
    slot_ticks = C.c_int64(0); last_ms = C.c_double(0)
    rc = _lib.gemma_engine_scheduler_stats(
        C.byref(B), C.byref(resident), C.byref(steps),
        C.byref(toks), C.byref(slot_ticks), C.byref(last_ms))
    if rc != 0: return {}
    return dict(B=B.value, resident=resident.value,
                total_steps=steps.value, total_tokens=toks.value,
                total_slot_ticks=slot_ticks.value, last_step_ms=last_ms.value)


def make_prompt(sid: int, n: int) -> list[int]:
    """Distinct deterministic prompt per stream-id, no in-batch dedup."""
    return [2] + [(50 + sid * 11 + j * 7) % 32000 for j in range(n - 1)]


def make_spec(sid: int, sampling: g.SamplingParams) -> g.StreamSpec:
    return g.StreamSpec(
        stream_id=sid, action=0,
        tokens=make_prompt(sid, PROMPT_TOK), sampling=sampling)


def main() -> int:
    print(f"=== sustained-throughput eval-shape probe ===")
    print(f"  budget:        {WALL_BUDGET_S:.0f}s")
    print(f"  in-flight:     M={M_INFLIGHT}")
    print(f"  prompt size:   {PROMPT_TOK} tok")
    print(f"  completion:    {COMPL_TOK} tok/req")
    rc = g.init(GGUF)
    if rc != 0:
        print(f"  init failed: rc={rc}"); return 1

    sampling = g.SamplingParams(
        temperature=0.7,
        max_new_tokens=COMPL_TOK,
        eos_token_id=999_999,   # impossible → never early-stops
    )

    # Track per-request birth-time so we can measure end-to-end latency.
    next_sid = 1_000_000
    birth_at: dict[int, float] = {}
    in_flight: set[int] = set()
    completed_latencies: list[float] = []
    total_completion_tokens = 0
    total_prompt_tokens = 0
    total_requests_completed = 0

    # Rolling window of recent completion timestamps for instantaneous t/s
    recent_completions: list[tuple[float, int]] = []  # (timestamp, tokens)

    def fill_to_M(now: float):
        """Submit new streams until in_flight reaches M."""
        nonlocal next_sid, total_prompt_tokens
        new_specs = []
        while len(in_flight) + len(new_specs) < M_INFLIGHT:
            sid = next_sid; next_sid += 1
            new_specs.append(make_spec(sid, sampling))
            birth_at[sid] = now
            total_prompt_tokens += PROMPT_TOK
        if new_specs:
            rc = g.submit(new_specs)
            if rc != 0:
                print(f"  submit rc={rc} mid-run"); return False
            for s in new_specs:
                in_flight.add(s.stream_id)
        return True

    print()
    print(f"  {'t':>5} | {'reqs':>5} | {'compl tok':>9} | "
          f"{'agg t/s':>8} | {'inst t/s':>8} | {'p50 ms':>7} | "
          f"{'p90 ms':>7} | {'in-flight':>9}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-"
          f"{'-'*7}-+-{'-'*7}-+-{'-'*9}")

    pre_stats = scheduler_stats()
    t0 = time.time()
    deadline = t0 + WALL_BUDGET_S
    last_print = t0
    if not fill_to_M(t0):
        return 2

    while time.time() < deadline:
        for u in g.poll(timeout_ms=100):
            total_completion_tokens += len(u.new_tokens)
            if u.state == 2:
                # Request done.
                done_at = time.time()
                latency = done_at - birth_at.pop(u.stream_id, done_at)
                completed_latencies.append(latency)
                recent_completions.append((done_at, u.completion_tokens_emitted))
                total_requests_completed += 1
                in_flight.discard(u.stream_id)

        # Refill.
        now = time.time()
        if not fill_to_M(now):
            return 3

        # Periodic print.
        if now - last_print >= 10.0:
            elapsed = now - t0
            agg_tok_s = total_completion_tokens / elapsed
            # Instantaneous tok/s from recent window.
            window_cut = now - WINDOW_S
            recent_completions = [(t, n) for (t, n) in recent_completions
                                   if t >= window_cut]
            window_toks = sum(n for (_, n) in recent_completions)
            inst_tok_s = window_toks / min(WINDOW_S, elapsed)
            # Recent latency percentiles.
            recent_lat = completed_latencies[-50:] if completed_latencies else [0]
            recent_lat_sorted = sorted(recent_lat)
            p50 = recent_lat_sorted[len(recent_lat_sorted)//2] * 1000
            p90 = (recent_lat_sorted[int(len(recent_lat_sorted)*0.9)] * 1000
                   if len(recent_lat_sorted) > 1 else p50)
            print(f"  {elapsed:>5.0f} | {total_requests_completed:>5} | "
                  f"{total_completion_tokens:>9} | "
                  f"{agg_tok_s:>8.1f} | {inst_tok_s:>8.1f} | "
                  f"{p50:>7.1f} | {p90:>7.1f} | "
                  f"{len(in_flight):>9}", flush=True)
            last_print = now

    # Drain remaining in-flight (within a tight deadline).
    drain_deadline = time.time() + 30.0
    while in_flight and time.time() < drain_deadline:
        for u in g.poll(timeout_ms=100):
            total_completion_tokens += len(u.new_tokens)
            if u.state == 2:
                done_at = time.time()
                completed_latencies.append(done_at - birth_at.pop(u.stream_id, done_at))
                total_requests_completed += 1
                in_flight.discard(u.stream_id)

    t1 = time.time()
    wall = t1 - t0
    post_stats = scheduler_stats()

    # ── final stats ──
    print()
    print(f"=== Summary (wall = {wall:.1f}s) ===")
    print(f"  Requests completed:      {total_requests_completed}")
    print(f"  Completion tokens:       {total_completion_tokens}")
    print(f"  Prompt tokens:           {total_prompt_tokens}")
    print(f"  Aggregate completion:    {total_completion_tokens / wall:7.1f} tok/s")
    print(f"  Aggregate prompt+gen:    {(total_completion_tokens + total_prompt_tokens) / wall:7.1f} tok/s")
    print(f"  Requests per second:     {total_requests_completed / wall:7.2f} req/s")
    if completed_latencies:
        s = sorted(completed_latencies)
        p50 = s[len(s)//2]
        p90 = s[int(len(s)*0.9)]
        p99 = s[int(len(s)*0.99)]
        print(f"  Per-request latency:     p50={p50*1000:.0f}ms  "
              f"p90={p90*1000:.0f}ms  p99={p99*1000:.0f}ms  "
              f"max={s[-1]*1000:.0f}ms  mean={statistics.fmean(s)*1000:.0f}ms")

    if pre_stats and post_stats:
        d_steps = post_stats["total_steps"] - pre_stats["total_steps"]
        d_slots = post_stats["total_slot_ticks"] - pre_stats["total_slot_ticks"]
        B = post_stats["B"]
        print(f"  Engine steps:            {d_steps}")
        print(f"  Avg step time:           {(wall*1000)/max(d_steps,1):.2f} ms")
        if B > 0 and d_steps > 0:
            slot_fill = d_slots / (d_steps * B)
            slot_fill_capped = min(slot_fill, 1.0) * 100
            oversubscription = max(slot_fill, 1.0)
            print(f"  Kernel slot fill:        {slot_fill_capped:.1f}% "
                  f"(oversubscription factor {oversubscription:.1f}x)")

    g.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
