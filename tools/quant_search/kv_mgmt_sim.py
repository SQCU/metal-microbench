"""KV cache memory-management simulator.

Compares allocation strategies for the bridge's KV-page pool against
realistic admission-burst workloads. Output: total burst wall, max
single-lock hold time, admission latency percentiles, lock-held-time
breakdown, blocked-progress fraction.

Strategies modeled (per codex spec, 2026-05-06):
  S1 PRE_RESERVE_SYNC_MEMSET    — current pathology (full lookahead
                                   pre-reserve + sync memset under lock)
  S2 LAZY_GROW_SYNC_MEMSET      — small initial reserve, grow on demand
  S3 LAZY_GROW_ASYNC_MEMSET     — lazy grow + async GPU zero-fill
  S4 PRE_ZEROED_POOL            — background-zeroed free list, alloc-only
                                   under lock

Event-driven, not fixed-timestep.

Usage::

    python tools/quant_search/kv_mgmt_sim.py
"""
from __future__ import annotations

import heapq
import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable

# ──────────────────────────────────────────────────────────────────────
# Calibration constants (from the actual code + measured numbers)
# ──────────────────────────────────────────────────────────────────────

NUM_PHYS_PAGES = 5000             # bridge's free_pages observed ~4036
PAGE_FULL_SIZE_TOKENS = 8         # PAGE_FULL = 8 tokens per page
PAGE_FULL_SIZE_BYTES = 800_000    # ~800 KB across 30 layers (K + V)
MEMSET_BANDWIDTH_GBPS = 250       # M5 Max memset ceiling
ALLOC_US = 5.0                    # alloc_fresh cost (search free list, etc.)
BLOCK_TABLE_US = 50.0             # write MAX_PAGES_PER_SLOT entries
AR_STEP_MS = 93.0                 # observed b8 AR tick wall time
PREFILL_TILE_MS = 133.0           # textMultiPrefill cost per 256-token tile
INITIAL_RUNWAY_TOKENS = 256       # S2/S3 initial reserve beyond prefill
GROW_QUANTUM_TOKENS = 128         # S2/S3 grow chunk size

# Background zero worker rate for S4 (pre-zeroed pool):
# overlaps with bridge work, so doesn't directly count against bridge
# wall, but does limit how fast pages refill into the zeroed free list.
BG_ZERO_THROUGHPUT_PAGES_PER_S = 1.0e9 / PAGE_FULL_SIZE_BYTES * MEMSET_BANDWIDTH_GBPS  # pages/s


def memset_time_s(n_pages: int) -> float:
    """Synchronous memset of n KV pages on the bridge thread."""
    bytes_total = n_pages * PAGE_FULL_SIZE_BYTES
    return bytes_total / (MEMSET_BANDWIDTH_GBPS * 1.0e9)


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class Session:
    sid: int
    admission_time: float
    max_new_tokens: int
    prefill_tokens: int
    position: int = 0
    allocated_pages: int = 0
    admitted_at: float | None = None
    ready_at: float | None = None         # first AR token can be emitted
    completed_at: float | None = None


@dataclass
class PagePool:
    free_zeroed: int                       # ready-to-use pages
    free_dirty: int = 0                    # need zeroing before use
    allocated: int = 0                     # in-use by sessions
    needs_zeroing_queue: list = field(default_factory=list)


@dataclass
class LockState:
    holder_sid: int | None = None          # session whose work currently holds
    held_since: float | None = None
    hold_intervals: list = field(default_factory=list)  # (start, end, op_tag)
    longest_hold: float = 0.0
    longest_hold_op: str = ""
    total_held_time: float = 0.0
    waiter_count_timeline: list = field(default_factory=list)  # (t, n_waiters)


@dataclass
class SimState:
    now: float = 0.0
    workload: list = field(default_factory=list)     # admission events
    sessions: dict = field(default_factory=dict)     # sid -> Session
    pool: PagePool = field(default_factory=lambda: PagePool(free_zeroed=NUM_PHYS_PAGES))
    lock: LockState = field(default_factory=LockState)
    events: list = field(default_factory=list)       # heap: (t, seq, kind, payload)
    seq: int = 0
    # bridge thread is the implicit single worker; bridge_busy_until is the
    # earliest time it can take a new task.
    bridge_busy_until: float = 0.0
    completed_sessions: list = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Lock helpers — track contention metrics
# ──────────────────────────────────────────────────────────────────────


def acquire_lock(state: SimState, sid: int, op: str, duration_s: float) -> float:
    """Schedule a lock-held operation. Returns the wall-clock time at
    which the lock will be released.

    With a single-bridge-thread model, contention shows up as the
    bridge_busy_until pushing forward — newly-arriving operations
    queue behind whatever's already running.
    """
    start = max(state.now, state.bridge_busy_until)
    end = start + duration_s
    state.lock.hold_intervals.append((start, end, op, sid))
    state.lock.total_held_time += duration_s
    if duration_s > state.lock.longest_hold:
        state.lock.longest_hold = duration_s
        state.lock.longest_hold_op = f"{op}/sid={sid}/dur={duration_s*1000:.1f}ms"
    state.bridge_busy_until = end
    return end


# ──────────────────────────────────────────────────────────────────────
# Strategy: PRE_RESERVE_SYNC_MEMSET (current pathology)
# ──────────────────────────────────────────────────────────────────────


def strategy_pre_reserve_sync(state: SimState, sess: Session) -> None:
    """At admission: compute lookahead, allocate ALL needed pages,
    synchronously memset every page, write block table — all under
    one big lock-hold."""
    lookahead_tokens = sess.prefill_tokens + sess.max_new_tokens + 8
    pages_needed = (lookahead_tokens + PAGE_FULL_SIZE_TOKENS - 1) // PAGE_FULL_SIZE_TOKENS
    if state.pool.free_zeroed < pages_needed:
        # Pool exhaustion — model as a long retry that fails, falls back
        # to whatever pages are available. Real engine just prints
        # "pool exhausted" and the admission stalls. We model as: sit
        # waiting for free pages from background zeroing (here there
        # is none, so we just fail to admit and move the wall forward
        # by what would've been the memset time anyway).
        avail = state.pool.free_zeroed
        if avail == 0:
            # Pure starvation; bridge ticks forward by 1ms and retries.
            sess.admitted_at = state.bridge_busy_until + 0.001
            return
        pages_needed = avail
    # Acquire lock + alloc + memset + block-table install
    alloc_s = pages_needed * ALLOC_US * 1e-6
    memset_s = memset_time_s(pages_needed)
    bt_s = BLOCK_TABLE_US * 1e-6
    total_held = alloc_s + memset_s + bt_s
    end_t = acquire_lock(state, sess.sid,
                          f"PRE_RESERVE alloc+memset {pages_needed}p",
                          total_held)
    state.pool.free_zeroed -= pages_needed
    state.pool.allocated += pages_needed
    sess.allocated_pages = pages_needed
    sess.admitted_at = end_t
    # Ready_at: includes prefill cost (must run before first AR token)
    prefill_tiles = (sess.prefill_tokens + 255) // 256
    prefill_s = prefill_tiles * PREFILL_TILE_MS * 1e-3
    prefill_end = acquire_lock(state, sess.sid,
                                 f"prefill {sess.prefill_tokens}t",
                                 prefill_s)
    sess.ready_at = prefill_end


# ──────────────────────────────────────────────────────────────────────
# Strategy: LAZY_GROW_SYNC_MEMSET
# ──────────────────────────────────────────────────────────────────────


def strategy_lazy_grow_sync(state: SimState, sess: Session) -> None:
    """At admission: reserve only prefill + INITIAL_RUNWAY pages.
    Sync memset of just those pages. Growth happens later as
    `position` advances — modelled below in advance_decode."""
    initial_tokens = sess.prefill_tokens + INITIAL_RUNWAY_TOKENS
    pages_needed = (initial_tokens + PAGE_FULL_SIZE_TOKENS - 1) // PAGE_FULL_SIZE_TOKENS
    if state.pool.free_zeroed < pages_needed:
        sess.admitted_at = state.bridge_busy_until + 0.001
        return
    alloc_s = pages_needed * ALLOC_US * 1e-6
    memset_s = memset_time_s(pages_needed)
    bt_s = BLOCK_TABLE_US * 1e-6
    end_t = acquire_lock(state, sess.sid,
                          f"LAZY init alloc+memset {pages_needed}p",
                          alloc_s + memset_s + bt_s)
    state.pool.free_zeroed -= pages_needed
    state.pool.allocated += pages_needed
    sess.allocated_pages = pages_needed
    sess.admitted_at = end_t
    prefill_tiles = (sess.prefill_tokens + 255) // 256
    prefill_s = prefill_tiles * PREFILL_TILE_MS * 1e-3
    prefill_end = acquire_lock(state, sess.sid,
                                 f"prefill {sess.prefill_tokens}t",
                                 prefill_s)
    sess.ready_at = prefill_end


# ──────────────────────────────────────────────────────────────────────
# Strategy: LAZY_GROW_ASYNC_MEMSET
# ──────────────────────────────────────────────────────────────────────


def strategy_lazy_grow_async(state: SimState, sess: Session) -> None:
    """Lazy reserve like S2, but memset is encoded as a Metal CB
    chained before the next AR step — overlaps with compute. Lock
    only covers the alloc metadata + block table (microseconds).
    """
    initial_tokens = sess.prefill_tokens + INITIAL_RUNWAY_TOKENS
    pages_needed = (initial_tokens + PAGE_FULL_SIZE_TOKENS - 1) // PAGE_FULL_SIZE_TOKENS
    if state.pool.free_zeroed < pages_needed:
        sess.admitted_at = state.bridge_busy_until + 0.001
        return
    alloc_s = pages_needed * ALLOC_US * 1e-6
    bt_s = BLOCK_TABLE_US * 1e-6
    # Lock held only for metadata work — memset happens off-bridge-thread
    end_t = acquire_lock(state, sess.sid,
                          f"ASYNC alloc+bt {pages_needed}p",
                          alloc_s + bt_s)
    state.pool.free_zeroed -= pages_needed
    state.pool.allocated += pages_needed
    sess.allocated_pages = pages_needed
    sess.admitted_at = end_t
    # Memset overlaps with prefill — assume prefill > memset for any
    # realistic page count, so prefill dominates.
    prefill_tiles = (sess.prefill_tokens + 255) // 256
    prefill_s = prefill_tiles * PREFILL_TILE_MS * 1e-3
    prefill_end = acquire_lock(state, sess.sid,
                                 f"prefill {sess.prefill_tokens}t",
                                 prefill_s)
    sess.ready_at = prefill_end


# ──────────────────────────────────────────────────────────────────────
# Strategy: PRE_ZEROED_POOL
# ──────────────────────────────────────────────────────────────────────


def strategy_pre_zeroed_pool(state: SimState, sess: Session) -> None:
    """Background-zeroed free list. Lock-held op is just popping pages
    + writing block table — sub-microsecond per page. Refilling the
    free list from dirty happens on a background worker we don't
    block on (idealized).
    """
    initial_tokens = sess.prefill_tokens + INITIAL_RUNWAY_TOKENS
    pages_needed = (initial_tokens + PAGE_FULL_SIZE_TOKENS - 1) // PAGE_FULL_SIZE_TOKENS
    if state.pool.free_zeroed < pages_needed:
        # Real fallback: block for zeroing or fail. Model as small wait.
        sess.admitted_at = state.bridge_busy_until + 0.001
        return
    alloc_s = pages_needed * ALLOC_US * 1e-6 * 0.5    # cheaper, no scan
    bt_s = BLOCK_TABLE_US * 1e-6
    end_t = acquire_lock(state, sess.sid,
                          f"POOL pop {pages_needed}p",
                          alloc_s + bt_s)
    state.pool.free_zeroed -= pages_needed
    state.pool.allocated += pages_needed
    sess.allocated_pages = pages_needed
    sess.admitted_at = end_t
    prefill_tiles = (sess.prefill_tokens + 255) // 256
    prefill_s = prefill_tiles * PREFILL_TILE_MS * 1e-3
    prefill_end = acquire_lock(state, sess.sid,
                                 f"prefill {sess.prefill_tokens}t",
                                 prefill_s)
    sess.ready_at = prefill_end


STRATEGIES = {
    "S1_pre_reserve_sync":  strategy_pre_reserve_sync,
    "S2_lazy_grow_sync":    strategy_lazy_grow_sync,
    "S3_lazy_grow_async":   strategy_lazy_grow_async,
    "S4_pre_zeroed_pool":   strategy_pre_zeroed_pool,
}


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────


def run(workload: list, strategy_name: str) -> dict:
    """Run a workload through a strategy. Returns a results dict."""
    state = SimState()
    state.workload = sorted(workload, key=lambda w: w[0])  # by admission_time
    strategy = STRATEGIES[strategy_name]

    for i, (admission_time, max_new_tokens, prefill_tokens) in enumerate(state.workload):
        sess = Session(sid=i,
                        admission_time=admission_time,
                        max_new_tokens=max_new_tokens,
                        prefill_tokens=prefill_tokens)
        state.sessions[i] = sess
        # Time advances: bridge becomes available at
        # max(admission_time, bridge_busy_until)
        state.now = max(state.now, admission_time)
        strategy(state, sess)

    # Compute results.
    admit_latencies = [(s.admitted_at - s.admission_time)
                        for s in state.sessions.values()
                        if s.admitted_at is not None]
    ready_latencies = [(s.ready_at - s.admission_time)
                        for s in state.sessions.values()
                        if s.ready_at is not None]

    def _pct(xs: list, p: float) -> float:
        if not xs: return float("nan")
        s = sorted(xs)
        return s[int(len(s) * p)]

    burst_wall = state.bridge_busy_until - state.workload[0][0] if state.workload else 0
    return {
        "strategy": strategy_name,
        "n_sessions": len(state.sessions),
        "n_admitted": sum(1 for s in state.sessions.values()
                            if s.admitted_at is not None),
        "burst_wall_s": burst_wall,
        "max_lock_hold_ms": state.lock.longest_hold * 1000,
        "max_lock_hold_op": state.lock.longest_hold_op,
        "total_lock_held_s": state.lock.total_held_time,
        "lock_held_fraction": (state.lock.total_held_time / max(burst_wall, 1e-9)
                                if burst_wall > 0 else float("nan")),
        "admit_lat_p50_ms":  _pct(admit_latencies, 0.5)*1000 if admit_latencies else 0,
        "admit_lat_p90_ms":  _pct(admit_latencies, 0.9)*1000 if admit_latencies else 0,
        "admit_lat_p99_ms":  _pct(admit_latencies, 0.99)*1000 if admit_latencies else 0,
        "admit_lat_max_ms":  max(admit_latencies)*1000 if admit_latencies else 0,
        "ready_lat_p50_ms":  _pct(ready_latencies, 0.5)*1000 if ready_latencies else 0,
        "ready_lat_max_ms":  max(ready_latencies)*1000 if ready_latencies else 0,
        "pool_free_zeroed_end": state.pool.free_zeroed,
        "pool_allocated_end":   state.pool.allocated,
    }


def fmt_result(r: dict) -> str:
    return (f"  {r['strategy']:<26}  "
            f"burst={r['burst_wall_s']*1000:>7.0f}ms  "
            f"max_lock={r['max_lock_hold_ms']:>7.0f}ms  "
            f"locked%={r['lock_held_fraction']*100:>5.1f}  "
            f"admit_p50={r['admit_lat_p50_ms']:>6.0f}ms  "
            f"p90={r['admit_lat_p90_ms']:>6.0f}ms  "
            f"max={r['admit_lat_max_ms']:>6.0f}ms  "
            f"ready_max={r['ready_lat_max_ms']:>6.0f}ms")


# ──────────────────────────────────────────────────────────────────────
# Workload presets
# ──────────────────────────────────────────────────────────────────────


def workload_burst_8_at_zero() -> list:
    """8 sessions all submitted at t=0 (admission burst), max_tokens=8192,
    realistic prefill mix (eval+judge bench mix)."""
    out = []
    for i in range(8):
        # mix: eval prefill ~80, judge prefill ~600 — average ~340
        prefill = 80 if i % 2 == 0 else 600
        out.append((0.0, 8192, prefill))
    return out


def workload_burst_8_then_4() -> list:
    """8 at t=0, 4 more at t=0.5s — represents the C=12 client semaphore."""
    out = workload_burst_8_at_zero()
    for i in range(4):
        out.append((0.5, 8192, 340))
    return out


def workload_realistic_probe2_first_minute() -> list:
    """First minute of a probe2-shape run: 30 admissions over 60 seconds,
    bursts at the top, trickle as evals complete."""
    out = []
    # Initial admission burst of 8
    for i in range(8):
        out.append((0.0, 8192, 80 if i % 2 == 0 else 600))
    # Trickle over the next 60s as rollouts complete and new ones admit
    import random
    rng = random.Random(42)
    t = 5.0
    for _ in range(22):
        out.append((t, 8192, 80 if rng.random() < 0.5 else 600))
        t += rng.uniform(2.0, 4.0)
    return out


def workload_short_prompts_short_max() -> list:
    """Imagined workload with sane max_tokens=512 (instead of 8192).
    Demonstrates how much pathology comes from over-reserving."""
    return [(0.0, 512, 80 if i % 2 == 0 else 600) for i in range(8)]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    print("# KV-cache memory-management strategy simulator")
    print(f"# config: pool={NUM_PHYS_PAGES}p, page=8tok×800KB, "
          f"memset_BW={MEMSET_BANDWIDTH_GBPS}GB/s, "
          f"AR_step={AR_STEP_MS}ms, prefill_tile={PREFILL_TILE_MS}ms")
    print()

    workloads = [
        ("8 admissions @ t=0, max_tokens=8192 (admission burst)",
         workload_burst_8_at_zero()),
        ("8 then 4 at t=0.5s, max_tokens=8192 (probe-shape C=12)",
         workload_burst_8_then_4()),
        ("first 60s of probe2-shape (30 admissions, max_tokens=8192)",
         workload_realistic_probe2_first_minute()),
        ("8 admissions, sane max_tokens=512 (counterfactual)",
         workload_short_prompts_short_max()),
    ]

    for label, w in workloads:
        print(f"=== {label} ===")
        results = []
        for sname in STRATEGIES:
            r = run(list(w), sname)
            results.append(r)
            print(fmt_result(r))
        # Speedup vs S1
        s1 = results[0]
        best = min(results, key=lambda r: r["burst_wall_s"])
        if best["strategy"] != s1["strategy"]:
            speedup = s1["burst_wall_s"] / best["burst_wall_s"]
            lock_reduction = s1["max_lock_hold_ms"] / max(best["max_lock_hold_ms"], 0.001)
            print(f"\n  → best ({best['strategy']}) is {speedup:.2f}x faster on burst wall, "
                  f"max-lock-hold {lock_reduction:.0f}x shorter")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
