"""Bridge AR-decode scheduler simulator.

Models the engine's per-tick decision: which subset of resident streams
get the 8 active slots this tick. Compares different policies against
the same workload, measures wall-time-to-completion.

Cost model (memory-bandwidth-bound batched decode on M5 Max):
  - Each AR tick loads the full active-experts weight set ONCE.
  - The b1, b2, b4, b8 kernel variants amortize that weight-load
    across however many lanes are populated.
  - Per-tick cost is dominated by weight-bandwidth, NOT by per-lane
    matmul work — so a b8 tick costs only marginally more than a b1
    tick, and the per-stream throughput rises ~linearly with active
    lanes.

Empirical anchors (calibrated to the observed 2.8 tok/s/stream at C=12
with sticky admission):
  - b8 tick wall:  ~10 ms  →  per-stream throughput at b8 = 100 tok/s
  - b4 tick wall:  ~9  ms  →  per-stream throughput at b4 = 56 tok/s
  - b2 tick wall:  ~8  ms  →  per-stream throughput at b2 = 31 tok/s
  - b1 tick wall:  ~7  ms  →  per-stream throughput at b1 = 17 tok/s

(Numbers are illustrative orders-of-magnitude; the simulation logic
holds for any concrete cost-vs-bin-size curve. Override `BIN_COSTS_MS`
to recalibrate.)

Usage::

    python tools/quant_search/sched_sim.py
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable

# ──────────────────────────────────────────────────────────────────────
# Cost model
# ──────────────────────────────────────────────────────────────────────

# Per-tick wall in milliseconds for each kernel batch size.
# Dominated by weight-bandwidth: per-tick floor ~7 ms (one full weight
# load), plus a small per-lane increment from extra activations.
BIN_COSTS_MS: dict[int, float] = {
    1: 7.0,
    2: 8.0,
    4: 9.0,
    8: 10.0,
}

# Slot count. Kernel zoo ceiling.
B = 8


def round_up_bin(n: int) -> int:
    """Active-stream count → next-power-of-two kernel bin (capped at B)."""
    if n <= 0: return 0
    if n <= 1: return 1
    if n <= 2: return 2
    if n <= 4: return 4
    return 8


def tick_cost_ms(active_count: int) -> float:
    return BIN_COSTS_MS[round_up_bin(active_count)]


# ──────────────────────────────────────────────────────────────────────
# Workload model
# ──────────────────────────────────────────────────────────────────────


@dataclass
class Stream:
    sid: int
    tokens_remaining: int
    tokens_emitted: int = 0
    finish_tick: int | None = None
    finish_wall_ms: float | None = None


# ──────────────────────────────────────────────────────────────────────
# Schedulers — pick which subset of resident streams gets the 8 slots
# this tick. Each scheduler takes the full resident pool plus per-stream
# state and returns the (up-to-8) sids to decode this tick.
# ──────────────────────────────────────────────────────────────────────


def sched_sticky_fifo(resident: list[Stream], slot_owners: list[int | None]) -> list[int]:
    """The current bridge behaviour. Slots are owned long-term: a stream
    holds its slot until completion. New streams only enter when an
    existing slot owner finishes."""
    # Already-slotted streams hold their slot.
    owned_sids = [s for s in slot_owners if s is not None]
    # Free slots (None entries) admit unslotted residents in arrival order.
    unslotted = [s.sid for s in resident
                  if s.sid not in set(owned_sids) and s.tokens_remaining > 0]
    free_count = sum(1 for s in slot_owners if s is None)
    new_admits = unslotted[:free_count]
    return owned_sids + new_admits


def sched_round_robin(resident: list[Stream], slot_owners: list[int | None],
                       _state: dict) -> list[int]:
    """Per-tick rotation: every active stream gets equal slot time over
    the long run. With N resident and B slots, each tick picks the B
    streams whose `tokens_emitted` is currently smallest (i.e. the
    ones that have had the fewest previous slot-ticks). Ties broken
    by sid for determinism."""
    active = [s for s in resident if s.tokens_remaining > 0]
    if not active: return []
    active.sort(key=lambda s: (s.tokens_emitted, s.sid))
    return [s.sid for s in active[:B]]


def sched_sjf(resident: list[Stream], slot_owners: list[int | None],
               _state: dict) -> list[int]:
    """Shortest-job-first: prefer streams with fewest tokens remaining.
    Drains short jobs to completion fast, then concentrates on the
    long ones."""
    active = [s for s in resident if s.tokens_remaining > 0]
    active.sort(key=lambda s: (s.tokens_remaining, s.sid))
    return [s.sid for s in active[:B]]


def sched_ljf(resident: list[Stream], slot_owners: list[int | None],
               _state: dict) -> list[int]:
    """Longest-job-first: classic makespan-minimization heuristic. Start
    long jobs early so the tail (when only a few streams remain) is
    populated by short ones that finish quickly."""
    active = [s for s in resident if s.tokens_remaining > 0]
    active.sort(key=lambda s: (-s.tokens_remaining, s.sid))
    return [s.sid for s in active[:B]]


# ──────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────


def simulate(workload: list[int],
              policy: str,
              max_ticks: int = 1_000_000) -> dict:
    """Run one scheduler policy on a workload (list of per-stream
    tokens_to_emit). Returns wall_ms, ticks, per-stream finish times,
    and bin-distribution stats."""
    streams = [Stream(sid=i, tokens_remaining=tok)
                for i, tok in enumerate(workload)]
    slot_owners: list[int | None] = [None] * B
    sticky = (policy == "sticky_fifo")

    state: dict = {}
    bin_hist: dict[int, int] = {1: 0, 2: 0, 4: 0, 8: 0}
    wall_ms = 0.0
    tick = 0
    while tick < max_ticks:
        # Free slots whose owner has finished.
        for slot_idx, sid in enumerate(slot_owners):
            if sid is None: continue
            owner = streams[sid]
            if owner.tokens_remaining <= 0:
                slot_owners[slot_idx] = None

        if all(s.tokens_remaining <= 0 for s in streams):
            break

        # Pick this tick's active streams.
        if policy == "sticky_fifo":
            active_sids = sched_sticky_fifo(streams, slot_owners)
            # Update slot_owners: any newly admitted stream takes a free slot.
            already = set(s for s in slot_owners if s is not None)
            for sid in active_sids:
                if sid in already: continue
                # Find a free slot
                for slot_idx in range(B):
                    if slot_owners[slot_idx] is None:
                        slot_owners[slot_idx] = sid
                        already.add(sid)
                        break
        elif policy == "round_robin":
            active_sids = sched_round_robin(streams, slot_owners, state)
        elif policy == "sjf":
            active_sids = sched_sjf(streams, slot_owners, state)
        elif policy == "ljf":
            active_sids = sched_ljf(streams, slot_owners, state)
        else:
            raise ValueError(f"unknown policy: {policy}")

        n_active = len(active_sids)
        if n_active == 0:
            break

        # Charge tick cost.
        bin_size = round_up_bin(n_active)
        bin_hist[bin_size] += 1
        wall_ms += BIN_COSTS_MS[bin_size]

        # Emit one token per active stream.
        for sid in active_sids:
            s = streams[sid]
            if s.tokens_remaining > 0:
                s.tokens_remaining -= 1
                s.tokens_emitted += 1
                if s.tokens_remaining <= 0:
                    s.finish_tick = tick
                    s.finish_wall_ms = wall_ms

        tick += 1

    finish_times = [s.finish_wall_ms for s in streams
                     if s.finish_wall_ms is not None]
    return {
        "policy": policy,
        "wall_ms": wall_ms,
        "wall_s": wall_ms / 1000,
        "ticks": tick,
        "completed_streams": sum(1 for s in streams if s.tokens_remaining <= 0),
        "total_streams": len(streams),
        "bin_dist": bin_hist,
        "median_finish_ms": statistics.median(finish_times) if finish_times else 0,
        "p10_finish_ms": sorted(finish_times)[len(finish_times)//10] if finish_times else 0,
        "p90_finish_ms": sorted(finish_times)[len(finish_times)*9//10] if finish_times else 0,
        "max_finish_ms": max(finish_times) if finish_times else 0,
        "agg_tok_per_s": sum(workload) / max(wall_ms/1000, 1e-9),
    }


# ──────────────────────────────────────────────────────────────────────
# Workload presets
# ──────────────────────────────────────────────────────────────────────


def workload_uniform_12x1000() -> list[int]:
    """Codex's example: 12 streams each generating 1000 tokens.
    Demonstrates sticky-FIFO's bin-degradation tail."""
    return [1000] * 12


def workload_realistic_probe2() -> list[int]:
    """Approximate the probe2 fp16 workload: 6 benchmarks × ~40 records
    each = 240 rollouts, mixed token counts.

    Per-rollout = eval call + judge call. Token totals derived from
    the observed bridge log (probe2 fp16 bridge):
      eval_tokens (decode):
        triviaqa  ~140  hellaswag ~210  algebra  ~280  humaneval ~230
        mmlu      ~430  svg       ~550 (per iter, 3 iters per rollout)
      judge_tokens (decode): ~50 each

    For a 12-stream concurrency cap workload, we'd present ~240 rollouts
    × 2 calls/rollout = 480 streams to the scheduler over the run.
    Simplification: model each "stream" as a single AR pipeline with
    decode-tokens = sum of eval+judge.
    """
    # 240 rollouts, mixed by benchmark proportion (40 each)
    n_per = 40
    bench_decode_tokens = {
        "triviaqa":  140 + 50,
        "hellaswag": 210 + 50,
        "humaneval": 230 + 50,
        "algebra":   280 + 50,
        "mmlu":      430 + 50,
        "svg":       550 * 3 + 0,    # 3 iters, no judge
    }
    out = []
    for tk in bench_decode_tokens.values():
        out.extend([tk] * n_per)
    return out


def workload_mixed_short_long(n_short: int = 60, n_long: int = 60,
                                  short_tok: int = 100, long_tok: int = 2000) -> list[int]:
    return [short_tok] * n_short + [long_tok] * n_long


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────


def fmt_result(r: dict) -> str:
    bd = r["bin_dist"]
    total_ticks = sum(bd.values()) or 1
    bin_pcts = {b: 100*c/total_ticks for b, c in bd.items()}
    return (f"  {r['policy']:<14}  wall={r['wall_s']:>6.1f}s  "
            f"ticks={r['ticks']:>6}  "
            f"agg={r['agg_tok_per_s']:>6.0f}tok/s  "
            f"bin% b1={bin_pcts[1]:>4.0f} b2={bin_pcts[2]:>4.0f} "
            f"b4={bin_pcts[4]:>4.0f} b8={bin_pcts[8]:>4.0f}  "
            f"finish_med={r['median_finish_ms']/1000:>5.1f}s "
            f"p90={r['p90_finish_ms']/1000:>5.1f}s "
            f"max={r['max_finish_ms']/1000:>5.1f}s")


def compare(workload: list[int], label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"  workload: {len(workload)} streams, "
          f"total tokens={sum(workload):,}, "
          f"per-stream median={statistics.median(workload):.0f}, "
          f"p90={sorted(workload)[len(workload)*9//10]:.0f}")
    results = []
    for policy in ["sticky_fifo", "round_robin", "sjf", "ljf"]:
        r = simulate(workload, policy)
        results.append(r)
        print(fmt_result(r))
    # Speedup of best vs sticky_fifo
    sticky = next(r for r in results if r["policy"] == "sticky_fifo")
    best = min(results, key=lambda r: r["wall_ms"])
    if best["policy"] != "sticky_fifo":
        print(f"\n  → best policy ({best['policy']}) is "
              f"{sticky['wall_ms']/best['wall_ms']:.2f}x faster than "
              f"sticky_fifo")


def main() -> int:
    print("# Bridge scheduler bin-packing simulator")
    print(f"# B={B} (kernel-zoo ceiling)")
    print(f"# bin costs: {BIN_COSTS_MS}")

    compare(workload_uniform_12x1000(),
             "Codex example: 12 streams × 1000 tokens (uniform)")
    compare(workload_realistic_probe2(),
             "probe2-shape workload: 240 rollouts, mixed by benchmark")
    compare(workload_mixed_short_long(n_short=60, n_long=60),
             "Mixed: 60 short (100 tok) + 60 long (2000 tok)")
    compare(workload_mixed_short_long(n_short=120, n_long=12,
                                          short_tok=50, long_tok=4000),
             "Heavy-tail: 120 judge-shape (50) + 12 svg-iter-shape (4000)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
