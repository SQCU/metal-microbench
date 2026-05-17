#!/usr/bin/env python3
"""Pipeline probe — fire K concurrent minimal streaming requests, measure
TTFT spread + observed parallelism + engine-side slot occupancy to
separate pipeline starvation from pipeline bubbles.

Runs in ≤10s wall on a warm server.

Three headline metrics:

  1. TTFT spread = max(ttft) - min(ttft).
     Small → prefill batches across slots; large → prefill staircases.
  2. Observed parallelism = sum(total_s) / max(total_s) ∈ [1, K].
     K → fully batched; 1 → fully serial.
     Starvation fraction = (K - observed) / (K - 1).
  3. Mean slotted slots during the probe window (scheduler poller).
     Should approach K if scheduler filled all slots.

Plus aggregated prefill/decode tok/s for reference.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, pathlib, sys, threading, time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svg_refinement_loop import chat_stream

BASE = "http://127.0.0.1:8000"


class SchedulerPoller:
    """Polls /api/extra/scheduler and /api/extra/perf every interval_s."""

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _fetch(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(BASE + path, timeout=2) as r:
                return json.load(r)
        except Exception:
            return {}

    def _run(self):
        while not self._stop.is_set():
            t = time.time()
            s = self._fetch("/api/extra/scheduler")
            p = self._fetch("/api/extra/perf")
            counts = s.get("counts") or {}
            activity = s.get("activity") or {}
            self.samples.append({
                "ts": t,
                "slotted":  counts.get("slotted", 0),
                "resident": counts.get("resident", 0),
                "wanting":  counts.get("wanting_slot_not_assigned", 0),
                "paused":   counts.get("paused", 0),
                "avg_active_since_start": activity.get("avg_active_slots_since_start", 0),
                "last_step_ms": activity.get("last_step_ms", 0),
                "total_completion_tokens": p.get("total_completion_tokens", 0),
                "uptime":   p.get("uptime_seconds", 0),
                "tps":      p.get("tokens_per_second", 0),
            })
            self._stop.wait(self.interval_s)

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._t: self._t.join(timeout=3)
        return self.samples


def one_request(idx: int, prompt: str, max_tokens: int,
                 temperature: float, seed: int) -> dict:
    messages = [{"role": "user", "content": prompt}]
    text, m = chat_stream(messages, max_tokens=max_tokens,
                           temperature=temperature, seed=seed)
    m["idx"] = idx
    m["text_len"] = len(text)
    return m


def fmt_or_na(x, spec=".2f"):
    if x is None: return "n/a"
    return format(x, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4,
                    help="concurrent clients (minimum 4)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--prompt", default="count: one two three four",
                    help="short text prompt (kept small to isolate prefill)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--warmup", action="store_true",
                    help="fire a single request first to warm caches")
    ap.add_argument("--save", type=pathlib.Path, default=None,
                    help="write raw probe data to this JSON path")
    args = ap.parse_args()

    if args.K < 4:
        print("K<4 rejected (natural concurrency is ≥4)", file=sys.stderr)
        sys.exit(2)

    if args.warmup:
        print("[probe] warmup …", file=sys.stderr)
        one_request(-1, args.prompt, max_tokens=4,
                     temperature=args.temperature, seed=0)

    poller = SchedulerPoller(interval_s=0.05); poller.start()
    time.sleep(0.2)  # baseline samples before launch
    t_pre = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.K) as exe:
        futs = [exe.submit(one_request, i, args.prompt, args.max_tokens,
                            args.temperature, 1000 + i)
                for i in range(args.K)]
        # Results sorted by completion order; we'll re-sort by submit below.
        results = [f.result() for f in futs]
    t_post = time.time()
    time.sleep(0.2)  # tail samples
    samples = poller.stop()

    wall = t_post - t_pre
    ttfts  = [r["ttft_s"]  for r in results]
    totals = [r["total_s"] for r in results]
    decs   = [r["decode_s"] for r in results]
    t_first_abs = [r["ts_submit"] + r["ttft_s"] for r in results]

    ttft_spread = max(ttfts) - min(ttfts)
    observed_par = sum(totals) / max(totals) if max(totals) > 0 else 0.0
    starvation_frac = ((args.K - observed_par) / (args.K - 1)
                        if args.K > 1 else 0.0)

    first_sorted = sorted(t_first_abs)
    gaps = [first_sorted[i+1] - first_sorted[i]
            for i in range(len(first_sorted) - 1)]

    window = [s for s in samples if t_pre <= s["ts"] <= t_post]
    if window:
        slotted_vals = [s["slotted"] for s in window]
        wanting_vals = [s["wanting"] for s in window]
        mean_slotted = sum(slotted_vals) / len(slotted_vals)
        max_slotted = max(slotted_vals)
        mean_wanting = sum(wanting_vals) / len(wanting_vals)
    else:
        mean_slotted = max_slotted = mean_wanting = None

    # engine-side window averages from since-start counters
    window_avg_active = window_tps = window_ctok = None
    if len(samples) >= 2:
        s0, s1 = samples[0], samples[-1]
        dt = s1["uptime"] - s0["uptime"]
        if dt > 0:
            window_avg_active = (
                (s1["avg_active_since_start"] * s1["uptime"]
                 - s0["avg_active_since_start"] * s0["uptime"]) / dt)
            window_ctok = s1["total_completion_tokens"] - s0["total_completion_tokens"]
            window_tps = window_ctok / dt

    sum_pt = sum((r["prompt_tokens"] or 0) for r in results)
    sum_ct = sum((r["completion_tokens"] or 0) for r in results)
    agg_prefill_bw = sum_pt / wall if wall > 0 else 0
    agg_decode_bw  = sum_ct / wall if wall > 0 else 0

    print(f"\n=== pipeline probe (K={args.K}) ===")
    print(f"wall: {wall*1000:.0f}ms  prompt_tokens_each≈{results[0]['prompt_tokens']}  "
          f"max_tokens={args.max_tokens}")

    print(f"\nper-request (sorted by submit order):")
    for r in sorted(results, key=lambda r: r["ts_submit"]):
        print(f"  req{r['idx']}: ttft={r['ttft_s']*1000:>6.0f}ms  "
              f"decode={r['decode_s']*1000:>6.0f}ms  "
              f"total={r['total_s']*1000:>6.0f}ms  "
              f"pt={r['prompt_tokens']} ct={r['completion_tokens']} "
              f"fin={r.get('finish_reason')}")

    print(f"\nheadline:")
    print(f"  1. TTFT spread: {ttft_spread*1000:.0f}ms  "
          f"(min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms)")
    print(f"     first-tok gaps (sorted): "
          + "  ".join(f"{g*1000:.0f}ms" for g in gaps))
    print(f"  2. Observed parallelism: {observed_par:.2f} / {args.K}  "
          f"(starvation fraction: {starvation_frac*100:.0f}%)")
    print(f"  3. Scheduler mean slotted (window): {fmt_or_na(mean_slotted)}  "
          f"max={max_slotted}  mean_wanting={fmt_or_na(mean_wanting)}")
    if window_avg_active is not None:
        print(f"     engine avg_active_slots during window: {window_avg_active:.2f}")
    if window_tps is not None:
        print(f"     engine tokens/sec during window: {window_tps:.1f}  "
              f"({window_ctok} tok / {s1['uptime']-s0['uptime']:.2f}s)")

    print(f"\naggregated bandwidth (sum / wall):")
    print(f"  prefill: {agg_prefill_bw:>7.1f} tok/s  "
          f"(sum prompt_tokens = {sum_pt})")
    print(f"  decode:  {agg_decode_bw:>7.1f} tok/s  "
          f"(sum completion_tokens = {sum_ct})")
    print(f"  total:   {(sum_pt + sum_ct)/wall:>7.1f} tok/s")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({
            "K": args.K,
            "wall_s": wall,
            "results": results,
            "scheduler_samples": samples,
            "window_avg_active": window_avg_active,
            "window_tps": window_tps,
            "window_ctok": window_ctok,
            "observed_parallelism": observed_par,
            "starvation_frac": starvation_frac,
            "ttft_spread_s": ttft_spread,
        }, indent=2, default=str))
        print(f"\nsaved → {args.save}")


if __name__ == "__main__":
    main()
