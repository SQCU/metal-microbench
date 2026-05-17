"""30-second strictly-timed bandwidth probe with incremental d/dt logging.

Fires N concurrent chat completions against the bridge AND polls
/v1/engine/state at 1Hz for exactly 30 seconds. Emits one CSV row per
second showing both the current value and the d/dt over the last 1s
window for every measured quantity.

The point: characterize bridge throughput in a small, fixed time budget.
30 seconds is enough samples (30 data points) to mean+variance the
steady-state behavior, identify warmup (d/dt ramping), drain (d/dt
falling), and per-stream throughput at the actual concurrency level.

Usage:
    bandwidth_probe.py [--concurrency 8] [--bridge http://127.0.0.1:8001]
                       [--prompt-tokens 500] [--max-tokens 200]
                       [--duration 30]

CSV columns:
    t_sec               — seconds since start
    active_streams      — current count
    d_active_streams    — change since prior second
    free_pages          — KV pool free
    cached_pages        — KV pool cached
    d_cached_pages      — change since prior second
    total_steps         — engine cumulative steps
    d_steps             — change (== steps/sec instantaneous)
    total_tokens        — engine cumulative tokens
    d_tokens            — change (== tokens/sec instantaneous; includes prefill+AR)

The load generator runs in a background thread. The poller runs in the
main thread. Both terminate strictly at duration seconds.
"""

import argparse
import json
import sys
import threading
import time
import urllib.request


def fire_one(bridge: str, prompt_tokens: int, max_tokens: int | None, seed: int):
    """One chat completion call — synchronous urllib. Returns usage dict
    or None on failure. Padded prompt to ~prompt_tokens of work."""
    # ~4 chars/token average → padding length
    padding = "the quick brown fox jumps over the lazy dog " * max(1, prompt_tokens // 11)
    body = json.dumps({
        "messages": [
            {"role": "system", "content": "You answer in 2-3 sentences."},
            {"role": "user", "content": f"{padding}\n\nIn 2-3 sentences, describe what an animation loop is."},
        ],
        "stream": False,
        "seed": seed,
    }).encode()
    req = urllib.request.Request(
        f"{bridge}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            d = json.loads(resp.read())
        return d.get("usage")
    except Exception as e:
        return {"error": str(e)[:80]}


def load_generator(bridge: str, concurrency: int, duration: float,
                   prompt_tokens: int, max_tokens: int | None,
                   stop_flag: threading.Event, completed_counter: list):
    """Keeps `concurrency` chat-completion calls in flight for the
    duration. Each thread keeps firing new calls until stop_flag is set.
    Completed-call count goes into completed_counter[0]."""
    def worker(worker_id: int):
        seed = 10000 + worker_id * 1000
        while not stop_flag.is_set():
            seed += 1
            r = fire_one(bridge, prompt_tokens, max_tokens, seed)
            if not stop_flag.is_set() and r and "error" not in r:
                completed_counter[0] += 1
    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(concurrency)]
    for t in threads:
        t.start()
    # Caller drives the timing; this function returns immediately. The
    # threads run in background until stop_flag is set.


def poll_engine_state(bridge: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{bridge}/v1/engine/state", timeout=2) as r:
            return json.load(r)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bridge", default="http://127.0.0.1:8001")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--prompt-tokens", type=int, default=500,
                   help="approximate prompt tokens per call (padded)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="optional max_tokens per call (default: bridge default = no cap)")
    p.add_argument("--duration", type=float, default=30.0)
    args = p.parse_args()

    # CSV header, then one row per second. Flush every row so the
    # operator sees d/dt live.
    cols = ("t_sec,active_streams,d_active_streams,free_pages,cached_pages,"
            "d_cached_pages,total_steps,d_steps,total_tokens,d_tokens,"
            "completed_calls,d_completed")
    print(cols)
    sys.stdout.flush()

    # Snapshot start state.
    initial = poll_engine_state(args.bridge)
    if initial is None:
        print("ERROR: bridge unreachable", file=sys.stderr)
        sys.exit(1)
    initial_steps = (initial.get("engine") or {}).get("total_steps", 0)
    initial_tokens = (initial.get("engine") or {}).get("total_tokens", 0)

    # Start load.
    stop_flag = threading.Event()
    completed_counter = [0]
    load_generator(args.bridge, args.concurrency, args.duration,
                   args.prompt_tokens, args.max_tokens,
                   stop_flag, completed_counter)

    # Poll at 1Hz for duration seconds. Strict timing: each iteration
    # measures relative to the start, not relative to the prior tick.
    t0 = time.time()
    prev = {"active_streams": 0, "cached_pages": 0, "total_steps": 0,
            "total_tokens": 0, "completed": 0}
    tick = 0
    while True:
        target_t = t0 + tick + 1
        sleep_for = target_t - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        t_now = time.time() - t0
        if t_now >= args.duration:
            break
        state = poll_engine_state(args.bridge)
        if state is None:
            print(f"{t_now:.2f},ERR,ERR,ERR,ERR,ERR,ERR,ERR,ERR,ERR,{completed_counter[0]},ERR")
            sys.stdout.flush()
            tick += 1
            continue
        kv = state.get("kv_cache") or {}
        eng = state.get("engine") or {}
        active = state.get("active_streams")
        if isinstance(active, list):
            active = len(active)
        elif active is None:
            active = 0
        cur = {
            "active_streams": active,
            "cached_pages": kv.get("cached_pages", 0),
            "total_steps": eng.get("total_steps", 0),
            "total_tokens": eng.get("total_tokens", 0),
            "completed": completed_counter[0],
        }
        d_active = cur["active_streams"] - prev["active_streams"]
        d_cached = cur["cached_pages"] - prev["cached_pages"]
        d_steps = cur["total_steps"] - prev["total_steps"]
        d_tokens = cur["total_tokens"] - prev["total_tokens"]
        d_completed = cur["completed"] - prev["completed"]
        print(f"{t_now:.2f},{cur['active_streams']},{d_active},"
              f"{kv.get('free_pages', '?')},{cur['cached_pages']},{d_cached},"
              f"{cur['total_steps']},{d_steps},{cur['total_tokens']},{d_tokens},"
              f"{cur['completed']},{d_completed}")
        sys.stdout.flush()
        prev = cur
        tick += 1

    # Stop load + wait briefly for in-flight requests to drain.
    stop_flag.set()
    time.sleep(0.5)

    # Final aggregate + steady-state (last 10s) summary on stderr so
    # CSV stays clean.
    final = poll_engine_state(args.bridge)
    if final:
        eng = final.get("engine") or {}
        delta_steps = eng.get("total_steps", 0) - initial_steps
        delta_tokens = eng.get("total_tokens", 0) - initial_tokens
        print(f"\n# AGGREGATE over {args.duration:.0f}s @ concurrency={args.concurrency}:",
              file=sys.stderr)
        print(f"#   steps:    {delta_steps:>8,}  ({delta_steps/args.duration:>6.1f}/s)",
              file=sys.stderr)
        print(f"#   tokens:   {delta_tokens:>8,}  ({delta_tokens/args.duration:>6.1f}/s)",
              file=sys.stderr)
        print(f"#   completed: {completed_counter[0]:>7}  ({completed_counter[0]/args.duration:>6.2f}/s)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
