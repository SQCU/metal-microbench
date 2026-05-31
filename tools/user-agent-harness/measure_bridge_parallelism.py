#!/usr/bin/env python3
"""Measure: does the bridge actually parallelize concurrent requests?

Fire identical short generation requests at N=1,2,4,8 parallelism
levels. If the bridge's multi-stream / batched-AR-decode story works,
wall time at N=8 should be much less than 8× wall time at N=1.

If wall time scales linearly with N (e.g. N=4 takes 4× N=1), the
bridge is effectively serial despite supporting concurrent streams,
and something downstream of the handler is acting as a global mutex.
"""
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BRIDGE = os.environ["BRIDGE_URL"].rstrip("/")

# Identical small request so per-call work is constant.
BODY = {
    "model": "gemma-4-a4b",
    "messages": [
        {"role": "system", "content": "You are a brief, factual assistant."},
        {"role": "user", "content": "say the word 'pong' once, no other text."},
    ],
    "stream": False,
    "max_tokens": 16,
    "temperature": 0.1,
}


def one_call(idx):
    t0 = time.monotonic()
    req = urllib.request.Request(
        f"{BRIDGE}/v1/chat/completions",
        data=json.dumps(BODY).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    elapsed = time.monotonic() - t0
    content = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return idx, elapsed, len(content)


def run_at_concurrency(n):
    """Fire n identical requests at the same time. Return (wall_time, per_call_times)."""
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(one_call, i) for i in range(n)]
        results = [f.result() for f in futs]
    wall = time.monotonic() - t0
    per_call = [r[1] for r in results]
    return wall, per_call


def main():
    # Warmup so we don't measure cold-start
    print("warmup...", end=" ", flush=True)
    one_call(0)
    print("done")

    print(f"{'N':>3}  {'wall':>8}  {'avg_call':>10}  {'min_call':>10}  {'max_call':>10}  {'speedup':>9}")
    print("─" * 60)
    baseline = None
    for n in (1, 2, 4, 8):
        wall, per_call = run_at_concurrency(n)
        avg = sum(per_call) / len(per_call)
        if baseline is None:
            baseline = wall
        speedup = (baseline * n) / wall if wall > 0 else 0.0
        print(f"{n:>3}  {wall:>8.3f}  {avg:>10.3f}  {min(per_call):>10.3f}  {max(per_call):>10.3f}  {speedup:>8.2f}x")
        # n=1 wall time becomes baseline serial-equivalent; for n>1,
        # the ideal-parallel wall time would equal baseline, so
        # speedup = (n * baseline) / actual_wall.


if __name__ == "__main__":
    main()
