#!/usr/bin/env python3
"""Engine bandwidth heartbeat — poll /health (+ /v1/engine/state) and log a
time-series of the three tensor-backend work rates plus occupancy.

What /health exposes cumulatively lets us derive, per interval:
  - DECODE   : tok/s  = d(total_tokens_emitted)/dt           (AR decode bandwidth)
  - VISION   : img/s  = d(vision_cache.misses)/dt   (each miss = one 27-layer
               encode ~280 soft tokens; soft_tok/s = img/s * 280)
  - STEPS    : steps/s = d(total_steps)/dt
plus active_stream_count (decode occupancy; the multimodal-stall tell when it
sticks at 0), last_step_ms, free KV pages, and the engine's own aggregate tok/s.
PREFILL tok/s is not cumulatively exposed by /health (would need a counter); we
surface steps/s + last_step_ms as the available prefill proxies.

Writes one JSON line per interval to --out and prints a live table. Runs until
killed (Ctrl-C / SIGTERM) or --duration seconds.
"""
from __future__ import annotations
import argparse, json, time, urllib.request

SOFT_PER_IMG = 280


def _get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--out", default=None, help="jsonl time-series output")
    ap.add_argument("--duration", type=float, default=0, help="0 = until killed")
    args = ap.parse_args()
    f = open(args.out, "w") if args.out else None

    prev = None
    t0 = time.time()
    print(f"{'t':>6} {'act':>3} {'decode_t/s':>10} {'vis_img/s':>9} {'vis_tok/s':>9} "
          f"{'steps/s':>7} {'last_ms':>8} {'freePg':>7} {'agg_t/s':>8}", flush=True)
    try:
        while True:
            if args.duration and time.time() - t0 > args.duration:
                break
            try:
                h = _get(args.base + "/health")
                es = _get(args.base + "/v1/engine/state")
            except Exception:
                time.sleep(args.interval)
                continue
            now = time.time()
            vmiss = (es.get("vision_cache") or {}).get("misses", 0)
            last_ms = (es.get("engine") or {}).get("last_step_ms", 0.0)
            cur = {"t": round(now - t0, 1), "active": h["active_stream_count"],
                   "total_tokens": h["total_tokens_emitted"], "total_steps": h["total_steps"],
                   "vis_miss": vmiss, "last_step_ms": round(last_ms, 1),
                   "free_pages": h["free_pages"], "agg_tok_s": round(h["aggregate_tok_per_sec"]),
                   "_t": now}
            if prev:
                dt = now - prev["_t"]
                dec = (cur["total_tokens"] - prev["total_tokens"]) / dt
                vimg = (cur["vis_miss"] - prev["vis_miss"]) / dt
                steps = (cur["total_steps"] - prev["total_steps"]) / dt
                cur["decode_tok_s"] = round(dec)
                cur["vision_img_s"] = round(vimg, 2)
                cur["vision_tok_s"] = round(vimg * SOFT_PER_IMG)
                cur["steps_s"] = round(steps)
                print(f"{cur['t']:>6.0f} {cur['active']:>3} {dec:>10.0f} {vimg:>9.2f} "
                      f"{vimg*SOFT_PER_IMG:>9.0f} {steps:>7.0f} {cur['last_step_ms']:>8.0f} "
                      f"{cur['free_pages']:>7} {cur['agg_tok_s']:>8}", flush=True)
                if f:
                    out = {k: v for k, v in cur.items() if k != "_t"}
                    f.write(json.dumps(out) + "\n"); f.flush()
            prev = cur
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if f:
            f.close()


if __name__ == "__main__":
    main()
