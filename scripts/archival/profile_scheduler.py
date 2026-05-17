#!/usr/bin/env python3
"""Poll /v1/kv/snapshot at 500 ms, track scheduler/throughput stats
while a workload (e.g. concurrent multi-agent trials) runs.

Outputs per-second:
  - number of resident sessions in each state
  - total session-token-positions advanced this sec (= aggregate
    tokens produced across all sessions — prefill + decode counted
    equally; good proxy for engine bandwidth)
  - any session id appearing / disappearing

Writes raw samples to --out as JSONL so we can analyse cadence
post-hoc.
"""
import argparse, json, time, urllib.request
from collections import defaultdict


def snapshot():
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/kv/snapshot",
                                  timeout=5) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notes/runs/profile.jsonl")
    ap.add_argument("--duration-s", type=float, default=60.0)
    ap.add_argument("--poll-interval-s", type=float, default=0.5)
    args = ap.parse_args()

    prev_positions: dict[int, int] = {}
    prev_sids: set[int] = set()
    rolling = defaultdict(lambda: {"secs": 0, "tokens": 0})

    t_end = time.time() + args.duration_s
    last_report = time.time()
    bucket_tokens = 0
    bucket_secs = 0
    state_hist_bucket = defaultdict(int)

    with open(args.out, "w") as fout:
        while time.time() < t_end:
            t = time.time()
            try:
                s = snapshot()
            except Exception as e:
                print(f"snapshot err: {e}"); time.sleep(args.poll_interval_s); continue
            # Count new position deltas
            new_pos = {row["sid"]: row["position"] for row in s["sessions"]}
            state_by_sid = {row["sid"]: row["state"] for row in s["sessions"]}
            delta = 0
            for sid, p in new_pos.items():
                prev = prev_positions.get(sid, p)
                if p >= prev:
                    delta += p - prev
            prev_positions = new_pos

            # State distribution this tick
            tick_states = defaultdict(int)
            for sid, st in state_by_sid.items():
                tick_states[st] += 1

            # Session arrivals/departures
            now_sids = set(new_pos)
            arrived = now_sids - prev_sids
            departed = prev_sids - now_sids
            prev_sids = now_sids

            # Write record
            fout.write(json.dumps({
                "ts": t, "tokens_delta": delta,
                "n_sessions": len(new_pos),
                "states": dict(tick_states),
                "arrived": list(arrived), "departed": list(departed),
                "per_sid_position": new_pos,
            }) + "\n")
            fout.flush()

            bucket_tokens += delta
            bucket_secs += args.poll_interval_s
            for st, n in tick_states.items():
                state_hist_bucket[st] += n

            # Report every 2 seconds
            if time.time() - last_report >= 2.0:
                n_ticks = bucket_secs / args.poll_interval_s
                tps = bucket_tokens / bucket_secs if bucket_secs > 0 else 0
                avg_gen = state_hist_bucket.get("generating", 0) / max(n_ticks, 1)
                avg_pri = state_hist_bucket.get("priming", 0) / max(n_ticks, 1)
                avg_oth = (sum(state_hist_bucket.values())
                            - state_hist_bucket.get("generating", 0)
                            - state_hist_bucket.get("priming", 0)) / max(n_ticks, 1)
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"tok/s={tps:>6.0f}  "
                      f"avg resident: gen={avg_gen:.2f}, "
                      f"prime={avg_pri:.2f}, other={avg_oth:.2f}")
                last_report = time.time()
                bucket_tokens = 0; bucket_secs = 0
                state_hist_bucket = defaultdict(int)

            time.sleep(max(0, args.poll_interval_s - (time.time() - t)))


if __name__ == "__main__":
    main()
