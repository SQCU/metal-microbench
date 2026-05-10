#!/usr/bin/env python3
"""§3 validator: oversupply continuous batching.

Submits N=12 streams in ONE batch (more than the engine's B=4 slot
budget). Engine should rotate them through slots: at any moment, up to
B streams are actively decoding; the rest sit in priming until a slot
frees. Bridge never knows B exists — it just ships everything inflight.

Two cases:
  (a) Distinct prompts: pure oversupply, no in-batch dedup. All 12
      streams should complete; status during the run shows
      generating_streams + priming_streams capped at B.
  (b) Identical prompts: oversupply meets §2 in-batch dedup. 1 leader
      should prefill cold; 11 followers each adopt the leader's pages.

Note: this test does NOT go through the HTTP bridge — it talks directly
to the unified FFI (gemma_ffi). The bridge's coordinator pattern
naturally produces the same shape; running through HTTP would just add
the asyncio.to_thread overhead without changing what we're measuring.
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemma_ffi as g  # noqa: E402

GGUF = os.environ.get(
    "GGUF_PATH",
    "/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
)
B_SLOTS = 4   # engine batch width


def collect_all(stream_ids: list[int], deadline_s: float = 120.0):
    pending = set(stream_ids)
    finals: dict[int, g.StreamUpdate] = {}
    deadline = time.time() + deadline_s
    max_active = 0
    max_generating = 0
    samples = 0
    while pending and time.time() < deadline:
        for u in g.poll(timeout_ms=200):
            if u.stream_id in pending and u.state == 2:
                finals[u.stream_id] = u
                pending.discard(u.stream_id)
        # Sample status periodically (avoid hot-loop)
        s = g.status()
        max_active = max(max_active, s.active_streams)
        max_generating = max(max_generating, s.generating_streams)
        samples += 1
    return finals, max_active, max_generating


def main() -> int:
    print("=== §3 oversupply continuous-batching ===")
    rc = g.init(GGUF)
    if rc != 0:
        print(f"  init failed: rc={rc}"); return 1

    sampling = g.SamplingParams(
        temperature=0.7, max_new_tokens=4, eos_token_id=106)

    # ---- (a) Distinct prompts: pure oversupply ----
    print("\n  [a] 12 streams, distinct prompts (pure oversupply)")
    distinct_streams = []
    for i, sid in enumerate(range(7001, 7013)):
        # Each stream's prompt has a unique seed-based head — different
        # first-page hashes, so no in-batch dedup fires.
        prompt = [2] + [(100 + sid * 19 + j * 7) % 32000 for j in range(40)]
        distinct_streams.append(g.StreamSpec(
            stream_id=sid, action=0, tokens=prompt, sampling=sampling))

    pre = g.status()
    print(f"    pre-submit: free={pre.free_pages}/{pre.total_pages} "
          f"cached={pre.cached_pages}")
    rc = g.submit(distinct_streams)
    if rc != 0:
        print(f"  FAIL submit distinct rc={rc}"); return 2

    finals_a, max_active_a, max_gen_a = collect_all(
        list(range(7001, 7013)))
    if len(finals_a) != 12:
        print(f"  FAIL only {len(finals_a)}/12 distinct streams completed")
        return 3
    print(f"    ✓ all 12 distinct streams completed")
    print(f"    max active_streams seen: {max_active_a}")
    print(f"    max generating_streams seen: {max_gen_a} (slot budget B={B_SLOTS})")
    if max_active_a < 8:
        print(f"  FAIL active never exceeded 8 — engine isn't holding "
              f"oversupply in residency"); return 4
    if max_gen_a > B_SLOTS:
        print(f"  FAIL generating exceeded slot budget {B_SLOTS}")
        return 5
    print(f"    ✓ oversupply queued correctly: active≥8, generating≤B")

    # ---- (b) Identical prompts: oversupply meets §2 dedup ----
    print("\n  [b] 12 streams, identical prompt (oversupply + §2 dedup)")
    same_prompt = [2] + [(50 + i * 13) % 32000 for i in range(60)]
    same_streams = [
        g.StreamSpec(stream_id=sid, action=0, tokens=same_prompt,
                     sampling=sampling)
        for sid in range(7101, 7113)
    ]
    rc = g.submit(same_streams)
    if rc != 0:
        print(f"  FAIL submit same rc={rc}"); return 6

    finals_b, max_active_b, max_gen_b = collect_all(
        list(range(7101, 7113)))
    if len(finals_b) != 12:
        print(f"  FAIL only {len(finals_b)}/12 identical streams completed")
        return 7

    leaders = [u for u in finals_b.values() if u.cache_hits == 0]
    followers = [u for u in finals_b.values() if u.cache_hits > 0]
    print(f"    ✓ all 12 identical streams completed")
    print(f"    leaders (cache_hits=0): {len(leaders)}")
    print(f"    followers (cache_hits>0): {len(followers)}")
    print(f"    max generating_streams seen: {max_gen_b} (B={B_SLOTS})")
    if len(leaders) != 1:
        print(f"  FAIL expected 1 leader, got {len(leaders)} — §2 dedup "
              f"didn't fire under oversupply"); return 8
    if len(followers) != 11:
        print(f"  FAIL expected 11 followers, got {len(followers)}")
        return 9
    if max_gen_b > B_SLOTS:
        print(f"  FAIL generating exceeded slot budget"); return 10

    follower_hits = followers[0].cache_hits
    same_hits = all(u.cache_hits == follower_hits for u in followers)
    if not same_hits:
        hits_set = sorted({u.cache_hits for u in followers})
        print(f"  WARN follower hits not uniform: {hits_set}")
    else:
        print(f"    ✓ all 11 followers adopted exactly {follower_hits} tok")

    print("\n  ALL §3 oversupply checks passed.")
    g.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
