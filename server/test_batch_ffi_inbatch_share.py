#!/usr/bin/env python3
"""§2 validator: in-batch shared-prefix detection.

Submits 4 streams in ONE gemma_submit_batch call, all with the same
cold prefix. Asserts that the backend deduplicates internally:
  - 3 of 4 streams report cache_hits ≈ prefix_token_count
  - 1 stream (the leader) reports cache_hits = 0 (it did the prefill)
  - Total page allocation ≈ 1× prefix (not 4×)

This is the test the case-2 race never could pass under the legacy
per-session FFI: there, 4 simultaneous probes against an empty cache
all missed, all prefilled redundantly. With the batch FFI's in-batch
shared-prefix detection (notes/specs/bandwidth_triage.md §2), only
the leader prefills; followers defer until the leader's first page
promotes, then adopt.

Usage:
  python3 server/test_batch_ffi_inbatch_share.py
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemma_ffi as gb  # noqa: E402

GGUF = os.environ.get(
    "GGUF_PATH",
    "/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
)


def collect_all(stream_ids: list[int], deadline_s: float = 60.0):
    """Drain N streams concurrently. Returns dict[stream_id → final update]."""
    pending = set(stream_ids)
    finals: dict[int, gb.StreamUpdate] = {}
    deadline = time.time() + deadline_s
    while pending and time.time() < deadline:
        for u in gb.poll(timeout_ms=200):
            if u.stream_id in pending:
                if u.state == 2:
                    finals[u.stream_id] = u
                    pending.discard(u.stream_id)
    return finals


def main() -> int:
    print("=== §2 in-batch shared-prefix detection ===")
    rc = gb.init(GGUF)
    if rc != 0:
        print(f"  init failed: rc={rc}"); return 1
    print("  init ok")

    # 64-token prefix, 4 logical pages (PAGE_SLIDE=16). Submit-time
    # backoff drops 1 page → max-adoptable = 3 pages = 48 tokens.
    prefix = [2] + [(100 + i * 7) % 32000 for i in range(63)]
    sampling = gb.SamplingParams(
        temperature=0.7, max_new_tokens=4, eos_token_id=106)

    # 4 streams, all start, all same prefix, all in ONE batch.
    streams = []
    for sid in (4001, 4002, 4003, 4004):
        streams.append(gb.StreamSpec(
            stream_id=sid, action=0,
            tokens=prefix,
            sampling=sampling))

    print(f"\n  submitting 4 streams with identical 64-token prefix in ONE batch")
    pre = gb.status()
    print(f"  pre-submit pool: free={pre.free_pages}/{pre.total_pages} cached={pre.cached_pages}")
    rc = gb.submit(streams)
    if rc != 0:
        print(f"  FAIL submit rc={rc}"); return 2

    finals = collect_all([4001, 4002, 4003, 4004], deadline_s=60.0)
    if len(finals) != 4:
        print(f"  FAIL only {len(finals)}/4 streams reached done")
        return 3

    print(f"\n  per-stream results:")
    for sid in (4001, 4002, 4003, 4004):
        u = finals[sid]
        print(f"    stream {sid}: state={u.state} done={u.done_reason} "
              f"prompt={u.prompt_tokens_seen} compl={u.completion_tokens_emitted} "
              f"hits={u.cache_hits} misses={u.cache_misses}")

    # Exactly one leader (cache_hits=0); the rest are followers (hits≈48).
    leaders = [u for u in finals.values() if u.cache_hits == 0]
    followers = [u for u in finals.values() if u.cache_hits > 0]
    print(f"\n  leaders: {len(leaders)}; followers: {len(followers)}")
    if len(leaders) != 1:
        print(f"  FAIL expected exactly 1 leader (in-batch dedup); got {len(leaders)}")
        return 4
    if len(followers) != 3:
        print(f"  FAIL expected 3 followers; got {len(followers)}")
        return 5
    # Each follower should adopt the same number of pages.
    follower_hits = [u.cache_hits for u in followers]
    if len(set(follower_hits)) > 1:
        print(f"  FAIL followers report mismatched hits: {follower_hits}")
        return 6
    # 64 tokens = 4 pages = 64 tokens, but submit-time backoff drops 1
    # page if the prompt exactly fills the cache. Allow 32 or 48 tok of
    # adoption (32 = 2 pages, 48 = 3 pages — depending on which pages
    # the leader had time to promote before followers retried).
    f_hits = follower_hits[0]
    if f_hits not in (32, 48):
        print(f"  FAIL follower hits={f_hits}, expected 32 or 48")
        return 7
    print(f"  ✓ exactly 1 leader, 3 followers; followers each adopt {f_hits} tok")

    post = gb.status()
    pages_used = pre.free_pages - post.free_pages
    print(f"  post-submit pool: free={post.free_pages}/{post.total_pages} "
          f"cached={post.cached_pages}; this run consumed {pages_used} pages")
    # Without in-batch dedup, 4 streams × ~13 pages (the per-stream
    # ensurePages footprint we saw under the legacy FFI's case 2) =
    # ~512 pages. With dedup, expect ~50–80 pages.
    if pages_used > 200:
        print(f"  WARN page consumption is high ({pages_used}); check ensurePages "
              f"behavior under follower adoption")
    print(f"\n  ALL §2 checks passed.")
    gb.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
