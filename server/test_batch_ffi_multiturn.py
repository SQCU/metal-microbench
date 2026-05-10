#!/usr/bin/env python3
"""Multi-turn prefix cache reuse + logprobs validation for the batch FFI.

Mirrors `test_prefix_cache.swift`'s case-3 (multi-turn rollout divergence)
shape, but exercises it through the new batch FFI surface and asserts on
cache_hits / cache_misses surfaced in StreamUpdate. Also validates the
logprobs payload (sampled token logprob + top-K) on a stream that opted
into capture_logits.

This is the gating milestone for §1: if these pass, the FFI surface
collapse is validated for the curriculum's actual usage pattern (each
turn opens a new stream with the full token-ID history; the backend's
content-hash cache picks up the prior turn's pages with no bridge
involvement).

Usage:
  python3 server/test_batch_ffi_multiturn.py
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


def drain(stream_id: int, deadline_s: float = 30.0, verbose: bool = False):
    deadline = time.time() + deadline_s
    last = None
    while time.time() < deadline:
        for u in gb.poll(timeout_ms=200):
            if u.stream_id == stream_id:
                last = u
                if verbose:
                    print(f"    update: state={u.state} done={u.done_reason} "
                          f"new={u.new_tokens} hits={u.cache_hits} "
                          f"misses={u.cache_misses} compl={u.completion_tokens_emitted}")
                if u.state == 2:
                    return u
    return last


def main() -> int:
    print("=== batch FFI multi-turn + logprobs smoke ===")
    rc = gb.init(GGUF)
    if rc != 0:
        print(f"  init failed: rc={rc}")
        return 1
    print("  init ok")

    # ------------------------------------------------------------------
    # Curriculum-shape pattern: a shared starter prompt, then per-rollout
    # divergent responses, then a turn-2 user prompt that should adopt
    # the rollout's full history (shared starter + own response).
    #
    # Each iter is its own stream because the curriculum opens a fresh
    # session per iter — exactly what the existing case-3 in Swift
    # exercises, just driven through the batch FFI.
    # ------------------------------------------------------------------
    shared = [2] + [(100 + i * 7) % 32000 for i in range(40)]   # 41 tok, 2.5 pages
    resp_a = [(200 + i * 5) % 32000 for i in range(20)]         # 20 tok
    resp_b = [(300 + i * 11) % 32000 for i in range(20)]
    user2  = [(400 + i * 13) % 32000 for i in range(8)]

    sampling = gb.SamplingParams(
        temperature=0.7, max_new_tokens=2, eos_token_id=106)

    # ---- Turn 1, rollout A: prompt = shared + resp_a ----
    print(f"\n  [A1] cold submit: shared(41) + resp_a(20) = {len(shared)+len(resp_a)} tok")
    a1 = gb.StreamSpec(
        stream_id=2001, action=0,
        tokens=shared + resp_a,
        sampling=sampling)
    rc = gb.submit([a1])
    if rc != 0:
        print(f"  FAIL submit A1 rc={rc}"); return 2
    fa1 = drain(2001)
    if fa1 is None or fa1.state != 2:
        print(f"  FAIL A1 didn't reach done; final={fa1}"); return 3
    print(f"  ✓ A1 done: hits={fa1.cache_hits} misses={fa1.cache_misses}")
    if fa1.cache_hits != 0:
        print(f"  FAIL A1 should have cache_hits=0 (cold)"); return 4

    # ---- Turn 1, rollout B: prompt = shared + resp_b ----
    print(f"\n  [B1] warm-shared submit: shared(41) + resp_b(20)")
    b1 = gb.StreamSpec(
        stream_id=2002, action=0,
        tokens=shared + resp_b,
        sampling=sampling)
    rc = gb.submit([b1])
    if rc != 0:
        print(f"  FAIL submit B1 rc={rc}"); return 5
    fb1 = drain(2002)
    if fb1 is None or fb1.state != 2:
        print(f"  FAIL B1 didn't reach done"); return 6
    print(f"  ✓ B1 done: hits={fb1.cache_hits} misses={fb1.cache_misses}")
    # B1 shares only the 'shared' portion with A1 (41 tokens).
    # PAGE_SLIDE=16, so 41 tok = 2 full pages + 9 tail. A1 promotes
    # 2 pages (32 tokens) to the cache; B1 should hit those exactly.
    expected_b1_hits = 32
    if fb1.cache_hits != expected_b1_hits:
        print(f"  FAIL B1 expected hits={expected_b1_hits}, got {fb1.cache_hits}")
        return 7

    # ---- Turn 2, rollout A: prompt = shared + resp_a + user2 ----
    print(f"\n  [A2] turn-2 A: shared+resp_a+user2 = "
          f"{len(shared)+len(resp_a)+len(user2)} tok")
    a2 = gb.StreamSpec(
        stream_id=2003, action=0,
        tokens=shared + resp_a + user2,
        sampling=sampling)
    rc = gb.submit([a2])
    if rc != 0:
        print(f"  FAIL submit A2 rc={rc}"); return 8
    fa2 = drain(2003)
    if fa2 is None or fa2.state != 2:
        print(f"  FAIL A2 didn't reach done"); return 9
    print(f"  ✓ A2 done: hits={fa2.cache_hits} misses={fa2.cache_misses}")
    # A2 shares shared(41) + resp_a(20) = 61 tok with A1. PAGE_SLIDE=16
    # so 61 tok = 3 full pages + 13 tail; A1 promoted 3 full pages = 48 tok.
    # A2 should adopt 48 tokens of cache_hits.
    expected_a2_hits = 48
    if fa2.cache_hits != expected_a2_hits:
        print(f"  FAIL A2 expected hits={expected_a2_hits}, got {fa2.cache_hits}")
        return 10
    print(f"  ✓ A2 adopted full A1 history (48 tok = 3 full pages)")

    # ---- Turn 2, rollout B: prompt = shared + resp_b + user2 ----
    print(f"\n  [B2] turn-2 B: shared+resp_b+user2")
    b2 = gb.StreamSpec(
        stream_id=2004, action=0,
        tokens=shared + resp_b + user2,
        sampling=sampling)
    rc = gb.submit([b2])
    if rc != 0:
        print(f"  FAIL submit B2 rc={rc}"); return 11
    fb2 = drain(2004)
    if fb2 is None or fb2.state != 2:
        print(f"  FAIL B2 didn't reach done"); return 12
    print(f"  ✓ B2 done: hits={fb2.cache_hits} misses={fb2.cache_misses}")
    # B2 shares shared+resp_b = 61 tok with B1; B1 promoted 3 pages.
    # B2 must NOT pick up A's pages (cross-rollout pollution check).
    if fb2.cache_hits != expected_a2_hits:
        print(f"  FAIL B2 expected hits={expected_a2_hits}, got {fb2.cache_hits}")
        return 13
    print(f"  ✓ B2 adopted full B1 history (no cross-rollout pollution)")

    # ------------------------------------------------------------------
    # Logprobs validation: open a stream with capture_logits + top_logprobs=5,
    # verify each emitted token has a sampled_logprob and 5 top entries.
    # ------------------------------------------------------------------
    print(f"\n  [LP] capture_logits stream with top_logprobs=5")
    lp_sampling = gb.SamplingParams(
        temperature=0.7, max_new_tokens=4, eos_token_id=106,
        top_logprobs=5)
    lp_spec = gb.StreamSpec(
        stream_id=2005, action=0,
        flags=0x01,                           # bit 0 = capture_logits
        tokens=shared,
        sampling=lp_sampling)
    rc = gb.submit([lp_spec])
    if rc != 0:
        print(f"  FAIL submit LP rc={rc}"); return 14

    # Collect ALL updates for this stream until done so we get the
    # logprobs from each tick.
    collected_lp = []
    deadline = time.time() + 30.0
    while time.time() < deadline:
        for u in gb.poll(timeout_ms=200):
            if u.stream_id != 2005:
                continue
            collected_lp.extend(u.logprobs)
            if u.state == 2:
                break
        else:
            continue
        break

    print(f"  ✓ LP stream done: {len(collected_lp)} logprob records")
    if len(collected_lp) == 0:
        print(f"  FAIL: expected logprob records, got none"); return 15
    for i, rec in enumerate(collected_lp):
        # sampled_logprob should be a finite negative number (or 0) under
        # log-softmax. Top entries should be sorted descending.
        if not (rec.sampled_logprob <= 1e-3):
            print(f"  FAIL: rec[{i}] sampled_logprob={rec.sampled_logprob} "
                  f"should be ≤ 0 (it's a logprob)")
            return 16
        if len(rec.top_logprobs) != 5:
            print(f"  FAIL: rec[{i}] expected 5 top entries, got "
                  f"{len(rec.top_logprobs)}")
            return 17
        # Top should be sorted desc by logprob.
        for j in range(1, 5):
            if rec.top_logprobs[j][1] > rec.top_logprobs[j-1][1] + 1e-5:
                print(f"  FAIL: rec[{i}] top entries not sorted desc")
                return 18
        print(f"    tok={rec.token} lp={rec.sampled_logprob:.4f} "
              f"top5={[(t, f'{lp:.3f}') for t, lp in rec.top_logprobs]}")

    print(f"\n  ALL multi-turn + logprobs checks passed.")
    gb.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
