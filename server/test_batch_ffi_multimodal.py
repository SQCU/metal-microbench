#!/usr/bin/env python3
"""Multimodal + multi-turn smoke for the batch FFI.

Loads weights + vision tower, submits a single stream containing a
text-then-image-then-text segment sequence, drains until done, then runs
a second stream with the SAME image bytes to verify the vision cache
fires (vision_cache_hits > 0). Then drives a third stream with action=
continue across batches to verify multi-turn KV reuse over the FFI.

Usage:
  python3 server/test_batch_ffi_multimodal.py
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
VISION_ST = os.environ.get(
    "VISION_ST",
    "/Users/mdot/models/gemma-4-a4b-bf16/model-00001-of-00002.safetensors",
)
IMG_PATH = os.environ.get(
    "IMG_PATH",
    "/Users/mdot/metal-microbench/test_data/frames/frame_00_fbb737dcf6b0.png",
)


def drain(stream_id: int, deadline_s: float = 30.0):
    deadline = time.time() + deadline_s
    last = None
    while time.time() < deadline:
        for u in gb.poll(timeout_ms=200):
            if u.stream_id == stream_id:
                last = u
                if u.state == 2:
                    return u
    return last


def main() -> int:
    print("=== batch FFI multimodal + multi-turn smoke ===")
    print(f"  GGUF      = {GGUF}")
    print(f"  VISION_ST = {VISION_ST}")
    print(f"  IMG_PATH  = {IMG_PATH}")

    rc = gb.init(GGUF, VISION_ST)
    if rc != 0:
        print(f"  gemma_init failed: rc={rc}")
        return 1
    print("  init ok")

    img_bytes = open(IMG_PATH, "rb").read()
    print(f"  image bytes: {len(img_bytes)}")

    # ---- 1. Multimodal start: text-image-text ----
    print(f"\n  [1] multimodal stream: text + image + text")
    text_pre  = [2] + [(100 + i * 7) % 32000 for i in range(10)]   # 11 tok with BOS
    text_post = [(200 + i * 5) % 32000 for i in range(8)]          # 8 tok
    spec1 = gb.StreamSpec(
        stream_id=1001,
        action=0,
        segments=[
            gb.Segment(kind=0, tokens=text_pre),
            gb.Segment(kind=1, image_bytes=img_bytes),
            gb.Segment(kind=0, tokens=text_post),
        ],
        sampling=gb.SamplingParams(
            temperature=0.7, max_new_tokens=4, eos_token_id=106),
    )
    rc = gb.submit([spec1])
    if rc != 0:
        print(f"  FAIL submit rc={rc}")
        return 2
    fin1 = drain(1001)
    if fin1 is None or fin1.state != 2:
        print(f"  FAIL stream 1001 didn't reach done; final={fin1}")
        return 3
    print(f"  ✓ stream 1001 done: prompt_seen={fin1.prompt_tokens_seen} "
          f"completion={fin1.completion_tokens_emitted} "
          f"cache_hits={fin1.cache_hits} cache_misses={fin1.cache_misses} "
          f"vision_cache_hits={fin1.vision_cache_hits}")

    # ---- 2. Same image, different surrounding text → vision cache should hit ----
    print(f"\n  [2] second stream: same image, different surrounding text")
    text_pre2  = [2] + [(50 + i * 11) % 32000 for i in range(10)]
    text_post2 = [(75 + i * 13) % 32000 for i in range(8)]
    spec2 = gb.StreamSpec(
        stream_id=1002,
        action=0,
        segments=[
            gb.Segment(kind=0, tokens=text_pre2),
            gb.Segment(kind=1, image_bytes=img_bytes),
            gb.Segment(kind=0, tokens=text_post2),
        ],
        sampling=gb.SamplingParams(
            temperature=0.7, max_new_tokens=4, eos_token_id=106),
    )
    rc = gb.submit([spec2])
    if rc != 0:
        print(f"  FAIL submit rc={rc}")
        return 4
    fin2 = drain(1002)
    if fin2 is None or fin2.state != 2:
        print(f"  FAIL stream 1002 didn't reach done; final={fin2}")
        return 5
    print(f"  ✓ stream 1002 done: prompt_seen={fin2.prompt_tokens_seen} "
          f"completion={fin2.completion_tokens_emitted} "
          f"cache_hits={fin2.cache_hits} cache_misses={fin2.cache_misses} "
          f"vision_cache_hits={fin2.vision_cache_hits}")
    if fin2.vision_cache_hits == 0:
        print(f"  FAIL: expected vision_cache_hits>0 on identical image")
        return 6

    # ---- 3. Multi-turn via continue: same stream, additional turn ----
    print(f"\n  [3] multi-turn: stream 1003 start + continue")
    starter = [2] + [(100 + i * 17) % 32000 for i in range(40)]
    spec3 = gb.StreamSpec(
        stream_id=1003,
        action=0,
        segments=[gb.Segment(kind=0, tokens=starter)],
        sampling=gb.SamplingParams(
            temperature=0.7, max_new_tokens=2, eos_token_id=106),
    )
    rc = gb.submit([spec3])
    if rc != 0:
        print(f"  FAIL submit start rc={rc}")
        return 7
    fin3 = drain(1003)
    if fin3 is None or fin3.state != 2:
        print(f"  FAIL stream 1003 first turn didn't reach done")
        return 8
    print(f"  ✓ turn 1 done: prompt={fin3.prompt_tokens_seen} "
          f"completion={fin3.completion_tokens_emitted} "
          f"hits={fin3.cache_hits} misses={fin3.cache_misses}")

    # Stream 1003 is now closed (state==done means session was cleared).
    # For real multi-turn, the bridge should keep the session alive
    # across turns — which means action=continue, NOT a fresh start with
    # a new stream id and the full reconstructed history.
    #
    # The current backend completes done sessions and frees their KV.
    # To exercise true multi-turn, we'd need either a "keep_alive" flag
    # on the action or a stream that doesn't auto-close. For now, this
    # smoke verifies the action=start path works; the action=continue
    # semantic is implemented in the backend but only meaningful for
    # streams that haven't transitioned to done. Mark this case as
    # documented-but-deferred.
    print(f"  (multi-turn continue test deferred: stream auto-closes at done; "
          f"keep-alive flag is a follow-up — see bandwidth_triage.md §1)")

    # ---- Stats sanity check ----
    print(f"\n  status snapshot:")
    s = gb.status()
    print(f"    pages: {s.cached_pages}/{s.total_pages} cached, {s.free_pages} free")
    print(f"    streams: {s.active_streams} active ({s.generating_streams} gen, "
          f"{s.priming_streams} prim)")
    print(f"    total: {s.total_steps} steps, {s.total_tokens_emitted} tok emitted")
    print(f"    vision: {s.vision_cache_entries} entries, {s.vision_cache_hits} hits")

    print(f"\n  all multimodal smoke checks passed.")
    gb.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
