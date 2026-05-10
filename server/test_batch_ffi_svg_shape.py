#!/usr/bin/env python3
"""Round-trip multimodal multi-turn validation for the batch FFI.

Mirrors the SVG-curriculum harness shape: each iter opens a fresh stream
with the full conversation history rebuilt as token IDs (system text +
image bytes + user1 + assistant1 + user2 + ...). Verifies:

  - vision_cache_hits fires on the second iter (same image bytes).
  - cache_hits covers the WHOLE prior-iter history including image
    soft-tokens + the actually-generated assistant tokens (because the
    test pulls iter 1's emitted tokens out of the response and folds
    them into iter 2's prompt — exactly what the bridge does).
  - cache_misses covers only the new user-turn portion of iter 2.

This is the gating end-to-end test for §1: the SVG curriculum's exact
hit pattern reproduced through the new FFI surface.

Usage:
  python3 server/test_batch_ffi_svg_shape.py
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


def collect(stream_id: int, deadline_s: float = 60.0):
    """Drain a stream, returning (final_update, all_emitted_tokens)."""
    deadline = time.time() + deadline_s
    last = None
    tokens: list[int] = []
    while time.time() < deadline:
        for u in gb.poll(timeout_ms=200):
            if u.stream_id == stream_id:
                last = u
                tokens.extend(u.new_tokens)
                if u.state == 2:
                    return last, tokens
    return last, tokens


def main() -> int:
    print("=== batch FFI SVG-shape multimodal multi-turn ===")
    rc = gb.init(GGUF, VISION_ST)
    if rc != 0:
        print(f"  init failed: rc={rc}"); return 1
    print("  init ok")
    img = open(IMG_PATH, "rb").read()
    print(f"  image bytes: {len(img)}")

    # Curriculum-shape conversation pieces. The exact tokens are
    # arbitrary — what matters is that iter 2's submission contains
    # iter 1's emitted output verbatim, so the cache should hit on
    # everything up through that emitted portion.
    sys_text = [2] + [(100 + i * 7) % 32000 for i in range(40)]   # 41 tok
    user1    = [(200 + i * 11) % 32000 for i in range(20)]         # 20 tok
    user2    = [(300 + i * 13) % 32000 for i in range(20)]         # 20 tok

    sampling = gb.SamplingParams(
        temperature=0.7, max_new_tokens=8, eos_token_id=106)

    # ---- ITER 1: cold, multimodal ----
    print("\n  [iter 1] cold multimodal stream")
    iter1 = gb.StreamSpec(
        stream_id=3001, action=0,
        segments=[
            gb.Segment(kind=0, tokens=sys_text),
            gb.Segment(kind=1, image_bytes=img),
            gb.Segment(kind=0, tokens=user1),
        ],
        sampling=sampling)
    rc = gb.submit([iter1])
    if rc != 0:
        print(f"  FAIL submit iter1 rc={rc}"); return 2
    f1, emitted_a1 = collect(3001)
    if f1 is None or f1.state != 2:
        print(f"  FAIL iter1 didn't reach done; final={f1}"); return 3
    print(f"  ✓ iter1 done: prompt={f1.prompt_tokens_seen} "
          f"completion={f1.completion_tokens_emitted} "
          f"cache_hits={f1.cache_hits} cache_misses={f1.cache_misses} "
          f"vision_cache_hits={f1.vision_cache_hits}")
    print(f"  emitted assistant tokens: {emitted_a1}")
    s_after_iter1 = gb.status()
    print(f"  status after iter1: cached_pages={s_after_iter1.cached_pages}, "
          f"free_pages={s_after_iter1.free_pages}")
    if f1.cache_hits != 0:
        print(f"  FAIL iter1 should be cold (cache_hits=0)"); return 4
    if f1.vision_cache_hits != 0:
        print(f"  FAIL iter1 vision should be cold (vision_cache_hits=0)"); return 5

    # ---- ITER 2: full history rebuilt, same image ----
    # iter 2's prompt = sys + image + user1 + assistant1 + user2.
    # Backend should hit cache on everything through assistant1, and
    # vision cache on the image. cache_misses should only cover user2
    # (and sampler-tail accounting from the trailing-page backoff).
    print("\n  [iter 2] same image + full history + new user turn")
    iter2 = gb.StreamSpec(
        stream_id=3002, action=0,
        segments=[
            gb.Segment(kind=0, tokens=sys_text),
            gb.Segment(kind=1, image_bytes=img),
            gb.Segment(kind=0, tokens=user1 + emitted_a1 + user2),
        ],
        sampling=sampling)
    rc = gb.submit([iter2])
    if rc != 0:
        print(f"  FAIL submit iter2 rc={rc}"); return 6
    f2, emitted_a2 = collect(3002)
    if f2 is None or f2.state != 2:
        print(f"  FAIL iter2 didn't reach done; final={f2}"); return 7
    print(f"  ✓ iter2 done: prompt={f2.prompt_tokens_seen} "
          f"completion={f2.completion_tokens_emitted} "
          f"cache_hits={f2.cache_hits} cache_misses={f2.cache_misses} "
          f"vision_cache_hits={f2.vision_cache_hits}")

    # Vision cache MUST hit (same image bytes).
    if f2.vision_cache_hits == 0:
        print(f"  FAIL iter2 expected vision_cache_hits>0"); return 8

    # Cache hits should be non-trivial. iter 1's prompt was
    # 41 (sys) + 282 (image: 280 softs + 2 BOI/EOI) + 20 (user1) +
    # 8 (emitted) = 351 tokens. iter 2 adds user2 (20 tokens) at the
    # tail. PAGE_SLIDE=16 means iter 2 should adopt floor(351/16) = 21
    # full pages = 336 tokens, missing only the trailing 15 + user2.
    # Allow some flex: require ≥ 320 hits (20 pages worth).
    if f2.cache_hits < 320:
        print(f"  FAIL iter2 expected cache_hits >= 320, got {f2.cache_hits}")
        print(f"       (turn-1 promoted ~21 pages × 16 tok = ~336 tok of cache)")
        return 9
    # cache_misses should be small relative to the prompt: only user2 +
    # any partial trailing page.
    if f2.cache_misses > 60:
        print(f"  FAIL iter2 expected cache_misses ≤ 60 (just user2 + tail), "
              f"got {f2.cache_misses}")
        return 10
    print(f"  ✓ iter2 cache reuse: hits={f2.cache_hits} (≥ 320 expected) "
          f"misses={f2.cache_misses} (≤ 60 expected)")

    # ---- ITER 3: same shape, again, to verify pages from iter 2 also
    #              promote and stack on top.
    print("\n  [iter 3] further turn — assistant2 + user3")
    user3 = [(400 + i * 17) % 32000 for i in range(20)]
    iter3 = gb.StreamSpec(
        stream_id=3003, action=0,
        segments=[
            gb.Segment(kind=0, tokens=sys_text),
            gb.Segment(kind=1, image_bytes=img),
            gb.Segment(kind=0, tokens=user1 + emitted_a1 + user2 + emitted_a2 + user3),
        ],
        sampling=sampling)
    rc = gb.submit([iter3])
    if rc != 0:
        print(f"  FAIL submit iter3 rc={rc}"); return 11
    f3, _ = collect(3003)
    if f3 is None or f3.state != 2:
        print(f"  FAIL iter3 didn't reach done"); return 12
    print(f"  ✓ iter3 done: hits={f3.cache_hits} misses={f3.cache_misses} "
          f"vision_cache_hits={f3.vision_cache_hits}")
    # iter 3 should have hits ≥ iter 2's, since it has more history to
    # adopt (iter 2's pages got promoted on iter 2's drain).
    if f3.cache_hits <= f2.cache_hits:
        print(f"  FAIL iter3 should have cache_hits > iter2 ({f2.cache_hits}); "
              f"got {f3.cache_hits}")
        return 13
    print(f"  ✓ iter3 grew its hit footprint: {f2.cache_hits} → {f3.cache_hits}")

    print(f"\n  ALL SVG-shape multimodal multi-turn checks passed.")
    s = gb.status()
    print(f"  status: pages cached={s.cached_pages}/{s.total_pages}, "
          f"vision entries={s.vision_cache_entries} hits={s.vision_cache_hits}, "
          f"total_tokens_emitted={s.total_tokens_emitted}")
    gb.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
