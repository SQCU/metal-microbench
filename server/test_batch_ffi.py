#!/usr/bin/env python3
"""End-to-end smoke test for the batch FFI (notes/specs/batch_ffi_abi.md).

Submits a single stream with a small prompt, polls until done, prints the
emitted tokens and per-stream usage. Validates wire format in both
directions and the basic engine integration.

Usage:
  python3 server/test_batch_ffi.py
"""
from __future__ import annotations
import os
import sys
import time

# Ensure we can import gemma_ffi_batch from this directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gemma_ffi as gb  # noqa: E402

GGUF = os.environ.get(
    "GGUF_PATH",
    "/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
)


def main() -> int:
    print(f"=== batch FFI smoke test ===")
    print(f"  loading {GGUF}")
    rc = gb.init(GGUF)
    if rc != 0:
        print(f"  gemma_init failed: rc={rc}")
        return 1

    # Build a 40-token prompt so it spans ≥ 2 logical pages (PAGE_SLIDE=16).
    # That way the second submission can actually adopt cached pages.
    # Real bridge will use the tokenizer; here we just need any deterministic
    # token IDs that span two full pages.
    prompt = [2] + [(100 + i * 17) % 32000 for i in range(40)]
    sampling = gb.SamplingParams(
        temperature=0.7,
        max_new_tokens=4,
        eos_token_id=106,
    )
    stream = gb.StreamSpec(
        stream_id=12345,
        action=0,  # start
        tokens=prompt,
        sampling=sampling,
    )
    print(f"  submitting stream 12345 with {len(prompt)} prompt tokens, "
          f"max_new_tokens={sampling.max_new_tokens}")
    rc = gb.submit([stream])
    if rc != 0:
        print(f"  gemma_submit_batch failed: rc={rc}")
        return 2
    print(f"  submit ok")

    final_update = _drain_to_done(stream_id=12345, deadline_s=30.0)
    if final_update is None:
        print("  FAIL: stream 12345 didn't reach done")
        return 3
    print(f"  ✓ basic completion: {final_update.completion_tokens_emitted} tokens, "
          f"cache_hits={final_update.cache_hits}, cache_misses={final_update.cache_misses}")

    # ------------------------------------------------------------------
    # Multi-turn smoke: open a fresh stream with the SAME prompt; expect
    # cache_hits ≈ prompt token count (minus trailing-page backoff).
    # ------------------------------------------------------------------
    print(f"\n  multi-turn cache reuse: re-submitting same prompt as stream 67890")
    stream2 = gb.StreamSpec(
        stream_id=67890,
        action=0,
        tokens=prompt,
        sampling=sampling,
    )
    rc = gb.submit([stream2])
    if rc != 0:
        print(f"  FAIL: gemma_submit_batch second-stream rc={rc}")
        return 4
    final2 = _drain_to_done(stream_id=67890, deadline_s=30.0)
    if final2 is None:
        print("  FAIL: stream 67890 didn't reach done")
        return 5
    print(f"  ✓ second stream done: completion={final2.completion_tokens_emitted}, "
          f"cache_hits={final2.cache_hits}, cache_misses={final2.cache_misses}")
    if final2.cache_hits == 0:
        print(f"  FAIL: expected cache_hits>0 on warm prompt re-submit")
        return 6
    print(f"\n  all batch FFI smoke checks passed.")
    return 0


def _drain_to_done(stream_id: int, deadline_s: float):
    deadline = time.time() + deadline_s
    last_update = None
    while time.time() < deadline:
        updates = gb.poll(timeout_ms=200)
        for u in updates:
            if u.stream_id != stream_id:
                continue
            last_update = u
            if u.state == 2:
                return u
    return last_update


if __name__ == "__main__":
    raise SystemExit(main())
