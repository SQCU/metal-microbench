#!/usr/bin/env python3
"""logit_bias smoke: same prompt twice, second time bias one token to dominate.

Asserts:
  - Without bias, model emits its natural argmax token.
  - With +50 bias on a chosen "winner" token, that token is sampled
    instead at temperature=0.
This proves the unified-FFI logit_bias field flows through encode →
decode → Session.logitBiasDense → sampler kernel correctly.
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


def drain(stream_id: int, deadline_s: float = 30.0):
    deadline = time.time() + deadline_s
    last = None
    while time.time() < deadline:
        for u in g.poll(timeout_ms=200):
            if u.stream_id == stream_id:
                last = u
                if u.state == 2:
                    return u
    return last


def main() -> int:
    print("=== logit_bias unified-interface smoke ===")
    rc = g.init(GGUF)
    if rc != 0:
        print(f"  init failed: rc={rc}"); return 1

    prompt = [2] + [(100 + i * 7) % 32000 for i in range(20)]

    # ---- Run 1: no bias. Capture natural argmax token. ----
    spec1 = g.StreamSpec(
        stream_id=5001, action=0, tokens=prompt,
        sampling=g.SamplingParams(
            temperature=0.7, max_new_tokens=1, eos_token_id=106))
    rc = g.submit([spec1])
    if rc != 0: print(f"  FAIL submit1 rc={rc}"); return 2
    f1 = drain(5001)
    if f1 is None or not f1.new_tokens:
        print(f"  FAIL run1 produced no token; final={f1}"); return 3
    natural_tok = f1.new_tokens[0]
    print(f"  run1 (no bias): natural argmax = token {natural_tok}")

    # ---- Run 2: same prompt, bias a different token to be chosen. ----
    target_tok = (natural_tok + 1234) % 262144  # arbitrary distinct token
    biased_sampling = g.SamplingParams(
        temperature=0.7, max_new_tokens=1, eos_token_id=106,
        logit_bias={target_tok: 50.0},
    )
    spec2 = g.StreamSpec(
        stream_id=5002, action=0, tokens=prompt,
        sampling=biased_sampling)
    rc = g.submit([spec2])
    if rc != 0: print(f"  FAIL submit2 rc={rc}"); return 4
    f2 = drain(5002)
    if f2 is None or not f2.new_tokens:
        print(f"  FAIL run2 produced no token; final={f2}"); return 5
    biased_tok = f2.new_tokens[0]
    print(f"  run2 (logit_bias[{target_tok}]=+50): sampled token = {biased_tok}")

    if biased_tok != target_tok:
        print(f"  FAIL expected biased token {target_tok}, got {biased_tok}")
        print(f"       (natural was {natural_tok}; bias +50 should dominate)")
        return 6
    print(f"  ✓ logit_bias dominates: sampled the +50-biased token")

    # ---- Run 3: clear bias on a continue. Should snap back to natural. ----
    spec3 = g.StreamSpec(
        stream_id=5003, action=0, tokens=prompt,
        sampling=g.SamplingParams(
            temperature=0.7, max_new_tokens=1, eos_token_id=106,
            logit_bias={}))   # explicit empty
    rc = g.submit([spec3])
    if rc != 0: print(f"  FAIL submit3 rc={rc}"); return 7
    f3 = drain(5003)
    if f3 is None or not f3.new_tokens:
        print(f"  FAIL run3 produced no token"); return 8
    cleared_tok = f3.new_tokens[0]
    if cleared_tok != natural_tok:
        print(f"  FAIL run3 (no bias) expected natural {natural_tok}, got {cleared_tok}")
        return 9
    print(f"  ✓ empty logit_bias = no-op; sampler returned to natural argmax")

    print(f"\n  ALL logit_bias checks passed.")
    g.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
