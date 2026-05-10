#!/usr/bin/env python3
"""Regression: chat-template multi-turn dialog produces token sequences
that prefix-extend cleanly across turns, enabling KV-cache reuse.

Failure mode this test catches:
  Turn 1 with add_generation_prompt=True ends with the no-thinking hint
  (`<|channel>thought\\n<channel|>`, atomic IDs 100/45518/107/101).
  Turn 2's canonical re-render does NOT include those tokens between
  `<|turn>model\\n` and the historical assistant content. So turn 1's
  KV at positions ~13..16 contains scaffolding while turn 2's tokens
  at the same positions contain assistant content — page hashes diverge,
  zero cache reuse on what should be a near-full hit.

Pre-existing tests (test_batch_ffi_multiturn.py, test_batch_ffi_inbatch_share.py)
exercise the engine's content-hash machinery with synthetic integer
arrays and never touch render_chat / tokenize_with_specials. They miss
this entire failure class.

Drives 3 turns through /v1/chat/completions; asserts each turn N+1
adopts ≥ floor(turn_N_total_KV / PAGE_SLIDE) * PAGE_SLIDE tokens.
"""
from __future__ import annotations
import json
import os
import sys
import urllib.request

BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8001")
PAGE_SLIDE = 16   # engine slide-page size; cache adoption is page-aligned


def post_chat(messages, max_tokens=20, temperature=0.7):
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{BRIDGE_URL}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    print("=== chat-template prefix continuity regression ===")

    with urllib.request.urlopen(f"{BRIDGE_URL}/health", timeout=5) as r:
        h = json.loads(r.read())
    print(f"  bridge: {h['model']} on {BRIDGE_URL}")

    # ── Turn 1 (cold) ──
    print("\n  Turn 1: cold start")
    msgs1 = [{"role": "user", "content": "Name one fun fact about cats."}]
    r1 = post_chat(msgs1)
    a1 = r1["choices"][0]["message"]["content"]
    u1 = r1["usage"]
    print(f"    prompt_tokens={u1['prompt_tokens']} "
          f"completion_tokens={u1['completion_tokens']}")
    print(f"    cache_hits={u1['cache_hits']} cache_misses={u1['cache_misses']}")
    print(f"    assistant: {a1[:60]!r}…")
    if u1['cache_hits'] != 0:
        print(f"  ✗ unexpected: turn 1 had cache_hits={u1['cache_hits']} (expected 0 cold)")
        return 1
    turn1_total_kv = u1['prompt_tokens'] + u1['completion_tokens']

    # ── Turn 2 (continuation) ──
    print("\n  Turn 2: should adopt turn-1 KV")
    msgs2 = msgs1 + [
        {"role": "assistant", "content": a1},
        {"role": "user", "content": "What about dogs?"},
    ]
    r2 = post_chat(msgs2)
    a2 = r2["choices"][0]["message"]["content"]
    u2 = r2["usage"]
    print(f"    prompt_tokens={u2['prompt_tokens']} "
          f"completion_tokens={u2['completion_tokens']}")
    print(f"    cache_hits={u2['cache_hits']} cache_misses={u2['cache_misses']}")
    expected_t2_min = (turn1_total_kv // PAGE_SLIDE) * PAGE_SLIDE
    if u2['cache_hits'] < expected_t2_min:
        print(f"  ✗ FAIL turn 2: hits={u2['cache_hits']}, "
              f"expected ≥ {expected_t2_min} "
              f"(turn-1 KV total = {turn1_total_kv})")
        return 2
    print(f"  ✓ turn 2 adopted {u2['cache_hits']} ≥ {expected_t2_min}")
    turn2_total_kv = u2['prompt_tokens'] + u2['completion_tokens']

    # ── Turn 3 (continuation) ──
    print("\n  Turn 3: should adopt turn-2 KV")
    msgs3 = msgs2 + [
        {"role": "assistant", "content": a2},
        {"role": "user", "content": "Cool, tell me about birds."},
    ]
    r3 = post_chat(msgs3)
    u3 = r3["usage"]
    print(f"    prompt_tokens={u3['prompt_tokens']} "
          f"completion_tokens={u3['completion_tokens']}")
    print(f"    cache_hits={u3['cache_hits']} cache_misses={u3['cache_misses']}")
    expected_t3_min = (turn2_total_kv // PAGE_SLIDE) * PAGE_SLIDE
    if u3['cache_hits'] < expected_t3_min:
        print(f"  ✗ FAIL turn 3: hits={u3['cache_hits']}, "
              f"expected ≥ {expected_t3_min} "
              f"(turn-2 KV total = {turn2_total_kv})")
        return 3
    print(f"  ✓ turn 3 adopted {u3['cache_hits']} ≥ {expected_t3_min}")

    print("\n  ALL prefix-continuity asserts passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
