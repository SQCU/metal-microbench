#!/usr/bin/env python3
"""Regression: multimodal multi-turn dialog (text + image_url content
parts) gets the same prefix-cache reuse as pure-text dialog.

Sends three turns where the second user turn embeds a small inline PNG
(data: URL). Because the new image is part of the tail (not the
historical prefix), turn 1's KV (which is text-only) should still be
fully adopted by turn 2.

Turn 3 sends a SECOND inline PNG. Turn 2's KV (text + iter-1's image
soft tokens + assistant response) should be fully adopted; only the
new image + closing scaffolding gets prefilled fresh.
"""
from __future__ import annotations
import base64
import json
import os
import urllib.request
import zlib
import struct

BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8001")
PAGE_SLIDE = 16


def _make_solid_png(rgb: tuple[int, int, int], size: int = 32) -> bytes:
    """Tiny deterministic PNG so the engine has actual image bytes to
    process. Made fresh per call so two calls produce IDENTICAL bytes
    (deterministic) but different colors → different SHA-256 cache keys."""
    raw_rows = []
    for _ in range(size):
        row = bytes([0]) + bytes(rgb) * size  # filter byte + RGB triplets
        raw_rows.append(row)
    raw = b"".join(raw_rows)
    compressed = zlib.compress(raw, level=9)

    def chunk(t: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(t + data)
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (sig + chunk(b"IHDR", ihdr)
             + chunk(b"IDAT", compressed)
             + chunk(b"IEND", b""))


def _png_data_url(rgb: tuple[int, int, int]) -> str:
    return "data:image/png;base64," + base64.b64encode(_make_solid_png(rgb)).decode()


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
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    print("=== chat-template prefix continuity (multimodal) ===")
    with urllib.request.urlopen(f"{BRIDGE_URL}/health", timeout=5) as r:
        h = json.loads(r.read())
    print(f"  bridge: {h['model']}")

    # Turn 1: text only (cold).
    print("\n  Turn 1: text-only cold start")
    msgs1 = [{"role": "user", "content": "Describe what red looks like in one sentence."}]
    r1 = post_chat(msgs1)
    a1 = r1["choices"][0]["message"]["content"]
    u1 = r1["usage"]
    print(f"    prompt_tokens={u1['prompt_tokens']} "
          f"completion_tokens={u1['completion_tokens']} "
          f"hits={u1['cache_hits']} misses={u1['cache_misses']}")
    if u1["cache_hits"] != 0:
        print(f"  ✗ turn 1 cold expected hits=0, got {u1['cache_hits']}"); return 1
    turn1_kv = u1["prompt_tokens"] + u1["completion_tokens"]

    # Turn 2: history + new user turn with embedded RED image.
    print("\n  Turn 2: + assistant + new user with image")
    msgs2 = msgs1 + [
        {"role": "assistant", "content": a1},
        {"role": "user", "content": [
            {"type": "text", "text": "Does this image match?"},
            {"type": "image_url",
             "image_url": {"url": _png_data_url((220, 30, 30))}},
        ]},
    ]
    r2 = post_chat(msgs2)
    a2 = r2["choices"][0]["message"]["content"]
    u2 = r2["usage"]
    print(f"    prompt_tokens={u2['prompt_tokens']} "
          f"completion_tokens={u2['completion_tokens']} "
          f"hits={u2['cache_hits']} misses={u2['cache_misses']} "
          f"vision_hits={u2['vision_cache_hits']}")
    # turn 1's KV is fully text → should be adopted by turn 2's prefix
    # (which begins with the SAME canonical bytes for system+user1+
    # assistant1, since we replay stored tokens). Round to PAGE_SLIDE.
    expected_min = (turn1_kv // PAGE_SLIDE) * PAGE_SLIDE
    if u2["cache_hits"] < expected_min:
        print(f"  ✗ FAIL turn 2: hits={u2['cache_hits']}, "
              f"expected ≥ {expected_min} (turn 1 KV total = {turn1_kv})")
        return 2
    print(f"  ✓ turn 2 adopted {u2['cache_hits']} ≥ {expected_min}")
    # Turn 2's KV = submission + emitted = (text prefix + image soft
    # tokens + closing text + emitted). Image contributes ~280 vision
    # soft tokens (Gemma-4 padded soft count).
    turn2_kv = u2["prompt_tokens"] + u2["completion_tokens"]

    # Turn 3: history + new user turn with a DIFFERENT image (green).
    print("\n  Turn 3: + assistant + new user with second image")
    msgs3 = msgs2 + [
        {"role": "assistant", "content": a2},
        {"role": "user", "content": [
            {"type": "text", "text": "And this one?"},
            {"type": "image_url",
             "image_url": {"url": _png_data_url((30, 200, 50))}},
        ]},
    ]
    r3 = post_chat(msgs3)
    u3 = r3["usage"]
    print(f"    prompt_tokens={u3['prompt_tokens']} "
          f"completion_tokens={u3['completion_tokens']} "
          f"hits={u3['cache_hits']} misses={u3['cache_misses']} "
          f"vision_hits={u3['vision_cache_hits']}")
    expected_min_t3 = (turn2_kv // PAGE_SLIDE) * PAGE_SLIDE
    if u3["cache_hits"] < expected_min_t3:
        print(f"  ✗ FAIL turn 3: hits={u3['cache_hits']}, "
              f"expected ≥ {expected_min_t3} (turn 2 KV total = {turn2_kv})")
        return 3
    print(f"  ✓ turn 3 adopted {u3['cache_hits']} ≥ {expected_min_t3}")

    print("\n  ALL multimodal prefix-continuity asserts passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
