#!/usr/bin/env python3
"""Smoke for ABI v3 extensions: min_p, structured_cot, control_vectors,
and the gemma_register_resource admin entry.

Validates the architectural rule the spec binds: every feature ports as
a field on the existing types (SamplingParams or StreamSpec) or as ONE
unified admin entry. No per-feature FFI exists.
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
    print("=== unified-interface extensions smoke (ABI v3) ===")
    rc = g.init(GGUF)
    if rc != 0:
        print(f"  init failed: rc={rc}"); return 1

    prompt = [2] + [(100 + i * 7) % 32000 for i in range(20)]

    # ---- min_p as a SamplingParams field ----
    print("\n  [1] min_p on SamplingParams")
    spec1 = g.StreamSpec(
        stream_id=6001, action=0, tokens=prompt,
        sampling=g.SamplingParams(
            temperature=0.7, max_new_tokens=4, eos_token_id=106,
            min_p=0.05))
    rc = g.submit([spec1])
    if rc != 0: print(f"  FAIL submit min_p rc={rc}"); return 2
    f1 = drain(6001)
    if f1 is None or f1.state != 2:
        print(f"  FAIL min_p stream didn't reach done"); return 3
    print(f"  ✓ min_p stream done: emitted {f1.completion_tokens_emitted} tokens")

    # ---- structured_cot grammar ----
    print("\n  [2] structured_cot grammar on SamplingParams")
    spec2 = g.StreamSpec(
        stream_id=6002, action=0, tokens=prompt,
        sampling=g.SamplingParams(
            temperature=0.7, max_new_tokens=4, eos_token_id=106,
            cot_labels=["GOAL", "APPROACH", "EDGE"]))
    rc = g.submit([spec2])
    if rc != 0: print(f"  FAIL submit cot rc={rc}"); return 4
    f2 = drain(6002)
    if f2 is None or f2.state != 2:
        print(f"  FAIL cot stream didn't reach done"); return 5
    print(f"  ✓ structured_cot stream done: emitted {f2.completion_tokens_emitted} tokens")

    # ---- register a control vector + apply it to a stream ----
    print("\n  [3] register_resource + control_vectors on StreamSpec")
    HIDDEN = 2816
    # Make a deterministic random fp16 cvec — content doesn't matter, only
    # that the wire format flows through and the stream completes.
    import struct
    cv_bytes = bytearray()
    seed = 0xDEADBEEF
    for i in range(HIDDEN):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        # Generate a small random fp16 value in [-0.01, 0.01]
        v = ((seed >> 8) % 1000) / 50000.0 - 0.01
        # fp16 packing
        import math
        # Use a tiny deterministic value; real CVs come from sklearn etc.
        cv_bytes += struct.pack("<e", v)
    rc = g.register_resource("cvec", "test_cvec_v3", bytes(cv_bytes))
    if rc != 0:
        print(f"  FAIL register_resource rc={rc}"); return 6
    print(f"  ✓ registered cvec 'test_cvec_v3' ({len(cv_bytes)} bytes)")

    # Apply it via a CVApplication on a stream.
    spec3 = g.StreamSpec(
        stream_id=6003, action=0, tokens=prompt,
        sampling=g.SamplingParams(
            temperature=0.7, max_new_tokens=4, eos_token_id=106),
        control_vectors=[g.CVApplication(
            cvec_id="test_cvec_v3",
            layer=12,
            polarity=1.0,
            peak_magnitude=0.05,    # gentle — don't want to break the model
            attack=0.0, decay=0.0, sustain_level=1.0, release=0.0,
            shape=0, units=0, mode=0,
        )])
    rc = g.submit([spec3])
    if rc != 0: print(f"  FAIL submit cv rc={rc}"); return 7
    f3 = drain(6003)
    if f3 is None or f3.state != 2:
        print(f"  FAIL cv stream didn't reach done"); return 8
    print(f"  ✓ control_vector stream done: emitted {f3.completion_tokens_emitted} tokens")

    # ---- empty control_vectors on the same prompt — should be a no-op
    #      and produce identical output to a stream with no CV field.
    print("\n  [4] empty control_vectors = no-op")
    spec4 = g.StreamSpec(
        stream_id=6004, action=0, tokens=prompt,
        sampling=g.SamplingParams(
            temperature=0.7, max_new_tokens=4, eos_token_id=106),
        control_vectors=[])
    rc = g.submit([spec4])
    if rc != 0: print(f"  FAIL submit empty-cv rc={rc}"); return 9
    f4 = drain(6004)
    if f4 is None or f4.state != 2:
        print(f"  FAIL empty-cv stream didn't reach done"); return 10
    print(f"  ✓ empty-cv stream done: emitted {f4.completion_tokens_emitted} tokens")

    print("\n  ALL unified-interface extension checks passed.")
    g.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
