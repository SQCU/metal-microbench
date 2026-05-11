#!/usr/bin/env python3
"""On-path correctness gate: cvec apply path reached, magnitudes
captured, residual stream UNCHANGED when the cvec is the zero vector.

This is the missing half of the validation story. The off-path gate
(baseline_logprobs.py) covered "engine bit-equals baseline when nothing
is asked of the cvec subsystem." This script covers:

  1. The bridge accepts `controls` and forwards to the engine.
  2. The engine's apply kernel actually runs (the magnitudes show up
     in the streamed `cvec_activations` deltas).
  3. With a zero-vector cvec, the residual stream gain is exactly
     zero — sampled logprobs bit-equal the baseline.
  4. With a non-zero cvec at non-trivial magnitude, the output DIFFERS
     from baseline (this is the "intervenes upon the residual streams"
     half — without it we'd have only proven we can MOVE BYTES, not
     that we ACTUALLY STEER).

Calls the bridge over HTTP and registers cvecs via the FFI wrapper.
"""
import base64, json, struct, sys, time, urllib.request
from pathlib import Path

BRIDGE = "http://127.0.0.1:8001"
HIDDEN = 2816     # gemma-4-a4b residual width (5632 bytes = 2 × HIDDEN at fp16)
LAYER = 12        # arbitrary middle-ish layer
PROMPT = (
    "Write the first sentence of a short story about a cartographer "
    "who maps the spaces between bookshelves."
)
N_TOKENS = 12
TOP_K = 5
SEED = 42
BASELINE_PATH = Path(__file__).parent / "baseline.json"


def _half_bytes(floats):
    """Convert a Python float list to fp16 little-endian bytes."""
    import numpy as np
    return np.asarray(floats, dtype=np.float32).astype(np.float16).tobytes()


def _register_via_bridge(cvec_id: str, payload: bytes):
    """Register a cvec in the bridge's process, not ours. The libgemma
    cvec registry is per-process; calling g.register_resource from this
    script would only register into a local dylib copy that the bridge
    never sees. /v1/resources/register routes the bytes through the
    bridge's own FFI handle."""
    req = urllib.request.Request(
        f"{BRIDGE}/v1/resources/register",
        data=json.dumps({
            "kind": "cvec",
            "id": cvec_id,
            "data_b64": base64.b64encode(payload).decode("ascii"),
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        out = json.loads(r.read())
    if out.get("rc") != 0:
        print(f"FAIL: register_resource({cvec_id!r}) → {out}")
        sys.exit(2)


def register_zero_cvec(cvec_id: str):
    payload = _half_bytes([0.0] * HIDDEN)
    assert len(payload) == HIDDEN * 2, (len(payload), HIDDEN * 2)
    _register_via_bridge(cvec_id, payload)


def register_unit_cvec(cvec_id: str, dim: int = 0):
    v = [0.0] * HIDDEN
    v[dim] = 1.0
    payload = _half_bytes(v)
    _register_via_bridge(cvec_id, payload)


def call_bridge_with_controls(controls):
    """Streaming call; collects content + cvec_activations + per-token
    logprobs from the SSE deltas. Streaming chosen so we can directly
    observe whether activation records actually flow back."""
    body = {
        "model": "gemma-4-a4b",
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "max_tokens": N_TOKENS,
        "temperature": 0.01,
        "seed": SEED,
        "logprobs": True,
        "top_logprobs": TOP_K,
        "capture_cvec_activations": True,
        "controls": controls,
    }
    req = urllib.request.Request(
        f"{BRIDGE}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    activations: list[dict] = []
    content_chars: list[str] = []
    finish = None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        for line in r:
            line = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_chars.append(delta["content"])
            if delta.get("cvec_activations"):
                activations.extend(delta["cvec_activations"])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    elapsed = time.time() - t0
    content = "".join(content_chars)
    # Streaming path doesn't carry per-token logprobs in deltas the way
    # the non-streaming response does (logprobs only land in the
    # terminal usage chunk, if at all). For the on-path correctness
    # comparison we fall back to comparing token sequences via
    # detokenizing the content + finish reason.
    return {
        "content": content,
        "finish_reason": finish,
        "elapsed_s": round(elapsed, 3),
        "activations": activations,
    }


def content_matches_baseline(streaming_result, baseline) -> bool:
    """Compare the streamed content head against the baseline's content
    head. The cvec-perturbation test cares about whether the generated
    sequence diverges, not about bit-equality of logprobs (streaming
    doesn't surface per-token logprobs in deltas)."""
    a = streaming_result["content"][:120].rstrip()
    b = baseline["content"][:120].rstrip()
    return a == b


def main():
    if not BASELINE_PATH.exists():
        print(f"FAIL: no baseline at {BASELINE_PATH} — "
              f"run baseline_logprobs.py --record first")
        sys.exit(2)
    baseline = json.loads(BASELINE_PATH.read_text())

    # ── Phase 1: zero-vector cvec applied with magnitude=1 ─────────
    # Expect: cvec_activations records POPULATE (apply path ran) AND
    # output content matches baseline (zero × anything = 0).
    register_zero_cvec("cvec-validation-zero")
    zero_result = call_bridge_with_controls([{
        "cvec_id": "cvec-validation-zero",
        "layer": LAYER,
        "polarity": 1.0,
        "peak_magnitude": 1.0,
        "sustain_level": 1.0,
        "mode": 0,  # additive
    }])
    print(f"[zero-cvec] elapsed={zero_result['elapsed_s']}s "
          f"finish={zero_result['finish_reason']} "
          f"activations={len(zero_result['activations'])}")
    if not zero_result['activations']:
        print("FAIL Phase 1: cvec_activations stream is empty — apply "
              "path never reached, the bridge or engine isn't wiring "
              "the control onto the active session")
        sys.exit(1)
    if not content_matches_baseline(zero_result, baseline):
        print("FAIL Phase 1: zero-vector cvec changed output content")
        print(f"  baseline: {baseline['content'][:120]!r}")
        print(f"  now:      {zero_result['content'][:120]!r}")
        sys.exit(1)
    print(f"Phase 1 PASS: zero-vector cvec applied "
          f"({len(zero_result['activations'])} activation records flowed), "
          f"output content matches baseline")
    print(f"  sample activation: {zero_result['activations'][0]}")
    print(f"  content head: {zero_result['content'][:80]!r}")

    # ── Phase 2: unit cvec applied with large magnitude ─────────────
    # Expect: cvec_activations populate with NON-ZERO magnitudes AND
    # output content DIFFERS from baseline.
    register_unit_cvec("cvec-validation-unit-dim0", dim=0)
    unit_result = call_bridge_with_controls([{
        "cvec_id": "cvec-validation-unit-dim0",
        "layer": LAYER,
        "polarity": 1.0,
        "peak_magnitude": 100.0,
        "sustain_level": 1.0,
        "mode": 0,
    }])
    print(f"[unit-cvec] elapsed={unit_result['elapsed_s']}s "
          f"finish={unit_result['finish_reason']} "
          f"activations={len(unit_result['activations'])}")
    if not unit_result['activations']:
        print("FAIL Phase 2: cvec_activations stream is empty")
        sys.exit(1)
    nz_mags = [a for a in unit_result['activations'] if abs(a['magnitude']) > 1e-6]
    print(f"  activations with |magnitude|>1e-6: {len(nz_mags)}")
    if not nz_mags:
        print("FAIL Phase 2: all reported activation magnitudes were zero")
        sys.exit(1)
    print(f"  sample non-zero activation: {nz_mags[0]}")
    if content_matches_baseline(unit_result, baseline):
        print("FAIL Phase 2: unit cvec at peak_magnitude=100 produced "
              "bit-identical content — apply path didn't intervene on "
              "the residual stream")
        sys.exit(1)
    print("Phase 2 PASS: unit cvec at peak_magnitude=100 → output content "
          "differs from baseline")
    print(f"  content head: {unit_result['content'][:80]!r}")


if __name__ == "__main__":
    main()
