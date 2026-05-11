#!/usr/bin/env python3
"""Bit-identity regression gate for cvec instrumentation work.

Captures top-K logprobs of the first N tokens generated for a fixed
prompt + fixed seed. Two runs of the same code must produce identical
top-K outputs (bit-equal float32). Before any change to the engine's
AR-step or sampling path, run with --record to capture a baseline JSON;
after the change, run with --verify to check the new output bit-equals
the baseline.

Logprobs are deterministic given the same prompt + same engine state —
they come from the model forward pass, not from sampling — so the
test is independent of seed if the test only inspects logprobs (not
sampled tokens). We do still pin the seed so the sampled tokens are
themselves stable, which keeps the prompt-extension path deterministic
across the N steps we measure.

Usage:
    python3 baseline_logprobs.py --record   # write baseline.json
    python3 baseline_logprobs.py --verify   # compare to baseline.json
"""
import argparse, json, struct, sys, time
from pathlib import Path
import urllib.request

BRIDGE = "http://127.0.0.1:8001"
PROMPT = (
    "Write the first sentence of a short story about a cartographer "
    "who maps the spaces between bookshelves."
)
N_TOKENS = 12      # capture top-K logprobs of first N generated tokens
TOP_K = 5          # top-K logprobs to capture per token
SEED = 42
HERE = Path(__file__).parent
BASELINE_PATH = HERE / "baseline.json"


def call_bridge(capture_cvec=False):
    body = {
        "model": "gemma-4-a4b",
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "max_tokens": N_TOKENS,
        "temperature": 0.01,   # near-greedy; 0.0 forbidden per project policy
        "seed": SEED,
        "logprobs": True,
        "top_logprobs": TOP_K,
    }
    if capture_cvec:
        body["capture_cvec_activations"] = True
    req = urllib.request.Request(
        f"{BRIDGE}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    elapsed = time.time() - t0
    choice = resp["choices"][0]
    msg = choice["message"]
    lp_blocks = (choice.get("logprobs") or {}).get("content") or []
    out = {
        "prompt": PROMPT,
        "n_tokens_requested": N_TOKENS,
        "top_k": TOP_K,
        "seed": SEED,
        "content": msg.get("content", ""),
        "finish_reason": choice.get("finish_reason"),
        "elapsed_s": round(elapsed, 3),
        "tokens": [
            {
                "token": entry.get("token"),
                "logprob": _f32_hex(entry.get("logprob")),
                "top": [
                    {"token": t.get("token"), "logprob": _f32_hex(t.get("logprob"))}
                    for t in (entry.get("top_logprobs") or [])
                ],
            }
            for entry in lp_blocks[:N_TOKENS]
        ],
    }
    return out


def _f32_hex(x):
    """Encode a float32 as its bit pattern (8-hex-char string). Bit-equal
    is the gate; numeric == on float can mask single-ULP perturbations
    that flag a real determinism break."""
    if x is None:
        return None
    return struct.pack("<f", float(x)).hex()


def record():
    out = call_bridge()
    BASELINE_PATH.write_text(json.dumps(out, indent=2))
    print(f"baseline recorded → {BASELINE_PATH}")
    print(f"  tokens captured: {len(out['tokens'])}")
    print(f"  first token logprob: {out['tokens'][0]['logprob'] if out['tokens'] else '(none)'}")
    print(f"  content head: {out['content'][:120]!r}")


def verify(capture_cvec=False):
    if not BASELINE_PATH.exists():
        print(f"FAIL: no baseline at {BASELINE_PATH} — run --record first")
        sys.exit(2)
    baseline = json.loads(BASELINE_PATH.read_text())
    now = call_bridge(capture_cvec=capture_cvec)
    diffs = []
    if baseline["content"] != now["content"]:
        diffs.append(
            f"content differs:\n  baseline: {baseline['content'][:200]!r}\n  now:      {now['content'][:200]!r}"
        )
    for i, (b, n) in enumerate(zip(baseline["tokens"], now["tokens"])):
        if b["token"] != n["token"]:
            diffs.append(f"token[{i}] differs: baseline={b['token']!r} now={n['token']!r}")
        if b["logprob"] != n["logprob"]:
            diffs.append(f"token[{i}] logprob differs: baseline={b['logprob']} now={n['logprob']}")
        for k, (bt, nt) in enumerate(zip(b["top"], n["top"])):
            if bt["token"] != nt["token"] or bt["logprob"] != nt["logprob"]:
                diffs.append(
                    f"token[{i}].top[{k}] differs: "
                    f"baseline=({bt['token']!r}, {bt['logprob']}) "
                    f"now=({nt['token']!r}, {nt['logprob']})"
                )
    if len(baseline["tokens"]) != len(now["tokens"]):
        diffs.append(
            f"token count differs: baseline={len(baseline['tokens'])} now={len(now['tokens'])}"
        )
    if diffs:
        print(f"FAIL: bit-identity check found {len(diffs)} differences:")
        for d in diffs[:20]:
            print(f"  • {d}")
        sys.exit(1)
    print(f"PASS: bit-identity verified across {len(now['tokens'])} tokens × {TOP_K + 1} logprobs each")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--record", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--verify-with-capture", action="store_true",
                   help="verify with capture_cvec_activations=true in the "
                        "request body (instrumented-but-unused regression "
                        "gate; the apply path runs but no cvec is "
                        "registered so the queue stays empty)")
    args = p.parse_args()
    if args.record:
        record()
    elif args.verify:
        verify()
    elif args.verify_with_capture:
        verify(capture_cvec=True)
    else:
        print("usage: --record | --verify | --verify-with-capture")
        sys.exit(2)
