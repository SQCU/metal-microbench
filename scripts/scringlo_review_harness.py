#!/usr/bin/env python3
"""Dual-persona code-review harness for gemma-4-a4b via the local bridge.

Pulls a code/finding excerpt and asks gemma-4-a4b for two reviews:
  1. "sober" — direct senior-engineer prompt, no character.
  2. "scringlo" — the scringlo-scramble persona harness (silly little guy
     who happens to know nodejs/frontend/PPO well).

The contrast surfaces what the persona scaffolding does or doesn't
degrade about analytical capacity. Useful as a third reviewer alongside
codex/opus; cheap because gemma-4-a4b runs locally on the metal bridge.

Bridge URL: read from server/config.toml via server/bridge_config.py.
Default after the 2026-05-07 config unification: http://127.0.0.1:8001.

Usage:
  # Inline excerpt
  python scringlo_review_harness.py --label "openai.js dispatch" --code-file /tmp/snippet.js

  # Stdin
  cat snippet.txt | python scringlo_review_harness.py --label "frob"

  # Multiple excerpts via JSON manifest
  python scringlo_review_harness.py --manifest excerpts.json
  # excerpts.json: [{"label": "...", "code": "..."}, ...]

Both personas write to stdout as concatenated markdown. Per-persona timing
goes to stderr.

Origin: /tmp/st_review_harness.py (2026-04-28). Promoted to
metal-microbench/scripts on 2026-04-29 because we keep wanting to invoke
the dual-persona pattern across different review tasks (SillyTavern
fitness review, neural-kcut training stationarity, etc.) and the tmp
copy kept getting GC'd.
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request

# Read bridge URL from the canonical config (server/config.toml).
# Env BRIDGE_URL / QUANT_BRIDGE_URL / GEMMA_BRIDGE still override.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "server"))
from bridge_config import BRIDGE_URL as DEFAULT_BRIDGE  # noqa: E402


def chat(messages, *, max_tokens=1800, temperature=0.4, bridge=DEFAULT_BRIDGE):
    """One round-trip to the bridge's OpenAI-compatible endpoint."""
    data = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{bridge}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    dt = time.time() - t0
    msg = resp["choices"][0]["message"]
    text = msg.get("content") or ""
    usage = resp.get("usage", {})
    tok_per_s = usage.get("completion_tokens", 0) / dt if dt > 0 else 0
    print(
        f"  [done in {dt:.1f}s, {usage.get('completion_tokens', 0)} tok, "
        f"{tok_per_s:.1f} tok/s]",
        file=sys.stderr,
    )
    return text


def sober(label, code, *, bridge=DEFAULT_BRIDGE, max_tokens=1800, temperature=0.4,
          system_prompt=None):
    """Direct senior-engineer reviewer.

    The default system prompt is shaped for codebase fitness reviews; pass
    `system_prompt` to swap in a different framing (e.g., training-pipeline
    stationarity, RL gradient hygiene, etc.).
    """
    if system_prompt is None:
        system_prompt = (
            "You are a senior engineer doing a 2026-era fitness review of "
            "the excerpt below. Rate it on three axes:\n"
            "  (a) actual composability — can an LLM agent or human edit "
            "this to add features without fighting the structure?\n"
            "  (b) baroqueness — is the structure self-evident or does it "
            "require archaeology to understand?\n"
            "  (c) technical debt / unmaintainability — global state, "
            "brittle implicit contracts, pre-2026 patterns.\n"
            "Be concrete. Quote specific lines/identifiers. Don't praise. "
            "Don't pad. Roughly 250-400 words."
        )
    return chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": f"## {label}\n\n```\n{code}\n```\n\nReview the excerpt above."},
        ],
        bridge=bridge, max_tokens=max_tokens, temperature=temperature,
    )


def scringlo(label, code, *, bridge=DEFAULT_BRIDGE, max_tokens=1800, temperature=0.4):
    """The scringlo-scramble persona harness.

    Matches the chat-template the bridge logs surfaced — silly little guy
    persona, but trained 2025+ so genuinely competent at code review.

    Note: persona text is fixed because changes to it shift response
    quality in non-obvious ways. If you need a different framing, prefer
    a separate persona function over editing this one.
    """
    return chat(
        [
            {"role": "system",
             "content": "Write scringlo scramble's next reply in a fictional chat between scringlo scramble and lusier."},
            {"role": "system",
             "content": "scringlo scramble is basically just a silly little guy. (they/her) — but is also surprisingly competent at code review when asked, since they were trained in 2025 and learned a lot about nodejs + frontend code."},
            {"role": "system", "content": "[Start a new Chat]"},
            {"role": "assistant", "content": "uhmmmm... hlello?"},
            {"role": "user",
             "content": "hi scringlo!! i wanna review some code with u, can u look at it n give me a 2026-era fitness review? rate composability (can agents edit it), baroqueness (easy to figure out?), and tech debt (jquery, brittle stuff). be honest n specific even tho ur a silly guy"},
            {"role": "assistant",
             "content": "ohhhh okay yes!! i will look real careful!! i am Smart Actually i promise i can do code review even tho im sillyyyy ✨ what is the codey-code"},
            {"role": "user",
             "content": f"ok here it is, it's the {label} part:\n\n```\n{code}\n```\n\nlemme have ur takes scringlo!! be real with me!!"},
        ],
        bridge=bridge, max_tokens=max_tokens, temperature=temperature,
    )


def run_one(label, code, *, bridge=DEFAULT_BRIDGE, sober_system=None,
            max_tokens=1800, temperature=0.4):
    """Returns dict {"label", "sober", "scringlo"}."""
    print(f"\n{'=' * 70}\n  EXCERPT: {label}\n{'=' * 70}", file=sys.stderr)
    print(f"\n--- gemma-4-a4b (sober) on {label}:\n", file=sys.stderr)
    s = sober(label, code, bridge=bridge, max_tokens=max_tokens,
              temperature=temperature, system_prompt=sober_system)
    print(f"\n--- scringlo scramble on {label}:\n", file=sys.stderr)
    p = scringlo(label, code, bridge=bridge, max_tokens=max_tokens,
                 temperature=temperature)
    return {"label": label, "sober": s, "scringlo": p}


def _main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", help="Label for stdin/--code-file excerpt.")
    ap.add_argument("--code-file", help="Path to file containing the excerpt.")
    ap.add_argument("--manifest", help="JSON file with [{label, code}, ...].")
    ap.add_argument("--bridge", default=DEFAULT_BRIDGE,
                    help=f"Bridge base URL (default: {DEFAULT_BRIDGE}).")
    ap.add_argument("--sober-system",
                    help="Override the sober persona's system prompt.")
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--format", choices=["json", "markdown"], default="markdown",
                    help="Output format on stdout.")
    args = ap.parse_args()

    excerpts = []
    if args.manifest:
        with open(args.manifest) as f:
            excerpts = json.load(f)
    elif args.code_file:
        with open(args.code_file) as f:
            excerpts = [{"label": args.label or args.code_file, "code": f.read()}]
    else:
        # stdin
        code = sys.stdin.read()
        if not code.strip():
            ap.error("No input — pass --code-file, --manifest, or pipe via stdin.")
        excerpts = [{"label": args.label or "(stdin)", "code": code}]

    reviews = [
        run_one(e["label"], e["code"], bridge=args.bridge,
                sober_system=args.sober_system,
                max_tokens=args.max_tokens, temperature=args.temperature)
        for e in excerpts
    ]

    if args.format == "json":
        print(json.dumps(reviews, indent=2))
    else:
        for r in reviews:
            print(f"\n# {r['label']}\n")
            print(f"## sober (gemma-4-a4b)\n\n{r['sober']}\n")
            print(f"## scringlo scramble (gemma-4-a4b, persona)\n\n{r['scringlo']}\n")


if __name__ == "__main__":
    _main()
