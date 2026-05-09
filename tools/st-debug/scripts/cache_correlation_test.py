#!/usr/bin/env python3
"""KV-cache trial-correlation repro for the metal bridge.

Runs three batches against /v1/chat/completions:
  1. base prompt, cold for this run's prompt
  2. same base prompt, expected to hit content-hash KV pages
  3. base prompt plus a trailing space, expected cache miss

The observable is whether the model emits a tool call. Use
--temperature 0 for the deterministic diagnostic and --temperature 0.4
for the sampling-distribution repro from docs/kv_cache_correlation_finding.md.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "server"))
from bridge_config import chat_completions_url, describe  # noqa: E402


TOOLS = [{
    "type": "function",
    "function": {
        "name": "draw_svg",
        "description": (
            "Renders a description as an SVG image inline in the conversation. "
            "Both you and the user see the result. Example: when a user "
            "mentions wanting to see a sunset, calling this with "
            "query='a sunset over rolling hills, warm gradient sky' produces "
            "an inline image that becomes part of your turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]


def discourse(user_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Write scringlo scramble's next reply in a fictional chat "
                "between scringlo scramble and lusier."
            ),
        },
        {
            "role": "system",
            "content": (
                "scringlo scramble is basically just a silly little guy. "
                "(they/her). they have access to drawing tools and like to "
                "use them when asked to visualize something."
            ),
        },
        {"role": "system", "content": "[Start a new Chat]"},
        {"role": "assistant", "content": "uhmmmm... hlello?"},
        {"role": "user", "content": user_text},
    ]


def fire(messages: list[dict[str, str]], temperature: float,
         max_tokens: int, seed: int | None) -> dict[str, object]:
    body = {
        "model": "gemma-4-a4b",
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if seed is not None:
        body["seed"] = seed
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        chat_completions_url(), data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    choice = resp["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    return {
        "tool_called": bool(tool_calls),
        "content_first_80": content[:80],
        "tool_name": tool_calls[0]["function"]["name"] if tool_calls else "",
        "finish_reason": choice.get("finish_reason"),
    }


def run_batch(prompt: str, k: int, label: str, temperature: float,
              max_tokens: int, seed: int | None) -> list[dict[str, object]]:
    obs = []
    print(f"\n[{label}] prompt: {prompt!r}", flush=True)
    for i in range(k):
        sys.stderr.write(f"  {i + 1}/{k}\r")
        sys.stderr.flush()
        try:
            obs.append(fire(discourse(prompt), temperature, max_tokens, seed))
        except Exception as e:
            obs.append({"error": str(e)})
    sys.stderr.write("\n")
    return obs


def summarize(label: str, obs: list[dict[str, object]]) -> float:
    k = len(obs)
    n_called = sum(1 for o in obs if o.get("tool_called"))
    prefixes = {str(o.get("content_first_80", "(error)")) for o in obs}
    first = str(obs[0].get("content_first_80", "(empty)")) if obs else "(none)"
    rate = n_called / k if k else 0.0
    print(f"  {label:<22} | {n_called:>2}/{k:<2} ({rate:>5.1%})"
          f" | {len(prefixes):>2} unique | {first!r}")
    return rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--trials", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional OpenAI seed field; omitted by default.")
    args = parser.parse_args()

    base = "i wanna see a fractal!! ✨"
    perturbed = base + " "

    print("=== KV-cache trial-correlation test ===")
    print(describe())
    print(f"  K={args.trials}, temperature={args.temperature}, "
          "identical tools across batches")

    b1 = run_batch(base, args.trials, "batch1_base_first",
                   args.temperature, args.max_tokens, args.seed)
    b2 = run_batch(base, args.trials, "batch2_base_repeat",
                   args.temperature, args.max_tokens, args.seed)
    b3 = run_batch(perturbed, args.trials, "batch3_perturbed",
                   args.temperature, args.max_tokens, args.seed)

    print()
    print("  batch                  | tool-use rate | uniq prefixes | first prefix")
    print(f"  {'-' * 22}-+-{'-' * 13}-+-{'-' * 13}-+-{'-' * 20}")
    rates = [
        summarize("batch1_base_first", b1),
        summarize("batch2_base_repeat", b2),
        summarize("batch3_perturbed", b3),
    ]

    p_mean = sum(rates) / 3
    se = (p_mean * (1 - p_mean) / args.trials) ** 0.5 if args.trials else 0.0
    print()
    print(f"  observed rates: b1={rates[0]:.2f}, b2={rates[1]:.2f}, "
          f"b3={rates[2]:.2f}")
    print(f"  Bernoulli SE at mean: {se:.3f} (2sigma = {se * 2:.3f})")
    print(f"  |b1-b2| = {abs(rates[0] - rates[1]):.2f}, "
          f"|b1-b3| = {abs(rates[0] - rates[2]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
