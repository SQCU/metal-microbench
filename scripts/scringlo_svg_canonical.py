#!/usr/bin/env python3
"""Reproduce the old SillyTavern-fork SVG-from-query workflow.

Uses the scringlo-scramble persona discourse to ask the model to render
an SVG via a tool call. Pre-2026-05-07 this worked (slowly) on bridge
versions that lacked the M:K + atomic-construction + content-hash KV
pool refactors. Then we hit the regression where tool_call markers
were BPE-split silently. After the fix chain (atomic stop sequence +
SPECIAL_TOKENS auto-load + server-side tool_calls extraction) it
should work fast AND produce OAI-shape tool_calls[].

Verifies:
  - tool_calls[] returned in OAI shape
  - function name matches
  - arguments parse as valid JSON
  - SVG content is structurally plausible
"""
import json
import sys
import time
import urllib.request
import pathlib

# Use the canonical config-driven URL.
sys.path.insert(0, str(pathlib.Path("/Users/mdot/metal-microbench/server")))
from bridge_config import BRIDGE_URL, chat_completions_url

# The scringlo-scramble canonical persona discourse, repurposed from
# code-review to SVG-rendering-via-subagent. The pattern matches what
# the SillyTavern-fork harness was producing pre-regression: persona
# scaffolding (3 system messages + assistant priming + user question),
# tools[] for render_svg, request asking for a specific scene.
CANONICAL_DISCOURSE = [
    {"role": "system",
     "content": "Write scringlo scramble's next reply in a fictional chat between scringlo scramble and lusier."},
    {"role": "system",
     "content": "scringlo scramble is basically just a silly little guy. (they/her) — but has a render_svg tool available and is happy to use it when asked to draw or visualize something."},
    {"role": "system", "content": "[Start a new Chat]"},
    {"role": "assistant", "content": "uhmmmm... hlello?"},
    {"role": "user",
     "content": "hi scringlo!! can u draw me a pretty sunset scene? use the render_svg tool — i want orange sky w sun + a couple lil hills 😊"},
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "render_svg",
        "description": "render an SVG markup string and display it to the user; use for any drawing, diagram, or visual",
        "parameters": {
            "type": "object",
            "properties": {
                "svg": {
                    "type": "string",
                    "description": "the complete SVG markup, including <svg>...</svg> tags",
                },
            },
            "required": ["svg"],
        },
    },
}]


def fire_chat() -> dict:
    """Fire one chat against the bridge with the canonical discourse + tools.

    Per the generation-config moratorium (lint_generation_config.mjs):
    no temperature / no max_tokens at the caller layer. Bridge defaults
    apply (temperature=1.0 + EOS termination). This is a regression
    smoke-test; the assertions downstream don't depend on sampling
    settings.
    """
    body = json.dumps({
        "model": "gemma-4-a4b",
        "messages": CANONICAL_DISCOURSE,
        "tools": TOOLS,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        chat_completions_url(),
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    elapsed = time.time() - t0
    return resp, elapsed


def main() -> int:
    print(f"=== scringlo-canonical SVG-via-subagent regression test ===")
    print(f"  bridge: {BRIDGE_URL}")
    print(f"  discourse: {len(CANONICAL_DISCOURSE)} messages "
          f"({sum(1 for m in CANONICAL_DISCOURSE if m['role']=='system')} system, "
          f"{sum(1 for m in CANONICAL_DISCOURSE if m['role']=='user')} user, "
          f"{sum(1 for m in CANONICAL_DISCOURSE if m['role']=='assistant')} assistant)")
    print(f"  tools:     {len(TOOLS)} ({TOOLS[0]['function']['name']})")
    print()

    resp, elapsed = fire_chat()
    choice = resp["choices"][0]
    msg = choice["message"]
    usage = resp.get("usage", {})
    finish = choice.get("finish_reason")

    completion_tokens = usage.get("completion_tokens", 0)
    tok_per_s = completion_tokens / elapsed if elapsed > 0 else 0

    print(f"  wall:        {elapsed:.2f}s")
    print(f"  finish:      {finish}")
    print(f"  prompt tok:  {usage.get('prompt_tokens', 0)}")
    print(f"  output tok:  {completion_tokens}  ({tok_per_s:.1f} tok/s)")
    print(f"  cache hits:  {usage.get('cache_hits', 0)}/{usage.get('cache_misses', 0)}")
    print()

    # Assertion 1: finish_reason should be "tool_calls"
    if finish != "tool_calls":
        print(f"  ✗ FAIL: finish_reason should be 'tool_calls', got {finish!r}")
        if msg.get("content"):
            print(f"  raw content (first 400b): {msg['content'][:400]!r}")
        return 1
    print(f"  ✓ finish_reason = 'tool_calls'")

    # Assertion 2: message.tool_calls present
    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        print(f"  ✗ FAIL: message.tool_calls missing or empty")
        return 2
    print(f"  ✓ tool_calls[] present, n={len(tool_calls)}")

    tc = tool_calls[0]

    # Assertion 3: function name matches
    fn_name = tc.get("function", {}).get("name")
    if fn_name != "render_svg":
        print(f"  ✗ FAIL: function name should be 'render_svg', got {fn_name!r}")
        return 3
    print(f"  ✓ function.name = 'render_svg'")

    # Assertion 4: arguments parse as JSON
    args_str = tc.get("function", {}).get("arguments", "")
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError as e:
        print(f"  ✗ FAIL: arguments not valid JSON: {e}")
        print(f"  raw arguments: {args_str!r}")
        return 4
    print(f"  ✓ arguments parse as JSON")

    # Assertion 5: svg arg is a non-empty string with <svg ...> bones
    svg = args.get("svg", "")
    if not isinstance(svg, str) or not svg.strip():
        print(f"  ✗ FAIL: svg argument is missing or empty")
        return 5
    if "<svg" not in svg or "</svg>" not in svg:
        print(f"  ✗ FAIL: svg argument doesn't look like SVG markup")
        print(f"  got: {svg[:300]!r}")
        return 6
    print(f"  ✓ svg argument is structurally plausible "
          f"({len(svg)} chars, has <svg> + </svg>)")

    # Display the SVG content (so the user can sanity-check it)
    print()
    print("─" * 72)
    print("  rendered SVG markup:")
    print("─" * 72)
    print(svg)
    print("─" * 72)

    # Optionally save it for visual inspection
    out_path = pathlib.Path("/tmp/scringlo_svg.svg")
    out_path.write_text(svg)
    print(f"\n  saved to {out_path} — open it in a browser to view the rendered scene")

    print()
    print("  ALL CHECKS PASS — SillyTavern-fork SVG-from-query workflow restored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
