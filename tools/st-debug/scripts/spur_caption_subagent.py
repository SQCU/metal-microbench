#!/usr/bin/env python3
"""Spur-subagent caption primitive.

When the primary agent (e.g., a query-to-svg toolcard service) is doing
slow descendant work — multi-iter SVG refinement, big code-gen, anything
where the user is waiting >5s with no UI affordance — fork off a tiny
subagent whose only job is to summarize the primary's recent activity in
one short sentence. Ship that sentence as a caption that surfaces in the
chat UI's placeholder bubble for the calling turn.

The point isn't fidelity (the caption summarizer doesn't see the model's
internal state, only its output stream). The point is to give the user
a continuously-updating answer to "what is the agent currently trying to
do?" — replacing the dead-air affordance with an animate one.

This file is the standalone PRIMITIVE. Inputs: a chunk of the primary
agent's recent output (text, code, SVG markup, anything). Output: one
short sentence summary. Cost: one chat-completions call, capped at
~32 generated tokens, expected to land in 1-2s on the bridge.

Wiring this into the toolcards plugin (so it actually surfaces during
real query-to-svg runs) is Phase-3 work — see
docs/descendant_agent_ux_spec.md design problem #3 for the integration
sketch. This script is the proof that the primitive itself is fast and
useful.

Usage:
    # one-shot: pipe an agent's output through this; get a caption back
    cat /tmp/primary_agent_output.txt | ./spur_caption_subagent.py

    # streaming: tail-follow a log file, emit captions as new content arrives
    ./spur_caption_subagent.py --tail /tmp/primary_agent.log --interval 5
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.request
import pathlib

sys.path.insert(0, str(pathlib.Path("/Users/mdot/metal-microbench/server")))
from bridge_config import chat_completions_url


# Caption agent's system prompt. Three principles applied:
#   - explicit role ("you are a caption-writer for live activity feeds")
#   - positive framing of the desired output shape (one sentence,
#     present-progressive describing the activity)
#   - a concrete example of input → caption mapping (show-don't-tell)
CAPTION_SYSTEM_PROMPT = """You are a caption-writer for a live activity \
feed. The user shows you the most recent text output from a different \
agent that is doing some longer-running work, and you write ONE short \
sentence (under 20 words) describing what that agent is currently \
doing — present-progressive, informal but precise. No preamble, no \
quoting, no commentary about your own role.

Example input:
    <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="sun" cx="50%" cy="40%" r="40%">
          <stop offset="0%" style="stop-color:#fffbe7;stop-opacity:1"/>

Example caption: drawing a radial-gradient sun in the upper-left.

Example input:
    iter 2/3: rendering 502-char SVG to PNG via playwright
    iter 2: got 502-char SVG, rasterizing
    iter 3/3: requesting refinement against vision feedback

Example caption: on iteration 3, asking the model to refine its SVG \
against the rendered preview.
"""


def caption(activity_text: str, *, max_tokens: int = 32, temperature: float = 0.4) -> dict:
    """Single bridge call. Returns {caption, elapsed_s, completion_tokens}."""
    body = json.dumps({
        "model": "gemma-4-a4b",
        "messages": [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": activity_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        chat_completions_url(), data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    elapsed = time.time() - t0
    msg = resp["choices"][0]["message"]
    return {
        "caption": (msg.get("content") or "").strip(),
        "elapsed_s": elapsed,
        "completion_tokens": resp.get("usage", {}).get("completion_tokens", 0),
        "prompt_tokens": resp.get("usage", {}).get("prompt_tokens", 0),
    }


# ── Quick built-in demo ────────────────────────────────────────────────
# Five real-shape activity-text snippets to verify the primitive
# produces useful captions and lands fast.
DEMO_INPUTS = [
    # 1. SVG markup mid-render
    """<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#ff7e5f;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#feb47b;stop-opacity:1" />
    </linearGradient>
  </defs>""",
    # 2. Tool-card-style progress events
    """target 256x256, 3 iter(s)
iter 1/3: requesting SVG
iter 1: got 502-char SVG, rasterizing
iter 1: done in 7.0s (502 chars)
iter 2/3: requesting refinement against vision feedback""",
    # 3. Python code mid-write
    """def refine_svg(prev_svg: str, feedback: str) -> str:
    \"\"\"Take the previous SVG attempt and the model's prose feedback,
    return an updated SVG that addresses the feedback.\"\"\"
    prompt = f\"Previous attempt:\\n{prev_svg}\\n\\nFeedback:\\n{feedback}\\n\\n\"
    prompt += \"Updated SVG:\"
    response = bridge.chat([
        {\"role\": \"user\", \"content\": prompt},
    ])""",
    # 4. Prose mid-think
    """The user mentioned a fractal — likely they're thinking of something \
visually rich and recursive. I should consider whether to produce a Mandelbrot-\
like nested pattern, or a simpler self-similar tree, or a Sierpinski-style \
triangle. Given the casual register ("i wanna see"), simpler is probably""",
    # 5. JSON tool-call shape mid-emit
    """{"name": "render_svg", "arguments": {"svg": "<svg width='200' height='200'>\\n  <circle cx='100' cy='100' r='80' fill='red' />""",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true",
                    help="Run the built-in five-input demo.")
    ap.add_argument("--input-file", type=pathlib.Path,
                    help="Read activity-text from a file instead of stdin.")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    if args.demo:
        print("=== caption-subagent demo ===")
        print(f"  bridge: {chat_completions_url()}")
        print(f"  budget: max_tokens={args.max_tokens}, temp={args.temperature}")
        print()
        elapsed_total = 0.0
        for i, snippet in enumerate(DEMO_INPUTS, 1):
            short_snip = snippet[:80].replace("\n", " ⏎ ")
            print(f"  [{i}] input ({len(snippet)} chars): {short_snip}…")
            result = caption(snippet, max_tokens=args.max_tokens,
                              temperature=args.temperature)
            elapsed_total += result["elapsed_s"]
            print(f"      caption ({result['elapsed_s']:.2f}s, "
                  f"{result['completion_tokens']} tok): "
                  f"{result['caption']!r}")
            print()
        print(f"  total wall: {elapsed_total:.1f}s for {len(DEMO_INPUTS)} captions")
        print(f"  per-caption mean: {elapsed_total/len(DEMO_INPUTS):.2f}s")
        return 0

    # Single-input mode (stdin or file).
    if args.input_file:
        text = args.input_file.read_text()
    else:
        text = sys.stdin.read()
    if not text.strip():
        print("error: no input", file=sys.stderr)
        return 1
    result = caption(text, max_tokens=args.max_tokens, temperature=args.temperature)
    print(result["caption"])
    print(f"({result['elapsed_s']:.2f}s, {result['completion_tokens']} tok)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
