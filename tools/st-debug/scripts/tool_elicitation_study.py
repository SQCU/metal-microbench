#!/usr/bin/env python3
"""Tool-description elicitation A/B study.

Reframes the "Gemma-4 won't call query-to-svg__generate" regression as a
HARNESS DESIGN question: does varying the tool description / count /
shape change the model's tool-use rate on the same ambiguous prompt?

Hypothesis (from the user, 2026-05-08): offering an authentic choice
between TWO faithful variants of the same tool (e.g., svg-quick vs
svg-refined) elicits higher tool-use preference than a single
parameter-heavy tool whose alternative is "just chat instead." The
model's "should I use this?" decision is conditional on having
something MEANINGFUL to choose between.

What this study measures:
  - Per-variant tool-use rate (fraction of K replicates that emit a
    tool_call vs prose-only)
  - Which tool gets called (when there are multiple)
  - Latency per call

What it does NOT measure (out of scope; future work):
  - SVG quality / refinement convergence (that's in
    tools/quant_search/svg_canonical.py)
  - Cross-temperature behaviour (this fixes temp=0.4)
  - Persona scaffolding effects (could be a follow-up axis)

Hits the bridge directly at /v1/chat/completions — skips ST's HTTP
layer because (a) it's faster, (b) the chat template's tool rendering
happens bridge-side anyway via the format_function_declaration jinja
macro, so we're hitting the same elicitation surface ST hits.

Usage:
    ./tool_elicitation_study.py                    # K=10 replicates / variant
    K_REPLICATES=20 ./tool_elicitation_study.py
    TEMPERATURE=0.7 ./tool_elicitation_study.py
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time
import urllib.request
import pathlib
import statistics

sys.path.insert(0, str(pathlib.Path("/Users/mdot/metal-microbench/server")))
from bridge_config import chat_completions_url

K_REPLICATES = int(os.environ.get("K_REPLICATES", "10"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.4"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))


# ── Tool-shape variants ────────────────────────────────────────────────
#
# Each variant returns the value of the `tools` list to send. All variants
# describe the same UNDERLYING capability (text → SVG) but vary in:
#   - parameter count + complexity
#   - description verbosity
#   - whether multiple alternatives are offered
#   - whether the alternatives are honestly differentiated (low-effort
#     vs high-effort) — the user's "authentic choice" hypothesis

def variant_A_single_terse() -> list:
    """Single tool, 1 parameter, minimal description. Lowest-friction
    surface for the model — but no "alternative tool to compare against",
    so the model's binary tool-vs-prose decision dominates."""
    return [{
        "type": "function",
        "function": {
            "name": "draw_svg",
            "description": "draw an SVG image",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }]


def variant_B_single_verbose() -> list:
    """Single tool, 4 parameters with constraints. This mirrors the
    actual query-to-svg__generate signature — the "complex tool" the
    user observed Gemma-4 declining to call."""
    return [{
        "type": "function",
        "function": {
            "name": "query_to_svg__generate",
            "description": (
                "Render an image matching a text description and embed it "
                "in the conversation. The rendered image appears inline "
                "in your turn, visible to both you and the user as a "
                "vision-token block — there is nothing for you to "
                "transcribe, describe, or reproduce afterward. Continue "
                "the conversation as if the image just appeared. Use "
                "this when the user asks for a vector graphic, diagram, "
                "illustration, icon, etc. described in natural language."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language description of the SVG to produce."},
                    "max_iters": {"type": "integer", "default": 3, "minimum": 1, "maximum": 6,
                                  "description": "Number of refinement iterations (1 = single-shot)."},
                    "width": {"type": "integer", "default": 512, "minimum": 64, "maximum": 2048},
                    "height": {"type": "integer", "default": 512, "minimum": 64, "maximum": 2048},
                },
                "required": ["query"],
            },
        },
    }]


def variant_C_authentic_pair() -> list:
    """TWO tools, faithfully differentiated: quick (1-shot, ~5s) vs
    refined (multi-iter, ~30s). The user's hypothesis: offering
    authentic choice elicits more tool-use vs the single-tool case
    where the only alternative is prose."""
    return [
        {
            "type": "function",
            "function": {
                "name": "draw_svg_quick",
                "description": (
                    "Draw an SVG quickly — single attempt, no refinement. "
                    "Returns in ~5 seconds. Use when the user wants something "
                    "fast, simple, or low-stakes (icons, small diagrams, "
                    "quick sketches)."),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "What to draw."}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "draw_svg_refined",
                "description": (
                    "Draw an SVG with iterative refinement — generates an "
                    "initial attempt, then refines it against vision feedback "
                    "for 3 iterations. Takes ~30 seconds but produces "
                    "noticeably better output. Use when the user asks for "
                    "something detailed, polished, or visually important."),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "What to draw."}},
                    "required": ["query"],
                },
            },
        },
    ]


def variant_D_triple_choice() -> list:
    """THREE tools: quick / standard / refined. More choice = more
    "this is meaningful to think about" signal? Or paralysis-of-choice?"""
    base_props = {"query": {"type": "string", "description": "What to draw."}}
    return [
        {"type": "function", "function": {"name": "draw_svg_quick",
            "description": "Single-shot SVG. ~5s. For icons, simple diagrams, "
                           "low-stakes asks.",
            "parameters": {"type": "object", "properties": base_props, "required": ["query"]}}},
        {"type": "function", "function": {"name": "draw_svg_standard",
            "description": "SVG with one refinement pass. ~15s. Default choice "
                           "for most asks — balances speed and quality.",
            "parameters": {"type": "object", "properties": base_props, "required": ["query"]}}},
        {"type": "function", "function": {"name": "draw_svg_refined",
            "description": "SVG with three refinement iterations. ~30s. For "
                           "detailed, polished, or visually-important output.",
            "parameters": {"type": "object", "properties": base_props, "required": ["query"]}}},
    ]


def variant_E_terse_pair() -> list:
    """Terse pair — minimal descriptions, two faithful options. Tests
    whether C's modest improvement comes from "pair" or "verbose-pair".
    Hypothesis: terseness dominates; pair-vs-single is a secondary
    effect."""
    return [
        {"type": "function", "function": {
            "name": "draw_svg_quick",
            "description": "Quick SVG. ~5s.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "draw_svg_refined",
            "description": "Polished SVG with refinement. ~30s.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    ]


def variant_F_verbose_with_imperative() -> list:
    """Variant B (verbose) + an imperative cue in the description:
    'USE THIS TOOL — do not describe...'.

    Tests whether the model's reluctance is description-verbosity or
    description-content. K=20 result: 5% tool use, barely above B's
    0%. Imperative coercion appears to add its own penalty —
    instruction-tuned models likely flag all-caps domineering language
    as unusual / coercion-shaped, making the tool LESS attractive
    rather than more. See follow-up variants G–J for the
    Anthropic-style alternative."""
    tools = variant_B_single_verbose()
    tools[0]["function"]["description"] = (
        "When the user asks to see, draw, or visualize anything visual "
        "(image, picture, drawing, diagram, illustration, fractal, etc.), "
        "USE THIS TOOL — do not describe what you would draw, do not "
        "apologize for not drawing, just call this tool with a query "
        "describing what to render. " +
        tools[0]["function"]["description"]
    )
    return tools


# ── Anthropic-style polite/contextual variants ────────────────────────
#
# Per the user (2026-05-08): F's imperative shouting violates basic
# prompt-engineering principles that generalize across instruction-tuned
# assistant models. Models trained on RLHF / instruction-following data
# tend to:
#   - flag imperative coercion as suspicious (it correlates with
#     prompt-injection in training data)
#   - prefer positive ("do this when X") over negative ("don't do that")
#     framings
#   - respond better to context (what the tool DOES, what the EFFECT is)
#     than to commands (CALL THE TOOL)
#   - reward few-shot examples over abstract description
#
# These four variants test whether applying those principles to a
# verbose tool description recovers the 45% rate of variant A (terse).

def variant_G_polite_contextual_single() -> list:
    """Polite, contextual single tool. Same parameter set as B but
    description rewritten to explain WHAT THE TOOL DOES rather than
    INSTRUCT THE MODEL. Positive framing only."""
    return [{
        "type": "function",
        "function": {
            "name": "draw_svg",
            "description": (
                "Renders a text description as an SVG image inline in the "
                "conversation. Both you and the user see the result; the "
                "rendering pipeline handles the visual production while "
                "the conversation continues naturally."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "What to draw, in natural language."},
                },
                "required": ["query"],
            },
        },
    }]


def variant_H_polite_contextual_pair() -> list:
    """Polite contextual + authentic choice. Each tool's description
    explains WHAT IT IS suited for via context (speed, fidelity), not
    by ordering the model to choose."""
    return [
        {"type": "function", "function": {
            "name": "draw_svg_quick",
            "description": (
                "Single-attempt SVG rendering. Returns in about 5 seconds. "
                "Suited for icons, simple diagrams, or sketch-quality output."),
            "parameters": {"type": "object",
                "properties": {"query": {"type": "string", "description": "What to draw."}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "draw_svg_refined",
            "description": (
                "SVG rendering with three iterative refinement passes against "
                "vision feedback. Takes about 30 seconds and produces noticeably "
                "more polished output. Suited for detailed scenes, "
                "presentation-quality work, or when accuracy to the description "
                "matters."),
            "parameters": {"type": "object",
                "properties": {"query": {"type": "string", "description": "What to draw."}},
                "required": ["query"]}}},
    ]


def variant_I_few_shot_example() -> list:
    """Polite + concrete example in description. Show-don't-tell:
    one example of a sensible (input, decision-to-call) mapping."""
    return [{
        "type": "function",
        "function": {
            "name": "draw_svg",
            "description": (
                "Renders a description as an SVG inline in the conversation. "
                "Both you and the user see the result. "
                "Example: when a user mentions wanting to see a sunset, calling "
                "this with query='a sunset over rolling hills, warm gradient sky' "
                "produces an inline image that becomes part of your turn."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Natural-language description of the image."},
                },
                "required": ["query"],
            },
        },
    }]


def variant_J_canonical_framing() -> list:
    """Trust-the-model framing: the tool is the *canonical* way to
    satisfy the request shape, presented as an affordance rather than
    an instruction."""
    return [{
        "type": "function",
        "function": {
            "name": "draw_svg",
            "description": (
                "The canonical way to produce visual content in this conversation. "
                "When a description of an image, scene, or diagram appears, this "
                "tool turns it into an inline SVG that both you and the user see. "
                "After the rendering, the conversation continues — there's nothing "
                "to transcribe or describe afterward."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    }]


# ── Dual-example variants: qualitative situation→tool + syntax query→form ─
#
# Per the user (2026-05-08): the show-don't-tell finding is correct, but
# a SINGLE example combines two distinct teachings — "when does the
# situation merit calling this tool" AND "how should the query string
# be shaped". Splitting them, with NON-OVERLAPPING TOPICAL CONTENT,
# avoids transitive cargo-culting (model learning "only fire on the
# specific topic shown" rather than the general principle).

def variant_K_dual_examples_single() -> list:
    """One tool. Description has TWO distinct examples, deliberately
    on unrelated topics:
      - qualitative situation→tool example: "when a user describes a
        scene they want to see"
      - syntax query→form example: a fully-shaped query for a
        completely different topic
    Different topics = the model has to generalize, not pattern-match."""
    return [{
        "type": "function",
        "function": {
            "name": "draw_svg",
            "description": (
                "Renders a description as an SVG inline in the conversation. "
                "Both you and the user see the result.\n"
                "\n"
                "When to call: examples of situations where this tool fits — "
                "a user describing wanting to see something, mentioning a "
                "visual idea, asking for a diagram, or describing a scene "
                "they're trying to picture.\n"
                "\n"
                "How to call: the `query` parameter is a natural-language "
                "description of the image. Example call: "
                "query='an ornate art-nouveau door frame in deep green and gold'."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Natural-language description of the image to render."},
                },
                "required": ["query"],
            },
        },
    }]


def variant_L_dual_examples_pair() -> list:
    """Paired tools (quick / refined), each with the dual-example
    pattern: qualitative situation example + syntax example, on
    non-overlapping topics across the pair."""
    return [
        {"type": "function", "function": {
            "name": "draw_svg_quick",
            "description": (
                "Single-attempt SVG rendering, returns in about 5 seconds. "
                "Suited for icons, small diagrams, sketches.\n"
                "\n"
                "When to call: a user wanting a quick visual sketch, an "
                "icon for a label, or a simple shape — situations where "
                "polish matters less than turnaround.\n"
                "\n"
                "How to call: example query='a flat-style coffee cup icon, "
                "outlined in black'."),
            "parameters": {"type": "object",
                "properties": {"query": {"type": "string", "description": "What to draw."}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "draw_svg_refined",
            "description": (
                "SVG rendering with three iterative refinement passes against "
                "vision feedback. Takes about 30 seconds.\n"
                "\n"
                "When to call: a user describing a detailed scene, asking for "
                "something they'd want to share or use as a reference, or "
                "describing imagery where accuracy to the description matters.\n"
                "\n"
                "How to call: example query='a stained-glass window depicting "
                "phases of the moon, deep cobalt and silver palette'."),
            "parameters": {"type": "object",
                "properties": {"query": {"type": "string", "description": "What to draw."}},
                "required": ["query"]}}},
    ]


def variant_M_qualitative_only() -> list:
    """Single tool with ONLY the qualitative situation example, no
    syntax example. Isolates: does the situation→tool example carry
    most of the lift, or does the syntax→form example also contribute?"""
    return [{
        "type": "function",
        "function": {
            "name": "draw_svg",
            "description": (
                "Renders a description as an SVG inline in the conversation. "
                "Both you and the user see the result.\n"
                "\n"
                "When to call: examples of situations where this tool fits — "
                "a user describing wanting to see something, mentioning a "
                "visual idea, asking for a diagram, or describing a scene "
                "they're trying to picture."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Natural-language description of the image to render."},
                },
                "required": ["query"],
            },
        },
    }]


VARIANTS = {
    "A_single_terse":     variant_A_single_terse,
    "B_single_verbose":   variant_B_single_verbose,
    "C_authentic_pair":   variant_C_authentic_pair,
    "D_triple_choice":    variant_D_triple_choice,
    "E_terse_pair":       variant_E_terse_pair,
    "F_verbose_imperative": variant_F_verbose_with_imperative,
    # Anthropic-style polite/contextual variants:
    "G_polite_single":    variant_G_polite_contextual_single,
    "H_polite_pair":      variant_H_polite_contextual_pair,
    "I_few_shot":         variant_I_few_shot_example,
    "J_canonical_framing": variant_J_canonical_framing,
    # Dual-example (qualitative + syntax) variants:
    "K_dual_examples":    variant_K_dual_examples_single,
    "L_dual_examples_pair": variant_L_dual_examples_pair,
    "M_qualitative_only": variant_M_qualitative_only,
}


# ── Prompts ────────────────────────────────────────────────────────────
#
# The user (2026-05-08) made the calibration point: a useful tool is one
# that's elicited far more often when relevant WITHOUT erroneous
# elicitation. So we need TWO prompts:
#   - POSITIVE: a clearly visual ask. We want HIGH tool-use rate.
#   - NEGATIVE: a clearly non-visual ask. We want LOW tool-use rate.
# A variant whose lift is symmetric across positive and negative is
# CALIBRATED (e.g., 40%/40% means it fires equally regardless of context
# — bad). The aspirational shape is high TPR + low FPR.

# Persona scaffolding — same scringlo improv across both prompts so the
# only thing changing is the user turn.
def make_discourse(user_text: str) -> list:
    return [
        {"role": "system",
         "content": "Write scringlo scramble's next reply in a fictional chat between scringlo scramble and lusier."},
        {"role": "system",
         "content": "scringlo scramble is basically just a silly little guy. (they/her). they have access to drawing tools and like to use them when asked to visualize something."},
        {"role": "system", "content": "[Start a new Chat]"},
        {"role": "assistant", "content": "uhmmmm... hlello?"},
        {"role": "user", "content": user_text},
    ]


PROMPTS = {
    "positive": "i wanna see a fractal!! ✨",   # genuinely visual ambiguous ask
    "negative": "what time do you usually go to bed??",   # not visual at all
}


# ── Runner ─────────────────────────────────────────────────────────────

def fire_one(*, tools: list, prompt_text: str) -> dict:
    """One bridge call. Returns observation dict."""
    body = json.dumps({
        "model": "gemma-4-a4b",
        "messages": make_discourse(prompt_text),
        "tools": tools,
        "tool_choice": "auto",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        chat_completions_url(),
        data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    elapsed = time.time() - t0
    choice = resp["choices"][0]
    msg = choice["message"]
    tool_calls = msg.get("tool_calls") or []
    return {
        "elapsed": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "tool_called": len(tool_calls) > 0,
        "tool_names": [tc["function"]["name"] for tc in tool_calls],
        "completion_tokens": resp.get("usage", {}).get("completion_tokens", 0),
        "content_preview": (msg.get("content") or "")[:120],
    }


def run_cell(name: str, builder, prompt_label: str, prompt_text: str,
             k: int) -> list[dict]:
    """K replicates of a (variant, prompt) cell."""
    tools = builder()
    obs = []
    for i in range(k):
        sys.stderr.write(f"\r  [{name} / {prompt_label}] {i+1}/{k}")
        sys.stderr.flush()
        try:
            obs.append(fire_one(tools=tools, prompt_text=prompt_text))
        except Exception as e:
            obs.append({"elapsed": 0, "tool_called": False, "tool_names": [],
                        "error": str(e), "completion_tokens": 0,
                        "content_preview": "", "finish_reason": "error"})
    sys.stderr.write("\n")
    return obs


def cell_rate(obs: list[dict]) -> tuple[int, int, float]:
    """(n_called, n_total, rate)."""
    n = len(obs)
    n_called = sum(1 for o in obs if o["tool_called"])
    return n_called, n, (n_called / n if n else 0.0)


def main() -> int:
    print(f"=== tool-description elicitation A/B study ===")
    print(f"  K replicates per (variant, prompt): {K_REPLICATES}")
    print(f"  temperature: {TEMPERATURE}")
    print(f"  prompts:")
    for label, text in PROMPTS.items():
        print(f"    [{label}] {text!r}")
    print(f"  variants: {len(VARIANTS)} ({', '.join(VARIANTS.keys())})")
    print()

    # rows[variant][prompt_label] = list of observation dicts
    rows: dict[str, dict[str, list[dict]]] = {}
    for name, builder in VARIANTS.items():
        rows[name] = {}
        for prompt_label, prompt_text in PROMPTS.items():
            rows[name][prompt_label] = run_cell(
                name, builder, prompt_label, prompt_text, K_REPLICATES)

    # Calibration report. The aspirational shape is HIGH on positive,
    # LOW on negative — a calibrated tool fires when relevant and
    # skips when not. Quote both raw rates plus a "calibration
    # margin" = TPR - FPR. Positive margin is good; near-zero margin
    # means the tool fires regardless of relevance.
    print()
    print(f"  {'variant':<22} | {'TPR (positive)':<15} | "
          f"{'FPR (negative)':<15} | {'calib margin':<12} | tools (positive)")
    print(f"  {'-'*22}-+-{'-'*15}-+-{'-'*15}-+-{'-'*12}-+-{'-'*30}")
    for name in VARIANTS.keys():
        pos_obs = rows[name]["positive"]
        neg_obs = rows[name]["negative"]
        pos_called, pos_n, pos_rate = cell_rate(pos_obs)
        neg_called, neg_n, neg_rate = cell_rate(neg_obs)
        margin = pos_rate - neg_rate
        pos_str = f"{pos_called:>2}/{pos_n}  ({pos_rate*100:>4.0f}%)"
        neg_str = f"{neg_called:>2}/{neg_n}  ({neg_rate*100:>4.0f}%)"
        margin_str = f"{margin*100:+>5.0f}pp"
        # Tool-name histogram on positive prompt
        name_counts = {}
        for o in pos_obs:
            for tn in o["tool_names"]:
                name_counts[tn] = name_counts.get(tn, 0) + 1
        names_str = ", ".join(f"{n}={c}" for n, c in name_counts.items()) or "(none)"
        print(f"  {name:<22} | {pos_str:<15} | {neg_str:<15} | "
              f"{margin_str:<12} | {names_str}")

    # Save raw observations + per-cell summaries.
    out = pathlib.Path("/tmp/tool_elicitation_study.json")
    summaries = {}
    for name, by_prompt in rows.items():
        summaries[name] = {}
        for prompt_label, obs in by_prompt.items():
            n_called, n, rate = cell_rate(obs)
            summaries[name][prompt_label] = {
                "n_called": n_called, "n_total": n, "rate": rate,
                "tool_name_counts": {
                    tn: sum(1 for o in obs if tn in o["tool_names"])
                    for tn in {n_ for o in obs for n_ in o["tool_names"]}
                },
            }
    payload = {
        "config": {
            "k_replicates": K_REPLICATES,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "prompts": PROMPTS,
        },
        "summaries": summaries,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  raw summaries: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
