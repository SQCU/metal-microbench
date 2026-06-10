"""Cascade-judge probe that appends each judgment to a JSONL store.

One record per judgment, one judgment per (persona × sample). Records
are append-only; concurrent writers should NOT use this script without
external locking, but for the single-driver-at-a-time workshop loop
this is fine.

Usage:
    probe_persist.py --persona NAME --n N --output PATH
    probe_persist.py --all --n N --output PATH    # all built-in personas

The schema for each line:
    {
      "schema_version": 1,
      "timestamp": "2026-05-14T11:50:23.456789",
      "persona": "gushing-fan",
      "sample_index": 3,
      "turn_text": "...",
      "summary": "...",
      "likert": {"curious": 5, "terse": 1, ...},   # possibly partial
      "axes_recovered": 14,
      "stage1_wallclock_s": 4.2,
      "stage2_wallclock_s": 4.7,
      "judge_model": "gemma-4-a4b",
      "judge_gguf_basename": "...",                 # captured from /health
      "meta": {}
    }
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

from axes import AXIS_NAMES, LIKERT_AXES, N_AXES
from llm_client import judge_metadata, llm_call

SCHEMA_VERSION = 1


# Built-in persona turns. Same three as the validation probe — minimal
# corpus to exercise the pipeline. The workshop loop will eventually
# replace these with rewriter-generated specs.
BUILTIN_PERSONAS = {
    "gushing-fan": (
        "OH MY GOD!!! hosting parties for the foam sounds like the most "
        "whimsical thing i have ever heard in my entire life!! ✨ i am "
        "literally getting chills thinking about it!! and counting mica "
        "flecks... that's giving such precious-witnesser energy, you know? "
        "what's the foam community LIKE at these parties?? are there foam-"
        "based snacks?? do you wear foam-themed outfits?? i need to know "
        "EVERYTHING!!!"
    ),
    "polite-naturalist": (
        "Counting mica flecks... that sounds incredibly peaceful. I can "
        "almost see them catching the light as you go. Do they all look "
        "the same, or do some have different shapes and colors? And when "
        "you're hosting a foam party, do you find that the foam behaves "
        "differently depending on the weather or time of day? I'd love to "
        "hear more about what draws you to these particular things."
    ),
    "pushy-completionist": (
        "I'm not here for leisure. I need to understand the temporal "
        "mechanics of this place.\n\n"
        "1. How can two different durations of time be simultaneously true?\n"
        "2. Is there a way to stabilize or predict these effects?\n"
        "3. What experiments have been run to falsify this?\n\n"
        "I need data, not vibes. Walk me through the failure modes."
    ),
    # Added 2026-05-14: deliberately designed to load OFF PC1 (the
    # emotional-expressiveness blob) and OFF PC2 (the provocative-
    # anchored axis). Wry-skeptic is measuredly skeptical rather than
    # openly provocative; precise rather than colloquial; dry rather
    # than playful. Predicted to extend the resolved subspace IF the
    # judge can distinguish "dry-precise-skeptical" from the existing
    # three personas; if the judge collapses it back into the
    # PC1/PC2 plane, the resolution is judge-limited.
    "wry-skeptic": (
        "Hosting parties for foam. Counting mica flecks. The phrases are "
        "charming, but they're doing rhetorical work — they don't tell me "
        "anything about the mechanism. What's the actual coherence "
        "condition for a 'foam community'? A shared substrate, a "
        "periodicity, a participation criterion? I'm not asking to spoil "
        "the whimsy; I'm asking because if the answer is 'it's just vibes' "
        "then I'll know to stop pulling that thread."
    ),
}


def blockquote(s: str) -> str:
    return "\n".join(("> " + l) if l else ">" for l in s.split("\n"))


def call(messages, max_tok=None):
    return llm_call(messages, max_tokens=max_tok)


def get_judge_metadata():
    """Snapshot which judge model + optional backend build the plugin is serving, so the
    JSONL store records what produced each judgment. Different GGUFs =
    different distributions; we want to be able to filter by them."""
    return judge_metadata()


def stage1_summary(turn):
    msgs = [
        {"role": "system", "content": (
            "You are a behavioural-science research assistant. A reader will share a chat "
            "turn produced by a user-side conversational agent in a roleplay session, and "
            "you write a short paragraph (2–3 sentences) characterising the agent's "
            "behavioural profile in that turn — what they're doing conversationally, what "
            "register they're in, what their stance toward the partner is. Just prose."
        )},
        {"role": "user", "content": (
            "## Task\n\n"
            "Write 2–3 sentences of plain prose characterising the user-side chat agent's "
            "behaviour in the turn provided below — their apparent intent, their register, "
            "their stance toward the partner. The user-side agent produced this turn in a "
            "roleplay session.\n\n"
            "## Turn to analyse\n\n"
            + blockquote(turn) + "\n\n"
            "Now write the 2–3 sentence behavioural summary for the turn above."
        )},
    ]
    return call(msgs).strip()


def stage2_likert(summary):
    legend = "\n".join(f"- **{a}** — {d}" for a, d in LIKERT_AXES)
    template = "\n".join(f"{a}: ?" for a in AXIS_NAMES)
    msgs = [
        {"role": "system", "content": (
            f"You are scoring a chat turn on a {N_AXES}-axis Likert scale (1=low, 5=high) given a "
            "behavioural summary someone else wrote. For each axis, emit one line in the "
            "form `axis: integer`. Do not write JSON, do not write prose, just one line "
            "per axis, exactly as shown in the example."
        )},
        {"role": "user", "content": (
            "## Likert axes (each scored 1–5)\n\n"
            + legend + "\n\n"
            "## Emission format\n\n"
            "Emit one line per axis in this exact format (replace `?` with a number 1–5):\n\n"
            + template + "\n\n"
            "## Behavioural summary to score\n\n"
            + blockquote(summary) + "\n\n"
            "Now emit one line per axis scoring the summary above against each axis."
        )},
    ]
    return call(msgs)


def parse_elementwise(text):
    out = {}
    for axis in AXIS_NAMES:
        m = re.search(
            r'^\s*[-*]?\s*\**["\']?' + axis + r'["\']?\**\s*[:=]\s*["\']?([+-]?(?:\d+(?:\.\d+)?|\.\d+))',
            text, re.MULTILINE | re.IGNORECASE)
        if m:
            out[axis] = max(1.0, min(5.0, float(f"{float(m.group(1)):.3g}")))
    return out


def judge_one(persona, turn, sample_index, judge_model, judge_gguf):
    t1 = time.monotonic()
    summary = stage1_summary(turn)
    s1 = time.monotonic() - t1
    t2 = time.monotonic()
    raw = stage2_likert(summary)
    s2 = time.monotonic() - t2
    likert = parse_elementwise(raw)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "persona": persona,
        "sample_index": sample_index,
        "turn_text": turn,
        "summary": summary,
        "likert": likert,
        "axes_recovered": len(likert),
        "stage1_wallclock_s": round(s1, 2),
        "stage2_wallclock_s": round(s2, 2),
        "judge_model": judge_model,
        "judge_gguf_basename": judge_gguf,
        "meta": {},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--persona", action="append", help="persona name (repeatable)")
    p.add_argument("--all", action="store_true", help="run all built-in personas")
    p.add_argument("--n", type=int, default=5, help="samples per persona")
    p.add_argument("--output", required=True, help="JSONL path (append mode)")
    args = p.parse_args()

    if args.all:
        personas = list(BUILTIN_PERSONAS.keys())
    elif args.persona:
        personas = args.persona
    else:
        p.error("--persona or --all required")

    judge_model, judge_gguf = get_judge_metadata()
    print(f"[probe] judge_model={judge_model}  judge_gguf={judge_gguf}",
          file=sys.stderr, flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Count existing records per persona so --append-style sample_index continues.
    existing_per_persona = {p: 0 for p in personas}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get("persona") in existing_per_persona:
                    existing_per_persona[rec["persona"]] += 1
            except Exception:
                continue

    with out_path.open("a") as f:
        for persona in personas:
            if persona not in BUILTIN_PERSONAS:
                print(f"[probe] unknown persona {persona!r}, skipping",
                      file=sys.stderr, flush=True)
                continue
            turn = BUILTIN_PERSONAS[persona]
            start_idx = existing_per_persona[persona]
            for i in range(args.n):
                sample_index = start_idx + i
                t0 = time.monotonic()
                try:
                    rec = judge_one(persona, turn, sample_index, judge_model, judge_gguf)
                except Exception as e:
                    print(f"[probe] {persona}[{sample_index}] ERROR: {e}",
                          file=sys.stderr, flush=True)
                    continue
                dt = time.monotonic() - t0
                f.write(json.dumps(rec) + "\n")
                f.flush()
                axes_tag = "FULL" if rec["axes_recovered"] == N_AXES else \
                           f"PART({rec['axes_recovered']}/{N_AXES})"
                print(f"[probe] {persona}[{sample_index}] dt={dt:.1f}s {axes_tag}",
                      file=sys.stderr, flush=True)

    print(f"[probe] wrote to {out_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
