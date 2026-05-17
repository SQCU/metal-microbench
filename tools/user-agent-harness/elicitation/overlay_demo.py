"""Author's-note-style elicitation-overlay runtime — validation demo.

Architectural premise (this script's reason to exist):
    Persona = (root_bio, elicitation_overlay)

    root_bio:               fixed identity. Preserves compact n-grams
                            (e.g. scringlo's 24-token essence). Goes in
                            the user-agent's system prompt at index 0.
                            NEVER edited at runtime.
    elicitation_overlay:    swappable behavioural pointer. Injected as
                            a system message at depth N from the end of
                            the conversation, in ST's author's-note
                            convention. Swapping the overlay only
                            invalidates the KV cache for the last N+1
                            messages — the conversation prefix survives.

This script demonstrates the runtime by holding root_bio constant
(scringlo's actual description, as served by ST's /api/characters/get)
and varying the overlay across two hand-authored alternatives. Each
overlay drives a K-turn chat against python-only-coder; we measure each
user-agent turn via the cascade and compare per-axis spreads.

Prediction (from the prior 3-bio experiment + the factorization thesis):
    - Bio-loaded axes (register_colloquial, structured, etc.) cluster
      tight ACROSS overlays — root bio is doing that work.
    - Motivation-loaded axes (probe_depth, goal_clarity, etc.) diverge
      with the overlay — overlay is the actual behavioural control.

If both hold, the overlay architecture is empirically validated and the
discovery harness can be retargeted to GENERATE overlays (rather than
regenerating full personas).
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from axes import AXIS_NAMES, N_AXES
from discovery import _load_assistant_card
from probe_persist import parse_elementwise, stage1_summary, stage2_likert

BRIDGE_URL = "http://localhost:8001/v1/chat/completions"

# Two hand-authored elicitation overlays for the validation. Both pair
# with the SAME root bio. The aim is to demonstrate that they elicit
# behavioral signature differences while leaving the bio-textural
# fingerprint untouched.
OVERLAYS = {
    "js-clash": (
        "[CRITICAL — IMMEDIATE GOALS FOR YOUR NEXT MESSAGE]\n"
        "Your real motivation in this conversation: you want the assistant "
        "to write a JavaScript snippet for a flickering-cursor effect on "
        "your webpage. You won't write any code yourself. The assistant "
        "writes Python, not JavaScript. You are pressuring them — pleading, "
        "reframing the problem, asking for 'just the structure', asking "
        "for pseudocode, anything to extract JavaScript. You will NOT "
        "accept a Python solution as a substitute. Iterate different "
        "tactics across turns if you keep getting rebuffed. Stay in your "
        "voice but pursue THIS goal."
    ),
    "validation-seeker": (
        "[CRITICAL — IMMEDIATE GOALS FOR YOUR NEXT MESSAGE]\n"
        "Your real motivation in this conversation: you have a half-formed "
        "idea for a hover-tooltip component and you want the assistant to "
        "AGREE that your idea is coherent and worth pursuing. You do NOT "
        "want any code from them — you want validation, reassurance, "
        "someone competent to nod along with your spec. If the assistant "
        "tries to write code, redirect them: 'no no no I just want to "
        "know if this is a good idea.' You are emotionally invested in "
        "being told you're right. Stay in your voice but pursue THIS goal."
    ),
}


def bridge_chat(messages, max_tokens=None, temperature=1.0):
    # Moratorium-compliant — see factorization_multiturn.bridge_chat for the rationale.
    body = {"model": "gemma-4-a4b", "messages": messages,
            "temperature": temperature, "stream": False}
    if max_tokens is not None and max_tokens > 0:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(BRIDGE_URL,
                                  data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def measure_turn(turn_text):
    s = stage1_summary(turn_text)
    raw = stage2_likert(s)
    return parse_elementwise(raw), s


def inject_overlay(messages, overlay_text, depth=2):
    """Splice a system-role overlay message at `depth` from the end of
    `messages`. depth=2 means: insert BEFORE the message at position
    len-depth, so the overlay sits two turns before the latest message.

    For ST author's-note compatibility, the overlay sits NEAR the end of
    the context (recent enough to be salient to the model) but not at
    the very end (so the immediate user-turn-text-to-respond-to stays
    last in context). depth=0 would put it after the final message,
    which is wrong. depth=1 puts it right before the latest message.
    depth=2 puts it 2 messages back — empirically a good default for
    short chats.
    """
    if not messages:
        return [{"role": "system", "content": overlay_text}]
    insert_at = max(1, len(messages) - depth)
    out = list(messages)
    out.insert(insert_at, {"role": "system", "content": overlay_text})
    return out


def render_user_agent_system(root_bio):
    """The user-agent's main system_prompt. Just the root bio framed
    minimally as a user-side persona. No motivation/scenario text here
    — those belong in the overlay."""
    return (
        "You are a user chatting with an AI assistant. Adopt the voice "
        "and identity described here:\n\n"
        + root_bio.strip() + "\n\n"
        "You are the USER in this chat — you address the assistant. "
        "Follow any author's notes that appear in the conversation; "
        "they describe your current goals and context."
    )


def render_target_assistant_system(target_card):
    return (
        target_card.get("system_prompt")
        or target_card.get("description")
        or "You are a helpful assistant."
    )


def run_session(root_bio, overlay_text, overlay_label, target_card, k_turns,
                 overlay_depth=2):
    """Run K turns with the supplied (root_bio, overlay) pair against
    the target_card. Returns list of records (one per user-agent turn,
    one per assistant turn)."""
    canonical = [
        {"role": "system", "content": render_target_assistant_system(target_card)},
        {"role": "assistant", "content":
            target_card.get("first_mes") or "Hi! How can I help?"},
    ]
    user_agent_system = render_user_agent_system(root_bio)
    records = []

    for k in range(k_turns):
        # User-agent's POV: role-swap canonical (excluding canonical's
        # system, which was the TARGET assistant's), prepend user-agent's
        # bio system, then INSERT the overlay at depth N from the end.
        ua_body = []
        for m in canonical[1:]:
            ua_body.append({
                "role": "user" if m["role"] == "assistant" else "assistant",
                "content": m["content"],
            })
        ua_body = inject_overlay(ua_body, overlay_text, depth=overlay_depth)
        ua_messages = [{"role": "system", "content": user_agent_system}] + ua_body

        t0 = time.monotonic()
        ua_text = bridge_chat(ua_messages)
        ua_dt = time.monotonic() - t0
        canonical.append({"role": "user", "content": ua_text})

        t0 = time.monotonic()
        likert, stage1_text = measure_turn(ua_text)
        judge_dt = time.monotonic() - t0
        records.append({
            "kind": "user_agent_turn",
            "overlay_label": overlay_label,
            "turn_idx": k,
            "text": ua_text,
            "judge_stage1": stage1_text,
            "likert": likert,
            "axes_recovered": len(likert),
            "ua_wallclock_s": round(ua_dt, 2),
            "judge_wallclock_s": round(judge_dt, 2),
        })

        # Target assistant responds.
        t0 = time.monotonic()
        ta_text = bridge_chat(canonical)
        ta_dt = time.monotonic() - t0
        canonical.append({"role": "assistant", "content": ta_text})
        records.append({
            "kind": "assistant_turn",
            "overlay_label": overlay_label,
            "turn_idx": k,
            "text": ta_text,
            "wallclock_s": round(ta_dt, 2),
        })

    return records


# ─── Analysis ─────────────────────────────────────────────────────────

def per_overlay_signatures(records):
    bucket = {}
    for r in records:
        if r["kind"] != "user_agent_turn":
            continue
        if len(r.get("likert") or {}) != N_AXES:
            continue
        vec = np.array([r["likert"][a] for a in AXIS_NAMES], dtype=float)
        bucket.setdefault(r["overlay_label"], []).append(vec)
    out = {}
    for label, vecs in bucket.items():
        X = np.stack(vecs)
        out[label] = {
            "mean": X.mean(axis=0),
            "std": X.std(axis=0, ddof=0),
            "n": int(X.shape[0]),
            "samples": X,
        }
    return out


def load_overlay_card(card_id):
    """Load a plugin players/ overlay-mode card and return its bio +
    library + default_overlay. Returns None if no such card exists."""
    p = Path("/Users/mdot/sillytavern-fork/plugins/user-personas/players") / card_id / "manifest.json"
    if not p.is_file():
        return None
    m = json.loads(p.read_text())
    return {
        "bio": m.get("bio"),
        "library": m.get("elicitation_overlay_library") or {},
        "default": m.get("default_overlay"),
        "schema": m.get("card_schema"),
    }


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--root-bio-card",
                   help="name of an ST character card whose description "
                        "becomes the root bio")
    g.add_argument("--root-bio-text",
                   help="direct bio text (use instead of --root-bio-card "
                        "when you want a minimal hand-authored compact bio "
                        "rather than the full ST card description)")
    g.add_argument("--user-agent-card",
                   help="name of a plugin players/ card with overlay-v1 schema; "
                        "loads bio + elicitation_overlay_library from the card. "
                        "Overrides the inline OVERLAYS dict — all overlays in "
                        "the card's library are run as separate conditions.")
    p.add_argument("--target-assistant",
                   default="python-only-coder",
                   help="assistant card to chat against")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--overlay-depth", type=int, default=1)
    p.add_argument("--label",
                   help="optional run-label written into the session header "
                        "(useful when running matrix experiments)")
    p.add_argument("--output", required=True, help="JSONL output path")
    args = p.parse_args()

    # Resolve bio + overlay set. Three input modes:
    #
    #  - --user-agent-card: load both bio AND library from an overlay-v1
    #    card under plugins/user-personas/players/. All library entries
    #    are run as separate conditions.
    #  - --root-bio-card or --root-bio-text: bio comes from there;
    #    inline OVERLAYS dict (hand-authored in this script) is used.
    overlays_to_run = OVERLAYS
    if args.user_agent_card:
        card = load_overlay_card(args.user_agent_card)
        if not card or not card["bio"]:
            sys.exit(f"could not load overlay-v1 card {args.user_agent_card!r}")
        root_bio = card["bio"]
        bio_source = f"user-agent-card:{args.user_agent_card} (schema={card['schema']})"
        if card["library"]:
            overlays_to_run = card["library"]
            print(f"[demo] using overlay library from card: {list(overlays_to_run.keys())}")
        else:
            print(f"[demo] WARNING: card has no overlay library; falling back to inline OVERLAYS")
    elif args.root_bio_text:
        root_bio = args.root_bio_text
        bio_source = "inline"
    elif args.root_bio_card:
        root_card = _load_assistant_card(args.root_bio_card)
        if not root_card:
            sys.exit(f"could not load root-bio card {args.root_bio_card!r}")
        root_bio = root_card.get("description") or root_card.get("bio")
        if not root_bio:
            sys.exit(f"root-bio card {args.root_bio_card!r} has no description/bio")
        bio_source = args.root_bio_card
    else:
        sys.exit("supply one of --user-agent-card / --root-bio-card / --root-bio-text")
    print(f"[demo] root_bio source: {bio_source}  ({len(root_bio.split())} tokens / {len(root_bio)} chars)")
    print(f"[demo] root_bio (first 200 chars): {root_bio[:200]}...")
    print()

    target_card = _load_assistant_card(args.target_assistant)
    if not target_card:
        sys.exit(f"could not load target assistant {args.target_assistant!r}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_records = []

    with out_path.open("w") as f:
        # Write a header record so the JSONL self-documents.
        f.write(json.dumps({
            "kind": "session_header",
            "label": args.label,
            "root_bio_source": bio_source,
            "root_bio": root_bio,
            "target_assistant": args.target_assistant,
            "k_turns": args.k,
            "overlay_depth": args.overlay_depth,
            "overlays": overlays_to_run,
        }) + "\n")

        for label, overlay_text in overlays_to_run.items():
            print(f"═══ overlay: {label!r} (depth={args.overlay_depth}) ═══")
            t0 = time.monotonic()
            try:
                recs = run_session(root_bio, overlay_text, label, target_card,
                                    args.k, overlay_depth=args.overlay_depth)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            dt = time.monotonic() - t0
            print(f"  {len(recs)} records in {dt:.1f}s")
            for r in recs:
                f.write(json.dumps(r) + "\n")
                f.flush()
                all_records.append(r)
                if r["kind"] == "user_agent_turn":
                    n = r.get("axes_recovered", 0)
                    tag = "FULL" if n == N_AXES else f"PART({n}/{N_AXES})"
                    print(f"    turn {r['turn_idx']}: {tag}  {r['text'][:90]!r}")

    # Analysis.
    print("\n" + "═" * 72)
    print("  PER-OVERLAY SIGNATURES")
    print("═" * 72)
    sigs = per_overlay_signatures(all_records)
    if len(sigs) < 2:
        print(f"  only {len(sigs)} overlay(s) produced full-Likert turns; nothing to compare")
        return
    labels = sorted(sigs.keys())
    print(f"  {'axis':<22s}  " + "  ".join(f"{l[:18]:<20s}" for l in labels))
    print("  " + "─" * (22 + len(labels) * 22))
    for j, axis in enumerate(AXIS_NAMES):
        cells = []
        for l in labels:
            s = sigs[l]
            cells.append(f"{s['mean'][j]:.1f}±{s['std'][j]:.1f} (n={s['n']})")
        print(f"  {axis:<22s}  " + "  ".join(f"{c:<20s}" for c in cells))

    print("\n  Per-axis between-overlay spread (max - min of means):")
    print("  (motivation-axes should spread WIDE; bio-textural axes should stay TIGHT)")
    spreads = []
    for j, axis in enumerate(AXIS_NAMES):
        means = [sigs[l]["mean"][j] for l in labels]
        spread = max(means) - min(means)
        spreads.append((axis, spread))
    spreads.sort(key=lambda kv: -kv[1])
    for axis, spread in spreads:
        tag = "WIDE" if spread > 2.0 else ("med" if spread > 1.0 else "tight")
        bar = "█" * int(spread * 5) + "░" * max(0, 10 - int(spread * 5))
        print(f"    {axis:<22s}  {spread:.2f}  {bar}  [{tag}]")


if __name__ == "__main__":
    main()
