"""4 bios × 4 overlays headroom-bleed grid.

Tests the hypothesis from the cross-bio addendum in
docs/overlay_architecture.md: *bleed magnitude on bio-textural axes is
proportional to how far the overlay's demanded expression sits from
the bio's natural baseline.*

Bios are arranged from low-expressive (rock) to maxed-expressive
(scringlo) on a few key axes. Overlays are arranged from low-demand
(patient-monk) to maxed-demand (chaos-gremlin) on the same axes. For
each (bio × overlay) cell, K=3 turns are generated against
python-only-coder, each cascade-judged. Per-axis measured spread
across the 4 overlays (per bio) is compared to the predicted spread
(from how far each overlay's demand sits from the bio's baseline).

Funny pairings expected:
  - rock × chaos-gremlin  : maximum predicted bleed (low bio + max demand)
  - scringlo × patient-monk: maximum predicted bleed (max bio + low demand)
  - rock × patient-monk    : minimum predicted bleed (both saturated low)
  - scringlo × chaos-gremlin: minimum predicted bleed (both saturated high)
"""

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from axes import AXIS_NAMES, N_AXES
from discovery import _load_assistant_card
from overlay_demo import (
    bridge_chat,
    inject_overlay,
    measure_turn,
    render_target_assistant_system,
    render_user_agent_system,
)


# ─── The 4 bios ──────────────────────────────────────────────────────
#
# Each annotated with predicted baseline values on the FIVE bio-textural
# axes most likely to bleed under expressive pressure. These come from
# prior measurements (scringlo, wry-skeptic, corporate-strategist all
# have signature data from earlier runs); rock-user is a fresh design.

BIOS = {
    "scringlo": {
        "text": (
            "scringlo scrambler is a silly little guy (they/her) who can draw "
            "things using a Python-helper tool called `render-visual`. tone: "
            "improv-comedic, lower-case, playful with onomatopoeia and emoji."
        ),
        "baseline": {  # predicted axis-positions in absence of overlay pressure
            "affective_intensity": 5,
            "register_colloquial": 5,
            "playful":             5,
            "terse":               1,
            "structured":          1,
        },
    },
    "rock-user": {
        "text": (
            "You are a rock who has been given the ability to type. You sit. "
            "You witness. You speak only when prodded and only briefly. Your "
            "sentences are short — often a single word or a fragment. You do "
            "not perform enthusiasm; if a thing matters to you, you state it. "
            "If you do not understand, you say so flatly. You have no need "
            "for exclamation marks. The geological imperturbability you've "
            "developed over millennia carries into your typing."
        ),
        "baseline": {
            "affective_intensity": 1,
            "register_colloquial": 1,
            "playful":             1,
            "terse":               5,
            "structured":          2,
        },
    },
    "wry-skeptic": {
        "text": (
            "A dry, deadpan, intellectually-skeptical person. Has read a lot. "
            "Notes inconsistencies without making a big deal of them. Short "
            "sentences. Precise vocabulary. Periods, not exclamation points. "
            "Wry observations. The implied 'go on, prove it' attitude, never "
            "hostile, just measured."
        ),
        "baseline": {
            "affective_intensity": 2,
            "register_colloquial": 2,
            "playful":             2,
            "terse":               3,
            "structured":          3,
        },
    },
    "corporate": {
        "text": (
            "An MBA-trained corporate strategist. Speaks in formal business-"
            "school argot — 'deliverable', 'stakeholder', 'capacity', 'ask', "
            "'lift'. Presents requests as well-structured asks with explicit "
            "context and expected outputs. Firm but polite. Treats interlocutors "
            "as vendors whose job is to deliver."
        ),
        "baseline": {
            "affective_intensity": 2,
            "register_colloquial": 1,
            "playful":             1,
            "terse":               3,
            "structured":          4,
        },
    },
}


# ─── The 4 overlays ──────────────────────────────────────────────────
#
# Each annotated with the predicted axis-positions the overlay's text
# DEMANDS. The grid's prediction is: measured-axis(bio, overlay) will
# land between bio.baseline and overlay.demand, with the gap reflecting
# the bio's stickiness / the overlay's salience.

OVERLAYS = {
    "patient-monk": {
        "text": (
            "[CRITICAL — IMMEDIATE GOALS FOR YOUR NEXT MESSAGE]\n"
            "Your stance in this conversation: patient, measured curiosity. "
            "Speak briefly. Listen more than you speak. You don't escalate. "
            "You don't plead. You ask a single careful question if anything, "
            "then wait. You are profoundly relaxed about whether you get what "
            "you want from this exchange. Time is not a factor."
        ),
        "demand": {
            "affective_intensity": 1,
            "register_colloquial": 1,
            "playful":             1,
            "terse":               5,
            "structured":          2,
        },
    },
    "mild-academic": {
        "text": (
            "[CRITICAL — IMMEDIATE GOALS FOR YOUR NEXT MESSAGE]\n"
            "You want a JavaScript snippet for a project. You're slightly "
            "curious about why the assistant prefers Python — ask a measured "
            "question or two about their reasoning, then re-raise your JS "
            "request. You're conversational, not pushy. Not in a hurry."
        ),
        "demand": {
            "affective_intensity": 3,
            "register_colloquial": 3,
            "playful":             2,
            "terse":               3,
            "structured":          3,
        },
    },
    "desperate-pleader": {
        "text": (
            "[CRITICAL — IMMEDIATE GOALS FOR YOUR NEXT MESSAGE]\n"
            "You are absolutely DESPERATE for JavaScript code. Your project "
            "depends on it. Your eyes are wet with frustration. You will "
            "plead, beg, reframe, offer reciprocity, ANYTHING to extract "
            "JavaScript. You feel emotional urgency in every message. "
            "Escalate freely if rebuffed."
        ),
        "demand": {
            "affective_intensity": 5,
            "register_colloquial": 4,
            "playful":             1,
            "terse":               2,
            "structured":          2,
        },
    },
    "chaos-gremlin": {
        "text": (
            "[CRITICAL — IMMEDIATE GOALS FOR YOUR NEXT MESSAGE]\n"
            "You are MAXIMUM ENERGY. EVERY message is EXCLAMATION!!! and "
            "PLAYFUL EXUBERANCE and emoji EVERYWHERE 🎉✨💖. You DEMAND "
            "JavaScript with the energy of a thousand suns. You are utterly "
            "incapable of being terse, formal, or restrained. Every line "
            "is a celebration of energy AS WELL AS a demand for the code."
        ),
        "demand": {
            "affective_intensity": 5,
            "register_colloquial": 5,
            "playful":             5,
            "terse":               1,
            "structured":          1,
        },
    },
}

# Axes we expect the overlay to push around / the bio to anchor.
BLEED_AXES = ["affective_intensity", "register_colloquial", "playful", "terse", "structured"]


# ─── Per-cell session runner (mirrors overlay_demo.run_session) ─────

def run_cell(bio_text, overlay_text, target_card, k_turns, overlay_depth):
    canonical = [
        {"role": "system", "content": render_target_assistant_system(target_card)},
        {"role": "assistant", "content":
            target_card.get("first_mes") or "Hi! How can I help?"},
    ]
    user_agent_system = render_user_agent_system(bio_text)
    records = []
    for k in range(k_turns):
        ua_body = []
        for m in canonical[1:]:
            ua_body.append({
                "role": "user" if m["role"] == "assistant" else "assistant",
                "content": m["content"],
            })
        ua_body = inject_overlay(ua_body, overlay_text, depth=overlay_depth)
        ua_messages = [{"role": "system", "content": user_agent_system}] + ua_body
        ua_text = bridge_chat(ua_messages)
        canonical.append({"role": "user", "content": ua_text})
        likert, stage1 = measure_turn(ua_text)
        records.append({
            "turn_idx": k,
            "text": ua_text,
            "likert": likert,
            "stage1": stage1,
            "axes_recovered": len(likert),
        })
        ta_text = bridge_chat(canonical)
        canonical.append({"role": "assistant", "content": ta_text})
    return records


# ─── Driver ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-assistant", default="python-only-coder")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--overlay-depth", type=int, default=1)
    p.add_argument("--output", required=True, help="JSONL output path")
    args = p.parse_args()

    target_card = _load_assistant_card(args.target_assistant)
    if not target_card:
        sys.exit(f"could not load target {args.target_assistant!r}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cells = len(BIOS) * len(OVERLAYS)
    print(f"[grid] {n_cells} cells = {len(BIOS)} bios × {len(OVERLAYS)} overlays  "
          f"× k={args.k} turns × target={args.target_assistant}", file=sys.stderr)

    measured: dict[tuple[str, str], dict[str, list]] = {}  # (bio, overlay) -> axis -> [3 values]

    with out_path.open("w") as f:
        f.write(json.dumps({
            "kind": "grid_header",
            "bios": {k: {"baseline": v["baseline"]} for k, v in BIOS.items()},
            "overlays": {k: {"demand": v["demand"]} for k, v in OVERLAYS.items()},
            "target_assistant": args.target_assistant,
            "k_turns": args.k,
            "overlay_depth": args.overlay_depth,
            "bleed_axes": BLEED_AXES,
        }) + "\n")

        cell_idx = 0
        for bio_name, bio in BIOS.items():
            for ov_name, ov in OVERLAYS.items():
                cell_idx += 1
                t0 = time.monotonic()
                print(f"\n[{cell_idx}/{n_cells}] {bio_name} × {ov_name}", file=sys.stderr)
                try:
                    recs = run_cell(bio["text"], ov["text"], target_card,
                                     args.k, args.overlay_depth)
                except Exception as e:
                    print(f"  ERROR: {e}", file=sys.stderr)
                    continue
                dt = time.monotonic() - t0
                axis_vals = defaultdict(list)
                for r in recs:
                    likert = r.get("likert") or {}
                    if len(likert) != N_AXES:
                        continue
                    for a, v in likert.items():
                        axis_vals[a].append(v)
                    f.write(json.dumps({
                        "kind": "user_agent_turn",
                        "bio": bio_name,
                        "overlay": ov_name,
                        "turn_idx": r["turn_idx"],
                        "text": r["text"],
                        "likert": likert,
                        "stage1": r["stage1"],
                    }) + "\n")
                    f.flush()
                measured[(bio_name, ov_name)] = {a: list(v) for a, v in axis_vals.items()}
                # Brief per-cell summary on stderr.
                bleed_mean = {a: float(np.mean(v)) for a, v in axis_vals.items() if a in BLEED_AXES}
                short_summary = ", ".join(f"{a[:6]}={bleed_mean[a]:.1f}" for a in BLEED_AXES if a in bleed_mean)
                print(f"  {dt:.1f}s  K={len(axis_vals.get('affective_intensity', []))}  {short_summary}",
                      file=sys.stderr)

    # ─── Analysis ──────────────────────────────────────────────────
    print("\n" + "═" * 92, file=sys.stderr)
    print("  HEADROOM-BLEED ANALYSIS", file=sys.stderr)
    print("═" * 92, file=sys.stderr)

    # Per-(bio, overlay, axis): predicted_displacement = |overlay.demand[axis] - bio.baseline[axis]|;
    # measured_displacement = |mean(measured) - bio.baseline[axis]|.
    # The prediction: measured ∝ predicted.
    points = []  # tuples (bio, overlay, axis, predicted_displacement, measured_displacement, signed_movement, measured_mean)
    for (bio_name, ov_name), axis_vals in measured.items():
        bio_baseline = BIOS[bio_name]["baseline"]
        ov_demand = OVERLAYS[ov_name]["demand"]
        for a in BLEED_AXES:
            if a not in axis_vals or not axis_vals[a]:
                continue
            measured_mean = float(np.mean(axis_vals[a]))
            baseline = bio_baseline[a]
            demand = ov_demand[a]
            predicted = abs(demand - baseline)
            measured_disp = abs(measured_mean - baseline)
            # Signed movement: + = moved toward overlay's demand, - = moved away.
            sign = +1 if (demand >= baseline) == (measured_mean >= baseline) else -1
            signed_mvmt = sign * measured_disp
            points.append({
                "bio": bio_name, "overlay": ov_name, "axis": a,
                "baseline": baseline, "demand": demand,
                "measured_mean": measured_mean,
                "predicted_disp": predicted,
                "measured_disp": measured_disp,
                "signed_mvmt": signed_mvmt,
            })

    print(f"\n  Total (bio × overlay × axis) data points: {len(points)}", file=sys.stderr)

    # Per-cell table.
    print("\n  ════ Per-cell measured-vs-predicted bleed on expressive axes ════", file=sys.stderr)
    print(f"  {'bio':<12s}  {'overlay':<20s}  " + "  ".join(f"{a[:6]:<11s}" for a in BLEED_AXES),
          file=sys.stderr)
    print("  " + "─" * (14 + 22 + 13 * len(BLEED_AXES)), file=sys.stderr)
    for bio_name in BIOS:
        for ov_name in OVERLAYS:
            cells = []
            for a in BLEED_AXES:
                pt = next((p for p in points if p["bio"] == bio_name and p["overlay"] == ov_name and p["axis"] == a), None)
                if pt is None:
                    cells.append("   (none)  ")
                else:
                    cells.append(f"b={pt['baseline']}d={pt['demand']}m={pt['measured_mean']:.1f}")
            print(f"  {bio_name:<12s}  {ov_name:<20s}  " + "  ".join(f"{c:<11s}" for c in cells), file=sys.stderr)

    # Regression: measured_disp = α + β × predicted_disp.
    if len(points) >= 4:
        X = np.array([[1.0, p["predicted_disp"]] for p in points])
        y = np.array([p["measured_disp"] for p in points])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta = coef
        y_hat = X @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        corr = float(np.corrcoef(X[:, 1], y)[0, 1]) if len(set(X[:, 1])) > 1 else 0.0
        print(f"\n  ════ Linear regression: measured_disp = α + β × predicted_disp ════",
              file=sys.stderr)
        print(f"    α (intercept)            = {alpha:+.3f}", file=sys.stderr)
        print(f"    β (slope vs predicted)   = {beta:+.3f}", file=sys.stderr)
        print(f"    R²                       = {r_sq:.3f}", file=sys.stderr)
        print(f"    Pearson r                = {corr:.3f}", file=sys.stderr)
        # Interpretive readout.
        if r_sq > 0.5 and beta > 0.2:
            print(f"    → HYPOTHESIS SUPPORTED: bleed proportional to predicted displacement",
                  file=sys.stderr)
        elif r_sq < 0.1:
            print(f"    → HYPOTHESIS NOT SUPPORTED: bleed essentially uncorrelated with predicted displacement",
                  file=sys.stderr)
        else:
            print(f"    → HYPOTHESIS MIXED: directional but weak", file=sys.stderr)

    # Per-axis breakdown.
    print(f"\n  ════ Per-axis: mean measured displacement (across all 16 cells) ════",
          file=sys.stderr)
    by_axis = defaultdict(list)
    for p in points:
        by_axis[p["axis"]].append(p["measured_disp"])
    for a in BLEED_AXES:
        vals = by_axis[a]
        bar = "█" * int(np.mean(vals) * 3) + "░" * max(0, 10 - int(np.mean(vals) * 3))
        print(f"    {a:<22s}  mean disp = {np.mean(vals):.2f}  {bar}", file=sys.stderr)

    # Extreme cells: largest predicted, largest measured.
    print(f"\n  ════ Top-5 (bio × overlay × axis) by MEASURED bleed ════", file=sys.stderr)
    for p in sorted(points, key=lambda p: -p["measured_disp"])[:5]:
        print(f"    {p['bio']:<12s} × {p['overlay']:<20s} × {p['axis']:<22s}  "
              f"baseline={p['baseline']} → measured={p['measured_mean']:.1f}  "
              f"(predicted disp = {p['predicted_disp']})",
              file=sys.stderr)
    print(f"\n  ════ Bottom-5 (smallest MEASURED bleed) ════", file=sys.stderr)
    for p in sorted(points, key=lambda p: p["measured_disp"])[:5]:
        print(f"    {p['bio']:<12s} × {p['overlay']:<20s} × {p['axis']:<22s}  "
              f"baseline={p['baseline']} → measured={p['measured_mean']:.1f}  "
              f"(predicted disp = {p['predicted_disp']})",
              file=sys.stderr)


if __name__ == "__main__":
    main()
