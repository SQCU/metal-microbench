#!/usr/bin/env python3
"""DEPRECATED: superseded by pyrepl.py (extraction+REPL) and pyrepl_refine.py
(the incremental harness). Kept for reference; do not build new harnesses on this.

Incremental edit-in-REPL refinement — the loop krepl.py NAMED but didn't build.

krepl.py drives the residual down, but every turn the model RE-EMITS the entire
program (the assistant turn IS a full rewrite, ~2k tokens, no diff). That is the
exact expense svg_repl.py was written to remove: an artifact that PERSISTS in a
REPL and is mutated by small, tolerantly-parsed EDIT blocks.

This module wires call_lm + render + judge around svg_repl.SvgRepl so the loop is:
  turn 0    : model emits a FULL python program that sets `svg` -> stored in the REPL.
  turn >=1  : model emits ONLY edit blocks (EDIT-SUITE grammar) against the program
              shown to it -> parsed + applied to the persisted program. The assistant
              message is the EDIT TEXT only (small); the whole program is never re-sent.

Feedback REUSES krepl._feedback verbatim (its render + RESIDUAL + DELTA images,
mse/ssim, checklist coverage, anchored faithfulness score AND the numbered current
program) and APPENDS, per edit that failed to apply, an
  "edit N did not apply: <reason> — re-anchor"
note so the model can fix its anchor next turn. No image or metric is dropped.

Every intermediate is dumped exactly like krepl:
  {prefix}_reference.png
  {prefix}_t{NN}.py / _t{NN}_render.png / _t{NN}_residual.png / _t{NN}_delta.png
and a {prefix}_report.json records, per turn: out_tokens(completion), op_count,
n_applied, n_errors, lines_changed_vs_prev, mse/ssim/anchored/checklist.

SERVERLESS-friendly: call_lm and the judge fns are module attributes the probe
monkeypatches; nothing here touches an API backend on its own beyond call_lm.
"""
from __future__ import annotations
import json, pathlib, sys

import numpy as np

# Reuse the canonical pieces — do NOT re-implement any of them.
import edit_elicit as _E                                             # sets sys.path
from edit_elicit import _extract_program, _run_program, _numbered    # noqa: E402
from svg_refinement_loop import render_svg, image_to_data_url, load_target_from_path  # noqa: E402
from elicit import call_lm                                           # noqa: E402
import judge as _judge                                              # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hresidual as R                                               # noqa: E402
import svg_repl                                                      # noqa: E402
# Reuse krepl's _feedback (all imagery + metrics) and its STRONG-anchor default.
import krepl as _krepl                                              # noqa: E402

# Re-export the fixed strong-reconstruction anchor so the probe can blank it
# (same knob krepl exposes). It is read lazily inside refine_incremental.
_ANCHOR_GOOD = _krepl._ANCHOR_GOOD

# ----------------------------------------------------------------------------- #
# System prompt: turn 0 = FULL program; every later turn = ONLY edit blocks.    #
# The EDIT-SUITE grammar is stated verbatim so the model can author it.         #
# ----------------------------------------------------------------------------- #
SYSTEM = (
    "You reconstruct a REFERENCE image as an SVG, built by a Python program that sets the "
    "string variable `svg`. You refine it across turns. Each turn you are shown: your current "
    "render; a false-color RESIDUAL where BRIGHT marks regions your render gets WRONG or MISSING "
    "versus the reference and dark marks matches; a DELTA showing what changed since last turn; "
    "the MSE and SSIM; a faithfulness score; a checklist of reference elements; and your CURRENT "
    "PROGRAM, line-numbered.\n\n"
    "TURN 0: emit your strongest one-shot reconstruction as a FULL Python program in a single "
    "```python``` block that sets `svg`. This program is then HELD in a REPL.\n\n"
    "EVERY LATER TURN: you do NOT re-emit the program — it is already submitted and persists in "
    "the REPL as Python. Emit ONLY one or more EDIT BLOCKS that mutate the line-numbered program "
    "shown to you; the harness applies them, re-runs the program, and re-renders. "
    "Make a SUBSTANTIAL revision each turn: darken the brightest residual, add what the checklist "
    "says is missing, fix wrong colours/places/sizes. Use this grammar (prose around blocks is "
    "ignored; op names are case-insensitive; the marker rune may be ### or *** or ===):\n\n"
    "  ### EDIT <OP> ###\n"
    "  --- FIND ---\n"
    "  <one or more EXACT lines copied from the CURRENT program>   (omit FIND for APPEND)\n"
    "  --- REPLACE ---\n"
    "  <replacement lines>                                         (omit REPLACE for DELETE)\n"
    "  ### END ###\n\n"
    "Ops (the FIND text must appear EXACTLY ONCE in the current program):\n"
    "  REPLACE      : substitute the FIND text with the REPLACE text (REPLACE may be empty).\n"
    "  INSERT_AFTER : insert the REPLACE lines immediately AFTER the FIND anchor.\n"
    "  DELETE       : remove the FIND text (no REPLACE).\n"
    "  APPEND       : no FIND; insert the REPLACE lines just before the final `svg = ...` line.\n\n"
    "An edit whose FIND is missing or matches MORE THAN ONCE is REJECTED and reported back to you "
    "next turn — re-anchor on a longer, unique snippet. `np` and the reference array `target` "
    "(H,W,3 uint8 RGB) are available in the program — measure it rather than guessing."
)

# Turn-0 instruction (full program) and the standing reminder for edit turns.
_TURN0_TEXT = (
    "REFERENCE image ({W}x{H}). Write your TURN-0 program now: a single ```python``` block that "
    "sets `svg` — your strongest one-shot reconstruction. Assign `svg` EXACTLY ONCE: do NOT include "
    "draft/alternative `svg =` assignments (a later assignment silently overwrites an earlier one, "
    "and edits you later make to the dead block would never render). From the next turn on you will "
    "ONLY emit edit blocks against this program; you will never re-send the whole program."
)
_EDIT_REMINDER = (
    "Emit ONLY edit blocks (### EDIT <OP> ### / --- FIND --- / --- REPLACE --- / ### END ###) that "
    "mutate the numbered program above. Do NOT re-emit the whole program — it is held in the REPL."
)


def _line_diff_count(a: str, b: str) -> int:
    """Lines that differ between two program texts (added/removed/changed), by a
    cheap line-multiset symmetric difference — the 'how much did the edit move the
    source' signal for the per-turn record. 0 means the program is unchanged."""
    from collections import Counter
    ca, cb = Counter(a.split("\n")), Counter(b.split("\n"))
    return int(sum(((ca - cb) + (cb - ca)).values()))


# Gemma-4 turn delimiters (<|turn>/<turn|>) and Gemma-3 (<start_of_turn>/<end_of_turn>)
# can glue onto edit-block markers (e.g. '### END ###<turn|>'), hiding the close. Strip
# them from model text before parsing edits.
_TURN_MARKER_RE = __import__("re").compile(
    r"<\|?/?turn\|?>|</?start_of_turn>|</?end_of_turn>")


def _strip_turn_markers(s: str) -> str:
    return _TURN_MARKER_RE.sub("", s) if s else s


def refine_incremental(target, rounds, seed, out_dir, prefix, max_tokens=16000):
    """Incremental edit-in-REPL refinement.

    turn 0 stores a FULL program in an svg_repl.SvgRepl; turns 1..rounds apply
    EDIT blocks to that persisted program. Returns a krepl-shaped report dict and
    dumps every intermediate + {prefix}_report.json.
    """
    prev_render = None
    prev_svg = None
    W, H = target.size
    tgt_arr = np.asarray(target.convert("RGB"))
    out_dir.mkdir(parents=True, exist_ok=True)
    target.save(out_dir / f"{prefix}_reference.png")

    def jchat(msgs):
        return call_lm(msgs, 512, 0.0, seed)[0]

    # Fixed per-image checklist (seed-0 so it is identical across seeds), exactly
    # as krepl computes it. Reuse the judge fn (monkeypatchable by the probe).
    checklist = _judge.salient_checklist(lambda m: call_lm(m, 512, 0.0, 0)[0], target) or []
    (out_dir / f"{prefix}_checklist.json").write_text(
        json.dumps(checklist, ensure_ascii=False, indent=2))

    # Anchored (calibrated) judge — same STRONG/WEAK anchor construction as krepl.
    from PIL import Image as _Image
    good_anchor = (_Image.open(_ANCHOR_GOOD).convert("RGB")
                   if _ANCHOR_GOOD and pathlib.Path(_ANCHOR_GOOD).exists() else None)
    bad_anchor = _Image.new("RGB", (good_anchor.size if good_anchor else (320, 180)),
                            (128, 128, 128))

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": _TURN0_TEXT.format(W=W, H=H)},
            {"type": "image_url", "image_url": {"url": image_to_data_url(target)}}]},
    ]

    repl = svg_repl.SvgRepl("svg = ''")     # persistent program_state
    traj = []
    for t in range(rounds + 1):
        text, finish, usage = call_lm(messages, max_tokens, 1.0, seed + t)
        messages.append({"role": "assistant", "content": text})

        apply_errors: list[dict] = []
        op_count = n_applied = n_errors = 0
        prev_program = repl.program

        if t == 0:
            # TURN 0: FULL program -> store in the REPL.
            prog = _extract_program(text)
            if prog is None:                            # never start blank: retry once
                messages.append({"role": "user", "content":
                    "No program found. Reply with ONLY a ```python``` block that sets `svg`."})
                text, finish, usage = call_lm(messages, max_tokens, 1.0, seed + 9000 + t)
                messages.append({"role": "assistant", "content": text})
                prog = _extract_program(text)
            if prog is not None:
                repl = svg_repl.SvgRepl(prog)
        else:
            # TURN >=1: parse EDIT blocks + apply to the PERSISTED program. The
            # assistant message above is the EDIT TEXT only (small) — no rewrite.
            ops = svg_repl.parse_edits(text)
            res = repl.apply(text)
            op_count = len(ops)
            n_applied = res["applied"]
            apply_errors = res["errors"]
            n_errors = len(apply_errors)

        # Re-run the PERSISTED Python program (REPL state) and render it. Python is
        # a superset of SVG — keeping the program (not a flattened SVG) is the point:
        # the model can compute, loop, sample `target`. Each turn we update the code
        # via edits and re-render the SAME canonical program.
        source = repl.program.split("\n")
        lines_changed = _line_diff_count(prev_program, repl.program)

        svg, err = _run_program(repl.program, tgt_arr)
        render = None
        if svg and not err:
            try:
                render = render_svg(svg, W, H)
            except Exception as e:
                err = repr(e)[:200]

        # render-unchanged guard: applied edits that do NOT move the picture are a
        # tell that the model anchored on a no-op / dead code (e.g. a draft `svg =`
        # block that a later assignment overwrites) — flag it so it re-anchors next
        # turn instead of silently spinning.
        render_unchanged = bool(t >= 1 and n_applied > 0
                                and prev_svg is not None and svg == prev_svg)

        # ---- DUMP EVERYTHING (identical layout to krepl) ----
        (out_dir / f"{prefix}_t{t:02d}.py").write_text(repl.program)
        stats, cl, aq = {"mse": None, "ssim": None}, None, None
        if render is not None:
            render.save(out_dir / f"{prefix}_t{t:02d}_render.png")
            R.false_color_residual(target, render).save(out_dir / f"{prefix}_t{t:02d}_residual.png")
            R.false_color_delta(prev_render, render).save(out_dir / f"{prefix}_t{t:02d}_delta.png")
            stats = R.residual_stats(target, render, prev_render)
            cl = _judge.checklist_match(jchat, target, render, checklist) if checklist else None
            aq = (_judge.anchored_score(jchat, target, render, good_anchor, bad_anchor)
                  if good_anchor is not None else None)

        row = {"turn": t, "out_tokens": usage.get("completion_tokens"),
               "finish": finish, "lines": len(source), "err": (err[:160] if err else None),
               "op_count": op_count, "n_applied": n_applied, "n_errors": n_errors,
               "lines_changed_vs_prev": lines_changed,
               "render_unchanged": render_unchanged,
               "edit_errors": apply_errors,
               **stats,
               "anchored_quality": aq,
               "checklist_present": (cl or {}).get("present_count"),
               "checklist_total": (cl or {}).get("total"),
               "checklist_fraction": (cl or {}).get("fraction"),
               "checklist_scores": (cl or {}).get("scores")}
        traj.append(row)

        if t < rounds:
            # REUSE krepl._feedback verbatim (render + residual + delta images,
            # mse/ssim, checklist coverage, anchored score, numbered program),
            # then APPEND the standing edit reminder and any apply-error notes.
            fb = _krepl._feedback(t, source, render, stats, cl, aq, err, target, prev_render)
            note = "\n\n" + _EDIT_REMINDER
            if render_unchanged:
                note += ("\nWARNING: your edits APPLIED but the RENDER DID NOT CHANGE — you "
                         "anchored on text that does not affect the picture. Edit the actual SVG "
                         "element lines shown above (a <rect>/<path>/<circle>/<text> attribute, or "
                         "INSERT_AFTER an existing element) so the image changes this turn.")
            for e in apply_errors:
                note += (f"\nedit {e['index']} did not apply: {e['reason']} "
                         f"(op {e['op']}, snippet {e['snippet']!r}) — re-anchor on a longer, "
                         f"UNIQUE snippet copied exactly from the program above.")
            # Append to the first text part so no image/metric is displaced.
            for part in fb:
                if part.get("type") == "text":
                    part["text"] = part["text"] + note
                    break
            else:
                fb.append({"type": "text", "text": note})
            messages.append({"role": "user", "content": fb})

        if render is not None:
            prev_render = render
        if svg and not err:
            prev_svg = svg

    # Report (krepl-shaped), plus the incremental-specific per-turn AR/edit columns.
    mses = [r["mse"] for r in traj if isinstance(r["mse"], (int, float))]
    cps = [r["checklist_present"] for r in traj if isinstance(r.get("checklist_present"), (int, float))]
    aqs = [r["anchored_quality"] for r in traj if isinstance(r.get("anchored_quality"), (int, float))]

    def _spread(xs):
        return round(max(xs) - min(xs), 3) if xs else None

    report = {
        "prefix": prefix, "seed": seed, "rounds": rounds, "mode": "incremental_edit_repl",
        "trajectory": traj, "checklist": checklist,
        "oneshot_mse": traj[0]["mse"], "final_mse": traj[-1]["mse"],
        "best_mse": (min(mses) if mses else None),
        "mse_drop_oneshot_to_best": (round(traj[0]["mse"] - min(mses), 6)
                                     if (mses and traj[0]["mse"] is not None) else None),
        "anchored_traj": aqs,
        "anchored_first_last": ([aqs[0], aqs[-1]] if aqs else None),
        "anchored_spread": _spread(aqs),
        "anchored_rises": (aqs[-1] > aqs[0]) if len(aqs) >= 2 else None,
        "checklist_present_traj": cps,
        "checklist_first_last": ([cps[0], cps[-1]] if cps else None),
        "checklist_spread": _spread(cps),
        # incremental-specific: the AR (completion) per turn and the edit accounting.
        "out_tokens_traj": [r["out_tokens"] for r in traj],
        "op_count_traj": [r["op_count"] for r in traj],
        "applied_traj": [r["n_applied"] for r in traj],
        "errors_traj": [r["n_errors"] for r in traj],
        "lines_changed_traj": [r["lines_changed_vs_prev"] for r in traj],
        "mean_change_between_turns": (round(float(np.mean(
            [r["mse_vs_prev"] for r in traj if isinstance(r.get("mse_vs_prev"), (int, float))])), 6)
            if any(isinstance(r.get("mse_vs_prev"), (int, float)) for r in traj) else None),
    }
    (out_dir / f"{prefix}_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main():
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=4, help="edit turns after the one-shot")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    frames = [(f"{pathlib.Path(f).parent.name}_{pathlib.Path(f).stem}", load_target_from_path(f))
              for f in args.frames]
    print(f"[krepl_repl] {len(frames)} images x {len(args.seeds)} seeds, "
          f"{args.rounds+1} turns, max_tokens={args.max_tokens}", flush=True)
    t0 = time.time()
    for stem, tgt in frames:
        for s in args.seeds:
            rep = refine_incremental(tgt, args.rounds, s, args.out_root,
                                     f"{stem}_s{s}", args.max_tokens)
            print(f"\n  {rep['prefix']}", flush=True)
            print(f"    mse:        {' -> '.join(str(r['mse']) for r in rep['trajectory'])}", flush=True)
            print(f"    out_tokens: {rep['out_tokens_traj']}", flush=True)
            print(f"    ops/appl/err: {rep['op_count_traj']} / {rep['applied_traj']} / {rep['errors_traj']}", flush=True)
            print(f"    anchored:   {rep['anchored_traj']}", flush=True)
    print(f"\n[krepl_repl] wall {time.time()-t0:.0f}s -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
