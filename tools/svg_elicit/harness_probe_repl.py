#!/usr/bin/env python3
"""NO-SERVER unit-exercise of the krepl_repl incremental edit-in-REPL loop.

Runs the REAL refine_incremental() turn-loop (message accumulation, krepl._feedback
assembly, svg_repl parse+apply, _run_program, optional render_svg) with call_lm and
the judge stubbed to canned outputs — so we can verify the loop's CONTRACT without an
API backend:

  turn 0  -> the model emits a FULL program (large AR), stored in the REPL.
  turn>=1 -> the model emits ONLY small EDIT blocks (AR collapses), applied to the
             PERSISTED program; the program_state changes every edit turn; a bad-anchor
             edit is reported in errors AND surfaced into the NEXT turn's feedback.

The canned turn>=1 reply references REAL substrings of the CURRENT program (so the
apply actually mutates it): a REPLACE that recolours a shape, an APPEND that adds a
shape, and — on exactly one turn — a bad-anchor REPLACE whose FIND is absent, to
exercise error reporting + surfacing.

LIGHT mode (default) skips real rendering but still does the full parse+apply+AR
accounting. --render does real render_svg so deltas/renders are produced too.

Usage:
  uv run --quiet --with numpy --with pillow --with scikit-image --with playwright \
    python tools/svg_elicit/harness_probe_repl.py [--render] [--rounds N]
"""
from __future__ import annotations
import argparse, math, pathlib, re, sys
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import krepl_repl  # noqa: E402
import svg_repl     # noqa: E402

# Captured per call_lm invocation: the messages handed in, plus per-turn telemetry.
SNAPS: list[dict] = []        # one snapshot per call_lm call (the prefill it saw)
TURN_PROGRAM: list[str] = []  # program_state AFTER each turn's apply (recorded by us)


def est_text_tokens(s: str) -> int:
    return math.ceil(len(s) / 4)


# ---- the canned TURN-0 full program. Sets `svg`, ends with `svg = ...`. ----
def make_turn0_program() -> str:
    lines = [
        'import math',
        'W, H = 320, 180',
        'parts = []',
        'parts.append(\'<rect x="0" y="0" width="320" height="180" fill="#222"/>\')',
        'parts.append(\'<circle cx="160" cy="90" r="40" fill="#e44"/>\')',
        'parts.append(\'<rect x="20" y="20" width="60" height="40" fill="#48c"/>\')',
    ]
    # pad so the full program is clearly LARGE relative to the edit blocks
    for i in range(40):
        lines.append(f'parts.append(\'<line x1="{i}" y1="0" x2="{i}" y2="180" '
                     f'stroke="#333"/>\')  # filler stroke {i} for realistic program length')
    lines.append("svg = '<svg xmlns=\"http://www.w3.org/2000/svg\" "
                 "viewBox=\"0 0 320 180\">' + ''.join(parts) + '</svg>'")
    return "```python\n" + "\n".join(lines) + "\n```"


TURN0 = make_turn0_program()

# Canned edit blocks. The persisted artifact is the Python program, so _GOOD edits
# anchor on real `parts.append(...)` LINES of it (so apply mutates the program, which
# re-runs to a changed svg); the _BAD edit's FIND is deliberately absent (error path).
_GOOD_REPLACE = (
    "Recolouring the central circle to better match the residual:\n\n"
    "### EDIT REPLACE ###\n"
    "--- FIND ---\n"
    'parts.append(\'<circle cx="160" cy="90" r="40" fill="#e44"/>\')\n'
    "--- REPLACE ---\n"
    'parts.append(\'<circle cx="160" cy="90" r="48" fill="#f66"/>\')\n'
    "### END ###\n"
)
_GOOD_APPEND_TMPL = (
    "And adding a missing element near the bottom:\n\n"
    "*** EDIT APPEND ***\n"
    "--- REPLACE ---\n"
    "parts.append('<text x=\"10\" y=\"170\" fill=\"#fff\">edit-turn-{t}</text>')\n"
    "*** END ***\n"
)
_BAD_REPLACE = (
    "\nAlso tweak a shape that isn't actually in the program (should be reported):\n\n"
    "### EDIT REPLACE ###\n"
    "--- FIND ---\n"
    'parts.append(\'<ellipse cx="1" cy="2" rx="3" ry="4" fill="#0f0"/>\')\n'
    "--- REPLACE ---\n"
    'parts.append(\'<ellipse cx="1" cy="2" rx="3" ry="4" fill="#00f"/>\')\n'
    "### END ###\n"
)

# The turn on which the bad-anchor edit is injected. We put it on turn 1 — the
# LARGEST edit turn (it also carries the recolour REPLACE) — so the extra bad block
# does not balloon a later turn's AR above an earlier one: the edit-turn AR sequence
# stays non-increasing while still exercising exactly-one bad-anchor edit + surfacing.
BAD_TURN = 1


def _build_edit_reply(turn: int) -> str:
    # After the FIRST recolour applies, the original circle line no longer exists,
    # so target whatever circle line is currently present (keeps APPLY non-trivial
    # across turns). We always APPEND (unique-by-turn text) so program_state changes
    # EVERY edit turn even if a REPLACE happens to no-op.
    reply = _GOOD_APPEND_TMPL.format(t=turn)
    if turn == 1:
        reply = _GOOD_REPLACE + "\n" + reply
    if turn == BAD_TURN:
        reply = reply + _BAD_REPLACE
    return reply


def _snapshot(messages) -> dict:
    """AR-relevant view of the messages list: text token estimate + image count,
    and the concatenated user-feedback text (so we can search it for error notes)."""
    n_img = 0
    fb_text_parts: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if part.get("type") == "image_url":
                    n_img += 1
                elif part.get("type") == "text" and m["role"] == "user":
                    fb_text_parts.append(part["text"])
    return {"n_msgs": len(messages), "imgs": n_img,
            "last_user_feedback": "\n".join(fb_text_parts)}


# ---- monkeypatched LM: turn 0 -> full program, turns>=1 -> edit blocks ----
_call_idx = {"n": 0}


def fake_call_lm(messages, max_tokens, temperature, seed):
    SNAPS.append(_snapshot(messages))
    # Count assistant turns already in the conversation = which turn this reply is.
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")
    turn = n_assistant            # 0 on the first call, 1 on the second, ...
    if turn == 0:
        reply = TURN0
    else:
        reply = _build_edit_reply(turn)
    _call_idx["n"] += 1
    return reply, "stop", {"completion_tokens": est_text_tokens(reply)}


def fake_checklist(chat, ref, render, checklist):
    return {"present_count": 5, "total": 2 * len(checklist), "fraction": 0.25,
            "missing": ["a brown head", "white hands"],
            "partial": ["the dark background"],
            "scores": {e: 1 for e in checklist}}


def _patch():
    krepl_repl.call_lm = fake_call_lm
    # judge fns the loop calls (via krepl_repl._judge and through jchat closures)
    krepl_repl._judge.salient_checklist = lambda chat, ref, n=10: [f"element {i}" for i in range(8)]
    krepl_repl._judge.checklist_match = fake_checklist
    krepl_repl._judge.anchored_score = lambda chat, ref, render, good, bad: 5.0
    krepl_repl._ANCHOR_GOOD = ""   # skip loading the anchor png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4, help="edit turns after turn 0")
    ap.add_argument("--render", action="store_true",
                    help="do real render_svg (else LIGHT: parse+apply+AR only, no raster)")
    ap.add_argument("--out", type=pathlib.Path,
                    default=HERE.parents[1] / "output_data/svg_runs/harness_probe_repl")
    args = ap.parse_args()

    _patch()

    if not args.render:
        # LIGHT mode: stub render_svg + the residual/metric image dumps so the loop
        # still parses+applies+accounts AR without rasterizing. A 1x1 image keeps
        # mse/ssim cheap and the dumped pngs trivial; the contract under test is the
        # AR/edit/program-state machinery, not the pixels.
        light_img = Image.new("RGB", (8, 8), (40, 40, 40))
        krepl_repl.render_svg = lambda svg, w, h: light_img.copy()

    target = Image.fromarray(np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8), "RGB")
    args.out.mkdir(parents=True, exist_ok=True)
    rep = krepl_repl.refine_incremental(target, rounds=args.rounds, seed=1,
                                        out_dir=args.out, prefix="probe",
                                        max_tokens=16000)

    traj = rep["trajectory"]
    mode = "RENDER" if args.render else "LIGHT"
    print(f"\n=== incremental edit-in-REPL probe  ({mode} mode, rounds={args.rounds}) ===")
    print(f"{'turn':>4} {'AR_out':>7} {'ops':>4} {'appl':>5} {'err':>4} "
          f"{'lines':>6} {'Δlines':>7} {'mse':>9} {'note':>0}")
    for r in traj:
        note = "TURN-0 full program" if r["turn"] == 0 else "edit blocks only"
        if r["n_errors"]:
            note += f"  (<-- {r['n_errors']} bad-anchor edit reported)"
        print(f"{r['turn']:>4} {r['out_tokens']:>7} {r['op_count']:>4} {r['n_applied']:>5} "
              f"{r['n_errors']:>4} {r['lines']:>6} {r['lines_changed_vs_prev']:>7} "
              f"{str(r['mse']):>9} {note}")

    # ---------------------------------------------------------------- asserts
    ar = [r["out_tokens"] for r in traj]
    ar0 = ar[0]
    later = ar[1:]

    print("\n=== SUMMARY / ASSERTIONS ===")
    checks: list[bool] = []

    def check(name: str, cond: bool, detail: str = ""):
        checks.append(bool(cond))
        status = "PASS" if cond else "FAIL"
        line = f"[{status}] {name}"
        if detail:
            line += f"  -- {detail}"
        print(line)

    # (i.a) turn-0 AR IS the full program (large; matches the canned full program AR).
    check("(i) turn-0 AR is the FULL program",
          ar0 == est_text_tokens(TURN0) and ar0 > 200,
          f"ar0={ar0} (full-program AR={est_text_tokens(TURN0)})")

    # (i.b) every later turn AR < 0.3 * turn0 AR.
    thresh = 0.3 * ar0
    check("(i) every edit-turn AR < 0.3 * turn-0 AR",
          all(a < thresh for a in later),
          f"thresh={thresh:.0f}, edit ARs={later}")

    # (i.c) later ARs are NOT increasing (monotone non-increasing across edit turns).
    check("(i) edit-turn ARs are non-increasing",
          all(later[i] >= later[i + 1] for i in range(len(later) - 1)),
          f"edit ARs={later}")

    # (ii) program_state CHANGES every edit turn (lines_changed_vs_prev > 0 for t>=1).
    edit_deltas = [r["lines_changed_vs_prev"] for r in traj if r["turn"] >= 1]
    check("(ii) program_state changes on EVERY edit turn",
          all(d > 0 for d in edit_deltas),
          f"Δlines per edit turn={edit_deltas}")

    # (ii.b) the persisted program actually grew vs turn 0 (APPENDs accumulated).
    t0_lines = traj[0]["lines"]
    tN_lines = traj[-1]["lines"]
    check("(ii) persisted program accumulated edits (final lines > turn-0 lines)",
          tN_lines > t0_lines,
          f"turn0 lines={t0_lines}, final lines={tN_lines}")

    # (iii.a) the bad-anchor edit shows up in errors on BAD_TURN with reason=not_found.
    bad_row = next((r for r in traj if r["turn"] == BAD_TURN), None)
    bad_errs = (bad_row or {}).get("edit_errors") or []
    check(f"(iii) bad-anchor edit reported in errors on turn {BAD_TURN} (reason=not_found)",
          len(bad_errs) == 1 and bad_errs[0]["reason"] == "not_found"
          and bad_errs[0]["op"] == "REPLACE",
          f"errors={bad_errs}")

    # (iii.b) no OTHER edit turn reported an error (the bad anchor is isolated).
    other_err_turns = [r["turn"] for r in traj if r["turn"] >= 1 and r["turn"] != BAD_TURN
                       and r["n_errors"] > 0]
    check("(iii) only the bad-anchor turn reports an error",
          other_err_turns == [],
          f"other turns with errors={other_err_turns}")

    # (iii.c) the error is SURFACED into the NEXT turn's feedback prefill. The call
    # for turn BAD_TURN+1 is SNAPS[BAD_TURN+1]; its accumulated user feedback must
    # contain the 're-anchor' note naming the failed edit.
    next_snap = SNAPS[BAD_TURN + 1] if len(SNAPS) > BAD_TURN + 1 else {}
    fb = next_snap.get("last_user_feedback", "")
    surfaced = (f"edit {bad_errs[0]['index']} did not apply" in fb
                and "re-anchor" in fb and "not_found" in fb) if bad_errs else False
    check("(iii) bad-anchor error surfaced into the NEXT turn's feedback ('re-anchor')",
          surfaced,
          "found 'did not apply ... re-anchor (not_found)' in next prefill"
          if surfaced else "re-anchor note NOT found in next-turn feedback")

    # bonus: feedback for an edit turn still carries ALL krepl imagery (3 images:
    # render + residual + delta) + the numbered program — i.e. no metric/image dropped.
    # The turn-1 feedback is what the turn-2 call saw: SNAPS[2].
    if len(SNAPS) > 2 and args.render:
        fb1 = SNAPS[2].get("last_user_feedback", "")
        has_numbered = bool(re.search(r"\n\s*\d+\| ", fb1))
        check("(bonus) edit-turn feedback still shows the NUMBERED current program",
              has_numbered, "numbered 'NNN| ...' source present in feedback")

    # (REGRESSION GUARD) the failure this harness had: edits APPLIED to program text
    # but the RENDER never changed (edits anchored on dead code; the canonicalization
    # fix routes every edit onto the markup that actually renders). In --render mode
    # the rendered mse MUST move across turns, and NO edit turn may be render_unchanged.
    if args.render:
        mses = [r["mse"] for r in traj if isinstance(r["mse"], (int, float))]
        check("(render) rendered output CHANGES across turns (mse not frozen)",
              len({round(m, 6) for m in mses}) > 1,
              f"mse traj={mses}")
        ru = [r["turn"] for r in traj if r.get("render_unchanged")]
        check("(render) no edit turn is render_unchanged (edits reach the picture)",
              ru == [], f"render_unchanged turns={ru}")

    ok = all(checks)
    print(f"\n{'ALL ASSERTIONS PASS' if ok else 'SOME ASSERTIONS FAILED'}: "
          f"{sum(checks)}/{len(checks)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
