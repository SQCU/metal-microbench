#!/usr/bin/env python3
"""NO-SERVER unit-exercise of the krepl refine() turn-loop.

Runs the REAL harness code (message accumulation, _feedback assembly, _extract_program,
_run_program, render_svg) with call_lm + the judge stubbed to canned outputs, so we can
SEE what the harness actually does with its context — without an API backend.

What it measures (the thing the live token-burden review needs):
  - per-designer-turn PREFILL the harness presents (snapshotted inside the fake call_lm)
  - how that prefill is composed: system / re-emitted programs / feedback text / images
  - whether the program is double-carried (assistant msg AND echoed in feedback)
  - whether anything is ever pruned

Token estimate: text ~= ceil(chars/4) (Gemma-ish); images counted separately and
priced at SOFT tokens/image (pass the engine-measured value via --soft).
Usage:
  uv run --with numpy --with pillow --with scikit-image --with playwright \
    python tools/svg_elicit/harness_probe.py --soft 256
"""
from __future__ import annotations
import argparse, math, pathlib, sys
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import krepl  # noqa: E402

SNAPS = []  # one entry per designer call_lm invocation: the messages it was handed


def est_text_tokens(s: str) -> int:
    return math.ceil(len(s) / 4)


def count_msg(content) -> tuple[int, int]:
    """(text_tokens, n_images) for one message's content (str or list of parts)."""
    if isinstance(content, str):
        return est_text_tokens(content), 0
    tt, ni = 0, 0
    for part in content:
        if part.get("type") == "text":
            tt += est_text_tokens(part["text"])
        elif part.get("type") == "image_url":
            ni += 1
    return tt, ni


def snapshot(messages):
    """Token-estimate + image-count of the full messages list, broken out by role/kind."""
    sys_t = prog_t = fb_t = 0
    imgs = 0
    for m in messages:
        tt, ni = count_msg(m["content"])
        imgs += ni
        if m["role"] == "system":
            sys_t += tt
        elif m["role"] == "assistant":
            prog_t += tt           # assistant turns ARE the (full) re-emitted programs
        else:
            fb_t += tt             # user feedback text (incl. echoed numbered program)
    return {"n_msgs": len(messages), "imgs": imgs,
            "sys_t": sys_t, "prog_t": prog_t, "fb_t": fb_t}


# ---- canned model + judge outputs (no server) ----
def make_canned_program(n_shapes=140) -> str:
    """A valid, ~real-length python program that sets `svg` (~1.8-2k tokens, like the live runs)."""
    lines = ["w,h = 320,180",
             "parts = ['<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 %d %d\">' % (w,h)]"]
    for i in range(n_shapes):
        x, y = (i * 7) % 300, (i * 13) % 160
        c = "#%02x%02x%02x" % ((i * 17) % 256, (i * 53) % 256, (i * 97) % 256)
        lines.append(f"parts.append('<rect x=\"{x}\" y=\"{y}\" width=\"18\" height=\"12\" "
                     f"fill=\"{c}\" opacity=\"0.8\"/>')  # shape {i}: descriptive comment to mimic real verbosity")
    lines.append("parts.append('</svg>')")
    lines.append("svg = ''.join(parts)")
    return "```python\n" + "\n".join(lines) + "\n```"


CANNED = make_canned_program()


def fake_call_lm(messages, max_tokens, temp, seed):
    SNAPS.append(snapshot(messages))
    return CANNED, "stop", {"completion_tokens": est_text_tokens(CANNED)}


def fake_checklist(chat, ref, render, checklist):
    return {"present_count": 5, "total": len(checklist), "fraction": 0.5,
            "missing": ["the brown head", "the white hands"],
            "partial": ["the dark background"], "scores": [1] * len(checklist)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--soft", type=int, default=256, help="engine-measured image soft-tokens/image")
    ap.add_argument("--out", type=pathlib.Path, default=HERE.parents[1] / "output_data/svg_runs/harness_probe")
    args = ap.parse_args()

    # monkeypatch: designer + judge -> canned, no API server touched
    krepl.call_lm = fake_call_lm
    krepl._judge.salient_checklist = lambda chat, ref, n=10: [f"element {i}" for i in range(10)]
    krepl._judge.checklist_match = fake_checklist
    krepl._judge.anchored_score = lambda chat, ref, render, good, bad: 4.0
    krepl._ANCHOR_GOOD = ""  # skip loading the anchor png

    target = Image.fromarray(np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8), "RGB")
    args.out.mkdir(parents=True, exist_ok=True)
    krepl.refine(target, rounds=args.rounds, seed=1, out_dir=args.out, prefix="probe")

    SOFT = args.soft
    print(f"\n=== designer PREFILL the harness builds, per turn (image=,{SOFT} soft tok) ===")
    print(f"{'turn':>4} {'msgs':>5} {'imgs':>5} {'img_tok':>8} {'sys':>5} {'prog_txt':>9} "
          f"{'fb_txt':>7} {'TEXT':>7} {'TOTAL':>8}")
    for i, s in enumerate(SNAPS):
        img_tok = s["imgs"] * SOFT
        text = s["sys_t"] + s["prog_t"] + s["fb_t"]
        total = text + img_tok
        print(f"{i:>4} {s['n_msgs']:>5} {s['imgs']:>5} {img_tok:>8} {s['sys_t']:>5} "
              f"{s['prog_t']:>9} {s['fb_t']:>7} {text:>7} {total:>8}")
    if SNAPS:
        f = SNAPS[-1]
        img_tok = f["imgs"] * SOFT
        text = f["sys_t"] + f["prog_t"] + f["fb_t"]
        total = text + img_tok
        print(f"\n=== FINAL-turn prefill composition (total ~{total} tok) ===")
        print(f"  re-emitted programs (assistant)   : {f['prog_t']:>6} tok  ({100*f['prog_t']/total:.0f}%)")
        print(f"  feedback text (incl. echoed prog) : {f['fb_t']:>6} tok  ({100*f['fb_t']/total:.0f}%)")
        print(f"  image soft tokens ({f['imgs']} imgs)        : {img_tok:>6} tok  ({100*img_tok/total:.0f}%)")
        print(f"  system                            : {f['sys_t']:>6} tok  ({100*f['sys_t']/total:.0f}%)")
        print(f"\n  pruned across the whole run? NO — n_msgs grew 1 -> {f['n_msgs']} monotonically.")
        print(f"  per-turn designer AR (re-emit)    : ~{est_text_tokens(CANNED)} tok EVERY turn (full rewrite, no diff).")


if __name__ == "__main__":
    main()
