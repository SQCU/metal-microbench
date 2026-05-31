#!/usr/bin/env python3
"""Incremental SVG refinement on the SHARED pyrepl core — the clean replacement for
the krepl_repl '### EDIT ###' grammar. The model drives a PERSISTENT Python REPL via
<pyrepl> cells (pyrepl.PyRepl); everything else (render, residual/metrics, judge,
feedback imagery) is REUSED from the canonical modules, not re-implemented:

  extract/REPL : pyrepl.extract_pyrepl + pyrepl.PyRepl   (tag protocol, transactional)
  render       : svg_refinement_loop.render_svg
  metrics      : hresidual (mse/ssim + false-color residual & delta)
  judge        : judge.salient_checklist / checklist_match / anchored_score
  feedback     : krepl._feedback  (render + residual + delta + mse/ssim + numbered REPL)
  client       : elicit.call_lm

Every intermediate is dumped: {prefix}_reference.png, {prefix}_t{NN}.py (the canonical
REPL cells), _t{NN}_render.png / _residual.png / _delta.png, {prefix}_report.json.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "archival"))   # svg_refinement_loop lives here

from pyrepl import PyRepl, extract_pyrepl, PYREPL_TAG_HELP   # noqa: E402
from svg_refinement_loop import render_svg, image_to_data_url, load_target_from_path  # noqa: E402
from elicit import call_lm                                   # noqa: E402
import hresidual as R                                        # noqa: E402
import judge as _judge                                       # noqa: E402
import krepl as _krepl                                       # noqa: E402  (reuse _feedback + anchor)
import batch_scaler as bs                                    # noqa: E402

_ANCHOR_GOOD = _krepl._ANCHOR_GOOD

SYSTEM = (
    "You reconstruct a REFERENCE image as an SVG by writing Python in a PERSISTENT "
    "REPL. Each turn you are shown: your current render; a false-color RESIDUAL where "
    "BRIGHT marks regions your render gets WRONG or MISSING and dark marks matches; a "
    "DELTA of what changed since last turn; the MSE and SSIM; a faithfulness score; a "
    "checklist of reference elements; and your REPL cells so far.\n\n"
    "TURN 0: emit your strongest one-shot reconstruction.\n"
    "EVERY LATER TURN: the namespace PERSISTS — the variables you defined are STILL in "
    "scope. Make a SUBSTANTIAL revision as a SMALL delta cell that mutates that state "
    "(recolour/move/resize a shape, append a missing element, fix the brightest "
    "residual) and recomputes `svg`. Do NOT rebuild the whole picture from scratch.\n\n"
    + PYREPL_TAG_HELP
)

_TURN0_TEXT = (
    "REFERENCE image ({W}x{H}). Write your TURN-0 <pyrepl> cell now — build `svg` as "
    "your strongest one-shot reconstruction. Keep the pieces in variables (e.g. a "
    "`parts` list) so later turns can tweak them incrementally."
)
_EDIT_REMINDER = (
    "Emit ONE small <pyrepl> delta cell that mutates the persisted state and recomputes "
    "`svg`. Variables from earlier turns are still defined — do not rebuild from scratch."
)


def _spread(xs):
    return round(max(xs) - min(xs), 3) if xs else None


def refine_pyrepl(target, rounds, seed, out_dir, prefix, max_tokens=8000):
    W, H = target.size
    tgt_arr = np.asarray(target.convert("RGB"))
    out_dir.mkdir(parents=True, exist_ok=True)
    target.save(out_dir / f"{prefix}_reference.png")

    def jchat(msgs):
        return call_lm(msgs, 512, 0.0, seed)[0]

    checklist = _judge.salient_checklist(lambda m: call_lm(m, 512, 0.0, 0)[0], target) or []
    (out_dir / f"{prefix}_checklist.json").write_text(json.dumps(checklist, ensure_ascii=False, indent=2))

    from PIL import Image as _Image
    good_anchor = (_Image.open(_ANCHOR_GOOD).convert("RGB")
                   if _ANCHOR_GOOD and pathlib.Path(_ANCHOR_GOOD).exists() else None)
    bad_anchor = _Image.new("RGB", (good_anchor.size if good_anchor else (320, 180)), (128, 128, 128))

    repl = PyRepl(seed_vars={"np": np, "target": tgt_arr, "W": W, "H": H})
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": _TURN0_TEXT.format(W=W, H=H)},
            {"type": "image_url", "image_url": {"url": image_to_data_url(target)}}]},
    ]

    prev_render = None
    prev_svg = None
    traj = []
    for t in range(rounds + 1):
        text, finish, usage = call_lm(messages, max_tokens, 1.0, seed + t)
        messages.append({"role": "assistant", "content": text})

        code = extract_pyrepl(text)
        if code is None and t == 0:                 # never start blank: re-ask once
            messages.append({"role": "user", "content":
                             "No <pyrepl> block found. Reply with ONLY a <pyrepl> ... </pyrepl> "
                             "cell that sets `svg`."})
            text, finish, usage = call_lm(messages, max_tokens, 1.0, seed + 9000 + t)
            messages.append({"role": "assistant", "content": text})
            code = extract_pyrepl(text)

        if code is None:
            res = {"ok": False, "error": "no <pyrepl> block in reply", "svg": repl.last_svg,
                   "rolled_back": False}
        else:
            res = repl.run_cell(code)
        svg, cell_ok, cell_err, rolled = res["svg"], res["ok"], res["error"], res["rolled_back"]

        render, err = None, (cell_err if not cell_ok else None)
        if svg:
            try:
                render = render_svg(svg, W, H)
            except Exception as e:
                err = repr(e)[:200]
                render = None

        render_unchanged = bool(t >= 1 and cell_ok and prev_svg is not None and svg == prev_svg)

        # ---- DUMP EVERYTHING ----
        (out_dir / f"{prefix}_t{t:02d}.py").write_text(repl.canonical_py or (code or ""))
        if code is not None and not cell_ok:        # keep the rejected cell for inspection
            (out_dir / f"{prefix}_t{t:02d}_rejected_cell.py").write_text(code)

        stats, cl, aq = {"mse": None, "ssim": None}, None, None
        if render is not None:
            render.save(out_dir / f"{prefix}_t{t:02d}_render.png")
            R.false_color_residual(target, render).save(out_dir / f"{prefix}_t{t:02d}_residual.png")
            R.false_color_delta(prev_render, render).save(out_dir / f"{prefix}_t{t:02d}_delta.png")
            stats = R.residual_stats(target, render, prev_render)
            cl = _judge.checklist_match(jchat, target, render, checklist) if checklist else None
            aq = (_judge.anchored_score(jchat, target, render, good_anchor, bad_anchor)
                  if good_anchor is not None else None)

        row = {"turn": t, "out_tokens": usage.get("completion_tokens"), "finish": finish,
               "cell_ok": cell_ok, "rolled_back": rolled, "cell_error": (cell_err[:160] if cell_err else None),
               "render_unchanged": render_unchanged, "err": (err[:160] if err else None),
               "n_cells": len(repl.cells), **stats, "anchored_quality": aq,
               "checklist_present": (cl or {}).get("present_count"),
               "checklist_total": (cl or {}).get("total"),
               "checklist_fraction": (cl or {}).get("fraction")}
        traj.append(row)

        if t < rounds:
            source = (repl.canonical_py or "").split("\n")
            fb = _krepl._feedback(t, source, render, stats, cl, aq, err, target, prev_render)
            note = "\n\n" + _EDIT_REMINDER
            if not cell_ok and cell_err:
                note += (f"\nYour <pyrepl> cell did NOT take effect: {cell_err}. The REPL was "
                         f"rolled back to the last working state shown above — fix the cell "
                         f"(keep it valid Python; leave a non-empty string in `svg`) and retry.")
            elif render_unchanged:
                note += ("\nWARNING: your cell ran but the RENDER DID NOT CHANGE — mutate the "
                         "elements that are bright in the residual so the picture actually moves.")
            for part in fb:
                if part.get("type") == "text":
                    part["text"] = part["text"] + note
                    break
            else:
                fb.append({"type": "text", "text": note})
            messages.append({"role": "user", "content": fb})

        if render is not None:
            prev_render = render
        if svg and cell_ok:
            prev_svg = svg

    mses = [r["mse"] for r in traj if isinstance(r["mse"], (int, float))]
    aqs = [r["anchored_quality"] for r in traj if isinstance(r.get("anchored_quality"), (int, float))]
    cps = [r["checklist_present"] for r in traj if isinstance(r.get("checklist_present"), (int, float))]
    report = {
        "prefix": prefix, "seed": seed, "rounds": rounds, "mode": "pyrepl_persistent_namespace",
        "trajectory": traj, "checklist": checklist,
        "out_tokens_traj": [r["out_tokens"] for r in traj],
        "cell_ok_traj": [r["cell_ok"] for r in traj],
        "rolled_back_traj": [r["rolled_back"] for r in traj],
        "mse_traj": mses, "mse_first_last": ([mses[0], mses[-1]] if mses else None),
        "anchored_traj": aqs, "anchored_rises": (aqs[-1] > aqs[0]) if len(aqs) >= 2 else None,
        "checklist_present_traj": cps, "checklist_spread": _spread(cps),
    }
    (out_dir / f"{prefix}_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    args = ap.parse_args()

    from PIL import Image
    items = [(f, s) for f in args.frames for s in args.seeds]
    print(f"[pyrepl_refine] {len(args.frames)} images x {len(args.seeds)} seeds = {len(items)} runs, "
          f"{args.rounds + 1} turns", flush=True)

    def run(item):
        frame, s = item
        target = Image.open(frame).convert("RGB")
        stem = pathlib.Path(frame).parent.name + "_" + pathlib.Path(frame).stem
        prefix = f"{stem}_s{s}"
        rep = refine_pyrepl(target, args.rounds, s, args.out_root, prefix, args.max_tokens)
        print(f"  {prefix}: mse={rep['mse_traj']}  cell_ok={rep['cell_ok_traj']}  "
              f"rolled={rep['rolled_back_traj']}  anchored={rep['anchored_traj']}", flush=True)
        return rep

    list(bs.saturated_map(run, items, ordered=False))


if __name__ == "__main__":
    main()
