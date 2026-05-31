#!/usr/bin/env python3
"""k-shot edit-in-REPL harness: drive the RESIDUAL down across turns.

One clean harness, composed from the canonical pieces (no inlined dupes):
  render  : svg_render.render_svg (headless Chromium, full SVG spec)
  protocol: shared pyrepl tag (<pyrepl> ... </pyrepl>) — the model emits its FULL
            program in one block each turn (full re-emission is this harness's design)
  exec    : edit_elicit._extract_program (pyrepl-primary, tolerant fallback) +
            _run_program (fresh pyrepl.PyRepl per turn -> stateless full-rewrite)
  metrics : hresidual (mse/ssim + false-color residual & delta)
  judge   : judge.feature_residual (9-axis comparative vector, >6 dims)
  client  : elicit.call_lm (streaming + usage)

Each turn the model sees its render, the false-color RESIDUAL (bright = wrong vs
reference), the DELTA (what it changed since last turn), mse/ssim, and the 9-axis
match vector — and is told to make a SUBSTANTIAL revision that darkens the brightest
residual and raises the weakest axes. EVERY intermediate image is dumped:
  {prefix}_reference.png
  {prefix}_t{NN}.py / _t{NN}_render.png / _t{NN}_residual.png / _t{NN}_delta.png
A near-black delta or a flat metric trajectory is the visible tell that the harness
gave the model the wrong instruction — review elicitation, don't excuse the model.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time

import numpy as np

import edit_elicit as _E                                            # sets sys.path
from edit_elicit import _extract_program, _run_program, _numbered   # noqa: E402
# Shared pyrepl core: _extract_program/_run_program (imported above) ALREADY route
# their extraction + program-exec through pyrepl (extract_pyrepl primary path; a
# FRESH PyRepl per call -> krepl's full-rewrite, stateless-per-turn semantics are
# preserved). We additionally switch the MODEL-FACING PROTOCOL to the <pyrepl> tag
# so the model emits the clean, unambiguous block the shared extractor captures.
from pyrepl import PYREPL_TAG_HELP                                  # noqa: E402
from svg_refinement_loop import render_svg, image_to_data_url, load_target_from_path  # noqa: E402
from elicit import call_lm                                          # noqa: E402
import judge as _judge                                              # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hresidual as R                                               # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import batch_scaler as bs                                           # noqa: E402

# Fixed STRONG-reconstruction anchor for the calibrated (anchored) judge — the
# ablation winner. The WEAK anchor is a synthetic flat field (built per-run). This
# is a hand-off knob: a clearly-faithful reconstruction the judge calibrates "9/10"
# against. Default = the visually-confirmed strong render from krepl_v2b.
_ANCHOR_GOOD = str(pathlib.Path(__file__).resolve().parents[2]
                   / "output_data/svg_runs/krepl_v2b/E4qMxJJAszg_frame_0010_s42_t04_render.png")

SYSTEM = (
    "You reconstruct a REFERENCE image as an SVG, built by a Python program that sets the "
    "string variable `svg`. You refine it across turns. Each turn you are shown: your current "
    "render; a false-color RESIDUAL where BRIGHT marks regions your render gets WRONG or MISSING "
    "versus the reference and dark marks matches; a DELTA showing what changed since last turn; "
    "the MSE and SSIM; and a 9-axis match score (1-5).\n\n"
    "Your job each turn: make a SUBSTANTIAL revision to the program so the NEXT render is clearly "
    "closer to the reference — darken the brightest residual regions and raise your weakest axes. "
    "Recolour, move, resize, add, or remove shapes as needed. Do NOT make cosmetic tweaks: if a "
    "region is bright in the residual, change it decisively this turn. `np` and the reference "
    "array `target` (H,W,3 uint8 RGB) are available — measure it rather than guessing. "
    "Emit your FULL updated program in one <pyrepl> ... </pyrepl> block (this is a full "
    "RE-EMISSION every turn — restate the entire program, not a delta).\n\n"
    + PYREPL_TAG_HELP
)


def _feedback(turn, source, render, stats, cl, aq, err, ref, prev):
    if render is None:
        return [{"type": "text", "text":
                 f"turn {turn}: your program FAILED ({err}). Fix it and emit a full working "
                 f"program in one <pyrepl> ... </pyrepl> block. Your current program:\n"
                 + _numbered(source)[:4000]}]
    c = cl or {}
    missing = c.get("missing") or []
    partial = c.get("partial") or []
    cov = f"{c.get('present_count')}/{c.get('total')}" if c else "?"
    qual = f"{aq:.0f}/10" if isinstance(aq, (int, float)) else "?"
    txt = (f"turn {turn}: faithfulness score = {qual} (calibrated 1-10; push it toward 10). "
           f"mse={stats['mse']} ssim={stats['ssim']} changed_since_prev(mse)={stats.get('mse_vs_prev')}.\n"
           f"reference-content reconstructed: {cov}.\n"
           f"STILL MISSING (add these): {missing}\n"
           f"only PARTIAL (fix colour/place/size): {partial}\n"
           "Below: current render, then RESIDUAL (bright = wrong/missing), then DELTA (what you "
           "changed). Make a SUBSTANTIAL revision this turn that ADDS the missing elements and "
           "fixes the partial ones to raise your faithfulness score — not a cosmetic tweak. "
           "Your current program:\n"
           + _numbered(source)[:4000])
    return [
        {"type": "text", "text": txt},
        {"type": "text", "text": "current render:"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(render)}},
        {"type": "text", "text": "RESIDUAL (bright = wrong/missing vs reference):"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(R.false_color_residual(ref, render))}},
        {"type": "text", "text": "DELTA (bright = what changed since last turn):"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(R.false_color_delta(prev, render))}},
    ]


def refine(target, rounds, seed, out_dir, prefix, max_tokens=16000):
    prev_render = None
    W, H = target.size
    tgt_arr = np.asarray(target.convert("RGB"))
    out_dir.mkdir(parents=True, exist_ok=True)
    target.save(out_dir / f"{prefix}_reference.png")

    def jchat(msgs):
        return call_lm(msgs, 512, 0.0, seed)[0]

    # fixed per-image checklist of salient reference elements (computed ONCE, seed-0
    # so it is IDENTICAL across seeds of the same image -> coverage is comparable).
    # The per-turn matching uses the run seed, so its variance is what we study.
    checklist = _judge.salient_checklist(lambda m: call_lm(m, 512, 0.0, 0)[0], target) or []
    (out_dir / f"{prefix}_checklist.json").write_text(json.dumps(checklist, ensure_ascii=False, indent=2))

    # Anchored (calibrated) judge — the ablation winner: a 1-10 faithfulness scalar
    # calibrated against a STRONG and a WEAK reconstruction shown in-context, which
    # recovers the dynamic range an absolute judge crushes. Anchors are fixed.
    from PIL import Image as _Image
    good_anchor = (_Image.open(_ANCHOR_GOOD).convert("RGB")
                   if _ANCHOR_GOOD and pathlib.Path(_ANCHOR_GOOD).exists() else None)
    bad_anchor = _Image.new("RGB", (good_anchor.size if good_anchor else (320, 180)), (128, 128, 128))

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"REFERENCE image ({W}x{H}). Write your first Python program "
             f"that sets `svg`, inside one <pyrepl> ... </pyrepl> block — your strongest one-shot "
             f"reconstruction. You will then refine it (re-emitting the FULL program each turn)."},
            {"type": "image_url", "image_url": {"url": image_to_data_url(target)}}]},
    ]
    source: list[str] = ["svg = ''"]
    traj = []
    for t in range(rounds + 1):
        text, finish, usage = call_lm(messages, max_tokens, 1.0, seed + t)
        messages.append({"role": "assistant", "content": text})
        prog = _extract_program(text)
        if prog is None and t == 0:                     # never start blank: retry once
            messages.append({"role": "user", "content":
                "No program found. Reply with ONLY a <pyrepl> ... </pyrepl> block that sets `svg`."})
            text, finish, usage = call_lm(messages, max_tokens, 1.0, seed + 9000 + t)
            messages.append({"role": "assistant", "content": text})
            prog = _extract_program(text)
        if prog is not None:
            source = prog.split("\n")                    # else keep prior program (no blanking)
        svg, err = _run_program("\n".join(source), tgt_arr)
        render = None
        if svg and not err:
            try:
                render = render_svg(svg, W, H)
            except Exception as e:
                err = repr(e)[:200]
        # ---- DUMP EVERYTHING ----
        (out_dir / f"{prefix}_t{t:02d}.py").write_text("\n".join(source))
        stats, cl, aq = {"mse": None, "ssim": None}, None, None
        if render is not None:
            render.save(out_dir / f"{prefix}_t{t:02d}_render.png")
            R.false_color_residual(target, render).save(out_dir / f"{prefix}_t{t:02d}_residual.png")
            R.false_color_delta(prev_render, render).save(out_dir / f"{prefix}_t{t:02d}_delta.png")
            stats = R.residual_stats(target, render, prev_render)
            cl = _judge.checklist_match(jchat, target, render, checklist) if checklist else None
            aq = (_judge.anchored_score(jchat, target, render, good_anchor, bad_anchor)
                  if good_anchor is not None else None)            # ablation winner: steering signal
        row = {"turn": t, "out_tokens": usage.get("completion_tokens"),
               "finish": finish, "lines": len(source), "err": (err[:160] if err else None),
               **stats,
               "anchored_quality": aq,
               "checklist_present": (cl or {}).get("present_count"),
               "checklist_total": (cl or {}).get("total"),
               "checklist_fraction": (cl or {}).get("fraction"),
               "checklist_scores": (cl or {}).get("scores")}
        traj.append(row)
        if t < rounds:
            messages.append({"role": "user", "content":
                             _feedback(t, source, render, stats, cl, aq, err, target, prev_render)})
        if render is not None:
            prev_render = render

    # trajectory: residual should DROP, the calibrated (anchored) faithfulness score
    # and checklist coverage should RISE turn-over-turn. The anchored score is the
    # ablation-winning STEERING signal the model now optimizes against.
    mses = [r["mse"] for r in traj if isinstance(r["mse"], (int, float))]
    cps = [r["checklist_present"] for r in traj if isinstance(r.get("checklist_present"), (int, float))]
    aqs = [r["anchored_quality"] for r in traj if isinstance(r.get("anchored_quality"), (int, float))]

    def _spread(xs):
        return round(max(xs) - min(xs), 3) if xs else None
    report = {
        "prefix": prefix, "seed": seed, "rounds": rounds, "trajectory": traj,
        "checklist": checklist,
        "oneshot_mse": traj[0]["mse"], "final_mse": traj[-1]["mse"], "best_mse": (min(mses) if mses else None),
        "mse_drop_oneshot_to_best": (round(traj[0]["mse"] - min(mses), 6)
                                     if (mses and traj[0]["mse"] is not None) else None),
        # the steering signal: does the calibrated faithfulness score RISE over turns?
        "anchored_traj": aqs,
        "anchored_first_last": ([aqs[0], aqs[-1]] if aqs else None),
        "anchored_spread": _spread(aqs),
        "anchored_rises": (aqs[-1] > aqs[0]) if len(aqs) >= 2 else None,
        "checklist_present_traj": cps,
        "checklist_first_last": ([cps[0], cps[-1]] if cps else None),
        "checklist_spread": _spread(cps),
        "mean_change_between_turns": (round(float(np.mean(
            [r["mse_vs_prev"] for r in traj if isinstance(r.get("mse_vs_prev"), (int, float))])), 6)
            if any(isinstance(r.get("mse_vs_prev"), (int, float)) for r in traj) else None),
    }
    (out_dir / f"{prefix}_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=4, help="edit turns after the one-shot")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13],
                    help="multiple seeds per image -> trajectory statistics + judge-collapse view")
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    frames = [(f"{pathlib.Path(f).parent.name}_{pathlib.Path(f).stem}", load_target_from_path(f))
              for f in args.frames]
    jobs = [(stem, tgt, s) for stem, tgt in frames for s in args.seeds]

    def run(item):
        stem, tgt, s = item
        return refine(tgt, args.rounds, s, args.out_root, f"{stem}_s{s}", args.max_tokens)

    print(f"[krepl] {len(frames)} images x {len(args.seeds)} seeds = {len(jobs)} runs, "
          f"{args.rounds+1} turns, max_tokens={args.max_tokens}", flush=True)
    t0 = time.time()
    for rep in bs.saturated_map(run, jobs, ordered=False):
        print(f"\n  {rep['prefix']}", flush=True)
        print(f"    mse:       {' -> '.join(str(r['mse']) for r in rep['trajectory'])}", flush=True)
        print(f"    anchored:  {rep['anchored_traj']}  (spread {rep['anchored_spread']}, rises {rep['anchored_rises']})", flush=True)
        print(f"    checklist: {rep['checklist_present_traj']}  (spread {rep['checklist_spread']}, "
              f"/{rep['trajectory'][0].get('checklist_total')})", flush=True)
        print(f"    change-between-turns(mse): {rep['mean_change_between_turns']}", flush=True)
    print(f"\n[krepl] wall {time.time()-t0:.0f}s -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
