#!/usr/bin/env python3
"""k-shot edit-in-REPL harness: drive the RESIDUAL down across turns.

One clean harness, composed from the canonical pieces (no inlined dupes):
  render  : svg_render.render_svg (headless Chromium, full SVG spec)
  exec    : edit_elicit._extract_program (tolerant) + _run_program (in-process)
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
from svg_refinement_loop import render_svg, image_to_data_url, load_target_from_path  # noqa: E402
from elicit import call_lm                                          # noqa: E402
import judge as _judge                                              # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hresidual as R                                               # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import batch_scaler as bs                                           # noqa: E402

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
    "Emit your FULL updated program in one ```python``` block."
)


def _feedback(turn, source, render, stats, jv, err, ref, prev):
    if render is None:
        return [{"type": "text", "text":
                 f"turn {turn}: your program FAILED ({err}). Fix it and emit a full working "
                 f"```python``` program. Your current program:\n" + _numbered(source)[:4000]}]
    j = jv or {}
    axes = ", ".join(f"{k}={j.get(k)}" for k in _judge.RESIDUAL_AXES)
    worst = j.get("worst", "?")
    txt = (f"turn {turn}: mse={stats['mse']} ssim={stats['ssim']} "
           f"changed_since_prev(mse)={stats.get('mse_vs_prev')}.\n"
           f"match (1-5): {axes}.  weakest axis: {worst}.\n"
           "Below: current render, then RESIDUAL (bright = wrong/missing), then DELTA (what you "
           "changed). Make a SUBSTANTIAL revision that darkens the brightest residual regions and "
           "lifts your weakest axes — not a cosmetic tweak. Your current program:\n"
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

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"REFERENCE image ({W}x{H}). Write your first Python program "
             f"that sets `svg` — your strongest one-shot reconstruction. You will then refine it."},
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
                "No program found. Reply with ONLY a ```python``` block that sets `svg`."})
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
        stats, jv = {"mse": None, "ssim": None}, None
        if render is not None:
            render.save(out_dir / f"{prefix}_t{t:02d}_render.png")
            R.false_color_residual(target, render).save(out_dir / f"{prefix}_t{t:02d}_residual.png")
            R.false_color_delta(prev_render, render).save(out_dir / f"{prefix}_t{t:02d}_delta.png")
            stats = R.residual_stats(target, render, prev_render)
            jv = _judge.feature_residual(jchat, target, render)
        row = {"turn": t, "out_tokens": usage.get("completion_tokens"),
               "finish": finish, "lines": len(source), "err": (err[:160] if err else None),
               **stats, "judge": jv, "judge_mean": (jv or {}).get("mean")}
        traj.append(row)
        if t < rounds:
            messages.append({"role": "user", "content":
                             _feedback(t, source, render, stats, jv, err, target, prev_render)})
        if render is not None:
            prev_render = render

    # trajectory deltas: residual should DROP, change-vs-prev should be LARGE, judge should RISE
    mses = [r["mse"] for r in traj if isinstance(r["mse"], (int, float))]
    jms = [r["judge_mean"] for r in traj if isinstance(r["judge_mean"], (int, float))]
    report = {
        "prefix": prefix, "rounds": rounds, "trajectory": traj,
        "oneshot_mse": traj[0]["mse"], "final_mse": traj[-1]["mse"],
        "best_mse": (min(mses) if mses else None),
        "mse_drop_oneshot_to_best": (round(traj[0]["mse"] - min(mses), 6)
                                     if (mses and traj[0]["mse"] is not None) else None),
        "judge_mean_first": (jms[0] if jms else None), "judge_mean_last": (jms[-1] if jms else None),
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    frames = [(f"{pathlib.Path(f).parent.name}_{pathlib.Path(f).stem}", load_target_from_path(f))
              for f in args.frames]

    def run(item):
        stem, tgt = item
        return refine(tgt, args.rounds, args.seed, args.out_root, stem, args.max_tokens)

    print(f"[krepl] {len(frames)} images, {args.rounds+1} turns, max_tokens={args.max_tokens}", flush=True)
    t0 = time.time()
    for rep in bs.saturated_map(run, frames, ordered=False):
        tj = rep["trajectory"]
        mtraj = " -> ".join(f"{r['mse']}" for r in tj)
        jtraj = " -> ".join(f"{r['judge_mean']}" for r in tj)
        print(f"\n  {rep['prefix']}", flush=True)
        print(f"    mse:   {mtraj}", flush=True)
        print(f"    judge: {jtraj}", flush=True)
        print(f"    change-between-turns(mse): {rep['mean_change_between_turns']}  "
              f"drop oneshot->best: {rep['mse_drop_oneshot_to_best']}", flush=True)
    print(f"\n[krepl] wall {time.time()-t0:.0f}s -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
