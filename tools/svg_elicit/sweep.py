#!/usr/bin/env python3
"""Confirm-sweep driver for repl_elicit's ablation axes.

Runs (cell x frame x seed) rollouts CONCURRENTLY via a thread pool — each
rollout is its own stream, so the engine batches their decodes (non-serial,
exercises the batched backend). Then aggregates best MSE/SSIM/judge/passes/
code-errors per cell (mean over frames/seeds).

Cells isolate one lever each against a full-stack anchor (2b.kshot.user.f3):
  control            : one-shot prior baseline (no kernel)
  2b.kshot.user.f3   : ANCHOR — kernel + target access + kshot + user voice + forced 3
  2a.kshot.user.f3   : - target access   (anchor - 2a  => instrument value)
  2b.kshot.terse.f3  : - user voice       (anchor - terse => voice value)
  2b.kshot.user.f0   : - forced depth     (anchor - voluntary => forced value)
  2b.plain.user.f3   : - kshot            (anchor - plain => elicitation value)

Outputs under output_data/ (never /tmp).
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, pathlib, statistics as st, time

import repl_elicit as R                                   # sets sys.path for svg_refinement_loop
from svg_refinement_loop import load_target_from_path     # noqa: E402

# (label, arm, elicit, voice, min_passes)
CELLS = [
    ("control",           "control", "plain", "terse", 0),
    ("2b.kshot.user.f3",  "2b", "kshot", "user",  3),
    ("2a.kshot.user.f3",  "2a", "kshot", "user",  3),
    ("2b.kshot.terse.f3", "2b", "kshot", "terse", 3),
    ("2b.kshot.user.f0",  "2b", "kshot", "user",  0),
    ("2b.plain.user.f3",  "2b", "plain", "user",  3),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    targets = {pathlib.Path(f).stem: (load_target_from_path(f), pathlib.Path(f).stem)
               for f in args.frames}
    jobs = [(label, arm, el, vo, mp, stem, seed)
            for stem in targets
            for seed in args.seeds
            for (label, arm, el, vo, mp) in CELLS]

    def run(job):
        label, arm, el, vo, mp, stem, seed = job
        tgt, _ = targets[stem]
        prefix = f"{stem}_{label}_s{seed}"
        try:
            rep = R.run_rollout(tgt, arm, args.max_turns, args.max_tokens, 1.0, seed,
                                args.out_root, prefix, elicit=el, min_passes=mp, voice=vo)
        except Exception as e:
            rep = {"best_ssim": None, "error": repr(e)}
        rep.update({"label": label, "frame": stem, "seed": seed})
        return rep

    print(f"[sweep] {len(jobs)} jobs, {args.workers} concurrent")
    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rep in ex.map(run, jobs):
            results.append(rep)
            print(f"  {rep['frame']:>10}/{rep['label']:<18} s{rep['seed']} "
                  f"ssim={rep.get('best_ssim')} judge={rep.get('judge_faithfulness')} "
                  f"passes={rep.get('accepted_passes')} code={rep.get('code_calls')}/"
                  f"{rep.get('code_errors')}err {rep.get('error','')}", flush=True)
    wall = time.time() - t0

    def m(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return round(st.mean(v), 4) if v else None

    agg = {}
    for (label, *_rest) in CELLS:
        rs = [r for r in results if r["label"] == label and r.get("best_ssim") is not None]
        if not rs:
            agg[label] = {"n": 0}; continue
        agg[label] = {
            "n": len(rs), "ssim": m(rs, "best_ssim"), "mse": m(rs, "best_mse"),
            "judge": m(rs, "judge_faithfulness"), "passes": m(rs, "accepted_passes"),
            "code_calls": m(rs, "code_calls"),
            "code_err_total": sum(r.get("code_errors", 0) for r in rs),
        }

    (args.out_root / "sweep_summary.json").write_text(json.dumps(
        {"cells": [list(c) for c in CELLS], "seeds": args.seeds,
         "frames": list(targets), "wall_s": round(wall, 1),
         "per_cell": agg, "results": results}, indent=2, default=str))

    print(f"\n=== per-cell means (n frames*seeds) — wall {wall:.0f}s ===")
    for label, a in agg.items():
        if a.get("n"):
            print(f"  {label:<20} n={a['n']} ssim={a['ssim']} mse={a['mse']} "
                  f"judge={a['judge']} passes={a['passes']} code_err={a['code_err_total']}")
        else:
            print(f"  {label:<20} (no valid results)")
    print(f"-> {args.out_root}/sweep_summary.json")


if __name__ == "__main__":
    main()
