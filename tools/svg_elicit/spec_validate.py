#!/usr/bin/env python3
"""Spec validation — the ceiling question, run at the engine's kernel width.

QUESTION: given an API that lets Gemma-4 ADDITIVELY compose detail into a
persisted SVG-generating python program turn-over-turn, guided in-loop by BOTH
the MSE-delta scalar AND a same-model JOINT correspondence judge
(composition / forms / color_texture), can it push the MSE residual BELOW its own
round-0 one-shot, and does it compose genuinely more detail, over N turns?

This driver fans (frames × seeds) tools-mode rollouts through batch_scaler so the
run SATURATES the engine's kernel batch width instead of underfilling it (the bug
that prompted this: 2 frames → 2 streams on an 8-wide kernel). It then reads each
rollout's trajectory and reports, per cell and in aggregate:
  - mse per round (round 0 = one-shot ceiling) → does the residual descend?
  - detail accumulation (source linecount round0 → final) → more detail composed?
  - judge axes per round (composition/forms/color_texture) → does correspondence rise?
  - mse_drop_vs_oneshot → did multi-turn beat the one-shot?

Honest-reporting contract: this is VALIDATION, not proof — small n, the judge is
the same model (circular), single seed family. Caveats travel with the numbers.

Usage:
  GEMMA_BASE=http://127.0.0.1:8001 \\
    uv run --with numpy --with pillow --with scikit-image --with playwright \\
      python tools/svg_elicit/spec_validate.py \\
      --frames F1.png F2.png F3.png F4.png --seeds 42 7 \\
      --out-root output_data/svg_runs/spec_validate
"""
from __future__ import annotations
import argparse, json, pathlib, statistics as st, sys, time

import edit_elicit as E                                       # sets sys.path
from svg_refinement_loop import load_target_from_path         # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # tools/ for batch_scaler
import batch_scaler as bs                                     # noqa: E402  saturate kernel width

_JUDGE_AXES = ("composition", "forms", "color_texture")


def _judge_mean(row: dict):
    vs = [row.get(a) for a in _JUDGE_AXES]
    vs = [v for v in vs if isinstance(v, (int, float))]
    return round(st.mean(vs), 3) if vs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--fill", type=int, default=1,
                    help="overschedule multiplier on the engine kernel width "
                         "(1=saturate; concurrency itself comes from batch_scaler)")
    ap.add_argument("--turn0", choices=["simple", "rich"], default="rich",
                    help="round-0 elicitation strategy (default rich: strong first attempt)")
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    # Key by parent/stem so identically-named frames from different source dirs
    # (e.g. frame_0010.png under four different videos) stay DISTINCT — a bare
    # stem collapses them into one and silently underfills the batch.
    def _label(f):
        p = pathlib.Path(f)
        return f"{p.parent.name}_{p.stem}"
    targets = {_label(f): load_target_from_path(f) for f in args.frames}
    if len(targets) < len(args.frames):
        print(f"[spec_validate] WARNING: {len(args.frames)} frames collapsed to "
              f"{len(targets)} distinct keys — duplicate parent/stem labels", flush=True)
    jobs = [(stem, seed) for stem in targets for seed in args.seeds]

    def run(job):
        stem, seed = job
        prefix = f"{stem}_s{seed}"
        try:
            rep = E.run_rollout(targets[stem], "tools", args.rounds, args.max_tokens,
                                1.0, seed, args.out_root, prefix,   # judge ON by default
                                turn0=args.turn0)
        except Exception as e:
            import traceback
            rep = {"best_mse": None, "error": repr(e), "tb": traceback.format_exc(limit=3)}
        rep.update({"frame": stem, "seed": seed})
        return rep

    print(f"[spec_validate] {len(jobs)} rollouts, {args.rounds} rounds each "
          f"(frames={list(targets)} seeds={args.seeds})", flush=True)
    t0 = time.time()
    results = []
    for r in bs.saturated_map(run, jobs, fill=args.fill, ordered=False):
        results.append(r)
        if r.get("error"):
            print(f"  {r['frame']}/s{r['seed']}  ERROR {r['error']}", flush=True)
            continue
        tj = r.get("trajectory") or []
        jr0, jrN = (tj[0] if tj else {}), (tj[-1] if tj else {})
        print(f"  {r['frame']}/s{r['seed']}  oneshot_mse={r.get('oneshot_mse')} "
              f"best_mse={r.get('best_mse')} (round {r.get('best_round')}) "
              f"drop_vs_oneshot={r.get('mse_drop_vs_oneshot')}  "
              f"lines {r.get('lines_round0')}->{r.get('lines_final')} "
              f"(+{r.get('detail_accumulation')})  "
              f"judge {_judge_mean(jr0)}->{_judge_mean(jrN)}  "
              f"edits {r.get('n_edits_applied')}/{r.get('n_edit_errors')}err", flush=True)
    wall = time.time() - t0

    # ── aggregate: per-round means across cells, and the headline deltas ──────
    ok = [r for r in results if not r.get("error") and r.get("trajectory")]
    by_round: dict[int, dict] = {}
    for r in ok:
        for row in r["trajectory"]:
            d = by_round.setdefault(row["round"], {"mse": [], "lines": [], "judge": []})
            if isinstance(row.get("mse"), (int, float)):
                d["mse"].append(row["mse"])
            if isinstance(row.get("lines"), (int, float)):
                d["lines"].append(row["lines"])
            jm = _judge_mean(row)
            if jm is not None:
                d["judge"].append(jm)

    def mean(xs):
        return round(st.mean(xs), 5) if xs else None

    per_round = []
    for rnd in sorted(by_round):
        d = by_round[rnd]
        per_round.append({"round": rnd, "n": len(d["mse"]),
                          "mse_mean": mean(d["mse"]), "lines_mean": mean(d["lines"]),
                          "judge_mean": mean(d["judge"])})

    def cell_mean(key):
        xs = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
        return round(st.mean(xs), 5) if xs else None

    n_beat = sum(1 for r in ok
                 if isinstance(r.get("mse_drop_vs_oneshot"), (int, float))
                 and r["mse_drop_vs_oneshot"] > 0)
    summary = {
        "n_cells": len(results), "n_ok": len(ok), "wall_s": round(wall, 1),
        "frames": list(targets), "seeds": args.seeds, "rounds": args.rounds,
        "saturated_workers": bs.target_workers(len(jobs), fill=args.fill),
        "kernel_batch": bs.kernel_batch(),
        "mean_oneshot_mse": cell_mean("oneshot_mse"),
        "mean_best_mse": cell_mean("best_mse"),
        "mean_mse_drop_vs_oneshot": cell_mean("mse_drop_vs_oneshot"),
        "mean_detail_accumulation": cell_mean("detail_accumulation"),
        "cells_beating_oneshot": f"{n_beat}/{len(ok)}",
        "per_round": per_round,
        "cells": [{k: r.get(k) for k in
                   ("frame", "seed", "oneshot_mse", "best_mse", "best_round",
                    "mse_drop_vs_oneshot", "detail_accumulation",
                    "judge_round0", "judge_final", "n_edits_applied", "n_edit_errors",
                    "error")} for r in results],
    }
    (args.out_root / "spec_validate_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"\n=== spec_validate — {len(ok)}/{len(results)} ok, wall {wall:.0f}s, "
          f"saturated {summary['saturated_workers']} workers ===")
    print(f"  one-shot MSE (round 0 ceiling) : {summary['mean_oneshot_mse']}")
    print(f"  best MSE (any round)           : {summary['mean_best_mse']}")
    print(f"  mean drop vs one-shot          : {summary['mean_mse_drop_vs_oneshot']} "
          f"(cells beating one-shot: {summary['cells_beating_oneshot']})")
    print(f"  mean detail accumulation       : {summary['mean_detail_accumulation']} lines")
    print(f"  per-round (round: n  mse  lines  judge):")
    for pr in per_round:
        print(f"    r{pr['round']:>2}  n={pr['n']}  mse={pr['mse_mean']}  "
              f"lines={pr['lines_mean']}  judge={pr['judge_mean']}")
    print(f"-> {args.out_root}/spec_validate_summary.json")


if __name__ == "__main__":
    main()
