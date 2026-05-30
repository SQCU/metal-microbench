#!/usr/bin/env python3
"""Score an edit_elicit run with the recalibrated metric stack.

Primary faithfulness signal = the Gemma-4 vision-encoder cosine distance to the
target (model-native, position-tolerant); MSE/SSIM/judge come from each cell's
report. Answers the two questions the 8-wide test is for:
  (1) is the turn-0 one-shot now strong (no longer a corrupted blank)?
  (2) does refinement push the PEAK below the one-shot in the vision latent?
Stratifies cells by a target-entropy proxy (simple vs dense) so the dense-scene
ceiling is reported apart from simple-scene success.

Usage:
  GEMMA_DYLIB=.../libgemma_metal.dylib uv run --with numpy --with pillow \\
    --with scikit-image python tools/svg_elicit/score_run.py --run-dir <dir>
"""
from __future__ import annotations
import argparse, glob, json, pathlib, statistics as st, sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import vision_metric as vm  # noqa: E402


def target_entropy(path: pathlib.Path) -> float:
    """Shannon entropy (bits) of the target's grayscale histogram — a cheap
    busy-ness proxy. Low ~ simple/flat, high ~ dense/textured."""
    g = np.asarray(Image.open(path).convert("L"))
    h = np.bincount(g.ravel(), minlength=256).astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=pathlib.Path, required=True)
    ap.add_argument("--entropy-split", type=float, default=7.0,
                    help="grayscale-entropy threshold splitting simple(<) vs dense(>=)")
    args = ap.parse_args()

    reports = sorted(glob.glob(str(args.run_dir / "*_report.json")))
    if not reports:
        print(f"no *_report.json under {args.run_dir}"); return
    vm.init()
    print(f"[score_run] {len(reports)} cells, vision tower ready\n")

    rows = []
    for rp in reports:
        rep = json.loads(pathlib.Path(rp).read_text())
        prefix = rep["prefix"]
        tgt = args.run_dir / f"{prefix}_target.png"
        if not tgt.exists():
            continue
        ent = target_entropy(tgt)
        et = vm.embed(tgt)
        # per-round vision cos-dist over the renders that exist
        traj = rep.get("trajectory") or []
        per = []
        for row in traj:
            nn = row["round"]
            rpng = args.run_dir / f"{prefix}_r{nn:02d}_render.png"
            if not rpng.exists():
                continue
            try:
                vis = vm.cosine_distance(et, vm.embed(rpng))
            except Exception:
                continue
            per.append({"round": nn, "vis": vis, "mse": row.get("mse"), "ssim": row.get("ssim"),
                        "judge": [row.get("composition"), row.get("forms"), row.get("color_texture")]})
        if not per:
            continue
        one = per[0]                                   # round-0 one-shot (now valid)
        peak = min(per, key=lambda x: x["vis"])        # best by vision latent
        rows.append({
            "cell": prefix, "entropy": round(ent, 3),
            "stratum": "dense" if ent >= args.entropy_split else "simple",
            "round0_parse_failed": rep.get("round0_parse_failed"),
            "turn0": rep.get("turn0"),
            "oneshot_vis": round(one["vis"], 4), "oneshot_round": one["round"],
            "peak_vis": round(peak["vis"], 4), "peak_round": peak["round"],
            "vis_gain_oneshot_minus_peak": round(one["vis"] - peak["vis"], 4),
            "refine_beats_oneshot": peak["vis"] < one["vis"] - 1e-4,
            "oneshot_ssim": one["ssim"], "peak_ssim": per[[p["round"] for p in per].index(peak["round"])]["ssim"],
            "oneshot_judge": one["judge"], "peak_judge": peak["judge"],
            "n_rounds_scored": len(per),
        })

    def agg(rs, key):
        v = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
        return round(st.mean(v), 4) if v else None

    summary = {"run_dir": str(args.run_dir), "n_cells": len(rows)}
    for label, subset in (("all", rows),
                          ("simple", [r for r in rows if r["stratum"] == "simple"]),
                          ("dense", [r for r in rows if r["stratum"] == "dense"])):
        if not subset:
            summary[label] = {"n": 0}; continue
        summary[label] = {
            "n": len(subset),
            "mean_oneshot_vis": agg(subset, "oneshot_vis"),
            "mean_peak_vis": agg(subset, "peak_vis"),
            "mean_vis_gain": agg(subset, "vis_gain_oneshot_minus_peak"),
            "cells_refine_beats_oneshot": f"{sum(r['refine_beats_oneshot'] for r in subset)}/{len(subset)}",
            "cells_parse_failed_turn0": sum(bool(r["round0_parse_failed"]) for r in subset),
        }
    summary["cells"] = rows
    out = args.run_dir / "score_run_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))

    print(f"{'cell':40} {'strat':6} {'1shot_vis':>9} {'peak_vis':>8} {'gain':>7} {'peak@':>5} parse_fail")
    for r in sorted(rows, key=lambda x: (x["stratum"], x["cell"])):
        print(f"{r['cell'][:40]:40} {r['stratum']:6} {r['oneshot_vis']:>9.4f} {r['peak_vis']:>8.4f} "
              f"{r['vis_gain_oneshot_minus_peak']:>+7.4f} {r['peak_round']:>5} "
              f"{'YES' if r['round0_parse_failed'] else '-'}")
    print()
    for label in ("all", "simple", "dense"):
        s = summary.get(label, {})
        if s.get("n"):
            print(f"  [{label:6}] n={s['n']}  1shot_vis={s['mean_oneshot_vis']}  peak_vis={s['mean_peak_vis']}  "
                  f"mean_gain={s['mean_vis_gain']}  refine>1shot={s['cells_refine_beats_oneshot']}  "
                  f"turn0_parsefail={s['cells_parse_failed_turn0']}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
