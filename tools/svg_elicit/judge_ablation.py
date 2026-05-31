#!/usr/bin/env python3
"""Judge-design ablation — find a judge with DYNAMIC RANGE, not a crushed/flat scale.

Rescore a FINISHED krepl run (saved reference + per-turn renders) with several judge
designs; no generation/render. The hypothesis: ABSOLUTE judges (correspondence,
feature_residual, an un-anchored rubric) floor every SVG at the same low score
because the raster reference is unreachable, so their scale has no resolution for
refinement; COMPARATIVE (rank, pairwise) and ANCHORED/calibrated designs recover it.

Metrics per design (normalized to the design's range so 'flatness' is comparable):
  spread_norm        = (max-min)/range over a trajectory  (HIGH = not flat)
  discrim_norm       = (last-first)/range                 (signed; + if refinement reads as better)
  noise_norm         = stddev of re-scoring ONE render k times / range (LOW = consistent)
  snr                = mean spread_norm / mean noise_norm
E4q_s42 (t0->t4 is a visually-confirmed large refinement) is the positive control:
a good judge must give it a clearly rising trajectory.

Usage:
  GEMMA_BASE=... uv run --with numpy --with pillow python tools/svg_elicit/judge_ablation.py \\
    --run-dir output_data/svg_runs/krepl_v2b --out output_data/svg_runs/krepl_v2b/judge_ablation.json
"""
from __future__ import annotations
import argparse, glob, json, pathlib, statistics as st, sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import judge as J            # noqa: E402
from elicit import call_lm   # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import batch_scaler as bs    # noqa: E402

CONTROL = "E4qMxJJAszg_frame_0010_s42"


def _chat(seed, temp=0.0):
    return lambda msgs: call_lm(msgs, 512, temp, seed)[0]


def _mean(d, keys):
    vs = [d[k] for k in keys if isinstance(d.get(k), (int, float))]
    return st.mean(vs) if vs else None


# design name -> (range, fn(ref, renders, chat) -> list[float|None] aligned to renders)
def _abs9(ref, renders, chat):
    out = []
    for r in renders:
        jr = J.feature_residual(chat, ref, r)
        out.append(_mean(jr or {}, J.RESIDUAL_AXES.keys()) if jr else None)
    return out

def _corr3(ref, renders, chat):
    out = []
    for r in renders:
        jr = J.correspondence(chat, ref, r)
        out.append(_mean(jr or {}, J.CORR_AXES.keys()) if jr else None)
    return out

def _checklist(ref, renders, chat):
    cl = J.salient_checklist(chat, ref)
    if not cl:
        return [None] * len(renders)
    return [(J.checklist_match(chat, ref, r, cl) or {}).get("present_count") for r in renders]

def _rubric(ref, renders, chat):
    return [(J.score(chat, ref, r, []) or {}).get("faithfulness") for r in renders]

def _rank(ref, renders, chat):
    return J.rank_renders(chat, ref, renders) or [None] * len(renders)

def _pairwise(ref, renders, chat):
    out = [0.0]
    for i in range(1, len(renders)):
        v = J.pairwise_improve(chat, ref, renders[i - 1], renders[i])
        out.append(out[-1] + (v if v is not None else 0))
    return out

_ANCHORS = {}
def _anchored(ref, renders, chat):
    good, bad = _ANCHORS["good"], _ANCHORS["bad"]
    return [J.anchored_score(chat, ref, r, good, bad) for r in renders]

DESIGNS = {
    "abs9":      (4.0, _abs9),       # absolute 9-axis (1-5) — expect flat
    "corr3":     (4.0, _corr3),      # absolute 3-axis (1-5) — expect flat
    "checklist": (20.0, _checklist), # absolute coverage 0-20 — informative-but-noisy
    "rubric":    (4.0, _rubric),     # un-anchored rubric (1-5) — the 'uncalibrated old judge'
    "anchored":  (9.0, _anchored),   # exemplar-anchored 1-10 — calibrated absolute
    "rank":      (None, _rank),      # comparative ranking 1..N — forces spread
    "pairwise":  (None, _pairwise),  # comparative directional cumulative
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--noise-reps", type=int, default=3)
    args = ap.parse_args()

    cells = []
    for ref in sorted(glob.glob(str(args.run_dir / "*_reference.png"))):
        stem = pathlib.Path(ref).name[:-len("_reference.png")]
        rends = sorted(glob.glob(str(args.run_dir / f"{stem}_t*_render.png")))
        if len(rends) >= 2:
            cells.append((stem, ref, rends))
    print(f"[judge_ablation] {len(cells)} cells, {len(DESIGNS)} designs", flush=True)

    # fixed anchors for the calibrated design: a strong render + a flat/gray "barely-started"
    good_path = args.run_dir / f"{CONTROL}_t04_render.png"
    _ANCHORS["good"] = Image.open(good_path).convert("RGB") if good_path.exists() else None
    _ANCHORS["bad"] = Image.new("RGB", _ANCHORS["good"].size if _ANCHORS["good"] else (320, 180), (128, 128, 128))

    def score_cell(item):
        stem, refp, rends = item
        ref = Image.open(refp).convert("RGB")
        rimgs = [Image.open(p).convert("RGB") for p in rends]
        seed = 100 + (hash(stem) % 1000)
        chat = _chat(seed)
        res = {"cell": stem, "n": len(rimgs), "designs": {}}
        for name, (_rng, fn) in DESIGNS.items():
            try:
                traj = fn(ref, rimgs, chat)
            except Exception as e:
                traj = None
                res.setdefault("errors", {})[name] = repr(e)[:120]
            res["designs"][name] = traj
        # noise: re-score the MIDDLE render k times per design (per-render designs only)
        if len(rimgs) >= 3:
            mid = rimgs[len(rimgs) // 2]
            res["noise"] = {}
            for name in ("abs9", "checklist", "rubric", "anchored"):
                rng, fn = DESIGNS[name]
                vals = []
                for k in range(args.noise_reps):
                    v = fn(ref, [mid], _chat(seed + 1000 + k, temp=0.7))[0]
                    if isinstance(v, (int, float)):
                        vals.append(v)
                res["noise"][name] = round(st.pstdev(vals), 4) if len(vals) >= 2 else None
        return res

    results = list(bs.saturated_map(score_cell, cells, ordered=False))

    # aggregate: normalized spread / discrimination / noise per design
    def rng_for(name, traj):
        r = DESIGNS[name][0]
        if r is not None:
            return r
        if name == "rank":
            return max(1, (len([x for x in traj if x is not None]) - 1))
        if name == "pairwise":
            return max(1, 2 * (len(traj) - 1))
        return 1.0

    agg = {}
    for name in DESIGNS:
        spreads, discrims, noises = [], [], []
        for r in results:
            traj = [x for x in (r["designs"].get(name) or []) if isinstance(x, (int, float))]
            if len(traj) >= 2:
                rg = rng_for(name, r["designs"][name])
                spreads.append((max(traj) - min(traj)) / rg)
                discrims.append((traj[-1] - traj[0]) / rg)
            nz = (r.get("noise") or {}).get(name)
            if isinstance(nz, (int, float)):
                noises.append(nz / (DESIGNS[name][0] or 1.0))
        agg[name] = {
            "mean_spread_norm": round(st.mean(spreads), 3) if spreads else None,
            "mean_discrim_norm": round(st.mean(discrims), 3) if discrims else None,
            "mean_noise_norm": round(st.mean(noises), 3) if noises else None,
            "snr": (round(st.mean(spreads) / st.mean(noises), 2)
                    if (spreads and noises and st.mean(noises) > 0) else None),
            "control_E4q_s42_traj": next((r["designs"].get(name) for r in results if r["cell"] == CONTROL), None),
        }

    args.out.write_text(json.dumps({"per_cell": results, "per_design": agg}, indent=2, default=str))

    order = sorted(agg, key=lambda n: -(agg[n]["mean_spread_norm"] or 0))
    print(f"\n=== judge-design dynamic range (rescored {len(results)} trajectories) ===")
    print(f"  {'design':10} {'spread':>7} {'discrim':>8} {'noise':>7} {'snr':>6}  control E4q_s42 traj (t0->t4)")
    for name in order:
        a = agg[name]
        ctrl = a["control_E4q_s42_traj"]
        ctrl_s = " -> ".join(str(round(x, 2)) if isinstance(x, (int, float)) else "-" for x in (ctrl or []))
        print(f"  {name:10} {str(a['mean_spread_norm']):>7} {str(a['mean_discrim_norm']):>8} "
              f"{str(a['mean_noise_norm']):>7} {str(a['snr']):>6}  {ctrl_s}")
    print(f"\nHIGH spread_norm = NOT flat; + discrim = refinement reads as better; LOW noise = consistent.")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
