#!/usr/bin/env python3
"""Tool-design / description / error-recovery ablation for edit_elicit's `tools`
mode — does any design lift the ATOMIC EDIT SUCCESS RATE (and semdist)?

Runs cells concurrently (thread pool -> batched decode). Cells isolate the
toolset design knobs against a rewrite reference. Aggregates edit success rate,
best MSE, semantic_distance, and whether the program APPENDED (lines grew).
Outputs under output_data/.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, pathlib, statistics as st, sys, time

import edit_elicit as E                                       # sets sys.path
from svg_refinement_loop import load_target_from_path         # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # tools/ for batch_scaler
import batch_scaler as bs                                     # noqa: E402  saturate kernel width

# (label, edit_mode, design, tool_desc, recovery)
CELLS = [
    ("rewrite.ref",        "rewrite", "full",    "terse",   False),
    ("full.terse",         "tools",   "full",    "terse",   False),
    ("full.verbose",       "tools",   "full",    "verbose", False),
    ("full.verbose.rec",   "tools",   "full",    "verbose", True),
    ("addonly.verbose",    "tools",   "addonly", "verbose", False),
    ("addonly.verbose.rec","tools",   "addonly", "verbose", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--fill", type=int, default=1,
                    help="overschedule multiplier on the engine's kernel batch width "
                         "(1=saturate; 2=keep kernel full despite stragglers). Concurrency "
                         "comes from the engine via batch_scaler, never guessed.")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    targets = {pathlib.Path(f).stem: load_target_from_path(f) for f in args.frames}
    jobs = [(lbl, em, dz, td, rc, stem, seed)
            for stem in targets for seed in args.seeds
            for (lbl, em, dz, td, rc) in CELLS]

    def run(job):
        lbl, em, dz, td, rc, stem, seed = job
        prefix = f"{stem}_{lbl}_s{seed}"
        try:
            rep = E.run_rollout(targets[stem], em, args.rounds, args.max_tokens, 1.0, seed,
                                args.out_root, prefix, tool_desc=td, design=dz, recovery=rc)
        except Exception as e:
            rep = {"best_mse": None, "error": repr(e)}
        tj = rep.get("trajectory") or []
        na, ne = rep.get("n_edits_applied", 0), rep.get("n_edit_errors", 0)
        rep["edit_success"] = round(na / (na + ne), 2) if (na + ne) else None
        rep["appended"] = (tj[-1]["lines"] - tj[0]["lines"]) if len(tj) >= 2 else 0
        rep.update({"label": lbl, "frame": stem, "seed": seed})
        return rep

    print(f"[edit_sweep] {len(jobs)} jobs, saturating engine kernel width (fill={args.fill})")
    t0 = time.time(); results = []
    for r in bs.saturated_map(run, jobs, fill=args.fill, ordered=False):
            results.append(r)
            print(f"  {r['frame']:>10}/{r['label']:<20} s{r['seed']} "
                  f"best_mse={r.get('best_mse')} semdist={r.get('semantic_distance')} "
                  f"smatch={r.get('subject_match')} edit_succ={r.get('edit_success')} "
                  f"appended={r.get('appended')} {r.get('error','')}", flush=True)
    wall = time.time() - t0

    def m(rs, k):
        v = [x[k] for x in rs if x.get(k) is not None]
        return round(st.mean(v), 4) if v else None

    agg = {}
    for (lbl, *_r) in CELLS:
        rs = [x for x in results if x["label"] == lbl and x.get("best_mse") is not None]
        agg[lbl] = ({"n": len(rs), "best_mse": m(rs, "best_mse"), "semdist": m(rs, "semantic_distance"),
                     "smatch": m(rs, "subject_match"), "edit_success": m(rs, "edit_success"),
                     "appended": m(rs, "appended")} if rs else {"n": 0})

    (args.out_root / "edit_sweep_summary.json").write_text(json.dumps(
        {"cells": [list(c) for c in CELLS], "wall_s": round(wall, 1),
         "per_cell": agg, "results": results}, indent=2, default=str))
    print(f"\n=== per-cell means — wall {wall:.0f}s ===")
    for lbl, a in agg.items():
        if a.get("n"):
            print(f"  {lbl:<22} n={a['n']} edit_succ={a['edit_success']} appended={a['appended']} "
                  f"best_mse={a['best_mse']} semdist={a['semdist']} smatch={a['smatch']}")
        else:
            print(f"  {lbl:<22} (no valid)")
    print(f"-> {args.out_root}/edit_sweep_summary.json")


if __name__ == "__main__":
    main()
