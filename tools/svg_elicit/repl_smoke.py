#!/usr/bin/env python3
"""REAL-SERVER regression test for the incremental edit-in-REPL harness.

The serverless probe (harness_probe_repl.py) CANNOT catch the failure that bit us:
the real model writes draft-then-final programs (two `svg =` assignments) and anchors
edits on the dead draft, so edits apply to program text that never renders -> the
render is byte-identical every turn (the zero-delta bug). Reproducing that needs the
LIVE model, not canned edits. This runs a short real refine_incremental against the
engine and ASSERTS the render actually moves when edits apply.

It re-derives each turn's svg from the DUMPED Python (the persisted REPL state) so the
check is independent of the loop's own bookkeeping.

Run (engine must be up at GEMMA_BASE, default http://127.0.0.1:8001):
  GEMMA_BASE=http://127.0.0.1:8001 uv run --quiet --with numpy --with pillow \
    --with scikit-image --with playwright python tools/svg_elicit/repl_smoke.py
Exit 0 = pass.
"""
from __future__ import annotations
import argparse, hashlib, pathlib, sys
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import krepl_repl                       # noqa: E402
from edit_elicit import _run_program    # noqa: E402


def _svg_sha(py_path: pathlib.Path, tgt_arr) -> tuple[str, bool]:
    """Re-run the dumped Python REPL program -> svg -> short hash (independent check)."""
    src = py_path.read_text()
    svg, err = _run_program(src, tgt_arr)
    return hashlib.sha1((svg or "").encode()).hexdigest()[:10], bool(err)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=pathlib.Path,
                    default=HERE.parents[1] / "test_data/frames_v2/QTG2yY1znxM/frame_0010.png")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=pathlib.Path,
                    default=HERE.parents[1] / "output_data/svg_runs/repl_smoke")
    args = ap.parse_args()

    target = Image.open(args.frame).convert("RGB")
    tgt_arr = np.asarray(target)
    print(f"[repl_smoke] live run: {args.frame.name} rounds={args.rounds} seed={args.seed}", flush=True)
    rep = krepl_repl.refine_incremental(target, rounds=args.rounds, seed=args.seed,
                                        out_dir=args.out, prefix="smoke", max_tokens=8000)
    traj = rep["trajectory"]

    shas, errs = [], []
    for r in traj:
        sha, err = _svg_sha(args.out / f"smoke_t{r['turn']:02d}.py", tgt_arr)
        shas.append(sha); errs.append(err)

    applied = [r.get("n_applied") for r in traj]
    nerr = [r.get("n_errors") for r in traj]
    mses = [r.get("mse") for r in traj]
    runch = [r.get("render_unchanged") for r in traj]

    print(f"\n{'turn':>4} {'applied':>8} {'edit_err':>9} {'render_unchg':>13} "
          f"{'mse':>9} {'svg_sha':>11}")
    for i, r in enumerate(traj):
        print(f"{r['turn']:>4} {str(applied[i]):>8} {str(nerr[i]):>9} {str(runch[i]):>13} "
              f"{str(round(mses[i], 6) if isinstance(mses[i], float) else mses[i]):>9} {shas[i]:>11}")

    checks: list[bool] = []

    def chk(name: str, cond: bool, detail: str = ""):
        checks.append(bool(cond))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

    print()
    chk("turn 0 produced a valid render (mse not None)",
        isinstance(mses[0], float), f"mse0={mses[0]}")
    edit_applied = [a for a in applied[1:] if a and a > 0]
    chk("at least one edit turn applied >=1 edit against real model output",
        len(edit_applied) > 0, f"applied per turn={applied}")
    chk("rendered SVG is NOT frozen across turns (the zero-delta bug)",
        len(set(shas)) > 1, f"{len(set(shas))} distinct of {len(shas)}; sha={shas}")
    distinct_mse = len({round(m, 6) for m in mses if isinstance(m, float)})
    chk("rendered mse moves across turns",
        distinct_mse > 1, f"mse={mses}")
    moved = any(applied[i] and applied[i] > 0 and shas[i] != shas[i - 1]
                for i in range(1, len(shas)))
    chk("an APPLIED edit changed the rendered SVG (edits reach the picture)",
        moved, f"applied={applied} sha={shas}")

    ok = all(checks)
    print(f"\n{'REAL-SERVER SMOKE PASS' if ok else 'REAL-SERVER SMOKE FAIL'}: "
          f"{sum(checks)}/{len(checks)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
