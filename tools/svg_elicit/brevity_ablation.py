#!/usr/bin/env python3
"""Randomized brevity ablation — treat prompt design as a statistics problem.

For each (frame, replicate) we draw ONE variant per constraint from brevity_bank
(a 2x2 demonstrated x asked grid each), assemble the system prompt, and run a short
append-only edit rollout with a LARGE max_tokens (we never truncate — truncation
only yields parse errors; if the model rambles that is the prompt's failure). We
record exact per-turn output tokens (from usage) and, offline, the vision-encoder
faithfulness of the best render.

Then we regress output tokens on the brevity vector (length 2*constraints):
which (constraint, dimension) coordinates actually compress output — and do they
cost faithfulness? This replaces a 4**constraints grid search with N random draws
plus a linear model.

Usage:
  GEMMA_BASE=http://127.0.0.1:8001 GEMMA_DYLIB=.../libgemma_metal.dylib \\
    uv run --with numpy --with pillow --with scikit-image --with playwright \\
      python tools/svg_elicit/brevity_ablation.py --frames F1 F2 ... \\
      --reps 8 --rounds 3 --out-root output_data/svg_runs/brevity
"""
from __future__ import annotations
import argparse, json, pathlib, random, statistics as st, sys, time

import numpy as np

import edit_elicit as E                                              # sets sys.path
from edit_elicit import _extract_program, _apply_tools, _run_program, _numbered  # noqa: E402
from svg_refinement_loop import (                                    # noqa: E402
    render_svg, mse_images, diff_heatmap, image_to_data_url, load_target_from_path)
from elicit import call_lm                                           # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # tools/
import batch_scaler as bs                                            # noqa: E402
import brevity_bank as bank                                          # noqa: E402
import vision_metric as vm                                           # noqa: E402

TURN0_ASK = "Target image ({W}x{H}). Write your initial program now."


def run_one(target, choice, rounds, seed, out_dir, prefix, max_tokens):
    W, H = target.size
    tgt_arr = np.asarray(target.convert("RGB"))
    out_dir.mkdir(parents=True, exist_ok=True)
    messages = [
        {"role": "system", "content": bank.assemble_system(choice)},
        {"role": "user", "content": [
            {"type": "text", "text": TURN0_ASK.format(W=W, H=H)},
            {"type": "image_url", "image_url": {"url": image_to_data_url(target)}}]},
    ]
    source: list[str] = []
    per_turn = []
    for rnd in range(rounds + 1):
        text, finish, usage = call_lm(messages, max_tokens, 1.0, seed + rnd)
        messages.append({"role": "assistant", "content": text})
        ntok = usage.get("completion_tokens")
        if rnd == 0:
            prog = _extract_program(text)
            source = (prog or "svg = ''").split("\n")
        else:
            source, _, _, _ = _apply_tools(source, text, "full")
        (out_dir / f"{prefix}_r{rnd:02d}.py").write_text("\n".join(source))
        svg, err = _run_program("\n".join(source), tgt_arr)
        render, mse = None, None
        if svg and not err:
            try:
                render = render_svg(svg, W, H)
                mse = round(float(mse_images(target, render)), 5)
                render.save(out_dir / f"{prefix}_r{rnd:02d}_render.png")
            except Exception as e:
                err = repr(e)[:300]
        per_turn.append({"round": rnd, "out_tokens": ntok, "finish": finish,
                         "lines": len(source), "mse": mse, "err": (err[:160] if err else None)})
        if rnd < rounds:
            if render is not None:
                fb = {"round": rnd, "mse": mse, "lines": len(source)}
                content = [
                    {"type": "text", "text": "result: " + json.dumps(fb)
                     + "\n\nyour program:\n" + _numbered(source)[:4000]},
                    {"type": "text", "text": "current render:"},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(render)}},
                    {"type": "text", "text": "residual (bright = wrong/missing):"},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(diff_heatmap(target, render))}}]
            else:
                content = [{"type": "text", "text": f"program error: {err}. fix it; your program:\n"
                            + _numbered(source)[:4000]}]
            messages.append({"role": "user", "content": content})
    target.save(out_dir / f"{prefix}_target.png")
    return {"prefix": prefix, "choice": choice, "brevity_vector": bank.brevity_vector(choice),
            "per_turn": per_turn, "rounds": rounds,
            "total_tokens": sum((t["out_tokens"] or 0) for t in per_turn),
            "mean_turn_tokens": round(st.mean([(t["out_tokens"] or 0) for t in per_turn]), 1),
            "n_length_trunc": sum(1 for t in per_turn if t["finish"] == "length"),
            "n_render_ok": sum(1 for t in per_turn if not t["err"])}


def ols(X, y):
    """OLS with intercept; return coefficients (excluding intercept) for centered cols."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3, help="edit rounds after the initial program")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    targets = {f"{pathlib.Path(f).parent.name}_{pathlib.Path(f).stem}": load_target_from_path(f)
               for f in args.frames}
    jobs = [(stem, rep) for stem in targets for rep in range(args.reps)]

    def run(job):
        stem, rep = job
        seed = args.base_seed + (hash((stem, rep)) % 100000)
        choice = bank.choose(random.Random(seed))
        prefix = f"{stem}_rep{rep:02d}"
        try:
            return run_one(targets[stem], choice, args.rounds, seed, args.out_root, prefix, args.max_tokens)
        except Exception as e:
            import traceback
            return {"prefix": prefix, "error": repr(e), "tb": traceback.format_exc(limit=3),
                    "brevity_vector": bank.brevity_vector(choice), "total_tokens": None}

    print(f"[brevity] {len(jobs)} rollouts ({len(targets)} frames x {args.reps} reps), "
          f"{args.rounds+1} turns, max_tokens={args.max_tokens}", flush=True)
    t0 = time.time(); rows = []
    for r in bs.saturated_map(run, jobs, ordered=False):
        rows.append(r)
        if r.get("error"):
            print(f"  {r['prefix']:30} ERROR {r['error'][:80]}", flush=True)
        else:
            print(f"  {r['prefix']:30} total_tok={r['total_tokens']:>6} mean/turn={r['mean_turn_tokens']:>6} "
                  f"render_ok={r['n_render_ok']}/{args.rounds+1} trunc={r['n_length_trunc']} "
                  f"vec={r['brevity_vector']}", flush=True)
    wall = time.time() - t0

    # ── offline faithfulness: vision-encoder cosine of the best render per cell ──
    print("\n[brevity] scoring faithfulness (vision encoder)...", flush=True)
    vm.init()
    for r in rows:
        if r.get("error"):
            r["best_vis"] = None; continue
        tgt = args.out_root / f"{r['prefix']}_target.png"
        et = vm.embed(tgt)
        best = None
        for t in r["per_turn"]:
            rp = args.out_root / f"{r['prefix']}_r{t['round']:02d}_render.png"
            if rp.exists():
                try:
                    d = vm.cosine_distance(et, vm.embed(rp))
                    best = d if best is None else min(best, d)
                except Exception:
                    pass
        r["best_vis"] = round(best, 4) if best is not None else None

    # ── regression: output tokens (and faithfulness) on the brevity vector ──
    ok = [r for r in rows if not r.get("error") and r.get("total_tokens")]
    cols = bank.VECTOR_COLS
    X = np.array([r["brevity_vector"] for r in ok], dtype=float)
    Xc = X - X.mean(axis=0)                                  # center binary predictors
    y_tok = np.log(np.array([r["total_tokens"] for r in ok], dtype=float))
    y_turn = np.log(np.array([max(1.0, r["mean_turn_tokens"]) for r in ok], dtype=float))
    fid = [r["best_vis"] for r in ok]
    have_fid = [(r, f) for r, f in zip(ok, fid) if f is not None]

    beta_tok = ols(Xc, y_tok)
    beta_turn = ols(Xc, y_turn)
    # marginal correlation of each brevity bit with log total tokens
    corr = [float(np.corrcoef(X[:, j], y_tok)[0, 1]) if X[:, j].std() > 0 else 0.0
            for j in range(X.shape[1])]

    eff = sorted(zip(cols, beta_tok, beta_turn, corr), key=lambda z: z[1])
    summary = {
        "n": len(ok), "wall_s": round(wall, 1), "rounds": args.rounds,
        "columns": cols,
        "effects_on_log_total_tokens": {c: {"ols_beta": round(b, 3), "ols_beta_per_turn": round(bt, 3),
                                            "marginal_corr": round(cc, 3)}
                                        for c, b, bt, cc in zip(cols, beta_tok, beta_turn, corr)},
        "mean_total_tokens": round(float(np.mean([r["total_tokens"] for r in ok])), 0),
        "rows": rows,
    }
    if len(have_fid) >= len(cols) + 2:
        Xf = np.array([r["brevity_vector"] for r, _ in have_fid], dtype=float)
        yf = np.array([f for _, f in have_fid], dtype=float)   # vision cos-dist (lower=better)
        beta_fid = ols(Xf - Xf.mean(axis=0), yf)
        summary["effects_on_vision_cosdist_lower_is_better"] = {
            c: round(b, 4) for c, b in zip(cols, beta_fid)}

    (args.out_root / "brevity_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n=== brevity ablation — n={len(ok)} runs, wall {wall:.0f}s ===")
    print(f"mean total output tokens/run: {summary['mean_total_tokens']:.0f}")
    print(f"\neffect of each brevity bit on LOG total output tokens (negative = compresses):")
    print(f"  {'bit':8} {'OLS_beta':>9} {'per_turn':>9} {'marginal_r':>11}")
    for c, b, bt, cc in eff:
        flag = "  <== compresses" if b < -0.05 else ("  (backfires)" if b > 0.05 else "")
        print(f"  {c:8} {b:>9.3f} {bt:>9.3f} {cc:>11.3f}{flag}")
    if "effects_on_vision_cosdist_lower_is_better" in summary:
        print(f"\neffect on faithfulness (vision cos-dist; +beta = brevity HURT fidelity):")
        for c, b in sorted(summary["effects_on_vision_cosdist_lower_is_better"].items(), key=lambda z: -abs(z[1])):
            print(f"  {c:8} {b:>+.4f}")
    print(f"\n-> {args.out_root}/brevity_summary.json")


if __name__ == "__main__":
    main()
