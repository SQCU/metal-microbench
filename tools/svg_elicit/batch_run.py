#!/usr/bin/env python3
"""Batched video-frame SVG throughput regime.

The oldest video-trial workload (svg_concurrent_bench curriculum) re-run through
the upgraded elicit harness: take substantial-length videos, sample frames with a
strategy that AVOIDS over-polling near-duplicate frames (perceptual aHash +
farthest-point sampling), then run N batches of K frames CONCURRENTLY (exercising
the engine's B=K batched decode) and report runtime statistics + collect the SVG
outputs.

Usage (videos):
  GEMMA_BASE=http://127.0.0.1:8001 \\
    uv run --with numpy --with pillow --with playwright --with scikit-image \\
      python tools/svg_elicit/batch_run.py \\
      --videos test_data/video_rips/KCrfDHS_YUw.mp4 test_data/video_rips/Vore-4VZ5rs.mp4 \\
      --extract-fps 1 --n-frames 64 --batch 8 --max-iters 1
Usage (pre-extracted frame dirs as the pool):
  ... --frames-dirs test_data/frames_v2/KCrfDHS_YUw test_data/frames_v2/Vore-4VZ5rs ...
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, time, urllib.request
import concurrent.futures as cf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "svg_elicit"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "scripts" / "archival"))
import batch_scaler as bs  # noqa: E402  saturate engine kernel width, never guess it
import elicit  # noqa: E402  (one_rollout, JUDGE_EXEMPLARS)
import judge as _judge  # noqa: E402
from svg_refinement_loop import load_target_from_path  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

BASE = os.environ.get("GEMMA_BASE", "http://127.0.0.1:8001")


def ahash(img: Image.Image, n: int = 8) -> np.ndarray:
    """8x8 average-hash bit vector — cheap perceptual fingerprint; Hamming
    distance between two ≈ perceptual difference (near-dupes ≈ 0)."""
    g = np.asarray(img.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float32)
    return (g > g.mean()).flatten()


def extract_frames(video: pathlib.Path, out_dir: pathlib.Path, fps: float) -> list[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.png"))
    if existing:
        return existing
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}",
                    str(out_dir / "frame_%05d.png")], check=True)
    return sorted(out_dir.glob("frame_*.png"))


def _frame_std(f: pathlib.Path) -> float:
    return float(np.asarray(Image.open(f).convert("L"), dtype=np.float32).std())


def diverse_sample(frames: list[pathlib.Path], n: int, *, seed: int = 0,
                   min_marginal_hamming: int = 8) -> tuple[list[pathlib.Path], dict]:
    """Online marginal-variance sampler: variance measured RELATIVE TO THE POOL
    SAMPLED SO FAR, not per-frame absolute. Probe frames in random order (random
    intervals through the video) and ACCEPT a frame iff its min aHash-Hamming to
    everything already selected is >= min_marginal_hamming — i.e. it adds spread
    the current set lacks.

    This is the right behavior the earlier farthest-point + hard-std-filter
    versions got wrong:
      - A black/blank frame IS a distinct mode, so the FIRST one encountered is
        accepted (high marginal variance vs a content-only set) — but a SECOND
        black frame has ~0 Hamming to the first → rejected. Result: exactly ONE
        boring full-black frame ever, not zero (hard-filter) and not many (plain
        farthest-point, which prioritizes outliers).
      - Near-duplicate consecutive frames are rejected (low marginal variance).
    Random order avoids seeding on the video's (often black) first frame and
    makes the candidate search a random-interval probe rather than an all-pairs
    scan."""
    rng = np.random.RandomState(seed)
    order = list(range(len(frames)))
    rng.shuffle(order)
    sel_idx, sel_hashes = [], []   # sel_hashes holds CONTENT hashes only
    have_blank, n_probed, min_std = False, 0, 12.0
    for i in order:
        n_probed += 1
        g = np.asarray(Image.open(frames[i]).convert("L"), dtype=np.float32)
        if g.std() < min_std:
            # Degenerate (black/blank) frames collapse to ONE canonical mode:
            # aHash is just noise on a near-uniform image, so we can't trust it
            # to dedup them. Accept the first blank ever; reject all others.
            if have_blank:
                continue
            have_blank = True
            sel_idx.append(i)
        else:
            h = ahash(Image.fromarray(g.astype("uint8")))
            if sel_hashes and min(int(np.count_nonzero(h != sh)) for sh in sel_hashes) < min_marginal_hamming:
                continue  # too similar to the content pool so far
            sel_idx.append(i)
            sel_hashes.append(h)
        if len(sel_idx) >= n:
            break
    # diagnostics
    nn = [min(int(np.count_nonzero(a != b)) for b in sel_hashes if b is not a)
          for a in sel_hashes] if len(sel_hashes) > 1 else [0]
    n_black = sum(1 for i in sel_idx if _frame_std(frames[i]) < min_std)
    return ([frames[i] for i in sel_idx],
            {"pool": len(frames), "probed": n_probed, "selected": len(sel_idx),
             "black_frames_in_selection": n_black,
             "min_marginal_hamming": min_marginal_hamming,
             "min_nn_hamming": min(nn), "mean_nn_hamming": round(float(np.mean(nn)), 2),
             "note": "online: accept iff min-Hamming-to-pool-so-far >= threshold; "
                     "≤1 black frame by construction; random-interval probe order"})


def health() -> dict:
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=4) as r:
            d = json.load(r)
        return {k: d.get(k) for k in ("vision_cache_hits", "cached_pages", "free_pages",
                                      "active_stream_count", "aggregate_tok_per_sec")}
    except Exception:
        return {}


def run_batch(frames, batch_idx, args, out_root) -> list[dict]:
    """Run one batch of frames concurrently (one OS thread each → engine B=K
    batched decode). Returns per-frame result dicts."""
    results = [None] * len(frames)
    # Concurrency = the engine's kernel width (clamped to this batch's frame
    # count), from batch_scaler — not len(frames), which would over/under-fill.
    with bs.SaturatingPool(n_items=len(frames)) as pool:
        futs = {}
        for j, fp in enumerate(frames):
            tgt = load_target_from_path(str(fp))
            w, h = tgt.size
            prefix = f"b{batch_idx:02d}f{j:02d}_{fp.parent.name}_{fp.stem}"
            fut = pool.submit(elicit.one_rollout, tgt, args.mode, w, h, args.max_iters,
                              args.max_tokens, args.temperature,
                              args.base_seed + batch_idx * 100 + j, args.primary_metric,
                              args.use_judge, out_root, prefix)
            futs[fut] = (j, str(fp), prefix)
        for fut in cf.as_completed(futs):
            j, fpath, prefix = futs[fut]
            try:
                rep = fut.result()
                valid = [hh for hh in rep["history"] if hh.get("valid")]
                ctoks = sum((hh.get("completion_tokens") or 0) for hh in rep["history"])
                fwall = sum((hh.get("elapsed_s") or 0) for hh in rep["history"])
                best = rep.get("best_primary_value")
                ssim_last = valid[-1]["ssim"] if valid else None
                results[j] = {"frame": fpath, "prefix": prefix, "ok": True,
                              "completion_tokens": ctoks, "frame_compute_s": round(fwall, 1),
                              "n_valid": rep["n_valid"], "n_invalid": rep["n_invalid"],
                              "best_primary": best, "ssim_last": ssim_last}
            except Exception as e:
                results[j] = {"frame": fpath, "prefix": prefix, "ok": False, "error": str(e)}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=pathlib.Path, nargs="*", default=None)
    ap.add_argument("--frames-dirs", type=pathlib.Path, nargs="*", default=None)
    ap.add_argument("--extract-fps", type=float, default=1.0)
    ap.add_argument("--n-frames", type=int, default=64)
    ap.add_argument("--batch", type=int, default=None,
                    help="frames per concurrent batch; default = the engine's kernel width "
                         "(batch_scaler.kernel_batch) so each batch saturates one decode step")
    ap.add_argument("--mode", choices=["svg", "python"], default="svg")
    ap.add_argument("--max-iters", type=int, default=1, help="1 = single-shot per frame (throughput regime)")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--primary-metric", choices=["ssim", "mse", "judge"], default="ssim")
    ap.add_argument("--judge", action="store_true", help="enable the calibrated judge (off by default for clean throughput)")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--out-root", type=pathlib.Path,
                    default=REPO / "output_data" / "svg_runs" / f"batch_{int(time.time())}")
    args = ap.parse_args()
    if args.batch is None:
        args.batch = bs.kernel_batch()  # saturate one decode step by default
    args.use_judge = args.judge
    if args.use_judge:
        elicit.JUDGE_EXEMPLARS = _judge.load_exemplars(
            REPO, REPO / "tools" / "svg_elicit" / "amongus_onpolicy_exemplars.json")
    args.out_root.mkdir(parents=True, exist_ok=True)

    # 1) Build the frame pool from videos (extract) or pre-extracted dirs.
    pool_by_src = []
    if args.videos:
        for v in args.videos:
            fdir = REPO / "test_data" / "extracted_frames" / v.stem
            fr = extract_frames(v, fdir, args.extract_fps)
            pool_by_src.append((v.stem, fr))
    elif args.frames_dirs:
        for d in args.frames_dirs:
            pool_by_src.append((d.name, sorted(d.glob("frame_*.png"))))
    else:
        raise SystemExit("need --videos or --frames-dirs")
    pool = [f for _, fr in pool_by_src for f in fr]
    print(f"[batch] pool: {len(pool)} frames from {len(pool_by_src)} sources "
          f"({', '.join(f'{n}:{len(fr)}' for n, fr in pool_by_src)})")

    # 2) Diverse-sample n_frames avoiding near-duplicates.
    selected, samp_stats = diverse_sample(pool, args.n_frames)
    print(f"[batch] diverse-sampled {len(selected)} frames: {samp_stats}")

    # 3) Run in batches of K concurrently; time each batch.
    batches = [selected[i:i + args.batch] for i in range(0, len(selected), args.batch)]
    h0 = health()
    job_t0 = time.time()
    per_batch, all_frames = [], []
    for bi, fb in enumerate(batches):
        hb0 = health()
        t0 = time.time()
        res = run_batch(fb, bi, args, args.out_root)
        wall = time.time() - t0
        hb1 = health()
        ok = [r for r in res if r and r.get("ok")]
        ctoks = sum(r["completion_tokens"] for r in ok)
        serial_equiv = sum(r["frame_compute_s"] for r in ok)  # sum of per-frame compute
        bstat = {"batch": bi, "n": len(fb), "n_ok": len(ok),
                 "wall_s": round(wall, 1),
                 "batch_completion_tokens": ctoks,
                 "agg_tok_per_s": round(ctoks / wall, 1) if wall else None,
                 "serial_equiv_s": round(serial_equiv, 1),
                 "batch_speedup_x": round(serial_equiv / wall, 2) if wall else None,
                 "vision_cache_hits_delta": (hb1.get("vision_cache_hits") or 0) - (hb0.get("vision_cache_hits") or 0)}
        per_batch.append(bstat)
        all_frames.extend(res)
        print(f"  batch {bi}: wall={bstat['wall_s']}s ok={len(ok)}/{len(fb)} "
              f"ctoks={ctoks} agg_tok/s={bstat['agg_tok_per_s']} "
              f"speedup={bstat['batch_speedup_x']}x cache_hits+={bstat['vision_cache_hits_delta']}")
    job_wall = time.time() - job_t0

    ok_all = [r for r in all_frames if r and r.get("ok")]
    tot_ctoks = sum(r["completion_tokens"] for r in ok_all)
    report = {
        "config": {"videos": [str(v) for v in (args.videos or [])],
                   "frames_dirs": [str(d) for d in (args.frames_dirs or [])],
                   "n_frames": len(selected), "batch": args.batch, "mode": args.mode,
                   "max_iters": args.max_iters, "judge": args.use_judge},
        "sampling": samp_stats,
        "totals": {
            "frames": len(selected), "frames_ok": len(ok_all),
            "frames_failed": len(all_frames) - len(ok_all),
            "job_wall_s": round(job_wall, 1),
            "frames_per_min": round(len(ok_all) / (job_wall / 60), 1) if job_wall else None,
            "total_completion_tokens": tot_ctoks,
            "aggregate_tok_per_s": round(tot_ctoks / job_wall, 1) if job_wall else None,
            "mean_batch_speedup_x": round(float(np.mean([b["batch_speedup_x"] for b in per_batch if b["batch_speedup_x"]])), 2),
            "health_start": h0, "health_end": health(),
        },
        "per_batch": per_batch,
        "per_frame": all_frames,
    }
    (args.out_root / "batch_report.json").write_text(json.dumps(report, indent=2, default=str))
    t = report["totals"]
    print(f"\n[batch] {t['frames_ok']}/{t['frames']} frames in {t['job_wall_s']}s "
          f"({t['frames_per_min']} frames/min) | agg {t['aggregate_tok_per_s']} tok/s "
          f"| mean batch speedup {t['mean_batch_speedup_x']}x")
    print(f"[batch] → {args.out_root}/batch_report.json")


if __name__ == "__main__":
    main()
