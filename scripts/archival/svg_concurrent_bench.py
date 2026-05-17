#!/usr/bin/env python3
"""Concurrent throughput benchmark + distribution-sampling curriculum
for the svg_refinement_loop workload.

Treats each refinement-loop instance as an independent API client
running in its own thread. Fires K clients in parallel via a thread
pool, logs per-iteration timings + MSE, samples engine-side metrics
in the background, outputs a structured JSON + a short ASCII summary.

Two intended use modes:

  1. `--profile`: short sweep over K ∈ {1, 2, 4, 8} × R runs each to
     establish throughput curve. Scopes the next run.
  2. `--curriculum <image_dir>`: long run — picks images from a
     directory (e.g., extracted video frames), assigns them to K
     parallel users with controlled variation (mode, seed, prompt
     flavor). Runs until max_wall_time or max_total_iterations.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import random
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svg_refinement_loop import (
    run_loop, load_target_from_path, generate_donut_target,
    synth_smoke_target,
)
from PIL import Image


BASE = "http://127.0.0.1:8000"
REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / "notes" / "runs"


# ── Engine-side metric sampler ───────────────────────────────────────

class EngineSampler:
    """Background thread that polls /api/extra/scheduler and /api/extra/perf
    at a specified frequency, accumulating per-tick records for
    post-hoc analysis."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(BASE + path, timeout=5) as r:
                return json.load(r)
        except Exception:
            return {}

    def _run(self):
        while not self._stop.is_set():
            t = time.time()
            sched = self._get("/api/extra/scheduler")
            perf = self._get("/api/extra/perf")
            if sched and perf:
                self.samples.append({
                    "ts": t,
                    "resident":   sched.get("counts", {}).get("resident", 0),
                    "slotted":    sched.get("counts", {}).get("slotted", 0),
                    "wanting":    sched.get("counts", {}).get("wanting_slot_not_assigned", 0),
                    "paused":     sched.get("counts", {}).get("paused", 0),
                    "avg_active": sched.get("activity", {}).get("avg_active_slots_since_start", 0),
                    "last_step_ms": sched.get("activity", {}).get("last_step_ms", 0),
                    "tps":        perf.get("tokens_per_second", 0),
                    "total_tokens": perf.get("total_completion_tokens", 0),
                    "idle":       perf.get("idle", 0),
                })
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread: self._thread.join(timeout=5)
        return self.samples


# ── One worker = one refinement-loop client ──────────────────────────

def one_client_run(client_id: int, target_img: Image.Image, target_id: str,
                     mode: str, size: int, max_iters: int,
                     out_base: pathlib.Path, seed_base: int) -> dict:
    """Run a single refinement loop and record its per-iter metrics."""
    t0 = time.time()
    out_dir = out_base / f"client_{client_id:04d}"
    try:
        report = run_loop(
            target_img=target_img, mode=mode, size=size,
            max_iterations=max_iters, mse_target=0.003,
            plateau_patience=2, include_heatmap=True,
            out_dir=out_dir, seed_base=seed_base)
    except Exception as e:
        return {
            "client_id": client_id, "target_id": target_id, "mode": mode,
            "error": str(e), "total_wall_s": time.time() - t0,
        }
    total_wall = time.time() - t0
    return {
        "client_id": client_id,
        "target_id": target_id,
        "mode": mode,
        "seed_base": seed_base,
        "best_mse": report["best_mse"],
        "best_iter": report["best_iter"],
        "n_iterations_completed": len(report["history"]),
        "per_iter": report["history"],
        "total_wall_s": total_wall,
    }


# ── Target-set loaders ───────────────────────────────────────────────

def load_curriculum_targets(image_dir: pathlib.Path, size: int,
                               limit: int | None = None) -> list[tuple[str, Image.Image]]:
    """Load all images in `image_dir`, resize to (size, size), return
    (stable_id, PIL.Image) pairs. stable_id is the relative file path
    without extension."""
    out = []
    for p in sorted(image_dir.glob("**/*")):
        if not p.is_file(): continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            continue
        try:
            img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        except Exception:
            continue
        target_id = p.relative_to(image_dir).with_suffix("").as_posix()
        out.append((target_id, img))
        if limit and len(out) >= limit: break
    return out


def mixed_synthetic_targets(size: int) -> list[tuple[str, Image.Image]]:
    """Built-in fallback set: mix of synth shapes + some donuts. Used
    for --profile and as default if no curriculum dir supplied."""
    targets = [("smoke", synth_smoke_target(size))]
    for seed in (7, 42, 101, 137):
        try:
            targets.append((f"donut-{seed}", generate_donut_target(seed, size)))
        except Exception as e:
            print(f"  [warn] donut-{seed} failed: {e}")
            break
    return targets


# ── Modes ────────────────────────────────────────────────────────────

def pick_mode(mode_spec: str, rng: random.Random) -> str:
    if mode_spec == "mix":
        return rng.choice(["svg", "python"])
    return mode_spec


# ── Benchmark runners ────────────────────────────────────────────────

def run_wave(
    clients_n: int, target_iter, max_iters: int, mode_spec: str,
    size: int, out_base: pathlib.Path, run_id: str, rng_seed: int,
) -> list[dict]:
    """Launch up to `clients_n` clients via a thread pool, each pulling
    one target from `target_iter`. Wait for all to complete. Returns
    per-client result list."""
    rng = random.Random(rng_seed)
    jobs = []
    for cid in range(clients_n):
        try:
            target_id, target_img = next(target_iter)
        except StopIteration:
            break
        mode = pick_mode(mode_spec, rng)
        seed_base = rng.randint(1, 10_000_000)
        jobs.append((cid, target_id, target_img, mode, seed_base))

    results = []
    with cf.ThreadPoolExecutor(max_workers=clients_n) as exe:
        futs = [
            exe.submit(one_client_run, cid, target_img, target_id, mode, size,
                        max_iters, out_base / run_id, seed_base)
            for (cid, target_id, target_img, mode, seed_base) in jobs
        ]
        for fut in cf.as_completed(futs):
            try: results.append(fut.result())
            except Exception as e:
                results.append({"error": f"future-level: {e}"})
    return results


def profile_sweep(
    K_values: list[int], R: int, size: int, max_iters: int,
    mode: str, out_base: pathlib.Path,
) -> dict:
    """Short sweep over K ∈ K_values × R wave-runs per K. Each wave
    gets R different (target, seed) pairs so we don't thrash cache."""
    all_results = []
    summary_by_K = {}

    for K in K_values:
        print(f"\n=== profile K={K} (R={R} waves of K clients each) ===")
        sampler = EngineSampler(interval_s=0.5); sampler.start()
        t_wave0 = time.time()

        wave_results = []
        for r_idx in range(R):
            wave_seed = 10_000 + K * 1000 + r_idx
            # Refresh targets each wave; mixed_synthetic gives us 5 items.
            targets = mixed_synthetic_targets(size)
            random.Random(wave_seed).shuffle(targets)
            target_iter = iter(targets)
            wave_id = f"K{K:02d}_R{r_idx:02d}"
            t0 = time.time()
            wave_r = run_wave(K, target_iter, max_iters, mode, size,
                                out_base, wave_id, wave_seed)
            wave_wall = time.time() - t0
            wave_iters = sum(r.get("n_iterations_completed", 0) for r in wave_r)
            bw = wave_bandwidth_report(wave_r, wave_wall)
            print(f"  wave {wave_id}: {len(wave_r)} clients, "
                  f"{wave_iters} iterations, {wave_wall:.1f}s wall, "
                  f"{wave_iters/max(wave_wall,1e-6):.3f} iter/s agg")
            print_bandwidth(bw, prefix="  ")
            wave_results.append({
                "wave_id": wave_id, "K": K, "wave_wall_s": wave_wall,
                "n_clients": len(wave_r), "n_iterations": wave_iters,
                "iters_per_sec": wave_iters / max(wave_wall, 1e-6),
                "bandwidth": bw,
                "clients": wave_r,
            })
            all_results.append(wave_results[-1])

        samples = sampler.stop()
        summary_by_K[K] = {
            "waves": wave_results,
            "total_wall_s": time.time() - t_wave0,
            "engine_samples": samples,
        }
        # Aggregate stats for K.
        total_iters = sum(w["n_iterations"] for w in wave_results)
        total_wall = sum(w["wave_wall_s"] for w in wave_results)
        avg_iterps = total_iters / max(total_wall, 1e-6)
        if samples:
            avg_slotted = sum(s["slotted"] for s in samples) / len(samples)
            max_slotted = max(s["slotted"] for s in samples)
            avg_active = (sum(s["avg_active"] for s in samples[-10:]) / 10
                          if len(samples) >= 10 else 0)
        else:
            avg_slotted = max_slotted = avg_active = 0
        print(f"  K={K} summary: {total_iters} iters / {total_wall:.1f}s "
              f"= {avg_iterps:.3f} iter/s agg (avg_slotted={avg_slotted:.2f}, "
              f"max_slotted={max_slotted}, end-of-sample avg_active={avg_active:.2f})")

    return {"by_K": summary_by_K, "all_waves": all_results}


def run_curriculum(
    image_dir: pathlib.Path, K: int, max_iters: int, mode_spec: str,
    size: int, out_base: pathlib.Path, max_wall_s: float,
    max_iters_total: int | None = None,
) -> dict:
    """Long-running curriculum: fire K clients in a rolling pool,
    sample from image_dir with controlled variation. Runs until
    max_wall_s or max_iters_total reached."""
    targets_all = load_curriculum_targets(image_dir, size)
    if not targets_all:
        raise RuntimeError(f"no images found in {image_dir}")
    print(f"[curriculum] loaded {len(targets_all)} targets from {image_dir}")

    sampler = EngineSampler(interval_s=1.0); sampler.start()
    t_start = time.time()
    all_results = []
    total_iters = 0
    trial_idx = 0
    rng = random.Random(42)

    try:
        while time.time() - t_start < max_wall_s:
            # Build next wave: K target-pair choices.
            wave_targets = rng.sample(targets_all, min(K, len(targets_all)))
            target_iter = iter(wave_targets)
            wave_id = f"trial_{trial_idx:04d}"
            t0 = time.time()
            wave_r = run_wave(K, target_iter, max_iters, mode_spec, size,
                                out_base, wave_id, rng.randint(1, 1_000_000_000))
            wave_wall = time.time() - t0
            wave_iters = sum(r.get("n_iterations_completed", 0) for r in wave_r)
            bw = wave_bandwidth_report(wave_r, wave_wall)
            total_iters += wave_iters
            elapsed = time.time() - t_start
            print(f"[curriculum] trial {trial_idx:04d} done "
                  f"({len(wave_r)} clients, {wave_iters} iters in {wave_wall:.1f}s) | "
                  f"total {total_iters} iters in {elapsed:.0f}s "
                  f"({total_iters/max(elapsed,1e-6):.2f} iter/s)")
            print_bandwidth(bw, prefix="[curriculum] ")
            all_results.append({
                "trial": trial_idx, "wave_wall_s": wave_wall,
                "wave_iters": wave_iters, "wall_elapsed_s": elapsed,
                "bandwidth": bw,
                "clients": wave_r,
            })
            trial_idx += 1
            if max_iters_total and total_iters >= max_iters_total:
                print(f"[curriculum] reached max_iters_total={max_iters_total}")
                break
    finally:
        samples = sampler.stop()

    return {
        "image_dir": str(image_dir),
        "K": K, "mode_spec": mode_spec, "size": size,
        "max_iters_per_client": max_iters,
        "total_trials": trial_idx,
        "total_iterations": total_iters,
        "total_wall_s": time.time() - t_start,
        "trials": all_results,
        "engine_samples": samples,
    }


# ── Per-request metric aggregation ───────────────────────────────────

def _collect_requests(wave_results: list[dict]) -> list[dict]:
    """Pull per-iter metric dicts out of each client's history."""
    out = []
    for c in wave_results:
        for h in c.get("per_iter", []):
            if not isinstance(h, dict): continue
            if "ttft_s" not in h: continue
            out.append({
                **h,
                "client_id": c.get("client_id"),
                "target_id": c.get("target_id"),
            })
    return out


def _class_agg(recs: list[dict], label: str, wave_wall_s: float) -> dict:
    if not recs:
        return {"class": label, "n": 0}
    pt = [r.get("prompt_tokens") for r in recs if r.get("prompt_tokens")]
    ct = [r.get("completion_tokens") for r in recs if r.get("completion_tokens")]
    ttft = [r["ttft_s"] for r in recs if r.get("ttft_s") is not None]
    dec = [r["decode_s"] for r in recs if r.get("decode_s") is not None]
    tot = [r["total_s"] for r in recs if r.get("total_s") is not None]
    sum_pt, sum_ct = sum(pt), sum(ct)
    sum_ttft, sum_dec, sum_tot = sum(ttft), sum(dec), sum(tot)
    # Per-request bandwidth (mean of tokens/time per request) —
    # reflects what one request sees when it's one of K contending.
    per_req_prefill = [
        r["prompt_tokens"] / r["ttft_s"]
        for r in recs if r.get("prompt_tokens") and r.get("ttft_s", 0) > 0
    ]
    per_req_decode = [
        r["completion_tokens"] / r["decode_s"]
        for r in recs if r.get("completion_tokens") and r.get("decode_s", 0) > 0
    ]
    return {
        "class": label, "n": len(recs),
        "sum_prompt_tokens": sum_pt,
        "sum_completion_tokens": sum_ct,
        "mean_ttft_s": sum_ttft / len(ttft) if ttft else None,
        "mean_decode_s": sum_dec / len(dec) if dec else None,
        "mean_total_s": sum_tot / len(tot) if tot else None,
        # Aggregate bandwidth — server throughput under this wave's load.
        "agg_prefill_tok_per_s": sum_pt / wave_wall_s if wave_wall_s else None,
        "agg_decode_tok_per_s": sum_ct / wave_wall_s if wave_wall_s else None,
        "agg_total_tok_per_s": (sum_pt + sum_ct) / wave_wall_s if wave_wall_s else None,
        # Per-request bandwidth (mean across requests).
        "mean_per_req_prefill_tok_per_s": (
            sum(per_req_prefill) / len(per_req_prefill) if per_req_prefill else None),
        "mean_per_req_decode_tok_per_s": (
            sum(per_req_decode) / len(per_req_decode) if per_req_decode else None),
    }


def wave_bandwidth_report(wave_results: list[dict], wave_wall_s: float) -> dict:
    """Compute prefill/decode/total tokens-per-sec overall + by image class.

    Classes: by num_images on the request.
      - text_only:    num_images == 0
      - single_image: num_images == 1
      - multi_image:  num_images >= 2
    """
    recs = _collect_requests(wave_results)
    zero = [r for r in recs if r.get("num_images", 0) == 0]
    one = [r for r in recs if r.get("num_images", 0) == 1]
    many = [r for r in recs if r.get("num_images", 0) >= 2]
    return {
        "wave_wall_s": wave_wall_s,
        "n_requests": len(recs),
        "overall": _class_agg(recs, "overall", wave_wall_s),
        "by_class": [
            _class_agg(zero, "text_only",    wave_wall_s),
            _class_agg(one,  "single_image", wave_wall_s),
            _class_agg(many, "multi_image",  wave_wall_s),
        ],
    }


def print_bandwidth(report: dict, prefix: str = "") -> None:
    o = report["overall"]
    if o["n"] == 0:
        print(f"{prefix}  (no per-request metrics captured)")
        return
    print(f"{prefix}  overall: n={o['n']}  "
          f"prefill={o['agg_prefill_tok_per_s']:.1f} tok/s  "
          f"decode={o['agg_decode_tok_per_s']:.1f} tok/s  "
          f"total={o['agg_total_tok_per_s']:.1f} tok/s")
    for c in report["by_class"]:
        if c["n"] == 0: continue
        print(f"{prefix}    [{c['class']:<12}] n={c['n']:<3}  "
              f"prefill_agg={c['agg_prefill_tok_per_s']:.1f}  "
              f"decode_agg={c['agg_decode_tok_per_s']:.1f}  "
              f"per_req_prefill={c['mean_per_req_prefill_tok_per_s'] or 0:.1f}  "
              f"per_req_decode={c['mean_per_req_decode_tok_per_s'] or 0:.1f}")


# ── ASCII summary utilities ──────────────────────────────────────────

def hist(values: list[float], buckets: int = 20, label: str = "") -> str:
    if not values: return f"{label}: (empty)"
    lo, hi = min(values), max(values)
    if hi == lo:
        return f"{label}: all {lo:.4f} (n={len(values)})"
    width = (hi - lo) / buckets
    counts = [0] * buckets
    for v in values:
        k = min(buckets - 1, int((v - lo) / width))
        counts[k] += 1
    mx = max(counts)
    bar_w = 40
    lines = [f"{label} n={len(values)} min={lo:.4f} max={hi:.4f} "
             f"median={sorted(values)[len(values)//2]:.4f}"]
    for i, c in enumerate(counts):
        b_lo = lo + i * width
        b_hi = lo + (i + 1) * width
        bar = "█" * int(bar_w * c / mx)
        lines.append(f"  [{b_lo:>8.4f} .. {b_hi:>8.4f}]  {c:>4}  {bar}")
    return "\n".join(lines)


def summarize_report(report: dict) -> None:
    if "by_K" in report:
        print("\n=== profile summary ===")
        for K, data in sorted(report["by_K"].items()):
            total_i = sum(w["n_iterations"] for w in data["waves"])
            total_w = sum(w["wave_wall_s"] for w in data["waves"])
            print(f"  K={K:>2}  {total_i:>4} iters / {total_w:>6.1f}s = "
                  f"{total_i/max(total_w,1e-6):>6.3f} iter/s agg  "
                  f"(per-client avg: "
                  f"{total_w/max(sum(w['n_clients'] for w in data['waves']), 1):.2f}s/client)")
    if "trials" in report:
        all_mse = []
        for t in report["trials"]:
            for c in t["clients"]:
                if isinstance(c, dict) and "best_mse" in c:
                    all_mse.append(c["best_mse"])
        print("\n=== curriculum summary ===")
        print(f"  trials: {report['total_trials']}")
        print(f"  iterations: {report['total_iterations']}")
        print(f"  wall: {report['total_wall_s']:.0f}s")
        print(f"  throughput: {report['total_iterations']/max(report['total_wall_s'],1e-6):.2f} iter/s")
        print(hist(all_mse, buckets=15, label="best_MSE distribution"))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest="cmd", required=True)

    p1 = subs.add_parser("profile",
        help="short throughput sweep over K ∈ {4,6,8,12}. K<4 is rejected "
             "— single-client performance is not a target.")
    p1.add_argument("--K-values", type=int, nargs="+", default=[4, 6, 8, 12])
    p1.add_argument("--R", type=int, default=1, help="waves per K")
    p1.add_argument("--size", type=int, default=128)
    p1.add_argument("--max-iters", type=int, default=3)
    p1.add_argument("--mode", default="svg", choices=["svg", "python", "mix"])

    p2 = subs.add_parser("curriculum",
        help="long run against an image directory")
    p2.add_argument("image_dir", type=pathlib.Path)
    p2.add_argument("--K", type=int, default=4)
    p2.add_argument("--max-iters", type=int, default=4)
    p2.add_argument("--max-wall-s", type=float, default=6 * 3600)
    p2.add_argument("--max-iters-total", type=int, default=None)
    p2.add_argument("--size", type=int, default=128)
    p2.add_argument("--mode", default="mix", choices=["svg", "python", "mix"])

    args = ap.parse_args()
    if args.cmd == "profile":
        bad = [k for k in args.K_values if k < 4]
        if bad:
            print(f"[bench] rejecting K-values < 4: {bad}. "
                  f"Minimum concurrency is 4 (natural client count).",
                  file=sys.stderr)
            sys.exit(2)
    elif args.cmd == "curriculum" and args.K < 4:
        print(f"[bench] rejecting --K {args.K}. Minimum is 4.",
              file=sys.stderr)
        sys.exit(2)

    out_base = RUNS / f"bench_{int(time.time())}"
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"[bench] output → {out_base}")

    if args.cmd == "profile":
        report = profile_sweep(
            K_values=args.K_values, R=args.R, size=args.size,
            max_iters=args.max_iters, mode=args.mode, out_base=out_base,
        )
        # Strip engine samples from report before save (they get huge).
        save = {"by_K": {K: {k: v for k, v in d.items() if k != "engine_samples"}
                          for K, d in report["by_K"].items()}}
        (out_base / "profile_report.json").write_text(json.dumps(save, indent=2))
        for K, d in report["by_K"].items():
            sfile = out_base / f"engine_samples_K{K:02d}.json"
            sfile.write_text(json.dumps(d["engine_samples"]))
        summarize_report(report)
    else:
        report = run_curriculum(
            image_dir=args.image_dir, K=args.K, max_iters=args.max_iters,
            mode_spec=args.mode, size=args.size, out_base=out_base,
            max_wall_s=args.max_wall_s, max_iters_total=args.max_iters_total,
        )
        save = {k: v for k, v in report.items() if k != "engine_samples"}
        (out_base / "curriculum_report.json").write_text(json.dumps(save, indent=2))
        (out_base / "engine_samples.json").write_text(
            json.dumps(report["engine_samples"]))
        summarize_report(report)


if __name__ == "__main__":
    main()
