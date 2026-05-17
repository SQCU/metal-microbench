#!/usr/bin/env python3
"""Multi-benchmark, multi-config quant-behavior orchestrator.

Iterates over (config × benchmark), runs each via framework.run_benchmark
with the per-config bridge managed by BridgeContext (clean SIGKILL
lifecycle — no shell pkill, no orphan bridges). After all combinations
collect, runs cross-config aggregation per benchmark and prints
a side-by-side summary.

Usage::

    N_ITEMS=30 K_SAMPLES=8 SAMPLE_TEMPERATURE=1.0 CONCURRENCY=12 \\
    BENCHMARKS=hellaswag,algebra \\
    OUTPUT_DIR=/tmp/multibench \\
    ./server/.venv/bin/python tools/quant_search/scripts/11_multibench_run.py

Configs are listed inline (CONFIGS) — extend that list to compare more
quant strategies.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools" / "quant_search"
sys.path.insert(0, str(TOOLS_DIR))

from bridge_lifecycle import BridgeContext           # noqa: E402
from framework import (                                              # noqa: E402
    RunConfig,
    run_benchmark,                # kept for legacy callers
    run_benchmarks_pooled,        # the unified-pool runner we use here
    aggregate_cross_config,
    render_summary,
)
import benchmarks as bm                              # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Configs to compare. Add more (Q5_K_M, Q4_0, etc.) when GGUFs land.
# ──────────────────────────────────────────────────────────────────────


# Each config is a fine-grained parameter-group quantization probe to
# compare against the fp16 reference. We are NOT testing uniform-
# quantization configs — without quantization-aware post-training there
# is no reason to expect every "anatomical unit" of an optimized policy
# to be equally tolerant of the same precision target. The probes
# below are heterogeneous per-tensor quantization strategies.
CONFIGS: list[tuple[str, str]] = [
    ("fp16",
     "/Users/mdot/models/gemma-4-a4b-quant-search/gemma-4-26B-A4B-it-fp16.gguf"),
    # Unsloth Dynamic ("UD-Q4_K_M.gguf"): heterogeneous per-tensor
    # quantization. Per gguf metadata:
    #   Q4_K  ×30  (50.8% of weight bytes)  — MoE ffn_gate_up_exps
    #   Q5_1  ×30  (33.9%)                  — MoE ffn_down_exps
    #   Q8_0  ×206 (15.0%)                  — attention (q/k/v/output),
    #                                          dense FFN, token embed
    #   F32   ×392 (0.3%)                   — norms, gate routing scales
    # Strategy: spend bandwidth on attention + routing where every-
    # token precision matters; aggressively compress the MoE expert
    # FFNs where the model already has heterogeneity to absorb loss.
    ("unsloth_dyn",
     "/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
]

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/multibench"))


def _selected_benchmarks() -> dict[str, "bm.Benchmark"]:
    sel = os.environ.get("BENCHMARKS", "").strip()
    if not sel:
        return bm.BENCHMARKS
    out = {}
    for name in [n.strip() for n in sel.split(",") if n.strip()]:
        if name not in bm.BENCHMARKS:
            print(f"[multibench] unknown benchmark {name!r}; "
                  f"available: {list(bm.BENCHMARKS)}", file=sys.stderr)
            sys.exit(2)
        out[name] = bm.BENCHMARKS[name]
    return out


def _path_for(label: str, benchmark: str) -> Path:
    return OUTPUT_DIR / f"{benchmark}_{label}.jsonl"


async def main() -> int:
    sel = _selected_benchmarks()
    print(f"[multibench] benchmarks: {list(sel)}", flush=True)
    print(f"[multibench] configs:    {[lab for lab, _ in CONFIGS]}",
          flush=True)
    print(f"[multibench] output_dir: {OUTPUT_DIR}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_items     = int(os.environ.get("N_ITEMS", 30))
    k_samples   = int(os.environ.get("K_SAMPLES", 8))
    temperature = float(os.environ.get("SAMPLE_TEMPERATURE", 1.0))
    concurrency = int(os.environ.get("CONCURRENCY", 12))
    # Whole-orchestrator wall-clock timeout. If the entire study runs
    # past this many seconds, the orchestrator gives up — different
    # mechanism than per-task timeouts (which we don't have, because
    # they conflate queue wait with decode time). Default 0 = no
    # timeout. Set ORCH_TIMEOUT_S to e.g. 1800 for a 30-minute global
    # ceiling. Implemented as a ground-truth wall-clock check at every
    # benchmark/config boundary; a slow benchmark won't be cancelled
    # mid-flight, but the orchestrator will stop scheduling new work
    # once the budget is exhausted.
    orch_timeout_s = float(os.environ.get("ORCH_TIMEOUT_S", "0") or "0")

    import time
    orch_t0 = time.time()
    skipped_for_timeout: list[tuple[str, str]] = []

    # Phase 1: collect rollouts.
    #
    # Per-config we run ALL selected benchmarks under a single shared
    # worker pool (run_benchmarks_pooled). Tickets from different
    # benchmarks compete for the same `concurrency`-sized semaphore;
    # there is no "drain hellaswag fully before mmlu starts" tail-end
    # serialization. The bridge sees a steady mix of streams from
    # whichever benchmarks have outstanding work, which keeps the
    # batched-decode utilization high until literally the last few
    # rollouts finish.
    for cfg_label, gguf in CONFIGS:
        if orch_timeout_s and (time.time() - orch_t0) >= orch_timeout_s:
            print(f"[multibench] orchestrator timeout ({orch_timeout_s}s) "
                  f"reached; skipping config {cfg_label} and beyond",
                  flush=True)
            skipped_for_timeout.append((cfg_label, "*"))
            continue
        print(f"\n{'=' * 64}\n  CONFIG: {cfg_label}  ({Path(gguf).name})\n"
              f"{'=' * 64}", flush=True)
        with BridgeContext(gguf) as bridge:
            # The model_name string is the OAI `model:` field we send
            # in chat-completion requests; it has no effect on which
            # weights the bridge actually loads (that's gguf, set by
            # BridgeContext). But it gets echoed in usage logs and
            # /v1/models, so we make it match the active config label
            # — `fp16`, `unsloth_dyn`, etc. — instead of a stale
            # uniform-quant string that misrepresents what's loaded.
            run_cfg = RunConfig(
                bridge_url=bridge.url,
                model_name=os.environ.get(
                    "MODEL_NAME", f"gemma-4-a4b-{cfg_label}"),
                n_items=n_items,
                k_samples=k_samples,
                sample_temperature=temperature,
                concurrency=concurrency,
            )
            # Point benchmark module's judge calls at the active bridge.
            bm.BRIDGE_URL = bridge.url
            bm.MODEL_NAME = run_cfg.model_name
            out_paths = {name: _path_for(cfg_label, name)
                          for name in sel}
            await run_benchmarks_pooled(sel, run_cfg, out_paths)
    if skipped_for_timeout:
        print(f"\n[multibench] skipped {len(skipped_for_timeout)} (config, "
              f"benchmark) cells due to orchestrator timeout: "
              f"{skipped_for_timeout}", flush=True)

    # Phase 2: aggregate. For each benchmark, compare each non-fp16
    # config to fp16 (the natural reference).
    print(f"\n{'=' * 64}\n  CROSS-CONFIG AGGREGATION\n{'=' * 64}",
          flush=True)
    ref_label = CONFIGS[0][0]
    summaries: list = []
    for bench_name, benchmark in sel.items():
        ref_recs = [json.loads(l) for l in _path_for(ref_label, bench_name).open()]
        for cmp_label, _ in CONFIGS[1:]:
            cmp_recs = [json.loads(l) for l in
                          _path_for(cmp_label, bench_name).open()]
            summary = aggregate_cross_config(
                ref_recs, cmp_recs, benchmark,
                ref_label=ref_label, cmp_label=cmp_label)
            summaries.append(summary)
            print()
            print(render_summary(summary))

    # Compact "quality vector" view across benchmarks for each cmp config.
    print(f"\n{'=' * 64}\n  QUALITY VECTORS (vs {ref_label})\n{'=' * 64}",
          flush=True)
    by_cmp: dict[str, list] = {}
    for s in summaries:
        by_cmp.setdefault(s.cmp_label, []).append(s)
    for cmp_label, ss in by_cmp.items():
        print(f"\n{cmp_label}:")
        for s in ss:
            print(f"  {s.benchmark:<10}  "
                  f"status_tv={s.status_tv:.3f}  "
                  f"len_W={s.gen_chars_wasserstein:>6.0f}  "
                  f"eos_diff={s.hit_eos_rate_diff:.3f}  "
                  f"committed_dist={s.ok_metric_distance:.3f}  "
                  f"paired_agree="
                  f"{s.paired_status_agreement*100:.1f}%")
    print()

    # Phase 3: SVG-MSE cross-config drift, computed on PNGs that were
    # rendered DURING the eval (each multi-turn refinement rollout
    # stores its final rendered PNG as base64 in judge_meta). No
    # rendering at this stage — just decode + MSE math.
    #
    # Per prompt p we compute three MSE distributions:
    #   - within-ref  (K choose 2 same-config pairs: noise floor)
    #   - within-cmp  (K choose 2 same-config pairs: noise floor)
    #   - cross r/c   (K × K cross-config pairs)
    # Drift signal: mean(cross) − max(mean(within_*)). Positive value
    # means cmp has shifted out of ref's elicitation manifold beyond
    # the within-config noise floor.
    if "svg" in sel:
        print(f"\n{'=' * 64}\n  SVG-MSE DRIFT (renders from in-loop "
              f"refinement)\n{'=' * 64}", flush=True)
        try:
            import statistics                                          # noqa: PLC0415
            from svg_canonical import png_pair_mse                     # noqa: PLC0415

            def _png_b64s_by_item(jsonl_path: Path) -> dict[str, list[str]]:
                """Group base64 PNGs by item_id, sample-ordered. Records
                without final_png_b64 (render failures, no_commit)
                contribute None placeholders — we drop them at MSE time."""
                by_item: dict[str, list] = {}
                for line in jsonl_path.open():
                    r = json.loads(line)
                    b64 = (r.get("judge_meta") or {}).get("final_png_b64")
                    by_item.setdefault(r["item_id"], []).append(
                        (r.get("sample_idx", 0), b64))
                # Sort by sample_idx, drop None.
                out: dict[str, list[str]] = {}
                for it, pairs in by_item.items():
                    pairs.sort(key=lambda x: x[0])
                    out[it] = [b for _, b in pairs if b is not None]
                return out

            ref_svg_path = _path_for(ref_label, "svg")
            for cmp_label, _ in CONFIGS[1:]:
                cmp_svg_path = _path_for(cmp_label, "svg")
                if not (ref_svg_path.exists() and cmp_svg_path.exists()):
                    print(f"[svg_mse] missing jsonl for {cmp_label}; skip",
                          flush=True)
                    continue
                ref_by = _png_b64s_by_item(ref_svg_path)
                cmp_by = _png_b64s_by_item(cmp_svg_path)
                common = sorted(set(ref_by) & set(cmp_by))

                per_prompt: list[dict] = []
                for it in common:
                    ref_pngs = ref_by[it]
                    cmp_pngs = cmp_by[it]
                    within_ref = []
                    for i in range(len(ref_pngs)):
                        for j in range(i + 1, len(ref_pngs)):
                            within_ref.append(png_pair_mse(ref_pngs[i], ref_pngs[j]))
                    within_cmp = []
                    for i in range(len(cmp_pngs)):
                        for j in range(i + 1, len(cmp_pngs)):
                            within_cmp.append(png_pair_mse(cmp_pngs[i], cmp_pngs[j]))
                    cross = [png_pair_mse(a, b)
                             for a in ref_pngs for b in cmp_pngs]
                    if not (cross and (within_ref or within_cmp)):
                        per_prompt.append({"item_id": it, "drift": float("nan"),
                                           "n_ref": len(ref_pngs), "n_cmp": len(cmp_pngs)})
                        continue
                    floor = max(
                        statistics.mean(within_ref) if within_ref else 0.0,
                        statistics.mean(within_cmp) if within_cmp else 0.0)
                    per_prompt.append({
                        "item_id": it,
                        "n_ref": len(ref_pngs), "n_cmp": len(cmp_pngs),
                        "within_ref_mean": statistics.mean(within_ref) if within_ref else float("nan"),
                        "within_cmp_mean": statistics.mean(within_cmp) if within_cmp else float("nan"),
                        "cross_mean":      statistics.mean(cross),
                        "within_floor":    floor,
                        "drift":           statistics.mean(cross) - floor,
                    })

                drifts = [p["drift"] for p in per_prompt
                            if isinstance(p["drift"], float)
                                and p["drift"] == p["drift"]]
                cross_means = [p["cross_mean"] for p in per_prompt
                                if "cross_mean" in p]
                floors = [p["within_floor"] for p in per_prompt
                            if "within_floor" in p]

                def _mean(xs):
                    return statistics.mean(xs) if xs else float("nan")

                summary = {
                    "n_prompts": len(per_prompt),
                    "n_prompts_with_data": len(drifts),
                    "mean_drift":   _mean(drifts),
                    "median_drift": (statistics.median(drifts) if drifts else float("nan")),
                    "max_drift":    (max(drifts) if drifts else float("nan")),
                    "mean_cross":   _mean(cross_means),
                    "mean_within_floor": _mean(floors),
                }

                print()
                print(f"  {cmp_label} vs {ref_label}:")
                print(f"    n_prompts          {summary['n_prompts_with_data']}/"
                      f"{summary['n_prompts']}")
                print(f"    mean_drift         {summary['mean_drift']:.6f}  "
                      f"(positive = behavioral shift above noise floor)")
                print(f"    median_drift       {summary['median_drift']:.6f}")
                print(f"    max_drift          {summary['max_drift']:.6f}")
                print(f"    mean_cross         {summary['mean_cross']:.6f}")
                print(f"    mean_within_floor  {summary['mean_within_floor']:.6f}")

                out_json = OUTPUT_DIR / f"svg_mse_{cmp_label}_vs_{ref_label}.json"
                out_json.write_text(json.dumps(
                    {"per_prompt": per_prompt, "summary": summary,
                     "ref_label": ref_label, "cmp_label": cmp_label},
                    indent=2))
                print(f"    [persisted] {out_json}")
        except Exception as e:                       # noqa: BLE001
            print(f"[svg_mse] aggregation failed: {e!r}", flush=True)
            import traceback
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
