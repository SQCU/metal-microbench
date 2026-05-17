#!/usr/bin/env python3
"""Multi-config benchmarking run for the quant search.

Iterates over configured GGUFs, restarting the bridge with each, running
the full workload (all four surviving harnesses: MMLU, GSM8K, SVG-MSE,
Tok/s), and writing results to JSONL on disk *as each harness finalizes*
— so a kill mid-run cannot lose any harness's already-finalized result.
Designed for an 8–10 hour contiguous run.

Per-config workflow:
  1. If GGUF is not already loaded, kill bridge → relaunch with new
     GEMMA_GGUF env var pointing at the new file → wait for /health ready.
  2. Run the workload (MMLU, GSM8K, SVG-MSE, Tok/s) at fixed activeB=8
     concurrency (matching the engine's B=8 kernel zoo cell). Per-harness
     budgets cap each harness's wall time; the orchestrator stops
     dispatching once a harness exceeds its budget.
  3. As each harness's `run()` coroutine returns, append a JSONL record:
       {"kind": "harness", "tag": ..., "harness": ..., "metrics": ...,
        "consumption": ..., "finalized_at": ...}
  4. After all harnesses for the config complete, append:
       {"kind": "config", "tag": ..., "gguf_path": ..., "started_at": ...,
        "elapsed_workload_s": ..., "concurrency_history": [...]}
  5. Continue with the next config.

Run from repo root:
    ./server/.venv/bin/python tools/quant_search/scripts/08_long_run.py

Configurable via env:
    LONG_RUN_RESULTS  output JSONL path (default /tmp/long_run_results.jsonl)
    LONG_RUN_CONFIGS  comma-separated list of config tags
                       (default: all GGUFs found in QUANT_CACHE_DIR)
    LONG_RUN_HARNESSES  comma-separated list of harness names
                         (default: all four)
    LONG_RUN_BUDGETS   JSON dict overriding per-harness budgets

Post-process with jq:
    jq -c 'select(.kind=="harness") | {tag, harness, metrics}' \
        /tmp/long_run_results.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "quant_search"))

from harnesses import (                                            # noqa: E402
    GSM8KHarness, MMLUHarness, SVGMSEHarness, TokSHarness,
)
from workload import HarnessBudget, Orchestrator                   # noqa: E402


BRIDGE_URL = os.environ.get("QUANT_BRIDGE_URL", "http://127.0.0.1:8001")
RESULTS_PATH = Path(os.environ.get(
    "LONG_RUN_RESULTS", "/tmp/long_run_results.jsonl"))
QUANT_CACHE = Path(os.environ.get(
    "QUANT_CACHE_DIR", "/Users/mdot/models/gemma-4-a4b-quant-search"))


# Harness factory map — name → constructor
HARNESS_CTORS = {
    "mmlu":       MMLUHarness,
    "gsm8k":      GSM8KHarness,
    "tok_s":      TokSHarness,
    "svg_mse":    SVGMSEHarness,
}

# Default per-harness budgets — large enough for each harness to run
# substantially through its full dataset (or saturate in tok_s/svg_mse
# which have fixed prompt sets). At expected aggregate throughput these
# add up to multi-hour wall clock per config; the budget governor stops
# any harness that would overshoot, so total per-config wall clock is
# bounded but generous.
DEFAULT_BUDGETS: dict[str, dict[str, int]] = {
    "mmlu":       {"prefill": 5_000_000,  "ar":   100_000},
    "gsm8k":      {"prefill": 1_000_000,  "ar": 1_000_000},
    "tok_s":      {"prefill":   500_000,  "ar":   200_000},
    "svg_mse":    {"prefill":   500_000,  "ar":   250_000},
}


def discover_configs() -> list[tuple[str, Path]]:
    """Find quantized GGUFs across known model dirs. Returns (tag, path)
    list. Filters out fp16/bf16 (no fp16 dense+MoE kernels in our
    engine — those GGUFs would fail to load)."""
    search_dirs = [
        QUANT_CACHE,
        Path("/Users/mdot/models/gemma-4-a4b"),
    ]
    out: list[tuple[str, Path]] = []
    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.gguf")):
            tag = (f.stem
                    .replace("gemma-4-26B-A4B-it-UD-", "")
                    .replace("gemma-4-26B-A4B-it-", "")
                    .replace("gemma-4-26B-A4B-", ""))
            # Skip non-runnable formats.
            if "fp16" in tag.lower() or "bf16" in tag.lower():
                continue
            out.append((tag, f))
    # Dedupe — sometimes a tag can appear in multiple dirs.
    seen = set()
    uniq: list[tuple[str, Path]] = []
    for tag, p in out:
        if tag in seen:
            continue
        seen.add(tag)
        uniq.append((tag, p))
    return uniq


def restart_bridge_with(gguf_path: Path) -> int:
    """Kill any running bridge, relaunch with GEMMA_GGUF=gguf_path,
    block until /health is ready. Returns the new bridge PID."""
    print(f"[long_run] restarting bridge with {gguf_path.name}", flush=True)
    # Kill any existing bridge.
    subprocess.run(["pkill", "-f", "server/serve.py"], check=False)
    time.sleep(2)
    # Launch new bridge.
    env = {**os.environ, "GEMMA_GGUF": str(gguf_path)}
    proc = subprocess.Popen(
        [str(REPO_ROOT / "server" / ".venv" / "bin" / "python"),
         str(REPO_ROOT / "server" / "serve.py")],
        env=env, cwd=REPO_ROOT,
        stdout=open("/tmp/bridge.log", "w"),
        stderr=subprocess.STDOUT,
    )
    # Wait for /health.
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{BRIDGE_URL}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ready":
                    print(f"[long_run] bridge ready (pid {proc.pid})",
                          flush=True)
                    return proc.pid
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("bridge failed to become ready within 120s")


def _append_jsonl(record: dict) -> None:
    """Append one JSONL line to RESULTS_PATH atomically. Used both for
    per-harness records (kind='harness') as each harness finalizes, and
    for the per-config summary record (kind='config') after all harnesses
    in a config complete. Post-processing joins by `tag`."""
    with RESULTS_PATH.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


async def run_one_config(tag: str, gguf_path: Path,
                          harnesses_to_run: list[str],
                          budgets: dict[str, dict[str, int]]) -> dict:
    """Restart bridge with this config's GGUF, run the workload, write
    a per-harness JSONL line as each harness finalizes, then write a
    config-summary line. Returns the summary dict for caller logging."""
    started_at = time.time()
    pid = restart_bridge_with(gguf_path)
    bridge_ready_at = time.time()

    harnesses = []
    h_budgets = {}
    for name in harnesses_to_run:
        if name not in HARNESS_CTORS:
            print(f"[long_run] unknown harness {name!r}, skipping", flush=True)
            continue
        harnesses.append(HARNESS_CTORS[name]())
        b = budgets.get(name, DEFAULT_BUDGETS.get(name, {}))
        h_budgets[name] = HarnessBudget(
            prefill_tokens=int(b.get("prefill", 1_000)),
            ar_tokens=int(b.get("ar", 1_000)),
        )

    print(f"[long_run] config={tag} harnesses={[h.name for h in harnesses]}",
          flush=True)

    # Persist each harness's result as soon as its `run()` coroutine
    # returns. The handover post-mortem documented that an 8–10 hour run
    # killed mid-config lost the one harness (KL_DIV) that had finished
    # in memory but not yet written, because persistence used to happen
    # only at config completion. This path makes that loss class
    # impossible: every finalized harness lives on disk before the
    # orchestrator's bookkeeping coroutines wind down.
    def _on_harness_finalized(
        harness_name: str,
        result: "dict | BaseException",
        consumption: dict,
    ) -> None:
        record = {
            "kind": "harness",
            "tag": tag,
            "harness": harness_name,
            "finalized_at": time.time(),
            "consumption": consumption,
        }
        if isinstance(result, BaseException):
            record["error"] = repr(result)
        else:
            record["metrics"] = result
        try:
            _append_jsonl(record)
        except Exception as e:                                  # noqa: BLE001
            # Failing to persist is bad but not fatal — keep the run
            # going so other harnesses get a chance.
            print(f"[long_run] persist({tag}/{harness_name}) failed: {e}",
                  flush=True)

    orch = Orchestrator(
        bridge_url=BRIDGE_URL,
        progress_cb=lambda m: print(f"[{tag}] {m}", flush=True),
        harness_finalized_cb=_on_harness_finalized,
    )

    workload_started_at = time.time()
    try:
        finals = await orch.run(harnesses, h_budgets)
    except Exception as e:
        print(f"[long_run] config {tag} crashed: {e}", flush=True)
        finals = {"_error": repr(e)}
    workload_finished_at = time.time()

    summary = {
        "kind": "config",
        "tag": tag,
        "gguf_path": str(gguf_path),
        "started_at": started_at,
        "bridge_ready_at": bridge_ready_at,
        "workload_started_at": workload_started_at,
        "workload_finished_at": workload_finished_at,
        "elapsed_total_s": workload_finished_at - started_at,
        "elapsed_workload_s": workload_finished_at - workload_started_at,
        "concurrency_history": orch.history,
        "harness_names": [h.name for h in harnesses],
    }
    # Config-summary record after all harness records for this tag have
    # already been flushed by the callback. Post-processing joins by tag.
    try:
        _append_jsonl(summary)
    except Exception as e:                                       # noqa: BLE001
        print(f"[long_run] persist config-summary {tag} failed: {e}",
              flush=True)
    summary["finals"] = finals  # for caller logging only; not persisted twice
    return summary


async def main() -> int:
    configs_env = os.environ.get("LONG_RUN_CONFIGS", "").strip()
    if configs_env:
        wanted = set(configs_env.split(","))
        all_configs = discover_configs()
        configs = [(t, p) for t, p in all_configs if t in wanted]
    else:
        configs = discover_configs()
    if not configs:
        print(f"[long_run] no configs found in {QUANT_CACHE}", file=sys.stderr)
        return 1

    harnesses_env = os.environ.get("LONG_RUN_HARNESSES", "").strip()
    if harnesses_env:
        harnesses_to_run = harnesses_env.split(",")
    else:
        harnesses_to_run = list(HARNESS_CTORS.keys())

    budgets_env = os.environ.get("LONG_RUN_BUDGETS", "").strip()
    if budgets_env:
        budgets = json.loads(budgets_env)
    else:
        budgets = DEFAULT_BUDGETS

    print(f"[long_run] {len(configs)} configs to run:")
    for t, p in configs:
        print(f"  {t}  ({p.name})")
    print(f"[long_run] harnesses: {harnesses_to_run}")
    print(f"[long_run] results → {RESULTS_PATH}")
    print()

    overall_t0 = time.time()
    for i, (tag, path) in enumerate(configs, 1):
        print(f"\n========== config {i}/{len(configs)}: {tag} ==========",
              flush=True)
        # `run_one_config` writes per-harness records as each harness
        # finalizes, then a config-summary record at the end. Nothing
        # extra to persist here — it's already on disk.
        await run_one_config(tag, path, harnesses_to_run, budgets)
        elapsed = time.time() - overall_t0
        print(f"[long_run] {tag} done; aggregate elapsed {elapsed/60:.1f} min",
              flush=True)

    total = time.time() - overall_t0
    print(f"\n[long_run] ALL DONE in {total/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
