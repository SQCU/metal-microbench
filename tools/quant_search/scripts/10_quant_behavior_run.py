#!/usr/bin/env python3
"""Quant-behavior study orchestrator: runs `09_hellaswag_kl_study` on
N configs sequentially with verified-clean bridge lifecycle between
them. No shell commands, no orphan bridges, no manual pkill cycles.

Configs are listed at the top of this file. Each entry is a (label,
gguf_path) tuple. The orchestrator:

  1. SIGKILL any existing bridges + verify (via bridge_lifecycle)
  2. Launch bridge with config A → wait /health ready
  3. Run the study, write per-config JSONL
  4. Exit context manager → SIGKILL bridge + verify
  5. Repeat for config B
  6. Post-process across the per-config JSONLs

Usage::

    N_ITEMS=30 K_SAMPLES=8 SAMPLE_TEMPERATURE=1.0 \\
    STRATEGIES=model_aware_mc CONCURRENCY=12 \\
    ./server/.venv/bin/python tools/quant_search/scripts/10_quant_behavior_run.py

Environment:
  N_ITEMS, K_SAMPLES, SAMPLE_TEMPERATURE, STRATEGIES, CONCURRENCY,
  TOP_LOGPROBS, MC_MAX_TOKENS, FREE_MAX_TOKENS — passed through to the
  study script (see 09_hellaswag_kl_study.py for defaults).
  OUTPUT_DIR — where per-config JSONLs land (default /tmp).
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools" / "quant_search"
SCRIPTS_DIR = TOOLS_DIR / "scripts"

sys.path.insert(0, str(TOOLS_DIR))

from bridge_lifecycle import BridgeContext   # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Config list — edit to add/remove quant configs to compare.
# ──────────────────────────────────────────────────────────────────────


CONFIGS: list[tuple[str, str]] = [
    ("fp16",
     "/Users/mdot/models/gemma-4-a4b-quant-search/gemma-4-26B-A4B-it-fp16.gguf"),
    # Unsloth Dynamic: heterogeneous per-tensor quantization
    # (Q4_K MoE up/gate + Q5_1 MoE down + Q8_0 attention/dense +
    # F32 norms). Filename happens to contain "Q4_K_M" but the
    # quantization is NOT uniform Q4_K_M — see gguf metadata for
    # the actual per-tensor breakdown.
    ("unsloth_dyn",
     "/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
]

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp"))


# ──────────────────────────────────────────────────────────────────────
# Load the study module by file path (numeric-prefixed filename can't
# be imported normally).
# ──────────────────────────────────────────────────────────────────────


_study_path = SCRIPTS_DIR / "09_hellaswag_kl_study.py"
_spec = importlib.util.spec_from_file_location("hellaswag_study", _study_path)
study = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(study)


async def run_one_config(label: str, gguf_path: str) -> Path:
    """Lifecycle-managed run for a single config. Returns the JSONL path."""
    out_path = OUTPUT_DIR / f"quant_behavior_{label}.jsonl"
    print(f"\n{'=' * 64}\n  CONFIG: {label}  ({Path(gguf_path).name})\n"
          f"{'=' * 64}", flush=True)
    with BridgeContext(gguf_path) as bridge:
        # Override the URL the study uses so it always talks to the bridge
        # we just launched. The study's defaults are already 127.0.0.1:8001
        # but we set explicitly for clarity in case BridgeContext picks a
        # non-default port in the future.
        study.BRIDGE_URL = bridge.url
        await study.run_collection(out_path)
    return out_path


async def main() -> int:
    paths: dict[str, Path] = {}
    for label, gguf in CONFIGS:
        paths[label] = await run_one_config(label, gguf)

    # Post-process across paths. Study's post_process reads FP16_PATH
    # and QUANT_PATH from environ; populate them from our results.
    if "fp16" in paths and len(paths) >= 2:
        os.environ["FP16_PATH"] = str(paths["fp16"])
        # Use the first non-fp16 entry as the "quant" comparison.
        quant_label = next(k for k in paths if k != "fp16")
        os.environ["QUANT_PATH"] = str(paths[quant_label])
        print(f"\n{'=' * 64}\n  POST-PROCESS (fp16 vs {quant_label})\n"
              f"{'=' * 64}", flush=True)
        study.post_process()
    else:
        print(f"\n[run] {len(paths)} configs collected; skipping "
              f"post-process (need fp16 + at least one other).",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
