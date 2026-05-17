#!/usr/bin/env bash
# Materialize the V1 search grid by invoking llama-quantize for each
# uniform-quant target. Cached by output filename — each invocation
# skips if its target already exists.
#
# After this script: 8 GGUFs in QUANT_CACHE_DIR, one per V1 grid config.

set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-/Users/mdot/llama.cpp}"
LLAMA_QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"
BASE_GGUF="${BASE_GGUF:-/Users/mdot/models/gemma-4-a4b-quant-search/gemma-4-26B-A4B-it-fp16.gguf}"
CACHE_DIR="${QUANT_CACHE_DIR:-/Users/mdot/models/gemma-4-a4b-quant-search}"

if [[ ! -x "$LLAMA_QUANTIZE" ]]; then
    echo "[03] FAIL — llama-quantize not built. Run 01_build_llama_quantize.sh first."
    exit 1
fi
if [[ ! -f "$BASE_GGUF" ]]; then
    echo "[03] FAIL — base GGUF not present. Run 02_generate_fp16_base.sh first."
    exit 1
fi

mkdir -p "$CACHE_DIR"

# Grid: tag → llama-quantize type. Mirrors initial_search_grid() in
# quant_driver.py, but expressed as bash-friendly tag strings.
#
# NOTE: this materializes uniform (or near-uniform) llama-quantize
# preset configs as INFRASTRUCTURE EXEMPLARS only. They are not
# valid research probes for this project: there is no reason to
# expect every "anatomical unit" of an optimized policy to be
# equally tolerant of the same precision target without
# quantization-aware post-training. Real probes are heterogeneous
# per-tensor configs (e.g. Unsloth Dynamic, custom mixes) — see
# scripts/11_multibench_run.py CONFIGS for the actual study set.
declare -a GRID=(
    "Q4_0"
    "Q4_K_M"          # llama.cpp standard K-quant mix (default Q5_K, MoE Q4_K).
                      #   NOT the same as Unsloth Dynamic — Unsloth Dynamic
                      #   uses Q4_K MoE up/gate + Q5_1 MoE down + Q8_0 for
                      #   attention/dense + F32 for norms.
    "Q4_K_S"          # smaller mix (default Q4_K, MoE Q4_K)
    "Q5_1"
    "Q5_K_M"          # default Q5_K, MoE Q5_K
    "Q6_K"
    "Q8_0"
)

echo "[03] materializing ${#GRID[@]} grid configs to $CACHE_DIR ..."
for QUANT in "${GRID[@]}"; do
    TARGET="$CACHE_DIR/gemma-4-26B-A4B-it-${QUANT}.gguf"
    if [[ -f "$TARGET" ]]; then
        SIZE=$(du -h "$TARGET" | awk '{print $1}')
        echo "[03]   ✓ $QUANT already at $TARGET ($SIZE)"
        continue
    fi
    echo "[03]   → materializing $QUANT ..."
    "$LLAMA_QUANTIZE" "$BASE_GGUF" "$TARGET" "$QUANT"
    if [[ ! -f "$TARGET" ]]; then
        echo "[03]   ! FAIL — llama-quantize produced no output for $QUANT"
        exit 1
    fi
    SIZE=$(du -h "$TARGET" | awk '{print $1}')
    echo "[03]   ✓ $QUANT done ($SIZE) at $TARGET"
done

echo "[03] OK — grid materialized:"
ls -lh "$CACHE_DIR"/*.gguf | awk '{print "  " $5 "  " $9}'
