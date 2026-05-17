#!/usr/bin/env bash
# Master setup: run all four prerequisites in order so a fresh checkout
# can land at "ready to search" with one command.
#
# Idempotent: each sub-script is itself idempotent. Re-running this is cheap.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  Quant search setup"
echo "=========================================="
echo
echo "Will run, in order:"
echo "  01_build_llama_quantize.sh   — build llama-quantize binary"
echo "  02_generate_fp16_base.sh     — convert bf16 safetensors → fp16 GGUF"
echo "  03_materialize_grid.sh       — produce 7 quantized GGUFs (~30 min)"
echo "  05_generate_svg_refs.sh      — produce 5 reference PNGs (~5 min)"
echo
echo "(04_toolcards_runner.mjs is a long-running service, started separately)"
echo

bash "$SCRIPT_DIR/01_build_llama_quantize.sh"
echo
bash "$SCRIPT_DIR/02_generate_fp16_base.sh"
echo
bash "$SCRIPT_DIR/03_materialize_grid.sh"
echo
echo "  → 05_generate_svg_refs.sh requires the toolcards runner to be"
echo "    running. In another terminal: "
echo "      node $SCRIPT_DIR/04_toolcards_runner.mjs"
echo "    then re-run this setup script (it'll skip the steps already done)."
echo
if curl -sf "${TOOLCARDS_URL:-http://127.0.0.1:8002}/health" > /dev/null 2>&1; then
    bash "$SCRIPT_DIR/05_generate_svg_refs.sh"
else
    echo "  (skipping 05_generate_svg_refs — toolcards runner not up)"
fi

echo
echo "=========================================="
echo "  Setup complete. To run the long-run multi-config benchmark:"
echo
echo "    cd $(realpath "$SCRIPT_DIR/../../..")"
echo "    ./server/.venv/bin/python tools/quant_search/scripts/08_long_run.py"
echo
echo "  Configurable via env (see 08_long_run.py docstring for the full list):"
echo "    LONG_RUN_RESULTS    output JSONL path (default /tmp/long_run_results.jsonl)"
echo "    LONG_RUN_CONFIGS    comma-separated list of config tags"
echo "    LONG_RUN_HARNESSES  comma-separated list of harness names"
echo "    LONG_RUN_BUDGETS    JSON dict overriding per-harness budgets"
echo
echo "  Post-process /tmp/long_run_results.jsonl with jq/python."
echo "=========================================="
