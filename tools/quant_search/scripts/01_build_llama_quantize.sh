#!/usr/bin/env bash
# Build llama.cpp's llama-quantize binary if not already present.
# Idempotent: noops if the binary already exists.
#
# Required for materialize_grid.sh and any quant_driver.py call.

set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-/Users/mdot/llama.cpp}"
TARGET="$LLAMA_CPP/build/bin/llama-quantize"

if [[ -x "$TARGET" ]]; then
    echo "[01] llama-quantize already built at $TARGET"
    exit 0
fi

echo "[01] building llama-quantize at $LLAMA_CPP/build ..."
cd "$LLAMA_CPP/build"
cmake --build . --config Release --target llama-quantize -j 8

if [[ ! -x "$TARGET" ]]; then
    echo "[01] FAIL — build did not produce $TARGET"
    exit 1
fi
echo "[01] OK — built $TARGET"
"$TARGET" --help 2>&1 | head -3
