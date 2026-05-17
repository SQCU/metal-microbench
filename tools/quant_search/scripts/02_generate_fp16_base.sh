#!/usr/bin/env bash
# Convert the Gemma-4-A4B bf16 safetensors → fp16 GGUF.
# This is the source artifact for all subsequent llama-quantize invocations
# in the search grid (each grid config quantizes from this base).
#
# Idempotent: skip if output GGUF already exists.

set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-/Users/mdot/llama.cpp}"
SOURCE_DIR="${SOURCE_DIR:-/Users/mdot/models/gemma-4-a4b-bf16}"
OUTPUT_GGUF="${OUTPUT_GGUF:-/Users/mdot/models/gemma-4-a4b-quant-search/gemma-4-26B-A4B-it-fp16.gguf}"

if [[ -f "$OUTPUT_GGUF" ]]; then
    SIZE=$(du -h "$OUTPUT_GGUF" | awk '{print $1}')
    echo "[02] FP16 base already exists ($SIZE) at $OUTPUT_GGUF"
    exit 0
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "[02] FAIL — source dir $SOURCE_DIR does not exist"
    exit 1
fi

if [[ ! -f "$SOURCE_DIR/config.json" ]]; then
    echo "[02] FAIL — $SOURCE_DIR has no config.json (not an HF model dir)"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_GGUF")"

echo "[02] converting $SOURCE_DIR → $OUTPUT_GGUF ..."
echo "[02] (this takes ~5 min and produces ~52GB on disk for a 26B-A4B bf16 model)"

# Use the venv that has gguf + transformers + torch; if not present, fall back
# to system python and let it fail loudly.
PYTHON="${PYTHON:-python3}"
for venv_path in \
    "$LLAMA_CPP/.venv/bin/python" \
    "$LLAMA_CPP/venv/bin/python"; do
    if [[ -x "$venv_path" ]]; then
        PYTHON="$venv_path"
        echo "[02] using $PYTHON"
        break
    fi
done

cd "$LLAMA_CPP"
"$PYTHON" convert_hf_to_gguf.py \
    "$SOURCE_DIR" \
    --outtype f16 \
    --outfile "$OUTPUT_GGUF"

if [[ ! -f "$OUTPUT_GGUF" ]]; then
    echo "[02] FAIL — conversion did not produce $OUTPUT_GGUF"
    exit 1
fi
SIZE=$(du -h "$OUTPUT_GGUF" | awk '{print $1}')
echo "[02] OK — produced $OUTPUT_GGUF ($SIZE)"
