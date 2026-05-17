# Quant search — substrate-aware quantization Pareto-frontier explorer

Tool for measuring (KL-divergence, tok/s) tradeoffs across GGUF
quantization choices on this engine, on this hardware, with the current
kernel set. Targets the question: *what's the fastest model on M5 Max
that maintains acceptable quality?* — which is the inverse of standard
GGUF tooling, which optimizes smallest-model-at-fixed-quality.

## Status

V1 scaffolding. Runs:

- **tok_s oracle**: `multi_stream_test.mjs` at activeB ∈ {1, 4, 8}, warm 3-trial median.
- **quality battery**: MMLU, GSM8K, and SVG-MSE harnesses driven entirely through `/v1/chat/completions` (greedy decode, scored against gold answers / reference rasterizations). Pure public-API — no engine bypass.
- **svg_mse oracle**: skeleton in place; full toolcard wiring is TODO (the existing SVG harness needs a CLI shim).

What's NOT in V1:

- Cost model / learned predictor.
- Anything fancier than grid search.

These are deliberately deferred. V1 establishes the plumbing; once we've evaluated the heterogeneous-quant probes and trust the metrics, the learned-cost-model step is a quant-driver patch.

> **Note on uniform quantization (post-2026-05-06 methodology correction).** An earlier version of this README listed a "uniform-quant grid" (Q4_0, Q4_K, Q5_K, Q8_0, etc.) as the V1 probe set. That framing was wrong. Without quantization-aware post-training, there is no reason to expect every "anatomical unit" of an optimized policy to be equally tolerant of the same precision target. PCA over fp16 coefficients reveals heterogeneous variance across parameter groups; uniform quantization ignores this and is not a valid research probe. The V1 probe set is heterogeneous-quant configs only (Unsloth Dynamic, custom mixes), measured against fp16. The uniform-grid materializer in `scripts/03_materialize_grid.sh` is retained as infrastructure for sanity checks but its outputs are not part of the active probe set.

## Quick start

```bash
# 1. Generate FP16 base GGUF from the bf16 safetensors (one-time, ~5 min)
#    via llama.cpp's convert_hf_to_gguf.py
python /Users/mdot/llama.cpp/convert_hf_to_gguf.py \
    /Users/mdot/models/gemma-4-a4b-bf16 \
    --outtype f16 \
    --outfile /Users/mdot/models/gemma-4-a4b-bf16/gemma-4-26B-A4B-it-bf16.gguf

# 2. Run the multi-config benchmarking long-run (full surviving harness
#    battery: MMLU, GSM8K, SVG-MSE, Tok/s — driven via /v1/chat/completions).
#    Restarts the bridge per config; appends results to JSONL.
./server/.venv/bin/python tools/quant_search/scripts/08_long_run.py

# Configurable via env (see the script docstring for the full list):
#   LONG_RUN_RESULTS    output JSONL path (default /tmp/long_run_results.jsonl)
#   LONG_RUN_CONFIGS    comma-separated list of config tags
#   LONG_RUN_HARNESSES  comma-separated list of harness names
#   LONG_RUN_BUDGETS    JSON dict overriding per-harness budgets

# 3. Post-process /tmp/long_run_results.jsonl with jq/python — e.g.
jq -c '{tag: .config_tag, mmlu: .finals.mmlu, gsm8k: .finals.gsm8k, tok_s: .finals.tok_s}' \
    /tmp/long_run_results.jsonl
```

## Oracle taxonomy

| Tier | Oracle | Cost / config | What it measures |
|---|---|---|---|
| 1 (cheap) | perplexity | ~30s | Distribution-level fidelity, coarse |
| 1 | kl_div | ~30s | Per-position divergence vs FP16 oracle |
| 1 | argmax/top-5 overlap | ~30s | Sharper than KL when KL is small |
| 2 (medium) | tok_s | ~5min (3 trials × 3 activeB) | Throughput at multi-stream |
| 2 | structured-output fidelity | ~1-5min | JSON schema / regex / AST |
| 2 | mermaid graph diff | ~1min | Continuous spatial-text grading |
| 3 (expensive) | svg_mse | ~10min (vision-feedback loop) | On-policy spatial reasoning |
| 3 | activation L2 per layer | ~5min | Per-layer attribution of damage |
| 3 | calibration drift | ~5min | Logit-magnitude vs accuracy |
| 4 | qualitative human review | manual | Catches weirdness metrics miss |

Run cheap-tier oracles for coarse pruning. Run expensive-tier only on Pareto-frontier candidates.

## Search strategy roadmap

1. **V1 (now): heterogeneous-quant probes** — fp16 baseline + Unsloth Dynamic (Q4_K MoE up/gate + Q5_1 MoE down + Q8_0 attention/dense + F32 norms) and any other published mixed-precision configs we can drop into a GGUF. Establishes baseline behavioral-divergence frontier, validates oracles. (The earlier "uniform-quant grid" plan is retired; see the methodology correction note above. Uniform-precision configs would only be valid probes for a quantization-aware-trained model, which Gemma-4-a4b is not.)
2. **V2: per-tensor-class local search** — start from V1 frontier, swap one tensor class at a time (e.g., MoE→Q4_K while attention stays Q5_K), measure if Pareto improves. Requires custom-quant materializer (patch to llama-quantize or python-side gguf-py quantization).
3. **V3: learned cost model** — after ~50 configs, fit a regressor predicting (tok_s, behavioral-divergence) from config vector. Use to skip benchmark for configs the model says are clearly Pareto-dominated.
4. **V4: re-run on kernel changes** — when V12-or-successor lands, the cost surface shifts. Re-evaluate the V1 probes + spot-check the V2 frontier.

## Storage schema

`results.sqlite` (default path):

```sql
CREATE TABLE configs (
    config_hash    TEXT PRIMARY KEY,
    config_json    TEXT,        -- the full quant assignment
    kernel_version TEXT,         -- e.g. "v11+v11_q4k_q5_1"
    created_at     REAL
);
CREATE TABLE metrics (
    config_hash  TEXT,
    metric       TEXT,           -- "tok_s", "kl_mean", "svg_mse", ...
    activeB      INTEGER,        -- relevant for tok_s; 0 elsewhere
    value        REAL,
    raw_json     TEXT,           -- full oracle response
    measured_at  REAL,
    PRIMARY KEY(config_hash, metric, activeB)
);
```

Pareto frontier is computed on-demand from this; no separate "frontier" table.

## Open questions

- **Reference image set for SVG MSE**: V1 expects pre-generated reference PNGs. Generate them once via the same toolcard against the FP16 reference model, cache in `test_data/svg_quant_refs/`.
- **Custom per-tensor quant**: requires either patching llama-quantize (~150 LOC of C++) or implementing GGML quant functions in Python (more work, more flexible). V2 decision point.
- **Vision-loop latency**: SVG oracle runtime dominated by the playwright rasterize step (~500ms per iter). Acceptable for V1 (small prompt set); if we scale prompts up, batch the rasterizer.
- **Bimodality at N=8**: tok_s measurements are bimodal (sometimes 130, sometimes 158). Take 3-trial median; report stdev. Configs with high stdev are penalized in Pareto pruning (use median - stdev as the lower bound).

## Why this is hobbyist-relevant

Existing GGUF tooling optimizes the wrong quantization target for the M-series Mac demographic. Phone-deploy framing (smallest model that loads at all) ignores that 64-128GB unified memory makes the *speed* axis the binding constraint. A substrate-aware Pareto explorer captures the actually-relevant tradeoff curve and produces an artifact that generalizes to future engines, future kernel improvements, and future Apple silicon.
