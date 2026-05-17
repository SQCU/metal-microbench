# elicitation — layered signature toolchain for user-agent judgments

Companion to:
- `docs/user_agent_likert_methodology.md` — why we Likert-judge user-agent turns
- `docs/user_agent_workshop_harness_design.md` — what the workshop harness is for

This subdir is the first stage of the workshop-harness pipeline: a
persistent JSONL judgment store + a layered signature estimator that
gracefully scales from "very few samples" to "enough samples for PCA."

## Files

- `axes.py` — single-source-of-truth Likert axis names + legends.
  Mirror of LIKERT_AXES in the user-personas plugin. Keep in sync.
- `probe_persist.py` — runs the two-stage cascade judge against built-in
  persona turn texts and appends each judgment to a JSONL file.
  Idempotent + append-friendly: sample_index continues from existing
  records in the file, so two `--n 30` runs on the same path produce
  60 distinct samples per persona.
- `signature.py` — analysis library. Four layers, sample-size-gated:
  - Layer 1 (n≥2 per axis): per-axis mean / std / median
  - Layer 2 (n≥6): bivariate Pearson correlation matrix
  - Layer 3 (n≥30 total, ≥5/persona): pooled within-persona covariance
    + Mahalanobis distance between persona means
  - Layer 4 (n≥2·N_AXES=28; ≥50 recommended): PCA + effective
    dimensionality at variance thresholds
- `analyze.py` — CLI that loads a JSONL file and prints the layered
  report. Layers that are sample-size-gated out are reported as
  "skipped" rather than silently degraded.

## Typical use

```bash
# Acquire data (N=30 per built-in persona, ~12 min on Q8_0 bridge).
/path/to/server/.venv/bin/python3 probe_persist.py \
    --all --n 30 \
    --output /Users/mdot/metal-microbench/data/elicitation_judgments.jsonl

# Inspect.
/path/to/server/.venv/bin/python3 analyze.py \
    /Users/mdot/metal-microbench/data/elicitation_judgments.jsonl
```

Re-running `probe_persist.py` against the same `--output` appends more
samples; `analyze.py` will incorporate them automatically.

## Why the venv path

`signature.py` needs numpy. The metal-microbench server venv has it.
The system python3 does not. Always invoke with
`/Users/mdot/metal-microbench/server/.venv/bin/python3` until we add a
dedicated venv for this subdir.

## What goes in `meta`

The JSONL schema reserves a `meta` field for the workshop loop to
attach round/target-shift metadata once it exists. For pilot probes
this field is `{}`.

## Layer-skip rules (sample-size gates)

The gates in `signature.py` are intentional, not negotiable:

| layer | gate | rationale |
|---|---|---|
| univariate | n ≥ 2 | std needs ≥2 points |
| correlation | n ≥ 6 | rule-of-thumb minimum for meaningful Pearson r |
| covariance | n ≥ 30 total AND ≥ 5 per persona | each persona contributes a centered block; pooled within-class df = N − K |
| PCA | n ≥ 2 × N_AXES = 28 | bare minimum for a non-degenerate sample covariance in 14-d; ≥100 recommended for stable PC directions |

A consumer that asks for layer N and gets `None` should not fall back
silently to layer N−1 — it should fail loudly or surface the
sample-size shortage to the user. Silent degradation is how
methodology debt accumulates.
