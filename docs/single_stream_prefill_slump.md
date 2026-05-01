# Single-stream prefill slump: diagnosis + falsified levers

## TL;DR

After the simdgroup-matmul prefill rewire (commit `bcfa0fd`, 2026-04-30), the
multi-stream operating point hits ~1.4 sec for 2048 tokens (B=8 streams ×
qLen=256), ≈1422 tok/s aggregate. The single-stream operating point (one user,
one prompt) hits ~85–195 tok/s depending on prompt length. That single-stream
gap is structural to running an 8-slot-batched engine at 1 active slot. Two
substrate-level levers were investigated as ways to close it; **neither
delivered**. This doc captures the falsifications so they aren't repeated.

Multi-user batching already aggregates correctly via
`lm_engine.swift::stepMultiSlotPrefill` — when 2+ sessions are priming, the
scheduler packs them into a B≥2 prefill CB. The rest of this doc is purely
about the case where exactly one user is in flight.

## Measured baseline (2026-05-01, M5 Max, Gemma-4-A4B Q4_K_M, MAX_Q_LEN=256)

Cold-prefill rates (random nonces in each prompt to defeat the prefix cache):

| prompt_tokens | wall (s) | rate (tok/s) |
|---|---|---|
| 330 | 2.19 | 195 |
| 579 | 3.83 | 174 |
| 1082 | 8.24 | 140 |
| 2082 | 19.28 | 111 |
| 4131 | 48.90 | 85 |

Per-token cost grows monotonically with prompt size — that's the attention
quadratic (5 of 30 layers are full-attention with no sliding window; per-token
attention cost scales linearly with prompt length, total scales O(N²)).

## Lever A — NR1 specialization for low-batch prefill (FALSIFIED, twice)

**Hypothesis**: at single-stream prefill chunk (numVecs=256), the canonical
matmul kernel (NR1=32) leaves only 8 X-TGs per output-col tile, while the
saturated case (numVecs=1024) has 32 X-TGs per tile. More X-TGs at the same
gridY co-read W from L2 → higher TFLOPS. A kernel with NR1=8 would have 4× more
X-TGs at numVecs=256, recovering the L2-reuse density.

**Falsification round 1** (commit `ea03163`): minimum-change NR1=8 kernel
(just `NR1=32` → `NR1=8` in the same body, keeping the canonical 2×2 col×row
simdgroup partition). Plateaus at ~3.5 TFLOPS regardless of numVecs because
half the `mc` tiles compute against duplicate-clamped sb data and get
discarded by the cooperative output store. The L2-reuse mechanism is real in
principle but the implementation has 50% wasted simdgroup-matmul FMA work.

**Falsification round 2** (this iteration): proper row-partition NR1=8
kernel where all 4 simdgroups produce useful `mc` output (each handles a
16-row stripe of NR0=64, ma[2] × mb[1] = 2 mc useful per simdgroup, 8 mc
useful per TG total). Numerically correct (RMSE 0.0004, fp16 floor) and 2.7×
faster than the minimum-change variant — peaks at 9.65 TFLOPS at numVecs=1024
versus 3.5 plateau. **But still loses to NR1=32 at every operating point**:

| numVecs | NR1=32 | NR1=8 row-part | Δ |
|---|---|---|---|
| 32 | 2.78 | 2.34 | -16% |
| 64 | 4.59 | 3.74 | -19% |
| 128 | 4.82 | 4.63 | -4% |
| 256 | 8.62 | 6.99 | -19% |
| 512 | 11.13 | 9.52 | -14% |
| 1024 | 13.69 | 9.65 | -29% |

The L2-reuse hypothesis was right that "more X-TGs per gridY = more L2 sharing
= higher TFLOPS" — that's why the row-partition variant beat the minimum-change
variant by 2.7×. But NR1=32 also benefits from L2 reuse via grid scaling once
numVecs ≥ 256 (gridX = numVecs/32 grows naturally), AND has 4× more useful
compute per TG. The row-partition design's 1/4 mc-per-TG density costs more
than the L2-reuse density gains.

**Conclusion**: NR1=32 is genuinely the best tile for this kernel design across
the entire numVecs range we care about. There is no kernel-zoo dispatch rule
based on NR1 that beats the canonical case. Not just "loser at one operating
point" — falsified across all of them.

## Lever B — MAX_Q_LEN bump (FALSIFIED)

**Hypothesis**: the canonical kernel scales cleanly to ~13.7 TFLOPS at
numVecs=1024. Bumping MAX_Q_LEN from 256 → 1024 lets single-stream chunks
reach the saturated regime. Engine knob only, no kernel work.

**Result**: falsified by bridge end-to-end measurement (2026-05-01). Cold
single-stream prefill rates were within 3% noise of the MAX_Q_LEN=256 baseline
across prompts from 330 to 4131 tokens. The matmul does run faster per-call
(profile-confirmed), but the per-token attention cost grows with chunk size:
`kv_attn` went from 5.5 µs/tok at MAX_Q_LEN=256 to 10.8 µs/tok at MAX_Q_LEN=1024
in the multi-stream profile. Attention's per-token cost growing 2× cancels the
matmul's 25% per-token speedup.

The mathematical reason: chunked prefill processes the same total attention
ops (O(N²/2)) regardless of MAX_Q_LEN partitioning, but the per-chunk attention
KERNEL is less efficient at larger qLen. Net: no end-to-end gain from MAX_Q_LEN
bump alone, plus 4× memory cost on `pre_logits`. Reverted.

## What this leaves

Both clean substrate-level NR1 levers are exhausted, and the row-partition
design that "should have worked" turned out to lose too — definitively closing
the matmul-kernel-zoo angle. The single-stream prefill operating point is
bound by a **mix** of attention quadratic and matmul under-fill, not any
single lever the simdgroup-matmul kernel design exposes.

Remaining substrate-only options for single-user prefill speed (none cheap):

- **Attention kernel zoo at high qLen**: a flex-attention variant tuned for
  qLen=1024+ at B=1, addressing the per-token cost growth. Likely needs
  different block-Q sizing, different threadgroup-memory budget. Real kernel
  work, not a knob.
- **CB consolidation** via norm-fused matmul kernels. 1.1-1.2× modest gain at
  most. Defer.

The bigger architectural truth is: **the engine is correctly designed for
multi-stream**. Single-user is a corner case. The bridge already aggregates
multi-user prefill via `stepMultiSlotPrefill`; if production single-user TTFT
is a real concern, the right answer is more concurrent users (the architectural
intent), not contorting the single-stream path against the architectural grain.

## Out of scope

- Anything that changes attention scope, KV cache contents, or model output
  distribution. The full-attention quadratic at 6k+ contexts is intrinsic to
  the architecture and accepted as the price of correctness.
  (See `feedback_no_destructive_algorithmic_changes.md`.)
- Speculative decoding, cross-chunk pipelining of the same user's prefill
  (causality-violating without speculation), and similar tricks that require
  output-distribution changes.

## Decision deferred

Two clean levers are exhausted, neither paid back. A third (the redesigned
NR1=8 kernel) is real kernel work that should be motivated by concrete pain
before being attempted. Hold the line at MAX_Q_LEN=256 + canonical NR1=32
matmul as the single-stream operating point. If a later workload proves single-
user prefill TTFT is binding, revisit the attention-kernel and row-partition
options above with a fresh round of profiling.
