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

## Lever A — NR1 specialization for low-batch prefill (FALSIFIED)

**Hypothesis**: at single-stream prefill chunk (numVecs=256), the canonical
matmul kernel (NR1=32) leaves only 8 X-TGs per output-col tile, while the
saturated case (numVecs=1024) has 32 X-TGs per tile. More X-TGs at the same
gridY co-read W from L2 → higher TFLOPS. A kernel with NR1=8 would have 4× more
X-TGs at numVecs=256, recovering the L2-reuse density.

**Result**: falsified by `q4k_mma_bench` measurement (commit `ea03163` — the
same commit that fixed a separate dispatch-grid axis bug in the bench). NR1=32
canonical scales 2.44 → 13.72 TFLOPS as numVecs grows 32 → 1024. NR1=8 as a
minimum-change kernel (just `NR1=32` → `NR1=8` in the same body) plateaus at
~3.5 TFLOPS regardless of numVecs because half the `mc` tiles compute against
duplicate-clamped sb data and get discarded by the cooperative output store.
The L2-reuse mechanism is real in principle but the implementation has 50%
wasted simdgroup-matmul FMA work, which more than offsets the L2 win.

A "real" NR1=8 kernel would need to redesign the simdgroup row-partition so
all 4 simdgroups produce useful `mc` tiles at NR1=8. A custom version was
attempted in the same investigation; failed correctness (RMSE 1.09 on tiny
shape) and was abandoned. Restarting that work isn't cheap — the row-partition
redesign means a different thread-mapping + sa/sb layout + output-store
pattern. **Not justified by current data.**

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

The single-stream prefill operating point is bound by a **mix** of attention
quadratic and matmul under-fill, not by any single lever the substrate exposes.
Closing it without algorithmic changes (forbidden — see
`feedback_no_destructive_algorithmic_changes.md`) requires a more invasive
redesign:

- **Attention kernel zoo at high qLen**: a flex-attention variant tuned for
  qLen=1024+ at B=1, addressing the per-token cost growth. Likely needs
  different block-Q sizing, different threadgroup-memory budget. Real kernel
  work, not a knob.
- **Properly-row-partitioned NR1=8 simdgroup matmul** (the design that the
  minimum-change variant approximated and lost on). A row-partition that keeps
  all 4 simdgroups productive at NR1=8 would deliver the L2-reuse win without
  the wasted-mc penalty. The simdgroup_matrix register layout makes this
  non-trivial; the first-cut attempt failed correctness.
- **CB consolidation**: lever C from the prior version of this doc
  (norm-fused matmul kernels). 1.1-1.2× modest gain at most. Defer.

The bigger architectural truth is: **the engine is correctly designed for
multi-stream**. Single-user is a corner case. The bridge already aggregates
multi-user prefill via `stepMultiSlotPrefill`; if production single-user TTFT
is a real concern, the right answer is more concurrent users, not making the
single-stream path beat the architectural grain.

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
