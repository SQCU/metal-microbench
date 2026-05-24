# Control-vectors + the prefix cache

What a cvector operation traces/recovers depends on whether it *measures* or
*applies*, and on whether the positions it touches are running fresh through
the forward pass or being served from a cached prefix.

## Two cvec operations, two contracts

| op | kernel | side-effect on residual | side-effect on K/V | traceable via |
| --- | --- | --- | --- | --- |
| **measure** | `measure_dot_fp16` | none (read-only) | none | `Session.pendingSamples` → `gemma_session_poll_samples_json` |
| **apply** | `add_scaled_cvector_fp16` (AR) / `add_scaled_cvector_prefill_fp16` (prefill) | residual += mag·cvec at layer L | layer L+1's QKV (and every subsequent layer) | `Session.pendingSamples` (records per-token magnitude) + the K/V cache itself |

Both hooks fire at the post-FFN residual site, identically in prefill and AR,
so a prompt submitted whole vs. split-and-resumed produces bit-identical
outputs under the same cvec configuration (verified by
`LM_TEST_CACHE_DIVERGENCE=1`).

## Cache interaction

The content-hash prefix cache keys each 16-token slide page on `hashPage(tokens,
cvecDigest)` where `cvecDigest` is an FNV digest of the *parameters* of every
`ActiveControl` whose envelope intersects that page's position range (layer,
cvecId, shape, attack/decay/sustain/release, peak, polarity, start-offset). See
`computeCvecDigest` in `lm_engine.swift`.

Practical consequences:

### Measurement
- **Never affects cache keys.** `DetectorAttachment` is not part of the digest.
- **Only fires during live forward passes** — a detector at layer L writes an
  intensity scalar via `measure_dot_fp16` into `gIntensityBuf` during each AR
  tick (slot 0, position at which the tick runs) or prefill tile (slot 0, the
  *last* position of that tile). Adopted/cached positions are NOT revisited, so
  measurement misses them.
- **To read residual values at a cached prefix position you must re-prefill it.**
  Force a cache miss (e.g., submit with a unique system-prompt prefix) or
  avoid promotion (no other session has promoted that exact `(tokens, digest)`
  key yet).

### Application
- **Every distinct envelope configuration partitions the cache.** Two sessions
  with identical tokens but different `attack` values (or different layer,
  different cvecId, etc.) get different keys and correctly miss each other's
  pages. Adoption of an unsteered session's pages by a steered session (or
  vice versa) is prevented at the hash level — their digests differ by design.
- **Matching configurations share pages.** A steered session whose envelope
  params hash to the same digest as an earlier steered session *will* adopt
  that session's K/V pages. The intervention is baked into the promoted K/V
  (steering at layer L modifies layer L+1's QKV projection input, which is
  then written to the K/V cache).
- **Adoption is in pairs.** Each 16-token slide page promotes two phys pages
  (slide primary + full-attention sibling) because Gemma-4 uses `PAGE_SLIDE=16`
  for sliding-window layers and `PAGE_FULL=8` for full-attention layers. A
  session that adopts N slide pages gets `2N` phys pages, and the intervention
  is recoverable at full precision across cache boundaries.

## Interaction modes worth knowing

### "Probe before you plan"
A trainer building a reward signal from detector readings over a shared-system-
prompt batch should submit the system prompt **once** (promotes pages), then
for each rollout submit `system + unique_user` with the detector attached.
Detectors fire on each user-prefix token (because the user prefix is always
freshly prefilled — it's past the adopted pages). The system-prefix tokens are
not measured.

### "Gate an intervention on a detector reading"
Supported today via `SessionTrigger` edge-detection (onExceed/onFall). The
trigger evaluator runs between ticks and can `restart()` an effector's
envelope. Because the measurement and the effector restart happen *across*
ticks, there's no in-CB coupling — which also means the earliest the effector
can act on a detection is the NEXT tick's forward pass. For single-token-
latency gating, make both detector and effector fire at the same layer on the
same tick's residual. (The measurement lands in `gIntensityBuf` after the
post-FFN write at the detector's layer, so the effector at a *later* layer in
the same forward pass still happens before the measurement is available to the
CPU — trigger logic is inherently cross-tick.)

### "Steer only the user prefix"
Scope the envelope with `units: .tokens`, `startPos: systemPromptLength`,
appropriate attack/sustain. The digest encodes `startOffset` relative to page
start, so two sessions with the same system prompt but different user prefix
lengths will cleanly partition if their envelopes land at different page-
relative offsets.

### "Replay a steered rollout from cache"
If you saved a `(prompt, cvec config)` pair and want the model's behavior on a
prefix to be bit-identical to when the cache was populated, submit the same
tokens with the same cvec params. Adoption is bit-exact (MSE=0 verified end-
to-end across 30 layers). No staleness risk because `allocFresh` invalidates
both pair members when either gets reclaimed.

## What you CAN'T get back

- **Past-token residual values through adopted pages.** The cache stores K/V,
  not pre-attention residuals. Re-prefill to measure.
- **Effector magnitude at adopted positions.** Recorded per-token in
  `pendingSamples` only for positions that were *freshly* prefilled or AR-
  decoded by this session. Cached positions don't re-record.
- **Cross-session detector traces.** Each session's `pendingSamples` is its
  own; there's no merged timeline. Build one client-side from the SSE
  telemetry frames.

## Diagnostic commands

```bash
# Unit tests on digest + hashPage composition (no model weights needed)
LM_TEST_CVEC_DIGEST=1 ./forward_graph

# Full integration: cvec partitioning + intervention preservation
LM_TEST_CVEC_CACHE=1 GGUF_PATH=... ./forward_graph

# Per-layer MSE dump — unsteered full-prefill vs split-prefill-resume;
# first non-zero layer localizes any future cache-replay regression
LM_TEST_CACHE_DIVERGENCE=1 GGUF_PATH=... ./forward_graph
```

A regression in pair-promotion or in either steering hook will surface in the
divergence dump as the first layer with MSE > 1e-4. A regression in digest
partitioning will surface in the integration test as an unexpected
adoption/miss count. Keep both passing and cvec+cache interactions stay
bit-exact.

## AR-vs-prefill kernel-pair K/V divergence (known-benign invariant)

The engine uses two distinct kernel paths for QKV projection:

- **AR path** — `dense_gemv_q8_0_btile_qkv_otf_b1` and its Q5_K / Q6_K /
  Q4_0 / Q4_1 / Q5_1 / F16 mirrors. Fused RMSNorm + QKV in one dispatch,
  per-thread scalar accumulation, 4-way simdgroup split-K reduction at
  the end. Hand-tuned for activeB ∈ {1, 2, 4, 8} via templated kernels.
- **Prefill path** — `prefill_mm_q8_0_swiz` and friends. Separate
  `encRMSNormG` dispatch + simdgroup-matmul tile (NR0=64 × NR1=32 ×
  NK=32) on Apple's `simdgroup_multiply_accumulate` hardware primitive
  (8×8 fp32 accumulators).

These two paths produce **~9% nDiff fp16 K/V on bit-identical inputs**
by reduction-order alone — measured via `LM_KV_OVERWRITE_CHECK=1` after
the `.primed` recover-step landed. The divergence is fundamental
(reduction-order-dependent fp16 accumulation, not a bug) and the
`test_KL_adopted_vs_fresh_first_token` harness confirms it's
statistically equivalent at the sampling-distribution level.

`.primed` recover-step routes through the **prefill** kernel at qLen=1
so single-page prompts (≤16 tokens) get bit-identical producer/consumer
K/V. Multi-page prompts still have residual drift because the prefill
kernel's simdgroup MMA output-tile position is qLen-dependent
(producer's row 31 in a qLen=32 tile vs consumer's row 0 in a qLen=1
tile uses different reduction orders within the same kernel).

A unified-kernel rewrite — making the AR kernel use simdgroup MMA — was
**scoped 2026-05-23 and rejected**. The simdgroup tile geometry is
~3% utilized at activeB=1 (NR1=32 → 1 row), so the rewrite would
regress AR decode speed for fp-rounding-magnitude correctness gain.
Existing tests (KL guard + per-layer divergence dump) catch any actual
sampling-distribution regression.
