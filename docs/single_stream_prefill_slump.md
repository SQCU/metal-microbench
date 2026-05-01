# Single-stream prefill slump: diagnosis + substrate-only fix menu

## TL;DR

After the simdgroup-matmul prefill rewire (commit `bcfa0fd`, 2026-04-30), the
multi-stream operating point hits ~1.4 sec for 2048 tokens (B=8 streams ×
qLen=256), ≈1422 tok/s aggregate. The single-stream operating point (one
user, one prompt) hits ≈96–236 tok/s depending on prompt length. That single-stream
slump is structural to running an 8-slot-batched engine at 1 active slot, and
is what this spec is about. Multi-user batching already aggregates correctly
(`lm_engine.swift::stepMultiSlotPrefill`); this is purely the case where exactly
one user is in flight.

## Measured baseline (commit `bcfa0fd`, M5 Max, Gemma-4-A4B Q4_K_M, MAX_Q_LEN=256)

| prompt_tokens | wall (s) | per-token (ms) | rate (tok/s) |
|---|---|---|---|
| 71 | 0.48 | 6.8 | 236 |
| 521 | 3.18 | 6.1 | 173 |
| 1521 | 10.06 | 6.6 | 154 |
| 3021 | 21.56 | 7.1 | 141 |
| 6021 | 62.70 | 10.4 | 96 |

Per-chunk (≈256 tokens each at MAX_Q_LEN=256):

| chunks | per-chunk wall (s) |
|---|---|
| 3 | 1.06 |
| 6 | 1.68 |
| 12 | 1.80 |
| 24 | 2.61 |

Two distinct effects compose:

- **Constant single-stream baseline**: ~1.06 s for one 256-token chunk at 1 active stream.
  Multi-stream profile says the same chunk under B=8 saturated takes 1.44 s for
  2048-token-equivalent batch. Per-token cost: 4.1 ms (single-stream) vs 0.7 ms
  (multi-stream, fully packed). **6× under-utilization at single-stream**.
- **Attention quadratic at long context**: per-chunk wall grows 2.5× from chunk 1
  (~1.06 s) to chunk 24 (~2.61 s). Five of thirty layers are full-attention (no
  sliding window), so each new chunk's queries attend to the full accumulated
  KV history. **This is intrinsic to dense full-attention layers and is not in
  scope for this spec — algorithmic attention changes are forbidden.**

## Where the 6× single-stream gap comes from

Three substrate-level contributors, multiplicative:

1. **Matmul tile under-fill**. The dense Q8_0 simdgroup matmul kernel
   (`prefill_mm_q8_0_swiz`) uses NR1=32 for the slot dimension. At single-stream
   chunk (qLen=256, B=1), each matmul gets `ceil(256/32) = 8` X-axis TGs. At
   B=8 saturated (256 × 8 = 2048 effective batch), the same matmul gets 64
   X-axis TGs. Bench measurement at FFN gate/up shape: `swiz_q8_0` runs at 7.59
   TFLOPS at B=256, 13.69 TFLOPS at B=1024 — about **1.8× directly attributable
   to tile-fill density**.

2. **MoE expert grid under-fill**. `prefill_mm_id_q4K_swiz` and
   `prefill_mm_id_q5_1_swiz` dispatch with `gridZ = E = 128` regardless of how
   many slot-fills exist. At B=8 streams, total routed slots ≈
   `B × MAX_Q_LEN × TOPK = 16384`, giving each expert ~128 slots → fills the
   NR1=32 slot tile cleanly. At B=1 stream, ~2048 routed slots, each expert
   ~16 slots → half-empty tiles. **Estimated 1.5–2× MoE-call lost** to partial
   tiles at single-stream (consistent with the bench's `mm_id_q4K_swiz` numbers
   at B=8 vs B=256).

3. **Per-CB / per-encoder overhead**. ~30 layers × ~10–15 encoders = ~300–450
   encoder boundaries per prefill CB. Each encoder pays a fixed
   `~50–100 µs` setup cost on Apple silicon. That's ~15–45 ms per CB of pure
   overhead. At B=8 saturated this is amortized over 2048 tokens (≈10 µs/tok);
   at B=1 it's amortized over 256 tokens (≈80 µs/tok). **Amounts to a ~70 µs/tok
   single-stream tax** — small but real, contributing maybe 1.1–1.2× to the gap.

Multiplicatively: 1.8 × 1.7 × 1.15 ≈ **3.5×**, leaving ~1.7× unaccounted-for —
likely a mix of activeB-non-aware kernels (some ancillary ops still dispatch
B-wide grids) and the matmul kernel's intra-TG simdgroup count being tuned for
the B=8 case.

## Substrate-only fix menu (to be evaluated; this spec scopes the work, not the
 selection)

### Lever A — NR1 specialization for low-batch prefill (kernel zoo)

Add a low-batch variant of each simdgroup matmul kernel with smaller NR1 (slot
tile). Same matmul body; smaller slot tile fills more cleanly at B=1×qLen=256.
Mirrors the existing AR-decode B_TILE zoo
(`dense_gemv_q8_0_btile_b{1,2,4,8}` and friends). Three new kernels:

- `prefill_mm_q8_0_swiz_nr1_8` (NR1=8, suitable for activeB=1×MAX_Q_LEN=256)
- `prefill_mm_id_q4K_swiz_nr1_8`
- `prefill_mm_id_q5_1_swiz_nr1_8`

Dispatcher picks NR1 from `activeB`:
- `activeB ≥ 8`: use NR1=32 (current)
- `activeB ∈ {2..7}`: use NR1=16 (could specialize further)
- `activeB == 1`: use NR1=8

Estimated win: 1.5–2× on the lever-A axis, recovers most of the tile-fill slump.

Risk: more MSL surface area, more kernel-zoo dispatcher complexity. Mitigation:
the fix-set-of-three is small (production target is fixed Gemma-4 shapes) and
follows the well-trodden B_TILE pattern.

### Lever B — MAX_Q_LEN tuning above 256

The matmul kernel scales cleanly at larger Q-batch. Bench at FFN gate/up:
- B=256: 7.59 TFLOPS
- B=512: 12.00 TFLOPS
- B=1024: 12.71 TFLOPS

Bumping MAX_Q_LEN from 256 to 512 should give ~1.6× per-chunk efficiency at
single-stream (since each chunk now has B=1×512 effective). At MAX_Q_LEN=1024,
~1.7× over MAX_Q_LEN=256. Past 1024 we hit the matmul kernel's roofline.

Cost: scratch buffers (`pre_*` in `bootstrap.swift`) grow proportionally.
Biggest is `pre_logits` at B × MAX_Q_LEN × VOCAB × 2 bytes:
- MAX_Q_LEN=256: 8 × 256 × 262144 × 2 = 1 GB
- MAX_Q_LEN=512: 2 GB
- MAX_Q_LEN=1024: 4 GB

Unified memory is 128 GB so all three fit comfortably. The fast-unembed-gather
path (`unembed_fast` in `profile_prefill.swift`) only writes the last position
per slot, sidestepping the full `pre_logits` buffer most of the time, but the
allocation still has to fit.

Estimated win: 1.5–1.7× across the per-chunk baseline at single-stream.

Risk: low. Engine-side change only, no kernel modifications. KL parity is
mechanical.

### Lever C — CB consolidation (fewer encoder boundaries)

Each `enc.endEncoding()` + new encoder is a fixed ~50–100 µs cost. Some prefill
stages currently use multiple encoders that could be merged:

- QKV is now 1 RMSNorm + 3 matmul = 4 encoders. Could fuse QKV into one matmul
  call writing into a [Q | K | V] concat output buffer.
- Shared FFN gate+up is currently 1 RMSNorm + 2 matmul + gelu_mul + matmul +
  RMSNorm = 6 encoders. Some of these can fuse if the kernel-zoo grows variants
  with norm-pre-fused matmul (similar to the existing
  `dense_gemv_q8_0_v6_rmsnorm` pattern, but on the simdgroup matmul).

Estimated win: 1.1–1.2× across all single-stream prefill (the encoder overhead
is real but small; this is gravy).

Risk: each fused kernel is more code surface and harder to A/B-test in isolation.
Defer until A and B are landed and we know the residual gap.

## Out of scope

- Anything that changes attention scope, KV cache contents, or model output
  distribution. The full-attention quadratic at 6k+ contexts is intrinsic to
  the architecture and accepted as the price of correctness.
  (See `feedback_no_destructive_algorithmic_changes.md`.)
- Speculative decoding, cross-chunk pipelining of the same user's prefill
  (causality-violating without speculation), and similar tricks that require
  output-distribution changes.

## Decision deferred

Not selecting between A/B/C now — wait until the multi-stream path is exercised
in production traffic and the actual single-user TTFT pain is concrete from
real users. When that happens:

1. Pick A first (biggest single-lever win, well-trodden code pattern).
2. If A leaves residual gap, add B (cheap, just engine-side knob).
3. If A+B leaves residual gap, evaluate C.

Expected post-A+B single-stream rate at 256-token chunk: ~400–500 tok/s
(2.5–3× over current 173 tok/s baseline at 521-token prompt). Expected
6021-token prefill: ~25–35 sec (down from current 62.7 sec). Attention
quadratic remains the asymptotic limit at very long context.
