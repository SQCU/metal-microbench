# Engineering memo: why split-K became the default low-batch dense GEMV kernel

**Audience:** kernel team, performance reviewer, future-self.
**Status:** decision landed; this is the writeup, not a proposal.

## Setup

The Gemma-4-A4B forward pass is dominated by dense GEMVs at low batch
(B=1..4) where the model is bandwidth-bound on the weight stream, not
compute-bound on activations. On Apple M5 Max, the simgroup-layout
choice for these GEMVs swung performance by 2-4× across plausible
implementations. Three candidates were measured head-to-head:

1. **Single-SG K-unroll**: one simdgroup per output row; the SG sweeps
   the K dimension end-to-end with manual 8-way unrolling.
2. **Multi-SG split-K**: 4 simdgroups per output row; each SG handles
   a contiguous K-quarter, with a final reduction across SGs at the
   threadgroup boundary.
3. **Tiled register-blocked**: B-row tile per SG, register-resident
   accumulators, structured weight prefetch.

## What the numbers said

At Gemma-4 dense shapes (D_in=2304, D_out=2304 for QKV; D_in=2304,
D_out=11008 for FFN gate/up; D_in=11008, D_out=2304 for FFN down), the
ranking on M5 Max was **split-K > K-unroll > register-tiled** at every
batch in {1, 2, 4, 8}. Margin against K-unroll was 2.1× at B=1 narrowing
to 1.4× at B=8. Register-tiled lost everywhere; profiler showed local-
memory spill from dynamic-bound array indexing.

## Why split-K wins on this silicon

Apple GPUs expose limited public async-copy primitives — every
threadgroup_barrier is a fully-stalled GPU sync we cannot hide. The
single-SG variant uses no barriers and naively looks attractive, but
its K-loop is completely serialized: the SG cannot make forward
progress on the next K-tile while the prior tile's MAD is in flight.
Multi-SG split-K parallelizes K across 4 SGs, each running its own
unrolled inner loop. The reduction at the end pays one barrier, but
that's amortized over 4× the K-throughput. The win is purely from
keeping more arithmetic in flight simultaneously, not from a smarter
memory pattern.

## What this rules out

The register-blocked variant is a dead end on Apple silicon for
GEMV-shaped (B small) ops. The kernel's computation pattern was sound;
the loss came from the compiler spilling B-indexed accumulators to
threadgroup memory under register pressure. We've seen this twice now
in different shapes; treat it as a hardware fact.

## What this opens up

With split-K as the default, per-shape specialization is justified:
the FFN-down shape (D_in=11008 → D_out=2304) has 4× the K-extent of
QKV, so split-K's parallelism wins are correspondingly larger there.
The kernel zoo can ship a B_TILE × shape matrix of variants; the
dispatcher picks at runtime based on activeB. Existing zoo entries
already cover B ∈ {1, 2, 4, 8} for dense_gemv_q8_0 and showed 5.5× at
B=1 over the previous V6 baseline.

## Caveats worth recording

- These results are M5-specific. M3 / M2 may rank differently because
  the SG-count-per-CU and barrier cost differ. We have not re-measured.
- Quantization format affects this: Q4_K_M's dequant cost is
  threadgroup-memory-resident, which can crowd out split-K's reduction
  scratch on smaller D_out shapes. We've observed this on attn_out
  specifically; the workaround was a tighter SG count there.
- The reduction barrier has a measurable absolute floor (~1 μs on M5).
  At very small N, that floor dominates and single-SG wins. We have
  not characterized the crossover; below D_in×D_out ≈ 1M elements it's
  worth re-measuring.
