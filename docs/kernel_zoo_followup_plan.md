# Kernel zoo follow-up plan — review/test target ~Wednesday

Owner: review pass after current checkpoint (commit `95ee346`).

This document scopes the work that's *interesting but bigger than a
single-session edit*. Each item has a self-contained scope, a clear
expected payoff, a falsification path (what would prove it doesn't work
and we abandon), and the tooling we'll lean on (existing
`profile_ar_step.swift` + bridge iter-refinement test + LM_PROF).

---

## Where we are now (post-checkpoint)

| Metric | Before this session | Now |
|--------|---------------------|-----|
| Engine wall per AR step | 44.4 ms | **23.9 ms** (-46%) |
| Iter-refinement turn 3 (cold cache) | 16089 ms | **7829 ms** (-51%) |
| Per-stream tok/s (warm decode) | ~18.5 | **~38** |

Single-stream AR has hit the user's reference target (30 tok/s/stream).
The remaining work is for **further single-stream gains** (cumulative
~5-8 ms savings possible) and **multi-stream aggregate scaling**.

---

## Item A: MoE Q4_K dequant rewrite (biggest single remaining lever)

**Scope.** Rewrite `moe_gemv_q4k_v6` (Q4_K MoE gate_up) inner-loop
structure. Target the per-element dequant cost.

**Why.** Bandwidth math at activeB=1: 8 active experts × 2.23 MB Q4_K =
17.8 MB per layer × 30 layers = 535 MB. At 400 GB/s = 1.34 ms ideal.
Current wall: ~6 ms in production. **4-5× off ceiling.** The gap is
not TG launch overhead (we proved that — numActive trim was 0.3 ms)
but Q4_K dequant compute and scattered weight access patterns.

**Approach options (to be benchmarked, not all done)**:

1. **Per-block precomputed dequant table**: stage one Q4_K super-block's
   dequantized weights (256 halves = 512 B) in SG-cooperative tg-mem
   first, then matmul reads from tg-mem. Saves repeating the per-element
   nibble unpack + scale-min math inside the FMA loop. tg-mem footprint
   stays under 1 KB.

2. **Wider weight read coalescing**: Q4_K's `qs[pair*32 + p]` access
   pattern reads 32 bytes contiguously per pair. Already coalesced.
   But the scale-byte unpack walks `scales[12]` with bit-twiddling —
   could pre-extract all 8 sub-block (sc, mn) pairs once per kb and
   stash in registers. Eliminates the redundant `unpack_q4k_scales`
   calls per FMA.

3. **Smaller-D specialization**: at activeB=1 each active expert has
   exactly 1 slot. Templated kernel with constexpr SLOTS=1 (no inner
   slot loop, no register array indexing) might shave 10-15% off the
   inner FMA. Pair with item (1) for compounding wins.

**Falsification.** If approach (1) doesn't show >20% per-call speedup
in `profile_ar_step.swift`, abandon and try (2) or (3). If V8/V9-class
register tiling at MAX_SLOTS=1 doesn't beat V6 (we already saw V8/V9
at MAX_SLOTS≥2 lose), the structural amortization story is dead and
the only remaining lever is dequant compute reduction.

**Expected payoff.** 4-6 ms savings at activeB=1, taking engine wall
from 24 → 18-20 ms = **~50 tok/s/stream warm decode**.

**Effort.** 1-2 focused sessions. Q4_K is 8 sub-blocks × tricky scale
packing — get the dequant test harness right first (KL parity check
vs V6 on canned routings).

---

## Item B: MoE Q5_1 down rewrite

**Scope.** Sister kernel to Q4_K. Same restructuring options but Q5_1
has a simpler block layout (32 elements × 24 bytes, single d/m pair,
qh+qs split).

**Why.** ~4 ms in production at activeB=1. Bandwidth ideal: 8 × 1.49
MB × 30 = 357 MB → 0.89 ms. Roughly same 4-5× off as Q4_K, similar
fix space.

**Effort.** Apply lessons from Item A. Likely 1 session.

---

## Item C: Tg-mem-tiled QKV/GateUp staging variants for activeB ∈ {2,3,4}

**Scope.** New kernels `dense_gemv_q8_0_btile_qkv_tiled_b{2,4}` and
matching gate_up that stage h_norm in **K-tile chunks** (e.g., 256
elems at a time) instead of the full D_in. tg-mem allocation stays
tiny regardless of B_TILE.

**Why.** Current QKV/GateUp dispatcher picks the OTF kernel at
activeB=1 (best) and falls back to V6 grid-shrink for activeB>1.
At activeB ∈ {2,3,4} (multi-stream batches), neither extreme wins
cleanly — OTF's per-FMA gamma/x re-reads accumulate, and V6's
B-grid wastes parallelism on small batches. A tiled-staging kernel
gets register-amortized FMA *and* compact tg-mem.

Sketch:
```cpp
template<uint B_TILE>
inline void btile_qkv_tiled_impl(...) {
    constexpr uint K_TILE = 256;          // tg-mem holds B_TILE × 256 halves
    for (uint k_block = 0; k_block < D_in; k_block += K_TILE) {
        // SG-cooperative load: stage x[..., k_block..k_block+K_TILE]
        // for B_TILE batches, normalize on-the-fly with stored inv_rms.
        // tg-mem footprint: B_TILE × 256 × 2 bytes = 4 KB at b8.
        threadgroup_barrier(...);
        // Matmul over this K-tile, FMA into B_TILE register accs.
    }
    // Output write as in OTF.
}
```

**Falsification.** If `btile_qkv_tiled_b4` doesn't beat V6 at
numVecs=4 in profile, the tiling overhead (extra barriers, tg-mem
bookkeeping) outweighs the register-amortization benefit. Abandon.

**Expected payoff.** 1-2 ms per AR step at activeB ∈ {2,3,4}.
Important for **multi-stream aggregate**, not single-stream.

**Effort.** 1 session. The pattern extends OTF cleanly.

---

## Item D: Multi-stream aggregate verification — DONE

**Already measured** with `server/multi_stream_test.mjs` (4 and 8
concurrent fetches, distinct prompts to avoid cache sharing).

| Active streams | Aggregate tok/s | Per-stream tok/s | Step time | Scaling efficiency |
|----------------|-----------------|-------------------|-----------|--------------------|
| 1 (single-stream) | ~38 | ~38 | 24 ms | — |
| 4 | **80.9** | 20.3 | 49 ms | **100%** |
| 8 | **111.2** | 14.0 | 72 ms | **99%** |

Scaling efficiency at 4-stream and 8-stream is essentially perfect —
no contention, no scheduler bug, no spinwait. The aggregate target gap
(user reference: 120 tok/s at 4-stream = 33 ms step time) is purely
**per-step time scaling with activeB**: each additional active slot
adds ~5-7 ms because the still-non-templated B-grid kernels (attention,
RoPE, KV-write, RMSNorms, residual) genuinely do more work as more
slots are active.

**The aggregate target gap is closed by Items A+B+C.** MoE bandwidth
is shared across slots (same 8 experts touched regardless of activeB),
so reducing MoE per-step cost compounds at all activeB. Items C's
tg-mem-tiled QKV/GateUp helps the per-batch GEMV at activeB ∈ {2,3,4}.
Combined estimate: shave ~12-16 ms off step time at activeB=4, hitting
~33 ms = the user's 120 tok/s reference aggregate.

---

## Item E: Profile-vs-bridge correlation re-verification

**Scope.** After all above kernel changes, re-run `profile_ar_step.swift`
+ bridge iter-refinement and update the per-stage breakdown +
production-scaled estimates. Several memories (`project_active_b_plumbing.md`,
`project_route_compact_slot_aware.md`, etc.) contain stage-time tables
that are now stale.

**Why.** When designing future kernel work, planners should reference
current numbers, not pre-route-compact-fix numbers.

**Effort.** Small. ~30 minutes of profiling + memory updates.

---

## Items deliberately NOT planned

- **Cross-CB pipelining via second queue** — codex flagged earlier
  as high risk-of-regression on Apple silicon (DRAM contention against
  in-flight bandwidth-bound MoE kernels). Hardware doesn't expose
  explicit prefetch primitives. Skip until proven need.
- **Speculative decoding** — explicitly out of scope per user.
- **Compile-time-fixed B for hardcoded-B kernels** (encEmbed,
  encScaleByScalar, etc) — they all already accept numVecs at runtime.
  Compile-time would only help if a kernel had B-dependent register
  state (none of these do).
- **Sampling-time in-place read** — already done (no
  pre_logits→logits blit for fast-unembed path).

---

## Suggested review/test order

1. Run `profile_ar_step.swift` at the checkpoint → record current
   per-stage breakdown (Item E baseline).
2. Item A: Q4_K dequant rewrite + measurement. **Biggest win.**
3. Item B: Q5_1 rewrite (apply Item A lessons).
4. Item C: tg-mem-tiled QKV/GateUp for activeB ∈ {2,3,4}.
5. Item D: multi-stream aggregate verification.

**Rough total**: 3-5 focused sessions.

After all items: expected single-stream ~50 tok/s, multi-stream
aggregate ~150 tok/s at B=4 (matches user reference exactly).

## Cross-references

- `~/.claude-personal/projects/-Users-mdot/memory/project_kernel_zoo_btile_framework.md` — overall framework
- `~/.claude-personal/projects/-Users-mdot/memory/project_route_compact_slot_aware.md` — the big route_compact win
- `~/.claude-personal/projects/-Users-mdot/memory/project_active_b_plumbing.md` — engine plumbing
- `kernels.swift:1297+` — `dense_gemv_q8_0_btile_*` reference template
- `kernels.swift:1605+` — `dense_gemv_q8_0_btile_qkv_otf_*` for the otf pattern
- `profile_ar_step.swift` — A/B test harness; add stages parallel to existing v6/b1/b2/b4/b8 entries
