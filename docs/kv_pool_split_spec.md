# KV-pool split: 2-level address arithmetic to lift the 8192-page cap

**Status**: Spec — no kernels touched yet. Estimated half-day implementation if
the assumptions below hold.

**Author / context**: 2026-05-07. Followup to the Session-deletion / atomic-
construction refactor (commit f6cfedd). The configuration audit
(commit 2f730db) flagged this as the one remaining structural limit blocking
multi-session-at-64k-context workloads.

## Problem

Apple Metal has an unidentified per-MTLBuffer or per-resource constraint
between **528 MB** (pool=8192, working) and **784 MB** (pool=12288, wedges
on first GPU dispatch). Documented at `bootstrap.swift:531-540`. Symptom:
allocation succeeds, kernel arguments encode fine, first dispatch hangs
with no error returned.

This caps the engine at **one session × 64k context, OR many sessions ×
shorter contexts** — but not both. With `MAX_PAGES_PER_SLOT=8192` and a
shared `TOTAL_PAGES=8192` pool, the moment session A holds all 8000 full-
attention pages for its 64k tokens, session B has zero pages to allocate.

## The half-day insight

Don't try to debug the Metal cliff. **Split each layer's K (and V) cache
into N_SUBBUF smaller MTLBuffers, each comfortably below the cliff, and do
2-level addressing in the kernels.** Same total KV memory; just distributed
across more buffer objects.

If we choose `PAGES_PER_SUBBUF` to be a **power of 2**, the address
decomposition is free on GPU:

```c
uint phys      = block_table[...];                    // unchanged: 0..TOTAL_PAGES-1
uint subbuf    = phys >> PAGES_PER_SUBBUF_LOG2;       // shift, single cycle
uint local     = phys &  (PAGES_PER_SUBBUF - 1);      // mask, single cycle
device half* K = K_subbuffers[subbuf]                 // indexed pointer load
                 + ((local * PAGE + off) * H + h) * D;
```

The block-table format **does not change**. Phys page IDs are still 0..
`TOTAL_PAGES-1`. The page manager is unchanged. Only the kernel read site
(and its CPU-side counterpart in lm_engine.swift) gain one indirection.

## Concrete shape

| Constant | Value | Rationale |
|---|---|---|
| `PAGES_PER_SUBBUF` | 4096 | Power of 2; gives 264 MB / slide-layer-K-buffer (50% of cliff) |
| `PAGES_PER_SUBBUF_LOG2` | 12 | bit-shift constant |
| `N_SUBBUF` | `TOTAL_PAGES / PAGES_PER_SUBBUF` | 2 at pool=8192, 8 at pool=32768 |
| `TOTAL_PAGES` (target) | 32768 (8× headroom over today's 8192) | 4× concurrent 64k-context sessions |

Per-layer slide K-cache size at pool=32768: 8 sub-buffers × 264 MB = 2112 MB
total (same memory as one buffer, just split). Per slide layer K+V = 4224 MB.
Across 25 slide layers + 5 full layers: **~26 GB → ~110 GB** at full
pool=32768. Comfortable on a 128 GB M5 alongside 10 GB Q4_K_M weights and
the rest of the working set.

## Binding model decision: flat args, not argument buffers

The project doesn't currently use Metal argument buffers
(`MTLArgumentEncoder`). We'd need to either set one up, or pass N_SUBBUF
separate `[[buffer(N)]]` arguments per kernel.

**Recommendation: flat args.** With `N_SUBBUF ≤ 8` it fits comfortably in
Metal's per-kernel binding budget (~31 buffers; the worst-loaded kernel
today binds ~8). Avoids `MTLArgumentEncoder` setup, which is its own
debugging adventure on a half-day budget.

Kernel signatures change like:

```c
// before
kernel void slide_attn_ar_v6(
    ...,
    device half*       K_cache       [[buffer(2)]],
    device half*       V_cache       [[buffer(3)]],
    ...);

// after (N_SUBBUF=8 example)
kernel void slide_attn_ar_v6(
    ...,
    device half*       K_subbuf_0    [[buffer(2)]],
    device half*       K_subbuf_1    [[buffer(3)]],
    device half*       K_subbuf_2    [[buffer(4)]],
    ... K_subbuf_3..K_subbuf_7 ...,
    device half*       V_subbuf_0    [[buffer(10)]],
    ... V_subbuf_1..V_subbuf_7 ...,
    ...);

// at the read site, replace:
//   device half* Kd = K_cache + offset_expr;
// with:
device half* Kd;
uint subbuf = phys >> PAGES_PER_SUBBUF_LOG2;
uint local  = phys &  (PAGES_PER_SUBBUF - 1);
uint inner  = ((local * PAGE + off) * H + h) * D;
switch (subbuf) {
    case 0: Kd = K_subbuf_0 + inner; break;
    case 1: Kd = K_subbuf_1 + inner; break;
    case 2: Kd = K_subbuf_2 + inner; break;
    ...
    case 7: Kd = K_subbuf_7 + inner; break;
}
```

The switch compiles to a small jump table on Apple GPUs. Branch divergence
is bounded (within a SIMD group, all lanes typically read pages in the
same sub-buffer because consecutive logical pages map to consecutive phys
pages most of the time — confirmed by inspecting page allocation order).

**Alternative (deferred)**: if `N_SUBBUF > 8` ever needs to hold (pool >
32768), graduate to argument buffers. Single-call refactor, probably
another half day.

## Change list — CPU side

### `bootstrap.swift`

1. **Add constants** near `TOTAL_PAGES`:
   ```swift
   let PAGES_PER_SUBBUF      = 4096
   let PAGES_PER_SUBBUF_LOG2 = 12
   let N_SUBBUF              = (TOTAL_PAGES + PAGES_PER_SUBBUF - 1)
                                / PAGES_PER_SUBBUF
   // assert TOTAL_PAGES % PAGES_PER_SUBBUF == 0  (or equivalent guard)
   ```

2. **`LmWeights.K_caches`** type change:
   ```swift
   let K_caches: [[MTLBuffer]]       // [layer][subbuf]
   let V_caches: [[MTLBuffer]]
   ```

3. **Allocation loop** (`bootstrap.swift:5767-5781`): inner loop over
   `subbuf in 0..<N_SUBBUF`, each `device.makeBuffer(length:
   PAGES_PER_SUBBUF * pg * KV_H * HD * 2)`. Total memory unchanged;
   just split into N_SUBBUF buffers.

4. **Kernel arg-encoding sites** (`bootstrap.swift:4218, 4632, 4960,
   5163`): each one currently does `enc.setBuffer(Kc, ...)` / similar.
   Loop over sub-buffers and bind to consecutive buffer indices.
   Helper function:
   ```swift
   func encodeKVCaches(_ enc: MTLComputeCommandEncoder,
                        K_subbufs: [MTLBuffer], V_subbufs: [MTLBuffer],
                        kBaseIdx: Int, vBaseIdx: Int) {
       for i in 0..<N_SUBBUF { enc.setBuffer(K_subbufs[i], offset: 0,
                                              index: kBaseIdx + i) }
       for i in 0..<N_SUBBUF { enc.setBuffer(V_subbufs[i], offset: 0,
                                              index: vBaseIdx + i) }
   }
   ```

### `lm_engine.swift`

5. **CPU-side KV access** (`lm_engine.swift:1245, 1297`): the zero-fill
   and KV-staging copies index by physical page. Replace single-buffer
   `K_caches[L].contents()` with sub-buffer-aware lookup:
   ```swift
   let subbuf = phys / PAGES_PER_SUBBUF
   let local  = phys % PAGES_PER_SUBBUF
   let dst    = weights.K_caches[L][subbuf].contents()
                  .advanced(by: local * pg * KV_H * HD * 2 + ...)
   ```

### `page_manager.swift`

6. **No changes**. Pages are still 0..TOTAL_PAGES-1 logical IDs; the sub-
   buffer mapping is purely a CPU-and-GPU access concern.

## Change list — GPU side (kernels.swift)

The 10 K/V argument-binding sites:

| Line | Kernel | Op |
|---|---|---|
| 607-622 | `kv_write_slide` | write K and V for prefill tile |
| 673-684 | `kv_write_full` | write K and V for prefill tile |
| 763-780 | `attn_prefill_v3` | read K and V |
| 9929-10056 | `slide_attn_ar_v6` | read K and V |
| 10177-10279 | `full_attn_ar_v6` | read K and V |
| 10408+ | (additional AR variant) | read K and V |

Each gets the same transform:

1. Replace `device half* K_cache [[buffer(N)]]` with N_SUBBUF separate
   `device half* K_subbuf_i [[buffer(N+i)]]` bindings.
2. Same for V_cache.
3. At the read/write site, replace the single `phys`-based offset with
   the `subbuf = phys >> 12; local = phys & 0xfff` decomposition + a
   switch on `subbuf` to pick the right sub-buffer.
4. Bump downstream `[[buffer(N)]]` indices by `2 * (N_SUBBUF - 1)` to
   make room.

## Validation strategy

The change is **bisectable at N_SUBBUF=1** (= today's behavior, just
gone through one extra indirection). Land in this order:

1. Land all CPU + GPU edits at `TOTAL_PAGES=8192, N_SUBBUF=1`. Run
   `python3 server/test_batch_ffi.py` — must produce **bit-identical**
   completion tokens to the pre-refactor build (greedy temp=0.0, same
   seed). This proves the address transform is mathematically equivalent.

2. Bump `TOTAL_PAGES=16384, N_SUBBUF=2`. Run the smoke test again — same
   tokens. Run `python3 server/probe_oversubscription.py` — same M=8
   throughput (~158 t/s). If both pass, the cliff is confirmed buffer-size
   bound.

3. Bump `TOTAL_PAGES=32768, N_SUBBUF=8`. Submit two sessions × 64k context
   simultaneously through `server/test_batch_ffi.py` extended. Should
   complete without pool exhaustion.

If step 1 produces non-bit-identical output, the bug is in the address
arithmetic — single point of failure, easy to debug.
If step 2 wedges, the cliff isn't per-buffer — open question, see below.

## Open questions to verify before starting

1. **Is the cliff really per-MTLBuffer, not per-resource-binding-set or per-
   command-buffer?** A 30-minute pre-flight: write a Swift script that
   allocates 8 separate 256-MB buffers (= 2 GB total) and does a single
   compute dispatch reading from one of them. If that wedges, the split
   doesn't help and the spec is wrong. If it works, proceed.

2. **Branch divergence cost of the switch?** Spot-check by running the
   N_SUBBUF=1 build under `MTLCaptureScope` at a profiling point. If
   the AR step time at N_SUBBUF=1 is meaningfully slower than the pre-
   refactor build (>2% degradation on the probe_oversubscription.py
   M=8 number), reconsider the binding model — argument buffer with a
   single indexed pointer-load might be cheaper than the switch.

3. **Argument count budget**: at N_SUBBUF=8, kernels gain 16 buffer args
   (8 K + 8 V). Today's hottest kernel (`slide_attn_ar_v6`) binds ~10
   args. 10 + 16 = 26 → fits in Metal's 31-arg budget. At N_SUBBUF=16
   (pool=65536), we'd be at 42 args → exceeds budget → graduate to
   argument buffers. So N_SUBBUF=8 is the practical max for the flat-arg
   approach. Pool can grow to ~32768 within this budget; beyond that,
   argument-buffer refactor is needed.

## Time budget

| Step | Estimate |
|---|---|
| Pre-flight: confirm cliff is per-buffer (multi-buffer dispatch test) | 30 min |
| Constants + LmWeights type change + allocation loop (bootstrap.swift) | 45 min |
| 6 kernel arg-encoding sites + helper function | 45 min |
| 10 kernel signature updates + 10 read-site transforms (kernels.swift) | 90 min |
| CPU-side KV access transform (lm_engine.swift) | 30 min |
| Validation step 1: bit-identical at N_SUBBUF=1 | 30 min |
| Validation step 2: throughput at N_SUBBUF=2 | 15 min |
| Validation step 3: 2 × 64k sessions at N_SUBBUF=8 | 30 min |
| Slack / unexpected | 60 min |
| **Total** | **~6 hours** |

Half a day if the pre-flight confirms the cliff is per-buffer and the
read-site transform produces bit-identical output at N_SUBBUF=1.

## What this enables

- **2 to 4 concurrent sessions × 64k context** at N_SUBBUF=8 / pool=32768.
- The agent-clique evaluator workloads that today silently fail or
  truncate when total pool demand exceeds 8192 pages.
- A clean path forward: if N_SUBBUF=8 isn't enough, graduate to argument
  buffers and grow pool to whatever fits in device memory. The 2-level
  arithmetic infrastructure is already in place; only the binding model
  changes.

## What this does NOT change

- Kernel performance characteristics: address arithmetic adds 2 cycles
  (shift + mask) per page access. Negligible vs the actual K/V load.
- The page_manager.swift API or behavior — pages are still anonymous
  refcounted slots, addressed by 0..TOTAL_PAGES-1 IDs.
- The block_table wire format — kernels still receive `uint*
  block_table` indexed by `[slot * MAX_PAGES_PER_SLOT + p]`.
- Anything in lm_engine.swift's scheduling, admission, or per-CB picker
  logic.
- The bridge or FFI ABI.

## Followup work (out of scope)

- Argument-buffer refactor (lifts the N_SUBBUF=8 ceiling).
- Splitting slide-cache and full-cache into separate pools (today they
  share TOTAL_PAGES; full-attention-only sessions waste slide budget).
- Per-session pool reservation / fair-share admission (today first-come-
  first-served can starve).
