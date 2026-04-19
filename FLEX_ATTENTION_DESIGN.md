# Tile-streaming flex attention for Gemma-4 on Apple Silicon

Status: design. Supersedes the ad-hoc split-KV kernels in `paged_attn_slide_gqa_compute`
and `paged_attn_full_gqa_compute`. Target: one kernel family that handles decode
(Q_LEN=1), prefill (Q_LEN>1), sliding-window, document-isolation for packed
sequences in one slot, and soft-capped scores — with block-sparse dispatch so
masked-out tiles never run.

## 1. Why

PyTorch `torch.nn.attention.flex_attention` is the current reference for
programmable attention: two user-supplied Python functions (`score_mod`,
`mask_mod`) get inlined into a block-sparse kernel at torch.compile time. Each
(Q_BLOCK, K_BLOCK) tile is classified offline as:

- **full**      — mask_mod is True for every (q, k) in the tile → skip the mask check inside the kernel
- **partial**   — some True, some False → apply mask inside the softmax loop
- **empty**     — all False → don't dispatch the tile at all

We can't JIT arbitrary Python in Metal, but we can build the same three-tier
dispatch pattern with a small enumerated family of mask patterns covering every
attention variant Gemma-4 (and the broader llama.cpp-inspired target set) needs.
What we get in return is a single kernel that replaces five variants today,
handles long contexts correctly (the recent SWA bug), and turns on prefill
throughput (one CB for S prompt tokens instead of S CBs).

## 2. Scope

In scope for v1:

- **Q_LEN ∈ {1, 8, 16, …, 64}** — decode and chunked prefill
- **mask_mod** enum: `causal`, `causal_sliding(window)`, `doc_isolation(doc_ids)`,
  `packed_bitmask(buffer)` — the last one is an escape hatch for anything we
  can't close-form in the others
- **score_mod** enum: `none`, `softcap(cap)` (Gemma-4 has logit soft-cap on lm_head
  but NOT on attention — it's still the right hook for future models)
- **Paged KV**: inherit the current block_table indirection unchanged
- **GQA**: Q_PER_TG = H_Q / H_KV covered by the outer kernel grid
- **Two head-dim presets**: D=256 (slide), D=512 (full). PAGE_K tuned per D
- **Split-K**: optional, ATTN_N_SPLITS controls the K-range split; reduce kernel
  is unchanged

Out of scope for v1 (defer to v2):

- Arbitrary D — initial version specializes via template-like `constexpr`
- Per-head mask (`mask_mod(b, h, q, k)`) with head axis used — easy to add but
  Gemma-4 doesn't need it
- Score_mod with per-head parameters
- Multi-query where Q_LEN crosses page boundaries in a single TG

## 3. API surface

Swift-side:

```swift
struct FlexAttnMaskMod {
    enum Kind { case causal, causalSliding, docIsolation, packedBitmask }
    var kind: Kind
    // Kind-specific parameters. Unused fields ignored.
    var slidingWindow: Int              // causal_sliding
    var docIdBuf: MTLBuffer?            // doc_isolation: [B, MAX_POSITIONS] u32
    var bitmaskBuf: MTLBuffer?          // packed_bitmask: [B, H, Q_BLOCKS, K_BLOCKS, 32] u32 bits
}

struct FlexAttnScoreMod {
    enum Kind { case none, softcap }
    var kind: Kind
    var softcap: Float                  // softcap: active cap value
}

struct FlexAttnBlockMask {
    // Produced by precomputeBlockMask below. Owned by the caller across one or
    // more attention dispatches.
    let qBlocks: Int
    let kBlocks: Int
    // Per-(slot, q_block): list of partially-masked K blocks + list of fully-
    // unmasked K blocks. The kernel processes full blocks faster (no per-k
    // predicate) than partial ones.
    let fullKVIndices: MTLBuffer        // [B, H_Q_groups, Q_blocks, MAX_FULL] u32
    let fullKVCount:   MTLBuffer        // [B, H_Q_groups, Q_blocks] u32
    let partialKVIndices: MTLBuffer     // same shape, partials
    let partialKVCount:   MTLBuffer
    // "Empty" blocks are neither in the full nor partial lists — the kernel
    // simply never sees them, achieving true block-sparse dispatch.
}

func precomputeBlockMask(_ maskMod: FlexAttnMaskMod,
                         qLen: Int, kLen: Int,
                         qBlock: Int = 8, kBlock: Int = PAGE) -> FlexAttnBlockMask

func flexAttention(_ cb: MTLCommandBuffer,
                   Q: MTLBuffer, O: MTLBuffer,
                   Kc: MTLBuffer, Vc: MTLBuffer,
                   blockTable: MTLBuffer,
                   numPagesBuf: MTLBuffer, kLenBuf: MTLBuffer,
                   qOffset: Int,           // position of Q's first row in each slot
                   qLen: Int,              // rows of Q per slot
                   headDim: Int,
                   H_Q: Int, H_KV: Int,
                   blockMask: FlexAttnBlockMask,
                   scoreMod: FlexAttnScoreMod)
```

Kernel-side: one PSO per (D, Q_BLOCK) tuple. Mask kind and score kind are
`constant uint&` kernel args and dispatched on inside the kernel — the branches
are resolved at SimdGroup-group granularity (all lanes take the same branch), so
they're free.

## 4. Kernel architecture

Grid: `MTLSize(width: B*H_KV, height: q_blocks, depth: N_SPLITS)`. Each TG owns
one (slot, kv_head, q_block, k_split). Threadgroup size: 32 lanes.

```
slot     = tg.x / H_KV
kv_head  = tg.x % H_KV
q_block  = tg.y
k_split  = tg.z
q_head_base = kv_head * Q_PER_TG         // GQA: Q_PER_TG = H_Q / H_KV
```

Per-TG threadgroup memory (v1 budget for D=256, Q_BLOCK=8):

- `Q_tile[Q_BLOCK * D]`          = 8 × 256 × 2 B = 4 KB
- `scores_tile[Q_BLOCK * K_BLOCK]` = 8 × 16 × 2 B = 256 B
- `O_acc[Q_PER_TG * Q_BLOCK * D]` = 2 × 8 × 256 × 4 B = 16 KB
- `m_state[Q_PER_TG * Q_BLOCK]`, `l_state[…]`, `scale_tile[…]` — 192 B
- Total ≈ 20 KB per TG, leaves headroom for V staging if needed

Per-split inner loop:

```
load Q_tile once (Q_PER_TG × Q_BLOCK × D halves)     [cooperative, 32 lanes]
init m_state = -INF, l_state = 0, O_acc = 0

// Iterate over this split's assigned K blocks (FULL list first, then PARTIAL).
for k_block_idx in partitioned_k_blocks(blockMask, slot, q_block, k_split):
    // Dereference the paged KV cache.
    phys = block_table[slot * MAX_PAGES + k_block_idx]
    K_base = K_cache + (phys * PAGE * H_KV + kv_head) * D
    V_base = V_cache + (phys * PAGE * H_KV + kv_head) * D

    // QK via simdgroup_matrix 8x8 MMA. Transposed K load.
    for pb in 0..(K_BLOCK / 8):
        mqk = 0
        for dt in 0..(D / 8):
            simdgroup_load(mq, Q_tile + dt*8, D)
            simdgroup_load(mk, K_base + (pb*8)*kv_row_stride + dt*8,
                           kv_row_stride, ulong2(0, 0), /*transpose=*/true)
            simdgroup_multiply_accumulate(mqk, mq, mk, mqk)
        simdgroup_store(mqk, scores_tile + pb*8, K_BLOCK)

    // Online softmax over this K block (per Q row).
    if (lid < Q_PER_TG * Q_BLOCK) {
        q_row, q_head = decode(lid)
        for k in 0..K_BLOCK:
            s = scores_tile[q_row * K_BLOCK + k] * qk_scale   // 1.0 for Gemma-4
            if (kind == PARTIAL) {
                q_idx = q_offset + q_row
                k_idx = k_block_idx * K_BLOCK + k
                if (!apply_mask_mod(b, h, q_idx, k_idx)) s = -INFINITY
            }
            if (score_mod == SOFTCAP) {
                s = softcap_value * tanh(s / softcap_value)
            }
            ...  // Flash update of m_state, l_state, scores_tile (exp'd)
    }

    // AV: accumulate scores × V into O_acc. Scalar cooperative as today.
    // ...

write (m_state, l_state, O_acc) to partials per Q head
```

### Block classification during dispatch

The outer loop `for k_block_idx in …` walks a **precomputed list**. If a block
was classified as empty during precomputeBlockMask, it's simply not in the list
and the TG never encounters it. This is the tile-sparsity win: at long contexts
with sliding_window=1024, the TG processing q_block=N only loops over the
at-most 64 K blocks that intersect the window (not all N/K_BLOCK of them).

The list split into full/partial matters because FULL blocks skip the per-k
mask check entirely — saves 2-4 cycles per k across the inner loop. At
decode (Q_LEN=1) with causal, every k block except the last is FULL. At prefill,
most (q_block, k_block) pairs with q_block > k_block are full, pairs on the
diagonal are partial, pairs with q_block < k_block are empty.

## 5. Mask / score mod registry

### mask_mod kinds

| Kind              | Parameter              | Predicate                                    |
| ----------------- | ---------------------- | -------------------------------------------- |
| `causal`          | —                      | `k_idx <= q_idx`                             |
| `causalSliding`   | `window: uint`         | `q_idx - window < k_idx && k_idx <= q_idx`   |
| `docIsolation`    | `doc_ids: MTLBuffer`   | `doc_ids[q_idx] == doc_ids[k_idx]` (plus causal) |
| `packedBitmask`   | `bitmask: MTLBuffer`   | `bitmask[b, h, q_idx, k_idx]` (escape hatch) |

All predicates must be monotonic on (q_idx, k_idx) tiles for tile classification
to work cheaply. Packed bitmask is the fallback when a new mask can't be
expressed closed-form — precompute does a full tile scan.

### score_mod kinds

| Kind      | Effect                                  |
| --------- | --------------------------------------- |
| `none`    | score unchanged                         |
| `softcap` | `score ← cap * tanh(score / cap)`       |

Adding new mods later: one `case` in the kernel switch, one variant in the
Swift enum, one branch in precomputeBlockMask (if mask). Score mods don't
affect the block classification.

## 6. Block-mask precomputation

Tile classification runs once per attention call and is cheap — each Q tile
checks at most `⌈kLen / K_BLOCK⌉` corner predicates. For k_len ≤ 8192 and
K_BLOCK=16, that's 512 calls. Trivially CPU-bound.

```swift
func precomputeBlockMask(maskMod, qLen, kLen, qBlock, kBlock) -> FlexAttnBlockMask {
    for slot in 0..<B:
        for q in 0..<qBlocks:
            for k in 0..<kBlocks:
                // Four-corner test (or more for non-monotonic masks).
                let tl = maskMod.apply(q * qBlock,      k * kBlock)
                let tr = maskMod.apply(q * qBlock,      k * kBlock + kBlock - 1)
                let bl = maskMod.apply(q * qBlock + qBlock - 1, k * kBlock)
                let br = maskMod.apply(q * qBlock + qBlock - 1, k * kBlock + kBlock - 1)
                let any = tl || tr || bl || br
                let all = tl && tr && bl && br
                if all      { fullList[slot, q].append(k) }
                else if any { partialList[slot, q].append(k) }
                // else: empty, don't emit
}
```

For packed_bitmask (non-monotonic), we do a full scan per tile instead of
corner-only. Still O(kLen) per Q tile and done on CPU once per forward.

Output lists go into two tight buffers so the kernel's outer loop is a simple
strided read:

```swift
fullKVIndices:    [B, q_blocks] × Int u32 indices, prefix-summed offsets
fullKVCount:      [B, q_blocks] count
```

For the common cases (causal, sliding-causal with small window) the precompute
is free enough to run on each forward pass. For packed_bitmask with large
masks, expose a cache keyed by `(maskModParams, qLen, kLen)`.

## 6a. Hardware budget (M5 Max)

- `maxThreadgroupMemoryLength`: **32 KB per TG** (Apple9)
- `maxThreadsPerThreadgroup`: 1024 across (x, y, z)
- `maxBufferLength`: 80 GB
- `hasUnifiedMemory`: true

Per-TG tg-mem accounting for the flash attention kernel with `Q_tile`, `scores_tile`, `O_acc`, `m/l/scale`:

```
tg_bytes(D, PAGE, Q_ROWS) =
  Q_tile:       2 * Q_ROWS * D            (halves)
  scores_tile:  2 * Q_ROWS * PAGE         (halves)
  O_acc:        4 * Q_ROWS * D            (floats)
  m/l/scale:    12 * Q_ROWS               (floats)
  = Q_ROWS * (6*D + 2*PAGE + 12)
```

| D   | PAGE | Q_ROWS | bytes  | fits 32 KB? |
|-----|------|--------|--------|-------------|
| 256 | 16   | 2      | 3160   | ✓ (AR v0)   |
| 256 | 16   | 16     | 25280  | ✓ (prefill slide Q_BLOCK=8)   |
| 256 | 16   | 32     | 50560  | ✗                             |
| 512 | 8    | 8      | 24800  | ✓ (AR v0 full) |
| 512 | 8    | 16     | 49600  | ✗                             |
| 512 | 8    | 32     | 99200  | ✗                             |

Naive v0 "grouped Q-heads per TG" (`Q_PER_TG = H_Q/H_KV`) breaks at D=512 when `Q_BLOCK > 1`: Q_ROWS scales as `Q_PER_TG * Q_BLOCK`, and at full-attn `Q_PER_TG=8`, any `Q_BLOCK>1` blows past 32 KB. Apparent workarounds (register O_acc, D-tile streaming, device-memory O_acc) are all unattractive — register budget is ~256 fp32s per lane so even `Q_ROWS=16` spills at D=512, and device-memory streaming moves GBs per K block.

**The actually-correct fix** is the llama.cpp geometry: dispatch **one TG per (slot, q_head, q_block)** rather than grouping H_Q/H_KV heads into one TG. This makes `Q_ROWS = Q_BLOCK` (not `Q_PER_TG * Q_BLOCK`), so at `Q_BLOCK=8, D=512`: Q_tile 8KB + O_acc 16KB + scores 128B + state ~100B = ~24 KB ✓. Multiple q_heads that share a KV head each re-read K — but Apple Silicon's cache hierarchy handles that duplicate read cheaply (unified memory + L1/L2). Net: 8× more TGs at full-attn than v0, which actually *improves* occupancy rather than hurting it.

Conclusions:

- **Slide v1 prefill**: grouped-Q geometry works at `Q_PER_TG=2, Q_BLOCK=8`. 24.7 KB, ships as `flex_attn_slide_v1_q8`.
- **Full v1 prefill**: un-grouped (one-TG-per-q_head) geometry at `Q_BLOCK=8`. 24.3 KB, ships as `flex_attn_full_prefill`.

Both bit-for-bit correct on 128-point numpy sweep (cos 0.999998). No layer-type special-casing, no D-dependent "fallback to AR inside prefill", no two implementations where one suffices.

## 6b. Prefill as a separate forward (not a mode toggle)

Prefill and AR decode differ in shape (Q_LEN), buffer sizing (B vs B*Q_LEN), dispatch grids, attention kernels, and MoE slot count. Instead of parameterizing every kernel and adding `if (prefill)` branches everywhere, we keep two disjoint forward paths:

- `buildStepCB(w) -> MTLCommandBuffer` — untouched AR path. Uses existing scratch buffers (`hidden`, `q_out`, etc.) sized `[B, …]`. Invokes `flex_attn_slide_v0`, `flex_attn_full_v0`. AR kernels never see Q_LEN.
- `buildPrefillCB(w, qLen) -> MTLCommandBuffer` — new path. Uses separate prefill scratch buffers sized `[B * MAX_Q_LEN, …]`. Invokes `flex_attn_slide_prefill`, `flex_attn_full_prefill`. Prefill kernels never see "one token per slot" code paths.

The existing kernels that already work per-row (rms_norm, gemv_q8_0_v6, moe_gemv_q4k_v6, etc.) accept a `numVecs` argument — they can be called from both AR and prefill with different numVecs without modification. They're essentially row-oblivious. The kernels that *aren't* row-oblivious (attention, kv_write, rope, embed) are the ones that need prefill-specific siblings.

Prefill-specific kernel catalog:

| AR kernel                      | Prefill sibling                 | Diff                                             |
|--------------------------------|----------------------------------|--------------------------------------------------|
| `embed_lookup`                  | `embed_lookup_multi`            | reads [B, Q_LEN] tokens, writes [B, Q_LEN, HIDDEN] |
| `kv_write`                     | `kv_write_multi`                | writes Q_LEN cache entries per batch (done)      |
| `rope_half`                    | `rope_half_multi`               | per-row position (done)                          |
| `flex_attn_slide_v0`            | `flex_attn_slide_prefill` (=v1_q8) | Q_BLOCK=8, per-row causal+SW mask (done)       |
| `flex_attn_full_v0`             | `flex_attn_full_prefill`        | Q_BLOCK=1 initially (serial Q inside prefill CB) |

Prefill-specific buffer allocations (one-time, at program start):

```swift
let MAX_Q_LEN = 8
// Expanded from [B, ...] → [B * MAX_Q_LEN, ...]. Memory overhead is modest:
// hidden: 4 * 8 * 2816 * 2 = 180 KB (vs 23 KB at AR)
// logits: 4 * 8 * 262144 * 2 = 16 MB (vs 2 MB at AR — but we can skip non-final
//         positions' logits if desired)
// Attention partials: 4 * 8 * 16 * 16 * 4 = 32 KB m_partials (vs 4 KB at AR)
// Etc. Nothing close to troubling.
let pre_hidden = halfBuf(B * MAX_Q_LEN * HIDDEN)
let pre_input_tokens = device.makeBuffer(length: B * MAX_Q_LEN * 4, ...)!
let pre_q_positions = device.makeBuffer(length: B * MAX_Q_LEN * 4, ...)!
// ... q_out, k_out, v_out, attn_out, mlp_out, moe_sum, ffn_combined, logits
// MoE routing: TOTAL_PREFILL_SLOTS = B * MAX_Q_LEN * TOPK = 256 (vs 32 AR)
let pre_expert_ids = device.makeBuffer(length: B * MAX_Q_LEN * TOPK * 4, ...)!
// ... gate_w, slot_token, batch_slots, group_start[E+1], m/l/O_partials
```

No shared state between AR and prefill paths except the GGUF weights + KV cache + block_table. Both paths can coexist in the same binary; the harness decides which to call based on workload (prefill for the initial prompt, AR for the subsequent token-at-a-time generation).

## 7. Phased implementation

1. **Phase 0 (skeleton)**: a single-PSO FlexAttention kernel for D=256, Q_BLOCK=8,
   `causalSliding` only. Replaces `paged_attn_slide_gqa_compute`. Validate
   against existing `hellolong` trajectory — KL should match 0.086 exactly.
   *Estimated effort: ~1 day.*

2. **Phase 1 (full-attn variant)**: D=512, Q_BLOCK=8, `causal` (no window). Replaces
   `paged_attn_full_gqa_compute`. Same trajectory-match test.
   *Estimated effort: ~4 hours.*

3. **Phase 2 (prefill support)**: Q_BLOCK up to 64 with Q_LEN>1. New buildStepCB
   variant that feeds S prompt tokens in one CB (vs S CBs). Measure
   tokens-per-sec vs current AR.
   *Estimated effort: ~1 day including new harness.*

4. **Phase 3 (doc isolation)**: `docIsolation` mask_mod. Test by packing two
   sequences into B=1 slot with doc_ids [0,0,0,1,1,1] and verifying cross-doc
   attention produces 0.
   *Estimated effort: ~4 hours.*

5. **Phase 4 (score_mod)**: `softcap` integration. Not strictly needed for
   Gemma-4 text attention but paves the way for future models and unifies
   the final-logit soft-cap path.
   *Estimated effort: ~2 hours.*

6. **Phase 5 (retire old kernels)**: delete `paged_attn_slide_gqa_compute`,
   `paged_attn_full_gqa_compute`, `paged_attn_slide_sgmm_compute`, and the
   non-split scalar paged_attn variants. Update all call-sites.
   *Estimated effort: ~2 hours, gated on Phases 0-4 passing.*

## 8. Testing plan

**Correctness suite**, in order:

1. **Decode, short**: `hellolong` (5 prompt + 16 AR), B=4. Expect mean KL ≤ 0.09, argmax 21/21.
2. **Decode, long**: synthetic 2048-token prompt. Validates sliding-window correctness past the 1024 boundary.
3. **Prefill**: same `hellolong` prompt, all 5 tokens in one CB. Expect per-position logits to match AR decode bit-for-bit (the prefill computes the same probabilities in parallel).
4. **Doc isolation**: two sequences packed into one slot, per-doc logits independent from the other doc.
5. **Soft-cap**: artificial model with attention soft-cap; verify logits match HF eager under `softcap` score_mod.

Each test: oracle dumped via `extract_lm_trajectory.py` (already wired) + per-layer hidden + per-position attn_out probes.

**Kernel-level unit tests** (new `flex_attn.swift` binary):

- Synthetic Q, K, V with known analytical output under each mask_mod, compare kernel output to numpy reference.
- Block-mask precompute: verify full/partial/empty counts match hand-computed values for small grids.
- GQA factoring: verify attention output is identical for Q_PER_TG=1 vs Q_PER_TG=8 given the same Q/K/V replicated.

## 9. Perf instrumentation

Per-dispatch metrics (extend existing `runLmForwardBench`):

- **Wall-clock per CB**: current 29 ms target. Post-flex expect similar or better at decode, significantly better at prefill (batched Q MMAs).
- **Dispatches per CB**: currently 545 (from the handover). Flex unifies slide+full, so it should drop.
- **Active-block ratio**: `full + partial` / `q_blocks * k_blocks`. For causal decode this trivially stays at ~0.5; for causal_sliding at long context it drops to `window / kLen` — directly visible perf win.

Log into memory handover after Phase 0 lands: new KL baseline, new per-step wall, tile-sparsity ratio at position 2048.

## 10. Non-goals / risks

- **Not matching FlexAttention's JIT flexibility**. We have a fixed enum;
  adding a new mask kind is a ~20-line Swift+Metal change, not a `@torch.compile`.
  Acceptable tradeoff.
- **Split-K mask interaction**. Each split sees a subset of K blocks. The
  mask precompute splits lists per-split so no split sees both halves of a
  masked boundary at once. Easier than the current implicit "split 0 does all
  the work when k_len is small" behavior.
- **Q_BLOCK > 8 with GQA**. For Q_PER_TG=8 and Q_BLOCK=8, the MMA uses
  all 8 rows productively (8 Q heads × 1 Q position). For Q_BLOCK=16 with
  Q_PER_TG=8, we'd need 2 MMA rows per Q position — the AV scalar path
  becomes 128 floats per K block per lane, past the register budget. Keep
  Q_BLOCK ≤ 8 in v1; revisit when prefill perf demands it.
