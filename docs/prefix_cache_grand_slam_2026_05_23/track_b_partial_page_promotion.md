# Track B — Partial-page promotion

> **STATUS:** Folded into Track D. Under D, partial pairs live as
> `TrieAnchor.partialTail: PartialPagePair?` (a field D's design
> already reserved). The hybrid (c)+(a) recommendation below is the
> right shape STANDALONE — but with D in the carrier, we go straight
> to true 1-token granularity via option (b) (partial-pair with CoW),
> because the trie's anchor structure makes the data-structure churn
> negligible. The pair-invariant analysis, kernel-write semantics, and
> KL-divergence test plan all apply unchanged.

---

# Partial-Page Promotion: Design Document

## 1. The Pair Invariant — Restated Formally

### Physical KV-cache geometry (Gemma-4)

The model has two attention regimes (see `bootstrap.swift:641-647`):

- **Slide layers**: `PAGE_SLIDE = 16`, `SLIDING_WINDOW = 1024`, head dim `D = 256`.
- **Full layers**: `PAGE_FULL = 8` (smaller page chosen for threadgroup-memory budget at D=512), unbounded attention.

Both layer types share a single per-session `block_table[slot][p]` indexed by `p ∈ [0, MAX_PAGES_PER_SLOT)`. Per `lm_engine.swift:1546-1563` (`ensurePages`), the table is sized in **PAGE_FULL=8 units** so the smaller-page (full) layers have enough entries.

Consequence: for any logical slide-page index `P` (16-token window starting at token `P*16`), the full layers' K/V for tokens `[P*16, P*16+15]` lives across **two consecutive entries** in `block_table` — `block_table[2P]` and `block_table[2P+1]` — each holding 8 tokens of full-attn K/V. The slide layers' K/V for that same range lives entirely in `block_table[2P]` (which is treated as a 16-token page by slide-attention).

### What each member holds (citing `page_manager.swift:55-68, 83-91` + `lm_engine.swift:1826-1833`)

- **`slidePrimary` = phys page at `block_table[2P]`**:
  - Slide K/V for slide-positions `[P*16, P*16+15]` (16 rows × `SLIDE_H` × 256).
  - Full K/V for full-positions `[P*16, P*16+7]` (8 rows × `FULL_H` × 512).
- **`fullSibling` = phys page at `block_table[2P+1]`**:
  - Full K/V for full-positions `[P*16+8, P*16+15]` (8 rows × `FULL_H` × 512).
  - (Slide-layer view of this index is unused for the 16-token-aligned slide tile; later prefill may write slide K/V from positions `[P*16+16, P*16+31]` here if extended further.)

### The kv-write addressing

`kv_write_multi` (kernels.swift:636-661) computes per-token destination as:
```
lp = pos / PAGE; off = pos % PAGE
phys = block_table[b * max_pages + lp]
dst = phys_kv[(local_phys * PAGE + off) * H + h] ...
```
The "PAGE" constant baked into the dispatch is the *layer's own* page size — `PAGE_SLIDE` for slide layers, `PAGE_FULL` for full layers. So a write at slide-position `5` lands at full-page `0` offset `5` AND at slide-page `0` offset `5` — both writes happen, into different phys-page buffers, because each layer has its own per-layer `K_chunks` and `V_chunks` arrays.

### The pair invariant

**A `(slidePrimary, fullSibling)` pair is bit-identical to fresh compute at logical-positions `[P*16, P*16+15]` if and only if:**

1. Both members were written by a prefill (or AR sequence) that observed the same input token sequence `tokens[0..P*16+16]` AND the same control-vector envelope state across those positions (encoded via `cvecDigestForPage`, see `page_manager.swift:185-212` and `lm_engine.swift:709`).
2. The writing session executed all 16 positions `[P*16, P*16+15]` in order, so K/V values at the latter positions saw correctly-populated K/V at the earlier ones (causal attention dependence runs through prior K rows).
3. No subsequent write has overwritten the page (refcount guarantees this — see `page_manager.swift:355-365` `decref` preserves content while refcount==0 unless `allocFresh` chooses it for forced eviction).

**Promoting only one member violates the invariant** for any read at full-positions `[P*16+8, P*16+15]` — the full layers' attention dispatcher reads `block_table[2P+1]`, finds whatever zeros/stale data live there, computes garbage (`page_manager.swift:60-68` documents the empirical `KL ~ 0.38`).

The codebase guarantees pair atomicity in `PageManager.promotePair` (`page_manager.swift:327-342`) — both phys pages share one `contentHash`, both point at each other via `pairMate`, both stored in the same `SharedPagePair` value. Adoption (`lm_engine.swift:1054-1064`) increfs both atomically.

---

## 2. Option (a) — Flush-Prefill with Sentinel Tokens

### Mechanism

At session teardown (`lm_engine.swift:1796-1810` `closeSession`), if `s.position % PAGE_SLIDE != 0`, run additional prefill iterations with sentinel tokens to pad the in-flight slide page to a 16-token boundary before decref. The completed pair is then promoted normally via `promoteFinishedPages` and stored in `contentIndex` with `validUpTo = (s.position before flush)`.

### Sentinel token choice

Three candidates:
1. **`weights.bosTokenId`** — already used as silenced-slot filler in `stepPrefillForSession` (`lm_engine.swift:2701`). Easy, but real prompts may begin with BOS, so a sentinel of BOS at an interior position is semantically odd. Functionally fine — sentinels never get sampled.
2. **`weights.padTokenId`** — semantically cleanest. Need to verify exists in tokenizer.
3. **A never-emitted token** (e.g. an unused vocab id reserved in tokenizer config) — the tightest choice; guarantees the sentinel is not a legitimate user token.

Recommendation: **pad token** (Gemma-style tokenizers reserve `<pad>`). If unavailable, BOS works — the K/V values produced are unimportant; only their addressing slot matters.

### Correctness for adopters (the bit-identity question)

**Sentinel K/V values are NOT bit-identical** to what a real session ending there would produce: a real session never emits at positions `[P*16+r, P*16+15]` (the session ended at position `P*16+r`), so "real" K/V at those slots is undefined. The sentinel run produces *some* K/V — the question is: **can an adopter read it?**

Adopters that resume at any position `Q ≤ validUpTo` execute attention with `q_pos = Q`, which reads K positions `[max(0, Q-W+1), Q]` where `W = SLIDING_WINDOW = 1024` for slide layers, or `[0, Q]` for full layers. By construction `Q ≤ validUpTo ≤ P*16+r-1 < P*16+r`, so adopter K-reads stay strictly below `validUpTo`.

**Does the attention kernel for `q_pos = validUpTo - 1` look at any position `≥ validUpTo`?** No — see `kernels.swift:10293-10305`, slide kernel masks `if (k_pos >= k_len) sv = -INFINITY` (causal) and `if (k_pos < window_lo) sv = -INFINITY` (sliding). The adopter sets `k_len = Q + 1`, so `k_pos ∈ [window_lo, Q]` are kept, `k_pos ≥ Q+1` masked. Sentinel K/V at `[validUpTo, P*16+15]` is fully masked.

**Caveat**: `validUpTo` must be reported to adopters and they must NOT extend `k_len` past it. The adoption call in `revisitCacheProbe`/`adoptSharedPrefixPages` currently sets `position = adopted * PAGE_SLIDE` (`lm_engine.swift:989, 1105`); with partial pairs it needs to set `position = validUpTo` instead.

### Cost

- **GPU cost at teardown**: `(PAGE_SLIDE - r)` extra prefill positions = 1..15 tokens. At a typical 200 tok/s prefill rate, that's 5-75 ms wall time per teardown — quietly added to the session's close path. Multiplied across thousands of sessions per hour this is meaningful (1-15 ms p50, ~75 ms p99).
- **CPU**: trivial — populate `posP[]` and `tokP[]` for r extra positions, same as current prefill driver.
- **Memory**: zero — pages were already allocated.
- **Cache benefit**: full prompt cached, granular to 1 token. The adopter saves the full prefill it would otherwise re-run.

### Implementation surface

- New: `Engine.flushTailToPageBoundary(_ s: Session)` (~60 LOC): builds a synthetic `chunkQueue` entry of `[pad, pad, ..., pad]` length `(PAGE_SLIDE - r)`, calls `stepPrefillForSession(s)` once.
- New: `PageManager.SharedPagePair` gains a `validUpTo: Int` (~5 LOC + sites).
- Modify: `promoteFinishedPages` to use the new `validUpTo` field for the trailing partial page (`lm_engine.swift:1816-1851`), specifically: after the flush, the last "fully written" page has `validUpTo < PAGE_SLIDE * (p+1)`.
- Modify: `adoptSharedPrefixPages`/`revisitCacheProbe` to set `position = pair.validUpTo` for the trailing partial page (not `(adopted+1) * PAGE_SLIDE`).
- Modify: `closeSession` to call flush before decref.

Total: **~120 LOC**. Touches PageManager, Session, Engine, but no kernels.

---

## 3. Option (b) — Partial-Pair with `validUpTo`

### Data structure

Two viable shapes:

**(b1) Single index, value-tagged**:
```swift
struct SharedPagePair {
    let slidePrimary: Int
    let fullSibling: Int?     // nil if only first 8 tokens valid
    let validUpTo: Int        // 1..16
}
private var contentIndex: [UInt64: SharedPagePair]
```
Old `validUpTo == 16` is the legacy fully-aligned case; new partial cases write `validUpTo < 16`. Backward compatible (a fully-aligned pair simply has `validUpTo = 16`).

**(b2) Two indices**:
```swift
private var contentIndex: [UInt64: SharedPagePair]       // 16-aligned (current)
private var partialIndex: [UInt64: PartialPagePair]      // <16-aligned
```
Adoption probes `contentIndex` first, falls back to `partialIndex`. Cleaner separation but doubles the lookup-path complexity.

Recommend **(b1)** — single field added, dict-lookup count unchanged.

### Adoption returns position-to-resume-from

`adoptSharedPrefixPages` currently returns `Int` (count of full pages adopted). New return shape: `(slidePagesAdopted: Int, partialTailTokens: Int)`. The caller computes `position = slidePagesAdopted * PAGE_SLIDE + partialTailTokens` and trims `chunkQueue` accordingly.

### Can prefill start mid-page?

**Yes.** Per `kernels.swift:636-661` (`kv_write_multi`):
```
pos = q_positions[q_flat];
lp = pos / PAGE; off = pos % PAGE;
phys = block_table[b * max_pages + lp];
```
The write position is whatever `posP[]` says — `pre_q_positions` is populated by `stepPrefillForSession` as `posP[sslot * thisTile + i] = positionStart + i` (`lm_engine.swift:2700`). Setting `positionStart = validUpTo` writes K/V at positions `[validUpTo, validUpTo + qLen - 1]`, which lands at offsets `[validUpTo % PAGE, ...]` within the page indexed by `block_table[validUpTo / PAGE]`.

**But there's a subtle issue**: the partial page is currently *adopted* by the consumer session via `incref`. The consumer is now WRITING to a page it shares with the donor's content-index entry. If a second consumer adopts the same partial pair concurrently, both would attempt to write to positions `[validUpTo, PAGE_SLIDE-1]` of the SAME phys page — a race + correctness disaster.

Fix: **adoption of a partial pair must allocate a fresh page and copy** (CoW). Or: **adopt-then-fork-on-extend** — keep the shared page read-only, allocate a fresh page when the consumer's prefill needs to write into it, copy the `validUpTo` prefix bytes into the fresh page, then prefill the rest into the fresh page.

### Copy cost

Per slide page: 16 tokens × `SLIDE_H * SLIDE_HD * 2 bytes` per layer × NUM_LAYERS. Concrete: at `SLIDE_H = 4`, `HD = 256`, that's 16 × 4 × 256 × 2 = 32 KB per slide layer per K (or V) buffer per page. Across 25 layers × 2 (K+V) = 1.6 MB per partial-pair adoption. At Apple's ~250 GB/s memcpy, ~6 μs. Negligible.

Plus the full-sibling copy: 16 tokens × `FULL_H * FULL_HD * 2` × layers. Similar order.

So total CoW copy on adoption of a partial pair: ~tens of microseconds per phys page, ~hundreds across all layers. **Cost is dominated by GPU memcpy bandwidth, ~tens of microseconds total** — well below a single prefill step (~30 ms at qLen=16).

### Alternative: prefill kernel writes only positions `[validUpTo, PAGE_SLIDE-1]`

If the consumer is willing to *not* copy and the only writer is a single consumer (no concurrent adoption), the prefill kernel can write to the shared page's tail positions directly. But concurrent adoption is impossible to prevent with the current refcount model — once `decref` lands the page on the free list, any adopter can `incref` it. CoW is mandatory.

### Implementation surface

- `PageManager.SharedPagePair` + `validUpTo`: ~10 LOC.
- `PageManager.promotePair` extended with `validUpTo` arg: ~5 LOC.
- `adoptSharedPrefixPages` returns `(Int, Int)` partial-aware: ~30 LOC.
- `revisitCacheProbe` consumes new return shape, sets `position` accordingly: ~25 LOC.
- `submit()` first-submit path likewise: ~25 LOC.
- CoW copy on adoption: `Engine.copyPartialPair(src: pair, dst: freshPair, tokensToCopy: validUpTo)` — walks K/V chunks per layer, memcpy. ~120 LOC.
- `promoteFinishedPages` + `closeSession`: at teardown if `position % PAGE_SLIDE != 0`, promote the trailing pair with `validUpTo = position % PAGE_SLIDE`. ~40 LOC.

Total: **~250 LOC**. No kernel changes (kernels already support arbitrary positions).

---

## 4. Option (c) — 8-Token Sub-Granularity

### Does promoting `slidePrimary` alone produce correct K/V for adopters reading positions `[P*16, P*16+7]`?

Per `page_manager.swift:55-68`, the slide primary holds:
- Slide K/V for slide-positions `[P*16, P*16+15]` (16 entries; only first 8 are valid if the session stopped at `P*16+7`).
- Full K/V for full-positions `[P*16, P*16+7]` (all 8 entries valid — they live in `block_table[2P]`, separate from `block_table[2P+1]`).

**For an adopter resuming at `position = P*16+8`:**
- Slide-layer attention reads K positions `[max(0, P*16+7-SLIDING_WINDOW+1), P*16+7]`. K/V for `[P*16, P*16+7]` is in the slide-primary's slide K/V buffer (offsets 0..7 of the 16-slot page). Valid. K/V for positions `[..P*16-1]` is in earlier pages, also adopted. Valid.
- Slide-layer K/V for slide-positions `[P*16+8, P*16+15]` is invalid (never written), but these positions are past adopter's resume point — adopter will write them next. Fine.
- Full-layer attention reads K positions `[0, P*16+7]`. Full K/V for `[P*16, P*16+7]` is in `block_table[2P]` — adopted, valid. Full K/V for `[P*16+8, P*16+15]` lives in `block_table[2P+1]` — NOT adopted, but adopter doesn't read past `P*16+7`. Fine.

**Correctness holds.** Adopting just `slidePrimary` gives the adopter a valid K/V state at positions `[0, P*16+7]` provided all earlier slide pages were also adopted (transitively).

### Cache-key design

The simplest unified scheme: **content-index entries always store a pair where `fullSibling` is `Int?` (nullable) AND `validUpTo` indicates the high water mark**. There is one `contentIndex: [UInt64: SharedPagePair]`. Three regimes:

1. **`fullSibling != nil, validUpTo = 16`**: fully-aligned pair (today's case).
2. **`fullSibling = nil, validUpTo = 8`**: half-pair, only first 8 tokens cached.
3. **`fullSibling != nil, validUpTo ∈ (8, 16)`**: 8-aligned but with extra valid tokens via flush (hybrid c+a).

The hash key for a half-pair is computed over `tokens[0..P*16+8]` — **a DIFFERENT key** than the fully-aligned `tokens[0..P*16+16]`. So they coexist in one dict without conflict. Adoption walks the prefix in 8-token increments and tries both keys at each boundary.

### Two-stage promotion semantics

At teardown:
- If `position >= P*16 + 8` for some `P` not yet promoted: promote slide-primary with `validUpTo=8` under hash `H(tokens[0..P*16+8])`.
- If `position >= P*16 + 16`: also promote full-sibling (upgrade to full pair) under hash `H(tokens[0..P*16+16])`. Note: this is a DIFFERENT hash key; two distinct entries in `contentIndex` may exist for the same `P`, one half-keyed, one full-keyed.

Idempotency: the existing `promotePair` already guards via `if contentIndex[contentHash] != nil { return }` (`page_manager.swift:331`). A new `promoteHalf(slidePrimary, contentHash)` follows the same pattern.

### Cost

- **Teardown CPU**: same as current promote loop, just runs over half-pages too. ~microseconds per page.
- **Teardown GPU**: ZERO. No additional prefill needed — the slide-primary is already written by the session's normal prefill at the moment the session passed `P*16+8`.
- **Memory**: ~2× content-index entries in the worst case. At ~thousands of entries, hundreds of KB — irrelevant.
- **Hit floor**: 8 tokens. Better than today's 16, worse than (a)/(b)'s 1.

### Implementation surface

- `PageManager.SharedPagePair`: `fullSibling: Int?`, `validUpTo: Int`: ~5 LOC.
- New `PageManager.promoteHalf(slidePrimary:, contentHash:)`: ~25 LOC.
- `promoteFinishedPages` loop: walk in 8-token increments, promote half-pairs at boundaries that don't have full coverage yet: ~40 LOC.
- `adoptSharedPrefixPages` 8-stride probing: ~50 LOC.
- `revisitCacheProbe` position-update to handle non-16-multiple advances: ~15 LOC.
- Backward-compat: when reading legacy code paths, `fullSibling: Int?` requires `.flatMap` or `if let` guards across ~10 sites.

Total: **~140 LOC**. No kernel changes.

---

## 5. Comparison Table

| Metric | Current (16-only) | (a) Flush-prefill | (b) Partial-pair | (c) 8-granular | (c+a) Hybrid |
|---|---|---|---|---|---|
| Cache-hit floor (min tokens) | 16 | **1** | **1** | 8 | **1** |
| Per-token CPU cost at teardown | 0 | trivial | trivial | trivial | trivial |
| Per-token GPU cost at teardown | 0 | **1-15 prefill steps × ~5 ms** | 0 | 0 | 1-7 prefill steps |
| Additional memory | 0 | 0 | ~1.6 MB CoW per adoption | ~2× index entries | ~2× index entries |
| Complexity (1-5) | — | **2** | 4 | 3 | **3** |
| Backward compat with dict lookup | — | full | full | full | full |
| Requires kernel changes | — | no | no | no | no |
| Concurrent-adoption safe | — | yes | requires CoW | yes | yes |

---

## 6. Recommendation

**Hybrid: (c) as the base, plus (a) for the partial tail.**

(NOTE — superseded under D-carrier execution: with Track D in scope, the trie's anchor structure makes option (b)'s data-structure churn negligible, so go directly to true 1-token granularity via (b) at trie anchors. The hybrid below is the right shape for STANDALONE Track B.)

Rationale:

1. **(c) is nearly free** (no GPU work, no copies, no kernel changes) and immediately halves the floor from 16 to 8. The 8-token granularity is enough for the ST 2491-token case to recover ~3-7 more tokens per session vs. today (whichever 8-aligned boundary is closest below the actual position).
2. **(a) on top of (c)** closes the remaining gap to 1-token granularity at modest GPU cost. The flush-prefill only runs over the trailing `(8 - r % 8)` tokens (1-7) instead of the full `(16 - r % 16)` (1-15), since (c) already captures the 8-boundary.
3. **(b) is rejected as the primary**: the CoW-on-adoption requirement adds a memcpy-and-allocate path that bumps adoption from O(1) refcount ops to O(layers) bytes-moved, and the data-structure churn (partial-pair index, fork-on-write logic) is hard to get right. The complexity is justified only if the GPU cost of (a)'s flush is shown to be the bottleneck — which is unlikely at 1-7 token tails.
4. **(c) is the lowest-risk first commit** — ship it standalone first, measure cache-hit lift, then decide whether (a)'s ~5-35 ms-per-close GPU cost is worth the additional 1-7 tokens of cache-hit floor reduction.

Phased rollout: **(c) first, (a) second**, gated by measured cache-hit-rate improvement from (c).

---

## 7. Test Plan

### Smoking-gun (correctness)

1. **15-token prompt** (`tokens = [t0..t14]`). Run session A to completion, close. Open session B with same 15 tokens.
   - With **(c)**: B's `adoptSharedPrefixPages` finds the half-pair at `H(tokens[0..8])`, adopts. B's `position = 8`. B prefills 7 tokens. Cache-hit accounting reports 8 tokens hit, 7 missed.
   - With **(a) + (c)**: B finds the full pair at hash with `validUpTo=15`. `position = 15`, prefills 1 token (the backstop tail-guard `lm_engine.swift:979-988, 1090-1104` still applies). Cache-hit reports 14 hit, 1 miss.
2. **Identical-prompt round trip**: 1000 sessions of the same 17-token prompt. Today: first session prefills 17 tokens, all subsequent sessions adopt 16 (hit) + prefill 1 (miss). With (c): first session prefills 17, subsequent adopt 16 from full pair OR 8 from half pair if the full pair wasn't promoted (it should be — both boundaries are crossed). With (a)+(c): subsequent adopt all 16 hit + tail-guard backstop forces 1 token re-prefill.

### Correctness (KL divergence)

For each partial-promotion case, run **paired-comparison** against fresh compute:
- Generate logits for `tokens[0..N]` from a cold session (no cache hit).
- Generate logits for the same `tokens[0..N]` from a warm session that adopted a partial page.
- Compute KL(cold || warm) per position.
- **Target**: KL < 1e-5 averaged across positions; max KL < 1e-3.
- The existing `kv_visualizer.swift` may already export per-position KV; otherwise add a logit-dump shim.

Particular boundary cases:
- Tokens crossing `P*16+7` → `P*16+8` (the half-pair boundary in (c)).
- Tokens crossing `P*16+15` → `P*16+16` (the full-pair boundary).
- Sliding-window edges: adopter at position `Q ≥ SLIDING_WINDOW` so window_lo > 0.

### Performance (ST cold + warm path)

Use existing `profile_prefill.swift` as the reference rig:
- Single-stream 2491-token prompt, **cold**: measure end-to-end tok/s (should match today's number).
- Same prompt **warm** (second submission, full cache hit + 11-token tail): with (c), expect to adopt 2480 / 16 = 155 slide pages → resume at position 2480, prefill 11 tokens. Compared to today: 2480 hit (same), 11 miss (same). Hybrid (c+a) adopts up to 2490 tokens, prefills 1 token (the tail-guard backstop's enforced miss).
- Crossover prompt sizes 8, 16, 24, 32 — verify cache adoption percentage.

---

## 8. Interactions with Other Tracks

### Track A (backstop removal)

The "tail-guard backstop" (`lm_engine.swift:979-988, 1090-1104`) currently unadopts the trailing slide page when all tokens would otherwise be cached, so prefill has ≥1 token to drive post-prefill sampling. **With partial pairs**: if the adopter sets `position = validUpTo` and `validUpTo == prompt.count`, the same all-cached condition fires. The backstop unadopts the trailing partial pair, decrefs both pages, and re-prefills the last 16 (or `validUpTo`) tokens.

**This negates the benefit of (a) for fully-cached prompts**. Track A removal (replacing the backstop with a "synthetic sample-only step that re-uses the cached K/V") is a prerequisite to extract full value from (a). Without Track A: (a)+(c) still helps partial cache hits (most real cases) but not 100%-cached repeats. With Track A: 100%-cached prompts become 0-prefill operations.

(c) standalone is **not** blocked on Track A — the half-pair adoption is just like a full-pair adoption from the position-tracking standpoint, no new code path.

### Track D (radix trie)

A radix trie stores prefix nodes at arbitrary character/token positions, not at fixed page boundaries. **Partial pairs fit naturally**: a trie node at position `N` (any 8-aligned position with (c), or any position with (a)/(b)) points at a `SharedPagePair` with `validUpTo = N % PAGE_SLIDE`. The trie's "longest matching prefix" query returns the node whose `validUpTo` reflects the deepest cached position.

The trie can also help (b)'s CoW concerns: instead of allocating a fresh page and copying on adoption, the trie node's pair can be marked "forking" — the next adopter who needs to extend gets a copy, while concurrent read-only adopters share the original. Trie nodes naturally support this versioning.

### Track C (cvecDigest)

`cvecDigestForPage` (`lm_engine.swift:709`) is currently computed over a 16-token range (`pageSize: PAGE_SLIDE`). With **(c)**, the half-pair's hash is computed over `tokens[0..P*16+8]` and the cvec digest must be computed over **the corresponding 8-token sub-range** to match. The digest function must accept a variable range:
```swift
func cvecDigestForRange(start: Int, end: Int) -> UInt64
```
Half-pair promote/probe uses `(P*16, P*16+8)`; full-pair uses `(P*16, P*16+16)`. The digest does NOT need to include `validUpTo` — `validUpTo` is determined by the range covered by the hash, which is already part of the key.

### Track E (rename pass)

New public-facing names that need entries in the naming table:
- `SharedPagePair.validUpTo: Int` — "how many tokens of this pair's logical 16-token range are bit-identical to fresh compute."
- `SharedPagePair.fullSibling: Int?` — now nullable; "nil means only the first PAGE_FULL=8 tokens are valid (half-pair); non-nil with `validUpTo == PAGE_SLIDE` is the legacy full pair."
- `PageManager.promoteHalf(slidePrimary:, contentHash:)` — the (c) entry point.
- `Engine.flushTailToPageBoundary(_:)` — the (a) entry point.
- Possibly `partialTailTokens` in adoption return shape.

---

### Critical Files for Implementation

- /Users/mdot/metal-microbench/page_manager.swift
- /Users/mdot/metal-microbench/lm_engine.swift
- /Users/mdot/metal-microbench/bootstrap.swift
- /Users/mdot/metal-microbench/lm_session.swift
- /Users/mdot/metal-microbench/kernels.swift
