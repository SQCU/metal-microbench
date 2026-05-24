# Track D — Token-granularity radix-trie prefix cache (THE CARRIER)

> **STATUS:** This is the carrier refactor. A/B/C/E fold into D's
> shape per the README. The 3-PR phased migration plan below
> ("Shadow → Read-through → Retire dict") is METR-sized and presumes
> human review cycles; in the multipolar subagent format this lands
> as a single integrated change with the test harness as the safety
> net, not the shadow-mode counter.

---

# Radix-Trie Prefix Cache: Design Document

**Status:** proposal
**Author:** Architecture planning pass, 2026-05-23
**Audience:** Senior engineer who will implement this in the metal-microbench Gemma-4 engine
**Scope:** Replace `PageManager.contentIndex: [UInt64: SharedPagePair]` with a token-granularity radix trie matching SGLang's RadixAttention semantics

---

## 1. Reference architecture

### SGLang's RadixAttention

SGLang stores its prefix cache as a **radix trie of token IDs**. Each edge is a *sequence* of one or more tokens; each node owns a child map keyed on the first token of the child's edge. Each node holds a handle to a contiguous range of KV-cache slots (in their case, a contiguous slice of `kv_indices` into a pool of "token-cells" rather than fixed-size pages — they don't quantize at the storage layer).

Three operations form the API:

1. **`match_prefix(token_ids)`** — walks from root, following edges that match successive tokens. Returns the longest matched prefix as `(matched_length, kv_indices_for_those_tokens, last_node)`. When the input diverges *partway* through an edge, the trie splits that edge at the divergence point so the matched portion becomes a new internal node.
2. **`insert(token_ids, kv_indices)`** — walks the trie following matching edges; at the divergence point, attaches a new leaf carrying the remaining tokens + the remaining KV indices. Insertion can also split an existing edge if a *shorter* prefix is inserted later.
3. **`evict(num_tokens)`** — LRU eviction at trie *leaves* only (internal nodes are pinned because they carry data shared with descendants). The eviction policy maintains a min-heap of leaves keyed on `last_access_time`; pop leaves until `num_tokens` cells have been freed. Active sessions hold refcounts ("locked" status) on nodes along their path; locked nodes are never evicted.

The defining property: **lookup returns the longest matched prefix of any length L**, not just page-aligned L. The prefill resumes at position L. Token granularity falls naturally out of the data structure.

### Differences from vLLM's block-level APC

vLLM's Automatic Prefix Cache (APC) is at **block granularity** (16 tokens, configurable). Its index is a `Dict[block_hash, PhysicalBlock]` where `block_hash = hash(parent_block_hash, tokens_in_this_block)` — a chained hash, exactly like our current setup. Lookup quantizes to block boundaries. The data structure is flat; there is no trie. To share a 7-token suffix of one prefix with another prefix that has the same 7 tokens, vLLM cannot — the hashes diverge.

SGLang's radix trie supports:
- Sub-block-granularity sharing (two prompts with identical first 23 tokens share 23 tokens, not just the first 16).
- Edge-splitting when a new request reveals a previously-unseen branch point.

The cost is a more complex data structure and a non-trivial split operation. The benefit is that **with bursty, near-duplicate prompts** (multi-turn chat, system-prompt sharing, few-shot eval, agent decompositions) the hit rate is materially higher than block-aligned caches.

### Where we are today

Our cache is a **degenerate block-level APC** with these twists:

- Block size is 16 tokens (`PAGE_SLIDE`), but each block actually owns TWO physical phys-pages (`SharedPagePair`) because Gemma-4's sliding-window layers use `PAGE_SLIDE=16` and full-attention layers use `PAGE_FULL=8`, sharing one `block_table`. A "block" in our cache means "a slide page plus its full-sibling that holds the second-half full-attn K/V".
- Block hash is `FNV-1a(consumedTokens[0..(P+1)*PAGE_SLIDE], cvecDigest)` — chained on tokens (the whole prefix is rehashed each page, not just `parent_hash + this_page_tokens`, but the chained property is equivalent).
- A per-page `cvecDigest` is XOR-mixed into the hash; this partitions the cache key namespace by steering configuration.
- Lookup loop is in `Session.adoptSharedPrefixPages` (`lm_engine.swift:1041`) — walks pages 0..N, queries `contentIndex[hash]` each step, stops at first miss.
- Insertion is in `LmEngine.promoteFinishedPages` (`lm_engine.swift:1816`) — runs after each prefill tile and after AR steps that cross a page boundary. Publishes via `PageManager.promotePair`.

Token-granularity lookup gets us hit rates on multi-turn chat where a tool-call result is only 7 tokens long (today: 0 tokens of that turn's prefix get cached; with a trie: all 7 of those tokens contribute to future hits when the next turn extends them).

---

## 2. Chosen data structure

**Pick: (c) Token-trie with page-aligned leaves.**

Rationale below. First, the disqualifying analysis on the alternatives.

### Why not (a) — pure radix trie of tokens

A pure SGLang-style trie has nodes whose leaves point to KV ranges of arbitrary length. To adopt this directly we would need to break the page-pair invariant: a leaf that covers 23 tokens cannot point at "one SharedPagePair" because the second page-pair starts at token 16, and 23 isn't a page boundary. The KV physical layout *is* still page-quantized at 16 (slide) / 8 (full) regardless of what the trie thinks. We would have to invent a "partial pair" concept inside the trie just to make leaves work at arbitrary lengths — and the GPU still can't *read* a partial pair (kv_write writes whole pages; the attention kernel reads via `block_table` which is page-quantized). The only thing token-granular trie lookup actually buys us is a **starting position** for the prefill tail. The KV cache itself cannot honor sub-page granularity without a much deeper refactor (a new "tail page" concept where the first N tokens are cached and the next 16-N must still be computed).

So the storage layer is fundamentally page-quantized at 16. We should not pretend otherwise in the trie's leaves.

### Why not (b) — dict + secondary sub-page dict

Adds the complexity of two indexes without giving us actual sub-page sharing. The "longest prefix" answer is still discrete at 16-token boundaries because *that's where the page-pairs exist*. The only thing the secondary index would do is collapse with the primary index for any meaningful query. Don't double-bookkeep what isn't there.

### Why (c) — token-trie, page-aligned leaves

The data structure is a trie of tokens. Internal nodes at non-page-boundary positions exist *only* to enable navigation. Leaves (or "anchor" nodes) exist *only* at positions divisible by `PAGE_SLIDE`, and they carry a `SharedPagePair` handle.

This gives us:

- **Longest-page-aligned-prefix lookup in O(L)** where L is the matched prefix length in tokens. We walk down the trie token-by-token until divergence or end-of-input; the answer is "the deepest anchor node we passed through on the way". For the common case of a 1000-token shared system prompt, that's 1000 token-comparisons (cheap — `UInt32` equality in a hashmap-keyed child lookup) vs. today's 62 FNV-1a hashes over slices of length 16..1000 (62 × ~average 500 = 31000 hash mixes). **The trie is actually faster than the current hash chain for long prefixes.**
- **Edge compression.** Long sequences of single-child nodes collapse to a single edge carrying a `[UInt32]` of tokens. Memory cost stays modest (one trie node per branch point + one per anchor, not one per token).
- **The KV layer doesn't change.** Each leaf still hands out a `SharedPagePair`; the GPU still reads page-aligned KV. We only gain token-granular *insight* into "which prefix length was matched."

For now, we deliberately do NOT support returning matched-but-not-aligned tail length. The lookup API can compute and *report* it ("you matched 23 of the 27 tokens you asked about; the prefill tail can start at token 16 because that's the last anchor"), but the engine treats the matched length as `floor(L_matched / PAGE_SLIDE) * PAGE_SLIDE` for actual page adoption. **Track B** (partial-page promotion, see §9) is the future work that lets us honor the unaligned matched length. The trie is designed to ingest partial pairs the day Track B lands without an API change — we add an `anchor.partial: PartialPagePair?` field and the lookup returns it when present.

### Properties

| Property | Value |
|---|---|
| Memory per active node | ~80 bytes (parent ptr + edge tokens slice header + child map header + anchor ptr) |
| Memory per cached prefix (1000 tokens, fully shared with N siblings) | 1 anchor every 16 tokens = 63 anchor nodes + 0 branch nodes = ~5 KB regardless of N (the win — N=1 and N=100 cost the same) |
| Memory per cached prefix (1000 tokens, no siblings) | One edge of length 1000 + one anchor every 16 = 63 anchors. ~5 KB. |
| Lookup worst case | O(L) token comparisons + O(L/16) anchor lookups. For L=10K, ~10K hashmap lookups. Each is ~50ns. ~500 µs. Today's amortized: O(L/16) FNV chains × O(L) per chain = O(L²/16). 50 ms for L=10K. **Trie is 100× faster at this scale.** |
| Lookup amortized (common case L=128) | ~6 µs. Today: ~3 µs. **Same order; trie has a small constant-factor overhead from per-token dispatch but it's noise.** |
| Insertion (extend by 16 tokens, no split) | O(16) — walk down, attach 1 anchor. ~1 µs. |
| Insertion (split at non-anchor) | O(edge_length) to split the edge. Bounded by edge compression. ~5 µs typical. |
| Implementation complexity (1–5) | **3.** Edge-compression and split logic is the tricky part. Anchor placement is a small additional constraint. ref-counting + LRU is well-trodden. |

---

## 3. Lookup API

```swift
struct PrefixMatch {
    /// Length in tokens of the longest matched prefix that has page-anchored
    /// KV. Always a multiple of PAGE_SLIDE. The caller advances `position`
    /// by exactly this number of tokens.
    let alignedMatchLength: Int

    /// Length in tokens of the longest matched prefix in the trie,
    /// regardless of anchor alignment. Always >= alignedMatchLength.
    /// alignedMatchLength + tail is what gets re-prefilled.
    /// Reported for telemetry / future Track B; not used today.
    let trieMatchLength: Int

    /// Page-pairs covering [0, alignedMatchLength). Each pair was incref'd
    /// before return; caller MUST decref on session teardown.
    /// Empty array means no anchor was reached (no usable cache hit).
    let pages: [SharedPagePair]

    /// Reserved for Track B. Always nil today.
    let partialTail: PartialPagePair?
}

extension RadixTrie {
    /// Walk the trie matching `tokens` token-by-token. Stop at first divergence
    /// or end-of-input. Returns the page-anchored prefix.
    ///
    /// The cvec-state envelope is keyed PER ANCHOR (see §6) so that two
    /// sessions with identical tokens but different active steering envelopes
    /// don't accidentally share K/V. Callers pass a closure that, given a
    /// page-start position, returns the cvec digest for that page — same
    /// signature as today's `Session.cvecDigestForPage`.
    func findLongestPrefix(
        tokens: ArraySlice<UInt32>,
        cvecDigestForPage: (Int) -> UInt64
    ) -> PrefixMatch
}
```

### Cvec handling in the lookup

The cvec digest is consulted **lazily, per anchor** (see §6 for the full discussion). The trie walk is token-only; the anchor adoption check is "does any of this anchor's `[cvecDigest: SharedPagePair]` map contain the digest the caller computed for *this page's* position range?". If yes, take that pair; if no, treat this anchor as not adoptable (we still recurse past it — the *next* anchor might also have an unsteered or matching-steered entry).

This is the key insight: the trie itself is token-only. Cvec is a *partitioning at the anchor leaves*. Walking past an anchor whose cvec-set doesn't contain our digest is *not* a divergence — the deeper trie might have an anchor that *does* contain it (in practice this won't happen often, but the data structure permits it).

### Ref-counting

`findLongestPrefix` calls `pageManager.incref` on every returned page (as today). Caller MUST balance with `decref` on session teardown (today's pattern, unchanged). The trie itself holds **no** ref on page-pairs — it stores phys page indices, and the page-pairs survive in `PageManager.contentIndex` as long as PageManager's LRU hasn't reclaimed them.

When PageManager evicts a page-pair (via `allocFresh`'s forced-eviction path), it must call back into the trie to invalidate the corresponding anchor. See §5 for the callback mechanism.

---

## 4. Insertion API

```swift
extension RadixTrie {
    /// Publish "I've extended page P of prefix [tokens[0..N])". Idempotent.
    /// Internally walks/extends the trie, creating intermediate edges and
    /// the final anchor.
    ///
    /// pageIndex: 0-based slide-page number (P).
    /// tokensCoveringFullPrefix: ArraySlice<UInt32> of length (P+1)*PAGE_SLIDE
    ///   covering [0, (P+1)*PAGE_SLIDE) of the session's prefix.
    /// pair: the SharedPagePair this anchor publishes.
    /// cvecDigest: per-page cvec state for [P*PAGE_SLIDE, (P+1)*PAGE_SLIDE).
    func insertAnchor(
        tokensCoveringFullPrefix: ArraySlice<UInt32>,
        pair: SharedPagePair,
        cvecDigest: UInt64
    )
}
```

### When does insertion happen

Same trigger as today: `promoteFinishedPages` (lm_engine.swift:1816) fires every time `s.position` crosses a `PAGE_SLIDE` boundary. The loop body changes from:

```swift
let hash = PageManager.hashPage(s.consumedTokens[0..<end], cvecDigest: digest)
pageManager.promotePair(slidePrimary: ..., fullSibling: ..., contentHash: hash)
```

to:

```swift
let pair = SharedPagePair(slidePrimary: ..., fullSibling: ...)
pageManager.promotePair(pair)   // unchanged — page manager still owns the phys-page state
trie.insertAnchor(
    tokensCoveringFullPrefix: s.consumedTokens[0..<end],
    pair: pair,
    cvecDigest: digest
)
```

The single FNV hash call is replaced by a trie walk of `end` tokens. For end=64 (typical first promotion) this is ~64 hashmap lookups instead of one FNV-mix. The cost is small in absolute terms (~3 µs) but worth noting because `promoteFinishedPages` is called every AR step at the page boundary. Mitigation: the trie walk for *subsequent* anchor insertions on the same session can start from a cached "last insertion cursor" — we already know we walked to depth `P*PAGE_SLIDE` last time and just need to extend by 16 more tokens. This is a minor optimization; defer until profiling shows it matters.

### Mid-prefill vs. teardown insertion

Insertion happens **mid-prefill**, exactly as today (`promoteFinishedPages` runs after every prefill tile). The trie sees anchors appear in order as the session progresses. There is no special teardown logic — at teardown we decref pages and the trie's anchors stay valid (pointing at now-refcount-0 page-pairs that PageManager still has in `contentIndex`).

### Interaction with Track B (partial-page promotion)

When Track B lands, `promoteFinishedPages` will gain a new path that publishes a *partial* page when a session ends with `s.position % PAGE_SLIDE != 0` (today such tails are silently lost). The new `insertAnchor` signature can accept either a full `SharedPagePair` or a `PartialPagePair`; the trie stores both kinds at anchor leaves, and `findLongestPrefix` returns the partial in `partialTail` when the deepest matched anchor is partial. Anchor placement constraint relaxes from "every 16 tokens" to "every 16 tokens OR at end-of-prefix" — small, additive change.

---

## 5. Eviction strategy

### Two-tier ownership

- **PageManager owns physical pages.** LRU on `pages[p].lastAccessTick`. Eviction happens in `allocFresh` when the free list is all-cached and we need a page.
- **Trie owns navigation structure.** No physical pages; just nodes pointing at phys page indices.

When PageManager evicts a page-pair, the trie has a stale anchor (its `slidePrimary`/`fullSibling` indices now point at pages with different content). Two solutions:

### Solution: callback on phys-page eviction

`PageManager.allocFresh`, on the forced-eviction path, currently does:

```swift
contentIndex.removeValue(forKey: h)
```

We replace this with a callback to the trie:

```swift
onPageEvicted?(physPage, hash)   // delegate
```

The trie's callback walks the path that *would* have led to that anchor (we keep a back-pointer: `pages[p].owningTrieAnchor: TrieAnchor?`). It removes the anchor; if the anchor's parent is now a single-child internal node, it edge-merges; if the anchor's parent has no other children and is also not an anchor, it prunes upward. This maintains the invariant "every trie leaf is an anchor pointing at a live page-pair".

Cost: per-eviction trie cleanup is O(depth_of_evicted_anchor + edge_merge), worst case O(tokens_in_prefix). For a 1000-token prefix, ~1000 hashmap ops + a few merges = ~50 µs. Eviction happens once per `allocFresh` *when the pool is full*, so the amortized cost across all `allocFresh` calls is much lower.

### Stale-leaf cleanup

With the callback above, stale leaves are eagerly cleaned at the moment of eviction. **No background sweep needed.** This is a major simplification vs. the alternative ("trie has its own LRU, periodically scan for stale entries"). Eviction is push-driven, not pull-driven.

### Refcounts on active sessions

Today: a session holds refcounts on its `ownedPages` (the phys pages); decrements on teardown; PageManager preserves the `contentIndex` entry even at refcount=0 (allowing later cache hits as long as the page isn't reclaimed). This continues to work.

In the trie design, **the trie nodes themselves are NOT refcounted by sessions**. A session refcounts physical pages (via PageManager); when those pages get reclaimed for fresh allocation, the eviction callback unlinks the corresponding trie anchor. There is no session ↔ trie-node refcount. This is simpler than SGLang's design (where active sessions "lock" trie nodes) because our two-tier separation does the same work via the PageManager refcount.

**Important consequence:** an anchor stays in the trie *exactly as long as* the underlying page-pair is in `PageManager.contentIndex`. The two go together. This means *adopting* a cached prefix via `incref` resurrects the pages from the free list but doesn't change anything about the trie — the anchor was always there. Perfect.

---

## 6. Steering-aware (cvec_digest) integration

Three options, evaluated:

### (i) Per-trie-node cvec annotation

Branch the trie at the point where a steered session diverges from an unsteered one. Edges become `(token, cvec_state)`-keyed.

**Rejected.** Cvec state is per-*page*, not per-token. A 47-token-long edge in an unsteered prefix would need to split at every position where some steered session has a different envelope. The branching factor blows up; the trie no longer enjoys edge compression. Memory cost scales with the *product* of unique token-paths and unique cvec-configurations.

### (ii) Per-page-pair cvec partition at anchors

Trie walk is **token-only**. At each anchor, we keep a small map:

```swift
class TrieAnchor {
    var byCvecDigest: [UInt64: SharedPagePair] = [:]  // typical size: 1
    // ...
}
```

Lookup procedure:
1. Walk the trie matching tokens. At each anchor we pass through, check whether `byCvecDigest[caller_digest_for_this_page]` is non-nil; if yes, this anchor contributes a page-pair to the result.
2. If `byCvecDigest` lacks the caller's digest, the anchor is not adoptable but we **keep walking** — deeper anchors might match (or might also miss; the answer is the deepest *adoptable* anchor).
3. Return the deepest adoptable anchor's page-pair plus all shallower adoptable anchors' page-pairs (in order). If anchor at depth 32 is adoptable and anchor at depth 48 is not, the answer is 32. If anchor at 32 is not but 48 is, the answer is 48 — but in practice this won't happen (cvec digest is per-page; if envelope intersects pages 0 and 1, it intersects all pages they share; the digest stays consistent across the prefix once steering activates).

**Picked.** The common case (unsteered sessions, `digest == 0`) is one entry in the map per anchor; lookup overhead is ~one hashmap lookup. The structural cost of cvec partitioning is paid exactly where it lives — at the anchor — without polluting the navigation trie.

### (iii) Separate trie per cvec_digest

Rejected on memory cost. We'd duplicate the entire navigation structure per unique steering configuration.

### Edge case: cvec changes mid-prefix

If a session's active controls change so that an envelope starts at position 100 (page 6, 7, ...) but not before, then:
- Anchors at pages 0–5 use the caller's `digest_for_page_p == 0` (no overlap with the envelope window) → they match the unsteered entry.
- Anchors at pages 6+ use `digest_for_page_p == <non-zero>` → they need a separate entry in the byCvecDigest map.

Today the same logic applies because `cvecDigestForPage` already handles per-page intersection (`lm_engine.swift:709` and the digest construction at `lm_engine.swift:386–479`). Behavior is preserved; the trie just stores the per-page result at each anchor.

**Could we drop cvecDigest entirely in favor of per-leaf cvec-state tags?** See §9 — yes, this is a useful follow-up but should not be co-mingled with this refactor.

---

## 7. Migration plan

The whole point of this structure is that the **trie can run alongside the existing dict**. Both are populated from `promoteFinishedPages`; both are queried from `adoptSharedPrefixPages` (the trie's answer is checked against the dict's; discrepancies are logged but not fatal).

### Step 1: Shadow mode (PR #1)

- Add `RadixTrie` class. Add `LmEngine.prefixTrie: RadixTrie`.
- In `promoteFinishedPages`, after the existing `pageManager.promotePair(...)` call, also `prefixTrie.insertAnchor(...)`.
- In `adoptSharedPrefixPages`, run BOTH lookups. Caller still uses the dict's answer. Log discrepancies (trie says match-length L_t, dict says L_d; if L_t != L_d, log session id + token prefix). Add a counter; expose in engine-state telemetry.
- **Pass criteria before advancing:**
  - All existing tests pass (`LM_TEST_CVEC_DIGEST=1 ./forward_graph`, `LM_TEST_CACHE_DIVERGENCE=1 ./forward_graph`, `LM_TEST_CVEC_CACHE=1 ./forward_graph` per `docs/CVEC_AND_PREFIX_CACHE.md`).
  - 24-hour bridge soak with realistic traffic shows zero `L_t != L_d` discrepancies for the page-aligned case. (Token-granular extra matches are *expected* and not an error — they're the whole reason for this refactor; suppress from the discrepancy counter when `L_t > L_d AND L_t < L_d + PAGE_SLIDE`.)
  - New test: synthetic two-session test where session B has a tokens-but-not-page-aligned prefix of session A; confirm trie reports `trieMatchLength > alignedMatchLength` while dict reports `alignedMatchLength`. Assert telemetry counters update accordingly.

### Step 2: Read through trie (PR #2)

- `adoptSharedPrefixPages` consults the trie only. Dict still populated by `promoteFinishedPages` (for safety / rollback ease).
- Eviction callback wired up (`PageManager.allocFresh` → `prefixTrie.invalidateAnchorFor(physPage:)`).
- **Pass criteria before advancing:**
  - One-week production soak with shadow-mode discrepancy counter staying at zero.
  - New test: a session adopts pages, those pages get evicted by allocFresh pressure, a follow-on session's lookup correctly returns the *new* state (deeper hit if there's a longer trie path elsewhere; otherwise no false hit).
  - End-to-end MSE divergence test (`LM_TEST_CACHE_DIVERGENCE=1`) confirms K/V remain bit-exact under trie-driven adoption.

### Step 3: Retire the dict (PR #3)

- `PageManager.contentIndex` deleted. `PageManager.findByHash` deleted. `PageManager.promotePair` no longer takes a `contentHash`; signature becomes `promotePair(_ pair: SharedPagePair)` recording just the phys pages + their pair relationship.
- The hash field on `PageInfo` is kept (it's used in `livePageSnapshot` for the visualizer); the trie's anchor invalidation callback receives the phys page index, not a hash, so this field is now write-only for visualization purposes. Consider deleting in a follow-up.
- **Pass criteria:**
  - One-week soak after Step 2 with no rollback events.
  - All cvec + cache integration tests pass.

### Optional Step 4: Drop cvec from the engine

See §9 — coordinate with Track C if/when it lands.

---

## 8. Code touchpoints estimate

| File | LoC delta | What changes |
|---|---|---|
| **NEW** `/Users/mdot/metal-microbench/radix_trie.swift` | +400 to +600 | The trie itself: `RadixTrie`, `TrieNode`, `TrieAnchor`, edge compression, split logic, insertion, lookup, invalidation callback. |
| `/Users/mdot/metal-microbench/page_manager.swift` | +30 / −40 (Step 3) | Add `onPageEvicted` callback wire-up (+10). At Step 3, delete `contentIndex`, `findByHash`, `promotePair`'s hash arg, content-index-mutation in `allocFresh` (−40). Net −10. |
| `/Users/mdot/metal-microbench/lm_engine.swift` | +120 / −60 | `adoptSharedPrefixPages` rewritten to call `prefixTrie.findLongestPrefix` and translate the result. `revisitCacheProbe` mostly unchanged (calls the same `adoptSharedPrefixPages`). `promoteFinishedPages` adds a trie insertion. `LmEngine` gains a `prefixTrie` field + constructor. Shadow-mode discrepancy logging adds ~50 lines during Step 1, removed in Step 3. |
| `/Users/mdot/metal-microbench/ffi_batch.swift` (`gemma_engine_state`) | +20 | New JSON fields under `kv_cache` for trie stats: `trie_nodes`, `trie_anchors`, `trie_depth_p50`, `trie_depth_p99`, `discrepancy_count` (Step 1 only). |
| `/Users/mdot/metal-microbench/server/bridge.py` | 0 | Bridge passes JSON through verbatim. No code change. New fields appear automatically. (Optional: update `docs/engine_telemetry_endpoint.md` documentation, +20 lines, no code.) |
| `/Users/mdot/metal-microbench/kv_visualizer.swift` | +0 / +30 | The visualizer's hash helper at lines 209–216 can stay (the visualizer doesn't need the trie). If we want it to *show* trie structure, add ~30 lines for a per-anchor depth annotation; not required for correctness. |
| `/Users/mdot/metal-microbench/docs/CVEC_AND_PREFIX_CACHE.md` | +40 | Document that the cache is now token-granular at lookup, with per-anchor cvec partition. |
| **NEW** tests in `/Users/mdot/metal-microbench/docs/tests/` (or wherever the existing trie tests live) | +200 | Unit tests for `RadixTrie`: insert/lookup/split, eviction callback, cvec partition at anchor, edge-compression invariant. |

**Total: ~+800 to +1000 added, ~−100 deleted across three PRs.**

---

## 9. Dependencies on other tracks

### Track A — backstop removal

The "backstop" refers to the `submit()` / `revisitCacheProbe` code that unadopts the trailing page when the entire prefill has been cache-hit so there is ≥1 token for the prefill-then-sample path to run on (`lm_engine.swift:979–988` and `1094–1104`). This backstop exists because the post-prefill sampling path only fires when `stepPrefillForSession` actually runs.

**The trie refactor does NOT require Track A.** The backstop logic is per-session bookkeeping on `ownedPages`/`promotedPageCount` after the lookup returns; it doesn't care whether the lookup answer came from a dict or a trie. **However**: once Track A lands (presumably by routing the "fully cached" case through a dedicated "no-prefill sample" code path), the backstop's `while skipPrefix * PAGE_SLIDE >= tokens.count` unadopt-loop disappears, which makes the trie lookup result cleaner to consume (no post-processing trimming). Lower friction, but not a dependency.

### Track B — partial-page promotion

The trie's anchor model is designed to ingest partial pairs from day one. The `TrieAnchor` struct should be declared with a single field that holds *either* a full pair OR a partial pair (enum/tagged union; default is full). When Track B lands:
- `promoteFinishedPages` gains a path that publishes the last partial page on session teardown.
- `insertAnchor` accepts the partial-pair variant.
- `findLongestPrefix` populates `PrefixMatch.partialTail` when the deepest matched anchor is partial.
- The caller (`adoptSharedPrefixPages`) gains a path to install the partial-pair K/V on the session's leading page.

**API stability:** the `findLongestPrefix` return type already has `partialTail: PartialPagePair?`. The signature is forward-compatible.

### Track C — cvecDigest tightening

There is a real opportunity here. Today, cvecDigest is mixed into the hash *and* the digest construction includes envelope parameters that, post-hoc, we know don't affect K/V values bit-identically — e.g., `transportScale`/`transportOffset` affect *measurement* but not *application*. The digest currently lumps them in defensively.

In the trie design, we can replace `cvecDigest: UInt64` with a richer `CvecAnchorTag` struct stored at the anchor:

```swift
struct CvecAnchorTag: Hashable {
    let layerLayerSetHash: UInt64    // which layers see steering
    let envelopeShapeHash: UInt64    // attack/decay/sustain/release/peak/shape
    // omit transportScale/transportOffset/target — measurement-only
}
```

Lookup keys the anchor's `byCvecAnchorTag` map by this struct instead of the precomputed UInt64. Conceptually cleaner; lets us *drop* fields from the tag once we verify they don't affect K/V. Practically the change is small (replace one type with another).

**Coordination with Track C:** if Track C tightens the digest by removing fields, the trie's `CvecAnchorTag` can be updated in lockstep. If Track C lands first, the trie design ingests the tightened tag directly. **Not a blocker either way.**

### Track E — rename pass

New types introduced: `RadixTrie`, `TrieNode`, `TrieAnchor`, `PrefixMatch`. The current names `SharedPagePair`, `PageManager.contentIndex`, `promoteFinishedPages`, `adoptSharedPrefixPages` are perfectly fine and should be kept. The natural rename touchpoints if Track E is happening:
- `contentIndex` is dead at Step 3; no rename needed.
- `promoteFinishedPages` and `adoptSharedPrefixPages` continue to make sense (they describe what the engine does, not the underlying data structure).
- `SharedPagePair` is unchanged.

**Coordinate with Track E only on the new type names.** Submit those names in the radix PR description and let Track E review before Step 1 lands. No back-and-forth expected.

---

## 10. Honest sizing

**This is a ~1000-line refactor across three PRs, spread over ~3 weeks.**

(NOTE — superseded under D-carrier execution. In the multipolar
subagent format we collapse the 3 PRs to 1 integrated change with the
test harness as the safety net.)

Breakdown:
- **PR #1 (Shadow mode):** ~700 lines added (the trie itself + insertion in `promoteFinishedPages` + shadow logging). 0 lines deleted. Carries the most risk: introduces a brand-new data structure, requires careful unit tests, requires soak time before advancing. Plan for ~1.5 weeks elapsed (1 week of dev, 0.5 week of soak).
- **PR #2 (Read-through):** ~150 lines added (eviction callback wiring), 50 deleted (shadow logging dropped, dict-read paths removed). Carries the second-most risk: switches the engine to trust the trie. Plan ~1 week elapsed (3 days dev, 4 days soak).
- **PR #3 (Retire dict):** ~50 lines deleted, no additions. Cleanup. Low risk; plan ~3 days elapsed.

**Should it ship in one PR?** No. The three-step shadow → read-through → retire pattern is the lowest-risk path for a refactor that touches the cache. The shadow-mode discrepancy counter is the safety net. If the trie has a bug we missed in tests, PR #1's logs will show it before any user traffic is affected.

**Should the three PRs ship contiguously?** Yes — there is no value in pausing between them once Step 1 has soaked. Lining them up keeps the cognitive load on the reviewer high but bounded.

### Why not larger

Because the surrounding architecture is *exactly* right for this refactor:
- PageManager already separates phys-page ownership from content indexing.
- The cache key (`hashPage(tokens, cvecDigest)`) is already a pure function of inputs.
- Insertion (`promoteFinishedPages`) and lookup (`adoptSharedPrefixPages`) are isolated functions, not scattered call sites.
- Engine-state telemetry is JSON; adding new fields is mechanical.

### Why not smaller

Because edge-compressed trie + cvec partitioning + eviction callback + extensive testing is not a 200-line job. The data-structure code itself is 400+ lines once you cover edge-split correctly. Tests are another 200. Telemetry, integration into the engine, and shadow-mode infrastructure add the rest.

---

## Critical Files for Implementation

- /Users/mdot/metal-microbench/page_manager.swift
- /Users/mdot/metal-microbench/lm_engine.swift
- /Users/mdot/metal-microbench/ffi_batch.swift
- /Users/mdot/metal-microbench/docs/CVEC_AND_PREFIX_CACHE.md
- (NEW) /Users/mdot/metal-microbench/radix_trie.swift
