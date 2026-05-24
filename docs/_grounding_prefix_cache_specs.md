# Prefix-cache / engine design specs — grounding document
# Sources: docs modified 2026-05-23

---

## 1. `docs/CVEC_AND_PREFIX_CACHE.md`

**Filepath:** `/Users/mdot/metal-microbench/docs/CVEC_AND_PREFIX_CACHE.md`

**Scope.** Specifies the contract between the control-vector (cvec) subsystem and the KV prefix cache. Covers what `measure` vs. `apply` operations do to the residual stream and how cvecDigest partitions cache keys.

**Problem statement.** The doc identifies two "things you CAN'T get back":
> "Past-token residual values through adopted pages. The cache stores K/V, not pre-attention residuals. Re-prefill to measure."
> "Effector magnitude at adopted positions. Recorded per-token in `pendingSamples` only for positions that were *freshly* prefilled or AR-decoded by this session. Cached positions don't re-record."

It also documents a known-benign invariant: AR-path and prefill-path kernels produce **~9% nDiff fp16 K/V on bit-identical inputs** by reduction-order alone.

**Proposed design.** The cache keys each 16-token slide page on:
> "`hashPage(tokens, cvecDigest)` where `cvecDigest` is an FNV digest of the *parameters* of every `ActiveControl` whose envelope intersects that page's position range (layer, cvecId, shape, attack/decay/sustain/release, peak, polarity, start-offset)."

The `.primed` recover-step routes single-page prompts (≤16 tokens) through the **prefill** kernel at `qLen=1` to achieve bit-identical producer/consumer K/V. A unified-kernel rewrite making the AR kernel use simdgroup MMA was:
> "**scoped 2026-05-23 and rejected**. The simdgroup tile geometry is ~3% utilized at activeB=1 (NR1=32 → 1 row), so the rewrite would regress AR decode speed for fp-rounding-magnitude correctness gain."

**Invariants stated.**
- A `measure` operation (`measure_dot_fp16`) never affects cache keys; `DetectorAttachment` is not part of the digest.
- Measurement only fires during live forward passes; adopted/cached positions are NOT revisited.
- Every distinct envelope configuration partitions the cache; adoption of an unsteered session's pages by a steered session is prevented at the hash level.
- Adoption is in pairs: each 16-token slide page promotes two phys pages (slide primary + full-attention sibling).
- > "A session that adopts N slide pages gets `2N` phys pages, and the intervention is recoverable at full precision across cache boundaries."
- `allocFresh` invalidates both pair members when either is reclaimed.

**Acceptance criteria.**
- `LM_TEST_CVEC_DIGEST=1 ./forward_graph` — unit tests on digest + hashPage composition, no model weights needed.
- `LM_TEST_CVEC_CACHE=1 GGUF_PATH=... ./forward_graph` — cvec partitioning + intervention preservation.
- `LM_TEST_CACHE_DIVERGENCE=1 GGUF_PATH=... ./forward_graph` — per-layer MSE dump; any cache-replay regression surfaces as first layer with MSE > 1e-4.
- Adoption is bit-exact: MSE=0 verified end-to-end across 30 layers.

**Risks / open questions.** The AR-vs-prefill K/V divergence (~9% nDiff) is documented as a known-benign invariant; the `test_KL_adopted_vs_fresh_first_token` harness confirms statistical equivalence at the sampling-distribution level. No open questions flagged—the doc is in steady-state.

---

## 2. `docs/prefix_cache_grand_slam_2026_05_23/README.md`

**Filepath:** `/Users/mdot/metal-microbench/docs/prefix_cache_grand_slam_2026_05_23/README.md`

**Scope.** Operator doctrine and execution plan for the five-track grand slam. Specifies that Track D is the carrier refactor and Tracks A/B/C/E fold into D's shape.

**Problem statement.**
> "A/B/C/E in isolation would produce code that D throws away (`PageManager.contentIndex`, `findByHash`, the `hashPagePrefix` wrapper, the backstop's bookkeeping arithmetic, `cvecDigest: UInt64` as a primitive). Doing them first pays the rename cost twice."

**Proposed design.** Track D is the carrier; A/B/C/E fold in:

| Track | Folded form |
|---|---|
| A — backstop removal | `.primed` SessionState refinement + recover-tick at any adopted position |
| B — partial-page promotion | `TrieAnchor.partialTail: PartialPagePair?` + CoW-on-extend; true 1-token granularity |
| C — cvecDigest tightening | `CvecAnchorTag` struct stored at each anchor |
| E — renames | Only surviving-code renames: `chunkQueue → primingQueue`, `promotedPageCount → myPromotedSlidePairCount`, `pendingPrimingCount → pendingPrimingTokenCount`, `SharedPagePair → SlidePairContents` |

Execution order: test harness first (gating), then 4-subagent parallel implementation fan-out, then main-agent integration, then surviving-code renames, then full validation.

**Invariants stated.** KL-divergence guard against fresh-compute: if KL stays < 1e-5 across a representative prompt corpus, high confidence.

**Acceptance criteria.**
- Full existing test suite: `LM_TEST_CVEC_DIGEST`, `LM_TEST_CACHE_DIVERGENCE`, `LM_TEST_CVEC_CACHE`.
- End-to-end ST test against live bridge.
- Bridge smoke: 2 identical 16-token curls → second reports `cache_hits=16`.
- Cold-prefill 107 tok/sec investigation (task #216) runs naturally at this step.

**Risks / open questions.**
> "D is a wholesale rewrite of the cache's lookup semantics. The test harness mitigates this but doesn't eliminate it — a sufficiently subtle correctness bug could survive even thorough tests."

---

## 3. `track_a_backstop_removal.md`

**Filepath:** `/Users/mdot/metal-microbench/docs/prefix_cache_grand_slam_2026_05_23/track_a_backstop_removal.md`

**Status note:** Folded into Track D as the `.primed` SessionState refinement (option c').

**Scope.** Specifies the removal of the backstop in `Session.submit()` / `revisitCacheProbe` that currently unadopts the trailing slide page when all tokens would otherwise be fully cached.

**Problem statement.**
> "Two bit-identical 16-token prompts: the second submits 16 tokens, adopts page 0, then the backstop sheds it (because `1 * 16 >= 16`). Net: `cache_hits = 0`, `cache_misses = 16`. This is the smoking gun reproducer."
> "Any prompt whose length is an exact multiple of `PAGE_SLIDE` loses its tail page to the backstop."

**Proposed design.** The chosen option is **(b)** (1-step AR tick at the adopted position), with a TODO to refactor into option **(c')** (first-class `.primed` state) after Track E lands.

Option b: when adoption leaves `chunkQueue` empty (`position > 0 && chunkQueue.isEmpty && state == .priming`), run a recover tick at `position - 1` with `consumedTokens[position - 1]` as input. The K/V write at `position - 1` lands on a shared page and produces bit-identical bytes (provably a no-op):
> "The K/V write at `position − 1` lands in a shared page that the consumer has incref'd. The write produces bit-identical bytes (same input, same KV[0,k-2], same cvecs). The PageManager doesn't have CoW logic and would not detect the overlap. **The write goes through and is benign.**"

Option c' (the long-term shape): a 4th `SessionState` `.primed` meaning "all prefill complete, K/V at position N exists, but no first generated token has been sampled."

**Invariants stated.**
- The `needsRecoverStep` condition is `position > 0 && chunkQueue.isEmpty && state == .priming`, not "page-aligned position" — must work at any position once Track D introduces token-granularity.
- Performance: trades one AR tick (~34ms) for skipping N-1 prefill tokens; net positive for N ≥ 2.

**Acceptance criteria.** Twelve named tests (R1–R12), including:
- R1: smoking-gun reproducer — second 16-token identical prompt: `cacheHitTokens == 16`, `cacheMissTokens == 0`.
- R2/R3: 32-token and all page-multiples {16, 32, 48, 64, 128, 256}: all hits, zero misses on second submission.
- R7: recover-step token-frequency histogram matches fresh-prefill histogram within sampling noise.
- R11: after triple-submission stress, `pagesInUse == 0` (no leak).
- Add `LM_KV_OVERWRITE_CHECK` debug assertion that rehashes the page post-tick and compares to pre-tick content hash.

**Risks / open questions.**
- The benign shared-KV overwrite is the one structural wrinkle; must be defended with a code comment + env-var assertion.
- Track B compatibility: the recover-step owner must confirm the `needsRecoverStep` flag gets set regardless of adoption entry point when Track B introduces a new path.

---

## 4. `track_b_partial_page_promotion.md`

**Filepath:** `/Users/mdot/metal-microbench/docs/prefix_cache_grand_slam_2026_05_23/track_b_partial_page_promotion.md`

**Status note:** Folded into Track D as `TrieAnchor.partialTail: PartialPagePair?` with true 1-token granularity.

**Scope.** Specifies caching of in-flight partial slide pages (sessions that end with `position % PAGE_SLIDE != 0`). Currently those trailing tokens are silently lost.

**Problem statement.** The pair invariant requires both members of a `(slidePrimary, fullSibling)` pair:
> "Promoting only one member violates the invariant for any read at full-positions `[P*16+8, P*16+15]` — the full layers' attention dispatcher reads `block_table[2P+1]`, finds whatever zeros/stale data live there, computes garbage (`page_manager.swift:60-68` documents the empirical `KL ~ 0.38`)."

Today, a 15-token prompt: first session prefills 15 tokens, closes — 0 tokens cached. Second session re-prefills all 15.

**Proposed design.** Under D-carrier, true 1-token granularity via option **(b)** (partial-pair with CoW). Standalone recommendation was hybrid (c)+(a): 8-token sub-granularity as the base, flush-prefill for the 1–7 token remainder.

Option b data structure: `SharedPagePair.validUpTo: Int` (1..16), with CoW on adoption of partial pairs:
> "**adoption of a partial pair must allocate a fresh page and copy** (CoW). Or: **adopt-then-fork-on-extend** — keep the shared page read-only, allocate a fresh page when the consumer's prefill needs to write into it, copy the `validUpTo` prefix bytes into the fresh page, then prefill the rest into the fresh page."

CoW copy cost: ~tens of microseconds total across all layers — well below a single prefill step (~30 ms at qLen=16).

Adoption returns `(slidePagesAdopted: Int, partialTailTokens: Int)`; caller computes `position = slidePagesAdopted * PAGE_SLIDE + partialTailTokens`.

**Invariants stated.**
- `adoptSharedPrefixPages` must set `position = validUpTo` for a trailing partial pair (not `(adopted+1) * PAGE_SLIDE`).
- Concurrent adoption of a partial pair is impossible without CoW — once `decref` lands a page on the free list, any adopter can `incref` it; CoW is mandatory.
- Track B requires Track C's `cvecDigestForRange(start:end:)` to compute the digest over an arbitrary sub-range: `func cvecDigestForRange(start: Int, end: Int) -> UInt64`.

**Acceptance criteria.**
- 15-token prompt round-trip: adopter sets `position = 8` (half-pair), prefills 7 tokens; reports 8 hits, 7 missed.
- KL divergence target for partial-promotion case: KL < 1e-5 averaged across positions; max KL < 1e-3.
- Track A is a prerequisite to extract full value from Track B for 100%-cached prompts; without Track A, Track B still helps partial cache hits (most real cases).

**Risks / open questions.** Track B + Track A interaction: without Track A, the backstop still unadopts the trailing partial pair on 100%-cached prompts, negating the benefit of full flush. Option (b) alone is blocked on Track A for the fully-cached case.

---

## 5. `track_c_cvec_digest_tightening.md`

**Filepath:** `/Users/mdot/metal-microbench/docs/prefix_cache_grand_slam_2026_05_23/track_c_cvec_digest_tightening.md`

**Status note:** Folded into Track D as `CvecAnchorTag` struct at each `TrieAnchor.byCvecAnchorTag` map.

**Scope.** Audit of `computeCvecDigest` (`lm_engine.swift:385-480`) identifying inputs that over-partition the cache key without corresponding K/V differences.

**Problem statement.**
> "Over-partitioning case: if the page lies entirely in sustain (i.e., `pageStart − startPosition ≥ attack+decay` AND `pageEnd ≤ stopPosition`, units=tokens), `magnitudeAt` returns `sustain = peak·sustainLevel` for every position regardless of `startOffset`."
> "`startTurn` [is] **Always mixed unconditionally**, even when `units == .tokens` — `startTurn` is dead bits for token-units envelopes."
> "Predicted hit-rate uplift on cached prefix length: **0% → ~(L - (attack+decay)) / L** where L = prompt length in pages. For typical envelopes (`attack ~ 4 tokens`, `decay ~ 4 tokens`, page = 16 tokens), only the first page is uncacheable; pages 2…N hit. For a 10-page prompt, this is **0% → ~90%**."

**Proposed design.** Five concrete changes to `computeCvecDigest`:
1. **Drop redundant `startPosition`/`startTurn`**: keep only `startOffset` when the attack/decay ramp overlaps the page.
2. **Phase-gate envelope params**: include attack/decay/sustain/release/shape only when the corresponding phase overlaps the page.
3. **Units-gate anchors**: include `startPosition` for tokens-units, `startTurn` for turns-units — never both.
4. **Quantize floats**: 16.16 fixed-point (`quantF32`) replacing raw bitpatterns for transport coefficients.
5. **Skip near-zero-magnitude controls**: closed-form peak over the page; if below `EPS_MAG`, skip the control entirely.

New helpers: `peakMagnitudeOverPage`, `phasesOverPage`, `quantF32`. New type `PageEnvPhase: OptionSet` with cases `.attack, .decay, .sustain, .release`.

**Invariants stated.**
- The tighter digest must NOT produce false hits. Risk analysis documents each proposed change:
  - Phase-gating sustain pages: **safe** (magnitudeAt returns same constant scalar in sustain; page index identical across sessions).
  - Float quantization (Q16.16, ~1e-4 tolerance): **only non-trivial false-hit risk**; residual delta ≤ ~5e-4 per dim, well within fp16 noise.
  - Peak-magnitude-over-page skip at `EPS_MAG = 1e-3 * peakMagnitude`: **safe** (perturbation ≤ fp16 noise floor; kernel's own `m == 0.0f` short-circuit already does this at exact zero).
- Track C signals to Track D: design the trie node structure assuming consecutive-page digest equality is the common case (long sustain runs share digests page-after-page).

**Acceptance criteria.** Twelve named unit tests including:
- Sustain-phase invariance to `startPosition`: pages in sustain produce equal digest for different start positions.
- Float tolerance: `transportScale=1.0` vs. `1.0 + 1e-7` produce equal digest.
- Units gating: turns-units envelope varying `startPosition` → equal digest.
- Decayed-to-zero: `sustainLevel=0`, no `stopPosition`, page far past `attack+decay` → digest equals zero (as if control absent).
- Integration: post-fix, two sessions with identical tokens and matching cvec config but one single-shot and one built via `continue` calls should share pages for sustain-phase pages.

**Risks / open questions.** Float quantization of transport coefficients is the only proposed change with non-trivial false-hit risk. Recommendation: restrict to transport mode only or choose tighter tolerance. Turns-based `startTurn` rebasing against `currentTurn` flagged as needing clarification in the proposed pseudocode.

---

## 6. `track_d_radix_trie.md`

**Filepath:** `/Users/mdot/metal-microbench/docs/prefix_cache_grand_slam_2026_05_23/track_d_radix_trie.md`

**Scope.** The carrier refactor: replace `PageManager.contentIndex: [UInt64: SharedPagePair]` with a token-granularity radix trie, matching SGLang's RadixAttention semantics. All other tracks fold into this shape.

**Problem statement.**
> "Our cache is a **degenerate block-level APC** … Lookup quantizes to block boundaries. The data structure is flat; there is no trie. To share a 7-token suffix of one prefix with another prefix that has the same 7 tokens, [current design] cannot — the hashes diverge."
> "Token-granularity lookup gets us hit rates on multi-turn chat where a tool-call result is only 7 tokens long (today: 0 tokens of that turn's prefix get cached; with a trie: all 7 of those tokens contribute to future hits when the next turn extends them)."

Lookup performance: current O(L²/16) for L-token prompts → trie O(L). At L=10K:
> "**Trie is 100× faster at this scale.** Today's amortized: O(L/16) FNV chains × O(L) per chain = O(L²/16). 50 ms for L=10K."

**Proposed design.** Chosen structure: **(c) token-trie with page-aligned leaves.** Internal nodes exist only for navigation at non-page-boundary positions; anchor nodes exist only at positions divisible by `PAGE_SLIDE` and carry a `SharedPagePair` handle.

Key API types:

```
struct PrefixMatch {
    let alignedMatchLength: Int    // always a multiple of PAGE_SLIDE
    let trieMatchLength: Int       // reported for telemetry / Track B; not used today
    let pages: [SharedPagePair]    // incref'd; caller MUST decref on teardown
    let partialTail: PartialPagePair?  // nil today; reserved for Track B
}
```

Cvec is a **partitioning at anchor leaves** (option ii), not a trie-node annotation:
```
class TrieAnchor {
    var byCvecDigest: [UInt64: SharedPagePair] = [:]  // typical size: 1
}
```
> "The trie itself is token-only. Cvec is a *partitioning at the anchor leaves*. Walking past an anchor whose cvec-set doesn't contain our digest is *not* a divergence."

Eviction: push-driven via `PageManager.allocFresh` callback (`onPageEvicted`), which calls into the trie to invalidate the anchor and edge-merge upward. No background sweep.
> "Stale-leaf cleanup: with the callback above, stale leaves are eagerly cleaned at the moment of eviction. **No background sweep needed.**"

Sessions hold no ref on trie nodes — only on physical pages via PageManager. The trie and PageManager's contentIndex stay in sync through the eviction callback.

Migration (METR-sized 3-PR plan, collapsed to 1 integrated change in multipolar subagent format): Shadow → Read-through → Retire dict.

**Invariants stated.**
- Every trie leaf is an anchor pointing at a live page-pair.
- The trie itself holds no ref on page-pairs; only PageManager does via contentIndex.
- `findLongestPrefix` incref's every returned page; caller MUST decref on teardown.
- `TrieAnchor.partialTail: PartialPagePair?` field reserved from day one (forward-compatible with Track B without API change).

**Acceptance criteria.**
- Shadow-mode: 24-hour bridge soak with zero `L_t != L_d` discrepancies for the page-aligned case.
- Trie-driven adoption: `LM_TEST_CACHE_DIVERGENCE=1` confirms K/V remains bit-exact.
- KL guard: KL < 1e-5 across representative prompt corpus.
- New file: `/Users/mdot/metal-microbench/radix_trie.swift` (~400–600 LoC).
- Net code change: ~+800 to +1000 LoC added, ~−100 deleted.

**Risks / open questions.** D is a wholesale rewrite of cache lookup semantics. The test harness is the primary safety net. The shadow-mode discrepancy counter in Step 1 catches bugs before user traffic is affected.

---

## 7. `track_e_naming_renames.md`

**Filepath:** `/Users/mdot/metal-microbench/docs/prefix_cache_grand_slam_2026_05_23/track_e_naming_renames.md`

**Status note:** Folded into Track D. Under D-carrier execution, only surviving-code renames apply; symbols deleted or signature-replaced by D are skipped.

**Scope.** This track is a naming cleanup. It is short and carries zero algorithmic risk. The doc is thorough but the scope is entirely mechanical.

**Problem statement.** A cluster of naming inconsistencies in the prefix-cache subsystem obscures the geometric vs. semantic distinction (physical KV-block geometry vs. cache-lookup stride) and creates ambiguity about units (phys pages vs. slide pairs vs. pair counts).

**Proposed design.** Surviving-code renames (those that make it through Track D):
- `chunkQueue → primingQueue` (~40 Swift hits, biggest single rename)
- `promotedPageCount → myPromotedSlidePairCount`
- `pendingPrimingCount → pendingPrimingTokenCount`
- `SharedPagePair → SlidePairContents`

Skipped (deleted or signature-replaced by D): `contentIndex`, `findByHash`, `promotePair`, `hashPagePrefix`, `adoptSharedPrefixPages`, `promoteFinishedPages`, `revisitCacheProbe`.

JSON wire field names (`cache_hits`, `cache_misses`, `pair_mate`, `pages_owned`) are **kept** regardless — consumed by static dashboards and recorded test fixtures.

**Invariants stated.** All wire keys (`cache_hits`, `cache_misses`, `cache_hit_tokens`, `cache_miss_tokens`, `pair_mate`, `pages_owned`, `promoted`) preserved on the JSON wire. Recorded SSE chunks in `docs/media/**/sse_chunks.jsonl` must not be rewritten.

**Acceptance criteria.** ~250 LoC across ~15 files; ~80% Swift (compile-checked), ~15% Markdown, ~5% Python. Zero JSON-wire breakage. Zero changes in `/Users/mdot/sillytavern-fork/`.

**Risks / open questions.** The only runtime-risk item is the serializer mapping for `cacheHitTokens → pageAlignedCacheHitTokens` — a wire-key typo silently degrades dashboards to "0 cache hits forever." Mitigation: `server/test_engine_state_keys.py` smoke test asserting the JSON snapshot contains the documented key set, gating PR-3.

---

## Consolidation

### End-to-end story across the five tracks

The five tracks address a single coherent deficiency: **the prefix cache is a flat, page-aligned, over-partitioned hash table that misses cache hits it should get.** Three independent failure modes compound:

1. **The backstop (Track A)** actively un-adopts correctly cached pages when a prompt length is an exact multiple of `PAGE_SLIDE`. A 16-token repeat goes 0-for-16 on cache hits. Track A fixes this by introducing a `.primed` session state that runs one AR recover-tick instead of unadopting.

2. **Page-aligned teardown (Track B)** silently discards partial tail pages when sessions end with `position % PAGE_SLIDE != 0`. A 15-token prompt caches nothing. Track B promotes partial pairs with a `validUpTo` field and CoW semantics on adoption.

3. **Over-partitioned cvec digest (Track C)** causes sessions with the same tokens and effectively identical steering (same sustain plateau, different absolute start position) to miss each other's pages entirely. Track C tightens the digest by phase-gating envelope parameters and dropping dead-bits fields, recovering 0% → ~90% hit rate on typical 10-page sustain-phase prompts.

4. **Flat hash dict (Track D)** caps the lookup granularity at 16-token page boundaries. Two prompts sharing 23 tokens reuse only 16. Track D replaces the dict with a radix trie, enabling O(L) token-granular lookup and 100× faster lookup at long-prefix scale. **Track D is the carrier** — it provides the structural home (TrieAnchor) where Track A's `.primed` state, Track B's `PartialPagePair`, and Track C's `CvecAnchorTag` all live.

5. **Naming gaps (Track E)** are folded into Track D's surviving-code surface. Only renames on symbols that survive the rewrite are applied; symbols deleted by D are not renamed.

The dependency graph: Track A and Track B both deliver more value after Track C reduces digest over-partitioning (fewer misses mean more hits to rescue). Track B's full value for 100%-cached prompts is blocked on Track A. Track D is not blocked by any of A/B/C but benefits from them — the trie's anchor hit rates improve as C tightens partitioning and B adds sub-page anchors.

### Conventions defined that other docs in this cluster depend on

- **`cvecDigest` / `CvecAnchorTag`** — the per-page FNV-1a digest that partitions the cache on the cvec axis. Track C tightens its inputs; Track D promotes it from a `UInt64` scalar to a typed struct stored at each trie anchor. Any doc touching cache adoption depends on this digest being computed over the right input set.

- **Pair invariant** — a `(slidePrimary, fullSibling)` pair is bit-identical to fresh compute iff both members were written by a prefill observing the same token sequence and cvec envelope over the full 16-token range. Promotion or adoption of only one member yields KL~0.38. All tracks respect this; Track B is the only one that relaxes it via `validUpTo`.

- **Partial-page promotion (`validUpTo`)** — Track B's field on `SlidePairContents` indicating how many tokens of the pair's 16-token range are valid. Track C's `cvecDigestForRange(start:end:)` must be range-aware to match. Track D's `TrieAnchor.partialTail: PartialPagePair?` is the structural home.

- **`.primed` SessionState** — Track A's recover-step state. Sessions in `.primed` have all prefill complete but have not yet sampled the first generated token. `prepareARStep` must treat `.primed` like `.generating` for the recover tick.

- **Eviction callback** — Track D's `onPageEvicted` delegate from `PageManager.allocFresh` into the trie. Required for trie correctness; the invariant "every trie leaf is an anchor pointing at a live page-pair" depends on this being called on every forced eviction.

- **Wire-name stability** — `cache_hits`, `cache_misses`, `pair_mate`, `pages_owned` JSON keys are frozen. Any internal Swift rename must map to these keys at the serializer site.

### TODOs surfaced

- Unified-kernel rewrite (AR kernel using simdgroup MMA) rejected 2026-05-23; existing KL guard + per-layer divergence dump are the regression safety net.
- `LM_KV_OVERWRITE_CHECK` debug assertion (Track A) — rehash the shared page post-recover-tick; remove after one month of production stability.
- `server/test_engine_state_keys.py` — new smoke test gating Track E PR-3 for wire-key correctness.
- Track C flag: if Track C tightens cvecDigest to include future-position-conditional evaluations, the adoption rate of the trailing page may change (fewer hits, not wrong correctness).
- Track D: `promoteFinishedPages` insertion loop could use a "last insertion cursor" to avoid re-walking the full prefix on each anchor insertion; defer until profiling shows it matters.
- Cold-prefill 107 tok/sec investigation (task #216) is scheduled to run naturally after full Track D integration.
- `kv_visualizer.swift`: optional ~30-line addition to annotate trie structure per anchor; not required for correctness.
- If Track C decides cvecDigest should additionally cover future positions, coordinate with Track A to ensure the recover-step's adoption condition is not invalidated.
