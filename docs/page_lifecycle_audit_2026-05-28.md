# KV-page lifecycle audit — 2026-05-28

Static audit (no runtime; the engine was throughput-contaminated by a
misplaced `neural-kcut` EDA solver during this session, so runtime validation
is deferred to a quiet machine). Three parallel auditors covered: (A) page
alloc/refcount/free, (B) page reclaim vs in-flight GPU use, (C) prefix-cache
reuse correctness. Motivating question (operator): *"there might not be a
validated lifecycle of reusing pages."*

## Verdict on the motivating question

**There IS a reuse lifecycle, and the current radix-trie design (post
2026-05-23 refactor) is materially sounder than the flat-`contentIndex`
design the KV-correlation docs (2026-05-08) diagnosed** — refcount-gated
sharing, eviction-driven trie-anchor invalidation, CoW for partial
divergence, cvec partition tags. **But it is not a *validated* lifecycle:**
`page_manager.swift` contains **zero** `assert`/`precondition`; every safety
property (no-UAF, position-alignment, shared-page immutability) is an
*implicit global invariant* held by construction, not a locally-enforced
contract. The planned refactors (lock-splitting; non-zero-offset adoption)
would silently break them.

## Reassuring (NOT bugs) — confirmed sound on the normal path

- **No active use-after-free on the serving path.** KV-touching CBs are
  synchronous (`syncTickStep` commits + `waitUntilCompleted` back-to-back,
  async chain deleted 2026-04-26); all `closeSession`/`decref` run under the
  single `gEngineLock` between CBs. Backing KV `MTLBuffer`s are never freed —
  only logical page indices recycle — and fresh pages are zeroed.
- **Prefix-reuse RoPE/position correctness is sound by construction.**
  Adoption fires only at `firstSubmit` (position 0) and walks the trie from
  root, so a page promoted for tokens at absolute positions [P,P+15] can only
  be adopted where the identical prefix sits at the *same* positions. The
  token-only `hashPage` is NOT the match key (the trie is), so the old "hash
  misses position/layer" worry is not a live defect.
- **The documented 2026-05-08 sampling shift** is most likely K=20 small-sample
  noise (the doc itself says "just outside 2σ") plus the now-fixed
  seed-propagation bug for *seeded* repros — not unsound K/V reuse.

## Confirmed defects (ranked)

1. **Page leak under pool pressure** *(confirmed, clean fix)* —
   `adoptSharedPrefixPages` CoW path (`lm_engine.swift:1257-1266`): if the 2nd
   `allocFresh()` throws (pool exhausted), `freshHead` (refcount=1) is neither
   appended to `ownedPages` nor decref'd → **permanent leak**, triggered
   exactly under the pressure that caused it. Fix: decref `freshHead` in the
   catch.
2. **`decref` silently swallows double-free/underflow** *(confirmed)* —
   `page_manager.swift:395-407`; `PageManagerError.doubleFree` is defined but
   never thrown. Masks every balance bug (so none surface as a crash). Fix:
   `precondition(refcount > 0)` (or log-once) instead of silent no-op.
3. **`LM_PARTIAL_COW_DISABLE` shares a partial page the consumer then WRITES**
   *(confirmed soundness hole, env-gated)* — `lm_engine.swift:1234-1252`,
   self-described "race-unsafe on concurrent extends." Two adopters diverge in
   the same physical page → cross-session K/V corruption. The default CoW path
   is sound. Fix: delete the escape hatch (or assert refcount==1 before write).
4. **`anchorByPhys` is single-valued** *(confirmed mechanism, inferred trigger)*
   — `radix_trie.swift:136`; a phys page referenced by both a full-pair and a
   partial anchor has its back-pointer overwritten by the 2nd `insertAnchor`,
   so eviction unlinks only one anchor → the other serves a now-reallocated
   page → stale adoption / cross-session corruption. Fix: multi-map, or walk
   all anchors referencing a phys on eviction.
5. **`gemma_shutdown` frees pages + nils the engine WITHOUT `gEngineLock` and
   without a GPU fence** *(confirmed structural gap)* — `ffi_batch.swift:1475`.
   A real UAF race if shutdown overlaps an in-flight `gemma_poll` (plausible
   during the `pkill serve.py` teardown the memory notes describe). Fix: take
   `gEngineLock` (and drain/await) in shutdown.
6. **No barrier in PageManager — the no-UAF guarantee is a whole-loop implicit
   invariant** *(confirmed; latent)* — there is no per-page GPU-completion
   gate; safety rests entirely on coarse-lock + synchronous CBs. The
   documented prior lock-removal wedge confirms this is load-bearing. The
   TODO'd lock-split (engine-state vs page-manager) would reintroduce a UAF
   class. Fix-before-refactor: stamp each page with the CB generation at
   block-table install; refuse to re-hand until that generation completes.
7. **"LRU" eviction is actually stack-top (MRU)** *(confirmed, perf-correctness)*
   — `page_manager.swift:291-302` pops most-recently-freed despite the comment
   promising LRU-oldest; `lastAccessTick` is tracked but never consulted →
   evicts hottest cache entries first under pressure.
8. **No enforced invariant that shared (refcount>1) pages are immutable**
   *(confirmed; convention-only)* — the `.primed` recover path + the
   `LM_KV_OVERWRITE_CHECK` env guard exist *because* this is known-violable.

## Proposed invariant set (the "validation" that's missing)

Add to `page_manager.swift` (none exist today):
- range precondition on `incref`/`decref`/`markPairContentIndexed`/`allocFresh`;
- single-location: a page is in exactly one of {on a free stack, refcount>0};
- no-underflow: `decref` preconditions/throws `doubleFree`;
- pair integrity: `pairMate` is symmetric and shares `contentHash`;
- shared-immutability: assert `pageRefcount(phys) <= 1` before any non-recover
  K/V write;
- adoption-at-root: assert `findLongestPrefix` adoption begins at depth 0 (so a
  future non-zero-offset adopter trips it instead of serving RoPE-mismatched
  K/V).

## Fix sequencing (validation deferred to a quiet machine)

- **Low-risk, host-independent, can land + build now:** #1 (leak), #5
  (shutdown lock), #3 (delete COW-disable hatch), #7 (real LRU), plus the
  invariant asserts.
- **Needs runtime validation at speed:** #2 (decref precondition — could trip a
  latent double-free; introduce as log-once first), #4 (anchor multi-map), #6
  (per-page GPU-completion gate, especially before any lock-split).

## Implemented 2026-05-28 (built + correctness-validated)

Landed and shipped in `libgemma_metal.dylib`:
- **#1 leak** — `lm_engine.swift` adopt-CoW: decref `freshHead` if the 2nd
  `allocFresh` throws.
- **#2 detector** — `page_manager.swift` `decref`: log-once on
  double-free / out-of-range (was silently swallowed).
- **#3** — deleted the `LM_PARTIAL_COW_DISABLE` shared-write hatch; partial
  tails are always CoW'd.
- **#4** — `radix_trie.swift` `anchorByPhys` is now `[Int: [AnchorBack]]`;
  eviction unlinks ALL anchors referencing a phys (+ `removeBack`/
  `pruneAnchorChain` helpers).
- **#5** — `gemma_shutdown` takes `gEngineLock`.
- **#7** — `allocFresh` forced-eviction picks the LRU (lowest
  `lastAccessTick`), not the stack top.

**Validation (correctness; host was throughput-contended so tok/s is NOT a
valid signal and was not used):** build exit 0; 3 lifecycle requests — full,
repeat (prefix-cache adoption), mid-stream abort (cancel→closeSession) — all
produced correct output; `free_pages` returned **16000→16000** after all
sessions closed (no leak); the #2 double-free detector stayed **silent** (clean
refcount balance); no crash, engine `ready`. NOT validated: throughput (host
contention, orthogonal to these fixes); the exact dual-anchor #4 trigger (can't
construct on demand); a clean-host before/after.

**Deferred by design — #6 (per-page GPU-completion gate).** Agent B confirmed
the serving path has **no active use-after-free** today (synchronous CBs +
coarse `gEngineLock`). #6 only defends a FUTURE engine-state/page-manager
lock-split that does not exist yet; a speculative blocking gate adds deadlock
surface to guard a non-live hazard. **Prerequisite for any lock-split**, to be
implemented *with* that refactor and validated on a quiet host.

## Update 2026-05-31 — defect #8 went LIVE on the serving path; fixed (16b5416)

Defect #8 ("no enforced invariant that shared (refcount>1) pages are immutable")
was not merely a latent convention gap: it FIRED on the production serving path.
Root cause: ONE flat block_table per slot serves both layer geometries —
full-attention layers (PAGE_FULL=8) address `ownedPages[pos/8]`, sliding-window
layers (PAGE_SLIDE=16) address `ownedPages[pos/16]`. Aligned adoption of N
slide-pairs incref-shares `ownedPages[0..2N-1]` read-only. For the FULL layers
every shared page [8k..8k+7] sits inside the shared prefix [0..16N-1] (sound),
but for the SLIDE layers pages k ∈ [N..2N-1] map to positions [16N..32N-1] —
PAST the shared prefix, i.e. this session's own divergent generation region.
Once the consumer generated past 16N it WROTE its divergent slide K/V into
`ownedPages[N..2N-1]`, pages the producer and sibling adopters still referenced
read-only → cross-session slide-K/V clobber → off-manifold activations →
Gemma-4 absorbing-state collapse ("multilingual token soup", running to the
max-tokens cap, no EOS) on the SillyTavern shareable-prefix /poll suggester.
State-dependent and concurrency-exposed: 0/40 serial, 12.5% at conc4, 17.7% at
conc8; direct-to-bridge with identical messages never collapsed.

Fix (commit 16b5416, 2026-05-31): refcount-gated copy-on-write of the
slide-divergent half `ownedPages[N..2N-1]` in `adoptSharedPrefixPages`
(lm_engine.swift). Whole-page copy preserves the still-valid shared FULL K/V
[8N..16N-1] while privatizing the slide rows the consumer is about to overwrite;
a no-op when refcount==1 (producer already released); degrades to prior shared
behavior on `allocFresh` pool-pressure failure rather than crashing. The
partial-tail path was independently re-verified sound under the same fix (slide
layers read the privatized `ownedPages[N]`; the partial-CoW's wrong slide
content at `ownedPages[2N]` is dead-for-slide and its full K/V is correct).

Also fixed in the same change: the `LM_KV_OVERWRITE_CHECK` detector
(`hashSessionKVAtPosition`) was itself mis-indexing slide layers at
`ownedPages[2*logSlide]` instead of the flat `ownedPages[logSlide]` — so the
very detector meant to catch a shared-page overwrite was hashing the WRONG
physical page and was BLIND to exactly this clobber. Corrected to the flat
per-layer ordinal (slide pos/16, full pos/8); re-run `LM_KV_OVERWRITE_CHECK=1`
as the regression guard for any future block_table sharing change. Validation:
conc8 reproducer 17.7% → 1.0% (below the 1.6% no-sharing floor; the residual is
baseline temp=1.0 sampling variance), slide-CoW fired on exactly [N..2N-1],
ZERO `KV BYTES CHANGED` in the divergent range, 0 leaks/double-frees, sharing
preserved. The geometry-vocabulary docs (SlidePairContents / PartialPagePair /
promoteFinishedPages / the detector comments) were also corrected — the field
name `slidePlusFullHead` is a historical misnomer carrying NO slide K/V; the
slide-layer page for slide-page P is the separate `ownedPages[P]`.

Cross-ref: `docs/kv_cache_correlation_finding.md` + `…_diagnosis.md` (the
2026-05-08 flat-dict era), `radix_trie.swift` (current match index),
`page_manager.swift`, `lm_engine.swift`, `ffi_batch.swift`.
