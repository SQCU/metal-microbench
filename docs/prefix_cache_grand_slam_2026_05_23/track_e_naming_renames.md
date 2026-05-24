# Track E — Naming-gap rename plan

> **STATUS:** Folded into Track D. Under D-carrier execution we apply
> ONLY the renames whose target symbols survive Track D:
> `chunkQueue → primingQueue`, `promotedPageCount → myPromotedSlidePairCount`,
> `pendingPrimingCount → pendingPrimingTokenCount`, `SharedPagePair →
> SlidePairContents`. We SKIP renames on `PageManager.contentIndex`,
> `findByHash`, `promotePair`, `hashPagePrefix`, `adoptSharedPrefixPages`,
> `promoteFinishedPages`, `revisitCacheProbe` — these are deleted or
> signature-replaced by D. JSON-wire field names (`cache_hits`,
> `cache_misses`, `pair_mate`, `pages_owned`) are KEPT regardless.

---

# Prefix-cache naming rename plan

Scope of grep: `/Users/mdot/metal-microbench/**/*.{swift,py,md,mjs,js,html,json}` excluding `node_modules`, `*.bak`. `/Users/mdot/sillytavern-fork/` had **zero hits** for any of the in-scope symbols — the plugin stays JSON-API-coupled and is unaffected.

All citations are absolute paths. Definition lines marked with `[def]`.

---

## 1. Renaming table

### 1.1 `PAGE_SLIDE = 16` → keep, but add a layer of meaning

**Final name:** **Keep `PAGE_SLIDE = 16` as the physical KV-block geometry constant** (it correctly names a slide-attention layer page size, distinct from `PAGE_FULL = 8`). **Introduce `let CACHE_LOOKUP_STRIDE = PAGE_SLIDE` (alias) at `bootstrap.swift:641`** and migrate every *cache lookup / promotion / adoption* site to `CACHE_LOOKUP_STRIDE`. That preserves the geometric meaning at KV-write/kernel-dispatch sites and exposes the *separate* semantic concept ("the quantum the cache walks at") at lookup sites — so Track D can rename `CACHE_LOOKUP_STRIDE` to whatever the trie wants without disturbing kernel code.

**Reviewer's `KV_SLIDE_BLOCK_TOKENS` is fine for the physical constant but doubles the churn at every kernel call site (~25 hits in bootstrap.swift alone, all unambiguously geometric).** Deviating to keep the geometric name and only introduce the new concept where it differs.

**Call sites (every `PAGE_SLIDE` occurrence; * = cache-lookup site → migrate to `CACHE_LOOKUP_STRIDE`):**

Physical geometry (keep as `PAGE_SLIDE`, 22 sites):
- `/Users/mdot/metal-microbench/bootstrap.swift:527, 641[def], 3306, 3339, 3362, 3364–3365, 3435, 3451, 3460–3461, 3476, 4511, 4948, 5271, 5479, 5606, 5674`
- `/Users/mdot/metal-microbench/profile_ar_step.swift:111, 241`
- `/Users/mdot/metal-microbench/profile_prefill.swift:140`
- `/Users/mdot/metal-microbench/weights.swift:616`
- `/Users/mdot/metal-microbench/lm_engine.swift:622, 928, 1531, 1550, 1580, 1627, 2232`
- `/Users/mdot/metal-microbench/harness.swift:1650, 1659, 1670, 1674, 1917`
- `/Users/mdot/metal-microbench/lm_session.swift:183`
- `/Users/mdot/metal-microbench/kv_visualizer.swift:41 (comment), 291`

Cache-lookup sites (migrate to `CACHE_LOOKUP_STRIDE`, 13 sites):
- `/Users/mdot/metal-microbench/lm_engine.swift:979, 989, 991, 994, 1048, 1049, 1050, 1051, 1073, 1094, 1103, 1817, 1820, 1825, 1838, 1840, 2566, 2571` (this is the AR-promotion + submit-adoption + cvec-digest range)
- `/Users/mdot/metal-microbench/server/test_batch_ffi_inbatch_share.py:55`
- `/Users/mdot/metal-microbench/server/test_chat_template_prefix_continuity.py:20, 29[def, py], 82, 102`
- `/Users/mdot/metal-microbench/server/test_chat_template_prefix_multimodal.py:23[def, py], 106, 107, 134`
- `/Users/mdot/metal-microbench/server/test_batch_ffi.py:35`
- `/Users/mdot/metal-microbench/server/test_batch_ffi_svg_shape.py:139`
- `/Users/mdot/metal-microbench/server/test_batch_ffi_multiturn.py:104, 125`

**Backcompat:** KEEP-OLD for physical sites + KEEP-AS-ALIAS for new lookup constant. Both Swift names are internal — no public exposure.
**LoC churn:** ~35 lines (just the lookup sites and the alias declaration). Mechanical.

---

### 1.2 `adoptSharedPrefixPages` → `adoptFullyPromotedLeadingSlidePairs`

**Final name:** `adoptFullyPromotedLeadingSlidePairs` (singular `Pair` is misleading since it walks ≥1 pair; plural). Reviewer's `adoptFullyPromotedLeadingSlidePages` is close, but the function returns a count of *slide pages* yet pushes *pairs of phys pages* to `ownedPages` — `Pair` is the right unit because every adopted entry is a `SharedPagePair`. Track B (partial-page) will add an `adoptPartialLeadingSlidePair` sibling — the parallel naming makes that obvious.

**Call sites:**
- `/Users/mdot/metal-microbench/lm_engine.swift:1041[def], 965, 1088, 1150 (comment)`
- `/Users/mdot/metal-microbench/docs/kv_cache_correlation_diagnosis.md:67`
- `/Users/mdot/metal-microbench/notes/spec_vs_inductions.md:102`
- `/Users/mdot/metal-microbench/notes/swift_comments_audit.md:241`
- `/Users/mdot/metal-microbench/notes/specs/bandwidth_triage.md:195` (referenced as part of FFI-line discussion — verify wording)

**Backcompat:** RENAME-ONLY-INTERNAL (private method, no FFI). **LoC churn:** 4 Swift lines, 4 doc lines = 8.

---

### 1.3 `promoteFinishedPages` → `promoteSlidePairsCompletedThisTick`

**Final name:** `promoteSlidePairsCompletedThisTick`. Reviewer's `promotePageAlignedCompletedSlideBlocks` is accurate but verbose; "completed this tick" is the operative truth — the function is called after each prefill tile commit and after each AR token (when `position % PAGE_SLIDE == 0`), and it walks `promotedPageCount → fullyWritten`. It does NOT promote on `.done` if `position % PAGE_SLIDE != 0` (i.e. "session ended" alone does not qualify), which "Finished" hides. "ThisTick" reads naturally with the call-site guard at `lm_engine.swift:2571`.

After Track A (backstop removal) lands, `.done` will still gate via the modulo check at the call site, so this name remains correct.

**Call sites:**
- `/Users/mdot/metal-microbench/lm_engine.swift:1816[def], 1148 (comment), 2552 (comment), 2568 (comment), 2572, 2847, 3343, 3541`
- `/Users/mdot/metal-microbench/docs/kv_cache_correlation_diagnosis.md:66`
- `/Users/mdot/metal-microbench/notes/spec_vs_inductions.md:104`
- `/Users/mdot/metal-microbench/notes/swift_comments_audit.md:258, 279`

**Backcompat:** RENAME-ONLY-INTERNAL. **LoC churn:** 8 Swift lines, 4 doc lines = 12.

---

### 1.4 `PageManager.hashPage` → split

**Final names:**
- `hashTokenRange(_ tokens: ArraySlice<UInt32>, cvecDigest: UInt64 = 0) -> UInt64` — the low-level hashing primitive (current body, renamed in place).
- `hashPagePrefix(consumed: [UInt32], pageIndex: Int, pageSize: Int, cvecDigest: UInt64) -> UInt64` — a new convenience wrapper that does `hashTokenRange(consumed[0..<(pageIndex+1)*pageSize], cvecDigest:)`. Callers should switch to this so the prefix-vs-page-content distinction becomes type-level instead of comment-level.

Agree with reviewer; this is the highest-value rename (prevents silent corruption from passing the wrong slice once Track D introduces token-granularity lookups that operate over partial pages).

**Call sites (5 direct call uses + comments):**
- `/Users/mdot/metal-microbench/page_manager.swift:201[def]` — body kept, signature renamed
- `/Users/mdot/metal-microbench/lm_engine.swift:1052` — migrate to `hashPagePrefix`
- `/Users/mdot/metal-microbench/lm_engine.swift:1841` — migrate to `hashPagePrefix`
- `/Users/mdot/metal-microbench/lm_engine.swift:383 (comment), 420 (comment), 707 (comment)` — comment refs
- `/Users/mdot/metal-microbench/kv_visualizer.swift:207 (comment)` — visualizer manually re-implements FNV-1a; comment needs to point at `hashTokenRange`
- `/Users/mdot/metal-microbench/docs/CVEC_AND_PREFIX_CACHE.md:21, 108`
- `/Users/mdot/metal-microbench/docs/kv_cache_correlation_finding.md:122`
- `/Users/mdot/metal-microbench/notes/swift_comments_audit.md:231, 337, 419, 524`
- `/Users/mdot/metal-microbench/notes/specs/bandwidth_triage.md:322`

**Backcompat:** RENAME-ONLY-INTERNAL. **LoC churn:** ~6 Swift + ~10 doc + ~15 lines for the new wrapper function = ~30.

**Track D coordination:** if the trie introduces token-granularity lookup, `hashTokenRange` is exactly what it needs and `hashPagePrefix` becomes one of several wrappers — the split is forward-compatible.

---

### 1.5 `findByHash` → `findPairByPrefixHash`

**Final name:** `findPairByPrefixHash` (reviewer's suggestion verbatim). Returns the `SharedPagePair` (renamed below) and the "Pair" makes the geometric coupling explicit.

**Call sites:**
- `/Users/mdot/metal-microbench/page_manager.swift:217[def], 320 (comment)`
- `/Users/mdot/metal-microbench/lm_engine.swift:1054, 1814 (comment)`
- `/Users/mdot/metal-microbench/notes/swift_comments_audit.md:391`
- `/Users/mdot/metal-microbench/notes/specs/bandwidth_triage.md:195`

**Backcompat:** RENAME-ONLY-INTERNAL. **LoC churn:** 2 Swift + 4 doc = 6.

---

### 1.6 `SharedPagePair` + `slidePrimary` / `fullSibling` → `SlidePairContents` / `slidePlusFullHead` / `fullTail`

**Final name:** `struct SlidePairContents { let slidePlusFullHead: Int; let fullTail: Int }`. Reviewer's `SlidePlusFullHeadPage` / `FullTailPage` is right at the field level but they're `Int` phys-page IDs, not page types — so put the descriptive name on the *field* (`slidePlusFullHead: Int` and `fullTail: Int`) and rename the outer struct to `SlidePairContents`. This is more accurate than `SharedPagePair`: the struct value is the *content* of a pair (the two phys IDs); the *pages themselves* aren't typed.

Track B (partial-page promotion) will likely need a sibling `PartialSlidePairContents { slidePlusFullHead: Int; fullTail: Int?; validUpTo: Int }` — the rename lays the pattern.

**Call sites:**
- `/Users/mdot/metal-microbench/page_manager.swift:88[def], 89 (slidePrimary), 90 (fullSibling), 340–341 (initializer), 326 (comment), 322 (header comment)`
- `/Users/mdot/metal-microbench/page_manager.swift:327, 332, 336, 337, 343–344` (`promotePair(slidePrimary:fullSibling:contentHash:)` — also rename to `promoteSlidePair(slidePlusFullHead:fullTail:contentHash:)`)
- `/Users/mdot/metal-microbench/lm_engine.swift:1061–1064` (`pair.slidePrimary`, `pair.fullSibling`, ownedPages append order)
- `/Users/mdot/metal-microbench/lm_engine.swift:1843–1845` (`promotePair(slidePrimary:fullSibling:contentHash:)` call)
- `/Users/mdot/metal-microbench/notes/spec_vs_inductions.md:101`
- `/Users/mdot/metal-microbench/notes/swift_comments_audit.md:384`

**Backcompat:** RENAME-ONLY-INTERNAL (struct never leaves Swift). **LoC churn:** ~14 Swift + 2 doc = 16.

---

### 1.7 `cacheHitTokens` / `cacheMissTokens` (Session fields) — RENAME-internal, **KEEP-OLD on JSON wire**

**Final names (Swift):** `pageAlignedCacheHitTokens` / `prefilledTokens` (NOT "miss tokens"). The current "miss" name hides that the trailing 1–15 tokens are *always* prefilled even when there was a cache hit — they're a deterministic remainder, not a miss. "Prefilled" describes the action taken; the ratio operators care about is `prefilled / (page_aligned_hit + prefilled)` which is honest.

**KEEP THE WIRE NAMES** `cache_hits`, `cache_misses`, `cache_hit_tokens`, `cache_miss_tokens` — these are public OpenAI-extension fields consumed by the static dashboards, by the recorded SSE chunks in `/Users/mdot/metal-microbench/docs/media/`, and (per docs) potentially by external dashboard tooling. The Swift→JSON serializer should map new Swift name → old JSON name.

**Swift-side call sites (rename to `pageAlignedCacheHitTokens` / `prefilledTokens`):**
- `/Users/mdot/metal-microbench/lm_engine.swift:644–645[def], 641–643 (comment), 991, 1002, 1111–1115` (note: the `cacheMissTokens &-= advU` arithmetic at 1112–1115 stays semantically identical under rename)
- `/Users/mdot/metal-microbench/ffi_batch.swift:1185–1186, 1392–1393` (serializer site — map to old JSON keys)
- `/Users/mdot/metal-microbench/notes/specs/bandwidth_triage.md:183, 671, 708`

**Wire / consumer sites (do NOT rename):**
- `/Users/mdot/metal-microbench/ffi_batch.swift:785–786, 830–831, 1073–1074` (StreamUpdateOut field names; could stay as `cacheHits`/`cacheMisses` since this struct is internal-but-wire-shaped — owner's call)
- `/Users/mdot/metal-microbench/server/gemma_ffi.py:194–195, 226, 455, 685` (Python decoder fields)
- `/Users/mdot/metal-microbench/server/bridge.py:350–352, 1054, 1077, 1467–1469, 1502–1504, 1792–1794, 1801–1803` (log lines + JSON serializer)
- `/Users/mdot/metal-microbench/server/static/{clients.html, steering.html, loom.html, index.html, labeler.html}` — ~15 hits, all read from `usage.cache_hits` / `usage.cache_misses`
- `/Users/mdot/metal-microbench/docs/engine_telemetry_endpoint.md:7–8, 72–73`, `docs/QUICKSTART.md:94`, `docs/user_agent_factorization_spec.md:138–139`
- `/Users/mdot/metal-microbench/docs/media/**/sse_chunks.jsonl` — recorded test fixtures, **do not edit**

**Add documentation** at the serializer site (`ffi_batch.swift:1392`) explaining "wire name = `cache_hit_tokens`, semantic = `page-aligned cache-hit tokens`; the trailing partial page is always counted in `cache_miss_tokens` even when the producer-session's prefix matched."

**Backcompat:** KEEP-OLD on wire, RENAME-ONLY-INTERNAL Swift-side. **LoC churn:** ~8 Swift renames + 1 comment block at the serializer = ~12.

---

### 1.8 Backstop comment → `shedTrailingSlidePairToForceAtLeastOnePrefillToken(s:)`

**Final name:** Extract the backstop into a named private method `shedTrailingSlidePairToForceAtLeastOnePrefillToken(_ s: Session, eng: LmEngine)`. The current logic is duplicated at:
- `/Users/mdot/metal-microbench/lm_engine.swift:979–989` (submit path)
- `/Users/mdot/metal-microbench/lm_engine.swift:1096–1104` (revisitCacheProbe path)

Both fragments do the same thing: pop two phys pages off `ownedPages`, decref, decrement `promotedPageCount`, subtract `PAGE_SLIDE` from the advance. Extract → name honestly → call from both sites. Reviewer's name verbatim is fine; "SlidePair" rather than "CachedPage" because each shed = 2 phys pages.

**Track A coordination (CRITICAL):** if the backstop is removed entirely, this rename is wasted work — the extracted method dies. **Recommendation: do not extract before Track A decides.** Instead, in this rename pass, only update the *comments* at lines 966–972, 974–988, and 1090–1104 to describe what the code does honestly ("shed the trailing slide pair so the prefill tail has ≥1 token") and leave the inline code in place. If Track A keeps the backstop, extract afterward.

**Backcompat:** RENAME-ONLY-INTERNAL. **LoC churn (comments only):** ~6.

---

### 1.9 `chunkQueue` → `primingQueue` (the field) + `unprefilledTailTokens` (the count helper)

**Final names:**
- Field: `var primingQueue: [PrimingChunk]` — already type-named `PrimingChunk`, so the field name should match.
- Existing accessor `pendingPrimingCount` → keep its name; semantics are "tokens still waiting to be teacher-forced" which is honest.
- Existing accessor `chunkQueueDepthForDebug` → `primingQueueDepthForDebug`.
- New accessor `unprefilledTailTokens` to distinguish from partial-page resumption (Track B). For now `unprefilledTailTokens == pendingPrimingCount`; once Track B introduces partial-page resumption (where the *first* chunk represents tokens whose K/V is partially valid), `unprefilledTailTokens = pendingPrimingCount - firstChunk.validUpTo`. Adding the accessor now (even as an alias) gives Track B a slot to fill.

Reviewer's distinction is right; the cleanest landing is to rename the field (mechanical) and stub the new accessor (forward-compatible).

**Call sites for `chunkQueue` rename (≈ 30 hits):**
- `/Users/mdot/metal-microbench/lm_engine.swift:46 (comment), 647[def], 955, 971 (comment), 1001, 1081 (comment), 1095, 1118, 1119, 1122, 1127, 1130, 1143, 1148 (comment), 1191, 1194, 1695 (comment), 1823 (comment), 1967, 1971, 1974, 1975, 1990, 2533, 2648, 2853, 2854, 2857, 2867, 2876, 2950, 3012, 3018, 3139, 3180, 3347, 3348, 3350, 3414, 3544, 3550, 3556`
- `/Users/mdot/metal-microbench/docs/dataflow_pipeline_spec.md:37`
- `/Users/mdot/metal-microbench/notes/spec_vs_inductions.md:96, 150, 217`
- `/Users/mdot/metal-microbench/notes/swift_comments_audit.md:225, 243`
- `/Users/mdot/metal-microbench/notes/decisions/2026-04-26-remove-session-concurrency-primitives.md:142`

**Backcompat:** RENAME-ONLY-INTERNAL (`fileprivate` field, no FFI). **LoC churn:** ~40 Swift + 6 doc = 46. By far the biggest churn item.

---

### 1.10 `bootstrap.swift:5553` comment "any common prefix" → rewrite

**Final wording (replace the doc comment at `bootstrap.swift:5552–5559`):**

> // PrefixCache: FNV-1a hash → shared phys pages. On a new prompt, hash
> // the prefix tokens **at PAGE_SLIDE-aligned (16-token) granularity** and
> // reuse existing phys pages for **any PAGE_SLIDE-aligned common prefix**
> // (the trailing 0–15 tokens are always prefilled fresh). Each hash entry
> // carries a refcount so pages stay live while in use by any slot.
> // Implementation only — the attn kernel doesn't know about sharing; it
> // only reads block_table, which we populate with shared phys IDs for
> // slots with a PAGE_SLIDE-aligned shared prefix.

**Backcompat:** N/A (comment). **LoC churn:** 8 lines rewritten in place.

---

### 1.11 `promotedPageCount` → `myPromotedSlidePairCount`

**Final name:** `myPromotedSlidePairCount` — emphasizes (a) it's per-session and (b) the unit is slide pairs (not phys pages — would be 2× higher — and not slide pages independent of full siblings). Reviewer's `myPagesPublishedToContentIndex` is good but "Pages" still has the slide-vs-full ambiguity that bit `SharedPagePair`.

**Call sites:**
- `/Users/mdot/metal-microbench/lm_engine.swift:639[def], 637–638 (comment), 987, 1069, 1102, 1807, 1817, 1818, 1819, 1850`
- `/Users/mdot/metal-microbench/notes/swift_comments_audit.md:224`

**Backcompat:** RENAME-ONLY-INTERNAL. **LoC churn:** 10 Swift + 1 doc = 11.

---

## 2. Doc-update list (comments that need rewriting, not just renaming)

| Where | Current text (paraphrased) | Proposed |
|---|---|---|
| `bootstrap.swift:5553–5559` | "hash the prefix tokens at PAGE-granularity and reuse for any common prefix" | See 1.10 — explicit about `PAGE_SLIDE` alignment and the 0–15-token tail |
| `lm_engine.swift:641–643` | "cacheHitTokens counts tokens covered by adopted cache pages on this stream's submits; cacheMissTokens counts tokens this stream had to prefill itself" | "pageAlignedCacheHitTokens counts tokens covered by PAGE_SLIDE-aligned adopted pages; prefilledTokens counts every token whose K/V this stream had to compute — including the always-prefilled trailing 0–15 of an otherwise fully-cached prompt. JSON wire still uses `cache_hits` / `cache_misses` for back-compat." |
| `lm_engine.swift:966–972` | "Guarantee the prefill tail covers ≥ 1 token … the post-prefill sampling path … only fires when stepPrefillForSession actually runs." | "Shed the trailing slide pair (2 phys pages) when the prompt is an exact PAGE_SLIDE multiple, so the sampler — which only fires from stepPrefillForSession — has at least one token to run on. NB: this is currently a workaround for the sampler-on-prefill coupling, not a fundamental requirement of the cache; Track A is removing it." |
| `lm_engine.swift:1090–1093` | "Backstop: keep at least 1 token of tail to prefill" | "Same shed-trailing-pair workaround as the submit() path. See `lm_engine.swift:966` for the underlying coupling." |
| `lm_engine.swift:46` (SessionState comment) | ".priming — chunkQueue non-empty" | ".priming — primingQueue non-empty (or just-finished prefill transitioning to .generating)" |
| `page_manager.swift:185–200` (hashPage docstring) | "FNV-1a over a page's token IDs, optionally mixed with a cvec-state digest" | Split into two docstrings — `hashTokenRange` keeps the FNV+cvec-digest description; new `hashPagePrefix` wrapper says "FNV-1a over `consumed[0..<(pageIndex+1)*pageSize]`. Callers MUST pass the cumulative prefix, not a single page slice — the cache key partitions on prefix content, not on page content alone." |
| `page_manager.swift:56–68` (SharedPagePair header) | "Pair bookkeeping … `pairMate` is the phys index of the other member" | Rewrite to spell out: "slidePlusFullHead = block_table[P] carries [slide K/V for tokens [P*16..P*16+15]] AND [full K/V for tokens [P*16..P*16+7]]. fullTail = block_table[P+1] carries [full K/V for tokens [P*16+8..P*16+15]]. The two are content-cached as a unit; adopting only one yields KL~0.38 divergence vs. fresh compute." |
| `lm_engine.swift:2566–2570` (promote-only-at-boundaries) | "Promote ONLY at page boundaries (every PAGE_SLIDE tokens) … promoteFinishedPages is internally bounded" | "Promote ONLY at PAGE_SLIDE boundaries (s.position % CACHE_LOOKUP_STRIDE == 0). 'Done' alone does not qualify if the final page is partial — partial-page promotion is Track B's responsibility." |
| `notes/spec_vs_inductions.md:96–104` | List of names | Update to new names with a one-line history note. |

---

## 3. Missed names (new findings)

| Existing name | Where | Complaint | Proposed | Backcompat | Churn |
|---|---|---|---|---|---|
| `PageManager.promotePair(slidePrimary:fullSibling:contentHash:)` | `page_manager.swift:327` | Same `Pair` ambiguity as `SharedPagePair`. | `promoteSlidePair(slidePlusFullHead:fullTail:contentHash:)` | RENAME-ONLY-INTERNAL | 2 Swift |
| `pairMate` (PageInfo, PageRecord, JSON `pair_mate`) | `page_manager.swift:79, 281, 398, 413`; `ffi_batch.swift:1417` | "Mate" is intuitive but undirected — doesn't say whether mate is slide-head or full-tail. | Swift: `companionInPair`. **JSON wire `pair_mate`: KEEP** (consumed by static dashboards). | KEEP-OLD on wire | 6 Swift |
| `ownedPageCount` (Session) | `lm_engine.swift:628`; `ffi_batch.swift:1391` (`pages_owned`) | "Pages" again — really counts phys pages (= 2 × slidePairs). Confusing when paired with `myPromotedSlidePairCount`. | Swift: `ownedPhysPageCount`. **JSON wire `pages_owned`: KEEP**. | KEEP-OLD on wire | 3 Swift |
| `pendingPrimingCount` | `lm_engine.swift:1191` | Sounds like "count of priming tasks pending" but actually sums tokens across all primingQueue chunks. | `pendingPrimingTokenCount` | RENAME-ONLY-INTERNAL | 8 Swift (also used at `lm_engine.swift:2818, 2822, 3312, 3316, 3511, 3514`) |
| `chunkQueueDepthForDebug` | `lm_engine.swift:1194` | Tied to old name. | `primingQueueDepthForDebug` | RENAME-ONLY-INTERNAL | 1 Swift (no callers in repo apart from FFI consumer search — verify before landing) |
| `promoted` (JSON key in `livePageSnapshot` payload) | `ffi_batch.swift:1414, 1417`; `docs/engine_telemetry_endpoint.md` | The boolean is `contentHash != nil`, i.e., "this page is content-indexed." "Promoted" reads as a verb. | Swift: `isContentIndexed`. **JSON wire `promoted`: KEEP** (consumed by static dashboards via `r.cached_pages`/etc; verify `promoted` isn't directly read — search showed it isn't). Safe to RENAME wire too. | RENAME-OK | 3 Swift + doc |
| `revisitCacheProbe` | `lm_engine.swift:1083` | "Revisit" doesn't say *why* — it's the post-extension re-probe after a multi-segment submit (text + image + text). | `reprobeCacheAfterTokenExtension` | RENAME-ONLY-INTERNAL | 3 Swift |
| `PageManager.promotePair` parameter ordering in callers | `lm_engine.swift:1843–1845` | Caller passes `s.ownedPages[2*p]` and `[2*p+1]` — positional, no compile-time check that you got the order right. | Considered, but Swift call-site label `(slidePlusFullHead: ownedPages[2*p], fullTail: ownedPages[2*p+1])` already enforces this once 1.6 lands. No additional change needed. | — | 0 |
| `LM_CACHE_DEBUG` env var | `lm_engine.swift:1047, 1070, 1085, 1161, 1846` | Operator-facing env var; fine name. | **KEEP** | KEEP-OLD | 0 |
| Log prefix `"  [cache]"` | `lm_engine.swift:1057, 1071, 1086, 1162, 1848` | Ops grep target — stable string. | **KEEP** | KEEP-OLD | 0 |
| `gemma_engine_state` JSON keys: `cached_pages`, `pages_in_use`, `pages_owned`, `cache_hit_tokens`, `cache_miss_tokens`, `pair_mate`, `promoted` | `ffi_batch.swift:1339–1419` | All consumed by `server/static/*.html` (verified). | **KEEP ALL** | KEEP-OLD | 0 |

---

## 4. Sequencing

**Recommendation: three sequential PRs, not one.**

(NOTE — superseded under D-carrier execution: only the surviving-code
renames in PR-2 apply, plus a final pass after D lands.)

**PR-1 (mechanical type/struct renames — lowest conflict surface):**
- 1.6 `SharedPagePair` → `SlidePairContents` + field renames + `promotePair` → `promoteSlidePair`
- 1.5 `findByHash` → `findPairByPrefixHash`
- 1.4 `hashPage` → `hashTokenRange` + add `hashPagePrefix` wrapper
- Missed 3.1 `companionInPair`, 3.3 `ownedPhysPageCount`
- ~70 LoC, all Swift, all compile-checked. Land first.

**PR-2 (function/method renames + private field renames):**
- 1.2 `adoptSharedPrefixPages` → `adoptFullyPromotedLeadingSlidePairs`
- 1.3 `promoteFinishedPages` → `promoteSlidePairsCompletedThisTick`
- 1.9 `chunkQueue` → `primingQueue` (biggest single rename — ~40 hits)
- 1.11 `promotedPageCount` → `myPromotedSlidePairCount`
- 3.4 `pendingPrimingCount` → `pendingPrimingTokenCount`
- 3.5 `chunkQueueDepthForDebug` → `primingQueueDepthForDebug`
- 3.7 `revisitCacheProbe` → `reprobeCacheAfterTokenExtension`
- ~80 LoC. Land second so it doesn't conflict with PR-1's struct surgery.

**PR-3 (constant introduction + comment rewrites + JSON-key telemetry split):**
- 1.1 `let CACHE_LOOKUP_STRIDE = PAGE_SLIDE` + migrate 13 lookup sites
- 1.7 `pageAlignedCacheHitTokens` / `prefilledTokens` Swift-side + serializer mapping comment
- 1.8 backstop comment-only rewrite (no extraction; wait for Track A)
- 1.10 `bootstrap.swift:5553` comment rewrite
- Section 2 doc updates
- ~60 LoC + comments. Land third, after PR-2 is shaken out.

**Don't merge as one PR** because (a) PR-2's `chunkQueue` rename alone churns ~46 lines and merge-conflicts everything; (b) splitting gives bisect granularity if Track A/B/D's later work surfaces a semantic regression masked as a "name-only" change; (c) the wire-name decisions in PR-3 are higher-risk and benefit from review independent of the obvious renames.

---

## 5. Risk assessment

**Compile-error tier (Swift, low risk):** all of 1.2, 1.3, 1.4, 1.5, 1.6, 1.9, 1.11, 3.1, 3.3, 3.4, 3.5, 3.7. If a call site is missed, Swift refuses to compile. No safety net needed.

**Comment-only tier (zero runtime risk):** 1.8, 1.10, Section 2 doc edits. Wrong-but-internally-consistent is the worst outcome (someone reads a stale comment); rerun the same audit one quarter later.

**Runtime-error tier (Python, medium risk):** the Swift `cacheHitTokens` → `pageAlignedCacheHitTokens` rename has a serializer site at `ffi_batch.swift:1392` that writes the JSON literal key `"cache_hit_tokens"`. If a typo lands in the literal during the rename, the Python decoder at `server/gemma_ffi.py:194` does `cache_hits: int` via the binary-decoder path (not the engine_state JSON path) — but the `/v1/engine/state` JSON path is consumed by `server/static/steering.html:376`, `server/static/loom.html:401`, etc., which do `s.usage.cache_hits || 0` (note the `|| 0`). **A wire-key typo silently degrades the dashboards to "0 cache hits forever" with no error.** Mitigation: add a `server/test_engine_state_keys.py` smoke test that asserts the JSON snapshot contains exactly the documented key set; gate PR-3 on it.

**Silent-semantic-bug tier (worst, high risk):** the *only* candidate in this set is 1.7's wire-name handling. Mitigations:
- KEEP all `cache_hits` / `cache_misses` / `cache_hit_tokens` / `cache_miss_tokens` / `pair_mate` / `pages_owned` JSON keys exactly as-is (this plan does).
- Add a *single* deprecation-window emission for any JSON key we choose to rename anyway (e.g., if 3.6 `promoted` → `is_content_indexed` is accepted, emit BOTH keys for two release cycles and document in `engine_telemetry_endpoint.md`).
- The recorded SSE chunks in `docs/media/**/sse_chunks.jsonl` are test fixtures; under no circumstances rewrite them — they document historical wire format. Add a `CHANGELOG.md` entry instead.

**Backstop comment rename (1.8) special case:** if Track A removes the backstop while PR-3 is in flight with the rewritten comments, the comments become orphan documentation of dead code. Mitigation: land PR-3's backstop-comment change in the *same commit* as Track A's removal, not before — or, gate PR-3's backstop-comment hunk on Track A's status at merge time.

---

## 6. Interactions with other tracks

**Track A (backstop removal).**
- 1.8 (`shedTrailingCachedPageToForceAtLeastOnePrefillToken`) becomes moot if backstop is removed entirely; symbol disappears.
- *Recommendation:* don't extract the backstop into a named method in this rename pass. Limit to comment rewrites at `lm_engine.swift:966–972` and `1090–1093`. If Track A keeps the backstop, extract afterward.
- The `myPromotedSlidePairCount -= 1` decrement at `lm_engine.swift:1102` is part of the backstop's bookkeeping — if Track A removes the surrounding `if` block, the decrement goes with it. Rename in place; Track A deletes.

**Track B (partial-page promotion, introduces `validUpTo`).**
- 1.9 `unprefilledTailTokens` accessor is forward-prepared for Track B's `firstChunk.validUpTo`. Stub it as an alias of `pendingPrimingTokenCount` now; Track B fills in the subtraction.
- 1.6 `SlidePairContents` rename anticipates a sibling `PartialSlidePairContents` from Track B. Parallel naming will read cleanly.
- 1.3 `promoteSlidePairsCompletedThisTick` — Track B will add a `promotePartialSlidePairs` companion or extend the existing function. Either way the name stays accurate ("Completed" = fully-written; partial-page promotion gets its own function).
- 2.* doc updates for `promote-at-boundaries` comment should note "Track B will relax this to per-token granularity once `validUpTo` lands."

**Track C (cvecDigest tightening).**
- 1.4 `hashTokenRange(_, cvecDigest:)` is exactly the surface cvecDigest tightening will touch. After the rename, that work becomes a focused change to one function with one wrapper.
- The cvecDigest comments at `page_manager.swift:185–200` should not be rewritten in this pass beyond the split (1.4) — leave the *semantics* alone for Track C to revise.

**Track D (token-granularity radix lookup, new trie).**
- 1.1 `CACHE_LOOKUP_STRIDE` is the bridge name: the trie will introduce a *new* lookup quantum (probably 1 token, possibly variable per node). Once the trie lands, the constant gets renamed/removed and lookup sites migrate to the trie API. The intermediate alias means lookup sites are *already isolated* from physical KV geometry sites — clean diff.
- 1.4 `hashTokenRange` is the trie's natural hashing primitive; `hashPagePrefix` becomes a legacy wrapper or gets retired with PageManager-based lookup.
- 1.5 `findPairByPrefixHash` becomes `trie.findLongestSharedPrefix(tokens:)` or similar; the rename now correctly signals "this is the old page-grained lookup" so the trie's API doesn't accidentally adopt the misleading name.

**Cross-track:** the rename pass is *expressly designed* not to land before any algorithmic track. Land PR-1+PR-2 immediately (mechanical, low conflict). Hold PR-3 until at least one of A/B/C/D is in flight so the wire-name and comment decisions can be revisited with concrete refactor diffs in hand.

---

**Total estimated churn:** ~250 LoC across ~15 files; ~80% Swift (compile-checked), ~15% Markdown, ~5% Python doc strings. Zero JSON-wire breakage if PRs are split as recommended. Zero changes in `/Users/mdot/sillytavern-fork/`.
