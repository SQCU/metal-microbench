// PageManager — per-session ownership of KV-cache pages, with explicit
// exhaustion behaviour (no silent wrap) and content-addressed prefix reuse.
//
// Previous behaviour: LmEngine.openSession laid down a fixed
// block_table[slot][p] = slot*MAX_PAGES_PER_SLOT + p mapping, so two
// sessions pinned to different slots automatically got disjoint KV strips.
// Perfect for 4 toy sessions; completely wrong once we want:
//   - more sessions than slots (slot reuse requires releasing pages)
//   - a session that outgrows MAX_PAGES_PER_SLOT
//   - two sessions sharing a common prompt prefix (the whole point of
//     prefix caching — avoid reprocessing the first N tokens)
//
// What this adds:
//   - A free list of physical pages; `alloc` / `release` are explicit.
//   - Per-page content hash (FNV-1a over the page's token IDs), indexed
//     so a second session submitting the same first K pages gets the
//     same physical pages handed back (refcounted, read-only sharing).
//   - Fresh-allocated pages are single-owner + single-refcount. Only
//     pages that came from the content cache are multi-owner; if a
//     session *diverges* from the shared prefix, it must copy-on-write
//     (take a fresh page and copy the KV from the shared one). We don't
//     need CoW yet — the engine never writes to pre-populated prefix
//     pages (they come from a completed prefill that the shared session
//     already ran), so read-sharing is safe until a session starts
//     *extending* beyond the shared prefix into pages it doesn't own.
//   - Exhaustion throws `PageManagerError.outOfPages` instead of the old
//     silent `% TOTAL_PAGES` wrap that used to corrupt pages 0..N-1
//     once we went past the pool.
//
// What this does NOT do yet:
//   - LRU / refcount-aware eviction. When the pool fills, callers must
//     release sessions to free pages. A real production scheduler will
//     want eviction (e.g., drop the lowest-refcount cached prefix to
//     admit a new session's pages). Leaving that for when we measure
//     actual cache pressure.
//   - Copy-on-write when a session writes to a shared page. Today we
//     assume sessions only write to their own fresh pages (which is
//     true in the engine — shared pages come pre-populated from a
//     previous session's prefill).
//   - Cross-layer divergence. Each GGUF layer has its own K/V cache
//     buffer but they share the same phys-page indices; PageManager
//     tracks phys pages not per-layer cells. That's fine: if phys page
//     P is "owned" by session S for positions [a, b), it's the same in
//     every layer.
import Foundation

enum PageManagerError: Error {
    case outOfPages(needed: Int, available: Int)
    case doubleFree(physPage: Int)
    // 2026-05-07: removed `notOwned` — no per-session owner concept.
    // decref() is a refcount op; double-decref is silently ignored.
}

// A logical page: its refcount + the content hash used for prefix-cache
// lookups.
//
// ONE PAGE=16 for EVERY layer: a 16-token logical page P is a SINGLE physical
// page (ownedPages[P]) covering positions [16P..16P+15] in every per-layer K/V
// buffer. Full and sliding-window layers both index block_table[slot][pos/16];
// there is no dual /8-vs-/16 granularity and no per-page partner — the prior
// "full-attn page pair" bookkeeping (pairMate / SlidePairContents) is deleted.
private struct PageInfo {
    // 2026-05-07 (anonymous-pool refactor): pages are no longer tagged
    // with the set of session IDs that own them. The pool is a uniform
    // anonymous resource; callers (engine, vision tower, etc.) hold
    // their own [Int] lists of phys pages they're currently using and
    // call decref() to release. The single number `refcount` replaces
    // the `owners: Set<Int>` ledger — same information (refcount ==
    // owners.count) but no per-session bookkeeping inside PageManager.
    var refcount: Int           // number of live page references
    var contentHash: UInt64?    // nil = page isn't content-indexed (fresh/dirty)
    var lastAccessTick: UInt64
    // 2026-06 KV-retention decoupling — BILINEAR reuse-value (citation × recency).
    //
    // The value function (kv_pages_are_not_connection_stateful memory + G3/G4/G5):
    // a page's worth to keep is NOT pure recency (LRU evicts a hot shared
    // tool-prefix before a marginally-newer one-off suffix) and NOT pure
    // citation-count (LFU pins a dead-but-historically-popular prefix forever).
    // It is a RECENCY-DECAYED CITATION score:
    //
    //   value(page) = Σ_citations decay(now − t_citation)
    //
    // maintained as an EWMA: on each new citation,
    //   citationScore = citationScore · decay(Δticks) + 1
    // where Δticks = clockTick − lastCitationTick and decay(Δ) = HALF_LIFE^Δ.
    //
    // A CITATION is a DISTINCT ADOPTER — a new generation adopting the page
    // via findLongestPrefix (adoptSharedPrefixPages). It is NOT a per-AR-step
    // re-read by the page's own generation; recency (lastAccessTick, bumped on
    // the forward-pass KV READ in touchAccess) already covers self-rereads, and
    // counting them would re-create the immortality LFU suffers from.
    //
    // Eviction picks the MIN decayed-citation score over the eligible set.
    // lastCitationTick lets the eviction scan decay the stored score to "now"
    // before comparing, so a once-popular-but-now-cold prefix ranks correctly
    // low. lastCitationTick==0 (never cited) keeps citationScore==0 = coldest.
    var citationScore: Double   // EWMA of decayed distinct-adopter citations
    var lastCitationTick: UInt64 // clockTick at last citation (for decay-to-now)
}


final class PageManager {
    // Total physical pages this manager describes (for `pages[]` array
    // sizing). Equals SCRATCH_STRIP + cache-pool count — the manager
    // tracks every physical page in the K/V buffer, but only the cache
    // range [basePage, basePage + numPoolPages) is allocated to
    // sessions; pages [0, SCRATCH_STRIP) are reserved for silenced-slot
    // scratch and never appear on the free list.
    let numPhysPages: Int
    // Cache pool range: alloc/free operate over [basePage, basePage + numPoolPages).
    // Scratch pages live OUTSIDE this range — see SCRATCH_PAGE_BASE.
    let basePage: Int
    let numPoolPages: Int
    let pageSize: Int

    // Physical page state: pages[p] describes phys page p. Sized at
    // numPhysPages so any phys index (including scratch-strip indices)
    // can be looked up; only the [basePage, basePage+numPoolPages)
    // range will ever be referenced through alloc/free paths.
    private var pages: [PageInfo]
    // Free pages split by whether they still carry reusable K/V content.
    // The previous single mixed freeList made allocFresh scan O(freePages)
    // for every new logical page to find an uncached entry. Long repeated
    // swipe/prefix-cache workloads turned that into pages_needed*freePages
    // allocator CPU. These two stacks preserve the policy preference
    // (uncached first, cached only under pressure) with O(1) pop/remove.
    private var freeUncached: [Int] = []
    private var freeCached: [Int] = []
    private var freeUncachedPos: [Int: Int] = [:]
    private var freeCachedPos: [Int: Int] = [:]
    // Followup 3 (2026-05-23): contentIndex deleted. The radix trie
    // (radix_trie.swift, RadixTrie) is now the sole source of truth
    // for adoption lookups; PageManager only tracks refcounts +
    // content-hash bookkeeping per phys page so the eviction callback
    // can notify the trie. Under ONE PAGE=16 a 16-token page is a single
    // phys page — no per-page partner bookkeeping.
    //
    // Eviction callback (Track D fold-in): invoked when allocFresh
    // forcibly evicts a cached page. LmEngine wires this up to
    // RadixTrie.invalidateAnchorFor so stale anchors get unlinked.
    // Receives (physPage, oldHash).
    var onPageEvicted: ((Int, UInt64) -> Void)?
    // Tier 1 cold-KV SSD demote (2026-06): fired at the SAME forced-eviction
    // victim point as onPageEvicted, but INSTEAD OF it, when Tier 1 is enabled.
    // (phys, oldHash) -> demoted? The engine's closure gathers the page's 60 K/V
    // slices, pwrites them to a free SSD slot, retags the trie anchor(s)
    // RAM(phys)->SSD(slot), and returns true. If it returns false (SSD full) OR
    // the closure is nil (Tier 1 disabled) the existing onPageEvicted DROP runs
    // verbatim. The victim is always a refcount==0 CACHED page (a refcount>0
    // page is structurally absent from the free stacks), so "NEVER demote a
    // refcount>0 page" holds BY CONSTRUCTION. PageManager stays Metal-free — the
    // gather/pwrite/retag all live in the engine closure, same decoupling as
    // onPageEvicted/onPageCommitted.
    var onPageDemoteCandidate: ((Int, UInt64) -> Bool)?
    // Tier 0 pin-on-grow (2026-06): fired EXACTLY ONCE per never-before-exposed
    // phys page (resident-frontier growth in growPool), for chunk-granular KV
    // wiring. The engine wires this to ensureChunkResidentForPage, which pins
    // the page's KV chunk (all 30 layers' K+V) into the KV residency set before
    // the page is handed out / first-touched. PageManager stays Metal-free (no
    // MTLBuffer / weights dependency) — same decoupling as onPageEvicted. The
    // reuse/decref paths (free-stack pop, freeUncached.append on decref) do NOT
    // call this — only the growFrontier advance does.
    var onPageCommitted: ((Int) -> Void)?
    // 2026-05-07: deleted `sessionPages: [Int: [Int]]`. Callers now hold
    // their own list of phys pages they reference and call decref()
    // when done. No per-session ledger lives in the page manager —
    // pages are anonymous, identified only by content hash + refcount.
    private var clockTick: UInt64 = 0

    // ── Dynamic pool growth (G2, 2026-06) ─────────────────────────────────
    // The free list is no longer eagerly populated over the whole pool. The
    // per-layer K/V device buffers are sized at the HARD CAP (basePage ..
    // basePage+poolCapPages) but are storage-mode-shared / lazy-committed —
    // physical RAM is only committed when a phys page is first WRITTEN
    // (zeroPhysPageKV). So the pool "grows on demand": allocFresh exposes the
    // next contiguous page (growFrontier) only when both free stacks are empty,
    // up to poolCapPages (the budget-derived cap). committedHighWater records
    // the highest watermark ever exposed so /health (stats().totalPages) shows
    // real growth, while poolCapPages bounds it to the memory budget.
    //
    // BIT-EXACTNESS: pages are exposed lowest-index-first, one at a time, ONLY
    // when the free stacks are empty — the identical order the old eager-init
    // (reversed range, popped from the tail) handed them out. A decref'd page
    // returns to the free stack and is reused before growFrontier advances
    // (allocFresh checks the free stacks first), exactly as before. So the
    // sequence of phys indices handed to logical pages is unchanged → no
    // physical page backing any position moves → K/V bytes are identical.
    private var growFrontier: Int        // next never-yet-exposed page index
    private let poolCapPages: Int        // hard cap on exposed pages (budget)
    private var committedHighWater: Int = 0  // pages ever exposed (telemetry)

    init(numPhysPages: Int, pageSize: Int, basePage: Int = 0,
          numPoolPages: Int? = nil, poolCapPages: Int? = nil) {
        // 2026-05-06: split scratch-strip from cache-pool addressing
        // (codex RCA). PageManager allocates physical-page indices
        // from [basePage, basePage + numPoolPages); pages outside that
        // range are reserved (typically scratch at low indices). The
        // single-arg `init(numPhysPages:pageSize:)` default keeps the
        // legacy behavior (pool = entire range starting at 0) so
        // existing tests / callers that don't reserve a scratch strip
        // are unchanged.
        let poolCount = numPoolPages ?? numPhysPages
        self.numPhysPages = numPhysPages
        self.basePage = basePage
        self.numPoolPages = poolCount
        self.pageSize = pageSize
        // Budget-derived hard cap. Defaults to the full pool (poolCount) so a
        // caller that doesn't pass a budget keeps the legacy capacity; the
        // engine passes the KV_MEM_BUDGET_FRAC-derived value. Clamp to [0,
        // poolCount] — the device buffers only back poolCount pages.
        self.poolCapPages = max(0, min(poolCount, poolCapPages ?? poolCount))
        self.pages = (0..<numPhysPages).map { _ in
            PageInfo(refcount: 0, contentHash: nil,
                     lastAccessTick: 0, citationScore: 0, lastCitationTick: 0)
        }
        // Lazy free list: nothing pre-pushed. growFrontier starts at the pool
        // base and advances (lowest-first) as allocFresh exposes pages.
        self.growFrontier = basePage
        self.freeUncachedPos.reserveCapacity(self.poolCapPages)
        self.freeCachedPos.reserveCapacity(self.poolCapPages)
    }

    // Expose the next never-yet-seen page if the budget allows. Returns the
    // freshly-exposed phys index (already pushed onto freeUncached), or nil if
    // the pool is at its budget cap. Lowest-index-first, one at a time.
    private func growPool() -> Int? {
        guard growFrontier < basePage + poolCapPages else { return nil }
        let phys = growFrontier
        growFrontier += 1
        committedHighWater = max(committedHighWater, growFrontier - basePage)
        // Newly-exposed page is uncached, refcount 0.
        freeUncachedPos[phys] = freeUncached.count
        freeUncached.append(phys)
        // Tier 0 pin-on-grow: this is the SOLE resident-frontier growth point.
        // Fire BEFORE returning so the page's KV chunk is wired-resident before
        // allocFresh hands the page to a session and zeroPhysPageKV first-touches
        // it (same call stack — completes synchronously). Idempotent on the
        // engine side: a page whose chunk is already resident short-circuits.
        onPageCommitted?(phys)
        return phys
    }

    // O(1) removal from a free stack at a known index. Uses the
    // swap-with-last-then-pop pattern so removal doesn't shift the
    // tail; the position map for the swapped element and removed value is
    // updated here.
    private func freeStackPop(_ list: inout [Int],
                              _ pos: inout [Int: Int],
                              at idx: Int) -> Int {
        let last = list.count - 1
        let removed = list[idx]
        if idx != last {
            let movedPhys = list[last]
            list[idx] = movedPhys
            pos[movedPhys] = idx
        }
        list.removeLast()
        pos.removeValue(forKey: removed)
        return removed
    }

    private func freeUncachedPop(at idx: Int) -> Int {
        return freeStackPop(&freeUncached, &freeUncachedPos, at: idx)
    }

    private func freeCachedPop(at idx: Int) -> Int {
        return freeStackPop(&freeCached, &freeCachedPos, at: idx)
    }

    // O(1) push onto the appropriate free stack with position tracking.
    private func freePush(_ phys: Int) {
        if pages[phys].contentHash == nil {
            freeUncachedPos[phys] = freeUncached.count
            freeUncached.append(phys)
        } else {
            freeCachedPos[phys] = freeCached.count
            freeCached.append(phys)
        }
    }

    private func removeFromFreeStacksIfPresent(_ phys: Int) {
        if let idx = freeUncachedPos[phys] {
            _ = freeUncachedPop(at: idx)
        }
        if let idx = freeCachedPos[phys] {
            _ = freeCachedPop(at: idx)
        }
    }

    private func moveFreePageToCachedIfPresent(_ phys: Int) {
        if let idx = freeUncachedPos[phys] {
            _ = freeUncachedPop(at: idx)
            freeCachedPos[phys] = freeCached.count
            freeCached.append(phys)
        }
    }

    private func moveFreePageToUncachedIfPresent(_ phys: Int) {
        if let idx = freeCachedPos[phys] {
            _ = freeCachedPop(at: idx)
            freeUncachedPos[phys] = freeUncached.count
            freeUncached.append(phys)
        }
    }

    // FNV-1a over a page's token IDs, optionally mixed with a cvec-state
    // digest that represents the control-vector intervention pattern
    // applied across the page's position range.
    //
    // `cvecDigest == 0` reproduces the pre-steering-aware hash exactly —
    // unsteered sessions keep hitting pages that were promoted before
    // this parameter existed, and two unsteered sessions continue to
    // share cached prefixes. Non-zero digests partition the namespace:
    // two sessions that submit identical tokens under different ADSR
    // envelope parameters / layers / cvec ids get different keys and
    // correctly MISS each other's pages, because their K/V values would
    // diverge (steering at layer L feeds layer L+1's QKV projection).
    //
    // Digest construction is the caller's responsibility (see
    // Session.cvecDigestForPage). We use xor-then-mix so digest==0 is a
    // strict identity; any non-zero digest perturbs the hash irrecoverably.
    static func hashPage(_ tokens: ArraySlice<UInt32>, cvecDigest: UInt64 = 0) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for t in tokens {
            h ^= UInt64(t)
            h = h &* 0x100000001b3
        }
        if cvecDigest != 0 {
            h ^= cvecDigest
            h = h &* 0x100000001b3
        }
        return h
    }

    // Followup 3 (2026-05-23): findByHash deleted; adoption goes
    // through RadixTrie.findLongestPrefix.

    // Grab a phys page for `sessionId`. The free list holds all
    // refcount==0 pages; some carry valid content hashes (cache entries
    // from prior sessions), some don't (genuinely uncached). We pick in
    // this order:
    //   1. A free uncached page (no hash to drop, free reuse).
    //   2. Otherwise grow the pool by one fresh page (budget permitting).
    //   3. Otherwise: forced eviction — the MIN decayed-citation cached
    //      page → evict its hash, reuse.
    //
    // Critically we DO NOT drop a content hash unless steps 1-2 found
    // nothing and we're forced to overwrite. The previous "drop on every
    // pop" logic shredded cache entries that the SAME request was about to
    // probe for: iter N's session pop-and-evicts iter N-1's pages off
    // the free list (because the entire free list was content-cached
    // immediately after iter N-1 closed), then iter N's later probes
    // for those same hashes miss. With no-eager-eviction, iter N
    // shareExisting-adopts iter N-1's pages first (refcount++, removes
    // them from free list), and only then allocFresh runs for the
    // genuine tail — by which point the free list contains uncached
    // pages.
    //
    // 2026-06 KV-retention decoupling: forced eviction now picks the MIN of
    // a recency-DECAYED CITATION score (value(page)=Σ_citations decay(now −
    // t_citation)) instead of the pure LRU lastAccessTick min — see PageInfo.
    // citationScore. Pure-LRU evicts a hot shared prefix before a marginally-
    // newer one-off suffix; pure-LFU pins a dead-but-popular prefix forever.
    // The decayed-citation min is the bilinear value that does neither. The
    // eligible set is freeCached (refcount==0 pages) — a page listed by a
    // live wants-slot generation has refcount>0 and is structurally absent
    // from the free stacks, so it is never an eviction victim (eligibility
    // = "not active-step AND not listed by any wants-slot session"; the
    // engine decrefs a session's pages at .done so finished/idle work falls
    // here and becomes evictable, while live working sets stay protected).
    //
    // 2026-05-07: anonymous-pool refactor — callers track their own
    // page references; PageManager only knows refcounts and content
    // hashes. Returns a phys page with refcount=1; caller MUST call
    // decref(physPage:) when done with it.
    func allocFresh() throws -> Int {
        let phys: Int
        if !freeUncached.isEmpty {
            // Prefer uncached pages. No prefix-cache entry is dropped.
            phys = freeUncachedPop(at: freeUncached.count - 1)
        } else if let grown = growPool() {
            // Pool grows on demand (budget permitting) before we evict any
            // cached page — an unexposed page has no hash to drop, exactly
            // as the eager-init pool would have preferred it (step 1).
            phys = freeUncachedPop(at: freeUncachedPos[grown]!)
        } else if !freeCached.isEmpty {
            // Forced eviction: pool at budget cap and every free page is
            // cached. Drop the MIN decayed-citation cached page (the lowest
            // bilinear reuse-value) — NOT the stack top, NOT pure LRU. Decay
            // each candidate's stored citationScore to "now" before
            // comparing, so a once-popular-but-now-cold prefix ranks low.
            // Ties (e.g. all never-cited score==0) break to the lower
            // lastAccessTick (recency) — preserving the audit-#7 LRU-not-MRU
            // property for the uncited common case. O(n) scan is fine: forced
            // eviction only fires when uncached pages and growth are exhausted.
            var victimIdx = 0
            var victimScore = Double.greatestFiniteMagnitude
            var victimTick = UInt64.max
            for (i, ph) in freeCached.enumerated() {
                let sc = decayedCitationScore(ph, at: clockTick)
                let tk = pages[ph].lastAccessTick
                if sc < victimScore || (sc == victimScore && tk < victimTick) {
                    victimScore = sc; victimTick = tk; victimIdx = i
                }
            }
            phys = freeCachedPop(at: victimIdx)
            // Drop the now-orphaned content hash + its pair.
            // Followup 3 (2026-05-23): contentIndex deleted; only the
            // PageInfo bookkeeping + eviction callback into the trie
            // remain. The trie is responsible for unlinking the anchor.
            var p = pages[phys]
            if let h = p.contentHash {
                // VALUE-GATED DEMOTE (2026-06): the demote-vs-drop decision now
                // consults the victim's PROVEN re-adoption value, not just
                // `contentHash != nil`. The min-scan above selected this page as
                // the LOWEST-value cached page; here we ask whether it has ANY
                // proven reuse worth a wasted SSD write. The signal is the
                // citation MAGNITUDE (p.citationScore), bumped ONLY by
                // recordCitation on a DISTINCT-ADOPTER adoption — never by a
                // page's own AR re-reads. A page that was never adopted/cited (a
                // one-off tail of a distinct prompt) has citationScore == 0 and
                // therefore ~0 reload value: it would never be re-adopted, so
                // demoting it is a pure wasted write (the 49,804-demote / 175 GB
                // distinct-prompt storm). Such a page DROPs (onPageEvicted, no
                // SSD write) exactly as if Tier 1 were disabled.
                //
                // We do NOT gate on decayedCitationScore (the eviction RANK):
                // a fresh uncited page has recency≈1 so its decayed value ≈ 1.0
                // (citationScore 0 ⇒ 1+0), which would NOT discriminate the
                // never-adopted tail. The citation magnitude does: it is >0 iff
                // some distinct generation adopted the page at least once.
                //
                // Threshold is in citation-magnitude units (EWMA of decayed
                // distinct-adopter citations), env-tunable via
                // KV_DEMOTE_VALUE_FLOOR; default 0.0 ⇒ predicate is
                // `citationScore > 0` = "demote ONLY pages with ≥1 proven
                // citation, drop the rest". No magic number — the floor lives in
                // the same units as the existing value model, and the default
                // maps to the principled "proven-adoption" rule.
                let worthDemoting = p.citationScore > PageManager.demoteValueFloor
                // Tier 1: try to DEMOTE the cold (refcount==0) cached page to
                // the SSD store instead of DROPPING it. The closure gathers +
                // pwrites + retags the trie anchor RAM->SSD and returns true. If
                // Tier 1 is disabled (nil closure) or the SSD store is full
                // (returns false) OR the page failed the value gate, fall
                // through to the existing DROP path (onPageEvicted ->
                // invalidateAnchorFor), bit-identical to before. Gather READS the
                // page bytes here BEFORE the caller (ensurePages) zeroes the
                // reused page, so the demote captures valid K/V. Track D: notify
                // the radix trie BEFORE we mutate PageInfo so the callback can
                // read accurate state.
                let demoted = worthDemoting
                    ? (onPageDemoteCandidate?(phys, h) ?? false)
                    : false
                if !demoted {
                    onPageEvicted?(phys, h)
                }
                p.contentHash = nil
            }
            pages[phys] = p
        } else {
            // No uncached page, pool at budget cap, and no evictable cached
            // page either (every page is refcount>0 = listed by a live
            // generation). Genuine exhaustion — caller (engine admission /
            // ensurePages) must shed a session or backpressure the submit.
            throw PageManagerError.outOfPages(needed: 1, available: 0)
        }
        clockTick += 1
        var p = pages[phys]
        p.refcount = 1
        p.lastAccessTick = clockTick
        pages[phys] = p
        return phys
    }

    // Recency-decayed citation value of a page, evaluated at `now` (ticks).
    // score(now) = storedScore · HALF_LIFE^(now − lastCitationTick).
    // A page never cited (lastCitationTick==0, storedScore==0) returns 0 —
    // the coldest possible value, evicted first. Decay base is configurable
    // via env KV_CITATION_HALF_LIFE_TICKS (default 4096 ticks); larger =
    // citations matter longer. Pure function of PageInfo + now: read-only,
    // no mutation, safe to call from the eviction scan.
    private func decayedCitationScore(_ phys: Int, at now: UInt64) -> Double {
        // BILINEAR reuse-value = recency × citation-magnitude. RECENCY is the
        // decaying MULTIPLIER, anchored on lastAccessTick (last forward
        // dependency): it drives the whole value toward 0 as a page goes
        // stale, so a once-popular-but-now-cold prefix falls BELOW a freshly-
        // touched uncited page (value ≈ 1) and is finally evicted. The prior
        // form decayed the CITATION SUM alone — that asymptotes >0 and never
        // crosses an uncited page's 0, so a stale heavily-cited prefix was
        // IMMORTAL (the G5 failure / the "stuck-en-cache forever" class). The
        // citation magnitude (1 + citationScore) is the breadth bonus that
        // keeps a HOT shared prefix last-evicted (G3). value =
        // HALF_LIFE^(now − lastAccessTick) · (1 + citationScore). Half-life
        // via KV_CITATION_HALF_LIFE_TICKS. Pure read-only fn; citationScore /
        // lastAccessTick never touch K/V bytes — eviction ranking is
        // decoupled from cache contents.
        let p = pages[phys]
        let age = now >= p.lastAccessTick ? Double(now - p.lastAccessTick) : 0
        let recency = pow(PageManager.citationDecayBase, age)
        return recency * (1.0 + p.citationScore)
    }

    // Value-gate floor for the demote-vs-drop decision (allocFresh forced
    // eviction). A cached eviction victim is written to SSD (DEMOTED) only if
    // its citation MAGNITUDE strictly exceeds this floor; otherwise it is
    // DROPPED (no SSD write). Units = citation magnitude (EWMA of decayed
    // distinct-adopter citations, same as PageInfo.citationScore). Default 0.0
    // ⇒ predicate `citationScore > 0` = "demote ONLY pages adopted ≥1 time".
    // Env KV_DEMOTE_VALUE_FLOOR raises the bar (e.g. require >1 distinct
    // adopter). Negative values are clamped to 0 (a floor below 0 would admit
    // never-cited pages, defeating the gate). Lazily read once from the env.
    static let demoteValueFloor: Double = {
        let env = ProcessInfo.processInfo.environment["KV_DEMOTE_VALUE_FLOOR"]
        let floor = (env.flatMap { Double($0) }) ?? 0.0
        return floor >= 0 ? floor : 0.0
    }()

    // decay(Δ=1) per-tick multiplier = HALF_LIFE^(1/halfLifeTicks) computed
    // as 2^(-1/halfLifeTicks). Lazily read once from the environment.
    static let citationDecayBase: Double = {
        let env = ProcessInfo.processInfo.environment["KV_CITATION_HALF_LIFE_TICKS"]
        let halfLife = (env.flatMap { Double($0) }) ?? 4096.0
        let hl = halfLife > 0 ? halfLife : 4096.0
        return pow(2.0, -1.0 / hl)
    }()

    // Record a CITATION (a distinct adopter took this page via the prefix
    // cache). EWMA update: decay the stored score to now, then add 1. Bumps
    // the citation timestamp; does NOT touch refcount, contentHash, pairMate,
    // or lastAccessTick (recency is a separate signal, bumped on the forward-
    // pass KV read via touchAccess). Out-of-range / scratch indices ignored.
    // citationScore never touches K/V bytes — it cannot perturb numerics.
    func recordCitation(physPage: Int) {
        guard physPage >= 0 && physPage < pages.count else { return }
        clockTick += 1
        var p = pages[physPage]
        let decayed = (p.lastCitationTick == 0)
            ? 0.0
            : p.citationScore * pow(PageManager.citationDecayBase,
                                    Double(clockTick - p.lastCitationTick))
        p.citationScore = decayed + 1.0
        p.lastCitationTick = clockTick
        pages[physPage] = p
    }

    // Increment refcount on an existing page (used to be `shareExisting`).
    // If the page was previously refcount=0 and on the free list, it's
    // pulled out (resurrection of a cached page). Content hash is
    // preserved (read-only sharing across whichever callers hold the
    // refcount). Caller is responsible for matching decref().
    func incref(physPage: Int) {
        clockTick += 1
        var p = pages[physPage]
        if p.refcount == 0 {
            // Page is in free list but still has valid content. Pull it
            // back out before handing out the new reference.
            removeFromFreeStacksIfPresent(physPage)
        }
        p.refcount += 1
        p.lastAccessTick = clockTick
        pages[physPage] = p
    }

    // PageInfo-only writer: stamps a phys page with the content hash so the
    // eviction callback in allocFresh can notify the trie. Under ONE PAGE=16
    // a 16-token page is a SINGLE phys page (ownedPages[P] covering
    // [16P..16P+15]) — no partner page bookkeeping.
    //
    // Idempotent: re-stamping with the same hash is a no-op modulo
    // lastAccessTick refresh; re-stamping with a DIFFERENT hash
    // overwrites (the trie's insertAnchor is similarly first-writer-
    // wins via its internal walk, so callers should follow trie
    // semantics).
    func markContentIndexed(phys: Int, contentHash: UInt64) {
        var p = pages[phys]
        p.contentHash = contentHash
        pages[phys] = p
        if p.refcount == 0 { moveFreePageToCachedIfPresent(phys) }
    }

    // Decrement refcount. If it drops to 0, the page goes on the free
    // list BUT its content-index entry stays — KV data is still valid
    // until something overwrites it (only at allocFresh time). This
    // enables cache hits across request lifetimes: request A finishes
    // and decrefs its pages; request B with the same prefix probes
    // contentIndex, hits the (now-refcount=0-but-still-cached) pages,
    // calls incref() to resurrect them. No prefill work needed.
    //
    // No sessionId arg, no `notOwned` error — pages are anonymous.
    // Caller is responsible for not double-decref'ing or decref'ing
    // pages they didn't incref.
    func decref(physPage: Int) {
        // Out-of-range phys = programmer error (a caller decref'ing a page
        // index that was never allocated). Hard precondition — crash loudly,
        // never a warn-and-proceed balance-bug masker.
        precondition(physPage >= 0 && physPage < pages.count,
                     "decref out-of-range phys=\(physPage) (pool=\(pages.count))")
        var p = pages[physPage]
        // decref() is idempotent by design (see field doc): double-decref of
        // an already-free page is a benign no-op. Refcount==0 falls through.
        if p.refcount > 0 {
            p.refcount -= 1
            pages[physPage] = p
            if p.refcount == 0 {
                freePush(physPage)
                // contentHash preserved for potential cache hit later.
            }
        }
    }

    // Decayed-citation reuse-value of a page evaluated at the current tick.
    // Public read accessor for the engine's admission-pressure-cancel ranking
    // (shed the lowest-value generation). 0 = never cited / coldest.
    func pageReuseValue(_ phys: Int) -> Double {
        guard phys >= 0 && phys < pages.count else { return 0 }
        return decayedCitationScore(phys, at: clockTick)
    }

    func pageRefcount(_ phys: Int) -> Int {
        guard phys >= 0 && phys < pages.count else { return 0 }
        return pages[phys].refcount
    }

    // B1 (2026-06 KV-retention decoupling): forward-read recency stamp.
    // Telemetry ONLY — bumps lastAccessTick so a page an in-flight forward
    // pass DEPENDS ON looks hot to the LRU eviction picker in allocFresh
    // (the freeCached min-scan). Unlike incref/allocFresh this does NOT touch
    // refcount, the free list, contentHash, or pairMate, so it cannot perturb
    // numerics or ownership — it only adjusts eviction recency.
    // Out-of-range / scratch indices are ignored.
    func touchAccess(physPage: Int) {
        guard physPage >= 0 && physPage < pages.count else { return }
        clockTick += 1
        pages[physPage].lastAccessTick = clockTick
    }

    // Diagnostic snapshot.
    //
    // Followup 3 (2026-05-23): `cachedHashes` historically counted the
    // size of contentIndex. With that dict gone, we count the number
    // of phys pages that currently carry a non-nil contentHash — same
    // semantic ("how many pages are content-addressed"), now derived
    // from PageInfo state. The richer adoption-anchor count lives in
    // RadixTrie.stats().anchorCount, surfaced in the engine-state
    // `prefix_trie` JSON block.
    struct Stats {
        let totalPages: Int
        let freePages: Int
        let cachedHashes: Int
        // 2026-05-07: pages-in-use count replaces "active sessions".
        // Pages are anonymous; "in-use" means refcount > 0.
        let pagesInUse: Int
        // G2 dynamic-pool telemetry (2026-06). totalPages reports the BUDGET
        // CAP (how big the pool may grow). committedPages is the high-water of
        // pages ever exposed — the "growth" /health should show. poolCapacity
        // duplicates totalPages explicitly for new consumers; legacy consumers
        // keep reading totalPages.
        let committedPages: Int
        let poolCapacityPages: Int
    }
    func stats() -> Stats {
        var cached = 0
        for p in pages { if p.contentHash != nil { cached += 1 } }
        // freePages = free stacks (exposed, refcount==0) PLUS the still-
        // growable headroom (budget cap minus pages ever exposed). Admission
        // backpressure compares freePages against a floor; on a fresh engine
        // committedHighWater is 0, so free MUST include the growth headroom or
        // every submit would be rejected for "0 free".
        let growHeadroom = poolCapPages - committedHighWater
        let freeNow = freeUncached.count + freeCached.count + growHeadroom
        return Stats(totalPages: poolCapPages,
                     freePages: freeNow,
                     cachedHashes: cached,
                     pagesInUse: committedHighWater - freeUncached.count - freeCached.count,
                     committedPages: committedHighWater,
                     poolCapacityPages: poolCapPages)
    }

    // Per-page diagnostic record (used by the engine-state endpoint that
    // backs /v1/engine/state for the static visualizers). A page is
    // returned IFF it has refcount > 0 (in-use) or carries a contentHash
    // (cached + currently free, eligible for adoption). Pages that have
    // never been touched are omitted to keep the snapshot small even
    // when the pool is 8192 entries.
    struct PageRecord {
        let phys: Int                  // physical page index
        let refcount: Int              // > 0 = in-use; 0 = free-but-cached
        let contentHash: UInt64?       // FNV-1a digest of the tokens on this page
    }

    func livePageSnapshot() -> [PageRecord] {
        var out: [PageRecord] = []
        // Cap the snapshot to avoid pathological cases where every page
        // is content-indexed; this is a visualization, not a forensic
        // dump. 4096 covers the worst realistic case (full pool in use
        // or fully cached) without producing megabyte-sized payloads.
        let cap = 4096
        for phys in basePage..<(basePage + numPoolPages) {
            let p = pages[phys]
            if p.refcount == 0 && p.contentHash == nil { continue }
            out.append(PageRecord(phys: phys, refcount: p.refcount,
                                   contentHash: p.contentHash))
            if out.count >= cap { break }
        }
        return out
    }
}
