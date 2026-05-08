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

// A logical page: which session owns it, at which logical index, plus the
// content hash used for prefix-cache lookups.
//
// Pair bookkeeping: Gemma-4 uses PAGE_SLIDE=16 for sliding-window layers
// and PAGE_FULL=8 for full-attention layers, but shares one block_table
// per slot. That means each 16-token slide page spans TWO full-attn
// pages in the full-K/V cache (the first-half lives at the same phys
// index as the slide page, the second-half lives at block_table[index+1]).
// To share a slide-page worth of K/V correctly across sessions, the
// content index has to track BOTH members as a pair — adopting one
// without the other leaves the full-attn K/V at positions [page_start+8,
// page_start+15] as zeros (or stale data), producing KL~0.38 divergence
// vs. fresh compute. `pairMate` is the phys index of the other member
// of this content-hash pair (nil when this page isn't part of a promoted
// pair).
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
    var pairMate: Int?          // other phys page sharing this hash (see header)
    var lastAccessTick: UInt64
}

// Value of contentIndex[hash]. Ordered: slidePrimary corresponds to
// block_table[P] (slide cache for 16 tokens + full cache for first 8);
// fullSibling corresponds to block_table[P+1] (full cache for the second
// 8 tokens at that 16-token slide-page range, plus whatever slide data
// gets written there by a later prefill tile).
struct SharedPagePair {
    let slidePrimary: Int
    let fullSibling: Int
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
    // Free list of phys pages with refcount==0. LIFO (cache-warm reuse).
    private var freeList: [Int] = []
    // 2026-05-07: parallel index map for O(1) removal from freeList by
    // physPage value. Previous shareExisting used `freeList.firstIndex(of:)`
    // which was O(|freeList|) — at numPoolPages=8192 with 10 page adoptions
    // per session admission, that was 80K linear ops per admission. With
    // freeListPos[physPage] = currentIndexInFreeList, both removal paths
    // (allocFresh's `remove(at:)` and shareExisting's `firstIndex(of:)
    // + remove(at:)`) become O(1) via the swap-with-last-and-pop helper
    // freeListPop(at:).
    private var freeListPos: [Int: Int] = [:]
    // contentIndex[contentHash] = pair of phys pages carrying the
    // 16-token slide-page's data in both cache layouts. See the
    // PageInfo header for why sharing has to be a pair.
    private var contentIndex: [UInt64: SharedPagePair] = [:]
    // 2026-05-07: deleted `sessionPages: [Int: [Int]]`. Callers now hold
    // their own list of phys pages they reference and call decref()
    // when done. No per-session ledger lives in the page manager —
    // pages are anonymous, identified only by content hash + refcount.
    private var clockTick: UInt64 = 0

    init(numPhysPages: Int, pageSize: Int, basePage: Int = 0,
          numPoolPages: Int? = nil) {
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
        self.pages = (0..<numPhysPages).map { _ in
            PageInfo(refcount: 0, contentHash: nil, pairMate: nil, lastAccessTick: 0)
        }
        // Free list spans only the pool range. Push in reverse so the
        // lowest-indexed pool page pops first (cache-warm-friendly).
        self.freeList = Array((basePage..<(basePage + poolCount)).reversed())
        // Build the position-tracking map (parallel structure for O(1)
        // freeList membership lookups + swap-and-pop removal).
        self.freeListPos.reserveCapacity(self.freeList.count)
        for (i, phys) in self.freeList.enumerated() {
            self.freeListPos[phys] = i
        }
    }

    // O(1) removal from `freeList` at a known index. Uses the
    // swap-with-last-then-pop pattern so removal doesn't shift the
    // tail; freeListPos for the swapped element is updated. Caller is
    // responsible for clearing freeListPos for the removed value.
    private func freeListPop(at idx: Int) -> Int {
        let last = freeList.count - 1
        let removed = freeList[idx]
        if idx != last {
            let movedPhys = freeList[last]
            freeList[idx] = movedPhys
            freeListPos[movedPhys] = idx
        }
        freeList.removeLast()
        freeListPos.removeValue(forKey: removed)
        return removed
    }

    // O(1) push onto freeList with position-tracking maintained.
    private func freeListPush(_ phys: Int) {
        freeListPos[phys] = freeList.count
        freeList.append(phys)
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

    // Try to find a shared pair already holding content with this hash.
    // nil if not cached. Does NOT increment refcount — caller probes,
    // then commits via `shareExisting` once they've decided to use it.
    func findByHash(_ hash: UInt64) -> SharedPagePair? {
        return contentIndex[hash]
    }

    // Grab a phys page for `sessionId`. The free list holds all
    // refcount==0 pages; some carry valid content hashes (cache entries
    // from prior sessions), some don't (genuinely uncached). We pick in
    // this order:
    //   1. The LRU-OLDEST uncached page (no hash to drop, free reuse).
    //   2. Otherwise: the LRU-OLDEST cached page → evict its hash, reuse.
    //
    // Critically we DO NOT drop a content hash unless step 1 found nothing
    // and we're forced to overwrite. The previous "drop on every pop"
    // logic shredded cache entries that the SAME request was about to
    // probe for: iter N's session pop-and-evicts iter N-1's pages off
    // the free list (because the entire free list was content-cached
    // immediately after iter N-1 closed), then iter N's later probes
    // for those same hashes miss. With LRU-no-eager-eviction, iter N
    // shareExisting-adopts iter N-1's pages first (refcount++, removes
    // them from free list), and only then allocFresh runs for the
    // genuine tail — by which point the free list contains uncached
    // pages.
    // 2026-05-07: anonymous-pool refactor — callers track their own
    // page references; PageManager only knows refcounts and content
    // hashes. Returns a phys page with refcount=1; caller MUST call
    // decref(physPage:) when done with it.
    func allocFresh() throws -> Int {
        guard !freeList.isEmpty else {
            throw PageManagerError.outOfPages(needed: 1, available: 0)
        }
        // Step 1: find the LRU-oldest uncached page on the free list.
        var bestUncached: (idx: Int, tick: UInt64)? = nil
        for (i, phys) in freeList.enumerated() {
            if pages[phys].contentHash == nil {
                let tick = pages[phys].lastAccessTick
                if bestUncached == nil || tick < bestUncached!.tick {
                    bestUncached = (i, tick)
                }
            }
        }
        let phys: Int
        if let pick = bestUncached {
            // Pop the LRU-oldest uncached page. No hash to drop.
            phys = freeListPop(at: pick.idx)
        } else {
            // Forced eviction: free list is all-cached. Pick LRU-oldest
            // cached entry and drop its hash.
            var oldest: (idx: Int, tick: UInt64)? = nil
            for (i, p) in freeList.enumerated() {
                let tick = pages[p].lastAccessTick
                if oldest == nil || tick < oldest!.tick {
                    oldest = (i, tick)
                }
            }
            let pick = oldest!  // freeList non-empty guaranteed above
            phys = freeListPop(at: pick.idx)
            // Drop the now-orphaned content hash + its pair.
            var p = pages[phys]
            if let h = p.contentHash {
                contentIndex.removeValue(forKey: h)
                if let partner = p.pairMate,
                   partner >= 0 && partner < pages.count,
                   pages[partner].contentHash == h {
                    pages[partner].contentHash = nil
                    pages[partner].pairMate = nil
                }
                p.contentHash = nil
                p.pairMate = nil
            }
            pages[phys] = p
        }
        clockTick += 1
        var p = pages[phys]
        p.refcount = 1
        p.lastAccessTick = clockTick
        pages[phys] = p
        return phys
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
            if let idx = freeListPos[physPage] {
                _ = freeListPop(at: idx)
            }
        }
        p.refcount += 1
        p.lastAccessTick = clockTick
        pages[physPage] = p
    }

    // After a fresh-allocated pair of pages has been written to (full
    // prefill CB committed for the slide-page's 16-token range + the
    // first 8 full-page tokens of that range + the second 8 full-page
    // tokens), promote the pair into the content index so a later
    // session submitting the same tokens can `findByHash` and adopt
    // both members as read-only shared pages.
    //
    // `slidePrimary` corresponds to the phys index at block_table[P]
    // (carries slide K/V for tokens [P*16, P*16+15] + full K/V for
    // tokens [P*16, P*16+7]). `fullSibling` corresponds to block_table
    // [P+1] (carries full K/V for tokens [P*16+8, P*16+15]).
    func promotePair(slidePrimary: Int, fullSibling: Int, contentHash: UInt64) {
        // Idempotence: if the same pair is already recorded under this
        // hash, nothing to do. If this hash is recorded against a
        // different pair, keep the older one (first-writer-wins).
        if contentIndex[contentHash] != nil { return }
        var p1 = pages[slidePrimary]
        var p2 = pages[fullSibling]
        p1.contentHash = contentHash
        p2.contentHash = contentHash
        p1.pairMate = fullSibling
        p2.pairMate = slidePrimary
        pages[slidePrimary] = p1
        pages[fullSibling]  = p2
        contentIndex[contentHash] = SharedPagePair(
            slidePrimary: slidePrimary, fullSibling: fullSibling)
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
        var p = pages[physPage]
        if p.refcount > 0 {
            p.refcount -= 1
            if p.refcount == 0 {
                freeListPush(physPage)
                // contentHash preserved for potential cache hit later.
            }
            pages[physPage] = p
        }
    }

    func pageRefcount(_ phys: Int) -> Int {
        guard phys >= 0 && phys < pages.count else { return 0 }
        return pages[phys].refcount
    }

    // Diagnostic snapshot.
    struct Stats {
        let totalPages: Int
        let freePages: Int
        let cachedHashes: Int
        // 2026-05-07: pages-in-use count replaces "active sessions".
        // Pages are anonymous; "in-use" means refcount > 0.
        let pagesInUse: Int
    }
    func stats() -> Stats {
        return Stats(totalPages: numPhysPages,
                     freePages: freeList.count,
                     cachedHashes: contentIndex.count,
                     pagesInUse: numPoolPages - freeList.count)
    }
}
