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

// SlidePairContents — pair of phys page IDs representing the K/V backing
// for a 16-token slide page. `slidePlusFullHead` corresponds to block_table[P]
// (slide cache for 16 tokens + full cache for first 8); `fullTail`
// corresponds to block_table[P+1] (full cache for the second 8 tokens
// at that 16-token slide-page range, plus whatever slide data gets
// written there by a later prefill tile).
//
// Followup 3 (2026-05-23): PageManager.contentIndex was deleted; the
// radix trie (radix_trie.swift) is the sole source of truth for
// adoption lookups. This struct survives as the value type stored at
// each trie anchor.
struct SlidePairContents {
    let slidePlusFullHead: Int
    let fullTail: Int
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
    // content-hash bookkeeping per phys page, plus the per-page
    // pairMate so the eviction callback can notify the trie about
    // both members of a pair.
    //
    // Eviction callback (Track D fold-in): invoked when allocFresh
    // forcibly evicts a cached page-pair. LmEngine wires this up to
    // RadixTrie.invalidateAnchorFor so stale anchors get unlinked.
    // Receives (physPage, oldHash).
    var onPageEvicted: ((Int, UInt64) -> Void)?
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
        self.freeUncached = Array((basePage..<(basePage + poolCount)).reversed())
        self.freeUncachedPos.reserveCapacity(self.freeUncached.count)
        self.freeCachedPos.reserveCapacity(poolCount)
        for (i, phys) in self.freeUncached.enumerated() {
            self.freeUncachedPos[phys] = i
        }
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
        guard !freeUncached.isEmpty || !freeCached.isEmpty else {
            throw PageManagerError.outOfPages(needed: 1, available: 0)
        }
        let phys: Int
        if !freeUncached.isEmpty {
            // Prefer uncached pages. No prefix-cache entry is dropped.
            phys = freeUncachedPop(at: freeUncached.count - 1)
        } else {
            // Forced eviction: every free page is cached. Drop the LRU
            // (lowest lastAccessTick) cached page — NOT the stack top.
            // The prior code popped most-recently-freed (MRU), evicting the
            // hottest cache entries first and defeating the prefix cache
            // under pressure (page_lifecycle_audit_2026-05-28 #7). O(n) scan
            // is fine: forced eviction only fires when freeUncached is empty.
            var lruIdx = 0
            var lruTick = UInt64.max
            for (i, ph) in freeCached.enumerated() {
                let t = pages[ph].lastAccessTick
                if t < lruTick { lruTick = t; lruIdx = i }
            }
            phys = freeCachedPop(at: lruIdx)
            // Drop the now-orphaned content hash + its pair.
            // Followup 3 (2026-05-23): contentIndex deleted; only the
            // PageInfo bookkeeping + eviction callback into the trie
            // remain. The trie is responsible for unlinking the anchor.
            var p = pages[phys]
            if let h = p.contentHash {
                // Track D: notify the radix trie BEFORE we mutate
                // PageInfo so the callback can read accurate state if
                // it wishes.
                onPageEvicted?(phys, h)
                if let partner = p.pairMate {
                    onPageEvicted?(partner, h)
                }
                if let partner = p.pairMate,
                   partner >= 0 && partner < pages.count,
                   pages[partner].contentHash == h {
                    pages[partner].contentHash = nil
                    pages[partner].pairMate = nil
                    moveFreePageToUncachedIfPresent(partner)
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
            removeFromFreeStacksIfPresent(physPage)
        }
        p.refcount += 1
        p.lastAccessTick = clockTick
        pages[physPage] = p
    }

    // Followup 3 (2026-05-23): the old promotePair(...) populated both
    // PageInfo bookkeeping (contentHash + pairMate) AND inserted into
    // contentIndex. With contentIndex deleted, this method survives as
    // a small PageInfo-only writer: it stamps each phys page with the
    // content hash + partner phys so the eviction callback in
    // allocFresh can notify the trie about the pair as a unit.
    //
    // `slidePlusFullHead` is block_table[P] (slide K/V for tokens
    // [P*16..P*16+15] + full K/V for [P*16..P*16+7]); `fullTail`
    // is block_table[P+1] (full K/V for [P*16+8..P*16+15]).
    //
    // Idempotent: re-stamping with the same hash is a no-op modulo
    // lastAccessTick refresh; re-stamping with a DIFFERENT hash
    // overwrites (the trie's insertAnchor is similarly first-writer-
    // wins via its internal walk, so callers should follow trie
    // semantics).
    func markPairContentIndexed(slidePlusFullHead: Int, fullTail: Int,
                                  contentHash: UInt64) {
        var p1 = pages[slidePlusFullHead]
        var p2 = pages[fullTail]
        p1.contentHash = contentHash
        p2.contentHash = contentHash
        p1.pairMate = fullTail
        p2.pairMate = slidePlusFullHead
        pages[slidePlusFullHead] = p1
        pages[fullTail]  = p2
        if p1.refcount == 0 { moveFreePageToCachedIfPresent(slidePlusFullHead) }
        if p2.refcount == 0 { moveFreePageToCachedIfPresent(fullTail) }
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
        guard physPage >= 0 && physPage < pages.count else {
            if !PageManager.warnedBadDecref {
                PageManager.warnedBadDecref = true
                FileHandle.standardError.write(Data("[pagemgr] WARN: decref out-of-range phys=\(physPage) (pool=\(pages.count)) — balance bug; see page_lifecycle_audit #2\n".utf8))
            }
            return
        }
        var p = pages[physPage]
        if p.refcount > 0 {
            p.refcount -= 1
            if p.refcount == 0 {
                pages[physPage] = p
                freePush(physPage)
                // contentHash preserved for potential cache hit later.
                return
            }
            pages[physPage] = p
        } else {
            // decref of an already-free page = a balance bug (double-free /
            // duplicate ownedPages entry). The old code silently swallowed
            // this, masking every page-leak/double-free in the engine. Log
            // ONCE (not a hard precondition — a latent double-free should be
            // surfaced, not crash production). page_lifecycle_audit #2.
            if !PageManager.warnedDoubleFree {
                PageManager.warnedDoubleFree = true
                FileHandle.standardError.write(Data("[pagemgr] WARN: decref of free page \(physPage) (refcount already 0) — double-free / balance bug; see page_lifecycle_audit #2\n".utf8))
            }
        }
    }
    static var warnedDoubleFree = false
    static var warnedBadDecref = false

    func pageRefcount(_ phys: Int) -> Int {
        guard phys >= 0 && phys < pages.count else { return 0 }
        return pages[phys].refcount
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
    }
    func stats() -> Stats {
        var cached = 0
        for p in pages { if p.contentHash != nil { cached += 1 } }
        return Stats(totalPages: numPhysPages,
                     freePages: freeUncached.count + freeCached.count,
                     cachedHashes: cached,
                     pagesInUse: numPoolPages - freeUncached.count - freeCached.count)
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
        let pairMate: Int?             // companion phys page (slide + full sibling)
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
                                   contentHash: p.contentHash,
                                   pairMate: p.pairMate))
            if out.count >= cap { break }
        }
        return out
    }
}
