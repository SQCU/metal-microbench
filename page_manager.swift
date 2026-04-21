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
    case notOwned(physPage: Int, sessionId: Int)
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
    var owners: Set<Int>        // session IDs currently holding this page
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
    let numPhysPages: Int
    let pageSize: Int

    // Physical page state: pages[p] describes phys page p.
    private var pages: [PageInfo]
    // Free list of phys pages with refcount==0. LIFO (cache-warm reuse).
    private var freeList: [Int] = []
    // contentIndex[contentHash] = pair of phys pages carrying the
    // 16-token slide-page's data in both cache layouts. See the
    // PageInfo header for why sharing has to be a pair.
    private var contentIndex: [UInt64: SharedPagePair] = [:]
    // Per-session ordered list of phys pages owned (for release + block-table build).
    private var sessionPages: [Int: [Int]] = [:]
    private var clockTick: UInt64 = 0

    init(numPhysPages: Int, pageSize: Int) {
        self.numPhysPages = numPhysPages
        self.pageSize = pageSize
        self.pages = (0..<numPhysPages).map { _ in
            PageInfo(owners: [], contentHash: nil, pairMate: nil, lastAccessTick: 0)
        }
        // Initially every page is free. Push in reverse so page 0 pops first.
        self.freeList = Array((0..<numPhysPages).reversed())
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

    // Grab a fresh phys page for `sessionId`. Fails with `outOfPages` if
    // the free list is empty. The page is single-owner and its content
    // hash is invalidated at alloc time (we're about to overwrite the KV).
    //
    // Released pages keep their content hash until they're re-allocated
    // here — that's how we let a later session probe findByHash and
    // adopt an already-computed prefix without prefilling it. Only at
    // the moment a fresh allocation *commits to overwriting* do we drop
    // the cached hash.
    func allocFresh(sessionId: Int) throws -> Int {
        guard let phys = freeList.popLast() else {
            throw PageManagerError.outOfPages(needed: 1, available: 0)
        }
        clockTick += 1
        var p = pages[phys]
        // Invalidate stale content hash — the KV is about to be rewritten.
        // Pair invariant: if this page was part of a promoted pair, the
        // partner's K/V alone is no longer useful (a content-hash hit
        // must produce BOTH phys pages to correctly cover slide + full
        // caches). Drop the index entry AND clear the partner's hash so
        // a later session can't findByHash and get an incomplete pair.
        if let h = p.contentHash {
            if contentIndex[h] != nil {
                contentIndex.removeValue(forKey: h)
            }
            if let partner = p.pairMate,
               partner >= 0 && partner < pages.count,
               pages[partner].contentHash == h {
                pages[partner].contentHash = nil
                pages[partner].pairMate = nil
            }
            p.contentHash = nil
            p.pairMate = nil
        }
        p.owners = [sessionId]
        p.lastAccessTick = clockTick
        pages[phys] = p
        sessionPages[sessionId, default: []].append(phys)
        return phys
    }

    // Adopt an existing cached page for `sessionId`. If the page was
    // released-but-not-yet-overwritten (owners.isEmpty, still on free
    // list), resurrect it: remove from the free list, set sessionId as
    // the sole owner. Otherwise just add sessionId to the owner set.
    // Either way, content hash is preserved (read-only sharing).
    func shareExisting(physPage: Int, sessionId: Int) {
        clockTick += 1
        var p = pages[physPage]
        if p.owners.isEmpty {
            // Page is in free list but still has valid content. Pull it
            // back out before handing out to this session.
            if let idx = freeList.firstIndex(of: physPage) {
                freeList.remove(at: idx)
            }
        }
        p.owners.insert(sessionId)
        p.lastAccessTick = clockTick
        pages[physPage] = p
        sessionPages[sessionId, default: []].append(physPage)
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

    // Release this session's claim on the page. If refcount drops to zero
    // the page goes on the free list BUT its content-index entry stays —
    // the KV data is still valid until something actually overwrites it
    // (which only happens at allocFresh time). This enables cache hits
    // across session lifetimes: close session A, open session B with the
    // same prefix, B's probe finds A's cached pages and resurrects them
    // via shareExisting without any prefill work.
    func releasePage(physPage: Int, sessionId: Int) throws {
        var p = pages[physPage]
        guard p.owners.contains(sessionId) else {
            throw PageManagerError.notOwned(physPage: physPage, sessionId: sessionId)
        }
        p.owners.remove(sessionId)
        if p.owners.isEmpty {
            freeList.append(physPage)
            // Do NOT drop contentHash here — leave it for potential reuse.
        }
        pages[physPage] = p
    }

    // Release all of a session's pages in one call — used by closeSession.
    func releaseAllForSession(_ sessionId: Int) {
        guard let owned = sessionPages.removeValue(forKey: sessionId) else { return }
        for phys in owned {
            // Ignore failures — we're tearing down the session anyway.
            try? releasePage(physPage: phys, sessionId: sessionId)
        }
    }

    // Peek at a session's owned pages (for block-table assembly).
    func pagesForSession(_ sessionId: Int) -> [Int] {
        return sessionPages[sessionId] ?? []
    }

    // Owners of a physical page. Used by the KV-snapshot FFI to surface
    // which sessions are citing a given page (refcount > 1 ⇒ shared).
    func ownersOfPage(_ phys: Int) -> [Int] {
        guard phys >= 0 && phys < pages.count else { return [] }
        return Array(pages[phys].owners).sorted()
    }

    func pageRefcount(_ phys: Int) -> Int {
        guard phys >= 0 && phys < pages.count else { return 0 }
        return pages[phys].owners.count
    }

    // Diagnostic snapshot.
    struct Stats {
        let totalPages: Int
        let freePages: Int
        let cachedHashes: Int
        let activeSessions: Int
    }
    func stats() -> Stats {
        return Stats(totalPages: numPhysPages,
                     freePages: freeList.count,
                     cachedHashes: contentIndex.count,
                     activeSessions: sessionPages.count)
    }
}
