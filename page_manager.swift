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
private struct PageInfo {
    var owners: Set<Int>        // session IDs currently holding this page
    var contentHash: UInt64?    // nil = page isn't content-indexed (fresh/dirty)
    var lastAccessTick: UInt64
}

final class PageManager {
    let numPhysPages: Int
    let pageSize: Int

    // Physical page state: pages[p] describes phys page p.
    private var pages: [PageInfo]
    // Free list of phys pages with refcount==0. LIFO (cache-warm reuse).
    private var freeList: [Int] = []
    // contentIndex[contentHash] = phys page holding that content. Used
    // by prefix-cache lookups during session submit.
    private var contentIndex: [UInt64: Int] = [:]
    // Per-session ordered list of phys pages owned (for release + block-table build).
    private var sessionPages: [Int: [Int]] = [:]
    private var clockTick: UInt64 = 0

    init(numPhysPages: Int, pageSize: Int) {
        self.numPhysPages = numPhysPages
        self.pageSize = pageSize
        self.pages = (0..<numPhysPages).map { _ in
            PageInfo(owners: [], contentHash: nil, lastAccessTick: 0)
        }
        // Initially every page is free. Push in reverse so page 0 pops first.
        self.freeList = Array((0..<numPhysPages).reversed())
    }

    // FNV-1a over a page's token IDs. Used both by the manager (on
    // `admitShared`) and by callers that want to probe before committing.
    static func hashPage(_ tokens: ArraySlice<UInt32>) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for t in tokens {
            h ^= UInt64(t)
            h = h &* 0x100000001b3
        }
        return h
    }

    // Try to find a phys page already holding content with this hash. nil
    // if not cached. Does NOT increment refcount — caller probes, then
    // commits via `shareExisting` once they've decided to use it.
    func findByHash(_ hash: UInt64) -> Int? {
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
        if let h = p.contentHash {
            if contentIndex[h] == phys { contentIndex.removeValue(forKey: h) }
            p.contentHash = nil
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

    // After a fresh-allocated page has been written to (KV cache populated
    // by a prefill CB), promote it into the content index so a later
    // session submitting the same tokens can `findByHash` it and share.
    func promoteToShared(physPage: Int, contentHash: UInt64) {
        var p = pages[physPage]
        // A page can only be in contentIndex once; if it was already
        // promoted, keep it (same tokens → same hash). If the content
        // index already points to a different page for this hash, prefer
        // the older one and leave this page unshared (caller can choose
        // to free the redundant page).
        if p.contentHash == contentHash { return }
        p.contentHash = contentHash
        pages[physPage] = p
        // Only index if no prior page claimed this hash.
        if contentIndex[contentHash] == nil {
            contentIndex[contentHash] = physPage
        }
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
