// kv_ssd_store.swift — Tier 1 cold-KV SSD store (2026-06).
//
// Builds on Tier 0 (pin-on-grow KV residency, committed). Tier 1 EXTENDS
// cold-cache capacity: a refcount-0 CACHED page that would otherwise be
// DROPPED (its trie anchor unlinked, K/V bytes abandoned) under forced
// eviction is instead DEMOTED to an NVMe-backed file, so a later session
// re-adopting the same content-addressed prefix can RELOAD it (pread +
// scatter into a fresh RAM page) instead of re-running prefill.
//
// Bandwidth justification (docs/kv_memory_pin_hot_tier_cold_2026-06.md §4):
//   reload = 220 KiB/token / ~6 GB/s = ~37.5 us/tok (~26,700 tok/s),
//   vs re-prefill ~1-1.7 ms/tok (600-1000 tok/s) -> 27-45x faster.
//
// DESIGN (per docs/kv_memory_pin_hot_tier_cold_2026-06.md and the approved
// Tier-1 design block):
//   - A single growable file, one fixed-size slot = perPageBytes (the SAME
//     Sigma_layers PAGE*KV_H*HD*2*2 the engine computes; passed in at init,
//     never re-derived here so it cannot drift from the kernel layout).
//   - Positioned I/O: pwrite/pread at slot*perPageBytes — no seek races.
//   - Slot ownership lives in the trie's PageRef.ssd(slot); this store is a
//     DUMB slab allocator (free-slot stack + high-water). No content
//     addressing, no anchor references inside the store.
//   - Volatile cache: the file is process-life only (truncated to 0 at
//     boot). On crash it is discarded — content-addressed re-prefill
//     rebuilds it. fsync is NOT required (KV pages are a content-addressed
//     cache, never trusted across process restarts).
//   - KV_SSD_TIER_GB env (Double, default 0 = DISABLED). When disabled the
//     engine never instantiates this store and never sets
//     onPageDemoteCandidate -> the drop path is bit-identical to today.
//
// CONCURRENCY: engine-owned single instance, guarded by gEngineLock like all
// PageManager ops. demote/reload both run under the engine CB serialization
// (same as growPool/allocFresh), so no internal locking is needed.

import Foundation

// Tier 1 capacity config. KV_SSD_TIER_GB (Double): 0 / unset = DISABLED.
// Symmetric with kvMemBudgetFrac() / kvPoolPagesOverride() in bootstrap.swift.
func kvSsdTierGB() -> Double {
    if let s = ProcessInfo.processInfo.environment["KV_SSD_TIER_GB"],
       let v = Double(s), v > 0 {
        return v
    }
    return 0.0
}

final class KvSsdStore {
    // Bytes per logical page = the engine's perPageBytes (Sigma over 30
    // layers of PAGE*KV_H*HD*2 for K and again for V). Passed in; NEVER
    // re-derived here so the slot size cannot drift from kvSliceLayout.
    let perPageBytes: Int
    // Hard cap on slots = floor(KV_SSD_TIER_GB * 1024^3 / perPageBytes).
    let maxSlots: Int
    let path: String

    private let fd: Int32
    // Free-slot reuse stack + high-water of never-used slots.
    private var freeSlots: [Int] = []
    private var highWater: Int = 0
    // Tier-1 in-tier LRU (2026-06): per-slot recency for coldestInUseSlot().
    // A monotonic tick (matches PageManager.clockTick's style; no wall clock)
    // is bumped on the two byte-I/O sites (write=demote, read=reload). slotTick
    // is a SPARSE dict (not a [UInt64] array) because freeSlot()/allocSlot()
    // reuse arbitrary indices: a reused slot gets a fresh stamp on its next
    // write, and freeSlot drops the stale stamp so a freed slot is never a
    // coldest candidate.
    private var globalTick: UInt64 = 0
    private var slotTick: [Int: UInt64] = [:]
    // Telemetry.
    private(set) var demoteCount: Int = 0
    private(set) var demoteBytes: UInt64 = 0
    private(set) var reloadCount: Int = 0
    private(set) var reloadBytes: UInt64 = 0

    // Returns nil if Tier 1 is disabled (gb<=0) OR the file cannot be opened.
    // The caller (engine init) treats nil as "Tier 1 off" and leaves the drop
    // path unchanged.
    static func makeIfEnabled(perPageBytes: Int) -> KvSsdStore? {
        let gb = kvSsdTierGB()
        guard gb > 0, perPageBytes > 0 else { return nil }
        let bytesCap = gb * 1024.0 * 1024.0 * 1024.0
        let maxSlots = Int(bytesCap / Double(perPageBytes))
        guard maxSlots > 0 else { return nil }
        // output_data is the canonical artifact dir (memory: outputs_never_to_tmp).
        let dir = "output_data/kv_ssd_tier"
        do {
            try FileManager.default.createDirectory(atPath: dir,
                withIntermediateDirectories: true, attributes: nil)
        } catch {
            FileHandle.standardError.write(Data(
                "[kv-ssd] FAILED to create \(dir): \(error) -> Tier 1 disabled\n".utf8))
            return nil
        }
        let path = "\(dir)/kv_tier1.bin"
        // O_TRUNC: the file is process-life only; never trust it across a
        // process restart (volatile content-addressed cache). 0600.
        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o600)
        guard fd >= 0 else {
            FileHandle.standardError.write(Data(
                "[kv-ssd] FAILED to open \(path): errno=\(errno) -> Tier 1 disabled\n".utf8))
            return nil
        }
        let store = KvSsdStore(fd: fd, path: path,
                               perPageBytes: perPageBytes, maxSlots: maxSlots)
        FileHandle.standardError.write(Data(String(format:
            "[kv-ssd] Tier 1 ENABLED: %.2f GB cap, perPage=%.2f MB -> %d slots (%@)\n",
            gb, Double(perPageBytes) / (1024*1024), maxSlots, path).utf8))
        return store
    }

    private init(fd: Int32, path: String, perPageBytes: Int, maxSlots: Int) {
        self.fd = fd
        self.path = path
        self.perPageBytes = perPageBytes
        self.maxSlots = maxSlots
        self.freeSlots.reserveCapacity(min(maxSlots, 4096))
    }

    deinit {
        if fd >= 0 { close(fd) }
        // Best-effort cleanup; the file is gitignored under output_data.
        unlink(path)
    }

    // Allocate a slot: reuse a freed slot if any, else advance the high-water
    // (growing the file lazily by ftruncate). Returns nil when the store is at
    // its KV_SSD_TIER_GB cap (FULL) — the caller then DECLINES the demote and
    // the existing DROP path runs.
    func allocSlot() -> Int? {
        if let s = freeSlots.popLast() { return s }
        guard highWater < maxSlots else { return nil }
        let s = highWater
        highWater += 1
        // Grow the backing file to cover the new slot (sparse until written).
        let wantLen = off_t(highWater) * off_t(perPageBytes)
        if ftruncate(fd, wantLen) != 0 {
            // Growth failed (disk full): roll back the high-water and decline.
            highWater -= 1
            FileHandle.standardError.write(Data(
                "[kv-ssd] ftruncate to \(wantLen) failed errno=\(errno); declining demote\n".utf8))
            return nil
        }
        return s
    }

    // Return a slot to the free stack (after a reload reclaims it OR an
    // in-tier LRU eviction orphans it). Drop the recency stamp so a freed
    // (not-yet-reused) slot is never picked by coldestInUseSlot.
    func freeSlot(_ slot: Int) {
        freeSlots.append(slot)
        slotTick[slot] = nil
    }

    // In-tier LRU victim selection (2026-06). The in-use set is the
    // [0, highWater) frontier MINUS freeSlots (exactly Stats.usedSlots). Return
    // the lowest-lastTick in-use slot. Returns nil only when there are no
    // in-use slots (cannot happen on the demote-evict-retry path, where
    // allocSlot just returned nil => highWater==maxSlots with every slot in
    // use, so coldest is non-nil). O(highWater) scan, fired only at cap.
    func coldestInUseSlot(excluding: Set<Int> = []) -> Int? {
        if highWater == 0 { return nil }
        let free = Set(freeSlots)   // small; O(1) membership
        var coldest: Int? = nil
        var coldestTick = UInt64.max
        for s in 0..<highWater where !free.contains(s) && !excluding.contains(s) {
            // A written slot always has a stamp; default 0 ranks an
            // (in-use but somehow unstamped) slot as coldest, which is safe.
            let tk = slotTick[s] ?? 0
            if tk < coldestTick { coldestTick = tk; coldest = s }
        }
        return coldest
    }

    // Write a gathered page (exactly perPageBytes) to slot. Caller has already
    // gathered the 60 K/V slices contiguously via the engine's kvSliceLayout.
    // Returns true on full write. positioned pwrite — no internal seek.
    func write(slot: Int, bytes: UnsafeRawPointer) -> Bool {
        let off = off_t(slot) * off_t(perPageBytes)
        var written = 0
        while written < perPageBytes {
            let n = pwrite(fd, bytes.advanced(by: written),
                           perPageBytes - written, off + off_t(written))
            if n <= 0 {
                FileHandle.standardError.write(Data(
                    "[kv-ssd] pwrite slot=\(slot) failed at \(written)/\(perPageBytes) errno=\(errno)\n".utf8))
                return false
            }
            written += n
        }
        globalTick += 1; slotTick[slot] = globalTick   // recency: demote
        demoteCount += 1
        demoteBytes += UInt64(perPageBytes)
        return true
    }

    // Read a slot's perPageBytes into dst. Caller scatters them via the same
    // kvSliceLayout. positioned pread — no internal seek.
    func read(slot: Int, into dst: UnsafeMutableRawPointer) -> Bool {
        let off = off_t(slot) * off_t(perPageBytes)
        var got = 0
        while got < perPageBytes {
            let n = pread(fd, dst.advanced(by: got),
                          perPageBytes - got, off + off_t(got))
            if n <= 0 {
                FileHandle.standardError.write(Data(
                    "[kv-ssd] pread slot=\(slot) failed at \(got)/\(perPageBytes) errno=\(errno)\n".utf8))
                return false
            }
            got += n
        }
        globalTick += 1; slotTick[slot] = globalTick   // recency: reload
        reloadCount += 1
        reloadBytes += UInt64(perPageBytes)
        return true
    }

    struct Stats {
        let usedSlots: Int
        let maxSlots: Int
        let demoteCount: Int
        let demoteBytes: UInt64
        let reloadCount: Int
        let reloadBytes: UInt64
    }
    func stats() -> Stats {
        return Stats(usedSlots: highWater - freeSlots.count,
                     maxSlots: maxSlots,
                     demoteCount: demoteCount, demoteBytes: demoteBytes,
                     reloadCount: reloadCount, reloadBytes: reloadBytes)
    }

    // Test-only factory: a tiny in-tmp store with `slots` capacity, used by
    // runKvSsdStoreRecencyTests. perPageBytes is a token small size (16) so the
    // pwrite/pread exercise the recency stamps without large allocations. The
    // backing file is created O_TRUNC under output_data and unlinked on deinit.
    static func makeForTest(perPageBytes: Int, maxSlots: Int) -> KvSsdStore? {
        let dir = "output_data/kv_ssd_tier"
        try? FileManager.default.createDirectory(atPath: dir,
            withIntermediateDirectories: true, attributes: nil)
        let path = "\(dir)/kv_tier1_test.bin"
        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o600)
        guard fd >= 0 else { return nil }
        return KvSsdStore(fd: fd, path: path,
                          perPageBytes: perPageBytes, maxSlots: maxSlots)
    }
}

// ============================================================================
// KvSsdStore in-tier-LRU recency tests — gated on LM_TEST_RADIX_TRIE (run
// alongside runRadixTrieTests from runtime.swift). Validates per-slot recency
// ordering + coldestInUseSlot victim selection.
// ============================================================================
func runKvSsdStoreRecencyTests() {
    print("\n=== KvSsdStore recency tests ===")
    var passed = 0, failed = 0
    func check(_ name: String, _ cond: Bool) {
        if cond { passed += 1; print("  ok  \(name)") }
        else    { failed += 1; print("  FAIL \(name)") }
    }
    let per = 16
    guard let store = KvSsdStore.makeForTest(perPageBytes: per, maxSlots: 4) else {
        print("  FAIL could not create test store"); exit(1)
    }
    let buf = UnsafeMutableRawPointer.allocate(byteCount: per, alignment: 16)
    defer { buf.deallocate() }
    memset(buf, 0xAB, per)

    // Fill all 4 slots with writes in order 0,1,2,3 (ticks ascending).
    var slots: [Int] = []
    for _ in 0..<4 {
        guard let s = store.allocSlot() else { check("alloc within cap", false); break }
        _ = store.write(slot: s, bytes: buf)
        slots.append(s)
    }
    check("recency: filled 4 slots", slots.count == 4)
    // Store now at cap: allocSlot returns nil.
    check("recency: store at cap (allocSlot nil)", store.allocSlot() == nil)
    // Coldest = the first-written slot (lowest tick) = slots[0].
    check("recency: coldest is the first-written slot",
          store.coldestInUseSlot() == slots[0])
    // Touch slots[0] via a read -> it becomes the freshest; coldest moves to slots[1].
    _ = store.read(slot: slots[0], into: buf)
    check("recency: read bumps recency (coldest now second-written)",
          store.coldestInUseSlot() == slots[1])
    // Re-write slots[1] (demote bump) -> coldest moves to slots[2].
    _ = store.write(slot: slots[1], bytes: buf)
    check("recency: write bumps recency (coldest now third-written)",
          store.coldestInUseSlot() == slots[2])
    // Free the current coldest; coldestInUseSlot must skip it (stamp dropped).
    let evicted = store.coldestInUseSlot()!
    store.freeSlot(evicted)
    check("recency: freed slot is not a coldest candidate",
          store.coldestInUseSlot() != evicted)
    // excluding: must skip an excluded slot even if it is coldest.
    let cur = store.coldestInUseSlot()!
    check("recency: excluding skips the named slot",
          store.coldestInUseSlot(excluding: [cur]) != cur)

    print("KvSsdStore recency tests: \(passed) passed, \(failed) failed")
    if failed > 0 { exit(1) }
}
