// ffi.swift — C-ABI exports for the Python bridge.
//
// The engine stays in Swift (Metal driver domain). This file is the only
// thing the Python FastAPI bridge links against. Everything here is
// thread-safe via a single recursive lock: the bridge runs a pump thread
// that loops gemma_tick() while there's work, and async request handlers
// call open/submit/append/poll/close concurrently. Serializing on the lock
// mirrors LmEngine's single-threaded-from-Swift-POV invariant.
//
// Conventions:
//   - All pointer args may be nil; nil pointers = "query the size instead".
//   - Returns: ≥0 on success (usually counts), -1 on error.
//   - Session handles are Int32 (1-indexed, stable for the process lifetime).

import Foundation
import Metal
import CryptoKit

private let ffiLock = NSRecursiveLock()
private var gEngine: LmEngine?
private var gVisionResidency: VisionResidency?
private var gPressureSource: DispatchSourceMemoryPressure?
private var gSessions: [Int32: Session] = [:]
private var gNextHandle: Int32 = 1

// Vision soft-tokens cache — keyed by SHA-256 of the raw PNG/JPEG bytes.
// On a cache hit we skip preprocessing + vision tower entirely (saves ~7 s
// per repeated image on M5) and just reuse the already-padded MTLBuffer.
// The same MTLBuffer can back N concurrent sessions: Session.submit(softTokens:)
// stores it in a .softTokens chunk which is read-only from the kernel side.
// Cache entry. `pendingCB` is the in-flight vision CB that produced this
// buffer; it gets cleared after the first .waitUntilCompleted(). Subsequent
// users of the same cache entry see pendingCB == nil and skip the wait.
// Under the async pipeline, LM decode can already be running by the time
// this wait happens — the CB has usually already completed on the GPU.
private class CachedSofts {
    let buffer: MTLBuffer       // padded to targetSoft rows, fp32
    let count: Int              // always targetSoft=280 currently
    var lastUsed: UInt64        // monotonic tick counter for LRU
    let bytes: Int              // buffer.length, for stats
    var pendingCB: MTLCommandBuffer?   // non-nil until the vision CB is waited on
    init(buffer: MTLBuffer, count: Int, lastUsed: UInt64, bytes: Int, pendingCB: MTLCommandBuffer?) {
        self.buffer = buffer; self.count = count; self.lastUsed = lastUsed
        self.bytes = bytes; self.pendingCB = pendingCB
    }
}
private var gVisionCache: [Data: CachedSofts] = [:]
private var gVisionCacheHits: UInt64 = 0
private var gVisionCacheMisses: UInt64 = 0
private var gVisionCacheTick: UInt64 = 0
private let gVisionCacheMaxEntries = 64    // ~200 MB at 280 × 2816 × 4 B each

// Dedicated command queue for vision tower work. Runs concurrently with
// the main LM queue on M5 Max — two queues can share ALU partitions,
// enabling vision(image N+1) ∥ LM.decode(label N) pipelining.
var gVisionQueue: MTLCommandQueue?

// --- Initialization ---

@_cdecl("gemma_init")
public func gemma_init(_ ggufPath: UnsafePointer<CChar>?) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let gp = ggufPath else { return -1 }
    let pathStr = String(cString: gp)
    do {
        // One-time side effects that used to be top-level statements in
        // bootstrap.swift (banners + active_exp fill). Must run before any
        // GPU kernel that reads active_exp — i.e. before the first tick().
        bootstrapGlobalState()
        let w = try loadLmWeights(ggufPath: pathStr)
        // K/V caches are the thing we MUST NOT evict — losing a layer's K/V
        // mid-conversation kills in-flight generations. Explicitly pin them
        // as .nonVolatile (buffers default non-volatile, but explicit makes
        // the policy legible and survives any ambient purgeability changes).
        for kc in w.K_caches { _ = kc.setPurgeableState(.nonVolatile) }
        for vc in w.V_caches { _ = vc.setPurgeableState(.nonVolatile) }
        gEngine = LmEngine(weights: w)
        // Subscribe to macOS memory-pressure events. .warn ⇒ just ask for
        // vision soft-cache flush; .critical ⇒ drop vision working weights
        // + image-softs cache entirely. Session KV stays pinned throughout.
        subscribePressureSource()
        return 0
    } catch {
        print("gemma_init failed: \(error)")
        return -1
    }
}

private func subscribePressureSource() {
    // On macOS, DispatchSource memory-pressure events fire asynchronously
    // from the kernel. We handle them on a dedicated serial queue so the
    // handler doesn't contend with in-flight FFI calls on the main thread.
    let q = DispatchQueue(label: "gemma.pressure", qos: .utility)
    let src = DispatchSource.makeMemoryPressureSource(
        eventMask: [.warning, .critical], queue: q)
    src.setEventHandler {
        let evt = src.data
        ffiLock.lock(); defer { ffiLock.unlock() }
        if evt.contains(.critical) {
            print("[gemma] memory pressure CRITICAL — dropping vision working set + soft cache")
            gVisionResidency?.forceDrop()
            let evicted = gVisionCache.count
            gVisionCache.removeAll()
            if evicted > 0 {
                print("[gemma] evicted \(evicted) soft-tokens cache entries")
            }
        } else if evt.contains(.warning) {
            print("[gemma] memory pressure WARN — marking vision volatile")
            gVisionResidency?.allowEvict()
        }
    }
    src.resume()
    gPressureSource = src
}

@_cdecl("gemma_is_ready")
public func gemma_is_ready() -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return gEngine != nil ? 1 : 0
}

// --- Session lifecycle ---

@_cdecl("gemma_open_session")
public func gemma_open_session(_ maxNewTokens: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let engine = gEngine else { return -1 }
    guard let s = engine.openSession(maxNewTokens: Int(maxNewTokens)) else { return -1 }
    let h = gNextHandle
    gNextHandle += 1
    gSessions[h] = s
    return h
}

@_cdecl("gemma_close_session")
public func gemma_close_session(_ sid: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let engine = gEngine, let s = gSessions[sid] else { return -1 }
    engine.closeSession(s)
    gSessions.removeValue(forKey: sid)
    return 0
}

@_cdecl("gemma_pause_session")
public func gemma_pause_session(_ sid: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    s.pause()
    return 0
}

// --- Submit / append tokens ---

@_cdecl("gemma_submit")
public func gemma_submit(_ sid: Int32, _ tokens: UnsafePointer<UInt32>?, _ n: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid], let t = tokens, n > 0 else { return -1 }
    let arr = Array(UnsafeBufferPointer(start: t, count: Int(n)))
    s.submit(arr)
    return 0
}

@_cdecl("gemma_append")
public func gemma_append(_ sid: Int32, _ tokens: UnsafePointer<UInt32>?, _ n: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid], let t = tokens, n > 0 else { return -1 }
    let arr = Array(UnsafeBufferPointer(start: t, count: Int(n)))
    s.append(arr)
    return 0
}

// --- Scheduler + output drain ---

@_cdecl("gemma_tick")
public func gemma_tick() -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let engine = gEngine else { return 0 }
    return Int32(engine.tick())
}

@_cdecl("gemma_has_work")
public func gemma_has_work() -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return (gEngine?.hasWork ?? false) ? 1 : 0
}

@_cdecl("gemma_poll")
public func gemma_poll(_ sid: Int32,
                       _ outBuf: UnsafeMutablePointer<UInt32>?,
                       _ maxTokens: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid], let buf = outBuf else { return -1 }
    var count: Int32 = 0
    while count < maxTokens, let tok = s.nextToken() {
        buf[Int(count)] = tok
        count += 1
    }
    return count
}

// State enum mirrored on the Python side:
//   0 = idle, 1 = priming, 2 = generating, 3 = paused, 4 = done
@_cdecl("gemma_session_state")
public func gemma_session_state(_ sid: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    switch s.state {
    case .idle:       return 0
    case .priming:    return 1
    case .generating: return 2
    case .paused:     return 3
    case .done:       return 4
    }
}

// Current session position (tokens consumed by K/V cache so far).
// Used by /v1/perplexity to poll for prefill/AR completion: after
// submit(N tokens), position advances by N once the prefill/AR tick
// has run. Polling position>expected tells us when logits are ready
// to read.
@_cdecl("gemma_session_position")
public func gemma_session_position(_ sid: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    return Int32(s.positionForDebug)
}

// --- Tokenizer (so Python doesn't have to bundle a second one) ---

// Tokenize text_len UTF-8 bytes. If outTokens is nil, returns the number of
// tokens that would be produced (query-size pattern). Otherwise fills up to
// maxTokens and returns the actual count written (clamped to maxTokens).
@_cdecl("gemma_tokenize")
public func gemma_tokenize(_ text: UnsafePointer<CChar>?, _ textLen: Int32,
                           _ addBos: Int32,
                           _ outTokens: UnsafeMutablePointer<UInt32>?,
                           _ maxTokens: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let engine = gEngine, let t = text, textLen >= 0 else { return -1 }
    let raw = UnsafeBufferPointer(start: t, count: Int(textLen))
    let bytes = raw.map { UInt8(bitPattern: $0) }
    let str = String(decoding: bytes, as: UTF8.self)
    let toks = engine.tokenize(str, addBos: addBos != 0)
    if outTokens == nil { return Int32(toks.count) }
    let n = min(toks.count, Int(maxTokens))
    for i in 0..<n { outTokens![i] = toks[i] }
    return Int32(n)
}

// Detokenize n tokens into UTF-8 bytes. If outBuf is nil, returns bytes needed.
// Result is NOT null-terminated — caller gets the exact byte count written.
@_cdecl("gemma_detokenize")
public func gemma_detokenize(_ tokens: UnsafePointer<UInt32>?, _ n: Int32,
                             _ outBuf: UnsafeMutablePointer<CChar>?,
                             _ maxBytes: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let engine = gEngine, let t = tokens, n >= 0 else { return -1 }
    let toks = Array(UnsafeBufferPointer(start: t, count: Int(n)))
    let str = engine.detokenize(toks)
    let data = Array(str.utf8)
    if outBuf == nil { return Int32(data.count) }
    let copyN = min(data.count, Int(maxBytes))
    for i in 0..<copyN { outBuf![i] = CChar(bitPattern: data[i]) }
    return Int32(copyN)
}

@_cdecl("gemma_bos_id")
public func gemma_bos_id() -> UInt32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return gEngine?.weights.bosTokenId ?? 0
}

@_cdecl("gemma_eos_id")
public func gemma_eos_id() -> UInt32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return gEngine?.weights.eosTokenId ?? 0
}

// --- Introspection ---

@_cdecl("gemma_active_session_count")
public func gemma_active_session_count() -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return Int32(gSessions.count)
}

// Fill out_sids with the currently-open session handles, return count.
// If out_sids is nil, returns the count that would be written.
@_cdecl("gemma_active_session_ids")
public func gemma_active_session_ids(_ outSids: UnsafeMutablePointer<Int32>?,
                                      _ maxN: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    let sorted = gSessions.keys.sorted()
    if outSids == nil { return Int32(sorted.count) }
    let n = min(sorted.count, Int(maxN))
    for i in 0..<n { outSids![i] = sorted[i] }
    return Int32(n)
}

// Snapshot one session's KV state for the cache-tenancy viz.
//   outPosition: session.position (current k_len, i.e. token count)
//   outState:    SessionState enum (0..4), see gemma_session_state
//   outPages:    fill with phys page IDs this session owns (ordered)
//   returns:     number of pages written, or query-size when outPages is nil
@_cdecl("gemma_session_snapshot")
public func gemma_session_snapshot(_ sid: Int32,
                                    _ outPosition: UnsafeMutablePointer<Int32>?,
                                    _ outState: UnsafeMutablePointer<Int32>?,
                                    _ outPages: UnsafeMutablePointer<UInt32>?,
                                    _ maxPages: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    if let op = outPosition { op.pointee = Int32(s.positionForDebug) }
    if let os = outState {
        switch s.state {
        case .idle: os.pointee = 0
        case .priming: os.pointee = 1
        case .generating: os.pointee = 2
        case .paused: os.pointee = 3
        case .done: os.pointee = 4
        }
    }
    let pages = s.ownedPagesForDebug
    if outPages == nil { return Int32(pages.count) }
    let n = min(pages.count, Int(maxPages))
    for i in 0..<n { outPages![i] = UInt32(pages[i]) }
    return Int32(n)
}

// Refcount (owner-count) for a physical page. > 1 ⇒ shared across sessions.
@_cdecl("gemma_page_refcount")
public func gemma_page_refcount(_ phys: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let engine = gEngine else { return 0 }
    return Int32(engine.pageManager.pageRefcount(Int(phys)))
}

// Which session IDs currently own a given phys page. Fill outSids, return count.
@_cdecl("gemma_page_owners")
public func gemma_page_owners(_ phys: Int32,
                               _ outSids: UnsafeMutablePointer<Int32>?,
                               _ maxN: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let engine = gEngine else { return 0 }
    let owners = engine.pageManager.ownersOfPage(Int(phys))
    if outSids == nil { return Int32(owners.count) }
    let n = min(owners.count, Int(maxN))
    for i in 0..<n { outSids![i] = Int32(owners[i]) }
    return Int32(n)
}

// Bulk counts for a session: page_count + shared_count (= number of this
// session's pages with refcount > 1). Consolidated into ONE FFI call so
// UI pollers don't spam one-per-page calls — which serialize against the
// pump's tick() via ffiLock and collapse AR throughput (measured 3-4×
// slowdown at 2 Hz polling, vs negligible with this shape).
@_cdecl("gemma_session_counts")
public func gemma_session_counts(_ sid: Int32,
                                  _ outPageCount: UnsafeMutablePointer<Int32>?,
                                  _ outSharedCount: UnsafeMutablePointer<Int32>?) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let engine = gEngine, let s = gSessions[sid] else { return -1 }
    let owned = s.ownedPagesForDebug
    var shared = 0
    for phys in owned {
        if engine.pageManager.pageRefcount(phys) > 1 { shared += 1 }
    }
    if let pc = outPageCount   { pc.pointee = Int32(owned.count) }
    if let sc = outSharedCount { sc.pointee = Int32(shared) }
    return 0
}

// One-shot snapshot of every active session's aggregate KV stats. Writes
// a packed Int32 array into outBuf of shape [N, 5] where each row is
// (sid, position, state, page_count, shared_count). Returns N (number
// of sessions written) on success, negative on overflow or error.
//
// Collapsing the whole poll to a single ffiLock acquisition: UI pollers
// previously fired 1 + 2N FFI calls per /v1/kv/snapshot (active_session_ids,
// then session_snapshot + session_counts per session), each of which
// waited up to one full gemma_tick() duration (~30 ms) for ffiLock.
// At 2 Hz polling with 2 active sessions = 5 × 30 ms = 150 ms/poll of
// serialized lock contention per second → measurable AR throughput
// collapse. This bulk variant takes one lock once, runs all the Swift-
// side bookkeeping, releases. Observed: 130 ms/poll → ~30 ms/poll.
@_cdecl("gemma_kv_snapshot_summary")
public func gemma_kv_snapshot_summary(_ outBuf: UnsafeMutablePointer<Int32>?,
                                       _ maxSessions: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let engine = gEngine, let out = outBuf else { return 0 }
    let sids = Array(gSessions.keys).sorted()
    let stride = 5
    var written = 0
    for sid in sids {
        if written >= Int(maxSessions) { break }
        guard let s = gSessions[sid] else { continue }
        let stateCode: Int32
        switch s.state {
        case .idle:       stateCode = 0
        case .priming:    stateCode = 1
        case .generating: stateCode = 2
        case .paused:     stateCode = 3
        case .done:       stateCode = 4
        }
        let owned = s.ownedPagesForDebug
        var shared = 0
        for phys in owned {
            if engine.pageManager.pageRefcount(phys) > 1 { shared += 1 }
        }
        let base = written * stride
        out[base + 0] = sid
        out[base + 1] = Int32(s.positionForDebug)
        out[base + 2] = stateCode
        out[base + 3] = Int32(owned.count)
        out[base + 4] = Int32(shared)
        written += 1
    }
    return Int32(written)
}

// --- Vision / multimodal ---

// Bind the vision tower to a safetensors file. This is a CHEAP call now:
// it only creates the zero-copy bf16 source buffers (mmap-backed, ~0 RAM).
// The fp16 working weights are hydrated on first gemma_submit_image_path,
// and may be dropped/reloaded across memory-pressure events. Returns 0 on
// success, -1 on failure.
@_cdecl("gemma_vision_init")
public func gemma_vision_init(_ safetensorsPath: UnsafePointer<CChar>?) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let p = safetensorsPath else { return -1 }
    let pathStr = String(cString: p)
    do {
        let st = try SafetensorsFile(pathStr)
        gVisionResidency = VisionResidency(file: st)
        if gVisionQueue == nil { gVisionQueue = device.makeCommandQueue() }
        return 0
    } catch {
        print("gemma_vision_init failed: \(error)")
        return -1
    }
}

@_cdecl("gemma_vision_is_ready")
public func gemma_vision_is_ready() -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return gVisionResidency != nil ? 1 : 0
}

// Current residency state: 0=unloaded, 1=volatile, 2=pinned, -1=not bound.
@_cdecl("gemma_vision_residency_state")
public func gemma_vision_residency_state() -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let r = gVisionResidency else { return -1 }
    switch r.state {
    case .unloaded: return 0
    case .volatile_: return 1
    case .pinned: return 2
    }
}

// Working-set size in bytes. 0 when unloaded.
@_cdecl("gemma_vision_residency_bytes")
public func gemma_vision_residency_bytes() -> UInt64 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return UInt64(gVisionResidency?.workingSetBytes() ?? 0)
}

// Manually flip to volatile (signal "we're idle, OS take this if needed").
// Auto-called by the pressure source on .warning.
@_cdecl("gemma_vision_allow_evict")
public func gemma_vision_allow_evict() -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    gVisionResidency?.allowEvict()
    return 0
}

// Manually drop the working set entirely (simulate a .critical pressure
// event from tests; real pressure source calls this automatically).
@_cdecl("gemma_vision_force_drop")
public func gemma_vision_force_drop() -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    gVisionResidency?.forceDrop()
    return 0
}

// Preprocess + vision-tower + submit soft tokens to a session, bracketed
// by BOI (255999) / EOI (258882) markers the way Gemma-4 expects for an
// inline image in a user turn. PNG bytes are read from the given file path
// (Python writes to tempfile, passes path, cleans up after).
//
// Cache-aware: hashes the raw file bytes (SHA-256); on hit, reuses the
// previously-padded soft-tokens MTLBuffer and skips preprocess + vision
// entirely. The same buffer can back concurrent sessions safely since
// Session.submit(softTokens:) treats the buffer as read-only. LRU-evicts
// when gVisionCacheMaxEntries is exceeded.
//
// Returns number of soft tokens submitted (280 on success), or -1 on error.
@_cdecl("gemma_submit_image_path")
public func gemma_submit_image_path(_ sid: Int32,
                                     _ pngPath: UnsafePointer<CChar>?) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    guard let p = pngPath else { return -1 }
    let pathStr = String(cString: p)

    guard let padded = ensureCachedSofts(pngPath: pathStr) else { return -1 }

    // Bracket with BOI + EOI the same way runLmMultimodal does. The softs
    // chunk carries `pendingCB` so the LM tick can wait on the vision CB
    // lazily — vision computation overlaps with LM decode on other sessions.
    let BOI: UInt32 = 255999
    let EOI: UInt32 = 258882
    s.submit([BOI])
    s.submit(softTokens: padded.buffer, count: padded.count, isFp32: true,
             pendingCB: padded.pendingCB)
    s.submit([EOI])
    // After the first handoff, clear pendingCB from the cache entry so the
    // next cache hit on the same image doesn't wait again.
    padded.pendingCB = nil
    return Int32(padded.count)
}

// Pre-warm the cache without attaching to a session. Intended for
// POST /v1/images/prewarm: an agent can pre-populate entries it expects
// to re-reference later so the first "real" request sees a hit.
//
// Unlike gemma_submit_image_path, this WAITS for the vision CB to
// complete before returning. The async pipeline exists to let vision
// overlap with LM decode on concurrent sessions, but for the prewarm
// use case the whole point is "ensure this image is ready before the
// caller's next step" — returning with the CB still in flight defeats
// that semantic and makes batched-decode workflows (labeler, etc.)
// silently lose the batching win.
//
// Returns soft-token count on success (cache hit or miss), -1 on error.
@_cdecl("gemma_vision_prewarm_path")
public func gemma_vision_prewarm_path(_ pngPath: UnsafePointer<CChar>?) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let p = pngPath else { return -1 }
    let pathStr = String(cString: p)
    guard let padded = ensureCachedSofts(pngPath: pathStr) else { return -1 }
    padded.pendingCB?.waitUntilCompleted()
    padded.pendingCB = nil
    return Int32(padded.count)
}

// Fills out_hex (65 bytes minimum, includes null terminator) with the
// SHA-256 hex of the last prewarm/submit. Useful for clients that want
// to know the cache key.
@_cdecl("gemma_vision_last_cache_key")
public func gemma_vision_last_cache_key(_ outHex: UnsafeMutablePointer<CChar>?,
                                         _ maxBytes: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let h = gLastCacheKeyHex else { return -1 }
    let data = Array(h.utf8)
    if outHex == nil { return Int32(data.count) }
    let n = min(data.count, Int(maxBytes))
    for i in 0..<n { outHex![i] = CChar(bitPattern: data[i]) }
    return Int32(n)
}

// Copy soft tokens out of the cache by SHA-256 hex key. Lets the client
// round-trip vision tower outputs: run an image once, receive the softs,
// hand them back on future turns without the server re-running anything.
//
// If outPtr is nil, returns the number of bytes required (so the caller
// can size its buffer). Otherwise copies min(required, maxBytes) bytes
// and returns the number copied. Returns -1 on cache miss or bad key.
@_cdecl("gemma_vision_fetch_softs_by_key")
public func gemma_vision_fetch_softs_by_key(_ hexKey: UnsafePointer<CChar>?,
                                             _ outPtr: UnsafeMutablePointer<UInt8>?,
                                             _ maxBytes: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let k = hexKey else { return -1 }
    let hex = String(cString: k)
    guard hex.count == 64 else { return -1 }
    var bytes = [UInt8](); bytes.reserveCapacity(32)
    var idx = hex.startIndex
    for _ in 0..<32 {
        let next = hex.index(idx, offsetBy: 2)
        guard let b = UInt8(hex[idx..<next], radix: 16) else { return -1 }
        bytes.append(b); idx = next
    }
    let hashData = Data(bytes)
    guard let entry = gVisionCache[hashData] else { return -1 }
    let need = entry.buffer.length
    if outPtr == nil { return Int32(need) }
    let n = min(need, Int(maxBytes))
    memcpy(outPtr!, entry.buffer.contents(), n)
    return Int32(n)
}

// Register a control vector by caller-assigned string id. Sessions
// reference the cvec via this id in gemma_session_add_control. Bytes
// must be exactly HIDDEN * 2 (fp16). Returns 0 on success, -1 on error.
// Re-registering the same id replaces the existing entry (consumers
// referencing it continue to see the OLD buffer until they re-attach —
// the buffer is strongly retained by each ActiveControl).
// Return the currently-registered cvec ids as a comma-separated string.
// Useful for UIs that want to warn "this id isn't on the server" before
// submitting a request that would 400.
@_cdecl("gemma_control_list_ids")
public func gemma_control_list_ids(_ outPtr: UnsafeMutablePointer<CChar>?,
                                    _ maxBytes: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    let ids = gCvecRegistry.keys.sorted().joined(separator: ",")
    let bytes = Array(ids.utf8)
    if outPtr == nil { return Int32(bytes.count) }
    let n = min(bytes.count, Int(maxBytes))
    for i in 0..<n { outPtr![i] = CChar(bitPattern: bytes[i]) }
    return Int32(n)
}

@_cdecl("gemma_control_register_fp16")
public func gemma_control_register_fp16(_ idPtr: UnsafePointer<CChar>?,
                                         _ dataPtr: UnsafePointer<UInt8>?,
                                         _ nBytes: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let ip = idPtr, let dp = dataPtr else { return -1 }
    let id = String(cString: ip)
    if id.isEmpty { return -1 }
    let expected = HIDDEN * 2
    guard Int(nBytes) == expected else {
        print("[cvec] register id=\(id): size mismatch (got \(nBytes), expected \(expected))")
        return -1
    }
    guard let buf = device.makeBuffer(length: expected, options: .storageModeShared) else { return -1 }
    memcpy(buf.contents(), dp, expected)
    gCvecRegistry[id] = buf
    return 0
}

// Attach a cvector to a session with an ADSR envelope. Activation time
// (startPosition / startTurn) is captured as the session's current
// counters — the envelope's elapsed time is measured relative to that.
// Shape: 0=linear, 1=expIn, 2=expOut, 3=cubic (smoothstep).
// Units: 0=tokens, 1=turns.
@_cdecl("gemma_session_add_control")
public func gemma_session_add_control(_ sid: Int32,
                                       _ cvecIdPtr: UnsafePointer<CChar>?,
                                       _ layer: Int32,
                                       _ polarity: Float,
                                       _ peakMagnitude: Float,
                                       _ attack: Float,
                                       _ decay: Float,
                                       _ sustainLevel: Float,
                                       _ release: Float,
                                       _ shape: Int32,
                                       _ units: Int32,
                                       _ mode: Int32) -> Int32 {
    // mode: 0 = additive (residual += mag * cvec)
    //       1 = project  (residual projection onto cvec coerced to mag)
    // Additive stays the default for backward compat; mode=1 adds
    // representation-engineering primitives (target=0 removes the
    // feature, nonzero target coerces to a specific level).
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    guard let ip = cvecIdPtr else { return -1 }
    let cvecId = String(cString: ip)
    guard let buf = gCvecRegistry[cvecId] else {
        print("[cvec] session \(sid) add_control: no cvec registered for id=\(cvecId)")
        return -1
    }
    let shapes: [CvecShape] = [.linear, .expIn, .expOut, .cubic]
    let shapeVal = shapes[max(0, min(shapes.count - 1, Int(shape)))]
    let unitsVal: CvecUnits = (units == 0 ? .tokens : .turns)
    let modeVal: CvecMode = (mode == 1 ? .project : .additive)
    let env = CvecEnvelope(attack: attack, decay: decay, sustainLevel: sustainLevel,
                            release: release, peakMagnitude: peakMagnitude,
                            shape: shapeVal, units: unitsVal)
    let t = s.currentTimeCoords
    let ctrl = ActiveControl(cvecId: cvecId, buffer: buf, layer: Int(layer),
                              envelope: env, polarity: polarity,
                              startPosition: t.position, startTurn: t.turn,
                              mode: modeVal)
    s.addControl(ctrl)
    return 0
}

// Drop all active controls on a session (e.g. at turn-end or on reset).
@_cdecl("gemma_session_clear_controls")
public func gemma_session_clear_controls(_ sid: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    s.clearControls()
    return 0
}

// Teacher-forcing + logit readback used by /v1/perplexity to walk a
// known completion under controls and compute per-position logprobs.
// The UI compares perplexity across lanes + control-configs to spot
// locally-unstable fitted directions (distributions collapsed to a
// single fixed-point token) versus genuinely-shifted trajectories.
//
// Workflow from Python:
//   1. open_session, attach controls, submit(prompt), drain.
//   2. read slot logits  → first completion token's logit distribution
//   3. set_next_input(completion[0]); tick(); read slot logits → next
//   4. ... repeat over completion length.
// The session's AR tick path uses nextGeneratedInput as the input
// token; forceNextInput overrides it with the caller's choice.

// Copy the current slot's logits ([VOCAB] fp16) into the caller's
// buffer. The logits represent the model's prediction for the
// position that the session is about to consume (i.e. the NEXT
// token's distribution). Returns VOCAB on success, negative on error.
@_cdecl("gemma_session_get_slot_logits")
public func gemma_session_get_slot_logits(_ sid: Int32,
                                           _ outBuf: UnsafeMutablePointer<Float16>?) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid], let slot = s.slot, let out = outBuf else { return -1 }
    let logP = logits.contents().assumingMemoryBound(to: Float16.self)
    let base = slot * VOCAB
    for i in 0..<VOCAB { out[i] = logP[base + i] }
    return Int32(VOCAB)
}

// Pause/resume used by /v1/perplexity. While paused, the pump skips
// the session (wantsSlot=false), so the caller can control exactly
// when forward passes happen relative to logit reads.
@_cdecl("gemma_session_pause")
public func gemma_session_pause(_ sid: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    s.pauseForExternal()
    return 0
}

@_cdecl("gemma_session_resume")
public func gemma_session_resume(_ sid: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    s.resumeFromExternalPause()
    return 0
}

// Teacher-force: override the token the next AR tick will consume.
// Session must be in .generating state (call after prefill has
// completed and the session has entered AR). Does not itself trigger
// a tick — caller drives tick() separately.
@_cdecl("gemma_session_set_next_input")
public func gemma_session_set_next_input(_ sid: Int32,
                                          _ token: UInt32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    s.forceNextInput(token)
    return 0
}

// Session sampling config. temperature=0 is greedy argmax (default,
// matches prior behavior exactly). Positive values enable stochastic
// softmax sampling with the session's own RNG — each re-run produces
// a different trajectory, useful for demonstrating intervention effect
// as a DISTRIBUTIONAL shift rather than a single argmax-flip. No top-p
// / top-k filtering yet; temperature alone is enough for visible
// trajectory diversity at the demo's scale.
@_cdecl("gemma_session_set_temperature")
public func gemma_session_set_temperature(_ sid: Int32,
                                           _ temperature: Float) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    s.samplingTemperature = max(0, temperature)
    return 0
}

// Signal a control's sustain → release transition (begin the release
// ramp NOW). Pass the same cvec_id used at add_control time.
@_cdecl("gemma_session_release_control")
public func gemma_session_release_control(_ sid: Int32,
                                           _ cvecIdPtr: UnsafePointer<CChar>?) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    guard let ip = cvecIdPtr else { return -1 }
    let t = s.currentTimeCoords
    s.releaseControl(cvecId: String(cString: ip), position: t.position, turn: t.turn)
    return 0
}

// Attach a detector: a named direction that gets measured against the
// post-FFN residual at the specified layer every tick. The resulting
// scalar intensity is visible to triggers and readable via
// gemma_session_read_intensity. `name` is a session-scoped alias used
// by gemma_session_add_trigger; `cvecId` picks an already-registered
// vector from the registry (detectors and effectors share the same
// registry since they're both HIDDEN-length fp16 vectors).
@_cdecl("gemma_session_add_detector")
public func gemma_session_add_detector(_ sid: Int32,
                                        _ namePtr: UnsafePointer<CChar>?,
                                        _ cvecIdPtr: UnsafePointer<CChar>?,
                                        _ layer: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    guard let np = namePtr, let ip = cvecIdPtr else { return -1 }
    let name = String(cString: np)
    let cvecId = String(cString: ip)
    guard let buf = gCvecRegistry[cvecId] else {
        print("[cvec] session \(sid) add_detector: no cvec registered for id=\(cvecId)")
        return -1
    }
    s.addDetector(DetectorAttachment(name: name, cvecId: cvecId,
                                      buffer: buf, layer: Int(layer)))
    return 0
}

// Attach a gated trigger: when `detectorName`'s intensity crosses
// `threshold` in the direction set by `conditionCode`, restart the
// envelope of the effector control whose cvecId matches
// `effectorCvecId`. conditionCode: 0 = onExceed, 1 = onFall.
@_cdecl("gemma_session_add_trigger")
public func gemma_session_add_trigger(_ sid: Int32,
                                       _ detNamePtr: UnsafePointer<CChar>?,
                                       _ conditionCode: Int32,
                                       _ threshold: Float,
                                       _ effectorCvecIdPtr: UnsafePointer<CChar>?) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    guard let dp = detNamePtr, let ep = effectorCvecIdPtr else { return -1 }
    let cond: TriggerCondition = (conditionCode == 0
        ? .onExceed(threshold: threshold)
        : .onFall(threshold: threshold))
    s.addTrigger(SessionTrigger(detectorName: String(cString: dp),
                                 condition: cond,
                                 effectorCvecId: String(cString: ep)))
    return 0
}

// Drop all detectors / triggers on a session.
@_cdecl("gemma_session_clear_detectors")
public func gemma_session_clear_detectors(_ sid: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    s.clearDetectors(); s.clearTriggers()
    return 0
}

// Peek the most recent intensity reading for a detector (for telemetry
// / UI, not for the main feedback path — triggers already consume this
// internally). Returns 0.0 for unknown detector names.
@_cdecl("gemma_session_read_intensity")
public func gemma_session_read_intensity(_ sid: Int32,
                                          _ namePtr: UnsafePointer<CChar>?) -> Float {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return 0 }
    guard let np = namePtr else { return 0 }
    let name = String(cString: np)
    return s.detectors.first(where: { $0.name == name })?.lastIntensity ?? 0
}

// Pairwise prose cvec constructor: set the layer whose residual should
// be blitted into the capture buffer after each tick's post-FFN write.
// Pass -1 to disable. Active for as long as it's set — caller is
// responsible for resetting after it's done capturing.
@_cdecl("gemma_set_capture_layer")
public func gemma_set_capture_layer(_ layer: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    gResidualCaptureLayer = Int(layer)
    return 0
}

// All-layer capture toggle. When enabled, every layer's post-FFN
// residual gets blitted into its L-indexed slot of gAllLayerCaptureBuf
// during each tick. Meant to be on briefly during screening passes
// and off the rest of the time (tiny but not zero per-layer blit cost).
@_cdecl("gemma_set_capture_all_layers")
public func gemma_set_capture_all_layers(_ enabled: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    gCaptureAllLayers = (enabled != 0)
    return 0
}

// Copy the NUM_LAYERS × HIDDEN × fp16 all-layer capture buffer into
// the caller's output. Returns bytes written, or if outPtr is nil,
// the needed size.
@_cdecl("gemma_get_all_layer_residuals")
public func gemma_get_all_layer_residuals(_ outPtr: UnsafeMutablePointer<UInt8>?,
                                           _ maxBytes: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    let need = NUM_LAYERS * HIDDEN * 2
    guard let p = outPtr else { return Int32(need) }
    let n = min(need, Int(maxBytes))
    memcpy(p, gAllLayerCaptureBuf.contents(), n)
    return Int32(n)
}

// Copy the most-recently-captured residual (HIDDEN × fp16 = 5632 B)
// into the caller's buffer. Returns bytes written (= HIDDEN * 2) or
// the buffer size needed if outPtr is nil. The buffer is overwritten
// on every tick while capture is active — caller times the read to
// pick up the residual from the tick they care about (typically the
// last priming tick before the session transitions to .generating).
@_cdecl("gemma_get_captured_residual")
public func gemma_get_captured_residual(_ outPtr: UnsafeMutablePointer<UInt8>?,
                                         _ maxBytes: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    let need = HIDDEN * 2
    guard let p = outPtr else { return Int32(need) }
    let n = min(need, Int(maxBytes))
    memcpy(p, gResidualCaptureBuf.contents(), n)
    return Int32(n)
}

// Drain per-token samples as a JSON array and copy into caller-provided
// buffer. Returns bytes written (truncated if buffer too small — caller
// should size generously, ~200 bytes per sample is a safe upper bound).
// If outPtr is nil, returns the number of bytes that WOULD be written
// without draining — use this to size your buffer. Otherwise the queue
// IS drained (consumed) on each call.
@_cdecl("gemma_session_poll_samples_json")
public func gemma_session_poll_samples_json(_ sid: Int32,
                                             _ outPtr: UnsafeMutablePointer<CChar>?,
                                             _ maxBytes: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    // When asked for a size only (outPtr == nil), don't drain — serialize
    // a snapshot of the current queue without consuming it. A subsequent
    // call with a buffer drains for real.
    if outPtr == nil {
        // Quick upper bound: each sample serializes to O(150 bytes)
        // typical; give the caller enough headroom.
        return Int32(s.pendingSamplesCount * 256 + 4)
    }
    let json = s.drainSamplesJson()
    let bytes = Array(json.utf8)
    let n = min(bytes.count, Int(maxBytes))
    for i in 0..<n { outPtr![i] = CChar(bitPattern: bytes[i]) }
    return Int32(n)
}

// Submit pre-computed soft tokens to a session. The client is handing
// back softs they received from a previous image submission — the server
// brackets with BOI/EOI and appends to the chunk queue without running
// the vision tower. This is the inverse of gemma_vision_fetch_softs_by_key.
//
// `byteCount` must equal `nTokens * HIDDEN * (isFp32 ? 4 : 2)`.
// Returns nTokens on success, -1 on error.
@_cdecl("gemma_submit_softs")
public func gemma_submit_softs(_ sid: Int32,
                                _ ptr: UnsafePointer<UInt8>?,
                                _ byteCount: Int32,
                                _ nTokens: Int32,
                                _ isFp32: Int32) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let s = gSessions[sid] else { return -1 }
    guard let src = ptr, nTokens > 0 else { return -1 }
    let fp32 = isFp32 != 0
    let bpe = fp32 ? 4 : 2
    let expected = Int(nTokens) * HIDDEN * bpe
    guard Int(byteCount) == expected else {
        print("gemma_submit_softs: size mismatch (got \(byteCount), expected \(expected) for \(nTokens) tokens at hidden=\(HIDDEN), fp32=\(fp32))")
        return -1
    }
    guard let buf = device.makeBuffer(length: expected, options: .storageModeShared) else { return -1 }
    memcpy(buf.contents(), src, expected)

    let BOI: UInt32 = 255999
    let EOI: UInt32 = 258882
    s.submit([BOI])
    s.submit(softTokens: buf, count: Int(nTokens), isFp32: fp32)
    s.submit([EOI])
    return nTokens
}

// Internal: hash the file, check/populate the cache. Shared by submit + prewarm.
private var gLastCacheKeyHex: String?
private func ensureCachedSofts(pngPath: String) -> CachedSofts? {
    guard let residency = gVisionResidency else {
        print("ensureCachedSofts: vision not initialized")
        return nil
    }
    guard let fileData = try? Data(contentsOf: URL(fileURLWithPath: pngPath)) else {
        print("ensureCachedSofts: failed to read \(pngPath)")
        return nil
    }
    let hash = SHA256.hash(data: fileData)
    let hashData = Data(hash)
    gLastCacheKeyHex = hash.map { String(format: "%02x", $0) }.joined()

    gVisionCacheTick &+= 1
    if var hit = gVisionCache[hashData] {
        hit.lastUsed = gVisionCacheTick
        gVisionCache[hashData] = hit
        gVisionCacheHits &+= 1
        return hit
    }
    gVisionCacheMisses &+= 1

    // Miss: run preprocess + vision tower. Ensure the residency is .pinned
    // for the duration of the forward pass, then flip to .volatile so the
    // OS can reclaim the working set when we're idle again.
    do {
        try residency.ensurePinned()
    } catch {
        print("ensureCachedSofts: vision hydrate failed: \(error)")
        return nil
    }
    defer { residency.allowEvict() }
    guard let visWeights = residency.weights else {
        print("ensureCachedSofts: residency returned nil weights")
        return nil
    }

    let batch: PatchBatch
    do { batch = try gemma4ImagePreprocess(path: pngPath, device: device) }
    catch {
        print("ensureCachedSofts: preprocess failed: \(error)")
        return nil
    }
    // Submit vision work on the dedicated vision queue so LM-queue CBs
    // (running concurrently from other sessions) aren't serialized behind it.
    let vq = gVisionQueue ?? queue
    let (rawSoftTokens, rawNPooled, cb) = runVisionTowerForwardAsync(
        batch: batch, weights: visWeights, device: device, queue: vq)
    if rawNPooled == 0 { return nil }

    // Allocate the padded [image_seq_length=280, HIDDEN] fp32 target and
    // zero-init on CPU (trivial — 3 MB memset). The GPU blit below copies
    // rawSoftTokens' computed rows into the head. This keeps the padded
    // tail bytes quietly at zero, matching what the old sync memcpy did.
    let targetSoft = 280
    let padded = device.makeBuffer(length: targetSoft * HIDDEN * 4,
                                    options: .storageModeShared)!
    memset(padded.contents(), 0, padded.length)
    let copyRows = min(rawNPooled, targetSoft)

    // Blit on the vision queue — runs after `cb` completes (same queue,
    // serialized). When `padCB` completes, `padded` is fully materialized.
    // The consumer (LM tick softTokens reader) waits on `padCB` before
    // reading, so vision work pipelines against LM decode: the LM queue
    // can process other sessions while vision's two CBs churn on vq.
    let padCB = vq.makeCommandBuffer()!
    let blit = padCB.makeBlitCommandEncoder()!
    blit.copy(from: rawSoftTokens, sourceOffset: 0,
              to: padded, destinationOffset: 0,
              size: copyRows * HIDDEN * 4)
    blit.endEncoding()
    padCB.commit()

    let entry = CachedSofts(buffer: padded, count: targetSoft,
                             lastUsed: gVisionCacheTick, bytes: padded.length,
                             pendingCB: padCB)
    gVisionCache[hashData] = entry

    // LRU evict if over cap.
    if gVisionCache.count > gVisionCacheMaxEntries {
        // Evict lowest lastUsed. O(N) at cap=64 so fine.
        if let (oldKey, _) = gVisionCache.min(by: { $0.value.lastUsed < $1.value.lastUsed }) {
            gVisionCache.removeValue(forKey: oldKey)
        }
    }
    return entry
}

@_cdecl("gemma_vision_cache_entries")
public func gemma_vision_cache_entries() -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return Int32(gVisionCache.count)
}

@_cdecl("gemma_vision_cache_hits")
public func gemma_vision_cache_hits() -> UInt64 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return gVisionCacheHits
}

@_cdecl("gemma_vision_cache_misses")
public func gemma_vision_cache_misses() -> UInt64 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return gVisionCacheMisses
}

@_cdecl("gemma_vision_cache_bytes")
public func gemma_vision_cache_bytes() -> UInt64 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return gVisionCache.values.reduce(0) { $0 + UInt64($1.bytes) }
}

@_cdecl("gemma_vision_cache_clear")
public func gemma_vision_cache_clear() -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    let n = gVisionCache.count
    gVisionCache.removeAll()
    return Int32(n)
}
