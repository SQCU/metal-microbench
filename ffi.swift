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
private var gVisionWeights: VisionWeights?
private var gSessions: [Int32: Session] = [:]
private var gNextHandle: Int32 = 1

// Vision soft-tokens cache — keyed by SHA-256 of the raw PNG/JPEG bytes.
// On a cache hit we skip preprocessing + vision tower entirely (saves ~7 s
// per repeated image on M5) and just reuse the already-padded MTLBuffer.
// The same MTLBuffer can back N concurrent sessions: Session.submit(softTokens:)
// stores it in a .softTokens chunk which is read-only from the kernel side.
private struct CachedSofts {
    let buffer: MTLBuffer       // padded to targetSoft rows, fp32
    let count: Int              // always targetSoft=280 currently
    var lastUsed: UInt64        // monotonic tick counter for LRU
    let bytes: Int              // buffer.length, for stats
}
private var gVisionCache: [Data: CachedSofts] = [:]
private var gVisionCacheHits: UInt64 = 0
private var gVisionCacheMisses: UInt64 = 0
private var gVisionCacheTick: UInt64 = 0
private let gVisionCacheMaxEntries = 64    // ~200 MB at 280 × 2816 × 4 B each

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
        gEngine = LmEngine(weights: w)
        return 0
    } catch {
        print("gemma_init failed: \(error)")
        return -1
    }
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

// --- Vision / multimodal ---

// Load vision weights from the Gemma-4 safetensors file. Must be called once
// before any gemma_submit_image_path. Returns 0 on success, -1 on failure.
@_cdecl("gemma_vision_init")
public func gemma_vision_init(_ safetensorsPath: UnsafePointer<CChar>?) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let p = safetensorsPath else { return -1 }
    let pathStr = String(cString: p)
    do {
        let st = try SafetensorsFile(pathStr)
        gVisionWeights = try loadVisionWeights(st, device: device)
        return 0
    } catch {
        print("gemma_vision_init failed: \(error)")
        return -1
    }
}

@_cdecl("gemma_vision_is_ready")
public func gemma_vision_is_ready() -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    return gVisionWeights != nil ? 1 : 0
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

    // Bracket with BOI + EOI the same way runLmMultimodal does.
    let BOI: UInt32 = 255999
    let EOI: UInt32 = 258882
    s.submit([BOI])
    s.submit(softTokens: padded.buffer, count: padded.count, isFp32: true)
    s.submit([EOI])
    return Int32(padded.count)
}

// Pre-warm the cache without attaching to a session. Intended for
// POST /v1/images/prewarm: an agent can pre-populate entries it expects
// to re-reference later so the first "real" request sees a hit.
//
// Returns soft-token count on success (cache hit or miss), -1 on error.
@_cdecl("gemma_vision_prewarm_path")
public func gemma_vision_prewarm_path(_ pngPath: UnsafePointer<CChar>?) -> Int32 {
    ffiLock.lock(); defer { ffiLock.unlock() }
    guard let p = pngPath else { return -1 }
    let pathStr = String(cString: p)
    guard let padded = ensureCachedSofts(pngPath: pathStr) else { return -1 }
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

// Internal: hash the file, check/populate the cache. Shared by submit + prewarm.
private var gLastCacheKeyHex: String?
private func ensureCachedSofts(pngPath: String) -> CachedSofts? {
    guard let visWeights = gVisionWeights else {
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

    // Miss: run preprocess + vision tower.
    let batch: PatchBatch
    do { batch = try gemma4ImagePreprocess(path: pngPath, device: device) }
    catch {
        print("ensureCachedSofts: preprocess failed: \(error)")
        return nil
    }
    let (rawSoftTokens, rawNPooled) = runVisionTowerForward(
        batch: batch, weights: visWeights, device: device, queue: queue)
    if rawNPooled == 0 { return nil }

    // Pad to image_seq_length=280, fp32.
    let targetSoft = 280
    let padded = device.makeBuffer(length: targetSoft * HIDDEN * 4,
                                    options: .storageModeShared)!
    memset(padded.contents(), 0, padded.length)
    let copyRows = min(rawNPooled, targetSoft)
    memcpy(padded.contents(), rawSoftTokens.contents(), copyRows * HIDDEN * 4)

    let entry = CachedSofts(buffer: padded, count: targetSoft,
                             lastUsed: gVisionCacheTick, bytes: padded.length)
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
