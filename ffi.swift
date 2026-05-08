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

// Module-internal so the engine's chainAdvance / per-path completion
// ffiLock and gEngine moved to bootstrap.swift so the executable build
// (which excludes ffi.swift) can still resolve them in lm_engine.swift /
// buildStepCB. ffi.swift retains gEngine's lifecycle (assigned at
// gemma_init, cleared at teardown).
private var gVisionResidency: VisionResidency?
private var gPressureSource: DispatchSourceMemoryPressure?
// 2026-05-07: gSessions / gNextHandle deleted. The per-session FFI surface
// (gemma_session_open / submit / poll / close) was retired earlier in
// favour of the batch FFI; this dict was only ever READ by the two
// diagnostic snapshot FFIs (gemma_scheduler_snapshot,
// gemma_kv_snapshot_summary) that always returned 0 entries because
// nothing populated it. Both snapshots have been deleted along with
// the dict.

// Vision soft-tokens cache — keyed by SHA-256 of the raw PNG/JPEG bytes.
// On a cache hit we skip preprocessing + vision tower entirely (saves ~7 s
// per repeated image on M5) and just reuse the already-padded MTLBuffer.
// The same MTLBuffer can back N concurrent sessions: Session.submit(softTokens:)
// stores it in a .softTokens chunk which is read-only from the kernel side.
// Cache entry. `eventTicket` is the value the vision pad-blit CB signals
// on `gVisionEvent` after writing this buffer; LM consumers
// `cb.encodeWaitForEvent(gVisionEvent, value: eventTicket)` so the GPU
// itself waits before reading. Under the async pipeline the CPU never
// blocks — the wait is purely a GPU-side queue ordering primitive.
//
// Once a consumer's CB has actually read the buffer, the ticket can be
// reset to 0 (no-wait) so subsequent consumers don't re-wait
// unnecessarily — but the wait is cheap (event already signaled) so
// leaving it in place is harmless.
internal class CachedSofts {
    let buffer: MTLBuffer       // padded to targetSoft rows, fp32
    let count: Int              // always targetSoft=280 currently
    var lastUsed: UInt64        // monotonic tick counter for LRU
    let bytes: Int              // buffer.length, for stats
    var eventTicket: UInt64     // 0 = no wait needed (already signaled past, or never had one)
    init(buffer: MTLBuffer, count: Int, lastUsed: UInt64, bytes: Int, eventTicket: UInt64) {
        self.buffer = buffer; self.count = count; self.lastUsed = lastUsed
        self.bytes = bytes; self.eventTicket = eventTicket
    }
}
private var gVisionCache: [Data: CachedSofts] = [:]
internal var gVisionCacheHits: UInt64 = 0
internal var gVisionCacheMisses: UInt64 = 0
private var gVisionCacheTick: UInt64 = 0
// Convenience for stats endpoints in ffi_batch.swift.
internal var gVisionCacheEntryCount: Int { return gVisionCache.count }
private let gVisionCacheMaxEntries = 64    // ~200 MB at 280 × 2816 × 4 B each

// Dedicated command queue for vision tower work. Runs concurrently with
// the main LM queue on M5 Max — two queues can share ALU partitions,
// enabling vision(image N+1) ∥ LM.decode(label N) pipelining.
var gVisionQueue: MTLCommandQueue?

// --- Initialization ---

@_cdecl("gemma_init")
public func gemma_init(_ ggufPath: UnsafePointer<CChar>?) -> Int32 {
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
    return gEngine != nil ? 1 : 0
}

// Per-session FFI exports (open_session / close_session / pause_session /
// gemma_submit / gemma_append / gemma_tick) deleted 2026-04-26. Replaced
// by the unified batch-shaped FFI in ffi_batch.swift: gemma_submit and
// gemma_poll take a BatchRequest carrying N streams' actions. The
// bridge no longer holds per-session handles; it submits StreamSpec
// records keyed by client-assigned stream_id. See notes/specs/
// batch_ffi_abi.md and notes/decisions/2026-04-26-remove-session-
// concurrency-primitives.md.

// Phase-1 validator entry — see docs/dataflow_pipeline_spec.md.
//
// Runs the `sample_token` kernel against caller-supplied logits +
// per-slot sampling params, writes chosen token ids to outSampledPtr.
// Self-contained: allocates scratch, dispatches the kernel on a fresh
// CB, blocks until done, returns. Intended ONLY for the Python-side
// unit/statistical validator; the real engine integration in Phase 2
// will dispatch the kernel as part of the step CB (no per-call CB).
//
// For T<=0 slots: bit-exact argmax vs CPU.
// For T>0 slots: per-slot inverse-CDF draw using philox(seed, step, slot).
//                Same tuple → same draw (reproducible); distribution
//                matches CPU softmax within sampling noise.
@_cdecl("gemma_test_sample_token")
public func gemma_test_sample_token(
    _ logitsPtr: UnsafePointer<UInt16>?,    // [B*VOCAB] fp16 bit-pattern
    _ biasPtr: UnsafePointer<Float>?,       // [B*VOCAB] fp32 — pass zeros if unused
    _ tempPtr: UnsafePointer<Float>?,       // [B]
    _ minPPtr: UnsafePointer<Float>?,       // [B]
    _ seedPtr: UnsafePointer<UInt32>?,      // [B]
    _ stepPtr: UnsafePointer<UInt32>?,      // [B]
    _ activePtr: UnsafePointer<UInt32>?,    // [B]
    _ outSampledPtr: UnsafeMutablePointer<UInt32>?
) -> Int32 {
    guard let lp = logitsPtr, let bp = biasPtr, let tp = tempPtr,
          let mp = minPPtr, let sp = seedPtr, let stp = stepPtr,
          let ap = activePtr, let op = outSampledPtr
    else { return -1 }

    let bVocabBytesHalf = B * VOCAB * 2
    let bVocabBytesFloat = B * VOCAB * 4
    guard let logitsBuf = device.makeBuffer(length: bVocabBytesHalf, options: .storageModeShared),
          let biasBuf   = device.makeBuffer(length: bVocabBytesFloat, options: .storageModeShared),
          let tempBuf   = device.makeBuffer(length: B * 4, options: .storageModeShared),
          let minPBuf   = device.makeBuffer(length: B * 4, options: .storageModeShared),
          let seedBuf   = device.makeBuffer(length: B * 4, options: .storageModeShared),
          let stepBuf   = device.makeBuffer(length: B * 4, options: .storageModeShared),
          let activeBuf = device.makeBuffer(length: B * 4, options: .storageModeShared),
          let tokBuf    = device.makeBuffer(length: B * 4, options: .storageModeShared)
    else { return -2 }

    memcpy(logitsBuf.contents(), lp, bVocabBytesHalf)
    memcpy(biasBuf.contents(),   bp, bVocabBytesFloat)
    memcpy(tempBuf.contents(),   tp, B * 4)
    memcpy(minPBuf.contents(),   mp, B * 4)
    memcpy(seedBuf.contents(),   sp, B * 4)
    memcpy(stepBuf.contents(),   stp, B * 4)
    memcpy(activeBuf.contents(), ap, B * 4)

    let cb = queue.makeCommandBuffer()!
    encSampleToken(cb, logits: logitsBuf,
                    samplingLogitBias: biasBuf,
                    samplingTemp: tempBuf, samplingMinP: minPBuf,
                    samplingSeed: seedBuf, samplingStep: stepBuf,
                    samplingActive: activeBuf,
                    inputTokens: tokBuf, vocab: VOCAB)
    cb.commit(); cb.waitUntilCompleted()
    if let e = cb.error { print("test_sample_token: GPU err \(e)"); return -3 }

    memcpy(op, tokBuf.contents(), B * 4)
    return 0
}

// Profiling readout for AR-step throughput debugging. Exposes the engine's
// running totals: AR steps observed, GPU exec ms summed, wall ms summed.
// gap = (wall - gpu) / steps  →  CPU/scheduling overhead per step.
@_cdecl("gemma_engine_ar_profile")
public func gemma_engine_ar_profile(_ outSteps: UnsafeMutablePointer<Int32>?,
                                     _ outGpuMs: UnsafeMutablePointer<Double>?,
                                     _ outWallMs: UnsafeMutablePointer<Double>?) -> Int32 {
    guard let engine = gEngine else { return -1 }
    outSteps?.pointee = Int32(engine.prof_arSteps)
    outGpuMs?.pointee = engine.prof_gpuMsSum
    outWallMs?.pointee = engine.prof_wallMsSum
    return 0
}

@_cdecl("gemma_engine_ar_profile_detailed")
public func gemma_engine_ar_profile_detailed(_ outSteps: UnsafeMutablePointer<Int32>?,
                                              _ outGpuMs: UnsafeMutablePointer<Double>?,
                                              _ outWallMs: UnsafeMutablePointer<Double>?,
                                              _ outHandlerMs: UnsafeMutablePointer<Double>?,
                                              _ outFinalizeMs: UnsafeMutablePointer<Double>?,
                                              _ outPrepMs: UnsafeMutablePointer<Double>?) -> Int32 {
    guard let engine = gEngine else { return -1 }
    outSteps?.pointee = Int32(engine.prof_arSteps)
    outGpuMs?.pointee = engine.prof_gpuMsSum
    outWallMs?.pointee = engine.prof_wallMsSum
    outHandlerMs?.pointee = engine.prof_handlerLatencyMsSum
    outFinalizeMs?.pointee = engine.prof_finalizeMsSum
    outPrepMs?.pointee = engine.prof_prepMsSum
    return 0
}

// Per-session FFI exports (gemma_has_work / gemma_poll / gemma_poll_all /
// gemma_session_set_structured_cot / gemma_session_state /
// gemma_session_position) deleted 2026-04-26. Has_work and the bulk
// drain are no longer meaningful since the batch-shaped gemma_poll
// internally drives the engine and returns updates for all live
// streams in one call. Structured-cot is now SamplingParams.cot_labels
// on every submit; per-session state/position are reported via the
// batch StreamUpdate's state + counters fields.

// --- Tokenizer (so Python doesn't have to bundle a second one) ---

// Tokenize text_len UTF-8 bytes. If outTokens is nil, returns the number of
// tokens that would be produced (query-size pattern). Otherwise fills up to
// maxTokens and returns the actual count written (clamped to maxTokens).
@_cdecl("gemma_tokenize")
public func gemma_tokenize(_ text: UnsafePointer<CChar>?, _ textLen: Int32,
                           _ addBos: Int32,
                           _ outTokens: UnsafeMutablePointer<UInt32>?,
                           _ maxTokens: Int32) -> Int32 {
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
    return gEngine?.weights.bosTokenId ?? 0
}

@_cdecl("gemma_eos_id")
public func gemma_eos_id() -> UInt32 {
    return gEngine?.weights.eosTokenId ?? 0
}

// --- Introspection ---
// gemma_active_session_count, gemma_active_session_ids,
// gemma_session_snapshot deleted 2026-04-26: per-session bookkeeping
// has moved to the bridge. Engine-level totals (cached_pages,
// active_streams, etc.) are reported by gemma_status; per-stream
// state + position is in StreamUpdate.

// Refcount (owner-count) for a physical page. > 1 ⇒ shared across sessions.
@_cdecl("gemma_page_refcount")
public func gemma_page_refcount(_ phys: Int32) -> Int32 {
    guard let engine = gEngine else { return 0 }
    return Int32(engine.pageManager.pageRefcount(Int(phys)))
}

// gemma_page_owners deleted 2026-05-07: per-page session ownership is no
// longer a meaningful concept. Under the anonymous-pool refactor pages
// have a refcount only — no per-page list of holding sessions exists.
// Use gemma_page_refcount for "is this page shared" (refcount > 1).

// gemma_session_counts deleted 2026-04-26: per-session page counts are
// now per-stream, surfaced via StreamUpdate.cache_hits / cache_misses
// (measured in tokens; pages are an internal implementation detail).

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
// ── Heretic per-write directional ablation ──────────────────────────
// Configure engine-level write ablations. `component` is 0 for the
// attention out-projection write (`mlp_out` pre-residual-add) and 1 for
// the FFN combined write (`ffn_combined` pre-residual-add-with-scale).
// `cvecPtr` points to HIDDEN fp16 halves (the unit-norm r̂ for this
// layer); we copy into a fresh MTLBuffer so the caller can free their
// buffer after the call. alpha is heretic's α(L) for this layer —
// typically 0-1 for ablation, negative for amplification-induction.
// Re-calling with the same (layer, component) replaces the prior entry.
@_cdecl("gemma_engine_set_write_ablation")
public func gemma_engine_set_write_ablation(
    _ layer: Int32, _ component: Int32,
    _ cvecPtr: UnsafePointer<UInt8>?, _ cvecByteCount: Int32,
    _ alpha: Float
) -> Int32 {
    guard let engine = gEngine else { return -1 }
    guard let comp = AblationComponent(rawValue: component) else { return -2 }
    guard let p = cvecPtr, cvecByteCount == Int32(HIDDEN * 2) else { return -3 }
    let buf = device.makeBuffer(length: Int(cvecByteCount), options: .storageModeShared)!
    memcpy(buf.contents(), p, Int(cvecByteCount))
    let entry = LayerComponentAblation(layer: Int(layer), component: comp,
                                         rHatBuf: buf, alpha: alpha)
    if let i = engine.writeAblations.firstIndex(
        where: { $0.layer == Int(layer) && $0.component == comp }) {
        engine.writeAblations[i] = entry
    } else {
        engine.writeAblations.append(entry)
    }
    return 0
}

@_cdecl("gemma_engine_clear_write_ablations")
public func gemma_engine_clear_write_ablations() -> Int32 {
    gEngine?.writeAblations.removeAll()
    return 0
}

// Count of currently-configured ablation entries. Handy for testing.
@_cdecl("gemma_engine_write_ablation_count")
public func gemma_engine_write_ablation_count() -> Int32 {
    return Int32(gEngine?.writeAblations.count ?? 0)
}

// Scheduler globals in one call: batch size, resident count, lifetime
// totals, last step ms. Used by the /api/extra/scheduler observability
// endpoint. Complements gemma_kv_snapshot_summary (per-session KV stats)
// with per-engine counters that the latter doesn't expose.
@_cdecl("gemma_engine_scheduler_stats")
public func gemma_engine_scheduler_stats(
    _ outB:              UnsafeMutablePointer<Int32>?,
    _ outResidentCount:  UnsafeMutablePointer<Int32>?,
    _ outTotalSteps:     UnsafeMutablePointer<Int64>?,
    _ outTotalTokens:    UnsafeMutablePointer<Int64>?,
    _ outTotalSlotTicks: UnsafeMutablePointer<Int64>?,
    _ outLastStepMs:     UnsafeMutablePointer<Double>?
) -> Int32 {
    guard let e = gEngine else { return -1 }
    if let p = outB              { p.pointee = Int32(B) }
    if let p = outResidentCount  { p.pointee = Int32(e.residentSessions.count) }
    if let p = outTotalSteps     { p.pointee = Int64(e.totalSteps) }
    if let p = outTotalTokens    { p.pointee = Int64(e.totalTokensGenerated) }
    if let p = outTotalSlotTicks { p.pointee = Int64(e.totalSlotTicks) }
    if let p = outLastStepMs     { p.pointee = e.lastStepMs }
    return 0
}

// gemma_scheduler_snapshot + gemma_kv_snapshot_summary deleted 2026-05-07:
// both iterated `gSessions` which was never written to, so they always
// returned 0 entries. Zero callers in Python or Swift confirmed via grep.
// Replacements (if anything ever wants this telemetry again): iterate
// engine.requestForStream.values directly — that's the canonical live-
// request set under the anonymous-pool refactor.

// --- Vision / multimodal ---

// Bind the vision tower to a safetensors file. This is a CHEAP call now:
// it only creates the zero-copy bf16 source buffers (mmap-backed, ~0 RAM).
// The fp16 working weights are hydrated on first gemma_submit_image_bytes,
// and may be dropped/reloaded across memory-pressure events. Returns 0 on
// success, -1 on failure.
@_cdecl("gemma_vision_init")
public func gemma_vision_init(_ safetensorsPath: UnsafePointer<CChar>?) -> Int32 {
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
    return gVisionResidency != nil ? 1 : 0
}

// Current residency state: 0=unloaded, 1=volatile, 2=pinned, -1=not bound.
@_cdecl("gemma_vision_residency_state")
public func gemma_vision_residency_state() -> Int32 {
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
    return UInt64(gVisionResidency?.workingSetBytes() ?? 0)
}

// Manually flip to volatile (signal "we're idle, OS take this if needed").
// Auto-called by the pressure source on .warning.
@_cdecl("gemma_vision_allow_evict")
public func gemma_vision_allow_evict() -> Int32 {
    gVisionResidency?.allowEvict()
    return 0
}

// Manually drop the working set entirely (simulate a .critical pressure
// event from tests; real pressure source calls this automatically).
@_cdecl("gemma_vision_force_drop")
public func gemma_vision_force_drop() -> Int32 {
    gVisionResidency?.forceDrop()
    return 0
}

// Preprocess + vision-tower + submit soft tokens to a session, bracketed
// by BOI (255999) / EOI (258882) markers the way Gemma-4 expects for an
// inline image in a user turn. Image bytes (PNG / JPEG / HEIC / whatever
// CGImageSource can decode) are passed directly from the caller — no
// filesystem round-trip.
//
// Cache-aware: SHA-256 of the raw bytes keys a padded soft-token buffer;
// on hit, reuses it and skips preprocess + vision entirely. The same
// buffer can back concurrent sessions safely since Session.submit(soft-
// Tokens:) treats the buffer as read-only. LRU-evicts when
// gVisionCacheMaxEntries is exceeded.
//
// Returns number of soft tokens submitted (280 on success), or -1 on error.
// gemma_submit_image_bytes (per-session) deleted 2026-04-26 — replaced
// by Segment(kind=1, image_bytes=...) on the unified gemma_submit path.
// The flow it ran (hash → ensureCachedSofts → submit BOI/softs/EOI) is
// preserved verbatim in ffi_batch.swift's submitImageSegment helper.

// Pre-warm the cache without attaching to a session. Intended for
// POST /v1/images/prewarm: an agent can pre-populate entries it expects
// to re-reference later so the first "real" request sees a hit.
//
// "Prewarmed" means the entry exists and a vision-event ticket is
// allocated. The pad-blit CB may still be running when this returns —
// the LM consumer's pre-prefill CB encodeWaitForEvent's the ticket and
// the GPU itself waits if needed. Per notes/engine_debloat.md: the HTTP
// handler does not block on GPU work; the consumer encodes the wait.
//
// Returns soft-token count on success (cache hit or miss), -1 on error.
@_cdecl("gemma_vision_prewarm_bytes")
public func gemma_vision_prewarm_bytes(_ ptr: UnsafePointer<UInt8>?,
                                         _ byteCount: Int32) -> Int32 {
    guard let p = ptr, byteCount > 0 else { return -1 }
    let data = Data(bytes: p, count: Int(byteCount))
    // No ffiLock — ensureCachedSofts has its own fine-grained locking and
    // runs preprocess in parallel across concurrent callers.
    guard let padded = ensureCachedSofts(data: data) else { return -1 }
    return Int32(padded.count)
}

// Fills out_hex (65 bytes minimum, includes null terminator) with the
// SHA-256 hex of the last prewarm/submit. Useful for clients that want
// to know the cache key.
@_cdecl("gemma_vision_last_cache_key")
public func gemma_vision_last_cache_key(_ outHex: UnsafeMutablePointer<CChar>?,
                                         _ maxBytes: Int32) -> Int32 {
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
    // CPU-CPU read API: caller is asking for bytes back. The pad-blit CB
    // may still be in flight on the vision queue; wait on the
    // shared-event ticket so the bytes are coherent before memcpy.
    // This is a legitimate CPU sync point — "give me the bytes" is a
    // synchronous semantic by definition.
    if entry.eventTicket > 0 {
        _ = gVisionEvent.wait(untilSignaledValue: entry.eventTicket, timeoutMS: 30_000)
    }
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
    let ids = gCvecRegistry.keys.sorted().joined(separator: ",")
    let bytes = Array(ids.utf8)
    if outPtr == nil { return Int32(bytes.count) }
    let n = min(bytes.count, Int(maxBytes))
    for i in 0..<n { outPtr![i] = CChar(bitPattern: bytes[i]) }
    return Int32(n)
}
// Read back an already-registered fp16 cvec. Clients that interpolate
// directions (heretic's fractional direction_index) or feed a registered
// direction to the heretic per-write ablation need the raw bytes. Writes
// HIDDEN fp16 halves into outPtr and returns bytes written, or -1 if the
// id isn't registered. Passing outPtr=nil returns the needed byte count.
@_cdecl("gemma_control_get_fp16")
public func gemma_control_get_fp16(_ idPtr: UnsafePointer<CChar>?,
                                     _ outPtr: UnsafeMutablePointer<UInt8>?,
                                     _ maxBytes: Int32) -> Int32 {
    let need = HIDDEN * 2
    guard let p = outPtr else { return Int32(need) }
    guard let ip = idPtr else { return -1 }
    let id = String(cString: ip)
    guard let buf = gCvecRegistry[id] else { return -1 }
    let n = min(need, Int(maxBytes))
    memcpy(p, buf.contents(), n)
    return Int32(n)
}
// Pairwise prose cvec constructor: set the layer whose residual should
// be blitted into the capture buffer after each tick's post-FFN write.
// Pass -1 to disable. Active for as long as it's set — caller is
// responsible for resetting after it's done capturing.
@_cdecl("gemma_set_capture_layer")
public func gemma_set_capture_layer(_ layer: Int32) -> Int32 {
    gResidualCaptureLayer = Int(layer)
    return 0
}

// All-layer capture toggle. When enabled, every layer's post-FFN
// residual gets blitted into its L-indexed slot of gAllLayerCaptureBuf
// during each tick. Meant to be on briefly during screening passes
// and off the rest of the time (tiny but not zero per-layer blit cost).
@_cdecl("gemma_set_capture_all_layers")
public func gemma_set_capture_all_layers(_ enabled: Int32) -> Int32 {
    gCaptureAllLayers = (enabled != 0)
    return 0
}

// Copy the NUM_LAYERS × HIDDEN × fp16 all-layer capture buffer for
// slot 0 into the caller's output. Returns bytes written, or if outPtr
// is nil, the needed size. Since the underlying buffer is now B-wide
// per layer, this gathers L-strided slot-0 strips.
@_cdecl("gemma_get_all_layer_residuals")
public func gemma_get_all_layer_residuals(_ outPtr: UnsafeMutablePointer<UInt8>?,
                                           _ maxBytes: Int32) -> Int32 {
    return gemma_get_all_slot_layer_residuals(0, outPtr, maxBytes)
}

// Copy NUM_LAYERS × HIDDEN × fp16 for a specific batch slot out of the
// B-wide all-layer capture buffer. Layout in the buffer is [L][slot][v]
// so per-slot extraction is a non-contiguous gather over L strides —
// 30 memcpys of 5.6 KB each on Gemma-4-A4B, negligible cost.
@_cdecl("gemma_get_all_slot_layer_residuals")
public func gemma_get_all_slot_layer_residuals(_ slot: Int32,
                                                 _ outPtr: UnsafeMutablePointer<UInt8>?,
                                                 _ maxBytes: Int32) -> Int32 {
    let need = NUM_LAYERS * HIDDEN * 2
    guard let p = outPtr else { return Int32(need) }
    let s = Int(slot)
    if s < 0 || s >= B { return -1 }
    let n = min(need, Int(maxBytes))
    let src = gAllLayerCaptureBuf.contents()
    let layerStride = B * HIDDEN * 2
    let slotOffset = s * HIDDEN * 2
    var written = 0
    let perLayerBytes = HIDDEN * 2
    for L in 0..<NUM_LAYERS {
        if written + perLayerBytes > n { break }
        memcpy(p.advanced(by: written),
                src.advanced(by: L * layerStride + slotOffset),
                perLayerBytes)
        written += perLayerBytes
    }
    return Int32(written)
}

// Per-layer Q/K/V capture toggle. When enabled, after each layer's
// q_norm+RoPE, k_norm+RoPE, and v_norm_noscale, the slot-0 per-head
// Q/K/V tensors get blitted into gQ/K/VCaptureBuf at layer-indexed
// slots. Used by the synthetic-KV fitting pipeline offline; cost is
// near-zero when disabled.
@_cdecl("gemma_set_capture_qkv")
public func gemma_set_capture_qkv(_ enabled: Int32) -> Int32 {
    gCaptureQKV = (enabled != 0)
    return 0
}

// Readback for the Q capture buffer. Layout is
// [NUM_LAYERS, MAX_Q_HEADS * MAX_HD] fp16 halves, conservatively sized
// so each per-layer slice has enough room for either SLIDE (H=16,
// HD=256) or FULL (H=16, HD=512) layers. Only the first (H_L × HD_L)
// halves per layer slice are valid; clients consult
// gemma_get_layer_head_shape to split correctly.
@_cdecl("gemma_get_captured_q")
public func gemma_get_captured_q(_ outPtr: UnsafeMutablePointer<UInt8>?,
                                  _ maxBytes: Int32) -> Int32 {
    let need = NUM_LAYERS * MAX_Q_HEADS * MAX_HD * 2
    guard let p = outPtr else { return Int32(need) }
    let n = min(need, Int(maxBytes))
    memcpy(p, gQCaptureBuf.contents(), n)
    return Int32(n)
}

@_cdecl("gemma_get_captured_k")
public func gemma_get_captured_k(_ outPtr: UnsafeMutablePointer<UInt8>?,
                                  _ maxBytes: Int32) -> Int32 {
    let need = NUM_LAYERS * MAX_KV_HEADS * MAX_HD * 2
    guard let p = outPtr else { return Int32(need) }
    let n = min(need, Int(maxBytes))
    memcpy(p, gKCaptureBuf.contents(), n)
    return Int32(n)
}

@_cdecl("gemma_get_captured_v")
public func gemma_get_captured_v(_ outPtr: UnsafeMutablePointer<UInt8>?,
                                  _ maxBytes: Int32) -> Int32 {
    let need = NUM_LAYERS * MAX_KV_HEADS * MAX_HD * 2
    guard let p = outPtr else { return Int32(need) }
    let n = min(need, Int(maxBytes))
    memcpy(p, gVCaptureBuf.contents(), n)
    return Int32(n)
}

// Return per-layer attention shape info so clients can slice the
// Q/K/V capture buffers correctly. Writes NUM_LAYERS × 4 int32 values
// as (num_q_heads, num_kv_heads, head_dim, is_full). Also returns the
// per-slice STRIDE as MAX_Q_HEADS * MAX_HD (for Q) and MAX_KV_HEADS *
// MAX_HD (for K and V) — clients use the stride for offset arithmetic
// and the per-layer (H, KV_H, HD) for valid-element count.
@_cdecl("gemma_get_layer_attn_shapes")
public func gemma_get_layer_attn_shapes(_ outPtr: UnsafeMutablePointer<Int32>?,
                                         _ maxInts: Int32) -> Int32 {
    let need = NUM_LAYERS * 4
    guard let p = outPtr else { return Int32(need) }
    guard let engine = gEngine else { return -1 }
    let n = min(need, Int(maxInts))
    var idx = 0
    for L in 0..<NUM_LAYERS {
        if idx + 4 > n { break }
        let lw = engine.weights.layers[L]
        let H: Int = lw.isFull ? FULL_H : SLIDE_H
        p[idx + 0] = Int32(H)
        p[idx + 1] = Int32(lw.KV_H)
        p[idx + 2] = Int32(lw.HD)
        p[idx + 3] = Int32(lw.isFull ? 1 : 0)
        idx += 4
    }
    return Int32(idx)
}

// Strides for slicing the capture buffers. Q stride is MAX_Q_HEADS *
// MAX_HD halves; K/V stride is MAX_KV_HEADS * MAX_HD halves.
@_cdecl("gemma_get_qkv_capture_strides")
public func gemma_get_qkv_capture_strides(_ outQStride: UnsafeMutablePointer<Int32>?,
                                           _ outKVStride: UnsafeMutablePointer<Int32>?) -> Int32 {
    if let p = outQStride { p.pointee = Int32(MAX_Q_HEADS * MAX_HD) }
    if let p = outKVStride { p.pointee = Int32(MAX_KV_HEADS * MAX_HD) }
    return 0
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
    let need = HIDDEN * 2
    guard let p = outPtr else { return Int32(need) }
    let n = min(need, Int(maxBytes))
    memcpy(p, gResidualCaptureBuf.contents(), n)
    return Int32(n)
}
// Fine-grained locks for the image-submit critical path. ffiLock is NOT
// held during preprocess / vision CB submission — text-only sessions'
// tick() threads proceed concurrently while vision work runs.
//
// - gVisionCacheLock: guards the cache dict + tick counter + hit/miss
//   counters + gLastCacheKeyHex. Held only for microseconds at a time
//   (dict reads/writes).
// - gResidencyLock: serializes residency state transitions (ensurePinned,
//   forceDrop). The first caller pays the hydrate cost (~10 ms GPU); all
//   subsequent callers fast-path through the .pinned check.
// - gVisionBatchLock: guards the pending-batch queue + dispatcher-active
//   flag. The first concurrent miss becomes the batch "leader" — waits a
//   short window for peers' preprocess to complete, then dispatches one
//   runVisionTowerBatchForwardAsync call covering all pending images.
//
// Residency is kept pinned indefinitely after first use — on an M5 Max
// the ~1.5 GB working set is a trivial budget, and the prior allowEvict
// defer raced with concurrent callers' in-flight vision CBs. Pressure-
// source forceDrop() still runs if the OS actually needs the pages back.
private let gVisionCacheLock = NSLock()
private let gResidencyLock = NSLock()
private let gVisionBatchLock = NSLock()
private var gLastCacheKeyHex: String?

// Vision-batch dispatcher: completion-driven gather + non-blocking submit
// (see notes/engine_debloat.md).
//
// HTTP submit pre-allocates the padded buffer + reserves an
// MTLSharedEvent ticket and installs the CachedSofts entry in the cache
// IMMEDIATELY. Then it enqueues a `VisionBatchPending` carrying that
// entry and returns to the caller. No semaphore. No leader. No poll.
//
// The vision dispatcher's completion handler then runs:
//   - For each pending item, encode a pad-blit CB that fills the
//     pre-allocated buffer from the raw vision-tower output.
//   - The pad-blit CB encodeSignalEvent's the reserved ticket.
//   - LM consumers' `encodeWaitForEvent(ticket)` (already wired in
//     prepareSinglePrefill / prepareMultiSlotSoftPrefill) gates the
//     downstream prefill on the pad-blit completing — entirely on GPU,
//     CPU never blocks.
//
// Cache hits return the existing entry instantly. The caller's chunk
// queue carries the entry's eventTicket; the LM prefill encodes a
// GPU-side wait that resolves immediately if the ticket is already
// signaled past, or waits for the in-flight pad-blit otherwise.
private struct VisionBatchPending {
    let batch: PatchBatch
    let entry: CachedSofts   // pre-allocated buffer + reserved ticket
}

private var gVisionBatchPending: [VisionBatchPending] = []
// In-flight flag: set true while a vision CB is committed and running
// (or while the completion handler is running). New submitters who find
// it true append to the queue and skip the kick — the in-flight CB's
// completion handler will drain them. Cleared inside the handler when
// the post-CB queue drain finds nothing left to do.
private var gVisionBatchInFlight: Bool = false

private func _installCachedSofts(hashData: Data, entry: CachedSofts) {
    gVisionCacheLock.lock()
    gVisionCache[hashData] = entry
    if gVisionCache.count > gVisionCacheMaxEntries {
        if let (oldKey, _) = gVisionCache.min(by: { $0.value.lastUsed < $1.value.lastUsed }) {
            gVisionCache.removeValue(forKey: oldKey)
        }
    }
    gVisionCacheLock.unlock()
}

// Drain the pending-batch queue into one runVisionTowerBatchForwardAsync
// call. Self-perpetuating: the vision CB's completion handler re-calls
// this to drain anything that arrived during the GPU work. Caller must
// hold the gVisionBatchInFlight=true invariant: either as the original
// kicker (set just before calling) or by virtue of being inside the
// completion handler chain (in-flight remains true across the handler).
//
// On entry: gVisionBatchInFlight is true. On exit: either dispatched a
// new vision CB (handler re-calls us) or cleared in-flight (queue empty).
private func _kickVisionDispatch(weights: VisionWeights) {
    // Atomically drain whatever is currently queued. If empty, clear the
    // in-flight flag inside the same critical section so we don't race
    // with a new submitter who would otherwise see in-flight=true and
    // not kick — that would leak a never-dispatched item.
    gVisionBatchLock.lock()
    let items = gVisionBatchPending
    gVisionBatchPending.removeAll(keepingCapacity: true)
    if items.isEmpty {
        gVisionBatchInFlight = false
        gVisionBatchLock.unlock()
        return
    }
    gVisionBatchLock.unlock()

    let vq = gVisionQueue ?? queue
    let batches = items.map { $0.batch }
    if ProcessInfo.processInfo.environment["VISION_BATCH_DEBUG"] != nil {
        FileHandle.standardError.write("[vision-batch] dispatching B=\(batches.count)\n".data(using: .utf8)!)
    }
    let (batchResults, visionCB) = runVisionTowerBatchForwardAsync(
        batches: batches, weights: weights, device: device, queue: vq)

    // Vision CB completion handler: encode per-item pad-blit CBs into
    // the pre-allocated entry buffers, signal each item's pre-reserved
    // event ticket, then re-kick to drain anything that arrived during
    // this CB. No CPU sync — LM consumers wait on event tickets via
    // GPU-side encodeWaitForEvent.
    visionCB.addCompletedHandler { _ in
        // One pad-blit CB for the whole batch: K blits + K signal-events
        // in one CB instead of K separate CBs (each ~100µs of Metal
        // overhead). Items run sequentially within this CB on the vision
        // queue; signals fire in encode order, so the event's signaledValue
        // passes through each ticket monotonically — LM consumers waiting
        // on any ticket get woken when that one is reached.
        let padCB = vq.makeCommandBuffer()!
        let blit = padCB.makeBlitCommandEncoder()!
        for (i, item) in items.enumerated() {
            let (rawSoftTokens, rawNPooled) = batchResults[i]
            if rawNPooled > 0 {
                let copyRows = min(rawNPooled, item.entry.count)
                blit.copy(from: rawSoftTokens, sourceOffset: 0,
                          to: item.entry.buffer, destinationOffset: 0,
                          size: copyRows * HIDDEN * 4)
            } else {
                FileHandle.standardError.write(
                    "[vision] image extraction returned 0 softs (failure)\n".data(using: .utf8)!)
            }
        }
        blit.endEncoding()
        // Signal each item's ticket at the same point in the CB timeline
        // (after the blits). Even failed items get signaled so LM
        // consumers don't deadlock; their buffers remain zero-init.
        for item in items {
            padCB.encodeSignalEvent(gVisionEvent, value: item.entry.eventTicket)
        }
        padCB.commit()
        _kickVisionDispatch(weights: weights)
    }
    visionCB.commit()
}

internal func ensureCachedSofts(data: Data) -> CachedSofts? {
    // Hash outside any lock — SHA-256 on ~10–200 KB is microseconds and
    // has no shared state.
    let hash = SHA256.hash(data: data)
    let hashData = Data(hash)
    let hashHex = hash.map { String(format: "%02x", $0) }.joined()

    // Fast path: cache hit. Brief critical section on the dict only.
    gVisionCacheLock.lock()
    gVisionCacheTick &+= 1
    gLastCacheKeyHex = hashHex
    if var hit = gVisionCache[hashData] {
        hit.lastUsed = gVisionCacheTick
        gVisionCache[hashData] = hit
        gVisionCacheHits &+= 1
        gVisionCacheLock.unlock()
        return hit
    }
    gVisionCacheMisses &+= 1
    let tickAtMiss = gVisionCacheTick
    gVisionCacheLock.unlock()

    // Miss: hydrate residency (first-call only; subsequent .pinned checks
    // are ~nanoseconds). Separate lock so the residency transition doesn't
    // block cache dict reads.
    gResidencyLock.lock()
    guard let residency = gVisionResidency else {
        gResidencyLock.unlock()
        print("ensureCachedSofts: vision not initialized")
        return nil
    }
    do {
        try residency.ensurePinned()
    } catch {
        gResidencyLock.unlock()
        print("ensureCachedSofts: vision hydrate failed: \(error)")
        return nil
    }
    guard let visWeights = residency.weights else {
        gResidencyLock.unlock()
        print("ensureCachedSofts: residency returned nil weights")
        return nil
    }
    gResidencyLock.unlock()

    // Preprocess: pure CPU, no shared state. Runs in parallel across
    // concurrent callers on M5 Max's plentiful cores.
    let batch: PatchBatch
    do { batch = try gemma4ImagePreprocess(data: data, device: device) }
    catch {
        print("ensureCachedSofts: preprocess failed: \(error)")
        return nil
    }

    // Pre-allocate the padded buffer + reserve an event ticket up front,
    // so the cache entry exists IMMEDIATELY (deduplication: a second
    // submit of the same hash hits the cache and shares the in-flight
    // ticket). The vision dispatcher's completion handler will fill the
    // buffer + signal the ticket. No semaphore, no CPU wait — LM
    // consumers gate downstream work via GPU-side encodeWaitForEvent.
    let targetSoft = 280
    guard let padded = device.makeBuffer(length: targetSoft * HIDDEN * 4,
                                          options: .storageModeShared)
    else {
        print("ensureCachedSofts: makeBuffer failed (out of memory?)")
        return nil
    }
    memset(padded.contents(), 0, padded.length)
    let ticket = nextVisionEventTicket()
    let entry = CachedSofts(buffer: padded, count: targetSoft,
                             lastUsed: tickAtMiss, bytes: padded.length,
                             eventTicket: ticket)
    _installCachedSofts(hashData: hashData, entry: entry)

    let pending = VisionBatchPending(batch: batch, entry: entry)
    let shouldKick: Bool
    gVisionBatchLock.lock()
    gVisionBatchPending.append(pending)
    shouldKick = !gVisionBatchInFlight
    if shouldKick { gVisionBatchInFlight = true }
    gVisionBatchLock.unlock()
    if shouldKick {
        _kickVisionDispatch(weights: visWeights)
    }
    return entry
}

@_cdecl("gemma_vision_cache_entries")
public func gemma_vision_cache_entries() -> Int32 {
    return Int32(gVisionCache.count)
}

@_cdecl("gemma_vision_cache_hits")
public func gemma_vision_cache_hits() -> UInt64 {
    return gVisionCacheHits
}

@_cdecl("gemma_vision_cache_misses")
public func gemma_vision_cache_misses() -> UInt64 {
    return gVisionCacheMisses
}

@_cdecl("gemma_vision_cache_bytes")
public func gemma_vision_cache_bytes() -> UInt64 {
    return gVisionCache.values.reduce(0) { $0 + UInt64($1.bytes) }
}

@_cdecl("gemma_vision_cache_clear")
public func gemma_vision_cache_clear() -> Int32 {
    let n = gVisionCache.count
    gVisionCache.removeAll()
    return Int32(n)
}

@_cdecl("gemma_max_q_len")
public func gemma_max_q_len() -> Int32 {
    return Int32(MAX_Q_LEN)
}
