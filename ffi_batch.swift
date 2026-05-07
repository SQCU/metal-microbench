// ffi_batch.swift — production-runtime FFI implementing the batch-shaped ABI
// from notes/specs/batch_ffi_abi.md.
//
// This is the small, correct surface the bridge calls: gemma_submit_batch
// enqueues a wave of stream actions; gemma_poll drives the engine forward
// and returns whatever progress happened. Status / init / shutdown are
// admin entries.
//
// All wire formats are little-endian binary, hand-rolled. Magic numbers
// match the ABI doc: 'GEMB' for requests, 'GEMR' for responses.
//
// Scope:
//   - Segment kinds: tokens + image_bytes both supported.
//   - Cache stats (cache_hits, cache_misses, vision_cache_hits) plumbed.
//   - Admin entries: gemma_status, gemma_shutdown.
//   - capture_logits payload: sampled_token_id, sampled_logprob, optional
//     top-K logprobs. Captured at tick time (per-poll loop iteration)
//     before the logits buffer is overwritten by the next CB.

import Foundation

// ----------------------------------------------------------------------
// Stream/session id mapping. The ABI says stream_id is a u64 the bridge
// owns; internally we use Session.id (Int). Keep both directions.
// ----------------------------------------------------------------------
private var gStreamToSession: [UInt64: Session] = [:]
private var gSessionToStream: [Int: UInt64] = [:]

// ----------------------------------------------------------------------
// Work-conserving FFI: intake queue + engine drive lock.
//
// gemma_submit pushes DecodedStream items onto gIntakeQueue (cheap, no
// engine-state mutation) and signals gIntakeCond. gemma_poll holds
// gEngineLock for the duration of its drive loop; at the top of each
// iteration it briefly takes gIntakeCond to drain the intake queue
// into engine state. This means submit and poll can run on different
// threadpool workers in parallel — submit pushes intake while poll is
// mid-tick, poll picks it up at the next iteration's drain step.
//
// The deadline argument to gemma_poll now ONLY gates the idle-wait
// (when engine has no work AND intake is empty); it never interrupts
// active driving. This is the work-conserving invariant: a stream's
// prefill chunks run back-to-back at engine speed without forced
// roundtrips through the bridge.
// ----------------------------------------------------------------------
private let gIntakeCond = NSCondition()    // protects gIntakeQueue + idle wait
private var gIntakeQueue: [DecodedStream] = []
// 2026-05-06: lock removal attempted, reverted after bisect — the
// bridge wedged after admission burst when gemma_poll ran without
// gEngineLock held across the drive loop. Some path in the engine
// state transitions / page-allocator interaction relies on the
// implicit serialization the coarse lock provided. The lock-removal
// refactor needs more careful work than a one-line delete; restoring
// for now so the rest of the changes (page allocator no-pre-reserve +
// pool size bump) can land cleanly. TODO: split into per-domain
// locks (engine state vs page manager) and add the instrumentation
// codex specified before another removal attempt.
private let gEngineLock = NSLock()

// ----------------------------------------------------------------------
// Binary readers — little-endian, alignment-safe via load(as:).
// ----------------------------------------------------------------------
@inline(__always)
private func readU8(_ ptr: UnsafePointer<UInt8>, _ off: Int) -> UInt8 {
    return ptr.advanced(by: off).pointee
}
@inline(__always)
private func readU16(_ ptr: UnsafePointer<UInt8>, _ off: Int) -> UInt16 {
    return UnsafeRawPointer(ptr.advanced(by: off)).loadUnaligned(as: UInt16.self)
}
@inline(__always)
private func readU32(_ ptr: UnsafePointer<UInt8>, _ off: Int) -> UInt32 {
    return UnsafeRawPointer(ptr.advanced(by: off)).loadUnaligned(as: UInt32.self)
}
@inline(__always)
private func readU64(_ ptr: UnsafePointer<UInt8>, _ off: Int) -> UInt64 {
    return UnsafeRawPointer(ptr.advanced(by: off)).loadUnaligned(as: UInt64.self)
}
@inline(__always)
private func readI32(_ ptr: UnsafePointer<UInt8>, _ off: Int) -> Int32 {
    return UnsafeRawPointer(ptr.advanced(by: off)).loadUnaligned(as: Int32.self)
}
@inline(__always)
private func readF32(_ ptr: UnsafePointer<UInt8>, _ off: Int) -> Float {
    return UnsafeRawPointer(ptr.advanced(by: off)).loadUnaligned(as: Float.self)
}

// ----------------------------------------------------------------------
// Binary writer — appends to a growing Data buffer.
// ----------------------------------------------------------------------
private struct BinWriter {
    var data: Data = Data()
    mutating func u8(_ v: UInt8)   { var x = v; data.append(Data(bytes: &x, count: 1)) }
    mutating func u16(_ v: UInt16) { var x = v; data.append(Data(bytes: &x, count: 2)) }
    mutating func u32(_ v: UInt32) { var x = v; data.append(Data(bytes: &x, count: 4)) }
    mutating func u64(_ v: UInt64) { var x = v; data.append(Data(bytes: &x, count: 8)) }
    mutating func i32(_ v: Int32)  { var x = v; data.append(Data(bytes: &x, count: 4)) }
    mutating func f32(_ v: Float)  { var x = v; data.append(Data(bytes: &x, count: 4)) }
    mutating func bytes(_ b: [UInt8]) { data.append(contentsOf: b) }
    mutating func zeros(_ n: Int) { data.append(contentsOf: [UInt8](repeating: 0, count: n)) }
    var count: Int { data.count }
    // Patch a u32 already written at `at`. Used to backfill heap_offset etc.
    mutating func patchU32(at: Int, _ v: UInt32) {
        var x = v
        data.replaceSubrange(at..<(at+4),
                              with: Data(bytes: &x, count: 4))
    }
}

// ----------------------------------------------------------------------
// Decoded request shape. Kept simple — caller converts directly into
// engine actions, no intermediate optimization.
// ----------------------------------------------------------------------
private struct DecodedSampling {
    var temperature: Float
    var topP: Float
    var topK: UInt32
    var repetitionPenalty: Float
    var maxNewTokens: UInt32
    var seed: UInt64
    var eosTokenId: Int32
    var stopTokens: [UInt32]
    var topLogprobs: UInt32
    var logitBias: [(UInt32, Float)]
    var minP: Float
    var cotLabels: [String]
    // Multi-token stop sequences. After the engine emits a token, if
    // the recent emitted tail equals any of these sequences, the
    // stream finishes with done_reason=1 (EOS-equivalent).
    var stopSequences: [[UInt32]]
}

private struct DecodedCV {
    var cvecId: String
    var layer: Int32
    var polarity: Float
    var peakMagnitude: Float
    var attack: Float
    var decay: Float
    var sustainLevel: Float
    var release: Float
    var shape: UInt8
    var units: UInt8
    var mode: UInt8
    var target: Float
    var transportScale: Float
    var transportOffset: Float
}

private struct DecodedSegment {
    var kind: UInt8           // 0=tokens, 1=image_bytes
    var tokens: [UInt32]      // populated when kind==0
    var imageBytes: Data      // populated when kind==1
}

private struct DecodedStream {
    var streamId: UInt64
    var action: UInt8         // 0=start, 1=continue, 2=cancel, 3=touch
    var flags: UInt8
    var segments: [DecodedSegment]
    var sampling: DecodedSampling
    var controlVectors: [DecodedCV]
}

private enum DecodeError: Error {
    case badMagic, badVersion, truncated, badAction, badSegmentKind
}

// Decode a wire-format BatchRequest. Bounds-checks at every read.
private func decodeBatchRequest(_ buf: UnsafePointer<UInt8>, _ len: Int) throws -> [DecodedStream] {
    guard len >= 16 else { throw DecodeError.truncated }
    let magic = readU32(buf, 0)
    guard magic == 0x424D4547 else { throw DecodeError.badMagic } // 'GEMB' little-endian
    let version = readU32(buf, 4)
    guard version == 4 else { throw DecodeError.badVersion }
    let streamCount = Int(readU32(buf, 8))
    let heapOffset = Int(readU32(buf, 12))
    let streamSpecsBase = 16
    // v4: SamplingParams=72 bytes (now includes ssq_count + ssq_offset
    // where v3 had 8 reserved bytes); StreamSpec=104 (cv_count + cvs_offset
    // unchanged). See notes/specs/batch_ffi_abi.md.
    let streamSpecBytes = 104
    let streamSpecsBytes = streamCount * streamSpecBytes
    guard streamSpecsBase + streamSpecsBytes <= len else { throw DecodeError.truncated }
    guard heapOffset >= streamSpecsBase + streamSpecsBytes,
          heapOffset <= len else { throw DecodeError.truncated }

    var out: [DecodedStream] = []
    out.reserveCapacity(streamCount)

    for i in 0..<streamCount {
        let base = streamSpecsBase + i * streamSpecBytes
        let sid = readU64(buf, base + 0)
        let action = readU8(buf, base + 8)
        let flags = readU8(buf, base + 9)
        // bytes 10-11 reserved
        let segCount = Int(readU32(buf, base + 12))
        let segOff = Int(readU32(buf, base + 16))
        // bytes 20-23 reserved
        // SamplingParams at base+24 .. base+72 (now 48 bytes; spec was 40 prior).
        let temp = readF32(buf, base + 24)
        let topP = readF32(buf, base + 28)
        let topK = readU32(buf, base + 32)
        let repPen = readF32(buf, base + 36)
        let maxNew = readU32(buf, base + 40)
        let seed = readU64(buf, base + 44)
        let eosId = readI32(buf, base + 52)
        let stopCount = Int(readU32(buf, base + 56))
        let stopOff = Int(readU32(buf, base + 60))
        let topLog = readU32(buf, base + 64)
        let lbCount = Int(readU32(buf, base + 68))
        let lbOff = Int(readU32(buf, base + 72))
        let minP = readF32(buf, base + 76)
        let cotCount = Int(readU32(buf, base + 80))
        let cotOff = Int(readU32(buf, base + 84))
        // v4: bytes 88-95 are now stop_sequences count + offset. (v3
        // had these as 8 reserved zero bytes; the version bump prevents
        // mis-decoding an old client that left them zero.)
        let ssqCount = Int(readU32(buf, base + 88))
        let ssqOff = Int(readU32(buf, base + 92))
        let cvCount = Int(readU32(buf, base + 96))
        let cvsOff = Int(readU32(buf, base + 100))
        // StreamSpec ends at 104.

        var lbEntries: [(UInt32, Float)] = []
        if lbCount > 0 {
            guard lbOff + lbCount * 8 <= len else { throw DecodeError.truncated }
            lbEntries.reserveCapacity(lbCount)
            for k in 0..<lbCount {
                let tid = readU32(buf, lbOff + k * 8)
                let val = readF32(buf, lbOff + k * 8 + 4)
                lbEntries.append((tid, val))
            }
        }

        var cotLabels: [String] = []
        if cotCount > 0 {
            var cur = cotOff
            for _ in 0..<cotCount {
                guard cur + 4 <= len else { throw DecodeError.truncated }
                let bc = Int(readU32(buf, cur))
                cur += 4
                guard cur + bc <= len else { throw DecodeError.truncated }
                let raw = UnsafeBufferPointer(
                    start: buf.advanced(by: cur), count: bc)
                // Direct String init from UnsafeBufferPointer — skips
                // the Array intermediate copy that the previous
                // `String(bytes: Array(raw), ...)` paid.
                if let s = String(bytes: raw, encoding: .utf8) {
                    cotLabels.append(s)
                }
                cur += bc
            }
        }

        var cvs: [DecodedCV] = []
        if cvCount > 0 {
            guard cvsOff + cvCount * 64 <= len else { throw DecodeError.truncated }
            for k in 0..<cvCount {
                let off = cvsOff + k * 64
                let idOff = Int(readU32(buf, off + 0))
                let idBc  = Int(readU32(buf, off + 4))
                guard idOff + idBc <= len else { throw DecodeError.truncated }
                let idRaw = UnsafeBufferPointer(
                    start: buf.advanced(by: idOff), count: idBc)
                // Direct UnsafeBufferPointer init — no Array intermediate.
                let id = String(bytes: idRaw, encoding: .utf8) ?? ""
                cvs.append(DecodedCV(
                    cvecId: id,
                    layer: Int32(bitPattern: readU32(buf, off + 8)),
                    polarity: readF32(buf, off + 12),
                    peakMagnitude: readF32(buf, off + 16),
                    attack: readF32(buf, off + 20),
                    decay: readF32(buf, off + 24),
                    sustainLevel: readF32(buf, off + 28),
                    release: readF32(buf, off + 32),
                    shape: readU8(buf, off + 36),
                    units: readU8(buf, off + 37),
                    mode:  readU8(buf, off + 38),
                    target: readF32(buf, off + 40),
                    transportScale: readF32(buf, off + 44),
                    transportOffset: readF32(buf, off + 48)))
            }
        }

        var stops: [UInt32] = []
        if stopCount > 0 {
            guard stopOff + stopCount * 4 <= len else { throw DecodeError.truncated }
            stops.reserveCapacity(stopCount)
            for k in 0..<stopCount {
                stops.append(readU32(buf, stopOff + k * 4))
            }
        }

        // Multi-token stop sequences: ssq_count back-to-back records,
        // each [u32 length][u32 tok0]...[u32 tokN-1]. ssq_off points to
        // the first length word.
        var stopSequences: [[UInt32]] = []
        if ssqCount > 0 {
            stopSequences.reserveCapacity(ssqCount)
            var cur = ssqOff
            for _ in 0..<ssqCount {
                guard cur + 4 <= len else { throw DecodeError.truncated }
                let seqLen = Int(readU32(buf, cur))
                cur += 4
                guard cur + seqLen * 4 <= len else { throw DecodeError.truncated }
                var seq: [UInt32] = []
                seq.reserveCapacity(seqLen)
                for k in 0..<seqLen {
                    seq.append(readU32(buf, cur + k * 4))
                }
                cur += seqLen * 4
                stopSequences.append(seq)
            }
        }

        var segs: [DecodedSegment] = []
        if segCount > 0 {
            let segArrBytes = segCount * 16
            guard segOff + segArrBytes <= len else { throw DecodeError.truncated }
            for k in 0..<segCount {
                let so = segOff + k * 16
                let kind = readU8(buf, so + 0)
                // bytes 1-3 reserved
                let count = Int(readU32(buf, so + 4))
                let dataOff = Int(readU32(buf, so + 8))
                // bytes 12-15 reserved
                switch kind {
                case 0: // tokens
                    guard dataOff + count * 4 <= len else { throw DecodeError.truncated }
                    var toks: [UInt32] = []
                    toks.reserveCapacity(count)
                    for j in 0..<count { toks.append(readU32(buf, dataOff + j * 4)) }
                    segs.append(DecodedSegment(kind: 0, tokens: toks, imageBytes: Data()))
                case 1: // image_bytes
                    guard dataOff + count <= len else { throw DecodeError.truncated }
                    let raw = UnsafeBufferPointer(start: buf.advanced(by: dataOff),
                                                   count: count)
                    let imgBytes = Data(buffer: raw)
                    segs.append(DecodedSegment(kind: 1, tokens: [], imageBytes: imgBytes))
                default:
                    throw DecodeError.badSegmentKind
                }
            }
        }

        out.append(DecodedStream(
            streamId: sid, action: action, flags: flags, segments: segs,
            sampling: DecodedSampling(
                temperature: temp, topP: topP, topK: topK,
                repetitionPenalty: repPen, maxNewTokens: maxNew,
                seed: seed, eosTokenId: eosId, stopTokens: stops,
                topLogprobs: topLog, logitBias: lbEntries,
                minP: minP, cotLabels: cotLabels,
                stopSequences: stopSequences),
            controlVectors: cvs))
    }
    return out
}

// ----------------------------------------------------------------------
// Per-stream usage counters. Cache stats are mostly derived from Session's
// own accumulators (cacheHitTokens / cacheMissTokens); vision stats need
// to be incremented at submit time around ensureCachedSofts.
// ----------------------------------------------------------------------
private struct StreamUsage {
    var promptTokensSeen: UInt32 = 0
    var completionTokensEmitted: UInt32 = 0
    var visionCacheHits: UInt32 = 0
}
private var gUsage: [UInt64: StreamUsage] = [:]

// Logprobs capture state. Sets/maps keyed by stream_id.
//   gCaptureLogits — stream_ids with capture_logits flag set
//   gTopLogprobs   — per-stream top-K count (0 = sampled-token only)
//   gLastOutCount  — last-seen Session.outputQueue length, so we can detect
//                    newly-emitted tokens after each tick
//   gLogprobsQ     — per-stream queue of (token, sampled_logprob, top_k_pairs)
//                    parallel to outputQueue; drained when tokens are.
private var gCaptureLogits: Set<UInt64> = []
private var gTopLogprobs: [UInt64: UInt32] = [:]
private var gLastOutCount: [UInt64: Int] = [:]
private struct LogprobRecord {
    let token: UInt32
    let sampledLogprob: Float
    let topKPairs: [(UInt32, Float)]
}
private var gLogprobsQ: [UInt64: [LogprobRecord]] = [:]

// In-batch shared-prefix deferral — see notes/specs/bandwidth_triage.md §2.
// When gemma_submit_batch arrives with N streams whose first PAGE_SLIDE
// tokens hash to the same value, only one (the leader) actually opens
// a session and prefills. Followers are stashed here, indexed by the
// shared first-page hash they're waiting on. On every poll_batch tick,
// we re-check the cache: once the leader has promoted its first page,
// followers apply with the standard adoptSharedPrefixPages path picking
// up the prefix automatically. This is what makes 4 simultaneous
// rollouts of the same prompt cost 1× prefill instead of 4×.
private struct DeferredStart {
    let spec: DecodedStream
    let firstPageHash: UInt64
    let leaderStreamId: UInt64   // who we're waiting on
}
private var gDeferred: [UInt64: DeferredStart] = [:]   // stream_id → deferred
// Leader assignment per first-page-hash within a batch. Used by deferred
// followers to know which session's prefill to wait for before applying.
private var gBatchLeaders: [UInt64: UInt64] = [:]      // hash → leader sid

// Streams whose state=2 (.done) update has already been appended to the
// current poll's updates buffer. Prevents duplicate done-emissions when
// the poll loop drives multiple CBs and re-iterates gStreamToSession on
// each. Reset at the top of every gemma_poll call.
private var gDoneEmittedThisPoll: Set<UInt64> = []

// ----------------------------------------------------------------------
// Compute the first-page hash for a stream's start segments. Returns 0
// (= "no hash") if the stream doesn't have ≥ PAGE_SLIDE leading text
// tokens. For first-cut, image-leading streams skip the deferral path
// — the multimodal curriculum's per-rollout system-text prefix is the
// case we care about, and that's always text-leading.
// ----------------------------------------------------------------------
private func firstPageHashOf(_ stream: DecodedStream) -> UInt64 {
    var leading: [UInt32] = []
    for seg in stream.segments {
        guard seg.kind == 0 else { break }   // image_bytes — stop accumulating
        leading.append(contentsOf: seg.tokens)
        if leading.count >= PAGE_SLIDE { break }
    }
    guard leading.count >= PAGE_SLIDE else { return 0 }
    return PageManager.hashPage(leading[0..<PAGE_SLIDE], cvecDigest: 0)
}

// ----------------------------------------------------------------------
// Apply a single decoded stream action to the engine.
// ----------------------------------------------------------------------
private func applyStreamAction(_ stream: DecodedStream, engine: LmEngine) {
    let sid = stream.streamId
    switch stream.action {
    case 0: // start
        guard gStreamToSession[sid] == nil else {
            print("  [batch_ffi] start on already-live stream_id \(sid); ignored")
            return
        }
        let maxNew = Int(stream.sampling.maxNewTokens > 0
                          ? stream.sampling.maxNewTokens : 512)
        let eosId: UInt32? = stream.sampling.eosTokenId >= 0
            ? UInt32(stream.sampling.eosTokenId) : nil
        guard let s = engine.openSession(eosId: eosId, maxNewTokens: maxNew) else {
            print("  [batch_ffi] openSession failed for stream_id \(sid)")
            return
        }
        s.samplingTemperature = stream.sampling.temperature
        gStreamToSession[sid] = s
        gSessionToStream[s.id] = sid
        gUsage[sid] = StreamUsage()
        if (stream.flags & 0x01) != 0 {
            gCaptureLogits.insert(sid)
            gTopLogprobs[sid] = stream.sampling.topLogprobs
            gLogprobsQ[sid] = []
            gLastOutCount[sid] = 0
        }
        // Sampler-side per-stream features.
        applyLogitBias(s, entries: stream.sampling.logitBias)
        applyMinP(s, minP: stream.sampling.minP)
        applyStructuredCot(s, labels: stream.sampling.cotLabels)
        s.stopSequences = stream.sampling.stopSequences
        // Forward-pass-side: control vectors.
        applyControlVectors(s, cvs: stream.controlVectors)
        // Submit segments. Each tokens-segment goes through s.submit; each
        // image_bytes segment goes through the vision tower (with caching)
        // then submits BOI/softTokens/EOI. Order is preserved: text and
        // image segments interleave per the original message.
        for seg in stream.segments {
            switch seg.kind {
            case 0:
                s.submit(seg.tokens)
                gUsage[sid]?.promptTokensSeen &+= UInt32(seg.tokens.count)
            case 1:
                submitImageSegment(s, sid: sid, imageBytes: seg.imageBytes)
            default:
                break
            }
        }
    case 1: // continue
        guard let s = gStreamToSession[sid] else {
            print("  [batch_ffi] continue on unknown stream_id \(sid); ignored")
            return
        }
        for seg in stream.segments {
            switch seg.kind {
            case 0:
                s.submit(seg.tokens)
                gUsage[sid]?.promptTokensSeen &+= UInt32(seg.tokens.count)
            case 1:
                submitImageSegment(s, sid: sid, imageBytes: seg.imageBytes)
            default:
                break
            }
        }
        // Re-apply sampling/control params on continue (clients may
        // adjust mid-conv). All these are no-ops when the stream's
        // values are unchanged from the prior submit.
        s.samplingTemperature = stream.sampling.temperature
        applyLogitBias(s, entries: stream.sampling.logitBias)
        applyMinP(s, minP: stream.sampling.minP)
        applyStructuredCot(s, labels: stream.sampling.cotLabels)
        s.stopSequences = stream.sampling.stopSequences
        applyControlVectors(s, cvs: stream.controlVectors)
    case 2: // cancel
        // A cancel might arrive for a deferred (not-yet-applied) stream
        // too. Drop it from gDeferred either way.
        gDeferred.removeValue(forKey: sid)
        guard let s = gStreamToSession[sid] else { return }
        engine.closeSession(s)
        gStreamToSession.removeValue(forKey: sid)
        gSessionToStream.removeValue(forKey: s.id)
    case 3: // touch — re-apply all sampling/control state without new tokens
        guard let s = gStreamToSession[sid] else { return }
        s.samplingTemperature = stream.sampling.temperature
        applyLogitBias(s, entries: stream.sampling.logitBias)
        applyMinP(s, minP: stream.sampling.minP)
        applyStructuredCot(s, labels: stream.sampling.cotLabels)
        s.stopSequences = stream.sampling.stopSequences
        applyControlVectors(s, cvs: stream.controlVectors)
    default:
        print("  [batch_ffi] unknown action \(stream.action) for stream_id \(sid)")
    }
}

// ----------------------------------------------------------------------
// Logprob capture for one stream's just-emitted token. Reads the slot's
// fp16 logits row from the global `logits` buffer, computes log-softmax,
// records sampled_logprob + optional top-K. Must be called AFTER
// syncTickStep and BEFORE the next syncTickStep (which overwrites the
// buffer). Cost is ~2 ms per call at VOCAB=262144 — acceptable for the
// RL/distillation path that opted into it via capture_logits.
// ----------------------------------------------------------------------
private func captureLogprobForLatestToken(_ sid: UInt64,
                                           _ s: Session,
                                           topK: UInt32) {
    guard let slot = s.slot else { return }
    // Total tokens this stream has produced ever = drained-already-counted
    // (in completionTokensEmitted) + still-pending in outputQueue.
    let prev = gLastOutCount[sid] ?? 0
    let curr = Int(gUsage[sid]?.completionTokensEmitted ?? 0) + s.pendingOutputCount
    if curr <= prev { return }

    // Read the logits row: B × VOCAB fp16 buffer; this slot starts at
    // slot * VOCAB. Capture the post-softmax distribution.
    let p = logits.contents().assumingMemoryBound(to: Float16.self)
    let row = UnsafeBufferPointer(
        start: p.advanced(by: slot * VOCAB), count: VOCAB)

    // log-softmax in two passes (max + sum-exp). Stable.
    var maxLogit: Float = -.infinity
    for i in 0..<VOCAB {
        let v = Float(row[i])
        if v > maxLogit { maxLogit = v }
    }
    var sumExp: Double = 0
    for i in 0..<VOCAB {
        sumExp += Double(exp(Float(row[i]) - maxLogit))
    }
    let logZ = maxLogit + Float(log(sumExp))

    // Sampled token(s): the most recently appended to outputQueue. In
    // simple AR there's one per tick.
    let recents = s.peekRecentOutputs(count: curr - prev)
    for token in recents {
        let sampledLogprob = Float(row[Int(token)]) - logZ

        var topPairs: [(UInt32, Float)] = []
        if topK > 0 {
            let k = Int(min(topK, 50))   // ABI cap
            // Build a min-heap of size k via simple replace-min.
            // For VOCAB=262144 and k=20, this is ~5 ms — acceptable.
            var heap: [(Int, Float)] = []
            heap.reserveCapacity(k)
            var heapMin: Float = .infinity
            for i in 0..<VOCAB {
                let lp = Float(row[i]) - logZ
                if heap.count < k {
                    heap.append((i, lp))
                    if heap.count == k {
                        heapMin = heap.min { $0.1 < $1.1 }!.1
                    }
                } else if lp > heapMin {
                    let idx = heap.firstIndex(where: { $0.1 == heapMin })!
                    heap[idx] = (i, lp)
                    heapMin = heap.min { $0.1 < $1.1 }!.1
                }
            }
            heap.sort { $0.1 > $1.1 }
            topPairs = heap.map { (UInt32($0.0), $0.1) }
        }

        gLogprobsQ[sid, default: []].append(LogprobRecord(
            token: token,
            sampledLogprob: sampledLogprob,
            topKPairs: topPairs))
    }
    gLastOutCount[sid] = curr
}

// ----------------------------------------------------------------------
// Sampler-side apply paths. All called from applyStreamAction on start /
// continue / touch — empty input is the no-op path so streams that
// don't use these features pay zero cost.
// ----------------------------------------------------------------------
private func applyLogitBias(_ s: Session, entries: [(UInt32, Float)]) {
    if entries.isEmpty {
        s.logitBiasDense = nil
        return
    }
    var dense = [Float](repeating: 0, count: VOCAB)
    for (tok, bias) in entries {
        let i = Int(tok)
        if i >= 0 && i < VOCAB { dense[i] = bias }
    }
    s.logitBiasDense = dense
}

private func applyMinP(_ s: Session, minP: Float) {
    s.minP = max(0, min(1, minP))
}

// Sampler-side: structured-cot grammar. Empty labels = no constraint
// (clears the existing cot state if any).
private func applyStructuredCot(_ s: Session, labels: [String]) {
    if labels.isEmpty {
        s.cot = nil
    } else {
        s.enableStructuredCot(labels: labels)
    }
}

// ----------------------------------------------------------------------
// Forward-pass-side apply path: control vectors. Each DecodedCV refers
// to a registered cvec id (uploaded via gemma_register_resource). We
// resolve the buffer, build a CvecEnvelope, and call s.addControl —
// the existing kernel hook in buildStepCB picks it up.
// ----------------------------------------------------------------------
private func applyControlVectors(_ s: Session, cvs: [DecodedCV]) {
    // Replace any existing controls with the freshly-specified set —
    // start/continue both re-apply, matching how the legacy bridge
    // re-attached on each chat turn.
    s.clearControls()
    for cv in cvs {
        guard let buf = gCvecRegistry[cv.cvecId] else {
            print("  [batch_ffi] cvec '\(cv.cvecId)' not registered; skipping")
            continue
        }
        let shape: CvecShape
        switch cv.shape {
        case 0: shape = .linear
        case 1: shape = .expIn
        case 2: shape = .expOut
        case 3: shape = .cubic
        default: shape = .linear
        }
        let units: CvecUnits = (cv.units == 1) ? .turns : .tokens
        let mode: CvecMode
        switch cv.mode {
        case 1: mode = .project
        case 2: mode = .transport
        default: mode = .additive
        }
        let env = CvecEnvelope(
            attack: cv.attack, decay: cv.decay,
            sustainLevel: cv.sustainLevel, release: cv.release,
            peakMagnitude: cv.peakMagnitude,
            shape: shape, units: units)
        let target: Float? = cv.target.isNaN ? nil : cv.target
        let ctrl = ActiveControl(
            cvecId: cv.cvecId, buffer: buf, layer: Int(cv.layer),
            envelope: env, polarity: cv.polarity,
            startPosition: s.positionForDebug,
            startTurn: s.turnIndex,
            mode: mode, target: target,
            transportScale: cv.transportScale,
            transportOffset: cv.transportOffset)
        s.addControl(ctrl)
    }
}

// ----------------------------------------------------------------------
// Admin entry: register a named resource (control vector today; LoRA /
// adapter / etc. reserved). Replaces the legacy gemma_control_register_fp16.
// kind="cvec" expects HIDDEN × 2 bytes (fp16).
// ----------------------------------------------------------------------
@_cdecl("gemma_register_resource")
public func gemma_register_resource(_ kindPtr: UnsafePointer<CChar>?,
                                     _ idPtr: UnsafePointer<CChar>?,
                                     _ bytes: UnsafePointer<UInt8>?,
                                     _ nBytes: Int32) -> Int32 {
    guard let kp = kindPtr, let ip = idPtr, let bp = bytes else { return -1 }
    let kind = String(cString: kp)
    let id = String(cString: ip)
    if id.isEmpty { return -1 }
    switch kind {
    case "cvec":
        let expected = HIDDEN * 2
        guard Int(nBytes) == expected else {
            print("[register_resource] cvec '\(id)': size mismatch (got \(nBytes), expected \(expected))")
            return -1
        }
        guard let buf = device.makeBuffer(length: expected, options: .storageModeShared) else {
            return -1
        }
        memcpy(buf.contents(), bp, expected)
        gCvecRegistry[id] = buf
        return 0
    default:
        print("[register_resource] unknown kind '\(kind)'")
        return -1
    }
}

// ----------------------------------------------------------------------
// Image segment submission: hash bytes, ensure cached softs (vision tower
// runs on miss), submit BOI / softTokens / EOI to the session. Mirrors
// the per-session FFI's gemma_session_submit_image_bytes path.
// ----------------------------------------------------------------------
private func submitImageSegment(_ s: Session, sid: UInt64, imageBytes: Data) {
    let hitsBefore = gVisionCacheHits
    guard let padded = ensureCachedSofts(data: imageBytes) else {
        print("  [batch_ffi] ensureCachedSofts failed for stream \(sid)")
        return
    }
    let hitDelta = gVisionCacheHits - hitsBefore
    if hitDelta > 0 {
        gUsage[sid]?.visionCacheHits &+= UInt32(hitDelta)
    }
    // Stable content hash of input bytes — same image in two streams produces
    // identical placeholder positions in the page-cache hash, so prefix
    // sharing across streams works as long as preceding text is identical.
    var imgHash: UInt64 = 0xcbf29ce484222325
    for byte in imageBytes {
        imgHash ^= UInt64(byte)
        imgHash = imgHash &* 0x100000001b3
    }
    let BOI: UInt32 = 255999
    let EOI: UInt32 = 258882
    s.submit([BOI])
    s.submit(softTokens: padded.buffer, count: padded.count, isFp32: true,
             eventTicket: padded.eventTicket, contentHash: imgHash)
    s.submit([EOI])
    // Account image-soft-tokens as prompt tokens for billing parity with
    // text. The 2 (BOI + EOI) are negligible.
    gUsage[sid]?.promptTokensSeen &+= UInt32(padded.count + 2)
}

// ----------------------------------------------------------------------
// Encode a BatchResponse from current engine state. For each live stream:
// emit a StreamUpdate carrying any pending output tokens and current state.
// Returns the serialized bytes.
//
// Streams with no progress since last poll are omitted to keep the response
// compact. The bridge re-derives state for those streams as "no change."
// ----------------------------------------------------------------------
private struct StreamUpdateOut {
    var streamId: UInt64
    var state: UInt8
    var doneReason: UInt8
    var newTokens: [UInt32]
    var errMsg: String
    var promptTokensSeen: UInt32
    var completionTokensEmitted: UInt32
    var cacheHits: UInt32
    var cacheMisses: UInt32
    var visionCacheHits: UInt32
    var logprobs: [LogprobRecord]   // empty if capture_logits not set
}

private func encodeBatchResponse(_ updates: [StreamUpdateOut]) -> Data {
    var w = BinWriter()
    w.u32(0x52454D47) // 'GEMR' (little-endian: 0x47 0x4D 0x45 0x52 → "GEMR")
    w.u32(1) // version
    w.u32(UInt32(updates.count))
    let heapOffsetSlot = w.count
    w.u32(0) // patched after we know array end

    // Reserve fixed-size StreamUpdate slots; record offsets to backfill heap.
    let updateBase = w.count
    w.zeros(updates.count * 64)

    // Heap starts now.
    let heapStart = w.count
    w.patchU32(at: heapOffsetSlot, UInt32(heapStart))

    for (i, u) in updates.enumerated() {
        let off = updateBase + i * 64
        var fix = BinWriter()
        fix.u64(u.streamId)
        fix.u8(u.state)
        fix.u8(u.doneReason)
        fix.u16(0)
        fix.u32(UInt32(u.newTokens.count))
        let newToksOffSlot = fix.count
        fix.u32(0)
        fix.u32(UInt32(u.errMsg.utf8.count))
        let errOffSlot = fix.count
        fix.u32(0)
        fix.u32(u.promptTokensSeen)
        fix.u32(u.completionTokensEmitted)
        fix.u32(u.cacheHits)
        fix.u32(u.cacheMisses)
        fix.u32(u.visionCacheHits)
        let lpByteSlot = fix.count
        fix.u32(0)
        let lpOffSlot = fix.count
        fix.u32(0)
        fix.zeros(8)

        let newToksHeapOff = w.count
        for t in u.newTokens { w.u32(t) }
        let errMsgHeapOff = w.count
        if !u.errMsg.isEmpty {
            w.bytes(Array(u.errMsg.utf8))
        }

        // Logprobs payload — per ABI: 12 + 8*top_count bytes per emitted
        // token, packed sequentially. Byte count covers the full payload.
        let lpHeapOff = w.count
        var lpBytes = 0
        for rec in u.logprobs {
            w.u32(rec.token)
            w.f32(rec.sampledLogprob)
            w.u32(UInt32(rec.topKPairs.count))
            lpBytes += 12
            for (tid, lp) in rec.topKPairs {
                w.u32(tid)
                w.f32(lp)
                lpBytes += 8
            }
        }

        var fixData = fix.data
        var nt = UInt32(newToksHeapOff)
        fixData.replaceSubrange(newToksOffSlot..<(newToksOffSlot+4),
                                with: Data(bytes: &nt, count: 4))
        var em = UInt32(errMsgHeapOff)
        fixData.replaceSubrange(errOffSlot..<(errOffSlot+4),
                                with: Data(bytes: &em, count: 4))
        var lb = UInt32(lpBytes)
        fixData.replaceSubrange(lpByteSlot..<(lpByteSlot+4),
                                with: Data(bytes: &lb, count: 4))
        var lo = UInt32(lpHeapOff)
        fixData.replaceSubrange(lpOffSlot..<(lpOffSlot+4),
                                with: Data(bytes: &lo, count: 4))

        w.data.replaceSubrange(off..<(off+64), with: fixData)
    }

    return w.data
}

// ----------------------------------------------------------------------
// Drain a session's pending tokens. Updates usage counters.
//
// 2026-05-07: pre-size with reserveCapacity to avoid Swift Array geometric
// regrowth on append. Per-stream per-poll allocation; with 8 streams ×
// ~12 polls/sec that's 96 small array allocs/sec — small individually
// but adds up. The reserveCapacity hint at output-token-budget bound
// prevents the first append from realloc'ing.
// ----------------------------------------------------------------------
private func drainSession(_ s: Session, sid: UInt64) -> [UInt32] {
    var out: [UInt32] = []
    out.reserveCapacity(8)   // typical per-poll drain is 1-8 tokens
    while let t = s.nextToken() {
        out.append(t)
    }
    if !out.isEmpty {
        gUsage[sid]?.completionTokensEmitted &+= UInt32(out.count)
    }
    return out
}

private func sessionStateByte(_ state: SessionState) -> UInt8 {
    switch state {
    case .idle:       return 0
    case .priming:    return 0
    case .generating: return 1
    case .paused:     return 0
    case .done:       return 2
    }
}

// Best-effort done_reason. Engine doesn't yet expose explicit reason;
// approximate from numGenerated and EOS comparison via context. For now
// return 1 (eos) when state is done — the bridge can refine when the
// engine grows a real done_reason field.
private func sessionDoneReason(_ s: Session) -> UInt8 {
    return s.state == .done ? 1 : 0
}

// ----------------------------------------------------------------------
// FFI exports.
// ----------------------------------------------------------------------

@_cdecl("gemma_submit")
public func gemma_submit(_ buf: UnsafePointer<UInt8>?,
                          _ len: Int32) -> Int32 {
    // Non-driving: decode the wire format (no engine state mutation),
    // push DecodedStream items onto the intake queue, signal
    // gIntakeCond. Returns immediately. The next call into gemma_poll
    // (or any thread already inside one) drains the intake at the top
    // of its drive loop and applies stream actions to engine state
    // under gEngineLock.
    //
    // This separates submission concurrency from driving concurrency:
    // bridge can call gemma_submit on threadpool worker A while
    // worker B is mid-poll driving the engine — they only contend on
    // the brief intake-lock acquisition during drain.
    guard gEngine != nil else { return -1 }
    guard let buf = buf, len > 0 else { return -2 }
    let decoded: [DecodedStream]
    do {
        decoded = try decodeBatchRequest(buf, Int(len))
    } catch {
        print("  [batch_ffi] decodeBatchRequest failed: \(error)")
        return -3
    }
    gIntakeCond.lock()
    gIntakeQueue.append(contentsOf: decoded)
    gIntakeCond.signal()
    gIntakeCond.unlock()
    return 0
}

// Drain currently-pending intake items into engine state. Caller MUST
// hold gEngineLock. Briefly takes gIntakeCond's lock to pop the
// queue, then releases it before applying stream actions (which can
// take a while — hashing, page-manager lookups, etc.).
private func drainIntakeIntoEngine(_ engine: LmEngine) {
    gIntakeCond.lock()
    let items = gIntakeQueue
    gIntakeQueue.removeAll(keepingCapacity: true)
    gIntakeCond.unlock()
    if items.isEmpty { return }

    let dbg = ProcessInfo.processInfo.environment["LM_BATCH_DEBUG"] != nil
    // 2026-05-07: shared-prefix follower deferral is now opt-in via
    // LM_BATCH_SHARED_PREFIX_DEFERRAL=1. The sched_sim_token simulation
    // (D2 break-even surface) showed it only wins for prompt_len>=512
    // with p_shared>=50% (agent-clique sampling). For typical live
    // serving (independent prompts, p_shared≈0), it costs 0-2% by
    // serializing followers behind leaders that could have prefilled
    // in parallel. Default OFF — every stream admits as its own
    // session and prefills its own KV pages.
    let useFollowerDeferral = (ProcessInfo.processInfo.environment["LM_BATCH_SHARED_PREFIX_DEFERRAL"] == "1")
    for stream in items {
        if useFollowerDeferral && stream.action == 0 { // start
            let h = firstPageHashOf(stream)
            if h != 0 && engine.pageManager.findByHash(h) == nil {
                if let leaderSid = gBatchLeaders[h],
                   gStreamToSession[leaderSid] != nil {
                    gDeferred[stream.streamId] = DeferredStart(
                        spec: stream, firstPageHash: h,
                        leaderStreamId: leaderSid)
                    if dbg {
                        print("  [batch] stream \(stream.streamId) deferred behind leader \(leaderSid) (hash=\(String(h, radix: 16)))")
                    }
                    continue
                }
                gBatchLeaders[h] = stream.streamId
                if dbg {
                    print("  [batch] stream \(stream.streamId) leader (hash=\(String(h, radix: 16)))")
                }
            }
        }
        applyStreamAction(stream, engine: engine)
    }
}

@_cdecl("gemma_poll")
public func gemma_poll(_ timeoutMs: Int32,
                       _ outBuf: UnsafeMutablePointer<UInt8>?,
                       _ outCap: Int32) -> Int32 {
    guard let engine = gEngine else { return -1 }
    guard let outBuf = outBuf, outCap > 0 else { return -2 }

    // Drive contract (work-conserving):
    //   1. drain intake into engine state (each iteration)
    //   2. retry deferred starts (each iteration; only matters when
    //      LM_BATCH_SHARED_PREFIX_DEFERRAL=1 is set — otherwise gDeferred
    //      stays empty)
    //   3. if engine has work: syncTickStep() — runs one prefill chunk
    //      OR one AR step
    //   4. collect updates from streams that emitted tokens; APPEND to
    //      the per-poll updates buffer (NOT a new local snapshot —
    //      see 2026-05-07 D3 fix below)
    //   5. (DELETED in 2026-05-07): the old drive contract returned
    //      ASAP on the first productive CB, "to surface tokens fast".
    //      The sched_sim_token simulation showed this costs ~1-9%
    //      aggregate tok/s by introducing an ~8 ms Python-side host
    //      gap between every productive CB. We now drive multiple
    //      CBs within one poll call, accumulating tokens, and only
    //      return when (a) no work remains, or (b) the deadline is hit.
    //   6. if more work to do AND deadline not exceeded: loop back
    //   7. if engine is idle AND intake is empty: cond_wait on
    //      gIntakeCond up to remaining timeout, then return
    //
    // Result: prefill chunks AND AR ticks run back-to-back at engine
    // speed inside a single poll call. The bridge does NOT need to
    // round-trip per chunk OR per produced token. New submits arriving
    // via gemma_submit on a different thread are picked up at the next
    // iteration's drain step (microsecond latency).
    gEngineLock.lock()
    defer { gEngineLock.unlock() }

    let deadline = Date().addingTimeInterval(Double(max(0, timeoutMs)) / 1000.0)
    var updates: [StreamUpdateOut] = []
    gDoneEmittedThisPoll.removeAll(keepingCapacity: true)

    while true {
        // 1. Drain new submissions into engine state.
        drainIntakeIntoEngine(engine)

        // 2. Retry deferred starts (in-batch shared-prefix follower
        //    rule: a follower waits until its leader's session is past
        //    prefill, then applies). gDeferred is empty in default
        //    operation (the LM_BATCH_SHARED_PREFIX_DEFERRAL gate is
        //    off by default per the 2026-05-07 D2 falsification).
        //    2026-05-07: snapshot the keys once into a stack-local
        //    array only when gDeferred has entries; the previous
        //    `Array(gDeferred.keys)` allocation ran every poll
        //    iteration regardless of whether the dict had entries.
        if !gDeferred.isEmpty {
            let dbg = ProcessInfo.processInfo.environment["LM_BATCH_DEBUG"] != nil
            // Collect keys to apply in this pass; can't iterate the
            // dict while mutating it (removeValue), so snapshot only
            // the ready-to-apply keys instead of the full key set.
            var readyKeys: [UInt64] = []
            readyKeys.reserveCapacity(gDeferred.count)
            for (sid, pending) in gDeferred {
                let leaderReady: Bool
                if let leader = gStreamToSession[pending.leaderStreamId] {
                    leaderReady = (leader.state == .generating || leader.state == .done)
                } else {
                    leaderReady = true
                }
                if leaderReady { readyKeys.append(sid) }
            }
            for sid in readyKeys {
                guard let pending = gDeferred[sid] else { continue }
                if dbg {
                    print("  [batch] stream \(sid) follower applying — leader \(pending.leaderStreamId) ready")
                }
                applyStreamAction(pending.spec, engine: engine)
                gDeferred.removeValue(forKey: sid)
            }
        }

        // 3. Drive one chunk if there's work.
        let hadWork = engine.hasWork
        if hadWork {
            engine.syncTickStep()
            for sid in gCaptureLogits {
                guard let s = gStreamToSession[sid] else { continue }
                captureLogprobForLatestToken(sid, s, topK: gTopLogprobs[sid] ?? 0)
            }
        }

        // 4. Collect updates from streams. APPEND to the per-poll
        //    `updates` buffer — see drive-contract note above. Tracks
        //    `doneAlreadyEmitted` so we don't re-emit a state=2 update
        //    on subsequent iterations after a stream finishes inside
        //    this poll. Cleanup of finished streams happens once at
        //    poll end, AFTER all driving is done.
        for (sid, s) in gStreamToSession {
            // If this stream finished and we already emitted its
            // state=2 update on a prior iteration of this poll, skip.
            // (gStreamToSession only loses it during the post-loop
            // cleanup pass, so still iterating over it here is fine.)
            if gDoneEmittedThisPoll.contains(sid) { continue }
            let newToks = drainSession(s, sid: sid)
            let stateByte = sessionStateByte(s.state)
            let doneByte = sessionDoneReason(s)
            if !newToks.isEmpty || s.state == .done {
                let usage = gUsage[sid] ?? StreamUsage()
                var lps: [LogprobRecord] = []
                if gCaptureLogits.contains(sid) {
                    let avail = gLogprobsQ[sid] ?? []
                    let n = min(avail.count, newToks.count)
                    lps = Array(avail.prefix(n))
                    if n > 0 {
                        gLogprobsQ[sid] = Array(avail.dropFirst(n))
                    }
                }
                updates.append(StreamUpdateOut(
                    streamId: sid,
                    state: stateByte,
                    doneReason: doneByte,
                    newTokens: newToks,
                    errMsg: "",
                    promptTokensSeen: usage.promptTokensSeen,
                    completionTokensEmitted: usage.completionTokensEmitted,
                    cacheHits: s.cacheHitTokens,
                    cacheMisses: s.cacheMissTokens,
                    visionCacheHits: usage.visionCacheHits,
                    logprobs: lps))
                if s.state == .done {
                    gDoneEmittedThisPoll.insert(sid)
                }
            }
        }

        // 5. (DELETED 2026-05-07): the old early-return-on-first-update
        //    introduced an ~8 ms Python round-trip gap per productive
        //    CB. Now we keep driving while there's work AND time.

        // 6. More work to do AND deadline not exceeded → keep driving.
        if hadWork && Date() < deadline {
            continue
        }
        // hadWork but deadline hit: return what we have.
        if hadWork && Date() >= deadline {
            break
        }

        // 7. Engine is idle. If we already have updates to deliver,
        //    return them immediately rather than waiting on intake —
        //    no point sleeping when there are tokens for the bridge.
        if !updates.isEmpty {
            break
        }
        // No updates yet. Check intake one more time before waiting.
        gIntakeCond.lock()
        let intakePending = !gIntakeQueue.isEmpty
        gIntakeCond.unlock()
        if intakePending {
            continue
        }

        // Truly idle. Wait on gIntakeCond up to remaining deadline,
        // then exit. The bridge will call gemma_poll again if it
        // wants more.
        let remaining = deadline.timeIntervalSinceNow
        if remaining <= 0 { break }

        gIntakeCond.lock()
        if gIntakeQueue.isEmpty {
            // NSCondition.wait(until:) returns false on timeout.
            // Release gEngineLock while waiting — admin FFI calls
            // (gemma_status etc.) can slip in here. Re-acquire after.
            gEngineLock.unlock()
            _ = gIntakeCond.wait(until: Date().addingTimeInterval(remaining))
            gIntakeCond.unlock()
            gEngineLock.lock()
        } else {
            gIntakeCond.unlock()
        }
        // Loop back; drain whatever arrived (or nothing, on timeout).
        // Outer-loop deadline check below catches the timeout case.
        if Date() >= deadline {
            break
        }
    }

    // Clean up any sessions that finished.
    for u in updates where u.state == 2 {
        if let s = gStreamToSession[u.streamId] {
            engine.closeSession(s)
            gSessionToStream.removeValue(forKey: s.id)
        }
        gStreamToSession.removeValue(forKey: u.streamId)
        gUsage.removeValue(forKey: u.streamId)
        gCaptureLogits.remove(u.streamId)
        gTopLogprobs.removeValue(forKey: u.streamId)
        gLastOutCount.removeValue(forKey: u.streamId)
        gLogprobsQ.removeValue(forKey: u.streamId)
    }

    let resp = encodeBatchResponse(updates)
    if resp.count > Int(outCap) {
        return -28 // -ENOSPC
    }
    resp.withUnsafeBytes { src in
        outBuf.update(from: src.baseAddress!.assumingMemoryBound(to: UInt8.self),
                       count: resp.count)
    }
    return Int32(resp.count)
}

// ----------------------------------------------------------------------
// Admin entries.
// ----------------------------------------------------------------------

// gemma_status: write a serialized ServerStats blob (ABI: 64 bytes).
@_cdecl("gemma_status")
public func gemma_status(_ outBuf: UnsafeMutablePointer<UInt8>?,
                          _ outCap: Int32) -> Int32 {
    guard let outBuf = outBuf, outCap >= 64 else { return -28 }
    var w = BinWriter()
    if let engine = gEngine {
        let stats = engine.pageManager.stats()
        var generating = 0
        var priming = 0
        for s in gStreamToSession.values {
            switch s.state {
            case .generating: generating += 1
            case .priming:    priming += 1
            default:          break
            }
        }
        w.u32(UInt32(stats.totalPages))
        w.u32(UInt32(stats.freePages))
        w.u32(UInt32(stats.cachedHashes))
        w.u32(UInt32(gStreamToSession.count))
        w.u32(UInt32(generating))
        w.u32(UInt32(priming))
        w.u64(UInt64(engine.totalSteps))
        w.u64(UInt64(engine.totalTokensGenerated))
        w.u32(UInt32(gVisionCacheEntryCount))
        w.zeros(4) // padding to align u64
        w.u64(gVisionCacheHits)
        w.zeros(8) // reserved
    } else {
        // Engine not initialized — return all-zero stats but valid frame.
        w.zeros(64)
    }
    let bytes = w.data
    bytes.withUnsafeBytes { src in
        outBuf.update(from: src.baseAddress!.assumingMemoryBound(to: UInt8.self),
                       count: bytes.count)
    }
    return Int32(bytes.count)
}

// gemma_shutdown: close all live streams, drop the engine, free state.
// Idempotent: safe to call multiple times.
@_cdecl("gemma_shutdown")
public func gemma_shutdown() -> Int32 {
    if let engine = gEngine {
        for (_, s) in gStreamToSession {
            engine.closeSession(s)
        }
    }
    gStreamToSession.removeAll()
    gSessionToStream.removeAll()
    gUsage.removeAll()
    gEngine = nil
    return 0
}
