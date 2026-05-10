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
// 2026-05-07 (anonymous-pool refactor): the old gStreamToSession +
// gSessionToStream double-dictionary is gone. Lookup by stream_id is
// now `engine.requestForStream[sid]` — the engine owns the index, the
// bridge owns nothing about the request between calls. Same goes for
// per-stream policy/usage/logprob state, which moved onto Session
// (see Session.streamId / .usage / .logprobsQueue / etc. in
// lm_engine.swift). The bridge layer is now a marshaling layer with
// no long-lived state.
// ----------------------------------------------------------------------

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

// 2026-05-07: gUsage / gCaptureLogits / gTopLogprobs / gLastOutCount /
// gLogprobsQ have been deleted. The state they held now lives on
// Session (see Session.usage / .captureLogits / .topLogprobs /
// .lastReportedOutputCount / .logprobsQueue). StreamUsage and
// LogprobRecord type definitions moved to lm_engine.swift since
// they're now Session fields. Iteration over "streams with the
// capture-logits flag" is now `engine.requestForStream.values
// where $0.captureLogits`.

// 2026-05-07: deleted in-batch shared-prefix follower deferral.
// Per the NO REMOTE LOCKS principle, gating stream B's submission on
// stream A's prefill progress is a cross-stream coupling — even though
// hash-keyed, it makes one client's work order depend on another's
// timing. The engine-side content-hash KV page cache (page_manager
// contentIndex) already accelerates B's prefill mid-stream when A's
// pages are present, with no waiting and no client coupling: B's
// prefill kernel runs to completion either way; engine just hands B
// cached pages where it can. Follower deferral was a pre-cache-aware
// optimization that became redundant once the page cache landed, and
// was strict-violation territory under the principle.
//
// Empirical sched_sim_token D2 sweep + production probe agreed: the
// follower-deferral path only won at p_shared >= 50% with prompt
// length >= 512 — i.e., the agent-clique sampling regime. Even then
// only modestly (3-11%) and only after paying a 1-CB silencing cost
// for the leader's single-slot prefill. For typical eval/serving
// workloads (independent prompts), it lost 0-2%.
//
// Removed: DeferredStart struct, gDeferred dict, gBatchLeaders dict,
// firstPageHashOf helper, and the LM_BATCH_SHARED_PREFIX_DEFERRAL
// gate logic. The gemma_poll deferred-retry loop is also stripped
// (was unused with gDeferred always empty).

// Streams whose state=2 (.done) update has already been appended to the
// current poll's updates buffer. Prevents duplicate done-emissions when
// the poll loop drives multiple CBs and re-iterates gStreamToSession on
// each. Reset at the top of every gemma_poll call.
private var gDoneEmittedThisPoll: Set<UInt64> = []

// ----------------------------------------------------------------------
// Apply a single decoded stream action to the engine.
// ----------------------------------------------------------------------
private func applyStreamAction(_ stream: DecodedStream, engine: LmEngine) {
    let sid = stream.streamId
    switch stream.action {
    case 0: // start — atomic creation via engine.submitRequest
        let maxNew = Int(stream.sampling.maxNewTokens > 0
                          ? stream.sampling.maxNewTokens : 512)
        let eosId: UInt32? = stream.sampling.eosTokenId >= 0
            ? UInt32(stream.sampling.eosTokenId) : nil

        // Build RequestInit from the wire payload. All per-request
        // configuration is set ONCE at birth — no setter dance.
        var initParams = LmEngine.RequestInit()
        initParams.eosId = eosId
        initParams.maxNewTokens = maxNew
        initParams.samplingTemperature = stream.sampling.temperature
        initParams.samplingSeed = stream.sampling.seed
        if (stream.flags & 0x01) != 0 {
            initParams.captureLogits = true
            initParams.topLogprobs = stream.sampling.topLogprobs
        }
        initParams.logitBiasDense = denseLogitBias(stream.sampling.logitBias)
        initParams.minP = max(0, min(1, stream.sampling.minP))
        initParams.stopSequences = stream.sampling.stopSequences
        initParams.cot = cotStateForLabels(stream.sampling.cotLabels)
        initParams.controls = controlsFromDecoded(stream.controlVectors,
                                                   startPosition: 0,
                                                   startTurn: 0)

        // Translate wire segments to engine-typed segments. Image
        // segments stay tagged as raw-bytes; the imageSubmit closure
        // resolves them through the vision tower at submit time.
        let segs: [LmEngine.InitialSegment] = stream.segments.compactMap { seg in
            switch seg.kind {
            case 0: return .tokens(seg.tokens)
            case 1: return .image(seg.imageBytes)
            default: return nil
            }
        }
        _ = engine.submitRequest(streamId: sid, init: initParams,
                                  segments: segs,
                                  imageSubmit: { s, bytes in
                                      submitImageSegment(s, imageBytes: bytes)
                                  })
    case 1: // continue — append more segments to a live request
        guard let s = engine.requestForStream[sid] else {
            print("  [batch_ffi] continue on unknown stream_id \(sid); ignored")
            return
        }
        for seg in stream.segments {
            switch seg.kind {
            case 0:
                s.submit(seg.tokens)
                s.usage.promptTokensSeen &+= UInt32(seg.tokens.count)
            case 1:
                submitImageSegment(s, imageBytes: seg.imageBytes)
            default:
                break
            }
        }
        // Re-apply sampling/control params on continue (clients may
        // adjust mid-conv). All these are no-ops when the stream's
        // values are unchanged from the prior submit.
        s.samplingTemperature = stream.sampling.temperature
        s.applySamplingSeed(stream.sampling.seed)
        s.logitBiasDense = denseLogitBias(stream.sampling.logitBias)
        s.minP = max(0, min(1, stream.sampling.minP))
        s.cot = cotStateForLabels(stream.sampling.cotLabels)
        s.stopSequences = stream.sampling.stopSequences
        s.clearControls()
        for c in controlsFromDecoded(stream.controlVectors,
                                      startPosition: s.positionForDebug,
                                      startTurn: s.turnIndex) {
            s.addControl(c)
        }
    case 2: // cancel
        guard let s = engine.requestForStream[sid] else { return }
        engine.closeSession(s)  // also unbinds streamId
    case 3: // touch — re-apply policy without new tokens (same code as continue without submit)
        guard let s = engine.requestForStream[sid] else { return }
        s.samplingTemperature = stream.sampling.temperature
        s.applySamplingSeed(stream.sampling.seed)
        s.logitBiasDense = denseLogitBias(stream.sampling.logitBias)
        s.minP = max(0, min(1, stream.sampling.minP))
        s.cot = cotStateForLabels(stream.sampling.cotLabels)
        s.stopSequences = stream.sampling.stopSequences
        s.clearControls()
        for c in controlsFromDecoded(stream.controlVectors,
                                      startPosition: s.positionForDebug,
                                      startTurn: s.turnIndex) {
            s.addControl(c)
        }
    default:
        print("  [batch_ffi] unknown action \(stream.action) for stream_id \(sid)")
    }
}

// ----------------------------------------------------------------------
// Logprob capture for one stream's just-emitted token. Reads the slot's
// 2026-05-07: this used to be a CPU-side computation that iterated
// VOCAB=262144 fp16 logits twice per slot per CB (~2 ms/call × B
// streams × 12 CBs/sec = ~192 ms/sec of CPU work in gemma_poll's
// critical path between AR ticks). It's been replaced by a GPU
// kernel `extract_logprobs` that runs after sample_token in
// buildStepCB, parallel-reducing max + sum_exp + top-K extraction
// in ~50 µs per slot × B = ~0.4 ms per CB. The CPU function below
// just READS the GPU output buffers — no compute, no allocation.
// ----------------------------------------------------------------------
private func captureLogprobForLatestToken(_ s: Session, topK: UInt32) {
    guard let slot = s.slot else { return }
    let prev = s.lastReportedOutputCount
    let curr = Int(s.usage.completionTokensEmitted) + s.pendingOutputCount
    if curr <= prev { return }

    // Read the GPU-computed sampled_logprob + top-K from the output
    // buffers. Buffers are populated by extract_logprobs which ran
    // immediately after sample_token in the just-completed CB (they
    // share the same gEngineLock-serialized CB, and waitUntilCompleted
    // has already returned on this code path).
    let sampledLpP = gpu_sampled_logprobs.contents()
        .assumingMemoryBound(to: Float.self)
    let sampledLogprob = sampledLpP[slot]

    let k = Int(min(topK, UInt32(MAX_TOPK_LOGPROBS)))
    var topPairs: [(UInt32, Float)] = []
    if k > 0 {
        topPairs.reserveCapacity(k)
        let topkIdsP = gpu_topk_token_ids.contents()
            .assumingMemoryBound(to: UInt32.self)
            .advanced(by: slot * MAX_TOPK_LOGPROBS)
        let topkLpsP = gpu_topk_logprobs.contents()
            .assumingMemoryBound(to: Float.self)
            .advanced(by: slot * MAX_TOPK_LOGPROBS)
        for i in 0..<k {
            topPairs.append((topkIdsP[i], topkLpsP[i]))
        }
    }

    // The GPU kernel only writes the LATEST sampled token's logprob (one
    // per CB). For simple AR (one token per tick) that matches the CPU
    // behavior; for prefill or multi-token-per-CB cases, additional
    // tokens past the latest are not captured (matches what the GPU has
    // visibility into — only the just-sampled token went through
    // sample_token).
    let recents = s.peekRecentOutputs(count: curr - prev)
    if let latest = recents.last {
        // Older tokens (recents.dropLast()) have no per-token logprob
        // available from this path — the buffer was overwritten on
        // each AR tick. Emit them with sentinel sampled_logprob = 0
        // and empty top-K so the bridge response shape stays consistent.
        for token in recents.dropLast() {
            s.logprobsQueue.append(LogprobRecord(
                token: token,
                sampledLogprob: 0.0,
                topKPairs: []))
        }
        s.logprobsQueue.append(LogprobRecord(
            token: latest,
            sampledLogprob: sampledLogprob,
            topKPairs: topPairs))
    }
    s.lastReportedOutputCount = curr
}

// ----------------------------------------------------------------------
// Wire-payload → engine-typed value builders. These are pure functions
// (no Session reference, no engine-state mutation) called from
// applyStreamAction's start case to populate RequestInit, and from the
// continue/touch paths to overwrite a live request's policy fields.
// Empty input is the no-op path; nil/empty returns mean "no constraint."
// ----------------------------------------------------------------------
private func denseLogitBias(_ entries: [(UInt32, Float)]) -> [Float]? {
    if entries.isEmpty { return nil }
    var dense = [Float](repeating: 0, count: VOCAB)
    for (tok, bias) in entries {
        let i = Int(tok)
        if i >= 0 && i < VOCAB { dense[i] = bias }
    }
    return dense
}

// Structured-cot grammar from caller-supplied phase labels. nil = no
// constraint. Mirrors the body of Session.enableStructuredCot but
// returns the value instead of attaching it.
private func cotStateForLabels(_ labels: [String]) -> CotState? {
    if labels.isEmpty { return nil }
    var phases: [CotPhase] = []
    phases.append(.literal(bytes: Array("<think>\n".utf8)))
    for label in labels {
        phases.append(.literal(bytes: Array((label + ": ").utf8)))
        phases.append(.freeLine)
    }
    phases.append(.literal(bytes: Array("</think>\n\n".utf8)))
    return CotState(phases: phases)
}

// Resolve every DecodedCV reference to an ActiveControl with concrete
// envelope/buffer pointers. Caller passes startPosition/startTurn so
// the envelope's t=0 anchor matches the request's current time
// coordinates (0/0 at request birth; current values during continue).
private func controlsFromDecoded(_ cvs: [DecodedCV],
                                  startPosition: Int,
                                  startTurn: Int) -> [ActiveControl] {
    var out: [ActiveControl] = []
    out.reserveCapacity(cvs.count)
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
        out.append(ActiveControl(
            cvecId: cv.cvecId, buffer: buf, layer: Int(cv.layer),
            envelope: env, polarity: cv.polarity,
            startPosition: startPosition,
            startTurn: startTurn,
            mode: mode, target: target,
            transportScale: cv.transportScale,
            transportOffset: cv.transportOffset))
    }
    return out
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
private func submitImageSegment(_ s: Session, imageBytes: Data) {
    let hitsBefore = gVisionCacheHits
    guard let padded = ensureCachedSofts(data: imageBytes) else {
        print("  [batch_ffi] ensureCachedSofts failed for stream \(s.streamId)")
        return
    }
    let hitDelta = gVisionCacheHits - hitsBefore
    if hitDelta > 0 {
        s.usage.visionCacheHits &+= UInt32(hitDelta)
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
    s.usage.promptTokensSeen &+= UInt32(padded.count + 2)
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
private func drainSession(_ s: Session) -> [UInt32] {
    var out: [UInt32] = []
    out.reserveCapacity(8)   // typical per-poll drain is 1-8 tokens
    while let t = s.nextToken() {
        out.append(t)
    }
    if !out.isEmpty {
        s.usage.completionTokensEmitted &+= UInt32(out.count)
    }
    return out
}

private func sessionStateByte(_ state: SessionState) -> UInt8 {
    switch state {
    case .priming:    return 0
    case .generating: return 1
    case .done:       return 2
    // .paused retired 2026-05-07; .idle retired 2026-05-07 (atomic
    // construction). Wire format: 0 = priming/inactive, 1 = generating,
    // 2 = terminal. External pollers that branched on stateCode 3
    // (.paused) will simply never see it.
    }
}

// Best-effort done_reason. Engine doesn't yet expose explicit reason;
// approximate from numGenerated and EOS comparison via context. For now
// return 1 (eos) when state is done — the bridge can refine when the
// engine grows a real done_reason field.
private func sessionDoneReason(_ s: Session) -> UInt8 {
    // s.doneReason is the engine-side termination code:
    //   0 = still running (state != .done)
    //   1 = stop / EOS (set by AR loop on EOS or stop_sequence match)
    //   2 = length / max_tokens (currently set by bridge override)
    //   3 = error (e.g. vision tower returned 0 soft tokens; errMsg populated)
    // Sessions that hit .done without any explicit doneReason being
    // set get 1 by default for back-compat with the legacy hardcode.
    if s.state != .done { return 0 }
    if s.doneReason == 0 { return 1 }
    return s.doneReason
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

    // 2026-05-07: every submitted stream is applied directly. No
    // shared-prefix follower deferral. Cross-stream coupling at the
    // bridge layer is forbidden under the NO REMOTE LOCKS principle;
    // the engine-side content-hash KV page cache delivers the same
    // benefit (cached prefix adoption) as a passive accelerator
    // without making B's submission wait on A's prefill progress.
    for stream in items {
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
    //   2. (no deferred-retry step; deleted 2026-05-07 — the
    //      LM_BATCH_SHARED_PREFIX_DEFERRAL feature created a
    //      cross-stream coupling that the NO REMOTE LOCKS principle
    //      forbids; KV-page-cache adoption at the page_manager layer
    //      gives the same prefix-reuse benefit passively, with no
    //      submission gating.)
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

        // 2. (Removed 2026-05-07): deferred-start retry. Shared-prefix
        //    follower deferral was a cross-stream coupling that the
        //    NO REMOTE LOCKS principle forbids; engine-side page-cache
        //    handles prefix reuse passively at the page_manager layer.

        // 3. Drive one chunk if there's work.
        // 2026-05-07: populate per-slot GPU capture buffers BEFORE
        // syncTickStep. The buildStepCB now encodes extract_logprobs
        // immediately after sample_token; the kernel reads
        // gpu_capture_active[slot] to decide whether this slot's
        // logprob+top-K is computed. captureLogprobForLatestToken
        // (called after syncTickStep) reads the GPU output buffers.
        let hadWork = engine.hasWork
        if hadWork {
            // 2026-05-07: run admission BEFORE populating per-slot
            // capture state. Freshly-admitted sessions don't have
            // s.slot assigned until admission runs, and admission
            // normally happens inside tick() — too late for our
            // populate-then-syncTickStep ordering. Calling the
            // public admission wrapper here ensures `s.slot` is
            // valid by the time we read it below.
            engine.runAdmissionPassPublic()
            // Reset capture state for all slots — defaults to "no capture".
            let captureActiveP = gpu_capture_active.contents()
                .bindMemory(to: UInt8.self, capacity: B)
            let captureTopKP = gpu_capture_topk.contents()
                .bindMemory(to: UInt32.self, capacity: B)
            for slot in 0..<B {
                captureActiveP[slot] = 0
                captureTopKP[slot] = 0
            }
            // Set per-slot capture flags for streams with logprobs=True.
            for s in engine.requestForStream.values where s.captureLogits {
                guard let slot = s.slot else { continue }
                captureActiveP[slot] = 1
                captureTopKP[slot] = s.topLogprobs
            }
            engine.syncTickStep()
            for s in engine.requestForStream.values where s.captureLogits {
                captureLogprobForLatestToken(s, topK: s.topLogprobs)
            }
        }

        // 4. Collect updates from streams. APPEND to the per-poll
        //    `updates` buffer — see drive-contract note above. Tracks
        //    `doneAlreadyEmitted` so we don't re-emit a state=2 update
        //    on subsequent iterations after a stream finishes inside
        //    this poll. Cleanup of finished streams happens once at
        //    poll end, AFTER all driving is done.
        for (sid, s) in engine.requestForStream {
            // If this stream finished and we already emitted its
            // state=2 update on a prior iteration of this poll, skip.
            // (engine.requestForStream only loses it during the post-
            // loop cleanup pass, so still iterating over it here is fine.)
            if gDoneEmittedThisPoll.contains(sid) { continue }
            let newToks = drainSession(s)
            let stateByte = sessionStateByte(s.state)
            let doneByte = sessionDoneReason(s)
            if !newToks.isEmpty || s.state == .done {
                var lps: [LogprobRecord] = []
                if s.captureLogits {
                    let avail = s.logprobsQueue
                    let n = min(avail.count, newToks.count)
                    lps = Array(avail.prefix(n))
                    if n > 0 {
                        s.logprobsQueue = Array(avail.dropFirst(n))
                    }
                }
                updates.append(StreamUpdateOut(
                    streamId: sid,
                    state: stateByte,
                    doneReason: doneByte,
                    newTokens: newToks,
                    errMsg: s.errMsg,
                    promptTokensSeen: s.usage.promptTokensSeen,
                    completionTokensEmitted: s.usage.completionTokensEmitted,
                    cacheHits: s.cacheHitTokens,
                    cacheMisses: s.cacheMissTokens,
                    visionCacheHits: s.usage.visionCacheHits,
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

    // Clean up any sessions that finished. closeSession decrefs page
    // refs, removes from residentSessions, and (via unbindStream)
    // drops the streamId entry — request state lives entirely on
    // Session and dies with it. No bridge-side state to clean.
    for u in updates where u.state == 2 {
        if let s = engine.requestForStream[u.streamId] {
            engine.closeSession(s)
        }
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
        for s in engine.requestForStream.values {
            switch s.state {
            case .generating: generating += 1
            case .priming:    priming += 1
            default:          break
            }
        }
        w.u32(UInt32(stats.totalPages))
        w.u32(UInt32(stats.freePages))
        w.u32(UInt32(stats.cachedHashes))
        w.u32(UInt32(engine.requestForStream.count))
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
        // closeSession unbinds the streamId; iterate on a snapshot of
        // values to avoid mutating the dict mid-iteration.
        for s in Array(engine.requestForStream.values) {
            engine.closeSession(s)
        }
    }
    gEngine = nil
    return 0
}
