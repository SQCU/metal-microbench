// Per-layer weights, paged KV cache, and tokenizer state for the LM forward.
//
// Extracted 2026-05-14 from bootstrap.swift as part of the 8-file split
// motivated by SourceKit indexing churn + cognitive load. Contains:
//   - PrefixCache: scratch class used by the shared-prefix prefill path
//   - LayerW: per-layer weight buffers (attention + FFN + MoE + norms)
//   - LmWeights: the bundled struct passed to every forward graph encoder
//   - loadLmWeights: GGUF parser + Q8_0/Q4_K/Q5_1 swizzle + KV alloc
//
// All declarations remain at module scope (no namespace), so existing
// callsites need no changes — this is purely a physical-file move.

import Metal
import Foundation

final class PrefixCache {
    private var byHash: [UInt64: (pages: [Int], length: Int, refCount: Int)] = [:]
    private var freeList: [Int] = []
    private(set) var nextPhys: Int = 0
    private let maxPhys: Int

    init(maxPhys: Int) {
        self.maxPhys = maxPhys
    }

    // FNV-1a hash of a sequence of UInt32 token IDs.
    static func hash(_ tokens: ArraySlice<UInt32>) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for t in tokens {
            h ^= UInt64(t)
            h = h &* 0x100000001b3
        }
        return h
    }

    // Return phys pages covering `tokens` — reused if an entry for this
    // exact hash+length is cached, else freshly allocated.
    func getOrAllocate(tokens: [UInt32], pageSize: Int) -> [Int] {
        let h = PrefixCache.hash(tokens[...])
        if var entry = byHash[h], entry.length == tokens.count {
            entry.refCount += 1
            byHash[h] = entry
            return entry.pages
        }
        let numPages = (tokens.count + pageSize - 1) / pageSize
        var pages: [Int] = []
        pages.reserveCapacity(numPages)
        for _ in 0..<numPages {
            if let recycled = freeList.popLast() {
                pages.append(recycled)
            } else {
                precondition(nextPhys < maxPhys, "PrefixCache out of phys pages")
                pages.append(nextPhys)
                nextPhys += 1
            }
        }
        byHash[h] = (pages: pages, length: tokens.count, refCount: 1)
        return pages
    }

    func release(tokens: [UInt32]) {
        let h = PrefixCache.hash(tokens[...])
        guard var entry = byHash[h] else { return }
        entry.refCount -= 1
        if entry.refCount <= 0 {
            freeList.append(contentsOf: entry.pages)
            byHash.removeValue(forKey: h)
        } else {
            byHash[h] = entry
        }
    }

    var entryCount: Int { byHash.count }
    var totalCachedPages: Int { byHash.values.reduce(0) { $0 + $1.pages.count } }
}


struct LayerW {
    let attnQ, attnK, attnOut: MTLBuffer              // dense, format may be Q8_0/Q5_K/Q6_K
    let attnV: MTLBuffer?                              // nil on full-attn layers:
                                                        // Gemma-4 drops the V projection and uses
                                                        // K as V at those layers (see llama.cpp
                                                        // gemma4-iswa.cpp:83-85)
    let ffnGate, ffnUp, ffnDown: MTLBuffer            // dense, format may be Q8_0/Q5_K/Q6_K
    let moeGateUp: MTLBuffer                           // MoE up, format Q4_K or Q5_K
    let moeDown: MTLBuffer                             // MoE down, format Q5_1/Q6_K/Q8_0
    // Per-tensor formats — populated from GGUF's actual dtype on load.
    // The format-aware dispatchers (encDenseMmPrefill / encMoeUpMmPrefill /
    // encMoeDownMmPrefill) read these to pick the right kernel PSO.
    let attnQFormat, attnKFormat, attnOutFormat: GGMLType
    let attnVFormat: GGMLType                          // valid iff attnV != nil
    let ffnGateFormat, ffnUpFormat, ffnDownFormat: GGMLType
    let moeGateUpFormat, moeDownFormat: GGMLType
    let attnNorm, postAttnNorm: MTLBuffer              // f16 from f32
    let attnQNorm, attnKNorm: MTLBuffer                // f16 from f32 (per-head)
    let ffnNorm, postFfn1Norm: MTLBuffer               // shared FFN pre/post
    let preFfn2Norm, postFfn2Norm: MTLBuffer           // MoE pre/post
    let postFfnNorm: MTLBuffer                          // combined post-FFN
    let routerW, routerScale, expertScale: MTLBuffer   // f32 passthrough
    let layerOutputScale: MTLBuffer                     // f32 scalar
    let isFull: Bool
    let KV_H: Int
    let HD: Int
}

// Bundled real-weight + per-layer KV cache + tokenizer state for the LM forward.
struct LmWeights {
    let layers: [LayerW]
    let embedTable: MTLBuffer        // fp16 [VOCAB, HIDDEN] dequantized from Q8_0
    let unembedW: MTLBuffer          // fp16 [HIDDEN, VOCAB] transposed tied view
    let outputNorm: MTLBuffer        // fp16 [HIDDEN] final RMSNorm gamma
    let embedScaleBuf: MTLBuffer     // fp32 single scalar = sqrt(HIDDEN)
    // Virtual-page-table KV cache, scatter-gather via argument buffers.
    //
    // 2026-05-14 argbuf refactor. Each layer's K and V cache live across
    // KV_NUM_CHUNKS=4 device buffers; phys-page-id `p` maps via
    //   chunk_idx = p / kvChunkPages
    //   local_phys = p - chunk_idx * kvChunkPages
    // and the chunks live at K_chunks[L][chunk_idx] / V_chunks[L][chunk_idx].
    //
    // Kernels access them through an argument buffer (KVChunks struct)
    // pre-encoded once at startup and bound at a single buffer index per
    // kernel invocation. Per-CB useResource() hints make the GPU's
    // residency tracker list only the chunks actually touched, not the
    // full ~100+ GB pool. This is the whole point: scatter-gather lets
    // the addressable pool grow to 110 GB+ while the per-CB working set
    // stays at ~6 GB.
    //
    // The legacy K_caches/V_caches (single buffer or lo/hi pair) is
    // deleted — every consumer goes through K_chunks_argbuf now.
    let K_chunks: [[MTLBuffer]]              // [NUM_LAYERS][KV_NUM_CHUNKS]
    let V_chunks: [[MTLBuffer]]              // [NUM_LAYERS][KV_NUM_CHUNKS]
    let K_chunks_argbuf: [MTLBuffer]         // [NUM_LAYERS], encoded argument buffer
    let V_chunks_argbuf: [MTLBuffer]         // [NUM_LAYERS], encoded argument buffer
    let kvChunkPages: Int                    // pages per chunk (rounded up)
    let bosTokenId: UInt32
    let eosTokenId: UInt32           // tokenizer.ggml.eos_token_id (Gemma4: 106 = <end_of_turn>)
    let addBosToken: Bool            // tokenizer.ggml.add_bos_token
    let vocabTokens: [String]        // decoded from tokenizer.ggml.tokens
    let merges: [String]             // tokenizer.ggml.merges — "TOKEN_A TOKEN_B" pairs in priority order
}

// Load every weight we need for a real Gemma-4-A4B LM forward from a Q4_K_M
// GGUF. Q8_0 dense weights get swizzled for the v6 kernel; Q4_K / Q5_1 MoE
// weights get per-expert-swizzled; norms load f32→f16; routing scales and
// layer scalars stay f32. Allocates per-layer paged K/V caches (zero-filled).
// ─────────────────────────────────────────────────────────────────────
// Persistent weight residency set (2026-05-28).
//
// Profiling the bridge showed generation was SUBMIT-bound, not compute-bound:
// the Metal CommandQueueDispatch thread sat ~100% in
// IOGPUCommandQueueSubmitCommandBuffers → iokit_user_client_trap, while
// waitUntilCompleted (actual GPU compute) barely registered and the bridge's
// asyncio thread was idle. Cause: every command buffer re-walks the residency
// list of the ~660 weight buffers (which are fully indexed on every forward
// pass and never change). Pin them ONCE in a persistent MTLResidencySet
// attached to the LM queue so per-CB submission skips them. The KV pool stays
// on the narrowed per-CB useResource path — its working set changes per CB and
// must not be made fully/permanently resident. macOS 15+ only; older systems
// fall back to the prior per-CB residency behavior.
private var gWeightResidencySet: AnyObject?

// ── Tier 0 KV pin-on-grow residency set (2026-06) ──────────────────────────
// SEPARATE from the weight set (gWeightResidencySet) by design:
//   (1) the weight set is committed ONCE at boot from a complete, immutable
//       buffer list (installWeightResidencySet); a residency set is committed
//       as a unit. Pin-on-grow ADDS allocations incrementally and re-commits/
//       re-requests as KV chunks enter the hot working set, so it needs its
//       own mutable lifecycle.
//   (2) the weight-set comment above explicitly states KV must NOT ride the
//       weight set's per-CB-skip path because its working set changes; a
//       distinct set keeps that boundary legible.
//   (3) keeps allWeightBuffers (argbuf POINTERS only) untouched, so
//       staticResidentBytesEstimate (the poolCap math input) stays correct.
// Both sets attach to the SAME LM queue (Metal allows multiple per queue).
// Process-life retain.
private var gKvResidencySet: AnyObject?

extension LmWeights {
    /// Every static weight buffer — fully indexed each forward, immutable.
    /// EXCLUDES the KV chunk pool (K_chunks/V_chunks); includes the static
    /// per-layer argument buffers (encoded once, read every CB).
    var allWeightBuffers: [MTLBuffer] {
        var bufs: [MTLBuffer] = [embedTable, unembedW, outputNorm, embedScaleBuf]
        for L in layers {
            bufs.append(contentsOf: [
                L.attnQ, L.attnK, L.attnOut, L.ffnGate, L.ffnUp, L.ffnDown,
                L.moeGateUp, L.moeDown,
                L.attnNorm, L.postAttnNorm, L.attnQNorm, L.attnKNorm,
                L.ffnNorm, L.postFfn1Norm, L.preFfn2Norm, L.postFfn2Norm, L.postFfnNorm,
                L.routerW, L.routerScale, L.expertScale, L.layerOutputScale])
            if let v = L.attnV { bufs.append(v) }
        }
        bufs.append(contentsOf: K_chunks_argbuf)
        bufs.append(contentsOf: V_chunks_argbuf)
        return bufs
    }

    /// G2 (2026-06): resident footprint of the static model weights in bytes
    /// — the sum of every static weight buffer's length (embeddings, per-layer
    /// projections/norms/router, argument buffers). EXCLUDES the lazy-committed
    /// K/V chunk pool (that is the budget being sized) and the vision tower
    /// (allocated separately, small relative to the LM). Used by LmEngine to
    /// reserve the model's RAM before handing the rest of KV_MEM_BUDGET_FRAC ·
    /// physicalMemory to the dynamic KV pool, so growth cannot OOM the box.
    func staticResidentBytesEstimate() -> Int {
        var total = 0
        for b in allWeightBuffers { total += b.length }
        return total
    }
}

func installWeightResidencySet(_ w: LmWeights) {
    if #available(macOS 15.0, *) {
        do {
            let set = try device.makeResidencySet(descriptor: MTLResidencySetDescriptor())
            let bufs = w.allWeightBuffers
            for b in bufs { set.addAllocation(b) }
            set.commit()
            set.requestResidency()
            queue.addResidencySet(set)
            gWeightResidencySet = set   // retain for process lifetime
            FileHandle.standardError.write(Data(
                "[residency] pinned \(bufs.count) weight buffers in a persistent MTLResidencySet (attached to LM queue)\n".utf8))
        } catch {
            FileHandle.standardError.write(Data(
                "[residency] WARN: MTLResidencySet failed (\(error)) — per-CB useResource fallback\n".utf8))
        }
    } else {
        FileHandle.standardError.write(Data(
            "[residency] macOS < 15: no MTLResidencySet; per-CB useResource path\n".utf8))
    }
    // G9 (2026-06): WIRE the static weights against IDLE eviction. The
    // MTLResidencySet above wires them for per-CB submission efficiency, NOT
    // against the OS reclaiming memory while the engine sits idle: the weights
    // are mmap'd (clean, file-backed, droppable) or copied (compressible), all
    // storageModeShared. Under accumulated host memory pressure macOS evicted
    // them and the next forward pass re-faulted ~25GB from the GGUF on SSD —
    // an ~80s cold prefill (decode was always fine). mlock() pins them
    // unconditionally (wired limit ~108GB >> ~25GB model); madvise(WILLNEED)
    // prefetches/keeps them hot. ONLY the static weights — NOT the KV chunks,
    // which are pin-on-grow + DESIGNED to be evictable under page pressure.
    // Env-gated (LM_WIRE_WEIGHTS=0 disables) in case wired pressure ever bites;
    // mlock failure is non-fatal (we degrade to the prior evict-and-refault).
    if ProcessInfo.processInfo.environment["LM_WIRE_WEIGHTS"] != "0" {
        var wiredBufs = 0, wiredBytes = 0, failed = 0
        for b in w.allWeightBuffers {
            let p = b.contents(); let n = b.length
            if n == 0 { continue }
            madvise(p, n, MADV_WILLNEED)
            if mlock(p, n) == 0 { wiredBufs += 1; wiredBytes += n } else { failed += 1 }
        }
        FileHandle.standardError.write(Data(
            "[residency] mlock+WILLNEED \(wiredBufs) static weight buffers (~\(wiredBytes / (1024*1024*1024)) GB wired against idle eviction; \(failed) mlock failures)\n".utf8))
    } else {
        FileHandle.standardError.write(Data(
            "[residency] LM_WIRE_WEIGHTS=0 — weights NOT wired (will evict+refault under idle pressure)\n".utf8))
    }
    // Tier 0: create the (empty) KV residency set + attach it to the LM queue
    // at boot. No KV chunk is addAllocation'd here — that is PIN-ON-GROW
    // (pinKvChunk, fired from the page allocator's grow frontier).
    makeKvResidencySetIfNeeded()
}

// Tier 0 pin-on-grow — create the persistent KV residency set ONCE at boot and
// attach it to the LM queue. The set starts EMPTY; chunks enter it lazily via
// pinKvChunk as the page allocator's resident frontier grows into them.
//
// Tier 0 REQUIRES pinning: on macOS < 15 (no MTLResidencySet) OR if the set
// cannot be created, this is a LOUD boot-time refusal (fail()), NOT a silent
// unpinned run — an unpinned KV pool is exactly the page-out fault surface this
// design removes (doc §4). (The weight set degrades to per-CB useResource on
// the same branch; for KV we must escalate because Tier 0 cannot fall back.)
func makeKvResidencySetIfNeeded() {
    if gKvResidencySet != nil { return }
    if #available(macOS 15.0, *) {
        do {
            let set = try device.makeResidencySet(descriptor: MTLResidencySetDescriptor())
            // Empty commit is well-defined; chunks are added incrementally.
            set.commit()
            queue.addResidencySet(set)
            gKvResidencySet = set   // retain for process lifetime
            FileHandle.standardError.write(Data(
                "[kv-residency] created persistent KV MTLResidencySet (attached to LM queue, pin-on-grow; 0 chunks resident at boot)\n".utf8))
        } catch {
            fail("[kv-residency] FATAL: could not create KV MTLResidencySet (\(error)); "
                + "Tier 0 REQUIRES a wired KV pool — refusing to run unpinned "
                + "(would reintroduce the hot-KV page-out fault, doc §4)")
        }
    } else {
        fail("[kv-residency] FATAL: macOS < 15 has no MTLResidencySet; "
            + "Tier 0 REQUIRES a wired KV pool — refusing to run unpinned "
            + "(would reintroduce the hot-KV page-out fault, doc §4)")
    }
}

// Deterministic teardown of the wired residency sets (2026-06). The weight +
// KV residency sets pin tens of GB of wired GPU memory; without an explicit
// release, reclamation relies on the OS tearing down the process image, which
// (a) is non-deterministic on graceful shutdown and (b) can lag/leak under an
// abrupt exit. gemma_shutdown calls this so a graceful stop (uvicorn SIGTERM ->
// app shutdown -> gemma_shutdown) UNWIRES immediately. Idempotent: nils the
// globals so a re-init recreates them via install/makeKvResidencySetIfNeeded.
//
// Supervisor contract: send SIGTERM (graceful drain) — NEVER SIGKILL — so this
// path runs. (SIGKILL skips it; the OS still reclaims on process death, but not
// deterministically.) Target the real serve.py PID, not the uv wrapper.
func releaseResidencySets() {
    if #available(macOS 15.0, *) {
        for g in [gWeightResidencySet, gKvResidencySet] {
            guard let set = g as? MTLResidencySet else { continue }
            queue.removeResidencySet(set)
            set.endResidency()
        }
        FileHandle.standardError.write(Data(
            "[residency] released weight + KV residency sets (unwired on shutdown)\n".utf8))
    }
    gWeightResidencySet = nil
    gKvResidencySet = nil
}

// Tier 0 pin-on-grow — wire ALL 30 layers' K and V buffers for chunk `chunkIdx`
// (= 60 MTLBuffers) into the KV residency set, then requestResidency.
//
// GEOMETRY: one phys page covers [16P..16P+15] in EVERY layer's K/V buffer
// (page_manager.swift:58-59), so committing one pool page touches that page's
// slice in all 30 layers' K AND V — all 60 buffers for the chunk must be wired
// together or the first forward READ of a slide/full layer faults.
//
// requestResidency is LOAD-BEARING (requirement 3): this is a GROW-TIME
// PRECONDITION. If the set is unavailable or wiring fails, this LOUD-fails at
// the grow point — it does NOT fall through to handing back an unpinned page.
// (macOS requestResidency() returns Void and does not signal a wired-limit
// failure on the success path; the LM_KV_POOL_PAGES test — vm_stat wired delta
// per chunk, pageins flat — is the validation that the call is load-bearing.)
func pinKvChunk(_ w: LmWeights, _ chunkIdx: Int) {
    precondition(chunkIdx >= 0 && chunkIdx < KV_NUM_CHUNKS,
                 "pinKvChunk: chunkIdx \(chunkIdx) out of range [0, \(KV_NUM_CHUNKS))")
    // Per-chunk byte size (telemetry + the LOUD-fail message).
    var chunkBytes = 0
    for L in 0..<NUM_LAYERS {
        chunkBytes += w.K_chunks[L][chunkIdx].length + w.V_chunks[L][chunkIdx].length
    }
    if #available(macOS 15.0, *) {
        guard let set = gKvResidencySet as? MTLResidencySet else {
            fail("[kv-residency] FATAL: failed to wire KV chunk \(chunkIdx) "
                + "(60 buffers, ~\(chunkBytes / (1024*1024)) MiB) at grow frontier: "
                + "KV residency set was never created — Tier 0 cannot serve an "
                + "unpinned page (doc §4)")
        }
        for L in 0..<NUM_LAYERS {
            set.addAllocation(w.K_chunks[L][chunkIdx])
            set.addAllocation(w.V_chunks[L][chunkIdx])
        }
        set.commit()
        set.requestResidency()
        FileHandle.standardError.write(Data(
            "[kv-residency] pinned KV chunk \(chunkIdx) (60 buffers, ~\(chunkBytes / (1024*1024)) MiB) into the KV residency set + requestResidency\n".utf8))
    } else {
        fail("[kv-residency] FATAL: failed to wire KV chunk \(chunkIdx) "
            + "(60 buffers, ~\(chunkBytes / (1024*1024)) MiB) at grow frontier: "
            + "macOS < 15 has no MTLResidencySet — Tier 0 cannot serve an "
            + "unpinned page (doc §4)")
    }
}

func loadLmWeights(ggufPath: String) throws -> LmWeights {
    let t0 = Date()
    print("  loading GGUF: \(ggufPath)")
    let g = try GGUFFile(ggufPath)
    print(String(format: "  GGUF parsed in %.1f ms (%d tensors, %d metadata)",
                 Date().timeIntervalSince(t0) * 1000, g.tensors.count, g.metadata.count))

    // Per-layer KV_H (Gemma-4 alternates: 5 sliding with KV_H=8, then 1 full with KV_H=2).
    var layerKVH: [Int] = []
    if let kvArr = g.metadata["gemma4.attention.head_count_kv"] as? [Any] {
        for v in kvArr {
            if let vi = v as? UInt32      { layerKVH.append(Int(vi)) }
            else if let vi = v as? Int32  { layerKVH.append(Int(vi)) }
            else if let vi = v as? Int    { layerKVH.append(vi) }
            else if let vi = v as? UInt64 { layerKVH.append(Int(vi)) }
            else if let vi = v as? Int64  { layerKVH.append(Int(vi)) }
        }
    }
    precondition(layerKVH.count == NUM_LAYERS,
                 "expected \(NUM_LAYERS) KV_H entries, got \(layerKVH.count)")
    print("  per-layer KV_H: \(layerKVH)")

    // ---------- Per-tensor loaders ----------
    func loadF32AsF16(_ name: String) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == .f32, "\(name): expected f32")
        let nElems = info.shape.reduce(1, *)
        let dst = device.makeBuffer(length: nElems * 2, options: .storageModeShared)!
        let sp = g.base.advanced(by: info.dataOffset).assumingMemoryBound(to: Float.self)
        let dp = dst.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<nElems { dp[i] = Float16(sp[i]) }
        return dst
    }
    func loadF32Raw(_ name: String) throws -> MTLBuffer {
        return try g.makeMetalBuffer(name, device: device)
    }
    // Load a 2D f32 GGUF tensor and convert to a half-precision buffer laid out
    // as [D_in, D_out] row-major, which is what dense_gemv_v5 expects when it
    // reads W[k*D_out + n]. GGUF shape reports [D_in, D_out] but stores bytes
    // as [D_out, D_in] row-major (GGUF axis 0 = fastest = D_in). Our per-layer
    // router weight `ffn_gate_inp.weight` is the only f32 2D tensor; it must
    // be transposed AND cast to half for the kernel to read correctly.
    func loadF32ToHalfTransposed2D(_ name: String) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == .f32, "\(name): expected f32")
        precondition(info.shape.count == 2, "\(name): expected 2D, got shape \(info.shape)")
        let D_in = info.shape[0], D_out = info.shape[1]
        let srcBuf = try g.makeMetalBuffer(name, device: device)
        let dst = device.makeBuffer(length: D_in * D_out * 2, options: .storageModeShared)!
        let src = srcBuf.contents().assumingMemoryBound(to: Float.self)
        let dp = dst.contents().assumingMemoryBound(to: Float16.self)
        // Source bytes are [D_out, D_in] row-major: src[e*D_in + k] = W[e, k].
        // Destination wants [D_in, D_out] row-major: dp[k*D_out + n] = W[n, k].
        for k in 0..<D_in {
            for n in 0..<D_out {
                dp[k * D_out + n] = Float16(src[n * D_in + k])
            }
        }
        return dst
    }
    func loadQ80Swizzled(_ name: String) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == .q8_0, "\(name): expected q8_0")
        let Din = info.shape[0], Dout = info.shape[1]
        let raw = try g.makeMetalBuffer(name, device: device)
        let sw = device.makeBuffer(length: raw.length, options: .storageModeShared)!
        repackQ80ToSwizzled(src: raw, dst: sw, Din: Din, Dout: Dout)
        return sw
    }
    // F16 has no block structure (every element is a plain 2-byte half),
    // so no swizzling is needed. The GGUF-native row-major [Din, Dout]
    // layout matches what the F16 kernels expect — return the raw
    // MTLBuffer directly. Used for dense F16 weights and (since the same
    // loader works for any rank-2/3 F16 tensor) MoE F16 weights too.
    func loadF16Raw(_ name: String) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == .f16, "\(name): expected f16")
        return try g.makeMetalBuffer(name, device: device)
    }
    // Auto-detecting dense loader: reads the GGUF tensor's actual dtype and
    // dispatches to the right swizzler. Returns (buffer, format) tuple so
    // LayerW can populate the matching format field. Supports Q8_0/Q5_K/Q6_K
    // (the dense formats in the V1 grid that have AR-decode + prefill kernels).
    // Resolve the tensor class from a GGUF tensor name like "blk.5.attn_q.weight"
    // → "attn_q". The class drives both capability validation and dispatch.
    func tensorClassFromName(_ name: String) -> String {
        // Strip "blk.<L>." prefix if present.
        var stripped = name
        if let dotAfterBlk = stripped.range(of: "blk.")?.upperBound {
            if let nextDot = stripped[dotAfterBlk...].firstIndex(of: ".") {
                stripped = String(stripped[stripped.index(after: nextDot)...])
            }
        }
        // Strip ".weight" suffix.
        if stripped.hasSuffix(".weight") {
            stripped = String(stripped.dropLast(".weight".count))
        }
        return stripped
    }

    func loadDenseAuto(_ name: String) throws -> (MTLBuffer, GGMLType) {
        let info = try g.tensor(name)
        // Single source of truth: validate against kernel_capabilities.json
        // before the dispatch switch picks the right loader.
        assertCapability(tensorClassFromName(name), info.dtype, tensorName: name)
        switch info.dtype {
        case .q8_0:
            return (try loadQ80Swizzled(name), .q8_0)
        case .q5_K:
            return (try loadDenseSwizzled(name, dtype: .q5_K, blkBytes: 176, blkElems: 256), .q5_K)
        case .q6_K:
            return (try loadDenseSwizzled(name, dtype: .q6_K, blkBytes: 210, blkElems: 256), .q6_K)
        case .q4_K:
            // Q4_K dense path (super-block 256, 144 B). Routes to the Q4_K
            // dense btile zoo via encDenseGemvAR .q4_K. Added 2026-06 alongside
            // dense_gemv_q4k_btile_b{1,2,4,8} so a mixed GGUF with a Q4_K dense
            // tensor loads + decodes through the btile fast path (the UD-Q4_K_M
            // goal model keeps dense at Q8_0, so this is for non-uniform mixes).
            return (try loadDenseSwizzled(name, dtype: .q4_K, blkBytes: 144, blkElems: 256), .q4_K)
        case .q5_1:
            return (try loadDenseSwizzled(name, dtype: .q5_1, blkBytes: 24, blkElems: 32), .q5_1)
        case .q4_0:
            return (try loadDenseSwizzled(name, dtype: .q4_0, blkBytes: 18, blkElems: 32), .q4_0)
        case .q4_1:
            return (try loadDenseSwizzled(name, dtype: .q4_1, blkBytes: 20, blkElems: 32), .q4_1)
        case .f16:
            return (try loadF16Raw(name), .f16)
        default:
            fail("loadDenseAuto(\(name)): unsupported dtype \(info.dtype) (capability check should have caught this; engine and capabilities matrix are out of sync)")
        }
    }
    // Auto-detecting MoE up loader (slot_token broadcast convention).
    // Supports Q4_K (Q4_K_M default), Q5_K (Q5_K_M default), Q4_0 (--pure Q4_0),
    // and (since 2026-05-13) Q8_0 for uniform-Q8_0 quant-ablation builds.
    func loadMoEUpAuto(_ name: String) throws -> (MTLBuffer, GGMLType) {
        let info = try g.tensor(name)
        assertCapability(tensorClassFromName(name), info.dtype, tensorName: name)
        switch info.dtype {
        case .q4_K:
            return (try loadMoESwizzled(name, dtype: .q4_K, blkBytes: 144, blkElems: 256, E: E_EXP), .q4_K)
        case .q5_K:
            return (try loadMoESwizzled(name, dtype: .q5_K, blkBytes: 176, blkElems: 256, E: E_EXP), .q5_K)
        case .q4_0:
            return (try loadMoESwizzled(name, dtype: .q4_0, blkBytes: 18, blkElems: 32, E: E_EXP), .q4_0)
        case .q4_1:
            return (try loadMoESwizzled(name, dtype: .q4_1, blkBytes: 20, blkElems: 32, E: E_EXP), .q4_1)
        case .q8_0:
            // Added 2026-05-13 alongside moe_gemv_q8_0_v11_up_b{1,2,4,8}.
            // Same block params + swizzle as the q8_0 down loader because
            // loadMoESwizzled is layout-agnostic between up and down — the
            // (expert_stride, super_block, col-major-per-block) pattern is
            // identical, only the dispatcher's hidden-pointer indirection
            // differs at compute time.
            return (try loadMoESwizzled(name, dtype: .q8_0, blkBytes: 34, blkElems: 32, E: E_EXP), .q8_0)
        case .f16:
            return (try loadF16Raw(name), .f16)
        default:
            fail("loadMoEUpAuto(\(name)): unsupported dtype \(info.dtype) (capability check should have caught this)")
        }
    }
    // Auto-detecting MoE down loader (per-slot convention). Each format
    // has its own dedicated per-slot kernel — no convention-mixing routes.
    func loadMoEDownAuto(_ name: String) throws -> (MTLBuffer, GGMLType) {
        let info = try g.tensor(name)
        assertCapability(tensorClassFromName(name), info.dtype, tensorName: name)
        switch info.dtype {
        case .q5_1:
            return (try loadMoESwizzled(name, dtype: .q5_1, blkBytes: 24, blkElems: 32, E: E_EXP), .q5_1)
        case .q6_K:
            return (try loadMoESwizzled(name, dtype: .q6_K, blkBytes: 210, blkElems: 256, E: E_EXP), .q6_K)
        case .q8_0:
            return (try loadMoESwizzled(name, dtype: .q8_0, blkBytes: 34, blkElems: 32, E: E_EXP), .q8_0)
        case .q4_0:
            return (try loadMoESwizzled(name, dtype: .q4_0, blkBytes: 18, blkElems: 32, E: E_EXP), .q4_0)
        case .q4_1:
            return (try loadMoESwizzled(name, dtype: .q4_1, blkBytes: 20, blkElems: 32, E: E_EXP), .q4_1)
        case .f16:
            return (try loadF16Raw(name), .f16)
        default:
            fail("loadMoEDownAuto(\(name)): unsupported dtype \(info.dtype) (capability check should have caught this)")
        }
    }
    // Generic dense weight swizzler for any quant. Source is GGUF-native
    // [Dout cols, nbc kb, blkBytes] (Dout rows, each row is nbc super-blocks).
    // Destination is v6 swizzled [n_super=Dout/32, nbc, 32 cols, blkBytes]:
    // 32 threads of an SG read 32 contiguous blocks per kb iteration. Same
    // shape as repackQ80ToSwizzled but parameterized by block geometry.
    func loadDenseSwizzled(_ name: String, dtype: GGMLType,
                            blkBytes: Int, blkElems: Int) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == dtype, "\(name): expected \(dtype)")
        let Din = info.shape[0], Dout = info.shape[1]
        precondition(Dout % 32 == 0, "\(name): Dout=\(Dout) must be a multiple of 32")
        precondition(Din % blkElems == 0, "\(name): Din=\(Din) must be a multiple of \(blkElems)")
        let raw = try g.makeMetalBuffer(name, device: device)
        let sw = device.makeBuffer(length: raw.length, options: .storageModeShared)!
        let nbc = Din / blkElems
        let colBytes = nbc * blkBytes
        let nSuper = Dout / 32
        let sp = raw.contents().assumingMemoryBound(to: UInt8.self)
        let dp = sw.contents().assumingMemoryBound(to: UInt8.self)
        for ns in 0..<nSuper {
            let srcColBase = ns * 32 * colBytes
            let dstSuperBase = ns * nbc * 32 * blkBytes
            for kb in 0..<nbc {
                for col in 0..<32 {
                    let srcOff = srcColBase + col * colBytes + kb * blkBytes
                    let dstOff = dstSuperBase + kb * 32 * blkBytes + col * blkBytes
                    memcpy(dp.advanced(by: dstOff), sp.advanced(by: srcOff), blkBytes)
                }
            }
        }
        return sw
    }
    func loadMoESwizzled(_ name: String, dtype: GGMLType, blkBytes: Int, blkElems: Int,
                          E: Int) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == dtype, "\(name): expected \(dtype)")
        let Din = info.shape[0], Dout = info.shape[1]
        precondition(info.shape[2] == E, "\(name): expected E=\(E)")
        let raw = try g.makeMetalBuffer(name, device: device)
        let sw = device.makeBuffer(length: raw.length, options: .storageModeShared)!
        let nbc = Din / blkElems
        let colBytes = nbc * blkBytes
        let nSuper = Dout / 32
        let sp = raw.contents().assumingMemoryBound(to: UInt8.self)
        let dp = sw.contents().assumingMemoryBound(to: UInt8.self)
        for expert in 0..<E {
            let srcExpBase = expert * Dout * colBytes
            let dstExpBase = expert * nSuper * nbc * 32 * blkBytes
            for ns in 0..<nSuper {
                let srcColBase = srcExpBase + ns * 32 * colBytes
                let dstSuperBase = dstExpBase + ns * nbc * 32 * blkBytes
                for kb in 0..<nbc {
                    for col in 0..<32 {
                        let srcOff = srcColBase + col * colBytes + kb * blkBytes
                        let dstOff = dstSuperBase + kb * 32 * blkBytes + col * blkBytes
                        memcpy(dp.advanced(by: dstOff), sp.advanced(by: srcOff), blkBytes)
                    }
                }
            }
        }
        return sw
    }

    // ---------- Load all 30 layers ----------
    let tLoad = Date()
    var layers: [LayerW] = []
    layers.reserveCapacity(NUM_LAYERS)
    for L in 0..<NUM_LAYERS {
        let p = "blk.\(L)."
        let lkv = layerKVH[L]
        let isFull = (lkv == 2)
        let hd = isFull ? FULL_HD : SLIDE_HD
        // Load each tensor through the auto-detecting loader; capture
        // (buffer, format) tuples and unpack into LayerW. The format
        // fields drive the prefill dispatcher's PSO selection downstream.
        let (attnQBuf, attnQFmt)   = try loadDenseAuto("\(p)attn_q.weight")
        let (attnKBuf, attnKFmt)   = try loadDenseAuto("\(p)attn_k.weight")
        let (attnOutBuf, attnOutFmt) = try loadDenseAuto("\(p)attn_output.weight")
        let attnVTuple: (MTLBuffer, GGMLType)? = (g.tensors["\(p)attn_v.weight"] != nil)
            ? try loadDenseAuto("\(p)attn_v.weight") : nil
        let (ffnGateBuf, ffnGateFmt)   = try loadDenseAuto("\(p)ffn_gate.weight")
        let (ffnUpBuf, ffnUpFmt)       = try loadDenseAuto("\(p)ffn_up.weight")
        let (ffnDownBuf, ffnDownFmt)   = try loadDenseAuto("\(p)ffn_down.weight")
        let (moeUpBuf, moeUpFmt)       = try loadMoEUpAuto("\(p)ffn_gate_up_exps.weight")
        let (moeDownBuf, moeDownFmt)   = try loadMoEDownAuto("\(p)ffn_down_exps.weight")
        let lw = LayerW(
            attnQ:    attnQBuf,
            attnK:    attnKBuf,
            attnOut:  attnOutBuf,
            attnV:    attnVTuple?.0,
            ffnGate:  ffnGateBuf,
            ffnUp:    ffnUpBuf,
            ffnDown:  ffnDownBuf,
            moeGateUp: moeUpBuf,
            moeDown:   moeDownBuf,
            attnQFormat:    attnQFmt,
            attnKFormat:    attnKFmt,
            attnOutFormat:  attnOutFmt,
            attnVFormat:    attnVTuple?.1 ?? .q8_0,    // unused when attnV is nil
            ffnGateFormat:  ffnGateFmt,
            ffnUpFormat:    ffnUpFmt,
            ffnDownFormat:  ffnDownFmt,
            moeGateUpFormat: moeUpFmt,
            moeDownFormat:   moeDownFmt,
            attnNorm:       try loadF32AsF16("\(p)attn_norm.weight"),
            postAttnNorm:   try loadF32AsF16("\(p)post_attention_norm.weight"),
            attnQNorm:      try loadF32AsF16("\(p)attn_q_norm.weight"),
            attnKNorm:      try loadF32AsF16("\(p)attn_k_norm.weight"),
            ffnNorm:        try loadF32AsF16("\(p)ffn_norm.weight"),
            postFfn1Norm:   try loadF32AsF16("\(p)post_ffw_norm_1.weight"),
            preFfn2Norm:    try loadF32AsF16("\(p)pre_ffw_norm_2.weight"),
            postFfn2Norm:   try loadF32AsF16("\(p)post_ffw_norm_2.weight"),
            postFfnNorm:    try loadF32AsF16("\(p)post_ffw_norm.weight"),
            routerW:          try loadF32ToHalfTransposed2D("\(p)ffn_gate_inp.weight"),
            routerScale:      try loadF32Raw("\(p)ffn_gate_inp.scale"),
            expertScale:      try loadF32Raw("\(p)ffn_down_exps.scale"),
            layerOutputScale: try loadF32Raw("\(p)layer_output_scale.weight"),
            isFull: isFull, KV_H: lkv, HD: hd
        )
        layers.append(lw)
        if L == 0 || L == NUM_LAYERS - 1 || (L + 1) % 10 == 0 || isFull {
            let vNote = (attnVTuple == nil) ? " (V reused from K)" : " (V=\(attnVTuple!.1))"
            print("    layer \(L): \(isFull ? "full" : "slide") KV_H=\(lkv) HD=\(hd) Q=\(attnQFmt) FFN-down=\(ffnDownFmt) MoE-up=\(moeUpFmt) MoE-dn=\(moeDownFmt)\(vNote)")
        }
    }
    print(String(format: "  %d layers loaded+repacked in %.1f sec",
                 NUM_LAYERS, Date().timeIntervalSince(tLoad)))

    // ---------- Dequant tied token_embd → fp16 twice (embed table + transposed unembed) ----------
    // Format varies: Q4_K_M uses Q8_0 token_embd; Q5_K_M / Q6_K use Q6_K.
    // Both produce the same output (fp16 per-row embed table + transposed unembed).
    let tEmbed = Date()
    let embedInfo = try g.tensor("token_embd.weight")
    let eDin = embedInfo.shape[0], eDout = embedInfo.shape[1]
    precondition(eDin == HIDDEN && eDout == VOCAB, "embed shape mismatch")
    let embedTable = device.makeBuffer(length: VOCAB * HIDDEN * 2, options: .storageModeShared)!
    let unembedW   = device.makeBuffer(length: HIDDEN * VOCAB * 2, options: .storageModeShared)!
    let srcBase = g.base.advanced(by: embedInfo.dataOffset)
    let embedDp = embedTable.contents().assumingMemoryBound(to: Float16.self)
    let unembedDp = unembedW.contents().assumingMemoryBound(to: Float16.self)
    switch embedInfo.dtype {
    case .q8_0:
        let nbc = HIDDEN / 32
        let BLK = 34
        let colBytes = nbc * BLK
        for vo in 0..<VOCAB {
            let colBase = vo * colBytes
            for kb in 0..<nbc {
                let blkOff = colBase + kb * BLK
                let dFloat = Float(srcBase.load(fromByteOffset: blkOff, as: Float16.self))
                let baseD = kb * 32
                for pi in 0..<32 {
                    let qsByte = srcBase.load(fromByteOffset: blkOff + 2 + pi, as: Int8.self)
                    let val = Float16(dFloat * Float(qsByte))
                    embedDp[vo * HIDDEN + baseD + pi] = val
                    unembedDp[(baseD + pi) * VOCAB + vo] = val
                }
            }
        }
    case .q6_K:
        // Q6_K block: 210 bytes / 256 elts. Layout: ql[128], qh[64], scales[16] (i8), d (half).
        // Mirror of dequantize_q6_K_llama in MSL — produces 16 elts per il_orig in [0,16).
        let nbc = HIDDEN / 256
        let BLK = 210
        let colBytes = nbc * BLK
        for vo in 0..<VOCAB {
            let colBase = vo * colBytes
            for kb in 0..<nbc {
                let blkOff = colBase + kb * BLK
                let blk = srcBase.advanced(by: blkOff)
                let dAll = Float(blk.load(fromByteOffset: 208, as: Float16.self))
                for il in 0..<16 {
                    let qlBase = 32*(il/8) + 16*((il/2) & 1) + 8*(il & 1)
                    let qhBase = 16*(il/8) + 8*(il & 1)
                    let sc = Float(blk.load(fromByteOffset: 192 + (il % 2) + 2*(il/2), as: Int8.self))
                    let phase = (il/2) & 3
                    let kmask1: UInt32 = phase > 1 ? (phase > 2 ? 0xC0C0C0C0 : 0x30303030) : (phase > 0 ? 0x0C0C0C0C : 0x03030303)
                    let kmask2: UInt32 = phase > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F
                    let ml  = dAll * sc * 32.0
                    let dl0 = dAll * sc
                    let dl1 = dl0 / 256.0
                    let dl2 = dl0 / (256.0 * 256.0)
                    let dl3 = dl0 / (256.0 * 256.0 * 256.0)
                    let shr_h: UInt32 = phase > 2 ? 2 : 0
                    let shl_h: UInt32 = phase > 1 ? 0 : (phase > 0 ? 2 : 4)
                    let shr_l: UInt32 = phase > 1 ? 4 : 0
                    let baseD = kb * 256 + il * 16
                    for i in 0..<4 {
                        let low_lo  = UInt32(blk.load(fromByteOffset: (qlBase + 2*i) * 2, as: UInt16.self))
                        let low_hi  = UInt32(blk.load(fromByteOffset: (qlBase + 2*i + 1) * 2, as: UInt16.self))
                        let high_lo = UInt32(blk.load(fromByteOffset: 128 + (qhBase + 2*i) * 2, as: UInt16.self))
                        let high_hi = UInt32(blk.load(fromByteOffset: 128 + (qhBase + 2*i + 1) * 2, as: UInt16.self))
                        let low  = (low_lo  | (low_hi  << 16)) & kmask2
                        let high = (high_lo | (high_hi << 16)) & kmask1
                        let q = ((high << shl_h) >> shr_h) | (low >> shr_l)
                        let v0 = Float16(dl0 * Float(q & 0xFF) - ml)
                        let v1 = Float16(dl1 * Float(q & 0xFF00) - ml)
                        let v2 = Float16(dl2 * Float(q & 0xFF0000) - ml)
                        let v3 = Float16(dl3 * Float(q & 0xFF000000) - ml)
                        let kIdx0 = baseD + i*4
                        embedDp[vo * HIDDEN + kIdx0 + 0] = v0
                        embedDp[vo * HIDDEN + kIdx0 + 1] = v1
                        embedDp[vo * HIDDEN + kIdx0 + 2] = v2
                        embedDp[vo * HIDDEN + kIdx0 + 3] = v3
                        unembedDp[(kIdx0 + 0) * VOCAB + vo] = v0
                        unembedDp[(kIdx0 + 1) * VOCAB + vo] = v1
                        unembedDp[(kIdx0 + 2) * VOCAB + vo] = v2
                        unembedDp[(kIdx0 + 3) * VOCAB + vo] = v3
                    }
                }
            }
        }
    case .f16:
        // Source is already fp16 in GGUF row-major [VOCAB, HIDDEN] layout
        // (each vocab row holds HIDDEN halves contiguously). embedTable
        // wants [VOCAB, HIDDEN] → straight memcpy. unembedW wants the
        // transpose [HIDDEN, VOCAB] → element-wise transpose loop.
        let srcHalf = srcBase.assumingMemoryBound(to: Float16.self)
        memcpy(embedDp, srcHalf, VOCAB * HIDDEN * 2)
        for vo in 0..<VOCAB {
            let rowBase = vo * HIDDEN
            for k in 0..<HIDDEN {
                unembedDp[k * VOCAB + vo] = srcHalf[rowBase + k]
            }
        }
    default:
        fail("token_embd unsupported dtype \(embedInfo.dtype) — expected Q8_0, Q6_K, or F16")
    }
    print(String(format: "  token_embd \(embedInfo.dtype) → fp16 dequant in %.1f sec", Date().timeIntervalSince(tEmbed)))
    let outputNorm = try loadF32AsF16("output_norm.weight")

    // ---------- Per-layer paged K/V caches (zero-filled) ----------
    //
    // GUARD: each per-layer K (or V) buffer must stay strictly below
    // the empirically-measured hardware/driver corruption cliff. The
    // 2026-05-14 bisection (with full Metal validation enabled) found:
    //   - per-buffer ≤ 768 MB (= 0x3000_0000 bytes): correct results
    //   - per-buffer ≥ 784 MB:                       SILENT KV
    //     corruption (model emits coherent-English-but-unrelated
    //     text; Metal validation reports NOTHING).
    // Apple Silicon driver / TLB limit; no Swift code change fixes
    // it within the current single-buffer-per-layer layout. The
    // ONLY workaround is the per-layer buffer-split refactor (split
    // each layer's K/V into N device buffers + 2-level kernel
    // addressing), which is a substantial Metal kernel rewrite.
    //
    // Until that refactor lands, this guard fails-fast at startup
    // if a config would push past the cliff, preventing the silent-
    // corruption mode where the bridge serves wrong content without
    // anyone noticing. The threshold is set with margin: 768 MB hard
    // ceiling reduces to 752 MB enforced ceiling (16 MB safety
    // margin matching the 2026-05-14 production setting).
    // Per-chunk hardware cliff. The 2026-05-14 bisection found Metal
    // silently produces wrong-but-in-bounds reads above ~768 MB per
    // device buffer. With KV_NUM_CHUNKS=4 chunks per layer, each
    // chunk's K (or V) buffer must stay below this ceiling.
    let KV_BUFFER_HARD_LIMIT_BYTES = 768 * 1024 * 1024     // 768 MB tested-OK
    let KV_BUFFER_SAFETY_MARGIN    =  16 * 1024 * 1024     // 16 MB margin
    let KV_BUFFER_ENFORCED_CEILING = KV_BUFFER_HARD_LIMIT_BYTES - KV_BUFFER_SAFETY_MARGIN

    // Virtual-page-table KV cache, scatter-gather via argument buffers.
    //
    // Per-layer K (and V) cache lives across KV_NUM_CHUNKS device
    // buffers. The kernels see a single argument-buffer reference (the
    // KVChunks struct from kernels.swift's MSL); the GPU dereferences
    // chunks[chunk_idx] dynamically. Per-CB useResource() hints (set
    // by the dispatchers) let the GPU's residency table list only the
    // chunks actually touched.
    //
    // KV_NUM_CHUNKS must match the MSL #define in kernels.swift.
    let kvChunkPages = (TOTAL_PAGES + KV_NUM_CHUNKS - 1) / KV_NUM_CHUNKS

    let tCache = Date()
    var K_chunks: [[MTLBuffer]] = []
    var V_chunks: [[MTLBuffer]] = []
    var K_chunks_argbuf: [MTLBuffer] = []
    var V_chunks_argbuf: [MTLBuffer] = []
    K_chunks.reserveCapacity(NUM_LAYERS)
    V_chunks.reserveCapacity(NUM_LAYERS)
    K_chunks_argbuf.reserveCapacity(NUM_LAYERS)
    V_chunks_argbuf.reserveCapacity(NUM_LAYERS)
    var kvBytes = 0
    // Argument-buffer encoding via direct gpuAddress packing. The MSL
    // KVChunks struct is `device const half* chunks[N]` — just an array
    // of GPU virtual addresses. We allocate one arg-buf per layer (one
    // for K, one for V) sized for N pointers and write each chunk's
    // gpuAddress directly. Tier 2 argument buffers on M-series let the
    // kernel dereference these at runtime; useResource() in the
    // dispatcher tells the GPU which chunks are legal targets.
    // This avoids MTLArgumentEncoder entirely — simpler API surface,
    // no kernel-bufferIndex coupling.
    let argBufBytes = KV_NUM_CHUNKS * MemoryLayout<UInt64>.size
    for L in 0..<NUM_LAYERS {
        let lw = layers[L]
        let pg = PAGE
        // Per-chunk page count. The last chunk may carry fewer pages
        // if TOTAL_PAGES doesn't divide evenly. Per-chunk allocation
        // uses kvChunkPages (rounded up) so all chunks are the same
        // size — last chunk's tail simply stays zero / unused.
        let perChunkPages = kvChunkPages
        let elemsPerChunk = perChunkPages * pg * lw.KV_H * lw.HD
        let bytesPerChunk = elemsPerChunk * 2
        if bytesPerChunk > KV_BUFFER_ENFORCED_CEILING {
            let safePagesPerChunk = KV_BUFFER_ENFORCED_CEILING / (pg * lw.KV_H * lw.HD * 2)
            let safePoolPages = safePagesPerChunk * KV_NUM_CHUNKS - SCRATCH_STRIP
            fail("""
              KV-pool guard tripped at layer \(L) (\(lw.isFull ? "full" : "slide")):
                per-chunk bytes = \(bytesPerChunk) (\(bytesPerChunk / (1024*1024)) MB)
                ceiling         = \(KV_BUFFER_ENFORCED_CEILING) (\(KV_BUFFER_ENFORCED_CEILING / (1024*1024)) MB)
                hardware cliff  = ~768 MB per buffer (empirical 2026-05-14)
              SCRATCH_PAGE_BASE=\(SCRATCH_PAGE_BASE) too large for KV_NUM_CHUNKS=\(KV_NUM_CHUNKS).
              Reduce SCRATCH_PAGE_BASE to ≤ \(safePoolPages), OR bump KV_NUM_CHUNKS
              (must match between bootstrap.swift and kernels.swift's MSL #define).
              """)
        }
        // Allocate the N chunks WITHOUT eager memset. Critical at large
        // pool sizes: each `storageModeShared` buffer is virtual address
        // space until first touch. macOS lazy-commits physical pages on
        // first write, and freshly-faulted pages are zero by default
        // — so we get the same "zero-on-first-read" guarantee without
        // the up-front 100+ GB commitment that an explicit memset would
        // force. The engine's zeroPhysPageKV already does an explicit
        // memset when a phys page is first ASSIGNED to a session, which
        // is the only place we actually need zeros for correctness.
        //
        // Eagerly memsetting all chunks at startup was tripping the
        // pool=33000 wedge earlier: 240 chunks × 519 MB = 124 GB of
        // forced page commits on a 128 GB system → swap thrash, GPU
        // contention, bluetooth audio glitches. The lazy path means
        // actual RAM consumption scales with the working set
        // (~6 GB for batch=8 with typical max_tokens), not the
        // total addressable pool (108 GB).
        var Lchunks: [MTLBuffer] = []
        var Vchunks: [MTLBuffer] = []
        Lchunks.reserveCapacity(KV_NUM_CHUNKS)
        Vchunks.reserveCapacity(KV_NUM_CHUNKS)
        for _ in 0..<KV_NUM_CHUNKS {
            let kbuf = device.makeBuffer(length: bytesPerChunk, options: .storageModeShared)!
            let vbuf = device.makeBuffer(length: bytesPerChunk, options: .storageModeShared)!
            Lchunks.append(kbuf)
            Vchunks.append(vbuf)
            kvBytes += kbuf.length + vbuf.length
        }
        K_chunks.append(Lchunks)
        V_chunks.append(Vchunks)
        // Pack chunk gpuAddresses directly. Each arg buffer carries
        // N=KV_NUM_CHUNKS UInt64 entries; the kernel side declares
        // `device const half* chunks[N]` which matches this layout
        // on M-series under Tier 2 argument buffers.
        let kArgBuf = device.makeBuffer(length: argBufBytes, options: .storageModeShared)!
        let vArgBuf = device.makeBuffer(length: argBufBytes, options: .storageModeShared)!
        let kPtr = kArgBuf.contents().assumingMemoryBound(to: UInt64.self)
        let vPtr = vArgBuf.contents().assumingMemoryBound(to: UInt64.self)
        for i in 0..<KV_NUM_CHUNKS {
            kPtr[i] = Lchunks[i].gpuAddress
            vPtr[i] = Vchunks[i].gpuAddress
        }
        K_chunks_argbuf.append(kArgBuf)
        V_chunks_argbuf.append(vArgBuf)
    }
    print(String(format: "  per-layer K/V caches allocated: %.1f MB in %.2f sec (KV_NUM_CHUNKS=%d, chunk %d MB / cliff %d MB)",
                 Double(kvBytes) / (1024*1024), Date().timeIntervalSince(tCache),
                 KV_NUM_CHUNKS,
                 (K_chunks[0][0].length) / (1024*1024),
                 KV_BUFFER_HARD_LIMIT_BYTES / (1024*1024)))

    // ---------- Gemma-4 embed scale = sqrt(hidden) ----------
    let embedScaleBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
    embedScaleBuf.contents().assumingMemoryBound(to: Float.self)[0] = Float(HIDDEN).squareRoot()

    // ---------- Tokenizer bits from GGUF metadata ----------
    func readU32(_ key: String, default def: UInt32) -> UInt32 {
        guard let v = g.metadata[key] else { return def }
        if let x = v as? UInt32      { return x }
        if let x = v as? Int32       { return UInt32(x) }
        if let x = v as? UInt64      { return UInt32(x) }
        return def
    }
    let bosTokenId: UInt32 = readU32("tokenizer.ggml.bos_token_id", default: 2)
    // Gemma-4 sets eos_token_id to 106 (<end_of_turn>) since chat-tuned builds
    // use that as the effective stopping token. Keeps turn-boundary generation
    // behavior aligned with HF's generate() default.
    let eosTokenId: UInt32 = readU32("tokenizer.ggml.eos_token_id", default: 106)
    var addBosToken = true
    if let v = g.metadata["tokenizer.ggml.add_bos_token"] as? Bool { addBosToken = v }
    var vocabTokens: [String] = []
    if let tarr = g.metadata["tokenizer.ggml.tokens"] as? [Any] {
        vocabTokens.reserveCapacity(tarr.count)
        for t in tarr { vocabTokens.append((t as? String) ?? "") }
    }
    var merges: [String] = []
    if let marr = g.metadata["tokenizer.ggml.merges"] as? [Any] {
        merges.reserveCapacity(marr.count)
        for m in marr { merges.append((m as? String) ?? "") }
    }
    print(String(format: "  tokenizer: bos=%d eos=%d add_bos=%@ vocab=%d tokens merges=%d",
                 Int(bosTokenId), Int(eosTokenId), addBosToken ? "true" : "false",
                 vocabTokens.count, merges.count))

    // ---------- Sanity spot-check on layer 0's attnQ swizzle ----------
    // Compare the first block (col=0, kb=0) of the swizzled buffer against the
    // raw source — should be byte-identical regardless of format. Use 32 bytes
    // as a format-agnostic minimum (smaller than any quant block size).
    let L0 = layers[0]
    let rawQ = try g.makeMetalBuffer("blk.0.attn_q.weight", device: device)
    let rawSp = rawQ.contents()
    let swDp = L0.attnQ.contents()
    var match = true
    for byte in 0..<32 {
        let rawB = rawSp.load(fromByteOffset: byte, as: UInt8.self)
        let swB  = swDp.load(fromByteOffset: byte, as: UInt8.self)
        if rawB != swB { match = false; break }
    }
    print("  spot-check: L0 attn_q[col=0,kb=0,32B] \(match ? "✓ matches" : "✗ MISMATCH") post-swizzle (\(L0.attnQFormat))")

    print(String(format: "  == TOTAL load: %.2f sec ==", Date().timeIntervalSince(t0)))
    let w = LmWeights(layers: layers, embedTable: embedTable, unembedW: unembedW,
                      outputNorm: outputNorm, embedScaleBuf: embedScaleBuf,
                      K_chunks: K_chunks, V_chunks: V_chunks,
                      K_chunks_argbuf: K_chunks_argbuf, V_chunks_argbuf: V_chunks_argbuf,
                      kvChunkPages: kvChunkPages,
                      bosTokenId: bosTokenId, eosTokenId: eosTokenId,
                      addBosToken: addBosToken, vocabTokens: vocabTokens, merges: merges)
    installWeightResidencySet(w)
    return w
}
