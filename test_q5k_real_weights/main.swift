// Validate the new Q5_K/Q6_K AR + prefill kernels against REAL weights from
// a llama-quantize-produced GGUF (Q5_K_M). Loads one tensor of each format,
// runs both the GPU kernel and a CPU dequant-then-matmul reference on the
// SAME byte-identical weights, and compares fp16 RMSE.
//
// This is a stronger correctness check than the random-weights test:
// real-model weight distributions may have outliers, signed zeros,
// near-zero scales, etc. that random inputs don't exercise.

import Metal
import Foundation

let q5kmPath = "/Users/mdot/models/gemma-4-a4b-quant-search/gemma-4-26B-A4B-it-Q5_K_M.gguf"

print("device: \(device.name)")
print("MSL library: \(lib.functionNames.count) functions")
print("loading Q5_K_M GGUF: \(q5kmPath)")
let g = try GGUFFile(q5kmPath)
print("  parsed OK — \(g.tensors.count) tensors")

// ── Q5_K dequant helpers (same as test_q5k_q6k_ar) ─────────────────
let BLK_Q5K_BYTES = 176
let BLK_Q5K = 256

@inline(__always)
func unpackQ4kScales(_ s: UnsafePointer<UInt8>, _ sb: Int) -> (UInt8, UInt8) {
    if sb < 4 {
        return (s[sb] & 0x3F, s[sb + 4] & 0x3F)
    } else {
        let k = sb - 4
        let sc = (s[k + 8] & 0x0F) | ((s[k + 0] & 0xC0) >> 2)
        let mn = (s[k + 8] >> 4)   | ((s[k + 4] & 0xC0) >> 2)
        return (sc, mn)
    }
}

func dequantQ5kSubtile(_ blk: UnsafePointer<UInt8>, _ il_orig: Int) -> [Float] {
    let raw = UnsafeRawPointer(blk)
    let dHalf    = Float(raw.load(fromByteOffset: 0, as: Float16.self))
    let dminHalf = Float(raw.load(fromByteOffset: 2, as: Float16.self))
    let scales   = blk + 4
    var qPtr     = blk + 48
    var qhPtr    = blk + 16

    let is_ = (il_orig/4) * 2
    qPtr  = qPtr  + 32 * (il_orig/4) + 16 * (il_orig & 1)
    qhPtr = qhPtr + 16 * (il_orig & 1)
    let ul: UInt8 = UInt8(1 << (il_orig / 2))
    let il = il_orig & 3

    let (sc, mn) = unpackQ4kScales(scales, is_ + (il / 2))
    let d  = il < 2 ? dHalf : dHalf / 16.0
    let m  = dminHalf
    let dl = d * Float(sc)
    let ml = m * Float(mn)

    let mask: UInt8 = il < 2 ? 0x0F : 0xF0
    let qh_val: Float = il < 2 ? 16.0 : 256.0

    var out = [Float](repeating: 0, count: 16)
    for i in 0..<16 {
        let lower = Float(qPtr[i] & mask)
        let high  = (qhPtr[i] & ul) != 0 ? qh_val : 0.0
        out[i] = dl * (lower + high) - ml
    }
    return out
}

func dequantQ5kBlock(_ blk: UnsafePointer<UInt8>) -> [Float] {
    var out = [Float](repeating: 0, count: 256)
    for il_orig in 0..<16 {
        let v = dequantQ5kSubtile(blk, il_orig)
        for i in 0..<16 { out[il_orig * 16 + i] = v[i] }
    }
    return out
}

// ── Q6_K dequant ────────────────────────────────────────────────────
let BLK_Q6K_BYTES = 210
let BLK_Q6K = 256

func dequantQ6kSubtile(_ blk: UnsafePointer<UInt8>, _ il_orig: Int) -> [Float] {
    let raw = UnsafeRawPointer(blk)
    let dAll = Float(raw.load(fromByteOffset: 208, as: Float16.self))

    @inline(__always) func ql_u16(_ idx: Int) -> UInt16 {
        return raw.load(fromByteOffset: idx * 2, as: UInt16.self)
    }
    @inline(__always) func qh_u16(_ idx: Int) -> UInt16 {
        return raw.load(fromByteOffset: 128 + idx * 2, as: UInt16.self)
    }

    let qlBase = 32*(il_orig/8) + 16*((il_orig/2) & 1) + 8*(il_orig & 1)
    let qhBase = 16*(il_orig/8) + 8*(il_orig & 1)
    let scIdx = (il_orig % 2) + 2 * (il_orig / 2)
    let sc = Float(Int8(bitPattern: blk[192 + scIdx]))
    let il = (il_orig / 2) & 3

    let kmask1: UInt32 = il > 1 ? (il > 2 ? 0xC0C0C0C0 : 0x30303030) : (il > 0 ? 0x0C0C0C0C : 0x03030303)
    let kmask2: UInt32 = il > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F
    let ml  = dAll * sc * 32.0
    let dl0 = dAll * sc
    let dl1 = dl0 / 256.0
    let dl2 = dl0 / (256.0 * 256.0)
    let dl3 = dl0 / (256.0 * 256.0 * 256.0)
    let shr_h: UInt32 = il > 2 ? 2 : 0
    let shl_h: UInt32 = il > 1 ? 0 : (il > 0 ? 2 : 4)
    let shr_l: UInt32 = il > 1 ? 4 : 0

    var out = [Float](repeating: 0, count: 16)
    for i in 0..<4 {
        let low  = (UInt32(ql_u16(qlBase + 2*i)) | (UInt32(ql_u16(qlBase + 2*i + 1)) << 16)) & kmask2
        let high = (UInt32(qh_u16(qhBase + 2*i)) | (UInt32(qh_u16(qhBase + 2*i + 1)) << 16)) & kmask1
        let q = ((high << shl_h) >> shr_h) | (low >> shr_l)
        out[i*4 + 0] = dl0 * Float(q & 0xFF)         - ml
        out[i*4 + 1] = dl1 * Float(q & 0xFF00)       - ml
        out[i*4 + 2] = dl2 * Float(q & 0xFF0000)     - ml
        out[i*4 + 3] = dl3 * Float(q & 0xFF000000)   - ml
    }
    return out
}

func dequantQ6kBlock(_ blk: UnsafePointer<UInt8>) -> [Float] {
    var out = [Float](repeating: 0, count: 256)
    for il_orig in 0..<16 {
        let v = dequantQ6kSubtile(blk, il_orig)
        for i in 0..<16 { out[il_orig * 16 + i] = v[i] }
    }
    return out
}

// ── v6 swizzle ──────────────────────────────────────────────────────
func swizzleBytes(_ src: UnsafePointer<UInt8>, _ dst: UnsafeMutablePointer<UInt8>,
                   N: Int, K: Int, blkBytes: Int, blkElems: Int) {
    let nbc = K / blkElems
    let colBytes = nbc * blkBytes
    let nSuper = N / 32
    for ns in 0..<nSuper {
        let srcColBase = ns * 32 * colBytes
        let dstSuperBase = ns * nbc * 32 * blkBytes
        for kb in 0..<nbc {
            for col in 0..<32 {
                let srcOff = srcColBase + col * colBytes + kb * blkBytes
                let dstOff = dstSuperBase + kb * 32 * blkBytes + col * blkBytes
                memcpy(dst.advanced(by: dstOff), src.advanced(by: srcOff), blkBytes)
            }
        }
    }
}

// ── CPU reference ──────────────────────────────────────────────────
func cpuRefQ5K(X: [Float16], W: UnsafePointer<UInt8>, B: Int, K: Int, N: Int) -> [Float] {
    let blocksPerRow = K / BLK_Q5K
    var Y = [Float](repeating: 0, count: B * N)
    for b in 0..<B {
        for n in 0..<N {
            var acc: Float = 0
            for kb in 0..<blocksPerRow {
                let blkOff = (n * blocksPerRow + kb) * BLK_Q5K_BYTES
                let dq = dequantQ5kBlock(W.advanced(by: blkOff))
                let kBase = kb * BLK_Q5K
                for elem in 0..<BLK_Q5K {
                    acc += Float(X[b * K + kBase + elem]) * dq[elem]
                }
            }
            Y[b * N + n] = acc
        }
    }
    return Y
}

func cpuRefQ6K(X: [Float16], W: UnsafePointer<UInt8>, B: Int, K: Int, N: Int) -> [Float] {
    let blocksPerRow = K / BLK_Q6K
    var Y = [Float](repeating: 0, count: B * N)
    for b in 0..<B {
        for n in 0..<N {
            var acc: Float = 0
            for kb in 0..<blocksPerRow {
                let blkOff = (n * blocksPerRow + kb) * BLK_Q6K_BYTES
                let dq = dequantQ6kBlock(W.advanced(by: blkOff))
                let kBase = kb * BLK_Q6K
                for elem in 0..<BLK_Q6K {
                    acc += Float(X[b * K + kBase + elem]) * dq[elem]
                }
            }
            Y[b * N + n] = acc
        }
    }
    return Y
}

// ── RMSE ────────────────────────────────────────────────────────────
func relRMSE(_ ref: [Float], _ gpu: [Float16]) -> Float {
    var refSq: Double = 0
    var s: Double = 0
    for i in 0..<ref.count {
        let r = Double(ref[i])
        refSq += r * r
        let d = r - Double(Float(gpu[i]))
        s += d * d
    }
    let rms = sqrt(refSq / Double(ref.count))
    let abs = sqrt(s / Double(ref.count))
    return rms > 1e-6 ? Float(abs / rms) : Float(abs)
}

// ── Helpers ─────────────────────────────────────────────────────────
func makeBuf<T>(_ arr: [T]) -> MTLBuffer {
    return arr.withUnsafeBufferPointer { ptr in
        device.makeBuffer(bytes: ptr.baseAddress!, length: arr.count * MemoryLayout<T>.stride, options: [])!
    }
}
func makeOutBuf(count: Int) -> MTLBuffer {
    return device.makeBuffer(length: count * MemoryLayout<Float16>.stride, options: [])!
}
func readOut(_ buf: MTLBuffer, count: Int) -> [Float16] {
    let p = buf.contents().bindMemory(to: Float16.self, capacity: count)
    return Array(UnsafeBufferPointer(start: p, count: count))
}

// ── Test: dense_gemv_q5_K_v4 against real attn_q.weight (Q5_K) ─────
do {
    let info = try g.tensor("blk.0.attn_q.weight")
    print("\nblk.0.attn_q.weight: dtype=\(info.dtype) shape=\(info.shape)")
    precondition(info.dtype == .q5_K, "attn_q expected q5_K in Q5_K_M")
    let K = info.shape[0]   // D_in
    let N = info.shape[1]   // D_out
    print("  K=\(K) N=\(N), bytes=\(info.byteSize)")

    let wPtr = g.base.advanced(by: info.dataOffset).assumingMemoryBound(to: UInt8.self)

    var rng = SystemRandomNumberGenerator()
    let B = 4
    let X16: [Float16] = (0..<(B*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }

    print("  computing CPU reference (B=\(B), \(B*N) outputs) ...")
    let t0 = Date()
    let yRef = cpuRefQ5K(X: X16, W: wPtr, B: B, K: K, N: N)
    print("    CPU reference in \(Int(Date().timeIntervalSince(t0)*1000)) ms")

    // GPU: dense_gemv_q5_K_v4 takes the STANDARD layout (not swizzled), since
    // the AR kernel uses col-major standard layout per its design.
    // Copy bytes into MTLBuffer.
    let wBuf = device.makeBuffer(length: info.byteSize, options: .storageModeShared)!
    memcpy(wBuf.contents(), wPtr, info.byteSize)
    let xBuf = makeBuf(X16)
    let yBuf = makeOutBuf(count: B * N)
    var Bv = UInt32(B), Kv = UInt32(K), Nv = UInt32(N)

    let p = pso("dense_gemv_q5_K_v4")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(xBuf, offset: 0, index: 0)
    enc.setBuffer(wBuf, offset: 0, index: 1)
    enc.setBuffer(yBuf, offset: 0, index: 2)
    enc.setBytes(&Bv, length: 4, index: 3)
    enc.setBytes(&Kv, length: 4, index: 4)
    enc.setBytes(&Nv, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: N/32, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let yGpu = readOut(yBuf, count: B*N)
    let r = relRMSE(yRef, yGpu)
    print(String(format: "  dense_gemv_q5_K_v4 vs CPU-Q5_K-dequant: rel-RMSE=%.5f %@",
                 r, r < 0.005 ? "PASS" : "FAIL"))
}

// ── Test: dense_gemv_q6_K_v4 against real attn_v.weight (Q6_K) ─────
do {
    let info = try g.tensor("blk.0.attn_v.weight")
    print("\nblk.0.attn_v.weight: dtype=\(info.dtype) shape=\(info.shape)")
    precondition(info.dtype == .q6_K, "attn_v expected q6_K in Q5_K_M")
    let K = info.shape[0]
    let N = info.shape[1]
    print("  K=\(K) N=\(N), bytes=\(info.byteSize)")

    let wPtr = g.base.advanced(by: info.dataOffset).assumingMemoryBound(to: UInt8.self)
    var rng = SystemRandomNumberGenerator()
    let B = 4
    let X16: [Float16] = (0..<(B*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }

    print("  computing CPU reference ...")
    let t0 = Date()
    let yRef = cpuRefQ6K(X: X16, W: wPtr, B: B, K: K, N: N)
    print("    CPU reference in \(Int(Date().timeIntervalSince(t0)*1000)) ms")

    let wBuf = device.makeBuffer(length: info.byteSize, options: .storageModeShared)!
    memcpy(wBuf.contents(), wPtr, info.byteSize)
    let xBuf = makeBuf(X16)
    let yBuf = makeOutBuf(count: B * N)
    var Bv = UInt32(B), Kv = UInt32(K), Nv = UInt32(N)

    let p = pso("dense_gemv_q6_K_v4")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(xBuf, offset: 0, index: 0)
    enc.setBuffer(wBuf, offset: 0, index: 1)
    enc.setBuffer(yBuf, offset: 0, index: 2)
    enc.setBytes(&Bv, length: 4, index: 3)
    enc.setBytes(&Kv, length: 4, index: 4)
    enc.setBytes(&Nv, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: N/32, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let yGpu = readOut(yBuf, count: B*N)
    let r = relRMSE(yRef, yGpu)
    print(String(format: "  dense_gemv_q6_K_v4 vs CPU-Q6_K-dequant: rel-RMSE=%.5f %@",
                 r, r < 0.005 ? "PASS" : "FAIL"))
}

// ── Test: prefill_mm_q5_K_swiz against real ffn_gate.weight (Q5_K) ─
do {
    let info = try g.tensor("blk.0.ffn_gate.weight")
    print("\nblk.0.ffn_gate.weight: dtype=\(info.dtype) shape=\(info.shape)")
    precondition(info.dtype == .q5_K, "ffn_gate expected q5_K in Q5_K_M")
    let K = info.shape[0]
    let N = info.shape[1]
    print("  K=\(K) N=\(N), bytes=\(info.byteSize)")

    let wPtr = g.base.advanced(by: info.dataOffset).assumingMemoryBound(to: UInt8.self)
    var rng = SystemRandomNumberGenerator()
    let B = 64
    let X16: [Float16] = (0..<(B*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }

    print("  computing CPU reference (B=\(B)) ...")
    let t0 = Date()
    let yRef = cpuRefQ5K(X: X16, W: wPtr, B: B, K: K, N: N)
    print("    CPU reference in \(Int(Date().timeIntervalSince(t0)*1000)) ms")

    // Swizzle the bytes into a new buffer (prefill kernel reads swizzled).
    let wBufSw = device.makeBuffer(length: info.byteSize, options: .storageModeShared)!
    swizzleBytes(wPtr, wBufSw.contents().assumingMemoryBound(to: UInt8.self),
                 N: N, K: K, blkBytes: BLK_Q5K_BYTES, blkElems: BLK_Q5K)

    let xBuf = makeBuf(X16)
    let yBuf = makeOutBuf(count: B * N)
    var Bv = UInt32(B), Kv = UInt32(K), Nv = UInt32(N)

    let p = pso("prefill_mm_q5_K_swiz")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(xBuf, offset: 0, index: 0)
    enc.setBuffer(wBufSw, offset: 0, index: 1)
    enc.setBuffer(yBuf, offset: 0, index: 2)
    enc.setBytes(&Bv, length: 4, index: 3)
    enc.setBytes(&Kv, length: 4, index: 4)
    enc.setBytes(&Nv, length: 4, index: 5)
    enc.setThreadgroupMemoryLength(4096 + 2048 + 4096, index: 0)
    let gx = (B + 31) / 32
    let gy = (N + 63) / 64
    enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let yGpu = readOut(yBuf, count: B*N)
    let r = relRMSE(yRef, yGpu)
    print(String(format: "  prefill_mm_q5_K_swiz vs CPU-Q5_K-dequant: rel-RMSE=%.5f %@",
                 r, r < 0.005 ? "PASS" : "FAIL"))
}

// ── Q8_0 dequant (for ffn_down_exps in Q5_K_M) ──────────────────────
let BLK_Q8_BYTES = 34
let BLK_Q8 = 32

func dequantQ8Block(_ blk: UnsafePointer<UInt8>) -> [Float] {
    let raw = UnsafeRawPointer(blk)
    let d = Float(raw.load(fromByteOffset: 0, as: Float16.self))
    let qs = (blk + 2).withMemoryRebound(to: Int8.self, capacity: 32) { $0 }
    var out = [Float](repeating: 0, count: 32)
    for i in 0..<32 { out[i] = d * Float(qs[i]) }
    return out
}

func cpuRefQ8(X: [Float16], W: UnsafePointer<UInt8>, B: Int, K: Int, N: Int) -> [Float] {
    let blocksPerRow = K / BLK_Q8
    var Y = [Float](repeating: 0, count: B * N)
    for b in 0..<B {
        for n in 0..<N {
            var acc: Float = 0
            for kb in 0..<blocksPerRow {
                let blkOff = (n * blocksPerRow + kb) * BLK_Q8_BYTES
                let dq = dequantQ8Block(W.advanced(by: blkOff))
                let kBase = kb * BLK_Q8
                for elem in 0..<BLK_Q8 {
                    acc += Float(X[b * K + kBase + elem]) * dq[elem]
                }
            }
            Y[b * N + n] = acc
        }
    }
    return Y
}

// ── Test: moe_gemv_q8_0_v6 + prefill_mm_id_q8_0_swiz on real ffn_down_exps ─
// Both kernels are per-slot/down convention. Single-expert harness:
// active_exp=[0], group_start=[0, B], W = expert-0 slice from real GGUF.
do {
    let info = try g.tensor("blk.0.ffn_down_exps.weight")
    print("\nblk.0.ffn_down_exps.weight: dtype=\(info.dtype) shape=\(info.shape)")
    precondition(info.dtype == .q8_0, "ffn_down_exps expected q8_0 in Q5_K_M")
    let K = info.shape[0]      // 704 — MOE_INT
    let N = info.shape[1]      // 2816 — HIDDEN
    let E = info.shape[2]      // 128 — n_experts

    // Slice expert 0 (offset 0, length = K * N / 32 * 34 bytes)
    let blocksPerCol = K / BLK_Q8
    let expertBytes = N * blocksPerCol * BLK_Q8_BYTES
    print("  K=\(K) N=\(N) E=\(E), expert-0 slice=\(expertBytes) bytes")
    let wPtr = g.base.advanced(by: info.dataOffset).assumingMemoryBound(to: UInt8.self)

    var rng = SystemRandomNumberGenerator()
    let B = 4   // single-expert "slots"
    let X16: [Float16] = (0..<(B*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }

    print("  computing CPU reference (B=\(B)) ...")
    let t0 = Date()
    let yRef = cpuRefQ8(X: X16, W: wPtr, B: B, K: K, N: N)
    print("    CPU reference in \(Int(Date().timeIntervalSince(t0)*1000)) ms")

    // ── AR-style: moe_gemv_q8_0_v6 with single-expert routing ─────
    do {
        let wBufSw = device.makeBuffer(length: expertBytes * E, options: .storageModeShared)!
        // Swizzle each expert's slice
        let dst = wBufSw.contents().assumingMemoryBound(to: UInt8.self)
        for e in 0..<E {
            swizzleBytes(wPtr.advanced(by: e * expertBytes),
                         dst.advanced(by: e * expertBytes),
                         N: N, K: K, blkBytes: BLK_Q8_BYTES, blkElems: BLK_Q8)
        }
        let slotTok: [UInt32] = (0..<UInt32(B)).map { $0 }
        let activeExp: [UInt32] = [0]
        let groupStart: [UInt32] = [0, UInt32(B)]
        let xBuf = makeBuf(X16)
        let stBuf = makeBuf(slotTok)
        let aeBuf = makeBuf(activeExp)
        let gsBuf = makeBuf(groupStart)
        let yBuf = makeOutBuf(count: B * N)
        var Kv = UInt32(K), Nv = UInt32(N)

        let p = pso("moe_gemv_q8_0_v6")
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(stBuf, offset: 0, index: 1)
        enc.setBuffer(wBufSw, offset: 0, index: 2)
        enc.setBuffer(aeBuf, offset: 0, index: 3)
        enc.setBuffer(gsBuf, offset: 0, index: 4)
        enc.setBuffer(yBuf, offset: 0, index: 5)
        enc.setBytes(&Kv, length: 4, index: 6)
        enc.setBytes(&Nv, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: N/32, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let yGpu = readOut(yBuf, count: B*N)
        let r = relRMSE(yRef, yGpu)
        print(String(format: "  moe_gemv_q8_0_v6 vs CPU-Q8-dequant: rel-RMSE=%.5f %@",
                     r, r < 0.005 ? "PASS" : "FAIL"))
    }

    // ── Prefill: prefill_mm_id_q8_0_swiz with single-expert routing ──
    do {
        let Bp = 32
        let Xp16: [Float16] = (0..<(Bp*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
        let yRefP = cpuRefQ8(X: Xp16, W: wPtr, B: Bp, K: K, N: N)

        let wBufSw = device.makeBuffer(length: expertBytes * E, options: .storageModeShared)!
        let dst = wBufSw.contents().assumingMemoryBound(to: UInt8.self)
        for e in 0..<E {
            swizzleBytes(wPtr.advanced(by: e * expertBytes),
                         dst.advanced(by: e * expertBytes),
                         N: N, K: K, blkBytes: BLK_Q8_BYTES, blkElems: BLK_Q8)
        }
        let activeExp: [UInt32] = [0]
        let groupStart: [UInt32] = [0, UInt32(Bp)]
        let xBuf = makeBuf(Xp16)
        let aeBuf = makeBuf(activeExp)
        let gsBuf = makeBuf(groupStart)
        let yBuf = makeOutBuf(count: Bp * N)
        var Kv = UInt32(K), Nv = UInt32(N)

        let p = pso("prefill_mm_id_q8_0_swiz")
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(wBufSw, offset: 0, index: 1)
        enc.setBuffer(aeBuf, offset: 0, index: 2)
        enc.setBuffer(gsBuf, offset: 0, index: 3)
        enc.setBuffer(yBuf, offset: 0, index: 4)
        enc.setBytes(&Kv, length: 4, index: 5)
        enc.setBytes(&Nv, length: 4, index: 6)
        enc.setThreadgroupMemoryLength(4096 + 2048 + 4096, index: 0)
        let gx = (Bp + 31) / 32
        let gy = (N + 63) / 64
        enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let yGpu = readOut(yBuf, count: Bp*N)
        let r = relRMSE(yRefP, yGpu)
        print(String(format: "  prefill_mm_id_q8_0_swiz vs CPU-Q8-dequant: rel-RMSE=%.5f %@",
                     r, r < 0.005 ? "PASS" : "FAIL"))
    }
}

print("\nDone — kernel-correctness validation complete on real Q5_K_M weights.")
