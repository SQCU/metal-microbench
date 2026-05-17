// test_q5k_q6k_ar — direct numeric correctness test for the new
// Q5_K and Q6_K AR-decode GEMV kernels (dense + MoE) added in
// kernels.swift on 2026-05-01. Validates against CPU references
// (ported from q4k_mma_bench.swift) at random inputs; passes if
// RMSE is at fp16 floor (< 0.01).

import Metal
import Foundation

// ── Q5_K block layout (176 B / 256 elts) ────────────────────────────
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

func randomQ5kBlock(_ rng: inout SystemRandomNumberGenerator) -> [UInt8] {
    var blk = [UInt8](repeating: 0, count: BLK_Q5K_BYTES)
    let d    = Float16(Float.random(in: 0.005...0.05, using: &rng))
    let dmin = Float16(Float.random(in: 0.001...0.02, using: &rng))
    blk.withUnsafeMutableBytes { buf in
        buf.bindMemory(to: Float16.self)[0] = d
        buf.bindMemory(to: Float16.self)[1] = dmin
    }
    for i in 4..<BLK_Q5K_BYTES { blk[i] = UInt8.random(in: 0...255, using: &rng) }
    return blk
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

func buildQ5kBlob(N: Int, K: Int) -> [UInt8] {
    precondition(K % BLK_Q5K == 0)
    let blocksPerRow = K / BLK_Q5K
    var rng = SystemRandomNumberGenerator()
    var out = [UInt8](repeating: 0, count: N * blocksPerRow * BLK_Q5K_BYTES)
    for n in 0..<N {
        for kb in 0..<blocksPerRow {
            let off = (n * blocksPerRow + kb) * BLK_Q5K_BYTES
            let blk = randomQ5kBlock(&rng)
            for i in 0..<BLK_Q5K_BYTES { out[off + i] = blk[i] }
        }
    }
    return out
}

func cpuRefQ5K(X: [Float16], W: [UInt8], B: Int, K: Int, N: Int) -> [Float] {
    let blocksPerRow = K / BLK_Q5K
    var Y = [Float](repeating: 0, count: B * N)
    W.withUnsafeBufferPointer { wPtr in
        for b in 0..<B {
            for n in 0..<N {
                var acc: Float = 0
                for kb in 0..<blocksPerRow {
                    let blkOff = (n * blocksPerRow + kb) * BLK_Q5K_BYTES
                    let dq = dequantQ5kBlock(wPtr.baseAddress!.advanced(by: blkOff))
                    let kBase = kb * BLK_Q5K
                    for elem in 0..<BLK_Q5K {
                        acc += Float(X[b * K + kBase + elem]) * dq[elem]
                    }
                }
                Y[b * N + n] = acc
            }
        }
    }
    return Y
}

// ── Q6_K block layout (210 B / 256 elts) ────────────────────────────
let BLK_Q6K_BYTES = 210
let BLK_Q6K = 256

func randomQ6kBlock(_ rng: inout SystemRandomNumberGenerator) -> [UInt8] {
    var blk = [UInt8](repeating: 0, count: BLK_Q6K_BYTES)
    for i in 0..<192 { blk[i] = UInt8.random(in: 0...255, using: &rng) }
    for i in 192..<208 { blk[i] = UInt8(bitPattern: Int8.random(in: -8...8, using: &rng)) }
    let d = Float16(Float.random(in: 0.005...0.05, using: &rng))
    blk.withUnsafeMutableBytes { buf in
        let p = buf.baseAddress!.advanced(by: 208)
        p.bindMemory(to: Float16.self, capacity: 1).pointee = d
    }
    return blk
}

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

func buildQ6kBlob(N: Int, K: Int) -> [UInt8] {
    precondition(K % BLK_Q6K == 0)
    let blocksPerRow = K / BLK_Q6K
    var rng = SystemRandomNumberGenerator()
    var out = [UInt8](repeating: 0, count: N * blocksPerRow * BLK_Q6K_BYTES)
    for n in 0..<N {
        for kb in 0..<blocksPerRow {
            let off = (n * blocksPerRow + kb) * BLK_Q6K_BYTES
            let blk = randomQ6kBlock(&rng)
            for i in 0..<BLK_Q6K_BYTES { out[off + i] = blk[i] }
        }
    }
    return out
}

func cpuRefQ6K(X: [Float16], W: [UInt8], B: Int, K: Int, N: Int) -> [Float] {
    let blocksPerRow = K / BLK_Q6K
    var Y = [Float](repeating: 0, count: B * N)
    W.withUnsafeBufferPointer { wPtr in
        for b in 0..<B {
            for n in 0..<N {
                var acc: Float = 0
                for kb in 0..<blocksPerRow {
                    let blkOff = (n * blocksPerRow + kb) * BLK_Q6K_BYTES
                    let dq = dequantQ6kBlock(wPtr.baseAddress!.advanced(by: blkOff))
                    let kBase = kb * BLK_Q6K
                    for elem in 0..<BLK_Q6K {
                        acc += Float(X[b * K + kBase + elem]) * dq[elem]
                    }
                }
                Y[b * N + n] = acc
            }
        }
    }
    return Y
}

// ── v6 swizzle: [n, kb, byte] → [n_super=N/32, kb, col, byte] ───────
func swizzleBlob(_ src: [UInt8], N: Int, K: Int, blkBytes: Int, blkElems: Int) -> [UInt8] {
    precondition(N % 32 == 0)
    let nbc = K / blkElems
    let colBytes = nbc * blkBytes
    var dst = [UInt8](repeating: 0, count: src.count)
    src.withUnsafeBufferPointer { sp in
        dst.withUnsafeMutableBufferPointer { dp in
            let nSuper = N / 32
            for ns in 0..<nSuper {
                let srcColBase = ns * 32 * colBytes
                let dstSuperBase = ns * nbc * 32 * blkBytes
                for kb in 0..<nbc {
                    for col in 0..<32 {
                        let srcOff = srcColBase + col * colBytes + kb * blkBytes
                        let dstOff = dstSuperBase + kb * 32 * blkBytes + col * blkBytes
                        memcpy(dp.baseAddress!.advanced(by: dstOff),
                               sp.baseAddress!.advanced(by: srcOff),
                               blkBytes)
                    }
                }
            }
        }
    }
    return dst
}

// ── RMSE helper ─────────────────────────────────────────────────────
func rmse(_ a: [Float], _ b: [Float16]) -> Float {
    var s: Double = 0
    for i in 0..<a.count {
        let d = Double(a[i]) - Double(Float(b[i]))
        s += d * d
    }
    return Float(sqrt(s / Double(a.count)))
}

// fp16 partial-sum drift threshold scales with sqrt(K) — outputs of a
// K-element dot product accumulated in fp32-then-rounded-to-fp16 have
// expected RMSE ~ output_magnitude * eps_fp16 * sqrt(K), where output
// magnitude itself ~ sqrt(K) * input_std * weight_std. Use a relative
// threshold against the reference RMS so big-K cases aren't flagged.
func relRMSE(_ ref: [Float], _ gpu: [Float16]) -> Float {
    var refSq: Double = 0
    for v in ref { refSq += Double(v) * Double(v) }
    let refRMS = Float(sqrt(refSq / Double(ref.count)))
    let abs = rmse(ref, gpu)
    return refRMS > 1e-6 ? abs / refRMS : abs
}

// ── Buffer helpers ─────────────────────────────────────────────────
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

// ── Tests ──────────────────────────────────────────────────────────
struct TestResult { let name: String; let rmse: Float; let pass: Bool }

func testDenseQ5K(B: Int, K: Int, N: Int) -> TestResult {
    var rng = SystemRandomNumberGenerator()
    let X16: [Float16] = (0..<(B*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
    let W = buildQ5kBlob(N: N, K: K)
    let yRef = cpuRefQ5K(X: X16, W: W, B: B, K: K, N: N)

    let xBuf = makeBuf(X16)
    let wBuf = makeBuf(W)
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
    return TestResult(name: "dense_gemv_q5_K_v4 [B=\(B), K=\(K), N=\(N)]", rmse: r, pass: r < 0.01)
}

func testDenseQ6K(B: Int, K: Int, N: Int) -> TestResult {
    var rng = SystemRandomNumberGenerator()
    let X16: [Float16] = (0..<(B*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
    let W = buildQ6kBlob(N: N, K: K)
    let yRef = cpuRefQ6K(X: X16, W: W, B: B, K: K, N: N)

    let xBuf = makeBuf(X16)
    let wBuf = makeBuf(W)
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
    return TestResult(name: "dense_gemv_q6_K_v4 [B=\(B), K=\(K), N=\(N)]", rmse: r, pass: r < 0.01)
}

// MoE Q5_K v6 — single-expert test (E=1, all slots route to expert 0).
// Kernel uses `tok = slot_token[slot]` indexing (gate/up convention).
func testMoeQ5K(numSlots: Int, K: Int, N: Int) -> TestResult {
    var rng = SystemRandomNumberGenerator()
    let X16: [Float16] = (0..<(numSlots*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
    let W = buildQ5kBlob(N: N, K: K)
    let yRef = cpuRefQ5K(X: X16, W: W, B: numSlots, K: K, N: N)

    let Wsw = swizzleBlob(W, N: N, K: K, blkBytes: BLK_Q5K_BYTES, blkElems: BLK_Q5K)

    let slotTok: [UInt32] = (0..<UInt32(numSlots)).map { $0 }   // identity
    let activeExp: [UInt32] = [0]
    let groupStart: [UInt32] = [0, UInt32(numSlots)]

    let xBuf = makeBuf(X16)
    let stBuf = makeBuf(slotTok)
    let wBuf = makeBuf(Wsw)
    let aeBuf = makeBuf(activeExp)
    let gsBuf = makeBuf(groupStart)
    let yBuf = makeOutBuf(count: numSlots * N)
    var Kv = UInt32(K), Nv = UInt32(N)

    let p = pso("moe_gemv_q5_K_v6")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(xBuf, offset: 0, index: 0)
    enc.setBuffer(stBuf, offset: 0, index: 1)
    enc.setBuffer(wBuf, offset: 0, index: 2)
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

    let yGpu = readOut(yBuf, count: numSlots*N)
    let r = relRMSE(yRef, yGpu)
    return TestResult(name: "moe_gemv_q5_K_v6 [slots=\(numSlots), K=\(K), N=\(N), E=1]", rmse: r, pass: r < 0.01)
}

// MoE Q6_K v6 — single-expert. Kernel uses per-slot X (down convention),
// so reference X is [numSlots, K] and we compare directly.
func testMoeQ6K(numSlots: Int, K: Int, N: Int) -> TestResult {
    var rng = SystemRandomNumberGenerator()
    let X16: [Float16] = (0..<(numSlots*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
    let W = buildQ6kBlob(N: N, K: K)
    let yRef = cpuRefQ6K(X: X16, W: W, B: numSlots, K: K, N: N)

    let Wsw = swizzleBlob(W, N: N, K: K, blkBytes: BLK_Q6K_BYTES, blkElems: BLK_Q6K)

    // slot_token field is unused by Q6_K kernel (per-slot indexing) but the
    // kernel still binds it; supply identity to satisfy buffer arity.
    let slotTok: [UInt32] = (0..<UInt32(numSlots)).map { $0 }
    let activeExp: [UInt32] = [0]
    let groupStart: [UInt32] = [0, UInt32(numSlots)]

    let xBuf = makeBuf(X16)
    let stBuf = makeBuf(slotTok)
    let wBuf = makeBuf(Wsw)
    let aeBuf = makeBuf(activeExp)
    let gsBuf = makeBuf(groupStart)
    let yBuf = makeOutBuf(count: numSlots * N)
    var Kv = UInt32(K), Nv = UInt32(N)

    let p = pso("moe_gemv_q6_K_v6")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(xBuf, offset: 0, index: 0)
    enc.setBuffer(stBuf, offset: 0, index: 1)
    enc.setBuffer(wBuf, offset: 0, index: 2)
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

    let yGpu = readOut(yBuf, count: numSlots*N)
    let r = relRMSE(yRef, yGpu)
    return TestResult(name: "moe_gemv_q6_K_v6 [slots=\(numSlots), K=\(K), N=\(N), E=1]", rmse: r, pass: r < 0.01)
}

// ─── Prefill matmul tests ──────────────────────────────────────────
// Dense prefill kernels: same buffer convention as prefill_mm_q8_0_swiz
// (X [B, K], W swizzled, Y [B, N], shmem 10240). Output layout:
// Y[b * D_out + n].

func testDensePrefillQ5K(B: Int, K: Int, N: Int) -> TestResult {
    var rng = SystemRandomNumberGenerator()
    let X16: [Float16] = (0..<(B*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
    let W = buildQ5kBlob(N: N, K: K)
    let yRef = cpuRefQ5K(X: X16, W: W, B: B, K: K, N: N)
    let Wsw = swizzleBlob(W, N: N, K: K, blkBytes: BLK_Q5K_BYTES, blkElems: BLK_Q5K)

    let xBuf = makeBuf(X16)
    let wBuf = makeBuf(Wsw)
    let yBuf = makeOutBuf(count: B * N)
    var Bv = UInt32(B), Kv = UInt32(K), Nv = UInt32(N)

    let p = pso("prefill_mm_q5_K_swiz")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(xBuf, offset: 0, index: 0)
    enc.setBuffer(wBuf, offset: 0, index: 1)
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
    return TestResult(name: "prefill_mm_q5_K_swiz [B=\(B), K=\(K), N=\(N)]", rmse: r, pass: r < 0.01)
}

func testDensePrefillQ6K(B: Int, K: Int, N: Int) -> TestResult {
    var rng = SystemRandomNumberGenerator()
    let X16: [Float16] = (0..<(B*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
    let W = buildQ6kBlob(N: N, K: K)
    let yRef = cpuRefQ6K(X: X16, W: W, B: B, K: K, N: N)
    let Wsw = swizzleBlob(W, N: N, K: K, blkBytes: BLK_Q6K_BYTES, blkElems: BLK_Q6K)

    let xBuf = makeBuf(X16)
    let wBuf = makeBuf(Wsw)
    let yBuf = makeOutBuf(count: B * N)
    var Bv = UInt32(B), Kv = UInt32(K), Nv = UInt32(N)

    let p = pso("prefill_mm_q6_K_swiz")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(xBuf, offset: 0, index: 0)
    enc.setBuffer(wBuf, offset: 0, index: 1)
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
    return TestResult(name: "prefill_mm_q6_K_swiz [B=\(B), K=\(K), N=\(N)]", rmse: r, pass: r < 0.01)
}

// MoE Q5_K prefill — single-expert via active_exp=[0], group_start=[0, B],
// slot_token=identity (gate/up: X indexed by slot_token[s_flat]).
// Output: [num_slots, D_out] slot-flat.
func testMoePrefillQ5K(numSlots: Int, K: Int, N: Int) -> TestResult {
    var rng = SystemRandomNumberGenerator()
    let X16: [Float16] = (0..<(numSlots*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
    let W = buildQ5kBlob(N: N, K: K)
    let yRef = cpuRefQ5K(X: X16, W: W, B: numSlots, K: K, N: N)

    let Wsw = swizzleBlob(W, N: N, K: K, blkBytes: BLK_Q5K_BYTES, blkElems: BLK_Q5K)

    let slotTok: [UInt32] = (0..<UInt32(numSlots)).map { $0 }
    let activeExp: [UInt32] = [0]
    let groupStart: [UInt32] = [0, UInt32(numSlots)]

    let xBuf = makeBuf(X16)
    let stBuf = makeBuf(slotTok)
    let wBuf = makeBuf(Wsw)
    let aeBuf = makeBuf(activeExp)
    let gsBuf = makeBuf(groupStart)
    let yBuf = makeOutBuf(count: numSlots * N)
    var Kv = UInt32(K), Nv = UInt32(N)

    let p = pso("prefill_mm_id_q5_K_swiz")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(xBuf, offset: 0, index: 0)
    enc.setBuffer(stBuf, offset: 0, index: 1)
    enc.setBuffer(wBuf, offset: 0, index: 2)
    enc.setBuffer(aeBuf, offset: 0, index: 3)
    enc.setBuffer(gsBuf, offset: 0, index: 4)
    enc.setBuffer(yBuf, offset: 0, index: 5)
    enc.setBytes(&Kv, length: 4, index: 6)
    enc.setBytes(&Nv, length: 4, index: 7)
    enc.setThreadgroupMemoryLength(4096 + 2048 + 4096, index: 0)
    let gx = (numSlots + 31) / 32
    let gy = (N + 63) / 64
    enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1),  // gz=1, single expert
                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let yGpu = readOut(yBuf, count: numSlots*N)
    let r = relRMSE(yRef, yGpu)
    return TestResult(name: "prefill_mm_id_q5_K_swiz [slots=\(numSlots), K=\(K), N=\(N), E=1]", rmse: r, pass: r < 0.01)
}

// MoE Q6_K prefill — single-expert, per-slot X (down convention).
func testMoePrefillQ6K(numSlots: Int, K: Int, N: Int) -> TestResult {
    var rng = SystemRandomNumberGenerator()
    let X16: [Float16] = (0..<(numSlots*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
    let W = buildQ6kBlob(N: N, K: K)
    let yRef = cpuRefQ6K(X: X16, W: W, B: numSlots, K: K, N: N)

    let Wsw = swizzleBlob(W, N: N, K: K, blkBytes: BLK_Q6K_BYTES, blkElems: BLK_Q6K)

    let activeExp: [UInt32] = [0]
    let groupStart: [UInt32] = [0, UInt32(numSlots)]

    let xBuf = makeBuf(X16)
    let wBuf = makeBuf(Wsw)
    let aeBuf = makeBuf(activeExp)
    let gsBuf = makeBuf(groupStart)
    let yBuf = makeOutBuf(count: numSlots * N)
    var Kv = UInt32(K), Nv = UInt32(N)

    let p = pso("prefill_mm_id_q6_K_swiz")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(xBuf, offset: 0, index: 0)
    enc.setBuffer(wBuf, offset: 0, index: 1)
    enc.setBuffer(aeBuf, offset: 0, index: 2)
    enc.setBuffer(gsBuf, offset: 0, index: 3)
    enc.setBuffer(yBuf, offset: 0, index: 4)
    enc.setBytes(&Kv, length: 4, index: 5)
    enc.setBytes(&Nv, length: 4, index: 6)
    enc.setThreadgroupMemoryLength(4096 + 2048 + 4096, index: 0)
    let gx = (numSlots + 31) / 32
    let gy = (N + 63) / 64
    enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let yGpu = readOut(yBuf, count: numSlots*N)
    let r = relRMSE(yRef, yGpu)
    return TestResult(name: "prefill_mm_id_q6_K_swiz [slots=\(numSlots), K=\(K), N=\(N), E=1]", rmse: r, pass: r < 0.01)
}

// ── Btile zoo correctness: each format's btile kernel must match its v4 baseline ──
// The btile zoo and v4 single-template kernels compute the SAME matmul on
// the same Q5_K/Q6_K/Q5_1 bytes — different dispatch shape (4-SG split-K vs
// single-SG) but identical mathematical output. Validate fp16-floor agreement.

func testDenseBtileVsV4(formatName: String, B: Int, K: Int, N: Int,
                          buildBlob: (Int, Int) -> [UInt8],
                          blkBytes: Int, blkElems: Int,
                          v4Pso: String, btilePso: String) -> TestResult {
    var rng = SystemRandomNumberGenerator()
    let X16: [Float16] = (0..<(B*K)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
    let W = buildBlob(N, K)
    let Wsw = swizzleBlob(W, N: N, K: K, blkBytes: blkBytes, blkElems: blkElems)

    let xBuf = makeBuf(X16)
    let wStdBuf = makeBuf(W)         // standard layout for v4
    let wSwBuf = makeBuf(Wsw)        // swizzled for btile
    let yV4 = makeOutBuf(count: B * N)
    let yBT = makeOutBuf(count: B * N)
    var Bv = UInt32(B), Kv = UInt32(K), Nv = UInt32(N)

    // v4 dispatch (32 threads, grid (N/32, 1, 1), standard W layout)
    do {
        let p = pso(v4Pso)
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(wStdBuf, offset: 0, index: 1)
        enc.setBuffer(yV4, offset: 0, index: 2)
        enc.setBytes(&Bv, length: 4, index: 3)
        enc.setBytes(&Kv, length: 4, index: 4)
        enc.setBytes(&Nv, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: N/32, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
    }
    // btile dispatch (128 threads = 4 SGs split-K, swizzled W layout)
    do {
        let p = pso(btilePso)
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(wSwBuf, offset: 0, index: 1)
        enc.setBuffer(yBT, offset: 0, index: 2)
        enc.setBytes(&Kv, length: 4, index: 3)
        enc.setBytes(&Nv, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: N/32, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
    }

    // Both should agree to fp16 floor.
    let v4Out = readOut(yV4, count: B*N).map { Float($0) }
    let btOut = readOut(yBT, count: B*N)
    let r = relRMSE(v4Out, btOut)
    return TestResult(name: "\(formatName) btile_b\(B) vs v4 [B=\(B), K=\(K), N=\(N)]", rmse: r, pass: r < 0.01)
}

// ── Run all tests ──────────────────────────────────────────────────
print("device: \(device.name)")
print("MSL library: \(lib.functionNames.count) functions\n")

var results: [TestResult] = []

// Cover small + Gemma-shaped cases. Gemma-4 FFN is K=2304, N=11008.
results.append(testDenseQ5K(B: 1, K: 256, N: 32))
results.append(testDenseQ5K(B: 4, K: 512, N: 64))
results.append(testDenseQ5K(B: 8, K: 2304, N: 256))   // FFN-shaped slice

results.append(testDenseQ6K(B: 1, K: 256, N: 32))
results.append(testDenseQ6K(B: 4, K: 512, N: 64))
results.append(testDenseQ6K(B: 8, K: 2304, N: 256))

results.append(testMoeQ5K(numSlots: 1, K: 256, N: 32))
results.append(testMoeQ5K(numSlots: 8, K: 2304, N: 256))

results.append(testMoeQ6K(numSlots: 1, K: 256, N: 32))
results.append(testMoeQ6K(numSlots: 8, K: 2304, N: 256))

// Prefill matmul tests (large-batch path, uses simdgroup matmul).
// NR0=64 → N must be ≥64; NR1=32 → B must be ≥32.
results.append(testDensePrefillQ5K(B: 32, K: 256, N: 64))
results.append(testDensePrefillQ5K(B: 64, K: 2304, N: 256))
results.append(testDensePrefillQ6K(B: 32, K: 256, N: 64))
results.append(testDensePrefillQ6K(B: 64, K: 2304, N: 256))
results.append(testMoePrefillQ5K(numSlots: 32, K: 256, N: 64))
results.append(testMoePrefillQ5K(numSlots: 64, K: 2304, N: 256))
results.append(testMoePrefillQ6K(numSlots: 32, K: 256, N: 64))
results.append(testMoePrefillQ6K(numSlots: 64, K: 2304, N: 256))

// btile vs v4 cross-check — same matmul, two kernel structures must agree.
// Use Gemma-FFN-shaped K=2816 (nbc=11 for Q5_K/Q6_K, nbc=88 for Q5_1) so the
// test exercises both divisible (Q5_1) and non-divisible (Q5_K/Q6_K) cases.
for B in [1, 2, 4, 8] {
    results.append(testDenseBtileVsV4(formatName: "Q5_K", B: B, K: 2816, N: 256,
                                        buildBlob: buildQ5kBlob,
                                        blkBytes: BLK_Q5K_BYTES, blkElems: BLK_Q5K,
                                        v4Pso: "dense_gemv_q5_K_v4",
                                        btilePso: "dense_gemv_q5_K_btile_b\(B)"))
    results.append(testDenseBtileVsV4(formatName: "Q6_K", B: B, K: 2816, N: 256,
                                        buildBlob: buildQ6kBlob,
                                        blkBytes: BLK_Q6K_BYTES, blkElems: BLK_Q6K,
                                        v4Pso: "dense_gemv_q6_K_v4",
                                        btilePso: "dense_gemv_q6_K_btile_b\(B)"))
}

print("Results:")
var failed = 0
for r in results {
    let mark = r.pass ? "PASS" : "FAIL"
    print(String(format: "  [%@] %@ rmse=%.5f", mark, r.name, r.rmse))
    if !r.pass { failed += 1 }
}
print()
if failed == 0 {
    print("ALL \(results.count) TESTS PASSED")
    exit(0)
} else {
    print("\(failed)/\(results.count) tests FAILED")
    exit(1)
}
