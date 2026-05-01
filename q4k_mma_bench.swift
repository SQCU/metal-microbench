// q4k_mma_bench — A/B tournament harness for Q4_K cooperative-matmul
// prefill kernels.
//
// Embeds candidate kernels, runs each at production-shape sizes (Gemma-4
// dense + MoE expert), validates against CPU reference, times wall-ms,
// prints a leaderboard. Also runs llama.cpp's bench number (3315 t/s
// pp512 on the same GGUF/hardware) as an external reference target.
//
// Build:
//   swiftc -O q4k_mma_bench.swift -o q4k_mma_bench \
//      -framework Metal -framework Foundation
//
// Run:
//   ./q4k_mma_bench [shape...]   (defaults to a few canonical shapes)

import Metal
import Foundation

// ────────────────────────────────────────────────────────────────────
// Q4_K block layout (must match kernels.swift dequant_q4k_one).
// Block is 144 bytes:
//   half d         (2)
//   half dmin      (2)
//   uchar scales[12]
//   uchar qs[128]   (256 × 4-bit quants packed pair-wise)
// ────────────────────────────────────────────────────────────────────

let BLK_BYTES = 144
let BLK_K = 256

// CPU side: unpack the 12-byte scales table to (sc, mn) for a given
// sub-block (0..7) — mirrors device-side unpack_q4k_scales.
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

// CPU dequant: produce one fp32 weight from a Q4_K block.
@inline(__always)
func dequantQ4kOne(_ blk: UnsafePointer<UInt8>, _ e: Int) -> Float {
    let raw = UnsafeRawPointer(blk)
    let d    = Float(raw.load(fromByteOffset: 0, as: Float16.self))
    let dmin = Float(raw.load(fromByteOffset: 2, as: Float16.self))
    let scales  = blk + 4
    let qs      = blk + 16
    let sb   = e / 32
    let p    = e % 32
    let pair = sb / 2
    let isHi = (sb & 1) != 0
    let byte = qs[pair * 32 + p]
    let nib  = isHi ? (Int((byte >> 4) & 0xF)) : (Int(byte & 0xF))
    let (sc, mn) = unpackQ4kScales(scales, sb)
    return d * Float(sc) * Float(nib) - dmin * Float(mn)
}

// Generate a random Q4_K block. Returns 144 bytes.
func randomQ4kBlock(_ rng: inout SystemRandomNumberGenerator) -> [UInt8] {
    var blk = [UInt8](repeating: 0, count: BLK_BYTES)
    // d, dmin: small fp16 values
    let d   = Float16(Float.random(in: 0.005...0.05, using: &rng))
    let dmin = Float16(Float.random(in: 0.001...0.02, using: &rng))
    blk.withUnsafeMutableBytes { buf in
        buf.bindMemory(to: Float16.self)[0] = d
        buf.bindMemory(to: Float16.self)[1] = dmin
    }
    // scales: 12 random bytes
    for i in 4..<16 { blk[i] = UInt8.random(in: 0...255, using: &rng) }
    // qs: 128 random bytes
    for i in 16..<144 { blk[i] = UInt8.random(in: 0...255, using: &rng) }
    return blk
}

// Build a [N, K] Q4_K weight blob. K must be a multiple of 256.
func buildQ4kBlob(N: Int, K: Int) -> [UInt8] {
    precondition(K % BLK_K == 0, "K must be multiple of 256 for Q4_K")
    let blocksPerRow = K / BLK_K
    var rng = SystemRandomNumberGenerator()
    var out = [UInt8](repeating: 0, count: N * blocksPerRow * BLK_BYTES)
    for n in 0..<N {
        for kb in 0..<blocksPerRow {
            let off = (n * blocksPerRow + kb) * BLK_BYTES
            let blk = randomQ4kBlock(&rng)
            for i in 0..<BLK_BYTES { out[off + i] = blk[i] }
        }
    }
    return out
}

// ── Q8_0 helpers ────────────────────────────────────────────────────
let BLK_Q8_BYTES = 34
let BLK_Q8 = 32

func randomQ8Block(_ rng: inout SystemRandomNumberGenerator) -> [UInt8] {
    var blk = [UInt8](repeating: 0, count: BLK_Q8_BYTES)
    let d = Float16(Float.random(in: 0.005...0.05, using: &rng))
    blk.withUnsafeMutableBytes { buf in
        buf.bindMemory(to: Float16.self)[0] = d
    }
    for i in 2..<34 { blk[i] = UInt8(bitPattern: Int8.random(in: -127...127, using: &rng)) }
    return blk
}

func buildQ8Blob(N: Int, K: Int) -> [UInt8] {
    precondition(K % BLK_Q8 == 0, "K must be multiple of 32 for Q8_0")
    let blocksPerRow = K / BLK_Q8
    var rng = SystemRandomNumberGenerator()
    var out = [UInt8](repeating: 0, count: N * blocksPerRow * BLK_Q8_BYTES)
    for n in 0..<N {
        for kb in 0..<blocksPerRow {
            let off = (n * blocksPerRow + kb) * BLK_Q8_BYTES
            let blk = randomQ8Block(&rng)
            for i in 0..<BLK_Q8_BYTES { out[off + i] = blk[i] }
        }
    }
    return out
}

@inline(__always)
func dequantQ8One(_ blk: UnsafePointer<UInt8>, _ e: Int) -> Float {
    let raw = UnsafeRawPointer(blk)
    let d = Float(raw.load(fromByteOffset: 0, as: Float16.self))
    let qsBase = blk + 2
    let q = Int8(bitPattern: qsBase[e])
    return Float(q) * d
}

func cpuRefQ8(X: [Float16], W: [UInt8], B: Int, K: Int, N: Int) -> [Float] {
    let blocksPerRow = K / BLK_Q8
    var Y = [Float](repeating: 0, count: B * N)
    W.withUnsafeBufferPointer { wPtr in
        for b in 0..<B {
            for n in 0..<N {
                var acc: Float = 0
                for k in 0..<K {
                    let kb = k / BLK_Q8
                    let elem = k % BLK_Q8
                    let blkOff = (n * blocksPerRow + kb) * BLK_Q8_BYTES
                    let w = dequantQ8One(wPtr.baseAddress!.advanced(by: blkOff), elem)
                    acc += Float(X[b * K + k]) * w
                }
                Y[b * N + n] = acc
            }
        }
    }
    return Y
}

// v6 swizzle: standard [n, kb, byte] → swizzled [n_super, kb, col, byte].
// Mirror of repackQ80ToSwizzled in bootstrap.swift, in-Swift on a flat blob.
// Requires N % 32 == 0 (super-row size).
func swizzleQ8Blob(_ src: [UInt8], N: Int, K: Int) -> [UInt8] {
    precondition(N % 32 == 0, "N must be multiple of 32 for v6 swizzle")
    precondition(K % BLK_Q8 == 0, "K must be multiple of 32 for Q8_0")
    let nbc = K / BLK_Q8
    let nSuper = N / 32
    let colBytes = nbc * BLK_Q8_BYTES   // bytes per row in standard layout
    var dst = [UInt8](repeating: 0, count: src.count)
    src.withUnsafeBufferPointer { sp in
        dst.withUnsafeMutableBufferPointer { dp in
            for ns in 0..<nSuper {
                let srcColBase = ns * 32 * colBytes
                let dstSuperBase = ns * nbc * 32 * BLK_Q8_BYTES
                for kb in 0..<nbc {
                    for col in 0..<32 {
                        let srcOff = srcColBase + col * colBytes + kb * BLK_Q8_BYTES
                        let dstOff = dstSuperBase + kb * 32 * BLK_Q8_BYTES + col * BLK_Q8_BYTES
                        memcpy(dp.baseAddress!.advanced(by: dstOff),
                               sp.baseAddress!.advanced(by: srcOff),
                               BLK_Q8_BYTES)
                    }
                }
            }
        }
    }
    return dst
}

// ── Q5_1 helpers ────────────────────────────────────────────────────
let BLK_Q51 = 32
let BLK_Q51_BYTES = 24

func randomQ51Block(_ rng: inout SystemRandomNumberGenerator) -> [UInt8] {
    var blk = [UInt8](repeating: 0, count: BLK_Q51_BYTES)
    let d = Float16(Float.random(in: 0.005...0.05, using: &rng))
    let m = Float16(Float.random(in: -0.1...0.1, using: &rng))
    blk.withUnsafeMutableBytes { buf in
        let f16 = buf.bindMemory(to: Float16.self)
        f16[0] = d
        f16[1] = m
    }
    for i in 4..<24 { blk[i] = UInt8.random(in: 0...255, using: &rng) }
    return blk
}

// Mirrors MSL dequantize_q5_1_llama in CPU. Returns 32 dequantized weights
// in K-position order (il=0 produces K[0..15], il=1 produces K[16..31]).
func dequantQ51Block(_ blk: UnsafePointer<UInt8>) -> [Float] {
    let raw = UnsafeRawPointer(blk)
    let d = Float(raw.load(fromByteOffset: 0, as: Float16.self))
    let m = Float(raw.load(fromByteOffset: 2, as: Float16.self))
    let qh = raw.load(fromByteOffset: 4, as: UInt32.self)
    let qsU16 = (raw + 8).assumingMemoryBound(to: UInt16.self)
    var out = [Float](repeating: 0, count: 32)
    for il in 0...1 {
        let mask: UInt16 = il == 1 ? 0x00F0 : 0x000F
        let x_mv: UInt16 = il == 1 ? 4 : 0
        let gh_mv: UInt32 = il == 1 ? 12 : 0
        let gh_bk: UInt32 = il == 1 ? 0 : 4
        for i in 0..<8 {
            let xh_0 = ((qh >> (gh_mv + UInt32(2*i  ))) << gh_bk) & 0x10
            let xh_1 = ((qh >> (gh_mv + UInt32(2*i+1))) << gh_bk) & 0x10
            let qsi = qsU16[i]
            let x0 = Int32((qsi & mask) >> x_mv) | Int32(xh_0)
            let x1 = Int32(((qsi >> 8) & mask) >> x_mv) | Int32(xh_1)
            // Linear reg index: reg_f[i/2][2*(i%2)+n] → (i/2)*4 + 2*(i%2) + n
            let regBase = (i/2)*4 + 2*(i%2)
            out[il*16 + regBase + 0] = d * Float(x0) + m
            out[il*16 + regBase + 1] = d * Float(x1) + m
        }
    }
    return out
}

func buildQ51Blob(N: Int, K: Int) -> [UInt8] {
    precondition(K % BLK_Q51 == 0, "K must be multiple of 32 for Q5_1")
    let blocksPerRow = K / BLK_Q51
    var rng = SystemRandomNumberGenerator()
    var out = [UInt8](repeating: 0, count: N * blocksPerRow * BLK_Q51_BYTES)
    for n in 0..<N {
        for kb in 0..<blocksPerRow {
            let off = (n * blocksPerRow + kb) * BLK_Q51_BYTES
            let blk = randomQ51Block(&rng)
            for i in 0..<BLK_Q51_BYTES { out[off + i] = blk[i] }
        }
    }
    return out
}

// ── Per-expert blob builders (MoE) ──────────────────────────────────
// Layout: W[E][N][K/blk] block-bytes — same per-expert structure as
// llama.cpp's mul_mm_id src0 with nb02 = N * (K/blk) * blkBytes.

func buildQ4kPerExpertBlob(E: Int, N: Int, K: Int) -> [UInt8] {
    precondition(K % BLK_K == 0)
    let blocksPerRow = K / BLK_K
    var rng = SystemRandomNumberGenerator()
    var out = [UInt8](repeating: 0, count: E * N * blocksPerRow * BLK_BYTES)
    for e in 0..<E {
        let expBase = e * N * blocksPerRow * BLK_BYTES
        for n in 0..<N {
            for kb in 0..<blocksPerRow {
                let off = expBase + (n * blocksPerRow + kb) * BLK_BYTES
                let blk = randomQ4kBlock(&rng)
                for i in 0..<BLK_BYTES { out[off + i] = blk[i] }
            }
        }
    }
    return out
}

func buildQ51PerExpertBlob(E: Int, N: Int, K: Int) -> [UInt8] {
    precondition(K % BLK_Q51 == 0)
    let blocksPerRow = K / BLK_Q51
    var rng = SystemRandomNumberGenerator()
    var out = [UInt8](repeating: 0, count: E * N * blocksPerRow * BLK_Q51_BYTES)
    for e in 0..<E {
        let expBase = e * N * blocksPerRow * BLK_Q51_BYTES
        for n in 0..<N {
            for kb in 0..<blocksPerRow {
                let off = expBase + (n * blocksPerRow + kb) * BLK_Q51_BYTES
                let blk = randomQ51Block(&rng)
                for i in 0..<BLK_Q51_BYTES { out[off + i] = blk[i] }
            }
        }
    }
    return out
}

// ── v6 swizzle for per-expert blobs ─────────────────────────────────
// Standard [E][N][kb][byte] → swizzled [E][ns][kb][col][byte] where
// ns=N/32, col=N%32. Same layout as repackQ80ToSwizzled in bootstrap.swift,
// applied independently per expert.

private func swizzlePerExpert(_ src: [UInt8], E: Int, N: Int, K: Int,
                                blkBytes: Int, blkElems: Int) -> [UInt8] {
    precondition(N % 32 == 0)
    precondition(K % blkElems == 0)
    let nbc = K / blkElems
    let nSuper = N / 32
    let colBytes = nbc * blkBytes      // bytes per row in standard layout
    var dst = [UInt8](repeating: 0, count: src.count)
    src.withUnsafeBufferPointer { sp in
        dst.withUnsafeMutableBufferPointer { dp in
            for e in 0..<E {
                let srcExpBase = e * N * colBytes
                let dstExpBase = e * nSuper * nbc * 32 * blkBytes
                for ns in 0..<nSuper {
                    let srcColBase = srcExpBase + ns * 32 * colBytes
                    let dstSuperBase = dstExpBase + ns * nbc * 32 * blkBytes
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
    }
    return dst
}

func swizzleQ4kPerExpert(_ src: [UInt8], E: Int, N: Int, K: Int) -> [UInt8] {
    swizzlePerExpert(src, E: E, N: N, K: K, blkBytes: BLK_BYTES, blkElems: BLK_K)
}

func swizzleQ51PerExpert(_ src: [UInt8], E: Int, N: Int, K: Int) -> [UInt8] {
    swizzlePerExpert(src, E: E, N: N, K: K, blkBytes: BLK_Q51_BYTES, blkElems: BLK_Q51)
}

// ── Routing generator + tpe/ids tables ──────────────────────────────

struct Routing {
    let tpe: [UInt32]              // [E] tokens-per-expert
    let ids: [Int32]               // [E * idsStride] ids[e*stride + s] = encoded(token, slot_in_topk)
    let idsStride: Int             // = B * TOPK (safe upper bound)
    let maxNeh1: Int               // max over experts of tpe[e] — for grid sizing
    let assignments: [[Int]]       // [B][TOPK] expert index — for CPU reference
}

func genRouting(B: Int, E: Int, TOPK: Int) -> Routing {
    var rng = SystemRandomNumberGenerator()
    let idsStride = B * TOPK
    var assignments = [[Int]](repeating: [Int](repeating: 0, count: TOPK), count: B)
    for b in 0..<B {
        var picked = Set<Int>()
        while picked.count < TOPK {
            picked.insert(Int.random(in: 0..<E, using: &rng))
        }
        assignments[b] = Array(picked)
    }
    var tpe = [UInt32](repeating: 0, count: E)
    var ids = [Int32](repeating: 0, count: E * idsStride)
    for b in 0..<B {
        for s in 0..<TOPK {
            let e = assignments[b][s]
            let slotInExp = Int(tpe[e])
            ids[e * idsStride + slotInExp] = Int32(b * TOPK + s)
            tpe[e] += 1
        }
    }
    let maxNeh1 = Int(tpe.max() ?? 0)
    return Routing(tpe: tpe, ids: ids, idsStride: idsStride,
                   maxNeh1: maxNeh1, assignments: assignments)
}

// ── MoE CPU references ──────────────────────────────────────────────
// Y[b][s][n] = sum_k X[b][k] * W[expert_for(b,s)][n][k].
// Output flattened as b*TOPK*N + s*N + n (matches mul_mm_id dst layout).

func cpuRefMoE_Q4K(X: [Float16], W: [UInt8], routing: Routing,
                    B: Int, TOPK: Int, K: Int, N: Int) -> [Float] {
    let blocksPerRow = K / BLK_K
    let perExp = N * blocksPerRow * BLK_BYTES
    var Y = [Float](repeating: 0, count: B * TOPK * N)
    W.withUnsafeBufferPointer { wPtr in
        let base = wPtr.baseAddress!
        for b in 0..<B {
            for s in 0..<TOPK {
                let e = routing.assignments[b][s]
                for n in 0..<N {
                    var acc: Float = 0
                    for k in 0..<K {
                        let kb = k / BLK_K
                        let elem = k % BLK_K
                        let blkOff = e*perExp + (n * blocksPerRow + kb) * BLK_BYTES
                        let w = dequantQ4kOne(base.advanced(by: blkOff), elem)
                        acc += Float(X[b * K + k]) * w
                    }
                    Y[b * TOPK * N + s * N + n] = acc
                }
            }
        }
    }
    return Y
}

func cpuRefMoE_Q51(X: [Float16], W: [UInt8], routing: Routing,
                    B: Int, TOPK: Int, K: Int, N: Int) -> [Float] {
    let blocksPerRow = K / BLK_Q51
    let perExp = N * blocksPerRow * BLK_Q51_BYTES
    var Y = [Float](repeating: 0, count: B * TOPK * N)
    W.withUnsafeBufferPointer { wPtr in
        let base = wPtr.baseAddress!
        for b in 0..<B {
            for s in 0..<TOPK {
                let e = routing.assignments[b][s]
                for n in 0..<N {
                    var acc: Float = 0
                    // Walk full K, dequanting one block at a time and reusing
                    for kb in 0..<blocksPerRow {
                        let blkOff = e*perExp + (n * blocksPerRow + kb) * BLK_Q51_BYTES
                        let dq = dequantQ51Block(base.advanced(by: blkOff))
                        for elem in 0..<BLK_Q51 {
                            let k = kb * BLK_Q51 + elem
                            acc += Float(X[b * K + k]) * dq[elem]
                        }
                    }
                    Y[b * TOPK * N + s * N + n] = acc
                }
            }
        }
    }
    return Y
}

// CPU reference matmul: Y[B, N] = X[B, K] × W[N, K]^T (W stored as Q4_K).
// Returns Y as fp32.
func cpuRef(X: [Float16], W: [UInt8], B: Int, K: Int, N: Int) -> [Float] {
    let blocksPerRow = K / BLK_K
    var Y = [Float](repeating: 0, count: B * N)
    W.withUnsafeBufferPointer { wPtr in
        for b in 0..<B {
            for n in 0..<N {
                var acc: Float = 0
                for k in 0..<K {
                    let kb = k / BLK_K
                    let elem = k % BLK_K
                    let blkOff = (n * blocksPerRow + kb) * BLK_BYTES
                    let w = dequantQ4kOne(wPtr.baseAddress!.advanced(by: blkOff), elem)
                    acc += Float(X[b * K + k]) * w
                }
                Y[b * N + n] = acc
            }
        }
    }
    return Y
}

// ────────────────────────────────────────────────────────────────────
// Kernel source (subset of kernels.swift — only what V1 needs).
// We embed the Q4_K dequant helpers + the matmul kernel here so the
// bench doesn't need the full engine library.
// ────────────────────────────────────────────────────────────────────

let mslSource = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ────────────────────────────────────────────────────────────────────
// V_LLAMA — VERBATIM port of llama.cpp's kernel_mul_mm template body
// specialized for q4_K_f16 (block_q4_K source0, half source1, half dst).
//
// Source: /Users/mdot/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal
//         lines 9305-9614 (kernel_mul_mm template body)
//         lines 675-697  (dequantize_q4_K + get_scale_min_k4_just2)
//         lines 327      (block_q4_K struct)
//
// Specialization at source level (no template machinery):
//   S0 = half, S0_4x4 = half4x4, S0_8x8 = simdgroup_half8x8
//   S1 = half, S1_2x4 = half2x4, S1_8x8 = simdgroup_half8x8
//   block_q = block_q4_K, nl = 16 (QK_NL)
//   dequantize_func = dequantize_q4_K
//   T0 = float (kernel-internal accumulator; we cast to half at write)
//   T1 = half  (input X is fp16)
//
// We omit:
//   - GGML_METAL_HAS_TENSOR branches (Apple's matmul2d API; not on M5
//     until newer Metal)
//   - FC_mul_mm_bc_inp / FC_mul_mm_bc_out function-constants — we
//     hard-code the bounds-check-when-needed paths (matches their
//     non-bc-inp path for inputs; both bc-out / non-bc-out for output)
//
// Buffer adapter: replace `args.neXX/nbXX` with scalar params for the
// 2D non-broadcast case (no batched batch). i12=i13=0, r2=r3=1.
// ────────────────────────────────────────────────────────────────────

struct block_q4K_metal {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

static inline void dequantize_q4_K_llama(device const block_q4K_metal * xb, short il, thread half4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.h;
    const float min_= float(xb->dmin);
    const float dl  = d * sc[0];
    const float ml  = min_ * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = half(dl * float(q[i] & mask) - ml);
    }
}

// Buffer adapter: scalars in lieu of args struct.
//   ne00 = D_in (K)
//   ne0  = D_out (N, output dim)
//   ne1  = B_count (batch dim)
//   ne12 = 1 (no batched batch); i12=i13=0; r2=r3=1
//   nb01 = bytes per row of W = (D_in/256) * sizeof(block_q4K_metal) = (D_in/256) * 144
//   nb02 = nb03 = 0 (single batch, broadcasts irrelevant)
//   nb10 = sizeof(half) = 2
//   nb11 = D_in * sizeof(half) = D_in * 2
//   nb12 = nb13 = 0
kernel void kernel_mul_mm_q4K_llama(
    device const half*  X               [[buffer(0)]],   // src1 — [B, K] half
    device const uchar* W_q4k           [[buffer(1)]],   // src0 — [N, K/256] block_q4K bytes
    device half*        Y               [[buffer(2)]],   // dst  — [B, N] half
    constant uint& B_count              [[buffer(3)]],   // ne1
    constant uint& D_in                 [[buffer(4)]],   // ne00
    constant uint& D_out                [[buffer(5)]],   // ne0
    threadgroup char*  shmem            [[threadgroup(0)]],
    uint3 tgpig                         [[threadgroup_position_in_grid]],
    ushort tiitg                        [[thread_index_in_threadgroup]],
    ushort sgitg                        [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;   // 2
    constexpr int NL1 = NK/8;    // 4

    constexpr short nl = 16;     // QK_NL for q4_K

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    // Fields from args struct, materialized inline:
    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;
    const int ne1  = (int)B_count;
    const int ne12 = 1;
    const int r2   = 1;
    const int r3   = 1;
    const ulong nb01 = (ulong)(D_in / 256) * 144ul;
    const ulong nb02 = 0ul;
    const ulong nb03 = 0ul;
    const ulong nb10 = sizeof(half);
    const ulong nb11 = (ulong)D_in * sizeof(half);
    const ulong nb12 = 0ul;
    const ulong nb13 = 0ul;

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (ne1 - r1 < NR1) ? short(ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    const ulong offset0 = (i12/r2)*nb02 + (i13/r3)*nb03;
    const short offset1 = il0/nl;

    device const block_q4K_metal * x = (device const block_q4K_metal *)
        ((device const char *)W_q4k + nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8 * (tiitg % NL1);

    device const half * y = (device const half *) ((device const char *)X
        + nb13*i13
        + nb12*i12
        + nb11*(r1 + lr1)
        + nb10*iy);

    simdgroup_half8x8     ma[4];
    simdgroup_half8x8     mb[2];
    simdgroup_float8x8    mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        // ── Stage W (q4_K dequant) into sa ──────────────────────────
        // (using the !FC_mul_mm_bc_inp branch — assume input bounds OK.
        // Can revisit if we ever push K below 256.)
        {
            half4x4 temp_a;
            dequantize_q4_K_llama(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        // ── Stage X into sb (no-bounds-check vectorized variant) ────
        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = *((device half2x4 *) y);
        }

        // ── Advance il and pointers (verbatim) ──────────────────────
        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Cooperative matmul (verbatim) ───────────────────────────
        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    // ── Output store (verbatim, with half-cast at end) ──────────────
    // llama.cpp writes float to dst. We want half. So go through tg-mem
    // (always — simpler than the bounds-check branching) and cast at
    // device write.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem)
                                   + 32*(sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1) {
            device half * D = Y + r0 + (r1 + j) * ne0 + im * ne1 * ne0;
            threadgroup float * C = ((threadgroup float *) shmem) + (j * NR0);
            int i = 0;
            for (; i < nr0; i++) {
                D[i] = (half) C[i];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Q5_1 struct + dequant (verbatim from llama.cpp/ggml-metal.metal:512).
// 24 bytes / 32 weights:
//   half d         (2 B — scale)
//   half m         (2 B — min)
//   uchar qh[4]    (4 B — 5th bits, packed)
//   uchar qs[16]   (16 B — 4 lower bits, 2 packed per byte)
// nl = 2 (each call returns 16 elts; il ∈ {0,1}).
// ────────────────────────────────────────────────────────────────────

struct block_q5_1_metal {
    half     d;
    half     m;
    uchar    qh[4];
    uchar    qs[16];
};

static inline void dequantize_q5_1_llama(device const block_q5_1_metal * xb, short il, thread half4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4); // skip d,m,qh = 8 bytes = 4 u16
    const float d = float(xb->d);
    const float m = float(xb->m);
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);
        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }
    reg = (half4x4) reg_f;
}

// ────────────────────────────────────────────────────────────────────
// Q8_0 verbatim port — same template body, swap dequant + block struct.
// Reference: llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:573-585
// (dequantize_q8_0) and :10097 (kernel_mul_mm_q8_0_f16 instantiation).
//
// block_q8_0:
//   half d         (2 B — block scale)
//   int8 qs[32]    (32 B — int8-quantized weights, scaled by d)
// total: 34 B per 32 weights.
//
// Q8_0's nl = QK8_0/16 = 2 in the template (each call to dequantize_q8_0
// returns 16 weights from one half of the 32-weight block; il=0 reads
// first half, il=1 reads second half).
// ────────────────────────────────────────────────────────────────────

struct block_q8_0_metal {
    half   d;
    int8_t qs[32];
};

static inline void dequantize_q8_0_llama(device const block_q8_0_metal * xb, short il, thread half4x4 & reg) {
    device const int8_t * qs = xb->qs;
    const float d = float(xb->d);
    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = float(qs[i + 16*il]) * d;
    }
    reg = (half4x4) reg_f;
}

kernel void kernel_mul_mm_q8_0_llama(
    device const half*   X               [[buffer(0)]],
    device const uchar*  W_q8            [[buffer(1)]],
    device half*         Y               [[buffer(2)]],
    constant uint& B_count               [[buffer(3)]],
    constant uint& D_in                  [[buffer(4)]],
    constant uint& D_out                 [[buffer(5)]],
    threadgroup char*    shmem           [[threadgroup(0)]],
    uint3 tgpig                          [[threadgroup_position_in_grid]],
    ushort tiitg                         [[thread_index_in_threadgroup]],
    ushort sgitg                         [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 2;
    constexpr int BLK_Q8_BYTES = 34;

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;
    const int ne1  = (int)B_count;
    const int ne12 = 1;
    const int r2   = 1;
    const int r3   = 1;
    const ulong nb01 = (ulong)(D_in / 32) * BLK_Q8_BYTES;
    const ulong nb02 = 0ul;
    const ulong nb03 = 0ul;
    const ulong nb10 = sizeof(half);
    const ulong nb11 = (ulong)D_in * sizeof(half);
    const ulong nb12 = 0ul;
    const ulong nb13 = 0ul;

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (ne1 - r1 < NR1) ? short(ne1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    const ulong offset0 = (i12/r2)*nb02 + (i13/r3)*nb03;
    const short offset1 = il0/nl;

    device const block_q8_0_metal * x = (device const block_q8_0_metal *)
        ((device const char *)W_q8 + nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8 * (tiitg % NL1);

    device const half * y = (device const half *) ((device const char *)X
        + nb13*i13 + nb12*i12 + nb11*(r1 + lr1) + nb10*iy);

    simdgroup_half8x8     ma[4];
    simdgroup_half8x8     mb[2];
    simdgroup_float8x8    mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            dequantize_q8_0_llama(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = *((device half2x4 *) y);
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem)
                                   + 32*(sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1) {
            device half * D = Y + r0 + (r1 + j) * ne0 + im * ne1 * ne0;
            threadgroup float * C = ((threadgroup float *) shmem) + (j * NR0);
            int i = 0;
            for (; i < nr0; i++) {
                D[i] = (half) C[i];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Q8_0 v6-swizzled variant: same matmul template, W indexed for the
// production decode-time swizzle (super-block of 32 output rows
// interleaved by [kb, col, byte]). Lets prefill share the same single
// in-memory weight buffer as AR decode.
//
// Swizzle (from repackQ80ToSwizzled in bootstrap.swift):
//   block at logical (n, kb) → byte offset:
//     ns = n / 32, col = n % 32
//     swizzled_block_index = ns*(nbc*32) + kb*32 + col
//   where nbc = D_in / 32 (blocks per row in K-direction).
//
// In the matmul, advancing kb by +1 means stepping the block pointer
// by +32 (one block-stride within a super-row), not +1 like standard.
// ────────────────────────────────────────────────────────────────────

kernel void kernel_mul_mm_q8_0_swiz(
    device const half*   X               [[buffer(0)]],
    device const uchar*  W_q8            [[buffer(1)]],
    device half*         Y               [[buffer(2)]],
    constant uint& B_count               [[buffer(3)]],
    constant uint& D_in                  [[buffer(4)]],
    constant uint& D_out                 [[buffer(5)]],
    threadgroup char*    shmem           [[threadgroup(0)]],
    uint3 tgpig                          [[threadgroup_position_in_grid]],
    ushort tiitg                         [[thread_index_in_threadgroup]],
    ushort sgitg                         [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 2;

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;
    const int ne1  = (int)B_count;
    const ulong nb10 = sizeof(half);
    const ulong nb11 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (ne1 - r1 < NR1) ? short(ne1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const short offset1 = il0/nl;

    // ─── swizzled W base pointer ───
    const int   nbc    = ne00 / 32;        // blocks-per-row in K-direction
    const short row_g  = (short)(r0 + lr0);
    const short ns     = row_g >> 5;       // / 32
    const short col    = row_g & 31;       // % 32
    device const block_q8_0_metal * x = (device const block_q8_0_metal *)W_q8
        + (int)ns * nbc * 32                // skip past ns super-rows
        + (int)offset1 * 32                 // step to kb=offset1
        + col;                              // col within (ns, kb)

    const short iy = 8 * (tiitg % NL1);
    device const half * y = (device const half *) ((device const char *)X
        + nb11*(r1 + lr1) + nb10*iy);

    simdgroup_half8x8     ma[4];
    simdgroup_half8x8     mb[2];
    simdgroup_float8x8    mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            dequantize_q8_0_llama(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = *((device half2x4 *) y);
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        // swizzle: +1 kb step = +32 block stride (32 cols per kb in super-row)
        x  = (il < 2) ? x + 32 : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem)
                                   + 32*(sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1) {
            device half * D = Y + r0 + (r1 + j) * ne0 + im * ne1 * ne0;
            threadgroup float * C = ((threadgroup float *) shmem) + (j * NR0);
            int i = 0;
            for (; i < nr0; i++) {
                D[i] = (half) C[i];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// kernel_mul_mm_id — verbatim port of llama.cpp's MoE-routed matmul
// (ggml-metal.metal:9684). Same matmul body as kernel_mul_mm; routing
// adds:
//   • per-expert W base pointer (im selects expert)
//   • early-exit when no tokens routed to this expert
//   • src1 (X) read indirected through ids[im*IDS_STRIDE + r1+lr1]
//   • dst write scattered into [token][slot][N] by ids
//
// Inputs:
//   X      [B, K] half — same X used for every routed slot of a token
//                        (broadcast: nb11=0, nb12=K*2)
//   W      [E, N, K/blk] block_q* — per-expert weights, standard layout
//   Y      [B, TOPK, N] half — per-(token, slot) output rows
//   tpe    [E] uint32 — tokens routed to expert e (host-computed)
//   ids    [E, IDS_STRIDE] int32 — encoded slot index = token*TOPK + slot
//   B_count, D_in, D_out, E_count, TOPK, IDS_STRIDE — scalars
//
// Grid: (ceil(max_neh1 / NR1), ceil(N / NR0), E).  Threads: 128 / TG.
// ────────────────────────────────────────────────────────────────────

kernel void kernel_mul_mm_id_q4K_llama(
    device const half*     X            [[buffer(0)]],
    device const uchar*    W_q4k        [[buffer(1)]],
    device half*           Y            [[buffer(2)]],
    device const uint*     tpe          [[buffer(3)]],
    device const int*      ids          [[buffer(4)]],
    constant uint& B_count              [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    constant uint& E_count              [[buffer(8)]],
    constant uint& TOPK                 [[buffer(9)]],
    constant uint& IDS_STRIDE           [[buffer(10)]],
    threadgroup char*  shmem            [[threadgroup(0)]],
    uint3 tgpig                         [[threadgroup_position_in_grid]],
    ushort tiitg                        [[thread_index_in_threadgroup]],
    ushort tiisg                        [[thread_index_in_simdgroup]],
    ushort sgitg                        [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 16;     // QK_NL for q4_K

    const int im = tgpig.z;       // expert
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const int neh1 = (int)tpe[im];   // tokens routed to this expert
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;

    const ulong nb01 = (ulong)(D_in / 256) * 144ul;
    const ulong nb02 = (ulong)D_out * nb01;          // bytes per expert
    const ulong nb11 = 0ul;                          // slot stride (broadcast X)
    const ulong nb12 = (ulong)D_in * sizeof(half);   // token stride

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id  = ids[im * (int)IDS_STRIDE + r1 + lr1];
    const int i12 = id / (int)TOPK;        // token
    const int i11 = id % (int)TOPK;        // slot in top-k

    const ulong offset0 = (ulong)im * nb02;
    const short offset1 = il0 / nl;

    device const block_q4K_metal * x = (device const block_q4K_metal *)
        ((device const char *)W_q4k + nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8 * (tiitg % NL1);
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * i12 + nb11 * i11 + sizeof(half) * iy);

    simdgroup_half8x8     ma[4];
    simdgroup_half8x8     mb[2];
    simdgroup_float8x8    mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            dequantize_q4_K_llama(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = *((device half2x4 *) y);
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem)
                                   + 32*(sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter rows back to (token, slot) addresses.
    for (short j = sgitg; j < nr1; j += 4) {
        const int id_j = ids[im * (int)IDS_STRIDE + r1 + j];
        const int ide = id_j % (int)TOPK;
        const int idt = id_j / (int)TOPK;
        device half * D = Y + r0 + ide * ne0 + idt * (int)TOPK * ne0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// kernel_mul_mm_id_q5_1_llama — same body, Q5_1 dequant + nl=2.
// Q5_1 block is 24 B per 32 weights, so nb01 = (D_in/32) * 24.
// ────────────────────────────────────────────────────────────────────

kernel void kernel_mul_mm_id_q5_1_llama(
    device const half*     X            [[buffer(0)]],
    device const uchar*    W_q51        [[buffer(1)]],
    device half*           Y            [[buffer(2)]],
    device const uint*     tpe          [[buffer(3)]],
    device const int*      ids          [[buffer(4)]],
    constant uint& B_count              [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    constant uint& E_count              [[buffer(8)]],
    constant uint& TOPK                 [[buffer(9)]],
    constant uint& IDS_STRIDE           [[buffer(10)]],
    threadgroup char*  shmem            [[threadgroup(0)]],
    uint3 tgpig                         [[threadgroup_position_in_grid]],
    ushort tiitg                        [[thread_index_in_threadgroup]],
    ushort tiisg                        [[thread_index_in_simdgroup]],
    ushort sgitg                        [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 2;
    constexpr int BLK_Q51_BYTES = 24;

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const int neh1 = (int)tpe[im];
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;

    const ulong nb01 = (ulong)(D_in / 32) * BLK_Q51_BYTES;
    const ulong nb02 = (ulong)D_out * nb01;
    const ulong nb11 = 0ul;
    const ulong nb12 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id  = ids[im * (int)IDS_STRIDE + r1 + lr1];
    const int i12 = id / (int)TOPK;
    const int i11 = id % (int)TOPK;

    const ulong offset0 = (ulong)im * nb02;
    const short offset1 = il0 / nl;

    device const block_q5_1_metal * x = (device const block_q5_1_metal *)
        ((device const char *)W_q51 + nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8 * (tiitg % NL1);
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * i12 + nb11 * i11 + sizeof(half) * iy);

    simdgroup_half8x8     ma[4];
    simdgroup_half8x8     mb[2];
    simdgroup_float8x8    mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            dequantize_q5_1_llama(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = *((device half2x4 *) y);
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem)
                                   + 32*(sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id_j = ids[im * (int)IDS_STRIDE + r1 + j];
        const int ide = id_j % (int)TOPK;
        const int idt = id_j / (int)TOPK;
        device half * D = Y + r0 + ide * ne0 + idt * (int)TOPK * ne0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// kernel_mul_mm_id_q4K_swiz — MoE-routed Q4_K matmul on per-expert v6
// swizzled weights. Same routing semantics as the standard-layout port;
// only the W base pointer + kb-step advance differ.
//
// Per-expert layout: each expert's weights are independently v6-swizzled
// over the (N, K/256) plane. Per-expert size in blocks = N * (K/256).
// ────────────────────────────────────────────────────────────────────

kernel void kernel_mul_mm_id_q4K_swiz(
    device const half*     X            [[buffer(0)]],
    device const uchar*    W_q4k        [[buffer(1)]],
    device half*           Y            [[buffer(2)]],
    device const uint*     tpe          [[buffer(3)]],
    device const int*      ids          [[buffer(4)]],
    constant uint& B_count              [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    constant uint& E_count              [[buffer(8)]],
    constant uint& TOPK                 [[buffer(9)]],
    constant uint& IDS_STRIDE           [[buffer(10)]],
    threadgroup char*  shmem            [[threadgroup(0)]],
    uint3 tgpig                         [[threadgroup_position_in_grid]],
    ushort tiitg                        [[thread_index_in_threadgroup]],
    ushort tiisg                        [[thread_index_in_simdgroup]],
    ushort sgitg                        [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 16;

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const int neh1 = (int)tpe[im];
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;
    const int nbc  = ne00 / 256;                        // blocks-per-row in K
    const int blocks_per_expert = ne0 * nbc;            // = N * nbc

    const ulong nb11 = 0ul;                             // X broadcast across slots
    const ulong nb12 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id  = ids[im * (int)IDS_STRIDE + r1 + lr1];
    const int i12 = id / (int)TOPK;
    const int i11 = id % (int)TOPK;

    const short offset1 = il0 / nl;

    // Swizzled W base: skip prior experts, then super-row, kb step, col.
    const short row_g = (short)(r0 + lr0);
    const short ns    = row_g >> 5;
    const short col   = row_g & 31;
    device const block_q4K_metal * x = ((device const block_q4K_metal *)W_q4k)
        + im * blocks_per_expert
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

    const short iy = 8 * (tiitg % NL1);
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * i12 + nb11 * i11 + sizeof(half) * iy);

    simdgroup_half8x8     ma[4];
    simdgroup_half8x8     mb[2];
    simdgroup_float8x8    mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            dequantize_q4_K_llama(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = *((device half2x4 *) y);
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        // swizzle: kb advance = +32 blocks (one super-row stride for q4_K too)
        x  = (il < 2) ? x + 32 : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem)
                                   + 32*(sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id_j = ids[im * (int)IDS_STRIDE + r1 + j];
        const int ide = id_j % (int)TOPK;
        const int idt = id_j / (int)TOPK;
        device half * D = Y + r0 + ide * ne0 + idt * (int)TOPK * ne0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// kernel_mul_mm_id_q5_1_swiz — same swizzle adaptation, Q5_1 dequant.
// Per-expert size in blocks = N * (K/32). nl=2.
// ────────────────────────────────────────────────────────────────────

kernel void kernel_mul_mm_id_q5_1_swiz(
    device const half*     X            [[buffer(0)]],
    device const uchar*    W_q51        [[buffer(1)]],
    device half*           Y            [[buffer(2)]],
    device const uint*     tpe          [[buffer(3)]],
    device const int*      ids          [[buffer(4)]],
    constant uint& B_count              [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    constant uint& E_count              [[buffer(8)]],
    constant uint& TOPK                 [[buffer(9)]],
    constant uint& IDS_STRIDE           [[buffer(10)]],
    threadgroup char*  shmem            [[threadgroup(0)]],
    uint3 tgpig                         [[threadgroup_position_in_grid]],
    ushort tiitg                        [[thread_index_in_threadgroup]],
    ushort tiisg                        [[thread_index_in_simdgroup]],
    ushort sgitg                        [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 2;

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const int neh1 = (int)tpe[im];
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;
    const int nbc  = ne00 / 32;
    const int blocks_per_expert = ne0 * nbc;

    const ulong nb11 = 0ul;
    const ulong nb12 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id  = ids[im * (int)IDS_STRIDE + r1 + lr1];
    const int i12 = id / (int)TOPK;
    const int i11 = id % (int)TOPK;

    const short offset1 = il0 / nl;

    const short row_g = (short)(r0 + lr0);
    const short ns    = row_g >> 5;
    const short col   = row_g & 31;
    device const block_q5_1_metal * x = ((device const block_q5_1_metal *)W_q51)
        + im * blocks_per_expert
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

    const short iy = 8 * (tiitg % NL1);
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * i12 + nb11 * i11 + sizeof(half) * iy);

    simdgroup_half8x8     ma[4];
    simdgroup_half8x8     mb[2];
    simdgroup_float8x8    mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            dequantize_q5_1_llama(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = *((device half2x4 *) y);
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + 32 : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem)
                                   + 32*(sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id_j = ids[im * (int)IDS_STRIDE + r1 + j];
        const int ide = id_j % (int)TOPK;
        const int idt = id_j / (int)TOPK;
        device half * D = Y + r0 + ide * ne0 + idt * (int)TOPK * ne0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}
"""

// ────────────────────────────────────────────────────────────────────
// Bench driver.
// ────────────────────────────────────────────────────────────────────

enum WeightFormat { case q4k, q8_0, q8_0_swiz }

struct KernelVariant {
    let name: String
    let pipeline: MTLComputePipelineState
    let tileQ: Int   // batch rows per TG
    let tileO: Int   // output cols per TG
    let threads: Int
    var threadgroupMemoryBytes: Int = 0
    var format: WeightFormat = .q4k
}


func runShape(_ device: MTLDevice, _ queue: MTLCommandQueue,
              _ variant: KernelVariant,
              B: Int, K: Int, N: Int, iters: Int) -> (rmseRel: Float, msPerIter: Double, tflops: Double) {
    let pipeline = variant.pipeline
    let tileQ = variant.tileQ
    let tileO = variant.tileO
    let threads = variant.threads
    let isQ8 = (variant.format == .q8_0 || variant.format == .q8_0_swiz)
    let isSwiz = (variant.format == .q8_0_swiz)
    if isQ8 { precondition(K % BLK_Q8 == 0) } else { precondition(K % BLK_K == 0) }
    if isSwiz { precondition(N % 32 == 0, "swizzled Q8_0 requires N % 32 == 0") }

    var rng = SystemRandomNumberGenerator()
    var X = [Float16](repeating: 0, count: B * K)
    for i in 0..<X.count { X[i] = Float16(Float.random(in: -1...1, using: &rng) * 0.1) }
    // Always build standard layout (the CPU reference reads it this way).
    // If the variant wants swizzled, derive a swizzled copy for the GPU buffer.
    let W: [UInt8] = isQ8 ? buildQ8Blob(N: N, K: K) : buildQ4kBlob(N: N, K: K)
    let Wgpu: [UInt8] = isSwiz ? swizzleQ8Blob(W, N: N, K: K) : W

    let xBuf = device.makeBuffer(bytes: X, length: X.count * 2, options: .storageModeShared)!
    let wBuf = device.makeBuffer(bytes: Wgpu, length: Wgpu.count, options: .storageModeShared)!
    let yBuf = device.makeBuffer(length: B * N * 2, options: .storageModeShared)!

    var bC = UInt32(B), kC = UInt32(K), nC = UInt32(N)

    func dispatch() {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(yBuf, offset: 0, index: 2)
        enc.setBytes(&bC, length: 4, index: 3)
        enc.setBytes(&kC, length: 4, index: 4)
        enc.setBytes(&nC, length: 4, index: 5)
        if variant.threadgroupMemoryBytes > 0 {
            enc.setThreadgroupMemoryLength(variant.threadgroupMemoryBytes, index: 0)
        }
        let grid = MTLSize(width: (N + tileO - 1) / tileO,
                            height: (B + tileQ - 1) / tileQ, depth: 1)
        let tg   = MTLSize(width: threads, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
    }

    // Correctness only on small shapes (CPU ref is O(B*K*N)).
    var rmseRel: Float = 0
    if B * K * N <= 1_000_000 {
        dispatch()
        let yRef = isQ8 ? cpuRefQ8(X: X, W: W, B: B, K: K, N: N)
                        : cpuRef(X: X, W: W, B: B, K: K, N: N)
        let yPtr = yBuf.contents().bindMemory(to: Float16.self, capacity: B * N)
        var sumSq: Double = 0
        for i in 0..<B * N {
            let d = Float(yPtr[i]) - yRef[i]
            sumSq += Double(d * d)
        }
        let rmse = Float(sqrt(sumSq / Double(B * N)))
        let avgMag = (yRef.map { abs($0) }.reduce(0,+)) / Float(yRef.count)
        rmseRel = rmse / max(avgMag, 1e-6)
    }

    // Warmup
    for _ in 0..<2 { dispatch() }
    // Time
    let t0 = Date()
    for _ in 0..<iters { dispatch() }
    let dt = Date().timeIntervalSince(t0) / Double(iters)
    let flops = 2.0 * Double(B) * Double(K) * Double(N)
    let tflops = flops / dt / 1e12
    return (rmseRel, dt * 1000, tflops)
}

// ────────────────────────────────────────────────────────────────────
// MoE-routed matmul driver. Tests kernel_mul_mm_id_{q4K,q5_1}_llama.
// Y[B][TOPK][N] = sum_k X[B][K] * W[expert(b,s)][N][K].
// FLOP count: 2 * B * TOPK * K * N (every routed slot does a full matmul).
// ────────────────────────────────────────────────────────────────────

enum MoEFormat { case q4k, q51 }

func runShapeMoE(_ device: MTLDevice, _ queue: MTLCommandQueue,
                  pipeline: MTLComputePipelineState, name: String,
                  B: Int, K: Int, N: Int, E: Int, TOPK: Int,
                  format: MoEFormat, swizzled: Bool, iters: Int)
                 -> (rmseRel: Float, msPerIter: Double, tflops: Double)
{
    if format == .q4k {
        precondition(K % BLK_K == 0, "Q4_K requires K%256==0")
    } else {
        precondition(K % BLK_Q51 == 0, "Q5_1 requires K%32==0")
    }
    if swizzled { precondition(N % 32 == 0, "v6 swizzle requires N%32==0") }

    var rng = SystemRandomNumberGenerator()
    var X = [Float16](repeating: 0, count: B * K)
    for i in 0..<X.count { X[i] = Float16(Float.random(in: -1...1, using: &rng) * 0.1) }

    // Standard-layout W for the CPU reference; if the kernel wants swizzled,
    // produce a swizzled copy for the GPU buffer (CPU ref is unchanged).
    let W: [UInt8] = (format == .q4k)
        ? buildQ4kPerExpertBlob(E: E, N: N, K: K)
        : buildQ51PerExpertBlob(E: E, N: N, K: K)
    let Wgpu: [UInt8]
    if swizzled {
        Wgpu = (format == .q4k)
            ? swizzleQ4kPerExpert(W, E: E, N: N, K: K)
            : swizzleQ51PerExpert(W, E: E, N: N, K: K)
    } else {
        Wgpu = W
    }
    let routing = genRouting(B: B, E: E, TOPK: TOPK)

    let xBuf = device.makeBuffer(bytes: X, length: X.count * 2, options: .storageModeShared)!
    let wBuf = device.makeBuffer(bytes: Wgpu, length: Wgpu.count, options: .storageModeShared)!
    let yBuf = device.makeBuffer(length: B * TOPK * N * 2, options: .storageModeShared)!
    let tpeBuf = device.makeBuffer(bytes: routing.tpe, length: routing.tpe.count * 4,
                                    options: .storageModeShared)!
    let idsBuf = device.makeBuffer(bytes: routing.ids, length: routing.ids.count * 4,
                                    options: .storageModeShared)!

    var bC = UInt32(B), kC = UInt32(K), nC = UInt32(N)
    var eC = UInt32(E), tkC = UInt32(TOPK), strideC = UInt32(routing.idsStride)

    func dispatch() {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(yBuf, offset: 0, index: 2)
        enc.setBuffer(tpeBuf, offset: 0, index: 3)
        enc.setBuffer(idsBuf, offset: 0, index: 4)
        enc.setBytes(&bC, length: 4, index: 5)
        enc.setBytes(&kC, length: 4, index: 6)
        enc.setBytes(&nC, length: 4, index: 7)
        enc.setBytes(&eC, length: 4, index: 8)
        enc.setBytes(&tkC, length: 4, index: 9)
        enc.setBytes(&strideC, length: 4, index: 10)
        enc.setThreadgroupMemoryLength(4096 + 2048 + 4096, index: 0)
        // Grid: (ceil(maxNeh1/NR1=32), ceil(N/NR0=64), E).
        let gx = (routing.maxNeh1 + 31) / 32
        let gy = (N + 63) / 64
        let grid = MTLSize(width: max(gx, 1), height: gy, depth: E)
        let tg   = MTLSize(width: 128, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
    }

    var rmseRel: Float = 0
    if B * TOPK * K * N <= 4_000_000 {
        // Zero output — uninitialized slots (any unrouted (token,slot)) must
        // stay zero so the RMSE comparison against cpuRefMoE (which only
        // writes routed pairs) is meaningful.
        memset(yBuf.contents(), 0, B * TOPK * N * 2)
        dispatch()
        let yRef = (format == .q4k)
            ? cpuRefMoE_Q4K(X: X, W: W, routing: routing, B: B, TOPK: TOPK, K: K, N: N)
            : cpuRefMoE_Q51(X: X, W: W, routing: routing, B: B, TOPK: TOPK, K: K, N: N)
        let yPtr = yBuf.contents().bindMemory(to: Float16.self, capacity: B * TOPK * N)
        var sumSq: Double = 0
        for i in 0..<(B * TOPK * N) {
            let d = Float(yPtr[i]) - yRef[i]
            sumSq += Double(d * d)
        }
        let rmse = Float(sqrt(sumSq / Double(B * TOPK * N)))
        let avgMag = (yRef.map { abs($0) }.reduce(0,+)) / Float(B * TOPK * N)
        rmseRel = rmse / max(avgMag, 1e-6)
    }

    for _ in 0..<2 { dispatch() }
    let t0 = Date()
    for _ in 0..<iters { dispatch() }
    let dt = Date().timeIntervalSince(t0) / Double(iters)
    let flops = 2.0 * Double(B) * Double(TOPK) * Double(K) * Double(N)
    let tflops = flops / dt / 1e12
    return (rmseRel, dt * 1000, tflops)
}

// ────────────────────────────────────────────────────────────────────

FileHandle.standardError.write("ALIVE: pre-device\n".data(using:.utf8)!); guard let device = MTLCreateSystemDefaultDevice() else { exit(1) }; FileHandle.standardError.write("ALIVE: post-device\n".data(using:.utf8)!)
let queue = device.makeCommandQueue()!
let opts = MTLCompileOptions()
opts.languageVersion = .version3_0
let library: MTLLibrary
do { library = try device.makeLibrary(source: mslSource, options: opts) }
catch { FileHandle.standardError.write("compile error: \(error)\n".data(using:.utf8)!); exit(1) }

// Single canonical kernel: v6-swizzled Q8_0 simdgroup matmul (matches
// production prefill path). Standard-layout reference kernels remain in
// the MSL source as documented baselines but are not dispatched.
let q8sfn = library.makeFunction(name: "kernel_mul_mm_q8_0_swiz")!
let q8spl = try device.makeComputePipelineState(function: q8sfn)
let variants: [KernelVariant] = [
    KernelVariant(
        name: "swiz_q8_0",
        pipeline: q8spl,
        tileQ: 32, tileO: 64, threads: 128,
        threadgroupMemoryBytes: 4096 + 2048 + 4096,
        format: .q8_0_swiz),
]
FileHandle.standardError.write("ALIVE: variants=\(variants.count)\n".data(using:.utf8)!)

// MoE-routed mul_mm_id pipelines: v6-swizzled per-expert (canonical).
let moeQ4Kspl = try device.makeComputePipelineState(function:
    library.makeFunction(name: "kernel_mul_mm_id_q4K_swiz")!)
let moeQ51spl = try device.makeComputePipelineState(function:
    library.makeFunction(name: "kernel_mul_mm_id_q5_1_swiz")!)

// Default to small shapes for debugging; pass --full for production matched-B=32 sweep;
// pass --batch-sweep for the FFN-gate shape across multiple Q-batch sizes (the operating
// point that matters for long-prompt prefill); pass --moe for MoE-routed mul_mm_id sweep.
let runFull = CommandLine.arguments.contains("--full")
let runBatchSweep = CommandLine.arguments.contains("--batch-sweep")
let runMoE = CommandLine.arguments.contains("--moe")
let shapes: [(B: Int, K: Int, N: Int, iters: Int, label: String)]
if runBatchSweep {
    shapes = [
        // FFN gate/up — the dominant matmul shape — across Q-batches
        (32,  2304, 11008, 30, "ffn-gate B=32 "),
        (64,  2304, 11008, 30, "ffn-gate B=64 "),
        (128, 2304, 11008, 20, "ffn-gate B=128"),
        (256, 2304, 11008, 15, "ffn-gate B=256"),
        (512, 2304, 11008, 10, "ffn-gate B=512"),
        (1024, 2304, 11008, 6, "ffn-gate B=1024"),
        // FFN down — also dominant
        (32,  11008, 2304, 30, "ffn-down B=32 "),
        (256, 11008, 2304, 15, "ffn-down B=256"),
        (1024, 11008, 2304, 6, "ffn-down B=1024"),
    ]
} else if runFull {
    shapes = [
        (32, 256, 32, 5, "tiny correctness"),
        (32, 512, 64, 5, "small correctness"),
        (32, 2304, 2304, 50, "QKV-shape"),
        (32, 2304, 11008, 20, "FFN gate/up"),
        (32, 11008, 2304, 20, "FFN down"),
    ]
} else {
    shapes = [
        (16, 256, 64, 5, "tiny"),    // N≥64 so swizzled v6 (N%32==0) is also exercised
        (32, 256, 64, 5, "small"),
        (32, 512, 64, 5, "med"),
    ]
}

func pad(_ s: String, _ w: Int) -> String {
    if s.count >= w { return s }
    return s + String(repeating: " ", count: w - s.count)
}

if !runMoE {
    print("\n=== A/B tournament ===")
    print("\(pad("shape", 28)) \(pad("variant", 14)) \(pad("iter (ms)", 12)) \(pad("TFLOPS", 10)) \(pad("rmse-rel", 12)) \(pad("tok/s/matmul", 14))")
    print(String(repeating: "─", count: 100))
    for s in shapes {
        var best: (name: String, tflops: Double, ms: Double) = ("", 0, .infinity)
        for v in variants {
            FileHandle.standardError.write("ALIVE: shape=\(s.label) variant=\(v.name)\n".data(using:.utf8)!)
            let r = runShape(device, queue, v, B: s.B, K: s.K, N: s.N, iters: s.iters)
            let tps = Double(s.B) / (r.msPerIter / 1000.0)
            let label = "\(s.label) (\(s.B)×\(s.K)×\(s.N))"
            let rmse = r.rmseRel > 0 ? String(format: "%.4f", r.rmseRel) : "  -   "
            let msStr = String(format: "%.3f", r.msPerIter)
            let tflopsStr = String(format: "%.2f", r.tflops)
            let tpsStr = String(format: "%.0f", tps)
            print("\(pad(label, 28)) \(pad(v.name, 14)) \(pad(msStr, 12)) \(pad(tflopsStr, 10)) \(pad(rmse, 12)) \(pad(tpsStr, 14))")
            if r.tflops > best.tflops { best = (v.name, r.tflops, r.msPerIter) }
        }
        print("  → winner: \(best.name)  (\(String(format: "%.2f", best.tflops)) TFLOPS, \(String(format: "%.2f", best.ms)) ms/iter)\n")
    }
    print("--- llama.cpp pp512 on same GGUF/M5 = 3315 tok/s end-to-end (full forward pass) ---")
    print("--- our numbers above are PER-MATMUL; ~8 matmuls per layer × 30 layers ≈ 240 per token ---")
}

if runMoE {
    // MoE-routed mul_mm_id sweep — both standard and v6-swizzled layouts.
    // Production Gemma-4 MoE: E=128, TOPK=8, gate_up K=2304 N=1408, down K=704 N=2304.
    // For correctness checks we use small E to keep the CPU reference tractable.
    let moeShapes: [(B: Int, K: Int, N: Int, E: Int, TOPK: Int, iters: Int, label: String, fmt: MoEFormat)] = [
        (8, 256, 64, 4, 2, 5, "tiny Q4K MoE",  .q4k),
        (8, 256, 64, 4, 2, 5, "tiny Q5_1 MoE", .q51),
        (8,  2304, 1408, 8, 2, 10, "Q4K gate_up B=8 ", .q4k),
        (8,   704, 2304, 8, 2, 10, "Q5_1 down  B=8 ", .q51),
        (256, 2304, 1408, 128, 8, 6, "Q4K gate_up B=256", .q4k),
        (256,  704, 2304, 128, 8, 6, "Q5_1 down  B=256", .q51),
    ]
    print("\n=== MoE prefill kernels (canonical v6-swizzled mul_mm_id) ===")
    print("\(pad("shape", 32)) \(pad("variant", 16)) \(pad("iter (ms)", 12)) \(pad("TFLOPS", 10)) \(pad("rmse-rel", 12))")
    print(String(repeating: "─", count: 100))
    for s in moeShapes {
        let pl   = (s.fmt == .q4k) ? moeQ4Kspl : moeQ51spl
        let name = (s.fmt == .q4k) ? "mm_id_q4K_swiz" : "mm_id_q5_1_swiz"
        FileHandle.standardError.write("ALIVE: moe \(s.label)\n".data(using:.utf8)!)
        let r = runShapeMoE(device, queue, pipeline: pl, name: name,
                             B: s.B, K: s.K, N: s.N, E: s.E, TOPK: s.TOPK,
                             format: s.fmt, swizzled: true, iters: s.iters)
        let label = "\(s.label) (B=\(s.B) E=\(s.E) K=\(s.K) N=\(s.N))"
        let rmse = r.rmseRel > 0 ? String(format: "%.4f", r.rmseRel) : "  -   "
        let msStr = String(format: "%.3f", r.msPerIter)
        let tflopsStr = String(format: "%.2f", r.tflops)
        print("\(pad(label, 32)) \(pad(name, 16)) \(pad(msStr, 12)) \(pad(tflopsStr, 10)) \(pad(rmse, 12))")
    }
    print("--- MoE FLOP count includes all TOPK slots per token (active compute) ---")
}
