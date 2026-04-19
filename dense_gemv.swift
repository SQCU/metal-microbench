import Metal
import Foundation

// ===========================================================================
// Dense batched-decode GEMV — the low-batch regime for every non-MoE op.
//
// Target: QKV/out projection, shared-FFN gate/up/down, unembed — all of the
// form [B, D_in] × [D_in, D_out] = [B, D_out] with B ∈ {1..8}. At these
// shapes the per-TG MMA primitive cannot run (M < 8 row threshold), and the
// op is pure memory-bound on weight streaming. Each TG loops over k_in,
// fetching one W row (32 cols = 64 bytes coalesced), reusing it across ALL
// B batch rows. This forces W amortization across batch — the whole point.
//
// Two variants:
//   dense_gemv_fp16 : fp16 weights, scalar load
//   dense_gemv_q4   : Q4-packed weights, dequant-at-register via uint4 load
//
// Tests the batched-decode value prop: at B=4, time should be ~constant
// across B=1..4 (all DRAM-floor-bound, compute is free).
// ===========================================================================

let mslSource = """
#include <metal_stdlib>
using namespace metal;

// v4: grid.y=1 (no B split), multi-batch-per-TG with inner B loop + K-unroll.
// Amortizes W loads ACROSS batch rows within a single TG (not just via L2).
// Per k-block: read 8 W once, then for each b: 8 hidden + 8 MACs. B×-less
// total W DRAM traffic vs v3. Trades grid parallelism for W-sharing guarantee.
kernel void dense_gemv_fp16_v4(
    device const half* hidden       [[buffer(0)]],
    device const half* W            [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant uint& B                [[buffer(3)]],
    constant uint& D_in             [[buffer(4)]],
    constant uint& D_out            [[buffer(5)]],
    uint2 tg_pos                    [[threadgroup_position_in_grid]],
    uint2 lid2                      [[thread_position_in_threadgroup]]
) {
    constexpr uint MAX_B = 8;
    const uint n_block = tg_pos.x;
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    device const half* w_col = W + n;

    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;

    for (uint k = 0; k < D_in; k += 8) {
        half w0 = w_col[(k + 0) * D_out];
        half w1 = w_col[(k + 1) * D_out];
        half w2 = w_col[(k + 2) * D_out];
        half w3 = w_col[(k + 3) * D_out];
        half w4 = w_col[(k + 4) * D_out];
        half w5 = w_col[(k + 5) * D_out];
        half w6 = w_col[(k + 6) * D_out];
        half w7 = w_col[(k + 7) * D_out];
        for (uint b = 0; b < B; ++b) {
            device const half* hid = hidden + b * D_in + k;
            accs[b] += float(hid[0]) * float(w0) + float(hid[1]) * float(w1)
                     + float(hid[2]) * float(w2) + float(hid[3]) * float(w3)
                     + float(hid[4]) * float(w4) + float(hid[5]) * float(w5)
                     + float(hid[6]) * float(w6) + float(hid[7]) * float(w7);
        }
    }
    for (uint b = 0; b < B; ++b) {
        output[b * D_out + n] = half(accs[b]);
    }
}

// v5: split-K via multi-SG. 128 threads (4 SGs) per TG. Each SG processes
// K/4 of the K range, accumulates partial sum. Reduce via tg-mem, SG 0
// writes output. Goal: pack more outstanding DRAM requests into fewer TGs
// when grid would otherwise be under-filled (small D_out).
kernel void dense_gemv_fp16_v5(
    device const half* hidden       [[buffer(0)]],
    device const half* W            [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant uint& D_in             [[buffer(3)]],
    constant uint& D_out            [[buffer(4)]],
    uint2 tg_pos                    [[threadgroup_position_in_grid]],
    uint2 lid2                      [[thread_position_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint N_SPLITS = 4;
    const uint n_block = tg_pos.x;
    const uint b = tg_pos.y;
    const uint lid_sg = lid2.x % 32;
    const uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    device const half* hid_b = hidden + b * D_in;
    device const half* w_col = W + n;

    const uint k_per_sg = D_in / N_SPLITS;
    const uint k_begin = sg_id * k_per_sg;
    const uint k_end = k_begin + k_per_sg;

    float acc = 0.0f;
    for (uint k = k_begin; k < k_end; k += 8) {
        half w0 = w_col[(k + 0) * D_out];
        half w1 = w_col[(k + 1) * D_out];
        half w2 = w_col[(k + 2) * D_out];
        half w3 = w_col[(k + 3) * D_out];
        half w4 = w_col[(k + 4) * D_out];
        half w5 = w_col[(k + 5) * D_out];
        half w6 = w_col[(k + 6) * D_out];
        half w7 = w_col[(k + 7) * D_out];
        half h0 = hid_b[k + 0];
        half h1 = hid_b[k + 1];
        half h2 = hid_b[k + 2];
        half h3 = hid_b[k + 3];
        half h4 = hid_b[k + 4];
        half h5 = hid_b[k + 5];
        half h6 = hid_b[k + 6];
        half h7 = hid_b[k + 7];
        acc += float(h0) * float(w0) + float(h1) * float(w1)
             + float(h2) * float(w2) + float(h3) * float(w3)
             + float(h4) * float(w4) + float(h5) * float(w5)
             + float(h6) * float(w6) + float(h7) * float(w7);
    }

    // Reduce across 4 SGs: each SG writes its partial to tg-mem[sg][lid_sg],
    // then SG 0 reads all 4 and sums. partials sized [4 SGs × 32 lanes].
    threadgroup float partials[N_SPLITS][32];
    partials[sg_id][lid_sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0) {
        float total = partials[0][lid_sg] + partials[1][lid_sg]
                    + partials[2][lid_sg] + partials[3][lid_sg];
        output[b * D_out + n] = half(total);
    }
}

// W int8 + A fp16 + K-unroll-by-8. Half W BW vs fp16, full-precision
// activations. Int8 × fp16 compute path (cast to float).
kernel void dense_gemv_i8w_v3(
    device const half* hidden       [[buffer(0)]],
    device const char* W            [[buffer(1)]],   // int8, [D_in, D_out]
    device half* output             [[buffer(2)]],
    constant uint& D_in             [[buffer(3)]],
    constant uint& D_out            [[buffer(4)]],
    constant float& w_scale         [[buffer(5)]],
    uint2 tg_pos                    [[threadgroup_position_in_grid]],
    uint2 lid2                      [[thread_position_in_threadgroup]]
) {
    const uint n_block = tg_pos.x;
    const uint b = tg_pos.y;
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    device const half* hid = hidden + b * D_in;
    device const char* w_col = W + n;

    float acc = 0.0f;
    for (uint k = 0; k < D_in; k += 8) {
        char w0 = w_col[(k + 0) * D_out];
        char w1 = w_col[(k + 1) * D_out];
        char w2 = w_col[(k + 2) * D_out];
        char w3 = w_col[(k + 3) * D_out];
        char w4 = w_col[(k + 4) * D_out];
        char w5 = w_col[(k + 5) * D_out];
        char w6 = w_col[(k + 6) * D_out];
        char w7 = w_col[(k + 7) * D_out];
        half h0 = hid[k + 0];
        half h1 = hid[k + 1];
        half h2 = hid[k + 2];
        half h3 = hid[k + 3];
        half h4 = hid[k + 4];
        half h5 = hid[k + 5];
        half h6 = hid[k + 6];
        half h7 = hid[k + 7];
        acc += float(h0) * float(w0) + float(h1) * float(w1)
             + float(h2) * float(w2) + float(h3) * float(w3)
             + float(h4) * float(w4) + float(h5) * float(w5)
             + float(h6) * float(w6) + float(h7) * float(w7);
    }
    output[b * D_out + n] = half(acc * w_scale);
}

// W int8 + A int8 + K-unroll-by-8. Both W and activations int8. Uses int32
// accumulator within an 8-way K-unroll chunk, applies per-tensor scales at
// the end. Real impl would use per-32-block fp16 activation scales; this
// microbench uses one global scale for each side to isolate the BW/compute
// characteristics. NOTE: this crosses the activation-quantization line —
// numerical divergence from fp16 training, see project_metal_inference_scope.
kernel void dense_gemv_i8_i8_v3(
    device const char* hidden       [[buffer(0)]],   // int8, [B, D_in]
    device const char* W            [[buffer(1)]],   // int8, [D_in, D_out]
    device half* output             [[buffer(2)]],
    constant uint& D_in             [[buffer(3)]],
    constant uint& D_out            [[buffer(4)]],
    constant float& w_scale         [[buffer(5)]],
    constant float& a_scale         [[buffer(6)]],
    uint2 tg_pos                    [[threadgroup_position_in_grid]],
    uint2 lid2                      [[thread_position_in_threadgroup]]
) {
    const uint n_block = tg_pos.x;
    const uint b = tg_pos.y;
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    device const char* hid = hidden + b * D_in;
    device const char* w_col = W + n;

    // int32 accumulator across the full K. int8×int8 → int16, summed to int32.
    // D_in=2816 × max value 127×127 = ~45M, far inside int32 range.
    int acc_i = 0;
    for (uint k = 0; k < D_in; k += 8) {
        int w0 = int(w_col[(k + 0) * D_out]);
        int w1 = int(w_col[(k + 1) * D_out]);
        int w2 = int(w_col[(k + 2) * D_out]);
        int w3 = int(w_col[(k + 3) * D_out]);
        int w4 = int(w_col[(k + 4) * D_out]);
        int w5 = int(w_col[(k + 5) * D_out]);
        int w6 = int(w_col[(k + 6) * D_out]);
        int w7 = int(w_col[(k + 7) * D_out]);
        int h0 = int(hid[k + 0]);
        int h1 = int(hid[k + 1]);
        int h2 = int(hid[k + 2]);
        int h3 = int(hid[k + 3]);
        int h4 = int(hid[k + 4]);
        int h5 = int(hid[k + 5]);
        int h6 = int(hid[k + 6]);
        int h7 = int(hid[k + 7]);
        acc_i += h0*w0 + h1*w1 + h2*w2 + h3*w3 + h4*w4 + h5*w5 + h6*w6 + h7*w7;
    }
    output[b * D_out + n] = half(float(acc_i) * w_scale * a_scale);
}

// Variant V3: grid.y = B + K-unroll by 8.
// Same grid as V2 but the K loop issues 8 W loads + 8 hidden loads per iter.
// Compiler should schedule the 8 loads in parallel (no data dependency
// between them), so each thread has 8 outstanding DRAM requests in flight
// instead of 1 — multiplies request parallelism, hides cacheline latency.
// Directly targets the small-D_out latency-bound regime where V2 stalls.
// Valid when D_in % 8 == 0 (true for every Gemma-4 D_in: 2816, 2112, 704).
kernel void dense_gemv_fp16_v3(
    device const half* hidden       [[buffer(0)]],
    device const half* W            [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant uint& D_in             [[buffer(3)]],
    constant uint& D_out            [[buffer(4)]],
    uint2 tg_pos                    [[threadgroup_position_in_grid]],
    uint2 lid2                      [[thread_position_in_threadgroup]]
) {
    const uint n_block = tg_pos.x;
    const uint b = tg_pos.y;
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    device const half* hid = hidden + b * D_in;
    device const half* w_col = W + n;

    float acc = 0.0f;
    for (uint k = 0; k < D_in; k += 8) {
        half w0 = w_col[(k + 0) * D_out];
        half w1 = w_col[(k + 1) * D_out];
        half w2 = w_col[(k + 2) * D_out];
        half w3 = w_col[(k + 3) * D_out];
        half w4 = w_col[(k + 4) * D_out];
        half w5 = w_col[(k + 5) * D_out];
        half w6 = w_col[(k + 6) * D_out];
        half w7 = w_col[(k + 7) * D_out];
        half h0 = hid[k + 0];
        half h1 = hid[k + 1];
        half h2 = hid[k + 2];
        half h3 = hid[k + 3];
        half h4 = hid[k + 4];
        half h5 = hid[k + 5];
        half h6 = hid[k + 6];
        half h7 = hid[k + 7];
        acc += float(h0) * float(w0) + float(h1) * float(w1)
             + float(h2) * float(w2) + float(h3) * float(w3)
             + float(h4) * float(w4) + float(h5) * float(w5)
             + float(h6) * float(w6) + float(h7) * float(w7);
    }
    output[b * D_out + n] = half(acc);
}

kernel void dense_gemv_q4_v3(
    device const half* hidden       [[buffer(0)]],
    device const uint4* W_q4        [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant uint& D_in             [[buffer(3)]],
    constant uint& D_out            [[buffer(4)]],
    constant float& q4_scale        [[buffer(5)]],
    uint2 tg_pos                    [[threadgroup_position_in_grid]],
    uint2 lid2                      [[thread_position_in_threadgroup]]
) {
    const uint n_block = tg_pos.x;
    const uint b = tg_pos.y;
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    device const half* hid = hidden + b * D_in;
    const uint k_stride = D_out / 32;
    device const uint4* w_col = W_q4 + n_block;
    const uint word_idx = lid / 8;
    const uint nib_shift = (lid % 8) * 4;

    float acc = 0.0f;
    for (uint k = 0; k < D_in; k += 8) {
        // Cluster 8 uint4 reads + 8 hidden reads. Each uint4 is one cacheline's
        // worth (16 bytes) at stride D_out/2 bytes = distinct DRAM request.
        uint4 p0 = w_col[(k + 0) * k_stride];
        uint4 p1 = w_col[(k + 1) * k_stride];
        uint4 p2 = w_col[(k + 2) * k_stride];
        uint4 p3 = w_col[(k + 3) * k_stride];
        uint4 p4 = w_col[(k + 4) * k_stride];
        uint4 p5 = w_col[(k + 5) * k_stride];
        uint4 p6 = w_col[(k + 6) * k_stride];
        uint4 p7 = w_col[(k + 7) * k_stride];
        half h0 = hid[k + 0];
        half h1 = hid[k + 1];
        half h2 = hid[k + 2];
        half h3 = hid[k + 3];
        half h4 = hid[k + 4];
        half h5 = hid[k + 5];
        half h6 = hid[k + 6];
        half h7 = hid[k + 7];
        // Per-lane nibble extract from each pack
        #define EXTRACT(p) float(int(((word_idx == 0) ? (p).x : (word_idx == 1) ? (p).y : (word_idx == 2) ? (p).z : (p).w) >> nib_shift & 0xF) - 8) * q4_scale
        float w0 = EXTRACT(p0);
        float w1 = EXTRACT(p1);
        float w2 = EXTRACT(p2);
        float w3 = EXTRACT(p3);
        float w4 = EXTRACT(p4);
        float w5 = EXTRACT(p5);
        float w6 = EXTRACT(p6);
        float w7 = EXTRACT(p7);
        #undef EXTRACT
        acc += float(h0) * w0 + float(h1) * w1 + float(h2) * w2 + float(h3) * w3
             + float(h4) * w4 + float(h5) * w5 + float(h6) * w6 + float(h7) * w7;
    }
    output[b * D_out + n] = half(acc);
}

// Variant V2: grid.y = B, one TG per (n_block, batch_row).
// Motivation: at small D_out, grid.x alone doesn't produce enough TGs to
// hide DRAM cacheline latency (each k-iter's load is a fresh cacheline at
// stride D_out*2 bytes). Multiplying by B gets more TGs in flight. W
// sharing across b-sibling TGs is now opportunistic via L2 rather than
// forced via in-kernel loops, but avoids the per-lane dependent-MAC chain.
kernel void dense_gemv_fp16_v2(
    device const half* hidden       [[buffer(0)]],   // [B, D_in]
    device const half* W            [[buffer(1)]],   // [D_in, D_out]
    device half* output             [[buffer(2)]],   // [B, D_out]
    constant uint& D_in             [[buffer(3)]],
    constant uint& D_out            [[buffer(4)]],
    uint2 tg_pos                    [[threadgroup_position_in_grid]],
    uint2 lid2                      [[thread_position_in_threadgroup]]
) {
    const uint n_block = tg_pos.x;
    const uint b = tg_pos.y;
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    device const half* hid = hidden + b * D_in;
    device const half* w_col = W + n;

    float acc = 0.0f;
    for (uint k = 0; k < D_in; ++k) {
        acc += float(hid[k]) * float(w_col[k * D_out]);
    }
    output[b * D_out + n] = half(acc);
}

kernel void dense_gemv_q4_v2(
    device const half* hidden       [[buffer(0)]],
    device const uint4* W_q4        [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant uint& D_in             [[buffer(3)]],
    constant uint& D_out            [[buffer(4)]],
    constant float& q4_scale        [[buffer(5)]],
    uint2 tg_pos                    [[threadgroup_position_in_grid]],
    uint2 lid2                      [[thread_position_in_threadgroup]]
) {
    const uint n_block = tg_pos.x;
    const uint b = tg_pos.y;
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    device const half* hid = hidden + b * D_in;
    const uint k_stride = D_out / 32;
    device const uint4* w_col = W_q4 + n_block;
    const uint word_idx = lid / 8;
    const uint nib_shift = (lid % 8) * 4;

    float acc = 0.0f;
    for (uint k = 0; k < D_in; ++k) {
        uint4 pack = w_col[k * k_stride];
        uint word = (word_idx == 0) ? pack.x :
                    (word_idx == 1) ? pack.y :
                    (word_idx == 2) ? pack.z : pack.w;
        int nib = int((word >> nib_shift) & 0xF);
        float w_val = float(nib - 8) * q4_scale;
        acc += float(hid[k]) * w_val;
    }
    output[b * D_out + n] = half(acc);
}

// fp16 dense GEMV. One TG per 32-col n-slab. 32 lanes each own 1 output col
// per batch row; loops B rows × D_in k-iters. W streamed once per TG.
kernel void dense_gemv_fp16(
    device const half* hidden       [[buffer(0)]],   // [B, D_in]
    device const half* W            [[buffer(1)]],   // [D_in, D_out]
    device half* output             [[buffer(2)]],   // [B, D_out]
    constant uint& B                [[buffer(3)]],
    constant uint& D_in             [[buffer(4)]],
    constant uint& D_out            [[buffer(5)]],
    uint2 tg_pos                    [[threadgroup_position_in_grid]],
    uint2 lid2                      [[thread_position_in_threadgroup]]
) {
    constexpr uint MAX_B = 8;
    const uint n_block = tg_pos.x;
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;

    device const half* w_col = W + n;
    for (uint k = 0; k < D_in; ++k) {
        float w = float(w_col[k * D_out]);
        for (uint b = 0; b < B; ++b) {
            accs[b] += float(hidden[b * D_in + k]) * w;
        }
    }
    for (uint b = 0; b < B; ++b) {
        output[b * D_out + n] = half(accs[b]);
    }
}

// Q4 dense GEMV. W packed as uint4 (32 nibbles per uint4 = 32 consecutive N).
// All 32 lanes read the same uint4 per k-iter (one 16-byte DRAM request),
// each extracts its nibble. Same B-loop amortization as fp16 variant.
kernel void dense_gemv_q4(
    device const half* hidden       [[buffer(0)]],
    device const uint4* W_q4        [[buffer(1)]],   // [D_in * D_out / 32] uint4
    device half* output             [[buffer(2)]],
    constant uint& B                [[buffer(3)]],
    constant uint& D_in             [[buffer(4)]],
    constant uint& D_out            [[buffer(5)]],
    constant float& q4_scale        [[buffer(6)]],
    uint2 tg_pos                    [[threadgroup_position_in_grid]],
    uint2 lid2                      [[thread_position_in_threadgroup]]
) {
    constexpr uint MAX_B = 8;
    const uint n_block = tg_pos.x;
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;

    const uint k_stride = D_out / 32;
    device const uint4* w_col = W_q4 + n_block;
    const uint word_idx = lid / 8;
    const uint nib_shift = (lid % 8) * 4;

    for (uint k = 0; k < D_in; ++k) {
        uint4 pack = w_col[k * k_stride];
        uint word = (word_idx == 0) ? pack.x :
                    (word_idx == 1) ? pack.y :
                    (word_idx == 2) ? pack.z : pack.w;
        int nib = int((word >> nib_shift) & 0xF);
        float w_val = float(nib - 8) * q4_scale;
        for (uint b = 0; b < B; ++b) {
            accs[b] += float(hidden[b * D_in + k]) * w_val;
        }
    }
    for (uint b = 0; b < B; ++b) {
        output[b * D_out + n] = half(accs[b]);
    }
}
"""

func fail(_ msg: String) -> Never {
    fputs("error: \(msg)\n", stderr); exit(1)
}

guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let queue = device.makeCommandQueue()!
let opts = MTLCompileOptions()
if #available(macOS 15.0, *) { opts.languageVersion = .version3_2 }
let library: MTLLibrary
do { library = try device.makeLibrary(source: mslSource, options: opts) }
catch { fail("MSL compile: \(error)") }
guard let fpFn    = library.makeFunction(name: "dense_gemv_fp16"),
      let q4Fn    = library.makeFunction(name: "dense_gemv_q4"),
      let fp2Fn   = library.makeFunction(name: "dense_gemv_fp16_v2"),
      let q42Fn   = library.makeFunction(name: "dense_gemv_q4_v2"),
      let fp3Fn   = library.makeFunction(name: "dense_gemv_fp16_v3"),
      let q43Fn   = library.makeFunction(name: "dense_gemv_q4_v3"),
      let i8wFn   = library.makeFunction(name: "dense_gemv_i8w_v3"),
      let i8i8Fn  = library.makeFunction(name: "dense_gemv_i8_i8_v3"),
      let fp4Fn   = library.makeFunction(name: "dense_gemv_fp16_v4"),
      let fp5Fn   = library.makeFunction(name: "dense_gemv_fp16_v5") else { fail("no kernel") }
let fpPSO   = try! device.makeComputePipelineState(function: fpFn)
let q4PSO   = try! device.makeComputePipelineState(function: q4Fn)
let fp2PSO  = try! device.makeComputePipelineState(function: fp2Fn)
let q42PSO  = try! device.makeComputePipelineState(function: q42Fn)
let fp3PSO  = try! device.makeComputePipelineState(function: fp3Fn)
let q43PSO  = try! device.makeComputePipelineState(function: q43Fn)
let i8wPSO  = try! device.makeComputePipelineState(function: i8wFn)
let i8i8PSO = try! device.makeComputePipelineState(function: i8i8Fn)
let fp4PSO  = try! device.makeComputePipelineState(function: fp4Fn)
let fp5PSO  = try! device.makeComputePipelineState(function: fp5Fn)

print("device: \(device.name)")
print("")

func makeRandomHalfBuf(_ elements: Int, seed: UInt32, scale: Float = 0.02) -> MTLBuffer {
    let b = device.makeBuffer(length: elements * 2, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: Float16.self, capacity: elements)
    var s = seed
    for i in 0..<elements {
        s = s &* 1664525 &+ 1013904223
        let f = Float(Int32(bitPattern: s) % 1000) / 500.0 - 1.0
        p[i] = Float16(f * scale)
    }
    return b
}

func makeRandomQ4Buf(bytes: Int, seed: UInt32) -> MTLBuffer {
    let b = device.makeBuffer(length: bytes, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: UInt8.self, capacity: bytes)
    var s = seed
    for i in 0..<bytes {
        s = s &* 1664525 &+ 1013904223
        p[i] = UInt8(s & 0xFF)
    }
    return b
}

struct Result { let time: Double; let wBytes: Double; let flops: Double }

func makeRandomInt8Buf(bytes: Int, seed: UInt32) -> MTLBuffer {
    let b = device.makeBuffer(length: bytes, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: UInt8.self, capacity: bytes)
    var s = seed
    for i in 0..<bytes { s = s &* 1664525 &+ 1013904223; p[i] = UInt8(s & 0xFF) }
    return b
}

func runFp16V4(B: Int, Din: Int, Dout: Int, iters: Int = 20, warmup: Int = 5) -> Result {
    precondition(Din % 8 == 0)
    let hidden = makeRandomHalfBuf(B * Din, seed: 0xaaaaaaaa)
    let W = makeRandomHalfBuf(Din * Dout, seed: 0xbbbbbbbb)
    let output = device.makeBuffer(length: B * Dout * 2, options: .storageModeShared)!
    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fp4PSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(W, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var Bu = UInt32(B), Du = UInt32(Din), Do = UInt32(Dout)
        enc.setBytes(&Bu, length: 4, index: 3)
        enc.setBytes(&Du, length: 4, index: 4)
        enc.setBytes(&Do, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return Result(time: times.min()!, wBytes: Double(Din) * Double(Dout) * 2.0, flops: 2.0 * Double(B) * Double(Din) * Double(Dout))
}

func runFp16V5(B: Int, Din: Int, Dout: Int, iters: Int = 20, warmup: Int = 5) -> Result {
    precondition(Din % 32 == 0)    // N_SPLITS=4 × K-unroll=8 → Din % 32
    let hidden = makeRandomHalfBuf(B * Din, seed: 0xaaaaaaaa)
    let W = makeRandomHalfBuf(Din * Dout, seed: 0xbbbbbbbb)
    let output = device.makeBuffer(length: B * Dout * 2, options: .storageModeShared)!
    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fp5PSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(W, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var Du = UInt32(Din), Do = UInt32(Dout)
        enc.setBytes(&Du, length: 4, index: 3)
        enc.setBytes(&Do, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: B, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return Result(time: times.min()!, wBytes: Double(Din) * Double(Dout) * 2.0, flops: 2.0 * Double(B) * Double(Din) * Double(Dout))
}

func runI8W(B: Int, Din: Int, Dout: Int, iters: Int = 20, warmup: Int = 5) -> Result {
    precondition(Din % 8 == 0)
    let hidden = makeRandomHalfBuf(B * Din, seed: 0xaaaaaaaa)
    let W = makeRandomInt8Buf(bytes: Din * Dout, seed: 0xbbbbbbbb)
    let output = device.makeBuffer(length: B * Dout * 2, options: .storageModeShared)!
    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(i8wPSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(W, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var Du = UInt32(Din), Do = UInt32(Dout)
        var wscale: Float = 0.02 / 127.0
        enc.setBytes(&Du, length: 4, index: 3)
        enc.setBytes(&Do, length: 4, index: 4)
        enc.setBytes(&wscale, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: B, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return Result(time: times.min()!, wBytes: Double(Din) * Double(Dout) * 1.0,
                  flops: 2.0 * Double(B) * Double(Din) * Double(Dout))
}

func runI8I8(B: Int, Din: Int, Dout: Int, iters: Int = 20, warmup: Int = 5) -> Result {
    precondition(Din % 8 == 0)
    let hidden = makeRandomInt8Buf(bytes: B * Din, seed: 0xaaaaaaaa)
    let W = makeRandomInt8Buf(bytes: Din * Dout, seed: 0xbbbbbbbb)
    let output = device.makeBuffer(length: B * Dout * 2, options: .storageModeShared)!
    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(i8i8PSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(W, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var Du = UInt32(Din), Do = UInt32(Dout)
        var wscale: Float = 0.02 / 127.0
        var ascale: Float = 0.1 / 127.0
        enc.setBytes(&Du, length: 4, index: 3)
        enc.setBytes(&Do, length: 4, index: 4)
        enc.setBytes(&wscale, length: 4, index: 5)
        enc.setBytes(&ascale, length: 4, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: B, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return Result(time: times.min()!, wBytes: Double(Din) * Double(Dout) * 1.0,
                  flops: 2.0 * Double(B) * Double(Din) * Double(Dout))
}

func runFp16V3(B: Int, Din: Int, Dout: Int, iters: Int = 20, warmup: Int = 5) -> Result {
    precondition(Din % 8 == 0, "V3 requires D_in divisible by 8")
    let hidden = makeRandomHalfBuf(B * Din, seed: 0xaaaaaaaa)
    let W = makeRandomHalfBuf(Din * Dout, seed: 0xbbbbbbbb)
    let output = device.makeBuffer(length: B * Dout * 2, options: .storageModeShared)!
    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fp3PSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(W, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var Du = UInt32(Din), Do = UInt32(Dout)
        enc.setBytes(&Du, length: 4, index: 3)
        enc.setBytes(&Do, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: B, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return Result(time: times.min()!, wBytes: Double(Din) * Double(Dout) * 2.0, flops: 2.0 * Double(B) * Double(Din) * Double(Dout))
}

func runQ4V3(B: Int, Din: Int, Dout: Int, iters: Int = 20, warmup: Int = 5) -> Result {
    precondition(Din % 8 == 0)
    let hidden = makeRandomHalfBuf(B * Din, seed: 0xaaaaaaaa)
    let Wq4 = makeRandomQ4Buf(bytes: Din * Dout / 2, seed: 0xbbbbbbbb)
    let output = device.makeBuffer(length: B * Dout * 2, options: .storageModeShared)!
    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(q43PSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(Wq4, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var Du = UInt32(Din), Do = UInt32(Dout)
        var scale: Float = 0.02
        enc.setBytes(&Du, length: 4, index: 3)
        enc.setBytes(&Do, length: 4, index: 4)
        enc.setBytes(&scale, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: B, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return Result(time: times.min()!, wBytes: Double(Din) * Double(Dout) * 0.5, flops: 2.0 * Double(B) * Double(Din) * Double(Dout))
}

func runFp16V2(B: Int, Din: Int, Dout: Int, iters: Int = 20, warmup: Int = 5) -> Result {
    let hidden = makeRandomHalfBuf(B * Din, seed: 0xaaaaaaaa)
    let W = makeRandomHalfBuf(Din * Dout, seed: 0xbbbbbbbb)
    let output = device.makeBuffer(length: B * Dout * 2, options: .storageModeShared)!
    precondition(Dout % 32 == 0)
    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fp2PSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(W, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var Du = UInt32(Din), Do = UInt32(Dout)
        enc.setBytes(&Du, length: 4, index: 3)
        enc.setBytes(&Do, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: B, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    let t = times.min()!
    return Result(time: t, wBytes: Double(Din) * Double(Dout) * 2.0, flops: 2.0 * Double(B) * Double(Din) * Double(Dout))
}

func runQ4V2(B: Int, Din: Int, Dout: Int, iters: Int = 20, warmup: Int = 5) -> Result {
    let hidden = makeRandomHalfBuf(B * Din, seed: 0xaaaaaaaa)
    let Wq4 = makeRandomQ4Buf(bytes: Din * Dout / 2, seed: 0xbbbbbbbb)
    let output = device.makeBuffer(length: B * Dout * 2, options: .storageModeShared)!
    precondition(Dout % 32 == 0)
    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(q42PSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(Wq4, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var Du = UInt32(Din), Do = UInt32(Dout)
        var scale: Float = 0.02
        enc.setBytes(&Du, length: 4, index: 3)
        enc.setBytes(&Do, length: 4, index: 4)
        enc.setBytes(&scale, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: B, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    let t = times.min()!
    return Result(time: t, wBytes: Double(Din) * Double(Dout) * 0.5, flops: 2.0 * Double(B) * Double(Din) * Double(Dout))
}

func runFp16(B: Int, Din: Int, Dout: Int, iters: Int = 20, warmup: Int = 5) -> Result {
    let hidden = makeRandomHalfBuf(B * Din, seed: 0xaaaaaaaa)
    let W = makeRandomHalfBuf(Din * Dout, seed: 0xbbbbbbbb)
    let output = device.makeBuffer(length: B * Dout * 2, options: .storageModeShared)!

    precondition(Dout % 32 == 0, "n-block of 32 requires D_out divisible by 32")

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fpPSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(W, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var Bu = UInt32(B), Du = UInt32(Din), Do = UInt32(Dout)
        enc.setBytes(&Bu, length: 4, index: 3)
        enc.setBytes(&Du, length: 4, index: 4)
        enc.setBytes(&Do, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    let t = times.min()!
    let wBytes = Double(Din) * Double(Dout) * 2.0
    let flops = 2.0 * Double(B) * Double(Din) * Double(Dout)
    return Result(time: t, wBytes: wBytes, flops: flops)
}

func runQ4(B: Int, Din: Int, Dout: Int, iters: Int = 20, warmup: Int = 5) -> Result {
    let hidden = makeRandomHalfBuf(B * Din, seed: 0xaaaaaaaa)
    let Wq4 = makeRandomQ4Buf(bytes: Din * Dout / 2, seed: 0xbbbbbbbb)
    let output = device.makeBuffer(length: B * Dout * 2, options: .storageModeShared)!

    precondition(Dout % 32 == 0)

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(q4PSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(Wq4, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var Bu = UInt32(B), Du = UInt32(Din), Do = UInt32(Dout)
        var scale: Float = 0.02
        enc.setBytes(&Bu, length: 4, index: 3)
        enc.setBytes(&Du, length: 4, index: 4)
        enc.setBytes(&Do, length: 4, index: 5)
        enc.setBytes(&scale, length: 4, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    let t = times.min()!
    let wBytes = Double(Din) * Double(Dout) * 0.5       // Q4
    let flops = 2.0 * Double(B) * Double(Din) * Double(Dout)
    return Result(time: t, wBytes: wBytes, flops: flops)
}

print("=== Gemma-4-A4B dense projections — low-batch decode regime ===")
print("   Each row: time, TFLOPS, W-stream GB/s (native bytes per variant)")
print("   Batched-decode hypothesis: time ≈ constant across B=1..8 (DRAM-bound)")
print("")

// Gemma-4-A4B shapes (from config.json). All dense, non-routed.
// Q_total for sliding = num_heads * head_dim = 16 * 256 = 4096
// K/V sliding = num_kv_heads * head_dim = 8 * 256 = 2048
// Q full = 16 * 512 = 8192, K/V full = 2 * 512 = 1024
// attention_k_eq_v means K and V share weights; modeled as one projection of 1024 or 2048 here
let shapes: [(String, Int, Int)] = [
    ("Q sliding     2816→4096", 2816, 4096),
    ("K/V sliding   2816→2048", 2816, 2048),
    ("Q full        2816→8192", 2816, 8192),
    ("K/V full      2816→1024", 2816, 1024),
    ("out           2816→2816", 2816, 2816),
    ("shared gate/up 2816→2112", 2816, 2112),
    ("shared down   2112→2816", 2112, 2816),
    // Unembed is huge — test separately to avoid allocation issues if tight
    ("unembed       2816→262144", 2816, 262144),
]

for (label, Din, Dout) in shapes {
    print("  \(label)")
    for B in [1, 2, 4, 8] {
        let rF3  = runFp16V3(B: B, Din: Din, Dout: Dout)
        let rF4  = runFp16V4(B: B, Din: Din, Dout: Dout)
        let rF5  = runFp16V5(B: B, Din: Din, Dout: Dout)
        let rI8w = runI8W(B: B, Din: Din, Dout: Dout)
        print(String(format: "    B=%d  fp16 v3 %6.1f μs   v4 %6.1f μs   v5(splitK) %6.1f μs   I8W v3 %6.1f μs",
                     B, rF3.time * 1e6, rF4.time * 1e6, rF5.time * 1e6, rI8w.time * 1e6))
    }
    print("")
}
