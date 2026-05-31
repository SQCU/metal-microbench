import Metal
import Foundation

// ===========================================================================
// Token-grouped MoE matmul — Gemma 4 top-8-of-128 routed FFN.
//
// Two kernel variants, dispatched by per-expert group size:
//   moe_proj_m8  : M_TILE=8,  1 acc/SG × 8 SGs, 256 threads. Group sizes ≡ 8.
//   moe_proj_m16 : M_TILE=16, 4 accs/SG × 4 SGs, 128 threads. Group sizes ≡ 16.
//
// Both use simdgroup_load for the B (W) tile — the scalar-read loop in the
// prior iteration was the dominant limiter. A-tile still scalar because row
// gather is scattered across tokens (different expert routes each token).
//
// Weight layout: W[E, D_in, D_out] pre-transposed. Test harness generates
// in this layout; GGUF loader will transpose once at model load.
//
// At Gemma-4 MoE operating points:
//   batch=128   top8 → g=8   → use m8
//   batch=256   top8 → g=16  → use m16 (1 m_block per TG)
//   prefill S=2048 → g=128   → use m16 (8 m_blocks per TG, heavy W reuse)
// ===========================================================================

let mslSource = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ----- Low-batch GEMV: fp16 baseline -----
// One TG per (n_block=32 cols, active_expert). 32 threads, each owns 1 output col.
// Per k-iter: 32 lanes each read W[k, n_base + lane] — stride-1 coalesced load,
// one 64-byte DRAM transaction. hidden[k] read once by lane 0, broadcast via
// simd_shuffle to all lanes. Scalar accumulator per lane.
// Memory-bound regime — compute is trivial, W stream dominates. W bytes per
// TG = D_in × 64 bytes (32 halves/iter × D_in iters). Tracks DRAM ceiling.
kernel void moe_gemv_fp16(
    device const half* hidden          [[buffer(0)]],   // [N_tokens, D_in]
    device const uint* slot_token      [[buffer(1)]],   // [total_slots]
    device const half* W               [[buffer(2)]],   // [E, D_in, D_out]
    device const uint* active_experts  [[buffer(3)]],   // [num_active]
    device const uint* group_start     [[buffer(4)]],   // [E+1]
    device half* output                [[buffer(5)]],   // [total_slots, D_out]
    constant uint& D_in                [[buffer(6)]],
    constant uint& D_out               [[buffer(7)]],
    uint2 tg_pos                       [[threadgroup_position_in_grid]],
    uint2 lid2                         [[thread_position_in_threadgroup]]
) {
    const uint n_block = tg_pos.x;
    const uint active_idx = tg_pos.y;
    const uint expert = active_experts[active_idx];
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    const uint g_begin = group_start[expert];
    const uint g_end = group_start[expert + 1];

    for (uint slot = g_begin; slot < g_end; ++slot) {
        const uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;
        device const half* w_col = W + expert * D_in * D_out + n;

        float acc = 0.0f;
        for (uint k = 0; k < D_in; ++k) {
            // Each lane reads W[k, n] at stride D_out across k — 32 lanes cover
            // 32 contiguous N at fixed k = one coalesced cacheline per k-iter.
            acc += float(hid[k]) * float(w_col[k * D_out]);
        }
        output[slot * D_out + n] = half(acc);
    }
}

// ----- Low-batch GEMV with split-K across 4 SGs (v5) -----
// 128 threads per TG = 4 SGs. Each SG handles K/4 of the K range for the
// same (expert, n_block, slot) output cell. Reduce partials in tg-mem, SG 0
// writes final. Expected 2-4× speedup over v3 at small per-expert work.
kernel void moe_gemv_fp16_v5(
    device const half* hidden          [[buffer(0)]],
    device const uint* slot_token      [[buffer(1)]],
    device const half* W               [[buffer(2)]],
    device const uint* active_experts  [[buffer(3)]],
    device const uint* group_start     [[buffer(4)]],
    device half* output                [[buffer(5)]],
    constant uint& D_in                [[buffer(6)]],
    constant uint& D_out               [[buffer(7)]],
    uint2 tg_pos                       [[threadgroup_position_in_grid]],
    uint2 lid2                         [[thread_position_in_threadgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint N_SPLITS = 4;
    const uint n_block = tg_pos.x;
    const uint active_idx = tg_pos.y;
    const uint expert = active_experts[active_idx];
    const uint lid_sg = lid2.x % 32;
    const uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    const uint g_begin = group_start[expert];
    const uint g_end = group_start[expert + 1];
    const uint k_per_sg = D_in / N_SPLITS;
    const uint k_begin = sg_id * k_per_sg;
    const uint k_end = k_begin + k_per_sg;

    threadgroup float partials[N_SPLITS][32];

    for (uint slot = g_begin; slot < g_end; ++slot) {
        const uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;
        device const half* w_col = W + expert * D_in * D_out + n;

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

        partials[sg_id][lid_sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            output[slot * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void moe_gemv_q4_v5(
    device const half* hidden          [[buffer(0)]],
    device const uint* slot_token      [[buffer(1)]],
    device const uint4* W_q4           [[buffer(2)]],
    device const uint* active_experts  [[buffer(3)]],
    device const uint* group_start     [[buffer(4)]],
    device half* output                [[buffer(5)]],
    constant uint& D_in                [[buffer(6)]],
    constant uint& D_out               [[buffer(7)]],
    constant float& q4_scale           [[buffer(8)]],
    uint2 tg_pos                       [[threadgroup_position_in_grid]],
    uint2 lid2                         [[thread_position_in_threadgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint N_SPLITS = 4;
    const uint n_block = tg_pos.x;
    const uint active_idx = tg_pos.y;
    const uint expert = active_experts[active_idx];
    const uint lid_sg = lid2.x % 32;
    const uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    const uint g_begin = group_start[expert];
    const uint g_end = group_start[expert + 1];
    const uint expert_stride = D_in * (D_out / 32);
    const uint k_stride = D_out / 32;
    device const uint4* w_exp = W_q4 + expert * expert_stride + n_block;
    const uint word_idx = lid_sg / 8;
    const uint nib_shift = (lid_sg % 8) * 4;

    const uint k_per_sg = D_in / N_SPLITS;
    const uint k_begin = sg_id * k_per_sg;
    const uint k_end = k_begin + k_per_sg;

    threadgroup float partials[N_SPLITS][32];

    for (uint slot = g_begin; slot < g_end; ++slot) {
        const uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;

        float acc = 0.0f;
        for (uint k = k_begin; k < k_end; k += 8) {
            uint4 p0 = w_exp[(k + 0) * k_stride];
            uint4 p1 = w_exp[(k + 1) * k_stride];
            uint4 p2 = w_exp[(k + 2) * k_stride];
            uint4 p3 = w_exp[(k + 3) * k_stride];
            uint4 p4 = w_exp[(k + 4) * k_stride];
            uint4 p5 = w_exp[(k + 5) * k_stride];
            uint4 p6 = w_exp[(k + 6) * k_stride];
            uint4 p7 = w_exp[(k + 7) * k_stride];
            half h0 = hid[k + 0];
            half h1 = hid[k + 1];
            half h2 = hid[k + 2];
            half h3 = hid[k + 3];
            half h4 = hid[k + 4];
            half h5 = hid[k + 5];
            half h6 = hid[k + 6];
            half h7 = hid[k + 7];
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

        partials[sg_id][lid_sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            output[slot * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ----- Low-batch GEMV with K-unroll-by-8 (v3) -----
// Same as moe_gemv_fp16 but unrolls the K loop so each thread issues 8 W
// loads and 8 hidden loads per iter. Compiler schedules them in parallel,
// giving 8× more outstanding DRAM requests per thread — kills the latency-
// bound stall at small per-expert work.
kernel void moe_gemv_fp16_v3(
    device const half* hidden          [[buffer(0)]],
    device const uint* slot_token      [[buffer(1)]],
    device const half* W               [[buffer(2)]],
    device const uint* active_experts  [[buffer(3)]],
    device const uint* group_start     [[buffer(4)]],
    device half* output                [[buffer(5)]],
    constant uint& D_in                [[buffer(6)]],
    constant uint& D_out               [[buffer(7)]],
    uint2 tg_pos                       [[threadgroup_position_in_grid]],
    uint2 lid2                         [[thread_position_in_threadgroup]]
) {
    const uint n_block = tg_pos.x;
    const uint active_idx = tg_pos.y;
    const uint expert = active_experts[active_idx];
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    const uint g_begin = group_start[expert];
    const uint g_end = group_start[expert + 1];

    for (uint slot = g_begin; slot < g_end; ++slot) {
        const uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;
        device const half* w_col = W + expert * D_in * D_out + n;

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
        output[slot * D_out + n] = half(acc);
    }
}

kernel void moe_gemv_q4_v3(
    device const half* hidden          [[buffer(0)]],
    device const uint* slot_token      [[buffer(1)]],
    device const uint4* W_q4           [[buffer(2)]],
    device const uint* active_experts  [[buffer(3)]],
    device const uint* group_start     [[buffer(4)]],
    device half* output                [[buffer(5)]],
    constant uint& D_in                [[buffer(6)]],
    constant uint& D_out               [[buffer(7)]],
    constant float& q4_scale           [[buffer(8)]],
    uint2 tg_pos                       [[threadgroup_position_in_grid]],
    uint2 lid2                         [[thread_position_in_threadgroup]]
) {
    const uint n_block = tg_pos.x;
    const uint active_idx = tg_pos.y;
    const uint expert = active_experts[active_idx];
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    const uint g_begin = group_start[expert];
    const uint g_end = group_start[expert + 1];
    const uint expert_stride = D_in * (D_out / 32);
    const uint k_stride = D_out / 32;
    device const uint4* w_exp = W_q4 + expert * expert_stride + n_block;
    const uint word_idx = lid / 8;
    const uint nib_shift = (lid % 8) * 4;

    for (uint slot = g_begin; slot < g_end; ++slot) {
        const uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;

        float acc = 0.0f;
        for (uint k = 0; k < D_in; k += 8) {
            uint4 p0 = w_exp[(k + 0) * k_stride];
            uint4 p1 = w_exp[(k + 1) * k_stride];
            uint4 p2 = w_exp[(k + 2) * k_stride];
            uint4 p3 = w_exp[(k + 3) * k_stride];
            uint4 p4 = w_exp[(k + 4) * k_stride];
            uint4 p5 = w_exp[(k + 5) * k_stride];
            uint4 p6 = w_exp[(k + 6) * k_stride];
            uint4 p7 = w_exp[(k + 7) * k_stride];
            half h0 = hid[k + 0];
            half h1 = hid[k + 1];
            half h2 = hid[k + 2];
            half h3 = hid[k + 3];
            half h4 = hid[k + 4];
            half h5 = hid[k + 5];
            half h6 = hid[k + 6];
            half h7 = hid[k + 7];
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
        output[slot * D_out + n] = half(acc);
    }
}

// ----- Low-batch GEMV: Q4 packed weights, dequant-at-register -----
// W_q4 layout: [E, D_in, D_out/8] as uint4 (each uint4 = 16 bytes = 32 packed
// Q4 values for 32 consecutive N). All 32 lanes read the same uint4, each
// extracts its own nibble via shift-and-mask. Quant scheme is simple centered
// Q4: value = (nibble - 8) * SCALE (real Gemma Q4_K would add per-block
// scales; elided here to isolate the dequant-at-register BW win).
// DRAM footprint: 4× smaller than fp16. Memory-bound regime → 4× throughput.
kernel void moe_gemv_q4(
    device const half* hidden          [[buffer(0)]],
    device const uint* slot_token      [[buffer(1)]],
    device const uint4* W_q4           [[buffer(2)]],   // [E * D_in * D_out / 32] uint4
    device const uint* active_experts  [[buffer(3)]],
    device const uint* group_start     [[buffer(4)]],
    device half* output                [[buffer(5)]],
    constant uint& D_in                [[buffer(6)]],
    constant uint& D_out               [[buffer(7)]],
    constant float& q4_scale           [[buffer(8)]],
    uint2 tg_pos                       [[threadgroup_position_in_grid]],
    uint2 lid2                         [[thread_position_in_threadgroup]]
) {
    const uint n_block = tg_pos.x;    // each block covers 32 consecutive N
    const uint active_idx = tg_pos.y;
    const uint expert = active_experts[active_idx];
    const uint lid = lid2.x;
    const uint n = n_block * 32 + lid;
    if (n >= D_out) return;

    const uint g_begin = group_start[expert];
    const uint g_end = group_start[expert + 1];

    // Q4 index math: W_q4[e * D_in * (D_out/32) + k * (D_out/32) + n_block]
    const uint expert_stride = D_in * (D_out / 32);
    const uint k_stride = D_out / 32;
    device const uint4* w_exp = W_q4 + expert * expert_stride + n_block;

    // Per-lane bit-extract of the 32-nibble uint4: word = pack[lane/8],
    // shift = (lane % 8) * 4, nibble = (word >> shift) & 0xF
    const uint word_idx = lid / 8;
    const uint nib_shift = (lid % 8) * 4;

    for (uint slot = g_begin; slot < g_end; ++slot) {
        const uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;

        float acc = 0.0f;
        for (uint k = 0; k < D_in; ++k) {
            uint4 pack = w_exp[k * k_stride];
            uint word = (word_idx == 0) ? pack.x :
                        (word_idx == 1) ? pack.y :
                        (word_idx == 2) ? pack.z : pack.w;
            int nib = int((word >> nib_shift) & 0xF);
            float w_val = float(nib - 8) * q4_scale;
            acc += float(hid[k]) * w_val;
        }
        output[slot * D_out + n] = half(acc);
    }
}

// ----- Gather: hidden[slot_token[slot], :] -> hidden_gathered[slot, :] -----
// Expert-grouped layout. Reads scattered hidden rows into a dense expert-
// grouped tensor so the MoE projection kernel can use simdgroup_load (not
// scalar scatter) for its A-tile. One TG per slot; threads split D_in.
kernel void moe_gather_hidden(
    device const half* hidden          [[buffer(0)]],   // [N_tokens, D_in]
    device const uint* slot_token      [[buffer(1)]],   // [total_slots]
    device half* hidden_gathered       [[buffer(2)]],   // [total_slots, D_in]
    constant uint& D_in                [[buffer(3)]],
    uint3 tg_pos                       [[threadgroup_position_in_grid]],
    uint3 lid3                         [[thread_position_in_threadgroup]]
) {
    const uint slot = tg_pos.x;
    const uint lid = lid3.x;
    const uint tok = slot_token[slot];
    device const half* src = hidden + tok * D_in;
    device half* dst = hidden_gathered + slot * D_in;
    // 128 threads, D_in=2816 → 22 elements per thread. Contiguous load/store.
    for (uint i = lid; i < D_in; i += 128) {
        dst[i] = src[i];
    }
}

// ----- M_TILE=16 + gathered hidden: A-load becomes simdgroup_load -----
// Structural change from moe_proj_m16: hidden is pre-gathered into expert-
// grouped row order, so the A-tile is 16 contiguous rows × K_TILE cols of
// a flat half* with stride D_in. Uses simdgroup_load for A (8×8 matrix
// datapath) instead of scalar scatter gather.
kernel void moe_proj_m16_gathered(
    device const half* hidden_gathered [[buffer(0)]],   // [total_slots, D_in] expert-grouped
    device const half* W               [[buffer(1)]],   // [E, D_in, D_out]
    device const uint* group_start     [[buffer(2)]],   // [E+1]
    device half* output                [[buffer(3)]],   // [total_slots, D_out]
    constant uint& D_in                [[buffer(4)]],
    constant uint& D_out               [[buffer(5)]],
    uint3 tg_pos                       [[threadgroup_position_in_grid]],
    uint3 lid3                         [[thread_position_in_threadgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint M_TILE = 16;
    constexpr uint N_TILE = 64;
    constexpr uint K_TILE = 32;
    constexpr uint N_SIMDS = 4;

    const uint n_block = tg_pos.x;
    const uint expert = tg_pos.y;

    const uint g_begin = group_start[expert];
    const uint g_end   = group_start[expert + 1];
    const uint g_size  = g_end - g_begin;
    if (g_size == 0) return;

    threadgroup half A_tile[M_TILE * K_TILE];
    threadgroup half B_tile[K_TILE * N_TILE];

    const uint sg_col = sg_id * 16;
    const uint n_out_base = n_block * N_TILE;

    for (uint m_start = 0; m_start < g_size; m_start += M_TILE) {
        simdgroup_half8x8 C_acc[2][2];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                C_acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

        for (uint k_base = 0; k_base < D_in; k_base += K_TILE) {
            // A-tile: 16×32 = 2 row × 4 col subtiles = 8 subtiles, 4 SGs × 2 each
            simdgroup_half8x8 atmp;
            for (uint s = sg_id; s < 8; s += N_SIMDS) {
                const uint ar = (s / 4) * 8;
                const uint ac = (s % 4) * 8;
                simdgroup_load(atmp,
                    hidden_gathered + (g_begin + m_start + ar) * D_in + k_base + ac,
                    D_in);
                simdgroup_store(atmp, A_tile + ar * K_TILE + ac, K_TILE);
            }
            // B-tile: 32×64 = 4 row × 8 col = 32 subtiles, 4 SGs × 8 each
            simdgroup_half8x8 btmp;
            for (uint s = sg_id; s < 32; s += N_SIMDS) {
                const uint br = (s / 8) * 8;
                const uint bc = (s % 8) * 8;
                simdgroup_load(btmp,
                    W + (expert * D_in + k_base + br) * D_out + n_out_base + bc,
                    D_out);
                simdgroup_store(btmp, B_tile + br * N_TILE + bc, N_TILE);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k_sub = 0; k_sub < K_TILE; k_sub += 8) {
                simdgroup_half8x8 a[2], b[2];
                for (int i = 0; i < 2; ++i)
                    simdgroup_load(a[i], A_tile + i * 8 * K_TILE + k_sub, K_TILE);
                for (int j = 0; j < 2; ++j)
                    simdgroup_load(b[j], B_tile + k_sub * N_TILE + sg_col + j * 8, N_TILE);
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j)
                        simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                simdgroup_store(C_acc[i][j],
                                output + (g_begin + m_start + i * 8) * D_out + n_out_base + sg_col + j * 8,
                                D_out);
    }
}

// ----- M_TILE=16, K-contiguous weight layout (the down-shape fix) -----
// Identical compute to moe_proj_m16, but the weight Wt is stored
// [E, D_out, D_in] (contraction axis D_in innermost/contiguous) instead of
// [E, D_in, D_out]. The B-tile is loaded with a transposed simdgroup_load,
// so the K-walk strides by D_in (small) rather than D_out (large). For the
// down projection (D_in=704 << D_out=2816) this restores the gate/up-shape's
// memory locality; the original kernel contracts along the strided axis and
// collapses to ~13 GB/s whenever D_out is large.
kernel void moe_proj_m16_kt(
    device const half* hidden       [[buffer(0)]],
    device const half* Wt           [[buffer(1)]],   // [E, D_out, D_in]  (transposed)
    device const uint* group_start  [[buffer(2)]],
    device const uint* slot_token   [[buffer(3)]],
    device half* output             [[buffer(4)]],
    constant uint& D_in             [[buffer(5)]],
    constant uint& D_out            [[buffer(6)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 lid3                      [[thread_position_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint M_TILE = 16;
    constexpr uint N_TILE = 64;
    constexpr uint K_TILE = 32;
    constexpr uint N_SIMDS = 4;
    constexpr uint THREADS = 128;

    const uint n_block = tg_pos.x;
    const uint expert = tg_pos.y;
    const uint lid = lid3.x;

    const uint g_begin = group_start[expert];
    const uint g_end   = group_start[expert + 1];
    const uint g_size  = g_end - g_begin;
    if (g_size == 0) return;

    threadgroup half A_tile[M_TILE * K_TILE];
    threadgroup half B_tile[K_TILE * N_TILE];

    const uint sg_col = sg_id * 16;        // 0, 16, 32, 48
    const uint n_out_base = n_block * N_TILE;

    for (uint m_start = 0; m_start < g_size; m_start += M_TILE) {
        const uint rows = min(M_TILE, g_size - m_start);

        simdgroup_half8x8 C_acc[2][2];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                C_acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

        for (uint k_base = 0; k_base < D_in; k_base += K_TILE) {
            for (uint i = 0; i < 4; ++i) {
                const uint flat = i * THREADS + lid;
                const uint r = flat / K_TILE;
                const uint c = flat % K_TILE;
                half v = 0.0h;
                if (r < rows) {
                    const uint tok = slot_token[g_begin + m_start + r];
                    v = hidden[tok * D_in + k_base + c];
                }
                A_tile[r * K_TILE + c] = v;
            }

            // B-tile from transposed weight: src block is [n][k] with row
            // stride D_in; transposed load yields [k][n] subtiles. Reads run
            // along contiguous k → coalesced regardless of D_out.
            simdgroup_half8x8 btmp;
            for (uint s = sg_id; s < 32; s += N_SIMDS) {
                const uint br = (s / 8) * 8;   // K offset within tile
                const uint bc = (s % 8) * 8;   // N offset within tile
                simdgroup_load(btmp,
                    Wt + (expert * D_out + n_out_base + bc) * D_in + k_base + br,
                    D_in, ulong2(0, 0), /*transpose=*/true);
                simdgroup_store(btmp, B_tile + br * N_TILE + bc, N_TILE);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k_sub = 0; k_sub < K_TILE; k_sub += 8) {
                simdgroup_half8x8 a[2], b[2];
                for (int i = 0; i < 2; ++i)
                    simdgroup_load(a[i], A_tile + i * 8 * K_TILE + k_sub, K_TILE);
                for (int j = 0; j < 2; ++j)
                    simdgroup_load(b[j], B_tile + k_sub * N_TILE + sg_col + j * 8, N_TILE);
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j)
                        simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                simdgroup_store(C_acc[i][j],
                                output + (g_begin + m_start + i * 8) * D_out + n_out_base + sg_col + j * 8,
                                D_out);
    }
}

// ----- M_TILE=8 variant: 8 SGs × 1 acc (8×8), 256 threads/TG -----
kernel void moe_proj_m8(
    device const half* hidden       [[buffer(0)]],
    device const half* W            [[buffer(1)]],
    device const uint* group_start  [[buffer(2)]],
    device const uint* slot_token   [[buffer(3)]],
    device half* output             [[buffer(4)]],
    constant uint& D_in             [[buffer(5)]],
    constant uint& D_out            [[buffer(6)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 lid3                      [[thread_position_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint M_TILE = 8;
    constexpr uint N_TILE = 64;
    constexpr uint K_TILE = 32;
    constexpr uint N_SIMDS = 8;
    constexpr uint THREADS = 256;

    const uint n_block = tg_pos.x;
    const uint expert = tg_pos.y;
    const uint lid = lid3.x;

    const uint g_begin = group_start[expert];
    const uint g_end   = group_start[expert + 1];
    const uint g_size  = g_end - g_begin;
    if (g_size == 0) return;

    threadgroup half A_tile[M_TILE * K_TILE];
    threadgroup half B_tile[K_TILE * N_TILE];

    const uint sg_col = sg_id * 8;
    const uint n_out_base = n_block * N_TILE;

    for (uint m_start = 0; m_start < g_size; m_start += M_TILE) {
        const uint rows = min(M_TILE, g_size - m_start);

        simdgroup_half8x8 C_acc = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

        for (uint k_base = 0; k_base < D_in; k_base += K_TILE) {
            // A-tile scalar scatter-gather: 256 threads × 1 element = 256 halves.
            {
                const uint r = lid / K_TILE;
                const uint c = lid % K_TILE;
                half v = 0.0h;
                if (r < rows) {
                    const uint tok = slot_token[g_begin + m_start + r];
                    v = hidden[tok * D_in + k_base + c];
                }
                A_tile[r * K_TILE + c] = v;
            }

            // B-tile: 32×64 = 4×8 subtile grid = 32 subtiles. 8 SGs × 4 each.
            simdgroup_half8x8 btmp;
            for (uint s = sg_id; s < 32; s += N_SIMDS) {
                const uint br = (s / 8) * 8;
                const uint bc = (s % 8) * 8;
                simdgroup_load(btmp,
                    W + (expert * D_in + k_base + br) * D_out + n_out_base + bc,
                    D_out);
                simdgroup_store(btmp, B_tile + br * N_TILE + bc, N_TILE);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k_sub = 0; k_sub < K_TILE; k_sub += 8) {
                simdgroup_half8x8 a, b;
                simdgroup_load(a, A_tile + k_sub, K_TILE);
                simdgroup_load(b, B_tile + k_sub * N_TILE + sg_col, N_TILE);
                simdgroup_multiply_accumulate(C_acc, a, b, C_acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_store(C_acc,
                        output + (g_begin + m_start) * D_out + n_out_base + sg_col,
                        D_out);
    }
}

// ----- M_TILE=16 + double-buffered K prefetch -----
// Same 4 SG × 4 acc layout as m16, but tg-mem holds two K tiles so the
// load for k_base+1 overlaps the compute on k_base. Halves barriers
// (one per k_iter instead of two). tg-mem: 2 × (A 1KB + B 4KB) = 10 KB.
kernel void moe_proj_m16_pipe(
    device const half* hidden       [[buffer(0)]],
    device const half* W            [[buffer(1)]],
    device const uint* group_start  [[buffer(2)]],
    device const uint* slot_token   [[buffer(3)]],
    device half* output             [[buffer(4)]],
    constant uint& D_in             [[buffer(5)]],
    constant uint& D_out            [[buffer(6)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 lid3                      [[thread_position_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint M_TILE = 16;
    constexpr uint N_TILE = 64;
    constexpr uint K_TILE = 32;
    constexpr uint N_SIMDS = 4;
    constexpr uint THREADS = 128;

    const uint n_block = tg_pos.x;
    const uint expert = tg_pos.y;
    const uint lid = lid3.x;

    const uint g_begin = group_start[expert];
    const uint g_end   = group_start[expert + 1];
    const uint g_size  = g_end - g_begin;
    if (g_size == 0) return;

    threadgroup half A_tile[2][M_TILE * K_TILE];
    threadgroup half B_tile[2][K_TILE * N_TILE];

    const uint sg_col = sg_id * 16;
    const uint n_out_base = n_block * N_TILE;

    // Helper: load A + B for a given (m_start, k_base) into slot `buf`.
    auto load_tile = [&](uint m_start, uint rows, uint k_base, uint buf) {
        // A: 16×32 = 512 halves, 128 threads × 4 each
        for (uint i = 0; i < 4; ++i) {
            const uint flat = i * THREADS + lid;
            const uint r = flat / K_TILE;
            const uint c = flat % K_TILE;
            half v = 0.0h;
            if (r < rows) {
                const uint tok = slot_token[g_begin + m_start + r];
                v = hidden[tok * D_in + k_base + c];
            }
            A_tile[buf][r * K_TILE + c] = v;
        }
        // B: 32×64 = 4×8 subtiles, 4 SGs × 8 each
        simdgroup_half8x8 btmp;
        for (uint s = sg_id; s < 32; s += N_SIMDS) {
            const uint br = (s / 8) * 8;
            const uint bc = (s % 8) * 8;
            simdgroup_load(btmp,
                W + (expert * D_in + k_base + br) * D_out + n_out_base + bc,
                D_out);
            simdgroup_store(btmp, B_tile[buf] + br * N_TILE + bc, N_TILE);
        }
    };

    for (uint m_start = 0; m_start < g_size; m_start += M_TILE) {
        const uint rows = min(M_TILE, g_size - m_start);

        simdgroup_half8x8 C_acc[2][2];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                C_acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

        // Prime: load tile[0] for k_base=0
        load_tile(m_start, rows, 0, 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Main loop: compute on buf, prefetch next tile into (1-buf).
        uint buf = 0;
        for (uint k_base = 0; k_base < D_in; k_base += K_TILE) {
            const uint next_k = k_base + K_TILE;
            const bool has_next = (next_k < D_in);

            // Prefetch next K tile (writes to 1-buf) while compute on buf runs below.
            // The store side of load_tile writes to B_tile[1-buf] / A_tile[1-buf],
            // distinct from the compute-read side at A_tile[buf] / B_tile[buf].
            if (has_next) {
                load_tile(m_start, rows, next_k, 1 - buf);
            }

            // Compute on the already-loaded buf. No barrier needed between load and
            // compute here because compute reads buf, load writes to 1-buf.
            for (uint k_sub = 0; k_sub < K_TILE; k_sub += 8) {
                simdgroup_half8x8 a[2], b[2];
                for (int i = 0; i < 2; ++i)
                    simdgroup_load(a[i], A_tile[buf] + i * 8 * K_TILE + k_sub, K_TILE);
                for (int j = 0; j < 2; ++j)
                    simdgroup_load(b[j], B_tile[buf] + k_sub * N_TILE + sg_col + j * 8, N_TILE);
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j)
                        simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
            }

            // Single barrier per k_iter: wait for the prefetch to finish before
            // next iter's compute reads from it, and wait for current compute
            // to finish reading buf before next iter's prefetch writes over it
            // (two iters ahead). With 2 buffers the "two iters ahead" clobber
            // is exactly what we synchronize here.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf = 1 - buf;
        }

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                simdgroup_store(C_acc[i][j],
                                output + (g_begin + m_start + i * 8) * D_out + n_out_base + sg_col + j * 8,
                                D_out);
    }
}

// ----- M_TILE=16 variant: 4 SGs × 4 accs (2×2), 128 threads/TG -----
// Per-SG: 16×16 output = 2×2 of 8×8. 4 SGs arranged 1×4 along N.
kernel void moe_proj_m16(
    device const half* hidden       [[buffer(0)]],
    device const half* W            [[buffer(1)]],
    device const uint* group_start  [[buffer(2)]],
    device const uint* slot_token   [[buffer(3)]],
    device half* output             [[buffer(4)]],
    constant uint& D_in             [[buffer(5)]],
    constant uint& D_out            [[buffer(6)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 lid3                      [[thread_position_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint M_TILE = 16;
    constexpr uint N_TILE = 64;
    constexpr uint K_TILE = 32;
    constexpr uint N_SIMDS = 4;
    constexpr uint THREADS = 128;

    const uint n_block = tg_pos.x;
    const uint expert = tg_pos.y;
    const uint lid = lid3.x;

    const uint g_begin = group_start[expert];
    const uint g_end   = group_start[expert + 1];
    const uint g_size  = g_end - g_begin;
    if (g_size == 0) return;

    threadgroup half A_tile[M_TILE * K_TILE];
    threadgroup half B_tile[K_TILE * N_TILE];

    // SG arrangement: 1 M-row × 4 N-cols. Each SG covers 16×16 output.
    const uint sg_col = sg_id * 16;        // 0, 16, 32, 48
    const uint n_out_base = n_block * N_TILE;

    for (uint m_start = 0; m_start < g_size; m_start += M_TILE) {
        const uint rows = min(M_TILE, g_size - m_start);

        simdgroup_half8x8 C_acc[2][2];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                C_acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

        for (uint k_base = 0; k_base < D_in; k_base += K_TILE) {
            // A-tile scalar gather: 16×32 = 512 halves, 128 threads × 4 each.
            for (uint i = 0; i < 4; ++i) {
                const uint flat = i * THREADS + lid;
                const uint r = flat / K_TILE;
                const uint c = flat % K_TILE;
                half v = 0.0h;
                if (r < rows) {
                    const uint tok = slot_token[g_begin + m_start + r];
                    v = hidden[tok * D_in + k_base + c];
                }
                A_tile[r * K_TILE + c] = v;
            }

            // B-tile: 32×64 = 4×8 = 32 subtiles. 4 SGs × 8 each.
            simdgroup_half8x8 btmp;
            for (uint s = sg_id; s < 32; s += N_SIMDS) {
                const uint br = (s / 8) * 8;
                const uint bc = (s % 8) * 8;
                simdgroup_load(btmp,
                    W + (expert * D_in + k_base + br) * D_out + n_out_base + bc,
                    D_out);
                simdgroup_store(btmp, B_tile + br * N_TILE + bc, N_TILE);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k_sub = 0; k_sub < K_TILE; k_sub += 8) {
                simdgroup_half8x8 a[2], b[2];
                for (int i = 0; i < 2; ++i)
                    simdgroup_load(a[i], A_tile + i * 8 * K_TILE + k_sub, K_TILE);
                for (int j = 0; j < 2; ++j)
                    simdgroup_load(b[j], B_tile + k_sub * N_TILE + sg_col + j * 8, N_TILE);
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j)
                        simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                simdgroup_store(C_acc[i][j],
                                output + (g_begin + m_start + i * 8) * D_out + n_out_base + sg_col + j * 8,
                                D_out);
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
guard let m8Fn       = library.makeFunction(name: "moe_proj_m8"),
      let m16Fn      = library.makeFunction(name: "moe_proj_m16"),
      let m16pFn     = library.makeFunction(name: "moe_proj_m16_pipe"),
      let m16gFn     = library.makeFunction(name: "moe_proj_m16_gathered"),
      let m16ktFn    = library.makeFunction(name: "moe_proj_m16_kt"),
      let gatherFn   = library.makeFunction(name: "moe_gather_hidden"),
      let gemvFpFn   = library.makeFunction(name: "moe_gemv_fp16"),
      let gemvQ4Fn   = library.makeFunction(name: "moe_gemv_q4"),
      let gemvFp3Fn  = library.makeFunction(name: "moe_gemv_fp16_v3"),
      let gemvQ43Fn  = library.makeFunction(name: "moe_gemv_q4_v3"),
      let gemvFp5Fn  = library.makeFunction(name: "moe_gemv_fp16_v5"),
      let gemvQ45Fn  = library.makeFunction(name: "moe_gemv_q4_v5") else { fail("no kernel") }
let m8PSO       = try! device.makeComputePipelineState(function: m8Fn)
let m16PSO      = try! device.makeComputePipelineState(function: m16Fn)
let m16pPSO     = try! device.makeComputePipelineState(function: m16pFn)
let m16gPSO     = try! device.makeComputePipelineState(function: m16gFn)
let m16ktPSO    = try! device.makeComputePipelineState(function: m16ktFn)
let gatherPSO   = try! device.makeComputePipelineState(function: gatherFn)
let gemvFpPSO   = try! device.makeComputePipelineState(function: gemvFpFn)
let gemvQ4PSO   = try! device.makeComputePipelineState(function: gemvQ4Fn)
let gemvFp3PSO  = try! device.makeComputePipelineState(function: gemvFp3Fn)
let gemvQ43PSO  = try! device.makeComputePipelineState(function: gemvQ43Fn)
let gemvFp5PSO  = try! device.makeComputePipelineState(function: gemvFp5Fn)
let gemvQ45PSO  = try! device.makeComputePipelineState(function: gemvQ45Fn)

print("device: \(device.name)")
print("m8:           tgMem=\(m8PSO.staticThreadgroupMemoryLength) B")
print("m16:          tgMem=\(m16PSO.staticThreadgroupMemoryLength) B")
print("m16_pipe:     tgMem=\(m16pPSO.staticThreadgroupMemoryLength) B")
print("m16_gathered: tgMem=\(m16gPSO.staticThreadgroupMemoryLength) B")
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

func buildUniformRouting(nTokens: Int, topK: Int, E: Int)
    -> (groupStart: MTLBuffer, slotToken: MTLBuffer, totalSlots: Int, groupSize: Int)
{
    let totalSlots = nTokens * topK
    precondition(totalSlots % E == 0, "need uniform slot distribution")
    let groupSize = totalSlots / E

    let groupStartBuf = device.makeBuffer(length: (E + 1) * 4, options: .storageModeShared)!
    let gsp = groupStartBuf.contents().bindMemory(to: UInt32.self, capacity: E + 1)
    for e in 0...E { gsp[e] = UInt32(e * groupSize) }

    let slotTokenBuf = device.makeBuffer(length: totalSlots * 4, options: .storageModeShared)!
    let stp = slotTokenBuf.contents().bindMemory(to: UInt32.self, capacity: totalSlots)
    for e in 0..<E {
        for i in 0..<groupSize {
            let slot = e * groupSize + i
            stp[slot] = UInt32((e + i * E / topK) % nTokens)
        }
    }
    return (groupStartBuf, slotTokenBuf, totalSlots, groupSize)
}

enum Variant { case m8, m16, m16p, m16g, m16kt }

struct ProjRun { let time: Double; let flops: Double; let wBytes: Double; var outSum: Double = 0 }

func runProj(variant: Variant,
             nTokens: Int, Din: Int, Dout: Int, E: Int, topK: Int,
             iters: Int = 10, warmup: Int = 3) -> ProjRun
{
    let hidden = makeRandomHalfBuf(nTokens * Din, seed: 0xaaaaaaaa)
    let W = makeRandomHalfBuf(E * Din * Dout, seed: 0xbbbbbbbb)
    let (gStart, slotTok, totalSlots, gSize) = buildUniformRouting(nTokens: nTokens, topK: topK, E: E)
    let output = device.makeBuffer(length: totalSlots * Dout * 2, options: .storageModeShared)!

    precondition(Dout % 64 == 0, "N_TILE=64 requires D_out divisible by 64")
    if variant == .m16 || variant == .m16p || variant == .m16g || variant == .m16kt {
        precondition(gSize % 16 == 0, "m16 requires group size divisible by 16")
    }

    let pso: MTLComputePipelineState
    let threads: Int
    switch variant {
    case .m8:    pso = m8PSO;    threads = 256
    case .m16:   pso = m16PSO;   threads = 128
    case .m16p:  pso = m16pPSO;  threads = 128
    case .m16g:  pso = m16gPSO;  threads = 128
    case .m16kt: pso = m16ktPSO; threads = 128
    }

    // m16kt contracts along the contiguous axis, so it needs the weight stored
    // [E, D_out, D_in] (transposed from the [E, D_in, D_out] the others use).
    // Built once here, outside the timed loop.
    let Wkt: MTLBuffer = {
        guard variant == .m16kt else { return W }
        let t = device.makeBuffer(length: E * Din * Dout * 2, options: .storageModeShared)!
        let src = W.contents().bindMemory(to: Float16.self, capacity: E * Din * Dout)
        let dst = t.contents().bindMemory(to: Float16.self, capacity: E * Din * Dout)
        for e in 0..<E {
            let base = e * Din * Dout
            for k in 0..<Din {
                let srow = base + k * Dout
                for n in 0..<Dout { dst[base + n * Din + k] = src[srow + n] }
            }
        }
        return t
    }()

    // For m16g: pre-gather hidden into expert-grouped rows. Includes both
    // the gather and the projection in the timed window, so the reported
    // time is apples-to-apples against the scatter-load variants.
    let hiddenGathered: MTLBuffer? = (variant == .m16g)
        ? device.makeBuffer(length: totalSlots * Din * 2, options: .storageModeShared)
        : nil

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!

        if variant == .m16g {
            let encG = cb.makeComputeCommandEncoder()!
            encG.setComputePipelineState(gatherPSO)
            encG.setBuffer(hidden, offset: 0, index: 0)
            encG.setBuffer(slotTok, offset: 0, index: 1)
            encG.setBuffer(hiddenGathered!, offset: 0, index: 2)
            var Du = UInt32(Din)
            encG.setBytes(&Du, length: 4, index: 3)
            encG.dispatchThreadgroups(
                MTLSize(width: totalSlots, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encG.endEncoding()
        }

        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        if variant == .m16g {
            enc.setBuffer(hiddenGathered!, offset: 0, index: 0)
            enc.setBuffer(W, offset: 0, index: 1)
            enc.setBuffer(gStart, offset: 0, index: 2)
            enc.setBuffer(output, offset: 0, index: 3)
            var Du = UInt32(Din), Do = UInt32(Dout)
            enc.setBytes(&Du, length: 4, index: 4)
            enc.setBytes(&Do, length: 4, index: 5)
        } else {
            enc.setBuffer(hidden, offset: 0, index: 0)
            enc.setBuffer(Wkt, offset: 0, index: 1)
            enc.setBuffer(gStart, offset: 0, index: 2)
            enc.setBuffer(slotTok, offset: 0, index: 3)
            enc.setBuffer(output, offset: 0, index: 4)
            var Du = UInt32(Din), Do = UInt32(Dout)
            enc.setBytes(&Du, length: 4, index: 5)
            enc.setBytes(&Do, length: 4, index: 6)
        }
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 64, height: E, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    let t = times.min()!
    let flops = 2.0 * Double(totalSlots) * Double(Din) * Double(Dout)
    let wBytes = Double(E) * Double(Din) * Double(Dout) * 2.0
    // checksum the output (first 4096 slots' col 0..63) for cross-variant verify
    let op = output.contents().bindMemory(to: Float16.self, capacity: totalSlots * Dout)
    var sum = 0.0
    let nCheck = min(totalSlots, 4096)
    for r in 0..<nCheck { for c in 0..<min(64, Dout) { sum += Double(op[r * Dout + c]) } }
    return ProjRun(time: t, flops: flops, wBytes: wBytes, outSum: sum)
}

// Correctness: m16 and m16kt must agree (same W seed; kt transposes it).
do {
    let a = runProj(variant: .m16,   nTokens: 256, Din: 2816, Dout: 704, E: 128, topK: 8, iters: 1, warmup: 0)
    let b = runProj(variant: .m16kt, nTokens: 256, Din: 2816, Dout: 704, E: 128, topK: 8, iters: 1, warmup: 0)
    let rel = abs(a.outSum - b.outSum) / max(1e-6, abs(a.outSum))
    print(String(format: "[verify] gate/up  m16 outSum=%.3f  m16kt=%.3f  rel=%.2e  %@",
                 a.outSum, b.outSum, rel, rel < 1e-2 ? "OK" : "MISMATCH!"))
    let c = runProj(variant: .m16,   nTokens: 256, Din: 704, Dout: 2816, E: 128, topK: 8, iters: 1, warmup: 0)
    let d = runProj(variant: .m16kt, nTokens: 256, Din: 704, Dout: 2816, E: 128, topK: 8, iters: 1, warmup: 0)
    let rel2 = abs(c.outSum - d.outSum) / max(1e-6, abs(c.outSum))
    print(String(format: "[verify] down     m16 outSum=%.3f  m16kt=%.3f  rel=%.2e  %@",
                 c.outSum, d.outSum, rel2, rel2 < 1e-2 ? "OK" : "MISMATCH!"))
}

print("=== Gemma-4 MoE token-grouped projection — top-8 of 128 ===")
print("   shape per expert: D_in × D_out. W tile is streamed once per TG and")
print("   reused across m_blocks — amortization grows with group size.")
print("")

let topK = 8
let E = 128

func line(_ label: String, _ r: ProjRun, _ note: String = "") {
    let tflops = r.flops / r.time / 1e12
    let wGbps = r.wBytes / r.time / 1e9
    let hdr = label.padding(toLength: 38, withPad: " ", startingAt: 0)
    let tail = note.isEmpty ? "" : "  [\(note)]"
    print("  \(hdr)  \(String(format: "%8.2f ms", r.time * 1000))   \(String(format: "%6.2f TFLOPS", tflops))   \(String(format: "%6.0f GB/s W 1×", wGbps))\(tail)")
}

// ===========================================================================
// Low-batch GEMV sweep (the actual batched-decode regime).
// At batch=4 × top_k=8 = 32 slots over 128 experts, ~32 active experts w/
// ~1 token each. Pure memory-bound on W stream. Q4 cuts DRAM by 4×.
// ===========================================================================

struct GemvResult { let time: Double; let wBytesFP16: Double; let flops: Double }

func runGemvGeneric(pso: MTLComputePipelineState, useQ4: Bool,
                    nTokens: Int, Din: Int, Dout: Int, E: Int, topK: Int,
                    threadsPerTG: Int = 32,
                    iters: Int = 20, warmup: Int = 5) -> GemvResult
{
    let hidden = makeRandomHalfBuf(nTokens * Din, seed: 0xaaaaaaaa)
    let (gStart, slotTok, totalSlots, gSize) = buildUniformRouting(nTokens: nTokens, topK: topK, E: E)
    let output = device.makeBuffer(length: totalSlots * Dout * 2, options: .storageModeShared)!

    let numActive = (gSize > 0) ? E : 0
    let activeBuf = device.makeBuffer(length: numActive * 4, options: .storageModeShared)!
    let ap = activeBuf.contents().bindMemory(to: UInt32.self, capacity: numActive)
    for i in 0..<numActive { ap[i] = UInt32(i) }

    let W: MTLBuffer
    if useQ4 {
        let bytes = E * Din * Dout / 2
        W = device.makeBuffer(length: bytes, options: .storageModeShared)!
        let p = W.contents().bindMemory(to: UInt8.self, capacity: bytes)
        var s: UInt32 = 0xbbbbbbbb
        for i in 0..<bytes { s = s &* 1664525 &+ 1013904223; p[i] = UInt8(s & 0xFF) }
    } else {
        W = makeRandomHalfBuf(E * Din * Dout, seed: 0xbbbbbbbb)
    }

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(slotTok, offset: 0, index: 1)
        enc.setBuffer(W, offset: 0, index: 2)
        enc.setBuffer(activeBuf, offset: 0, index: 3)
        enc.setBuffer(gStart, offset: 0, index: 4)
        enc.setBuffer(output, offset: 0, index: 5)
        var Du = UInt32(Din), Do = UInt32(Dout)
        enc.setBytes(&Du, length: 4, index: 6)
        enc.setBytes(&Do, length: 4, index: 7)
        if useQ4 {
            var scale: Float = 0.02
            enc.setBytes(&scale, length: 4, index: 8)
        }
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: numActive, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    let t = times.min()!
    let bytesPerElement: Double = useQ4 ? 0.5 : 2.0
    let wBytes = Double(numActive) * Double(Din) * Double(Dout) * bytesPerElement
    let flops = 2.0 * Double(totalSlots) * Double(Din) * Double(Dout)
    return GemvResult(time: t, wBytesFP16: wBytes, flops: flops)
}

func runGemvFp16(nTokens: Int, Din: Int, Dout: Int, E: Int, topK: Int,
                 iters: Int = 20, warmup: Int = 5) -> GemvResult
{
    let hidden = makeRandomHalfBuf(nTokens * Din, seed: 0xaaaaaaaa)
    let W = makeRandomHalfBuf(E * Din * Dout, seed: 0xbbbbbbbb)
    let (gStart, slotTok, totalSlots, gSize) = buildUniformRouting(nTokens: nTokens, topK: topK, E: E)
    let output = device.makeBuffer(length: totalSlots * Dout * 2, options: .storageModeShared)!

    // Active-expert compaction: uniform routing → all E are active if gSize>0.
    let numActive = (gSize > 0) ? E : 0
    let activeBuf = device.makeBuffer(length: numActive * 4, options: .storageModeShared)!
    let ap = activeBuf.contents().bindMemory(to: UInt32.self, capacity: numActive)
    for i in 0..<numActive { ap[i] = UInt32(i) }

    precondition(Dout % 32 == 0, "N-block of 32 requires D_out divisible by 32")

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(gemvFpPSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(slotTok, offset: 0, index: 1)
        enc.setBuffer(W, offset: 0, index: 2)
        enc.setBuffer(activeBuf, offset: 0, index: 3)
        enc.setBuffer(gStart, offset: 0, index: 4)
        enc.setBuffer(output, offset: 0, index: 5)
        var Du = UInt32(Din), Do = UInt32(Dout)
        enc.setBytes(&Du, length: 4, index: 6)
        enc.setBytes(&Do, length: 4, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: numActive, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    let t = times.min()!
    let wBytes = Double(numActive) * Double(Din) * Double(Dout) * 2.0   // active experts only
    let flops = 2.0 * Double(totalSlots) * Double(Din) * Double(Dout)
    return GemvResult(time: t, wBytesFP16: wBytes, flops: flops)
}

func runGemvQ4(nTokens: Int, Din: Int, Dout: Int, E: Int, topK: Int,
               iters: Int = 20, warmup: Int = 5) -> GemvResult
{
    let hidden = makeRandomHalfBuf(nTokens * Din, seed: 0xaaaaaaaa)
    // Q4 buffer: E * D_in * D_out / 2 bytes, organized as uint4 = 16 B = 32 nibbles
    let q4Count = E * Din * Dout / 2   // bytes
    let Wq4 = device.makeBuffer(length: q4Count, options: .storageModeShared)!
    let qp = Wq4.contents().bindMemory(to: UInt8.self, capacity: q4Count)
    var s: UInt32 = 0xbbbbbbbb
    for i in 0..<q4Count {
        s = s &* 1664525 &+ 1013904223
        qp[i] = UInt8(s & 0xFF)
    }

    let (gStart, slotTok, totalSlots, gSize) = buildUniformRouting(nTokens: nTokens, topK: topK, E: E)
    let output = device.makeBuffer(length: totalSlots * Dout * 2, options: .storageModeShared)!

    let numActive = (gSize > 0) ? E : 0
    let activeBuf = device.makeBuffer(length: numActive * 4, options: .storageModeShared)!
    let ap = activeBuf.contents().bindMemory(to: UInt32.self, capacity: numActive)
    for i in 0..<numActive { ap[i] = UInt32(i) }

    precondition(Dout % 32 == 0)

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(gemvQ4PSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(slotTok, offset: 0, index: 1)
        enc.setBuffer(Wq4, offset: 0, index: 2)
        enc.setBuffer(activeBuf, offset: 0, index: 3)
        enc.setBuffer(gStart, offset: 0, index: 4)
        enc.setBuffer(output, offset: 0, index: 5)
        var Du = UInt32(Din), Do = UInt32(Dout)
        var scale: Float = 0.02
        enc.setBytes(&Du, length: 4, index: 6)
        enc.setBytes(&Do, length: 4, index: 7)
        enc.setBytes(&scale, length: 4, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(width: Dout / 32, height: numActive, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    let t = times.min()!
    let wBytes = Double(numActive) * Double(Din) * Double(Dout) * 0.5    // Q4 = 0.5 bytes
    let flops = 2.0 * Double(totalSlots) * Double(Din) * Double(Dout)
    return GemvResult(time: t, wBytesFP16: wBytes, flops: flops)
}

print("=== Low-batch GEMV — batched-decode regime (memory-bound on W) ===")
print("   test config: batch × top_k=8 slots, uniform-distributed over E=128")
print("   (all experts active; real router would give ~32 active at batch=4)")
print("")
for (label, Din, Dout) in [
    ("gate/up 2816→704", 2816, 704),
    ("down    704→2816", 704, 2816),
] {
    print("  --- \(label) (GEMV v3 vs v5 split-K) ---")
    for N in [16, 32] {
        if (N * 8) % 128 != 0 { continue }
        let rFp3 = runGemvGeneric(pso: gemvFp3PSO, useQ4: false, nTokens: N, Din: Din, Dout: Dout, E: 128, topK: 8)
        let rFp5 = runGemvGeneric(pso: gemvFp5PSO, useQ4: false, nTokens: N, Din: Din, Dout: Dout, E: 128, topK: 8, threadsPerTG: 128)
        let rQ43 = runGemvGeneric(pso: gemvQ43PSO, useQ4: true,  nTokens: N, Din: Din, Dout: Dout, E: 128, topK: 8)
        let rQ45 = runGemvGeneric(pso: gemvQ45PSO, useQ4: true,  nTokens: N, Din: Din, Dout: Dout, E: 128, topK: 8, threadsPerTG: 128)
        print(String(format: "  N=%d g=%d/expert  fp16: v3 %7.1f μs | v5 %7.1f μs (%4.1f×)  Q4: v3 %7.1f μs | v5 %7.1f μs (%4.1f×)",
                     N, N * 8 / 128,
                     rFp3.time * 1e6, rFp5.time * 1e6, rFp3.time / rFp5.time,
                     rQ43.time * 1e6, rQ45.time * 1e6, rQ43.time / rQ45.time))
    }
    print("")
}

print("=== Token-grouped MMA path (for reference at higher batches) ===")
print("")
for (label, Din, Dout) in [
    ("gate/up 2816→704", 2816, 704),
    ("down    704→2816", 704, 2816),
] {
    print("  --- \(label) ---")
    // N=128 (g=8): m8 only.
    line("N=128  g=8   m8",  runProj(variant: .m8,  nTokens: 128,  Din: Din, Dout: Dout, E: E, topK: topK))
    // N=256 (g=16): baseline m8, m16, m16_pipe, m16_gathered.
    line("N=256  g=16  m8",       runProj(variant: .m8,   nTokens: 256,  Din: Din, Dout: Dout, E: E, topK: topK))
    line("N=256  g=16  m16",      runProj(variant: .m16,  nTokens: 256,  Din: Din, Dout: Dout, E: E, topK: topK))
    line("N=256  g=16  m16_pipe", runProj(variant: .m16p, nTokens: 256,  Din: Din, Dout: Dout, E: E, topK: topK))
    line("N=256  g=16  m16_gath", runProj(variant: .m16g, nTokens: 256,  Din: Din, Dout: Dout, E: E, topK: topK))
    line("N=256  g=16  m16_KT  ", runProj(variant: .m16kt, nTokens: 256, Din: Din, Dout: Dout, E: E, topK: topK), "K-contiguous W")
    // N=2048 (g=128): heavy W reuse.
    line("N=2048 g=128 m8",       runProj(variant: .m8,   nTokens: 2048, Din: Din, Dout: Dout, E: E, topK: topK))
    line("N=2048 g=128 m16",      runProj(variant: .m16,  nTokens: 2048, Din: Din, Dout: Dout, E: E, topK: topK))
    line("N=2048 g=128 m16_pipe", runProj(variant: .m16p, nTokens: 2048, Din: Din, Dout: Dout, E: E, topK: topK))
    line("N=2048 g=128 m16_gath", runProj(variant: .m16g, nTokens: 2048, Din: Din, Dout: Dout, E: E, topK: topK))
    line("N=2048 g=128 m16_KT  ", runProj(variant: .m16kt, nTokens: 2048, Din: Din, Dout: Dout, E: E, topK: topK), "K-contiguous W")
    print("")
}
