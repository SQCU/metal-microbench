import Metal
import Foundation

// ===========================================================================
// Tile contract (the protocol future kernels will plug into)
// ===========================================================================
//
// A "tile stage" is a kernel that computes one fixed-size output block by
// consuming fixed-size input tiles staged in threadgroup memory.
//
// Conventions:
//   - Each threadgroup computes one output block. All state for that block
//     lives on-chip: input tiles in tg-mem, accumulator in simdgroup regs.
//   - Tile shapes are compile-time constants per stage (M_TG, N_TG, K_TG here).
//   - Cooperative load / store primitives move a tile between DRAM and tg-mem
//     with coalesced access; adjacent lanes read adjacent device addresses.
//   - Within a tg-scoped kernel, stages are separated by threadgroup_barrier
//     on mem_threadgroup. The next stage reads what the previous stage wrote.
//   - Future multi-stage fused kernels (qkv_proj → rope → attention → ...)
//     will layer additional named tile regions into the same tg-mem slab and
//     hand tiles off via barriers without touching DRAM between stages.
//
// GEMM stage layout:
//   M_TG × N_TG = 64 × 64 output block per threadgroup
//   4 SIMD groups per TG (128 threads), each owning a 32×32 sub-block (2×2 grid)
//   K is blocked in K_TG = 32; A_tile (64×32 half) + B_tile (32×64 half) = 8 KB
//   Accumulator = 4 × 4 × simdgroup_half8x8 per SIMD group (2 KB regs / sg)
// ===========================================================================

let mslSource = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// --- Tile-contract compile-time shape constants ---
constant constexpr uint M_TG = 64;
constant constexpr uint N_TG = 64;
constant constexpr uint K_TG = 32;
constant constexpr uint THREADS_PER_TG = 128;
constant constexpr uint SIMDS_PER_TG   = 4;

// --- Cooperative load primitives: DRAM → tg-mem via simdgroup_matrix path ---
// Each 8×8 fp16 subtile moves via `simdgroup_load` (DRAM → register) and
// `simdgroup_store` (register → tg-mem). This exercises the vectorized matrix
// datapath instead of scalar half loads (avoids 64-byte cacheline waste).
//
// A tile is 64×32 half = 8 subtile-rows × 4 subtile-cols = 32 subtiles.
// B tile is 32×64 half = 4 subtile-rows × 8 subtile-cols = 32 subtiles.
// 4 SIMD groups per TG each cover 8 subtiles of A and 8 subtiles of B.
static inline void load_A_tile(
    threadgroup half* A_tile,
    device const half* A, uint A_stride, uint2 A_origin,
    uint sg_id
) {
    simdgroup_half8x8 tmp;
    // 32 subtiles of A, strided by sg_id across 4 simdgroups → 8 each.
    for (uint s = sg_id; s < 32; s += SIMDS_PER_TG) {
        const uint sub_row = (s / 4) * 8;   // 0..56
        const uint sub_col = (s % 4) * 8;   // 0..24
        simdgroup_load(tmp,
                       A + (A_origin.y + sub_row) * A_stride + A_origin.x + sub_col,
                       A_stride);
        simdgroup_store(tmp,
                        A_tile + sub_row * K_TG + sub_col,
                        K_TG);
    }
}

static inline void load_B_tile(
    threadgroup half* B_tile,
    device const half* B, uint B_stride, uint2 B_origin,
    uint sg_id
) {
    simdgroup_half8x8 tmp;
    // 32 subtiles of B arranged as 4 rows × 8 cols.
    for (uint s = sg_id; s < 32; s += SIMDS_PER_TG) {
        const uint sub_row = (s / 8) * 8;   // 0..24
        const uint sub_col = (s % 8) * 8;   // 0..56
        simdgroup_load(tmp,
                       B + (B_origin.y + sub_row) * B_stride + B_origin.x + sub_col,
                       B_stride);
        simdgroup_store(tmp,
                        B_tile + sub_row * N_TG + sub_col,
                        N_TG);
    }
}

// --- Per-simdgroup GEMM accumulator & compute step ---
// One SIMD group owns a 32×32 output sub-block = 4 × 4 = 16 simdgroup_half8x8
// accumulator matrices, held in registers across the K loop.

// One K-substep of 8 (one simdgroup_multiply_accumulate sweep):
// loads 4 A tiles (32 rows × 8) + 4 B tiles (8 rows × 32) from tg-mem, runs
// 16 MACs. Ratio: 16 MACs / 8 tg-mem loads = 2 MAC/load on tg-mem side.
static inline void simdgroup_gemm_substep(
    threadgroup const half* A_tile,
    threadgroup const half* B_tile,
    thread simdgroup_half8x8 (&C_acc)[4][4],
    uint sg_row, uint sg_col, uint k_sub
) {
    simdgroup_half8x8 a[4], b[4];
    for (int i = 0; i < 4; ++i) {
        simdgroup_load(a[i], A_tile + (sg_row + i * 8) * K_TG + k_sub, K_TG);
    }
    for (int j = 0; j < 4; ++j) {
        simdgroup_load(b[j], B_tile + k_sub * N_TG + sg_col + j * 8, N_TG);
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
        }
    }
}

// --- Accumulator-count sweep (all no tg-mem, single simdgroup per TG) ---
// Differ only in output tile per simdgroup → register pressure scan.

// 1 accumulator: 8×8 output per TG (equivalent to naive matmul_simd8x8).
kernel void gemm_sg8x8(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]]
) {
    const uint row = tg_pos.y * 8;
    const uint col = tg_pos.x * 8;
    simdgroup_half8x8 a, b;
    simdgroup_half8x8 c = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(a, A + row * K + k, K);
        simdgroup_load(b, B + k * N + col, N);
        simdgroup_multiply_accumulate(c, a, b, c);
    }
    simdgroup_store(c, C + row * N + col, N);
}

// 4 accumulators (2×2): 16×16 output per TG.
kernel void gemm_sg16x16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]]
) {
    const uint row0 = tg_pos.y * 16;
    const uint col0 = tg_pos.x * 16;
    simdgroup_half8x8 C_acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C_acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    for (uint k = 0; k < K; k += 8) {
        simdgroup_half8x8 a[2], b[2];
        for (int i = 0; i < 2; ++i)
            simdgroup_load(a[i], A + (row0 + i * 8) * K + k, K);
        for (int j = 0; j < 2; ++j)
            simdgroup_load(b[j], B + k * N + col0 + j * 8, N);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            simdgroup_store(C_acc[i][j], C + (row0 + i * 8) * N + col0 + j * 8, N);
}

// 16 accumulators (4×4): 32×32 output per TG.
kernel void gemm_sg32x32(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]]
) {
    const uint row0 = tg_pos.y * 32;
    const uint col0 = tg_pos.x * 32;

    simdgroup_half8x8 C_acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            C_acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_half8x8 a[4], b[4];
        for (int i = 0; i < 4; ++i)
            simdgroup_load(a[i], A + (row0 + i * 8) * K + k, K);
        for (int j = 0; j < 4; ++j)
            simdgroup_load(b[j], B + k * N + col0 + j * 8, N);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
    }

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            simdgroup_store(C_acc[i][j],
                            C + (row0 + i * 8) * N + col0 + j * 8,
                            N);
}

// --- Q8 variant: B loaded as int8, dequantized in tg-mem before matmul ---
// B_device is M*N int8 bytes (half the DRAM footprint of fp16). Cooperative
// load reads int8, each thread casts to half with a global scale, writes fp16
// into B_tile. A_tile and compute path unchanged from v2.
// Measures: does dequant-in-tg-mem impose any runtime cost, and does the
// halved DRAM BW free up any headroom for concurrent work?
kernel void gemm_tiled_q8(
    device const half* A [[buffer(0)]],
    device const char* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TILE_M = 64;
    constexpr uint TILE_N = 64;
    constexpr uint TILE_K = 32;
    constexpr uint N_SIMDS = 16;
    constexpr half DEQUANT = half(1.0h / 127.0h);

    threadgroup half A_tile[TILE_M * TILE_K];
    threadgroup half B_tile[TILE_K * TILE_N];

    const uint lid = lid2.x;
    const uint sg_row = (sg_id / 4) * 16;
    const uint sg_col = (sg_id % 4) * 16;
    const uint C_row0 = tg_pos.y * TILE_M + sg_row;
    const uint C_col0 = tg_pos.x * TILE_N + sg_col;

    simdgroup_half8x8 C_acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C_acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        simdgroup_half8x8 tmp;
        for (uint s = sg_id; s < 32; s += N_SIMDS) {
            const uint a_r = (s / 4) * 8;
            const uint a_c = (s % 4) * 8;
            simdgroup_load(tmp, A + (tg_pos.y * TILE_M + a_r) * K + k_base + a_c, K);
            simdgroup_store(tmp, A_tile + a_r * TILE_K + a_c, TILE_K);
        }
        // B int8 → fp16 cooperative dequant-into-tg-mem.
        // 2048 bytes total, 512 threads × 4 bytes each.
        for (uint i = 0; i < 4; ++i) {
            const uint flat = lid + i * 512;
            const uint tile_row = flat / TILE_N;
            const uint tile_col = flat % TILE_N;
            char q = B[(k_base + tile_row) * N + tg_pos.x * TILE_N + tile_col];
            B_tile[tile_row * TILE_N + tile_col] = half(q) * DEQUANT;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
            simdgroup_half8x8 a[2], b[2];
            for (int i = 0; i < 2; ++i)
                simdgroup_load(a[i], A_tile + (sg_row + i * 8) * TILE_K + k_sub, TILE_K);
            for (int j = 0; j < 2; ++j)
                simdgroup_load(b[j], B_tile + k_sub * TILE_N + sg_col + j * 8, TILE_N);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            simdgroup_store(C_acc[i][j], C + (C_row0 + i * 8) * N + C_col0 + j * 8, N);
}

// --- v6: MLX-default-sized TG (32×32 output, 4 SGs × 4 accs, 128 threads) ---
// Delta from v5: halves the per-TG output tile to match MLX's BlockMMA
// defaults (BM=BN=32, BK=16, WM=WN=2). 4× more TGs → much higher
// per-core occupancy. Keeps float acc + padding + swizzle.
kernel void gemm_tiled_v6(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TILE_M = 32;
    constexpr uint TILE_N = 32;
    constexpr uint TILE_K = 16;
    constexpr uint K_PAD = 24;   // TILE_K + 8
    constexpr uint N_PAD = 40;   // TILE_N + 8
    constexpr uint N_SIMDS = 4;
    constexpr uint SWIZZLE_LOG = 3;

    const uint tiles_m = M / TILE_M;
    const uint tiles_n = N / TILE_N;
    const uint tid_y = (tg_pos.y << SWIZZLE_LOG) + (tg_pos.x & ((1u << SWIZZLE_LOG) - 1u));
    const uint tid_x = tg_pos.x >> SWIZZLE_LOG;
    if (tid_x >= tiles_n || tid_y >= tiles_m) return;

    threadgroup half A_tile[TILE_M * K_PAD];  // 32 * 24 = 768
    threadgroup half B_tile[TILE_K * N_PAD];  // 16 * 40 = 640

    // 2×2 SG layout within 32×32 output. Each SG owns 16×16 = 4 accs.
    const uint sg_row = (sg_id / 2) * 16;
    const uint sg_col = (sg_id % 2) * 16;
    const uint C_row0 = tid_y * TILE_M + sg_row;
    const uint C_col0 = tid_x * TILE_N + sg_col;

    simdgroup_float8x8 C_acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C_acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        // A_tile: 32 rows × 16 cols = 4×2 subtiles = 8 subtiles
        // B_tile: 16 rows × 32 cols = 2×4 subtiles = 8 subtiles
        // 4 SGs × 2 subtiles each
        simdgroup_half8x8 tmp;
        for (uint s = sg_id; s < 8; s += N_SIMDS) {
            const uint a_r = (s / 2) * 8;
            const uint a_c = (s % 2) * 8;
            simdgroup_load(tmp, A + (tid_y * TILE_M + a_r) * K + k_base + a_c, K);
            simdgroup_store(tmp, A_tile + a_r * K_PAD + a_c, K_PAD);
        }
        for (uint s = sg_id; s < 8; s += N_SIMDS) {
            const uint b_r = (s / 4) * 8;
            const uint b_c = (s % 4) * 8;
            simdgroup_load(tmp, B + (k_base + b_r) * N + tid_x * TILE_N + b_c, N);
            simdgroup_store(tmp, B_tile + b_r * N_PAD + b_c, N_PAD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
            simdgroup_half8x8 a[2], b[2];
            for (int i = 0; i < 2; ++i)
                simdgroup_load(a[i], A_tile + (sg_row + i * 8) * K_PAD + k_sub, K_PAD);
            for (int j = 0; j < 2; ++j)
                simdgroup_load(b[j], B_tile + k_sub * N_PAD + sg_col + j * 8, N_PAD);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            simdgroup_store(C_acc[i][j], C + (C_row0 + i * 8) * N + C_col0 + j * 8, N);
}

// --- v5: v4 + TG swizzle (L2-friendly block scheduling) ---
// Delta from v4: remap (tg_pos.x, tg_pos.y) so adjacent dispatched TGs share
// the same A-row block (tid_y). Host dispatches an oversized grid of
// tiles_n×8 × tiles_m/8 and the kernel discards out-of-range TGs.
kernel void gemm_tiled_v5(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TILE_M = 64;
    constexpr uint TILE_N = 64;
    constexpr uint TILE_K = 32;
    constexpr uint K_PAD = 40;
    constexpr uint N_PAD = 72;
    constexpr uint N_SIMDS = 16;
    constexpr uint SWIZZLE_LOG = 3;

    const uint tiles_m = M / TILE_M;
    const uint tiles_n = N / TILE_N;
    const uint tid_y = (tg_pos.y << SWIZZLE_LOG) + (tg_pos.x & ((1u << SWIZZLE_LOG) - 1u));
    const uint tid_x = tg_pos.x >> SWIZZLE_LOG;
    if (tid_x >= tiles_n || tid_y >= tiles_m) return;

    threadgroup half A_tile[TILE_M * K_PAD];
    threadgroup half B_tile[TILE_K * N_PAD];

    const uint sg_row = (sg_id / 4) * 16;
    const uint sg_col = (sg_id % 4) * 16;
    const uint C_row0 = tid_y * TILE_M + sg_row;
    const uint C_col0 = tid_x * TILE_N + sg_col;

    simdgroup_float8x8 C_acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C_acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        simdgroup_half8x8 tmp;
        for (uint s = sg_id; s < 32; s += N_SIMDS) {
            const uint a_r = (s / 4) * 8;
            const uint a_c = (s % 4) * 8;
            simdgroup_load(tmp, A + (tid_y * TILE_M + a_r) * K + k_base + a_c, K);
            simdgroup_store(tmp, A_tile + a_r * K_PAD + a_c, K_PAD);
        }
        for (uint s = sg_id; s < 32; s += N_SIMDS) {
            const uint b_r = (s / 8) * 8;
            const uint b_c = (s % 8) * 8;
            simdgroup_load(tmp, B + (k_base + b_r) * N + tid_x * TILE_N + b_c, N);
            simdgroup_store(tmp, B_tile + b_r * N_PAD + b_c, N_PAD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
            simdgroup_half8x8 a[2], b[2];
            for (int i = 0; i < 2; ++i)
                simdgroup_load(a[i], A_tile + (sg_row + i * 8) * K_PAD + k_sub, K_PAD);
            for (int j = 0; j < 2; ++j)
                simdgroup_load(b[j], B_tile + k_sub * N_PAD + sg_col + j * 8, N_PAD);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            simdgroup_store(C_acc[i][j], C + (C_row0 + i * 8) * N + C_col0 + j * 8, N);
}

// --- v4: v3 + tg-mem row padding (bank-conflict avoidance) ---
// Delta from v3: A_tile leading dim becomes K_PAD = TILE_K + 8 (16 B),
// B_tile leading dim becomes N_PAD = TILE_N + 8. Shifts addresses so
// 32-bank tg-mem doesn't collide on stride-power-of-2 8×8 reads.
kernel void gemm_tiled_v4(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TILE_M = 64;
    constexpr uint TILE_N = 64;
    constexpr uint TILE_K = 32;
    constexpr uint K_PAD = 40;   // TILE_K + 8
    constexpr uint N_PAD = 72;   // TILE_N + 8
    constexpr uint N_SIMDS = 16;

    threadgroup half A_tile[TILE_M * K_PAD];  // 64 * 40 = 2560 halves
    threadgroup half B_tile[TILE_K * N_PAD];  // 32 * 72 = 2304 halves

    const uint sg_row = (sg_id / 4) * 16;
    const uint sg_col = (sg_id % 4) * 16;
    const uint C_row0 = tg_pos.y * TILE_M + sg_row;
    const uint C_col0 = tg_pos.x * TILE_N + sg_col;

    simdgroup_float8x8 C_acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C_acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        simdgroup_half8x8 tmp;
        for (uint s = sg_id; s < 32; s += N_SIMDS) {
            const uint a_r = (s / 4) * 8;
            const uint a_c = (s % 4) * 8;
            simdgroup_load(tmp, A + (tg_pos.y * TILE_M + a_r) * K + k_base + a_c, K);
            simdgroup_store(tmp, A_tile + a_r * K_PAD + a_c, K_PAD);
        }
        for (uint s = sg_id; s < 32; s += N_SIMDS) {
            const uint b_r = (s / 8) * 8;
            const uint b_c = (s % 8) * 8;
            simdgroup_load(tmp, B + (k_base + b_r) * N + tg_pos.x * TILE_N + b_c, N);
            simdgroup_store(tmp, B_tile + b_r * N_PAD + b_c, N_PAD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
            simdgroup_half8x8 a[2], b[2];
            for (int i = 0; i < 2; ++i)
                simdgroup_load(a[i], A_tile + (sg_row + i * 8) * K_PAD + k_sub, K_PAD);
            for (int j = 0; j < 2; ++j)
                simdgroup_load(b[j], B_tile + k_sub * N_PAD + sg_col + j * 8, N_PAD);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            simdgroup_store(C_acc[i][j], C + (C_row0 + i * 8) * N + C_col0 + j * 8, N);
}

// --- v3: float32 accumulators (fp16 inputs, fp32 acc) ---
// Delta from v2: C_acc is simdgroup_float8x8 not half. MSL's
// simdgroup_multiply_accumulate supports mixed-precision (T_acc ≠ T_in).
// Output buffer becomes float to avoid a cast-store step for this bench.
kernel void gemm_tiled_v3(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TILE_M = 64;
    constexpr uint TILE_N = 64;
    constexpr uint TILE_K = 32;
    constexpr uint N_SIMDS = 16;

    threadgroup half A_tile[TILE_M * TILE_K];
    threadgroup half B_tile[TILE_K * TILE_N];

    const uint sg_row = (sg_id / 4) * 16;
    const uint sg_col = (sg_id % 4) * 16;
    const uint C_row0 = tg_pos.y * TILE_M + sg_row;
    const uint C_col0 = tg_pos.x * TILE_N + sg_col;

    simdgroup_float8x8 C_acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C_acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        simdgroup_half8x8 tmp;
        for (uint s = sg_id; s < 32; s += N_SIMDS) {
            const uint a_r = (s / 4) * 8;
            const uint a_c = (s % 4) * 8;
            simdgroup_load(tmp, A + (tg_pos.y * TILE_M + a_r) * K + k_base + a_c, K);
            simdgroup_store(tmp, A_tile + a_r * TILE_K + a_c, TILE_K);
        }
        for (uint s = sg_id; s < 32; s += N_SIMDS) {
            const uint b_r = (s / 8) * 8;
            const uint b_c = (s % 8) * 8;
            simdgroup_load(tmp, B + (k_base + b_r) * N + tg_pos.x * TILE_N + b_c, N);
            simdgroup_store(tmp, B_tile + b_r * TILE_N + b_c, TILE_N);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
            simdgroup_half8x8 a[2], b[2];
            for (int i = 0; i < 2; ++i)
                simdgroup_load(a[i], A_tile + (sg_row + i * 8) * TILE_K + k_sub, TILE_K);
            for (int j = 0; j < 2; ++j)
                simdgroup_load(b[j], B_tile + k_sub * TILE_N + sg_col + j * 8, TILE_N);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            simdgroup_store(C_acc[i][j], C + (C_row0 + i * 8) * N + C_col0 + j * 8, N);
}

// --- Tile-contract GEMM v2: 16 SGs × 4 accs each, 64×64 per TG ---
// Keeps every simdgroup at 4 accumulators (sweet spot from the sweep) while
// letting 16 simdgroups share a single A/B staging in tg-mem. 512 threads/TG.
kernel void gemm_tiled_v2(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TILE_M = 64;
    constexpr uint TILE_N = 64;
    constexpr uint TILE_K = 32;
    constexpr uint N_SIMDS = 16;  // 4×4 arrangement within 64×64 output

    threadgroup half A_tile[TILE_M * TILE_K];  // 4 KB
    threadgroup half B_tile[TILE_K * TILE_N];  // 4 KB

    // Per-SG sub-tile: 16×16 output = 2×2 of 8×8 accumulators
    const uint sg_row = (sg_id / 4) * 16;  // 0, 16, 32, 48
    const uint sg_col = (sg_id % 4) * 16;
    const uint C_row0 = tg_pos.y * TILE_M + sg_row;
    const uint C_col0 = tg_pos.x * TILE_N + sg_col;

    simdgroup_half8x8 C_acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C_acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        // Cooperative DRAM→tg-mem load. A_tile has 32 8×8 subtiles (8×4),
        // B_tile has 32 8×8 subtiles (4×8). 16 SGs each handle 2 of each.
        simdgroup_half8x8 tmp;
        for (uint s = sg_id; s < 32; s += N_SIMDS) {
            const uint a_r = (s / 4) * 8;
            const uint a_c = (s % 4) * 8;
            simdgroup_load(tmp,
                           A + (tg_pos.y * TILE_M + a_r) * K + k_base + a_c,
                           K);
            simdgroup_store(tmp, A_tile + a_r * TILE_K + a_c, TILE_K);
        }
        for (uint s = sg_id; s < 32; s += N_SIMDS) {
            const uint b_r = (s / 8) * 8;
            const uint b_c = (s % 8) * 8;
            simdgroup_load(tmp,
                           B + (k_base + b_r) * N + tg_pos.x * TILE_N + b_c,
                           N);
            simdgroup_store(tmp, B_tile + b_r * TILE_N + b_c, TILE_N);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: 4 K-sub iters × (2 A + 2 B tg-mem loads + 4 MACs)
        for (uint k_sub = 0; k_sub < TILE_K; k_sub += 8) {
            simdgroup_half8x8 a[2], b[2];
            for (int i = 0; i < 2; ++i)
                simdgroup_load(a[i], A_tile + (sg_row + i * 8) * TILE_K + k_sub, TILE_K);
            for (int j = 0; j < 2; ++j)
                simdgroup_load(b[j], B_tile + k_sub * TILE_N + sg_col + j * 8, TILE_N);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    simdgroup_multiply_accumulate(C_acc[i][j], a[i], b[j], C_acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            simdgroup_store(C_acc[i][j], C + (C_row0 + i * 8) * N + C_col0 + j * 8, N);
}

// --- Top-level GEMM stage as tile-contract instance ---
kernel void gemm_tiled(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]]
) {
    const uint lid = lid2.x;
    threadgroup half A_tile[M_TG * K_TG];  // 4 KB
    threadgroup half B_tile[K_TG * N_TG];  // 4 KB

    // 2×2 SIMD-group grid within the 64×64 output block
    const uint sg_row = (sg_id / 2) * 32;  // 0 or 32
    const uint sg_col = (sg_id % 2) * 32;

    // Output origin in C
    const uint C_row0 = tg_pos.y * M_TG + sg_row;
    const uint C_col0 = tg_pos.x * N_TG + sg_col;

    // Accumulator in SIMD-group registers
    simdgroup_half8x8 C_acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            C_acc[i][j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    // K loop: stage one K_TG-wide slice of A and B at a time
    (void)lid;  // no longer needed by simdgroup-based loads
    for (uint k_base = 0; k_base < K; k_base += K_TG) {
        load_A_tile(A_tile, A, K, uint2(k_base, tg_pos.y * M_TG), sg_id);
        load_B_tile(B_tile, B, N, uint2(tg_pos.x * N_TG, k_base), sg_id);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_sub = 0; k_sub < K_TG; k_sub += 8) {
            simdgroup_gemm_substep(A_tile, B_tile, C_acc, sg_row, sg_col, k_sub);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store 32×32 sub-block to device memory (4×4 simdgroup stores)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            simdgroup_store(C_acc[i][j],
                            C + (C_row0 + i * 8) * N + C_col0 + j * 8,
                            N);
        }
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

guard let tiledFn  = library.makeFunction(name: "gemm_tiled"),
      let tiled2Fn = library.makeFunction(name: "gemm_tiled_v2"),
      let tiled3Fn = library.makeFunction(name: "gemm_tiled_v3"),
      let tiled4Fn = library.makeFunction(name: "gemm_tiled_v4"),
      let tiled5Fn = library.makeFunction(name: "gemm_tiled_v5"),
      let tiled6Fn = library.makeFunction(name: "gemm_tiled_v6"),
      let q8Fn     = library.makeFunction(name: "gemm_tiled_q8"),
      let sg8Fn    = library.makeFunction(name: "gemm_sg8x8"),
      let sg16Fn   = library.makeFunction(name: "gemm_sg16x16"),
      let sg32Fn   = library.makeFunction(name: "gemm_sg32x32") else { fail("kernel lookup failed") }
let tiledPSO  = try! device.makeComputePipelineState(function: tiledFn)
let tiled2PSO = try! device.makeComputePipelineState(function: tiled2Fn)
let tiled3PSO = try! device.makeComputePipelineState(function: tiled3Fn)
let tiled4PSO = try! device.makeComputePipelineState(function: tiled4Fn)
let tiled5PSO = try! device.makeComputePipelineState(function: tiled5Fn)
let tiled6PSO = try! device.makeComputePipelineState(function: tiled6Fn)
let q8PSO     = try! device.makeComputePipelineState(function: q8Fn)
let sg8PSO    = try! device.makeComputePipelineState(function: sg8Fn)
let sg16PSO   = try! device.makeComputePipelineState(function: sg16Fn)
let sg32PSO   = try! device.makeComputePipelineState(function: sg32Fn)

print("device: \(device.name)")
print("")

func makeFilledBufferHalf(_ elements: Int, seed: UInt32) -> MTLBuffer {
    let b = device.makeBuffer(length: elements * 2, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: Float16.self, capacity: elements)
    var s = seed
    for i in 0..<elements {
        s = s &* 1664525 &+ 1013904223
        p[i] = Float16(Float(Int32(bitPattern: s) % 17) / 17.0)
    }
    return b
}

enum Variant { case tiled, tiledV2, tiledV3, tiledV4, tiledV5, tiledV6, q8, sg8x8, sg16x16, sg32x32 }

func runGemm(variant: Variant, M: Int, N: Int, K: Int, iters: Int = 10, warmup: Int = 3) {
    let A = makeFilledBufferHalf(M * K, seed: 0x12345678)
    let B: MTLBuffer
    if case .q8 = variant {
        // int8 B buffer, K*N bytes, filled with byte pattern
        B = device.makeBuffer(length: K * N, options: .storageModeShared)!
        let p = B.contents().bindMemory(to: Int8.self, capacity: K * N)
        var s: UInt32 = 0x87654321
        for i in 0..<(K * N) {
            s = s &* 1664525 &+ 1013904223
            p[i] = Int8(bitPattern: UInt8(s & 0xFF))
        }
    } else {
        B = makeFilledBufferHalf(K * N, seed: 0x87654321)
    }
    let outBytes: Int
    switch variant {
    case .tiledV3, .tiledV4, .tiledV5, .tiledV6: outBytes = 4
    default:                                     outBytes = 2
    }
    let C = device.makeBuffer(length: M * N * outBytes, options: .storageModeShared)!

    let pso: MTLComputePipelineState
    let grid: MTLSize
    let tg: MTLSize
    let name: String
    switch variant {
    case .tiled:
        assert(M % 64 == 0 && N % 64 == 0 && K % 32 == 0)
        pso = tiledPSO
        grid = MTLSize(width: N / 64, height: M / 64, depth: 1)
        tg = MTLSize(width: 128, height: 1, depth: 1)
        name = "gemm_tiled    (16 acc, 4 SG, tg-mem)"
    case .tiledV2:
        assert(M % 64 == 0 && N % 64 == 0 && K % 32 == 0)
        pso = tiled2PSO
        grid = MTLSize(width: N / 64, height: M / 64, depth: 1)
        tg = MTLSize(width: 512, height: 1, depth: 1)
        name = "gemm_tiled_v2 (half acc)       "
    case .tiledV3:
        assert(M % 64 == 0 && N % 64 == 0 && K % 32 == 0)
        pso = tiled3PSO
        grid = MTLSize(width: N / 64, height: M / 64, depth: 1)
        tg = MTLSize(width: 512, height: 1, depth: 1)
        name = "gemm_tiled_v3 (+float acc)     "
    case .tiledV4:
        assert(M % 64 == 0 && N % 64 == 0 && K % 32 == 0)
        pso = tiled4PSO
        grid = MTLSize(width: N / 64, height: M / 64, depth: 1)
        tg = MTLSize(width: 512, height: 1, depth: 1)
        name = "gemm_tiled_v4 (+tg-mem padding)"
    case .tiledV5:
        assert(M % 64 == 0 && N % 64 == 0 && K % 32 == 0)
        pso = tiled5PSO
        grid = MTLSize(width: (N / 64) * 8, height: (M / 64 + 7) / 8, depth: 1)
        tg = MTLSize(width: 512, height: 1, depth: 1)
        name = "gemm_tiled_v5 (+TG swizzle)    "
    case .tiledV6:
        assert(M % 32 == 0 && N % 32 == 0 && K % 16 == 0)
        pso = tiled6PSO
        grid = MTLSize(width: (N / 32) * 8, height: (M / 32 + 7) / 8, depth: 1)
        tg = MTLSize(width: 128, height: 1, depth: 1)
        name = "gemm_tiled_v6 (MLX-sized TG)   "
    case .q8:
        assert(M % 64 == 0 && N % 64 == 0 && K % 32 == 0)
        pso = q8PSO
        grid = MTLSize(width: N / 64, height: M / 64, depth: 1)
        tg = MTLSize(width: 512, height: 1, depth: 1)
        name = "gemm_tiled_q8 (int8 B dequant) "
    case .sg8x8:
        assert(M % 8 == 0 && N % 8 == 0 && K % 8 == 0)
        pso = sg8PSO
        grid = MTLSize(width: N / 8, height: M / 8, depth: 1)
        tg = MTLSize(width: 32, height: 1, depth: 1)
        name = "gemm_sg8x8    (1 acc)"
    case .sg16x16:
        assert(M % 16 == 0 && N % 16 == 0 && K % 8 == 0)
        pso = sg16PSO
        grid = MTLSize(width: N / 16, height: M / 16, depth: 1)
        tg = MTLSize(width: 32, height: 1, depth: 1)
        name = "gemm_sg16x16  (4 acc)"
    case .sg32x32:
        assert(M % 32 == 0 && N % 32 == 0 && K % 8 == 0)
        pso = sg32PSO
        grid = MTLSize(width: N / 32, height: M / 32, depth: 1)
        tg = MTLSize(width: 32, height: 1, depth: 1)
        name = "gemm_sg32x32  (16 acc)"
    }

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(A, offset: 0, index: 0)
        enc.setBuffer(B, offset: 0, index: 1)
        enc.setBuffer(C, offset: 0, index: 2)
        var Mu = UInt32(M), Nu = UInt32(N), Ku = UInt32(K)
        enc.setBytes(&Mu, length: 4, index: 3)
        enc.setBytes(&Nu, length: 4, index: 4)
        enc.setBytes(&Ku, length: 4, index: 5)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            fputs("GPU error: \(err)\n", stderr); exit(1)
        }
        if i >= warmup {
            times.append(cb.gpuEndTime - cb.gpuStartTime)
        }
    }
    let best = times.min()!
    let flops = 2.0 * Double(M) * Double(N) * Double(K)
    let tflops = flops / best / 1e12
    let label = "\(name)  \(M)x\(N)x\(K)".padding(toLength: 56, withPad: " ", startingAt: 0)
    let msStr = String(format: "%8.3f ms", best * 1000)
    let tfStr = String(format: "%.2f TFLOPS", tflops)
    print("  \(label) \(msStr)   \(tfStr)")
}

print("=== fp16 baseline vs int8 dequant-at-load ===")
for size in [(2048,2048,2048), (4096,4096,4096)] {
    runGemm(variant: .tiledV2, M: size.0, N: size.1, K: size.2)
    runGemm(variant: .q8,      M: size.0, N: size.1, K: size.2)
    print("")
}

// --- Gemma-4-A4B dense projections ---
// Shapes from config.json: hidden_size=2816, intermediate_size=2112 (shared
// expert FFN), moe_intermediate_size=704 (per-expert FFN). QKV/out projection
// land at hidden_size × hidden_size. M is token count: batched-decode B=128
// or prefill chunks.
print("=== Gemma-4-A4B dense-projection shapes ===")
print("  (M=token count; batch-128 decode vs 2048-token prefill chunk)")
let gemmaDense: [(String, Int, Int)] = [
    ("QKV / out proj   ", 2816, 2816),
    ("shared FFN gate/up", 2816, 2112),
    ("shared FFN down   ", 2112, 2816),
]
for (label, K, N) in gemmaDense {
    for M in [128, 2048] {
        print("  \(label) M=\(M)")
        runGemm(variant: .tiledV2, M: M, N: N, K: K)
        runGemm(variant: .q8,      M: M, N: N, K: K)
    }
    print("")
}
