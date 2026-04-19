import Metal
import Foundation

// ===========================================================================
// Paged decode attention — single-slot (batch=1, H=1) first variant.
//
// Tile-contract equivalents:
//   - Q tile: [D]          (one query token; stays in tg-mem for all K pages)
//   - K/V page: [PAGE][D]  (cooperatively loaded from DRAM per outer iter)
//   - scores: [PAGE]       (computed per page, reduced to max, exp'd in place)
//   - Running softmax state: (m, ℓ) pair in tg-mem + float O_acc[D]
//
// Flash online-softmax update per page p:
//   S_p   = Q · K_p^T                              ; apply score_mod
//   m'    = max(m, max(S_p))
//   scale = exp(m - m')
//   O     = O·scale + exp(S_p - m') · V_p
//   ℓ     = ℓ·scale + sum(exp(S_p - m'))
//   m     = m'
// Final: O = O / ℓ
//
// Intent: this kernel is the skeleton. Batched multi-slot decode reuses the
// primitives with one TG per slot; prefill reuses primitives with contiguous
// K-chunks and a per-Q-chunk outer loop.
// ===========================================================================

func mslSource(D: Int, PAGE: Int) -> String { return """
#include <metal_stdlib>
using namespace metal;

constant constexpr uint D = \(D);
constant constexpr uint PAGE = \(PAGE);
constant constexpr uint THREADS = 32;

// Shared impl. tg-mem must be declared in the calling kernel and passed in.
static inline void decode_attn_impl(
    threadgroup half* Q_tile,
    threadgroup half* K_tile,
    threadgroup half* V_tile,
    threadgroup float* scores,
    threadgroup float* m_state,
    threadgroup float* l_state,
    threadgroup float* O_acc,
    threadgroup float* page_max,
    device const half* Q,
    device const half* K_cache,
    device const half* V_cache,
    device const uint* block_table,
    device half* O,
    const uint num_pages,
    const uint k_len,
    const uint max_pages_per_slot,
    const float qk_scale,
    const uint slot,
    const uint lid
) {
    // --- init ---
    device const half* Q_s = Q + slot * D;
    device half* O_s = O + slot * D;
    device const uint* bt_s = block_table + slot * max_pages_per_slot;

    for (uint i = lid; i < D; i += THREADS) {
        Q_tile[i] = Q_s[i];
        O_acc[i] = 0.0f;
    }
    if (lid == 0) {
        m_state[0] = -INFINITY;
        l_state[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- loop over logical pages ---
    for (uint p = 0; p < num_pages; ++p) {
        const uint phys = bt_s[p];

        // Cooperative load of K and V pages (PAGE*D = 2048 halves each)
        for (uint i = lid; i < PAGE * D; i += THREADS) {
            K_tile[i] = K_cache[phys * PAGE * D + i];
            V_tile[i] = V_cache[phys * PAGE * D + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scores S[r] per active lane — in register, no tg-mem write.
        float my_score = -INFINITY;
        if (lid < PAGE) {
            float s = 0.0f;
            for (uint d = 0; d < D; ++d) {
                s += float(Q_tile[d]) * float(K_tile[lid * D + d]);
            }
            s *= qk_scale;
            uint k_pos = p * PAGE + lid;
            if (k_pos < k_len) my_score = s;
        }

        // SIMD-wide max and sum — no serial thread-0 work.
        float page_max_val = simd_max(my_score);
        float m_old = m_state[0];
        float m_new = max(m_old, page_max_val);
        float scale = exp(m_old - m_new);
        float my_exp = (lid < PAGE) ? exp(my_score - m_new) : 0.0f;
        float page_sum = simd_sum(my_exp);

        // Stage exp'd scores into tg-mem for the O update that follows.
        if (lid < PAGE) scores[lid] = my_exp;
        if (lid == 0) {
            l_state[0] = l_state[0] * scale + page_sum;
            m_state[0] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O_acc = O_acc * scale + sum_r scores[r] * V_tile[r][d]
        // Each thread owns D/THREADS = 4 elements of O_acc (D=128, THREADS=32).
        for (uint d = lid; d < D; d += THREADS) {
            float acc = O_acc[d] * scale;
            for (uint r = 0; r < PAGE; ++r) {
                acc += scores[r] * float(V_tile[r * D + d]);
            }
            O_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalize and store.
    const float l_final = l_state[0];
    for (uint d = lid; d < D; d += THREADS) {
        O_s[d] = half(O_acc[d] / l_final);
    }
}

kernel void decode_attn_single(
    device const half* Q            [[buffer(0)]],
    device const half* K_cache      [[buffer(1)]],
    device const half* V_cache      [[buffer(2)]],
    device const uint* block_table  [[buffer(3)]],
    device half* O                  [[buffer(4)]],
    constant uint& num_pages        [[buffer(5)]],
    constant uint& k_len            [[buffer(6)]],
    constant float& qk_scale        [[buffer(7)]],
    uint2 lid2                      [[thread_position_in_threadgroup]]
) {
    threadgroup half Q_tile[D];
    threadgroup half K_tile[PAGE * D];
    threadgroup half V_tile[PAGE * D];
    threadgroup float scores[PAGE];
    threadgroup float m_state[1];
    threadgroup float l_state[1];
    threadgroup float O_acc[D];
    threadgroup float page_max[1];
    decode_attn_impl(Q_tile, K_tile, V_tile, scores, m_state, l_state, O_acc, page_max,
                     Q, K_cache, V_cache, block_table, O,
                     num_pages, k_len, num_pages, qk_scale,
                     0, lid2.x);
}

// Split-KV impl: same Flash loop but over page range [page_start, page_end),
// and writes un-normalized (m, l, O_acc) to partials buffers instead of
// producing final O. No final normalization — that happens in the reduce
// kernel after combining across splits.
static inline void decode_attn_split_impl(
    threadgroup half* Q_tile,
    threadgroup half* K_tile,
    threadgroup half* V_tile,
    threadgroup float* scores,
    threadgroup float* m_state,
    threadgroup float* l_state,
    threadgroup float* O_acc,
    device const half* Q,
    device const half* K_cache,
    device const half* V_cache,
    device const uint* block_table,
    device float* m_partials,   // [slot, split]
    device float* l_partials,   // [slot, split]
    device float* O_partials,   // [slot, split, D]
    const uint k_len,
    const uint max_pages_per_slot,
    const uint page_start,
    const uint page_end,
    const uint num_splits,
    const float qk_scale,
    const uint slot,
    const uint split,
    const uint lid
) {
    device const half* Q_s = Q + slot * D;
    device const uint* bt_s = block_table + slot * max_pages_per_slot;

    for (uint i = lid; i < D; i += THREADS) {
        Q_tile[i] = Q_s[i];
        O_acc[i] = 0.0f;
    }
    if (lid == 0) {
        m_state[0] = -INFINITY;
        l_state[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = page_start; p < page_end; ++p) {
        const uint phys = bt_s[p];
        for (uint i = lid; i < PAGE * D; i += THREADS) {
            K_tile[i] = K_cache[phys * PAGE * D + i];
            V_tile[i] = V_cache[phys * PAGE * D + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float my_score = -INFINITY;
        if (lid < PAGE) {
            float s = 0.0f;
            for (uint d = 0; d < D; ++d)
                s += float(Q_tile[d]) * float(K_tile[lid * D + d]);
            s *= qk_scale;
            uint k_pos = p * PAGE + lid;
            if (k_pos < k_len) my_score = s;
        }
        float page_max_val = simd_max(my_score);
        float m_old = m_state[0];
        float m_new = max(m_old, page_max_val);
        float scale = exp(m_old - m_new);
        float my_exp = (lid < PAGE) ? exp(my_score - m_new) : 0.0f;
        float page_sum = simd_sum(my_exp);

        if (lid < PAGE) scores[lid] = my_exp;
        if (lid == 0) {
            l_state[0] = l_state[0] * scale + page_sum;
            m_state[0] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = lid; d < D; d += THREADS) {
            float acc = O_acc[d] * scale;
            for (uint r = 0; r < PAGE; ++r) {
                acc += scores[r] * float(V_tile[r * D + d]);
            }
            O_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials. O_acc is un-normalized on purpose.
    const uint idx = slot * num_splits + split;
    if (lid == 0) {
        // If this split got no pages, write sentinels so reduce ignores it.
        const bool empty = (page_start >= page_end);
        m_partials[idx] = empty ? -INFINITY : m_state[0];
        l_partials[idx] = empty ? 0.0f : l_state[0];
    }
    device float* O_part = O_partials + idx * D;
    for (uint d = lid; d < D; d += THREADS) {
        O_part[d] = O_acc[d];
    }
}

kernel void decode_attn_split_compute(
    device const half* Q                 [[buffer(0)]],
    device const half* K_cache           [[buffer(1)]],
    device const half* V_cache           [[buffer(2)]],
    device const uint* block_table       [[buffer(3)]],
    device const uint* num_pages_per_slot[[buffer(4)]],
    device const uint* k_len_per_slot    [[buffer(5)]],
    device float* m_partials             [[buffer(6)]],
    device float* l_partials             [[buffer(7)]],
    device float* O_partials             [[buffer(8)]],
    constant float& qk_scale             [[buffer(9)]],
    constant uint& max_pages_per_slot    [[buffer(10)]],
    constant uint& num_splits            [[buffer(11)]],
    uint2 tg_pos                         [[threadgroup_position_in_grid]],
    uint2 lid2                           [[thread_position_in_threadgroup]]
) {
    const uint slot = tg_pos.x;
    const uint split = tg_pos.y;
    const uint total_pages = num_pages_per_slot[slot];
    const uint pages_per_split = (total_pages + num_splits - 1) / num_splits;
    const uint page_start = split * pages_per_split;
    const uint page_end = min((split + 1) * pages_per_split, total_pages);

    threadgroup half Q_tile[D];
    threadgroup half K_tile[PAGE * D];
    threadgroup half V_tile[PAGE * D];
    threadgroup float scores[PAGE];
    threadgroup float m_state[1];
    threadgroup float l_state[1];
    threadgroup float O_acc[D];
    decode_attn_split_impl(Q_tile, K_tile, V_tile, scores, m_state, l_state, O_acc,
                           Q, K_cache, V_cache, block_table,
                           m_partials, l_partials, O_partials,
                           k_len_per_slot[slot],
                           max_pages_per_slot,
                           page_start, page_end, num_splits,
                           qk_scale, slot, split, lid2.x);
}

// Prefill: one TG per Q token. Contiguous K/V access (no block table),
// causal mask via k_pos <= q_pos. Grid = (S_q,). Each TG's K-loop runs
// (q_pos+1) positions. Load imbalance across TGs (later q_pos does more
// work) is fine — the GPU scheduler will oversubscribe cores to amortize.
kernel void prefill_attn_causal(
    device const half* Q         [[buffer(0)]],   // [S_q, D]
    device const half* K         [[buffer(1)]],   // [S_kv, D]
    device const half* V         [[buffer(2)]],   // [S_kv, D]
    device half* O               [[buffer(3)]],   // [S_q, D]
    constant uint& S_q           [[buffer(4)]],
    constant uint& S_kv          [[buffer(5)]],
    constant float& qk_scale     [[buffer(6)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint2 lid2                   [[thread_position_in_threadgroup]]
) {
    const uint q_pos = tg_pos.x;
    const uint lid = lid2.x;
    if (q_pos >= S_q) return;

    threadgroup half Q_tile[D];
    threadgroup half K_tile[PAGE * D];
    threadgroup half V_tile[PAGE * D];
    threadgroup float scores[PAGE];
    threadgroup float m_state[1];
    threadgroup float l_state[1];
    threadgroup float O_acc[D];

    for (uint i = lid; i < D; i += THREADS) {
        Q_tile[i] = Q[q_pos * D + i];
        O_acc[i] = 0.0f;
    }
    if (lid == 0) { m_state[0] = -INFINITY; l_state[0] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k_end = q_pos + 1;                 // causal
    const uint num_pages = (k_end + PAGE - 1) / PAGE;

    for (uint p = 0; p < num_pages; ++p) {
        for (uint i = lid; i < PAGE * D; i += THREADS) {
            K_tile[i] = K[p * PAGE * D + i];
            V_tile[i] = V[p * PAGE * D + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float my_score = -INFINITY;
        if (lid < PAGE) {
            float s = 0.0f;
            for (uint d = 0; d < D; ++d)
                s += float(Q_tile[d]) * float(K_tile[lid * D + d]);
            s *= qk_scale;
            uint k_pos = p * PAGE + lid;
            if (k_pos < k_end) my_score = s;  // causal mask + pad mask combined
        }
        float page_max_val = simd_max(my_score);
        float m_old = m_state[0];
        float m_new = max(m_old, page_max_val);
        float scale = exp(m_old - m_new);
        float my_exp = (lid < PAGE) ? exp(my_score - m_new) : 0.0f;
        float page_sum = simd_sum(my_exp);

        if (lid < PAGE) scores[lid] = my_exp;
        if (lid == 0) {
            l_state[0] = l_state[0] * scale + page_sum;
            m_state[0] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = lid; d < D; d += THREADS) {
            float acc = O_acc[d] * scale;
            for (uint r = 0; r < PAGE; ++r) {
                acc += scores[r] * float(V_tile[r * D + d]);
            }
            O_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float l_final = l_state[0];
    for (uint d = lid; d < D; d += THREADS) {
        O[q_pos * D + d] = half(O_acc[d] / l_final);
    }
}

kernel void decode_attn_split_reduce(
    device const float* m_partials [[buffer(0)]],
    device const float* l_partials [[buffer(1)]],
    device const float* O_partials [[buffer(2)]],
    device half* O                 [[buffer(3)]],
    constant uint& num_splits      [[buffer(4)]],
    uint2 tg_pos                   [[threadgroup_position_in_grid]],
    uint2 lid2                     [[thread_position_in_threadgroup]]
) {
    const uint slot = tg_pos.x;
    const uint lid = lid2.x;

    // Load per-split (m, l) into per-lane registers; unused lanes get sentinels.
    float my_m = -INFINITY;
    float my_l = 0.0f;
    if (lid < num_splits) {
        my_m = m_partials[slot * num_splits + lid];
        my_l = l_partials[slot * num_splits + lid];
    }
    float m_global = simd_max(my_m);
    float my_scale = (my_m == -INFINITY) ? 0.0f : exp(my_m - m_global);
    float l_global = simd_sum(my_scale * my_l);

    // Broadcast per-split scales to all threads via tg-mem.
    threadgroup float scales_tg[32];
    if (lid < num_splits) scales_tg[lid] = my_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread handles D/THREADS output elements.
    for (uint d = lid; d < D; d += THREADS) {
        float acc = 0.0f;
        for (uint s = 0; s < num_splits; ++s) {
            acc += scales_tg[s] * O_partials[(slot * num_splits + s) * D + d];
        }
        O[slot * D + d] = half(acc / l_global);
    }
}

// Batched: one TG per slot. Each slot has its own num_pages/k_len/block_table
// entry — stored as parallel arrays indexed by slot. All slots share K_cache
// and V_cache via block_table indirection.
kernel void decode_attn_batched(
    device const half* Q                 [[buffer(0)]],   // [batch, D]
    device const half* K_cache           [[buffer(1)]],
    device const half* V_cache           [[buffer(2)]],
    device const uint* block_table       [[buffer(3)]],   // [batch, max_pages]
    device half* O                       [[buffer(4)]],   // [batch, D]
    device const uint* num_pages_per_slot[[buffer(5)]],
    device const uint* k_len_per_slot    [[buffer(6)]],
    constant float& qk_scale             [[buffer(7)]],
    constant uint& max_pages_per_slot    [[buffer(8)]],
    uint2 tg_pos                         [[threadgroup_position_in_grid]],
    uint2 lid2                           [[thread_position_in_threadgroup]]
) {
    const uint slot = tg_pos.x;
    threadgroup half Q_tile[D];
    threadgroup half K_tile[PAGE * D];
    threadgroup half V_tile[PAGE * D];
    threadgroup float scores[PAGE];
    threadgroup float m_state[1];
    threadgroup float l_state[1];
    threadgroup float O_acc[D];
    threadgroup float page_max[1];
    decode_attn_impl(Q_tile, K_tile, V_tile, scores, m_state, l_state, O_acc, page_max,
                     Q, K_cache, V_cache, block_table, O,
                     num_pages_per_slot[slot],
                     k_len_per_slot[slot],
                     max_pages_per_slot,
                     qk_scale,
                     slot, lid2.x);
}
""" }

func fail(_ msg: String) -> Never {
    fputs("error: \(msg)\n", stderr); exit(1)
}

guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let queue = device.makeCommandQueue()!
let opts = MTLCompileOptions()
if #available(macOS 15.0, *) { opts.languageVersion = .version3_2 }

struct AttnPSOs {
    let single: MTLComputePipelineState
    let batch: MTLComputePipelineState
    let splitComp: MTLComputePipelineState
    let splitRed: MTLComputePipelineState
    let prefill: MTLComputePipelineState
    let D: Int
    let PAGE: Int
}

func buildPSOs(D: Int, PAGE: Int) -> AttnPSOs {
    let lib: MTLLibrary
    do { lib = try device.makeLibrary(source: mslSource(D: D, PAGE: PAGE), options: opts) }
    catch { fail("MSL compile (D=\(D) PAGE=\(PAGE)): \(error)") }
    guard let singleFn    = lib.makeFunction(name: "decode_attn_single"),
          let batchFn     = lib.makeFunction(name: "decode_attn_batched"),
          let splitCompFn = lib.makeFunction(name: "decode_attn_split_compute"),
          let splitRedFn  = lib.makeFunction(name: "decode_attn_split_reduce"),
          let prefillFn   = lib.makeFunction(name: "prefill_attn_causal") else { fail("no kernel") }
    return AttnPSOs(
        single:    try! device.makeComputePipelineState(function: singleFn),
        batch:     try! device.makeComputePipelineState(function: batchFn),
        splitComp: try! device.makeComputePipelineState(function: splitCompFn),
        splitRed:  try! device.makeComputePipelineState(function: splitRedFn),
        prefill:   try! device.makeComputePipelineState(function: prefillFn),
        D: D, PAGE: PAGE)
}

let psos128 = buildPSOs(D: 128, PAGE: 16)   // legacy head_dim baseline
let psos256 = buildPSOs(D: 256, PAGE: 16)   // Gemma-4 sliding attention
let psos512 = buildPSOs(D: 512, PAGE: 8)    // Gemma-4 full attention (smaller PAGE to fit tg-mem)

print("device: \(device.name)")
print("kernel (D=128): threadExecWidth=\(psos128.single.threadExecutionWidth) maxTPerTG=\(psos128.single.maxTotalThreadsPerThreadgroup) staticTGMem=\(psos128.single.staticThreadgroupMemoryLength) B")
print("kernel (D=256): staticTGMem=\(psos256.single.staticThreadgroupMemoryLength) B")
print("kernel (D=512): staticTGMem=\(psos512.single.staticThreadgroupMemoryLength) B")
print("")

func makeFilledHalfBuf(_ elements: Int, seed: UInt32, scale: Float = 0.1) -> MTLBuffer {
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

func runDecode(psos: AttnPSOs, kLen: Int, iters: Int = 30, warmup: Int = 5) -> Double {
    let D = psos.D
    let PAGE = psos.PAGE
    let numPages = (kLen + PAGE - 1) / PAGE

    let Q = makeFilledHalfBuf(D, seed: 0x11111111)

    // For simplicity: block_table is the identity mapping (logical page i → physical page i).
    // In real paged attention, multiple slots would share physical pages.
    let K = makeFilledHalfBuf(numPages * PAGE * D, seed: 0x22222222)
    let V = makeFilledHalfBuf(numPages * PAGE * D, seed: 0x33333333)

    let blockTable = device.makeBuffer(length: numPages * 4, options: .storageModeShared)!
    let btp = blockTable.contents().bindMemory(to: UInt32.self, capacity: numPages)
    for i in 0..<numPages { btp[i] = UInt32(i) }

    let O = device.makeBuffer(length: D * 2, options: .storageModeShared)!

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psos.single)
        enc.setBuffer(Q, offset: 0, index: 0)
        enc.setBuffer(K, offset: 0, index: 1)
        enc.setBuffer(V, offset: 0, index: 2)
        enc.setBuffer(blockTable, offset: 0, index: 3)
        enc.setBuffer(O, offset: 0, index: 4)
        var np = UInt32(numPages), kl = UInt32(kLen)
        var sc = 1.0 / Float(D).squareRoot()
        enc.setBytes(&np, length: 4, index: 5)
        enc.setBytes(&kl, length: 4, index: 6)
        enc.setBytes(&sc, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return times.min()!
}

print("=== single-slot decode (1 TG, underutilizes GPU by design) — D=128 legacy ===")
for kLen in [2048, 8192] {
    let t = runDecode(psos: psos128, kLen: kLen)
    let numPages = (kLen + psos128.PAGE - 1) / psos128.PAGE
    let kvBytes = Double(numPages) * Double(psos128.PAGE) * Double(psos128.D) * 2.0 * 2.0
    let gbps = kvBytes / t / 1e9
    let label = "K_len=\(kLen)".padding(toLength: 14, withPad: " ", startingAt: 0)
    print("  \(label)  \(String(format: "%8.1f μs", t * 1e6))   \(String(format: "%6.1f GB/s", gbps))")
}

func runBatched(psos: AttnPSOs, batch: Int, kLen: Int, iters: Int = 30, warmup: Int = 5) -> Double {
    let D = psos.D
    let PAGE = psos.PAGE
    let numPagesPerSlot = (kLen + PAGE - 1) / PAGE
    let maxPages = numPagesPerSlot

    // Q and O: [batch, D]
    let Q = makeFilledHalfBuf(batch * D, seed: 0x11111111)
    let O = device.makeBuffer(length: batch * D * 2, options: .storageModeShared)!

    // KV cache: enough physical pages for all slots, no sharing in this baseline.
    let totalPages = batch * numPagesPerSlot
    let K = makeFilledHalfBuf(totalPages * PAGE * D, seed: 0x22222222)
    let V = makeFilledHalfBuf(totalPages * PAGE * D, seed: 0x33333333)

    // Block table: slot s → pages [s*numPagesPerSlot .. (s+1)*numPagesPerSlot - 1]
    let blockTable = device.makeBuffer(length: batch * maxPages * 4, options: .storageModeShared)!
    let btp = blockTable.contents().bindMemory(to: UInt32.self, capacity: batch * maxPages)
    for s in 0..<batch {
        for p in 0..<numPagesPerSlot {
            btp[s * maxPages + p] = UInt32(s * numPagesPerSlot + p)
        }
    }

    let numPagesPerSlotBuf = device.makeBuffer(length: batch * 4, options: .storageModeShared)!
    let kLenPerSlotBuf     = device.makeBuffer(length: batch * 4, options: .storageModeShared)!
    let np = numPagesPerSlotBuf.contents().bindMemory(to: UInt32.self, capacity: batch)
    let kl = kLenPerSlotBuf.contents().bindMemory(to: UInt32.self, capacity: batch)
    for s in 0..<batch { np[s] = UInt32(numPagesPerSlot); kl[s] = UInt32(kLen) }

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psos.batch)
        enc.setBuffer(Q, offset: 0, index: 0)
        enc.setBuffer(K, offset: 0, index: 1)
        enc.setBuffer(V, offset: 0, index: 2)
        enc.setBuffer(blockTable, offset: 0, index: 3)
        enc.setBuffer(O, offset: 0, index: 4)
        enc.setBuffer(numPagesPerSlotBuf, offset: 0, index: 5)
        enc.setBuffer(kLenPerSlotBuf, offset: 0, index: 6)
        var sc = 1.0 / Float(D).squareRoot()
        var mp = UInt32(maxPages)
        enc.setBytes(&sc, length: 4, index: 7)
        enc.setBytes(&mp, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: batch, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return times.min()!
}

print("\n=== batched decode (D=128 legacy) ===")
for (batch, kLen) in [(8, 2048), (32, 2048), (128, 2048), (32, 8192), (128, 8192)] {
    let t = runBatched(psos: psos128, batch: batch, kLen: kLen)
    let numPages = (kLen + psos128.PAGE - 1) / psos128.PAGE
    let kvBytes = Double(batch) * Double(numPages) * Double(psos128.PAGE) * Double(psos128.D) * 2.0 * 2.0
    let gbps = kvBytes / t / 1e9
    let label = "batch=\(batch) K_len=\(kLen)".padding(toLength: 24, withPad: " ", startingAt: 0)
    print("  \(label)  \(String(format: "%8.1f μs", t * 1e6))   \(String(format: "%6.1f GB/s", gbps))")
}

// Shared-prefix variant: all slots point block_table[0..prefix_pages-1] at
// the same physical pages. Only the tail pages differ per slot.
func runBatchedSharedPrefix(psos: AttnPSOs, batch: Int, kLen: Int, prefixLen: Int,
                            iters: Int = 30, warmup: Int = 5) -> Double {
    let D = psos.D
    let PAGE = psos.PAGE
    let numPagesPerSlot = (kLen + PAGE - 1) / PAGE
    let prefixPages = (prefixLen + PAGE - 1) / PAGE
    let tailPagesPerSlot = numPagesPerSlot - prefixPages
    let maxPages = numPagesPerSlot

    let Q = makeFilledHalfBuf(batch * D, seed: 0x11111111)
    let O = device.makeBuffer(length: batch * D * 2, options: .storageModeShared)!

    // Physical pages: shared prefix pages are at [0..prefixPages-1].
    // Per-slot tail pages at [prefixPages + slot*tailPagesPerSlot ..].
    let totalPhysPages = prefixPages + batch * tailPagesPerSlot
    let K = makeFilledHalfBuf(totalPhysPages * PAGE * D, seed: 0x22222222)
    let V = makeFilledHalfBuf(totalPhysPages * PAGE * D, seed: 0x33333333)

    let blockTable = device.makeBuffer(length: batch * maxPages * 4, options: .storageModeShared)!
    let btp = blockTable.contents().bindMemory(to: UInt32.self, capacity: batch * maxPages)
    for s in 0..<batch {
        for p in 0..<prefixPages {
            btp[s * maxPages + p] = UInt32(p)   // all slots point to same prefix
        }
        for p in 0..<tailPagesPerSlot {
            btp[s * maxPages + prefixPages + p] = UInt32(prefixPages + s * tailPagesPerSlot + p)
        }
    }

    let numPagesPerSlotBuf = device.makeBuffer(length: batch * 4, options: .storageModeShared)!
    let kLenPerSlotBuf     = device.makeBuffer(length: batch * 4, options: .storageModeShared)!
    let np = numPagesPerSlotBuf.contents().bindMemory(to: UInt32.self, capacity: batch)
    let kl = kLenPerSlotBuf.contents().bindMemory(to: UInt32.self, capacity: batch)
    for s in 0..<batch { np[s] = UInt32(numPagesPerSlot); kl[s] = UInt32(kLen) }

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psos.batch)
        enc.setBuffer(Q, offset: 0, index: 0)
        enc.setBuffer(K, offset: 0, index: 1)
        enc.setBuffer(V, offset: 0, index: 2)
        enc.setBuffer(blockTable, offset: 0, index: 3)
        enc.setBuffer(O, offset: 0, index: 4)
        enc.setBuffer(numPagesPerSlotBuf, offset: 0, index: 5)
        enc.setBuffer(kLenPerSlotBuf, offset: 0, index: 6)
        var sc = 1.0 / Float(D).squareRoot()
        var mp = UInt32(maxPages)
        enc.setBytes(&sc, length: 4, index: 7)
        enc.setBytes(&mp, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: batch, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return times.min()!
}

func runSplit(psos: AttnPSOs, batch: Int, kLen: Int, numSplits: Int,
              iters: Int = 30, warmup: Int = 5) -> Double {
    let D = psos.D
    let PAGE = psos.PAGE
    let numPagesPerSlot = (kLen + PAGE - 1) / PAGE
    let maxPages = numPagesPerSlot

    let Q = makeFilledHalfBuf(batch * D, seed: 0x11111111)
    let O = device.makeBuffer(length: batch * D * 2, options: .storageModeShared)!
    let totalPages = batch * numPagesPerSlot
    let K = makeFilledHalfBuf(totalPages * PAGE * D, seed: 0x22222222)
    let V = makeFilledHalfBuf(totalPages * PAGE * D, seed: 0x33333333)

    let blockTable = device.makeBuffer(length: batch * maxPages * 4, options: .storageModeShared)!
    let btp = blockTable.contents().bindMemory(to: UInt32.self, capacity: batch * maxPages)
    for s in 0..<batch {
        for p in 0..<numPagesPerSlot { btp[s * maxPages + p] = UInt32(s * numPagesPerSlot + p) }
    }
    let numPagesPerSlotBuf = device.makeBuffer(length: batch * 4, options: .storageModeShared)!
    let kLenPerSlotBuf     = device.makeBuffer(length: batch * 4, options: .storageModeShared)!
    let np = numPagesPerSlotBuf.contents().bindMemory(to: UInt32.self, capacity: batch)
    let kl = kLenPerSlotBuf.contents().bindMemory(to: UInt32.self, capacity: batch)
    for s in 0..<batch { np[s] = UInt32(numPagesPerSlot); kl[s] = UInt32(kLen) }

    let mPart = device.makeBuffer(length: batch * numSplits * 4, options: .storageModeShared)!
    let lPart = device.makeBuffer(length: batch * numSplits * 4, options: .storageModeShared)!
    let oPart = device.makeBuffer(length: batch * numSplits * D * 4, options: .storageModeShared)!

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        // Phase 1: compute partials
        let encC = cb.makeComputeCommandEncoder()!
        encC.setComputePipelineState(psos.splitComp)
        encC.setBuffer(Q, offset: 0, index: 0)
        encC.setBuffer(K, offset: 0, index: 1)
        encC.setBuffer(V, offset: 0, index: 2)
        encC.setBuffer(blockTable, offset: 0, index: 3)
        encC.setBuffer(numPagesPerSlotBuf, offset: 0, index: 4)
        encC.setBuffer(kLenPerSlotBuf, offset: 0, index: 5)
        encC.setBuffer(mPart, offset: 0, index: 6)
        encC.setBuffer(lPart, offset: 0, index: 7)
        encC.setBuffer(oPart, offset: 0, index: 8)
        var sc = 1.0 / Float(D).squareRoot()
        var mp = UInt32(maxPages)
        var ns = UInt32(numSplits)
        encC.setBytes(&sc, length: 4, index: 9)
        encC.setBytes(&mp, length: 4, index: 10)
        encC.setBytes(&ns, length: 4, index: 11)
        encC.dispatchThreadgroups(MTLSize(width: batch, height: numSplits, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encC.endEncoding()
        // Phase 2: reduce
        let encR = cb.makeComputeCommandEncoder()!
        encR.setComputePipelineState(psos.splitRed)
        encR.setBuffer(mPart, offset: 0, index: 0)
        encR.setBuffer(lPart, offset: 0, index: 1)
        encR.setBuffer(oPart, offset: 0, index: 2)
        encR.setBuffer(O, offset: 0, index: 3)
        encR.setBytes(&ns, length: 4, index: 4)
        encR.dispatchThreadgroups(MTLSize(width: batch, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encR.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return times.min()!
}

func runPrefill(psos: AttnPSOs, S: Int, iters: Int = 20, warmup: Int = 5) -> Double {
    let D = psos.D
    let Q = makeFilledHalfBuf(S * D, seed: 0x11111111)
    let K = makeFilledHalfBuf(S * D, seed: 0x22222222)
    let V = makeFilledHalfBuf(S * D, seed: 0x33333333)
    let O = device.makeBuffer(length: S * D * 2, options: .storageModeShared)!

    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psos.prefill)
        enc.setBuffer(Q, offset: 0, index: 0)
        enc.setBuffer(K, offset: 0, index: 1)
        enc.setBuffer(V, offset: 0, index: 2)
        enc.setBuffer(O, offset: 0, index: 3)
        var Su = UInt32(S)
        var sc = 1.0 / Float(D).squareRoot()
        enc.setBytes(&Su, length: 4, index: 4)
        enc.setBytes(&Su, length: 4, index: 5)
        enc.setBytes(&sc, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: S, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU: \(err)") }
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return times.min()!
}

print("\n=== prefill attention (D=128 legacy) ===")
for S in [256, 512, 1024, 2048, 4096] {
    let t = runPrefill(psos: psos128, S: S)
    let flops = Double(S) * Double(S + 1) / 2.0 * Double(psos128.D) * 4.0
    let tflops = flops / t / 1e12
    let label = "S=\(S)".padding(toLength: 10, withPad: " ", startingAt: 0)
    print("  \(label)  \(String(format: "%9.1f μs", t * 1e6))   \(String(format: "%.3f TFLOPS", tflops))")
}

print("\n=== split-KV decode (D=128 legacy) ===")
for (kLen, splits) in [(2048, 1), (2048, 4), (2048, 8), (2048, 16), (2048, 32),
                       (8192, 1), (8192, 8), (8192, 32),
                       (32768, 1), (32768, 32), (32768, 128)] {
    let t = runSplit(psos: psos128, batch: 1, kLen: kLen, numSplits: splits)
    let numPages = (kLen + psos128.PAGE - 1) / psos128.PAGE
    let kvBytes = Double(numPages) * Double(psos128.PAGE) * Double(psos128.D) * 2.0 * 2.0
    let gbps = kvBytes / t / 1e9
    let label = "K=\(kLen) splits=\(splits)".padding(toLength: 22, withPad: " ", startingAt: 0)
    print("  \(label)  \(String(format: "%8.1f μs", t * 1e6))   \(String(format: "%6.1f GB/s", gbps))")
}

print("\n=== shared-prefix decode (D=128 legacy) ===")
for batch in [32, 128] {
    for (kLen, prefLen) in [(2048, 0), (2048, 1024), (2048, 1920), (8192, 0), (8192, 4096), (8192, 7680)] {
        let t = runBatchedSharedPrefix(psos: psos128, batch: batch, kLen: kLen, prefixLen: prefLen)
        let numPages = (kLen + psos128.PAGE - 1) / psos128.PAGE
        let kvLogicalBytes = Double(batch) * Double(numPages) * Double(psos128.PAGE) * Double(psos128.D) * 2.0 * 2.0
        let effGbps = kvLogicalBytes / t / 1e9
        let prefPages = (prefLen + psos128.PAGE - 1) / psos128.PAGE
        let physBytes = Double(prefPages + batch * (numPages - prefPages)) * Double(psos128.PAGE) * Double(psos128.D) * 2.0 * 2.0
        let physGbps = physBytes / t / 1e9
        let label = "batch=\(batch) K=\(kLen) pref=\(prefLen)".padding(toLength: 30, withPad: " ", startingAt: 0)
        print("  \(label)  \(String(format: "%8.1f μs", t * 1e6))   logical \(String(format: "%5.0f", effGbps)) / physical \(String(format: "%5.0f", physGbps)) GB/s")
    }
    print("")
}

// ===========================================================================
// Gemma-4-A4B attention shapes.
// - Sliding layers (25/30): head_dim=256, kv_heads=8, sliding_window=1024.
//   Use D=256 PAGE=16. Per-KV-head perspective; Q replicates across KV-heads.
//   Context is capped at 1024 tokens → K_len sweep bounded.
// - Full layers (5/30):     head_dim=512, kv_heads=2, up to 256K context.
//   Use D=512 PAGE=8 (tg-mem budget forces smaller page).
// ===========================================================================

print("\n=== Gemma-4 sliding attention (D=256 PAGE=16, K_len ≤ 1024) ===")
for (batch, kLen) in [(128, 256), (128, 1024), (256, 1024)] {
    let t = runBatched(psos: psos256, batch: batch, kLen: kLen)
    let numPages = (kLen + psos256.PAGE - 1) / psos256.PAGE
    let kvBytes = Double(batch) * Double(numPages) * Double(psos256.PAGE) * Double(psos256.D) * 2.0 * 2.0
    let gbps = kvBytes / t / 1e9
    let label = "batch=\(batch) K_len=\(kLen)".padding(toLength: 24, withPad: " ", startingAt: 0)
    print("  \(label)  \(String(format: "%8.1f μs", t * 1e6))   \(String(format: "%6.1f GB/s", gbps))")
}

print("\n=== Gemma-4 sliding shared-prefix (D=256 PAGE=16) ===")
for batch in [4, 32, 128] {
    for (kLen, prefLen) in [(1024, 0), (1024, 512), (1024, 960)] {
        let t = runBatchedSharedPrefix(psos: psos256, batch: batch, kLen: kLen, prefixLen: prefLen)
        let numPages = (kLen + psos256.PAGE - 1) / psos256.PAGE
        let kvLogicalBytes = Double(batch) * Double(numPages) * Double(psos256.PAGE) * Double(psos256.D) * 2.0 * 2.0
        let effGbps = kvLogicalBytes / t / 1e9
        let prefPages = (prefLen + psos256.PAGE - 1) / psos256.PAGE
        let physBytes = Double(prefPages + batch * (numPages - prefPages)) * Double(psos256.PAGE) * Double(psos256.D) * 2.0 * 2.0
        let physGbps = physBytes / t / 1e9
        let label = "batch=\(batch) K=\(kLen) pref=\(prefLen)".padding(toLength: 30, withPad: " ", startingAt: 0)
        print("  \(label)  \(String(format: "%8.1f μs", t * 1e6))   logical \(String(format: "%5.0f", effGbps)) / physical \(String(format: "%5.0f", physGbps)) GB/s")
    }
    print("")
}

print("=== Gemma-4 full attention (D=512 PAGE=8, long context) ===")
for (batch, kLen) in [(32, 4096), (128, 4096), (32, 32768), (128, 32768)] {
    let t = runBatched(psos: psos512, batch: batch, kLen: kLen)
    let numPages = (kLen + psos512.PAGE - 1) / psos512.PAGE
    let kvBytes = Double(batch) * Double(numPages) * Double(psos512.PAGE) * Double(psos512.D) * 2.0 * 2.0
    let gbps = kvBytes / t / 1e9
    let label = "batch=\(batch) K_len=\(kLen)".padding(toLength: 24, withPad: " ", startingAt: 0)
    print("  \(label)  \(String(format: "%8.1f μs", t * 1e6))   \(String(format: "%6.1f GB/s", gbps))")
}

print("\n=== Gemma-4 full attention shared-prefix (D=512 PAGE=8) ===")
// Long-context prefix sharing: 2 distinct prompts, 75-80% shared prefix, 4 slots.
for (batch, kLen, prefLen) in [(4, 8192, 6144), (4, 32768, 24576), (8, 32768, 26000), (32, 8192, 6144)] {
    let t = runBatchedSharedPrefix(psos: psos512, batch: batch, kLen: kLen, prefixLen: prefLen)
    let numPages = (kLen + psos512.PAGE - 1) / psos512.PAGE
    let kvLogicalBytes = Double(batch) * Double(numPages) * Double(psos512.PAGE) * Double(psos512.D) * 2.0 * 2.0
    let effGbps = kvLogicalBytes / t / 1e9
    let prefPages = (prefLen + psos512.PAGE - 1) / psos512.PAGE
    let physBytes = Double(prefPages + batch * (numPages - prefPages)) * Double(psos512.PAGE) * Double(psos512.D) * 2.0 * 2.0
    let physGbps = physBytes / t / 1e9
    let label = "batch=\(batch) K=\(kLen) pref=\(prefLen)".padding(toLength: 30, withPad: " ", startingAt: 0)
    print("  \(label)  \(String(format: "%8.1f μs", t * 1e6))   logical \(String(format: "%5.0f", effGbps)) / physical \(String(format: "%5.0f", physGbps)) GB/s")
}

print("\n=== Gemma-4 full-attention split-KV (D=512 PAGE=8, single-slot long-context) ===")
for (kLen, splits) in [(32768, 1), (32768, 32), (32768, 128), (131072, 128)] {
    let t = runSplit(psos: psos512, batch: 1, kLen: kLen, numSplits: splits)
    let numPages = (kLen + psos512.PAGE - 1) / psos512.PAGE
    let kvBytes = Double(numPages) * Double(psos512.PAGE) * Double(psos512.D) * 2.0 * 2.0
    let gbps = kvBytes / t / 1e9
    let label = "K=\(kLen) splits=\(splits)".padding(toLength: 22, withPad: " ", startingAt: 0)
    print("  \(label)  \(String(format: "%8.1f μs", t * 1e6))   \(String(format: "%6.1f GB/s", gbps))")
}
