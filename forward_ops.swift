import Metal
import Foundation

// ===========================================================================
// Phase 1 forward-pass primitives (Gemma-4-A4B) — the small kernels every
// transformer needs beyond the projection/attention heavy hitters.
//
// Kernels here:
//   rms_norm             — hidden → normalized hidden × gamma
//   rope_half            — rotate Q/K in-place (rotate_half convention)
//   kv_cache_write       — write K,V to paged cache slot
//   softmax_topk_128     — router softmax + top-k=8 selection
//   moe_combine          — scatter-add expert outputs weighted by gate
//   embed_lookup         — token_id → embedding row
//   logit_softcap        — tanh(logits/cap)*cap
//
// All at Gemma-4 shapes (hidden=2816). No fusion yet — the point is to
// measure per-launch overhead so we can quantify what fusion earns.
// ===========================================================================

let mslSource = """
#include <metal_stdlib>
using namespace metal;

// ----- RMSNorm: one TG per batch row, simdgroup reduction -----
kernel void rms_norm(
    device const half* x         [[buffer(0)]],   // [B, D]
    device const half* gamma     [[buffer(1)]],   // [D]
    device half* y               [[buffer(2)]],   // [B, D]
    constant uint& D             [[buffer(3)]],
    constant float& eps          [[buffer(4)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint2 lid2                   [[thread_position_in_threadgroup]]
) {
    const uint b = tg_pos.x;
    const uint lid = lid2.x;

    // Phase 1: sum of squares across D, simdgroup-reduced.
    float sum_sq = 0.0f;
    for (uint i = lid; i < D; i += 32) {
        float v = float(x[b * D + i]);
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    // Phase 2: compute rsqrt, apply to every element * gamma.
    const float scale = rsqrt(sum_sq / float(D) + eps);
    for (uint i = lid; i < D; i += 32) {
        float v = float(x[b * D + i]);
        y[b * D + i] = half(v * scale * float(gamma[i]));
    }
}

// ----- RoPE with rotate_half: first D/2 and last D/2 form the pairs -----
// For Gemma-4 sliding (default) — theta=10000, rotary_dims = D.
// For Gemma-4 full — theta=1M, rotary_dims = D/4 (partial_rotary_factor=0.25).
// Input: x [B, H, D]. Position per batch row in `positions[B]`.
kernel void rope_half(
    device half* x               [[buffer(0)]],   // [B, H, D] in-place
    device const uint* positions [[buffer(1)]],   // [B]
    constant uint& H             [[buffer(2)]],
    constant uint& D             [[buffer(3)]],
    constant uint& rotary_dims   [[buffer(4)]],   // ≤ D
    constant float& theta        [[buffer(5)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint3 lid3                   [[thread_position_in_threadgroup]]
) {
    const uint b = tg_pos.x;
    const uint h = tg_pos.y;
    const uint lid = lid3.x;
    const uint pos = positions[b];
    const uint half_dim = rotary_dims / 2;

    device half* x_bh = x + (b * H + h) * D;

    // Each lane handles a (i, i + half_dim) pair within rotary_dims.
    // Dims beyond rotary_dims pass through unchanged (partial rotary case).
    for (uint i = lid; i < half_dim; i += 32) {
        float freq = 1.0f / pow(theta, 2.0f * float(i) / float(rotary_dims));
        float angle = float(pos) * freq;
        float c = cos(angle);
        float s = sin(angle);
        float x0 = float(x_bh[i]);
        float x1 = float(x_bh[i + half_dim]);
        x_bh[i]              = half(x0 * c - x1 * s);
        x_bh[i + half_dim]   = half(x0 * s + x1 * c);
    }
}

// ----- KV cache write: one TG per (b, h), each TG writes D values -----
// K and V written side-by-side (separate cache tensors, same indexing).
kernel void kv_cache_write(
    device const half* K           [[buffer(0)]],   // [B, H, D]
    device const half* V           [[buffer(1)]],
    device half* K_cache           [[buffer(2)]],   // [total_pages, PAGE, H, D]
    device half* V_cache           [[buffer(3)]],
    device const uint* block_table [[buffer(4)]],   // [B, max_pages]
    device const uint* positions   [[buffer(5)]],   // [B]
    constant uint& H               [[buffer(6)]],
    constant uint& D               [[buffer(7)]],
    constant uint& PAGE            [[buffer(8)]],
    constant uint& max_pages       [[buffer(9)]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]],
    uint3 lid3                     [[thread_position_in_threadgroup]]
) {
    const uint b = tg_pos.x;
    const uint h = tg_pos.y;
    const uint lid = lid3.x;
    const uint pos = positions[b];
    const uint logical_page = pos / PAGE;
    const uint offset = pos % PAGE;
    const uint phys = block_table[b * max_pages + logical_page];

    device const half* Ksrc = K + (b * H + h) * D;
    device const half* Vsrc = V + (b * H + h) * D;
    device half* Kdst = K_cache + ((phys * PAGE + offset) * H + h) * D;
    device half* Vdst = V_cache + ((phys * PAGE + offset) * H + h) * D;

    for (uint i = lid; i < D; i += 32) {
        Kdst[i] = Ksrc[i];
        Vdst[i] = Vsrc[i];
    }
}

// ----- Softmax + top-k=8 over 128 logits -----
// One TG per batch row. 32 threads each hold 4 logits. Find max, exp,
// normalize, then linear scan to pull top-k.
kernel void softmax_topk_128(
    device const half* logits    [[buffer(0)]],   // [B, 128]
    device uint* expert_ids      [[buffer(1)]],   // [B, 8]
    device float* gate_weights   [[buffer(2)]],   // [B, 8]
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint2 lid2                   [[thread_position_in_threadgroup]]
) {
    constexpr uint E = 128;
    constexpr uint K = 8;
    const uint b = tg_pos.x;
    const uint lid = lid2.x;
    device const half* lg = logits + b * E;

    // Each lane holds 4 logits: lane i owns positions {i, i+32, i+64, i+96}.
    float vals[4];
    for (uint j = 0; j < 4; ++j) vals[j] = float(lg[lid + j * 32]);

    // Max across TG
    float my_max = max(max(vals[0], vals[1]), max(vals[2], vals[3]));
    float global_max = simd_max(my_max);

    // Exp + local sum
    float my_sum = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        vals[j] = exp(vals[j] - global_max);
        my_sum += vals[j];
    }
    float global_sum = simd_sum(my_sum);
    float inv_sum = 1.0f / global_sum;
    for (uint j = 0; j < 4; ++j) vals[j] *= inv_sum;

    // Write normalized probs to tg-mem so one lane can scan for top-k.
    threadgroup float probs[E];
    for (uint j = 0; j < 4; ++j) probs[lid + j * 32] = vals[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Single-lane top-k via linear selection (K*E = 1024 ops, fine at K=8).
    if (lid == 0) {
        float selected_vals[K];
        uint selected_ids[K];
        for (uint k = 0; k < K; ++k) {
            float best_val = -INFINITY;
            uint best_id = 0;
            for (uint e = 0; e < E; ++e) {
                if (probs[e] > best_val) {
                    best_val = probs[e];
                    best_id = e;
                }
            }
            selected_vals[k] = best_val;
            selected_ids[k] = best_id;
            probs[best_id] = -INFINITY;   // mark used
        }
        // Write out
        for (uint k = 0; k < K; ++k) {
            expert_ids[b * K + k] = selected_ids[k];
            gate_weights[b * K + k] = selected_vals[k];
        }
    }
}

// ----- MoE combine: scatter-add weighted expert outputs into hidden -----
// For each (b, d_block), read top_k slots' outputs, multiply by gate_weight,
// sum, add to hidden (residual). No atomics needed — one TG owns each
// (b, d_block) output. Grid: (D/32, B).
kernel void moe_combine(
    device const half* expert_out  [[buffer(0)]],   // [total_slots, D]
    device const uint* batch_slots [[buffer(1)]],   // [B, top_k] — slot index for each (b, k)
    device const float* gate_weights [[buffer(2)]], // [B, top_k]
    device half* hidden            [[buffer(3)]],   // [B, D] — accumulated in-place
    constant uint& top_k           [[buffer(4)]],
    constant uint& D               [[buffer(5)]],
    uint2 tg_pos                   [[threadgroup_position_in_grid]],
    uint2 lid2                     [[thread_position_in_threadgroup]]
) {
    const uint d_block = tg_pos.x;
    const uint b = tg_pos.y;
    const uint lid = lid2.x;
    const uint d = d_block * 32 + lid;
    if (d >= D) return;

    float acc = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
        uint slot = batch_slots[b * top_k + k];
        float w = gate_weights[b * top_k + k];
        acc += w * float(expert_out[slot * D + d]);
    }
    hidden[b * D + d] = half(float(hidden[b * D + d]) + acc);
}

// ----- Embed lookup: token_id → embedding row, B independent copies -----
kernel void embed_lookup(
    device const uint* token_ids   [[buffer(0)]],   // [B]
    device const half* embed_table [[buffer(1)]],   // [vocab, D]
    device half* hidden            [[buffer(2)]],   // [B, D]
    constant uint& D               [[buffer(3)]],
    uint2 tg_pos                   [[threadgroup_position_in_grid]],
    uint2 lid2                     [[thread_position_in_threadgroup]]
) {
    const uint b = tg_pos.x;
    const uint lid = lid2.x;
    uint tok = token_ids[b];
    device const half* src = embed_table + tok * D;
    device half* dst = hidden + b * D;
    for (uint i = lid; i < D; i += 32) dst[i] = src[i];
}

// ----- Logit softcap: tanh(x/cap) * cap, in-place over [B, vocab] -----
kernel void logit_softcap(
    device half* logits            [[buffer(0)]],   // [B, vocab]
    constant uint& vocab           [[buffer(1)]],
    constant float& cap            [[buffer(2)]],
    uint2 tg_pos                   [[threadgroup_position_in_grid]],
    uint2 lid2                     [[thread_position_in_threadgroup]]
) {
    const uint b = tg_pos.x;
    const uint lid = lid2.x;
    device half* lg = logits + b * vocab;
    const float inv_cap = 1.0f / cap;
    for (uint i = lid; i < vocab; i += 32) {
        float v = float(lg[i]);
        lg[i] = half(tanh(v * inv_cap) * cap);
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

guard let rmsFn      = library.makeFunction(name: "rms_norm"),
      let ropeFn     = library.makeFunction(name: "rope_half"),
      let kvwFn      = library.makeFunction(name: "kv_cache_write"),
      let topkFn     = library.makeFunction(name: "softmax_topk_128"),
      let combineFn  = library.makeFunction(name: "moe_combine"),
      let embedFn    = library.makeFunction(name: "embed_lookup"),
      let softcapFn  = library.makeFunction(name: "logit_softcap") else { fail("no kernel") }
let rmsPSO     = try! device.makeComputePipelineState(function: rmsFn)
let ropePSO    = try! device.makeComputePipelineState(function: ropeFn)
let kvwPSO     = try! device.makeComputePipelineState(function: kvwFn)
let topkPSO    = try! device.makeComputePipelineState(function: topkFn)
let combinePSO = try! device.makeComputePipelineState(function: combineFn)
let embedPSO   = try! device.makeComputePipelineState(function: embedFn)
let softcapPSO = try! device.makeComputePipelineState(function: softcapFn)

print("device: \(device.name)")
print("")

func randomHalfBuf(_ n: Int, seed: UInt32, scale: Float = 0.02) -> MTLBuffer {
    let b = device.makeBuffer(length: n * 2, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: Float16.self, capacity: n)
    var s = seed
    for i in 0..<n {
        s = s &* 1664525 &+ 1013904223
        let f = Float(Int32(bitPattern: s) % 1000) / 500.0 - 1.0
        p[i] = Float16(f * scale)
    }
    return b
}
func randomUIntBuf(_ n: Int, mod: Int, seed: UInt32) -> MTLBuffer {
    let b = device.makeBuffer(length: n * 4, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: UInt32.self, capacity: n)
    var s = seed
    for i in 0..<n {
        s = s &* 1664525 &+ 1013904223
        p[i] = UInt32(Int(s % UInt32(mod)))
    }
    return b
}

// --- Individual per-op bench (min of several iters, single-kernel CBs) ---
func timeOne(_ block: (MTLCommandBuffer) -> Void, iters: Int = 20, warmup: Int = 5) -> Double {
    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        block(cb)
        cb.commit()
        cb.waitUntilCompleted()
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return times.min()!
}

let B = 4
let D = 2816
let H_slide = 16
let headD_slide = 256
let H_full = 16
let headD_full = 512
let numExperts = 128
let topK = 8

// RMSNorm at Gemma-4 shape.
let x = randomHalfBuf(B * D, seed: 0x11)
let gamma = randomHalfBuf(D, seed: 0x22)
let y = device.makeBuffer(length: B * D * 2, options: .storageModeShared)!
let tRms = timeOne { cb in
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(rmsPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(gamma, offset: 0, index: 1)
    enc.setBuffer(y, offset: 0, index: 2)
    var Du = UInt32(D); var eps: Float = 1e-6
    enc.setBytes(&Du, length: 4, index: 3)
    enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}
print(String(format: "  RMSNorm [B=%d, D=%d]         %6.1f μs", B, D, tRms * 1e6))

// RoPE on Q (sliding).
let q_slide = randomHalfBuf(B * H_slide * headD_slide, seed: 0x33)
let positions = randomUIntBuf(B, mod: 4096, seed: 0x44)
let tRopeSlide = timeOne { cb in
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(ropePSO)
    enc.setBuffer(q_slide, offset: 0, index: 0)
    enc.setBuffer(positions, offset: 0, index: 1)
    var Hv = UInt32(H_slide), Dv = UInt32(headD_slide), Rv = UInt32(headD_slide)
    var theta: Float = 10000.0
    enc.setBytes(&Hv, length: 4, index: 2)
    enc.setBytes(&Dv, length: 4, index: 3)
    enc.setBytes(&Rv, length: 4, index: 4)
    enc.setBytes(&theta, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: B, height: H_slide, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}
print(String(format: "  RoPE sliding [B=%d, H=%d, D=%d full rotary]  %6.1f μs", B, H_slide, headD_slide, tRopeSlide * 1e6))

// RoPE on Q (full-attn, partial rotary 0.25).
let q_full = randomHalfBuf(B * H_full * headD_full, seed: 0x55)
let tRopeFull = timeOne { cb in
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(ropePSO)
    enc.setBuffer(q_full, offset: 0, index: 0)
    enc.setBuffer(positions, offset: 0, index: 1)
    var Hv = UInt32(H_full), Dv = UInt32(headD_full), Rv = UInt32(headD_full / 4)
    var theta: Float = 1_000_000.0
    enc.setBytes(&Hv, length: 4, index: 2)
    enc.setBytes(&Dv, length: 4, index: 3)
    enc.setBytes(&Rv, length: 4, index: 4)
    enc.setBytes(&theta, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: B, height: H_full, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}
print(String(format: "  RoPE full    [B=%d, H=%d, D=%d 0.25 partial] %6.1f μs", B, H_full, headD_full, tRopeFull * 1e6))

// KV cache write (sliding shape).
let kv_K = randomHalfBuf(B * 8 * headD_slide, seed: 0x66)   // 8 kv heads sliding
let kv_V = randomHalfBuf(B * 8 * headD_slide, seed: 0x77)
let K_cache = device.makeBuffer(length: 1024 * 16 * 8 * headD_slide * 2, options: .storageModeShared)!
let V_cache = device.makeBuffer(length: 1024 * 16 * 8 * headD_slide * 2, options: .storageModeShared)!
let block_table = randomUIntBuf(B * 64, mod: 1024, seed: 0x88)
let tKvw = timeOne { cb in
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(kvwPSO)
    enc.setBuffer(kv_K, offset: 0, index: 0)
    enc.setBuffer(kv_V, offset: 0, index: 1)
    enc.setBuffer(K_cache, offset: 0, index: 2)
    enc.setBuffer(V_cache, offset: 0, index: 3)
    enc.setBuffer(block_table, offset: 0, index: 4)
    enc.setBuffer(positions, offset: 0, index: 5)
    var Hk: UInt32 = 8, Dh = UInt32(headD_slide), PAGE: UInt32 = 16, maxP: UInt32 = 64
    enc.setBytes(&Hk, length: 4, index: 6)
    enc.setBytes(&Dh, length: 4, index: 7)
    enc.setBytes(&PAGE, length: 4, index: 8)
    enc.setBytes(&maxP, length: 4, index: 9)
    enc.dispatchThreadgroups(MTLSize(width: B, height: 8, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}
print(String(format: "  KV write     [B=%d, H=8, D=%d]             %6.1f μs", B, headD_slide, tKvw * 1e6))

// Router softmax + top-k.
let rlogits = randomHalfBuf(B * numExperts, seed: 0x99)
let exp_ids = device.makeBuffer(length: B * topK * 4, options: .storageModeShared)!
let gate_w = device.makeBuffer(length: B * topK * 4, options: .storageModeShared)!
let tTopk = timeOne { cb in
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(topkPSO)
    enc.setBuffer(rlogits, offset: 0, index: 0)
    enc.setBuffer(exp_ids, offset: 0, index: 1)
    enc.setBuffer(gate_w, offset: 0, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}
print(String(format: "  Softmax+topK [B=%d, E=%d, K=%d]              %6.1f μs", B, numExperts, topK, tTopk * 1e6))

// MoE combine.
let totalSlots = B * topK
let expert_out = randomHalfBuf(totalSlots * D, seed: 0xaa)
let batch_slots = randomUIntBuf(B * topK, mod: totalSlots, seed: 0xbb)
let gate_w_f = device.makeBuffer(length: B * topK * 4, options: .storageModeShared)!
let hidden = device.makeBuffer(length: B * D * 2, options: .storageModeShared)!
let tComb = timeOne { cb in
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(combinePSO)
    enc.setBuffer(expert_out, offset: 0, index: 0)
    enc.setBuffer(batch_slots, offset: 0, index: 1)
    enc.setBuffer(gate_w_f, offset: 0, index: 2)
    enc.setBuffer(hidden, offset: 0, index: 3)
    var tk = UInt32(topK), Dv = UInt32(D)
    enc.setBytes(&tk, length: 4, index: 4)
    enc.setBytes(&Dv, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: D / 32, height: B, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}
print(String(format: "  MoE combine  [B=%d, D=%d, topK=%d]          %6.1f μs", B, D, topK, tComb * 1e6))

// Embed lookup.
let tokens = randomUIntBuf(B, mod: 262144, seed: 0xcc)
let embed_table = randomHalfBuf(262144 * D, seed: 0xdd)
let hidden2 = device.makeBuffer(length: B * D * 2, options: .storageModeShared)!
let tEmbed = timeOne { cb in
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(embedPSO)
    enc.setBuffer(tokens, offset: 0, index: 0)
    enc.setBuffer(embed_table, offset: 0, index: 1)
    enc.setBuffer(hidden2, offset: 0, index: 2)
    var Dv = UInt32(D)
    enc.setBytes(&Dv, length: 4, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}
print(String(format: "  Embed lookup [B=%d, D=%d]                    %6.1f μs", B, D, tEmbed * 1e6))

// Softcap (operates on full vocab logits).
let vocab_logits = randomHalfBuf(B * 262144, seed: 0xee)
let tSoftcap = timeOne { cb in
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(softcapPSO)
    enc.setBuffer(vocab_logits, offset: 0, index: 0)
    var vv = UInt32(262144); var cap: Float = 30.0
    enc.setBytes(&vv, length: 4, index: 1)
    enc.setBytes(&cap, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}
print(String(format: "  Softcap      [B=%d, vocab=262144]            %6.1f μs", B, tSoftcap * 1e6))

// --- Fusion overhead test: N kernels in one command buffer vs N separate ---
// Compare run-N-in-one-CB (amortized launch) to N × individual-CB timings.
// Per-op time stays the same; CB-start/finish fixed cost shows up as gap.
func timeNinOneCB(_ n: Int, block: (MTLCommandBuffer) -> Void,
                  iters: Int = 10, warmup: Int = 3) -> Double {
    var times: [Double] = []
    for i in 0..<(iters + warmup) {
        let cb = queue.makeCommandBuffer()!
        for _ in 0..<n { block(cb) }
        cb.commit()
        cb.waitUntilCompleted()
        if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    return times.min()!
}

print("")
print("=== Fusion overhead: N-kernel CB vs N × single-kernel CBs ===")
print("   (RMSNorm repeated; diff = per-CB fixed overhead)")
for n in [1, 2, 4, 8, 16, 32, 60] {
    let tBatch = timeNinOneCB(n) { cb in
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(rmsPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var Du = UInt32(D); var eps: Float = 1e-6
        enc.setBytes(&Du, length: 4, index: 3)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }
    let perCall = tBatch / Double(n)
    let overheadVsSolo = (tRms - perCall) * 1e6  // positive = solo has more overhead per call
    print(String(format: "  n=%3d in 1 CB: %7.1f μs total, %6.1f μs/op  (solo=%6.1f, savings=%+6.1f μs/op)",
                 n, tBatch * 1e6, perCall * 1e6, tRms * 1e6, overheadVsSolo))
}

// Layer-synthetic: chain all Phase-1 primitives in one CB, one batch of them
// ≈ a simplified per-layer overhead test (sans actual projections/MoE).
print("")
print("=== One synthetic layer (all Phase-1 primitives chained) ===")
let tLayer = timeOne { cb in
    // RMSNorm
    do {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(rmsPSO)
        enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(gamma, offset: 0, index: 1); enc.setBuffer(y, offset: 0, index: 2)
        var Du = UInt32(D); var eps: Float = 1e-6
        enc.setBytes(&Du, length: 4, index: 3); enc.setBytes(&eps, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }
    // RoPE sliding
    do {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(ropePSO)
        enc.setBuffer(q_slide, offset: 0, index: 0); enc.setBuffer(positions, offset: 0, index: 1)
        var Hv = UInt32(H_slide), Dv = UInt32(headD_slide), Rv = UInt32(headD_slide); var theta: Float = 10000
        enc.setBytes(&Hv, length: 4, index: 2); enc.setBytes(&Dv, length: 4, index: 3)
        enc.setBytes(&Rv, length: 4, index: 4); enc.setBytes(&theta, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: B, height: H_slide, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }
    // KV write
    do {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(kvwPSO)
        enc.setBuffer(kv_K, offset: 0, index: 0); enc.setBuffer(kv_V, offset: 0, index: 1)
        enc.setBuffer(K_cache, offset: 0, index: 2); enc.setBuffer(V_cache, offset: 0, index: 3)
        enc.setBuffer(block_table, offset: 0, index: 4); enc.setBuffer(positions, offset: 0, index: 5)
        var Hk: UInt32 = 8, Dh = UInt32(headD_slide), PAGE: UInt32 = 16, maxP: UInt32 = 64
        enc.setBytes(&Hk, length: 4, index: 6); enc.setBytes(&Dh, length: 4, index: 7)
        enc.setBytes(&PAGE, length: 4, index: 8); enc.setBytes(&maxP, length: 4, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: B, height: 8, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }
    // RMSNorm second (pre-FFN)
    do {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(rmsPSO)
        enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(gamma, offset: 0, index: 1); enc.setBuffer(y, offset: 0, index: 2)
        var Du = UInt32(D); var eps: Float = 1e-6
        enc.setBytes(&Du, length: 4, index: 3); enc.setBytes(&eps, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }
    // Softmax + topK
    do {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(topkPSO)
        enc.setBuffer(rlogits, offset: 0, index: 0); enc.setBuffer(exp_ids, offset: 0, index: 1); enc.setBuffer(gate_w, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }
    // MoE combine
    do {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(combinePSO)
        enc.setBuffer(expert_out, offset: 0, index: 0); enc.setBuffer(batch_slots, offset: 0, index: 1)
        enc.setBuffer(gate_w_f, offset: 0, index: 2); enc.setBuffer(hidden, offset: 0, index: 3)
        var tk = UInt32(topK), Dv = UInt32(D)
        enc.setBytes(&tk, length: 4, index: 4); enc.setBytes(&Dv, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: D / 32, height: B, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }
}
let sumOfIndividuals = tRms + tRopeSlide + tKvw + tRms + tTopk + tComb
print(String(format: "  chained in 1 CB: %6.1f μs", tLayer * 1e6))
print(String(format: "  sum of solo runs: %6.1f μs", sumOfIndividuals * 1e6))
print(String(format: "  CB-amortization savings: %6.1f μs (%.1f%%)",
             (sumOfIndividuals - tLayer) * 1e6, (sumOfIndividuals - tLayer) / sumOfIndividuals * 100))

// ==========================================================================
// "Fake full forward" test — simulate 30-layer Gemma-4 forward's dispatch
// density (≈600 kernel dispatches per step) two ways:
//   A) One CB per dispatch (worst case, what default bench harnesses do)
//   B) One CB with N back-to-back dispatches (graph-capture-equivalent)
// The delta is the architectural upside from CB-reuse.
// ==========================================================================

print("")
print("=== Fake forward pass — 600 dispatches (≈30 layers × 20 ops/layer) ===")
print("   Using RMSNorm as a stand-in 'unit op' (real forward would mix kernels,")
print("   but CB overhead is kernel-independent).")
print("")

let FORWARD_DISPATCHES = 600

// Pattern A: one CB per dispatch — this is what our per-op benches did.
func patternA(n: Int) -> Double {
    // Full warm-up: submit a couple of full-forwards first.
    for _ in 0..<2 {
        for _ in 0..<n {
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(rmsPSO)
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(gamma, offset: 0, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
            var Du = UInt32(D); var eps: Float = 1e-6
            enc.setBytes(&Du, length: 4, index: 3)
            enc.setBytes(&eps, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }
    }
    // Timed run: measure wall clock from first commit to last completion
    let t0 = Date()
    for _ in 0..<n {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(rmsPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var Du = UInt32(D); var eps: Float = 1e-6
        enc.setBytes(&Du, length: 4, index: 3)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    return Date().timeIntervalSince(t0)
}

// Pattern B: one CB containing N dispatches — what a captured graph gives us.
func patternB(n: Int) -> Double {
    // Warmup
    for _ in 0..<2 {
        let cb = queue.makeCommandBuffer()!
        for _ in 0..<n {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(rmsPSO)
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(gamma, offset: 0, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
            var Du = UInt32(D); var eps: Float = 1e-6
            enc.setBytes(&Du, length: 4, index: 3)
            enc.setBytes(&eps, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
    }
    let t0 = Date()
    let cb = queue.makeCommandBuffer()!
    for _ in 0..<n {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(rmsPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var Du = UInt32(D); var eps: Float = 1e-6
        enc.setBytes(&Du, length: 4, index: 3)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }
    cb.commit()
    cb.waitUntilCompleted()
    return Date().timeIntervalSince(t0)
}

for n in [30, 100, 300, 600] {
    let tA = patternA(n: n)
    let tB = patternB(n: n)
    print(String(format: "  n=%3d dispatches: A (CB-per-op) %6.2f ms    B (one CB) %6.2f ms    %.1f× speedup",
                 n, tA * 1000, tB * 1000, tA / tB))
}
