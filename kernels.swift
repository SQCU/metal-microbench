// Metal Shading Language kernel source strings for the forward_graph
// build. Extracted from the former monofile forward_graph.swift in the
// 2026-04-18 refactor. This file defines only:
//   let msl: String — the full concatenated kernel library.
// Keep every kernel here in one piece so the library compiles with a
// single `device.makeLibrary(source:options:)` call.

let msl = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// RMSNorm with learnable scale gamma.
kernel void rms_norm(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device half* y [[buffer(2)]], constant uint& D [[buffer(3)]], constant float& eps [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    float s = 0.0f;
    for (uint i = t; i < D; i += 32) { float v = float(x[b*D+i]); s += v*v; }
    s = simd_sum(s);
    float sc = rsqrt(s/float(D) + eps);
    for (uint i = t; i < D; i += 32) { y[b*D+i] = half(float(x[b*D+i]) * sc * float(gamma[i])); }
}

// RMSNorm without gamma (Gemma-4 uses this for v_norm in attention, and
// inside the router pre-projection). Just rsqrt(mean_sq + eps) normalize.
kernel void rms_norm_noscale(
    device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
    constant uint& D [[buffer(2)]], constant float& eps [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    float s = 0.0f;
    for (uint i = t; i < D; i += 32) { float v = float(x[b*D+i]); s += v*v; }
    s = simd_sum(s);
    float sc = rsqrt(s/float(D) + eps);
    for (uint i = t; i < D; i += 32) { y[b*D+i] = half(float(x[b*D+i]) * sc); }
}

// Per-layer scalar multiply (layer_output_scale applied at the end of each block).
kernel void scale_by_scalar(
    device half* x [[buffer(0)]], device const float* scalar_buf [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    float s = scalar_buf[0];
    for (uint i = t; i < N; i += 32) { x[b*N+i] = half(float(x[b*N+i]) * s); }
}

// Router pre-projection: RMSNorm_noscale(x) * ffn_gate_inp.scale[D] * (1/sqrt(D))
// Fuses the norm + per-dim scale + sqrt(D) divisor in one kernel.
kernel void router_prenorm_scale(
    device const half* x [[buffer(0)]], device const float* per_dim_scale [[buffer(1)]],
    device half* y [[buffer(2)]], constant uint& D [[buffer(3)]], constant float& eps [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    float s = 0.0f;
    for (uint i = t; i < D; i += 32) { float v = float(x[b*D+i]); s += v*v; }
    s = simd_sum(s);
    float rsq = rsqrt(s/float(D) + eps);
    float inv_sqrt_d = rsqrt(float(D));
    for (uint i = t; i < D; i += 32) {
        y[b*D+i] = half(float(x[b*D+i]) * rsq * per_dim_scale[i] * inv_sqrt_d);
    }
}

// GELU(gate) * up — in-place on gate. Tanh-approx (Gemma uses gelu_pytorch_tanh):
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
kernel void gelu_mul_inplace(
    device half* gate [[buffer(0)]], device const half* up [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    const float c = 0.7978845608f;        // sqrt(2/π)
    for (uint i = t; i < N; i += 32) {
        float g = float(gate[b*N+i]);
        float u = float(up[b*N+i]);
        // Clamp argument to tanh to avoid observed MSL fast-math NaN at
        // |inner|≈45 (likely from exp overflow in the internal tanh expansion).
        // tanh saturates to ±1 well before |x|=20 anyway.
        float inner = c * (g + 0.044715f * g * g * g);
        inner = clamp(inner, -20.0f, 20.0f);
        float gelu_g = 0.5f * g * (1.0f + tanh(inner));
        gate[b*N+i] = half(gelu_g * u);
    }
}

// MoE combine writing to a dedicated output buffer (not adding to hidden).
// Used in the Gemma-4 flow where MoE output needs its own post-norm before
// being summed with the MLP branch and then added to the residual.
kernel void moe_combine_write(
    device const half* expert_out  [[buffer(0)]],
    device const uint* batch_slots [[buffer(1)]],
    device const float* gate_w     [[buffer(2)]],
    device half* out               [[buffer(3)]],
    constant uint& top_k           [[buffer(4)]],
    constant uint& D               [[buffer(5)]],
    uint2 tg                       [[threadgroup_position_in_grid]],
    uint2 lid                      [[thread_position_in_threadgroup]])
{
    uint db = tg.x; uint b = tg.y; uint t = lid.x; uint d = db * 32 + t;
    if (d >= D) return;
    float acc = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
        uint slot = batch_slots[b * top_k + k];
        float w = gate_w[b * top_k + k];
        acc += w * float(expert_out[slot * D + d]);
    }
    out[b * D + d] = half(acc);
}

// Fused: y = RMSNorm(x, gamma) + residual. Replaces 3 kernels
// (norm-in-place + copy + add) with one — saves ~40 μs per use.
kernel void rms_norm_add(
    device const half* x         [[buffer(0)]],
    device const half* gamma     [[buffer(1)]],
    device const half* residual  [[buffer(2)]],
    device half* out             [[buffer(3)]],
    constant uint& N             [[buffer(4)]],
    constant float& eps          [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    float s = 0.0f;
    for (uint i = t; i < N; i += 32) { float v = float(x[b*N+i]); s += v * v; }
    s = simd_sum(s);
    float scale = rsqrt(s / float(N) + eps);
    for (uint i = t; i < N; i += 32) {
        float v = float(x[b*N+i]) * scale * float(gamma[i]);
        out[b*N+i] = half(v + float(residual[b*N+i]));
    }
}

// Fused: y = (RMSNorm(x, gamma) + residual) * scalar. Replaces 4 kernels
// (norm + copy + add + scale) — saves ~60 μs per use. Used at end of layer
// to combine post_ffw_norm + residual add + layer_output_scale in one pass.
kernel void rms_norm_add_scale(
    device const half* x         [[buffer(0)]],
    device const half* gamma     [[buffer(1)]],
    device const half* residual  [[buffer(2)]],
    device const float* scalar_buf [[buffer(3)]],
    device half* out             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    constant float& eps          [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    float s = 0.0f;
    for (uint i = t; i < N; i += 32) { float v = float(x[b*N+i]); s += v * v; }
    s = simd_sum(s);
    float scale = rsqrt(s / float(N) + eps);
    float ls = scalar_buf[0];
    for (uint i = t; i < N; i += 32) {
        float v = float(x[b*N+i]) * scale * float(gamma[i]);
        out[b*N+i] = half((v + float(residual[b*N+i])) * ls);
    }
}

// Elementwise add: dst += src, over [B, N] tensors.
kernel void add_inplace(
    device half* dst [[buffer(0)]], device const half* src [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    for (uint i = t; i < N; i += 32) { dst[b*N+i] = half(float(dst[b*N+i]) + float(src[b*N+i])); }
}

// Residual-stream measurement: intensities[b] = dot(dst[b, :], meas[:])
// Pure read — doesn't modify the residual. One 32-thread TG per slot;
// per-lane partial sum is combined via simd_sum then lane 0 writes the
// scalar out. Paired with add_scaled_cvector_fp16 to form the detect/
// trigger/effect loop: this kernel's output gets read back CPU-side
// between ticks and fed into the trigger evaluator, which optionally
// restarts an effector ADSR envelope for the NEXT tick. No in-CB
// coupling — the detector and effector are fully decoupled feedforward.
kernel void measure_dot_fp16(
    device const half*  src          [[buffer(0)]],  // [B, N] residual stream
    device const half*  meas         [[buffer(1)]],  // [N]    measurement direction
    device       float* intensities  [[buffer(2)]],  // [B]    output, one scalar per slot
    constant uint&      N            [[buffer(3)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    float acc = 0.0f;
    for (uint i = t; i < N; i += 32) {
        acc += float(src[b*N + i]) * float(meas[i]);
    }
    acc = simd_sum(acc);
    if (t == 0) intensities[b] = acc;
}

// Residual-stream control-vector injection: dst[b, :] += mag * cvec[:]
// One cvec is shared across every batch slot (same steering vector applies
// to every sequence in the active batch). Bandwidth-bound and tiny — cvec
// is HIDDEN halves (~5.6 KB at HIDDEN=2816), streams once per dispatch,
// fan-out across B slots is free. Caller passes `mag` as a scalar uniform;
// the ADSR evaluator lives CPU-side and writes a fresh mag per tick.
kernel void add_scaled_cvector_fp16(
    device       half*  dst   [[buffer(0)]],   // [B, N] residual stream
    device const half*  cvec  [[buffer(1)]],   // [N] control vector
    constant uint&      N     [[buffer(2)]],
    constant float&     mag   [[buffer(3)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    for (uint i = t; i < N; i += 32) {
        dst[b*N + i] = half(float(dst[b*N + i]) + mag * float(cvec[i]));
    }
}

// Dense GEMV v5 (split-K, 4 SGs × 128 threads/TG) — best for D_out ≤ 8192
kernel void dense_gemv_v5(
    device const half* hidden [[buffer(0)]], device const half* W [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& D_in [[buffer(3)]], constant uint& D_out [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    constexpr uint N_SPLITS = 4;
    uint n_block = tg.x; uint b = tg.y; uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    device const half* hid_b = hidden + b * D_in;
    device const half* w_col = W + n;
    uint k_per_sg = D_in / N_SPLITS;
    uint k_begin = sg_id * k_per_sg;
    uint k_end = k_begin + k_per_sg;
    float acc = 0.0f;
    for (uint k = k_begin; k < k_end; k += 8) {
        half w0 = w_col[(k+0)*D_out], w1 = w_col[(k+1)*D_out], w2 = w_col[(k+2)*D_out], w3 = w_col[(k+3)*D_out];
        half w4 = w_col[(k+4)*D_out], w5 = w_col[(k+5)*D_out], w6 = w_col[(k+6)*D_out], w7 = w_col[(k+7)*D_out];
        half h0 = hid_b[k+0], h1 = hid_b[k+1], h2 = hid_b[k+2], h3 = hid_b[k+3];
        half h4 = hid_b[k+4], h5 = hid_b[k+5], h6 = hid_b[k+6], h7 = hid_b[k+7];
        acc += float(h0)*float(w0) + float(h1)*float(w1) + float(h2)*float(w2) + float(h3)*float(w3)
             + float(h4)*float(w4) + float(h5)*float(w5) + float(h6)*float(w6) + float(h7)*float(w7);
    }
    threadgroup float partials[4][32];
    partials[sg_id][lid_sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        float total = partials[0][lid_sg] + partials[1][lid_sg] + partials[2][lid_sg] + partials[3][lid_sg];
        output[b * D_out + n] = half(total);
    }
}

// Dense GEMV v4 (multi-batch per TG, K-unroll) — best for huge D_out (unembed)
kernel void dense_gemv_v4(
    device const half* hidden [[buffer(0)]], device const half* W [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& B [[buffer(3)]],
    constant uint& D_in [[buffer(4)]], constant uint& D_out [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    constexpr uint MAX_B = 8;
    uint n_block = tg.x; uint lidx = lid.x;
    uint n = n_block * 32 + lidx;
    if (n >= D_out) return;
    device const half* w_col = W + n;
    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;
    for (uint k = 0; k < D_in; k += 8) {
        half w0 = w_col[(k+0)*D_out], w1 = w_col[(k+1)*D_out], w2 = w_col[(k+2)*D_out], w3 = w_col[(k+3)*D_out];
        half w4 = w_col[(k+4)*D_out], w5 = w_col[(k+5)*D_out], w6 = w_col[(k+6)*D_out], w7 = w_col[(k+7)*D_out];
        for (uint b = 0; b < B; ++b) {
            device const half* hid = hidden + b * D_in + k;
            accs[b] += float(hid[0])*float(w0) + float(hid[1])*float(w1) + float(hid[2])*float(w2) + float(hid[3])*float(w3)
                     + float(hid[4])*float(w4) + float(hid[5])*float(w5) + float(hid[6])*float(w6) + float(hid[7])*float(w7);
        }
    }
    for (uint b = 0; b < B; ++b) output[b * D_out + n] = half(accs[b]);
}

// Fused dense GEMV v4 + Gemma softcap (tanh(y/cap)*cap). Unembed at
// D_out=262144 is the only production user. The standalone softcap kernel
// was launching 4 TGs of 32 threads across a 40-core GPU, dispatch-bound at
// ~2 ms for what's 13 µs of DRAM bandwidth; folding it into the GEMV write
// eliminates a whole kernel dispatch and its DRAM round-trip.
kernel void dense_gemv_v4_softcap(
    device const half* hidden [[buffer(0)]], device const half* W [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& B [[buffer(3)]],
    constant uint& D_in [[buffer(4)]], constant uint& D_out [[buffer(5)]],
    constant float& cap [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    constexpr uint MAX_B = 8;
    uint n_block = tg.x; uint lidx = lid.x;
    uint n = n_block * 32 + lidx;
    if (n >= D_out) return;
    device const half* w_col = W + n;
    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;
    for (uint k = 0; k < D_in; k += 8) {
        half w0 = w_col[(k+0)*D_out], w1 = w_col[(k+1)*D_out], w2 = w_col[(k+2)*D_out], w3 = w_col[(k+3)*D_out];
        half w4 = w_col[(k+4)*D_out], w5 = w_col[(k+5)*D_out], w6 = w_col[(k+6)*D_out], w7 = w_col[(k+7)*D_out];
        for (uint b = 0; b < B; ++b) {
            device const half* hid = hidden + b * D_in + k;
            accs[b] += float(hid[0])*float(w0) + float(hid[1])*float(w1) + float(hid[2])*float(w2) + float(hid[3])*float(w3)
                     + float(hid[4])*float(w4) + float(hid[5])*float(w5) + float(hid[6])*float(w6) + float(hid[7])*float(w7);
        }
    }
    float inv = 1.0f / cap;
    for (uint b = 0; b < B; ++b) output[b * D_out + n] = half(tanh(accs[b] * inv) * cap);
}

// RoPE rotate_half — Gemma-4 convention: split at head_dim/2, rotate first
// rotary_dims/2 elements paired with elements at [D/2, D/2 + rotary_dims/2).
// For full rotary (rotary == D): splits at D/2, rotates whole head.
// For partial rotary (rotary < D): rotates only first rotary_dims, still
// paired across the D/2 boundary (NOT at rotary/2 — that was a bug).
kernel void rope_half(
    device half* x [[buffer(0)]], device const uint* positions [[buffer(1)]],
    constant uint& H [[buffer(2)]], constant uint& D [[buffer(3)]], constant uint& rotary [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint h = tg.y; uint t = lid.x;
    uint pos = positions[b];
    uint half_head = D / 2;             // split index (Gemma's rotate_half)
    uint n_pairs = rotary / 2;          // how many pair-dims to actually rotate
    device half* xbh = x + (b * H + h) * D;
    // Gemma-4 frequency formula: inv_freq[i] = 1 / base^(2*i / head_dim).
    // The denominator is head_dim, NOT rotary — for full-attention layers
    // (head_dim=512, rotary=128) these differ by 4×, and the rotated sub-
    // spectrum must use the SAME base-of-exponent as a full-rotation layer
    // would, just truncated to fewer pairs. HF's `_compute_proportional_rope_
    // parameters` builds exactly this: the first rope_angles freqs use head_
    // dim as denominator, remaining angles are zero (identity).
    for (uint i = t; i < n_pairs; i += 32) {
        float freq = 1.0f / pow(theta, 2.0f * float(i) / float(D));
        float ang = float(pos) * freq; float c = cos(ang); float s = sin(ang);
        float x0 = float(xbh[i]);
        float x1 = float(xbh[i + half_head]);
        xbh[i]             = half(x0 * c - x1 * s);
        xbh[i + half_head] = half(x0 * s + x1 * c);
    }
}

// Multi-position variant of kv_write. Writes q_len positions per batch
// at slots q_positions[b, q_local]. Used by prefill.
// K, V layout: [B, Q_LEN, H, D]. q_positions: [B, Q_LEN].
kernel void kv_write_multi(
    device const half* K [[buffer(0)]], device const half* V [[buffer(1)]],
    device half* K_cache [[buffer(2)]], device half* V_cache [[buffer(3)]],
    device const uint* block_table [[buffer(4)]], device const uint* q_positions [[buffer(5)]],
    constant uint& H [[buffer(6)]], constant uint& D [[buffer(7)]],
    constant uint& PAGE [[buffer(8)]], constant uint& max_pages [[buffer(9)]],
    constant uint& q_len [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint q_local = tg.y; uint h = tg.z; uint t = lid.x;
    uint q_flat = b * q_len + q_local;
    uint pos = q_positions[q_flat];
    uint lp = pos / PAGE; uint off = pos % PAGE;
    uint phys = block_table[b * max_pages + lp];
    device const half* Ks = K + (q_flat * H + h) * D;
    device const half* Vs = V + (q_flat * H + h) * D;
    device half* Kd = K_cache + ((phys * PAGE + off) * H + h) * D;
    device half* Vd = V_cache + ((phys * PAGE + off) * H + h) * D;
    for (uint i = t; i < D; i += 32) { Kd[i] = Ks[i]; Vd[i] = Vs[i]; }
}

// Multi-position variant of rope_half. Rotates each (b, q_local) row by
// its own position q_positions[b, q_local]. Used by prefill for Q and K.
// x layout: [B, Q_LEN, H, D]. q_positions: [B, Q_LEN].
kernel void rope_half_multi(
    device half* x [[buffer(0)]], device const uint* q_positions [[buffer(1)]],
    constant uint& H [[buffer(2)]], constant uint& D [[buffer(3)]],
    constant uint& rotary [[buffer(4)]], constant float& theta [[buffer(5)]],
    constant uint& q_len [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint q_local = tg.y; uint h = tg.z; uint t = lid.x;
    uint q_flat = b * q_len + q_local;
    uint pos = q_positions[q_flat];
    uint half_head = D / 2;
    uint n_pairs = rotary / 2;
    device half* xbh = x + (q_flat * H + h) * D;
    for (uint i = t; i < n_pairs; i += 32) {
        float freq = 1.0f / pow(theta, 2.0f * float(i) / float(D));
        float ang = float(pos) * freq; float c = cos(ang); float s = sin(ang);
        float x0 = float(xbh[i]);
        float x1 = float(xbh[i + half_head]);
        xbh[i]             = half(x0 * c - x1 * s);
        xbh[i + half_head] = half(x0 * s + x1 * c);
    }
}

// bf16 → fp16 conversion. bf16 is the top 16 bits of fp32; we reinterpret
// src[i] as a ushort, shift it into the top half of a uint, cast to float,
// then cast to half. Used by VisionResidency to rehydrate vision weights
// from a mmap-backed (file-evictable) bf16 source into a working fp16
// buffer that the kernels actually read. 1 thread per element; the dispatch
// is pure memory-bandwidth, ~10 ms to convert 1 GB on M5.
kernel void bf16_to_fp16(
    device const ushort* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    uint bits = uint(src[i]) << 16;
    float f = as_type<float>(bits);
    dst[i] = half(f);
}

// KV cache write
kernel void kv_write(
    device const half* K [[buffer(0)]], device const half* V [[buffer(1)]],
    device half* K_cache [[buffer(2)]], device half* V_cache [[buffer(3)]],
    device const uint* block_table [[buffer(4)]], device const uint* positions [[buffer(5)]],
    constant uint& H [[buffer(6)]], constant uint& D [[buffer(7)]],
    constant uint& PAGE [[buffer(8)]], constant uint& max_pages [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint h = tg.y; uint t = lid.x;
    uint pos = positions[b]; uint lp = pos / PAGE; uint off = pos % PAGE;
    uint phys = block_table[b * max_pages + lp];
    device const half* Ks = K + (b * H + h) * D; device const half* Vs = V + (b * H + h) * D;
    device half* Kd = K_cache + ((phys * PAGE + off) * H + h) * D;
    device half* Vd = V_cache + ((phys * PAGE + off) * H + h) * D;
    for (uint i = t; i < D; i += 32) { Kd[i] = Ks[i]; Vd[i] = Vs[i]; }
}

// ---- Split-KV paged attention with simdgroup_matrix QK/AV ----
// Pattern stolen from llama.cpp's kernel_flash_attn_ext_impl:
//   - K loaded direct from device (no tg-mem round-trip) via simdgroup_load
//     with transpose=true
//   - QK: 8×8 simdgroup_multiply_accumulate per d-tile
//   - AV: 8×8 simdgroup_multiply_accumulate per d-tile
// For decode Q=1: we pad Q to 8 rows (7 unused). The 8×8 MMA still beats
// scalar-cooperative QK because each MMA instruction is ~1 cycle of MAC
// throughput equivalent to 64 scalar MACs.

kernel void paged_attn_slide_sgmm_compute(
    device const half* Q                    [[buffer(0)]],
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],
    device float* m_partials                [[buffer(4)]],
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],
    device const uint* num_pages_per_slot   [[buffer(7)]],
    device const uint* k_len_per_slot       [[buffer(8)]],
    constant float& qk_scale                [[buffer(9)]],
    constant uint& max_pages                [[buffer(10)]],
    constant uint& H_Q                      [[buffer(11)]],
    constant uint& H_KV                     [[buffer(12)]],
    constant uint& N_SPLITS                 [[buffer(13)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 256;
    constexpr uint PAGE = 16;
    constexpr uint THREADS = 32;
    constexpr uint D8 = D / 8;            // 32
    const uint vs = tg_pos.x;
    const uint split = tg_pos.y;
    const uint slot = vs / H_Q;
    const uint q_head = vs % H_Q;
    const uint kv_head = (q_head * H_KV) / H_Q;
    const uint lid = lid3.x;

    // Q_tile: padded to 8×D (row 0 = real Q, rows 1-7 = zero). In tg-mem so
    // simdgroup_load can read 8×8 slices efficiently.
    threadgroup half Q_tile[8 * D];
    // Scores tile: 8×PAGE accumulator for QK^T output. Padded the same way.
    threadgroup half scores_tile[8 * PAGE];
    // O_acc: 8×D (row 0 = real output, rest ignored).
    threadgroup float O_acc[D];
    threadgroup float m_state[1];
    threadgroup float l_state[1];

    device const half* Q_s = Q + (slot * H_Q + q_head) * D;
    device const uint* bt_s = block_table + slot * max_pages;

    // Initialize: row 0 = Q, rows 1-7 = 0. O_acc = 0. m, l initial.
    for (uint i = lid; i < 8 * D; i += THREADS) {
        uint r = i / D;
        Q_tile[i] = (r == 0) ? Q_s[i % D] : half(0);
    }
    for (uint i = lid; i < D; i += THREADS) O_acc[i] = 0.0f;
    if (lid == 0) { m_state[0] = -INFINITY; l_state[0] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pages = num_pages_per_slot[slot];
    const uint pages_per_split = (total_pages + N_SPLITS - 1) / N_SPLITS;
    const uint p_begin = split * pages_per_split;
    const uint p_end   = min(p_begin + pages_per_split, total_pages);
    const uint k_len   = k_len_per_slot[slot];
    const uint kv_row_stride = H_KV * D;  // stride in halves between KV rows in cache

    for (uint p = p_begin; p < p_end; ++p) {
        const uint phys = bt_s[p];
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;

        // --- QK: mqk[8x8] = Q_tile[8xD_slice] * K^T[D_slice x 8] ---
        // PAGE=16 requires TWO 8x8 K-blocks (pb=0,1). For each block and each
        // d-tile, accumulate mma into mqk.
        for (uint pb = 0; pb < PAGE / 8; ++pb) {
            simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            device const half* pk = Kbase + (pb * 8) * kv_row_stride;
            for (uint dt = 0; dt < D8; ++dt) {
                simdgroup_half8x8 mq, mk;
                simdgroup_load(mq, Q_tile + dt * 8, D);
                // Load K tile with transpose: K[8 rows × 8 cols] -> 8×8 matrix
                // where the loaded 8×8 is K^T (cols become rows in the matrix).
                simdgroup_load(mk, pk + dt * 8, kv_row_stride, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }
            // Store the 8×8 scores for this K-block into scores_tile
            simdgroup_store(mqk, scores_tile + pb * 8, PAGE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Online softmax on row 0 of scores_tile (the real scores) ---
        // lid < PAGE reads its score column; compute max/sum via simd ops.
        float my_score = -INFINITY;
        if (lid < PAGE) {
            my_score = float(scores_tile[lid]) * qk_scale;
            uint k_pos = p * PAGE + lid;
            if (k_pos >= k_len) my_score = -INFINITY;
        }
        float page_max = simd_max(my_score);
        float m_old = m_state[0];
        float m_new = max(m_old, page_max);
        float scale = exp(m_old - m_new);
        float my_exp = (lid < PAGE) ? exp(my_score - m_new) : 0.0f;
        float page_sum = simd_sum(my_exp);
        if (lid < PAGE) {
            // Write exp'd scores back to row 0 of scores_tile; rows 1-7 zero.
            scores_tile[lid] = half(my_exp);
        }
        // Zero the padded rows of scores_tile (simdgroup_load would bring in garbage)
        if (lid < PAGE * 7) {
            uint pad_i = PAGE + lid;   // position in scores_tile past row 0
            scores_tile[pad_i] = half(0);
        }
        if (lid == 0) { l_state[0] = l_state[0] * scale + page_sum; m_state[0] = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- AV: O_acc[8×D] += scores[8×PAGE] × V[PAGE×D] ---
        // Each d-tile: one MMA. Scale existing O_acc by `scale` first (Flash update).
        for (uint dt = 0; dt < D8; ++dt) {
            // Scale existing O_acc slab for this d-tile
            uint d0 = dt * 8;
            // O_acc[d0..d0+8] *= scale (only row 0 matters; 8 floats)
            // Done in a single lane to keep simple (PAGE < 8 so only a few elements)
            // Actually each thread handles its own float; use lid for dim coverage
            if (lid < 8) {
                O_acc[d0 + lid] *= scale;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // mvv = scores × V(d_tile)
            simdgroup_half8x8 mv;
            simdgroup_half8x8 ms;
            simdgroup_float8x8 mo = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_load(ms, scores_tile, PAGE);   // 8×PAGE but we only use 8 cols at a time below
            // Inner loop over PAGE/8=2 K-blocks of V
            for (uint pb = 0; pb < PAGE / 8; ++pb) {
                device const half* pv = Vbase + (pb * 8) * kv_row_stride + d0;
                simdgroup_load(mv, pv, kv_row_stride);
                simdgroup_multiply_accumulate(mo, ms, mv, mo);
                // Shift scores_tile view for next K-block (reuse ms load)
                simdgroup_load(ms, scores_tile + (pb + 1) * 8, PAGE);
            }
            // Extract row 0 of mo (the real O slice) and add to O_acc
            // Use tg-mem as a scratch for the extraction
            threadgroup float o_scratch[64];
            simdgroup_store(mo, o_scratch, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < 8) {
                O_acc[d0 + lid] += o_scratch[lid];   // row 0, cols 0..7
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write partials
    uint pidx = vs * N_SPLITS + split;
    const bool empty = (p_begin >= p_end);
    if (lid == 0) {
        m_partials[pidx] = empty ? -INFINITY : m_state[0];
        l_partials[pidx] = empty ? 0.0f : l_state[0];
    }
    device float* O_part = O_partials + pidx * D;
    for (uint d = lid; d < D; d += THREADS) O_part[d] = O_acc[d];
}

// GQA-grouped sliding-attn: one TG per (slot, kv_head, split), processing
// all H_Q/H_KV=2 Q heads that share this kv_head. Half the KV DRAM reads,
// half the TGs. Per-MMA utilization stays low (2/8 rows) so AV is scalar.
kernel void paged_attn_slide_gqa_compute(
    device const half* Q                    [[buffer(0)]],
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],
    device float* m_partials                [[buffer(4)]],
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],
    device const uint* num_pages_per_slot   [[buffer(7)]],
    device const uint* k_len_per_slot       [[buffer(8)]],
    constant float& qk_scale                [[buffer(9)]],
    constant uint& max_pages                [[buffer(10)]],
    constant uint& H_Q                      [[buffer(11)]],
    constant uint& H_KV                     [[buffer(12)]],
    constant uint& N_SPLITS                 [[buffer(13)]],
    constant uint& sliding_window           [[buffer(14)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 256;
    constexpr uint PAGE = 16;
    constexpr uint THREADS = 32;
    constexpr uint Q_PER_TG = 2;          // H_Q/H_KV for sliding-attn
    constexpr uint D8 = D / 8;            // 32

    const uint vs = tg_pos.x;             // 0..B*H_KV
    const uint split = tg_pos.y;
    const uint slot = vs / H_KV;
    const uint kv_head = vs % H_KV;
    const uint q_head_base = kv_head * Q_PER_TG;
    const uint lid = lid3.x;

    // Q_tile padded to 8 rows so simdgroup_load works; rows 0..Q_PER_TG-1 real.
    threadgroup half  Q_tile[8 * D];
    threadgroup half  scores_tile[8 * PAGE];   // only rows 0..Q_PER_TG-1 real
    threadgroup float O_acc[Q_PER_TG * D];
    threadgroup float m_state[Q_PER_TG];
    threadgroup float l_state[Q_PER_TG];
    threadgroup float scale_tile[Q_PER_TG];

    device const half* Qbase = Q + (slot * H_Q + q_head_base) * D;
    device const uint* bt_s = block_table + slot * max_pages;

    // Load real Q rows, zero the padded rows (so sgmm of padded rows is 0).
    for (uint i = lid; i < 8 * D; i += THREADS) {
        uint r = i / D;
        Q_tile[i] = (r < Q_PER_TG) ? Qbase[r * D + (i % D)] : half(0);
    }
    for (uint i = lid; i < Q_PER_TG * D; i += THREADS) O_acc[i] = 0.0f;
    if (lid < Q_PER_TG) { m_state[lid] = -INFINITY; l_state[lid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pages = num_pages_per_slot[slot];
    const uint pages_per_split = (total_pages + N_SPLITS - 1) / N_SPLITS;
    const uint p_begin = split * pages_per_split;
    const uint p_end   = min(p_begin + pages_per_split, total_pages);
    const uint k_len   = k_len_per_slot[slot];
    const uint kv_row_stride = H_KV * D;
    // Sliding-window lower bound (inclusive): any k_pos < window_lo is masked.
    // 0 means "no lower bound" (short context or sliding_window==0 disables).
    const uint window_lo = (sliding_window > 0 && k_len > sliding_window)
                           ? (k_len - sliding_window) : 0u;

    for (uint p = p_begin; p < p_end; ++p) {
        // Tile-level skip: page entirely before the sliding window, or
        // entirely past the causal horizon. Either way no entries contribute.
        const uint page_kpos_hi = p * PAGE + PAGE;   // exclusive upper bound
        const uint page_kpos_lo = p * PAGE;          // inclusive lower bound
        if (page_kpos_hi <= window_lo) continue;     // entirely pre-window
        if (page_kpos_lo >= k_len)     continue;     // entirely past causal

        const uint phys = bt_s[p];
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;

        // QK: PAGE=16 needs two 8-col K blocks; D=256 → 32 d-tiles each.
        for (uint pb = 0; pb < PAGE / 8; ++pb) {
            simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            device const half* pk = Kbase + (pb * 8) * kv_row_stride;
            for (uint dt = 0; dt < D8; ++dt) {
                simdgroup_half8x8 mq, mk;
                simdgroup_load(mq, Q_tile + dt * 8, D);
                simdgroup_load(mk, pk + dt * 8, kv_row_stride, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }
            simdgroup_store(mqk, scores_tile + pb * 8, PAGE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row online softmax (lanes 0..Q_PER_TG-1 each handle one row).
        // Intra-tile mask: -INF if k_pos is past causal horizon OR before the
        // sliding window. Both conditions get collapsed here so partial pages
        // at either end of the valid range are handled uniformly.
        if (lid < Q_PER_TG) {
            const uint q = lid;
            float row_max = -INFINITY;
            float s_loc[PAGE];
            for (uint k = 0; k < PAGE; ++k) {
                float sv = float(scores_tile[q * PAGE + k]) * qk_scale;
                uint k_pos = p * PAGE + k;
                if (k_pos >= k_len || k_pos < window_lo) sv = -INFINITY;
                s_loc[k] = sv;
                row_max = max(row_max, sv);
            }
            float m_old = m_state[q];
            float m_new = max(m_old, row_max);
            float scale = exp(m_old - m_new);
            float sum = 0.0f;
            for (uint k = 0; k < PAGE; ++k) {
                float e = exp(s_loc[k] - m_new);
                scores_tile[q * PAGE + k] = half(e);
                sum += e;
            }
            m_state[q] = m_new;
            l_state[q] = l_state[q] * scale + sum;
            scale_tile[q] = scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // AV: scalar cooperative. Each lane owns D/THREADS=8 d-values.
        // Stage V[0..PAGE, d] in registers, reuse across Q heads.
        for (uint d = lid; d < D; d += THREADS) {
            half V_reg[PAGE];
            for (uint k = 0; k < PAGE; ++k) {
                V_reg[k] = Vbase[k * kv_row_stride + d];
            }
            for (uint q = 0; q < Q_PER_TG; ++q) {
                float acc = O_acc[q * D + d] * scale_tile[q];
                for (uint k = 0; k < PAGE; ++k) {
                    acc += float(scores_tile[q * PAGE + k]) * float(V_reg[k]);
                }
                O_acc[q * D + d] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials per Q head (pidx layout matches reduce kernel expectations).
    const bool empty = (p_begin >= p_end);
    for (uint q = 0; q < Q_PER_TG; ++q) {
        const uint q_head = q_head_base + q;
        const uint pidx = (slot * H_Q + q_head) * N_SPLITS + split;
        if (lid == 0) {
            m_partials[pidx] = empty ? -INFINITY : m_state[q];
            l_partials[pidx] = empty ? 0.0f : l_state[q];
        }
        device float* O_part = O_partials + pidx * D;
        for (uint d = lid; d < D; d += THREADS) O_part[d] = O_acc[q * D + d];
    }
}

// Same sgmm pattern for full-attn (D=512 PAGE=8). This is where long-context
// gap vs llama.cpp is biggest, so sgmm-ing here matters most.
// Hybrid attention for full-attn layers (D=512 PAGE=8):
//   - QK via simdgroup_matrix (batch K cols 8-at-a-time → fast)
//   - AV via scalar-cooperative (no wasted rows at Q=1)
// Avoids the per-d-tile scratch extraction problem that sunk the all-sgmm
// variant at D=512.
kernel void paged_attn_full_sgmm_compute(
    device const half* Q                    [[buffer(0)]],
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],
    device float* m_partials                [[buffer(4)]],
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],
    device const uint* num_pages_per_slot   [[buffer(7)]],
    device const uint* k_len_per_slot       [[buffer(8)]],
    constant float& qk_scale                [[buffer(9)]],
    constant uint& max_pages                [[buffer(10)]],
    constant uint& H_Q                      [[buffer(11)]],
    constant uint& H_KV                     [[buffer(12)]],
    constant uint& N_SPLITS                 [[buffer(13)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 512;
    constexpr uint PAGE = 8;
    constexpr uint THREADS = 32;
    constexpr uint D8 = D / 8;                // 64
    const uint vs = tg_pos.x;
    const uint split = tg_pos.y;
    const uint slot = vs / H_Q;
    const uint q_head = vs % H_Q;
    const uint kv_head = (q_head * H_KV) / H_Q;
    const uint lid = lid3.x;

    threadgroup half Q_tile[8 * D];           // 8 KB — padded Q for sgmm QK
    threadgroup half V_tile[PAGE * D];        // 8 KB — V in tg-mem for scalar AV
    threadgroup half scores_tile[8 * PAGE];   // QK output, row 0 = real scores
    threadgroup float O_acc[D];
    threadgroup float m_state[1];
    threadgroup float l_state[1];

    device const half* Q_s = Q + (slot * H_Q + q_head) * D;
    device const uint* bt_s = block_table + slot * max_pages;

    for (uint i = lid; i < 8 * D; i += THREADS) {
        uint r = i / D;
        Q_tile[i] = (r == 0) ? Q_s[i % D] : half(0);
    }
    for (uint i = lid; i < D; i += THREADS) O_acc[i] = 0.0f;
    if (lid == 0) { m_state[0] = -INFINITY; l_state[0] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pages = num_pages_per_slot[slot];
    const uint pages_per_split = (total_pages + N_SPLITS - 1) / N_SPLITS;
    const uint p_begin = split * pages_per_split;
    const uint p_end   = min(p_begin + pages_per_split, total_pages);
    const uint k_len   = k_len_per_slot[slot];
    const uint kv_row_stride = H_KV * D;

    for (uint p = p_begin; p < p_end; ++p) {
        const uint phys = bt_s[p];
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;

        // Load V into tg-mem (for scalar AV later). Cooperative, 32 threads.
        // PAGE*D = 4096 halves; per-thread 128 scalar loads.
        for (uint i = lid; i < PAGE * D; i += THREADS) {
            uint row = i / D; uint d = i % D;
            V_tile[i] = V_cache[((phys * PAGE + row) * H_KV + kv_head) * D + d];
        }
        // K does not need tg-mem — simdgroup_load with transpose reads directly.

        // QK via simdgroup_matrix (batches K cols 8 at a time). Transposed K load
        // from K_cache directly — the real win at D=512 (64 d-tiles).
        simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        for (uint dt = 0; dt < D8; ++dt) {
            simdgroup_half8x8 mq, mk;
            simdgroup_load(mq, Q_tile + dt * 8, D);
            simdgroup_load(mk, Kbase + dt * 8, kv_row_stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }
        simdgroup_store(mqk, scores_tile, PAGE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax update
        float my_score = -INFINITY;
        if (lid < PAGE) {
            my_score = float(scores_tile[lid]) * qk_scale;
            uint k_pos = p * PAGE + lid;
            if (k_pos >= k_len) my_score = -INFINITY;
        }
        float page_max = simd_max(my_score);
        float m_old = m_state[0];
        float m_new = max(m_old, page_max);
        float scale = exp(m_old - m_new);
        float my_exp = (lid < PAGE) ? exp(my_score - m_new) : 0.0f;
        float page_sum = simd_sum(my_exp);
        if (lid < PAGE) scores_tile[lid] = half(my_exp);
        if (lid == 0) { l_state[0] = l_state[0] * scale + page_sum; m_state[0] = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // AV via scalar-cooperative (each lane owns D/32 dims, no wasted compute).
        // O_acc[d] = O_acc[d]*scale + sum_r(scores[r] * V_tile[r, d])
        for (uint d = lid; d < D; d += THREADS) {
            float acc = O_acc[d] * scale;
            for (uint r = 0; r < PAGE; ++r) {
                acc += float(scores_tile[r]) * float(V_tile[r * D + d]);
            }
            O_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint pidx = vs * N_SPLITS + split;
    const bool empty = (p_begin >= p_end);
    if (lid == 0) {
        m_partials[pidx] = empty ? -INFINITY : m_state[0];
        l_partials[pidx] = empty ? 0.0f : l_state[0];
    }
    device float* O_part = O_partials + pidx * D;
    for (uint d = lid; d < D; d += THREADS) O_part[d] = O_acc[d];
}

// GQA-grouped full-attn: one TG per (slot, kv_head, split), processing all
// H_Q/H_KV=8 Q heads that share this kv_head. Two wins over the per-Q-head
// variants:
//   1) KV DRAM reads collapse 8:1 (one TG reads each KV row once instead of
//      eight separate TGs reading the same row). At P=16384 this is the
//      dominant bandwidth cost.
//   2) QK simdgroup_matrix now has Q=8 REAL rows — the 8x8 MMA is fully
//      utilized (64 real outputs/op), not 7/8 wasted as at Q=1.
// Partials layout is unchanged: reduce kernel sees (slot*H_Q + q_head, split)
// pidx, so each TG writes Q_PER_TG=8 pidx values before exit.
kernel void paged_attn_full_gqa_compute(
    device const half* Q                    [[buffer(0)]],
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],
    device float* m_partials                [[buffer(4)]],
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],
    device const uint* num_pages_per_slot   [[buffer(7)]],
    device const uint* k_len_per_slot       [[buffer(8)]],
    constant float& qk_scale                [[buffer(9)]],
    constant uint& max_pages                [[buffer(10)]],
    constant uint& H_Q                      [[buffer(11)]],
    constant uint& H_KV                     [[buffer(12)]],
    constant uint& N_SPLITS                 [[buffer(13)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 512;
    constexpr uint PAGE = 8;
    constexpr uint THREADS = 32;
    constexpr uint Q_PER_TG = 8;           // H_Q/H_KV at full-attn layers
    constexpr uint D8 = D / 8;             // 64 d-tiles

    const uint vs = tg_pos.x;              // 0..B*H_KV
    const uint split = tg_pos.y;           // 0..N_SPLITS
    const uint slot = vs / H_KV;
    const uint kv_head = vs % H_KV;
    const uint q_head_base = kv_head * Q_PER_TG;
    const uint lid = lid3.x;

    threadgroup half  Q_tile[Q_PER_TG * D];          // 8 KB (8 real Q rows)
    threadgroup half  scores_tile[Q_PER_TG * PAGE];  // 128 B (8 x PAGE)
    threadgroup float O_acc[Q_PER_TG * D];           // 16 KB (float acc)
    threadgroup float m_state[Q_PER_TG];             // 32 B
    threadgroup float l_state[Q_PER_TG];             // 32 B
    threadgroup float scale_tile[Q_PER_TG];          // 32 B

    device const half* Qbase = Q + (slot * H_Q + q_head_base) * D;
    device const uint* bt_s = block_table + slot * max_pages;

    for (uint i = lid; i < Q_PER_TG * D; i += THREADS) Q_tile[i] = Qbase[i];
    for (uint i = lid; i < Q_PER_TG * D; i += THREADS) O_acc[i] = 0.0f;
    if (lid < Q_PER_TG) { m_state[lid] = -INFINITY; l_state[lid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pages = num_pages_per_slot[slot];
    const uint pages_per_split = (total_pages + N_SPLITS - 1) / N_SPLITS;
    const uint p_begin = split * pages_per_split;
    const uint p_end   = min(p_begin + pages_per_split, total_pages);
    const uint k_len   = k_len_per_slot[slot];
    const uint kv_row_stride = H_KV * D;

    for (uint p = p_begin; p < p_end; ++p) {
        const uint phys = bt_s[p];
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;

        // QK: 8x8 MMA, Q=8 real rows. One MMA per d-tile; 64 d-tiles at D=512.
        // K loaded transposed directly from device memory (stride=kv_row_stride).
        simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        for (uint dt = 0; dt < D8; ++dt) {
            simdgroup_half8x8 mq, mk;
            simdgroup_load(mq, Q_tile + dt * 8, D);
            simdgroup_load(mk, Kbase + dt * 8, kv_row_stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }
        simdgroup_store(mqk, scores_tile, PAGE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row online softmax: lane q in 0..7 owns Q-head q. Lanes 8..31 idle
        // during this short scalar phase (PAGE=8 ops × a few math ops is cheap).
        if (lid < Q_PER_TG) {
            const uint q = lid;
            float row_max = -INFINITY;
            float s_loc[PAGE];
            for (uint k = 0; k < PAGE; ++k) {
                float sv = float(scores_tile[q * PAGE + k]) * qk_scale;
                uint k_pos = p * PAGE + k;
                if (k_pos >= k_len) sv = -INFINITY;
                s_loc[k] = sv;
                row_max = max(row_max, sv);
            }
            float m_old = m_state[q];
            float m_new = max(m_old, row_max);
            float scale = exp(m_old - m_new);
            float sum = 0.0f;
            for (uint k = 0; k < PAGE; ++k) {
                float e = exp(s_loc[k] - m_new);
                scores_tile[q * PAGE + k] = half(e);
                sum += e;
            }
            m_state[q] = m_new;
            l_state[q] = l_state[q] * scale + sum;
            scale_tile[q] = scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // AV: scalar cooperative. Each lane owns D/32=16 d-values. For each d,
        // stage V[0..PAGE, d] in registers (8 halves = 16 B), then reuse across
        // the 8 Q heads. No V tg-mem staging needed. Sgmm AV was tried but
        // the per-row scale step + simdgroup_load/store tg-mem round-trips per
        // d-tile (2 × 64 = 128) cost more than the MMA throughput saved.
        for (uint d = lid; d < D; d += THREADS) {
            half V_reg[PAGE];
            for (uint k = 0; k < PAGE; ++k) {
                V_reg[k] = Vbase[k * kv_row_stride + d];
            }
            for (uint q = 0; q < Q_PER_TG; ++q) {
                float acc = O_acc[q * D + d] * scale_tile[q];
                for (uint k = 0; k < PAGE; ++k) {
                    acc += float(scores_tile[q * PAGE + k]) * float(V_reg[k]);
                }
                O_acc[q * D + d] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials for all 8 Q heads. pidx matches the per-Q-head layout
    // the shared reduce kernel expects.
    const bool empty = (p_begin >= p_end);
    for (uint q = 0; q < Q_PER_TG; ++q) {
        const uint q_head = q_head_base + q;
        const uint pidx = (slot * H_Q + q_head) * N_SPLITS + split;
        if (lid == 0) {
            m_partials[pidx] = empty ? -INFINITY : m_state[q];
            l_partials[pidx] = empty ? 0.0f : l_state[q];
        }
        device float* O_part = O_partials + pidx * D;
        for (uint d = lid; d < D; d += THREADS) O_part[d] = O_acc[q * D + d];
    }
}

// ---- Split-KV paged attention: compute partials + reduce ----
// Grid compute: (B*H_Q, N_SPLITS) TGs. Each TG owns one (slot, q_head, split)
// and processes its page range via Flash online softmax, writing partial
// (m, l, O_unnorm) to partials buffers.
// Grid reduce: (B*H_Q,) TGs. Each combines N_SPLITS partials → final O using
// Flash-associative merge. 4× more parallel TGs than single-TG attention —
// fixes the ~66% latency-starved portion of attention cost on M5.

kernel void paged_attn_slide_split_compute(
    device const half* Q                    [[buffer(0)]],
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],
    device float* m_partials                [[buffer(4)]],    // [B*H_Q, N_SPLITS]
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],    // [B*H_Q, N_SPLITS, D]
    device const uint* num_pages_per_slot   [[buffer(7)]],
    device const uint* k_len_per_slot       [[buffer(8)]],
    constant float& qk_scale                [[buffer(9)]],
    constant uint& max_pages                [[buffer(10)]],
    constant uint& H_Q                      [[buffer(11)]],
    constant uint& H_KV                     [[buffer(12)]],
    constant uint& N_SPLITS                 [[buffer(13)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 256;
    constexpr uint PAGE = 16;
    constexpr uint THREADS = 32;
    const uint vs = tg_pos.x;
    const uint split = tg_pos.y;
    const uint slot = vs / H_Q;
    const uint q_head = vs % H_Q;
    const uint kv_head = (q_head * H_KV) / H_Q;
    const uint lid = lid3.x;

    threadgroup half Q_tile[D];
    threadgroup half K_tile[PAGE * D];
    threadgroup half V_tile[PAGE * D];
    threadgroup float scores[PAGE];
    threadgroup float m_state[1];
    threadgroup float l_state[1];
    threadgroup float O_acc[D];
    threadgroup float scores_tg[PAGE];

    device const half* Q_s = Q + (slot * H_Q + q_head) * D;
    device const uint* bt_s = block_table + slot * max_pages;
    for (uint i = lid; i < D; i += THREADS) { Q_tile[i] = Q_s[i]; O_acc[i] = 0.0f; }
    if (lid == 0) { m_state[0] = -INFINITY; l_state[0] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pages = num_pages_per_slot[slot];
    const uint pages_per_split = (total_pages + N_SPLITS - 1) / N_SPLITS;
    const uint p_begin = split * pages_per_split;
    const uint p_end   = min(p_begin + pages_per_split, total_pages);
    const uint k_len   = k_len_per_slot[slot];

    const uint kv_row_stride = H_KV * D;
    for (uint p = p_begin; p < p_end; ++p) {
        const uint phys = bt_s[p];
        simdgroup_half8x8 tmp;
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;
        for (uint s = 0; s < 64; ++s) {
            const uint br = (s / 32) * 8;
            const uint bc = (s % 32) * 8;
            simdgroup_load(tmp, Kbase + br * kv_row_stride + bc, kv_row_stride);
            simdgroup_store(tmp, K_tile + br * D + bc, D);
            simdgroup_load(tmp, Vbase + br * kv_row_stride + bc, kv_row_stride);
            simdgroup_store(tmp, V_tile + br * D + bc, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative QK
        for (uint r = 0; r < PAGE; ++r) {
            float my_partial = 0.0f;
            for (uint di = 0; di < D / 32; ++di) {
                uint d = lid + di * 32;
                my_partial += float(Q_tile[d]) * float(K_tile[r * D + d]);
            }
            float s = simd_sum(my_partial);
            if (lid == 0) {
                uint k_pos = p * PAGE + r;
                scores_tg[r] = (k_pos < k_len) ? s * qk_scale : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float my_score = (lid < PAGE) ? scores_tg[lid] : -INFINITY;
        float page_max = simd_max(my_score);
        float m_old = m_state[0];
        float m_new = max(m_old, page_max);
        float scale = exp(m_old - m_new);
        float my_exp = (lid < PAGE) ? exp(my_score - m_new) : 0.0f;
        float page_sum = simd_sum(my_exp);
        if (lid < PAGE) scores[lid] = my_exp;
        if (lid == 0) { l_state[0] = l_state[0] * scale + page_sum; m_state[0] = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = lid; d < D; d += THREADS) {
            float acc = O_acc[d] * scale;
            for (uint r = 0; r < PAGE; ++r) acc += scores[r] * float(V_tile[r * D + d]);
            O_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials (no final normalization — reduce will combine + normalize)
    uint pidx = vs * N_SPLITS + split;
    const bool empty = (p_begin >= p_end);
    if (lid == 0) {
        m_partials[pidx] = empty ? -INFINITY : m_state[0];
        l_partials[pidx] = empty ? 0.0f : l_state[0];
    }
    device float* O_part = O_partials + pidx * D;
    for (uint d = lid; d < D; d += THREADS) O_part[d] = O_acc[d];
}

// Same logic for D=512 PAGE=8 (full-attn layers).
kernel void paged_attn_full_split_compute(
    device const half* Q                    [[buffer(0)]],
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],
    device float* m_partials                [[buffer(4)]],
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],
    device const uint* num_pages_per_slot   [[buffer(7)]],
    device const uint* k_len_per_slot       [[buffer(8)]],
    constant float& qk_scale                [[buffer(9)]],
    constant uint& max_pages                [[buffer(10)]],
    constant uint& H_Q                      [[buffer(11)]],
    constant uint& H_KV                     [[buffer(12)]],
    constant uint& N_SPLITS                 [[buffer(13)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 512;
    constexpr uint PAGE = 8;
    constexpr uint THREADS = 32;
    const uint vs = tg_pos.x;
    const uint split = tg_pos.y;
    const uint slot = vs / H_Q;
    const uint q_head = vs % H_Q;
    const uint kv_head = (q_head * H_KV) / H_Q;
    const uint lid = lid3.x;

    threadgroup half Q_tile[D];
    threadgroup half K_tile[PAGE * D];
    threadgroup half V_tile[PAGE * D];
    threadgroup float scores[PAGE];
    threadgroup float m_state[1];
    threadgroup float l_state[1];
    threadgroup float O_acc[D];
    threadgroup float scores_tg[PAGE];

    device const half* Q_s = Q + (slot * H_Q + q_head) * D;
    device const uint* bt_s = block_table + slot * max_pages;
    for (uint i = lid; i < D; i += THREADS) { Q_tile[i] = Q_s[i]; O_acc[i] = 0.0f; }
    if (lid == 0) { m_state[0] = -INFINITY; l_state[0] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pages = num_pages_per_slot[slot];
    const uint pages_per_split = (total_pages + N_SPLITS - 1) / N_SPLITS;
    const uint p_begin = split * pages_per_split;
    const uint p_end   = min(p_begin + pages_per_split, total_pages);
    const uint k_len   = k_len_per_slot[slot];

    const uint kv_row_stride = H_KV * D;
    for (uint p = p_begin; p < p_end; ++p) {
        const uint phys = bt_s[p];
        simdgroup_half8x8 tmp;
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;
        for (uint s = 0; s < 64; ++s) {
            const uint br = 0;
            const uint bc = s * 8;
            simdgroup_load(tmp, Kbase + br * kv_row_stride + bc, kv_row_stride);
            simdgroup_store(tmp, K_tile + br * D + bc, D);
            simdgroup_load(tmp, Vbase + br * kv_row_stride + bc, kv_row_stride);
            simdgroup_store(tmp, V_tile + br * D + bc, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < PAGE; ++r) {
            float my_partial = 0.0f;
            for (uint di = 0; di < D / 32; ++di) {
                uint d = lid + di * 32;
                my_partial += float(Q_tile[d]) * float(K_tile[r * D + d]);
            }
            float s = simd_sum(my_partial);
            if (lid == 0) {
                uint k_pos = p * PAGE + r;
                scores_tg[r] = (k_pos < k_len) ? s * qk_scale : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float my_score = (lid < PAGE) ? scores_tg[lid] : -INFINITY;
        float page_max = simd_max(my_score);
        float m_old = m_state[0];
        float m_new = max(m_old, page_max);
        float scale = exp(m_old - m_new);
        float my_exp = (lid < PAGE) ? exp(my_score - m_new) : 0.0f;
        float page_sum = simd_sum(my_exp);
        if (lid < PAGE) scores[lid] = my_exp;
        if (lid == 0) { l_state[0] = l_state[0] * scale + page_sum; m_state[0] = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = lid; d < D; d += THREADS) {
            float acc = O_acc[d] * scale;
            for (uint r = 0; r < PAGE; ++r) acc += scores[r] * float(V_tile[r * D + d]);
            O_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint pidx = vs * N_SPLITS + split;
    const bool empty = (p_begin >= p_end);
    if (lid == 0) {
        m_partials[pidx] = empty ? -INFINITY : m_state[0];
        l_partials[pidx] = empty ? 0.0f : l_state[0];
    }
    device float* O_part = O_partials + pidx * D;
    for (uint d = lid; d < D; d += THREADS) O_part[d] = O_acc[d];
}

// Reduce across N_SPLITS per (slot, q_head): Flash-associative merge.
// Shared between sliding (D=256) and full (D=512) via runtime D param.
kernel void paged_attn_split_reduce(
    device const float* m_partials          [[buffer(0)]],
    device const float* l_partials          [[buffer(1)]],
    device const float* O_partials          [[buffer(2)]],
    device half* O                          [[buffer(3)]],
    constant uint& D                        [[buffer(4)]],
    constant uint& N_SPLITS                 [[buffer(5)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    const uint vs = tg_pos.x;
    const uint lid = lid3.x;

    float my_m = -INFINITY, my_l = 0.0f;
    if (lid < N_SPLITS) {
        my_m = m_partials[vs * N_SPLITS + lid];
        my_l = l_partials[vs * N_SPLITS + lid];
    }
    float m_global = simd_max(my_m);
    float my_scale = (my_m == -INFINITY) ? 0.0f : exp(my_m - m_global);
    // Guard against my_l being NaN/Inf from a split that saw no kept K —
    // the per-TG kernel now zeroes in that case, but belt-and-suspenders
    // so a stale partial can't poison l_global via 0*NaN = NaN.
    float my_contrib = (my_m == -INFINITY) ? 0.0f : (my_scale * my_l);
    float l_global = simd_sum(my_contrib);

    threadgroup float scales_tg[32];
    if (lid < N_SPLITS) scales_tg[lid] = my_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // If no split produced any kept K for this (slot, q_pos, q_head) — all
    // partials were -INF — output zero rather than NaN (0/0). Won't happen
    // with a valid causal mask + self-Q, but the guard costs nothing.
    const bool all_empty = (l_global == 0.0f);
    for (uint d = lid; d < D; d += 32) {
        float acc = 0.0f;
        for (uint s = 0; s < N_SPLITS; ++s) {
            acc += scales_tg[s] * O_partials[(vs * N_SPLITS + s) * D + d];
        }
        O[vs * D + d] = all_empty ? half(0) : half(acc / l_global);
    }
}

// Real paged decode attention — OPTIMIZED: all 32 lanes cooperate on each score
// via simd_sum reduction (instead of 16 lanes each doing scalar D-loop). Also
// avoids tg-mem bank conflicts that the lane-per-row pattern triggers at D=256.
//
// Trade-off vs original: simd_sum per score × PAGE=16 scores = 16 simd_sums
// per page. simd_sum is ~10 cycles. So 160 cycles of reduction overhead per
// page, but the QK compute phase drops from 256-cycle scalar chain to
// 8-cycle per-lane with parallel reduction.
//
// Real paged decode attention, D=256 PAGE=16 (sliding). Flash online softmax.
// Grid: (B × H_Q) TGs. Each TG handles one (slot, q_head); kv_head derived
// from q_head via integer divide for GQA. Per TG tg-mem:
//   Q_tile[256] = 512 B, K_tile[16*256]=8KB, V_tile[16*256]=8KB, scores/m/l = small
// Total ~17 KB per TG — fits under 32 KB limit.
kernel void paged_attn_slide(
    device const half* Q                    [[buffer(0)]],   // [B, H_Q, 256]
    device const half* K_cache              [[buffer(1)]],   // [pages, 16, H_KV, 256]
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],   // [B, max_pages]
    device half* O                           [[buffer(4)]],   // [B, H_Q, 256]
    device const uint* num_pages_per_slot   [[buffer(5)]],
    device const uint* k_len_per_slot       [[buffer(6)]],
    constant float& qk_scale                [[buffer(7)]],
    constant uint& max_pages                [[buffer(8)]],
    constant uint& H_Q                      [[buffer(9)]],
    constant uint& H_KV                     [[buffer(10)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 256;
    constexpr uint PAGE = 16;
    constexpr uint THREADS = 32;
    const uint vs = tg_pos.x;
    const uint slot = vs / H_Q;
    const uint q_head = vs % H_Q;
    const uint kv_head = (q_head * H_KV) / H_Q;
    const uint lid = lid3.x;

    threadgroup half Q_tile[D];
    threadgroup half K_tile[PAGE * D];
    threadgroup half V_tile[PAGE * D];
    threadgroup float scores[PAGE];
    threadgroup float m_state[1];
    threadgroup float l_state[1];
    threadgroup float O_acc[D];

    device const half* Q_s = Q + (slot * H_Q + q_head) * D;
    device half* O_s = O + (slot * H_Q + q_head) * D;
    device const uint* bt_s = block_table + slot * max_pages;

    for (uint i = lid; i < D; i += THREADS) { Q_tile[i] = Q_s[i]; O_acc[i] = 0.0f; }
    if (lid == 0) { m_state[0] = -INFINITY; l_state[0] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_pages = num_pages_per_slot[slot];
    const uint k_len = k_len_per_slot[slot];

    threadgroup float scores_tg[PAGE];

    for (uint p = 0; p < num_pages; ++p) {
        const uint phys = bt_s[p];
        // simdgroup_load: 8×8 half blocks direct from DRAM into tg-mem,
        // fewer instructions than 4096 scalar loads. K_tile is 16×256,
        // V_tile same: 2×32 = 64 8×8 blocks each.
        simdgroup_half8x8 tmp;
        const uint kv_row_stride = H_KV * D;   // stride in halves between KV rows in cache
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;
        for (uint s = 0; s < 64; ++s) {
            const uint br = (s / 32) * 8;    // 0 or 8
            const uint bc = (s % 32) * 8;    // 0..248
            simdgroup_load(tmp, Kbase + br * kv_row_stride + bc, kv_row_stride);
            simdgroup_store(tmp, K_tile + br * D + bc, D);
            simdgroup_load(tmp, Vbase + br * kv_row_stride + bc, kv_row_stride);
            simdgroup_store(tmp, V_tile + br * D + bc, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // COOPERATIVE QK: all 32 lanes compute each score in parallel.
        for (uint r = 0; r < PAGE; ++r) {
            float my_partial = 0.0f;
            for (uint di = 0; di < D / 32; ++di) {
                uint d = lid + di * 32;
                my_partial += float(Q_tile[d]) * float(K_tile[r * D + d]);
            }
            float s = simd_sum(my_partial);
            if (lid == 0) {
                uint k_pos = p * PAGE + r;
                scores_tg[r] = (k_pos < k_len) ? s * qk_scale : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float my_score = (lid < PAGE) ? scores_tg[lid] : -INFINITY;
        float page_max = simd_max(my_score);
        float m_old = m_state[0];
        float m_new = max(m_old, page_max);
        float scale = exp(m_old - m_new);
        float my_exp = (lid < PAGE) ? exp(my_score - m_new) : 0.0f;
        float page_sum = simd_sum(my_exp);
        if (lid < PAGE) scores[lid] = my_exp;
        if (lid == 0) { l_state[0] = l_state[0] * scale + page_sum; m_state[0] = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = lid; d < D; d += THREADS) {
            float acc = O_acc[d] * scale;
            for (uint r = 0; r < PAGE; ++r) acc += scores[r] * float(V_tile[r * D + d]);
            O_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float l_final = l_state[0];
    for (uint d = lid; d < D; d += THREADS) O_s[d] = half(O_acc[d] / l_final);
}

// Real paged decode attention, D=512 PAGE=8 (full-attn). Same structure but
// smaller PAGE to fit tg-mem budget at larger head_dim.
kernel void paged_attn_full(
    device const half* Q                    [[buffer(0)]],
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],
    device half* O                           [[buffer(4)]],
    device const uint* num_pages_per_slot   [[buffer(5)]],
    device const uint* k_len_per_slot       [[buffer(6)]],
    constant float& qk_scale                [[buffer(7)]],
    constant uint& max_pages                [[buffer(8)]],
    constant uint& H_Q                      [[buffer(9)]],
    constant uint& H_KV                     [[buffer(10)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 512;
    constexpr uint PAGE = 8;
    constexpr uint THREADS = 32;
    const uint vs = tg_pos.x;
    const uint slot = vs / H_Q;
    const uint q_head = vs % H_Q;
    const uint kv_head = (q_head * H_KV) / H_Q;
    const uint lid = lid3.x;

    threadgroup half Q_tile[D];
    threadgroup half K_tile[PAGE * D];
    threadgroup half V_tile[PAGE * D];
    threadgroup float scores[PAGE];
    threadgroup float m_state[1];
    threadgroup float l_state[1];
    threadgroup float O_acc[D];

    device const half* Q_s = Q + (slot * H_Q + q_head) * D;
    device half* O_s = O + (slot * H_Q + q_head) * D;
    device const uint* bt_s = block_table + slot * max_pages;

    for (uint i = lid; i < D; i += THREADS) { Q_tile[i] = Q_s[i]; O_acc[i] = 0.0f; }
    if (lid == 0) { m_state[0] = -INFINITY; l_state[0] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_pages = num_pages_per_slot[slot];
    const uint k_len = k_len_per_slot[slot];

    threadgroup float scores_tg[PAGE];

    for (uint p = 0; p < num_pages; ++p) {
        const uint phys = bt_s[p];
        // simdgroup_load for K/V tiles. PAGE=8 × D=512 = 1×64 blocks of 8×8.
        simdgroup_half8x8 tmp;
        const uint kv_row_stride = H_KV * D;
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;
        for (uint s = 0; s < 64; ++s) {
            const uint br = 0;                // single 8-row band (PAGE=8)
            const uint bc = s * 8;            // 0..504
            simdgroup_load(tmp, Kbase + br * kv_row_stride + bc, kv_row_stride);
            simdgroup_store(tmp, K_tile + br * D + bc, D);
            simdgroup_load(tmp, Vbase + br * kv_row_stride + bc, kv_row_stride);
            simdgroup_store(tmp, V_tile + br * D + bc, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative QK (all 32 lanes per score)
        for (uint r = 0; r < PAGE; ++r) {
            float my_partial = 0.0f;
            for (uint di = 0; di < D / 32; ++di) {
                uint d = lid + di * 32;
                my_partial += float(Q_tile[d]) * float(K_tile[r * D + d]);
            }
            float s = simd_sum(my_partial);
            if (lid == 0) {
                uint k_pos = p * PAGE + r;
                scores_tg[r] = (k_pos < k_len) ? s * qk_scale : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float my_score = (lid < PAGE) ? scores_tg[lid] : -INFINITY;
        float page_max = simd_max(my_score);
        float m_old = m_state[0];
        float m_new = max(m_old, page_max);
        float scale = exp(m_old - m_new);
        float my_exp = (lid < PAGE) ? exp(my_score - m_new) : 0.0f;
        float page_sum = simd_sum(my_exp);
        if (lid < PAGE) scores[lid] = my_exp;
        if (lid == 0) { l_state[0] = l_state[0] * scale + page_sum; m_state[0] = m_new; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = lid; d < D; d += THREADS) {
            float acc = O_acc[d] * scale;
            for (uint r = 0; r < PAGE; ++r) acc += scores[r] * float(V_tile[r * D + d]);
            O_acc[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float l_final = l_state[0];
    for (uint d = lid; d < D; d += THREADS) O_s[d] = half(O_acc[d] / l_final);
}

// Attention BW-model placeholder. Real paged attention (paged_attention.swift)
// does Flash online-softmax with simdgroup_load; measured at ~80-500 μs per
// layer at batch=4 K_len=1024. This simpler model reads K+V once each
// (representative of the Flash pattern's amortized BW) and writes O. NOT a
// correct attention — just a memory-BW cost proxy for the forward-graph sim.
// For a real forward, swap this for paged_attention's decode_attn_batched.
kernel void fake_attention(
    device const half* Q [[buffer(0)]],
    device half* O [[buffer(1)]],
    device const half* K_cache [[buffer(2)]],
    device const half* V_cache [[buffer(3)]],
    device const uint* block_table [[buffer(4)]],
    constant uint& num_pages [[buffer(5)]],
    constant uint& H [[buffer(6)]], constant uint& D [[buffer(7)]], constant uint& PAGE [[buffer(8)]],
    constant uint& max_pages [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint h = tg.y; uint t = lid.x;
    device const half* q = Q + (b * H + h) * D;
    device half* o = O + (b * H + h) * D;
    // Single-pass scan: read all K+V pages in big linear sweep, accumulate.
    // Models the DRAM BW an optimized Flash attention would actually pull.
    float acc = 0.0f;
    for (uint p = 0; p < num_pages; ++p) {
        uint phys = block_table[b * max_pages + p];
        device const half* Kp = K_cache + (phys * PAGE * H + h) * D;
        device const half* Vp = V_cache + (phys * PAGE * H + h) * D;
        // Each lane touches D/32 elements × PAGE rows. Read pattern is
        // contiguous across the 16-row page (stride H*D between rows), which
        // is how Flash reads K/V tiles.
        for (uint r = 0; r < PAGE; ++r) {
            for (uint i = t; i < D; i += 32) {
                acc += float(Kp[r * H * D + i]) + float(Vp[r * H * D + i]);
            }
        }
    }
    // Write result (one value per lane's slice of O)
    for (uint i = t; i < D; i += 32) {
        o[i] = half(float(q[i]) + acc * 0.0001f);
    }
}

// Softmax + top-k=8 + renormalize + per-expert scale multiply.
// Output gate_weights[b, k] = softmax(logits)[sel_i[k]] / sum * per_expert_scale[sel_i[k]]
kernel void softmax_topk(
    device const half* logits [[buffer(0)]],
    device uint* expert_ids [[buffer(1)]], device float* gate_weights [[buffer(2)]],
    device const float* per_expert_scale [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    constexpr uint E = 128; constexpr uint K = 8;
    uint b = tg.x; uint t = lid.x;
    device const half* lg = logits + b * E;
    float vals[4];
    for (uint j = 0; j < 4; ++j) vals[j] = float(lg[t + j * 32]);
    float my_max = max(max(vals[0], vals[1]), max(vals[2], vals[3]));
    float gm = simd_max(my_max);
    float my_sum = 0.0f;
    for (uint j = 0; j < 4; ++j) { vals[j] = exp(vals[j] - gm); my_sum += vals[j]; }
    float gs = simd_sum(my_sum); float inv = 1.0f / gs;
    for (uint j = 0; j < 4; ++j) vals[j] *= inv;
    threadgroup float p[E];
    for (uint j = 0; j < 4; ++j) p[t + j * 32] = vals[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float sel_v[K]; uint sel_i[K];
        for (uint k = 0; k < K; ++k) {
            float bv = -INFINITY; uint bi = 0;
            for (uint e = 0; e < E; ++e) if (p[e] > bv) { bv = p[e]; bi = e; }
            sel_v[k] = bv; sel_i[k] = bi;
            p[bi] = -INFINITY;
        }
        // Renormalize top-k weights to sum=1, then multiply by per-expert
        // learned scale (ffn_down_exps.scale[E]) — Gemma-4 convention.
        float sum = 0.0f;
        for (uint k = 0; k < K; ++k) sum += sel_v[k];
        float inv_sum = 1.0f / sum;
        for (uint k = 0; k < K; ++k) {
            expert_ids[b * K + k] = sel_i[k];
            gate_weights[b * K + k] = sel_v[k] * inv_sum * per_expert_scale[sel_i[k]];
        }
    }
}

// MoE routing compaction: histogram + prefix sum + scatter.
// Given expert_ids[B*K] from softmax_topk, build:
//   group_start[E+1]   CSR prefix-sum of slot counts per expert (monotone, ascending)
//   slot_token[B*K]    expert-grouped order -> batch index b
//   batch_slots[B*K]   (b,k) order -> slot index in expert-grouped order
// Dispatch: 1 TG, 128 threads. Thread e owns expert e. active_exp stays
// static (identity 0..E-1); empty experts early-return in MoE kernels via
// group_start[e+1]==group_start[e].
kernel void route_compact(
    device const uint* expert_ids   [[buffer(0)]],   // [B*K]
    device uint* group_start        [[buffer(1)]],   // [E+1]
    device uint* slot_token         [[buffer(2)]],   // [B*K]
    device uint* batch_slots        [[buffer(3)]],   // [B*K]
    constant uint& B                [[buffer(4)]],
    constant uint& K                [[buffer(5)]],
    uint2 lid                       [[thread_position_in_threadgroup]])
{
    constexpr uint E = 128;
    threadgroup uint counts[E];
    uint e = lid.x;
    uint BK = B * K;
    if (e >= E) return;

    // Pass 1: count matches for my expert.
    uint myCount = 0;
    for (uint i = 0; i < BK; ++i) {
        if (expert_ids[i] == e) ++myCount;
    }
    counts[e] = myCount;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: thread 0 serializes the E=128 prefix sum into group_start.
    if (e == 0) {
        uint running = 0;
        for (uint i = 0; i < E; ++i) {
            group_start[i] = running;
            running += counts[i];
        }
        group_start[E] = running;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: scatter. Thread e walks (b,k) in row-major order and emits
    // slots for its own expert. Same iteration order as pass 1 -> consistent
    // local offset sequence, so batch_slots is well-defined.
    uint base = group_start[e];
    uint localOff = 0;
    for (uint i = 0; i < BK; ++i) {
        if (expert_ids[i] == e) {
            uint slotIdx = base + localOff;
            uint b = i / K;
            slot_token[slotIdx] = b;
            batch_slots[i] = slotIdx;
            ++localOff;
        }
    }
}

// -------- Q4_K GGUF-native dequant helpers --------
// block_q4_K layout (144 bytes per 256-element super-block):
//   half d;              // 2 B — super-scale for the 8 sub-block scales
//   half dmin;           // 2 B — super-scale for the 8 sub-block mins
//   uchar scales[12];    // 12 B — packed (scale_6b, min_6b) × 8 sub-blocks
//   uchar qs[128];       // 128 B — 256 × 4-bit quants
// Each of 8 sub-blocks covers 32 consecutive elements.
// Scales packing (llama.cpp convention):
//   sub-block sb in [0..3]: scale=scales[sb]&0x3F, min=scales[sb+4]&0x3F
//   sub-block sb in [4..7]: scale=(scales[sb+4]&0xF)|((scales[sb-4]>>2)&0x30),
//                            min=(scales[sb+4]>>4)|((scales[sb]>>2)&0x30)
// Quant indexing for element e in [0..256):
//   sb = e/32; p = e%32;
//   byte = qs[sb*16 + p%16]; nibble = (p<16) ? (byte&0xF) : (byte>>4)
// Dequant: w = d*scale_6b*nibble - dmin*min_6b

static inline void unpack_q4k_scales(device const uchar* s, thread uchar &sc, thread uchar &mn, uint sb) {
    if (sb < 4) {
        sc = s[sb] & 0x3F;
        mn = s[sb + 4] & 0x3F;
    } else {
        uint k = sb - 4;
        sc = (s[k + 8] & 0x0F) | ((s[k + 0] & 0xC0) >> 2);
        mn = (s[k + 8] >> 4)   | ((s[k + 4] & 0xC0) >> 2);
    }
}

// Dequant one element from a Q4_K super-block.
// Canonical llama.cpp layout: sub-blocks are paired (2p, 2p+1); both share the
// 32-byte qs slice `qs[pair*32 : pair*32 + 32]`. The even sub-block takes low
// nibbles, the odd takes high nibbles.
static inline float dequant_q4k_one(device const half* blk_d, device const half* blk_dmin,
                                    device const uchar* scales, device const uchar* qs, uint e) {
    uint sb = e / 32;
    uint p  = e % 32;
    uint pair = sb / 2;
    uint is_hi = sb & 1u;
    uchar byte = qs[pair * 32 + p];
    uint nib = is_hi ? ((byte >> 4) & 0xF) : (byte & 0xF);
    uchar sc, mn;
    unpack_q4k_scales(scales, sc, mn, sb);
    float d = float(*blk_d);
    float dmin = float(*blk_dmin);
    return d * float(sc) * float(nib) - dmin * float(mn);
}

// MoE GEMV with Q4_K weights. Layout: W_q4k[E * (D_out * D_in/256) * 144 bytes].
// For expert e, column n, k-block kb: block offset = (e*D_out*D_in/256 + n*D_in/256 + kb) * 144.
// Each thread owns one output column (like simple-Q4 kernel), but per k-block
// the thread reads ONE block (144 bytes) and dequantizes 256 elements inline.
kernel void moe_gemv_q4k_v3(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const uchar* W_q4k           [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    // Bytes per super-block: 144 = 2 + 2 + 12 + 128
    constexpr uint BLK = 144;
    uint n_blocks_per_col = D_in / 256;                         // K super-blocks per column
    uint expert_bytes = D_out * n_blocks_per_col * BLK;         // bytes per expert's W
    uint col_bytes = n_blocks_per_col * BLK;                    // bytes per column
    device const uchar* W_exp = W_q4k + expert * expert_bytes + n * col_bytes;

    for (uint slot = gb; slot < ge; ++slot) {
        uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;
        float acc = 0.0f;

        for (uint kb = 0; kb < n_blocks_per_col; ++kb) {
            device const uchar* blk = W_exp + kb * BLK;
            device const half*  blk_d    = (device const half*)(blk + 0);
            device const half*  blk_dmin = (device const half*)(blk + 2);
            device const uchar* scales   = blk + 4;
            device const uchar* qs       = blk + 16;
            float d = float(*blk_d);
            float dmin = float(*blk_dmin);

            // Process 8 sub-blocks × 32 elements each
            for (uint sb = 0; sb < 8; ++sb) {
                uchar sc, mn;
                unpack_q4k_scales(scales, sc, mn, sb);
                float dl = d * float(sc);
                float ml = dmin * float(mn);
                // Unroll the 32 elements of this sub-block
                for (uint p = 0; p < 16; ++p) {
                    uchar byte = qs[sb * 16 + p];
                    float w_lo = dl * float(byte & 0xF) - ml;
                    float w_hi = dl * float((byte >> 4) & 0xF) - ml;
                    uint base_k = kb * 256 + sb * 32;
                    acc += float(hid[base_k + p])     * w_lo
                         + float(hid[base_k + p + 16]) * w_hi;
                }
            }
        }
        output[slot * D_out + n] = half(acc);
    }
}

// MoE Q4_K v4: k-outer / slot-inner. Dequantizes each (kb, sb, p) exactly once
// and applies to all slots sharing this expert, amortizing the Q4_K bit-unpack
// math when multiple tokens route to the same expert. At 1-slot-per-expert
// the compiler should compile down to v3's inner loop with a slight hoist
// benefit; at 2-4 slots the savings are linear in n_slots.
kernel void moe_gemv_q4k_v4(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const uchar* W_q4k           [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint MAX_SLOTS = 4;                    // B=4 top-8 upper bound
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    uint n_slots = ge - gb;
    if (n_slots == 0) return;

    constexpr uint BLK = 144;
    uint n_blocks_per_col = D_in / 256;
    uint expert_bytes = D_out * n_blocks_per_col * BLK;
    uint col_bytes = n_blocks_per_col * BLK;
    device const uchar* W_exp = W_q4k + expert * expert_bytes + n * col_bytes;

    device const half* hid_slots[MAX_SLOTS];
    for (uint s = 0; s < n_slots; ++s) {
        hid_slots[s] = hidden + slot_token[gb + s] * D_in;
    }
    float accs[MAX_SLOTS] = {0};

    for (uint kb = 0; kb < n_blocks_per_col; ++kb) {
        device const uchar* blk = W_exp + kb * BLK;
        float d    = float(*(device const half*)(blk + 0));
        float dmin = float(*(device const half*)(blk + 2));
        device const uchar* scales = blk + 4;
        device const uchar* qs     = blk + 16;

        for (uint sb = 0; sb < 8; ++sb) {
            uchar sc, mn;
            unpack_q4k_scales(scales, sc, mn, sb);
            float dl = d * float(sc);
            float ml = dmin * float(mn);
            uint base_k = kb * 256 + sb * 32;
            for (uint p = 0; p < 16; ++p) {
                uchar byte = qs[sb * 16 + p];
                float w_lo = dl * float(byte & 0xF) - ml;
                float w_hi = dl * float((byte >> 4) & 0xF) - ml;
                for (uint s = 0; s < n_slots; ++s) {
                    accs[s] += float(hid_slots[s][base_k + p])      * w_lo
                             + float(hid_slots[s][base_k + p + 16]) * w_hi;
                }
            }
        }
    }

    for (uint s = 0; s < n_slots; ++s) {
        output[(gb + s) * D_out + n] = half(accs[s]);
    }
}

// MoE Q5_1 v4: same k-outer / slot-inner structure as Q4_K v4. Q5_1 dequant
// (5-bit from split nibble+high-bit) is more expensive per element, so slot
// amortization helps more when sharing is present.
kernel void moe_gemv_q5_1_v4(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const uchar* W_q51           [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint MAX_SLOTS = 4;
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    uint n_slots = ge - gb;
    if (n_slots == 0) return;

    constexpr uint BLK = 24;
    uint nbc = D_in / 32;
    uint expert_bytes = D_out * nbc * BLK;
    uint col_bytes = nbc * BLK;
    device const uchar* W_exp = W_q51 + expert * expert_bytes + n * col_bytes;

    device const half* hid_slots[MAX_SLOTS];
    for (uint s = 0; s < n_slots; ++s) {
        hid_slots[s] = hidden + slot_token[gb + s] * D_in;
    }
    float accs[MAX_SLOTS] = {0};

    for (uint kb = 0; kb < nbc; ++kb) {
        device const uchar* blk = W_exp + kb * BLK;
        float d = float(*(device const half*)(blk));
        float m = float(*(device const half*)(blk + 2));
        device const uchar* qh = blk + 4;
        device const uchar* qs = blk + 8;
        uint base_k = kb * 32;
        for (uint p = 0; p < 16; ++p) {
            uchar qsp = qs[p];
            uint h_lo = (qh[p / 8]       >> (p       % 8)) & 1u;
            uint h_hi = (qh[(p + 16) / 8] >> ((p + 16) % 8)) & 1u;
            uint q_lo = (uint(qsp) & 0xFu)        | (h_lo << 4);
            uint q_hi = ((uint(qsp) >> 4) & 0xFu) | (h_hi << 4);
            float w_lo = d * float(q_lo) + m;
            float w_hi = d * float(q_hi) + m;
            for (uint s = 0; s < n_slots; ++s) {
                accs[s] += float(hid_slots[s][base_k + p])      * w_lo
                         + float(hid_slots[s][base_k + p + 16]) * w_hi;
            }
        }
    }

    for (uint s = 0; s < n_slots; ++s) {
        output[(gb + s) * D_out + n] = half(accs[s]);
    }
}

// Dense GEMV with Q4_K weights, v4 pattern (multi-batch-per-TG amortizes W).
// Each thread owns one output column n, loops k-blocks over all B batch rows.
kernel void dense_gemv_q4k_v4(
    device const half* hidden           [[buffer(0)]],
    device const uchar* W_q4k           [[buffer(1)]],
    device half* output                 [[buffer(2)]],
    constant uint& B                    [[buffer(3)]],
    constant uint& D_in                 [[buffer(4)]],
    constant uint& D_out                [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint MAX_B = 8;
    uint n_block = tg.x; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;

    constexpr uint BLK = 144;
    uint n_blocks_per_col = D_in / 256;
    uint col_bytes = n_blocks_per_col * BLK;
    device const uchar* W_col = W_q4k + n * col_bytes;

    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;

    for (uint kb = 0; kb < n_blocks_per_col; ++kb) {
        device const uchar* blk = W_col + kb * BLK;
        device const half*  blk_d    = (device const half*)(blk + 0);
        device const half*  blk_dmin = (device const half*)(blk + 2);
        device const uchar* scales   = blk + 4;
        device const uchar* qs       = blk + 16;
        float d = float(*blk_d); float dmin = float(*blk_dmin);

        // Precompute 8 (dl, ml) pairs once per block (shared across batch)
        float dl[8], ml[8];
        for (uint sb = 0; sb < 8; ++sb) {
            uchar sc, mn; unpack_q4k_scales(scales, sc, mn, sb);
            dl[sb] = d * float(sc); ml[sb] = dmin * float(mn);
        }

        for (uint sb = 0; sb < 8; ++sb) {
            for (uint p = 0; p < 16; ++p) {
                uchar byte = qs[sb * 16 + p];
                float w_lo = dl[sb] * float(byte & 0xF) - ml[sb];
                float w_hi = dl[sb] * float((byte >> 4) & 0xF) - ml[sb];
                uint base_k = kb * 256 + sb * 32;
                for (uint b = 0; b < B; ++b) {
                    accs[b] += float(hidden[b * D_in + base_k + p])      * w_lo
                             + float(hidden[b * D_in + base_k + p + 16]) * w_hi;
                }
            }
        }
    }
    for (uint b = 0; b < B; ++b) output[b * D_out + n] = half(accs[b]);
}

// -------- Q8_0 GGUF-native dequant (used for most Gemma-4 dense weights) --------
// block_q8_0 (34 bytes per 32-element block):
//   half d;            // 2 B — single scale for the whole block
//   int8 qs[32];       // 32 signed-int8 quantized values
// Dequant: w = d * qs[i]
// Used for Q/K/V/out, shared FFN gate/up/down, token_embd, and router.

kernel void dense_gemv_q8_0_v5(
    device const half* hidden           [[buffer(0)]],
    device const uchar* W_q80           [[buffer(1)]],
    device half* output                 [[buffer(2)]],
    constant uint& D_in                 [[buffer(3)]],
    constant uint& D_out                [[buffer(4)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]])
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 34;
    uint n_block = tg.x; uint b = tg.y; uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    uint nbc = D_in / 32;
    uint col_bytes = nbc * BLK;
    device const uchar* W_col = W_q80 + n * col_bytes;

    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    float acc = 0.0f;
    device const half* hid_b = hidden + b * D_in;
    for (uint kb = kb_begin; kb < kb_end; ++kb) {
        device const uchar* blk = W_col + kb * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        // Inner loop: 32 int8 multiply-adds, scaled by d
        for (uint p = 0; p < 32; ++p) {
            acc += float(hid_b[base_k + p]) * d * float(qs[p]);
        }
    }
    threadgroup float partials[N_SPLITS][32];
    partials[sg_id][lid_sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        float total = partials[0][lid_sg] + partials[1][lid_sg] + partials[2][lid_sg] + partials[3][lid_sg];
        output[b * D_out + n] = half(total);
    }
}

// Q8_0 dense GEMV v6: SWIZZLED layout. The weight buffer for this kernel is
// laid out as [n_super = D_out/32, nbc, 32 cols, 34 bytes] so that 32 threads
// of a simdgroup read 1088 contiguous bytes per kb iteration — fully
// cacheline-coalesced — instead of 32 scattered 34-byte reads at col_bytes
// stride. Expected 2–3× DRAM throughput vs v5 for D_in=2816 projections.
// Caller is responsible for providing the repacked weight buffer (see
// repackQ80 on the Swift side).
kernel void dense_gemv_q8_0_v6(
    device const half* hidden           [[buffer(0)]],
    device const uchar* W_sw            [[buffer(1)]],
    device half* output                 [[buffer(2)]],
    constant uint& D_in                 [[buffer(3)]],
    constant uint& D_out                [[buffer(4)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]])
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 34;
    uint n_block = tg.x; uint b = tg.y; uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    uint nbc = D_in / 32;
    // Super-column base: one "super" is 32 adjacent d_out columns of all kb blocks.
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    float acc = 0.0f;
    device const half* hid_b = hidden + b * D_in;
    for (uint kb = kb_begin; kb < kb_end; ++kb) {
        // In swizzled layout, the 32 adjacent-column blocks for this kb are
        // packed contiguously: offset = kb * 32 * 34 + lid_sg * 34.
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        for (uint p = 0; p < 32; ++p) {
            acc += float(hid_b[base_k + p]) * d * float(qs[p]);
        }
    }
    threadgroup float partials[N_SPLITS][32];
    partials[sg_id][lid_sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        float total = partials[0][lid_sg] + partials[1][lid_sg] + partials[2][lid_sg] + partials[3][lid_sg];
        output[b * D_out + n] = half(total);
    }
}

// Q8_0 v6 + fused RMSNorm. Same pattern as v5_rmsnorm but with swizzled W.
kernel void dense_gemv_q8_0_v6_rmsnorm(
    device const half* x                [[buffer(0)]],
    device const uchar* W_sw            [[buffer(1)]],
    device const half* gamma            [[buffer(2)]],
    device half* output                 [[buffer(3)]],
    constant uint& D_in                 [[buffer(4)]],
    constant uint& D_out                [[buffer(5)]],
    constant float& eps                 [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]])
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 34;
    constexpr uint MAX_D_IN = 2816;
    constexpr uint THREADS = 128;
    threadgroup half h_norm[MAX_D_IN];

    uint n_block = tg.x; uint b = tg.y; uint tid = lid.x; uint lid_sg = tid % 32;

    device const half* xb = x + b * D_in;
    float local_ss = 0.0f;
    for (uint i = tid; i < D_in; i += THREADS) {
        float v = float(xb[i]);
        local_ss += v * v;
    }
    float sg_ss = simd_sum(local_ss);
    threadgroup float ss_stage[N_SPLITS];
    if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
    float inv_rms = rsqrt(total_ss / float(D_in) + eps);

    for (uint i = tid; i < D_in; i += THREADS) {
        h_norm[i] = half(float(xb[i]) * inv_rms * float(gamma[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;
    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    float acc = 0.0f;
    for (uint kb = kb_begin; kb < kb_end; ++kb) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        for (uint p = 0; p < 32; ++p) {
            acc += float(h_norm[base_k + p]) * d * float(qs[p]);
        }
    }
    threadgroup float partials[N_SPLITS][32];
    partials[sg_id][lid_sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        float total = partials[0][lid_sg] + partials[1][lid_sg] + partials[2][lid_sg] + partials[3][lid_sg];
        output[b * D_out + n] = half(total);
    }
}

// Fused: RMSNorm + Q8_0 v6 swizzled GEMV for Q, K, V in a single dispatch.
// Each TG claims one 32-column slab of the concatenated output space
// [Q_slabs | K_slabs | V_slabs] based on tg.x, routes to the corresponding
// weight buffer, and writes to the corresponding output buffer. RMS and
// h_norm tg-mem staging happen once per (TG, batch) instead of three times.
kernel void dense_gemv_q8_0_v6_rmsnorm_qkv(
    device const half* x                [[buffer(0)]],
    device const half* gamma            [[buffer(1)]],
    device const uchar* Wq_sw           [[buffer(2)]],
    device const uchar* Wk_sw           [[buffer(3)]],
    device const uchar* Wv_sw           [[buffer(4)]],
    device half* out_q                  [[buffer(5)]],
    device half* out_k                  [[buffer(6)]],
    device half* out_v                  [[buffer(7)]],
    constant uint& D_in                 [[buffer(8)]],
    constant uint& Q_nb                 [[buffer(9)]],     // D_out_q / 32
    constant uint& K_nb                 [[buffer(10)]],    // D_out_k / 32
    constant uint& V_nb                 [[buffer(11)]],    // D_out_v / 32
    constant uint& D_out_q              [[buffer(12)]],
    constant uint& D_out_k              [[buffer(13)]],
    constant uint& D_out_v              [[buffer(14)]],
    constant float& eps                 [[buffer(15)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]])
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 34;
    constexpr uint MAX_D_IN = 2816;
    constexpr uint THREADS = 128;
    threadgroup half h_norm[MAX_D_IN];

    uint slab = tg.x; uint b = tg.y; uint tid = lid.x; uint lid_sg = tid % 32;

    // === RMS reduction + h_norm staging (shared by Q/K/V for this b) ===
    device const half* xb = x + b * D_in;
    float local_ss = 0.0f;
    for (uint i = tid; i < D_in; i += THREADS) {
        float v = float(xb[i]);
        local_ss += v * v;
    }
    float sg_ss = simd_sum(local_ss);
    threadgroup float ss_stage[N_SPLITS];
    if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
    float inv_rms = rsqrt(total_ss / float(D_in) + eps);
    for (uint i = tid; i < D_in; i += THREADS) {
        h_norm[i] = half(float(xb[i]) * inv_rms * float(gamma[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Dispatch slab to the right projection ===
    uint n_block;
    uint D_out;
    device const uchar* W_sw;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab;
        D_out = D_out_q;
        W_sw = Wq_sw;
        out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb;
        D_out = D_out_k;
        W_sw = Wk_sw;
        out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb;
        D_out = D_out_v;
        W_sw = Wv_sw;
        out = out_v;
    }

    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;
    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    float acc = 0.0f;
    for (uint kb = kb_begin; kb < kb_end; ++kb) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        for (uint p = 0; p < 32; ++p) {
            acc += float(h_norm[base_k + p]) * d * float(qs[p]);
        }
    }
    threadgroup float partials[N_SPLITS][32];
    partials[sg_id][lid_sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        float total = partials[0][lid_sg] + partials[1][lid_sg] + partials[2][lid_sg] + partials[3][lid_sg];
        out[b * D_out + n] = half(total);
    }
}

// Fused: RMSNorm + Q8_0 v6 for [gate, up] dense shared-FFN pair in a single
// dispatch. Output layout is [B, 2*D_out] with first D_out halves = gate,
// second D_out halves = up — identical to the MoE fused layout, so a
// gelu_mul kernel written for that layout handles this too.
kernel void dense_gemv_q8_0_v6_rmsnorm_gate_up(
    device const half* x                [[buffer(0)]],
    device const half* gamma            [[buffer(1)]],
    device const uchar* Wg_sw           [[buffer(2)]],
    device const uchar* Wu_sw           [[buffer(3)]],
    device half* fused_out              [[buffer(4)]],
    constant uint& D_in                 [[buffer(5)]],
    constant uint& D_out                [[buffer(6)]],
    constant float& eps                 [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]])
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 34;
    constexpr uint MAX_D_IN = 2816;
    constexpr uint THREADS = 128;
    threadgroup half h_norm[MAX_D_IN];

    uint slab = tg.x; uint b = tg.y; uint tid = lid.x; uint lid_sg = tid % 32;
    uint n_blocks_per_branch = D_out / 32;
    bool is_up = slab >= n_blocks_per_branch;
    uint n_block = is_up ? slab - n_blocks_per_branch : slab;

    device const half* xb = x + b * D_in;
    float local_ss = 0.0f;
    for (uint i = tid; i < D_in; i += THREADS) {
        float v = float(xb[i]);
        local_ss += v * v;
    }
    float sg_ss = simd_sum(local_ss);
    threadgroup float ss_stage[N_SPLITS];
    if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
    float inv_rms = rsqrt(total_ss / float(D_in) + eps);
    for (uint i = tid; i < D_in; i += THREADS) {
        h_norm[i] = half(float(xb[i]) * inv_rms * float(gamma[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_sw = is_up ? Wu_sw : Wg_sw;
    device const uchar* W_super = W_sw + n_block * super_bytes;
    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    float acc = 0.0f;
    for (uint kb = kb_begin; kb < kb_end; ++kb) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        for (uint p = 0; p < 32; ++p) {
            acc += float(h_norm[base_k + p]) * d * float(qs[p]);
        }
    }
    threadgroup float partials[N_SPLITS][32];
    partials[sg_id][lid_sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        float total = partials[0][lid_sg] + partials[1][lid_sg] + partials[2][lid_sg] + partials[3][lid_sg];
        uint col = is_up ? (D_out + n) : n;
        fused_out[b * 2 * D_out + col] = half(total);
    }
}

// Fused: RMSNorm(x, gamma) → Q8_0 dense GEMV v5. Reads x once from DRAM,
// normalizes in-TG-memory, then projects. Eliminates the separate RMSNorm
// dispatch and its round-trip write of the scaled activation. Usable any
// time a GEMV immediately follows RMSNorm with the same D_in as hidden_dim.
// tg-mem scratch is sized for HIDDEN=2816 halves = 5632 B.
kernel void dense_gemv_q8_0_v5_rmsnorm(
    device const half* x                [[buffer(0)]],
    device const uchar* W_q80           [[buffer(1)]],
    device const half* gamma            [[buffer(2)]],
    device half* output                 [[buffer(3)]],
    constant uint& D_in                 [[buffer(4)]],
    constant uint& D_out                [[buffer(5)]],
    constant float& eps                 [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]])
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 34;
    constexpr uint MAX_D_IN = 2816;           // HIDDEN cap for Gemma-4
    constexpr uint THREADS = 128;             // 4 SGs × 32
    threadgroup half h_norm[MAX_D_IN];        // gamma-scaled, normalized x for this batch

    uint n_block = tg.x; uint b = tg.y; uint tid = lid.x; uint lid_sg = tid % 32;

    // === Phase 1: per-TG RMS reduction over x[b, :] ===
    device const half* xb = x + b * D_in;
    float local_ss = 0.0f;
    for (uint i = tid; i < D_in; i += THREADS) {
        float v = float(xb[i]);
        local_ss += v * v;
    }
    float sg_ss = simd_sum(local_ss);
    threadgroup float ss_stage[N_SPLITS];
    if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
    float inv_rms = rsqrt(total_ss / float(D_in) + eps);

    // === Phase 2: stage gamma-scaled normalized x into tg-mem ===
    for (uint i = tid; i < D_in; i += THREADS) {
        h_norm[i] = half(float(xb[i]) * inv_rms * float(gamma[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Phase 3: v5 split-K GEMV over h_norm ===
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    uint col_bytes = nbc * BLK;
    device const uchar* W_col = W_q80 + n * col_bytes;
    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    float acc = 0.0f;
    for (uint kb = kb_begin; kb < kb_end; ++kb) {
        device const uchar* blk = W_col + kb * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        for (uint p = 0; p < 32; ++p) {
            acc += float(h_norm[base_k + p]) * d * float(qs[p]);
        }
    }
    threadgroup float partials[N_SPLITS][32];
    partials[sg_id][lid_sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        float total = partials[0][lid_sg] + partials[1][lid_sg] + partials[2][lid_sg] + partials[3][lid_sg];
        output[b * D_out + n] = half(total);
    }
}

// Q8_0 dense GEMV with multi-batch-per-TG amortization (for unembed at D_out=262144).
kernel void dense_gemv_q8_0_v4(
    device const half* hidden           [[buffer(0)]],
    device const uchar* W_q80           [[buffer(1)]],
    device half* output                 [[buffer(2)]],
    constant uint& B                    [[buffer(3)]],
    constant uint& D_in                 [[buffer(4)]],
    constant uint& D_out                [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint MAX_B = 8;
    constexpr uint BLK = 34;
    uint n_block = tg.x; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;

    uint nbc = D_in / 32;
    uint col_bytes = nbc * BLK;
    device const uchar* W_col = W_q80 + n * col_bytes;

    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;

    for (uint kb = 0; kb < nbc; ++kb) {
        device const uchar* blk = W_col + kb * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        for (uint p = 0; p < 32; ++p) {
            float w = d * float(qs[p]);
            for (uint b = 0; b < B; ++b) {
                accs[b] += float(hidden[b * D_in + base_k + p]) * w;
            }
        }
    }
    for (uint b = 0; b < B; ++b) output[b * D_out + n] = half(accs[b]);
}

// -------- Q5_1 GGUF-native dequant (used for Gemma-4 MoE down projection) --------
// block_q5_1 (24 bytes per 32-element block):
//   half d;            // 2 B — scale
//   half m;            // 2 B — offset (min)
//   uchar qh[4];       // 4 B — high bit of each 5-bit quant (bit i of qh[i/8] = 5th bit of q[i])
//   uchar qs[16];      // 16 B — low 4 bits of each quant, packed 2 per byte
// Element e (in [0..31]) uses:
//   lo_nibble = (e < 16) ? (qs[e] & 0xF) : (qs[e-16] >> 4)
//   h5 = (qh[e/8] >> (e%8)) & 1
//   q5 = lo_nibble | (h5 << 4)            // 5-bit unsigned value 0..31
//   w  = d * q5 + m
kernel void moe_gemv_q5_1_v3(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const uchar* W_q51           [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 24;
    uint nbc = D_in / 32;
    uint expert_bytes = D_out * nbc * BLK;
    uint col_bytes = nbc * BLK;
    device const uchar* W_exp = W_q51 + expert * expert_bytes + n * col_bytes;

    for (uint slot = gb; slot < ge; ++slot) {
        uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;
        float acc = 0.0f;
        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_exp + kb * BLK;
            float d = float(*(device const half*)(blk));
            float m = float(*(device const half*)(blk + 2));
            device const uchar* qh = blk + 4;
            device const uchar* qs = blk + 8;
            uint base_k = kb * 32;
            for (uint p = 0; p < 16; ++p) {
                uchar qsp = qs[p];
                uint h_lo = (qh[p / 8]       >> (p       % 8)) & 1u;
                uint h_hi = (qh[(p + 16) / 8] >> ((p + 16) % 8)) & 1u;
                uint q_lo = (uint(qsp) & 0xFu)       | (h_lo << 4);
                uint q_hi = ((uint(qsp) >> 4) & 0xFu) | (h_hi << 4);
                float w_lo = d * float(q_lo) + m;
                float w_hi = d * float(q_hi) + m;
                acc += float(hid[base_k + p])      * w_lo
                     + float(hid[base_k + p + 16]) * w_hi;
            }
        }
        output[slot * D_out + n] = half(acc);
    }
}

// MoE Q4_K GEMV v6: swizzled [expert, n_super=D_out/32, nbc=D_in/256, 32 cols, 144 bytes].
// 32 threads of an SG read 4608 contiguous bytes per kb iter — cache-line coalesced.
kernel void moe_gemv_q4k_v6(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const uchar* W_sw            [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 144;
    uint nbc = D_in / 256;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    for (uint slot = gb; slot < ge; ++slot) {
        uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;
        float acc = 0.0f;
        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d    = float(*(device const half*)(blk + 0));
            float dmin = float(*(device const half*)(blk + 2));
            device const uchar* scales = blk + 4;
            device const uchar* qs     = blk + 16;
            // Q4_K sub-blocks are paired: (sb_lo, sb_hi) = (2·p, 2·p+1) share
            // qs[p·32 : (p+1)·32]. Low nibbles → sb_lo elements, high → sb_hi.
            for (uint pair = 0; pair < 4; ++pair) {
                uint sb_lo = pair * 2;
                uint sb_hi = pair * 2 + 1;
                uchar sc_lo, mn_lo, sc_hi, mn_hi;
                unpack_q4k_scales(scales, sc_lo, mn_lo, sb_lo);
                unpack_q4k_scales(scales, sc_hi, mn_hi, sb_hi);
                float dl_lo = d * float(sc_lo), ml_lo = dmin * float(mn_lo);
                float dl_hi = d * float(sc_hi), ml_hi = dmin * float(mn_hi);
                uint base_lo = kb * 256 + sb_lo * 32;
                uint base_hi = kb * 256 + sb_hi * 32;
                for (uint p = 0; p < 32; ++p) {
                    uchar byte = qs[pair * 32 + p];
                    float w_lo = dl_lo * float(byte & 0xF)        - ml_lo;
                    float w_hi = dl_hi * float((byte >> 4) & 0xF) - ml_hi;
                    acc += float(hid[base_lo + p]) * w_lo
                         + float(hid[base_hi + p]) * w_hi;
                }
            }
        }
        output[slot * D_out + n] = half(acc);
    }
}

// MoE Q5_1 GEMV v6: swizzled layout [expert, n_super=D_out/32, nbc, 32 cols, 24 bytes].
// 32 threads of an SG read 768 contiguous bytes per kb iter — no col_bytes-stride
// scatter. Biggest remaining MoE DRAM lever (Q5_1 down projection).
kernel void moe_gemv_q5_1_v6(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const uchar* W_sw            [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 24;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    for (uint slot = gb; slot < ge; ++slot) {
        // Q5_1 is the MoE DOWN projection: its input `hidden` is per-slot
        // (already expanded through gate/up in earlier stages), layout
        // [TOTAL_SLOTS, D_in]. Use `slot * D_in`, NOT slot_token[slot] * D_in
        // (that's the per-batch index, which only makes sense for the first
        // MoE GEMV, where we fan out from hidden_norm[B, HIDDEN]).
        device const half* hid = hidden + slot * D_in;
        float acc = 0.0f;
        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d = float(*(device const half*)(blk));
            float m = float(*(device const half*)(blk + 2));
            device const uchar* qh = blk + 4;
            device const uchar* qs = blk + 8;
            uint base_k = kb * 32;
            for (uint p = 0; p < 16; ++p) {
                uchar qsp = qs[p];
                uint h_lo = (qh[p / 8]       >> (p       % 8)) & 1u;
                uint h_hi = (qh[(p + 16) / 8] >> ((p + 16) % 8)) & 1u;
                uint q_lo = (uint(qsp) & 0xFu)        | (h_lo << 4);
                uint q_hi = ((uint(qsp) >> 4) & 0xFu) | (h_hi << 4);
                float w_lo = d * float(q_lo) + m;
                float w_hi = d * float(q_hi) + m;
                acc += float(hid[base_k + p])      * w_lo
                     + float(hid[base_k + p + 16]) * w_hi;
            }
        }
        output[slot * D_out + n] = half(acc);
    }
}

// -------- Fused gate+up activation --------
// Reads a [slots, 2*N_half] tensor where first half is gate_proj and second
// half is up_proj (matches Gemma-4's ffn_gate_up_exps layout after matmul).
// Applies gelu(gate) * up → [slots, N_half].
kernel void moe_gelu_mul_fused(
    device const half* fused            [[buffer(0)]],
    device half* out                    [[buffer(1)]],
    constant uint& N_half               [[buffer(2)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    const float c = 0.7978845608f;
    uint stride = N_half * 2;
    // Layout: fused[b, 0..N_half) = gate, fused[b, N_half..2*N_half) = up.
    // Confirmed by KL-harness empirical test — swapping gate/up worsens
    // divergence, so this order matches the GGUF-of-Gemma-4 convention.
    for (uint i = t; i < N_half; i += 32) {
        float g = float(fused[b * stride + i]);
        float u = float(fused[b * stride + N_half + i]);
        float inner = c * (g + 0.044715f * g * g * g);
        // Metal's tanh(x) naively computes (exp(x)-exp(-x))/(exp(x)+exp(-x))
        // and returns NaN when |x| is large enough that exp overflows.
        // Clamp inner so tanh saturates cleanly to ±1.
        inner = clamp(inner, -20.0f, 20.0f);
        float gelu_g = 0.5f * g * (1.0f + tanh(inner));
        out[b * N_half + i] = half(gelu_g * u);
    }
}

// -------- Q4_0 GGUF-native dequant (for tensors with K not divisible by 256) --------
// block_q4_0 layout (18 bytes per 32-element block):
//   half d;            // 2 B — single scale for the whole block
//   uchar qs[16];      // 16 B — 32 × 4-bit quants
// Nibble packing: element e in [0..31]:
//   byte = qs[e % 16]
//   nibble = (e < 16) ? (byte & 0xF) : (byte >> 4)
// Dequant: w = d * (nibble - 8)   [centered around 0]
// Gemma-4 uses this for tensors with K % 256 != 0: MoE down (K=704),
// shared FFN down (K=2112), and likely any mis-aligned projections.

kernel void moe_gemv_q4_0_v3(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const uchar* W_q40           [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 18;
    uint n_blocks_per_col = D_in / 32;                  // K blocks per output column
    uint expert_bytes = D_out * n_blocks_per_col * BLK;
    uint col_bytes = n_blocks_per_col * BLK;
    device const uchar* W_exp = W_q40 + expert * expert_bytes + n * col_bytes;

    for (uint slot = gb; slot < ge; ++slot) {
        uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;
        float acc = 0.0f;

        for (uint kb = 0; kb < n_blocks_per_col; ++kb) {
            device const uchar* blk = W_exp + kb * BLK;
            device const half* blk_d = (device const half*)(blk);
            device const uchar* qs = blk + 2;
            float d = float(*blk_d);
            uint base_k = kb * 32;
            for (uint p = 0; p < 16; ++p) {
                uchar byte = qs[p];
                float w_lo = d * float(int(byte & 0xF) - 8);
                float w_hi = d * float(int((byte >> 4) & 0xF) - 8);
                acc += float(hid[base_k + p])      * w_lo
                     + float(hid[base_k + p + 16]) * w_hi;
            }
        }
        output[slot * D_out + n] = half(acc);
    }
}

// Dense Q4_0 v4 (multi-batch-per-TG) — for shared FFN down (K=2112), other
// mis-aligned dense projections. K must be divisible by 32 (Gemma: 2112 ✓).
kernel void dense_gemv_q4_0_v4(
    device const half* hidden           [[buffer(0)]],
    device const uchar* W_q40           [[buffer(1)]],
    device half* output                 [[buffer(2)]],
    constant uint& B                    [[buffer(3)]],
    constant uint& D_in                 [[buffer(4)]],
    constant uint& D_out                [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint MAX_B = 8;
    constexpr uint BLK = 18;
    uint n_block = tg.x; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint n_blocks_per_col = D_in / 32;
    uint col_bytes = n_blocks_per_col * BLK;
    device const uchar* W_col = W_q40 + n * col_bytes;

    float accs[MAX_B]; for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;

    for (uint kb = 0; kb < n_blocks_per_col; ++kb) {
        device const uchar* blk = W_col + kb * BLK;
        device const half* blk_d = (device const half*)(blk);
        device const uchar* qs = blk + 2;
        float d = float(*blk_d);
        uint base_k = kb * 32;
        for (uint p = 0; p < 16; ++p) {
            uchar byte = qs[p];
            float w_lo = d * float(int(byte & 0xF) - 8);
            float w_hi = d * float(int((byte >> 4) & 0xF) - 8);
            for (uint b = 0; b < B; ++b) {
                accs[b] += float(hidden[b * D_in + base_k + p])      * w_lo
                         + float(hidden[b * D_in + base_k + p + 16]) * w_hi;
            }
        }
    }
    for (uint b = 0; b < B; ++b) output[b * D_out + n] = half(accs[b]);
}

// MoE GEMV Q4 — uint4 packed W (32 nibbles per pack), K-unroll-by-8,
// dequant-at-register. All 32 lanes in a TG share the same uint4 read per
// k-iter (coalesced broadcast), each extracts its lane's nibble.
kernel void moe_gemv_q4_v3(
    device const half* hidden [[buffer(0)]],
    device const uint* slot_token [[buffer(1)]],
    device const uint4* W_q4 [[buffer(2)]],
    device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    constant float& q4_scale [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    uint expert_stride = D_in * (D_out / 32);
    uint k_stride = D_out / 32;
    device const uint4* w_exp = W_q4 + expert * expert_stride + n_block;
    uint word_idx = t / 8;
    uint nib_shift = (t % 8) * 4;

    for (uint slot = gb; slot < ge; ++slot) {
        uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;
        float acc = 0.0f;
        for (uint k = 0; k < D_in; k += 8) {
            uint4 p0 = w_exp[(k+0)*k_stride], p1 = w_exp[(k+1)*k_stride];
            uint4 p2 = w_exp[(k+2)*k_stride], p3 = w_exp[(k+3)*k_stride];
            uint4 p4 = w_exp[(k+4)*k_stride], p5 = w_exp[(k+5)*k_stride];
            uint4 p6 = w_exp[(k+6)*k_stride], p7 = w_exp[(k+7)*k_stride];
            half h0 = hid[k+0], h1 = hid[k+1], h2 = hid[k+2], h3 = hid[k+3];
            half h4 = hid[k+4], h5 = hid[k+5], h6 = hid[k+6], h7 = hid[k+7];
            #define EX(p) float(int(((word_idx==0)?(p).x:(word_idx==1)?(p).y:(word_idx==2)?(p).z:(p).w) >> nib_shift & 0xF) - 8) * q4_scale
            acc += float(h0)*EX(p0) + float(h1)*EX(p1) + float(h2)*EX(p2) + float(h3)*EX(p3)
                 + float(h4)*EX(p4) + float(h5)*EX(p5) + float(h6)*EX(p6) + float(h7)*EX(p7);
            #undef EX
        }
        output[slot * D_out + n] = half(acc);
    }
}

// Dense GEMV v4 + int8 W — half the W stream bytes vs fp16, preserve batch-
// amortization pattern. Better than Q4 at high B due to simpler dequant.
kernel void dense_gemv_i8w_v4(
    device const half* hidden [[buffer(0)]],
    device const char* W [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& B [[buffer(3)]], constant uint& D_in [[buffer(4)]], constant uint& D_out [[buffer(5)]],
    constant float& w_scale [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    constexpr uint MAX_B = 8;
    uint n_block = tg.x; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    device const char* w_col = W + n;
    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;
    for (uint k = 0; k < D_in; k += 8) {
        char w0 = w_col[(k+0)*D_out], w1 = w_col[(k+1)*D_out], w2 = w_col[(k+2)*D_out], w3 = w_col[(k+3)*D_out];
        char w4 = w_col[(k+4)*D_out], w5 = w_col[(k+5)*D_out], w6 = w_col[(k+6)*D_out], w7 = w_col[(k+7)*D_out];
        for (uint b = 0; b < B; ++b) {
            device const half* hid = hidden + b * D_in + k;
            accs[b] += float(hid[0])*float(w0) + float(hid[1])*float(w1) + float(hid[2])*float(w2) + float(hid[3])*float(w3)
                     + float(hid[4])*float(w4) + float(hid[5])*float(w5) + float(hid[6])*float(w6) + float(hid[7])*float(w7);
        }
    }
    for (uint b = 0; b < B; ++b) output[b * D_out + n] = half(accs[b] * w_scale);
}

// MoE GEMV (per-expert projection) — same v3 pattern with K-unroll-by-8.
// Launches E × (D_out/32) TGs; early-returns for experts with g=0.
kernel void moe_gemv_v3(
    device const half* hidden [[buffer(0)]],
    device const uint* slot_token [[buffer(1)]],
    device const half* W [[buffer(2)]],
    device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;
    for (uint slot = gb; slot < ge; ++slot) {
        uint tok = slot_token[slot];
        device const half* hid = hidden + tok * D_in;
        device const half* w_col = W + expert * D_in * D_out + n;
        float acc = 0.0f;
        for (uint k = 0; k < D_in; k += 8) {
            half w0 = w_col[(k+0)*D_out], w1 = w_col[(k+1)*D_out], w2 = w_col[(k+2)*D_out], w3 = w_col[(k+3)*D_out];
            half w4 = w_col[(k+4)*D_out], w5 = w_col[(k+5)*D_out], w6 = w_col[(k+6)*D_out], w7 = w_col[(k+7)*D_out];
            half h0 = hid[k+0], h1 = hid[k+1], h2 = hid[k+2], h3 = hid[k+3];
            half h4 = hid[k+4], h5 = hid[k+5], h6 = hid[k+6], h7 = hid[k+7];
            acc += float(h0)*float(w0) + float(h1)*float(w1) + float(h2)*float(w2) + float(h3)*float(w3)
                 + float(h4)*float(w4) + float(h5)*float(w5) + float(h6)*float(w6) + float(h7)*float(w7);
        }
        output[slot * D_out + n] = half(acc);
    }
}

// MoE combine (weighted scatter-add into hidden)
kernel void moe_combine(
    device const half* expert_out [[buffer(0)]], device const uint* batch_slots [[buffer(1)]],
    device const float* gate_w [[buffer(2)]], device half* hidden [[buffer(3)]],
    constant uint& top_k [[buffer(4)]], constant uint& D [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint db = tg.x; uint b = tg.y; uint t = lid.x; uint d = db * 32 + t;
    if (d >= D) return;
    float acc = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
        uint slot = batch_slots[b * top_k + k];
        float w = gate_w[b * top_k + k];
        acc += w * float(expert_out[slot * D + d]);
    }
    hidden[b * D + d] = half(float(hidden[b * D + d]) + acc);
}

// Embed lookup
kernel void embed_lookup(
    device const uint* tokens [[buffer(0)]], device const half* embed_table [[buffer(1)]],
    device half* hidden [[buffer(2)]], constant uint& D [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    uint tok = tokens[b];
    for (uint i = t; i < D; i += 32) hidden[b * D + i] = embed_table[tok * D + i];
}

// Vision 2D positional embedding add. The weight tensor is
// [2, POS_MAX, hidden] BF16 — y-table at `pos_table + 0`, x-table at
// `pos_table + POS_MAX * hidden`. For each patch, out[p] = in[p] +
// pos_y[y_idx] + pos_x[x_idx]. Simple broadcast across hidden dim.
kernel void vision_pos_embed_add_fp16(
    device const half* x                [[buffer(0)]],
    device const half* pos_y_table      [[buffer(1)]],
    device const half* pos_x_table      [[buffer(2)]],
    device half* out                    [[buffer(3)]],
    constant uint& nPatchesX            [[buffer(4)]],
    constant uint& hidden               [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint patch = tg.x; uint t = lid.x;
    uint yi = patch / nPatchesX;
    uint xi = patch % nPatchesX;
    device const half* py = pos_y_table + yi * hidden;
    device const half* px = pos_x_table + xi * hidden;
    device const half* xi_ptr = x + patch * hidden;
    device half* out_ptr = out + patch * hidden;
    for (uint i = t; i < hidden; i += 32) {
        out_ptr[i] = half(float(xi_ptr[i]) + float(py[i]) + float(px[i]));
    }
}

// Vision 2D NEOX RoPE applied in-place to Q or K. NEOX ordering: for each
// head's D-dim vector, split into first half [0..D/2) and second half
// [D/2..D). First half rotated by pos_x with its own D/2-dim frequency
// cycle, second half rotated by pos_y independently. (See llama.cpp
// gemma4v.cpp: build_rope_2d with NEOX flag.)
kernel void vision_2d_rope_neox_fp16(
    device half* x                      [[buffer(0)]],
    device const uint* pos_x            [[buffer(1)]],
    device const uint* pos_y            [[buffer(2)]],
    constant uint& H                    [[buffer(3)]],
    constant uint& HD                   [[buffer(4)]],
    constant float& theta               [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint token = tg.x; uint head = tg.y; uint t = lid.x;
    uint half_dim = HD / 2;           // 36 for HD=72
    uint quarter = half_dim / 2;      // 18 pairs per half
    device half* vec = x + (token * H + head) * HD;

    // First half rotated by pos_x
    uint px = pos_x[token];
    for (uint i = t; i < quarter; i += 32) {
        float freq = 1.0f / pow(theta, 2.0f * float(i) / float(half_dim));
        float ang = float(px) * freq;
        float c = cos(ang), s = sin(ang);
        float a = float(vec[i]);
        float b = float(vec[i + quarter]);
        vec[i]           = half(a * c - b * s);
        vec[i + quarter] = half(a * s + b * c);
    }
    // Second half rotated by pos_y
    uint py = pos_y[token];
    for (uint i = t; i < quarter; i += 32) {
        float freq = 1.0f / pow(theta, 2.0f * float(i) / float(half_dim));
        float ang = float(py) * freq;
        float c = cos(ang), s = sin(ang);
        float a = float(vec[half_dim + i]);
        float b = float(vec[half_dim + i + quarter]);
        vec[half_dim + i]           = half(a * c - b * s);
        vec[half_dim + i + quarter] = half(a * s + b * c);
    }
}

// Dense fp16 GEMV v5 — split-K pattern, 4 SGs per TG, for the vision tower's
// Q/K/V/O projections, MLP gate/up/down, and vision→text embedding
// projection. Row-major W [D_out, D_in]; not swizzled (add that later if
// perf pinned on vision tower).
kernel void dense_gemv_fp16_v5(
    device const half* x                [[buffer(0)]],
    device const half* W                [[buffer(1)]],
    device half* output                 [[buffer(2)]],
    constant uint& D_in                 [[buffer(3)]],
    constant uint& D_out                [[buffer(4)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]])
{
    constexpr uint N_SPLITS = 4;
    uint n_block = tg.x; uint b = tg.y; uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint k_per_sg = D_in / N_SPLITS;
    uint k_begin = sg_id * k_per_sg;
    uint k_end = k_begin + k_per_sg;
    float acc = 0.0f;
    device const half* x_b = x + b * D_in;
    device const half* W_row = W + n * D_in;
    for (uint k = k_begin; k < k_end; ++k) {
        acc += float(x_b[k]) * float(W_row[k]);
    }
    threadgroup float partials[N_SPLITS][32];
    partials[sg_id][lid_sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        float total = partials[0][lid_sg] + partials[1][lid_sg] + partials[2][lid_sg] + partials[3][lid_sg];
        output[b * D_out + n] = half(total);
    }
}

// Vision encoder attention (prefill). Bidirectional (no causal mask), no
// KV cache — Q, K, V are computed on the fly from the current token
// sequence and discarded after this forward. One TG per (token, head)
// computes the 72-dim output via two-pass scaled softmax over all N tokens.
// Stores full scores[N] in tg-mem (fp16) — at N=2520 that's 5 KB, fits.
// This is a simple correctness-first implementation; flash-attention-style
// tiling is a later perf pass.
kernel void vision_attn_prefill_fp16(
    device const half* Q                [[buffer(0)]],  // [N, H, HD]
    device const half* K                [[buffer(1)]],  // [N, H, HD]
    device const half* V                [[buffer(2)]],  // [N, H, HD]
    device half* O                      [[buffer(3)]],  // [N, H, HD]
    constant uint& N                    [[buffer(4)]],
    constant uint& H                    [[buffer(5)]],
    constant uint& HD                   [[buffer(6)]],
    constant float& qk_scale            [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint THREADS = 32;
    constexpr uint N_MAX = 3072;      // big enough for 2520 patches + slack
    // scores kept in float — with kq_scale=1.0 (Gemma4V) dot products can
    // exceed half's 65504 cap for large sequence length, and even a single
    // +Inf in half softmax produces NaN after max-subtraction.
    threadgroup float scores[N_MAX];
    threadgroup float state[3];       // [row_max, sum_exp, unused]

    uint q_tok = tg.x; uint head = tg.y; uint t = lid.x;
    device const half* Q_qh = Q + (q_tok * H + head) * HD;
    device const half* K_h  = K + head * HD;                     // stride (N, HD)
    device const half* V_h  = V + head * HD;
    uint K_stride = H * HD;

    // === Pass 1: scores[k] = <Q, K[k]> * qk_scale ===
    for (uint k = t; k < N; k += THREADS) {
        float s = 0;
        device const half* K_k = K_h + k * K_stride;
        for (uint d = 0; d < HD; ++d) {
            s += float(Q_qh[d]) * float(K_k[d]);
        }
        scores[k] = s * qk_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Pass 2: find row max ===
    float local_max = -INFINITY;
    for (uint k = t; k < N; k += THREADS) local_max = max(local_max, scores[k]);
    float row_max = simd_max(local_max);
    if (t == 0) state[0] = row_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    row_max = state[0];

    // === Pass 3: exp(score - max), sum ===
    float local_sum = 0;
    for (uint k = t; k < N; k += THREADS) {
        float e = exp(scores[k] - row_max);
        scores[k] = e;
        local_sum += e;
    }
    float row_sum = simd_sum(local_sum);
    if (t == 0) state[1] = row_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / state[1];

    // === Pass 4: O[q, head, :] = sum_k scores[k] * V[k, head, :] * inv_sum ===
    device half* O_qh = O + (q_tok * H + head) * HD;
    for (uint d = t; d < HD; d += THREADS) {
        float acc = 0;
        for (uint k = 0; k < N; ++k) {
            acc += scores[k] * float(V_h[k * K_stride + d]);
        }
        O_qh[d] = half(acc * inv_sum);
    }
}

// Vision 2D average pool (kernel_size × kernel_size, stride=kernel_size).
// Input: [grid_h * grid_w, hidden] row-major over the 2D grid.
// Output: [out_h * out_w, hidden] where out_h = grid_h / kernel_size,
// out_w = grid_w / kernel_size. Exact-divisibility only; non-divisible
// cases should pad or truncate on the CPU side before calling.
kernel void vision_pool_2d_fp16(
    device const half* x                [[buffer(0)]],
    device half* out                    [[buffer(1)]],
    constant uint& grid_w               [[buffer(2)]],
    constant uint& out_w                [[buffer(3)]],
    constant uint& kernel_size          [[buffer(4)]],
    constant uint& hidden               [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint out_idx = tg.x; uint t = lid.x;
    uint oy = out_idx / out_w;
    uint ox = out_idx % out_w;
    uint y_start = oy * kernel_size;
    uint x_start = ox * kernel_size;
    float inv_area = 1.0f / float(kernel_size * kernel_size);
    for (uint i = t; i < hidden; i += 32) {
        float acc = 0;
        for (uint dy = 0; dy < kernel_size; ++dy) {
            for (uint dx = 0; dx < kernel_size; ++dx) {
                uint px = (y_start + dy) * grid_w + (x_start + dx);
                acc += float(x[px * hidden + i]);
            }
        }
        out[out_idx * hidden + i] = half(acc * inv_area);
    }
}

// ---------- fp32-residual-stream variants for vision tower -----------
// Gemma4 has outlier-valued RMSNorm gammas (e.g. post_ffn_norm.weight[294]
// ≈ 16.875) that amplify rounding error into specific channels. Keeping the
// residual stream `x` in fp32 (while everything else stays fp16) eliminates
// most of the compounding drift over 27 encoder layers.

kernel void dense_gemv_fp16in_fp32out_v5(
    device const half* x                [[buffer(0)]],
    device const half* W                [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant uint& D_in                 [[buffer(3)]],
    constant uint& D_out                [[buffer(4)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]])
{
    constexpr uint N_SPLITS = 4;
    uint n_block = tg.x; uint b = tg.y; uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint k_per_sg = D_in / N_SPLITS;
    uint k_begin = sg_id * k_per_sg;
    uint k_end = k_begin + k_per_sg;
    float acc = 0.0f;
    device const half* x_b = x + b * D_in;
    device const half* W_row = W + n * D_in;
    for (uint k = k_begin; k < k_end; ++k) {
        acc += float(x_b[k]) * float(W_row[k]);
    }
    threadgroup float partials[N_SPLITS][32];
    partials[sg_id][lid_sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        output[b * D_out + n] = partials[0][lid_sg] + partials[1][lid_sg] + partials[2][lid_sg] + partials[3][lid_sg];
    }
}

kernel void vision_pos_embed_add_fp32(
    device const float* x               [[buffer(0)]],
    device const half* pos_y_table      [[buffer(1)]],
    device const half* pos_x_table      [[buffer(2)]],
    device float* out                   [[buffer(3)]],
    constant uint& nPatchesX            [[buffer(4)]],
    constant uint& hidden               [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint patch = tg.x; uint t = lid.x;
    uint yi = patch / nPatchesX;
    uint xi = patch % nPatchesX;
    device const half* py = pos_y_table + yi * hidden;
    device const half* px = pos_x_table + xi * hidden;
    device const float* xi_ptr = x + patch * hidden;
    device float* out_ptr = out + patch * hidden;
    for (uint i = t; i < hidden; i += 32) {
        out_ptr[i] = xi_ptr[i] + float(py[i]) + float(px[i]);
    }
}

// Post-block RMSNorm with gamma — reads fp16 sub-block output, writes fp32.
// Used for post_attention_layernorm and post_feedforward_layernorm so the
// value added to the fp32 residual stream carries full fp32 precision
// (preventing fp16 rounding on outlier-gamma channels like dim 294 with
// post_ffn_norm.weight = 16.875).
kernel void rms_norm_fp16in_fp32out(
    device const half* x                [[buffer(0)]],
    device float* y                     [[buffer(1)]],
    device const half* gamma            [[buffer(2)]],
    constant uint& D                    [[buffer(3)]],
    constant float& eps                 [[buffer(4)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    float s = 0.0f;
    for (uint i = t; i < D; i += 32) { float v = float(x[b*D+i]); s += v*v; }
    s = simd_sum(s);
    float sc = rsqrt(s/float(D) + eps);
    for (uint i = t; i < D; i += 32) {
        y[b*D+i] = float(x[b*D+i]) * sc * float(gamma[i]);
    }
}

// Residual-stream add where both sides are fp32.
kernel void add_inplace_fp32_fp32(
    device float* dst                   [[buffer(0)]],
    device const float* src             [[buffer(1)]],
    constant uint& N                    [[buffer(2)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    for (uint i = t; i < N; i += 32) {
        dst[b*N+i] = dst[b*N+i] + src[b*N+i];
    }
}

// Same normalization math as rms_norm but reads fp32 x (residual stream).
// Output stays fp16 because tmp downstream is fp16 and Q/K/V projections
// take fp16 input.
kernel void rms_norm_fp32in(
    device const float* x               [[buffer(0)]],
    device half* y                      [[buffer(1)]],
    device const half* gamma            [[buffer(2)]],
    constant uint& D                    [[buffer(3)]],
    constant float& eps                 [[buffer(4)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    float s = 0.0f;
    for (uint i = t; i < D; i += 32) { float v = x[b*D+i]; s += v*v; }
    s = simd_sum(s);
    float sc = rsqrt(s/float(D) + eps);
    for (uint i = t; i < D; i += 32) {
        y[b*D+i] = half(x[b*D+i] * sc * float(gamma[i]));
    }
}

// dst is the fp32 residual stream; src is a fp16 post-norm sub-block output.
// Equivalent to `x += post_norm(sub)` but without the fp16 round-trip on x.
kernel void add_inplace_fp32dst_fp16src(
    device float* dst                   [[buffer(0)]],
    device const half* src              [[buffer(1)]],
    constant uint& N                    [[buffer(2)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    for (uint i = t; i < N; i += 32) {
        dst[b*N+i] = dst[b*N+i] + float(src[b*N+i]);
    }
}

// Pool reads the fp32 residual stream and emits fp16 pooled output. The
// subsequent standardize / RMSNorm / projection can safely stay fp16 since
// they renormalize per-token and there's no residual accumulation after.
kernel void vision_pool_2d_fp32in_fp16out(
    device const float* x               [[buffer(0)]],
    device half* out                    [[buffer(1)]],
    constant uint& grid_w               [[buffer(2)]],
    constant uint& out_w                [[buffer(3)]],
    constant uint& kernel_size          [[buffer(4)]],
    constant uint& hidden               [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint out_idx = tg.x; uint t = lid.x;
    uint oy = out_idx / out_w;
    uint ox = out_idx % out_w;
    uint y_start = oy * kernel_size;
    uint x_start = ox * kernel_size;
    float inv_area = 1.0f / float(kernel_size * kernel_size);
    for (uint i = t; i < hidden; i += 32) {
        float acc = 0;
        for (uint dy = 0; dy < kernel_size; ++dy) {
            for (uint dx = 0; dx < kernel_size; ++dx) {
                uint px = (y_start + dy) * grid_w + (x_start + dx);
                acc += x[px * hidden + i];
            }
        }
        out[out_idx * hidden + i] = half(acc * inv_area);
    }
}

// Vision std-normalize + sqrt(hidden) scale fused. Matches gemma4v.cpp:
//   y = (x * sqrt(hidden) - std_bias) * std_scale
// Applied after pool, before embed_vision projection. Per-element over a
// [num_vecs, D] tensor; gamma-like broadcasts of bias and scale over D.
kernel void vision_scaled_std_normalize_fp16(
    device const half* x                [[buffer(0)]],
    device const half* bias             [[buffer(1)]],
    device const half* scale            [[buffer(2)]],
    device half* out                    [[buffer(3)]],
    constant uint& D                    [[buffer(4)]],
    constant float& global_scale        [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    for (uint i = t; i < D; i += 32) {
        float v = float(x[b * D + i]) * global_scale;
        v -= float(bias[i]);
        out[b * D + i] = half(v * float(scale[i]));
    }
}

// Vision patch embed (fp16), HWC-within-patch flatten order. Turns a
// [3, H, W] image tensor in CHW layout into [N_patches, hidden=1152] where
// N_patches = (H/PATCH) * (W/PATCH). Matches Gemma4ImageProcessor's
// `convert_image_to_patches`: reshape (C, pH, P, pW, P) → permute (1, 3, 2, 4, 0)
// → flatten, producing per-patch 768-vector in order (py_in, px_in, C) —
// HWC with channel innermost. Any aspect ratio works (no square-mode);
// W and H must each be a multiple of PATCH.
kernel void vision_patch_embed_fp16(
    device const half* img          [[buffer(0)]],   // [3, H, W] CHW
    device const half* W            [[buffer(1)]],   // [D_out, D_in=768]
    device half* output             [[buffer(2)]],   // [N_patches, D_out]
    constant uint& imgH             [[buffer(3)]],
    constant uint& imgW             [[buffer(4)]],
    constant uint& D_out            [[buffer(5)]],
    uint2 tg                        [[threadgroup_position_in_grid]],
    uint2 lid                       [[thread_position_in_threadgroup]])
{
    constexpr uint PATCH = 16;
    constexpr uint D_IN = 3 * PATCH * PATCH;    // 768
    uint n_block = tg.x; uint patch = tg.y; uint t = lid.x;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;

    uint nPx = imgW / PATCH;                    // patches per row
    uint py = patch / nPx;
    uint px = patch % nPx;

    uint plane = imgH * imgW;                   // pixels per channel
    uint origin_y = py * PATCH;
    uint origin_x = px * PATCH;

    float acc = 0.0f;
    device const half* W_row = W + n * D_IN;

    // HWC flatten: k = (y_in * PATCH + x_in) * 3 + c. Iterate y_in, x_in, c
    // in that order so the W reads march linearly through the 768-vector.
    for (uint y = 0; y < PATCH; ++y) {
        uint img_y = (origin_y + y) * imgW;
        uint w_y_base = y * PATCH * 3;
        for (uint x = 0; x < PATCH; ++x) {
            uint img_yx = img_y + (origin_x + x);
            uint w_yx_base = w_y_base + x * 3;
            float p0 = float(img[0 * plane + img_yx]);
            float p1 = float(img[1 * plane + img_yx]);
            float p2 = float(img[2 * plane + img_yx]);
            acc += p0 * float(W_row[w_yx_base + 0]);
            acc += p1 * float(W_row[w_yx_base + 1]);
            acc += p2 * float(W_row[w_yx_base + 2]);
        }
    }
    output[patch * D_out + n] = half(acc);
}

// Softcap
kernel void softcap(
    device half* x [[buffer(0)]], constant uint& N [[buffer(1)]], constant float& cap [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    device half* row = x + b * N;
    float inv = 1.0f / cap;
    for (uint i = t; i < N; i += 32) { float v = float(row[i]); row[i] = half(tanh(v * inv) * cap); }
}

// ========================================================================
// Flex Attention — tile-streaming, block-sparse, parameterized mask_mod.
//
// Replaces the hard-coded k-page loop with two precomputed lists of k_block
// indices: FULL (mask is True throughout the tile → skip per-k predicate)
// and PARTIAL (intra-tile predicate required). EMPTY tiles are never in a
// list, so they never dispatch.
//
// v0: slide layer (D=256, PAGE=16, Q_PER_TG=2, Q_BLOCK=1), causal_sliding
// mask_mod only. Same MMA+softmax+AV structure as paged_attn_slide_gqa_compute
// so correctness is a pure A/B test.
// ========================================================================
kernel void flex_attn_slide_v0(
    device const half* Q                    [[buffer(0)]],
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],
    device float* m_partials                [[buffer(4)]],
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],
    device const uint* full_kv_offsets      [[buffer(7)]],
    device const uint* full_kv_indices      [[buffer(8)]],
    device const uint* partial_kv_offsets   [[buffer(9)]],
    device const uint* partial_kv_indices   [[buffer(10)]],
    device const uint* k_len_per_slot       [[buffer(11)]],
    constant float& qk_scale                [[buffer(12)]],
    constant uint& max_pages                [[buffer(13)]],
    constant uint& H_Q                      [[buffer(14)]],
    constant uint& H_KV                     [[buffer(15)]],
    constant uint& N_SPLITS                 [[buffer(16)]],    // internal — partitions CSR work (per_split = n_total/N_SPLITS)
    constant uint& sliding_window           [[buffer(17)]],
    constant uint& prefix_pages             [[buffer(18)]],    // skip logical pages < this (tail mode). 0 = process all.
    constant uint& split_offset             [[buffer(19)]],    // write partials at total_splits_out stride, + split_offset + tg.y. 0 = default.
    constant uint& total_splits_out         [[buffer(20)]],    // output layout stride. If 0, falls back to N_SPLITS (v0 back-compat).
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 256;
    constexpr uint PAGE = 16;
    constexpr uint THREADS = 32;
    constexpr uint Q_PER_TG = 2;
    constexpr uint D8 = D / 8;

    const uint vs = tg_pos.x;
    const uint split = tg_pos.y;
    const uint slot = vs / H_KV;
    const uint kv_head = vs % H_KV;
    const uint q_head_base = kv_head * Q_PER_TG;
    const uint lid = lid3.x;

    // v0: Q_BLOCK=1, q_blocks=1 → CSR index is just the slot.
    const uint csr_idx = slot;

    threadgroup half  Q_tile[8 * D];
    threadgroup half  scores_tile[8 * PAGE];
    threadgroup float O_acc[Q_PER_TG * D];
    threadgroup float m_state[Q_PER_TG];
    threadgroup float l_state[Q_PER_TG];
    threadgroup float scale_tile[Q_PER_TG];

    device const half* Qbase = Q + (slot * H_Q + q_head_base) * D;
    device const uint* bt_s = block_table + slot * max_pages;

    for (uint i = lid; i < 8 * D; i += THREADS) {
        uint r = i / D;
        Q_tile[i] = (r < Q_PER_TG) ? Qbase[r * D + (i % D)] : half(0);
    }
    for (uint i = lid; i < Q_PER_TG * D; i += THREADS) O_acc[i] = 0.0f;
    if (lid < Q_PER_TG) { m_state[lid] = -INFINITY; l_state[lid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k_len = k_len_per_slot[slot];
    const uint kv_row_stride = H_KV * D;
    const uint window_lo = (sliding_window > 0 && k_len > sliding_window)
                           ? (k_len - sliding_window) : 0u;

    // Each split gets a contiguous slice of the per-slot list. We split the
    // FULL list first and PARTIAL list second, concatenated. This keeps
    // roughly equal work per split when both lists are non-empty.
    uint full_lo  = full_kv_offsets[csr_idx];
    uint full_hi  = full_kv_offsets[csr_idx + 1];
    uint part_lo  = partial_kv_offsets[csr_idx];
    uint part_hi  = partial_kv_offsets[csr_idx + 1];
    uint n_full   = full_hi - full_lo;
    uint n_part   = part_hi - part_lo;
    uint n_total  = n_full + n_part;
    uint per_split = (n_total + N_SPLITS - 1) / N_SPLITS;
    uint ix_begin = split * per_split;
    uint ix_end   = min(ix_begin + per_split, n_total);

    // Walk the split's assigned blocks. FULL blocks come first, then PARTIAL,
    // so we dispatch based on where ix lands. When `prefix_pages > 0` we skip
    // any logical page index < prefix_pages (those positions are handled by
    // the shared-prefix broadcast kernel at split=0).
    for (uint ix = ix_begin; ix < ix_end; ++ix) {
        uint p;
        bool is_partial;
        if (ix < n_full) {
            p = full_kv_indices[full_lo + ix];
            is_partial = false;
        } else {
            p = partial_kv_indices[part_lo + (ix - n_full)];
            is_partial = true;
        }
        // Skip logical pages owned by the shared-prefix broadcast kernel.
        if (p < prefix_pages) continue;

        const uint phys = bt_s[p];
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;

        // QK: PAGE=16 needs two 8-col K blocks.
        for (uint pb = 0; pb < PAGE / 8; ++pb) {
            simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            device const half* pk = Kbase + (pb * 8) * kv_row_stride;
            for (uint dt = 0; dt < D8; ++dt) {
                simdgroup_half8x8 mq, mk;
                simdgroup_load(mq, Q_tile + dt * 8, D);
                simdgroup_load(mk, pk + dt * 8, kv_row_stride, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }
            simdgroup_store(mqk, scores_tile + pb * 8, PAGE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row online softmax. FULL blocks skip the per-k predicate.
        if (lid < Q_PER_TG) {
            const uint q = lid;
            float row_max = -INFINITY;
            float s_loc[PAGE];
            if (is_partial) {
                for (uint k = 0; k < PAGE; ++k) {
                    float sv = float(scores_tile[q * PAGE + k]) * qk_scale;
                    uint k_pos = p * PAGE + k;
                    // mask_mod(causalSliding): reject k_pos past causal or before window
                    if (k_pos >= k_len || k_pos < window_lo) sv = -INFINITY;
                    s_loc[k] = sv;
                    row_max = max(row_max, sv);
                }
            } else {
                for (uint k = 0; k < PAGE; ++k) {
                    float sv = float(scores_tile[q * PAGE + k]) * qk_scale;
                    s_loc[k] = sv;
                    row_max = max(row_max, sv);
                }
            }
            float m_old = m_state[q];
            float m_new = max(m_old, row_max);
            float scale = exp(m_old - m_new);
            float sum = 0.0f;
            for (uint k = 0; k < PAGE; ++k) {
                float e = exp(s_loc[k] - m_new);
                scores_tile[q * PAGE + k] = half(e);
                sum += e;
            }
            m_state[q] = m_new;
            l_state[q] = l_state[q] * scale + sum;
            scale_tile[q] = scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // AV: scalar cooperative (same as paged_attn_slide_gqa_compute).
        for (uint d = lid; d < D; d += THREADS) {
            half V_reg[PAGE];
            for (uint k = 0; k < PAGE; ++k) {
                V_reg[k] = Vbase[k * kv_row_stride + d];
            }
            for (uint q = 0; q < Q_PER_TG; ++q) {
                float acc = O_acc[q * D + d] * scale_tile[q];
                for (uint k = 0; k < PAGE; ++k) {
                    acc += float(scores_tile[q * PAGE + k]) * float(V_reg[k]);
                }
                O_acc[q * D + d] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials per Q head. Output layout stride = total_splits_out
    // (falls back to N_SPLITS for pre-broadcast callers that don't set it).
    // With split_offset>0, this kernel writes to the tail slice of a shared
    // partials buffer whose split=0 is owned by the shared-prefix kernel.
    const uint out_stride = (total_splits_out > 0) ? total_splits_out : N_SPLITS;
    const bool empty_split = (ix_begin >= ix_end);
    for (uint q = 0; q < Q_PER_TG; ++q) {
        const uint q_head = q_head_base + q;
        const uint pidx = (slot * H_Q + q_head) * out_stride + split_offset + split;
        if (lid == 0) {
            m_partials[pidx] = empty_split ? -INFINITY : m_state[q];
            l_partials[pidx] = empty_split ? 0.0f : l_state[q];
        }
        device float* O_part = O_partials + pidx * D;
        for (uint d = lid; d < D; d += THREADS) O_part[d] = O_acc[q * D + d];
    }
}

// Cross-slot K/V broadcast kernel for AR decode over a SHARED PREFIX.
//
// Problem it solves: when B sessions share a KV prefix (same system prompt,
// same multi-turn history, same cached tool-call result), every slot's AR
// attention step independently re-reads the same K/V from DRAM. For a 1000-
// token shared prefix at D=256, that's 1000 × 256 × 2 B × B reads per layer.
// This kernel loads each shared page's K/V ONCE into threadgroup memory, then
// runs Q@K for every slot's Q against that single staged K. Saves (B-1)/B of
// the K/V bandwidth on the shared range (75% at B=4).
//
// Grid: (H_KV, 1, 1) — one TG per kv_head, fans out to all B slots in-TG.
// Output: partials at (slot, q_head, split=0). Paired with a tail-only kernel
// (the standard per-slot kernel called with a starting page offset) writing
// split=1, then paged_attn_split_reduce with N_SPLITS=2 merges them into the
// final per-slot attention output.
//
// Scope for v1: slide layers only (D=256, PAGE=16, Q_PER_TG=2). Full-attn has
// the same pattern at D=512, PAGE=8 but different tg-mem math; left for a
// follow-up. Causal + sliding-window applied per-slot in softmax using
// per-slot k_len from `k_len_per_slot`. Each slot's SW horizon can differ,
// so the mask is still per-slot even though K/V loads are shared.
kernel void paged_attn_slide_ar_shared(
    device const half* Q                    [[buffer(0)]],   // [B, H_Q, D]
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* shared_pages         [[buffer(3)]],   // [prefix_pages] phys-page list
    device float* m_partials                [[buffer(4)]],   // [B, H_Q, N_SPLITS]
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],   // [B, H_Q, N_SPLITS, D]
    device const uint* k_len_per_slot       [[buffer(7)]],
    constant float& qk_scale                [[buffer(8)]],
    constant uint& H_Q                      [[buffer(9)]],
    constant uint& H_KV                     [[buffer(10)]],
    constant uint& N_SPLITS                 [[buffer(11)]],
    constant uint& prefix_pages             [[buffer(12)]],
    constant uint& sliding_window           [[buffer(13)]],
    constant uint& B_batch                  [[buffer(14)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 256;
    constexpr uint PAGE = 16;
    constexpr uint THREADS = 128;       // 4 simdgroups
    constexpr uint Q_PER_TG = 2;
    constexpr uint MAX_B = 4;            // cap; B_batch runtime arg picks actual
    constexpr uint D8 = D / 8;

    const uint kv_head = tg_pos.x;
    const uint q_head_base = kv_head * Q_PER_TG;
    const uint lid = lid3.x;

    // Threadgroup state. See kernel header for tg-mem budget.
    threadgroup half  Q_tile[MAX_B * Q_PER_TG * D];       // 4*2*256*2 = 4KB
    threadgroup half  K_stage[PAGE * D];                  // 16*256*2 = 8KB
    threadgroup half  V_stage[PAGE * D];                  // 16*256*2 = 8KB
    threadgroup half  scores_tile[MAX_B * Q_PER_TG * PAGE]; // 4*2*16*2 = 256B
    threadgroup float O_acc[MAX_B * Q_PER_TG * D];        // 4*2*256*4 = 8KB
    threadgroup float m_state[MAX_B * Q_PER_TG];
    threadgroup float l_state[MAX_B * Q_PER_TG];
    threadgroup float scale_tile[MAX_B * Q_PER_TG];

    // Load all active slots' Q for this kv_head's Q_PER_TG grouping.
    for (uint i = lid; i < B_batch * Q_PER_TG * D; i += THREADS) {
        uint b = i / (Q_PER_TG * D);
        uint r = (i / D) % Q_PER_TG;
        uint d = i % D;
        uint q_head = q_head_base + r;
        Q_tile[i] = Q[(b * H_Q + q_head) * D + d];
    }
    // Clear O_acc for all active slots.
    for (uint i = lid; i < B_batch * Q_PER_TG * D; i += THREADS) O_acc[i] = 0.0f;
    if (lid < B_batch * Q_PER_TG) {
        m_state[lid] = -INFINITY;
        l_state[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kv_row_stride = H_KV * D;

    // Iterate over SHARED pages. Each page's K/V is loaded once and reused
    // across all active slots' Q@K + score@V computations.
    for (uint p = 0; p < prefix_pages; ++p) {
        const uint phys = shared_pages[p];
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;

        // Stage K into tg-mem. Layout: K_stage[k * D + d].
        for (uint i = lid; i < PAGE * D; i += THREADS) {
            uint k = i / D;
            uint d = i % D;
            K_stage[i] = Kbase[k * kv_row_stride + d];
        }
        // Stage V into tg-mem. Same layout.
        for (uint i = lid; i < PAGE * D; i += THREADS) {
            uint k = i / D;
            uint d = i % D;
            V_stage[i] = Vbase[k * kv_row_stride + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Q@K for every slot's Q, against the shared K. Each slot's Q_PER_TG
        // rows share a K column (PAGE=16 → 2 passes of 8 K cols each).
        for (uint b = 0; b < B_batch; ++b) {
            for (uint pb = 0; pb < PAGE / 8; ++pb) {
                simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                threadgroup const half* pk = K_stage + (pb * 8) * D;
                for (uint dt = 0; dt < D8; ++dt) {
                    simdgroup_half8x8 mq, mk;
                    simdgroup_load(mq, Q_tile + b * Q_PER_TG * D + dt * 8, D);
                    simdgroup_load(mk, pk + dt * 8, D, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                }
                simdgroup_store(mqk, scores_tile + (b * Q_PER_TG) * PAGE + pb * 8, PAGE);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row online softmax with PER-SLOT mask (k_len and SW may differ
        // across slots even when the prefix pages themselves are shared).
        if (lid < B_batch * Q_PER_TG) {
            uint b = lid / Q_PER_TG;
            uint row_idx = lid;
            uint k_len = k_len_per_slot[b];
            uint window_lo = (sliding_window > 0 && k_len > sliding_window)
                             ? (k_len - sliding_window) : 0u;

            float row_max = -INFINITY;
            float s_loc[PAGE];
            for (uint k = 0; k < PAGE; ++k) {
                float sv = float(scores_tile[row_idx * PAGE + k]) * qk_scale;
                uint k_pos = p * PAGE + k;
                if (k_pos >= k_len || k_pos < window_lo) sv = -INFINITY;
                s_loc[k] = sv;
                row_max = max(row_max, sv);
            }
            float m_old = m_state[row_idx];
            float m_new = max(m_old, row_max);
            float scale = exp(m_old - m_new);
            float sum = 0.0f;
            for (uint k = 0; k < PAGE; ++k) {
                float e = exp(s_loc[k] - m_new);
                scores_tile[row_idx * PAGE + k] = half(e);
                sum += e;
            }
            m_state[row_idx] = m_new;
            l_state[row_idx] = l_state[row_idx] * scale + sum;
            scale_tile[row_idx] = scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // AV: each lane owns D/THREADS dims; for each (slot, q_head) row,
        // multiply by rescale then accumulate scores @ V.
        for (uint sqd = lid; sqd < B_batch * Q_PER_TG * D; sqd += THREADS) {
            uint b = sqd / (Q_PER_TG * D);
            uint q = (sqd / D) % Q_PER_TG;
            uint d = sqd % D;
            uint row_idx = b * Q_PER_TG + q;
            float acc = O_acc[row_idx * D + d] * scale_tile[row_idx];
            for (uint k = 0; k < PAGE; ++k) {
                acc += float(scores_tile[row_idx * PAGE + k])
                     * float(V_stage[k * D + d]);
            }
            O_acc[row_idx * D + d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials at split=0 per (slot, q_head). Empty-prefix edge case:
    // if prefix_pages==0 we never entered the loop; m_state is still -INF
    // (correctly signalling "nothing seen") and O_acc is zero.
    if (lid < B_batch * Q_PER_TG) {
        uint b = lid / Q_PER_TG;
        uint q = lid % Q_PER_TG;
        uint q_head = q_head_base + q;
        uint pidx = (b * H_Q + q_head) * N_SPLITS + 0;
        m_partials[pidx] = m_state[lid];
        l_partials[pidx] = l_state[lid];
    }
    for (uint sqd = lid; sqd < B_batch * Q_PER_TG * D; sqd += THREADS) {
        uint b = sqd / (Q_PER_TG * D);
        uint q = (sqd / D) % Q_PER_TG;
        uint d = sqd % D;
        uint q_head = q_head_base + q;
        uint pidx = (b * H_Q + q_head) * N_SPLITS + 0;
        O_partials[pidx * D + d] = O_acc[(b * Q_PER_TG + q) * D + d];
    }
}

// Flex attention v1 — slide layer with Q_BLOCK=8 (prefill).
// Each TG covers 8 consecutive Q positions × Q_PER_TG=2 Q-heads → 16 real Q rows.
// MMA runs in 2 passes of 8 rows each. Per-Q-row softmax applies a per-row
// causal+sliding mask using per-row q_positions. Partials laid out per (q_pos, q_head).
kernel void flex_attn_slide_v1_q8(
    device const half* Q                    [[buffer(0)]],   // [B, Q_LEN, H_Q, D]
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],
    device float* m_partials                [[buffer(4)]],   // [B, Q_LEN, H_Q, N_SPLITS]
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],   // [B, Q_LEN, H_Q, N_SPLITS, D]
    device const uint* full_kv_offsets      [[buffer(7)]],   // CSR [B*Q_BLOCKS+1]
    device const uint* full_kv_indices      [[buffer(8)]],
    device const uint* partial_kv_offsets   [[buffer(9)]],
    device const uint* partial_kv_indices   [[buffer(10)]],
    device const uint* q_positions          [[buffer(11)]],  // [B, Q_LEN] u32
    device const uint* k_len_per_slot       [[buffer(12)]],
    constant float& qk_scale                [[buffer(13)]],
    constant uint& max_pages                [[buffer(14)]],
    constant uint& H_Q                      [[buffer(15)]],
    constant uint& H_KV                     [[buffer(16)]],
    constant uint& N_SPLITS                 [[buffer(17)]],
    constant uint& sliding_window           [[buffer(18)]],  // legacy: ignored when mask bitmap drives masking
    constant uint& q_len                    [[buffer(19)]],
    device const uint* partial_block_masks  [[buffer(20)]],  // [total_partials, Q_BLOCK] uint; bit k set = keep
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 256;
    constexpr uint PAGE = 16;
    constexpr uint THREADS = 32;
    constexpr uint Q_PER_TG = 2;
    constexpr uint Q_BLOCK = 8;
    constexpr uint Q_ROWS = Q_PER_TG * Q_BLOCK;   // 16
    constexpr uint D8 = D / 8;

    const uint vs = tg_pos.x;
    const uint q_block_idx = tg_pos.y;   // which Q tile
    const uint split = tg_pos.z;
    const uint slot = vs / H_KV;
    const uint kv_head = vs % H_KV;
    const uint q_head_base = kv_head * Q_PER_TG;
    const uint lid = lid3.x;

    const uint q_blocks_per_slot = (q_len + Q_BLOCK - 1) / Q_BLOCK;
    const uint csr_idx = slot * q_blocks_per_slot + q_block_idx;
    const uint q_local_base = q_block_idx * Q_BLOCK;

    threadgroup half  Q_tile[Q_ROWS * D];
    threadgroup half  scores_tile[Q_ROWS * PAGE];
    threadgroup float O_acc[Q_ROWS * D];
    threadgroup float m_state[Q_ROWS];
    threadgroup float l_state[Q_ROWS];
    threadgroup float scale_tile[Q_ROWS];
    threadgroup uint  q_pos_tg[Q_BLOCK];

    device const uint* bt_s = block_table + slot * max_pages;

    // Load Q_tile with layout [row=q_local*Q_PER_TG+h_local, dim=d]. Pad
    // rows where q_local_base+q_local >= q_len to zero so their MMA
    // contributes zero.
    for (uint i = lid; i < Q_ROWS * D; i += THREADS) {
        uint r = i / D;
        uint q_local = r / Q_PER_TG;
        uint h_local = r % Q_PER_TG;
        uint q_pos_in_seq = q_local_base + q_local;
        if (q_pos_in_seq >= q_len) {
            Q_tile[i] = half(0);
        } else {
            uint q_flat = (slot * q_len + q_pos_in_seq) * H_Q + q_head_base + h_local;
            Q_tile[i] = Q[q_flat * D + (i % D)];
        }
    }
    for (uint i = lid; i < Q_ROWS * D; i += THREADS) O_acc[i] = 0.0f;
    if (lid < Q_ROWS) { m_state[lid] = -INFINITY; l_state[lid] = 0.0f; }
    if (lid < Q_BLOCK) {
        uint q_pos_in_seq = q_local_base + lid;
        q_pos_tg[lid] = (q_pos_in_seq < q_len)
                        ? q_positions[slot * q_len + q_pos_in_seq] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k_len = k_len_per_slot[slot];
    const uint kv_row_stride = H_KV * D;

    uint full_lo  = full_kv_offsets[csr_idx];
    uint full_hi  = full_kv_offsets[csr_idx + 1];
    uint part_lo  = partial_kv_offsets[csr_idx];
    uint part_hi  = partial_kv_offsets[csr_idx + 1];
    uint n_full   = full_hi - full_lo;
    uint n_part   = part_hi - part_lo;
    uint n_total  = n_full + n_part;
    uint per_split = (n_total + N_SPLITS - 1) / N_SPLITS;
    uint ix_begin = split * per_split;
    uint ix_end   = min(ix_begin + per_split, n_total);

    for (uint ix = ix_begin; ix < ix_end; ++ix) {
        uint p;
        bool is_partial = (ix >= n_full);
        uint partial_idx = 0;
        if (!is_partial) {
            p = full_kv_indices[full_lo + ix];
        } else {
            partial_idx = part_lo + (ix - n_full);
            p = partial_kv_indices[partial_idx];
        }

        const uint phys = bt_s[p];
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;

        // QK via simdgroup_matrix. Q_ROWS=16 rows → 2 MMA passes per K col
        // slab. PAGE=16 → 2 K col slabs (pb=0, 1).
        for (uint pb = 0; pb < PAGE / 8; ++pb) {
            device const half* pk = Kbase + (pb * 8) * kv_row_stride;
            for (uint qp = 0; qp < Q_ROWS / 8; ++qp) {
                simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                for (uint dt = 0; dt < D8; ++dt) {
                    simdgroup_half8x8 mq, mk;
                    simdgroup_load(mq, Q_tile + (qp * 8) * D + dt * 8, D);
                    simdgroup_load(mk, pk + dt * 8, kv_row_stride, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                }
                simdgroup_store(mqk, scores_tile + (qp * 8) * PAGE + pb * 8, PAGE);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row online softmax. For FULL blocks every element is kept; for
        // PARTIAL blocks we read a per-(q_local, block) bitmap produced by
        // the CPU mask-mod policy. Bit k of the per-row uint = 1 ⇒ keep, 0
        // ⇒ -∞. Rows past the real sequence end stay at -INF.
        if (lid < Q_ROWS) {
            const uint r = lid;
            const uint q_local = r / Q_PER_TG;
            const uint q_pos_in_seq = q_local_base + q_local;
            if (q_pos_in_seq < q_len) {
                uint mask_word = 0u;
                if (is_partial) {
                    // Partial blocks carry a Q_BLOCK-row bitmap. Within the
                    // tile, q_local = r / Q_PER_TG already collapses the Q_PER_TG
                    // head-grouping, so several rows share the same q_local
                    // and the same mask row.
                    mask_word = partial_block_masks[partial_idx * 8u + q_local];
                }
                float row_max = -INFINITY;
                float s_loc[PAGE];
                for (uint k = 0; k < PAGE; ++k) {
                    float sv = float(scores_tile[r * PAGE + k]) * qk_scale;
                    bool keep = is_partial ? (((mask_word >> k) & 1u) != 0u) : true;
                    if (!keep) sv = -INFINITY;
                    s_loc[k] = sv;
                    row_max = max(row_max, sv);
                }
                // Skip the online-softmax update when this block contributes
                // nothing for this Q row (all K masked). Otherwise we hit
                // `exp(m_old - m_new)` with both = -INFINITY → exp(NaN) = NaN
                // → poisons l_state → NaN in reduce. Happens for a Q row
                // whose full causal horizon lies in earlier pages and the
                // CSR split places a past-horizon partial page in this TG.
                if (row_max == -INFINITY) {
                    scale_tile[r] = 1.0f;
                    for (uint k = 0; k < PAGE; ++k) scores_tile[r * PAGE + k] = half(0);
                } else {
                    float m_old = m_state[r];
                    float m_new = max(m_old, row_max);
                    float scale = exp(m_old - m_new);
                    float sum = 0.0f;
                    for (uint k = 0; k < PAGE; ++k) {
                        float e = exp(s_loc[k] - m_new);
                        scores_tile[r * PAGE + k] = half(e);
                        sum += e;
                    }
                    m_state[r] = m_new;
                    l_state[r] = l_state[r] * scale + sum;
                    scale_tile[r] = scale;
                }
            } else {
                // zero out padding row's contribution this iteration
                scale_tile[r] = 1.0f;
                for (uint k = 0; k < PAGE; ++k) scores_tile[r * PAGE + k] = half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // AV: scalar cooperative. Each lane owns D/THREADS=8 dims.
        for (uint d = lid; d < D; d += THREADS) {
            half V_reg[PAGE];
            for (uint k = 0; k < PAGE; ++k) V_reg[k] = Vbase[k * kv_row_stride + d];
            for (uint r = 0; r < Q_ROWS; ++r) {
                float acc = O_acc[r * D + d] * scale_tile[r];
                for (uint k = 0; k < PAGE; ++k) {
                    acc += float(scores_tile[r * PAGE + k]) * float(V_reg[k]);
                }
                O_acc[r * D + d] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials per real (q_pos, q_head). Padding rows skipped.
    const bool empty_split = (ix_begin >= ix_end);
    for (uint r = 0; r < Q_ROWS; ++r) {
        const uint q_local = r / Q_PER_TG;
        const uint h_local = r % Q_PER_TG;
        const uint q_pos_in_seq = q_local_base + q_local;
        if (q_pos_in_seq >= q_len) continue;
        const uint q_head = q_head_base + h_local;
        const uint pidx = ((slot * q_len + q_pos_in_seq) * H_Q + q_head) * N_SPLITS + split;
        if (lid == 0) {
            m_partials[pidx] = empty_split ? -INFINITY : m_state[r];
            l_partials[pidx] = empty_split ? 0.0f : l_state[r];
        }
        device float* O_part = O_partials + pidx * D;
        for (uint d = lid; d < D; d += THREADS) O_part[d] = O_acc[r * D + d];
    }
}

// Flex attention v1 full-attention for prefill.
// llama.cpp-style geometry: ONE TG per (slot, q_head, q_block). Each TG owns
// Q_BLOCK=8 queries of a SINGLE q_head (not Q_PER_TG grouped like v0).
// That keeps the tg-mem budget below 32 KB at D=512:
//   Q_tile:      8 * 512 * 2 = 8192 B
//   scores_tile: 8 * 8 * 2   = 128 B
//   O_acc:       8 * 512 * 4 = 16384 B
//   m/l/scale:   8 * 12       = 96 B
//   q_pos_tg:    8 * 4         = 32 B
//   ---------------------------------
//   total:                     ~24.5 KB ✓
// Grid: (B * H_Q, q_blocks, N_SPLITS). kv_head = (q_head * H_KV) / H_Q is
// derived inside the kernel; multiple q_heads that share a KV head re-read
// the same K — on Apple Silicon with unified memory this hits L1/L2 cache.
// Mask: pure causal (full-attn layers have no sliding window).
kernel void flex_attn_full_prefill(
    device const half* Q                    [[buffer(0)]],
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],
    device float* m_partials                [[buffer(4)]],
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],
    device const uint* full_kv_offsets      [[buffer(7)]],
    device const uint* full_kv_indices      [[buffer(8)]],
    device const uint* partial_kv_offsets   [[buffer(9)]],
    device const uint* partial_kv_indices   [[buffer(10)]],
    device const uint* q_positions          [[buffer(11)]],
    device const uint* k_len_per_slot       [[buffer(12)]],
    constant float& qk_scale                [[buffer(13)]],
    constant uint& max_pages                [[buffer(14)]],
    constant uint& H_Q                      [[buffer(15)]],
    constant uint& H_KV                     [[buffer(16)]],
    constant uint& N_SPLITS                 [[buffer(17)]],
    constant uint& q_len                    [[buffer(18)]],
    device const uint* partial_block_masks  [[buffer(19)]],  // [total_partials, Q_BLOCK] uint
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 512;
    constexpr uint PAGE = 8;
    constexpr uint THREADS = 32;
    constexpr uint Q_BLOCK = 8;
    constexpr uint Q_ROWS = Q_BLOCK;           // one q_head per TG
    constexpr uint D8 = D / 8;                 // 64

    const uint vs = tg_pos.x;                  // 0..B*H_Q
    const uint q_block_idx = tg_pos.y;
    const uint split = tg_pos.z;
    const uint slot = vs / H_Q;
    const uint q_head = vs % H_Q;
    const uint kv_head = (q_head * H_KV) / H_Q;
    const uint lid = lid3.x;

    const uint q_blocks_per_slot = (q_len + Q_BLOCK - 1) / Q_BLOCK;
    const uint csr_idx = slot * q_blocks_per_slot + q_block_idx;
    const uint q_local_base = q_block_idx * Q_BLOCK;

    threadgroup half  Q_tile[Q_ROWS * D];
    threadgroup half  scores_tile[Q_ROWS * PAGE];
    threadgroup float O_acc[Q_ROWS * D];
    threadgroup float m_state[Q_ROWS];
    threadgroup float l_state[Q_ROWS];
    threadgroup float scale_tile[Q_ROWS];
    threadgroup uint  q_pos_tg[Q_BLOCK];

    device const uint* bt_s = block_table + slot * max_pages;

    // Load Q for this (slot, q_head) × [q_local_base, q_local_base+Q_BLOCK) rows.
    for (uint i = lid; i < Q_ROWS * D; i += THREADS) {
        uint r = i / D;
        uint q_pos_in_seq = q_local_base + r;
        if (q_pos_in_seq >= q_len) {
            Q_tile[i] = half(0);
        } else {
            uint q_flat = (slot * q_len + q_pos_in_seq) * H_Q + q_head;
            Q_tile[i] = Q[q_flat * D + (i % D)];
        }
    }
    for (uint i = lid; i < Q_ROWS * D; i += THREADS) O_acc[i] = 0.0f;
    if (lid < Q_ROWS) { m_state[lid] = -INFINITY; l_state[lid] = 0.0f; }
    if (lid < Q_BLOCK) {
        uint q_pos_in_seq = q_local_base + lid;
        q_pos_tg[lid] = (q_pos_in_seq < q_len)
                        ? q_positions[slot * q_len + q_pos_in_seq] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k_len = k_len_per_slot[slot];
    const uint kv_row_stride = H_KV * D;

    uint full_lo  = full_kv_offsets[csr_idx];
    uint full_hi  = full_kv_offsets[csr_idx + 1];
    uint part_lo  = partial_kv_offsets[csr_idx];
    uint part_hi  = partial_kv_offsets[csr_idx + 1];
    uint n_full   = full_hi - full_lo;
    uint n_part   = part_hi - part_lo;
    uint n_total  = n_full + n_part;
    uint per_split = (n_total + N_SPLITS - 1) / N_SPLITS;
    uint ix_begin = split * per_split;
    uint ix_end   = min(ix_begin + per_split, n_total);

    for (uint ix = ix_begin; ix < ix_end; ++ix) {
        uint p;
        bool is_partial = (ix >= n_full);
        uint partial_idx = 0;
        if (!is_partial) {
            p = full_kv_indices[full_lo + ix];
        } else {
            partial_idx = part_lo + (ix - n_full);
            p = partial_kv_indices[partial_idx];
        }

        const uint phys = bt_s[p];
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;

        // One MMA pass (Q_ROWS=8 fits 8x8) × PAGE/8=1 K slab × D8=64 d-tiles.
        simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        for (uint dt = 0; dt < D8; ++dt) {
            simdgroup_half8x8 mq, mk;
            simdgroup_load(mq, Q_tile + dt * 8, D);
            simdgroup_load(mk, Kbase + dt * 8, kv_row_stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }
        simdgroup_store(mqk, scores_tile, PAGE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row online softmax. FULL blocks keep every element (no mask
        // consulted); PARTIAL blocks read a Q_BLOCK-row bitmap produced by
        // the CPU mask-mod policy and zero-mask via -INF per cell.
        if (lid < Q_ROWS) {
            const uint r = lid;
            const uint q_pos_in_seq = q_local_base + r;
            if (q_pos_in_seq < q_len) {
                uint mask_word = 0u;
                if (is_partial) {
                    // Full-attention prefill: one q_head per TG ⇒ one row per
                    // q_pos_in_seq, so mask row index = r directly.
                    mask_word = partial_block_masks[partial_idx * 8u + r];
                }
                float row_max = -INFINITY;
                float s_loc[PAGE];
                for (uint k = 0; k < PAGE; ++k) {
                    float sv = float(scores_tile[r * PAGE + k]) * qk_scale;
                    bool keep = is_partial ? (((mask_word >> k) & 1u) != 0u) : true;
                    if (!keep) sv = -INFINITY;
                    s_loc[k] = sv;
                    row_max = max(row_max, sv);
                }
                // All-masked-this-block guard: skip online-softmax update so
                // we don't hit `exp(-INF - -INF)` = NaN when m_state is still
                // -INF from a split whose first page was fully past-horizon.
                if (row_max == -INFINITY) {
                    scale_tile[r] = 1.0f;
                    for (uint k = 0; k < PAGE; ++k) scores_tile[r * PAGE + k] = half(0);
                } else {
                    float m_old = m_state[r];
                    float m_new = max(m_old, row_max);
                    float scale = exp(m_old - m_new);
                    float sum = 0.0f;
                    for (uint k = 0; k < PAGE; ++k) {
                        float e = exp(s_loc[k] - m_new);
                        scores_tile[r * PAGE + k] = half(e);
                        sum += e;
                    }
                    m_state[r] = m_new;
                    l_state[r] = l_state[r] * scale + sum;
                    scale_tile[r] = scale;
                }
            } else {
                scale_tile[r] = 1.0f;
                for (uint k = 0; k < PAGE; ++k) scores_tile[r * PAGE + k] = half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // AV scalar cooperative: each lane owns D/THREADS=16 dims.
        for (uint d = lid; d < D; d += THREADS) {
            half V_reg[PAGE];
            for (uint k = 0; k < PAGE; ++k) V_reg[k] = Vbase[k * kv_row_stride + d];
            for (uint r = 0; r < Q_ROWS; ++r) {
                float acc = O_acc[r * D + d] * scale_tile[r];
                for (uint k = 0; k < PAGE; ++k) {
                    acc += float(scores_tile[r * PAGE + k]) * float(V_reg[k]);
                }
                O_acc[r * D + d] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials per (q_pos, q_head). Partial layout identical to v0:
    // pidx = ((slot * q_len + q_pos_in_seq) * H_Q + q_head) * N_SPLITS + split.
    const bool empty_split = (ix_begin >= ix_end);
    for (uint r = 0; r < Q_ROWS; ++r) {
        const uint q_pos_in_seq = q_local_base + r;
        if (q_pos_in_seq >= q_len) continue;
        const uint pidx = ((slot * q_len + q_pos_in_seq) * H_Q + q_head) * N_SPLITS + split;
        if (lid == 0) {
            m_partials[pidx] = empty_split ? -INFINITY : m_state[r];
            l_partials[pidx] = empty_split ? 0.0f : l_state[r];
        }
        device float* O_part = O_partials + pidx * D;
        for (uint d = lid; d < D; d += THREADS) O_part[d] = O_acc[r * D + d];
    }
}

// Full-attention cross-slot K/V broadcast kernel (shared-prefix AR).
// Mirrors paged_attn_slide_ar_shared but for D=512, PAGE=8 full-attn
// layers. tg-mem budget forces Q_PER_TG=1 (one q_head per TG); grid is
// (H_Q, 1, 1) = 16 TGs per layer, each broadcasting its q_head across
// all B active slots. Q_heads sharing a kv_head will all issue the same
// K/V load pattern — on Apple Silicon these hit L2 after the first load,
// so we still get most of the bandwidth benefit of true broadcast.
//
// tg-mem ≈ 28 KB at B=4: Q_tile 4 KB + K_stage 8 KB + V_stage 8 KB +
// O_acc 8 KB + small m/l state.
kernel void paged_attn_full_ar_shared(
    device const half* Q                    [[buffer(0)]],   // [B, H_Q, D]
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* shared_pages         [[buffer(3)]],   // [prefix_pages]
    device float* m_partials                [[buffer(4)]],
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],
    device const uint* k_len_per_slot       [[buffer(7)]],
    constant float& qk_scale                [[buffer(8)]],
    constant uint& H_Q                      [[buffer(9)]],
    constant uint& H_KV                     [[buffer(10)]],
    constant uint& N_SPLITS                 [[buffer(11)]],
    constant uint& prefix_pages             [[buffer(12)]],
    constant uint& B_batch                  [[buffer(13)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 512;
    constexpr uint PAGE = 8;
    constexpr uint THREADS = 128;
    constexpr uint MAX_B = 4;
    constexpr uint D8 = D / 8;

    const uint q_head = tg_pos.x;
    const uint kv_head = (q_head * H_KV) / H_Q;   // per GQA grouping
    const uint lid = lid3.x;

    threadgroup half  Q_tile[MAX_B * D];          // 4*512*2 = 4 KB
    threadgroup half  K_stage[PAGE * D];          // 8*512*2 = 8 KB
    threadgroup half  V_stage[PAGE * D];          // 8*512*2 = 8 KB
    threadgroup half  scores_tile[MAX_B * PAGE];  // 4*8*2 = 64 B
    threadgroup float O_acc[MAX_B * D];           // 4*512*4 = 8 KB
    threadgroup float m_state[MAX_B];
    threadgroup float l_state[MAX_B];
    threadgroup float scale_tile[MAX_B];

    // Load all active slots' Q[this q_head] into Q_tile
    for (uint i = lid; i < B_batch * D; i += THREADS) {
        uint b = i / D;
        uint d = i % D;
        Q_tile[i] = Q[(b * H_Q + q_head) * D + d];
    }
    for (uint i = lid; i < B_batch * D; i += THREADS) O_acc[i] = 0.0f;
    if (lid < B_batch) {
        m_state[lid] = -INFINITY;
        l_state[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kv_row_stride = H_KV * D;

    for (uint p = 0; p < prefix_pages; ++p) {
        const uint phys = shared_pages[p];
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;

        // Stage K & V into tg-mem (shared by all slots' compute below).
        for (uint i = lid; i < PAGE * D; i += THREADS) {
            uint k = i / D; uint d = i % D;
            K_stage[i] = Kbase[k * kv_row_stride + d];
        }
        for (uint i = lid; i < PAGE * D; i += THREADS) {
            uint k = i / D; uint d = i % D;
            V_stage[i] = Vbase[k * kv_row_stride + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Q@K for every slot. MAX_B=4 slots, each one 1×D × D×8 = 1×8 score row.
        // Pad Q to 8 rows for 8×8 MMA (slots beyond B_batch get all-zero Q).
        for (uint b = 0; b < B_batch; ++b) {
            simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            for (uint dt = 0; dt < D8; ++dt) {
                simdgroup_half8x8 mq, mk;
                simdgroup_load(mq, Q_tile + b * D + dt * 8, D);
                simdgroup_load(mk, K_stage + dt * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }
            // Only row 0 of the 8x8 MMA matters (Q_PER_TG=1 → 1 real Q row);
            // store full 8×8 to scores buffer and ignore rows 1..7.
            threadgroup half row_buf[8 * PAGE];
            simdgroup_store(mqk, row_buf, PAGE);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = lid; k < PAGE; k += THREADS) {
                scores_tile[b * PAGE + k] = row_buf[k];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Per-slot softmax with per-slot k_len mask.
        if (lid < B_batch) {
            uint b = lid;
            uint k_len = k_len_per_slot[b];
            float row_max = -INFINITY;
            float s_loc[PAGE];
            for (uint k = 0; k < PAGE; ++k) {
                float sv = float(scores_tile[b * PAGE + k]) * qk_scale;
                uint k_pos = p * PAGE + k;
                if (k_pos >= k_len) sv = -INFINITY;
                s_loc[k] = sv;
                row_max = max(row_max, sv);
            }
            float m_old = m_state[b];
            float m_new = max(m_old, row_max);
            float scale = exp(m_old - m_new);
            float sum = 0.0f;
            for (uint k = 0; k < PAGE; ++k) {
                float e = exp(s_loc[k] - m_new);
                scores_tile[b * PAGE + k] = half(e);
                sum += e;
            }
            m_state[b] = m_new;
            l_state[b] = l_state[b] * scale + sum;
            scale_tile[b] = scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // AV: each lane owns D/THREADS dims per slot.
        for (uint bd = lid; bd < B_batch * D; bd += THREADS) {
            uint b = bd / D;
            uint d = bd % D;
            float acc = O_acc[b * D + d] * scale_tile[b];
            for (uint k = 0; k < PAGE; ++k) {
                acc += float(scores_tile[b * PAGE + k])
                     * float(V_stage[k * D + d]);
            }
            O_acc[b * D + d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials at split=0 for each active slot.
    if (lid < B_batch) {
        uint b = lid;
        uint pidx = (b * H_Q + q_head) * N_SPLITS + 0;
        m_partials[pidx] = m_state[b];
        l_partials[pidx] = l_state[b];
    }
    for (uint bd = lid; bd < B_batch * D; bd += THREADS) {
        uint b = bd / D;
        uint d = bd % D;
        uint pidx = (b * H_Q + q_head) * N_SPLITS + 0;
        O_partials[pidx * D + d] = O_acc[b * D + d];
    }
}

// Flex attention v0 for full-attention layers: D=512, PAGE=8, Q_PER_TG=8,
// Q_BLOCK=1, mask_mod = causal (no sliding window — full layers are global).
// Same list-driven structure as the slide variant.
kernel void flex_attn_full_v0(
    device const half* Q                    [[buffer(0)]],
    device const half* K_cache              [[buffer(1)]],
    device const half* V_cache              [[buffer(2)]],
    device const uint* block_table          [[buffer(3)]],
    device float* m_partials                [[buffer(4)]],
    device float* l_partials                [[buffer(5)]],
    device float* O_partials                [[buffer(6)]],
    device const uint* full_kv_offsets      [[buffer(7)]],
    device const uint* full_kv_indices      [[buffer(8)]],
    device const uint* partial_kv_offsets   [[buffer(9)]],
    device const uint* partial_kv_indices   [[buffer(10)]],
    device const uint* k_len_per_slot       [[buffer(11)]],
    constant float& qk_scale                [[buffer(12)]],
    constant uint& max_pages                [[buffer(13)]],
    constant uint& H_Q                      [[buffer(14)]],
    constant uint& H_KV                     [[buffer(15)]],
    constant uint& N_SPLITS                 [[buffer(16)]],
    constant uint& prefix_pages             [[buffer(17)]],    // skip pages < this (tail mode)
    constant uint& split_offset             [[buffer(18)]],    // write at split_offset+tg.y in output layout
    constant uint& total_splits_out         [[buffer(19)]],    // output layout stride (0 → fallback to N_SPLITS)
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 512;
    constexpr uint PAGE = 8;
    constexpr uint THREADS = 32;
    constexpr uint Q_PER_TG = 8;
    constexpr uint D8 = D / 8;           // 64

    const uint vs = tg_pos.x;
    const uint split = tg_pos.y;
    const uint slot = vs / H_KV;
    const uint kv_head = vs % H_KV;
    const uint q_head_base = kv_head * Q_PER_TG;
    const uint lid = lid3.x;

    const uint csr_idx = slot;

    threadgroup half  Q_tile[Q_PER_TG * D];
    threadgroup half  scores_tile[Q_PER_TG * PAGE];
    threadgroup float O_acc[Q_PER_TG * D];
    threadgroup float m_state[Q_PER_TG];
    threadgroup float l_state[Q_PER_TG];
    threadgroup float scale_tile[Q_PER_TG];

    device const half* Qbase = Q + (slot * H_Q + q_head_base) * D;
    device const uint* bt_s = block_table + slot * max_pages;

    for (uint i = lid; i < Q_PER_TG * D; i += THREADS) Q_tile[i] = Qbase[i];
    for (uint i = lid; i < Q_PER_TG * D; i += THREADS) O_acc[i] = 0.0f;
    if (lid < Q_PER_TG) { m_state[lid] = -INFINITY; l_state[lid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k_len = k_len_per_slot[slot];
    const uint kv_row_stride = H_KV * D;

    uint full_lo  = full_kv_offsets[csr_idx];
    uint full_hi  = full_kv_offsets[csr_idx + 1];
    uint part_lo  = partial_kv_offsets[csr_idx];
    uint part_hi  = partial_kv_offsets[csr_idx + 1];
    uint n_full   = full_hi - full_lo;
    uint n_part   = part_hi - part_lo;
    uint n_total  = n_full + n_part;
    uint per_split = (n_total + N_SPLITS - 1) / N_SPLITS;
    uint ix_begin = split * per_split;
    uint ix_end   = min(ix_begin + per_split, n_total);

    for (uint ix = ix_begin; ix < ix_end; ++ix) {
        uint p;
        bool is_partial;
        if (ix < n_full) {
            p = full_kv_indices[full_lo + ix];
            is_partial = false;
        } else {
            p = partial_kv_indices[part_lo + (ix - n_full)];
            is_partial = true;
        }
        // Tail mode: skip logical pages owned by the full-attn shared-prefix kernel.
        if (p < prefix_pages) continue;

        const uint phys = bt_s[p];
        device const half* Kbase = K_cache + (phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = V_cache + (phys * PAGE * H_KV + kv_head) * D;

        // QK: one 8x8 MMA per d-tile; 64 d-tiles at D=512. Q=8 real rows.
        simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        for (uint dt = 0; dt < D8; ++dt) {
            simdgroup_half8x8 mq, mk;
            simdgroup_load(mq, Q_tile + dt * 8, D);
            simdgroup_load(mk, Kbase + dt * 8, kv_row_stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }
        simdgroup_store(mqk, scores_tile, PAGE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row online softmax. FULL blocks skip the per-k predicate.
        if (lid < Q_PER_TG) {
            const uint q = lid;
            float row_max = -INFINITY;
            float s_loc[PAGE];
            if (is_partial) {
                for (uint k = 0; k < PAGE; ++k) {
                    float sv = float(scores_tile[q * PAGE + k]) * qk_scale;
                    uint k_pos = p * PAGE + k;
                    if (k_pos >= k_len) sv = -INFINITY;
                    s_loc[k] = sv;
                    row_max = max(row_max, sv);
                }
            } else {
                for (uint k = 0; k < PAGE; ++k) {
                    float sv = float(scores_tile[q * PAGE + k]) * qk_scale;
                    s_loc[k] = sv;
                    row_max = max(row_max, sv);
                }
            }
            float m_old = m_state[q];
            float m_new = max(m_old, row_max);
            float scale = exp(m_old - m_new);
            float sum = 0.0f;
            for (uint k = 0; k < PAGE; ++k) {
                float e = exp(s_loc[k] - m_new);
                scores_tile[q * PAGE + k] = half(e);
                sum += e;
            }
            m_state[q] = m_new;
            l_state[q] = l_state[q] * scale + sum;
            scale_tile[q] = scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // AV: scalar cooperative.
        for (uint d = lid; d < D; d += THREADS) {
            half V_reg[PAGE];
            for (uint k = 0; k < PAGE; ++k) {
                V_reg[k] = Vbase[k * kv_row_stride + d];
            }
            for (uint q = 0; q < Q_PER_TG; ++q) {
                float acc = O_acc[q * D + d] * scale_tile[q];
                for (uint k = 0; k < PAGE; ++k) {
                    acc += float(scores_tile[q * PAGE + k]) * float(V_reg[k]);
                }
                O_acc[q * D + d] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint out_stride = (total_splits_out > 0) ? total_splits_out : N_SPLITS;
    const bool empty_split = (ix_begin >= ix_end);
    for (uint q = 0; q < Q_PER_TG; ++q) {
        const uint q_head = q_head_base + q;
        const uint pidx = (slot * H_Q + q_head) * out_stride + split_offset + split;
        if (lid == 0) {
            m_partials[pidx] = empty_split ? -INFINITY : m_state[q];
            l_partials[pidx] = empty_split ? 0.0f : l_state[q];
        }
        device float* O_part = O_partials + pidx * D;
        for (uint d = lid; d < D; d += THREADS) O_part[d] = O_acc[q * D + d];
    }
}

// =======================================================================
// fp32-path vision encoder kernels.
//
// Why: Gemma-4's vision tower has outlier channels (dim 213 hits 2934 at
// layer 26, dims 294/366/989/etc. are secondary hotspots) trained in bf16
// (fp32 exponent range). Our fp16 intermediate buffers lose too much
// precision on these channels — per-layer MSE doubles past layer 12,
// reaching ~28% relative noise at layer 26, enough to destroy LM image
// understanding downstream. Promoting intermediate buffers to fp32 fixes
// the precision floor at the cost of ~2× memory per intermediate buffer
// (tiny compared to the already-fp32 residual stream).
//
// These kernels mirror their fp16 counterparts 1-for-1, just with fp32
// I/O. No algorithmic changes.
// =======================================================================

// RMSNorm with gamma, fp32 in/out.
kernel void rms_norm_fp32(
    device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
    device const half* gamma [[buffer(2)]],
    constant uint& D [[buffer(3)]], constant float& eps [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    float s = 0.0f;
    for (uint i = t; i < D; i += 32) { float v = x[b*D+i]; s += v*v; }
    s = simd_sum(s);
    float sc = rsqrt(s/float(D) + eps);
    for (uint i = t; i < D; i += 32) { y[b*D+i] = x[b*D+i] * sc * float(gamma[i]); }
}

// RMSNorm no-scale, fp32 in/out.
kernel void rms_norm_noscale_fp32(
    device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
    constant uint& D [[buffer(2)]], constant float& eps [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    float s = 0.0f;
    for (uint i = t; i < D; i += 32) { float v = x[b*D+i]; s += v*v; }
    s = simd_sum(s);
    float sc = rsqrt(s/float(D) + eps);
    for (uint i = t; i < D; i += 32) { y[b*D+i] = x[b*D+i] * sc; }
}

// Dense GEMV with fp32 input, fp16 weights, fp32 output. Mirrors
// dense_gemv_fp16_v5's SG-split-K pattern but reads/writes fp32 vectors.
kernel void dense_gemv_fp32in_fp32out_v5(
    device const float* x               [[buffer(0)]],
    device const half*  W               [[buffer(1)]],
    device float*       output          [[buffer(2)]],
    constant uint& D_in                 [[buffer(3)]],
    constant uint& D_out                [[buffer(4)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]])
{
    constexpr uint N_SPLITS = 4;
    uint n_block = tg.x; uint b = tg.y; uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint k_per_sg = D_in / N_SPLITS;
    uint k_begin = sg_id * k_per_sg;
    uint k_end = k_begin + k_per_sg;
    float acc = 0.0f;
    device const float* x_b = x + b * D_in;
    device const half*  W_row = W + n * D_in;
    for (uint k = k_begin; k < k_end; ++k) {
        acc += x_b[k] * float(W_row[k]);
    }
    threadgroup float partials[N_SPLITS][32];
    partials[sg_id][lid_sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        output[b * D_out + n] = partials[0][lid_sg] + partials[1][lid_sg]
                              + partials[2][lid_sg] + partials[3][lid_sg];
    }
}

// 2D NeoX RoPE in-place on an fp32 buffer. Same math as the fp16 variant
// vision_2d_rope_neox_fp16 — first half of head-dim rotated by pos_x,
// second half by pos_y, NeoX pairs are (i, i+quarter) within each half.
kernel void vision_2d_rope_neox_fp32(
    device float* x                     [[buffer(0)]],
    device const uint* pos_x            [[buffer(1)]],
    device const uint* pos_y            [[buffer(2)]],
    constant uint& N                    [[buffer(3)]],
    constant uint& H                    [[buffer(4)]],
    constant uint& HD                   [[buffer(5)]],
    constant float& theta               [[buffer(6)]],
    uint3 tg                            [[threadgroup_position_in_grid]],
    uint3 lid                           [[thread_position_in_threadgroup]])
{
    uint token = tg.x; uint head = tg.y; uint t = lid.x;
    uint half_dim = HD / 2;
    uint quarter = half_dim / 2;
    device float* vec = x + (token * H + head) * HD;

    uint px = pos_x[token];
    for (uint i = t; i < quarter; i += 32) {
        float freq = 1.0f / pow(theta, 2.0f * float(i) / float(half_dim));
        float ang = float(px) * freq;
        float c = cos(ang), s = sin(ang);
        float a = vec[i];
        float b = vec[i + quarter];
        vec[i]           = a * c - b * s;
        vec[i + quarter] = a * s + b * c;
    }
    uint py = pos_y[token];
    for (uint i = t; i < quarter; i += 32) {
        float freq = 1.0f / pow(theta, 2.0f * float(i) / float(half_dim));
        float ang = float(py) * freq;
        float c = cos(ang), s = sin(ang);
        float a = vec[half_dim + i];
        float b = vec[half_dim + i + quarter];
        vec[half_dim + i]           = a * c - b * s;
        vec[half_dim + i + quarter] = a * s + b * c;
    }
}

// Bidirectional attention (no causal mask), fp32 Q/K/V/O. One TG per
// (query token, head). Same N_MAX tg-mem scratch as the fp16 variant.
kernel void vision_attn_prefill_fp32(
    device const float* Q               [[buffer(0)]],
    device const float* K               [[buffer(1)]],
    device const float* V               [[buffer(2)]],
    device float* O                     [[buffer(3)]],
    constant uint& N                    [[buffer(4)]],
    constant uint& H                    [[buffer(5)]],
    constant uint& HD                   [[buffer(6)]],
    constant float& qk_scale            [[buffer(7)]],
    device const uchar* padding_mask    [[buffer(8)]],   // 1 byte per K pos; 1 = padded, mask out of softmax
    constant uint& use_padding_mask     [[buffer(9)]],   // non-zero → consult padding_mask
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint THREADS = 32;
    constexpr uint N_MAX = 3072;
    threadgroup float scores[N_MAX];
    threadgroup float state[3];

    uint q_tok = tg.x; uint head = tg.y; uint t = lid.x;
    device const float* Q_qh = Q + (q_tok * H + head) * HD;
    device const float* K_h  = K + head * HD;
    device const float* V_h  = V + head * HD;
    uint K_stride = H * HD;

    // Pass 1: Q@K with optional -INF for masked-out K positions.
    for (uint k = t; k < N; k += THREADS) {
        float s = 0;
        device const float* K_k = K_h + k * K_stride;
        for (uint d = 0; d < HD; ++d) { s += Q_qh[d] * K_k[d]; }
        float score = s * qk_scale;
        if (use_padding_mask != 0u && padding_mask[k] != 0u) {
            score = -INFINITY;
        }
        scores[k] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint k = t; k < N; k += THREADS) local_max = max(local_max, scores[k]);
    float row_max = simd_max(local_max);
    if (t == 0) state[0] = row_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    row_max = state[0];

    float local_sum = 0;
    for (uint k = t; k < N; k += THREADS) {
        float e = exp(scores[k] - row_max);
        scores[k] = e;
        local_sum += e;
    }
    float row_sum = simd_sum(local_sum);
    if (t == 0) state[1] = row_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / state[1];

    device float* O_qh = O + (q_tok * H + head) * HD;
    for (uint d = t; d < HD; d += THREADS) {
        float acc = 0;
        for (uint k = 0; k < N; ++k) { acc += scores[k] * V_h[k * K_stride + d]; }
        O_qh[d] = acc * inv_sum;
    }
}

// Batched MMA-based GEMM for the vision tower's dense layers. X [B, D_in]
// fp32, W [D_out, D_in] fp16 (row-major: each row is one output neuron's
// weights), Y [B, D_out] fp32. Uses simdgroup_matrix<half, 8, 8> MMA with
// fp32 accumulator. Each TG owns an 8-token × 8-output tile and iterates
// over D_in in 8-sized K-tiles; the weight for this (o_block) is loaded
// ONCE per TG rather than 2520 times (once per batch element) as the
// old dense_gemv_fp32in_fp32out_v5 does. Expected speedup at vision
// shapes: ~15× on 1152×1152 projections, more on the 4304-expanded FFN.
//
// Grid: (ceil(D_out/8), ceil(B/8)). 32 threads per TG.
// D_in and D_out must be multiples of 8.
kernel void vision_gemm_fp32_mma(
    device const float* X               [[buffer(0)]],   // [B, D_in]
    device const half*  W               [[buffer(1)]],   // [D_out, D_in]
    device float*       Y               [[buffer(2)]],   // [B, D_out]
    constant uint& B_count              [[buffer(3)]],
    constant uint& D_in                 [[buffer(4)]],
    constant uint& D_out                [[buffer(5)]],
    constant uint& quant_out            [[buffer(6)]],   // 0=fp32, 1=bf16-rounded fp32
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint Q_TILE  = 8;
    constexpr uint K_TILE  = 8;
    constexpr uint O_TILE  = 8;
    constexpr uint THREADS = 32;

    const uint o_block = tg.x;
    const uint q_block = tg.y;
    const uint o_start = o_block * O_TILE;
    const uint q_start = q_block * Q_TILE;
    const uint lid0    = lid.x;

    // Staging tiles: small, both in tg-mem.
    threadgroup half  x_stage[Q_TILE * K_TILE];  // 128 B
    threadgroup half  w_stage[K_TILE * O_TILE];  // 128 B
    threadgroup float y_stage[Q_TILE * O_TILE];  // 256 B

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < D_in; k += K_TILE) {
        // Stage X[q_start..q_start+8, k..k+8] as half.
        for (uint i = lid0; i < Q_TILE * K_TILE; i += THREADS) {
            uint q = i / K_TILE; uint kk = i % K_TILE;
            uint q_abs = q_start + q;
            uint k_abs = k + kk;
            x_stage[i] = (q_abs < B_count && k_abs < D_in)
                ? half(X[q_abs * D_in + k_abs]) : half(0);
        }
        // Stage W-tile: w_stage[kk, oo] = W[o_start+oo, k+kk].
        // (Transposed layout feeds simdgroup_multiply_accumulate(C, A=X, B=W^T, C).)
        for (uint i = lid0; i < K_TILE * O_TILE; i += THREADS) {
            uint kk = i / O_TILE; uint oo = i % O_TILE;
            uint o_abs = o_start + oo;
            uint k_abs = k + kk;
            w_stage[i] = (o_abs < D_out && k_abs < D_in)
                ? W[o_abs * D_in + k_abs] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 mx, mw;
        simdgroup_load(mx, x_stage, K_TILE);
        simdgroup_load(mw, w_stage, O_TILE);
        simdgroup_multiply_accumulate(acc, mx, mw, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, y_stage, O_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid0; i < Q_TILE * O_TILE; i += THREADS) {
        uint q = i / O_TILE; uint oo = i % O_TILE;
        uint q_abs = q_start + q;
        uint o_abs = o_start + oo;
        if (q_abs < B_count && o_abs < D_out) {
            float v = y_stage[i];
            if (quant_out != 0u) {
                uint bits = as_type<uint>(v);
                uint lsb  = (bits >> 16) & 1u;
                v = as_type<float>((bits + 0x7FFFu + lsb) & 0xFFFF0000u);
            }
            Y[q_abs * D_out + o_abs] = v;
        }
    }
}

// 16×16 variant of vision_gemm_fp32_mma with four simdgroup_float8x8
// accumulators per TG. Quadruples the work per barrier/stage pair and
// doubles arithmetic intensity over v1 (staging cost 2× but arithmetic
// 4×). Target: close the 0.93→≥3 TFLOPS gap measured on the v1 kernel.
//
// Layout: each TG owns a 16-token × 16-output output tile, computed as
// a 2×2 grid of 8×8 fp32 accumulators. X staged at [16, 8] half; W
// staged at [8, 16] half. Same K_TILE=8 stride as v1.
//
// Grid: (ceil(D_out/16), ceil(B/16)). 32 threads per TG (single simdgroup).
// D_in and D_out should be multiples of 8 (tail-safe at 16-boundary via
// the B_count/D_out checks at store time; unused tile entries zero out
// during staging).
kernel void vision_gemm_fp32_mma_v2(
    device const float* X               [[buffer(0)]],   // [B, D_in]
    device const half*  W               [[buffer(1)]],   // [D_out, D_in]
    device float*       Y               [[buffer(2)]],   // [B, D_out]
    constant uint& B_count              [[buffer(3)]],
    constant uint& D_in                 [[buffer(4)]],
    constant uint& D_out                [[buffer(5)]],
    constant uint& quant_out            [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint Q_TILE  = 16;
    constexpr uint K_TILE  = 8;
    constexpr uint O_TILE  = 16;
    constexpr uint THREADS = 32;

    const uint o_block = tg.x;
    const uint q_block = tg.y;
    const uint o_start = o_block * O_TILE;
    const uint q_start = q_block * Q_TILE;
    const uint lid0    = lid.x;

    threadgroup half  x_stage[Q_TILE * K_TILE];   // 256 B (16×8 half)
    threadgroup half  w_stage[K_TILE * O_TILE];   // 256 B (8×16 half)
    threadgroup float y_stage[Q_TILE * O_TILE];   // 1024 B (16×16 fp32)

    // Four 8×8 fp32 accumulators arranged as a 2×2 grid:
    //   acc[qi][oi] corresponds to output tile rows [qi*8..qi*8+8),
    //   cols [oi*8..oi*8+8).
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < D_in; k += K_TILE) {
        // Stage X[q_start..q_start+16, k..k+8] as half.
        for (uint i = lid0; i < Q_TILE * K_TILE; i += THREADS) {
            uint q = i / K_TILE; uint kk = i % K_TILE;
            uint q_abs = q_start + q;
            uint k_abs = k + kk;
            x_stage[i] = (q_abs < B_count && k_abs < D_in)
                ? half(X[q_abs * D_in + k_abs]) : half(0);
        }
        // Stage W[o_start..o_start+16, k..k+8] transposed to [kk, oo] for
        // simdgroup_multiply_accumulate(C = A @ B^T).
        for (uint i = lid0; i < K_TILE * O_TILE; i += THREADS) {
            uint kk = i / O_TILE; uint oo = i % O_TILE;
            uint o_abs = o_start + oo;
            uint k_abs = k + kk;
            w_stage[i] = (o_abs < D_out && k_abs < D_in)
                ? W[o_abs * D_in + k_abs] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load 2 Q tiles (rows 0-7, 8-15) × 2 W tiles (cols 0-7, 8-15).
        simdgroup_half8x8 mx0, mx1, mw0, mw1;
        simdgroup_load(mx0, x_stage,               K_TILE);
        simdgroup_load(mx1, x_stage + 8 * K_TILE,  K_TILE);
        simdgroup_load(mw0, w_stage,               O_TILE);
        simdgroup_load(mw1, w_stage + 8,           O_TILE);

        // Four MMAs — 2× arithmetic per barrier relative to v1's single MMA.
        simdgroup_multiply_accumulate(acc00, mx0, mw0, acc00);
        simdgroup_multiply_accumulate(acc01, mx0, mw1, acc01);
        simdgroup_multiply_accumulate(acc10, mx1, mw0, acc10);
        simdgroup_multiply_accumulate(acc11, mx1, mw1, acc11);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store 4 accumulators into the 16×16 y_stage.
    simdgroup_store(acc00, y_stage,                         O_TILE);
    simdgroup_store(acc01, y_stage + 8,                     O_TILE);
    simdgroup_store(acc10, y_stage + 8 * O_TILE,            O_TILE);
    simdgroup_store(acc11, y_stage + 8 * O_TILE + 8,        O_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write 16×16 tile to Y with bounds check + optional bf16 quant fold.
    for (uint i = lid0; i < Q_TILE * O_TILE; i += THREADS) {
        uint q = i / O_TILE; uint oo = i % O_TILE;
        uint q_abs = q_start + q;
        uint o_abs = o_start + oo;
        if (q_abs < B_count && o_abs < D_out) {
            float v = y_stage[i];
            if (quant_out != 0u) {
                uint bits = as_type<uint>(v);
                uint lsb  = (bits >> 16) & 1u;
                v = as_type<float>((bits + 0x7FFFu + lsb) & 0xFFFF0000u);
            }
            Y[q_abs * D_out + o_abs] = v;
        }
    }
}

// v3: 16×16 output tile with K-unroll=2. Stages a 16×16 X-tile and a
// 16×16 W-tile per outer iteration (32 K elements in each direction
// doubled: Q×K_CHUNK for X, K_CHUNK×O for W), then issues 2 inner
// MMA passes (8 MMAs total) before the next barrier. Halves the
// barrier count vs v2 (1152/16 = 72 barriers vs 144) while keeping
// the same 1-simdgroup / 32-thread shape.
//
// D_in must be a multiple of K_CHUNK=16. For vision this holds for all
// projections (1152 and 4304 both divisible by 16).
kernel void vision_gemm_fp32_mma_v3(
    device const float* X               [[buffer(0)]],
    device const half*  W               [[buffer(1)]],
    device float*       Y               [[buffer(2)]],
    constant uint& B_count              [[buffer(3)]],
    constant uint& D_in                 [[buffer(4)]],
    constant uint& D_out                [[buffer(5)]],
    constant uint& quant_out            [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint Q_TILE  = 16;
    constexpr uint K_TILE  = 8;
    constexpr uint K_UNROLL = 2;
    constexpr uint K_CHUNK = K_TILE * K_UNROLL;   // 16
    constexpr uint O_TILE  = 16;
    constexpr uint THREADS = 32;

    const uint o_block = tg.x;
    const uint q_block = tg.y;
    const uint o_start = o_block * O_TILE;
    const uint q_start = q_block * Q_TILE;
    const uint lid0    = lid.x;

    threadgroup half  x_stage[Q_TILE * K_CHUNK];   // 16×16 half = 512 B
    threadgroup half  w_stage[K_CHUNK * O_TILE];   // 16×16 half = 512 B
    threadgroup float y_stage[Q_TILE * O_TILE];    // 16×16 fp32 = 1024 B

    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < D_in; k += K_CHUNK) {
        // Stage X[q_start..q_start+16, k..k+16] as half (256 halves).
        for (uint i = lid0; i < Q_TILE * K_CHUNK; i += THREADS) {
            uint q = i / K_CHUNK; uint kk = i % K_CHUNK;
            uint q_abs = q_start + q;
            uint k_abs = k + kk;
            x_stage[i] = (q_abs < B_count && k_abs < D_in)
                ? half(X[q_abs * D_in + k_abs]) : half(0);
        }
        // Stage W transposed: w_stage[kk, oo] = W[o_start+oo, k+kk].
        for (uint i = lid0; i < K_CHUNK * O_TILE; i += THREADS) {
            uint kk = i / O_TILE; uint oo = i % O_TILE;
            uint o_abs = o_start + oo;
            uint k_abs = k + kk;
            w_stage[i] = (o_abs < D_out && k_abs < D_in)
                ? W[o_abs * D_in + k_abs] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner K-loop: K_UNROLL=2 passes, 4 MMAs each = 8 MMAs per barrier.
        #pragma unroll
        for (uint kk = 0; kk < K_UNROLL; ++kk) {
            simdgroup_half8x8 mx0, mx1, mw0, mw1;
            // X leading-dim = K_CHUNK; step 8 along K per inner pass.
            simdgroup_load(mx0, x_stage + kk * 8,                   K_CHUNK);
            simdgroup_load(mx1, x_stage + 8 * K_CHUNK + kk * 8,     K_CHUNK);
            // W leading-dim = O_TILE; step (8 * O_TILE) rows per inner pass.
            simdgroup_load(mw0, w_stage + kk * 8 * O_TILE,          O_TILE);
            simdgroup_load(mw1, w_stage + kk * 8 * O_TILE + 8,      O_TILE);

            simdgroup_multiply_accumulate(acc00, mx0, mw0, acc00);
            simdgroup_multiply_accumulate(acc01, mx0, mw1, acc01);
            simdgroup_multiply_accumulate(acc10, mx1, mw0, acc10);
            simdgroup_multiply_accumulate(acc11, mx1, mw1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc00, y_stage,                         O_TILE);
    simdgroup_store(acc01, y_stage + 8,                     O_TILE);
    simdgroup_store(acc10, y_stage + 8 * O_TILE,            O_TILE);
    simdgroup_store(acc11, y_stage + 8 * O_TILE + 8,        O_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid0; i < Q_TILE * O_TILE; i += THREADS) {
        uint q = i / O_TILE; uint oo = i % O_TILE;
        uint q_abs = q_start + q;
        uint o_abs = o_start + oo;
        if (q_abs < B_count && o_abs < D_out) {
            float v = y_stage[i];
            if (quant_out != 0u) {
                uint bits = as_type<uint>(v);
                uint lsb  = (bits >> 16) & 1u;
                v = as_type<float>((bits + 0x7FFFu + lsb) & 0xFFFF0000u);
            }
            Y[q_abs * D_out + o_abs] = v;
        }
    }
}

// Flash-attention variant of vision_attn_prefill_fp32. Bidirectional (no
// causal mask), optional padding mask, per-(head, q_block=8) threadgroup,
// online softmax, simdgroup_matrix<half, 8, 8> MMA for QK accumulated into
// fp32 (simdgroup_float8x8). AV remains scalar-cooperative (8 K positions
// per tile is small enough that the MMA overhead doesn't pay off — same
// tradeoff as flex_attn_slide_v1_q8 on the LM side).
//
// Inputs are fp32; the kernel converts each K/V tile to half in tg-mem
// once per iteration, which is the cheap part. QK then runs as 9 MMA
// instructions per K tile (D=72 → D8=9). Per-(head, q_block) TG wall is
// ~15 ms on M5 at N=2520, vs ~100 ms for the scalar kernel — 6-8× speedup
// on the attention portion of the vision forward.
//
// Grid: (H, ceil(N/8)). THREADS per TG: 32 (one simdgroup).
// HD must equal 72 (vision head dim). N can be up to 2520.
kernel void vision_attn_flash_fp32(
    device const float* Q               [[buffer(0)]],   // [B, N, H, D]
    device const float* K               [[buffer(1)]],   // [B, N, H, D]
    device const float* V               [[buffer(2)]],   // [B, N, H, D]
    device float* O                     [[buffer(3)]],   // [B, N, H, D]
    constant uint& N                    [[buffer(4)]],
    constant uint& H                    [[buffer(5)]],
    constant uint& HD                   [[buffer(6)]],
    constant float& qk_scale            [[buffer(7)]],
    device const uchar* padding_mask    [[buffer(8)]],   // [B, N]
    constant uint& use_padding_mask     [[buffer(9)]],
    uint3 tg_in                         [[threadgroup_position_in_grid]],
    uint3 lid_in                        [[thread_position_in_threadgroup]])
{
    constexpr uint Q_BLOCK = 8;
    constexpr uint K_BLOCK = 8;
    constexpr uint D       = 72;          // vision HD
    constexpr uint D8      = D / 8;       // 9 MMA d-tiles
    constexpr uint THREADS = 32;

    const uint head    = tg_in.x;
    const uint q_block = tg_in.y;
    const uint b       = tg_in.z;              // batch slot
    const uint q_start = q_block * Q_BLOCK;
    const uint lid     = lid_in.x;

    // Offset to this batch slot's Q/K/V/O and padding mask.
    // Slot-parallel pattern: each slot's K/V lives in its own [N,H,D] region
    // so cross-slot attention is impossible by construction (no mask needed).
    const uint slot_stride = N * H * D;
    device const float* Qb = Q + b * slot_stride;
    device const float* Kb = K + b * slot_stride;
    device const float* Vb = V + b * slot_stride;
    device       float* Ob = O + b * slot_stride;
    device const uchar* maskb = padding_mask + b * N;

    // tg-mem budget: 1152 + 576 + 576 + 128 + 2304 + 32 + 32 + 256 = ~5 KB.
    threadgroup half  Q_tile[Q_BLOCK * D];
    threadgroup half  K_stage[K_BLOCK * D];
    threadgroup half  V_stage[K_BLOCK * D];
    threadgroup half  scores_tile[Q_BLOCK * K_BLOCK];
    threadgroup float O_acc[Q_BLOCK * D];
    threadgroup float m_state[Q_BLOCK];
    threadgroup float l_state[Q_BLOCK];
    threadgroup float scale_tile[Q_BLOCK];
    threadgroup float scores_raw[Q_BLOCK * K_BLOCK];

    // Load Q_tile once (fp32 → half).
    for (uint i = lid; i < Q_BLOCK * D; i += THREADS) {
        uint q = i / D; uint d = i % D;
        uint q_abs = q_start + q;
        Q_tile[i] = half(q_abs < N ? Qb[(q_abs * H + head) * D + d] : 0.0f);
    }
    for (uint i = lid; i < Q_BLOCK * D; i += THREADS) O_acc[i] = 0.0f;
    if (lid < Q_BLOCK) { m_state[lid] = -INFINITY; l_state[lid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate K in blocks of 8.
    for (uint k_start = 0; k_start < N; k_start += K_BLOCK) {
        // Stage K and V as half in tg-mem. Out-of-range rows zero-padded.
        for (uint i = lid; i < K_BLOCK * D; i += THREADS) {
            uint kr = i / D; uint d = i % D;
            uint k_abs = k_start + kr;
            float kv = 0.0f, vv = 0.0f;
            if (k_abs < N) {
                kv = Kb[(k_abs * H + head) * D + d];
                vv = Vb[(k_abs * H + head) * D + d];
            }
            K_stage[i] = half(kv);
            V_stage[i] = half(vv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QK via MMA. D_MMA tiles × K_BLOCK × Q_BLOCK = 9 × 8 × 8 ≈ 576 FMA
        // equivalents folded into 9 MMA instructions per head.
        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint dt = 0; dt < D8; ++dt) {
            simdgroup_half8x8 mq, mk;
            simdgroup_load(mq, Q_tile + dt * 8, D);
            simdgroup_load(mk, K_stage + dt * 8, D, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }
        simdgroup_store(mqk, scores_raw, K_BLOCK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row online softmax (one lane per Q row).
        if (lid < Q_BLOCK) {
            uint r = lid;
            float row_max = -INFINITY;
            float s_loc[K_BLOCK];
            for (uint k = 0; k < K_BLOCK; ++k) {
                uint k_abs = k_start + k;
                float sv = scores_raw[r * K_BLOCK + k] * qk_scale;
                bool masked = (k_abs >= N) ||
                              (use_padding_mask != 0u && maskb[k_abs] != 0u);
                if (masked) sv = -INFINITY;
                s_loc[k] = sv;
                row_max = max(row_max, sv);
            }
            if (row_max == -INFINITY) {
                // All K in this tile masked out for this Q row — skip update
                // (avoids exp(-INF - -INF) = NaN propagation).
                scale_tile[r] = 1.0f;
                for (uint k = 0; k < K_BLOCK; ++k) scores_tile[r * K_BLOCK + k] = half(0);
            } else {
                float m_old = m_state[r];
                float m_new = max(m_old, row_max);
                float scale = exp(m_old - m_new);
                float sum = 0.0f;
                for (uint k = 0; k < K_BLOCK; ++k) {
                    float e = exp(s_loc[k] - m_new);
                    scores_tile[r * K_BLOCK + k] = half(e);
                    sum += e;
                }
                m_state[r] = m_new;
                l_state[r] = l_state[r] * scale + sum;
                scale_tile[r] = scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // AV scalar cooperative: each lane owns D/THREADS ≈ 2 dims.
        for (uint d = lid; d < D; d += THREADS) {
            half V_reg[K_BLOCK];
            for (uint k = 0; k < K_BLOCK; ++k) V_reg[k] = V_stage[k * D + d];
            for (uint r = 0; r < Q_BLOCK; ++r) {
                float acc = O_acc[r * D + d] * scale_tile[r];
                for (uint k = 0; k < K_BLOCK; ++k) {
                    acc += float(scores_tile[r * K_BLOCK + k]) * float(V_reg[k]);
                }
                O_acc[r * D + d] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize O_acc / l_state and write to output.
    for (uint i = lid; i < Q_BLOCK * D; i += THREADS) {
        uint q = i / D; uint d = i % D;
        uint q_abs = q_start + q;
        if (q_abs < N) {
            float inv_l = (l_state[q] > 0.0f) ? (1.0f / l_state[q]) : 0.0f;
            Ob[(q_abs * H + head) * D + d] = O_acc[i] * inv_l;
        }
    }
}

// GELU(gate) * up → gate, fp32 in-place on gate, reads fp32 up.
kernel void gelu_mul_fp32(
    device float* gate                  [[buffer(0)]],
    device const float* up              [[buffer(1)]],
    constant uint& N                    [[buffer(2)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    device float* g = gate + b * N;
    device const float* u = up + b * N;
    for (uint i = t; i < N; i += 32) {
        float x = g[i];
        // clamp the tanh argument to prevent overflow on outlier channels
        float arg = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        if (arg > 20.0f) arg = 20.0f; else if (arg < -20.0f) arg = -20.0f;
        float gelu_x = 0.5f * x * (1.0f + tanh(arg));
        g[i] = gelu_x * u[i];
    }
}

// Quantize an fp32 buffer to bf16 precision in-place (round mantissa to
// 7 bits, keep fp32 exponent). Used at vision encoder layer boundaries
// to match HF's bf16 residual stream. Each fp32 is 1+8+23 bits; bf16 is
// 1+8+7; we drop the low 16 bits (round-to-nearest-even via add-then-mask).
kernel void quantize_fp32_to_bf16_inplace(
    device float* x                     [[buffer(0)]],
    constant uint& N                    [[buffer(1)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    for (uint i = t; i < N; i += 32) {
        uint idx = b * N + i;
        // Round-to-nearest-even: add 0x7FFF + ((bits >> 16) & 1) before mask.
        uint bits = as_type<uint>(x[idx]);
        uint lsb = (bits >> 16) & 1u;
        uint rounded = (bits + 0x7FFFu + lsb) & 0xFFFF0000u;
        x[idx] = as_type<float>(rounded);
    }
}

// Fused vision FFN: gate_proj + up_proj + gelu(bf16(gate))*bf16(up), output
// bf16-rounded. Replaces 6 dispatches (gate GEMM, q, up GEMM, q, gelu*up, q)
// with 1. Preserves HF's bf16 rounding trajectory: gate_raw and up_raw are
// each bf16-rounded before the gelu-combine, and the final product is
// bf16-rounded on write. Grid: (D_out/8, ceil(B/8)); 32 threads per TG.
kernel void vision_ffn_gate_up_gelu_fp32_mma(
    device const float* X                [[buffer(0)]],   // [B, D_in]
    device const half*  W_gate           [[buffer(1)]],   // [D_out, D_in]
    device const half*  W_up             [[buffer(2)]],   // [D_out, D_in]
    device float*       Y                [[buffer(3)]],   // [B, D_out] bf16-rounded
    constant uint& B_count               [[buffer(4)]],
    constant uint& D_in                  [[buffer(5)]],
    constant uint& D_out                 [[buffer(6)]],
    uint2 tg                             [[threadgroup_position_in_grid]],
    uint2 lid                            [[thread_position_in_threadgroup]])
{
    constexpr uint Q_TILE  = 8;
    constexpr uint K_TILE  = 8;
    constexpr uint O_TILE  = 8;
    constexpr uint THREADS = 32;

    const uint o_block = tg.x;
    const uint q_block = tg.y;
    const uint o_start = o_block * O_TILE;
    const uint q_start = q_block * Q_TILE;
    const uint lid0    = lid.x;

    threadgroup half  x_stage[Q_TILE * K_TILE];
    threadgroup half  wg_stage[K_TILE * O_TILE];
    threadgroup half  wu_stage[K_TILE * O_TILE];
    threadgroup float yg_stage[Q_TILE * O_TILE];
    threadgroup float yu_stage[Q_TILE * O_TILE];

    simdgroup_float8x8 acc_g = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_u = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < D_in; k += K_TILE) {
        for (uint i = lid0; i < Q_TILE * K_TILE; i += THREADS) {
            uint q = i / K_TILE; uint kk = i % K_TILE;
            uint q_abs = q_start + q;
            uint k_abs = k + kk;
            x_stage[i] = (q_abs < B_count && k_abs < D_in)
                ? half(X[q_abs * D_in + k_abs]) : half(0);
        }
        for (uint i = lid0; i < K_TILE * O_TILE; i += THREADS) {
            uint kk = i / O_TILE; uint oo = i % O_TILE;
            uint o_abs = o_start + oo;
            uint k_abs = k + kk;
            bool ok = (o_abs < D_out && k_abs < D_in);
            wg_stage[i] = ok ? W_gate[o_abs * D_in + k_abs] : half(0);
            wu_stage[i] = ok ? W_up  [o_abs * D_in + k_abs] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 mx, mwg, mwu;
        simdgroup_load(mx,  x_stage,  K_TILE);
        simdgroup_load(mwg, wg_stage, O_TILE);
        simdgroup_load(mwu, wu_stage, O_TILE);
        simdgroup_multiply_accumulate(acc_g, mx, mwg, acc_g);
        simdgroup_multiply_accumulate(acc_u, mx, mwu, acc_u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc_g, yg_stage, O_TILE);
    simdgroup_store(acc_u, yu_stage, O_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid0; i < Q_TILE * O_TILE; i += THREADS) {
        uint q = i / O_TILE; uint oo = i % O_TILE;
        uint q_abs = q_start + q;
        uint o_abs = o_start + oo;
        if (q_abs >= B_count || o_abs >= D_out) continue;

        // bf16-round gate_raw and up_raw to match HF's bf16 residual trajectory.
        uint gbits = as_type<uint>(yg_stage[i]);
        uint glsb  = (gbits >> 16) & 1u;
        float g    = as_type<float>((gbits + 0x7FFFu + glsb) & 0xFFFF0000u);
        uint ubits = as_type<uint>(yu_stage[i]);
        uint ulsb  = (ubits >> 16) & 1u;
        float u    = as_type<float>((ubits + 0x7FFFu + ulsb) & 0xFFFF0000u);

        // GELU-tanh with clamp (match gelu_mul_fp32 + tanh-overflow guard).
        float arg = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        if (arg > 20.0f) arg = 20.0f; else if (arg < -20.0f) arg = -20.0f;
        float gelu_g = 0.5f * g * (1.0f + tanh(arg));
        float out    = gelu_g * u;

        // bf16-round the final product on write (folds q_geluMul).
        uint obits   = as_type<uint>(out);
        uint olsb    = (obits >> 16) & 1u;
        Y[q_abs * D_out + o_abs] = as_type<float>((obits + 0x7FFFu + olsb) & 0xFFFF0000u);
    }
}

// v2: 16×16 output tile with 8 fp32 accumulators (4 for gate, 4 for up).
// Same fused semantics + bf16 trajectory as vision_ffn_gate_up_gelu_fp32_mma,
// but 4× arithmetic per stage-barrier pair. Grid: (ceil(D_out/16), ceil(B/16)).
kernel void vision_ffn_gate_up_gelu_fp32_mma_v2(
    device const float* X                [[buffer(0)]],
    device const half*  W_gate           [[buffer(1)]],
    device const half*  W_up             [[buffer(2)]],
    device float*       Y                [[buffer(3)]],
    constant uint& B_count               [[buffer(4)]],
    constant uint& D_in                  [[buffer(5)]],
    constant uint& D_out                 [[buffer(6)]],
    uint2 tg                             [[threadgroup_position_in_grid]],
    uint2 lid                            [[thread_position_in_threadgroup]])
{
    constexpr uint Q_TILE  = 16;
    constexpr uint K_TILE  = 8;
    constexpr uint O_TILE  = 16;
    constexpr uint THREADS = 32;

    const uint o_block = tg.x;
    const uint q_block = tg.y;
    const uint o_start = o_block * O_TILE;
    const uint q_start = q_block * Q_TILE;
    const uint lid0    = lid.x;

    threadgroup half  x_stage[Q_TILE * K_TILE];    // 16×8 half = 256 B
    threadgroup half  wg_stage[K_TILE * O_TILE];   // 8×16 half = 256 B
    threadgroup half  wu_stage[K_TILE * O_TILE];   // 8×16 half = 256 B
    threadgroup float yg_stage[Q_TILE * O_TILE];   // 16×16 fp32 = 1024 B
    threadgroup float yu_stage[Q_TILE * O_TILE];   // 16×16 fp32 = 1024 B

    simdgroup_float8x8 acc_g00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_g01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_g10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_g11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_u00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_u01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_u10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_u11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < D_in; k += K_TILE) {
        for (uint i = lid0; i < Q_TILE * K_TILE; i += THREADS) {
            uint q = i / K_TILE; uint kk = i % K_TILE;
            uint q_abs = q_start + q;
            uint k_abs = k + kk;
            x_stage[i] = (q_abs < B_count && k_abs < D_in)
                ? half(X[q_abs * D_in + k_abs]) : half(0);
        }
        for (uint i = lid0; i < K_TILE * O_TILE; i += THREADS) {
            uint kk = i / O_TILE; uint oo = i % O_TILE;
            uint o_abs = o_start + oo;
            uint k_abs = k + kk;
            bool ok = (o_abs < D_out && k_abs < D_in);
            wg_stage[i] = ok ? W_gate[o_abs * D_in + k_abs] : half(0);
            wu_stage[i] = ok ? W_up  [o_abs * D_in + k_abs] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 mx0, mx1, mwg0, mwg1, mwu0, mwu1;
        simdgroup_load(mx0,  x_stage,               K_TILE);
        simdgroup_load(mx1,  x_stage + 8 * K_TILE,  K_TILE);
        simdgroup_load(mwg0, wg_stage,              O_TILE);
        simdgroup_load(mwg1, wg_stage + 8,          O_TILE);
        simdgroup_load(mwu0, wu_stage,              O_TILE);
        simdgroup_load(mwu1, wu_stage + 8,          O_TILE);

        // 8 MMAs: (2 Q tiles) × (2 O tiles) × (gate/up). 8× arithmetic per
        // barrier vs v1's 2 MMAs.
        simdgroup_multiply_accumulate(acc_g00, mx0, mwg0, acc_g00);
        simdgroup_multiply_accumulate(acc_g01, mx0, mwg1, acc_g01);
        simdgroup_multiply_accumulate(acc_g10, mx1, mwg0, acc_g10);
        simdgroup_multiply_accumulate(acc_g11, mx1, mwg1, acc_g11);
        simdgroup_multiply_accumulate(acc_u00, mx0, mwu0, acc_u00);
        simdgroup_multiply_accumulate(acc_u01, mx0, mwu1, acc_u01);
        simdgroup_multiply_accumulate(acc_u10, mx1, mwu0, acc_u10);
        simdgroup_multiply_accumulate(acc_u11, mx1, mwu1, acc_u11);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc_g00, yg_stage,                     O_TILE);
    simdgroup_store(acc_g01, yg_stage + 8,                 O_TILE);
    simdgroup_store(acc_g10, yg_stage + 8 * O_TILE,        O_TILE);
    simdgroup_store(acc_g11, yg_stage + 8 * O_TILE + 8,    O_TILE);
    simdgroup_store(acc_u00, yu_stage,                     O_TILE);
    simdgroup_store(acc_u01, yu_stage + 8,                 O_TILE);
    simdgroup_store(acc_u10, yu_stage + 8 * O_TILE,        O_TILE);
    simdgroup_store(acc_u11, yu_stage + 8 * O_TILE + 8,    O_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid0; i < Q_TILE * O_TILE; i += THREADS) {
        uint q = i / O_TILE; uint oo = i % O_TILE;
        uint q_abs = q_start + q;
        uint o_abs = o_start + oo;
        if (q_abs >= B_count || o_abs >= D_out) continue;

        uint gbits = as_type<uint>(yg_stage[i]);
        uint glsb  = (gbits >> 16) & 1u;
        float g    = as_type<float>((gbits + 0x7FFFu + glsb) & 0xFFFF0000u);
        uint ubits = as_type<uint>(yu_stage[i]);
        uint ulsb  = (ubits >> 16) & 1u;
        float u    = as_type<float>((ubits + 0x7FFFu + ulsb) & 0xFFFF0000u);

        float arg = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        if (arg > 20.0f) arg = 20.0f; else if (arg < -20.0f) arg = -20.0f;
        float gelu_g = 0.5f * g * (1.0f + tanh(arg));
        float out    = gelu_g * u;

        uint obits   = as_type<uint>(out);
        uint olsb    = (obits >> 16) & 1u;
        Y[q_abs * D_out + o_abs] = as_type<float>((obits + 0x7FFFu + olsb) & 0xFFFF0000u);
    }
}

// v3 fused FFN: 16×16 output tile + K-unroll=2. 16 MMAs per barrier pair
// (8 gate + 8 up), halves the barrier count vs v2 for the D_in=1152 → 4304
// gate/up projection pair. Same bf16 trajectory as earlier variants.
kernel void vision_ffn_gate_up_gelu_fp32_mma_v3(
    device const float* X                [[buffer(0)]],
    device const half*  W_gate           [[buffer(1)]],
    device const half*  W_up             [[buffer(2)]],
    device float*       Y                [[buffer(3)]],
    constant uint& B_count               [[buffer(4)]],
    constant uint& D_in                  [[buffer(5)]],
    constant uint& D_out                 [[buffer(6)]],
    uint2 tg                             [[threadgroup_position_in_grid]],
    uint2 lid                            [[thread_position_in_threadgroup]])
{
    constexpr uint Q_TILE   = 16;
    constexpr uint K_TILE   = 8;
    constexpr uint K_UNROLL = 2;
    constexpr uint K_CHUNK  = K_TILE * K_UNROLL;   // 16
    constexpr uint O_TILE   = 16;
    constexpr uint THREADS  = 32;

    const uint o_block = tg.x;
    const uint q_block = tg.y;
    const uint o_start = o_block * O_TILE;
    const uint q_start = q_block * Q_TILE;
    const uint lid0    = lid.x;

    threadgroup half  x_stage[Q_TILE * K_CHUNK];     // 512 B
    threadgroup half  wg_stage[K_CHUNK * O_TILE];    // 512 B
    threadgroup half  wu_stage[K_CHUNK * O_TILE];    // 512 B
    threadgroup float yg_stage[Q_TILE * O_TILE];     // 1024 B
    threadgroup float yu_stage[Q_TILE * O_TILE];     // 1024 B

    simdgroup_float8x8 acc_g00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_g01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_g10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_g11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_u00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_u01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_u10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_u11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < D_in; k += K_CHUNK) {
        for (uint i = lid0; i < Q_TILE * K_CHUNK; i += THREADS) {
            uint q = i / K_CHUNK; uint kk = i % K_CHUNK;
            uint q_abs = q_start + q;
            uint k_abs = k + kk;
            x_stage[i] = (q_abs < B_count && k_abs < D_in)
                ? half(X[q_abs * D_in + k_abs]) : half(0);
        }
        for (uint i = lid0; i < K_CHUNK * O_TILE; i += THREADS) {
            uint kk = i / O_TILE; uint oo = i % O_TILE;
            uint o_abs = o_start + oo;
            uint k_abs = k + kk;
            bool ok = (o_abs < D_out && k_abs < D_in);
            wg_stage[i] = ok ? W_gate[o_abs * D_in + k_abs] : half(0);
            wu_stage[i] = ok ? W_up  [o_abs * D_in + k_abs] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < K_UNROLL; ++kk) {
            simdgroup_half8x8 mx0, mx1, mwg0, mwg1, mwu0, mwu1;
            simdgroup_load(mx0,  x_stage + kk * 8,                  K_CHUNK);
            simdgroup_load(mx1,  x_stage + 8 * K_CHUNK + kk * 8,    K_CHUNK);
            simdgroup_load(mwg0, wg_stage + kk * 8 * O_TILE,        O_TILE);
            simdgroup_load(mwg1, wg_stage + kk * 8 * O_TILE + 8,    O_TILE);
            simdgroup_load(mwu0, wu_stage + kk * 8 * O_TILE,        O_TILE);
            simdgroup_load(mwu1, wu_stage + kk * 8 * O_TILE + 8,    O_TILE);

            simdgroup_multiply_accumulate(acc_g00, mx0, mwg0, acc_g00);
            simdgroup_multiply_accumulate(acc_g01, mx0, mwg1, acc_g01);
            simdgroup_multiply_accumulate(acc_g10, mx1, mwg0, acc_g10);
            simdgroup_multiply_accumulate(acc_g11, mx1, mwg1, acc_g11);
            simdgroup_multiply_accumulate(acc_u00, mx0, mwu0, acc_u00);
            simdgroup_multiply_accumulate(acc_u01, mx0, mwu1, acc_u01);
            simdgroup_multiply_accumulate(acc_u10, mx1, mwu0, acc_u10);
            simdgroup_multiply_accumulate(acc_u11, mx1, mwu1, acc_u11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc_g00, yg_stage,                     O_TILE);
    simdgroup_store(acc_g01, yg_stage + 8,                 O_TILE);
    simdgroup_store(acc_g10, yg_stage + 8 * O_TILE,        O_TILE);
    simdgroup_store(acc_g11, yg_stage + 8 * O_TILE + 8,    O_TILE);
    simdgroup_store(acc_u00, yu_stage,                     O_TILE);
    simdgroup_store(acc_u01, yu_stage + 8,                 O_TILE);
    simdgroup_store(acc_u10, yu_stage + 8 * O_TILE,        O_TILE);
    simdgroup_store(acc_u11, yu_stage + 8 * O_TILE + 8,    O_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid0; i < Q_TILE * O_TILE; i += THREADS) {
        uint q = i / O_TILE; uint oo = i % O_TILE;
        uint q_abs = q_start + q;
        uint o_abs = o_start + oo;
        if (q_abs >= B_count || o_abs >= D_out) continue;

        uint gbits = as_type<uint>(yg_stage[i]);
        uint glsb  = (gbits >> 16) & 1u;
        float g    = as_type<float>((gbits + 0x7FFFu + glsb) & 0xFFFF0000u);
        uint ubits = as_type<uint>(yu_stage[i]);
        uint ulsb  = (ubits >> 16) & 1u;
        float u    = as_type<float>((ubits + 0x7FFFu + ulsb) & 0xFFFF0000u);

        float arg = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        if (arg > 20.0f) arg = 20.0f; else if (arg < -20.0f) arg = -20.0f;
        float gelu_g = 0.5f * g * (1.0f + tanh(arg));
        float out    = gelu_g * u;

        uint obits   = as_type<uint>(out);
        uint olsb    = (obits >> 16) & 1u;
        Y[q_abs * D_out + o_abs] = as_type<float>((obits + 0x7FFFu + olsb) & 0xFFFF0000u);
    }
}

// Vision 2D pool fp32 → fp32 (spatial 3×3 average over patches).
kernel void vision_pool_2d_fp32in_fp32out(
    device const float* x               [[buffer(0)]],
    device float* out                   [[buffer(1)]],
    constant uint& grid_w               [[buffer(2)]],
    constant uint& out_w                [[buffer(3)]],
    constant uint& kernel_size          [[buffer(4)]],
    constant uint& hidden               [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint out_idx = tg.x; uint t = lid.x;
    uint oy = out_idx / out_w;
    uint ox = out_idx % out_w;
    uint y_start = oy * kernel_size;
    uint x_start = ox * kernel_size;
    float inv_area = 1.0f / float(kernel_size * kernel_size);
    for (uint i = t; i < hidden; i += 32) {
        float acc = 0;
        for (uint dy = 0; dy < kernel_size; ++dy) {
            for (uint dx = 0; dx < kernel_size; ++dx) {
                uint px = (y_start + dy) * grid_w + (x_start + dx);
                acc += x[px * hidden + i];
            }
        }
        out[out_idx * hidden + i] = acc * inv_area;
    }
}

// Vision std-normalize + sqrt(hidden) scale, fp32 I/O version.
kernel void vision_scaled_std_normalize_fp32(
    device const float* x               [[buffer(0)]],
    device const half* bias             [[buffer(1)]],
    device const half* scale            [[buffer(2)]],
    device float* out                   [[buffer(3)]],
    constant uint& D                    [[buffer(4)]],
    constant float& global_scale        [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    for (uint i = t; i < D; i += 32) {
        float v = x[b * D + i] * global_scale;
        v -= float(bias[i]);
        out[b * D + i] = v * float(scale[i]);
    }
}
"""
