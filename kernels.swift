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

// ─────────────────────────────────────────────────────────────────────
// KV virtual-page-table indirection.
//
// Each per-layer K/V cache is split across KV_NUM_CHUNKS device buffers.
// Phys-page id `p` maps to (chunk_idx, local_phys) via:
//     chunk_idx = p / chunk_pages
//     local_phys = p - chunk_idx * chunk_pages
//
// Kernels receive an argument buffer (Tier 2; M-series supported) that
// encodes the per-layer chunk pointer array. The kernel indexes into
// the array at runtime; the GPU's residency tracker only sees the
// chunks listed by the dispatcher's useResource() hints. This is the
// scatter-gather pattern: 100+ GB of total KV addressable, but only
// the ~6 GB working-set actually touched per CB is residency-tracked.
//
// KV_NUM_CHUNKS is hardcoded here and must match the Swift-side
// constant. Bumping it doubles addressable pool capacity (split-by-4
// allows ~32K pages = ~105 GB KV; split-by-8 allows ~64K = ~210 GB).
#define KV_NUM_CHUNKS 4

// Argument buffer carrying KV_NUM_CHUNKS device pointers. Same shape
// for K and V chunks; layer-level Swift allocation creates one of these
// per layer's K and one per layer's V.
struct KVChunks {
    device const half* chunks[KV_NUM_CHUNKS];
};
// Write-side variant (non-const) for kv_write / kv_write_multi.
struct KVChunksW {
    device half* chunks[KV_NUM_CHUNKS];
};

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

// Projection-steer: set the residual's projection onto cvec to a specific
// target magnitude. Unlike add_scaled_cvector which pushes BY an amount,
// this coerces the residual INTO a specific projection value — the
// representation-engineering primitive. target=0 removes the direction
// entirely (obliteratus-style ablation); target>0 elicits/amplifies;
// target<0 expresses the opposite valence. Also returns the PRE-WRITE
// projection value per slot so callers can read the "natural" feature
// level at each position — measurement + coercion in one dispatch.
//
// cvec MUST be unit-norm for the projection arithmetic to be clean:
//   current_proj = <hidden, cvec>
//   delta        = target - current_proj
//   hidden      += delta * cvec      (after which <hidden, cvec> == target)
kernel void project_cvector_fp16(
    device       half*  dst          [[buffer(0)]],  // [B, N] residual, written in place
    device const half*  cvec         [[buffer(1)]],  // [N] unit direction
    device       float* currentProj  [[buffer(2)]],  // [B] output, pre-write projection
    constant uint&      N            [[buffer(3)]],
    constant float&     target       [[buffer(4)]],  // desired projection magnitude
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint t = lid.x;
    // Phase 1: dot product. 32-thread simdgroup reduction via simd_sum.
    float acc = 0.0f;
    for (uint i = t; i < N; i += 32) {
        acc += float(dst[b*N + i]) * float(cvec[i]);
    }
    acc = simd_sum(acc);
    if (t == 0) currentProj[b] = acc;
    // Phase 2: every lane computes the same delta and writes its stripe.
    float delta = target - acc;
    for (uint i = t; i < N; i += 32) {
        dst[b*N + i] = half(float(dst[b*N + i]) + delta * float(cvec[i]));
    }
}

// Per-slot variant of project_cvector_fp16: caller offsets dst by
// slot*N*sizeof(half) and the kernel sees numVecs=1. Used in buildStepCB
// for per-session independent projection steering, same pattern as
// encAddScaledCvecSlot.
kernel void project_cvector_slot_fp16(
    device       half*  dst          [[buffer(0)]],  // dst+slot*N, [1, N]
    device const half*  cvec         [[buffer(1)]],
    device       float* currentProj  [[buffer(2)]],  // [1], pre-write projection
    constant uint&      N            [[buffer(3)]],
    constant float&     target       [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]])
{
    float acc = 0.0f;
    for (uint i = lid; i < N; i += 32) {
        acc += float(dst[i]) * float(cvec[i]);
    }
    acc = simd_sum(acc);
    if (lid == 0) currentProj[0] = acc;
    float delta = target - acc;
    for (uint i = lid; i < N; i += 32) {
        dst[i] = half(float(dst[i]) + delta * float(cvec[i]));
    }
}

// Prefill variant: per-row target magnitudes over [numVecs, N]. rows
// with target == sentinel skip entirely (so silenced slots + positions
// outside the envelope window pay near-zero cost, same pattern as
// add_scaled_cvector_prefill_fp16). We can't use target==0 as a skip
// sentinel because 0 is a meaningful coerce-to-zero value; instead use
// NaN (Float.nan from Swift-side writes the marker). currentProj[r]
// receives the pre-write projection for every active row.
kernel void project_cvector_prefill_fp16(
    device       half*  dst          [[buffer(0)]],  // [numVecs, N]
    device const half*  cvec         [[buffer(1)]],  // [N]
    device const float* targets      [[buffer(2)]],  // [numVecs] per-row target (NaN = skip)
    device       float* currentProj  [[buffer(3)]],  // [numVecs] output
    constant uint&      N            [[buffer(4)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint r = tg.x; uint t = lid.x;
    float target = targets[r];
    // NaN check: a row is skipped if its target is NaN. isnan() is a
    // valid Metal intrinsic.
    bool skip = isnan(target);
    float acc = 0.0f;
    for (uint i = t; i < N; i += 32) {
        acc += float(dst[r*N + i]) * float(cvec[i]);
    }
    acc = simd_sum(acc);
    if (t == 0) currentProj[r] = skip ? 0.0f : acc;
    if (skip) return;
    float delta = target - acc;
    for (uint i = t; i < N; i += 32) {
        dst[r*N + i] = half(float(dst[r*N + i]) + delta * float(cvec[i]));
    }
}

// Transport variant (per-slot): Gaussian optimal-transport coerce.
// Instead of a scalar target projection, we get per-class Gaussian
// statistics (μ_src, σ_src) for the source class and (μ_tgt, σ_tgt)
// for the target class. The Brenier map between two 1D Gaussians is
//    a' = μ_tgt + (a - μ_src) * (σ_tgt / σ_src)
// which can be rewritten as a' = scale*a + offset with
//    scale  = σ_tgt / σ_src
//    offset = μ_tgt - scale * μ_src
// We take scale+offset as kernel inputs (computed client-side — the
// kernel doesn't need to know the individual μ/σ values). This
// preserves within-class variation when applied per-PC: an input that
// was ±kσ_src from μ_src ends up at ±kσ_tgt from μ_tgt. No overshoot
// past the class distribution → no Korean-unembedding cartography.
kernel void transport_cvector_slot_fp16(
    device       half*  dst          [[buffer(0)]],  // dst+slot*N, [1, N]
    device const half*  cvec         [[buffer(1)]],
    device       float* currentProj  [[buffer(2)]],  // [1] pre-write projection
    constant uint&      N            [[buffer(3)]],
    constant float&     scale        [[buffer(4)]],
    constant float&     offset       [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]])
{
    float a = 0.0f;
    for (uint i = lid; i < N; i += 32) {
        a += float(dst[i]) * float(cvec[i]);
    }
    a = simd_sum(a);
    if (lid == 0) currentProj[0] = a;
    // a' = scale*a + offset; delta = a' - a = (scale-1)*a + offset.
    float delta = (scale - 1.0f) * a + offset;
    for (uint i = lid; i < N; i += 32) {
        dst[i] = half(float(dst[i]) + delta * float(cvec[i]));
    }
}

// Transport variant (prefill, per-row). Rows with scale == NaN (or
// whatever sentinel we pick — matching project-variant convention) are
// skipped. scales[r]/offsets[r] per row; same logic as slot variant
// extended over [numVecs, N].
kernel void transport_cvector_prefill_fp16(
    device       half*  dst          [[buffer(0)]],  // [numVecs, N]
    device const half*  cvec         [[buffer(1)]],  // [N]
    device const float* scales       [[buffer(2)]],  // [numVecs] per-row scale (NaN = skip)
    device const float* offsets      [[buffer(3)]],  // [numVecs] per-row offset
    device       float* currentProj  [[buffer(4)]],  // [numVecs] output
    constant uint&      N            [[buffer(5)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint r = tg.x; uint t = lid.x;
    float scale = scales[r];
    float offset = offsets[r];
    bool skip = isnan(scale);
    float a = 0.0f;
    for (uint i = t; i < N; i += 32) {
        a += float(dst[r*N + i]) * float(cvec[i]);
    }
    a = simd_sum(a);
    if (t == 0) currentProj[r] = skip ? 0.0f : a;
    if (skip) return;
    float delta = (scale - 1.0f) * a + offset;
    for (uint i = t; i < N; i += 32) {
        dst[r*N + i] = half(float(dst[r*N + i]) + delta * float(cvec[i]));
    }
}

// Heretic-style per-write directional ablation (abliteration).
// Applies  y -= alpha * r_hat * dot(r_hat, y)  to every row of a
// [numVecs, N] fp16 buffer. Same direction and alpha for all rows
// (model-level intervention — all batch slots at a given site get
// the same treatment). Algebraically equivalent to orthogonalizing
// the preceding matrix W against r_hat with magnitude alpha:
//   (W - alpha * r_hat * r_hat^T * W) x  =  y - alpha * r_hat * (r_hat^T y)
// so this post-matmul hook gives us the same effect without ever
// touching the weights. alpha=0 is a no-op; alpha=1 fully zeroes the
// r_hat component; alpha<0 amplifies (behavior induction). Two-phase:
// (1) simdgroup-reduce dot(r_hat, y[r]) per row; (2) rank-1 subtract.
kernel void orthogonalize_write_fp16(
    device       half*  y      [[buffer(0)]],  // [numVecs, N]
    device const half*  r_hat  [[buffer(1)]],  // [N]
    constant uint&      N      [[buffer(2)]],
    constant float&     alpha  [[buffer(3)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint r = tg.x; uint t = lid.x;
    float g = 0.0f;
    for (uint i = t; i < N; i += 32) {
        g += float(y[r*N + i]) * float(r_hat[i]);
    }
    g = simd_sum(g);
    float s = alpha * g;
    for (uint i = t; i < N; i += 32) {
        y[r*N + i] = half(float(y[r*N + i]) - s * float(r_hat[i]));
    }
}

// Prefill-variant: per-row magnitudes. dst[r, :] += mags[r] * cvec[:], with
// rows indexed over the full prefill tile (numVecs = B * qLen). mags[r] == 0
// is a no-op row — the dispatch wastes ~32 thread-cycles reading zeros, but
// we avoid a per-(slot, position) dispatch-count explosion and keep the CPU
// scheduler out of the hot path. Called once per (layer, active-control)
// pair by encodePrefillTileInto.
kernel void add_scaled_cvector_prefill_fp16(
    device       half*  dst   [[buffer(0)]],   // [numVecs, N] residual rows
    device const half*  cvec  [[buffer(1)]],   // [N] control vector
    device const float* mags  [[buffer(2)]],   // [numVecs] per-row magnitude
    constant uint&      N     [[buffer(3)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint r = tg.x; uint t = lid.x;
    float m = mags[r];
    if (m == 0.0f) return;
    for (uint i = t; i < N; i += 32) {
        dst[r*N + i] = half(float(dst[r*N + i]) + m * float(cvec[i]));
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

// V4 prefill: V4 with grid extended over vec-blocks. Each TG handles
// MAX_B=8 vec rows at v_base = tg.y * MAX_B and produces 32 output cols.
// For huge D_out (unembed) the weight matrix overflows L2, so V5's per-vec
// grid pays DRAM bandwidth ~numVecs× (its L2 amortization between vec
// iterations breaks down). V4 with vec-block tile reads weight 1× (per
// slab) and FMAs into MAX_B accumulators in registers. Profiled at
// numVecs=256 unembed: V5 ≈ 150 ms; V4_p target ≈ 5 ms (BW-bound 1.4 GB
// at ~400 GB/s).
//
// numVecs is now a kernel arg (not B) so prefill (numVecs up to MAX_Q_LEN
// × B) and AR (numVecs ≤ B) share the kernel.
kernel void dense_gemv_v4_p(
    device const half* hidden [[buffer(0)]], device const half* W [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& D_in [[buffer(3)]],
    constant uint& D_out [[buffer(4)]], constant uint& numVecs [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    // MAX_B sweet spot from 2026-04-28 sweep: =8 → 106 ms; =16 → 152 ms;
    // =32 → 251 ms (V5 baseline 150 ms). Above 8, register pressure /
    // occupancy loss erases the bandwidth-amortization win.
    constexpr uint MAX_B = 8;
    uint n_block = tg.x;
    uint v_base = tg.y * MAX_B;
    uint lidx = lid.x;
    uint n = n_block * 32 + lidx;
    if (n >= D_out) return;
    if (v_base >= numVecs) return;
    uint b_count = (numVecs - v_base < MAX_B) ? (numVecs - v_base) : MAX_B;

    device const half* w_col = W + n;
    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;
    for (uint k = 0; k < D_in; k += 8) {
        half w0 = w_col[(k+0)*D_out], w1 = w_col[(k+1)*D_out], w2 = w_col[(k+2)*D_out], w3 = w_col[(k+3)*D_out];
        half w4 = w_col[(k+4)*D_out], w5 = w_col[(k+5)*D_out], w6 = w_col[(k+6)*D_out], w7 = w_col[(k+7)*D_out];
        for (uint b = 0; b < b_count; ++b) {
            device const half* hid = hidden + (v_base + b) * D_in + k;
            accs[b] += float(hid[0])*float(w0) + float(hid[1])*float(w1) + float(hid[2])*float(w2) + float(hid[3])*float(w3)
                     + float(hid[4])*float(w4) + float(hid[5])*float(w5) + float(hid[6])*float(w6) + float(hid[7])*float(w7);
        }
    }
    for (uint b = 0; b < b_count; ++b) output[(v_base + b) * D_out + n] = half(accs[b]);
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
    device const KVChunksW& k_chunks [[buffer(2)]],
    device const KVChunksW& v_chunks [[buffer(3)]],
    device const uint* block_table [[buffer(4)]], device const uint* q_positions [[buffer(5)]],
    constant uint& H [[buffer(6)]], constant uint& D [[buffer(7)]],
    constant uint& PAGE [[buffer(8)]], constant uint& max_pages [[buffer(9)]],
    constant uint& q_len [[buffer(10)]],
    constant uint& chunk_pages [[buffer(27)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint q_local = tg.y; uint h = tg.z; uint t = lid.x;
    uint q_flat = b * q_len + q_local;
    uint pos = q_positions[q_flat];
    uint lp = pos / PAGE; uint off = pos % PAGE;
    uint phys = block_table[b * max_pages + lp];
    uint chunk_idx = phys / chunk_pages;
    uint local_phys = phys - chunk_idx * chunk_pages;
    device half* Kx = k_chunks.chunks[chunk_idx];
    device half* Vx = v_chunks.chunks[chunk_idx];
    device const half* Ks = K + (q_flat * H + h) * D;
    device const half* Vs = V + (q_flat * H + h) * D;
    device half* Kd = Kx + ((local_phys * PAGE + off) * H + h) * D;
    device half* Vd = Vx + ((local_phys * PAGE + off) * H + h) * D;
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

// KV cache write — scatter-gathers writes across KV_NUM_CHUNKS device
// buffers per layer K/V via argument buffer indirection. See the
// KVChunks struct + KV_NUM_CHUNKS #define at top of file.
kernel void kv_write(
    device const half* K [[buffer(0)]], device const half* V [[buffer(1)]],
    device const KVChunksW& k_chunks [[buffer(2)]],
    device const KVChunksW& v_chunks [[buffer(3)]],
    device const uint* block_table [[buffer(4)]], device const uint* positions [[buffer(5)]],
    constant uint& H [[buffer(6)]], constant uint& D [[buffer(7)]],
    constant uint& PAGE [[buffer(8)]], constant uint& max_pages [[buffer(9)]],
    constant uint& chunk_pages [[buffer(27)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])
{
    uint b = tg.x; uint h = tg.y; uint t = lid.x;
    uint pos = positions[b]; uint lp = pos / PAGE; uint off = pos % PAGE;
    uint phys = block_table[b * max_pages + lp];
    uint chunk_idx = phys / chunk_pages;
    uint local_phys = phys - chunk_idx * chunk_pages;
    device half* Kx = k_chunks.chunks[chunk_idx];
    device half* Vx = v_chunks.chunks[chunk_idx];
    device const half* Ks = K + (b * H + h) * D; device const half* Vs = V + (b * H + h) * D;
    device half* Kd = Kx + ((local_phys * PAGE + off) * H + h) * D;
    device half* Vd = Vx + ((local_phys * PAGE + off) * H + h) * D;
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


// ---- Split-KV paged attention: compute partials + reduce ----
// Grid compute: (B*H_Q, N_SPLITS) TGs. Each TG owns one (slot, q_head, split)
// and processes its page range via Flash online softmax, writing partial
// (m, l, O_unnorm) to partials buffers.
// Grid reduce: (B*H_Q,) TGs. Each combines N_SPLITS partials → final O using
// Flash-associative merge. 4× more parallel TGs than single-TG attention —
// fixes the ~66% latency-starved portion of attention cost on M5.


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


// Attention BW-model placeholder. Real paged attention (paged_attention.swift)
// does Flash online-softmax with simdgroup_load; measured at ~80-500 μs per
// layer at batch=4 K_len=1024. This simpler model reads K+V once each
// (representative of the Flash pattern's amortized BW) and writes O. NOT a
// correct attention — just a memory-BW cost proxy for the forward-graph sim.
// For a real forward, swap this for paged_attention's decode_attn_batched.
kernel void fake_attention(
    device const half* Q [[buffer(0)]],
    device half* O [[buffer(1)]],
    device const KVChunks& k_chunks [[buffer(2)]],
    device const KVChunks& v_chunks [[buffer(3)]],
    device const uint* block_table [[buffer(4)]],
    constant uint& num_pages [[buffer(5)]],
    constant uint& H [[buffer(6)]], constant uint& D [[buffer(7)]], constant uint& PAGE [[buffer(8)]],
    constant uint& max_pages [[buffer(9)]],
    constant uint& chunk_pages [[buffer(27)]],
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
        uint chunk_idx = phys / chunk_pages;
        uint local_phys = phys - chunk_idx * chunk_pages;
        device const half* Kx = k_chunks.chunks[chunk_idx];
        device const half* Vx = v_chunks.chunks[chunk_idx];
        device const half* Kp = Kx + (local_phys * PAGE * H + h) * D;
        device const half* Vp = Vx + (local_phys * PAGE * H + h) * D;
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
//
// V2 takes aB (active batch count). Iterates only [0, aB*K) in passes 1
// and 3 — silenced slots' garbage routings (computed by softmax_topk
// over hidden states for non-active slots) are skipped, so MoE matmul
// only consumes real-slot routings. For aB == B, behavior is identical
// to V1. For aB < B, only the first aB slots' routings end up in
// slot_token[]/batch_slots[] and group_start reflects their counts only.
//
// V3 ALSO writes a compact active_experts[] list (overwrites the
// static identity map at runtime). For aB=1 with TOPK=8 distinct
// routings, this packs the 8 active expert IDs into
// active_experts[0..8) and pads the rest with sentinel E (= 128).
// MoE kernels check the sentinel and early-return. Dispatcher passes
// numActive = TOPK * aB so only the relevant TGs launch — saves
// 120 wasted TG launches per slab at aB=1.
// Size the MoE-prefill matmul grid to the ACTUAL routing, not the worst case.
// Reads group_start[E+1] (per-expert compacted offsets) and writes one
// MTLDispatchThreadgroupsIndirectArguments triple {gx, gy, gz}:
//   gx = ceil(maxTokens/32)   — token dim, bounded by the single hottest expert
//   gy = passed in (ceil(Dout/64))
//   gz = activeExperts        — non-empty groups only
// maxTokens = max over experts of group_start[e+1]-group_start[e]. One thread;
// each MoE matmul re-emits this with its own gy right before dispatching.
kernel void moe_write_dispatch_args(
    device const uint* group_start  [[buffer(0)]],   // [E+1]
    device uint*       args         [[buffer(1)]],   // 3 uints: {gx,gy,gz}
    constant uint&     gy           [[buffer(2)]],
    constant uint&     gxCap        [[buffer(3)]],   // structural max token-tiles
    uint lid                        [[thread_position_in_threadgroup]])
{
    if (lid != 0) return;
    constexpr uint E = 128;
    // gxCap is the STRUCTURAL ceiling on the X (token-tile) axis: a token
    // routes to any expert at most once, so an expert holds <= N tokens
    // (N = numSlots/TOPK) => at most ceil(N/32) tiles. The grid X axis is
    // GPU-computed here and is otherwise UNBOUNDED. Without this clamp a
    // single non-monotonic / stale / pre-route-compact group_start value
    // makes the unsigned subtraction below underflow to ~4.29e9, gx blows
    // up to ~1e8 threadgroups, and the GPU watchdog can't drain the grid —
    // WindowServer (shared GPU) hangs and the machine needs a hard reboot.
    // (RCA 2026-05-28: the kernel-hang-the-whole-box bug.) Every CPU path
    // this replaced was already numSlots-bounded; this restores that ceiling.
    uint maxc = 0, active = 0;
    for (uint e = 0; e < E; ++e) {
        uint a = group_start[e];
        uint b = group_start[e + 1];
        uint c = (b > a) ? (b - a) : 0u;   // saturating: never underflow
        if (c > maxc) maxc = c;
        if (c > 0) active += 1;
    }
    uint gx = (maxc + 31) / 32;
    if (gx == 0)     gx = 1;               // never a zero-size dispatch
    if (gxCap > 0 && gx > gxCap) gx = gxCap;// hard structural ceiling
    if (active == 0) active = 1;
    if (active > E)  active = E;            // gz can never exceed E experts
    args[0] = gx; args[1] = gy; args[2] = active;
}

kernel void route_compact(
    device const uint* expert_ids   [[buffer(0)]],   // [B*K]
    device uint* group_start        [[buffer(1)]],   // [E+1]
    device uint* slot_token         [[buffer(2)]],   // [B*K]
    device uint* batch_slots        [[buffer(3)]],   // [B*K]
    constant uint& aB               [[buffer(4)]],   // active batch count (was B)
    constant uint& K                [[buffer(5)]],
    device uint* active_experts     [[buffer(6)]],   // [E] — overwritten with compact list + sentinel padding
    uint2 lid                       [[thread_position_in_threadgroup]])
{
    constexpr uint E = 128;
    threadgroup uint counts[E];
    uint e = lid.x;
    uint BK = aB * K;
    if (e >= E) return;

    // Pass 1: count matches for my expert across [0, aB*K) only —
    // silenced slots [aB, B) are skipped.
    uint myCount = 0;
    for (uint i = 0; i < BK; ++i) {
        if (expert_ids[i] == e) ++myCount;
    }
    counts[e] = myCount;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: thread 0 serializes the E=128 prefix sum into group_start
    // AND compacts the active expert IDs into active_experts[].
    // Inactive experts get sentinel E (=128) which the MoE kernels
    // recognize as "skip TG entirely".
    if (e == 0) {
        uint running = 0;
        uint active_count = 0;
        for (uint i = 0; i < E; ++i) {
            group_start[i] = running;
            running += counts[i];
            if (counts[i] > 0) {
                active_experts[active_count] = i;
                active_count += 1;
            }
        }
        group_start[E] = running;
        // Pad sentinel for [active_count, E).
        for (uint i = active_count; i < E; ++i) {
            active_experts[i] = E;     // sentinel = invalid expert
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: scatter. Thread e walks (b,k) in row-major order over
    // [0, aB*K) and emits slots for its own expert. Same iteration order
    // as pass 1 -> consistent local offset sequence, so batch_slots is
    // well-defined for active slots. batch_slots[aB*K..B*K) is left
    // stale; downstream consumers (encMoeCombineWriteInto) will read
    // stale values for silenced slots but write to scratch outputs that
    // the engine ignores anyway.
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
    constexpr uint MAX_SLOTS = 16;
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

// -------- Q5_K dequant helpers + AR GEMV --------
// block_q5_K (176 bytes / 256 elts):
//   half d, half dmin, uchar scales[12], uchar qh[32], uchar qs[128]
// Sub-tile structure (16 sub-tiles of 16 elts each, indexed by il_orig ∈ [0,16)):
//   q_off  = 32*(il_orig/4) + 16*(il_orig & 1)   // base byte in qs[128]
//   qh_off = 16*(il_orig & 1)                    // base byte in qh[32]
//   ul     = 1 << (il_orig / 2)                  // bit selector in qh
//   is     = (il_orig / 4) * 2                   // scales-pair index
//   ph     = il_orig & 3                         // phase ∈ {0..3}
// Phase ph: lo nibble for ph<2, hi nibble for ph>=2; d /= 16 for ph>=2;
//           qh adds 16 for ph<2, 256 for ph>=2; sc index (ph/2) ∈ {0,1}.

// Forward-shadowed in AR section to avoid ordering against the int-typed
// version defined in the prefill section (line ~7500). Mirrors that function
// verbatim with uint args.
static inline uchar2 get_scale_min_k4_just2(
    uint j, uint k, device const uchar* q)
{
    if (j < 4) {
        return uchar2(q[j + 0 + k] & 0x3F, q[j + 4 + k] & 0x3F);
    }
    return uchar2((q[j + 4 + k] & 0x0F) | ((q[j - 4 + k] & 0xC0) >> 2),
                  (q[j + 4 + k] >> 4)   | ((q[j - 0 + k] & 0xC0) >> 2));
}

// Dense GEMV with Q5_K weights, v4 pattern (one thread per output column,
// MAX_B accumulators per thread amortizes W reads across batch).
kernel void dense_gemv_q5_K_v4(
    device const half* hidden           [[buffer(0)]],
    device const uchar* W_q5k           [[buffer(1)]],
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

    constexpr uint BLK = 176;
    uint nbc = D_in / 256;
    uint col_bytes = nbc * BLK;
    device const uchar* W_col = W_q5k + n * col_bytes;

    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;

    for (uint kb = 0; kb < nbc; ++kb) {
        device const uchar* blk = W_col + kb * BLK;
        float d_all = float(*(device const half*)(blk + 0));
        float dmin  = float(*(device const half*)(blk + 2));
        device const uchar* scales = blk + 4;
        device const uchar* qh     = blk + 16;
        device const uchar* qs     = blk + 48;

        for (uint il_orig = 0; il_orig < 16; ++il_orig) {
            uint q_off  = 32u * (il_orig / 4u) + 16u * (il_orig & 1u);
            uint qh_off = 16u * (il_orig & 1u);
            uchar ul    = uchar(1u << (il_orig / 2u));
            uint is     = (il_orig / 4u) * 2u;
            uint ph     = il_orig & 3u;
            uchar2 sc   = get_scale_min_k4_just2(is, ph / 2u, scales);
            float d_eff = (ph < 2u) ? d_all : (d_all / 16.0f);
            float dl    = d_eff * float(sc[0]);
            float ml    = dmin  * float(sc[1]);
            uchar mask  = (ph < 2u) ? 0x0F : 0xF0;
            float qhv   = (ph < 2u) ? 16.0f : 256.0f;
            uint base_k = kb * 256u + il_orig * 16u;
            for (uint i = 0; i < 16; ++i) {
                float v = float(qs[q_off + i] & mask)
                        + (((qh[qh_off + i] & ul) != 0) ? qhv : 0.0f);
                float w = dl * v - ml;
                for (uint b = 0; b < B; ++b) {
                    accs[b] += float(hidden[b * D_in + base_k + i]) * w;
                }
            }
        }
    }
    for (uint b = 0; b < B; ++b) output[b * D_out + n] = half(accs[b]);
}

// -------- Q6_K dequant + AR GEMV --------
// block_q6_K (210 bytes / 256 elts):
//   uchar ql[128], uchar qh[64], int8 scales[16], half d
// 16 sub-tiles of 16 elts (il ∈ [0,16)), each producing 4 packed 32-bit q's
// of 4 bytes each (16 elts total). Centered Q6 (-32 offset baked into ml).

kernel void dense_gemv_q6_K_v4(
    device const half* hidden           [[buffer(0)]],
    device const uchar* W_q6k           [[buffer(1)]],
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

    constexpr uint BLK = 210;
    uint nbc = D_in / 256;
    uint col_bytes = nbc * BLK;
    device const uchar* W_col = W_q6k + n * col_bytes;

    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;

    for (uint kb = 0; kb < nbc; ++kb) {
        device const uchar* blk = W_col + kb * BLK;
        device const ushort* ql_u16    = (device const ushort*)(blk + 0);
        device const ushort* qh_u16    = (device const ushort*)(blk + 128);
        device const char*   scales_i8 = (device const char*)(blk + 192);
        float d_all = float(*(device const half*)(blk + 208));

        for (uint il = 0; il < 16; ++il) {
            uint ql_base = 32u * (il / 8u) + 16u * ((il / 2u) & 1u) + 8u * (il & 1u);
            uint qh_base = 16u * (il / 8u) + 8u * (il & 1u);
            float sc = float(scales_i8[(il % 2u) + 2u * (il / 2u)]);
            uint ph = (il / 2u) & 3u;

            uint kmask1 = (ph > 1u) ? ((ph > 2u) ? 0xC0C0C0C0u : 0x30303030u)
                                    : ((ph > 0u) ? 0x0C0C0C0Cu : 0x03030303u);
            uint kmask2 = (ph > 1u) ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
            float ml  = d_all * sc * 32.0f;
            float dl0 = d_all * sc;
            float dl1 = dl0 / 256.0f;
            float dl2 = dl1 / 256.0f;
            float dl3 = dl2 / 256.0f;
            uint shr_h = (ph > 2u) ? 2u : 0u;
            uint shl_h = (ph > 1u) ? 0u : ((ph > 0u) ? 2u : 4u);
            uint shr_l = (ph > 1u) ? 4u : 0u;

            uint base_k = kb * 256u + il * 16u;
            for (uint i = 0; i < 4; ++i) {
                uint low  = (uint(ql_u16[ql_base + 2u*i]) | (uint(ql_u16[ql_base + 2u*i + 1u]) << 16)) & kmask2;
                uint high = (uint(qh_u16[qh_base + 2u*i]) | (uint(qh_u16[qh_base + 2u*i + 1u]) << 16)) & kmask1;
                uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
                float w0 = dl0 * float(q & 0x000000FFu)         - ml;
                float w1 = dl1 * float(q & 0x0000FF00u)         - ml;
                float w2 = dl2 * float(q & 0x00FF0000u)         - ml;
                float w3 = dl3 * float(q & 0xFF000000u)         - ml;
                uint k0 = base_k + i * 4u;
                for (uint b = 0; b < B; ++b) {
                    device const half* hb = hidden + b * D_in;
                    accs[b] += float(hb[k0 + 0]) * w0
                             + float(hb[k0 + 1]) * w1
                             + float(hb[k0 + 2]) * w2
                             + float(hb[k0 + 3]) * w3;
                }
            }
        }
    }
    for (uint b = 0; b < B; ++b) output[b * D_out + n] = half(accs[b]);
}

// -------- Q5_1 dense AR GEMV (32-elt blocks, 24 B) --------
// block_q5_1 (24 bytes per 32 elts):
//   half d, half m, uchar qh[4], uchar qs[16]
// Per-element: q_lo = (qs[p] & 0xF) | (((qh[p/8] >> (p%8)) & 1) << 4); w_lo = d*q_lo + m
//              q_hi = ((qs[p] >> 4) & 0xF) | (((qh[(p+16)/8] >> ((p+16)%8)) & 1) << 4); w_hi = d*q_hi + m

kernel void dense_gemv_q5_1_v4(
    device const half* hidden           [[buffer(0)]],
    device const uchar* W_q51           [[buffer(1)]],
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

    constexpr uint BLK = 24;
    uint nbc = D_in / 32;
    uint col_bytes = nbc * BLK;
    device const uchar* W_col = W_q51 + n * col_bytes;

    float accs[MAX_B];
    for (uint b = 0; b < MAX_B; ++b) accs[b] = 0.0f;

    for (uint kb = 0; kb < nbc; ++kb) {
        device const uchar* blk = W_col + kb * BLK;
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
            for (uint b = 0; b < B; ++b) {
                accs[b] += float(hidden[b * D_in + base_k + p])      * w_lo
                         + float(hidden[b * D_in + base_k + p + 16]) * w_hi;
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
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
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
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
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

// =============================================================================
// Q8_0 GEMV — kernel zoo at compile-time fixed B_TILE ∈ {1, 2, 4, 8}.
//
// Architecture: each TG handles ALL B_TILE batches in registers via a per-
// thread accumulator array `accs[B_TILE]`. Compared to V6 (1 batch per TG,
// B TGs in grid), this reads weights ONCE per K-tile per TG and amortizes
// across B_TILE batches at the register level — no L2 round-trip dependence.
//
// Compile-time B_TILE is critical: Apple's compiler can fully unroll the
// inner B-loop and keep `accs[B_TILE]` in registers (we hit a hard wall on
// runtime-bounded array indexing in V8/V9 — accs spilled to local mem,
// 2-3× slower). With B_TILE constexpr, no spill.
//
// Scheduler picks the variant by rounding active-batch UP to the nearest
// power of 2: activeB ∈ [1] → b1; [2] → b2; [3,4] → b4; [5..8] → b8.
// Silenced slots in the tail (e.g., 5 active → b8 with 3 silenced) still
// pay weight-bandwidth-amortized work; the per-stream cost is pessimal at
// activeB just above a power-of-2 boundary, optimal exactly on it.
//
// Grid: (D_out/32) — one TG per 32-output slab. Threads: 4 SGs × 32 = 128.
// Split-K: each SG covers a quarter of K. Per-batch reduction at end.
// Caller dispatches with grid = (D_out/32, 1, 1) regardless of B_TILE.
// Templated impl: caller (kernel) supplies the threadgroup partials array
// because MSL forbids `threadgroup` declarations inside non-kernel functions.
template<uint B_TILE>
inline void dense_gemv_q8_0_btile_impl(
    device const half* hidden,
    device const uchar* W_sw,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 34;

    uint n_block = tg.x;
    uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    // Original kb_per_sg split: contiguous range per SG, restored to keep
    // Q8_0 logit output bit-identical to the pre-LUT baseline. For Q8_0 at
    // K=2816, nbc=88, divides cleanly across N_SPLITS=4. Other formats
    // with non-divisible nbc use a strided variant — see q5_K_btile_impl etc.
    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    // B_TILE accumulators per thread, fully unrollable since B_TILE is constexpr.
    float accs[B_TILE];
    for (uint b = 0; b < B_TILE; ++b) accs[b] = 0.0f;

    for (uint kb = kb_begin; kb < kb_end; ++kb) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        for (uint p = 0; p < 32; ++p) {
            float w_p = d * float(qs[p]);
            // Compile-time unrolled inner loop: B_TILE FMAs into B_TILE register accs.
            for (uint b = 0; b < B_TILE; ++b) {
                accs[b] += float(hidden[b * D_in + base_k + p]) * w_p;
            }
        }
    }

    // Per-batch reduction across N_SPLITS SGs, write each output.
    for (uint b = 0; b < B_TILE; ++b) {
        partials[sg_id][lid_sg] = accs[b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            output[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q8_0_btile_b1")]]
kernel void dense_gemv_q8_0_btile_b1(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_impl<1>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_b2")]]
kernel void dense_gemv_q8_0_btile_b2(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_impl<2>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_b4")]]
kernel void dense_gemv_q8_0_btile_b4(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_impl<4>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_b8")]]
kernel void dense_gemv_q8_0_btile_b8(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_impl<8>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

// =============================================================================
// F16 btile GEMV — unquantized baseline riding the same dispatcher as the
// quant zoo. Structural twin of dense_gemv_q8_0_btile_impl with two
// simplifications: (1) W is plain row-major [D_out, D_in] half (matches GGUF
// native layout, no swizzle, no super-column packing); (2) no per-block scale
// and no int->float dequant — values flow straight through float(half).
// Same N_SPLITS=4 split-K, same kb=32-wide K-tiling for SG-coalesced loads,
// same B_TILE register accumulators, same per-batch threadgroup reduction.
// Each TG handles 32 output rows; thread lid_sg maps to row n_block*32 + lid_sg.
// =============================================================================
template<uint B_TILE>
inline void dense_gemv_f16_btile_impl(
    device const half* hidden,
    device const half* W,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;

    uint n_block = tg.x;
    uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    uint nbc = D_in / 32;

    // Contiguous-range split per SG, matching Q8_0 btile. For divisible nbc,
    // each SG owns kb_per_sg = nbc / N_SPLITS consecutive blocks.
    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    // B_TILE accumulators per thread, fully unrollable since B_TILE is constexpr.
    float accs[B_TILE];
    for (uint b = 0; b < B_TILE; ++b) accs[b] = 0.0f;

    // Row base for this thread's output: 32 contiguous fp16 elements per kb.
    device const half* w_row = W + n * D_in;

    for (uint kb = kb_begin; kb < kb_end; ++kb) {
        uint base_k = kb * 32;
        device const half* w_blk = w_row + base_k;
        for (uint p = 0; p < 32; ++p) {
            float w_p = float(w_blk[p]);
            // Compile-time unrolled inner loop: B_TILE FMAs into B_TILE register accs.
            for (uint b = 0; b < B_TILE; ++b) {
                accs[b] += float(hidden[b * D_in + base_k + p]) * w_p;
            }
        }
    }

    // Per-batch reduction across N_SPLITS SGs, write each output.
    for (uint b = 0; b < B_TILE; ++b) {
        partials[sg_id][lid_sg] = accs[b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            output[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_f16_btile_b1")]]
kernel void dense_gemv_f16_btile_b1(
    device const half* hidden  [[buffer(0)]],
    device const half* W       [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_impl<1>(hidden, W, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_f16_btile_b2")]]
kernel void dense_gemv_f16_btile_b2(
    device const half* hidden  [[buffer(0)]],
    device const half* W       [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_impl<2>(hidden, W, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_f16_btile_b4")]]
kernel void dense_gemv_f16_btile_b4(
    device const half* hidden  [[buffer(0)]],
    device const half* W       [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_impl<4>(hidden, W, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_f16_btile_b8")]]
kernel void dense_gemv_f16_btile_b8(
    device const half* hidden  [[buffer(0)]],
    device const half* W       [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_impl<8>(hidden, W, output, D_in, D_out, partials, tg, lid, sg_id);
}

// =============================================================================
// Q5_K btile GEMV — structural mirror of dense_gemv_q8_0_btile_impl, swapping
// only the per-element dequant. Same N_SPLITS=4 split-K, same swizzled W
// layout [n_super, kb, 32 cols, BLK], same B_TILE register accumulator
// pattern, same per-batch threadgroup reduction. Block size differs:
// 256 elts × 176 B for Q5_K vs 32 elts × 34 B for Q8_0.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q5_K_btile_impl(
    device const half* hidden,
    device const uchar* W_sw,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 176;
    constexpr uint K_PER_BLK = 256;

    uint n_block = tg.x;
    uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint b = 0; b < B_TILE; ++b) accs[b] = 0.0f;

    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d_all = float(*(device const half*)(blk + 0));
        float dmin  = float(*(device const half*)(blk + 2));
        device const uchar* scales = blk + 4;
        device const uchar* qh     = blk + 16;
        device const uchar* qs     = blk + 48;
        for (uint il_orig = 0; il_orig < 16; ++il_orig) {
            uint q_off  = 32u * (il_orig / 4u) + 16u * (il_orig & 1u);
            uint qh_off = 16u * (il_orig & 1u);
            uchar ul    = uchar(1u << (il_orig / 2u));
            uint is     = (il_orig / 4u) * 2u;
            uint ph     = il_orig & 3u;
            uchar2 sc   = get_scale_min_k4_just2(is, ph / 2u, scales);
            float d_eff = (ph < 2u) ? d_all : (d_all / 16.0f);
            float dl    = d_eff * float(sc[0]);
            float ml    = dmin  * float(sc[1]);
            uchar mask  = (ph < 2u) ? 0x0F : 0xF0;
            float qhv   = (ph < 2u) ? 16.0f : 256.0f;
            uint base_k = kb * K_PER_BLK + il_orig * 16u;
            for (uint i = 0; i < 16; ++i) {
                float v = float(qs[q_off + i] & mask)
                        + (((qh[qh_off + i] & ul) != 0) ? qhv : 0.0f);
                float w_p = dl * v - ml;
                for (uint b = 0; b < B_TILE; ++b) {
                    accs[b] += float(hidden[b * D_in + base_k + i]) * w_p;
                }
            }
        }
    }

    for (uint b = 0; b < B_TILE; ++b) {
        partials[sg_id][lid_sg] = accs[b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            output[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q5_K_btile_b1")]]
kernel void dense_gemv_q5_K_btile_b1(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_impl<1>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_K_btile_b2")]]
kernel void dense_gemv_q5_K_btile_b2(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_impl<2>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_K_btile_b4")]]
kernel void dense_gemv_q5_K_btile_b4(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_impl<4>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_K_btile_b8")]]
kernel void dense_gemv_q5_K_btile_b8(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_impl<8>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

// =============================================================================
// Q6_K btile GEMV — same skeleton, Q6_K dequant inside the kb loop.
// Block: 210 B / 256 elts. ql[128], qh[64], int8 scales[16], half d (offset 208).
// Per-subtile (il ∈ [0,16)) produces 16 weights via the kmask-shift-pack pattern.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q6_K_btile_impl(
    device const half* hidden,
    device const uchar* W_sw,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 210;
    constexpr uint K_PER_BLK = 256;

    uint n_block = tg.x;
    uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint b = 0; b < B_TILE; ++b) accs[b] = 0.0f;

    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        device const ushort* ql_u16    = (device const ushort*)(blk + 0);
        device const ushort* qh_u16    = (device const ushort*)(blk + 128);
        device const char*   scales_i8 = (device const char*)(blk + 192);
        float d_all = float(*(device const half*)(blk + 208));

        for (uint il = 0; il < 16; ++il) {
            uint ql_base = 32u * (il / 8u) + 16u * ((il / 2u) & 1u) + 8u * (il & 1u);
            uint qh_base = 16u * (il / 8u) + 8u * (il & 1u);
            float sc = float(scales_i8[(il % 2u) + 2u * (il / 2u)]);
            uint ph = (il / 2u) & 3u;

            uint kmask1 = (ph > 1u) ? ((ph > 2u) ? 0xC0C0C0C0u : 0x30303030u)
                                    : ((ph > 0u) ? 0x0C0C0C0Cu : 0x03030303u);
            uint kmask2 = (ph > 1u) ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
            float ml  = d_all * sc * 32.0f;
            float dl0 = d_all * sc;
            float dl1 = dl0 / 256.0f;
            float dl2 = dl1 / 256.0f;
            float dl3 = dl2 / 256.0f;
            uint shr_h = (ph > 2u) ? 2u : 0u;
            uint shl_h = (ph > 1u) ? 0u : ((ph > 0u) ? 2u : 4u);
            uint shr_l = (ph > 1u) ? 4u : 0u;

            uint base_k = kb * K_PER_BLK + il * 16u;
            for (uint i = 0; i < 4; ++i) {
                uint low  = (uint(ql_u16[ql_base + 2u*i]) | (uint(ql_u16[ql_base + 2u*i + 1u]) << 16)) & kmask2;
                uint high = (uint(qh_u16[qh_base + 2u*i]) | (uint(qh_u16[qh_base + 2u*i + 1u]) << 16)) & kmask1;
                uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
                float w0 = dl0 * float(q & 0x000000FFu) - ml;
                float w1 = dl1 * float(q & 0x0000FF00u) - ml;
                float w2 = dl2 * float(q & 0x00FF0000u) - ml;
                float w3 = dl3 * float(q & 0xFF000000u) - ml;
                uint k0 = base_k + i * 4u;
                for (uint b = 0; b < B_TILE; ++b) {
                    device const half* hb = hidden + b * D_in;
                    accs[b] += float(hb[k0 + 0]) * w0
                             + float(hb[k0 + 1]) * w1
                             + float(hb[k0 + 2]) * w2
                             + float(hb[k0 + 3]) * w3;
                }
            }
        }
    }

    for (uint b = 0; b < B_TILE; ++b) {
        partials[sg_id][lid_sg] = accs[b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            output[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q6_K_btile_b1")]]
kernel void dense_gemv_q6_K_btile_b1(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_impl<1>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q6_K_btile_b2")]]
kernel void dense_gemv_q6_K_btile_b2(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_impl<2>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q6_K_btile_b4")]]
kernel void dense_gemv_q6_K_btile_b4(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_impl<4>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q6_K_btile_b8")]]
kernel void dense_gemv_q6_K_btile_b8(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_impl<8>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

// =============================================================================
// Q5_1 btile GEMV — block: 24 B / 32 elts. half d, half m, uchar qh[4], uchar qs[16].
// Same outer skeleton as Q8_0 (32-elt blocks); inner dequant is 5-bit unpack.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q5_1_btile_impl(
    device const half* hidden,
    device const uchar* W_sw,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 24;
    constexpr uint K_PER_BLK = 32;

    uint n_block = tg.x;
    uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint b = 0; b < B_TILE; ++b) accs[b] = 0.0f;

    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        float m = float(*(device const half*)(blk + 2));
        device const uchar* qh = blk + 4;
        device const uchar* qs = blk + 8;
        uint base_k = kb * K_PER_BLK;
        for (uint p = 0; p < 16; ++p) {
            uchar qsp = qs[p];
            uint h_lo = (qh[p / 8]       >> (p       % 8)) & 1u;
            uint h_hi = (qh[(p + 16) / 8] >> ((p + 16) % 8)) & 1u;
            uint q_lo = (uint(qsp) & 0xFu)        | (h_lo << 4);
            uint q_hi = ((uint(qsp) >> 4) & 0xFu) | (h_hi << 4);
            float w_lo = d * float(q_lo) + m;
            float w_hi = d * float(q_hi) + m;
            for (uint b = 0; b < B_TILE; ++b) {
                accs[b] += float(hidden[b * D_in + base_k + p])      * w_lo
                         + float(hidden[b * D_in + base_k + p + 16]) * w_hi;
            }
        }
    }

    for (uint b = 0; b < B_TILE; ++b) {
        partials[sg_id][lid_sg] = accs[b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            output[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q5_1_btile_b1")]]
kernel void dense_gemv_q5_1_btile_b1(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_impl<1>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_1_btile_b2")]]
kernel void dense_gemv_q5_1_btile_b2(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_impl<2>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_1_btile_b4")]]
kernel void dense_gemv_q5_1_btile_b4(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_impl<4>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_1_btile_b8")]]
kernel void dense_gemv_q5_1_btile_b8(
    device const half* hidden  [[buffer(0)]],
    device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_impl<8>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

// =============================================================================
// Q4_0 btile GEMV — block: 18 B / 32 elts. half d (offset 0), uchar qs[16].
// Per element: w = d * (nib - 8). Simplest 4bpw on Apple Silicon — fewest
// ALU ops per FMA in the dequant chain. Same outer skeleton as Q5_1.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q4_0_btile_impl(
    device const half* hidden,
    device const uchar* W_sw,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 18;
    constexpr uint K_PER_BLK = 32;

    uint n_block = tg.x;
    uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint b = 0; b < B_TILE; ++b) accs[b] = 0.0f;

    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const uchar* qs = blk + 2;
        uint base_k = kb * K_PER_BLK;
        for (uint p = 0; p < 16; ++p) {
            uchar byte = qs[p];
            float w_lo = d * float(int(byte & 0xF) - 8);
            float w_hi = d * float(int((byte >> 4) & 0xF) - 8);
            for (uint b = 0; b < B_TILE; ++b) {
                accs[b] += float(hidden[b * D_in + base_k + p])      * w_lo
                         + float(hidden[b * D_in + base_k + p + 16]) * w_hi;
            }
        }
    }

    for (uint b = 0; b < B_TILE; ++b) {
        partials[sg_id][lid_sg] = accs[b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            output[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q4_0_btile_b1")]]
kernel void dense_gemv_q4_0_btile_b1(
    device const half* hidden  [[buffer(0)]], device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]], constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_impl<1>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_0_btile_b2")]]
kernel void dense_gemv_q4_0_btile_b2(
    device const half* hidden  [[buffer(0)]], device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]], constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_impl<2>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_0_btile_b4")]]
kernel void dense_gemv_q4_0_btile_b4(
    device const half* hidden  [[buffer(0)]], device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]], constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_impl<4>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_0_btile_b8")]]
kernel void dense_gemv_q4_0_btile_b8(
    device const half* hidden  [[buffer(0)]], device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]], constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_impl<8>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}

// =============================================================================
// Q4_1 btile GEMV — block: 20 B / 32 elts. half d (offset 0), half m (offset 2),
// uchar qs[16] (offset 4). Per element: w = d * nib + m (unsigned 4-bit nibble,
// no -8 bias; m is the additive offset). Same outer skeleton as Q4_0 / Q5_1.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q4_1_btile_impl(
    device const half* hidden,
    device const uchar* W_sw,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 20;
    constexpr uint K_PER_BLK = 32;

    uint n_block = tg.x;
    uint lid_sg = lid.x % 32;
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;

    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint b = 0; b < B_TILE; ++b) accs[b] = 0.0f;

    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        float m = float(*(device const half*)(blk + 2));
        device const uchar* qs = blk + 4;
        uint base_k = kb * K_PER_BLK;
        for (uint p = 0; p < 16; ++p) {
            uchar byte = qs[p];
            float w_lo = d * float(byte & 0xF) + m;
            float w_hi = d * float((byte >> 4) & 0xF) + m;
            for (uint b = 0; b < B_TILE; ++b) {
                accs[b] += float(hidden[b * D_in + base_k + p])      * w_lo
                         + float(hidden[b * D_in + base_k + p + 16]) * w_hi;
            }
        }
    }

    for (uint b = 0; b < B_TILE; ++b) {
        partials[sg_id][lid_sg] = accs[b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            output[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q4_1_btile_b1")]]
kernel void dense_gemv_q4_1_btile_b1(
    device const half* hidden  [[buffer(0)]], device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]], constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_impl<1>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_1_btile_b2")]]
kernel void dense_gemv_q4_1_btile_b2(
    device const half* hidden  [[buffer(0)]], device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]], constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_impl<2>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_1_btile_b4")]]
kernel void dense_gemv_q4_1_btile_b4(
    device const half* hidden  [[buffer(0)]], device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]], constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_impl<4>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_1_btile_b8")]]
kernel void dense_gemv_q4_1_btile_b8(
    device const half* hidden  [[buffer(0)]], device const uchar* W_sw   [[buffer(1)]],
    device half* output        [[buffer(2)]], constant uint& D_in        [[buffer(3)]],
    constant uint& D_out       [[buffer(4)]],
    uint2 tg                   [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_impl<8>(hidden, W_sw, output, D_in, D_out, partials, tg, lid, sg_id);
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
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
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
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
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

// F16 mirror of dense_gemv_q8_0_v6_rmsnorm_qkv: same fused RMSNorm + QKV
// dispatch but weights are plain row-major fp16 [D_out, D_in] (no swizzle,
// no per-block scales). Inner loop drops the dequant mul and reads
// weights directly. nbc = D_in / 32 K-tiling preserved verbatim.
kernel void dense_gemv_f16_v6_rmsnorm_qkv(
    device const half* x                [[buffer(0)]],
    device const half* gamma            [[buffer(1)]],
    device const half* Wq               [[buffer(2)]],
    device const half* Wk               [[buffer(3)]],
    device const half* Wv               [[buffer(4)]],
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
    device const half* W;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab;
        D_out = D_out_q;
        W = Wq;
        out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb;
        D_out = D_out_k;
        W = Wk;
        out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb;
        D_out = D_out_v;
        W = Wv;
        out = out_v;
    }

    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    device const half* w_row = W + n * D_in;

    float acc = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        uint base_k = kb * 32;
        device const half* w_blk = w_row + base_k;
        for (uint p = 0; p < 32; ++p) {
            acc += float(h_norm[base_k + p]) * float(w_blk[p]);
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

// =============================================================================
// Q8_0 RMSNorm + QKV kernel zoo at compile-time fixed B_TILE ∈ {1, 2, 4}.
//
// Same spirit as V7 (one TG handles all B_TILE batches in registers, weight
// read once per K-tile, V_count is runtime ≤ B_TILE) but B_TILE is now a
// template parameter so the compiler:
//   (a) sizes h_norms[B_TILE * MAX_D_IN] tg-mem to exactly what's needed
//       — V7 hard-coded VEC_TILE=4 → 22.5 KB always, wasteful for activeB=1
//   (b) unrolls the inner FMA loop across B_TILE accumulators
//   (c) drops dead code paths for slot indices ≥ B_TILE
//
// tg-mem budget at MAX_D_IN=2816, half = 2 B:
//   B_TILE=1 →  5.6 KB | B_TILE=2 → 11.2 KB | B_TILE=4 → 22.5 KB
//   B_TILE=8 → 45 KB — over the 32 KB cap; needs a different staging
//   strategy (e.g., k-tiled normalize on-the-fly). Future work.
//
// Scheduler maps activeB → kernel:
//   activeB == 1 → btile_qkv_b1 (exact)
//   activeB == 2 → btile_qkv_b2 (exact)
//   activeB ∈ {3,4} → btile_qkv_b4 (1 predicated slot for activeB=3)
//   activeB > 4 → fall back to V6 (no btile_qkv_b8 yet)
template<uint B_TILE>
inline void dense_gemv_q8_0_btile_qkv_impl(
    device const half* x,
    device const half* gamma,
    device const uchar* Wq_sw,
    device const uchar* Wk_sw,
    device const uchar* Wv_sw,
    device half* out_q,
    device half* out_k,
    device half* out_v,
    constant uint& D_in,
    constant uint& Q_nb,
    constant uint& K_nb,
    constant uint& V_nb,
    constant uint& D_out_q,
    constant uint& D_out_k,
    constant uint& D_out_v,
    constant float& eps,
    constant uint& numVecs,
    threadgroup half* h_norms,            // [B_TILE * MAX_D_IN]
    threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint MAX_D_IN = 2816;
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 34;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    // Phase 1: stage RMS-normalized x for each of v_count batches into h_norms.
    // Constexpr loop bound B_TILE; predicate body on (v < v_count).
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        float inv_rms = rsqrt(total_ss / float(D_in) + eps);
        for (uint i = tid; i < D_in; i += THREADS) {
            h_norms[v * MAX_D_IN + i] = half(float(xb[i]) * inv_rms * float(gamma[i]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: route slab to Q/K/V.
    uint n_block;
    uint D_out;
    device const uchar* W_sw;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab; D_out = D_out_q; W_sw = Wq_sw; out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb; D_out = D_out_k; W_sw = Wk_sw; out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb; D_out = D_out_v; W_sw = Wv_sw; out = out_v;
    }
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;
    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    // Phase 3: register-amortized matmul. B_TILE accs in registers.
    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        for (uint p = 0; p < 32; ++p) {
            float w_p = d * float(qs[p]);
            for (uint v = 0; v < B_TILE; ++v) {
                if (v < v_count) {
                    accs[v] += float(h_norms[v * MAX_D_IN + base_k + p]) * w_p;
                }
            }
        }
    }

    // Phase 4: per-batch reduction across N_SPLITS SGs and write outputs.
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            out[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q8_0_btile_qkv_b1")]]
kernel void dense_gemv_q8_0_btile_qkv_b1(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup half h_norms[1 * 2816];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_qkv_impl<1>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                       out_q, out_k, out_v,
                                       D_in, Q_nb, K_nb, V_nb,
                                       D_out_q, D_out_k, D_out_v,
                                       eps, numVecs,
                                       h_norms, ss_stage, partials,
                                       tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_qkv_b2")]]
kernel void dense_gemv_q8_0_btile_qkv_b2(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup half h_norms[2 * 2816];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_qkv_impl<2>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                       out_q, out_k, out_v,
                                       D_in, Q_nb, K_nb, V_nb,
                                       D_out_q, D_out_k, D_out_v,
                                       eps, numVecs,
                                       h_norms, ss_stage, partials,
                                       tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_qkv_b4")]]
kernel void dense_gemv_q8_0_btile_qkv_b4(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup half h_norms[4 * 2816];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_qkv_impl<4>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                       out_q, out_k, out_v,
                                       D_in, Q_nb, K_nb, V_nb,
                                       D_out_q, D_out_k, D_out_v,
                                       eps, numVecs,
                                       h_norms, ss_stage, partials,
                                       tg, lid, sg_id);
}

// =============================================================================
// Q8_0 RMSNorm + QKV kernel zoo OTF (on-the-fly normalize) at compile-time
// fixed B_TILE ∈ {1, 2, 4, 8}.
//
// Same layout/grid/register-amortization story as the staging-based btile_qkv
// above, but the RMSNorm fusion is split: phase 1 computes a per-batch
// scalar inv_rms and stashes it in tg-mem (only B_TILE floats). Phase 3
// reads x and gamma directly per element and applies normalize-on-the-fly,
// so there's NO full-D h_norm tg-mem staging.
//
// tg-mem footprint: B_TILE×4B (inv_rms) + 16B (ss_stage) + 512B (partials)
// = ~560B at B_TILE=8 vs 22.5 KB for the staging-based b4 variant.
// This is what unlocks B_TILE=8 (the staging design overflowed the 32 KB
// tg-mem cap at b8).
//
// Cost vs staging: phase 3 reads x and gamma per FMA element (extra L1/L2
// traffic) and adds 1 mul per FMA (gamma scaling) instead of computing
// h_norm once into tg-mem. At AR scales the L1-resident x and gamma
// across the kb loop dominate; the otf math is hot-cache-amortized.
template<uint B_TILE>
inline void dense_gemv_q8_0_btile_qkv_otf_impl(
    device const half* x,
    device const half* gamma,
    device const uchar* Wq_sw,
    device const uchar* Wk_sw,
    device const uchar* Wv_sw,
    device half* out_q,
    device half* out_k,
    device half* out_v,
    constant uint& D_in,
    constant uint& Q_nb,
    constant uint& K_nb,
    constant uint& V_nb,
    constant uint& D_out_q,
    constant uint& D_out_k,
    constant uint& D_out_v,
    constant float& eps,
    constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE],
    threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 34;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    // Phase 1: per-batch RMS reduction. Compile-time-bounded loop unrolls;
    // predicate body on (v < v_count).
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) {
            inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: route slab to Q/K/V projection.
    uint n_block;
    uint D_out;
    device const uchar* W_sw;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab; D_out = D_out_q; W_sw = Wq_sw; out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb; D_out = D_out_k; W_sw = Wk_sw; out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb; D_out = D_out_v; W_sw = Wv_sw; out = out_v;
    }
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;
    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    // Phase 3: GEMV with on-the-fly RMSNorm. B_TILE register accs.
    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        for (uint p = 0; p < 32; ++p) {
            float w_p = d * float(qs[p]);
            float gamma_p = float(gamma[base_k + p]);
            for (uint v = 0; v < B_TILE; ++v) {
                if (v < v_count) {
                    uint b = v_base + v;
                    float x_val = float(x[b * D_in + base_k + p]);
                    // Normalize-on-the-fly: x · inv_rms · gamma · w
                    accs[v] += x_val * inv_rms[v] * gamma_p * w_p;
                }
            }
        }
    }

    // Phase 4: per-batch reduction across N_SPLITS SGs and write outputs.
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            out[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q8_0_btile_qkv_otf_b1")]]
kernel void dense_gemv_q8_0_btile_qkv_otf_b1(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_qkv_otf_impl<1>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                           out_q, out_k, out_v,
                                           D_in, Q_nb, K_nb, V_nb,
                                           D_out_q, D_out_k, D_out_v,
                                           eps, numVecs,
                                           inv_rms, ss_stage, partials,
                                           tg, lid, sg_id);
}

// =============================================================================
// F16 mirror of the Q8_0 RMSNorm + QKV otf zoo. Same per-batch inv_rms
// staging and register-amortized matmul, but weights are plain row-major
// fp16 [D_out, D_in] (no swizzle, no per-block scales). Inner loop drops
// the d * qs[p] dequant and reads w_row[base_k + p] directly. Only B_TILE=1
// is needed by this scope (b1 = the AR-decode path).
template<uint B_TILE>
inline void dense_gemv_f16_btile_qkv_otf_impl(
    device const half* x,
    device const half* gamma,
    device const half* Wq,
    device const half* Wk,
    device const half* Wv,
    device half* out_q,
    device half* out_k,
    device half* out_v,
    constant uint& D_in,
    constant uint& Q_nb,
    constant uint& K_nb,
    constant uint& V_nb,
    constant uint& D_out_q,
    constant uint& D_out_k,
    constant uint& D_out_v,
    constant float& eps,
    constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE],
    threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    // Phase 1: per-batch RMS reduction (format-agnostic).
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) {
            inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: route slab to Q/K/V projection.
    uint n_block;
    uint D_out;
    device const half* W;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab; D_out = D_out_q; W = Wq; out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb; D_out = D_out_k; W = Wk; out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb; D_out = D_out_v; W = Wv; out = out_v;
    }
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    device const half* w_row = W + n * D_in;

    // Phase 3: GEMV with on-the-fly RMSNorm. B_TILE register accs.
    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        uint base_k = kb * 32;
        device const half* w_blk = w_row + base_k;
        for (uint p = 0; p < 32; ++p) {
            float w_p = float(w_blk[p]);
            float gamma_p = float(gamma[base_k + p]);
            for (uint v = 0; v < B_TILE; ++v) {
                if (v < v_count) {
                    uint b = v_base + v;
                    float x_val = float(x[b * D_in + base_k + p]);
                    // Normalize-on-the-fly: x · inv_rms · gamma · w
                    accs[v] += x_val * inv_rms[v] * gamma_p * w_p;
                }
            }
        }
    }

    // Phase 4: per-batch reduction across N_SPLITS SGs and write outputs.
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            out[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_f16_btile_qkv_otf_b1")]]
kernel void dense_gemv_f16_btile_qkv_otf_b1(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const half* Wq       [[buffer(2)]],
    device const half* Wk       [[buffer(3)]],
    device const half* Wv       [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_qkv_otf_impl<1>(x, gamma, Wq, Wk, Wv,
                                          out_q, out_k, out_v,
                                          D_in, Q_nb, K_nb, V_nb,
                                          D_out_q, D_out_k, D_out_v,
                                          eps, numVecs,
                                          inv_rms, ss_stage, partials,
                                          tg, lid, sg_id);
}

// F16 OTF QKV — B_TILE ∈ {2, 4, 8}. Mechanical mirrors of the b1 wrapper
// using the same templated impl with different specializations. Reason
// for shipping these (the Q8_0 zoo had b{2,4,8} but production V6
// fallback skipped them at activeB > 1): F16 has no per-FMA dequant
// cost, so the OTF inner loop is much tighter than Q8_0's. The V6
// fallback redoes the RMS reduction in every (slab, batch) TG, which
// is a significant redundant-work cost at activeB=8 — OTF amortizes
// that across the B_TILE batches sharing a TG.
[[host_name("dense_gemv_f16_btile_qkv_otf_b2")]]
kernel void dense_gemv_f16_btile_qkv_otf_b2(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const half* Wq       [[buffer(2)]],
    device const half* Wk       [[buffer(3)]],
    device const half* Wv       [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_qkv_otf_impl<2>(x, gamma, Wq, Wk, Wv,
                                          out_q, out_k, out_v,
                                          D_in, Q_nb, K_nb, V_nb,
                                          D_out_q, D_out_k, D_out_v,
                                          eps, numVecs,
                                          inv_rms, ss_stage, partials,
                                          tg, lid, sg_id);
}

[[host_name("dense_gemv_f16_btile_qkv_otf_b4")]]
kernel void dense_gemv_f16_btile_qkv_otf_b4(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const half* Wq       [[buffer(2)]],
    device const half* Wk       [[buffer(3)]],
    device const half* Wv       [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_qkv_otf_impl<4>(x, gamma, Wq, Wk, Wv,
                                          out_q, out_k, out_v,
                                          D_in, Q_nb, K_nb, V_nb,
                                          D_out_q, D_out_k, D_out_v,
                                          eps, numVecs,
                                          inv_rms, ss_stage, partials,
                                          tg, lid, sg_id);
}

[[host_name("dense_gemv_f16_btile_qkv_otf_b8")]]
kernel void dense_gemv_f16_btile_qkv_otf_b8(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const half* Wq       [[buffer(2)]],
    device const half* Wk       [[buffer(3)]],
    device const half* Wv       [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_qkv_otf_impl<8>(x, gamma, Wq, Wk, Wv,
                                          out_q, out_k, out_v,
                                          D_in, Q_nb, K_nb, V_nb,
                                          D_out_q, D_out_k, D_out_v,
                                          eps, numVecs,
                                          inv_rms, ss_stage, partials,
                                          tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_qkv_otf_b2")]]
kernel void dense_gemv_q8_0_btile_qkv_otf_b2(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_qkv_otf_impl<2>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                           out_q, out_k, out_v,
                                           D_in, Q_nb, K_nb, V_nb,
                                           D_out_q, D_out_k, D_out_v,
                                           eps, numVecs,
                                           inv_rms, ss_stage, partials,
                                           tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_qkv_otf_b4")]]
kernel void dense_gemv_q8_0_btile_qkv_otf_b4(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_qkv_otf_impl<4>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                           out_q, out_k, out_v,
                                           D_in, Q_nb, K_nb, V_nb,
                                           D_out_q, D_out_k, D_out_v,
                                           eps, numVecs,
                                           inv_rms, ss_stage, partials,
                                           tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_qkv_otf_b8")]]
kernel void dense_gemv_q8_0_btile_qkv_otf_b8(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_qkv_otf_impl<8>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                           out_q, out_k, out_v,
                                           D_in, Q_nb, K_nb, V_nb,
                                           D_out_q, D_out_k, D_out_v,
                                           eps, numVecs,
                                           inv_rms, ss_stage, partials,
                                           tg, lid, sg_id);
}

// =============================================================================
// Q5_K btile_qkv_otf — structural mirror of dense_gemv_q8_0_btile_qkv_otf.
// Same 4-phase shape (RMS reduction → slab routing → otf-RMSNorm GEMV →
// per-batch reduction); only Phase 3's per-element dequant differs.
// Block size: 256 elts × 176 B for Q5_K vs 32 elts × 34 B for Q8_0.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q5_K_btile_qkv_otf_impl(
    device const half* x,
    device const half* gamma,
    device const uchar* Wq_sw,
    device const uchar* Wk_sw,
    device const uchar* Wv_sw,
    device half* out_q,
    device half* out_k,
    device half* out_v,
    constant uint& D_in,
    constant uint& Q_nb,
    constant uint& K_nb,
    constant uint& V_nb,
    constant uint& D_out_q,
    constant uint& D_out_k,
    constant uint& D_out_v,
    constant float& eps,
    constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE],
    threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 176;
    constexpr uint K_PER_BLK = 256;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    // Phase 1: per-batch RMS reduction (format-agnostic, same as Q8_0).
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) {
            inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: slab routing — same as Q8_0 (not format-dependent).
    uint n_block;
    uint D_out;
    device const uchar* W_sw;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab; D_out = D_out_q; W_sw = Wq_sw; out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb; D_out = D_out_k; W_sw = Wk_sw; out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb; D_out = D_out_v; W_sw = Wv_sw; out = out_v;
    }
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    // Phase 3: GEMV with on-the-fly RMSNorm. Q5_K dequant inside.
    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d_all = float(*(device const half*)(blk + 0));
        float dmin  = float(*(device const half*)(blk + 2));
        device const uchar* scales = blk + 4;
        device const uchar* qh     = blk + 16;
        device const uchar* qs     = blk + 48;
        for (uint il_orig = 0; il_orig < 16; ++il_orig) {
            uint q_off  = 32u * (il_orig / 4u) + 16u * (il_orig & 1u);
            uint qh_off = 16u * (il_orig & 1u);
            uchar ul    = uchar(1u << (il_orig / 2u));
            uint is     = (il_orig / 4u) * 2u;
            uint ph     = il_orig & 3u;
            uchar2 sc   = get_scale_min_k4_just2(is, ph / 2u, scales);
            float d_eff = (ph < 2u) ? d_all : (d_all / 16.0f);
            float dl    = d_eff * float(sc[0]);
            float ml    = dmin  * float(sc[1]);
            uchar mask  = (ph < 2u) ? 0x0F : 0xF0;
            float qhv   = (ph < 2u) ? 16.0f : 256.0f;
            uint base_k = kb * K_PER_BLK + il_orig * 16u;
            for (uint i = 0; i < 16; ++i) {
                float v_q = float(qs[q_off + i] & mask)
                          + (((qh[qh_off + i] & ul) != 0) ? qhv : 0.0f);
                float w_p = dl * v_q - ml;
                float gamma_p = float(gamma[base_k + i]);
                for (uint v = 0; v < B_TILE; ++v) {
                    if (v < v_count) {
                        uint b = v_base + v;
                        float x_val = float(x[b * D_in + base_k + i]);
                        accs[v] += x_val * inv_rms[v] * gamma_p * w_p;
                    }
                }
            }
        }
    }

    // Phase 4: per-batch reduction across N_SPLITS SGs (same as Q8_0).
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            out[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q5_K_btile_qkv_otf_b1")]]
kernel void dense_gemv_q5_K_btile_qkv_otf_b1(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_qkv_otf_impl<1>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                           out_q, out_k, out_v,
                                           D_in, Q_nb, K_nb, V_nb,
                                           D_out_q, D_out_k, D_out_v,
                                           eps, numVecs,
                                           inv_rms, ss_stage, partials,
                                           tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_K_btile_qkv_otf_b2")]]
kernel void dense_gemv_q5_K_btile_qkv_otf_b2(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_qkv_otf_impl<2>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                           out_q, out_k, out_v,
                                           D_in, Q_nb, K_nb, V_nb,
                                           D_out_q, D_out_k, D_out_v,
                                           eps, numVecs,
                                           inv_rms, ss_stage, partials,
                                           tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_K_btile_qkv_otf_b4")]]
kernel void dense_gemv_q5_K_btile_qkv_otf_b4(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_qkv_otf_impl<4>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                           out_q, out_k, out_v,
                                           D_in, Q_nb, K_nb, V_nb,
                                           D_out_q, D_out_k, D_out_v,
                                           eps, numVecs,
                                           inv_rms, ss_stage, partials,
                                           tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_K_btile_qkv_otf_b8")]]
kernel void dense_gemv_q5_K_btile_qkv_otf_b8(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_qkv_otf_impl<8>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                           out_q, out_k, out_v,
                                           D_in, Q_nb, K_nb, V_nb,
                                           D_out_q, D_out_k, D_out_v,
                                           eps, numVecs,
                                           inv_rms, ss_stage, partials,
                                           tg, lid, sg_id);
}

// =============================================================================
// Q6_K btile_qkv_otf — structural mirror with Q6_K dequant. Block: 210 B /
// 256 elts. ql[128], qh[64], int8 scales[16], half d (offset 208).
// Per il_orig ∈ [0,16): 16 weights via the kmask-shift-pack formula.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q6_K_btile_qkv_otf_impl(
    device const half* x,
    device const half* gamma,
    device const uchar* Wq_sw,
    device const uchar* Wk_sw,
    device const uchar* Wv_sw,
    device half* out_q,
    device half* out_k,
    device half* out_v,
    constant uint& D_in,
    constant uint& Q_nb,
    constant uint& K_nb,
    constant uint& V_nb,
    constant uint& D_out_q,
    constant uint& D_out_k,
    constant uint& D_out_v,
    constant float& eps,
    constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE],
    threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 210;
    constexpr uint K_PER_BLK = 256;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    // Phase 1: per-batch RMS — same as Q5_K / Q8_0.
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) {
            inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: slab routing.
    uint n_block;
    uint D_out;
    device const uchar* W_sw;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab; D_out = D_out_q; W_sw = Wq_sw; out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb; D_out = D_out_k; W_sw = Wk_sw; out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb; D_out = D_out_v; W_sw = Wv_sw; out = out_v;
    }
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    // Phase 3: GEMV with on-the-fly RMSNorm. Q6_K dequant inside.
    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        device const ushort* ql_u16    = (device const ushort*)(blk + 0);
        device const ushort* qh_u16    = (device const ushort*)(blk + 128);
        device const char*   scales_i8 = (device const char*)(blk + 192);
        float d_all = float(*(device const half*)(blk + 208));
        for (uint il = 0; il < 16; ++il) {
            uint ql_base = 32u * (il / 8u) + 16u * ((il / 2u) & 1u) + 8u * (il & 1u);
            uint qh_base = 16u * (il / 8u) + 8u * (il & 1u);
            float sc = float(scales_i8[(il % 2u) + 2u * (il / 2u)]);
            uint ph = (il / 2u) & 3u;
            uint kmask1 = (ph > 1u) ? ((ph > 2u) ? 0xC0C0C0C0u : 0x30303030u)
                                    : ((ph > 0u) ? 0x0C0C0C0Cu : 0x03030303u);
            uint kmask2 = (ph > 1u) ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
            float ml  = d_all * sc * 32.0f;
            float dl0 = d_all * sc;
            float dl1 = dl0 / 256.0f;
            float dl2 = dl1 / 256.0f;
            float dl3 = dl2 / 256.0f;
            uint shr_h = (ph > 2u) ? 2u : 0u;
            uint shl_h = (ph > 1u) ? 0u : ((ph > 0u) ? 2u : 4u);
            uint shr_l = (ph > 1u) ? 4u : 0u;
            uint base_k = kb * K_PER_BLK + il * 16u;
            for (uint i = 0; i < 4; ++i) {
                uint low  = (uint(ql_u16[ql_base + 2u*i]) | (uint(ql_u16[ql_base + 2u*i + 1u]) << 16)) & kmask2;
                uint high = (uint(qh_u16[qh_base + 2u*i]) | (uint(qh_u16[qh_base + 2u*i + 1u]) << 16)) & kmask1;
                uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
                float w0 = dl0 * float(q & 0x000000FFu) - ml;
                float w1 = dl1 * float(q & 0x0000FF00u) - ml;
                float w2 = dl2 * float(q & 0x00FF0000u) - ml;
                float w3 = dl3 * float(q & 0xFF000000u) - ml;
                uint k0 = base_k + i * 4u;
                float gamma0 = float(gamma[k0 + 0]);
                float gamma1 = float(gamma[k0 + 1]);
                float gamma2 = float(gamma[k0 + 2]);
                float gamma3 = float(gamma[k0 + 3]);
                for (uint v = 0; v < B_TILE; ++v) {
                    if (v < v_count) {
                        uint b = v_base + v;
                        device const half* xb = x + b * D_in;
                        float ir = inv_rms[v];
                        accs[v] += float(xb[k0 + 0]) * ir * gamma0 * w0
                                 + float(xb[k0 + 1]) * ir * gamma1 * w1
                                 + float(xb[k0 + 2]) * ir * gamma2 * w2
                                 + float(xb[k0 + 3]) * ir * gamma3 * w3;
                    }
                }
            }
        }
    }

    // Phase 4: per-batch reduction + write.
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            out[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q6_K_btile_qkv_otf_b1")]]
kernel void dense_gemv_q6_K_btile_qkv_otf_b1(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_qkv_otf_impl<1>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q6_K_btile_qkv_otf_b2")]]
kernel void dense_gemv_q6_K_btile_qkv_otf_b2(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_qkv_otf_impl<2>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q6_K_btile_qkv_otf_b4")]]
kernel void dense_gemv_q6_K_btile_qkv_otf_b4(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_qkv_otf_impl<4>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q6_K_btile_qkv_otf_b8")]]
kernel void dense_gemv_q6_K_btile_qkv_otf_b8(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_qkv_otf_impl<8>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}

// =============================================================================
// Q5_1 btile_qkv_otf — structural mirror with Q5_1 dequant. Block: 24 B /
// 32 elts. half d, half m, uchar qh[4], uchar qs[16]. Same outer skeleton
// as Q8_0 (32-elt blocks); inner dequant is the 5-bit unpack.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q5_1_btile_qkv_otf_impl(
    device const half* x,
    device const half* gamma,
    device const uchar* Wq_sw,
    device const uchar* Wk_sw,
    device const uchar* Wv_sw,
    device half* out_q,
    device half* out_k,
    device half* out_v,
    constant uint& D_in,
    constant uint& Q_nb,
    constant uint& K_nb,
    constant uint& V_nb,
    constant uint& D_out_q,
    constant uint& D_out_k,
    constant uint& D_out_v,
    constant float& eps,
    constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE],
    threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 24;
    constexpr uint K_PER_BLK = 32;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) {
            inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint n_block;
    uint D_out;
    device const uchar* W_sw;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab; D_out = D_out_q; W_sw = Wq_sw; out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb; D_out = D_out_k; W_sw = Wk_sw; out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb; D_out = D_out_v; W_sw = Wv_sw; out = out_v;
    }
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        float m = float(*(device const half*)(blk + 2));
        device const uchar* qh = blk + 4;
        device const uchar* qs = blk + 8;
        uint base_k = kb * K_PER_BLK;
        for (uint p = 0; p < 16; ++p) {
            uchar qsp = qs[p];
            uint h_lo = (qh[p / 8]       >> (p       % 8)) & 1u;
            uint h_hi = (qh[(p + 16) / 8] >> ((p + 16) % 8)) & 1u;
            uint q_lo = (uint(qsp) & 0xFu)        | (h_lo << 4);
            uint q_hi = ((uint(qsp) >> 4) & 0xFu) | (h_hi << 4);
            float w_lo = d * float(q_lo) + m;
            float w_hi = d * float(q_hi) + m;
            float gamma_lo = float(gamma[base_k + p]);
            float gamma_hi = float(gamma[base_k + p + 16]);
            for (uint v = 0; v < B_TILE; ++v) {
                if (v < v_count) {
                    uint b = v_base + v;
                    float x_lo = float(x[b * D_in + base_k + p]);
                    float x_hi = float(x[b * D_in + base_k + p + 16]);
                    float ir = inv_rms[v];
                    accs[v] += x_lo * ir * gamma_lo * w_lo
                             + x_hi * ir * gamma_hi * w_hi;
                }
            }
        }
    }

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            out[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q5_1_btile_qkv_otf_b1")]]
kernel void dense_gemv_q5_1_btile_qkv_otf_b1(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_qkv_otf_impl<1>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q5_1_btile_qkv_otf_b2")]]
kernel void dense_gemv_q5_1_btile_qkv_otf_b2(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_qkv_otf_impl<2>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q5_1_btile_qkv_otf_b4")]]
kernel void dense_gemv_q5_1_btile_qkv_otf_b4(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_qkv_otf_impl<4>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q5_1_btile_qkv_otf_b8")]]
kernel void dense_gemv_q5_1_btile_qkv_otf_b8(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_qkv_otf_impl<8>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}

// =============================================================================
// Q4_0 btile_qkv_otf — structural mirror with Q4_0 dequant (simplest 4bpw).
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q4_0_btile_qkv_otf_impl(
    device const half* x, device const half* gamma,
    device const uchar* Wq_sw, device const uchar* Wk_sw, device const uchar* Wv_sw,
    device half* out_q, device half* out_k, device half* out_v,
    constant uint& D_in,
    constant uint& Q_nb, constant uint& K_nb, constant uint& V_nb,
    constant uint& D_out_q, constant uint& D_out_k, constant uint& D_out_v,
    constant float& eps, constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE], threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 18;
    constexpr uint K_PER_BLK = 32;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint n_block;
    uint D_out;
    device const uchar* W_sw;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab; D_out = D_out_q; W_sw = Wq_sw; out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb; D_out = D_out_k; W_sw = Wk_sw; out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb; D_out = D_out_v; W_sw = Wv_sw; out = out_v;
    }
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const uchar* qs = blk + 2;
        uint base_k = kb * K_PER_BLK;
        for (uint p = 0; p < 16; ++p) {
            uchar byte = qs[p];
            float w_lo = d * float(int(byte & 0xF) - 8);
            float w_hi = d * float(int((byte >> 4) & 0xF) - 8);
            float gamma_lo = float(gamma[base_k + p]);
            float gamma_hi = float(gamma[base_k + p + 16]);
            for (uint v = 0; v < B_TILE; ++v) {
                if (v < v_count) {
                    uint b = v_base + v;
                    float x_lo = float(x[b * D_in + base_k + p]);
                    float x_hi = float(x[b * D_in + base_k + p + 16]);
                    float ir = inv_rms[v];
                    accs[v] += x_lo * ir * gamma_lo * w_lo
                             + x_hi * ir * gamma_hi * w_hi;
                }
            }
        }
    }

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            out[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q4_0_btile_qkv_otf_b1")]]
kernel void dense_gemv_q4_0_btile_qkv_otf_b1(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_qkv_otf_impl<1>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_0_btile_qkv_otf_b2")]]
kernel void dense_gemv_q4_0_btile_qkv_otf_b2(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_qkv_otf_impl<2>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_0_btile_qkv_otf_b4")]]
kernel void dense_gemv_q4_0_btile_qkv_otf_b4(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_qkv_otf_impl<4>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_0_btile_qkv_otf_b8")]]
kernel void dense_gemv_q4_0_btile_qkv_otf_b8(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_qkv_otf_impl<8>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}

// =============================================================================
// Q4_1 btile_qkv_otf — structural mirror with Q4_1 dequant (w = d*nib + m).
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q4_1_btile_qkv_otf_impl(
    device const half* x, device const half* gamma,
    device const uchar* Wq_sw, device const uchar* Wk_sw, device const uchar* Wv_sw,
    device half* out_q, device half* out_k, device half* out_v,
    constant uint& D_in,
    constant uint& Q_nb, constant uint& K_nb, constant uint& V_nb,
    constant uint& D_out_q, constant uint& D_out_k, constant uint& D_out_v,
    constant float& eps, constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE], threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 20;
    constexpr uint K_PER_BLK = 32;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint n_block;
    uint D_out;
    device const uchar* W_sw;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab; D_out = D_out_q; W_sw = Wq_sw; out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb; D_out = D_out_k; W_sw = Wk_sw; out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb; D_out = D_out_v; W_sw = Wv_sw; out = out_v;
    }
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        float m = float(*(device const half*)(blk + 2));
        device const uchar* qs = blk + 4;
        uint base_k = kb * K_PER_BLK;
        for (uint p = 0; p < 16; ++p) {
            uchar byte = qs[p];
            float w_lo = d * float(byte & 0xF) + m;
            float w_hi = d * float((byte >> 4) & 0xF) + m;
            float gamma_lo = float(gamma[base_k + p]);
            float gamma_hi = float(gamma[base_k + p + 16]);
            for (uint v = 0; v < B_TILE; ++v) {
                if (v < v_count) {
                    uint b = v_base + v;
                    float x_lo = float(x[b * D_in + base_k + p]);
                    float x_hi = float(x[b * D_in + base_k + p + 16]);
                    float ir = inv_rms[v];
                    accs[v] += x_lo * ir * gamma_lo * w_lo
                             + x_hi * ir * gamma_hi * w_hi;
                }
            }
        }
    }

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            out[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q4_1_btile_qkv_otf_b1")]]
kernel void dense_gemv_q4_1_btile_qkv_otf_b1(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_qkv_otf_impl<1>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_1_btile_qkv_otf_b2")]]
kernel void dense_gemv_q4_1_btile_qkv_otf_b2(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_qkv_otf_impl<2>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_1_btile_qkv_otf_b4")]]
kernel void dense_gemv_q4_1_btile_qkv_otf_b4(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_qkv_otf_impl<4>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_1_btile_qkv_otf_b8")]]
kernel void dense_gemv_q4_1_btile_qkv_otf_b8(
    device const half* x        [[buffer(0)]], device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]], device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]], device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]], device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]], constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]], constant uint& V_nb        [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]], constant uint& D_out_k     [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]], constant float& eps        [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_qkv_otf_impl<8>(x, gamma, Wq_sw, Wk_sw, Wv_sw, out_q, out_k, out_v,
        D_in, Q_nb, K_nb, V_nb, D_out_q, D_out_k, D_out_v, eps, numVecs,
        inv_rms, ss_stage, partials, tg, lid, sg_id);
}

// Templated QKV with K-tile staging (Item C of the kernel zoo followup
// plan). Targets activeB ∈ {2,3,4} where neither extreme of the existing
// dispatcher wins cleanly: OTF's per-FMA gamma + inv_rms re-reads
// accumulate, and V6 grid-shrink wastes parallelism on small batches.
//
// Design relative to OTF:
//   - Same Phase 1 (per-batch inv_rms reduction) and Phase 2 (slab routing).
//   - Phase 3 changes: instead of OTF's per-FMA reads of x and gamma from
//     DRAM, each SG independently stages K_TILE=256 elements per batch
//     into its own slice of tg-mem, with inv_rms and gamma applied
//     during the load. The inner FMA loop then reads h_norm from tg-mem
//     and applies only the W coefficient — pure mul-acc that the
//     compiler can SIMD-coalesce.
//   - Per-SG staging avoids cross-SG synchronization (each SG sees only
//     its 1/4 K-slice). The N_SPLITS=4 reduction structure is preserved.
//
// Costs vs OTF: simdgroup_barrier per K-tile (cheap, SG-internal), plus
// the staging buffer in tg-mem. tg-mem footprint:
//   B_TILE  | per-SG slice | total (4 SGs)
//   1       | 512 B        | 2 KB
//   2       | 1 KB         | 4 KB
//   4       | 2 KB         | 8 KB
//   8       | 4 KB         | 16 KB
// All comfortably under the 32 KB tg-mem cap.
//
// Expected payoff (per the plan): 1-2 ms per AR step at activeB ∈ {2,3,4}.
// Falsification: if tiled_b4 doesn't beat V6 at numVecs=4 in profile, the
// barriers + staging overhead outweigh the gamma/inv_rms hoist benefit.
template<uint B_TILE>
inline void dense_gemv_q8_0_btile_qkv_tiled_impl(
    device const half* x,
    device const half* gamma,
    device const uchar* Wq_sw,
    device const uchar* Wk_sw,
    device const uchar* Wv_sw,
    device half* out_q,
    device half* out_k,
    device half* out_v,
    constant uint& D_in,
    constant uint& Q_nb,
    constant uint& K_nb,
    constant uint& V_nb,
    constant uint& D_out_q,
    constant uint& D_out_k,
    constant uint& D_out_v,
    constant float& eps,
    constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE],
    threadgroup float (&ss_stage)[4],
    threadgroup half* h_norm_tile_global,    // [4 SGs * B_TILE * K_TILE]
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint K_TILE = 256;             // elements per tile
    constexpr uint K_TILE_KB = K_TILE / 32;  // = 8 kbs per tile
    constexpr uint BLK = 34;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    // Phase 1: per-batch inv_rms (same as OTF).
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) {
            inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: route slab to Q/K/V (same as OTF).
    uint n_block;
    uint D_out;
    device const uchar* W_sw;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab; D_out = D_out_q; W_sw = Wq_sw; out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb; D_out = D_out_k; W_sw = Wk_sw; out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb; D_out = D_out_v; W_sw = Wv_sw; out = out_v;
    }
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;
    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    // Phase 3: K-tile staging + FMA.
    threadgroup half* my_tile = h_norm_tile_global + sg_id * B_TILE * K_TILE;
    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;

    for (uint kb_tile_start = kb_begin; kb_tile_start < kb_end; kb_tile_start += K_TILE_KB) {
        uint k_elem_start = kb_tile_start * 32;
        // Cooperative SG-local load: 32 threads load K_TILE=256 elements
        // per batch (8 elems per thread per batch). Apply inv_rms and
        // gamma during the load so the FMA loop sees pre-normalized values.
        for (uint v = 0; v < B_TILE; ++v) {
            if (v >= v_count) break;
            uint b = v_base + v;
            float ir = inv_rms[v];
            for (uint i = lid_sg; i < K_TILE; i += 32) {
                uint global_i = k_elem_start + i;
                float xv = float(x[b * D_in + global_i]);
                float gv = float(gamma[global_i]);
                my_tile[v * K_TILE + i] = half(xv * ir * gv);
            }
        }
        // SG-internal barrier ensures the load completes before the FMA
        // reads from my_tile. Cheap (within-SG-only, no cross-SG sync).
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // FMA over this K-tile. K_TILE_KB=8 kb iterations, each a
        // 32-element block.
        for (uint kb_local = 0; kb_local < K_TILE_KB; ++kb_local) {
            uint kb = kb_tile_start + kb_local;
            device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
            float d = float(*(device const half*)(blk));
            device const char* qs = (device const char*)(blk + 2);
            uint base_local = kb_local * 32;
            for (uint p = 0; p < 32; ++p) {
                float w_p = d * float(qs[p]);
                for (uint v = 0; v < B_TILE; ++v) {
                    if (v < v_count) {
                        accs[v] += float(my_tile[v * K_TILE + base_local + p]) * w_p;
                    }
                }
            }
        }
        // Barrier before next tile load to ensure FMA completes.
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 4: per-batch reduction across N_SPLITS SGs and write outputs.
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            out[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q8_0_btile_qkv_tiled_b1")]]
kernel void dense_gemv_q8_0_btile_qkv_tiled_b1(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1];
    threadgroup float ss_stage[4];
    threadgroup half h_norm_tile[4 * 1 * 256];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_qkv_tiled_impl<1>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                             out_q, out_k, out_v,
                                             D_in, Q_nb, K_nb, V_nb,
                                             D_out_q, D_out_k, D_out_v,
                                             eps, numVecs,
                                             inv_rms, ss_stage, h_norm_tile, partials,
                                             tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_qkv_tiled_b2")]]
kernel void dense_gemv_q8_0_btile_qkv_tiled_b2(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2];
    threadgroup float ss_stage[4];
    threadgroup half h_norm_tile[4 * 2 * 256];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_qkv_tiled_impl<2>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                             out_q, out_k, out_v,
                                             D_in, Q_nb, K_nb, V_nb,
                                             D_out_q, D_out_k, D_out_v,
                                             eps, numVecs,
                                             inv_rms, ss_stage, h_norm_tile, partials,
                                             tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_qkv_tiled_b4")]]
kernel void dense_gemv_q8_0_btile_qkv_tiled_b4(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4];
    threadgroup float ss_stage[4];
    threadgroup half h_norm_tile[4 * 4 * 256];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_qkv_tiled_impl<4>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                             out_q, out_k, out_v,
                                             D_in, Q_nb, K_nb, V_nb,
                                             D_out_q, D_out_k, D_out_v,
                                             eps, numVecs,
                                             inv_rms, ss_stage, h_norm_tile, partials,
                                             tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_qkv_tiled_b8")]]
kernel void dense_gemv_q8_0_btile_qkv_tiled_b8(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wq_sw   [[buffer(2)]],
    device const uchar* Wk_sw   [[buffer(3)]],
    device const uchar* Wv_sw   [[buffer(4)]],
    device half* out_q          [[buffer(5)]],
    device half* out_k          [[buffer(6)]],
    device half* out_v          [[buffer(7)]],
    constant uint& D_in         [[buffer(8)]],
    constant uint& Q_nb         [[buffer(9)]],
    constant uint& K_nb         [[buffer(10)]],
    constant uint& V_nb         [[buffer(11)]],
    constant uint& D_out_q      [[buffer(12)]],
    constant uint& D_out_k      [[buffer(13)]],
    constant uint& D_out_v      [[buffer(14)]],
    constant float& eps         [[buffer(15)]],
    constant uint& numVecs      [[buffer(16)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8];
    threadgroup float ss_stage[4];
    threadgroup half h_norm_tile[4 * 8 * 256];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_qkv_tiled_impl<8>(x, gamma, Wq_sw, Wk_sw, Wv_sw,
                                             out_q, out_k, out_v,
                                             D_in, Q_nb, K_nb, V_nb,
                                             D_out_q, D_out_k, D_out_v,
                                             eps, numVecs,
                                             inv_rms, ss_stage, h_norm_tile, partials,
                                             tg, lid, sg_id);
}

// V7 family: same swizzled Q8_0 layout as V6, but each threadgroup
// amortizes weight reads across VEC_TILE input vectors via per-thread
// register accumulators (the same pattern V4 uses across B for AR).
// V6 dispatched (D_out/32, numVecs) TGs; V7 dispatches
// (D_out/32, ceil(numVecs / VEC_TILE)) TGs with each handling VEC_TILE
// vectors. Per K-tile: load weight ONCE, FMA into VEC_TILE accumulators.
// At numVecs ≥ VEC_TILE this is a structural ~VEC_TILE× weight-bandwidth
// reduction on prefill matmuls.
//
// VEC_TILE=4 chosen because tg-mem h_norms[VEC_TILE * MAX_D_IN] halves =
// 4 × 2816 × 2 = 22.5 KB, comfortably under Apple GPU's 32 KB tg-mem cap.
// VEC_TILE=8 would be 45 KB and exceed the budget.

kernel void dense_gemv_q8_0_v7_rmsnorm_qkv(
    device const half* x                [[buffer(0)]],
    device const half* gamma            [[buffer(1)]],
    device const uchar* Wq_sw           [[buffer(2)]],
    device const uchar* Wk_sw           [[buffer(3)]],
    device const uchar* Wv_sw           [[buffer(4)]],
    device half* out_q                  [[buffer(5)]],
    device half* out_k                  [[buffer(6)]],
    device half* out_v                  [[buffer(7)]],
    constant uint& D_in                 [[buffer(8)]],
    constant uint& Q_nb                 [[buffer(9)]],
    constant uint& K_nb                 [[buffer(10)]],
    constant uint& V_nb                 [[buffer(11)]],
    constant uint& D_out_q              [[buffer(12)]],
    constant uint& D_out_k              [[buffer(13)]],
    constant uint& D_out_v              [[buffer(14)]],
    constant float& eps                 [[buffer(15)]],
    constant uint& numVecs              [[buffer(16)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]])
{
    constexpr uint VEC_TILE = 4;
    constexpr uint MAX_D_IN = 2816;
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 34;
    constexpr uint THREADS = 128;
    threadgroup half h_norms[VEC_TILE * MAX_D_IN];
    threadgroup float ss_stage[N_SPLITS];

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * VEC_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < VEC_TILE) ? (numVecs - v_base) : VEC_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    // Phase 1: For each vector v ∈ [0..v_count), compute RMS-normalized
    // h_norms[v * MAX_D_IN + i]. Each iteration uses ss_stage transiently;
    // a barrier between iterations ensures stage doesn't get re-clobbered
    // before all threads have read total_ss.
    for (uint v = 0; v < v_count; ++v) {
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        float inv_rms = rsqrt(total_ss / float(D_in) + eps);
        for (uint i = tid; i < D_in; i += THREADS) {
            h_norms[v * MAX_D_IN + i] = half(float(xb[i]) * inv_rms * float(gamma[i]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: Route slab to Q/K/V projection.
    uint n_block;
    uint D_out;
    device const uchar* W_sw;
    device half* out;
    if (slab < Q_nb) {
        n_block = slab;
        D_out = D_out_q; W_sw = Wq_sw; out = out_q;
    } else if (slab < Q_nb + K_nb) {
        n_block = slab - Q_nb;
        D_out = D_out_k; W_sw = Wk_sw; out = out_k;
    } else {
        n_block = slab - Q_nb - K_nb;
        D_out = D_out_v; W_sw = Wv_sw; out = out_v;
    }

    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_super = W_sw + n_block * super_bytes;
    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    // Phase 3: Amortized matmul. Each thread holds VEC_TILE accumulators
    // in registers. Per K-tile: load weights once, FMA against all v_count
    // vectors. The compile-time VEC_TILE bound guarantees the inner loop
    // unrolls; v_count gates which iterations run actual work vs no-op.
    float accs[VEC_TILE];
    for (uint v = 0; v < VEC_TILE; ++v) accs[v] = 0.0f;

    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        for (uint p = 0; p < 32; ++p) {
            float w_p = d * float(qs[p]);
            for (uint v = 0; v < VEC_TILE; ++v) {
                if (v < v_count) {
                    accs[v] += float(h_norms[v * MAX_D_IN + base_k + p]) * w_p;
                }
            }
        }
    }

    // Phase 4: Reduce partials across N_SPLITS subgroups, write VEC_TILE
    // outputs. Reuse one partials buffer across vectors with a barrier
    // between iterations.
    threadgroup float partials[N_SPLITS][32];
    for (uint v = 0; v < v_count; ++v) {
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            out[b * D_out + n] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
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

// F16 mirror of dense_gemv_q8_0_v6_rmsnorm_gate_up: same fused [gate|up]
// dispatch with [slot, 2*D_out] output layout, but weights are plain
// row-major fp16 [D_out, D_in] (no swizzle, no per-block scales). Inner
// loop drops the dequant mul.
kernel void dense_gemv_f16_v6_rmsnorm_gate_up(
    device const half* x                [[buffer(0)]],
    device const half* gamma            [[buffer(1)]],
    device const half* Wg               [[buffer(2)]],
    device const half* Wu               [[buffer(3)]],
    device half* fused_out              [[buffer(4)]],
    constant uint& D_in                 [[buffer(5)]],
    constant uint& D_out                [[buffer(6)]],
    constant float& eps                 [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]])
{
    constexpr uint N_SPLITS = 4;
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
    device const half* W = is_up ? Wu : Wg;
    device const half* w_row = W + n * D_in;

    float acc = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        uint base_k = kb * 32;
        device const half* w_blk = w_row + base_k;
        for (uint p = 0; p < 32; ++p) {
            acc += float(h_norm[base_k + p]) * float(w_blk[p]);
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

// =============================================================================
// Q8_0 RMSNorm + gate_up kernel zoo OTF (on-the-fly normalize) at compile-
// time fixed B_TILE ∈ {1, 2, 4, 8}. Mirror of the QKV otf zoo but writes
// to fused [slots, 2*D_out] gate|up output instead of three separate Q/K/V
// buffers. Same per-batch inv_rms staging (only B_TILE * 4 B in tg-mem).
template<uint B_TILE>
inline void dense_gemv_q8_0_btile_gate_up_otf_impl(
    device const half* x,
    device const half* gamma,
    device const uchar* Wg_sw,
    device const uchar* Wu_sw,
    device half* fused_out,
    constant uint& D_in,
    constant uint& D_out,             // = N_half (gate output dim = up output dim)
    constant float& eps,
    constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE],
    threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 34;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    // Slab routing: first D_out/32 slabs are gate, next D_out/32 are up.
    uint n_blocks_per_branch = D_out / 32;
    bool is_up = slab >= n_blocks_per_branch;
    uint n_block = is_up ? slab - n_blocks_per_branch : slab;

    // Phase 1: per-batch RMS — same as QKV otf.
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) {
            inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: GEMV with on-the-fly normalize. Read x and gamma directly.
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_sw = is_up ? Wu_sw : Wg_sw;
    device const uchar* W_super = W_sw + n_block * super_bytes;
    uint kb_per_sg = nbc / N_SPLITS;
    uint kb_begin = sg_id * kb_per_sg;
    uint kb_end = kb_begin + kb_per_sg;

    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const char* qs = (device const char*)(blk + 2);
        uint base_k = kb * 32;
        for (uint p = 0; p < 32; ++p) {
            float w_p = d * float(qs[p]);
            float gamma_p = float(gamma[base_k + p]);
            for (uint v = 0; v < B_TILE; ++v) {
                if (v < v_count) {
                    uint b = v_base + v;
                    float x_val = float(x[b * D_in + base_k + p]);
                    accs[v] += x_val * inv_rms[v] * gamma_p * w_p;
                }
            }
        }
    }

    // Phase 3: per-batch reduction and write to fused [slot, 2*D_out] layout.
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            uint col = is_up ? (D_out + n) : n;
            fused_out[b * 2 * D_out + col] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q8_0_btile_gate_up_otf_b1")]]
kernel void dense_gemv_q8_0_btile_gate_up_otf_b1(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wg_sw   [[buffer(2)]],
    device const uchar* Wu_sw   [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_gate_up_otf_impl<1>(x, gamma, Wg_sw, Wu_sw, fused_out,
                                                D_in, D_out, eps, numVecs,
                                                inv_rms, ss_stage, partials,
                                                tg, lid, sg_id);
}

// =============================================================================
// F16 mirror of the Q8_0 RMSNorm + gate_up otf zoo. Same fused
// [slot, 2*D_out] output and per-batch inv_rms staging, but weights are
// plain row-major fp16 [D_out, D_in]. Only B_TILE=1 is needed by this
// scope (b1 = the AR-decode path).
template<uint B_TILE>
inline void dense_gemv_f16_btile_gate_up_otf_impl(
    device const half* x,
    device const half* gamma,
    device const half* Wg,
    device const half* Wu,
    device half* fused_out,
    constant uint& D_in,
    constant uint& D_out,             // = N_half (gate output dim = up output dim)
    constant float& eps,
    constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE],
    threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    // Slab routing: first D_out/32 slabs are gate, next D_out/32 are up.
    uint n_blocks_per_branch = D_out / 32;
    bool is_up = slab >= n_blocks_per_branch;
    uint n_block = is_up ? slab - n_blocks_per_branch : slab;

    // Phase 1: per-batch RMS — same as QKV otf.
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) {
            inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: GEMV with on-the-fly normalize. Read x and gamma directly.
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / 32;
    device const half* W = is_up ? Wu : Wg;
    device const half* w_row = W + n * D_in;

    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        uint base_k = kb * 32;
        device const half* w_blk = w_row + base_k;
        for (uint p = 0; p < 32; ++p) {
            float w_p = float(w_blk[p]);
            float gamma_p = float(gamma[base_k + p]);
            for (uint v = 0; v < B_TILE; ++v) {
                if (v < v_count) {
                    uint b = v_base + v;
                    float x_val = float(x[b * D_in + base_k + p]);
                    accs[v] += x_val * inv_rms[v] * gamma_p * w_p;
                }
            }
        }
    }

    // Phase 3: per-batch reduction and write to fused [slot, 2*D_out] layout.
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            uint col = is_up ? (D_out + n) : n;
            fused_out[b * 2 * D_out + col] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_f16_btile_gate_up_otf_b1")]]
kernel void dense_gemv_f16_btile_gate_up_otf_b1(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const half* Wg       [[buffer(2)]],
    device const half* Wu       [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_gate_up_otf_impl<1>(x, gamma, Wg, Wu, fused_out,
                                              D_in, D_out, eps, numVecs,
                                              inv_rms, ss_stage, partials,
                                              tg, lid, sg_id);
}

// F16 OTF gate+up — B_TILE ∈ {2, 4, 8}. Same rationale as the QKV
// b{2,4,8} additions above: F16 has no per-FMA dequant cost, so the
// OTF inner loop is tighter than Q8_0's, and amortizing RMS across
// B_TILE batches per TG (avoiding V6's per-(slab,batch) RMS redo)
// is a clear win at activeB > 1.
[[host_name("dense_gemv_f16_btile_gate_up_otf_b2")]]
kernel void dense_gemv_f16_btile_gate_up_otf_b2(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const half* Wg       [[buffer(2)]],
    device const half* Wu       [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_gate_up_otf_impl<2>(x, gamma, Wg, Wu, fused_out,
                                              D_in, D_out, eps, numVecs,
                                              inv_rms, ss_stage, partials,
                                              tg, lid, sg_id);
}

[[host_name("dense_gemv_f16_btile_gate_up_otf_b4")]]
kernel void dense_gemv_f16_btile_gate_up_otf_b4(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const half* Wg       [[buffer(2)]],
    device const half* Wu       [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_gate_up_otf_impl<4>(x, gamma, Wg, Wu, fused_out,
                                              D_in, D_out, eps, numVecs,
                                              inv_rms, ss_stage, partials,
                                              tg, lid, sg_id);
}

[[host_name("dense_gemv_f16_btile_gate_up_otf_b8")]]
kernel void dense_gemv_f16_btile_gate_up_otf_b8(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const half* Wg       [[buffer(2)]],
    device const half* Wu       [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_f16_btile_gate_up_otf_impl<8>(x, gamma, Wg, Wu, fused_out,
                                              D_in, D_out, eps, numVecs,
                                              inv_rms, ss_stage, partials,
                                              tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_gate_up_otf_b2")]]
kernel void dense_gemv_q8_0_btile_gate_up_otf_b2(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wg_sw   [[buffer(2)]],
    device const uchar* Wu_sw   [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_gate_up_otf_impl<2>(x, gamma, Wg_sw, Wu_sw, fused_out,
                                                D_in, D_out, eps, numVecs,
                                                inv_rms, ss_stage, partials,
                                                tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_gate_up_otf_b4")]]
kernel void dense_gemv_q8_0_btile_gate_up_otf_b4(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wg_sw   [[buffer(2)]],
    device const uchar* Wu_sw   [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_gate_up_otf_impl<4>(x, gamma, Wg_sw, Wu_sw, fused_out,
                                                D_in, D_out, eps, numVecs,
                                                inv_rms, ss_stage, partials,
                                                tg, lid, sg_id);
}

[[host_name("dense_gemv_q8_0_btile_gate_up_otf_b8")]]
kernel void dense_gemv_q8_0_btile_gate_up_otf_b8(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wg_sw   [[buffer(2)]],
    device const uchar* Wu_sw   [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q8_0_btile_gate_up_otf_impl<8>(x, gamma, Wg_sw, Wu_sw, fused_out,
                                                D_in, D_out, eps, numVecs,
                                                inv_rms, ss_stage, partials,
                                                tg, lid, sg_id);
}

// =============================================================================
// Q5_K btile_gate_up_otf — structural mirror of the Q8_0 gate_up_otf zoo.
// Same 3-phase shape (RMS reduction → slab-routed GEMV with OTF normalize
// → per-batch reduction writing to fused [slot, 2*D_out] layout). Phase 2's
// per-element dequant is Q5_K (256 elts × 176 B) instead of Q8_0 (32 × 34).
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q5_K_btile_gate_up_otf_impl(
    device const half* x,
    device const half* gamma,
    device const uchar* Wg_sw,
    device const uchar* Wu_sw,
    device half* fused_out,
    constant uint& D_in,
    constant uint& D_out,
    constant float& eps,
    constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE],
    threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 176;
    constexpr uint K_PER_BLK = 256;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    // Slab routing: gate then up (same as Q8_0).
    uint n_blocks_per_branch = D_out / 32;
    bool is_up = slab >= n_blocks_per_branch;
    uint n_block = is_up ? slab - n_blocks_per_branch : slab;

    // Phase 1: per-batch RMS reduction (format-agnostic).
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) {
            inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: GEMV with on-the-fly normalize, Q5_K dequant.
    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_sw = is_up ? Wu_sw : Wg_sw;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d_all = float(*(device const half*)(blk + 0));
        float dmin  = float(*(device const half*)(blk + 2));
        device const uchar* scales = blk + 4;
        device const uchar* qh     = blk + 16;
        device const uchar* qs     = blk + 48;
        for (uint il_orig = 0; il_orig < 16; ++il_orig) {
            uint q_off  = 32u * (il_orig / 4u) + 16u * (il_orig & 1u);
            uint qh_off = 16u * (il_orig & 1u);
            uchar ul    = uchar(1u << (il_orig / 2u));
            uint is     = (il_orig / 4u) * 2u;
            uint ph     = il_orig & 3u;
            uchar2 sc   = get_scale_min_k4_just2(is, ph / 2u, scales);
            float d_eff = (ph < 2u) ? d_all : (d_all / 16.0f);
            float dl    = d_eff * float(sc[0]);
            float ml    = dmin  * float(sc[1]);
            uchar mask  = (ph < 2u) ? 0x0F : 0xF0;
            float qhv   = (ph < 2u) ? 16.0f : 256.0f;
            uint base_k = kb * K_PER_BLK + il_orig * 16u;
            for (uint i = 0; i < 16; ++i) {
                float v_q = float(qs[q_off + i] & mask)
                          + (((qh[qh_off + i] & ul) != 0) ? qhv : 0.0f);
                float w_p = dl * v_q - ml;
                float gamma_p = float(gamma[base_k + i]);
                for (uint v = 0; v < B_TILE; ++v) {
                    if (v < v_count) {
                        uint b = v_base + v;
                        float x_val = float(x[b * D_in + base_k + i]);
                        accs[v] += x_val * inv_rms[v] * gamma_p * w_p;
                    }
                }
            }
        }
    }

    // Phase 3: per-batch reduction + write to fused [slot, 2*D_out] layout.
    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            uint col = is_up ? (D_out + n) : n;
            fused_out[b * 2 * D_out + col] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q5_K_btile_gate_up_otf_b1")]]
kernel void dense_gemv_q5_K_btile_gate_up_otf_b1(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wg_sw   [[buffer(2)]],
    device const uchar* Wu_sw   [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_gate_up_otf_impl<1>(x, gamma, Wg_sw, Wu_sw, fused_out,
                                                D_in, D_out, eps, numVecs,
                                                inv_rms, ss_stage, partials,
                                                tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_K_btile_gate_up_otf_b2")]]
kernel void dense_gemv_q5_K_btile_gate_up_otf_b2(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wg_sw   [[buffer(2)]],
    device const uchar* Wu_sw   [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_gate_up_otf_impl<2>(x, gamma, Wg_sw, Wu_sw, fused_out,
                                                D_in, D_out, eps, numVecs,
                                                inv_rms, ss_stage, partials,
                                                tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_K_btile_gate_up_otf_b4")]]
kernel void dense_gemv_q5_K_btile_gate_up_otf_b4(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wg_sw   [[buffer(2)]],
    device const uchar* Wu_sw   [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_gate_up_otf_impl<4>(x, gamma, Wg_sw, Wu_sw, fused_out,
                                                D_in, D_out, eps, numVecs,
                                                inv_rms, ss_stage, partials,
                                                tg, lid, sg_id);
}

[[host_name("dense_gemv_q5_K_btile_gate_up_otf_b8")]]
kernel void dense_gemv_q5_K_btile_gate_up_otf_b8(
    device const half* x        [[buffer(0)]],
    device const half* gamma    [[buffer(1)]],
    device const uchar* Wg_sw   [[buffer(2)]],
    device const uchar* Wu_sw   [[buffer(3)]],
    device half* fused_out      [[buffer(4)]],
    constant uint& D_in         [[buffer(5)]],
    constant uint& D_out        [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    constant uint& numVecs      [[buffer(8)]],
    uint2 tg                    [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8];
    threadgroup float ss_stage[4];
    threadgroup float partials[4][32];
    dense_gemv_q5_K_btile_gate_up_otf_impl<8>(x, gamma, Wg_sw, Wu_sw, fused_out,
                                                D_in, D_out, eps, numVecs,
                                                inv_rms, ss_stage, partials,
                                                tg, lid, sg_id);
}

// =============================================================================
// Q6_K btile_gate_up_otf — structural mirror with Q6_K dequant.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q6_K_btile_gate_up_otf_impl(
    device const half* x, device const half* gamma,
    device const uchar* Wg_sw, device const uchar* Wu_sw,
    device half* fused_out,
    constant uint& D_in, constant uint& D_out, constant float& eps, constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE], threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 210;
    constexpr uint K_PER_BLK = 256;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    uint n_blocks_per_branch = D_out / 32;
    bool is_up = slab >= n_blocks_per_branch;
    uint n_block = is_up ? slab - n_blocks_per_branch : slab;

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) {
            inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_sw = is_up ? Wu_sw : Wg_sw;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        device const ushort* ql_u16    = (device const ushort*)(blk + 0);
        device const ushort* qh_u16    = (device const ushort*)(blk + 128);
        device const char*   scales_i8 = (device const char*)(blk + 192);
        float d_all = float(*(device const half*)(blk + 208));
        for (uint il = 0; il < 16; ++il) {
            uint ql_base = 32u * (il / 8u) + 16u * ((il / 2u) & 1u) + 8u * (il & 1u);
            uint qh_base = 16u * (il / 8u) + 8u * (il & 1u);
            float sc = float(scales_i8[(il % 2u) + 2u * (il / 2u)]);
            uint ph = (il / 2u) & 3u;
            uint kmask1 = (ph > 1u) ? ((ph > 2u) ? 0xC0C0C0C0u : 0x30303030u)
                                    : ((ph > 0u) ? 0x0C0C0C0Cu : 0x03030303u);
            uint kmask2 = (ph > 1u) ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
            float ml  = d_all * sc * 32.0f;
            float dl0 = d_all * sc;
            float dl1 = dl0 / 256.0f;
            float dl2 = dl1 / 256.0f;
            float dl3 = dl2 / 256.0f;
            uint shr_h = (ph > 2u) ? 2u : 0u;
            uint shl_h = (ph > 1u) ? 0u : ((ph > 0u) ? 2u : 4u);
            uint shr_l = (ph > 1u) ? 4u : 0u;
            uint base_k = kb * K_PER_BLK + il * 16u;
            for (uint i = 0; i < 4; ++i) {
                uint low  = (uint(ql_u16[ql_base + 2u*i]) | (uint(ql_u16[ql_base + 2u*i + 1u]) << 16)) & kmask2;
                uint high = (uint(qh_u16[qh_base + 2u*i]) | (uint(qh_u16[qh_base + 2u*i + 1u]) << 16)) & kmask1;
                uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
                float w0 = dl0 * float(q & 0x000000FFu) - ml;
                float w1 = dl1 * float(q & 0x0000FF00u) - ml;
                float w2 = dl2 * float(q & 0x00FF0000u) - ml;
                float w3 = dl3 * float(q & 0xFF000000u) - ml;
                uint k0 = base_k + i * 4u;
                float gamma0 = float(gamma[k0 + 0]);
                float gamma1 = float(gamma[k0 + 1]);
                float gamma2 = float(gamma[k0 + 2]);
                float gamma3 = float(gamma[k0 + 3]);
                for (uint v = 0; v < B_TILE; ++v) {
                    if (v < v_count) {
                        uint b = v_base + v;
                        device const half* xb = x + b * D_in;
                        float ir = inv_rms[v];
                        accs[v] += float(xb[k0 + 0]) * ir * gamma0 * w0
                                 + float(xb[k0 + 1]) * ir * gamma1 * w1
                                 + float(xb[k0 + 2]) * ir * gamma2 * w2
                                 + float(xb[k0 + 3]) * ir * gamma3 * w3;
                    }
                }
            }
        }
    }

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            uint col = is_up ? (D_out + n) : n;
            fused_out[b * 2 * D_out + col] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q6_K_btile_gate_up_otf_b1")]]
kernel void dense_gemv_q6_K_btile_gate_up_otf_b1(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_gate_up_otf_impl<1>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q6_K_btile_gate_up_otf_b2")]]
kernel void dense_gemv_q6_K_btile_gate_up_otf_b2(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_gate_up_otf_impl<2>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q6_K_btile_gate_up_otf_b4")]]
kernel void dense_gemv_q6_K_btile_gate_up_otf_b4(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_gate_up_otf_impl<4>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q6_K_btile_gate_up_otf_b8")]]
kernel void dense_gemv_q6_K_btile_gate_up_otf_b8(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q6_K_btile_gate_up_otf_impl<8>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}

// =============================================================================
// Q5_1 btile_gate_up_otf — structural mirror with Q5_1 dequant.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q5_1_btile_gate_up_otf_impl(
    device const half* x, device const half* gamma,
    device const uchar* Wg_sw, device const uchar* Wu_sw,
    device half* fused_out,
    constant uint& D_in, constant uint& D_out, constant float& eps, constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE], threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 24;
    constexpr uint K_PER_BLK = 32;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    uint n_blocks_per_branch = D_out / 32;
    bool is_up = slab >= n_blocks_per_branch;
    uint n_block = is_up ? slab - n_blocks_per_branch : slab;

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) {
            inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_sw = is_up ? Wu_sw : Wg_sw;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        float m = float(*(device const half*)(blk + 2));
        device const uchar* qh = blk + 4;
        device const uchar* qs = blk + 8;
        uint base_k = kb * K_PER_BLK;
        for (uint p = 0; p < 16; ++p) {
            uchar qsp = qs[p];
            uint h_lo = (qh[p / 8]       >> (p       % 8)) & 1u;
            uint h_hi = (qh[(p + 16) / 8] >> ((p + 16) % 8)) & 1u;
            uint q_lo = (uint(qsp) & 0xFu)        | (h_lo << 4);
            uint q_hi = ((uint(qsp) >> 4) & 0xFu) | (h_hi << 4);
            float w_lo = d * float(q_lo) + m;
            float w_hi = d * float(q_hi) + m;
            float gamma_lo = float(gamma[base_k + p]);
            float gamma_hi = float(gamma[base_k + p + 16]);
            for (uint v = 0; v < B_TILE; ++v) {
                if (v < v_count) {
                    uint b = v_base + v;
                    float x_lo = float(x[b * D_in + base_k + p]);
                    float x_hi = float(x[b * D_in + base_k + p + 16]);
                    float ir = inv_rms[v];
                    accs[v] += x_lo * ir * gamma_lo * w_lo
                             + x_hi * ir * gamma_hi * w_hi;
                }
            }
        }
    }

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            uint col = is_up ? (D_out + n) : n;
            fused_out[b * 2 * D_out + col] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q5_1_btile_gate_up_otf_b1")]]
kernel void dense_gemv_q5_1_btile_gate_up_otf_b1(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_gate_up_otf_impl<1>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q5_1_btile_gate_up_otf_b2")]]
kernel void dense_gemv_q5_1_btile_gate_up_otf_b2(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_gate_up_otf_impl<2>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q5_1_btile_gate_up_otf_b4")]]
kernel void dense_gemv_q5_1_btile_gate_up_otf_b4(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_gate_up_otf_impl<4>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q5_1_btile_gate_up_otf_b8")]]
kernel void dense_gemv_q5_1_btile_gate_up_otf_b8(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q5_1_btile_gate_up_otf_impl<8>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}

// =============================================================================
// Q4_0 btile_gate_up_otf — structural mirror with Q4_0 dequant.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q4_0_btile_gate_up_otf_impl(
    device const half* x, device const half* gamma,
    device const uchar* Wg_sw, device const uchar* Wu_sw,
    device half* fused_out,
    constant uint& D_in, constant uint& D_out, constant float& eps, constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE], threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 18;
    constexpr uint K_PER_BLK = 32;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    uint n_blocks_per_branch = D_out / 32;
    bool is_up = slab >= n_blocks_per_branch;
    uint n_block = is_up ? slab - n_blocks_per_branch : slab;

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_sw = is_up ? Wu_sw : Wg_sw;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        device const uchar* qs = blk + 2;
        uint base_k = kb * K_PER_BLK;
        for (uint p = 0; p < 16; ++p) {
            uchar byte = qs[p];
            float w_lo = d * float(int(byte & 0xF) - 8);
            float w_hi = d * float(int((byte >> 4) & 0xF) - 8);
            float gamma_lo = float(gamma[base_k + p]);
            float gamma_hi = float(gamma[base_k + p + 16]);
            for (uint v = 0; v < B_TILE; ++v) {
                if (v < v_count) {
                    uint b = v_base + v;
                    float x_lo = float(x[b * D_in + base_k + p]);
                    float x_hi = float(x[b * D_in + base_k + p + 16]);
                    float ir = inv_rms[v];
                    accs[v] += x_lo * ir * gamma_lo * w_lo
                             + x_hi * ir * gamma_hi * w_hi;
                }
            }
        }
    }

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            uint col = is_up ? (D_out + n) : n;
            fused_out[b * 2 * D_out + col] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q4_0_btile_gate_up_otf_b1")]]
kernel void dense_gemv_q4_0_btile_gate_up_otf_b1(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_gate_up_otf_impl<1>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_0_btile_gate_up_otf_b2")]]
kernel void dense_gemv_q4_0_btile_gate_up_otf_b2(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_gate_up_otf_impl<2>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_0_btile_gate_up_otf_b4")]]
kernel void dense_gemv_q4_0_btile_gate_up_otf_b4(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_gate_up_otf_impl<4>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_0_btile_gate_up_otf_b8")]]
kernel void dense_gemv_q4_0_btile_gate_up_otf_b8(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_0_btile_gate_up_otf_impl<8>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}

// =============================================================================
// Q4_1 btile_gate_up_otf — structural mirror with Q4_1 dequant.
// =============================================================================

template<uint B_TILE>
inline void dense_gemv_q4_1_btile_gate_up_otf_impl(
    device const half* x, device const half* gamma,
    device const uchar* Wg_sw, device const uchar* Wu_sw,
    device half* fused_out,
    constant uint& D_in, constant uint& D_out, constant float& eps, constant uint& numVecs,
    threadgroup float (&inv_rms)[B_TILE], threadgroup float (&ss_stage)[4],
    threadgroup float (&partials)[4][32],
    uint2 tg, uint2 lid, uint sg_id)
{
    constexpr uint N_SPLITS = 4;
    constexpr uint BLK = 20;
    constexpr uint K_PER_BLK = 32;
    constexpr uint THREADS = 128;

    uint slab = tg.x;
    uint vec_block = tg.y;
    uint v_base = vec_block * B_TILE;
    if (v_base >= numVecs) return;
    uint v_count = (numVecs - v_base < B_TILE) ? (numVecs - v_base) : B_TILE;

    uint tid = lid.x;
    uint lid_sg = tid % 32;

    uint n_blocks_per_branch = D_out / 32;
    bool is_up = slab >= n_blocks_per_branch;
    uint n_block = is_up ? slab - n_blocks_per_branch : slab;

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        uint b = v_base + v;
        device const half* xb = x + b * D_in;
        float local_ss = 0.0f;
        for (uint i = tid; i < D_in; i += THREADS) {
            float val = float(xb[i]);
            local_ss += val * val;
        }
        float sg_ss = simd_sum(local_ss);
        if (lid_sg == 0) ss_stage[sg_id] = sg_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_ss = ss_stage[0] + ss_stage[1] + ss_stage[2] + ss_stage[3];
        if (tid == 0) inv_rms[v] = rsqrt(total_ss / float(D_in) + eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint n = n_block * 32 + lid_sg;
    if (n >= D_out) return;
    uint nbc = D_in / K_PER_BLK;
    uint super_bytes = nbc * 32 * BLK;
    device const uchar* W_sw = is_up ? Wu_sw : Wg_sw;
    device const uchar* W_super = W_sw + n_block * super_bytes;

    float accs[B_TILE];
    for (uint v = 0; v < B_TILE; ++v) accs[v] = 0.0f;
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
        device const uchar* blk = W_super + kb * 32 * BLK + lid_sg * BLK;
        float d = float(*(device const half*)(blk));
        float m = float(*(device const half*)(blk + 2));
        device const uchar* qs = blk + 4;
        uint base_k = kb * K_PER_BLK;
        for (uint p = 0; p < 16; ++p) {
            uchar byte = qs[p];
            float w_lo = d * float(byte & 0xF) + m;
            float w_hi = d * float((byte >> 4) & 0xF) + m;
            float gamma_lo = float(gamma[base_k + p]);
            float gamma_hi = float(gamma[base_k + p + 16]);
            for (uint v = 0; v < B_TILE; ++v) {
                if (v < v_count) {
                    uint b = v_base + v;
                    float x_lo = float(x[b * D_in + base_k + p]);
                    float x_hi = float(x[b * D_in + base_k + p + 16]);
                    float ir = inv_rms[v];
                    accs[v] += x_lo * ir * gamma_lo * w_lo
                             + x_hi * ir * gamma_hi * w_hi;
                }
            }
        }
    }

    for (uint v = 0; v < B_TILE; ++v) {
        if (v >= v_count) break;
        partials[sg_id][lid_sg] = accs[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0) {
            float total = partials[0][lid_sg] + partials[1][lid_sg]
                        + partials[2][lid_sg] + partials[3][lid_sg];
            uint b = v_base + v;
            uint col = is_up ? (D_out + n) : n;
            fused_out[b * 2 * D_out + col] = half(total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[host_name("dense_gemv_q4_1_btile_gate_up_otf_b1")]]
kernel void dense_gemv_q4_1_btile_gate_up_otf_b1(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[1]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_gate_up_otf_impl<1>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_1_btile_gate_up_otf_b2")]]
kernel void dense_gemv_q4_1_btile_gate_up_otf_b2(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[2]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_gate_up_otf_impl<2>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_1_btile_gate_up_otf_b4")]]
kernel void dense_gemv_q4_1_btile_gate_up_otf_b4(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[4]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_gate_up_otf_impl<4>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
}
[[host_name("dense_gemv_q4_1_btile_gate_up_otf_b8")]]
kernel void dense_gemv_q4_1_btile_gate_up_otf_b8(
    device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]],
    device const uchar* Wg_sw [[buffer(2)]], device const uchar* Wu_sw [[buffer(3)]],
    device half* fused_out [[buffer(4)]],
    constant uint& D_in [[buffer(5)]], constant uint& D_out [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant uint& numVecs [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float inv_rms[8]; threadgroup float ss_stage[4]; threadgroup float partials[4][32];
    dense_gemv_q4_1_btile_gate_up_otf_impl<8>(x, gamma, Wg_sw, Wu_sw, fused_out,
        D_in, D_out, eps, numVecs, inv_rms, ss_stage, partials, tg, lid, sg_id);
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
    for (uint kb = sg_id; kb < nbc; kb += N_SPLITS) {
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
    if (expert >= 128u) return;       // sentinel: route_compact pads tail with E=128
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
    if (expert >= 128u) return;       // sentinel from route_compact tail padding
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

// MoE Q5_K GEMV v6: swizzled [expert, n_super=D_out/32, nbc=D_in/256, 32 cols, 176 bytes].
// 32 threads of an SG read 5632 contiguous bytes per kb iter — cache-line coalesced.
// Slot-flat: hidden indexed by slot_token[slot] (first MoE GEMV — fan-out from
// hidden_norm[B, HIDDEN]). Mirror moe_gemv_q4k_v6 with Q5_K dequant.
kernel void moe_gemv_q5_K_v6(
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
    if (expert >= 128u) return;       // sentinel: route_compact pads tail with E=128
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 176;
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
            float d_all = float(*(device const half*)(blk + 0));
            float dmin  = float(*(device const half*)(blk + 2));
            device const uchar* scales = blk + 4;
            device const uchar* qh     = blk + 16;
            device const uchar* qs     = blk + 48;
            for (uint il_orig = 0; il_orig < 16; ++il_orig) {
                uint q_off  = 32u * (il_orig / 4u) + 16u * (il_orig & 1u);
                uint qh_off = 16u * (il_orig & 1u);
                uchar ul    = uchar(1u << (il_orig / 2u));
                uint is     = (il_orig / 4u) * 2u;
                uint ph     = il_orig & 3u;
                uchar2 sc   = get_scale_min_k4_just2(is, ph / 2u, scales);
                float d_eff = (ph < 2u) ? d_all : (d_all / 16.0f);
                float dl    = d_eff * float(sc[0]);
                float ml    = dmin  * float(sc[1]);
                uchar mask  = (ph < 2u) ? 0x0F : 0xF0;
                float qhv   = (ph < 2u) ? 16.0f : 256.0f;
                uint base_k = kb * 256u + il_orig * 16u;
                for (uint i = 0; i < 16; ++i) {
                    float v = float(qs[q_off + i] & mask)
                            + (((qh[qh_off + i] & ul) != 0) ? qhv : 0.0f);
                    acc += float(hid[base_k + i]) * (dl * v - ml);
                }
            }
        }
        output[slot * D_out + n] = half(acc);
    }
}

// MoE Q6_K GEMV v6: swizzled [expert, n_super=D_out/32, nbc=D_in/256, 32 cols, 210 bytes].
// 32 threads of an SG read 6720 contiguous bytes per kb iter — cache-line coalesced.
// Down-projection convention: hidden indexed by slot * D_in (per-slot layout).
kernel void moe_gemv_q6_K_v6(
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
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 210;
    uint nbc = D_in / 256;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    for (uint slot = gb; slot < ge; ++slot) {
        // Q6_K is the MoE DOWN projection per the V1 grid (Q6_K replacement
        // for the high-precision down weights). Use slot * D_in layout —
        // input is per-slot (already gate/up expanded), not per-batch-token.
        device const half* hid = hidden + slot * D_in;
        float acc = 0.0f;
        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            device const ushort* ql_u16    = (device const ushort*)(blk + 0);
            device const ushort* qh_u16    = (device const ushort*)(blk + 128);
            device const char*   scales_i8 = (device const char*)(blk + 192);
            float d_all = float(*(device const half*)(blk + 208));
            for (uint il = 0; il < 16; ++il) {
                uint ql_base = 32u * (il / 8u) + 16u * ((il / 2u) & 1u) + 8u * (il & 1u);
                uint qh_base = 16u * (il / 8u) + 8u * (il & 1u);
                float sc = float(scales_i8[(il % 2u) + 2u * (il / 2u)]);
                uint ph = (il / 2u) & 3u;

                uint kmask1 = (ph > 1u) ? ((ph > 2u) ? 0xC0C0C0C0u : 0x30303030u)
                                        : ((ph > 0u) ? 0x0C0C0C0Cu : 0x03030303u);
                uint kmask2 = (ph > 1u) ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
                float ml  = d_all * sc * 32.0f;
                float dl0 = d_all * sc;
                float dl1 = dl0 / 256.0f;
                float dl2 = dl1 / 256.0f;
                float dl3 = dl2 / 256.0f;
                uint shr_h = (ph > 2u) ? 2u : 0u;
                uint shl_h = (ph > 1u) ? 0u : ((ph > 0u) ? 2u : 4u);
                uint shr_l = (ph > 1u) ? 4u : 0u;

                uint base_k = kb * 256u + il * 16u;
                for (uint i = 0; i < 4; ++i) {
                    uint low  = (uint(ql_u16[ql_base + 2u*i]) | (uint(ql_u16[ql_base + 2u*i + 1u]) << 16)) & kmask2;
                    uint high = (uint(qh_u16[qh_base + 2u*i]) | (uint(qh_u16[qh_base + 2u*i + 1u]) << 16)) & kmask1;
                    uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
                    float w0 = dl0 * float(q & 0x000000FFu) - ml;
                    float w1 = dl1 * float(q & 0x0000FF00u) - ml;
                    float w2 = dl2 * float(q & 0x00FF0000u) - ml;
                    float w3 = dl3 * float(q & 0xFF000000u) - ml;
                    uint k0 = base_k + i * 4u;
                    acc += float(hid[k0 + 0]) * w0
                         + float(hid[k0 + 1]) * w1
                         + float(hid[k0 + 2]) * w2
                         + float(hid[k0 + 3]) * w3;
                }
            }
        }
        output[slot * D_out + n] = half(acc);
    }
}

// MoE Q8_0 GEMV v6: swizzled [expert, n_super=D_out/32, nbc=D_in/32, 32 cols, 34 bytes].
// 32 threads of an SG read 1088 contiguous bytes per kb iter. Per-slot X
// (down convention) — used for ffn_down_exps when llama-quantize keeps it
// at Q8_0 (Q5_K_M default). Modeled on moe_gemv_q5_1_v6 with Q8_0 dequant.
kernel void moe_gemv_q8_0_v6(
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
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 34;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    for (uint slot = gb; slot < ge; ++slot) {
        // Down-projection convention: hidden indexed by slot * D_in (per-slot).
        device const half* hid = hidden + slot * D_in;
        float acc = 0.0f;
        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d = float(*(device const half*)(blk));
            device const char* qs = (device const char*)(blk + 2);
            uint base_k = kb * 32;
            for (uint p = 0; p < 32; ++p) {
                acc += float(hid[base_k + p]) * d * float(qs[p]);
            }
        }
        output[slot * D_out + n] = half(acc);
    }
}

// MoE Q4_K GEMV v8 — V4-style register slot-amortization with chunked
// MAX_SLOTS=16 (matches prefill numVecs=256 × top-8 / 128 experts ≈ 16
// slots/expert avg). V6 walks slots one-by-one and re-reads the weight
// slab from L2 each time; V8 reads the weight slab ONCE per slot-chunk
// and FMAs into MAX_SLOTS accumulators in registers. At full prefill
// numVecs the L2 amortization V6 relies on is already breaking down
// (weight matrix is 285 MB, much bigger than L2), so register-level
// amortization captures the bandwidth headroom.
kernel void moe_gemv_q4k_v8(
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
    constexpr uint MAX_SLOTS = 4;     // matches V4's register footprint; chunked outer loop handles n_slots > 4
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

    // Outer chunk-of-slots loop. Each chunk processes up to MAX_SLOTS
    // slots with weight read once via the kb loop.
    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < chunk; ++s) {
            hid_slots[s] = hidden + slot_token[slot_base + s] * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d    = float(*(device const half*)(blk + 0));
            float dmin = float(*(device const half*)(blk + 2));
            device const uchar* scales = blk + 4;
            device const uchar* qs     = blk + 16;
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
                    for (uint s = 0; s < chunk; ++s) {
                        accs[s] += float(hid_slots[s][base_lo + p]) * w_lo
                                 + float(hid_slots[s][base_hi + p]) * w_hi;
                    }
                }
            }
        }
        for (uint s = 0; s < chunk; ++s) {
            output[(slot_base + s) * D_out + n] = half(accs[s]);
        }
        slot_base += chunk;
    }
}

// MoE Q5_1 GEMV v8 — same structure as q4k_v8 but for Q5_1 down. Down-proj
// uses slot * D_in indexing (post-gate/up expanded layout) per the v6
// note above.
kernel void moe_gemv_q5_1_v8(
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
    constexpr uint MAX_SLOTS = 4;     // matches V4's register footprint; chunked outer loop handles n_slots > 4
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

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < chunk; ++s) {
            // Down-proj reads pre-expanded [TOTAL_SLOTS, D_in] layout: index
            // by absolute slot, NOT slot_token (matches v6 commentary).
            hid_slots[s] = hidden + (slot_base + s) * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

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
                for (uint s = 0; s < chunk; ++s) {
                    accs[s] += float(hid_slots[s][base_k + p])      * w_lo
                             + float(hid_slots[s][base_k + p + 16]) * w_hi;
                }
            }
        }
        for (uint s = 0; s < chunk; ++s) {
            output[(slot_base + s) * D_out + n] = half(accs[s]);
        }
        slot_base += chunk;
    }
}

// MoE Q4_K GEMV v9 — V8's chunked register slot-amortization, but inner
// FMA loop uses constexpr MAX_SLOTS=8 bound + predicated `if (s < chunk)`
// so Apple's compiler can unroll the s-loop and keep accs[]/hid_slots[]
// in registers (the V7 QKV pattern). V8's `for s = 0..<chunk` was
// runtime-bounded → array spilled to local mem and ran 2-3× slower than
// V6. V9's biggest expected win is dequant amortization: V6 does N_slots
// dequants per (kb, p); V9 does 1 per chunk, so 8× fewer Q4_K unpack ops
// on each TG.
kernel void moe_gemv_q4k_v9(
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
    constexpr uint MAX_SLOTS = 16;
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

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            // Safe load: out-of-range lanes alias slot_base[0] but their
            // FMAs are predicated off below.
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + slot_token[idx] * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d    = float(*(device const half*)(blk + 0));
            float dmin = float(*(device const half*)(blk + 2));
            device const uchar* scales = blk + 4;
            device const uchar* qs     = blk + 16;
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
                    // Constexpr-bounded predicated FMA (V7 pattern).
                    for (uint s = 0; s < MAX_SLOTS; ++s) {
                        if (s < chunk) {
                            accs[s] += float(hid_slots[s][base_lo + p]) * w_lo
                                     + float(hid_slots[s][base_hi + p]) * w_hi;
                        }
                    }
                }
            }
        }
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

// MoE Q4_K v10 — fused matmul + GELU(gate)·up. Each TG handles ONE
// 32-column slab of the gate-side outputs (Dout = N_half = MOE_INT).
// Inner K-loop reads BOTH the gate weight slab and the up weight slab in
// lockstep, accumulating into two parallel register arrays. After the
// K-loop, applies GELU(gate)·up in-register and writes [slots, N_half]
// directly to the post-activation buffer (pre_gate_proj). This removes
// the separate moe_gelu_mul_fused dispatch + the [slots, 2*N_half]
// fused intermediate's DRAM round-trip.
//
// Caveat: profiler shows moe_gelu_mul_fused at 0.022 ms/call, so the
// structural win is bounded at ~0.7 ms/pass total. The bigger expected
// effect would be cross-slab L2 sharing of the hidden vector (same
// hidden read drives gate+up in the same TG instead of two separate
// TGs). Magnitude TBD empirically vs v9.
//
// Register footprint: 2 × MAX_SLOTS=8 floats + MAX_SLOTS device ptrs +
// loop temps ≈ 24 32-bit slots/thread; under the ~32-48 cliff that V8
// hit at MAX_SLOTS=16.
//
// Buffer layout: same swizzled per-expert layout as v6/v9. We reach
// the up slab via offset (n_block + N_half/32) * super_bytes within
// the same expert's region. D_out_fused = 2 * N_half is passed so the
// kernel can compute the slab stride.
kernel void moe_gemv_q4k_v10_fused_gelu(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const uchar* W_sw            [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],     // [slots, N_half]
    constant uint& D_in                 [[buffer(6)]],
    constant uint& N_half               [[buffer(7)]],     // = MOE_INT
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint MAX_SLOTS = 8;
    uint n_block = tg.x;                               // [0, N_half/32)
    uint ai = tg.y; uint expert = active_experts[ai];
    uint t = lid.x;
    uint n = n_block * 32 + t;                         // gate output column
    if (n >= N_half) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 144;
    uint nbc = D_in / 256;
    uint super_bytes = nbc * 32 * BLK;
    // Fused weight tensor has 2 * N_half output columns; expert region
    // covers (2 * N_half / 32) super-blocks.
    uint expert_bytes = (2 * N_half / 32) * super_bytes;
    device const uchar* W_gate = W_sw + expert * expert_bytes + n_block * super_bytes;
    device const uchar* W_up   = W_sw + expert * expert_bytes
                                       + (n_block + N_half / 32) * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + slot_token[idx] * D_in;
        }
        float acc_g[MAX_SLOTS];
        float acc_u[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) { acc_g[s] = 0.0f; acc_u[s] = 0.0f; }

        for (uint kb = 0; kb < nbc; ++kb) {
            // ---- gate slab ----
            {
                device const uchar* blk = W_gate + kb * 32 * BLK + t * BLK;
                float d    = float(*(device const half*)(blk + 0));
                float dmin = float(*(device const half*)(blk + 2));
                device const uchar* scales = blk + 4;
                device const uchar* qs     = blk + 16;
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
                        for (uint s = 0; s < MAX_SLOTS; ++s) {
                            if (s < chunk) {
                                acc_g[s] += float(hid_slots[s][base_lo + p]) * w_lo
                                          + float(hid_slots[s][base_hi + p]) * w_hi;
                            }
                        }
                    }
                }
            }
            // ---- up slab (same hidden vectors, different weights) ----
            {
                device const uchar* blk = W_up + kb * 32 * BLK + t * BLK;
                float d    = float(*(device const half*)(blk + 0));
                float dmin = float(*(device const half*)(blk + 2));
                device const uchar* scales = blk + 4;
                device const uchar* qs     = blk + 16;
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
                        for (uint s = 0; s < MAX_SLOTS; ++s) {
                            if (s < chunk) {
                                acc_u[s] += float(hid_slots[s][base_lo + p]) * w_lo
                                          + float(hid_slots[s][base_hi + p]) * w_hi;
                            }
                        }
                    }
                }
            }
        }

        // GELU(gate) · up — clamped tanh, matches moe_gelu_mul_fused.
        const float c = 0.7978845608f;
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                float g = acc_g[s];
                float inner = c * (g + 0.044715f * g * g * g);
                inner = clamp(inner, -20.0f, 20.0f);
                float gelu_g = 0.5f * g * (1.0f + tanh(inner));
                output[(slot_base + s) * N_half + n] = half(gelu_g * acc_u[s]);
            }
        }
        slot_base += chunk;
    }
}

// MoE Q5_1 GEMV v9 — same constexpr-unroll pattern as q4k_v9 but for the
// down projection (Q5_1, BLK=24, slot index uses absolute slot per the
// down-proj layout note in v6).
kernel void moe_gemv_q5_1_v9(
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
    constexpr uint MAX_SLOTS = 16;
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

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + idx * D_in;     // absolute slot indexing for down-proj
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

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
                for (uint s = 0; s < MAX_SLOTS; ++s) {
                    if (s < chunk) {
                        accs[s] += float(hid_slots[s][base_k + p])      * w_lo
                                 + float(hid_slots[s][base_k + p + 16]) * w_hi;
                    }
                }
            }
        }
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

// MoE Q4_K v11 — V9 + scale-hoist (Approach 2) + per-pair pre-dequant
// register scratch (Approach 1) + templated MAX_SLOTS (Approach 3).
//
// Empirically (N=1: ties V6 at 15.9 tok/s; N=4: +4.3% vs V6 at 85 tok/s;
// N=8: within ±6% of V6) the pre-dequant scratch DOES promote to
// registers on Apple's MSL compiler when accessed via constexpr-bounded
// loops. An earlier diagnosis attempting to fix an N=1 regression by
// removing the scratch was wrong — it was reading my own stale baseline
// memory, not an actual regression.
//
// Three deltas vs V9:
//
// (1) Scales hoisted out of the pair loop. V6/V9 call unpack_q4k_scales 8x
//     per kb (twice per pair × 4 pairs); each call does ~5 bit ops on the
//     packed scales[12] array. V11 hoists all 8 (sc, mn) pairs into
//     register arrays dl[8], ml[8] once per kb. Inner pair loop just
//     reads dl[sb_lo] etc. Saves ~16 multiplies/kb plus 4 redundant
//     unpack calls.
//
// (2) Per-pair pre-dequant to register scratch W_lo[32], W_hi[32]. The
//     slot FMA loop reads pure half[32] arrays — multiply-accumulate
//     that the compiler can SIMD-coalesce more easily than V9's
//     interleaved dequant-and-FMA pattern.
//
// (3) MAX_SLOTS as a template param. At activeB=1 (every expert holds 1
//     slot) MAX_SLOTS=1 gives the cleanest possible inner loop; at
//     activeB=4-8 with mild routing collisions, MAX_SLOTS=4/8 absorbs
//     the typical chunk without chunking overhead.
//
// Buffer layout, output write convention, and grid shape match V6/V9.
template<uint MAX_SLOTS>
inline void moe_gemv_q4k_v11_impl(
    device const half* hidden,
    device const uint* slot_token,
    device const uchar* W_sw,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;          // sentinel: route_compact pads tail with E=128
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 144;
    uint nbc = D_in / 256;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        // Per-slot input pointers. Out-of-range lanes alias slot_base[0]
        // but their FMAs are predicated off, so the read of an aliased
        // hidden vector is harmless (the result is multiplied by zero in
        // the sense that it's never written through to accs).
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + slot_token[idx] * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d    = float(*(device const half*)(blk + 0));
            float dmin = float(*(device const half*)(blk + 2));
            device const uchar* scales = blk + 4;
            device const uchar* qs     = blk + 16;

            // (1) Hoist scales out of the pair loop. dl[sb] = d * sc[sb],
            //     ml[sb] = dmin * mn[sb] for all 8 sub-blocks at once.
            //     The unroll keeps `sb` compile-time constant so dl/ml
            //     stay in registers without dynamic indexing.
            float dl_arr[8], ml_arr[8];
            for (uint sb = 0; sb < 8; ++sb) {
                uchar sc, mn;
                unpack_q4k_scales(scales, sc, mn, sb);
                dl_arr[sb] = d * float(sc);
                ml_arr[sb] = dmin * float(mn);
            }

            for (uint pair = 0; pair < 4; ++pair) {
                uint sb_lo = pair * 2;
                uint sb_hi = pair * 2 + 1;
                float dl_lo = dl_arr[sb_lo], ml_lo = ml_arr[sb_lo];
                float dl_hi = dl_arr[sb_hi], ml_hi = ml_arr[sb_hi];

                // (Approach 1) Pre-dequant this pair's 64 weights to
                // register scratch. Empirically this WINS at multi-stream
                // (N=4: +4% vs V6, N=8 within noise) — the Apple compiler
                // does promote half W_lo[32] / W_hi[32] to registers when
                // accessed via a constexpr-bounded loop, contrary to my
                // earlier worry. The slot FMA loop becomes pure half-array
                // multiply-accumulate that the SIMD path can coalesce.
                half W_lo[32], W_hi[32];
                for (uint p = 0; p < 32; ++p) {
                    uchar byte = qs[pair * 32 + p];
                    W_lo[p] = half(dl_lo * float(byte & 0xF)        - ml_lo);
                    W_hi[p] = half(dl_hi * float((byte >> 4) & 0xF) - ml_hi);
                }
                uint base_lo = kb * 256 + sb_lo * 32;
                uint base_hi = kb * 256 + sb_hi * 32;

                // Predicated multi-slot FMA. V7/V9 pattern: outer s-loop
                // unrolled across constexpr MAX_SLOTS, inner predicate
                // masks inactive lanes. accs[s] stays in registers
                // because `s` is compile-time-constant after unroll.
                for (uint s = 0; s < MAX_SLOTS; ++s) {
                    if (s < chunk) {
                        device const half* hid = hid_slots[s];
                        float a = accs[s];
                        for (uint p = 0; p < 32; ++p) {
                            a += float(hid[base_lo + p]) * float(W_lo[p])
                               + float(hid[base_hi + p]) * float(W_hi[p]);
                        }
                        accs[s] = a;
                    }
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_q4k_v11_b1")]]
kernel void moe_gemv_q4k_v11_b1_kern(
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
    moe_gemv_q4k_v11_impl<1>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q4k_v11_b2")]]
kernel void moe_gemv_q4k_v11_b2_kern(
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
    moe_gemv_q4k_v11_impl<2>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q4k_v11_b4")]]
kernel void moe_gemv_q4k_v11_b4_kern(
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
    moe_gemv_q4k_v11_impl<4>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q4k_v11_b8")]]
kernel void moe_gemv_q4k_v11_b8_kern(
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
    moe_gemv_q4k_v11_impl<8>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

// MoE Q5_1 v11 — port of Q4K V11's structure to the simpler Q5_1 block
// layout (32 elements × 24 bytes, single (d, m) pair per block, no
// paired sub-blocks). The down projection uses this kernel.
//
// Two of Q4K V11's three sub-approaches port directly:
//
// (1) Per-block pre-dequant register scratch. Q5_1 has 32 elements per
//     block split into low/high halves (p ∈ [0,16) → element p AND element
//     p+16, sharing a qs byte). We pre-dequant W_lo[16] (elements 0..15)
//     and W_hi[16] (elements 16..31) once per kb, then FMA across slots.
//     Scratch footprint: 32 halves = 16 32-bit registers — even smaller
//     than Q4K's per-pair 64-half scratch, well below the spill cliff.
//
// (2) Templated MAX_SLOTS={1,2,4,8}. Same predicated multi-slot FMA
//     pattern as Q4K V11. Dispatcher picks specialization by activeB.
//
// Q4K's Approach 2 (scale hoist) does NOT apply here — Q5_1 has only ONE
// (d, m) pair per kb, already loaded once at the top of the kb loop.
//
// CRITICAL: Q5_1 is the down projection. Its input `hidden` is
// per-slot post-gate-up activations, layout [TOTAL_SLOTS, D_in]. Use
// `(slot_base + s) * D_in`, NOT `slot_token[…] * D_in` (that's the
// per-batch index used by the up projection where we fan out from
// hidden_norm[B, HIDDEN]). slot_token is unused in this kernel —
// kept in the buffer-binding signature for ABI parity with Q4K V11
// so the dispatcher can share argument-marshaling code.
template<uint MAX_SLOTS>
inline void moe_gemv_q5_1_v11_impl(
    device const half* hidden,
    device const uint* slot_token,    // unused — see comment above
    device const uchar* W_sw,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    (void)slot_token;
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 24;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        // Per-slot input pointers. Q5_1 is per-slot indexed (down proj),
        // not slot_token-indexed.
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + idx * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d = float(*(device const half*)(blk + 0));
            float m = float(*(device const half*)(blk + 2));
            device const uchar* qh = blk + 4;
            device const uchar* qs = blk + 8;

            // (Approach 1) Pre-dequant the 32 weights of this block to
            // half W_lo[16] (elements 0..15) and W_hi[16] (elements 16..31).
            // The Q5_1 5-bit format packs 4 low bits in qs[p] and the 5th
            // bit in qh, with element p+16 sharing the same qs byte.
            half W_lo[16], W_hi[16];
            for (uint p = 0; p < 16; ++p) {
                uchar qsp = qs[p];
                uint h_lo = (qh[p / 8]       >> (p       % 8)) & 1u;
                uint h_hi = (qh[(p + 16) / 8] >> ((p + 16) % 8)) & 1u;
                uint q_lo = (uint(qsp) & 0xFu)        | (h_lo << 4);
                uint q_hi = ((uint(qsp) >> 4) & 0xFu) | (h_hi << 4);
                W_lo[p] = half(d * float(q_lo) + m);
                W_hi[p] = half(d * float(q_hi) + m);
            }
            uint base_k = kb * 32;

            // Predicated multi-slot FMA. Same V7/V9 pattern as Q4K V11.
            for (uint s = 0; s < MAX_SLOTS; ++s) {
                if (s < chunk) {
                    device const half* hid = hid_slots[s];
                    float a = accs[s];
                    for (uint p = 0; p < 16; ++p) {
                        a += float(hid[base_k + p])      * float(W_lo[p])
                           + float(hid[base_k + p + 16]) * float(W_hi[p]);
                    }
                    accs[s] = a;
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_q5_1_v11_b1")]]
kernel void moe_gemv_q5_1_v11_b1_kern(
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
    moe_gemv_q5_1_v11_impl<1>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q5_1_v11_b2")]]
kernel void moe_gemv_q5_1_v11_b2_kern(
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
    moe_gemv_q5_1_v11_impl<2>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q5_1_v11_b4")]]
kernel void moe_gemv_q5_1_v11_b4_kern(
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
    moe_gemv_q5_1_v11_impl<4>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q5_1_v11_b8")]]
kernel void moe_gemv_q5_1_v11_b8_kern(
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
    moe_gemv_q5_1_v11_impl<8>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

// =============================================================================
// MoE Q4_0 v11 — structural mirror of moe_gemv_q4k_v11_impl (slot_token
// broadcast convention). Q4_0 dequant: w = d * (nib - 8). Pre-dequant the
// 32 weights of each block to register array W[32], then run the predicated
// multi-slot FMA loop.
// =============================================================================

template<uint MAX_SLOTS>
inline void moe_gemv_q4_0_v11_impl(
    device const half* hidden,
    device const uint* slot_token,
    device const uchar* W_sw,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 18;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + slot_token[idx] * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d = float(*(device const half*)(blk));
            device const uchar* qs = blk + 2;

            half W_lo[16], W_hi[16];
            for (uint p = 0; p < 16; ++p) {
                uchar byte = qs[p];
                W_lo[p] = half(d * float(int(byte & 0xF) - 8));
                W_hi[p] = half(d * float(int((byte >> 4) & 0xF) - 8));
            }
            uint base_k = kb * 32;

            for (uint s = 0; s < MAX_SLOTS; ++s) {
                if (s < chunk) {
                    device const half* hid = hid_slots[s];
                    float a = accs[s];
                    for (uint p = 0; p < 16; ++p) {
                        a += float(hid[base_k + p])      * float(W_lo[p])
                           + float(hid[base_k + p + 16]) * float(W_hi[p]);
                    }
                    accs[s] = a;
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_q4_0_v11_b1")]]
kernel void moe_gemv_q4_0_v11_b1_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_0_v11_impl<1>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_0_v11_b2")]]
kernel void moe_gemv_q4_0_v11_b2_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_0_v11_impl<2>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_0_v11_b4")]]
kernel void moe_gemv_q4_0_v11_b4_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_0_v11_impl<4>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_0_v11_b8")]]
kernel void moe_gemv_q4_0_v11_b8_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_0_v11_impl<8>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }

// =============================================================================
// MoE Q4_0 v11 (per-slot / down convention) — same kernel body as
// moe_gemv_q4_0_v11_impl but reads `hidden + idx * D_in` instead of
// `hidden + slot_token[idx] * D_in`. Used for ffn_down_exps in --pure Q4_0
// configs, or anywhere ffn_down at a layer is Q4_0. The slot_token buffer
// is kept in the binding signature for ABI parity but unused.
// =============================================================================

template<uint MAX_SLOTS>
inline void moe_gemv_q4_0_v11_down_impl(
    device const half* hidden,
    device const uint* slot_token,    // unused — per-slot convention
    device const uchar* W_sw,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    (void)slot_token;
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 18;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + idx * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d = float(*(device const half*)(blk));
            device const uchar* qs = blk + 2;

            half W_lo[16], W_hi[16];
            for (uint p = 0; p < 16; ++p) {
                uchar byte = qs[p];
                W_lo[p] = half(d * float(int(byte & 0xF) - 8));
                W_hi[p] = half(d * float(int((byte >> 4) & 0xF) - 8));
            }
            uint base_k = kb * 32;

            for (uint s = 0; s < MAX_SLOTS; ++s) {
                if (s < chunk) {
                    device const half* hid = hid_slots[s];
                    float a = accs[s];
                    for (uint p = 0; p < 16; ++p) {
                        a += float(hid[base_k + p])      * float(W_lo[p])
                           + float(hid[base_k + p + 16]) * float(W_hi[p]);
                    }
                    accs[s] = a;
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_q4_0_v11_down_b1")]]
kernel void moe_gemv_q4_0_v11_down_b1_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_0_v11_down_impl<1>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_0_v11_down_b2")]]
kernel void moe_gemv_q4_0_v11_down_b2_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_0_v11_down_impl<2>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_0_v11_down_b4")]]
kernel void moe_gemv_q4_0_v11_down_b4_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_0_v11_down_impl<4>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_0_v11_down_b8")]]
kernel void moe_gemv_q4_0_v11_down_b8_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_0_v11_down_impl<8>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }

// =============================================================================
// MoE Q4_1 v11 (slot_token broadcast / up convention). Q4_1 dequant:
// w = d * nib + m. Block: 20 B/32 elts (half d, half m, uchar qs[16]).
// =============================================================================

template<uint MAX_SLOTS>
inline void moe_gemv_q4_1_v11_impl(
    device const half* hidden,
    device const uint* slot_token,
    device const uchar* W_sw,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 20;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + slot_token[idx] * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d = float(*(device const half*)(blk));
            float m = float(*(device const half*)(blk + 2));
            device const uchar* qs = blk + 4;

            half W_lo[16], W_hi[16];
            for (uint p = 0; p < 16; ++p) {
                uchar byte = qs[p];
                W_lo[p] = half(d * float(byte & 0xF) + m);
                W_hi[p] = half(d * float((byte >> 4) & 0xF) + m);
            }
            uint base_k = kb * 32;

            for (uint s = 0; s < MAX_SLOTS; ++s) {
                if (s < chunk) {
                    device const half* hid = hid_slots[s];
                    float a = accs[s];
                    for (uint p = 0; p < 16; ++p) {
                        a += float(hid[base_k + p])      * float(W_lo[p])
                           + float(hid[base_k + p + 16]) * float(W_hi[p]);
                    }
                    accs[s] = a;
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_q4_1_v11_b1")]]
kernel void moe_gemv_q4_1_v11_b1_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_1_v11_impl<1>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_1_v11_b2")]]
kernel void moe_gemv_q4_1_v11_b2_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_1_v11_impl<2>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_1_v11_b4")]]
kernel void moe_gemv_q4_1_v11_b4_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_1_v11_impl<4>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_1_v11_b8")]]
kernel void moe_gemv_q4_1_v11_b8_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_1_v11_impl<8>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }

// =============================================================================
// MoE Q4_1 v11 (per-slot / down convention).
// =============================================================================

template<uint MAX_SLOTS>
inline void moe_gemv_q4_1_v11_down_impl(
    device const half* hidden,
    device const uint* slot_token,    // unused — per-slot convention
    device const uchar* W_sw,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    (void)slot_token;
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 20;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + idx * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d = float(*(device const half*)(blk));
            float m = float(*(device const half*)(blk + 2));
            device const uchar* qs = blk + 4;

            half W_lo[16], W_hi[16];
            for (uint p = 0; p < 16; ++p) {
                uchar byte = qs[p];
                W_lo[p] = half(d * float(byte & 0xF) + m);
                W_hi[p] = half(d * float((byte >> 4) & 0xF) + m);
            }
            uint base_k = kb * 32;

            for (uint s = 0; s < MAX_SLOTS; ++s) {
                if (s < chunk) {
                    device const half* hid = hid_slots[s];
                    float a = accs[s];
                    for (uint p = 0; p < 16; ++p) {
                        a += float(hid[base_k + p])      * float(W_lo[p])
                           + float(hid[base_k + p + 16]) * float(W_hi[p]);
                    }
                    accs[s] = a;
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_q4_1_v11_down_b1")]]
kernel void moe_gemv_q4_1_v11_down_b1_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_1_v11_down_impl<1>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_1_v11_down_b2")]]
kernel void moe_gemv_q4_1_v11_down_b2_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_1_v11_down_impl<2>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_1_v11_down_b4")]]
kernel void moe_gemv_q4_1_v11_down_b4_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_1_v11_down_impl<4>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q4_1_v11_down_b8")]]
kernel void moe_gemv_q4_1_v11_down_b8_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q4_1_v11_down_impl<8>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }

// =============================================================================
// MoE Q5_K v11 — structural mirror of moe_gemv_q4k_v11_impl (slot_token
// broadcast convention, used for ffn_gate_up_exps in Q5_K_M). Same
// MAX_SLOTS register-tile pattern; only the per-element dequant differs.
// Q5_K block: 256 elts × 176 B (vs Q4_K's 256 × 144). 16 sub-tiles per
// super-block, each producing 16 weights via the (q_off, qh_off, ul,
// mask, qhv, dl, ml) chain.
//
// We pre-dequant 64 weights at a time (4 contiguous sub-tiles = K range
// [group*64, group*64+64)) into register array W[64], then run the
// predicated multi-slot FMA loop — same skeleton as Q4_K V11's pair loop
// but each "group" is 4 sub-tiles of 16 weights instead of 1 pair of 32.
// =============================================================================

template<uint MAX_SLOTS>
inline void moe_gemv_q5_K_v11_impl(
    device const half* hidden,
    device const uint* slot_token,
    device const uchar* W_sw,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 176;
    uint nbc = D_in / 256;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + slot_token[idx] * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d_all = float(*(device const half*)(blk + 0));
            float dmin  = float(*(device const half*)(blk + 2));
            device const uchar* scales = blk + 4;
            device const uchar* qh     = blk + 16;
            device const uchar* qs     = blk + 48;

            // 4 groups of 4 contiguous sub-tiles, 64 weights per group.
            for (uint group = 0; group < 4; ++group) {
                half W[64];
                for (uint il_in_grp = 0; il_in_grp < 4; ++il_in_grp) {
                    uint il_orig = group * 4 + il_in_grp;
                    uint q_off  = 32u * (il_orig / 4u) + 16u * (il_orig & 1u);
                    uint qh_off = 16u * (il_orig & 1u);
                    uchar ul    = uchar(1u << (il_orig / 2u));
                    uint is     = (il_orig / 4u) * 2u;
                    uint ph     = il_orig & 3u;
                    uchar2 sc   = get_scale_min_k4_just2(is, ph / 2u, scales);
                    float d_eff = (ph < 2u) ? d_all : (d_all / 16.0f);
                    float dl    = d_eff * float(sc[0]);
                    float ml    = dmin  * float(sc[1]);
                    uchar mask  = (ph < 2u) ? 0x0F : 0xF0;
                    float qhv   = (ph < 2u) ? 16.0f : 256.0f;
                    for (uint i = 0; i < 16; ++i) {
                        float v = float(qs[q_off + i] & mask)
                                + (((qh[qh_off + i] & ul) != 0) ? qhv : 0.0f);
                        W[il_in_grp * 16 + i] = half(dl * v - ml);
                    }
                }
                uint base_k = kb * 256 + group * 64;

                // Predicated multi-slot FMA. Mirrors Q4_K V11.
                for (uint s = 0; s < MAX_SLOTS; ++s) {
                    if (s < chunk) {
                        device const half* hid = hid_slots[s];
                        float a = accs[s];
                        for (uint i = 0; i < 64; ++i) {
                            a += float(hid[base_k + i]) * float(W[i]);
                        }
                        accs[s] = a;
                    }
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_q5_K_v11_b1")]]
kernel void moe_gemv_q5_K_v11_b1_kern(
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
    moe_gemv_q5_K_v11_impl<1>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q5_K_v11_b2")]]
kernel void moe_gemv_q5_K_v11_b2_kern(
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
    moe_gemv_q5_K_v11_impl<2>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q5_K_v11_b4")]]
kernel void moe_gemv_q5_K_v11_b4_kern(
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
    moe_gemv_q5_K_v11_impl<4>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q5_K_v11_b8")]]
kernel void moe_gemv_q5_K_v11_b8_kern(
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
    moe_gemv_q5_K_v11_impl<8>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

// =============================================================================
// MoE Q8_0 v11 — structural mirror of moe_gemv_q5_1_v11_impl (per-slot
// convention, used for ffn_down_exps when format is Q8_0 in Q5_K_M).
// Same MAX_SLOTS register-tile pattern; per-element dequant is Q8_0
// (single int8 multiply by block scale d).
// =============================================================================

template<uint MAX_SLOTS>
inline void moe_gemv_q8_0_v11_impl(
    device const half* hidden,
    device const uint* slot_token,    // unused — per-slot convention
    device const uchar* W_sw,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    (void)slot_token;
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 34;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + idx * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d = float(*(device const half*)(blk + 0));
            device const char* qs = (device const char*)(blk + 2);

            // Pre-dequant 32 weights to register array W[32].
            half W[32];
            for (uint p = 0; p < 32; ++p) {
                W[p] = half(d * float(qs[p]));
            }
            uint base_k = kb * 32;

            // Predicated multi-slot FMA. Mirrors Q5_1 V11.
            for (uint s = 0; s < MAX_SLOTS; ++s) {
                if (s < chunk) {
                    device const half* hid = hid_slots[s];
                    float a = accs[s];
                    for (uint p = 0; p < 32; ++p) {
                        a += float(hid[base_k + p]) * float(W[p]);
                    }
                    accs[s] = a;
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_q8_0_v11_b1")]]
kernel void moe_gemv_q8_0_v11_b1_kern(
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
    moe_gemv_q8_0_v11_impl<1>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q8_0_v11_b2")]]
kernel void moe_gemv_q8_0_v11_b2_kern(
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
    moe_gemv_q8_0_v11_impl<2>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q8_0_v11_b4")]]
kernel void moe_gemv_q8_0_v11_b4_kern(
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
    moe_gemv_q8_0_v11_impl<4>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q8_0_v11_b8")]]
kernel void moe_gemv_q8_0_v11_b8_kern(
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
    moe_gemv_q8_0_v11_impl<8>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

// =============================================================================
// MoE F16 v11 (per-slot convention) — fp16 weight-streaming mirror of
// moe_gemv_q8_0_v11_impl with the dequant step removed. Buffer is per-expert
// row-major fp16: [E][D_out, D_in], expert stride = D_out * D_in halves.
// All slot routing arithmetic preserved verbatim.
// =============================================================================

template<uint MAX_SLOTS>
inline void moe_gemv_f16_v11_impl(
    device const half* hidden,
    device const uint* slot_token,    // unused — per-slot convention
    device const half* W,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    (void)slot_token;
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    uint nbc = D_in / 32;
    device const half* W_row = W + (uint64_t)expert * (uint64_t)D_out * (uint64_t)D_in + (uint64_t)n * (uint64_t)D_in;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + idx * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            // Direct fp16 weight loads — no dequant.
            half Wreg[32];
            for (uint p = 0; p < 32; ++p) {
                Wreg[p] = W_row[kb * 32 + p];
            }
            uint base_k = kb * 32;

            // Predicated multi-slot FMA. Mirrors Q8_0 V11.
            for (uint s = 0; s < MAX_SLOTS; ++s) {
                if (s < chunk) {
                    device const half* hid = hid_slots[s];
                    float a = accs[s];
                    for (uint p = 0; p < 32; ++p) {
                        a += float(hid[base_k + p]) * float(Wreg[p]);
                    }
                    accs[s] = a;
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_f16_v11_b1")]]
kernel void moe_gemv_f16_v11_b1_kern(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const half* W                [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    moe_gemv_f16_v11_impl<1>(hidden, slot_token, W, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_f16_v11_b2")]]
kernel void moe_gemv_f16_v11_b2_kern(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const half* W                [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    moe_gemv_f16_v11_impl<2>(hidden, slot_token, W, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_f16_v11_b4")]]
kernel void moe_gemv_f16_v11_b4_kern(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const half* W                [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    moe_gemv_f16_v11_impl<4>(hidden, slot_token, W, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_f16_v11_b8")]]
kernel void moe_gemv_f16_v11_b8_kern(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const half* W                [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    moe_gemv_f16_v11_impl<8>(hidden, slot_token, W, active_experts, group_start, output, D_in, D_out, tg, lid);
}

// =============================================================================
// MoE F16 v11 UP (slot_token-broadcast convention) — for ffn_gate_up_exps
// where multiple expert slots may share the same token's hidden vector.
//
// Differs from moe_gemv_f16_v11_impl above by exactly one line: the slot's
// hidden pointer is `hidden + slot_token[idx] * D_in` instead of
// `hidden + idx * D_in`. That is the standard moe_up indirection that the
// Q4_K and Q5_K moe_up V11 kernels also do — when one source token is
// routed to k experts, all k slots referring to that token must read the
// SAME source row of `hidden`. Per-slot indexing here would silently read
// neighbor slots' rows (which represent different tokens or different
// experts of the same token), producing structurally-wrong-but-bounded
// activations that propagate through the residual stream and degenerate
// the output distribution.
//
// The down kernel (moe_gemv_f16_v11_impl above) stays per-slot — that
// convention is correct for ffn_down_exps where each slot has its own
// expert-output row from the prior MoE-up step.
// =============================================================================

template<uint MAX_SLOTS>
inline void moe_gemv_f16_v11_up_impl(
    device const half* hidden,
    device const uint* slot_token,    // slot_token-broadcast convention
    device const half* W,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    uint nbc = D_in / 32;
    device const half* W_row = W + (uint64_t)expert * (uint64_t)D_out * (uint64_t)D_in + (uint64_t)n * (uint64_t)D_in;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            // slot_token-broadcast: multiple slots may share a token, so
            // indirect through slot_token[] to find this slot's source row.
            hid_slots[s] = hidden + slot_token[idx] * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            // Direct fp16 weight loads — no dequant.
            half Wreg[32];
            for (uint p = 0; p < 32; ++p) {
                Wreg[p] = W_row[kb * 32 + p];
            }
            uint base_k = kb * 32;

            // Predicated multi-slot FMA. Mirrors the Q4_K up V11 pattern.
            for (uint s = 0; s < MAX_SLOTS; ++s) {
                if (s < chunk) {
                    device const half* hid = hid_slots[s];
                    float a = accs[s];
                    for (uint p = 0; p < 32; ++p) {
                        a += float(hid[base_k + p]) * float(Wreg[p]);
                    }
                    accs[s] = a;
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_f16_v11_up_b1")]]
kernel void moe_gemv_f16_v11_up_b1_kern(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const half* W                [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    moe_gemv_f16_v11_up_impl<1>(hidden, slot_token, W, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_f16_v11_up_b2")]]
kernel void moe_gemv_f16_v11_up_b2_kern(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const half* W                [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    moe_gemv_f16_v11_up_impl<2>(hidden, slot_token, W, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_f16_v11_up_b4")]]
kernel void moe_gemv_f16_v11_up_b4_kern(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const half* W                [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    moe_gemv_f16_v11_up_impl<4>(hidden, slot_token, W, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_f16_v11_up_b8")]]
kernel void moe_gemv_f16_v11_up_b8_kern(
    device const half* hidden           [[buffer(0)]],
    device const uint* slot_token       [[buffer(1)]],
    device const half* W                [[buffer(2)]],
    device const uint* active_experts   [[buffer(3)]],
    device const uint* group_start      [[buffer(4)]],
    device half* output                 [[buffer(5)]],
    constant uint& D_in                 [[buffer(6)]],
    constant uint& D_out                [[buffer(7)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    moe_gemv_f16_v11_up_impl<8>(hidden, slot_token, W, active_experts, group_start, output, D_in, D_out, tg, lid);
}

// =============================================================================
// MoE Q8_0 v11 UP (slot_token-broadcast convention) — for ffn_gate_up_exps
// where multiple expert slots may share the same token's hidden vector.
//
// Structural intersection of moe_gemv_q8_0_v11_impl (above, per-slot down
// convention) and moe_gemv_f16_v11_up_impl (above, broadcast up convention):
//   • Swizzled Q8_0 byte layout + 2-byte half scale + 32 int8 quants per
//     block — identical to the down kernel; the swizzle is layout-agnostic
//     between up and down because loadMoESwizzled writes the same pattern
//     for both tensor classes.
//   • slot_token[idx]-broadcast hidden-pointer indirection — identical to
//     the F16 up kernel; required because one source token may be routed
//     to k experts, so all k slots referring to that token must read the
//     SAME row of `hidden`.
//
// Written 2026-05-13 to close the engine's last MoE-quant gap: previously
// only Q4_K/Q5_K/Q4_0/Q4_1/F16 were available for ffn_gate_up_exps. Adding
// Q8_0 lets us run a "Q8_0 dense + Q8_0 MoE up + Q8_0 MoE down" build,
// which is the highest-precision uniform-Q8_0 the engine can express
// without the F16 dense slow path. The on-policy LLM-as-judge probe
// surfaced a precision-vs-completion-rate trend that this kernel makes
// the next datapoint testable.
// =============================================================================

template<uint MAX_SLOTS>
inline void moe_gemv_q8_0_v11_up_impl(
    device const half* hidden,
    device const uint* slot_token,    // slot_token-broadcast convention
    device const uchar* W_sw,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 34;
    uint nbc = D_in / 32;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            // slot_token-broadcast: multiple slots may share a token, so
            // indirect through slot_token[] to find this slot's source row.
            hid_slots[s] = hidden + slot_token[idx] * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d = float(*(device const half*)(blk + 0));
            device const char* qs = (device const char*)(blk + 2);

            // Pre-dequant 32 weights to register array W[32].
            half W[32];
            for (uint p = 0; p < 32; ++p) {
                W[p] = half(d * float(qs[p]));
            }
            uint base_k = kb * 32;

            // Predicated multi-slot FMA. Mirrors Q5_K up V11 / Q8_0 down V11.
            for (uint s = 0; s < MAX_SLOTS; ++s) {
                if (s < chunk) {
                    device const half* hid = hid_slots[s];
                    float a = accs[s];
                    for (uint p = 0; p < 32; ++p) {
                        a += float(hid[base_k + p]) * float(W[p]);
                    }
                    accs[s] = a;
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_q8_0_v11_up_b1")]]
kernel void moe_gemv_q8_0_v11_up_b1_kern(
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
    moe_gemv_q8_0_v11_up_impl<1>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q8_0_v11_up_b2")]]
kernel void moe_gemv_q8_0_v11_up_b2_kern(
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
    moe_gemv_q8_0_v11_up_impl<2>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q8_0_v11_up_b4")]]
kernel void moe_gemv_q8_0_v11_up_b4_kern(
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
    moe_gemv_q8_0_v11_up_impl<4>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

[[host_name("moe_gemv_q8_0_v11_up_b8")]]
kernel void moe_gemv_q8_0_v11_up_b8_kern(
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
    moe_gemv_q8_0_v11_up_impl<8>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid);
}

// =============================================================================
// MoE Q6_K v11 (per-slot convention) — structural mirror of Q5_K V11 with
// Q6_K dequant. Used for ffn_down_exps if a future GGUF puts Q6_K there.
// 16 sub-tiles × 16 weights via the kmask/shift formula; we pre-dequant
// 64 weights per group (4 sub-tiles) into register array W[64], then run
// the predicated multi-slot FMA loop.
// =============================================================================

template<uint MAX_SLOTS>
inline void moe_gemv_q6_K_v11_impl(
    device const half* hidden,
    device const uint* slot_token,    // unused — per-slot convention
    device const uchar* W_sw,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    uint2 tg,
    uint2 lid)
{
    (void)slot_token;
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 210;
    uint nbc = D_in / 256;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + idx * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            device const ushort* ql_u16    = (device const ushort*)(blk + 0);
            device const ushort* qh_u16    = (device const ushort*)(blk + 128);
            device const char*   scales_i8 = (device const char*)(blk + 192);
            float d_all = float(*(device const half*)(blk + 208));

            // 4 groups of 4 contiguous sub-tiles, 64 weights per group.
            for (uint group = 0; group < 4; ++group) {
                half W[64];
                for (uint il_in_grp = 0; il_in_grp < 4; ++il_in_grp) {
                    uint il = group * 4 + il_in_grp;
                    uint ql_base = 32u * (il / 8u) + 16u * ((il / 2u) & 1u) + 8u * (il & 1u);
                    uint qh_base = 16u * (il / 8u) + 8u * (il & 1u);
                    float sc = float(scales_i8[(il % 2u) + 2u * (il / 2u)]);
                    uint ph = (il / 2u) & 3u;
                    uint kmask1 = (ph > 1u) ? ((ph > 2u) ? 0xC0C0C0C0u : 0x30303030u)
                                            : ((ph > 0u) ? 0x0C0C0C0Cu : 0x03030303u);
                    uint kmask2 = (ph > 1u) ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
                    float ml  = d_all * sc * 32.0f;
                    float dl0 = d_all * sc;
                    float dl1 = dl0 / 256.0f;
                    float dl2 = dl1 / 256.0f;
                    float dl3 = dl2 / 256.0f;
                    uint shr_h = (ph > 2u) ? 2u : 0u;
                    uint shl_h = (ph > 1u) ? 0u : ((ph > 0u) ? 2u : 4u);
                    uint shr_l = (ph > 1u) ? 4u : 0u;
                    for (uint i = 0; i < 4; ++i) {
                        uint low  = (uint(ql_u16[ql_base + 2u*i]) | (uint(ql_u16[ql_base + 2u*i + 1u]) << 16)) & kmask2;
                        uint high = (uint(qh_u16[qh_base + 2u*i]) | (uint(qh_u16[qh_base + 2u*i + 1u]) << 16)) & kmask1;
                        uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
                        W[il_in_grp * 16 + i*4 + 0] = half(dl0 * float(q & 0x000000FFu) - ml);
                        W[il_in_grp * 16 + i*4 + 1] = half(dl1 * float(q & 0x0000FF00u) - ml);
                        W[il_in_grp * 16 + i*4 + 2] = half(dl2 * float(q & 0x00FF0000u) - ml);
                        W[il_in_grp * 16 + i*4 + 3] = half(dl3 * float(q & 0xFF000000u) - ml);
                    }
                }
                uint base_k = kb * 256 + group * 64;
                for (uint s = 0; s < MAX_SLOTS; ++s) {
                    if (s < chunk) {
                        device const half* hid = hid_slots[s];
                        float a = accs[s];
                        for (uint i = 0; i < 64; ++i) {
                            a += float(hid[base_k + i]) * float(W[i]);
                        }
                        accs[s] = a;
                    }
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
}

[[host_name("moe_gemv_q6_K_v11_b1")]]
kernel void moe_gemv_q6_K_v11_b1_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q6_K_v11_impl<1>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q6_K_v11_b2")]]
kernel void moe_gemv_q6_K_v11_b2_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q6_K_v11_impl<2>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q6_K_v11_b4")]]
kernel void moe_gemv_q6_K_v11_b4_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q6_K_v11_impl<4>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }
[[host_name("moe_gemv_q6_K_v11_b8")]]
kernel void moe_gemv_q6_K_v11_b8_kern(
    device const half* hidden [[buffer(0)]], device const uint* slot_token [[buffer(1)]],
    device const uchar* W_sw [[buffer(2)]], device const uint* active_experts [[buffer(3)]],
    device const uint* group_start [[buffer(4)]], device half* output [[buffer(5)]],
    constant uint& D_in [[buffer(6)]], constant uint& D_out [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{ moe_gemv_q6_K_v11_impl<8>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, tg, lid); }

// MoE Q4_K v12 — V11 + per-kb tg-mem nibble lookup table (compute-bound
// dequant rewrite). Targets the 32% gap between V11's measured MoE wall
// (~26 ms/step at B=8) and the bandwidth ceiling (~18 ms). The gap is
// instruction-dispatch latency on the Q4_K dequant chain (nibble extract
// + scale-mul + bias-sub per byte), not bandwidth.
//
// Structural change vs V11:
//
//   For each kb iteration, before the FMA loop:
//     - V11 computes dl_arr[8] / ml_arr[8] in registers (kept here).
//     - V12 ADDITIONALLY pre-computes the FULL 8×16 nibble lookup table
//       in tg-mem: nibble_table[sb][v] = dl_arr[sb] * v - ml_arr[sb].
//       128 entries built cooperatively across the SG's 32 threads,
//       4 entries per thread. simdgroup_barrier seals the build.
//
//   Inner FMA loop replaces V11's per-byte mul-sub with a tg-mem load:
//     V11: w_lo = dl_lo * float(byte & 0xF) - ml_lo;     // 2 ALU ops
//     V12: w_lo = nibble_table[sb_lo][byte & 0xF];        // 1 tg-mem read
//
// Per-byte cycle count (estimated for Apple M-series GPU):
//   V11: 2 nibble extracts + 2 fp32 mul-sub + 2 cvt-to-half ≈ 6 ALU cycles
//        + 16 FMAs at MAX_SLOTS=8 = ~22 cycles per byte
//   V12: 2 tg-mem reads (1-2 cycles each) + 16 FMAs ≈ 18-20 cycles per byte
//   Saving: 3-4 cycles per byte × 32 bytes/pair × 4 pairs × 11 kbs × 64
//   experts × 30 layers ≈ 5-6 ms per step at B=8 (estimate).
//
// Register pressure DROPS vs V11: V11 held W_lo[32] / W_hi[32] half scratch
// (32 32-bit slots/thread); V12 doesn't need it. Lower register pressure
// → potentially higher SM occupancy.
//
// tg-mem footprint per TG: 8 × 16 × 2 bytes = 256 B. Negligible.
//
// Dispatch shape, buffer layout, output write convention, slot_token
// indexing — all match V6/V11.
template<uint MAX_SLOTS>
inline void moe_gemv_q4k_v12_impl(
    device const half* hidden,
    device const uint* slot_token,
    device const uchar* W_sw,
    device const uint* active_experts,
    device const uint* group_start,
    device half* output,
    constant uint& D_in,
    constant uint& D_out,
    threadgroup half (&nibble_table)[8][16],
    uint2 tg,
    uint2 lid)
{
    uint n_block = tg.x; uint ai = tg.y; uint expert = active_experts[ai]; uint t = lid.x;
    if (expert >= 128u) return;
    uint n = n_block * 32 + t;
    if (n >= D_out) return;
    uint gb = group_start[expert]; uint ge = group_start[expert + 1];
    if (ge == gb) return;

    constexpr uint BLK = 144;
    uint nbc = D_in / 256;
    uint super_bytes = nbc * 32 * BLK;
    uint expert_bytes = (D_out / 32) * super_bytes;
    device const uchar* W_super = W_sw + expert * expert_bytes + n_block * super_bytes;

    uint slot_base = gb;
    while (slot_base < ge) {
        uint chunk = (ge - slot_base < MAX_SLOTS) ? (ge - slot_base) : MAX_SLOTS;
        device const half* hid_slots[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) {
            uint idx = (s < chunk) ? (slot_base + s) : slot_base;
            hid_slots[s] = hidden + slot_token[idx] * D_in;
        }
        float accs[MAX_SLOTS];
        for (uint s = 0; s < MAX_SLOTS; ++s) accs[s] = 0.0f;

        for (uint kb = 0; kb < nbc; ++kb) {
            device const uchar* blk = W_super + kb * 32 * BLK + t * BLK;
            float d    = float(*(device const half*)(blk + 0));
            float dmin = float(*(device const half*)(blk + 2));
            device const uchar* scales = blk + 4;
            device const uchar* qs     = blk + 16;

            // Hoist scales (V11 Approach 2, kept).
            float dl_arr[8], ml_arr[8];
            for (uint sb = 0; sb < 8; ++sb) {
                uchar sc, mn;
                unpack_q4k_scales(scales, sc, mn, sb);
                dl_arr[sb] = d * float(sc);
                ml_arr[sb] = dmin * float(mn);
            }

            // V12: build per-kb nibble lookup table cooperatively across
            // the SG's 32 threads. 128 entries / 32 threads = 4 entries
            // per thread. Layout: nibble_table[sb][v] = dl[sb]*v - ml[sb].
            // Each thread builds entries [t*4 .. t*4+3] of the flattened
            // table.
            //
            // CAVEAT: each thread t computes its OWN dl_arr/ml_arr from
            // its OWN column's d/dmin. So the table THIS THREAD writes is
            // specific to its column. Other threads write tables for
            // their columns. The result: 32 different per-thread "table
            // entries" living in the shared tg-mem region. This is wrong
            // for shared-table semantics — each thread's entries clobber
            // its neighbors.
            //
            // FIX: each thread builds its OWN private 8x16 table. Simplest
            // way: keep it in registers. 128 halves = 64 32-bit per thread
            // — too high. Better: build only the needed pair's entries
            // per-pair iteration. 32 entries per pair = 16 32-bit. Doable.
            //
            // (Switching the kernel to per-pair private register table.)
            for (uint pair = 0; pair < 4; ++pair) {
                uint sb_lo = pair * 2;
                uint sb_hi = pair * 2 + 1;
                float dl_lo = dl_arr[sb_lo], ml_lo = ml_arr[sb_lo];
                float dl_hi = dl_arr[sb_hi], ml_hi = ml_arr[sb_hi];

                // Per-thread private nibble table for this pair.
                // 16 lo entries + 16 hi entries = 32 halves in registers.
                half tbl_lo[16], tbl_hi[16];
                for (uint v = 0; v < 16; ++v) {
                    tbl_lo[v] = half(dl_lo * float(v) - ml_lo);
                    tbl_hi[v] = half(dl_hi * float(v) - ml_hi);
                }
                uint base_lo = kb * 256 + sb_lo * 32;
                uint base_hi = kb * 256 + sb_hi * 32;

                for (uint p = 0; p < 32; ++p) {
                    uchar byte = qs[pair * 32 + p];
                    half w_lo = tbl_lo[byte & 0xF];
                    half w_hi = tbl_hi[(byte >> 4) & 0xF];
                    for (uint s = 0; s < MAX_SLOTS; ++s) {
                        if (s < chunk) {
                            device const half* hid = hid_slots[s];
                            accs[s] += float(hid[base_lo + p]) * float(w_lo)
                                     + float(hid[base_hi + p]) * float(w_hi);
                        }
                    }
                }
            }
        }

        for (uint s = 0; s < MAX_SLOTS; ++s) {
            if (s < chunk) {
                output[(slot_base + s) * D_out + n] = half(accs[s]);
            }
        }
        slot_base += chunk;
    }
    (void)nibble_table;  // unused — kept in signature for ABI parity if we
                        // later want to switch to a tg-mem table layout.
}

[[host_name("moe_gemv_q4k_v12_b1")]]
kernel void moe_gemv_q4k_v12_b1_kern(
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
    threadgroup half nibble_table[8][16];
    moe_gemv_q4k_v12_impl<1>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, nibble_table, tg, lid);
}

[[host_name("moe_gemv_q4k_v12_b2")]]
kernel void moe_gemv_q4k_v12_b2_kern(
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
    threadgroup half nibble_table[8][16];
    moe_gemv_q4k_v12_impl<2>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, nibble_table, tg, lid);
}

[[host_name("moe_gemv_q4k_v12_b4")]]
kernel void moe_gemv_q4k_v12_b4_kern(
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
    threadgroup half nibble_table[8][16];
    moe_gemv_q4k_v12_impl<4>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, nibble_table, tg, lid);
}

[[host_name("moe_gemv_q4k_v12_b8")]]
kernel void moe_gemv_q4k_v12_b8_kern(
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
    threadgroup half nibble_table[8][16];
    moe_gemv_q4k_v12_impl<8>(hidden, slot_token, W_sw, active_experts, group_start, output, D_in, D_out, nibble_table, tg, lid);
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

// Batched 2D position-embedding add. One dispatch over ALL B*N rows.
// Instead of recomputing grid coords from a single per-image nPatchesX
// (which was wrong across a batch of images with differing gridW, and
// forced a per-image Swift dispatch loop), this indexes the precomputed
// per-row grid coordinates posY[row]/posX[row] ([B*N], filled CPU-side
// from each image's patch positions, (0,0) for padded rows). Correct by
// construction for any batch composition; B=1 is byte-identical since
// posY[i]==i/gridW, posX[i]==i%gridW for a single image's real patches.
kernel void vision_pos_embed_add_fp32(
    device const float* x               [[buffer(0)]],
    device const half* pos_y_table      [[buffer(1)]],
    device const half* pos_x_table      [[buffer(2)]],
    device float* out                   [[buffer(3)]],
    device const uint* posY             [[buffer(4)]],   // [B*N] grid-y per row
    device const uint* posX             [[buffer(5)]],   // [B*N] grid-x per row
    constant uint& hidden               [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint row = tg.x; uint t = lid.x;
    uint yi = posY[row];
    uint xi = posX[row];
    device const half* py = pos_y_table + yi * hidden;
    device const half* px = pos_x_table + xi * hidden;
    device const float* xi_ptr = x + row * hidden;
    device float* out_ptr = out + row * hidden;
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

// Soft-prompt softs ingest: copy-and-cast a [n_rows, HIDDEN] fp32 region of
// `src` (vision-tower padded output) into a [n_rows, HIDDEN] slice of `dst`
// (LM pre_hidden, fp16). Replaces the CPU memcpy + fp32→fp16 loop that
// used to follow `pendingCB.waitUntilCompleted()` in the soft-prefill prep.
// Encoded into a pre-prefill CB on the LM queue, gated by
// encodeWaitForEvent on the cross-queue vision event — GPU waits for the
// vision pad-blit, then runs this copy, then the prefill CB consumes
// pre_hidden in queue order. CPU never blocks. See notes/engine_debloat.md.
kernel void vision_softs_copy_fp32_to_fp16(
    device const float* src     [[buffer(0)]],
    device half* dst            [[buffer(1)]],
    constant uint& src_row_off  [[buffer(2)]],
    constant uint& dst_row_off  [[buffer(3)]],
    constant uint& n_rows       [[buffer(4)]],
    constant uint& hidden       [[buffer(5)]],
    uint gid                    [[thread_position_in_grid]])
{
    uint total = n_rows * hidden;
    if (gid >= total) { return; }
    dst[(dst_row_off * hidden) + gid] = half(src[(src_row_off * hidden) + gid]);
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
// ONE kernel source. Specialized per-PSO via function constants:
//   FC_D          — head dim (256 for slide layers, 512 for full)
//   FC_PAGE       — KV-cache page size in tokens (16 for slide, 8 for full)
//   FC_Q_PER_TG   — real Q rows per threadgroup (2 for slide, 8 for full)
//   FC_USE_SLIDE  — 1 enables per-row sliding-window mask; 0 = causal only
//
// Metal function constants are resolved at PSO compile time; the compiler
// constant-folds the `if (FC_USE_SLIDE)` branches and fixes array sizes
// (threadgroup allocations, Q_tile/scores_tile) so each PSO lowers to
// the same asm the old per-variant kernels did.
// ========================================================================
constant int  FC_D         [[function_constant(0)]];
constant int  FC_PAGE      [[function_constant(1)]];
constant int  FC_Q_PER_TG  [[function_constant(2)]];
constant bool FC_USE_SLIDE [[function_constant(3)]];

kernel void flex_attn_v0(
    device const half* Q                    [[buffer(0)]],
    device const KVChunks& k_chunks         [[buffer(1)]],
    device const KVChunks& v_chunks         [[buffer(2)]],
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
    constant uint& sliding_window           [[buffer(17)]],    // value ignored when !FC_USE_SLIDE; always bound by wrapper
    constant uint& prefix_pages             [[buffer(18)]],    // skip logical pages < this (tail mode). 0 = process all.
    constant uint& split_offset             [[buffer(19)]],    // write partials at total_splits_out stride, + split_offset + tg.y.
    constant uint& total_splits_out         [[buffer(20)]],    // output layout stride. 0 → fallback to N_SPLITS.
    constant uint& chunk_pages              [[buffer(27)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint THREADS = 32;
    constexpr uint MMA = 8;                 // simdgroup_half8x8 is fixed 8×8

    // MSL library compile happens before function constants resolve, so
    // threadgroup/stack array sizes can't use FC_* directly. We size to
    // the max over the two specialized PSOs (D=512, PAGE=16, Q_PER_TG=8)
    // and only use the FC_*-sized prefix at runtime. The slide PSO leaves
    // the tail of each array unused — a few KB of wasted threadgroup
    // memory, well within the 32 KB/TG Apple GPU budget.
    constexpr uint MAX_D        = 512;
    constexpr uint MAX_PAGE     = 16;
    constexpr uint MAX_Q_PER_TG = 8;

    const     uint D        = uint(FC_D);
    const     uint PAGE     = uint(FC_PAGE);
    const     uint Q_PER_TG = uint(FC_Q_PER_TG);
    const     uint D8       = D / 8;

    const uint vs = tg_pos.x;
    const uint split = tg_pos.y;
    const uint slot = vs / H_KV;
    const uint kv_head = vs % H_KV;
    const uint q_head_base = kv_head * Q_PER_TG;
    const uint lid = lid3.x;

    // Q_BLOCK=1, q_blocks=1 → CSR index is just the slot.
    const uint csr_idx = slot;

    // Q_tile always holds MMA rows of up-to-MAX_D columns. When Q_PER_TG <
    // MMA the trailing rows are zero-padded so they contribute 0 to the
    // 8×8 MMA output and are ignored by the softmax/AV loops (which
    // iterate q < Q_PER_TG only).
    //
    // 2026-05-06 refactor: O_acc moved to per-lane registers
    // (`O_local[MAX_Q_PER_TG][MAX_D_PER_LANE]`). Each lane owns
    // MAX_D/THREADS = 16 disjoint d-slots × MAX_Q_PER_TG = 8 rows
    // = 128 floats = 512 B per lane. Drops static threadgroup-memory
    // budget by 16 KB (was 24.9 KB → 8.5 KB), keeping the doubled
    // accounting (simdgroup_matrix operand-tile double-buffering)
    // safely under the 32 KB hardware limit.
    threadgroup half  Q_tile[MMA * MAX_D];
    threadgroup half  scores_tile[MMA * MAX_PAGE];
    threadgroup float m_state[MAX_Q_PER_TG];
    threadgroup float l_state[MAX_Q_PER_TG];
    threadgroup float scale_tile[MAX_Q_PER_TG];

    // Per-lane register accumulator for O. Sized to the max specialization
    // (Q_PER_TG=8, D_PER_LANE=MAX_D/THREADS=16). Slide PSO uses only the
    // first FC_Q_PER_TG rows × FC_D/THREADS d-slots; remaining slots are
    // unused but cost only register pressure, no memory traffic.
    constexpr uint MAX_D_PER_LANE = MAX_D / THREADS;   // 16
    const     uint D_PER_LANE     = D / THREADS;       // 8 (slide) or 16 (full)
    float O_local[MAX_Q_PER_TG][MAX_D_PER_LANE];

    device const half* Qbase = Q + (slot * H_Q + q_head_base) * D;
    device const uint* bt_s = block_table + slot * max_pages;

    // Load Q_PER_TG real rows; zero-pad up to MMA for the simdgroup MMA.
    for (uint i = lid; i < MMA * D; i += THREADS) {
        uint r = i / D;
        Q_tile[i] = (r < Q_PER_TG) ? Qbase[r * D + (i % D)] : half(0);
    }
    // Zero per-lane O accumulator. Pure register init — no threadgroup
    // memory traffic, no barrier needed (each lane writes only its own
    // private slots).
    for (uint q = 0; q < Q_PER_TG; ++q) {
        for (uint i = 0; i < D_PER_LANE; ++i) O_local[q][i] = 0.0f;
    }
    if (lid < Q_PER_TG) { m_state[lid] = -INFINITY; l_state[lid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k_len = k_len_per_slot[slot];
    const uint kv_row_stride = H_KV * D;
    const uint window_lo = (FC_USE_SLIDE && sliding_window > 0 && k_len > sliding_window)
                           ? (k_len - sliding_window) : 0u;

    // Each split gets a contiguous slice of the per-slot list (FULL first,
    // PARTIAL second, concatenated). Keeps work balanced across splits.
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
        // Tail mode: skip logical pages owned by the shared-prefix broadcast.
        if (p < prefix_pages) continue;

        const uint phys = bt_s[p];
        const uint chunk_idx = phys / chunk_pages;
        const uint local_phys = phys - chunk_idx * chunk_pages;
        device const half* Kx = k_chunks.chunks[chunk_idx];
        device const half* Vx = v_chunks.chunks[chunk_idx];
        device const half* Kbase = Kx + (local_phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = Vx + (local_phys * PAGE * H_KV + kv_head) * D;

        // QK: PAGE/8 passes of 8 K-columns each (1 pass at PAGE=8, 2 at PAGE=16).
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

        // Per-row online softmax. Only the partial-block path checks masks.
        if (lid < Q_PER_TG) {
            const uint q = lid;
            float row_max = -INFINITY;
            float s_loc[MAX_PAGE];
            if (is_partial) {
                for (uint k = 0; k < PAGE; ++k) {
                    float sv = float(scores_tile[q * PAGE + k]) * qk_scale;
                    uint k_pos = p * PAGE + k;
                    // causal: k_pos past k_len is always masked
                    if (k_pos >= k_len) sv = -INFINITY;
                    // sliding: reject below window_lo (elided when !FC_USE_SLIDE)
                    if (FC_USE_SLIDE) {
                        if (k_pos < window_lo) sv = -INFINITY;
                    }
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

        // AV: scalar cooperative. Each lane owns D/THREADS d-slots in
        // per-lane registers (`O_local[q][i]`) across all ix iterations.
        for (uint i = 0; i < D_PER_LANE; ++i) {
            uint d = lid + i * THREADS;
            half V_reg[MAX_PAGE];
            for (uint k = 0; k < PAGE; ++k) {
                V_reg[k] = Vbase[k * kv_row_stride + d];
            }
            for (uint q = 0; q < Q_PER_TG; ++q) {
                float acc = O_local[q][i] * scale_tile[q];
                for (uint k = 0; k < PAGE; ++k) {
                    acc += float(scores_tile[q * PAGE + k]) * float(V_reg[k]);
                }
                O_local[q][i] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials per Q head. When split_offset>0, this writes the tail
    // slice of a shared partials buffer whose split=0 belongs to the
    // shared-prefix broadcast kernel.
    const uint out_stride = (total_splits_out > 0) ? total_splits_out : N_SPLITS;
    const bool empty_split = (ix_begin >= ix_end);
    for (uint q = 0; q < Q_PER_TG; ++q) {
        const uint q_head = q_head_base + q;
        const uint pidx = (slot * H_Q + q_head) * out_stride + split_offset + split;
        if (lid == 0) {
            m_partials[pidx] = empty_split ? -INFINITY : m_state[q];
            l_partials[pidx] = empty_split ? 0.0f : l_state[q];
        }
        // One device store per (lane, q, d-slot) — covers D/THREADS
        // contiguous-strided d positions per lane.
        device float* O_part = O_partials + pidx * D;
        for (uint i = 0; i < D_PER_LANE; ++i) {
            uint d = lid + i * THREADS;
            O_part[d] = O_local[q][i];
        }
    }
}


// Flex attention v1 — slide layer with Q_BLOCK=8 (prefill).
//
// 2026-05-06 refactor: switched from Q_PER_TG=2 (16 Q rows per TG) to
// Q_PER_TG=1 (8 Q rows per TG, ONE q_head per TG). Matches the
// flex_attn_full_prefill geometry. Reason: the Q_PER_TG=2 layout
// declared 25,312 bytes of static threadgroup memory which Metal
// counts as ~50 KB after compiler-side simdgroup-matrix double-
// buffering — over the 32 KB hardware limit. Apple Silicon tolerated
// the overflow at small KV-cache sizes (production worked at
// pool=8192) but the threadgroup-mem assertion is real and fires
// under MTL_DEBUG_LAYER. Single-q-head-per-TG drops static usage
// to ~12.6 KB, well under 32 KB even with compiler doubling.
//
// Grid changes: x dim was B*H_KV (each TG handled Q_PER_TG q_heads
// for one kv_head); now B*H_Q (each TG handles ONE q_head). H_KV is
// derived inside the kernel via kv_head = (q_head * H_KV) / H_Q. The
// dispatcher in encFlexAttnSlidePrefill must use the new grid.
//
// Each TG covers 8 consecutive Q positions × 1 Q-head → 8 real Q rows.
// Per-Q-row softmax applies a per-row causal+sliding mask using
// per-row q_positions. Partials laid out per (q_pos, q_head).
kernel void flex_attn_slide_v1_q8(
    device const half* Q                    [[buffer(0)]],   // [B, Q_LEN, H_Q, D]
    device const KVChunks& k_chunks         [[buffer(1)]],
    device const KVChunks& v_chunks         [[buffer(2)]],
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
    constant uint& chunk_pages              [[buffer(27)]],
    uint3 tg_pos                            [[threadgroup_position_in_grid]],
    uint3 lid3                              [[thread_position_in_threadgroup]])
{
    constexpr uint D = 256;
    constexpr uint PAGE = 16;
    constexpr uint THREADS = 32;
    constexpr uint Q_BLOCK = 8;
    constexpr uint Q_ROWS = Q_BLOCK;          // 8 (was Q_PER_TG * Q_BLOCK = 16)
    constexpr uint D8 = D / 8;

    const uint vs = tg_pos.x;                 // ranges B * H_Q (was B * H_KV)
    const uint q_block_idx = tg_pos.y;        // which Q tile
    const uint split = tg_pos.z;
    const uint slot = vs / H_Q;               // (was vs / H_KV)
    const uint q_head = vs % H_Q;             // (was kv_head, q_head_base = kv_head * Q_PER_TG)
    const uint kv_head = (q_head * H_KV) / H_Q;
    const uint lid = lid3.x;

    const uint q_blocks_per_slot = (q_len + Q_BLOCK - 1) / Q_BLOCK;
    const uint csr_idx = slot * q_blocks_per_slot + q_block_idx;
    const uint q_local_base = q_block_idx * Q_BLOCK;

    threadgroup half  Q_tile[Q_ROWS * D];           // 8 * 256 * 2 =  4096 B
    threadgroup half  scores_tile[Q_ROWS * PAGE];   // 8 *  16 * 2 =   256 B
    threadgroup float O_acc[Q_ROWS * D];            // 8 * 256 * 4 = 8192 B
    threadgroup float m_state[Q_ROWS];              // 8 *       4 =   32 B
    threadgroup float l_state[Q_ROWS];              //                  32 B
    threadgroup float scale_tile[Q_ROWS];           //                  32 B
    threadgroup uint  q_pos_tg[Q_BLOCK];            // 8 *       4 =   32 B
                                                     // total: ~12,672 B static

    device const uint* bt_s = block_table + slot * max_pages;

    // Load Q_tile with layout [row=q_local, dim=d]. Each row corresponds
    // to one Q position for our single q_head. Padding rows zero out so
    // their MMA contributes zero.
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
        const uint chunk_idx = phys / chunk_pages;
        const uint local_phys = phys - chunk_idx * chunk_pages;
        device const half* Kx = k_chunks.chunks[chunk_idx];
        device const half* Vx = v_chunks.chunks[chunk_idx];
        device const half* Kbase = Kx + (local_phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = Vx + (local_phys * PAGE * H_KV + kv_head) * D;

        // QK via simdgroup_matrix. Q_ROWS=8 rows → 1 MMA pass (was 2 at
        // Q_ROWS=16). PAGE=16 → 2 K col slabs (pb=0, 1).
        for (uint pb = 0; pb < PAGE / 8; ++pb) {
            device const half* pk = Kbase + (pb * 8) * kv_row_stride;
            // Q_ROWS / 8 = 1 — single qp pass.
            simdgroup_half8x8 mqk = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            for (uint dt = 0; dt < D8; ++dt) {
                simdgroup_half8x8 mq, mk;
                simdgroup_load(mq, Q_tile + dt * 8, D);
                simdgroup_load(mk, pk + dt * 8, kv_row_stride, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }
            simdgroup_store(mqk, scores_tile + pb * 8, PAGE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row online softmax. For FULL blocks every element is kept; for
        // PARTIAL blocks we read a per-(q_local, block) bitmap produced by
        // the CPU mask-mod policy. Bit k of the per-row uint = 1 ⇒ keep, 0
        // ⇒ -∞. Rows past the real sequence end stay at -INF.
        //
        // With Q_PER_TG=1 (was =2), q_local = r directly. The mask layout
        // is [total_partials, Q_BLOCK=8] which still indexes per (q_pos
        // within tile = q_local), so partial_block_masks[partial_idx * 8 +
        // q_local] is unchanged.
        if (lid < Q_ROWS) {
            const uint r = lid;
            const uint q_local = r;
            const uint q_pos_in_seq = q_local_base + q_local;
            if (q_pos_in_seq < q_len) {
                uint mask_word = 0u;
                if (is_partial) {
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

    // Write partials per real (q_pos, q_head=this TG's q_head). One Q row
    // per Q position now (was Q_PER_TG rows per position).
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

// Flex attention v1 full-attention for prefill.
// llama.cpp-style geometry: ONE TG per (slot, q_head, q_block). Each TG owns
// Q_BLOCK=8 queries of a SINGLE q_head (not Q_PER_TG grouped like v0).
//
// 2026-05-06 refactor: moved the 16 KB O_acc accumulator out of threadgroup
// memory and into per-lane registers (`O_local[Q_ROWS][D_PER_LANE]`). Each
// of the 32 lanes in the threadgroup owns D/32 = 16 disjoint d-slots
// across all 8 query rows = 128 floats = 512 B per lane — well within
// Apple GPU's register budget, and faster than threadgroup memory in the
// inner accumulate loop. Threadgroup-memory budget at D=512:
//   Q_tile:      8 * 512 * 2 = 8192 B
//   scores_tile: 8 * 8 * 2   = 128 B
//   m/l/scale:   8 * 12       = 96 B
//   q_pos_tg:    8 * 4         = 32 B
//   ---------------------------------
//   total:                     ~8.4 KB (was ~24.5 KB before O_acc → reg)
// Why it matters: simdgroup_matrix usage causes Metal's compiler to
// double-buffer operand tiles, putting the effective threadgroup-mem
// reservation near 2× the static accounting. At ~24.5 KB static the
// kernel was clearing 49 KB doubled — over the 32 KB hardware limit.
// At ~8.4 KB static the doubled accounting (~17 KB) is comfortably
// under, with substantial headroom.
//
// Grid: (B * H_Q, q_blocks, N_SPLITS). kv_head = (q_head * H_KV) / H_Q is
// derived inside the kernel; multiple q_heads that share a KV head re-read
// the same K — on Apple Silicon with unified memory this hits L1/L2 cache.
// Mask: pure causal (full-attn layers have no sliding window).
kernel void flex_attn_full_prefill(
    device const half* Q                    [[buffer(0)]],
    device const KVChunks& k_chunks         [[buffer(1)]],
    device const KVChunks& v_chunks         [[buffer(2)]],
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
    constant uint& chunk_pages              [[buffer(27)]],
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
    threadgroup float m_state[Q_ROWS];
    threadgroup float l_state[Q_ROWS];
    threadgroup float scale_tile[Q_ROWS];
    threadgroup uint  q_pos_tg[Q_BLOCK];

    // Per-lane register accumulator. Replaces the 16 KB threadgroup
    // O_acc[Q_ROWS * D] tile. Each lane owns disjoint d-slots
    // {lid, lid+THREADS, lid+2*THREADS, ...} across all Q_ROWS query
    // rows. D/THREADS = 512/32 = 16 d-slots × 8 rows = 128 floats per
    // lane = 512 B, fits comfortably in Apple GPU registers.
    constexpr uint D_PER_LANE = D / THREADS;   // 16
    float O_local[Q_ROWS][D_PER_LANE];

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
    // Zero the per-lane accumulator (pure-register init, no threadgroup-
    // memory traffic and no barrier needed because each lane only touches
    // its own private slots).
    for (uint r = 0; r < Q_ROWS; ++r) {
        for (uint i = 0; i < D_PER_LANE; ++i) O_local[r][i] = 0.0f;
    }
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
        const uint chunk_idx = phys / chunk_pages;
        const uint local_phys = phys - chunk_idx * chunk_pages;
        device const half* Kx = k_chunks.chunks[chunk_idx];
        device const half* Vx = v_chunks.chunks[chunk_idx];
        device const half* Kbase = Kx + (local_phys * PAGE * H_KV + kv_head) * D;
        device const half* Vbase = Vx + (local_phys * PAGE * H_KV + kv_head) * D;

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

        // AV scalar cooperative: each lane owns D/THREADS=16 dims, kept
        // in per-lane registers (`O_local[r][i]`) across all ix iterations.
        for (uint i = 0; i < D_PER_LANE; ++i) {
            uint d = lid + i * THREADS;
            half V_reg[PAGE];
            for (uint k = 0; k < PAGE; ++k) V_reg[k] = Vbase[k * kv_row_stride + d];
            for (uint r = 0; r < Q_ROWS; ++r) {
                float acc = O_local[r][i] * scale_tile[r];
                for (uint k = 0; k < PAGE; ++k) {
                    acc += float(scores_tile[r * PAGE + k]) * float(V_reg[k]);
                }
                O_local[r][i] = acc;
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
        // One device store per (lane, r, d-slot) — each lane covers
        // D_PER_LANE = D/THREADS contiguous-strided d positions.
        device float* O_part = O_partials + pidx * D;
        for (uint i = 0; i < D_PER_LANE; ++i) {
            uint d = lid + i * THREADS;
            O_part[d] = O_local[r][i];
        }
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
// Batched 2D average-pool. ONE dispatch over ALL pooled tokens across the
// whole batch (totalPooled = Σ_i nPooled_i). Each global pooled token g maps
// back to its source image via tokenImage[g]; per-image grid width / output
// width come from imgGridW/imgOutW; the image's patch rows start at img*N in
// the batched [B*N, hidden] input. Writes its own disjoint out[g] slot — no
// shared scratch, no per-image dispatch loop, correct for any batch.
kernel void vision_pool_2d_fp32in_fp32out(
    device const float* x               [[buffer(0)]],   // [B*N, hidden]
    device float* out                   [[buffer(1)]],   // [totalPooled, hidden]
    device const uint* tokenImage       [[buffer(2)]],   // [totalPooled] -> image idx
    device const uint* pooledStart      [[buffer(3)]],   // [B] prefix sum of nPooled
    device const uint* imgGridW         [[buffer(4)]],   // [B]
    device const uint* imgOutW          [[buffer(5)]],   // [B]
    constant uint& N                    [[buffer(6)]],    // patches per image (uniform)
    constant uint& kernel_size          [[buffer(7)]],
    constant uint& hidden               [[buffer(8)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    uint g = tg.x; uint t = lid.x;
    uint img   = tokenImage[g];
    uint local = g - pooledStart[img];
    uint out_w = imgOutW[img];
    uint gw    = imgGridW[img];
    uint base  = img * N;                       // patch-row base of this image
    uint oy = local / out_w;
    uint ox = local % out_w;
    uint y_start = oy * kernel_size;
    uint x_start = ox * kernel_size;
    float inv_area = 1.0f / float(kernel_size * kernel_size);
    for (uint i = t; i < hidden; i += 32) {
        float acc = 0;
        for (uint dy = 0; dy < kernel_size; ++dy) {
            for (uint dx = 0; dx < kernel_size; ++dx) {
                uint px = base + (y_start + dy) * gw + (x_start + dx);
                acc += x[px * hidden + i];
            }
        }
        out[g * hidden + i] = acc * inv_area;
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

// ============================================================================
// GPU-side sampling kernel — Phase 1+1b of the dataflow pipeline spec.
// See docs/dataflow_pipeline_spec.md §2.1.
//
// Direct port of CPU `sampleTokenFromLogits` (lm_engine.swift): same
// inverse-CDF softmax-sampling algorithm, same temperature/min_p
// semantics, same T<=0 argmax fast path, same additive per-slot
// logit_bias. The ONLY substitution is the PRNG — Swift's stdlib RNG
// cannot be called from Metal, so we use philox-4x32-10 keyed on
// (seed, step, slot) for the per-slot uniform draw. Output distribution
// is identical to the CPU path; specific draws differ because the PRNG
// differs.
//
// Grid: (B, 1, 1). One TG per slot. 32 threads per TG (one simdgroup)
// scan VOCAB cooperatively for the max/sum-exp reductions; the final
// inverse-CDF walk runs on lid==0 only (~ms at VOCAB=262144, amortized
// against a ~100ms step).
//
// logit_bias is a dense [B, VOCAB] fp32 buffer. Slots without a
// client-set bias hold all-zeros, so the add is a no-op uniform across
// threads. No data-conditional branch inside the kernel.
//
// When `sampling_active[slot] == 0` the kernel leaves input_tokens
// untouched — caller's responsibility to ignore idle slots.
// ============================================================================

// philox-4x32-10 constants.
#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

static inline uint _mulhi_u32(uint x, uint y) {
    return uint((ulong(x) * ulong(y)) >> 32);
}

// One uniform float in [0, 1) per (seed, step, slot). Same tuple →
// same draw, different tuples → statistically-independent draws. 10
// rounds matches the standard Random123 `philox4x32_10` (Salmon et al.
// SC'11). We consume the high 24 bits of ctr.x to build a float —
// avoids LSB bias that some PRNG outputs have.
static inline float philox_uniform(uint seed, uint step_id, uint slot) {
    uint4 ctr = uint4(step_id, 0u, slot, 0u);
    uint2 key = uint2(seed, 0xA3C59A5Du);  // fixed key suffix
    for (uint r = 0; r < 10; ++r) {
        uint hi0 = _mulhi_u32(ctr.x, PHILOX_M0);
        uint lo0 = ctr.x * PHILOX_M0;
        uint hi1 = _mulhi_u32(ctr.z, PHILOX_M1);
        uint lo1 = ctr.z * PHILOX_M1;
        uint4 c;
        c.x = hi1 ^ ctr.y ^ key.x;
        c.y = lo1;
        c.z = hi0 ^ ctr.w ^ key.y;
        c.w = lo0;
        ctr = c;
        if (r < 9) {
            key.x += PHILOX_W0;
            key.y += PHILOX_W1;
        }
    }
    return float(ctr.x >> 8) * (1.0f / 16777216.0f);   // 2^-24
}

kernel void sample_token(
    device const half* logits                [[buffer(0)]],   // [B, VOCAB]
    device const float* sampling_logit_bias  [[buffer(1)]],   // [B, VOCAB]
    device const float* sampling_temperature [[buffer(2)]],   // [B]
    device const float* sampling_min_p       [[buffer(3)]],   // [B]
    device const uint* sampling_seed         [[buffer(4)]],   // [B]
    device const uint* sampling_step         [[buffer(5)]],   // [B]
    device const uint* sampling_active       [[buffer(6)]],   // [B] 0/1
    device uint* input_tokens                [[buffer(7)]],   // [B] — WRITE
    constant uint& VOCAB                     [[buffer(8)]],
    uint3 tg_pos                             [[threadgroup_position_in_grid]],
    uint3 lid3                               [[thread_position_in_threadgroup]])
{
    const uint slot = tg_pos.x;
    const uint lid  = lid3.x;
    constexpr uint THREADS = 32;

    if (sampling_active[slot] == 0) return;

    device const half* L = logits + slot * VOCAB;
    device const float* BIAS = sampling_logit_bias + slot * VOCAB;
    const float temperature = sampling_temperature[slot];

    // T <= 0: argmax fast path over (logit + bias). Matches CPU
    // `rawLogit(v)` which always applies bias regardless of temperature.
    // Ties broken toward the lowest token id (strict `>` in CPU; same
    // tie rule here).
    if (temperature <= 0.0f) {
        uint local_best_id = 0xFFFFFFFFu;
        float local_best_val = -INFINITY;
        for (uint v = lid; v < VOCAB; v += THREADS) {
            float x = float(L[v]) + BIAS[v];
            if (x > local_best_val || (x == local_best_val && v < local_best_id)) {
                local_best_val = x;
                local_best_id  = v;
            }
        }
        threadgroup float vals[THREADS];
        threadgroup uint  ids [THREADS];
        vals[lid] = local_best_val;
        ids [lid] = local_best_id;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = THREADS / 2; s > 0; s /= 2) {
            if (lid < s) {
                float oV = vals[lid + s];
                uint  oI = ids [lid + s];
                if (oV > vals[lid] || (oV == vals[lid] && oI < ids[lid])) {
                    vals[lid] = oV;
                    ids [lid] = oI;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0) input_tokens[slot] = ids[0];
        return;
    }

    // T > 0: inverse-CDF softmax sampling. Three vocab passes, matching
    // the CPU algorithm structure pass-for-pass. CPU's `rawLogit(v) * tInv`
    // becomes `(logit[v] + bias[v]) * tInv` here — same expression.
    const float tInv = 1.0f / temperature;
    const float minP = sampling_min_p[slot];

    // Pass 1: max of (logit + bias) * tInv.
    float local_max = -INFINITY;
    for (uint v = lid; v < VOCAB; v += THREADS) {
        float x = (float(L[v]) + BIAS[v]) * tInv;
        if (x > local_max) local_max = x;
    }
    threadgroup float mx[THREADS];
    mx[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = THREADS / 2; s > 0; s /= 2) {
        if (lid < s) mx[lid] = max(mx[lid], mx[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float m = mx[0];
    const float cutoff = (minP > 0.0f) ? (m + log(minP)) : -INFINITY;

    // Pass 2: sum_exp over eligible logits (x >= cutoff).
    float local_sum = 0.0f;
    for (uint v = lid; v < VOCAB; v += THREADS) {
        float x = (float(L[v]) + BIAS[v]) * tInv;
        if (x >= cutoff) local_sum += exp(x - m);
    }
    threadgroup float sm[THREADS];
    sm[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = THREADS / 2; s > 0; s /= 2) {
        if (lid < s) sm[lid] += sm[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sumExp = sm[0];

    // Draw one uniform r via philox. Only lid==0 runs; broadcast via
    // threadgroup memory.
    threadgroup float r_shared;
    if (lid == 0) {
        r_shared = philox_uniform(
            sampling_seed[slot], sampling_step[slot], slot);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float target = r_shared * sumExp;

    // Pass 3: inverse-CDF walk. Sequential on lid==0.
    if (lid == 0) {
        float cum = 0.0f;
        uint chosen = VOCAB - 1;
        for (uint v = 0; v < VOCAB; ++v) {
            float x = (float(L[v]) + BIAS[v]) * tInv;
            if (x < cutoff) continue;
            cum += exp(x - m);
            if (cum >= target) { chosen = v; break; }
        }
        input_tokens[slot] = chosen;
    }
}

// ====================================================================
// GPU-side logprob + top-K extraction.
//
// 2026-05-07: replaces the CPU-side captureLogprobForLatestToken
// (ffi_batch.swift) which iterated VOCAB=262144 logits twice per slot
// per CB to compute log-softmax, plus a third pass for top-K via a
// linear-scan replace-min heap. With B=8 streams × ~12 CBs/sec ×
// ~2 ms/call, that was ~192 ms/sec of CPU work directly in the
// gemma_poll outer loop's critical path between AR ticks.
//
// This kernel runs on the GPU after sample_token, reusing the same
// sampling_logit_bias as the sampler. One TG per slot; 32 threads
// parallel-reduce max + sum_exp; each lane maintains a local top-K
// (replace-min over its VOCAB/THREADS = 8192-element slice); finally
// lane 0 selection-sorts the THREADS*K candidate buffer in
// threadgroup memory to write the global top-K out.
//
// Static threadgroup memory: at K_MAX=50:
//   mx[32]:       128 B
//   sm[32]:       128 B
//   cand_lps:     32 * 50 * 4 = 6,400 B
//   cand_ids:     32 * 50 * 4 = 6,400 B
//   total:        ~13 KB  (under 32 KB hardware limit)
//
// Per-lane register pressure: K_MAX × 8 bytes = 400 B (only first K
// of these slots are touched; trailing slots are dead reg pressure).
//
// Skipped per-slot via `capture_active[slot]==0` — slots without
// logprobs=True flag never run the work.
kernel void extract_logprobs(
    device const half*  logits                 [[buffer(0)]],   // [B, VOCAB]
    device const float* sampling_logit_bias    [[buffer(1)]],   // [B, VOCAB]
    device const uint*  sampling_active        [[buffer(2)]],   // [B] 0/1
    device const uchar* capture_active         [[buffer(3)]],   // [B] 0/1
    device const uint*  capture_topk           [[buffer(4)]],   // [B] requested K (0..MAX_TOPK)
    device const uint*  input_tokens           [[buffer(5)]],   // [B] sampled token (written by sample_token)
    device float*       sampled_logprob_out    [[buffer(6)]],   // [B]
    device uint*        topk_ids_out           [[buffer(7)]],   // [B, MAX_TOPK]
    device float*       topk_logprobs_out      [[buffer(8)]],   // [B, MAX_TOPK]
    constant uint&      VOCAB                  [[buffer(9)]],
    constant uint&      MAX_TOPK               [[buffer(10)]],
    uint3 tg_pos                               [[threadgroup_position_in_grid]],
    uint3 lid3                                 [[thread_position_in_threadgroup]])
{
    const uint slot = tg_pos.x;
    const uint lid  = lid3.x;
    constexpr uint THREADS = 32;
    constexpr uint K_MAX   = 50;   // ABI cap — matches captureLogprobForLatestToken's `min(topK, 50)`

    if (sampling_active[slot] == 0 || capture_active[slot] == 0) return;

    device const half*  L    = logits              + slot * VOCAB;
    device const float* BIAS = sampling_logit_bias + slot * VOCAB;

    // Pass 1: max of (logit + bias). Parallel reduce over THREADS.
    float local_max = -INFINITY;
    for (uint v = lid; v < VOCAB; v += THREADS) {
        float x = float(L[v]) + BIAS[v];
        if (x > local_max) local_max = x;
    }
    threadgroup float mx[THREADS];
    mx[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = THREADS / 2; s > 0; s /= 2) {
        if (lid < s) mx[lid] = max(mx[lid], mx[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float m = mx[0];

    // Pass 2: sum_exp of (logit + bias - m). Parallel reduce.
    float local_sum = 0.0f;
    for (uint v = lid; v < VOCAB; v += THREADS) {
        float x = float(L[v]) + BIAS[v];
        local_sum += exp(x - m);
    }
    threadgroup float sm[THREADS];
    sm[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = THREADS / 2; s > 0; s /= 2) {
        if (lid < s) sm[lid] += sm[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float logZ = m + log(sm[0]);

    // Sampled logprob: log_softmax(sampled_token).
    if (lid == 0) {
        const uint sampled = input_tokens[slot];
        sampled_logprob_out[slot] = float(L[sampled]) + BIAS[sampled] - logZ;
    }

    // Top-K extraction.
    const uint K_req = capture_topk[slot];
    const uint K = K_req < K_MAX ? K_req : K_MAX;
    if (K == 0) return;

    // Lane-0-only top-K via threadgroup scratch. Slower than parallel-
    // reduce (~6 ms per slot for VOCAB=262144 vs ~50 µs parallel) but
    // correct and simple. The parallel version had a per-lane stack
    // array that the Metal compiler may have placed in threadgroup
    // memory or spilled in a way that broke the recompute-min logic.
    // If profiling shows top-K dispatch dominates, revisit with a
    // proper parallel-reduce design.
    threadgroup float scratch_lps[K_MAX];
    threadgroup uint  scratch_ids[K_MAX];
    if (lid == 0) {
        for (uint i = 0; i < K_MAX; ++i) {
            scratch_lps[i] = -INFINITY;
            scratch_ids[i] = 0xFFFFFFFFu;
        }
        uint min_idx = 0;
        float min_val = -INFINITY;
        for (uint v = 0; v < VOCAB; ++v) {
            const float lp = float(L[v]) + BIAS[v] - logZ;
            if (lp > min_val) {
                scratch_lps[min_idx] = lp;
                scratch_ids[min_idx] = v;
                // Recompute local min over the K-element scratch.
                float new_min_v = scratch_lps[0];
                uint  new_min_i = 0;
                for (uint i = 1; i < K; ++i) {
                    if (scratch_lps[i] < new_min_v) {
                        new_min_v = scratch_lps[i];
                        new_min_i = i;
                    }
                }
                min_val = new_min_v;
                min_idx = new_min_i;
            }
        }
        // Now sort scratch[0..K) descending so output is top-1, top-2, ...
        for (uint k = 0; k < K; ++k) {
            float best_v = scratch_lps[k];
            uint  best_i = k;
            for (uint i = k + 1; i < K; ++i) {
                if (scratch_lps[i] > best_v) {
                    best_v = scratch_lps[i];
                    best_i = i;
                }
            }
            if (best_i != k) {
                float tlp = scratch_lps[k]; scratch_lps[k] = scratch_lps[best_i]; scratch_lps[best_i] = tlp;
                uint  tid = scratch_ids[k]; scratch_ids[k] = scratch_ids[best_i]; scratch_ids[best_i] = tid;
            }
            topk_ids_out[slot * MAX_TOPK + k]      = scratch_ids[k];
            topk_logprobs_out[slot * MAX_TOPK + k] = scratch_lps[k];
        }
    }
}

// ====================================================================
// Q4_K cooperative matmul (V1) — first iteration of the missing
// simdgroup_float8x8 prefill kernel. Mirrors llama.cpp's
// kernel_mul_mm_q4_K_f32 in shape (cooperative-matrix tiles via
// simdgroup_multiply_accumulate, in-flight Q4_K dequant) but starts
// with smaller, simpler dimensions so we can validate correctness +
// run the A/B tournament against larger tile variants.
//
// V1 dimensions (deliberately conservative — A/B tournament will
// search the larger configurations from this baseline):
//   Output tile: 16 batch rows × 16 output cols
//   K_TILE:      8  (one simdgroup_load worth of K per substep)
//   Threads/TG:  32 (one simdgroup)
//   Accumulators: 4× simdgroup_float8x8 (2×2 grid of the 16×16 output)
//
// Per K_TILE iteration:
//   1. Stage X[16 batch, 8 K] into tg-mem fp16 (128 elements / 32 thr = 4 each)
//   2. Stage W[8 K, 16 outputs] into tg-mem fp16 via per-element Q4_K
//      dequant (128 elements / 32 thr = 4 each — V2 will batch dequants)
//   3. simdgroup_load × 4 (2 X tiles × 2 W tiles)
//   4. simdgroup_multiply_accumulate × 4 (one per accumulator)
//
// Buffer 0: X        [B,    K]    fp16
// Buffer 1: W_q4k    [N * (K/256) * 144 bytes] uchar (Q4_K blob)
// Buffer 2: Y        [B,    N]    fp16
// Buffer 3: B_count  uint
// Buffer 4: D_in (K) uint
// Buffer 5: D_out (N) uint
//
// Grid: (ceil(D_out/16), ceil(B/16)). 32 threads/TG.
// D_in MUST be a multiple of 256 (the Q4_K block size).
kernel void dense_gemm_q4k_mma_v1(
    device const half*  X               [[buffer(0)]],
    device const uchar* W_q4k           [[buffer(1)]],
    device half*        Y               [[buffer(2)]],
    constant uint& B_count              [[buffer(3)]],
    constant uint& D_in                 [[buffer(4)]],
    constant uint& D_out                [[buffer(5)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint2 lid                           [[thread_position_in_threadgroup]])
{
    constexpr uint Q_TILE  = 16;   // batch rows per TG
    constexpr uint O_TILE  = 16;   // output cols per TG
    constexpr uint K_TILE  = 8;    // K elements per simdgroup substep
    constexpr uint THREADS = 32;
    constexpr uint BLK_BYTES = 144;

    const uint o_block = tg.x;
    const uint q_block = tg.y;
    const uint o_start = o_block * O_TILE;
    const uint q_start = q_block * Q_TILE;
    const uint t       = lid.x;

    // tg-mem staging tiles. X laid out as [Q_TILE, K_TILE] (row-major
    // batch × K), W as [K_TILE, O_TILE] (transposed: K × output).
    threadgroup half x_stage[Q_TILE * K_TILE];          // 256 B
    threadgroup half w_stage[K_TILE * O_TILE];          // 256 B
    threadgroup float y_stage[Q_TILE * O_TILE];         // 1024 B

    // 2x2 grid of 8x8 fp32 accumulators covering the 16x16 output tile.
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint blocks_per_row = D_in / 256;             // Q4_K super-blocks per output row

    for (uint k = 0; k < D_in; k += K_TILE) {
        // ── Stage X ──────────────────────────────────────────────
        // x_stage[Q_TILE, K_TILE] = X[q_start..q_start+16, k..k+8]
        // 16*8 = 128 elements, 32 threads, 4 each.
        for (uint i = t; i < Q_TILE * K_TILE; i += THREADS) {
            uint q  = i / K_TILE;
            uint kk = i % K_TILE;
            uint q_abs = q_start + q;
            uint k_abs = k + kk;
            x_stage[i] = (q_abs < B_count && k_abs < D_in)
                ? X[q_abs * D_in + k_abs] : half(0);
        }

        // ── Stage W via in-flight Q4_K dequant ───────────────────
        // w_stage[K_TILE, O_TILE] (transposed — for matmul C = X * W^T
        // we want B-input as W^T; Apple's simdgroup_multiply_accumulate
        // computes acc += A * B where A is [M,K] and B is [K,N], so
        // staging W[K,N] directly is correct).
        // 8*16 = 128 elements, 32 threads, 4 each.
        for (uint i = t; i < K_TILE * O_TILE; i += THREADS) {
            uint kk = i / O_TILE;
            uint oo = i % O_TILE;
            uint o_abs = o_start + oo;
            uint k_abs = k + kk;
            half w_val = half(0);
            if (o_abs < D_out && k_abs < D_in) {
                // Find the right block + element.
                uint kb = k_abs / 256;            // block index along K
                uint elem_in_blk = k_abs % 256;   // element within the block
                uint blk_off = (o_abs * blocks_per_row + kb) * BLK_BYTES;
                device const half*  blk_d    = (device const half*)(W_q4k + blk_off);
                device const half*  blk_dmin = (device const half*)(W_q4k + blk_off + 2);
                device const uchar* scales   = (W_q4k + blk_off + 4);
                device const uchar* qs       = (W_q4k + blk_off + 16);
                w_val = half(dequant_q4k_one(blk_d, blk_dmin, scales, qs, elem_in_blk));
            }
            w_stage[i] = w_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Cooperative matmul ───────────────────────────────────
        // Load 8x8 tiles from tg-mem and accumulate into the 2x2
        // grid of fp32 accumulators. The K dimension of this iter is
        // K_TILE=8 — exactly one simdgroup_matrix multiply.
        simdgroup_half8x8 mx0, mx1;
        simdgroup_half8x8 mw0, mw1;
        // x_stage row-stride = K_TILE = 8, so the second 8 batch rows
        // start at offset Q_TILE/2 * K_TILE = 8 * 8 = 64.
        simdgroup_load(mx0, x_stage,                 K_TILE);   // batch[ 0..7], K[k..k+8]
        simdgroup_load(mx1, x_stage + 8 * K_TILE,    K_TILE);   // batch[ 8..15], K[k..k+8]
        // w_stage row-stride = O_TILE = 16; we want B[K,N] so load
        // 8x8 tiles at column offsets 0 and 8.
        simdgroup_load(mw0, w_stage,                 O_TILE);   // K[k..k+8], output[ 0..7]
        simdgroup_load(mw1, w_stage + 8,             O_TILE);   // K[k..k+8], output[ 8..15]
        simdgroup_multiply_accumulate(acc00, mx0, mw0, acc00);
        simdgroup_multiply_accumulate(acc01, mx0, mw1, acc01);
        simdgroup_multiply_accumulate(acc10, mx1, mw0, acc10);
        simdgroup_multiply_accumulate(acc11, mx1, mw1, acc11);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Store the 16x16 output tile ─────────────────────────────
    // Each accumulator stores into its quadrant of y_stage.
    simdgroup_store(acc00, y_stage,                       O_TILE);  // batch[0..7],  out[0..7]
    simdgroup_store(acc01, y_stage + 8,                   O_TILE);  // batch[0..7],  out[8..15]
    simdgroup_store(acc10, y_stage + 8 * O_TILE,          O_TILE);  // batch[8..15], out[0..7]
    simdgroup_store(acc11, y_stage + 8 * O_TILE + 8,      O_TILE);  // batch[8..15], out[8..15]
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // tg-mem → device, with bounds checks at the tile edge.
    for (uint i = t; i < Q_TILE * O_TILE; i += THREADS) {
        uint q  = i / O_TILE;
        uint oo = i % O_TILE;
        uint q_abs = q_start + q;
        uint o_abs = o_start + oo;
        if (q_abs < B_count && o_abs < D_out) {
            Y[q_abs * D_out + o_abs] = half(y_stage[i]);
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// PREFILL MATMUL — verbatim port of llama.cpp's `kernel_mul_mm` template
// (ggml-metal.metal:9305-9614) specialized for Q8_0 v6-swizzled weights.
// Uses simdgroup_float8x8 cooperative matmul + in-flight Q8_0 dequant.
//
// Replaces the GEMV-shaped Q8_0 dispatchers (encGemvQ80V6 etc.) in the
// prefill path. At prefill batch sizes (B*MAX_Q_LEN tokens per encode) the
// matmul kernel scales 2 → 13 TFLOPS as Q-batch grows from 32 → 1024,
// whereas the GEMV peaks at ~1.5 TFLOPS regardless of batch.
//
// Validated bit-for-bit (RMSE 0.0004 vs CPU ref) in q4k_mma_bench.swift.
// W layout: production v6 swizzle (super-block of 32 output rows
// interleaved by [kb, col, byte]) — single source of truth shared with
// AR decode kernels. See repackQ80ToSwizzled in bootstrap.swift.
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

kernel void prefill_mm_q8_0_swiz(
    device const half*   X            [[buffer(0)]],
    device const uchar*  W_q8         [[buffer(1)]],
    device half*         Y            [[buffer(2)]],
    constant uint& B_count            [[buffer(3)]],
    constant uint& D_in               [[buffer(4)]],
    constant uint& D_out              [[buffer(5)]],
    threadgroup char*    shmem        [[threadgroup(0)]],
    uint3 tgpig                       [[threadgroup_position_in_grid]],
    ushort tiitg                      [[thread_index_in_threadgroup]],
    ushort sgitg                      [[simdgroup_index_in_threadgroup]])
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

    const int   nbc    = ne00 / 32;
    const short row_g  = (short)(r0 + lr0);
    const short ns     = row_g >> 5;
    const short col    = row_g & 31;
    device const block_q8_0_metal * x = (device const block_q8_0_metal *)W_q8
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

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
        // swizzle: kb step = +32 blocks (one super-row stride)
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
// F16 dense prefill — mirror of prefill_mm_q8_0_swiz with the in-flight
// dequant replaced by a direct row-major fp16 read. Same NR0/NR1/NK
// tile geometry, same threadgroup layout, same simdgroup mainloop.
// W layout: plain row-major half [D_out, D_in]. No swizzle (fp16 has no
// blocks to hide). Used for attn QKV/O and ffn_gate/up/down at prefill
// when the tensor is already fp16 in the GGUF.
// ────────────────────────────────────────────────────────────────────
kernel void prefill_mm_f16_swiz(
    device const half*   X            [[buffer(0)]],
    device const half*   W            [[buffer(1)]],
    device half*         Y            [[buffer(2)]],
    constant uint& B_count            [[buffer(3)]],
    constant uint& D_in               [[buffer(4)]],
    constant uint& D_out              [[buffer(5)]],
    threadgroup char*    shmem        [[threadgroup(0)]],
    uint3 tgpig                       [[threadgroup_position_in_grid]],
    ushort tiitg                      [[thread_index_in_threadgroup]],
    ushort sgitg                      [[simdgroup_index_in_threadgroup]])
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

    // F16 has no per-block swizzle; W is plain row-major [ne0, ne00].
    // Each thread owns row (r0+lr0) and reads 16 halves at K offset
    // (loop_k + 16*il) — the same 16-elt sub-tile that dequantize_q8_0
    // produced from the int8 block.
    device const half * x = W + (int)(r0 + lr0) * ne00;

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
            // Direct row-major copy in place of dequantize_q8_0_llama.
            half4x4 temp_a;
            device const half * w_tile = x + loop_k + 16 * (int)il;
            for (short i = 0; i < 16; i++) {
                temp_a[i/4][i%4] = w_tile[i];
            }

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
        // No swizzle pointer-walk for fp16: K advance comes from loop_k.
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
// MoE matmul ports — simdgroup_float8x8 cooperative matmul for Q4_K
// (gate+up) and Q5_1 (down), reading per-expert v6-swizzled weights and
// using our existing routing convention (slot_token, group_start,
// active_experts) instead of llama.cpp's tpe[E]/ids[E*stride] format.
// Output is written in slot-flat order, matching what moe_combine_write
// expects downstream.
//
// Gate+up reads X with broadcast (X[slot_token[s] * K + k]) — input is
// per-token hidden state, fanned to all top-k slots of that token.
// Down reads X per-slot (X[s * K + k]) — input is gelu_mul output that's
// already in slot-flat layout.
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

struct block_q5_1_metal {
    half     d;
    half     m;
    uchar    qh[4];
    uchar    qs[16];
};

static inline void dequantize_q5_1_llama(device const block_q5_1_metal * xb, short il, thread half4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
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

// ── Q5_K block + simdgroup dequant (port of llama.cpp ggml-metal:699) ──
// 176 bytes per 256 weights:
//   half d, half dmin, uchar scales[12], uchar qh[32], uchar qs[128]
struct block_q5K_metal {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qh[32];
    uchar qs[128];
};

static inline void dequantize_q5_K_llama(device const block_q5K_metal * xb, short il, thread half4x4 & reg) {
    device const uchar * q  = xb->qs;
    device const uchar * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q  + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uchar ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.0f;
    const float min_= float(xb->dmin);
    const float dl  = d * sc[0];
    const float ml  = min_ * sc[1];
    const ushort mask  = il < 2 ? 0x0F : 0xF0;
    const float qh_val = il < 2 ? 16.0f : 256.0f;
    for (int i = 0; i < 16; ++i) {
        const float v = (q[i] & mask) + (qh[i] & ul ? qh_val : 0.0f);
        reg[i/4][i%4] = half(dl * v - ml);
    }
}

// ── Q6_K block + simdgroup dequant (port of llama.cpp ggml-metal:722) ──
// 210 bytes per 256 weights:
//   uchar ql[128], uchar qh[64], int8 scales[16], half d
struct block_q6K_metal {
    uchar  ql[128];
    uchar  qh[64];
    int8_t scales[16];
    half   d;
};

static inline void dequantize_q6_K_llama(device const block_q6K_metal * xb, short il, thread half4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t   * scales = xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0u : 0x30303030u)
                                 : (il>0 ? 0x0C0C0C0Cu : 0x03030303u);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
    const float ml  = float(d_all) * sc * 32.0f;
    const float dl0 = float(d_all) * sc;
    const float dl1 = dl0 / 256.0f;
    const float dl2 = dl0 / (256.0f * 256.0f);
    const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
    const uchar shr_h = il>2 ? 2 : 0;
    const uchar shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uchar shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (uint32_t(ql[2*i]) | (uint32_t(ql[2*i+1]) << 16)) & kmask2;
        const uint32_t high = (uint32_t(qh[2*i]) | (uint32_t(qh[2*i+1]) << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = half(dl0 * float(q & 0xFFu)         - ml);
        reg[i][1] = half(dl1 * float(q & 0xFF00u)       - ml);
        reg[i][2] = half(dl2 * float(q & 0xFF0000u)     - ml);
        reg[i][3] = half(dl3 * float(q & 0xFF000000u)   - ml);
    }
}

// ─── MoE Q4_K gate+up: simdgroup matmul, broadcast X via slot_token ─

kernel void prefill_mm_id_q4K_swiz(
    device const half*    X            [[buffer(0)]],   // [N_tokens, K] hidden
    device const uint*    slot_token   [[buffer(1)]],   // [N_slots] → token idx
    device const uchar*   W_q4k        [[buffer(2)]],   // [E, N, K/256] v6 swizzled per-expert
    device const uint*    active_exp   [[buffer(3)]],   // [E] (sentinel = 128 for tail)
    device const uint*    group_start  [[buffer(4)]],   // [E+1] slot ranges by raw expert id
    device half*          Y            [[buffer(5)]],   // [N_slots, N] slot-flat output
    constant uint& D_in                [[buffer(6)]],
    constant uint& D_out               [[buffer(7)]],
    threadgroup char*  shmem           [[threadgroup(0)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    ushort tiitg                       [[thread_index_in_threadgroup]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 16;

    const uint ai = tgpig.z;                    // compact active-expert index
    const uint expert = active_exp[ai];
    if (expert >= 128u) return;                 // sentinel (route_compact tail)

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const uint gb = group_start[expert];
    const uint ge = group_start[expert + 1];
    const int neh1 = (int)(ge - gb);
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;
    const int nbc  = ne00 / 256;
    const int blocks_per_expert = ne0 * nbc;

    const ulong nb12 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    // Per-thread slot + token (broadcast X).
    const uint s_flat = gb + (uint)(r1 + lr1);
    const uint t_idx  = slot_token[s_flat];

    const short offset1 = il0 / nl;

    // Swizzled W base: per expert + ns super-row + kb step + col.
    const short row_g = (short)(r0 + lr0);
    const short ns    = row_g >> 5;
    const short col   = row_g & 31;
    device const block_q4K_metal * x = ((device const block_q4K_metal *)W_q4k)
        + (int)expert * blocks_per_expert
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

    const short iy = 8 * (tiitg % NL1);
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * (ulong)t_idx + sizeof(half) * (ulong)iy);

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
        x  = (il < 2) ? x + 32 : x;          // swizzle: kb advance = +32 blocks
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

    // Slot-flat output write: Y[s_flat, r0..r0+nr0).
    for (short j = sgitg; j < nr1; j += 4) {
        const uint s_out = gb + (uint)(r1 + j);
        device half * D = Y + (int)s_out * ne0 + r0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}

// ─── MoE Q5_1 down: simdgroup matmul, per-slot X (no slot_token lookup) ─

kernel void prefill_mm_id_q5_1_swiz(
    device const half*    X            [[buffer(0)]],   // [N_slots, K] gelu output (per-slot)
    device const uchar*   W_q51        [[buffer(1)]],   // [E, N, K/32] v6 swizzled per-expert
    device const uint*    active_exp   [[buffer(2)]],
    device const uint*    group_start  [[buffer(3)]],
    device half*          Y            [[buffer(4)]],   // [N_slots, N] slot-flat
    constant uint& D_in                [[buffer(5)]],
    constant uint& D_out               [[buffer(6)]],
    threadgroup char*  shmem           [[threadgroup(0)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    ushort tiitg                       [[thread_index_in_threadgroup]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 2;

    const uint ai = tgpig.z;
    const uint expert = active_exp[ai];
    if (expert >= 128u) return;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const uint gb = group_start[expert];
    const uint ge = group_start[expert + 1];
    const int neh1 = (int)(ge - gb);
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;
    const int nbc  = ne00 / 32;
    const int blocks_per_expert = ne0 * nbc;

    const ulong nb12 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const uint s_flat = gb + (uint)(r1 + lr1);

    const short offset1 = il0 / nl;

    const short row_g = (short)(r0 + lr0);
    const short ns    = row_g >> 5;
    const short col   = row_g & 31;
    device const block_q5_1_metal * x = ((device const block_q5_1_metal *)W_q51)
        + (int)expert * blocks_per_expert
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

    const short iy = 8 * (tiitg % NL1);
    // Per-slot X read: directly indexed by s_flat (no slot_token broadcast).
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * (ulong)s_flat + sizeof(half) * (ulong)iy);

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
        const uint s_out = gb + (uint)(r1 + j);
        device half * D = Y + (int)s_out * ne0 + r0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}

// ─── Dense Q5_K prefill matmul: simdgroup matmul, swizzled W ───
// Same skeleton as prefill_mm_q8_0_swiz; differs in block size (256/176B),
// dequant call, and nl=16 (K-quant layout has 16 sub-tiles per super-block).

kernel void prefill_mm_q5_K_swiz(
    device const half*   X            [[buffer(0)]],
    device const uchar*  W_q5k        [[buffer(1)]],
    device half*         Y            [[buffer(2)]],
    constant uint& B_count            [[buffer(3)]],
    constant uint& D_in               [[buffer(4)]],
    constant uint& D_out              [[buffer(5)]],
    threadgroup char*    shmem        [[threadgroup(0)]],
    uint3 tgpig                       [[threadgroup_position_in_grid]],
    ushort tiitg                      [[thread_index_in_threadgroup]],
    ushort sgitg                      [[simdgroup_index_in_threadgroup]])
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

    const int   nbc    = ne00 / 256;
    const short row_g  = (short)(r0 + lr0);
    const short ns     = row_g >> 5;
    const short col    = row_g & 31;
    device const block_q5K_metal * x = (device const block_q5K_metal *)W_q5k
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

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
            dequantize_q5_K_llama(x, il, temp_a);

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

// ─── Dense Q6_K prefill matmul ───

kernel void prefill_mm_q6_K_swiz(
    device const half*   X            [[buffer(0)]],
    device const uchar*  W_q6k        [[buffer(1)]],
    device half*         Y            [[buffer(2)]],
    constant uint& B_count            [[buffer(3)]],
    constant uint& D_in               [[buffer(4)]],
    constant uint& D_out              [[buffer(5)]],
    threadgroup char*    shmem        [[threadgroup(0)]],
    uint3 tgpig                       [[threadgroup_position_in_grid]],
    ushort tiitg                      [[thread_index_in_threadgroup]],
    ushort sgitg                      [[simdgroup_index_in_threadgroup]])
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

    const int   nbc    = ne00 / 256;
    const short row_g  = (short)(r0 + lr0);
    const short ns     = row_g >> 5;
    const short col    = row_g & 31;
    device const block_q6K_metal * x = (device const block_q6K_metal *)W_q6k
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

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
            dequantize_q6_K_llama(x, il, temp_a);

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

// ─── Dense Q5_1 prefill matmul ───
// Used for ffn_down at layers where llama-quantize emits Q5_1 instead of
// Q8_0 in Q5_K_M output. Same skeleton as prefill_mm_q8_0_swiz; differs in
// block size (32-elt/24B), nl=2, and dequantize_q5_1_llama call.

kernel void prefill_mm_q5_1_swiz(
    device const half*   X            [[buffer(0)]],
    device const uchar*  W_q51        [[buffer(1)]],
    device half*         Y            [[buffer(2)]],
    constant uint& B_count            [[buffer(3)]],
    constant uint& D_in               [[buffer(4)]],
    constant uint& D_out              [[buffer(5)]],
    threadgroup char*    shmem        [[threadgroup(0)]],
    uint3 tgpig                       [[threadgroup_position_in_grid]],
    ushort tiitg                      [[thread_index_in_threadgroup]],
    ushort sgitg                      [[simdgroup_index_in_threadgroup]])
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

    const int   nbc    = ne00 / 32;        // 32-elt blocks for Q5_1
    const short row_g  = (short)(r0 + lr0);
    const short ns     = row_g >> 5;
    const short col    = row_g & 31;
    device const block_q5_1_metal * x = (device const block_q5_1_metal *)W_q51
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

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

// ─── MoE Q5_K gate+up: simdgroup matmul, broadcast X via slot_token ─
// Slot-flat adaptation of bench's kernel_mul_mm_id_q5_K_swiz (which used
// llama.cpp tpe/ids). Mirrors prefill_mm_id_q4K_swiz with Q5_K dequant.

kernel void prefill_mm_id_q5_K_swiz(
    device const half*    X            [[buffer(0)]],
    device const uint*    slot_token   [[buffer(1)]],
    device const uchar*   W_q5k        [[buffer(2)]],
    device const uint*    active_exp   [[buffer(3)]],
    device const uint*    group_start  [[buffer(4)]],
    device half*          Y            [[buffer(5)]],
    constant uint& D_in                [[buffer(6)]],
    constant uint& D_out               [[buffer(7)]],
    threadgroup char*  shmem           [[threadgroup(0)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    ushort tiitg                       [[thread_index_in_threadgroup]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 16;

    const uint ai = tgpig.z;
    const uint expert = active_exp[ai];
    if (expert >= 128u) return;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const uint gb = group_start[expert];
    const uint ge = group_start[expert + 1];
    const int neh1 = (int)(ge - gb);
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;
    const int nbc  = ne00 / 256;
    const int blocks_per_expert = ne0 * nbc;

    const ulong nb12 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const uint s_flat = gb + (uint)(r1 + lr1);
    const uint t_idx  = slot_token[s_flat];

    const short offset1 = il0 / nl;

    const short row_g = (short)(r0 + lr0);
    const short ns    = row_g >> 5;
    const short col   = row_g & 31;
    device const block_q5K_metal * x = ((device const block_q5K_metal *)W_q5k)
        + (int)expert * blocks_per_expert
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

    const short iy = 8 * (tiitg % NL1);
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * (ulong)t_idx + sizeof(half) * (ulong)iy);

    simdgroup_half8x8     ma[4];
    simdgroup_half8x8     mb[2];
    simdgroup_float8x8    mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            dequantize_q5_K_llama(x, il, temp_a);
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
        const uint s_out = gb + (uint)(r1 + j);
        device half * D = Y + (int)s_out * ne0 + r0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}

// ─── MoE Q6_K down: simdgroup matmul, per-slot X (no slot_token broadcast) ─
// Slot-flat adaptation; Q6_K serves as the down-projection format in V1.

kernel void prefill_mm_id_q6_K_swiz(
    device const half*    X            [[buffer(0)]],
    device const uchar*   W_q6k        [[buffer(1)]],
    device const uint*    active_exp   [[buffer(2)]],
    device const uint*    group_start  [[buffer(3)]],
    device half*          Y            [[buffer(4)]],
    constant uint& D_in                [[buffer(5)]],
    constant uint& D_out               [[buffer(6)]],
    threadgroup char*  shmem           [[threadgroup(0)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    ushort tiitg                       [[thread_index_in_threadgroup]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 16;

    const uint ai = tgpig.z;
    const uint expert = active_exp[ai];
    if (expert >= 128u) return;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const uint gb = group_start[expert];
    const uint ge = group_start[expert + 1];
    const int neh1 = (int)(ge - gb);
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;
    const int nbc  = ne00 / 256;
    const int blocks_per_expert = ne0 * nbc;

    const ulong nb12 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const uint s_flat = gb + (uint)(r1 + lr1);

    const short offset1 = il0 / nl;

    const short row_g = (short)(r0 + lr0);
    const short ns    = row_g >> 5;
    const short col   = row_g & 31;
    device const block_q6K_metal * x = ((device const block_q6K_metal *)W_q6k)
        + (int)expert * blocks_per_expert
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

    const short iy = 8 * (tiitg % NL1);
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * (ulong)s_flat + sizeof(half) * (ulong)iy);

    simdgroup_half8x8     ma[4];
    simdgroup_half8x8     mb[2];
    simdgroup_float8x8    mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            dequantize_q6_K_llama(x, il, temp_a);
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
        const uint s_out = gb + (uint)(r1 + j);
        device half * D = Y + (int)s_out * ne0 + r0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}

// ─── MoE Q8_0 down: simdgroup matmul, per-slot X (no slot_token broadcast) ─
// Slot-flat adaptation of bench's kernel_mul_mm_id_q8_0_swiz. Used for
// ffn_down_exps when llama-quantize emits Q8_0 (Q5_K_M default).

kernel void prefill_mm_id_q8_0_swiz(
    device const half*    X            [[buffer(0)]],
    device const uchar*   W_q80        [[buffer(1)]],
    device const uint*    active_exp   [[buffer(2)]],
    device const uint*    group_start  [[buffer(3)]],
    device half*          Y            [[buffer(4)]],
    constant uint& D_in                [[buffer(5)]],
    constant uint& D_out               [[buffer(6)]],
    threadgroup char*  shmem           [[threadgroup(0)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    ushort tiitg                       [[thread_index_in_threadgroup]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 2;       // Q8_0 has 2 sub-tiles per 32-elt block

    const uint ai = tgpig.z;
    const uint expert = active_exp[ai];
    if (expert >= 128u) return;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const uint gb = group_start[expert];
    const uint ge = group_start[expert + 1];
    const int neh1 = (int)(ge - gb);
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;
    const int nbc  = ne00 / 32;       // 32-elt super-blocks for Q8_0
    const int blocks_per_expert = ne0 * nbc;

    const ulong nb12 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const uint s_flat = gb + (uint)(r1 + lr1);

    const short offset1 = il0 / nl;

    const short row_g = (short)(r0 + lr0);
    const short ns    = row_g >> 5;
    const short col   = row_g & 31;
    device const block_q8_0_metal * x = ((device const block_q8_0_metal *)W_q80)
        + (int)expert * blocks_per_expert
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

    const short iy = 8 * (tiitg % NL1);
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * (ulong)s_flat + sizeof(half) * (ulong)iy);

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
        const uint s_out = gb + (uint)(r1 + j);
        device half * D = Y + (int)s_out * ne0 + r0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}

// ─── MoE Q8_0 UP prefill: slot_token-broadcast convention ───────────────
// Mirror of prefill_mm_id_q8_0_swiz with the slot_token-broadcast indirection
// from prefill_mm_id_f16_up_swiz spliced in. Used for ffn_gate_up_exps when
// the loaded GGUF puts Q8_0 there — added 2026-05-13 alongside the AR
// moe_gemv_q8_0_v11_up_b{1,2,4,8} family to close the engine's last
// MoE-quant gap.
//
// Buffer layout matches the Q4_K / Q5_K / F16 moe_up prefill exactly:
//   (0) X, (1) slot_token, (2) W, (3) active_exp, (4) group_start,
//   (5) Y, (6) D_in, (7) D_out.
// X is indexed via t_idx = slot_token[s_flat] (broadcast: multiple slots
// referring to the same source token read the same X row), while the
// down kernel above uses s_flat directly (per-slot).
// ─────────────────────────────────────────────────────────────────────────

kernel void prefill_mm_id_q8_0_up_swiz(
    device const half*    X            [[buffer(0)]],
    device const uint*    slot_token   [[buffer(1)]],
    device const uchar*   W_q80        [[buffer(2)]],
    device const uint*    active_exp   [[buffer(3)]],
    device const uint*    group_start  [[buffer(4)]],
    device half*          Y            [[buffer(5)]],
    constant uint& D_in                [[buffer(6)]],
    constant uint& D_out               [[buffer(7)]],
    threadgroup char*  shmem           [[threadgroup(0)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    ushort tiitg                       [[thread_index_in_threadgroup]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 2;

    const uint ai = tgpig.z;
    const uint expert = active_exp[ai];
    if (expert >= 128u) return;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const uint gb = group_start[expert];
    const uint ge = group_start[expert + 1];
    const int neh1 = (int)(ge - gb);
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;
    const int nbc  = ne00 / 32;
    const int blocks_per_expert = ne0 * nbc;

    const ulong nb12 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const uint s_flat = gb + (uint)(r1 + lr1);
    // slot_token-broadcast: lookup the source token for this slot.
    const uint t_idx  = slot_token[s_flat];

    const short offset1 = il0 / nl;

    const short row_g = (short)(r0 + lr0);
    const short ns    = row_g >> 5;
    const short col   = row_g & 31;
    device const block_q8_0_metal * x = ((device const block_q8_0_metal *)W_q80)
        + (int)expert * blocks_per_expert
        + (int)ns * nbc * 32
        + (int)offset1 * 32
        + col;

    const short iy = 8 * (tiitg % NL1);
    // Index X by t_idx, not s_flat (multiple slots may share a token).
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * (ulong)t_idx + sizeof(half) * (ulong)iy);

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
        const uint s_out = gb + (uint)(r1 + j);
        device half * D = Y + (int)s_out * ne0 + r0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}

// ─── MoE F16: simdgroup matmul, per-slot X, plain row-major fp16 W ──────
// Mirror of prefill_mm_id_q8_0_swiz with the in-flight dequant replaced by
// a direct fp16 read. Single kernel covers both call sites
// (ffn_gate_up_exps and ffn_down_exps) because the I/O shape is identical
// once the routing is provided via active_exp / group_start.
//
// W layout: row-major half [E, D_out, D_in]. Per-expert base pointer is
// just expert * D_out * D_in (no swizzled-block math).
kernel void prefill_mm_id_f16_swiz(
    device const half*    X            [[buffer(0)]],
    device const half*    W            [[buffer(1)]],
    device const uint*    active_exp   [[buffer(2)]],
    device const uint*    group_start  [[buffer(3)]],
    device half*          Y            [[buffer(4)]],
    constant uint& D_in                [[buffer(5)]],
    constant uint& D_out               [[buffer(6)]],
    threadgroup char*  shmem           [[threadgroup(0)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    ushort tiitg                       [[thread_index_in_threadgroup]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 2;

    const uint ai = tgpig.z;
    const uint expert = active_exp[ai];
    if (expert >= 128u) return;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const uint gb = group_start[expert];
    const uint ge = group_start[expert + 1];
    const int neh1 = (int)(ge - gb);
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;

    const ulong nb12 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    const uint s_flat = gb + (uint)(r1 + lr1);

    // F16 per-expert base: simple row-major stride product, no block math.
    // Each thread owns row (r0+lr0) of expert `expert`.
    device const half * x = W
        + (ulong)expert * (ulong)ne0 * (ulong)ne00
        + (ulong)(r0 + lr0) * (ulong)ne00;

    const short iy = 8 * (tiitg % NL1);
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * (ulong)s_flat + sizeof(half) * (ulong)iy);

    simdgroup_half8x8     ma[4];
    simdgroup_half8x8     mb[2];
    simdgroup_float8x8    mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        {
            // Direct row-major copy in place of dequantize_q8_0_llama.
            half4x4 temp_a;
            device const half * w_tile = x + loop_k + 16 * (int)il;
            for (short i = 0; i < 16; i++) {
                temp_a[i/4][i%4] = w_tile[i];
            }
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
        // No swizzle pointer-walk for fp16: K advance comes from loop_k.
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
        const uint s_out = gb + (uint)(r1 + j);
        device half * D = Y + (int)s_out * ne0 + r0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}

// ─── MoE F16 UP prefill: slot_token-broadcast convention ────────────────
// Mirror of prefill_mm_id_q4K_swiz (Q4_K is moe_up only, so its kernel
// uses slot_token-broadcast for X). The F16 down kernel above is wrong
// to call from moe_up — its buffer signature lacks slot_token, so the
// dispatcher's index-1 binding (slot_token) gets read as the weight
// matrix pointer and produces structurally-wrong output.
//
// Buffer layout matches Q4_K moe_up exactly:
//   (0) X, (1) slot_token, (2) W, (3) active_exp, (4) group_start,
//   (5) Y, (6) D_in, (7) D_out.
// X is indexed via t_idx = slot_token[s_flat] (broadcast: multiple slots
// referring to the same source token read the same X row).
// ─────────────────────────────────────────────────────────────────────────
kernel void prefill_mm_id_f16_up_swiz(
    device const half*    X            [[buffer(0)]],
    device const uint*    slot_token   [[buffer(1)]],
    device const half*    W            [[buffer(2)]],
    device const uint*    active_exp   [[buffer(3)]],
    device const uint*    group_start  [[buffer(4)]],
    device half*          Y            [[buffer(5)]],
    constant uint& D_in                [[buffer(6)]],
    constant uint& D_out               [[buffer(7)]],
    threadgroup char*  shmem           [[threadgroup(0)]],
    uint3 tgpig                        [[threadgroup_position_in_grid]],
    ushort tiitg                       [[thread_index_in_threadgroup]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = 2;

    const uint ai = tgpig.z;
    const uint expert = active_exp[ai];
    if (expert >= 128u) return;

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const uint gb = group_start[expert];
    const uint ge = group_start[expert + 1];
    const int neh1 = (int)(ge - gb);
    if (r1 >= neh1) return;

    const int ne00 = (int)D_in;
    const int ne0  = (int)D_out;

    const ulong nb12 = (ulong)D_in * sizeof(half);

    const short nr0 = (ne0 - r0 < NR0) ? short(ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : (nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : (nr1 - 1);

    const short il0 = (tiitg % NL0);
    short il = il0;

    // slot_token-broadcast: lookup the source token for this slot.
    const uint s_flat = gb + (uint)(r1 + lr1);
    const uint t_idx  = slot_token[s_flat];

    // F16 per-expert base: simple row-major stride product, no block math.
    device const half * x = W
        + (ulong)expert * (ulong)ne0 * (ulong)ne00
        + (ulong)(r0 + lr0) * (ulong)ne00;

    const short iy = 8 * (tiitg % NL1);
    // Index X by t_idx, not s_flat (multiple slots may share a token).
    device const half * y = (device const half *) ((device const char *)X
        + nb12 * (ulong)t_idx + sizeof(half) * (ulong)iy);

    simdgroup_half8x8     ma[4];
    simdgroup_half8x8     mb[2];
    simdgroup_float8x8    mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            device const half * w_tile = x + loop_k + 16 * (int)il;
            for (short i = 0; i < 16; i++) {
                temp_a[i/4][i%4] = w_tile[i];
            }
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
        const uint s_out = gb + (uint)(r1 + j);
        device half * D = Y + (int)s_out * ne0 + r0;
        threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
        for (int i = tiisg; i < nr0; i += 32) {
            D[i] = (half) C[i];
        }
    }
}
"""
