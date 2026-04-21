import Metal
import Foundation
import ImageIO
import CoreGraphics
import Accelerate

// ===========================================================================
// ForwardGraph demo — one-CB-per-step architecture for Gemma-4-A4B decode.
//
// This is a STRUCTURAL demo: it wires all the kernel types from our
// microbench corpus into a single-CB forward-pass simulator at the correct
// per-layer dispatch density (~17 ops/layer × 30 layers + I/O = ~515 ops).
//
// What's real here:
//   - All PSOs compiled once at setup
//   - Persistent buffers for weights, KV cache, routing, etc.
//   - One CB encoding the whole forward, committed per step
//   - Synthetic but realistically-sized kernels (actual Gemma shapes)
//
// What's not (yet):
//   - Real GGUF weight loading (we use random halves)
//   - Real attention compute (placeholder kernel representing memory cost)
//   - Correctness verification (no CPU reference)
//   - Sampling logic
//
// Goal: measure per-step wall clock at B=4 and compare to our projections.
// ===========================================================================


// device / queue / lib / pso() moved to common.swift so other non-main files
// (vision_tower.swift, future llm_*.swift) can initialize their top-level PSO
// bindings in a defined order.

let rmsPSO          = pso("rms_norm")
let rmsNoScalePSO   = pso("rms_norm_noscale")
let scaleByScalarPSO = pso("scale_by_scalar")
let routerPreNormPSO = pso("router_prenorm_scale")
let addInplacePSO   = pso("add_inplace")
let geluMulPSO      = pso("gelu_mul_inplace")
let moeCombineWritePSO = pso("moe_combine_write")
let gemvV5PSO      = pso("dense_gemv_v5")
let gemvV4PSO      = pso("dense_gemv_v4")
let gemvV4SoftcapPSO = pso("dense_gemv_v4_softcap")
let gemvI8V4PSO    = pso("dense_gemv_i8w_v4")
let ropePSO        = pso("rope_half")
let kvwPSO         = pso("kv_write")
let fakeAttnPSO    = pso("fake_attention")
let pagedSlidePSO  = pso("paged_attn_slide")
let pagedFullPSO   = pso("paged_attn_full")
let topkPSO        = pso("softmax_topk")
let routeCompactPSO = pso("route_compact")
let moePSO         = pso("moe_gemv_v3")
let moeQ4PSO       = pso("moe_gemv_q4_v3")
let moeQ4KPSO      = pso("moe_gemv_q4k_v3")
let moeQ4KV4PSO    = pso("moe_gemv_q4k_v4")
let moeQ51V4PSO    = pso("moe_gemv_q5_1_v4")
let moeQ51V6PSO    = pso("moe_gemv_q5_1_v6")
let moeQ4KV6PSO    = pso("moe_gemv_q4k_v6")
let denseQ4KV4PSO  = pso("dense_gemv_q4k_v4")
let moeQ40PSO      = pso("moe_gemv_q4_0_v3")
let denseQ40V4PSO  = pso("dense_gemv_q4_0_v4")
let denseQ80V5PSO  = pso("dense_gemv_q8_0_v5")
let denseQ80V5RmsPSO = pso("dense_gemv_q8_0_v5_rmsnorm")
let denseQ80V6PSO    = pso("dense_gemv_q8_0_v6")
let denseQ80V6RmsPSO = pso("dense_gemv_q8_0_v6_rmsnorm")
let denseQ80V6RmsQkvPSO = pso("dense_gemv_q8_0_v6_rmsnorm_qkv")
let denseQ80V6RmsGateUpPSO = pso("dense_gemv_q8_0_v6_rmsnorm_gate_up")
let denseQ80V4PSO  = pso("dense_gemv_q8_0_v4")
let moeQ51PSO      = pso("moe_gemv_q5_1_v3")
let moeGeluMulFusedPSO = pso("moe_gelu_mul_fused")
let pagedSlideSplitPSO = pso("paged_attn_slide_split_compute")
let pagedFullSplitPSO  = pso("paged_attn_full_split_compute")
let pagedSplitReducePSO = pso("paged_attn_split_reduce")
let pagedSlideSgmmPSO  = pso("paged_attn_slide_sgmm_compute")
let pagedFullSgmmPSO   = pso("paged_attn_full_sgmm_compute")
let pagedFullGqaPSO    = pso("paged_attn_full_gqa_compute")
let pagedSlideGqaPSO   = pso("paged_attn_slide_gqa_compute")
let flexAttnSlideV0PSO = pso("flex_attn_slide_v0")
let pagedAttnSlideArSharedPSO = pso("paged_attn_slide_ar_shared")
let pagedAttnFullArSharedPSO  = pso("paged_attn_full_ar_shared")
let flexAttnFullV0PSO  = pso("flex_attn_full_v0")
let flexAttnSlideV1Q8PSO = pso("flex_attn_slide_v1_q8")
let flexAttnFullPrefillPSO = pso("flex_attn_full_prefill")
let kvWriteMultiPSO      = pso("kv_write_multi")
let ropeHalfMultiPSO     = pso("rope_half_multi")
let rmsNormAddPSO   = pso("rms_norm_add")
let rmsNormAddScalePSO = pso("rms_norm_add_scale")
let combinePSO     = pso("moe_combine")
let embedPSO       = pso("embed_lookup")
let softcapPSO     = pso("softcap")

// --- Gemma-4-A4B constants ---
let NUM_LAYERS = 30
let HIDDEN = 2816
let VOCAB = 262144
let E_EXP = 128
let TOPK = 8
let MOE_INT = 704
let SHARED_INT = 2112
// Sliding-attention layer params (25 of 30 layers)
let SLIDE_H = 16
let SLIDE_HD = 256
let SLIDE_KV_H = 8
// Full-attention layer params (5 of 30 layers)
let FULL_H = 16
let FULL_HD = 512
let FULL_KV_H = 2

// Unified phys-page pool parameters. At 8192 pages:
//   slide coverage: 128k positions / slot (SW=1024 clamps real usage)
//   full coverage:  64k positions / slot — the binding constraint for
//                    max per-user context until we split slide/full pools.
// KV-cache memory at 8192 pages, Gemma-4 (25 slide + 5 full):
//   slide 25×64KB×8192×2 ≈ 26 GB  +  full 5×16KB×8192×2 ≈ 1.3 GB  = 27 GB
// Fits comfortably alongside the 10 GB Q4_K_M weights on a 128 GB M5.
let MAX_PAGES_PER_SLOT = 8192   // per-session max context (full-cache-bound at 64k tokens)

// Physical page pool must be large enough that every slot gets a disjoint set
// of phys pages, so parallel batch items cannot clobber each other's KV cache.
// Prefix-sharing can intentionally alias pages later, but the default
// disjoint-init depends on this bound.
// (Previous TOTAL_PAGES=256 with B*MAX_PAGES_PER_SLOT=512 wrapped batches 2,3
// onto batches 0,1 — only undetected because our B=4 test replicates the same
// tokens across slots, so writes were idempotent.)

// --- Batch config ---
let B = 4
// At B=4 top_k=8 = 32 slots; uniform routing gives ~32 active experts w/ g=1
let TOTAL_SLOTS = B * TOPK
let ACTIVE_EXPERTS = E_EXP   // worst case — we always launch all 128 TGs (early-return pattern)

// --- Allocate all persistent buffers once ---

func halfBuf(_ n: Int, seed: UInt32 = 0, scale: Float = 0.02) -> MTLBuffer {
    let buf = device.makeBuffer(length: n * 2, options: .storageModeShared)!
    let p = buf.contents().bindMemory(to: Float16.self, capacity: n)
    var s = seed == 0 ? UInt32(n & 0xFFFFFFFF) | 0xdeadbeef : seed
    for i in 0..<n {
        s = s &* 1664525 &+ 1013904223
        let f = Float(Int32(bitPattern: s) % 1000) / 500.0 - 1.0
        p[i] = Float16(f * scale)
    }
    return buf
}
func uintBuf(_ n: Int, mod: Int = 1024, seed: UInt32 = 0) -> MTLBuffer {
    let buf = device.makeBuffer(length: n * 4, options: .storageModeShared)!
    let p = buf.contents().bindMemory(to: UInt32.self, capacity: n)
    var s = seed == 0 ? 0xc0ffee : seed
    for i in 0..<n { s = s &* 1664525 &+ 1013904223; p[i] = UInt32(s) % UInt32(mod) }
    return buf
}
func emptyHalf(_ n: Int) -> MTLBuffer {
    return device.makeBuffer(length: n * 2, options: .storageModeShared)!
}
func emptyUint(_ n: Int) -> MTLBuffer {
    return device.makeBuffer(length: n * 4, options: .storageModeShared)!
}
func emptyFloat(_ n: Int) -> MTLBuffer {
    return device.makeBuffer(length: n * 4, options: .storageModeShared)!
}

// Hidden states: main buffer + pre-norm scratch + two branch outputs + residual save
let hidden        = halfBuf(B * HIDDEN, seed: 0x01)   // carries residual through block
let hidden_norm   = halfBuf(B * HIDDEN, seed: 0x02)   // post-norm scratch
let hidden_resid  = halfBuf(B * HIDDEN, seed: 0x03)   // saved residual before FFN
let mlp_out       = halfBuf(B * HIDDEN, seed: 0x04)   // shared MLP branch output
let moe_sum       = halfBuf(B * HIDDEN, seed: 0x05)   // MoE combined output (post scatter-combine)
let ffn_combined  = halfBuf(B * HIDDEN, seed: 0x06)   // mlp + moe

// Q4_K fused gate+up layout (144 B / 256-elt block). Output dim 1408 = gate(704) || up(704).
let MOE_FUSED_DOUT = MOE_INT * 2

// One-time CPU repack helper: un-swizzled [n, kb, byte] → swizzled [n_super, kb, col, byte].
// Used by the GGUF loader to repack Q8_0 weights into the v6-kernel layout.
func repackQ80ToSwizzled(src: MTLBuffer, dst: MTLBuffer, Din: Int, Dout: Int) {
    let nbc = Din / 32
    let BLK = 34
    let colBytes = nbc * BLK
    let sp = src.contents().assumingMemoryBound(to: UInt8.self)
    let dp = dst.contents().assumingMemoryBound(to: UInt8.self)
    let nSuper = Dout / 32
    for ns in 0..<nSuper {
        let srcColBase = ns * 32 * colBytes
        let dstSuperBase = ns * nbc * 32 * BLK
        for kb in 0..<nbc {
            for col in 0..<32 {
                let srcOff = srcColBase + col * colBytes + kb * BLK
                let dstOff = dstSuperBase + kb * 32 * BLK + col * BLK
                memcpy(dp.advanced(by: dstOff), sp.advanced(by: srcOff), BLK)
            }
        }
    }
}

// Fused gate+up output buffer (total_slots × 2×MOE_INT = 32 × 1408 = 45K halves)
let gate_up_fused = emptyHalf(TOTAL_SLOTS * MOE_FUSED_DOUT)

// Split-KV partials. B*H_Q virtual slots × N_SPLITS partials × (m, l, O_unnorm).
// Shared between sliding and full (different D, so we size for the max = FULL_HD).
let ATTN_N_SPLITS = 16
let NUM_VSLOTS = B * max(SLIDE_H, FULL_H)   // both use 16 Q heads
let m_partials = device.makeBuffer(length: NUM_VSLOTS * ATTN_N_SPLITS * 4, options: .storageModeShared)!
let l_partials = device.makeBuffer(length: NUM_VSLOTS * ATTN_N_SPLITS * 4, options: .storageModeShared)!
let O_partials = device.makeBuffer(length: NUM_VSLOTS * ATTN_N_SPLITS * FULL_HD * 4, options: .storageModeShared)!
let OUT_PROJ_MAX_DIN = max(SLIDE_H * SLIDE_HD, FULL_H * FULL_HD)

// Intermediate buffers (allocated once, reused across every forward step)
let q_slide_out = emptyHalf(B * SLIDE_H * SLIDE_HD)
let k_slide_out = emptyHalf(B * SLIDE_KV_H * SLIDE_HD)
let v_slide_out = emptyHalf(B * SLIDE_KV_H * SLIDE_HD)
let q_full_out  = emptyHalf(B * FULL_H * FULL_HD)
let k_full_out  = emptyHalf(B * FULL_KV_H * FULL_HD)
let v_full_out  = emptyHalf(B * FULL_KV_H * FULL_HD)
let attn_out    = emptyHalf(B * max(SLIDE_H * SLIDE_HD, FULL_H * FULL_HD))
let router_lg   = emptyHalf(B * E_EXP)
let expert_ids  = emptyUint(B * TOPK)
let gate_w      = emptyFloat(B * TOPK)
let gate_proj   = emptyHalf(TOTAL_SLOTS * MOE_INT)
let moe_down_out = emptyHalf(TOTAL_SLOTS * HIDDEN)
let shrd_gate   = emptyHalf(B * SHARED_INT)
let shrd_gate_up_fused = emptyHalf(B * 2 * SHARED_INT)
let logits      = emptyHalf(B * VOCAB)

// Paging. TOTAL_PAGES is per-layer (per-layer K/V caches allocated at GGUF
// load time). 256 pages × PAGE_SLIDE=16 tokens = 4096 slide tokens of
// per-layer capacity; at B=4 disjoint sequences that's ~1024 tokens/seq.
// The scratch strip at the end (SCRATCH_PAGE_BASE..) is not assigned to
// any session; it absorbs KV writes from slots we deliberately silence
// during a single-session prefill. Without this, running prefill for
// slot S would over-write the other slots' active KV at positions
// 0..qLen-1 — which is exactly where earlier sessions started.
// One shared scratch strip is enough because silenced slots all write
// ignored garbage; concurrent writes may race per-cell, but nothing reads
// the result. Cuts KV-cache memory overhead from 2× to 1.25×.
// TOTAL_PAGES is the shared phys-page pool for up to MAX_RESIDENT_SESSIONS
// sessions (not B — sessions and active-slot occupancy are decoupled by
// the scheduler). Per-layer KV cache is sized to this; scratch strip
// sits at the very end for silencing inactive slots during prefill.
//
// With 8192 pages + 256 scratch = 8448 total. Slide K/V at 64 KB/page ×
// 25 layers × 2 ≈ 27 GB. Full K/V tiny in comparison.
let MAX_RESIDENT_SESSIONS = 16                    // logical users held in KV
let SCRATCH_STRIP = 256                           // silenced-slot scratch (shared)
let SCRATCH_PAGE_BASE = 8192                      // pool starts here; scratch at [8192, 8192+256)
let TOTAL_PAGES = SCRATCH_PAGE_BASE + SCRATCH_STRIP
// Load-time assert: if this ever drops below B*MAX_PAGES_PER_SLOT, default
// initLmState's `(b*MAX + p) % TOTAL_PAGES` routing will alias batches.
let PAGE_SLIDE = 16
let PAGE_FULL  = 8      // smaller PAGE at D=512 to fit tg-mem (K+V tile 16 KB at PAGE=8)

// Gemma-4 sliding-window for slide-attention layers (from config.json). The
// slide kernel masks K positions with k_pos < k_len - SLIDING_WINDOW. Set to
// 0 to disable the window (equivalent to global attention).
let SLIDING_WINDOW = 1024

// Per-slot attention metadata, written by CPU per step.
let num_pages_slide = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let num_pages_full  = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let k_len_slide     = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let k_len_full      = device.makeBuffer(length: B * 4, options: .storageModeShared)!

// ========================================================================
// Prefill pipeline — a COMPLETELY SEPARATE forward from autoregressive decode.
// Prefill kernels and buffers are disjoint from AR's; the only shared state
// is GGUF weights + KV cache + block_table. At 32 KB tg-mem/TG on M5 Max,
// parallel-Q attention fits for slide (D=256, Q_BLOCK=8) but not full
// (D=512, Q_BLOCK>1 overflows O_acc at 64 KB). Full prefill serializes Q
// inside the prefill CB — still a single CB for all 30 layers, just not
// fully parallel on the 5/30 full-attn layers.
// ========================================================================
let MAX_Q_LEN = 8

// Prefill scratch buffers sized for B * MAX_Q_LEN "virtual batches".
let pre_hidden       = halfBuf(B * MAX_Q_LEN * HIDDEN, seed: 0x21)
let pre_hidden_norm  = emptyHalf(B * MAX_Q_LEN * HIDDEN)
let pre_mlp_out      = emptyHalf(B * MAX_Q_LEN * HIDDEN)
let pre_moe_sum      = emptyHalf(B * MAX_Q_LEN * HIDDEN)
let pre_ffn_combined = emptyHalf(B * MAX_Q_LEN * HIDDEN)
let pre_q_slide_out  = emptyHalf(B * MAX_Q_LEN * SLIDE_H  * SLIDE_HD)
let pre_k_slide_out  = emptyHalf(B * MAX_Q_LEN * SLIDE_KV_H * SLIDE_HD)
let pre_v_slide_out  = emptyHalf(B * MAX_Q_LEN * SLIDE_KV_H * SLIDE_HD)
let pre_q_full_out   = emptyHalf(B * MAX_Q_LEN * FULL_H  * FULL_HD)
let pre_k_full_out   = emptyHalf(B * MAX_Q_LEN * FULL_KV_H * FULL_HD)
let pre_v_full_out   = emptyHalf(B * MAX_Q_LEN * FULL_KV_H * FULL_HD)
let pre_attn_out     = emptyHalf(B * MAX_Q_LEN * max(SLIDE_H * SLIDE_HD, FULL_H * FULL_HD))
let pre_input_tokens = device.makeBuffer(length: B * MAX_Q_LEN * 4, options: .storageModeShared)!
let pre_q_positions  = device.makeBuffer(length: B * MAX_Q_LEN * 4, options: .storageModeShared)!

// Staging buffers for multi-tile-in-one-CB prefill. CPU populates the wide
// staging buffers with every tile's input in one shot before committing,
// and each tile's per-CB encoding starts with a blit from the staging
// region for THAT tile into the compact working buffers above. This lets
// N tiles share one commit (~100 μs saved per tile) without needing to
// widen every single pre_* residual buffer — the residual buffers stay
// at MAX_Q_LEN capacity and get overwritten tile-by-tile.
//
// MAX_PREFILL_TILES sets the cap on tiles-per-single-CB. A 280-soft-token
// image rounds up to 35 tiles, so 64 is a comfortable headroom.
let MAX_PREFILL_TILES = 64
let MAX_PREFILL_TOKENS = MAX_PREFILL_TILES * MAX_Q_LEN   // 512
let pre_input_tokens_wide = device.makeBuffer(length: B * MAX_PREFILL_TOKENS * 4, options: .storageModeShared)!
let pre_q_positions_wide  = device.makeBuffer(length: B * MAX_PREFILL_TOKENS * 4, options: .storageModeShared)!
// Soft-token staging: fp16 rows ready to copy into pre_hidden slot-s' range.
let pre_hidden_wide       = emptyHalf(B * MAX_PREFILL_TOKENS * HIDDEN)
let pre_logits       = emptyHalf(B * MAX_Q_LEN * VOCAB)

// Prefill MoE state. TOTAL_PREFILL_SLOTS = B * MAX_Q_LEN * TOPK = 256 at Q_LEN=8.
let TOTAL_PREFILL_SLOTS = B * MAX_Q_LEN * TOPK
let pre_expert_ids    = emptyUint(B * MAX_Q_LEN * TOPK)
let pre_gate_w        = device.makeBuffer(length: B * MAX_Q_LEN * TOPK * 4, options: .storageModeShared)!
let pre_slot_token    = emptyUint(TOTAL_PREFILL_SLOTS)
let pre_batch_slots   = emptyUint(B * MAX_Q_LEN * TOPK)
let pre_group_start   = emptyUint(E_EXP + 1)
let pre_router_lg     = emptyHalf(B * MAX_Q_LEN * E_EXP)
let pre_shrd_gate_up_fused = emptyHalf(B * MAX_Q_LEN * 2 * SHARED_INT)
let pre_shrd_gate     = emptyHalf(B * MAX_Q_LEN * SHARED_INT)
let pre_gate_up_fused = emptyHalf(TOTAL_PREFILL_SLOTS * MOE_FUSED_DOUT)
let pre_gate_proj     = emptyHalf(TOTAL_PREFILL_SLOTS * MOE_INT)
let pre_moe_down_out  = emptyHalf(TOTAL_PREFILL_SLOTS * HIDDEN)

// Prefill attention partials, sized [B * MAX_Q_LEN, H_Q, N_SPLITS, D].
let PRE_NUM_VSLOTS = B * MAX_Q_LEN * max(SLIDE_H, FULL_H)
let pre_m_partials = device.makeBuffer(length: PRE_NUM_VSLOTS * ATTN_N_SPLITS * 4, options: .storageModeShared)!
let pre_l_partials = device.makeBuffer(length: PRE_NUM_VSLOTS * ATTN_N_SPLITS * 4, options: .storageModeShared)!
let pre_O_partials = device.makeBuffer(length: PRE_NUM_VSLOTS * ATTN_N_SPLITS * FULL_HD * 4, options: .storageModeShared)!

// Prefill block-mask buffers. At Q_BLOCK=Q_LEN=MAX_Q_LEN=8, q_blocks=1 per slot,
// so CSR offsets are [B+1]. At smaller q_len, still 1 block (padded to Q_BLOCK).
let pre_slide_full_offsets = device.makeBuffer(length: (B + 1) * 4, options: .storageModeShared)!
let pre_slide_full_indices = device.makeBuffer(length: B * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!
let pre_slide_part_offsets = device.makeBuffer(length: (B + 1) * 4, options: .storageModeShared)!
let pre_slide_part_indices = device.makeBuffer(length: B * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!

// Per-partial-block attention mask bitmap. Each partial block is a Q_BLOCK ×
// PAGE mask; we store one uint32 per Q-row (low PAGE bits set = visible).
// Kernel-agnostic: CPU fills these from whatever mask_mod policy the caller
// wants (causal, sliding, doc-isolation, prefix-bidi, arbitrary). The kernel
// just reads "is bit k of row r set" to decide whether to include k in
// softmax or send it to -infinity.
//
// Sized for the worst case: every page is partial. Slide uses PAGE=16 (fits
// in low 16 bits); full uses PAGE=8 (fits in low 8 bits). Q_BLOCK=8 rows.
let FLEX_Q_BLOCK = 8   // compile-time; must match kernel's Q_BLOCK constant
let pre_slide_part_masks = device.makeBuffer(length: B * MAX_PAGES_PER_SLOT * FLEX_Q_BLOCK * 4,
                                               options: .storageModeShared)!
let flex_full_part_masks = device.makeBuffer(length: B * MAX_PAGES_PER_SLOT * FLEX_Q_BLOCK * 4,
                                               options: .storageModeShared)!

let pre_k_len_slide = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let pre_k_len_full  = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let pre_num_pages_slide = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let pre_num_pages_full  = device.makeBuffer(length: B * 4, options: .storageModeShared)!

// Flex-attention block-mask buffers. Sized for the worst case (Q_BLOCK=1,
// so Q_blocks=1 per slot). CSR offsets: [B+1] u32; indices: [B * MAX_PAGES_PER_SLOT] u32.
// Separate pairs for FULL and PARTIAL tiles so the kernel skips the mask
// predicate on the fast path.
let flex_full_offsets    = device.makeBuffer(length: (B + 1) * 4, options: .storageModeShared)!
let flex_full_indices    = device.makeBuffer(length: B * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!
let flex_partial_offsets = device.makeBuffer(length: (B + 1) * 4, options: .storageModeShared)!
let flex_partial_indices = device.makeBuffer(length: B * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!
// Separate buffers for full-attention layer block masks (PAGE=8, no window).
let flex_full_full_offsets = device.makeBuffer(length: (B + 1) * 4, options: .storageModeShared)!
let flex_full_full_indices = device.makeBuffer(length: B * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!
let flex_full_partial_offsets = device.makeBuffer(length: (B + 1) * 4, options: .storageModeShared)!
let flex_full_partial_indices = device.makeBuffer(length: B * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!

// Flex attention is the default as of Phase 1 (bit-for-bit match with legacy
// validated on hellolong + 99-token prose). Set `LEGACY_ATTN=1` to force the
// old paged_attn_{slide,full}_gqa_compute kernels instead (bisection escape).
let USE_FLEX_ATTN = ProcessInfo.processInfo.environment["LEGACY_ATTN"] == nil

// Dynamic routing/control buffers (populated by the forward pass every step).
let positions    = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let block_table  = device.makeBuffer(length: B * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!

// Populated per-step by the scheduler when active slots share a leading
// prefix in their block_tables. `shared_phys_pages[p]` = the (single)
// phys page ID that all active slots agree on for logical page p.
// Read by paged_attn_{slide,full}_ar_shared during AR broadcast.
let shared_phys_pages = device.makeBuffer(length: MAX_PAGES_PER_SLOT * 4,
                                           options: .storageModeShared)!
let active_exp   = device.makeBuffer(length: E_EXP * 4, options: .storageModeShared)!
let group_start  = device.makeBuffer(length: (E_EXP + 1) * 4, options: .storageModeShared)!
let slot_token   = device.makeBuffer(length: TOTAL_SLOTS * 4, options: .storageModeShared)!
let batch_slots  = device.makeBuffer(length: B * TOPK * 4, options: .storageModeShared)!
let input_tokens = device.makeBuffer(length: B * 4, options: .storageModeShared)!

// active_exp stays static at identity [0..E_EXP-1]: route_compact writes
// group_start with prefix counts (empty groups == zero-width), and MoE
// kernels always dispatch E=128 TG rows, each reading active_exp[e_idx]==e_idx
// and early-returning on group_start[e+1]==group_start[e].
// Populated by bootstrapGlobalState() — see below.

// --- Layer dispatch helpers ---

// Generalized RMSNorm: any shape [numVecs, D] with a [D] gamma.
func encRMSNormG(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer, out: MTLBuffer, D: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(rmsPSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(gammaBuf, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
    var Dv = UInt32(D); var eps: Float = 1e-6
    enc.setBytes(&Dv, length: 4, index: 3); enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// RMSNorm with no learnable scale (for v_norm in attention).
func encRMSNormNoScale(_ cb: MTLCommandBuffer, x: MTLBuffer, out: MTLBuffer, D: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(rmsNoScalePSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
    var Dv = UInt32(D); var eps: Float = 1e-6
    enc.setBytes(&Dv, length: 4, index: 2); enc.setBytes(&eps, length: 4, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Per-layer scalar multiply (layer_output_scale applied at end of block).
func encScaleByScalar(_ cb: MTLCommandBuffer, x: MTLBuffer, scalar: MTLBuffer, N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(scaleByScalarPSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(scalar, offset: 0, index: 1)
    var Nv = UInt32(N)
    enc.setBytes(&Nv, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Router pre-norm: RMSNorm_noscale(x) * per_dim_scale * 1/sqrt(D).
func encRouterPreNorm(_ cb: MTLCommandBuffer, x: MTLBuffer, per_dim_scale: MTLBuffer, out: MTLBuffer,
                       numVecs: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(routerPreNormPSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(per_dim_scale, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
    var Dv = UInt32(HIDDEN); var eps: Float = 1e-6
    enc.setBytes(&Dv, length: 4, index: 3); enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Fused norm+add: y = RMSNorm(x, gamma) + residual.
func encRmsNormAdd(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                   residual: MTLBuffer, out: MTLBuffer, N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(rmsNormAddPSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(gammaBuf, offset: 0, index: 1)
    enc.setBuffer(residual, offset: 0, index: 2); enc.setBuffer(out, offset: 0, index: 3)
    var Nv = UInt32(N); var eps: Float = 1e-6
    enc.setBytes(&Nv, length: 4, index: 4); enc.setBytes(&eps, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Fused norm+add+scalar-multiply.
func encRmsNormAddScale(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                        residual: MTLBuffer, scalar: MTLBuffer, out: MTLBuffer,
                        N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(rmsNormAddScalePSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(gammaBuf, offset: 0, index: 1)
    enc.setBuffer(residual, offset: 0, index: 2); enc.setBuffer(scalar, offset: 0, index: 3)
    enc.setBuffer(out, offset: 0, index: 4)
    var Nv = UInt32(N); var eps: Float = 1e-6
    enc.setBytes(&Nv, length: 4, index: 5); enc.setBytes(&eps, length: 4, index: 6)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// In-place residual add: dst += src, [numVecs, N] tensors.
func encAddInplace(_ cb: MTLCommandBuffer, dst: MTLBuffer, src: MTLBuffer, N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addInplacePSO)
    enc.setBuffer(dst, offset: 0, index: 0); enc.setBuffer(src, offset: 0, index: 1)
    var Nv = UInt32(N)
    enc.setBytes(&Nv, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encGemvV5(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer, Din: Int, Dout: Int, numVecs: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(gemvV5PSO)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 3)
    enc.setBytes(&dou, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numVecs, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

func encGemvV4(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer, Din: Int, Dout: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(gemvV4PSO)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var bv = UInt32(B), du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&bv, length: 4, index: 3)
    enc.setBytes(&du, length: 4, index: 4)
    enc.setBytes(&dou, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encGemvV4Softcap(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer,
                       Din: Int, Dout: Int, cap: Float) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(gemvV4SoftcapPSO)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var bv = UInt32(B), du = UInt32(Din), dou = UInt32(Dout), cv = cap
    enc.setBytes(&bv, length: 4, index: 3)
    enc.setBytes(&du, length: 4, index: 4)
    enc.setBytes(&dou, length: 4, index: 5)
    enc.setBytes(&cv, length: 4, index: 6)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encGemvI8V4(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer, Din: Int, Dout: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(gemvI8V4PSO)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(W, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
    var bv = UInt32(B), du = UInt32(Din), dou = UInt32(Dout); var ws: Float = 0.02 / 127.0
    enc.setBytes(&bv, length: 4, index: 3); enc.setBytes(&du, length: 4, index: 4)
    enc.setBytes(&dou, length: 4, index: 5); enc.setBytes(&ws, length: 4, index: 6)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Q8_0 split-K dense GEMV (v5 — for small-to-medium D_out).
func encGemvQ80V5(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq80: MTLBuffer, _ out: MTLBuffer, Din: Int, Dout: Int, numVecs: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseQ80V5PSO)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(Wq80, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 3); enc.setBytes(&dou, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numVecs, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Fused RMSNorm + Q8_0 v5: reads x once, normalizes+scales-by-gamma inside
// TG-mem, then projects. Replaces encRMSNormG + encGemvQ80V5 pair when the
// GEMV is the first consumer of the normed activation.
func encGemvQ80V5Rmsnorm(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                          W: MTLBuffer, out: MTLBuffer, Din: Int, Dout: Int, numVecs: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseQ80V5RmsPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(gammaBuf, offset: 0, index: 2)
    enc.setBuffer(out, offset: 0, index: 3)
    var du = UInt32(Din), dou = UInt32(Dout); var eps: Float = 1e-6
    enc.setBytes(&du, length: 4, index: 4)
    enc.setBytes(&dou, length: 4, index: 5)
    enc.setBytes(&eps, length: 4, index: 6)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numVecs, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Q8_0 swizzled GEMV v6: consumes repacked weights for 32-way coalesced SG reads.
func encGemvQ80V6(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ WswBuf: MTLBuffer,
                   _ out: MTLBuffer, Din: Int, Dout: Int, numVecs: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseQ80V6PSO)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(WswBuf, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 3); enc.setBytes(&dou, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numVecs, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Fused RMSNorm + Q8_0 v6 swizzled.
func encGemvQ80V6Rmsnorm(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                          W: MTLBuffer, out: MTLBuffer, Din: Int, Dout: Int, numVecs: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseQ80V6RmsPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(gammaBuf, offset: 0, index: 2)
    enc.setBuffer(out, offset: 0, index: 3)
    var du = UInt32(Din), dou = UInt32(Dout); var eps: Float = 1e-6
    enc.setBytes(&du, length: 4, index: 4)
    enc.setBytes(&dou, length: 4, index: 5)
    enc.setBytes(&eps, length: 4, index: 6)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numVecs, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Fused RMSNorm + Q8_0 v6 gate+up (shared FFN). Output is [numVecs, 2*D_out] in
// the same layout as the MoE gate_up_fused buffer.
func encGemvQ80V6RmsnormGateUp(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                                Wg: MTLBuffer, Wu: MTLBuffer, fusedOut: MTLBuffer,
                                Din: Int, Dout: Int, numVecs: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseQ80V6RmsGateUpPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(gammaBuf, offset: 0, index: 1)
    enc.setBuffer(Wg, offset: 0, index: 2)
    enc.setBuffer(Wu, offset: 0, index: 3)
    enc.setBuffer(fusedOut, offset: 0, index: 4)
    var du = UInt32(Din), dou = UInt32(Dout); var eps: Float = 1e-6
    enc.setBytes(&du, length: 4, index: 5)
    enc.setBytes(&dou, length: 4, index: 6)
    enc.setBytes(&eps, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: 2 * (Dout / 32), height: numVecs, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Fused RMSNorm + Q8_0 v6 for Q/K/V in a single dispatch.
func encGemvQ80V6RmsnormQKV(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                             Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                             outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                             Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                             numVecs: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseQ80V6RmsQkvPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(gammaBuf, offset: 0, index: 1)
    enc.setBuffer(Wq, offset: 0, index: 2)
    enc.setBuffer(Wk, offset: 0, index: 3)
    enc.setBuffer(Wv, offset: 0, index: 4)
    enc.setBuffer(outQ, offset: 0, index: 5)
    enc.setBuffer(outK, offset: 0, index: 6)
    enc.setBuffer(outV, offset: 0, index: 7)
    var du = UInt32(Din)
    var qnb = UInt32(DoutQ / 32), knb = UInt32(DoutK / 32), vnb = UInt32(DoutV / 32)
    var douq = UInt32(DoutQ), douk = UInt32(DoutK), douv = UInt32(DoutV)
    var eps: Float = 1e-6
    enc.setBytes(&du, length: 4, index: 8)
    enc.setBytes(&qnb, length: 4, index: 9)
    enc.setBytes(&knb, length: 4, index: 10)
    enc.setBytes(&vnb, length: 4, index: 11)
    enc.setBytes(&douq, length: 4, index: 12)
    enc.setBytes(&douk, length: 4, index: 13)
    enc.setBytes(&douv, length: 4, index: 14)
    enc.setBytes(&eps, length: 4, index: 15)
    let totalSlabs = (DoutQ + DoutK + DoutV) / 32
    enc.dispatchThreadgroups(MTLSize(width: totalSlabs, height: numVecs, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Q8_0 multi-batch dense GEMV (v4 — for large D_out like unembed=262144).
func encGemvQ80V4(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq80: MTLBuffer, _ out: MTLBuffer, Din: Int, Dout: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseQ80V4PSO)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(Wq80, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
    var bv = UInt32(B), du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&bv, length: 4, index: 3); enc.setBytes(&du, length: 4, index: 4); enc.setBytes(&dou, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Q5_1 MoE GEMV.
func encMoeGemvQ51(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq51: MTLBuffer, _ out: MTLBuffer,
                    Din: Int, Dout: Int, numActive: Int, useV4: Bool = false, useV6: Bool = false,
                    slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let enc = cb.makeComputeCommandEncoder()!
    let pso = useV6 ? moeQ51V6PSO : (useV4 ? moeQ51V4PSO : moeQ51PSO)
    enc.setComputePipelineState(pso)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(slotTokenBuf ?? slot_token, offset: 0, index: 1)
    enc.setBuffer(Wq51, offset: 0, index: 2); enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(groupStartBuf ?? group_start, offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&dou, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// GELU(gate) * up over the fused [slots, 2*N_half] tensor.
func encMoeGeluMulFused(_ cb: MTLCommandBuffer, fused: MTLBuffer, out: MTLBuffer, N_half: Int, numSlots: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(moeGeluMulFusedPSO)
    enc.setBuffer(fused, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
    var Nv = UInt32(N_half)
    enc.setBytes(&Nv, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: numSlots, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encMoeGemvQ40(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq40: MTLBuffer, _ out: MTLBuffer, Din: Int, Dout: Int, numActive: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(moeQ40PSO)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(slot_token, offset: 0, index: 1)
    enc.setBuffer(Wq40, offset: 0, index: 2); enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(group_start, offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&dou, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encGemvQ40V4(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq40: MTLBuffer, _ out: MTLBuffer, Din: Int, Dout: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseQ40V4PSO)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(Wq40, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
    var bv = UInt32(B), du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&bv, length: 4, index: 3); enc.setBytes(&du, length: 4, index: 4); enc.setBytes(&dou, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// GGUF-native Q4_K MoE GEMV (valid when D_in divisible by 256). Uses v4
// (k-outer / slot-inner) pattern — amortizes Q4_K dequant across slots
// sharing an expert. Falls back to v3 when useV4=false for benchmarking.
func encMoeGemvQ4K(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq4k: MTLBuffer, _ out: MTLBuffer,
                    Din: Int, Dout: Int, numActive: Int, useV4: Bool = false, useV6: Bool = false,
                    slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let enc = cb.makeComputeCommandEncoder()!
    let pso = useV6 ? moeQ4KV6PSO : (useV4 ? moeQ4KV4PSO : moeQ4KPSO)
    enc.setComputePipelineState(pso)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(slotTokenBuf ?? slot_token, offset: 0, index: 1)
    enc.setBuffer(Wq4k, offset: 0, index: 2); enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(groupStartBuf ?? group_start, offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&dou, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encMoeGemvQ4(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq4: MTLBuffer, _ out: MTLBuffer, Din: Int, Dout: Int, numActive: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(moeQ4PSO)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(slot_token, offset: 0, index: 1)
    enc.setBuffer(Wq4, offset: 0, index: 2); enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(group_start, offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
    var du = UInt32(Din), dou = UInt32(Dout); var scale: Float = 0.02
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&dou, length: 4, index: 7)
    enc.setBytes(&scale, length: 4, index: 8)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encRope(_ cb: MTLCommandBuffer, _ x: MTLBuffer, H: Int, D: Int, rotary: Int, theta: Float) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(ropePSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(positions, offset: 0, index: 1)
    var hv = UInt32(H), dv = UInt32(D), rv = UInt32(rotary); var tv = theta
    enc.setBytes(&hv, length: 4, index: 2)
    enc.setBytes(&dv, length: 4, index: 3)
    enc.setBytes(&rv, length: 4, index: 4)
    enc.setBytes(&tv, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: B, height: H, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encKVWrite(_ cb: MTLCommandBuffer, K: MTLBuffer, V: MTLBuffer, Kc: MTLBuffer, Vc: MTLBuffer,
                H: Int, D: Int, page: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(kvwPSO)
    enc.setBuffer(K, offset: 0, index: 0); enc.setBuffer(V, offset: 0, index: 1)
    enc.setBuffer(Kc, offset: 0, index: 2); enc.setBuffer(Vc, offset: 0, index: 3)
    enc.setBuffer(block_table, offset: 0, index: 4); enc.setBuffer(positions, offset: 0, index: 5)
    var hv = UInt32(H), dv = UInt32(D), pv = UInt32(page), mv = UInt32(MAX_PAGES_PER_SLOT)
    enc.setBytes(&hv, length: 4, index: 6); enc.setBytes(&dv, length: 4, index: 7)
    enc.setBytes(&pv, length: 4, index: 8); enc.setBytes(&mv, length: 4, index: 9)
    enc.dispatchThreadgroups(MTLSize(width: B, height: H, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Sliding split-KV with simdgroup_matrix QK/AV (llama.cpp-style). Same
// reduce kernel as the scalar split variant — partials layout is identical.
func encPagedAttnSlideSgmm(_ cb: MTLCommandBuffer, Q: MTLBuffer, O: MTLBuffer,
                            Kc: MTLBuffer, Vc: MTLBuffer,
                            numPagesBuf: MTLBuffer, kLenBuf: MTLBuffer,
                            H_Q: Int, H_KV: Int, D: Int) {
    let enc1 = cb.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(pagedSlideSgmmPSO)
    enc1.setBuffer(Q, offset: 0, index: 0); enc1.setBuffer(Kc, offset: 0, index: 1)
    enc1.setBuffer(Vc, offset: 0, index: 2); enc1.setBuffer(block_table, offset: 0, index: 3)
    enc1.setBuffer(m_partials, offset: 0, index: 4)
    enc1.setBuffer(l_partials, offset: 0, index: 5)
    enc1.setBuffer(O_partials, offset: 0, index: 6)
    enc1.setBuffer(numPagesBuf, offset: 0, index: 7); enc1.setBuffer(kLenBuf, offset: 0, index: 8)
    var scale: Float = 1.0   // Gemma-4: attn.scaling == 1.0; q is already RMS-normed via q_norm
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV), ns = UInt32(ATTN_N_SPLITS)
    enc1.setBytes(&scale, length: 4, index: 9); enc1.setBytes(&mv, length: 4, index: 10)
    enc1.setBytes(&hq, length: 4, index: 11); enc1.setBytes(&hkv, length: 4, index: 12)
    enc1.setBytes(&ns, length: 4, index: 13)
    enc1.dispatchThreadgroups(MTLSize(width: B * H_Q, height: ATTN_N_SPLITS, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc1.endEncoding()

    let enc2 = cb.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(pagedSplitReducePSO)
    enc2.setBuffer(m_partials, offset: 0, index: 0)
    enc2.setBuffer(l_partials, offset: 0, index: 1)
    enc2.setBuffer(O_partials, offset: 0, index: 2)
    enc2.setBuffer(O, offset: 0, index: 3)
    var Dv = UInt32(D)
    enc2.setBytes(&Dv, length: 4, index: 4)
    enc2.setBytes(&ns, length: 4, index: 5)
    enc2.dispatchThreadgroups(MTLSize(width: B * H_Q, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc2.endEncoding()
}

// Sliding GQA split-KV: (B*H_KV, N_SPLITS) grid, Q_PER_TG=2. Halves KV DRAM.
func encPagedAttnSlideGqa(_ cb: MTLCommandBuffer, Q: MTLBuffer, O: MTLBuffer,
                           Kc: MTLBuffer, Vc: MTLBuffer,
                           numPagesBuf: MTLBuffer, kLenBuf: MTLBuffer,
                           H_Q: Int, H_KV: Int, D: Int) {
    let enc1 = cb.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(pagedSlideGqaPSO)
    enc1.setBuffer(Q, offset: 0, index: 0); enc1.setBuffer(Kc, offset: 0, index: 1)
    enc1.setBuffer(Vc, offset: 0, index: 2); enc1.setBuffer(block_table, offset: 0, index: 3)
    enc1.setBuffer(m_partials, offset: 0, index: 4)
    enc1.setBuffer(l_partials, offset: 0, index: 5)
    enc1.setBuffer(O_partials, offset: 0, index: 6)
    enc1.setBuffer(numPagesBuf, offset: 0, index: 7); enc1.setBuffer(kLenBuf, offset: 0, index: 8)
    var scale: Float = 1.0   // Gemma-4: attn.scaling == 1.0; q is already RMS-normed via q_norm
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV), ns = UInt32(ATTN_N_SPLITS)
    var sw = UInt32(SLIDING_WINDOW)
    enc1.setBytes(&scale, length: 4, index: 9); enc1.setBytes(&mv, length: 4, index: 10)
    enc1.setBytes(&hq, length: 4, index: 11); enc1.setBytes(&hkv, length: 4, index: 12)
    enc1.setBytes(&ns, length: 4, index: 13)
    enc1.setBytes(&sw, length: 4, index: 14)
    enc1.dispatchThreadgroups(MTLSize(width: B * H_KV, height: ATTN_N_SPLITS, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc1.endEncoding()

    let enc2 = cb.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(pagedSplitReducePSO)
    enc2.setBuffer(m_partials, offset: 0, index: 0)
    enc2.setBuffer(l_partials, offset: 0, index: 1)
    enc2.setBuffer(O_partials, offset: 0, index: 2)
    enc2.setBuffer(O, offset: 0, index: 3)
    var Dv = UInt32(D)
    enc2.setBytes(&Dv, length: 4, index: 4)
    enc2.setBytes(&ns, length: 4, index: 5)
    enc2.dispatchThreadgroups(MTLSize(width: B * H_Q, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc2.endEncoding()
}

// Host-side block-mask precompute for the flex attention kernel. v0 supports
// Q_BLOCK=1 (decode) with causal_sliding mask_mod. Populates four buffers in
// CSR layout. Classification per k_block B of width PAGE_SLIDE, for query
// position q_pos = k_len - 1 (the most recent token; the only Q row at decode):
//   window_lo  = max(0, k_len - sliding_window)
//   valid k range: [window_lo, q_pos]
//   block FULL     ⟺ B*PAGE >= window_lo  AND  (B+1)*PAGE - 1 <= q_pos
//   block EMPTY    ⟺ (B+1)*PAGE - 1 < window_lo  OR  B*PAGE > q_pos
//   block PARTIAL  otherwise
// sliding_window == 0 disables the lower bound (pure causal).
func precomputeFlexBlockMaskSlide(slidingWindow: Int) {
    let klsP = k_len_slide.contents().assumingMemoryBound(to: UInt32.self)
    let fullOffPtr = flex_full_offsets.contents().assumingMemoryBound(to: UInt32.self)
    let fullIdxPtr = flex_full_indices.contents().assumingMemoryBound(to: UInt32.self)
    let partOffPtr = flex_partial_offsets.contents().assumingMemoryBound(to: UInt32.self)
    let partIdxPtr = flex_partial_indices.contents().assumingMemoryBound(to: UInt32.self)
    var fullCursor = 0
    var partCursor = 0
    fullOffPtr[0] = 0
    partOffPtr[0] = 0
    for b in 0..<B {
        let k_len = Int(klsP[b])
        let q_pos = k_len - 1
        let window_lo = (slidingWindow > 0 && k_len > slidingWindow)
                        ? (k_len - slidingWindow) : 0
        let kBlocks = (k_len + PAGE_SLIDE - 1) / PAGE_SLIDE
        for K in 0..<kBlocks {
            let lo = K * PAGE_SLIDE
            let hi = lo + PAGE_SLIDE - 1
            if hi < window_lo || lo > q_pos {
                // EMPTY — don't emit.
            } else if lo >= window_lo && hi <= q_pos {
                fullIdxPtr[fullCursor] = UInt32(K)
                fullCursor += 1
            } else {
                partIdxPtr[partCursor] = UInt32(K)
                partCursor += 1
            }
        }
        fullOffPtr[b + 1] = UInt32(fullCursor)
        partOffPtr[b + 1] = UInt32(partCursor)
    }
}

// Mask-mod policy. Given an absolute (q_pos, k_pos) and a per-slot context
// blob, return true if attention from q to k should be kept (softmax sees
// the score) or false if it should be masked to -infinity. Called
// Q_BLOCK × PAGE times per partial block at precompute time — NOT on the
// GPU. The kernel only ever reads the resulting bitmap.
//
// Context fields are whatever the policy wants to consume; for Gemma the
// classic ones are k_len (for truncation), slidingWindow (for local attn),
// docIds (for doc-isolation during packed-sequence training-style prefills).
// Adding a new mask type = new case + the bitmap plumbing stays the same.
struct MaskModContext {
    var kLen: Int
    var slidingWindow: Int   // 0 = global; >0 = keep only k >= q+1-window
    // Future: docIds [Int]? for doc-isolation; prefixLen Int? for bidi prefix.
}

enum MaskMod {
    case causal                  // k <= q
    case causalSliding(Int)      // k <= q && k >= q+1-window
    case none                    // always keep (trivial)
    // Future: .docIsolation([Int]), .prefixBidi(prefixLen: Int), ...

    @inline(__always)
    func keep(q: Int, k: Int, ctx: MaskModContext) -> Bool {
        switch self {
        case .causal:
            return k <= q && k < ctx.kLen
        case .causalSliding(let w):
            return k <= q && k < ctx.kLen && (w <= 0 || k + w > q)
        case .none:
            return k < ctx.kLen
        }
    }
}

// Precompute prefill block masks for slide + full attention. For each
// partial block, emit (a) its block index in CSR indices, (b) a Q_BLOCK-row
// bitmap (one uint32 per row) where bit k = 1 iff mask.keep(q_pos, k_pos)
// returns true. FULL blocks get the "all keep" classification; EMPTY blocks
// are skipped entirely.
//
// Kernel is mask-policy-agnostic — it just consumes the bitmap. Supports
// any mask_mod by changing the policy passed to this function.
func precomputeFlexPrefillMasks(qLen: Int, positionStart: Int,
                                 slideMask: MaskMod = .causalSliding(SLIDING_WINDOW),
                                 fullMask: MaskMod = .causal) {
    // Per-slot q positions live in pre_q_positions[b * qLen + i]. We still
    // take a positionStart arg for back-compat with single-slot callers;
    // when multi-slot prefill is active (slots have different positionStarts)
    // each slot's q_first/q_last is read from pre_q_positions instead, so
    // the masks classify FULL/PARTIAL/EMPTY for each slot's own Q range.
    let qBlock = 8
    let qBlocks = (qLen + qBlock - 1) / qBlock
    precondition(qBlocks == 1, "v1 prefill assumes single q_block per slot")
    let qPosP = pre_q_positions.contents().assumingMemoryBound(to: UInt32.self)

    // ---- Slide (PAGE_SLIDE=16) ----
    do {
        let fullOff = pre_slide_full_offsets.contents().assumingMemoryBound(to: UInt32.self)
        let fullIdx = pre_slide_full_indices.contents().assumingMemoryBound(to: UInt32.self)
        let partOff = pre_slide_part_offsets.contents().assumingMemoryBound(to: UInt32.self)
        let partIdx = pre_slide_part_indices.contents().assumingMemoryBound(to: UInt32.self)
        let partMask = pre_slide_part_masks.contents().assumingMemoryBound(to: UInt32.self)
        var fc = 0, pc = 0
        fullOff[0] = 0; partOff[0] = 0
        for b in 0..<B {
            let k_len = Int(pre_k_len_slide.contents().assumingMemoryBound(to: UInt32.self)[b])
            // Read per-slot q positions; fall back to positionStart-based when
            // pre_q_positions[b*qLen] is 0 AND positionStart > 0 (single-slot
            // path that didn't populate silenced slots' positions).
            let slotPos = Int(qPosP[b * qLen])
            let q_first = (slotPos != 0 || positionStart == 0) ? slotPos : positionStart
            let q_last  = q_first + qLen - 1
            let ctx = MaskModContext(kLen: k_len, slidingWindow: SLIDING_WINDOW)
            let kBlocks = (k_len + PAGE_SLIDE - 1) / PAGE_SLIDE
            for K in 0..<kBlocks {
                let lo = K * PAGE_SLIDE
                let hi = lo + PAGE_SLIDE - 1
                // Decide FULL/PARTIAL/EMPTY by sampling corners — safe for
                // monotonic masks like causal + sliding. If the mask_mod is
                // non-monotonic the classifier may over-classify as FULL;
                // fall back to per-cell check when that matters.
                let topLeft     = slideMask.keep(q: q_first, k: lo, ctx: ctx)
                let topRight    = slideMask.keep(q: q_first, k: hi, ctx: ctx)
                let botLeft     = slideMask.keep(q: q_last,  k: lo, ctx: ctx)
                let botRight    = slideMask.keep(q: q_last,  k: hi, ctx: ctx)
                if !(topLeft || topRight || botLeft || botRight) { continue }   // all-empty
                let allKeep = topLeft && topRight && botLeft && botRight
                if allKeep {
                    fullIdx[fc] = UInt32(K); fc += 1
                } else {
                    partIdx[pc] = UInt32(K)
                    // Emit bitmap for this partial block: one uint32 per Q row.
                    for qrow in 0..<qBlock {
                        let q_abs = q_first + qrow
                        var row: UInt32 = 0
                        if q_abs <= q_last {   // don't mask beyond real qLen
                            for kcell in 0..<PAGE_SLIDE {
                                let k_abs = lo + kcell
                                if slideMask.keep(q: q_abs, k: k_abs, ctx: ctx) {
                                    row |= UInt32(1) << kcell
                                }
                            }
                        }
                        partMask[pc * FLEX_Q_BLOCK + qrow] = row
                    }
                    pc += 1
                }
            }
            fullOff[b + 1] = UInt32(fc)
            partOff[b + 1] = UInt32(pc)
        }
    }

    // ---- Full (PAGE_FULL=8, no sliding window) ----
    do {
        let fullOff = flex_full_full_offsets.contents().assumingMemoryBound(to: UInt32.self)
        let fullIdx = flex_full_full_indices.contents().assumingMemoryBound(to: UInt32.self)
        let partOff = flex_full_partial_offsets.contents().assumingMemoryBound(to: UInt32.self)
        let partIdx = flex_full_partial_indices.contents().assumingMemoryBound(to: UInt32.self)
        let partMask = flex_full_part_masks.contents().assumingMemoryBound(to: UInt32.self)
        var fc = 0, pc = 0
        fullOff[0] = 0; partOff[0] = 0
        for b in 0..<B {
            let k_len = Int(pre_k_len_full.contents().assumingMemoryBound(to: UInt32.self)[b])
            let slotPos = Int(qPosP[b * qLen])
            let q_first = (slotPos != 0 || positionStart == 0) ? slotPos : positionStart
            let q_last  = q_first + qLen - 1
            let ctx = MaskModContext(kLen: k_len, slidingWindow: 0)
            let kBlocks = (k_len + PAGE_FULL - 1) / PAGE_FULL
            for K in 0..<kBlocks {
                let lo = K * PAGE_FULL
                let hi = lo + PAGE_FULL - 1
                let topLeft  = fullMask.keep(q: q_first, k: lo, ctx: ctx)
                let topRight = fullMask.keep(q: q_first, k: hi, ctx: ctx)
                let botLeft  = fullMask.keep(q: q_last,  k: lo, ctx: ctx)
                let botRight = fullMask.keep(q: q_last,  k: hi, ctx: ctx)
                if !(topLeft || topRight || botLeft || botRight) { continue }
                let allKeep = topLeft && topRight && botLeft && botRight
                if allKeep {
                    fullIdx[fc] = UInt32(K); fc += 1
                } else {
                    partIdx[pc] = UInt32(K)
                    for qrow in 0..<qBlock {
                        let q_abs = q_first + qrow
                        var row: UInt32 = 0
                        if q_abs <= q_last {
                            for kcell in 0..<PAGE_FULL {
                                let k_abs = lo + kcell
                                if fullMask.keep(q: q_abs, k: k_abs, ctx: ctx) {
                                    row |= UInt32(1) << kcell
                                }
                            }
                        }
                        partMask[pc * FLEX_Q_BLOCK + qrow] = row
                    }
                    pc += 1
                }
            }
            fullOff[b + 1] = UInt32(fc)
            partOff[b + 1] = UInt32(pc)
        }
    }
}

// Prefill attention dispatchers (slide + full). Called from buildPrefillCB;
// dispatch the flex v1 kernels with (B*H_KV or B*H_Q, q_blocks, N_SPLITS).
// Dispatcher for the cross-slot broadcast AR shared-prefix attention.
// Caller supplies:
//   Q:                [B, H_Q, D] — AR-step Qs for every active slot
//   K_cache, V_cache: [phys_pages, PAGE*H_KV, D]
//   sharedPhysPages:  fp32 UInt32[prefix_pages] — one phys-page list used by ALL slots
//   kLenBuf:          UInt32[B] — each slot's k_len (for per-slot SW mask)
//   mPart/lPart/OPart: partials buffers sized [B, H_Q, N_SPLITS, …]; kernel
//                      writes split=0 only. Caller runs a tail kernel into
//                      split=1, then paged_attn_split_reduce to merge.
// N_SPLITS must be 2 (one for shared, one for tail).
func encPagedAttnSlideArShared(_ cb: MTLCommandBuffer,
                                Q: MTLBuffer, Kc: MTLBuffer, Vc: MTLBuffer,
                                sharedPhysPages: MTLBuffer,
                                mPart: MTLBuffer, lPart: MTLBuffer, OPart: MTLBuffer,
                                kLenBuf: MTLBuffer,
                                H_Q: Int, H_KV: Int,
                                prefixPages: Int, slidingWindow: Int, bBatch: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pagedAttnSlideArSharedPSO)
    enc.setBuffer(Q,  offset: 0, index: 0)
    enc.setBuffer(Kc, offset: 0, index: 1)
    enc.setBuffer(Vc, offset: 0, index: 2)
    enc.setBuffer(sharedPhysPages, offset: 0, index: 3)
    enc.setBuffer(mPart, offset: 0, index: 4)
    enc.setBuffer(lPart, offset: 0, index: 5)
    enc.setBuffer(OPart, offset: 0, index: 6)
    enc.setBuffer(kLenBuf, offset: 0, index: 7)
    var sc: Float = 1.0
    var hq = UInt32(H_Q), hkv = UInt32(H_KV), ns = UInt32(2)
    var pp = UInt32(prefixPages), sw = UInt32(slidingWindow), bb = UInt32(bBatch)
    enc.setBytes(&sc, length: 4, index: 8)
    enc.setBytes(&hq, length: 4, index: 9)
    enc.setBytes(&hkv, length: 4, index: 10)
    enc.setBytes(&ns, length: 4, index: 11)
    enc.setBytes(&pp, length: 4, index: 12)
    enc.setBytes(&sw, length: 4, index: 13)
    enc.setBytes(&bb, length: 4, index: 14)
    enc.dispatchThreadgroups(MTLSize(width: H_KV, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Tail-only AR attention — runs flex_attn_slide_v0 with prefix_pages > 0 so
// it skips logical pages already handled by the shared-prefix broadcast
// kernel, and writes its partials at split_offset=1 into a 2-split layout.
// The caller provides the same CSR (flex_full_*/flex_partial_*) as a
// regular v0 call; filtering happens per-block inside the kernel.
func encFlexAttnSlideV0Tail(_ cb: MTLCommandBuffer, Q: MTLBuffer,
                             Kc: MTLBuffer, Vc: MTLBuffer,
                             kLenBuf: MTLBuffer,
                             mPart: MTLBuffer, lPart: MTLBuffer, OPart: MTLBuffer,
                             H_Q: Int, H_KV: Int, D: Int,
                             prefixPages: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(flexAttnSlideV0PSO)
    enc.setBuffer(Q, offset: 0, index: 0); enc.setBuffer(Kc, offset: 0, index: 1)
    enc.setBuffer(Vc, offset: 0, index: 2); enc.setBuffer(block_table, offset: 0, index: 3)
    enc.setBuffer(mPart, offset: 0, index: 4)
    enc.setBuffer(lPart, offset: 0, index: 5)
    enc.setBuffer(OPart, offset: 0, index: 6)
    enc.setBuffer(flex_full_offsets, offset: 0, index: 7)
    enc.setBuffer(flex_full_indices, offset: 0, index: 8)
    enc.setBuffer(flex_partial_offsets, offset: 0, index: 9)
    enc.setBuffer(flex_partial_indices, offset: 0, index: 10)
    enc.setBuffer(kLenBuf, offset: 0, index: 11)
    var scale: Float = 1.0
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV)
    // Internal N_SPLITS=1: the tail kernel doesn't sub-partition its CSR
    // work across TGs in this variant. total_splits_out=2 reserves split=0
    // for the shared-prefix broadcast kernel.
    var ns: UInt32 = 1, sw = UInt32(SLIDING_WINDOW)
    var pp = UInt32(prefixPages), so: UInt32 = 1, tso: UInt32 = 2
    enc.setBytes(&scale, length: 4, index: 12); enc.setBytes(&mv, length: 4, index: 13)
    enc.setBytes(&hq,  length: 4, index: 14); enc.setBytes(&hkv, length: 4, index: 15)
    enc.setBytes(&ns,  length: 4, index: 16); enc.setBytes(&sw,  length: 4, index: 17)
    enc.setBytes(&pp,  length: 4, index: 18); enc.setBytes(&so,  length: 4, index: 19)
    enc.setBytes(&tso, length: 4, index: 20)
    enc.dispatchThreadgroups(MTLSize(width: B * H_KV, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Dispatcher for the full-attention shared-prefix broadcast kernel.
// Grid (H_Q, 1, 1): one TG per q_head, fans out across B slots. Output
// partials at (slot, q_head, split=0). Pair with full_v0_tail writing
// split=1, then paged_attn_split_reduce with N_SPLITS=2.
func encPagedAttnFullArShared(_ cb: MTLCommandBuffer,
                               Q: MTLBuffer, Kc: MTLBuffer, Vc: MTLBuffer,
                               sharedPhysPages: MTLBuffer,
                               mPart: MTLBuffer, lPart: MTLBuffer, OPart: MTLBuffer,
                               kLenBuf: MTLBuffer,
                               H_Q: Int, H_KV: Int,
                               prefixPages: Int, bBatch: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pagedAttnFullArSharedPSO)
    enc.setBuffer(Q,  offset: 0, index: 0)
    enc.setBuffer(Kc, offset: 0, index: 1)
    enc.setBuffer(Vc, offset: 0, index: 2)
    enc.setBuffer(sharedPhysPages, offset: 0, index: 3)
    enc.setBuffer(mPart, offset: 0, index: 4)
    enc.setBuffer(lPart, offset: 0, index: 5)
    enc.setBuffer(OPart, offset: 0, index: 6)
    enc.setBuffer(kLenBuf, offset: 0, index: 7)
    var sc: Float = 1.0
    var hq = UInt32(H_Q), hkv = UInt32(H_KV), ns = UInt32(2)
    var pp = UInt32(prefixPages), bb = UInt32(bBatch)
    enc.setBytes(&sc, length: 4, index: 8)
    enc.setBytes(&hq, length: 4, index: 9)
    enc.setBytes(&hkv, length: 4, index: 10)
    enc.setBytes(&ns, length: 4, index: 11)
    enc.setBytes(&pp, length: 4, index: 12)
    enc.setBytes(&bb, length: 4, index: 13)
    enc.dispatchThreadgroups(MTLSize(width: H_Q, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Tail-only full-attn AR — flex_attn_full_v0 with prefix_pages>0 + split_offset=1.
func encFlexAttnFullV0Tail(_ cb: MTLCommandBuffer, Q: MTLBuffer,
                            Kc: MTLBuffer, Vc: MTLBuffer,
                            kLenBuf: MTLBuffer,
                            mPart: MTLBuffer, lPart: MTLBuffer, OPart: MTLBuffer,
                            H_Q: Int, H_KV: Int, D: Int,
                            prefixPages: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(flexAttnFullV0PSO)
    enc.setBuffer(Q, offset: 0, index: 0); enc.setBuffer(Kc, offset: 0, index: 1)
    enc.setBuffer(Vc, offset: 0, index: 2); enc.setBuffer(block_table, offset: 0, index: 3)
    enc.setBuffer(mPart, offset: 0, index: 4)
    enc.setBuffer(lPart, offset: 0, index: 5)
    enc.setBuffer(OPart, offset: 0, index: 6)
    enc.setBuffer(flex_full_full_offsets, offset: 0, index: 7)
    enc.setBuffer(flex_full_full_indices, offset: 0, index: 8)
    enc.setBuffer(flex_full_partial_offsets, offset: 0, index: 9)
    enc.setBuffer(flex_full_partial_indices, offset: 0, index: 10)
    enc.setBuffer(kLenBuf, offset: 0, index: 11)
    var scale: Float = 1.0
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV)
    var ns: UInt32 = 1
    var pp = UInt32(prefixPages), so: UInt32 = 1, tso: UInt32 = 2
    enc.setBytes(&scale, length: 4, index: 12); enc.setBytes(&mv, length: 4, index: 13)
    enc.setBytes(&hq, length: 4, index: 14); enc.setBytes(&hkv, length: 4, index: 15)
    enc.setBytes(&ns, length: 4, index: 16)
    enc.setBytes(&pp, length: 4, index: 17)
    enc.setBytes(&so, length: 4, index: 18)
    enc.setBytes(&tso, length: 4, index: 19)
    enc.dispatchThreadgroups(MTLSize(width: B * H_KV, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// End-to-end shared+tail+reduce AR attention for the full layer.
func encPagedAttnFullArSharedAndTail(_ cb: MTLCommandBuffer,
                                      Q: MTLBuffer, O: MTLBuffer,
                                      Kc: MTLBuffer, Vc: MTLBuffer,
                                      sharedPhysPages: MTLBuffer,
                                      mPart: MTLBuffer, lPart: MTLBuffer, OPart: MTLBuffer,
                                      kLenBuf: MTLBuffer,
                                      H_Q: Int, H_KV: Int, D: Int,
                                      prefixPages: Int, bBatch: Int) {
    encPagedAttnFullArShared(cb, Q: Q, Kc: Kc, Vc: Vc,
                              sharedPhysPages: sharedPhysPages,
                              mPart: mPart, lPart: lPart, OPart: OPart,
                              kLenBuf: kLenBuf,
                              H_Q: H_Q, H_KV: H_KV,
                              prefixPages: prefixPages, bBatch: bBatch)
    encFlexAttnFullV0Tail(cb, Q: Q, Kc: Kc, Vc: Vc,
                           kLenBuf: kLenBuf,
                           mPart: mPart, lPart: lPart, OPart: OPart,
                           H_Q: H_Q, H_KV: H_KV, D: D,
                           prefixPages: prefixPages)
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pagedSplitReducePSO)
    enc.setBuffer(mPart, offset: 0, index: 0)
    enc.setBuffer(lPart, offset: 0, index: 1)
    enc.setBuffer(OPart, offset: 0, index: 2)
    enc.setBuffer(O,     offset: 0, index: 3)
    var Dv = UInt32(D), ns: UInt32 = 2
    enc.setBytes(&Dv, length: 4, index: 4); enc.setBytes(&ns, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: bBatch * H_Q, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// End-to-end shared+tail+reduce AR attention for the slide layer.
// Caller provides the shared-phys-page list (one per logical page) which
// applies to ALL active slots; each slot's block_table must agree on those
// pages. The tail range is driven by each slot's own block_table + k_len.
func encPagedAttnSlideArSharedAndTail(_ cb: MTLCommandBuffer,
                                       Q: MTLBuffer, O: MTLBuffer,
                                       Kc: MTLBuffer, Vc: MTLBuffer,
                                       sharedPhysPages: MTLBuffer,
                                       mPart: MTLBuffer, lPart: MTLBuffer, OPart: MTLBuffer,
                                       kLenBuf: MTLBuffer,
                                       H_Q: Int, H_KV: Int, D: Int,
                                       prefixPages: Int, slidingWindow: Int, bBatch: Int) {
    // 1. Shared kernel writes split=0.
    encPagedAttnSlideArShared(cb, Q: Q, Kc: Kc, Vc: Vc,
                               sharedPhysPages: sharedPhysPages,
                               mPart: mPart, lPart: lPart, OPart: OPart,
                               kLenBuf: kLenBuf,
                               H_Q: H_Q, H_KV: H_KV,
                               prefixPages: prefixPages,
                               slidingWindow: slidingWindow, bBatch: bBatch)
    // 2. Tail kernel writes split=1.
    encFlexAttnSlideV0Tail(cb, Q: Q, Kc: Kc, Vc: Vc,
                            kLenBuf: kLenBuf,
                            mPart: mPart, lPart: lPart, OPart: OPart,
                            H_Q: H_Q, H_KV: H_KV, D: D,
                            prefixPages: prefixPages)
    // 3. Reduce merges (split=0, split=1) → final O per (slot, q_head).
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pagedSplitReducePSO)
    enc.setBuffer(mPart, offset: 0, index: 0)
    enc.setBuffer(lPart, offset: 0, index: 1)
    enc.setBuffer(OPart, offset: 0, index: 2)
    enc.setBuffer(O, offset: 0, index: 3)
    var Dv = UInt32(D), ns: UInt32 = 2
    enc.setBytes(&Dv, length: 4, index: 4); enc.setBytes(&ns, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: bBatch * H_Q, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encFlexAttnSlidePrefill(_ cb: MTLCommandBuffer, Q: MTLBuffer, O: MTLBuffer,
                              Kc: MTLBuffer, Vc: MTLBuffer,
                              kLenBuf: MTLBuffer, qPositions: MTLBuffer,
                              H_Q: Int, H_KV: Int, D: Int, qLen: Int) {
    let qBlocks = (qLen + 8 - 1) / 8
    let enc1 = cb.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(flexAttnSlideV1Q8PSO)
    enc1.setBuffer(Q, offset: 0, index: 0); enc1.setBuffer(Kc, offset: 0, index: 1)
    enc1.setBuffer(Vc, offset: 0, index: 2); enc1.setBuffer(block_table, offset: 0, index: 3)
    enc1.setBuffer(pre_m_partials, offset: 0, index: 4)
    enc1.setBuffer(pre_l_partials, offset: 0, index: 5)
    enc1.setBuffer(pre_O_partials, offset: 0, index: 6)
    enc1.setBuffer(pre_slide_full_offsets, offset: 0, index: 7)
    enc1.setBuffer(pre_slide_full_indices, offset: 0, index: 8)
    enc1.setBuffer(pre_slide_part_offsets, offset: 0, index: 9)
    enc1.setBuffer(pre_slide_part_indices, offset: 0, index: 10)
    enc1.setBuffer(qPositions, offset: 0, index: 11)
    enc1.setBuffer(kLenBuf, offset: 0, index: 12)
    enc1.setBuffer(pre_slide_part_masks, offset: 0, index: 20)
    var sc: Float = 1.0, mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV)
    var ns = UInt32(ATTN_N_SPLITS), sw = UInt32(SLIDING_WINDOW), ql = UInt32(qLen)
    enc1.setBytes(&sc, length: 4, index: 13); enc1.setBytes(&mv, length: 4, index: 14)
    enc1.setBytes(&hq, length: 4, index: 15); enc1.setBytes(&hkv, length: 4, index: 16)
    enc1.setBytes(&ns, length: 4, index: 17); enc1.setBytes(&sw, length: 4, index: 18)
    enc1.setBytes(&ql, length: 4, index: 19)
    enc1.dispatchThreadgroups(MTLSize(width: B * H_KV, height: qBlocks, depth: ATTN_N_SPLITS),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc1.endEncoding()

    // Reduce per (slot, q_pos, q_head). Grid: B*qLen*H_Q.
    let enc2 = cb.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(pagedSplitReducePSO)
    enc2.setBuffer(pre_m_partials, offset: 0, index: 0)
    enc2.setBuffer(pre_l_partials, offset: 0, index: 1)
    enc2.setBuffer(pre_O_partials, offset: 0, index: 2)
    enc2.setBuffer(O, offset: 0, index: 3)
    var Dv = UInt32(D)
    enc2.setBytes(&Dv, length: 4, index: 4); enc2.setBytes(&ns, length: 4, index: 5)
    enc2.dispatchThreadgroups(MTLSize(width: B * qLen * H_Q, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc2.endEncoding()
}

func encFlexAttnFullPrefill(_ cb: MTLCommandBuffer, Q: MTLBuffer, O: MTLBuffer,
                             Kc: MTLBuffer, Vc: MTLBuffer,
                             kLenBuf: MTLBuffer, qPositions: MTLBuffer,
                             H_Q: Int, H_KV: Int, D: Int, qLen: Int) {
    let qBlocks = (qLen + 8 - 1) / 8
    let enc1 = cb.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(flexAttnFullPrefillPSO)
    enc1.setBuffer(Q, offset: 0, index: 0); enc1.setBuffer(Kc, offset: 0, index: 1)
    enc1.setBuffer(Vc, offset: 0, index: 2); enc1.setBuffer(block_table, offset: 0, index: 3)
    enc1.setBuffer(pre_m_partials, offset: 0, index: 4)
    enc1.setBuffer(pre_l_partials, offset: 0, index: 5)
    enc1.setBuffer(pre_O_partials, offset: 0, index: 6)
    enc1.setBuffer(flex_full_full_offsets, offset: 0, index: 7)
    enc1.setBuffer(flex_full_full_indices, offset: 0, index: 8)
    enc1.setBuffer(flex_full_partial_offsets, offset: 0, index: 9)
    enc1.setBuffer(flex_full_partial_indices, offset: 0, index: 10)
    enc1.setBuffer(qPositions, offset: 0, index: 11)
    enc1.setBuffer(kLenBuf, offset: 0, index: 12)
    enc1.setBuffer(flex_full_part_masks, offset: 0, index: 19)
    var sc: Float = 1.0, mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV)
    var ns = UInt32(ATTN_N_SPLITS), ql = UInt32(qLen)
    enc1.setBytes(&sc, length: 4, index: 13); enc1.setBytes(&mv, length: 4, index: 14)
    enc1.setBytes(&hq, length: 4, index: 15); enc1.setBytes(&hkv, length: 4, index: 16)
    enc1.setBytes(&ns, length: 4, index: 17); enc1.setBytes(&ql, length: 4, index: 18)
    enc1.dispatchThreadgroups(MTLSize(width: B * H_Q, height: qBlocks, depth: ATTN_N_SPLITS),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc1.endEncoding()

    let enc2 = cb.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(pagedSplitReducePSO)
    enc2.setBuffer(pre_m_partials, offset: 0, index: 0)
    enc2.setBuffer(pre_l_partials, offset: 0, index: 1)
    enc2.setBuffer(pre_O_partials, offset: 0, index: 2)
    enc2.setBuffer(O, offset: 0, index: 3)
    var Dv = UInt32(D)
    enc2.setBytes(&Dv, length: 4, index: 4); enc2.setBytes(&ns, length: 4, index: 5)
    enc2.dispatchThreadgroups(MTLSize(width: B * qLen * H_Q, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc2.endEncoding()
}

// Multi-position KV write. K/V layout [B, q_len, H, D]. q_positions [B, q_len].
// Grid: B × q_len × H, 32 lanes each.
func encKVWriteMulti(_ cb: MTLCommandBuffer, K: MTLBuffer, V: MTLBuffer,
                      Kc: MTLBuffer, Vc: MTLBuffer, q_positions: MTLBuffer,
                      H: Int, D: Int, page: Int, qLen: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(kvWriteMultiPSO)
    enc.setBuffer(K, offset: 0, index: 0); enc.setBuffer(V, offset: 0, index: 1)
    enc.setBuffer(Kc, offset: 0, index: 2); enc.setBuffer(Vc, offset: 0, index: 3)
    enc.setBuffer(block_table, offset: 0, index: 4); enc.setBuffer(q_positions, offset: 0, index: 5)
    var hv = UInt32(H), dv = UInt32(D), pv = UInt32(page)
    var mv = UInt32(MAX_PAGES_PER_SLOT), qv = UInt32(qLen)
    enc.setBytes(&hv, length: 4, index: 6); enc.setBytes(&dv, length: 4, index: 7)
    enc.setBytes(&pv, length: 4, index: 8); enc.setBytes(&mv, length: 4, index: 9)
    enc.setBytes(&qv, length: 4, index: 10)
    enc.dispatchThreadgroups(MTLSize(width: B, height: qLen, depth: H),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Multi-position RoPE. x layout [B, q_len, H, D]. q_positions [B, q_len].
func encRopeMulti(_ cb: MTLCommandBuffer, _ x: MTLBuffer, q_positions: MTLBuffer,
                   H: Int, D: Int, rotary: Int, theta: Float, qLen: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(ropeHalfMultiPSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(q_positions, offset: 0, index: 1)
    var hv = UInt32(H), dv = UInt32(D), rv = UInt32(rotary); var tv = theta
    var qv = UInt32(qLen)
    enc.setBytes(&hv, length: 4, index: 2); enc.setBytes(&dv, length: 4, index: 3)
    enc.setBytes(&rv, length: 4, index: 4); enc.setBytes(&tv, length: 4, index: 5)
    enc.setBytes(&qv, length: 4, index: 6)
    enc.dispatchThreadgroups(MTLSize(width: B, height: qLen, depth: H),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Precompute for full-attention (pure causal, no sliding window).
// Same CSR layout as slide but writes into the full-specific buffers and
// uses PAGE_FULL and k_len_full.
func precomputeFlexBlockMaskFull() {
    let klfP = k_len_full.contents().assumingMemoryBound(to: UInt32.self)
    let fullOffPtr = flex_full_full_offsets.contents().assumingMemoryBound(to: UInt32.self)
    let fullIdxPtr = flex_full_full_indices.contents().assumingMemoryBound(to: UInt32.self)
    let partOffPtr = flex_full_partial_offsets.contents().assumingMemoryBound(to: UInt32.self)
    let partIdxPtr = flex_full_partial_indices.contents().assumingMemoryBound(to: UInt32.self)
    var fullCursor = 0
    var partCursor = 0
    fullOffPtr[0] = 0
    partOffPtr[0] = 0
    for b in 0..<B {
        let k_len = Int(klfP[b])
        let q_pos = k_len - 1
        let kBlocks = (k_len + PAGE_FULL - 1) / PAGE_FULL
        for K in 0..<kBlocks {
            let lo = K * PAGE_FULL
            let hi = lo + PAGE_FULL - 1
            if lo > q_pos {
                // EMPTY (past causal horizon).
            } else if hi <= q_pos {
                fullIdxPtr[fullCursor] = UInt32(K); fullCursor += 1
            } else {
                partIdxPtr[partCursor] = UInt32(K); partCursor += 1
            }
        }
        fullOffPtr[b + 1] = UInt32(fullCursor)
        partOffPtr[b + 1] = UInt32(partCursor)
    }
}

// Flex attention dispatcher for full-attn layers (D=512, Q_PER_TG=8, causal).
func encFlexAttnFullV0(_ cb: MTLCommandBuffer, Q: MTLBuffer, O: MTLBuffer,
                        Kc: MTLBuffer, Vc: MTLBuffer,
                        kLenBuf: MTLBuffer,
                        H_Q: Int, H_KV: Int, D: Int) {
    let enc1 = cb.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(flexAttnFullV0PSO)
    enc1.setBuffer(Q, offset: 0, index: 0); enc1.setBuffer(Kc, offset: 0, index: 1)
    enc1.setBuffer(Vc, offset: 0, index: 2); enc1.setBuffer(block_table, offset: 0, index: 3)
    enc1.setBuffer(m_partials, offset: 0, index: 4)
    enc1.setBuffer(l_partials, offset: 0, index: 5)
    enc1.setBuffer(O_partials, offset: 0, index: 6)
    enc1.setBuffer(flex_full_full_offsets, offset: 0, index: 7)
    enc1.setBuffer(flex_full_full_indices, offset: 0, index: 8)
    enc1.setBuffer(flex_full_partial_offsets, offset: 0, index: 9)
    enc1.setBuffer(flex_full_partial_indices, offset: 0, index: 10)
    enc1.setBuffer(kLenBuf, offset: 0, index: 11)
    var scale: Float = 1.0
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV), ns = UInt32(ATTN_N_SPLITS)
    enc1.setBytes(&scale, length: 4, index: 12); enc1.setBytes(&mv, length: 4, index: 13)
    enc1.setBytes(&hq, length: 4, index: 14); enc1.setBytes(&hkv, length: 4, index: 15)
    enc1.setBytes(&ns, length: 4, index: 16)
    // New params (defaults match pre-broadcast behavior).
    var pp: UInt32 = 0, so: UInt32 = 0, tso: UInt32 = 0
    enc1.setBytes(&pp,  length: 4, index: 17)
    enc1.setBytes(&so,  length: 4, index: 18)
    enc1.setBytes(&tso, length: 4, index: 19)
    enc1.dispatchThreadgroups(MTLSize(width: B * H_KV, height: ATTN_N_SPLITS, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc1.endEncoding()

    let enc2 = cb.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(pagedSplitReducePSO)
    enc2.setBuffer(m_partials, offset: 0, index: 0)
    enc2.setBuffer(l_partials, offset: 0, index: 1)
    enc2.setBuffer(O_partials, offset: 0, index: 2)
    enc2.setBuffer(O, offset: 0, index: 3)
    var Dv = UInt32(D)
    enc2.setBytes(&Dv, length: 4, index: 4)
    enc2.setBytes(&ns, length: 4, index: 5)
    enc2.dispatchThreadgroups(MTLSize(width: B * H_Q, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc2.endEncoding()
}

// Flex attention dispatcher (v0: slide D=256, Q_BLOCK=1, causal_sliding).
// Consumes the precomputed full/partial lists and shares the same split-reduce
// kernel as the legacy encPagedAttnSlideGqa so pidx layout is unchanged.
func encFlexAttnSlideV0(_ cb: MTLCommandBuffer, Q: MTLBuffer, O: MTLBuffer,
                         Kc: MTLBuffer, Vc: MTLBuffer,
                         kLenBuf: MTLBuffer,
                         H_Q: Int, H_KV: Int, D: Int) {
    let enc1 = cb.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(flexAttnSlideV0PSO)
    enc1.setBuffer(Q, offset: 0, index: 0); enc1.setBuffer(Kc, offset: 0, index: 1)
    enc1.setBuffer(Vc, offset: 0, index: 2); enc1.setBuffer(block_table, offset: 0, index: 3)
    enc1.setBuffer(m_partials, offset: 0, index: 4)
    enc1.setBuffer(l_partials, offset: 0, index: 5)
    enc1.setBuffer(O_partials, offset: 0, index: 6)
    enc1.setBuffer(flex_full_offsets, offset: 0, index: 7)
    enc1.setBuffer(flex_full_indices, offset: 0, index: 8)
    enc1.setBuffer(flex_partial_offsets, offset: 0, index: 9)
    enc1.setBuffer(flex_partial_indices, offset: 0, index: 10)
    enc1.setBuffer(kLenBuf, offset: 0, index: 11)
    var scale: Float = 1.0   // Gemma-4 attn.scaling
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV), ns = UInt32(ATTN_N_SPLITS)
    var sw = UInt32(SLIDING_WINDOW)
    enc1.setBytes(&scale, length: 4, index: 12); enc1.setBytes(&mv, length: 4, index: 13)
    enc1.setBytes(&hq, length: 4, index: 14); enc1.setBytes(&hkv, length: 4, index: 15)
    enc1.setBytes(&ns, length: 4, index: 16)
    enc1.setBytes(&sw, length: 4, index: 17)
    // New params — zeroed for the default path (no prefix skipping, pidx
    // stride == N_SPLITS so output layout matches pre-broadcast v0 clients).
    var pp: UInt32 = 0, so: UInt32 = 0, tso: UInt32 = 0
    enc1.setBytes(&pp,  length: 4, index: 18)
    enc1.setBytes(&so,  length: 4, index: 19)
    enc1.setBytes(&tso, length: 4, index: 20)
    enc1.dispatchThreadgroups(MTLSize(width: B * H_KV, height: ATTN_N_SPLITS, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc1.endEncoding()

    // Shared reduce kernel: same as paged_attn_split_reduce.
    let enc2 = cb.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(pagedSplitReducePSO)
    enc2.setBuffer(m_partials, offset: 0, index: 0)
    enc2.setBuffer(l_partials, offset: 0, index: 1)
    enc2.setBuffer(O_partials, offset: 0, index: 2)
    enc2.setBuffer(O, offset: 0, index: 3)
    var Dv = UInt32(D)
    enc2.setBytes(&Dv, length: 4, index: 4)
    enc2.setBytes(&ns, length: 4, index: 5)
    enc2.dispatchThreadgroups(MTLSize(width: B * H_Q, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc2.endEncoding()
}

// Full-attn GQA-grouped split-KV. Grid: (B*H_KV, N_SPLITS), one TG handles
// all H_Q/H_KV=8 Q heads that share a kv_head — kills the 8× KV-read
// redundancy and gives QK-sgmm real Q=8 rows. Shared reduce kernel unchanged.
func encPagedAttnFullGqa(_ cb: MTLCommandBuffer, Q: MTLBuffer, O: MTLBuffer,
                          Kc: MTLBuffer, Vc: MTLBuffer,
                          numPagesBuf: MTLBuffer, kLenBuf: MTLBuffer,
                          H_Q: Int, H_KV: Int, D: Int) {
    let enc1 = cb.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(pagedFullGqaPSO)
    enc1.setBuffer(Q, offset: 0, index: 0); enc1.setBuffer(Kc, offset: 0, index: 1)
    enc1.setBuffer(Vc, offset: 0, index: 2); enc1.setBuffer(block_table, offset: 0, index: 3)
    enc1.setBuffer(m_partials, offset: 0, index: 4)
    enc1.setBuffer(l_partials, offset: 0, index: 5)
    enc1.setBuffer(O_partials, offset: 0, index: 6)
    enc1.setBuffer(numPagesBuf, offset: 0, index: 7); enc1.setBuffer(kLenBuf, offset: 0, index: 8)
    var scale: Float = 1.0   // Gemma-4: attn.scaling == 1.0; q is already RMS-normed via q_norm
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV), ns = UInt32(ATTN_N_SPLITS)
    enc1.setBytes(&scale, length: 4, index: 9); enc1.setBytes(&mv, length: 4, index: 10)
    enc1.setBytes(&hq, length: 4, index: 11); enc1.setBytes(&hkv, length: 4, index: 12)
    enc1.setBytes(&ns, length: 4, index: 13)
    enc1.dispatchThreadgroups(MTLSize(width: B * H_KV, height: ATTN_N_SPLITS, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc1.endEncoding()

    let enc2 = cb.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(pagedSplitReducePSO)
    enc2.setBuffer(m_partials, offset: 0, index: 0)
    enc2.setBuffer(l_partials, offset: 0, index: 1)
    enc2.setBuffer(O_partials, offset: 0, index: 2)
    enc2.setBuffer(O, offset: 0, index: 3)
    var Dv = UInt32(D)
    enc2.setBytes(&Dv, length: 4, index: 4)
    enc2.setBytes(&ns, length: 4, index: 5)
    enc2.dispatchThreadgroups(MTLSize(width: B * H_Q, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc2.endEncoding()
}

// Split-KV attention: compute partials + reduce. 4× more parallelism than
// single-TG variant — directly addresses the latency-bound bottleneck on M5.
func encPagedAttnSplit(_ cb: MTLCommandBuffer, Q: MTLBuffer, O: MTLBuffer,
                        Kc: MTLBuffer, Vc: MTLBuffer,
                        numPagesBuf: MTLBuffer, kLenBuf: MTLBuffer,
                        H_Q: Int, H_KV: Int, D: Int, isFull: Bool) {
    // Compute: grid (B*H_Q, N_SPLITS)
    let enc1 = cb.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(isFull ? pagedFullSplitPSO : pagedSlideSplitPSO)
    enc1.setBuffer(Q, offset: 0, index: 0); enc1.setBuffer(Kc, offset: 0, index: 1)
    enc1.setBuffer(Vc, offset: 0, index: 2); enc1.setBuffer(block_table, offset: 0, index: 3)
    enc1.setBuffer(m_partials, offset: 0, index: 4)
    enc1.setBuffer(l_partials, offset: 0, index: 5)
    enc1.setBuffer(O_partials, offset: 0, index: 6)
    enc1.setBuffer(numPagesBuf, offset: 0, index: 7); enc1.setBuffer(kLenBuf, offset: 0, index: 8)
    var scale: Float = 1.0   // Gemma-4: attn.scaling == 1.0; q is already RMS-normed via q_norm
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV), ns = UInt32(ATTN_N_SPLITS)
    enc1.setBytes(&scale, length: 4, index: 9); enc1.setBytes(&mv, length: 4, index: 10)
    enc1.setBytes(&hq, length: 4, index: 11); enc1.setBytes(&hkv, length: 4, index: 12)
    enc1.setBytes(&ns, length: 4, index: 13)
    enc1.dispatchThreadgroups(MTLSize(width: B * H_Q, height: ATTN_N_SPLITS, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc1.endEncoding()

    // Reduce: grid (B*H_Q,)
    let enc2 = cb.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(pagedSplitReducePSO)
    enc2.setBuffer(m_partials, offset: 0, index: 0)
    enc2.setBuffer(l_partials, offset: 0, index: 1)
    enc2.setBuffer(O_partials, offset: 0, index: 2)
    enc2.setBuffer(O, offset: 0, index: 3)
    var Dv = UInt32(D)
    enc2.setBytes(&Dv, length: 4, index: 4)
    enc2.setBytes(&ns, length: 4, index: 5)
    enc2.dispatchThreadgroups(MTLSize(width: B * H_Q, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc2.endEncoding()
}

func encPagedAttn(_ cb: MTLCommandBuffer, Q: MTLBuffer, O: MTLBuffer, Kc: MTLBuffer, Vc: MTLBuffer,
                  numPagesBuf: MTLBuffer, kLenBuf: MTLBuffer,
                  H_Q: Int, H_KV: Int, D: Int, isFull: Bool) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(isFull ? pagedFullPSO : pagedSlidePSO)
    enc.setBuffer(Q, offset: 0, index: 0); enc.setBuffer(Kc, offset: 0, index: 1)
    enc.setBuffer(Vc, offset: 0, index: 2); enc.setBuffer(block_table, offset: 0, index: 3)
    enc.setBuffer(O, offset: 0, index: 4); enc.setBuffer(numPagesBuf, offset: 0, index: 5)
    enc.setBuffer(kLenBuf, offset: 0, index: 6)
    var scale: Float = 1.0   // Gemma-4: attn.scaling == 1.0; q is already RMS-normed via q_norm
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV)
    enc.setBytes(&scale, length: 4, index: 7)
    enc.setBytes(&mv, length: 4, index: 8)
    enc.setBytes(&hq, length: 4, index: 9)
    enc.setBytes(&hkv, length: 4, index: 10)
    // Grid: B × H_Q virtual slots (each TG handles one q-head for one slot)
    enc.dispatchThreadgroups(MTLSize(width: B * H_Q, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encSoftmaxTopk(_ cb: MTLCommandBuffer, expertScaleBuf: MTLBuffer) {
    encSoftmaxTopkInto(cb, logits: router_lg, expertIds: expert_ids, gateW: gate_w,
                        expertScaleBuf: expertScaleBuf, numVecs: B)
}

// Row-generic softmax+topk+renorm+per-expert-scale. Processes numVecs token
// rows of logits[numVecs, E_EXP] into (expert_ids, gate_w)[numVecs, TOPK].
// Underlying MSL kernel uses tg.x as the token row index; it already works
// for numVecs > B — just dispatch wider.
func encSoftmaxTopkInto(_ cb: MTLCommandBuffer, logits: MTLBuffer,
                         expertIds: MTLBuffer, gateW: MTLBuffer,
                         expertScaleBuf: MTLBuffer, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(topkPSO)
    enc.setBuffer(logits, offset: 0, index: 0)
    enc.setBuffer(expertIds, offset: 0, index: 1)
    enc.setBuffer(gateW, offset: 0, index: 2)
    enc.setBuffer(expertScaleBuf, offset: 0, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// MoE routing compaction: reads expert_ids[numVecs*K] written by softmax_topk,
// writes group_start[E+1], slot_token[numVecs*K], batch_slots[numVecs*K].
// Single TG, 128 threads. Must run after softmax_topk and before any MoE GEMV.
// The kernel takes B as a runtime arg, so pass B*qLen for prefill.
func encRouteCompact(_ cb: MTLCommandBuffer) {
    encRouteCompactInto(cb, expertIds: expert_ids, groupStart: group_start,
                         slotToken: slot_token, batchSlots: batch_slots, numVecs: B)
}

func encRouteCompactInto(_ cb: MTLCommandBuffer, expertIds: MTLBuffer,
                          groupStart: MTLBuffer, slotToken: MTLBuffer,
                          batchSlots: MTLBuffer, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(routeCompactPSO)
    enc.setBuffer(expertIds, offset: 0, index: 0)
    enc.setBuffer(groupStart, offset: 0, index: 1)
    enc.setBuffer(slotToken, offset: 0, index: 2)
    enc.setBuffer(batchSlots, offset: 0, index: 3)
    var bv = UInt32(numVecs), kv = UInt32(TOPK)
    enc.setBytes(&bv, length: 4, index: 4)
    enc.setBytes(&kv, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

func encMoeGemv(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer, Din: Int, Dout: Int, numActive: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(moePSO)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(slot_token, offset: 0, index: 1)
    enc.setBuffer(W, offset: 0, index: 2)
    enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(group_start, offset: 0, index: 4)
    enc.setBuffer(out, offset: 0, index: 5)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&dou, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encMoeCombine(_ cb: MTLCommandBuffer) {
    encMoeCombineInto(cb, moeOut: moe_down_out, batchSlots: batch_slots, gateW: gate_w,
                       hiddenInplace: hidden, numVecs: B)
}

// MoE combine writing to a dedicated output buffer (used in the new Gemma-4
// flow where we need the MoE output separate for post_ffw_norm_2).
func encMoeCombineWrite(_ cb: MTLCommandBuffer, to outBuf: MTLBuffer) {
    encMoeCombineWriteInto(cb, moeOut: moe_down_out, batchSlots: batch_slots, gateW: gate_w,
                            outBuf: outBuf, numVecs: B)
}

// Row-generic MoE combine (in-place add to hidden). For prefill pass
// numVecs=B*qLen and use prefill-sized moeOut/batchSlots/gateW buffers.
func encMoeCombineInto(_ cb: MTLCommandBuffer, moeOut: MTLBuffer, batchSlots: MTLBuffer,
                        gateW: MTLBuffer, hiddenInplace: MTLBuffer, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(combinePSO)
    enc.setBuffer(moeOut, offset: 0, index: 0)
    enc.setBuffer(batchSlots, offset: 0, index: 1)
    enc.setBuffer(gateW, offset: 0, index: 2)
    enc.setBuffer(hiddenInplace, offset: 0, index: 3)
    var tk = UInt32(TOPK), dv = UInt32(HIDDEN)
    enc.setBytes(&tk, length: 4, index: 4); enc.setBytes(&dv, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: HIDDEN / 32, height: numVecs, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Row-generic MoE combine writing to a separate output buffer.
func encMoeCombineWriteInto(_ cb: MTLCommandBuffer, moeOut: MTLBuffer, batchSlots: MTLBuffer,
                             gateW: MTLBuffer, outBuf: MTLBuffer, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(moeCombineWritePSO)
    enc.setBuffer(moeOut, offset: 0, index: 0)
    enc.setBuffer(batchSlots, offset: 0, index: 1)
    enc.setBuffer(gateW, offset: 0, index: 2)
    enc.setBuffer(outBuf, offset: 0, index: 3)
    var tk = UInt32(TOPK), dv = UInt32(HIDDEN)
    enc.setBytes(&tk, length: 4, index: 4); enc.setBytes(&dv, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: HIDDEN / 32, height: numVecs, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// GELU×mul (in-place on gate, reads up).
func encGeluMul(_ cb: MTLCommandBuffer, gate: MTLBuffer, up: MTLBuffer, N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(geluMulPSO)
    enc.setBuffer(gate, offset: 0, index: 0); enc.setBuffer(up, offset: 0, index: 1)
    var Nv = UInt32(N)
    enc.setBytes(&Nv, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Blit-copy helper (fast memcpy between buffers via MTLBlitCommandEncoder).
func encBufferCopy(_ cb: MTLCommandBuffer, src: MTLBuffer, dst: MTLBuffer, bytes: Int) {
    let blit = cb.makeBlitCommandEncoder()!
    blit.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: bytes)
    blit.endEncoding()
}

// Row-generic embed lookup into a caller-provided hidden buffer.
func encEmbedInto(_ cb: MTLCommandBuffer, tokens: MTLBuffer, embedTable: MTLBuffer,
                   out: MTLBuffer, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(embedPSO)
    enc.setBuffer(tokens, offset: 0, index: 0)
    enc.setBuffer(embedTable, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var dv = UInt32(HIDDEN)
    enc.setBytes(&dv, length: 4, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encEmbed(_ cb: MTLCommandBuffer, embedTable: MTLBuffer) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(embedPSO)
    enc.setBuffer(input_tokens, offset: 0, index: 0)
    enc.setBuffer(embedTable, offset: 0, index: 1)
    enc.setBuffer(hidden, offset: 0, index: 2)
    var dv = UInt32(HIDDEN)
    enc.setBytes(&dv, length: 4, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encSoftcap(_ cb: MTLCommandBuffer) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(softcapPSO)
    enc.setBuffer(logits, offset: 0, index: 0)
    var nv = UInt32(VOCAB); var cap: Float = 30.0
    enc.setBytes(&nv, length: 4, index: 1); enc.setBytes(&cap, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Row-generic softcap: applies tanh(row[i]/cap)*cap across all N elements
// of each row, numVecs rows total. Used in the prefill path where the
// logits tensor has shape [B*qLen, VOCAB].
func encSoftcapInto(_ cb: MTLCommandBuffer, buf: MTLBuffer, N: Int, numVecs: Int, cap: Float) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(softcapPSO)
    enc.setBuffer(buf, offset: 0, index: 0)
    var nv = UInt32(N); var cv = cap
    enc.setBytes(&nv, length: 4, index: 1); enc.setBytes(&cv, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// --- One step: build CB for full forward, commit, wait ---
//
// Per-layer sequence (HF Gemma-4 text decoder):
//   h = input_layernorm(hidden); q/k/v = Wq/Wk/Wv(h)  [fused triplet]
//   q = q_norm(q); q = RoPE(q); k = k_norm(k); k = RoPE(k); v = v_norm_noscale(v)
//   KV write; attn(q, K_cache, V_cache) @ Wout → mlp_out
//   hidden = post_attn_norm(mlp_out) + hidden
//   shared_ffn = Wdn(gelu(Wgate(pre_ffn_norm(hidden))) * Wup(..))
//   shared_ffn = post_ffn_norm_1(shared_ffn)
//   router: RMSNorm_noscale(hidden)*scale*1/sqrt(D) → Wrouter → softmax_topk → compact
//   moe = MoE(hidden, pre_ffn_norm_2, gate, down, combine)
//   moe = post_ffn_norm_2(moe)
//   ffn = shared_ffn + moe; ffn = post_ffn_norm(ffn)
//   hidden = (ffn + hidden) * layer_output_scale
// Env-var debug toggles for bisecting the forward:
//   LM_SKIP_MOE=1      zero the MoE branch contribution (drops expert output)
//   LM_SKIP_SHARED=1   zero the shared-FFN branch contribution
//   LM_SKIP_ATTN=1     zero the attention output (still writes KV cache)
//   LM_SKIP_SCALE=1    skip the layer_output_scale multiply (use 1.0 instead)
let LM_SKIP_MOE    = ProcessInfo.processInfo.environment["LM_SKIP_MOE"]    != nil
let LM_SKIP_SHARED = ProcessInfo.processInfo.environment["LM_SKIP_SHARED"] != nil
let LM_SKIP_ATTN   = ProcessInfo.processInfo.environment["LM_SKIP_ATTN"]   != nil

// When LM_DUMP_LAYERS=<dir> is set, allocate a staging buffer sized for
// (NUM_LAYERS + 1) * HIDDEN fp16 slot-0 snapshots and blit into it at each
// layer boundary inside buildStepCB. runLmLayerDump reads back after each CB.
//   index 0             = residual after embed + sqrt(hidden) scale (pre layer 0)
//   index L+1 (L=0..29) = residual after decoder_layer[L] (post-ffn norm/resid/scale)
let LM_DUMP_STAGING: MTLBuffer? = {
    guard ProcessInfo.processInfo.environment["LM_DUMP_LAYERS"] != nil else { return nil }
    return device.makeBuffer(length: (NUM_LAYERS + 1) * HIDDEN * 2, options: .storageModeShared)!
}()

// When LM_DUMP_L0_INTERNALS=<dir> is set, allocate staging buffers for
// intra-layer probes captured inside decoder layer 0:
//   slot 0 = hidden after post_attn_norm + residual add (pre_feedforward_layernorm input)
//   slot 1 = mlp_out after post_feedforward_layernorm_1 (shared FFN path output)
//   slot 2 = moe_sum after post_feedforward_layernorm_2 (MoE path output)
//   slot 3 = hidden_norm after pre_feedforward_layernorm_2 (experts' input)
//   slot 4 = moe_sum pre-post_feedforward_layernorm_2 (raw scatter-sum of experts)
// Plus a tiny router staging buffer: expert_ids[K] (uint32) followed by gate_w[K] (float32).
let LM_DUMP_L0_STAGING: MTLBuffer? = {
    guard ProcessInfo.processInfo.environment["LM_DUMP_L0_INTERNALS"] != nil else { return nil }
    return device.makeBuffer(length: 5 * HIDDEN * 2, options: .storageModeShared)!
}()
let LM_DUMP_L0_ROUTER: MTLBuffer? = {
    guard ProcessInfo.processInfo.environment["LM_DUMP_L0_INTERNALS"] != nil else { return nil }
    // Layout: expert_ids (TOPK*u32) + gate_w (TOPK*f32) + router_lg (E_EXP*f16) + hidden_norm (HIDDEN*f16).
    return device.makeBuffer(length: TOPK * 4 + TOPK * 4 + E_EXP * 2 + HIDDEN * 2, options: .storageModeShared)!
}()

// Per-slot MoE intermediates at layer 0 for drilling into the expert-compute
// bug. Layout (in order):
//   moe_down_out      TOTAL_SLOTS * HIDDEN * f16         (post-Q5_1, pre-combine)
//   gate_up_fused     TOTAL_SLOTS * 2*MOE_INT * f16      (post-Q4_K, pre-gelu*up)
//   gate_proj         TOTAL_SLOTS * MOE_INT * f16        (post-gelu*up, pre-Q5_1)
//   slot_token        TOTAL_SLOTS * u32
//   batch_slots       TOTAL_SLOTS * u32
//   group_start       (E_EXP + 1) * u32
// attn_out slot 0 at layer 0 (pre-o_proj), sized for the slide layer
// H_Q*HD = 16*256 = 4096 halves.
let LM_DUMP_L0_ATTN: MTLBuffer? = {
    guard ProcessInfo.processInfo.environment["LM_DUMP_L0_INTERNALS"] != nil else { return nil }
    return device.makeBuffer(length: SLIDE_H * SLIDE_HD * 2, options: .storageModeShared)!
}()

let LM_DUMP_L0_MOE_SLOTS: MTLBuffer? = {
    guard ProcessInfo.processInfo.environment["LM_DUMP_L0_INTERNALS"] != nil else { return nil }
    let size = TOTAL_SLOTS * HIDDEN * 2
        + TOTAL_SLOTS * 2 * MOE_INT * 2
        + TOTAL_SLOTS * MOE_INT * 2
        + TOTAL_SLOTS * 4
        + TOTAL_SLOTS * 4
        + (E_EXP + 1) * 4
    return device.makeBuffer(length: size, options: .storageModeShared)!
}()

// Threshold (in pages) below which broadcast attention isn't worth the
// overhead — per-slot v0 stays faster at tiny shared regions. Env-
// tunable: SHARED_PREFIX_THRESHOLD=<pages>. Default 4 (=64 tokens).
let SHARED_PREFIX_THRESHOLD_PAGES =
    Int(ProcessInfo.processInfo.environment["SHARED_PREFIX_THRESHOLD"] ?? "4") ?? 4

func buildStepCB(_ w: LmWeights, sharedPrefixPages: Int = 0) -> MTLCommandBuffer {
    let cb = queue.makeCommandBuffer()!
    // Decide once per build whether the broadcast path applies. Above
    // threshold, each layer's attention routes through the shared+tail
    // kernel pair; below, we stay on the standard per-slot v0 path.
    let useBroadcast = sharedPrefixPages >= SHARED_PREFIX_THRESHOLD_PAGES

    // Embed lookup + Gemma-4 sqrt(hidden) scale on token embeddings.
    encEmbed(cb, embedTable: w.embedTable)
    encScaleByScalar(cb, x: hidden, scalar: w.embedScaleBuf, N: HIDDEN, numVecs: B)
    if let dump = LM_DUMP_STAGING {
        let blit = cb.makeBlitCommandEncoder()!
        blit.copy(from: hidden, sourceOffset: 0,
                  to: dump, destinationOffset: 0, size: HIDDEN * 2)
        blit.endEncoding()
    }

    for L in 0..<NUM_LAYERS {
        let lw = w.layers[L]
        let isFull = lw.isFull
        let H = isFull ? FULL_H : SLIDE_H
        let KV_H = lw.KV_H
        let HD = lw.HD
        let theta: Float = isFull ? 1_000_000 : 10_000
        let rotary = isFull ? (FULL_HD / 4) : SLIDE_HD
        let q_out = isFull ? q_full_out : q_slide_out
        let k_out = isFull ? k_full_out : k_slide_out
        let v_out = isFull ? v_full_out : v_slide_out
        let Kc = w.K_caches[L]
        let Vc = w.V_caches[L]

        // Fused Q/K/V: one RMSNorm of hidden (with attn_norm gamma), one h_norm
        // TG-mem stage, then route each TG to Q/K/V projection by slab.
        // Gemma-4 full-attn layers omit the V projection in GGUF — pass attnK
        // as Wv so the kernel runs to completion; v_norm_noscale follows.
        let Wv = lw.attnV ?? lw.attnK
        encGemvQ80V6RmsnormQKV(cb, x: hidden, gammaBuf: lw.attnNorm,
                                Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                outQ: q_out, outK: k_out, outV: v_out,
                                Din: HIDDEN,
                                DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD)

        encRMSNormG(cb, x: q_out, gammaBuf: lw.attnQNorm, out: q_out, D: HD, numVecs: B * H)
        encRope(cb, q_out, H: H, D: HD, rotary: rotary, theta: theta)
        encRMSNormG(cb, x: k_out, gammaBuf: lw.attnKNorm, out: k_out, D: HD, numVecs: B * KV_H)
        encRope(cb, k_out, H: KV_H, D: HD, rotary: rotary, theta: theta)
        encRMSNormNoScale(cb, x: v_out, out: v_out, D: HD, numVecs: B * KV_H)

        let pg = isFull ? PAGE_FULL : PAGE_SLIDE
        encKVWrite(cb, K: k_out, V: v_out, Kc: Kc, Vc: Vc, H: KV_H, D: HD, page: pg)

        let npBuf = isFull ? num_pages_full : num_pages_slide
        let klBuf = isFull ? k_len_full : k_len_slide
        if useBroadcast && isFull {
            // Broadcast shared-prefix K/V across all B slots, then per-slot
            // tail kernel writes split=1, split_reduce merges.
            encPagedAttnFullArSharedAndTail(cb, Q: q_out, O: attn_out,
                                             Kc: Kc, Vc: Vc,
                                             sharedPhysPages: shared_phys_pages,
                                             mPart: m_partials,
                                             lPart: l_partials,
                                             OPart: O_partials,
                                             kLenBuf: klBuf,
                                             H_Q: H, H_KV: KV_H, D: HD,
                                             prefixPages: sharedPrefixPages,
                                             bBatch: B)
        } else if useBroadcast {
            encPagedAttnSlideArSharedAndTail(cb, Q: q_out, O: attn_out,
                                              Kc: Kc, Vc: Vc,
                                              sharedPhysPages: shared_phys_pages,
                                              mPart: m_partials,
                                              lPart: l_partials,
                                              OPart: O_partials,
                                              kLenBuf: klBuf,
                                              H_Q: H, H_KV: KV_H, D: HD,
                                              prefixPages: sharedPrefixPages,
                                              slidingWindow: SLIDING_WINDOW,
                                              bBatch: B)
        } else if isFull && USE_FLEX_ATTN {
            encFlexAttnFullV0(cb, Q: q_out, O: attn_out, Kc: Kc, Vc: Vc,
                              kLenBuf: klBuf, H_Q: H, H_KV: KV_H, D: HD)
        } else if isFull {
            encPagedAttnFullGqa(cb, Q: q_out, O: attn_out, Kc: Kc, Vc: Vc,
                                numPagesBuf: npBuf, kLenBuf: klBuf,
                                H_Q: H, H_KV: KV_H, D: HD)
        } else if USE_FLEX_ATTN {
            encFlexAttnSlideV0(cb, Q: q_out, O: attn_out, Kc: Kc, Vc: Vc,
                               kLenBuf: klBuf, H_Q: H, H_KV: KV_H, D: HD)
        } else {
            encPagedAttnSlideGqa(cb, Q: q_out, O: attn_out, Kc: Kc, Vc: Vc,
                                 numPagesBuf: npBuf, kLenBuf: klBuf,
                                 H_Q: H, H_KV: KV_H, D: HD)
        }
        if LM_SKIP_ATTN {
            let blit = cb.makeBlitCommandEncoder()!
            blit.fill(buffer: attn_out, range: 0..<(B * max(SLIDE_H * SLIDE_HD, FULL_H * FULL_HD) * 2), value: 0)
            blit.endEncoding()
        }

        // Out projection → mlp_out; fused post-attn norm + residual add on `hidden`.
        if L == 0, let attnDump = LM_DUMP_L0_ATTN {
            // Capture slot-0 attn_out (pre-o_proj) so we can isolate softmax+AV
            // from o_proj and post-attn norm.
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: attn_out, sourceOffset: 0,
                      to: attnDump, destinationOffset: 0, size: H * HD * 2)
            blit.endEncoding()
        }
        encGemvQ80V6(cb, attn_out, lw.attnOut, mlp_out, Din: H * HD, Dout: HIDDEN)
        encRmsNormAdd(cb, x: mlp_out, gammaBuf: lw.postAttnNorm,
                      residual: hidden, out: hidden, N: HIDDEN, numVecs: B)
        if L == 0, let dump = LM_DUMP_L0_STAGING {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: hidden, sourceOffset: 0,
                      to: dump, destinationOffset: 0, size: HIDDEN * 2)
            blit.endEncoding()
        }

        // Shared MLP branch: fused (rmsnorm + gate + up) → gelu*mul → down → post_ffn_1.
        encGemvQ80V6RmsnormGateUp(cb, x: hidden, gammaBuf: lw.ffnNorm,
                                   Wg: lw.ffnGate, Wu: lw.ffnUp,
                                   fusedOut: shrd_gate_up_fused,
                                   Din: HIDDEN, Dout: SHARED_INT)
        encMoeGeluMulFused(cb, fused: shrd_gate_up_fused, out: shrd_gate, N_half: SHARED_INT, numSlots: B)
        encGemvQ80V6(cb, shrd_gate, lw.ffnDown, mlp_out, Din: SHARED_INT, Dout: HIDDEN)
        encRMSNormG(cb, x: mlp_out, gammaBuf: lw.postFfn1Norm, out: mlp_out, D: HIDDEN, numVecs: B)
        if L == 0, let dump = LM_DUMP_L0_STAGING {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: mlp_out, sourceOffset: 0,
                      to: dump, destinationOffset: 1 * HIDDEN * 2, size: HIDDEN * 2)
            blit.endEncoding()
        }

        // Router: pre-norm with per-dim scale and 1/sqrt(D) divisor, project to
        // logits, softmax+topk+renorm*per_expert_scale, then compact for MoE.
        encRouterPreNorm(cb, x: hidden, per_dim_scale: lw.routerScale, out: hidden_norm)
        if L == 0, let routerDump = LM_DUMP_L0_ROUTER {
            // hidden_norm (slot 0) f16[HIDDEN] — capture BEFORE the router proj.
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: hidden_norm, sourceOffset: 0,
                      to: routerDump, destinationOffset: TOPK * 4 + TOPK * 4 + E_EXP * 2, size: HIDDEN * 2)
            blit.endEncoding()
        }
        encGemvV5(cb, hidden_norm, lw.routerW, router_lg, Din: HIDDEN, Dout: E_EXP)
        if L == 0, let routerDump = LM_DUMP_L0_ROUTER {
            // router_lg (slot 0) f16[E_EXP] — captured AFTER proj, BEFORE softmax.
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: router_lg, sourceOffset: 0,
                      to: routerDump, destinationOffset: TOPK * 4 + TOPK * 4, size: E_EXP * 2)
            blit.endEncoding()
        }
        encSoftmaxTopk(cb, expertScaleBuf: lw.expertScale)
        if L == 0, let routerDump = LM_DUMP_L0_ROUTER {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: expert_ids, sourceOffset: 0,
                      to: routerDump, destinationOffset: 0, size: TOPK * 4)
            blit.copy(from: gate_w, sourceOffset: 0,
                      to: routerDump, destinationOffset: TOPK * 4, size: TOPK * 4)
            blit.endEncoding()
        }
        encRouteCompact(cb)

        // MoE branch: pre_ffn_2(hidden) → fused Q4_K gate_up → gelu*mul → Q5_1 down → combine → post_ffn_2.
        encRMSNormG(cb, x: hidden, gammaBuf: lw.preFfn2Norm, out: hidden_norm, D: HIDDEN, numVecs: B)
        if L == 0, let dump = LM_DUMP_L0_STAGING {
            // Slot 3: pre_feedforward_layernorm_2(hidden) = input to experts.
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: hidden_norm, sourceOffset: 0,
                      to: dump, destinationOffset: 3 * HIDDEN * 2, size: HIDDEN * 2)
            blit.endEncoding()
        }
        encMoeGemvQ4K(cb, hidden_norm, lw.moeGateUp, gate_up_fused,
                      Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: E_EXP, useV6: true)
        encMoeGeluMulFused(cb, fused: gate_up_fused, out: gate_proj, N_half: MOE_INT, numSlots: TOTAL_SLOTS)
        encMoeGemvQ51(cb, gate_proj, lw.moeDown, moe_down_out, Din: MOE_INT, Dout: HIDDEN, numActive: E_EXP, useV6: true)
        if L == 0, let slotsDump = LM_DUMP_L0_MOE_SLOTS {
            let blit = cb.makeBlitCommandEncoder()!
            var off = 0
            blit.copy(from: moe_down_out, sourceOffset: 0,
                      to: slotsDump, destinationOffset: off, size: TOTAL_SLOTS * HIDDEN * 2)
            off += TOTAL_SLOTS * HIDDEN * 2
            blit.copy(from: gate_up_fused, sourceOffset: 0,
                      to: slotsDump, destinationOffset: off, size: TOTAL_SLOTS * 2 * MOE_INT * 2)
            off += TOTAL_SLOTS * 2 * MOE_INT * 2
            blit.copy(from: gate_proj, sourceOffset: 0,
                      to: slotsDump, destinationOffset: off, size: TOTAL_SLOTS * MOE_INT * 2)
            off += TOTAL_SLOTS * MOE_INT * 2
            blit.copy(from: slot_token, sourceOffset: 0, to: slotsDump, destinationOffset: off, size: TOTAL_SLOTS * 4)
            off += TOTAL_SLOTS * 4
            blit.copy(from: batch_slots, sourceOffset: 0, to: slotsDump, destinationOffset: off, size: TOTAL_SLOTS * 4)
            off += TOTAL_SLOTS * 4
            blit.copy(from: group_start, sourceOffset: 0, to: slotsDump, destinationOffset: off, size: (E_EXP + 1) * 4)
            blit.endEncoding()
        }
        encMoeCombineWrite(cb, to: moe_sum)
        if L == 0, let dump = LM_DUMP_L0_STAGING {
            // Slot 4: moe_sum BEFORE post_feedforward_layernorm_2 (raw scatter-sum experts output).
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: moe_sum, sourceOffset: 0,
                      to: dump, destinationOffset: 4 * HIDDEN * 2, size: HIDDEN * 2)
            blit.endEncoding()
        }
        encRMSNormG(cb, x: moe_sum, gammaBuf: lw.postFfn2Norm, out: moe_sum, D: HIDDEN, numVecs: B)
        if L == 0, let dump = LM_DUMP_L0_STAGING {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: moe_sum, sourceOffset: 0,
                      to: dump, destinationOffset: 2 * HIDDEN * 2, size: HIDDEN * 2)
            blit.endEncoding()
        }
        if LM_SKIP_MOE {
            let blit = cb.makeBlitCommandEncoder()!
            blit.fill(buffer: moe_sum, range: 0..<(B * HIDDEN * 2), value: 0)
            blit.endEncoding()
        }
        if LM_SKIP_SHARED {
            let blit = cb.makeBlitCommandEncoder()!
            blit.fill(buffer: mlp_out, range: 0..<(B * HIDDEN * 2), value: 0)
            blit.endEncoding()
        }

        // Combine + final post_ffn_norm + residual add + layer_output_scale (fused).
        encBufferCopy(cb, src: mlp_out, dst: ffn_combined, bytes: B * HIDDEN * 2)
        encAddInplace(cb, dst: ffn_combined, src: moe_sum, N: HIDDEN, numVecs: B)
        encRmsNormAddScale(cb, x: ffn_combined, gammaBuf: lw.postFfnNorm,
                           residual: hidden, scalar: lw.layerOutputScale,
                           out: hidden, N: HIDDEN, numVecs: B)
        if let dump = LM_DUMP_STAGING {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: hidden, sourceOffset: 0,
                      to: dump, destinationOffset: (L + 1) * HIDDEN * 2, size: HIDDEN * 2)
            blit.endEncoding()
        }
    }

    // Final RMSNorm + fused fp16 unembed + final-logit softcap (cap=30.0).
    encRMSNormG(cb, x: hidden, gammaBuf: w.outputNorm, out: hidden_norm, D: HIDDEN, numVecs: B)
    encGemvV4Softcap(cb, hidden_norm, w.unembedW, logits,
                     Din: HIDDEN, Dout: VOCAB, cap: 30.0)
    return cb
}

// ====================================================================
// Prefill forward: B * qLen "virtual slots" through all 30 layers in one CB.
// Distinct from AR `buildStepCB` — separate scratch (pre_*), multi-position
// KV write + RoPE, flex v1 attention kernels (slide Q_BLOCK=8 + full Q_BLOCK=8
// one-TG-per-q_head). All row-generic dispatchers get numVecs = B * qLen.
//
// Caller is responsible for:
//   1) Writing pre_input_tokens[b * qLen + i] and pre_q_positions[b * qLen + i]
//   2) Writing pre_k_len_slide[b] and pre_k_len_full[b] (post-prefill values)
//   3) Setting block_table for the target slots (call initLmState or manage manually)
//   4) Calling precomputeFlexPrefillMasks(qLen:, positionStart:) before build
//
// Returns the built (uncommitted) CB. After commit+wait, pre_logits is
// populated with [B, qLen, VOCAB] fp16 logits.
// Encode the prefill pipeline (embed/scale or skip → 30 layers → optional
// unembed+softcap) into a caller-provided CB. Factored out so multi-tile
// drivers can chain several tiles into one CB with one commit.
//   skipEmbed=true  → caller pre-populated pre_hidden (e.g. image soft
//                     tokens already in text hidden space); kernel skips
//                     embed_lookup + embed-scale.
//   skipUnembed=true→ intermediate tiles whose logits we don't use —
//                     saves the HIDDEN→VOCAB=262144 GEMV + softcap per
//                     skipped tile.
func encodePrefillTileInto(_ cb: MTLCommandBuffer, _ w: LmWeights,
                            qLen: Int, skipEmbed: Bool = false,
                            skipUnembed: Bool = false) {
    precondition(qLen <= MAX_Q_LEN, "qLen \(qLen) exceeds MAX_Q_LEN=\(MAX_Q_LEN)")
    let N = B * qLen               // total token rows
    let NS = B * qLen * TOPK       // total MoE slots

    if !skipEmbed {
        // Embed + Gemma-4 sqrt(hidden) scale, now over N = B*qLen tokens.
        encEmbedInto(cb, tokens: pre_input_tokens, embedTable: w.embedTable,
                     out: pre_hidden, numVecs: N)
        encScaleByScalar(cb, x: pre_hidden, scalar: w.embedScaleBuf, N: HIDDEN, numVecs: N)
    }
    // When skipEmbed=true, caller has already written pre_hidden with the
    // vision-tower-produced soft tokens (already scaled). Image tokens
    // skip embed_lookup entirely — the vision projection already mapped
    // them into the text hidden space.

    for L in 0..<NUM_LAYERS {
        let lw = w.layers[L]
        let isFull = lw.isFull
        let H = isFull ? FULL_H : SLIDE_H
        let KV_H = lw.KV_H
        let HD = lw.HD
        let theta: Float = isFull ? 1_000_000 : 10_000
        let rotary = isFull ? (FULL_HD / 4) : SLIDE_HD
        let q_out = isFull ? pre_q_full_out : pre_q_slide_out
        let k_out = isFull ? pre_k_full_out : pre_k_slide_out
        let v_out = isFull ? pre_v_full_out : pre_v_slide_out
        let Kc = w.K_caches[L]
        let Vc = w.V_caches[L]

        // Fused RMSNorm + Q/K/V projection over all N tokens.
        let Wv = lw.attnV ?? lw.attnK
        encGemvQ80V6RmsnormQKV(cb, x: pre_hidden, gammaBuf: lw.attnNorm,
                                Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                outQ: q_out, outK: k_out, outV: v_out,
                                Din: HIDDEN,
                                DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                                numVecs: N)

        // Per-head norms (numVecs = N * head-count).
        encRMSNormG(cb, x: q_out, gammaBuf: lw.attnQNorm, out: q_out, D: HD, numVecs: N * H)
        encRopeMulti(cb, q_out, q_positions: pre_q_positions, H: H, D: HD,
                     rotary: rotary, theta: theta, qLen: qLen)
        encRMSNormG(cb, x: k_out, gammaBuf: lw.attnKNorm, out: k_out, D: HD, numVecs: N * KV_H)
        encRopeMulti(cb, k_out, q_positions: pre_q_positions, H: KV_H, D: HD,
                     rotary: rotary, theta: theta, qLen: qLen)
        encRMSNormNoScale(cb, x: v_out, out: v_out, D: HD, numVecs: N * KV_H)

        // Multi-position KV write: qLen entries per batch.
        let pg = isFull ? PAGE_FULL : PAGE_SLIDE
        encKVWriteMulti(cb, K: k_out, V: v_out, Kc: Kc, Vc: Vc,
                        q_positions: pre_q_positions,
                        H: KV_H, D: HD, page: pg, qLen: qLen)

        // Attention — flex v1 (Q_BLOCK=8).
        let klBuf = isFull ? pre_k_len_full : pre_k_len_slide
        if isFull {
            encFlexAttnFullPrefill(cb, Q: q_out, O: pre_attn_out, Kc: Kc, Vc: Vc,
                                    kLenBuf: klBuf, qPositions: pre_q_positions,
                                    H_Q: H, H_KV: KV_H, D: HD, qLen: qLen)
        } else {
            encFlexAttnSlidePrefill(cb, Q: q_out, O: pre_attn_out, Kc: Kc, Vc: Vc,
                                     kLenBuf: klBuf, qPositions: pre_q_positions,
                                     H_Q: H, H_KV: KV_H, D: HD, qLen: qLen)
        }

        // o_proj (Q8_0) → pre_mlp_out; fused post-attn norm + residual add on pre_hidden.
        encGemvQ80V6(cb, pre_attn_out, lw.attnOut, pre_mlp_out,
                     Din: H * HD, Dout: HIDDEN, numVecs: N)
        encRmsNormAdd(cb, x: pre_mlp_out, gammaBuf: lw.postAttnNorm,
                      residual: pre_hidden, out: pre_hidden, N: HIDDEN, numVecs: N)

        // Shared MLP (fused RMSNorm + gate_up → gelu*up → down → post_ffn_1_norm).
        encGemvQ80V6RmsnormGateUp(cb, x: pre_hidden, gammaBuf: lw.ffnNorm,
                                   Wg: lw.ffnGate, Wu: lw.ffnUp,
                                   fusedOut: pre_shrd_gate_up_fused,
                                   Din: HIDDEN, Dout: SHARED_INT, numVecs: N)
        encMoeGeluMulFused(cb, fused: pre_shrd_gate_up_fused, out: pre_shrd_gate,
                            N_half: SHARED_INT, numSlots: N)
        encGemvQ80V6(cb, pre_shrd_gate, lw.ffnDown, pre_mlp_out,
                     Din: SHARED_INT, Dout: HIDDEN, numVecs: N)
        encRMSNormG(cb, x: pre_mlp_out, gammaBuf: lw.postFfn1Norm, out: pre_mlp_out,
                    D: HIDDEN, numVecs: N)

        // Router (on pre_hidden = post-attn residual): pre-norm → proj →
        // softmax+topk+renorm → compact. Uses pre_* routing buffers.
        encRouterPreNorm(cb, x: pre_hidden, per_dim_scale: lw.routerScale,
                          out: pre_hidden_norm, numVecs: N)
        encGemvV5(cb, pre_hidden_norm, lw.routerW, pre_router_lg,
                  Din: HIDDEN, Dout: E_EXP, numVecs: N)
        encSoftmaxTopkInto(cb, logits: pre_router_lg, expertIds: pre_expert_ids,
                            gateW: pre_gate_w, expertScaleBuf: lw.expertScale, numVecs: N)
        encRouteCompactInto(cb, expertIds: pre_expert_ids, groupStart: pre_group_start,
                             slotToken: pre_slot_token, batchSlots: pre_batch_slots,
                             numVecs: N)

        // MoE: pre_ffn_2(pre_hidden) → Q4_K gate_up → gelu*up → Q5_1 down → combine → post_ffn_2.
        encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.preFfn2Norm, out: pre_hidden_norm,
                    D: HIDDEN, numVecs: N)
        encMoeGemvQ4K(cb, pre_hidden_norm, lw.moeGateUp, pre_gate_up_fused,
                      Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: E_EXP, useV6: true,
                      slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
        encMoeGeluMulFused(cb, fused: pre_gate_up_fused, out: pre_gate_proj,
                            N_half: MOE_INT, numSlots: NS)
        encMoeGemvQ51(cb, pre_gate_proj, lw.moeDown, pre_moe_down_out,
                      Din: MOE_INT, Dout: HIDDEN, numActive: E_EXP, useV6: true,
                      slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
        encMoeCombineWriteInto(cb, moeOut: pre_moe_down_out, batchSlots: pre_batch_slots,
                                gateW: pre_gate_w, outBuf: pre_moe_sum, numVecs: N)
        encRMSNormG(cb, x: pre_moe_sum, gammaBuf: lw.postFfn2Norm, out: pre_moe_sum,
                    D: HIDDEN, numVecs: N)

        // Combine + final post_ffn_norm + residual add + layer_output_scale.
        encBufferCopy(cb, src: pre_mlp_out, dst: pre_ffn_combined, bytes: N * HIDDEN * 2)
        encAddInplace(cb, dst: pre_ffn_combined, src: pre_moe_sum, N: HIDDEN, numVecs: N)
        encRmsNormAddScale(cb, x: pre_ffn_combined, gammaBuf: lw.postFfnNorm,
                           residual: pre_hidden, scalar: lw.layerOutputScale,
                           out: pre_hidden, N: HIDDEN, numVecs: N)
    }

    // Final output norm + unembed + softcap. v4 softcap's accumulator is
    // MAX_B=8 wide — prefill's N = B*qLen can exceed that, so split into a
    // numVecs-generic GEMV followed by an in-place softcap pass.
    if !skipUnembed {
        encRMSNormG(cb, x: pre_hidden, gammaBuf: w.outputNorm, out: pre_hidden_norm,
                    D: HIDDEN, numVecs: N)
        encGemvV5(cb, pre_hidden_norm, w.unembedW, pre_logits,
                  Din: HIDDEN, Dout: VOCAB, numVecs: N)
        encSoftcapInto(cb, buf: pre_logits, N: VOCAB, numVecs: N, cap: 30.0)
    }
}

// Back-compat wrapper for callers (LmSession, runLmPrefillValidate) that
// want a single-tile CB handed to them ready to commit.
func buildPrefillCB(_ w: LmWeights, qLen: Int, skipEmbed: Bool = false) -> MTLCommandBuffer {
    let cb = queue.makeCommandBuffer()!
    encodePrefillTileInto(cb, w, qLen: qLen, skipEmbed: skipEmbed, skipUnembed: false)
    return cb
}

// Diagnostic: split prefill into (first K layers) + (commit + NaN check on
// pre_hidden) for each K in 0..<NUM_LAYERS, to locate where NaN first
// appears when buildPrefillCB reads from a session whose KV cache was
// partially written by AR.
func debugPrefillLayerBisect(_ w: LmWeights, qLen: Int, sslot: Int) {
    print("  [prefill-bisect qLen=\(qLen) sslot=\(sslot)]")
    let N = B * qLen
    // Embed once at the start (not re-entrant into this probe).
    let cb0 = queue.makeCommandBuffer()!
    encEmbedInto(cb0, tokens: pre_input_tokens, embedTable: w.embedTable,
                 out: pre_hidden, numVecs: N)
    encScaleByScalar(cb0, x: pre_hidden, scalar: w.embedScaleBuf, N: HIDDEN, numVecs: N)
    cb0.commit(); cb0.waitUntilCompleted()
    // Check pre_hidden post-embed
    checkNanOne("post-embed", pre_hidden, count: N * HIDDEN, sslot: sslot, qLen: qLen)

    for L in 0..<NUM_LAYERS {
        if L == 0 || L == 1 {
            encodeOnePrefillLayerSplit(w, L: L, qLen: qLen, sslot: sslot)
        } else {
            let cb = queue.makeCommandBuffer()!
            encodeOnePrefillLayer(cb, w, L: L, qLen: qLen)
            cb.commit(); cb.waitUntilCompleted()
            checkNanOne("after layer \(L)", pre_hidden, count: N * HIDDEN, sslot: sslot, qLen: qLen)
        }
    }
}

// Fine-grained bisect: split a single prefill layer into sub-steps, commit
// each, read back and print NaN stats. Used to localize NaN emergence.
func encodeOnePrefillLayerSplit(_ w: LmWeights, L: Int, qLen: Int, sslot: Int) {
    let N = B * qLen
    let NS = B * qLen * TOPK
    let lw = w.layers[L]
    let isFull = lw.isFull
    let H = isFull ? FULL_H : SLIDE_H
    let KV_H = lw.KV_H
    let HD = lw.HD
    let theta: Float = isFull ? 1_000_000 : 10_000
    let rotary = isFull ? (FULL_HD / 4) : SLIDE_HD
    let q_out = isFull ? pre_q_full_out : pre_q_slide_out
    let k_out = isFull ? pre_k_full_out : pre_k_slide_out
    let v_out = isFull ? pre_v_full_out : pre_v_slide_out
    let Kc = w.K_caches[L]
    let Vc = w.V_caches[L]
    let Wv = lw.attnV ?? lw.attnK

    func commitAndCheck(_ tag: String, buf: MTLBuffer, count: Int, _ block: (MTLCommandBuffer) -> Void) {
        let cb = queue.makeCommandBuffer()!
        block(cb); cb.commit(); cb.waitUntilCompleted()
        checkNanOne("L\(L) \(tag)", buf, count: count, sslot: sslot, qLen: qLen)
    }

    commitAndCheck("qkv_proj", buf: q_out, count: N * H * HD) { cb in
        encGemvQ80V6RmsnormQKV(cb, x: pre_hidden, gammaBuf: lw.attnNorm,
                                Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                outQ: q_out, outK: k_out, outV: v_out,
                                Din: HIDDEN, DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                                numVecs: N)
    }
    commitAndCheck("q_norm+rope", buf: q_out, count: N * H * HD) { cb in
        encRMSNormG(cb, x: q_out, gammaBuf: lw.attnQNorm, out: q_out, D: HD, numVecs: N * H)
        encRopeMulti(cb, q_out, q_positions: pre_q_positions, H: H, D: HD,
                     rotary: rotary, theta: theta, qLen: qLen)
    }
    commitAndCheck("k_norm+rope", buf: k_out, count: N * KV_H * HD) { cb in
        encRMSNormG(cb, x: k_out, gammaBuf: lw.attnKNorm, out: k_out, D: HD, numVecs: N * KV_H)
        encRopeMulti(cb, k_out, q_positions: pre_q_positions, H: KV_H, D: HD,
                     rotary: rotary, theta: theta, qLen: qLen)
    }
    commitAndCheck("v_norm", buf: v_out, count: N * KV_H * HD) { cb in
        encRMSNormNoScale(cb, x: v_out, out: v_out, D: HD, numVecs: N * KV_H)
    }
    let pg = isFull ? PAGE_FULL : PAGE_SLIDE
    commitAndCheck("kv_write", buf: Kc, count: min(N * KV_H * HD, 4096)) { cb in
        encKVWriteMulti(cb, K: k_out, V: v_out, Kc: Kc, Vc: Vc,
                        q_positions: pre_q_positions,
                        H: KV_H, D: HD, page: pg, qLen: qLen)
    }
    // Sample K_cache[layer] at each valid position [0, positionStart + qLen)
    // for sslot. Non-writes would be whatever the page was last left at.
    // This finds the first position with NaN K.
    do {
        let positionStart = Int(pre_q_positions.contents().assumingMemoryBound(to: UInt32.self)[sslot * qLen])
        let kLen = positionStart + qLen
        let btP = block_table.contents().assumingMemoryBound(to: UInt32.self)
        let Kp = Kc.contents().assumingMemoryBound(to: Float16.self)
        var firstNaNPos = -1, firstInfPos = -1, sampledPositions = 0
        var byRange = [0: (nan: 0, total: 0), 1: (nan: 0, total: 0), 2: (nan: 0, total: 0)]
        for pos in 0..<kLen {
            let lp = pos / pg
            let off = pos % pg
            let phys = Int(btP[sslot * MAX_PAGES_PER_SLOT + lp])
            // K_cache layout: [phys * pg + off, kv_head, d]. Sample kv_head=0, all d.
            let baseEl = (phys * pg + off) * KV_H * HD
            var hasNan = false
            for d in 0..<HD {
                let v = Float(Kp[baseEl + d])
                if v.isNaN { hasNan = true; if firstNaNPos < 0 { firstNaNPos = pos } }
                else if v.isInfinite { if firstInfPos < 0 { firstInfPos = pos } }
            }
            sampledPositions += 1
            let r = pos < positionStart - qLen ? 0 : (pos < positionStart ? 1 : 2)
            var e = byRange[r]!
            e.total += 1; if hasNan { e.nan += 1 }
            byRange[r] = e
        }
        print(String(format: "    L\(L) K_cache[sslot=\(sslot), pos=0..\(kLen-1)] sampled=\(sampledPositions) firstNaN=\(firstNaNPos) firstInf=\(firstInfPos)"))
        print("       by-range: prefill-early=\(byRange[0]!), AR-written=\(byRange[1]!), new-prefill=\(byRange[2]!)")
    }
    let klBuf = isFull ? pre_k_len_full : pre_k_len_slide
    commitAndCheck("attention", buf: pre_attn_out, count: N * H * HD) { cb in
        if isFull {
            encFlexAttnFullPrefill(cb, Q: q_out, O: pre_attn_out, Kc: Kc, Vc: Vc,
                                    kLenBuf: klBuf, qPositions: pre_q_positions,
                                    H_Q: H, H_KV: KV_H, D: HD, qLen: qLen)
        } else {
            encFlexAttnSlidePrefill(cb, Q: q_out, O: pre_attn_out, Kc: Kc, Vc: Vc,
                                     kLenBuf: klBuf, qPositions: pre_q_positions,
                                     H_Q: H, H_KV: KV_H, D: HD, qLen: qLen)
        }
    }
    // Per-Q-row breakdown of sslot's attention output (positionStart+i for i in 0..<qLen).
    do {
        let p = pre_attn_out.contents().assumingMemoryBound(to: Float16.self)
        let perSlotAlloc = pre_attn_out.length / 2 / B
        let sbase = sslot * perSlotAlloc
        let rowStride = H * HD
        let positionStart = Int(pre_q_positions.contents().assumingMemoryBound(to: UInt32.self)[sslot * qLen])
        var rowReport: [String] = []
        for q in 0..<qLen {
            var rn = 0
            for i in 0..<rowStride {
                let v = Float(p[sbase + q * rowStride + i])
                if v.isNaN { rn += 1 }
            }
            rowReport.append("Q=\(positionStart + q):\(rn == 0 ? "ok" : "NaN\(rn)")")
        }
        print("    L\(L) per-Q-row: " + rowReport.joined(separator: " "))
    }
    commitAndCheck("o_proj+resid1", buf: pre_hidden, count: N * HIDDEN) { cb in
        encGemvQ80V6(cb, pre_attn_out, lw.attnOut, pre_mlp_out,
                     Din: H * HD, Dout: HIDDEN, numVecs: N)
        encRmsNormAdd(cb, x: pre_mlp_out, gammaBuf: lw.postAttnNorm,
                      residual: pre_hidden, out: pre_hidden, N: HIDDEN, numVecs: N)
    }
    // Remaining (shared MLP + MoE) — as a single commit, probably not needed
    // past the first NaN source.
    commitAndCheck("ffn+moe+resid2", buf: pre_hidden, count: N * HIDDEN) { cb in
        encGemvQ80V6RmsnormGateUp(cb, x: pre_hidden, gammaBuf: lw.ffnNorm,
                                   Wg: lw.ffnGate, Wu: lw.ffnUp,
                                   fusedOut: pre_shrd_gate_up_fused,
                                   Din: HIDDEN, Dout: SHARED_INT, numVecs: N)
        encMoeGeluMulFused(cb, fused: pre_shrd_gate_up_fused, out: pre_shrd_gate,
                            N_half: SHARED_INT, numSlots: N)
        encGemvQ80V6(cb, pre_shrd_gate, lw.ffnDown, pre_mlp_out,
                     Din: SHARED_INT, Dout: HIDDEN, numVecs: N)
        encRMSNormG(cb, x: pre_mlp_out, gammaBuf: lw.postFfn1Norm, out: pre_mlp_out,
                    D: HIDDEN, numVecs: N)
        encRouterPreNorm(cb, x: pre_hidden, per_dim_scale: lw.routerScale,
                          out: pre_hidden_norm, numVecs: N)
        encGemvV5(cb, pre_hidden_norm, lw.routerW, pre_router_lg,
                  Din: HIDDEN, Dout: E_EXP, numVecs: N)
        encSoftmaxTopkInto(cb, logits: pre_router_lg, expertIds: pre_expert_ids,
                            gateW: pre_gate_w, expertScaleBuf: lw.expertScale, numVecs: N)
        encRouteCompactInto(cb, expertIds: pre_expert_ids, groupStart: pre_group_start,
                             slotToken: pre_slot_token, batchSlots: pre_batch_slots,
                             numVecs: N)
        encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.preFfn2Norm, out: pre_hidden_norm,
                    D: HIDDEN, numVecs: N)
        encMoeGemvQ4K(cb, pre_hidden_norm, lw.moeGateUp, pre_gate_up_fused,
                      Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: E_EXP, useV6: true,
                      slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
        encMoeGeluMulFused(cb, fused: pre_gate_up_fused, out: pre_gate_proj,
                            N_half: MOE_INT, numSlots: NS)
        encMoeGemvQ51(cb, pre_gate_proj, lw.moeDown, pre_moe_down_out,
                      Din: MOE_INT, Dout: HIDDEN, numActive: E_EXP, useV6: true,
                      slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
        encMoeCombineWriteInto(cb, moeOut: pre_moe_down_out, batchSlots: pre_batch_slots,
                                gateW: pre_gate_w, outBuf: pre_moe_sum, numVecs: N)
        encRMSNormG(cb, x: pre_moe_sum, gammaBuf: lw.postFfn2Norm, out: pre_moe_sum,
                    D: HIDDEN, numVecs: N)
        encBufferCopy(cb, src: pre_mlp_out, dst: pre_ffn_combined, bytes: N * HIDDEN * 2)
        encAddInplace(cb, dst: pre_ffn_combined, src: pre_moe_sum, N: HIDDEN, numVecs: N)
        encRmsNormAddScale(cb, x: pre_ffn_combined, gammaBuf: lw.postFfnNorm,
                           residual: pre_hidden, scalar: lw.layerOutputScale,
                           out: pre_hidden, N: HIDDEN, numVecs: N)
    }
}

private func checkNanOne(_ tag: String, _ buf: MTLBuffer, count: Int, sslot: Int, qLen: Int) {
    let p = buf.contents().assumingMemoryBound(to: Float16.self)
    let fullCount = buf.length / 2
    var nan = 0, inf = 0, maxAbs: Float = 0
    for i in 0..<fullCount {
        let v = Float(p[i])
        if v.isNaN { nan += 1 }
        else if v.isInfinite { inf += 1 }
        else if abs(v) > maxAbs { maxAbs = abs(v) }
    }
    // Per-slot split: assume the buffer is [B, ...] row-major, so each slot
    // owns fullCount/B halfs. Prints sslot's contribution vs others'.
    let perSlot = fullCount / B
    var snan = 0, smaxAbs: Float = 0, onan = 0, omaxAbs: Float = 0
    for b in 0..<B {
        for i in 0..<perSlot {
            let v = Float(p[b * perSlot + i])
            if b == sslot {
                if v.isNaN { snan += 1 }
                else if abs(v) > smaxAbs { smaxAbs = abs(v) }
            } else {
                if v.isNaN { onan += 1 }
                else if abs(v) > omaxAbs { omaxAbs = abs(v) }
            }
        }
    }
    print(String(format: "    %-20@  ALL: NaN=%d max=%.3f | sslot=%d: NaN=%d max=%.3f | others: NaN=%d max=%.3f",
                 tag as NSString, nan, maxAbs, sslot, snan, smaxAbs, onan, omaxAbs))
}

// Encode ONE prefill layer's work into `cb`. Used by debugPrefillLayerBisect
// to commit per-layer and read back pre_hidden. Mirrors the layer body
// inside encodePrefillTileInto.
func encodeOnePrefillLayer(_ cb: MTLCommandBuffer, _ w: LmWeights, L: Int, qLen: Int) {
    let N = B * qLen
    let NS = B * qLen * TOPK
    let lw = w.layers[L]
    let isFull = lw.isFull
    let H = isFull ? FULL_H : SLIDE_H
    let KV_H = lw.KV_H
    let HD = lw.HD
    let theta: Float = isFull ? 1_000_000 : 10_000
    let rotary = isFull ? (FULL_HD / 4) : SLIDE_HD
    let q_out = isFull ? pre_q_full_out : pre_q_slide_out
    let k_out = isFull ? pre_k_full_out : pre_k_slide_out
    let v_out = isFull ? pre_v_full_out : pre_v_slide_out
    let Kc = w.K_caches[L]
    let Vc = w.V_caches[L]
    let Wv = lw.attnV ?? lw.attnK
    encGemvQ80V6RmsnormQKV(cb, x: pre_hidden, gammaBuf: lw.attnNorm,
                            Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                            outQ: q_out, outK: k_out, outV: v_out,
                            Din: HIDDEN,
                            DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                            numVecs: N)
    encRMSNormG(cb, x: q_out, gammaBuf: lw.attnQNorm, out: q_out, D: HD, numVecs: N * H)
    encRopeMulti(cb, q_out, q_positions: pre_q_positions, H: H, D: HD,
                 rotary: rotary, theta: theta, qLen: qLen)
    encRMSNormG(cb, x: k_out, gammaBuf: lw.attnKNorm, out: k_out, D: HD, numVecs: N * KV_H)
    encRopeMulti(cb, k_out, q_positions: pre_q_positions, H: KV_H, D: HD,
                 rotary: rotary, theta: theta, qLen: qLen)
    encRMSNormNoScale(cb, x: v_out, out: v_out, D: HD, numVecs: N * KV_H)
    let pg = isFull ? PAGE_FULL : PAGE_SLIDE
    encKVWriteMulti(cb, K: k_out, V: v_out, Kc: Kc, Vc: Vc,
                    q_positions: pre_q_positions,
                    H: KV_H, D: HD, page: pg, qLen: qLen)
    let klBuf = isFull ? pre_k_len_full : pre_k_len_slide
    if isFull {
        encFlexAttnFullPrefill(cb, Q: q_out, O: pre_attn_out, Kc: Kc, Vc: Vc,
                                kLenBuf: klBuf, qPositions: pre_q_positions,
                                H_Q: H, H_KV: KV_H, D: HD, qLen: qLen)
    } else {
        encFlexAttnSlidePrefill(cb, Q: q_out, O: pre_attn_out, Kc: Kc, Vc: Vc,
                                 kLenBuf: klBuf, qPositions: pre_q_positions,
                                 H_Q: H, H_KV: KV_H, D: HD, qLen: qLen)
    }
    encGemvQ80V6(cb, pre_attn_out, lw.attnOut, pre_mlp_out,
                 Din: H * HD, Dout: HIDDEN, numVecs: N)
    encRmsNormAdd(cb, x: pre_mlp_out, gammaBuf: lw.postAttnNorm,
                  residual: pre_hidden, out: pre_hidden, N: HIDDEN, numVecs: N)
    encGemvQ80V6RmsnormGateUp(cb, x: pre_hidden, gammaBuf: lw.ffnNorm,
                               Wg: lw.ffnGate, Wu: lw.ffnUp,
                               fusedOut: pre_shrd_gate_up_fused,
                               Din: HIDDEN, Dout: SHARED_INT, numVecs: N)
    encMoeGeluMulFused(cb, fused: pre_shrd_gate_up_fused, out: pre_shrd_gate,
                        N_half: SHARED_INT, numSlots: N)
    encGemvQ80V6(cb, pre_shrd_gate, lw.ffnDown, pre_mlp_out,
                 Din: SHARED_INT, Dout: HIDDEN, numVecs: N)
    encRMSNormG(cb, x: pre_mlp_out, gammaBuf: lw.postFfn1Norm, out: pre_mlp_out,
                D: HIDDEN, numVecs: N)
    encRouterPreNorm(cb, x: pre_hidden, per_dim_scale: lw.routerScale,
                      out: pre_hidden_norm, numVecs: N)
    encGemvV5(cb, pre_hidden_norm, lw.routerW, pre_router_lg,
              Din: HIDDEN, Dout: E_EXP, numVecs: N)
    encSoftmaxTopkInto(cb, logits: pre_router_lg, expertIds: pre_expert_ids,
                        gateW: pre_gate_w, expertScaleBuf: lw.expertScale, numVecs: N)
    encRouteCompactInto(cb, expertIds: pre_expert_ids, groupStart: pre_group_start,
                         slotToken: pre_slot_token, batchSlots: pre_batch_slots,
                         numVecs: N)
    encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.preFfn2Norm, out: pre_hidden_norm,
                D: HIDDEN, numVecs: N)
    encMoeGemvQ4K(cb, pre_hidden_norm, lw.moeGateUp, pre_gate_up_fused,
                  Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: E_EXP, useV6: true,
                  slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
    encMoeGeluMulFused(cb, fused: pre_gate_up_fused, out: pre_gate_proj,
                        N_half: MOE_INT, numSlots: NS)
    encMoeGemvQ51(cb, pre_gate_proj, lw.moeDown, pre_moe_down_out,
                  Din: MOE_INT, Dout: HIDDEN, numActive: E_EXP, useV6: true,
                  slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
    encMoeCombineWriteInto(cb, moeOut: pre_moe_down_out, batchSlots: pre_batch_slots,
                            gateW: pre_gate_w, outBuf: pre_moe_sum, numVecs: N)
    encRMSNormG(cb, x: pre_moe_sum, gammaBuf: lw.postFfn2Norm, out: pre_moe_sum,
                D: HIDDEN, numVecs: N)
    encBufferCopy(cb, src: pre_mlp_out, dst: pre_ffn_combined, bytes: N * HIDDEN * 2)
    encAddInplace(cb, dst: pre_ffn_combined, src: pre_moe_sum, N: HIDDEN, numVecs: N)
    encRmsNormAddScale(cb, x: pre_ffn_combined, gammaBuf: lw.postFfnNorm,
                       residual: pre_hidden, scalar: lw.layerOutputScale,
                       out: pre_hidden, N: HIDDEN, numVecs: N)
}

// ====================================================================
// PrefixCache: FNV-1a hash → shared phys pages. On a new prompt, hash
// the prefix tokens at PAGE-granularity and reuse existing phys pages
// for any common prefix. Each hash entry carries a refcount so pages
// stay live while in use by any slot. Implementation only — the attn
// kernel doesn't need to know about sharing; it only reads block_table,
// which we populate with shared phys IDs for slots with a shared prefix.
// ====================================================================
final class PrefixCache {
    private var byHash: [UInt64: (pages: [Int], length: Int, refCount: Int)] = [:]
    private var freeList: [Int] = []
    private(set) var nextPhys: Int = 0
    private let maxPhys: Int

    init(maxPhys: Int) {
        self.maxPhys = maxPhys
    }

    // FNV-1a hash of a sequence of UInt32 token IDs.
    static func hash(_ tokens: ArraySlice<UInt32>) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for t in tokens {
            h ^= UInt64(t)
            h = h &* 0x100000001b3
        }
        return h
    }

    // Return phys pages covering `tokens` — reused if an entry for this
    // exact hash+length is cached, else freshly allocated.
    func getOrAllocate(tokens: [UInt32], pageSize: Int) -> [Int] {
        let h = PrefixCache.hash(tokens[...])
        if var entry = byHash[h], entry.length == tokens.count {
            entry.refCount += 1
            byHash[h] = entry
            return entry.pages
        }
        let numPages = (tokens.count + pageSize - 1) / pageSize
        var pages: [Int] = []
        pages.reserveCapacity(numPages)
        for _ in 0..<numPages {
            if let recycled = freeList.popLast() {
                pages.append(recycled)
            } else {
                precondition(nextPhys < maxPhys, "PrefixCache out of phys pages")
                pages.append(nextPhys)
                nextPhys += 1
            }
        }
        byHash[h] = (pages: pages, length: tokens.count, refCount: 1)
        return pages
    }

    func release(tokens: [UInt32]) {
        let h = PrefixCache.hash(tokens[...])
        guard var entry = byHash[h] else { return }
        entry.refCount -= 1
        if entry.refCount <= 0 {
            freeList.append(contentsOf: entry.pages)
            byHash.removeValue(forKey: h)
        } else {
            byHash[h] = entry
        }
    }

    var entryCount: Int { byHash.count }
    var totalCachedPages: Int { byHash.values.reduce(0) { $0 + $1.pages.count } }
}

// ====================================================================
// Real-weight pipeline. Loads + repacks all per-layer weights from a
// Gemma-4-A4B Q4_K_M GGUF into swizzled-v6 buffers, plus dequants the
// tied token_embd into fp16 for embed + unembed. Validates the full
// weight-loading pipeline end-to-end — every shape, every dtype,
// every repack — ahead of running a real forward pass.
// ====================================================================
// Paged-attention shared-prefix nondivergence test. Sets up B=4 slots whose
// block_tables all point at the SAME phys pages (i.e., shared prefix),
// populates the KV cache with deterministic values, and verifies:
//   A) Identical-Q run: all 4 slot outputs are bitwise identical
//      (kernel must be slot-order-invariant).
//   B) Distinct-Q run: each slot's output is a function only of its own Q
//      and the shared pages — not of any other slot's Q. We prove this by
//      re-running with other slots' Qs zeroed and checking slot 0's output
//      is unchanged.
// If either check fails, the shared-prefix decode is not equivalent to
// serial decode from the same prefix state.
func runPagedAttnSharedPrefixTest() {
    // Deterministic fill of a half buffer (seeded, matches halfBuf).
    func fillHalf(_ buf: MTLBuffer, seed: UInt64) {
        let p = buf.contents().assumingMemoryBound(to: Float16.self)
        let n = buf.length / 2
        var s = seed
        for i in 0..<n {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let u = Float((s >> 32) & 0xFFFFFFFF) / Float(UInt32.max)
            p[i] = Float16(u - 0.5) * Float16(0.1)
        }
    }
    func halfAt(_ buf: MTLBuffer, _ offset: Int) -> Float16 {
        buf.contents().load(fromByteOffset: offset * 2, as: Float16.self)
    }

    // Helper: set up shared block_tables (all slots point to same phys 0..N-1).
    func setSharedBlockTable(numPages: Int, kLen: Int, numPagesBuf: MTLBuffer, kLenBuf: MTLBuffer) {
        let bt = block_table.contents().assumingMemoryBound(to: UInt32.self)
        for slot in 0..<B {
            for p in 0..<numPages {
                bt[slot * MAX_PAGES_PER_SLOT + p] = UInt32(p)
            }
        }
        let np = numPagesBuf.contents().assumingMemoryBound(to: UInt32.self)
        let kl = kLenBuf.contents().assumingMemoryBound(to: UInt32.self)
        for slot in 0..<B {
            np[slot] = UInt32(numPages)
            kl[slot] = UInt32(kLen)
        }
    }

    // Measure max abs diff between two half tensors of given element count.
    func maxDiff(_ a: MTLBuffer, _ b: MTLBuffer, count: Int) -> Float {
        let ap = a.contents().assumingMemoryBound(to: Float16.self)
        let bp = b.contents().assumingMemoryBound(to: Float16.self)
        var m: Float = 0
        for i in 0..<count {
            m = max(m, abs(Float(ap[i]) - Float(bp[i])))
        }
        return m
    }

    func sliceBuffer(_ buf: MTLBuffer, start: Int, count: Int) -> [Float] {
        let p = buf.contents().assumingMemoryBound(to: Float16.self)
        return (0..<count).map { Float(p[start + $0]) }
    }

    func runOne(isFull: Bool) {
        let H = isFull ? FULL_H : SLIDE_H
        let KV_H = isFull ? FULL_KV_H : SLIDE_KV_H
        let HD = isFull ? FULL_HD : SLIDE_HD
        let PAGE = isFull ? PAGE_FULL : PAGE_SLIDE
        // Local K/V caches sized just for this test (TOTAL_PAGES global pool).
        let Kc = emptyHalf(TOTAL_PAGES * PAGE * KV_H * HD)
        let Vc = emptyHalf(TOTAL_PAGES * PAGE * KV_H * HD)
        let npBuf = isFull ? num_pages_full : num_pages_slide
        let klBuf = isFull ? k_len_full : k_len_slide
        let qBuf = isFull ? q_full_out : q_slide_out

        let NUM_PAGES = 8
        let K_LEN = NUM_PAGES * PAGE
        let slotQElems = H * HD          // halves per slot in Q
        let slotOutElems = H * HD        // halves per slot in attn_out

        // Populate KV cache pages 0..NUM_PAGES-1 deterministically. Other pages
        // are left random so we can detect if any slot reaches past its shared
        // block_table range.
        fillHalf(Kc, seed: 0xD00D0000 + (isFull ? 1 : 0))
        fillHalf(Vc, seed: 0xD00D1000 + (isFull ? 1 : 0))
        setSharedBlockTable(numPages: NUM_PAGES, kLen: K_LEN, numPagesBuf: npBuf, kLenBuf: klBuf)

        let label = isFull ? "full-attn GQA" : "sliding GQA"
        print("  --- \(label): H_Q=\(H) H_KV=\(KV_H) HD=\(HD) PAGE=\(PAGE) K_len=\(K_LEN) ---")

        // ======= Mode A: all 4 slots share same Q =======
        // Fill slot 0's Q, then copy to slots 1-3. Run attention. All outputs equal.
        fillHalf(qBuf, seed: 0xBEEF_0000)
        let qp = qBuf.contents().assumingMemoryBound(to: Float16.self)
        for slot in 1..<B {
            for i in 0..<slotQElems {
                qp[slot * slotQElems + i] = qp[i]
            }
        }
        let outSameBuf = emptyHalf(B * slotOutElems)
        let cbA = queue.makeCommandBuffer()!
        if isFull {
            encPagedAttnFullGqa(cbA, Q: qBuf, O: outSameBuf, Kc: Kc, Vc: Vc,
                                 numPagesBuf: npBuf, kLenBuf: klBuf,
                                 H_Q: H, H_KV: KV_H, D: HD)
        } else {
            encPagedAttnSlideGqa(cbA, Q: qBuf, O: outSameBuf, Kc: Kc, Vc: Vc,
                                  numPagesBuf: npBuf, kLenBuf: klBuf,
                                  H_Q: H, H_KV: KV_H, D: HD)
        }
        cbA.commit(); cbA.waitUntilCompleted()

        var maxCrossSlot: Float = 0
        let outp = outSameBuf.contents().assumingMemoryBound(to: Float16.self)
        for slot in 1..<B {
            for i in 0..<slotOutElems {
                let d = abs(Float(outp[i]) - Float(outp[slot * slotOutElems + i]))
                if d > maxCrossSlot { maxCrossSlot = d }
            }
        }
        print(String(format: "    Mode A (identical Q × 4 slots): max cross-slot diff = %.3e", maxCrossSlot))
        let aPass = maxCrossSlot == 0
        print("    Mode A: \(aPass ? "✓ BITWISE IDENTICAL across slots" : "✗ SLOTS DIVERGE")")

        // ======= Mode B: 4 distinct Qs, verify slot-independence =======
        // Run 1: 4 different Qs (different seeds per slot)
        for slot in 0..<B {
            let slotPtr = UnsafeMutableRawPointer(mutating: qp.advanced(by: slot * slotQElems))
            let slotBuf = device.makeBuffer(bytesNoCopy: slotPtr, length: slotQElems * 2,
                                             options: .storageModeShared, deallocator: nil)!
            fillHalf(slotBuf, seed: 0xCAFE_0000 &+ UInt64(slot))
        }
        let outRun1 = emptyHalf(B * slotOutElems)
        let cbB1 = queue.makeCommandBuffer()!
        if isFull {
            encPagedAttnFullGqa(cbB1, Q: qBuf, O: outRun1, Kc: Kc, Vc: Vc,
                                 numPagesBuf: npBuf, kLenBuf: klBuf,
                                 H_Q: H, H_KV: KV_H, D: HD)
        } else {
            encPagedAttnSlideGqa(cbB1, Q: qBuf, O: outRun1, Kc: Kc, Vc: Vc,
                                  numPagesBuf: npBuf, kLenBuf: klBuf,
                                  H_Q: H, H_KV: KV_H, D: HD)
        }
        cbB1.commit(); cbB1.waitUntilCompleted()

        // Run 2: keep slot 0's Q, zero out slots 1-3
        for slot in 1..<B {
            for i in 0..<slotQElems { qp[slot * slotQElems + i] = 0 }
        }
        let outRun2 = emptyHalf(B * slotOutElems)
        let cbB2 = queue.makeCommandBuffer()!
        if isFull {
            encPagedAttnFullGqa(cbB2, Q: qBuf, O: outRun2, Kc: Kc, Vc: Vc,
                                 numPagesBuf: npBuf, kLenBuf: klBuf,
                                 H_Q: H, H_KV: KV_H, D: HD)
        } else {
            encPagedAttnSlideGqa(cbB2, Q: qBuf, O: outRun2, Kc: Kc, Vc: Vc,
                                  numPagesBuf: npBuf, kLenBuf: klBuf,
                                  H_Q: H, H_KV: KV_H, D: HD)
        }
        cbB2.commit(); cbB2.waitUntilCompleted()

        // Slot 0 of run 1 (4 distinct Qs) vs slot 0 of run 2 (only slot 0's Q).
        // Should match — slot 0's output doesn't depend on other slots' Qs.
        var maxSlot0Diff: Float = 0
        let r1 = outRun1.contents().assumingMemoryBound(to: Float16.self)
        let r2 = outRun2.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<slotOutElems {
            maxSlot0Diff = max(maxSlot0Diff, abs(Float(r1[i]) - Float(r2[i])))
        }
        print(String(format: "    Mode B (distinct Qs, serial-equiv check): slot-0 diff run1 vs run2 = %.3e", maxSlot0Diff))
        let bPass = maxSlot0Diff == 0
        print("    Mode B: \(bPass ? "✓ slot 0 is isolated from other slots' Qs" : "✗ CROSS-SLOT CONTAMINATION")")

        // Print a few sample output values so regressions are grep-able.
        let sample = sliceBuffer(outRun1, start: 0, count: 8)
        print(String(format: "    slot 0 output[0..7] = [%@]",
                     sample.map { String(format: "%.4f", $0) }.joined(separator: ", ")))
    }

    runOne(isFull: false)
    runOne(isFull: true)

    // Smoke test the PrefixCache as well.
    print("  --- PrefixCache smoke test ---")
    let pc = PrefixCache(maxPhys: 1024)
    let prompt = Array<UInt32>(stride(from: UInt32(0), to: UInt32(64), by: 1))  // 64 tokens
    let pages1 = pc.getOrAllocate(tokens: prompt, pageSize: 8)
    let pages2 = pc.getOrAllocate(tokens: prompt, pageSize: 8)
    precondition(pages1 == pages2, "identical prefix must dedupe")
    let differentPrompt = Array<UInt32>(stride(from: UInt32(1000), to: UInt32(1064), by: 1))
    let pages3 = pc.getOrAllocate(tokens: differentPrompt, pageSize: 8)
    precondition(Set(pages1).isDisjoint(with: Set(pages3)), "different prefixes must not share pages")
    print("    cache: \(pc.entryCount) entries, \(pc.totalCachedPages) pages allocated")
    print("    identical-prompt hit: pages match (\(pages1.count) pages)")
    print("    disjoint-prompt miss: new pages allocated (\(pages3.count) pages)")
}

struct LayerW {
    let attnQ, attnK, attnOut: MTLBuffer              // Q8_0 v6 swizzled
    let attnV: MTLBuffer?                              // nil on full-attn layers:
                                                        // Gemma-4 drops the V projection and uses
                                                        // K as V at those layers (see llama.cpp
                                                        // gemma4-iswa.cpp:83-85)
    let ffnGate, ffnUp, ffnDown: MTLBuffer            // Q8_0 v6 swizzled
    let moeGateUp: MTLBuffer                           // Q4_K v6 swizzled
    let moeDown: MTLBuffer                             // Q5_1 v6 swizzled
    let attnNorm, postAttnNorm: MTLBuffer              // f16 from f32
    let attnQNorm, attnKNorm: MTLBuffer                // f16 from f32 (per-head)
    let ffnNorm, postFfn1Norm: MTLBuffer               // shared FFN pre/post
    let preFfn2Norm, postFfn2Norm: MTLBuffer           // MoE pre/post
    let postFfnNorm: MTLBuffer                          // combined post-FFN
    let routerW, routerScale, expertScale: MTLBuffer   // f32 passthrough
    let layerOutputScale: MTLBuffer                     // f32 scalar
    let isFull: Bool
    let KV_H: Int
    let HD: Int
}

// Bundled real-weight + per-layer KV cache + tokenizer state for the LM forward.
struct LmWeights {
    let layers: [LayerW]
    let embedTable: MTLBuffer        // fp16 [VOCAB, HIDDEN] dequantized from Q8_0
    let unembedW: MTLBuffer          // fp16 [HIDDEN, VOCAB] transposed tied view
    let outputNorm: MTLBuffer        // fp16 [HIDDEN] final RMSNorm gamma
    let embedScaleBuf: MTLBuffer     // fp32 single scalar = sqrt(HIDDEN)
    let K_caches: [MTLBuffer]        // per-layer K cache (size depends on isFull)
    let V_caches: [MTLBuffer]        // per-layer V cache (size depends on isFull)
    let bosTokenId: UInt32
    let eosTokenId: UInt32           // tokenizer.ggml.eos_token_id (Gemma4: 106 = <end_of_turn>)
    let addBosToken: Bool            // tokenizer.ggml.add_bos_token
    let vocabTokens: [String]        // decoded from tokenizer.ggml.tokens
    let merges: [String]             // tokenizer.ggml.merges — "TOKEN_A TOKEN_B" pairs in priority order
}

// Load every weight we need for a real Gemma-4-A4B LM forward from a Q4_K_M
// GGUF. Q8_0 dense weights get swizzled for the v6 kernel; Q4_K / Q5_1 MoE
// weights get per-expert-swizzled; norms load f32→f16; routing scales and
// layer scalars stay f32. Allocates per-layer paged K/V caches (zero-filled).
func loadLmWeights(ggufPath: String) throws -> LmWeights {
    let t0 = Date()
    print("  loading GGUF: \(ggufPath)")
    let g = try GGUFFile(ggufPath)
    print(String(format: "  GGUF parsed in %.1f ms (%d tensors, %d metadata)",
                 Date().timeIntervalSince(t0) * 1000, g.tensors.count, g.metadata.count))

    // Per-layer KV_H (Gemma-4 alternates: 5 sliding with KV_H=8, then 1 full with KV_H=2).
    var layerKVH: [Int] = []
    if let kvArr = g.metadata["gemma4.attention.head_count_kv"] as? [Any] {
        for v in kvArr {
            if let vi = v as? UInt32      { layerKVH.append(Int(vi)) }
            else if let vi = v as? Int32  { layerKVH.append(Int(vi)) }
            else if let vi = v as? Int    { layerKVH.append(vi) }
            else if let vi = v as? UInt64 { layerKVH.append(Int(vi)) }
            else if let vi = v as? Int64  { layerKVH.append(Int(vi)) }
        }
    }
    precondition(layerKVH.count == NUM_LAYERS,
                 "expected \(NUM_LAYERS) KV_H entries, got \(layerKVH.count)")
    print("  per-layer KV_H: \(layerKVH)")

    // ---------- Per-tensor loaders ----------
    func loadF32AsF16(_ name: String) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == .f32, "\(name): expected f32")
        let nElems = info.shape.reduce(1, *)
        let dst = device.makeBuffer(length: nElems * 2, options: .storageModeShared)!
        let sp = g.base.advanced(by: info.dataOffset).assumingMemoryBound(to: Float.self)
        let dp = dst.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<nElems { dp[i] = Float16(sp[i]) }
        return dst
    }
    func loadF32Raw(_ name: String) throws -> MTLBuffer {
        return try g.makeMetalBuffer(name, device: device)
    }
    // Load a 2D f32 GGUF tensor and convert to a half-precision buffer laid out
    // as [D_in, D_out] row-major, which is what dense_gemv_v5 expects when it
    // reads W[k*D_out + n]. GGUF shape reports [D_in, D_out] but stores bytes
    // as [D_out, D_in] row-major (GGUF axis 0 = fastest = D_in). Our per-layer
    // router weight `ffn_gate_inp.weight` is the only f32 2D tensor; it must
    // be transposed AND cast to half for the kernel to read correctly.
    func loadF32ToHalfTransposed2D(_ name: String) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == .f32, "\(name): expected f32")
        precondition(info.shape.count == 2, "\(name): expected 2D, got shape \(info.shape)")
        let D_in = info.shape[0], D_out = info.shape[1]
        let srcBuf = try g.makeMetalBuffer(name, device: device)
        let dst = device.makeBuffer(length: D_in * D_out * 2, options: .storageModeShared)!
        let src = srcBuf.contents().assumingMemoryBound(to: Float.self)
        let dp = dst.contents().assumingMemoryBound(to: Float16.self)
        // Source bytes are [D_out, D_in] row-major: src[e*D_in + k] = W[e, k].
        // Destination wants [D_in, D_out] row-major: dp[k*D_out + n] = W[n, k].
        for k in 0..<D_in {
            for n in 0..<D_out {
                dp[k * D_out + n] = Float16(src[n * D_in + k])
            }
        }
        return dst
    }
    func loadQ80Swizzled(_ name: String) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == .q8_0, "\(name): expected q8_0")
        let Din = info.shape[0], Dout = info.shape[1]
        let raw = try g.makeMetalBuffer(name, device: device)
        let sw = device.makeBuffer(length: raw.length, options: .storageModeShared)!
        repackQ80ToSwizzled(src: raw, dst: sw, Din: Din, Dout: Dout)
        return sw
    }
    func loadMoESwizzled(_ name: String, dtype: GGMLType, blkBytes: Int, blkElems: Int,
                          E: Int) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == dtype, "\(name): expected \(dtype)")
        let Din = info.shape[0], Dout = info.shape[1]
        precondition(info.shape[2] == E, "\(name): expected E=\(E)")
        let raw = try g.makeMetalBuffer(name, device: device)
        let sw = device.makeBuffer(length: raw.length, options: .storageModeShared)!
        let nbc = Din / blkElems
        let colBytes = nbc * blkBytes
        let nSuper = Dout / 32
        let sp = raw.contents().assumingMemoryBound(to: UInt8.self)
        let dp = sw.contents().assumingMemoryBound(to: UInt8.self)
        for expert in 0..<E {
            let srcExpBase = expert * Dout * colBytes
            let dstExpBase = expert * nSuper * nbc * 32 * blkBytes
            for ns in 0..<nSuper {
                let srcColBase = srcExpBase + ns * 32 * colBytes
                let dstSuperBase = dstExpBase + ns * nbc * 32 * blkBytes
                for kb in 0..<nbc {
                    for col in 0..<32 {
                        let srcOff = srcColBase + col * colBytes + kb * blkBytes
                        let dstOff = dstSuperBase + kb * 32 * blkBytes + col * blkBytes
                        memcpy(dp.advanced(by: dstOff), sp.advanced(by: srcOff), blkBytes)
                    }
                }
            }
        }
        return sw
    }

    // ---------- Load all 30 layers ----------
    let tLoad = Date()
    var layers: [LayerW] = []
    layers.reserveCapacity(NUM_LAYERS)
    for L in 0..<NUM_LAYERS {
        let p = "blk.\(L)."
        let lkv = layerKVH[L]
        let isFull = (lkv == 2)
        let hd = isFull ? FULL_HD : SLIDE_HD
        let attnVBuf: MTLBuffer? = (g.tensors["\(p)attn_v.weight"] != nil)
            ? try loadQ80Swizzled("\(p)attn_v.weight") : nil
        let lw = LayerW(
            attnQ:    try loadQ80Swizzled("\(p)attn_q.weight"),
            attnK:    try loadQ80Swizzled("\(p)attn_k.weight"),
            attnOut:  try loadQ80Swizzled("\(p)attn_output.weight"),
            attnV:    attnVBuf,
            ffnGate:  try loadQ80Swizzled("\(p)ffn_gate.weight"),
            ffnUp:    try loadQ80Swizzled("\(p)ffn_up.weight"),
            ffnDown:  try loadQ80Swizzled("\(p)ffn_down.weight"),
            moeGateUp: try loadMoESwizzled("\(p)ffn_gate_up_exps.weight",
                                           dtype: .q4_K, blkBytes: 144, blkElems: 256, E: E_EXP),
            moeDown:   try loadMoESwizzled("\(p)ffn_down_exps.weight",
                                           dtype: .q5_1, blkBytes: 24, blkElems: 32, E: E_EXP),
            attnNorm:       try loadF32AsF16("\(p)attn_norm.weight"),
            postAttnNorm:   try loadF32AsF16("\(p)post_attention_norm.weight"),
            attnQNorm:      try loadF32AsF16("\(p)attn_q_norm.weight"),
            attnKNorm:      try loadF32AsF16("\(p)attn_k_norm.weight"),
            ffnNorm:        try loadF32AsF16("\(p)ffn_norm.weight"),
            postFfn1Norm:   try loadF32AsF16("\(p)post_ffw_norm_1.weight"),
            preFfn2Norm:    try loadF32AsF16("\(p)pre_ffw_norm_2.weight"),
            postFfn2Norm:   try loadF32AsF16("\(p)post_ffw_norm_2.weight"),
            postFfnNorm:    try loadF32AsF16("\(p)post_ffw_norm.weight"),
            routerW:          try loadF32ToHalfTransposed2D("\(p)ffn_gate_inp.weight"),
            routerScale:      try loadF32Raw("\(p)ffn_gate_inp.scale"),
            expertScale:      try loadF32Raw("\(p)ffn_down_exps.scale"),
            layerOutputScale: try loadF32Raw("\(p)layer_output_scale.weight"),
            isFull: isFull, KV_H: lkv, HD: hd
        )
        layers.append(lw)
        if L == 0 || L == NUM_LAYERS - 1 || (L + 1) % 10 == 0 || isFull {
            let vNote = (attnVBuf == nil) ? " (V reused from K)" : ""
            print("    layer \(L): \(isFull ? "full" : "slide") KV_H=\(lkv) HD=\(hd) loaded\(vNote)")
        }
    }
    print(String(format: "  %d layers loaded+repacked in %.1f sec",
                 NUM_LAYERS, Date().timeIntervalSince(tLoad)))

    // ---------- Dequant tied token_embd Q8_0 → fp16 twice (embed table + transposed unembed) ----------
    let tEmbed = Date()
    let embedInfo = try g.tensor("token_embd.weight")
    precondition(embedInfo.dtype == .q8_0, "token_embd expected Q8_0")
    let eDin = embedInfo.shape[0], eDout = embedInfo.shape[1]
    precondition(eDin == HIDDEN && eDout == VOCAB, "embed shape mismatch")
    let embedTable = device.makeBuffer(length: VOCAB * HIDDEN * 2, options: .storageModeShared)!
    let unembedW   = device.makeBuffer(length: HIDDEN * VOCAB * 2, options: .storageModeShared)!
    let srcBase = g.base.advanced(by: embedInfo.dataOffset)
    let embedDp = embedTable.contents().assumingMemoryBound(to: Float16.self)
    let unembedDp = unembedW.contents().assumingMemoryBound(to: Float16.self)
    let nbc = HIDDEN / 32
    let BLK = 34
    let colBytes = nbc * BLK
    for vo in 0..<VOCAB {
        let colBase = vo * colBytes
        for kb in 0..<nbc {
            let blkOff = colBase + kb * BLK
            let dFloat = Float(srcBase.load(fromByteOffset: blkOff, as: Float16.self))
            let baseD = kb * 32
            for pi in 0..<32 {
                let qsByte = srcBase.load(fromByteOffset: blkOff + 2 + pi, as: Int8.self)
                let val = Float16(dFloat * Float(qsByte))
                embedDp[vo * HIDDEN + baseD + pi] = val
                unembedDp[(baseD + pi) * VOCAB + vo] = val
            }
        }
    }
    print(String(format: "  token_embd Q8_0 → fp16 dequant in %.1f sec", Date().timeIntervalSince(tEmbed)))
    let outputNorm = try loadF32AsF16("output_norm.weight")

    // ---------- Per-layer paged K/V caches (zero-filled) ----------
    let tCache = Date()
    var K_caches: [MTLBuffer] = []
    var V_caches: [MTLBuffer] = []
    K_caches.reserveCapacity(NUM_LAYERS)
    V_caches.reserveCapacity(NUM_LAYERS)
    var kvBytes = 0
    for L in 0..<NUM_LAYERS {
        let lw = layers[L]
        let pg = lw.isFull ? PAGE_FULL : PAGE_SLIDE
        let cacheElems = TOTAL_PAGES * pg * lw.KV_H * lw.HD
        let Kbuf = device.makeBuffer(length: cacheElems * 2, options: .storageModeShared)!
        let Vbuf = device.makeBuffer(length: cacheElems * 2, options: .storageModeShared)!
        memset(Kbuf.contents(), 0, Kbuf.length)
        memset(Vbuf.contents(), 0, Vbuf.length)
        K_caches.append(Kbuf)
        V_caches.append(Vbuf)
        kvBytes += Kbuf.length + Vbuf.length
    }
    print(String(format: "  per-layer K/V caches allocated: %.1f MB in %.2f sec",
                 Double(kvBytes) / (1024*1024), Date().timeIntervalSince(tCache)))

    // ---------- Gemma-4 embed scale = sqrt(hidden) ----------
    let embedScaleBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
    embedScaleBuf.contents().assumingMemoryBound(to: Float.self)[0] = Float(HIDDEN).squareRoot()

    // ---------- Tokenizer bits from GGUF metadata ----------
    func readU32(_ key: String, default def: UInt32) -> UInt32 {
        guard let v = g.metadata[key] else { return def }
        if let x = v as? UInt32      { return x }
        if let x = v as? Int32       { return UInt32(x) }
        if let x = v as? UInt64      { return UInt32(x) }
        return def
    }
    let bosTokenId: UInt32 = readU32("tokenizer.ggml.bos_token_id", default: 2)
    // Gemma-4 sets eos_token_id to 106 (<end_of_turn>) since chat-tuned builds
    // use that as the effective stopping token. Keeps turn-boundary generation
    // behavior aligned with HF's generate() default.
    let eosTokenId: UInt32 = readU32("tokenizer.ggml.eos_token_id", default: 106)
    var addBosToken = true
    if let v = g.metadata["tokenizer.ggml.add_bos_token"] as? Bool { addBosToken = v }
    var vocabTokens: [String] = []
    if let tarr = g.metadata["tokenizer.ggml.tokens"] as? [Any] {
        vocabTokens.reserveCapacity(tarr.count)
        for t in tarr { vocabTokens.append((t as? String) ?? "") }
    }
    var merges: [String] = []
    if let marr = g.metadata["tokenizer.ggml.merges"] as? [Any] {
        merges.reserveCapacity(marr.count)
        for m in marr { merges.append((m as? String) ?? "") }
    }
    print(String(format: "  tokenizer: bos=%d eos=%d add_bos=%@ vocab=%d tokens merges=%d",
                 Int(bosTokenId), Int(eosTokenId), addBosToken ? "true" : "false",
                 vocabTokens.count, merges.count))

    // ---------- Sanity spot-check on layer 0's attnQ swizzle ----------
    let L0 = layers[0]
    let rawQ = try g.makeMetalBuffer("blk.0.attn_q.weight", device: device)
    let rawSp = rawQ.contents()
    let swDp = L0.attnQ.contents()
    var match = true
    for byte in 0..<BLK {
        let rawB = rawSp.load(fromByteOffset: byte, as: UInt8.self)
        let swB  = swDp.load(fromByteOffset: byte, as: UInt8.self)
        if rawB != swB { match = false; break }
    }
    print("  spot-check: L0 attn_q block[n=0,kb=0] \(match ? "✓ matches" : "✗ MISMATCH") post-swizzle")

    print(String(format: "  == TOTAL load: %.2f sec ==", Date().timeIntervalSince(t0)))
    return LmWeights(layers: layers, embedTable: embedTable, unembedW: unembedW,
                      outputNorm: outputNorm, embedScaleBuf: embedScaleBuf,
                      K_caches: K_caches, V_caches: V_caches,
                      bosTokenId: bosTokenId, eosTokenId: eosTokenId,
                      addBosToken: addBosToken, vocabTokens: vocabTokens, merges: merges)
}

// Reset shared paged-attn state for step 0. After kv_write of position=0,
// the cache contains exactly 1 token, so attention reads with k_len=1
// (num_pages=1). Each slot b gets a disjoint strip of phys pages starting
// at b*MAX_PAGES_PER_SLOT; the first KV write lands in page b,0 for each layer.
func initLmState(bos: UInt32) {
    let posP = positions.contents().bindMemory(to: UInt32.self, capacity: B)
    let tokP = input_tokens.contents().bindMemory(to: UInt32.self, capacity: B)
    let npsP = num_pages_slide.contents().bindMemory(to: UInt32.self, capacity: B)
    let npfP = num_pages_full.contents().bindMemory(to: UInt32.self, capacity: B)
    let klsP = k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
    let klfP = k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
    for b in 0..<B {
        posP[b] = 0; tokP[b] = bos
        klsP[b] = 1; klfP[b] = 1
        npsP[b] = 1; npfP[b] = 1
    }
    let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
    for b in 0..<B {
        for p in 0..<MAX_PAGES_PER_SLOT {
            btP[b * MAX_PAGES_PER_SLOT + p] = UInt32(b * MAX_PAGES_PER_SLOT + p) % UInt32(TOTAL_PAGES)
        }
    }
    if USE_FLEX_ATTN {
        precomputeFlexBlockMaskSlide(slidingWindow: SLIDING_WINDOW)
        precomputeFlexBlockMaskFull()
    }
}

// Advance AR state by one token: next input_tokens = nextTokens,
// position → position+1, k_len includes the KV row this next step will write.
func advanceLmState(nextTokens: [UInt32]) {
    let posP = positions.contents().bindMemory(to: UInt32.self, capacity: B)
    let tokP = input_tokens.contents().bindMemory(to: UInt32.self, capacity: B)
    let npsP = num_pages_slide.contents().bindMemory(to: UInt32.self, capacity: B)
    let npfP = num_pages_full.contents().bindMemory(to: UInt32.self, capacity: B)
    let klsP = k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
    let klfP = k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
    for b in 0..<B {
        posP[b] &+= 1
        tokP[b] = nextTokens[b]
        // Let k_len grow freely; the slide kernel applies the SW mask at
        // softmax time and skips entire pages that fall before the window.
        // num_pages still tracks total pages so the kernel can short-circuit.
        let newKLS = Int(klsP[b]) + 1
        let newKLF = Int(klfP[b]) + 1
        klsP[b] = UInt32(newKLS)
        klfP[b] = UInt32(newKLF)
        npsP[b] = UInt32((newKLS + PAGE_SLIDE - 1) / PAGE_SLIDE)
        npfP[b] = UInt32((newKLF + PAGE_FULL - 1) / PAGE_FULL)
    }
    if USE_FLEX_ATTN {
        precomputeFlexBlockMaskSlide(slidingWindow: SLIDING_WINDOW)
        precomputeFlexBlockMaskFull()
    }
}

// Diagnostic: count finite / NaN / Inf / min / max in an fp16 buffer region.
@inline(__always)
func scanHalfBuf(_ buf: MTLBuffer, count: Int) -> (nan: Int, inf: Int, mn: Float, mx: Float) {
    let p = buf.contents().assumingMemoryBound(to: Float16.self)
    var nNaN = 0, nInf = 0
    var mn = Float.infinity, mx = -Float.infinity
    for i in 0..<count {
        let v = Float(p[i])
        if v.isNaN { nNaN += 1 }
        else if !v.isFinite { nInf += 1 }
        else {
            if v < mn { mn = v }
            if v > mx { mx = v }
        }
    }
    return (nNaN, nInf, mn, mx)
}

// Build a partial forward that stops after `stopAfterLayer` transformer
// layers (inclusive): embed+scale → layer[0..stopAfterLayer] → (no final
// norm/unembed). The `hidden` buffer holds the residual stream at the end;
// inspect it to localize NaN/Inf introduction. stopAfterLayer=-1 means run
// only embed+scale (no layers).
func buildPartialStepCB(_ w: LmWeights, stopAfterLayer: Int) -> MTLCommandBuffer {
    let cb = queue.makeCommandBuffer()!
    encEmbed(cb, embedTable: w.embedTable)
    encScaleByScalar(cb, x: hidden, scalar: w.embedScaleBuf, N: HIDDEN, numVecs: B)
    if stopAfterLayer < 0 { return cb }

    for L in 0...stopAfterLayer {
        let lw = w.layers[L]
        let isFull = lw.isFull
        let H = isFull ? FULL_H : SLIDE_H
        let KV_H = lw.KV_H
        let HD = lw.HD
        let theta: Float = isFull ? 1_000_000 : 10_000
        let rotary = isFull ? (FULL_HD / 4) : SLIDE_HD
        let q_out = isFull ? q_full_out : q_slide_out
        let k_out = isFull ? k_full_out : k_slide_out
        let v_out = isFull ? v_full_out : v_slide_out
        let Kc = w.K_caches[L]; let Vc = w.V_caches[L]

        let Wv = lw.attnV ?? lw.attnK
        encGemvQ80V6RmsnormQKV(cb, x: hidden, gammaBuf: lw.attnNorm,
                                Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                outQ: q_out, outK: k_out, outV: v_out,
                                Din: HIDDEN, DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD)
        encRMSNormG(cb, x: q_out, gammaBuf: lw.attnQNorm, out: q_out, D: HD, numVecs: B * H)
        encRope(cb, q_out, H: H, D: HD, rotary: rotary, theta: theta)
        encRMSNormG(cb, x: k_out, gammaBuf: lw.attnKNorm, out: k_out, D: HD, numVecs: B * KV_H)
        encRope(cb, k_out, H: KV_H, D: HD, rotary: rotary, theta: theta)
        encRMSNormNoScale(cb, x: v_out, out: v_out, D: HD, numVecs: B * KV_H)
        let pg = isFull ? PAGE_FULL : PAGE_SLIDE
        encKVWrite(cb, K: k_out, V: v_out, Kc: Kc, Vc: Vc, H: KV_H, D: HD, page: pg)
        let npBuf = isFull ? num_pages_full : num_pages_slide
        let klBuf = isFull ? k_len_full : k_len_slide
        if isFull && USE_FLEX_ATTN {
            encFlexAttnFullV0(cb, Q: q_out, O: attn_out, Kc: Kc, Vc: Vc,
                              kLenBuf: klBuf, H_Q: H, H_KV: KV_H, D: HD)
        } else if isFull {
            encPagedAttnFullGqa(cb, Q: q_out, O: attn_out, Kc: Kc, Vc: Vc,
                                numPagesBuf: npBuf, kLenBuf: klBuf, H_Q: H, H_KV: KV_H, D: HD)
        } else if USE_FLEX_ATTN {
            encFlexAttnSlideV0(cb, Q: q_out, O: attn_out, Kc: Kc, Vc: Vc,
                               kLenBuf: klBuf, H_Q: H, H_KV: KV_H, D: HD)
        } else {
            encPagedAttnSlideGqa(cb, Q: q_out, O: attn_out, Kc: Kc, Vc: Vc,
                                 numPagesBuf: npBuf, kLenBuf: klBuf, H_Q: H, H_KV: KV_H, D: HD)
        }
        encGemvQ80V6(cb, attn_out, lw.attnOut, mlp_out, Din: H * HD, Dout: HIDDEN)
        encRmsNormAdd(cb, x: mlp_out, gammaBuf: lw.postAttnNorm,
                      residual: hidden, out: hidden, N: HIDDEN, numVecs: B)
        encGemvQ80V6RmsnormGateUp(cb, x: hidden, gammaBuf: lw.ffnNorm,
                                   Wg: lw.ffnGate, Wu: lw.ffnUp, fusedOut: shrd_gate_up_fused,
                                   Din: HIDDEN, Dout: SHARED_INT)
        encMoeGeluMulFused(cb, fused: shrd_gate_up_fused, out: shrd_gate, N_half: SHARED_INT, numSlots: B)
        encGemvQ80V6(cb, shrd_gate, lw.ffnDown, mlp_out, Din: SHARED_INT, Dout: HIDDEN)
        encRMSNormG(cb, x: mlp_out, gammaBuf: lw.postFfn1Norm, out: mlp_out, D: HIDDEN, numVecs: B)
        encRouterPreNorm(cb, x: hidden, per_dim_scale: lw.routerScale, out: hidden_norm)
        encGemvV5(cb, hidden_norm, lw.routerW, router_lg, Din: HIDDEN, Dout: E_EXP)
        encSoftmaxTopk(cb, expertScaleBuf: lw.expertScale)
        encRouteCompact(cb)
        encRMSNormG(cb, x: hidden, gammaBuf: lw.preFfn2Norm, out: hidden_norm, D: HIDDEN, numVecs: B)
        encMoeGemvQ4K(cb, hidden_norm, lw.moeGateUp, gate_up_fused,
                      Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: E_EXP, useV6: true)
        encMoeGeluMulFused(cb, fused: gate_up_fused, out: gate_proj, N_half: MOE_INT, numSlots: TOTAL_SLOTS)
        encMoeGemvQ51(cb, gate_proj, lw.moeDown, moe_down_out,
                      Din: MOE_INT, Dout: HIDDEN, numActive: E_EXP, useV6: true)
        encMoeCombineWrite(cb, to: moe_sum)
        encRMSNormG(cb, x: moe_sum, gammaBuf: lw.postFfn2Norm, out: moe_sum, D: HIDDEN, numVecs: B)
        encBufferCopy(cb, src: mlp_out, dst: ffn_combined, bytes: B * HIDDEN * 2)
        encAddInplace(cb, dst: ffn_combined, src: moe_sum, N: HIDDEN, numVecs: B)
        encRmsNormAddScale(cb, x: ffn_combined, gammaBuf: lw.postFfnNorm,
                           residual: hidden, scalar: lw.layerOutputScale,
                           out: hidden, N: HIDDEN, numVecs: B)
    }
    return cb
}

// Run N forward steps with real weights, greedy-sample each slot's argmax,
// and feed it back as the next token. Reports finite-check + top-5 tokens
// after step 0; wall/GPU/encoding timing across all N measured steps.
func runLmForwardBench(ggufPath: String) {
    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { fail("loadLmWeights: \(error)") }
    print("")

    // Initialize per-slot state to position 0 with BOS as the input token.
    initLmState(bos: w.bosTokenId)

    // Diagnostic: report hidden-stream finiteness at embed+scale and after
    // every layer. First NaN/Inf pinpoints the broken stage.
    if ProcessInfo.processInfo.environment["LM_DIAG"] != nil {
        print("  [diag] hidden stream scan (B*HIDDEN = \(B * HIDDEN) half-elts per snapshot)")
        for stop in -1..<NUM_LAYERS {
            initLmState(bos: w.bosTokenId)
            let cb = buildPartialStepCB(w, stopAfterLayer: stop)
            cb.commit(); cb.waitUntilCompleted()
            if let err = cb.error { print("  GPU at L\(stop): \(err)"); break }
            let s = scanHalfBuf(hidden, count: B * HIDDEN)
            let label = (stop < 0 ? "embed+scale" : "after L\(stop)")
                .padding(toLength: 14, withPad: " ", startingAt: 0)
            print("    \(label)  NaN=\(s.nan)  Inf=\(s.inf)  "
                  + String(format: "min=%.3f  max=%.3f", s.mn, s.mx))
            if s.nan > 0 || s.inf > 0 {
                // Dump all intermediate buffers from the failing layer to
                // localize which sub-step went bad.
                print("    [drill-down after L\(stop)]")
                let probes: [(String, MTLBuffer, Int)] = [
                    ("q_slide_out", q_slide_out, B * SLIDE_H * SLIDE_HD),
                    ("k_slide_out", k_slide_out, B * SLIDE_KV_H * SLIDE_HD),
                    ("v_slide_out", v_slide_out, B * SLIDE_KV_H * SLIDE_HD),
                    ("q_full_out",  q_full_out,  B * FULL_H * FULL_HD),
                    ("k_full_out",  k_full_out,  B * FULL_KV_H * FULL_HD),
                    ("v_full_out",  v_full_out,  B * FULL_KV_H * FULL_HD),
                    ("attn_out",    attn_out,    B * max(SLIDE_H*SLIDE_HD, FULL_H*FULL_HD)),
                    ("mlp_out",     mlp_out,     B * HIDDEN),
                    ("moe_sum",     moe_sum,     B * HIDDEN),
                    ("ffn_combined",ffn_combined,B * HIDDEN),
                    ("router_lg",   router_lg,   B * E_EXP),
                    ("shrd_gate_up_fused", shrd_gate_up_fused, B * 2 * SHARED_INT),
                    ("shrd_gate",   shrd_gate,   B * SHARED_INT),
                    ("gate_up_fused", gate_up_fused, TOTAL_SLOTS * MOE_FUSED_DOUT),
                    ("gate_proj",   gate_proj,   TOTAL_SLOTS * MOE_INT),
                    ("moe_down_out",moe_down_out,TOTAL_SLOTS * HIDDEN),
                ]
                for (name, buf, n) in probes {
                    let ss = scanHalfBuf(buf, count: n)
                    let lab = name.padding(toLength: 20, withPad: " ", startingAt: 0)
                    print("      \(lab)  NaN=\(ss.nan)  Inf=\(ss.inf)  "
                          + String(format: "min=%.3f  max=%.3f", ss.mn, ss.mx))
                }
                // Find first few NaN indices in shrd_gate and print their
                // corresponding (gate, up) inputs from shrd_gate_up_fused.
                let sp = shrd_gate.contents().assumingMemoryBound(to: Float16.self)
                let fp = shrd_gate_up_fused.contents().assumingMemoryBound(to: Float16.self)
                let N_half = SHARED_INT
                print("      [shrd_gate NaN locations → (gate, up) inputs]")
                var shown = 0
                for b in 0..<B {
                    for i in 0..<N_half where shown < 10 {
                        let v = Float(sp[b * N_half + i])
                        if v.isNaN {
                            let g = Float(fp[b * (2 * N_half) + i])
                            let u = Float(fp[b * (2 * N_half) + N_half + i])
                            print(String(format: "        b=%d i=%d  gate=%.3f  up=%.3f",
                                         b, i, g, u))
                            shown += 1
                        }
                    }
                }
                break
            }
        }
        print("")
        initLmState(bos: w.bosTokenId)
    }

    // Warmup.
    let WARMUP = 2
    for _ in 0..<WARMUP {
        let cb = buildStepCB(w)
        cb.commit(); cb.waitUntilCompleted()
        if let err = cb.error { fail("GPU (warmup): \(err)") }
        // warmup steps consume the same input_tokens repeatedly; don't advance state
        // (re-init to keep k_len comparable across the measured loop).
        initLmState(bos: w.bosTokenId)
    }

    // First real step: read logits back, check finite, print top-5 for slot 0.
    let cbStep0 = buildStepCB(w)
    cbStep0.commit(); cbStep0.waitUntilCompleted()
    if let err = cbStep0.error { fail("GPU (step 0): \(err)") }
    inspectLogitsTop5(w: w, label: "step 0 (input=BOS)")

    // Advance state by one token (greedy argmax per slot), then run the
    // measured loop. We reset to match step 0's conditions so timing
    // compares apples-to-apples.
    initLmState(bos: w.bosTokenId)
    var wallTimes: [Double] = []
    var gpuTimes: [Double] = []
    var encodeTimes: [Double] = []
    let MEASURED = 10
    var nextToks = [UInt32](repeating: w.bosTokenId, count: B)
    for step in 0..<MEASURED {
        let t0 = Date()
        let cb = buildStepCB(w)
        let tEnc = Date().timeIntervalSince(t0)
        cb.commit(); cb.waitUntilCompleted()
        let tWall = Date().timeIntervalSince(t0)
        if let err = cb.error { fail("GPU (step \(step)): \(err)") }
        wallTimes.append(tWall)
        gpuTimes.append(cb.gpuEndTime - cb.gpuStartTime)
        encodeTimes.append(tEnc)
        // Greedy sample from logits and feed back.
        nextToks = greedyArgmaxPerSlot(w: w)
        advanceLmState(nextTokens: nextToks)
    }

    let minT = wallTimes.min()!
    let medT = wallTimes.sorted()[wallTimes.count / 2]
    let minGpu = gpuTimes.min()!
    let minEnc = encodeTimes.min()!
    print("")
    print(String(format: "=== Per-step wall clock (B=%d, one CB, %d dispatches) ===",
                 B, NUM_LAYERS * 18 + 5))
    print(String(format: "  wall min: %.2f ms | median: %.2f ms", minT * 1000, medT * 1000))
    print(String(format: "  CPU encoding: %.2f ms", minEnc * 1000))
    print(String(format: "  GPU execute:  %.2f ms", minGpu * 1000))
    print("")
    print(String(format: "  aggregate throughput (min): %.0f tok/s", Double(B) / minT))
    print(String(format: "  per-stream throughput (min): %.0f tok/s", 1.0 / minT))
    print("")

    // Final snapshot of slot 0's logits so we can eyeball the AR trajectory.
    inspectLogitsTop5(w: w, label: "step \(MEASURED - 1)")
}

// Read `logits` back from GPU, check finite, print top-5 tokens for slot 0
// with decoded strings from GGUF tokenizer vocab.
func inspectLogitsTop5(w: LmWeights, label: String) {
    let p = logits.contents().assumingMemoryBound(to: Float16.self)
    var nNaN = 0, nInf = 0
    var gMin = Float.infinity, gMax = -Float.infinity
    for i in 0..<(B * VOCAB) {
        let v = Float(p[i])
        if v.isNaN { nNaN += 1 }
        else if !v.isFinite { nInf += 1 }
        else {
            if v < gMin { gMin = v }
            if v > gMax { gMax = v }
        }
    }
    print("  [\(label)] logits: NaN=\(nNaN) Inf=\(nInf) min=\(String(format: "%.3f", gMin)) max=\(String(format: "%.3f", gMax))")
    // Top-5 for slot 0.
    var idxVal: [(Int, Float)] = []
    idxVal.reserveCapacity(5)
    for v in 0..<VOCAB {
        let val = Float(p[0 * VOCAB + v])
        if !val.isFinite { continue }
        if idxVal.count < 5 {
            idxVal.append((v, val))
            idxVal.sort { $0.1 > $1.1 }
        } else if val > idxVal[4].1 {
            idxVal[4] = (v, val)
            idxVal.sort { $0.1 > $1.1 }
        }
    }
    var line = "  [\(label)] slot0 top-5:"
    for (id, val) in idxVal {
        let tok = (id < w.vocabTokens.count ? w.vocabTokens[id] : "<oov>")
            .replacingOccurrences(of: "\n", with: "\\n")
        line += String(format: "  %d=%@(%.2f)", id, tok, val)
    }
    print(line)
}

// Greedy argmax per batch slot. B values read back from the `logits` buffer.
func greedyArgmaxPerSlot(w: LmWeights) -> [UInt32] {
    let p = logits.contents().assumingMemoryBound(to: Float16.self)
    var out = [UInt32](repeating: 0, count: B)
    for b in 0..<B {
        var bestI = 0
        var bestV: Float = -Float.infinity
        for v in 0..<VOCAB {
            let val = Float(p[b * VOCAB + v])
            if val > bestV { bestV = val; bestI = v }
        }
        out[b] = UInt32(bestI)
    }
    return out
}

// ====================================================================
// GGUF real-weight pipeline. Two modes gated on env vars:
//   GGUF_VALIDATE=<path> — single-tensor kernel validation (tight loop,
//     load Q8_0 blk.0.attn_q, repack, run v6 kernel, compare CPU reference)
//   GGUF_PATH=<path>     — full real-weight forward: load all 30 layers'
//     weights, run ONE forward step, dump logits
// ====================================================================

// ---------- Env-var-driven test harnesses ----------
// All harness bodies live in harness.swift. main.swift is just a
// dispatcher that invokes the right harness based on which env vars
// the user set. Multiple harnesses can fire in one binary invocation.
//
// Wrapped in runEnvDrivenDemos() so the library target (libgemma_metal
// .dylib, built for the Python FFI bridge) can compile all of main.swift
// without executing any of this on dylib load. The executable target
// (forward_graph) calls the function at the bottom of this file.
func runEnvDrivenDemos() {
if let stPath = ProcessInfo.processInfo.environment["VISION_ST"],
   ProcessInfo.processInfo.environment["VISION_FORWARD"] == nil,
   ProcessInfo.processInfo.environment["VISION_ASPECT_DIR"] == nil,
   ProcessInfo.processInfo.environment["VISION_SWEEP_DIR"] == nil {
    // VISION_ST alone → patch-embed smoke test. If the user passed
    // VISION_FORWARD/VISION_ASPECT_DIR/VISION_SWEEP_DIR, let those drive.
    runVisionPatchEmbedSmokeTest(stPath: stPath)
}
if let framePath = ProcessInfo.processInfo.environment["VISION_FORWARD"],
   let stPath = ProcessInfo.processInfo.environment["VISION_ST"] {
    runVisionEndToEndForward(framePath: framePath, stPath: stPath)
}
if let batchDir = ProcessInfo.processInfo.environment["VISION_BATCH_DIR"],
   let stPath = ProcessInfo.processInfo.environment["VISION_ST"] {
    runVisionBatchForward(batchDir: batchDir, stPath: stPath)
}
if let batchDir = ProcessInfo.processInfo.environment["VISION_CONCURRENT_DIR"],
   let stPath = ProcessInfo.processInfo.environment["VISION_ST"] {
    let n = Int(ProcessInfo.processInfo.environment["VISION_CONCURRENT_N"] ?? "2") ?? 2
    runVisionConcurrentQueues(batchDir: batchDir, stPath: stPath, nQueues: n)
}
if let aspectDir = ProcessInfo.processInfo.environment["VISION_ASPECT_DIR"],
   let stPath = ProcessInfo.processInfo.environment["VISION_ST"] {
    runVisionAspectSweep(aspectDir: aspectDir, stPath: stPath)
}
if let sweepDir = ProcessInfo.processInfo.environment["VISION_SWEEP_DIR"],
   let stPath = ProcessInfo.processInfo.environment["VISION_ST"],
   let refPath = ProcessInfo.processInfo.environment["VISION_REF"],
   let orderPath = ProcessInfo.processInfo.environment["VISION_SWEEP_ORDER"] {
    runVisionMultiFrameSweep(sweepDir: sweepDir, stPath: stPath,
                              refPath: refPath, orderPath: orderPath)
}
if let stPath = ProcessInfo.processInfo.environment["VISION_LOAD"] {
    runVisionWeightLoadSmokeTest(stPath: stPath)
}
if let png = ProcessInfo.processInfo.environment["VISION_PREPROCESS"] {
    runVisionPreprocessSmokeTest(png: png)
}
if ProcessInfo.processInfo.environment["PAGED_ATTN_TEST"] != nil {
    runPagedAttnHarness()
}
let isDumpRun = ProcessInfo.processInfo.environment["LM_DUMP_LAYERS"] != nil
    || ProcessInfo.processInfo.environment["LM_DUMP_L0_INTERNALS"] != nil
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_KL_REF"] == nil,
   ProcessInfo.processInfo.environment["LM_PREFILL_VALIDATE"] == nil,
   ProcessInfo.processInfo.environment["LM_GENERATE"] == nil,
   ProcessInfo.processInfo.environment["LM_MULTISESSION"] == nil,
   !isDumpRun {
    // GGUF_PATH alone → LM forward benchmark. If LM_KL_REF, LM_PREFILL_VALIDATE,
    // LM_GENERATE, LM_MULTISESSION, or any dump flag is also set, let those
    // harnesses drive (all reuse loadLmWeights).
    runGgufPathHarness(ggufPath: ggufPath)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   let prompt = ProcessInfo.processInfo.environment["LM_GENERATE"] {
    let maxN = Int(ProcessInfo.processInfo.environment["LM_GENERATE_MAX"] ?? "64") ?? 64
    let eos = (ProcessInfo.processInfo.environment["LM_GENERATE_EOS"]).flatMap { UInt32($0) }
    runLmGenerate(ggufPath: ggufPath, prompt: prompt, maxNewTokens: maxN, eos: eos)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   let prompts = ProcessInfo.processInfo.environment["LM_MULTISESSION"] {
    let maxN = Int(ProcessInfo.processInfo.environment["LM_MULTISESSION_MAX"] ?? "32") ?? 32
    runLmMultisession(ggufPath: ggufPath, promptsStr: prompts, maxNewPerSession: maxN)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_MULTITURN_DEMO"] != nil {
    let turnsStr = ProcessInfo.processInfo.environment["LM_MULTITURN_TURNS"]
    let maxN = Int(ProcessInfo.processInfo.environment["LM_MULTITURN_MAX_PER_TURN"] ?? "24") ?? 24
    runLmMultiturnDemo(ggufPath: ggufPath, turnsStr: turnsStr, maxPerTurn: maxN)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_COMPOSITE_DEMO"] != nil {
    let n = Int(ProcessInfo.processInfo.environment["LM_COMPOSITE_N"] ?? "4") ?? 4
    let maxN = Int(ProcessInfo.processInfo.environment["LM_COMPOSITE_MAX_PER_TURN"] ?? "24") ?? 24
    runLmCompositeDemo(ggufPath: ggufPath, nUsers: n, maxPerTurn: maxN)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   let stPath = ProcessInfo.processInfo.environment["VISION_ST"],
   let imagePath = ProcessInfo.processInfo.environment["LM_MULTIMODAL"] {
    let prefix = ProcessInfo.processInfo.environment["LM_MULTIMODAL_PREFIX"]
        ?? "<|turn>user\n"
    let suffix = ProcessInfo.processInfo.environment["LM_MULTIMODAL_SUFFIX"]
        ?? "\nDescribe this image in one short sentence.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    let maxN = Int(ProcessInfo.processInfo.environment["LM_MULTIMODAL_MAX"] ?? "48") ?? 48
    runLmMultimodal(ggufPath: ggufPath, stPath: stPath, imagePath: imagePath,
                     prefix: prefix, suffix: suffix, maxNew: maxN)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   let refDir = ProcessInfo.processInfo.environment["LM_KL_REF"],
   !isDumpRun {
    let tag = ProcessInfo.processInfo.environment["LM_KL_TAG"] ?? "hello"
    runLmKLHarness(ggufPath: ggufPath, refDir: refDir, tag: tag)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   let tag = ProcessInfo.processInfo.environment["LM_PREFILL_VALIDATE"] {
    let refDir = ProcessInfo.processInfo.environment["LM_KL_REF"]
        ?? "/Users/mdot/metal-microbench/test_data/reference"
    runLmPrefillValidate(ggufPath: ggufPath, refDir: refDir, tag: tag)
}
if let outDir = ProcessInfo.processInfo.environment["FLEX_ATTN_TEST"] {
    runFlexAttnSlideV1Test(outDir: outDir)
    runFlexAttnFullPrefillTest(outDir: outDir)
}
if ProcessInfo.processInfo.environment["ATTN_BENCH"] != nil {
    runAttnBench()
}
if ProcessInfo.processInfo.environment["SHARED_PREFIX_SMOKE"] != nil {
    runSharedPrefixSmoke()
}
if ProcessInfo.processInfo.environment["SHARED_PREFIX_PARITY"] != nil {
    runSharedPrefixParity()
}
if ProcessInfo.processInfo.environment["SHARED_PREFIX_PARITY_FULL"] != nil {
    runSharedPrefixParityFull()
}
if ProcessInfo.processInfo.environment["KV_VIZ"] != nil {
    runKvVisualizer()
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_SHARED_PREFIX_DEMO"] != nil {
    let maxN = Int(ProcessInfo.processInfo.environment["LM_SHARED_PREFIX_MAX"] ?? "16") ?? 16
    runLmSharedPrefixDemo(ggufPath: ggufPath, maxNewPerSession: maxN)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_BRANCH_DEMO"] != nil {
    let maxN = Int(ProcessInfo.processInfo.environment["LM_BRANCH_MAX"] ?? "16") ?? 16
    runLmBranchDemo(ggufPath: ggufPath, maxNewPerBranch: maxN)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_PAUSE_DEMO"] != nil {
    runLmPauseResumeDemo(ggufPath: ggufPath)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   let outDir = ProcessInfo.processInfo.environment["LM_DUMP_EXPERT_W"] {
    let id = Int(ProcessInfo.processInfo.environment["LM_DUMP_EXPERT_ID"] ?? "52") ?? 52
    runDumpExpertWeights(ggufPath: ggufPath, expertId: id, outDir: outDir)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"], isDumpRun,
   ProcessInfo.processInfo.environment["LM_DUMP_EXPERT_W"] == nil {
    // LM_DUMP_LAYERS=<dir> and/or LM_DUMP_L0_INTERNALS=<dir> activate dumps.
    // Tokens come from LM_KL_REF's lm_<tag>_tokens.npy (reuses oracle files).
    let refDir = ProcessInfo.processInfo.environment["LM_KL_REF"]
        ?? "/Users/mdot/metal-microbench/test_data/reference"
    let tag = ProcessInfo.processInfo.environment["LM_KL_TAG"] ?? "hello"
    let outDir = ProcessInfo.processInfo.environment["LM_DUMP_LAYERS"]
        ?? ProcessInfo.processInfo.environment["LM_DUMP_L0_INTERNALS"]!
    runLmLayerDump(ggufPath: ggufPath, refDir: refDir, tag: tag, outDir: outDir)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_VALIDATE"] {
    runGgufValidateHarness(ggufPath: ggufPath)
}
} // end runEnvDrivenDemos

// Run all the one-time top-level side effects (banners + active_exp fill)
// that used to be top-level statements. The executable (main.swift) calls
// this before runEnvDrivenDemos(); the library (ffi.swift/gemma_init) calls
// it right after weights load. Idempotent: safe to call multiple times.
private var _bootstrapped = false
func bootstrapGlobalState() {
    if _bootstrapped { return }
    _bootstrapped = true
    print("device: \(device.name)")
    print("")
    print("config: B=\(B), 30 layers, hidden=\(HIDDEN), experts=\(E_EXP) top_k=\(TOPK)")
    print("")
    let ap = active_exp.contents().bindMemory(to: UInt32.self, capacity: E_EXP)
    for e in 0..<E_EXP { ap[e] = UInt32(e) }
}
