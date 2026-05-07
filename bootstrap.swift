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

// Vision→LM cross-queue ordering primitive. Vision pad-blit CBs encode
// signal-event with a monotonic ticket; LM pre-prefill CBs encode
// wait-event for that ticket. GPU waits, CPU never blocks. See
// notes/engine_debloat.md.
let gVisionEvent: MTLSharedEvent = device.makeSharedEvent()!
let gVisionEventCounterLock = NSLock()
private var _gVisionEventCounter: UInt64 = 0
func nextVisionEventTicket() -> UInt64 {
    gVisionEventCounterLock.lock(); defer { gVisionEventCounterLock.unlock() }
    _gVisionEventCounter += 1
    return _gVisionEventCounter
}

let visionSoftsCopyFp32PSO = pso("vision_softs_copy_fp32_to_fp16")

let rmsPSO          = pso("rms_norm")
let rmsNoScalePSO   = pso("rms_norm_noscale")
let scaleByScalarPSO = pso("scale_by_scalar")
let routerPreNormPSO = pso("router_prenorm_scale")
let addInplacePSO   = pso("add_inplace")
let addScaledCvecPSO = pso("add_scaled_cvector_fp16")
let addScaledCvecPrefillPSO = pso("add_scaled_cvector_prefill_fp16")
let projectCvecPSO          = pso("project_cvector_fp16")
let projectCvecSlotPSO      = pso("project_cvector_slot_fp16")
let projectCvecPrefillPSO   = pso("project_cvector_prefill_fp16")
let transportCvecSlotPSO    = pso("transport_cvector_slot_fp16")
let transportCvecPrefillPSO = pso("transport_cvector_prefill_fp16")
let orthogWritePSO          = pso("orthogonalize_write_fp16")
let measureDotPSO    = pso("measure_dot_fp16")
let geluMulPSO      = pso("gelu_mul_inplace")
let moeCombineWritePSO = pso("moe_combine_write")
let gemvV5PSO      = pso("dense_gemv_v5")
let gemvV4PSO      = pso("dense_gemv_v4")
let gemvV4PPSO     = pso("dense_gemv_v4_p")
let gemvV4SoftcapPSO = pso("dense_gemv_v4_softcap")
let gemvI8V4PSO    = pso("dense_gemv_i8w_v4")
let ropePSO        = pso("rope_half")
let kvwPSO         = pso("kv_write")
let fakeAttnPSO    = pso("fake_attention")
let topkPSO        = pso("softmax_topk")
let routeCompactPSO = pso("route_compact")
let moePSO         = pso("moe_gemv_v3")
let moeQ4PSO       = pso("moe_gemv_q4_v3")
let moeQ4KPSO      = pso("moe_gemv_q4k_v3")
let moeQ4KV4PSO    = pso("moe_gemv_q4k_v4")
let moeQ51V4PSO    = pso("moe_gemv_q5_1_v4")
let moeQ51V6PSO    = pso("moe_gemv_q5_1_v6")
let moeQ4KV6PSO    = pso("moe_gemv_q4k_v6")
let moeQ51V8PSO    = pso("moe_gemv_q5_1_v8")
let moeQ4KV8PSO    = pso("moe_gemv_q4k_v8")
let moeQ51V9PSO    = pso("moe_gemv_q5_1_v9")
let moeQ4KV9PSO    = pso("moe_gemv_q4k_v9")
let moeQ4KV10PSO   = pso("moe_gemv_q4k_v10_fused_gelu")
// MoE Q4_K v11 — templated MAX_SLOTS={1,2,4,8} with scale-hoist + per-pair
// pre-dequant register scratch. Dispatcher picks the right specialization
// from activeB (= the actual chunk size at the typical-routing case).
let moeQ4KV11B1PSO = pso("moe_gemv_q4k_v11_b1")
let moeQ4KV11B2PSO = pso("moe_gemv_q4k_v11_b2")
let moeQ4KV11B4PSO = pso("moe_gemv_q4k_v11_b4")
let moeQ4KV11B8PSO = pso("moe_gemv_q4k_v11_b8")
// MoE Q5_1 v11 — port of Q4K V11's structure to Q5_1's simpler 32-elem
// blocks (single d/m pair, no paired sub-blocks). Down-projection kernel.
let moeQ51V11B1PSO = pso("moe_gemv_q5_1_v11_b1")
let moeQ51V11B2PSO = pso("moe_gemv_q5_1_v11_b2")
let moeQ51V11B4PSO = pso("moe_gemv_q5_1_v11_b4")
let moeQ51V11B8PSO = pso("moe_gemv_q5_1_v11_b8")
// MoE Q4_K v12 — V11 + per-pair private register nibble lookup table.
// Compute-bound dequant rewrite attempt (Item-A precomputed-table
// sub-option from the original kernel-zoo plan).
let moeQ4KV12B1PSO = pso("moe_gemv_q4k_v12_b1")
let moeQ4KV12B2PSO = pso("moe_gemv_q4k_v12_b2")
let moeQ4KV12B4PSO = pso("moe_gemv_q4k_v12_b4")
let moeQ4KV12B8PSO = pso("moe_gemv_q4k_v12_b8")
let denseQ4KV4PSO  = pso("dense_gemv_q4k_v4")
let moeQ40PSO      = pso("moe_gemv_q4_0_v3")
let denseQ40V4PSO  = pso("dense_gemv_q4_0_v4")
let denseQ80V5PSO  = pso("dense_gemv_q8_0_v5")
// Q8_0 GEMV kernel zoo: compile-time fixed B_TILE specializations.
// Scheduler picks by activeB rounded up to nearest power of 2.
let denseQ80BtileB1PSO = pso("dense_gemv_q8_0_btile_b1")
let denseQ80BtileB2PSO = pso("dense_gemv_q8_0_btile_b2")
let denseQ80BtileB4PSO = pso("dense_gemv_q8_0_btile_b4")
let denseQ80BtileB8PSO = pso("dense_gemv_q8_0_btile_b8")
// Q8_0 RMSNorm+QKV templated zoo (staging-based — kept for reference).
// B_TILE=8 omitted; tg-mem h_norm staging at b8 = 45 KB > 32 KB cap.
let denseQ80BtileQkvB1PSO = pso("dense_gemv_q8_0_btile_qkv_b1")
let denseQ80BtileQkvB2PSO = pso("dense_gemv_q8_0_btile_qkv_b2")
let denseQ80BtileQkvB4PSO = pso("dense_gemv_q8_0_btile_qkv_b4")
// Q8_0 RMSNorm+QKV templated zoo (on-the-fly normalize). Tg-mem usage
// drops from h_norm staging's 5.6-22.5 KB to ~600 B (just B_TILE
// inv_rms scalars + partials), so b8 fits cleanly. Cost: per-FMA
// gamma+x re-read from L1/L2 instead of TG-mem h_norm broadcast.
let denseQ80BtileQkvOtfB1PSO = pso("dense_gemv_q8_0_btile_qkv_otf_b1")
let denseQ80BtileQkvOtfB2PSO = pso("dense_gemv_q8_0_btile_qkv_otf_b2")
let denseQ80BtileQkvOtfB4PSO = pso("dense_gemv_q8_0_btile_qkv_otf_b4")
let denseQ80BtileQkvOtfB8PSO = pso("dense_gemv_q8_0_btile_qkv_otf_b8")
// QKV K-tile staging family (Item C). Targets activeB ∈ {2,3,4} where
// neither OTF (per-FMA gamma reads accumulate) nor V6 grid-shrink (small
// batches waste B-grid parallelism) wins cleanly.
let denseQ80BtileQkvTiledB1PSO = pso("dense_gemv_q8_0_btile_qkv_tiled_b1")
let denseQ80BtileQkvTiledB2PSO = pso("dense_gemv_q8_0_btile_qkv_tiled_b2")
let denseQ80BtileQkvTiledB4PSO = pso("dense_gemv_q8_0_btile_qkv_tiled_b4")
let denseQ80BtileQkvTiledB8PSO = pso("dense_gemv_q8_0_btile_qkv_tiled_b8")
// Q8_0 RMSNorm + gate_up templated zoo (otf — same pattern as QKV otf).
let denseQ80BtileGateUpOtfB1PSO = pso("dense_gemv_q8_0_btile_gate_up_otf_b1")
let denseQ80BtileGateUpOtfB2PSO = pso("dense_gemv_q8_0_btile_gate_up_otf_b2")
let denseQ80BtileGateUpOtfB4PSO = pso("dense_gemv_q8_0_btile_gate_up_otf_b4")
let denseQ80BtileGateUpOtfB8PSO = pso("dense_gemv_q8_0_btile_gate_up_otf_b8")
let denseQ80V5RmsPSO = pso("dense_gemv_q8_0_v5_rmsnorm")
let denseQ80V6PSO    = pso("dense_gemv_q8_0_v6")
let denseQ80V6RmsPSO = pso("dense_gemv_q8_0_v6_rmsnorm")
let denseQ80V6RmsQkvPSO = pso("dense_gemv_q8_0_v6_rmsnorm_qkv")
let denseQ80V7RmsQkvPSO = pso("dense_gemv_q8_0_v7_rmsnorm_qkv")
let denseQ80V6RmsGateUpPSO = pso("dense_gemv_q8_0_v6_rmsnorm_gate_up")
// Verbatim simdgroup-matmul port for prefill (replaces GEMV at high Q-batch).
// Reads v6-swizzled Q8_0 weights — same layout as the AR decode kernels.
let prefillMmQ80SwizPSO    = pso("prefill_mm_q8_0_swiz")
let prefillMmQ5KSwizPSO    = pso("prefill_mm_q5_K_swiz")
let prefillMmQ6KSwizPSO    = pso("prefill_mm_q6_K_swiz")
let prefillMmQ51SwizPSO    = pso("prefill_mm_q5_1_swiz")
// MoE simdgroup-matmul ports — slot-flat conventions matching route_compact
// + moe_combine_write. Per-expert v6-swizzled weights, broadcast X for
// gate+up (Q4_K), per-slot X for down (Q5_1).
let prefillMmIdQ4KSwizPSO  = pso("prefill_mm_id_q4K_swiz")
let prefillMmIdQ51SwizPSO  = pso("prefill_mm_id_q5_1_swiz")
let prefillMmIdQ5KSwizPSO  = pso("prefill_mm_id_q5_K_swiz")
let prefillMmIdQ6KSwizPSO  = pso("prefill_mm_id_q6_K_swiz")
let prefillMmIdQ80SwizPSO  = pso("prefill_mm_id_q8_0_swiz")

// F16 weight-streaming kernel zoo. F16 is the IEEE half-precision baseline:
// no block structure, no dequant, plain row-major [Dout, Din] weights. Used
// when the loaded GGUF tensor is fp16 (no quantization). Same dispatch
// shapes as the Q8_0 zoo since only the per-element weight type differs.
let denseF16BtileB1PSO          = pso("dense_gemv_f16_btile_b1")
let denseF16BtileB2PSO          = pso("dense_gemv_f16_btile_b2")
let denseF16BtileB4PSO          = pso("dense_gemv_f16_btile_b4")
let denseF16BtileB8PSO          = pso("dense_gemv_f16_btile_b8")
let denseF16BtileQkvOtfB1PSO    = pso("dense_gemv_f16_btile_qkv_otf_b1")
let denseF16BtileQkvOtfB2PSO    = pso("dense_gemv_f16_btile_qkv_otf_b2")
let denseF16BtileQkvOtfB4PSO    = pso("dense_gemv_f16_btile_qkv_otf_b4")
let denseF16BtileQkvOtfB8PSO    = pso("dense_gemv_f16_btile_qkv_otf_b8")
let denseF16BtileGateUpOtfB1PSO = pso("dense_gemv_f16_btile_gate_up_otf_b1")
let denseF16BtileGateUpOtfB2PSO = pso("dense_gemv_f16_btile_gate_up_otf_b2")
let denseF16BtileGateUpOtfB4PSO = pso("dense_gemv_f16_btile_gate_up_otf_b4")
let denseF16BtileGateUpOtfB8PSO = pso("dense_gemv_f16_btile_gate_up_otf_b8")
let denseF16V6RmsQkvPSO         = pso("dense_gemv_f16_v6_rmsnorm_qkv")
let denseF16V6RmsGateUpPSO      = pso("dense_gemv_f16_v6_rmsnorm_gate_up")
let moeGemvF16V11B1PSO          = pso("moe_gemv_f16_v11_b1")
let moeGemvF16V11B2PSO          = pso("moe_gemv_f16_v11_b2")
let moeGemvF16V11B4PSO          = pso("moe_gemv_f16_v11_b4")
let moeGemvF16V11B8PSO          = pso("moe_gemv_f16_v11_b8")
// F16 MoE-up (slot_token-broadcast convention) PSOs — separate from the
// per-slot down kernels above because moe_up needs slot_token indirection
// when one source token is routed to multiple experts.
let moeGemvF16V11UpB1PSO        = pso("moe_gemv_f16_v11_up_b1")
let moeGemvF16V11UpB2PSO        = pso("moe_gemv_f16_v11_up_b2")
let moeGemvF16V11UpB4PSO        = pso("moe_gemv_f16_v11_up_b4")
let moeGemvF16V11UpB8PSO        = pso("moe_gemv_f16_v11_up_b8")
let prefillMmF16SwizPSO         = pso("prefill_mm_f16_swiz")
let prefillMmIdF16SwizPSO       = pso("prefill_mm_id_f16_swiz")
// F16 moe-up prefill (slot_token-broadcast convention) — separate PSO from
// the down kernel above because moe_up needs slot_token at buffer(1) and
// indexes X via slot_token[s_flat] rather than s_flat directly. Sharing
// the down kernel for moe_up causes off-by-one buffer bindings AND
// per-slot indexing where slot_token-broadcast is required.
let prefillMmIdF16UpSwizPSO     = pso("prefill_mm_id_f16_up_swiz")

// AR-decode (single-template) GEMV PSOs for non-Q8_0 dense + non-Q4_K/Q5_1 MoE.
// Used by encDenseGemvAR / encMoeUpGemvAR / encMoeDownGemvAR when the format
// LUT lacks a templated/fused fast path. These are the type-completeness
// fallbacks: any format that has a kernel here can be dispatched correctly,
// even if not at peak throughput.
let denseGemvQ5KV4PSO      = pso("dense_gemv_q5_K_v4")
let denseGemvQ6KV4PSO      = pso("dense_gemv_q6_K_v4")
let denseGemvQ51V4PSO      = pso("dense_gemv_q5_1_v4")
let moeGemvQ5KV6PSO        = pso("moe_gemv_q5_K_v6")
let moeGemvQ6KV6PSO        = pso("moe_gemv_q6_K_v6")
let moeGemvQ80V6PSO        = pso("moe_gemv_q8_0_v6")

// Q5_K / Q6_K / Q5_1 dense AR btile zoo — structural mirror of Q8_0's
// dense_gemv_q8_0_btile_b{1,2,4,8}. Same template-on-B_TILE pattern + 4-SG
// split-K reduction; only the per-element dequant differs. Used by
// encDenseGemvAR via the format LUT to put non-Q8_0 dense AR at the same
// performance ceiling as Q8_0 (modulo per-element dequant ALU cost).
let denseGemvQ5KBtileB1PSO = pso("dense_gemv_q5_K_btile_b1")
let denseGemvQ5KBtileB2PSO = pso("dense_gemv_q5_K_btile_b2")
let denseGemvQ5KBtileB4PSO = pso("dense_gemv_q5_K_btile_b4")
let denseGemvQ5KBtileB8PSO = pso("dense_gemv_q5_K_btile_b8")
let denseGemvQ6KBtileB1PSO = pso("dense_gemv_q6_K_btile_b1")
let denseGemvQ6KBtileB2PSO = pso("dense_gemv_q6_K_btile_b2")
let denseGemvQ6KBtileB4PSO = pso("dense_gemv_q6_K_btile_b4")
let denseGemvQ6KBtileB8PSO = pso("dense_gemv_q6_K_btile_b8")
let denseGemvQ51BtileB1PSO = pso("dense_gemv_q5_1_btile_b1")
let denseGemvQ51BtileB2PSO = pso("dense_gemv_q5_1_btile_b2")
let denseGemvQ51BtileB4PSO = pso("dense_gemv_q5_1_btile_b4")
let denseGemvQ51BtileB8PSO = pso("dense_gemv_q5_1_btile_b8")

// Q5_K fused-RMSNorm QKV variants — structural mirror of Q8_0's
// dense_gemv_q8_0_btile_qkv_otf_b{1,2,4,8}. Same 4-phase shape (RMS reduction,
// slab routing, OTF-RMSNorm GEMV with the 3 weight projections, per-batch
// reduction); Phase 3's dequant inside the kb loop is Q5_K instead of Q8_0.
let denseGemvQ5KBtileQkvOtfB1PSO = pso("dense_gemv_q5_K_btile_qkv_otf_b1")
let denseGemvQ5KBtileQkvOtfB2PSO = pso("dense_gemv_q5_K_btile_qkv_otf_b2")
let denseGemvQ5KBtileQkvOtfB4PSO = pso("dense_gemv_q5_K_btile_qkv_otf_b4")
let denseGemvQ5KBtileQkvOtfB8PSO = pso("dense_gemv_q5_K_btile_qkv_otf_b8")

// Q5_K fused-RMSNorm gate+up variants — structural mirror of Q8_0's
// dense_gemv_q8_0_btile_gate_up_otf_b{1,2,4,8}. Writes interleaved fused
// [slot, 2*D_out] layout that encMoeGeluMulFused consumes downstream.
let denseGemvQ5KBtileGateUpOtfB1PSO = pso("dense_gemv_q5_K_btile_gate_up_otf_b1")
let denseGemvQ5KBtileGateUpOtfB2PSO = pso("dense_gemv_q5_K_btile_gate_up_otf_b2")
let denseGemvQ5KBtileGateUpOtfB4PSO = pso("dense_gemv_q5_K_btile_gate_up_otf_b4")
let denseGemvQ5KBtileGateUpOtfB8PSO = pso("dense_gemv_q5_K_btile_gate_up_otf_b8")

// MoE V11 templated zoo for Q5_K (slot_token broadcast / gate-up convention),
// Q8_0 (per-slot / down convention), and Q6_K (per-slot / down convention).
// Same MAX_SLOTS register-tile pattern as moe_gemv_q4k_v11 / moe_gemv_q5_1_v11.
let moeGemvQ5KV11B1PSO = pso("moe_gemv_q5_K_v11_b1")
let moeGemvQ5KV11B2PSO = pso("moe_gemv_q5_K_v11_b2")
let moeGemvQ5KV11B4PSO = pso("moe_gemv_q5_K_v11_b4")
let moeGemvQ5KV11B8PSO = pso("moe_gemv_q5_K_v11_b8")
let moeGemvQ80V11B1PSO = pso("moe_gemv_q8_0_v11_b1")
let moeGemvQ80V11B2PSO = pso("moe_gemv_q8_0_v11_b2")
let moeGemvQ80V11B4PSO = pso("moe_gemv_q8_0_v11_b4")
let moeGemvQ80V11B8PSO = pso("moe_gemv_q8_0_v11_b8")
let moeGemvQ6KV11B1PSO = pso("moe_gemv_q6_K_v11_b1")
let moeGemvQ6KV11B2PSO = pso("moe_gemv_q6_K_v11_b2")
let moeGemvQ6KV11B4PSO = pso("moe_gemv_q6_K_v11_b4")
let moeGemvQ6KV11B8PSO = pso("moe_gemv_q6_K_v11_b8")

// Q6_K + Q5_1 fused-RMSNorm QKV/gate_up zoos — structural mirrors of the
// Q8_0 / Q5_K versions for hypothetical configs that put Q6_K or Q5_1 on
// attention/shared-FFN dense weights.
let denseGemvQ6KBtileQkvOtfB1PSO    = pso("dense_gemv_q6_K_btile_qkv_otf_b1")
let denseGemvQ6KBtileQkvOtfB2PSO    = pso("dense_gemv_q6_K_btile_qkv_otf_b2")
let denseGemvQ6KBtileQkvOtfB4PSO    = pso("dense_gemv_q6_K_btile_qkv_otf_b4")
let denseGemvQ6KBtileQkvOtfB8PSO    = pso("dense_gemv_q6_K_btile_qkv_otf_b8")
let denseGemvQ51BtileQkvOtfB1PSO    = pso("dense_gemv_q5_1_btile_qkv_otf_b1")
let denseGemvQ51BtileQkvOtfB2PSO    = pso("dense_gemv_q5_1_btile_qkv_otf_b2")
let denseGemvQ51BtileQkvOtfB4PSO    = pso("dense_gemv_q5_1_btile_qkv_otf_b4")
let denseGemvQ51BtileQkvOtfB8PSO    = pso("dense_gemv_q5_1_btile_qkv_otf_b8")
let denseGemvQ6KBtileGateUpOtfB1PSO = pso("dense_gemv_q6_K_btile_gate_up_otf_b1")
let denseGemvQ6KBtileGateUpOtfB2PSO = pso("dense_gemv_q6_K_btile_gate_up_otf_b2")
let denseGemvQ6KBtileGateUpOtfB4PSO = pso("dense_gemv_q6_K_btile_gate_up_otf_b4")
let denseGemvQ6KBtileGateUpOtfB8PSO = pso("dense_gemv_q6_K_btile_gate_up_otf_b8")
let denseGemvQ51BtileGateUpOtfB1PSO = pso("dense_gemv_q5_1_btile_gate_up_otf_b1")
let denseGemvQ51BtileGateUpOtfB2PSO = pso("dense_gemv_q5_1_btile_gate_up_otf_b2")
let denseGemvQ51BtileGateUpOtfB4PSO = pso("dense_gemv_q5_1_btile_gate_up_otf_b4")
let denseGemvQ51BtileGateUpOtfB8PSO = pso("dense_gemv_q5_1_btile_gate_up_otf_b8")

// Q4_0 zoo — full structural-mirror coverage. Q4_0 is the simplest 4bpw
// format (no min, no per-pair scales — w = d * (nib - 8)) and historically
// the fastest 4bpw on Apple Silicon because the dequant chain is minimal.
let denseGemvQ40BtileB1PSO = pso("dense_gemv_q4_0_btile_b1")
let denseGemvQ40BtileB2PSO = pso("dense_gemv_q4_0_btile_b2")
let denseGemvQ40BtileB4PSO = pso("dense_gemv_q4_0_btile_b4")
let denseGemvQ40BtileB8PSO = pso("dense_gemv_q4_0_btile_b8")
let denseGemvQ40BtileQkvOtfB1PSO    = pso("dense_gemv_q4_0_btile_qkv_otf_b1")
let denseGemvQ40BtileQkvOtfB2PSO    = pso("dense_gemv_q4_0_btile_qkv_otf_b2")
let denseGemvQ40BtileQkvOtfB4PSO    = pso("dense_gemv_q4_0_btile_qkv_otf_b4")
let denseGemvQ40BtileQkvOtfB8PSO    = pso("dense_gemv_q4_0_btile_qkv_otf_b8")
let denseGemvQ40BtileGateUpOtfB1PSO = pso("dense_gemv_q4_0_btile_gate_up_otf_b1")
let denseGemvQ40BtileGateUpOtfB2PSO = pso("dense_gemv_q4_0_btile_gate_up_otf_b2")
let denseGemvQ40BtileGateUpOtfB4PSO = pso("dense_gemv_q4_0_btile_gate_up_otf_b4")
let denseGemvQ40BtileGateUpOtfB8PSO = pso("dense_gemv_q4_0_btile_gate_up_otf_b8")
let moeGemvQ40V11B1PSO = pso("moe_gemv_q4_0_v11_b1")
let moeGemvQ40V11B2PSO = pso("moe_gemv_q4_0_v11_b2")
let moeGemvQ40V11B4PSO = pso("moe_gemv_q4_0_v11_b4")
let moeGemvQ40V11B8PSO = pso("moe_gemv_q4_0_v11_b8")
// Q4_0 V11 down (per-slot) — paired with the up variant above so Q4_0 can
// serve any MoE role (gate_up via slot_token broadcast OR ffn_down per-slot)
// without the silent-misroute hazard of routing one through the other.
let moeGemvQ40V11DownB1PSO = pso("moe_gemv_q4_0_v11_down_b1")
let moeGemvQ40V11DownB2PSO = pso("moe_gemv_q4_0_v11_down_b2")
let moeGemvQ40V11DownB4PSO = pso("moe_gemv_q4_0_v11_down_b4")
let moeGemvQ40V11DownB8PSO = pso("moe_gemv_q4_0_v11_down_b8")

// Q4_1 zoo — same dequant shape as Q4_0 but with additive offset m
// (w = d * nib + m, no -8 bias). Block: 20 B / 32 elts.
let denseGemvQ41BtileB1PSO = pso("dense_gemv_q4_1_btile_b1")
let denseGemvQ41BtileB2PSO = pso("dense_gemv_q4_1_btile_b2")
let denseGemvQ41BtileB4PSO = pso("dense_gemv_q4_1_btile_b4")
let denseGemvQ41BtileB8PSO = pso("dense_gemv_q4_1_btile_b8")
let denseGemvQ41BtileQkvOtfB1PSO    = pso("dense_gemv_q4_1_btile_qkv_otf_b1")
let denseGemvQ41BtileQkvOtfB2PSO    = pso("dense_gemv_q4_1_btile_qkv_otf_b2")
let denseGemvQ41BtileQkvOtfB4PSO    = pso("dense_gemv_q4_1_btile_qkv_otf_b4")
let denseGemvQ41BtileQkvOtfB8PSO    = pso("dense_gemv_q4_1_btile_qkv_otf_b8")
let denseGemvQ41BtileGateUpOtfB1PSO = pso("dense_gemv_q4_1_btile_gate_up_otf_b1")
let denseGemvQ41BtileGateUpOtfB2PSO = pso("dense_gemv_q4_1_btile_gate_up_otf_b2")
let denseGemvQ41BtileGateUpOtfB4PSO = pso("dense_gemv_q4_1_btile_gate_up_otf_b4")
let denseGemvQ41BtileGateUpOtfB8PSO = pso("dense_gemv_q4_1_btile_gate_up_otf_b8")
let moeGemvQ41V11B1PSO = pso("moe_gemv_q4_1_v11_b1")
let moeGemvQ41V11B2PSO = pso("moe_gemv_q4_1_v11_b2")
let moeGemvQ41V11B4PSO = pso("moe_gemv_q4_1_v11_b4")
let moeGemvQ41V11B8PSO = pso("moe_gemv_q4_1_v11_b8")
let moeGemvQ41V11DownB1PSO = pso("moe_gemv_q4_1_v11_down_b1")
let moeGemvQ41V11DownB2PSO = pso("moe_gemv_q4_1_v11_down_b2")
let moeGemvQ41V11DownB4PSO = pso("moe_gemv_q4_1_v11_down_b4")
let moeGemvQ41V11DownB8PSO = pso("moe_gemv_q4_1_v11_down_b8")
let denseQ80V4PSO  = pso("dense_gemv_q8_0_v4")
let moeQ51PSO      = pso("moe_gemv_q5_1_v3")
let moeGeluMulFusedPSO = pso("moe_gelu_mul_fused")
let pagedSplitReducePSO = pso("paged_attn_split_reduce")
let sampleTokenPSO      = pso("sample_token")
// Two specialized PSOs compiled from the one unified `flex_attn_v0` kernel
// source. Function constants set head-dim / page / Q-per-TG / mask style;
// the MSL compiler produces the same asm as the old per-variant kernels.
let flexAttnSlideV0PSO: MTLComputePipelineState = psoFC("flex_attn_v0") { fcv in
    var d: Int32 = 256, page: Int32 = 16, qPerTg: Int32 = 2
    var useSlide: Bool = true
    fcv.setConstantValue(&d,        type: .int,  index: 0)
    fcv.setConstantValue(&page,     type: .int,  index: 1)
    fcv.setConstantValue(&qPerTg,   type: .int,  index: 2)
    fcv.setConstantValue(&useSlide, type: .bool, index: 3)
}
let flexAttnFullV0PSO: MTLComputePipelineState = psoFC("flex_attn_v0") { fcv in
    var d: Int32 = 512, page: Int32 = 8, qPerTg: Int32 = 8
    var useSlide: Bool = false
    fcv.setConstantValue(&d,        type: .int,  index: 0)
    fcv.setConstantValue(&page,     type: .int,  index: 1)
    fcv.setConstantValue(&qPerTg,   type: .int,  index: 2)
    fcv.setConstantValue(&useSlide, type: .bool, index: 3)
}
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

// Engine handle for the dylib's FFI entry points. Defined here (not in
// ffi.swift) so the executable build — which excludes ffi.swift — still
// resolves it. ffi.swift owns gEngine's lifecycle (assigns at gemma_init,
// clears at teardown). In the executable build gEngine stays nil and the
// writeAblation paths in buildStepCB no-op.
//
// There is no engine-state mutex in Swift. The bridge layer is responsible
// for serializing FFI entry — see notes/decisions/2026-04-26-remove-
// session-concurrency-primitives.md (Phase B).
var gEngine: LmEngine?

// --- Batch config ---
let B = 8
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
//
// 2026-05-06: empirical pool-growth cliff above 8192 pages confirmed
// to NOT be about scratch-address arithmetic. Tried decoupling scratch
// to LOW addresses (phys [0, SCRATCH_STRIP)) and growing pool to
// 12288. K/V allocation succeeded, but bridge STILL wedged on first
// GPU dispatch — same symptom as direct pool=12288 / 16384.
//
// So the cliff is the BUFFER SIZE itself, not the addressing pattern
// within it. At pool=8192 each sliding-layer K (or V) buffer is
// 528 MB; at pool=12288 it's 784 MB. Apple Metal apparently has
// some per-resource constraint between these that we haven't yet
// pinpointed (per-buffer binding limit? per-CB resource limit? a
// kernel argument-table thing? we don't know yet).
//
// Real fix: split per-layer K/V cache into multiple smaller device
// buffers, update kernel address arithmetic to a 2-level index
// (which buffer + offset within buffer). Substantial Metal kernel
// refactor; deferred until a debugging session can narrow the limit.
//
// Reverted to 8192 pages. The pre-grow synthetic over-reservation in
// runAdmissionPass is removed (lm_engine.swift:1436), so 8192 is now
// sized for actual concurrent demand: 8 streams × ~140 pages typical
// = ~1100 pages active, with ~7× headroom.
let MAX_RESIDENT_SESSIONS = 16                    // logical users held in KV
let SCRATCH_STRIP = 256                           // silenced-slot scratch (shared)
let SCRATCH_PAGE_BASE = 8192                     // DEBUG: pool=12288 to capture all Metal validation messages
let REAL_PAGE_BASE = 0
let PHYS_POOL_PAGES = SCRATCH_PAGE_BASE
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
// Settled at 256 (2026-05-01) after empirical bridge measurement showed
// 256 is at-or-near the end-to-end pareto optimum for single-stream
// prefill. Earlier history: 32 → 256 alongside the simdgroup-matmul
// rewire (commit bcfa0fd) when the matmul kernel was Q-batch-starved
// at MAX_Q_LEN=32.
//
// Tried 256 → 1024 to feed the matmul kernel at its saturated regime
// (bench: 8.0 TFLOPS at numVecs=256 vs 13.7 TFLOPS at numVecs=1024).
// Net: SLOWER end-to-end. The per-token attention cost grows with
// chunk_qLen (kv_attn went 5.5 → 10.8 µs/tok between MAX_Q_LEN=256
// and 1024 in profile_prefill), and that loss eats the matmul gain.
// Cold bridge measurement at ~3000-token prompt: 141 tok/s @ MQL=256
// vs 85 tok/s @ MQL=1024.
//
// MAX_Q_LEN is the prefill CHUNK size / per-kernel-dispatch work size.
// It is NOT the KV cache size (that is MAX_PAGES_PER_SLOT × PAGE_FULL
// ≈ 64k tokens). Long prompts are chunked through the engine's
// multi-tile prefill path (lm_engine.swift:2099 `thisTile = min(qLen,
// MAX_Q_LEN)`); the chunk size is the GPU work granularity, not a
// per-prompt limit. Callers that hit MAX_Q_LEN as a per-call cap
// (e.g. the single-shot teacher-forced FFI) are FFI-implementation
// limited — the right fix is to chunk in the FFI, not to inflate the
// chunk size.
//
// Scratch budget: pre_logits (B × MAX_Q_LEN × VOCAB fp16) is the
// binding constraint; at B=8 / VOCAB=262144 / MAX_Q_LEN=256 → 1 GB.
let MAX_Q_LEN = 256
// Max number of 8-row Q-blocks that can fit in one prefill step. The
// kernels (`flex_attn_full_prefill`, `flex_attn_slide_v1_q8`) tile along
// the q_block axis at runtime; the v2 mask precompute below emits one
// CSR entry per (slot, q_block) tuple, making MAX_Q_LEN ≥ 8 actually
// usable. Buffer sizing below scales with this.
let MAX_Q_BLOCKS = (MAX_Q_LEN + 7) / 8

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
// MAX_PREFILL_TILES sets the cap on tiles-per-single-CB. With MAX_Q_LEN
// bumped from 8 → 256, a 280-soft-token image now fits in 2 tiles, so
// the prior 64-tile headroom is way overprovisioned. Drop to 16 — that
// caps prefill at 16 × 256 = 4096 tokens per single CB. Larger prompts
// chunk into multiple CBs (per-CB submission overhead is ~100µs vs
// per-CB compute of tens of ms, so the chunking cost is negligible).
// Wide-staging buffer pre_hidden_wide is B × MAX_PREFILL_TOKENS ×
// HIDDEN fp16 = 8 × 4096 × 2304 × 2 = 150 MB at this setting.
let MAX_PREFILL_TILES = 16
let MAX_PREFILL_TOKENS = MAX_PREFILL_TILES * MAX_Q_LEN   // = 4096
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

// Prefill block-mask buffers (slide-attention layers).
// CSR layout: one entry per (slot, q_block) → B * MAX_Q_BLOCKS + 1 offsets.
// Indices/masks: worst-case every k_block × every q_block is partial.
let pre_slide_full_offsets = device.makeBuffer(length: (B * MAX_Q_BLOCKS + 1) * 4, options: .storageModeShared)!
let pre_slide_full_indices = device.makeBuffer(length: B * MAX_Q_BLOCKS * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!
let pre_slide_part_offsets = device.makeBuffer(length: (B * MAX_Q_BLOCKS + 1) * 4, options: .storageModeShared)!
let pre_slide_part_indices = device.makeBuffer(length: B * MAX_Q_BLOCKS * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!

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
// One uint32 mask per Q-row × per partial-block. Worst case: every k_block
// is partial in every q_block of every slot.
let pre_slide_part_masks = device.makeBuffer(length: B * MAX_Q_BLOCKS * MAX_PAGES_PER_SLOT * FLEX_Q_BLOCK * 4,
                                               options: .storageModeShared)!
let flex_full_part_masks = device.makeBuffer(length: B * MAX_Q_BLOCKS * MAX_PAGES_PER_SLOT * FLEX_Q_BLOCK * 4,
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
// Same per-(slot, q_block) CSR layout as the slide buffers above.
let flex_full_full_offsets = device.makeBuffer(length: (B * MAX_Q_BLOCKS + 1) * 4, options: .storageModeShared)!
let flex_full_full_indices = device.makeBuffer(length: B * MAX_Q_BLOCKS * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!
let flex_full_partial_offsets = device.makeBuffer(length: (B * MAX_Q_BLOCKS + 1) * 4, options: .storageModeShared)!
let flex_full_partial_indices = device.makeBuffer(length: B * MAX_Q_BLOCKS * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!

// Dynamic routing/control buffers (populated by the forward pass every step).
let positions    = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let block_table  = device.makeBuffer(length: B * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!

let active_exp   = device.makeBuffer(length: E_EXP * 4, options: .storageModeShared)!
let group_start  = device.makeBuffer(length: (E_EXP + 1) * 4, options: .storageModeShared)!
let slot_token   = device.makeBuffer(length: TOTAL_SLOTS * 4, options: .storageModeShared)!
let batch_slots  = device.makeBuffer(length: B * TOPK * 4, options: .storageModeShared)!
let input_tokens = device.makeBuffer(length: B * 4, options: .storageModeShared)!

// Per-slot sampling-param buffers used by the GPU-side `sample_token`
// kernel (see docs/dataflow_pipeline_spec.md §2). CPU writes per-step
// from session state before committing the step CB; GPU reads during
// dispatch; GPU writes the sampled token into `gpu_sampled_tokens`
// which the post-wait CPU compares against the CPU sampler output
// (Phase 2 validation) or uses directly (post-validation).
let sampling_temperature = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let sampling_min_p       = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let sampling_seed        = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let sampling_step        = device.makeBuffer(length: B * 4, options: .storageModeShared)!
let sampling_active      = device.makeBuffer(length: B * 4, options: .storageModeShared)!
// Alias gpu_sampled_tokens → input_tokens. Sampler writes the next-step
// input directly. This lets CB N+1 read input_tokens written by CB N's
// sampler with no CPU bridge — Metal queue serialization handles the
// dependency. Foundation for two-CB-in-flight pipelining: the CPU no
// longer needs to wait for CB N's completion to populate input_tokens
// for CB N+1's AR slots.
let gpu_sampled_tokens = input_tokens

// Dense per-slot logit-bias buffer. Sampler always reads it — slots
// without a client-set bias hold all-zeros, so the add is a no-op.
// 4 × 262144 × 4B = 4 MB total; per-step BW cost ≈ 80 μs at M5's
// unified-memory bandwidth, negligible vs a ~100 ms step. Written
// each step from session state by step(): memcpy when a session has
// `logitBiasDense`, memset(0) when it doesn't.
let sampling_logit_bias  = device.makeBuffer(length: B * VOCAB * 4, options: .storageModeShared)!

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

// Scaled cvector injection into residual: dst[b, :] += mag * cvec[:] for
// every b in [0, numVecs). Cheap — one 32-thread TG per row, weight-bound
// on the HIDDEN-length cvec read. Intended to be called at one or more
// designated post-residual sites inside the LM layer loop.
func encAddScaledCvec(_ cb: MTLCommandBuffer, dst: MTLBuffer, cvec: MTLBuffer,
                       mag: Float, N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addScaledCvecPSO)
    enc.setBuffer(dst, offset: 0, index: 0); enc.setBuffer(cvec, offset: 0, index: 1)
    var Nv = UInt32(N); var magv = mag
    enc.setBytes(&Nv, length: 4, index: 2)
    enc.setBytes(&magv, length: 4, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Per-slot dot-product measurement. Writes a single float into
// `intensities[slot]` representing the raw projection of the residual
// row onto the measurement direction. Pair with the Phase C-Read pump
// logic that reads this buffer back between ticks and gates effector
// envelopes on threshold crossings.
func encMeasureDotSlot(_ cb: MTLCommandBuffer, src: MTLBuffer, slot: Int,
                        meas: MTLBuffer, intensities: MTLBuffer,
                        intensitySlotOffsetBytes: Int, N: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(measureDotPSO)
    enc.setBuffer(src, offset: slot * N * 2, index: 0)
    enc.setBuffer(meas, offset: 0, index: 1)
    enc.setBuffer(intensities, offset: intensitySlotOffsetBytes, index: 2)
    var Nv = UInt32(N)
    enc.setBytes(&Nv, length: 4, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Per-slot variant: only the row at dst[slot, :] gets the injection.
// buildStepCB calls this once per (slot, active-control) pair per layer
// so concurrent sessions can carry independent steering state. dst is
// offset by slot * N * sizeof(half) and the kernel sees numVecs=1.
func encAddScaledCvecSlot(_ cb: MTLCommandBuffer, dst: MTLBuffer, slot: Int,
                           cvec: MTLBuffer, mag: Float, N: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addScaledCvecPSO)
    enc.setBuffer(dst, offset: slot * N * 2, index: 0)
    enc.setBuffer(cvec, offset: 0, index: 1)
    var Nv = UInt32(N); var magv = mag
    enc.setBytes(&Nv, length: 4, index: 2)
    enc.setBytes(&magv, length: 4, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Prefill-variant cvec injection: dst[r, :] += mags[r] * cvec[:] for every
// row r in [0, numVecs). mags[r]==0 short-circuits in the kernel so slots
// without this control active at a given position pay near-zero cost. One
// dispatch per (layer, active-control) pair during encodePrefillTileInto.
// `magsBuf` is a [numVecs] float buffer the engine fills CPU-side by
// evaluating each control's envelope at pre_q_positions[b*qLen+i].
func encAddScaledCvecPrefill(_ cb: MTLCommandBuffer, dst: MTLBuffer,
                              cvec: MTLBuffer, magsBuf: MTLBuffer,
                              N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addScaledCvecPrefillPSO)
    enc.setBuffer(dst, offset: 0, index: 0)
    enc.setBuffer(cvec, offset: 0, index: 1)
    enc.setBuffer(magsBuf, offset: 0, index: 2)
    var Nv = UInt32(N)
    enc.setBytes(&Nv, length: 4, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// ------------------------------------------------------------------
// Projection-steer dispatchers. Each writes the residual's projection
// onto cvec to a caller-specified target value AND writes the pre-
// write projection to currentProjBuf — one dispatch that both
// coerces AND measures, enabling representation-engineering flows
// where the user wants to read a feature's natural level at each
// position alongside any coercion they apply.
// ------------------------------------------------------------------
func encProjectCvec(_ cb: MTLCommandBuffer, dst: MTLBuffer, cvec: MTLBuffer,
                     currentProjBuf: MTLBuffer, target: Float,
                     N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(projectCvecPSO)
    enc.setBuffer(dst, offset: 0, index: 0)
    enc.setBuffer(cvec, offset: 0, index: 1)
    enc.setBuffer(currentProjBuf, offset: 0, index: 2)
    var Nv = UInt32(N); var t = target
    enc.setBytes(&Nv, length: 4, index: 3)
    enc.setBytes(&t, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encProjectCvecSlot(_ cb: MTLCommandBuffer, dst: MTLBuffer, slot: Int,
                         cvec: MTLBuffer, currentProjBuf: MTLBuffer,
                         currentProjSlotOffsetBytes: Int, target: Float, N: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(projectCvecSlotPSO)
    enc.setBuffer(dst, offset: slot * N * 2, index: 0)
    enc.setBuffer(cvec, offset: 0, index: 1)
    enc.setBuffer(currentProjBuf, offset: currentProjSlotOffsetBytes, index: 2)
    var Nv = UInt32(N); var t = target
    enc.setBytes(&Nv, length: 4, index: 3)
    enc.setBytes(&t, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Heretic per-write ablation: y -= alpha * r_hat * dot(r_hat, y) over
// every row of a [numVecs, N] fp16 buffer. Same (r_hat, alpha) for all
// rows — model-level intervention, one dispatch per (layer, component)
// covering all B batch slots.
func encOrthogonalizeWrite(_ cb: MTLCommandBuffer, y: MTLBuffer,
                            rHat: MTLBuffer, alpha: Float,
                            N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(orthogWritePSO)
    enc.setBuffer(y,    offset: 0, index: 0)
    enc.setBuffer(rHat, offset: 0, index: 1)
    var Nv = UInt32(N); var a = alpha
    enc.setBytes(&Nv, length: 4, index: 2)
    enc.setBytes(&a,  length: 4, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Prefill variant. targets buffer holds [numVecs] fp32; a row with
// target == Float.nan is skipped (kernel checks isnan). Matches the
// add_scaled_cvector_prefill_fp16 idiom of "zero magnitude = no-op row"
// but with a non-zero sentinel so target=0 remains meaningful.
func encProjectCvecPrefill(_ cb: MTLCommandBuffer, dst: MTLBuffer,
                            cvec: MTLBuffer, targetsBuf: MTLBuffer,
                            currentProjBuf: MTLBuffer, N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(projectCvecPrefillPSO)
    enc.setBuffer(dst, offset: 0, index: 0)
    enc.setBuffer(cvec, offset: 0, index: 1)
    enc.setBuffer(targetsBuf, offset: 0, index: 2)
    enc.setBuffer(currentProjBuf, offset: 0, index: 3)
    var Nv = UInt32(N)
    enc.setBytes(&Nv, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Transport slot variant: takes (scale, offset) scalars; projection is
// coerced via the Brenier map a → scale*a + offset, which is per-PC
// Gaussian OT when scale=σ_tgt/σ_src, offset=μ_tgt-scale*μ_src. Same
// dispatch/threadgroup shape as encProjectCvecSlot; the only kernel
// difference is the target formula (constant vs linear-in-projection).
func encTransportCvecSlot(_ cb: MTLCommandBuffer, dst: MTLBuffer, slot: Int,
                           cvec: MTLBuffer, currentProjBuf: MTLBuffer,
                           currentProjSlotOffsetBytes: Int,
                           scale: Float, offset: Float, N: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(transportCvecSlotPSO)
    enc.setBuffer(dst, offset: slot * N * 2, index: 0)
    enc.setBuffer(cvec, offset: 0, index: 1)
    enc.setBuffer(currentProjBuf, offset: currentProjSlotOffsetBytes, index: 2)
    var Nv = UInt32(N); var sc = scale; var off = offset
    enc.setBytes(&Nv, length: 4, index: 3)
    enc.setBytes(&sc, length: 4, index: 4)
    enc.setBytes(&off, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Transport prefill variant. scalesBuf[r] = NaN → row skipped. offsets
// read per-row from offsetsBuf. Mirrors encProjectCvecPrefill otherwise.
func encTransportCvecPrefill(_ cb: MTLCommandBuffer, dst: MTLBuffer,
                              cvec: MTLBuffer,
                              scalesBuf: MTLBuffer, offsetsBuf: MTLBuffer,
                              currentProjBuf: MTLBuffer,
                              N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(transportCvecPrefillPSO)
    enc.setBuffer(dst, offset: 0, index: 0)
    enc.setBuffer(cvec, offset: 0, index: 1)
    enc.setBuffer(scalesBuf, offset: 0, index: 2)
    enc.setBuffer(offsetsBuf, offset: 0, index: 3)
    enc.setBuffer(currentProjBuf, offset: 0, index: 4)
    var Nv = UInt32(N)
    enc.setBytes(&Nv, length: 4, index: 5)
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

// V4 with grid extended over vec-blocks. Same kernel as V4 but each TG
// targets a 32-output × MAX_B=8-vec tile. For huge D_out (unembed) where
// V5's per-vec L2 amortization breaks down, this pattern reads weight
// once per slab and FMAs into MAX_B accumulators in registers.
func encGemvV4P(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer,
                 Din: Int, Dout: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(gemvV4PPSO)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var du = UInt32(Din), dou = UInt32(Dout), nv = UInt32(numVecs)
    enc.setBytes(&du, length: 4, index: 3)
    enc.setBytes(&dou, length: 4, index: 4)
    enc.setBytes(&nv, length: 4, index: 5)
    let MAX_B = 8           // must match dense_gemv_v4_p's MAX_B
    let yBlocks = (numVecs + MAX_B - 1) / MAX_B
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: yBlocks, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encGemvV4Softcap(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer,
                       Din: Int, Dout: Int, cap: Float, activeB: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(gemvV4SoftcapPSO)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    // bv = activeB so the kernel's `for b in 0..<B` inner loop runs only
    // over real slots. MAX_B=8 hardcoded in the kernel sizes the per-thread
    // accs[] array; passing activeB just early-exits the inner loop.
    var bv = UInt32(activeB), du = UInt32(Din), dou = UInt32(Dout), cv = cap
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

// Q8_0 GEMV kernel-zoo dispatcher: picks the B_TILE-specialized variant
// for the given activeB. Each variant holds B_TILE accumulators in
// registers (compile-time-fixed, fully unrolled inner FMA loop) and reads
// the weight slab once per K-tile per TG. Grid is 1D (D_out/32 TGs total)
// regardless of activeB — all the per-batch work is in registers.
//
// Scheduler rounds activeB UP to the nearest power of 2: 1→b1, 2→b2,
// {3,4}→b4, {5..8}→b8. For non-power-of-2 actual counts, the unused
// register slots accumulate against silenced-slot inputs; the caller is
// expected to populate hidden[0..activeB) with real data and may leave
// the tail rows uninitialized (output[activeB..B_TILE) is overwritten
// with garbage and discarded by downstream consumers that index by
// active slot only).
func encGemvQ80Btile(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ WswBuf: MTLBuffer,
                      _ out: MTLBuffer, Din: Int, Dout: Int, activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let pso: MTLComputePipelineState
    switch activeB {
    case 1: pso = denseQ80BtileB1PSO
    case 2: pso = denseQ80BtileB2PSO
    case 3, 4: pso = denseQ80BtileB4PSO
    default: pso = denseQ80BtileB8PSO   // 5..8
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(WswBuf, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 3)
    enc.setBytes(&dou, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Q8_0 RMSNorm + QKV kernel-zoo dispatcher. activeB ≤ 4 picks the
// templated b1/b2/b4 variant (one TG per slab, all batches in registers,
// per-batch tg-mem h_norms). activeB > 4 falls back to V6 grid-shrink
// (numVecs=activeB) — no btile_qkv_b8 yet because full-D h_norm staging
// for 8 batches exceeds the 32 KB tg-mem budget.
func encGemvQ80BtileQKV(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                         Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                         outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                         Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                         activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    if activeB > 4 {
        // No btile_qkv_b8 yet — fall back to V6 with numVecs=activeB.
        encGemvQ80V6RmsnormQKV(cb, x: x, gammaBuf: gammaBuf,
                                Wq: Wq, Wk: Wk, Wv: Wv,
                                outQ: outQ, outK: outK, outV: outV,
                                Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                                numVecs: activeB)
        return
    }
    let pso: MTLComputePipelineState
    let bTile: Int
    switch activeB {
    case 1: pso = denseQ80BtileQkvB1PSO; bTile = 1
    case 2: pso = denseQ80BtileQkvB2PSO; bTile = 2
    default: pso = denseQ80BtileQkvB4PSO; bTile = 4    // activeB ∈ {3,4}
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
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
    var nv = UInt32(activeB)
    enc.setBytes(&du, length: 4, index: 8)
    enc.setBytes(&qnb, length: 4, index: 9)
    enc.setBytes(&knb, length: 4, index: 10)
    enc.setBytes(&vnb, length: 4, index: 11)
    enc.setBytes(&douq, length: 4, index: 12)
    enc.setBytes(&douk, length: 4, index: 13)
    enc.setBytes(&douv, length: 4, index: 14)
    enc.setBytes(&eps, length: 4, index: 15)
    enc.setBytes(&nv, length: 4, index: 16)
    let totalSlabs = (DoutQ + DoutK + DoutV) / 32
    let yBlocks = (activeB + bTile - 1) / bTile
    enc.dispatchThreadgroups(MTLSize(width: totalSlabs, height: yBlocks, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Q8_0 RMSNorm + QKV kernel-zoo dispatcher (OTF — on-the-fly normalize).
// All four otf widths {1, 2, 4, 8} compile and run, but only b1 actually
// beats V6 in measurement. V6's grid (totalSlabs × B) parallelism wins
// over the otf "all-batches-in-one-TG" design at higher widths because
// per-TG work scales with B_TILE while V6 spreads across 8 SMs cleanly.
// Policy:
//   activeB == 1 → otf_b1 (best single-stream)
//   activeB >= 2 → V6 grid-shrink (won the Item-C falsification A/B)
// Item C (K-tile staging at B_TILE=2/4) was falsified empirically:
// tiled_b2 lost N=2 by -18%, tiled_b4 lost N=4 by -5%. The tg-mem
// barriers + cooperative load overhead outweigh the gamma/inv_rms hoist
// benefit at these batch sizes; V6's per-batch-TG design with cheap
// launch overhead and L1/L2-amortized weight reads wins at {2,3,4}.
// The OTF b2/b4/b8 and tiled b1/2/4/8 PSOs stay registered as diagnostic
// kernels for future revisits if the substrate changes.
func encGemvQ80BtileQKVOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                            Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                            outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                            Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                            activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    // Three-tier dispatch:
    //   activeB == 1            → OTF B_TILE=1 (best at single-stream)
    //   activeB ∈ {2, 3, 4}     → tiled B_TILE=2/4 (Item C — K-tile staging
    //                             closes the gap where OTF's per-FMA gamma
    //                             reads accumulate but V6 grid-shrink wastes
    //                             B-parallelism on small batches)
    //   activeB ∈ {5, 6, 7, 8}  → V6 grid-shrink (large enough B that the
    //                             B-grid amortizes scheduling overhead)
    let pso: MTLComputePipelineState
    let bTile: Int
    if activeB == 1 {
        pso = denseQ80BtileQkvOtfB1PSO
        bTile = 1
    } else {
        encGemvQ80V6RmsnormQKV(cb, x: x, gammaBuf: gammaBuf,
                                Wq: Wq, Wk: Wk, Wv: Wv,
                                outQ: outQ, outK: outK, outV: outV,
                                Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                                numVecs: activeB)
        return
    }
    let _ = (pso, bTile)
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
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
    var nv = UInt32(activeB)
    enc.setBytes(&du, length: 4, index: 8)
    enc.setBytes(&qnb, length: 4, index: 9)
    enc.setBytes(&knb, length: 4, index: 10)
    enc.setBytes(&vnb, length: 4, index: 11)
    enc.setBytes(&douq, length: 4, index: 12)
    enc.setBytes(&douk, length: 4, index: 13)
    enc.setBytes(&douv, length: 4, index: 14)
    enc.setBytes(&eps, length: 4, index: 15)
    enc.setBytes(&nv, length: 4, index: 16)
    let totalSlabs = (DoutQ + DoutK + DoutV) / 32
    let yBlocks = (activeB + bTile - 1) / bTile
    enc.dispatchThreadgroups(MTLSize(width: totalSlabs, height: yBlocks, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Q8_0 RMSNorm + gate_up kernel-zoo dispatcher (OTF). Same policy as
// QKV otf: otf_b1 wins at activeB=1, V6 grid-shrink wins at higher
// widths because V6's TG-grid parallelism (2*D_out/32 × B TGs) beats
// otf's all-batches-in-one-TG layout when there are enough batches to
// amortize SM scheduling.
func encGemvQ80BtileGateUpOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                                Wg: MTLBuffer, Wu: MTLBuffer, fusedOut: MTLBuffer,
                                Din: Int, Dout: Int, activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    if activeB > 1 {
        encGemvQ80V6RmsnormGateUp(cb, x: x, gammaBuf: gammaBuf,
                                    Wg: Wg, Wu: Wu, fusedOut: fusedOut,
                                    Din: Din, Dout: Dout, numVecs: activeB)
        return
    }
    let pso = denseQ80BtileGateUpOtfB1PSO
    let bTile = 1
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(gammaBuf, offset: 0, index: 1)
    enc.setBuffer(Wg, offset: 0, index: 2)
    enc.setBuffer(Wu, offset: 0, index: 3)
    enc.setBuffer(fusedOut, offset: 0, index: 4)
    var du = UInt32(Din), dou = UInt32(Dout); var eps: Float = 1e-6
    var nv = UInt32(activeB)
    enc.setBytes(&du, length: 4, index: 5)
    enc.setBytes(&dou, length: 4, index: 6)
    enc.setBytes(&eps, length: 4, index: 7)
    enc.setBytes(&nv, length: 4, index: 8)
    let yBlocks = (activeB + bTile - 1) / bTile
    enc.dispatchThreadgroups(MTLSize(width: 2 * (Dout / 32), height: yBlocks, depth: 1),
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

// Prefill simdgroup-matmul dispatcher for Q8_0 v6-swizzled weights.
// Tile: NR0=64 output cols, NR1=32 batch rows, 128 threads, 4 simdgroups,
// 4096 + 2048 + 4096 = 10240 B threadgroup memory (sa + sb + reduction).
//
// Y is written as a contiguous [numVecs, Dout] tensor — caller is responsible
// for picking the right output buffer (and offset, via MTLBuffer.makeBuffer
// or setBuffer with explicit offset) when fanning gate/up into a shared
// scratch region. Both Dout and numVecs must be multiples of 32 (the kernel
// clamps inside-tile but expects grid-aligned outer dims for production
// shapes). For Gemma-4 these are always satisfied.
func encMatMulQ80SwizPrefill(_ cb: MTLCommandBuffer,
                              x: MTLBuffer, W: MTLBuffer, Y: MTLBuffer, yOffset: Int = 0,
                              Din: Int, Dout: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(prefillMmQ80SwizPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(Y, offset: yOffset, index: 2)
    var bC = UInt32(numVecs), kC = UInt32(Din), nC = UInt32(Dout)
    enc.setBytes(&bC, length: 4, index: 3)
    enc.setBytes(&kC, length: 4, index: 4)
    enc.setBytes(&nC, length: 4, index: 5)
    enc.setThreadgroupMemoryLength(4096 + 2048 + 4096, index: 0)
    let gx = (numVecs + 31) / 32
    let gy = (Dout + 63) / 64
    enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// MoE Q4_K gate+up prefill dispatcher (simdgroup matmul, v6 swizzled
// per-expert, slot-flat output). Broadcasts X via slot_token: each slot
// reads from X[slot_token[s] * D_in].
//
// Grid: gridX upper-bounded by ceil(N_slots / NR1=32) — the kernel
// early-exits cheaply when r1 >= neh1 for this expert. gridZ = E (all
// experts; sentinel ID=128 from route_compact tail also early-exits).
func encMmIdQ4KSwizPrefill(_ cb: MTLCommandBuffer,
                            x: MTLBuffer, W: MTLBuffer, Y: MTLBuffer,
                            slotTokenBuf: MTLBuffer, activeExpBuf: MTLBuffer,
                            groupStartBuf: MTLBuffer,
                            Din: Int, Dout: Int, numSlots: Int, E: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(prefillMmIdQ4KSwizPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(slotTokenBuf, offset: 0, index: 1)
    enc.setBuffer(W, offset: 0, index: 2)
    enc.setBuffer(activeExpBuf, offset: 0, index: 3)
    enc.setBuffer(groupStartBuf, offset: 0, index: 4)
    enc.setBuffer(Y, offset: 0, index: 5)
    var kC = UInt32(Din), nC = UInt32(Dout)
    enc.setBytes(&kC, length: 4, index: 6)
    enc.setBytes(&nC, length: 4, index: 7)
    enc.setThreadgroupMemoryLength(4096 + 2048 + 4096, index: 0)
    let gx = (numSlots + 31) / 32                  // worst-case slot block count
    let gy = (Dout + 63) / 64
    enc.dispatchThreadgroups(MTLSize(width: max(gx, 1), height: gy, depth: E),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// MoE Q5_1 down prefill dispatcher (simdgroup matmul, v6 swizzled
// per-expert, slot-flat output). Per-slot X read (no slot_token lookup,
// since down's input is already slot-flat after gelu_mul).
func encMmIdQ51SwizPrefill(_ cb: MTLCommandBuffer,
                            x: MTLBuffer, W: MTLBuffer, Y: MTLBuffer,
                            activeExpBuf: MTLBuffer, groupStartBuf: MTLBuffer,
                            Din: Int, Dout: Int, numSlots: Int, E: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(prefillMmIdQ51SwizPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(activeExpBuf, offset: 0, index: 2)
    enc.setBuffer(groupStartBuf, offset: 0, index: 3)
    enc.setBuffer(Y, offset: 0, index: 4)
    var kC = UInt32(Din), nC = UInt32(Dout)
    enc.setBytes(&kC, length: 4, index: 5)
    enc.setBytes(&nC, length: 4, index: 6)
    enc.setThreadgroupMemoryLength(4096 + 2048 + 4096, index: 0)
    let gx = (numSlots + 31) / 32
    let gy = (Dout + 63) / 64
    enc.dispatchThreadgroups(MTLSize(width: max(gx, 1), height: gy, depth: E),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// ─────────────────────────────────────────────────────────────────────
// Format-aware prefill dispatchers. Same buffer layout across formats —
// only the PSO differs. Used wherever the loaded GGUF tensor's quant may
// vary (Q5_K_M loads weights as Q5_K, Q6_K, or Q8_0 depending on tensor).
//
// `encDenseMmPrefill`: dense matmul (no expert dim, no slot routing)
// `encMoeUpMmPrefill`: MoE gate+up (slot_token broadcast: X[slot_token[s]])
// `encMoeDownMmPrefill`: MoE down (per-slot X: X[slot])
// ─────────────────────────────────────────────────────────────────────

func encDenseMmPrefill(_ cb: MTLCommandBuffer,
                        x: MTLBuffer, W: MTLBuffer, format: GGMLType,
                        Y: MTLBuffer, yOffset: Int = 0,
                        Din: Int, Dout: Int, numVecs: Int) {
    let psoSel: MTLComputePipelineState
    switch format {
    case .q8_0: psoSel = prefillMmQ80SwizPSO
    case .q5_K: psoSel = prefillMmQ5KSwizPSO
    case .q6_K: psoSel = prefillMmQ6KSwizPSO
    case .q5_1: psoSel = prefillMmQ51SwizPSO
    case .f16:  psoSel = prefillMmF16SwizPSO
    default: fail("encDenseMmPrefill: unsupported format \(format)")
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(Y, offset: yOffset, index: 2)
    var bC = UInt32(numVecs), kC = UInt32(Din), nC = UInt32(Dout)
    enc.setBytes(&bC, length: 4, index: 3)
    enc.setBytes(&kC, length: 4, index: 4)
    enc.setBytes(&nC, length: 4, index: 5)
    enc.setThreadgroupMemoryLength(4096 + 2048 + 4096, index: 0)
    let gx = (numVecs + 31) / 32
    let gy = (Dout + 63) / 64
    enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

func encMoeUpMmPrefill(_ cb: MTLCommandBuffer,
                        x: MTLBuffer, W: MTLBuffer, format: GGMLType, Y: MTLBuffer,
                        slotTokenBuf: MTLBuffer, activeExpBuf: MTLBuffer,
                        groupStartBuf: MTLBuffer,
                        Din: Int, Dout: Int, numSlots: Int, E: Int) {
    let psoSel: MTLComputePipelineState
    switch format {
    case .q4_K: psoSel = prefillMmIdQ4KSwizPSO
    case .q5_K: psoSel = prefillMmIdQ5KSwizPSO
    case .f16:  psoSel = prefillMmIdF16UpSwizPSO   // slot_token-broadcast variant
    default: fail("encMoeUpMmPrefill: unsupported format \(format)")
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(slotTokenBuf, offset: 0, index: 1)
    enc.setBuffer(W, offset: 0, index: 2)
    enc.setBuffer(activeExpBuf, offset: 0, index: 3)
    enc.setBuffer(groupStartBuf, offset: 0, index: 4)
    enc.setBuffer(Y, offset: 0, index: 5)
    var kC = UInt32(Din), nC = UInt32(Dout)
    enc.setBytes(&kC, length: 4, index: 6)
    enc.setBytes(&nC, length: 4, index: 7)
    enc.setThreadgroupMemoryLength(4096 + 2048 + 4096, index: 0)
    let gx = (numSlots + 31) / 32
    let gy = (Dout + 63) / 64
    enc.dispatchThreadgroups(MTLSize(width: max(gx, 1), height: gy, depth: E),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

func encMoeDownMmPrefill(_ cb: MTLCommandBuffer,
                          x: MTLBuffer, W: MTLBuffer, format: GGMLType, Y: MTLBuffer,
                          activeExpBuf: MTLBuffer, groupStartBuf: MTLBuffer,
                          Din: Int, Dout: Int, numSlots: Int, E: Int) {
    let psoSel: MTLComputePipelineState
    switch format {
    case .q5_1: psoSel = prefillMmIdQ51SwizPSO
    case .q6_K: psoSel = prefillMmIdQ6KSwizPSO
    case .q8_0: psoSel = prefillMmIdQ80SwizPSO
    case .f16:  psoSel = prefillMmIdF16SwizPSO
    default: fail("encMoeDownMmPrefill: unsupported format \(format)")
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(activeExpBuf, offset: 0, index: 2)
    enc.setBuffer(groupStartBuf, offset: 0, index: 3)
    enc.setBuffer(Y, offset: 0, index: 4)
    var kC = UInt32(Din), nC = UInt32(Dout)
    enc.setBytes(&kC, length: 4, index: 5)
    enc.setBytes(&nC, length: 4, index: 6)
    enc.setThreadgroupMemoryLength(4096 + 2048 + 4096, index: 0)
    let gx = (numSlots + 31) / 32
    let gy = (Dout + 63) / 64
    enc.dispatchThreadgroups(MTLSize(width: max(gx, 1), height: gy, depth: E),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// ─────────────────────────────────────────────────────────────────────
// Format-aware AR-decode dispatchers. Mirror of the prefill family but
// for the AR path. Each tensor class (dense, MoE-up, MoE-down) has a
// single-buffer wrapper that takes a `format: GGMLType` and routes to
// the right kernel via a per-format LUT (expressed as a switch).
//
// The LUT entries are:
//   dense:    Q8_0 → btile zoo (kept as fast path),   Q5_K/Q6_K/Q5_1 → v4 single-template
//   MoE-up:   Q4_K → V11 templated zoo,                Q5_K          → v6 single-template
//   MoE-down: Q5_1 → V11 templated zoo,                Q6_K/Q8_0     → v6 single-template
//
// When the engine is fed a Q4_K_M GGUF, all routes resolve to the existing
// optimized fast paths (zero perf regression). When fed Q5_K_M or any other
// mix, routes fall back to the simple-template kernels that still produce
// correct outputs at fp16 floor.
// ─────────────────────────────────────────────────────────────────────

// Single-buffer dense GEMV. Used for attn-out and ffn-down (no preceding
// fused-RMSNorm). For QKV and gate+up which DO share a preceding RMSNorm,
// see encQKVDenseAR / encGateUpDenseAR which handle the fast-path branching.
//
// Every format dispatches through the SAME structural path: 128-thread TG,
// 4-SG split-K, B_TILE templated at b1/b2/b4/b8 (rounded up by power-of-2).
// Per-element dequant is the only thing that differs across formats.
func encDenseGemvAR(_ cb: MTLCommandBuffer,
                     _ xbuf: MTLBuffer, _ W: MTLBuffer, format: GGMLType, _ out: MTLBuffer,
                     Din: Int, Dout: Int, activeB: Int) {
    switch format {
    case .q8_0:
        encGemvQ80Btile(cb, xbuf, W, out, Din: Din, Dout: Dout, activeB: activeB)
    case .q5_K:
        encDenseGemvBtile(cb, xbuf, W, out, psoZoo: q5kBtileZoo,
                           Din: Din, Dout: Dout, activeB: activeB)
    case .q6_K:
        encDenseGemvBtile(cb, xbuf, W, out, psoZoo: q6kBtileZoo,
                           Din: Din, Dout: Dout, activeB: activeB)
    case .q5_1:
        encDenseGemvBtile(cb, xbuf, W, out, psoZoo: q51BtileZoo,
                           Din: Din, Dout: Dout, activeB: activeB)
    case .q4_0:
        encDenseGemvBtile(cb, xbuf, W, out, psoZoo: q40BtileZoo,
                           Din: Din, Dout: Dout, activeB: activeB)
    case .q4_1:
        encDenseGemvBtile(cb, xbuf, W, out, psoZoo: q41BtileZoo,
                           Din: Din, Dout: Dout, activeB: activeB)
    case .f16:
        encDenseGemvF16Btile(cb, xbuf, W, out,
                              Din: Din, Dout: Dout, activeB: activeB)
    default:
        fail("encDenseGemvAR: unsupported format \(format)")
    }
}

// F16 dense GEMV btile dispatcher — structural mirror of encGemvQ80Btile.
// Picks PSO from denseF16BtileB{1,2,4,8}PSO by activeB rounded up to power
// of 2 (1→b1, 2→b2, {3,4}→b4, {5..8}→b8). Same grid + threadgroup shape:
// 128-thread TG, 4-SG split-K, grid (Dout/32, 1, 1).
func encDenseGemvF16Btile(_ cb: MTLCommandBuffer,
                           _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer,
                           Din: Int, Dout: Int, activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let pso: MTLComputePipelineState
    switch activeB {
    case 1: pso = denseF16BtileB1PSO
    case 2: pso = denseF16BtileB2PSO
    case 3, 4: pso = denseF16BtileB4PSO
    default: pso = denseF16BtileB8PSO   // 5..8
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 3)
    enc.setBytes(&dou, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Per-format btile PSO zoo — index by [activeB → b1/b2/b4/b8] like
// encGemvQ80Btile does. Same dispatch shape (128 threads, grid (Dout/32, 1)).
private let q5kBtileZoo: [MTLComputePipelineState] = [
    denseGemvQ5KBtileB1PSO, denseGemvQ5KBtileB2PSO,
    denseGemvQ5KBtileB4PSO, denseGemvQ5KBtileB8PSO,
]
private let q6kBtileZoo: [MTLComputePipelineState] = [
    denseGemvQ6KBtileB1PSO, denseGemvQ6KBtileB2PSO,
    denseGemvQ6KBtileB4PSO, denseGemvQ6KBtileB8PSO,
]
private let q51BtileZoo: [MTLComputePipelineState] = [
    denseGemvQ51BtileB1PSO, denseGemvQ51BtileB2PSO,
    denseGemvQ51BtileB4PSO, denseGemvQ51BtileB8PSO,
]
private let q40BtileZoo: [MTLComputePipelineState] = [
    denseGemvQ40BtileB1PSO, denseGemvQ40BtileB2PSO,
    denseGemvQ40BtileB4PSO, denseGemvQ40BtileB8PSO,
]
private let q41BtileZoo: [MTLComputePipelineState] = [
    denseGemvQ41BtileB1PSO, denseGemvQ41BtileB2PSO,
    denseGemvQ41BtileB4PSO, denseGemvQ41BtileB8PSO,
]

// Underlying btile dispatcher: picks PSO by activeB rounded up to power of 2,
// dispatches with the same grid + threadgroup shape as Q8_0's btile zoo.
func encDenseGemvBtile(_ cb: MTLCommandBuffer,
                        _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer,
                        psoZoo: [MTLComputePipelineState],
                        Din: Int, Dout: Int, activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let psoSel: MTLComputePipelineState
    switch activeB {
    case 1: psoSel = psoZoo[0]
    case 2: psoSel = psoZoo[1]
    case 3, 4: psoSel = psoZoo[2]
    default: psoSel = psoZoo[3]   // 5..8
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 3)
    enc.setBytes(&dou, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Underlying simple-template AR GEMV dispatcher (used by Q5_K/Q6_K/Q5_1).
// Dispatch shape: 32-thread TGs, grid = (Dout/32, 1, 1), inner-loop over B
// in the kernel (B reads as a uniform from buffer 3).
func encDenseGemvSimpleAR(_ cb: MTLCommandBuffer,
                           _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer,
                           pso psoSel: MTLComputePipelineState,
                           Din: Int, Dout: Int, B: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var bC = UInt32(B), kC = UInt32(Din), nC = UInt32(Dout)
    enc.setBytes(&bC, length: 4, index: 3)
    enc.setBytes(&kC, length: 4, index: 4)
    enc.setBytes(&nC, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// QKV-shaped dense AR with fused-RMSNorm fast path. Each format gets a
// structurally-equivalent kernel zoo (4-phase: RMS reduction, slab routing,
// OTF-RMSNorm GEMV, per-batch reduction); only the per-element dequant in
// Phase 3 differs across formats. Falls back to unfused (RMSNorm + 3 plain
// GEMVs) only when the format mix is genuinely heterogeneous (Q5_K_M can
// have attn_q at Q5_K but attn_v at Q6_K — no single fused kernel covers
// that, so unfused is the correct path there).
func encQKVDenseAR(_ cb: MTLCommandBuffer,
                    x: MTLBuffer, gammaBuf: MTLBuffer, xNormBuf: MTLBuffer,
                    Wq: MTLBuffer, qFmt: GGMLType,
                    Wk: MTLBuffer, kFmt: GGMLType,
                    Wv: MTLBuffer, vFmt: GGMLType,
                    outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                    Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                    activeB: Int) {
    // All-Q8_0: existing fused fast path.
    if qFmt == .q8_0 && kFmt == .q8_0 && vFmt == .q8_0 {
        encGemvQ80BtileQKVOtf(cb, x: x, gammaBuf: gammaBuf,
                                Wq: Wq, Wk: Wk, Wv: Wv,
                                outQ: outQ, outK: outK, outV: outV,
                                Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                                activeB: activeB)
        return
    }
    // All-Q5_K: structural mirror of Q8_0 fused fast path, Q5_K dequant.
    if qFmt == .q5_K && kFmt == .q5_K && vFmt == .q5_K {
        encGemvQ5KBtileQKVOtf(cb, x: x, gammaBuf: gammaBuf,
                                Wq: Wq, Wk: Wk, Wv: Wv,
                                outQ: outQ, outK: outK, outV: outV,
                                Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                                activeB: activeB)
        return
    }
    // All-Q6_K: structural mirror with Q6_K dequant.
    if qFmt == .q6_K && kFmt == .q6_K && vFmt == .q6_K {
        encGemvQ6KBtileQKVOtf(cb, x: x, gammaBuf: gammaBuf,
                                Wq: Wq, Wk: Wk, Wv: Wv,
                                outQ: outQ, outK: outK, outV: outV,
                                Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                                activeB: activeB)
        return
    }
    // All-Q5_1: structural mirror with Q5_1 dequant.
    if qFmt == .q5_1 && kFmt == .q5_1 && vFmt == .q5_1 {
        encGemvQ51BtileQKVOtf(cb, x: x, gammaBuf: gammaBuf,
                               Wq: Wq, Wk: Wk, Wv: Wv,
                               outQ: outQ, outK: outK, outV: outV,
                               Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                               activeB: activeB)
        return
    }
    // All-Q4_0: structural mirror with Q4_0 dequant (simplest 4bpw).
    if qFmt == .q4_0 && kFmt == .q4_0 && vFmt == .q4_0 {
        encGemvQ40BtileQKVOtf(cb, x: x, gammaBuf: gammaBuf,
                               Wq: Wq, Wk: Wk, Wv: Wv,
                               outQ: outQ, outK: outK, outV: outV,
                               Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                               activeB: activeB)
        return
    }
    // All-Q4_1: structural mirror with Q4_1 dequant (4bpw + additive offset).
    if qFmt == .q4_1 && kFmt == .q4_1 && vFmt == .q4_1 {
        encGemvQ41BtileQKVOtf(cb, x: x, gammaBuf: gammaBuf,
                               Wq: Wq, Wk: Wk, Wv: Wv,
                               outQ: outQ, outK: outK, outV: outV,
                               Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                               activeB: activeB)
        return
    }
    // All-F16: structural mirror of Q8_0 fused fast path with raw fp16 weights.
    if qFmt == .f16 && kFmt == .f16 && vFmt == .f16 {
        encGemvF16BtileQKVOtf(cb, x: x, gammaBuf: gammaBuf,
                               Wq: Wq, Wk: Wk, Wv: Wv,
                               outQ: outQ, outK: outK, outV: outV,
                               Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                               activeB: activeB)
        return
    }
    // Heterogeneous format mix: unfused (RMSNorm + 3 plain GEMVs).
    encRMSNormG(cb, x: x, gammaBuf: gammaBuf, out: xNormBuf, D: Din, numVecs: activeB)
    encDenseGemvAR(cb, xNormBuf, Wq, format: qFmt, outQ, Din: Din, Dout: DoutQ, activeB: activeB)
    encDenseGemvAR(cb, xNormBuf, Wk, format: kFmt, outK, Din: Din, Dout: DoutK, activeB: activeB)
    encDenseGemvAR(cb, xNormBuf, Wv, format: vFmt, outV, Din: Din, Dout: DoutV, activeB: activeB)
}

// Q5_K fused-RMSNorm QKV dispatcher — same shape as encGemvQ80BtileQKVOtf,
// uses the Q5_K btile_qkv_otf zoo. activeB-templated PSO selection mirrors
// Q8_0's policy (b{1,2,4,8} rounded up).
func encGemvQ5KBtileQKVOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                            Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                            outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                            Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                            activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let psoSel: MTLComputePipelineState
    let bTile: Int
    switch activeB {
    case 1: psoSel = denseGemvQ5KBtileQkvOtfB1PSO; bTile = 1
    case 2: psoSel = denseGemvQ5KBtileQkvOtfB2PSO; bTile = 2
    case 3, 4: psoSel = denseGemvQ5KBtileQkvOtfB4PSO; bTile = 4
    default: psoSel = denseGemvQ5KBtileQkvOtfB8PSO; bTile = 8
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
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
    var nv = UInt32(activeB)
    enc.setBytes(&du, length: 4, index: 8)
    enc.setBytes(&qnb, length: 4, index: 9)
    enc.setBytes(&knb, length: 4, index: 10)
    enc.setBytes(&vnb, length: 4, index: 11)
    enc.setBytes(&douq, length: 4, index: 12)
    enc.setBytes(&douk, length: 4, index: 13)
    enc.setBytes(&douv, length: 4, index: 14)
    enc.setBytes(&eps, length: 4, index: 15)
    enc.setBytes(&nv, length: 4, index: 16)
    let totalSlabs = (DoutQ + DoutK + DoutV) / 32
    let yBlocks = (activeB + bTile - 1) / bTile
    enc.dispatchThreadgroups(MTLSize(width: totalSlabs, height: yBlocks, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// F16 fused-RMSNorm QKV dispatcher.
//
// History: only b1 OTF was originally shipped, mirroring Q8_0's policy
// (V6 fallback at activeB > 1).
//
// 2026-05-07 falsification probe: shipped F16 OTF b{2,4,8} kernels and
// routed them at activeB > 1, hypothesizing that without the per-FMA
// dequant cost Q8_0 carries, F16's OTF would beat V6 by amortizing RMS
// across B_TILE batches per TG. Result: production AR-tick wall went
// from ~85 ms to ~96 ms (-13% throughput). The hypothesis was wrong.
// The likely cause is GPU saturation: V6 dispatches `slabs × activeB`
// TGs (~500-770 for full QKV at activeB=8) while OTF b8 dispatches
// only `slabs` TGs (~64-96), starving the M5 Max's 40 GPU cores of
// parallel work. The 8× weight-bandwidth amortization OTF gains is
// outweighed by the saturation loss.
//
// Current policy: keep the b{2,4,8} kernels available (zero cost to
// ship), but route via V6 fallback at activeB > 1. activeB=1 still
// uses OTF b1 since V6 has nothing to amortize at single-stream.
func encGemvF16BtileQKVOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                            Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                            outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                            Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                            activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    if activeB > 1 {
        encGemvF16V6RmsnormQKV(cb, x: x, gammaBuf: gammaBuf,
                                Wq: Wq, Wk: Wk, Wv: Wv,
                                outQ: outQ, outK: outK, outV: outV,
                                Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                                numVecs: activeB)
        return
    }
    let pso = denseF16BtileQkvOtfB1PSO
    let bTile = 1
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
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
    var nv = UInt32(activeB)
    enc.setBytes(&du, length: 4, index: 8)
    enc.setBytes(&qnb, length: 4, index: 9)
    enc.setBytes(&knb, length: 4, index: 10)
    enc.setBytes(&vnb, length: 4, index: 11)
    enc.setBytes(&douq, length: 4, index: 12)
    enc.setBytes(&douk, length: 4, index: 13)
    enc.setBytes(&douv, length: 4, index: 14)
    enc.setBytes(&eps, length: 4, index: 15)
    enc.setBytes(&nv, length: 4, index: 16)
    let totalSlabs = (DoutQ + DoutK + DoutV) / 32
    let yBlocks = (activeB + bTile - 1) / bTile
    enc.dispatchThreadgroups(MTLSize(width: totalSlabs, height: yBlocks, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// F16 V6 RMSNorm + QKV fallback — structural mirror of encGemvQ80V6RmsnormQKV.
// Used by encGemvF16BtileQKVOtf when activeB > 1 (the F16 zoo only ships
// the otf_b1 PSO; V6 grid-shrink is the higher-batch path).
func encGemvF16V6RmsnormQKV(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                             Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                             outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                             Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                             numVecs: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseF16V6RmsQkvPSO)
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

// Bundle of gate+up+gelu_mul as one logical operation. Always produces
// post-gelu activation in `gateOut`. Internally:
//   - All-Q8_0: fused-RMSNorm gate+up kernel writes interleaved layout
//     into `fusedScratch`; encMoeGeluMulFused reads interleaved, writes
//     to gateOut.
//   - All-Q5_K: structural mirror — fused-RMSNorm gate+up Q5_K kernel
//     writes the same interleaved layout; encMoeGeluMulFused unchanged.
//   - Mixed/other: explicit RMSNorm into xNormBuf, then plain gate
//     GEMV into gateOut and plain up GEMV into fusedScratch (both as
//     contiguous half buffers, NOT interleaved); encGeluMulInplace
//     consumes the two separate halves and writes back into gateOut.
//
// Either way, gateOut holds the post-gelu_mul activation when this
// function returns. Downstream ffn_down GEMV reads from gateOut as a
// plain [activeB, Dout] buffer.
func encGateUpAR(_ cb: MTLCommandBuffer,
                  x: MTLBuffer, gammaBuf: MTLBuffer, xNormBuf: MTLBuffer,
                  Wg: MTLBuffer, gateFmt: GGMLType,
                  Wu: MTLBuffer, upFmt: GGMLType,
                  gateOut: MTLBuffer, fusedScratch: MTLBuffer,
                  Din: Int, Dout: Int, activeB: Int) {
    if gateFmt == .q8_0 && upFmt == .q8_0 {
        // All-Q8_0 fast path: existing fused-RMSNorm gate+up + interleaved gelu_mul.
        encGemvQ80BtileGateUpOtf(cb, x: x, gammaBuf: gammaBuf,
                                   Wg: Wg, Wu: Wu, fusedOut: fusedScratch,
                                   Din: Din, Dout: Dout, activeB: activeB)
        encMoeGeluMulFused(cb, fused: fusedScratch, out: gateOut,
                            N_half: Dout, numSlots: activeB)
        return
    }
    if gateFmt == .q5_K && upFmt == .q5_K {
        // All-Q5_K fast path: structural mirror of Q8_0 with Q5_K dequant.
        // Produces the same interleaved [slot, 2*Dout] layout, so the
        // existing encMoeGeluMulFused consumes it unchanged.
        encGemvQ5KBtileGateUpOtf(cb, x: x, gammaBuf: gammaBuf,
                                  Wg: Wg, Wu: Wu, fusedOut: fusedScratch,
                                  Din: Din, Dout: Dout, activeB: activeB)
        encMoeGeluMulFused(cb, fused: fusedScratch, out: gateOut,
                            N_half: Dout, numSlots: activeB)
        return
    }
    if gateFmt == .q6_K && upFmt == .q6_K {
        encGemvQ6KBtileGateUpOtf(cb, x: x, gammaBuf: gammaBuf,
                                  Wg: Wg, Wu: Wu, fusedOut: fusedScratch,
                                  Din: Din, Dout: Dout, activeB: activeB)
        encMoeGeluMulFused(cb, fused: fusedScratch, out: gateOut,
                            N_half: Dout, numSlots: activeB)
        return
    }
    if gateFmt == .q5_1 && upFmt == .q5_1 {
        encGemvQ51BtileGateUpOtf(cb, x: x, gammaBuf: gammaBuf,
                                  Wg: Wg, Wu: Wu, fusedOut: fusedScratch,
                                  Din: Din, Dout: Dout, activeB: activeB)
        encMoeGeluMulFused(cb, fused: fusedScratch, out: gateOut,
                            N_half: Dout, numSlots: activeB)
        return
    }
    if gateFmt == .q4_0 && upFmt == .q4_0 {
        encGemvQ40BtileGateUpOtf(cb, x: x, gammaBuf: gammaBuf,
                                  Wg: Wg, Wu: Wu, fusedOut: fusedScratch,
                                  Din: Din, Dout: Dout, activeB: activeB)
        encMoeGeluMulFused(cb, fused: fusedScratch, out: gateOut,
                            N_half: Dout, numSlots: activeB)
        return
    }
    if gateFmt == .q4_1 && upFmt == .q4_1 {
        encGemvQ41BtileGateUpOtf(cb, x: x, gammaBuf: gammaBuf,
                                  Wg: Wg, Wu: Wu, fusedOut: fusedScratch,
                                  Din: Din, Dout: Dout, activeB: activeB)
        encMoeGeluMulFused(cb, fused: fusedScratch, out: gateOut,
                            N_half: Dout, numSlots: activeB)
        return
    }
    if gateFmt == .f16 && upFmt == .f16 {
        // All-F16 fast path: structural mirror of Q8_0 with raw fp16 weights.
        encGemvF16BtileGateUpOtf(cb, x: x, gammaBuf: gammaBuf,
                                  Wg: Wg, Wu: Wu, fusedOut: fusedScratch,
                                  Din: Din, Dout: Dout, activeB: activeB)
        encMoeGeluMulFused(cb, fused: fusedScratch, out: gateOut,
                            N_half: Dout, numSlots: activeB)
        return
    }
    // Heterogeneous: explicit RMSNorm + 2 plain GEMVs + split-halves gelu_mul.
    encRMSNormG(cb, x: x, gammaBuf: gammaBuf, out: xNormBuf, D: Din, numVecs: activeB)
    encDenseGemvAR(cb, xNormBuf, Wg, format: gateFmt, gateOut,
                    Din: Din, Dout: Dout, activeB: activeB)
    encDenseGemvAR(cb, xNormBuf, Wu, format: upFmt, fusedScratch,
                    Din: Din, Dout: Dout, activeB: activeB)
    encGeluMulInplace(cb, gate: gateOut, up: fusedScratch,
                       N_half: Dout, numSlots: activeB)
}

// Q5_K fused-RMSNorm gate+up dispatcher — same buffer convention as
// encGemvQ80BtileGateUpOtf, picks Q5_K btile_gate_up_otf zoo by activeB.
func encGemvQ5KBtileGateUpOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                                Wg: MTLBuffer, Wu: MTLBuffer, fusedOut: MTLBuffer,
                                Din: Int, Dout: Int, activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let psoSel: MTLComputePipelineState
    let bTile: Int
    switch activeB {
    case 1: psoSel = denseGemvQ5KBtileGateUpOtfB1PSO; bTile = 1
    case 2: psoSel = denseGemvQ5KBtileGateUpOtfB2PSO; bTile = 2
    case 3, 4: psoSel = denseGemvQ5KBtileGateUpOtfB4PSO; bTile = 4
    default: psoSel = denseGemvQ5KBtileGateUpOtfB8PSO; bTile = 8
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(gammaBuf, offset: 0, index: 1)
    enc.setBuffer(Wg, offset: 0, index: 2)
    enc.setBuffer(Wu, offset: 0, index: 3)
    enc.setBuffer(fusedOut, offset: 0, index: 4)
    var du = UInt32(Din), dou = UInt32(Dout); var eps: Float = 1e-6
    var nv = UInt32(activeB)
    enc.setBytes(&du, length: 4, index: 5)
    enc.setBytes(&dou, length: 4, index: 6)
    enc.setBytes(&eps, length: 4, index: 7)
    enc.setBytes(&nv, length: 4, index: 8)
    let yBlocks = (activeB + bTile - 1) / bTile
    enc.dispatchThreadgroups(MTLSize(width: 2 * (Dout / 32), height: yBlocks, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// F16 fused-RMSNorm gate+up dispatcher. Same falsification story as
// encGemvF16BtileQKVOtf above (b{2,4,8} routing measured slower than
// V6 fallback at activeB > 1, ~13% AR-tick wall regression). Kernels
// are shipped for future use but production routes via V6 fallback
// at activeB > 1; only activeB=1 uses OTF b1.
func encGemvF16BtileGateUpOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                                Wg: MTLBuffer, Wu: MTLBuffer, fusedOut: MTLBuffer,
                                Din: Int, Dout: Int, activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    if activeB > 1 {
        encGemvF16V6RmsnormGateUp(cb, x: x, gammaBuf: gammaBuf,
                                    Wg: Wg, Wu: Wu, fusedOut: fusedOut,
                                    Din: Din, Dout: Dout, numVecs: activeB)
        return
    }
    let pso = denseF16BtileGateUpOtfB1PSO
    let bTile = 1
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(gammaBuf, offset: 0, index: 1)
    enc.setBuffer(Wg, offset: 0, index: 2)
    enc.setBuffer(Wu, offset: 0, index: 3)
    enc.setBuffer(fusedOut, offset: 0, index: 4)
    var du = UInt32(Din), dou = UInt32(Dout); var eps: Float = 1e-6
    var nv = UInt32(activeB)
    enc.setBytes(&du, length: 4, index: 5)
    enc.setBytes(&dou, length: 4, index: 6)
    enc.setBytes(&eps, length: 4, index: 7)
    enc.setBytes(&nv, length: 4, index: 8)
    let yBlocks = (activeB + bTile - 1) / bTile
    enc.dispatchThreadgroups(MTLSize(width: 2 * (Dout / 32), height: yBlocks, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// F16 V6 RMSNorm + gate+up fallback — structural mirror of
// encGemvQ80V6RmsnormGateUp. Used by encGemvF16BtileGateUpOtf when activeB
// > 1 (only otf_b1 ships in the F16 zoo).
func encGemvF16V6RmsnormGateUp(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                                 Wg: MTLBuffer, Wu: MTLBuffer, fusedOut: MTLBuffer,
                                 Din: Int, Dout: Int, numVecs: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseF16V6RmsGateUpPSO)
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

// MoE up GEMV with format-aware LUT. Q4_K + Q5_K → V11 templated zoo
// (same MAX_SLOTS register-tile pattern, slot_token broadcast convention).
func encMoeUpGemvAR(_ cb: MTLCommandBuffer,
                     _ xbuf: MTLBuffer, _ W: MTLBuffer, format: GGMLType, _ out: MTLBuffer,
                     Din: Int, Dout: Int, numActive: Int, activeB: Int) {
    switch format {
    case .q4_K:
        encMoeGemvQ4KV11(cb, xbuf, W, out,
                          Din: Din, Dout: Dout, numActive: numActive, activeB: activeB)
    case .q5_K:
        encMoeGemvQ5KV11(cb, xbuf, W, out,
                          Din: Din, Dout: Dout, numActive: numActive, activeB: activeB)
    case .q4_0:
        encMoeGemvQ40V11(cb, xbuf, W, out,
                          Din: Din, Dout: Dout, numActive: numActive, activeB: activeB)
    case .q4_1:
        encMoeGemvQ41V11(cb, xbuf, W, out,
                          Din: Din, Dout: Dout, numActive: numActive, activeB: activeB)
    case .f16:
        // Use the slot_token-broadcast moe-up variant. Sharing the per-slot
        // moe-down kernel here was the cause of structurally-wrong logits
        // when the fp16 GGUF first ran end-to-end (2026-05-05): each MoE
        // layer read neighbor slots' rows for hidden activations instead
        // of the source-token row, corrupting the residual stream.
        encMoeGemvF16V11Up(cb, xbuf, W, out,
                            Din: Din, Dout: Dout, numActive: numActive, activeB: activeB)
    default:
        fail("encMoeUpGemvAR: unsupported MoE-up format \(format)")
    }
}

// MoE down GEMV with format-aware LUT. All formats → V11 templated zoo
// with explicit per-slot convention (kernel reads `hidden + idx * D_in`,
// not `hidden + slot_token[idx] * D_in`). Each format has its own dedicated
// down-convention kernel — no convention-routing tricks.
func encMoeDownGemvAR(_ cb: MTLCommandBuffer,
                       _ xbuf: MTLBuffer, _ W: MTLBuffer, format: GGMLType, _ out: MTLBuffer,
                       Din: Int, Dout: Int, numActive: Int, activeB: Int) {
    switch format {
    case .q5_1:
        encMoeGemvQ51V11(cb, xbuf, W, out,
                          Din: Din, Dout: Dout, numActive: numActive, activeB: activeB)
    case .q8_0:
        encMoeGemvQ80V11(cb, xbuf, W, out,
                          Din: Din, Dout: Dout, numActive: numActive, activeB: activeB)
    case .q6_K:
        encMoeGemvQ6KV11(cb, xbuf, W, out,
                          Din: Din, Dout: Dout, numActive: numActive, activeB: activeB)
    case .q4_0:
        encMoeGemvQ40V11Down(cb, xbuf, W, out,
                              Din: Din, Dout: Dout, numActive: numActive, activeB: activeB)
    case .q4_1:
        encMoeGemvQ41V11Down(cb, xbuf, W, out,
                              Din: Din, Dout: Dout, numActive: numActive, activeB: activeB)
    case .f16:
        encMoeGemvF16V11(cb, xbuf, W, out,
                          Din: Din, Dout: Dout, numActive: numActive, activeB: activeB)
    default:
        fail("encMoeDownGemvAR: unsupported MoE-down format \(format)")
    }
}

// Q5_K MoE V11 dispatcher — structural mirror of encMoeGemvQ4KV11.
// activeB rounded up to power of 2: 1→b1, 2→b2, {3,4}→b4, {5..8}→b8.
func encMoeGemvQ5KV11(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq5k: MTLBuffer, _ out: MTLBuffer,
                       Din: Int, Dout: Int, numActive: Int, activeB: Int,
                       slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let psoSel: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:    psoSel = moeGemvQ5KV11B1PSO
    case 2:    psoSel = moeGemvQ5KV11B2PSO
    case 3, 4: psoSel = moeGemvQ5KV11B4PSO
    default:   psoSel = moeGemvQ5KV11B8PSO
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(slotTokenBuf ?? slot_token, offset: 0, index: 1)
    enc.setBuffer(Wq5k, offset: 0, index: 2); enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(groupStartBuf ?? group_start, offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&dou, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// F16 MoE-DOWN V11 dispatcher (per-slot convention) — structural mirror of
// encMoeGemvQ80V11. Same MAX_SLOTS register-tile pattern; per-element
// weights are raw fp16 (no dequant). The kernel ignores slot_token because
// at moe_down each slot has its own input row from the prior MoE-up.
// Use ONLY for ffn_down_exps. For ffn_gate_up_exps see
// encMoeGemvF16V11Up below — that one uses the slot_token-broadcast
// convention which is required when one source token is routed to
// multiple experts.
func encMoeGemvF16V11(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wf16: MTLBuffer, _ out: MTLBuffer,
                       Din: Int, Dout: Int, numActive: Int, activeB: Int,
                       slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let psoSel: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:    psoSel = moeGemvF16V11B1PSO
    case 2:    psoSel = moeGemvF16V11B2PSO
    case 3, 4: psoSel = moeGemvF16V11B4PSO
    default:   psoSel = moeGemvF16V11B8PSO
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(slotTokenBuf ?? slot_token, offset: 0, index: 1)
    enc.setBuffer(Wf16, offset: 0, index: 2); enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(groupStartBuf ?? group_start, offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&dou, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// F16 MoE-UP V11 dispatcher (slot_token-broadcast convention) — for
// ffn_gate_up_exps where one source token may activate multiple experts.
// Mirrors encMoeGemvQ4KV11 / encMoeGemvQ5KV11 in indirection pattern;
// uses the new moeGemvF16V11Up{B1,B2,B4,B8}PSO zoo. Same MAX_SLOTS
// register-tile + per-element fp16 reads as the down kernel; only the
// per-slot hidden-pointer indirection differs.
func encMoeGemvF16V11Up(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wf16: MTLBuffer, _ out: MTLBuffer,
                         Din: Int, Dout: Int, numActive: Int, activeB: Int,
                         slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let psoSel: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:    psoSel = moeGemvF16V11UpB1PSO
    case 2:    psoSel = moeGemvF16V11UpB2PSO
    case 3, 4: psoSel = moeGemvF16V11UpB4PSO
    default:   psoSel = moeGemvF16V11UpB8PSO
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(slotTokenBuf ?? slot_token, offset: 0, index: 1)
    enc.setBuffer(Wf16, offset: 0, index: 2); enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(groupStartBuf ?? group_start, offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&dou, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Q8_0 MoE V11 dispatcher (per-slot convention) — structural mirror of
// encMoeGemvQ51V11. Used for ffn_down_exps when format is Q8_0 (Q5_K_M).
func encMoeGemvQ80V11(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq80: MTLBuffer, _ out: MTLBuffer,
                       Din: Int, Dout: Int, numActive: Int, activeB: Int,
                       slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let psoSel: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:    psoSel = moeGemvQ80V11B1PSO
    case 2:    psoSel = moeGemvQ80V11B2PSO
    case 3, 4: psoSel = moeGemvQ80V11B4PSO
    default:   psoSel = moeGemvQ80V11B8PSO
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(slotTokenBuf ?? slot_token, offset: 0, index: 1)
    enc.setBuffer(Wq80, offset: 0, index: 2); enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(groupStartBuf ?? group_start, offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&dou, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Q6_K MoE V11 dispatcher (per-slot convention) — structural mirror of
// encMoeGemvQ51V11. For hypothetical configs with Q6_K MoE-down weights.
func encMoeGemvQ6KV11(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq6k: MTLBuffer, _ out: MTLBuffer,
                       Din: Int, Dout: Int, numActive: Int, activeB: Int,
                       slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let psoSel: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:    psoSel = moeGemvQ6KV11B1PSO
    case 2:    psoSel = moeGemvQ6KV11B2PSO
    case 3, 4: psoSel = moeGemvQ6KV11B4PSO
    default:   psoSel = moeGemvQ6KV11B8PSO
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(slotTokenBuf ?? slot_token, offset: 0, index: 1)
    enc.setBuffer(Wq6k, offset: 0, index: 2); enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(groupStartBuf ?? group_start, offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&dou, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Q6_K fused-RMSNorm QKV dispatcher — structural mirror of encGemvQ80BtileQKVOtf.
func encGemvQ6KBtileQKVOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                            Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                            outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                            Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                            activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let psoSel: MTLComputePipelineState
    let bTile: Int
    switch activeB {
    case 1: psoSel = denseGemvQ6KBtileQkvOtfB1PSO; bTile = 1
    case 2: psoSel = denseGemvQ6KBtileQkvOtfB2PSO; bTile = 2
    case 3, 4: psoSel = denseGemvQ6KBtileQkvOtfB4PSO; bTile = 4
    default: psoSel = denseGemvQ6KBtileQkvOtfB8PSO; bTile = 8
    }
    encQKVOtfDispatchCommon(cb, pso: psoSel, x: x, gammaBuf: gammaBuf,
                             Wq: Wq, Wk: Wk, Wv: Wv,
                             outQ: outQ, outK: outK, outV: outV,
                             Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                             activeB: activeB, bTile: bTile)
}

// Q5_1 fused-RMSNorm QKV dispatcher.
func encGemvQ51BtileQKVOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                            Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                            outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                            Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                            activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let psoSel: MTLComputePipelineState
    let bTile: Int
    switch activeB {
    case 1: psoSel = denseGemvQ51BtileQkvOtfB1PSO; bTile = 1
    case 2: psoSel = denseGemvQ51BtileQkvOtfB2PSO; bTile = 2
    case 3, 4: psoSel = denseGemvQ51BtileQkvOtfB4PSO; bTile = 4
    default: psoSel = denseGemvQ51BtileQkvOtfB8PSO; bTile = 8
    }
    encQKVOtfDispatchCommon(cb, pso: psoSel, x: x, gammaBuf: gammaBuf,
                             Wq: Wq, Wk: Wk, Wv: Wv,
                             outQ: outQ, outK: outK, outV: outV,
                             Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                             activeB: activeB, bTile: bTile)
}

// Q6_K fused-RMSNorm gate+up dispatcher.
func encGemvQ6KBtileGateUpOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                                Wg: MTLBuffer, Wu: MTLBuffer, fusedOut: MTLBuffer,
                                Din: Int, Dout: Int, activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let psoSel: MTLComputePipelineState
    let bTile: Int
    switch activeB {
    case 1: psoSel = denseGemvQ6KBtileGateUpOtfB1PSO; bTile = 1
    case 2: psoSel = denseGemvQ6KBtileGateUpOtfB2PSO; bTile = 2
    case 3, 4: psoSel = denseGemvQ6KBtileGateUpOtfB4PSO; bTile = 4
    default: psoSel = denseGemvQ6KBtileGateUpOtfB8PSO; bTile = 8
    }
    encGateUpOtfDispatchCommon(cb, pso: psoSel, x: x, gammaBuf: gammaBuf,
                                Wg: Wg, Wu: Wu, fusedOut: fusedOut,
                                Din: Din, Dout: Dout, activeB: activeB, bTile: bTile)
}

// Q5_1 fused-RMSNorm gate+up dispatcher.
func encGemvQ51BtileGateUpOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                                Wg: MTLBuffer, Wu: MTLBuffer, fusedOut: MTLBuffer,
                                Din: Int, Dout: Int, activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let psoSel: MTLComputePipelineState
    let bTile: Int
    switch activeB {
    case 1: psoSel = denseGemvQ51BtileGateUpOtfB1PSO; bTile = 1
    case 2: psoSel = denseGemvQ51BtileGateUpOtfB2PSO; bTile = 2
    case 3, 4: psoSel = denseGemvQ51BtileGateUpOtfB4PSO; bTile = 4
    default: psoSel = denseGemvQ51BtileGateUpOtfB8PSO; bTile = 8
    }
    encGateUpOtfDispatchCommon(cb, pso: psoSel, x: x, gammaBuf: gammaBuf,
                                Wg: Wg, Wu: Wu, fusedOut: fusedOut,
                                Din: Din, Dout: Dout, activeB: activeB, bTile: bTile)
}

// Q4_0 fused-RMSNorm QKV dispatcher.
func encGemvQ40BtileQKVOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                            Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                            outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                            Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                            activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let psoSel: MTLComputePipelineState
    let bTile: Int
    switch activeB {
    case 1: psoSel = denseGemvQ40BtileQkvOtfB1PSO; bTile = 1
    case 2: psoSel = denseGemvQ40BtileQkvOtfB2PSO; bTile = 2
    case 3, 4: psoSel = denseGemvQ40BtileQkvOtfB4PSO; bTile = 4
    default: psoSel = denseGemvQ40BtileQkvOtfB8PSO; bTile = 8
    }
    encQKVOtfDispatchCommon(cb, pso: psoSel, x: x, gammaBuf: gammaBuf,
                             Wq: Wq, Wk: Wk, Wv: Wv,
                             outQ: outQ, outK: outK, outV: outV,
                             Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                             activeB: activeB, bTile: bTile)
}

// Q4_0 fused-RMSNorm gate+up dispatcher.
func encGemvQ40BtileGateUpOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                                Wg: MTLBuffer, Wu: MTLBuffer, fusedOut: MTLBuffer,
                                Din: Int, Dout: Int, activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let psoSel: MTLComputePipelineState
    let bTile: Int
    switch activeB {
    case 1: psoSel = denseGemvQ40BtileGateUpOtfB1PSO; bTile = 1
    case 2: psoSel = denseGemvQ40BtileGateUpOtfB2PSO; bTile = 2
    case 3, 4: psoSel = denseGemvQ40BtileGateUpOtfB4PSO; bTile = 4
    default: psoSel = denseGemvQ40BtileGateUpOtfB8PSO; bTile = 8
    }
    encGateUpOtfDispatchCommon(cb, pso: psoSel, x: x, gammaBuf: gammaBuf,
                                Wg: Wg, Wu: Wu, fusedOut: fusedOut,
                                Din: Din, Dout: Dout, activeB: activeB, bTile: bTile)
}

// Q4_0 MoE V11 dispatcher (slot_token broadcast / gate-up convention).
func encMoeGemvQ40V11(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq40: MTLBuffer, _ out: MTLBuffer,
                       Din: Int, Dout: Int, numActive: Int, activeB: Int,
                       slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let psoSel: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:    psoSel = moeGemvQ40V11B1PSO
    case 2:    psoSel = moeGemvQ40V11B2PSO
    case 3, 4: psoSel = moeGemvQ40V11B4PSO
    default:   psoSel = moeGemvQ40V11B8PSO
    }
    encMoeV11DispatchCommon(cb, pso: psoSel, xbuf: xbuf, W: Wq40, out: out,
                             Din: Din, Dout: Dout, numActive: numActive,
                             slotTokenBuf: slotTokenBuf, groupStartBuf: groupStartBuf)
}

// Q4_0 MoE V11 dispatcher (per-slot / down convention).
func encMoeGemvQ40V11Down(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq40: MTLBuffer, _ out: MTLBuffer,
                            Din: Int, Dout: Int, numActive: Int, activeB: Int,
                            slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let psoSel: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:    psoSel = moeGemvQ40V11DownB1PSO
    case 2:    psoSel = moeGemvQ40V11DownB2PSO
    case 3, 4: psoSel = moeGemvQ40V11DownB4PSO
    default:   psoSel = moeGemvQ40V11DownB8PSO
    }
    encMoeV11DispatchCommon(cb, pso: psoSel, xbuf: xbuf, W: Wq40, out: out,
                             Din: Din, Dout: Dout, numActive: numActive,
                             slotTokenBuf: slotTokenBuf, groupStartBuf: groupStartBuf)
}

// Q4_1 MoE V11 (slot_token broadcast / up).
func encMoeGemvQ41V11(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq41: MTLBuffer, _ out: MTLBuffer,
                       Din: Int, Dout: Int, numActive: Int, activeB: Int,
                       slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let psoSel: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:    psoSel = moeGemvQ41V11B1PSO
    case 2:    psoSel = moeGemvQ41V11B2PSO
    case 3, 4: psoSel = moeGemvQ41V11B4PSO
    default:   psoSel = moeGemvQ41V11B8PSO
    }
    encMoeV11DispatchCommon(cb, pso: psoSel, xbuf: xbuf, W: Wq41, out: out,
                             Din: Din, Dout: Dout, numActive: numActive,
                             slotTokenBuf: slotTokenBuf, groupStartBuf: groupStartBuf)
}

// Q4_1 MoE V11 (per-slot / down).
func encMoeGemvQ41V11Down(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq41: MTLBuffer, _ out: MTLBuffer,
                            Din: Int, Dout: Int, numActive: Int, activeB: Int,
                            slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let psoSel: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:    psoSel = moeGemvQ41V11DownB1PSO
    case 2:    psoSel = moeGemvQ41V11DownB2PSO
    case 3, 4: psoSel = moeGemvQ41V11DownB4PSO
    default:   psoSel = moeGemvQ41V11DownB8PSO
    }
    encMoeV11DispatchCommon(cb, pso: psoSel, xbuf: xbuf, W: Wq41, out: out,
                             Din: Din, Dout: Dout, numActive: numActive,
                             slotTokenBuf: slotTokenBuf, groupStartBuf: groupStartBuf)
}

// Q4_1 fused-RMSNorm QKV.
func encGemvQ41BtileQKVOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                            Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                            outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                            Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                            activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let psoSel: MTLComputePipelineState
    let bTile: Int
    switch activeB {
    case 1: psoSel = denseGemvQ41BtileQkvOtfB1PSO; bTile = 1
    case 2: psoSel = denseGemvQ41BtileQkvOtfB2PSO; bTile = 2
    case 3, 4: psoSel = denseGemvQ41BtileQkvOtfB4PSO; bTile = 4
    default: psoSel = denseGemvQ41BtileQkvOtfB8PSO; bTile = 8
    }
    encQKVOtfDispatchCommon(cb, pso: psoSel, x: x, gammaBuf: gammaBuf,
                             Wq: Wq, Wk: Wk, Wv: Wv,
                             outQ: outQ, outK: outK, outV: outV,
                             Din: Din, DoutQ: DoutQ, DoutK: DoutK, DoutV: DoutV,
                             activeB: activeB, bTile: bTile)
}

// Q4_1 fused-RMSNorm gate+up.
func encGemvQ41BtileGateUpOtf(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                                Wg: MTLBuffer, Wu: MTLBuffer, fusedOut: MTLBuffer,
                                Din: Int, Dout: Int, activeB: Int) {
    precondition(activeB >= 1 && activeB <= 8, "activeB \(activeB) out of [1,8]")
    let psoSel: MTLComputePipelineState
    let bTile: Int
    switch activeB {
    case 1: psoSel = denseGemvQ41BtileGateUpOtfB1PSO; bTile = 1
    case 2: psoSel = denseGemvQ41BtileGateUpOtfB2PSO; bTile = 2
    case 3, 4: psoSel = denseGemvQ41BtileGateUpOtfB4PSO; bTile = 4
    default: psoSel = denseGemvQ41BtileGateUpOtfB8PSO; bTile = 8
    }
    encGateUpOtfDispatchCommon(cb, pso: psoSel, x: x, gammaBuf: gammaBuf,
                                Wg: Wg, Wu: Wu, fusedOut: fusedOut,
                                Din: Din, Dout: Dout, activeB: activeB, bTile: bTile)
}

// Shared encoding for MoE V11 dispatchers — kernel signature is identical
// across formats and conventions (the "up" vs "down" distinction is in the
// kernel body, not the buffer set), so eliminating the duplicated buffer
// binding code makes the dispatch surface uniform. Same pattern as
// encQKVOtfDispatchCommon / encGateUpOtfDispatchCommon.
private func encMoeV11DispatchCommon(_ cb: MTLCommandBuffer,
                                       pso psoSel: MTLComputePipelineState,
                                       xbuf: MTLBuffer, W: MTLBuffer, out: MTLBuffer,
                                       Din: Int, Dout: Int, numActive: Int,
                                       slotTokenBuf: MTLBuffer?, groupStartBuf: MTLBuffer?) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(slotTokenBuf ?? slot_token, offset: 0, index: 1)
    enc.setBuffer(W, offset: 0, index: 2); enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(groupStartBuf ?? group_start, offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&dou, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Shared encoding for QKV-otf dispatchers — kernel signature is identical
// across formats; only the PSO differs. Eliminates duplicated buffer-binding
// code from each per-format wrapper.
private func encQKVOtfDispatchCommon(_ cb: MTLCommandBuffer,
                                      pso psoSel: MTLComputePipelineState,
                                      x: MTLBuffer, gammaBuf: MTLBuffer,
                                      Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                                      outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                                      Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                                      activeB: Int, bTile: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
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
    var nv = UInt32(activeB)
    enc.setBytes(&du, length: 4, index: 8)
    enc.setBytes(&qnb, length: 4, index: 9)
    enc.setBytes(&knb, length: 4, index: 10)
    enc.setBytes(&vnb, length: 4, index: 11)
    enc.setBytes(&douq, length: 4, index: 12)
    enc.setBytes(&douk, length: 4, index: 13)
    enc.setBytes(&douv, length: 4, index: 14)
    enc.setBytes(&eps, length: 4, index: 15)
    enc.setBytes(&nv, length: 4, index: 16)
    let totalSlabs = (DoutQ + DoutK + DoutV) / 32
    let yBlocks = (activeB + bTile - 1) / bTile
    enc.dispatchThreadgroups(MTLSize(width: totalSlabs, height: yBlocks, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Shared encoding for gate+up-otf dispatchers.
private func encGateUpOtfDispatchCommon(_ cb: MTLCommandBuffer,
                                         pso psoSel: MTLComputePipelineState,
                                         x: MTLBuffer, gammaBuf: MTLBuffer,
                                         Wg: MTLBuffer, Wu: MTLBuffer, fusedOut: MTLBuffer,
                                         Din: Int, Dout: Int, activeB: Int, bTile: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(gammaBuf, offset: 0, index: 1)
    enc.setBuffer(Wg, offset: 0, index: 2)
    enc.setBuffer(Wu, offset: 0, index: 3)
    enc.setBuffer(fusedOut, offset: 0, index: 4)
    var du = UInt32(Din), dou = UInt32(Dout); var eps: Float = 1e-6
    var nv = UInt32(activeB)
    enc.setBytes(&du, length: 4, index: 5)
    enc.setBytes(&dou, length: 4, index: 6)
    enc.setBytes(&eps, length: 4, index: 7)
    enc.setBytes(&nv, length: 4, index: 8)
    let yBlocks = (activeB + bTile - 1) / bTile
    enc.dispatchThreadgroups(MTLSize(width: 2 * (Dout / 32), height: yBlocks, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

// Underlying simple-template MoE-up V6 dispatcher (used for Q5_K).
// 32 threads/TG, grid (Dout/32, numActive, 1). Reads slot_token for X-broadcast.
func encMoeUpGemvSimpleAR(_ cb: MTLCommandBuffer,
                           _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer,
                           pso psoSel: MTLComputePipelineState,
                           Din: Int, Dout: Int, numActive: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(slot_token, offset: 0, index: 1)
    enc.setBuffer(W, offset: 0, index: 2)
    enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(group_start, offset: 0, index: 4)
    enc.setBuffer(out, offset: 0, index: 5)
    var kC = UInt32(Din), nC = UInt32(Dout)
    enc.setBytes(&kC, length: 4, index: 6)
    enc.setBytes(&nC, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Underlying simple-template MoE-down V6 dispatcher (used for Q6_K, Q8_0).
// Same layout but no slot_token (per-slot X).
func encMoeDownGemvSimpleAR(_ cb: MTLCommandBuffer,
                             _ xbuf: MTLBuffer, _ W: MTLBuffer, _ out: MTLBuffer,
                             pso psoSel: MTLComputePipelineState,
                             Din: Int, Dout: Int, numActive: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoSel)
    enc.setBuffer(xbuf, offset: 0, index: 0)
    enc.setBuffer(slot_token, offset: 0, index: 1)   // unused by Q6_K/Q8_0 v6 (per-slot conv) but bound for consistency
    enc.setBuffer(W, offset: 0, index: 2)
    enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(group_start, offset: 0, index: 4)
    enc.setBuffer(out, offset: 0, index: 5)
    var kC = UInt32(Din), nC = UInt32(Dout)
    enc.setBytes(&kC, length: 4, index: 6)
    enc.setBytes(&nC, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// gelu_mul_inplace dispatcher — reads two separate gate/up half-buffers,
// applies gelu(gate) * up, writes back into gate. Used by the prefill
// matmul rewire which produces gate and up as separate tensors (instead
// of the AR-path's interleaved fused buffer).
func encGeluMulInplace(_ cb: MTLCommandBuffer,
                        gate: MTLBuffer, gateOffset: Int = 0,
                        up: MTLBuffer, upOffset: Int = 0,
                        N_half: Int, numSlots: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(geluMulPSO)
    enc.setBuffer(gate, offset: gateOffset, index: 0)
    enc.setBuffer(up,   offset: upOffset,   index: 1)
    var Nv = UInt32(N_half)
    enc.setBytes(&Nv, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: numSlots, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
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
//
// V7 attempt (vector-amortized via per-thread accumulator array, like
// V4-for-AR) was benchmarked 2026-04-26 and ran SLOWER than V6 at every
// B in {4, 8, 16}: v7 collapses TG count (4× fewer at VEC_TILE=4) and
// 4×'s tg-mem (22.5 KB), both of which hurt occupancy more than register-
// level weight-amortization helps. Apple GPU's L2 cache was already
// amortizing weight reads across adjacent (slab, vec) TGs, so the
// "redundant DRAM reads" v7 targeted were going to L2 not DRAM, and
// were ~free. v7 kernel + its dispatch helper retained in-tree for
// reference (kernels.swift `dense_gemv_q8_0_v7_rmsnorm_qkv`); not used.
// Real prefill-bandwidth wins live elsewhere — see notes/specs/
// bandwidth_triage.md §5 follow-on items.
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

// V7: vector-amortized variant of encGemvQ80V6RmsnormQKV. Same args,
// different dispatch shape: (totalSlabs, ceil(numVecs/VEC_TILE)) with
// each TG handling VEC_TILE=4 vectors. Per K-tile loads weights once,
// FMAs into VEC_TILE accumulators in registers — structural 4×
// weight-bandwidth reduction for prefill. See kernels.swift:1422.
func encGemvQ80V7RmsnormQKV(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                             Wq: MTLBuffer, Wk: MTLBuffer, Wv: MTLBuffer,
                             outQ: MTLBuffer, outK: MTLBuffer, outV: MTLBuffer,
                             Din: Int, DoutQ: Int, DoutK: Int, DoutV: Int,
                             numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseQ80V7RmsQkvPSO)
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
    var nv = UInt32(numVecs)
    enc.setBytes(&du, length: 4, index: 8)
    enc.setBytes(&qnb, length: 4, index: 9)
    enc.setBytes(&knb, length: 4, index: 10)
    enc.setBytes(&vnb, length: 4, index: 11)
    enc.setBytes(&douq, length: 4, index: 12)
    enc.setBytes(&douk, length: 4, index: 13)
    enc.setBytes(&douv, length: 4, index: 14)
    enc.setBytes(&eps, length: 4, index: 15)
    enc.setBytes(&nv, length: 4, index: 16)
    let VEC_TILE = 4
    let totalSlabs = (DoutQ + DoutK + DoutV) / 32
    let yBlocks = (numVecs + VEC_TILE - 1) / VEC_TILE
    enc.dispatchThreadgroups(MTLSize(width: totalSlabs, height: yBlocks, depth: 1),
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
                    useV8: Bool = false, useV9: Bool = false,
                    slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let enc = cb.makeComputeCommandEncoder()!
    let pso = useV9 ? moeQ51V9PSO
            : (useV8 ? moeQ51V8PSO
            : (useV6 ? moeQ51V6PSO : (useV4 ? moeQ51V4PSO : moeQ51PSO)))
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

// MoE Q4_K v12 dispatcher — picks MAX_SLOTS specialization from activeB.
// Same mapping as V11. Intended for A/B against V11.
func encMoeGemvQ4KV12(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq4k: MTLBuffer, _ out: MTLBuffer,
                       Din: Int, Dout: Int, numActive: Int, activeB: Int,
                       slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let pso: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:           pso = moeQ4KV12B1PSO
    case 2:           pso = moeQ4KV12B2PSO
    case 3, 4:        pso = moeQ4KV12B4PSO
    default:          pso = moeQ4KV12B8PSO
    }
    let enc = cb.makeComputeCommandEncoder()!
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

// MoE Q5_1 v11 dispatcher — picks MAX_SLOTS specialization from activeB,
// same mapping as encMoeGemvQ4KV11. Used for the down projection.
//
// slotTokenBuf parameter is accepted for ABI parity with the V6 dispatcher
// but not actually consumed by V11's kernel (Q5_1 is per-slot indexed for
// the down projection — see moe_gemv_q5_1_v11_impl). groupStartBuf IS used
// (same expert→slot-range mapping as Q4K).
func encMoeGemvQ51V11(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq51: MTLBuffer, _ out: MTLBuffer,
                       Din: Int, Dout: Int, numActive: Int, activeB: Int,
                       slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let pso: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:           pso = moeQ51V11B1PSO
    case 2:           pso = moeQ51V11B2PSO
    case 3, 4:        pso = moeQ51V11B4PSO
    default:          pso = moeQ51V11B8PSO
    }
    let enc = cb.makeComputeCommandEncoder()!
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

// V10 fused matmul + GELU(gate)·up. Output is post-activation
// [slots, N_half], so caller passes the post-activation buffer (e.g.
// pre_gate_proj) directly and skips moe_gelu_mul_fused. Inputs identical
// to encMoeGemvQ4K.
func encMoeGemvQ4KFusedGelu(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq4kFused: MTLBuffer,
                              _ outPostAct: MTLBuffer,
                              Din: Int, NHalf: Int, numActive: Int,
                              slotTokenBuf: MTLBuffer? = nil,
                              groupStartBuf: MTLBuffer? = nil) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(moeQ4KV10PSO)
    enc.setBuffer(xbuf, offset: 0, index: 0); enc.setBuffer(slotTokenBuf ?? slot_token, offset: 0, index: 1)
    enc.setBuffer(Wq4kFused, offset: 0, index: 2); enc.setBuffer(active_exp, offset: 0, index: 3)
    enc.setBuffer(groupStartBuf ?? group_start, offset: 0, index: 4); enc.setBuffer(outPostAct, offset: 0, index: 5)
    var du = UInt32(Din), nh = UInt32(NHalf)
    enc.setBytes(&du, length: 4, index: 6); enc.setBytes(&nh, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: NHalf / 32, height: numActive, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// GGUF-native Q4_K MoE GEMV (valid when D_in divisible by 256). Uses v4
// (k-outer / slot-inner) pattern — amortizes Q4_K dequant across slots
// sharing an expert. Falls back to v3 when useV4=false for benchmarking.
func encMoeGemvQ4K(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq4k: MTLBuffer, _ out: MTLBuffer,
                    Din: Int, Dout: Int, numActive: Int, useV4: Bool = false, useV6: Bool = false,
                    useV8: Bool = false, useV9: Bool = false,
                    slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let enc = cb.makeComputeCommandEncoder()!
    let pso = useV9 ? moeQ4KV9PSO
            : (useV8 ? moeQ4KV8PSO
            : (useV6 ? moeQ4KV6PSO : (useV4 ? moeQ4KV4PSO : moeQ4KPSO)))
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

// MoE Q4_K v11 dispatcher — picks the MAX_SLOTS specialization from
// the typical chunk size at the given activeB. Mapping logic:
//
//   activeB=1 → MAX_SLOTS=1  (with TOPK=8, every active expert holds 1 slot
//                             at single-stream; the SLOTS=1 instantiation
//                             collapses the chunked-slot loop and drops
//                             the predicated-FMA register array entirely)
//   activeB=2 → MAX_SLOTS=2  (worst case 2 colliding tokens → 2 slots)
//   activeB=3-4 → MAX_SLOTS=4
//   activeB=5-8 → MAX_SLOTS=8
//
// MAX_SLOTS=N means the kernel processes up to N slots per chunk; if a
// particular expert has fewer than N slots, the unused lanes are masked
// out by `if (s < chunk)` predicates — no work executes for them.
// MAX_SLOTS larger than the actual chunk only costs unused register
// allocation (an MAX_SLOTS=8 kernel with chunk=1 burns 7 unused
// accumulators), but no spurious memory traffic.
//
// Falls back to V6 when activeB is 0 or unknown.
func encMoeGemvQ4KV11(_ cb: MTLCommandBuffer, _ xbuf: MTLBuffer, _ Wq4k: MTLBuffer, _ out: MTLBuffer,
                       Din: Int, Dout: Int, numActive: Int, activeB: Int,
                       slotTokenBuf: MTLBuffer? = nil, groupStartBuf: MTLBuffer? = nil) {
    let pso: MTLComputePipelineState
    switch max(1, activeB) {
    case 1:           pso = moeQ4KV11B1PSO
    case 2:           pso = moeQ4KV11B2PSO
    case 3, 4:        pso = moeQ4KV11B4PSO
    default:          pso = moeQ4KV11B8PSO
    }
    let enc = cb.makeComputeCommandEncoder()!
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

// activeB-aware grid-shrink: dispatch only [0, activeB) slot rows.
// The kernel itself reads `b = tg.x` and operates on whichever slot
// it gets — silenced slots [activeB, B) simply aren't dispatched.
// Default activeB=B preserves legacy AR-batch-full behavior.
func encRope(_ cb: MTLCommandBuffer, _ x: MTLBuffer, H: Int, D: Int, rotary: Int, theta: Float,
              activeB: Int = B) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(ropePSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(positions, offset: 0, index: 1)
    var hv = UInt32(H), dv = UInt32(D), rv = UInt32(rotary); var tv = theta
    enc.setBytes(&hv, length: 4, index: 2)
    enc.setBytes(&dv, length: 4, index: 3)
    enc.setBytes(&rv, length: 4, index: 4)
    enc.setBytes(&tv, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: activeB, height: H, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encKVWrite(_ cb: MTLCommandBuffer, K: MTLBuffer, V: MTLBuffer, Kc: MTLBuffer, Vc: MTLBuffer,
                H: Int, D: Int, page: Int, activeB: Int = B) {
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
    // v2: emit one CSR entry per (slot, q_block_idx) so qLen > 8 actually
    // works. Kernel-side csr_idx = slot * q_blocks_per_slot + q_block_idx
    // is already in place (`flex_attn_full_prefill`, `flex_attn_slide_v1_q8`),
    // it was just waiting for the host-side widened CSR.
    let qBlock = 8
    let qBlocks = (qLen + qBlock - 1) / qBlock
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
            // Slot's first q position. Fallback to positionStart for the
            // single-slot path that doesn't populate silenced slots.
            let slotPos = Int(qPosP[b * qLen])
            let slotFirst = (slotPos != 0 || positionStart == 0) ? slotPos : positionStart
            let ctx = MaskModContext(kLen: k_len, slidingWindow: SLIDING_WINDOW)
            let kBlocks = (k_len + PAGE_SLIDE - 1) / PAGE_SLIDE
            for qb in 0..<qBlocks {
                let q_first = slotFirst + qb * qBlock
                // Last real q row in this block — capped at the slot's
                // qLen so we don't classify mask cells for padding rows.
                let blockLastIdx = min((qb + 1) * qBlock, qLen) - 1
                let q_last  = slotFirst + blockLastIdx
                let csrIdx = b * qBlocks + qb
                for K in 0..<kBlocks {
                    let lo = K * PAGE_SLIDE
                    let hi = lo + PAGE_SLIDE - 1
                    let topLeft     = slideMask.keep(q: q_first, k: lo, ctx: ctx)
                    let topRight    = slideMask.keep(q: q_first, k: hi, ctx: ctx)
                    let botLeft     = slideMask.keep(q: q_last,  k: lo, ctx: ctx)
                    let botRight    = slideMask.keep(q: q_last,  k: hi, ctx: ctx)
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
                fullOff[csrIdx + 1] = UInt32(fc)
                partOff[csrIdx + 1] = UInt32(pc)
            }
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
            let slotFirst = (slotPos != 0 || positionStart == 0) ? slotPos : positionStart
            let ctx = MaskModContext(kLen: k_len, slidingWindow: 0)
            let kBlocks = (k_len + PAGE_FULL - 1) / PAGE_FULL
            for qb in 0..<qBlocks {
                let q_first = slotFirst + qb * qBlock
                let blockLastIdx = min((qb + 1) * qBlock, qLen) - 1
                let q_last  = slotFirst + blockLastIdx
                let csrIdx = b * qBlocks + qb
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
                fullOff[csrIdx + 1] = UInt32(fc)
                partOff[csrIdx + 1] = UInt32(pc)
            }
        }
    }
}

// Prefill attention dispatchers (slide + full). Called from buildPrefillCB;
// dispatch the flex v1 kernels with (B*H_KV or B*H_Q, q_blocks, N_SPLITS).
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
    // 2026-05-06: x-dim is now B*H_Q (was B*H_KV) after switching
    // flex_attn_slide_v1_q8 to one-q-head-per-TG geometry. Each TG
    // handles a single q_head; previously Q_PER_TG=2 q_heads shared
    // a TG. See kernels.swift comment block on the kernel for the
    // threadgroup-memory rationale (drops static usage from 25,312
    // to 12,672 bytes — under the 32 KB Metal hardware limit even
    // accounting for compiler-side simdgroup-matrix doubling).
    enc1.dispatchThreadgroups(MTLSize(width: B * H_Q, height: qBlocks, depth: ATTN_N_SPLITS),
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

// Per-slot AR attention dispatcher. The `isFull` flag selects the
// head-dim and mask variant of the PSO. Gemma-4's architecture is
// compile-time-partitioned per layer into full-attention (D=512,
// causal) and sliding-window (D=256, causal + window mask) layers,
// so this is a per-layer config flag, not a runtime decision.
func encAttn(_ cb: MTLCommandBuffer,
             Q: MTLBuffer, O: MTLBuffer,
             Kc: MTLBuffer, Vc: MTLBuffer,
             kLenBuf: MTLBuffer,
             H_Q: Int, H_KV: Int, D: Int,
             isFull: Bool,
             activeB: Int = B) {
    encFlexAttnV0(cb, Q: Q, O: O, Kc: Kc, Vc: Vc,
                   kLenBuf: kLenBuf, H_Q: H_Q, H_KV: H_KV, D: D,
                   isFull: isFull, activeB: activeB)
}

// GPU-side sampling dispatcher — Phase 1 of the dataflow pipeline spec.
// Encodes the `sample_token` MSL kernel as the last dispatch of a step
// CB. Reads per-slot logits, writes sampled token ids into `inputTokens`
// which the NEXT step's embed kernel reads.
//
// Direct port of CPU `sampleTokenFromLogits`: inverse-CDF softmax
// sampling with temperature + min_p, argmax fast path at T <= 0. PRNG
// is philox-4x32-10 keyed on (seed, step, slot) — statistically
// equivalent to Swift's stdlib RNG, specific draws differ.
//
//   logits         [B, VOCAB] fp16 — typically the engine's `logits` buffer
//   samplingTemp   [B] fp32
//   samplingMinP   [B] fp32
//   samplingSeed   [B] uint32 — per-slot RNG seed
//   samplingStep   [B] uint32 — advances each AR step for fresh draws
//   samplingActive [B] uint32 — 0=skip this slot (idle/closed)
//   inputTokens    [B] uint32 — output buffer; kernel writes sampled ids
//
// Logit-bias is not yet supported by this kernel — sessions that set a
// bias must use the CPU sampling fallback (Phase 1b adds GPU bias).
// Soft-prompt softs ingest. One compute encoder, K dispatches: copies-and-
// casts `qLen × HIDDEN` fp32 rows from each slot's `src` buffer (at
// `srcByteOffset` bytes) into `dst` rows [dstSlot * qLen ..) (fp16).
//
// Why a single encoder instead of K separate encoders: with K separate
// encoders we observed K≥2 garbage output (`<unused6226>` repetition).
// Empirically the pre_hidden writes from later dispatches don't actually
// land — likely a per-encoder argument-table reuse with the inline
// setBytes constants. One encoder + K dispatchThreads, with full
// rebinding per dispatch, sidesteps that and gives correct output at
// any K.
struct VisionSoftsIngestSlot {
    let src: MTLBuffer
    let srcByteOffset: Int
    let dstSlot: Int
}

func encVisionSoftsCopyFp32Multi(_ cb: MTLCommandBuffer,
                                   slots: [VisionSoftsIngestSlot],
                                   dst: MTLBuffer,
                                   qLen: Int) {
    if slots.isEmpty { return }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(visionSoftsCopyFp32PSO)
    enc.setBuffer(dst, offset: 0, index: 1)
    var nRows = UInt32(qLen)
    var hidden = UInt32(HIDDEN)
    enc.setBytes(&nRows, length: 4, index: 4)
    enc.setBytes(&hidden, length: 4, index: 5)
    let total = qLen * HIDDEN
    let tg = 256
    let grid = MTLSize(width: total, height: 1, depth: 1)
    let tgSize = MTLSize(width: tg, height: 1, depth: 1)
    for s in slots {
        enc.setBuffer(s.src, offset: 0, index: 0)
        var srcRowOff = UInt32(s.srcByteOffset / (HIDDEN * 4))
        var dstRowOff = UInt32(s.dstSlot * qLen)
        enc.setBytes(&srcRowOff, length: 4, index: 2)
        enc.setBytes(&dstRowOff, length: 4, index: 3)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tgSize)
    }
    enc.endEncoding()
}

// Single-slot wrapper kept for the single-slot prefill path (which already
// works correctly because there's only one dispatch per CB).
func encVisionSoftsCopyFp32(_ cb: MTLCommandBuffer,
                              src: MTLBuffer, srcByteOffset: Int,
                              dst: MTLBuffer,
                              dstSlot: Int, qLen: Int) {
    encVisionSoftsCopyFp32Multi(cb, slots: [
        VisionSoftsIngestSlot(src: src, srcByteOffset: srcByteOffset, dstSlot: dstSlot)
    ], dst: dst, qLen: qLen)
}

func encSampleToken(_ cb: MTLCommandBuffer,
                     logits: MTLBuffer,
                     samplingLogitBias: MTLBuffer,
                     samplingTemp: MTLBuffer,
                     samplingMinP: MTLBuffer,
                     samplingSeed: MTLBuffer,
                     samplingStep: MTLBuffer,
                     samplingActive: MTLBuffer,
                     inputTokens: MTLBuffer,
                     vocab: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(sampleTokenPSO)
    enc.setBuffer(logits,            offset: 0, index: 0)
    enc.setBuffer(samplingLogitBias, offset: 0, index: 1)
    enc.setBuffer(samplingTemp,      offset: 0, index: 2)
    enc.setBuffer(samplingMinP,      offset: 0, index: 3)
    enc.setBuffer(samplingSeed,      offset: 0, index: 4)
    enc.setBuffer(samplingStep,      offset: 0, index: 5)
    enc.setBuffer(samplingActive,    offset: 0, index: 6)
    enc.setBuffer(inputTokens,       offset: 0, index: 7)
    var v = UInt32(vocab)
    enc.setBytes(&v, length: 4, index: 8)
    // One TG per slot, 32 threads (one simdgroup) per TG.
    enc.dispatchThreadgroups(MTLSize(width: B, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Unified flex-attention dispatcher — one Swift path, one MSL source.
// Picks the specialized PSO (flexAttn{Slide,Full}V0PSO) and the matching
// CSR buffer set (flex_full_* at PAGE=16 vs flex_full_full_* at PAGE=8)
// by the per-layer `isFull` flag. Both PSOs are compiled from the same
// `flex_attn_v0` kernel source; function constants select head-dim,
// page size, Q-per-TG, and sliding-window enablement.
//
// This is called from encAttn when the scheduler determines a layer
// should NOT take the shared-prefix broadcast path.
func encFlexAttnV0(_ cb: MTLCommandBuffer, Q: MTLBuffer, O: MTLBuffer,
                    Kc: MTLBuffer, Vc: MTLBuffer,
                    kLenBuf: MTLBuffer,
                    H_Q: Int, H_KV: Int, D: Int,
                    isFull: Bool,
                    activeB: Int = B) {
    let psoSel: MTLComputePipelineState = isFull ? flexAttnFullV0PSO : flexAttnSlideV0PSO
    let fullOff = isFull ? flex_full_full_offsets   : flex_full_offsets
    let fullIdx = isFull ? flex_full_full_indices   : flex_full_indices
    let partOff = isFull ? flex_full_partial_offsets : flex_partial_offsets
    let partIdx = isFull ? flex_full_partial_indices : flex_partial_indices

    let enc1 = cb.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(psoSel)
    enc1.setBuffer(Q, offset: 0, index: 0); enc1.setBuffer(Kc, offset: 0, index: 1)
    enc1.setBuffer(Vc, offset: 0, index: 2); enc1.setBuffer(block_table, offset: 0, index: 3)
    enc1.setBuffer(m_partials, offset: 0, index: 4)
    enc1.setBuffer(l_partials, offset: 0, index: 5)
    enc1.setBuffer(O_partials, offset: 0, index: 6)
    enc1.setBuffer(fullOff, offset: 0, index: 7)
    enc1.setBuffer(fullIdx, offset: 0, index: 8)
    enc1.setBuffer(partOff, offset: 0, index: 9)
    enc1.setBuffer(partIdx, offset: 0, index: 10)
    enc1.setBuffer(kLenBuf, offset: 0, index: 11)
    var scale: Float = 1.0
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV), ns = UInt32(ATTN_N_SPLITS)
    // sliding_window bound at index 17 in both PSOs — the full-attn PSO's
    // FC_USE_SLIDE=false makes the kernel dead-strip all use of this value.
    // Keeping the binding uniform is a Swift-side convenience.
    var sw: UInt32 = isFull ? 0 : UInt32(SLIDING_WINDOW)
    var pp: UInt32 = 0, so: UInt32 = 0, tso: UInt32 = 0
    enc1.setBytes(&scale, length: 4, index: 12); enc1.setBytes(&mv, length: 4, index: 13)
    enc1.setBytes(&hq, length: 4, index: 14);    enc1.setBytes(&hkv, length: 4, index: 15)
    enc1.setBytes(&ns, length: 4, index: 16);    enc1.setBytes(&sw, length: 4, index: 17)
    enc1.setBytes(&pp, length: 4, index: 18);    enc1.setBytes(&so, length: 4, index: 19)
    enc1.setBytes(&tso, length: 4, index: 20)
    enc1.dispatchThreadgroups(MTLSize(width: activeB * H_KV, height: ATTN_N_SPLITS, depth: 1),
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
    enc2.dispatchThreadgroups(MTLSize(width: activeB * H_Q, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc2.endEncoding()
}




// activeB-aware softmax+topk: only [0, activeB) slot rows get routing.
// Silenced slots' router_lg is garbage (driven by garbage hidden) — this
// fix prevents downstream route_compact from seeing those bogus topk
// picks. Together with slot-aware route_compact this fully isolates the
// MoE pipeline from silenced-slot pollution.
func encSoftmaxTopk(_ cb: MTLCommandBuffer, expertScaleBuf: MTLBuffer, activeB: Int = B) {
    encSoftmaxTopkInto(cb, logits: router_lg, expertIds: expert_ids, gateW: gate_w,
                        expertScaleBuf: expertScaleBuf, numVecs: activeB)
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
// route_compact dispatch — `activeB` is passed to the kernel as the
// active-batch count. Silenced slots [activeB, B) are skipped, so MoE
// kernels see only real-slot routings in slot_token/group_start.
//
// V3: also writes a compact active_experts[] list (active expert IDs
// up front, sentinel E_EXP=128 in the tail). MoE dispatchers can then
// trim numActive to TOPK*activeB (or less) and rely on the sentinel
// for any padding entries.
func encRouteCompact(_ cb: MTLCommandBuffer, activeB: Int = B) {
    encRouteCompactInto(cb, expertIds: expert_ids, groupStart: group_start,
                         slotToken: slot_token, batchSlots: batch_slots,
                         activeExperts: active_exp, numVecs: activeB)
}

func encRouteCompactInto(_ cb: MTLCommandBuffer, expertIds: MTLBuffer,
                          groupStart: MTLBuffer, slotToken: MTLBuffer,
                          batchSlots: MTLBuffer, activeExperts: MTLBuffer,
                          numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(routeCompactPSO)
    enc.setBuffer(expertIds, offset: 0, index: 0)
    enc.setBuffer(groupStart, offset: 0, index: 1)
    enc.setBuffer(slotToken, offset: 0, index: 2)
    enc.setBuffer(batchSlots, offset: 0, index: 3)
    var bv = UInt32(numVecs), kv = UInt32(TOPK)
    enc.setBytes(&bv, length: 4, index: 4)
    enc.setBytes(&kv, length: 4, index: 5)
    enc.setBuffer(activeExperts, offset: 0, index: 6)
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
// flow where we need the MoE output separate for post_ffw_norm_2). The
// `activeB` cap restricts the per-batch combine to slots [0, activeB);
// silenced slots get no combine writes (their MoE route was zero anyway
// after slot-aware route_compact).
func encMoeCombineWrite(_ cb: MTLCommandBuffer, to outBuf: MTLBuffer, activeB: Int = B) {
    encMoeCombineWriteInto(cb, moeOut: moe_down_out, batchSlots: batch_slots, gateW: gate_w,
                            outBuf: outBuf, numVecs: activeB)
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

// Phase A control-vector injection: ONE cvector at ONE layer with a
// constant scalar magnitude, loaded from a raw fp16 binary of length
// HIDDEN (=2816) halves. Activated via env vars:
//   LM_CVEC_PATH  = path to fp16 binary
//   LM_CVEC_LAYER = layer index (0..NUM_LAYERS-1)
//   LM_CVEC_MAG   = scalar float magnitude (positive or negative)
// Validates the kernel cost is negligible before we invest in the
// ADSR evaluator + multi-vector / multi-layer plumbing of Phase B.
// Phase B: named cvector registry keyed by caller-assigned id string,
// populated via gemma_control_register_fp16 FFI. Per-session active-control
// lists reference entries here by id. Phase A's LM_CVEC_* env-gated path
// still works — the kernel dispatch site checks both sources.
var gCvecRegistry: [String: MTLBuffer] = [:]

// Residual capture for the pairwise prose cvec constructor. When
// gResidualCaptureLayer >= 0, buildStepCB blits `hidden` into
// gResidualCaptureBuf after that layer's post-FFN residual write.
// The most recent captured residual is the one from the most recent
// tick — caller is responsible for observing session state
// transitions (.priming → .generating) to read at the intended moment.
// Host-visible; caller reads directly via gemma_get_captured_residual.
var gResidualCaptureLayer: Int = -1
let gResidualCaptureBuf: MTLBuffer = device.makeBuffer(
    length: HIDDEN * 2, options: .storageModeShared)!

// All-layer capture: when enabled, every layer's post-FFN residual gets
// blitted into gAllLayerCaptureBuf. Layout:
//   [NUM_LAYERS, B, HIDDEN] fp16  — one strip per (layer, batch-slot) pair.
// A single blit per layer copies B * HIDDEN halves at once (the `hidden`
// buffer is already slot-major [B, HIDDEN]). Extracting one slot's
// per-layer residuals is a non-contiguous gather over L strides, done
// by gemma_get_all_slot_layer_residuals — cheap (30 * 5.6 KB memcpys).
// Legacy gemma_get_all_layer_residuals returns slot 0's strip, same
// semantics as the pre-batch layout.
var gCaptureAllLayers: Bool = false
let gAllLayerCaptureBuf: MTLBuffer = device.makeBuffer(
    length: NUM_LAYERS * B * HIDDEN * 2, options: .storageModeShared)!

// Per-layer Q/K/V capture for synthetic-KV fitting.
// When gCaptureQKV is true, after each layer's q_norm+RoPE, k_norm+RoPE,
// and v_norm_noscale, we blit the slot-0 q/k/v tensors into layer-
// indexed slots in these buffers. Slot 0 only — we use these captures
// from single-slot teacher-forcing during residual collection, not
// from the B=4 batched AR path. Layout:
//   Q: [NUM_LAYERS, MAX_Q_HEADS * MAX_HD] halves   (conservative sizing)
//   K: [NUM_LAYERS, MAX_KV_HEADS * MAX_HD] halves
//   V: [NUM_LAYERS, MAX_KV_HEADS * MAX_HD] halves
// MAX_HD covers both SLIDE (256) and FULL (512) layer types.
// Only the first (H_L * HD_L) halves of each per-layer slice are valid
// for that particular layer's shape; the client splits based on
// per-layer metadata.
let MAX_Q_HEADS: Int = max(SLIDE_H, FULL_H)       // 16
let MAX_KV_HEADS: Int = max(SLIDE_KV_H, FULL_KV_H) // 8
let MAX_HD: Int = max(SLIDE_HD, FULL_HD)           // 512
var gCaptureQKV: Bool = false
let gQCaptureBuf: MTLBuffer = device.makeBuffer(
    length: NUM_LAYERS * MAX_Q_HEADS * MAX_HD * 2, options: .storageModeShared)!
let gKCaptureBuf: MTLBuffer = device.makeBuffer(
    length: NUM_LAYERS * MAX_KV_HEADS * MAX_HD * 2, options: .storageModeShared)!
let gVCaptureBuf: MTLBuffer = device.makeBuffer(
    length: NUM_LAYERS * MAX_KV_HEADS * MAX_HD * 2, options: .storageModeShared)!

// Per-tick staging: for each slot, the list of (cvec buffer, layer, mag)
// triples to apply at their respective layers this step. Populated by
// step() from each session's activeControls + envelope evaluation, then
// read by buildStepCB inside the per-layer loop.
// For additive mode: `mag` = scalar the residual gets pushed BY.
// For project mode: `mag` = target projection value the residual's
// feature level gets coerced TO. Which mode is active is encoded in
// `mode`. measureOutSlot is the linear slot index in
// gProjectMeasureBuf where the kernel writes this control's
// pre-write projection (only populated when mode == .project); lets
// the pump read the natural feature level at each tick. ADSR
// envelope is applied to both modes identically (so you can e.g.
// ramp a target projection value over time, or fade out an
// obliteratus-style removal).
struct SlotControl {
    let buffer: MTLBuffer
    let layer: Int
    let mag: Float             // additive mag OR project target (mode-dependent)
    let mode: CvecMode
    let measureOutSlot: Int    // index into gProjectMeasureBuf
    // Transport mode only: Brenier-map scale + offset. Applied as
    // a' = scale * a + offset where a is the measured projection.
    let transportScale: Float
    let transportOffset: Float
}
var gSlotControls: [[SlotControl]] = Array(repeating: [], count: B)

// Scratch buffer for project-mode kernels' pre-write projection
// measurements. Each active project-mode control gets a unique slot
// index allocated during step()'s staging; the kernel writes that
// scalar into gProjectMeasureBuf[slot_idx]. The pump reads these
// post-CB to populate TokenSample.effectors[].projection or similar
// representation-engineering telemetry.
let MAX_PROJECT_CONTROLS_PER_SLOT = 8
let gProjectMeasureBuf: MTLBuffer = device.makeBuffer(
    length: B * MAX_PROJECT_CONTROLS_PER_SLOT * 4, options: .storageModeShared)!

// Per-prefill-tile staging: one entry per (layer, cvec) pair that should
// fire during the tile. `magsBuf` holds [B * qLen] floats, evaluated
// CPU-side from ActiveControl.magnitudeAt at each (slot, position) using
// pre_q_positions. mag==0 rows short-circuit inside the kernel, so silent
// slots and not-yet-active envelope positions pay near-zero cost.
// stepPrefillForSession rebuilds this list before each buildPrefillCB
// and clears it after the commit returns.
// Two flavors: additive uses magsBuf as the per-row scalar to add;
// project uses targetsBuf as the per-row TARGET projection (with
// Float.nan = skip-row sentinel). projectMeasuresBuf receives the
// pre-write projections per row when mode == .project (unused
// otherwise). Only one of magsBuf/targetsBuf is meaningful per entry
// based on mode, but both fields exist on the struct for simplicity.
struct PrefillControl {
    let buffer: MTLBuffer
    let layer: Int
    let mode: CvecMode
    let magsBuf: MTLBuffer          // additive-mode per-row scalars
    let targetsBuf: MTLBuffer       // project-mode per-row targets (NaN = skip)
    let projectMeasuresBuf: MTLBuffer  // project- & transport-mode pre-write readback
    // Transport-mode per-row buffers (NaN sentinel in scalesBuf skips row).
    let transportScalesBuf: MTLBuffer
    let transportOffsetsBuf: MTLBuffer
}
var gPrefillControls: [PrefillControl] = []

// Scratch pool of mag buffers reused across prefill tiles — one per
// potential (layer, control) pair in flight. Sized at B*MAX_Q_LEN floats
// each. Pool grows as needed; reused in FIFO order. Since prefill runs
// one tile at a time and commit-and-waits before returning, buffers are
// safe to reuse as soon as gPrefillControls.removeAll() fires.
var gPrefillMagBufPool: [MTLBuffer] = []
func acquirePrefillMagBuf(_ tileIdx: Int) -> MTLBuffer {
    while gPrefillMagBufPool.count <= tileIdx {
        gPrefillMagBufPool.append(device.makeBuffer(
            length: B * MAX_Q_LEN * 4, options: .storageModeShared)!)
    }
    return gPrefillMagBufPool[tileIdx]
}

// Phase C-Read staging. Detectors-to-dispatch this tick, per slot, as
// (buffer, layer, intensity-output-slot-index). intensity-slot-index is
// the linear offset into gIntensityBuf where the kernel writes this
// detector's scalar output.
struct SlotDetector { let buffer: MTLBuffer; let layer: Int; let outSlot: Int }
var gSlotDetectors: [[SlotDetector]] = Array(repeating: [], count: B)

// Host-visible scratch for per-tick intensity readback. Max 8 detectors
// per slot is plenty for Phase C-Read prototyping; layout is linear
// [slot][detector-index-within-slot] of float32.
let MAX_DETECTORS_PER_SLOT = 8
let gIntensityBuf: MTLBuffer = device.makeBuffer(
    length: B * MAX_DETECTORS_PER_SLOT * 4,
    options: .storageModeShared)!

let LM_CVEC_LAYER: Int = Int(ProcessInfo.processInfo.environment["LM_CVEC_LAYER"] ?? "") ?? -1
let LM_CVEC_MAG:   Float = Float(ProcessInfo.processInfo.environment["LM_CVEC_MAG"] ?? "") ?? 0.0
let LM_CVEC_BUF:   MTLBuffer? = {
    guard let path = ProcessInfo.processInfo.environment["LM_CVEC_PATH"],
          LM_CVEC_LAYER >= 0, !path.isEmpty else { return nil }
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
        print("[cvec] failed to read \(path)"); return nil
    }
    let expected = HIDDEN * 2
    guard data.count == expected else {
        print("[cvec] \(path) is \(data.count) bytes, expected \(expected) (HIDDEN=\(HIDDEN) × fp16)")
        return nil
    }
    let buf = device.makeBuffer(length: expected, options: .storageModeShared)!
    _ = data.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress, expected) }
    print("[cvec] loaded \(path), injecting at layer \(LM_CVEC_LAYER) with mag=\(LM_CVEC_MAG)")
    return buf
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

// AR step CB. `activeB` = the count of slots [0, activeB) carrying real
// session data this tick — kernel-zoo dispatchers use it to pick the
// compile-time-fixed B_TILE specialization that exactly matches the
// active workload (no silenced-slot wasted weight reads). Slot
// assignment policy ("lowest free first" in runAdmissionPass) keeps
// real sessions packed at [0, activeB).
//
// Default activeB=B preserves legacy behaviour for callers that haven't
// computed the actual count yet.
func buildStepCB(_ w: LmWeights, activeB: Int = B) -> MTLCommandBuffer {
    let cb = queue.makeCommandBuffer()!
    let aB = max(1, min(B, activeB))

    // Embed lookup + Gemma-4 sqrt(hidden) scale on token embeddings.
    encEmbed(cb, embedTable: w.embedTable)
    encScaleByScalar(cb, x: hidden, scalar: w.embedScaleBuf, N: HIDDEN, numVecs: aB)
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

        // Fused RMSNorm + Q/K/V projection — kernel-zoo dispatcher picks
        // the OTF b1 specialization for activeB=1 (22% faster than V6 at
        // numVecs=1) and falls back to V6 grid-shrink for activeB>1.
        // Gemma-4 full-attn layers omit the V projection in GGUF — pass
        // attnK as Wv so the kernel runs to completion; v_norm_noscale
        // follows.
        let Wv = lw.attnV ?? lw.attnK
        let WvFmt = (lw.attnV != nil) ? lw.attnVFormat : lw.attnKFormat
        encQKVDenseAR(cb, x: hidden, gammaBuf: lw.attnNorm, xNormBuf: hidden_norm,
                       Wq: lw.attnQ, qFmt: lw.attnQFormat,
                       Wk: lw.attnK, kFmt: lw.attnKFormat,
                       Wv: Wv, vFmt: WvFmt,
                       outQ: q_out, outK: k_out, outV: v_out,
                       Din: HIDDEN,
                       DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                       activeB: aB)

        encRMSNormG(cb, x: q_out, gammaBuf: lw.attnQNorm, out: q_out, D: HD, numVecs: aB * H)
        encRope(cb, q_out, H: H, D: HD, rotary: rotary, theta: theta, activeB: aB)
        encRMSNormG(cb, x: k_out, gammaBuf: lw.attnKNorm, out: k_out, D: HD, numVecs: aB * KV_H)
        encRope(cb, k_out, H: KV_H, D: HD, rotary: rotary, theta: theta, activeB: aB)
        encRMSNormNoScale(cb, x: v_out, out: v_out, D: HD, numVecs: aB * KV_H)

        // Q/K/V capture: slot-0 per-layer snapshot after q_norm/RoPE,
        // k_norm/RoPE, v_norm_noscale — i.e. exactly the tensors that
        // would enter attention's dot products and be written to the
        // KV cache. Used by the synthetic-KV fitting pipeline offline;
        // no cost when gCaptureQKV is false.
        if gCaptureQKV {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: q_out, sourceOffset: 0,
                      to: gQCaptureBuf, destinationOffset: L * MAX_Q_HEADS * MAX_HD * 2,
                      size: H * HD * 2)
            blit.copy(from: k_out, sourceOffset: 0,
                      to: gKCaptureBuf, destinationOffset: L * MAX_KV_HEADS * MAX_HD * 2,
                      size: KV_H * HD * 2)
            blit.copy(from: v_out, sourceOffset: 0,
                      to: gVCaptureBuf, destinationOffset: L * MAX_KV_HEADS * MAX_HD * 2,
                      size: KV_H * HD * 2)
            blit.endEncoding()
        }

        let pg = isFull ? PAGE_FULL : PAGE_SLIDE
        encKVWrite(cb, K: k_out, V: v_out, Kc: Kc, Vc: Vc, H: KV_H, D: HD, page: pg, activeB: aB)

        let npBuf = isFull ? num_pages_full : num_pages_slide
        let klBuf = isFull ? k_len_full : k_len_slide
        encAttn(cb, Q: q_out, O: attn_out, Kc: Kc, Vc: Vc,
                kLenBuf: klBuf, H_Q: H, H_KV: KV_H, D: HD,
                isFull: isFull, activeB: aB)
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
        // o_proj — format-aware AR dispatch (Q8_0 → btile zoo, others → simple GEMV).
        encDenseGemvAR(cb, attn_out, lw.attnOut, format: lw.attnOutFormat, mlp_out,
                        Din: H * HD, Dout: HIDDEN, activeB: aB)
        // Heretic-style attn-out ablation (niche, leave at B).
        if let engine = gEngine {
            for wa in engine.writeAblations where wa.layer == L && wa.component == .attnOut {
                encOrthogonalizeWrite(cb, y: mlp_out, rHat: wa.rHatBuf,
                                       alpha: wa.alpha, N: HIDDEN, numVecs: B)
            }
        }
        encRmsNormAdd(cb, x: mlp_out, gammaBuf: lw.postAttnNorm,
                      residual: hidden, out: hidden, N: HIDDEN, numVecs: aB)
        if L == 0, let dump = LM_DUMP_L0_STAGING {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: hidden, sourceOffset: 0,
                      to: dump, destinationOffset: 0, size: HIDDEN * 2)
            blit.endEncoding()
        }

        // Shared MLP branch: rmsnorm + gate + up + gelu*mul → ffn_down → post_ffn_1.
        // Format-aware: Q8_0 takes the fused-RMSNorm fast path, others fall
        // back to explicit RMSNorm + 2 plain GEMVs + split-halves gelu_mul.
        encGateUpAR(cb, x: hidden, gammaBuf: lw.ffnNorm, xNormBuf: hidden_norm,
                     Wg: lw.ffnGate, gateFmt: lw.ffnGateFormat,
                     Wu: lw.ffnUp,   upFmt:   lw.ffnUpFormat,
                     gateOut: shrd_gate, fusedScratch: shrd_gate_up_fused,
                     Din: HIDDEN, Dout: SHARED_INT, activeB: aB)
        encDenseGemvAR(cb, shrd_gate, lw.ffnDown, format: lw.ffnDownFormat, mlp_out,
                        Din: SHARED_INT, Dout: HIDDEN, activeB: aB)
        encRMSNormG(cb, x: mlp_out, gammaBuf: lw.postFfn1Norm, out: mlp_out, D: HIDDEN, numVecs: aB)
        if L == 0, let dump = LM_DUMP_L0_STAGING {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: mlp_out, sourceOffset: 0,
                      to: dump, destinationOffset: 1 * HIDDEN * 2, size: HIDDEN * 2)
            blit.endEncoding()
        }

        // Router: pre-norm with per-dim scale and 1/sqrt(D) divisor, project to
        // logits, softmax+topk+renorm*per_expert_scale, then compact for MoE.
        encRouterPreNorm(cb, x: hidden, per_dim_scale: lw.routerScale, out: hidden_norm,
                          numVecs: aB)
        if L == 0, let routerDump = LM_DUMP_L0_ROUTER {
            // hidden_norm (slot 0) f16[HIDDEN] — capture BEFORE the router proj.
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: hidden_norm, sourceOffset: 0,
                      to: routerDump, destinationOffset: TOPK * 4 + TOPK * 4 + E_EXP * 2, size: HIDDEN * 2)
            blit.endEncoding()
        }
        encGemvV5(cb, hidden_norm, lw.routerW, router_lg, Din: HIDDEN, Dout: E_EXP, numVecs: aB)
        if L == 0, let routerDump = LM_DUMP_L0_ROUTER {
            // router_lg (slot 0) f16[E_EXP] — captured AFTER proj, BEFORE softmax.
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: router_lg, sourceOffset: 0,
                      to: routerDump, destinationOffset: TOPK * 4 + TOPK * 4, size: E_EXP * 2)
            blit.endEncoding()
        }
        encSoftmaxTopk(cb, expertScaleBuf: lw.expertScale, activeB: aB)
        if L == 0, let routerDump = LM_DUMP_L0_ROUTER {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: expert_ids, sourceOffset: 0,
                      to: routerDump, destinationOffset: 0, size: TOPK * 4)
            blit.copy(from: gate_w, sourceOffset: 0,
                      to: routerDump, destinationOffset: TOPK * 4, size: TOPK * 4)
            blit.endEncoding()
        }
        encRouteCompact(cb, activeB: aB)

        // MoE branch: pre_ffn_2(hidden) → fused Q4_K gate_up → gelu*mul → Q5_1 down → combine → post_ffn_2.
        encRMSNormG(cb, x: hidden, gammaBuf: lw.preFfn2Norm, out: hidden_norm, D: HIDDEN, numVecs: aB)
        if L == 0, let dump = LM_DUMP_L0_STAGING {
            // Slot 3: pre_feedforward_layernorm_2(hidden) = input to experts.
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: hidden_norm, sourceOffset: 0,
                      to: dump, destinationOffset: 3 * HIDDEN * 2, size: HIDDEN * 2)
            blit.endEncoding()
        }
        // numActive trimmed via the compact active_experts list that
        // route_compact (V3) writes. Layout:
        //   active_experts[0 .. num_unique_active) — actual expert IDs
        //   active_experts[num_unique_active .. E)  — sentinel E (=128)
        // MoE V6 kernels check `if (expert >= 128) return;` so any TGs
        // we launch beyond the actual unique-active count just sentinel-
        // bail; the saving is the (E_EXP - TOPK*aB) wasted launches we
        // never fire in the first place.
        let aMoeNumActive = min(TOPK * aB, E_EXP)
        encMoeUpGemvAR(cb, hidden_norm, lw.moeGateUp, format: lw.moeGateUpFormat, gate_up_fused,
                        Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: aMoeNumActive, activeB: aB)
        encMoeGeluMulFused(cb, fused: gate_up_fused, out: gate_proj,
                            N_half: MOE_INT, numSlots: TOPK * aB)
        encMoeDownGemvAR(cb, gate_proj, lw.moeDown, format: lw.moeDownFormat, moe_down_out,
                          Din: MOE_INT, Dout: HIDDEN, numActive: aMoeNumActive, activeB: aB)
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
        encMoeCombineWrite(cb, to: moe_sum, activeB: aB)
        if L == 0, let dump = LM_DUMP_L0_STAGING {
            // Slot 4: moe_sum BEFORE post_feedforward_layernorm_2 (raw scatter-sum experts output).
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: moe_sum, sourceOffset: 0,
                      to: dump, destinationOffset: 4 * HIDDEN * 2, size: HIDDEN * 2)
            blit.endEncoding()
        }
        encRMSNormG(cb, x: moe_sum, gammaBuf: lw.postFfn2Norm, out: moe_sum, D: HIDDEN, numVecs: aB)
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
        encBufferCopy(cb, src: mlp_out, dst: ffn_combined, bytes: aB * HIDDEN * 2)
        encAddInplace(cb, dst: ffn_combined, src: moe_sum, N: HIDDEN, numVecs: aB)
        // Heretic-style ffn-out ablation (niche, leave at B).
        if let engine = gEngine {
            for wa in engine.writeAblations where wa.layer == L && wa.component == .ffnOut {
                encOrthogonalizeWrite(cb, y: ffn_combined, rHat: wa.rHatBuf,
                                       alpha: wa.alpha, N: HIDDEN, numVecs: B)
            }
        }
        encRmsNormAddScale(cb, x: ffn_combined, gammaBuf: lw.postFfnNorm,
                           residual: hidden, scalar: lw.layerOutputScale,
                           out: hidden, N: HIDDEN, numVecs: aB)
        // Control-vector injection at the post-FFN residual site.
        //
        // Phase A: LM_CVEC_* env vars apply ONE cvec at ONE layer to ALL
        // slots at constant magnitude (still supported for microbench
        // validation without touching the FFI).
        if let cvec = LM_CVEC_BUF, L == LM_CVEC_LAYER, LM_CVEC_MAG != 0.0 {
            encAddScaledCvec(cb, dst: hidden, cvec: cvec, mag: LM_CVEC_MAG,
                              N: HIDDEN, numVecs: B)
        }
        // Phase B: per-slot active controls, evaluated by step() each tick
        // and staged in gSlotControls. For additive mode, sc.mag is the
        // scalar to push the residual BY; for project mode, sc.mag is the
        // TARGET projection value to coerce the residual TO. Project-mode
        // dispatches ALSO write the pre-write projection into
        // gProjectMeasureBuf[measureOutSlot] so the pump can read back
        // the natural feature level at this tick (representation-
        // engineering measurement primitive).
        for slot in 0..<B {
            for sc in gSlotControls[slot] where sc.layer == L {
                switch sc.mode {
                case .additive:
                    if sc.mag != 0.0 {
                        encAddScaledCvecSlot(cb, dst: hidden, slot: slot,
                                              cvec: sc.buffer, mag: sc.mag, N: HIDDEN)
                    }
                case .project:
                    // Always dispatch for project mode — the target may
                    // legitimately be 0 (= remove the feature entirely).
                    // Only way to skip would be an explicit "disabled"
                    // flag, which we don't have; envelope evaluation
                    // handles fade-in/out timing separately.
                    encProjectCvecSlot(cb, dst: hidden, slot: slot,
                                        cvec: sc.buffer,
                                        currentProjBuf: gProjectMeasureBuf,
                                        currentProjSlotOffsetBytes: sc.measureOutSlot * 4,
                                        target: sc.mag, N: HIDDEN)
                case .transport:
                    // Gaussian OT: same reduction + write shape as
                    // project, but target is a linear function of the
                    // measured projection (scale*a + offset) rather
                    // than a constant. Preserves within-class variation.
                    encTransportCvecSlot(cb, dst: hidden, slot: slot,
                                          cvec: sc.buffer,
                                          currentProjBuf: gProjectMeasureBuf,
                                          currentProjSlotOffsetBytes: sc.measureOutSlot * 4,
                                          scale: sc.transportScale,
                                          offset: sc.transportOffset,
                                          N: HIDDEN)
                }
            }
        }
        // Phase C-Read: after writes land, measure the residual against
        // each active detector whose layer matches this one. Each
        // dispatch writes one scalar into gIntensityBuf at its assigned
        // slot — pump reads them back after CB completion.
        for slot in 0..<B {
            for sd in gSlotDetectors[slot] where sd.layer == L {
                encMeasureDotSlot(cb, src: hidden, slot: slot,
                                   meas: sd.buffer,
                                   intensities: gIntensityBuf,
                                   intensitySlotOffsetBytes: sd.outSlot * 4,
                                   N: HIDDEN)
            }
        }
        // Residual capture for the pairwise prose constructor. Captures
        // slot 0 only — the constructor always runs one-off sessions so
        // slot 0 is the only real one.
        if gResidualCaptureLayer == L {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: hidden, sourceOffset: 0,
                      to: gResidualCaptureBuf, destinationOffset: 0,
                      size: HIDDEN * 2)
            blit.endEncoding()
            if ProcessInfo.processInfo.environment["LM_CVEC_LOG"] != nil {
                print("[capture] blit fired in buildStepCB @ layer \(L)")
            }
        }
        // All-layer capture: one blit per layer, copies the FULL B-slot
        // strip of `hidden` (layout [B, HIDDEN] fp16) into the layer's
        // B-wide strip in gAllLayerCaptureBuf at offset L*B*HIDDEN*2.
        // Per-slot extraction happens on readback via
        // gemma_get_all_slot_layer_residuals; legacy callers using
        // gemma_get_all_layer_residuals still see slot 0's strip.
        if gCaptureAllLayers {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: hidden, sourceOffset: 0,
                      to: gAllLayerCaptureBuf,
                      destinationOffset: L * B * HIDDEN * 2,
                      size: B * HIDDEN * 2)
            blit.endEncoding()
        }
        if let dump = LM_DUMP_STAGING {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: hidden, sourceOffset: 0,
                      to: dump, destinationOffset: (L + 1) * HIDDEN * 2, size: HIDDEN * 2)
            blit.endEncoding()
        }
    }

    // Final RMSNorm + fused fp16 unembed + final-logit softcap (cap=30.0).
    // numVecs/activeB shrink to aB so only active slots get unembed work
    // (each is a 1.4 GB weight read, so this cuts ~7 of 8 wasted reads
    // when activeB=1). Inactive slot logits are stale but the engine
    // only reads gpu_sampled_tokens[active_slot] anyway.
    encRMSNormG(cb, x: hidden, gammaBuf: w.outputNorm, out: hidden_norm, D: HIDDEN, numVecs: aB)
    encGemvV4Softcap(cb, hidden_norm, w.unembedW, logits,
                     Din: HIDDEN, Dout: VOCAB, cap: 30.0, activeB: aB)

    // Phase 2 of the dataflow-pipeline spec: GPU-side sampling. Reads
    // logits + per-slot sampling params (populated by step() before
    // commit), writes chosen token ids into `gpu_sampled_tokens`.
    // Post-wait, step() reads gpu_sampled_tokens and compares vs the
    // CPU sampler (validation phase) or uses it directly.
    encSampleToken(cb,
                    logits: logits,
                    samplingLogitBias: sampling_logit_bias,
                    samplingTemp: sampling_temperature,
                    samplingMinP: sampling_min_p,
                    samplingSeed: sampling_seed,
                    samplingStep: sampling_step,
                    samplingActive: sampling_active,
                    inputTokens: gpu_sampled_tokens,
                    vocab: VOCAB)
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
                            skipUnembed: Bool = false,
                            fullPrefillLogits: Bool = false) {
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

        // RMSNorm + Q/K/V projection over all N tokens. Unfused (norm into
        // pre_hidden_norm, then 3 separate matmuls) so the projections can
        // use the simdgroup-matmul kernel — at MAX_Q_LEN=256 this beats
        // the fused-RMSNorm GEMV path by ~5×.
        let Wv = lw.attnV ?? lw.attnK
        let WvFmt = (lw.attnV != nil) ? lw.attnVFormat : lw.attnKFormat
        encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.attnNorm, out: pre_hidden_norm,
                     D: HIDDEN, numVecs: N)
        encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.attnQ, format: lw.attnQFormat, Y: q_out,
                           Din: HIDDEN, Dout: H * HD, numVecs: N)
        encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.attnK, format: lw.attnKFormat, Y: k_out,
                           Din: HIDDEN, Dout: KV_H * HD, numVecs: N)
        encDenseMmPrefill(cb, x: pre_hidden_norm, W: Wv, format: WvFmt, Y: v_out,
                           Din: HIDDEN, Dout: KV_H * HD, numVecs: N)

        // Per-head norms (numVecs = N * head-count).
        encRMSNormG(cb, x: q_out, gammaBuf: lw.attnQNorm, out: q_out, D: HD, numVecs: N * H)
        encRopeMulti(cb, q_out, q_positions: pre_q_positions, H: H, D: HD,
                     rotary: rotary, theta: theta, qLen: qLen)
        encRMSNormG(cb, x: k_out, gammaBuf: lw.attnKNorm, out: k_out, D: HD, numVecs: N * KV_H)
        encRopeMulti(cb, k_out, q_positions: pre_q_positions, H: KV_H, D: HD,
                     rotary: rotary, theta: theta, qLen: qLen)
        encRMSNormNoScale(cb, x: v_out, out: v_out, D: HD, numVecs: N * KV_H)

        // Q/K/V capture during prefill. Mirror of the AR site: blits
        // slot-0 LAST-position q/k/v into the capture buffers at
        // layer-indexed offsets. q/k/v are laid out as [B*qLen, H*HD]
        // halves, so slot 0 position qLen-1 starts at offset
        // (qLen - 1) * H * HD halves.
        if gCaptureQKV {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: q_out,
                      sourceOffset: (qLen - 1) * H * HD * 2,
                      to: gQCaptureBuf,
                      destinationOffset: L * MAX_Q_HEADS * MAX_HD * 2,
                      size: H * HD * 2)
            blit.copy(from: k_out,
                      sourceOffset: (qLen - 1) * KV_H * HD * 2,
                      to: gKCaptureBuf,
                      destinationOffset: L * MAX_KV_HEADS * MAX_HD * 2,
                      size: KV_H * HD * 2)
            blit.copy(from: v_out,
                      sourceOffset: (qLen - 1) * KV_H * HD * 2,
                      to: gVCaptureBuf,
                      destinationOffset: L * MAX_KV_HEADS * MAX_HD * 2,
                      size: KV_H * HD * 2)
            blit.endEncoding()
        }

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

        // o_proj (simdgroup matmul) → pre_mlp_out; fused post-attn norm + residual add.
        encDenseMmPrefill(cb, x: pre_attn_out, W: lw.attnOut, format: lw.attnOutFormat, Y: pre_mlp_out,
                           Din: H * HD, Dout: HIDDEN, numVecs: N)
        encRmsNormAdd(cb, x: pre_mlp_out, gammaBuf: lw.postAttnNorm,
                      residual: pre_hidden, out: pre_hidden, N: HIDDEN, numVecs: N)

        // Shared MLP. RMSNorm into pre_hidden_norm, then two separate
        // simdgroup matmuls (gate → pre_shrd_gate, up → pre_shrd_gate_up_fused
        // first half), gelu_mul_inplace combines them in-place into
        // pre_shrd_gate, ffn_down matmul produces pre_mlp_out, post-norm.
        encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.ffnNorm, out: pre_hidden_norm,
                     D: HIDDEN, numVecs: N)
        encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.ffnGate, format: lw.ffnGateFormat, Y: pre_shrd_gate,
                           Din: HIDDEN, Dout: SHARED_INT, numVecs: N)
        encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.ffnUp, format: lw.ffnUpFormat, Y: pre_shrd_gate_up_fused,
                           Din: HIDDEN, Dout: SHARED_INT, numVecs: N)
        encGeluMulInplace(cb, gate: pre_shrd_gate, up: pre_shrd_gate_up_fused,
                           N_half: SHARED_INT, numSlots: N)
        encDenseMmPrefill(cb, x: pre_shrd_gate, W: lw.ffnDown, format: lw.ffnDownFormat, Y: pre_mlp_out,
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
                             activeExperts: active_exp,
                             numVecs: N)

        // MoE: pre_ffn_2(pre_hidden) → Q4_K gate_up (simdgroup matmul) →
        // gelu*up → Q5_1 down (simdgroup matmul) → combine → post_ffn_2.
        // Both matmuls write slot-flat outputs matching moe_combine_write's
        // expected layout. Q4_K reads X via slot_token (broadcast per token),
        // Q5_1 reads X per-slot (gelu output is already slot-flat).
        encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.preFfn2Norm, out: pre_hidden_norm,
                    D: HIDDEN, numVecs: N)
        encMoeUpMmPrefill(cb, x: pre_hidden_norm, W: lw.moeGateUp, format: lw.moeGateUpFormat,
                           Y: pre_gate_up_fused,
                           slotTokenBuf: pre_slot_token, activeExpBuf: active_exp,
                           groupStartBuf: pre_group_start,
                           Din: HIDDEN, Dout: MOE_FUSED_DOUT, numSlots: NS, E: E_EXP)
        encMoeGeluMulFused(cb, fused: pre_gate_up_fused, out: pre_gate_proj,
                            N_half: MOE_INT, numSlots: NS)
        encMoeDownMmPrefill(cb, x: pre_gate_proj, W: lw.moeDown, format: lw.moeDownFormat,
                             Y: pre_moe_down_out,
                             activeExpBuf: active_exp, groupStartBuf: pre_group_start,
                             Din: MOE_INT, Dout: HIDDEN, numSlots: NS, E: E_EXP)
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
        // Prefill-time control-vector injection. Same post-FFN residual
        // site as buildStepCB so a span that gets split into prefill-then-
        // prefill-resume (or prefill-then-AR) sees identical steering
        // semantics to one that ran fully through AR. Each PrefillControl
        // carries a per-row mags buffer (zero rows are no-ops), populated
        // CPU-side before encode from ActiveControl.magnitudeAt(position,
        // turn) over pre_q_positions.
        for pc in gPrefillControls where pc.layer == L {
            switch pc.mode {
            case .additive:
                encAddScaledCvecPrefill(cb, dst: pre_hidden, cvec: pc.buffer,
                                         magsBuf: pc.magsBuf, N: HIDDEN, numVecs: N)
            case .project:
                encProjectCvecPrefill(cb, dst: pre_hidden, cvec: pc.buffer,
                                       targetsBuf: pc.targetsBuf,
                                       currentProjBuf: pc.projectMeasuresBuf,
                                       N: HIDDEN, numVecs: N)
            case .transport:
                encTransportCvecPrefill(cb, dst: pre_hidden, cvec: pc.buffer,
                                         scalesBuf: pc.transportScalesBuf,
                                         offsetsBuf: pc.transportOffsetsBuf,
                                         currentProjBuf: pc.projectMeasuresBuf,
                                         N: HIDDEN, numVecs: N)
            }
        }
        // Residual capture for the pairwise prose constructor — mirror
        // of the buildStepCB hook. Captures slot 0's LAST-position
        // residual (position qLen-1) at the designated layer. This is
        // the representation the model would use to predict the next
        // token, the conventional snapshot for representation engineering.
        if gResidualCaptureLayer == L {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: pre_hidden,
                      sourceOffset: (qLen - 1) * HIDDEN * 2,
                      to: gResidualCaptureBuf, destinationOffset: 0,
                      size: HIDDEN * 2)
            blit.endEncoding()
        }
        // All-layer capture during prefill: one blit per layer, last-
        // position slot only (prefill is single-slot in this path).
        // Writes into layer L's slot-0 strip of the B-wide capture
        // buffer, leaving the other slots whatever they had from the
        // prior AR tick (batched callers should drain+capture via
        // buildStepCB's blit, not the prefill path).
        if gCaptureAllLayers {
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: pre_hidden,
                      sourceOffset: (qLen - 1) * HIDDEN * 2,
                      to: gAllLayerCaptureBuf,
                      destinationOffset: L * B * HIDDEN * 2,
                      size: HIDDEN * 2)
            blit.endEncoding()
        }
    }

    // Final output norm + unembed + softcap.
    //
    // Default fast path (fullPrefillLogits=false): only the B last-q rows
    // feed sampling, so we gather those rows from pre_hidden, RMSNorm at
    // numVecs=B, and run V4Softcap (fused matmul+softcap) into the
    // AR-shaped logits[B, VOCAB] directly. Collapses unembed work 32× at
    // qLen=32 (~150 ms → ~5 ms).
    //
    // Slow path (fullPrefillLogits=true): legacy full [B*qLen, VOCAB]
    // unembed; required by runLmPrefillValidate which reads pre_logits
    // for per-position KL math.
    if !skipUnembed {
        if fullPrefillLogits {
            encRMSNormG(cb, x: pre_hidden, gammaBuf: w.outputNorm, out: pre_hidden_norm,
                        D: HIDDEN, numVecs: N)
            encGemvV5(cb, pre_hidden_norm, w.unembedW, pre_logits,
                      Din: HIDDEN, Dout: VOCAB, numVecs: N)
            encSoftcapInto(cb, buf: pre_logits, N: VOCAB, numVecs: N, cap: 30.0)
            let blit = cb.makeBlitCommandEncoder()!
            for slot in 0..<B {
                let srcOff = (slot * qLen + (qLen - 1)) * VOCAB * 2
                let dstOff = slot * VOCAB * 2
                blit.copy(from: pre_logits, sourceOffset: srcOff,
                          to: logits, destinationOffset: dstOff,
                          size: VOCAB * 2)
            }
            blit.endEncoding()
        } else {
            let gatherBlit = cb.makeBlitCommandEncoder()!
            for slot in 0..<B {
                let srcOff = (slot * qLen + (qLen - 1)) * HIDDEN * 2
                let dstOff = slot * HIDDEN * 2
                gatherBlit.copy(from: pre_hidden, sourceOffset: srcOff,
                                 to: hidden, destinationOffset: dstOff,
                                 size: HIDDEN * 2)
            }
            gatherBlit.endEncoding()
            encRMSNormG(cb, x: hidden, gammaBuf: w.outputNorm, out: hidden_norm,
                        D: HIDDEN, numVecs: B)
            encGemvV4Softcap(cb, hidden_norm, w.unembedW, logits,
                              Din: HIDDEN, Dout: VOCAB, cap: 30.0)
        }

        // GPU-side sampling at the end of prefill — same kernel, same
        // buffers, same contract as the AR step. Sampling params are
        // populated by the caller (stepPrefillForSession /
        // stepMultiSlotPrefill / stepMultiSlotSoftPrefill) before commit.
        encSampleToken(cb,
                        logits: logits,
                        samplingLogitBias: sampling_logit_bias,
                        samplingTemp: sampling_temperature,
                        samplingMinP: sampling_min_p,
                        samplingSeed: sampling_seed,
                        samplingStep: sampling_step,
                        samplingActive: sampling_active,
                        inputTokens: gpu_sampled_tokens,
                        vocab: VOCAB)
    }
}

// Back-compat wrapper for callers (LmSession, runLmPrefillValidate) that
// want a single-tile CB handed to them ready to commit.
//
// skipUnembed: defaults to false to preserve existing callers (validation
// harness needs pre_logits populated; LmSession's last prefill tick needs
// gpu_sampled_tokens). Multi-tile callers should pass skipUnembed=true on
// non-final ticks — the unembed at Dout=262144 is ~150 ms per CB and is
// pure waste when the logits won't be sampled.
func buildPrefillCB(_ w: LmWeights, qLen: Int, skipEmbed: Bool = false,
                     skipUnembed: Bool = false,
                     fullPrefillLogits: Bool = false) -> MTLCommandBuffer {
    let cb = queue.makeCommandBuffer()!
    encodePrefillTileInto(cb, w, qLen: qLen, skipEmbed: skipEmbed,
                            skipUnembed: skipUnembed,
                            fullPrefillLogits: fullPrefillLogits)
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
    let WvFmt = (lw.attnV != nil) ? lw.attnVFormat : lw.attnKFormat

    func commitAndCheck(_ tag: String, buf: MTLBuffer, count: Int, _ block: (MTLCommandBuffer) -> Void) {
        let cb = queue.makeCommandBuffer()!
        block(cb); cb.commit(); cb.waitUntilCompleted()
        checkNanOne("L\(L) \(tag)", buf, count: count, sslot: sslot, qLen: qLen)
    }

    commitAndCheck("qkv_proj", buf: q_out, count: N * H * HD) { cb in
        encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.attnNorm, out: pre_hidden_norm,
                     D: HIDDEN, numVecs: N)
        encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.attnQ, format: lw.attnQFormat, Y: q_out,
                           Din: HIDDEN, Dout: H * HD, numVecs: N)
        encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.attnK, format: lw.attnKFormat, Y: k_out,
                           Din: HIDDEN, Dout: KV_H * HD, numVecs: N)
        encDenseMmPrefill(cb, x: pre_hidden_norm, W: Wv, format: WvFmt, Y: v_out,
                           Din: HIDDEN, Dout: KV_H * HD, numVecs: N)
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
        encDenseMmPrefill(cb, x: pre_attn_out, W: lw.attnOut, format: lw.attnOutFormat,
                           Y: pre_mlp_out, Din: H * HD, Dout: HIDDEN, numVecs: N)
        encRmsNormAdd(cb, x: pre_mlp_out, gammaBuf: lw.postAttnNorm,
                      residual: pre_hidden, out: pre_hidden, N: HIDDEN, numVecs: N)
    }
    // Remaining (shared MLP + MoE) — as a single commit, probably not needed
    // past the first NaN source.
    commitAndCheck("ffn+moe+resid2", buf: pre_hidden, count: N * HIDDEN) { cb in
        encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.ffnNorm, out: pre_hidden_norm,
                     D: HIDDEN, numVecs: N)
        encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.ffnGate, format: lw.ffnGateFormat,
                           Y: pre_shrd_gate, Din: HIDDEN, Dout: SHARED_INT, numVecs: N)
        encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.ffnUp, format: lw.ffnUpFormat,
                           Y: pre_shrd_gate_up_fused, Din: HIDDEN, Dout: SHARED_INT, numVecs: N)
        encGeluMulInplace(cb, gate: pre_shrd_gate, up: pre_shrd_gate_up_fused,
                           N_half: SHARED_INT, numSlots: N)
        encDenseMmPrefill(cb, x: pre_shrd_gate, W: lw.ffnDown, format: lw.ffnDownFormat,
                           Y: pre_mlp_out, Din: SHARED_INT, Dout: HIDDEN, numVecs: N)
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
                             activeExperts: active_exp,
                             numVecs: N)
        encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.preFfn2Norm, out: pre_hidden_norm,
                    D: HIDDEN, numVecs: N)
        encMoeUpMmPrefill(cb, x: pre_hidden_norm, W: lw.moeGateUp, format: lw.moeGateUpFormat,
                           Y: pre_gate_up_fused,
                           slotTokenBuf: pre_slot_token, activeExpBuf: active_exp,
                           groupStartBuf: pre_group_start,
                           Din: HIDDEN, Dout: MOE_FUSED_DOUT, numSlots: NS, E: E_EXP)
        encMoeGeluMulFused(cb, fused: pre_gate_up_fused, out: pre_gate_proj,
                            N_half: MOE_INT, numSlots: NS)
        encMoeDownMmPrefill(cb, x: pre_gate_proj, W: lw.moeDown, format: lw.moeDownFormat,
                             Y: pre_moe_down_out,
                             activeExpBuf: active_exp, groupStartBuf: pre_group_start,
                             Din: MOE_INT, Dout: HIDDEN, numSlots: NS, E: E_EXP)
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
    // Diagnostic prefill path — type-complete via the same format-aware
    // dispatchers used by buildPrefillCB. RMSNorm is unfused (the simdgroup
    // matmul path doesn't fuse RMSNorm in any format).
    let Wv = lw.attnV ?? lw.attnK
    let WvFmt = (lw.attnV != nil) ? lw.attnVFormat : lw.attnKFormat
    encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.attnNorm, out: pre_hidden_norm,
                 D: HIDDEN, numVecs: N)
    encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.attnQ, format: lw.attnQFormat, Y: q_out,
                       Din: HIDDEN, Dout: H * HD, numVecs: N)
    encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.attnK, format: lw.attnKFormat, Y: k_out,
                       Din: HIDDEN, Dout: KV_H * HD, numVecs: N)
    encDenseMmPrefill(cb, x: pre_hidden_norm, W: Wv, format: WvFmt, Y: v_out,
                       Din: HIDDEN, Dout: KV_H * HD, numVecs: N)
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
    encDenseMmPrefill(cb, x: pre_attn_out, W: lw.attnOut, format: lw.attnOutFormat, Y: pre_mlp_out,
                       Din: H * HD, Dout: HIDDEN, numVecs: N)
    encRmsNormAdd(cb, x: pre_mlp_out, gammaBuf: lw.postAttnNorm,
                  residual: pre_hidden, out: pre_hidden, N: HIDDEN, numVecs: N)
    // Shared FFN: RMSNorm + gate + up + gelu_mul (unfused) + ffn_down.
    encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.ffnNorm, out: pre_hidden_norm,
                 D: HIDDEN, numVecs: N)
    encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.ffnGate, format: lw.ffnGateFormat,
                       Y: pre_shrd_gate, Din: HIDDEN, Dout: SHARED_INT, numVecs: N)
    encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.ffnUp, format: lw.ffnUpFormat,
                       Y: pre_shrd_gate_up_fused, Din: HIDDEN, Dout: SHARED_INT, numVecs: N)
    encGeluMulInplace(cb, gate: pre_shrd_gate, up: pre_shrd_gate_up_fused,
                       N_half: SHARED_INT, numSlots: N)
    encDenseMmPrefill(cb, x: pre_shrd_gate, W: lw.ffnDown, format: lw.ffnDownFormat,
                       Y: pre_mlp_out, Din: SHARED_INT, Dout: HIDDEN, numVecs: N)
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
                         activeExperts: active_exp,
                         numVecs: N)
    encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.preFfn2Norm, out: pre_hidden_norm,
                D: HIDDEN, numVecs: N)
    encMoeUpMmPrefill(cb, x: pre_hidden_norm, W: lw.moeGateUp, format: lw.moeGateUpFormat,
                       Y: pre_gate_up_fused,
                       slotTokenBuf: pre_slot_token, activeExpBuf: active_exp,
                       groupStartBuf: pre_group_start,
                       Din: HIDDEN, Dout: MOE_FUSED_DOUT, numSlots: NS, E: E_EXP)
    encMoeGeluMulFused(cb, fused: pre_gate_up_fused, out: pre_gate_proj,
                        N_half: MOE_INT, numSlots: NS)
    encMoeDownMmPrefill(cb, x: pre_gate_proj, W: lw.moeDown, format: lw.moeDownFormat,
                         Y: pre_moe_down_out,
                         activeExpBuf: active_exp, groupStartBuf: pre_group_start,
                         Din: MOE_INT, Dout: HIDDEN, numSlots: NS, E: E_EXP)
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


struct LayerW {
    let attnQ, attnK, attnOut: MTLBuffer              // dense, format may be Q8_0/Q5_K/Q6_K
    let attnV: MTLBuffer?                              // nil on full-attn layers:
                                                        // Gemma-4 drops the V projection and uses
                                                        // K as V at those layers (see llama.cpp
                                                        // gemma4-iswa.cpp:83-85)
    let ffnGate, ffnUp, ffnDown: MTLBuffer            // dense, format may be Q8_0/Q5_K/Q6_K
    let moeGateUp: MTLBuffer                           // MoE up, format Q4_K or Q5_K
    let moeDown: MTLBuffer                             // MoE down, format Q5_1/Q6_K/Q8_0
    // Per-tensor formats — populated from GGUF's actual dtype on load.
    // The format-aware dispatchers (encDenseMmPrefill / encMoeUpMmPrefill /
    // encMoeDownMmPrefill) read these to pick the right kernel PSO.
    let attnQFormat, attnKFormat, attnOutFormat: GGMLType
    let attnVFormat: GGMLType                          // valid iff attnV != nil
    let ffnGateFormat, ffnUpFormat, ffnDownFormat: GGMLType
    let moeGateUpFormat, moeDownFormat: GGMLType
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
    // F16 has no block structure (every element is a plain 2-byte half),
    // so no swizzling is needed. The GGUF-native row-major [Din, Dout]
    // layout matches what the F16 kernels expect — return the raw
    // MTLBuffer directly. Used for dense F16 weights and (since the same
    // loader works for any rank-2/3 F16 tensor) MoE F16 weights too.
    func loadF16Raw(_ name: String) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == .f16, "\(name): expected f16")
        return try g.makeMetalBuffer(name, device: device)
    }
    // Auto-detecting dense loader: reads the GGUF tensor's actual dtype and
    // dispatches to the right swizzler. Returns (buffer, format) tuple so
    // LayerW can populate the matching format field. Supports Q8_0/Q5_K/Q6_K
    // (the dense formats in the V1 grid that have AR-decode + prefill kernels).
    // Resolve the tensor class from a GGUF tensor name like "blk.5.attn_q.weight"
    // → "attn_q". The class drives both capability validation and dispatch.
    func tensorClassFromName(_ name: String) -> String {
        // Strip "blk.<L>." prefix if present.
        var stripped = name
        if let dotAfterBlk = stripped.range(of: "blk.")?.upperBound {
            if let nextDot = stripped[dotAfterBlk...].firstIndex(of: ".") {
                stripped = String(stripped[stripped.index(after: nextDot)...])
            }
        }
        // Strip ".weight" suffix.
        if stripped.hasSuffix(".weight") {
            stripped = String(stripped.dropLast(".weight".count))
        }
        return stripped
    }

    func loadDenseAuto(_ name: String) throws -> (MTLBuffer, GGMLType) {
        let info = try g.tensor(name)
        // Single source of truth: validate against kernel_capabilities.json
        // before the dispatch switch picks the right loader.
        assertCapability(tensorClassFromName(name), info.dtype, tensorName: name)
        switch info.dtype {
        case .q8_0:
            return (try loadQ80Swizzled(name), .q8_0)
        case .q5_K:
            return (try loadDenseSwizzled(name, dtype: .q5_K, blkBytes: 176, blkElems: 256), .q5_K)
        case .q6_K:
            return (try loadDenseSwizzled(name, dtype: .q6_K, blkBytes: 210, blkElems: 256), .q6_K)
        case .q5_1:
            return (try loadDenseSwizzled(name, dtype: .q5_1, blkBytes: 24, blkElems: 32), .q5_1)
        case .q4_0:
            return (try loadDenseSwizzled(name, dtype: .q4_0, blkBytes: 18, blkElems: 32), .q4_0)
        case .q4_1:
            return (try loadDenseSwizzled(name, dtype: .q4_1, blkBytes: 20, blkElems: 32), .q4_1)
        case .f16:
            return (try loadF16Raw(name), .f16)
        default:
            fail("loadDenseAuto(\(name)): unsupported dtype \(info.dtype) (capability check should have caught this; engine and capabilities matrix are out of sync)")
        }
    }
    // Auto-detecting MoE up loader (slot_token broadcast convention).
    // Supports Q4_K (Q4_K_M default), Q5_K (Q5_K_M default), Q4_0 (--pure Q4_0).
    func loadMoEUpAuto(_ name: String) throws -> (MTLBuffer, GGMLType) {
        let info = try g.tensor(name)
        assertCapability(tensorClassFromName(name), info.dtype, tensorName: name)
        switch info.dtype {
        case .q4_K:
            return (try loadMoESwizzled(name, dtype: .q4_K, blkBytes: 144, blkElems: 256, E: E_EXP), .q4_K)
        case .q5_K:
            return (try loadMoESwizzled(name, dtype: .q5_K, blkBytes: 176, blkElems: 256, E: E_EXP), .q5_K)
        case .q4_0:
            return (try loadMoESwizzled(name, dtype: .q4_0, blkBytes: 18, blkElems: 32, E: E_EXP), .q4_0)
        case .q4_1:
            return (try loadMoESwizzled(name, dtype: .q4_1, blkBytes: 20, blkElems: 32, E: E_EXP), .q4_1)
        case .f16:
            return (try loadF16Raw(name), .f16)
        default:
            fail("loadMoEUpAuto(\(name)): unsupported dtype \(info.dtype) (capability check should have caught this)")
        }
    }
    // Auto-detecting MoE down loader (per-slot convention). Each format
    // has its own dedicated per-slot kernel — no convention-mixing routes.
    func loadMoEDownAuto(_ name: String) throws -> (MTLBuffer, GGMLType) {
        let info = try g.tensor(name)
        assertCapability(tensorClassFromName(name), info.dtype, tensorName: name)
        switch info.dtype {
        case .q5_1:
            return (try loadMoESwizzled(name, dtype: .q5_1, blkBytes: 24, blkElems: 32, E: E_EXP), .q5_1)
        case .q6_K:
            return (try loadMoESwizzled(name, dtype: .q6_K, blkBytes: 210, blkElems: 256, E: E_EXP), .q6_K)
        case .q8_0:
            return (try loadMoESwizzled(name, dtype: .q8_0, blkBytes: 34, blkElems: 32, E: E_EXP), .q8_0)
        case .q4_0:
            return (try loadMoESwizzled(name, dtype: .q4_0, blkBytes: 18, blkElems: 32, E: E_EXP), .q4_0)
        case .q4_1:
            return (try loadMoESwizzled(name, dtype: .q4_1, blkBytes: 20, blkElems: 32, E: E_EXP), .q4_1)
        case .f16:
            return (try loadF16Raw(name), .f16)
        default:
            fail("loadMoEDownAuto(\(name)): unsupported dtype \(info.dtype) (capability check should have caught this)")
        }
    }
    // Generic dense weight swizzler for any quant. Source is GGUF-native
    // [Dout cols, nbc kb, blkBytes] (Dout rows, each row is nbc super-blocks).
    // Destination is v6 swizzled [n_super=Dout/32, nbc, 32 cols, blkBytes]:
    // 32 threads of an SG read 32 contiguous blocks per kb iteration. Same
    // shape as repackQ80ToSwizzled but parameterized by block geometry.
    func loadDenseSwizzled(_ name: String, dtype: GGMLType,
                            blkBytes: Int, blkElems: Int) throws -> MTLBuffer {
        let info = try g.tensor(name)
        precondition(info.dtype == dtype, "\(name): expected \(dtype)")
        let Din = info.shape[0], Dout = info.shape[1]
        precondition(Dout % 32 == 0, "\(name): Dout=\(Dout) must be a multiple of 32")
        precondition(Din % blkElems == 0, "\(name): Din=\(Din) must be a multiple of \(blkElems)")
        let raw = try g.makeMetalBuffer(name, device: device)
        let sw = device.makeBuffer(length: raw.length, options: .storageModeShared)!
        let nbc = Din / blkElems
        let colBytes = nbc * blkBytes
        let nSuper = Dout / 32
        let sp = raw.contents().assumingMemoryBound(to: UInt8.self)
        let dp = sw.contents().assumingMemoryBound(to: UInt8.self)
        for ns in 0..<nSuper {
            let srcColBase = ns * 32 * colBytes
            let dstSuperBase = ns * nbc * 32 * blkBytes
            for kb in 0..<nbc {
                for col in 0..<32 {
                    let srcOff = srcColBase + col * colBytes + kb * blkBytes
                    let dstOff = dstSuperBase + kb * 32 * blkBytes + col * blkBytes
                    memcpy(dp.advanced(by: dstOff), sp.advanced(by: srcOff), blkBytes)
                }
            }
        }
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
        // Load each tensor through the auto-detecting loader; capture
        // (buffer, format) tuples and unpack into LayerW. The format
        // fields drive the prefill dispatcher's PSO selection downstream.
        let (attnQBuf, attnQFmt)   = try loadDenseAuto("\(p)attn_q.weight")
        let (attnKBuf, attnKFmt)   = try loadDenseAuto("\(p)attn_k.weight")
        let (attnOutBuf, attnOutFmt) = try loadDenseAuto("\(p)attn_output.weight")
        let attnVTuple: (MTLBuffer, GGMLType)? = (g.tensors["\(p)attn_v.weight"] != nil)
            ? try loadDenseAuto("\(p)attn_v.weight") : nil
        let (ffnGateBuf, ffnGateFmt)   = try loadDenseAuto("\(p)ffn_gate.weight")
        let (ffnUpBuf, ffnUpFmt)       = try loadDenseAuto("\(p)ffn_up.weight")
        let (ffnDownBuf, ffnDownFmt)   = try loadDenseAuto("\(p)ffn_down.weight")
        let (moeUpBuf, moeUpFmt)       = try loadMoEUpAuto("\(p)ffn_gate_up_exps.weight")
        let (moeDownBuf, moeDownFmt)   = try loadMoEDownAuto("\(p)ffn_down_exps.weight")
        let lw = LayerW(
            attnQ:    attnQBuf,
            attnK:    attnKBuf,
            attnOut:  attnOutBuf,
            attnV:    attnVTuple?.0,
            ffnGate:  ffnGateBuf,
            ffnUp:    ffnUpBuf,
            ffnDown:  ffnDownBuf,
            moeGateUp: moeUpBuf,
            moeDown:   moeDownBuf,
            attnQFormat:    attnQFmt,
            attnKFormat:    attnKFmt,
            attnOutFormat:  attnOutFmt,
            attnVFormat:    attnVTuple?.1 ?? .q8_0,    // unused when attnV is nil
            ffnGateFormat:  ffnGateFmt,
            ffnUpFormat:    ffnUpFmt,
            ffnDownFormat:  ffnDownFmt,
            moeGateUpFormat: moeUpFmt,
            moeDownFormat:   moeDownFmt,
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
            let vNote = (attnVTuple == nil) ? " (V reused from K)" : " (V=\(attnVTuple!.1))"
            print("    layer \(L): \(isFull ? "full" : "slide") KV_H=\(lkv) HD=\(hd) Q=\(attnQFmt) FFN-down=\(ffnDownFmt) MoE-up=\(moeUpFmt) MoE-dn=\(moeDownFmt)\(vNote)")
        }
    }
    print(String(format: "  %d layers loaded+repacked in %.1f sec",
                 NUM_LAYERS, Date().timeIntervalSince(tLoad)))

    // ---------- Dequant tied token_embd → fp16 twice (embed table + transposed unembed) ----------
    // Format varies: Q4_K_M uses Q8_0 token_embd; Q5_K_M / Q6_K use Q6_K.
    // Both produce the same output (fp16 per-row embed table + transposed unembed).
    let tEmbed = Date()
    let embedInfo = try g.tensor("token_embd.weight")
    let eDin = embedInfo.shape[0], eDout = embedInfo.shape[1]
    precondition(eDin == HIDDEN && eDout == VOCAB, "embed shape mismatch")
    let embedTable = device.makeBuffer(length: VOCAB * HIDDEN * 2, options: .storageModeShared)!
    let unembedW   = device.makeBuffer(length: HIDDEN * VOCAB * 2, options: .storageModeShared)!
    let srcBase = g.base.advanced(by: embedInfo.dataOffset)
    let embedDp = embedTable.contents().assumingMemoryBound(to: Float16.self)
    let unembedDp = unembedW.contents().assumingMemoryBound(to: Float16.self)
    switch embedInfo.dtype {
    case .q8_0:
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
    case .q6_K:
        // Q6_K block: 210 bytes / 256 elts. Layout: ql[128], qh[64], scales[16] (i8), d (half).
        // Mirror of dequantize_q6_K_llama in MSL — produces 16 elts per il_orig in [0,16).
        let nbc = HIDDEN / 256
        let BLK = 210
        let colBytes = nbc * BLK
        for vo in 0..<VOCAB {
            let colBase = vo * colBytes
            for kb in 0..<nbc {
                let blkOff = colBase + kb * BLK
                let blk = srcBase.advanced(by: blkOff)
                let dAll = Float(blk.load(fromByteOffset: 208, as: Float16.self))
                for il in 0..<16 {
                    let qlBase = 32*(il/8) + 16*((il/2) & 1) + 8*(il & 1)
                    let qhBase = 16*(il/8) + 8*(il & 1)
                    let sc = Float(blk.load(fromByteOffset: 192 + (il % 2) + 2*(il/2), as: Int8.self))
                    let phase = (il/2) & 3
                    let kmask1: UInt32 = phase > 1 ? (phase > 2 ? 0xC0C0C0C0 : 0x30303030) : (phase > 0 ? 0x0C0C0C0C : 0x03030303)
                    let kmask2: UInt32 = phase > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F
                    let ml  = dAll * sc * 32.0
                    let dl0 = dAll * sc
                    let dl1 = dl0 / 256.0
                    let dl2 = dl0 / (256.0 * 256.0)
                    let dl3 = dl0 / (256.0 * 256.0 * 256.0)
                    let shr_h: UInt32 = phase > 2 ? 2 : 0
                    let shl_h: UInt32 = phase > 1 ? 0 : (phase > 0 ? 2 : 4)
                    let shr_l: UInt32 = phase > 1 ? 4 : 0
                    let baseD = kb * 256 + il * 16
                    for i in 0..<4 {
                        let low_lo  = UInt32(blk.load(fromByteOffset: (qlBase + 2*i) * 2, as: UInt16.self))
                        let low_hi  = UInt32(blk.load(fromByteOffset: (qlBase + 2*i + 1) * 2, as: UInt16.self))
                        let high_lo = UInt32(blk.load(fromByteOffset: 128 + (qhBase + 2*i) * 2, as: UInt16.self))
                        let high_hi = UInt32(blk.load(fromByteOffset: 128 + (qhBase + 2*i + 1) * 2, as: UInt16.self))
                        let low  = (low_lo  | (low_hi  << 16)) & kmask2
                        let high = (high_lo | (high_hi << 16)) & kmask1
                        let q = ((high << shl_h) >> shr_h) | (low >> shr_l)
                        let v0 = Float16(dl0 * Float(q & 0xFF) - ml)
                        let v1 = Float16(dl1 * Float(q & 0xFF00) - ml)
                        let v2 = Float16(dl2 * Float(q & 0xFF0000) - ml)
                        let v3 = Float16(dl3 * Float(q & 0xFF000000) - ml)
                        let kIdx0 = baseD + i*4
                        embedDp[vo * HIDDEN + kIdx0 + 0] = v0
                        embedDp[vo * HIDDEN + kIdx0 + 1] = v1
                        embedDp[vo * HIDDEN + kIdx0 + 2] = v2
                        embedDp[vo * HIDDEN + kIdx0 + 3] = v3
                        unembedDp[(kIdx0 + 0) * VOCAB + vo] = v0
                        unembedDp[(kIdx0 + 1) * VOCAB + vo] = v1
                        unembedDp[(kIdx0 + 2) * VOCAB + vo] = v2
                        unembedDp[(kIdx0 + 3) * VOCAB + vo] = v3
                    }
                }
            }
        }
    case .f16:
        // Source is already fp16 in GGUF row-major [VOCAB, HIDDEN] layout
        // (each vocab row holds HIDDEN halves contiguously). embedTable
        // wants [VOCAB, HIDDEN] → straight memcpy. unembedW wants the
        // transpose [HIDDEN, VOCAB] → element-wise transpose loop.
        let srcHalf = srcBase.assumingMemoryBound(to: Float16.self)
        memcpy(embedDp, srcHalf, VOCAB * HIDDEN * 2)
        for vo in 0..<VOCAB {
            let rowBase = vo * HIDDEN
            for k in 0..<HIDDEN {
                unembedDp[k * VOCAB + vo] = srcHalf[rowBase + k]
            }
        }
    default:
        fail("token_embd unsupported dtype \(embedInfo.dtype) — expected Q8_0, Q6_K, or F16")
    }
    print(String(format: "  token_embd \(embedInfo.dtype) → fp16 dequant in %.1f sec", Date().timeIntervalSince(tEmbed)))
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
    // Compare the first block (col=0, kb=0) of the swizzled buffer against the
    // raw source — should be byte-identical regardless of format. Use 32 bytes
    // as a format-agnostic minimum (smaller than any quant block size).
    let L0 = layers[0]
    let rawQ = try g.makeMetalBuffer("blk.0.attn_q.weight", device: device)
    let rawSp = rawQ.contents()
    let swDp = L0.attnQ.contents()
    var match = true
    for byte in 0..<32 {
        let rawB = rawSp.load(fromByteOffset: byte, as: UInt8.self)
        let swB  = swDp.load(fromByteOffset: byte, as: UInt8.self)
        if rawB != swB { match = false; break }
    }
    print("  spot-check: L0 attn_q[col=0,kb=0,32B] \(match ? "✓ matches" : "✗ MISMATCH") post-swizzle (\(L0.attnQFormat))")

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
    precomputeFlexBlockMaskSlide(slidingWindow: SLIDING_WINDOW)
    precomputeFlexBlockMaskFull()
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
    precomputeFlexBlockMaskSlide(slidingWindow: SLIDING_WINDOW)
    precomputeFlexBlockMaskFull()
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
        let WvFmt = (lw.attnV != nil) ? lw.attnVFormat : lw.attnKFormat
        encQKVDenseAR(cb, x: hidden, gammaBuf: lw.attnNorm, xNormBuf: hidden_norm,
                       Wq: lw.attnQ, qFmt: lw.attnQFormat,
                       Wk: lw.attnK, kFmt: lw.attnKFormat,
                       Wv: Wv, vFmt: WvFmt,
                       outQ: q_out, outK: k_out, outV: v_out,
                       Din: HIDDEN, DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                       activeB: B)
        encRMSNormG(cb, x: q_out, gammaBuf: lw.attnQNorm, out: q_out, D: HD, numVecs: B * H)
        encRope(cb, q_out, H: H, D: HD, rotary: rotary, theta: theta)
        encRMSNormG(cb, x: k_out, gammaBuf: lw.attnKNorm, out: k_out, D: HD, numVecs: B * KV_H)
        encRope(cb, k_out, H: KV_H, D: HD, rotary: rotary, theta: theta)
        encRMSNormNoScale(cb, x: v_out, out: v_out, D: HD, numVecs: B * KV_H)
        let pg = isFull ? PAGE_FULL : PAGE_SLIDE
        encKVWrite(cb, K: k_out, V: v_out, Kc: Kc, Vc: Vc, H: KV_H, D: HD, page: pg)
        let npBuf = isFull ? num_pages_full : num_pages_slide
        let klBuf = isFull ? k_len_full : k_len_slide
        encAttn(cb, Q: q_out, O: attn_out, Kc: Kc, Vc: Vc,
                kLenBuf: klBuf, H_Q: H, H_KV: KV_H, D: HD, isFull: isFull)
        encDenseGemvAR(cb, attn_out, lw.attnOut, format: lw.attnOutFormat, mlp_out,
                        Din: H * HD, Dout: HIDDEN, activeB: B)
        encRmsNormAdd(cb, x: mlp_out, gammaBuf: lw.postAttnNorm,
                      residual: hidden, out: hidden, N: HIDDEN, numVecs: B)
        encGateUpAR(cb, x: hidden, gammaBuf: lw.ffnNorm, xNormBuf: hidden_norm,
                     Wg: lw.ffnGate, gateFmt: lw.ffnGateFormat,
                     Wu: lw.ffnUp,   upFmt:   lw.ffnUpFormat,
                     gateOut: shrd_gate, fusedScratch: shrd_gate_up_fused,
                     Din: HIDDEN, Dout: SHARED_INT, activeB: B)
        encDenseGemvAR(cb, shrd_gate, lw.ffnDown, format: lw.ffnDownFormat, mlp_out,
                        Din: SHARED_INT, Dout: HIDDEN, activeB: B)
        encRMSNormG(cb, x: mlp_out, gammaBuf: lw.postFfn1Norm, out: mlp_out, D: HIDDEN, numVecs: B)
        encRouterPreNorm(cb, x: hidden, per_dim_scale: lw.routerScale, out: hidden_norm)
        encGemvV5(cb, hidden_norm, lw.routerW, router_lg, Din: HIDDEN, Dout: E_EXP)
        encSoftmaxTopk(cb, expertScaleBuf: lw.expertScale)
        encRouteCompact(cb)
        encRMSNormG(cb, x: hidden, gammaBuf: lw.preFfn2Norm, out: hidden_norm, D: HIDDEN, numVecs: B)
        encMoeUpGemvAR(cb, hidden_norm, lw.moeGateUp, format: lw.moeGateUpFormat, gate_up_fused,
                        Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: E_EXP, activeB: B)
        encMoeGeluMulFused(cb, fused: gate_up_fused, out: gate_proj, N_half: MOE_INT, numSlots: TOTAL_SLOTS)
        encMoeDownGemvAR(cb, gate_proj, lw.moeDown, format: lw.moeDownFormat, moe_down_out,
                          Din: MOE_INT, Dout: HIDDEN, numActive: E_EXP, activeB: B)
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
let isDumpRun = ProcessInfo.processInfo.environment["LM_DUMP_LAYERS"] != nil
    || ProcessInfo.processInfo.environment["LM_DUMP_L0_INTERNALS"] != nil
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_KL_REF"] == nil,
   ProcessInfo.processInfo.environment["LM_PREFILL_VALIDATE"] == nil,
   ProcessInfo.processInfo.environment["LM_GENERATE"] == nil,
   ProcessInfo.processInfo.environment["LM_MULTISESSION"] == nil,
   ProcessInfo.processInfo.environment["LM_TEST_CVEC_CACHE"] == nil,
   ProcessInfo.processInfo.environment["LM_TEST_CACHE_DIVERGENCE"] == nil,
   ProcessInfo.processInfo.environment["LM_PROFILE_PREFILL"] == nil,
   ProcessInfo.processInfo.environment["LM_PROFILE_AR"] == nil,
   !isDumpRun {
    // GGUF_PATH alone → LM forward benchmark. If LM_KL_REF, LM_PREFILL_VALIDATE,
    // LM_GENERATE, LM_MULTISESSION, LM_TEST_CVEC_CACHE or any dump flag is also
    // set, let those harnesses drive (all reuse loadLmWeights).
    runGgufPathHarness(ggufPath: ggufPath)
}
// LM_GENERATE was a reach-inside demo that took a raw prompt string
// and ran generation directly. Equivalent behavior is available through
// /v1/chat/completions via the HTTP API.
// LM_MULTISESSION / LM_MULTITURN_DEMO / LM_COMPOSITE_DEMO / LM_MULTIMODAL
// demos were inline-chat-template harnesses that bypassed the FFI/HTTP API
// and hand-crafted `<|turn>user\n...` strings. Removed — equivalent
// behavior is available through /v1/chat/completions with the same
// messages payloads, using the model's real jinja template.
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
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_PROFILE_PREFILL"] != nil {
    runLmPrefillProfile(ggufPath: ggufPath)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_PROFILE_AR"] != nil {
    runLmARProfile(ggufPath: ggufPath)
}
if let outDir = ProcessInfo.processInfo.environment["FLEX_ATTN_TEST"] {
    runFlexAttnSlideV1Test(outDir: outDir)
    runFlexAttnFullPrefillTest(outDir: outDir)
}
if ProcessInfo.processInfo.environment["ATTN_BENCH"] != nil {
    runAttnBench()
}
if ProcessInfo.processInfo.environment["KV_VIZ"] != nil {
    runKvVisualizer()
}
// LM_SHARED_PREFIX_DEMO / LM_BRANCH_DEMO / LM_PAUSE_DEMO were inline-
// template harnesses replicating what /v1/chat/completions does. Removed.
// The scheduler features these demos exercised (prefix caching, branch,
// pause/resume) are exercised through the HTTP API by normal clients.
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
// CVEC correctness harness. LM_TEST_CVEC_DIGEST=1 runs the pure-logic
// digest unit tests (fast, no weights). LM_TEST_CVEC_CACHE=1 runs the
// full integration test (requires GGUF_PATH).
if ProcessInfo.processInfo.environment["LM_TEST_CVEC_DIGEST"] != nil {
    runCvecDigestUnitTests()
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_TEST_CVEC_CACHE"] != nil {
    runCvecDigestUnitTests()
    runCvecCacheIntegrationTest(ggufPath: ggufPath)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_TEST_CACHE_DIVERGENCE"] != nil {
    runPrefillCacheDivergenceDump(ggufPath: ggufPath)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_TEST_PREFIX_CACHE"] != nil {
    runPrefixCacheSmoke(ggufPath: ggufPath)
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
