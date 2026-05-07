// profile_ar_step — per-stage GPU timing of buildStepCB (AR decode).
//
// Mirrors profile_prefill.swift's design: split each AR step's forward
// pass into per-kernel-family CBs, time each via gpuStartTime/gpuEndTime,
// aggregate across reps.
//
// Driven by env: LM_PROFILE_AR=1 GGUF_PATH=… [LM_PROFILE_AR_REPS=2]
//   GGUF_PATH                LM weights
//   LM_PROFILE_AR_REPS       reps to time (default 5)
//   LM_PROFILE_AR_KCTX       k_len to pre-fill K/V cache to (default 64)
//
// Stages per layer (matching prefill structure, but AR-shaped — numVecs=B):
//   qkv         — fused RMSNorm+Q/K/V proj
//   qkn_rope    — Q/K/V per-head norms + RoPE (single-pos)
//   kv_attn     — KV cache write (single position) + paged attention
//   oproj_norm  — o_proj + post-attn RMSNormAdd
//   shrd_ffn    — shared FFN gate_up + gelu + down + post-FFN1 norm
//   router      — pre-norm + GEMV + softmax/topk + compact
//   moe         — pre-FFN2 norm + Q4_K gate_up + gelu + Q5_1 down + combine + post-FFN2 norm
//   resid       — copy + add + final post-FFN RMSNormAdd_scale
// + head: embed + scale
// + tail: output norm + V4Softcap + sample

import Metal
import Foundation

private struct ARStage {
    let name: String
    let layer: Int    // -1 for head/tail
    let gpuMs: Double
    let cpuMs: Double
}

private func runARStage(_ name: String, layer: Int,
                         _ encode: (MTLCommandBuffer) -> Void) -> ARStage {
    let cb = queue.makeCommandBuffer()!
    encode(cb)
    let cpuT0 = Date()
    cb.commit()
    cb.waitUntilCompleted()
    let cpuT1 = Date()
    let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
    let cpuMs = cpuT1.timeIntervalSince(cpuT0) * 1000.0
    return ARStage(name: name, layer: layer, gpuMs: gpuMs, cpuMs: cpuMs)
}

func runLmARProfile(ggufPath: String) {
    print("\n=== AR step profile (per-stage GPU timing) ===")
    let repsEnv = ProcessInfo.processInfo.environment["LM_PROFILE_AR_REPS"].flatMap { Int($0) }
    let reps = max(1, repsEnv ?? 5)
    let kctxEnv = ProcessInfo.processInfo.environment["LM_PROFILE_AR_KCTX"].flatMap { Int($0) }
    let kctx = max(1, kctxEnv ?? 64)
    print("  B=\(B)  reps=\(reps) (+1 warmup)  k_len=\(kctx) (preset, single AR step at end)")

    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }

    // This profiler dispatches Q8_0-specific kernels by name (the whole point
    // is to A/B compare Q8_0 kernel variants). For non-Q8_0 dense formats the
    // measurements would be meaningless because the kernels would interpret
    // weight bytes incorrectly. Fail loud rather than silently produce garbage.
    let l0 = w.layers[0]
    precondition(l0.attnQFormat == .q8_0 && l0.ffnDownFormat == .q8_0,
                 "LM_PROFILE_AR requires Q8_0 dense layers (got Q=\(l0.attnQFormat), FFN-down=\(l0.ffnDownFormat)). Use the Q4_K_M GGUF for AR-step profiling.")

    // Pre-fill all B slots' K/V cache to kctx positions via a normal
    // prefill, so the AR step we time has a realistic K/V context to
    // attend over (not k_len=1). Use the existing fast-prefill path.
    let qLen = min(MAX_Q_LEN, kctx)
    let tokP = pre_input_tokens.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
    let posP = pre_q_positions.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
    for b in 0..<B {
        for i in 0..<qLen {
            tokP[b * qLen + i] = 1
            posP[b * qLen + i] = UInt32(i)
        }
    }
    let pklsP = pre_k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
    let pklfP = pre_k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
    for b in 0..<B { pklsP[b] = UInt32(qLen); pklfP[b] = UInt32(qLen) }
    let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
    for b in 0..<B {
        for p in 0..<MAX_PAGES_PER_SLOT {
            btP[b * MAX_PAGES_PER_SLOT + p] = UInt32(b * MAX_PAGES_PER_SLOT + p) % UInt32(TOTAL_PAGES)
        }
    }
    precomputeFlexPrefillMasks(qLen: qLen, positionStart: 0)
    // skipUnembed=true: the priming pass only seeds K/V; logits aren't
    // needed (the AR-step profile does its own unembed at the end).
    // Skipping avoids the sampling kernel reading uninitialized
    // sampling_* buffers (the harness doesn't call populateSamplingParams).
    let prefillCB = buildPrefillCB(w, qLen: qLen, skipUnembed: true)
    prefillCB.commit(); prefillCB.waitUntilCompleted()
    if let err = prefillCB.error { print("  prefill priming error: \(err)"); return }
    print("  prefill priming done (k_len=\(qLen) seeded across B=\(B) slots)")

    // Set up AR-step state: positions point at next position (= qLen),
    // k_len_* set so attention sees qLen positions, num_pages set to
    // ceil(qLen / PAGE).
    let posBuf = positions.contents().bindMemory(to: UInt32.self, capacity: B)
    let arklsP = k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
    let arklfP = k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
    let npsBuf = num_pages_slide.contents().bindMemory(to: UInt32.self, capacity: B)
    let npfBuf = num_pages_full.contents().bindMemory(to: UInt32.self, capacity: B)
    let inputBuf = input_tokens.contents().bindMemory(to: UInt32.self, capacity: B)
    for b in 0..<B {
        posBuf[b] = UInt32(qLen)
        arklsP[b] = UInt32(qLen + 1)        // attention sees positions 0..qLen
        arklfP[b] = UInt32(qLen + 1)
        npsBuf[b] = UInt32((qLen + PAGE_SLIDE - 1) / PAGE_SLIDE)
        npfBuf[b] = UInt32((qLen + PAGE_FULL - 1) / PAGE_FULL)
        inputBuf[b] = 1
    }

    func onePass() -> [ARStage] {
        var stages: [ARStage] = []

        stages.append(runARStage("embed", layer: -1) { cb in
            encEmbed(cb, embedTable: w.embedTable)
            encScaleByScalar(cb, x: hidden, scalar: w.embedScaleBuf, N: HIDDEN, numVecs: B)
        })

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

            stages.append(runARStage("qkv", layer: L) { cb in
                let Wv = lw.attnV ?? lw.attnK
                encGemvQ80V6RmsnormQKV(cb, x: hidden, gammaBuf: lw.attnNorm,
                                        Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                        outQ: q_out, outK: k_out, outV: v_out,
                                        Din: HIDDEN,
                                        DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD)
            })
            stages.append(runARStage("qkv_b1", layer: L) { cb in
                // numVecs=1: dispatch grid height = 1, only b=0's TG runs.
                // Tests user hypothesis: is per-stage time dominated by 7 silenced-slot TGs?
                let Wv = lw.attnV ?? lw.attnK
                encGemvQ80V6RmsnormQKV(cb, x: hidden, gammaBuf: lw.attnNorm,
                                        Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                        outQ: q_out, outK: k_out, outV: v_out,
                                        Din: HIDDEN,
                                        DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                                        numVecs: 1)
            })
            // V7-as-btile_qkv_b4-at-numVecs=1: existing V7 with VEC_TILE=4
            // dispatched with numVecs=1, so v_count=1 inside (3 predicated
            // accumulator slots wasted — proxy for templated b1 win).
            stages.append(runARStage("qkv_v7_n1", layer: L) { cb in
                let Wv = lw.attnV ?? lw.attnK
                encGemvQ80V7RmsnormQKV(cb, x: hidden, gammaBuf: lw.attnNorm,
                                        Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                        outQ: q_out, outK: k_out, outV: v_out,
                                        Din: HIDDEN,
                                        DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                                        numVecs: 1)
            })
            // Templated QKV kernel zoo: per-width specialized.
            stages.append(runARStage("btile_qkv_b1", layer: L) { cb in
                let Wv = lw.attnV ?? lw.attnK
                encGemvQ80BtileQKV(cb, x: hidden, gammaBuf: lw.attnNorm,
                                    Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                    outQ: q_out, outK: k_out, outV: v_out,
                                    Din: HIDDEN,
                                    DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                                    activeB: 1)
            })
            stages.append(runARStage("btile_qkv_b2", layer: L) { cb in
                let Wv = lw.attnV ?? lw.attnK
                encGemvQ80BtileQKV(cb, x: hidden, gammaBuf: lw.attnNorm,
                                    Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                    outQ: q_out, outK: k_out, outV: v_out,
                                    Din: HIDDEN,
                                    DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                                    activeB: 2)
            })
            stages.append(runARStage("btile_qkv_b4", layer: L) { cb in
                let Wv = lw.attnV ?? lw.attnK
                encGemvQ80BtileQKV(cb, x: hidden, gammaBuf: lw.attnNorm,
                                    Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                    outQ: q_out, outK: k_out, outV: v_out,
                                    Din: HIDDEN,
                                    DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                                    activeB: 4)
            })
            // OTF (on-the-fly normalize) variants — full {1,2,4,8} coverage.
            stages.append(runARStage("btile_qkv_otf_b1", layer: L) { cb in
                let Wv = lw.attnV ?? lw.attnK
                encGemvQ80BtileQKVOtf(cb, x: hidden, gammaBuf: lw.attnNorm,
                                       Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                       outQ: q_out, outK: k_out, outV: v_out,
                                       Din: HIDDEN,
                                       DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                                       activeB: 1)
            })
            stages.append(runARStage("btile_qkv_otf_b4", layer: L) { cb in
                let Wv = lw.attnV ?? lw.attnK
                encGemvQ80BtileQKVOtf(cb, x: hidden, gammaBuf: lw.attnNorm,
                                       Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                       outQ: q_out, outK: k_out, outV: v_out,
                                       Din: HIDDEN,
                                       DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                                       activeB: 4)
            })
            stages.append(runARStage("btile_qkv_otf_b8", layer: L) { cb in
                let Wv = lw.attnV ?? lw.attnK
                encGemvQ80BtileQKVOtf(cb, x: hidden, gammaBuf: lw.attnNorm,
                                       Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                       outQ: q_out, outK: k_out, outV: v_out,
                                       Din: HIDDEN,
                                       DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                                       activeB: 8)
            })

            stages.append(runARStage("qkn_rope", layer: L) { cb in
                encRMSNormG(cb, x: q_out, gammaBuf: lw.attnQNorm, out: q_out, D: HD, numVecs: B * H)
                encRope(cb, q_out, H: H, D: HD, rotary: rotary, theta: theta)
                encRMSNormG(cb, x: k_out, gammaBuf: lw.attnKNorm, out: k_out, D: HD, numVecs: B * KV_H)
                encRope(cb, k_out, H: KV_H, D: HD, rotary: rotary, theta: theta)
                encRMSNormNoScale(cb, x: v_out, out: v_out, D: HD, numVecs: B * KV_H)
            })

            stages.append(runARStage("kv_attn", layer: L) { cb in
                let pg = isFull ? PAGE_FULL : PAGE_SLIDE
                encKVWrite(cb, K: k_out, V: v_out, Kc: Kc, Vc: Vc, H: KV_H, D: HD, page: pg)
                let klBuf = isFull ? k_len_full : k_len_slide
                encAttn(cb, Q: q_out, O: attn_out, Kc: Kc, Vc: Vc,
                        kLenBuf: klBuf, H_Q: H, H_KV: KV_H, D: HD,
                        isFull: isFull)
            })

            stages.append(runARStage("oproj_norm", layer: L) { cb in
                encGemvQ80V6(cb, attn_out, lw.attnOut, mlp_out, Din: H * HD, Dout: HIDDEN)
                encRmsNormAdd(cb, x: mlp_out, gammaBuf: lw.postAttnNorm,
                              residual: hidden, out: hidden, N: HIDDEN, numVecs: B)
            })
            stages.append(runARStage("oproj_norm_b1", layer: L) { cb in
                encGemvQ80V6(cb, attn_out, lw.attnOut, mlp_out, Din: H * HD, Dout: HIDDEN, numVecs: 1)
                encRmsNormAdd(cb, x: mlp_out, gammaBuf: lw.postAttnNorm,
                              residual: hidden, out: hidden, N: HIDDEN, numVecs: 1)
            })
            // Kernel zoo A/B: templated dense_gemv_q8_0 at compile-time
            // fixed B_TILE. Pure GEMV cost (no RMSNormAdd appended) so
            // we can compare apples-to-apples to the V6 GEMV inside.
            stages.append(runARStage("btile_oproj_b1", layer: L) { cb in
                encGemvQ80Btile(cb, attn_out, lw.attnOut, mlp_out, Din: H * HD, Dout: HIDDEN, activeB: 1)
            })
            stages.append(runARStage("btile_oproj_b2", layer: L) { cb in
                encGemvQ80Btile(cb, attn_out, lw.attnOut, mlp_out, Din: H * HD, Dout: HIDDEN, activeB: 2)
            })
            stages.append(runARStage("btile_oproj_b4", layer: L) { cb in
                encGemvQ80Btile(cb, attn_out, lw.attnOut, mlp_out, Din: H * HD, Dout: HIDDEN, activeB: 4)
            })
            stages.append(runARStage("btile_oproj_b8", layer: L) { cb in
                encGemvQ80Btile(cb, attn_out, lw.attnOut, mlp_out, Din: H * HD, Dout: HIDDEN, activeB: 8)
            })

            stages.append(runARStage("shrd_ffn", layer: L) { cb in
                encGemvQ80V6RmsnormGateUp(cb, x: hidden, gammaBuf: lw.ffnNorm,
                                           Wg: lw.ffnGate, Wu: lw.ffnUp,
                                           fusedOut: shrd_gate_up_fused,
                                           Din: HIDDEN, Dout: SHARED_INT)
                encMoeGeluMulFused(cb, fused: shrd_gate_up_fused, out: shrd_gate, N_half: SHARED_INT, numSlots: B)
                encGemvQ80V6(cb, shrd_gate, lw.ffnDown, mlp_out, Din: SHARED_INT, Dout: HIDDEN)
                encRMSNormG(cb, x: mlp_out, gammaBuf: lw.postFfn1Norm, out: mlp_out, D: HIDDEN, numVecs: B)
            })
            stages.append(runARStage("shrd_ffn_b1", layer: L) { cb in
                encGemvQ80V6RmsnormGateUp(cb, x: hidden, gammaBuf: lw.ffnNorm,
                                           Wg: lw.ffnGate, Wu: lw.ffnUp,
                                           fusedOut: shrd_gate_up_fused,
                                           Din: HIDDEN, Dout: SHARED_INT, numVecs: 1)
                encMoeGeluMulFused(cb, fused: shrd_gate_up_fused, out: shrd_gate, N_half: SHARED_INT, numSlots: 1)
                encGemvQ80V6(cb, shrd_gate, lw.ffnDown, mlp_out, Din: SHARED_INT, Dout: HIDDEN, numVecs: 1)
                encRMSNormG(cb, x: mlp_out, gammaBuf: lw.postFfn1Norm, out: mlp_out, D: HIDDEN, numVecs: 1)
            })

            stages.append(runARStage("router", layer: L) { cb in
                encRouterPreNorm(cb, x: hidden, per_dim_scale: lw.routerScale, out: hidden_norm)
                encGemvV5(cb, hidden_norm, lw.routerW, router_lg, Din: HIDDEN, Dout: E_EXP)
                encSoftmaxTopk(cb, expertScaleBuf: lw.expertScale)
                encRouteCompact(cb)
            })

            stages.append(runARStage("moe", layer: L) { cb in
                encRMSNormG(cb, x: hidden, gammaBuf: lw.preFfn2Norm, out: hidden_norm, D: HIDDEN, numVecs: B)
                encMoeGemvQ4K(cb, hidden_norm, lw.moeGateUp, gate_up_fused,
                              Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: E_EXP, useV6: true)
                encMoeGeluMulFused(cb, fused: gate_up_fused, out: gate_proj, N_half: MOE_INT, numSlots: TOTAL_SLOTS)
                encMoeGemvQ51(cb, gate_proj, lw.moeDown, moe_down_out, Din: MOE_INT, Dout: HIDDEN, numActive: E_EXP, useV6: true)
                encMoeCombineWrite(cb, to: moe_sum)
                encRMSNormG(cb, x: moe_sum, gammaBuf: lw.postFfn2Norm, out: moe_sum, D: HIDDEN, numVecs: B)
            })
            stages.append(runARStage("moe_b1", layer: L) { cb in
                // numActive=TOPK: only the 8 experts of slot 0 are dispatched.
                // numSlots=TOPK: gelu_mul over the 8 expert slots of slot 0 only.
                encRMSNormG(cb, x: hidden, gammaBuf: lw.preFfn2Norm, out: hidden_norm, D: HIDDEN, numVecs: 1)
                encMoeGemvQ4K(cb, hidden_norm, lw.moeGateUp, gate_up_fused,
                              Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: TOPK, useV6: true)
                encMoeGeluMulFused(cb, fused: gate_up_fused, out: gate_proj, N_half: MOE_INT, numSlots: TOPK)
                encMoeGemvQ51(cb, gate_proj, lw.moeDown, moe_down_out, Din: MOE_INT, Dout: HIDDEN, numActive: TOPK, useV6: true)
                encMoeCombineWrite(cb, to: moe_sum)
                encRMSNormG(cb, x: moe_sum, gammaBuf: lw.postFfn2Norm, out: moe_sum, D: HIDDEN, numVecs: 1)
            })

            stages.append(runARStage("resid", layer: L) { cb in
                encBufferCopy(cb, src: mlp_out, dst: ffn_combined, bytes: B * HIDDEN * 2)
                encAddInplace(cb, dst: ffn_combined, src: moe_sum, N: HIDDEN, numVecs: B)
                encRmsNormAddScale(cb, x: ffn_combined, gammaBuf: lw.postFfnNorm,
                                   residual: hidden, scalar: lw.layerOutputScale,
                                   out: hidden, N: HIDDEN, numVecs: B)
            })
            stages.append(runARStage("resid_b1", layer: L) { cb in
                encBufferCopy(cb, src: mlp_out, dst: ffn_combined, bytes: 1 * HIDDEN * 2)
                encAddInplace(cb, dst: ffn_combined, src: moe_sum, N: HIDDEN, numVecs: 1)
                encRmsNormAddScale(cb, x: ffn_combined, gammaBuf: lw.postFfnNorm,
                                   residual: hidden, scalar: lw.layerOutputScale,
                                   out: hidden, N: HIDDEN, numVecs: 1)
            })
        }

        stages.append(runARStage("unembed", layer: -1) { cb in
            encRMSNormG(cb, x: hidden, gammaBuf: w.outputNorm, out: hidden_norm,
                        D: HIDDEN, numVecs: B)
            encGemvV4Softcap(cb, hidden_norm, w.unembedW, logits,
                              Din: HIDDEN, Dout: VOCAB, cap: 30.0)
        })

        return stages
    }

    print("  warmup pass (discarded)..."); fflush(stdout)
    _ = onePass()

    var aggGpu: [String: Double] = [:]
    var aggCpu: [String: Double] = [:]
    var aggCount: [String: Int] = [:]
    var fullPassGpu: [Double] = []
    var fullPassCpu: [Double] = []
    var perLayerGpu: [String: [Double]] = [:]

    for r in 0..<reps {
        print("  rep \(r+1)/\(reps)..."); fflush(stdout)
        let stages = onePass()
        var passGpu = 0.0, passCpu = 0.0
        for s in stages {
            aggGpu[s.name, default: 0] += s.gpuMs
            aggCpu[s.name, default: 0] += s.cpuMs
            aggCount[s.name, default: 0] += 1
            passGpu += s.gpuMs
            passCpu += s.cpuMs
            if r == 0 {
                perLayerGpu[s.name, default: []].append(s.gpuMs)
            }
        }
        fullPassGpu.append(passGpu)
        fullPassCpu.append(passCpu)
    }
    var fullLayerIndices: [Int] = []
    var slideLayerIndices: [Int] = []
    for L in 0..<NUM_LAYERS {
        if w.layers[L].isFull { fullLayerIndices.append(L) } else { slideLayerIndices.append(L) }
    }

    func pad(_ s: String, _ width: Int) -> String {
        return s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }
    func padf(_ d: Double, _ width: Int = 12, _ p: Int = 3) -> String {
        let s = String(format: "%.\(p)f", d)
        return s.count >= width ? s : String(repeating: " ", count: width - s.count) + s
    }

    let stageOrder = ["embed",
                       "qkv", "qkv_b1", "qkv_v7_n1",
                       "btile_qkv_b1", "btile_qkv_b2", "btile_qkv_b4",
                       "btile_qkv_otf_b1", "btile_qkv_otf_b4", "btile_qkv_otf_b8",
                       "qkn_rope", "kv_attn",
                       "oproj_norm", "oproj_norm_b1",
                       "btile_oproj_b1", "btile_oproj_b2", "btile_oproj_b4", "btile_oproj_b8",
                       "shrd_ffn", "shrd_ffn_b1",
                       "router",
                       "moe", "moe_b1",
                       "resid", "resid_b1",
                       "unembed"]
    print("")
    print("=== aggregate per-stage GPU time (across \(reps) AR steps, summed across layers/passes) ===")
    print("  \(pad("stage", 14))\(pad("count", 8))\(pad("total_gpu_ms", 14))\(pad("avg_per_call_ms", 18))\(pad("per_step_ms", 14))")
    var grandGpu = 0.0
    for name in stageOrder {
        guard let g = aggGpu[name], let c = aggCount[name] else { continue }
        let perStep = g / Double(reps)
        let perCall = g / Double(c)
        grandGpu += g
        print("  \(pad(name, 14))\(pad(String(c), 8))\(padf(g, 14, 2))\(padf(perCall, 18, 4))\(padf(perStep, 14, 2))")
    }
    let grandPerStep = grandGpu / Double(reps)
    print("  \(pad("TOTAL", 14))\(pad("", 8))\(padf(grandGpu, 14, 2))\(padf(0, 18, 4))\(padf(grandPerStep, 14, 2))")
    print("")
    let avgWall = fullPassGpu.reduce(0,+) / Double(reps)
    let avgCpuWall = fullPassCpu.reduce(0,+) / Double(reps)
    print(String(format: "  per-step GPU wall (sum of stage GPU times): %.2f ms", avgWall))
    print(String(format: "  per-step CPU wall (sum of stage CPU times): %.2f ms (CB-boundary overhead)", avgCpuWall))
    let aggregateTokSec = Double(B) / (grandPerStep / 1000.0)
    let perStreamTokSec = 1.0 / (grandPerStep / 1000.0)
    print(String(format: "  aggregate tok/s @ B=%d: %.2f", B, aggregateTokSec))
    print(String(format: "  per-stream tok/s:        %.2f", perStreamTokSec))

    print("")
    print("=== slide (\(slideLayerIndices.count) layers) vs full (\(fullLayerIndices.count) layers) ===")
    let attnStages = ["qkv", "qkn_rope", "kv_attn", "oproj_norm"]
    print("  \(pad("stage", 14))\(pad("slide_avg_ms", 16))\(pad("full_avg_ms", 16))\(pad("ratio", 10))")
    for name in attnStages {
        guard let arr = perLayerGpu[name], arr.count == NUM_LAYERS else { continue }
        var slideSum = 0.0, fullSum = 0.0
        for L in slideLayerIndices { slideSum += arr[L] }
        for L in fullLayerIndices { fullSum += arr[L] }
        let slideAvg = slideSum / Double(slideLayerIndices.count)
        let fullAvg = fullSum / Double(fullLayerIndices.count)
        let ratio = fullAvg / max(slideAvg, 1e-9)
        print("  \(pad(name, 14))\(padf(slideAvg, 16, 4))\(padf(fullAvg, 16, 4))\(padf(ratio, 10, 2))×")
    }

    fflush(stdout)
}
