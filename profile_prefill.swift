// profile_prefill — per-stage GPU timing of buildPrefillCB.
//
// Mirrors profile_vision_tower's design: split the prefill forward into
// stages, each committed as its own MTLCommandBuffer so
// gpuStartTime/gpuEndTime gives an honest GPU wall per stage. CB-boundary
// overhead (~100 μs each, 8 stages × 30 layers + 2 head/tail = 242 CBs ≈
// 24 ms) is small vs ~10 s prefill = <0.3%.
//
// Driven by env: LM_PROFILE_PREFILL=1 GGUF_PATH=… [LM_PROFILE_QLEN=32]
//   GGUF_PATH      LM weights (loaded via existing loadLmWeights)
//   LM_PROFILE_QLEN  prefill batch size N=B*qLen (default uses MAX_Q_LEN)
//   LM_PROFILE_REPS  warm passes counted (default 1, +1 untimed warmup)
//
// Stages (per layer):
//   qkv         — fused RMSNorm+Q/K/V proj  (encGemvQ80V6RmsnormQKV)
//   qkn_rope    — Q/K/V per-head norms + RoPE (5 dispatches)
//   kv_attn     — KV cache write + flex attention (slide or full)
//   oproj_norm  — o_proj + post-attn RMSNormAdd (residual fold)
//   shrd_ffn    — shared FFN gate_up + gelu·up + down + post-FFN1 norm
//   router      — router pre-norm + GEMV + softmax/topk + route compact
//   moe         — pre-FFN2 norm + Q4_K gate_up + gelu + Q5_1 down + combine + post-FFN2 norm
//   resid       — copy + add + final post-FFN RMSNormAdd_scale
// + head: embed + scale
// + tail: output norm + unembed + softcap

import Metal
import Foundation

private struct PrefillStage {
    let name: String
    let layer: Int      // -1 for head/tail stages
    let gpuMs: Double
    let cpuMs: Double
}

private func runStage(_ name: String, layer: Int,
                       _ encode: (MTLCommandBuffer) -> Void) -> PrefillStage {
    let cb = queue.makeCommandBuffer()!
    encode(cb)
    let cpuT0 = Date()
    cb.commit()
    cb.waitUntilCompleted()
    let cpuT1 = Date()
    let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
    let cpuMs = cpuT1.timeIntervalSince(cpuT0) * 1000.0
    return PrefillStage(name: name, layer: layer, gpuMs: gpuMs, cpuMs: cpuMs)
}

func runLmPrefillProfile(ggufPath: String) {
    print("\n=== prefill profile (per-stage GPU timing) ===")
    let qLenEnv = ProcessInfo.processInfo.environment["LM_PROFILE_QLEN"].flatMap { Int($0) }
    let qLen = min(qLenEnv ?? MAX_Q_LEN, MAX_Q_LEN)
    let repsEnv = ProcessInfo.processInfo.environment["LM_PROFILE_REPS"].flatMap { Int($0) }
    let reps = max(1, repsEnv ?? 1)
    print("  qLen=\(qLen)  B=\(B)  N=B*qLen=\(B*qLen)  reps=\(reps) (+1 warmup)")

    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }

    // Fake prompt: token id 1 across all positions, q_positions = 0..qLen-1.
    // This is enough to fire all kernels — we don't care about output values.
    let tokP = pre_input_tokens.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
    let posP = pre_q_positions.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
    for b in 0..<B {
        for i in 0..<qLen {
            tokP[b * qLen + i] = 1
            posP[b * qLen + i] = UInt32(i)
        }
    }
    let klsP = pre_k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
    let klfP = pre_k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
    for b in 0..<B { klsP[b] = UInt32(qLen); klfP[b] = UInt32(qLen) }
    let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
    for b in 0..<B {
        for p in 0..<MAX_PAGES_PER_SLOT {
            btP[b * MAX_PAGES_PER_SLOT + p] = UInt32(b * MAX_PAGES_PER_SLOT + p) % UInt32(TOTAL_PAGES)
        }
    }
    precomputeFlexPrefillMasks(qLen: qLen, positionStart: 0)

    let N = B * qLen
    let NS = B * qLen * TOPK

    // One pass of profiled forward. Returns the stages collected.
    func onePass() -> [PrefillStage] {
        var stages: [PrefillStage] = []

        stages.append(runStage("embed", layer: -1) { cb in
            encEmbedInto(cb, tokens: pre_input_tokens, embedTable: w.embedTable,
                         out: pre_hidden, numVecs: N)
            encScaleByScalar(cb, x: pre_hidden, scalar: w.embedScaleBuf, N: HIDDEN, numVecs: N)
        })

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

            stages.append(runStage("qkv", layer: L) { cb in
                let Wv = lw.attnV ?? lw.attnK
                encGemvQ80V6RmsnormQKV(cb, x: pre_hidden, gammaBuf: lw.attnNorm,
                                        Wq: lw.attnQ, Wk: lw.attnK, Wv: Wv,
                                        outQ: q_out, outK: k_out, outV: v_out,
                                        Din: HIDDEN,
                                        DoutQ: H * HD, DoutK: KV_H * HD, DoutV: KV_H * HD,
                                        numVecs: N)
            })

            stages.append(runStage("qkn_rope", layer: L) { cb in
                encRMSNormG(cb, x: q_out, gammaBuf: lw.attnQNorm, out: q_out, D: HD, numVecs: N * H)
                encRopeMulti(cb, q_out, q_positions: pre_q_positions, H: H, D: HD,
                             rotary: rotary, theta: theta, qLen: qLen)
                encRMSNormG(cb, x: k_out, gammaBuf: lw.attnKNorm, out: k_out, D: HD, numVecs: N * KV_H)
                encRopeMulti(cb, k_out, q_positions: pre_q_positions, H: KV_H, D: HD,
                             rotary: rotary, theta: theta, qLen: qLen)
                encRMSNormNoScale(cb, x: v_out, out: v_out, D: HD, numVecs: N * KV_H)
            })

            stages.append(runStage("kv_attn", layer: L) { cb in
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
            })

            stages.append(runStage("oproj_norm", layer: L) { cb in
                encGemvQ80V6(cb, pre_attn_out, lw.attnOut, pre_mlp_out,
                             Din: H * HD, Dout: HIDDEN, numVecs: N)
                encRmsNormAdd(cb, x: pre_mlp_out, gammaBuf: lw.postAttnNorm,
                              residual: pre_hidden, out: pre_hidden, N: HIDDEN, numVecs: N)
            })

            stages.append(runStage("shrd_ffn", layer: L) { cb in
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
            })

            stages.append(runStage("router", layer: L) { cb in
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
            })

            // Split the moe stage into 4 sub-stages so we can see per-kernel
            // timings (gate_up Q4_K vs gelu vs down Q5_1 vs combine+norms).
            stages.append(runStage("moe_q4k_v6", layer: L) { cb in
                encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.preFfn2Norm, out: pre_hidden_norm,
                            D: HIDDEN, numVecs: N)
                encMoeGemvQ4K(cb, pre_hidden_norm, lw.moeGateUp, pre_gate_up_fused,
                              Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: E_EXP, useV6: true,
                              slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
            })
            stages.append(runStage("moe_q4k_v8", layer: L) { cb in
                // V8 dispatch shadow run — same input, throwaway output (overwrites
                // pre_gate_up_fused). Gives a parallel A/B timing.
                encMoeGemvQ4K(cb, pre_hidden_norm, lw.moeGateUp, pre_gate_up_fused,
                              Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: E_EXP, useV8: true,
                              slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
            })
            stages.append(runStage("moe_q4k_v9", layer: L) { cb in
                encMoeGemvQ4K(cb, pre_hidden_norm, lw.moeGateUp, pre_gate_up_fused,
                              Din: HIDDEN, Dout: MOE_FUSED_DOUT, numActive: E_EXP, useV9: true,
                              slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
            })
            stages.append(runStage("moe_q4k_v10_fused", layer: L) { cb in
                // Writes [slots, MOE_INT] post-activation directly to
                // pre_gate_proj. AB compares this single stage vs the
                // v6/v9 matmul stage + the moe_gelu stage combined.
                encMoeGemvQ4KFusedGelu(cb, pre_hidden_norm, lw.moeGateUp, pre_gate_proj,
                                        Din: HIDDEN, NHalf: MOE_INT, numActive: E_EXP,
                                        slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
            })
            stages.append(runStage("moe_gelu", layer: L) { cb in
                encMoeGeluMulFused(cb, fused: pre_gate_up_fused, out: pre_gate_proj,
                                    N_half: MOE_INT, numSlots: NS)
            })
            stages.append(runStage("moe_q51_v6", layer: L) { cb in
                encMoeGemvQ51(cb, pre_gate_proj, lw.moeDown, pre_moe_down_out,
                              Din: MOE_INT, Dout: HIDDEN, numActive: E_EXP, useV6: true,
                              slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
            })
            stages.append(runStage("moe_q51_v8", layer: L) { cb in
                encMoeGemvQ51(cb, pre_gate_proj, lw.moeDown, pre_moe_down_out,
                              Din: MOE_INT, Dout: HIDDEN, numActive: E_EXP, useV8: true,
                              slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
            })
            stages.append(runStage("moe_q51_v9", layer: L) { cb in
                encMoeGemvQ51(cb, pre_gate_proj, lw.moeDown, pre_moe_down_out,
                              Din: MOE_INT, Dout: HIDDEN, numActive: E_EXP, useV9: true,
                              slotTokenBuf: pre_slot_token, groupStartBuf: pre_group_start)
            })
            stages.append(runStage("moe_tail", layer: L) { cb in
                encMoeCombineWriteInto(cb, moeOut: pre_moe_down_out, batchSlots: pre_batch_slots,
                                        gateW: pre_gate_w, outBuf: pre_moe_sum, numVecs: N)
                encRMSNormG(cb, x: pre_moe_sum, gammaBuf: lw.postFfn2Norm, out: pre_moe_sum,
                            D: HIDDEN, numVecs: N)
            })

            stages.append(runStage("resid", layer: L) { cb in
                encBufferCopy(cb, src: pre_mlp_out, dst: pre_ffn_combined, bytes: N * HIDDEN * 2)
                encAddInplace(cb, dst: pre_ffn_combined, src: pre_moe_sum, N: HIDDEN, numVecs: N)
                encRmsNormAddScale(cb, x: pre_ffn_combined, gammaBuf: lw.postFfnNorm,
                                   residual: pre_hidden, scalar: lw.layerOutputScale,
                                   out: pre_hidden, N: HIDDEN, numVecs: N)
            })
        }

        stages.append(runStage("unembed_v5", layer: -1) { cb in
            encRMSNormG(cb, x: pre_hidden, gammaBuf: w.outputNorm, out: pre_hidden_norm,
                        D: HIDDEN, numVecs: N)
            encGemvV5(cb, pre_hidden_norm, w.unembedW, pre_logits,
                      Din: HIDDEN, Dout: VOCAB, numVecs: N)
            encSoftcapInto(cb, buf: pre_logits, N: VOCAB, numVecs: N, cap: 30.0)
        })
        stages.append(runStage("unembed_v4p", layer: -1) { cb in
            encGemvV4P(cb, pre_hidden_norm, w.unembedW, pre_logits,
                       Din: HIDDEN, Dout: VOCAB, numVecs: N)
            encSoftcapInto(cb, buf: pre_logits, N: VOCAB, numVecs: N, cap: 30.0)
        })
        stages.append(runStage("unembed_fast", layer: -1) { cb in
            // gather B last-rows + RMSNorm + V4Softcap → logits[B, VOCAB]
            // matches encodePrefillTileInto's fast-path unembed branch
            let blit = cb.makeBlitCommandEncoder()!
            for slot in 0..<B {
                let srcOff = (slot * qLen + (qLen - 1)) * HIDDEN * 2
                let dstOff = slot * HIDDEN * 2
                blit.copy(from: pre_hidden, sourceOffset: srcOff,
                          to: hidden, destinationOffset: dstOff, size: HIDDEN * 2)
            }
            blit.endEncoding()
            encRMSNormG(cb, x: hidden, gammaBuf: w.outputNorm, out: hidden_norm,
                        D: HIDDEN, numVecs: B)
            encGemvV4Softcap(cb, hidden_norm, w.unembedW, logits,
                              Din: HIDDEN, Dout: VOCAB, cap: 30.0)
        })

        return stages
    }

    print("  warmup pass (discarded)..."); fflush(stdout)
    _ = onePass()

    // Aggregate over reps.
    var aggGpu: [String: Double] = [:]
    var aggCpu: [String: Double] = [:]
    var aggCount: [String: Int] = [:]
    var fullPassGpu: [Double] = []   // sum of stages per pass
    var fullPassCpu: [Double] = []
    // Layer-by-layer sample for the slide/full split (collected from rep 0).
    var perLayerGpu: [String: [Double]] = [:]
    var fullLayerIndices: [Int] = []
    var slideLayerIndices: [Int] = []

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
    // Identify which layer indices are isFull (collected once).
    for L in 0..<NUM_LAYERS {
        if w.layers[L].isFull { fullLayerIndices.append(L) } else { slideLayerIndices.append(L) }
    }

    func pad(_ s: String, _ width: Int) -> String {
        return s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }
    func padf(_ d: Double, _ width: Int = 12, _ p: Int = 2) -> String {
        let s = String(format: "%.\(p)f", d)
        return s.count >= width ? s : String(repeating: " ", count: width - s.count) + s
    }

    let stageOrder = ["embed",
                       "qkv", "qkn_rope", "kv_attn", "oproj_norm",
                       "shrd_ffn", "router",
                       "moe_q4k_v6", "moe_q4k_v8", "moe_q4k_v9", "moe_q4k_v10_fused",
                       "moe_gelu",
                       "moe_q51_v6", "moe_q51_v8", "moe_q51_v9",
                       "moe_tail",
                       "resid",
                       "unembed_v5", "unembed_v4p", "unembed_fast"]
    print("")
    print("=== aggregate per-stage GPU time (across \(reps) reps, summed across layers/passes) ===")
    print("  \(pad("stage", 14))\(pad("count", 8))\(pad("total_gpu_ms", 14))\(pad("avg_per_call_ms", 18))\(pad("per_pass_ms", 14))")
    var grandGpu = 0.0
    for name in stageOrder {
        guard let g = aggGpu[name], let c = aggCount[name] else { continue }
        let perPass = g / Double(reps)
        let perCall = g / Double(c)
        grandGpu += g
        print("  \(pad(name, 14))\(pad(String(c), 8))\(padf(g, 14))\(padf(perCall, 18, 3))\(padf(perPass, 14))")
    }
    let grandPerPass = grandGpu / Double(reps)
    print("  \(pad("TOTAL", 14))\(pad("", 8))\(padf(grandGpu, 14))\(padf(0, 18))\(padf(grandPerPass, 14))")
    print("")
    let avgWall = fullPassGpu.reduce(0,+) / Double(reps)
    let avgCpuWall = fullPassCpu.reduce(0,+) / Double(reps)
    print(String(format: "  per-pass GPU wall (sum of stage GPU times): %.2f ms", avgWall))
    print(String(format: "  per-pass CPU wall (sum of stage CPU times): %.2f ms (CB-boundary overhead)", avgCpuWall))

    // Slide vs full split for the per-layer attention stages.
    print("")
    print("=== slide (\(slideLayerIndices.count) layers) vs full (\(fullLayerIndices.count) layers) for attention-bearing stages (rep 0) ===")
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
        print("  \(pad(name, 14))\(padf(slideAvg, 16, 3))\(padf(fullAvg, 16, 3))\(padf(ratio, 10, 2))×")
    }

    fflush(stdout)
}
