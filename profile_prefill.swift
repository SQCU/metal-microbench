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
    // LM_PROFILE_ACTIVEB=1 (default 0): set numActiveRows=qLen (single-slot
    // prefill geometry) and numActiveSlots=1 across every dispatch.
    let activeBProfile = ProcessInfo.processInfo.environment["LM_PROFILE_ACTIVEB"] == "1"
    let activeSlots = activeBProfile ? 1 : B
    print("  qLen=\(qLen)  B=\(B)  numActiveSlots=\(activeSlots)  N=numActive*qLen=\(activeSlots*qLen)  reps=\(reps) (+1 warmup)")

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
    precomputeFlexPrefillMasks(qLen: qLen, positionStart: 0, numActiveSlots: activeSlots)

    let N = activeSlots * qLen
    let NS = activeSlots * qLen * TOPK

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
            let kArgBuf = w.K_chunks_argbuf[L]
            let vArgBuf = w.V_chunks_argbuf[L]
            let kChunks = w.K_chunks[L]
            let vChunks = w.V_chunks[L]

            // QKV: RMSNorm + 3 simdgroup matmuls (Q, K, V) on v6-swizzled Q8_0.
            stages.append(runStage("qkv", layer: L) { cb in
                let Wv = lw.attnV ?? lw.attnK
                encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.attnNorm, out: pre_hidden_norm,
                             D: HIDDEN, numVecs: N)
                let WvFmt = (lw.attnV != nil) ? lw.attnVFormat : lw.attnKFormat
                encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.attnQ, format: lw.attnQFormat, Y: q_out,
                                   Din: HIDDEN, Dout: H * HD, numVecs: N)
                encDenseMmPrefill(cb, x: pre_hidden_norm, W: lw.attnK, format: lw.attnKFormat, Y: k_out,
                                   Din: HIDDEN, Dout: KV_H * HD, numVecs: N)
                encDenseMmPrefill(cb, x: pre_hidden_norm, W: Wv, format: WvFmt, Y: v_out,
                                   Din: HIDDEN, Dout: KV_H * HD, numVecs: N)
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

            let activeChunkIdxs = activeKVChunkIdxsFromKLen(
                blockTable: block_table,
                kLenSlide: pre_k_len_slide, kLenFull: pre_k_len_full,
                activeB: B, kvChunkPages: w.kvChunkPages)
            stages.append(runStage("kv_attn", layer: L) { cb in
                let pg = PAGE
                encKVWriteMulti(cb, K: k_out, V: v_out,
                                kArgBuf: kArgBuf, vArgBuf: vArgBuf,
                                kChunks: kChunks, vChunks: vChunks, chunkPages: w.kvChunkPages,
                                activeChunkIdxs: activeChunkIdxs,
                                q_positions: pre_q_positions,
                                H: KV_H, D: HD, page: pg, qLen: qLen,
                                numActiveSlots: activeSlots)
                let klBuf = isFull ? pre_k_len_full : pre_k_len_slide
                if isFull {
                    encFlexAttnFullPrefill(cb, Q: q_out, O: pre_attn_out,
                                            kArgBuf: kArgBuf, vArgBuf: vArgBuf,
                                            kChunks: kChunks, vChunks: vChunks, chunkPages: w.kvChunkPages,
                                            activeChunkIdxs: activeChunkIdxs,
                                            kLenBuf: klBuf, qPositions: pre_q_positions,
                                            H_Q: H, H_KV: KV_H, D: HD, qLen: qLen,
                                            numActiveSlots: activeSlots)
                } else {
                    encFlexAttnSlidePrefill(cb, Q: q_out, O: pre_attn_out,
                                             kArgBuf: kArgBuf, vArgBuf: vArgBuf,
                                             kChunks: kChunks, vChunks: vChunks, chunkPages: w.kvChunkPages,
                                             activeChunkIdxs: activeChunkIdxs,
                                             kLenBuf: klBuf, qPositions: pre_q_positions,
                                             H_Q: H, H_KV: KV_H, D: HD, qLen: qLen,
                                             numActiveSlots: activeSlots)
                }
            })

            // o_proj (simdgroup matmul, format-aware) + post-attn RMSNormAdd.
            stages.append(runStage("oproj_norm", layer: L) { cb in
                encDenseMmPrefill(cb, x: pre_attn_out, W: lw.attnOut, format: lw.attnOutFormat,
                                   Y: pre_mlp_out, Din: H * HD, Dout: HIDDEN, numVecs: N)
                encRmsNormAdd(cb, x: pre_mlp_out, gammaBuf: lw.postAttnNorm,
                              residual: pre_hidden, out: pre_hidden, N: HIDDEN, numVecs: N)
            })

            // Shared FFN: RMSNorm + 2 matmuls (gate, up) + gelu_mul_inplace + ffn_down + post-norm.
            stages.append(runStage("shrd_ffn", layer: L) { cb in
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

            // MoE: pre-FFN2 RMSNorm + format-aware gate_up matmul (slot-flat, broadcast X).
            stages.append(runStage("moe_up", layer: L) { cb in
                encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.preFfn2Norm, out: pre_hidden_norm,
                            D: HIDDEN, numVecs: N)
                encMoeUpMmPrefill(cb, x: pre_hidden_norm, W: lw.moeGateUp, format: lw.moeGateUpFormat,
                                   Y: pre_gate_up_fused,
                                   slotTokenBuf: pre_slot_token, activeExpBuf: active_exp,
                                   groupStartBuf: pre_group_start,
                                   Din: HIDDEN, Dout: MOE_FUSED_DOUT, numSlots: NS, E: E_EXP,
                                   dispatchArgsBuf: pre_moe_dispatch_args)
            })
            stages.append(runStage("moe_gelu", layer: L) { cb in
                encMoeGeluMulFused(cb, fused: pre_gate_up_fused, out: pre_gate_proj,
                                    N_half: MOE_INT, numSlots: NS)
            })
            // Format-aware MoE down matmul (slot-flat, per-slot X).
            stages.append(runStage("moe_down", layer: L) { cb in
                encMoeDownMmPrefill(cb, x: pre_gate_proj, W: lw.moeDown, format: lw.moeDownFormat,
                                     Y: pre_moe_down_out,
                                     activeExpBuf: active_exp, groupStartBuf: pre_group_start,
                                     Din: MOE_INT, Dout: HIDDEN, numSlots: NS, E: E_EXP,
                                     dispatchArgsBuf: pre_moe_dispatch_args)
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

        stages.append(runStage("unembed_fast", layer: -1) { cb in
            // gather activeSlots last-rows + RMSNorm + V4Softcap → logits[activeSlots, VOCAB]
            // matches encodePrefillTileInto's fast-path unembed branch
            let blit = cb.makeBlitCommandEncoder()!
            for slot in 0..<activeSlots {
                let srcOff = (slot * qLen + (qLen - 1)) * HIDDEN * 2
                let dstOff = slot * HIDDEN * 2
                blit.copy(from: pre_hidden, sourceOffset: srcOff,
                          to: hidden, destinationOffset: dstOff, size: HIDDEN * 2)
            }
            blit.endEncoding()
            encRMSNormG(cb, x: hidden, gammaBuf: w.outputNorm, out: hidden_norm,
                        D: HIDDEN, numVecs: activeSlots)
            encGemvV4Softcap(cb, hidden_norm, w.unembedW, logits,
                              Din: HIDDEN, Dout: VOCAB, cap: 30.0, activeB: activeSlots)
        })

        return stages
    }

    print("  warmup pass (discarded)..."); fflush(stdout)
    _ = onePass()

    // Correctness fingerprint: argmax + checksum of slot-0 logits after the
    // forward. Used to prove kernel changes (e.g. MoE-down re-tile) are
    // bit-safe — argmax must match, sum must agree to fp tolerance.
    do {
        let lp = logits.contents().bindMemory(to: Float16.self, capacity: activeSlots * VOCAB)
        var best = -Float.infinity; var bestIdx = -1; var sum = 0.0
        for v in 0..<VOCAB {
            let x = Float(lp[v]); sum += Double(x)
            if x > best { best = x; bestIdx = v }
        }
        print(String(format: "  [fingerprint] slot0 argmax=%d  maxLogit=%.4f  sumLogits=%.3f",
                     bestIdx, best, sum)); fflush(stdout)
    }

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
                       "qkv", "qkn_rope", "kv_attn",
                       "oproj_norm", "shrd_ffn",
                       "router",
                       "moe_up", "moe_gelu", "moe_down", "moe_tail",
                       "resid",
                       "unembed_fast"]
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

// =====================================================================
// runLmPrefillBandwidthSweep
//
// 2026-05-23: characterizes per-stage prefill cost across the cross
// product of (qLen, activeSlots). For each cell, builds + commits a
// single CB that runs the FULL prefill forward (identical encode path
// to encodePrefillTileInto, just split per stage for GPU timing), then
// reports per-pass GPU wall + dominant-stage breakdown + derived tok/s.
//
// Aggregated into /tmp/prefill_bandwidth_sweep.csv (or LM_PROFILE_CSV
// override). Markdown summary printed at end.
//
// Invocation:
//   LM_PROFILE_PREFILL_SWEEP=1 GGUF_PATH=/path/to/Q8.gguf ./forward_graph
// Optional env:
//   LM_SWEEP_QLENS=64,128,256          (default = 64,128,256; >MAX_Q_LEN skipped)
//   LM_SWEEP_ACTIVE_SLOTS=1,2,3,4,5,6,7,8 (default = full B-range)
//   LM_SWEEP_REPS=3                    (default 3; +1 untimed warmup PER CELL)
//   LM_PROFILE_CSV=/tmp/prefill_bandwidth_sweep.csv
//
// Loads weights ONCE then iterates the grid so we don't re-pay the
// ~30 s cold load per cell.
// =====================================================================
private struct PrefillCellResult {
    let qLen: Int
    let activeSlots: Int
    let perPassMs: Double          // sum of stage gpuMs averaged over reps
    let stageMs: [String: Double]  // per-pass ms by stage name (averaged)
}

func runLmPrefillBandwidthSweep(ggufPath: String) {
    print("\n=== prefill bandwidth sweep (qLen × active_slots) ===")
    let qLensEnv = ProcessInfo.processInfo.environment["LM_SWEEP_QLENS"]
    let activeSlotsEnv = ProcessInfo.processInfo.environment["LM_SWEEP_ACTIVE_SLOTS"]
    let repsEnv = ProcessInfo.processInfo.environment["LM_SWEEP_REPS"].flatMap { Int($0) }
    let csvPath = ProcessInfo.processInfo.environment["LM_PROFILE_CSV"]
        ?? "/tmp/prefill_bandwidth_sweep.csv"
    let reps = max(1, repsEnv ?? 3)

    var rawQLens = (qLensEnv ?? "64,128,256").split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    var skippedQLens: [Int] = []
    rawQLens = rawQLens.filter { q in
        if q > MAX_Q_LEN { skippedQLens.append(q); return false }
        if q < 1 { return false }
        return true
    }
    let qLens = rawQLens

    let activeSlotsList = (activeSlotsEnv ?? "1,2,3,4,5,6,7,8").split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        .filter { $0 >= 1 && $0 <= B }

    print("  qLens=\(qLens) (skipped over MAX_Q_LEN=\(MAX_Q_LEN): \(skippedQLens))")
    print("  active_slots=\(activeSlotsList)  reps=\(reps) (+1 warmup per cell)")
    print("  csv → \(csvPath)")
    fflush(stdout)

    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }

    // Per-cell encode helper. Returns per-stage list summed across layers
    // for one pass. Stage names match the profile-mode output so reads can
    // cross-reference.
    func encodeOnePass(qLen: Int, activeSlots: Int) -> [String: Double] {
        let N = activeSlots * qLen
        let NS = activeSlots * qLen * TOPK

        // Re-init token/position state for this qLen. Only the active
        // slots' rows actually matter; silenced rows aren't touched by
        // matmuls but we set them anyway for cleanliness.
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
        precomputeFlexPrefillMasks(qLen: qLen, positionStart: 0,
                                    numActiveSlots: activeSlots)

        var stages: [String: Double] = [:]

        func add(_ name: String, _ stage: PrefillStage) {
            stages[name, default: 0] += stage.gpuMs
        }

        add("embed", runStage("embed", layer: -1) { cb in
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
            let kArgBuf = w.K_chunks_argbuf[L]
            let vArgBuf = w.V_chunks_argbuf[L]
            let kChunks = w.K_chunks[L]
            let vChunks = w.V_chunks[L]

            add("qkv", runStage("qkv", layer: L) { cb in
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
            })

            add("qkn_rope", runStage("qkn_rope", layer: L) { cb in
                encRMSNormG(cb, x: q_out, gammaBuf: lw.attnQNorm, out: q_out, D: HD, numVecs: N * H)
                encRopeMulti(cb, q_out, q_positions: pre_q_positions, H: H, D: HD,
                             rotary: rotary, theta: theta, qLen: qLen)
                encRMSNormG(cb, x: k_out, gammaBuf: lw.attnKNorm, out: k_out, D: HD, numVecs: N * KV_H)
                encRopeMulti(cb, k_out, q_positions: pre_q_positions, H: KV_H, D: HD,
                             rotary: rotary, theta: theta, qLen: qLen)
                encRMSNormNoScale(cb, x: v_out, out: v_out, D: HD, numVecs: N * KV_H)
            })

            let activeChunkIdxs = activeKVChunkIdxsFromKLen(
                blockTable: block_table,
                kLenSlide: pre_k_len_slide, kLenFull: pre_k_len_full,
                activeB: B, kvChunkPages: w.kvChunkPages)
            add("kv_attn", runStage("kv_attn", layer: L) { cb in
                let pg = PAGE
                encKVWriteMulti(cb, K: k_out, V: v_out,
                                kArgBuf: kArgBuf, vArgBuf: vArgBuf,
                                kChunks: kChunks, vChunks: vChunks, chunkPages: w.kvChunkPages,
                                activeChunkIdxs: activeChunkIdxs,
                                q_positions: pre_q_positions,
                                H: KV_H, D: HD, page: pg, qLen: qLen,
                                numActiveSlots: activeSlots)
                let klBuf = isFull ? pre_k_len_full : pre_k_len_slide
                if isFull {
                    encFlexAttnFullPrefill(cb, Q: q_out, O: pre_attn_out,
                                            kArgBuf: kArgBuf, vArgBuf: vArgBuf,
                                            kChunks: kChunks, vChunks: vChunks, chunkPages: w.kvChunkPages,
                                            activeChunkIdxs: activeChunkIdxs,
                                            kLenBuf: klBuf, qPositions: pre_q_positions,
                                            H_Q: H, H_KV: KV_H, D: HD, qLen: qLen,
                                            numActiveSlots: activeSlots)
                } else {
                    encFlexAttnSlidePrefill(cb, Q: q_out, O: pre_attn_out,
                                             kArgBuf: kArgBuf, vArgBuf: vArgBuf,
                                             kChunks: kChunks, vChunks: vChunks, chunkPages: w.kvChunkPages,
                                             activeChunkIdxs: activeChunkIdxs,
                                             kLenBuf: klBuf, qPositions: pre_q_positions,
                                             H_Q: H, H_KV: KV_H, D: HD, qLen: qLen,
                                             numActiveSlots: activeSlots)
                }
            })

            add("oproj_norm", runStage("oproj_norm", layer: L) { cb in
                encDenseMmPrefill(cb, x: pre_attn_out, W: lw.attnOut, format: lw.attnOutFormat,
                                   Y: pre_mlp_out, Din: H * HD, Dout: HIDDEN, numVecs: N)
                encRmsNormAdd(cb, x: pre_mlp_out, gammaBuf: lw.postAttnNorm,
                              residual: pre_hidden, out: pre_hidden, N: HIDDEN, numVecs: N)
            })

            add("shrd_ffn", runStage("shrd_ffn", layer: L) { cb in
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
            })

            add("router", runStage("router", layer: L) { cb in
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

            add("moe_up", runStage("moe_up", layer: L) { cb in
                encRMSNormG(cb, x: pre_hidden, gammaBuf: lw.preFfn2Norm, out: pre_hidden_norm,
                            D: HIDDEN, numVecs: N)
                encMoeUpMmPrefill(cb, x: pre_hidden_norm, W: lw.moeGateUp, format: lw.moeGateUpFormat,
                                   Y: pre_gate_up_fused,
                                   slotTokenBuf: pre_slot_token, activeExpBuf: active_exp,
                                   groupStartBuf: pre_group_start,
                                   Din: HIDDEN, Dout: MOE_FUSED_DOUT, numSlots: NS, E: E_EXP)
            })
            add("moe_gelu", runStage("moe_gelu", layer: L) { cb in
                encMoeGeluMulFused(cb, fused: pre_gate_up_fused, out: pre_gate_proj,
                                    N_half: MOE_INT, numSlots: NS)
            })
            add("moe_down", runStage("moe_down", layer: L) { cb in
                encMoeDownMmPrefill(cb, x: pre_gate_proj, W: lw.moeDown, format: lw.moeDownFormat,
                                     Y: pre_moe_down_out,
                                     activeExpBuf: active_exp, groupStartBuf: pre_group_start,
                                     Din: MOE_INT, Dout: HIDDEN, numSlots: NS, E: E_EXP)
            })
            add("moe_tail", runStage("moe_tail", layer: L) { cb in
                encMoeCombineWriteInto(cb, moeOut: pre_moe_down_out, batchSlots: pre_batch_slots,
                                        gateW: pre_gate_w, outBuf: pre_moe_sum, numVecs: N)
                encRMSNormG(cb, x: pre_moe_sum, gammaBuf: lw.postFfn2Norm, out: pre_moe_sum,
                            D: HIDDEN, numVecs: N)
            })

            add("resid", runStage("resid", layer: L) { cb in
                encBufferCopy(cb, src: pre_mlp_out, dst: pre_ffn_combined, bytes: N * HIDDEN * 2)
                encAddInplace(cb, dst: pre_ffn_combined, src: pre_moe_sum, N: HIDDEN, numVecs: N)
                encRmsNormAddScale(cb, x: pre_ffn_combined, gammaBuf: lw.postFfnNorm,
                                   residual: pre_hidden, scalar: lw.layerOutputScale,
                                   out: pre_hidden, N: HIDDEN, numVecs: N)
            })
        }

        add("unembed_fast", runStage("unembed_fast", layer: -1) { cb in
            let blit = cb.makeBlitCommandEncoder()!
            for slot in 0..<activeSlots {
                let srcOff = (slot * qLen + (qLen - 1)) * HIDDEN * 2
                let dstOff = slot * HIDDEN * 2
                blit.copy(from: pre_hidden, sourceOffset: srcOff,
                          to: hidden, destinationOffset: dstOff, size: HIDDEN * 2)
            }
            blit.endEncoding()
            encRMSNormG(cb, x: hidden, gammaBuf: w.outputNorm, out: hidden_norm,
                        D: HIDDEN, numVecs: activeSlots)
            encGemvV4Softcap(cb, hidden_norm, w.unembedW, logits,
                              Din: HIDDEN, Dout: VOCAB, cap: 30.0, activeB: activeSlots)
        })

        return stages
    }

    // Run grid. For each (qLen, activeSlots) we do one warmup + reps timed.
    var results: [PrefillCellResult] = []
    for qLen in qLens {
        for activeSlots in activeSlotsList {
            print("  cell qLen=\(qLen) active_slots=\(activeSlots)  warmup..."); fflush(stdout)
            _ = encodeOnePass(qLen: qLen, activeSlots: activeSlots)
            var sumStages: [String: Double] = [:]
            var sumPerPass = 0.0
            for r in 0..<reps {
                let stages = encodeOnePass(qLen: qLen, activeSlots: activeSlots)
                var pass = 0.0
                for (k, v) in stages {
                    sumStages[k, default: 0] += v
                    pass += v
                }
                sumPerPass += pass
                if r == 0 { _ = r }
            }
            let avgStages = sumStages.mapValues { $0 / Double(reps) }
            let avgPerPass = sumPerPass / Double(reps)
            let toksPerSec = Double(activeSlots * qLen) / (avgPerPass / 1000.0)
            print(String(format: "    per-pass=%.2f ms  tok/s=%.0f  qkv=%.2f moe_up=%.2f moe_down=%.2f kv_attn=%.2f shrd_ffn=%.2f",
                          avgPerPass, toksPerSec,
                          avgStages["qkv"] ?? 0, avgStages["moe_up"] ?? 0,
                          avgStages["moe_down"] ?? 0, avgStages["kv_attn"] ?? 0,
                          avgStages["shrd_ffn"] ?? 0))
            fflush(stdout)
            results.append(PrefillCellResult(
                qLen: qLen, activeSlots: activeSlots,
                perPassMs: avgPerPass, stageMs: avgStages))
        }
    }

    // -------- CSV emit --------
    let stageOrderCsv = ["embed", "qkv", "qkn_rope", "kv_attn",
                          "oproj_norm", "shrd_ffn", "router",
                          "moe_up", "moe_gelu", "moe_down", "moe_tail",
                          "resid", "unembed_fast"]
    var csv = "qLen,active_slots,tokens,per_pass_ms,tokens_per_sec"
    for s in stageOrderCsv { csv += ",ms_\(s)" }
    csv += "\n"
    for r in results {
        let toks = r.activeSlots * r.qLen
        let tps = Double(toks) / (r.perPassMs / 1000.0)
        csv += "\(r.qLen),\(r.activeSlots),\(toks),"
        csv += String(format: "%.3f,%.2f", r.perPassMs, tps)
        for s in stageOrderCsv {
            csv += String(format: ",%.4f", r.stageMs[s] ?? 0)
        }
        csv += "\n"
    }
    do {
        try csv.write(toFile: csvPath, atomically: true, encoding: .utf8)
        print("\n  CSV written: \(csvPath) (\(results.count) cells)")
    } catch {
        print("\n  CSV write failed: \(error)")
    }

    // -------- Headline tok/s table --------
    print("\n=== headline tok/s (qLen × active_slots) ===")
    var header = "  qLen \\ AS"
    for a in activeSlotsList { header += String(format: "%8d", a) }
    print(header)
    for q in qLens {
        var row = String(format: "  qLen=%-4d", q)
        for a in activeSlotsList {
            if let r = results.first(where: { $0.qLen == q && $0.activeSlots == a }) {
                let tps = Double(a * q) / (r.perPassMs / 1000.0)
                row += String(format: "%8.0f", tps)
            } else {
                row += "      —"
            }
        }
        print(row)
    }

    // -------- Per-pass ms table --------
    print("\n=== per-pass GPU wall ms (qLen × active_slots) ===")
    print(header)
    for q in qLens {
        var row = String(format: "  qLen=%-4d", q)
        for a in activeSlotsList {
            if let r = results.first(where: { $0.qLen == q && $0.activeSlots == a }) {
                row += String(format: "%8.2f", r.perPassMs)
            } else {
                row += "      —"
            }
        }
        print(row)
    }

    fflush(stdout)
}
