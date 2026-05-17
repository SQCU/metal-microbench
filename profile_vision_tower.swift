// profile_vision_tower — per-stage GPU timing of the vision tower's
// forward pass. Splits the 27-layer tower into separately-committed CBs
// (one per stage) so MTLCommandBuffer.gpuStartTime/gpuEndTime gives
// honest GPU wall per stage.
//
// Output:
//   - patch_embed + pos_embed
//   - per-layer attention sub-block
//   - per-layer FFN sub-block
//   - tail (pool + std_norm + pre_proj_norm + embed_vision_proj)
//
// Each layer is split into TWO CBs (attention | FFN) so we can attribute
// time within a layer. The 30 kernel dispatches per layer all stay in
// one CB per sub-block — we don't pay extra synchronization, just one
// CB-boundary-per-sub-block instead of one-per-image. Total: 1 (patch)
// + 27*2 (encoder) + 1 (tail) = 56 CBs. ~5.6ms of CB-boundary overhead
// vs ~11s baseline = <0.1% — small enough to ignore for profile purposes.

import Metal
import Foundation

struct StageTime {
    let name: String
    let gpuMs: Double
    let cpuMs: Double
}

private func stage(_ name: String, _ queue: MTLCommandQueue,
                    _ encode: (MTLCommandBuffer) -> Void) -> StageTime {
    let cb = queue.makeCommandBuffer()!
    encode(cb)
    let cpuT0 = Date()
    cb.commit()
    cb.waitUntilCompleted()
    let cpuT1 = Date()
    let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
    let cpuMs = cpuT1.timeIntervalSince(cpuT0) * 1000.0
    return StageTime(name: name, gpuMs: gpuMs, cpuMs: cpuMs)
}

func profileVisionTower(image: PatchBatch, weights: VisionWeights,
                         device: MTLDevice, queue: MTLCommandQueue) -> [StageTime] {
    let B = 1
    let N = image.maxPatches
    let h = weights.hidden, H = weights.numHeads, HD = weights.headDim
    let interm = weights.interm
    let BN = B * N
    var stages: [StageTime] = []

    // Setup: positions, padding mask, patches concat (one-time, not in profile)
    let posYBuf = device.makeBuffer(length: BN * 4, options: .storageModeShared)!
    let posXBuf = device.makeBuffer(length: BN * 4, options: .storageModeShared)!
    let paddingMaskBuf = device.makeBuffer(length: BN, options: .storageModeShared)!
    let pyp = posYBuf.contents().assumingMemoryBound(to: UInt32.self)
    let pxp = posXBuf.contents().assumingMemoryBound(to: UInt32.self)
    let pmp = paddingMaskBuf.contents().assumingMemoryBound(to: UInt8.self)
    for i in 0..<N {
        pyp[i] = UInt32(max(image.positions[i].0, 0))
        pxp[i] = UInt32(max(image.positions[i].1, 0))
        pmp[i] = (i < image.numRealPatches) ? 0 : 1
    }
    let patchesBuf = device.makeBuffer(length: BN * 768 * 2, options: .storageModeShared)!
    memcpy(patchesBuf.contents(), image.patches.contents(), N * 768 * 2)

    let x = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let tmp = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let qBuf = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let kBuf = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let vBuf = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let attnOut = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let gateAct = device.makeBuffer(length: BN * interm * 4, options: .storageModeShared)!
    let mlpOutB = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let postNormFp32 = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!

    // Stage 1: patch embed + pos embed
    stages.append(stage("patch+pos_embed", queue) { cb in
        encGemvFp16InFp32Out(cb, x: patchesBuf, W: weights.patchEmbedW, out: x,
                              B: BN, Din: 768, Dout: h)
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(visionPosEmbedFp32PSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weights.posEmbedTable, offset: weights.posMax * h * 2, index: 1)
        enc.setBuffer(weights.posEmbedTable, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        var nx = UInt32(image.gridW), hh = UInt32(h)
        enc.setBytes(&nx, length: 4, index: 4)
        enc.setBytes(&hh, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    })

    // Stage 2..28: per-layer attention then FFN
    let qkScale: Float = 1.0
    for L in 0..<weights.numLayers {
        let lw = weights.layers[L]
        stages.append(stage("layer_\(L)_attn", queue) { cb in
            encRMSNormGFp32(cb, x: x, gammaBuf: lw.inputNorm, out: tmp, D: h, numVecs: BN)
            encQuantizeFp32ToBf16(cb, x: tmp, N: h, numVecs: BN)
            encGemvFp32V5(cb, x: tmp, W: lw.qProj, out: qBuf, B: BN, Din: h, Dout: h, quantOut: true)
            encGemvFp32V5(cb, x: tmp, W: lw.kProj, out: kBuf, B: BN, Din: h, Dout: h, quantOut: true)
            encGemvFp32V5(cb, x: tmp, W: lw.vProj, out: vBuf, B: BN, Din: h, Dout: h, quantOut: true)
            encRMSNormGFp32(cb, x: qBuf, gammaBuf: lw.qNorm, out: qBuf, D: HD, numVecs: BN * H)
            encQuantizeFp32ToBf16(cb, x: qBuf, N: h, numVecs: BN)
            encRMSNormGFp32(cb, x: kBuf, gammaBuf: lw.kNorm, out: kBuf, D: HD, numVecs: BN * H)
            encQuantizeFp32ToBf16(cb, x: kBuf, N: h, numVecs: BN)
            encRMSNormNoScaleFp32(cb, x: vBuf, out: vBuf, D: HD, numVecs: BN * H)
            encQuantizeFp32ToBf16(cb, x: vBuf, N: h, numVecs: BN)
            encVision2DRopeFp32(cb, x: qBuf, posX: posXBuf, posY: posYBuf,
                                 N: BN, H: H, HD: HD, theta: 100.0)
            encQuantizeFp32ToBf16(cb, x: qBuf, N: h, numVecs: BN)
            encVision2DRopeFp32(cb, x: kBuf, posX: posXBuf, posY: posYBuf,
                                 N: BN, H: H, HD: HD, theta: 100.0)
            encQuantizeFp32ToBf16(cb, x: kBuf, N: h, numVecs: BN)
            encVisionAttnFlashFp32(cb, Q: qBuf, K: kBuf, V: vBuf, O: attnOut,
                                     N: N, H: H, HD: HD, qkScale: qkScale,
                                     paddingMask: paddingMaskBuf, B: B)
            encQuantizeFp32ToBf16(cb, x: attnOut, N: h, numVecs: BN)
            encGemvFp32V5(cb, x: attnOut, W: lw.oProj, out: tmp, B: BN, Din: h, Dout: h, quantOut: true)
            encRMSNormGFp32(cb, x: tmp, gammaBuf: lw.postAttnNorm, out: postNormFp32, D: h, numVecs: BN)
            encQuantizeFp32ToBf16(cb, x: postNormFp32, N: h, numVecs: BN)
            encAddInplaceFp32Fp32(cb, dst: x, src: postNormFp32, N: h, numVecs: BN)
            encQuantizeFp32ToBf16(cb, x: x, N: h, numVecs: BN)
        })
        stages.append(stage("layer_\(L)_ffn", queue) { cb in
            encRMSNormGFp32(cb, x: x, gammaBuf: lw.preFfnNorm, out: tmp, D: h, numVecs: BN)
            encQuantizeFp32ToBf16(cb, x: tmp, N: h, numVecs: BN)
            encVisionFfnGateUpGeluFp32(cb, x: tmp, Wgate: lw.gateProj, Wup: lw.upProj,
                                         out: gateAct, B: BN, Din: h, Dout: interm)
            encGemvFp32V5(cb, x: gateAct, W: lw.downProj, out: mlpOutB, B: BN, Din: interm, Dout: h, quantOut: true)
            encRMSNormGFp32(cb, x: mlpOutB, gammaBuf: lw.postFfnNorm, out: postNormFp32, D: h, numVecs: BN)
            encQuantizeFp32ToBf16(cb, x: postNormFp32, N: h, numVecs: BN)
            encAddInplaceFp32Fp32(cb, dst: x, src: postNormFp32, N: h, numVecs: BN)
            encQuantizeFp32ToBf16(cb, x: x, N: h, numVecs: BN)
        })
    }

    // Stage 29: tail (pool + std_norm + pre_proj_norm + embed_vision_proj)
    stages.append(stage("tail_pool+norm+proj", queue) { cb in
        let outH = image.gridH / 3, outW = image.gridW / 3
        let nPooled = outH * outW
        let pooled = device.makeBuffer(length: nPooled * h * 4, options: .storageModeShared)!
        let stdNormed = device.makeBuffer(length: nPooled * h * 4, options: .storageModeShared)!
        let softTokens = device.makeBuffer(length: nPooled * weights.textHidden * 4, options: .storageModeShared)!
        let sqrtH = Float(h).squareRoot()

        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(visionPool2DFp32InFp32OutPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(pooled, offset: 0, index: 1)
        var gw = UInt32(image.gridW), ow = UInt32(outW), ks: UInt32 = 3, hh = UInt32(h)
        enc.setBytes(&gw, length: 4, index: 2); enc.setBytes(&ow, length: 4, index: 3)
        enc.setBytes(&ks, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: nPooled, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()

        encVisionScaledStdNormFp32(cb, x: pooled, bias: weights.stdBias, scale: weights.stdScale,
                                     out: stdNormed, D: h, numVecs: nPooled, globalScale: sqrtH)
        encRMSNormNoScaleFp32(cb, x: stdNormed, out: stdNormed, D: h, numVecs: nPooled)
        encGemvFp32V5(cb, x: stdNormed, W: weights.embedVisionProj, out: softTokens,
                        B: nPooled, Din: h, Dout: weights.textHidden)
    })

    return stages
}

// Entry point — load weights + tokenizer, decode an image, run profile,
// print summary. Compile/run with:
//   make libgemma_metal.dylib && \
//   swiftc -O <forward_graph_lib_srcs> ffi.swift ffi_batch.swift profile_vision_tower.swift \
//     profile_vision_tower_main.swift -o profile_vision_tower \
//     -framework Metal -framework Foundation
// (driver in profile_vision_tower_main.swift)

func runVisionProfileMain(imagePath: String, safetensorsPath: String) {
    // Use the GLOBAL device + queue from common.swift. PSOs registered
    // by kernels.swift/vision_tower.swift bind to the global device at
    // import time; making a local device here would mean the PSOs are
    // for a different device than our command buffers, which crashes.

    print("loading vision weights from \(safetensorsPath)..."); fflush(stdout)
    let t0 = Date()
    let weights: VisionWeights
    do {
        print("  opening safetensors..."); fflush(stdout)
        let st = try SafetensorsFile(safetensorsPath)
        print("  opened; calling loadVisionWeights..."); fflush(stdout)
        weights = try loadVisionWeights(st, device: device)
    } catch {
        print("vision weight load failed: \(error)"); exit(1)
    }
    print("  loaded in \(String(format: "%.2f", -t0.timeIntervalSinceNow))s"); fflush(stdout)

    print("preprocessing image \(imagePath)..."); fflush(stdout)
    let t1 = Date()
    let image: PatchBatch
    do {
        image = try gemma4ImagePreprocessFromPath(path: imagePath, device: device)
    } catch {
        print("preprocess failed: \(error)"); exit(1)
    }
    print("  resized to \(image.resizedW)×\(image.resizedH) → \(image.gridW)×\(image.gridH) grid (\(image.numRealPatches) real / \(image.maxPatches) max patches), in \(String(format: "%.2f", -t1.timeIntervalSinceNow))s"); fflush(stdout)

    // Warm pass — first run pays JIT/PSO compile + initial weight residency.
    print("warmup pass (discarded)..."); fflush(stdout)
    _ = profileVisionTower(image: image, weights: weights, device: device, queue: queue)
    print("  warmup done"); fflush(stdout)

    print("profiling pass..."); fflush(stdout)
    let stages = profileVisionTower(image: image, weights: weights, device: device, queue: queue)
    print("  profiling done, \(stages.count) stages collected"); fflush(stdout)

    // Aggregate.
    var attnGpu: Double = 0, attnCpu: Double = 0
    var ffnGpu: Double = 0, ffnCpu: Double = 0
    var headTail: [StageTime] = []
    for s in stages {
        if s.name.contains("_attn") {
            attnGpu += s.gpuMs; attnCpu += s.cpuMs
        } else if s.name.contains("_ffn") {
            ffnGpu += s.gpuMs; ffnCpu += s.cpuMs
        } else {
            headTail.append(s)
        }
    }
    let totalGpu = stages.reduce(0.0) { $0 + $1.gpuMs }
    let totalCpu = stages.reduce(0.0) { $0 + $1.cpuMs }

    func pad(_ s: String, _ w: Int) -> String {
        return s.count >= w ? s : s + String(repeating: " ", count: w - s.count)
    }
    func padf(_ d: Double, _ w: Int = 12, _ p: Int = 2) -> String {
        let s = String(format: "%.\(p)f", d)
        return s.count >= w ? s : String(repeating: " ", count: w - s.count) + s
    }

    print("")
    print("=== per-stage timings (ms) ===")
    print("  \(pad("stage", 24))\(pad("gpu_ms", 12))\(pad("cpu_ms", 12))")
    for s in headTail {
        print("  \(pad(s.name, 24))\(padf(s.gpuMs))\(padf(s.cpuMs))")
    }
    print("  \(pad("all_layer_attn", 24))\(padf(attnGpu))\(padf(attnCpu))   (sum across 27)")
    print("  \(pad("all_layer_ffn", 24))\(padf(ffnGpu))\(padf(ffnCpu))   (sum across 27)")
    print("  \(pad("TOTAL", 24))\(padf(totalGpu))\(padf(totalCpu))")
    print("  \(pad("avg_layer_attn", 24))\(padf(attnGpu / Double(weights.numLayers)))")
    print("  \(pad("avg_layer_ffn", 24))\(padf(ffnGpu / Double(weights.numLayers)))")

    // Save first/last/middle layer for hot-spot inspection
    print("")
    print("=== layer-by-layer (first/middle/last attn+ffn) ===")
    let interesting = [0, weights.numLayers/2, weights.numLayers-1]
    for L in interesting {
        for s in stages where s.name == "layer_\(L)_attn" || s.name == "layer_\(L)_ffn" {
            print("  \(pad(s.name, 24))\(padf(s.gpuMs))")
        }
    }
    fflush(stdout)
}
