// harness.swift — environment-variable-driven test & validation harnesses.
// Extracted from main.swift in the 2026-04-18 refactor so that env-var
// block bodies are reusable free functions rather than scattered top-level
// script code. main.swift now just checks each env var and calls these.
//
// Contract: each runXxxHarness() is callable independently, prints its own
// header and body, and never aborts (internal catches convert errors to
// printed diagnostics). Depends on common.swift (device/queue/GGUF/safetensors),
// kernels.swift, and vision_tower.swift (for runVisionTowerBatchForward etc.).

import Metal
import Foundation


// Env-var driver: VISION_ST
func runVisionPatchEmbedSmokeTest(stPath: String) {
    print("\n=== Vision patch-embed smoke test ===")
    do {
        let st = try SafetensorsFile(stPath)
        print("  loaded safetensors: \(st.tensors.count) tensors, dataStart @ \(st.dataStart)")

        // Load patch embedder weight: [D_out=1152, D_in=768] BF16 → FP16.
        let wName = "model.vision_tower.patch_embedder.input_proj.weight"
        let wInfo = try st.tensor(wName)
        precondition(wInfo.shape == [1152, 768], "unexpected patch embedder shape: \(wInfo.shape)")
        let W = try st.loadBF16AsFP16(wName, device: device)
        let D_out = wInfo.shape[0]
        let D_in = wInfo.shape[1]
        print(String(format: "  W loaded: [%d, %d] FP16 (%.2f MB)", D_out, D_in, Double(W.length) / 1e6))

        // Synthetic test image: [3, 32, 32] so we get 4 patches (2×2).
        // Fill with a seeded random pattern so the output is deterministic.
        let imgH = 32, imgW = 32
        let img = device.makeBuffer(length: 3 * imgH * imgW * 2, options: .storageModeShared)!
        let ip = img.contents().assumingMemoryBound(to: Float16.self)
        var s: UInt64 = 0xAA55_AA55_AA55_AA55
        for i in 0..<(3 * imgH * imgW) {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let u = Float((s >> 32) & 0xFFFFFF) / Float(0xFFFFFF)
            ip[i] = Float16(u - 0.5)
        }
        let nPatches = (imgH / 16) * (imgW / 16)
        let out = device.makeBuffer(length: nPatches * D_out * 2, options: .storageModeShared)!

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(visionPatchEmbedPSO)
        enc.setBuffer(img, offset: 0, index: 0)
        enc.setBuffer(W, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var ih = UInt32(imgH), iw = UInt32(imgW), dou = UInt32(D_out)
        enc.setBytes(&ih, length: 4, index: 3)
        enc.setBytes(&iw, length: 4, index: 4)
        enc.setBytes(&dou, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: D_out / 32, height: nPatches, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()

        // CPU reference: unfold each patch, dot-product against W rows. Compare first 128 d_out.
        let wp = W.contents().assumingMemoryBound(to: Float16.self)
        let op = out.contents().assumingMemoryBound(to: Float16.self)
        var maxAbs: Float = 0, maxRel: Float = 0, refRms: Float = 0
        let checkDout = min(D_out, 128)
        for patch in 0..<nPatches {
            let py = patch / (imgW / 16); let px = patch % (imgW / 16)
            // HWC flatten to match the kernel and Gemma4ImageProcessor:
            // k = (y_in * 16 + x_in) * 3 + c.
            var unfold = [Float](repeating: 0, count: 768)
            for y in 0..<16 {
                for x in 0..<16 {
                    for c in 0..<3 {
                        unfold[(y * 16 + x) * 3 + c] =
                            Float(ip[c * imgH * imgW + (py * 16 + y) * imgW + (px * 16 + x)])
                    }
                }
            }
            for n in 0..<checkDout {
                var ref: Float = 0
                for k in 0..<768 {
                    ref += unfold[k] * Float(wp[n * 768 + k])
                }
                let got = Float(op[patch * D_out + n])
                let diff = abs(got - ref)
                let rel = diff / max(abs(ref), 1e-4)
                maxAbs = max(maxAbs, diff)
                maxRel = max(maxRel, rel)
                refRms += ref * ref
            }
        }
        refRms = sqrt(refRms / Float(nPatches * checkDout))
        print(String(format: "  patches=%d, checked %d d_out: max abs %.4e, max rel %.3e, ||ref||_rms %.3e",
                     nPatches, checkDout, maxAbs, maxRel, refRms))
        if maxRel < 0.01 {
            print("  ✓ patch-embed kernel matches CPU reference within 1%")
        } else {
            print("  ✗ patch-embed kernel diverges — inspect layout assumptions")
        }
    } catch {
        print("  vision smoke test failed: \(error)")
    }
}

// Env-var driver: VISION_FORWARD+VISION_ST
func runVisionEndToEndForward(framePath: String, stPath: String) {
    print("\n=== Vision end-to-end forward ===")
    do {
        let st = try SafetensorsFile(stPath)
        let weights = try loadVisionWeights(st, device: device)

        var batch = try gemma4ImagePreprocessFromPath(path: framePath, device: device)
        // VISION_FORCE_PATCHES=<fp16-bin> — override our preprocessor with an
        // externally-supplied patch buffer (e.g. ref's torchvision bicubic output)
        // to isolate model divergence from preprocessor divergence.
        if let overridePath = ProcessInfo.processInfo.environment["VISION_FORCE_PATCHES"] {
            let data = try Data(contentsOf: URL(fileURLWithPath: overridePath))
            let expectedBytes = batch.numRealPatches * 768 * 2
            precondition(data.count == expectedBytes,
                         "patch override size \(data.count) ≠ expected \(expectedBytes)")
            let dst = batch.patches.contents()
            data.withUnsafeBytes { src in
                memcpy(dst, src.baseAddress, expectedBytes)
            }
            print("  patches overridden from \(overridePath)")
        }
        print(String(format: "  image %@ → %d × %d grid, %d patches",
                     (framePath as NSString).lastPathComponent, batch.gridH, batch.gridW, batch.numRealPatches))

        let t0 = Date()
        let results = runVisionTowerBatchForward(
            batches: [batch], weights: weights, device: device, queue: queue)
        let (softTokens, nPooled) = results[0]
        let fwdMs = Date().timeIntervalSince(t0) * 1000
        print(String(format: "  forward: %d pooled tokens in %.1f ms (%.2f tok/ms)",
                     nPooled, fwdMs, Double(nPooled) / fwdMs))

        // Sample some output stats. softTokens is now fp32.
        let stPtr = softTokens.contents().assumingMemoryBound(to: Float.self)
        let total = nPooled * weights.textHidden
        var mean: Double = 0, sq: Double = 0
        var minV: Float = .infinity, maxV: Float = -.infinity
        for i in 0..<total {
            let v = stPtr[i]
            mean += Double(v); sq += Double(v*v)
            if v.isFinite { minV = min(minV, v); maxV = max(maxV, v) }
        }
        mean /= Double(total)
        let std = (sq / Double(total) - mean * mean).squareRoot()
        print(String(format: "  output stats: shape=[%d, %d] fp32, mean=%.4f, std=%.4f, min=%.3f, max=%.3f",
                     nPooled, weights.textHidden, mean, std, minV, maxV))
        print(String(format: "  first 8 values of token 0: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                     stPtr[0], stPtr[1], stPtr[2], stPtr[3], stPtr[4], stPtr[5], stPtr[6], stPtr[7]))

        // Optional: compare to Python reference .npy (fp16 bf16-run oracle).
        if let refPath = ProcessInfo.processInfo.environment["VISION_REF"] {
            let fd = open(refPath, O_RDONLY)
            if fd >= 0 {
                var stBuf = stat()
                fstat(fd, &stBuf)
                let size = Int(stBuf.st_size)
                if let p = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED {
                    let magic = p.assumingMemoryBound(to: UInt8.self)
                    precondition(magic[0] == 0x93 && magic[1] == 0x4E && magic[2] == 0x55, "bad npy magic")
                    let major = magic[6]
                    let headerLen: Int
                    let dataOffset: Int
                    if major == 1 {
                        headerLen = Int(p.load(fromByteOffset: 8, as: UInt16.self))
                        dataOffset = 10 + headerLen
                    } else {
                        headerLen = Int(p.load(fromByteOffset: 8, as: UInt32.self))
                        dataOffset = 12 + headerLen
                    }
                    let frameBytes = nPooled * weights.textHidden * 2   // ref is still fp16
                    precondition(dataOffset + frameBytes <= size, "ref npy smaller than expected")
                    let refPtr = p.advanced(by: dataOffset).assumingMemoryBound(to: Float16.self)
                    var sumSqErr: Double = 0, sumSqRef: Double = 0, maxAbs: Float = 0
                    for i in 0..<total {
                        let r = Float(refPtr[i])
                        let g = stPtr[i]
                        let d = g - r
                        sumSqErr += Double(d * d)
                        sumSqRef += Double(r * r)
                        maxAbs = max(maxAbs, abs(d))
                    }
                    let mse = sumSqErr / Double(total)
                    let refRms = (sumSqRef / Double(total)).squareRoot()
                    let relRms = (sumSqErr / max(sumSqRef, 1e-9)).squareRoot()
                    print(String(format: "  MSE vs ref (frame 0): %.6e, max_abs_err: %.4f, ref_rms: %.4f, rel_rms_err: %.3e",
                                 mse, maxAbs, refRms, relRms))
                    print(String(format: "  ref first 8:  [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                                 Float(refPtr[0]), Float(refPtr[1]), Float(refPtr[2]), Float(refPtr[3]),
                                 Float(refPtr[4]), Float(refPtr[5]), Float(refPtr[6]), Float(refPtr[7])))
                    munmap(p, size)
                }
                close(fd)
            } else {
                print("  VISION_REF file open failed")
            }
        }
    } catch {
        print("  forward failed: \(error)")
    }
}

// VISION_CONCURRENT_QUEUES=<n> — submit N vision forwards in parallel on
// N distinct MTLCommandQueues. Answers: does the M5 Max GPU actually run
// multiple CBs from different queues concurrently, or does it serialize
// them? Single-queue baseline vs N-queue wall-clock tells us directly.
func runVisionConcurrentQueues(batchDir: String, stPath: String, nQueues: Int) {
    print("\n=== Vision concurrent-queue test (N=\(nQueues)) ===")
    do {
        let st = try SafetensorsFile(stPath)
        let weights = try loadVisionWeights(st, device: device)
        let fm = FileManager.default
        let pngs = (try fm.contentsOfDirectory(atPath: batchDir))
            .filter { $0.hasSuffix(".png") }.sorted().prefix(nQueues)
        precondition(pngs.count >= 1, "need ≥1 PNG in VISION_BATCH_DIR")
        var batches: [PatchBatch] = []
        for p in pngs {
            batches.append(try gemma4ImagePreprocessFromPath(path: "\(batchDir)/\(p)", device: device))
        }

        // Baseline: single queue, serial (batched-B=1 per image, looped).
        let tSerial0 = Date()
        for b in batches {
            _ = runVisionTowerBatchForward(batches: [b], weights: weights, device: device, queue: queue)
        }
        let tSerial = Date().timeIntervalSince(tSerial0) * 1000
        print(String(format: "  serial on 1 queue:  %.1f ms (%d images)", tSerial, batches.count))

        // Concurrent: N queues, submit all async, wait on all.
        let queues = (0..<batches.count).map { _ in device.makeCommandQueue()! }
        let tPar0 = Date()
        var cbs: [MTLCommandBuffer] = []
        for (i, b) in batches.enumerated() {
            let (_, cb) = runVisionTowerBatchForwardAsync(batches: [b], weights: weights,
                                                           device: device, queue: queues[i])
            cb.commit()
            cbs.append(cb)
        }
        for cb in cbs { cb.waitUntilCompleted() }
        let tPar = Date().timeIntervalSince(tPar0) * 1000
        print(String(format: "  concurrent on %d queues: %.1f ms  (%.2fx speedup)",
                     batches.count, tPar, tSerial / tPar))
    } catch {
        print("  concurrent-queue test failed: \(error)")
    }
}

// VISION_BATCH_DIR=<dir> — run the batched forward over the first 4
// PNGs in the directory, compare per-image soft-tokens against a looped
// B=1 reference path. Validates numerical equivalence (batched B>1 kernel
// dispatches vs looped B=1 of the same kernels) and measures wall-clock
// speedup (one B=N CB vs N sequential B=1 CBs).
func runVisionBatchForward(batchDir: String, stPath: String) {
    print("\n=== Vision batched forward (B=4) ===")
    do {
        let st = try SafetensorsFile(stPath)
        let weights = try loadVisionWeights(st, device: device)
        let fm = FileManager.default
        let pngs = (try fm.contentsOfDirectory(atPath: batchDir))
            .filter { $0.hasSuffix(".png") }.sorted().prefix(4)
        precondition(pngs.count >= 2, "need ≥2 PNGs in VISION_BATCH_DIR")

        var batches: [PatchBatch] = []
        for p in pngs {
            let b = try gemma4ImagePreprocessFromPath(path: "\(batchDir)/\(p)", device: device)
            batches.append(b)
            print(String(format: "  %@ → %d×%d grid, %d patches",
                         p, b.gridH, b.gridW, b.numRealPatches))
        }
        let B = batches.count

        // Reference: each image through batched-B=1 sequentially.
        let tSerial0 = Date()
        var serialOut: [(MTLBuffer, Int)] = []
        for b in batches {
            let r = runVisionTowerBatchForward(batches: [b], weights: weights,
                                                 device: device, queue: queue)
            serialOut.append(r[0])
        }
        let tSerial = Date().timeIntervalSince(tSerial0) * 1000
        print(String(format: "  B=1 × %d sequential: %.1f ms", B, tSerial))

        // Batched forward.
        let tBatch0 = Date()
        let batchOut = runVisionTowerBatchForward(batches: batches, weights: weights,
                                                    device: device, queue: queue)
        let tBatch = Date().timeIntervalSince(tBatch0) * 1000
        print(String(format: "  batched B=%d: %.1f ms  (%.2fx speedup)",
                     B, tBatch, tSerial / tBatch))

        // Per-image bit-equivalence check.
        precondition(serialOut.count == batchOut.count)
        for i in 0..<B {
            precondition(serialOut[i].1 == batchOut[i].1, "nPooled mismatch for image \(i)")
            let nPooled = serialOut[i].1
            let total = nPooled * weights.textHidden
            let sp = serialOut[i].0.contents().assumingMemoryBound(to: Float.self)
            let bp = batchOut[i].0.contents().assumingMemoryBound(to: Float.self)
            var maxAbsDiff: Float = 0
            var sumSqDiff: Double = 0, sumSqRef: Double = 0
            for k in 0..<total {
                let d = sp[k] - bp[k]
                maxAbsDiff = max(maxAbsDiff, abs(d))
                sumSqDiff += Double(d * d)
                sumSqRef  += Double(sp[k] * sp[k])
            }
            let mse = sumSqDiff / Double(total)
            let relRms = (sumSqDiff / max(sumSqRef, 1e-9)).squareRoot()
            print(String(format: "  image %d: serial first8=[%.4f,%.4f,%.4f,%.4f,...] batch first8=[%.4f,%.4f,%.4f,%.4f,...] max|Δ|=%.3e MSE=%.3e rel_rms=%.3e",
                         i, sp[0], sp[1], sp[2], sp[3], bp[0], bp[1], bp[2], bp[3],
                         maxAbsDiff, mse, relRms))
        }
    } catch {
        print("  batched forward failed: \(error)")
    }
}

// Env-var driver: VISION_ASPECT_DIR+VISION_ST
func runVisionAspectSweep(aspectDir: String, stPath: String) {
    print("\n=== Vision aspect-ratio sweep ===")
    do {
        let st = try SafetensorsFile(stPath)
        let weights = try loadVisionWeights(st, device: device)

        let fm = FileManager.default
        let all = try fm.contentsOfDirectory(atPath: aspectDir)
            .filter { $0.hasPrefix("aspect_") && $0.hasSuffix(".png") }
            .sorted()
        for png in all {
            let pngPath = "\(aspectDir)/\(png)"
            let stem = String(png.dropLast(4))
            let refPath = "\(aspectDir)/\(stem)_ref.npy"
            guard fm.fileExists(atPath: refPath) else {
                print("  skip \(png): missing \(stem)_ref.npy")
                continue
            }
            let batch = try gemma4ImagePreprocessFromPath(path: pngPath, device: device)
            let t0 = Date()
            let r = runVisionTowerBatchForward(
                batches: [batch], weights: weights, device: device, queue: queue)
            let (softTokens, nPooled) = r[0]
            let fwdMs = Date().timeIntervalSince(t0) * 1000

            // Load ref (shape is [N_soft, 2816] fp16; a single frame).
            let fd = open(refPath, O_RDONLY)
            precondition(fd >= 0, "open \(refPath) failed")
            var stBuf = stat(); fstat(fd, &stBuf)
            let size = Int(stBuf.st_size)
            guard let rp = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0), rp != MAP_FAILED else {
                print("  mmap \(refPath) failed"); close(fd); continue
            }
            let magic = rp.assumingMemoryBound(to: UInt8.self)
            precondition(magic[0] == 0x93 && magic[1] == 0x4E && magic[2] == 0x55, "bad npy magic")
            let major = magic[6]
            let dataOffset: Int
            if major == 1 {
                dataOffset = 10 + Int(rp.load(fromByteOffset: 8, as: UInt16.self))
            } else {
                dataOffset = 12 + Int(rp.load(fromByteOffset: 8, as: UInt32.self))
            }
            let refPtr = rp.advanced(by: dataOffset).assumingMemoryBound(to: Float16.self)
            let total = nPooled * weights.textHidden
            let stPtr = softTokens.contents().assumingMemoryBound(to: Float.self)
            var sumSqErr: Double = 0, sumSqRef: Double = 0, maxAbs: Float = 0
            for j in 0..<total {
                let r = Float(refPtr[j])
                let g = stPtr[j]
                let d = g - r
                sumSqErr += Double(d * d); sumSqRef += Double(r * r)
                maxAbs = max(maxAbs, abs(d))
            }
            let mse = sumSqErr / Double(total)
            let relRms = (sumSqErr / max(sumSqRef, 1e-9)).squareRoot()
            munmap(rp, size); close(fd)

            print(String(format: "  %-25s grid=%d×%d=%d patches  nSoft=%d  MSE=%.4e  rel=%.2f%%  maxAbs=%.2f  %.0fms",
                         png, batch.gridH, batch.gridW, batch.numRealPatches, nPooled,
                         mse, relRms * 100, maxAbs, fwdMs))
        }
    } catch {
        print("  aspect sweep failed: \(error)")
    }
}

// Env-var driver: VISION_SWEEP_DIR+VISION_ST+VISION_REF+VISION_SWEEP_ORDER
func runVisionMultiFrameSweep(sweepDir: String, stPath: String, refPath: String, orderPath: String) {
    print("\n=== Vision multi-frame MSE sweep ===")
    do {
        let st = try SafetensorsFile(stPath)
        let weights = try loadVisionWeights(st, device: device)
        // Parse frame order file (one filename per line — the stored npy is
        // ordered to match).
        let orderText = try String(contentsOfFile: orderPath, encoding: .utf8)
        let frameNames = orderText.split(separator: "\n").map(String.init).filter { !$0.isEmpty }
        let maxFrames = Int(ProcessInfo.processInfo.environment["VISION_SWEEP_N"] ?? "414") ?? 414
        let nFrames = min(maxFrames, frameNames.count)

        // mmap the reference npy once.
        let fd = open(refPath, O_RDONLY)
        precondition(fd >= 0, "open VISION_REF failed")
        var stBuf = stat()
        fstat(fd, &stBuf)
        let refSize = Int(stBuf.st_size)
        let rp = mmap(nil, refSize, PROT_READ, MAP_PRIVATE, fd, 0)!
        precondition(rp != MAP_FAILED, "mmap VISION_REF failed")
        let magic = rp.assumingMemoryBound(to: UInt8.self)
        precondition(magic[0] == 0x93 && magic[1] == 0x4E && magic[2] == 0x55, "bad npy magic")
        let major = magic[6]
        let dataOffset: Int
        if major == 1 {
            dataOffset = 10 + Int(rp.load(fromByteOffset: 8, as: UInt16.self))
        } else {
            dataOffset = 12 + Int(rp.load(fromByteOffset: 8, as: UInt32.self))
        }
        let refBase = rp.advanced(by: dataOffset).assumingMemoryBound(to: Float16.self)
        // Stored layout: [N_frames_total, 272, 2816] fp16
        let refStridePerFrame = 272 * weights.textHidden

        print("  sweeping \(nFrames) frames from \(sweepDir)")

        var mseList = [Double]()
        var relRmsList = [Double]()
        var maxAbsList = [Double]()
        var tFwdList = [Double]()
        for i in 0..<nFrames {
            let fp = "\(sweepDir)/\(frameNames[i])"
            let batch = try gemma4ImagePreprocessFromPath(path: fp, device: device)
            let t0 = Date()
            let r = runVisionTowerBatchForward(
                batches: [batch], weights: weights, device: device, queue: queue)
            let (softTokens, nPooled) = r[0]
            let fwdMs = Date().timeIntervalSince(t0) * 1000
            tFwdList.append(fwdMs)
            let total = nPooled * weights.textHidden
            precondition(total == refStridePerFrame, "frame \(i) expected \(refStridePerFrame) elements, got \(total)")
            let stPtr = softTokens.contents().assumingMemoryBound(to: Float.self)
            let refPtrF = refBase.advanced(by: i * refStridePerFrame)

            var sumSqErr: Double = 0, sumSqRef: Double = 0, maxAbs: Float = 0
            for j in 0..<total {
                let r = Float(refPtrF[j])
                let g = stPtr[j]
                let d = g - r
                sumSqErr += Double(d * d)
                sumSqRef += Double(r * r)
                maxAbs = max(maxAbs, abs(d))
            }
            let mse = sumSqErr / Double(total)
            let relRms = (sumSqErr / max(sumSqRef, 1e-9)).squareRoot()
            mseList.append(mse)
            relRmsList.append(relRms)
            maxAbsList.append(Double(maxAbs))
            if i < 5 || i == nFrames - 1 || (i % 20 == 0) {
                print(String(format: "  [%3d/%3d] %-40s %d×%d g  MSE=%.4e rel=%.2f%%  maxAbs=%.2f  %.0fms",
                             i, nFrames, frameNames[i],
                             batch.gridH, batch.gridW,
                             mse, relRms * 100, maxAbs, fwdMs))
            }
        }
        munmap(rp, refSize)
        close(fd)

        // Summary stats
        func stats(_ xs: [Double], _ label: String, _ fmt: String) {
            let sorted = xs.sorted()
            let n = sorted.count
            let mean = xs.reduce(0.0, +) / Double(n)
            let sq = xs.reduce(0.0) { $0 + $1 * $1 }
            let std = (sq / Double(n) - mean * mean).squareRoot()
            let p50 = sorted[n / 2]
            let p95 = sorted[Int(Double(n) * 0.95)]
            let p99 = sorted[min(n - 1, Int(Double(n) * 0.99))]
            let mn = sorted[0], mx = sorted[n - 1]
            print(String(format: "  \(label): mean=\(fmt)  std=\(fmt)  min=\(fmt)  p50=\(fmt)  p95=\(fmt)  p99=\(fmt)  max=\(fmt)",
                         mean, std, mn, p50, p95, p99, mx))
        }
        print("\n=== Sweep summary (n=\(nFrames)) ===")
        stats(mseList, "MSE      ", "%.4e")
        stats(relRmsList.map { $0 * 100 }, "rel-rms %", "%.3f")
        stats(maxAbsList, "max|err| ", "%.3f")
        stats(tFwdList, "t_fwd ms ", "%.0f")
    } catch {
        print("  sweep failed: \(error)")
    }
}

// Env-var driver: VISION_LOAD
func runVisionWeightLoadSmokeTest(stPath: String) {
    print("\n=== Vision full-weight load smoke test ===")
    do {
        let st = try SafetensorsFile(stPath)
        let t0 = Date()
        let w = try loadVisionWeights(st, device: device)
        let loadMs = Date().timeIntervalSince(t0) * 1000
        let totalBytes = w.patchEmbedW.length + w.posEmbedTable.length + w.stdBias.length + w.stdScale.length + w.embedVisionProj.length +
            w.layers.reduce(0) { acc, l in
                acc + l.inputNorm.length + l.qProj.length + l.kProj.length + l.vProj.length + l.oProj.length +
                    l.qNorm.length + l.kNorm.length + l.postAttnNorm.length + l.preFfnNorm.length +
                    l.gateProj.length + l.upProj.length + l.downProj.length + l.postFfnNorm.length
            }
        print(String(format: "  loaded %d layers + 5 global in %.1f ms", w.numLayers, loadMs))
        print(String(format: "  total fp16 vision weights: %.2f GB", Double(totalBytes) / 1e9))
        // Sanity: layer 0 q_proj should be 1152*1152*2 = ~2.65 MB
        print(String(format: "  layer 0 q_proj size: %.2f MB (%d bytes)",
                     Double(w.layers[0].qProj.length) / (1024*1024), w.layers[0].qProj.length))
        // Sample first few values of layer 0 input_norm gamma
        let normP = w.layers[0].inputNorm.contents().assumingMemoryBound(to: Float16.self)
        print(String(format: "  L0 input_layernorm gamma[0..7] = [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                     Float(normP[0]), Float(normP[1]), Float(normP[2]), Float(normP[3]),
                     Float(normP[4]), Float(normP[5]), Float(normP[6]), Float(normP[7])))
    } catch {
        print("  vision load failed: \(error)")
    }
}

// Env-var driver: VISION_PREPROCESS
func runVisionPreprocessSmokeTest(png: String) {
    print("\n=== Vision preprocessor smoke test ===")
    do {
        let batch = try gemma4ImagePreprocessFromPath(path: png, device: device)
        print("  resized: \(batch.resizedH) × \(batch.resizedW)")
        print("  grid: \(batch.gridH) × \(batch.gridW) = \(batch.numRealPatches) real patches")
        print("  after pool÷3: \((batch.gridH/3)) × \((batch.gridW/3)) = \((batch.gridH/3) * (batch.gridW/3)) soft tokens")
        // Sample a real patch
        let p = batch.patches.contents().assumingMemoryBound(to: Float16.self)
        let patchBase = 0
        print(String(format: "  patch 0, first 12 values: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                     Float(p[patchBase+0]), Float(p[patchBase+1]), Float(p[patchBase+2]),
                     Float(p[patchBase+3]), Float(p[patchBase+4]), Float(p[patchBase+5]),
                     Float(p[patchBase+6]), Float(p[patchBase+7]), Float(p[patchBase+8]),
                     Float(p[patchBase+9]), Float(p[patchBase+10]), Float(p[patchBase+11])))
        // Sample a real patch from the middle of the image
        let midPatchIdx = batch.numRealPatches / 2
        let midBase = midPatchIdx * 768
        print(String(format: "  patch %d, value range: min=%.4f, max=%.4f",
                     midPatchIdx,
                     (0..<768).map { Float(p[midBase + $0]) }.min() ?? 0,
                     (0..<768).map { Float(p[midBase + $0]) }.max() ?? 0))
        // VISION_DUMP_PATCHES=<path.bin> — write the fp16 patches as raw bytes.
        if let dumpPath = ProcessInfo.processInfo.environment["VISION_DUMP_PATCHES"] {
            let nBytes = batch.numRealPatches * 768 * 2
            let data = Data(bytes: batch.patches.contents(), count: nBytes)
            try data.write(to: URL(fileURLWithPath: dumpPath))
            print("  dumped \(batch.numRealPatches) × 768 fp16 patches → \(dumpPath)")
        }
    } catch {
        print("  preprocess failed: \(error)")
    }
}

// Env-var driver: GGUF_PATH — load weights and run the LM forward benchmark
// (10 AR-decode steps at B=4 from BOS, with finite-check + top-5 token dump).
func runGgufPathHarness(ggufPath: String) {
    print("\n=== GGUF real-weight LM forward ===")
    runLmForwardBench(ggufPath: ggufPath)
}

// -------- Tiny npy reader helper (inline; matches the .npy v1/v2 format). --------
// Returns (dataPointer, byteSize, shape, dtypeTag) where dtypeTag is a short
// string like "<f4" or "<i4" read verbatim from the header. Mmap the file —
// the returned pointer is valid for the lifetime of the file descriptor.
struct NpyMmap {
    let ptr: UnsafeMutableRawPointer   // base of the mmapped file
    let size: Int                       // total file bytes
    let dataOffset: Int                 // bytes to skip past .npy header
    let shape: [Int]
    let dtype: String
    let fd: Int32
    func release() { munmap(ptr, size); close(fd) }
}

func mmapNpy(_ path: String) -> NpyMmap? {
    let fd = open(path, O_RDONLY)
    if fd < 0 { return nil }
    var st = stat()
    if fstat(fd, &st) != 0 { close(fd); return nil }
    let size = Int(st.st_size)
    guard let p = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
        close(fd); return nil
    }
    let magic = p.assumingMemoryBound(to: UInt8.self)
    guard magic[0] == 0x93, magic[1] == 0x4E, magic[2] == 0x55,
          magic[3] == 0x4D, magic[4] == 0x50, magic[5] == 0x59 else {
        munmap(p, size); close(fd); return nil
    }
    let major = magic[6]
    let headerLen: Int
    let dataOffset: Int
    if major == 1 {
        headerLen = Int(p.load(fromByteOffset: 8, as: UInt16.self))
        dataOffset = 10 + headerLen
    } else {
        headerLen = Int(p.load(fromByteOffset: 8, as: UInt32.self))
        dataOffset = 12 + headerLen
    }
    let headerStart = (major == 1) ? 10 : 12
    let headerBytes = UnsafeBufferPointer(
        start: p.advanced(by: headerStart).assumingMemoryBound(to: UInt8.self),
        count: headerLen)
    let header = String(decoding: headerBytes, as: UTF8.self)
    // header looks like: {'descr': '<f4', 'fortran_order': False, 'shape': (8, 262144), }
    func between(_ s: String, _ lo: String, _ hi: String) -> String? {
        guard let r1 = s.range(of: lo) else { return nil }
        guard let r2 = s.range(of: hi, range: r1.upperBound..<s.endIndex) else { return nil }
        return String(s[r1.upperBound..<r2.lowerBound])
    }
    let dtype = between(header, "'descr': '", "'") ?? "?"
    let shapeStr = between(header, "'shape': (", ")") ?? ""
    let shape: [Int] = shapeStr.split(separator: ",").compactMap {
        Int($0.trimmingCharacters(in: .whitespaces))
    }
    return NpyMmap(ptr: p, size: size, dataOffset: dataOffset,
                    shape: shape, dtype: dtype, fd: fd)
}

// Env-var driver: LM_KL_REF=<dir>  +  LM_KL_TAG=<tag>  +  GGUF_PATH=<gguf>
// Loads lm_<tag>_tokens.npy (int32[S]) and lm_<tag>_logits.npy (f32[S, VOCAB])
// from <dir>, then re-runs the same token sequence through the Swift LM
// forward (B=4 with all four slots fed the same tokens — slot 0 is the one
// we compare against; others are currently wasted compute, to be reclaimed
// when B becomes a runtime parameter). Per position p ∈ [0, S) computes:
//   log-space L2:  sqrt(mean((lm_swift[p,:] - lm_oracle[p,:])^2))
//   KL(oracle ∥ swift):  sum_i softmax(oracle)[i] * (log_softmax(oracle)[i] - log_softmax(swift)[i])
//   top-5 overlap:  |top5(oracle) ∩ top5(swift)|
//   argmax match:  oracle.argmax == swift.argmax
func runLmKLHarness(ggufPath: String, refDir: String, tag: String) {
    print("\n=== LM KL-divergence harness vs Python oracle ===")
    let tokensPath = "\(refDir)/lm_\(tag)_tokens.npy"
    let logitsPath = "\(refDir)/lm_\(tag)_logits.npy"
    guard let tokNpy = mmapNpy(tokensPath) else { print("  cannot open \(tokensPath)"); return }
    guard let logNpy = mmapNpy(logitsPath) else { print("  cannot open \(logitsPath)"); tokNpy.release(); return }
    defer { tokNpy.release(); logNpy.release() }

    // Validate shapes / dtypes.
    precondition(tokNpy.shape.count == 1, "tokens must be 1D, got shape \(tokNpy.shape)")
    precondition(tokNpy.dtype == "<i4", "tokens must be int32, got \(tokNpy.dtype)")
    precondition(logNpy.shape.count == 2, "logits must be 2D, got shape \(logNpy.shape)")
    precondition(logNpy.dtype == "<f4", "logits must be float32, got \(logNpy.dtype)")
    let S = tokNpy.shape[0]
    let V = logNpy.shape[1]
    precondition(logNpy.shape[0] == S, "logits rows \(logNpy.shape[0]) != tokens \(S)")
    precondition(V == VOCAB, "vocab mismatch: oracle \(V) vs swift \(VOCAB)")
    print("  prompt length S=\(S); vocab=\(V)")

    let toks: [UInt32] = {
        let p = tokNpy.ptr.advanced(by: tokNpy.dataOffset)
            .assumingMemoryBound(to: Int32.self)
        return (0..<S).map { UInt32(bitPattern: p[$0]) }
    }()
    let oracleBase = logNpy.ptr.advanced(by: logNpy.dataOffset)
        .assumingMemoryBound(to: Float.self)

    // Load weights and initialize state with the first prompt token.
    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    print("")

    // Per-position swift logits captured from slot 0. Float16 → read to
    // [Float] on CPU for KL math.
    var swiftLogits = [Float](repeating: 0, count: S * VOCAB)

    // Step 0 uses the first prompt token as the initial input. initLmState
    // primes positions=0, k_len=1, num_pages=1, block_table per-slot-disjoint.
    initLmState(bos: toks[0])
    let cb0 = buildStepCB(w); cb0.commit(); cb0.waitUntilCompleted()
    if let err = cb0.error { print("  GPU step 0: \(err)"); return }
    copySlot0LogitsToFloat(dst: &swiftLogits, destOffset: 0)

    // Subsequent positions: advance state with tokens[p] as the new input,
    // run buildStepCB, capture slot-0 logits.
    for p in 1..<S {
        let nextToks = [UInt32](repeating: toks[p], count: B)
        advanceLmState(nextTokens: nextToks)
        let cb = buildStepCB(w); cb.commit(); cb.waitUntilCompleted()
        if let err = cb.error { print("  GPU step \(p): \(err)"); return }
        copySlot0LogitsToFloat(dst: &swiftLogits, destOffset: p * VOCAB)
    }

    // Per-position metrics.
    print("  per-position metrics (oracle vs swift, slot 0):")
    print("    pos  token     L2        KL(ora‖swi)   top1?  top5∩  ora-next                    swi-next")
    var totalL2: Double = 0
    var totalKL: Double = 0
    var totalTop1: Int = 0
    var totalTop5Overlap: Int = 0
    for p in 0..<S {
        let orp = UnsafeBufferPointer(start: oracleBase.advanced(by: p * VOCAB), count: VOCAB)
        let (l2, kl, top1Match, top5Overlap, oraTop, swiTop) = swiftLogits.withUnsafeBufferPointer { swBuf -> (Float, Float, Bool, Int, Int, Int) in
            let swp = UnsafeBufferPointer(start: swBuf.baseAddress!.advanced(by: p * VOCAB), count: VOCAB)
            return scoreLogitPosition(orp, swp)
        }
        totalL2 += Double(l2)
        totalKL += Double(kl)
        totalTop1 += top1Match ? 1 : 0
        totalTop5Overlap += top5Overlap
        let oraTokStr = safeVocabToken(w, oraTop).padding(toLength: 20, withPad: " ", startingAt: 0)
        let swiTokStr = safeVocabToken(w, swiTop).padding(toLength: 20, withPad: " ", startingAt: 0)
        let tokStr = safeVocabToken(w, Int(toks[p])).padding(toLength: 8, withPad: " ", startingAt: 0)
        let posStr = String(format: "%3d", p)
        let l2Str = String(format: "%7.4f", l2)
        let klStr = String(format: "%9.4f", kl)
        let oraIdStr = String(format: "%6d", oraTop)
        let swiIdStr = String(format: "%6d", swiTop)
        print("    \(posStr)  \(tokStr)  \(l2Str)   \(klStr)   \(top1Match ? "✓" : "✗")      "
              + "\(top5Overlap)/5    \(oraIdStr)=\(oraTokStr)  \(swiIdStr)=\(swiTokStr)")
    }
    print("")
    print(String(format: "  mean L2      : %.4f", totalL2 / Double(S)))
    print(String(format: "  mean KL      : %.4f", totalKL / Double(S)))
    print(String(format: "  argmax match : %d / %d", totalTop1, S))
    print(String(format: "  top-5 overlap: %d / %d (max %d)", totalTop5Overlap, 5 * S, 5 * S))
}

// Copy logits[slot=0, :] (fp16) from the GPU buffer into a Float slice on CPU,
// performing fp16→fp32 widening. `destOffset` is the starting index in `dst`.
private func copySlot0LogitsToFloat(dst: inout [Float], destOffset: Int) {
    let p = logits.contents().assumingMemoryBound(to: Float16.self)
    for v in 0..<VOCAB {
        dst[destOffset + v] = Float(p[v])
    }
}

// Position-level scoring: log-softmax L2, KL(oracle‖swift), top-1 match,
// size of top-5 set intersection, and each side's argmax.
private func scoreLogitPosition(_ oracle: UnsafeBufferPointer<Float>,
                                _ swift: UnsafeBufferPointer<Float>)
    -> (l2: Float, kl: Float, top1: Bool, top5: Int, oraArg: Int, swiArg: Int)
{
    precondition(oracle.count == swift.count)
    let V = oracle.count
    // log-softmax for both (numerically stable).
    var oraMax = -Float.infinity
    var swiMax = -Float.infinity
    for i in 0..<V {
        if oracle[i] > oraMax { oraMax = oracle[i] }
        if swift[i]  > swiMax { swiMax  = swift[i]  }
    }
    var oraSumExp: Double = 0
    var swiSumExp: Double = 0
    for i in 0..<V {
        oraSumExp += Double(exp(oracle[i] - oraMax))
        swiSumExp += Double(exp(swift[i]  - swiMax))
    }
    let oraLogZ = oraMax + Float(log(oraSumExp))
    let swiLogZ = swiMax + Float(log(swiSumExp))
    // Streaming KL + L2 + top-1/top-5 in one pass.
    var kl: Double = 0
    var sqL2: Double = 0
    var oraArg = 0; var oraArgV = -Float.infinity
    var swiArg = 0; var swiArgV = -Float.infinity
    var oraTop5: [(Int, Float)] = []
    var swiTop5: [(Int, Float)] = []
    for i in 0..<V {
        let ol = oracle[i] - oraLogZ
        let sl = swift[i]  - swiLogZ
        let p = exp(ol)
        kl += Double(p) * Double(ol - sl)
        let diff = ol - sl
        sqL2 += Double(diff * diff)
        if oracle[i] > oraArgV { oraArgV = oracle[i]; oraArg = i }
        if swift[i]  > swiArgV { swiArgV = swift[i];  swiArg = i }
        // Top-5 accumulation on raw logits.
        if oraTop5.count < 5 {
            oraTop5.append((i, oracle[i]))
            oraTop5.sort { $0.1 > $1.1 }
        } else if oracle[i] > oraTop5[4].1 {
            oraTop5[4] = (i, oracle[i])
            oraTop5.sort { $0.1 > $1.1 }
        }
        if swiTop5.count < 5 {
            swiTop5.append((i, swift[i]))
            swiTop5.sort { $0.1 > $1.1 }
        } else if swift[i] > swiTop5[4].1 {
            swiTop5[4] = (i, swift[i])
            swiTop5.sort { $0.1 > $1.1 }
        }
    }
    let oraSet = Set(oraTop5.map { $0.0 })
    let swiSet = Set(swiTop5.map { $0.0 })
    let overlap = oraSet.intersection(swiSet).count
    let l2 = Float((sqL2 / Double(V)).squareRoot())
    return (l2, Float(kl), oraArg == swiArg, overlap, oraArg, swiArg)
}

// Decode a vocab token string (with newlines escaped) — or "<oov>" if id out of range.
private func safeVocabToken(_ w: LmWeights, _ id: Int) -> String {
    guard id >= 0, id < w.vocabTokens.count else { return "<oov>" }
    return w.vocabTokens[id].replacingOccurrences(of: "\n", with: "\\n")
}

// Env-var driver: GGUF_VALIDATE
func runGgufValidateHarness(ggufPath: String) {
    print("\n=== GGUF validation ===")
    do {
        let g = try GGUFFile(ggufPath)
        let info = try g.tensor("blk.0.attn_q.weight")
        precondition(info.dtype == .q8_0, "expected Q8_0")
        let Din = info.shape[0], Dout = info.shape[1]
        print("  tensor: \(info.name) Q8_0 \(Din)×\(Dout) (\(info.byteSize / (1024*1024)) MB)")

        // Wrap raw GGUF bytes as a read-only MTLBuffer (zero-copy mmap).
        let Wraw = try g.makeMetalBuffer(info.name, device: device)

        // Allocate swizzled destination and repack in place.
        let Wsw = device.makeBuffer(length: Wraw.length, options: .storageModeShared)!
        repackQ80ToSwizzled(src: Wraw, dst: Wsw, Din: Din, Dout: Dout)

        // Build a small test input (B=1 rows of 2816 halves, seeded random).
        let Bv = 1
        let xBuf = halfBuf(Bv * Din, seed: 0xC0FE)
        let yBuf = emptyHalf(Bv * Dout)

        // Run v6 kernel (note: encGemvQ80V6 uses the module-level B; drive
        // it manually so we can use Bv=1 here).
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(denseQ80V6PSO)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(Wsw, offset: 0, index: 1)
        enc.setBuffer(yBuf, offset: 0, index: 2)
        var du = UInt32(Din), dou = UInt32(Dout)
        enc.setBytes(&du, length: 4, index: 3)
        enc.setBytes(&dou, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: Bv, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        // CPU reference: direct Q8_0 dequant + dot, over the raw (un-swizzled)
        // GGUF bytes. Iterates 32-element blocks along D_in, accumulating in
        // float (kernel uses float too — no precision drift).
        let x = xBuf.contents().assumingMemoryBound(to: Float16.self)
        let wRaw = Wraw.contents()
        let nbc = Din / 32
        let BLK = 34
        let colBytes = nbc * BLK

        var maxAbsDiff: Float = 0
        var maxRelDiff: Float = 0
        var refNorm: Float = 0
        var nChecked = 0
        let checkN = min(Dout, 256)   // check first 256 output cols
        for n in 0..<checkN {
            var acc: Float = 0
            let colBase = n * colBytes
            for kb in 0..<nbc {
                let blkOff = colBase + kb * BLK
                let dBits = wRaw.load(fromByteOffset: blkOff, as: Float16.self)
                let dFloat = Float(dBits)
                let baseK = kb * 32
                for p in 0..<32 {
                    let qsByte = wRaw.load(fromByteOffset: blkOff + 2 + p, as: Int8.self)
                    let xVal = Float(x[baseK + p])
                    acc += xVal * dFloat * Float(qsByte)
                }
            }
            let ref = acc
            let got = Float(yBuf.contents().load(fromByteOffset: n * 2, as: Float16.self))
            let diff = abs(got - ref)
            let rel = diff / max(abs(ref), 1e-6)
            maxAbsDiff = max(maxAbsDiff, diff)
            maxRelDiff = max(maxRelDiff, rel)
            refNorm += ref * ref
            nChecked += 1
        }
        refNorm = sqrt(refNorm / Float(nChecked))
        print(String(format: "  checked first %d outputs: max abs diff %.4e, max rel diff %.3e, ||ref||_rms %.3e",
                     nChecked, maxAbsDiff, maxRelDiff, refNorm))

        if maxRelDiff < 0.01 {
            print("  ✓ GGUF load + swizzle repack + v6 kernel match reference within 1%")
        } else {
            print("  ✗ divergence above 1% — likely repack or kernel bug")
        }
    } catch {
        print("  GGUF validate failed: \(error)")
    }
}

// Phase 2a unit test: synthesize Q/K/V for a single slot and single Q tile,
// dispatch flex_attn_slide_v1_q8, write inputs + output to npy files so a
// Python reference can verify correctness. B_test=1, Q_LEN=8, H_Q=16, H_KV=8,
// HD=256. Sliding window disabled. Causal mask via q_positions=[0..7],
// k_len=8 (one page of K).
//
// Env: FLEX_ATTN_TEST=<outdir>
func runFlexAttnSlideV1Test(outDir: String) {
    print("\n=== flex_attn_slide_v1_q8 unit test ===")
    let H_Q = 16, H_KV = 8, D = 256, PAGE = 16, Q_LEN = 8
    let B_test = 1
    let qElems  = B_test * Q_LEN * H_Q * D
    let kvElems = PAGE * H_KV * D                 // one physical page
    let oElems  = qElems

    // Synth Q, K, V with deterministic pseudorandom fp16 values.
    func make(_ n: Int, seed: UInt32, scale: Float = 0.02) -> MTLBuffer {
        let buf = device.makeBuffer(length: n * 2, options: .storageModeShared)!
        let p = buf.contents().assumingMemoryBound(to: Float16.self)
        var s = seed
        for i in 0..<n {
            s = s &* 1664525 &+ 1013904223
            let r = Float(Int32(bitPattern: s) & 0xFFFF) / 65535.0 - 0.5
            p[i] = Float16(r * scale)
        }
        return buf
    }
    let Q   = make(qElems,  seed: 0xA001, scale: 1.0)
    let Kc  = make(kvElems, seed: 0xB001, scale: 1.0)
    let Vc  = make(kvElems, seed: 0xC001, scale: 1.0)
    let O   = device.makeBuffer(length: oElems * 2, options: .storageModeShared)!
    memset(O.contents(), 0, O.length)

    // Test-specific partials (big enough for Q_LEN * H_Q, N_SPLITS=1).
    let testNSplits = 1
    let mP = device.makeBuffer(length: B_test * Q_LEN * H_Q * testNSplits * 4, options: .storageModeShared)!
    let lP = device.makeBuffer(length: B_test * Q_LEN * H_Q * testNSplits * 4, options: .storageModeShared)!
    let oP = device.makeBuffer(length: B_test * Q_LEN * H_Q * testNSplits * D * 4, options: .storageModeShared)!

    // Block table: slot 0 → phys page 0.
    let bt = device.makeBuffer(length: B_test * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!
    let btP = bt.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0..<(B_test * MAX_PAGES_PER_SLOT) { btP[i] = 0 }
    // q_positions: [0, 1, 2, 3, 4, 5, 6, 7]
    let qPos = device.makeBuffer(length: B_test * Q_LEN * 4, options: .storageModeShared)!
    let qPosP = qPos.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0..<Q_LEN { qPosP[i] = UInt32(i) }
    // k_len = Q_LEN (just the one page of K is active so far).
    let kLen = device.makeBuffer(length: B_test * 4, options: .storageModeShared)!
    kLen.contents().assumingMemoryBound(to: UInt32.self)[0] = UInt32(Q_LEN)

    // Block mask (Q_BLOCKS=1 since Q_LEN==Q_BLOCK=8): one K block, non-empty.
    // With causal only (no window): block K=0 has k_pos 0..15, max q_pos=7.
    // k_pos > q_pos for k in 8..15, so block spans valid+invalid → PARTIAL.
    let fullOff = device.makeBuffer(length: (B_test + 1) * 4, options: .storageModeShared)!
    let partOff = device.makeBuffer(length: (B_test + 1) * 4, options: .storageModeShared)!
    let fullIdx = device.makeBuffer(length: 1 * 4, options: .storageModeShared)!   // empty
    let partIdx = device.makeBuffer(length: 1 * 4, options: .storageModeShared)!
    fullOff.contents().assumingMemoryBound(to: UInt32.self)[0] = 0
    fullOff.contents().assumingMemoryBound(to: UInt32.self)[1] = 0   // 0 full blocks
    partOff.contents().assumingMemoryBound(to: UInt32.self)[0] = 0
    partOff.contents().assumingMemoryBound(to: UInt32.self)[1] = 1   // 1 partial block
    partIdx.contents().assumingMemoryBound(to: UInt32.self)[0] = 0   // block index 0

    // Dispatch.
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(flexAttnSlideV1Q8PSO)
    enc.setBuffer(Q,   offset: 0, index: 0); enc.setBuffer(Kc, offset: 0, index: 1)
    enc.setBuffer(Vc,  offset: 0, index: 2); enc.setBuffer(bt, offset: 0, index: 3)
    enc.setBuffer(mP,  offset: 0, index: 4)
    enc.setBuffer(lP,  offset: 0, index: 5)
    enc.setBuffer(oP,  offset: 0, index: 6)
    enc.setBuffer(fullOff, offset: 0, index: 7)
    enc.setBuffer(fullIdx, offset: 0, index: 8)
    enc.setBuffer(partOff, offset: 0, index: 9)
    enc.setBuffer(partIdx, offset: 0, index: 10)
    enc.setBuffer(qPos, offset: 0, index: 11)
    enc.setBuffer(kLen, offset: 0, index: 12)
    var scale: Float = 1.0
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV)
    var ns = UInt32(testNSplits), sw = UInt32(0), ql = UInt32(Q_LEN)
    enc.setBytes(&scale, length: 4, index: 13)
    enc.setBytes(&mv,    length: 4, index: 14)
    enc.setBytes(&hq,    length: 4, index: 15)
    enc.setBytes(&hkv,   length: 4, index: 16)
    enc.setBytes(&ns,    length: 4, index: 17)
    enc.setBytes(&sw,    length: 4, index: 18)
    enc.setBytes(&ql,    length: 4, index: 19)
    // Grid: (B_test * H_KV, Q_blocks=1, N_SPLITS=1) TGs of 32 lanes each.
    enc.dispatchThreadgroups(MTLSize(width: B_test * H_KV, height: 1, depth: testNSplits),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()

    // Reduce partials → O. Using existing paged_attn_split_reduce with N_SPLITS=1.
    let enc2 = cb.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(pagedSplitReducePSO)
    enc2.setBuffer(mP, offset: 0, index: 0); enc2.setBuffer(lP, offset: 0, index: 1)
    enc2.setBuffer(oP, offset: 0, index: 2); enc2.setBuffer(O,  offset: 0, index: 3)
    var Dv = UInt32(D)
    enc2.setBytes(&Dv, length: 4, index: 4); enc2.setBytes(&ns, length: 4, index: 5)
    enc2.dispatchThreadgroups(MTLSize(width: B_test * Q_LEN * H_Q, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc2.endEncoding()

    cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { print("  GPU error: \(err)"); return }

    // Write Q, K (unswizzled page 0 as [PAGE, H_KV, D]), V, O to disk as fp32 npy.
    func dumpHalf(_ buf: MTLBuffer, count: Int, path: String, shape: [Int]) {
        var fp32 = [Float](repeating: 0, count: count)
        let src = buf.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<count { fp32[i] = Float(src[i]) }
        fp32.withUnsafeBufferPointer { bp in
            writeNpyFloat32(path, data: bp.baseAddress!, shape: shape)
        }
    }
    dumpHalf(Q,  count: qElems,  path: "\(outDir)/flex_test_Q.npy",  shape: [B_test, Q_LEN, H_Q, D])
    dumpHalf(Kc, count: kvElems, path: "\(outDir)/flex_test_K.npy",  shape: [PAGE, H_KV, D])
    dumpHalf(Vc, count: kvElems, path: "\(outDir)/flex_test_V.npy",  shape: [PAGE, H_KV, D])
    dumpHalf(O,  count: oElems,  path: "\(outDir)/flex_test_O_kernel.npy", shape: [B_test, Q_LEN, H_Q, D])
    print("  wrote flex_test_{Q,K,V,O_kernel}.npy to \(outDir)")
    print("  verify with: .venv/bin/python flex_attn_verify.py")
}

// Attention-kernel latency & bandwidth microbench.
// Fires on `ATTN_BENCH=1`. Dispatches each of the four flex attention kernels
// (slide AR v0, slide prefill v1_q8, full AR v0, full prefill) at a sweep of
// (k_len, Q_LEN) configurations, reports wall ms + achieved bandwidth.
// B=1 to isolate single-stream perf; the cache is allocated per-variant so we
// don't touch the LM loader's KV memory.

private struct AttnBenchVariant {
    let name: String
    let slide: Bool          // D=256 PAGE=16 H_KV=8 H_Q=16, else D=512 PAGE=8 H_KV=2 H_Q=16
    let prefill: Bool        // true → multi-Q prefill kernel, false → AR v0
}

private func attnBandwidthBytesPerCall(variant: AttnBenchVariant, kLen: Int, qLen: Int, slidingWindow: Int) -> Int {
    // Per-call DRAM traffic in bytes (B=1):
    //   Q_read : Q_LEN * H_Q * D * 2
    //   K_read : k_eff  * H_KV * D * 2
    //   V_read : k_eff  * H_KV * D * 2
    //   O_write: Q_LEN * H_Q * D * 2
    // k_eff = min(k_len, sliding_window) for slide layers, else k_len.
    let H_Q = 16
    let H_KV = variant.slide ? 8 : 2
    let D = variant.slide ? 256 : 512
    let k_eff = variant.slide ? min(kLen, slidingWindow) : kLen
    let qBytes = qLen * H_Q  * D * 2
    let kBytes = k_eff * H_KV * D * 2
    let vBytes = k_eff * H_KV * D * 2
    let oBytes = qLen * H_Q  * D * 2
    return qBytes + kBytes + vBytes + oBytes
}

func runAttnBench() {
    print("\n=== Attention kernel latency + bandwidth sweep (B=1, M5 Max) ===")
    let B_b = 1
    let iterations = 50
    let warmup = 5
    let slidingWindow = 1024

    let variants: [AttnBenchVariant] = [
        AttnBenchVariant(name: "slide_ar",      slide: true,  prefill: false),
        AttnBenchVariant(name: "slide_prefill", slide: true,  prefill: true),
        AttnBenchVariant(name: "full_ar",       slide: false, prefill: false),
        AttnBenchVariant(name: "full_prefill",  slide: false, prefill: true),
    ]
    let kLens = [1024, 4096, 8192, 16384]
    let qLensPerVariant: [String: [Int]] = [
        "slide_ar":      [1],
        "slide_prefill": [8],
        "full_ar":       [1],
        "full_prefill":  [8],
    ]

    // Max-sized cache: PAGE=8 (full) × 2048 pages = 16k tokens × H_KV=8 (slide max) × D=512 (full max).
    // Reuse one big buffer; slide variants index it with PAGE=16 geometry so phys_page 0
    // actually spans two "full" pages. We allocate per-variant caches so layouts match kernels.
    let H_Q = 16
    let maxPagesSlide = 16384 / 16   // 1024
    let maxPagesFull  = 16384 / 8    // 2048

    func makeHalfBuf(_ n: Int, seed: UInt32) -> MTLBuffer {
        let buf = device.makeBuffer(length: n * 2, options: .storageModeShared)!
        let p = buf.contents().assumingMemoryBound(to: Float16.self)
        var s = seed
        for i in 0..<n {
            s = s &* 1664525 &+ 1013904223
            let r = Float(Int32(bitPattern: s) & 0xFFFF) / 65535.0 - 0.5
            p[i] = Float16(r)
        }
        return buf
    }

    // Slide cache: [maxPagesSlide, PAGE=16, H_KV=8, D=256].
    let slideKc = makeHalfBuf(maxPagesSlide * 16 * 8 * 256, seed: 0x11)
    let slideVc = makeHalfBuf(maxPagesSlide * 16 * 8 * 256, seed: 0x12)
    // Full cache: [maxPagesFull, PAGE=8, H_KV=2, D=512].
    let fullKc = makeHalfBuf(maxPagesFull * 8 * 2 * 512, seed: 0x13)
    let fullVc = makeHalfBuf(maxPagesFull * 8 * 2 * 512, seed: 0x14)
    // Big Q / O buffers sized for Q_LEN=8, H_Q=16, D=512.
    let qMax = 8 * H_Q * 512
    let Q = makeHalfBuf(qMax, seed: 0x15)
    let O = device.makeBuffer(length: qMax * 2, options: .storageModeShared)!

    // Per-variant block_table, big enough for kLen=16k.
    let btSlide = device.makeBuffer(length: B_b * maxPagesSlide * 4, options: .storageModeShared)!
    let btFull  = device.makeBuffer(length: B_b * maxPagesFull  * 4, options: .storageModeShared)!
    for buf in [btSlide, btFull] {
        let p = buf.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<(buf.length / 4) { p[i] = UInt32(i) }
    }

    // kLen buffer.
    let kLenBuf = device.makeBuffer(length: B_b * 4, options: .storageModeShared)!
    // q_positions buffer (for prefill kernels).
    let qPosBuf = device.makeBuffer(length: B_b * 8 * 4, options: .storageModeShared)!
    // Attn partials sized for Q_LEN=8, H_Q=16, N_SPLITS=ATTN_N_SPLITS, D=512.
    let mP = device.makeBuffer(length: B_b * 8 * H_Q * ATTN_N_SPLITS * 4, options: .storageModeShared)!
    let lP = device.makeBuffer(length: B_b * 8 * H_Q * ATTN_N_SPLITS * 4, options: .storageModeShared)!
    let oP = device.makeBuffer(length: B_b * 8 * H_Q * ATTN_N_SPLITS * 512 * 4, options: .storageModeShared)!
    // Block-mask buffers (CSR): one Q-block per slot at Q_LEN=8 or 1. Full list empty,
    // partial list has all non-empty K blocks up to kLen.
    let maxBlocks = max(maxPagesSlide, maxPagesFull)
    let fullOff = device.makeBuffer(length: (B_b + 1) * 4, options: .storageModeShared)!
    let partOff = device.makeBuffer(length: (B_b + 1) * 4, options: .storageModeShared)!
    let fullIdx = device.makeBuffer(length: maxBlocks * 4, options: .storageModeShared)!
    let partIdx = device.makeBuffer(length: maxBlocks * 4, options: .storageModeShared)!

    // Formatting helpers.
    func fmtRow(_ cells: [String], widths: [Int]) -> String {
        var s = ""
        for (i, c) in cells.enumerated() { s += c.padding(toLength: widths[i], withPad: " ", startingAt: 0) }
        return s
    }
    let colW = [16, 10, 12, 14, 14, 12]
    print(fmtRow(["variant", "k_len", "Q_LEN", "wall ms", "BW GB/s", "bytes/call"], widths: colW))
    print(String(repeating: "-", count: colW.reduce(0, +)))

    for v in variants {
        let qLens = qLensPerVariant[v.name] ?? [1]
        let D = v.slide ? 256 : 512
        let PAGE = v.slide ? 16 : 8
        let H_KV = v.slide ? 8 : 2
        let Kc = v.slide ? slideKc : fullKc
        let Vc = v.slide ? slideVc : fullVc
        let bt = v.slide ? btSlide : btFull
        let maxPages = v.slide ? maxPagesSlide : maxPagesFull

        for kLen in kLens {
            for qLen in qLens {
                // k_len (= valid entries)
                kLenBuf.contents().assumingMemoryBound(to: UInt32.self)[0] = UInt32(kLen)
                // q_positions for prefill: the last qLen positions. For AR: single position = kLen-1.
                let qStart = kLen - qLen
                let qp = qPosBuf.contents().assumingMemoryBound(to: UInt32.self)
                for i in 0..<qLen { qp[i] = UInt32(qStart + i) }
                // Block mask: all k blocks in [0, kLen) that intersect the window+causal are PARTIAL.
                // For speed, mark them all PARTIAL (conservative; fine for perf bench).
                let k_blocks = (kLen + PAGE - 1) / PAGE
                let fOff = fullOff.contents().assumingMemoryBound(to: UInt32.self)
                let pOff = partOff.contents().assumingMemoryBound(to: UInt32.self)
                let pIdx = partIdx.contents().assumingMemoryBound(to: UInt32.self)
                fOff[0] = 0; fOff[1] = 0
                pOff[0] = 0
                // With sliding window on slide kernels, skip K blocks before window_lo.
                let windowLo = (v.slide && kLen > slidingWindow) ? (kLen - slidingWindow) : 0
                var pc = 0
                for k in 0..<k_blocks {
                    let lo = k * PAGE, hi = lo + PAGE - 1
                    if hi < windowLo { continue }
                    if lo >= kLen { continue }
                    pIdx[pc] = UInt32(k); pc += 1
                }
                pOff[1] = UInt32(pc)

                // Warmup + timed runs.
                for _ in 0..<warmup {
                    let cb = queue.makeCommandBuffer()!
                    dispatchAttnBench(cb: cb, v: v, Q: Q, O: O, Kc: Kc, Vc: Vc, bt: bt,
                                       mP: mP, lP: lP, oP: oP, kLenBuf: kLenBuf, qPosBuf: qPosBuf,
                                       fullOff: fullOff, fullIdx: fullIdx, partOff: partOff, partIdx: partIdx,
                                       B: B_b, maxPages: maxPages, H_Q: H_Q, H_KV: H_KV, D: D,
                                       qLen: qLen, slidingWindow: slidingWindow)
                    cb.commit(); cb.waitUntilCompleted()
                }

                // Timed.
                let t0 = Date()
                for _ in 0..<iterations {
                    let cb = queue.makeCommandBuffer()!
                    dispatchAttnBench(cb: cb, v: v, Q: Q, O: O, Kc: Kc, Vc: Vc, bt: bt,
                                       mP: mP, lP: lP, oP: oP, kLenBuf: kLenBuf, qPosBuf: qPosBuf,
                                       fullOff: fullOff, fullIdx: fullIdx, partOff: partOff, partIdx: partIdx,
                                       B: B_b, maxPages: maxPages, H_Q: H_Q, H_KV: H_KV, D: D,
                                       qLen: qLen, slidingWindow: slidingWindow)
                    cb.commit(); cb.waitUntilCompleted()
                }
                let wallMs = Date().timeIntervalSince(t0) * 1000.0 / Double(iterations)
                let bytes = attnBandwidthBytesPerCall(variant: v, kLen: kLen, qLen: qLen, slidingWindow: slidingWindow)
                let gbps = Double(bytes) / (wallMs / 1000.0) / 1e9
                print(fmtRow([
                    v.name,
                    "\(kLen)",
                    "\(qLen)",
                    String(format: "%.3f", wallMs),
                    String(format: "%.1f", gbps),
                    String(format: "%d", bytes),
                ], widths: colW))
            }
        }
        print("")
    }
}

private func dispatchAttnBench(cb: MTLCommandBuffer, v: AttnBenchVariant,
                                Q: MTLBuffer, O: MTLBuffer, Kc: MTLBuffer, Vc: MTLBuffer,
                                bt: MTLBuffer, mP: MTLBuffer, lP: MTLBuffer, oP: MTLBuffer,
                                kLenBuf: MTLBuffer, qPosBuf: MTLBuffer,
                                fullOff: MTLBuffer, fullIdx: MTLBuffer,
                                partOff: MTLBuffer, partIdx: MTLBuffer,
                                B: Int, maxPages: Int,
                                H_Q: Int, H_KV: Int, D: Int, qLen: Int,
                                slidingWindow: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    switch v.name {
    case "slide_ar":
        enc.setComputePipelineState(flexAttnSlideV0PSO)
        enc.setBuffer(Q, offset: 0, index: 0); enc.setBuffer(Kc, offset: 0, index: 1)
        enc.setBuffer(Vc, offset: 0, index: 2); enc.setBuffer(bt, offset: 0, index: 3)
        enc.setBuffer(mP, offset: 0, index: 4); enc.setBuffer(lP, offset: 0, index: 5)
        enc.setBuffer(oP, offset: 0, index: 6)
        enc.setBuffer(fullOff, offset: 0, index: 7); enc.setBuffer(fullIdx, offset: 0, index: 8)
        enc.setBuffer(partOff, offset: 0, index: 9); enc.setBuffer(partIdx, offset: 0, index: 10)
        enc.setBuffer(kLenBuf, offset: 0, index: 11)
        var sc: Float = 1.0, mv = UInt32(maxPages), hq = UInt32(H_Q), hkv = UInt32(H_KV), ns = UInt32(ATTN_N_SPLITS), sw = UInt32(slidingWindow)
        enc.setBytes(&sc, length: 4, index: 12); enc.setBytes(&mv, length: 4, index: 13)
        enc.setBytes(&hq, length: 4, index: 14); enc.setBytes(&hkv, length: 4, index: 15)
        enc.setBytes(&ns, length: 4, index: 16); enc.setBytes(&sw, length: 4, index: 17)
        enc.dispatchThreadgroups(MTLSize(width: B * H_KV, height: ATTN_N_SPLITS, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    case "slide_prefill":
        enc.setComputePipelineState(flexAttnSlideV1Q8PSO)
        enc.setBuffer(Q, offset: 0, index: 0); enc.setBuffer(Kc, offset: 0, index: 1)
        enc.setBuffer(Vc, offset: 0, index: 2); enc.setBuffer(bt, offset: 0, index: 3)
        enc.setBuffer(mP, offset: 0, index: 4); enc.setBuffer(lP, offset: 0, index: 5)
        enc.setBuffer(oP, offset: 0, index: 6)
        enc.setBuffer(fullOff, offset: 0, index: 7); enc.setBuffer(fullIdx, offset: 0, index: 8)
        enc.setBuffer(partOff, offset: 0, index: 9); enc.setBuffer(partIdx, offset: 0, index: 10)
        enc.setBuffer(qPosBuf, offset: 0, index: 11); enc.setBuffer(kLenBuf, offset: 0, index: 12)
        var sc: Float = 1.0, mv = UInt32(maxPages), hq = UInt32(H_Q), hkv = UInt32(H_KV)
        var ns = UInt32(ATTN_N_SPLITS), sw = UInt32(slidingWindow), ql = UInt32(qLen)
        enc.setBytes(&sc, length: 4, index: 13); enc.setBytes(&mv, length: 4, index: 14)
        enc.setBytes(&hq, length: 4, index: 15); enc.setBytes(&hkv, length: 4, index: 16)
        enc.setBytes(&ns, length: 4, index: 17); enc.setBytes(&sw, length: 4, index: 18)
        enc.setBytes(&ql, length: 4, index: 19)
        enc.dispatchThreadgroups(MTLSize(width: B * H_KV, height: 1, depth: ATTN_N_SPLITS),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    case "full_ar":
        enc.setComputePipelineState(flexAttnFullV0PSO)
        enc.setBuffer(Q, offset: 0, index: 0); enc.setBuffer(Kc, offset: 0, index: 1)
        enc.setBuffer(Vc, offset: 0, index: 2); enc.setBuffer(bt, offset: 0, index: 3)
        enc.setBuffer(mP, offset: 0, index: 4); enc.setBuffer(lP, offset: 0, index: 5)
        enc.setBuffer(oP, offset: 0, index: 6)
        enc.setBuffer(fullOff, offset: 0, index: 7); enc.setBuffer(fullIdx, offset: 0, index: 8)
        enc.setBuffer(partOff, offset: 0, index: 9); enc.setBuffer(partIdx, offset: 0, index: 10)
        enc.setBuffer(kLenBuf, offset: 0, index: 11)
        var sc: Float = 1.0, mv = UInt32(maxPages), hq = UInt32(H_Q), hkv = UInt32(H_KV), ns = UInt32(ATTN_N_SPLITS)
        enc.setBytes(&sc, length: 4, index: 12); enc.setBytes(&mv, length: 4, index: 13)
        enc.setBytes(&hq, length: 4, index: 14); enc.setBytes(&hkv, length: 4, index: 15)
        enc.setBytes(&ns, length: 4, index: 16)
        enc.dispatchThreadgroups(MTLSize(width: B * H_KV, height: ATTN_N_SPLITS, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    case "full_prefill":
        enc.setComputePipelineState(flexAttnFullPrefillPSO)
        enc.setBuffer(Q, offset: 0, index: 0); enc.setBuffer(Kc, offset: 0, index: 1)
        enc.setBuffer(Vc, offset: 0, index: 2); enc.setBuffer(bt, offset: 0, index: 3)
        enc.setBuffer(mP, offset: 0, index: 4); enc.setBuffer(lP, offset: 0, index: 5)
        enc.setBuffer(oP, offset: 0, index: 6)
        enc.setBuffer(fullOff, offset: 0, index: 7); enc.setBuffer(fullIdx, offset: 0, index: 8)
        enc.setBuffer(partOff, offset: 0, index: 9); enc.setBuffer(partIdx, offset: 0, index: 10)
        enc.setBuffer(qPosBuf, offset: 0, index: 11); enc.setBuffer(kLenBuf, offset: 0, index: 12)
        var sc: Float = 1.0, mv = UInt32(maxPages), hq = UInt32(H_Q), hkv = UInt32(H_KV)
        var ns = UInt32(ATTN_N_SPLITS), ql = UInt32(qLen)
        enc.setBytes(&sc, length: 4, index: 13); enc.setBytes(&mv, length: 4, index: 14)
        enc.setBytes(&hq, length: 4, index: 15); enc.setBytes(&hkv, length: 4, index: 16)
        enc.setBytes(&ns, length: 4, index: 17); enc.setBytes(&ql, length: 4, index: 18)
        enc.dispatchThreadgroups(MTLSize(width: B * H_Q, height: 1, depth: ATTN_N_SPLITS),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    default: break
    }
    enc.endEncoding()
}

// Phase 2b full-attn prefill unit test. B_test=1, Q_LEN=8, H_Q=16, H_KV=2,
// HD=512. Grid: (B*H_Q, 1, 1) = 16 TGs, each owns one q_head's 8 queries.
// kv_head = (q_head * H_KV) / H_Q = q_head / 8.
func runFlexAttnFullPrefillTest(outDir: String) {
    print("\n=== flex_attn_full_prefill unit test ===")
    let H_Q = 16, H_KV = 2, D = 512, PAGE = 8, Q_LEN = 8
    let B_test = 1
    let qElems  = B_test * Q_LEN * H_Q * D
    let kvElems = PAGE * H_KV * D
    let oElems  = qElems

    func make(_ n: Int, seed: UInt32) -> MTLBuffer {
        let buf = device.makeBuffer(length: n * 2, options: .storageModeShared)!
        let p = buf.contents().assumingMemoryBound(to: Float16.self)
        var s = seed
        for i in 0..<n {
            s = s &* 1664525 &+ 1013904223
            let r = Float(Int32(bitPattern: s) & 0xFFFF) / 65535.0 - 0.5
            p[i] = Float16(r)
        }
        return buf
    }
    let Q   = make(qElems,  seed: 0xD001)
    let Kc  = make(kvElems, seed: 0xE001)
    let Vc  = make(kvElems, seed: 0xF001)
    let O   = device.makeBuffer(length: oElems * 2, options: .storageModeShared)!
    memset(O.contents(), 0, O.length)

    let testNSplits = 1
    let mP = device.makeBuffer(length: B_test * Q_LEN * H_Q * testNSplits * 4, options: .storageModeShared)!
    let lP = device.makeBuffer(length: B_test * Q_LEN * H_Q * testNSplits * 4, options: .storageModeShared)!
    let oP = device.makeBuffer(length: B_test * Q_LEN * H_Q * testNSplits * D * 4, options: .storageModeShared)!

    let bt = device.makeBuffer(length: B_test * MAX_PAGES_PER_SLOT * 4, options: .storageModeShared)!
    let btP = bt.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0..<(B_test * MAX_PAGES_PER_SLOT) { btP[i] = 0 }
    let qPos = device.makeBuffer(length: B_test * Q_LEN * 4, options: .storageModeShared)!
    let qPosP = qPos.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0..<Q_LEN { qPosP[i] = UInt32(i) }
    let kLen = device.makeBuffer(length: B_test * 4, options: .storageModeShared)!
    kLen.contents().assumingMemoryBound(to: UInt32.self)[0] = UInt32(Q_LEN)

    // Block mask: 1 K block, PARTIAL (Q_LEN=8 queries need causal masking).
    let fullOff = device.makeBuffer(length: (B_test + 1) * 4, options: .storageModeShared)!
    let partOff = device.makeBuffer(length: (B_test + 1) * 4, options: .storageModeShared)!
    let fullIdx = device.makeBuffer(length: 1 * 4, options: .storageModeShared)!
    let partIdx = device.makeBuffer(length: 1 * 4, options: .storageModeShared)!
    fullOff.contents().assumingMemoryBound(to: UInt32.self)[0] = 0
    fullOff.contents().assumingMemoryBound(to: UInt32.self)[1] = 0
    partOff.contents().assumingMemoryBound(to: UInt32.self)[0] = 0
    partOff.contents().assumingMemoryBound(to: UInt32.self)[1] = 1
    partIdx.contents().assumingMemoryBound(to: UInt32.self)[0] = 0

    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(flexAttnFullPrefillPSO)
    enc.setBuffer(Q,   offset: 0, index: 0); enc.setBuffer(Kc, offset: 0, index: 1)
    enc.setBuffer(Vc,  offset: 0, index: 2); enc.setBuffer(bt, offset: 0, index: 3)
    enc.setBuffer(mP,  offset: 0, index: 4)
    enc.setBuffer(lP,  offset: 0, index: 5)
    enc.setBuffer(oP,  offset: 0, index: 6)
    enc.setBuffer(fullOff, offset: 0, index: 7)
    enc.setBuffer(fullIdx, offset: 0, index: 8)
    enc.setBuffer(partOff, offset: 0, index: 9)
    enc.setBuffer(partIdx, offset: 0, index: 10)
    enc.setBuffer(qPos, offset: 0, index: 11)
    enc.setBuffer(kLen, offset: 0, index: 12)
    var scale: Float = 1.0
    var mv = UInt32(MAX_PAGES_PER_SLOT), hq = UInt32(H_Q), hkv = UInt32(H_KV)
    var ns = UInt32(testNSplits), ql = UInt32(Q_LEN)
    enc.setBytes(&scale, length: 4, index: 13)
    enc.setBytes(&mv,    length: 4, index: 14)
    enc.setBytes(&hq,    length: 4, index: 15)
    enc.setBytes(&hkv,   length: 4, index: 16)
    enc.setBytes(&ns,    length: 4, index: 17)
    enc.setBytes(&ql,    length: 4, index: 18)
    // Grid: (B_test * H_Q, 1 q_block, 1 split)
    enc.dispatchThreadgroups(MTLSize(width: B_test * H_Q, height: 1, depth: testNSplits),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()

    let enc2 = cb.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(pagedSplitReducePSO)
    enc2.setBuffer(mP, offset: 0, index: 0); enc2.setBuffer(lP, offset: 0, index: 1)
    enc2.setBuffer(oP, offset: 0, index: 2); enc2.setBuffer(O,  offset: 0, index: 3)
    var Dv = UInt32(D)
    enc2.setBytes(&Dv, length: 4, index: 4); enc2.setBytes(&ns, length: 4, index: 5)
    enc2.dispatchThreadgroups(MTLSize(width: B_test * Q_LEN * H_Q, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc2.endEncoding()

    cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { print("  GPU error: \(err)"); return }

    func dumpHalf(_ buf: MTLBuffer, count: Int, path: String, shape: [Int]) {
        var fp32 = [Float](repeating: 0, count: count)
        let src = buf.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<count { fp32[i] = Float(src[i]) }
        fp32.withUnsafeBufferPointer { bp in
            writeNpyFloat32(path, data: bp.baseAddress!, shape: shape)
        }
    }
    dumpHalf(Q,  count: qElems,  path: "\(outDir)/flex_full_test_Q.npy",  shape: [B_test, Q_LEN, H_Q, D])
    dumpHalf(Kc, count: kvElems, path: "\(outDir)/flex_full_test_K.npy",  shape: [PAGE, H_KV, D])
    dumpHalf(Vc, count: kvElems, path: "\(outDir)/flex_full_test_V.npy",  shape: [PAGE, H_KV, D])
    dumpHalf(O,  count: oElems,  path: "\(outDir)/flex_full_test_O_kernel.npy", shape: [B_test, Q_LEN, H_Q, D])
    print("  wrote flex_full_test_{Q,K,V,O_kernel}.npy to \(outDir)")
}

// Dump swizzled per-expert bytes for layer-0 MoE weights. Byte-level probe:
// Python can inverse-swizzle + dequantize and compare to HF to detect whether
// our loadMoESwizzled layout matches the kernel's assumed layout.
// Env: LM_DUMP_EXPERT_W=<outDir>, optional LM_DUMP_EXPERT_ID=<int, default 52>.
func runDumpExpertWeights(ggufPath: String, expertId: Int, outDir: String) {
    print("\n=== Layer-0 MoE expert-weight byte dump ===")
    print("  expert id: \(expertId)")
    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    let lw = w.layers[0]

    // Q4_K gate_up: Din=HIDDEN=2816, Dout=2*MOE_INT=1408*... wait MOE_INT=704, so
    // Dout here = 2 * MOE_INT_FUSED = 1408. blkElems=256, blkBytes=144.
    // Per-expert bytes = Dout * (Din/blkElems) * blkBytes = 1408 * 11 * 144 = 2,230,272.
    do {
        let Din = HIDDEN, Dout = 2 * MOE_INT, blkElems = 256, blkBytes = 144
        let nbc = Din / blkElems
        let perExpert = Dout * nbc * blkBytes
        precondition(expertId >= 0 && expertId < E_EXP)
        let src = lw.moeGateUp.contents().advanced(by: expertId * perExpert)
        let data = Data(bytes: src, count: perExpert)
        let path = "\(outDir)/lm_swift_l0_expert\(expertId)_gate_up_swizzled.bin"
        try? data.write(to: URL(fileURLWithPath: path))
        print("  wrote \(path) (\(perExpert) bytes = Dout=\(Dout) × nbc=\(nbc) × blkBytes=\(blkBytes))")
    }

    // Q5_1 down: Din=MOE_INT=704, Dout=HIDDEN=2816, blkElems=32, blkBytes=24.
    // Per-expert bytes = 2816 * (704/32) * 24 = 2816 * 22 * 24 = 1,486,848.
    do {
        let Din = MOE_INT, Dout = HIDDEN, blkElems = 32, blkBytes = 24
        let nbc = Din / blkElems
        let perExpert = Dout * nbc * blkBytes
        let src = lw.moeDown.contents().advanced(by: expertId * perExpert)
        let data = Data(bytes: src, count: perExpert)
        let path = "\(outDir)/lm_swift_l0_expert\(expertId)_down_swizzled.bin"
        try? data.write(to: URL(fileURLWithPath: path))
        print("  wrote \(path) (\(perExpert) bytes = Dout=\(Dout) × nbc=\(nbc) × blkBytes=\(blkBytes))")
    }
}

// Minimal .npy v1 writer for f32/i32 arrays. Header padded so data starts on a
// 64-byte boundary per the numpy spec; trailing 0x0a newline terminates the
// header. Blindly trusts the caller.
private func writeNpyImpl(_ path: String, descr: String, raw: UnsafePointer<UInt8>,
                          byteCount: Int, shape: [Int]) {
    var headerStr = "{'descr': '\(descr)', 'fortran_order': False, 'shape': ("
    for s in shape { headerStr += "\(s), " }
    headerStr += "), }"
    let preamble = 10
    while (preamble + headerStr.count + 1) % 64 != 0 { headerStr += " " }
    headerStr += "\n"
    var file = Data()
    file.append(contentsOf: [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])
    file.append(contentsOf: [0x01, 0x00])
    let hdrLen = UInt16(headerStr.count).littleEndian
    file.append(UInt8(hdrLen & 0xff))
    file.append(UInt8((hdrLen >> 8) & 0xff))
    file.append(headerStr.data(using: .ascii)!)
    file.append(UnsafeBufferPointer(start: raw, count: byteCount))
    try? file.write(to: URL(fileURLWithPath: path))
}

func writeNpyFloat32(_ path: String, data: UnsafePointer<Float>, shape: [Int]) {
    let count = shape.reduce(1, *)
    data.withMemoryRebound(to: UInt8.self, capacity: count * 4) { raw in
        writeNpyImpl(path, descr: "<f4", raw: raw, byteCount: count * 4, shape: shape)
    }
}

func writeNpyInt32(_ path: String, data: UnsafePointer<Int32>, shape: [Int]) {
    let count = shape.reduce(1, *)
    data.withMemoryRebound(to: UInt8.self, capacity: count * 4) { raw in
        writeNpyImpl(path, descr: "<i4", raw: raw, byteCount: count * 4, shape: shape)
    }
}

// Env-var driver: LM_DUMP_LAYERS=<outDir>  +  LM_KL_REF=<refDir>  +  LM_KL_TAG=<tag>  +  GGUF_PATH=<gguf>
// Runs the same AR prefill as runLmKLHarness (using the oracle's tokens file)
// but with LM_DUMP_STAGING allocated, so buildStepCB blits slot-0's residual
// at each of the NUM_LAYERS+1 boundaries. After each step, copies the
// staging snapshot (fp16) into an accumulator and writes a single
// lm_swift_hiddens.npy of shape [NUM_LAYERS+1, S, HIDDEN] fp32 for
// compare_lm_hiddens.py to diff against lm_<tag>_hiddens.npy.
func runLmLayerDump(ggufPath: String, refDir: String, tag: String, outDir: String) {
    print("\n=== LM per-layer hidden-state dump ===")
    let staging = LM_DUMP_STAGING
    let l0Staging = LM_DUMP_L0_STAGING
    if staging == nil && l0Staging == nil {
        print("  no dump staging allocated; set LM_DUMP_LAYERS and/or LM_DUMP_L0_INTERNALS")
        return
    }
    let tokensPath = "\(refDir)/lm_\(tag)_tokens.npy"
    guard let tokNpy = mmapNpy(tokensPath) else { print("  cannot open \(tokensPath)"); return }
    defer { tokNpy.release() }
    precondition(tokNpy.shape.count == 1, "tokens must be 1D")
    precondition(tokNpy.dtype == "<i4", "tokens must be int32")
    let S = tokNpy.shape[0]
    let toks: [UInt32] = {
        let p = tokNpy.ptr.advanced(by: tokNpy.dataOffset).assumingMemoryBound(to: Int32.self)
        return (0..<S).map { UInt32(bitPattern: p[$0]) }
    }()
    print("  S=\(S), NUM_LAYERS=\(NUM_LAYERS), HIDDEN=\(HIDDEN)")

    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    print("")

    // Accumulators. Layer-boundary dump: [NUM_LAYERS+1, S, HIDDEN] fp32.
    // L0 intra-layer probes: [5, S, HIDDEN] fp32 — slots 0-2 = post_attn+res,
    // ffw_norm_1 out, ffw_norm_2 out; slot 3 = pre_ffw_norm_2 out (experts in);
    // slot 4 = raw moe_sum pre-postFfn2Norm.
    // L0 router: expert_ids[S, TOPK] (uint32) + gate_w[S, TOPK] (float32).
    let nBoundaries = NUM_LAYERS + 1
    let nL0 = 5
    var hiddens = [Float](repeating: 0, count: nBoundaries * S * HIDDEN)
    var l0Probes = [Float](repeating: 0, count: nL0 * S * HIDDEN)
    let routerStaging = LM_DUMP_L0_ROUTER
    var l0ExpertIds = [UInt32](repeating: 0, count: S * TOPK)
    var l0GateW = [Float](repeating: 0, count: S * TOPK)
    var l0RouterLg = [Float](repeating: 0, count: S * E_EXP)
    var l0HiddenNorm = [Float](repeating: 0, count: S * HIDDEN)
    let attnStaging = LM_DUMP_L0_ATTN
    var l0AttnOut = [Float](repeating: 0, count: S * SLIDE_H * SLIDE_HD)
    let moeSlotsStaging = LM_DUMP_L0_MOE_SLOTS
    var l0MoeDownOut = [Float](repeating: 0, count: S * TOTAL_SLOTS * HIDDEN)
    var l0GateUpFused = [Float](repeating: 0, count: S * TOTAL_SLOTS * 2 * MOE_INT)
    var l0GateProj = [Float](repeating: 0, count: S * TOTAL_SLOTS * MOE_INT)
    var l0SlotToken = [UInt32](repeating: 0, count: S * TOTAL_SLOTS)
    var l0BatchSlots = [UInt32](repeating: 0, count: S * TOTAL_SLOTS)
    var l0GroupStart = [UInt32](repeating: 0, count: S * (E_EXP + 1))

    func readAfterCB(position p: Int) {
        if let st = staging {
            let src = st.contents().assumingMemoryBound(to: Float16.self)
            for L in 0..<nBoundaries {
                let srcBase = L * HIDDEN
                let dstBase = L * S * HIDDEN + p * HIDDEN
                for i in 0..<HIDDEN { hiddens[dstBase + i] = Float(src[srcBase + i]) }
            }
        }
        if let l0 = l0Staging {
            let src = l0.contents().assumingMemoryBound(to: Float16.self)
            for probe in 0..<nL0 {
                let srcBase = probe * HIDDEN
                let dstBase = probe * S * HIDDEN + p * HIDDEN
                for i in 0..<HIDDEN { l0Probes[dstBase + i] = Float(src[srcBase + i]) }
            }
        }
        if let r = routerStaging {
            let idsPtr = r.contents().assumingMemoryBound(to: UInt32.self)
            let wPtr = r.contents().advanced(by: TOPK * 4).assumingMemoryBound(to: Float.self)
            let lgPtr = r.contents().advanced(by: TOPK * 4 + TOPK * 4).assumingMemoryBound(to: Float16.self)
            let hnPtr = r.contents().advanced(by: TOPK * 4 + TOPK * 4 + E_EXP * 2).assumingMemoryBound(to: Float16.self)
            for k in 0..<TOPK {
                l0ExpertIds[p * TOPK + k] = idsPtr[k]
                l0GateW[p * TOPK + k] = wPtr[k]
            }
            for e in 0..<E_EXP { l0RouterLg[p * E_EXP + e] = Float(lgPtr[e]) }
            for i in 0..<HIDDEN { l0HiddenNorm[p * HIDDEN + i] = Float(hnPtr[i]) }
        }
        if let a = attnStaging {
            let p_src = a.contents().assumingMemoryBound(to: Float16.self)
            for i in 0..<(SLIDE_H * SLIDE_HD) {
                l0AttnOut[p * SLIDE_H * SLIDE_HD + i] = Float(p_src[i])
            }
        }
        if let m = moeSlotsStaging {
            let base = m.contents()
            var off = 0
            let downPtr = base.advanced(by: off).assumingMemoryBound(to: Float16.self)
            for i in 0..<(TOTAL_SLOTS * HIDDEN) {
                l0MoeDownOut[p * TOTAL_SLOTS * HIDDEN + i] = Float(downPtr[i])
            }
            off += TOTAL_SLOTS * HIDDEN * 2
            let gufPtr = base.advanced(by: off).assumingMemoryBound(to: Float16.self)
            for i in 0..<(TOTAL_SLOTS * 2 * MOE_INT) {
                l0GateUpFused[p * TOTAL_SLOTS * 2 * MOE_INT + i] = Float(gufPtr[i])
            }
            off += TOTAL_SLOTS * 2 * MOE_INT * 2
            let gpPtr = base.advanced(by: off).assumingMemoryBound(to: Float16.self)
            for i in 0..<(TOTAL_SLOTS * MOE_INT) {
                l0GateProj[p * TOTAL_SLOTS * MOE_INT + i] = Float(gpPtr[i])
            }
            off += TOTAL_SLOTS * MOE_INT * 2
            let stokPtr = base.advanced(by: off).assumingMemoryBound(to: UInt32.self)
            for s in 0..<TOTAL_SLOTS { l0SlotToken[p * TOTAL_SLOTS + s] = stokPtr[s] }
            off += TOTAL_SLOTS * 4
            let bsPtr = base.advanced(by: off).assumingMemoryBound(to: UInt32.self)
            for s in 0..<TOTAL_SLOTS { l0BatchSlots[p * TOTAL_SLOTS + s] = bsPtr[s] }
            off += TOTAL_SLOTS * 4
            let gsPtr = base.advanced(by: off).assumingMemoryBound(to: UInt32.self)
            for e in 0..<(E_EXP + 1) { l0GroupStart[p * (E_EXP + 1) + e] = gsPtr[e] }
        }
    }

    initLmState(bos: toks[0])
    let cb0 = buildStepCB(w); cb0.commit(); cb0.waitUntilCompleted()
    if let err = cb0.error { print("  GPU step 0: \(err)"); return }
    readAfterCB(position: 0)

    for p in 1..<S {
        let nextToks = [UInt32](repeating: toks[p], count: B)
        advanceLmState(nextTokens: nextToks)
        let cb = buildStepCB(w); cb.commit(); cb.waitUntilCompleted()
        if let err = cb.error { print("  GPU step \(p): \(err)"); return }
        readAfterCB(position: p)
    }

    if staging != nil {
        let outPath = "\(outDir)/lm_swift_hiddens.npy"
        hiddens.withUnsafeBufferPointer { bp in
            writeNpyFloat32(outPath, data: bp.baseAddress!, shape: [nBoundaries, S, HIDDEN])
        }
        print("  per-boundary min/max (slot 0, first position):")
        for L in 0..<nBoundaries {
            var mn = Float.infinity, mx = -Float.infinity
            var nNaN = 0, nInf = 0
            for i in 0..<HIDDEN {
                let v = hiddens[L * S * HIDDEN + 0 * HIDDEN + i]
                if v.isNaN { nNaN += 1 }
                else if !v.isFinite { nInf += 1 }
                else { if v < mn { mn = v }; if v > mx { mx = v } }
            }
            let label = (L == 0 ? "embed" : "layer \(L - 1)").padding(toLength: 9, withPad: " ", startingAt: 0)
            print(String(format: "    %@  NaN=%d  Inf=%d  min=%.3f  max=%.3f", label, nNaN, nInf, mn, mx))
        }
        print("  wrote \(outPath)")
    }

    if l0Staging != nil {
        let outPath = "\(outDir)/lm_swift_l0_probes.npy"
        l0Probes.withUnsafeBufferPointer { bp in
            writeNpyFloat32(outPath, data: bp.baseAddress!, shape: [nL0, S, HIDDEN])
        }
        let names = ["post_attn+res ", "ffw_norm_1 out", "ffw_norm_2 out",
                     "pre_ffw_2 out ", "moe_sum raw   "]
        print("  L0 probe min/max (slot 0, first position):")
        for probe in 0..<nL0 {
            var mn = Float.infinity, mx = -Float.infinity
            for i in 0..<HIDDEN {
                let v = l0Probes[probe * S * HIDDEN + 0 * HIDDEN + i]
                if v < mn { mn = v }
                if v > mx { mx = v }
            }
            print(String(format: "    %@  min=%.3f  max=%.3f", names[probe], mn, mx))
        }
        print("  wrote \(outPath)")
    }

    if attnStaging != nil {
        let path = "\(outDir)/lm_swift_l0_attn_out.npy"
        l0AttnOut.withUnsafeBufferPointer { bp in
            writeNpyFloat32(path, data: bp.baseAddress!, shape: [S, SLIDE_H, SLIDE_HD])
        }
        print("  wrote \(path)  [S, H_Q=\(SLIDE_H), HD=\(SLIDE_HD)]")
    }

    // Dump layer-0 K and V cache page 0 (slot 0) after the full AR sweep.
    // Shape: K_cache[L=0] page 0 = [PAGE=16, H_KV=8, HD=256] halves.
    // We snapshot it at the end; by then entries 0..S-1 should hold the
    // post-rope/post-k_norm K from each AR step.
    do {
        // Diagnostic dump samples chunk 0 only (the pages used by slot 0
        // in this harness's setup fall in chunk 0).
        let Kc = w.K_chunks[0][0]
        let Vc = w.V_chunks[0][0]
        // Layer 0 is sliding (PAGE=16, H_KV=8, HD=256).
        let pageElems = PAGE * 8 * 256
        var kFp32 = [Float](repeating: 0, count: pageElems)
        var vFp32 = [Float](repeating: 0, count: pageElems)
        let kp = Kc.contents().assumingMemoryBound(to: Float16.self)
        let vp = Vc.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<pageElems {
            kFp32[i] = Float(kp[i])
            vFp32[i] = Float(vp[i])
        }
        kFp32.withUnsafeBufferPointer { bp in
            writeNpyFloat32("\(outDir)/lm_swift_l0_K_page0.npy",
                             data: bp.baseAddress!, shape: [PAGE, 8, 256])
        }
        vFp32.withUnsafeBufferPointer { bp in
            writeNpyFloat32("\(outDir)/lm_swift_l0_V_page0.npy",
                             data: bp.baseAddress!, shape: [PAGE, 8, 256])
        }
        print("  wrote lm_swift_l0_K_page0.npy, lm_swift_l0_V_page0.npy  (slot 0, phys page 0)")
    }

    if routerStaging != nil {
        let idsPath = "\(outDir)/lm_swift_l0_expert_ids.npy"
        let wPath = "\(outDir)/lm_swift_l0_gate_w.npy"
        let lgPath = "\(outDir)/lm_swift_l0_router_lg.npy"
        let hnPath = "\(outDir)/lm_swift_l0_hidden_norm.npy"
        l0ExpertIds.withUnsafeBufferPointer { bp in
            let asInt32 = UnsafeRawPointer(bp.baseAddress!).assumingMemoryBound(to: Int32.self)
            writeNpyInt32(idsPath, data: asInt32, shape: [S, TOPK])
        }
        l0GateW.withUnsafeBufferPointer { bp in
            writeNpyFloat32(wPath, data: bp.baseAddress!, shape: [S, TOPK])
        }
        l0RouterLg.withUnsafeBufferPointer { bp in
            writeNpyFloat32(lgPath, data: bp.baseAddress!, shape: [S, E_EXP])
        }
        l0HiddenNorm.withUnsafeBufferPointer { bp in
            writeNpyFloat32(hnPath, data: bp.baseAddress!, shape: [S, HIDDEN])
        }
        print("  L0 router (slot 0, position 0):")
        let ids = (0..<TOPK).map { "\(l0ExpertIds[0 * TOPK + $0])" }.joined(separator: ",")
        let ws = (0..<TOPK).map { String(format: "%.4f", l0GateW[0 * TOPK + $0]) }.joined(separator: ",")
        print("    expert_ids: [\(ids)]")
        print("    gate_w    : [\(ws)]")
        var mn = Float.infinity, mx = -Float.infinity
        for e in 0..<E_EXP {
            let v = l0RouterLg[e]
            if v < mn { mn = v }; if v > mx { mx = v }
        }
        print(String(format: "    router_lg min=%.3f  max=%.3f", mn, mx))
        print("  wrote \(idsPath), \(wPath), \(lgPath), \(hnPath)")
    }

    if moeSlotsStaging != nil {
        let downPath = "\(outDir)/lm_swift_l0_moe_down_out.npy"
        let gufPath  = "\(outDir)/lm_swift_l0_gate_up_fused.npy"
        let gpPath   = "\(outDir)/lm_swift_l0_gate_proj.npy"
        let stokPath = "\(outDir)/lm_swift_l0_slot_token.npy"
        let bsPath   = "\(outDir)/lm_swift_l0_batch_slots.npy"
        let gsPath   = "\(outDir)/lm_swift_l0_group_start.npy"
        l0MoeDownOut.withUnsafeBufferPointer { bp in
            writeNpyFloat32(downPath, data: bp.baseAddress!, shape: [S, TOTAL_SLOTS, HIDDEN])
        }
        l0GateUpFused.withUnsafeBufferPointer { bp in
            writeNpyFloat32(gufPath, data: bp.baseAddress!, shape: [S, TOTAL_SLOTS, 2 * MOE_INT])
        }
        l0GateProj.withUnsafeBufferPointer { bp in
            writeNpyFloat32(gpPath, data: bp.baseAddress!, shape: [S, TOTAL_SLOTS, MOE_INT])
        }
        l0SlotToken.withUnsafeBufferPointer { bp in
            let p = UnsafeRawPointer(bp.baseAddress!).assumingMemoryBound(to: Int32.self)
            writeNpyInt32(stokPath, data: p, shape: [S, TOTAL_SLOTS])
        }
        l0BatchSlots.withUnsafeBufferPointer { bp in
            let p = UnsafeRawPointer(bp.baseAddress!).assumingMemoryBound(to: Int32.self)
            writeNpyInt32(bsPath, data: p, shape: [S, TOTAL_SLOTS])
        }
        l0GroupStart.withUnsafeBufferPointer { bp in
            let p = UnsafeRawPointer(bp.baseAddress!).assumingMemoryBound(to: Int32.self)
            writeNpyInt32(gsPath, data: p, shape: [S, E_EXP + 1])
        }
        // Quick sanity log for position 0.
        let stoks0 = (0..<TOTAL_SLOTS).map { "\(l0SlotToken[$0])" }.joined(separator: ",")
        let bslt0  = (0..<TOTAL_SLOTS).map { "\(l0BatchSlots[$0])" }.joined(separator: ",")
        print("  L0 routing (pos 0):")
        print("    slot_token : [\(stoks0)]")
        print("    batch_slots: [\(bslt0)]")
        // group_start — only print non-repeat entries (expert-start milestones).
        var ms: [String] = []
        for e in 0..<E_EXP {
            let s = l0GroupStart[e]
            let sNext = l0GroupStart[e + 1]
            if s != sNext { ms.append("e\(e):\(s)..\(sNext)") }
        }
        print("    groups     : [\(ms.joined(separator: ", "))]")
        print("  wrote \(downPath), \(stokPath), \(bsPath), \(gsPath)")
    }
}

// ====================================================================
// Prefill validation harness — drives `buildPrefillCB` against the same
// lm_<tag>_tokens / lm_<tag>_logits oracle as runLmKLHarness, but in a
// single forward where all qLen positions are computed in parallel.
//
// Env-var driver: LM_PREFILL_VALIDATE=<tag>  +  GGUF_PATH=<gguf>  +
//                 [LM_KL_REF=<dir>]  (defaults to our test_data/reference)
//
// Geometry: B=4 slots each fed the same prompt; slot 0 is scored, others
// are currently wasted compute (prefill slot plumbing is per-slot; a
// later pass can feed different prompts per slot). qLen = S (the full
// oracle prompt length, capped at MAX_Q_LEN=8). Each position p ∈ [0,S)
// produces a VOCAB-wide logit row that we compare to oracle[p].
//
// Expected: mean KL(oracle‖swift) ≤ 0.2 (same floor as the AR path),
// per-position argmax agreement = S/S.
// ====================================================================
func runLmPrefillValidate(ggufPath: String, refDir: String, tag: String) {
    print("\n=== LM prefill validation (B*qLen single-CB prefill) ===")
    let tokensPath = "\(refDir)/lm_\(tag)_tokens.npy"
    let logitsPath = "\(refDir)/lm_\(tag)_logits.npy"
    guard let tokNpy = mmapNpy(tokensPath) else { print("  cannot open \(tokensPath)"); return }
    guard let logNpy = mmapNpy(logitsPath) else { print("  cannot open \(logitsPath)"); tokNpy.release(); return }
    defer { tokNpy.release(); logNpy.release() }

    precondition(tokNpy.shape.count == 1 && tokNpy.dtype == "<i4", "tokens must be int32 1D")
    precondition(logNpy.shape.count == 2 && logNpy.dtype == "<f4", "logits must be float32 2D")
    let S = tokNpy.shape[0]
    let V = logNpy.shape[1]
    precondition(V == VOCAB, "vocab mismatch: oracle \(V) vs swift \(VOCAB)")
    // LM_PREFILL_SPLIT=<k>: prefill tokens[0..k-1], then AR for the remainder.
    // Defaults to qLen = min(S, MAX_Q_LEN) so long oracle prompts still fit.
    let splitEnv = ProcessInfo.processInfo.environment["LM_PREFILL_SPLIT"].flatMap { Int($0) }
    let qLen = splitEnv ?? min(S, MAX_Q_LEN)
    precondition(qLen <= MAX_Q_LEN, "prefill qLen=\(qLen) exceeds MAX_Q_LEN=\(MAX_Q_LEN)")
    precondition(qLen <= S, "prefill qLen=\(qLen) exceeds oracle prompt length S=\(S)")
    print("  oracle S=\(S); prefill qLen=\(qLen); vocab=\(V); B=\(B) slots fed the same prompt prefix")

    let toks: [UInt32] = {
        let p = tokNpy.ptr.advanced(by: tokNpy.dataOffset)
            .assumingMemoryBound(to: Int32.self)
        return (0..<S).map { UInt32(bitPattern: p[$0]) }
    }()
    let oracleBase = logNpy.ptr.advanced(by: logNpy.dataOffset)
        .assumingMemoryBound(to: Float.self)

    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    print("")

    // Populate prefill inputs. All B slots get the same prompt; per-slot
    // q_positions = [0, 1, …, qLen-1]; per-slot k_len = qLen (each query
    // attends to itself plus all earlier tokens in the same prefill batch).
    let tokP = pre_input_tokens.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
    let posP = pre_q_positions.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
    for b in 0..<B {
        for i in 0..<qLen {
            tokP[b * qLen + i] = toks[i]
            posP[b * qLen + i] = UInt32(i)
        }
    }
    let klsP = pre_k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
    let klfP = pre_k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
    for b in 0..<B { klsP[b] = UInt32(qLen); klfP[b] = UInt32(qLen) }

    // Block table — reuse the AR per-slot-disjoint layout. Each slot's
    // pages are physically distinct so batches can't alias.
    let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
    for b in 0..<B {
        for p in 0..<MAX_PAGES_PER_SLOT {
            btP[b * MAX_PAGES_PER_SLOT + p] = UInt32(b * MAX_PAGES_PER_SLOT + p) % UInt32(TOTAL_PAGES)
        }
    }

    // CSR mask precompute — causal + (for slide) SW. positionStart=0 since
    // we're doing the initial prefill.
    precomputeFlexPrefillMasks(qLen: qLen, positionStart: 0)

    let cb = buildPrefillCB(w, qLen: qLen, fullPrefillLogits: true)
    let t0 = Date()
    cb.commit(); cb.waitUntilCompleted()
    let dtMs = Date().timeIntervalSince(t0) * 1000
    if let err = cb.error { print("  GPU prefill: \(err)"); return }
    print(String(format: "  prefill CB wall: %.2f ms (B=%d, qLen=%d, %d rows)",
                 dtMs, B, qLen, B * qLen))

    // Copy slot-0 logits [qLen, VOCAB] to a Float buffer for KL math.
    let logitsFp16 = pre_logits.contents().assumingMemoryBound(to: Float16.self)
    var swiftLogits = [Float](repeating: 0, count: qLen * VOCAB)
    // pre_logits layout: [B*qLen, VOCAB]. Slot 0 rows are at row b=0 → offsets
    // [0*qLen + 0 .. 0*qLen + qLen-1] (row index = b*qLen + i).
    for i in 0..<qLen {
        let srcRow = 0 * qLen + i
        for v in 0..<VOCAB {
            swiftLogits[i * VOCAB + v] = Float(logitsFp16[srcRow * VOCAB + v])
        }
    }

    // Activeness A/B test — when LM_PREFILL_AB=1, ALSO run a numActiveRows=qLen
    // dispatch (single-slot prefill geometry) and assert that slot-0's
    // pre_logits rows are byte-identical to the full B*qLen baseline above.
    // This is the byte-equality correctness gate from the activeB-aware
    // refactor task. Per-row independence of every prefill matmul means
    // slot-0 outputs must be unchanged — if they differ, the refactor's
    // per-row-independence assumption is violated.
    if ProcessInfo.processInfo.environment["LM_PREFILL_AB"] == "1" {
        // Snapshot baseline slot-0 logits as fp16 bytes.
        var baselineSlot0 = [Float16](repeating: 0, count: qLen * VOCAB)
        for i in 0..<qLen {
            let srcRow = 0 * qLen + i
            for v in 0..<VOCAB {
                baselineSlot0[i * VOCAB + v] = logitsFp16[srcRow * VOCAB + v]
            }
        }
        // Zero pre_logits before the activeB dispatch so we can verify rows
        // 1..(B*qLen-1) remain zeroed (i.e., the dispatch really only wrote
        // slot 0's rows).
        let preLogitsBytes = B * qLen * VOCAB * MemoryLayout<Float16>.stride
        memset(pre_logits.contents(), 0, preLogitsBytes)

        // Re-precompute masks at numActiveSlots=1 (matches the single-slot
        // prepare path) and re-dispatch with numActiveRows=qLen.
        precomputeFlexPrefillMasks(qLen: qLen, positionStart: 0, numActiveSlots: 1)
        let cb2 = buildPrefillCB(w, qLen: qLen, fullPrefillLogits: true,
                                  numActiveRows: qLen)
        let t1 = Date()
        cb2.commit(); cb2.waitUntilCompleted()
        let dtMs2 = Date().timeIntervalSince(t1) * 1000
        if let err = cb2.error { print("  GPU prefill (activeB): \(err)"); return }
        print(String(format: "  prefill CB wall (activeB=1): %.2f ms (B=%d, qLen=%d, %d rows) → %.2f× speedup",
                     dtMs2, B, qLen, qLen, dtMs / dtMs2))

        // Byte-equality check: slot-0 rows in pre_logits must match the
        // baseline fp16 bit pattern exactly.
        let logitsFp16v2 = pre_logits.contents().assumingMemoryBound(to: Float16.self)
        var mismatches = 0
        var maxAbsDiff: Float = 0
        for i in 0..<qLen {
            let srcRow = 0 * qLen + i
            for v in 0..<VOCAB {
                let a = baselineSlot0[i * VOCAB + v]
                let b = logitsFp16v2[srcRow * VOCAB + v]
                if a.bitPattern != b.bitPattern {
                    mismatches += 1
                    let d = abs(Float(a) - Float(b))
                    if d > maxAbsDiff { maxAbsDiff = d }
                }
            }
        }
        let total = qLen * VOCAB
        if mismatches == 0 {
            print(String(format: "  AB byte-equality: PASS (slot-0 logits %d/%d bytes identical)",
                         total, total))
        } else {
            print(String(format: "  AB byte-equality: FAIL (%d/%d mismatches, max abs diff %.6f)",
                         mismatches, total, maxAbsDiff))
        }

        // Restore CSR + re-run baseline so downstream AR-step path (if any)
        // sees the expected full-B mask state.
        precomputeFlexPrefillMasks(qLen: qLen, positionStart: 0)
    }

    // Per-position metrics.
    print("  per-position metrics (oracle vs swift, slot 0):")
    print("    pos  token     L2        KL(ora‖swi)   top1?  top5∩  ora-next                    swi-next")
    var totalL2: Double = 0
    var totalKL: Double = 0
    var totalTop1: Int = 0
    var totalTop5Overlap: Int = 0
    for p in 0..<qLen {
        let orp = UnsafeBufferPointer(start: oracleBase.advanced(by: p * VOCAB), count: VOCAB)
        let (l2, kl, top1Match, top5Overlap, oraTop, swiTop) = swiftLogits.withUnsafeBufferPointer { swBuf -> (Float, Float, Bool, Int, Int, Int) in
            let swp = UnsafeBufferPointer(start: swBuf.baseAddress!.advanced(by: p * VOCAB), count: VOCAB)
            return scoreLogitPosition(orp, swp)
        }
        totalL2 += Double(l2); totalKL += Double(kl)
        totalTop1 += top1Match ? 1 : 0
        totalTop5Overlap += top5Overlap
        let oraTokStr = safeVocabToken(w, oraTop).padding(toLength: 20, withPad: " ", startingAt: 0)
        let swiTokStr = safeVocabToken(w, swiTop).padding(toLength: 20, withPad: " ", startingAt: 0)
        let tokStr = safeVocabToken(w, Int(toks[p])).padding(toLength: 8, withPad: " ", startingAt: 0)
        let posStr = String(format: "%3d", p)
        let l2Str = String(format: "%7.4f", l2)
        let klStr = String(format: "%9.4f", kl)
        let oraIdStr = String(format: "%6d", oraTop)
        let swiIdStr = String(format: "%6d", swiTop)
        print("    \(posStr)  \(tokStr)  \(l2Str)   \(klStr)   \(top1Match ? "✓" : "✗")      "
              + "\(top5Overlap)/5    \(oraIdStr)=\(oraTokStr)  \(swiIdStr)=\(swiTokStr)")
    }
    print("")
    print(String(format: "  mean L2      : %.4f", totalL2 / Double(qLen)))
    print(String(format: "  mean KL      : %.4f", totalKL / Double(qLen)))
    print(String(format: "  argmax match : %d / %d", totalTop1, qLen))
    print(String(format: "  top-5 overlap: %d / %d (max %d)", totalTop5Overlap, 5 * qLen, 5 * qLen))

    // Scheduling controller — if LM_PREFILL_AR_STEPS=N is set, run the
    // prefill then continue with N AR steps, verifying each AR step's KL
    // against the oracle. This exercises the prefill→AR state handoff:
    // the AR kernels must read the prefill-populated KV cache and extend
    // it one row at a time. Uses oracle tokens (teacher-forcing) so the
    // KL reflects handoff fidelity, not sampling noise.
    //
    // NOTE: this needs LM_PREFILL_AR_STEPS <= S - qLen_prefill. When
    // LM_PREFILL_SPLIT=<k> is also set, prefill runs on tokens[0..k-1]
    // and AR runs on tokens[k..qLen-1], using oracle[k..qLen-1] as refs.
    guard let arStepsStr = ProcessInfo.processInfo.environment["LM_PREFILL_AR_STEPS"],
          let arSteps = Int(arStepsStr), arSteps > 0 else { return }
    print("\n  --- prefill→AR handoff: \(arSteps) AR steps after qLen=\(qLen) prefill ---")

    // Seed AR state to match the post-prefill geometry:
    //   positions[b] = qLen - 1 (next step writes at qLen)
    //   k_len_*[b]   = qLen     (advanceLmState will bump to qLen+1)
    //   num_pages_*  = ceil(qLen / PAGE)
    // The AR kernels use the same block_table, so the KV cache prefill
    // wrote through pre_* plumbing is already live for AR to consume.
    let arPosP = positions.contents().bindMemory(to: UInt32.self, capacity: B)
    let arNpsP = num_pages_slide.contents().bindMemory(to: UInt32.self, capacity: B)
    let arNpfP = num_pages_full.contents().bindMemory(to: UInt32.self, capacity: B)
    let arKlsP = k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
    let arKlfP = k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
    for b in 0..<B {
        arPosP[b] = UInt32(qLen - 1)
        arKlsP[b] = UInt32(qLen); arKlfP[b] = UInt32(qLen)
        arNpsP[b] = UInt32((qLen + PAGE - 1) / PAGE)
        arNpfP[b] = UInt32((qLen + PAGE  - 1) / PAGE)
    }

    // Teacher-force each AR step with the token that actually follows in
    // the oracle prompt. First AR step consumes token[qLen-1]'s argmax
    // position (already in pre_logits) and asks the model to predict the
    // token AT qLen. Since the oracle stops at position S-1, we only have
    // oracle references up to position S-1 — cap arSteps at S - qLen + 1
    // (the +1 is: we can score the new logit at position qLen-1 that we
    // already have from prefill, plus arSteps-1 new AR positions if S is
    // large enough). For the hello 5-tok oracle, arSteps=0 new positions
    // are scorable; hellolong has S=21 so arSteps up to 21-5=16 work.
    let maxScorable = max(0, S - qLen)
    if maxScorable == 0 {
        print("  oracle has no positions after qLen=\(qLen); skipping AR handoff scoring")
        return
    }
    let nSteps = min(arSteps, maxScorable)
    print("  scoring first \(nSteps) AR positions vs oracle[\(qLen)..\(qLen + nSteps - 1)]")

    var arTop1 = 0
    var arKL: Double = 0
    for i in 0..<nSteps {
        // Teacher-force: feed the oracle token that lives at the absolute
        // position this AR step will write. advanceLmState increments
        // positions from qLen-1+i to qLen+i, so the new KV row lands at
        // position qLen+i — and we want its input embedding to be
        // toks[qLen+i] (not toks[qLen-1+i], which prefill already wrote).
        let nextTok = toks[qLen + i]
        let nextToks = [UInt32](repeating: nextTok, count: B)
        advanceLmState(nextTokens: nextToks)
        let cbAr = buildStepCB(w); cbAr.commit(); cbAr.waitUntilCompleted()
        if let err = cbAr.error { print("  GPU AR step \(i): \(err)"); return }
        // Score the new logit at oracle position qLen + i.
        let p = qLen + i
        let orp = UnsafeBufferPointer(start: oracleBase.advanced(by: p * VOCAB), count: VOCAB)
        var swiftAR = [Float](repeating: 0, count: VOCAB)
        let logP = logits.contents().assumingMemoryBound(to: Float16.self)
        for v in 0..<VOCAB { swiftAR[v] = Float(logP[v]) }
        let (l2, kl, top1Match, top5Overlap, oraTop, swiTop) = swiftAR.withUnsafeBufferPointer { sw in
            scoreLogitPosition(orp, UnsafeBufferPointer(start: sw.baseAddress!, count: VOCAB))
        }
        arTop1 += top1Match ? 1 : 0
        arKL += Double(kl)
        let oraTokStr = safeVocabToken(w, oraTop).padding(toLength: 20, withPad: " ", startingAt: 0)
        let swiTokStr = safeVocabToken(w, swiTop).padding(toLength: 20, withPad: " ", startingAt: 0)
        print(String(format: "    AR%3d  fed=%6d  L2=%.4f  KL=%.4f  top1=%@  top5∩=%d/5  ora=%d/%@ swi=%d/%@",
                     p, Int(nextTok), l2, kl, top1Match ? "✓" : "✗", top5Overlap,
                     oraTop, oraTokStr, swiTop, swiTokStr))
    }
    print(String(format: "  AR mean KL: %.4f   argmax: %d / %d", arKL / Double(nSteps), arTop1, nSteps))
}

