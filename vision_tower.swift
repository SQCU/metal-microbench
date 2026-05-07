// vision_tower.swift — Gemma-4 vision encoder (27-layer SigLIP2-style).
// Extracted from the monofile forward_graph.swift in the 2026-04-18 refactor.
//
// Contents:
//   - Swift preprocessor (PIL-compatible bicubic+antialias resize, patchify)
//   - Vision tower weight loader (BF16 safetensors → FP16)
//   - Kernel dispatch wrappers (encGemvFp16V5, encVision*, encRMSNormGFp32*, …)
//   - runVisionTowerBatchForward — 27-layer forward producing per-image
//     [nPooled, 2816] fp32 soft tokens. Handles B=1 and B>1 uniformly;
//     the flat [B*N, ...] buffer layout reduces to [N, ...] at B=1 with
//     the same kernel dispatches.
//     soft tokens for the multimodal LM input stream
//
// Depends on kernels.swift (MSL source) and common.swift (device/queue/safetensors).

import Metal
import Foundation
import ImageIO
import CoreGraphics
import Accelerate

// ===========================================================================
// Gemma-4 image preprocessor (Option A: full Swift-side pipeline, no Python
// handoff). Replicates image_processing_gemma4.py:
//   1. Decode PNG/JPEG to RGB uint8 via CoreGraphics.
//   2. Aspect-preserving resize to (target_h, target_w) — largest multiple
//      of `pooling_kernel_size * patch_size = 48` px per side fitting under
//      `max_soft_tokens * pooling_kernel_size^2` patches total.
//   3. Bicubic/Lanczos resample via vImage HighQuality flag. Not bit-
//      identical to torchvision's bicubic+antialias, but within fp16
//      rounding for our MSE validation.
//   4. Rescale to [0, 1] floating-point CHW.
//   5. Patchify into [N_patches, 768] with HWC flatten per patch (matches
//      Gemma4's convert_image_to_patches output).
//   6. Pad to max_patches along the first dim; positions padded with -1.
// Works for any aspect ratio — no square-mode assumptions.
// ===========================================================================

struct PatchBatch {
    let patches: MTLBuffer       // [maxPatches, 768] fp16, zero-padded past numRealPatches
    let positions: [(Int32, Int32)]   // per-patch (y, x); real patches first, then (0,0) for pad
    let numRealPatches: Int
    let maxPatches: Int          // total slots including zero-padded tail
    let gridH: Int               // patch grid after resize
    let gridW: Int
    let resizedH: Int
    let resizedW: Int
}

enum PreprocessError: Error {
    case imageDecodeFailed(String)
    case invalidGeometry(String)
    case vImageFailed(Int)
}

/// Gemma-4 aspect-preserving resize target, exactly matching
/// `get_aspect_ratio_preserving_size` in image_processing_pil_gemma4.py.
func gemma4AspectResizeTarget(height h: Int, width w: Int,
                                patchSize: Int, maxPatches: Int,
                                poolingKernel: Int) -> (Int, Int) {
    let totalPx = Double(h * w)
    let targetPx = Double(maxPatches * patchSize * patchSize)
    let factor = (targetPx / totalPx).squareRoot()
    let idealH = factor * Double(h)
    let idealW = factor * Double(w)
    let sideMult = poolingKernel * patchSize
    var tH = Int(floor(idealH / Double(sideMult))) * sideMult
    var tW = Int(floor(idealW / Double(sideMult))) * sideMult
    let maxSideLen = (maxPatches / (poolingKernel * poolingKernel)) * sideMult
    if tH == 0 && tW == 0 {
        return (sideMult, sideMult)
    }
    if tH == 0 {
        tH = sideMult
        tW = min(Int(floor(Double(w) / Double(h))) * sideMult, maxSideLen)
    } else if tW == 0 {
        tW = sideMult
        tH = min(Int(floor(Double(h) / Double(w))) * sideMult, maxSideLen)
    }
    return (tH, tW)
}

/// Decode PNG/JPEG to a [3, H, W] RGB uint8 buffer via CoreGraphics. Force
/// an alpha-ignored render into a tightly-packed RGBA bitmap, then pluck
/// the three RGB channels out.
private func loadImageRGBFromBytes(_ data: Data) throws -> (width: Int, height: Int, rgb: [UInt8]) {
    guard let src = CGImageSourceCreateWithData(data as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        throw PreprocessError.imageDecodeFailed("CGImageSource")
    }
    let w = cg.width, h = cg.height
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    let cs = CGColorSpaceCreateDeviceRGB()
    // CGImageAlphaInfo.noneSkipLast + byteOrder32Big → memory order R, G, B, skipped.
    let bmInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    guard let ctx = rgba.withUnsafeMutableBytes({ ptr -> CGContext? in
        CGContext(data: ptr.baseAddress, width: w, height: h, bitsPerComponent: 8,
                  bytesPerRow: w * 4, space: cs, bitmapInfo: bmInfo)
    }) else { throw PreprocessError.imageDecodeFailed("CGContext") }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    // rgba is now [R, G, B, skipped] per pixel, row-major top-to-bottom.
    var rgb = [UInt8](repeating: 0, count: w * h * 3)
    for i in 0..<(w * h) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0]
        rgb[i * 3 + 1] = rgba[i * 4 + 1]
        rgb[i * 3 + 2] = rgba[i * 4 + 2]
    }
    return (w, h, rgb)
}

/// Resize one channel (Planar8) to target size using PIL-style bicubic
/// resample with antialiasing, matching `torchvision.transforms.v2.functional.
/// resize(..., interpolation=BICUBIC, antialias=True)` which Gemma4's
/// Hugging Face image processor uses. The implementation follows PIL's
/// `ImagingResampleHorizontal/Vertical` (src/libImaging/Resample.c): a
/// separable 1D bicubic with Keys' a=-0.5 and the antialias convention of
/// scaling the kernel support by max(1, in/out) when downsampling.
///
/// Returns float [dstH, dstW] (not UInt8 — avoids a redundant round-trip,
/// patchify then consumes floats directly).
private func bicubicResizePlaneFloat(src: UnsafePointer<UInt8>, srcW: Int, srcH: Int,
                                       dstW: Int, dstH: Int) -> [Float] {
    @inline(__always) func bicubicWeight(_ x: Float) -> Float {
        // Keys' bicubic with a = -0.5 (PIL/torchvision default).
        let a: Float = -0.5
        let xx = abs(x)
        if xx < 1.0 { return ((a + 2.0) * xx - (a + 3.0)) * xx * xx + 1.0 }
        if xx < 2.0 { return (((xx - 5.0) * xx + 8.0) * xx - 4.0) * a }
        return 0.0
    }
    // Precompute 1D resample weights for output dimension of length `outSize`
    // from input dimension of length `inSize`. Returns:
    //   bounds[i] = (xmin, xcount) — range of input pixels contributing to output i
    //   weights[i * kmax + j] = weight for input pixel (xmin + j)
    // kmax is the worst-case contributor count across all outputs.
    func precomputeCoeffs(inSize: Int, outSize: Int) -> (bounds: [Int], weights: [Float], kmax: Int) {
        let scale = Float(inSize) / Float(outSize)
        let filterscale = max(1.0, scale)
        let support: Float = 2.0 * filterscale
        let invFilterScale = 1.0 / filterscale
        let kmax = Int(support * 2) + 2  // generous upper bound
        var bounds = [Int](repeating: 0, count: outSize * 2)
        var weights = [Float](repeating: 0, count: outSize * kmax)
        for xx in 0..<outSize {
            let center = (Float(xx) + 0.5) * scale
            var xmin = Int(center - support + 0.5)   // truncate toward 0; center > 0
            if xmin < 0 { xmin = 0 }
            var xmaxAbs = Int(center + support + 0.5)
            if xmaxAbs > inSize { xmaxAbs = inSize }
            let xcount = xmaxAbs - xmin
            var wsum: Float = 0
            for x in 0..<xcount {
                let w = bicubicWeight((Float(x + xmin) - center + 0.5) * invFilterScale)
                weights[xx * kmax + x] = w
                wsum += w
            }
            if wsum != 0 {
                for x in 0..<xcount {
                    weights[xx * kmax + x] /= wsum
                }
            }
            bounds[xx * 2 + 0] = xmin
            bounds[xx * 2 + 1] = xcount
        }
        return (bounds, weights, kmax)
    }
    // Step 1: UInt8 → Float buffer (keep input-scale, no [0,1] normalize yet).
    var srcF = [Float](repeating: 0, count: srcW * srcH)
    for i in 0..<(srcW * srcH) { srcF[i] = Float(src[i]) }

    // Step 2: horizontal resize [srcH, srcW] → [srcH, dstW].
    let (xBounds, xWeights, xKmax) = precomputeCoeffs(inSize: srcW, outSize: dstW)
    var mid = [Float](repeating: 0, count: srcH * dstW)
    for yy in 0..<srcH {
        let rowIn = yy * srcW
        let rowOut = yy * dstW
        for xx in 0..<dstW {
            let xmin = xBounds[xx * 2 + 0]
            let xcount = xBounds[xx * 2 + 1]
            var s: Float = 0
            for x in 0..<xcount {
                s += srcF[rowIn + xmin + x] * xWeights[xx * xKmax + x]
            }
            mid[rowOut + xx] = s
        }
    }

    // Step 3: vertical resize [srcH, dstW] → [dstH, dstW].
    let (yBounds, yWeights, yKmax) = precomputeCoeffs(inSize: srcH, outSize: dstH)
    var dst = [Float](repeating: 0, count: dstH * dstW)
    for yy in 0..<dstH {
        let ymin = yBounds[yy * 2 + 0]
        let ycount = yBounds[yy * 2 + 1]
        let rowOut = yy * dstW
        for xx in 0..<dstW {
            var s: Float = 0
            for y in 0..<ycount {
                s += mid[(ymin + y) * dstW + xx] * yWeights[yy * yKmax + y]
            }
            dst[rowOut + xx] = s
        }
    }
    return dst
}

/// Full preprocessor: PNG → [maxPatches, 768] fp16 patch tensor.
// Demo/test-harness convenience: read the file, call gemma4ImagePreprocess.
// The serving FFI (gemma_submit_image_bytes) never touches this — it gets
// bytes directly from the HTTP body.
func gemma4ImagePreprocessFromPath(path: String,
                                     maxSoftTokens: Int = 280,
                                     patchSize: Int = 16,
                                     poolingKernel: Int = 3,
                                     device: MTLDevice) throws -> PatchBatch {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try gemma4ImagePreprocess(data: data,
                                       maxSoftTokens: maxSoftTokens,
                                       patchSize: patchSize,
                                       poolingKernel: poolingKernel,
                                       device: device)
}

func gemma4ImagePreprocess(data: Data,
                             maxSoftTokens: Int = 280,
                             patchSize: Int = 16,
                             poolingKernel: Int = 3,
                             device: MTLDevice) throws -> PatchBatch {
    // Step 1: decode image (PNG/JPEG/HEIC/etc. — whatever CGImageSource handles).
    let (origW, origH, rgbBytes) = try loadImageRGBFromBytes(data)

    // Step 2: target size
    let maxPatches = maxSoftTokens * poolingKernel * poolingKernel
    let (targetH, targetW) = gemma4AspectResizeTarget(
        height: origH, width: origW,
        patchSize: patchSize, maxPatches: maxPatches,
        poolingKernel: poolingKernel)

    // Step 3: separate into 3 planes, resize each via PIL-compatible
    // bicubic+antialias. Output is float (preserves resample precision).
    var plane0 = [UInt8](repeating: 0, count: origW * origH)
    var plane1 = [UInt8](repeating: 0, count: origW * origH)
    var plane2 = [UInt8](repeating: 0, count: origW * origH)
    for i in 0..<(origW * origH) {
        plane0[i] = rgbBytes[i * 3 + 0]
        plane1[i] = rgbBytes[i * 3 + 1]
        plane2[i] = rgbBytes[i * 3 + 2]
    }
    let r = plane0.withUnsafeBufferPointer { bicubicResizePlaneFloat(
        src: $0.baseAddress!, srcW: origW, srcH: origH,
        dstW: targetW, dstH: targetH) }
    let g = plane1.withUnsafeBufferPointer { bicubicResizePlaneFloat(
        src: $0.baseAddress!, srcW: origW, srcH: origH,
        dstW: targetW, dstH: targetH) }
    let b = plane2.withUnsafeBufferPointer { bicubicResizePlaneFloat(
        src: $0.baseAddress!, srcW: origW, srcH: origH,
        dstW: targetW, dstH: targetH) }

    // Step 4+5: rescale [0,255] float → [-1, +1] fp16, patchify with HWC flatten.
    // Output layout [maxPatches, 768] with real patches first, zero-padded after.
    let gridH = targetH / patchSize
    let gridW = targetW / patchSize
    let nReal = gridH * gridW
    guard nReal <= maxPatches else {
        throw PreprocessError.invalidGeometry("resized grid \(gridH)×\(gridW)=\(nReal) exceeds maxPatches \(maxPatches)")
    }

    let patchBuf = device.makeBuffer(length: maxPatches * 768 * 2,
                                       options: .storageModeShared)!
    memset(patchBuf.contents(), 0, patchBuf.length)
    let dp = patchBuf.contents().assumingMemoryBound(to: Float16.self)
    // Gemma4V rescales [0, 255] → [-1, 1] before patch embed:
    //   /255 (rescale) then 2*x - 1 = (byte/255) * 2 - 1 = byte/127.5 - 1.
    // Bake this into one multiply+add per pixel.
    let scale = Float(2.0 / 255.0)
    let bias = Float(-1.0)

    // Clamp+round to uint8 to match torchvision's BICUBIC path on a uint8
    // input tensor (it clamps overshoot and quantizes to 1/255 levels before
    // the /255 rescale). Skipping this adds ~0.02 per-pixel error from
    // bicubic ringing.
    @inline(__always) func u8round(_ v: Float) -> Float {
        let r = v.rounded()
        if r < 0 { return 0 }
        if r > 255 { return 255 }
        return r
    }
    for py in 0..<gridH {
        for px in 0..<gridW {
            let patchIdx = py * gridW + px
            let oy = py * patchSize
            let ox = px * patchSize
            let baseOut = patchIdx * 768
            for y in 0..<patchSize {
                let rowY = oy + y
                for x in 0..<patchSize {
                    let src = rowY * targetW + (ox + x)
                    let k = (y * patchSize + x) * 3
                    dp[baseOut + k + 0] = Float16(u8round(r[src]) * scale + bias)
                    dp[baseOut + k + 1] = Float16(u8round(g[src]) * scale + bias)
                    dp[baseOut + k + 2] = Float16(u8round(b[src]) * scale + bias)
                }
            }
        }
    }

    // Positions: real patches get (y, x); padding slots get (0, 0) so kernels
    // don't index OOB when we let them run over the full max_patches range.
    // The attention padding mask (separate buffer) is what actually silences
    // these slots — the (0,0) pos value is just a safe default.
    var positions = [(Int32, Int32)](repeating: (0, 0), count: maxPatches)
    for py in 0..<gridH {
        for px in 0..<gridW {
            positions[py * gridW + px] = (Int32(py), Int32(px))
        }
    }

    return PatchBatch(patches: patchBuf, positions: positions,
                       numRealPatches: nReal, maxPatches: maxPatches,
                       gridH: gridH, gridW: gridW,
                       resizedH: targetH, resizedW: targetW)
}


// ---- Vision PSO declarations ----
let visionPatchEmbedPSO = pso("vision_patch_embed_fp16")
let visionPosEmbedPSO   = pso("vision_pos_embed_add_fp16")
let vision2dRopePSO     = pso("vision_2d_rope_neox_fp16")
let denseGemvFp16V5PSO  = pso("dense_gemv_fp16_v5")
let visionAttnPrefillPSO = pso("vision_attn_prefill_fp16")
let visionPool2DPSO      = pso("vision_pool_2d_fp16")
let visionStdNormPSO     = pso("vision_scaled_std_normalize_fp16")
let denseGemvFp32OutPSO  = pso("dense_gemv_fp16in_fp32out_v5")
let visionPosEmbedFp32PSO = pso("vision_pos_embed_add_fp32")
let rmsNormFp32InPSO     = pso("rms_norm_fp32in")
let addInplaceFp32PSO    = pso("add_inplace_fp32dst_fp16src")
let visionPool2DFp32InPSO = pso("vision_pool_2d_fp32in_fp16out")
let rmsNormFp32OutPSO    = pso("rms_norm_fp16in_fp32out")
let addInplaceFp32FpPSO  = pso("add_inplace_fp32_fp32")

// Fp32-intermediate vision path PSOs. Mirror the fp16-path set above,
// just with fp32 I/O on all the per-layer intermediate buffers. See the
// kernel block in kernels.swift for rationale (outlier channels need
// fp32 precision or MSE compounds past layer 12).
let rmsNormFp32PSO       = pso("rms_norm_fp32")
let rmsNormNoScaleFp32PSO = pso("rms_norm_noscale_fp32")
let denseGemvFp32In32OutPSO = pso("dense_gemv_fp32in_fp32out_v5")
let vision2dRopeFp32PSO  = pso("vision_2d_rope_neox_fp32")
let visionAttnPrefillFp32PSO = pso("vision_attn_prefill_fp32")
let visionAttnFlashFp32PSO   = pso("vision_attn_flash_fp32")
let visionGemmFp32MmaPSO     = pso("vision_gemm_fp32_mma")
let visionGemmFp32MmaV2PSO   = pso("vision_gemm_fp32_mma_v2")
let visionGemmFp32MmaV3PSO   = pso("vision_gemm_fp32_mma_v3")
let visionFfnGateUpGeluFp32MmaPSO   = pso("vision_ffn_gate_up_gelu_fp32_mma")
let visionFfnGateUpGeluFp32MmaV2PSO = pso("vision_ffn_gate_up_gelu_fp32_mma_v2")
let visionFfnGateUpGeluFp32MmaV3PSO = pso("vision_ffn_gate_up_gelu_fp32_mma_v3")
let geluMulFp32PSO       = pso("gelu_mul_fp32")
let visionPool2DFp32InFp32OutPSO = pso("vision_pool_2d_fp32in_fp32out")
let visionStdNormFp32PSO = pso("vision_scaled_std_normalize_fp32")
let quantizeFp32ToBf16PSO = pso("quantize_fp32_to_bf16_inplace")

// Vision tower: per-layer weight struct + loader.
// Each of 27 encoder layers has the same tensor set; we load them from the
// Gemma-4 bf16 safetensors and convert BF16 → FP16 for our kernels.
// ===========================================================================
struct VisionLayerW {
    let inputNorm: MTLBuffer            // RMSNorm pre-attn gamma (1152)
    let qProj, kProj, vProj, oProj: MTLBuffer   // [1152, 1152] each
    let qNorm, kNorm: MTLBuffer         // per-head gamma (72)
    let postAttnNorm: MTLBuffer         // post-attention RMSNorm gamma (1152)
    let preFfnNorm: MTLBuffer           // pre-FFN RMSNorm gamma (1152)
    let gateProj, upProj, downProj: MTLBuffer  // [4304, 1152], [4304, 1152], [1152, 4304]
    let postFfnNorm: MTLBuffer          // post-FFN RMSNorm gamma (1152)
}

struct VisionWeights {
    let patchEmbedW: MTLBuffer           // [1152, 768] — patch_embedder.input_proj.weight
    let posEmbedTable: MTLBuffer         // [2, POS_MAX=10240, 1152] BF16 → FP16; y at offset 0, x at POS_MAX*1152
    let stdBias: MTLBuffer               // [1152]
    let stdScale: MTLBuffer              // [1152]
    let embedVisionProj: MTLBuffer       // [2816, 1152] — embed_vision.embedding_projection.weight
    let layers: [VisionLayerW]           // 27
    let hidden: Int = 1152
    let interm: Int = 4304
    let numHeads: Int = 16
    let headDim: Int = 72
    let posMax: Int = 10240
    let numLayers: Int = 27
    let textHidden: Int = 2816
}

func loadVisionWeights(_ st: SafetensorsFile, device: MTLDevice) throws -> VisionWeights {
    let t0 = Date()
    print("  loading vision weights ...")
    let patchEmbed = try st.loadBF16AsFP16("model.vision_tower.patch_embedder.input_proj.weight", device: device)
    let posTable = try st.loadBF16AsFP16("model.vision_tower.patch_embedder.position_embedding_table", device: device)
    let stdBias = try st.loadBF16AsFP16("model.vision_tower.std_bias", device: device)
    let stdScale = try st.loadBF16AsFP16("model.vision_tower.std_scale", device: device)
    let embedVision = try st.loadBF16AsFP16("model.embed_vision.embedding_projection.weight", device: device)

    var layers: [VisionLayerW] = []
    layers.reserveCapacity(27)
    for L in 0..<27 {
        let p = "model.vision_tower.encoder.layers.\(L)."
        let lw = VisionLayerW(
            inputNorm:    try st.loadBF16AsFP16("\(p)input_layernorm.weight", device: device),
            qProj:        try st.loadBF16AsFP16("\(p)self_attn.q_proj.linear.weight", device: device),
            kProj:        try st.loadBF16AsFP16("\(p)self_attn.k_proj.linear.weight", device: device),
            vProj:        try st.loadBF16AsFP16("\(p)self_attn.v_proj.linear.weight", device: device),
            oProj:        try st.loadBF16AsFP16("\(p)self_attn.o_proj.linear.weight", device: device),
            qNorm:        try st.loadBF16AsFP16("\(p)self_attn.q_norm.weight", device: device),
            kNorm:        try st.loadBF16AsFP16("\(p)self_attn.k_norm.weight", device: device),
            postAttnNorm: try st.loadBF16AsFP16("\(p)post_attention_layernorm.weight", device: device),
            preFfnNorm:   try st.loadBF16AsFP16("\(p)pre_feedforward_layernorm.weight", device: device),
            gateProj:     try st.loadBF16AsFP16("\(p)mlp.gate_proj.linear.weight", device: device),
            upProj:       try st.loadBF16AsFP16("\(p)mlp.up_proj.linear.weight", device: device),
            downProj:     try st.loadBF16AsFP16("\(p)mlp.down_proj.linear.weight", device: device),
            postFfnNorm:  try st.loadBF16AsFP16("\(p)post_feedforward_layernorm.weight", device: device)
        )
        layers.append(lw)
    }
    print(String(format: "  vision weights loaded in %.2f sec", Date().timeIntervalSince(t0)))
    return VisionWeights(
        patchEmbedW: patchEmbed, posEmbedTable: posTable,
        stdBias: stdBias, stdScale: stdScale,
        embedVisionProj: embedVision, layers: layers)
}

// ---- Vision-kernel dispatch wrappers ----
// FP16 dense GEMV v5 (row-major, split-K). Grid (D_out/32, B), 128 threads/TG.
func encGemvFp16V5(_ cb: MTLCommandBuffer, x: MTLBuffer, W: MTLBuffer, out: MTLBuffer,
                    B: Int, Din: Int, Dout: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseGemvFp16V5PSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 3); enc.setBytes(&dou, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: B, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

func encVisionPosEmbedAdd(_ cb: MTLCommandBuffer, x: MTLBuffer, posTable: MTLBuffer,
                           posMax: Int, out: MTLBuffer, N: Int, nPatchesX: Int, hidden: Int) {
    // Per gemma4v.cpp: X-table at offset 0, Y-table at offset pos_size * nb1.
    // Kernel binds index 1 → pos_y_table, index 2 → pos_x_table — so Y comes
    // from offset POS_MAX*hidden and X from offset 0.
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(visionPosEmbedPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(posTable, offset: posMax * hidden * 2, index: 1)   // Y-table
    enc.setBuffer(posTable, offset: 0, index: 2)                     // X-table
    enc.setBuffer(out, offset: 0, index: 3)
    var nx = UInt32(nPatchesX), h = UInt32(hidden)
    enc.setBytes(&nx, length: 4, index: 4); enc.setBytes(&h, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encVision2DRope(_ cb: MTLCommandBuffer, x: MTLBuffer, posX: MTLBuffer, posY: MTLBuffer,
                      N: Int, H: Int, HD: Int, theta: Float) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(vision2dRopePSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(posX, offset: 0, index: 1); enc.setBuffer(posY, offset: 0, index: 2)
    var Hv = UInt32(H), HDv = UInt32(HD), thv = theta
    enc.setBytes(&Hv, length: 4, index: 3)
    enc.setBytes(&HDv, length: 4, index: 4)
    enc.setBytes(&thv, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: N, height: H, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encVisionAttnPrefill(_ cb: MTLCommandBuffer, Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer, O: MTLBuffer,
                            N: Int, H: Int, HD: Int, qkScale: Float) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(visionAttnPrefillPSO)
    enc.setBuffer(Q, offset: 0, index: 0); enc.setBuffer(K, offset: 0, index: 1)
    enc.setBuffer(V, offset: 0, index: 2); enc.setBuffer(O, offset: 0, index: 3)
    var Nv = UInt32(N), Hv = UInt32(H), HDv = UInt32(HD), sc = qkScale
    enc.setBytes(&Nv, length: 4, index: 4)
    enc.setBytes(&Hv, length: 4, index: 5)
    enc.setBytes(&HDv, length: 4, index: 6)
    enc.setBytes(&sc, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: N, height: H, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// ---- Fp32-intermediate dispatch wrappers (mirror the fp16 path above) ----

func encGemvFp32V5(_ cb: MTLCommandBuffer, x: MTLBuffer, W: MTLBuffer, out: MTLBuffer,
                    B: Int, Din: Int, Dout: Int, quantOut: Bool = false) {
    // Route through the MMA-based batched GEMM unless explicitly disabled.
    // The scalar GEMV kernel reloads each W row from DRAM for every batch
    // element (~7 GB of weight traffic per projection at 2520 patches);
    // vision_gemm_fp32_mma amortizes by tiling 8 tokens × 8 outputs per TG.
    // VISION_GEMM_SCALAR=1 keeps the old kernel reachable for A/B.
    if ProcessInfo.processInfo.environment["VISION_GEMM_SCALAR"] == nil &&
       Din % 8 == 0 && Dout % 8 == 0 {
        return encVisionGemmMmaFp32(cb, x: x, W: W, out: out, B: B, Din: Din, Dout: Dout, quantOut: quantOut)
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseGemvFp32In32OutPSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var du = UInt32(Din), dou = UInt32(Dout)
    enc.setBytes(&du, length: 4, index: 3); enc.setBytes(&dou, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: Dout / 32, height: B, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
    // Scalar fallback doesn't quant internally; emit a separate pass.
    if quantOut { encQuantizeFp32ToBf16(cb, x: out, N: Dout, numVecs: B) }
}

// Batched MMA GEMM dispatcher. Kernel picked by env:
//   default:        v2 — 16×16 tile, 4 accums (best measured on M5 Max).
//   VISION_GEMM_V3: v3 — v2 + K-unroll=2. Halves barriers but MMA issue
//                        rate is the actual ceiling on this hardware, so
//                        it doesn't improve on v2 (kept reachable for A/B
//                        on future targets where barriers are the bottleneck).
//   VISION_GEMM_V1: v1 — 8×8 tile, 1 accum (legacy baseline, 1.25× slower).
// quantOut=true folds the following bf16-quantize dispatch into the output store.
func encVisionGemmMmaFp32(_ cb: MTLCommandBuffer, x: MTLBuffer, W: MTLBuffer, out: MTLBuffer,
                           B: Int, Din: Int, Dout: Int, quantOut: Bool = false) {
    precondition(Din % 8 == 0 && Dout % 8 == 0,
                 "vision_gemm_fp32_mma requires Din and Dout multiples of 8 (got \(Din), \(Dout))")
    let env = ProcessInfo.processInfo.environment
    let useV1 = env["VISION_GEMM_V1"] != nil
    let useV3 = env["VISION_GEMM_V3"] != nil && (Din % 16 == 0)
    let pso: MTLComputePipelineState
    let tile: Int
    if useV1 {
        pso = visionGemmFp32MmaPSO; tile = 8
    } else if useV3 {
        pso = visionGemmFp32MmaV3PSO; tile = 16
    } else {
        pso = visionGemmFp32MmaV2PSO; tile = 16
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var Bv = UInt32(B), Dv = UInt32(Din), Ov = UInt32(Dout), Qv = UInt32(quantOut ? 1 : 0)
    enc.setBytes(&Bv, length: 4, index: 3)
    enc.setBytes(&Dv, length: 4, index: 4)
    enc.setBytes(&Ov, length: 4, index: 5)
    enc.setBytes(&Qv, length: 4, index: 6)
    let qBlocks = (B + tile - 1) / tile
    let oBlocks = (Dout + tile - 1) / tile
    enc.dispatchThreadgroups(MTLSize(width: oBlocks, height: qBlocks, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Fused FFN gate+up GEMM + bf16-rounded gelu-combine. Writes Y bf16-rounded
// so the downstream downProj sees the same rounded residual HF does.
// Replaces: gate GEMM, q_gate, up GEMM, q_up, gelu*up, q_gelu (6 → 1).
func encVisionFfnGateUpGeluFp32(_ cb: MTLCommandBuffer,
                                 x: MTLBuffer, Wgate: MTLBuffer, Wup: MTLBuffer,
                                 out: MTLBuffer, B: Int, Din: Int, Dout: Int) {
    precondition(Din % 8 == 0 && Dout % 8 == 0,
                 "vision_ffn_gate_up_gelu requires Din and Dout multiples of 8 (got \(Din), \(Dout))")
    let env = ProcessInfo.processInfo.environment
    let useV1 = env["VISION_GEMM_V1"] != nil
    let useV3 = env["VISION_GEMM_V3"] != nil && (Din % 16 == 0)
    let pso: MTLComputePipelineState
    let tile: Int
    if useV1 {
        pso = visionFfnGateUpGeluFp32MmaPSO; tile = 8
    } else if useV3 {
        pso = visionFfnGateUpGeluFp32MmaV3PSO; tile = 16
    } else {
        pso = visionFfnGateUpGeluFp32MmaV2PSO; tile = 16
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(x,     offset: 0, index: 0)
    enc.setBuffer(Wgate, offset: 0, index: 1)
    enc.setBuffer(Wup,   offset: 0, index: 2)
    enc.setBuffer(out,   offset: 0, index: 3)
    var Bv = UInt32(B), Dv = UInt32(Din), Ov = UInt32(Dout)
    enc.setBytes(&Bv, length: 4, index: 4)
    enc.setBytes(&Dv, length: 4, index: 5)
    enc.setBytes(&Ov, length: 4, index: 6)
    let qBlocks = (B + tile - 1) / tile
    let oBlocks = (Dout + tile - 1) / tile
    enc.dispatchThreadgroups(MTLSize(width: oBlocks, height: qBlocks, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encRMSNormGFp32(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer, out: MTLBuffer,
                      D: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(rmsNormFp32PSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
    enc.setBuffer(gammaBuf, offset: 0, index: 2)
    var Dv = UInt32(D); var eps: Float = 1e-6
    enc.setBytes(&Dv, length: 4, index: 3); enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encRMSNormNoScaleFp32(_ cb: MTLCommandBuffer, x: MTLBuffer, out: MTLBuffer, D: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(rmsNormNoScaleFp32PSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
    var Dv = UInt32(D); var eps: Float = 1e-6
    enc.setBytes(&Dv, length: 4, index: 2); enc.setBytes(&eps, length: 4, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encVision2DRopeFp32(_ cb: MTLCommandBuffer, x: MTLBuffer, posX: MTLBuffer, posY: MTLBuffer,
                          N: Int, H: Int, HD: Int, theta: Float) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(vision2dRopeFp32PSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(posX, offset: 0, index: 1); enc.setBuffer(posY, offset: 0, index: 2)
    var Nv = UInt32(N), Hv = UInt32(H), HDv = UInt32(HD), thv = theta
    enc.setBytes(&Nv, length: 4, index: 3)
    enc.setBytes(&Hv, length: 4, index: 4)
    enc.setBytes(&HDv, length: 4, index: 5)
    enc.setBytes(&thv, length: 4, index: 6)
    enc.dispatchThreadgroups(MTLSize(width: N, height: H, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encVisionAttnPrefillFp32(_ cb: MTLCommandBuffer, Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer, O: MTLBuffer,
                                N: Int, H: Int, HD: Int, qkScale: Float,
                                paddingMask: MTLBuffer? = nil) {
    // Route through the flash-attention port unless explicitly disabled.
    // Bidirectional (no causal) + optional byte-mask + fp32 I/O with half
    // MMA inside the kernel. Vision HD is 72 = 9 × 8, a natural fit for
    // simdgroup_matrix<T, 8, 8>. Falls back to the scalar kernel when
    // VISION_ATTN_SCALAR=1 so the old path stays reachable for A/B.
    if ProcessInfo.processInfo.environment["VISION_ATTN_SCALAR"] == nil {
        return encVisionAttnFlashFp32(cb, Q: Q, K: K, V: V, O: O,
                                        N: N, H: H, HD: HD, qkScale: qkScale,
                                        paddingMask: paddingMask)
    }
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(visionAttnPrefillFp32PSO)
    enc.setBuffer(Q, offset: 0, index: 0); enc.setBuffer(K, offset: 0, index: 1)
    enc.setBuffer(V, offset: 0, index: 2); enc.setBuffer(O, offset: 0, index: 3)
    var Nv = UInt32(N), Hv = UInt32(H), HDv = UInt32(HD), sc = qkScale
    enc.setBytes(&Nv, length: 4, index: 4)
    enc.setBytes(&Hv, length: 4, index: 5)
    enc.setBytes(&HDv, length: 4, index: 6)
    enc.setBytes(&sc, length: 4, index: 7)
    enc.setBuffer(paddingMask ?? Q, offset: 0, index: 8)
    var use: UInt32 = (paddingMask != nil) ? 1 : 0
    enc.setBytes(&use, length: 4, index: 9)
    enc.dispatchThreadgroups(MTLSize(width: N, height: H, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Flash-attention dispatch. Grid: (H, ceil(N/8), B). 32 threads per TG.
// Slot-parallel batching: buffers are [B, N, H, D], each slot's K/V is
// distinct memory, so cross-slot attention is impossible by construction.
// B defaults to 1 for the single-image case — dispatcher degenerates cleanly.
func encVisionAttnFlashFp32(_ cb: MTLCommandBuffer, Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer, O: MTLBuffer,
                             N: Int, H: Int, HD: Int, qkScale: Float,
                             paddingMask: MTLBuffer? = nil, B: Int = 1) {
    precondition(HD == 72, "vision_attn_flash_fp32 hardcodes D=72 for Gemma-4")
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(visionAttnFlashFp32PSO)
    enc.setBuffer(Q, offset: 0, index: 0); enc.setBuffer(K, offset: 0, index: 1)
    enc.setBuffer(V, offset: 0, index: 2); enc.setBuffer(O, offset: 0, index: 3)
    var Nv = UInt32(N), Hv = UInt32(H), HDv = UInt32(HD), sc = qkScale
    enc.setBytes(&Nv, length: 4, index: 4)
    enc.setBytes(&Hv, length: 4, index: 5)
    enc.setBytes(&HDv, length: 4, index: 6)
    enc.setBytes(&sc, length: 4, index: 7)
    enc.setBuffer(paddingMask ?? Q, offset: 0, index: 8)
    var use: UInt32 = (paddingMask != nil) ? 1 : 0
    enc.setBytes(&use, length: 4, index: 9)
    let nBlocks = (N + 7) / 8
    enc.dispatchThreadgroups(MTLSize(width: H, height: nBlocks, depth: B),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encGeluMulFp32(_ cb: MTLCommandBuffer, gate: MTLBuffer, up: MTLBuffer, N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(geluMulFp32PSO)
    enc.setBuffer(gate, offset: 0, index: 0); enc.setBuffer(up, offset: 0, index: 1)
    var Nv = UInt32(N)
    enc.setBytes(&Nv, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encVisionPool2DFp32InFp32Out(_ cb: MTLCommandBuffer, x: MTLBuffer, out: MTLBuffer,
                                    gridW: Int, outW: Int, outH: Int, kernelSize: Int, hidden: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(visionPool2DFp32InFp32OutPSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
    var gw = UInt32(gridW), ow = UInt32(outW), ks = UInt32(kernelSize), h = UInt32(hidden)
    enc.setBytes(&gw, length: 4, index: 2); enc.setBytes(&ow, length: 4, index: 3)
    enc.setBytes(&ks, length: 4, index: 4); enc.setBytes(&h, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: outH * outW, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Round an fp32 buffer's values to bf16 precision in-place. Called at
// vision encoder layer boundaries to match HF's bf16 residual stream —
// HF runs `.type_as(bf16_hidden_states)` at the end of every RMSNorm,
// so every layer's input and output is bf16-rounded. Without this step
// our fp32 pipeline drifts from HF's reference by compounding rounding
// noise over 27 layers (measured MSE 0.054 at ev_out, enough to break
// LM image understanding downstream).
func encQuantizeFp32ToBf16(_ cb: MTLCommandBuffer, x: MTLBuffer, N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(quantizeFp32ToBf16PSO)
    enc.setBuffer(x, offset: 0, index: 0)
    var Nv = UInt32(N)
    enc.setBytes(&Nv, length: 4, index: 1)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encVisionScaledStdNormFp32(_ cb: MTLCommandBuffer, x: MTLBuffer, bias: MTLBuffer, scale: MTLBuffer,
                                 out: MTLBuffer, D: Int, numVecs: Int, globalScale: Float) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(visionStdNormFp32PSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(bias, offset: 0, index: 1)
    enc.setBuffer(scale, offset: 0, index: 2); enc.setBuffer(out, offset: 0, index: 3)
    var Dv = UInt32(D); var gs = globalScale
    enc.setBytes(&Dv, length: 4, index: 4); enc.setBytes(&gs, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encVisionPool2D(_ cb: MTLCommandBuffer, x: MTLBuffer, out: MTLBuffer,
                      gridW: Int, outW: Int, outH: Int, kernelSize: Int, hidden: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(visionPool2DPSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
    var gw = UInt32(gridW), ow = UInt32(outW), ks = UInt32(kernelSize), h = UInt32(hidden)
    enc.setBytes(&gw, length: 4, index: 2); enc.setBytes(&ow, length: 4, index: 3)
    enc.setBytes(&ks, length: 4, index: 4); enc.setBytes(&h, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: outH * outW, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encVisionScaledStdNorm(_ cb: MTLCommandBuffer, x: MTLBuffer, bias: MTLBuffer, scale: MTLBuffer,
                             out: MTLBuffer, D: Int, numVecs: Int, globalScale: Float) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(visionStdNormPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(bias, offset: 0, index: 1)
    enc.setBuffer(scale, offset: 0, index: 2)
    enc.setBuffer(out, offset: 0, index: 3)
    var Dv = UInt32(D); var gs = globalScale
    enc.setBytes(&Dv, length: 4, index: 4)
    enc.setBytes(&gs, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// Local helper: add `src` element-wise into `dst` (fp16). Reuses the
// text-side `add_inplace` MSL kernel if present. Fallback: walk device
// buffers as in the pre-existing encAddInplace.
// (encAddInplace already defined elsewhere in the file — we use it directly.)

// ---------- fp32-residual-stream dispatchers for vision tower -----------

func encGemvFp16InFp32Out(_ cb: MTLCommandBuffer, x: MTLBuffer, W: MTLBuffer, out: MTLBuffer,
                            B: Int, Din: Int, Dout: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(denseGemvFp32OutPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(W, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    var Dinv = UInt32(Din), Doutv = UInt32(Dout)
    enc.setBytes(&Dinv, length: 4, index: 3)
    enc.setBytes(&Doutv, length: 4, index: 4)
    let n_blocks = (Dout + 31) / 32
    enc.dispatchThreadgroups(MTLSize(width: n_blocks, height: B, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}

func encVisionPosEmbedAddFp32(_ cb: MTLCommandBuffer, x: MTLBuffer, posTable: MTLBuffer,
                                posMax: Int, out: MTLBuffer, N: Int, nPatchesX: Int, hidden: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(visionPosEmbedFp32PSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(posTable, offset: posMax * hidden * 2, index: 1)   // Y-table
    enc.setBuffer(posTable, offset: 0, index: 2)                     // X-table
    enc.setBuffer(out, offset: 0, index: 3)
    var nx = UInt32(nPatchesX), h = UInt32(hidden)
    enc.setBytes(&nx, length: 4, index: 4); enc.setBytes(&h, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encRMSNormGFp32In(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                         out: MTLBuffer, D: Int, numVecs: Int, eps: Float = 1e-6) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(rmsNormFp32InPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(out, offset: 0, index: 1)
    enc.setBuffer(gammaBuf, offset: 0, index: 2)
    var Dv = UInt32(D); var ev = eps
    enc.setBytes(&Dv, length: 4, index: 3); enc.setBytes(&ev, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encAddInplaceFp32(_ cb: MTLCommandBuffer, dst: MTLBuffer, src: MTLBuffer, N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addInplaceFp32PSO)
    enc.setBuffer(dst, offset: 0, index: 0)
    enc.setBuffer(src, offset: 0, index: 1)
    var Nv = UInt32(N); enc.setBytes(&Nv, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encRMSNormGFp32Out(_ cb: MTLCommandBuffer, x: MTLBuffer, gammaBuf: MTLBuffer,
                          out: MTLBuffer, D: Int, numVecs: Int, eps: Float = 1e-6) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(rmsNormFp32OutPSO)
    enc.setBuffer(x, offset: 0, index: 0)
    enc.setBuffer(out, offset: 0, index: 1)
    enc.setBuffer(gammaBuf, offset: 0, index: 2)
    var Dv = UInt32(D); var ev = eps
    enc.setBytes(&Dv, length: 4, index: 3); enc.setBytes(&ev, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encAddInplaceFp32Fp32(_ cb: MTLCommandBuffer, dst: MTLBuffer, src: MTLBuffer, N: Int, numVecs: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addInplaceFp32FpPSO)
    enc.setBuffer(dst, offset: 0, index: 0)
    enc.setBuffer(src, offset: 0, index: 1)
    var Nv = UInt32(N); enc.setBytes(&Nv, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: numVecs, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

func encVisionPool2DFp32In(_ cb: MTLCommandBuffer, x: MTLBuffer, out: MTLBuffer,
                             gridW: Int, outW: Int, outH: Int, kernelSize: Int, hidden: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(visionPool2DFp32InPSO)
    enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
    var gw = UInt32(gridW), ow = UInt32(outW), ks = UInt32(kernelSize), h = UInt32(hidden)
    enc.setBytes(&gw, length: 4, index: 2); enc.setBytes(&ow, length: 4, index: 3)
    enc.setBytes(&ks, length: 4, index: 4); enc.setBytes(&h, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: outH * outW, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// ===========================================================================
// Vision tower forward orchestrator. Takes a preprocessed PatchBatch (output
// of `gemma4ImagePreprocess`) and a fully-loaded `VisionWeights`, runs the
// 27-layer encoder + pool + std-normalize + embed_vision projection, returns
// a MTLBuffer of soft tokens shape [N_pooled, 2816] fp16. Single command
// buffer: one encode, one commit, one wait.
// ===========================================================================
// Quick buffer peek (commits any pending CB before reading).
func dumpFp16(_ buf: MTLBuffer, _ label: String, count: Int = 8, stride: Int = 0) {
    let p = buf.contents().assumingMemoryBound(to: Float16.self)
    var s = "    \(label):"
    for i in 0..<count {
        let v = Float(p[stride + i])
        s += v.isNaN ? " NaN" : (v.isInfinite ? " Inf" : String(format: " %+.3f", v))
    }
    print(s)
}


// Batched vision forward over B images at once. Slot-parallel layout:
// per-row ops (GEMMs, RMSNorm, RoPE, residuals) run with numVecs = B × N_max,
// amortizing weight loads across the batch; attention dispatches once per
// layer with a batch-slot axis in the grid (each slot's K/V lives in its
// own [N, H, D] region — cross-slot attention is impossible by construction).
//
// Handles B=1 and B>1 uniformly: the flat [B*N, ...] buffer layout reduces
// to [N, ...] at B=1 and runs the same kernel dispatches.
//
// Returns per-image (softTokens, nPooled) — each image has its own grid
// and thus its own pooled-token count, so outputs remain variable-length.
//
// _core encodes all vision work into a single command buffer but does not
// commit or wait. Two public wrappers call it: the sync variant commits
// and waits before returning; the async variant commits and returns the
// CB so the caller can wait lazily while other queues proceed.
private func _runVisionTowerBatchForwardCore(batches: [PatchBatch], weights: VisionWeights,
                                               device: MTLDevice, queue: MTLCommandQueue)
                                               -> ([(MTLBuffer, Int)], MTLCommandBuffer) {
    precondition(!batches.isEmpty)
    let B = batches.count
    let N = batches[0].maxPatches
    precondition(batches.allSatisfy { $0.maxPatches == N },
                 "all batches must share maxPatches (got mismatched N)")
    let h = weights.hidden, H = weights.numHeads, HD = weights.headDim
    let interm = weights.interm
    let BN = B * N

    // Per-image: positions, padding mask — concatenated into flat [B*N] buffers
    // so per-row kernels walk the whole batch in one dispatch.
    let posYBuf = device.makeBuffer(length: BN * 4, options: .storageModeShared)!
    let posXBuf = device.makeBuffer(length: BN * 4, options: .storageModeShared)!
    let paddingMaskBuf = device.makeBuffer(length: BN, options: .storageModeShared)!
    let pyp = posYBuf.contents().assumingMemoryBound(to: UInt32.self)
    let pxp = posXBuf.contents().assumingMemoryBound(to: UInt32.self)
    let pmp = paddingMaskBuf.contents().assumingMemoryBound(to: UInt8.self)
    for (b, batch) in batches.enumerated() {
        let base = b * N
        for i in 0..<N {
            pyp[base + i] = UInt32(max(batch.positions[i].0, 0))
            pxp[base + i] = UInt32(max(batch.positions[i].1, 0))
            pmp[base + i] = (i < batch.numRealPatches) ? 0 : 1
        }
    }

    // Concat patches — each image's [N, 768] fp16 into one [BN, 768] fp16.
    let patchesBuf = device.makeBuffer(length: BN * 768 * 2, options: .storageModeShared)!
    for (b, batch) in batches.enumerated() {
        let dst = patchesBuf.contents().advanced(by: b * N * 768 * 2)
        memcpy(dst, batch.patches.contents(), N * 768 * 2)
    }

    // Batched working buffers: all per-row state at [BN, ...] layout.
    let x       = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let tmp     = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let qBuf    = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let kBuf    = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let vBuf    = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let attnOut = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let gateAct = device.makeBuffer(length: BN * interm * 4, options: .storageModeShared)!
    let upAct   = device.makeBuffer(length: BN * interm * 4, options: .storageModeShared)!
    let mlpOutB = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!
    let postNormFp32 = device.makeBuffer(length: BN * h * 4, options: .storageModeShared)!

    // Single CB for the whole batched forward.
    let cb = queue.makeCommandBuffer()!

    // 1) Patch embed — one big dispatch over BN rows, weight streamed once.
    encGemvFp16InFp32Out(cb, x: patchesBuf, W: weights.patchEmbedW, out: x,
                          B: BN, Din: 768, Dout: h)

    // 2) Pos embed — per-image (each has its own gridW). Dispatch B times
    // with buffer offsets into the batched x.
    for (b, batch) in batches.enumerated() {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(visionPosEmbedFp32PSO)
        enc.setBuffer(x, offset: b * N * h * 4, index: 0)
        enc.setBuffer(weights.posEmbedTable, offset: weights.posMax * h * 2, index: 1)
        enc.setBuffer(weights.posEmbedTable, offset: 0, index: 2)
        enc.setBuffer(x, offset: b * N * h * 4, index: 3)
        var nx = UInt32(batch.gridW), hh = UInt32(h)
        enc.setBytes(&nx, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    // 3) 27 encoder layers — all per-row ops take numVecs = BN; attention
    // uses the batched grid axis. Same bf16-quant trajectory as B=1.
    let qkScale: Float = 1.0
    for L in 0..<weights.numLayers {
        let lw = weights.layers[L]
        // Attention sub-block
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
        // One batched attention dispatch: grid (H, ceil(N/8), B).
        encVisionAttnFlashFp32(cb, Q: qBuf, K: kBuf, V: vBuf, O: attnOut,
                                 N: N, H: H, HD: HD, qkScale: qkScale,
                                 paddingMask: paddingMaskBuf, B: B)
        encQuantizeFp32ToBf16(cb, x: attnOut, N: h, numVecs: BN)
        encGemvFp32V5(cb, x: attnOut, W: lw.oProj, out: tmp, B: BN, Din: h, Dout: h, quantOut: true)
        encRMSNormGFp32(cb, x: tmp, gammaBuf: lw.postAttnNorm, out: postNormFp32, D: h, numVecs: BN)
        encQuantizeFp32ToBf16(cb, x: postNormFp32, N: h, numVecs: BN)
        encAddInplaceFp32Fp32(cb, dst: x, src: postNormFp32, N: h, numVecs: BN)
        encQuantizeFp32ToBf16(cb, x: x, N: h, numVecs: BN)
        // FFN sub-block
        encRMSNormGFp32(cb, x: x, gammaBuf: lw.preFfnNorm, out: tmp, D: h, numVecs: BN)
        encQuantizeFp32ToBf16(cb, x: tmp, N: h, numVecs: BN)
        encVisionFfnGateUpGeluFp32(cb, x: tmp, Wgate: lw.gateProj, Wup: lw.upProj,
                                     out: gateAct, B: BN, Din: h, Dout: interm)
        encGemvFp32V5(cb, x: gateAct, W: lw.downProj, out: mlpOutB, B: BN, Din: interm, Dout: h, quantOut: true)
        encRMSNormGFp32(cb, x: mlpOutB, gammaBuf: lw.postFfnNorm, out: postNormFp32, D: h, numVecs: BN)
        encQuantizeFp32ToBf16(cb, x: postNormFp32, N: h, numVecs: BN)
        encAddInplaceFp32Fp32(cb, dst: x, src: postNormFp32, N: h, numVecs: BN)
        encQuantizeFp32ToBf16(cb, x: x, N: h, numVecs: BN)
    }

    // 4) Tail: pool + std_norm + pre_proj_norm + embed_vision_proj.
    // Each image has its own grid shape and pooled-token count; dispatch
    // per-image with buffer offsets. Weights (stdBias/stdScale/embedVisionProj)
    // are still streamed once across B per-image dispatches — same amortization
    // as the per-row ops since we're inside one CB.
    var results: [(MTLBuffer, Int)] = []
    results.reserveCapacity(B)
    let sqrtH = Float(h).squareRoot()
    for (b, batch) in batches.enumerated() {
        let outH = batch.gridH / 3, outW = batch.gridW / 3
        let nPooled = outH * outW
        let pooled    = device.makeBuffer(length: nPooled * h * 4, options: .storageModeShared)!
        let stdNormed = device.makeBuffer(length: nPooled * h * 4, options: .storageModeShared)!
        let softTokens = device.makeBuffer(length: nPooled * weights.textHidden * 4, options: .storageModeShared)!

        // Pool with per-image grid, reading from x at offset.
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(visionPool2DFp32InFp32OutPSO)
        enc.setBuffer(x, offset: b * N * h * 4, index: 0)
        enc.setBuffer(pooled, offset: 0, index: 1)
        var gw = UInt32(batch.gridW), ow = UInt32(outW), ks: UInt32 = 3, hh = UInt32(h)
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
        results.append((softTokens, nPooled))
    }

    return (results, cb)
}

// Sync wrapper — commits the CB and blocks until all vision kernels finish.
// Use when the caller needs the soft-token buffers immediately.
func runVisionTowerBatchForward(batches: [PatchBatch], weights: VisionWeights,
                                  device: MTLDevice, queue: MTLCommandQueue) -> [(MTLBuffer, Int)] {
    let (results, cb) = _runVisionTowerBatchForwardCore(
        batches: batches, weights: weights, device: device, queue: queue)
    cb.commit(); cb.waitUntilCompleted()
    if let e = cb.error { print("  [vision batch] GPU err: \(e)") }
    return results
}

// Async wrapper — encodes the vision forward and returns the **uncommitted**
// CB plus the per-image (rawSoftTokens, nPooled) results. The caller is
// responsible for `cb.commit()` (after attaching any
// `addCompletedHandler`s — Metal asserts when handlers are added after
// commit) and for ensuring the soft-token buffers are read only after
// the CB completes (either via downstream same-queue CB ordering or a
// completion handler). Used so vision work overlaps with LM decode on
// other queues.
func runVisionTowerBatchForwardAsync(batches: [PatchBatch], weights: VisionWeights,
                                      device: MTLDevice, queue: MTLCommandQueue)
                                      -> ([(MTLBuffer, Int)], MTLCommandBuffer) {
    let (results, cb) = _runVisionTowerBatchForwardCore(
        batches: batches, weights: weights, device: device, queue: queue)
    return (results, cb)
}
