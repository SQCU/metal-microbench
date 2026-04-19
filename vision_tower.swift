// vision_tower.swift — Gemma-4 vision encoder (27-layer SigLIP2-style).
// Extracted from the monofile forward_graph.swift in the 2026-04-18 refactor.
//
// Contents:
//   - Swift preprocessor (PIL-compatible bicubic+antialias resize, patchify)
//   - Vision tower weight loader (BF16 safetensors → FP16)
//   - Kernel dispatch wrappers (encGemvFp16V5, encVision*, encRMSNormGFp32*, …)
//   - runVisionTowerForward — 27-layer forward producing [nPooled, 2816] fp32
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
    let patches: MTLBuffer       // [maxPatches, 768] fp16
    let positions: [(Int32, Int32)]   // per-patch (y, x) positions; (-1, -1) for pad
    let numRealPatches: Int
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
private func loadImageRGB(_ path: String) throws -> (width: Int, height: Int, rgb: [UInt8]) {
    let url = URL(fileURLWithPath: path)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        throw PreprocessError.imageDecodeFailed(path)
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
func gemma4ImagePreprocess(path: String,
                             maxSoftTokens: Int = 280,
                             patchSize: Int = 16,
                             poolingKernel: Int = 3,
                             device: MTLDevice) throws -> PatchBatch {
    // Step 1: decode image
    let (origW, origH, rgbBytes) = try loadImageRGB(path)

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

    // Positions: (y, x) for real, (-1, -1) for padding.
    var positions = [(Int32, Int32)](repeating: (-1, -1), count: maxPatches)
    for py in 0..<gridH {
        for px in 0..<gridW {
            positions[py * gridW + px] = (Int32(py), Int32(px))
        }
    }

    return PatchBatch(patches: patchBuf, positions: positions,
                       numRealPatches: nReal, gridH: gridH, gridW: gridW,
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

func runVisionTowerForward(batch: PatchBatch, weights: VisionWeights,
                             device: MTLDevice, queue: MTLCommandQueue) -> (MTLBuffer, Int) {
    let debug = ProcessInfo.processInfo.environment["VISION_DEBUG"] != nil
    let N = batch.numRealPatches
    let gridH = batch.gridH, gridW = batch.gridW
    let h = weights.hidden, H = weights.numHeads, HD = weights.headDim
    let interm = weights.interm
    let outH = gridH / 3, outW = gridW / 3
    let nPooled = outH * outW

    // Positions (per-patch). batch.positions[i] is (y, x) for real patches.
    let posYBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!
    let posXBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!
    let pyp = posYBuf.contents().assumingMemoryBound(to: UInt32.self)
    let pxp = posXBuf.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0..<N {
        pyp[i] = UInt32(batch.positions[i].0)
        pxp[i] = UInt32(batch.positions[i].1)
    }

    // Scratch buffers. The residual stream `x` is fp32 (keeps accumulated
    // rounding from compounding over 27 layers on outlier-gamma channels);
    // soft token output is fp32 (downstream LM wants max precision pre-norm).
    // Intermediates between kernels stay fp16 — they're recomputed every layer
    // or renormalized per-token, so no accumulation drift.
    let x       = device.makeBuffer(length: N * h * 4, options: .storageModeShared)!   // fp32
    let tmp     = device.makeBuffer(length: N * h * 2, options: .storageModeShared)!
    let qBuf    = device.makeBuffer(length: N * h * 2, options: .storageModeShared)!
    let kBuf    = device.makeBuffer(length: N * h * 2, options: .storageModeShared)!
    let vBuf    = device.makeBuffer(length: N * h * 2, options: .storageModeShared)!
    let attnOut = device.makeBuffer(length: N * h * 2, options: .storageModeShared)!
    let gateAct = device.makeBuffer(length: N * interm * 2, options: .storageModeShared)!
    let upAct   = device.makeBuffer(length: N * interm * 2, options: .storageModeShared)!
    let mlpOutB = device.makeBuffer(length: N * h * 2, options: .storageModeShared)!
    // Post-norm outputs promoted to fp32 so residual-stream adds don't round
    // on outlier-gamma channels. One shared buffer is fine since post_attn_norm
    // and post_ffn_norm are sequential in the same layer.
    let postNormFp32 = device.makeBuffer(length: N * h * 4, options: .storageModeShared)!
    let pooled  = device.makeBuffer(length: nPooled * h * 2, options: .storageModeShared)!
    let stdNormed = device.makeBuffer(length: nPooled * h * 2, options: .storageModeShared)!
    let softTokens = device.makeBuffer(length: nPooled * weights.textHidden * 4, options: .storageModeShared)!  // fp32

    // Multi-CB in debug mode so we can inspect buffers between stages.
    func commitOne(_ cb: MTLCommandBuffer) {
        cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { print("  GPU err: \(e)") }
    }

    // VISION_DUMP_STAGES=<dir> — dump per-stage tensors into dir. Use bytesPerElem
    // to distinguish fp16 vs fp32 dumps.
    let dumpStagesDir = ProcessInfo.processInfo.environment["VISION_DUMP_STAGES"]
    let debugLayer = Int(ProcessInfo.processInfo.environment["VISION_DEBUG_LAYER"] ?? "") ?? -1
    // Fast path: single CB for the entire forward when neither dumping nor
    // per-step debugging is requested. Slow path: commit at each dump point
    // so buffer contents can be read back.
    let fastPath = (dumpStagesDir == nil) && (debugLayer < 0) && !debug
    func dumpStage(_ name: String, _ buf: MTLBuffer, _ nElem: Int, bytesPerElem: Int = 2) {
        guard let d = dumpStagesDir else { return }
        let nBytes = nElem * bytesPerElem
        let data = Data(bytes: buf.contents(), count: nBytes)
        let url = URL(fileURLWithPath: d).appendingPathComponent("swift_\(name).bin")
        try? data.write(to: url)
    }

    // Single-CB fast path: encode everything into one CB, commit once at end.
    var cb0 = queue.makeCommandBuffer()!
    // 1) Patch embed: [N, 768] @ W.T → [N, 1152] — write fp32 to x.
    encGemvFp16InFp32Out(cb0, x: batch.patches, W: weights.patchEmbedW, out: x,
                           B: N, Din: 768, Dout: h)
    // 2) 2D positional embedding add: fp32 x + fp16 pos → fp32 x.
    encVisionPosEmbedAddFp32(cb0, x: x, posTable: weights.posEmbedTable,
                               posMax: weights.posMax, out: x,
                               N: N, nPatchesX: gridW, hidden: h)
    if !fastPath {
        commitOne(cb0)
        dumpStage("patch_embed", x, N * h, bytesPerElem: 4)
    }

    // 3) 27 encoder layers. Gemma4V uses kq_scale = 1.0 (not 1/sqrt(HD))
    // per gemma4v.cpp line 93 — Q/K are RMSNorm'd so no additional scaling.
    let qkScale: Float = 1.0

    for L in 0..<weights.numLayers {
        let lw = weights.layers[L]
        let splitCBs = (L == debugLayer)
        // In fast path we never make a new layerCB — everything appends to cb0.
        let layerCB: MTLCommandBuffer = fastPath ? cb0 : queue.makeCommandBuffer()!

        // Helper: either batch into a single CB, or commit+dump+sample after
        // each step when splitCBs is true. At-CB granularity lets us pin the
        // NaN source.
        func step(_ tag: String, _ op: (MTLCommandBuffer) -> Void, dumpBuf: MTLBuffer, dumpStride: Int = 0, sampleCount: Int = 0) {
            if splitCBs {
                let c = queue.makeCommandBuffer()!
                op(c)
                commitOne(c)
                dumpFp16(dumpBuf, "L\(L) \(tag)", stride: dumpStride)
                if sampleCount > 0 {
                    let p = dumpBuf.contents().assumingMemoryBound(to: Float16.self)
                    var nan = 0, inf = 0, maxAbs: Float = 0, firstNaNIdx = -1
                    for s in 0..<sampleCount {
                        let v = Float(p[s])
                        if v.isNaN { nan += 1; if firstNaNIdx < 0 { firstNaNIdx = s } }
                        else if v.isInfinite { inf += 1 }
                        else { maxAbs = max(maxAbs, abs(v)) }
                    }
                    if nan > 0 || inf > 0 || maxAbs > 500 {
                        print("      ⚠ \(tag): \(nan) NaN (first @ \(firstNaNIdx)), \(inf) Inf / maxAbs=\(String(format: "%.2f", maxAbs))")
                    }
                }
            } else {
                op(layerCB)
            }
        }

        // === Attention sub-block ===
        let total = N * h
        step("input_norm", { encRMSNormGFp32In($0, x: x, gammaBuf: lw.inputNorm, out: tmp, D: h, numVecs: N) }, dumpBuf: tmp, sampleCount: total)
        step("qProj",      { encGemvFp16V5($0, x: tmp, W: lw.qProj, out: qBuf, B: N, Din: h, Dout: h) }, dumpBuf: qBuf, sampleCount: total)
        step("kProj",      { encGemvFp16V5($0, x: tmp, W: lw.kProj, out: kBuf, B: N, Din: h, Dout: h) }, dumpBuf: kBuf, sampleCount: total)
        step("vProj",      { encGemvFp16V5($0, x: tmp, W: lw.vProj, out: vBuf, B: N, Din: h, Dout: h) }, dumpBuf: vBuf, sampleCount: total)
        step("qNorm",      { encRMSNormG($0, x: qBuf, gammaBuf: lw.qNorm, out: qBuf, D: HD, numVecs: N * H) }, dumpBuf: qBuf, sampleCount: total)
        step("kNorm",      { encRMSNormG($0, x: kBuf, gammaBuf: lw.kNorm, out: kBuf, D: HD, numVecs: N * H) }, dumpBuf: kBuf, sampleCount: total)
        step("vNorm",      { encRMSNormNoScale($0, x: vBuf, out: vBuf, D: HD, numVecs: N * H) }, dumpBuf: vBuf, sampleCount: total)
        step("qRoPE",      { encVision2DRope($0, x: qBuf, posX: posXBuf, posY: posYBuf, N: N, H: H, HD: HD, theta: 100.0) }, dumpBuf: qBuf, sampleCount: total)
        step("kRoPE",      { encVision2DRope($0, x: kBuf, posX: posXBuf, posY: posYBuf, N: N, H: H, HD: HD, theta: 100.0) }, dumpBuf: kBuf, sampleCount: total)
        step("attn",       { encVisionAttnPrefill($0, Q: qBuf, K: kBuf, V: vBuf, O: attnOut, N: N, H: H, HD: HD, qkScale: qkScale) }, dumpBuf: attnOut, sampleCount: total)
        if splitCBs {
            // Sample stats at a few strided indices (avoid full-buffer scan).
            let qp = qBuf.contents().assumingMemoryBound(to: Float16.self)
            let vp = vBuf.contents().assumingMemoryBound(to: Float16.self)
            let ap = attnOut.contents().assumingMemoryBound(to: Float16.self)
            var qNaN = 0, vNaN = 0, aNaN = 0
            for s in stride(from: 0, to: N * h, by: 997) {
                if Float(qp[s]).isNaN { qNaN += 1 }
                if Float(vp[s]).isNaN { vNaN += 1 }
                if Float(ap[s]).isNaN { aNaN += 1 }
            }
            print("    L\(L) sampled ~\(N * h / 997): Q NaN=\(qNaN), V NaN=\(vNaN), attn NaN=\(aNaN)")
        }
        step("oProj",      { encGemvFp16V5($0, x: attnOut, W: lw.oProj, out: tmp, B: N, Din: h, Dout: h) }, dumpBuf: tmp, sampleCount: total)
        step("postAttnNorm", { encRMSNormGFp32Out($0, x: tmp, gammaBuf: lw.postAttnNorm, out: postNormFp32, D: h, numVecs: N) }, dumpBuf: postNormFp32, sampleCount: total)
        step("resid1",     { encAddInplaceFp32Fp32($0, dst: x, src: postNormFp32, N: h, numVecs: N) }, dumpBuf: x, sampleCount: total)
        // === FFN sub-block ===
        step("preFfnNorm", { encRMSNormGFp32In($0, x: x, gammaBuf: lw.preFfnNorm, out: tmp, D: h, numVecs: N) }, dumpBuf: tmp, sampleCount: total)
        step("gateProj",   { encGemvFp16V5($0, x: tmp, W: lw.gateProj, out: gateAct, B: N, Din: h, Dout: interm) }, dumpBuf: gateAct, sampleCount: N * interm)
        step("upProj",     { encGemvFp16V5($0, x: tmp, W: lw.upProj, out: upAct, B: N, Din: h, Dout: interm) }, dumpBuf: upAct, sampleCount: N * interm)
        if splitCBs {
            // Peek gate and up at the problem index (643, 2040) pre-GELU.
            let gp = gateAct.contents().assumingMemoryBound(to: Float16.self)
            let upp = upAct.contents().assumingMemoryBound(to: Float16.self)
            let patchBad = 643, dimBad = 2040
            let idx = patchBad * interm + dimBad
            print(String(format: "    pre-gelu probe: gate[%d,%d]=%.4f up[%d,%d]=%.4f",
                         patchBad, dimBad, Float(gp[idx]),
                         patchBad, dimBad, Float(upp[idx])))
        }
        step("geluMul",    { encGeluMul($0, gate: gateAct, up: upAct, N: interm, numVecs: N) }, dumpBuf: gateAct, sampleCount: N * interm)
        if splitCBs {
            let gp = gateAct.contents().assumingMemoryBound(to: Float16.self)
            let patchBad = 643, dimBad = 2040
            let idx = patchBad * interm + dimBad
            let v = Float(gp[idx])
            print(String(format: "    post-gelu probe: gateAct[%d,%d]=%@",
                         patchBad, dimBad, v.isNaN ? "NaN" : String(format: "%.4f", v)))
            // Scan a small neighborhood
            for offset in -8...8 {
                let ni = idx + offset
                if ni >= 0 && ni < N * interm {
                    let nv = Float(gp[ni])
                    if nv.isNaN { print("     NaN at linear idx \(ni) (offset \(offset))") }
                }
            }
        }
        step("downProj",   { encGemvFp16V5($0, x: gateAct, W: lw.downProj, out: mlpOutB, B: N, Din: interm, Dout: h) }, dumpBuf: mlpOutB, sampleCount: total)
        step("postFfnNorm",{ encRMSNormGFp32Out($0, x: mlpOutB, gammaBuf: lw.postFfnNorm, out: postNormFp32, D: h, numVecs: N) }, dumpBuf: postNormFp32, sampleCount: total)
        step("resid2",     { encAddInplaceFp32Fp32($0, dst: x, src: postNormFp32, N: h, numVecs: N) }, dumpBuf: x, sampleCount: total)

        if !splitCBs && !fastPath {
            commitOne(layerCB)
        }
        if dumpStagesDir != nil {
            dumpStage("layer_\(L)", x, N * h, bytesPerElem: 4)
        }
        if debug && (L == 0 || L == 1 || L == 13 || L == 26) && !splitCBs {
            dumpFp16(x, "after layer \(L) (x[tok0, 0..7])", stride: 0)
        }
    }

    // 4) Pool 3×3 avg over the patch grid. Reads fp32 x, writes fp16 pooled.
    let cbTail: MTLCommandBuffer = fastPath ? cb0 : queue.makeCommandBuffer()!
    encVisionPool2DFp32In(cbTail, x: x, out: pooled, gridW: gridW, outW: outW,
                            outH: outH, kernelSize: 3, hidden: h)

    // 5+6) Fused scale(sqrt(h)) + std-normalize: (x*sqrt(h) - std_bias) * std_scale.
    // sqrt(h) from Gemma4VisionPooler.forward:628; std-normalize from
    // Gemma4VisionModel.forward:1945.
    let sqrtH = Float(h).squareRoot()
    if !fastPath {
        commitOne(cbTail)
        dumpStage("pooled_raw", pooled, nPooled * h)
    }

    let cbStd: MTLCommandBuffer = fastPath ? cb0 : queue.makeCommandBuffer()!
    encVisionScaledStdNorm(cbStd, x: pooled, bias: weights.stdBias, scale: weights.stdScale,
                            out: stdNormed, D: h, numVecs: nPooled, globalScale: sqrtH)
    if !fastPath {
        commitOne(cbStd)
        dumpStage("std_normed", stdNormed, nPooled * h)
    }

    // 7) embed_vision.embedding_pre_projection_norm: RMSNorm(no scale).
    // modeling_gemma4.py:1973 — applied before the projection.
    let cbPre: MTLCommandBuffer = fastPath ? cb0 : queue.makeCommandBuffer()!
    encRMSNormNoScale(cbPre, x: stdNormed, out: stdNormed, D: h, numVecs: nPooled)
    if !fastPath {
        commitOne(cbPre)
        dumpStage("pre_proj_norm", stdNormed, nPooled * h)
    }

    // 8) embed_vision.embedding_projection: [nPooled, 1152] @ [2816, 1152].T → [nPooled, 2816].
    //    Emit fp32 — downstream LM input stream stays fp32 until the first RMSNorm.
    let cbProj: MTLCommandBuffer = fastPath ? cb0 : queue.makeCommandBuffer()!
    encGemvFp16InFp32Out(cbProj, x: stdNormed, W: weights.embedVisionProj, out: softTokens,
                           B: nPooled, Din: h, Dout: weights.textHidden)

    // Commit the single CB once (fast path) or the final tail CB (slow path).
    commitOne(cbProj)
    dumpStage("ev_out", softTokens, nPooled * weights.textHidden, bytesPerElem: 4)
    return (softTokens, nPooled)
}
