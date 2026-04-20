// vision_residency.swift
//
// Vision tower weight residency — memory-pressure-aware hydration.
//
// Goal: let this process co-tenant on a macOS user's desktop without pinning
// ~1.5 GB of vision weights in RAM forever, even though the vision tower is
// bursty (seconds of work per image, idle between requests).
//
// Strategy, three-level:
//
//   1. Source tier — zero-copy bf16 MTLBuffers pointing directly at the
//      mmap'd safetensors file. ALWAYS present. File-backed pages are clean
//      and non-anonymous, so the OS can evict them for free under pressure
//      and re-page on next access (no swap, no disk write). Near-zero RAM
//      cost when evicted. Startup cost: microseconds.
//
//   2. Working tier — fp16 MTLBuffers the vision kernels actually read.
//      Allocated lazily by `ensurePinned()`; populated by the `bf16_to_fp16`
//      GPU kernel reading from the source tier. Cost: ~1.5 GB of RAM and
//      ~10 ms GPU time to hydrate (a memory-bandwidth bound pass; fast even
//      on cold cache).
//
//   3. Residency policy — while pinned (actively serving an image request),
//      working buffers are `.nonVolatile` (guaranteed resident). When the
//      request finishes, `allowEvict()` marks them `.volatile` so macOS can
//      reclaim them under pressure. If pages were reclaimed, the next
//      `ensurePinned()` rehydrates from source.
//
//   4. Pressure source — `DispatchSource.makeMemoryPressureSource` fires on
//      `.warn` and `.critical`. `.warn` ⇒ flip to .volatile (passive). On
//      `.critical` ⇒ `forceDrop()` (active — drop working buffers + image
//      softs cache immediately; session K/V pages stay pinned so in-flight
//      generations aren't destroyed).
//
// Long-context corner: on a 128 GB M-series with a large model running a
// long-context conversation, K/V cache can reach 30+ GB. Vision weights
// become the natural eviction target — the three-state residency plus
// pressure subscription makes that tradeoff automatic.

import Foundation
import Metal

// Compute pipeline for bf16→fp16 conversion. Initialized by the module-
// level PSO table in bootstrap.swift.
let bf16ToFp16PSO: MTLComputePipelineState = pso("bf16_to_fp16")

// Dispatch helper: run bf16→fp16 kernel, n elements.
func encBf16ToFp16(_ cb: MTLCommandBuffer, src: MTLBuffer, dst: MTLBuffer, n: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(bf16ToFp16PSO)
    enc.setBuffer(src, offset: 0, index: 0)
    enc.setBuffer(dst, offset: 0, index: 1)
    var nv = UInt32(n)
    enc.setBytes(&nv, length: 4, index: 2)
    let tpg = 256
    let groups = (n + tpg - 1) / tpg
    enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
    enc.endEncoding()
}

final class VisionResidency {
    enum State { case unloaded, volatile_, pinned }

    private(set) var state: State = .unloaded
    private(set) var weights: VisionWeights?

    private let file: SafetensorsFile
    private var working: [MTLBuffer] = []        // fp16 working copies. Allocated on pinned/volatile.
                                                 // bf16 staging buffers live only for the duration of
                                                 // hydrate() — the mmap (SafetensorsFile) stays live
                                                 // so re-reading on reload is basically free (OS
                                                 // page cache).

    // Callback hooks for the pressure source. Set by the FFI on init.
    var onPressureCritical: (() -> Void)?

    init(file: SafetensorsFile) {
        self.file = file
    }

    // Hydrate to .pinned. Called by gemma_submit_image_path before vision forward.
    // Idempotent: no-op if already pinned. Re-pins from .volatile if pages
    // still live; otherwise rehydrates from source tier.
    func ensurePinned() throws {
        switch state {
        case .pinned:
            return
        case .volatile_:
            // Try to re-pin without re-converting. setPurgeableState returns
            // the PREVIOUS state — if any working buffer is .empty, the OS
            // reclaimed its pages and we have to rebuild.
            var evicted = false
            for buf in working {
                let prior = buf.setPurgeableState(.nonVolatile)
                if prior == .empty { evicted = true }
            }
            if !evicted {
                state = .pinned
                return
            }
            // Pages lost — drop refs and rehydrate from source.
            working.removeAll(keepingCapacity: true)
            weights = nil
            state = .unloaded
            fallthrough
        case .unloaded:
            try hydrate()
            for buf in working { _ = buf.setPurgeableState(.nonVolatile) }
            state = .pinned
        }
    }

    // Called after a vision forward completes. Flips working buffers to
    // .volatile so macOS can reclaim them under pressure — but keeps the
    // refs alive so an immediate next request doesn't have to rehydrate.
    func allowEvict() {
        guard state == .pinned else { return }
        for buf in working { _ = buf.setPurgeableState(.volatile) }
        state = .volatile_
    }

    // Called from the pressure source on .critical. Drops working buffers
    // entirely; source tier (mmap) stays since it costs ~0.
    func forceDrop() {
        working.removeAll(keepingCapacity: true)
        weights = nil
        state = .unloaded
        onPressureCritical?()
    }

    // --- hydrate ---

    private func hydrate() throws {
        let t0 = Date()
        working.removeAll(keepingCapacity: true)

        // Per-tensor staging buffers live only for this CB. Kept in a local
        // array so they stay alive until cb.waitUntilCompleted returns; the
        // array + buffers release right after.
        var stages: [MTLBuffer] = []
        stages.reserveCapacity(64)

        let cb = queue.makeCommandBuffer()!

        // Helper: bf16 staging (memcpy from mmap) + fresh fp16 working +
        // schedule GPU conversion.
        func load(_ name: String) throws -> MTLBuffer {
            let stage = try file.makeBF16StagingBuffer(name, device: device)
            let info = try file.tensor(name)
            let nElems = info.byteSize / 2
            let dst = device.makeBuffer(length: nElems * 2, options: .storageModeShared)!
            dst.label = "fp16:\(name)"
            stages.append(stage)
            working.append(dst)
            encBf16ToFp16(cb, src: stage, dst: dst, n: nElems)
            return dst
        }

        let patchEmbed = try load("model.vision_tower.patch_embedder.input_proj.weight")
        let posTable   = try load("model.vision_tower.patch_embedder.position_embedding_table")
        let stdBias    = try load("model.vision_tower.std_bias")
        let stdScale   = try load("model.vision_tower.std_scale")
        let embedVision = try load("model.embed_vision.embedding_projection.weight")

        var layers: [VisionLayerW] = []; layers.reserveCapacity(27)
        for L in 0..<27 {
            let p = "model.vision_tower.encoder.layers.\(L)."
            let lw = VisionLayerW(
                inputNorm:    try load("\(p)input_layernorm.weight"),
                qProj:        try load("\(p)self_attn.q_proj.linear.weight"),
                kProj:        try load("\(p)self_attn.k_proj.linear.weight"),
                vProj:        try load("\(p)self_attn.v_proj.linear.weight"),
                oProj:        try load("\(p)self_attn.o_proj.linear.weight"),
                qNorm:        try load("\(p)self_attn.q_norm.weight"),
                kNorm:        try load("\(p)self_attn.k_norm.weight"),
                postAttnNorm: try load("\(p)post_attention_layernorm.weight"),
                preFfnNorm:   try load("\(p)pre_feedforward_layernorm.weight"),
                gateProj:     try load("\(p)mlp.gate_proj.linear.weight"),
                upProj:       try load("\(p)mlp.up_proj.linear.weight"),
                downProj:     try load("\(p)mlp.down_proj.linear.weight"),
                postFfnNorm:  try load("\(p)post_feedforward_layernorm.weight")
            )
            layers.append(lw)
        }

        cb.commit(); cb.waitUntilCompleted()

        self.weights = VisionWeights(
            patchEmbedW: patchEmbed, posEmbedTable: posTable,
            stdBias: stdBias, stdScale: stdScale,
            embedVisionProj: embedVision, layers: layers)

        let dt = Date().timeIntervalSince(t0)
        print(String(format: "  [vision] hydrated bf16→fp16 in %.2f s (%d buffers, ~%.1f MB)",
                     dt, working.count,
                     Double(working.reduce(0) { $0 + $1.length }) / (1024.0 * 1024.0)))
    }

    // --- stats ---

    func workingSetBytes() -> Int {
        return working.reduce(0) { $0 + $1.length }
    }
}
