// Standalone probe: print static threadgroup memory + max threads/TG for
// the kernels we suspect of overflowing the 32 KB Metal hardware limit.
// Compiles only kernels.swift (which exports `msl`) — no bootstrap, no
// weight loading. Tells us the static-allocation accounting Metal records
// at PSO compile time.
//
// Usage:
//   swiftc -O probe_threadgroup_mem.swift kernels.swift \
//       -o probe_threadgroup_mem -framework Metal -framework Foundation
//   ./probe_threadgroup_mem

import Foundation
import Metal

@main
struct ProbeThreadgroupMem {
    static func main() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fputs("no Metal device\n", stderr); exit(1)
        }
        print("device: \(device.name)")
        print("max tg-memory length:        \(device.maxThreadgroupMemoryLength) B")
        print("max threads per threadgroup: \(device.maxThreadsPerThreadgroup)")
        print("rec max working-set size:    \(device.recommendedMaxWorkingSetSize / (1024*1024)) MB")
        print()

        let opts = MTLCompileOptions()
        if #available(macOS 15.0, *) { opts.languageVersion = .version3_2 }
        let lib: MTLLibrary
        do { lib = try device.makeLibrary(source: msl, options: opts) }
        catch { fputs("MSL compile failed: \(error)\n", stderr); exit(1) }

        // Plain PSOs (no function constants).
        let plainNames = [
            // Refactored slide kernel (now Q_PER_TG=1).
            "flex_attn_slide_v1_q8",
            // Full prefill kernel (D=512, Q_BLOCK=8, single-q-head-per-TG).
            "flex_attn_full_prefill",
            // Q8_0 fused-RMS QKV: claimed-suspect for h_norms[B*2816].
            "dense_gemv_q8_0_btile_qkv_b1",
            "dense_gemv_q8_0_btile_qkv_b2",
            "dense_gemv_q8_0_btile_qkv_b4",
            // No b8 of the non-otf variant (would be 45 KB).
            // OTF variant — re-derives RMS-norm inside kb loop (no h_norms).
            "extract_logprobs",
            "sample_token",
            "dense_gemv_q8_0_btile_qkv_otf_b1",
            "dense_gemv_q8_0_btile_qkv_otf_b2",
            "dense_gemv_q8_0_btile_qkv_otf_b4",
            "dense_gemv_q8_0_btile_qkv_otf_b8",
            "dense_gemv_f16_btile_qkv_otf_b1",
            "dense_gemv_f16_btile_qkv_otf_b2",
            "dense_gemv_f16_btile_qkv_otf_b4",
            "dense_gemv_f16_btile_qkv_otf_b8",
            "dense_gemv_f16_btile_gate_up_otf_b1",
            "dense_gemv_f16_btile_gate_up_otf_b2",
            "dense_gemv_f16_btile_gate_up_otf_b4",
            "dense_gemv_f16_btile_gate_up_otf_b8",
            "dense_gemv_f16_v6_rmsnorm_qkv",
            "dense_gemv_f16_v6_rmsnorm_gate_up",
            "dense_gemv_q8_0_v6_rmsnorm_qkv",
            // Tiled variant — claimed 16,448 B static.
            "dense_gemv_q8_0_btile_qkv_tiled_b1",
            "dense_gemv_q8_0_btile_qkv_tiled_b2",
            "dense_gemv_q8_0_btile_qkv_tiled_b4",
            "dense_gemv_q8_0_btile_qkv_tiled_b8",
        ]

        // Function-constant variants. flex_attn_v0 has 3 FCs (PAGE is a
        // true constant 16 in the kernel now):
        //   0=FC_D, 1=FC_Q_PER_TG, 2=FC_USE_SLIDE
        struct FCSpec { let name: String; let label: String;
                         let d: Int32; let qPerTG: Int32; let useSlide: Bool }
        let fcSpecs: [FCSpec] = [
            FCSpec(name: "flex_attn_v0", label: "flex_attn_v0/SLIDE [D=256,PAGE=16,Q_PER_TG=2]",
                   d: 256, qPerTG: 2, useSlide: true),
            FCSpec(name: "flex_attn_v0", label: "flex_attn_v0/FULL  [D=512,PAGE=16,Q_PER_TG=8]",
                   d: 512, qPerTG: 8, useSlide: false),
        ]

        let fmt: (Int) -> String = { v in
            String(format: "%6d B  (%5.2f KB)  [%2d%% of 32K]",
                   v, Double(v)/1024.0, (v * 100) / 32768)
        }
        print("=== plain PSOs ===")
        for n in plainNames {
            guard let f = lib.makeFunction(name: n) else {
                print("  [missing] \(n)"); continue
            }
            do {
                let pso = try device.makeComputePipelineState(function: f)
                let tg = pso.staticThreadgroupMemoryLength
                let mt = pso.maxTotalThreadsPerThreadgroup
                let mark = tg > 32768 ? "  ⚠️ OVER 32K" : (tg > 16384 ? "  (>16K, simdgroup_matrix doubling could push over)" : "")
                print("  \(fmt(tg))  maxThreads=\(mt)  \(n)\(mark)")
            } catch {
                print("  [error] \(n): \(error)")
            }
        }
        print()
        print("=== function-constant PSOs ===")
        for s in fcSpecs {
            let fcv = MTLFunctionConstantValues()
            var d = s.d, q = s.qPerTG, u = s.useSlide
            fcv.setConstantValue(&d, type: .int, index: 0)
            fcv.setConstantValue(&q, type: .int, index: 1)
            fcv.setConstantValue(&u, type: .bool, index: 2)
            do {
                let f = try lib.makeFunction(name: s.name, constantValues: fcv)
                let pso = try device.makeComputePipelineState(function: f)
                let tg = pso.staticThreadgroupMemoryLength
                let mt = pso.maxTotalThreadsPerThreadgroup
                let mark = tg > 32768 ? "  ⚠️ OVER 32K" : (tg > 16384 ? "  (>16K, simdgroup_matrix doubling could push over)" : "")
                print("  \(fmt(tg))  maxThreads=\(mt)  \(s.label)\(mark)")
            } catch {
                print("  [error] \(s.label): \(error)")
            }
        }
    }
}
