import Metal
import Foundation

// Memory mountain: sweep working-set size S from 4 KB to 1 GB, measure
// sustained read BW. Each thread walks the first n_elems of a fixed 1 GB
// buffer for a fixed iter count, wrapping via power-of-2 bitmask. The
// working-set-size-at-which-BW-drops is a direct read of the cache-tier
// step functions on this silicon.

let mslSource = """
#include <metal_stdlib>
using namespace metal;

kernel void mm_read(
    device const uint4* buf [[buffer(0)]],
    device uint4* sink [[buffer(1)]],
    constant uint& iters [[buffer(2)]],
    constant uint& mask [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid & mask;
    uint4 acc = uint4(0);
    for (uint i = 0; i < iters; i++) {
        acc ^= buf[idx];
        idx = (idx + 1) & mask;
    }
    sink[gid & 1023] = acc;
}

// Write-BW variant: XOR a running value into sink[idx] at each step. Forces
// device writes (unlike the read variant's 4 KB coalesced sink). Same wrap.
kernel void mm_write(
    device uint4* buf [[buffer(0)]],
    constant uint& iters [[buffer(1)]],
    constant uint& mask [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid & mask;
    uint4 v = uint4(gid, gid ^ 0x5A5A5A5A, gid * 2654435761u, gid + 1);
    for (uint i = 0; i < iters; i++) {
        buf[idx] ^= v;
        v += uint4(1);
        idx = (idx + 1) & mask;
    }
}

// Random-stride read: each step lands at a pseudo-random index within the
// working set. Defeats HW prefetcher and linear coalescing, so BW drops to
// whatever cache tier actually holds the line. Reveals latency-bound ceiling.
kernel void mm_read_random(
    device const uint4* buf [[buffer(0)]],
    device uint4* sink [[buffer(1)]],
    constant uint& iters [[buffer(2)]],
    constant uint& mask [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid & mask;
    uint4 acc = uint4(0);
    for (uint i = 0; i < iters; i++) {
        acc ^= buf[idx];
        idx = ((idx * 2654435761u) + 1) & mask;
    }
    sink[gid & 1023] = acc;
}
"""

func fail(_ msg: String) -> Never {
    fputs("error: \(msg)\n", stderr); exit(1)
}

guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let queue = device.makeCommandQueue()!

let opts = MTLCompileOptions()
if #available(macOS 15.0, *) { opts.languageVersion = .version3_2 }

let library: MTLLibrary
do { library = try device.makeLibrary(source: mslSource, options: opts) }
catch { fail("MSL compile: \(error)") }

guard let readFn = library.makeFunction(name: "mm_read"),
      let writeFn = library.makeFunction(name: "mm_write"),
      let randFn = library.makeFunction(name: "mm_read_random") else { fail("function lookup failed") }
let readPSO = try! device.makeComputePipelineState(function: readFn)
let writePSO = try! device.makeComputePipelineState(function: writeFn)
let randPSO = try! device.makeComputePipelineState(function: randFn)

print("device: \(device.name)")
print("unified memory: \(device.hasUnifiedMemory), max tg-mem: \(device.maxThreadgroupMemoryLength) B")
print("")

let maxSizeBytes = 1 * 1024 * 1024 * 1024
let bigBuf: MTLBuffer = {
    let b = device.makeBuffer(length: maxSizeBytes, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: UInt32.self, capacity: maxSizeBytes / 4)
    var s: UInt32 = 0xDEADBEEF
    for i in 0..<(maxSizeBytes / 4) {
        s = s &* 1664525 &+ 1013904223
        p[i] = s
    }
    return b
}()
let sink = device.makeBuffer(length: 4096 * 16, options: .storageModeShared)!

// Fixed grid size; iters per thread chosen so total loads ≈ 2 GiB regardless
// of working set (keeps dispatch time comparable across sizes).
let gridWidth = 65536
let totalLoadsTarget: UInt64 = 2 * 1024 * 1024 * 1024 / 16  // 2 GiB of uint4 loads
let itersPerThread = UInt32(totalLoadsTarget / UInt64(gridWidth))

func runReadSweep(_ pso: MTLComputePipelineState, writeMode: Bool) {
    var size = 4096
    while size <= maxSizeBytes {
        let nVec4 = size / 16
        var mask = UInt32(nVec4 - 1)
        var iters = itersPerThread
        var bestTime = Double.infinity
        let warmup = 2
        let reps = 8
        for r in 0..<(reps + warmup) {
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            enc.setBuffer(bigBuf, offset: 0, index: 0)
            if writeMode {
                enc.setBytes(&iters, length: 4, index: 1)
                enc.setBytes(&mask, length: 4, index: 2)
            } else {
                enc.setBuffer(sink, offset: 0, index: 1)
                enc.setBytes(&iters, length: 4, index: 2)
                enc.setBytes(&mask, length: 4, index: 3)
            }
            let tg = MTLSize(width: pso.threadExecutionWidth, height: 1, depth: 1)
            let grid = MTLSize(width: gridWidth, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            let t = cb.gpuEndTime - cb.gpuStartTime
            if r >= warmup { bestTime = min(bestTime, t) }
        }
        let totalBytes = Double(gridWidth) * Double(iters) * 16.0
        let gbps = totalBytes / bestTime / 1e9
        let sizeStr = String(format: "%9d B", size).padding(toLength: 11, withPad: " ", startingAt: 0)
        let humanStr = "(\(humanSize(size)))".padding(toLength: 9, withPad: " ", startingAt: 0)
        let msStr = String(format: "%6.2f ms", bestTime * 1000)
        let gbpsStr = String(format: "%7.1f GB/s", gbps)
        print("  \(sizeStr) \(humanStr)  \(msStr)   \(gbpsStr)")
        fflush(stdout)
        size *= 2
    }
}

func humanSize(_ b: Int) -> String {
    if b < 1024 { return "\(b)B" }
    if b < 1024*1024 { return "\(b/1024)KB" }
    if b < 1024*1024*1024 { return "\(b/1024/1024)MB" }
    return "\(b/1024/1024/1024)GB"
}

print("=== sequential-stride read (note: effective touched region capped at ~1 MB ===")
print("    because threads all start near gid and walk +1 for \(itersPerThread) steps, so this")
print("    measures sustained L2-hot-loop BW, not a true variable-working-set sweep)")
runReadSweep(readPSO, writeMode: false)

print("\n=== random-stride read (real memory mountain: whole working set visited) ===")
runReadSweep(randPSO, writeMode: false)

print("\n=== sequential-stride RMW write (read-modify-write, heavy L2 reuse) ===")
runReadSweep(writePSO, writeMode: true)
