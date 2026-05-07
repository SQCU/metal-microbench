// common.swift — shared infrastructure for the Metal Gemma-4 inference engine.
// Extracted from forward_graph.swift in the 2026-04-18 refactor.
//
// Contents:
//   - fail()          — uniform error-exit helper
//   - device/queue    — singleton MTLDevice + MTLCommandQueue
//   - lib             — compiled MTLLibrary from kernels.swift's `msl` source
//   - pso(_:)         — PSO lookup helper
//   - GGUFFile        — GGUF weight loader (Q4_0/Q4_K/Q5_1/Q8_0/etc.)
//   - SafetensorsFile — safetensors reader (bf16→fp16 conversion)
//
// These are top-level `let`/`func` declarations, so they initialize before
// any main.swift top-level statement runs. Other files (vision_tower.swift,
// kernels.swift, …) can depend on `lib`, `pso`, etc. safely.

import Metal
import Foundation

func fail(_ m: String) -> Never { fputs("error: \(m)\n", stderr); exit(1) }

let device: MTLDevice = {
    guard let d = MTLCreateSystemDefaultDevice() else { fail("no Metal device available") }
    return d
}()

let queue: MTLCommandQueue = {
    guard let q = device.makeCommandQueue() else { fail("failed to create MTLCommandQueue") }
    return q
}()

// Compile the MSL library once. Depends on `msl` (defined in kernels.swift).
let lib: MTLLibrary = {
    let opts = MTLCompileOptions()
    if #available(macOS 15.0, *) { opts.languageVersion = .version3_2 }
    do { return try device.makeLibrary(source: msl, options: opts) }
    catch { fail("MSL compile failed: \(error)") }
}()

func pso(_ name: String) -> MTLComputePipelineState {
    guard let f = lib.makeFunction(name: name) else { fail("no MSL function: \(name)") }
    return try! device.makeComputePipelineState(function: f)
}

// PSO with function-constant specialization. Caller's closure populates an
// MTLFunctionConstantValues; we then materialize the specialized variant
// of the kernel and build a PSO from it.
func psoFC(_ name: String, _ setup: (MTLFunctionConstantValues) -> Void) -> MTLComputePipelineState {
    let fcv = MTLFunctionConstantValues()
    setup(fcv)
    do {
        let f = try lib.makeFunction(name: name, constantValues: fcv)
        return try device.makeComputePipelineState(function: f)
    } catch {
        fail("psoFC(\(name)): \(error)")
    }
}

// ===========================================================================
// Kernel-capability matrix — the single source of truth for which
// (tensor_class, format) cells the engine has kernels for. Same JSON file
// the Python search reads (tools/quant_search/quant_driver.py); both
// interfaces share one definition so they can never silently drift.
//
// At engine boot we load and validate it (asserting every format declared
// has the right block geometry; every tensor_class has a non-empty allowed
// list). The auto-loaders in bootstrap.swift then check each tensor's
// dtype against `allowedFormats(for: tensor_class)` and fail loud with a
// helpful message if a GGUF demands something not declared here.
// ===========================================================================

struct KernelFormatInfo: Codable {
    let blk_bytes: Int
    let blk_elems: Int
    let bpw_eff: Float
    let notes: String
}

struct KernelTensorClassInfo: Codable {
    let kind: String
    let role: String
    let allowed: [String]
}

struct KernelLlamaQuantizeMix: Codable {
    let tag: String
    let kind: String
    let format: String?
    let moe_up: String?
    let moe_down: String?
    let dense_default: String?
    let note: String?
}

struct KernelCapabilities: Codable {
    let version: Int
    let description: String
    let formats: [String: KernelFormatInfo]
    let tensor_classes: [String: KernelTensorClassInfo]
    let llama_quantize_mixes: [KernelLlamaQuantizeMix]
}

let kernelCapabilities: KernelCapabilities = {
    let envPath = ProcessInfo.processInfo.environment["KERNEL_CAPABILITIES_JSON"]
    let candidates: [String] = [
        envPath,
        "/Users/mdot/metal-microbench/kernel_capabilities.json",
        FileManager.default.currentDirectoryPath + "/kernel_capabilities.json",
    ].compactMap { $0 }
    var loaded: KernelCapabilities? = nil
    var triedPaths: [String] = []
    for path in candidates {
        triedPaths.append(path)
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { continue }
        do {
            loaded = try JSONDecoder().decode(KernelCapabilities.self, from: data)
            break
        } catch {
            fail("kernel_capabilities.json at \(path) failed to decode: \(error)")
        }
    }
    guard let caps = loaded else {
        fail("kernel_capabilities.json not found at any of: \(triedPaths.joined(separator: ", "))")
    }
    precondition(caps.version == 1, "kernel_capabilities.json version \(caps.version) — engine expects 1")
    precondition(!caps.formats.isEmpty, "kernel_capabilities.json: empty formats")
    precondition(!caps.tensor_classes.isEmpty, "kernel_capabilities.json: empty tensor_classes")
    return caps
}()

// Map each format's GGMLType to its declared block geometry. Used by the
// auto-loaders to pick the right loadDenseSwizzled / loadMoESwizzled args.
func kernelFormatBlockGeometry(_ dtype: GGMLType) -> (blkBytes: Int, blkElems: Int)? {
    let tag = ggmlTypeToCapabilityTag(dtype)
    guard let info = kernelCapabilities.formats[tag] else { return nil }
    return (info.blk_bytes, info.blk_elems)
}

// Convert GGMLType to the JSON's format tag (uppercase).
func ggmlTypeToCapabilityTag(_ dtype: GGMLType) -> String {
    switch dtype {
    case .q8_0: return "Q8_0"
    case .q6_K: return "Q6_K"
    case .q5_K: return "Q5_K"
    case .q5_1: return "Q5_1"
    case .q5_0: return "Q5_0"
    case .q4_K: return "Q4_K"
    case .q4_1: return "Q4_1"
    case .q4_0: return "Q4_0"
    case .q3_K: return "Q3_K"
    case .q2_K: return "Q2_K"
    case .f16:  return "F16"
    case .f32:  return "F32"
    case .bf16: return "BF16"
    default:    return "\(dtype)"
    }
}

// Assert that `dtype` is allowed for `tensorClass` per the capabilities
// matrix. Called from the auto-loaders before each tensor load.
func assertCapability(_ tensorClass: String, _ dtype: GGMLType, tensorName: String) {
    guard let cls = kernelCapabilities.tensor_classes[tensorClass] else {
        fail("kernel_capabilities.json has no entry for tensor class '\(tensorClass)'")
    }
    let tag = ggmlTypeToCapabilityTag(dtype)
    if !cls.allowed.contains(tag) {
        fail("\(tensorName): dtype \(tag) not in allowed list for class '\(tensorClass)' (allowed: \(cls.allowed)). Either add a kernel + extend kernel_capabilities.json, or re-quantize with an allowed format.")
    }
}

// ===========================================================================
// Inlined GGUF v3 reader (was gguf_loader.swift; inlined here so swiftc
// treats forward_graph.swift as a single top-level source file).
// ===========================================================================

enum GGUFValueType: UInt32 {
    case uint8 = 0, int8 = 1, uint16 = 2, int16 = 3
    case uint32 = 4, int32 = 5, float32 = 6, bool = 7
    case string = 8, array = 9, uint64 = 10, int64 = 11, float64 = 12
}

enum GGMLType: UInt32 {
    case f32 = 0, f16 = 1
    case q4_0 = 2, q4_1 = 3
    case q5_0 = 6, q5_1 = 7, q8_0 = 8, q8_1 = 9
    case q2_K = 10, q3_K = 11, q4_K = 12, q5_K = 13, q6_K = 14, q8_K = 15
    case iq2_xxs = 16, iq2_xs = 17, iq3_xxs = 18, iq1_s = 19, iq4_nl = 20
    case iq3_s = 21, iq2_s = 22, iq4_xs = 23
    case i8 = 24, i16 = 25, i32 = 26, i64 = 27
    case f64 = 28
    case iq1_m = 29, bf16 = 30
    case tq1_0 = 34, tq2_0 = 35
    case iq4_nl_4_4 = 36, iq4_nl_4_8 = 37, iq4_nl_8_8 = 38
}

enum GGUFError: Error {
    case badMagic
    case unsupportedVersion(UInt32)
    case badMmap(String)
    case tensorNotFound(String)
    case readPastEnd
}

struct GGUFTensorInfo {
    let name: String
    let dtype: GGMLType
    let shape: [Int]
    let dataOffset: Int
    let byteSize: Int
}

final class GGUFFile {
    let path: String
    let fd: Int32
    let size: Int
    let base: UnsafeMutableRawPointer
    var cursor: Int = 0

    var metadata: [String: Any] = [:]
    var tensors: [String: GGUFTensorInfo] = [:]
    var alignment: Int = 32
    var dataSectionStart: Int = 0

    init(_ path: String) throws {
        self.path = path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw GGUFError.badMmap("open: \(String(cString: strerror(errno)))") }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            close(fd); throw GGUFError.badMmap("fstat failed")
        }
        self.fd = fd
        self.size = Int(st.st_size)
        guard let p = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
            close(fd); throw GGUFError.badMmap("mmap failed")
        }
        self.base = p
        try parse()
    }

    deinit {
        munmap(base, size)
        close(fd)
    }

    private func ensure(_ n: Int) throws {
        if cursor + n > size { throw GGUFError.readPastEnd }
    }
    private func readU32() throws -> UInt32 {
        try ensure(4)
        let v = base.advanced(by: cursor).load(as: UInt32.self); cursor += 4; return v
    }
    private func readU64() throws -> UInt64 {
        try ensure(8)
        let v = base.advanced(by: cursor).load(as: UInt64.self); cursor += 8; return v
    }
    private func readI32() throws -> Int32 {
        try ensure(4)
        let v = base.advanced(by: cursor).load(as: Int32.self); cursor += 4; return v
    }
    private func readI64() throws -> Int64 {
        try ensure(8)
        let v = base.advanced(by: cursor).load(as: Int64.self); cursor += 8; return v
    }
    private func readF32() throws -> Float {
        try ensure(4)
        let v = base.advanced(by: cursor).load(as: Float.self); cursor += 4; return v
    }
    private func readString() throws -> String {
        let len = try Int(readU64())
        try ensure(len)
        let bytes = UnsafeBufferPointer(
            start: base.advanced(by: cursor).assumingMemoryBound(to: UInt8.self),
            count: len)
        cursor += len
        return String(decoding: bytes, as: UTF8.self)
    }
    private func readValue(type: GGUFValueType) throws -> Any {
        switch type {
        case .uint8:   try ensure(1); let v = base.advanced(by: cursor).load(as: UInt8.self);  cursor += 1; return v
        case .int8:    try ensure(1); let v = base.advanced(by: cursor).load(as: Int8.self);   cursor += 1; return v
        case .uint16:  try ensure(2); let v = base.advanced(by: cursor).load(as: UInt16.self); cursor += 2; return v
        case .int16:   try ensure(2); let v = base.advanced(by: cursor).load(as: Int16.self);  cursor += 2; return v
        case .uint32:  return try readU32()
        case .int32:   return try readI32()
        case .float32: return try readF32()
        case .bool:    try ensure(1); let v = base.advanced(by: cursor).load(as: UInt8.self);  cursor += 1; return v != 0
        case .string:  return try readString()
        case .uint64:  return try readU64()
        case .int64:   return try readI64()
        case .float64: try ensure(8); let v = base.advanced(by: cursor).load(as: Double.self); cursor += 8; return v
        case .array:
            let innerRaw = try readU32()
            guard let inner = GGUFValueType(rawValue: innerRaw) else { throw GGUFError.readPastEnd }
            let count = try Int(readU64())
            var arr: [Any] = []; arr.reserveCapacity(count)
            for _ in 0..<count { arr.append(try readValue(type: inner)) }
            return arr
        }
    }

    private func parse() throws {
        let magic = try readU32()
        guard magic == 0x46554747 else { throw GGUFError.badMagic }
        let version = try readU32()
        guard version == 3 else { throw GGUFError.unsupportedVersion(version) }
        let tensorCount = try Int(readU64())
        let metaCount   = try Int(readU64())
        for _ in 0..<metaCount {
            let key = try readString()
            let typeRaw = try readU32()
            guard let type = GGUFValueType(rawValue: typeRaw) else { continue }
            let value = try readValue(type: type)
            metadata[key] = value
        }
        if let a = metadata["general.alignment"] as? UInt32 { alignment = Int(a) }

        var infos: [GGUFTensorInfo] = []
        infos.reserveCapacity(tensorCount)
        for _ in 0..<tensorCount {
            let name = try readString()
            let nDims = try Int(readU32())
            var shape: [Int] = []; shape.reserveCapacity(nDims)
            for _ in 0..<nDims { shape.append(try Int(readU64())) }
            let typeRaw = try readU32()
            guard let dtype = GGMLType(rawValue: typeRaw) else { continue }
            let offset = try Int(readU64())
            infos.append(GGUFTensorInfo(name: name, dtype: dtype, shape: shape,
                                          dataOffset: offset, byteSize: 0))
        }
        let rem = cursor % alignment
        if rem != 0 { cursor += alignment - rem }
        dataSectionStart = cursor
        for info in infos {
            let abs = dataSectionStart + info.dataOffset
            let bytes = ggmlTypeByteSize(info.dtype, shape: info.shape)
            tensors[info.name] = GGUFTensorInfo(
                name: info.name, dtype: info.dtype, shape: info.shape,
                dataOffset: abs, byteSize: bytes)
        }
    }

    private func ggmlTypeByteSize(_ t: GGMLType, shape: [Int]) -> Int {
        let nElems = shape.reduce(1, *)
        switch t {
        case .f32:  return nElems * 4
        case .f16, .bf16: return nElems * 2
        case .f64:  return nElems * 8
        case .i8:   return nElems
        case .i16:  return nElems * 2
        case .i32:  return nElems * 4
        case .i64:  return nElems * 8
        case .q4_0: return (nElems / 32) * 18
        case .q4_1: return (nElems / 32) * 20
        case .q5_0: return (nElems / 32) * 22
        case .q5_1: return (nElems / 32) * 24
        case .q8_0: return (nElems / 32) * 34
        case .q8_1: return (nElems / 32) * 36
        case .q2_K: return (nElems / 256) * 84
        case .q3_K: return (nElems / 256) * 110
        case .q4_K: return (nElems / 256) * 144
        case .q5_K: return (nElems / 256) * 176
        case .q6_K: return (nElems / 256) * 210
        case .q8_K: return (nElems / 256) * 292
        case .iq4_nl: return (nElems / 32) * 18
        case .iq4_xs: return (nElems / 256) * 136
        default: return 0
        }
    }

    func tensor(_ name: String) throws -> GGUFTensorInfo {
        guard let info = tensors[name] else { throw GGUFError.tensorNotFound(name) }
        return info
    }

    func makeMetalBuffer(_ name: String, device: MTLDevice) throws -> MTLBuffer {
        let info = try tensor(name)
        let ptr = base.advanced(by: info.dataOffset)
        guard let buf = device.makeBuffer(
            bytesNoCopy: ptr, length: info.byteSize,
            options: .storageModeShared, deallocator: nil
        ) else {
            throw GGUFError.badMmap("makeBuffer(bytesNoCopy) failed for \(name)")
        }
        buf.label = name
        return buf
    }

    func tensorsMatching(_ prefix: String) -> [GGUFTensorInfo] {
        return tensors.values.filter { $0.name.hasPrefix(prefix) }
            .sorted { $0.name < $1.name }
    }
}

// ===========================================================================
// Inlined safetensors v1 reader. Format: 8-byte LE UInt64 = header length,
// followed by that many bytes of JSON metadata (one entry per tensor plus
// optional __metadata__), followed by the raw tensor bytes. Used to load
// Gemma-4's BF16 vision-tower weights from the HF safetensors shards —
// kept distinct from the GGUF loader because the serialization formats
// are unrelated despite both being "weight blobs".
// ===========================================================================

struct SafetensorInfo {
    let name: String
    let dtype: String         // "BF16", "F16", "F32", ...
    let shape: [Int]
    let byteOffset: Int       // absolute offset into the mmap base
    let byteSize: Int
}

enum SafetensorsError: Error {
    case badMmap(String)
    case badHeader(String)
    case tensorNotFound(String)
    case unsupportedDtype(String)
    case failedAlloc(name: String)
}

final class SafetensorsFile {
    let path: String
    let fd: Int32
    let size: Int
    let base: UnsafeMutableRawPointer
    var tensors: [String: SafetensorInfo] = [:]
    var metadata: [String: String] = [:]
    var dataStart: Int = 0

    init(_ path: String) throws {
        self.path = path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw SafetensorsError.badMmap("open: \(String(cString: strerror(errno)))") }
        var st = stat()
        guard fstat(fd, &st) == 0 else { close(fd); throw SafetensorsError.badMmap("fstat") }
        self.fd = fd
        self.size = Int(st.st_size)
        guard let p = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
            close(fd); throw SafetensorsError.badMmap("mmap failed")
        }
        self.base = p
        try parse()
    }
    deinit { munmap(base, size); close(fd) }

    private func parse() throws {
        guard size >= 8 else { throw SafetensorsError.badHeader("file too small") }
        let hdrLen = Int(base.load(as: UInt64.self))
        guard 8 + hdrLen <= size else { throw SafetensorsError.badHeader("header length out of range") }
        dataStart = 8 + hdrLen
        let hdrBytes = UnsafeBufferPointer(
            start: base.advanced(by: 8).assumingMemoryBound(to: UInt8.self),
            count: hdrLen)
        let data = Data(buffer: hdrBytes)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw SafetensorsError.badHeader("json parse failed")
        }
        for (name, raw) in obj {
            if name == "__metadata__" {
                if let md = raw as? [String: String] { metadata = md }
                continue
            }
            guard let entry = raw as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shapeArr = entry["shape"] as? [Any],
                  let offArr = entry["data_offsets"] as? [Any],
                  offArr.count == 2 else { continue }
            let shape: [Int] = shapeArr.compactMap {
                if let i = $0 as? Int { return i }
                if let i = $0 as? NSNumber { return i.intValue }
                return nil
            }
            let off0 = (offArr[0] as? Int) ?? (offArr[0] as? NSNumber)?.intValue ?? 0
            let off1 = (offArr[1] as? Int) ?? (offArr[1] as? NSNumber)?.intValue ?? 0
            tensors[name] = SafetensorInfo(
                name: name, dtype: dtype, shape: shape,
                byteOffset: dataStart + off0, byteSize: off1 - off0)
        }
    }

    func tensor(_ name: String) throws -> SafetensorInfo {
        guard let t = tensors[name] else { throw SafetensorsError.tensorNotFound(name) }
        return t
    }

    /// Convert BF16 tensor bytes → FP16 MTLBuffer. BF16 is the top 16 bits of
    /// FP32 (1 sign + 8 exp + 7 mantissa); `fp32_bits = bf16_bits << 16`, then
    /// `Float16(fp32)` rounds to nearest representable FP16.
    func loadBF16AsFP16(_ name: String, device: MTLDevice) throws -> MTLBuffer {
        let info = try tensor(name)
        guard info.dtype == "BF16" else { throw SafetensorsError.unsupportedDtype(info.dtype) }
        let nElems = info.byteSize / 2
        let dst = device.makeBuffer(length: nElems * 2, options: .storageModeShared)!
        let src = base.advanced(by: info.byteOffset)
        let dp = dst.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<nElems {
            let bf16Bits = src.load(fromByteOffset: i * 2, as: UInt16.self)
            let f32 = Float(bitPattern: UInt32(bf16Bits) << 16)
            dp[i] = Float16(f32)
        }
        dst.label = name
        return dst
    }

    /// Convert F32 tensor bytes → FP16 MTLBuffer.
    func loadF32AsFP16(_ name: String, device: MTLDevice) throws -> MTLBuffer {
        let info = try tensor(name)
        guard info.dtype == "F32" else { throw SafetensorsError.unsupportedDtype(info.dtype) }
        let nElems = info.byteSize / 4
        let dst = device.makeBuffer(length: nElems * 2, options: .storageModeShared)!
        let src = base.advanced(by: info.byteOffset).assumingMemoryBound(to: Float.self)
        let dp = dst.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<nElems { dp[i] = Float16(src[i]) }
        dst.label = name
        return dst
    }

    /// BF16 tensor bytes staged into a shared MTLBuffer (copied from the
    /// mmap — cheaper than CPU-side bf16→fp16 conversion because there's
    /// no per-element arithmetic, just a memcpy). Intended as the "bf16
    /// staging" half of a hydrate pass: the GPU converter reads from this
    /// and writes into the working fp16 buffer, then we drop this staging
    /// buffer. Page-aligned bytesNoCopy would be zero-cost but tensor
    /// offsets inside a safetensors file aren't page-aligned, so a
    /// staging copy is the portable path.
    ///
    /// Cost: ~150 ms for the full vision tower (1.5 GB at ~10 GB/s memcpy).
    /// Net hydrate cost from unloaded: ~160 ms including GPU convert;
    /// ~10x faster than the old CPU per-element loop.
    func makeBF16StagingBuffer(_ name: String, device: MTLDevice) throws -> MTLBuffer {
        let info = try tensor(name)
        guard info.dtype == "BF16" else { throw SafetensorsError.unsupportedDtype(info.dtype) }
        let src = base.advanced(by: info.byteOffset)
        guard let buf = device.makeBuffer(length: info.byteSize, options: .storageModeShared) else {
            throw SafetensorsError.failedAlloc(name: name)
        }
        memcpy(buf.contents(), src, info.byteSize)
        buf.label = "bf16-stage:\(name)"
        return buf
    }

    /// Zero-copy wrap of tensor bytes as an MTLBuffer (for F32 passthrough etc).
    func makeMetalBuffer(_ name: String, device: MTLDevice) throws -> MTLBuffer {
        let info = try tensor(name)
        let ptr = base.advanced(by: info.byteOffset)
        guard let buf = device.makeBuffer(
            bytesNoCopy: ptr, length: info.byteSize,
            options: .storageModeShared, deallocator: nil) else {
            throw SafetensorsError.badMmap("makeBuffer(bytesNoCopy) failed for \(name)")
        }
        buf.label = name
        return buf
    }
}

