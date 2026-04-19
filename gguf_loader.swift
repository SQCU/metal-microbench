import Foundation
import Metal

// ===========================================================================
// Minimal GGUF v3 reader for Gemma-4-A4B-it-UD-Q4_K_M.gguf.
// Parses header + metadata + tensor table, mmaps data section, exposes
// tensors by name with mmap pointers that can be wrapped as MTLBuffer with
// bytesNoCopy (zero-copy weight loading).
//
// Scope: just enough to load Gemma-4. Not a full GGUF library.
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
    let shape: [Int]               // ggml order: slowest-changing first
    let dataOffset: Int            // absolute file offset into mmap
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

    // ---------- Raw read helpers ----------
    private func ensure(_ n: Int) throws {
        if cursor + n > size { throw GGUFError.readPastEnd }
    }

    private func readU32() throws -> UInt32 {
        try ensure(4)
        let v = base.advanced(by: cursor).load(as: UInt32.self)
        cursor += 4
        return v
    }
    private func readU64() throws -> UInt64 {
        try ensure(8)
        let v = base.advanced(by: cursor).load(as: UInt64.self)
        cursor += 8
        return v
    }
    private func readI32() throws -> Int32 {
        try ensure(4)
        let v = base.advanced(by: cursor).load(as: Int32.self)
        cursor += 4
        return v
    }
    private func readI64() throws -> Int64 {
        try ensure(8)
        let v = base.advanced(by: cursor).load(as: Int64.self)
        cursor += 8
        return v
    }
    private func readF32() throws -> Float {
        try ensure(4)
        let v = base.advanced(by: cursor).load(as: Float.self)
        cursor += 4
        return v
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
            guard let inner = GGUFValueType(rawValue: innerRaw) else {
                throw GGUFError.readPastEnd  // treat as malformed
            }
            let count = try Int(readU64())
            var arr: [Any] = []
            arr.reserveCapacity(count)
            for _ in 0..<count { arr.append(try readValue(type: inner)) }
            return arr
        }
    }

    // ---------- Header + metadata + tensor table parse ----------
    private func parse() throws {
        // Magic + version
        let magic = try readU32()
        guard magic == 0x46554747 else { throw GGUFError.badMagic }      // 'GGUF' little-endian
        let version = try readU32()
        guard version == 3 else { throw GGUFError.unsupportedVersion(version) }

        let tensorCount = try Int(readU64())
        let metaCount   = try Int(readU64())

        // Metadata
        for _ in 0..<metaCount {
            let key = try readString()
            let typeRaw = try readU32()
            guard let type = GGUFValueType(rawValue: typeRaw) else { continue }
            let value = try readValue(type: type)
            metadata[key] = value
        }
        if let a = metadata["general.alignment"] as? UInt32 { alignment = Int(a) }

        // Tensor info entries
        var infos: [GGUFTensorInfo] = []
        infos.reserveCapacity(tensorCount)
        for _ in 0..<tensorCount {
            let name = try readString()
            let nDims = try Int(readU32())
            var shape: [Int] = []
            shape.reserveCapacity(nDims)
            for _ in 0..<nDims { shape.append(try Int(readU64())) }
            let typeRaw = try readU32()
            guard let dtype = GGMLType(rawValue: typeRaw) else {
                continue   // skip unknown types (or throw — for now skip)
            }
            let offset = try Int(readU64())
            infos.append(GGUFTensorInfo(
                name: name, dtype: dtype, shape: shape,
                dataOffset: offset, byteSize: 0))   // size computed after data-section start known
        }

        // Align cursor to `alignment` boundary — data section starts there
        let rem = cursor % alignment
        if rem != 0 { cursor += alignment - rem }
        dataSectionStart = cursor

        // Now compute absolute offsets and byte sizes
        // Byte size can be inferred from dtype + shape; see ggml_type_size etc.
        for info in infos {
            let abs = dataSectionStart + info.dataOffset
            let bytes = ggmlTypeByteSize(info.dtype, shape: info.shape)
            tensors[info.name] = GGUFTensorInfo(
                name: info.name, dtype: info.dtype, shape: info.shape,
                dataOffset: abs, byteSize: bytes)
        }
    }

    // Compute byte size for a tensor given dtype + shape (ggml layout: columns-first).
    // For quantized types, the inner dim (shape[0]) must be a multiple of the block size.
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
        // Block-quantized types
        case .q4_0: return (nElems / 32) * 18        // 2 + 16
        case .q4_1: return (nElems / 32) * 20        // 2 + 2 + 16
        case .q5_0: return (nElems / 32) * 22
        case .q5_1: return (nElems / 32) * 24
        case .q8_0: return (nElems / 32) * 34        // 2 + 32
        case .q8_1: return (nElems / 32) * 36
        case .q2_K: return (nElems / 256) * 84
        case .q3_K: return (nElems / 256) * 110
        case .q4_K: return (nElems / 256) * 144      // 2 + 2 + 12 + 128
        case .q5_K: return (nElems / 256) * 176
        case .q6_K: return (nElems / 256) * 210
        case .q8_K: return (nElems / 256) * 292
        case .iq4_nl: return (nElems / 32) * 18
        case .iq4_xs: return (nElems / 256) * 136
        default: return 0    // unsupported / unsized
        }
    }

    // ---------- Public accessors ----------
    func tensor(_ name: String) throws -> GGUFTensorInfo {
        guard let info = tensors[name] else { throw GGUFError.tensorNotFound(name) }
        return info
    }

    /// Zero-copy wrap of a tensor's bytes as an MTLBuffer. The buffer shares
    /// the mmap'd memory — do not free or modify while the buffer is in use.
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

    // ---------- Debug helpers ----------
    func printSummary() {
        print("=== GGUF: \(path) ===")
        print("  size: \(size / (1024*1024)) MB")
        print("  metadata entries: \(metadata.count)")
        print("  tensors: \(tensors.count)")
        print("  data section starts at byte: \(dataSectionStart)")
        print("  alignment: \(alignment)")
        // Key metadata we care about
        let keys = [
            "general.architecture", "general.name", "general.file_type",
            "gemma4.context_length", "gemma4.embedding_length",
            "gemma4.block_count", "gemma4.attention.head_count",
            "gemma4.attention.head_count_kv", "gemma4.expert_count",
            "gemma4.expert_used_count", "gemma4.expert_feed_forward_length",
            "gemma4.feed_forward_length", "gemma4.rope.freq_base",
        ]
        print("  selected metadata:")
        for k in keys {
            if let v = metadata[k] {
                print("    \(k) = \(v)")
            }
        }
        // Count tensor dtypes
        var dtypeCount: [GGMLType: Int] = [:]
        for (_, info) in tensors {
            dtypeCount[info.dtype, default: 0] += 1
        }
        print("  tensor dtype histogram:")
        for (dt, c) in dtypeCount.sorted(by: { $0.value > $1.value }) {
            print("    \(dt): \(c)")
        }
    }

    /// List tensor names matching a prefix, sorted.
    func tensorsMatching(_ prefix: String) -> [GGUFTensorInfo] {
        return tensors.values.filter { $0.name.hasPrefix(prefix) }
            .sorted { $0.name < $1.name }
    }
}

// CLI entry point moved to gguf_tool.swift so this file can be linked as a
// library into forward_graph without side-effects at module init time.
