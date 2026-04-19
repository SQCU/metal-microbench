import Foundation

// Standalone CLI wrapper for GGUFFile. Prints summary + a few tensor examples.
// Kept separate from gguf_loader.swift so the loader can be linked as a
// library into forward_graph without running top-level code at init.

if CommandLine.arguments.count >= 2 {
    let path = CommandLine.arguments[1]
    do {
        let g = try GGUFFile(path)
        g.printSummary()
        print("\n  first layer's tensors:")
        for info in g.tensorsMatching("blk.0.").prefix(30) {
            let shapeStr = info.shape.map(String.init).joined(separator: "×")
            print("    \(info.name.padding(toLength: 40, withPad: " ", startingAt: 0)) \(info.dtype) \(shapeStr) (\(info.byteSize / 1024) KB)")
        }
        print("\n  top-level tensors:")
        for name in ["token_embd.weight", "output.weight", "output_norm.weight"] {
            if let info = g.tensors[name] {
                let shapeStr = info.shape.map(String.init).joined(separator: "×")
                print("    \(name.padding(toLength: 40, withPad: " ", startingAt: 0)) \(info.dtype) \(shapeStr) (\(info.byteSize / (1024*1024)) MB)")
            }
        }
    } catch {
        fputs("error: \(error)\n", stderr)
        exit(1)
    }
} else {
    fputs("usage: gguf_loader <path-to-gguf>\n", stderr)
    exit(2)
}
