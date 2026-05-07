// Gemma-4 BPE tokenizer + detokenizer.
//
// Why a hand-rolled BPE: we load weights directly from a Q4_K_M GGUF, so the
// tokenizer metadata (tokens[262144], merges[514906], add_bos, eos) is all
// sitting in `LmWeights` already. Shelling out to Python would double the
// process count for every generation and break the "one binary, one command
// buffer" story.
//
// Algorithm (SentencePiece-BPE, same as HF's GemmaTokenizerFast):
//   1. Preprocess: replace ASCII space with ▁ (U+2581).
//   2. Split into initial symbols. Each Unicode scalar is a symbol if that
//      exact string is in the vocab; otherwise we fall back to the byte
//      tokens <0x00>..<0xFF> one byte at a time.
//   3. Greedy-merge: scan adjacent pairs, apply the one with lowest merge
//      rank, repeat until no merge applies. O(n²) per step with O(n) steps
//      → O(n³) worst case; fine for prompts up to a few thousand tokens.
//      Optimisation to a priority-queue-based O(n log n) is straightforward
//      if this becomes a bottleneck.
//   4. Map final symbols to IDs; unknown symbols → UNK (0 or model-specific).
//
// Decoder: iterate tokens; for normal tokens replace ▁ with space; for
// byte tokens <0xHH> accumulate bytes into a scratch buffer and UTF-8-decode
// runs of bytes so multi-byte emoji etc. reconstruct correctly.
import Foundation

struct GemmaBpe {
    let vocab: [String]
    let tokenToId: [String: UInt32]
    // mergeRankByFirst[firstSymbol] = [secondSymbol: rank]. Priority = rank
    // (lower is better — merges[0] is the most aggressive merge).
    let mergeRankByFirst: [String: [String: Int]]
    let bosId: UInt32
    let eosId: UInt32
    let unkId: UInt32
    let addBos: Bool
    // 2026-05-07: pre-computed UTF-8 byte payload per token. Built once
    // at init; tokenBytes(_:) just reads from this table. Avoids the
    // 3-allocs-per-call (replacingOccurrences + Array(s.utf8) + return)
    // hot loop pattern in cotMask, which iterates all VOCAB=262,144
    // tokens per CoT-active sampling tick.
    let tokenBytesTable: [[UInt8]]

    init(weights: LmWeights, unkId: UInt32 = 3) {
        self.vocab = weights.vocabTokens
        self.bosId = weights.bosTokenId
        self.eosId = weights.eosTokenId
        self.unkId = unkId
        self.addBos = weights.addBosToken

        var t2i: [String: UInt32] = [:]
        t2i.reserveCapacity(vocab.count)
        for (i, s) in vocab.enumerated() where !s.isEmpty {
            t2i[s] = UInt32(i)
        }
        self.tokenToId = t2i

        // Parse merges into nested dict. Each merge is "A B" with a single
        // ASCII space delimiter; tokens themselves contain no ASCII space
        // (spaces map to ▁). Split on the FIRST space.
        var nested: [String: [String: Int]] = [:]
        for (rank, m) in weights.merges.enumerated() {
            guard let spaceIdx = m.firstIndex(of: " ") else { continue }
            let a = String(m[..<spaceIdx])
            let b = String(m[m.index(after: spaceIdx)...])
            if a.isEmpty || b.isEmpty { continue }
            nested[a, default: [:]][b] = rank
        }
        self.mergeRankByFirst = nested

        // Build the tokenBytes lookup table. One pass over vocab.
        var table: [[UInt8]] = []
        table.reserveCapacity(vocab.count)
        let underscoreBar = "▁"
        for s in vocab {
            // Empty entries: no byte payload.
            if s.isEmpty { table.append([]); continue }
            // Byte token "<0xHH>" → single byte.
            if s.count == 6, s.hasPrefix("<0x"), s.hasSuffix(">"),
               let b = UInt8(s.dropFirst(3).dropLast(), radix: 16) {
                table.append([b])
                continue
            }
            // Special tokens: framing only, no byte payload.
            if s.hasPrefix("<") && s.hasSuffix(">") && s.count > 2 {
                table.append([])
                continue
            }
            // Normal token: ▁ → space, then UTF-8 bytes.
            let mapped = s.replacingOccurrences(of: underscoreBar, with: " ")
            table.append(Array(mapped.utf8))
        }
        self.tokenBytesTable = table
    }

    // Encode a prompt into token IDs. When `addBos` is nil, the default from
    // the GGUF metadata applies (Gemma-4 defaults to true).
    func encode(_ text: String, addBos: Bool? = nil) -> [UInt32] {
        var out: [UInt32] = []
        if addBos ?? self.addBos { out.append(bosId) }

        // Step 1 — SentencePiece whitespace substitution. Gemma replaces
        // every ASCII space with ▁; newlines and tabs stay literal.
        let pre = text.replacingOccurrences(of: " ", with: "▁")

        // Step 2 — initial atomization. Walk Unicode scalars; if the scalar
        // is a vocab token, use it; otherwise fall back to UTF-8 bytes via
        // <0xHH> tokens (every byte value is in the vocab as a type-6 token).
        var symbols: [String] = []
        symbols.reserveCapacity(pre.count * 2)
        for scalar in pre.unicodeScalars {
            let s = String(scalar)
            if tokenToId[s] != nil {
                symbols.append(s)
            } else {
                for b in s.utf8 {
                    symbols.append(String(format: "<0x%02X>", b))
                }
            }
        }
        if symbols.isEmpty { return out }

        // Step 3 — greedy merge. Scan for the best (lowest-rank) adjacent
        // pair, merge, repeat. Early-exit when no pair has a rank.
        while symbols.count >= 2 {
            var bestRank = Int.max
            var bestI = -1
            for i in 0..<(symbols.count - 1) {
                guard let inner = mergeRankByFirst[symbols[i]],
                      let r = inner[symbols[i + 1]] else { continue }
                if r < bestRank { bestRank = r; bestI = i }
            }
            if bestI < 0 { break }
            symbols[bestI] = symbols[bestI] + symbols[bestI + 1]
            symbols.remove(at: bestI + 1)
        }

        // Step 4 — map to IDs. Any symbol that somehow didn't resolve falls
        // back to UNK, but after a successful merge pass every symbol should
        // be in vocab (single chars + byte tokens are all type-1/type-6).
        for s in symbols {
            if let id = tokenToId[s] { out.append(id) }
            else { out.append(unkId) }
        }
        return out
    }

    // Decode token IDs back to a string. Byte-fallback runs get UTF-8-decoded
    // as a group so multi-byte characters don't fragment across tokens.
    // Special tokens (type-3/type-4 in GGUF) are rendered with their raw
    // vocab string so callers can see e.g. `<bos>` in debug output.
    //
    // 2026-05-07: rewrote to accumulate ALL output as UTF-8 bytes, then
    // build the final String once at the end. The previous version did
    // `out += t.replacingOccurrences(of: "▁", with: " ")` per normal
    // token (= 1 String alloc + 1 String concat per token) and
    // `out += s` per byte-run flush. For a 200-token completion that's
    // ~400 String allocations per decode — noticeable per-request.
    // The new version uses tokenBytesTable directly (built at init)
    // and only constructs one String at the end.
    func decode(_ tokens: [UInt32], skipSpecial: Bool = false) -> String {
        var byteBuf: [UInt8] = []
        byteBuf.reserveCapacity(tokens.count * 4)   // typical ~3-4 UTF-8 bytes/token

        for id in tokens {
            let i = Int(id)
            guard i < vocab.count else {
                // OOV: append literal "<oov:NNN>" bytes.
                let s = "<oov:\(id)>"
                byteBuf.append(contentsOf: s.utf8)
                continue
            }
            let t = vocab[i]
            // Specials: <bos>, <eos>, <end_of_turn>, etc. — pass through
            // unless caller wants them stripped. tokenBytesTable[i] is
            // empty for specials (by design), so we go through the raw
            // vocab string here.
            if t.hasPrefix("<") && t.hasSuffix(">") && t.count > 2
               && !(t.count == 6 && t.hasPrefix("<0x")) {
                if !skipSpecial { byteBuf.append(contentsOf: t.utf8) }
                continue
            }
            // Byte token or normal token — both pre-resolved in tokenBytesTable.
            byteBuf.append(contentsOf: tokenBytesTable[i])
        }
        // One String alloc at end, vs N during the loop.
        if let s = String(bytes: byteBuf, encoding: .utf8) {
            return s
        }
        // Fallback for invalid UTF-8: render each invalid byte as \xHH.
        var out = ""
        out.reserveCapacity(byteBuf.count)
        for b in byteBuf {
            if let scalar = Unicode.Scalar(UInt32(b)), b < 0x80 {
                out.unicodeScalars.append(scalar)
            } else {
                out += String(format: "\\x%02X", b)
            }
        }
        return out
    }

    // Lookup helper for debug/logging: "tokenize-then-stringify" single token.
    func tokenString(_ id: UInt32) -> String {
        guard Int(id) < vocab.count else { return "<oov:\(id)>" }
        return vocab[Int(id)]
    }

    // Bytes that this token ID contributes to the output text stream when
    // emitted as an AR step. Used by grammar-constrained sampling
    // (structured-cot etc.) to compute per-step token masks.
    //
    //   - byte tokens "<0xHH>" → [byte]
    //   - special tokens (e.g. "<bos>", "<end_of_turn>") → empty (don't
    //     contribute to output bytes; control-only)
    //   - normal tokens → UTF-8 bytes of (token with ▁ → ASCII space)
    func tokenBytes(_ id: UInt32) -> [UInt8] {
        let i = Int(id)
        guard i < tokenBytesTable.count else { return [] }
        return tokenBytesTable[i]
    }
}
