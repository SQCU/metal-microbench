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
    //
    // 2026-05-07: heap-based BPE merger replaces the old O(N²)
    // greedy scan + remove-at:. The previous algorithm did:
    //   - O(N) scan over all adjacent pairs to find lowest-rank merge
    //   - O(N) `symbols.remove(at:)` shift after each merge
    //   - O(K) String concat per merge
    // → O(N²) scans + O(N²) shifts + O(N × avg_token_len) concat
    //
    // For a 64K-token prefill (Gemma-4 supports up to 256K context),
    // that's 4 BILLION ops — a minutes-long encode call that violates
    // the model's basic spec.
    //
    // Replacement is the standard heap-based SentencePiece BPE:
    //   - doubly-linked list over symbol slots (next/prev) → O(1) merge
    //   - min-heap of pending (rank, leftIdx, rightIdx, gens) entries
    //   - generation counter per slot invalidates stale heap entries
    //     instead of needing eager-removal from the heap
    //   - per-merge: 1 heap pop + (up to) 2 heap pushes for new pairs
    //     created at the merge boundary; O(log N) work per merge
    // → O(N log N) total, with a one-time O(N) String concat budget
    // bounded by total text length.
    //
    // For 64K tokens: ~64K × 17 = ~1M ops. Sub-second.
    func encode(_ text: String, addBos: Bool? = nil) -> [UInt32] {
        var out: [UInt32] = []
        if addBos ?? self.addBos { out.append(bosId) }

        // Step 1 — SentencePiece whitespace substitution. Gemma replaces
        // every ASCII space with ▁; newlines and tabs stay literal.
        let pre = text.replacingOccurrences(of: " ", with: "▁")

        // Step 2 — initial atomization. Walk Unicode scalars; if the scalar
        // is a vocab token, use it; otherwise fall back to UTF-8 bytes via
        // <0xHH> tokens (every byte value is in the vocab as a type-6 token).
        var symbols: [String?] = []
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
        if symbols.count == 1 {
            // Trivial: no merges possible.
            if let s = symbols[0] {
                out.append(tokenToId[s] ?? unkId)
            }
            return out
        }

        // Step 3 — heap-based BPE merge.
        let n = symbols.count
        // Doubly-linked list over the symbol-slot array.
        // prevIdx[i] = index of previous live slot (or -1 at head).
        // nextIdx[i] = index of next live slot (or -1 at tail).
        // gen[i] = monotonically incrementing generation number; used
        //   to invalidate stale heap entries without eager-removal.
        var prevIdx = [Int](repeating: -1, count: n)
        var nextIdx = [Int](repeating: -1, count: n)
        var gen = [UInt32](repeating: 0, count: n)
        for i in 0..<n {
            prevIdx[i] = (i == 0) ? -1 : (i - 1)
            nextIdx[i] = (i == n - 1) ? -1 : (i + 1)
        }

        // Min-heap entries. Sorted by (rank ASC, leftIdx ASC) for
        // deterministic tiebreak. genL/genR record the generation
        // each slot had when this entry was created; if either has
        // since advanced, this entry is stale (its pair was already
        // merged on a different side).
        struct HeapEntry {
            let rank: Int
            let leftIdx: Int
            let rightIdx: Int
            let genL: UInt32
            let genR: UInt32
        }
        // Manual binary min-heap on Array<HeapEntry>. ~30 lines, no
        // external dependency. Tiebreak: lower leftIdx wins (matches
        // greedy left-to-right scan order of the previous algorithm
        // when all pairs have equal rank).
        var heap: [HeapEntry] = []
        heap.reserveCapacity(n)
        @inline(__always)
        func heapBetter(_ a: HeapEntry, _ b: HeapEntry) -> Bool {
            if a.rank != b.rank { return a.rank < b.rank }
            return a.leftIdx < b.leftIdx
        }
        @inline(__always)
        func heapPush(_ e: HeapEntry) {
            heap.append(e)
            var i = heap.count - 1
            while i > 0 {
                let p = (i - 1) >> 1
                if heapBetter(heap[i], heap[p]) {
                    heap.swapAt(i, p); i = p
                } else { break }
            }
        }
        @inline(__always)
        func heapPop() -> HeapEntry? {
            guard !heap.isEmpty else { return nil }
            let top = heap[0]
            let last = heap.removeLast()
            if !heap.isEmpty {
                heap[0] = last
                var i = 0
                let n = heap.count
                while true {
                    let l = 2*i + 1, r = 2*i + 2
                    var best = i
                    if l < n && heapBetter(heap[l], heap[best]) { best = l }
                    if r < n && heapBetter(heap[r], heap[best]) { best = r }
                    if best == i { break }
                    heap.swapAt(i, best); i = best
                }
            }
            return top
        }
        @inline(__always)
        func tryPushPair(_ leftIdx: Int, _ rightIdx: Int) {
            guard leftIdx >= 0, rightIdx >= 0,
                  let leftSym = symbols[leftIdx],
                  let rightSym = symbols[rightIdx],
                  let inner = mergeRankByFirst[leftSym],
                  let rank = inner[rightSym] else { return }
            heapPush(HeapEntry(rank: rank,
                                leftIdx: leftIdx, rightIdx: rightIdx,
                                genL: gen[leftIdx], genR: gen[rightIdx]))
        }

        // Initial heap: all adjacent live pairs.
        for i in 0..<(n - 1) {
            tryPushPair(i, i + 1)
        }

        // Process merges in rank order.
        while let entry = heapPop() {
            let l = entry.leftIdx, r = entry.rightIdx
            // Stale checks: either symbol consumed, or its generation
            // has advanced since this entry was created.
            guard symbols[l] != nil, symbols[r] != nil,
                  gen[l] == entry.genL, gen[r] == entry.genR,
                  nextIdx[l] == r else { continue }
            // Merge r into l.
            symbols[l] = symbols[l]! + symbols[r]!
            symbols[r] = nil
            gen[l] &+= 1
            // Splice r out of the linked list.
            let rNext = nextIdx[r]
            nextIdx[l] = rNext
            if rNext >= 0 { prevIdx[rNext] = l }
            // (prevIdx[r] / nextIdx[r] are now ignored.)
            // Push new pairs at the merge boundary.
            let lPrev = prevIdx[l]
            tryPushPair(lPrev, l)
            tryPushPair(l, rNext)
        }

        // Step 4 — walk the linked list from head (the lowest live
        // index) and emit token IDs. A symbol that somehow didn't
        // resolve falls back to UNK; after a successful merge pass
        // every symbol should be in vocab (singletons + byte tokens
        // are all in vocab).
        var head = 0
        while head < n && symbols[head] == nil { head += 1 }
        var i = head
        while i >= 0 {
            if let s = symbols[i] {
                out.append(tokenToId[s] ?? unkId)
            }
            i = nextIdx[i]
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
            // vocab string here. Channel-block / EOS suppression for
            // the bridge's output stream lives at the engine layer
            // (lm_engine.swift:Session.maybeAppendOutput); decode
            // remains a faithful tokens→bytes mapper.
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
