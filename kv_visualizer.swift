// Prefix-discipline visualizer.
//
// Purpose: give users a mental model of KV-cache sharing by showing exactly
// which pages would be cached-and-reused vs freshly-prefilled for a set of
// submitted sessions. Runs entirely on CPU — no GPU, no weights loaded —
// so it's usable as a "what if I prompt it this way" planning tool before
// spending any GPU time.
//
// The common anti-pattern this visualizer diagnoses: users think they're
// being clever by editing a prefix mid-way through a chat (e.g. revising
// the system prompt, or swapping out a tool's output) and assume the
// engine will "patch" the cache. It cannot. KV at position N is computed
// from every earlier position; changing any token in [0, N) invalidates
// KV for every position in [N, end]. The visualizer makes this concrete
// by rendering cache vs reprefill decisions per-page, in color, with the
// "break point" highlighted.
//
// Invoke: KV_VIZ=1 ./forward_graph with LM_VIZ_SESSION_1=<prompt1> etc.
import Foundation

// A single virtual session for the visualizer — just a prompt and an
// optional label; no real KV, no tokens generated. The visualizer
// mirrors what the engine would do if it were fed this prompt.
struct VizSession {
    let label: String
    let tokens: [UInt32]
}

// One cell of the render: a page index, its prefix hash, and the slot
// assignment history (which session first wrote it, who reused it).
private struct PageCell {
    let pageIndex: Int
    let prefixHash: UInt64
    var owners: [Int]   // session indices that share this page
    var firstWriter: Int   // the session that originally filled it
}

// Visualizer for K page-aligned sessions sharing a global content cache.
struct PrefixDisciplineViz {
    let sessions: [VizSession]
    let pageSize: Int   // must match PAGE for the engine

    // Render everything: a header, a per-session "submission timeline"
    // row showing cache vs reprefill per page, then a summary table.
    func render(toStdout: Bool = true) -> String {
        var out = ""
        out += header()

        // Build the cache as sessions submit in-order (session 0 first,
        // then 1, ...). For each session, walk its pages; the first time
        // we see a given (hash, length-of-prefix) pair, it's a reprefill;
        // subsequent matches are cache-hits.
        //
        // Hash granularity is per-PAGE, keyed by the FULL prefix up to and
        // including that page. Two sessions sharing the first N pages of
        // tokens will produce the same hashes for those pages and share
        // them; the first page where they diverge starts a new cache entry.
        var cache: [UInt64: Int] = [:]  // prefix_hash -> page index
        var nextPageIndex: Int = 0

        var sessionPages: [[PageCell]] = []

        for (sIdx, sess) in sessions.enumerated() {
            let nPages = (sess.tokens.count + pageSize - 1) / pageSize
            var pages: [PageCell] = []
            for p in 0..<nPages {
                let end = min((p + 1) * pageSize, sess.tokens.count)
                let prefixHash = fnv1a(ArraySlice(sess.tokens[0..<end]))
                if let existing = cache[prefixHash] {
                    // Cache hit: this session shares the page.
                    let pageIdx = existing
                    // Find and update the PageCell in some earlier session's record.
                    var firstWriter = sIdx
                    for priorSession in 0..<sIdx {
                        if sessionPages[priorSession].indices.contains(p),
                           sessionPages[priorSession][p].pageIndex == pageIdx {
                            sessionPages[priorSession][p].owners.append(sIdx)
                            firstWriter = sessionPages[priorSession][p].firstWriter
                            break
                        }
                    }
                    pages.append(PageCell(pageIndex: pageIdx,
                                           prefixHash: prefixHash,
                                           owners: [firstWriter, sIdx],
                                           firstWriter: firstWriter))
                } else {
                    // Fresh — this session is the first writer.
                    let pageIdx = nextPageIndex; nextPageIndex += 1
                    cache[prefixHash] = pageIdx
                    pages.append(PageCell(pageIndex: pageIdx,
                                           prefixHash: prefixHash,
                                           owners: [sIdx],
                                           firstWriter: sIdx))
                }
            }
            sessionPages.append(pages)
        }

        // Render per-session timelines.
        out += "\n"
        out += "  Per-session submission timeline:\n"
        out += "  (each cell is one PAGE = \(pageSize) tokens; \(esc("●", color: "green")) = reprefill required, \(esc("◌", color: "gray")) = cache hit, \(esc("◆", color: "cyan")) = first writer of a shared page)\n\n"

        let maxPages = sessionPages.map { $0.count }.max() ?? 0
        // Column headers (page index 0..maxPages-1).
        out += "  " + String(repeating: " ", count: 20)
        for p in 0..<maxPages {
            out += String(format: "%3d ", p)
        }
        out += "\n"

        for (sIdx, sess) in sessions.enumerated() {
            let label = sess.label.padding(toLength: 16, withPad: " ", startingAt: 0)
            out += "  s\(sIdx) \(label): "
            for (p, cell) in sessionPages[sIdx].enumerated() {
                _ = p
                if cell.firstWriter != sIdx {
                    // This session is sharing someone else's page.
                    out += " " + esc("◌", color: "gray") + "  "
                } else if cell.owners.count > 1 {
                    // This session wrote it first AND others reuse it.
                    out += " " + esc("◆", color: "cyan") + "  "
                } else {
                    // Fresh, nobody else shares it.
                    out += " " + esc("●", color: "green") + "  "
                }
            }
            out += "\n"
        }

        out += "\n"

        // Summary per session: cache-hit page count vs reprefill count.
        out += "  Summary:\n"
        var totalHits = 0, totalWrites = 0
        for (sIdx, pages) in sessionPages.enumerated() {
            let hits = pages.filter { $0.firstWriter != sIdx }.count
            let writes = pages.count - hits
            totalHits += hits; totalWrites += writes
            let tokens = sessions[sIdx].tokens.count
            let pct = pages.count > 0 ? 100 * hits / pages.count : 0
            out += String(format: "  s%d: %d tokens → %d pages (%d cached, %d reprefilled — %d%% cache hit)\n",
                          sIdx, tokens, pages.count, hits, writes, pct)
        }
        let totalPages = totalHits + totalWrites
        let sharingPct = totalPages > 0 ? 100 * totalHits / totalPages : 0
        out += String(format: "  TOTAL: %d page-writes skipped / %d total page demands (%d%% sharing)\n",
                      totalHits, totalPages, sharingPct)

        // Break-point diagnostic: if session sIdx shares session (sIdx-1)'s
        // first N-1 pages but diverges at page K, say so. If there's NO
        // sharing at all between a pair that "should" share (users often
        // insert a stray whitespace or reorder tokens), call it out.
        out += "\n"
        out += "  Cache-boundary diagnostics:\n"
        for sIdx in 1..<sessions.count {
            var sharedPages = 0
            for p in 0..<min(sessionPages[sIdx].count, sessionPages[sIdx - 1].count) {
                if sessionPages[sIdx][p].prefixHash == sessionPages[sIdx - 1][p].prefixHash {
                    sharedPages += 1
                } else {
                    break
                }
            }
            let breakPage = sharedPages
            let breakToken = breakPage * pageSize
            let myLen = sessions[sIdx].tokens.count
            let prevLen = sessions[sIdx - 1].tokens.count
            let commonLen = min(myLen, prevLen)
            var divergeToken = commonLen
            for i in 0..<commonLen where sessions[sIdx].tokens[i] != sessions[sIdx - 1].tokens[i] {
                divergeToken = i; break
            }
            if sharedPages == 0 && divergeToken > 0 {
                out += String(format: "  ⚠  s%d vs s%d: %d tokens share first \(divergeToken) token-ids but DIFFERENT prefix-hashes on page 0. "
                              + "A trailing-whitespace-edit or reordering is invalidating cache eligibility.\n",
                              sIdx, sIdx - 1, divergeToken)
            } else if breakToken > 0 && divergeToken < myLen {
                out += String(format: "  s%d vs s%d: shared first %d pages (%d tokens). Divergence at token %d (page %d).\n",
                              sIdx, sIdx - 1, breakPage, breakToken, divergeToken, breakPage)
                // If the divergence is MID-PAGE (not on page boundary), flag it.
                if divergeToken % pageSize != 0 && divergeToken >= breakToken {
                    let wasted = pageSize - (divergeToken % pageSize)
                    out += String(format: "    Note: divergence is mid-page — ~%d tokens of page-%d work are 'lost to' a single-token difference.\n",
                                  wasted, breakPage)
                }
            } else if breakToken > 0 && divergeToken == myLen {
                out += String(format: "  s%d vs s%d: s%d is a strict prefix of s%d (all %d pages shared).\n",
                              sIdx - 1, sIdx, sIdx - 1, sIdx, sharedPages)
            } else if sharedPages == 0 {
                out += String(format: "  s%d vs s%d: no shared prefix.\n", sIdx, sIdx - 1)
            }
        }
        return out
    }

    // -------- Helpers --------
    private func header() -> String {
        var s = ""
        s += "\n"
        s += "  " + String(repeating: "─", count: 72) + "\n"
        s += "  Prefix-discipline visualizer\n"
        s += "  " + String(repeating: "─", count: 72) + "\n"
        return s
    }

    // FNV-1a matching PageManager.hashPage so the visualizer's decisions
    // match what the runtime actually does.
    private func fnv1a(_ tokens: ArraySlice<UInt32>) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for t in tokens {
            h ^= UInt64(t)
            h = h &* 0x100000001b3
        }
        return h
    }

    // ANSI colour escape. Falls through to no-color if NO_COLOR env var is
    // set or if we're not on a TTY.
    private func esc(_ s: String, color: String) -> String {
        if ProcessInfo.processInfo.environment["NO_COLOR"] != nil { return s }
        if isatty(1) == 0 { return s }
        let code: String
        switch color {
        case "green":  code = "32"
        case "red":    code = "31"
        case "cyan":   code = "36"
        case "yellow": code = "33"
        case "gray":   code = "90"
        default:       code = "0"
        }
        return "\u{001B}[\(code)m\(s)\u{001B}[0m"
    }
}

// ----------------------------------------------------------------------
// Env-var demo driver. Constructs a small didactic scenario showing how
// shared prefixes produce cache sharing, and how a subtle mid-prefix edit
// destroys it. No GPU, no weights. Intended for terminal teaching.
//   KV_VIZ=1 ./forward_graph
// Optionally the GGUF can be loaded to tokenize LM_VIZ_SESSION_N strings
// rather than fabricating token IDs:
//   KV_VIZ=1 GGUF_PATH=<gguf> LM_VIZ_SESSION_1="..." LM_VIZ_SESSION_2="..."
// ----------------------------------------------------------------------
func runKvVisualizer() {
    print("\n=== KV-cache sharing visualizer (no GPU) ===")

    // If a GGUF is supplied AND user passed LM_VIZ_SESSION_N env vars, use
    // the real tokenizer. Otherwise fall back to a fabricated scenario
    // built from fake token IDs — still pedagogically useful.
    var sessions: [VizSession] = []
    let env = ProcessInfo.processInfo.environment
    if let ggufPath = env["GGUF_PATH"],
       env.keys.contains(where: { $0.hasPrefix("LM_VIZ_SESSION_") }) {
        let w: LmWeights
        do { w = try loadLmWeights(ggufPath: ggufPath) }
        catch { print("  loadLmWeights failed: \(error)"); return }
        let tok = GemmaBpe(weights: w)
        for i in 1...16 {
            if let text = env["LM_VIZ_SESSION_\(i)"], !text.isEmpty {
                let toks = tok.encode(text, addBos: true)
                sessions.append(VizSession(label: "s\(i)", tokens: toks))
            }
        }
    }
    if sessions.isEmpty {
        // Built-in didactic scenario. 4 sessions demonstrate:
        // (1) full shared prefix, small divergent tail (ideal)
        // (2) same start, ONE WORD inserted mid-prefix (worst-case cache miss)
        // (3) strict-prefix of (1) — full sharing, no tail
        // (4) totally different prompt — no sharing, standalone entry
        let systemPrompt: [UInt32] = (0..<48).map { UInt32(100 + $0) }
        let image: [UInt32] = (200..<480).map { UInt32($0) }   // 280-token "image"
        let suffix1: [UInt32] = [9000, 9001, 9002]
        let suffix2: [UInt32] = [9003, 9004, 9005, 9006]
        let extraWord: UInt32 = 7777
        sessions.append(VizSession(
            label: "sys+img+q1",
            tokens: systemPrompt + image + suffix1))
        sessions.append(VizSession(
            label: "sys+WORD+img+q2",
            tokens: systemPrompt + [extraWord] + image + suffix2))
        sessions.append(VizSession(
            label: "sys+img only",
            tokens: systemPrompt + image))
        sessions.append(VizSession(
            label: "different prompt",
            tokens: (0..<32).map { UInt32(5000 + $0) }))
    }

    let viz = PrefixDisciplineViz(sessions: sessions, pageSize: PAGE)
    print(viz.render())
}
