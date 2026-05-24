// radix_trie.swift — Token-granularity prefix cache (Track D of the
// 2026-05-23 prefix-cache grand-slam refactor).
//
// Replaces `PageManager.contentIndex: [UInt64: SlidePairContents]` with a
// radix trie of token IDs. Edge-compressed; anchors live at PAGE_SLIDE
// multiples (and at end-of-prefix for Track-B partial pairs). Each
// anchor carries a `[CvecAnchorTag: SlidePairContents]` map plus a
// `[CvecAnchorTag: PartialPagePair]` map for partial-pair sharing.
//
// See docs/prefix_cache_grand_slam_2026_05_23/track_d_radix_trie.md
// for the full design rationale.

import Foundation

// CvecAnchorTag — the per-anchor cvec partition key. Replaces the
// scalar `cvecDigest: UInt64` with a structured value so we can
// (a) inspect WHICH control caused a partition during debugging, and
// (b) phase-/units-/magnitude-gate the contributing fields per
// Track C's audit.
//
// `unsteered` is the sentinel zero value — equivalent to "no
// intersecting controls"; unsteered sessions hash-collide on this
// value and share pages with each other.
struct CvecAnchorTag: Hashable {
    // Each entry describes one control that meaningfully perturbs
    // this page's K/V. Empty array → unsteered tag.
    struct Entry: Hashable {
        let layer: Int
        let cvecId: String
        let mode: String         // CvecMode.rawValue
        let units: String        // CvecUnits.rawValue
        // Phase-gated envelope params, all quantized to Q16.16
        // fixed-point so 1-ulp float noise doesn't false-partition.
        let attackQ: Int64?      // nil if attack phase doesn't overlap page
        let decayQ: Int64?
        let sustainLevelQ: Int64?
        let releaseQ: Int64?
        let shapeIfRamp: String? // shape.rawValue when any ramp overlaps
        let peakTimesPolarityQ: Int64
        // Units-gated start/stop anchors, rebased to pageStart.
        // tokens-units → startOffset (Int); turns-units → startTurn (Int).
        let startAnchor: Int?
        let stopAnchor: Int?
        // project-mode target, transport scale/offset (mode-gated).
        let targetQ: Int64?
        let transportScaleQ: Int64?
        let transportOffsetQ: Int64?
    }
    let entries: [Entry]

    static let unsteered = CvecAnchorTag(entries: [])

    var isUnsteered: Bool { entries.isEmpty }
}

// PartialPagePair — a (slidePlusFullHead, optional fullTail, validUpTo)
// triple used by Track B (partial-page promotion at teardown). The
// underlying physical pages are valid for positions [pageStart,
// pageStart + validUpTo).
//
// `validUpTo` is in [1, PAGE_SLIDE-1]: 0 means "no progress" (no
// partial pair worth promoting), PAGE_SLIDE means "full pair" (use
// SlidePairContents instead).
//
// `fullTail` is nil when validUpTo ≤ PAGE_FULL (8): the second
// full-attn page (block_table[2P+1]) hasn't been touched at all,
// so we don't reference any sibling phys page.
struct PartialPagePair {
    let slidePlusFullHead: Int    // phys @ block_table[2P]
    let fullTail: Int?             // phys @ block_table[2P+1], nil if validUpTo ≤ 8
    let validUpTo: Int             // 1..PAGE_SLIDE-1
}

// PrefixMatch — the lookup result. `pages` are pre-incref'd; caller
// MUST decref each on session teardown. `partialTail` is set when
// the deepest matched anchor carries a partial pair; the caller is
// expected to CoW the partial bytes onto a fresh page.
struct PrefixMatch {
    let alignedMatchLength: Int    // multiple of PAGE_SLIDE; pages.count*2*PAGE_SLIDE/2
    let trieMatchLength: Int       // longest token-walk; reported for telemetry
    let pages: [SlidePairContents]    // page-pairs covering [0, alignedMatchLength)
    let partialTail: PartialPagePair?
}

// ============================================================================
// Trie nodes.
//
// Edge-compression: a TrieNode owns an `edgeTokens` slice representing
// the sequence of tokens consumed on the *incoming* edge (i.e., the
// suffix beyond the parent's depth). The root has empty edgeTokens.
// Children are keyed on the FIRST token of THEIR incoming edge.
// ============================================================================
class TrieNode {
    // Incoming edge tokens (excluding the parent's last token).
    // For the root this is empty.
    var edgeTokens: [UInt32]
    // depth = parent.depth + edgeTokens.count.
    var depth: Int
    weak var parent: TrieNode?
    // Keyed on first token of the child's edge.
    var children: [UInt32: TrieNode] = [:]

    init(edgeTokens: [UInt32], depth: Int, parent: TrieNode?) {
        self.edgeTokens = edgeTokens
        self.depth = depth
        self.parent = parent
    }
}

// TrieAnchor — a TrieNode whose `depth` is a PAGE_SLIDE multiple (or
// is an end-of-prefix anchor for a partial pair). Carries the cvec
// partition maps plus the underlying physical page indices used for
// the eviction callback.
final class TrieAnchor: TrieNode {
    // Full pairs (validUpTo == PAGE_SLIDE) keyed by CvecAnchorTag.
    var byCvecAnchorTag: [CvecAnchorTag: SlidePairContents] = [:]
    // Partial pairs (validUpTo < PAGE_SLIDE) keyed by CvecAnchorTag.
    var byCvecAnchorTagPartial: [CvecAnchorTag: PartialPagePair] = [:]
}

// PageManager-agnostic radix trie. Phys pages are owned by PageManager;
// the trie holds back-pointers via SlidePairContents / PartialPagePair
// values stored at anchors.
final class RadixTrie {
    private let pageSlide: Int
    private let root: TrieAnchor

    // Reverse map: phys page → anchor that holds it (and key into the
    // anchor's maps). Used by invalidateAnchorFor(physPage:) to unlink
    // an anchor when its pages get evicted by PageManager.
    private struct AnchorBack {
        weak var anchor: TrieAnchor?
        let tag: CvecAnchorTag
        let isPartial: Bool
    }
    private var anchorByPhys: [Int: AnchorBack] = [:]

    init(pageSlide: Int = 16) {
        self.pageSlide = pageSlide
        self.root = TrieAnchor(edgeTokens: [], depth: 0, parent: nil)
    }

    // Stats for engine-state telemetry.
    struct Stats {
        let nodeCount: Int
        let anchorCount: Int
        let maxDepth: Int
        let avgDepth: Double
    }
    func stats() -> Stats {
        var nodes = 0
        var anchors = 0
        var depthSum = 0
        var maxDepth = 0
        var depthSamples = 0
        func walk(_ n: TrieNode) {
            nodes += 1
            if let a = n as? TrieAnchor {
                if a !== root || !a.byCvecAnchorTag.isEmpty || !a.byCvecAnchorTagPartial.isEmpty {
                    anchors += 1
                    depthSum += a.depth
                    if a.depth > maxDepth { maxDepth = a.depth }
                    depthSamples += 1
                }
            }
            for c in n.children.values { walk(c) }
        }
        walk(root)
        let avg = depthSamples == 0 ? 0.0 : Double(depthSum) / Double(depthSamples)
        return Stats(nodeCount: nodes, anchorCount: anchors,
                     maxDepth: maxDepth, avgDepth: avg)
    }

    // ========================================================================
    // Insertion API.
    // ========================================================================

    // Insert a full page-pair anchor at depth (tokensCoveringFullPrefix.count)
    // along the path matching those tokens. Walks the trie, splitting
    // edges as needed; lands at (or creates) an anchor node at the
    // target depth and records the pair under cvecTag.
    func insertAnchor(tokensCoveringFullPrefix: ArraySlice<UInt32>,
                       pair: SlidePairContents,
                       cvecTag: CvecAnchorTag) {
        let target = tokensCoveringFullPrefix.count
        guard target > 0 else { return }
        let anchor = walkToAnchor(tokens: tokensCoveringFullPrefix,
                                  createIfMissing: true, targetDepth: target)
        guard let anchor = anchor else { return }
        anchor.byCvecAnchorTag[cvecTag] = pair
        anchorByPhys[pair.slidePlusFullHead] = AnchorBack(
            anchor: anchor, tag: cvecTag, isPartial: false)
        anchorByPhys[pair.fullTail] = AnchorBack(
            anchor: anchor, tag: cvecTag, isPartial: false)
    }

    // Insert a partial-pair anchor. Same walk as full; lands at anchor
    // whose depth equals tokensCoveringFullPrefix.count (NOT necessarily
    // a PAGE_SLIDE multiple — partials sit at end-of-prefix).
    func insertPartialAnchor(tokensCoveringFullPrefix: ArraySlice<UInt32>,
                              pair: PartialPagePair,
                              cvecTag: CvecAnchorTag) {
        let target = tokensCoveringFullPrefix.count
        guard target > 0 else { return }
        let anchor = walkToAnchor(tokens: tokensCoveringFullPrefix,
                                  createIfMissing: true, targetDepth: target)
        guard let anchor = anchor else { return }
        anchor.byCvecAnchorTagPartial[cvecTag] = pair
        anchorByPhys[pair.slidePlusFullHead] = AnchorBack(
            anchor: anchor, tag: cvecTag, isPartial: true)
        if let ft = pair.fullTail {
            anchorByPhys[ft] = AnchorBack(
                anchor: anchor, tag: cvecTag, isPartial: true)
        }
    }

    // Walk the trie consuming tokens. If `createIfMissing` is true,
    // creates intermediate nodes and (possibly) splits edges so that
    // a node lands exactly at `targetDepth`. Returns that node if it
    // is (or becomes) a TrieAnchor; nil if walk fails (shouldn't
    // happen with createIfMissing=true).
    private func walkToAnchor(tokens: ArraySlice<UInt32>,
                               createIfMissing: Bool,
                               targetDepth: Int) -> TrieAnchor? {
        var node: TrieNode = root
        var depthConsumed = 0
        let tokArr = Array(tokens)
        while depthConsumed < targetDepth {
            let firstTok = tokArr[depthConsumed]
            if let child = node.children[firstTok] {
                // Match child's edge against tokens[depthConsumed...].
                let edge = child.edgeTokens
                let remaining = targetDepth - depthConsumed
                let maxMatch = min(edge.count, remaining)
                var matched = 0
                while matched < maxMatch && edge[matched] == tokArr[depthConsumed + matched] {
                    matched += 1
                }
                if matched == edge.count {
                    // Full edge match — descend.
                    node = child
                    depthConsumed += matched
                } else if matched == 0 {
                    // Shouldn't happen (firstTok matched), but defensive.
                    return nil
                } else {
                    // Partial match — split the edge.
                    if !createIfMissing { return nil }
                    let splitDepth = node.depth + matched
                    let isAnchor = (splitDepth % pageSlide == 0) || (splitDepth == targetDepth)
                    let splitNode: TrieNode = isAnchor
                        ? TrieAnchor(edgeTokens: Array(edge[0..<matched]),
                                     depth: splitDepth, parent: node)
                        : TrieNode(edgeTokens: Array(edge[0..<matched]),
                                    depth: splitDepth, parent: node)
                    // Re-parent the existing child under splitNode with
                    // shortened edge.
                    child.edgeTokens = Array(edge[matched..<edge.count])
                    child.parent = splitNode
                    splitNode.children[child.edgeTokens[0]] = child
                    // Hook splitNode into node.
                    node.children[firstTok] = splitNode
                    node = splitNode
                    depthConsumed += matched
                }
            } else {
                // No matching child — create a new edge for the tail.
                if !createIfMissing { return nil }
                let edgeStart = depthConsumed
                let edgeEnd = targetDepth
                let edge = Array(tokArr[edgeStart..<edgeEnd])
                let newDepth = node.depth + edge.count
                // We're at the final landing depth.
                let newAnchor = TrieAnchor(edgeTokens: edge,
                                            depth: newDepth, parent: node)
                node.children[firstTok] = newAnchor
                node = newAnchor
                depthConsumed = edgeEnd
            }
        }
        // We're at depthConsumed == targetDepth. If the node isn't
        // already an anchor, promote it.
        if let a = node as? TrieAnchor { return a }
        // Promote: replace with anchor, preserving children + edge.
        guard let parent = node.parent else { return nil }
        let promoted = TrieAnchor(edgeTokens: node.edgeTokens,
                                   depth: node.depth, parent: parent)
        promoted.children = node.children
        for (_, c) in promoted.children { c.parent = promoted }
        let key = node.edgeTokens.first ?? 0
        parent.children[key] = promoted
        return promoted
    }

    // ========================================================================
    // Lookup API.
    // ========================================================================
    //
    // Walk the trie matching tokens. At each anchor we pass through,
    // check whether `byCvecAnchorTag[cvecTagFor(pageStart)]` is non-nil.
    // The result is the deepest adoptable contiguous-from-root anchor
    // chain. Per Track D §6 we also consider partial-pair anchors that
    // appear deeper than the last full anchor.

    func findLongestPrefix(
        tokens: ArraySlice<UInt32>,
        cvecTagFor: (Int) -> CvecAnchorTag
    ) -> PrefixMatch {
        // Walk token-by-token; collect adoptable anchors in order.
        var node: TrieNode = root
        var depthConsumed = 0
        let tokArr = Array(tokens)
        let totalLen = tokArr.count

        // Collect pairs in order along the walk. Each full-pair anchor
        // contributes one SlidePairContents (== two phys pages).
        var collectedPairs: [SlidePairContents] = []
        var lastAdoptedDepth = 0
        var partialTail: PartialPagePair? = nil

        outer: while depthConsumed < totalLen {
            let firstTok = tokArr[depthConsumed]
            guard let child = node.children[firstTok] else { break }
            let edge = child.edgeTokens
            let remaining = totalLen - depthConsumed
            let maxMatch = min(edge.count, remaining)
            var matched = 0
            while matched < maxMatch && edge[matched] == tokArr[depthConsumed + matched] {
                matched += 1
            }
            if matched == 0 { break }
            if matched < edge.count {
                // Partial edge match — we matched some tokens but can't
                // descend further. Before stopping, see if `child`
                // (the trie node at the end of this edge) holds a
                // partial-pair OR a full-pair anchor whose K/V we can
                // reuse as a partial for the matched-but-not-anchored
                // prefix of the edge.
                //
                // We need at least one token beyond lastAdoptedDepth
                // AND we cannot bridge the (lastAdoptedDepth ..
                // pageStart) gap if it's non-zero (the anchor's pair
                // covers K/V starting at pageStart, not at our
                // lastAdoptedDepth). So this only works when the edge
                // started right at a page boundary (lastAdoptedDepth
                // == pageStart of the partial).
                let divergeDepth = node.depth + matched
                if let anchor = child as? TrieAnchor {
                    let pageStart = ((anchor.depth - 1) / pageSlide) * pageSlide
                    if lastAdoptedDepth == pageStart {
                        var usable = divergeDepth - pageStart
                        // Convention (test contract): partial never
                        // consumes the LAST token of the caller's prompt
                        // — leave ≥ 1 token for the consumer's prefill.
                        let maxUsable = totalLen - 1 - pageStart
                        if usable > maxUsable { usable = maxUsable }
                        if usable > 0 && usable < pageSlide {
                            let tag = cvecTagFor(pageStart)
                            // Prefer explicit partial-pair entry; fall
                            // back to deriving from the full-pair entry.
                            if let pp = anchor.byCvecAnchorTagPartial[tag] {
                                let trimmed = min(pp.validUpTo, usable)
                                if trimmed > 0 {
                                    partialTail = PartialPagePair(
                                        slidePlusFullHead: pp.slidePlusFullHead,
                                        fullTail: pp.fullTail,
                                        validUpTo: trimmed)
                                }
                            } else if let pair = anchor.byCvecAnchorTag[tag] {
                                // Synthesize a partial from the full pair.
                                // slidePlusFullHead is always usable for slide
                                // layers AND for full layers up to PAGE_FULL.
                                // fullTail carries full K/V for the
                                // second-half (positions [PAGE_FULL..PAGE_SLIDE)).
                                let fullTail: Int? =
                                    (usable > pageSlide / 2) ? pair.fullTail : nil
                                partialTail = PartialPagePair(
                                    slidePlusFullHead: pair.slidePlusFullHead,
                                    fullTail: fullTail,
                                    validUpTo: usable)
                            }
                        }
                    }
                }
                depthConsumed += matched
                break outer
            }
            // Full descent.
            node = child
            depthConsumed += matched
            // Try to adopt the anchor at this node, if any.
            if let anchor = node as? TrieAnchor {
                // For full pairs the anchor is at PAGE_SLIDE multiples;
                // pageStart for that anchor = anchor.depth - PAGE_SLIDE.
                if anchor.depth % pageSlide == 0 && anchor.depth >= pageSlide {
                    let fullPageStart = anchor.depth - pageSlide
                    let tag = cvecTagFor(fullPageStart)
                    if let pair = anchor.byCvecAnchorTag[tag] {
                        collectedPairs.append(pair)
                        lastAdoptedDepth = anchor.depth
                    }
                    // else: not adoptable; keep walking (deeper anchors might match).
                }
                // For partial pairs the anchor sits at end-of-prefix
                // (any depth). Check for a partial that extends past
                // lastAdoptedDepth (and where the partial's validUpTo
                // bridges the gap from lastAdoptedDepth).
                if anchor.depth > lastAdoptedDepth {
                    let partialPageStart = (anchor.depth / pageSlide) * pageSlide
                    let tag = cvecTagFor(partialPageStart)
                    if let pp = anchor.byCvecAnchorTagPartial[tag] {
                        if anchor.depth - lastAdoptedDepth == pp.validUpTo {
                            // Test contract: never consume the LAST
                            // token of the caller's prompt.
                            let maxUsable = totalLen - 1 - lastAdoptedDepth
                            if maxUsable > 0 {
                                let trimmed = min(pp.validUpTo, maxUsable)
                                if trimmed > 0 {
                                    partialTail = PartialPagePair(
                                        slidePlusFullHead: pp.slidePlusFullHead,
                                        fullTail: pp.fullTail,
                                        validUpTo: trimmed)
                                }
                            }
                        }
                    }
                }
            }
        }
        let aligned = lastAdoptedDepth
        let trieLen = depthConsumed
        return PrefixMatch(alignedMatchLength: aligned,
                           trieMatchLength: trieLen,
                           pages: collectedPairs,
                           partialTail: partialTail)
    }

    // ========================================================================
    // Eviction callback.
    // ========================================================================
    //
    // PageManager calls this when it forcibly evicts a phys page (its
    // contents are about to be overwritten by a fresh allocation).
    // We must unlink the anchor that referenced this page so future
    // lookups don't return stale pointers.
    func invalidateAnchorFor(physPage: Int) {
        guard let back = anchorByPhys.removeValue(forKey: physPage),
              let anchor = back.anchor else { return }
        if back.isPartial {
            if let pp = anchor.byCvecAnchorTagPartial.removeValue(forKey: back.tag) {
                anchorByPhys.removeValue(forKey: pp.slidePlusFullHead)
                if let ft = pp.fullTail { anchorByPhys.removeValue(forKey: ft) }
            }
        } else {
            if let pair = anchor.byCvecAnchorTag.removeValue(forKey: back.tag) {
                anchorByPhys.removeValue(forKey: pair.slidePlusFullHead)
                anchorByPhys.removeValue(forKey: pair.fullTail)
            }
        }
        // Try to prune up the chain: if this anchor has no pairs AND
        // no children AND isn't root, unlink it from its parent.
        var n: TrieNode = anchor
        while n !== root {
            let canPrune: Bool
            if let a = n as? TrieAnchor {
                canPrune = a.children.isEmpty &&
                           a.byCvecAnchorTag.isEmpty &&
                           a.byCvecAnchorTagPartial.isEmpty
            } else {
                canPrune = n.children.isEmpty
            }
            if !canPrune { break }
            guard let parent = n.parent,
                  let firstTok = n.edgeTokens.first else { break }
            parent.children.removeValue(forKey: firstTok)
            n = parent
        }
    }
}

// ============================================================================
// Inline unit tests — gated on LM_TEST_RADIX_TRIE env var.
// ============================================================================
func runRadixTrieTests() {
    print("\n=== RadixTrie unit tests ===")
    var failed = 0
    var passed = 0
    func check(_ name: String, _ cond: Bool) {
        if cond { passed += 1; print("  ok  \(name)") }
        else    { failed += 1; print("  FAIL \(name)") }
    }

    // (1) Empty trie returns empty match.
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<32).map { UInt32($0) }
        let m = t.findLongestPrefix(tokens: tokens[0..<32]) { _ in .unsteered }
        check("empty trie → no match", m.alignedMatchLength == 0 && m.pages.isEmpty)
    }
    // (2) Single-page insert + exact lookup.
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<16).map { UInt32($0) }
        let pair = SlidePairContents(slidePlusFullHead: 100, fullTail: 101)
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16],
                       pair: pair, cvecTag: .unsteered)
        let m = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        check("single page insert+lookup matches 16",
              m.alignedMatchLength == 16 && m.pages.count == 1 &&
              m.pages[0].slidePlusFullHead == 100)
    }
    // (3) Multi-page insert + lookup covers chain.
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<64).map { UInt32($0) }
        for p in 0..<4 {
            let end = (p + 1) * 16
            let pair = SlidePairContents(slidePlusFullHead: 200 + p*2,
                                       fullTail: 200 + p*2 + 1)
            t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<end],
                           pair: pair, cvecTag: .unsteered)
        }
        let m = t.findLongestPrefix(tokens: tokens[0..<64]) { _ in .unsteered }
        check("multi-page chain returns 4 pairs",
              m.alignedMatchLength == 64 && m.pages.count == 4)
    }
    // (4) Divergent suffix: lookup tokens that share first 32 with cached.
    do {
        let t = RadixTrie(pageSlide: 16)
        let a: [UInt32] = (0..<48).map { UInt32($0) }
        for p in 0..<3 {
            let end = (p + 1) * 16
            let pair = SlidePairContents(slidePlusFullHead: 300 + p*2,
                                       fullTail: 300 + p*2 + 1)
            t.insertAnchor(tokensCoveringFullPrefix: a[0..<end],
                           pair: pair, cvecTag: .unsteered)
        }
        var b: [UInt32] = (0..<48).map { UInt32($0) }
        b[40] = 9999  // diverge mid-page-2
        let m = t.findLongestPrefix(tokens: b[0..<48]) { _ in .unsteered }
        check("divergent at pos 40 → only 2 pages adopt (32 tokens)",
              m.alignedMatchLength == 32 && m.pages.count == 2)
    }
    // (5) Cvec partition: same tokens, different tag → no match.
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<16).map { UInt32($0) }
        let pair = SlidePairContents(slidePlusFullHead: 400, fullTail: 401)
        let tagA = CvecAnchorTag(entries: [
            CvecAnchorTag.Entry(layer: 5, cvecId: "x", mode: "additive", units: "tokens",
                                attackQ: nil, decayQ: nil, sustainLevelQ: 100, releaseQ: nil,
                                shapeIfRamp: nil, peakTimesPolarityQ: 100,
                                startAnchor: nil, stopAnchor: nil,
                                targetQ: nil, transportScaleQ: nil, transportOffsetQ: nil)
        ])
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16],
                       pair: pair, cvecTag: tagA)
        let m = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        check("cvec mismatch → no match", m.alignedMatchLength == 0)
        let m2 = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in tagA }
        check("cvec match → 1 pair", m2.alignedMatchLength == 16 && m2.pages.count == 1)
    }
    // (6) Eviction callback unlinks the anchor.
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<32).map { UInt32($0) }
        let p0 = SlidePairContents(slidePlusFullHead: 500, fullTail: 501)
        let p1 = SlidePairContents(slidePlusFullHead: 502, fullTail: 503)
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16], pair: p0, cvecTag: .unsteered)
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<32], pair: p1, cvecTag: .unsteered)
        t.invalidateAnchorFor(physPage: 502)
        let m = t.findLongestPrefix(tokens: tokens[0..<32]) { _ in .unsteered }
        check("eviction unlinks deeper anchor; first still adopts",
              m.alignedMatchLength == 16 && m.pages.count == 1)
    }
    // (7) Edge split: insert AB...P then ABQ... — second insert forces split.
    do {
        let t = RadixTrie(pageSlide: 16)
        let a: [UInt32] = (0..<32).map { UInt32($0) }
        let pA0 = SlidePairContents(slidePlusFullHead: 600, fullTail: 601)
        let pA1 = SlidePairContents(slidePlusFullHead: 602, fullTail: 603)
        t.insertAnchor(tokensCoveringFullPrefix: a[0..<16], pair: pA0, cvecTag: .unsteered)
        t.insertAnchor(tokensCoveringFullPrefix: a[0..<32], pair: pA1, cvecTag: .unsteered)
        var b: [UInt32] = a
        b[20] = 7777
        let pB1 = SlidePairContents(slidePlusFullHead: 604, fullTail: 605)
        // This insertion shares the first 16 tokens (page 0) and diverges
        // mid-page-1. We expect a node split at position 20 (well, we
        // currently land at depth 32 for B too with a separate anchor).
        t.insertAnchor(tokensCoveringFullPrefix: b[0..<32], pair: pB1, cvecTag: .unsteered)
        let mA = t.findLongestPrefix(tokens: a[0..<32]) { _ in .unsteered }
        let mB = t.findLongestPrefix(tokens: b[0..<32]) { _ in .unsteered }
        check("after split: A still adopts 32",
              mA.alignedMatchLength == 32 && mA.pages.count == 2)
        check("after split: B adopts 16 (page 0 only — page 1 differs)",
              mB.alignedMatchLength == 16 && mB.pages.count == 1 &&
              mB.pages[0].slidePlusFullHead == 600)
    }
    // (8) Partial pair: insert short partial, lookup returns it.
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<22).map { UInt32($0) }
        let pair0 = SlidePairContents(slidePlusFullHead: 700, fullTail: 701)
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16],
                       pair: pair0, cvecTag: .unsteered)
        let partial = PartialPagePair(slidePlusFullHead: 702, fullTail: nil, validUpTo: 6)
        t.insertPartialAnchor(tokensCoveringFullPrefix: tokens[0..<22],
                               pair: partial, cvecTag: .unsteered)
        let m = t.findLongestPrefix(tokens: tokens[0..<22]) { _ in .unsteered }
        check("partial pair returned in partialTail",
              m.alignedMatchLength == 16 && m.partialTail != nil &&
              m.partialTail?.validUpTo == 6)
    }

    print("RadixTrie tests: \(passed) passed, \(failed) failed")
    if failed > 0 {
        exit(1)
    }
}
