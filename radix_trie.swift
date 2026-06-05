// radix_trie.swift — Token-granularity prefix cache (Track D of the
// 2026-05-23 prefix-cache grand-slam refactor).
//
// Replaces `PageManager.contentIndex: [UInt64: Int]` with a radix trie of
// token IDs. Edge-compressed; anchors live at PAGE multiples (and at
// end-of-prefix for Track-B partial pages). Each anchor carries a
// `[CvecAnchorTag: Int]` map (the single phys page) plus a
// `[CvecAnchorTag: PartialPage]` map for partial-page sharing.
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
    // Full 64-bit digest of the IMAGE(s) prefilled into this page's prefix.
    // Folded into the partition key so two streams with byte-identical TEXT but
    // DIFFERENT images get different tags and can NEVER adopt each other's
    // soft-token K/V pages (cross-stream image-leak fix) — even when the 32-bit
    // soft-token placeholders in consumedTokens collide. 0 = no image (text-only),
    // which preserves all existing text-prefix sharing unchanged.
    var imageDigest: UInt64 = 0

    static let unsteered = CvecAnchorTag(entries: [])

    var isUnsteered: Bool { entries.isEmpty }
}

// PartialPage — a (phys, validUpTo) pair used by Track B (partial-page
// promotion at teardown). Under ONE PAGE=16 for all layers, a 16-token page
// is ONE physical page (ownedPages[P]); the K/V for positions
// [pageStart, pageStart + validUpTo) live in that single page.
//
// `validUpTo` is in [1, PAGE-1]: 0 means "no progress" (no partial worth
// promoting), PAGE means "full page" (a plain phys Int anchor instead).
struct PartialPage {
    let phys: Int                  // ownedPages[P] covering [16P..16P+15]
    let validUpTo: Int             // 1..PAGE-1
}

// PageRef — Tier 1 (2026-06): a full-page anchor entry is either RAM-resident
// (phys page index) or SSD-tiered (demoted to a KvSsdStore slot, no phys until
// reloaded). The `contentHash` rides along on the SSD case so a reload can
// re-establish PageManager.markContentIndexed without re-walking tokens (the
// hash was already computed at promote time). RAM entries don't need the hash
// (PageManager already holds it on the live phys page).
//
// Tier 0 (pin-on-grow KV residency) is untouched: a reloaded page is a freshly
// allocated RAM page whose chunk Tier 0 pins on grow. Partial pages stay
// RAM-only (PartialPage, below) — they are teardown-only producer hints and are
// never demoted, limiting the tiering blast radius.
enum PageRef {
    case ram(Int)                       // phys page index (RAM-resident)
    case ssd(slot: Int, contentHash: UInt64)  // demoted to KvSsdStore slot
}

// PrefixMatch — the lookup result. `pages` are pre-incref'd; caller
// MUST decref each on session teardown. `partialTail` is set when
// the deepest matched anchor carries a partial page; the caller is
// expected to CoW the partial bytes onto a fresh page.
//
// Each `pages` entry is the SINGLE page for one 16-token page (ownedPages[P]);
// the consumer reconstructs ownedPages page-for-page in order. Tier 1: an entry
// is .ram(phys) (adopt directly) or .ssd(slot,hash) (reload first, then adopt).
// Read-only by construction (one PAGE=16 means there is no slide-divergent half
// to privatize — the old CoW is gone).
struct PrefixMatch {
    let alignedMatchLength: Int    // == pages.count * PAGE (one page per entry)
    let trieMatchLength: Int       // longest token-walk; reported for telemetry
    let pages: [PageRef]           // RAM|SSD page refs covering [0, alignedMatchLength)
    let partialTail: PartialPage?
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

// TrieAnchor — a TrieNode whose `depth` is a PAGE multiple (or
// is an end-of-prefix anchor for a partial pair). Carries the cvec
// partition maps plus the underlying physical page indices used for
// the eviction callback.
final class TrieAnchor: TrieNode {
    // Full pages (validUpTo == PAGE) keyed by CvecAnchorTag. Tier 1: the value
    // is a PageRef — .ram(phys) (resident) or .ssd(slot,hash) (demoted to the
    // cold SSD store, no phys until reloaded).
    var byCvecAnchorTag: [CvecAnchorTag: PageRef] = [:]
    // Partial pages (validUpTo < PAGE) keyed by CvecAnchorTag. RAM-only — never
    // demoted (teardown-only producer hints; a partial whose phys gets demoted
    // simply loses its partial entry via the existing invalidate path).
    var byCvecAnchorTagPartial: [CvecAnchorTag: PartialPage] = [:]
}

// PageManager-agnostic radix trie. Phys pages are owned by PageManager;
// the trie holds back-pointers via the phys Int / PartialPage values
// stored at anchors.
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
    // Multi-valued (page_lifecycle_audit_2026-05-28 #4): a phys page can be
    // referenced by more than one anchor (a full-pair AND a partial-pair at
    // different depths). A single-valued map let the 2nd insert clobber the
    // 1st, so eviction unlinked only one anchor and the other served a
    // reallocated page (stale adoption). Track ALL backs per phys.
    private var anchorByPhys: [Int: [AnchorBack]] = [:]

    // Tier 1 (2026-06): reverse map ssdSlot -> the single anchor entry holding
    // .ssd(slot) for reloadAnchor lookup. An SSD-tiered page has NO phys (it is
    // absent from anchorByPhys) until reloaded; this is its back-pointer. A slot
    // is owned by exactly one (anchor, tag) — demote moves the entry off phys
    // and into this map; reload moves it back. The KvSsdStore itself never
    // references anchors (it is a dumb slab allocator); all anchor<->slot
    // ownership lives here in the trie.
    private struct SsdBack {
        weak var anchor: TrieAnchor?
        let tag: CvecAnchorTag
    }
    // Multi-valued, mirroring anchorByPhys: a phys page may be referenced by
    // more than one full anchor (same bytes via different paths/tags). Demote
    // gathers the bytes ONCE into one slot and points ALL of that phys's full
    // backs at it; reload flips them all back to the same fresh phys.
    private var anchorBySsdSlot: [Int: [SsdBack]] = [:]

    // Tier 1 in-tier LRU (2026-06): belt-and-suspenders reclaim callback. Fired
    // whenever a DEFENSIVE removal path (invalidateAnchorFor) clears a
    // byCvecAnchorTag entry that still held a .ssd(slot,_) value, so the engine
    // can reclaim the orphaned slab slot (wired in LmEngine.init to
    // ssdStore.freeSlot). NOT fired by the explicit eviction path
    // (orphanSsdSlot): there the engine calls store.freeSlot directly, so
    // firing here too would double-free. See clearTagEntry below.
    var onSsdSlotOrphaned: ((Int) -> Void)? = nil

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
    @discardableResult
    func insertAnchor(tokensCoveringFullPrefix: ArraySlice<UInt32>,
                       phys: Int,
                       cvecTag: CvecAnchorTag) -> Bool {
        let target = tokensCoveringFullPrefix.count
        guard target > 0 else { return false }
        let anchor = walkToAnchor(tokens: tokensCoveringFullPrefix,
                                  createIfMissing: true, targetDepth: target)
        guard let anchor = anchor else { return false }
        if anchor.byCvecAnchorTag[cvecTag] != nil {
            return false
        }
        anchor.byCvecAnchorTag[cvecTag] = .ram(phys)
        anchorByPhys[phys, default: []].append(
            AnchorBack(anchor: anchor, tag: cvecTag, isPartial: false))
        return true
    }

    // Insert a partial-pair anchor. Same walk as full; lands at anchor
    // whose depth equals tokensCoveringFullPrefix.count (NOT necessarily
    // a PAGE multiple — partials sit at end-of-prefix).
    @discardableResult
    func insertPartialAnchor(tokensCoveringFullPrefix: ArraySlice<UInt32>,
                              partial: PartialPage,
                              cvecTag: CvecAnchorTag) -> Bool {
        let target = tokensCoveringFullPrefix.count
        guard target > 0 else { return false }
        let anchor = walkToAnchor(tokens: tokensCoveringFullPrefix,
                                  createIfMissing: true, targetDepth: target)
        guard let anchor = anchor else { return false }
        if anchor.byCvecAnchorTagPartial[cvecTag] != nil {
            return false
        }
        anchor.byCvecAnchorTagPartial[cvecTag] = partial
        anchorByPhys[partial.phys, default: []].append(
            AnchorBack(anchor: anchor, tag: cvecTag, isPartial: true))
        return true
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

        // Collect page refs in order along the walk. Each full anchor
        // contributes one PageRef — .ram(phys) (resident) or .ssd(slot,hash)
        // (Tier 1 demoted; the consumer reloads it before adoption).
        var collectedPages: [PageRef] = []
        var lastAdoptedDepth = 0
        var partialTail: PartialPage? = nil

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
                        let usable = divergeDepth - pageStart
                        if usable > 0 && usable < pageSlide {
                            let tag = cvecTagFor(pageStart)
                            // Prefer explicit partial entry; fall back to
                            // deriving from the full-page entry (same phys,
                            // valid up to `usable` positions).
                            if let pp = anchor.byCvecAnchorTagPartial[tag] {
                                let trimmed = min(pp.validUpTo, usable)
                                if trimmed > 0 {
                                    partialTail = PartialPage(
                                        phys: pp.phys, validUpTo: trimmed)
                                }
                            } else if case .ram(let phys)? = anchor.byCvecAnchorTag[tag] {
                                // Only a RAM-resident full page can be sliced
                                // into a partial-tail here. An SSD-tiered page
                                // is not loaded; skip the partial derivation
                                // (the suffix re-prefills, harmless).
                                partialTail = PartialPage(
                                    phys: phys, validUpTo: usable)
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
                // For full pages the anchor is at PAGE multiples;
                // pageStart for that anchor = anchor.depth - PAGE.
                if anchor.depth % pageSlide == 0 && anchor.depth >= pageSlide {
                    let fullPageStart = anchor.depth - pageSlide
                    let tag = cvecTagFor(fullPageStart)
                    if let ref = anchor.byCvecAnchorTag[tag] {
                        // Tier 1: an .ssd entry is still adoptable — it is a
                        // contiguous-from-root page, just needs reload first
                        // (the consumer pread+scatters into a fresh RAM page).
                        // gap-stop / contiguous logic is unchanged.
                        collectedPages.append(ref)
                        lastAdoptedDepth = anchor.depth
                        if ProcessInfo.processInfo.environment["LM_TRIE_DEBUG"] != nil {
                            FileHandle.standardError.write(Data("[trie] COLLECT depth=\(anchor.depth) tagUnsteered=\(tag.isUnsteered) ref=\(ref)\n".utf8))
                        }
                    } else {
                        if ProcessInfo.processInfo.environment["LM_TRIE_DEBUG"] != nil {
                            FileHandle.standardError.write(Data("[trie] GAP depth=\(anchor.depth) wantTagUnsteered=\(tag.isUnsteered) nKeys=\(anchor.byCvecAnchorTag.count) keysUnsteered=\(anchor.byCvecAnchorTag.keys.map{ $0.isUnsteered })\n".utf8))
                        }
                        // GAP: this page boundary is NOT adoptable (the anchor
                        // was unlinked by a MID-CHAIN eviction — pruneAnchorChain
                        // can't prune it while it has deeper children — or the
                        // cvec tag doesn't match this session). Adoption is
                        // contiguous-from-root by definition: ownedPages[p] must
                        // back token positions [p·PAGE .. p·PAGE+PAGE-1], so we
                        // CANNOT skip this page and graft deeper pages onto its
                        // slot. Stop the adoptable run here. This is the source
                        // that GUARANTEES the PrefixMatch contract
                        //   alignedMatchLength == pages.count * pageSlide
                        // (consumed by adoptSharedPrefixPages: position is set to
                        // pages.count·PAGE, and ownedPages[i]=pages[i]). Walking
                        // past the gap divorced the position tag from the
                        // physical bytes — the exact full/slide-aliasing class of
                        // corruption the ONE-PAGE=16 refactor set out to kill,
                        // here re-entering through the trie collector.
                        break outer
                    }
                }
                // For partial pages the anchor sits at end-of-prefix
                // (any depth). Check for a partial that extends past
                // lastAdoptedDepth (and where the partial's validUpTo
                // bridges the gap from lastAdoptedDepth).
                if anchor.depth > lastAdoptedDepth {
                    let partialPageStart = (anchor.depth / pageSlide) * pageSlide
                    let tag = cvecTagFor(partialPageStart)
                    if let pp = anchor.byCvecAnchorTagPartial[tag] {
                        if anchor.depth - lastAdoptedDepth == pp.validUpTo {
                            let trimmed = pp.validUpTo
                            if trimmed > 0 {
                                partialTail = PartialPage(
                                    phys: pp.phys, validUpTo: trimmed)
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
                           pages: collectedPages,
                           partialTail: partialTail)
    }

    // ========================================================================
    // Tier 1 demote / reload API (2026-06).
    // ========================================================================

    // DEMOTE (RAM -> SSD). The engine has gathered phys's K/V into SSD slot
    // `ssdSlot` (perPageBytes) and is about to reuse the RAM page. For EACH full
    // (non-partial) anchor referencing this phys: flip its map entry
    // .ram(phys) -> .ssd(slot, contentHash) and register it in anchorBySsdSlot.
    // CRITICAL: drop this phys's back-entries from anchorByPhys so a LATER
    // eviction of the now-reused RAM page does NOT touch these anchors (the
    // back-map must only ever index RAM-resident pages). Partial backs on this
    // phys cannot be tiered (RAM-only) -> drop them (remove the partial entry),
    // exactly as the existing drop path would. Returns true if at least one
    // full anchor was tiered (so the engine knows the slot is in use).
    @discardableResult
    func demoteAnchor(physPage: Int, ssdSlot: Int, contentHash: UInt64) -> Bool {
        guard let backs = anchorByPhys.removeValue(forKey: physPage) else {
            return false
        }
        var tiered = false
        var prunedPartials: [TrieAnchor] = []
        for back in backs {
            guard let anchor = back.anchor else { continue }
            if back.isPartial {
                // Partials are not tiered — drop, like the existing path.
                anchor.byCvecAnchorTagPartial.removeValue(forKey: back.tag)
                prunedPartials.append(anchor)
                continue
            }
            // Flip the full-page entry to SSD-resident.
            anchor.byCvecAnchorTag[back.tag] = .ssd(slot: ssdSlot,
                                                    contentHash: contentHash)
            anchorBySsdSlot[ssdSlot, default: []].append(
                SsdBack(anchor: anchor, tag: back.tag))
            tiered = true
        }
        // A partial whose phys got tiered may now be a prunable empty anchor.
        for anchor in prunedPartials { pruneAnchorChain(anchor) }
        return tiered
    }

    // RELOAD (SSD -> RAM). The engine has pread+scattered slot `ssdSlot` into a
    // freshly-allocated RAM page `intoPhys` (bit-exact). Flip EVERY anchor entry
    // holding .ssd(slot) back to .ram(intoPhys) and re-establish the
    // anchorByPhys back-entry so a future eviction can re-demote/drop it. The
    // slot itself is reclaimed by the caller (KvSsdStore.freeSlot).
    func reloadAnchor(ssdSlot: Int, intoPhys: Int) {
        guard let backs = anchorBySsdSlot.removeValue(forKey: ssdSlot) else {
            return
        }
        for back in backs {
            guard let anchor = back.anchor else { continue }
            anchor.byCvecAnchorTag[back.tag] = .ram(intoPhys)
            anchorByPhys[intoPhys, default: []].append(
                AnchorBack(anchor: anchor, tag: back.tag, isPartial: false))
        }
    }

    // ORPHAN (SSD slot evicted by the in-tier LRU). Tier 1 (2026-06): the engine
    // is reclaiming SSD slab slot `slot` to make room for a new demote (the
    // store hit its cap). Drop ALL trie bookkeeping for this slot WITHOUT any
    // byte I/O and WITHOUT firing onSsdSlotOrphaned (the engine caller owns the
    // explicit store.freeSlot for `slot`). For each anchor entry that still
    // points at THIS slot, REMOVE the byCvecAnchorTag entry (no .ram
    // replacement): that prefix becomes a normal GAP, so the next lookup finds
    // no entry and the prefix simply re-prefills — correct + safe, NEVER serves
    // stale bytes. Then pruneAnchorChain reclaims the now-empty dead anchor
    // (this is the leak fix: orphanSsdSlot empties the map FIRST so canPrune,
    // which requires byCvecAnchorTag.isEmpty, can finally fire).
    func orphanSsdSlot(_ slot: Int) {
        guard let backs = anchorBySsdSlot.removeValue(forKey: slot) else { return }
        var touched: [TrieAnchor] = []
        for back in backs {
            guard let anchor = back.anchor else { continue }
            // Only remove if it still points at THIS slot (defensive; under
            // gEngineLock single-threaded this is always true — a concurrent
            // reload would have removed it from anchorBySsdSlot first).
            if case .ssd(let s, _)? = anchor.byCvecAnchorTag[back.tag], s == slot {
                anchor.byCvecAnchorTag.removeValue(forKey: back.tag)
                touched.append(anchor)
            }
        }
        for anchor in touched { pruneAnchorChain(anchor) }
    }

    // Clear a full-page tag entry and, if it held a .ssd(slot) value, fire the
    // belt-and-suspenders onSsdSlotOrphaned callback AND drop the matching
    // anchorBySsdSlot back so the store slot is reclaimed even on the DEFENSIVE
    // removal paths (invalidateAnchorFor). Routes the one full-branch removal in
    // invalidateAnchorFor through here so any future .ssd-holding clear is
    // covered. orphanSsdSlot does NOT use this (it must not fire the callback).
    private func clearTagEntry(_ anchor: TrieAnchor, _ tag: CvecAnchorTag) {
        guard let removed = anchor.byCvecAnchorTag.removeValue(forKey: tag) else { return }
        if case .ssd(let slot, _) = removed {
            // Drop the matching back so anchorBySsdSlot stays consistent.
            if var backs = anchorBySsdSlot[slot] {
                backs.removeAll { $0.anchor === anchor && $0.tag == tag }
                if backs.isEmpty { anchorBySsdSlot[slot] = nil }
                else { anchorBySsdSlot[slot] = backs }
            }
            onSsdSlotOrphaned?(slot)
        }
    }

    // TEST-ONLY (in-tier-LRU regression #16): forge a full-page anchor whose
    // entry is .ssd(slot,hash) AND is ALSO indexed by anchorByPhys[phys], i.e.
    // the defensive co-existence that the public demote path never produces
    // (demote removes the phys back). Lets the test drive invalidateAnchorFor's
    // clearTagEntry .ssd branch (the belt-and-suspenders fire). Not used outside
    // tests; pure trie bookkeeping, no byte I/O.
    func debugForceSsdEntryWithPhysBack(tokens: ArraySlice<UInt32>, phys: Int,
                                        slot: Int, contentHash: UInt64,
                                        cvecTag: CvecAnchorTag) {
        let target = tokens.count
        guard target > 0,
              let anchor = walkToAnchor(tokens: tokens, createIfMissing: true,
                                        targetDepth: target) else { return }
        anchor.byCvecAnchorTag[cvecTag] = .ssd(slot: slot, contentHash: contentHash)
        anchorBySsdSlot[slot, default: []].append(SsdBack(anchor: anchor, tag: cvecTag))
        anchorByPhys[phys, default: []].append(
            AnchorBack(anchor: anchor, tag: cvecTag, isPartial: false))
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
        // Unlink ALL anchors referencing this phys (see #4 above). The evicted
        // page's own back-list is removed here; each referenced anchor's entry
        // is dropped. Under ONE PAGE=16 each anchor entry is a SINGLE phys, so
        // there is no partner page to prune.
        guard let backs = anchorByPhys.removeValue(forKey: physPage) else { return }
        var touched: [TrieAnchor] = []
        for back in backs {
            guard let anchor = back.anchor else { continue }
            if back.isPartial {
                anchor.byCvecAnchorTagPartial.removeValue(forKey: back.tag)
            } else {
                // .ssd-aware clear (belt-and-suspenders): if this entry still
                // held a .ssd value (in practice unreachable via the
                // anchorByPhys loop — a .ssd page is never in anchorByPhys —
                // but defensive against future removal-site edits), the
                // callback reclaims the slab slot.
                clearTagEntry(anchor, back.tag)
            }
            touched.append(anchor)
        }
        for anchor in touched { pruneAnchorChain(anchor) }
    }

    // Prune empty anchors up to (not including) root.
    private func pruneAnchorChain(_ anchor: TrieAnchor) {
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
    // Tier 1: pages are now [PageRef]. Unwrap .ram(phys) for the assertions
    // (insertAnchor registers RAM pages; -1 sentinel for any SSD entry, which
    // these tests never produce).
    func ram(_ r: PageRef) -> Int { if case .ram(let p) = r { return p }; return -1 }

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
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16],
                       phys: 100, cvecTag: .unsteered)
        let m = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        check("single page insert+lookup matches 16",
              m.alignedMatchLength == 16 && m.pages.count == 1 &&
              ram(m.pages[0]) == 100)
    }
    // (3) Multi-page insert + lookup covers chain.
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<64).map { UInt32($0) }
        for p in 0..<4 {
            let end = (p + 1) * 16
            t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<end],
                           phys: 200 + p, cvecTag: .unsteered)
        }
        let m = t.findLongestPrefix(tokens: tokens[0..<64]) { _ in .unsteered }
        check("multi-page chain returns 4 pages",
              m.alignedMatchLength == 64 && m.pages.count == 4)
    }
    // (4) Divergent suffix: lookup tokens that share first 32 with cached.
    do {
        let t = RadixTrie(pageSlide: 16)
        let a: [UInt32] = (0..<48).map { UInt32($0) }
        for p in 0..<3 {
            let end = (p + 1) * 16
            t.insertAnchor(tokensCoveringFullPrefix: a[0..<end],
                           phys: 300 + p, cvecTag: .unsteered)
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
        let tagA = CvecAnchorTag(entries: [
            CvecAnchorTag.Entry(layer: 5, cvecId: "x", mode: "additive", units: "tokens",
                                attackQ: nil, decayQ: nil, sustainLevelQ: 100, releaseQ: nil,
                                shapeIfRamp: nil, peakTimesPolarityQ: 100,
                                startAnchor: nil, stopAnchor: nil,
                                targetQ: nil, transportScaleQ: nil, transportOffsetQ: nil)
        ])
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16],
                       phys: 400, cvecTag: tagA)
        let m = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        check("cvec mismatch → no match", m.alignedMatchLength == 0)
        let m2 = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in tagA }
        check("cvec match → 1 page", m2.alignedMatchLength == 16 && m2.pages.count == 1)
    }
    // (6) Eviction callback unlinks the anchor.
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<32).map { UInt32($0) }
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16], phys: 500, cvecTag: .unsteered)
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<32], phys: 502, cvecTag: .unsteered)
        t.invalidateAnchorFor(physPage: 502)
        let m = t.findLongestPrefix(tokens: tokens[0..<32]) { _ in .unsteered }
        check("eviction unlinks deeper anchor; first still adopts",
              m.alignedMatchLength == 16 && m.pages.count == 1)
    }
    // (6b) MID-CHAIN eviction: evict an interior page whose anchor still has
    // deeper children (so pruneAnchorChain cannot remove it). The walk MUST
    // stop the adoptable run at the gap — it must NOT skip the evicted page
    // and graft the surviving deeper page onto the gap's logical-page slot.
    // (Regression guard for the state-dependent adopted-prefix corruption:
    //  without the gap-stop, this returned alignedMatchLength=48/pages=[s1,s2]
    //  — the deeper pages mismapped one logical page early, divorcing the
    //  position tag from the physical bytes.)
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<48).map { UInt32($0) }
        for p in 0..<3 {
            let end = (p + 1) * 16
            t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<end],
                           phys: 800 + p, cvecTag: .unsteered)
        }
        // Evict the MIDDLE page (logical page 1, depth 32). Its anchor keeps
        // its depth-48 child, so the anchor node survives (loses only its phys).
        t.invalidateAnchorFor(physPage: 801)
        let m = t.findLongestPrefix(tokens: tokens[0..<48]) { _ in .unsteered }
        check("mid-chain eviction stops adoption at the gap (contiguous-from-root)",
              m.alignedMatchLength == 16 && m.pages.count == 1 && ram(m.pages[0]) == 800)
        check("mid-chain eviction preserves the contract alignedMatchLength == pages.count*PAGE",
              m.alignedMatchLength == m.pages.count * 16)
    }
    // (7) Edge split: insert AB...P then ABQ... — second insert forces split.
    do {
        let t = RadixTrie(pageSlide: 16)
        let a: [UInt32] = (0..<32).map { UInt32($0) }
        t.insertAnchor(tokensCoveringFullPrefix: a[0..<16], phys: 600, cvecTag: .unsteered)
        t.insertAnchor(tokensCoveringFullPrefix: a[0..<32], phys: 602, cvecTag: .unsteered)
        var b: [UInt32] = a
        b[20] = 7777
        // This insertion shares the first 16 tokens (page 0) and diverges
        // mid-page-1. We expect a node split at position 20 (well, we
        // currently land at depth 32 for B too with a separate anchor).
        t.insertAnchor(tokensCoveringFullPrefix: b[0..<32], phys: 604, cvecTag: .unsteered)
        let mA = t.findLongestPrefix(tokens: a[0..<32]) { _ in .unsteered }
        let mB = t.findLongestPrefix(tokens: b[0..<32]) { _ in .unsteered }
        check("after split: A still adopts 32",
              mA.alignedMatchLength == 32 && mA.pages.count == 2)
        check("after split: B adopts its own 32-token branch",
              mB.alignedMatchLength == 32 && mB.pages.count == 2 &&
              ram(mB.pages[0]) == 600 &&
              ram(mB.pages[1]) == 604)
    }
    // (8) Partial page: insert short partial, lookup returns it.
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<22).map { UInt32($0) }
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16],
                       phys: 700, cvecTag: .unsteered)
        let partial = PartialPage(phys: 702, validUpTo: 6)
        t.insertPartialAnchor(tokensCoveringFullPrefix: tokens[0..<22],
                               partial: partial, cvecTag: .unsteered)
        let m = t.findLongestPrefix(tokens: tokens[0..<22]) { _ in .unsteered }
        check("partial page returned in partialTail",
              m.alignedMatchLength == 16 && m.partialTail != nil &&
              m.partialTail?.validUpTo == 6)
    }

    // (9) Tier 1: demote flips RAM->SSD, lookup yields .ssd; reload flips back.
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<16).map { UInt32($0) }
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16],
                       phys: 900, cvecTag: .unsteered)
        // Demote phys 900 -> SSD slot 5 with a content hash.
        let didTier = t.demoteAnchor(physPage: 900, ssdSlot: 5, contentHash: 0xDEAD)
        let mDemoted = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        var sawSsd = false
        if mDemoted.pages.count == 1, case .ssd(let slot, let h) = mDemoted.pages[0] {
            sawSsd = (slot == 5 && h == 0xDEAD)
        }
        check("demote: anchor flips RAM->SSD, lookup returns .ssd(slot,hash)",
              didTier && mDemoted.alignedMatchLength == 16 && sawSsd)
        // A demoted page's phys back-entry is gone: invalidating the OLD phys is
        // a no-op (does NOT re-drop the now-SSD anchor).
        t.invalidateAnchorFor(physPage: 900)
        let mStillSsd = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        check("demote: old phys invalidation does not touch the SSD anchor",
              mStillSsd.alignedMatchLength == 16)
        // Reload SSD slot 5 -> fresh phys 901; anchor flips back to RAM.
        t.reloadAnchor(ssdSlot: 5, intoPhys: 901)
        let mReloaded = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        check("reload: anchor flips SSD->RAM(newphys)",
              mReloaded.alignedMatchLength == 16 && mReloaded.pages.count == 1 &&
              ram(mReloaded.pages[0]) == 901)
        // Now eviction of the NEW phys correctly drops it (back-entry restored).
        t.invalidateAnchorFor(physPage: 901)
        let mGone = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        check("reload: new phys back-entry restored (eviction now drops it)",
              mGone.alignedMatchLength == 0)
    }

    // (10) Tier 1 REPRO: MULTI-PAGE full demote. A 3-page prefix whose pages are
    // ALL demoted to SSD must still be fully collected by findLongestPrefix (the
    // re-adopt-after-full-eviction path that fails in the engine: reloads=0).
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<48).map { UInt32($0) }
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16], phys: 100, cvecTag: .unsteered)
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<32], phys: 101, cvecTag: .unsteered)
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<48], phys: 102, cvecTag: .unsteered)
        let mBefore = t.findLongestPrefix(tokens: tokens[0..<48]) { _ in .unsteered }
        check("multipage: 3 RAM pages adopt before demote",
              mBefore.pages.count == 3 && mBefore.alignedMatchLength == 48)
        _ = t.demoteAnchor(physPage: 100, ssdSlot: 10, contentHash: 0xA0)
        _ = t.demoteAnchor(physPage: 101, ssdSlot: 11, contentHash: 0xA1)
        _ = t.demoteAnchor(physPage: 102, ssdSlot: 12, contentHash: 0xA2)
        let mAfter = t.findLongestPrefix(tokens: tokens[0..<48]) { _ in .unsteered }
        var allSsd = (mAfter.pages.count == 3)
        for p in mAfter.pages { if case .ssd = p {} else { allSsd = false } }
        check("multipage: all 3 pages collected as .ssd after FULL demote (re-adopt repro)",
              mAfter.pages.count == 3 && mAfter.alignedMatchLength == 48 && allSsd)
    }

    // (11) Tier 1 REPRO: MIXED demote (eviction hits pages in arbitrary order).
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<48).map { UInt32($0) }
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16], phys: 200, cvecTag: .unsteered)
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<32], phys: 201, cvecTag: .unsteered)
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<48], phys: 202, cvecTag: .unsteered)
        _ = t.demoteAnchor(physPage: 201, ssdSlot: 20, contentHash: 0xB1)  // middle page first
        let mMid = t.findLongestPrefix(tokens: tokens[0..<48]) { _ in .unsteered }
        check("mixed: middle page demoted still yields 3 contiguous pages",
              mMid.pages.count == 3 && mMid.alignedMatchLength == 48)
    }

    // (12) Tier 1 REPRO (H4): a demoted page's RAM phys is REUSED by a new anchor,
    // then that new anchor is DROPPED (evicted). The original demoted anchor must
    // stay .ssd — the reused-phys drop must NOT clobber it. (Test 9 demotes+invalidates
    // the SAME phys without an intervening reuse, so it misses this.)
    do {
        let t = RadixTrie(pageSlide: 16)
        let xtok: [UInt32] = (100..<116).map { UInt32($0) }
        let ftok: [UInt32] = (200..<216).map { UInt32($0) }
        t.insertAnchor(tokensCoveringFullPrefix: xtok[0..<16], phys: 500, cvecTag: .unsteered)
        _ = t.demoteAnchor(physPage: 500, ssdSlot: 30, contentHash: 0xC0)   // X demoted; phys 500 freed
        t.insertAnchor(tokensCoveringFullPrefix: ftok[0..<16], phys: 500, cvecTag: .unsteered)  // filler REUSES phys 500
        t.invalidateAnchorFor(physPage: 500)                                // filler dropped
        let mX = t.findLongestPrefix(tokens: xtok[0..<16]) { _ in .unsteered }
        var xSsd = false
        if mX.pages.count == 1, case .ssd(let s, _) = mX.pages[0] { xSsd = (s == 30) }
        check("H4: reused-phys drop does NOT clobber the demoted X anchor (.ssd survives)",
              mX.alignedMatchLength == 16 && xSsd)
    }
    // (13) Tier 1 REPRO (H3): re-PROMOTE of DUPLICATE content at a NEW phys. If
    // insertAnchor returns false for the dup and does NOT register an anchorByPhys
    // back-entry for the new phys, demoteAnchor(newPhys) returns false -> engine
    // DROPs -> the anchor entry is later nil'd. Here we assert the dup phys is
    // demotable (has a back-entry) OR the original remains adoptable.
    do {
        let t = RadixTrie(pageSlide: 16)
        let xtok: [UInt32] = (300..<316).map { UInt32($0) }
        let ins1 = t.insertAnchor(tokensCoveringFullPrefix: xtok[0..<16], phys: 600, cvecTag: .unsteered)
        let ins2 = t.insertAnchor(tokensCoveringFullPrefix: xtok[0..<16], phys: 601, cvecTag: .unsteered)  // DUP content, new phys
        // Whatever phys the anchor now points at, demoting THAT phys must tier it.
        let mNow = t.findLongestPrefix(tokens: xtok[0..<16]) { _ in .unsteered }
        var curPhys = -1
        if mNow.pages.count == 1, case .ram(let p) = mNow.pages[0] { curPhys = p }
        let tier = (curPhys >= 0) ? t.demoteAnchor(physPage: curPhys, ssdSlot: 31, contentHash: 0xD0) : false
        let mAfter = t.findLongestPrefix(tokens: xtok[0..<16]) { _ in .unsteered }
        var ssdOk = false
        if mAfter.pages.count == 1, case .ssd = mAfter.pages[0] { ssdOk = true }
        check("H3: dup-content re-promote then demote-current-phys still tiers (ins1=\(ins1) ins2=\(ins2) phys=\(curPhys))",
              tier && ssdOk && mAfter.alignedMatchLength == 16)
    }

    // (14) Tier 1 REPRO: SHARED sub-page prefix (chat template) creates a trie
    // SPLIT, THEN demote. X and a filler share [2,105,2364,107] then diverge
    // (split at depth 4). After demoting X's pages, X must still collect as .ssd.
    // This is the path the real bridge hits that distinct-token tests miss.
    do {
        let t = RadixTrie(pageSlide: 16)
        let shared: [UInt32] = [2, 105, 2364, 107]
        let x: [UInt32] = shared + (1000..<1028).map { UInt32($0) }   // 32 tokens = 2 pages
        let f: [UInt32] = shared + (5000..<5028).map { UInt32($0) }   // 32 tokens = 2 pages
        _ = t.insertAnchor(tokensCoveringFullPrefix: x[0..<16], phys: 700, cvecTag: .unsteered)
        _ = t.insertAnchor(tokensCoveringFullPrefix: x[0..<32], phys: 701, cvecTag: .unsteered)
        _ = t.insertAnchor(tokensCoveringFullPrefix: f[0..<16], phys: 800, cvecTag: .unsteered)  // SPLIT at depth 4
        _ = t.insertAnchor(tokensCoveringFullPrefix: f[0..<32], phys: 801, cvecTag: .unsteered)
        let mPre = t.findLongestPrefix(tokens: x[0..<32]) { _ in .unsteered }
        check("shared-split: X adopts 2 RAM pages before demote (split intact)",
              mPre.pages.count == 2 && mPre.alignedMatchLength == 32)
        _ = t.demoteAnchor(physPage: 700, ssdSlot: 40, contentHash: 0xE0)
        _ = t.demoteAnchor(physPage: 701, ssdSlot: 41, contentHash: 0xE1)
        let mX = t.findLongestPrefix(tokens: x[0..<32]) { _ in .unsteered }
        var ok = (mX.pages.count == 2)
        for p in mX.pages { if case .ssd = p {} else { ok = false } }
        check("shared-split: X's 2 pages collectable as .ssd after filler split + demote",
              mX.alignedMatchLength == 32 && ok)
        // And the filler (sharing the split) is unaffected.
        let mF = t.findLongestPrefix(tokens: f[0..<32]) { _ in .unsteered }
        check("shared-split: filler still adopts its own 2 RAM pages",
              mF.pages.count == 2 && mF.alignedMatchLength == 32)
    }

    // (15) Tier 1 in-tier LRU: orphanSsdSlot drops the trie bookkeeping. Demote
    // a page to slot S, orphanSsdSlot(S) -> findLongestPrefix now GAPs that page
    // (re-prefill) AND anchorBySsdSlot[S] is gone (so a stale reload can't fire).
    do {
        let t = RadixTrie(pageSlide: 16)
        let tokens: [UInt32] = (0..<16).map { UInt32($0) }
        t.insertAnchor(tokensCoveringFullPrefix: tokens[0..<16],
                       phys: 1100, cvecTag: .unsteered)
        _ = t.demoteAnchor(physPage: 1100, ssdSlot: 7, contentHash: 0xF00D)
        // Sanity: it collects as .ssd before the orphan.
        let mPre = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        var preSsd = false
        if mPre.pages.count == 1, case .ssd(let s, _) = mPre.pages[0] { preSsd = (s == 7) }
        check("orphan: page is .ssd before orphanSsdSlot", preSsd && mPre.alignedMatchLength == 16)
        t.orphanSsdSlot(7)
        let mPost = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        check("orphan: orphanSsdSlot GAPs the page (re-prefill)",
              mPost.alignedMatchLength == 0 && mPost.pages.isEmpty)
        // A subsequent reloadAnchor on the orphaned slot is a no-op (back gone).
        t.reloadAnchor(ssdSlot: 7, intoPhys: 1101)
        let mAfterReload = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        check("orphan: anchorBySsdSlot[S] gone (stale reload is a no-op)",
              mAfterReload.alignedMatchLength == 0)
    }

    // (16) Tier 1 in-tier LRU: onSsdSlotOrphaned FIRES with the right slot when a
    // DEFENSIVE removal path (invalidateAnchorFor -> clearTagEntry) clears an
    // entry that still holds a .ssd value. The public demote ALWAYS removes the
    // phys back, so this defensive co-existence (a .ssd entry still reachable
    // from anchorByPhys) is only producible by a future removal-site edit; we
    // construct it directly with the test-only debugForceSsdEntryWithPhysBack
    // hook and assert the callback reports the correct slot AND that
    // anchorBySsdSlot is cleaned up (no double-free, no leak).
    do {
        let t = RadixTrie(pageSlide: 16)
        var fired: [Int] = []
        t.onSsdSlotOrphaned = { fired.append($0) }
        let tokens: [UInt32] = (0..<16).map { UInt32($0) }
        // Force a .ssd(slot 9) entry on phys 1200 that IS indexed by anchorByPhys.
        t.debugForceSsdEntryWithPhysBack(
            tokens: tokens[0..<16], phys: 1200, slot: 9, contentHash: 0x1234,
            cvecTag: .unsteered)
        // Defensive eviction of that phys clears the .ssd entry -> callback fires.
        t.invalidateAnchorFor(physPage: 1200)
        check("callback: defensive invalidate of a .ssd entry fires with the right slot",
              fired == [9])
        // The prefix now GAPs (re-prefill) and a stale reload is a no-op
        // (anchorBySsdSlot[9] was cleaned by clearTagEntry — no leak/double-free).
        let mGap = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        check("callback: cleared .ssd entry GAPs the prefix", mGap.alignedMatchLength == 0)
        t.reloadAnchor(ssdSlot: 9, intoPhys: 1201)
        let mStill = t.findLongestPrefix(tokens: tokens[0..<16]) { _ in .unsteered }
        check("callback: anchorBySsdSlot cleaned (stale reload no-op)",
              mStill.alignedMatchLength == 0)
    }

    // (16b) orphanSsdSlot does NOT fire onSsdSlotOrphaned — the engine's
    // evict-retry calls store.freeSlot directly for the orphaned slot, so firing
    // here too would DOUBLE-FREE. Verify the no-fire invariant explicitly.
    do {
        let t = RadixTrie(pageSlide: 16)
        var fired: [Int] = []
        t.onSsdSlotOrphaned = { fired.append($0) }
        let toks: [UInt32] = (0..<16).map { UInt32($0) }
        t.insertAnchor(tokensCoveringFullPrefix: toks[0..<16], phys: 1300, cvecTag: .unsteered)
        _ = t.demoteAnchor(physPage: 1300, ssdSlot: 12, contentHash: 0x9ABC)
        t.orphanSsdSlot(12)
        check("callback: orphanSsdSlot does NOT fire (engine owns freeSlot, no double-free)",
              fired.isEmpty)
    }

    // (17) Tier 1 in-tier LRU BOUND: after K demotes + K orphans, NO .ssd anchor
    // and NO anchorBySsdSlot entry leaks. We probe leakage indirectly: every
    // orphaned page must GAP (re-prefill) on lookup, and a fresh demote into a
    // recycled slot must still be collectable as .ssd (proving the trie's
    // .ssd-anchor set stays bounded == live demotes, never accumulating dead
    // entries).
    do {
        let t = RadixTrie(pageSlide: 16)
        let K = 8
        // K distinct single-page prefixes, each demoted to its own slot.
        for i in 0..<K {
            let base = UInt32(2000 + i * 16)
            let toks: [UInt32] = (0..<16).map { base + UInt32($0) }
            t.insertAnchor(tokensCoveringFullPrefix: toks[0..<16],
                           phys: 1400 + i, cvecTag: .unsteered)
            _ = t.demoteAnchor(physPage: 1400 + i, ssdSlot: 50 + i, contentHash: UInt64(i))
        }
        // Orphan every slot (simulating the LRU evicting them all over time).
        for i in 0..<K { t.orphanSsdSlot(50 + i) }
        // Every orphaned prefix must now GAP (no dead .ssd anchor lingers).
        var allGapped = true
        for i in 0..<K {
            let base = UInt32(2000 + i * 16)
            let toks: [UInt32] = (0..<16).map { base + UInt32($0) }
            let m = t.findLongestPrefix(tokens: toks[0..<16]) { _ in .unsteered }
            if m.alignedMatchLength != 0 { allGapped = false }
        }
        check("bound: all K orphaned .ssd anchors GAP (no leak)", allGapped)
        // A subsequent reload on any orphaned slot is a no-op (anchorBySsdSlot
        // entries all gone -> bound on the .ssd-slot set is exactly live demotes).
        for i in 0..<K { t.reloadAnchor(ssdSlot: 50 + i, intoPhys: 1500 + i) }
        var stillGapped = true
        for i in 0..<K {
            let base = UInt32(2000 + i * 16)
            let toks: [UInt32] = (0..<16).map { base + UInt32($0) }
            let m = t.findLongestPrefix(tokens: toks[0..<16]) { _ in .unsteered }
            if m.alignedMatchLength != 0 { stillGapped = false }
        }
        check("bound: orphaned anchorBySsdSlot entries gone (stale reloads no-op)", stillGapped)
    }

    print("RadixTrie tests: \(passed) passed, \(failed) failed")
    if failed > 0 {
        exit(1)
    }
}
