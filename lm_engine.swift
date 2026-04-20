// Multi-session batched-decode engine.
//
// Problem it solves: agent scenarios where a single logical user has
// several concurrent conversations in flight — each one independently
// blocked or unblocked by tool-call returns from different API servers
// answering at different speeds. A classic single-session inference loop
// would force these to serialise; this engine interleaves them into one
// AR step per scheduler tick, reusing the weight stream across all
// ready sessions for free (batched GEMV is the whole point of B>1).
//
// Architecture:
//   - Up to B sessions concurrent, each bound to a fixed slot in [0, B).
//   - Each session owns a disjoint strip of the paged KV cache; sessions
//     cannot contaminate each other's history.
//   - One `engine.step()` call emits one `buildStepCB` covering every
//     busy slot. Per slot, each session decides its own:
//         input_tokens[slot] — next token to feed (prompt, tool result,
//                              or previously-sampled token for generating)
//         positions[slot]    — absolute position this step writes at
//         k_len_*[slot]      — k_len including the row just written
//     Idle slots (no active session) run a no-op single-position forward
//     against their own dedicated pages; their outputs are ignored. This
//     is wasted work equal to `(B - active) / B` of each CB — acceptable
//     until we add per-slot "skip" gating in the kernels.
//
// State machine per session:
//   idle → (submit tokens) → priming → (queue drains) → generating
//                                                       ↓ (EOS / maxTokens)
//                                                       done
//   Back-edges: generating → priming is valid (tool return arrives mid-
//   generation; we just append the tool-result tokens to the priming
//   queue and the next step resumes teacher-forcing).
//
// API shape intentionally mirrors llama.cpp's surface (open_session /
// submit / step / next_token / close_session), translated to Swift,
// without adopting llama.cpp's internal data model.
import Foundation
import Metal

enum SessionState: Equatable {
    case idle           // no pending work — e.g. session just opened, nothing submitted
    case priming        // chunkQueue non-empty — teacher-forcing prompt/tool tokens
    case generating     // sampling; pushing to outputQueue
    case paused         // explicit pause (caller is waiting for tool result); KV retained
    case done           // EOS / maxTokens / caller closed

    // Does this session want a slot on the next scheduler tick?
    var wantsSlot: Bool {
        switch self {
        case .priming, .generating: return true
        default: return false
        }
    }
    // Legacy alias — some code paths still read .isBusy.
    var isBusy: Bool { return wantsSlot }
}

// A chunk of work queued into a session's priming lane. Text and image
// chunks travel together in one queue so the scheduler can preserve the
// caller's interleaving: "user: [text prefix] [image] [text suffix]"
// becomes three chunks processed in order. The engine turns each chunk
// into the right prefill/AR dispatch.
enum PrimingChunk {
    // Plain text tokens — go through embed_lookup + the full prefill pipe.
    case tokens([UInt32])

    // Pre-embedded image soft tokens produced by the vision tower. Layout
    // is [count, HIDDEN] (row-major) and the storage dtype is either fp16
    // or fp32. Prefill copies these into pre_hidden and skips embed_lookup
    // + embed-scale (the vision projection already did both).
    // `byteOffset` lets a chunk point at a sub-range of the same buffer,
    // which is how multi-tile soft-token prefills walk through a big
    // image one MAX_Q_LEN-sized chunk per tick.
    case softTokens(buffer: MTLBuffer, count: Int, isFp32: Bool, byteOffset: Int)

    var count: Int {
        switch self {
        case .tokens(let ts): return ts.count
        case .softTokens(_, let c, _, _): return c
        }
    }
}

final class Session {
    let id: Int
    // Which active-batch slot this session is currently occupying, or nil
    // when the session is *resident* (KV pages retained, block_table not
    // populated) but not in the active batch. The scheduler moves sessions
    // between resident-without-slot and resident-with-slot each tick.
    var slot: Int?
    let eosId: UInt32
    var maxNewTokens: Int
    fileprivate weak var engine: LmEngine?

    fileprivate(set) var state: SessionState = .idle
    // Phys-page IDs owned by this session, in logical-page order:
    //   ownedPages[p] = phys page number for this session's page p
    //   ownedPages.count * PAGE_SLIDE bounds the max k_len this session
    //   can reach without growing. Pages are allocated on-demand as the
    //   session's position advances (see growPagesFor(kLen:)).
    fileprivate var ownedPages: [Int] = []
    // Full token history for per-page prefix hashing. Append-only —
    // submit() extends it, and hash(consumedTokens[0..(P+1)*PAGE]) is
    // the content identity of logical page P. Used for:
    //   (a) cache probe at first-submit: find shared pages in the global
    //       PageManager content index and adopt them read-only
    //   (b) post-prefill promotion: announce newly-written pages to the
    //       content index so the NEXT session with the same prefix hits.
    fileprivate var consumedTokens: [UInt32] = []
    // Logical pages already promoted to PageManager.contentIndex. Kept
    // so we don't re-promote on every prefill tile commit.
    fileprivate var promotedPageCount: Int = 0
    // Ordered chunks to teacher-force (text tokens or image soft tokens).
    fileprivate var chunkQueue: [PrimingChunk] = []
    // When state == .generating, the last-sampled token becomes the next
    // step's input; kept separate from the chunk queue for state clarity.
    fileprivate var nextGeneratedInput: UInt32 = 0
    // Next KV-cache write position. k_len after a step == position + 1.
    fileprivate var position: Int = 0
    fileprivate var numGenerated: Int = 0

    // Tokens the caller can consume.
    fileprivate var outputQueue: [UInt32] = []

    fileprivate init(id: Int, eosId: UInt32, maxNewTokens: Int, engine: LmEngine) {
        self.id = id; self.slot = nil
        self.eosId = eosId; self.maxNewTokens = maxNewTokens
        self.engine = engine
    }

    // Queue more input tokens to be teacher-forced. Valid in any state:
    // calling while .generating/.paused/.idle flips back to .priming.
    //
    // At the *first* submit (session still at position=0, no owned pages),
    // we probe the PageManager's content index for cache hits on the
    // leading pages. Any hit is adopted read-only — ownedPages gets the
    // shared phys page, position advances by PAGE_SLIDE, and the queued
    // chunk only covers the un-cached tail. This is what makes multiple
    // sessions with the same system prompt / same image prefix skip the
    // redundant prefill work.
    //
    // Subsequent submits (tool-call returns, continuations) don't probe —
    // they always prefill fresh, since mid-conversation tails are unique.
    func submit(_ tokens: [UInt32]) {
        guard !tokens.isEmpty else { return }
        guard let eng = engine else {
            chunkQueue.append(.tokens(tokens))
            if state != .done { state = .priming }
            return
        }
        // Extend the canonical history (used for hashing).
        consumedTokens.append(contentsOf: tokens)
        // First-submit cache probe.
        let firstSubmit = (position == 0 && ownedPages.isEmpty)
        var skipPrefix = 0
        if firstSubmit {
            skipPrefix = adoptSharedPrefixPages(engine: eng)
            position = skipPrefix * PAGE_SLIDE
        }
        // Queue the un-cached tail for prefill.
        let tailStart = skipPrefix * PAGE_SLIDE
        if tailStart < tokens.count {
            // tokens held by THIS submit call; cached pages came from the
            // head of this same tokens array (or the very-start of
            // consumedTokens, which at first-submit is identical).
            let tail = Array(tokens[tailStart...])
            if !tail.isEmpty { chunkQueue.append(.tokens(tail)) }
        }
        if state != .done { state = .priming }
    }
    func submit(text: String, addBos: Bool? = nil) {
        guard let eng = engine else { return }
        submit(eng.tokenizer.encode(text, addBos: addBos))
    }

    // Explicit KV-page sharing: borrow this session's first `pageCount`
    // pages from `source` and install them as our own read-only prefix.
    // Unlike content-hash auto-sharing (which needs token-level hashable
    // prefixes), adoptKvFrom operates on phys pages directly — so it
    // works for ANY kind of prefix, including image soft tokens that
    // aren't easily fingerprinted from their fp16 content.
    //
    // Usage (the "same image, multiple suffixes" pattern):
    //   let base = engine.openSession(); base.submit(prefix + image)
    //   while base.state == .priming { engine.tick() }
    //   let pagesToShare = base.position / PAGE_SLIDE       // full pages only
    //   for query in queries {
    //       let s = engine.openSession()
    //       s.adoptKvFrom(base, pageCount: pagesToShare)
    //       s.submit(query)
    //   }
    //   while engine.hasWork { engine.tick() }              // all concurrent
    //
    // Preconditions: this session must be fresh (position=0, no owned
    // pages yet) and `pageCount` must not exceed source's current owned
    // page count. Fails silently (returns false) otherwise.
    @discardableResult
    func adoptKvFrom(_ source: Session, pageCount: Int) -> Bool {
        guard position == 0 && ownedPages.isEmpty else { return false }
        guard pageCount >= 0, pageCount <= source.ownedPages.count else { return false }
        guard let eng = engine else { return false }
        // Share each of source's leading phys pages. shareExisting handles
        // the release-from-freelist path if source has since closed but
        // the page's content hasn't been overwritten yet.
        for p in 0..<pageCount {
            let phys = source.ownedPages[p]
            eng.pageManager.shareExisting(physPage: phys, sessionId: id)
            ownedPages.append(phys)
        }
        // Advance our position past the shared range. The next submit's
        // tokens will prefill starting at this position.
        position = pageCount * PAGE_SLIDE
        // Copy source's consumedTokens over the shared range so that
        // post-prefill promotion for our future pages uses a prefix hash
        // that stays consistent with source's (i.e. a THIRD session
        // sharing BOTH of us gets a consistent cache hit).
        let tokensToCopy = min(source.consumedTokens.count, pageCount * PAGE_SLIDE)
        if tokensToCopy > 0 {
            consumedTokens.append(contentsOf: source.consumedTokens[0..<tokensToCopy])
        }
        // Adopted pages were already content-indexed by source's post-
        // prefill promotion; skip re-promoting them.
        promotedPageCount = pageCount
        // A session that adopted pages is priming-ready: it has KV for
        // positions [0, pageCount*PAGE_SLIDE) but no queued chunks yet.
        // The first submit() will queue the divergent suffix.
        return true
    }

    // Walk the PageManager's content index for leading pages of
    // consumedTokens. Returns the number of pages successfully adopted.
    // Idempotent-ish: safe to call, just returns 0 if none match.
    private func adoptSharedPrefixPages(engine: LmEngine) -> Int {
        var adopted = 0
        while (adopted + 1) * PAGE_SLIDE <= consumedTokens.count {
            let end = (adopted + 1) * PAGE_SLIDE
            let prefixHash = PageManager.hashPage(consumedTokens[0..<end])
            guard let phys = engine.pageManager.findByHash(prefixHash) else { break }
            engine.pageManager.shareExisting(physPage: phys, sessionId: id)
            ownedPages.append(phys)
            adopted += 1
        }
        // Adopted pages are already content-indexed — don't re-promote.
        if adopted > 0 {
            promotedPageCount = adopted
            if ProcessInfo.processInfo.environment["LM_CACHE_DEBUG"] != nil {
                print("  [cache] session \(id) adopted \(adopted) shared prefix pages "
                      + "(= \(adopted * PAGE_SLIDE) tokens cached)")
            }
        }
        return adopted
    }
    func submit(softTokens: MTLBuffer, count: Int, isFp32: Bool) {
        guard count > 0 else { return }
        chunkQueue.append(.softTokens(buffer: softTokens, count: count,
                                       isFp32: isFp32, byteOffset: 0))
        if state != .done { state = .priming }
    }

    // Explicit pause — retain KV, release the active slot. Caller uses this
    // while waiting for a tool call to complete from an external API; the
    // subsequent submit() of the tool-result tokens flips back to .priming
    // and re-admits the session on the next scheduler tick.
    func pause() { if state != .done { state = .paused } }

    // Pull the next generated token, or nil if none ready.
    func nextToken() -> UInt32? {
        guard !outputQueue.isEmpty else { return nil }
        return outputQueue.removeFirst()
    }

    var pendingOutputCount: Int { outputQueue.count }
    var pendingPrimingCount: Int { chunkQueue.reduce(0) { $0 + $1.count } }

    // Mark as done; engine will release pages + slot on the next tick.
    func finish() { state = .done }
}

final class LmEngine {
    let weights: LmWeights
    let tokenizer: GemmaBpe
    // All resident sessions (keyed by id). A session is "resident" if the
    // engine holds its KV pages; it may or may not currently own an active
    // batch slot. Limited to MAX_RESIDENT_SESSIONS — beyond that, callers
    // queue externally (a separate admission layer handles that).
    private(set) var residentSessions: [Int: Session] = [:]
    // Active-batch slot table: slotAssignment[s] is a session id or nil.
    // Decoupled from residentSessions — sessions move in and out of slots
    // each tick based on `wantsSlot` state.
    private var slotAssignment: [Int?]
    private var nextId: Int = 1

    // Round-robin cursor for slot admission — avoids sticky-bias toward
    // low-id sessions when more ready sessions exist than free slots.
    private var admissionCursor: Int = 0

    // Page manager owns the KV cache's physical page pool.
    let pageManager: PageManager

    // Instrumentation — helps the scheduler-behaviour tests measure how
    // well we're batching. One CB per step regardless of active count.
    private(set) var totalSteps: Int = 0
    private(set) var totalTokensGenerated: Int = 0
    private(set) var lastStepMs: Double = 0

    init(weights: LmWeights) {
        self.weights = weights
        self.tokenizer = GemmaBpe(weights: weights)
        self.slotAssignment = Array(repeating: nil, count: B)
        self.pageManager = PageManager(numPhysPages: SCRATCH_PAGE_BASE,
                                        pageSize: PAGE_SLIDE)
    }

    // Grow a session's owned pages so its logical page count covers k_len.
    // Called right before admission to a slot and between steps when the
    // session's position advances past its current allocation.
    // Fails if the page pool is exhausted — caller must close a session or
    // evict another resident to make room.
    fileprivate func ensurePages(_ s: Session, forKLen kLen: Int) -> Bool {
        let needed = (kLen + PAGE_SLIDE - 1) / PAGE_SLIDE
        while s.ownedPages.count < needed {
            do {
                let p = try pageManager.allocFresh(sessionId: s.id)
                s.ownedPages.append(p)
            } catch {
                print("  ensurePages: pool exhausted for session \(s.id) at logical page \(s.ownedPages.count)")
                return false
            }
        }
        return true
    }

    // Install a session's owned pages into block_table[slot][:]. Entries
    // past ownedPages.count get filled with SCRATCH_PAGE_BASE so that any
    // accidental read past num_pages lands in scratch (detectable garbage)
    // rather than aliasing with the session's real pages. Kernels should
    // only read [0..num_pages-1] but the safety net matters when we're
    // growing pages lazily.
    fileprivate func installBlockTable(_ s: Session, slot: Int) {
        let btP = block_table.contents().bindMemory(to: UInt32.self,
                    capacity: B * MAX_PAGES_PER_SLOT)
        let base = slot * MAX_PAGES_PER_SLOT
        let scratchGuard = UInt32(SCRATCH_PAGE_BASE)
        for p in 0..<MAX_PAGES_PER_SLOT {
            btP[base + p] = p < s.ownedPages.count ? UInt32(s.ownedPages[p]) : scratchGuard
        }
    }

    // Open a new session on the first free slot. Returns nil if the engine
    // is at capacity (B active sessions) OR the page pool is exhausted.
    // Caller owns the Session and must call `closeSession` to free both
    // the slot and the allocated pages.
    // Open a new resident session. Pages are claimed on-demand as the
    // session accumulates KV; no up-front allocation. Slot is assigned by
    // the scheduler on the next tick() call when the session is ready.
    func openSession(eosId: UInt32? = nil, maxNewTokens: Int = 128) -> Session? {
        guard residentSessions.count < MAX_RESIDENT_SESSIONS else {
            print("  openSession: residency cap \(MAX_RESIDENT_SESSIONS) reached")
            return nil
        }
        let sessionId = nextId; nextId += 1
        let s = Session(id: sessionId,
                        eosId: eosId ?? weights.eosTokenId,
                        maxNewTokens: maxNewTokens, engine: self)
        residentSessions[sessionId] = s
        return s
    }

    // Close: release slot (if any), return pages, drop from residency.
    func closeSession(_ s: Session) {
        s.state = .done
        if let slot = s.slot {
            slotAssignment[slot] = nil
            s.slot = nil
        }
        pageManager.releaseAllForSession(s.id)
        s.ownedPages.removeAll()
        s.consumedTokens.removeAll()
        s.promotedPageCount = 0
        residentSessions.removeValue(forKey: s.id)
    }

    // After a prefill tile commits, walk any fully-written logical pages
    // that haven't been promoted yet and publish them to the content
    // index so the next session with the same prefix can findByHash.
    // Called at the end of stepPrefillForSession.
    fileprivate func promoteFinishedPages(_ s: Session) {
        let fullyWritten = s.position / PAGE_SLIDE
        while s.promotedPageCount < fullyWritten {
            let p = s.promotedPageCount
            // We need the first (p+1)*PAGE_SLIDE tokens of this session's
            // submitted history to form the page's prefix hash. If the
            // session's consumedTokens hasn't caught up (unusual — happens
            // only when a submit staged tokens in chunkQueue that were
            // consumed without being added to consumedTokens), skip.
            let end = (p + 1) * PAGE_SLIDE
            guard end <= s.consumedTokens.count, p < s.ownedPages.count else {
                break
            }
            let hash = PageManager.hashPage(s.consumedTokens[0..<end])
            pageManager.promoteToShared(physPage: s.ownedPages[p], contentHash: hash)
            s.promotedPageCount += 1
        }
    }

    func poolStats() -> PageManager.Stats { return pageManager.stats() }

    // Longest common prefix (in pages) across ALL currently-slotted busy
    // sessions. Used by the AR scheduler to decide whether to route
    // attention through the K/V-broadcast shared+tail+reduce path. Zero
    // when there's only one active session (no broadcast benefit) or
    // when slots' block_tables disagree at page 0.
    func detectSharedPrefix() -> Int {
        let busy = activeSessions.filter { $0.state.wantsSlot }
        if busy.count < 2 { return 0 }
        let minPages = busy.map { $0.ownedPages.count }.min() ?? 0
        if minPages == 0 { return 0 }
        var p = 0
        while p < minPages {
            let first = busy[0].ownedPages[p]
            if busy.dropFirst().allSatisfy({ $0.ownedPages[p] == first }) {
                p += 1
            } else { break }
        }
        return p
    }

    // Sessions currently occupying active batch slots (length ≤ B).
    var activeSessions: [Session] {
        return slotAssignment.compactMap { $0.flatMap { residentSessions[$0] } }
    }
    // Resident sessions that want a slot (priming or generating).
    var readyResidents: [Session] {
        return residentSessions.values.filter { $0.state.wantsSlot }
    }
    var hasWork: Bool { readyResidents.count > 0 }

    // Admission pass: evict slots whose session no longer wants one, admit
    // ready-but-unslotted residents into free slots round-robin. Grows
    // owned pages + installs block_table entries for freshly-admitted sessions.
    private func runAdmissionPass() {
        for slot in 0..<B {
            if let sid = slotAssignment[slot],
               let s = residentSessions[sid], !s.state.wantsSlot {
                s.slot = nil
                slotAssignment[slot] = nil
            }
        }
        let ready = readyResidents.filter { $0.slot == nil }.sorted { $0.id < $1.id }
        if ready.isEmpty { return }
        var cursor = admissionCursor % ready.count
        for slot in 0..<B where slotAssignment[slot] == nil {
            let pick = ready[cursor % ready.count]
            if pick.slot != nil { cursor += 1; continue }
            pick.slot = slot
            slotAssignment[slot] = pick.id
            // Pre-grow pages. Use the max of (current prefill/gen tail) and a
            // baseline window so short-prompt runs don't keep re-growing one
            // page at a time during AR priming. 1024 tokens (= 64 slide pages)
            // is a reasonable floor — cheap relative to the pool, and matches
            // the size of a typical chat turn's KV residency.
            let pendingPrime = pick.pendingPrimingCount
            let lookahead = max(pick.position + pendingPrime + pick.maxNewTokens + 8, 1024)
            _ = ensurePages(pick, forKLen: lookahead)
            installBlockTable(pick, slot: slot)
            cursor += 1
        }
        admissionCursor = (admissionCursor + 1) % max(ready.count, 1)
    }

    // Take one token off the head chunk if it's a .tokens chunk. Returns
    // nil if the head is .softTokens (that chunk needs fast prefill, not
    // AR priming) or the queue is empty. Also removes emptied chunks and
    // updates session state.
    private func popArPrimingToken(_ s: Session) -> UInt32? {
        while let head = s.chunkQueue.first {
            switch head {
            case .tokens(var ts):
                if ts.isEmpty {
                    s.chunkQueue.removeFirst(); continue
                }
                let t = ts.removeFirst()
                if ts.isEmpty { s.chunkQueue.removeFirst() }
                else { s.chunkQueue[0] = .tokens(ts) }
                return t
            case .softTokens:
                // Head is image soft tokens — can't consume via AR. Caller
                // should use the fast-prefill path for this chunk.
                return nil
            }
        }
        return nil
    }

    // True if this session has a pending chunk that must go through fast
    // prefill (soft-tokens OR a .tokens chunk of size ≥ 2 for efficiency).
    // Used by the scheduler to decide when to run single-slot prefill.
    private func hasPrefillChunk(_ s: Session, minTokensThreshold: Int = 2) -> Bool {
        guard let head = s.chunkQueue.first else { return false }
        switch head {
        case .tokens(let ts): return ts.count >= minTokensThreshold
        case .softTokens:    return true
        }
    }

    // Run exactly one buildStepCB covering every slot, with per-slot state
    // driven by each session's queue. Returns the number of tokens emitted
    // into output queues this step (across all sessions).
    //
    // Sessions whose next chunk is .softTokens get parked (their slot runs
    // a no-op forward this step). The caller must invoke `tick()` — which
    // routes those sessions through fast prefill — rather than calling
    // `step()` directly in the multimodal case.
    @discardableResult
    func step() -> Int {
        runAdmissionPass()
        let busy = activeSessions.filter { $0.state.isBusy }
        if busy.isEmpty { return 0 }

        // Per-slot inputs (AR path).
        let tokP = input_tokens.contents().bindMemory(to: UInt32.self, capacity: B)
        let posP = positions.contents().bindMemory(to: UInt32.self, capacity: B)
        let klsP = k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let klfP = k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
        let npsP = num_pages_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let npfP = num_pages_full.contents().bindMemory(to: UInt32.self, capacity: B)

        // Track which slots run REAL work this step.
        var realSlot = [Bool](repeating: false, count: B)

        for slot in 0..<B {
            if let sid = slotAssignment[slot],
               let s = residentSessions[sid], s.state.isBusy {
                // Grow pages if needed for the step that's about to run.
                _ = ensurePages(s, forKLen: s.position + 2)
                installBlockTable(s, slot: slot)

                let inputTok: UInt32?
                if s.state == .priming {
                    inputTok = popArPrimingToken(s)
                } else {
                    inputTok = s.nextGeneratedInput
                }
                if let tok = inputTok {
                    tokP[slot] = tok
                    posP[slot] = UInt32(s.position)
                    let kLen = s.position + 1
                    klsP[slot] = UInt32(kLen); klfP[slot] = UInt32(kLen)
                    npsP[slot] = UInt32((kLen + PAGE_SLIDE - 1) / PAGE_SLIDE)
                    npfP[slot] = UInt32((kLen + PAGE_FULL  - 1) / PAGE_FULL)
                    realSlot[slot] = true
                    continue
                }
            }
            // Park: BOS at position 0, k_len=1, 1 page. The park slot writes
            // to whatever phys page is installed in block_table[slot][0] —
            // which for an unassigned slot is the scratch strip via the
            // guard-page fallback in installBlockTable. Safe no-op.
            tokP[slot] = weights.bosTokenId
            posP[slot] = 0
            klsP[slot] = 1; klfP[slot] = 1
            npsP[slot] = 1; npfP[slot] = 1
            if slotAssignment[slot] == nil {
                // No session here; redirect block_table[slot][0] to scratch
                // so this park step's KV write can't corrupt someone else.
                let btP = block_table.contents().bindMemory(to: UInt32.self,
                            capacity: B * MAX_PAGES_PER_SLOT)
                btP[slot * MAX_PAGES_PER_SLOT + 0] = UInt32(SCRATCH_PAGE_BASE)
            }
        }

        if USE_FLEX_ATTN {
            precomputeFlexBlockMaskSlide(slidingWindow: SLIDING_WINDOW)
            precomputeFlexBlockMaskFull()
        }

        // Detect cross-slot shared-prefix length and populate the shared
        // phys-pages buffer. Below threshold, buildStepCB's default path
        // runs (fast at low batch, fast at no-sharing).
        let sharedPrefix = detectSharedPrefix()
        if sharedPrefix >= SHARED_PREFIX_THRESHOLD_PAGES,
           let reference = activeSessions.filter({ $0.state.wantsSlot }).first {
            let spp = shared_phys_pages.contents().assumingMemoryBound(to: UInt32.self)
            for p in 0..<sharedPrefix {
                spp[p] = UInt32(reference.ownedPages[p])
            }
        }

        let t0 = Date()
        let cb = buildStepCB(weights, sharedPrefixPages: sharedPrefix)
        cb.commit(); cb.waitUntilCompleted()
        lastStepMs = Date().timeIntervalSince(t0) * 1000
        totalSteps += 1
        if let err = cb.error { print("  GPU step error: \(err)"); return 0 }

        let logP = logits.contents().assumingMemoryBound(to: Float16.self)
        var emitted = 0
        for slot in 0..<B where realSlot[slot] {
            guard let sid = slotAssignment[slot],
                  let s = residentSessions[sid], s.state.isBusy else { continue }
            let base = slot * VOCAB
            var bestI = 0; var bestV: Float = -.infinity
            for v in 0..<VOCAB {
                let x = Float(logP[base + v])
                if x > bestV { bestV = x; bestI = v }
            }
            let sampled = UInt32(bestI)
            s.position += 1

            if s.state == .priming {
                // Drained ALL chunks? The logit we just computed is the
                // first generated token's prediction. Flip to .generating,
                // emit, check EOS.
                if s.chunkQueue.isEmpty {
                    s.state = .generating
                    s.outputQueue.append(sampled)
                    s.consumedTokens.append(sampled)
                    s.nextGeneratedInput = sampled
                    s.numGenerated += 1; emitted += 1; totalTokensGenerated += 1
                    if sampled == s.eosId || s.numGenerated >= s.maxNewTokens {
                        s.state = .done
                    }
                }
                // else: more priming to do — discard this logit.
            } else {
                s.outputQueue.append(sampled)
                s.nextGeneratedInput = sampled
                s.numGenerated += 1; emitted += 1; totalTokensGenerated += 1
                if sampled == s.eosId || s.numGenerated >= s.maxNewTokens {
                    s.state = .done
                }
            }
        }
        return emitted
    }

    // Run a single-slot fast prefill: the given session's next chunk is
    // dispatched as a proper buildPrefillCB filling only its slot. Other
    // slots are silenced via block_table redirect to the scratch strip.
    // Returns true if a prefill actually ran (chunk was consumed).
    @discardableResult
    func stepPrefillForSession(_ s: Session) -> Bool {
        guard s.state == .priming, let head = s.chunkQueue.first else { return false }
        // Ensure the session owns an active slot. If not, run admission first.
        if s.slot == nil { runAdmissionPass() }
        guard let sslot = s.slot else { return false }
        let qLen = head.count
        precondition(qLen >= 1)
        let thisTile = min(qLen, MAX_Q_LEN)
        let remaining = qLen - thisTile
        let positionStart = s.position

        // Ensure pages cover positionStart..positionStart+thisTile.
        if !ensurePages(s, forKLen: positionStart + thisTile + 1) { return false }
        installBlockTable(s, slot: sslot)

        // --- Save block_table; redirect non-target slots to scratch strip ---
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        var savedBT = [UInt32](repeating: 0, count: B * MAX_PAGES_PER_SLOT)
        for i in 0..<(B * MAX_PAGES_PER_SLOT) { savedBT[i] = btP[i] }
        for slot in 0..<B where slot != sslot {
            // All silenced slots redirect every logical page to the scratch
            // strip. Silenced slots write garbage that gets discarded; same
            // scratch pages serve all silenced slots (writes race, reads
            // are ignored). Wrapping via % keeps us inside the strip.
            for p in 0..<MAX_PAGES_PER_SLOT {
                btP[slot * MAX_PAGES_PER_SLOT + p] =
                    UInt32(SCRATCH_PAGE_BASE + (p % SCRATCH_STRIP))
            }
        }
        defer {
            for i in 0..<(B * MAX_PAGES_PER_SLOT) { btP[i] = savedBT[i] }
        }

        // --- Populate prefill scratch for all B slots, but only s.slot
        // carries real data. The silenced slots get meaningless filler that
        // doesn't matter (their outputs hit scratch pages).
        let tokP = pre_input_tokens.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
        let posP = pre_q_positions.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
        for b in 0..<B {
            for i in 0..<thisTile {
                posP[b * thisTile + i] = UInt32(positionStart + i)
                tokP[b * thisTile + i] = weights.bosTokenId  // silenced-slot filler
            }
        }
        let klsP = pre_k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let klfP = pre_k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
        for b in 0..<B {
            klsP[b] = UInt32(positionStart + thisTile)
            klfP[b] = UInt32(positionStart + thisTile)
        }

        // --- Chunk-specific setup ---
        var skipEmbed = false
        switch head {
        case .tokens(let ts):
            // Real text tokens go in s.slot's row.
            for i in 0..<thisTile {
                tokP[sslot * thisTile + i] = ts[i]
            }
        case let .softTokens(buf, _, isFp32, byteOffset):
            // Vision tower output. Copy the thisTile-row window starting
            // at `byteOffset` into pre_hidden at rows [s.slot*thisTile,
            // s.slot*thisTile+thisTile), downcasting fp32→fp16 if needed.
            // Other slots' rows are untouched (they'll compute junk that
            // gets discarded via block_table redirect to scratch).
            let dstPtr = pre_hidden.contents().assumingMemoryBound(to: Float16.self)
            let dstBase = (sslot * thisTile) * HIDDEN
            let srcRaw = buf.contents().advanced(by: byteOffset)
            if isFp32 {
                let srcPtr = srcRaw.assumingMemoryBound(to: Float.self)
                if ProcessInfo.processInfo.environment["LM_MM_DEBUG"] != nil && byteOffset == 0 {
                    var mn: Float = .infinity, mx: Float = -.infinity, sumAbs: Float = 0
                    for i in 0..<(thisTile * HIDDEN) {
                        let v = srcPtr[i]
                        if v < mn { mn = v }; if v > mx { mx = v }
                        sumAbs += abs(v)
                    }
                    print(String(format: "  [softTokens tile0] min=%.3f max=%.3f mean|v|=%.3f (first-tile, fp32)",
                                 mn, mx, sumAbs / Float(thisTile * HIDDEN)))
                }
                for i in 0..<(thisTile * HIDDEN) {
                    dstPtr[dstBase + i] = Float16(srcPtr[i])
                }
            } else {
                memcpy(dstPtr.advanced(by: dstBase), srcRaw, thisTile * HIDDEN * 2)
            }
            skipEmbed = true
        }

        precomputeFlexPrefillMasks(qLen: thisTile, positionStart: positionStart)
        let t0 = Date()
        let cb = buildPrefillCB(weights, qLen: thisTile, skipEmbed: skipEmbed)
        cb.commit(); cb.waitUntilCompleted()
        lastStepMs = Date().timeIntervalSince(t0) * 1000
        totalSteps += 1
        if let err = cb.error { print("  GPU prefill error: \(err)"); return false }

        // --- Advance session state by thisTile positions. Copy slot-s's
        // last-position logit into the AR `logits` buffer so that the next
        // .step() or sampling sees a coherent post-prefill state.
        s.position += thisTile
        // Publish fully-written pages to the global content index so a
        // later session that submits the same prefix will findByHash this
        // page and share it read-only.
        promoteFinishedPages(s)
        let srcPtr = pre_logits.contents().assumingMemoryBound(to: Float16.self)
        let dstPtr = logits.contents().assumingMemoryBound(to: Float16.self)
        let src = srcPtr.advanced(by: (sslot * thisTile + (thisTile - 1)) * VOCAB)
        let dst = dstPtr.advanced(by: sslot * VOCAB)
        memcpy(dst, src, VOCAB * 2)

        // Pop / trim the head chunk.
        switch head {
        case .tokens(var ts):
            ts.removeFirst(thisTile)
            if ts.isEmpty { s.chunkQueue.removeFirst() }
            else          { s.chunkQueue[0] = .tokens(ts) }
        case let .softTokens(buf, _, isFp32, byteOffset):
            if remaining == 0 {
                s.chunkQueue.removeFirst()
            } else {
                // Leave the chunk in the queue with its offset advanced
                // by thisTile rows; next tick picks up where we left off.
                let bpe = isFp32 ? 4 : 2
                let newOffset = byteOffset + thisTile * HIDDEN * bpe
                s.chunkQueue[0] = .softTokens(buffer: buf, count: remaining,
                                               isFp32: isFp32, byteOffset: newOffset)
            }
        }

        // If the queue is now empty, sample the post-prefill logit as the
        // first generated token so callers don't see a stall step.
        if s.chunkQueue.isEmpty {
            let base = sslot * VOCAB
            let logP = logits.contents().assumingMemoryBound(to: Float16.self)
            var bestI = 0; var bestV: Float = -.infinity
            for v in 0..<VOCAB {
                let x = Float(logP[base + v])
                if x > bestV { bestV = x; bestI = v }
            }
            let sampled = UInt32(bestI)
            s.state = .generating
            s.outputQueue.append(sampled)
            s.consumedTokens.append(sampled)
            s.nextGeneratedInput = sampled
            s.numGenerated += 1; totalTokensGenerated += 1
            if sampled == s.eosId || s.numGenerated >= s.maxNewTokens {
                s.state = .done
            }
        }
        return true
    }

    // Unified scheduler tick. Picks between fast prefill and AR batch each
    // call. Policy for v1 — mostly AR-batch, use prefill only when it's
    // strictly better or strictly required:
    //
    //   1. Any session has a .softTokens head chunk  →  single-slot fast
    //      prefill for that session (image tokens can't go through AR;
    //      paused AR work for other sessions is unavoidable here).
    //   2. Exactly one session busy AND its head is tokens-chunk ≥ 2  →
    //      fast prefill (no AR-batch advantage to forfeit since it's the
    //      only busy slot anyway).
    //   3. Otherwise  →  AR step across all busy sessions. Even for many
    //      long prompts, parallel AR-prime beats serial fast prefill:
    //      prefill wastes (B-1)/B of each CB, AR doesn't.
    //
    // Returns tokens emitted this tick (0 during prefill unless the chunk
    // drained and we sampled the first generated token).
    @discardableResult
    func tick() -> Int {
        // Run admission first so we don't decide "no busy slots" while ready
        // residents are still unslotted (→ infinite spin in the scheduler loop).
        runAdmissionPass()
        let busy = activeSessions.filter { $0.state.isBusy }
        if busy.isEmpty { return 0 }
        // 1. Soft-tokens forced prefill.
        if let s = busy.first(where: { sess in
            if case .softTokens = sess.chunkQueue.first { return true }
            return false
        }) {
            let before = s.outputQueue.count
            _ = stepPrefillForSession(s)
            return s.outputQueue.count - before
        }
        // 2. Single-session fast prefill.
        if busy.count == 1, hasPrefillChunk(busy[0], minTokensThreshold: 2) {
            let before = busy[0].outputQueue.count
            _ = stepPrefillForSession(busy[0])
            return busy[0].outputQueue.count - before
        }
        // 3. AR batch across all busy sessions.
        return step()
    }

    // Pump the scheduler until all sessions hit .done or a budget elapses.
    // Returns the total tokens emitted across all sessions during the run.
    @discardableResult
    func runUntilIdle(maxSteps: Int = 10_000) -> Int {
        var emitted = 0
        for _ in 0..<maxSteps {
            if !hasWork { break }
            emitted += tick()
        }
        return emitted
    }

    // Tokenizer passthroughs so callers don't need to reach into tokenizer.
    func tokenize(_ text: String, addBos: Bool? = nil) -> [UInt32] {
        return tokenizer.encode(text, addBos: addBos)
    }
    func detokenize(_ tokens: [UInt32]) -> String {
        return tokenizer.decode(tokens)
    }
}

// ----------------------------------------------------------------------
// Env-var demo driver. Prompts are passed as numbered env vars so we
// don't have to invent a delimiter that doesn't collide with the chat
// template (which already contains '|', '<', '>', etc.):
//
//   LM_SESSION_1="prompt 1"
//   LM_SESSION_2="prompt 2"
//   ...  (up to LM_SESSION_<B>)
//   LM_MULTISESSION=1                # presence toggles this harness
//   GGUF_PATH=<gguf>
//   [LM_MULTISESSION_MAX=32]         # max new tokens per session
//   [LM_ADD_BOS=1]                   # BOS on (default); =0 to match oracle
//
// Now that residency is decoupled from the active batch, you can submit
// up to MAX_RESIDENT_SESSIONS prompts in one run — the scheduler cycles
// them through the B active slots as earlier ones hit EOS and free up.
// For convenience, LM_MULTISESSION="text1§text2§text3" (section-sign
// separator, U+00A7) still works as a fallback since that character
// doesn't appear in the Gemma chat template.
// ----------------------------------------------------------------------
func runLmMultisession(ggufPath: String, promptsStr: String, maxNewPerSession: Int) {
    print("\n=== LM multisession engine demo ===")
    var prompts: [String] = []
    // Numbered env-var path first (the robust one for chat templates).
    for i in 1...MAX_RESIDENT_SESSIONS {
        if let p = ProcessInfo.processInfo.environment["LM_SESSION_\(i)"], !p.isEmpty {
            prompts.append(p)
        }
    }
    if prompts.isEmpty {
        // Fallback: split promptsStr on § (U+00A7) — safe because the
        // Gemma-4 chat template doesn't use it.
        prompts = promptsStr.split(separator: "\u{00A7}", omittingEmptySubsequences: false).map(String.init)
    }
    guard !prompts.isEmpty else { print("  no prompts"); return }
    if prompts.count > MAX_RESIDENT_SESSIONS {
        print("  warning: \(prompts.count) prompts provided; residency cap is \(MAX_RESIDENT_SESSIONS), extras truncated")
    }
    let active = Array(prompts.prefix(MAX_RESIDENT_SESSIONS))

    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    print("")

    let addBosEnv = ProcessInfo.processInfo.environment["LM_ADD_BOS"]
        .flatMap { ["0", "false", "no"].contains($0.lowercased()) ? false : true }

    let engine = LmEngine(weights: w)
    var sessions: [Session] = []
    for (i, p) in active.enumerated() {
        guard let s = engine.openSession(maxNewTokens: maxNewPerSession) else { break }
        let toks = engine.tokenize(p, addBos: addBosEnv)
        let slotStr = s.slot.map(String.init) ?? "pending"
        print("  session \(s.id) (slot \(slotStr)): prompt=\(p.debugDescription) (\(toks.count) tokens)")
        s.submit(toks)
        sessions.append(s)
        _ = i
    }
    print("")
    print("  --- scheduler pump (interleaved per-session output) ---")

    // Per-session text accumulator so we can show the whole output at the
    // end, AND stream tokens in arrival order to show interleaving.
    var allOutputs: [Int: String] = [:]
    for s in sessions { allOutputs[s.id] = "" }

    let tStart = Date()
    while engine.hasWork {
        _ = engine.tick()
        // Drain any tokens that landed this step, per session, in slot order.
        for s in sessions {
            while let tok = s.nextToken() {
                let frag = engine.detokenize([tok])
                allOutputs[s.id]! += frag
                let tag = "[s\(s.id)]"
                let display = frag.replacingOccurrences(of: "\n", with: "\\n")
                print("  \(tag) \(display)")
            }
        }
    }
    let dtMs = Date().timeIntervalSince(tStart) * 1000

    print("")
    print("  --- done ---")
    print(String(format: "  wall: %.1f ms  steps: %d  total tokens: %d  mean step: %.2f ms",
                 dtMs, engine.totalSteps, engine.totalTokensGenerated,
                 dtMs / Double(engine.totalSteps)))
    print("")
    print("  --- final outputs ---")
    for s in sessions {
        let slotStr = s.slot.map(String.init) ?? "released"
        print("  s\(s.id) (slot \(slotStr), state=\(s.state)): \(allOutputs[s.id]!.debugDescription)")
        engine.closeSession(s)
    }
}

// ----------------------------------------------------------------------
// Shared-prefix demo driver. Opens N sessions sequentially (wait for each
// to finish before submitting the next) so the post-prefill page promotion
// from session i populates the content index that session i+1 probes at
// submit. Expected: session 1 does full work, sessions 2..N show
// "[cache] adopted P shared prefix pages" in the log and finish much
// faster since their prefill tails are tiny.
//
//   LM_SHARED_PREFIX_DEMO=1
//   LM_SHARED_PREFIX=<common prompt prefix>
//   LM_SUFFIX_1=<q1>, LM_SUFFIX_2=<q2>, ...  (distinguishes sessions)
//   [LM_SHARED_PREFIX_MAX=16]    # max new per session
//   GGUF_PATH=<gguf>
//   [LM_ADD_BOS=1]
//   [LM_CACHE_DEBUG=1]           # show per-session adoption counts
// ----------------------------------------------------------------------
func runLmSharedPrefixDemo(ggufPath: String, maxNewPerSession: Int) {
    print("\n=== LM shared-prefix demo ===")
    let env = ProcessInfo.processInfo.environment
    let addBos = env["LM_ADD_BOS"].flatMap {
        ["0", "false", "no"].contains($0.lowercased()) ? false : true
    }
    guard let prefix = env["LM_SHARED_PREFIX"], !prefix.isEmpty else {
        print("  Missing LM_SHARED_PREFIX. See the header comment in lm_engine.swift for usage.")
        return
    }
    var suffixes: [String] = []
    for i in 1...16 {
        if let s = env["LM_SUFFIX_\(i)"], !s.isEmpty { suffixes.append(s) }
    }
    if suffixes.isEmpty {
        print("  No LM_SUFFIX_N env vars — using defaults.")
        suffixes = ["What is the capital of France? One word.",
                    "What is the capital of Germany? One word.",
                    "What is the capital of Japan? One word."]
    }

    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    let engine = LmEngine(weights: w)

    // Build the shared-prefix token sequence once.
    let prefixToks = engine.tokenize(prefix, addBos: addBos)
    print("  shared prefix: \(prefix.debugDescription) (\(prefixToks.count) tokens ≈ \((prefixToks.count + PAGE_SLIDE - 1) / PAGE_SLIDE) pages)")
    print("  running \(suffixes.count) sessions SEQUENTIALLY (each completes before next submits)")
    print("")

    var sessionWalls: [Double] = []
    var sessionOutputs: [String] = []
    var sessionStats: [(adopted: Int, totalPages: Int)] = []

    for (i, suffix) in suffixes.enumerated() {
        let tag = "s\(i+1)"
        let fullPromptToks = prefixToks + engine.tokenize(suffix, addBos: false)
        guard let session = engine.openSession(maxNewTokens: maxNewPerSession) else {
            print("  [\(tag)] openSession failed"); continue
        }
        session.submit(fullPromptToks)

        let adoptedAtSubmit = session.ownedPages.count
        let totalPages = (fullPromptToks.count + PAGE_SLIDE - 1) / PAGE_SLIDE
        sessionStats.append((adopted: adoptedAtSubmit, totalPages: totalPages))

        let t0 = Date()
        var output = ""
        while engine.hasWork {
            _ = engine.tick()
            while let tok = session.nextToken() {
                output += engine.detokenize([tok])
            }
        }
        let dtMs = Date().timeIntervalSince(t0) * 1000
        sessionWalls.append(dtMs)
        sessionOutputs.append(output)

        let pct = totalPages > 0 ? 100 * adoptedAtSubmit / totalPages : 0
        print(String(format: "  [\(tag)] prompt=%d toks / %d pages  adopted=%d (%d%%)  wall=%.1f ms  answer=%@",
                     fullPromptToks.count, totalPages, adoptedAtSubmit, pct, dtMs, output.debugDescription))
        engine.closeSession(session)
    }

    // Summary
    print("")
    print("  --- sharing effectiveness ---")
    let firstWall = sessionWalls.first ?? 0
    for (i, wall) in sessionWalls.enumerated() {
        let stats = sessionStats[i]
        let speedup = firstWall > 0 ? firstWall / wall : 1.0
        let cachedPct = stats.totalPages > 0 ? 100 * stats.adopted / stats.totalPages : 0
        print(String(format: "  s%d: %.1f ms  (%.2f× vs s1 baseline)  — %d%% of pages cache-adopted",
                     i + 1, wall, speedup, cachedPct))
    }
}

// ----------------------------------------------------------------------
// Branch demo driver. Opens a BASE session with a shared prefix, fully
// prefills it (so KV is populated), then opens N branch sessions that
// each adoptKvFrom(base, pageCount: prefixPages) and submit their own
// divergent suffix. All N branches decode CONCURRENTLY — real compute
// reuse from a single shared prefix.
//
//   LM_BRANCH_DEMO=1
//   LM_BRANCH_PREFIX=<common prompt>
//   LM_BRANCH_SUFFIX_1=<q1>, LM_BRANCH_SUFFIX_2=<q2>, ...
//   [LM_BRANCH_MAX=16]       # max new per branch
//   GGUF_PATH=<gguf> [LM_ADD_BOS=1]
//
// This is the text-only prototype of what the real "same image, multiple
// suffixes" demo will look like — same structure, just with an image
// chunk in the prefix. Once image scatter semantics are fixed, swap
// LM_BRANCH_PREFIX text for actual vision-tower soft tokens and the
// branching API is unchanged.
// ----------------------------------------------------------------------
func runLmBranchDemo(ggufPath: String, maxNewPerBranch: Int) {
    print("\n=== LM prefix-branch demo (adoptKvFrom) ===")
    let env = ProcessInfo.processInfo.environment
    let addBos = env["LM_ADD_BOS"].flatMap {
        ["0", "false", "no"].contains($0.lowercased()) ? false : true
    }
    guard let prefix = env["LM_BRANCH_PREFIX"], !prefix.isEmpty else {
        print("  Missing LM_BRANCH_PREFIX."); return
    }
    var suffixes: [String] = []
    for i in 1...16 {
        if let s = env["LM_BRANCH_SUFFIX_\(i)"], !s.isEmpty { suffixes.append(s) }
    }
    if suffixes.isEmpty {
        suffixes = [" What is the capital of France? One word.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
                    " What is the capital of Japan? One word.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
                    " What is the capital of Germany? One word.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"]
    }

    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    let engine = LmEngine(weights: w)

    // --- 1. Prime the BASE session with the shared prefix ---
    let prefixToks = engine.tokenize(prefix, addBos: addBos)
    print("  shared prefix: \(prefix.debugDescription)")
    print("  (\(prefixToks.count) tokens ≈ \((prefixToks.count + PAGE_SLIDE - 1) / PAGE_SLIDE) pages; \(prefixToks.count / PAGE_SLIDE) full pages shareable)")

    guard let base = engine.openSession(maxNewTokens: 0) else {
        print("  openSession(base) failed"); return
    }
    base.submit(prefixToks)

    let tPrime = Date()
    // Pump until base has consumed all prefix tokens. We don't need it to
    // generate anything — just get its KV populated so the full pages are
    // promoted to the content index. base's maxNewTokens=0 stops gen
    // immediately once the last prefill token flips state to generating.
    while base.state == .priming {
        _ = engine.tick()
    }
    let primeMs = Date().timeIntervalSince(tPrime) * 1000
    let pagesToShare = base.position / PAGE_SLIDE
    print(String(format: "  base prefill complete: pos=%d, %d full pages available for sharing, wall=%.1f ms",
                 base.position, pagesToShare, primeMs))

    // --- 2. Open N branch sessions, each adopts base's first N pages ---
    var branches: [Session] = []
    for (i, sfx) in suffixes.enumerated() {
        guard let branch = engine.openSession(maxNewTokens: maxNewPerBranch) else {
            print("  openSession(branch \(i+1)) failed"); break
        }
        // Adopt base's pages — this is the "skip the prefix prefill" step.
        _ = branch.adoptKvFrom(base, pageCount: pagesToShare)
        branch.submit(engine.tokenize(sfx, addBos: false))
        branches.append(branch)
    }
    print("  opened \(branches.count) branches; each inherits \(pagesToShare) pages of KV from base without reprefilling")

    // --- 3. Pump concurrently until all branches finish ---
    var outputs: [Int: String] = [:]
    for b in branches { outputs[b.id] = "" }
    let tRun = Date()
    while engine.hasWork {
        _ = engine.tick()
        for b in branches {
            while let tok = b.nextToken() {
                outputs[b.id, default: ""] += engine.detokenize([tok])
            }
        }
    }
    let runMs = Date().timeIntervalSince(tRun) * 1000
    print(String(format: "  branches ran concurrently: %.1f ms (mean step %.1f ms × %d steps)",
                 runMs, engine.lastStepMs, engine.totalSteps))

    // --- 4. Report outputs + stats ---
    print("")
    print("  --- branch outputs ---")
    for (i, b) in branches.enumerated() {
        print("  branch \(i+1) (session \(b.id)): \(outputs[b.id]!.debugDescription)")
        engine.closeSession(b)
    }
    engine.closeSession(base)

    let pagesPerBranch = pagesToShare
    let sharedWork = pagesPerBranch * branches.count
    print("")
    print(String(format: "  sharing: %d pages × %d branches = %d page-writes skipped vs naive \"reprefill per branch\"",
                 pagesPerBranch, branches.count, sharedWork))
}

// ----------------------------------------------------------------------
// Multimodal demo driver — feeds vision-tower output through a session.
// Env vars:
//   GGUF_PATH=<gguf>                 # LM weights
//   VISION_ST=<safetensors path>     # vision tower weights
//   LM_MULTIMODAL=<image png path>   # image to embed and prompt with
//   [LM_MULTIMODAL_PREFIX="…"]       # text before image tokens
//   [LM_MULTIMODAL_SUFFIX="…"]       # text after image tokens
//   [LM_MULTIMODAL_MAX=48]           # generation budget
//   [LM_ADD_BOS=1]
//
// Runs: vision_tower(image) → soft_tokens, opens a session, submits
// prefix-text + soft_tokens + suffix-text into the session queue, then
// pumps the scheduler. Each chunk dispatches as either a text-token
// fast prefill or a soft-token fast prefill (skip embed). Stream
// output per session.
// ----------------------------------------------------------------------
func runLmMultimodal(ggufPath: String, stPath: String, imagePath: String,
                      prefix: String, suffix: String, maxNew: Int) {
    print("\n=== LM multimodal engine demo ===")
    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }

    let visWeights: VisionWeights
    do {
        let st = try SafetensorsFile(stPath)
        visWeights = try loadVisionWeights(st, device: device)
    } catch {
        print("  loadVisionWeights failed: \(error)"); return
    }

    let batch: PatchBatch
    do { batch = try gemma4ImagePreprocess(path: imagePath, device: device) }
    catch { print("  gemma4ImagePreprocess failed: \(error)"); return }
    print("  image preprocessed: \(batch.numRealPatches) real patches, grid \(batch.gridH)×\(batch.gridW)")

    let (softTokens, nPooled) = runVisionTowerForward(batch: batch, weights: visWeights,
                                                       device: device, queue: queue)
    print("  vision tower → \(nPooled) soft tokens (fp32, [\(nPooled), \(HIDDEN)])")
    if nPooled == 0 {
        print("  0 soft tokens — image too small? aborting"); return
    }

    let addBosEnv = ProcessInfo.processInfo.environment["LM_ADD_BOS"]
        .flatMap { ["0", "false", "no"].contains($0.lowercased()) ? false : true }

    let engine = LmEngine(weights: w)
    guard let sess = engine.openSession(maxNewTokens: maxNew) else {
        print("  no free slot"); return
    }
    let prefixToks = engine.tokenize(prefix, addBos: addBosEnv)
    let suffixToks = engine.tokenize(suffix, addBos: false)
    print("  prefix tokens: \(prefixToks.count); suffix tokens: \(suffixToks.count)")
    sess.submit(prefixToks)
    // Bracket the soft tokens with Gemma-4's BOI/EOI markers so the model
    // sees the same turn-boundary signal it was trained on:
    //   boi_token_id = 255999 (<|image>)
    //   eoi_token_id = 258882 (<image|>)
    // Without these markers, image-region attention heads have no signal
    // that the following hidden states are image features, and the model
    // falls back to <pad> as a uniform prior.
    let BOI: UInt32 = 255999
    let EOI: UInt32 = 258882
    sess.submit([BOI])
    sess.submit(softTokens: softTokens, count: nPooled, isFp32: true)
    sess.submit([EOI])

    print("")
    print("  --- generation ---")
    print("  \(prefix)<image:\(nPooled)soft>\(suffix)", terminator: "")
    fflush(stdout)

    let tStart = Date()
    var output = ""
    while engine.hasWork {
        _ = engine.tick()
        while let tok = sess.nextToken() {
            let frag = engine.detokenize([tok])
            output += frag
            print(frag.replacingOccurrences(of: "\n", with: "\\n"), terminator: "")
            fflush(stdout)
        }
    }
    print("")
    let dtMs = Date().timeIntervalSince(tStart) * 1000
    print("")
    print(String(format: "  wall: %.1f ms  steps: %d  tokens: %d",
                 dtMs, engine.totalSteps, engine.totalTokensGenerated))
    print("  output: \(output.debugDescription)")
    engine.closeSession(sess)
}

