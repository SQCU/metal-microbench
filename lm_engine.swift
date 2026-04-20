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

    // Mid-generation append: extend a session that already has KV history
    // (generated some tokens, hit a turn boundary, got paused by the caller)
    // with new tokens/softs. The next tick prefills them against the existing
    // block_table starting at the current `position`, then AR resumes.
    //
    // Differs from submit() in two ways:
    //   - reopens a .done session (submit() preserves .done as terminal)
    //   - resets numGenerated so maxNewTokens governs the next turn, not the
    //     running total (opt out via resetBudget: false for cumulative cap)
    //
    // Appropriate for multiturn chat, tool-call responses, injecting a
    // rendered-SVG image result back to the agent that requested it, etc.
    func append(_ tokens: [UInt32], resetBudget: Bool = true) {
        guard !tokens.isEmpty else { return }
        if state == .done { state = .paused }
        if resetBudget { numGenerated = 0 }
        submit(tokens)
    }
    func append(text: String, resetBudget: Bool = true) {
        guard let eng = engine else { return }
        append(eng.tokenizer.encode(text, addBos: false), resetBudget: resetBudget)
    }
    func append(softTokens: MTLBuffer, count: Int, isFp32: Bool, resetBudget: Bool = true) {
        guard count > 0 else { return }
        if state == .done { state = .paused }
        if resetBudget { numGenerated = 0 }
        submit(softTokens: softTokens, count: count, isFp32: isFp32)
    }

    // Pull the next generated token, or nil if none ready.
    func nextToken() -> UInt32? {
        guard !outputQueue.isEmpty else { return nil }
        return outputQueue.removeFirst()
    }

    var pendingOutputCount: Int { outputQueue.count }
    var pendingPrimingCount: Int { chunkQueue.reduce(0) { $0 + $1.count } }
    var ownedPagesForDebug: [Int] { ownedPages }
    var positionForDebug: Int { position }

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
    // call:
    //
    //   1. Any session has a .softTokens head chunk  →  single-slot fast
    //      prefill for that session (image tokens can't go through AR).
    //   2. ALL busy sessions are .priming with .tokens chunks of ≥2 remaining
    //      → multi-slot fast prefill in ONE CB, all slots active. This is
    //      the simultaneous-submission case: 4 users POST-ing at once should
    //      settle into peak throughput, not 16 AR-steps of zero emits.
    //   3. Exactly one session busy AND its head is tokens-chunk ≥ 2  →
    //      single-slot fast prefill.
    //   4. Otherwise  →  AR step across all busy sessions. Handles mixed
    //      prime+gen naturally: priming slots consume priming tokens, gen
    //      slots emit — so staggered submissions pipeline automatically.
    //
    // Returns tokens emitted this tick (usually 0 during prefill unless the
    // chunk drained and we sampled the first generated token).
    @discardableResult
    func tick() -> Int {
        runAdmissionPass()
        let busy = activeSessions.filter { $0.state.isBusy }
        if busy.isEmpty { return 0 }
        // 1a. Multi-slot soft-tokens prefill: ALL busy sessions have a
        //     .softTokens chunk at head → one buildPrefillCB dispatch
        //     processes every slot's softs simultaneously (skipEmbed=true,
        //     per-slot rows in pre_hidden). The point is weight-load
        //     amortization: one dense-GEMV of attnQ/attnK/attnV runs once
        //     and feeds all 4 slots' Q/K/V projections, instead of 4
        //     single-slot dispatches each reloading the weights. Same
        //     story for every layer's MLP/MoE/projection.
        let softBusy = busy.filter { sess in
            if case .softTokens = sess.chunkQueue.first { return true }
            return false
        }
        if softBusy.count == busy.count && softBusy.count > 1 {
            let beforeCounts = softBusy.map { $0.outputQueue.count }
            _ = stepMultiSlotSoftPrefill(softBusy)
            return zip(softBusy, beforeCounts).reduce(0) { $0 + ($1.0.outputQueue.count - $1.1) }
        }
        // 1b. Single-session soft-tokens path (or mixed: only one session
        //     has softs pending while others are elsewhere in their queue).
        if let s = softBusy.first {
            let before = s.outputQueue.count
            _ = stepPrefillForSession(s)
            return s.outputQueue.count - before
        }
        // 2. Multi-slot prefill: all busy sessions priming with ≥2 prefill
        //    tokens remaining. Peak aggregate throughput for cold-start of
        //    many concurrent users — each slot primes its own tokens in the
        //    same buildPrefillCB call, no AR-stepping zero-emit steps.
        if busy.count > 1, allPrimeReady(busy, minTokensThreshold: 2) {
            let beforeCounts = busy.map { $0.outputQueue.count }
            _ = stepMultiSlotPrefill(busy)
            return zip(busy, beforeCounts).reduce(0) { $0 + ($1.0.outputQueue.count - $1.1) }
        }
        // 3. Single-session fast prefill.
        if busy.count == 1, hasPrefillChunk(busy[0], minTokensThreshold: 2) {
            let before = busy[0].outputQueue.count
            _ = stepPrefillForSession(busy[0])
            return busy[0].outputQueue.count - before
        }
        // 4. AR batch across all busy sessions.
        return step()
    }

    // True if every session in `sessions` is priming with a .tokens head
    // chunk of at least `minTokensThreshold` tokens remaining. Gate for
    // multi-slot fast prefill; single-token tails fall through to AR step
    // where 1-token priming is nearly free (~34 ms vs ~133 ms for fast prefill).
    private func allPrimeReady(_ sessions: [Session], minTokensThreshold: Int) -> Bool {
        for s in sessions {
            guard s.state == .priming else { return false }
            guard let head = s.chunkQueue.first else { return false }
            guard case .tokens(let ts) = head, ts.count >= minTokensThreshold else { return false }
        }
        return true
    }

    // Multi-slot fast prefill: one buildPrefillCB dispatch that primes every
    // slot's own session simultaneously. Every slot's block_table points to
    // its session's real phys pages; each slot writes K/V at its own position
    // range via multi-position kv_write_multi + per-slot CSR.
    //
    // qLen is min(MAX_Q_LEN, min(remaining prefill tokens across sessions))
    // so no slot reads past the end of its chunk. Sessions with more tokens
    // than qLen stay in .priming; the next tick processes the next tile.
    @discardableResult
    func stepMultiSlotPrefill(_ sessions: [Session]) -> Bool {
        runAdmissionPass()
        // Gather each busy slot's priming session + chunk.
        var slotSession: [Session?] = Array(repeating: nil, count: B)
        var slotTokens: [[UInt32]] = Array(repeating: [], count: B)
        for s in sessions {
            guard let sslot = s.slot else { continue }
            guard case .tokens(let ts) = s.chunkQueue.first, !ts.isEmpty else { continue }
            slotSession[sslot] = s
            slotTokens[sslot] = ts
        }
        // min remaining across active slots; cap at MAX_Q_LEN.
        var qLen = MAX_Q_LEN
        var any = false
        for b in 0..<B {
            if let _ = slotSession[b] {
                any = true
                qLen = min(qLen, slotTokens[b].count)
            }
        }
        guard any, qLen >= 1 else { return false }

        // Each participating slot: ensure pages + install real block_table.
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        var savedBT = [UInt32](repeating: 0, count: B * MAX_PAGES_PER_SLOT)
        for i in 0..<(B * MAX_PAGES_PER_SLOT) { savedBT[i] = btP[i] }
        for b in 0..<B {
            if let s = slotSession[b] {
                if !ensurePages(s, forKLen: s.position + qLen + 1) { return false }
                installBlockTable(s, slot: b)
            } else {
                // Silence: point at scratch strip. Guards against stale K from
                // a prior single-slot prefill leaving real-page pointers here.
                for p in 0..<MAX_PAGES_PER_SLOT {
                    btP[b * MAX_PAGES_PER_SLOT + p] =
                        UInt32(SCRATCH_PAGE_BASE + (p % SCRATCH_STRIP))
                }
            }
        }
        defer { for i in 0..<(B * MAX_PAGES_PER_SLOT) { btP[i] = savedBT[i] } }

        // Populate pre_input_tokens, pre_q_positions, pre_k_len_*.
        let tokP = pre_input_tokens.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
        let posP = pre_q_positions.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
        let klsP = pre_k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let klfP = pre_k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
        for b in 0..<B {
            if let s = slotSession[b] {
                let ts = slotTokens[b]
                for i in 0..<qLen {
                    tokP[b * qLen + i] = ts[i]
                    posP[b * qLen + i] = UInt32(s.position + i)
                }
                klsP[b] = UInt32(s.position + qLen)
                klfP[b] = UInt32(s.position + qLen)
            } else {
                for i in 0..<qLen {
                    tokP[b * qLen + i] = weights.bosTokenId
                    posP[b * qLen + i] = 0
                }
                klsP[b] = 1
                klfP[b] = 1
            }
        }

        // precomputeFlexPrefillMasks reads per-slot q_first from pre_q_positions.
        precomputeFlexPrefillMasks(qLen: qLen, positionStart: 0)
        let cb = buildPrefillCB(weights, qLen: qLen, skipEmbed: false)
        cb.commit(); cb.waitUntilCompleted()
        totalSteps += 1
        if let err = cb.error { print("  GPU multi-prefill error: \(err)"); return false }

        // Per-slot: advance position, pop chunk, promote pages, copy logit,
        // and if the chunk drained, transition to .generating + sample first
        // gen token from the slot's last-position logit.
        let srcPtr = pre_logits.contents().assumingMemoryBound(to: Float16.self)
        let dstPtr = logits.contents().assumingMemoryBound(to: Float16.self)
        for b in 0..<B {
            guard let s = slotSession[b] else { continue }
            s.position += qLen
            promoteFinishedPages(s)
            // Copy this slot's final-Q-row logit into AR logits for downstream.
            let src = srcPtr.advanced(by: (b * qLen + (qLen - 1)) * VOCAB)
            let dst = dstPtr.advanced(by: b * VOCAB)
            memcpy(dst, src, VOCAB * 2)
            // Pop qLen tokens from the session's chunk head.
            var ts = slotTokens[b]
            ts.removeFirst(qLen)
            if ts.isEmpty { s.chunkQueue.removeFirst() }
            else          { s.chunkQueue[0] = .tokens(ts) }
            // Transition if chunk is drained.
            if s.chunkQueue.isEmpty {
                let base = b * VOCAB
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
        }
        return true
    }

    // Multi-slot fast prefill for .softTokens chunks. Mirrors
    // stepMultiSlotPrefill but pre-populates pre_hidden from each slot's
    // own soft-tokens buffer (at its current byteOffset) and commits with
    // skipEmbed=true — the vision-tower-produced rows already live in
    // text-hidden space, no embed_lookup needed.
    //
    // Savings: on the tetraplex-with-4-images demo, serial single-slot
    // soft prefill runs 4 × 35 tiles × ~130 ms = ~18 s wall. This path
    // runs 35 tiles × ~150 ms = ~5 s wall, because each dense-GEMV loads
    // its weights once and feeds all 4 slots' projections.
    @discardableResult
    func stepMultiSlotSoftPrefill(_ sessions: [Session]) -> Bool {
        runAdmissionPass()
        // Gather each busy slot's current soft-tokens chunk.
        struct SoftRef {
            let session: Session
            let buffer: MTLBuffer
            let remainingCount: Int
            let isFp32: Bool
            let byteOffset: Int
        }
        var slotSoft: [Int: SoftRef] = [:]
        for s in sessions {
            guard let sslot = s.slot else { continue }
            guard case let .softTokens(buf, count, isFp32, byteOffset) = s.chunkQueue.first
            else { continue }
            slotSoft[sslot] = SoftRef(session: s, buffer: buf, remainingCount: count,
                                       isFp32: isFp32, byteOffset: byteOffset)
        }
        guard !slotSoft.isEmpty else { return false }
        // qLen = min over slots of remaining rows, clamped to MAX_Q_LEN.
        var qLen = MAX_Q_LEN
        for (_, sr) in slotSoft { qLen = min(qLen, sr.remainingCount) }
        guard qLen >= 1 else { return false }

        // Install real block_table entries for participating slots; silence
        // the rest by redirecting their pages to the scratch strip.
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        var savedBT = [UInt32](repeating: 0, count: B * MAX_PAGES_PER_SLOT)
        for i in 0..<(B * MAX_PAGES_PER_SLOT) { savedBT[i] = btP[i] }
        for b in 0..<B {
            if let sr = slotSoft[b] {
                if !ensurePages(sr.session, forKLen: sr.session.position + qLen + 1) { return false }
                installBlockTable(sr.session, slot: b)
            } else {
                for p in 0..<MAX_PAGES_PER_SLOT {
                    btP[b * MAX_PAGES_PER_SLOT + p] =
                        UInt32(SCRATCH_PAGE_BASE + (p % SCRATCH_STRIP))
                }
            }
        }
        defer { for i in 0..<(B * MAX_PAGES_PER_SLOT) { btP[i] = savedBT[i] } }

        // Populate pre_hidden[slot * qLen * HIDDEN ..] with each slot's softs.
        // Layout: [B, qLen, HIDDEN] fp16 (pre_hidden is fp16 even though the
        // source softs may be fp32 — we convert row-by-row). Silenced slots
        // get zeros so garbage doesn't feed downstream kernels.
        let pH = pre_hidden.contents().assumingMemoryBound(to: Float16.self)
        let posP = pre_q_positions.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
        let klsP = pre_k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let klfP = pre_k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
        for b in 0..<B {
            let dstBase = (b * qLen) * HIDDEN
            if let sr = slotSoft[b] {
                let srcRaw = sr.buffer.contents().advanced(by: sr.byteOffset)
                if sr.isFp32 {
                    let srcPtr = srcRaw.assumingMemoryBound(to: Float.self)
                    for i in 0..<(qLen * HIDDEN) {
                        pH[dstBase + i] = Float16(srcPtr[i])
                    }
                } else {
                    memcpy(pH.advanced(by: dstBase), srcRaw, qLen * HIDDEN * 2)
                }
                for i in 0..<qLen {
                    posP[b * qLen + i] = UInt32(sr.session.position + i)
                }
                klsP[b] = UInt32(sr.session.position + qLen)
                klfP[b] = UInt32(sr.session.position + qLen)
            } else {
                for i in 0..<(qLen * HIDDEN) { pH[dstBase + i] = 0 }
                for i in 0..<qLen { posP[b * qLen + i] = 0 }
                klsP[b] = 1
                klfP[b] = 1
            }
        }

        precomputeFlexPrefillMasks(qLen: qLen, positionStart: 0)
        let cb = buildPrefillCB(weights, qLen: qLen, skipEmbed: true)
        cb.commit(); cb.waitUntilCompleted()
        totalSteps += 1
        if let err = cb.error { print("  GPU multi-soft-prefill error: \(err)"); return false }

        // Per-slot: advance position, promote pages, copy final logit,
        // update the soft-chunk's byteOffset (or pop it + sample first
        // gen token if this was the last chunk in the queue).
        let srcPtr = pre_logits.contents().assumingMemoryBound(to: Float16.self)
        let dstPtr = logits.contents().assumingMemoryBound(to: Float16.self)
        for b in 0..<B {
            guard let sr = slotSoft[b] else { continue }
            let s = sr.session
            s.position += qLen
            promoteFinishedPages(s)
            let src = srcPtr.advanced(by: (b * qLen + (qLen - 1)) * VOCAB)
            let dst = dstPtr.advanced(by: b * VOCAB)
            memcpy(dst, src, VOCAB * 2)
            let remaining = sr.remainingCount - qLen
            if remaining <= 0 {
                s.chunkQueue.removeFirst()
            } else {
                let bpe = sr.isFp32 ? 4 : 2
                let newOffset = sr.byteOffset + qLen * HIDDEN * bpe
                s.chunkQueue[0] = .softTokens(buffer: sr.buffer, count: remaining,
                                               isFp32: sr.isFp32, byteOffset: newOffset)
            }
            // If this was the last chunk of the session's priming queue,
            // transition to .generating + sample the first token from the
            // slot's final-Q-row logit (mirrors stepMultiSlotPrefill's tail).
            if s.chunkQueue.isEmpty {
                let base = b * VOCAB
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
        }
        return true
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
// Async pause/resume demo — models an agent with concurrent sessions,
// some of which stall mid-generation waiting for tool-call results.
// Demonstrates that a paused session retains its KV pages (so the tool
// response resumes decode in-place without reprefill) and that OTHER
// sessions keep making progress on the freed slot while the paused
// session is dormant.
//
//   LM_PAUSE_DEMO=1
//   LM_PAUSE_PROMPT_1=<prompt for session 1>
//   LM_PAUSE_PROMPT_2=<prompt for session 2>
//   LM_PAUSE_AFTER_TOKENS=3              # pause session 1 after N gen tokens
//   LM_PAUSE_TOOL_RESULT=" the answer"    # faked tool-result text to submit on resume
//   LM_PAUSE_RESUME_AFTER_STEPS=10       # how many ticks to wait before resume
//   GGUF_PATH=<gguf> [LM_ADD_BOS=1]
//
// Output logs the tick at which each event occurs so you can see the
// interleaving. Session 2 keeps decoding the whole time — the engine
// correctly drops paused sessions from the active batch and reassigns
// the slot until the caller calls submit() again.
// ----------------------------------------------------------------------
func runLmPauseResumeDemo(ggufPath: String) {
    print("\n=== LM pause/resume demo ===")
    let env = ProcessInfo.processInfo.environment
    let addBos = env["LM_ADD_BOS"].flatMap {
        ["0", "false", "no"].contains($0.lowercased()) ? false : true
    }
    guard let p1 = env["LM_PAUSE_PROMPT_1"], !p1.isEmpty,
          let p2 = env["LM_PAUSE_PROMPT_2"], !p2.isEmpty else {
        print("  Missing LM_PAUSE_PROMPT_1 / LM_PAUSE_PROMPT_2."); return
    }
    let pauseAfter = Int(env["LM_PAUSE_AFTER_TOKENS"] ?? "3") ?? 3
    let resumeAfter = Int(env["LM_PAUSE_RESUME_AFTER_STEPS"] ?? "10") ?? 10
    let toolResult = env["LM_PAUSE_TOOL_RESULT"] ?? " the capital is Paris."

    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    let engine = LmEngine(weights: w)

    guard let s1 = engine.openSession(maxNewTokens: 32),
          let s2 = engine.openSession(maxNewTokens: 32) else {
        print("  openSession failed"); return
    }
    s1.submit(engine.tokenize(p1, addBos: addBos))
    s2.submit(engine.tokenize(p2, addBos: addBos))

    var out1 = "", out2 = ""
    var s1TokensBeforePause = 0
    var pausedAtTick = -1
    var resumedAtTick = -1
    var tick = 0

    print("  session \(s1.id) prompt: \(p1.debugDescription)")
    print("  session \(s2.id) prompt: \(p2.debugDescription)")
    print("  policy: pause s\(s1.id) after it generates \(pauseAfter) tokens; resume after \(resumeAfter) more ticks; inject tool-result=\(toolResult.debugDescription)")
    print("")
    print("  tick  event")

    while engine.hasWork {
        _ = engine.tick()
        tick += 1
        // Drain output tokens per session.
        while let tok = s1.nextToken() {
            let frag = engine.detokenize([tok])
            out1 += frag
            // Count only generating tokens (priming doesn't emit).
            s1TokensBeforePause += 1
            print("  \(String(format: "%3d", tick))   [s\(s1.id)] \(frag.debugDescription)")
            // Trigger the pause once s1 has generated N tokens AND isn't
            // already paused/resumed.
            if s1TokensBeforePause >= pauseAfter, pausedAtTick < 0 {
                s1.pause()
                pausedAtTick = tick
                print("  \(String(format: "%3d", tick))   [s\(s1.id)] → pause (simulated tool call; KV retained, slot released)")
            }
        }
        while let tok = s2.nextToken() {
            out2 += engine.detokenize([tok])
            print("  \(String(format: "%3d", tick))   [s\(s2.id)] \(engine.detokenize([tok]).debugDescription)")
        }
        // Resume s1 after `resumeAfter` ticks have elapsed since the pause.
        if pausedAtTick > 0, resumedAtTick < 0,
           tick - pausedAtTick >= resumeAfter {
            // Inject a faked tool response — session state flips back to
            // .priming and re-admits on the next tick.
            let toolToks = engine.tokenize(toolResult, addBos: false)
            s1.submit(toolToks)
            resumedAtTick = tick
            print("  \(String(format: "%3d", tick))   [s\(s1.id)] ← submit(tool_result=\(toolToks.count) tokens); re-admits next tick")
        }
        // Safety: cap at many ticks to avoid infinite loops.
        if tick > 400 { print("  tick cap reached, aborting"); break }
    }

    print("")
    print("  --- final outputs ---")
    print("  s\(s1.id): \(out1.debugDescription)   (paused at tick \(pausedAtTick), resumed at tick \(resumedAtTick))")
    print("  s\(s2.id): \(out2.debugDescription)")

    print("")
    print("  --- sanity check ---")
    print("  during s\(s1.id)'s pause (ticks \(pausedAtTick)..\(resumedAtTick)), s\(s2.id) continued producing tokens:")
    // Because outputs per tick were printed interleaved, the user can
    // inspect above. We also note that after resume, s1 picked up its
    // pre-pause KV state exactly — tool-response tokens get teacher-
    // forced against the KV already in place without reprefilling
    // anything.

    engine.closeSession(s1)
    engine.closeSession(s2)
}

// ----------------------------------------------------------------------
// Multiturn demo — exercises Session.append() for the chat-loop case:
// one session lives across multiple user turns, each turn's KV staying
// resident so subsequent turns pay only their own prefill cost.
//
// Flow:
//   turn 1 user prompt  → submit()            → generate N tokens
//   (simulate turn end) → pause()
//   turn 2 user prompt  → append()            → generate N more
//   (turn end)          → pause()
//   turn 3              → append()            → generate N more
//
// The append() primitive is the engine-side contract that enables:
//   - multiturn chat (what this demo shows)
//   - tool-call response injection (runLmPauseResumeDemo pattern, now
//     usable via append() instead of raw submit())
//   - mid-session image injection (tool-call returns a rendered SVG
//     the agent needs to look at: append(softTokens:))
//   - agent interruption (another agent's "turn" injected into an
//     ongoing session by an external coordinator)
//
// Env:
//   LM_MULTITURN_DEMO=1            # presence toggles this harness
//   GGUF_PATH=<path>
//   [LM_MULTITURN_TURNS="prompt1§prompt2§prompt3"]   # § separator
//   [LM_MULTITURN_MAX_PER_TURN=24] # per-turn generation cap
//   [LM_MULTITURN_FRESH=1]         # opt-in: close+reopen each turn
//                                   (alternative path: replay full history
//                                    to a fresh session; not needed for
//                                    correctness, but useful for A/B).
//
func runLmMultiturnDemo(ggufPath: String, turnsStr: String?, maxPerTurn: Int) {
    print("\n=== LM multiturn demo (Session.append) ===")
    let defaultTurns = [
        "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
        "<turn|>\n<|turn>user\nAnd Germany?<turn|>\n<|turn>model\n",
        "<turn|>\n<|turn>user\nItaly?<turn|>\n<|turn>model\n",
    ]
    let turns: [String]
    if let s = turnsStr, !s.isEmpty {
        turns = s.components(separatedBy: "§")
    } else {
        turns = defaultTurns
    }
    guard !turns.isEmpty else {
        print("  no turns to run"); return
    }
    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    let engine = LmEngine(weights: w)
    guard let s = engine.openSession(maxNewTokens: maxPerTurn) else {
        print("  openSession failed"); return
    }
    print("  session \(s.id) — \(turns.count) turns, \(maxPerTurn) tok/turn cap")
    print("")

    let freshMode = ProcessInfo.processInfo.environment["LM_MULTITURN_FRESH"] != nil
    var currentSession = s
    var accumulatedHistory: [UInt32] = []
    for (i, turn) in turns.enumerated() {
        if i == 0 {
            let toks = engine.tokenize(turn, addBos: true)
            accumulatedHistory.append(contentsOf: toks)
            currentSession.submit(toks)
        } else if freshMode {
            // Control experiment: fresh session + replay full history as prompt.
            // Isolates whether the garbling is in reused-session state or KV itself.
            engine.closeSession(currentSession)
            guard let s2 = engine.openSession(maxNewTokens: maxPerTurn) else {
                print("  openSession failed on turn \(i + 1)"); return
            }
            currentSession = s2
            let newToks = engine.tokenize(turn, addBos: false)
            accumulatedHistory.append(contentsOf: newToks)
            currentSession.submit(accumulatedHistory)
            print("    [fresh session \(currentSession.id) replaying \(accumulatedHistory.count) tokens]")
        } else {
            currentSession.append(engine.tokenize(turn, addBos: false))
        }
        let s = currentSession
        print("  turn \(i + 1) input: \(turn.debugDescription)")
        var turnOutput = ""
        var turnTokenIds: [UInt32] = []
        var tick = 0
        let budgetFloor = s.numGenerated  // safety: break if we stall
        while engine.hasWork && s.state != .done {
            _ = engine.tick()
            tick += 1
            while let tok = s.nextToken() {
                turnOutput += engine.detokenize([tok])
                turnTokenIds.append(tok)
            }
            // Per-turn generation cap — session's own maxNewTokens will trip
            // .done which exits the loop; this is belt-and-suspenders.
            if tick > 400 { print("    (tick cap)"); break }
            if s.numGenerated - budgetFloor > maxPerTurn + 4 { break }
        }
        print("  turn \(i + 1) output (\(s.numGenerated - budgetFloor) tok, \(tick) ticks): \(turnOutput.debugDescription)")
        print("    ids: \(turnTokenIds.prefix(12))")
        // End of turn — pause so the next append resets the budget.
        s.pause()
        print("    position=\(s.position), ownedPages=\(s.ownedPagesForDebug.count), state=\(s.state)")
        print("")
    }

    print("  === summary ===")
    print("  session held KV across \(turns.count) turns; final position=\(currentSession.position) tokens")
    print("  per-turn prefill cost: only the new user-message tokens (no re-prefill of history)")
    engine.closeSession(currentSession)
}

// ----------------------------------------------------------------------
// Composite benchmark — the headline multiuser/multiturn workload.
//
// N concurrent users. Shared system prompt (exercises content-hash cache:
// user 1 primes the system prompt, users 2..N hit-and-adopt its pages). K
// turns per user. All users submit each turn simultaneously — the scheduler
// must handle that cold-start gracefully (multi-slot prefill, not AR-prime).
//
// Measures per-user per-turn:
//   - TTFT (submit → first emitted token)
//   - gen rate (tokens / last_tok_time - first_tok_time)
// And aggregate: total tokens / total wall.
//
// This is what we'd use to pitch the engine: "4 users × 3 turns, X tok/s
// sustained throughput, Y ms median TTFT, Z page-cache hits."
//
// Env:
//   LM_COMPOSITE_DEMO=1           # presence toggles
//   GGUF_PATH=<path>
//   [LM_COMPOSITE_N=4]            # concurrent users (≤ MAX_RESIDENT_SESSIONS)
//   [LM_COMPOSITE_MAX_PER_TURN=24]
//
func runLmCompositeDemo(ggufPath: String, nUsers: Int, maxPerTurn: Int) {
    print("\n=== LM composite benchmark: N=\(nUsers) users × 3 turns, shared system prompt ===")
    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    let engine = LmEngine(weights: w)

    // Shared system prompt. Same bytes for every user → content-hash cache
    // gives user 1's filled pages to users 2..N read-only.
    let system = "<|turn>system\nYou are a concise assistant. Give one-sentence answers.<turn|>\n"
    // Per-user, per-turn user message. All users get the same turn sequence
    // for the benchmark; a real server would have different messages.
    let userTurns = [
        "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        "<turn|>\n<|turn>user\nWhat about Germany?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        "<turn|>\n<|turn>user\nAnd Italy?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    ]

    struct TurnMetrics {
        var submitTime: Date = Date()
        var firstTokenTime: Date?
        var lastTokenTime: Date?
        var tokenCount: Int = 0
        var text: String = ""
    }

    // Warmup: the first buildStepCB / buildPrefillCB pair pays ~5s of
    // Metal pipeline compilation. That cost is one-time per process, not
    // per-request, but naively-measured TTFT eats it on turn 1. Run one
    // tiny throwaway decode first so the real measurement reflects steady
    // state.
    do {
        print("  (warmup: one-session 3-token decode to compile pipelines)")
        let wStart = Date()
        guard let ws = engine.openSession(maxNewTokens: 3) else { return }
        ws.submit(engine.tokenize("<|turn>user\nhi<turn|>\n<|turn>model\n", addBos: true))
        while engine.hasWork { _ = engine.tick() }
        while let _ = ws.nextToken() {}
        engine.closeSession(ws)
        print(String(format: "  warmup took %.2fs", Date().timeIntervalSince(wStart)))
    }

    // Open N sessions up front.
    var sessions: [Session] = []
    for _ in 0..<nUsers {
        guard let s = engine.openSession(maxNewTokens: maxPerTurn) else {
            print("  openSession failed at user \(sessions.count)"); return
        }
        sessions.append(s)
    }
    var metrics: [[TurnMetrics]] = Array(repeating: [], count: nUsers)

    let benchStart = Date()
    let preStats = engine.poolStats()

    for turnIdx in 0..<userTurns.count {
        let text = userTurns[turnIdx]
        // Simultaneous submit — this is the case multi-slot prefill targets.
        for i in 0..<nUsers {
            metrics[i].append(TurnMetrics(submitTime: Date()))
            let payload = (turnIdx == 0 ? system + text : text)
            if turnIdx == 0 {
                sessions[i].submit(engine.tokenize(payload, addBos: true))
            } else {
                sessions[i].append(engine.tokenize(payload, addBos: false))
            }
        }

        // Pump until every session finishes this turn (.done) or we hit a cap.
        var ticks = 0
        while engine.hasWork {
            _ = engine.tick()
            ticks += 1
            for i in 0..<nUsers {
                while let tok = sessions[i].nextToken() {
                    let now = Date()
                    if metrics[i][turnIdx].firstTokenTime == nil {
                        metrics[i][turnIdx].firstTokenTime = now
                    }
                    metrics[i][turnIdx].lastTokenTime = now
                    metrics[i][turnIdx].tokenCount += 1
                    metrics[i][turnIdx].text += engine.detokenize([tok])
                }
            }
            if ticks > 2000 { print("  tick cap reached on turn \(turnIdx+1)"); break }
        }
        // Pause for next turn so append() reopens.
        for s in sessions { s.pause() }
    }

    let benchEnd = Date()
    let postStats = engine.poolStats()
    let wall = benchEnd.timeIntervalSince(benchStart)

    // Report.
    print("")
    print("  --- per-user × per-turn ---")
    var totalTokens = 0
    for i in 0..<nUsers {
        print("  user \(i + 1):")
        for (t, tm) in metrics[i].enumerated() {
            totalTokens += tm.tokenCount
            let ttft = tm.firstTokenTime.map { $0.timeIntervalSince(tm.submitTime) * 1000 } ?? -1
            let genSecs = (tm.lastTokenTime != nil && tm.firstTokenTime != nil)
                ? tm.lastTokenTime!.timeIntervalSince(tm.firstTokenTime!) : 0
            let tokps = genSecs > 0.001 ? Double(tm.tokenCount) / genSecs : 0
            let preview = tm.text.replacingOccurrences(of: "\n", with: "\\n").prefix(64)
            print(String(format: "    turn %d: TTFT=%.0fms, gen=%d tok @ %.1f tok/s → %@",
                         t + 1, ttft, tm.tokenCount, tokps, String(preview)))
        }
    }
    print("")
    print("  --- aggregate ---")
    let agg = wall > 0.001 ? Double(totalTokens) / wall : 0
    // Peak batched rate: average per-user gen rate × min(nUsers, B). Taken
    // from gen-only time per turn (first→last token interval). This is what
    // the GPU can sustain once everyone is past priming.
    var perUserRates: [Double] = []
    for userM in metrics {
        for tm in userM {
            if let ft = tm.firstTokenTime, let lt = tm.lastTokenTime, tm.tokenCount > 1 {
                let dt = lt.timeIntervalSince(ft)
                if dt > 0.001 { perUserRates.append(Double(tm.tokenCount) / dt) }
            }
        }
    }
    let meanPerUser = perUserRates.isEmpty ? 0 : perUserRates.reduce(0, +) / Double(perUserRates.count)
    let peakBatched = meanPerUser * Double(min(nUsers, B))
    print(String(format: "  wall: %.2fs, tokens: %d", wall, totalTokens))
    print(String(format: "  aggregate: %.1f tok/s   (observed over the whole benchmark, prefill included)", agg))
    print(String(format: "  per-stream: %.1f tok/s  (mean across users × turns, gen-only)", meanPerUser))
    print(String(format: "  peak batched: %.1f tok/s  (per-stream × min(N, B=%d) — what the GPU sustains during pure-gen)", peakBatched, B))
    let headroom = peakBatched > 0.1 ? (1.0 - agg / peakBatched) * 100 : 0
    print(String(format: "  headroom: %.0f%% (the gap between aggregate and peak is prefill wall time; shrinks as prompt:gen ratio falls)", headroom))
    print("  pages: " +
          "used before \(preStats.totalPages - preStats.freePages), " +
          "used after \(postStats.totalPages - postStats.freePages), " +
          "shared content-hashes: \(postStats.cachedHashes - preStats.cachedHashes)")
    print("  note: simultaneous submission → no same-turn cache hits (nothing promoted yet when siblings probe); cache kicks in for later sessions submitting against a fresh engine with the same prefix — see LM_SHARED_PREFIX_DEMO.")

    for s in sessions { engine.closeSession(s) }
}

// ----------------------------------------------------------------------
// Multimodal demo driver — feeds vision-tower output through a session.
//
// STATUS NOTE: generation quality through this path is currently poor.
// The plumbing works end-to-end (vision tower → 272 soft tokens →
// prefill → AR) and all engine-level invariants hold, but the model
// produces degenerate output regardless of (a) BOI/EOI marker wrapping,
// (b) padding to image_seq_length=280, or (c) soft-token magnitude
// scaling. Remaining hypotheses:
//   - Our Gemma-4 vision tower has a subtle numerical mismatch with
//     HF's reference image_features (weight loading, layer ordering,
//     or pre-RMS normalization). Confirming requires a side-by-side
//     run against transformers' Gemma4ForConditionalGeneration on the
//     same preprocessed image.
//   - The HF-compatible scatter-after-embed path (which differs from
//     our skipEmbed injection only if soft tokens don't match image_
//     features byte-for-byte) may reveal the divergence point.
// Both are deferred to a focused debugging session with Python HF
// reference access; the infrastructure below (count padding, scale
// factor, skipEmbed prefill) is in place to make that iteration quick.
//
// Env vars:
//   GGUF_PATH=<gguf>                 # LM weights
//   VISION_ST=<safetensors path>     # vision tower weights
//   LM_MULTIMODAL=<image png path>   # image to embed and prompt with
//   [LM_MULTIMODAL_PREFIX="…"]       # text before image tokens
//   [LM_MULTIMODAL_SUFFIX="…"]       # text after image tokens
//   [LM_MULTIMODAL_MAX=48]           # generation budget
//   [LM_MM_NO_PAD=1]                 # disable 280 padding (run raw count)
//   [LM_MM_SOFT_SCALE=<f>]           # soft-token magnitude scale (probe tool)
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

    // LM_MM_FORCE_SOFTS=<path.bin> — skip the vision tower and feed the
    // given binary as soft-token input. Binary is fp16 row-major [N, HIDDEN].
    // Use for isolating LM-side multimodal bugs from vision-tower bugs:
    // dump HF's reference softs for a frame, feed them here, and if output
    // is STILL garbage the bug is downstream in LM (not in vision).
    let forceSoftsPath = ProcessInfo.processInfo.environment["LM_MM_FORCE_SOFTS"]
    let rawSoftTokens: MTLBuffer
    let rawNPooled: Int
    let forceIsFp32: Bool
    if let path = forceSoftsPath,
       let data = try? Data(contentsOf: URL(fileURLWithPath: path)) {
        let bytesPerElem = 2   // fp16
        let elements = data.count / bytesPerElem
        precondition(elements % HIDDEN == 0, "force-softs file size must be a multiple of HIDDEN*2 bytes")
        let n = elements / HIDDEN
        let buf = device.makeBuffer(length: data.count, options: .storageModeShared)!
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress, data.count)
        }
        rawSoftTokens = buf
        rawNPooled = n
        forceIsFp32 = false
        print("  LM_MM_FORCE_SOFTS: skipping vision tower, loaded \(n) fp16 soft tokens from \(path)")
    } else {
        forceIsFp32 = true
        let (st, np) = runVisionTowerForward(batch: batch, weights: visWeights,
                                              device: device, queue: queue)
        rawSoftTokens = st; rawNPooled = np
    }
    if rawNPooled == 0 {
        print("  0 soft tokens — image too small? aborting"); return
    }

    // Pad/truncate to Gemma-4's image_seq_length = 280. Default ON. Set
    // LM_MM_NO_PAD=1 to run at the raw vision-tower count for comparison.
    let targetSoft = 280
    let noPad = ProcessInfo.processInfo.environment["LM_MM_NO_PAD"] != nil
    // Scale factor applied to soft-token magnitudes. Vision-tower raw
    // values measured ±1–13 while post-embed-scale text tokens sit ±1–3.
    // If the imbalance causes image positions to dominate attention
    // softmax, scaling corrects it. LM_MM_SOFT_SCALE=<float> to override;
    // default 1.0 (no scaling).
    let softScale: Float = Float(ProcessInfo.processInfo.environment["LM_MM_SOFT_SCALE"] ?? "") ?? 1.0
    // With LM_MM_FORCE_SOFTS the raw buffer is fp16; otherwise (vision
    // tower path) it's fp32. Pipe through the padding/scale helpers accordingly.
    let softTokens: MTLBuffer
    let nPooled: Int
    let softTokensIsFp32: Bool
    if noPad && softScale == 1.0 {
        softTokens = rawSoftTokens; nPooled = rawNPooled
        softTokensIsFp32 = forceIsFp32
        print("  soft tokens: \(nPooled) (isFp32=\(forceIsFp32); padding + scale skipped)")
    } else {
        let outN = noPad ? rawNPooled : targetSoft
        let bytesPerElem = forceIsFp32 ? 4 : 2
        let buf = device.makeBuffer(length: outN * HIDDEN * bytesPerElem, options: .storageModeShared)!
        memset(buf.contents(), 0, buf.length)
        let copyRows = min(rawNPooled, outN)
        if softScale == 1.0 {
            memcpy(buf.contents(), rawSoftTokens.contents(), copyRows * HIDDEN * bytesPerElem)
        } else if forceIsFp32 {
            let src = rawSoftTokens.contents().assumingMemoryBound(to: Float.self)
            let dst = buf.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<(copyRows * HIDDEN) { dst[i] = src[i] * softScale }
        } else {
            let src = rawSoftTokens.contents().assumingMemoryBound(to: Float16.self)
            let dst = buf.contents().assumingMemoryBound(to: Float16.self)
            for i in 0..<(copyRows * HIDDEN) { dst[i] = Float16(Float(src[i]) * softScale) }
        }
        softTokens = buf
        nPooled = outN
        softTokensIsFp32 = forceIsFp32
        print(String(format: "  soft tokens: %d raw → %d (isFp32=%@, scale=%.3f)",
                     rawNPooled, outN, forceIsFp32 ? "true" : "false", softScale))
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
    sess.submit(softTokens: softTokens, count: nPooled, isFp32: softTokensIsFp32)
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

