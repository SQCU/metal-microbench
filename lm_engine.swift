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
    case idle           // no pending work
    case priming        // primingQueue is non-empty; teacher-forcing
    case generating     // sampling; pushing to generated queue
    case done           // EOS / maxTokens / explicit close

    var isBusy: Bool {
        switch self {
        case .priming, .generating: return true
        default: return false
        }
    }
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
    // is [count, HIDDEN] and the storage dtype is either fp16 or fp32.
    // Prefill copies these into pre_hidden and skips embed_lookup +
    // embed-scale (the vision projection already did both).
    case softTokens(buffer: MTLBuffer, count: Int, isFp32: Bool)

    var count: Int {
        switch self {
        case .tokens(let ts): return ts.count
        case .softTokens(_, let c, _): return c
        }
    }
}

final class Session {
    let id: Int
    let slot: Int
    let eosId: UInt32
    var maxNewTokens: Int
    fileprivate weak var engine: LmEngine?

    fileprivate(set) var state: SessionState = .idle
    // Ordered chunks to teacher-force (text tokens or image soft tokens).
    // The scheduler pops the front chunk and dispatches it — either as a
    // fast prefill (whole chunk at once) or as token-by-token AR priming,
    // depending on scheduler mode and concurrent session load.
    fileprivate var chunkQueue: [PrimingChunk] = []
    // When state == .generating, the token we sampled on the previous step
    // becomes the next step's input. Kept separate from the chunk queue so
    // the state machine doesn't have to pun inputs.
    fileprivate var nextGeneratedInput: UInt32 = 0
    // Next KV-cache write position. k_len after a step == position + 1.
    fileprivate var position: Int = 0
    fileprivate var numGenerated: Int = 0

    // Tokens the caller can consume. Generated-only (prompt tokens are not
    // echoed). Caller pulls via `nextToken()`.
    fileprivate var outputQueue: [UInt32] = []

    fileprivate init(id: Int, slot: Int, eosId: UInt32, maxNewTokens: Int, engine: LmEngine) {
        self.id = id; self.slot = slot
        self.eosId = eosId; self.maxNewTokens = maxNewTokens
        self.engine = engine
    }

    // Queue more input tokens to be teacher-forced. Valid in any state:
    // calling during .generating flips back to .priming, which is how
    // tool-call continuations re-enter the stream.
    func submit(_ tokens: [UInt32]) {
        guard !tokens.isEmpty else { return }
        chunkQueue.append(.tokens(tokens))
        if state == .idle || state == .generating { state = .priming }
    }

    // Convenience: tokenize and submit.
    func submit(text: String, addBos: Bool? = nil) {
        guard let eng = engine else { return }
        submit(eng.tokenizer.encode(text, addBos: addBos))
    }

    // Queue a pre-computed block of vision-tower soft tokens (shape
    // [count, HIDDEN], already projected into the text hidden space).
    // `isFp32` distinguishes the vision tower's default fp32 output from
    // a caller who's already downcast to fp16.
    func submit(softTokens: MTLBuffer, count: Int, isFp32: Bool) {
        guard count > 0 else { return }
        chunkQueue.append(.softTokens(buffer: softTokens, count: count, isFp32: isFp32))
        if state == .idle || state == .generating { state = .priming }
    }

    // Pull the next generated token, or nil if none ready. Caller should
    // call `engine.step()` to make progress.
    func nextToken() -> UInt32? {
        guard !outputQueue.isEmpty else { return nil }
        return outputQueue.removeFirst()
    }

    // How many tokens are ready to consume.
    var pendingOutputCount: Int { outputQueue.count }
    // Sum of all queued priming work (for scheduler heuristics).
    var pendingPrimingCount: Int { chunkQueue.reduce(0) { $0 + $1.count } }

    // Mark as done; engine will not schedule any more steps for this session.
    func finish() { state = .done }
}

final class LmEngine {
    let weights: LmWeights
    let tokenizer: GemmaBpe
    // Slot 0..B-1 is either nil (free) or owns exactly one Session.
    private var sessionBySlot: [Session?]
    private var nextId: Int = 1

    // Instrumentation — helps the scheduler-behaviour tests measure how
    // well we're batching. One CB per step regardless of active count.
    private(set) var totalSteps: Int = 0
    private(set) var totalTokensGenerated: Int = 0
    private(set) var lastStepMs: Double = 0

    init(weights: LmWeights) {
        self.weights = weights
        self.tokenizer = GemmaBpe(weights: weights)
        self.sessionBySlot = Array(repeating: nil, count: B)
    }

    // Open a new session on the first free slot. Returns nil if the engine
    // is at capacity (B active sessions). Caller owns the Session and must
    // call `closeSession` to free the slot.
    func openSession(eosId: UInt32? = nil, maxNewTokens: Int = 128) -> Session? {
        for slot in 0..<B where sessionBySlot[slot] == nil {
            let s = Session(id: nextId, slot: slot,
                            eosId: eosId ?? weights.eosTokenId,
                            maxNewTokens: maxNewTokens, engine: self)
            nextId += 1
            sessionBySlot[slot] = s
            // Prime per-slot block_table and reset the slot's KV pages to
            // its own dedicated strip. Each session gets pages
            // [slot*MAX_PAGES_PER_SLOT, (slot+1)*MAX_PAGES_PER_SLOT) —
            // disjoint from every other slot's strip.
            let btP = block_table.contents().bindMemory(to: UInt32.self,
                        capacity: B * MAX_PAGES_PER_SLOT)
            for p in 0..<MAX_PAGES_PER_SLOT {
                btP[slot * MAX_PAGES_PER_SLOT + p] =
                    UInt32(slot * MAX_PAGES_PER_SLOT + p) % UInt32(TOTAL_PAGES)
            }
            return s
        }
        return nil
    }

    // Close a session and free its slot. The KV strip stays allocated (it
    // will get overwritten when a new session lands on this slot).
    func closeSession(_ s: Session) {
        guard sessionBySlot[s.slot]?.id == s.id else { return }
        s.state = .done
        sessionBySlot[s.slot] = nil
    }

    var activeSessions: [Session] { sessionBySlot.compactMap { $0 } }
    var hasWork: Bool { activeSessions.contains { $0.state.isBusy } }

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
        let busy = activeSessions.filter { $0.state.isBusy }
        if busy.isEmpty { return 0 }

        // Per-slot inputs (AR path).
        let tokP = input_tokens.contents().bindMemory(to: UInt32.self, capacity: B)
        let posP = positions.contents().bindMemory(to: UInt32.self, capacity: B)
        let klsP = k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let klfP = k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
        let npsP = num_pages_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let npfP = num_pages_full.contents().bindMemory(to: UInt32.self, capacity: B)

        // Track which slots run REAL work this step — used downstream when
        // we read logits to decide whether to emit a sampled token.
        var realSlot = [Bool](repeating: false, count: B)

        for slot in 0..<B {
            if let s = sessionBySlot[slot], s.state.isBusy {
                let inputTok: UInt32?
                if s.state == .priming {
                    inputTok = popArPrimingToken(s)
                    // If nil, head chunk is .softTokens — can't AR-prime
                    // here. Park this slot; caller should run fast prefill.
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
            // Park: BOS at position 0, k_len=1, 1 page. Writes land in the
            // slot's own dedicated page strip and can't disturb any other
            // session's KV.
            tokP[slot] = weights.bosTokenId
            posP[slot] = 0
            klsP[slot] = 1; klfP[slot] = 1
            npsP[slot] = 1; npfP[slot] = 1
        }

        if USE_FLEX_ATTN {
            precomputeFlexBlockMaskSlide(slidingWindow: SLIDING_WINDOW)
            precomputeFlexBlockMaskFull()
        }

        let t0 = Date()
        let cb = buildStepCB(weights)
        cb.commit(); cb.waitUntilCompleted()
        lastStepMs = Date().timeIntervalSince(t0) * 1000
        totalSteps += 1
        if let err = cb.error { print("  GPU step error: \(err)"); return 0 }

        let logP = logits.contents().assumingMemoryBound(to: Float16.self)
        var emitted = 0
        for slot in 0..<B where realSlot[slot] {
            guard let s = sessionBySlot[slot], s.state.isBusy else { continue }
            let base = s.slot * VOCAB
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
        let qLen = head.count
        precondition(qLen >= 1)
        // Current prefill kernel supports Q_BLOCK=8 (one tile). If the chunk
        // is larger, process the first MAX_Q_LEN tokens here and leave the
        // tail for subsequent calls.
        let thisTile = min(qLen, MAX_Q_LEN)
        let remaining = qLen - thisTile
        let positionStart = s.position

        // --- Save block_table; redirect non-target slots to scratch strip ---
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        var savedBT = [UInt32](repeating: 0, count: B * MAX_PAGES_PER_SLOT)
        for i in 0..<(B * MAX_PAGES_PER_SLOT) { savedBT[i] = btP[i] }
        for slot in 0..<B where slot != s.slot {
            for p in 0..<MAX_PAGES_PER_SLOT {
                // All silenced slots redirect to the single scratch strip;
                // their KV writes race but nobody reads the outputs.
                btP[slot * MAX_PAGES_PER_SLOT + p] = UInt32(SCRATCH_PAGE_BASE + p)
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
                tokP[s.slot * thisTile + i] = ts[i]
            }
        case let .softTokens(buf, _, isFp32):
            // Vision tower output. Copy softTokens[0..thisTile*HIDDEN]
            // into pre_hidden at rows [s.slot*thisTile, s.slot*thisTile+thisTile),
            // converting fp32→fp16 if needed. Other slots' rows are left as
            // whatever was there; they'll compute junk that gets discarded.
            let dstPtr = pre_hidden.contents().assumingMemoryBound(to: Float16.self)
            let dstBase = (s.slot * thisTile) * HIDDEN
            if isFp32 {
                let srcPtr = buf.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<(thisTile * HIDDEN) {
                    dstPtr[dstBase + i] = Float16(srcPtr[i])
                }
            } else {
                let srcPtr = buf.contents().assumingMemoryBound(to: Float16.self)
                memcpy(dstPtr.advanced(by: dstBase), srcPtr, thisTile * HIDDEN * 2)
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
        let srcPtr = pre_logits.contents().assumingMemoryBound(to: Float16.self)
        let dstPtr = logits.contents().assumingMemoryBound(to: Float16.self)
        let src = srcPtr.advanced(by: (s.slot * thisTile + (thisTile - 1)) * VOCAB)
        let dst = dstPtr.advanced(by: s.slot * VOCAB)
        memcpy(dst, src, VOCAB * 2)

        // Pop / trim the head chunk.
        switch head {
        case .tokens(var ts):
            ts.removeFirst(thisTile)
            if ts.isEmpty { s.chunkQueue.removeFirst() }
            else          { s.chunkQueue[0] = .tokens(ts) }
        case let .softTokens(_, count, _):
            if remaining == 0 {
                s.chunkQueue.removeFirst()
            } else {
                // Not yet supported: image chunks larger than MAX_Q_LEN
                // would need sub-offset bookkeeping. Vision tower output
                // for 224×224 Gemma-4 images is 256 soft tokens — well
                // beyond the single-tile budget — so this is flagged for
                // the next iteration along with multi-tile prefill.
                print("  WARN: soft-tokens chunk (\(count)) > MAX_Q_LEN=\(MAX_Q_LEN); partial prefill dropping the tail")
                s.chunkQueue.removeFirst()
            }
        }

        // If the queue is now empty, sample the post-prefill logit as the
        // first generated token so callers don't see a stall step.
        if s.chunkQueue.isEmpty {
            let base = s.slot * VOCAB
            let logP = logits.contents().assumingMemoryBound(to: Float16.self)
            var bestI = 0; var bestV: Float = -.infinity
            for v in 0..<VOCAB {
                let x = Float(logP[base + v])
                if x > bestV { bestV = x; bestI = v }
            }
            let sampled = UInt32(bestI)
            s.state = .generating
            s.outputQueue.append(sampled)
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
// For convenience, LM_MULTISESSION="text1§text2§text3" (section-sign
// separator, U+00A7) still works as a fallback since that character
// doesn't appear in the Gemma chat template.
// ----------------------------------------------------------------------
func runLmMultisession(ggufPath: String, promptsStr: String, maxNewPerSession: Int) {
    print("\n=== LM multisession engine demo ===")
    var prompts: [String] = []
    // Numbered env-var path first (the robust one for chat templates).
    for i in 1...B {
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
    if prompts.count > B {
        print("  warning: \(prompts.count) prompts provided; engine capacity is B=\(B), extras truncated")
    }
    let active = Array(prompts.prefix(B))

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
        print("  session \(s.id) on slot \(s.slot): prompt=\(p.debugDescription) (\(toks.count) tokens)")
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
        print("  s\(s.id) (slot \(s.slot), state=\(s.state)): \(allOutputs[s.id]!.debugDescription)")
        engine.closeSession(s)
    }
}
