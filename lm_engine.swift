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

final class Session {
    let id: Int
    let slot: Int
    let eosId: UInt32
    var maxNewTokens: Int
    fileprivate weak var engine: LmEngine?

    fileprivate(set) var state: SessionState = .idle
    // Tokens queued to be teacher-forced (prompt prefix or tool-result
    // continuations). Consumed one per step while state == .priming.
    fileprivate var primingQueue: [UInt32] = []
    // When state == .generating, the token we sampled on the previous step
    // becomes the next step's input. Kept separate from primingQueue so
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

    // Queue more input tokens to be teacher-forced on subsequent steps.
    // Valid in any state: calling during .generating flips back to .priming,
    // which is how tool-call continuations re-enter the stream.
    func submit(_ tokens: [UInt32]) {
        guard !tokens.isEmpty else { return }
        primingQueue.append(contentsOf: tokens)
        if state == .idle || state == .generating { state = .priming }
    }

    // Convenience: tokenize and submit.
    func submit(text: String, addBos: Bool? = nil) {
        guard let eng = engine else { return }
        submit(eng.tokenizer.encode(text, addBos: addBos))
    }

    // Pull the next generated token, or nil if none ready. Caller should
    // call `engine.step()` to make progress.
    func nextToken() -> UInt32? {
        guard !outputQueue.isEmpty else { return nil }
        return outputQueue.removeFirst()
    }

    // How many tokens are ready to consume.
    var pendingOutputCount: Int { outputQueue.count }

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

    // Run exactly one buildStepCB covering every slot, with per-slot state
    // driven by each session's queue. Returns the number of tokens emitted
    // into output queues this step (across all sessions).
    @discardableResult
    func step() -> Int {
        let busy = activeSessions.filter { $0.state.isBusy }
        if busy.isEmpty { return 0 }

        // --- Prepare per-slot inputs ---
        let tokP = input_tokens.contents().bindMemory(to: UInt32.self, capacity: B)
        let posP = positions.contents().bindMemory(to: UInt32.self, capacity: B)
        let klsP = k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let klfP = k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
        let npsP = num_pages_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let npfP = num_pages_full.contents().bindMemory(to: UInt32.self, capacity: B)

        for slot in 0..<B {
            if let s = sessionBySlot[slot], s.state.isBusy {
                // What token goes in this step?
                let inputTok: UInt32
                if s.state == .priming {
                    inputTok = s.primingQueue.removeFirst()
                } else {
                    inputTok = s.nextGeneratedInput
                }
                tokP[slot] = inputTok
                posP[slot] = UInt32(s.position)
                let kLen = s.position + 1
                klsP[slot] = UInt32(kLen)
                klfP[slot] = UInt32(kLen)
                npsP[slot] = UInt32((kLen + PAGE_SLIDE - 1) / PAGE_SLIDE)
                npfP[slot] = UInt32((kLen + PAGE_FULL  - 1) / PAGE_FULL)
            } else {
                // No session or session is done: park this slot at position
                // 0 feeding BOS. Its KV write lands in its own dedicated
                // page[0] and can't disturb other slots. Wasted compute,
                // but correct.
                tokP[slot] = weights.bosTokenId
                posP[slot] = 0
                klsP[slot] = 1
                klfP[slot] = 1
                npsP[slot] = 1
                npfP[slot] = 1
            }
        }

        // Flex mask precompute — reads positions/k_len we just wrote.
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

        // --- Distribute outputs ---
        let logP = logits.contents().assumingMemoryBound(to: Float16.self)
        var emitted = 0
        for s in busy {
            // Greedy argmax per-slot — picks the next token this slot would
            // sample given its logit.
            let base = s.slot * VOCAB
            var bestI = 0; var bestV: Float = -.infinity
            for v in 0..<VOCAB {
                let x = Float(logP[base + v])
                if x > bestV { bestV = x; bestI = v }
            }
            let sampled = UInt32(bestI)

            // Advance position: we just wrote at s.position, next step
            // writes at s.position + 1.
            s.position += 1

            if s.state == .priming {
                // If that was the last priming token, the logit we just
                // computed IS the first generated token's prediction —
                // sample it immediately and flip to .generating so the
                // caller sees no stall.
                if s.primingQueue.isEmpty {
                    s.state = .generating
                    s.outputQueue.append(sampled)
                    s.nextGeneratedInput = sampled
                    s.numGenerated += 1; emitted += 1; totalTokensGenerated += 1
                    if sampled == s.eosId || s.numGenerated >= s.maxNewTokens {
                        s.state = .done
                    }
                }
                // else: still priming; discard logit, feed next queue token
                // on the next step.
            } else {
                // .generating: this is the next output token.
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

    // Pump the scheduler until all sessions hit .done or a budget elapses.
    // Returns the total tokens emitted across all sessions during the run.
    @discardableResult
    func runUntilIdle(maxSteps: Int = 10_000) -> Int {
        var emitted = 0
        for _ in 0..<maxSteps {
            if !hasWork { break }
            emitted += step()
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
        _ = engine.step()
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
