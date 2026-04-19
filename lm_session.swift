// LmSession — the real "prompt string → generated tokens" inference API.
//
// What belongs here: request lifecycle + scheduling. The per-layer kernel
// plumbing lives in buildStepCB (AR) and buildPrefillCB (prefill); this
// file just picks which one to call, wires the state transitions between
// them, and returns tokens.
//
// Intentionally minimal for v1: all B slots run the same prompt (so B=4
// gives 4× throughput of effectively the same request, which is useful
// for benchmarking but not yet for multi-tenant serving). The per-slot
// buffers and block_table are already disjoint, so the next revision can
// feed B different prompts by populating input_tokens + q_positions per
// slot before commit.
//
// Timing instrumentation is first-class: every public entry point emits
// a TimingRecord so callers can reason about prefill vs AR latency without
// wrapping their own clocks.
import Foundation

struct TimingRecord {
    var prefillMs: Double = 0      // wall of buildPrefillCB commit+wait, 0 if no prefill
    var arMs: Double = 0           // wall of AR steps (sum)
    var arStepCount: Int = 0
    var promptTokens: Int = 0      // tokens in prompt after tokenization
    var generatedTokens: Int = 0   // tokens actually generated (≤ maxNewTokens)
    var hitEos: Bool = false
    var tokenizeMs: Double = 0
    var detokenizeMs: Double = 0
}

final class LmSession {
    let weights: LmWeights
    let tokenizer: GemmaBpe
    // The currently-running request's output per slot. Parallel to B; slot 0
    // is the primary one we stream/decode. Cleared on each generate() call.
    private(set) var generatedPerSlot: [[UInt32]] = Array(repeating: [], count: B)

    init(weights: LmWeights) {
        self.weights = weights
        self.tokenizer = GemmaBpe(weights: weights)
    }

    // Primary entry point. Tokenizes, prefills as much as fits in one tile,
    // then AR-decodes until EOS or maxNewTokens. `stream` is called with
    // (tokenId, decodedFragment) on each generated token from slot 0. Return
    // false from `stream` to stop generation early.
    //
    // Returns slot-0 generated token IDs (prompt tokens NOT included). Use
    // `tokenizer.decode(result)` to turn them back into a string, or read
    // `generatedPerSlot` for all B slots.
    @discardableResult
    func generate(prompt: String,
                  maxNewTokens: Int = 64,
                  eos: UInt32? = nil,
                  stream: ((UInt32, String) -> Bool)? = nil) -> (tokens: [UInt32], timing: TimingRecord) {
        var tr = TimingRecord()
        let eosId = eos ?? weights.eosTokenId

        // ---- Tokenize ----
        let t0 = Date()
        let addBosOverride = ProcessInfo.processInfo.environment["LM_ADD_BOS"]
            .flatMap { ["0", "false", "no"].contains($0.lowercased()) ? false : true }
        var promptTokens = tokenizer.encode(prompt, addBos: addBosOverride)
        tr.tokenizeMs = Date().timeIntervalSince(t0) * 1000
        tr.promptTokens = promptTokens.count
        precondition(!promptTokens.isEmpty, "empty prompt after tokenization")

        // Reset per-slot output log.
        generatedPerSlot = Array(repeating: [], count: B)

        // ---- Prime AR state with prompt ----
        // Path A (prompt ≤ MAX_Q_LEN):  single-tile prefill, then AR.
        // Path B (prompt >  MAX_Q_LEN): prefill first MAX_Q_LEN tokens, then
        //   teacher-force the remainder one AR step at a time. Multi-tile
        //   prefill isn't wired yet (precomputeFlexPrefillMasks asserts
        //   qBlocks==1), so we lean on AR for the long-prompt tail. This is
        //   slower than prefill but still correct.
        var positionAfterPrime: Int
        if promptTokens.count <= MAX_Q_LEN && promptTokens.count > 1 {
            positionAfterPrime = runPrefillTile(tokens: promptTokens, tr: &tr)
        } else if promptTokens.count == 1 {
            // Single token: nothing to prefill, just run a single AR step
            // after init to populate position 0's KV and produce a logit.
            initLmState(bos: promptTokens[0])
            let cb = buildStepCB(weights); cb.commit(); cb.waitUntilCompleted()
            if let err = cb.error { print("  GPU step 0: \(err)"); return (tokens: [], timing: tr) }
            positionAfterPrime = 0
        } else {
            // Long prompt: prefill MAX_Q_LEN, AR-prime the tail.
            let prefix = Array(promptTokens.prefix(MAX_Q_LEN))
            positionAfterPrime = runPrefillTile(tokens: prefix, tr: &tr)
            // AR-prime tokens MAX_Q_LEN..promptTokens.count-1.
            positionAfterPrime = arPrimeTail(
                promptTokens: promptTokens,
                startIndex: MAX_Q_LEN,
                positionAfterPrefill: positionAfterPrime,
                tr: &tr)
        }

        // ---- Generate up to maxNewTokens ----
        for _ in 0..<maxNewTokens {
            // The most-recent buildStepCB / buildPrefillCB run already left
            // its logits in `logits` (AR) or `pre_logits` (prefill). In the
            // prefill path, runPrefillTile copied the last-position slot-0
            // logits into `logits` via a blit so the AR path can read
            // uniformly.
            let sampled = greedyArgmaxPerSlot(w: weights)
            for b in 0..<B { generatedPerSlot[b].append(sampled[b]) }
            tr.generatedTokens += 1

            let slot0Tok = sampled[0]
            if slot0Tok == eosId { tr.hitEos = true; break }

            // Stream callback for slot 0.
            if let stream = stream {
                let frag = tokenizer.decode([slot0Tok])
                if !stream(slot0Tok, frag) { break }
            }

            // Advance AR state with sampled tokens (per-slot — each slot
            // picks its own sample, even though in v1 they all saw the same
            // prompt, so their samples will be argmax-identical until FP
            // drift diverges them).
            let tAr = Date()
            advanceLmState(nextTokens: sampled)
            let cb = buildStepCB(weights); cb.commit(); cb.waitUntilCompleted()
            tr.arMs += Date().timeIntervalSince(tAr) * 1000
            tr.arStepCount += 1
            if let err = cb.error { print("  GPU AR step: \(err)"); break }
        }

        return (tokens: generatedPerSlot[0], timing: tr)
    }

    // ---- Internal: single-tile prefill ----
    // Writes `tokens` to all B slots, builds the prefill CB, copies slot-0's
    // last-position logits into `logits` (so AR can pick up seamlessly),
    // and seeds AR state such that advanceLmState will correctly extend.
    // Returns the absolute position of the last prompt token written (qLen-1).
    private func runPrefillTile(tokens: [UInt32], tr: inout TimingRecord) -> Int {
        let qLen = tokens.count
        precondition(qLen <= MAX_Q_LEN)

        // Populate per-slot prompt in pre_input_tokens + pre_q_positions.
        let tokP = pre_input_tokens.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
        let posP = pre_q_positions.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
        for b in 0..<B {
            for i in 0..<qLen {
                tokP[b * qLen + i] = tokens[i]
                posP[b * qLen + i] = UInt32(i)
            }
        }
        let klsP = pre_k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let klfP = pre_k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
        for b in 0..<B { klsP[b] = UInt32(qLen); klfP[b] = UInt32(qLen) }

        // Per-slot-disjoint block_table.
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        for b in 0..<B {
            for p in 0..<MAX_PAGES_PER_SLOT {
                btP[b * MAX_PAGES_PER_SLOT + p] = UInt32(b * MAX_PAGES_PER_SLOT + p) % UInt32(TOTAL_PAGES)
            }
        }

        precomputeFlexPrefillMasks(qLen: qLen, positionStart: 0)
        let cb = buildPrefillCB(weights, qLen: qLen)
        let t0 = Date()
        cb.commit(); cb.waitUntilCompleted()
        tr.prefillMs = Date().timeIntervalSince(t0) * 1000
        if let err = cb.error { print("  GPU prefill: \(err)") }

        // Seed AR state so the NEXT buildStepCB writes at absolute position
        // qLen. advanceLmState increments positions by 1, so pre-seed to
        // qLen-1 with k_len=qLen (num_pages match prefill's final state).
        let arPosP = positions.contents().bindMemory(to: UInt32.self, capacity: B)
        let arNpsP = num_pages_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let arNpfP = num_pages_full.contents().bindMemory(to: UInt32.self, capacity: B)
        let arKlsP = k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let arKlfP = k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
        for b in 0..<B {
            arPosP[b] = UInt32(qLen - 1)
            arKlsP[b] = UInt32(qLen); arKlfP[b] = UInt32(qLen)
            arNpsP[b] = UInt32((qLen + PAGE_SLIDE - 1) / PAGE_SLIDE)
            arNpfP[b] = UInt32((qLen + PAGE_FULL  - 1) / PAGE_FULL)
        }

        // Copy slot-0's last-position logits (row b=0, q=qLen-1) from
        // pre_logits into the AR `logits` buffer so greedyArgmaxPerSlot
        // sees the prefill output. Also copy slot-0's logits to all B rows
        // of `logits` since for a single shared prompt they all agree.
        let srcPtr = pre_logits.contents().assumingMemoryBound(to: Float16.self)
        let dstPtr = logits.contents().assumingMemoryBound(to: Float16.self)
        // Prefill laid out pre_logits as [b*qLen + i, VOCAB]. For each slot
        // b ∈ [0, B), the last-position row is at (b*qLen + qLen-1)*VOCAB.
        for b in 0..<B {
            let src = srcPtr.advanced(by: (b * qLen + (qLen - 1)) * VOCAB)
            let dst = dstPtr.advanced(by: b * VOCAB)
            memcpy(dst, src, VOCAB * 2)
        }

        return qLen - 1
    }

    // ---- Internal: AR-prime a long-prompt tail ----
    // For prompts longer than MAX_Q_LEN, runPrefillTile handles the first
    // MAX_Q_LEN tokens; this function teacher-forces the remainder one AR
    // step at a time, advancing state + committing one buildStepCB per
    // token. Returns the absolute position of the last token primed.
    private func arPrimeTail(promptTokens: [UInt32], startIndex: Int,
                              positionAfterPrefill: Int, tr: inout TimingRecord) -> Int {
        var lastPos = positionAfterPrefill
        for i in startIndex..<promptTokens.count {
            let tok = promptTokens[i]
            let toks = [UInt32](repeating: tok, count: B)
            let tAr = Date()
            advanceLmState(nextTokens: toks)
            let cb = buildStepCB(weights); cb.commit(); cb.waitUntilCompleted()
            tr.arMs += Date().timeIntervalSince(tAr) * 1000
            tr.arStepCount += 1
            if let err = cb.error { print("  GPU AR prime step \(i): \(err)"); return lastPos }
            lastPos = i
        }
        return lastPos
    }
}

// ---- Env-var driver: end-to-end generation ----
// LM_GENERATE="prompt string"  GGUF_PATH=<gguf>  [LM_GENERATE_MAX=64]
//   [LM_GENERATE_EOS=106]
// Loads weights, builds a session, generates up to LM_GENERATE_MAX tokens,
// streams each one to stdout as it arrives.
func runLmGenerate(ggufPath: String, prompt: String,
                    maxNewTokens: Int, eos: UInt32?) {
    print("\n=== LM generate ===")
    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    print("")

    let session = LmSession(weights: w)
    let addBosEnv = ProcessInfo.processInfo.environment["LM_ADD_BOS"]
        .flatMap { ["0", "false", "no"].contains($0.lowercased()) ? false : true }
    print("  prompt (raw): \(prompt.debugDescription)")
    let encoded = session.tokenizer.encode(prompt, addBos: addBosEnv)
    print("  prompt (tokens, \(encoded.count)): \(encoded.prefix(16))\(encoded.count > 16 ? " …" : "")")
    let decoded = session.tokenizer.decode(encoded)
    print("  prompt (round-trip): \(decoded.debugDescription)")
    print("  eos_id: \(eos ?? w.eosTokenId), max_new: \(maxNewTokens), add_bos: \(addBosEnv ?? w.addBosToken)")
    print("")

    print("  --- generation (slot 0) ---")
    print("  ", terminator: "")
    print(prompt.replacingOccurrences(of: "\n", with: "\\n"), terminator: "")
    fflush(stdout)

    let (tokens, timing) = session.generate(
        prompt: prompt,
        maxNewTokens: maxNewTokens,
        eos: eos,
        stream: { _, frag in
            print(frag.replacingOccurrences(of: "\n", with: "\\n"), terminator: "")
            fflush(stdout)
            return true
        })
    print("")   // end-of-generation newline
    print("")

    print("  --- timing ---")
    print(String(format: "  tokenize       : %.2f ms", timing.tokenizeMs))
    print(String(format: "  prefill CB     : %.2f ms (%d prompt tokens)", timing.prefillMs, timing.promptTokens))
    print(String(format: "  AR total       : %.2f ms (%d steps)", timing.arMs, timing.arStepCount))
    if timing.arStepCount > 0 {
        print(String(format: "  AR per-step    : %.2f ms (%.1f tok/s slot, %.1f tok/s batch)",
                     timing.arMs / Double(timing.arStepCount),
                     1000.0 / (timing.arMs / Double(timing.arStepCount)),
                     Double(B) * 1000.0 / (timing.arMs / Double(timing.arStepCount))))
    }
    print(String(format: "  generated      : %d tokens, hit_eos=%@", timing.generatedTokens, timing.hitEos ? "yes" : "no"))

    // Full per-slot dump.
    print("\n  --- all B slots final output ---")
    for b in 0..<B {
        let slotToks = session.generatedPerSlot[b]
        let slotText = session.tokenizer.decode(slotToks, skipSpecial: false)
        print("  slot \(b) (\(slotToks.count) toks): \(slotText.debugDescription)")
    }
    _ = tokens  // silence unused
}
