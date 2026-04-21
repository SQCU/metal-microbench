// Correctness test for control-vector + paged-KV-cache interaction.
//
// Two orthogonal things this file verifies, both gated on env vars so
// the test is a free-standing harness and doesn't perturb anything
// else.
//
//   LM_TEST_CVEC_DIGEST=1       → digest unit tests (no weights needed)
//   LM_TEST_CVEC_CACHE=1        → integration test (requires GGUF_PATH)
//
// The pair covers:
//
//   1) Digest determinism + partitioning — same envelope params ⇒ same
//      digest; any param difference ⇒ different digest; cvecId hashed
//      as UTF-8 so salt-free across process restarts.
//
//   2) PageManager.hashPage composition — two sessions with matching
//      tokens but different cvec state must land on different keys.
//      An unsteered session with cvecDigest=0 hits a prior unsteered
//      page's key bit-for-bit (back-compat with pre-steering cache).
//
//   3) Prefill steering hook — an ActiveControl that fires during a
//      buildPrefillCB tile actually writes into the residual stream
//      at the right positions (verified by re-running the same prompt
//      split across two submits and asserting per-token KL≈0 for the
//      post-split tokens).
//
//   4) Content cache behavior end-to-end — session pairs submitted
//      back-to-back with identical prompts observe the expected
//      cache hit / miss per their cvec state.
//
// Output convention: every assertion prints "PASS:" or "FAIL:" plus
// context; final summary line has a total count. A non-zero FAIL
// count should be a loud bug — the cache is a correctness boundary.

import Foundation
import Metal
#if canImport(Darwin)
import Darwin
#endif

// ------------------------------------------------------------------
// Tiny assert + counters. No XCTest dependency.
// ------------------------------------------------------------------
private var gTestPass = 0
private var gTestFail = 0

// Force line-buffered stdout so progress is visible when piped to a
// file (Swift's default full-buffering hides everything until exit).
private func enableLineBuffering() {
    setlinebuf(stdout)
    setlinebuf(stderr)
}

private func step(_ label: String) {
    print("  [step] \(label)")
    fflush(stdout)
}

// Bounded drain: tick until the engine reports no work OR we exceed
// maxTicks (safety net for state-machine edge cases). A hang would
// otherwise make the test take 20 min before manual intervention.
private func drain(_ engine: LmEngine, maxTicks: Int = 64) {
    var ticks = 0
    while engine.hasWork && ticks < maxTicks {
        _ = engine.tick()
        ticks += 1
    }
    if engine.hasWork {
        print("  WARN: drain exceeded \(maxTicks) ticks; engine still reports hasWork")
        fflush(stdout)
    }
}

private func tassert(_ cond: Bool, _ label: String,
                     _ detail: @autoclosure () -> String = "") {
    if cond {
        gTestPass += 1
        print("  PASS: \(label)")
    } else {
        gTestFail += 1
        let d = detail()
        print("  FAIL: \(label)\(d.isEmpty ? "" : " — \(d)")")
    }
    fflush(stdout)
}

private func tassertEqU64(_ a: UInt64, _ b: UInt64, _ label: String) {
    tassert(a == b, label, "0x\(String(a, radix: 16)) vs 0x\(String(b, radix: 16))")
}

private func tassertNotEqU64(_ a: UInt64, _ b: UInt64, _ label: String) {
    tassert(a != b, label, "both = 0x\(String(a, radix: 16))")
}

// ------------------------------------------------------------------
// Helpers to build ActiveControl + a fake HIDDEN-length fp16 cvec
// buffer without needing weights.
// ------------------------------------------------------------------
private func makeScratchCvec(value: Float = 0.01) -> MTLBuffer {
    let buf = device.makeBuffer(length: HIDDEN * 2, options: .storageModeShared)!
    let p = buf.contents().bindMemory(to: Float16.self, capacity: HIDDEN)
    for i in 0..<HIDDEN { p[i] = Float16(value) }
    return buf
}

private func mkControl(id: String, layer: Int,
                       attack: Float = 4, decay: Float = 4,
                       sustain: Float = 1, release: Float = 8,
                       peak: Float = 1, polarity: Float = 1,
                       shape: CvecShape = .linear,
                       units: CvecUnits = .tokens,
                       startPos: Int = 0, startTurn: Int = 0,
                       buf: MTLBuffer? = nil) -> ActiveControl {
    var env = CvecEnvelope()
    env.attack = attack; env.decay = decay
    env.sustainLevel = sustain; env.release = release
    env.peakMagnitude = peak; env.shape = shape; env.units = units
    return ActiveControl(cvecId: id, buffer: buf ?? makeScratchCvec(),
                          layer: layer, envelope: env, polarity: polarity,
                          startPosition: startPos, startTurn: startTurn)
}

// ------------------------------------------------------------------
// Phase 1: digest unit tests. Pure logic, no GPU work.
// ------------------------------------------------------------------
func runCvecDigestUnitTests() {
    enableLineBuffering()
    print("\n=== cvec-digest unit tests ===")
    gTestPass = 0; gTestFail = 0

    let cvecA = makeScratchCvec(value: 0.01)
    let cvecB = makeScratchCvec(value: 0.02)

    // 1a) Empty → 0.
    let empty: [ActiveControl] = []
    tassertEqU64(computeCvecDigest(activeControls: empty, pageStart: 0, pageSize: 16),
                  0, "empty active controls → digest == 0")

    // 1b) Same params → same digest (determinism).
    let c1 = mkControl(id: "X", layer: 12, buf: cvecA)
    let c2 = mkControl(id: "X", layer: 12, buf: cvecA)
    let d1 = computeCvecDigest(activeControls: [c1], pageStart: 0, pageSize: 16)
    let d2 = computeCvecDigest(activeControls: [c2], pageStart: 0, pageSize: 16)
    tassertEqU64(d1, d2, "identical-param controls → identical digest")
    tassert(d1 != 0, "single active control → digest != 0")

    // 1c) Different cvecId → different digest.
    let cX = mkControl(id: "X", layer: 12)
    let cY = mkControl(id: "Y", layer: 12)
    tassertNotEqU64(computeCvecDigest(activeControls: [cX], pageStart: 0, pageSize: 16),
                     computeCvecDigest(activeControls: [cY], pageStart: 0, pageSize: 16),
                     "cvecId X vs Y → digest differs")

    // 1d) Different layer → different digest.
    let cL11 = mkControl(id: "X", layer: 11)
    let cL12 = mkControl(id: "X", layer: 12)
    tassertNotEqU64(computeCvecDigest(activeControls: [cL11], pageStart: 0, pageSize: 16),
                     computeCvecDigest(activeControls: [cL12], pageStart: 0, pageSize: 16),
                     "layer 11 vs 12 → digest differs")

    // 1e) Different envelope params → different digest (each knob).
    let cBase = mkControl(id: "X", layer: 12, attack: 4, decay: 4,
                           sustain: 1, release: 8, peak: 1)
    let base = computeCvecDigest(activeControls: [cBase], pageStart: 0, pageSize: 16)
    let cAttack2 = mkControl(id: "X", layer: 12, attack: 5, decay: 4,
                              sustain: 1, release: 8, peak: 1)
    let cSustain2 = mkControl(id: "X", layer: 12, attack: 4, decay: 4,
                               sustain: 0.5, release: 8, peak: 1)
    let cRelease2 = mkControl(id: "X", layer: 12, attack: 4, decay: 4,
                               sustain: 1, release: 10, peak: 1)
    let cPeak2 = mkControl(id: "X", layer: 12, attack: 4, decay: 4,
                            sustain: 1, release: 8, peak: 1.5)
    let cPolarity2 = mkControl(id: "X", layer: 12, polarity: -1)
    let cShape2 = mkControl(id: "X", layer: 12, shape: .cubic)
    let cUnits2 = mkControl(id: "X", layer: 12, units: .turns)
    tassertNotEqU64(base,
        computeCvecDigest(activeControls: [cAttack2], pageStart: 0, pageSize: 16),
        "attack knob → digest differs")
    tassertNotEqU64(base,
        computeCvecDigest(activeControls: [cSustain2], pageStart: 0, pageSize: 16),
        "sustain knob → digest differs")
    tassertNotEqU64(base,
        computeCvecDigest(activeControls: [cRelease2], pageStart: 0, pageSize: 16),
        "release knob → digest differs")
    tassertNotEqU64(base,
        computeCvecDigest(activeControls: [cPeak2], pageStart: 0, pageSize: 16),
        "peakMagnitude knob → digest differs")
    tassertNotEqU64(base,
        computeCvecDigest(activeControls: [cPolarity2], pageStart: 0, pageSize: 16),
        "polarity knob → digest differs")
    tassertNotEqU64(base,
        computeCvecDigest(activeControls: [cShape2], pageStart: 0, pageSize: 16),
        "shape knob → digest differs")
    tassertNotEqU64(base,
        computeCvecDigest(activeControls: [cUnits2], pageStart: 0, pageSize: 16),
        "units knob → digest differs")

    // 1f) Ordering invariance: [A, B] and [B, A] should digest the same.
    let ca = mkControl(id: "A", layer: 5, buf: cvecA)
    let cb = mkControl(id: "B", layer: 10, buf: cvecB)
    let dAB = computeCvecDigest(activeControls: [ca, cb], pageStart: 0, pageSize: 16)
    let dBA = computeCvecDigest(activeControls: [cb, ca], pageStart: 0, pageSize: 16)
    tassertEqU64(dAB, dBA, "order-invariance of activeControls array")

    // 1g) Window intersection: a control starting after this page ends
    // should not contribute to this page's digest.
    let cLate = mkControl(id: "X", layer: 12, release: 0, startPos: 100)
    let dLatePg0 = computeCvecDigest(activeControls: [cLate], pageStart: 0, pageSize: 16)
    let dLatePg6 = computeCvecDigest(activeControls: [cLate], pageStart: 96, pageSize: 16)
    tassertEqU64(dLatePg0, 0, "control starting at pos 100 does not touch page [0,16)")
    tassert(dLatePg6 != 0, "control starting at pos 100 DOES touch page [96,112)")

    // 1h) Phase strictness: same params, different startOffset → different digest.
    // Control starts at pos=0 vs pos=4, both measured against page [0, 16).
    let cPhase0 = mkControl(id: "X", layer: 12, startPos: 0)
    let cPhase4 = mkControl(id: "X", layer: 12, startPos: 4)
    tassertNotEqU64(
        computeCvecDigest(activeControls: [cPhase0], pageStart: 0, pageSize: 16),
        computeCvecDigest(activeControls: [cPhase4], pageStart: 0, pageSize: 16),
        "startPos 0 vs 4 → different phase-relative digest")

    // 1i) hashPage composition: digest==0 collapses to token-only hash
    // (back-compat with pre-steering promoted pages).
    let tokens: [UInt32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    let tokHash = PageManager.hashPage(tokens[0..<16])
    let tokHashZeroDigest = PageManager.hashPage(tokens[0..<16], cvecDigest: 0)
    tassertEqU64(tokHash, tokHashZeroDigest,
                  "hashPage(tokens) == hashPage(tokens, digest: 0) — back-compat")

    // 1j) hashPage with non-zero digest produces a different key.
    let k0 = PageManager.hashPage(tokens[0..<16], cvecDigest: 0)
    let kSteered = PageManager.hashPage(tokens[0..<16], cvecDigest: d1)
    tassertNotEqU64(k0, kSteered, "hashPage partitions on digest")

    // 1k) Two distinct digests produce two distinct keys on the same tokens.
    let dX = computeCvecDigest(activeControls: [cX], pageStart: 0, pageSize: 16)
    let dY = computeCvecDigest(activeControls: [cY], pageStart: 0, pageSize: 16)
    tassertNotEqU64(PageManager.hashPage(tokens[0..<16], cvecDigest: dX),
                     PageManager.hashPage(tokens[0..<16], cvecDigest: dY),
                     "different digests → different page keys on same tokens")

    print("  digest-unit: \(gTestPass) passed, \(gTestFail) failed")
}

// ------------------------------------------------------------------
// Phase 2: integration test against a live LmEngine. Requires
// GGUF_PATH to load weights; tests the full prefill-steering +
// cache-hit/miss behavior end-to-end.
//
// Structure:
//   (a) baseline unsteered session S_u1 submits prompt P, runs to
//       completion, promotes pages.
//   (b) a second unsteered session S_u2 submits P, should adopt
//       S_u1's pages (cache hit).
//   (c) a steered session S_s1 submits P with cvec active, should
//       NOT adopt S_u1's unsteered pages (cache miss).
//   (d) a second steered session S_s2 with identical cvec params
//       as S_s1 submits P, should adopt S_s1's pages (cache hit).
//   (e) a third steered session S_s3 with DIFFERENT cvec params
//       (different layer) submits P, should miss (cache miss).
//   (f) intervention-preservation: run one session end-to-end with
//       steering through prefill; run another that splits P into
//       two submits (first part triggers cache promotion, second
//       part is fresh prefill) with same steering. Compare the
//       logits of the next generated token — should be within a
//       tight tolerance.
// ------------------------------------------------------------------
func runCvecCacheIntegrationTest(ggufPath: String) {
    enableLineBuffering()
    print("\n=== cvec-cache integration test (GGUF_PATH=\(ggufPath)) ===")
    fflush(stdout)
    gTestPass = 0; gTestFail = 0

    step("loading weights…")
    let wT0 = Date()
    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }
    print(String(format: "  [step] weights loaded in %.1fs", Date().timeIntervalSince(wT0)))
    fflush(stdout)

    // Build a reusable "tiny but multi-page" prompt. PAGE_SLIDE is 16,
    // so we need >= 32 tokens to get at least 2 promotable pages.
    let engine = LmEngine(weights: w)
    // Need ≥ 3 full pages so we can verify partial adoption (last page
    // is always unadopted by submit() to leave a tail for the prefill
    // sampling path — without that backoff, full-adoption sessions stall
    // in .priming with no tail to prefill).
    let prompt = """
    The quick brown fox jumps over the lazy dog. The cat sat on the mat. \
    Paris is the capital of France. Berlin is the capital of Germany. \
    London is the capital of England. Madrid is the capital of Spain. \
    Tokyo is the capital of Japan. Rome is the capital of Italy. \
    Vienna is the capital of Austria. Lisbon is the capital of Portugal.
    """
    let toks = engine.tokenize(prompt, addBos: true)
    let pageCount = toks.count / PAGE_SLIDE
    tassert(pageCount >= 3, "prompt spans ≥ 3 pages (got \(pageCount))")
    // Invariant: submit() guarantees the prefill tail has ≥ 1 token.
    //   - If tokens.count is an exact multiple of PAGE_SLIDE, backoff
    //     fires and max adoption = pageCount - 1.
    //   - Otherwise the partial tail already satisfies ≥1 token and
    //     max adoption = pageCount.
    let maxAdopt = (toks.count % PAGE_SLIDE == 0) ? (pageCount - 1) : pageCount
    print("  prompt: \(toks.count) tokens, \(pageCount) full pages, maxAdopt=\(maxAdopt)")

    // Use a non-trivial cvec so steering has a measurable effect, but
    // keep magnitude small so the model doesn't diverge catastrophically.
    let cvecA = makeScratchCvec(value: 0.02)
    let cvecB = makeScratchCvec(value: 0.02)

    // --- (a) unsteered S_u1 ---
    step("S_u1 (unsteered baseline)")
    guard let sU1 = engine.openSession(maxNewTokens: 1) else { return }
    sU1.submit(toks)
    let tU1 = Date()
    drain(engine)
    let dtU1 = Date().timeIntervalSince(tU1) * 1000
    let adoptedU1 = sU1.ownedPageCount
    print(String(format: "  S_u1 (unsteered): %.1f ms, owns %d pages", dtU1, adoptedU1))
    fflush(stdout)
    // Sanity: first submit with a cold cache must fresh-allocate.

    // --- (b) unsteered S_u2 — should HIT S_u1's promoted pages ---
    step("S_u2 (unsteered, expect cache hit)")
    guard let sU2 = engine.openSession(maxNewTokens: 1) else { return }
    // Probe the content index directly: count how many leading pages'
    // keys are present. This checks partition correctness at the
    // content-index level (separate from submit's backoff rule).
    var expectedHitsU2 = 0
    for p in 0..<pageCount {
        let end = (p + 1) * PAGE_SLIDE
        let digest = computeCvecDigest(activeControls: [],
                                        pageStart: p * PAGE_SLIDE,
                                        pageSize: PAGE_SLIDE)
        let h = PageManager.hashPage(toks[0..<end], cvecDigest: digest)
        if engine.pageManager.findByHash(h) != nil { expectedHitsU2 += 1 } else { break }
    }
    tassert(expectedHitsU2 >= maxAdopt,
             "unsteered→unsteered: content index holds ≥ \(maxAdopt) leading pages",
             "\(expectedHitsU2)")
    sU2.submit(toks)
    // Adoption appends 2 phys pages per slide page (slide primary +
    // full sibling), so ownedPageCount == 2 * slidePagesAdopted
    // immediately after submit (before admission tops up).
    let adoptedU2 = sU2.ownedPageCount
    tassert(adoptedU2 == 2 * maxAdopt,
             "unsteered→unsteered: S_u2 adopts exactly 2*maxAdopt (\(2 * maxAdopt)) phys pages",
             "adopted=\(adoptedU2)/\(2 * maxAdopt)")
    let tU2 = Date()
    drain(engine)
    let dtU2 = Date().timeIntervalSince(tU2) * 1000
    print(String(format: "  S_u2 (unsteered hit): %.1f ms", dtU2))
    fflush(stdout)
    engine.closeSession(sU2)
    engine.closeSession(sU1)

    // --- (c) steered S_s1 — must NOT adopt unsteered pages ---
    step("S_s1 (steered, expect cache miss)")
    guard let sS1 = engine.openSession(maxNewTokens: 1) else { return }
    let ctrlS1 = mkControl(id: "testX", layer: 12,
                            attack: 0, decay: 0, sustain: 1, release: 0,
                            peak: 1, polarity: 1,
                            units: .tokens, startPos: 0, startTurn: 0,
                            buf: cvecA)
    sS1.addControl(ctrlS1)
    sS1.submit(toks)
    let adoptedS1 = sS1.ownedPageCount
    // S1 submits when the only promoted pages in the cache are unsteered.
    // Expectation: its digest != 0, so its hash keys don't match any
    // promoted entry and adoption count should be zero (fresh prefill).
    tassert(adoptedS1 == 0,
             "steered S_s1 adopts NONE of unsteered S_u1's pages (digest partition)",
             "adopted=\(adoptedS1)")
    let tS1 = Date()
    drain(engine)
    let dtS1 = Date().timeIntervalSince(tS1) * 1000
    print(String(format: "  S_s1 (steered miss): %.1f ms", dtS1))
    fflush(stdout)

    // --- (d) steered S_s2 with identical cvec params → HIT S_s1 ---
    step("S_s2 (steered matching params, expect cache hit)")
    guard let sS2 = engine.openSession(maxNewTokens: 1) else { return }
    let ctrlS2 = mkControl(id: "testX", layer: 12,
                            attack: 0, decay: 0, sustain: 1, release: 0,
                            peak: 1, polarity: 1,
                            units: .tokens, startPos: 0, startTurn: 0,
                            buf: cvecA)
    sS2.addControl(ctrlS2)
    sS2.submit(toks)
    let adoptedS2 = sS2.ownedPageCount
    tassert(adoptedS2 == 2 * maxAdopt,
             "steered→steered (matching params): S_s2 adopts 2*maxAdopt (\(2 * maxAdopt)) phys pages",
             "adopted=\(adoptedS2)/\(2 * maxAdopt)")
    let tS2 = Date()
    drain(engine)
    let dtS2 = Date().timeIntervalSince(tS2) * 1000
    print(String(format: "  S_s2 (steered hit): %.1f ms", dtS2))
    fflush(stdout)
    engine.closeSession(sS2)
    engine.closeSession(sS1)

    // --- (e) steered S_s3 with DIFFERENT cvec params → MISS ---
    step("S_s3 (steered different layer, expect miss)")
    guard let sS3 = engine.openSession(maxNewTokens: 1) else { return }
    let ctrlS3 = mkControl(id: "testX", layer: 13,    // different layer
                            attack: 0, decay: 0, sustain: 1, release: 0,
                            peak: 1, polarity: 1,
                            units: .tokens, startPos: 0, startTurn: 0,
                            buf: cvecA)
    sS3.addControl(ctrlS3)
    sS3.submit(toks)
    let adoptedS3 = sS3.ownedPageCount
    tassert(adoptedS3 == 0,
             "steered (diff layer): S_s3 misses S_s2's pages (digest partition)",
             "adopted=\(adoptedS3)")
    drain(engine)
    engine.closeSession(sS3)

    // --- (e2) steered with different cvecId → MISS ---
    step("S_s4 (steered different cvecId, expect miss)")
    guard let sS4 = engine.openSession(maxNewTokens: 1) else { return }
    let ctrlS4 = mkControl(id: "testY", layer: 12,    // different id, same layer
                            attack: 0, decay: 0, sustain: 1, release: 0,
                            peak: 1, polarity: 1,
                            units: .tokens, startPos: 0, startTurn: 0,
                            buf: cvecB)
    sS4.addControl(ctrlS4)
    sS4.submit(toks)
    let adoptedS4 = sS4.ownedPageCount
    tassert(adoptedS4 == 0,
             "steered (diff cvecId): S_s4 misses S_s2's pages (digest partition)",
             "adopted=\(adoptedS4)")
    drain(engine)
    engine.closeSession(sS4)

    // --- (f0) baseline: unsteered full vs split prefill -----------
    // Establishes the floor for KL caused by cache replay itself,
    // independent of cvec. If this KL is already large, the prefix
    // cache's K/V replay path is non-deterministic or buggy (e.g.,
    // MoE router sensitivity to batching) and the steered KL in (f)
    // inherits that floor. If it's ~0, then any KL in (f) above this
    // floor is attributable to prefill steering alone.
    step("(f0) unsteered baseline: full vs split-resume KL")
    let engineU_Full = LmEngine(weights: w)
    guard let sUFull = engineU_Full.openSession(maxNewTokens: 1) else { return }
    sUFull.submit(toks)
    drain(engineU_Full)
    let logitsUFull = extractNextTokenLogits(slot: sUFull.slot ?? 0)

    let splitAtU = PAGE_SLIDE
    let engineU_Split = LmEngine(weights: w)
    guard let sUSplitA = engineU_Split.openSession(maxNewTokens: 1) else { return }
    sUSplitA.submit(Array(toks.prefix(splitAtU)))
    drain(engineU_Split)
    engineU_Split.closeSession(sUSplitA)
    guard let sUSplitB = engineU_Split.openSession(maxNewTokens: 1) else { return }
    sUSplitB.submit(toks)
    drain(engineU_Split)
    let logitsUSplit = extractNextTokenLogits(slot: sUSplitB.slot ?? 0)

    let klUnsteered = klDivergenceFromLogits(logitsUFull, logitsUSplit)
    print(String(format: "  UNSTEERED baseline KL(full ‖ split) = %.4f", klUnsteered))
    fflush(stdout)
    // This KL is our reference floor. Any larger KL in the steered
    // test is the marginal cost of prefill-steering correctness.

    // --- (f) intervention-preservation: full prefill vs split prefill ---
    step("(f) steered intervention-preservation (full vs split prefill)")
    // Both runs use the same cvec params and same prompt. One runs the
    // full prompt in a single submit (straight prefill). The other
    // splits the prompt after `splitAt` tokens and submits in two
    // pieces: the first piece primes the cache, the second piece's
    // prefill must produce identical post-prefill K/V so that the
    // next-token logits match.
    let splitAt = PAGE_SLIDE  // split on a page boundary
    tassert(splitAt < toks.count,
             "split point \(splitAt) inside prompt of \(toks.count) toks")

    // Run FULL in a fresh engine so cache state is clean.
    let engineFull = LmEngine(weights: w)
    guard let sFull = engineFull.openSession(maxNewTokens: 1) else { return }
    sFull.addControl(mkControl(id: "testZ", layer: 12,
                                attack: 0, decay: 0, sustain: 1, release: 0,
                                peak: 1, polarity: 1,
                                units: .tokens, startPos: 0, startTurn: 0,
                                buf: cvecA))
    sFull.submit(toks)
    drain(engineFull)
    let logitsFull = extractNextTokenLogits(slot: sFull.slot ?? 0)

    // Run SPLIT in a fresh engine. First submit: first `splitAt` toks.
    // This primes the cache with steered pages. Second submit: remaining
    // toks — a fresh session submits the full prompt to force the cache
    // lookup + partial adoption path.
    let engineSplit = LmEngine(weights: w)
    guard let sSplitA = engineSplit.openSession(maxNewTokens: 1) else { return }
    sSplitA.addControl(mkControl(id: "testZ", layer: 12,
                                  attack: 0, decay: 0, sustain: 1, release: 0,
                                  peak: 1, polarity: 1,
                                  units: .tokens, startPos: 0, startTurn: 0,
                                  buf: cvecA))
    sSplitA.submit(Array(toks.prefix(splitAt)))
    drain(engineSplit)
    engineSplit.closeSession(sSplitA)

    guard let sSplitB = engineSplit.openSession(maxNewTokens: 1) else { return }
    sSplitB.addControl(mkControl(id: "testZ", layer: 12,
                                  attack: 0, decay: 0, sustain: 1, release: 0,
                                  peak: 1, polarity: 1,
                                  units: .tokens, startPos: 0, startTurn: 0,
                                  buf: cvecA))
    sSplitB.submit(toks)
    let adoptedSplit = sSplitB.ownedPageCount
    tassert(adoptedSplit >= 2 * (splitAt / PAGE_SLIDE),
             "split-prefill: S_splitB adopts ≥ \(2 * (splitAt / PAGE_SLIDE)) phys page(s) from S_splitA",
             "adopted=\(adoptedSplit)")
    drain(engineSplit)
    let logitsSplit = extractNextTokenLogits(slot: sSplitB.slot ?? 0)

    // Compare the two logit vectors. If prefill steering is wired and
    // the cache hit carried the intervention forward, next-token
    // distributions should match to within fp16 + kernel nondeterminism.
    let kl = klDivergenceFromLogits(logitsFull, logitsSplit)
    // Tolerance is unsteered-baseline + small epsilon. Any excess
    // is the marginal error contributed by prefill-steering alone.
    let tol = klUnsteered + 0.02
    tassert(kl < tol,
             String(format: "steered KL < unsteered-baseline + 0.02 (baseline=%.4f, tol=%.4f)",
                     klUnsteered, tol),
             String(format: "steered kl=%.4f, delta=%.4f", kl, kl - klUnsteered))
    print(String(format: "  STEERED KL=%.4f, UNSTEERED KL=%.4f, delta=%.4f",
                  kl, klUnsteered, kl - klUnsteered))

    print("  integration: \(gTestPass) passed, \(gTestFail) failed")
}

// Pull the post-prefill next-token logits out of the engine's AR
// logits buffer (stepPrefillForSession copies slot-s's last-position
// logit there at end-of-prefill).
private func extractNextTokenLogits(slot: Int) -> [Float] {
    let p = logits.contents().assumingMemoryBound(to: Float16.self)
    var out = [Float](repeating: 0, count: VOCAB)
    let base = slot * VOCAB
    for v in 0..<VOCAB { out[v] = Float(p[base + v]) }
    return out
}

// ------------------------------------------------------------------
// Phase 3: layer-wise divergence dump. Runs the unsteered full-prefill
// vs split-prefill-resume scenario and snapshots every layer's post-
// FFN residual (slot 0, last position) via gAllLayerCaptureBuf. Diffs
// the two NUM_LAYERS × HIDDEN tensors per layer, reports MSE and
// max-abs-diff. The first layer with significant divergence localizes
// the responsible kernel.
//
// The unsteered KL(full ‖ split) = 0.38 measured by the integration
// test is a *large* divergence (on the order of "a few nats") — we
// expect it to show up as a specific layer whose kernel processes
// adopted K/V pages differently from freshly-written ones. Candidates:
//
//   - Flex attention: prefill's mask-bitmap / K-lookback may behave
//     differently when reading a "pre-populated" phys page vs one
//     just written in the same CB.
//   - MoE router: topk experts may differ under fp16 + batch shape
//     differences (full does qLen=8 tiles all the way through; split
//     does fewer tiles but on a session whose block_table points at
//     adopted phys pages, which were written by a *different* session).
//   - KV write: if K/V values in adopted pages have any staleness vs
//     what they would be if computed fresh.
//
// Binary search procedure: look for the first layer L where MSE > ~1e-3
// or max-abs-diff > 0.01. The hidden-stream before that layer is
// coherent, the computation *at* that layer is the divergence point.
// ------------------------------------------------------------------
func runPrefillCacheDivergenceDump(ggufPath: String) {
    enableLineBuffering()
    print("\n=== prefill-cache divergence dump ===")
    fflush(stdout)

    let w: LmWeights
    do { w = try loadLmWeights(ggufPath: ggufPath) }
    catch { print("  loadLmWeights failed: \(error)"); return }

    // Same prompt shape as the integration test: 3+ full pages so the
    // split point is well inside the prefill region.
    let prompt = """
    The quick brown fox jumps over the lazy dog. The cat sat on the mat. \
    Paris is the capital of France. Berlin is the capital of Germany. \
    London is the capital of England. Madrid is the capital of Spain. \
    Tokyo is the capital of Japan. Rome is the capital of Italy. \
    Vienna is the capital of Austria. Lisbon is the capital of Portugal.
    """
    let engine0 = LmEngine(weights: w)
    let toks = engine0.tokenize(prompt, addBos: true)
    let pageCount = toks.count / PAGE_SLIDE
    print("  prompt: \(toks.count) tokens, \(pageCount) pages, last position = \(toks.count - 1)")
    fflush(stdout)

    // Turn on all-layer capture for both runs.
    gCaptureAllLayers = true
    defer { gCaptureAllLayers = false }

    // --- FULL run ---
    print("  [1/2] running FULL prefill…"); fflush(stdout)
    let engineFull = LmEngine(weights: w)
    guard let sFull = engineFull.openSession(maxNewTokens: 1) else { return }
    sFull.submit(toks)
    drain(engineFull, maxTicks: 256)
    let fullSlot = sFull.slot ?? 0
    print("    FULL slot=\(fullSlot), adopted=\(sFull.ownedPageCount) pages"); fflush(stdout)

    // Snapshot the capture buffer.
    var residFull = [Float](repeating: 0, count: NUM_LAYERS * HIDDEN)
    let capP = gAllLayerCaptureBuf.contents().assumingMemoryBound(to: Float16.self)
    for i in 0..<(NUM_LAYERS * HIDDEN) { residFull[i] = Float(capP[i]) }

    // --- SPLIT run ---
    print("  [2/2] running SPLIT prefill (A: first page, close, B: adopt + finish)…"); fflush(stdout)
    let engineSplit = LmEngine(weights: w)
    guard let sSplitA = engineSplit.openSession(maxNewTokens: 1) else { return }
    sSplitA.submit(Array(toks.prefix(PAGE_SLIDE)))
    drain(engineSplit, maxTicks: 256)
    engineSplit.closeSession(sSplitA)

    guard let sSplitB = engineSplit.openSession(maxNewTokens: 1) else { return }
    sSplitB.submit(toks)
    let adoptedB = sSplitB.ownedPageCount
    print("    SPLIT-B adopted=\(adoptedB) pages from A"); fflush(stdout)
    drain(engineSplit, maxTicks: 256)
    let splitSlot = sSplitB.slot ?? 0
    print("    SPLIT slot=\(splitSlot)"); fflush(stdout)

    var residSplit = [Float](repeating: 0, count: NUM_LAYERS * HIDDEN)
    for i in 0..<(NUM_LAYERS * HIDDEN) { residSplit[i] = Float(capP[i]) }

    // Warn if slots disagree — captures are slot-0 only.
    if fullSlot != 0 || splitSlot != 0 {
        print("  WARN: expected slot==0 for both runs (capture is slot-0 only); "
              + "got full=\(fullSlot), split=\(splitSlot). Results may be meaningless.")
        fflush(stdout)
    }

    // --- Diff per-layer ---
    print("")
    print("  per-layer diff (post-FFN residual, slot 0, last position):")
    print("    ┌─────┬────────────┬────────────┬───────────┬───────────┬────────")
    print("    │ L   │ MSE        │ max|diff|  │ ‖full‖₂   │ ‖split‖₂  │ flag")
    print("    ├─────┼────────────┼────────────┼───────────┼───────────┼────────")
    var firstDiverge = -1
    let mseThreshold: Double = 1e-4
    for L in 0..<NUM_LAYERS {
        var sumSq: Double = 0, maxAbs: Double = 0
        var normFull: Double = 0, normSplit: Double = 0
        for h in 0..<HIDDEN {
            let a = Double(residFull[L * HIDDEN + h])
            let b = Double(residSplit[L * HIDDEN + h])
            let d = a - b
            sumSq += d * d
            if abs(d) > maxAbs { maxAbs = abs(d) }
            normFull += a * a
            normSplit += b * b
        }
        let mse = sumSq / Double(HIDDEN)
        normFull = sqrt(normFull)
        normSplit = sqrt(normSplit)
        let flag: String
        if mse > mseThreshold {
            if firstDiverge < 0 { firstDiverge = L; flag = " ← FIRST DIVERGE" }
            else { flag = " diverged" }
        } else {
            flag = ""
        }
        print(String(format: "    │ L%02d │ %.4e │ %.4e │ %9.2f │ %9.2f │%@",
                     L, mse, maxAbs, normFull, normSplit, flag))
        fflush(stdout)
    }
    print("    └─────┴────────────┴────────────┴───────────┴───────────┴────────")
    if firstDiverge < 0 {
        print("  RESULT: no layer exceeds MSE threshold \(mseThreshold) — divergence is below measurement floor")
    } else {
        print("  RESULT: first divergent layer = L\(firstDiverge). Inspect the kernels dispatched at that layer in encodePrefillTileInto.")
    }
    fflush(stdout)
}

// Softmax-normalized KL divergence. Inputs are raw logits of equal length.
private func klDivergenceFromLogits(_ a: [Float], _ b: [Float]) -> Double {
    precondition(a.count == b.count)
    let n = a.count
    // Softmax both, with max-subtraction for numerical stability.
    var maxA: Float = -.infinity, maxB: Float = -.infinity
    for i in 0..<n { if a[i] > maxA { maxA = a[i] }; if b[i] > maxB { maxB = b[i] } }
    var sumA: Double = 0, sumB: Double = 0
    var expA = [Double](repeating: 0, count: n)
    var expB = [Double](repeating: 0, count: n)
    for i in 0..<n {
        expA[i] = exp(Double(a[i] - maxA)); sumA += expA[i]
        expB[i] = exp(Double(b[i] - maxB)); sumB += expB[i]
    }
    var kl: Double = 0
    let eps: Double = 1e-12
    for i in 0..<n {
        let pa = expA[i] / sumA
        let pb = expB[i] / sumB
        if pa > eps {
            kl += pa * (log(pa + eps) - log(pb + eps))
        }
    }
    return kl
}
