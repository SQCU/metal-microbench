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
import Foundation
import Metal

// 2026-05-07 (anonymous-pool refactor): collapsed from a 5-state FSM
// (.idle/.priming/.generating/.paused/.done) to the three phases a
// stateless chat-completions request goes through. The deleted .idle
// and .paused cases existed only because the old API was two-phase
// (openSession then submit then setters then maybe more submits). Under
// atomic construction (LmEngine.submitRequest) a Session is born with
// chunks already queued, so .priming covers "has prefill work" and
// there's no half-state where a Session exists with no work.
enum SessionState: Equatable {
    case priming        // primingQueue non-empty (or just-finished prefill on transition to generating)
    case primed         // Track A fold-in: all prefill done, K/V at pos N exists,
                        // no first generated token sampled yet. Needs a 1-token
                        // recover tick at position-1 to produce the first
                        // sampled token; then transitions to .generating.
    case generating     // sampling; pushing to outputQueue
    case done           // EOS / maxTokens

    // Does this session want a slot on the next scheduler tick?
    var wantsSlot: Bool {
        switch self {
        case .priming, .primed, .generating: return true
        case .done: return false
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
    // `eventTicket` is the value the vision-tower pad-blit CB signals on
    // `gVisionEvent`. The LM consumer's pre-prefill CB encodes
    // `encodeWaitForEvent(gVisionEvent, value: eventTicket)` so the GPU
    // itself waits for the pad-blit before reading `buffer`. CPU never
    // blocks. 0 = no wait needed (vision long since complete, or never
    // had a pending CB). See notes/engine_debloat.md.
    case softTokens(buffer: MTLBuffer, count: Int, isFp32: Bool, byteOffset: Int,
                    eventTicket: UInt64)

    var count: Int {
        switch self {
        case .tokens(let ts): return ts.count
        case .softTokens(_, let c, _, _, _): return c
        }
    }
}

// MARK: - Structured chain-of-thought (grammar-constrained sampling)
//
// Implements the structured-cot intervention from
// github.com/andthattoo/structured-cot. Once enabled on a session, the
// AR sampler's logit-bias mask is set per-step to enforce a fixed
// reasoning-trace shape:
//
//   <think>\n LABEL_0: <free-line> LABEL_1: <free-line> ... </think>\n\n
//
// Each LABEL_i is provided by the caller (default GOAL/APPROACH/EDGE).
// Inside the think block, every token is masked: literal phases force
// tokens whose bytes prefix-match the remaining literal; free-line
// phases allow any token whose bytes don't contain a mid-token
// newline (only an ending newline transitions to the next literal).
// After the closing </think>\n\n, the grammar deactivates and the
// model decodes free-form for the answer.
//
// Reference findings (HumanEval+, Qwen3.6-35B-A3B-Q4_K_M): same pass@1
// as free-form thinking with 22× fewer thinking tokens.
enum CotPhase {
    case literal(bytes: [UInt8])   // emit these bytes exactly across N tokens
    case freeLine                   // any non-newline-mid token, ends on \n
}

final class CotState {
    var phases: [CotPhase]      // remaining phases (current = phases.first)
    var literalCursor: Int = 0   // bytes already emitted into current literal phase

    init(phases: [CotPhase]) { self.phases = phases }

    var isDone: Bool { phases.isEmpty }

    // Advance the state by the bytes of the just-sampled token. Returns
    // true if the state is now done (no more constraints — caller can
    // drop the CotState reference). Mismatches print a warning and
    // mark the state done (defensive: shouldn't happen if mask is
    // correct, but a stuck state is worse than just letting the model
    // continue free-form).
    @discardableResult
    func advance(by tokBytes: [UInt8]) -> Bool {
        guard let phase = phases.first else { return true }
        if tokBytes.isEmpty { return false }
        switch phase {
        case .literal(let bytes):
            let remaining = bytes.count - literalCursor
            if tokBytes.count >= remaining {
                // Token completes (and possibly overshoots) the literal.
                let prefix = Array(bytes[literalCursor..<bytes.count])
                if Array(tokBytes.prefix(remaining)) == prefix {
                    phases.removeFirst()
                    literalCursor = 0
                    // Overshoot bytes feed into the next phase (rare —
                    // mask should usually prevent this, but if BPE
                    // produced a token spanning the boundary, propagate).
                    if tokBytes.count > remaining {
                        return advance(by: Array(tokBytes.dropFirst(remaining)))
                    }
                } else {
                    fputs("[cot] literal mismatch: expected \(prefix), got \(tokBytes); deactivating grammar\n", stderr)
                    phases.removeAll()
                }
            } else {
                // Token is a strict prefix of remaining literal.
                let expected = Array(bytes[literalCursor..<(literalCursor + tokBytes.count)])
                if expected == tokBytes {
                    literalCursor += tokBytes.count
                } else {
                    fputs("[cot] literal mismatch mid-cursor; deactivating grammar\n", stderr)
                    phases.removeAll()
                }
            }
        case .freeLine:
            // The line ends if the chosen token ends with newline. Any
            // mid-token newline shouldn't have been masked-in; if it
            // happens, treat as line-end (newline at any position).
            if tokBytes.contains(0x0A) {
                phases.removeFirst()
            }
            // else: still in this line, no advance.
        }
        return phases.isEmpty
    }
}

// MARK: - Control vectors (Phase B)
//
// A cvector is an (injection-layer-L, HIDDEN-length half) pair applied as
// residual[L, b, :] += envelope(t) · polarity · cvec[:] at every AR step.
// Per-session active-controls list is evaluated each tick; the scheduler
// then dispatches one tiny add kernel per (slot, control) pair per layer.
//
// `position` and `turnIndex` on Session are the two time axes. The
// envelope chooses which via its `units`.

enum CvecShape: String, Codable {
    case linear = "linear"
    case expIn  = "exp-in"     // f(t) = t²  — slow start, fast end
    case expOut = "exp-out"    // f(t) = 1-(1-t)² — fast start, slow end
    case cubic  = "cubic"      // f(t) = 3t² - 2t³ — smoothstep
    func apply(_ p: Float) -> Float {
        let t = max(0, min(1, p))
        switch self {
        case .linear: return t
        case .expIn:  return t * t
        case .expOut: return 1 - (1 - t) * (1 - t)
        case .cubic:  return t * t * (3 - 2 * t)
        }
    }
}

enum CvecUnits: String, Codable {
    case tokens, turns
}

struct CvecEnvelope {
    var attack: Float = 0         // ramp-up duration in `units`
    var decay: Float = 0          // peak → sustain duration
    var sustainLevel: Float = 1   // 0..1 fraction of peak magnitude
    var release: Float = 0        // sustain → 0 duration after stop
    var peakMagnitude: Float = 1  // the "A" peak in ADSR
    var shape: CvecShape = .linear
    var units: CvecUnits = .tokens
}

// Mode for how a control interacts with the residual stream at its
// injection layer.
//   .additive  — residual += mag * cvec  (current default).
//                good for "push the residual along this axis."
//   .project   — residual's projection onto cvec is SET to target
//                (via current_proj measurement + delta write). same
//                kernel reads the pre-write projection so callers
//                get measurement-for-free. representation-engineering
//                primitive — target=0 removes the feature entirely,
//                nonzero targets coerce to a specific feature level.
enum CvecMode: String, Codable { case additive, project, transport }

// Heretic-style per-write directional ablation site. Applied as a
// post-matmul hook on the output buffer of either attention-out-proj
// (component=.attnOut, buffer `mlp_out`) or FFN combined output
// (component=.ffnOut, buffer `ffn_combined`) before each feeds into
// the residual-add. Per-layer per-component (r̂, α); engine-level
// state (NOT per-session — ablation is a model-level transform, uniform
// across batch slots). Configured via gemma_engine_set_write_ablation
// from the Python TPE harness and/or user-invoked calibration pipeline.
enum AblationComponent: Int32 {
    case attnOut = 0
    case ffnOut  = 1
}

struct LayerComponentAblation {
    let layer: Int
    let component: AblationComponent
    let rHatBuf: MTLBuffer   // fp16, HIDDEN halves, unit-norm r̂
    let alpha: Float          // heretic α(L); 0 = no-op, 1 = full ablation, <0 = amplify
}

// One active control vector on a session. `magnitudeAt(position:turn:)`
// is pure and cheap — evaluated once per tick before the engine builds
// the step CB, so the kernel sees a precomputed scalar.
final class ActiveControl {
    let cvecId: String          // key into the cvec registry
    let buffer: MTLBuffer       // resolved cvec MTLBuffer (half × HIDDEN)
    let layer: Int              // which LM layer to inject at
    let envelope: CvecEnvelope
    let polarity: Float         // +1 or -1 (or arbitrary multiplier)
    let mode: CvecMode          // additive or project
    // For project mode only: when non-nil, envelope scalar becomes a
    // GATE (0 = skip dispatch, nonzero = fire) and `target` is the
    // coerce-to value. When nil (default), envelope scalar is the
    // target directly — current obliteratus-style "always-on" semantics
    // where peak=0 means coerce-to-zero. Splitting this out enables
    // detector → trigger → ablation scoping: the envelope is the gate
    // signal (restarted by trigger), target stays fixed at e.g. 0.
    let target: Float?
    // For transport mode (CvecMode.transport). The Brenier map for
    // per-PC Gaussian OT is a' = scale*a + offset, so we precompute
    // these once at attach and reuse them every tick. Client computes:
    //   scale  = σ_tgt / σ_src
    //   offset = μ_tgt − scale * μ_src
    // Zero scale + zero offset is a valid "no-op" config (but a project-
    // mode control with target=current-projection would be cheaper).
    let transportScale: Float
    let transportOffset: Float
    // For project-mode controls: the pre-write projection the kernel
    // measured this tick, before coercing the residual to the target.
    // This is the representation-engineering "natural activation
    // level" — what projection the model was heading toward before
    // our coerce overrode it. Populated by step() after each CB commit
    // from gProjectMeasureBuf. Unused (nil) for additive mode.
    var lastProjectMeasurement: Float? = nil
    // start{Position,Turn} are mutable so a gated trigger can RESTART
    // the envelope at an arbitrary later moment without allocating a
    // new ActiveControl. stop fields clear on restart.
    var startPosition: Int
    var startTurn: Int
    var stopPosition: Int? = nil
    var stopTurn: Int? = nil
    init(cvecId: String, buffer: MTLBuffer, layer: Int,
         envelope: CvecEnvelope, polarity: Float,
         startPosition: Int, startTurn: Int,
         mode: CvecMode = .additive,
         target: Float? = nil,
         transportScale: Float = 0,
         transportOffset: Float = 0) {
        self.cvecId = cvecId; self.buffer = buffer; self.layer = layer
        self.envelope = envelope; self.polarity = polarity
        self.mode = mode
        self.target = target
        self.transportScale = transportScale
        self.transportOffset = transportOffset
        self.startPosition = startPosition; self.startTurn = startTurn
    }
    // Reset the envelope clock to (position, turn). Triggered by a
    // detector event; behaves exactly as if the control were freshly
    // attached — attack ramp replays from zero.
    func restart(position: Int, turn: Int) {
        startPosition = position; startTurn = turn
        stopPosition = nil; stopTurn = nil
    }
    func magnitudeAt(position: Int, turn: Int) -> Float {
        let elapsed: Float
        switch envelope.units {
        case .tokens: elapsed = Float(position - startPosition)
        case .turns:  elapsed = Float(turn - startTurn)
        }
        if elapsed < 0 { return 0 }
        let peak = envelope.peakMagnitude * polarity
        let sustain = peak * envelope.sustainLevel
        // Attack: 0 → peak over `attack` units.
        if elapsed < envelope.attack {
            return peak * envelope.shape.apply(elapsed / max(envelope.attack, 1e-9))
        }
        // Decay: peak → sustain over `decay` units.
        let afterAttack = elapsed - envelope.attack
        if afterAttack < envelope.decay {
            let p = envelope.shape.apply(afterAttack / max(envelope.decay, 1e-9))
            return peak + (sustain - peak) * p
        }
        // Sustain (if no stop) or release (if stop has been triggered).
        let stopRef: Int?
        let nowRef: Int
        switch envelope.units {
        case .tokens: stopRef = stopPosition; nowRef = position
        case .turns:  stopRef = stopTurn;     nowRef = turn
        }
        guard let stop = stopRef else { return sustain }
        let rel = Float(nowRef - stop)
        if rel < 0 { return sustain }
        if rel < envelope.release {
            return sustain * (1 - envelope.shape.apply(rel / max(envelope.release, 1e-9)))
        }
        return 0
    }
}

// SplitMix64: deterministic, seedable PRNG. One UInt64 of state, one
// multiply-xor-shift sequence per draw. Used for per-session seeded
// sampling — replacing SystemRandomNumberGenerator lets a test harness
// reproduce a trajectory bit-for-bit across runs.
struct SeedableRNG: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64 = 0x9E3779B97F4A7C15) {
        self.state = seed == 0 ? 0x9E3779B97F4A7C15 : seed
    }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}


// Pure-function digest over the cvec state that touches a given page's
// position range. Extracted from Session so the digest is testable in
// isolation (no engine/weights required) and so a future RL trainer can
// compute the digest for a planned rollout before instantiating the
// session.
//
// Strictness contract (agreed with user for RL-grade train/test
// determinism):
//   - Digest is over envelope *parameters* (exact), never evaluated
//     magnitudes (floaty, non-deterministic across kernel variants).
//   - Includes the start-offset-relative-to-page-start, so two sessions
//     must have triggered the cvec at the same relative phase to share
//     pages. Strict — if you want looser sharing, loosen here explicitly.
//   - Stable across process restarts: cvecId hashed as UTF-8 byte
//     sequence; no reliance on Swift's seeded String.hashValue.
//   - Empty activeControls → returns 0 → hashPage collapses to the
//     pre-existing token-only hash. Unsteered sessions keep sharing.
func computeCvecDigest(activeControls: [ActiveControl],
                        pageStart: Int, pageSize: Int) -> UInt64 {
    if activeControls.isEmpty { return 0 }
    let pageEnd = pageStart + pageSize
    var h: UInt64 = 0xcbf29ce484222325
    struct DigestEntry {
        let layer: Int; let cvecId: String
        let startOffset: Int; let stopOffset: Int
        let env: CvecEnvelope; let polarity: Float
        let startPosition: Int; let startTurn: Int
        let stopPosition: Int?; let stopTurn: Int?
        let mode: CvecMode
        let target: Float?
        let transportScale: Float
        let transportOffset: Float
    }
    var entries: [DigestEntry] = []
    for c in activeControls {
        let cStart = c.startPosition
        let cStop = c.stopPosition.map { $0 + Int(c.envelope.release.rounded(.up)) }
            ?? Int.max
        let intersects = cStart < pageEnd && cStop > pageStart
        if !intersects { continue }
        entries.append(DigestEntry(
            layer: c.layer, cvecId: c.cvecId,
            startOffset: cStart - pageStart,
            stopOffset: (c.stopPosition ?? Int.max) - pageStart,
            env: c.envelope, polarity: c.polarity,
            startPosition: c.startPosition, startTurn: c.startTurn,
            stopPosition: c.stopPosition, stopTurn: c.stopTurn,
            mode: c.mode, target: c.target,
            transportScale: c.transportScale,
            transportOffset: c.transportOffset))
    }
    // If no control's window intersected this page, the page is
    // effectively unsteered — return 0 so hashPage collapses to the
    // token-only key and sharing with unsteered pages is possible.
    if entries.isEmpty { return 0 }
    entries.sort { (a, b) -> Bool in
        if a.layer != b.layer { return a.layer < b.layer }
        if a.cvecId != b.cvecId { return a.cvecId < b.cvecId }
        return a.startOffset < b.startOffset
    }
    @inline(__always) func mixU64(_ x: UInt64) {
        var v = x
        for _ in 0..<8 {
            h ^= v & 0xff
            h = h &* 0x100000001b3
            v >>= 8
        }
    }
    @inline(__always) func mixF32(_ f: Float) {
        mixU64(UInt64(f.bitPattern))
    }
    @inline(__always) func mixInt(_ i: Int) {
        mixU64(UInt64(bitPattern: Int64(i)))
    }
    @inline(__always) func mixString(_ s: String) {
        for b in s.utf8 {
            h ^= UInt64(b)
            h = h &* 0x100000001b3
        }
        h ^= 0xff
        h = h &* 0x100000001b3
    }
    mixInt(entries.count)
    for e in entries {
        mixInt(e.layer)
        mixString(e.cvecId)
        mixInt(e.startOffset)
        mixInt(e.stopOffset)
        mixInt(e.startPosition)
        mixInt(e.startTurn)
        mixInt(e.stopPosition ?? Int.min)
        mixInt(e.stopTurn ?? Int.min)
        mixF32(e.polarity)
        mixF32(e.env.attack)
        mixF32(e.env.decay)
        mixF32(e.env.sustainLevel)
        mixF32(e.env.release)
        mixF32(e.env.peakMagnitude)
        mixString(e.env.shape.rawValue)
        mixString(e.env.units.rawValue)
        mixString(e.mode.rawValue)
        // Target presence bit + value. nil → 0 tag (envelope IS target);
        // non-nil → 1 tag + bits (envelope is gate, target is coerce value).
        if let t = e.target { mixU64(1); mixF32(t) } else { mixU64(0) }
        // Transport-mode params. Zero for non-transport modes; they
        // don't affect digest collision in practice because mode is
        // included above.
        mixF32(e.transportScale)
        mixF32(e.transportOffset)
    }
    if h == 0 { h = 0x9e3779b97f4a7c15 }
    return h
}

// Track C: tighter cvec partition tag. Returns CvecAnchorTag.unsteered
// when no control's effective magnitude overlaps the page. Units-gated,
// phase-gated, magnitude-thresholded, floats quantized to Q16.16.
//
// This is the value the radix trie uses as a per-anchor partition key.
// Two CvecAnchorTags hash-equal exactly when their cvec-induced K/V
// perturbation over the page is provably identical (modulo the
// quantization tolerance).
func computeCvecAnchorTag(activeControls: [ActiveControl],
                           pageStart: Int, pageSize: Int) -> CvecAnchorTag {
    if activeControls.isEmpty { return .unsteered }
    let pageEnd = pageStart + pageSize
    @inline(__always) func quantF32(_ f: Float) -> Int64 {
        // Q16.16 fixed point. 1.5e-5 grid, well within fp16 noise.
        return Int64((f * 65536.0).rounded())
    }
    var entries: [CvecAnchorTag.Entry] = []
    for c in activeControls {
        let cStart = c.startPosition
        let cStop = c.stopPosition.map { $0 + Int(c.envelope.release.rounded(.up)) }
            ?? Int.max
        let intersects = cStart < pageEnd && cStop > pageStart
        if !intersects { continue }

        // Magnitude epsilon: skip controls whose effective peak over
        // the page is below fp16 noise. Sustain-decayed-to-zero
        // controls fall through here.
        let peakAbs = abs(c.envelope.peakMagnitude * c.polarity)
        // Closed-form: pick the max |magnitudeAt| at phase boundaries
        // clipped into the page. For tokens-units we sample at the
        // four boundaries (pageStart, attack-end, decay-end, stopPos,
        // release-end) clamped. Cheap and good enough.
        let testPositions: [Int] = [
            pageStart, pageEnd - 1,
            cStart + Int(c.envelope.attack.rounded(.up)),
            cStart + Int((c.envelope.attack + c.envelope.decay).rounded(.up)),
            c.stopPosition ?? pageEnd,
            (c.stopPosition ?? pageStart) + Int(c.envelope.release.rounded(.up))
        ]
        var peakHere: Float = 0
        for tp in testPositions {
            let clamped = max(pageStart, min(pageEnd - 1, tp))
            // Note: magnitudeAt uses position OR turn depending on units.
            // For tag computation we just need a representative value.
            let m = abs(c.magnitudeAt(position: clamped, turn: 0))
            if m > peakHere { peakHere = m }
        }
        let EPS_MAG: Float = 1e-3
        if peakAbs > 0 && peakHere < EPS_MAG * peakAbs { continue }

        // Phase membership over the page.
        // Attack phase: elapsed in [0, attack). Decay: [attack, attack+decay).
        // Sustain: [attack+decay, stop). Release: [stop, stop+release).
        let attackEnd = cStart + Int(c.envelope.attack.rounded(.up))
        let decayEnd = attackEnd + Int(c.envelope.decay.rounded(.up))
        let stopPos = c.stopPosition ?? Int.max
        let releaseEnd = stopPos == Int.max
            ? Int.max
            : stopPos + Int(c.envelope.release.rounded(.up))
        let inAttack = cStart < pageEnd && attackEnd > pageStart
        let inDecay  = attackEnd < pageEnd && decayEnd > pageStart
        let inSustain = decayEnd < pageEnd && stopPos > pageStart
        let inRelease = stopPos < pageEnd && releaseEnd > pageStart
        let anyRamp = inAttack || inDecay || inRelease

        let attackQ: Int64? = inAttack ? quantF32(c.envelope.attack) : nil
        let decayQ:  Int64? = inDecay  ? quantF32(c.envelope.decay)  : nil
        let sustainQ: Int64? = inSustain ? quantF32(c.envelope.sustainLevel) : nil
        let releaseQ: Int64? = inRelease ? quantF32(c.envelope.release) : nil
        let shapeIfRamp: String? = anyRamp ? c.envelope.shape.rawValue : nil

        // Units-gated start/stop anchors.
        var startAnchor: Int? = nil
        var stopAnchor: Int? = nil
        if inAttack || inDecay {
            switch c.envelope.units {
            case .tokens: startAnchor = c.startPosition - pageStart
            case .turns:  startAnchor = c.startTurn
            }
        }
        if inRelease {
            switch c.envelope.units {
            case .tokens: stopAnchor = (c.stopPosition ?? Int.min) - pageStart
            case .turns:  stopAnchor = c.stopTurn ?? Int.min
            }
        }

        // Mode-gated extras.
        let targetQ: Int64? = (c.mode == .project)
            ? c.target.map { quantF32($0) } : nil
        let transportScaleQ: Int64? = (c.mode == .transport)
            ? quantF32(c.transportScale) : nil
        let transportOffsetQ: Int64? = (c.mode == .transport)
            ? quantF32(c.transportOffset) : nil

        entries.append(CvecAnchorTag.Entry(
            layer: c.layer,
            cvecId: c.cvecId,
            mode: c.mode.rawValue,
            units: c.envelope.units.rawValue,
            attackQ: attackQ,
            decayQ: decayQ,
            sustainLevelQ: sustainQ,
            releaseQ: releaseQ,
            shapeIfRamp: shapeIfRamp,
            peakTimesPolarityQ: quantF32(c.envelope.peakMagnitude * c.polarity),
            startAnchor: startAnchor,
            stopAnchor: stopAnchor,
            targetQ: targetQ,
            transportScaleQ: transportScaleQ,
            transportOffsetQ: transportOffsetQ
        ))
    }
    if entries.isEmpty { return .unsteered }
    // Order-invariant: sort by (layer, cvecId, startAnchor).
    entries.sort { (a, b) -> Bool in
        if a.layer != b.layer { return a.layer < b.layer }
        if a.cvecId != b.cvecId { return a.cvecId < b.cvecId }
        return (a.startAnchor ?? Int.min) < (b.startAnchor ?? Int.min)
    }
    return CvecAnchorTag(entries: entries)
}

// Phase C-Read: measurement direction attached to a session. Each tick,
// a dot product at the attached layer is written into a host-visible
// intensity buffer; pump reads it back after CB completion and feeds
// the trigger evaluator. Detectors are purely observational — they
// don't modify the residual stream.
final class DetectorAttachment {
    let name: String           // session-scoped alias used by triggers
    let cvecId: String         // registry key for the underlying vector
    let buffer: MTLBuffer      // HIDDEN × fp16
    let layer: Int             // which LM layer's post-FFN residual to measure
    var lastIntensity: Float = 0      // this tick's reading (post-step update)
    var prevIntensity: Float = 0      // last tick's reading (for edge detection)
    init(name: String, cvecId: String, buffer: MTLBuffer, layer: Int) {
        self.name = name; self.cvecId = cvecId; self.buffer = buffer; self.layer = layer
    }
}

// Gate condition for detector → effector coupling. Edge-triggered: fires
// ONCE per threshold crossing, not every tick where the condition holds.
// Passing the prev→curr pair is enough to implement both rise and fall
// edges without any state beyond what the detector already carries.
enum TriggerCondition {
    case onExceed(threshold: Float)   // rising edge past threshold
    case onFall(threshold: Float)     // falling edge past threshold
    func fires(prev: Float, curr: Float) -> Bool {
        switch self {
        case .onExceed(let th): return prev <= th && curr > th
        case .onFall(let th):   return prev >= th && curr < th
        }
    }
}

// Detector → effector gate. When the condition fires, the matching
// effector's ADSR envelope is RESTARTED (startPosition ← now,
// stopPosition ← nil) so its attack/decay plays out fresh from the
// trigger instant. Triggers carry no continuous coupling to the
// detector's intensity — only the edge event transfers.
struct SessionTrigger {
    let detectorName: String
    let condition: TriggerCondition
    let effectorCvecId: String
}

final class Session {
    let id: Int
    // Termination cause exposed to the bridge. 0 while running; set to
    // 1 (stop / EOS), 2 (length / max_tokens), or 3 (error) when state
    // transitions to .done. The bridge maps these onto OpenAI-shape
    // finish_reason. See ffi_batch.swift:sessionDoneReason which now
    // reads this field directly.
    var doneReason: UInt8 = 0
    // Optional human-readable explanation when doneReason == 3 (error).
    // Surfaces back to the bridge via StreamUpdate.errMsg → HTTP 5xx
    // body. Empty for non-error terminations.
    var errMsg: String = ""
    // True between a sampled `<|channel>` (id 100) and its matching
    // `<channel|>` (id 101). gemma-4 emits empty thinking-channel
    // blocks (`<|channel>thought\n<channel|>`) at the start of every
    // assistant turn as a learned formatting habit; with
    // enable_thinking=False the contents are scaffolding, not
    // user-visible content. Tokens sampled while this flag is set
    // are NOT pushed to outputQueue, so streaming clients never see
    // the channel block. They still count toward numGenerated /
    // history (consumedTokens) so the model's own self-attention
    // sees its own emissions normally. The flag clears when token
    // 101 is sampled (also dropped from outputQueue). See validation
    // artifact: tools/st-debug/tests/test-results/
    // 21_post_cleanup_validation/04_after_response.png — without
    // the suppression, the channel body leaks into chat as
    // "thought\n*gasps* tessellation time!!" prefix.
    var inChannelBlock: Bool = false
    // 2026-05-07 (M:K permutation): `slot` is now per-CB scratch.
    //   - Set by the engine's per-CB picker (pickKernelMappingForAR /
    //     pickKernelMappingForPrefill) BEFORE each CB runs, to the
    //     kernel-batch position [0..B) the session is currently
    //     occupying for THIS CB.
    //   - Cleared/repicked at the start of the NEXT CB.
    //   - Read by existing prepare/finalize code that uses s.slot to
    //     index kernel buffers (input_tokens, sampling, block_table).
    //
    // The previous semantics (persistent slot ownership for the life
    // of the session, set by runAdmissionPass) is gone — sessions
    // live in residentSessions[Int: Session] indefinitely (M >> B
    // logical sessions allowed) and rotate through kernel positions.
    var slot: Int?
    // 2026-05-07: client-controlled flags moved onto Session so the
    // engine doesn't depend on bridge-side dictionaries. Set by
    // applyStreamAction at start; engine reads them per CB to
    // populate gpu_capture_active / gpu_capture_topk for the
    // chosen kpos.
    var captureLogits: Bool = false
    var topLogprobs: UInt32 = 0
    let eosId: UInt32
    var maxNewTokens: Int
    // Sampling temperature. 0 (default) = greedy argmax, preserving
    // deterministic behavior for cache-replay + steering-comparison
    // demos. Positive values enable softmax sampling with the session's
    // own RNG (trajectory variation visible across re-runs).
    var samplingTemperature: Float = 0.0
    // Multi-token stop sequences. Each inner array is a contiguous run
    // of token IDs that, when matched against the recent emitted tail,
    // terminates the stream with done_reason=1 (EOS-equivalent). Used
    // for tool-call early-stop: bridge tokenizes "<tool_call|>" once
    // and passes it here so the engine self-terminates without the
    // bridge needing to detect-and-cancel from outside.
    var stopSequences: [[UInt32]] = []
    // min_p threshold (0 disables). Dense additive logit bias, lazily
    // materialized as VOCAB-length fp32 from the FFI sparse setter;
    // nil means "no bias," the hot path shortcuts on that.
    var minP: Float = 0.0
    var logitBiasDense: [Float]? = nil
    // Optional structured-cot grammar state. When non-nil, the AR
    // sampler's bias buffer is overwritten per-step to mask invalid
    // tokens for the current grammar phase. Cleared (set nil) when the
    // grammar reaches done. See CotState above.
    var cot: CotState? = nil
    // Seedable RNG — FFI swaps in a fresh SeedableRNG(seed:) per request
    // when the client passes `seed`. Default-seeded instance still varies
    // per-session due to gemma_session_set_seed being optional.
    var rng = SeedableRNG()
    // GPU-side RNG seed for the `sample_token` kernel (see
    // docs/dataflow_pipeline_spec.md §2.1). Keyed into philox as
    // (gpuRngSeed, numGenerated_as_step, slot). Initialized randomly at
    // session creation; updated when the client sets a new seed via
    // gemma_session_set_seed so CPU/GPU paths stay in lockstep.
    var gpuRngSeed: UInt32 = UInt32.random(in: 0 ..< UInt32.max)

    // Scope-helper so the four sample sites can pass an UnsafePointer<Float>?
    // into sampleTokenFromLogits without each one writing the same
    // withUnsafeBufferPointer dance. `nil` path is the common case and stays
    // zero-cost.
    @inline(__always)
    fileprivate weak var engine: LmEngine?

    // Default .priming: under atomic construction (LmEngine.submitRequest)
    // a Session is created with its first segment already queued. The
    // pre-submit window where state was .idle is gone.
    fileprivate(set) var state: SessionState = .priming
    // Phys-page IDs owned by this session, in logical-page order:
    //   ownedPages[p] = phys page number for this session's page p
    //   ownedPages.count * PAGE_SLIDE bounds the max k_len this session
    //   can reach without growing. Pages are allocated on-demand as the
    //   session's position advances (see growPagesFor(kLen:)).
    fileprivate var ownedPages: [Int] = []
    // Diagnostic + test accessor: number of phys pages this session
    // owns. Read after submit() to measure cache-adoption counts.
    var ownedPageCount: Int { ownedPages.count }
    // Full token history for per-page prefix hashing. Append-only —
    // submit() extends it, and hash(consumedTokens[0..(P+1)*PAGE]) is
    // the content identity of logical page P. Used for:
    //   (a) cache probe at first-submit: find shared pages in the global
    //       PageManager content index and adopt them read-only
    //   (b) post-prefill promotion: announce newly-written pages to the
    //       content index so the NEXT session with the same prefix hits.
    fileprivate var consumedTokens: [UInt32] = []
    // Logical pages already published to the prefix trie. Kept so we
    // don't re-promote on every prefill tile commit. (Pre-Followup 3:
    // tracked promotion into the deleted PageManager.contentIndex.)
    fileprivate var myPromotedSlidePairCount: Int = 0
    // Track B: prompt-end position (high-water mark of teacher-forced
    // input). Updated when prefill drains or when adoption sets
    // position before the first generated token. Used at teardown to
    // promote a partial-pair anchor at the PROMPT boundary, not the
    // generation boundary (generation tokens diverge across sessions).
    fileprivate var promptEndPosition: Int = 0
    // Per-stream cache accounting (in tokens). Surfaced via the batch FFI
    // as billing-line items in BatchResponse.StreamUpdate. cacheHitTokens
    // counts tokens covered by adopted cache pages on this stream's submits;
    // cacheMissTokens counts tokens this stream had to prefill itself.
    var cacheHitTokens: UInt32 = 0
    var cacheMissTokens: UInt32 = 0
    // Ordered chunks to teacher-force (text tokens or image soft tokens).
    fileprivate var primingQueue: [PrimingChunk] = []
    // Synthetic prefix K/V staging. When non-nil, the next prefill of
    // this session first allocates at least one page, writes these
    // bytes to position 0 of that page per layer, and bumps position
    // to 1 so real prefill tokens start at position 1+. K and V are
    // arrays of per-layer blobs; each blob is KV_H_L × HD_L fp16
    // halves = KV_H_L × HD_L × 2 bytes.
    internal var prefixKvStaged: (k: [Data], v: [Data])? = nil
    internal var prefixKvInstalled: Bool = false
    // When state == .generating, the last-sampled token becomes the next
    // step's input; kept separate from the chunk queue for state clarity.
    fileprivate var nextGeneratedInput: UInt32 = 0
    // forceNextInput / pauseForExternal / resumeFromExternalPause /
    // pausedStateCache deleted 2026-05-07: under stateless chat-
    // completions the bridge never reaches past the API to pause or
    // teacher-force a live request. Each new conversation turn is a
    // fresh request submitted with the full token history; content-hash
    // KV-page sharing is the *only* mechanism for skipping prefix work
    // (no private cross-request reach-arounds).

    // Next KV-cache write position. k_len after a step == position + 1.
    fileprivate var position: Int = 0
    fileprivate var numGenerated: Int = 0

    // Control-vector state (Phase B). activeControls is evaluated once
    // per tick inside step(); turnIndex is set at construction (always
    // 0 under stateless chat-completions — one turn per request).
    fileprivate(set) var activeControls: [ActiveControl] = []
    fileprivate(set) var turnIndex: Int = 0

    // Phase C-Read state. Detectors produce per-tick intensity readings;
    // triggers gate effector-envelope restarts on edge events.
    fileprivate(set) var detectors: [DetectorAttachment] = []
    fileprivate(set) var triggers: [SessionTrigger] = []

    // Per-emitted-token telemetry for the steering UI. One sample per
    // token, captured at emit time with that step's detector intensities
    // and effector magnitudes. Drained by gemma_session_poll_samples_json
    // between pump cycles — same pattern as the output token queue.
    struct TokenSample {
        let token: UInt32
        let position: Int
        // Keep as ordered pairs so the JSON serializer has a stable order
        // and clients can correlate sample[i].name with their UI list.
        let detectors: [(name: String, intensity: Float)]
        // effectors now carry a projection field: the pre-write natural
        // activation level the project-mode kernel measured this tick,
        // or nil for additive-mode controls.
        let effectors: [(cvecId: String, magnitude: Float, projection: Float?)]
    }
    fileprivate var pendingSamples: [TokenSample] = []

    func addControl(_ c: ActiveControl) { activeControls.append(c) }
    func removeControls(cvecId: String) {
        activeControls.removeAll { $0.cvecId == cvecId }
    }
    func clearControls() { activeControls.removeAll(keepingCapacity: false) }

    // Digest every ActiveControl whose envelope window intersects the
    // position range [pageStart, pageStart + pageSize). Output feeds
    // PageManager.hashPage(_:cvecDigest:) so the content-cache key
    // partitions by cvec state as well as tokens.
    func cvecDigestForPage(pageStart: Int, pageSize: Int) -> UInt64 {
        return computeCvecDigest(activeControls: activeControls,
                                  pageStart: pageStart, pageSize: pageSize)
    }

    // Track C / D: per-anchor cvec partition tag. Replaces cvecDigest
    // as the trie's per-anchor partition key. Phase-gated, units-gated,
    // float-quantized so 1-ulp transport-coefficient drift doesn't
    // false-partition. Returns `.unsteered` when no control meaningfully
    // perturbs K/V at this page.
    func cvecAnchorTagForPage(pageStart: Int, pageSize: Int) -> CvecAnchorTag {
        return computeCvecAnchorTag(activeControls: activeControls,
                                     pageStart: pageStart, pageSize: pageSize)
    }

    // Apply a client-provided seed to both host and GPU sampling state.
    // Seed 0 means "unspecified" on the batch FFI wire, so preserve the
    // randomly initialized per-session GPU seed in that case.
    func applySamplingSeed(_ seed: UInt64) {
        guard seed != 0 else { return }
        rng = SeedableRNG(seed: seed)
        var mixer = SeedableRNG(seed: seed)
        let mixed = mixer.next()
        gpuRngSeed = UInt32(truncatingIfNeeded: mixed ^ (mixed >> 32))
    }

    // Signal a control's sustain → release transition. Caller passes the
    // current (position, turn) so the ADSR decays relative to that point.
    func releaseControl(cvecId: String, position: Int, turn: Int) {
        for c in activeControls where c.cvecId == cvecId && c.stopPosition == nil {
            c.stopPosition = position; c.stopTurn = turn
        }
    }
    // Detector / trigger APIs — detectors added via FFI, triggers are
    // pure value types (no buffer refs) so attachment is trivial.
    func addDetector(_ d: DetectorAttachment) { detectors.append(d) }
    func removeDetectors(name: String) { detectors.removeAll { $0.name == name } }
    func clearDetectors() { detectors.removeAll(keepingCapacity: false) }
    func addTrigger(_ t: SessionTrigger) { triggers.append(t) }
    func clearTriggers() { triggers.removeAll(keepingCapacity: false) }
    // Read-only accessors for the pump (intensity readback, trigger eval).
    var detectorCount: Int { detectors.count }
    func detectorAt(_ i: Int) -> DetectorAttachment { detectors[i] }
    var pendingSamplesCount: Int { pendingSamples.count }
    // Record a per-token sample at emit time. Skips the work entirely
    // when the session has neither detectors nor controls — cost is zero
    // for non-steered sessions. Called once per emitted token from step().
    func recordSample(token: UInt32) {
        if detectors.isEmpty && activeControls.isEmpty { return }
        var dets: [(name: String, intensity: Float)] = []
        dets.reserveCapacity(detectors.count)
        for d in detectors { dets.append((name: d.name, intensity: d.lastIntensity)) }
        var effs: [(cvecId: String, magnitude: Float, projection: Float?)] = []
        effs.reserveCapacity(activeControls.count)
        for c in activeControls {
            let m = c.magnitudeAt(position: position, turn: turnIndex)
            // For project-mode controls, the projection field carries
            // the pre-write natural activation level. For additive,
            // projection is nil (the additive kernel doesn't measure).
            effs.append((cvecId: c.cvecId, magnitude: m,
                          projection: c.lastProjectMeasurement))
        }
        pendingSamples.append(TokenSample(token: token, position: position,
                                           detectors: dets, effectors: effs))
    }
    // Drain pending samples into a JSON string. Called by the FFI; the
    // bridge polls this alongside the token queue and streams the JSON
    // into an SSE side-channel frame the UI consumes for its heatmap.
    func drainSamplesJson() -> String {
        if pendingSamples.isEmpty { return "[]" }
        var out = "["
        for (i, s) in pendingSamples.enumerated() {
            if i > 0 { out += "," }
            let dets = s.detectors.map { "\"\(jsonEscape($0.name))\":\($0.intensity)" }.joined(separator: ",")
            // Effectors carry magnitude (scheduled envelope value) +
            // optional projection (pre-write natural activation,
            // project-mode only). Projections are serialized under a
            // separate "projections" key so additive-mode clients that
            // only parse "effectors" keep working unchanged.
            let effs = s.effectors.map { "\"\(jsonEscape($0.cvecId))\":\($0.magnitude)" }.joined(separator: ",")
            let projs = s.effectors.compactMap { (cid, _, pr) -> String? in
                guard let p = pr else { return nil }
                return "\"\(jsonEscape(cid))\":\(p)"
            }.joined(separator: ",")
            out += "{\"token\":\(s.token),\"position\":\(s.position),"
            out += "\"detectors\":{\(dets)},\"effectors\":{\(effs)},"
            out += "\"projections\":{\(projs)}}"
        }
        out += "]"
        pendingSamples.removeAll(keepingCapacity: true)
        return out
    }
    private func jsonEscape(_ s: String) -> String {
        // Escape only what matters for identifiers (backslash + quote);
        // names can't contain control chars in practice.
        return s.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
    }
    // Evaluate every trigger against current detector state, firing
    // effector restarts as edges cross. Called by step() AFTER intensity
    // readback and BEFORE the next tick's dispatch, so restarts affect
    // the immediately-following tick's envelope magnitude.
    func evaluateTriggers(position: Int, turn: Int,
                           log: ((String) -> Void)? = nil) {
        for tr in triggers {
            guard let det = detectors.first(where: { $0.name == tr.detectorName }) else { continue }
            if tr.condition.fires(prev: det.prevIntensity, curr: det.lastIntensity) {
                for c in activeControls where c.cvecId == tr.effectorCvecId {
                    c.restart(position: position, turn: turn)
                }
                log?("[cvec-trigger] s\(id) detector=\(tr.detectorName) prev=\(det.prevIntensity) curr=\(det.lastIntensity) → restart \(tr.effectorCvecId)")
            }
        }
    }
    // Snapshot of (position, turn) at the moment the caller reads them —
    // used by the FFI when activating a control so the envelope's start
    // time is captured relative to session state without exposing the
    // fileprivate fields.
    var currentTimeCoords: (position: Int, turn: Int) {
        return (position, turnIndex)
    }

    // Tokens the caller can consume.
    fileprivate var outputQueue: [UInt32] = []

    // Thinking-channel content. Populated by maybeAppendOutput when
    // `enableThinking` is true and the session is inside a
    // `<|channel>...<channel|>` block. The bridge surfaces these to
    // the client as `reasoning_content` (OpenAI o1-shape). When
    // `enableThinking` is false, channel-block content is dropped
    // (the model echoes a `<|channel>thought\n<channel|>` empty
    // marker as a learned habit; without thinking enabled it's
    // scaffolding the user shouldn't see).
    fileprivate var thinkingQueue: [UInt32] = []
    var enableThinking: Bool = false

    // Buffer for channel-block tokens generated when enableThinking
    // is false. Normally these are dropped (thinking scaffold, client
    // didn't ask for it). But if the model emits CHANNEL_OPEN_ID and
    // hits EOS/max_tokens before CHANNEL_CLOSE_ID — a learned-habit
    // failure mode where the model writes its WHOLE response inside a
    // thinking channel that never gets closed — we recover the
    // contents at .done via flushUnclosedChannel() so the response
    // isn't silently empty. See docs/streamed_toolcall_marker_leak.md
    // for the related rendering issue and 2026-05-10 user report.
    fileprivate var pendingChannelTokens: [UInt32] = []

    // Cvec activations capture. When `captureCvecActivations` is
    // true, every (ActiveControl, AR step) pair pushes one record
    // onto `cvecActivationsQueue`; the FFI pump drains the queue
    // and surfaces records as `choices[0].delta.cvec_activations`
    // in the bridge's SSE stream. When the flag is false, the
    // capture path is dormant — recordCvecActivation is a no-op
    // early-return — so the apply path bit-equals pre-
    // instrumentation behavior. Validated by
    // tools/cvec_validation/baseline_logprobs.py.
    var captureCvecActivations: Bool = false
    fileprivate var cvecActivationsQueue: [(tokenPosition: UInt32,
                                            layer: UInt32,
                                            magnitude: Float)] = []

    // ────────────────────────────────────────────────────────────────────
    // Bridge-side per-request state, pulled inside the request 2026-05-07
    // ────────────────────────────────────────────────────────────────────
    // Previously these lived in `private var gFoo: [UInt64: ...]`
    // dictionaries inside ffi_batch.swift, indexed by stream_id with a
    // companion gStreamToSession[UInt64: Session] to resolve them. That
    // was the long-range client-server coupling the user objected to:
    // bridge-side state survives across requests, indexed by an external
    // ID. Now each request owns its own state and the bridge is a thin
    // marshaling layer over `engine.requestForStream[sid]`.
    //
    // streamId: the bridge's external request identity. Set by the
    // bridge immediately after openSession returns. 0 = no bridge.
    var streamId: UInt64 = 0
    // Token-accounting that gets surfaced via StreamUpdate (billing).
    var usage = StreamUsage()
    // Logprob capture: previously `gCaptureLogits: Set<UInt64>` and
    // `gTopLogprobs: [UInt64: UInt32]`. captureLogits and topLogprobs
    // already live on Session above; this is the per-request emit queue
    // and the high-water mark of "outputs we've already reported."
    var logprobsQueue: [LogprobRecord] = []
    var lastReportedOutputCount: Int = 0

    // Forward-progress liveness pair. The engine-internal stuck-
    // detection layer. After the 2026-05-23 bridge⇄engine separation
    // there is no consumer-coupling here — this is purely "did my own
    // position counter advance?" The bridge owns disconnect detection
    // and queue-depth backpressure on its side; the engine self-cleans
    // any session whose own forward progress stalls.
    //
    // Signal: `position` strictly increases during BOTH prefill (every
    // chunk consumed advances by chunk size) AND AR generation (every
    // sampled token advances by 1). If position hasn't advanced for
    // engineProgressDeadline seconds AND the session is in .priming
    // or .generating, the session is stuck and gets transitioned to
    // .done with doneReason=3 + an explanatory errMsg.
    //
    // Chose position-advance over "tokens drained by FFI pump" because
    // it handles long prefills correctly: the pump's drainSession
    // returns [] during prefill (no AR tokens emit) but position is
    // advancing chunk by chunk. A fanout-based signal would falsely
    // expire those sessions.
    //
    // Maintained entirely by expireStalledSessions (called from
    // gemma_poll's drive loop); no other call site touches these
    // fields. bindStream initializes both at request-time so a brand-
    // new session gets ≥1 deadline window of grace.
    var lastSeenPosition: Int = 0
    var lastForwardProgressAt: Date = Date()

    fileprivate init(id: Int, eosId: UInt32, maxNewTokens: Int, engine: LmEngine) {
        self.id = id; self.slot = nil
        self.eosId = eosId; self.maxNewTokens = maxNewTokens
        self.engine = engine
    }

    // Queue more input tokens to be teacher-forced. Valid in any state:
    // calling while .generating flips back to .priming. .done is
    // terminal — submit on a .done session is a no-op.
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
    // Enable structured-cot grammar (github.com/andthattoo/structured-cot).
    // Forces the model's output to begin with `<think>\nLABEL_0: <free>\n
    // LABEL_1: <free>\n ... </think>\n\n` then decode unconstrained.
    // Empty `labels` → no-op. Default labels are GOAL / APPROACH / EDGE
    // (the HumanEval+ grammar from the reference repo).
    func enableStructuredCot(labels: [String] = ["GOAL", "APPROACH", "EDGE"]) {
        guard !labels.isEmpty else { cot = nil; return }
        var phases: [CotPhase] = []
        phases.append(.literal(bytes: Array("<think>\n".utf8)))
        for label in labels {
            phases.append(.literal(bytes: Array((label + ": ").utf8)))
            phases.append(.freeLine)
        }
        phases.append(.literal(bytes: Array("</think>\n\n".utf8)))
        cot = CotState(phases: phases)
    }

    func submit(_ tokens: [UInt32]) {
        guard !tokens.isEmpty else { return }
        guard let eng = engine else {
            primingQueue.append(.tokens(tokens))
            if state != .done { state = .priming }
            return
        }
        // Extend the canonical history (used for hashing).
        consumedTokens.append(contentsOf: tokens)
        // First-submit cache probe.
        let firstSubmit = (position == 0 && ownedPages.isEmpty)
        var adoptedTokens = 0           // total tokens covered (full + partial)
        var adoptedFullTokens = 0       // tokens covered by full page-pairs only
        if firstSubmit {
            (adoptedFullTokens, adoptedTokens) = adoptSharedPrefixPages(engine: eng)
            // Track A fold-in: the old backstop unadopted the trailing page
            // whenever adoption would leave primingQueue empty. Under the
            // .primed state machine that's gone — we set state=.primed
            // instead and let prepareARStep run a 1-token recover tick.
            position = adoptedTokens
            cacheHitTokens &+= UInt32(adoptedTokens)
        }
        // Track B: mark prompt-end at the *final* prompt position
        // (after all submits). We update this on every submit so it
        // grows with the accumulated prompt history.
        promptEndPosition = consumedTokens.count
        // Queue the un-cached tail for prefill.
        let tailStart = adoptedTokens
        if tailStart < tokens.count {
            let tail = Array(tokens[tailStart...])
            if !tail.isEmpty {
                primingQueue.append(.tokens(tail))
                cacheMissTokens &+= UInt32(tail.count)
            }
        }
        // State transition: .primed when fully cached at >0 position
        // (Track A path). .priming otherwise.
        if state != .done {
            if primingQueue.isEmpty && position > 0 {
                state = .primed
            } else {
                state = .priming
            }
        }
        // _ keep adoptedFullTokens reserved for future partial-pair fixups
        _ = adoptedFullTokens
        // If this is NOT the first submit (firstSubmit was false above),
        // we still want to probe the cache against the now-extended
        // consumedTokens. The first-submit path already probed; this is
        // for the multi-segment case (text → image-softs → text) where
        // earlier probes saw an incomplete prefix.
        if !firstSubmit {
            revisitCacheProbe(engine: eng)
        }
    }
    func submit(text: String, addBos: Bool? = nil) {
        guard let eng = engine else { return }
        submit(eng.tokenizer.encode(text, addBos: addBos))
    }

    // adoptKvFrom deleted 2026-05-07: explicit cross-request KV-page
    // sharing was a private reach-around the bridge layer used for
    // "same image, multiple suffixes." Under the anonymous-pool /
    // content-hash design, the only way one request shares another's
    // pages is via the prefix trie (RadixTrie) — content-addressed,
    // no identity coupling. Callers that need image-prefix sharing
    // should arrange for the soft-token fp16 buffer to hash
    // deterministically. (Pre-Followup 3 the source of truth was
    // PageManager.contentIndex, since deleted.)

    // Walk the PageManager's content index for leading pages of
    // consumedTokens. Returns the number of SLIDE pages successfully
    // adopted (each adopted slide page appends TWO phys pages to
    // ownedPages — the slide primary and the full-sibling — because
    // Gemma-4's full-attn uses half the page size of its slide-attn
    // layers and a single 16-token slide page straddles two full pages
    // in the full-K/V cache).
    //
    // Hash key includes a per-page cvec-state digest (see
    // cvecDigestForPage). Sessions with identical tokens but different
    // active controls get different keys and correctly MISS each
    // other's pages — preventing steered K/V from polluting unsteered
    // sessions (and vice versa).
    // Returns (alignedTokensAdoptedThisCall, totalTokensAdoptedThisCall).
    // Total may exceed aligned by [1, PAGE_SLIDE-1] when a partial pair
    // is CoW-copied onto fresh pages (Track B fold-in).
    private func adoptSharedPrefixPages(engine: LmEngine) -> (Int, Int) {
        // Idempotent: resume from how many slide pages we've already
        // adopted (each contributes 2 entries to ownedPages — slide
        // primary + full sibling).
        let beforeAdopted = ownedPages.count / 2
        let dbg = ProcessInfo.processInfo.environment["LM_CACHE_DEBUG"] != nil
        // Build closure capturing self's cvecAnchorTagForPage.
        let match = engine.prefixTrie.findLongestPrefix(
            tokens: consumedTokens[0..<consumedTokens.count],
            cvecTagFor: { pageStart in
                self.cvecAnchorTagForPage(pageStart: pageStart, pageSize: PAGE_SLIDE)
            })
        // We only want the *new* portion past what we've already adopted.
        let alreadyAdoptedTokens = beforeAdopted * PAGE_SLIDE
        let alignedMatch = match.alignedMatchLength
        guard alignedMatch >= alreadyAdoptedTokens else {
            // Nothing new. (Possible if the trie was trimmed by eviction
            // since the last call; we don't regress already-adopted pages.)
            return (0, 0)
        }
        // Adopt the new full pairs.
        for i in beforeAdopted..<match.pages.count {
            let pair = match.pages[i]
            engine.pageManager.incref(physPage: pair.slidePlusFullHead)
            engine.pageManager.incref(physPage: pair.fullTail)
            ownedPages.append(pair.slidePlusFullHead)
            ownedPages.append(pair.fullTail)
        }
        let newAlignedAdopted = (match.pages.count - beforeAdopted) * PAGE_SLIDE
        if alignedMatch > 0 && newAlignedAdopted > 0 {
            myPromotedSlidePairCount = match.pages.count
            if dbg {
                print("  [cache] session \(id) adopted \(match.pages.count - beforeAdopted) "
                      + "additional shared prefix pages via trie (total \(match.pages.count) = "
                      + "\(match.pages.count * PAGE_SLIDE) tokens; trie matched \(match.trieMatchLength))")
            }
        }
        // Partial-pair adoption (Track B fold-in): if the trie returned
        // a partial extension past the aligned match, allocate FRESH
        // pages and CoW-copy the partial's K/V bytes onto them. The
        // consumer's prefill then writes positions [validUpTo, ...)
        // into pages it solely owns — no shared writes, race-safe for
        // multi-consumer concurrent extends.
        //
        // Followup 1 (Grand-slam 2026-05-23): replaces the previous
        // direct-incref-and-share approach, which was race-safe only
        // for serialized workloads. The producer continues to hold
        // its partial-pair pages via PageManager's content-index /
        // trie back-pointer, so the producer-side pages are not
        // reclaimed by this consumer's adoption — they remain
        // available for future adopters.
        //
        // Operator escape hatch: LM_PARTIAL_COW_DISABLE=1 reverts to
        // the old direct-incref behavior for A/B comparison.
        var newPartialAdopted = 0
        if let pp = match.partialTail {
            let cowDisabled = ProcessInfo.processInfo.environment["LM_PARTIAL_COW_DISABLE"] != nil
            if cowDisabled {
                // Legacy direct-incref behavior (race-unsafe on concurrent extends).
                engine.pageManager.incref(physPage: pp.slidePlusFullHead)
                ownedPages.append(pp.slidePlusFullHead)
                if let ft = pp.fullTail {
                    engine.pageManager.incref(physPage: ft)
                    ownedPages.append(ft)
                } else {
                    do {
                        let fresh = try engine.pageManager.allocFresh()
                        engine.zeroPhysPageKV(fresh)
                        ownedPages.append(fresh)
                    } catch {
                        engine.pageManager.decref(physPage: pp.slidePlusFullHead)
                        ownedPages.removeLast()
                        return (newAlignedAdopted, newAlignedAdopted)
                    }
                }
                newPartialAdopted = pp.validUpTo
            } else {
                // CoW path: allocate fresh pair, copy bytes from partial.
                let freshHead: Int
                let freshTail: Int
                do {
                    freshHead = try engine.pageManager.allocFresh()
                    engine.zeroPhysPageKV(freshHead)
                    freshTail = try engine.pageManager.allocFresh()
                    engine.zeroPhysPageKV(freshTail)
                } catch {
                    // Couldn't allocate fresh pages; abandon the partial
                    // (we already have the full-aligned pages above).
                    return (newAlignedAdopted, newAlignedAdopted)
                }
                // Copy K/V bytes for [0, PAGE_SLIDE) worth of rows
                // (whole-page copy; positions past validUpTo are either
                // producer-written-but-soon-overwritten or zero).
                engine.copyPhysPageKV(srcPhys: pp.slidePlusFullHead,
                                       dstPhys: freshHead)
                if let ft = pp.fullTail {
                    engine.copyPhysPageKV(srcPhys: ft, dstPhys: freshTail)
                }
                // Note: we do NOT decref the original partial pages —
                // the producer (via PageManager.contentIndex / trie
                // anchor) still holds them; they remain adoptable for
                // future consumers.
                ownedPages.append(freshHead)
                ownedPages.append(freshTail)
                newPartialAdopted = pp.validUpTo
                if dbg {
                    print("  [cache] session \(id) partial-CoW: head \(pp.slidePlusFullHead)→\(freshHead), "
                          + "tail \(pp.fullTail.map(String.init) ?? "nil")→\(freshTail), validUpTo=\(pp.validUpTo)")
                }
            }
            // Don't bump myPromotedSlidePairCount — the partial isn't a full page
            // yet from this session's perspective.
        }
        return (newAlignedAdopted, newAlignedAdopted + newPartialAdopted)
    }

    // Re-probe the cache after consumedTokens has grown (e.g. additional
    // text or soft submit). Adopts any newly-matching pages, advances
    // `position` past them, and trims primingQueue from the front so prefill
    // doesn't redundantly compute K/V for already-cached positions.
    fileprivate func revisitCacheProbe(engine: LmEngine) {
        guard position == 0 || ownedPages.count > 0 else { return }
        if ProcessInfo.processInfo.environment["LM_CACHE_DEBUG"] != nil {
            print("  [cache] session \(id) revisitProbe consumedTokens.count=\(consumedTokens.count) ownedPages.count=\(ownedPages.count) position=\(position)")
        }
        let (newlyAlignedTokens, _) = adoptSharedPrefixPages(engine: engine)
        guard newlyAlignedTokens > 0 else { return }
        // Track A fold-in: backstop is gone. Sessions whose primingQueue
        // empties out get .primed state and the recover-tick path picks
        // them up to produce the first sampled token.
        var advance = newlyAlignedTokens
        position += advance
        // Move billing from miss → hit. The tokens we just adopted were
        // queued by an earlier submit() call, which accounted for them
        // as misses in `cacheMissTokens`. Now that we've recognized them
        // as cache hits, swap the books.
        let advU = UInt32(advance)
        cacheHitTokens &+= advU
        if cacheMissTokens >= advU {
            cacheMissTokens &-= advU
        } else {
            cacheMissTokens = 0
        }
        // Drop chunks (or chunk prefixes) covering positions we just adopted.
        while advance > 0 && !primingQueue.isEmpty {
            let head = primingQueue[0]
            let headCount = head.count
            if headCount <= advance {
                primingQueue.removeFirst()
                advance -= headCount
            } else {
                switch head {
                case .tokens(let ts):
                    primingQueue[0] = .tokens(Array(ts[advance...]))
                case .softTokens(let buf, let count, let isFp32, let byteOff, let evt):
                    let bytesPerTok = isFp32 ? (HIDDEN * 4) : (HIDDEN * 2)
                    primingQueue[0] = .softTokens(buffer: buf,
                                                 count: count - advance,
                                                 isFp32: isFp32,
                                                 byteOffset: byteOff + advance * bytesPerTok,
                                                 eventTicket: evt)
                }
                advance = 0
            }
        }
        // If we drained the queue and have a non-zero position, transition
        // to .primed so prepareARStep runs a recover tick.
        if state != .done && primingQueue.isEmpty && position > 0 {
            state = .primed
        }
    }
    func submit(softTokens: MTLBuffer, count: Int, isFp32: Bool,
                  eventTicket: UInt64 = 0, contentHash: UInt64 = 0) {
        guard count > 0 else { return }
        primingQueue.append(.softTokens(buffer: softTokens, count: count,
                                       isFp32: isFp32, byteOffset: 0,
                                       eventTicket: eventTicket))
        // Extend consumedTokens with content-derived placeholders for the
        // soft positions, so:
        //   - promoteFinishedPages requires end <= consumedTokens.count,
        //     so pages straddling soft positions can promote
        //   - adoptSharedPrefixPages walks consumedTokens, so cache probes
        //     see image identity in subsequent sessions
        //
        // CRITICAL: hash the image's INPUT BYTES, not the soft-token buffer.
        // The soft buffer is GPU-written by the vision tower on its own
        // queue and may be all zeros (or stale) at the moment submit runs
        // (caller sequences readers via the eventTicket but CPU reads
        // here would race the GPU writes). Using the input image hash
        // gives a stable, identical key across iters of the same image.
        // Caller passes contentHash from gVisionCache's image-bytes key.
        let imgHash = contentHash != 0 ? contentHash : 0xcbf29ce484222325
        if ProcessInfo.processInfo.environment["LM_CACHE_DEBUG"] != nil {
            print("  [cache] session \(id) submit(softs) imgHash=\(String(imgHash, radix: 16, uppercase: false)) count=\(count) (pre-consumedTokens.count=\(consumedTokens.count))")
        }
        for p in 0..<count {
            // Mix imgHash with position so each soft position is distinct.
            let mixed = imgHash &+ UInt64(p) &* 0x9E3779B97F4A7C15
            consumedTokens.append(UInt32(truncatingIfNeeded: mixed))
        }
        if state != .done { state = .priming }
        promptEndPosition = consumedTokens.count
        // Re-probe the cache now that consumedTokens has grown.
        if let eng = engine {
            revisitCacheProbe(engine: eng)
        }
    }

    // pause() / append() deleted 2026-05-07: multi-turn-with-retained-
    // KV ("inject tool result back into a paused session") was the
    // motivating use case. Under stateless chat-completions every new
    // turn is a fresh request submitted with the full history — content-
    // hash KV-page adoption recovers the prefix work without any
    // bridge-side notion of "this stream is the continuation of that
    // one." See applyStreamAction in ffi_batch.swift.

    // Pull the next generated token, or nil if none ready.
    func nextToken() -> UInt32? {
        guard !outputQueue.isEmpty else { return nil }
        return outputQueue.removeFirst()
    }

    var pendingOutputCount: Int { outputQueue.count }
    var pendingPrimingTokenCount: Int { primingQueue.reduce(0) { $0 + $1.count } }
    var ownedPagesForDebug: [Int] { ownedPages }
    var positionForDebug: Int { position }
    var primingQueueDepthForDebug: Int { primingQueue.count }
    // Peek the most recent N tokens in outputQueue without removing them.
    // Used by the batch FFI's logprob capture (which must inspect the
    // just-sampled token and pair it with the post-step logits row).
    func peekRecentOutputs(count: Int) -> [UInt32] {
        let n = min(count, outputQueue.count)
        if n == 0 { return [] }
        return Array(outputQueue.suffix(n))
    }

    // Mark as done; engine will release pages + slot on the next tick.
    func finish() { state = .done }

    // Token IDs the engine treats as a self-contained suppression span.
    // gemma-4 emits `<|channel>...<channel|>` blocks at the start of
    // every assistant turn (id 100 → ... → id 101). With
    // enable_thinking=False these are scaffolding, not content. The
    // helper drops the markers AND any tokens between them from the
    // output stream surfaced to the bridge. The tokens still count
    // toward numGenerated and consumedTokens so the model's own
    // self-attention sees a normal sequence.
    static let CHANNEL_OPEN_ID: UInt32 = 100   // <|channel>
    static let CHANNEL_CLOSE_ID: UInt32 = 101  // <channel|>

    func maybeAppendOutput(_ sampled: UInt32) {
        if sampled == Session.CHANNEL_OPEN_ID {
            inChannelBlock = true
            pendingChannelTokens.removeAll(keepingCapacity: true)
            return
        }
        if sampled == Session.CHANNEL_CLOSE_ID {
            inChannelBlock = false
            // Channel closed normally: discard the pending buffer.
            // When enableThinking was on we've already routed those
            // tokens to thinkingQueue in real time; the buffer is only
            // populated as a fallback for the !enableThinking case and
            // a closed channel means the model finished its thinking
            // scaffold properly, so the bytes were legitimate
            // scaffolding the client doesn't need.
            pendingChannelTokens.removeAll(keepingCapacity: true)
            return
        }
        if inChannelBlock {
            // Inside a thinking-channel block. Two routes:
            //   - enableThinking → live-stream to thinkingQueue so the
            //     bridge can emit `delta.reasoning_content` in real time.
            //   - !enableThinking → BUFFER the tokens in
            //     pendingChannelTokens. If the channel closes normally
            //     (CHANNEL_CLOSE_ID branch above) we discard the buffer.
            //     If the session ends with the channel still open
            //     (flushUnclosedChannel called from the FFI pump on
            //     .done) we route the buffered tokens to outputQueue —
            //     the model was thinking but never closed the block,
            //     so the bytes ARE the actual response wearing a
            //     thinking-channel costume. Shipping them as content
            //     prevents the empty-response failure mode the user
            //     reported on 2026-05-10.
            if enableThinking {
                thinkingQueue.append(sampled)
            } else {
                pendingChannelTokens.append(sampled)
            }
            return
        }
        // The EOS / turn-end marker is a stop signal, not response
        // content. Engine sites that follow this call still set
        // state=.done based on `sampled == eosId`, so suppression
        // here just prevents the marker from bleeding into the
        // detokenized client-visible text.
        if sampled == eosId { return }
        outputQueue.append(sampled)
    }

    // Drain pending thinking tokens. Returns and clears whatever has
    // accumulated since the last drain. Called from the FFI pump on
    // each StreamUpdate so the bridge can stream `reasoning_content`
    // alongside main `content`.
    func drainThinking() -> [UInt32] {
        if thinkingQueue.isEmpty { return [] }
        let out = thinkingQueue
        thinkingQueue.removeAll(keepingCapacity: true)
        return out
    }

    // Called by the FFI pump at .done transition (and on explicit
    // closeSession). If the session ends with `inChannelBlock == true`
    // the model emitted CHANNEL_OPEN_ID and never CHANNEL_CLOSE_ID — a
    // failure mode where the model writes its whole response inside an
    // unclosed thinking channel and runs out of tokens before reaching
    // the close. We recover the buffered tokens here:
    //   - enableThinking → tokens already flowed to thinkingQueue in
    //     real time; nothing extra to do. Just clear the in-channel
    //     flag so subsequent reuse (none currently, but defensive)
    //     sees a clean state.
    //   - !enableThinking → buffered tokens were withheld pending the
    //     channel close. Since the close never came, flush them to
    //     outputQueue so the assistant turn isn't silently empty.
    // Idempotent: safe to call multiple times. Safe to call when no
    // channel was ever opened (no-op).
    func flushUnclosedChannel() {
        guard inChannelBlock else {
            pendingChannelTokens.removeAll(keepingCapacity: true)
            return
        }
        if !enableThinking && !pendingChannelTokens.isEmpty {
            outputQueue.append(contentsOf: pendingChannelTokens)
        }
        pendingChannelTokens.removeAll(keepingCapacity: true)
        inChannelBlock = false
    }

    // Record a (token_position, layer, magnitude) tuple for one
    // (control, AR-step) pair. Early-returns when capture is off so
    // the call site stays a single conditional branch with no
    // allocation, matching the off-path bit-identity invariant. Call
    // sites: the AR-step + prefill-step loops in lm_engine.swift
    // where `c.magnitudeAt(position:turn:)` is evaluated.
    func recordCvecActivation(tokenPosition: Int, layer: Int,
                              magnitude: Float) {
        if !captureCvecActivations { return }
        cvecActivationsQueue.append((
            tokenPosition: UInt32(tokenPosition),
            layer: UInt32(layer),
            magnitude: magnitude))
    }

    // Drain pending cvec activation records. Mirrors drainThinking.
    func drainCvecActivations() -> [CvecActivationRecord] {
        if cvecActivationsQueue.isEmpty { return [] }
        let out = cvecActivationsQueue.map {
            CvecActivationRecord(
                tokenPosition: $0.tokenPosition,
                layer: $0.layer,
                magnitude: $0.magnitude)
        }
        cvecActivationsQueue.removeAll(keepingCapacity: true)
        return out
    }
}

// ============================================================================
// Per-request bridge state (moved out of ffi_batch.swift's gFoo dictionaries
// 2026-05-07; see Session.streamId / Session.usage / Session.logprobsQueue
// for the field definitions that hold these). These are simple value types
// with no engine-private semantics — they're "what the bridge needs to
// remember about a request between submit() and poll()."
// ============================================================================
struct StreamUsage {
    var promptTokensSeen: UInt32 = 0
    var completionTokensEmitted: UInt32 = 0
    var visionCacheHits: UInt32 = 0
}

struct LogprobRecord {
    let token: UInt32
    let sampledLogprob: Float
    let topKPairs: [(UInt32, Float)]
}

final class LmEngine {
    let weights: LmWeights
    let tokenizer: GemmaBpe
    // All resident sessions (keyed by id). A session is "resident" if the
    // engine holds its KV pages; it may or may not currently own an active
    // batch slot. Limited to MAX_RESIDENT_SESSIONS — beyond that, callers
    // queue externally (a separate admission layer handles that).
    private(set) var residentSessions: [Int: Session] = [:]
    // 2026-05-07: streamId-keyed lookup. Replaces the bridge's
    // gStreamToSession dictionary. The bridge calls openSession (returns
    // a Session with engine-internal id) then bindStream(s, sid) so the
    // engine can resolve incoming bridge calls by stream_id.
    private(set) var requestForStream: [UInt64: Session] = [:]
    func bindStream(_ s: Session, streamId sid: UInt64) {
        s.streamId = sid
        // Reset progress clock on bind so a freshly-submitted session
        // gets ≥1 full engineProgressDeadline of grace before
        // expireStalledSessions could ever sample its position.
        s.lastSeenPosition = s.position
        s.lastForwardProgressAt = Date()
        requestForStream[sid] = s
    }
    func unbindStream(_ s: Session) {
        if s.streamId != 0 {
            requestForStream.removeValue(forKey: s.streamId)
            s.streamId = 0
        }
    }

    // Engine-progress deadline for the stalled-session reaper.
    //
    // SEMANTIC (2026-05-23 rename): this is NOT a "consumer liveness"
    // signal — the engine does not observe consumer state. The check
    // is purely engine-internal: did the session's `position` advance
    // within the last `engineProgressDeadline` seconds?
    //
    // 60s is generous: a healthy session advances position at AR-tick
    // speed (~25ms per token) or prefill-chunk speed (multiple tokens
    // per chunk). 60s with no position advance means the session is
    // genuinely stuck (an admission-backpressure stream that hasn't
    // seen a slot, a runaway condition in the scheduler, etc.). Such
    // sessions get reaped with doneReason=3.
    //
    // The previous name (`consumerLivenessDeadline`) was misleading:
    // it conflated "engine session not advancing" with "bridge consumer
    // is dead". After the 2026-05-23 separation, those are independent
    // concerns and only the former is the engine's business — the
    // bridge handles all client/consumer disconnect logic on its side
    // and the engine continues regardless. See Session.lastForwardProgressAt.
    static let engineProgressDeadline: TimeInterval = 60.0

    /// Reap sessions whose forward progress has stalled. A session is
    /// stalled when it is in .priming or .generating AND its
    /// `position` hasn't advanced for `engineProgressDeadline`
    /// seconds. We transition each such session to .done with
    /// doneReason=3 (engine error) AND IMMEDIATELY call closeSession
    /// to release pages + unbind from requestForStream on the same
    /// tick. Earlier this function only mutated `s.state` and relied
    /// on the FFI pump's cleanup pass (`for u in updates where u.state
    /// == 2 { closeSession(u) }`) to do the actual reap — but that
    /// cleanup only fires when a state==2 StreamUpdate exists in the
    /// pump's poll output, and expireStalledSessions doesn't emit
    /// one. So reaped sessions stayed in `requestForStream` forever,
    /// accumulating to 14+ over a long session (the 400x slowdown
    /// pathology that motivated task #179). Reap and unbind here.
    ///
    /// Position-advance is the signal because it's monotonically
    /// increasing during both prefill (chunk consumption) and AR
    /// (token generation). A token-drained-by-pump signal would
    /// falsely expire long-prefill sessions whose chunks aren't yet
    /// producing AR tokens.
    ///
    /// 2026-05-23 rename note: was `expireAbandonedSessions`. This is
    /// engine-internal stuck-detection — NOT consumer coupling. The
    /// engine has no view into bridge or client state; it only checks
    /// whether its own position counter advanced. Bridge consumer
    /// liveness is fully owned by the bridge.
    ///
    /// Called from gemma_poll's drive loop before each work step. The
    /// O(active_streams) iteration cost is negligible against the AR-
    /// tick budget — there are typically ≤ 16 active streams.
    func expireStalledSessions() {
        let now = Date()
        let deadline = LmEngine.engineProgressDeadline
        // Collect-then-close: closeSession mutates requestForStream
        // (via unbindStream); iterating that dict's values while
        // removing from it is undefined behavior in Swift's Dictionary.
        var toClose: [Session] = []
        for s in requestForStream.values {
            // Only reap sessions actively asking for slot time. .done
            // sessions get caught by the FFI pump's straggler sweep
            // below — see ffi_batch.swift's gemma_poll cleanup.
            guard s.state.wantsSlot else { continue }
            if s.position > s.lastSeenPosition {
                // Forward progress — refresh snapshot.
                s.lastSeenPosition = s.position
                s.lastForwardProgressAt = now
                continue
            }
            let stalled = now.timeIntervalSince(s.lastForwardProgressAt)
            if stalled > deadline {
                print("[engine] expireStalledSessions: "
                    + "sid=\(s.streamId) state=\(s.state) "
                    + "stalled=\(Int(stalled))s position=\(s.position) "
                    + "numGenerated=\(s.numGenerated) "
                    + "→ .done(engine-progress stall) + closeSession")
                s.state = .done
                s.doneReason = 3   // engine error (surfaces as 500 / SSE error)
                s.errMsg = "engine-progress stall: position stuck at "
                    + "\(s.position) for \(Int(stalled))s "
                    + "(deadline=\(Int(deadline))s, "
                    + "stream_id=\(s.streamId))"
                toClose.append(s)
            }
        }
        for s in toClose {
            closeSession(s)
        }
    }

    /// Defense-in-depth: sweep `requestForStream` for any session that
    /// is .done but still bound. Such stragglers happen when a session
    /// transitions to .done outside the work-step path (e.g., a future
    /// code path that mutates state from a callback) and no
    /// corresponding state==2 StreamUpdate gets emitted for the FFI
    /// pump's cleanup pass to catch. Called from gemma_poll AFTER the
    /// update-driven cleanup so it never duplicates work but always
    /// catches what the update-driven path misses.
    func sweepDoneStragglers() -> Int {
        var toClose: [Session] = []
        for s in requestForStream.values where s.state == .done {
            toClose.append(s)
        }
        for s in toClose {
            print("[engine] sweepDoneStragglers: "
                + "sid=\(s.streamId) doneReason=\(s.doneReason) "
                + "→ closeSession")
            closeSession(s)
        }
        return toClose.count
    }
    // 2026-05-07 (M:K): slotAssignment removed. With the per-CB picker
    // model, kernel positions are assigned dynamically via Session.slot.
    // Keeping the field declarations as @unused stubs would be misleading;
    // they're gone. Code that searches sid → kpos can iterate residents
    // with `where $0.slot == kpos`.
    // Decoupled from residentSessions — sessions move in and out of slots
    // each tick based on `wantsSlot` state.
    private var nextId: Int = 1

    // 2026-05-07 (M:K): admissionCursor + slotAssignment retired. Picker
    // owns its own cursor (pickerCursor); admission is now slot-free.

    // Page manager owns the KV cache's physical page pool.
    let pageManager: PageManager
    // Track D: token-granularity radix trie of cached prefixes.
    // Sole source of truth for adoption lookups (Followup 3 deleted
    // PageManager.contentIndex; the trie's `byCvecAnchorTag` /
    // `byCvecAnchorTagPartial` maps replace it entirely).
    let prefixTrie: RadixTrie

    // Instrumentation — helps the scheduler-behaviour tests measure how
    // well we're batching. One CB per step regardless of active count.
    private(set) var totalSteps: Int = 0
    private(set) var totalTokensGenerated: Int = 0

    // (lockYielder + waitYielded removed Phase 4 — see notes/engine_debloat.md.
    // The wait they used to wrap doesn't exist on the hot path anymore.
    // The harness sync wrappers (step/stepPrefillForSession/...) still
    // call cb.waitUntilCompleted() directly, but they're only used by
    // runUntilIdle and validators, never the production pump.)
    private(set) var lastStepMs: Double = 0
    // Heretic-style write ablations — engine-level, applied uniformly
    // across all batch slots each tick. Configured externally via the
    // FFI; iterated by buildStepCB at the per-layer attn-out and ffn-out
    // hook points. Empty list = no ablation (engine in baseline mode).
    var writeAblations: [LayerComponentAblation] = []
    // Slot occupancy over lifetime: for each non-idle tick, we add the
    // number of busy sessions that were scheduled in (== slot count this
    // tick). avg_active_slots = totalSlotTicks / totalSteps. Exposed via
    // the scheduler-snapshot FFI for bottleneck debugging ("why aren't
    // all B slots occupied when I have ≥B sessions ready?").
    private(set) var totalSlotTicks: Int = 0

    // 2026-05-07: per-CB scratch to avoid heap-allocating 256 KB Swift
    // Array<UInt32> + 65,536-element copy loops every single-slot /
    // multi-slot / soft prefill (3 sites used to do this). Single CB at
    // a time (gEngineLock serializes), so one scratch buffer suffices
    // across all three prefill paths. Sized at engine init to
    // B * MAX_PAGES_PER_SLOT * sizeof(UInt32) = 8 * 8192 * 4 = 256 KB.
    // memcpy in/out instead of per-element loops.
    private let _savedBTScratch: UnsafeMutablePointer<UInt32>
    private let _savedBTScratchByteCount: Int

    init(weights: LmWeights) {
        self.weights = weights
        self.tokenizer = GemmaBpe(weights: weights)
        // 2026-05-07 (M:K): slotAssignment removed. Picker assigns kpos per CB.
        self.pageManager = PageManager(numPhysPages: SCRATCH_PAGE_BASE,
                                        pageSize: PAGE_SLIDE)
        self.prefixTrie = RadixTrie(pageSlide: PAGE_SLIDE)
        self._savedBTScratchByteCount = B * MAX_PAGES_PER_SLOT * MemoryLayout<UInt32>.stride
        self._savedBTScratch = UnsafeMutablePointer<UInt32>.allocate(
            capacity: B * MAX_PAGES_PER_SLOT)
        // Pre-compute the structured-cot free-line mask once. ~5ms at
        // VOCAB=262144 — runs alongside model load, doesn't show up in
        // request hot path. Reused by every cot-enabled session.
        self.buildFreeLineMask()
        // Track D: wire eviction callback so PageManager's forced-eviction
        // path can unlink stale trie anchors.
        let trieRef = self.prefixTrie
        self.pageManager.onPageEvicted = { phys, _ in
            trieRef.invalidateAnchorFor(physPage: phys)
        }
    }

    // Grow a session's owned pages so its logical page count covers k_len.
    // Called right before admission to a slot and between steps when the
    // session's position advances past its current allocation.
    // Fails if the page pool is exhausted — caller must close a session or
    // evict another resident to make room.
    fileprivate func ensurePages(_ s: Session, forKLen kLen: Int) -> Bool {
        // Allocate at PAGE_FULL granularity (the SMALLER of slide/full
        // page sizes) so block_table has enough entries for the
        // full-attention layers, which use PAGE_FULL=8 while slide uses
        // PAGE_SLIDE=16.
        let needed = (kLen + PAGE_FULL - 1) / PAGE_FULL
        while s.ownedPages.count < needed {
            do {
                let p = try pageManager.allocFresh()
                zeroPhysPageKV(p)
                s.ownedPages.append(p)
            } catch {
                print("  ensurePages: pool exhausted for session \(s.id) at logical page \(s.ownedPages.count)")
                return false
            }
        }
        return true
    }

    // Wipe the K/V cache rows for a freshly-allocated phys page across all
    // layers. Without this, an attention READ of a row that was never
    // written by the new owner (e.g. a partial-block tail row above the
    // current k_len, masked to score=-INF) does `0 * V_stale`. IEEE says
    // `0 * NaN = NaN` — so a single stale-NaN row poisons attention output
    // forever. Zero-fill makes that path return `0 * 0 = 0` safely.
    // Cost: ~32 KB per page across 25 layers ≈ 800 KB memset, ~3 μs on
    // M5's ~250 GB/s memcpy bandwidth. Negligible vs the ~30 ms step.
    fileprivate func zeroPhysPageKV(_ phys: Int) {
        // Virtual-page-table-aware zero-fill. Map phys → (chunk_idx,
        // local_phys) using the same arithmetic the kernels do.
        let chunkIdx = phys / weights.kvChunkPages
        let localPhys = phys - chunkIdx * weights.kvChunkPages
        for L in 0..<NUM_LAYERS {
            let lw = weights.layers[L]
            let pg = lw.isFull ? PAGE_FULL : PAGE_SLIDE
            let bytesPerPage = pg * lw.KV_H * lw.HD * 2
            let off = localPhys * bytesPerPage
            let kBuf = weights.K_chunks[L][chunkIdx]
            let vBuf = weights.V_chunks[L][chunkIdx]
            memset(kBuf.contents().advanced(by: off), 0, bytesPerPage)
            memset(vBuf.contents().advanced(by: off), 0, bytesPerPage)
        }
    }

    // Followup 2 (LM_KV_OVERWRITE_CHECK assertion): compute a content
    // hash over the K/V row(s) at a single logical position across all
    // layers, walking the session's ownedPages map to find the right
    // phys page per layer (slide layers use ownedPages[2*P], full
    // layers may use ownedPages[2*P] or ownedPages[2*P+1] depending on
    // position % 16 < 8 vs ≥ 8).
    //
    // Used to verify the .primed recover-step's "benign overwrite"
    // property: the K/V write at position lands in pages shared with
    // the producer; per the Track A analysis the bytes produced are
    // bit-identical (same input + same KV[0..k-2] + same cvecs).
    //
    // Cost: walks NUM_LAYERS * (K_row + V_row) bytes ≈ a few KB total.
    // FNV-1a over fp16 halves. Off the hot path by default (env-var
    // gated); on, runs only for .primed sessions.
    //
    // NOTE: we deliberately hash ONLY the recover-position row(s), not
    // whole pages. The benign-overwrite analysis only claims byte
    // identity at the single row being written.
    fileprivate func hashSessionKVAtPosition(_ s: Session, position: Int) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        let logSlide = position / PAGE_SLIDE       // logical slide page index
        let logFull = position / PAGE_FULL          // logical full page index
        // Slide-layer phys: ownedPages[2 * logSlide]
        // Full-layer phys: ownedPages[logFull] (full layers index block_table
        // directly by logFull, which corresponds to ownedPages[logFull])
        for L in 0..<NUM_LAYERS {
            let lw = weights.layers[L]
            let pg = lw.isFull ? PAGE_FULL : PAGE_SLIDE
            let physOwnedIdx = lw.isFull ? logFull : (2 * logSlide)
            guard physOwnedIdx < s.ownedPages.count else { continue }
            let phys = s.ownedPages[physOwnedIdx]
            let chunkIdx = phys / weights.kvChunkPages
            let localPhys = phys - chunkIdx * weights.kvChunkPages
            let rowBytes = lw.KV_H * lw.HD * 2
            let rowOff = (localPhys * pg + (position % pg)) * rowBytes
            let kBase = weights.K_chunks[L][chunkIdx].contents().advanced(by: rowOff)
            let vBase = weights.V_chunks[L][chunkIdx].contents().advanced(by: rowOff)
            for buf in [kBase, vBase] {
                let p = buf.bindMemory(to: UInt8.self, capacity: rowBytes)
                for i in 0..<rowBytes {
                    h ^= UInt64(p[i])
                    h = h &* 0x100000001b3
                }
            }
        }
        return h
    }

    // Followup 1 (CoW-on-extend for partial pairs): copy K/V bytes from
    // a source phys page to a destination phys page, across all layers.
    // The whole per-layer page is copied (PAGE_SLIDE or PAGE_FULL rows
    // worth of K + V halves) because (a) the cost is dominated by the
    // ~tens-of-microseconds memcpy bandwidth, not the byte count, and
    // (b) trailing rows in the source past validUpTo are either zeros
    // (fresh-allocated source) or producer-written bytes that the
    // consumer's later prefill will overwrite anyway. Cheaper than
    // computing per-row offsets and slicing.
    //
    // CPU memcpy via Swift `memcpy`; the K/V buffers are CPU/GPU shared
    // (managed/host-visible) so the copy is visible to subsequent GPU
    // dispatches without an explicit synchronize.
    fileprivate func copyPhysPageKV(srcPhys: Int, dstPhys: Int) {
        let srcChunkIdx = srcPhys / weights.kvChunkPages
        let srcLocal = srcPhys - srcChunkIdx * weights.kvChunkPages
        let dstChunkIdx = dstPhys / weights.kvChunkPages
        let dstLocal = dstPhys - dstChunkIdx * weights.kvChunkPages
        for L in 0..<NUM_LAYERS {
            let lw = weights.layers[L]
            let pg = lw.isFull ? PAGE_FULL : PAGE_SLIDE
            let bytesPerPage = pg * lw.KV_H * lw.HD * 2
            let srcOff = srcLocal * bytesPerPage
            let dstOff = dstLocal * bytesPerPage
            let srcK = weights.K_chunks[L][srcChunkIdx]
            let srcV = weights.V_chunks[L][srcChunkIdx]
            let dstK = weights.K_chunks[L][dstChunkIdx]
            let dstV = weights.V_chunks[L][dstChunkIdx]
            memcpy(dstK.contents().advanced(by: dstOff),
                   srcK.contents().advanced(by: srcOff),
                   bytesPerPage)
            memcpy(dstV.contents().advanced(by: dstOff),
                   srcV.contents().advanced(by: srcOff),
                   bytesPerPage)
        }
    }

    // Idempotent: if this session has staged prefix K/V and hasn't yet
    // installed, allocate one page (if needed), write prefix K/V bytes
    // to position 0 per layer, bump s.position to 1. Call BEFORE any
    // code that captures s.position for prefill positioning.
    fileprivate func installPendingPrefixKVIfAny(_ s: Session) {
        guard !s.prefixKvInstalled, let pkv = s.prefixKvStaged,
              s.position == 0 else { return }
        if s.ownedPages.isEmpty {
            do {
                let p = try pageManager.allocFresh()
                zeroPhysPageKV(p)
                s.ownedPages.append(p)
            } catch {
                print("  prefix-kv install: alloc failed for session \(s.id)")
                return
            }
        }
        installPrefixKV(s, k: pkv.k, v: pkv.v)
        s.prefixKvInstalled = true
        s.position = 1
        print("[prefix-kv] installed for session \(s.id), page=\(s.ownedPages[0]), position bumped to 1")
    }

    // Write per-layer K/V bytes to position 0 of session s's FIRST
    // owned page. Per-layer K cache layout per Kbuf:
    //   [TOTAL_PAGES, PAGE_size_L, KV_H_L, HD_L] halves
    // Byte offset for phys-page P, position p within page, head h, dim d:
    //   ((P * PAGE_size_L + p) * KV_H_L + h) * HD_L + d   (halves)
    // For p=0, h covers KV_H_L heads: write KV_H_L * HD_L halves = that
    // many * 2 bytes.
    fileprivate func installPrefixKV(_ s: Session, k: [Data], v: [Data]) {
        let phys = s.ownedPages[0]
        // Virtual-page-table-aware K/V write. Same routing as kernels.
        let chunkIdx = phys / weights.kvChunkPages
        let localPhys = phys - chunkIdx * weights.kvChunkPages
        for L in 0..<NUM_LAYERS {
            let lw = weights.layers[L]
            let pg = lw.isFull ? PAGE_FULL : PAGE_SLIDE
            let sliceHalves = lw.KV_H * lw.HD
            let byteOffset = localPhys * pg * sliceHalves * 2
            // K and V destinations at that offset.
            guard L < k.count, L < v.count else { continue }
            let kBlob = k[L]; let vBlob = v[L]
            guard kBlob.count == sliceHalves * 2,
                  vBlob.count == sliceHalves * 2 else {
                print("  installPrefixKV: L\(L) size mismatch " +
                      "(got k=\(kBlob.count), v=\(vBlob.count), " +
                      "expected \(sliceHalves * 2))")
                continue
            }
            let kBuf = weights.K_chunks[L][chunkIdx]
            let vBuf = weights.V_chunks[L][chunkIdx]
            let kDst = kBuf.contents().advanced(by: byteOffset)
            let vDst = vBuf.contents().advanced(by: byteOffset)
            kBlob.withUnsafeBytes { memcpy(kDst, $0.baseAddress, sliceHalves * 2) }
            vBlob.withUnsafeBytes { memcpy(vDst, $0.baseAddress, sliceHalves * 2) }
        }
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
    // 2026-05-07: openSession went private. Sessions are no longer
    // creatable from outside the engine — the only public path is
    // `submitRequest(...)` below, which creates the Session, binds its
    // streamId, applies sampling/control state, and submits initial
    // segments all under the same lock acquisition. There is no
    // pre-first-submit window where a Session exists with no work.
    fileprivate func openSession(eosId: UInt32? = nil, maxNewTokens: Int = 4096) -> Session? {
        // 2026-05-07: default bumped 128 → 4096. Gemma-4 rates 128k context
        // and the engine handles 64k+ full-cache positions per slot. The
        // 128 default was from before long-context support landed.
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

    // ────────────────────────────────────────────────────────────────────
    // Atomic request creation (2026-05-07).
    //
    // RequestInit bundles per-request configuration that's set once at
    // birth — no setter dance from the bridge, no observable half-state.
    // Initial segments are submitted at construction; the Session is born
    // in .priming with primingQueue populated.
    //
    // The body of submitRequest does what applyStreamAction's old "start"
    // case used to do, but inside the engine: open the session, bind the
    // streamId, write all the per-request fields, queue every initial
    // segment, and return the live Session. The bridge holds nothing
    // about the request afterward — `engine.requestForStream[sid]` is
    // the only handle.
    //
    // Image segments need vision-tower work the bridge knows about (the
    // softs cache lives in ffi_batch.swift). For those, the caller passes
    // a closure `submitImageSegment` that produces the soft-token blit
    // for a given image-bytes blob and submits it onto the Session. This
    // keeps the vision orchestration in the bridge while the Session
    // lifecycle stays atomic from the engine's perspective.
    // ────────────────────────────────────────────────────────────────────
    struct RequestInit {
        var eosId: UInt32? = nil
        var maxNewTokens: Int = 4096   // bumped 128 → 4096 (2026-05-07): see openSession default note
        var samplingTemperature: Float = 0.0
        var samplingSeed: UInt64 = 0
        var captureLogits: Bool = false
        var topLogprobs: UInt32 = 0
        var logitBiasDense: [Float]? = nil
        var minP: Float = 0.0
        var stopSequences: [[UInt32]] = []
        var cot: CotState? = nil
        var controls: [ActiveControl] = []
        var enableThinking: Bool = false
        var captureCvecActivations: Bool = false
    }

    enum InitialSegment {
        case tokens([UInt32])
        case image(Data)   // raw image bytes; bridge resolves via vision tower
    }

    @discardableResult
    func submitRequest(streamId sid: UInt64,
                       init initParams: RequestInit,
                       segments: [InitialSegment],
                       imageSubmit: ((Session, Data) -> Void)? = nil) -> Session? {
        // Refuse re-binding a live stream — bridge mistake or leaked sid.
        if requestForStream[sid] != nil {
            print("  submitRequest: stream_id \(sid) already live; ignored")
            return nil
        }
        // Admission backpressure: refuse new submissions when the page
        // pool is below the absolute floor needed for one prefill cycle.
        // The 2026-05-13 page-pool exhaustion under /sweep load motivated
        // this: 26 concurrent submissions with no admission cap let each
        // session grow into the shared pool until ensurePages began
        // failing mid-prefill, wedging the engine. Refusing admission at
        // submit time means the FAILURE surfaces immediately as a clean
        // "try again" signal rather than a half-prefilled session stuck
        // waiting for pages that won't come.
        //
        // Floor = ADMISSION_FREE_PAGE_FLOOR (set at boot from bootstrap
        // constants). Rationale: one prefill cycle for a typical user-
        // facing prompt (~1K-4K tokens) needs 64-256 pages. We refuse
        // admission when fewer pages are available than the floor, so
        // the new request can at least PROBABLY make progress without
        // immediately starving its peers.
        let poolStats = pageManager.stats()
        if poolStats.freePages < ADMISSION_FREE_PAGE_FLOOR {
            print("  submitRequest: admission backpressure (free=\(poolStats.freePages) < floor=\(ADMISSION_FREE_PAGE_FLOOR), live=\(poolStats.pagesInUse))")
            return nil
        }
        guard let s = openSession(eosId: initParams.eosId,
                                  maxNewTokens: initParams.maxNewTokens) else {
            return nil
        }
        bindStream(s, streamId: sid)
        s.samplingTemperature = initParams.samplingTemperature
        s.applySamplingSeed(initParams.samplingSeed)
        s.captureLogits = initParams.captureLogits
        s.topLogprobs = initParams.topLogprobs
        s.enableThinking = initParams.enableThinking
        s.captureCvecActivations = initParams.captureCvecActivations
        s.logitBiasDense = initParams.logitBiasDense
        s.minP = initParams.minP
        s.stopSequences = initParams.stopSequences
        s.cot = initParams.cot
        for c in initParams.controls { s.addControl(c) }
        // Queue every segment in declaration order. Submission has to
        // happen here (not at the call site) so the Session is in
        // .priming with chunks before submitRequest returns — no
        // observable empty-queue window.
        for seg in segments {
            switch seg {
            case .tokens(let toks):
                s.submit(toks)
                s.usage.promptTokensSeen &+= UInt32(toks.count)
            case .image(let bytes):
                imageSubmit?(s, bytes)
            }
        }
        return s
    }

    // Close: release slot (if any), return pages, drop from residency.
    func closeSession(_ s: Session) {
        s.state = .done
        // 2026-05-07 (M:K): no persistent slotAssignment to clear; slot
        // is per-CB scratch. Just nil out the per-CB field for cleanliness.
        s.slot = nil
        // Track B fold-in: partial-page promotion at teardown. We promote
        // a partial-pair anchor at the PROMPT boundary (not s.position,
        // which has been extended by generation — generation tokens
        // diverge across sessions and aren't useful to share).
        let promptEnd = s.promptEndPosition
        // Partial covers everything EXCEPT the last token of the prompt
        // (which gets re-prefilled by the consumer to drive the post-
        // prefill sample naturally; see B1/B3 test expectations).
        let mod = promptEnd % PAGE_SLIDE
        if mod >= 2 && promptEnd > 0 {
            let partialValid = mod - 1
            let p = promptEnd / PAGE_SLIDE      // index of trailing partial page
            let slideIdx = 2 * p
            let fullIdx = 2 * p + 1
            // Only promote if we have the trailing page pair allocated.
            if slideIdx < s.ownedPages.count &&
               promptEnd <= s.consumedTokens.count {
                let slidePrim = s.ownedPages[slideIdx]
                let fullSib = (fullIdx < s.ownedPages.count) ? s.ownedPages[fullIdx] : nil
                let pageStart = p * PAGE_SLIDE
                let tag = s.cvecAnchorTagForPage(pageStart: pageStart,
                                                  pageSize: PAGE_SLIDE)
                // The trie key is the first (pageStart + partialValid) tokens,
                // which is the prefix that B's lookup will match against
                // (B submits its own prefix; trie finds anchor at that depth).
                let trieDepth = pageStart + partialValid
                // Register pair with PageManager so contentIndex prevents
                // premature eviction. Synthetic hash so identical-prefix
                // partials dedupe across sessions.
                let synthHash = PageManager.hashPage(
                    s.consumedTokens[0..<trieDepth],
                    cvecDigest: s.cvecDigestForPage(pageStart: pageStart,
                                                     pageSize: partialValid))
                if let fs = fullSib {
                    pageManager.markPairContentIndexed(slidePlusFullHead: slidePrim,
                                                       fullTail: fs,
                                                       contentHash: synthHash)
                }
                let partial = PartialPagePair(
                    slidePlusFullHead: slidePrim,
                    fullTail: (partialValid > PAGE_FULL) ? fullSib : nil,
                    validUpTo: partialValid)
                prefixTrie.insertPartialAnchor(
                    tokensCoveringFullPrefix: s.consumedTokens[0..<trieDepth],
                    pair: partial, cvecTag: tag)
            }
        }
        // 2026-05-07 (anonymous-pool refactor): caller (this session) is
        // the canonical holder of its phys-page references; decref each
        // one. PageManager has no per-session ledger to release from.
        for phys in s.ownedPages { pageManager.decref(physPage: phys) }
        s.ownedPages.removeAll()
        s.consumedTokens.removeAll()
        s.myPromotedSlidePairCount = 0
        residentSessions.removeValue(forKey: s.id)
        unbindStream(s)
    }

    // After a prefill tile commits, walk any fully-written logical pages
    // that haven't been promoted yet and publish them to the radix
    // trie so the next session with the same prefix can adopt via
    // RadixTrie.findLongestPrefix. Called at the end of
    // stepPrefillForSession. (Pre-Followup 3 this also wrote into the
    // deleted PageManager.contentIndex dict.)
    fileprivate func promoteFinishedPages(_ s: Session) {
        let fullyWritten = s.position / PAGE_SLIDE
        while s.myPromotedSlidePairCount < fullyWritten {
            let p = s.myPromotedSlidePairCount
            // We need the first (p+1)*PAGE_SLIDE tokens of this session's
            // submitted history to form the page's prefix hash. If the
            // session's consumedTokens hasn't caught up (unusual — happens
            // only when a submit staged tokens in primingQueue that were
            // consumed without being added to consumedTokens), skip.
            let end = (p + 1) * PAGE_SLIDE
            // Each slide page P occupies TWO phys pages in ownedPages:
            // ownedPages[2P] is the slide primary (holds slide K/V for
            // positions [P*16, P*16+15] + full K/V for [P*16, P*16+7]),
            // and ownedPages[2P+1] is the full sibling (holds full K/V
            // for [P*16+8, P*16+15]).  Skip promotion until both are
            // allocated.
            let slideIdx = 2 * p
            let fullIdx  = 2 * p + 1
            guard end <= s.consumedTokens.count,
                  fullIdx < s.ownedPages.count else {
                break
            }
            let pageStart = p * PAGE_SLIDE
            let digest = s.cvecDigestForPage(pageStart: pageStart,
                                              pageSize: PAGE_SLIDE)
            let hash = PageManager.hashPage(s.consumedTokens[0..<end],
                                             cvecDigest: digest)
            let pair = SlidePairContents(slidePlusFullHead: s.ownedPages[slideIdx],
                                       fullTail: s.ownedPages[fullIdx])
            // Followup 3 (2026-05-23): contentIndex deleted; only the
            // PageInfo bookkeeping (so the eviction callback knows the
            // pair partner) survives. The radix trie below is the
            // source of truth for adoption lookups.
            pageManager.markPairContentIndexed(slidePlusFullHead: pair.slidePlusFullHead,
                                                fullTail: pair.fullTail,
                                                contentHash: hash)
            // Track D: insert into the radix trie. The trie is the
            // source-of-truth for adoption lookups.
            let tag = s.cvecAnchorTagForPage(pageStart: pageStart,
                                              pageSize: PAGE_SLIDE)
            prefixTrie.insertAnchor(
                tokensCoveringFullPrefix: s.consumedTokens[0..<end],
                pair: pair, cvecTag: tag)
            if ProcessInfo.processInfo.environment["LM_CACHE_DEBUG"] != nil {
                let head = s.consumedTokens[pageStart..<end].prefix(4).map(String.init).joined(separator: ",")
                print("  [cache] session \(s.id) PROMOTE page \(p) hash=\(String(hash, radix: 16, uppercase: false)) head=[\(head),…]")
            }
            s.myPromotedSlidePairCount += 1
        }
    }

    func poolStats() -> PageManager.Stats { return pageManager.stats() }

    // 2026-05-07 (M:K): "active" sessions are now all residents in a
    // busy state (.priming or .generating). Length can be > B; the
    // per-CB picker selects up to B of them for kernel positions.
    var activeSessions: [Session] {
        return residentSessions.values.filter { $0.state.isBusy }
    }
    // Resident sessions that want a slot (priming or generating).
    var readyResidents: [Session] {
        return residentSessions.values.filter { $0.state.wantsSlot }
    }
    var hasWork: Bool { readyResidents.count > 0 }

    // Admission pass: evict slots whose session no longer wants one, admit
    // ready-but-unslotted residents into free slots round-robin. Grows
    // owned pages + installs block_table entries for freshly-admitted sessions.
    // Public wrapper so the bridge can run admission before populating
    // per-slot state (e.g. gpu_capture_active for logprob extraction).
    // 2026-05-07: needed because the bridge's gCaptureLogits → gpu_capture_active
    // populate happens BEFORE syncTickStep, and freshly-admitted sessions
    // don't have s.slot assigned until tick()'s internal admission pass.
    func runAdmissionPassPublic() { runAdmissionPass() }

    private func runAdmissionPass() {
        // 2026-05-07 (M:K permutation): admission is now a no-op.
        // Sessions are added to residentSessions by openSession();
        // page allocation happens lazily in prepare* paths (per CB)
        // via ensurePages with the exact kLen the CB needs. Pre-
        // reserving a 1024-token floor at admission worked when M=B=8
        // (8 × 128 pages = 1024 pages, headroom in 8192-page pool),
        // but at M=64 it reserves 8192 pages = the entire pool, and
        // each session is over-allocated 4× what it actually uses.
        //
        // Per-CB ensurePages provides correct block_table coverage
        // for the kernel positions chosen this CB; non-chosen
        // sessions don't need any pages until their next CB.
    }

    // 2026-05-07: per-CB kernel-batch position picker.
    //
    // Picks up to B of the M=residentSessions.count sessions to occupy
    // kernel-batch positions [0..B). Returns a [Session?] of length B
    // indexed by kpos; nil entries are silenced (the kernel still runs
    // those positions but with scratch-strip block_table + sampling_active=0).
    //
    // Selection: round-robin by session id. The cursor advances by 1 per
    // CB, so over many CBs each session gets equal kernel-batch share.
    // For workloads with M >> B (e.g., M=64 sessions, B=8), each session
    // gets ~B/M ≈ 12.5% of the kernel-batch share. Per-session decode
    // rate drops; aggregate throughput stays at the kernel ceiling.
    //
    // Side effects: sets `Session.slot = kpos` on each chosen session
    // for use by existing prepare/finalize code that indexes kernel
    // buffers via s.slot. Non-chosen sessions get s.slot = nil.
    //
    // `state` filter selects which sessions are eligible: .generating
    // for AR, .priming for prefill paths.
    private func pickKernelMappingForState(
        _ acceptable: (Session) -> Bool
    ) -> [Session?] {
        // Clear all residents' slots — only chosen ones get a kpos.
        for s in residentSessions.values { s.slot = nil }
        let eligible = residentSessions.values
            .filter(acceptable)
            .sorted { $0.id < $1.id }
        var mapping: [Session?] = Array(repeating: nil, count: B)
        if eligible.isEmpty { return mapping }
        let pickStart = pickerCursor % eligible.count
        for i in 0..<min(B, eligible.count) {
            let s = eligible[(pickStart + i) % eligible.count]
            s.slot = i
            mapping[i] = s
        }
        pickerCursor &+= 1
        return mapping
    }

    private var pickerCursor: Int = 0

    private func pickKernelMappingForAR() -> [Session?] {
        // AR can serve both .generating sessions (real decode) AND
        // .priming sessions with a 1-token tail (popArPrimingToken
        // consumes them one token at a time). The pickChainPath
        // already routed multi-token-priming sessions to a prefill
        // path; whatever lands here is decoder-eligible.
        return pickKernelMappingForState { $0.state.isBusy }
    }

    // For prefill paths, the caller supplies a list of sessions in priming
    // state (already filtered by chunk type — text vs soft). The picker
    // assigns up to B of them to kernel positions and silences the rest.
    private func pickKernelMappingForPrefill(_ sessions: [Session]) -> [Session?] {
        for s in residentSessions.values { s.slot = nil }
        let eligible = sessions.filter { residentSessions[$0.id] != nil }
            .sorted { $0.id < $1.id }
        var mapping: [Session?] = Array(repeating: nil, count: B)
        if eligible.isEmpty { return mapping }
        let pickStart = pickerCursor % eligible.count
        for i in 0..<min(B, eligible.count) {
            let s = eligible[(pickStart + i) % eligible.count]
            s.slot = i
            mapping[i] = s
        }
        pickerCursor &+= 1
        return mapping
    }

    // Take one token off the head chunk if it's a .tokens chunk. Returns
    // nil if the head is .softTokens (that chunk needs fast prefill, not
    // AR priming) or the queue is empty. Also removes emptied chunks and
    // updates session state.
    private func popArPrimingToken(_ s: Session) -> UInt32? {
        while let head = s.primingQueue.first {
            switch head {
            case .tokens(var ts):
                if ts.isEmpty {
                    s.primingQueue.removeFirst(); continue
                }
                let t = ts.removeFirst()
                if ts.isEmpty { s.primingQueue.removeFirst() }
                else { s.primingQueue[0] = .tokens(ts) }
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
        guard let head = s.primingQueue.first else { return false }
        switch head {
        case .tokens(let ts): return ts.count >= minTokensThreshold
        case .softTokens:    return true
        }
    }

    // Populate the 6 sampling buffers (temperature, min_p, seed, step,
    // active, dense logit_bias) for the GPU `sample_token` kernel. Must
    // be called BEFORE committing any CB that ends with encSampleToken
    // (buildStepCB, buildPrefillCB via its unembed path).
    //
    // `activeSlotSession[slot]` is the session whose sampled token will
    // be read from gpu_sampled_tokens[slot] post-wait. Non-nil entries
    // get their params + bias populated; nil entries are zeroed and
    // marked inactive so the kernel short-circuits that slot.
    // Track whether each slot's bias row currently holds non-zero data.
    // Without this every CB does a 1MB memset per slot for the common
    // case (no logit bias) — pure waste on the critical path. The buffer
    // is initialized to zero at allocation; we only write when going
    // from zero→non-zero or non-zero→zero, and skip otherwise.
    private var slotBiasIsDirty = [Bool](repeating: false, count: B)

    // Pre-computed mask of the per-vocab "free-line" set: 0 for tokens
    // whose bytes don't contain a mid-token newline (and aren't framing
    // specials), -INF for tokens that would put a newline mid-stream
    // or contribute zero bytes. Computed once in init() — used for
    // every structured-cot session in the .freeLine phase. Tokens
    // ending in \n (terminal-newline) are valid; they advance the
    // grammar to the next literal phase.
    private var freeLineMask: [Float] = []

    // 2026-05-07: removed cotLiteralMaskCache. The field was declared
    // but never populated or read — pure dead code. The cotMask
    // function (now writing directly to the GPU buffer) recomputes
    // each call; if literal-phase mask compute ever shows up in
    // profiles, a real cache should be added with eviction tied to
    // CoT-state-mutation events (enable/advance/disable) rather than
    // session id (which doesn't capture phase advancement).

    // Write the mask for a CotState's CURRENT phase + cursor directly
    // to `dst` (a [VOCAB] Float pointer in the GPU sampling-bias buffer).
    // 0 = allowed, -INF = forbidden. Caller additively combines with
    // any user logit_bias (the sampling_logit_bias buffer is read by
    // the sample_token kernel as a pure additive bias — -INF saturates
    // regardless of user bias).
    //
    // 2026-05-07: rewrote to write directly into the GPU buffer instead
    // of allocating a 1 MB Swift [Float] per call. The literal-phase
    // path was the dominant thrasher: 262,144 floats × 4 bytes = 1 MB
    // heap alloc per CoT-active slot per AR tick. At B=8 with 1 CoT
    // slot, 12 AR ticks/sec, that's 12 MB/sec; at all 8 slots active,
    // 96 MB/sec.
    private func cotMask(state: CotState, dst: UnsafeMutablePointer<Float>) {
        let neg: Float = -.infinity
        guard let phase = state.phases.first else {
            // No active phase — clear to zero (allow all).
            memset(dst, 0, VOCAB * MemoryLayout<Float>.stride)
            return
        }
        switch phase {
        case .literal(let bytes):
            // Fill with -INF, then mark allowed tokens with 0.
            // memset_pattern4 would be ideal but Foundation's Darwin
            // memset_pattern4 is the way; fall back to a tight loop
            // if it's unavailable.
            var negPattern: Float = neg
            withUnsafePointer(to: &negPattern) { p in
                memset_pattern4(dst, p, VOCAB * MemoryLayout<Float>.stride)
            }
            // Iterate over the suffix of literal bytes we still owe.
            // No Array allocation: bytes[state.literalCursor...] is a
            // SubSequence we use directly via `dropFirst` semantics.
            let cursor = state.literalCursor
            let bytesEnd = bytes.count
            for v in 0..<VOCAB {
                let tBytes = tokenizer.tokenBytes(UInt32(v))
                if tBytes.isEmpty { continue }
                let need = tBytes.count
                if cursor + need > bytesEnd { continue }
                // Manual prefix-equality without slicing/Array.
                var ok = true
                for i in 0..<need {
                    if bytes[cursor + i] != tBytes[i] { ok = false; break }
                }
                if ok { dst[v] = 0 }
            }
        case .freeLine:
            // freeLineMask is a one-time-built [Float]; copy via
            // memcpy with the underlying buffer pointer.
            freeLineMask.withUnsafeBufferPointer { src in
                memcpy(dst, src.baseAddress, VOCAB * MemoryLayout<Float>.stride)
            }
        }
    }

    // Build freeLineMask once. Tokens with mid-token newline or empty
    // bytes (specials) are masked out; everything else is allowed.
    private func buildFreeLineMask() {
        let neg: Float = -.infinity
        var m = [Float](repeating: 0, count: VOCAB)
        for v in 0..<VOCAB {
            let tBytes = tokenizer.tokenBytes(UInt32(v))
            if tBytes.isEmpty { m[v] = neg; continue }
            // Find any \n NOT at the last position.
            for i in 0..<(tBytes.count - 1) {
                if tBytes[i] == 0x0A { m[v] = neg; break }
            }
        }
        freeLineMask = m
    }

    private func populateSamplingParams(_ activeSlotSession: [Session?]) {
        let sTempP   = sampling_temperature.contents().bindMemory(to: Float.self,  capacity: B)
        let sMinPP   = sampling_min_p.contents().bindMemory(to: Float.self,  capacity: B)
        let sSeedP   = sampling_seed.contents().bindMemory(to: UInt32.self, capacity: B)
        let sStepP   = sampling_step.contents().bindMemory(to: UInt32.self, capacity: B)
        let sActiveP = sampling_active.contents().bindMemory(to: UInt32.self, capacity: B)
        let sBiasP   = sampling_logit_bias.contents().bindMemory(to: Float.self, capacity: B * VOCAB)
        for slot in 0..<B {
            if let s = activeSlotSession[slot] {
                sTempP[slot]   = s.samplingTemperature
                sMinPP[slot]   = s.minP
                sSeedP[slot]   = s.gpuRngSeed
                sStepP[slot]   = UInt32(s.numGenerated)
                sActiveP[slot] = 1
                let biasRow = sBiasP.advanced(by: slot * VOCAB)
                // Composition order:
                //   1. If structured-cot is active, fill bias with the
                //      grammar's mask (-INF for forbidden tokens, 0 for
                //      allowed). This dominates user logit_bias for any
                //      forbidden token (-INF + anything = -INF).
                //   2. Else if user has dense logit_bias, memcpy that.
                //   3. Else clear bias if dirty.
                if let cot = s.cot {
                    // Direct-to-GPU-buffer write (no [Float] allocation).
                    cotMask(state: cot, dst: biasRow)
                    // If user ALSO has dense bias, add it on top
                    // (componentwise) so allowed tokens carry the user's
                    // preferences while forbidden ones stay -INF.
                    if let bias = s.logitBiasDense {
                        bias.withUnsafeBufferPointer { src in
                            for i in 0..<VOCAB { biasRow[i] += src[i] }
                        }
                    }
                    slotBiasIsDirty[slot] = true
                } else if let bias = s.logitBiasDense {
                    bias.withUnsafeBufferPointer { src in
                        memcpy(biasRow, src.baseAddress, VOCAB * 4)
                    }
                    slotBiasIsDirty[slot] = true
                } else if slotBiasIsDirty[slot] {
                    memset(biasRow, 0, VOCAB * 4)
                    slotBiasIsDirty[slot] = false
                }
                // else: bias row is already zero from allocation or a
                // prior cleanup, kernel reads zeros, no write needed.
            } else {
                sTempP[slot]   = 0.0
                sMinPP[slot]   = 0.0
                sSeedP[slot]   = 0
                sStepP[slot]   = 0
                sActiveP[slot] = 0
                if slotBiasIsDirty[slot] {
                    memset(sBiasP.advanced(by: slot * VOCAB), 0, VOCAB * 4)
                    slotBiasIsDirty[slot] = false
                }
            }
        }
    }

    // Snapshot captured at AR-step prep time and consumed by finalize. The
    // chain (chainAdvance) commits the CB and returns; finalize runs from
    // the completion handler, by which time `slotAssignment` may have
    // shifted under admission/close — so we snapshot the slot→session map.
    struct PreparedAR {
        let cb: MTLCommandBuffer
        let t0: Date
        let realSlot: [Bool]
        let slotSession: [Session?]
        // Followup 2 (LM_KV_OVERWRITE_CHECK): pre-step K/V byte hash for
        // each .primed slot's trailing-page phys (slidePlusFullHead). Only
        // populated when env-var is set; finalize compares post-step.
        let primedRecoverPagePreHash: [Int: (phys: Int, hash: UInt64, position: Int)]
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
        guard let p = prepareARStep() else { return 0 }
        p.cb.commit(); p.cb.waitUntilCompleted()
        return finalizeARStep(p)
    }

    // Build the AR CB and snapshot finalize-time state. Returns nil if no
    // session has busy work. Does not commit; caller commits and (sync or
    // via completion handler) calls finalizeARStep.
    func prepareARStep() -> PreparedAR? {
        runAdmissionPass()
        let busy = activeSessions.filter { $0.state.isBusy }
        if busy.isEmpty { return nil }

        // Per-slot inputs (AR path).
        let tokP = input_tokens.contents().bindMemory(to: UInt32.self, capacity: B)
        let posP = positions.contents().bindMemory(to: UInt32.self, capacity: B)
        let klsP = k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let klfP = k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
        let npsP = num_pages_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let npfP = num_pages_full.contents().bindMemory(to: UInt32.self, capacity: B)

        // 2026-05-07 (M:K permutation): pick up to B sessions in
        // .generating state from residentSessions to occupy the kernel
        // batch positions. The picker assigns Session.slot = kpos for
        // each chosen session; the existing per-kpos populate code
        // below uses s.slot as the kernel index. Non-chosen kernel
        // positions are silenced (block_table[kpos][0] → scratch,
        // sampling_active=0 implicit via realSlot=false).
        let arMapping = pickKernelMappingForAR()

        // Track which slots run REAL work this step.
        var realSlot = [Bool](repeating: false, count: B)
        // Followup 2 (LM_KV_OVERWRITE_CHECK): record pre-step K/V byte
        // hash for each .primed recover-step slot. finalize compares
        // post-step to catch any silent corruption of the producer's
        // shared pages. Empty when env-var is not set.
        let kvCheckEnabled = ProcessInfo.processInfo.environment["LM_KV_OVERWRITE_CHECK"] != nil
        var primedRecoverPagePreHash: [Int: (phys: Int, hash: UInt64, position: Int)] = [:]

        for slot in 0..<B {
            if let s = arMapping[slot], s.state.isBusy {
                // Track A fold-in: .primed sessions get a recover tick.
                // Input is the LAST consumed prompt token; position is
                // the LAST cached position.
                //
                // 2026-05-23 (kernel-pair fix): .primed sessions are now
                // routed to prepareRecoverPrefill via pickChainPath's
                // .recoverPrefill path, which dispatches the prefill
                // kernel (same as producer) for bit-identical K/V. If
                // a .primed session reaches HERE, the routing has
                // regressed — log it loudly. The AR-kernel branch below
                // is kept for defense-in-depth but should never fire.
                if s.state == .primed {
                    fputs("[recover-step] WARNING: .primed session sid=\(s.id) "
                          + "reached prepareARStep — pickChainPath routing "
                          + "regression. Falling back to AR-kernel recover "
                          + "(will re-trigger KV-overwrite drift).\n", stderr)
                    let recoverPos = s.position - 1
                    guard recoverPos >= 0,
                          recoverPos < s.consumedTokens.count else {
                        // Defensive: invalid recover state; park.
                        tokP[slot] = weights.bosTokenId
                        posP[slot] = 0
                        klsP[slot] = 1; klfP[slot] = 1
                        npsP[slot] = 1; npfP[slot] = 1
                        continue
                    }
                    // ensurePages for K/V at recoverPos (it should already
                    // exist via adoption, but defensive).
                    _ = ensurePages(s, forKLen: recoverPos + 2)
                    installBlockTable(s, slot: slot)
                    let tok = s.consumedTokens[recoverPos]
                    tokP[slot] = tok
                    posP[slot] = UInt32(recoverPos)
                    let kLen = recoverPos + 1
                    klsP[slot] = UInt32(kLen); klfP[slot] = UInt32(kLen)
                    npsP[slot] = UInt32((kLen + PAGE_SLIDE - 1) / PAGE_SLIDE)
                    npfP[slot] = UInt32((kLen + PAGE_FULL  - 1) / PAGE_FULL)
                    realSlot[slot] = true
                    // KV-overwrite check: hash the K/V row at the
                    // recover position (across all layers) pre-step.
                    // finalize verifies it's unchanged post-step.
                    if kvCheckEnabled {
                        let h = hashSessionKVAtPosition(s, position: recoverPos)
                        // We don't track a single phys page anymore;
                        // hashSessionKVAtPosition walks ownedPages.
                        // Store the session ref via slotSession in the
                        // PreparedAR; here we just store the hash + pos.
                        primedRecoverPagePreHash[slot] = (phys: -1, hash: h, position: recoverPos)
                    }
                    continue
                }
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
            // Park: BOS at position 0, k_len=1, 1 page. Non-chosen kernel
            // positions get scratch-strip block_table to keep KV writes
            // from this park step from corrupting other sessions' pages.
            tokP[slot] = weights.bosTokenId
            posP[slot] = 0
            klsP[slot] = 1; klfP[slot] = 1
            npsP[slot] = 1; npfP[slot] = 1
            if arMapping[slot] == nil {
                let btP = block_table.contents().bindMemory(to: UInt32.self,
                            capacity: B * MAX_PAGES_PER_SLOT)
                btP[slot * MAX_PAGES_PER_SLOT + 0] = UInt32(SCRATCH_PAGE_BASE)
            }
        }

        precomputeFlexBlockMaskSlide(slidingWindow: SLIDING_WINDOW)
        precomputeFlexBlockMaskFull()

        // Sampling params for the GPU sample_token dispatch that
        // encodes at the end of buildStepCB. See populateSamplingParams.
        // 2026-05-07 (M:K): use arMapping (from picker) instead of the
        // deleted slotAssignment.
        var arSlotSession: [Session?] = Array(repeating: nil, count: B)
        for slot in 0..<B where realSlot[slot] {
            arSlotSession[slot] = arMapping[slot]
        }
        populateSamplingParams(arSlotSession)

        // Control-vector per-tick staging. For every occupied slot, evaluate
        // each of its active controls' envelopes at the current (position,
        // turn) and stash the resulting (buffer, layer, mag) triples in
        // gSlotControls[slot]. Silenced slots just get an empty list.
        // Per-slot active-control staging. Each control's envelope-
        // evaluated scalar becomes either the ADD magnitude (additive
        // mode) or the TARGET projection (project mode); the dispatch
        // picks the appropriate kernel based on mode. Project-mode
        // controls claim a slot in gProjectMeasureBuf so the kernel
        // can write back pre-write projections for telemetry.
        for slot in 0..<B { gSlotControls[slot].removeAll(keepingCapacity: true) }
        for slot in 0..<B where realSlot[slot] {
            guard let s = arMapping[slot] else { continue }
            let measBase = slot * MAX_PROJECT_CONTROLS_PER_SLOT
            var measIdx = 0
            for c in s.activeControls {
                let m = c.magnitudeAt(position: s.position, turn: s.turnIndex)
                s.recordCvecActivation(
                    tokenPosition: s.position,
                    layer: c.layer,
                    magnitude: m)
                switch c.mode {
                case .additive:
                    if m != 0 {
                        gSlotControls[slot].append(SlotControl(
                            buffer: c.buffer, layer: c.layer, mag: m,
                            mode: .additive, measureOutSlot: 0,
                            transportScale: 0, transportOffset: 0))
                    }
                case .project:
                    // Two semantics:
                    //  - c.target == nil (back-compat): envelope m IS the
                    //    target projection. Always stage.
                    //  - c.target != nil (gated): envelope m is a GATE;
                    //    m == 0 → skip dispatch entirely, else coerce to
                    //    c.target. This is what enables scoped obliteratus:
                    //    detector restarts the envelope on refusal-rise,
                    //    ablation only fires while the gate is hot.
                    let stage: Bool
                    let targetValue: Float
                    if let t = c.target {
                        stage = (m != 0)
                        targetValue = t
                    } else {
                        stage = true
                        targetValue = m
                    }
                    if stage && measIdx < MAX_PROJECT_CONTROLS_PER_SLOT {
                        gSlotControls[slot].append(SlotControl(
                            buffer: c.buffer, layer: c.layer, mag: targetValue,
                            mode: .project, measureOutSlot: measBase + measIdx,
                            transportScale: 0, transportOffset: 0))
                        measIdx += 1
                    }
                case .transport:
                    // Gaussian OT: kernel computes target = scale*a + offset
                    // using the same per-dispatch reduction that project
                    // uses. Envelope m acts as a GATE identical to the
                    // gated-project semantics (m == 0 → skip). c.target
                    // isn't used; scale/offset are the Brenier-map
                    // coefficients precomputed at attach.
                    let stage = (m != 0)
                    if stage && measIdx < MAX_PROJECT_CONTROLS_PER_SLOT {
                        gSlotControls[slot].append(SlotControl(
                            buffer: c.buffer, layer: c.layer, mag: 0,
                            mode: .transport, measureOutSlot: measBase + measIdx,
                            transportScale: c.transportScale,
                            transportOffset: c.transportOffset))
                        measIdx += 1
                    }
                }
            }
        }
        // Phase C-Read: stage detectors, assigning each a linear offset
        // into gIntensityBuf so the CB kernel knows where to write its
        // scalar output. Max MAX_DETECTORS_PER_SLOT detectors per slot.
        for slot in 0..<B { gSlotDetectors[slot].removeAll(keepingCapacity: true) }
        for slot in 0..<B where realSlot[slot] {
            guard let s = arMapping[slot] else { continue }
            let base = slot * MAX_DETECTORS_PER_SLOT
            for (i, d) in s.detectors.prefix(MAX_DETECTORS_PER_SLOT).enumerated() {
                gSlotDetectors[slot].append(SlotDetector(
                    buffer: d.buffer, layer: d.layer, outSlot: base + i))
            }
        }

        // Compute activeB for the kernel-zoo dispatchers: highest-occupied
        // slot index + 1. Slot policy "lowest free first" (runAdmissionPass)
        // packs active sessions at [0, activeB), so the b1/b2/b4/b8 PSO
        // selection runs the right slots' work and silences nothing past aB.
        var activeB = 0
        for slot in 0..<B where realSlot[slot] {
            activeB = slot + 1
        }
        if activeB == 0 { activeB = 1 }      // park step still needs slot 0 dispatched

        let t0 = Date()
        let cb = buildStepCB(weights, activeB: activeB)
        return PreparedAR(cb: cb, t0: t0, realSlot: realSlot, slotSession: arSlotSession,
                          primedRecoverPagePreHash: primedRecoverPagePreHash)
    }

    // Post-CB readback. Runs once the GPU has finished the AR step CB —
    // either inline after waitYielded (sync `step()`) or from the chain's
    // addCompletedHandler (async `chainAdvance`). Uses the captured
    // slot→session snapshot, not slotAssignment, so admission churn
    // during GPU compute can't reroute the readback.
    @discardableResult
    func finalizeARStep(_ p: PreparedAR) -> Int {
        lastStepMs = Date().timeIntervalSince(p.t0) * 1000
        totalSteps += 1
        if let err = p.cb.error { print("  GPU step error: \(err)"); return 0 }
        // Followup 2 (LM_KV_OVERWRITE_CHECK): verify the .primed
        // recover-step's K/V write at position-1 produced bit-identical
        // bytes (the "benign overwrite" property from Track A's design).
        // On mismatch, abort the affected session and print a loud
        // warning — this is a correctness failure that would silently
        // corrupt downstream sampling.
        if !p.primedRecoverPagePreHash.isEmpty {
            // Followup 2: warn-only mode. The Track A design predicted
            // "benign overwrite" — i.e., bit-identical K/V bytes at the
            // recover position — based on (same input + same KV[0..k-2]
            // + same cvecs) → same K/V output. In practice prefill and
            // AR kernels use different reduction orders (q_len=16 vs
            // q_len=1) and produce K/V bytes that differ at the fp16
            // ULP level, even though the downstream sampling and KL-
            // divergence guards are satisfied by these tiny differences.
            //
            // We keep the assertion as a debug/diagnostic signal: if
            // the differences ever grow large enough to corrupt
            // downstream behavior, the warning will be the first
            // forensic breadcrumb. But we do NOT abort the session,
            // because the differences are known-benign for the test
            // harness's KL guard and the production hot path.
            //
            // Set LM_KV_OVERWRITE_CHECK=1 to enable; LM_KV_OVERWRITE_ABORT=1
            // additionally upgrades the warning to session abort.
            let abortOnMismatch = ProcessInfo.processInfo.environment["LM_KV_OVERWRITE_ABORT"] != nil
            for (slot, pre) in p.primedRecoverPagePreHash {
                guard let s = p.slotSession[slot] else { continue }
                let postH = hashSessionKVAtPosition(s, position: pre.position)
                if postH != pre.hash {
                    let sid = s.id
                    let preHex = String(pre.hash, radix: 16, uppercase: false)
                    let postHex = String(postH, radix: 16, uppercase: false)
                    fputs("[recover-step] KV BYTES CHANGED at position=\(pre.position) "
                          + "slot=\(slot) sid=\(sid) "
                          + "pre=\(preHex) post=\(postHex)"
                          + (abortOnMismatch ? " — aborting session" : " (warn-only)") + "\n",
                          stderr)
                    if abortOnMismatch {
                        s.state = .done
                        s.doneReason = 3
                    }
                }
            }
        }
        // Periodic profiling dump every 100 AR steps. Gated by LM_PROF env.
        // Splits step time into: GPU compute, handler latency, finalize, prep.
        // GPU time is wall - cpu_overhead. Anything left is bandwidth-bound or
        // dispatch-launch-bound on GPU.
        if ProcessInfo.processInfo.environment["LM_PROF"] != nil &&
           prof_arSteps > 0 && prof_arSteps % 100 == 0 {
            let n = Double(prof_arSteps)
            let gpuAvg = prof_gpuMsSum / n
            let wallAvg = prof_wallMsSum / n
            let handlerAvg = prof_handlerLatencyMsSum / n
            let finalizeAvg = prof_finalizeMsSum / n
            let prepAvg = prof_prepMsSum / n
            let cpuAvg = handlerAvg + finalizeAvg + prepAvg
            let line = String(format: "[PROF] ar=%d wall=%.1f gpu=%.1f cpu=%.1f (handler=%.2f final=%.2f prep=%.2f) sched(ar=%d sM=%d sS=%d tM=%d tS=%d)\n",
                              prof_arSteps, wallAvg, gpuAvg, cpuAvg,
                              handlerAvg, finalizeAvg, prepAvg,
                              sched_arCount, sched_softMultiCount, sched_softSingleCount,
                              sched_textMultiCount, sched_textSingleCount)
            if let data = line.data(using: .utf8) {
                FileHandle.standardError.write(data)
            }
        }

        // Token + logit trace (LM_TOKEN_TRACE=1). For each slot, prints the
        // sampled token, its raw logit value, the max logit value, and which
        // token won. Combined with NaN trace below, narrows the moment a
        // degenerate logit distribution appears.
        if ProcessInfo.processInfo.environment["LM_TOKEN_TRACE"] != nil {
            let logP = logits.contents().bindMemory(to: Float16.self,
                                                     capacity: B * VOCAB)
            let gpuTokP = gpu_sampled_tokens.contents().bindMemory(
                to: UInt32.self, capacity: B)
            for slot in 0..<B where p.realSlot[slot] {
                guard let s = p.slotSession[slot], s.state.isBusy else { continue }
                let sampled = gpuTokP[slot]
                var maxLogit: Float = -Float.infinity
                var maxIdx = 0
                let base = slot * VOCAB
                for v in 0..<VOCAB {
                    let lv = Float(logP[base + v])
                    if lv > maxLogit { maxLogit = lv; maxIdx = v }
                }
                let sampledLogit = Float(logP[base + Int(sampled)])
                // Print on every step that samples token in suspicious region,
                // OR every 50 steps for general health.
                let suspicious = (sampled >= 6000 && sampled <= 6500)
                if suspicious || (s.numGenerated % 50 == 0 && s.numGenerated > 0) {
                    let line = "[TOK] step=\(totalSteps) slot=\(slot) sid=\(s.id) numGen=\(s.numGenerated) tok=\(sampled) logit=\(sampledLogit) | argmax=\(maxIdx) maxlogit=\(maxLogit)\n"
                    if let data = line.data(using: .utf8) {
                        FileHandle.standardError.write(data)
                    }
                }
            }
        }
        // NaN trace (gated by LM_NAN_TRACE). Scans hidden + logits + every
        // K/V cache entry that any active slot's last-written page references,
        // prints the FIRST slot/buffer/index that goes non-finite. Wires to
        // the broadcast-bug RCA — find where NaN first appears.
        if ProcessInfo.processInfo.environment["LM_NAN_TRACE"] != nil {
            let hidP = hidden.contents().bindMemory(to: UInt16.self,
                                                     capacity: B * HIDDEN)
            let logP = logits.contents().bindMemory(to: UInt16.self,
                                                     capacity: B * VOCAB)
            for slot in 0..<B where p.realSlot[slot] {
                guard let s = p.slotSession[slot] else { continue }
                // Hidden: scan slot's row.
                var hidNan = -1
                for d in 0..<HIDDEN {
                    let h16 = hidP[slot * HIDDEN + d]
                    // fp16 NaN: exponent all 1s, mantissa nonzero
                    if (h16 & 0x7C00) == 0x7C00 && (h16 & 0x03FF) != 0 {
                        hidNan = d; break
                    }
                }
                if hidNan >= 0 {
                    print("[NAN] step=\(totalSteps) slot=\(slot) sid=\(s.id) hidden[\(hidNan)] is NaN, pos=\(s.position) numGen=\(s.numGenerated)")
                }
                // Logits: scan slot's row, log first NaN index.
                var logNan = -1
                for v in 0..<VOCAB {
                    let l16 = logP[slot * VOCAB + v]
                    if (l16 & 0x7C00) == 0x7C00 && (l16 & 0x03FF) != 0 {
                        logNan = v; break
                    }
                }
                if logNan >= 0 {
                    print("[NAN] step=\(totalSteps) slot=\(slot) sid=\(s.id) logits[\(logNan)] is NaN, pos=\(s.position)")
                }
            }
        }

        // Phase C-Read readback. Pull intensities out of gIntensityBuf
        // (host-visible) into each session's DetectorAttachment state —
        // prev gets last tick's curr, curr gets this tick's measurement.
        // Then evaluate gated triggers for edge-fires and restart any
        // effector envelopes they activate. The effector restarts take
        // effect on the NEXT tick's magnitudeAt() evaluation — classic
        // cross-tick side-chain, no in-CB coupling.
        let intP = gIntensityBuf.contents().bindMemory(to: Float.self,
                                                       capacity: B * MAX_DETECTORS_PER_SLOT)
        // Project-mode measurement readback: the project kernel writes
        // the pre-write projection into gProjectMeasureBuf at each
        // control's measureOutSlot. The staging in prepareARStep assigned
        // these slots by iterating activeControls in order and
        // incrementing a per-session counter for project-mode controls.
        // We replay the same iteration to map back from slot index to
        // which ActiveControl receives the reading.
        let projP = gProjectMeasureBuf.contents().bindMemory(to: Float.self,
                                                              capacity: B * MAX_PROJECT_CONTROLS_PER_SLOT)
        let logCvec = ProcessInfo.processInfo.environment["LM_CVEC_LOG"] != nil
        for slot in 0..<B where p.realSlot[slot] {
            guard let s = p.slotSession[slot] else { continue }
            let base = slot * MAX_DETECTORS_PER_SLOT
            for (i, d) in s.detectors.prefix(MAX_DETECTORS_PER_SLOT).enumerated() {
                d.prevIntensity = d.lastIntensity
                d.lastIntensity = intP[base + i]
            }
            // Project-mode pre-write projections. Replay staging order
            // EXACTLY — gated project controls whose envelope evaluates
            // to 0 are skipped above, so we skip them here too. Nil out
            // their measurement (nothing was written this tick).
            let projBase = slot * MAX_PROJECT_CONTROLS_PER_SLOT
            var projIdx = 0
            for c in s.activeControls where (c.mode == .project || c.mode == .transport) {
                let m = c.magnitudeAt(position: s.position,
                                       turn: s.turnIndex)
                let staged: Bool
                if c.mode == .transport {
                    staged = (m != 0)
                } else {
                    staged = (c.target == nil) || (m != 0)
                }
                if !staged { c.lastProjectMeasurement = nil; continue }
                if projIdx >= MAX_PROJECT_CONTROLS_PER_SLOT { break }
                c.lastProjectMeasurement = projP[projBase + projIdx]
                projIdx += 1
            }
            s.evaluateTriggers(position: s.position, turn: s.turnIndex,
                                log: logCvec ? { print($0) } : nil)
        }

        let gpuTokP = gpu_sampled_tokens.contents().bindMemory(
            to: UInt32.self, capacity: B)
        var emitted = 0
        for slot in 0..<B where p.realSlot[slot] {
            guard let s = p.slotSession[slot], s.state.isBusy else { continue }
            // GPU sampler is the source of truth for every slot —
            // inverse-CDF softmax with logit_bias + temperature + min_p,
            // written during the step CB. CPU `sampleTokenFromLogits`
            // is no longer called from the hot path.
            let sampled = gpuTokP[slot]
            // Track A fold-in: .primed sessions don't bump position
            // (the recover tick ran at position-1, not position).
            // The sampled token is the first generated token.
            if s.state == .primed {
                s.state = .generating
                s.maybeAppendOutput(sampled)
                s.consumedTokens.append(sampled)
                s.nextGeneratedInput = sampled
                s.recordSample(token: sampled)
                s.numGenerated += 1; emitted += 1; totalTokensGenerated += 1
                advanceCotIfActive(s, sampled: sampled)
                if sampled == s.eosId || matchesAnyStopSequence(s) {
                    s.state = .done
                    s.doneReason = 1
                } else if s.numGenerated >= s.maxNewTokens {
                    s.state = .done
                    s.doneReason = 2
                }
                // After the recover tick the cached K/V is intact
                // (benign overwrite of bit-identical bytes). Continue.
                s.position += 1
                continue
            }
            s.position += 1

            if s.state == .priming {
                // Drained ALL chunks? The logit we just computed is the
                // first generated token's prediction. Flip to .generating,
                // emit, check EOS.
                if s.primingQueue.isEmpty {
                    s.state = .generating
                    s.maybeAppendOutput(sampled)
                    s.consumedTokens.append(sampled)
                    s.nextGeneratedInput = sampled
                    s.recordSample(token: sampled)
                    s.numGenerated += 1; emitted += 1; totalTokensGenerated += 1
                    advanceCotIfActive(s, sampled: sampled)
                    if sampled == s.eosId || matchesAnyStopSequence(s) {
                        s.state = .done
                        s.doneReason = 1   // stop / EOS or stop_sequence
                    } else if s.numGenerated >= s.maxNewTokens {
                        s.state = .done
                        s.doneReason = 2   // length / max_tokens budget
                    }
                }
                // else: more priming to do — discard this logit.
            } else {
                s.maybeAppendOutput(sampled)
                // Extend canonical history so promoteFinishedPages can
                // promote pages covering AR-generated positions.
                s.consumedTokens.append(sampled)
                s.nextGeneratedInput = sampled
                s.recordSample(token: sampled)
                s.numGenerated += 1; emitted += 1; totalTokensGenerated += 1
                advanceCotIfActive(s, sampled: sampled)
                if sampled == s.eosId || matchesAnyStopSequence(s) {
                    s.state = .done
                    s.doneReason = 1   // stop / EOS or stop_sequence
                } else if s.numGenerated >= s.maxNewTokens {
                    s.state = .done
                    s.doneReason = 2   // length / max_tokens budget
                }
                // Promote ONLY at page boundaries (every PAGE_SLIDE tokens)
                // to avoid the per-step hash-compute cost dominating AR
                // throughput. promoteFinishedPages is internally bounded
                // by myPromotedSlidePairCount → fullyWritten anyway, but checking
                // before calling skips the function-call overhead.
                if s.position % PAGE_SLIDE == 0 || s.state == .done {
                    promoteFinishedPages(s)
                }
            }
        }
        return emitted
    }

    // True if any of the session's stopSequences matches the tail of
    // s.consumedTokens. O(num_seqs × max_seq_len) per call — for a
    // typical 1-2 sequences of 5-10 tokens that's noise. Sequences
    // longer than the current emit history are skipped.
    @inline(__always)
    private func matchesAnyStopSequence(_ s: Session) -> Bool {
        if s.stopSequences.isEmpty { return false }
        let history = s.consumedTokens
        for seq in s.stopSequences {
            let n = seq.count
            if n == 0 || history.count < n { continue }
            let start = history.count - n
            var match = true
            for k in 0..<n {
                if history[start + k] != seq[k] { match = false; break }
            }
            if match { return true }
        }
        return false
    }

    // If the session has an active structured-cot grammar, advance its
    // state by the bytes of the just-sampled token. Clears `s.cot`
    // when the grammar reaches done — subsequent steps run unconstrained.
    private func advanceCotIfActive(_ s: Session, sampled: UInt32) {
        guard let cot = s.cot else { return }
        let bytes = tokenizer.tokenBytes(sampled)
        if ProcessInfo.processInfo.environment["COT_DEBUG"] != nil {
            let phaseDesc = cot.phases.first.map { p -> String in
                switch p {
                case .literal(let b): return "literal(\(String(decoding: b, as: UTF8.self).debugDescription))@\(cot.literalCursor)"
                case .freeLine: return "freeLine"
                }
            } ?? "(done)"
            let asStr = String(decoding: bytes, as: UTF8.self)
            fputs("[cot] sid=\(s.id) tok=\(sampled) bytes=\(asStr.debugDescription) phase=\(phaseDesc)\n", stderr)
        }
        let done = cot.advance(by: bytes)
        if done { s.cot = nil }
    }

    // Snapshot for single-slot prefill chain. block_table restoration is
    // moved from a `defer` in the old function body into finalize, since
    // we no longer wait inside prep — the kernel reads block_table during
    // CB execution, so restoration can only happen after completion.
    struct PreparedSinglePrefill {
        let cb: MTLCommandBuffer
        let t0: Date
        let session: Session
        let sslot: Int
        let thisTile: Int
        let remaining: Int
        // savedBT is now in engine._savedBTScratch (single CB at a time
        // means one scratch suffices). finalize memcpys back from there.
        let head: PrimingChunk
    }

    // Run a single-slot fast prefill: the given session's next chunk is
    // dispatched as a proper buildPrefillCB filling only its slot. Other
    // slots are silenced via block_table redirect to the scratch strip.
    // Returns true if a prefill actually ran (chunk was consumed).
    @discardableResult
    func stepPrefillForSession(_ s: Session) -> Bool {
        guard let p = prepareSinglePrefill(s) else { return false }
        p.cb.commit(); p.cb.waitUntilCompleted()
        return finalizeSinglePrefill(p)
    }

    func prepareSinglePrefill(_ s: Session) -> PreparedSinglePrefill? {
        guard s.state == .priming, let head = s.primingQueue.first else { return nil }
        // 2026-05-07 (M:K): single-slot prefill puts THIS session at
        // kpos 0, silences the rest. The picker isn't involved (only
        // one session in the prefill); we just stamp s.slot=0 so
        // existing code that uses s.slot reads kpos=0.
        runAdmissionPass()    // ensure pages allocated
        for other in residentSessions.values where other.id != s.id { other.slot = nil }
        s.slot = 0
        let sslot = 0
        let qLen = head.count
        precondition(qLen >= 1)
        let thisTile = min(qLen, MAX_Q_LEN)
        let remaining = qLen - thisTile
        // Synthetic prefix KV install (if staged): allocates ≥1 page,
        // writes prefix bytes to position 0, bumps s.position to 1.
        // Must happen BEFORE positionStart capture so real tokens
        // don't overwrite our synthetic entry.
        installPendingPrefixKVIfAny(s)
        if !ensurePages(s, forKLen: s.position + thisTile + 1) { return nil }
        let positionStart = s.position
        installBlockTable(s, slot: sslot)

        // --- Save block_table; redirect non-target slots to scratch strip ---
        // Restoration runs in finalize after the GPU CB completes. The kernel
        // reads block_table during execution, so restoration cannot happen
        // before completion.
        //
        // 2026-05-07: pre-allocated engine._savedBTScratch + memcpy
        // replaces the per-CB Swift Array allocation (256 KB heap
        // alloc per single-slot prefill) and 65,536-element copy
        // loop. Single CB-at-a-time invariant means one scratch
        // suffices across all prefill paths.
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        memcpy(_savedBTScratch, btP, _savedBTScratchByteCount)
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
        case let .softTokens(buf, _, isFp32, byteOffset, eventTicket):
            // Vision tower output. The pad-blit CB on the vision queue
            // signaled `gVisionEvent` at value `eventTicket`; we encode a
            // pre-prefill CB on the LM queue that encodeWaitForEvent's the
            // same ticket and then dispatches a copy-and-cast kernel that
            // writes pre_hidden[sslot * thisTile ..) from `buf` at byte
            // offset `byteOffset`. Queue ordering on the LM queue means
            // the prefill CB (committed below) sees the populated rows.
            // CPU never blocks. See notes/engine_debloat.md.
            precondition(isFp32, "softTokens fp16 path not yet ported to GPU copy")
            let preCB = queue.makeCommandBuffer()!
            if eventTicket > 0 {
                preCB.encodeWaitForEvent(gVisionEvent, value: eventTicket)
            }
            encVisionSoftsCopyFp32(preCB, src: buf, srcByteOffset: byteOffset,
                                    dst: pre_hidden, dstSlot: sslot, qLen: thisTile)
            preCB.commit()
            skipEmbed = true
        }

        // Single-slot prefill: only slot 0 is active, so the flex-attention
        // CSR only needs to be built for slot 0 (numActiveSlots=1). The mask
        // dispatchers below also restrict their slot-axis dispatch to slot 0.
        precomputeFlexPrefillMasks(qLen: thisTile, positionStart: positionStart,
                                    numActiveSlots: 1)
        // Prefill-time cvec staging. Mirrors the AR-side evaluation in
        // step(): for every ActiveControl on this session, allocate a
        // [B*thisTile] mag buffer, evaluate the envelope at each
        // (slot=sslot, position=positionStart+i), and leave mag=0 for
        // silenced slots + positions outside the envelope window. Hook
        // in encodePrefillTileInto shorts-circuits zero-mag rows in the
        // kernel so idle slots cost near-zero. Pool-backed to avoid per-
        // tile allocations.
        // Prefill staging: for each active control, allocate a per-row
        // buffer sized [B * thisTile] floats. Additive mode fills this
        // with per-row ADD magnitudes (0 = silenced row); project mode
        // fills it with per-row TARGET projections (Float.nan = silenced
        // row, so target=0 can still coerce to zero meaningfully). We
        // allocate TWO buffers per control (mags + targets) from the
        // pool for simplicity; only one is meaningful per dispatch.
        gPrefillControls.removeAll(keepingCapacity: true)
        let prefillRows = B * thisTile
        for (pcIdx, c) in s.activeControls.enumerated() {
            // Per-control scratch buffers. Five per control now:
            //  0 mags (additive), 1 targets (project), 2 measures,
            //  3 transport scales, 4 transport offsets.
            let magsBuf = acquirePrefillMagBuf(pcIdx * 5 + 0)
            let targetsBuf = acquirePrefillMagBuf(pcIdx * 5 + 1)
            let measuresBuf = acquirePrefillMagBuf(pcIdx * 5 + 2)
            let scalesBuf = acquirePrefillMagBuf(pcIdx * 5 + 3)
            let offsetsBuf = acquirePrefillMagBuf(pcIdx * 5 + 4)
            let magsP = magsBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
            let targP = targetsBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
            let scaleP = scalesBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
            let offP = offsetsBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
            for r in 0..<prefillRows {
                magsP[r] = 0                   // additive-mode silenced
                targP[r] = Float.nan           // project-mode silenced
                scaleP[r] = Float.nan          // transport-mode silenced
                offP[r] = 0
            }
            for i in 0..<thisTile {
                let pos = positionStart + i
                let m = c.magnitudeAt(position: pos, turn: s.turnIndex)
                switch c.mode {
                case .additive: magsP[sslot * thisTile + i] = m
                case .project:
                    // Gated project: envelope m is a gate, target is the
                    // coerce value. Gate 0 → NaN sentinel (kernel skips).
                    if let t = c.target {
                        targP[sslot * thisTile + i] = (m != 0) ? t : Float.nan
                    } else {
                        targP[sslot * thisTile + i] = m
                    }
                case .transport:
                    // Envelope as gate; scale/offset precomputed at attach.
                    if m != 0 {
                        scaleP[sslot * thisTile + i] = c.transportScale
                        offP[sslot * thisTile + i] = c.transportOffset
                    }
                }
            }
            gPrefillControls.append(PrefillControl(
                buffer: c.buffer, layer: c.layer, mode: c.mode,
                magsBuf: magsBuf, targetsBuf: targetsBuf,
                projectMeasuresBuf: measuresBuf,
                transportScalesBuf: scalesBuf,
                transportOffsetsBuf: offsetsBuf))
        }
        // gPrefillControls clears at end of prep — kernel arguments are
        // already baked into the CB by buildPrefillCB; the buffers it
        // references are retained by Metal until CB completion.
        defer { gPrefillControls.removeAll(keepingCapacity: true) }

        // GPU sampling params — only sslot is active for this prefill;
        // the kernel short-circuits all other slots. The sampled token
        // (if this prefill drains the chunk queue) lands in
        // gpu_sampled_tokens[sslot] via buildPrefillCB's dispatch.
        var prefillSlotSession: [Session?] = Array(repeating: nil, count: B)
        prefillSlotSession[sslot] = s
        populateSamplingParams(prefillSlotSession)

        // Skip unembed (~150 ms at Dout=262144) on non-final prefill ticks.
        // pendingPrimingTokenCount == thisTile iff this tile drains every queued
        // priming token, which is the only tick whose sampled token feeds
        // AR. False negative would skip the post-prefill sample → bug; the
        // == comparison is exact, so safe.
        let isLastPrefillTick = s.pendingPrimingTokenCount == thisTile
        let t0 = Date()
        // activeB-aware dispatch: single-slot prefill computes ONLY slot 0's
        // qLen real rows (numActiveRows=thisTile) rather than the full
        // B*qLen=2048 rows. Per-row independence of prefill matmuls means
        // slot-0 outputs are byte-identical to the legacy dispatch; the
        // 7-of-8 silenced rows of the legacy path were pure waste (writing
        // garbage to scratch pages that got discarded). At qLen=256, B=8
        // this collapses every matmul stage's M-axis dispatch 8×.
        let cb = buildPrefillCB(weights, qLen: thisTile, skipEmbed: skipEmbed,
                                 skipUnembed: !isLastPrefillTick,
                                 numActiveRows: thisTile)
        return PreparedSinglePrefill(cb: cb, t0: t0, session: s, sslot: sslot,
                                      thisTile: thisTile, remaining: remaining,
                                      head: head)
    }

    @discardableResult
    func finalizeSinglePrefill(_ p: PreparedSinglePrefill) -> Bool {
        lastStepMs = Date().timeIntervalSince(p.t0) * 1000
        totalSteps += 1
        if let err = p.cb.error { print("  GPU prefill error: \(err)"); return false }

        // Restore block_table now that the kernel has finished reading it.
        // Source: engine._savedBTScratch (memcpy'd in at prepare time).
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        memcpy(btP, _savedBTScratch, _savedBTScratchByteCount)

        let s = p.session
        // --- Advance session state by thisTile positions. GPU blit
        // inside the prefill CB already copied pre_logits[sslot, last, :]
        // into logits[sslot, :]; sampled token is in gpu_sampled_tokens[sslot].
        s.position += p.thisTile
        promoteFinishedPages(s)

        // Pop / trim the head chunk.
        switch p.head {
        case .tokens(var ts):
            ts.removeFirst(p.thisTile)
            if ts.isEmpty { s.primingQueue.removeFirst() }
            else          { s.primingQueue[0] = .tokens(ts) }
        case let .softTokens(buf, _, isFp32, byteOffset, _):
            if p.remaining == 0 {
                s.primingQueue.removeFirst()
            } else {
                // Leave the chunk in the queue with its offset advanced
                // by thisTile rows; next tick picks up where we left off.
                // eventTicket = 0 because the next tile's pre-CB doesn't
                // need to re-wait — the previous tile already encoded a
                // wait on this ticket, and queue ordering on the LM queue
                // ensures subsequent CBs see the result.
                let bpe = isFp32 ? 4 : 2
                let newOffset = byteOffset + p.thisTile * HIDDEN * bpe
                s.primingQueue[0] = .softTokens(buffer: buf, count: p.remaining,
                                               isFp32: isFp32, byteOffset: newOffset,
                                               eventTicket: 0)
            }
        }

        // If the queue is now empty, read the GPU-sampled post-prefill
        // logit as the first generated token (sample_token kernel ran
        // at the end of the prefill CB).
        if s.primingQueue.isEmpty {
            let gpuTokP = gpu_sampled_tokens.contents()
                .bindMemory(to: UInt32.self, capacity: B)
            let sampled = gpuTokP[p.sslot]
            s.state = .generating
            s.maybeAppendOutput(sampled)
            s.consumedTokens.append(sampled)
            s.nextGeneratedInput = sampled
            s.numGenerated += 1; totalTokensGenerated += 1
            if sampled == s.eosId {
                s.state = .done
                s.doneReason = 1   // stop / EOS
            } else if s.numGenerated >= s.maxNewTokens {
                s.state = .done
                s.doneReason = 2   // length / max_tokens budget
            }
        }
        return true
    }

    // Snapshot for the .primed → .generating recover-step prefill. Mirrors
    // PreparedSinglePrefill but carries the fact that this is a recover
    // dispatch (thisTile=1, no chunk to pop, position must NOT advance,
    // sampled token is the first generated token).
    struct PreparedRecoverPrefill {
        let cb: MTLCommandBuffer
        let t0: Date
        let session: Session
        let sslot: Int
        // KV byte-hash captured pre-step at the recover position. Used by
        // LM_KV_OVERWRITE_CHECK to verify the prefill kernel re-writes
        // bit-identical K/V bytes (same-kernel-as-producer invariant).
        let preKvHash: UInt64?
        let recoverPos: Int
    }

    // 2026-05-23 (.primed recover-step kernel-pair fix):
    //
    // The .primed → .generating transition needs ONE GPU dispatch to
    // produce the first generated token's logit. Previously this ran via
    // prepareARStep's special-case branch which dispatched the
    // dense_gemv_q8_0_btile_qkv_otf_b1 AR kernel (fused QKV+RMSNorm,
    // serial accumulation) and re-wrote K/V at the prompt's last
    // position. But the cached prefix's producer had written that same
    // position via the prefill_mm_q8_0_swiz kernel (simdgroup matmul,
    // separate RMSNorm dispatch). Same input bits, different fp16 bytes
    // by reduction order alone — LM_KV_OVERWRITE_CHECK reported
    // [recover-step] KV BYTES CHANGED across all 30 layers.
    //
    // Fix: dispatch the prefill kernel at qLen=1 for the recover step.
    // Same kernel as producer → bit-identical K/V → no overwrite.
    //
    // This mirrors prepareSinglePrefill but:
    //   - no primingQueue head requirement (synthesizes the 1-token
    //     "chunk" from consumedTokens[position-1])
    //   - thisTile = 1 (single-position prefill)
    //   - positionStart = s.position - 1 (the recover position, already
    //     covered by adopted K/V from the producer)
    //   - kls/klf set to recoverPos + 1 (the inclusive cached length)
    //   - DOES NOT pop primingQueue (it's empty by .primed contract)
    //   - DOES NOT advance s.position (the recover overwrite is at an
    //     existing position; the sampled token IS the first generated
    //     token that subsequent AR steps consume)
    //
    // Cost: one prefill CB at qLen=1 has B-dim utilization ~3%
    // (NR1=32 tile × 1 row), roughly 2-3× an AR step wall-time. Happens
    // ONCE per adopted-prefix session at admission; trivially amortized.
    func prepareRecoverPrefill(_ s: Session) -> PreparedRecoverPrefill? {
        guard s.state == .primed else { return nil }
        let recoverPos = s.position - 1
        guard recoverPos >= 0,
              recoverPos < s.consumedTokens.count else { return nil }
        // 2026-05-07 (M:K): single-slot prefill puts THIS session at
        // kpos 0, silences the rest. Mirror prepareSinglePrefill.
        runAdmissionPass()    // ensure pages allocated
        for other in residentSessions.values where other.id != s.id { other.slot = nil }
        s.slot = 0
        let sslot = 0
        let thisTile = 1
        // recoverPos is already covered by adopted K/V. Defensive: ensure
        // pages exist for [0..recoverPos] (no growth needed in practice).
        if !ensurePages(s, forKLen: recoverPos + 2) { return nil }
        let positionStart = recoverPos
        installBlockTable(s, slot: sslot)

        // Save block_table; redirect non-target slots to scratch strip.
        // Restoration runs in finalize after the GPU CB completes.
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        memcpy(_savedBTScratch, btP, _savedBTScratchByteCount)
        for slot in 0..<B where slot != sslot {
            for p in 0..<MAX_PAGES_PER_SLOT {
                btP[slot * MAX_PAGES_PER_SLOT + p] =
                    UInt32(SCRATCH_PAGE_BASE + (p % SCRATCH_STRIP))
            }
        }

        // Populate prefill scratch. Only sslot carries real data.
        let tokP = pre_input_tokens.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
        let posP = pre_q_positions.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
        for b in 0..<B {
            for i in 0..<thisTile {
                posP[b * thisTile + i] = UInt32(positionStart + i)
                tokP[b * thisTile + i] = weights.bosTokenId  // silenced-slot filler
            }
        }
        // Real token in sslot's row: the LAST consumed prompt token.
        // This is the same token the producer's prefill wrote at this
        // position; re-dispatching with the same upstream K/V (cached)
        // and same cvecs (adoption invariant) produces bit-identical
        // K/V bytes at recoverPos.
        tokP[sslot * thisTile + 0] = s.consumedTokens[recoverPos]

        let klsP = pre_k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let klfP = pre_k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
        // kLen = positionStart + thisTile = recoverPos + 1 (the inclusive
        // cached length INCLUDING the recover position itself, which is
        // what the prefill kernel expects — same shape as a single-tile
        // prefill that filled positions [0..recoverPos]).
        for b in 0..<B {
            klsP[b] = UInt32(positionStart + thisTile)
            klfP[b] = UInt32(positionStart + thisTile)
        }

        precomputeFlexPrefillMasks(qLen: thisTile, positionStart: positionStart)

        // Cvec staging. By the adoption invariant a .primed session's
        // active controls produce the SAME magnitudes at recoverPos as
        // the producer used (otherwise the adoption would have missed),
        // so the recover prefill sees the same cvec inputs as the
        // producer did. Stage them identically to prepareSinglePrefill
        // (one-row case).
        gPrefillControls.removeAll(keepingCapacity: true)
        let prefillRows = B * thisTile
        for (pcIdx, c) in s.activeControls.enumerated() {
            let magsBuf = acquirePrefillMagBuf(pcIdx * 5 + 0)
            let targetsBuf = acquirePrefillMagBuf(pcIdx * 5 + 1)
            let measuresBuf = acquirePrefillMagBuf(pcIdx * 5 + 2)
            let scalesBuf = acquirePrefillMagBuf(pcIdx * 5 + 3)
            let offsetsBuf = acquirePrefillMagBuf(pcIdx * 5 + 4)
            let magsP = magsBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
            let targP = targetsBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
            let scaleP = scalesBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
            let offP = offsetsBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
            for r in 0..<prefillRows {
                magsP[r] = 0
                targP[r] = Float.nan
                scaleP[r] = Float.nan
                offP[r] = 0
            }
            let pos = positionStart
            let m = c.magnitudeAt(position: pos, turn: s.turnIndex)
            switch c.mode {
            case .additive:
                magsP[sslot * thisTile + 0] = m
            case .project:
                if let t = c.target {
                    targP[sslot * thisTile + 0] = (m != 0) ? t : Float.nan
                } else {
                    targP[sslot * thisTile + 0] = m
                }
            case .transport:
                if m != 0 {
                    scaleP[sslot * thisTile + 0] = c.transportScale
                    offP[sslot * thisTile + 0] = c.transportOffset
                }
            }
            gPrefillControls.append(PrefillControl(
                buffer: c.buffer, layer: c.layer, mode: c.mode,
                magsBuf: magsBuf, targetsBuf: targetsBuf,
                projectMeasuresBuf: measuresBuf,
                transportScalesBuf: scalesBuf,
                transportOffsetsBuf: offsetsBuf))
        }
        defer { gPrefillControls.removeAll(keepingCapacity: true) }

        // Sampling params — only sslot is active; this IS the last
        // prefill tick for the .primed session, so we want the unembed
        // + sample_token dispatch to run and write gpu_sampled_tokens[sslot].
        var prefillSlotSession: [Session?] = Array(repeating: nil, count: B)
        prefillSlotSession[sslot] = s
        populateSamplingParams(prefillSlotSession)

        // LM_KV_OVERWRITE_CHECK: pre-step K/V hash so finalize can
        // verify bit-identicality (the whole point of this fix).
        let kvCheckEnabled = ProcessInfo.processInfo.environment["LM_KV_OVERWRITE_CHECK"] != nil
        let preH: UInt64? = kvCheckEnabled
            ? hashSessionKVAtPosition(s, position: recoverPos)
            : nil

        let t0 = Date()
        // Always unembed: the recover step's sampled token IS the first
        // generated token for the .primed session.
        // numActiveRows=thisTile (i.e. 1) — recover-prefill is single-slot
        // single-token; activeB-aware dispatch collapses every matmul stage
        // to 1 row instead of B*qLen=2048. (Per the activeB refactor; see
        // bootstrap.swift::encodePrefillTileInto numActiveRows threading.)
        let cb = buildPrefillCB(weights, qLen: thisTile, skipEmbed: false,
                                 skipUnembed: false, numActiveRows: thisTile)
        return PreparedRecoverPrefill(cb: cb, t0: t0, session: s, sslot: sslot,
                                       preKvHash: preH, recoverPos: recoverPos)
    }

    @discardableResult
    func finalizeRecoverPrefill(_ p: PreparedRecoverPrefill) -> Bool {
        lastStepMs = Date().timeIntervalSince(p.t0) * 1000
        totalSteps += 1
        if let err = p.cb.error { print("  GPU recover-prefill error: \(err)"); return false }

        // Restore block_table.
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        memcpy(btP, _savedBTScratch, _savedBTScratchByteCount)

        let s = p.session

        // LM_KV_OVERWRITE_CHECK: post-step K/V hash. With the fix
        // (same kernel as producer), these MUST match.
        if let preH = p.preKvHash {
            let postH = hashSessionKVAtPosition(s, position: p.recoverPos)
            if postH != preH {
                let abortOnMismatch = ProcessInfo.processInfo.environment["LM_KV_OVERWRITE_ABORT"] != nil
                let preHex = String(preH, radix: 16, uppercase: false)
                let postHex = String(postH, radix: 16, uppercase: false)
                fputs("[recover-prefill] KV BYTES CHANGED at position=\(p.recoverPos) "
                      + "slot=\(p.sslot) sid=\(s.id) "
                      + "pre=\(preHex) post=\(postHex)"
                      + (abortOnMismatch ? " — aborting session" : " (warn-only)") + "\n",
                      stderr)
                if abortOnMismatch {
                    s.state = .done
                    s.doneReason = 3
                    return true
                }
            }
        }

        // .primed contract: position does NOT advance (the recover write
        // is at an EXISTING position). promoteFinishedPages still safe
        // (no new pages crossed any boundary).
        promoteFinishedPages(s)

        // Read GPU-sampled first generated token and transition to
        // .generating. Mirrors finalizeSinglePrefill's drained-queue path
        // exactly — no consumedTokens append for the recover position
        // (that token is already there from the prompt); only append the
        // newly sampled token.
        let gpuTokP = gpu_sampled_tokens.contents()
            .bindMemory(to: UInt32.self, capacity: B)
        let sampled = gpuTokP[p.sslot]
        s.state = .generating
        s.maybeAppendOutput(sampled)
        s.consumedTokens.append(sampled)
        s.nextGeneratedInput = sampled
        s.numGenerated += 1; totalTokensGenerated += 1
        // s.position += 1 — the prompt-end was at s.position; the first
        // generated token now lives at s.position, and subsequent AR
        // steps will write at s.position + 1, s.position + 2, ... Mirror
        // the .primed → .generating advance from prepareARStep's old path.
        s.position += 1
        if sampled == s.eosId {
            s.state = .done
            s.doneReason = 1
        } else if s.numGenerated >= s.maxNewTokens {
            s.state = .done
            s.doneReason = 2
        }
        return true
    }

    @discardableResult
    func stepRecoverPrefillForSession(_ s: Session) -> Bool {
        guard let p = prepareRecoverPrefill(s) else { return false }
        p.cb.commit(); p.cb.waitUntilCompleted()
        return finalizeRecoverPrefill(p)
    }

    // Unified scheduler tick. Picks between fast prefill and AR batch each
    // call. Path priority (rewritten 2026-05-07 — the prior "max wins"
    // rule was sched_sim_token-falsified, see pickChainPath body):
    //
    //   1. Any session has a .softTokens head chunk  →  soft prefill
    //      (image tokens can't go through AR; multi if ≥ 2 priming).
    //   2. Any session has a .tokens chunk of ≥2 remaining  →  text
    //      prefill (multi if ≥ 2 priming, else single). ALWAYS
    //      preferred over AR: a single-slot prefill costs 1 CB of
    //      all-8-silenced, but saves prompt_len-1 silenced-slot-ticks
    //      from AR-priming the would-be priming session through AR.
    //      For any prompt_len > 1 it wins; sched_sim D1 confirms.
    //   3. Otherwise  →  AR step across all busy sessions.
    //
    // Returns tokens emitted this tick (usually 0 during prefill unless the
    // chunk drained and we sampled the first generated token).
    // Scheduler path enum. Mutually-exclusive categories that map to one
    // of the three kernel shapes (soft-prefill / text-prefill / AR).
    enum ChainPath {
        case idle
        case softMultiPrefill([Session])
        case softSinglePrefill(Session)
        case textMultiPrefill([Session])
        case textSinglePrefill(Session)
        // 2026-05-23 (.primed kernel-pair fix): single-slot prefill at
        // qLen=1 that re-runs the producer's prefill kernel on the
        // recover position, producing bit-identical K/V bytes and the
        // first generated token's logit. Routed BEFORE arStep so .primed
        // sessions don't fall into prepareARStep's AR-kernel branch
        // (whose different reduction order causes the KV-overwrite
        // mismatch that this fix exists to eliminate).
        case recoverPrefill(Session)
        case arStep
    }

    // Pick the kernel category whose ready-slot count is highest. Replaces
    // the old hardcoded cascade ("softs ≥2, then 1, then text ≥2, then 1,
    // then AR") which would run a wasteful 1-slot prefill while leaving 3
    // AR-ready sessions parked. AR / prefill / softs each need their own
    // kernel shape — there is no merged "mixed-mode" CB; the scheduler's
    // job is to choose which kernel to launch, not to merge them.
    //
    // Tiebreak: soft > text > AR. On a tie, prefer the prefill category
    // because completing a prefill grows the future AR pool (a session
    // transitions to .generating once its priming queue drains). Without
    // this preference, two prefilling sessions and two AR-decoding sessions
    // would alternate paths every CB, never letting prefill complete to
    // grow the AR batch toward B.
    // Per-path step counters for pipeline-bubble diagnosis.
    var sched_softMultiCount: Int = 0
    var sched_softSingleCount: Int = 0
    var sched_textMultiCount: Int = 0
    var sched_textSingleCount: Int = 0
    var sched_arCount: Int = 0
    // 2026-05-23 (.primed kernel-pair fix): count of recover-prefill
    // dispatches scheduled. Should equal the number of adopted-prefix
    // sessions admitted; trivially small.
    var sched_recoverPrefillCount: Int = 0
    private func pickChainPath() -> ChainPath {
        let busy = activeSessions.filter { $0.state.isBusy }
        if busy.isEmpty { return .idle }

        var softBusy: [Session] = []
        var textPrefillBusy: [Session] = []
        var primedBusy: [Session] = []
        var nAR = 0
        for s in busy {
            // 2026-05-23 fix: .primed sessions need a 1-token prefill-
            // kernel dispatch (NOT the AR kernel) to recover the first
            // generated token's logit while leaving K/V bit-identical.
            // Peel them off here so they bypass arStep's AR-kernel path.
            if s.state == .primed {
                primedBusy.append(s)
                continue
            }
            switch s.primingQueue.first {
            case .some(.softTokens):
                softBusy.append(s)
            case .some(.tokens(let ts)) where ts.count >= 2:
                textPrefillBusy.append(s)
            default:
                // Either .generating (no chunks consumed by AR step) or
                // .priming with a 1-token tail (popArPrimingToken handles
                // it inside step's per-slot input population).
                nAR += 1
            }
        }
        let nSoft = softBusy.count
        let nText = textPrefillBusy.count

        // 2026-05-07: rewrote path selection to ALWAYS prefer prefill
        // over AR when any session is priming. Previously the rule was
        // "max wins" — pick whichever category (soft/text/AR) has more
        // slots. With nAR=7 and nText=1, that picked AR, which then
        // silenced the 1 priming session's slot for prompt_len AR
        // ticks via popArPrimingToken (slot runs kernel with the
        // prompt token loaded; sampled output discarded). For a
        // 76-token prompt, that's 76 silenced AR ticks for that slot
        // — a ~10% throughput loss per fresh-session admission.
        //
        // Single-slot prefill IS more expensive in the moment (1 CB
        // of all-8-slots silenced for prefill vs 1 CB of 7 AR + 1
        // silenced for AR-priming), but saves prompt_len-1 silenced
        // slot-ticks afterwards. For any prompt_len > 1 it wins; the
        // sched_sim_token D1 sweep confirmed +4-5% steady-state win
        // for "always prefill" over "max wins".
        //
        // Order: soft > text > AR. Within soft/text, multi if ≥ 2
        // priming, else single.

        // .primed recover-prefill: single-slot, qLen=1, prefill kernel.
        // Highest priority because (a) it's at-most-one-per-tick (a
        // .primed session transitions to .generating on its first tick),
        // (b) it unblocks the session for AR participation on the next
        // tick, and (c) running it before any other prefill avoids ever
        // dispatching the AR kernel on a .primed slot (the whole point
        // of the kernel-pair fix). Sort by sid for determinism if more
        // than one is primed concurrently.
        if !primedBusy.isEmpty {
            let pick = primedBusy.min(by: { $0.id < $1.id })!
            sched_recoverPrefillCount += 1
            return .recoverPrefill(pick)
        }
        // Soft (image) prefill takes priority — image tokens can't
        // be AR-primed, so any soft chunk MUST go through prefill.
        if nSoft >= 2 { sched_softMultiCount += 1; return .softMultiPrefill(softBusy) }
        if nSoft == 1 { sched_softSingleCount += 1; return .softSinglePrefill(softBusy[0]) }
        // Text prefill: any priming session with a multi-token chunk
        // wins over AR. The 1-CB silencing cost during single-slot
        // prefill is outweighed by avoiding prompt_len ticks of slot-
        // silencing during AR-priming.
        if nText >= 2 { sched_textMultiCount += 1; return .textMultiPrefill(textPrefillBusy) }
        if nText == 1 { sched_textSingleCount += 1; return .textSinglePrefill(textPrefillBusy[0]) }
        // No priming work — pure AR step.
        if nAR == 0 { return .idle }
        sched_arCount += 1
        return .arStep
    }

    @discardableResult
    // Walk resident sessions whose head chunk is a soft-token chunk
    // with a vision ticket marked failed by the vision dispatcher,
    // and transition them to .done with doneReason=3 (error). This
    // catches the vision-tower-returned-0-softs failure mode BEFORE
    // the LM consumer reads the zero-filled entry buffer and decodes
    // garbage. Side-table approach (gFailedVisionTickets) keeps the
    // chunk struct unchanged. Called at the top of tick() and
    // syncTickStep() so failed sessions never enter pickChainPath.
    private func processFailedVisionSessions() {
        for s in residentSessions.values {
            guard s.state.isBusy, let head = s.primingQueue.first else { continue }
            if case let .softTokens(_, _, _, _, ticket) = head,
               ticket > 0, isVisionTicketFailed(ticket) {
                s.state = .done
                s.doneReason = 3
                s.errMsg = "vision tower returned 0 soft tokens for an image in this stream (rawNPooled<=0; see [vision] stderr line for ticket \(ticket))"
                s.primingQueue.removeAll()
                clearVisionTicket(ticket)
            }
        }
    }

    func tick() -> Int {
        runAdmissionPass()
        processFailedVisionSessions()
        let busy = activeSessions.filter { $0.state.isBusy }
        if busy.isEmpty { return 0 }
        // Accrues only on productive ticks; matches totalSteps cadence.
        totalSlotTicks += busy.count

        switch pickChainPath() {
        case .idle:
            return 0
        case .softMultiPrefill(let sessions):
            let beforeCounts = sessions.map { $0.outputQueue.count }
            _ = stepMultiSlotSoftPrefill(sessions)
            return zip(sessions, beforeCounts).reduce(0) { $0 + ($1.0.outputQueue.count - $1.1) }
        case .softSinglePrefill(let s), .textSinglePrefill(let s):
            let before = s.outputQueue.count
            _ = stepPrefillForSession(s)
            return s.outputQueue.count - before
        case .textMultiPrefill(let sessions):
            let beforeCounts = sessions.map { $0.outputQueue.count }
            _ = stepMultiSlotPrefill(sessions)
            return zip(sessions, beforeCounts).reduce(0) { $0 + ($1.0.outputQueue.count - $1.1) }
        case .recoverPrefill(let s):
            let before = s.outputQueue.count
            _ = stepRecoverPrefillForSession(s)
            return s.outputQueue.count - before
        case .arStep:
            return step()
        }
    }

    // ============================================================
    // syncTickStep: synchronous step. One call = one CB worth of work.
    //
    // The bridge calls this in a loop (while gemma_has_work). Each call
    // picks a path, builds a CB, commits it, blocks for completion, runs
    // the finalize step, and returns. No completion handlers, no
    // background-thread state mutations, no chainInFlight invariant —
    // Swift goes vroom for one CB and gives control back.
    //
    // The contract: only one thread enters this function at a time. The
    // bridge serializes — see notes/decisions/2026-04-26-remove-session-
    // concurrency-primitives.md and the "Phase B" follow-up there.
    //
    // (The previous async chain — chainAdvance + per-path completion
    // handlers running on Metal's background queue — was removed
    // 2026-04-26 per user directive: "the swift backend is a fast metal
    // kernel dispatching machine. it goes vroom. it is not a place that
    // does async spinwaiting semantics, because another different
    // program can give the metal backend correctly templated sequential
    // lists of work queue, or deltas wrt the last work queue.")
    func syncTickStep() {
        runAdmissionPass()
        processFailedVisionSessions()
        let busy = activeSessions.filter { $0.state.isBusy }
        if busy.isEmpty { return }
        totalSlotTicks += busy.count

        switch pickChainPath() {
        case .idle:
            return
        case .softMultiPrefill(let sessions):
            guard let p = prepareMultiSlotSoftPrefill(sessions) else { return }
            p.cb.commit()
            p.cb.waitUntilCompleted()
            _ = finalizeMultiSlotSoftPrefill(p)
        case .softSinglePrefill(let s), .textSinglePrefill(let s):
            guard let p = prepareSinglePrefill(s) else { return }
            p.cb.commit()
            p.cb.waitUntilCompleted()
            _ = finalizeSinglePrefill(p)
        case .textMultiPrefill(let sessions):
            guard let p = prepareMultiSlotPrefill(sessions) else { return }
            p.cb.commit()
            p.cb.waitUntilCompleted()
            _ = finalizeMultiSlotPrefill(p)
        case .recoverPrefill(let s):
            guard let p = prepareRecoverPrefill(s) else { return }
            p.cb.commit()
            p.cb.waitUntilCompleted()
            _ = finalizeRecoverPrefill(p)
        case .arStep:
            // Bridge-vs-engine 10ms/step gap investigation (2026-04-29).
            // Attribute prep (mask precompute, block_table install,
            // sampling-params populate) and finalize (output-queue
            // drain, page promotion) explicitly so the profiler reports
            // the full host-side breakdown.
            let t_prep_start = Date()
            guard let p = prepareARStep() else { return }
            let t_commit = Date()
            p.cb.commit()
            p.cb.waitUntilCompleted()
            let t_done = Date()
            _ = finalizeARStep(p)
            let t_finalize_done = Date()
            let gpuMs = (p.cb.gpuEndTime - p.cb.gpuStartTime) * 1000
            prof_arSteps += 1
            prof_gpuMsSum += gpuMs
            prof_wallMsSum += t_done.timeIntervalSince(t_commit) * 1000
            prof_prepMsSum += t_commit.timeIntervalSince(t_prep_start) * 1000
            prof_finalizeMsSum += t_finalize_done.timeIntervalSince(t_done) * 1000
        }
    }

    // Profiling: GPU + wall ms accumulators for the AR path. Inter-CB
    // gap measurement is gone now that the chain is synchronous (the
    // bridge owns the inter-call gap).
    var prof_arSteps: Int = 0
    var prof_gpuMsSum: Double = 0
    var prof_wallMsSum: Double = 0
    // Retained for FFI-output compatibility; always 0 under sync tick.
    var prof_handlerLatencyMsSum: Double = 0
    var prof_finalizeMsSum: Double = 0
    var prof_prepMsSum: Double = 0
    // ============================================================

    // True if every session in `sessions` is priming with a .tokens head
    // chunk of at least `minTokensThreshold` tokens remaining. Gate for
    // multi-slot fast prefill; single-token tails fall through to AR step
    // where 1-token priming is nearly free (~34 ms vs ~133 ms for fast prefill).
    private func allPrimeReady(_ sessions: [Session], minTokensThreshold: Int) -> Bool {
        for s in sessions {
            guard s.state == .priming else { return false }
            guard let head = s.primingQueue.first else { return false }
            guard case .tokens(let ts) = head, ts.count >= minTokensThreshold else { return false }
        }
        return true
    }

    // Snapshot for multi-slot text-prefill chain. block_table restoration
    // and chunk pop deferred to finalize.
    struct PreparedMultiPrefill {
        let cb: MTLCommandBuffer
        let qLen: Int
        // savedBT lives in engine._savedBTScratch (single-CB-at-a-time);
        // see PreparedSinglePrefill comment.
        let slotSession: [Session?]
        let slotTokens: [[UInt32]]
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
        guard let p = prepareMultiSlotPrefill(sessions) else { return false }
        p.cb.commit(); p.cb.waitUntilCompleted()
        return finalizeMultiSlotPrefill(p)
    }

    func prepareMultiSlotPrefill(_ sessions: [Session]) -> PreparedMultiPrefill? {
        runAdmissionPass()
        // 2026-05-07 (M:K): pick up to B priming sessions and assign
        // them kpos 0..K-1. Non-chosen kernel positions silenced.
        let mapping = pickKernelMappingForPrefill(sessions)
        var slotSession: [Session?] = Array(repeating: nil, count: B)
        var slotTokens: [[UInt32]] = Array(repeating: [], count: B)
        for kpos in 0..<B {
            guard let s = mapping[kpos] else { continue }
            guard case .tokens(let ts) = s.primingQueue.first, !ts.isEmpty else { continue }
            slotSession[kpos] = s
            slotTokens[kpos] = ts
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
        guard any, qLen >= 1 else { return nil }

        // Each participating slot: ensure pages + install real block_table.
        // Restoration happens in finalize, after the GPU has read block_table.
        // 2026-05-07: memcpy into engine._savedBTScratch replaces the
        // per-CB 256 KB Swift Array<UInt32> heap alloc + 65,536-element
        // copy loop. This was THE big thrasher (3 prefill paths × every
        // CB × 256 KB).
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        memcpy(_savedBTScratch, btP, _savedBTScratchByteCount)
        for b in 0..<B {
            if let s = slotSession[b] {
                installPendingPrefixKVIfAny(s)
                if !ensurePages(s, forKLen: s.position + qLen + 1) { return nil }
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

        // Multi-slot cvec staging. Same row-per-position pattern as the
        // single-session stepPrefillForSession path, but iterating every
        // participating slot's controls. Each control gets its own
        // PrefillControl with magsBuf/targetsBuf where ONLY that slot's
        // rows are populated — all other rows stay silenced (mag=0 or
        // target=NaN sentinel) so the kernel no-ops on them. Without
        // this, 4-session simultaneous submits take this fast-prefill
        // path and SKIP all project/additive controls, leaving KV pages
        // that were computed un-steered. AR then fires the project
        // kernel against that un-steered KV state and produces
        // degenerate output (the "de-facto" loop we repro'd with the
        // on-policy matrix).
        gPrefillControls.removeAll(keepingCapacity: true)
        let prefillRows = B * qLen
        var pcSlot = 0
        for b in 0..<B {
            guard let s = slotSession[b] else { continue }
            for c in s.activeControls {
                let magsBuf = acquirePrefillMagBuf(pcSlot * 5 + 0)
                let targetsBuf = acquirePrefillMagBuf(pcSlot * 5 + 1)
                let measuresBuf = acquirePrefillMagBuf(pcSlot * 5 + 2)
                let scalesBuf = acquirePrefillMagBuf(pcSlot * 5 + 3)
                let offsetsBuf = acquirePrefillMagBuf(pcSlot * 5 + 4)
                pcSlot += 1
                let magsP = magsBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
                let targP = targetsBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
                let scaleP = scalesBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
                let offP = offsetsBuf.contents().bindMemory(to: Float.self, capacity: prefillRows)
                for r in 0..<prefillRows {
                    magsP[r] = 0; targP[r] = Float.nan
                    scaleP[r] = Float.nan; offP[r] = 0
                }
                for i in 0..<qLen {
                    let pos = s.position + i
                    let m = c.magnitudeAt(position: pos, turn: s.turnIndex)
                    switch c.mode {
                    case .additive:
                        magsP[b * qLen + i] = m
                    case .project:
                        if let t = c.target {
                            targP[b * qLen + i] = (m != 0) ? t : Float.nan
                        } else {
                            targP[b * qLen + i] = m
                        }
                    case .transport:
                        if m != 0 {
                            scaleP[b * qLen + i] = c.transportScale
                            offP[b * qLen + i] = c.transportOffset
                        }
                    }
                }
                gPrefillControls.append(PrefillControl(
                    buffer: c.buffer, layer: c.layer, mode: c.mode,
                    magsBuf: magsBuf, targetsBuf: targetsBuf,
                    projectMeasuresBuf: measuresBuf,
                    transportScalesBuf: scalesBuf,
                    transportOffsetsBuf: offsetsBuf))
            }
        }
        defer { gPrefillControls.removeAll(keepingCapacity: true) }

        // Populate GPU sampling params for every active slot in this
        // multi-slot prefill. buildPrefillCB's sample_token dispatch
        // reads these; its output lands in gpu_sampled_tokens[slot].
        populateSamplingParams(slotSession)

        // Skip unembed when no slot is on its final prefill tick (none of
        // the slots' total pendingPrimingTokenCount equals qLen). Conservative:
        // ANY slot on final tick → keep unembed so its sample fires.
        var anyOnFinalTick = false
        for s in slotSession {
            if let s = s, s.pendingPrimingTokenCount == qLen { anyOnFinalTick = true; break }
        }
        let cb = buildPrefillCB(weights, qLen: qLen, skipEmbed: false,
                                 skipUnembed: !anyOnFinalTick)
        return PreparedMultiPrefill(cb: cb, qLen: qLen,
                                     slotSession: slotSession, slotTokens: slotTokens)
    }

    @discardableResult
    func finalizeMultiSlotPrefill(_ p: PreparedMultiPrefill) -> Bool {
        totalSteps += 1
        if let err = p.cb.error { print("  GPU multi-prefill error: \(err)"); return false }

        // Restore block_table now that the kernel has finished reading it.
        // Source: engine._savedBTScratch (memcpy'd in at prepare time).
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        memcpy(btP, _savedBTScratch, _savedBTScratchByteCount)

        // Per-slot: advance position, pop chunk, promote pages. GPU
        // blit inside buildPrefillCB already copied final-Q-row logits
        // into `logits`; sampled tokens (when a chunk drains) are in
        // gpu_sampled_tokens[b].
        let gpuTokP = gpu_sampled_tokens.contents()
            .bindMemory(to: UInt32.self, capacity: B)
        for b in 0..<B {
            guard let s = p.slotSession[b] else { continue }
            s.position += p.qLen
            promoteFinishedPages(s)
            // Pop qLen tokens from the session's chunk head.
            var ts = p.slotTokens[b]
            ts.removeFirst(p.qLen)
            if ts.isEmpty { s.primingQueue.removeFirst() }
            else          { s.primingQueue[0] = .tokens(ts) }
            // Transition if chunk is drained.
            if s.primingQueue.isEmpty {
                let sampled = gpuTokP[b]
                s.state = .generating
                s.maybeAppendOutput(sampled)
                s.consumedTokens.append(sampled)
                s.nextGeneratedInput = sampled
                s.numGenerated += 1; totalTokensGenerated += 1
                if sampled == s.eosId {
                    s.state = .done
                    s.doneReason = 1   // stop / EOS
                } else if s.numGenerated >= s.maxNewTokens {
                    s.state = .done
                    s.doneReason = 2   // length / max_tokens budget
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
    // SoftRef captures one slot's view of a softTokens chunk. Lifted to file
    // scope under LmEngine so PreparedSoftPrefill can carry it across the
    // prepare→finalize boundary.
    struct SoftRef {
        let session: Session
        let buffer: MTLBuffer
        let remainingCount: Int
        let isFp32: Bool
        let byteOffset: Int
        let eventTicket: UInt64
    }

    // Snapshot for multi-slot soft-prefill chain.
    struct PreparedSoftPrefill {
        let cb: MTLCommandBuffer
        let qLen: Int
        // savedBT lives in engine._savedBTScratch (single-CB-at-a-time);
        // see PreparedSinglePrefill comment.
        let slotSoft: [Int: SoftRef]
    }

    @discardableResult
    func stepMultiSlotSoftPrefill(_ sessions: [Session]) -> Bool {
        guard let p = prepareMultiSlotSoftPrefill(sessions) else { return false }
        p.cb.commit(); p.cb.waitUntilCompleted()
        return finalizeMultiSlotSoftPrefill(p)
    }

    func prepareMultiSlotSoftPrefill(_ sessions: [Session]) -> PreparedSoftPrefill? {
        runAdmissionPass()
        // 2026-05-07 (M:K): same picker pattern as text prefill.
        let mapping = pickKernelMappingForPrefill(sessions)
        var slotSoft: [Int: SoftRef] = [:]
        for kpos in 0..<B {
            guard let s = mapping[kpos] else { continue }
            guard case let .softTokens(buf, count, isFp32, byteOffset, eventTicket) = s.primingQueue.first
            else { continue }
            slotSoft[kpos] = SoftRef(session: s, buffer: buf, remainingCount: count,
                                       isFp32: isFp32, byteOffset: byteOffset,
                                       eventTicket: eventTicket)
        }
        guard !slotSoft.isEmpty else { return nil }
        // qLen = min over slots of remaining rows, clamped to MAX_Q_LEN.
        var qLen = MAX_Q_LEN
        for (_, sr) in slotSoft { qLen = min(qLen, sr.remainingCount) }
        guard qLen >= 1 else { return nil }

        // Install real block_table entries for participating slots; silence
        // the rest by redirecting their pages to the scratch strip.
        // block_table restoration runs in finalize.
        // 2026-05-07: memcpy into engine._savedBTScratch — see other
        // prefill paths for rationale.
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        memcpy(_savedBTScratch, btP, _savedBTScratchByteCount)
        for b in 0..<B {
            if let sr = slotSoft[b] {
                if !ensurePages(sr.session, forKLen: sr.session.position + qLen + 1) { return nil }
                installBlockTable(sr.session, slot: b)
            } else {
                for p in 0..<MAX_PAGES_PER_SLOT {
                    btP[b * MAX_PAGES_PER_SLOT + p] =
                        UInt32(SCRATCH_PAGE_BASE + (p % SCRATCH_STRIP))
                }
            }
        }

        // Populate pre_hidden[slot * qLen * HIDDEN ..] via a pre-prefill
        // CB on the LM queue. The CB encodeWaitForEvent's the max ticket
        // across participating slots (vision pad-blits signaled in queue
        // order, so the latest ticket implies all earlier ones complete),
        // then dispatches the GPU copy-and-cast kernel per slot. Silenced
        // slots get a CPU-side zero of pre_hidden — they don't need a
        // wait. Queue ordering on the LM queue ensures the prefill CB
        // (built below) sees the populated rows. CPU never blocks.
        let posP = pre_q_positions.contents().bindMemory(to: UInt32.self, capacity: B * MAX_Q_LEN)
        let klsP = pre_k_len_slide.contents().bindMemory(to: UInt32.self, capacity: B)
        let klfP = pre_k_len_full.contents().bindMemory(to: UInt32.self, capacity: B)
        let pH = pre_hidden.contents().assumingMemoryBound(to: Float16.self)
        // Zero silenced-slot rows once on CPU — silenced slots never had
        // a pendingCB so no wait is needed for those rows; their
        // pre_hidden values get discarded post-prefill via block-table
        // redirection but downstream kernels still read them (and NaN
        // in a silenced slot would corrupt across-slot reductions).
        for b in 0..<B {
            if slotSoft[b] != nil { continue }
            let dstBase = (b * qLen) * HIDDEN
            for i in 0..<(qLen * HIDDEN) { pH[dstBase + i] = 0 }
        }
        var maxTicket: UInt64 = 0
        for (_, sr) in slotSoft { maxTicket = max(maxTicket, sr.eventTicket) }
        let preCB = queue.makeCommandBuffer()!
        if maxTicket > 0 {
            preCB.encodeWaitForEvent(gVisionEvent, value: maxTicket)
        }
        var ingestSlots: [VisionSoftsIngestSlot] = []
        for b in 0..<B {
            guard let sr = slotSoft[b] else { continue }
            precondition(sr.isFp32, "softTokens fp16 path not yet ported to GPU copy")
            ingestSlots.append(VisionSoftsIngestSlot(
                src: sr.buffer, srcByteOffset: sr.byteOffset, dstSlot: b))
        }
        encVisionSoftsCopyFp32Multi(preCB, slots: ingestSlots,
                                     dst: pre_hidden, qLen: qLen)
        preCB.commit()

        // Per-slot positions / k_len bookkeeping (silenced and active alike).
        for b in 0..<B {
            if let sr = slotSoft[b] {
                for i in 0..<qLen {
                    posP[b * qLen + i] = UInt32(sr.session.position + i)
                }
                klsP[b] = UInt32(sr.session.position + qLen)
                klfP[b] = UInt32(sr.session.position + qLen)
            } else {
                for i in 0..<qLen { posP[b * qLen + i] = 0 }
                klsP[b] = 1
                klfP[b] = 1
            }
        }

        precomputeFlexPrefillMasks(qLen: qLen, positionStart: 0)

        // Populate GPU sampling params for every slot participating in
        // this soft-prefill; buildPrefillCB's sample_token runs at end.
        var softSessionSlots: [Session?] = Array(repeating: nil, count: B)
        for b in 0..<B {
            if let sr = slotSoft[b] { softSessionSlots[b] = sr.session }
        }
        populateSamplingParams(softSessionSlots)

        // Skip unembed unless some slot is on its final tick (same logic as
        // multi-slot prefill). softTokens chunks always have count == qLen
        // for the slot on its final tile, so pendingPrimingTokenCount == qLen.
        var anyOnFinalTick = false
        for s in softSessionSlots {
            if let s = s, s.pendingPrimingTokenCount == qLen { anyOnFinalTick = true; break }
        }
        let cb = buildPrefillCB(weights, qLen: qLen, skipEmbed: true,
                                 skipUnembed: !anyOnFinalTick)
        return PreparedSoftPrefill(cb: cb, qLen: qLen, slotSoft: slotSoft)
    }

    @discardableResult
    func finalizeMultiSlotSoftPrefill(_ p: PreparedSoftPrefill) -> Bool {
        totalSteps += 1
        if let err = p.cb.error { print("  GPU multi-soft-prefill error: \(err)"); return false }

        // Restore block_table now that the kernel has finished reading it.
        // Source: engine._savedBTScratch (memcpy'd in at prepare time).
        let btP = block_table.contents().bindMemory(to: UInt32.self, capacity: B * MAX_PAGES_PER_SLOT)
        memcpy(btP, _savedBTScratch, _savedBTScratchByteCount)

        // Per-slot: advance position, promote pages, advance chunk
        // state. Final-Q-row logit copy is done on-GPU by the prefill
        // CB; sampled first-gen tokens (if a session's queue drains)
        // are in gpu_sampled_tokens[b].
        let gpuTokP = gpu_sampled_tokens.contents()
            .bindMemory(to: UInt32.self, capacity: B)
        for b in 0..<B {
            guard let sr = p.slotSoft[b] else { continue }
            let s = sr.session
            s.position += p.qLen
            promoteFinishedPages(s)
            let remaining = sr.remainingCount - p.qLen
            if remaining <= 0 {
                s.primingQueue.removeFirst()
            } else {
                let bpe = sr.isFp32 ? 4 : 2
                let newOffset = sr.byteOffset + p.qLen * HIDDEN * bpe
                // eventTicket = 0: subsequent tile's pre-CB inherits LM-
                // queue ordering vs this CB; no further wait required.
                s.primingQueue[0] = .softTokens(buffer: sr.buffer, count: remaining,
                                               isFp32: sr.isFp32, byteOffset: newOffset,
                                               eventTicket: 0)
            }
            // If this was the last chunk of the session's priming queue,
            // transition to .generating + use the GPU-sampled first token.
            if s.primingQueue.isEmpty {
                let sampled = gpuTokP[b]
                s.state = .generating
                s.maybeAppendOutput(sampled)
                s.consumedTokens.append(sampled)
                s.nextGeneratedInput = sampled
                s.numGenerated += 1; totalTokensGenerated += 1
                if sampled == s.eosId {
                    s.state = .done
                    s.doneReason = 1   // stop / EOS
                } else if s.numGenerated >= s.maxNewTokens {
                    s.state = .done
                    s.doneReason = 2   // length / max_tokens budget
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

