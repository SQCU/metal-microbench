# Track C — cvecDigest tightening

> **STATUS:** Folded into Track D. Under D the per-page UInt64 digest
> becomes a `CvecAnchorTag` struct stored at each `TrieAnchor.byCvecAnchorTag`
> map (per Track D §6 "(ii) Per-page-pair cvec partition at anchors").
> All over-partitioning fixes below (drop redundant `startPosition`/
> `startTurn`, units-gate the anchors, phase-gate the envelope params,
> quantize float bitpatterns, skip near-zero-magnitude controls) apply
> unchanged — they become the fields of `CvecAnchorTag` instead of
> entropy mixed into a single UInt64.

---

# Audit report: `cvecDigestForPage` over-partitioning

Read-only audit of `/Users/mdot/metal-microbench/lm_engine.swift:385-480` (`computeCvecDigest`) and its callers at `/Users/mdot/metal-microbench/lm_engine.swift:1051` and `/Users/mdot/metal-microbench/lm_engine.swift:1839`.

## 1. Enumeration of digest inputs

The digest mixes the following per intersecting `ActiveControl`, in this order (`lm_engine.swift:450-477`):

| Input | Source (line) | What it physically affects in K/V | When it doesn't affect THIS page's K/V |
|---|---|---|---|
| `entries.count` | `lm_engine.swift:450` | Number of controls intersecting the page; each is an independent residual perturbation. | If two pages have the same count but different control identities, this collapses identically — only safe because subsequent fields disambiguate. Necessary as a length prefix to prevent ambiguity (control A alone vs. A+B-cancel). Keep. |
| `e.layer` | `lm_engine.swift:452` | The injection happens at post-FFN residual of layer `L` (`bootstrap.swift:5036-5060`). It perturbs K/V at layers `L+1 … NUM_LAYERS-1`. | Always affects some K/V layer set (unless `L == NUM_LAYERS-1`, in which case it affects no K/V — only the final residual that feeds unembed). For `L == NUM_LAYERS-1` the control is K/V-irrelevant and could be excluded; in practice rarely used. Keep but consider exclusion sentinel. |
| `e.cvecId` (UTF-8) | `lm_engine.swift:453` | Identifies which direction is added/projected (kernels read `cvec` buffer at `kernels.swift:236-461`). | Two cvecIds bound to byte-identical buffers would produce identical K/V but different digests. Negligible in practice (the bridge dedup'd via registry, `bridge.py:1087-1110`). Keep. |
| `e.startOffset` (`cStart - pageStart`) | `lm_engine.swift:410, 454` | Together with envelope params, this fixes envelope phase relative to the page's first position. Affects K/V via `magnitudeAt(position, turn)` for `position ∈ [pageStart, pageEnd)`. | **Over-partitioning case: if the page lies entirely in sustain (i.e., `pageStart − startPosition ≥ attack+decay` AND `pageEnd ≤ stopPosition`, units=tokens), `magnitudeAt` returns `sustain = peak·sustainLevel` for every position regardless of `startOffset`** (see `lm_engine.swift:332-339`). Also redundant with `startPosition` (they're a linear pair). |
| `e.stopOffset` | `lm_engine.swift:411, 455` | Together with `release`, fixes the release ramp shape. Affects K/V only on positions in `[stopPosition, stopPosition+release)`. | If the page is fully before `stopPosition` or fully after `stopPosition + release`, the release-phase boundary is invisible to this page. **`stopOffset = Int.max - pageStart` is mixed for unstopped controls — needless variation.** |
| `e.startPosition` (raw) | `lm_engine.swift:456` | Same physical effect as `startOffset` above — redundant with it once `pageStart` is fixed. | **Redundant**: `startOffset = startPosition − pageStart`; the digest is computed per page, so mixing both is double-counting and exposes the absolute clock to the hash. Two sessions whose envelopes share the same relative phase but were anchored at different absolute positions get different digests for no K/V reason. |
| `e.startTurn` (raw) | `lm_engine.swift:457` | Same as `startPosition` but in turns-units envelopes. Affects K/V only when `envelope.units == .turns`. | **Always mixed unconditionally**, even when `units == .tokens` — `startTurn` is dead bits for token-units envelopes (the kernel never reads it via `magnitudeAt`). Symmetrically, `startPosition` is dead bits for turn-units envelopes. |
| `e.stopPosition`, `e.stopTurn` | `lm_engine.swift:458-459` | Same conditional logic as start fields. | Same over-partitioning: dead bits for the wrong-units envelope, and the absolute (un-rebased) value is mixed. |
| `e.polarity` | `lm_engine.swift:460` | Scales every magnitude (`magnitudeAt: peak = peakMagnitude * polarity`). | Always affects K/V if envelope is non-zero in this page. Keep. Note: `polarity * peakMagnitude` is the only product the kernel sees; mixing them separately is fine but two configurations with the same product produce identical K/V and different digests — minor over-partitioning. |
| `e.env.attack` | `lm_engine.swift:461` | Shapes magnitude for `elapsed < attack`. | If page is past `startPosition + attack` everywhere, attack value is K/V-irrelevant for this page. Over-partitioning. |
| `e.env.decay` | `lm_engine.swift:462` | Shapes magnitude for `attack ≤ elapsed < attack+decay`. | If page is past `startPosition + attack + decay` everywhere, decay value is K/V-irrelevant. Over-partitioning. |
| `e.env.sustainLevel` | `lm_engine.swift:463` | Sets the plateau magnitude after decay. | If the page is entirely inside attack (and there's no decay-to-sustain transition within or before the page) AND there's no release within the page, sustainLevel is K/V-irrelevant. Rare. |
| `e.env.release` | `lm_engine.swift:464` | Shapes the release ramp. | If page doesn't cross `[stopPosition, stopPosition+release]`, release is K/V-irrelevant. |
| `e.env.peakMagnitude` | `lm_engine.swift:465` | Scales every magnitude. | Always affects K/V if envelope is non-zero in page. Keep. |
| `e.env.shape.rawValue` | `lm_engine.swift:466` | Picks the easing function (`linear/expIn/expOut/cubic`). | Only matters during attack, decay, OR release ramps. If page is fully in sustain (or fully magnitude-zero), shape is irrelevant. |
| `e.env.units.rawValue` | `lm_engine.swift:467` | Picks `position - startPosition` vs `turn - startTurn` for elapsed. | Always affects which time axis matters. Keep but use it to gate `startPosition`/`startTurn` inclusion. |
| `e.mode.rawValue` | `lm_engine.swift:468` | Picks kernel: additive `+=`, project (coerce projection), transport (Brenier map). | Always affects K/V values produced. Keep. |
| `e.target` (presence + value) | `lm_engine.swift:471` | Only meaningful for `mode == .project` with non-nil target (gated coerce). For additive/transport this field is unused (`bootstrap.swift:5043-5060`). | For `mode != .project`, target is dead bits. |
| `e.transportScale` (raw bits) | `lm_engine.swift:475` | Only used by `transport_cvector_*_fp16` kernels (`kernels.swift:358-409`). | For `mode != .transport`, dead bits. Also: **float-bitpattern equality** rejects `0.999999f` vs `1.000000f` as different, even though K/V differ only at the noise floor. |
| `e.transportOffset` (raw bits) | `lm_engine.swift:476` | Same as scale. | Same over-partitioning. |

The `intersects` test at `lm_engine.swift:406` correctly excludes controls whose window doesn't reach the page — that part is fine. The collapse-to-zero at `lm_engine.swift:422` (no controls intersect → digest=0) is also fine and necessary for prefix-cache reuse with unsteered sessions.

## 2. Detailed analysis of suspect inputs

### `c.startPosition` / `c.startTurn`

The chain from start fields to K/V is:

```
startPosition → elapsed = position - startPosition  (lm_engine.swift:316)
             → mag = magnitudeAt(position, turn)     (lm_engine.swift:313-346)
             → residual[L] += mag * cvec             (kernels.swift:236-246, bootstrap.swift:5046)
             → QKV_proj at L+1, …                    (K/V at this page diverges)
```

Walk through: **envelope same-shape, different `startPosition`, page P at `[pageStart, pageEnd)`, units=tokens**:

1. **If `pageEnd ≤ startPosition`**: `elapsed < 0` for every position in page → mag=0 → no residual perturbation → K/V identical to unsteered. Already handled by the `intersects` test (`cStart < pageEnd && cStop > pageStart`).

2. **If `pageStart < startPosition < pageEnd`**: the attack ramp starts inside the page. Magnitudes within page depend on exact `startPosition`. Two different `startPosition` values → different K/V. **MUST include.**

3. **If `startPosition + attack + decay ≤ pageStart < pageEnd ≤ stopPosition` (page fully in sustain)**: `magnitudeAt` returns `sustain` (constant) for every position in the page. **`startPosition` has zero effect on K/V at this page.** Two sessions with the same envelope, same cvecId, same layer, same sustain plateau but anchored at different earlier positions produce IDENTICAL K/V at this page, but the current digest gives them different keys. **Over-partitioning.**

4. **If `startPosition + attack + decay ≤ pageStart < stopPosition < pageEnd < stopPosition + release`**: page covers a release ramp; `magnitudeAt` depends on `stopPosition`, not `startPosition`. **`startPosition` again has zero effect on K/V.** **Over-partitioning.**

5. **If `pageStart ≥ stopPosition + release`**: `intersects` excludes the control (with the +release rounding at `lm_engine.swift:404`). Good.

So `startPosition` only affects K/V at the page when the **attack-or-decay ramp overlaps `[pageStart, pageEnd)`**, i.e., when `pageStart - startPosition < attack + decay` AND `pageEnd > startPosition`. Outside that window, including `startPosition` (or its derivative `startOffset`) in the digest is pure over-partitioning.

Symmetrically, `stopPosition` only affects K/V when the **release ramp overlaps `[pageStart, pageEnd)`**, i.e., `stopPosition < pageEnd < stopPosition + release` (or, more precisely, the page intersects `[stopPosition, stopPosition + release]`).

For `units == .turns`: `startPosition` is entirely dead bits — the kernel never consults it (`magnitudeAt` switches on `units` and uses `startTurn`). Same in reverse.

**Real-world frequency of this bug**: high. Common patterns where two sessions have the same envelope but different start positions:
- Same prompt with one prior turn vs. two prior turns of warm-up → second turn's controls re-anchored via `touch`/`continue` (`ffi_batch.swift:512-516, 528-533`) get `startPosition = s.positionForDebug`, which is unique per session.
- Multi-turn conversations where the user adds tokens between control installs — the controls are cleared and re-installed every `continue`, picking up the current position as the new anchor.
- Two parallel sessions with identical config but different conversation lengths at install time.

### `transportScale` / `transportOffset` as float bitpatterns

These are client-supplied Brenier-map coefficients (`bridge.py:1226-1227`). The bridge accepts arbitrary floats from JSON. In practice these come from either:
- A config file (typically 4-6 significant decimal digits) — bitpattern-stable across reloads.
- A UI slider (any IEEE 754 float matching keystrokes) — bitpattern-unstable; `1.5` typed vs. `1.5000001` interpolated mouse drag will collide differently.
- A Python-side `μ_tgt − scale * μ_src` computation, where the inputs themselves come from float32 PC statistics. Two runs that retrain PC stats on the same data may produce values that differ in the bottom-byte mantissa.

The K/V effect of scale/offset goes through `delta = (scale-1)·a + offset` (`kernels.swift:374`), then the residual update. A 1-ulp difference in scale produces an O(2^-23 · |residual projection|) difference in residual ≈ noise floor of fp16 K/V storage. For digest purposes, **bitpattern equality is much stricter than K/V equivalence.**

A practical tolerance: quantize to ~16-20 bits of mantissa precision (round-to-fixed at ~1e-4 absolute). This recovers identical digests for "config file reloaded" cases without colliding distinct UI presets. Cleanest implementation: `mixU64(UInt64(bitPattern: Int64((scale * 65536).rounded())))` (Q16.16) or similar fixed-point hash.

Note: `transportScale = transportOffset = 0` is mixed for non-transport modes too (the per-control `0` is constant, so it adds no entropy — just wasted FNV cycles). Cheap to fix.

### Envelope decay tail

If a session has been steering for many turns and the envelope's release has fully decayed at `position < pageStart - release` (assuming a stop has fired), `magnitudeAt` returns 0 for every position in the page. The `intersects` test partly handles this: `cStop = stopPosition + ceil(release)`, and if `cStop ≤ pageStart`, the control is excluded.

But two failure cases remain:
- **Unstopped controls (no `stopPosition`)**: `cStop = Int.max`, so the control always intersects future pages, even if `sustainLevel == 0` (envelope effectively decayed to zero in sustain). `magnitudeAt` returns `sustain = peak·sustainLevel·polarity = 0` for every position past attack+decay. K/V is unperturbed but digest is non-zero.
- **Sustained-but-zero**: same as above. The intersect test doesn't look at magnitude values.

A tighter test would compute the peak magnitude over the page (closed-form from envelope params) and skip if it's below an epsilon (say, `1e-3 * peakMagnitude`).

## 3. Proposed tighter `cvecDigestForPage`

Pseudocode (Swift sketch). New helpers:

```swift
// Closed-form: peak |magnitude| over [pageStart, pageEnd) for this control.
// O(1), no per-position loop.
func peakMagnitudeOverPage(_ c: ActiveControl,
                            pageStart: Int, pageEnd: Int) -> Float {
    // For tokens-units: use (position - startPosition); turns: skip the
    // per-position bound check and use turnIndex (constant within a page
    // boundary advance — turn only changes between user messages).
    // Compute mag at each phase boundary clipped into [pageStart, pageEnd-1]
    // (attack-end, decay-end, stop, release-end) and take the max-abs.
    // For turns-units envelopes, magnitude is constant across positions
    // within a single page → evaluate once at `position = pageStart`.
    ...
}

// Quantize float to ~1e-4 absolute precision for tolerance-friendly hashing.
@inline(__always) func quantF32(_ f: Float, scale: Float = 65536) -> Int64 {
    return Int64((f * scale).rounded())
}

// Classify which envelope phase(s) the page overlaps, given units.
enum PageEnvPhase: OptionSet {
    case attack, decay, sustain, release
}
func phasesOverPage(_ c: ActiveControl, pageStart: Int, pageEnd: Int,
                    currentTurn: Int) -> PageEnvPhase { ... }
```

Replace the per-entry mix loop with:

```swift
mixInt(entries.count)
for e in entries {
    // (1) Identity always — these define which physical perturbation runs.
    mixInt(e.layer)
    mixString(e.cvecId)
    mixString(e.mode.rawValue)
    mixString(e.env.units.rawValue)

    // (2) Skip controls whose peak |magnitude| over this page is below
    //     an epsilon. They don't perturb K/V to within fp16 precision.
    let peakHere = peakMagnitudeOverPage(c, pageStart, pageEnd)
    if abs(peakHere) < EPS_MAG * abs(e.env.peakMagnitude) {
        // Treat as if it didn't intersect: contribute identity bits only,
        // not envelope params. Better: drop from entries[] entirely (and
        // recompute entries.isEmpty short-circuit).
        continue
    }

    // (3) Phase-conditional inclusion of envelope params.
    let phases = phasesOverPage(c, pageStart, pageEnd, currentTurn)
    if phases.contains(.attack)  { mixU64(quantF32(e.env.attack))  }
    if phases.contains(.decay)   { mixU64(quantF32(e.env.decay))   }
    if phases.contains(.sustain) { mixU64(quantF32(e.env.sustainLevel)) }
    if phases.contains(.release) { mixU64(quantF32(e.env.release)) }
    // Shape only matters during a non-trivial ramp.
    if phases.contains(.attack) || phases.contains(.decay) || phases.contains(.release) {
        mixString(e.env.shape.rawValue)
    }
    // Peak/polarity scale magnitudes everywhere; mix the product (only the
    // product is observable downstream).
    mixU64(quantF32(e.env.peakMagnitude * e.polarity))

    // (4) Anchor inclusion ONLY when the corresponding ramp crosses the page.
    //     For .attack/.decay overlap: include start{Position|Turn}-relative-
    //     to-page-start. For .release overlap: include stop{Position|Turn}-
    //     relative-to-page-start. For pure-sustain pages, neither is needed.
    if phases.contains(.attack) || phases.contains(.decay) {
        switch e.env.units {
        case .tokens: mixInt(e.startPosition - pageStart)
        case .turns:  mixInt(e.startTurn)  // turns rebased against current turn — see Risk surface
        }
    }
    if phases.contains(.release) {
        switch e.env.units {
        case .tokens: mixInt((e.stopPosition ?? Int.min) - pageStart)
        case .turns:  mixInt(e.stopTurn ?? Int.min)
        }
    }

    // (5) Target only for project mode.
    if e.mode == .project {
        if let t = e.target { mixU64(1); mixU64(quantF32(t)) } else { mixU64(0) }
    }

    // (6) Transport params only for transport mode, quantized.
    if e.mode == .transport {
        mixU64(quantF32(e.transportScale))
        mixU64(quantF32(e.transportOffset))
    }
}
```

Five concrete changes vs. current code:
1. **Drop redundant `startPosition`/`startTurn`**: keep only `startOffset` (when relevant).
2. **Phase-gate the envelope params**: attack/decay/sustain/release/shape only when the corresponding phase overlaps the page.
3. **Units-gate the anchors**: include `startPosition` for tokens-units, `startTurn` for turns-units — never both.
4. **Quantize floats**: 16.16 fixed-point or similar tolerance, replacing raw bitpatterns.
5. **Skip near-zero-magnitude controls**: closed-form peak over the page; if below `EPS_MAG`, contribute identity bits only (or drop from entries entirely so `entries.isEmpty` collapse to digest=0 fires).

## 4. Quantification of hit-rate recovery

These are predictions based on the digest's input dependence; integration measurement should follow.

### Case A — "same prompt, same cvec config, installed at slightly different turns"

The most common multi-session pattern (e.g., A/B-testing one persona's responses against another after the same opener).

- **Current behavior**: Session 1 installs control at `(pos=0, turn=0)` via initial submit (`ffi_batch.swift:457-459`); session 2 enters mid-conversation via `continue` which fires `clearControls() + addControl(...) with startPosition=s.positionForDebug, startTurn=s.turnIndex` (`ffi_batch.swift:511-516, 528-533`). Even though the prompts match, `startPosition`/`startTurn` differ → digests differ → 0% hit rate on the common prefix.
- **Proposed behavior**: For pages fully past the attack+decay window of session 1's envelope (and not crossing release), the digest no longer depends on `startPosition`. Expected hit rate ≈ token-only hit rate (~80-100% depending on prompt overlap). For pages within attack/decay, no hit (correct behavior — different attack timing → different K/V).

Predicted hit-rate uplift on cached prefix length: **0% → ~(L - (attack+decay)) / L** where L = prompt length in pages. For typical envelopes (`attack ~ 4 tokens`, `decay ~ 4 tokens`, page = 16 tokens), only the first page is uncacheable; pages 2…N hit. For a 10-page prompt, this is **0% → ~90%**.

### Case B — "same prompt, identical cvec config"

Should always hit. Is there a current bug preventing this?

- Sessions submitting identical `controls` payload to the bridge: both routed through `controlsFromDecoded`. If both are initial submits (action=0), both get `startPosition=0, startTurn=0` (`ffi_batch.swift:457-459`). Same envelope params, same cvecId → identical digest → hit. **Works correctly today.**
- However, any mix of initial vs. continue (e.g., session 1 came in with the prompt as one shot, session 2 was built turn-by-turn via continue) will produce different `startPosition`/`startTurn`, causing different digests → miss. **This is the bug from Case A surfacing again.** With the fix, sustain-phase pages reconvene.

### Case C — "same prompt, cvec config differing only in long-decayed envelope from 50 pages ago"

E.g., session 1 installed a short transient steer in turn 1 (released by turn 2), and at turn 10 wants to share the page-100 K/V with session 2 (same prompt, no such transient).

- **Current behavior**: session 1's control has `stopPosition + release < pageStart` for page 100 → the `intersects` test at `lm_engine.swift:406-407` correctly excludes it → digest=0 → hit. **Works correctly today** *for stopped controls.*
- **Edge case**: if the user forgot to `release`/`stop` the control (no `stopPosition`), it sits at `sustainLevel=0` indefinitely. `cStop = Int.max` → intersects every future page → non-zero digest → miss. **Currently broken.** Proposed fix: peak-magnitude-over-page test catches this (peakHere ≈ 0 → skip).

## 5. Test plan

### Unit tests for `computeCvecDigest`

Each test fixes a baseline `ActiveControl` and varies one input; asserts digest is equal or unequal as expected. Test file would live as `Tests/CvecDigestTests.swift` or similar.

Token-axis envelope:
1. **Identity**: two identical controls → equal digest.
2. **Sustain-phase invariance to `startPosition`**: envelope with `attack=4, decay=4, sustainLevel=0.5, peakMagnitude=1`, page=[64,80); control 1 `startPosition=0`, control 2 `startPosition=16`. Both pages lie fully in sustain. Current: different. Proposed: equal.
3. **Attack-phase sensitivity to `startPosition`**: same envelope, page=[0,16); control 1 `startPosition=0`, control 2 `startPosition=4`. Both: different (correct).
4. **Release-phase boundary**: same envelope, `stopPosition=64, release=8`, page=[64,80). `release` matters; `attack/decay` don't. Vary `attack` only → proposed: equal; vary `release` only → proposed: different.
5. **Decayed-to-zero**: `sustainLevel=0`, no `stopPosition`, page far past `attack+decay`. Proposed: digest equals digest with this control absent (i.e., 0 if it was the only control).
6. **Float tolerance**: `transportScale=1.0` vs. `1.0 + 1e-7` (sub-ulp at that magnitude). Current: different. Proposed: equal.
7. **Units gating**: turns-units envelope with `startTurn=5`, vary `startPosition` over wildly different values. Proposed: equal (startPosition is dead bits).
8. **Mode gating**: additive-mode control with `transportScale=42`. Vary `transportScale`. Proposed: equal (dead bits).
9. **Target gating**: additive-mode control with `target=Float.nan`. Vary `target`. Proposed: equal (dead bits).
10. **No-intersect**: control with `startPosition=100, pageEnd=80` → digest = 0 (unchanged from current).

Cross-axis:
11. **Order invariance**: two controls in different addition order → equal digest (current code already sorts at `lm_engine.swift:423-427`; preserve).
12. **Distinct controls → distinct digests**: A+B vs. A alone → different.

### Integration tests

- Spin up two sessions via bridge:
  - Session 1: POST `/v1/chat/completions` with full prompt + `controls=[{cvec_id, layer=10, peak_magnitude=1, attack=4, decay=4, sustain_level=0.5}]`, action=submit.
  - Session 2: POST `/v1/chat/completions` with same prompt + same controls.
  - Wait for both to enter prefill; capture `pageManager.stats()` before/after. Assert page-share count grows by `prompt_pages - 1` (all pages except page 0 covering the attack ramp).
- Pre-fix baseline measurement: assert current code yields 0 shared pages.
- A/B-testing pattern: session 1 single-shot, session 2 built via two `continue` calls (segment 1 then segment 2). Assert post-fix hit rate matches single-shot/single-shot case for sustain-phase pages.

## 6. Risk surface — can the tighter digest produce a FALSE HIT?

A false hit means two sessions share a page whose K/V values actually differ. Run through the proposed simplifications:

### Risk 1 — Phase-gating sustain pages

Two sessions in pure sustain at page P, but with different `startPosition`. **K/V are identical** as long as `magnitudeAt` returns the same scalar — which it does in sustain (it returns `sustain = peak·sustainLevel` independent of position). The cvec direction, layer, and magnitude scalar are all that the kernel sees (`kernels.swift:236-246`). **Safe.**

Caveat: this assumes the *page boundary* itself is at the same position offset for both sessions. PAGE_SLIDE=16 is a hard alignment (`bootstrap.swift:641`); since `pageStart = p * PAGE_SLIDE` is identical across sessions for the same page index `p`, and tokens are positionally-anchored, this holds. **No risk.**

### Risk 2 — Float quantization of transport coefficients

A 1-ulp quantization tolerance produces K/V deltas at the residual write of order `tolerance · |residual_projection| · cvec[i]`. For `tolerance = 1.5e-5` (the Q16.16 grid), and `|residual| ≤ ~5` (post-RMSNorm and layer scaling), the residual delta per dim is ≤ ~7.5e-5 ≈ 5 ULPs of fp16. K/V then runs through QKV projection (multiplies and accumulates over HIDDEN=2816 dims), then RoPE, then softmax — each stage may amplify. Worst case: divergent attention pattern only if the residual was near a softmax tipping point.

This is **the only proposed change with non-trivial false-hit risk.** Bound it by:
- Choose tolerance ≈ 1e-4 (Q12.16 grid) → residual delta ≤ ~5e-4 per dim → still well within fp16 noise.
- Or restrict tolerance to transport mode only (project/additive use `peakMagnitude` which is typically integer-friendly user input, less benefit from quantization).
- Or keep raw bitpatterns and document the residual variability in client-side guidance.

### Risk 3 — Peak-magnitude-over-page epsilon skip

If `EPS_MAG = 1e-3 * peakMagnitude` and we skip a control, the residual perturbation we're failing to account for is at most `1e-3 * peakMagnitude * |cvec|` per dim. For cvecs normalized to ‖cvec‖ ≈ 1 and `peakMagnitude ≈ 1`, this is ≤ 1e-3 per dim — clearly below fp16 noise (fp16 resolution ≈ 1e-3 for values ≈ 1). **Safe within an order of magnitude.** Tightening to `EPS_MAG = 1e-5` adds margin at negligible hit-rate cost.

### Risk 4 — Anchor exclusion for sustain pages

Already covered under Risk 1. **No additional risk.**

### Bounding citation

The kernel that bounds the residual perturbation: `add_scaled_cvector_prefill_fp16` at `kernels.swift:448-462`. Lines 457-458 short-circuit on `m == 0.0f` (exact), so a digest skip on "effective zero" matches the kernel's own behavior if we make `EPS_MAG = 0` strict. For non-zero but small `m`, the write at line 460 is `dst[r*N+i] = half(float(dst[r*N+i]) + m * float(cvec[i]))`. The fp16 round-to-nearest tie-breaking means writes with `|m * cvec[i]| < 2^-24 * |dst|` are no-ops, giving the kernel a built-in noise floor we can lean on.

## 7. Interactions with other tracks

### Track A — Backstop removal

If "backstop" refers to a fallback hash-collision safety check during page adoption, the proposed digest does NOT change collision probability (still 64-bit FNV with same finalization). The backstop's role is independent. **No interaction.**

However: if the backstop currently masks digest-driven misses (i.e., adoption falls back to a tokens-only key if the digest-keyed lookup misses), removing it would expose the over-partitioning more sharply. **Removing the backstop is safer AFTER this fix lands** — otherwise users will see a hit-rate regression.

### Track B — Partial-page promotion

The current digest is computed per full page (`pageSize = PAGE_SLIDE = 16`, `lm_engine.swift:1051, 1839`). Partial-page promotion would require computing the digest over `[pageStart, pageStart + partialLen)` instead. The proposed phase-gating logic naturally generalizes: `phasesOverPage(c, pageStart, pageEnd=pageStart+partialLen, ...)` is well-defined for any `partialLen ∈ (0, PAGE_SLIDE]`. **No invalidation; the proposal is partial-page-friendly.**

One subtlety: a partial page that ends mid-attack-ramp depends on `startPosition` even though the full page would not. Partial-page digests will hit less often than full-page digests in this case — expected, not a bug. **Make sure Track B uses the same `phasesOverPage` predicate as full pages for consistency.**

### Track D — Radix-trie refactor

A radix trie keyed on `(token-prefix, cvec-state)` doesn't change what cvec-state needs to encode; the digest is still the disambiguation key on the cvec axis. The proposed tighter digest makes more sessions' page-edges collide on the cvec axis, which **increases** trie node-sharing — strictly beneficial for memory and lookup.

One radix-trie-specific concern: the digest is computed per page (per node depth in the trie). If the trie folds multiple pages into a single node (compressed path), each page along the path may have a different digest (e.g., sustain → release → zero). The trie must either:
- Re-split a path whenever the digest changes along it (current page-keyed approach maps directly), or
- Carry per-position cvec state on the edge (more complex but enables more sharing).

The proposed phase-gating doesn't force a choice here, but it makes "digest equals across consecutive pages" common (long sustain runs share digests page-after-page), which favors the path-compression approach. **Flag for Track D: design the trie node structure assuming consecutive-page digest equality is the common case.**

---

## Key files referenced

- `/Users/mdot/metal-microbench/lm_engine.swift:188-216` — `CvecShape`, `CvecEnvelope`, `CvecUnits`
- `/Users/mdot/metal-microbench/lm_engine.swift:228` — `CvecMode`
- `/Users/mdot/metal-microbench/lm_engine.swift:253-347` — `ActiveControl` and `magnitudeAt`
- `/Users/mdot/metal-microbench/lm_engine.swift:385-480` — `computeCvecDigest` (the audit target)
- `/Users/mdot/metal-microbench/lm_engine.swift:709-712` — `Session.cvecDigestForPage` wrapper
- `/Users/mdot/metal-microbench/lm_engine.swift:1041-1077` — `adoptSharedPrefixPages` (the consumer)
- `/Users/mdot/metal-microbench/lm_engine.swift:1818-1851` — `promoteFinishedPages` (the producer)
- `/Users/mdot/metal-microbench/page_manager.swift:185-212` — `hashPage` with `cvecDigest` xor mix
- `/Users/mdot/metal-microbench/kernels.swift:236-462` — steering kernels (where K/V is actually perturbed)
- `/Users/mdot/metal-microbench/bootstrap.swift:5036-5060` — prefill-time cvec injection site (post-FFN residual of layer L)
- `/Users/mdot/metal-microbench/ffi_batch.swift:457-459, 511-516, 528-533, 642-683` — `controlsFromDecoded` (where `startPosition`/`startTurn` get their values)
- `/Users/mdot/metal-microbench/server/bridge.py:1199-1228` — wire format for control vector installs (no `start_position` field exposed to client)
