# Dataflow pipeline: eliminate CPU inversion-of-control

## 0. Motivation

Today's pump/tick architecture is CPU-stepped: `engine.tick()` commits one
CB, calls `waitUntilCompleted`, reads back sampled logits on CPU, samples
the next token with a Swift RNG, writes it into `input_tokens`, and
commits the next CB. The ~100 ms GPU step is mirrored by a ~100 ms CPU
wait that blocks the pump thread (and, through `ffiLock`, everyone else).

There is no causal dependency in this codebase that legitimately requires
a >30 ns CPU wait. Every long wait — `cb.waitUntilCompleted()` in
`step()`/`stepPrefill*`, `pendingCB.waitUntilCompleted()` for softs,
`slot.sema.wait()` in image submit — exists only because we put the CPU
in the middle of a sequence of pure GPU operations whose causal
dependencies could all be expressed on-device via Metal queue ordering
and `MTLSharedEvent`s.

The target architecture treats the server as a dataflow graph: inputs
stream in (tokens, image bytes), GPU runs chained CBs, outputs stream
out through per-session ring buffers. CPU threads exist only to admit
new work, read session ring buffers for streaming HTTP responses, and
react to external cancel/close signals. No CPU thread blocks on GPU
completion in the hot path.


## 1. Target architecture

```
                        ┌──────────────────────────┐
  HTTP POST /chat  ──▶  │  bridge.py submit_messages│
                        │  (tokenize + chunk queue) │
                        └───────┬──────────────────┘
                                │ (chunks)
                                ▼
                        ┌──────────────────────────┐
                        │  Session.chunkQueue       │
                        │  Session.outputRing       │◀──── streamer reads
                        └───────┬──────────────────┘
                                │ (slotted this tick)
                                ▼
   Metal vision queue           Metal LM queue
   ─────────────────            ──────────────────
   cb_vis_0 ─signal event──▶   cb_lm_0 (wait event;
                               embed; 30 layers;
                               sampling kernel; writes
                               sampled_token → next
                               step's input_tokens)
                                   │ addCompletedHandler
                                   ▼
                               cb_lm_1 (already
                               enqueued; uses the
                               just-written input_tokens)
                                   │ …
                                   ▼
                               ring append: (slot, token)
                                   │ condvar.signal
                                   ▼
                               HTTP streamer wakes; reads
                               new tokens; sends SSE frame.
```

Pump thread: watches for admission work, kill signals, memory pressure.
It does NOT drive step timing. The completion-handler of each committed
CB decides whether to commit the next CB.


## 2. Component specifications

### 2.1 GPU sampling kernel — `sample_token`

**File:** `kernels.swift` (new MSL kernel).

**Purpose:** Consume per-slot logits, apply sampling parameters, write a
sampled token ID into the next step's `input_tokens` buffer.

**Inputs (buffers):**
- `logits [B, VOCAB] fp16` — output of the final unembed projection
- `sampling_params [B] struct { temperature: float, min_p: float, seed: uint32, step: uint32, active: uint32 }` — per-slot sampler config; `active==0` skips sampling for that slot (slot is idle or closed)
- `logit_bias_csr_offsets [B+1] uint32` + `logit_bias_csr_ids [total] uint32` + `logit_bias_csr_vals [total] fp16` — sparse per-slot logit bias
- `input_tokens [B] uint32` — WRITE target; sampling kernel writes the chosen token ID here for the next step

**Algorithm (per slot, one TG):**
1. Apply `logit_bias` to `logits[slot, id]` for each `(id, val)` in CSR slice
2. Divide by `temperature` (clamped at the API boundary to ≥ 0.01;
   `temperature == 0.0` is forbidden — see `bridge.py:_parse_sampling`),
   softmax, min_p prune, renormalize
3. Inverse-CDF sample via philox(seed, step)
4. Write sampled token to `input_tokens[slot]`

**Grid:** `(B, 1, 1)`, threads-per-TG = 32 (one SIMD group scans VOCAB tiles cooperatively).

**RNG:** philox-4x32-10 keyed on `(seed, step, slot)`. Used uniformly for
`T > 0` (the only allowed regime; `T = 0` is rejected at the API
boundary). The kernel retains a defensive `T <= 0` argmax branch as
dead code in case of future regressions, but no production path can
reach it.

**Why temperature=0 is forbidden** (2026-05-10): greedy/argmax
sampling skips the stochastic regime we actually deploy under, which
makes tests pass under conditions different from production. Instruments
that measure on-policy token behavior should sample from the same
distribution clients hit. Tests that previously used `temperature=0`
for bit-exactness now combine non-zero temperature with a fixed `seed`
to get reproducibility without the determinism cheat.

**Validation:**
- Unit test: feed fixed logits, T=0 → argmax token matches CPU path bit-for-bit.
- Unit test: T=1, fixed seed+step+slot → reproducible token.
- Statistical: T=1 with uniform logits, 10k samples, chi-squared against uniform expected.


### 2.2 Swift wrapper — `encSampleToken`

**File:** `bootstrap.swift` (alongside other `enc*` wrappers).

```swift
func encSampleToken(_ cb: MTLCommandBuffer,
                    logits: MTLBuffer,
                    samplingParams: MTLBuffer,
                    biasOffsets: MTLBuffer, biasIds: MTLBuffer, biasVals: MTLBuffer,
                    inputTokens: MTLBuffer,
                    vocab: Int)
```

Encodes the sampling kernel as the last dispatch in the step CB, before
the CB is committed. `inputTokens` is the same buffer that the NEXT
step's `encEmbed` reads. No barrier needed between steps — queue
ordering guarantees the next CB observes sampling's writes.

**Validation:** A/B test — run a CB that does `buildStepCB` then
`encSampleToken`, read `inputTokens` vs the token CPU-side sampling
would have produced. Assert match (at T=0) or in-distribution (T>0).


### 2.3 CB chaining — `chainStepCB`

**File:** `lm_engine.swift` (new method on `LmEngine`).

**Purpose:** Replace the pump-thread-driven `step()` with a self-
chaining CB pipeline. Each committed CB's completion handler triggers
commit of the next CB if the session still has work.

```swift
func chainStepCB(forSession s: Session) {
    guard s.state.isBusy else { return }
    let cb = buildStepCB(weights, sharedPrefixPages: detectSharedPrefix())
    encSampleToken(cb, logits: logits, ..., inputTokens: input_tokens, vocab: VOCAB)
    cb.addCompletedHandler { [weak self] _ in
        self?.handleStepComplete(sessionsSnapshot)
    }
    cb.commit()
}

private func handleStepComplete(_ snapshot: [Session]) {
    // Runs on a GCD queue after the CB completes. Reads per-slot
    // sampled_token from input_tokens (now written by GPU), appends to
    // each active session's outputRing, checks stop conditions.
    // If any session still has work, chain the next CB.
    engineLock.lock()
    for s in snapshot {
        let slot = s.slot ?? continue
        let tok = input_tokens_host[slot]
        s.outputRing.append(tok)
        s.position += 1
        if tok == s.eosId || s.numGenerated >= s.maxNewTokens {
            s.state = .done
            continue
        }
    }
    let stillBusy = residentSessions.values.filter { $0.state.isBusy }
    engineLock.unlock()
    if !stillBusy.isEmpty { chainStepCB(forSession: stillBusy[0]) }
}
```

**Key invariants:**
- `input_tokens` buffer is shared-storage; GPU writes it at end of step
  N, CPU reads it at start of step N+1's completion handler, next CB's
  embed reads from same buffer. Metal queue ordering guarantees the
  read in CB_{N+1} observes the write from CB_N.
- `handleStepComplete` runs on a background GCD queue (Metal default),
  holds `engineLock` (new, replaces `ffiLock` for engine-state access)
  only during the per-slot state mutation — microseconds, not ms.
- Session close/cancel sets a kill flag; next step's sampling kernel
  checks `active==0` → writes a sentinel token; completion handler sees
  sentinel and finalizes the session.

**Validation:** chain 16 steps starting from a fixed prompt; assert
output ring ends up with the same 16 tokens the old CPU-stepped path
produced at T=0. At T>0, in-distribution via log-likelihood against a
reference run.


### 2.4 Cross-queue event — `visionToLmEvent`

**File:** `bootstrap.swift` or `common.swift` (global Metal event).

**Purpose:** Replace `pendingCB.waitUntilCompleted()` waits in the LM
tick's prefill path with `MTLSharedEvent` cross-queue sync.

**Design:**
- Global `gVisionLmEvent: MTLSharedEvent` + monotonic signal value
  `gVisionLmEventCounter: atomic UInt64`.
- Vision queue's pad-blit CB calls `cb.encodeSignalEvent(gVisionLmEvent, value: myTicket)`.
- LM queue's prefill CB calls `cb.encodeWaitForEvent(gVisionLmEvent, value: myTicket)` before the dispatch that reads the softs buffer.
- `CachedSofts` tracks the ticket value instead of a `pendingCB`.

**Swift integration points:**
- `ensureCachedSofts` (ffi.swift): after submitting pad-blit CB, encode
  signal-event with a fresh ticket; store ticket on the `CachedSofts`.
- LM prefill wrapper: before the softs-reading dispatch, encode
  wait-event with the stored ticket.

**Validation:** bit-exact vs current path — same tokens generated,
same bit-exact softs. Concurrent stress: vision submits + LM tick
interleaved; assert no race + no CPU ever blocks on `waitUntilCompleted`.


### 2.5 Per-session output ring — `SessionOutputRing`

**File:** `lm_session.swift` (new type; replaces `outputQueue: [UInt32]`).

**Purpose:** Lock-free (or micro-lock) MPSC/SPSC queue that the
completion handler writes tokens into and the HTTP streamer reads from.

```swift
final class SessionOutputRing {
    private var tokens: [UInt32] = []
    private var position: Int = 0  // next-token read offset
    private let lock = os_unfair_lock_t.allocate(capacity: 1)
    private let condvar = DispatchCondition()
    private var finished: Bool = false

    // Called from completion handler (GCD background thread). µs critical section.
    func append(_ token: UInt32) { ... }
    func appendBatch(_ tokens: [UInt32]) { ... }
    func finish() { ... }

    // Called from HTTP streamer. Blocks on condvar if no tokens ready.
    func readAvailable() -> (tokens: [UInt32], done: Bool) { ... }
    func waitForNext(timeout: TimeInterval) -> (tokens: [UInt32], done: Bool) { ... }
}
```

**Validation:** stream a 32-token generation through the ring; assert
all 32 tokens arrive at the reader in order + `done=true` at the end.


### 2.6 Pump thread (reshaped) — `watchdogLoop`

**File:** `lm_engine.swift` (replaces `_pump_loop` in bridge.py +
`gemma_tick` FFI).

**Purpose:** Long-lived Swift thread spawned at engine init. Responsible
for:
- Periodic admission (once per ~N ms): scan pending session submissions
  and slot them into B slots.
- Cancellation dispatch: set per-session kill flag when close_session or
  abort is received; next sampling-kernel invocation observes it.
- Residency pressure: flip vision residency to `.volatile` when idle.
- Bootstrapping the chain: when the first session has work and no CB is
  in flight, call `chainStepCB` to start the pipeline.

The watchdog's cadence is self-paced (sleeps between iterations). It
never blocks on CB completion — completion handlers drive steps.

**FFI changes:**
- `gemma_tick()` FFI becomes a no-op (deprecated) OR is removed in favor
  of pump-runs-in-Swift. Bridge.py's `_pump_loop` thread is deleted.
- `gemma_start_engine()` starts the watchdog thread.
- `gemma_has_work()` reads an atomic counter, not `engine.hasWork`.


### 2.7 FFI entry changes — `ffi.swift`

- `gemma_submit_image_bytes` (already re-done in #102): unchanged.
  Still places image work into batch queue, returns immediately.
- `gemma_submit_tokens` (and friends): append chunks to session's chunk
  queue under per-session lock (µs); the watchdog picks them up on
  next admission pass.
- `gemma_poll_session` (new or repurposed): read session's output ring.
  Returns `(tokens[], done)` non-blocking. HTTP streamer loops on this.
- `gemma_close_session`: set kill flag, queue finalize work; returns
  immediately. Actual cleanup happens on the next completion handler.

Remove `ffiLock`. Replace with:
- `gSessionsLock` (fine-grained) for `gSessions` dict reads/writes.
- `engineLock` (fine-grained) for engine state transitions the pump +
  FFI both touch (session state machine, slot assignment).
- Already fine-grained: `gVisionCacheLock`, `gResidencyLock`, `gVisionBatchLock`.


## 3. Implementation order & validation gates

Each phase lands independently, behind a feature flag (`DATAFLOW_PIPELINE=1`)
so the old CPU-stepped path remains available for A/B bit-exact
validation until the new one is proven.

| Phase | Component(s) | Validation gate |
|---|---|---|
| P1 | GPU sampling kernel (2.1) + `encSampleToken` wrapper (2.2) | argmax bit-exact vs `sampleTokenFromLogits`; T>0 in-distribution |
| P2 | `input_tokens` GPU-written integration — still CPU-stepped, just sampling runs on GPU | generates same token stream (T=0) as baseline |
| P3 | Per-session output ring (2.5) + streamer read path | single-session chat works end-to-end with SSE |
| P4 | CB chaining (2.3) — two steps in flight, completion-handler-driven | 16-token generation produces correct token stream; no CPU wait in engine |
| P5 | Cross-queue vision event (2.4) | vision+chat works; softs bit-exact; no `pendingCB.waitUntilCompleted` in hot path |
| P6 | Watchdog thread (2.6) + `ffiLock` removal (2.7) | multi-session concurrent submits race-free; bandwidth probe shows expected throughput |
| P7 | Delete feature flag; remove old CPU-stepped path | final cleanup |

At each gate: rebuild, run `scripts/vision_refactor_validator.py` +
existing multi-turn chat test + bandwidth probe. Any regression
(bit-exact broken, chat garbage, throughput regression >10%) pauses
the migration until root-caused.


## 4. Open questions / risks

1. **Philox RNG vs Swift RNG.** T>0 sampling output diverges from CPU
   reference. Document as "new sampler"; validate via distribution
   statistics rather than bit-exact. Offer an env flag to force CPU
   sampling for debugging (falls back to old path).

2. **Handling slots that don't have input_tokens yet** (new session just
   slotted; its first step reads from a prompt chunk, not from previous
   step's sampled token). Answer: the first step of a session reads
   from a `first_step_token` buffer populated by admission pass; after
   that, reads from `input_tokens` (GPU-written).

3. **Cancellation latency.** When `gemma_close_session` is called
   mid-generation, the kernel needs to see the kill flag before writing
   another token. Answer: sampling kernel reads `sampling_params[slot].active`
   each step; `close_session` atomically sets it to 0; within one step
   (~100ms) the session is effectively stopped. No completion-handler
   latency.

4. **Debugging.** GPU-side control flow is harder to step through than
   CPU. Mitigation: keep the CPU-stepped path behind the feature flag
   for at least one release; add GPU-side assertion buffers for
   sanity-checking kernel behavior.

5. **Backpressure.** If the HTTP streamer is slow, the output ring
   fills up. Answer: bound the ring (e.g., 4096 tokens); sampling
   kernel checks ring availability via an atomic counter; completion
   handler stalls commit of next CB if ring is full. Back-pressures
   naturally through the pipeline.

6. **Residency pressure during chain.** Vision weights may get evicted
   under memory pressure mid-chain. Answer: watchdog pins residency
   before starting a chain involving vision; keeps pinned for the
   lifetime of active multimodal sessions.

7. **Per-slot logit_bias sparsity.** CSR format works but assumes each
   slot's bias set is small (< ~100 entries). For wider bias sets, fall
   back to dense per-slot bias (B × VOCAB × fp16 = 2 MB at B=4) or chunk
   across multiple kernel dispatches.


## 5. What gets deleted at the end

- `engine.tick()`, `engine.step()`, `stepPrefillForSession()`,
  `stepMultiSlotPrefill()`, `stepMultiSlotSoftPrefill()` (all the
  `commit(); waitUntilCompleted(); readback` shape) — replaced by
  `chainStepCB` + completion handlers.
- `ffiLock` (global recursive NSLock) — replaced by fine-grained locks.
- `_pump_loop` in bridge.py, `pump_thread` — replaced by watchdog
  running in Swift.
- `gemma_tick()` FFI — deprecated (no-op or removed).
- `sampleTokenFromLogits` — still used for reference in validator
  only; not in hot path.
- `pendingCB?.waitUntilCompleted()` at prefill readback points —
  replaced by `MTLSharedEvent` wait.
- `slot.sema.wait()` in `ensureCachedSofts` — replaced by
  CachedSofts-placeholder pattern from #105 (which becomes trivial
  once chain-CB is in place).
