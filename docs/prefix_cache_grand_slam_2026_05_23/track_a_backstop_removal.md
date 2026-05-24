# Track A — Backstop removal

> **STATUS:** Folded into Track D. Under D the recover-step lives in a
> `.primed` SessionState refinement (Track A's option c'); the
> `needsRecoverStep` flag in option (b) is not needed when D's anchor
> adoption already routes through a clean state machine. The
> correctness analysis, test plan, and "benign overwrite of shared KV"
> reasoning below all apply unchanged.

---

# Design: Eliminate the Backstop in `Session.submit()` / `revisitCacheProbe`

Audience: senior engineer who has read the original review.
Scope: read-only design — no edits.

---

## 1. Why the backstop exists (the actual invariant being protected)

**Invariant:** every `priming` session must eventually be selected by `pickChainPath()` for a path that runs `sample_token` (i.e., a CB with `skipUnembed == false`) so that the priming→generating transition (which is conditional on `chunkQueue.isEmpty` and reads `gpu_sampled_tokens[slot]`) actually fires.

That invariant is enforced today by the weaker rule **"`chunkQueue` must be non-empty when `state == priming`"** — because every `priming → generating` transition site in the engine is *only* reached as a side-effect of consuming a chunk. There is no codepath that takes a `priming` session with an empty `chunkQueue` and either (i) runs an unembed-bearing CB or (ii) transitions it. So the session sits forever in `.priming`, never reaches `.generating`, never reaches `.done`, and the bridge stream never produces a finish_reason — exactly what the comment at `/Users/mdot/metal-microbench/lm_engine.swift:967-972` predicts.

### Concrete dependency sites (the things that need a non-empty chunkQueue or a logit-bearing CB to advance a priming session)

All line numbers are in `/Users/mdot/metal-microbench/lm_engine.swift` unless noted:

- **`pickChainPath()` at 2942-2999.** For a priming session whose `chunkQueue.first` is nil, no branch fires. `nText == 0`, `nSoft == 0`, and `nAR` only increments for sessions whose `chunkQueue.first` is `nil` *and* whose state is `.generating` (because the default arm assumes either generating or 1-token-priming). Empty-queue + priming = nobody picks it.
- **`popArPrimingToken()` at 1966-1984.** Returns nil when chunkQueue empty. `prepareARStep()` at 2222-2236 then falls through to the "park: BOS at position 0" branch (line 2238), which silences the slot — no real work, the slot is just clocked through. Even if forced into AR, **the AR step would issue a forward at `s.position` for token BOS at position 0** — wrong tokens, wrong position. The post-step block at 2520-2575 *would* run sample_token (AR always unembeds), but `realSlot[slot]` is false for parked slots, so the session is skipped in the per-slot finalize loop.
- **`prepareSinglePrefill()` at 2647-2648 / `finalizeSinglePrefill` at 2876.** The transition lives at `if s.chunkQueue.isEmpty` *after* a prefill tile commits. If chunkQueue is already empty before prefill, the guard at 2648 returns nil — no CB, no transition.
- **`prepareMultiSlotPrefill()` at 3180.** Same — head-must-be-tokens guard.
- **`buildPrefillCB(..., skipUnembed:)` at 2825, 3318, 3517.** Sample is only emitted on the *last* prefill tick of a chunk-draining tile. With empty chunkQueue there is no last tick.
- **`finalizeARStep` at 2533.** The priming→generating transition for the AR path is *only* taken when `s.chunkQueue.isEmpty` *and* an AR step actually ran for that slot — i.e., we consumed a 1-token-priming tail through `popArPrimingToken`.

### What the backstop does (lines 979-988 + 1090-1104)

`submit` (979) and `revisitCacheProbe` (1090) preserve the invariant by **unadopting** the trailing slide page when adoption would otherwise leave `chunkQueue` empty. Two phys pages are decref'd (slide primary + full sibling), `position` is rewound by `PAGE_SLIDE = 16`, `promotedPageCount` decremented, and the corresponding 16 tokens go back into the prefill chunk. **No content-index entry is touched** — only this session's refs.

### Cost ledger

- **Two bit-identical 16-token prompts:** the second submits 16 tokens, adopts page 0, then the backstop sheds it (because `1 * 16 >= 16`). Net: `cache_hits = 0`, `cache_misses = 16`. This is the smoking gun reproducer.
- **General case:** any prompt whose length is an exact multiple of `PAGE_SLIDE` loses its tail page to the backstop. A 32-token prompt with both pages cached → only page 0 adopted, 16 wasted prefill tokens. A 48-token prompt with all 3 cached → only first 2 adopted, 16 wasted. The relative cost decays as prompts get longer, but the *absurd* failure mode (length exactly equal to a cached prefix) is permanent.
- **Worse than it looks:** identical prompts of length `k*PAGE_SLIDE` are common in agent workloads where the prompt is "system + few-shot exemplars + question," and the system+few-shot block is bit-identical across requests.

---

## 2. Option-by-option comparison

### Option (a) — Cache the post-prefill logit alongside the K/V

The producer's final prefill tick writes `logits[sslot, :]` into a session-side `lastPostPrefillLogit: [Float16]?`. At `promoteFinishedPages` (called at end of every prefill finalize), once the session reaches `.generating` we also publish that vector into a new index `PageManager.postPrefillLogits[contentHashOfLastPage] → [Float16]` (size VOCAB × 2 = 524288 bytes). The adopting session, when it would have been backstopped, instead:

1. consults `pageManager.postPrefillLogits[hashOfLastAdoptedPage]`,
2. if found, **runs the GPU `sample_token` kernel on that logit directly** (no embed/forward, just unembed-skip + sampling), getting the first generated token, then
3. transitions to `.generating` and joins the AR pool normally.

**Implementation surface:**

- `page_manager.swift` (~30 LoC): add `postPrefillLogits: [UInt64: MTLBuffer]` (key = same content hash as the slide-primary pair), `setPostPrefillLogit(hash:, buf:)`, `findPostPrefillLogit(hash:)`. Evict-on-pair-eviction (when `allocFresh` drops a hash, also drop the logit entry).
- `lm_engine.swift`:
  - `Session`: add `lastPostPrefillLogitBuf: MTLBuffer?` field (~5 LoC).
  - `promoteFinishedPages` (1816): after the last page promotes, if `s.state == .generating`, capture logits row → registry (~20 LoC).
  - `finalizeSinglePrefill` / `finalizeMultiSlotPrefill` / `finalizeMultiSlotSoftPrefill`: where they currently read `gpu_sampled_tokens[slot]`, also copy `logits[slot, :]` into a per-session retained buffer before chunk-drain commits (~10 LoC × 3).
  - New entry point `runSamplerOnly(s:, logitBuf:)` — encodes a tiny CB with just `populateSamplingParams` + `sample_token` dispatch on `logitBuf`, copies result to `gpu_sampled_tokens[0]`, mirrors the priming→generating block from `finalizeARStep` (~80 LoC).
  - `submit()` and `revisitCacheProbe`: replace the backstop with a probe — *don't* shed; instead set a "pending sample" flag, and let `pickChainPath` route to a new `samplerOnly(s)` path that calls `runSamplerOnly` (~30 LoC).
  - `pickChainPath` + new `ChainPath.samplerOnly(Session)` case (~15 LoC).

  Total: ~200 LoC.

**Correctness analysis:**

- **Under cvecDigest changes (Track C):** the logit is conditioned on the K/V which is digest-keyed. The producer's K/V was computed with its `activeControls`; the consumer's `activeControls` are checked at adopt-time via the hash key (different digest → no adopt). So if adopt succeeds, the K/V is identical → the *forward path* up to the logit is identical → the cached logit is correct. **However:** post-forward logit processing (logit_bias, COT mask, structured-cot, temperature, RNG seed, repetition penalty) varies per-request and is *not* reflected in the cached logit. **This is fine**, because option (a) only caches the raw `logits[slot, :]` vector (which is the model's output before sampling), and `runSamplerOnly` re-runs `populateSamplingParams` + `sample_token` on that vector with the consumer's own sampling state. Re-confirming explicitly: yes, this is correct **iff the cached vector is the raw post-unembed logit, not the post-sample anything**.
- **Steering subtlety (the parenthetical in the task):** the user worried "the steering machinery may post-process logits independently of K/V." Scanning `populateSamplingParams` and the AR finalize path, the only logit-time interventions are `logitBiasDense`, COT mask, and the GPU sampler's temperature/min_p/seed/step/active. None of these change the *unembed* output, only the sampling distribution applied to it. The cvec injections happen at the residual stream during layer forward, so they're baked into the K/V (and hence into the next-step logit produced *from* that K/V via unembed of hidden state — wait, the logit at the post-prefill moment is computed by the *current* prefill CB from the current hidden, not from K/V. The cached logit therefore reflects the producer's hidden at the moment of unembed, which depends on the producer's full layer stack including its cvec injections.) **Conclusion:** since the producer's activeControls are baked into the page's content hash (Session.cvecDigestForPage line 709-712), and adopt only succeeds on hash match, the consumer's cvecs == producer's cvecs over the adopted range, so the logit is identical to what the consumer would have produced. ✓
- **Under partial-page adoption (Track B):** if Track B lets a session adopt 7/16 tokens of the last page, we need the *logit at position k*, not the logit at the page boundary. Option (a) only stores the page-boundary logit. Track B work would have to either (i) extend the index to store per-position logits (16× the memory) or (ii) defer to option (b) for sub-page adoption. **Risk: option (a) is invalidated for sub-page adoption.**
- **Under abnormal termination:** if the producer terminates before reaching the last page boundary, the logit was never captured → consumer falls back to running a real prefill (the backstop's behavior). Safe.

**Memory cost:** 524288 bytes per cached page that has a logit entry. The cache will only ever populate entries for pages where the producer transitioned `priming → generating` exactly at that page boundary — i.e., the last page of a prompt whose length is a multiple of PAGE_SLIDE. In practice few cached pages will have logits attached; the rest will have nil entries. Worst case (every cached page has a logit): at the current 8192-page pool that's 8192 × 512KB = 4GB. **Need a separate small LRU pool of, say, 256 cached logits = 128MB**, evicted independently of the K/V page cache. So this is one more cache to budget, not a free lunch.

**Risk surface:** introduces a new scheduler path (`samplerOnly`), a new CB kind, a new index. Forces the scheduler's `pickChainPath` to honor a 4th case. Risks: (i) the sampler-only CB still has to set `block_table[0]` to scratch (otherwise reads silenced slots' KV — but the sampler doesn't read KV at all, only `logits` + `sampling_logit_bias`, so might be encodable as a 1-dispatch CB with no per-slot setup beyond the sampling params). (ii) Race on `lastPostPrefillLogitBuf`: producer's prefill CB and consumer's eventual adoption could race if both run in the same tick; the producer's blit-into-session-buffer is on the LM queue and queue-ordered before any consumer's adopt, so an `MTLEvent` is not needed but the *retention* of the buffer matters (consumer must incref/decref via the same anonymous-pool mechanism PageManager already provides — best to use a small dedicated buffer pool indexed by hash).

**Migration plan:** ships incrementally behind a env-var feature flag (`LM_POST_PREFILL_LOGIT_CACHE`). When off, fall through to current backstop. Can be enabled per-deploy and A/B'd.

**Accounting effect on cache_hits / cache_misses:** the consumer correctly counts every adopted token (including the previously-backstopped last page) as a hit. The 16 tokens that previously went into the prefill chunk as misses are now zero. Net: smoking-gun reproducer goes from `hits=0,misses=16` to `hits=16,misses=0`. **The semantic is correct** — we genuinely did skip prefill on those tokens. No retroactive bookkeeping needed (no equivalent of revisitCacheProbe's miss→hit swap).

---

### Option (b) — Run a 1-step AR tick at the adopted position

Instead of caching the logit, we run the *forward pass* at the adopted position to produce the first sampled token. This is exactly what `prepareARStep` does for a `.generating` session — it issues a 1-token forward at `s.position` using `s.nextGeneratedInput` as the input token.

The hitch: at the adopted position, what is the "input token"? It's the token at `s.position` *in the consumed history*, which is `consumedTokens[s.position - 1]` — the last token of the prompt. We forward that token through the model with the cached K/V (read-only from positions `[0, s.position-1]`, write at position `s.position - 1` — wait, but position `s.position - 1` is *already* filled by the producer's prefill into the shared page, so we'd overwrite it on a shared page → CoW issue).

Re-examining: at `s.position == 32` (2 pages adopted), the AR step at position 32 reads K/V for `[0, 31]` (all cached, in shared pages) and writes K/V for position 32 (which is in *page 2*, a new page this session would allocate fresh). So no shared-page write. **But there's no input token at position 32** — position 32 is the first *generated* position. The input token for an AR step is the prompt's last token, *which is already at position 31 in the cached K/V*. To produce a logit *at* position 31 (which is what an AR step does — it computes hidden at position k from input k and KV[0,k-1], then unembeds), we need to forward the input token for position 31, which is `consumedTokens[31]`.

So: **the AR tick is at position 31, not 32**. Inputs: `tok = consumedTokens[31]`, `pos = 31`. KV reads `[0, 30]` (all in cached pages, hashed correctly per Track C). **KV writes at position 31** — which is also in the cached page (page 1, indices 16-31). That's the CoW problem: the cached page is shared (refcount ≥ 2) and the engine cannot safely overwrite K/V[31].

But wait — the cached K/V at position 31 was *already written by the producer's prefill*, with the *same* token at position 31, same upstream KV, same cvecs (digest match). The kernel would compute byte-identical K/V at position 31 and overwrite with the same bytes. **Functionally a no-op write.** PageManager forbids it because the *general* rule is "shared pages are read-only"; but in this specific case the write is provably idempotent.

The cleaner fix is to **skip the K/V write entirely on this tick** and only run the unembed branch — i.e., a CB variant that does forward through layers but does *not* commit K/V writes. The existing prefill kernels can't do this without modification (`kv_write` is encoded into the layer loop). Easier: forward the same token at position 31, knowing the K/V write produces identical bytes; **don't take the CoW penalty because we know it's a no-op**.

**Implementation surface:**

- `lm_engine.swift`:
  - `pickChainPath`: a session with `state == priming`, `chunkQueue.isEmpty`, and `position > 0` is routed to a new `arRecoverPath(Session)` case (~15 LoC).
  - New `prepareARRecoverStep(s)` / `finalizeARRecoverStep(p)`: builds a 1-slot CB at `position = s.position - 1`, input token = `consumedTokens[s.position - 1]`, klen = `s.position`. Reuses `buildStepCB` machinery with `activeB = 1` and silences other slots via `arMapping[slot] != nil` check. After CB completes, runs the same transition block as `finalizeARStep` at 2533-2548 (~150 LoC).
  - `submit()` / `revisitCacheProbe`: delete the backstop (lines 979-988 + 1090-1104) (~25 LoC removed). Add a post-adopt check that sets a session-side flag `needsRecoverStep = true` (~10 LoC). When `position % PAGE_SLIDE == 0` (a clean page boundary) AND `chunkQueue.isEmpty` AND `position > 0`, that's the signal.

  Total: ~200 LoC (similar to option a, but fewer cross-file touches — no PageManager work).

**Correctness analysis:**

- **Under cvecDigest:** the recover tick re-runs the forward, using the consumer's own `activeControls`. The K/V read from cached pages was produced under the *producer's* cvecs, but since cvecDigest match was required for adopt, those cvecs are equivalent. The new forward at position k-1 reads those K/V → produces hidden at k-1 → unembeds → samples. Identical to what the consumer would have produced from scratch. ✓
- **Under partial-page adoption (Track B):** trivially handled. Whatever `position` ends up at after partial adoption, the recover tick runs at `position - 1` with `consumedTokens[position - 1]` as input. No K/V boundary assumption. **Track B compatibility: clean.**
- **Under abnormal termination:** no producer state to lose. Pure consumer-side recovery; no producer cooperation needed.
- **CoW (kv_write at the recovered position):** the K/V write at `position - 1` lands in a shared page that the consumer has incref'd. The write produces bit-identical bytes (same input, same KV[0,k-2], same cvecs). The PageManager doesn't have CoW logic and would not detect the overlap. **The write goes through and is benign.** Slightly cursed (the engine *appears* to violate the "shared pages are read-only" invariant), but the violation is provably a no-op. Add a code comment + an assertion that the K/V bytes match post-write (gated by an env var; `LM_KV_OVERWRITE_CHECK`) so any future divergence is caught early.

**Memory cost:** zero per-cached-entry. One extra `bool needsRecoverStep` per session (~1 byte).

**Risk surface:** smaller than option (a). No new index, no new buffer pool, no logit-cache eviction policy. The "benign overwrite" of shared KV is the one wrinkle, and it's mechanically sound. The new scheduler path (`arRecoverPath`) is a stripped-down clone of `arStep` that only touches one slot.

**Migration plan:** ships incrementally behind `LM_BACKSTOP_RECOVER` flag. Off = current backstop. Easy to A/B.

**Accounting effect on cache_hits / cache_misses:** same as (a) — adopted page is counted as hit; the recover tick is a 1-token forward that is *not* counted as a miss (it's the "sample-the-first-token" step, equivalent to the first generated token's AR step, not prefill). Same smoking-gun result: `hits=16, misses=0`. The recover tick does cost 1 CB of GPU time (~34ms per AR step at current numbers), so the throughput win is "skipped prefill of N tokens" minus "one AR tick" — net positive for N ≥ 2.

---

### Option (c) — **Producer-side reservation** (my proposed third option)

Inversion of the backstop: instead of the consumer un-adopting the last page, the producer **never promotes the last page** until it has emitted at least one generated token. Promotion of page P requires `s.position > (P+1) * PAGE_SLIDE`, not `s.position >= (P+1) * PAGE_SLIDE`.

Examine `promoteFinishedPages` at 1816-1852: today the loop runs while `s.promotedPageCount < fullyWritten` where `fullyWritten = s.position / PAGE_SLIDE`. Change it to `fullyWritten = (s.position - 1) / PAGE_SLIDE` (or equivalently, only promote pages where the *next* position has been generated).

**Why this is wrong:** the producer doesn't know it's the producer. It promotes pages as it goes. If we hold back the last page until "at least one generated token," then:
- a producer that emits exactly one token, hits EOS at position `(P+1)*PAGE_SLIDE + 1`, will have promoted all pages because `(s.position - 1) / PAGE_SLIDE = (P+1)*PAGE_SLIDE / PAGE_SLIDE = P+1 ≥ P+1`. Wait, we need `s.position - 1 > (P+1)*PAGE_SLIDE`, i.e., `s.position > (P+1)*PAGE_SLIDE + 1`. So the producer has to emit ≥ 2 generated tokens for the last page to promote.
- A producer that emits 1 token then EOS never promotes its last page. **The consumer permanently misses on that prefix.** That's also wrong — it just relocates the absurd failure mode.

**Verdict on (c):** rejected. The producer can't know whether a future consumer will arrive with a prompt exactly matching the producer's last page. The right place to fix this is the consumer, where the empty-queue condition actually arises.

---

### Option (c'), a better third option — **First-class "primed" state**

Add a 4th `SessionState`: `.primed`, meaning "all prefill complete, K/V at position N exists, but no first generated token has been sampled." Then:

- `prepareARStep` admits `.primed` sessions and treats them like `.generating`: pulls `consumedTokens[position-1]` as the input token (instead of `nextGeneratedInput`), runs a 1-token forward at `position-1`, samples, transitions `.primed → .generating` on success.

This is essentially option (b) but expressed as a state-machine refinement instead of a new scheduler path. The CoW caveat is the same. The code surface is slightly larger because `state` is read in many places (~25 sites by `grep state ==` and `state.isBusy` etc.) and each needs to be audited for what to do with `.primed`. But it composes more cleanly with the existing scheduler and removes the need for a dedicated `arRecoverPath`.

**Verdict on (c'):** structurally cleaner than (b) but more touch sites. I think it's the right *long-term* shape, but option (b) is a strictly easier on-ramp that can be refactored into (c') later. (c') is what we should land if Track E's rename pass is happening anyway — folding in a state refinement is cheap mid-rename.

---

## 3. Recommendation

**Pick option (b)**, with a TODO to refactor into option (c') after Track E lands.

- **Smallest correctness surface.** No new index, no eviction policy, no new buffer pool, no producer-consumer coordination. Just a 1-slot AR tick at `position - 1` whose K/V "overwrite" is provably a no-op.
- **Track B compatible.** Sub-page adoption works trivially because we run at `position - 1` regardless of page boundaries.
- **Track C compatible.** cvecDigest already gates adoption; the recover tick uses the consumer's own cvecs, which (by adopt invariant) are equivalent to the producer's over the adopted range.
- **Track D compatible.** A token-granularity radix lookup will hand us a `position` (not a page count); option (b) operates on `position`. Option (a) is tied to *page-boundary* logits and would need redesign for token-granularity.
- **Track E neutral.** Naming changes don't affect the recover-step code.
- **Performance:** trades one AR tick (~34ms) for skipping `N - 1` prefill tokens (~1-2ms/token in single-slot prefill). Net positive for any prompt ≥ 17 tokens.

**The one wrinkle to call out in code review:** the benign overwrite of shared KV. Defend it with a code comment + an `LM_KV_OVERWRITE_CHECK` debug assertion that rehashes the page post-tick and compares to the pre-tick content hash. Cheap to leave in for the first month after ship.

---

## 4. Test plan

The smoking-gun reproducer:
- **R1.** Two identical 16-token prompts, sequentially: `submit([t0..t15])`, drain to done. Submit `[t0..t15]` again, drain to done. Assert second submission's `cacheHitTokens == 16`, `cacheMissTokens == 0`, and the second submission produces a real generated token sequence (not stuck in priming).

Additional tests:
- **R2.** Two identical 32-token prompts: assert second has `hits=32, misses=0`. (Today: backstop would leave `hits=16, misses=16`.)
- **R3.** Sequential prompts at all page-multiple lengths {16, 32, 48, 64, 128, 256}: assert all hits, zero misses on the second submission.
- **R4.** Sub-page-boundary length (33 tokens): assert second has `hits=32` (2 pages adopted), `misses=1` (the 33rd token goes through normal prefill). Verifies the recover-step doesn't trigger for non-page-aligned prompts.
- **R5.** Triple submission stress: 3 sequential identical 16-token prompts. Second + third both report `hits=16, misses=0`. Verifies cached page survives the consumer's lifecycle and that the "shared KV overwrite" really is a no-op (the third probe still finds the same hash, which would not be true if the second consumer corrupted bytes).
- **R6.** cvecDigest mismatch: producer with control X, consumer with control Y, same tokens. Assert consumer adopts 0 pages (digest mismatch) and runs full prefill — recover-step does NOT fire. Verifies the digest gate still works.
- **R7.** Output equivalence: the first-generated-token distribution from a recover-step path must equal the first-generated-token distribution from a fresh-prefill path. Run 100 fresh-prefill samples (no cache) with a fixed seed; run 100 adopt+recover samples; assert the token-frequency histograms match within sampling noise (or, at temperature=0 with seed-fixed greedy via the temperature floor of 0.01, assert exact match modulo the 0.01 floor — the GPU sampler uses the same RNG seed for both paths).
- **R8.** EOS-as-first-token: producer's first generated token is EOS; consumer adopts; consumer's recover-step must also emit a token and (probably) hit EOS too (assuming deterministic same seed) → `state = .done`. Verifies the recover path's transition block handles immediate EOS correctly.
- **R9.** Stop-sequence first match: same as R8 but with a `stopSequences` match. Verifies the stop_sequence check fires in the recover-step's finalize.
- **R10.** Backstop deletion regression: the env var `LM_BACKSTOP_RECOVER=0` (legacy mode) still passes the *original* test suite (so the new path doesn't break the old path).
- **R11.** Page-pool refcount accounting: after R5, run `pageManager.stats()`, assert `pagesInUse == 0` (no leak). Verifies the recover-step's finalize correctly decrefs.
- **R12.** Multi-modal: a soft-token + text-tokens prompt where the entire prompt is cached. Recover-step fires after `revisitCacheProbe` on the text segment. Verifies the recover-step works at the `revisitCacheProbe` site too, not just `submit`.

---

## 5. Interactions with other tracks

- **Track B (partial-page promotion at teardown).** Option (b) is robust: partial-page adoption produces a `position` not on a `PAGE_SLIDE` boundary, but the recover tick runs at `position - 1` regardless. Worth confirming with Track B owner that *their* adoption code calls into the same `submit`/`revisitCacheProbe` path so the new `needsRecoverStep` flag gets set. If Track B introduces a *different* adoption entry point, we have to mirror the flag-set there.

- **Track C (cvecDigest tightening).** Option (b) inherits Track C's correctness: digest match is the precondition for adoption, recover step reuses adopted K/V. **If Track C tightens to include things the digest currently misses** (e.g., position-conditional cvec evaluations on the page-tail tokens), option (b) is unaffected because it doesn't add any new digest assumption. **Flag:** if Track C decides cvecDigest should additionally cover *future* positions (e.g., the cvec that will be active at `position`), it could invalidate adoption of the trailing page even when token-history matches. That would *increase* the rate of misses but not break option (b)'s correctness.

- **Track D (token-granularity radix lookup).** Option (b) is **strictly preferable to option (a) under Track D.** Token-granularity lookup means adoption can end at any `position`, not just `PAGE_SLIDE` multiples. Option (a)'s page-keyed logit cache cannot serve sub-page adoptions; option (b) works at any position. **Flag:** Track D should plan for the `needsRecoverStep` flag to be set whenever adoption leaves an empty chunkQueue, regardless of granularity. The condition is `position > 0 && chunkQueue.isEmpty && state == .priming`, not "page-aligned position."

- **Track E (naming rename pass).** Option (b) adds: `Session.needsRecoverStep`, `ChainPath.arRecoverStep(Session)`, `prepareARRecoverStep`, `finalizeARRecoverStep`. **Flag:** these names should be chosen with Track E's rename conventions in mind. If "recover" doesn't fit Track E's vocabulary, suggest alternatives: `arResumeStep`, `samplePrimedSession`, `firstSampleAfterAdopt`. Recommend coordinating naming with Track E author before merge to avoid a follow-up rename. Better still: if Track E is also doing the `.primed` state refinement that (c') describes, fold this work in and skip the `needsRecoverStep` flag entirely — the state itself encodes the condition.

---

### Critical Files for Implementation

- /Users/mdot/metal-microbench/lm_engine.swift
- /Users/mdot/metal-microbench/page_manager.swift
- /Users/mdot/metal-microbench/ffi_batch.swift
- /Users/mdot/metal-microbench/bootstrap.swift
- /Users/mdot/metal-microbench/server/bridge.py
