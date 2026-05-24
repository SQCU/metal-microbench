# Prefix-cache grand slam — 2026-05-23

## Operator doctrine

> i think D is simply higher priority work, and is only weeks long in
> metr-time-horizon, which presumes all sorts of coordination and
> cognitive boundaries not present in this user-claude<->subclaude
> multipolar code writing and code review format! it's pretty much not
> worth writing code which isn't at least as bold as D, inverting
> several baseline intuitions about modesty and time value of writing
> code

> this is all main code, which we blast unrelentingly to the actual
> runtime, which we restart and sync to test the current working code
> as needed, persisting those design docs ofc, and like, yeah. we can
> resume the 'why the sillytavern slow???' case study *after* getting
> all of these yummy granular prefix matching correctness and speedup
> wins.

The METR-style 3-week sizing in the Track D doc presumes PR review
queues, alignment meetings, onboarding overhead, async standups,
branch-pollination concerns — all artifacts of a multi-human shipping
process that don't apply when integration is one agent and parallel
implementation is N subagents. In this format the ~1000-LoC refactor
is hours of wall-clock + thousands of tokens, not weeks.

A/B/C/E in isolation would produce code that D throws away
(`PageManager.contentIndex`, `findByHash`, the `hashPagePrefix`
wrapper, the backstop's bookkeeping arithmetic, `cvecDigest: UInt64`
as a primitive). Doing them first pays the rename cost twice.

## The carrier-and-folded-in structure

**Track D (radix trie) is the carrier refactor.** Tracks A/B/C/E are
folded in to its shape:

| Track | Original separate shape | Folded form inside D |
|---|---|---|
| **A** — backstop removal | `Session.needsRecoverStep` flag + `arRecoverPath` scheduler case | `.primed` SessionState refinement (Track A's option c') + recover-tick at any adopted position. D's anchors expose the fully-cached case more often, so A becomes a correctness requirement of D, not a separate fix. |
| **B** — partial-page promotion | hybrid (c) 8-token sub-granularity + (a) flush-prefill | `TrieAnchor.partialTail: PartialPagePair?` + CoW-on-extend at anchor (D's design reserves this field). True 1-token granularity, not the 8-token floor of standalone (c). |
| **C** — cvecDigest tightening | UInt64 digest with phase-gates added to `computeCvecDigest` | `CvecAnchorTag` struct stored at each anchor (per Track D §6 "(ii) Per-page-pair cvec partition at anchors"). Units-gated, phase-gated, quantized-floats — all per Track C's audit, but living in the trie shape from day one. |
| **E** — renames | Three-PR full Track E plan (~250 LoC) | Only the renames on **surviving code**: `chunkQueue → primingQueue`, `promotedPageCount → myPromotedSlidePairCount`, `pendingPrimingCount → pendingPrimingTokenCount`, `SharedPagePair → SlidePairContents`. **Skip** renames on dict/findByHash/promotePair/hashPagePrefix/adoptSharedPrefixPages/promoteFinishedPages/revisitCacheProbe — these are deleted or signature-replaced by D. |

## Execution order

```
1. Test harness (gating)
   ├─ Build comprehensive failure-mode test harness
   ├─ Validate it FAILS on current code (proves discriminating power)
   └─ Locks the smoking-gun reproducers as regression tests

2. Implementation fan-out (parallel — 4 subagents on main):
   ├─ subagent 1 — radix_trie.swift (new file, ~600 LoC)
   ├─ subagent 2 — lm_engine.swift: .primed state + recover-step + lookup-call-site rewrite
   ├─ subagent 3 — page_manager.swift refactor: drop dict, eviction-callback, partial-pair support
   └─ subagent 4 — CvecAnchorTag impl + tighter digest semantics

3. Integration (main agent)
   ├─ Merge edits, resolve cross-touchpoint conflicts
   ├─ Build, run test harness, iterate via targeted subagent edits until green
   └─ Bridge restart + smoke test against live runtime

4. Surviving-code renames (1 subagent)
   └─ Apply Track E's surviving-code renames across the integrated branch

5. Full validation
   ├─ Full existing test suite (LM_TEST_CVEC_DIGEST, LM_TEST_CACHE_DIVERGENCE,
   │  LM_TEST_CVEC_CACHE per docs/CVEC_AND_PREFIX_CACHE.md)
   ├─ End-to-end ST test against the live bridge
   └─ Cold-prefill 107 tok/sec investigation (task #216) runs naturally here
```

## Bridge handling

Per operator directive: bridge is mine to kill, relaunch, and validate
after each integration step.

- Kill: `pkill -KILL -f 'serve.py'` (per memory: never `kill <uv_run_pid>` alone)
- Relaunch: `nohup /Users/mdot/metal-microbench/server/.venv/bin/python /Users/mdot/metal-microbench/server/serve.py > /tmp/bridge_serve.log 2>&1 &`
- Verify: `curl -fsS http://127.0.0.1:8001/health | jq -e '.status == "ready"'`
- Smoke: 2 identical 16-token curls → second should report `cache_hits=16`

## Risk acknowledgment

D is a wholesale rewrite of the cache's lookup semantics. The test
harness mitigates this but doesn't eliminate it — a sufficiently
subtle correctness bug could survive even thorough tests. The harness
includes a KL-divergence guard against fresh-compute. If KL stays
< 1e-5 across a representative prompt corpus, high confidence.

## Files in this directory

- `track_a_backstop_removal.md` — Track A subagent's full design doc
- `track_b_partial_page_promotion.md` — Track B subagent's full design doc
- `track_c_cvec_digest_tightening.md` — Track C subagent's full audit report
- `track_d_radix_trie.md` — Track D subagent's full design doc (the carrier)
- `track_e_naming_renames.md` — Track E subagent's full rename plan
