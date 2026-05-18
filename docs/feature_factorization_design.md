# Feature factorization: linear algebra, pseudohaskell, ball-and-stick

The user-agent harness is, at heart, a **fixed-point iteration over two
nested functors** (bio designer outside, user-agent designer inside),
**driven by an LLM-as-judge** scoring sparse projections of a wide
behavioral feature space. This note pins down the structure so we can
talk about what's built, what's missing, and what to build next without
re-deriving the spec each session.

See also: `user_agent_factorization_spec.md` (canonical vocabulary &
schema), `user_agent_workshop_harness_design.md` (the DESIGNER/PERFORMER/
JUDGE loop motif from the discovery harness), `discovery_harness_findings.md`
(prior entanglement findings, e.g. `provocative`).

## 1. Linear algebra

Let X ⊆ R^N be the **total measurable behavioral feature space**.
Axes are 1–5 Likert scalars with rubrics; N is the registry size,
realistically ~10² growing to ~10³.

Two action subspaces:

- **A ⊆ X** — agent-controllable. Axes that a depth-1 author's-note
  user-agent overlay can move from one bio. Dispositional / move-set:
  `theft_aggressiveness`, `romantic_advance`, `confrontation_style`,
  `risk_tolerance`, …
- **B ⊆ X** — bio-controllable. Axes that bio prose can move.
  Identity / register: `astrology_sagittarian`, `vocabulary_register`,
  `voice_warmth`, … and (typically) `B ⊇ A` since bios also condition
  the agent-controllable axes (witness: the Rogue-Cancer bio caps
  reachable `theft_aggressiveness` to 3.5 unless paired with high
  `romantic_advance`).

The measurement operator m is **sparse**:
```
  m : Chat × IndexSubset_k → R^k    (k ≪ N, k ≤ ~14)
```
i.e. a judge call asks about k axes at a time, not all N. The infra
for this is the sparse extended-schema judge (#121).

**Inner fixed point** (built). Given `bio ∈ B`, target `t_A ∈ A`,
counterparty `cp`:
```
  F_A(t_A, bio, cp) : agent_text ∈ overlay-space such that
    m(chat(bio, agent_text, cp)) |_A  ≈  t_A
```
Iteration: pass measured signature back to the agent-designer LLM so it
rewrites `agent_text`. Converges (small ε) or hits K_max.

**Outer fixed point** (built). Given target `t_B ∈ B`, fixed agent
targets `{t_A^i}`, cp:
```
  F_B(t_B, {t_A^i}, cp) : bio_prose such that
    aggregate_i m(chat(bio_prose, F_A(t_A^i, bio_prose, cp), cp)) |_B  ≈  t_B
```
The aggregation projects all per-turn measurements over all inner runs'
best agents onto the B-axes and means them.

**Outer-outer selector** (MISSING). The vision:
```
  S(corpus, registry) : (t_B, {t_A^i})
    such that running F_B with these targets is expected to
    INCREASE pca_eff_dim(corpus ∪ {new_bio})
```
Plus a parallel **counterparty-variation** selector so converged bios
are validated transferable across multiple `cp`.

**Axis splitter** (MISSING). Diagnostic: for axis `a ∈ X`, if
`E[a | context_X] ≠ E[a | context_Y]` across trajectories where the
designer-LLM hit different ceilings on `a`, propose splitting
`a → {a_X, a_Y}`. Re-judge the historical trajectories under the
proposed pair and accept the split iff separability exceeds a threshold.

## 2. Pseudohaskell

```haskell
-- ── types ────────────────────────────────────────────────────────────
type AxisName     = Text
type Axis         = (AxisName, Rubric)
type Sig          = Map AxisName (Maybe Int)        -- sparse, 1..5
type Bio          = Text                            -- prose
type AgentText    = Text                            -- depth-1 overlay
type Chat         = [Turn]
type Counterparty = Character
type Target       = Map AxisName Int

data Attempt a = Attempt
  { artifact :: a              -- Bio or AgentText
  , sig      :: Sig
  , maxOff   :: Double
  , chat     :: Maybe Chat     -- inner: present; outer: aggregated
  }

-- ── atomic ops (all built) ────────────────────────────────────────────
runChat   :: Bio -> AgentText -> Counterparty -> Int -> IO Chat
judge     :: [Axis] -> Turn -> IO Sig                 -- sparse, k axes
aggregate :: [Sig] -> Sig                              -- per-axis mean

-- ── inner: agent designer (BUILT in lock_in_iterative.mjs) ───────────
designAgent :: Bio -> Target -> Counterparty -> IO [Attempt AgentText]
designAgent bio tA cp = fixUntil (converged tA `or` velocityStalled) $ \prior -> do
  txt   <- designerLLM_A bio tA prior          -- LLM sees prior measurements
  chat  <- runChat bio txt cp nTurns
  sig   <- aggregate <$> mapM (judge (axesOf tA)) (userTurns chat)
  pure $ Attempt txt sig (maxOff sig tA) (Just chat)

-- ── outer: bio designer (BUILT) ───────────────────────────────────────
designBio :: Target -> [Target] -> Counterparty -> IO [Attempt Bio]
designBio tB tAs cp = fixUntil (converged tB `or` velocityStalled) $ \prior -> do
  bio    <- designerLLM_B tB prior
  inners <- mapM (\tA -> designAgent bio tA cp) tAs
  bioSig <- aggregate <$> mapM (judge (axesOf tB))
                                (concatMap (userTurns . chat . best) inners)
  pure $ Attempt bio bioSig (maxOff bioSig tB) Nothing

-- ── outer-outer: target selector (MISSING) ────────────────────────────
exploreCorpus :: Registry -> Counterparty -> IO ()  -- forever
exploreCorpus reg cp = forever $ do
  corpus     <- loadCorpus
  (tB, tAs)  <- selectNextTargets reg corpus EffDimMaximizer
  attempts   <- designBio tB tAs cp
  appendCorpus (best attempts)

-- ── axis splitter (MISSING) ───────────────────────────────────────────
attemptSplit :: AxisName -> [Chat] -> IO SplitResult
attemptSplit a trajectories = do
  -- LLM proposes 2-3 candidate splits given the entangled trajectories
  hyps <- proposeSplits a trajectories
  -- Re-judge under each candidate (axis pair) and measure separation
  scored <- forM hyps $ \(a1, a2) -> do
    sigs <- mapM (judge [a1, a2] . everyTurn) trajectories
    pure (a1, a2, separability sigs)
  case filter (passesThreshold . sep) scored of
    []       -> pure (NoSplitFound { tried = hyps })
    (h:rest) -> pure (Split { winning = h, alsoRan = rest })
```

The two existing layers compose as `designBio = fix outerStep` where each
`outerStep` invokes `designAgent = fix innerStep`. The pending pieces
(`selectNextTargets`, `attemptSplit`) sit *outside* that pair without
modifying it.

## 3. Ball-and-stick

```
         ┌──────── OUTER-OUTER (MISSING — items 1,2,3,4) ──────────┐
         │  selectNextTargets : (Registry, Corpus, Objective)      │
         │      → (t_B ∈ B, [t_A ∈ A])                             │
         │  objective ∈ { eff-dim-maxx, decorrelate-from-cloud,    │
         │                resolve-entanglement }                   │
         │  consume entanglement signals → axis-splitter           │
         └──────────────────────┬──────────────────────────────────┘
                                ↓ proposes targets
         ┌──── OUTER (BUILT in lock_in_iterative) ────┐
         │  bio designer ⇄ aggregated measurements    │
         │  velocity-stall stop (item 5)              │
         └──────────────────────┬─────────────────────┘
                                ↓ proposes bio prose
         ┌──── INNER (BUILT) ────┐
         │ agent designer ⇄      │
         │ per-turn measurements │
         └──────────┬────────────┘
                    ↓ runs chat
                ┌── CHAT (BUILT) ──┐
                │ bio+agent vs CP  │←── multi-CP via /transfer-test (easy)
                └────────┬─────────┘
                         ↓ trajectory
                ┌── JUDGE (BUILT) ──┐
                │ sparse-sample k    │←── axis controller (item 2)
                │ from registry      │←── axis registry  (item 1)
                │ of ~10²–10³ axes   │
                └────────────────────┘
```

## 4. Status checklist

### Built

- Bridge / engine: prefix-maxx, KV-share, parallel judge fan-out at honest
  bandwidth (#143, #149, #156).
- Schemas: bio-v2, agent-v1, author's-note injection (#112, #113).
- Sparse extended judge schema: core 14 axes + opt-in extended + free
  observations (#121). The *protocol* for sparse sampling exists.
- Merged two-stage judge cascade with turn-text cross-check (#149, #122, #123).
- One-shot PCA effective-dimensionality measurement on bio corpus (#69).
- `/transfer-test` cross-counterparty cardability scan (#137).
- Strategy-diversity LLM-as-summarizer (#78); drift analysis (#107).
- Designer.html, suggester.html UI surfaces; /compare-agents (#132);
  derived_from (#130); card-quality badges (#131).
- **Nested fixed-point demonstrated end-to-end**: `lock_in_iterative.mjs`
  on 2 bios × 2 agent targets × 4 experiment axes. Rogue-Cancer surfaced
  internal entanglement on `theft_aggressiveness`.

### Missing — IRREDUCIBLE gap to the ~1k-axis vision

| #   | Piece                              | Est.  | Status                                            |
| --- | ---------------------------------- | ----- | ------------------------------------------------- |
| 1   | Axis registry                      | 1d    | starter shipped — `tools/user-agent-harness/axis_registry.mjs` (22 axes seeded, derived-axis storage) |
| 2   | Sparse-sampling controller         | 2d    | seed in axis_registry (`pickSubset` with recency); needs caller integration |
| 3   | Outer-outer w/ eff-dim objective   | 2-3d  | not yet started                                   |
| 4   | Entanglement detector + splitter   | 3d    | starter shipped — `tools/user-agent-harness/axis_splitter.mjs`; first run NO_SPLIT_FOUND (§5) |
| 5   | Velocity-stall convergence         | 30min | shipped in lock_in_iterative.mjs (`isStalled`)    |

### Missing — easy, just-not-composed

- Multi-counterparty validation in the outer loop (calls existing /transfer-test).
- Counterparty variation as an additional design axis.
- Persistence for 1M-scale (durable run store, restart-safe job queue, dedup).

## 5. First splitter run — Rogue-Cancer × theft_aggressiveness

The 2-bio iterative run surfaced the entanglement candidate
`theft_aggressiveness × astrology_cancerian` in the Rogue-Cancer bio
(see §1 and lock_in_iterative.mjs output). Same parent axis, divergent
measurements across chat contexts (`steals` vs `romances-and-steals`).
We fed the trajectory to `axis_splitter.mjs` to demonstrate the
entanglement-detection → split-hypothesis → re-judge → accept-or-reject
leg of the loop end-to-end.

### Acceptance criteria (encoded in the splitter)

A proposed split is accepted only if, on the historical turns:

1. **Sign-recovery**: the winning sub-axis's Cohen's d between contexts
   has the SAME SIGN as the parent's Cohen's d. (Otherwise the split has
   measured something orthogonal — possibly real, but not a factorization
   of the parent's entanglement.)
2. **Magnitude-recovery**: the winning sub-axis's |d| MEETS OR EXCEEDS
   the parent's own |d| on the same per-turn data. (The split must
   improve discrimination, not just re-score it.)
3. **Threshold**: the qualified sub-axis |d| ≥ 0.8 (large effect).

### Result: NO_SPLIT_FOUND (run at 2026-05-18T04:44Z)

Parent on the bucketed turns:
- `steals` context (n=10): mean theft_aggressiveness = 3.70
- `romances-and-steals` context (n=6): mean = 4.50
- parent Cohen's d = −0.88 (steals < r-and-s; |d| = 0.88)

The DESIGNER_S proposed three hypotheses:

| H  | Sub-axes                                          | Rationale (LLM)                                                                         |
|----|---------------------------------------------------|-----------------------------------------------------------------------------------------|
| H1 | `material_intent` / `tactical_execution`          | steals = purposeful theft; r-and-s = high intent + erratic execution                    |
| H2 | `scavenging_instinct` / `opportunistic_impulse`   | methodical scavenging vs emotional snatching during social roleplay                     |
| H3 | `search_intensity` / `theft_audacity`             | r-and-s = emotional desperation (high search); steals = stealthy professionalism        |

JUDGE_S re-scored all 16 user turns under each hypothesis. Every sub-axis
in every hypothesis came out HIGHER in `steals` than in
`romances-and-steals` — the wrong sign relative to the parent. No
hypothesis recovered the parent's gap direction; the qualified-max-d
collapsed to 0 across the board; verdict **NO_SPLIT_FOUND**, top
qualified |d| = 0.00.

Full evidence: `data/axis_splits/theft_aggressiveness-2026-05-18T04-44-54-147Z.json`.

### What this means

This is the user's "failing to split because the first few hypotheses
weren't separating enough" case made concrete. Three interpretations,
not mutually exclusive:

- **The parent gap may be noise**: |d|=0.88 sounds like a large effect,
  but it's measured on n=10 vs n=6 with N_TURNS_PER_CHAT=2 — the harness
  is operating at the noise floor of its own protocol.
- **The split-design LLM is biased toward steals-flavored sub-axes**:
  every hypothesis described the `steals` context with sharper / more
  positive theft language than `romances-and-steals` (where the steal-y
  behavior is intertwined with romance and reads as more diffuse). The
  designer-LLM's intuitions about what makes "theft" tracked the
  *clarity* of the theft expression, not the *aggressiveness* level the
  parent rubric pointed at.
- **The entanglement may be genuine and not splittable at this resolution**:
  the romance context modulates theft globally in a way no clean axis
  pair can separate. Recording this as "axis appears genuinely entangled
  at this resolution" and either expanding the bio sample or accepting
  the joint axis as canonical are both legitimate next steps.

The demonstration succeeded in the relevant sense: the splitter fired,
the LLM produced plausible candidates, the principled criteria rejected
all of them, no derived axes were committed to the registry. This is the
exact opposite of the kind of failure mode where an under-constrained
threshold accepts a noisy "split" and pollutes the registry with
axes that don't measure what their names claim.

### Follow-up tasks the run surfaced

- **Run-confounded sample-size**: re-run lock_in_iterative with a larger
  N_TURNS_PER_CHAT (say 4-6) on Rogue-Cancer specifically, then re-feed
  the splitter. If the parent gap shrinks with more samples, the
  entanglement was protocol-noise; if it grows, it was real.
- **Designer-LLM prompt refinement**: the proposeSplits prompt should
  emphasize that the sub-axes need to *flip*, not *agree* — i.e. one
  sub-axis HIGH where parent was LOW. Today's prompt asks for splits
  that "separate the contexts" without pinning sign-recovery as a
  hard requirement on the designer side.
- **Velocity-stall now wired** (item 5, lock_in_iterative.mjs): inner
  and outer loops accept "stalled" as a stop reason distinct from
  "converged" so the run records WHICH stop-reason hit. Next iterative
  run will produce richer evidence for the splitter.
