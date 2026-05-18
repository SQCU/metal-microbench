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

| #   | Piece                              | Est.  | Status                  |
| --- | ---------------------------------- | ----- | ----------------------- |
| 1   | Axis registry                      | 1d    | starter in progress     |
| 2   | Sparse-sampling controller         | 2d    | starter after #1        |
| 3   | Outer-outer w/ eff-dim objective   | 2-3d  | depends on #1, #2       |
| 4   | Entanglement detector + splitter   | 3d    | starter after #1        |
| 5   | Velocity-stall convergence         | 30min | sketched, wire pending  |

### Missing — easy, just-not-composed

- Multi-counterparty validation in the outer loop (calls existing /transfer-test).
- Counterparty variation as an additional design axis.
- Persistence for 1M-scale (durable run store, restart-safe job queue, dedup).

## 5. The demonstration target

The 2-bio run already surfaced one entanglement candidate:
`theft_aggressiveness` and `astrology_cancerian` are coupled in the
Rogue-Cancer bio — pure-theft floors at 3.5 over three inner iterations,
but pairing the same bio with `romantic_advance=5` immediately unlocks
`theft_aggressiveness=5` on iter 0. The pattern is identical in shape to
the `provocative` finding in `discovery_harness_findings.md`.

**Concrete demo**: feed that trajectory pair to the axis-splitter (item 4).
Expected outcomes (both informative):

- **Split succeeds**: e.g. propose `{ furtive_theft, brazen_theft }` or
  `{ theft_when_alone, theft_when_courting }`; re-judge the historical
  user turns under the new pair; separability passes threshold; commit
  the new axes to the registry; the next outer-outer iteration picks
  targets that exercise the new pair independently.
- **Split fails**: tried 2-3 split hypotheses, none separated the
  historical trajectories beyond noise; record "axis appears genuinely
  entangled with bio-context at this resolution" and either reduce the
  axis vocabulary or accept the joint axis as canonical.

Either way: we've demonstrated the *entanglement → splitting* leg of
the loop without play-pretending that the entanglement isn't real.
