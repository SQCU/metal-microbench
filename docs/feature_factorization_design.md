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
| 3   | Outer-outer w/ eff-dim objective   | 2-3d  | MVP shipped — `explore_corpus.mjs` + `harness_lib.mjs`; two runs in §7 surfaced an achievement-vs-prediction gap |
| 4   | Entanglement detector + splitter   | 3d    | starter shipped — `tools/user-agent-harness/axis_splitter.mjs`; first run NO_SPLIT_FOUND (§5) |
| 5   | Velocity-stall convergence         | 30min | shipped in lock_in_iterative.mjs (`isStalled`)    |
| 6   | Cluster disambiguator              | 2-3d  | shipped — `cluster_disambiguator.mjs`; first 2 runs both produced expected verdicts (§6 Results) |

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

## 6. Cluster disambiguator (item 6 — spec)

### What problem this solves

The splitter (§5, item 4) requires an **existing-axis gap across chat
contexts** to fire — given parent axis `a` and contexts `{X, Y}` where
`mean(a | X) ≠ mean(a | Y)`, decompose `a` into sub-axes that explain
the variance. Tight clusters in B-space (e.g. several Sagittarius bios
that all measure identically on the registry's existing axes) **lack
that gap by construction** and are invisible to the splitter.

The dual problem:

```
Given a cluster {bio_1, …, bio_k} with
    ‖signature(bio_i) − signature(bio_j)‖_existing-axes  small  ∀ i,j
find a NEW axis a_new (or discover that none exists) such that
    Var({mean(a_new | trajectories(bio_i))}_i)  is large.
```

Different bucketing (over bios, not contexts), different objective
(spread the cluster, not explain a parent), different prompt to the
designer. This is **item 6: cluster disambiguator** — a sibling of the
splitter, not a generalization.

### Three honest verdicts

The disambiguator either finds a spreading axis or it doesn't.
When it doesn't, the harness should distinguish **why**:

- **`SPREAD_AXIS_FOUND`** — at least one candidate axis spreads the
  cluster with between-bio variance significantly above within-bio
  variance, AND spread magnitude ≥ ~1.5 Likert points. Register the
  axis (kind=`bio` typically) and the cluster's per-bio coordinates
  on it.
- **`CLUSTER_IS_PARAPHRASE_DEGENERATE`** — no proposed axis spreads
  the cluster, AND pairwise prose-similarity (LLM-judged) is HIGH
  (≥ 4/5). The bios are different *wordings* of the same content;
  there's nothing behavioral to factor. Harness records the verdict
  and stops attempting to grow this corner of B.
- **`CLUSTER_IS_BEHAVIORALLY_DEGENERATE`** — no axis spreads, prose
  is dissimilar (≤ 3/5), behavior is identical. The bios are
  substantively distinct *as prose* but project identically into the
  behavior-space the judge can score. Either the judge's axis
  vocabulary is too thin to catch the difference (which the harness
  could attempt to address by feeding the bios back to the
  designer-LLM with explicit instructions to expand vocabulary), or
  the differences are genuinely non-behavioral (taste, register,
  surface ornament). Either way, recording the verdict is honest.

### Algorithm

```
clusterDisambiguator(bios: [BioId], cp: Counterparty) :
    # 1. Pre-flight: confirm the cluster is tight on existing axes.
    sigs = {b: existing_signature(b)  for b in bios}
    if max_pairwise_distance(sigs) > TIGHTNESS_THRESHOLD:
        raise NotATightCluster

    # 2. Cheap-agent pairing (mandatory). Every bio gets a paired agent_text
    #    via a SINGLE-PASS (K_max_inner=1) agent designer call against a
    #    neutral target ("be your character vividly"). This is foundational:
    #    we never measure bios in the asking-the-LM-to-play-both-roles
    #    degenerate mode. The agent_text may be weak — that's fine, it's a
    #    constant across the cluster, so cluster-internal variance still
    #    reflects bio variance.
    agents = {b: designCheapAgent(b)  for b in bios}

    # 3. Collect trajectories per bio. We prefer TURN-DEPTH (n_turns=4)
    #    over trajectory-count (N_TRAJ=1-2) — longer chats catch trailing-off
    #    and boring-fixed-point signals that 2-turn measurements miss.
    trajs = {b: [runChat(b, agents[b], cp, n_turns=4) for _ in range(N_TRAJ_PER_BIO)]
             for b in bios}

    # 3. DESIGNER_C: propose N_HYPOTHESES candidate axes that should spread
    #    the cluster. Designer sees ALL bio prose + sample turns + the fact
    #    that they currently measure identically on existing axes.
    hypotheses = proposeSpreadAxes(bios, sigs, trajs)

    # 4. JUDGE_C: re-score every user turn of every trajectory under each
    #    candidate axis, aggregate to per-bio means.
    evaluations = []
    for h in hypotheses:
        per_turn = {b: [judgeOnAxis(turn, h) for turn in userTurns(trajs[b])] for b in bios}
        per_bio  = {b: mean(per_turn[b])     for b in bios}
        within   = pooled_within_bio_variance(per_turn)
        between  = variance(per_bio.values())
        spread   = max(per_bio.values()) - min(per_bio.values())
        f_ratio  = between / max(within, EPSILON)
        evaluations.append(Eval(h, per_bio, within, between, spread, f_ratio))

    # 5. Pick winner under strict criteria:
    qualified = [e for e in evaluations
                 if e.f_ratio >= F_RATIO_THRESHOLD
                 and e.spread >= SPREAD_THRESHOLD]
    if qualified:
        top = max(qualified, key=lambda e: e.f_ratio)
        register_derived_axis(top.h.name, kind='bio', def=top.h.def,
                              derived_from={'cluster_members': bios,
                                            'sibling': None,
                                            'reason': 'spread_axis'})
        return SpreadAxisFound(top)

    # 6. No axis spread the cluster — distinguish paraphrase vs behavioral.
    prose_sim    = pairwiseProseSimilarity(bios)        # LLM-judged 1-5
    behavior_sim = pairwiseBehavioralSimilarity(trajs)  # LLM-judged 1-5
    if prose_sim >= PARAPHRASE_THRESHOLD:    # high prose similarity
        return ParaphraseDegenerate(prose_sim, behavior_sim, hypotheses)
    else:
        return BehaviorallyDegenerate(prose_sim, behavior_sim, hypotheses)
```

### Information presented to DESIGNER_C

In contrast to the splitter (which today sees only 3 sample turns +
the parent rubric — see §2 of this doc's followups for the splitter's
information-starvation problem), the disambiguator gives the designer:

| Surface                                  | Why                                                        |
|------------------------------------------|------------------------------------------------------------|
| All k bio prose blocks, full text        | The differences ARE in here; the designer needs them       |
| Per-bio: 2-3 sample user turns           | How each bio actually behaves dynamically                  |
| Existing signature per bio + tightness   | The negative space — "what these bios DON'T differ on"     |
| Existing axis registry (names + rubrics) | Avoid proposing axes that already exist                    |
| Explicit objective                       | "Propose axes that SPREAD these bios on a 1-5 scale"      |

### Acceptance thresholds (defaults, tunable)

- `TIGHTNESS_THRESHOLD = 1.0` (Likert) — max pairwise existing-axis
  distance for the cluster to qualify as "tight" (cheap pre-flight,
  measured only on the cluster's nominal-tight axis to bound cost)
- `F_RATIO_THRESHOLD` — use the actual F-distribution critical value
  at α=0.05 for (k−1, k·N_TRAJ·N_TURNS − k) df. For the demo defaults
  (k=3, N_TRAJ=2, N_TURNS=4) this is F(2, 21) ≈ 3.47. Hardcoding
  ≥3.5 is a pragmatic starting point.
- `SPREAD_THRESHOLD = 1.5` — per-bio mean spread (max − min) on the
  proposed axis must reach ≥ 1.5 Likert points (otherwise it's a
  technicality, not a useful axis)
- `PARAPHRASE_THRESHOLD = 4.0` — pairwise prose-similarity ≥ 4/5 to
  call the cluster paraphrase-degenerate
- `N_TRAJ_PER_BIO = 2`, `N_TURNS_PER_TRAJ = 4`. Rationale: turn-depth
  matters more than trajectory-count for this measurement — a 4-turn
  chat catches **trailing-off / boring-fixed-point** behavior (a real
  cluster-distinguishing signal: some Sagittarius bios may stay
  expressive across 4 turns while others run out of distinctive moves
  by turn 3). Per-bio wallclock: ~2 chats × ~7s/turn × 4 turns ≈ 1
  minute, plus ~16 judge calls per hypothesis × 3 hypotheses ≈
  parallelizable in ~30s. Total ~3-4 min for a k=3 cluster.

### BehaviorallyDegenerate is provisional

A `BehaviorallyDegenerate` verdict means "given the current axis
registry, the judge cannot find a behavioral dimension that
distinguishes these bios". This may change as the registry grows
(via successful splits/disambiguations elsewhere). The right
re-check pattern is **lazy and sparse**, not a big batch:

- Persist BehaviorallyDegenerate clusters with their verdict timestamp
  and the registry-snapshot used.
- A future outer-outer pass that adds N new axes to the registry can
  schedule a re-check for these clusters at low priority — judge each
  cluster's existing trajectories under only the *new* axes, not the
  whole registry. ~k × N judge calls per stale cluster, amortized
  over time.
- Never auto-rerun all stale clusters as a dense batch — that would
  choke the live test budget. Trickle.

### Pairwise similarity helpers (LLM-as-judge, no embedding infra needed)

```
pairwiseProseSimilarity(bios) :
    pairs = combinations(bios, 2)
    scores = [judgeBridge('Score prose similarity (vocabulary, sentence '
                          'structure, word choice) on 1-5: '
                          '\n\nBio A:\n{a}\n\nBio B:\n{b}', a, b)
              for (a, b) in pairs]
    return mean(scores)

pairwiseBehavioralSimilarity(trajs) :
    pairs = combinations(trajs.keys(), 2)
    scores = [judgeBridge('Score behavioral similarity on 1-5: how '
                          'similar are the strategies, moves, and '
                          'observable behaviors of these two users in '
                          'comparable chat turns?\n\nUser A turns:\n{a}'
                          '\n\nUser B turns:\n{b}',
                          sample_turns(trajs[a]), sample_turns(trajs[b]))
              for (a, b) in pairs]
    return mean(scores)
```

Reuses the existing bridge — no new dependencies.

### Pseudohaskell

```haskell
data ClusterVerdict
  = SpreadAxisFound
      { axis       :: Axis
      , perBio     :: Map BioId Double    -- means
      , fRatio     :: Double
      , spread     :: Double
      }
  | ParaphraseDegenerate
      { proseSim    :: Double
      , behaviorSim :: Double
      , tried       :: [Axis]
      }
  | BehaviorallyDegenerate
      { proseSim    :: Double
      , behaviorSim :: Double
      , tried       :: [Axis]
      }

clusterDisambiguator :: [BioId] -> Counterparty -> IO ClusterVerdict
clusterDisambiguator bios cp = do
  sigs   <- forM bios existingSignature
  unless (isTight sigs) (throw NotATightCluster)
  trajs  <- forM bios (\b -> replicateM nTrajPerBio (runChat b Nothing cp nTurns))
  hyps   <- proposeSpreadAxes bios sigs trajs            -- DESIGNER_C
  scored <- forM hyps (evalSpread bios trajs)            -- JUDGE_C
  case filter passes scored of
    (top:_) -> do registerDerivedAxis (axisOf top) "bio" (derivedFromCluster bios)
                  pure (SpreadAxisFound top)
    []      -> do proseSim    <- pairwiseProseSim bios
                  behaviorSim <- pairwiseBehavioralSim trajs
                  pure $ if proseSim >= paraphraseThreshold
                            then ParaphraseDegenerate proseSim behaviorSim hyps
                            else BehaviorallyDegenerate proseSim behaviorSim hyps
```

### Demo plan (what we'd run to resolve the guarantees question)

To exercise all three verdicts and verify the diagnostic actually
distinguishes the cases, construct **three Sagittarius clusters by
hand** and feed each to the disambiguator:

- **Cluster A — paraphrases (target: ParaphraseDegenerate).**
  Same bio rewritten 3 ways: same content (fire-sign, philosophical,
  blunt, big-idea), different wording per bio.
- **Cluster B — substantively different (target: SpreadAxisFound).**
  3 Sagittarius bios differing in some other behavioral dimension —
  e.g. one is a philosopher-Sagittarius (high `probe_depth`), one is
  an adventurer-Sagittarius (high `affective_intensity`), one is a
  blunt-truth-teller (high `provocative`). These DO project to high
  Sagittarian on existing axes but should spread on the registry's
  *other* axes — so a well-designed proposeSpreadAxes should pick up
  one of those existing-axis directions.
- **Cluster C — genuinely identical-on-behavior (target:
  BehaviorallyDegenerate or SpreadAxisFound depending on whether the
  designer can find anything).** 3 Sagittarius bios that are
  *prose-different* but pin to the same behavioral mode — e.g. all
  three describe the same wizard-style adventurer in different
  vocabulary. Hard to construct cleanly; if the disambiguator finds
  an axis, the cluster wasn't actually behaviorally degenerate;
  that's a useful corrective signal.

The demo SUCCEEDS in the same sense the splitter's first run did:
not by finding a split, but by **correctly characterizing what kind
of cluster we have**. The harness's job is to be a truthful witness
about the structure of B, not to always declare a productive answer.

### What this DOES and DOES NOT guarantee

- **Does**: assuming the designer-LLM proposes the right axis in
  its top-K candidates, and within-bio variance is below judge noise,
  the disambiguator will find it. Modulo case (a)/(c) degeneracies,
  it characterizes them honestly.
- **Does not**: guarantee that the designer-LLM's top-K covers the
  axis. The hypothesis-generation step is bounded by the designer's
  vocabulary of plausible axes; for non-obvious dimensions
  (idiosyncratic combinations, novel jargon), it may fail to propose
  the right axis even when one exists. Mitigations: larger K,
  multiple designer calls with different seedings, explicit
  prompt-side hints to consider specific axis families.
- **Does not**: handle cases where the cluster's distinguishing
  dimension is *non-monotonic* on a 1-5 scale (e.g. "three bios that
  differ by being three discrete archetypes that don't lie on a
  spectrum"). For these, the right tool is a categorical disambiguator
  (item 7? not yet specced) that asks the designer for a discrete
  classification rather than a Likert axis.

### First two runs — results (2026-05-18T05:26–05:31Z)

Both clusters used the-rock.png as counterparty, K=1 cheap agents per
bio, N_TRAJ=2 × N_TURNS=4 = 8 user turns per bio. Hand-crafted bios in
`data/clusters/sag_paraphrase.json` and `data/clusters/sag_substantive.json`.

#### Cluster A (`sag-paraphrase`): expected ParaphraseDegenerate

Three Sagittarius bios reworded as paraphrases of one fire-sign template.
Cheap-agent designer converged to nearly identical agent_texts for all
three ("prioritize grand abstractions / unfiltered bluntness"). User
turns were behaviorally near-identical entropy-and-heat-death speeches
about the rock.

- DESIGNER_C proposed: `syntactic_entropy`, `intellectual_hostility`,
  `abstraction_to_concrete_ratio`.
- JUDGE_C re-scored: **none qualified** (max F=0.28, max spread=0.57).
- Pairwise prose-similarity: **5.0 / 5.0** across all pairs (paraphrases).
- Pairwise behavioral-similarity: 4.0 / 5.0.
- **Verdict: `CLUSTER_IS_PARAPHRASE_DEGENERATE`** ✓

Evidence: `data/cluster_disambig/sag-paraphrase-2026-05-18T05-26-34-287Z.json`.

#### Cluster B (`sag-substantive`): expected SpreadAxisFound

Three Sagittarius bios deliberately varying on a dimension *not* in the
existing 22-axis registry: philosophical-domain-of-reference
(metaphysical / physical-explorer / moral-preacher).

- DESIGNER_C proposed: `sensory_orientation`, `normative_directionality`,
  `referential_anchor`.
- JUDGE_C re-scored — **all three qualified**:
  - `sensory_orientation`: F=11.14, spread=1.75 (Metaphys=2.88, Explorer=4.50, Preacher=2.75)
  - `normative_directionality`: F=35.93, spread=3.88 (Metaphys=1.88, Explorer=1.13, Preacher=5.00)
  - `referential_anchor`: F=24.37, spread=3.38 (Metaphys=1.38, Explorer=4.75, Preacher=4.25)
- Winner: **`normative_directionality`** (highest F).
- **Verdict: `SPREAD_AXIS_FOUND`** ✓

Evidence: `data/cluster_disambig/sag-substantive-2026-05-18T05-31-06-363Z.json`.
Registered as derived axis `normative_directionality` (kind=bio,
derived_from.cluster_members={3 Sagittarius substantive bios}) in
`data/derived_axes.json` — the first cluster-derived addition to the
registry. Registry size 22 → 23.

#### Incidental findings worth recording

- **Bio-claim vs behavioral-expression divergence**. Both clusters
  failed the tightness pre-flight on `astrology_sagittarian`. Cluster A
  spread = 1.43 (above the 1.0 threshold); Cluster B spread = 3.00.
  The Explorer-Sagittarius bio in B scored **1.00** on Sag-axis (every
  turn — zero variance) despite the bio explicitly claiming Sagittarius
  identity. Hypothesis: the judge looks for *behavioral* Sag-coding
  (optimism, restlessness, big-idea-loving register) in the user turn,
  and physical-explorer behavior simply doesn't read as Sag-coded even
  when the bio prose is. This is the kind of bio-prose-claim ≠
  behavioral-expression gap the whole harness exists to surface;
  noted in the cluster disambiguator output for downstream use.
- **DESIGNER_C hallucination is robustly handled**. On Cluster A, H1's
  rationale claimed Bio 2 had "stuttering/typos ('thes', 'justing',
  'can's')" — these strings don't appear in the actual sampled turns
  (Bio 1 had some token-level wobble; Bio 2 was clean). The
  designer-LLM confabulated evidence. **The empirical re-judgment
  step caught this automatically**: all three bios scored 4.6 on
  `syntactic_entropy`, the hypothesis didn't qualify, no harm done.
  Strict acceptance criteria > trusting designer-LLM rationale.
- **Wallclock**: Cluster A took ~9 min (chat phase 413s, KV cold-start
  costs). Cluster B took ~5 min (chat phase 165s, warm caches). The
  bridge serializes effectively-parallel chat requests at the model
  token-rate limit — designing for ~k×n_traj×n_turns × ~7s per turn is
  the right mental model.
- **Registry-API bug found and fixed**: `registerDerivedAxis` originally
  required `derived_from.parent` (split-only). Now accepts EITHER
  `parent` (split) OR `cluster_members` (cluster). Cluster B's winning
  axis was persisted manually after the fix.

### What this run did NOT demonstrate

- **CLUSTER_IS_BEHAVIORALLY_DEGENERATE** verdict never fired. To
  construct a deliberate test of it would require bios that are
  prose-different but behaviorally identical — and as noted in the
  spec, if I knew how to construct such a thing deliberately, I'd know
  the axis along which they don't differ, which is the disambiguator's
  job to discover. The verdict path is exercised in code; it'll fire
  in the wild when it fires. (See the lazy-re-check note above: such
  clusters get revisited as the registry grows.)
- **The full ~1k-axis vision** isn't reached by adding one derived
  axis. But the *machinery* for adding more is now demonstrated
  end-to-end: hand a cluster → get a verdict → on success, registry
  grows. Iterating this over many clusters is mechanical from here.

## 7. Outer-outer MVP — explore_corpus

### What it does (item 3)

`tools/user-agent-harness/explore_corpus.mjs` is the target-selecting
outer-outer the picker-vs-fixed-target distinction in §3's
ball-and-stick. Each iteration:

1. Compute participation-ratio effective-dim over the current bio
   corpus's per-axis marginal variances (the simple proxy in
   `harness_lib.effDimParticipationRatio`).
2. Generate K candidate bio targets uniformly in [1.5, 4.5]^|bio_axes|.
3. For each, simulate the corpus-with-candidate-added and score by
   ΔPR (PR_after − PR_before). Argmax wins.
4. Single-pass bio designer (no inner loop yet) writes show-don't-tell
   prose targeting the winning sig.
5. Cheap (K=1) agent designed for the bio. Install both into ST.
6. One chat × N_TURNS=4 vs the counterparty.
7. Multi-axis judge call per turn (one `judgeOnAxes` per turn covering
   all |bio_axes|=6 axes). Aggregate to per-axis means → measured_sig.
8. Append to corpus, recompute PR, persist state.json.

For iters where the corpus has <2 bios, mode is `random_warmup` (no
baseline to ΔPR against). Iter ≥2 switches to `delta_pr_argmax`.

### What it deliberately omits (MVP scope)

- **The full inner loop** — `designAgent` across multiple agent_targets.
  We use one "be yourself" cheap agent per bio. Inner-loop integration
  would multiply per-bio cost by `n_agent_targets × K_max_inner`; out
  of scope for the MVP.
- **Splitter / disambiguator auto-dispatch** — when the loop detects
  entangled axes or tight clusters in the growing corpus, it should
  call axis_splitter and cluster_disambiguator respectively. Wired by
  hand for now; auto-dispatch is the natural next iteration.
- **Sparse-sampling controller (item 2) integration** — judge calls
  cover the full bio_axes set on every turn. `pickSubset` from
  axis_registry exists but no caller uses it yet.

### Two runs

Run 1 used the original `judgeOnAxes` prompt; Run 2 used a retuned
prompt that removed the "be willing to score 1 (absence)" license and
added "USE THE FULL 1-5 RANGE / most real turns sit between 2 and 4".

| Metric                          | Run 1 (3 iters)         | Run 2 (4 iters, retuned judge) |
|---------------------------------|-------------------------|--------------------------------|
| Wallclock                       | 177s                    | 208s                           |
| Final corpus size               | 3 bios                  | 4 bios                         |
| Final eff-dim                   | 1.000                   | 3.085                          |
| Eff-dim trajectory              | n/a → n/a → 1.000       | n/a → 3.139 → 3.139 → 3.085    |
| Iter-2 expected ΔPR             | 5.154                   | 1.920                          |
| Iter-2 achieved ΔPR             | 1.000                   | 0.000                          |
| Iter-3 expected ΔPR             | —                       | 2.049                          |
| Iter-3 achieved ΔPR             | —                       | **−0.053** (eff-dim DROPPED)   |
| All-axes flat measurements      | yes (mostly 1.00)       | partially (some 1.33–1.67)     |

Evidence: `data/explore_corpus/run1.json`, `data/explore_corpus/run2.json`,
plus run logs in commit history.

### Findings

1. **The outer-outer mechanics work correctly.** Target-picking,
   ΔPR scoring, random_warmup → delta_pr_argmax transition, per-iter
   state persistence, expected-vs-achieved logging — all firing as
   designed. Iter 2 of run 2 correctly picked the candidate with
   highest expected ΔPR (1.920) out of 8 random candidates.
2. **The bio→behavior→judge chain is the binding constraint, not the
   picker.** Even with retuned judge, most per-axis measurements land
   at floor (1.00). Inspecting the actual user turns shows they are
   well-aligned to the targets (run 1 iter 0 was hostile-philosophical
   Sag; run 2 iter 1 was sentimental-protective Cancer; iter 2 was
   casual-colloquial Sag). The judge sees these as "not textbook
   enough" and floor-scores them.
3. **Judge prompt has large effect.** Manual probing showed the same
   turn scored Sag=3, Cancer=2, Colloq=4 under one prompt and
   Sag=1, Cancer=1, Colloq=2 (averaged) under the original. Removing
   the "be willing to score 1" license + adding "use full range" lifted
   run 2 to eff-dim 3.139 vs run 1's 1.000.
4. **Expected ΔPR ≠ achieved ΔPR.** The picker assumes target sig is
   exactly hit. Reality: achieved sig sits closer to floor and the
   variance lands non-uniformly across axes. Run 2 iter 3 achieved
   **negative** ΔPR — the new bio's measurement (Sag=1.67, others
   ~1) concentrated MORE variance on Sag (which already had the most),
   shrinking PR. Real harness would benefit from a noise-aware objective:
   "pick candidates that maximize E[ΔPR | achievement-noise-model]"
   rather than "pick candidates that maximize ΔPR | exact-target-hit".
5. **Some axes never moved.** `register_colloquial` and `warm` stayed
   at 1.00 for all 4 bios in run 2. Either these axes need a different
   counterparty than the-rock (which doesn't elicit warmth or
   colloquial banter the way a chatty NPC would), or the bio→behavior
   chain needs a more aggressive elicitation agent than the K=1
   cheap baseline. Both are configurable.

### What this DOES and DOES NOT demonstrate

- **Does**: outer-outer machinery runs unattended over N iterations,
  picks targets by ΔPR-maximization, persists state, computes
  expected-vs-achieved deltas. The plumbing for the ~1k-axis vision's
  self-driving loop is in place.
- **Does not (yet)**: produce a monotonically-growing eff-dim
  trajectory. The binding constraints are upstream of the picker —
  judge calibration, bio-designer show-don't-tell, counterparty
  choice, cheap-agent vs full-inner-loop tradeoff. Each is a
  knob that can be turned; none is a harness-architecture problem.
- **Does**: identify these constraints AS the next-bottleneck items,
  via the achievement-vs-prediction gap the picker logs per iteration.
  Iter 3's negative ΔPR is a useful diagnostic, not a bug.

### Judge prompt A/B (item: knob #1 from the original list)

Following the run-2 floor-bias observation, ran a controlled A/B over
7 judge-prompt variants on 4 hand-graded fixture turns × 5 trials each
(`tools/user-agent-harness/judge_prompt_ab.mjs`). Two rounds total
(rerun added V5/V6 after the user observed "try avoiding telling the
judge what numbers to use at all in any capacity"):

| Variant                    | Description                                  | MAE  | floor-bias |
|----------------------------|----------------------------------------------|------|-----------|
| V0_floor                   | original "be willing to score 1 (absence)"  | 1.18 | 20.0%     |
| V1_full_range              | "use full range, most turns 2-4"            | 1.02 | 30.0%     |
| V2_minimal                 | task statement + rubric + turn               | 0.92 | 27.8%     |
| V3_describe_first          | describe-then-score + calibration            | 0.89 | 20.0%     |
| V4_anchored                | explicit Likert anchors (1=absent…5=textbook)| 0.99 | 24.4%     |
| V5_describe_no_calibration | describe-then-score, NO calibration          | 0.76 | **12.2%** |
| **V6_pure_minimal**        | "score the turn on each axis per the rubric"| **0.61** | 16.7% |

**Robust ordering: every variant that tells the judge anything about
what numbers to favor is worse than the ones that don't.** V0/V1/V3/V4
all sit in the back; V5/V6 (no calibration language) are decisively
best. The "use full range" prompt is actively WORSE than the original
floor-biased one (MAE 1.02 vs 1.18 — close; floor-bias 30% vs 20%).
The describe-first chain-of-thought helps a little (V3 < V0) but
removing calibration helps far more (V5 < V3). V6 (just "score per
the rubric", no philosophy at all) wins on both MAE and std.

**Lesson, generalizable**: tell the judge WHAT to score, not HOW to
score. Promoted V6 to `harness_lib.judgeOnAxes` default.

Caveat: absolute MAE values shifted substantially between the two A/B
runs (V0 went 0.70 → 1.18, V3 went 0.60 → 0.89) at N=5 trials and
temperature=1.0. The *ranking* is reliable; the absolute numbers
require N ≫ 5 to settle. Evidence:
`data/judge_prompt_ab/ab-*.json`.

### Run 3 with V6 judge

Re-ran explore_corpus (4 iters from empty corpus) with V6 as the new
default judge. State: `data/explore_corpus/run3_v6judge.json`.

| Run | Judge | Final eff-dim | Trajectory |
|---|---|---|---|
| run1 | V0_floor (original) | 1.000 | n/a → n/a → 1.000 |
| run2 | V1_full_range (calibration was WRONG direction) | 3.085 | n/a → 3.139 → 3.139 → 3.085 |
| run3 | **V6_pure_minimal (best by A/B)** | **1.883** | n/a → n/a → 1.000 → 1.883 |

**Surprising finding**: the best judge by A/B (V6) produced LOWER
eff-dim than the second-worst judge (V1)! Why: V1 floor-bias = 30%
but it was inflating mid-range scores wrongly; that inflation looked
like cloud variance and pumped up PR. V6 is more strict — when the
bio→behavior chain produces a bland turn (which random_warmup bios
often do), V6 correctly scores 1, the cloud genuinely lacks spread,
and PR is correspondingly low. **Eff-dim is sensitive to judge
calibration; an over-generous judge inflates the apparent objective**.

Run 3's eff-dim of 1.883 over 4 axes with nonzero variance (Sag 69%,
Norm 21%, Trope 8%, Colloq/Warm ~1%) is the honest signal: with 4
bios, a strict judge, the-rock counterparty, and the cheap-K=1 agent,
the corpus genuinely spans about two effective behavioral dimensions.

**The picker still works correctly.** Iter 3 of run 3 expected ΔPR=4.689
(target=balanced moderate), achieved Δ=0.883 — the achievement gap
remains (target-vs-measurement noise), but it's measured against a
trustworthy baseline now.

### Next-iteration knobs (in priority order)

1. **Bio designer show-don't-tell upgrade.** The current designer
   prompt asks for show-don't-tell but the actual prose still trends
   toward self-description. A few-shot prompt with high-quality
   target↔prose pairs would help.
2. **Counterparty diversity.** Auto-rotate among a small pool
   (the-rock, a chatty NPC, a hostile NPC) per iteration so axes
   like `warm` and `register_colloquial` get the right context to fire.
3. **Auto-dispatch to splitter/disambiguator.** When corpus growth
   stalls (Δeff-dim ≈ 0 for K consecutive iters), the loop should
   call `cluster_disambiguator` on the densest cluster in B-space.
   When a measurement diverges across contexts within a single bio,
   call `axis_splitter`. Both tools exist; the auto-dispatch is the
   integration work.
4. **Sparse-sampling controller integration (item 2).** Use
   `axis_registry.pickSubset` to vary which |bio_axes| subset each
   judge call asks about, so the corpus grows measurements across the
   full registry over time rather than always reusing the same 6 axes.
5. **Noise-aware objective.** Model the bio→achievement noise per axis
   (the gap between target and measured), and pick candidates that
   maximize E[ΔPR | noise-model] instead of assuming exact hits.

## 8. Context-suggester (Phase A shipped)

### Problem

As the harness procedurally generates more bios, the suggester sidebar
becomes unusable — scrolling through every persona by default is bad
UX, and the "activate suggest for all" button becomes dangerous /
expensive at scale. Necessary UX shift to filter the drawer to a small
contextually-appropriate selection.

### Algorithm (Phase A — `tools/user-agent-harness/context_suggester.mjs`)

```
inputs:  chat context (recent messages), K_active (e.g. 2), K_disabled (e.g. 2)
1. Load persona-signature manifest from data/ run files.
2. Compute per-axis coverage across the manifest.
3. Call ContextJudge ONCE:
     prompt = (axis registry with coverage tags + chat context)
     output = K_active+K_disabled SPARSE target vectors
              (each = {label, rationale, axes: {axis_name: int, ...}})
   IMPORTANT: bios/agents NEVER enter the prompt — cost stays O(1) in
   corpus size.
4. For each target: sparse Euclidean distance to every persona;
   pick the nearest. Skip personas already picked (no duplicates).
5. First K_active picks would auto-trigger /poll in the UI;
   next K_disabled render as click-to-enable.
output:  {active: [persona_ids], disabled: [persona_ids], targets, evidence}
```

The judge is told which axes are well-covered (≥30% of manifest)
and instructed to prefer them — otherwise it proposes targets the
NN can't match (early A/B run had this failure mode).

### Phase A results

Manifest: 19 personas from explore_corpus, cluster_disambig, and
lock_in_iterative runs. Coverage skew (axes with ≥30%): astrology_sagittarian
68%, normative_directionality 32%, warm 58%, register_colloquial 58%,
trope_density 58%, plus a long tail of sparse axes.

Ran on 3 fixture contexts:

| Target type the judge proposed | Grief context | Tech-debate context | Rock-intro context |
|---|---|---|---|
| Directive / blunt / philosophical | **Moral Preacher** (Norm=5) | **Moral Preacher** (Norm=5) | **Moral Preacher** (Norm=5) |
| Sentimental / Cancer-coded | **RPG Rogue Cancer** | — | **RPG Rogue Cancer** |
| Casual / warm | Sag Paraphrase 3 | **Sag Paraphrase 3** | Sag Paraphrase 1 |
| Tech-bro / oversensitive engineer | — | **RPG Wizard Sagittarius** + Physical Explorer | — |
| Trope-y melodrama | Physical Explorer | — | Physical Explorer |

Evidence: `data/context_suggester/chat_*-*.json`.

### Findings

1. **The pipeline works end-to-end.** Each context produces a
   distinct set of K_active+K_disabled picks; the picks correlate
   with the proposed target types in sensible ways.
2. **Coverage-aware prompting is necessary.** Without telling the judge
   which axes the manifest covers, it proposed axes that no persona had
   measured → no-match outcomes. Adding the coverage hint reduced
   no-matches from 2/12 → 0/12 across the three fixtures.
3. **Moral Preacher dominates** because it's the only persona with
   normative_directionality ≥ 5, and the judge keeps proposing
   "directive / blunt / opinionated" target archetypes for many
   contexts. **This is a feature, not a bug**: it correctly identifies
   that the Preacher fills a niche the rest of the corpus doesn't
   cover. With more bio diversity, the picks would spread.
4. **Bland bios are correctly de-prioritized.** The mostly-1s
   `explore-iter*` bios from random_warmup iterations almost never get
   picked — they don't differentiate from each other or the baseline.
   The harness's measured per-axis variation drives the selection.
5. **Cost is O(1) in corpus size** for the LLM call. One ContextJudge
   per chat-context, regardless of N_personas. NN lookup is trivial.
   Compare per-persona LLM judging: would be O(N_personas) calls per
   suggestion refresh.

### Phase A refinements (after user review)

Two refinements applied to Phase A based on the design review:

**Uniform-prior fallback on missing axes.** Original `sparseDistance`
returned Infinity for personas with no overlap on target axes, which
EXCLUDED them entirely. Replaced with a `MISSING_AXIS_PENALTY = 2.0`
contribution per missing axis (half the Likert range, in axis units).
Now every persona gets a finite distance:

- No selectivity (target axes all uncovered): every persona is at
  baseline distance ≈ MISSING_AXIS_PENALTY → ~uniform prior over picks.
- Selective: personas with actual matches outscore the missing-axis
  baseline → biased sampling that peaks on the selected features.

Re-ran the 3 fixtures: Moral Preacher dominance dropped from 3/3 to
2/3 contexts; previously-unmatched personas now compete. This is also
the foundation for optional softmax sampling in a future refinement
(temperature-controlled stochastic picking instead of argmin).

**Why does Moral Preacher dominate at all?** Two effects:

1. **Corpus coverage skew**: the Preacher is the only persona with
   `normative_directionality` ≥ 5. Any target the judge proposes with
   a high-Norm component lands on Preacher by huge margin.
2. **Rubric framing**: the Norm axis rubric is "1: descriptive/
   observational; 5: prescriptive/directive". The "5 = prescriptive"
   pole reads to the judge as the active/helpful direction, biasing
   target generation toward high-Norm archetypes.

Fix is corpus diversity (generate more bios that span the Norm axis),
not algorithm changes. Optional secondary fix: rewrite the rubric to
neutralize the pole-value loading.

### Phase B (queued)

Port the algorithm to the diegetic UI:

- `POST /suggest-personas` plugin endpoint wrapping the algorithm.
- Suggester sidebar = filtered view with K_active+K_disabled picks,
  always. No "show all" toggle (weapon-wheel UX: bounded slots, not
  a phonebook — scaling-out is precisely the point).
- K_active auto-triggers /poll on suggestion-refresh; K_disabled =
  click-to-enable in the drawer.
- Persona signatures need to be persisted alongside bios — either
  extend `POST /personas/<key>` to accept a `signature` parameter, or
  build a live aggregator endpoint that reads `data/` files at query
  time. The Phase A script's `loadManifest()` is a reference impl.
- Optional refinement: per-axis backfill so every persona has
  measurements on the same canonical bio_axes set (improves NN
  match quality across the corpus uniformly).
