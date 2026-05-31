# Petit retrospective — the last week and a half (2026-05-16 → 2026-05-28)

**Author:** Opus 4.8 (1M ctx), session 2026-05-28.
**Scope:** a review of what this project was actually grinding through over
the back half of May, written from the artifacts (git history, harness code,
docs, and a fresh eval run done while writing this). Garnished with concrete
findings from a three-step dig (a→c) into the SVG harness lineage.

This is a narrative companion to the indexed `docs/project_archeology.md`, not
a replacement. Where archeology timestamps *events*, this asks *what the
project was trying to learn* and *what changed in how it tried*.

---

## 1. The one-sentence thesis of the whole repo

Both halves of this codebase run a single program:

> **empirically characterize a fixed stochastic backend against its own
> physics, where the deliverable is a response surface (a ceiling, a Pareto
> frontier, an effective-dimensionality map) and the two cardinal sins are
> (a) measuring a path production doesn't run, and (b) collapse of the
> quantity you're trying to maximize.**

The only thing that changed across the **May 7–8 seam** is that the "backend
under study" stopped being a Metal kernel and became the language model
itself. The engine became the instrument for studying the thing it serves.

This inheritance is *documented, not just inferred*:
`docs/user_agent_likert_methodology.md` carries the header *"Companion to:
`docs/quant_search_motivation.md` — the substrate-aware framing this note
inherits from,"* and `docs/tool_elicitation_findings.md` calls itself *"a
direct-bridge A/B benchmark … analogous to the harnesses under
`tools/quant_search/`. It is not a client."*

---

## 2. Two eras, one disposition

### Era 1 — kernel microbenchmarks (Apr, into early May)
Roofline-style saturation studies. One template: *isolate one kernel in a
standalone binary → sweep the one knob that should move it → measure the
regime-appropriate metric (GB/s, TFLOPS, occupancy) via honest GPU wall →
locate the knee and attribute it (bandwidth / compute / occupancy / hard
Metal resource limit).* The thesis is stated outright in
`docs/kernel_throughput_ceiling.md`: *"The plateau is the kernel ceiling, not
a scheduler ceiling. To go past 135 tok/s the kernel itself has to change
shape."* Levers that didn't move it were **falsified in writing** so they
wouldn't be re-attempted (`docs/single_stream_prefill_slump.md`).

### Era 2 — user-agent behavior harnesses (May 8 → 24)
The same shape, re-aimed at model behavior. `tools/user-agent-harness/`
sweeps seed / format / prompt / axis / overlay / persona-target and measures
**convergence / lock-in, distinctness / diversity / collapse, drift,
entanglement / factorization, strategic variety** — with Cohen's d, ANOVA
F-ratio, Mahalanobis distance, and PCA effective-dimensionality as the report
vocabulary. The harness self-describes as a *fixed-point iteration over two
nested functors (bio designer outside, user-agent designer inside), driven by
an LLM-as-judge.*

### The weld
`docs/kv_cache_correlation_finding.md` + `…_diagnosis.md` are the literal join:
an **engine-level RNG-plumbing bug** (the client `seed` was parsed, serialized,
and decoded into `DecodedSampling.seed` but never copied into
`Session.gpuRngSeed`) was hunted *because* an agent-behavior A/B study's error
bars wouldn't shrink — "trials don't carry independent samples." Kernel
correctness work, driven by behavioral-benchmark hygiene. (This is also the
one fix Codex was credited diagnosing — at the seam, on 2026-05-08.)

> **Authorship note, for the record.** The premise that "Opus 4.7 did the
> work, then it was shirked onto gpt-5.5-codex" is not what the git record
> shows. Opus 4.7 is the through-line co-author across *both* eras (173 of
> 239 commits). Codex appears only as a brief diagnostic/review partner
> clustered at the May 7–8 seam (3 commits: a stale-comment audit, the
> KV-seed diagnosis, the toolcards FIFO fix). Sonnet 4.6 did tail-end test
> cleanup on May 24. There was no handoff — there was one continuous Opus
> trunk with a short Codex consult exactly where the project pivoted.

---

## 3. The specimen: the SVG refinement harness (a→c findings)

The clearest single instance of the whole disposition is a lightly-documented
harness that takes a raster image and iteratively asks the VLM for an SVG,
rendering each attempt and scoring it by pixel MSE — a true fixed-point
iterator. It exists in two layers, and only the second is well-documented.

- **Layer 1 (the science):** `scripts/archival/svg_refinement_loop.py` +
  `svg_render.py` (headless-Chromium rasterizer) + `svg_concurrent_bench.py`.
  Black-box, edge-silicon: *"no backwards pass, no hidden-representation
  poking — just Gemma's chat API, PIL rasterization, numpy for MSE."*
  Generalized into a CoT-vs-baseline eval: `output_data/perf_probes/svg_cot_eval.py`
  + `notes/svg_cot_curriculum.md`.
- **Layer 2 (the product):** the SillyTavern `image-to-svg` / `query-to-svg`
  toolcards, productized from Layer 1 (the e2e spec header says so verbatim).

### (a) Fixed: a stale import that bit-rotted the eval
`svg_cot_eval.py` did `sys.path.insert(0, REPO/"notes")`, but
`svg_refinement_loop.py` had been parked under `scripts/archival/` — and that
module's own `from svg_render import render_svg` needs the same directory on
the path. So the eval was broken **two ways** by one stale line. Repointed at
`scripts/archival/`; all 12 imported symbols now resolve. Also corrected the
usage docstring's stale `--with resvg-py` (the rasterizer is Playwright/
Chromium; resvg was dropped earlier because it silently swallowed `<text>`).

### (b) Ran it — real numbers on the "−18% MSE" pilot
Bridge was live (`gemma-4-a4b`, multimodal, 16k free pages). Ran 1 frame ×
2 arms × 2 rollouts × 3 iters (`output_data/svg_runs/eval_retro_demo/`):

| arm | iter-0 MSE | iter-1 | iter-2 | best | mean_best | completion tokens |
|---|---|---|---|---|---|---|
| baseline | **0.0126** | 0.0288 | 0.0278 | 0.0119 | **0.0126** | 6 247 |
| structured-CoT | 0.0597 | 0.0229 | 0.0281 | 0.0229 | **0.0229** | 9 272 |

**First-pass (wrong) reading.** Pixel MSE *rises* across iterations: baseline
iter-0 is the lowest, and `argmin MSE` selects it as "best." The tempting
conclusion — "the loop doesn't converge, fixed-point iteration fails, the
behavioral twin of the kernel ceiling" — is what an earlier draft of this doc
said. **It is wrong, and it's wrong in exactly the way this project's own
later methodology was built to catch: MSE is one scalar, and a bad one.**

**Corrected reading (after looking at the actual images).** The target frame
is the word **"Linux"** in glossy chrome cursive over a faint Tron-style
perspective grid on black. Direct VLM inspection of the rendered trajectory
(Claude-as-judge, this session) inverts the MSE verdict:

- **The auto-selected "best" (iter-0) is the *least* semantically faithful.**
  Both baseline rollouts' iter-0 is a few sparse teal strokes on a mostly-black
  canvas. It scores low MSE largely *because the target is ~90% black and so is
  a near-empty canvas* — the metric rewards leaving the canvas alone.
- **Multi-turn iteration improved semantic fidelity while MSE worsened.**
  baseline r00 iter-2 reads clearly as connected "Linux" cursive **with the
  inner highlight stroke** (it captured the chrome-highlight structure) and the
  dotted grid. baseline r01 iter-2 *added a white outline around the teal
  strokes* — moving toward the target's silver-chrome material — and was
  penalized (0.0119→0.0137) for the extra colored pixels. Adding faithful teal
  detail over black is pure squared-error cost under MSE regardless of whether
  it's *right*.
- **The errors are of a different kind, not merely "more."** The persistent
  material error is hue-vs-lightness: the model renders solid teal/cyan; the
  target is white-silver chrome with teal *edges*. The model locked onto the
  edge hue and missed the lightness — a defect an LLM-judge can name
  ("right hue family, wrong material/lightness") but MSE just folds into a
  scalar. The CoT survivor (cot r00) failed *differently again*: it
  over-indexed on the **grid lines** (two large gray diagonals) and nearly
  abandoned the text — a different decomposition, not a worse one.

So: **fixed-point iteration did not fail here; the objective did.** The loop's
best-selector actively chose the worst semantic result. This is the canonical
"MSE is not a perceptual metric" trap, and it is precisely the gap the later
Likert/VLM-judge harnesses close. The early SVG harness predates that
methodology and never received it (see §3.5).

**On the CoT arm — doubly confounded, do not cite.** mean_best MSE was higher
(0.0229 vs 0.0126) at +48% tokens, opposite the pilot's "−18% MSE." But (1)
n=1 frame / 2 rollouts is noise-dominated as `notes/svg_cot_curriculum.md`
itself warns, and (2) **one of the two CoT rollouts (`cot_r01`) produced no
valid SVG on any iteration — it hit the 2048 `max_tokens` cap every turn**
(the structured-CoT preamble ate the budget), so the CoT mean rests on a
single surviving rollout. That's a harness/arm interaction bug (CoT needs a
larger token budget), not evidence about CoT's effect on the task.

### (c) The productization shed the science — twice
Comparing Layer 1 to the toolcard descendant surfaces a clean pattern: the
card kept the loop's *choreography* and dropped its *rigor*, in two places.

- **Dropped the objective.** `image-to-svg` is single-shot (no MSE, no loop);
  `query-to-svg` loops but stops on a model-judged "DONE" token, not a pixel
  metric. The numeric fixed point — the thing that made it a fixed-point
  iterator — did not survive.
- **Dropped the IID sampling discipline.** The original harness seeds *every*
  call distinctly: `seed_base + it` per iteration and `base_seed + r*1000`
  per rollout (`svg_cot_eval.py:155,220`; module `:418`), with the CLI help
  noting *"seed = seed_base + iter; lets you compare arms with same RNG."*
  The toolcard's server-side dispatch instead stamps a **constant** seed on
  every descendant call:

  ```js
  // plugins/toolcards/index.mjs:397
  const seed = profile.seed ?? 0;          // same value on every llm_call
  const body = { …, temperature, seed };
  ```

  This collides directly with `docs/kv_cache_correlation_finding.md`. Two
  cases:
  - `profile.seed` unset → `seed=0` → engine treats it as "random" (post-fix
    semantics), so RNG is at least independent — but every iteration still
    adopts the *same cached image-prefill prefix pages*, the exact
    cache-adoption-shifts-the-distribution effect the finding measured.
  - `profile.seed` set → after the May-8 seed-propagation fix that value is
    now *actually applied* to `Session.gpuRngSeed`. So K refinement calls (or
    K replicate rollouts through a card) run with **identical seed + identical
    cached prefix = maximally correlated**, the precise non-IID trap. The
    seed fix made seeding load-bearing, which means a fixed `profile.seed`
    now actively *collapses* replicate diversity instead of being a harmless
    no-op.

  **The irony worth flagging:** the original harness used per-call seed
  variation *specifically* to keep samples independent — the discipline the
  KV-correlation finding prescribes. The productized card threw exactly that
  away. So the toolcard is the loop with both halves of the experiment
  removed: no objective, and no independence.

### 3.5 The real gap: the SVG harness never got the judge methodology

The §3.(b) inversion is not a one-off; it is structural. The SVG harness was
built in the *early* style — a single numeric objective, like a kernel
benchmark reporting TFLOPS. But the project subsequently built an entire
**LLM-as-judge** methodology (`docs/user_agent_likert_methodology.md`:
multi-axis Likert scoring, summary→score cascade, per-judge effective-dim
fingerprint; `vision-review` toolcard: PASS/FAIL VLM verdicts; the
`strategy_diversity` notion of *what changed*, not just *how far*) — and
**none of it was ever wired back to the SVG task.** Concretely, the harness
has zero hits for `judge|ssim|clip|perceptual|lpips`; the only signal in the
loop is `mse` (the literal string fed to the model: *"Your previous attempt
scored MSE = X (lower is better, 0 = perfect pixel match)"*).

This matters three ways, all confirmed by inspection this session:
1. **The objective fed back is the wrong one.** Telling the model "lower MSE
   is better" actively steers it toward mean-matching the black background, not
   toward drawing "Linux." The feedback channel optimizes the metric we
   already know is bad.
2. **The selector throws away the good answer.** `best = argmin MSE` picked
   each rollout's *sparsest, least-faithful* iter-0. A perceptual or judge
   score would have picked iter-2.
3. **"What changed across iterations" was never measured.** The interesting
   behavioral question — does the model add detail, restructure, switch which
   feature it foregrounds (text vs grid)? — needs a judge that reads the
   images. Looking at them by hand shows baseline *added highlight/outline
   structure* while CoT *switched to foregrounding the grid*. That's exactly
   the `strategy_diversity` / `discovery`-style signal the later harnesses
   produce, and the SVG harness is blind to it.

**Submission mode was also under-explored.** `--mode {svg,python}` exists
(direct SVG emission vs. emit a `make_svg()` Python program that is exec'd),
but the eval sweeps only the baseline-vs-CoT *prompt* arm at the default
`svg`. The `python` mode — which lets the model compute geometry instead of
hand-placing path points — was never run in this study, and multi-turn
*strategy* (heatmap on/off, iteration count, restructure-vs-refine framing)
was never treated as a swept variable. The "fixed-point iteration fails"
narrative rests on one cell of a grid that was mostly never run.

### 3.6 The token cap was an un-overseen hyperparameter — and it biased the arm

`SVG_MAX_TOKENS` defaults to **2048**, with this justifying comment
(`svg_cot_eval.py:146`): *"SVG bodies typically run 800-1500 tokens; the
structured-cot block adds another 50-200; budget needs slack for refinement."*
That is an **estimate, not a measurement**, and this run falsifies it:

- Every baseline iteration finished cleanly (`finish_reason='stop'`, 790–1200
  completion tokens) — within the estimate.
- **Every `cot_r01` iteration hit `finish_reason='length'` at exactly 2048**
  and returned `err='no <svg> block found'` — the `</svg>` was truncated off
  an otherwise-valid drawing (the raw output shows it had rendered the grid +
  was mid-way through the `L i n …` strokes when it was cut). The structured-
  CoT preamble cost far more than the comment's "+50-200" in the tail.

This is a textbook **LLM-selected hyperparameter without oversight**: a round
power-of-two default, justified by a plausible-sounding token estimate that
was never validated against the *tail* of the output-length distribution, and
falsified the first time the study actually stressed it. Three compounding
failures:

1. **It silently biased the comparison against the arm under study.** The CoT
   arm by construction emits more tokens; capping both arms at the same 2048
   truncates CoT preferentially. A shared cap is *not* a fair control when the
   treatment adds tokens — `max_tokens` is a variable that interacts with the
   arm, so it must scale with the arm or be set safely above the tail for all.
2. **The harness had the signal to catch it and ignored it.**
   `finish_reason` is recorded in every history entry, but nothing treats
   `'length'` as a *censored / invalid* sample. The truncated rollout silently
   became `mse=None`, dropped out of the aggregate, and left the CoT mean
   resting on n=1 with no warning printed. A study that silently discards 50%
   of one arm's samples is the "passes against a broken implementation"
   anti-pattern the project's own UX-debt doc forbids.
3. **The project had already learned this exact lesson and didn't apply it.**
   `elicitation/drift_compare.py` exists specifically to measure how *removing*
   `max_tokens` caps shifted the 14-axis signatures — its thesis: "Big shift →
   the old corpus was contaminated." The SVG eval re-introduced precisely the
   contamination that doc was written to detect.

Minimum fix: set the cap from the measured tail (CoT here wanted >2048;
4096–6144 is the safe floor), and **treat `finish_reason=='length'` as a hard
invalid sample** — retry with a larger budget or discard-and-log, never fold a
truncated rollout into the mean as a silent `None`.

---

## 4. What the week and a half was really about

Stepping back, mid-to-late May was the project **finishing the pivot from
characterizing the engine to characterizing the model on the engine**, and
then **productizing harnesses into the SillyTavern client** — where the
recurring hazard is that productization quietly drops the measurement rigor
that made the harness a harness. The SVG card is the cleanest example, but the
pattern rhymes with the README's "shame and dismay" protocol: features (and
here, *experimental controls*) silently lost in work claimed as additive.

The healthy habit the artifacts show, and that this retro tries to continue:
**falsify in writing** (the prefill-slump doc, the KV-correlation finding) so
the next agent doesn't re-walk a dead lever — or, worse, ship a loop that
looks like science but no longer measures anything.

---

## 5. Open follow-ups (small, concrete)

1. **Fix the token cap + add a truncation guard BEFORE any real run.** Raise
   `SVG_MAX_TOKENS` to the measured tail (≥4096; CoT wanted >2048 here) and
   make `finish_reason=='length'` a hard invalid sample (retry-with-bigger-
   budget or discard-and-log) — never let a truncated rollout fold into the
   mean as a silent `None`. Without this, every CoT number is biased low.
2. **Wire a perceptual / VLM-judge score alongside MSE.** Add SSIM (cheap) and
   a `vision-review`-style gemma-as-judge axis pass (faithfulness, material,
   composition, "what changed since last iter") so the loop's *feedback* and
   its *best-selector* stop being pure pixel-MSE. This is the §3.5 gap; it is
   the single highest-leverage fix and directly tests whether iteration helps
   *semantically* (which by-hand inspection suggests it does).
3. **Sweep the strategy, not just the prompt.** Run `--mode python` vs `svg`,
   heatmap on/off, and iteration count as actual variables. The current
   conclusion rests on one mostly-unrun cell of a grid.
4. **Run the real CoT curriculum.** Only after 1–3: run
   `svg_cot_eval.py --rollouts 4 --frames-stride 1` over all three frame
   sources to put a real effect size (with error bars) on the −18% pilot.
   The current n=1 demo is noise-dominated and must not be cited as a result.
5. **Decide whether the toolcard should seed-vary.** If `query-to-svg` is ever
   used for replicate measurement (not just one-shot product calls), the
   constant `profile.seed ?? 0` should become `seed_base + iter`-style per
   call. If it's purely a product feature, document that it is *not* an IID
   sampler and must not back a study.
6. **Un-archive or re-home the SVG harness.** It's live-but-parked under
   `scripts/archival/`; the eval driver assumes a sibling layout. Either
   promote it out of `archival/` or add a one-line README there so the next
   reader doesn't conclude it's dead.

---

*Marker for rebase hygiene, per README convention: written against `f32321d`
(2026-05-24). The only code change in this session is the import/docstring fix
to `output_data/perf_probes/svg_cot_eval.py`; the eval outputs live under
`output_data/svg_runs/eval_retro_demo/` and are regenerable.*
