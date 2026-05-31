# Causal-ophthalmology harness — design note

**Status:** design + first implementation (elicitation/measurement/calibration
upgrade). Lesion studies are the *motivation*, **out of scope for this pass**.
**Author:** Opus 4.8, 2026-05-28. Supersedes the elicitation/measurement
design of the archived `scripts/archival/svg_refinement_loop.py`.
**Companion to:** `docs/retro_2026-05-28_petit_retrospective.md` (§3.5–3.6, the
findings that motivated this), and — in framing — `docs/quant_search_motivation.md`
(substrate-aware measurement) and `docs/user_agent_likert_methodology.md`
(LLM-as-judge as instrument).

---

## 1. What the instrument is *for* (motivation — deferred)

The eventual purpose of this harness is **causal ophthalmology**: a
psychophysical, lesion-study readout of what the gemma-4 vision+LM stack can
*perceive*. The wager (operator's, 2026-05-28):

> "Image features that gemma can autoencode through a program are *probably*
> qualitatively accessible to the gemma-4 residual stream — whether or not you
> believe in classification-probe accuracy or self-report. This is a causal
> ophthalmology study which, as a side effect, makes visually diverse SVGs
> from presenting similar images to multiple model runs."

Program-emitted SVG is a **high-bandwidth behavioral readout of internal
representation**: the model must commit to a full *structural hypothesis* about
the image, which is harder to fake than a linear-probe scalar and richer than a
caption. The intended use is as a sensitive scorer for representational
pathologies introduced by post-training quantization, continued training /
distillation, LoRA, or activation interventions — on the LM weights *or* the
vision-encoder weights.

**Epistemic shape (for when lesions are in scope).** Successful reconstruction
is strong evidence a feature is represented *and* decodable. Failure is
three-ways ambiguous — perception failure / articulation (code-writing) failure
/ search failure. The disambiguator is the *paired differential* design: hold
the program-scaffold and one stack fixed, lesion only the other, and a
per-feature fidelity delta localizes the cause. That protocol, the
feature-isolating probe battery, and the fixed-external-reference judge are
**documented here but not built in this pass** (see §5).

---

## 2. Scope of THIS pass — make the instrument capable and honest

Before any lesion can be measured, the instrument must be (a) *able* to let the
model express what it perceives, (b) *multi-measure* so a single bad scalar
can't invert the verdict, and (c) *calibrated* so the readout isn't silently
truncated. The retrospective showed the archived harness failed all three:

- **Construct error in elicitation.** Both system prompts banned *exactly the
  primitives the targets require* — *"No text, no images, no gradients, no
  filters — shapes + solid colors only."* The ban was a vestigial accommodation
  of the old `resvg` rasterizer; the renderer is now headless Chromium (full
  SVG spec), so the restriction is dead weight. The model rendered flat teal
  for a *metallic-chrome-on-perspective-grid* target because flat color was all
  it was allowed. That is not a model limitation — it is a blind spot designed
  into the probe.
- **Single, inverted measure.** Pixel MSE was the only feedback *and* the
  best-of-N selector. On a mostly-black target it rewards mean-matching the
  background; `argmin MSE` selected each rollout's *sparsest, least-faithful*
  iteration as "best." Direct VLM inspection inverted the ranking.
- **Mis-calibrated generation config.** A round `max_tokens=2048` default,
  justified by an unvalidated token estimate, truncated the structured-CoT arm
  on every iteration (`finish_reason='length'`) and the truncated samples were
  silently folded in as missing — biasing the comparison.

**This pass delivers the fixes, demonstrated, with no lesion:** programmatic
emission + full SVG feature set, multi-measure, multi-channel feedback, a
non-MSE selector, and a correctly calibrated generation config with a
truncation guard.

---

## 3. Elicitation upgrade

1. **Full SVG feature set, unbanned.** New system prompts explicitly invite
   gradients (`linear/radialGradient`), filters (`feGaussianBlur` glow/blur),
   patterns (repeated texture / lattices), `<text>`, `clipPath`, opacity,
   blend, and transforms — and tell the model to *pick the primitive that
   matches the structure* (metallic = gradient + highlight band; perspective
   grid = lines under a transform; texture = a `<pattern>` tile). Chromium
   renders all of it.
2. **Programmatic / scripted emission as a first-class, swept mode.** `--mode`
   ∈ {`svg`, `python`, `both`}. The `python` mode asks for
   `def make_svg() -> str:` and explicitly directs the model to **use loops and
   computation** for repetitive/parametric structure that is intractable to
   hand-place within a token budget: perspective lattices, gradient-stop
   arrays, hatching/stippling, motifs repeated along a path, procedurally
   placed shapes. (The architecture generalizes to other scripting languages;
   Python is implemented first because the exec sandbox already exists.)
   *Rationale:* hand-emitted SVG is a flat list the model places by hand and
   loses count on; a program expresses generative structure compactly and
   shifts the error distribution from un-actionable coordinate drift to
   diagnosable parametric/logic error.

## 4. Measurement & calibration upgrade

1. **Multi-measure, reported side by side (never collapsed to one scalar).**
   - **MSE (pixel)** — kept, model-independent. *Retained as a feedback channel
     and a recorded covariate, NOT as the selector.* The operator's call: the
     pixel-delta is the one signal that depends on no model's judgment, and
     whether the model can *act on a delta-heatmap* is itself a percept probe.
   - **SSIM (perceptual, model-independent)** — structural similarity;
     correlates with perceived fidelity far better than MSE.
   - **VLM-judge faithfulness (semantic)** — a gemma-as-judge score +
     structured "what's missing / biggest fix" critique. *On-device and
     circular by construction* (gemma grading gemma); acceptable here because
     **this pass measures capability, not a lesion**. For any differential
     study the judge MUST become a fixed external reference (§5).
2. **Multi-channel feedback.** Each refinement turn shows the model: the scalar
   MSE *and* SSIM, the rendered previous attempt, the amplified pixel-difference
   heatmap, *and* the judge's structured critique (prioritized missing
   elements). The model optimizes a richer target than "lower MSE," which on
   the chrome/grid target steered it toward an empty canvas.
3. **Non-MSE selector.** `best` is chosen by a configurable `--primary-metric`
   (default **SSIM**, higher = better), with all measures recorded so
   MSE-vs-perceptual *disagreement* is preserved as signal.
4. **Calibrated generation config + truncation guard.**
   - `--max-tokens` default **6144** (set above the measured output tail;
     programmatic + full-featureset SVG is longer and more variable than the
     old shape-only output).
   - **`finish_reason=='length'` is a hard event, not silent.** On truncation
     the call retries once at 2× budget; if it still truncates the sample is
     marked **invalid and logged** — never averaged in as a missing `None`.
   - `--temperature` and per-(rollout, iter) **distinct seeds** are explicit
     parameters (independent samples; the seed discipline the toolcard dropped).

## 5. Explicitly deferred to the lesion-study pass (NOT in this pass)

- **Fixed external reference judge.** For a differential lesion study,
  gemma-judging-gemma self-cancels (producer and referee degrade together). The
  primary referee must be stable while the subject changes — frozen fp16
  gemma-4 and/or Claude, with SSIM as a model-independent anchor.
- **Feature-isolating probe battery at threshold.** Random video frames
  confound feature classes. A real instrument needs Ishihara/Snellen-style
  *plates* (pure gradient, pure lattice, path-following, color-discrimination,
  spatial-frequency grating), tuned near the model's ability threshold where
  degradation shows first.
- **Reconstruction-diversity / mode-collapse axis.** Spread of reconstructions
  across IID seeds is a second pathology readout (collapse = variance loss).
  Requires the IID/seed discipline this pass installs.
- **Paired differential protocol + perception-vs-articulation localization.**

---

## 6. Layout

- `tools/svg_elicit/elicit.py` — the upgraded harness (this pass).
- Reuses rasterizer + extractors from `scripts/archival/svg_refinement_loop.py`
  (`render_svg`, `extract_svg_directly`, `extract_python_and_run`,
  `mse_images`, `diff_heatmap`, `image_to_data_url`, `load_target_from_path`).
- Outputs under `output_data/svg_runs/elicit_<tag>/`.

---

## 7. First demonstration run (2026-05-28)

Same frame that defeated the restricted harness — "Linux" chrome cursive on a
perspective grid (`KCrfDHS_YUw/frame_0000`, 976×720). One rollout per mode, 3
iterations, judge on, SSIM selector. Output:
`output_data/svg_runs/elicit_retro_demo/`.

| mode | SSIM trajectory | MSE trajectory | best | judge |
|---|---|---|---|---|
| svg | 0.724 → 0.556 → 0.694 | 0.0113 → 0.0191 → 0.0139 | SSIM 0.724 @ iter0 | 1/1/1 |
| python | 0.160 → 0.184 → **0.615** | 0.0146 → 0.0210 → 0.0144 | SSIM 0.615 @ iter2 | 1/1/1 |

**What worked (the elicitation thesis, confirmed):**
- **The model used the unbanned primitives immediately.** svg-mode output
  contained `<text>` (it wrote the word as actual text, in a serif face),
  `feGaussianBlur` (glow), `<pattern>` (grid), and a `transform` (the diagonal).
  python-mode used `for` loops to procedurally emit a **74 KB** SVG with a
  `<pattern>` background texture + `feGaussianBlur` glow — structure that is
  intractable to hand-place. Rendered output is recognizable glyphs-with-glow
  over a textured ground, not the old flat-teal squiggles.
- **Multi-turn iteration HELPED in python mode** (SSIM 0.160→0.615, monotone
  climb driven by the multi-channel feedback) — the direct counter to the
  archived harness's "iteration makes it worse" artifact. The earlier artifact
  was the MSE objective + selector, not the loop.
- **Calibration held:** 0 invalid samples; the 6144 cap never truncated even
  the 74 KB programmatic output. The finish_reason guard had nothing to catch
  this run, which is the point.

**What the run exposed (findings, not failures):**
- **The on-device judge is degenerate — floor-biased to 1/5 on every render.**
  This is exactly the floor-bias pathology `judge_prompt_ab.mjs` was built to
  detect, observed live. It confirms §5's insistence *empirically*: a circular
  gemma-as-judge is uninformative as a scalar and must be validated +
  externalized before it can score anything. (Its *textual* critique still fed
  forward as feedback; only the scalar is useless.)
- **SSIM is a far better selector than MSE, but not sufficient alone.** It
  correctly picked python iter-2 (the real improvement); for svg it picked
  iter-0 over a close iter-2. A perceptual + validated-judge ensemble is the
  right selector — single-scalar selection is still lossy.
- **Programmatic mode is expensive** (391 s vs 131 s) and can over-generate
  (74 KB). Worth a complexity/token budget or a size-penalty term later.

Net: the elicitation + calibration upgrades are demonstrated working; the
measurement upgrade is half-done — SSIM/MSE multi-measure is solid, but the
*judge* leg needs validation + externalization (the §5 work) before it carries
weight. No lesion was run.

---

## 8. Judge calibration — prompt-variant + k-shot sweep (2026-05-28)

The §7 floor-bias is an *elicitation-strategy* problem, not a perception
ceiling. `tools/svg_elicit/judge_calibrate.py` validates the image judge the
way `judge_prompt_ab.mjs` validated the text judge: against a
**controlled-degradation test set** (each candidate a known perturbation of a
reference frame → ground-truth semantic rank with no human/external model in
the loop). Test corpus: the twerking-Among-Us gif (the primordial reference),
16 poses extracted to `test_data/amongus_frames/`. Output:
`output_data/svg_runs/judge_calib_demo.json`. 20 test items / 2 frames, 10
degradations each (identity→blank, incl. grayscale & hue-rotate as deliberate
SSIM-vs-semantic dissociation cases). k-shot exemplars span the 1–5 range from
*held-out* frames (no leakage).

| variant | floor-bias | rankρ vs gt | discrim | imgs/call | vision-cache hits Δ | wall |
|---|---|---|---|---|---|---|
| V0 holistic ("be strict", no rubric) | **0.31** | 0.37 | 1.46 | 2 | 26 | 72 s |
| V1 anchored rubric (0-shot) | **0.00** | 0.39 | 1.42 | 2 | 40 | 49 s |
| 1-shot | 0.00 | 0.42 | 1.50 | 4 | 81 | 50 s |
| 2-shot | 0.00 | 0.38 | 1.13 | 6 | 119 | 57 s |
| 3-shot | 0.00 | **0.46** | **1.67** | 8 | 161 | 57 s |

**Findings:**
1. **Floor-bias is an elicitation bug; the anchored rubric eliminates it**
   (0.31 → 0.00). This is the large, robust effect — the single cheapest fix,
   and it confirms the diagnosis: the degenerate judge had no scale to anchor
   to and defaulted to the floor. **Promote the rubric to the `elicit.py`
   judge default.**
2. **k-shot helps, modestly and non-monotonically** (rankρ 0.39→0.42→0.38→0.46;
   3-shot best on both rank-corr and discrimination). At n=20 single-trial the
   per-step deltas are within noise; the 2-shot dip is not load-bearing. The
   honest read: k-shot is a real but small win on top of the rubric, needing
   more items + repeated trials for a tight effect size.
3. **No multi-image-context ceiling at 8 images.** The "could-harm" hypothesis
   (gemma never warmed up on 6+ interleaved images) did NOT trigger — quality
   held/rose through 8 images. gemma-4 can do 6–8-image interleaved ICL judging.
4. **The cache amortization is empirically confirmed.** vision-cache hits scale
   with shot count (26→161) because the fixed exemplar images are reused across
   every test call, and **wall time barely moved as images/call went 2→8**
   (72 s→57 s; V0 was slowest only because it ran cold-cache first). The k-shot
   prefix is ~free, exactly as predicted from the image-hash vision cache +
   content-hash prefix cache.

**Caveats:** that sweep used controlled-degradation exemplars (the self-contained
fallback); absolute rankρ (~0.4–0.46) was moderate. The on-policy cut (§9) does
much better.

---

## 9. On-policy exemplars + exact-token recording (2026-05-28)

Two upgrades, per operator direction:

**(a) Exact token in/out recording — from the bridge, not re-tokenization.**
The bridge returned only *strings*; it now honors `return_token_ids: true`
(non-streaming) and returns, per `server/bridge.py`:
`choices[0].token_ids` (raw emitted completion ids — `u.new_tokens`),
`thinking_token_ids`, and `prompt_token_layout` (per-segment text token ids —
BOS/turn markers included since they come from `render_chat` — plus image
segment positions and the exact `image_soft_tokens_total`). Pure-Python change,
no Swift rebuild. `tools/svg_elicit/recorder.py` consumes these and writes one
JSONL transcript per call mapping **exact token-in spans (incl. image soft-token
[start,end) spans tagged by image sha), exact token-out ids, strings in/out**,
and sha-addressed artifacts (target/candidate png + the candidate's binary `.svg`
+ parsed code when on-policy). Verified record: a 2-image judge call lays out as
`[text 30, image_soft 282, text 5, image_soft 282, text 28] = 627` tokens
(= reported total), text ids beginning `[2, 105, …]` (BOS + turn), out ids
ending `106` (end-of-turn). This is the recording convention the oldest video
trials got, extended to exact token-index fidelity.

**(b) On-policy SVG-render exemplars** (operator's spec: real frames × SVG
outputs that disagree across SSIM/MSE/semantic). Built from the amongus
candidate DB incl. the dissociation pair — a faithful render with the *worst*
MSE (0.129) labeled 4 "judge subject, not pixels." Exemplar-target frames are
held out of the test set (no leakage); floor anchor uses a non-amongus frame.
Run: `output_data/svg_runs/judge_calib_onpolicy/` (120 transcripts, 27 artifacts).

| variant | floor | rankρ vs gt | discrim | imgs/call | cache hits Δ | wall |
|---|---|---|---|---|---|---|
| V0 holistic | 0.31 | 0.58 | 1.96 | 2 | 22 | 48 s |
| V1 rubric (0-shot) | 0.00 | 0.75 | 2.42 | 2 | 40 | 36 s |
| 1-shot | 0.00 | 0.76 | 2.54 | 4 | 79 | 48 s |
| 2-shot | 0.00 | 0.84 | 2.54 | 6 | 119 | 48 s |
| 3-shot | 0.00 | **0.87** | **2.67** | 8 | 159 | 48 s |
| 4-shot | 0.00 | 0.87 | 2.42 | 10 | 199 | 50 s |

**Findings:**
1. **On-policy exemplars substantially beat controlled-degradation ones** —
   rankρ 0.75→**0.87** at 3-shot here vs 0.39→0.46 in §8. The operator's
   instinct was right: real SVG renders that dissociate SSIM/MSE/semantic teach
   the judge more than synthetic perturbations. And here the k-shot curve is
   **monotonic** (0.75→0.76→0.84→0.87), plateauing at 3 shots (4th adds nothing).
2. **Rubric eliminates floor-bias** (0.31→0.00) — confirmed a second time.
3. **No multi-image ceiling even at 10 images/call**; cache amortization holds
   (hits 22→199, wall flat ~48 s as imgs/call went 2→10).

**Promote:** the anchored rubric + ~3 on-policy dissociation exemplars is the
calibrated default for the `elicit.py` judge. Remaining for a study-grade judge:
repeated trials (error bars) and an external referee (§5).

---

## 10. Calibrated judge back in the multi-turn loop (2026-05-28)

`tools/svg_elicit/judge.py` packages the calibrated judge (rubric + on-policy
k-shot, loaded from `amongus_onpolicy_exemplars.json`); `elicit.py` now uses it
for both in-loop feedback and best-of-N selection (`--primary-metric judge`).
Re-ran the multi-turn harness on a HELD-OUT pose (amongus frame_08 — the
full bent-twerk; frame_00 was the exemplar target). Run:
`output_data/svg_runs/elicit_kshot_judge/`.

| mode | judge traj | SSIM traj | MSE traj | best |
|---|---|---|---|---|
| svg | 2,2,2 | 0.630→0.624→0.663 | 0.077→0.080→0.058 | judge 2.0 |
| python | 3,2,2 | 0.616→0.601→0.623 | 0.083→0.093→0.082 | judge 3.0 |

**Findings:**
1. **The judge now discriminates in-loop** (2–3, not floored-1 nor saturated-5)
   and **ranks the modes correctly**: python (3) > svg (2). Visual inspection
   confirms it — the python output renders a recognizable crewmate *with a
   clear outlined visor* + gradient body; the svg output is a near-featureless
   yellow blob (no visor, no legs). The judge's harsher score for frame_08 vs
   frame_00 (4–5) is also fair: frame_08 is the hard bent pose and both renders
   dropped the splayed legs.
2. **Real structured geometry.** Outputs use cubic-Bézier body silhouettes,
   radial-gradient + feGaussianBlur shading, background/floor composition paths;
   python mode used loops. Not the flat-stroke squiggles of the restricted
   harness.
3. **Iteration was flat on the judge metric here** (judge picked iter-0 both
   modes). Unlike the Linux frame (python SSIM 0.16→0.62 across iters, §7), the
   model did not recover frame_08's missing legs/pose through refinement — so
   "multi-turn helps" remains target/mode-dependent, and the judge correctly
   reflects the non-improvement rather than rewarding churn. Note judge and
   pixel metrics disagreed on best-iter for svg (judge=iter0; SSIM/MSE=iter2) —
   the multi-measure record preserves that disagreement.

---

## 11. Batched-video regime (2026-05-28) — what it's like, and three pitfalls

`tools/svg_elicit/batch_run.py` re-runs the oldest video-trial workload through
the upgraded harness: 2 freshly re-ripped YouTube sources (`KCrfDHS_YUw` 37.8 min,
`Vore-4VZ5rs` 29.1 min) → ffmpeg extract → diverse-sample → N batches of K
concurrent single-shot SVGs. Three findings dominated over the throughput
numbers:

1. **B=8 concurrent *multimodal* generation deadlocks the engine.** 8 streams
   admitted, `total_steps` frozen, no forward progress. Earlier runs were
   effectively B≤2; the old `svg_concurrent_bench` K=8 used lighter single-image
   text prompts. **B=4 progresses.** A real batched-multimodal ceiling.
2. **Throughput here is host-memory-bound, not kernel-bound.** With the host at
   **7% free memory** (heavy compressor/swap), decode fell to ~8 tok/s — the
   documented `kv_pool_vs_model_size` page-fault ~50× slowdown (`vm_stat`
   confirmed; the memory note's "check vm_stat before suspecting kernels" was
   right). The regime needs free host RAM to run at speed.
3. **The diverse-frame sampler had a pitfall, now fixed.** First cut
   (farthest-point over aHash) *over-selected* black frames — the dark video has
   ~10 near-black frames, and FPS prioritizes outliers, so the VLM kept getting
   black inputs and correctly refused ("the reference image is completely
   black / not encoded correctly"). Fix (operator's framing): score variance
   **relative to the pool sampled so far** over a random-interval probe, and
   **collapse degenerate (low-σ) frames to a single canonical "blank" mode**
   (aHash is just noise on a near-uniform image). Result: **exactly one black
   frame ever** (not zero, not many) + diverse content, verified stable across
   seeds (64/64, mean-NN-Hamming ~9.7). This is the "avoid over-polling similar
   frames" requirement done right — it also rejects near-duplicate consecutive
   frames.

Status: sampler fixed + verified (no LM). The full 64-frame LM gallery +
per-batch throughput table is deferred until the host has free RAM — at 7% free
/ 8 tok/s the engine isn't the bottleneck and the numbers would measure swap,
not the regime.
