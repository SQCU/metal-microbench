# Quant search execution plan — staging the off-device run

**Companion to:** `tools/quant_search/README.md` (the tool surface) and `docs/kernel_zoo_outcome_report.md` (the kernel arc this builds on).
**Goal:** drive a V1 quant Pareto-frontier search to completion against a 7×RTX-5090 cloud fleet, with all validation gates explicit and falsifiable.
**Status as of this doc:** scaffolding shipped; oracles smoke-tested where the M5 bridge can validate them; cloud deploy + actual search execution pending.

---

## Validated vs unvalidated state, by component

| Component | Built | M5 smoke-tested | E2E validated | Notes |
|---|---|---|---|---|
| `store.py` (Pareto + SQLite) | ✓ | ✓ | ✓ | Pareto sort verified on synthetic 3-point set |
| `quant_driver.py` (GGUF materialization) | ✓ | — | — | Calls `llama-quantize`; built but not exercised yet |
| `oracles.measure_tok_s` | ✓ | ✓ | ✓ | M5 only; baseline measurements in record |
| `oracles.measure_kl_div` (M5 LM_KL_REF path) | ✓ | ✓ | ✓ | Uses `forward_graph` binary; M5 only |
| `oracles.measure_svg_mse` (toolcards) | ✓ | ✓ | ✓ | Standalone toolcards runner verified |
| **`oracles.measure_perplexity`** | ✓ | ✗ | ✗ | Needs `/v1/completions` with `echo`; **llama-server only** |
| **`oracles.compute_kl_baseline` + `measure_kl_oai`** | ✓ | ✓ | partial | Self-comparison validates round-trip; FP16-vs-quantized KL not yet measured |
| **`oracles.measure_mmlu`** | ✓ | ✓ (n=5) | partial | Margin + accuracy correct on tiny subset; full n=1000 not yet exercised |
| **`oracles.measure_hellaswag`** | ✓ | ✗ | ✗ | Loader works; eval needs `/v1/completions` |
| `torch_dequant.py` (GPU-native dequant) | ✓ | ✓ (CPU) | partial | 6/8 formats bit-exact vs gguf-py reference; Q5_K/Q6_K pending Q5_K_M/Q6_K GGUFs |
| `cloud/Dockerfile` (CUDA 12.8 + Blackwell archs) | ✓ | — | — | Not built or run yet |
| `cloud/yeet_to_runpod.sh` (deploy) | ✓ | — | — | Bash-syntax-clean; not exercised |
| Multi-bridge dispatcher in `search.py` | ✗ | — | — | **Still single-bridge; needs ~30 LOC of `concurrent.futures` for fan-out** |
| Cost model (M5 tok/s predictor) | ✗ | — | — | Not started |
| FP16 GGUF + KL reference baseline | ✗ | — | — | Script 02 not run yet |
| V1 grid of materialized GGUFs | ✗ | — | — | Script 03 not run yet |
| **M5 kernel coverage for all V1 formats** | partial | partial | — | See "Kernel coverage map" below — Q5_K and Q6_K kernels currently missing |

## Kernel coverage map (M5 substrate)

For each V1 grid format, what kernels exist on the M5 production engine and what's missing as of 2026-05-01 (post-prefill-rewire commits `bcfa0fd`/`c76eb68`):

| Format | Prefill dense matmul | Prefill MoE mul_mm_id | AR dense GEMV | AR MoE GEMV |
|---|---|---|---|---|
| **Q4_0**  | ✗ (kernel needed) | ✗ (kernel needed) | ✓ `dense_gemv_q4_0_v4` | ✓ `moe_gemv_q4_0_v3` |
| **Q4_K**  | ✗ (kernel needed) | ✓ `prefill_mm_id_q4K_swiz` | ✓ `dense_gemv_q4k_v4` | ✓ `moe_gemv_q4k_v11_b{1,2,4,8}` |
| **Q5_1**  | ✗ (kernel needed) | ✓ `prefill_mm_id_q5_1_swiz` | partial (no dense path used in prod) | ✓ `moe_gemv_q5_1_v11_b{1,2,4,8}` |
| **Q5_K**  | ✗ (kernel needed) | ✗ (kernel needed) | ✗ (kernel needed) | ✗ (kernel needed) |
| **Q6_K**  | ✗ (kernel needed) | ✗ (kernel needed) | ✗ (kernel needed) | ✗ (kernel needed) |
| **Q8_0**  | ✓ `prefill_mm_q8_0_swiz` | ✗ (kernel needed) | ✓ `dense_gemv_q8_0_btile_b{1,2,4,8}` | n/a (Gemma-4 doesn't use Q8_0 MoE) |

Of the 9 V1 grid configs, only **Q8_0-uniform partially runs** on M5 today (its MoE path is missing — Gemma-4's Q4_K_M reference doesn't use Q8_0 MoE so production never needed it). The other 8 configs need kernel work before M5 calibration can happen.

This is the gap closure the rest of this update tracks. Each missing kernel is mechanical: copy the existing simdgroup-matmul template, swap the per-format dequant function and block struct. ~1-2 hours each, validated against a CPU reference at fp16 floor (RMSE ≤ 0.001).

## Substrate priors for the cost model

This session (commits `ea03163`, `c76eb68`) produced concrete kernel-level TFLOPS measurements for the formats currently in production, at saturated batch on M5:

| Format | Peak TFLOPS @ saturated batch | Per-element dequant complexity | Block geometry |
|---|---|---|---|
| Q8_0 dense (swizzled simdgroup matmul) | ~13.7 | one int8 multiply, no bit-packing | 32 elts / 34 B |
| Q4_K MoE (swizzled mul_mm_id) | ~5.0  | bit-unpacking (4-bit) + paired 6-bit scales + min-subtract | 256 elts / 144 B |
| Q5_1 MoE (swizzled mul_mm_id) | ~4.35 | 5-bit bit-math (4-bit + per-elt qh bit) | 32 elts / 24 B |

The rank ordering — **Q8_0 ≫ Q4_K > Q5_1** — tracks dequant ALU cycles per element, not bandwidth. This is a substrate prior for the V1 cost model: predict Q8_0 fastest, then Q4_K, then Q5_1, with Q4_0 likely between Q4_K and Q5_1 (simpler scale than Q4_K but more bits than Q4_K's K-block sharing). Q5_K and Q6_K don't have measurements yet — bench them once the kernels land.

Use this rank ordering as a Bayesian prior on the linear-regression cost-model coefficients in Phase 4.2: any per-format coefficient that violates the ordering is a red flag (kernel issue, not a quant property).

## Stale-data alert: bench grid axis bug (commit `ea03163`)

Any TFLOPS numbers from `q4k_mma_bench.swift` measurements **before commit `ea03163`** are inflated by a factor of `gridX_dim_count` (the bench dispatched more TGs than the kernel needed; FLOP accounting credited the wasted work). Production dispatch in `bootstrap.swift::encMatMulQ80SwizPrefill` was always correct, so production tok/s measurements via `forward_graph` / dylib are **unaffected**.

If any numbers in `tools/quant_search/results.sqlite` came from the standalone bench harness (none should — `oracles.measure_tok_s` goes through the real engine path), invalidate them. Re-measurement uses `kernel_version` set to "prefill-mm-swiz-2026-04-30+post-bench-fix" to namespace the corrected runs cleanly.

## Cache-discipline note for `oracles.measure_tok_s`

The bridge has a content-hash prefix cache (`PageManager`). Repeated identical prompts hit the cache as warm-prefill, returning sub-100-ms TTFT for what should be cold work. The plan's "warm 3-trial median" is fine if "warm" means "kernel JITed and L2-warmed" — but **not** if it means "prefix-cache hit" (a different thing). For unambiguous cold-prefill measurements, either:
- Use a unique nonce in each prompt to defeat the prefix cache, or
- Restart the bridge between configs (already implied by Phase 4.1's "restarting the bridge each time"), or
- Hit a cache-eviction endpoint between trials

Document which approach `measure_tok_s` uses; the resulting tok/s numbers should be unambiguously labeled as cold or warm.

---

## Phased execution

Each phase has a primary artifact, a validation gate, and a clear stop condition if the gate fails.

### Phase 0.5 — Teacher-forced HTTP endpoint on the M5 bridge

**Why this phase exists:** the multi-token teacher-forced quality oracles (perplexity, HellaSwag) currently assume llama-server's `/v1/completions echo=true logprobs=N` shape. The metal-microbench engine has its OWN teacher-forced logit codepath (already used by `runLmPrefillValidate` for KL validation against FP16 oracles), but it's only accessible at the binary level via env vars — not over HTTP.

Wiring this codepath to an HTTP endpoint:
- Lets perplexity + HellaSwag oracles validate locally on M5 BEFORE we trust llama-server's API surface
- Removes the dependency on llama-server's `echo` semantics matching expectations
- Gives us a substrate-pinned ground-truth path for cross-checking cloud-derived numbers
- Demotes cloud parallelization from "critical path" to "optional acceleration"

**Implementation breakdown** (concrete; the engine already has the GPU work):

**Sub-task 0.5.1: FFI export.** New C-ABI entry point in `ffi.swift`:
```
gemma_eval_teacher_forced(
    tokens: UnsafePointer<UInt32>, n_tokens: Int32,
    top_k: Int32, out_per_position: UnsafeMutableRawPointer
) -> Int32
```

Body: populate `pre_input_tokens` and `pre_q_positions` buffers from the input array, call `buildPrefillCB(w, qLen: n_tokens, fullPrefillLogits: true)`, wait, copy slot-0 rows of `pre_logits` to a heap buffer, for each position extract (a) the logprob of the actual token at that position, (b) top-K alternatives. Write structured output bytes to `out_per_position`.

Wire into the same FFI surface as the existing `gemma_status` / `gemma_submit_image_bytes` / etc. ~50-80 LOC.

**Sub-task 0.5.2: Bridge HTTP route.** New endpoint in `server/bridge.py`:
```
POST /v1/eval/teacher_forced
  body: {input: str, top_logprobs: int = 0}
  → {
      tokens: [int, ...],
      per_position: [
          {token_id, token, logprob, top_logprobs?: [{token, logprob}, ...]},
          ...
      ]
  }
```

Tokenizer call (already plumbed through `g.tokenize`), FFI call to `gemma_eval_teacher_forced`, JSON serialization of the result. ~50 LOC.

**Sub-task 0.5.3: MAX_Q_LEN handling for long sequences.** Current `MAX_Q_LEN` (8 or 32 depending on which prefill rev shipped) caps single-pass teacher-forcing. For perplexity over a 50K-token corpus, need stride-windowed evaluation:
- Window of length `L = MAX_Q_LEN`, stride `S = L/2`
- Each window contributes log-probs only for positions `[S, L)` (first `S` tokens are burn-in, providing context)
- Total compute is `2× corpus_tokens / L` prefills

Either implement chunking inside the FFI export (more complex, hides the chunking from callers) or expose single-window calls and chunk in the oracle (simpler, but oracle becomes substrate-aware). I lean toward chunking in the oracle — it's where the corpus-level orchestration lives anyway.

**Sub-task 0.5.4: Update oracles.measure_perplexity and measure_hellaswag.** Add a small dispatcher: prefer `/v1/eval/teacher_forced` if the bridge advertises it (via a flag on `/health`), fall back to `/v1/completions echo=true` otherwise. Both code paths produce the same oracle output shape. ~30 LOC of changes.

**Validation gate:** `measure_perplexity('http://127.0.0.1:8001')` returns a sensible perplexity (~5-15 for current Q4_K_M Gemma-4) on WikiText-2. Compare against an off-engine reference (e.g., `llama-cpp-python`'s perplexity utility on the same GGUF) to confirm absolute numbers are within ~10%. If they diverge significantly, the teacher-forced eval has an off-by-one in token alignment.

**Stop condition:** if the FFI returns NaN/Inf logprobs at any position, the prefill path has a numerical issue with the test corpus (most likely: tokens in the corpus that the engine's tokenizer handles differently than the oracle's expectation). Diagnose by inspecting per-position outputs at the boundary.

**Wall time estimate:** 1-2 focused sessions of engine work. The GPU work is done; this is wrapper code.

**What this phase deliberately defers:** stride-window optimization, batched multi-prompt eval (running B parallel teacher-forced prompts in one prefill), KV-cache reuse across calls. All can come later if HTTP overhead becomes the bottleneck — for V1 with 9 configs × few-thousand-token corpora, single-call teacher-forcing is fast enough.

### Phase 1 — Build artifacts the search depends on

**Sub-task 1.1: FP16 base GGUF.** Run `scripts/02_generate_fp16_base.sh`. Produces `~/models/gemma-4-a4b-quant-search/gemma-4-26B-A4B-it-fp16.gguf` (~52GB).

  **Validation gate:** load the produced GGUF with `llama-cpp-cli` and run a single-token completion against a canonical prompt ("The capital of France is"). Output should be "Paris". If not: the convert step lost information; rerun with `--outtype f16` instead of bf16-default behavior.

  **Stop condition:** if `llama-quantize` rejects the GGUF when we try to derive Q4_K_M from it, the source has a metadata problem (missing chat template, missing rope-scaling config). Diagnose by inspecting the GGUF metadata via `gguf-py`'s `dump_gguf.py`.

**Sub-task 1.2: V1 grid materialization.** Run `scripts/03_materialize_grid.sh` with the FP16 base. Produces 7 quantized GGUFs (Q4_0, Q4_K_M, Q4_K_S, Q5_1, Q5_K_M, Q6_K, Q8_0).

  **Validation gate:** smoke each GGUF by running `llama-cpp-cli -m <path> -p "Hello" -n 5`. Each should produce non-garbage output. Catch: if any GGUF produces gibberish or NaNs, the quantization pipeline broke for that format.

  **Stop condition:** any GGUF that fails the smoke test gets removed from the search grid. Document why (e.g. "Q4_0 produces NaN on this model — known issue with Gemma-4's specific activation distribution").

**Sub-task 1.3: KL reference distribution baseline.** Run `compute_kl_baseline()` against the FP16 base GGUF (loaded via a temporary local llama-server) on a 100-prompt set. Save the resulting top-K logprobs as `data_cache/kl_baseline_fp16.json`.

  **Prompt set composition:**
  - 25 conversational ("What's the weather like?", "Tell me about your day")
  - 25 factual ("What is the capital of …?", "Who wrote …?")
  - 25 reasoning ("If a train travels …", "Compare X and Y")
  - 25 generative ("Write a haiku about …", "Describe a sunset")
  Diverse enough that KL signal isn't dominated by a single domain.

  **Validation gate:** the baseline should NOT have any zero-distribution entries. Any prompt that produces empty top-K is a bug; fix loader or prompt before accepting.

  **Stop condition:** if the FP16 model itself produces nonsensical outputs on the prompt set, the GGUF conversion lost essential metadata; back to 1.1.

### Phase 2 — Smoke-test cloud infra on a single GPU

**Sub-task 2.1: Build the worker image.** `docker build -t metal-microbench/quant-worker:latest -f tools/quant_search/cloud/Dockerfile .` on a host with Docker + CUDA 12.8+ runtime.

  **Validation gate:** image builds without errors. If `nvcc` complains about SM_100 or SM_120, the base image is too old or CUDA version mismatched.

**Sub-task 2.2: Single-worker local validation.** On a development box with at least one Blackwell-class GPU (or whatever's locally available — even an RTX 4090 will smoke-test the build), run:

  ```bash
  docker run --rm --gpus '"device=0"' -p 8080:8080 \
      -v ~/models/gemma-4-a4b-quant-search:/models:ro \
      -e GGUF=/models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
      metal-microbench/quant-worker:latest
  ```

  Then exercise each oracle from Python:
  - `measure_perplexity('http://127.0.0.1:8080')` — verifies `/v1/completions echo=true logprobs=1` works on llama-server
  - `measure_kl_oai(...)` — verifies chat-completions logprobs on llama-server (probably already works since same shape as M5 bridge)
  - `measure_mmlu(..., n_samples=5)` — sanity
  - `measure_hellaswag(..., n_samples=5)` — sanity

  **Validation gate:** each oracle produces sensible numbers. Perplexity should be ~5-15 on WikiText-2 for a Q4_K_M Gemma-4 model. MMLU margin should be positive on most questions. KL vs FP16 baseline should be small but nonzero (this is the actual Q4_K_M-vs-FP16 signal we'll see throughout the search).

  **Stop condition (perplexity-specific):** if `/v1/completions` doesn't return logprobs or if `echo=true` is ignored, llama-server's API differs from spec. Alternative endpoint: `/completion` (legacy llama.cpp native) supports the same semantics with a different request shape. Code switch is ~20 LOC.

  **Stop condition (KL-specific):** if FP16-vs-Q4K KL is suspiciously small (~0.001) or large (>1.0), our KL math is wrong. Diagnose: compare against a known reference (e.g., llama-cpp-python's `llama_get_logits` direct output).

### Phase 3 — Cloud deploy + parallel evaluation

**Sub-task 3.1: Provision RunPod fleet.** Run `cloud/yeet_to_runpod.sh runpod 7 ~/models/gemma-4-a4b-quant-search rtx-5090`. This:
  - Provisions a 7×RTX-5090 spot instance
  - rsyncs the V1 grid GGUFs (7 files × ~10GB each = ~70GB)
  - Starts one worker per GPU
  - Writes `workers.json` with the 7 URLs

  **Validation gate:** all 7 workers respond to `/health`. Hit each with a single completion request to verify the model loaded.

  **Stop condition:** if RunPod can't allocate 7 spot RTX-5090s, fall back to a single 8×H100 instance (also fine; H100 has slightly less bandwidth but everything else identical) or a 4×B200 (overkill but available).

**Sub-task 3.2: Multi-bridge dispatcher in search.py.** Currently `search.py` assumes one bridge. Need to:
  - Accept `--workers workers.json` arg
  - Use `concurrent.futures.ThreadPoolExecutor(max_workers=N)` to dispatch evals across the workers in parallel
  - Each (config, worker) pair: pin the worker to that config (one config per worker, sticky), run the full quality bench
  - Aggregate results back into the same Store

  **Code estimate: ~30 LOC** of glue around the existing `evaluate_config_quality` function.

  **Validation gate:** dry-run with the 7 V1 grid configs across 7 workers in parallel. Total wall ~10-15 min for tier 1 (perplexity + KL) on all 7. Compare to "what if we ran them serially" sanity check (~1 hour).

**Sub-task 3.3: Run the V1 grid.** With workers provisioned and dispatcher wired:

  ```bash
  python tools/quant_search/search.py \
      --workers tools/quant_search/cloud/workers.json \
      --kernel-version v11+v11_q4k_q5_1 \
      --kl-baseline data_cache/kl_baseline_fp16.json \
      --tier all
  ```

  Walks the V1 grid (7 uniform configs + 2 mixes = 9 configs, but workers=7), runs all four cloud oracles per config, stores in SQLite.

  **Expected wall:** ~30-60 min for full tier-3 (perplexity + KL + MMLU + HellaSwag + SVG MSE) across 9 configs on 7 workers.

  **Validation gate:** the resulting (perplexity, KL, MMLU, HellaSwag) vectors should monotonically degrade as quantization gets more aggressive — Q8_0 > Q6_K > Q5_K_M > Q4_K_M > Q4_K_S > Q4_0. If the ordering is scrambled, either the quantization is broken or the metric is broken.

### Phase 4 — M5 cost model + final Pareto

**Sub-task 4.1: M5 calibration.** Run `measure_tok_s` on M5 for each of the 9 V1 grid configs (one at a time, restarting the bridge each time). Store the (config, tok_s_at_N=1, tok_s_at_N=4, tok_s_at_N=8) tuples.

  **Wall: ~9 × 5 min = 45 min on M5.**

  **Stop condition:** if a config measures 0.1× expected tok/s, the M5 engine had a kernel issue with that quant format (e.g., we don't have a Q4_0 kernel in metal-microbench). Document and exclude.

**Sub-task 4.2: Fit cost model.** Linear regression mapping (per-tensor format counts) → predicted tok/s, using the 9 calibration points.

  **Form:**
  ```python
  predicted_tok_s = baseline_tok_s + sum(
      coefficient[fmt] * fraction_of_tensors_in_fmt
      for fmt in formats
  )
  ```
  Or possibly a small MLP if the linear form has too much residual. With 9 calibration points, linear is the right starting choice.

  **Validation gate:** leave-one-out cross-validation. Each config's predicted-vs-actual tok/s should be within ±5% of the true measurement. If not, the model needs more calibration points (run V2 partial set) or a non-linear form.

**Sub-task 4.3: Pareto frontier construction.** Combine the 9 quality vectors (cloud-measured) with their 9 cost-model-predicted M5 tok/s values. Compute the Pareto frontier over (composite_quality_loss, predicted_tok_s) where:

  - composite_quality_loss = weighted Mahalanobis distance from FP16 baseline across (perplexity, KL_mean, MMLU_margin, HellaSwag_diff, SVG_MSE)
  - lower-is-better for quality_loss, higher-is-better for tok_s

  **Output artifact:** `data_cache/v1_pareto_frontier.json` listing the configs on the frontier with their (quality_loss, predicted_tok_s, actual_tok_s) tuples.

**Sub-task 4.4: M5 verification on the frontier.** For the 2-3 configs on the Pareto frontier, validate predicted-vs-actual tok/s by re-measuring on M5 (already done in 4.1 for the 9 configs, but the predicted side comes from the cost model).

  **Validation gate:** prediction error on frontier configs ≤ 5%. If it's worse, the cost model has bias toward dominated configs and we need to refit.

---

## Cost + time budget

| Phase | Wall time | Cloud spend |
|---|---|---|
| 1 (FP16 base + grid materialization) | ~45 min (5 + 40) | $0 (CPU work locally) |
| 2 (build + single-worker smoke) | ~30 min | $1-2 (one GPU for 30 min) |
| 3 (RunPod fleet + V1 grid eval) | ~90 min | $5-10 (7 GPUs × ~1.5 hours @ ~$3-5/hour spot) |
| 4 (M5 calibration + cost model + frontier) | ~90 min | $0 (M5 only) |
| **TOTAL V1** | **~4 hours** | **~$10** |

V2 (per-tensor heterogeneous, ~100 configs): same infrastructure, search runs ~3-4 hours wall, ~$30-50 cloud spend.

V3 (smart sampling + learned cost model, ~1000 configs): same infrastructure, ~10-15 hours wall, ~$150-250 cloud spend.

---

## What's still ambiguous and worth pinning down before phase 3

1. **Exact llama-server `/v1/completions` shape.** I've assumed OpenAI-spec behavior with `echo=true`. If llama-server differs (does it return `text_offset`? what's the logprobs token count when `max_tokens=0`?), perplexity + HellaSwag oracles need adjustment. **Mitigation:** sub-task 2.2 catches this in 30 min of single-worker smoke.

2. **Sticky worker-to-config binding.** A worker is pinned to one GGUF (loaded into VRAM). Switching configs means restarting the worker (~30s for model reload). The dispatcher should batch all evals for a single config sequentially on its sticky worker, then move to the next config. **Mitigation:** explicit in the dispatcher design — never bounce a worker mid-config.

3. **Kvar storage between phases.** SQLite stores quality metrics; M5 calibration produces tok_s tuples; cost model needs both. **Mitigation:** `Store.kernel_version` field already namespaces by kernel-build, which is how we'd separate V1 results from V2 results from cost-model-fit data.

4. **Reproducibility.** The whole flow is deterministic IF: same llama.cpp commit (pinned in Dockerfile via `LLAMA_CPP_REF`), same GGUF source, same prompt set. `kernel_version` tag in the store captures the M5 side. Worth committing a `requirements.txt` snapshot of the dev venv too. **Mitigation:** add to phase 1 — pin everything before running 3.

5. **Failure recovery.** If RunPod yanks the spot instance mid-search, we lose in-flight work but the SQLite store is on the M5 side and survives. Resume by calling `search.py` with the same args; it skips configs whose oracle results are already in the store.

---

## What this plan deliberately defers

- **V2 per-tensor heterogeneous search.** Requires custom-quant materializer beyond what `llama-quantize` exposes (currently only the standard "_M" mixes). The right time to write this is *after* V1 confirms the cost model + Pareto frontier infrastructure works end-to-end.

- **Multi-turn coherence + GSM8K + HumanEval oracles.** Currently we have 4 quality oracles. The plan trusts that these are sufficient for V1; if the V1 frontier is unsatisfying or non-discriminative (e.g., all configs cluster), we'd add the multi-turn / capability oracles. But starting with 4 is the right cardinality for the search to be tractable.

- **Cloud oracle ROI quantification.** "How many configs do we need to evaluate before the Pareto frontier converges" — answer is empirical. V1 has 9 configs (small grid). V2 has hundreds. The plan handles V1 directly; V2 needs convergence diagnostics.

- **PyTorch-native end-to-end inference.** We have GPU-native dequant kernels. We don't have a PyTorch-native MoE forward pass that uses them. For V1 this is fine (llama-server is the substrate). For research extensions (per-layer activation L2 distance, custom mixed-precision experiments), we'd want this. Defer until there's a concrete use case.

---

## Definition of done for V1

V1 quant search is "done" when all of the following hold:

1. The 9-config grid has been evaluated on the cloud fleet, all 4 quality oracles produce non-NaN values for every config.
2. The cost model predicts M5 tok/s within ±5% on leave-one-out CV across the 9 calibration measurements.
3. The Pareto frontier contains at least 3 configs and at least one is *not* the standard Q4_K_M shipped baseline. (If the standard baseline dominates, the search added no signal — diagnose: did we actually exercise enough format variety?)
4. A `docs/quant_search_v1_outcome.md` companion document exists summarizing the frontier, the cost model fit quality, and the recommended config(s) for actual production use.

That last document — written *after* the search runs — is the artifact that justifies the work. Without it the run was an experiment, not a deliverable.
