# Quant search — project goals and inertia

**Companion to:**
- `docs/handover_2026_05_05_quant_search.md` — the post-mortem of what went wrong on 2026-05-04 → 05-05.
- `tools/quant_search/README.md` — the tool surface and current status.
- `docs/quant_search_execution_plan.md` — the V1 plan being executed.

This document captures **why** the project exists and **why the bench has the shape it has** — not what was run on any particular day. The two are different questions and tend to drift apart over long sessions; this is the checkpoint for the second one.

---

## What this whole project is

There is an inference engine here — a vLLM-shaped Metal-backed runtime for Gemma 4 26B A4B on Apple Silicon (M5 Max, 128 GB unified memory). The engine is **not** a single-stream llama.cpp speedup; that comparison is explicitly not wanted, because llama.cpp doesn't do what this engine is for. **What this engine is for** is parallel decoding from shared prefixes at batchsize > 1 — multiple inference streams that share 75–80 % of their KV state, dispatched together through custom Metal kernels for paged attention, MoE grouped matmul, Q-chunked prefill, and tiled GEMM. The engine inherits I/O conventions, GGUF format, and the tokenizer from llama.cpp, but the entire forward pass and KV cache update loop is its own.

The hardware shapes the engineering: 128 GB unified memory means **memory budget is never the binding constraint** for any practical quantization. The binding constraint at production shapes is the per-layer weight stream into Metal threadgroups during autoregressive decode — that's the AR-decode latency bottleneck. So when the engine quantizes, it's not for storage; it's to reduce the bytes-per-element streamed off DRAM into the GPU per token.

## What "quant search" is for

Standard GGUF tooling answers a question like *"what's the smallest model that still loads and produces acceptable quality on a phone?"* — smallest-model-at-fixed-quality. That is the wrong question for this substrate. The right question is the inverse: **"what's the fastest model on M5 Max that maintains acceptable quality?"** — fastest-at-fixed-quality. Quant search is the tool meant to answer that question by sweeping a grid of GGUF quantization configurations, measuring each one along two axes — production-shape throughput on one, quality degradation versus the bf16 reference on the other — and producing the Pareto frontier as the artifact.

The framing is substrate-aware: a config is on the frontier *for this engine, with these kernels, on this hardware*. The artifact generalizes — re-run when kernels change, when Apple silicon advances, when the model class shifts — but the numbers are not a universal claim about Q4_K vs Q5_K vs Q8_0; they are claims about Q4_K on this engine on M5 Max with the kernel zoo as it stands at the commit you ran against.

## Why the bench has the shape it has

Two principles fall out of the substrate framing.

**First principle: production-shape, always.** The engine has a kernel zoo with `B_TILE ∈ {1, 2, 4, 8}`, each cell compiled separately. Production runs at concurrency activeB ≈ 8. So bandwidth must be measured at activeB = 8 — measuring at activeB = 1 would give a number from a *different kernel cell* and is a category error analogous to a physics simulator running 1/10 speed: not slow-correct, just wrong. Quality also has to be measured the way the model is actually used in production — generation-based, on-policy outputs, multimodal where multimodal matters. SVG-MSE (multi-turn vision-feedback) is the centerpiece because it's the canonical edge-deployment workload for this engine.

**Second principle: bridge is the contract.** The bridge is the API server production talks to. The set of endpoints production uses defines the contract. Benchmarks are clients of that contract, the same way every other OpenAI-compatible runtime client is a client of OpenAI's API. If a metric is awkward to compute through that contract, the answer is to measure it differently, accept that it's not available, or open an explicit conversation about extending the contract — not to silently add a "for-bench-only" endpoint.

## How the bridge extends

Two things stay constant about this engine's bridge:

1. **It exposes one inference contract** — `/v1/chat/completions` and its companions (`/health`, `/v1/tokenize`, `/v1/models`). Production talks to those. Benchmarks talk to those. Any other client talks to those.

2. **It runs over a kernel zoo with deep specialization** — different cells for different `B_TILE`, different formats per tensor class, different swizzlers per quantization shape. The dispatcher inside the engine decides which kernel to call based on the input shape and the loaded weights' format. **That dispatcher is the engine's reason for existing.**

When a new feature is genuinely useful — say, logit capture by top-p instead of top-K, or returning logits from positions the sampler didn't pick, or returning embeddings — the right shape of patch is small:

- **A new optional parameter on `/v1/chat/completions`.** Multiple commercial APIs (Qwen, DeepSeek) ship `logprobs` / `top_logprobs` / `return_logits` as request parameters on their chat completions endpoint. That's the precedent. The existing endpoint already returns logprob structure when `logprobs=true`; widening the cell doesn't fork the API.
- **A three-to-six line patch at the end of the decoding head function** inside the existing kernel, conditional on the new parameter. Logit capture is a tiny tail of a forward pass that already happens; capturing it is reading values out of registers that already hold them.

That patch composes with every kernel cell the dispatcher might pick. Q4_K MoE-up at `B_TILE=4` still gets the logit capture. Q5_K_M dense at `B_TILE=8` still gets it. Q8_0 with paged attention still gets it. The engine's specialization-by-shape stays intact because we did not reimplement the engine — we extended the leaf of one kernel.

The **wrong** shape of patch is large, and it's what the rejected endpoints were:

- A parallel `/v1/eval/teacher_forced` running a *different* prefill path with *its own* batching assumptions, not riding the kernel zoo's dispatcher.
- A parallel `/v1/completions` with `echo=true` handcoding a logprob-extraction codepath instead of widening `/v1/chat/completions`.
- A parallel `/v1/render` exposing tokenizer state because client-side chat-template guessing got dinged — the right answer was to use the production tokenizer round-trip via `/v1/tokenize`, which already existed.

Each of those endpoints duplicated infrastructure the engine already had, but did so **without the kernel dispatcher's shape specialization**. Two consequences fall out:

1. **The numbers measure a parallel inference path, not the one production runs.** The whole point of running an 8–10 hour quant search is to characterize production behaviour. Bench-only paths characterize themselves.
2. **Every parallel path is a permanent integration tax.** Q5_K_M support has to land in it separately. MoE has to land in it separately. Every kernel zoo follow-up has to land in it separately. A bench endpoint that ships once but gets maintained forever is worse than no bench endpoint at all.

The principle generalizes: **add parameters, not endpoints; patch leaves of kernels, not parallel backends.**

## On lm-eval-harness specifically

The lm-eval-harness ontology — MMLU as multiple-choice, GSM8K as math-extraction, HellaSwag as continuation-ranking, perplexity as sliding-window log-likelihood — is fine. It's a reasonable taxonomy of what "quality" means for an autoregressive model. We can want it. We can use it.

What's not fine is inheriting lm-eval-harness's *implementation patterns*. The reference implementations in that repo are research-grade: often single-stream, often blocking on Python loops, often assuming a private model handle rather than an API client. That's *their* quality bar. **It is not ours.** There's nothing about evaluating a model on MMLU that requires sequential dispatch, or a private bench endpoint, or a side-channel into raw logits. MMLU is "give the model a question and four options, take its first token, score it." That fits in `/v1/chat/completions` with `max_tokens=1` and `logprobs=true`, batched at `activeB=8` like every other workload, dispatched through the same kernel zoo as every other inference. The methodology is fine; the implementation is ours to write at our quality bar.

The recurring failure mode this cleanup post-mortems was *not* "lm-eval methodology was imported." It was **"lm-eval implementation patterns were imported as if they were the methodology."** Teacher-forced echo+logprobs is not the only way to compute KL — it's just how the reference repo does it. Per-position log-probability over the full vocabulary is a degenerate generalization of `top_logprobs=K` with K → vocab_size, achievable as a parameter widening, not a parallel endpoint. The *names* of the metrics are welcome; the *plumbing* the reference uses to compute them is not, because that plumbing presupposes none of the kernel-zoo specialization that defines this engine.

If, in the future, we genuinely need a metric that the existing parameter surface can't produce — e.g. full-vocab logprobs at every position of a long context — the path is: identify the smallest parameter widening on `/v1/chat/completions` that exposes it, write the leaf-of-kernel patch, ship it as a contract change that production can also use. The path is *not* to add `/v1/eval/whatever`.

## What's left to be true

The actually-requested suite, with all the unauthorized-implementation methodology stripped out, reduces to four things, all driven through `/v1/chat/completions`:

- **MMLU** scored by argmax over the model's first-token output (multiple choice via chat completions, not via teacher-forced log-prob ranking).
- **GSM8K** scored by extracting the numeric answer from generation (chat completions, not echo+logprobs).
- **SVG-MSE** scored by rasterizing the model's SVG output and computing pixel-MSE against a reference (existing toolcards-on-chat-completions path).
- **Tok/s** measured at `activeB = 8` production shape (concurrent chat completions).

The bridge surface that supports all four is exactly four endpoints: `/health`, `/v1/tokenize`, `/v1/models` (+ `/models`), `/v1/chat/completions` (+ `/chat/completions`). That's the cleanup's end state. From here the actual run becomes possible:

1. Pick configs (Q4_K_M and Q5_K_M are locally available; more need `llama-quantize` materialization).
2. Establish the bf16 baseline — the Y-axis of the Pareto plot is meaningless without it.
3. Fix the saturation-probe in `tools/quant_search/workload.py` so concurrency is hardcoded to the B=8 cell rather than ramped from B=1.
4. Fix per-harness JSONL persistence so killed-mid-run results aren't lost.
5. Run.

Each measurement then reflects the production engine and the production kernel dispatcher — not a bench-modified parallel path. That's the artifact this project is here to produce.

## Inertia worth preserving

A few framing points that are easy to lose between sessions:

- **Substrate-aware Pareto, not universal Pareto.** Numbers are claims about *this engine on this hardware at this commit.* They generalize through methodology, not through portability of numbers.
- **Bandwidth-bound, not memory-bound.** Quantization is for streaming bandwidth, not storage. Cost models look like `latency = bandwidth_term(bpw) + dequant_term(format) + fma_term`.
- **Fastest-at-fixed-quality, not smallest-at-fixed-quality.** Inverts the standard GGUF tooling target.
- **Multi-turn multimodal is the centerpiece.** SVG-MSE > MMLU > GSM8K > Tok/s in priority order, because edge-deployment-shaped workloads are the canonical use case.
- **Bridge is the contract; benchmarks are clients.** Not just a coding rule — the only way the numbers mean anything.
- **Add parameters, not endpoints. Patch leaves of kernels, not parallel backends.** The kernel dispatcher is the engine; reimplementing around it discards the engine.
- **lm-eval ontology is welcome; lm-eval implementation patterns are not.** Quality bar is ours, not theirs.
