# API compatibility — what we are, what we aren't, and why

This server presents an OpenAI + llama.cpp-server + koboldcpp compatible
surface. It is NOT trying to be a drop-in for every endpoint either of
those projects expose. The sorting rule:

- **First-order features** are things we actually built the engine to
  do well — **streaming batched AR decode across B=4 sessions, live
  control-vector injection, detector-trigger-effector composition,
  multimodal soft-token prefill, content-addressed prefix caching**,
  and the measurement/debug endpoints that let you inspect those
  live features from an HTTP client.

  These work, they're tested, they're what the metal kernels were
  written for, and they're what makes the project worth running.

- **Second-order features** are parts of the OpenAI/llama.cpp/kobold
  surface that make us look like a normal LLM server to off-the-shelf
  clients. We implement enough to pass typical frontend handshakes
  (SillyTavern, LibreChat, open-webui, kobold-lite) so you can plug
  them in without touching client code.

- **Politely ignored features** are parts of those APIs that either
  don't apply to causal transformer serving, reflect obsolete research
  patterns, or have active mechinterp-level reasons to NOT be
  reintroduced. We accept the relevant body fields or endpoints and
  log a warning; we do not emulate the behavior.

## The politely-ignored list

**Embedding-model endpoints** (`/v1/embeddings`, `/rerank`, dedicated
BERT-style bi-encoders). Exposing a separate embedding model's hidden
states is strictly inferior to exposing a real language model's —
you can do mechinterp, distillation, subliminal-learning-style probes,
and feature discovery on a causal LM's residuals, and none of that on
a frozen embed head. OpenAI is deprecating their embedding endpoints
one-at-a-time for adjacent reasons. This project takes the position
that dedicated embed models are obsolete and won't be first-class.
Anyone who wants to fork and implement the 6k lines of Metal kernels
needed to run a modern BERT at high FLOPs efficiency can do so — if
their fork passes tests, we'll pull it. Nobody will.

**`top_k` sampling.** Modern consensus: `min_p` (threshold at a
fraction of the top token's probability) dominates `top_k` on every
metric that matters — `top_k` with small k hard-cuts the distribution
and reliably produces degenerate-loop fixed points; `min_p` degrades
gracefully to greedy-ish as needed. We accept `top_k` in the body and
log a one-line "ignoring" warning. We implement `min_p`.

**Fill-in-middle (`/infill`).** Gemma-4-a4b isn't FIM-trained; the
endpoint returns garbage for this model. Accept the request, return
a stub "not applicable for this backend".

**Audio transcription (`/v1/audio/transcriptions`).** Needs a Whisper
model loaded. Not in scope.

**Image generation (`/v1/images/generations`).** Stable-diffusion
endpoint. Not in scope.

**Runtime LoRA swap (`/lora-adapters`).** We don't hot-swap adapters;
we do hot-swap control vectors at every request, which is strictly
more flexible and doesn't need the disk I/O.

**`response_format` JSON mode / `tools` / `tool_choice`.** Structured
decoding requires grammar enforcement on the sampler. Not first-order
for our research workloads. Accept the fields, log, ignore. May be
implemented via the existing control-vector path (gated on JSON
structure detectors) as future work.

## The first-order list (committed)

- **Batched AR decode at B=4**, streaming via SSE, content-addressed
  KV cache across sessions. This is what the Metal kernels exist to
  do; the API must let clients exploit it.

- **Multimodal input** — `image_url` content parts in the OpenAI
  `/v1/chat/completions` message format. `/v1/media/extract` for
  clients that want to cache soft tokens client-side.

- **Real streaming usage telemetry** — per-frame deltas + final
  aggregate usage (prompt_tokens, completion_tokens, total_tokens).

- **Server-side stop sequences** — checked after each streamed token.
  Required for off-the-shelf roleplay clients; no way to fake this.

- **Seeded RNG per session** — deterministic replay for test
  harnesses.

- **`min_p` sampling** in addition to temperature.

- **`logit_bias`** per-session, applied before softmax. Used for
  safety-token biasing, symbol steering, and creative work.

- **Abortable generation** — `/api/extra/abort` drives a session
  pause/close by request id.

## What being "a llama.cpp / kobold server" means here

If you point a client at this server using either project's URL
configuration, you get:
- All the shape fields they expect (`/health`, `/props`, `/slots`,
  `/tokenize`, `/detokenize`, `/completion`, `/api/v1/generate`,
  `/api/v1/info/...`)
- `/v1/chat/completions` with the full OpenAI body field set accepted
  (unimplemented fields log warnings)
- Streaming SSE with normal opening/delta/close frames + kobold's
  extended streaming at `/api/extra/generate/stream`
- Abort semantics that actually stop generation engine-side

You do NOT get:
- BERT embedding endpoints
- Runtime LoRA swap
- `top_k` enforced (body field is accepted; logged; ignored)
- FIM, audio, image-gen

You DO get things they don't:
- Live activation steering during generation
- Residual/Q/K/V capture endpoints for mechinterp
- Direct KV-cache injection at session creation
- Per-session control-vector attachment
- Multi-session batched prefill sharing KV pages by content hash

## When to fork rather than file an issue

If you want a feature from the "politely ignored" list, fork and
implement it. If it passes the existing test suite without
regressing the first-order features, file the PR and we'll pull it.
The bar is tests + no regressions, not a philosophical agreement
about what LLM serving should look like in 2026.
