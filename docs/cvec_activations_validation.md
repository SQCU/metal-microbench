# Cvec activations API + validation harness

**Added 2026-05-10.** Exposes per-(`ActiveControl`, AR-step) magnitude
records on the chat-completions SSE stream when a client opts in, plus a
four-test validation harness covering both halves of the
"intervenes-on-residual-streams-without-introducing-bugs" requirement.

## The API surface

### Request (extension fields on `/v1/chat/completions`)

```jsonc
{
  "model": "gemma-4-a4b",
  "messages": [...],
  "stream": true,
  "capture_cvec_activations": true,   // opt-in: routes engine instrumentation
  "controls": [                       // 0 or more cvec applications
    {
      "cvec_id": "<id>",              // must be pre-registered (see below)
      "layer": 12,                    // LM layer to inject at
      "polarity": 1.0,
      "peak_magnitude": 2.5,          // envelope peak; sustain = peak × sustain_level
      "attack": 0.0,
      "decay": 0.0,
      "sustain_level": 1.0,
      "release": 0.0,
      "mode": 0                       // 0=additive, 1=project, 2=transport
    }
  ]
}
```

### Pre-registering a cvec

The bridge process owns its own libgemma registry. Use
`POST /v1/resources/register`:

```jsonc
{
  "kind": "cvec",
  "id": "my-direction",
  "data_b64": "<base64 of HIDDEN×fp16 bytes>"   // HIDDEN=2816 for gemma-4-a4b
}
```

A client process calling `gemma_register_resource` via ctypes registers
into its OWN dylib copy and the bridge will never see the vector. Always
register through the bridge endpoint.

### Response (SSE delta extension)

When `capture_cvec_activations` is set and one or more controls were
active, each non-empty poll emits an extra delta:

```jsonc
{
  "id": "chatcmpl-...",
  "object": "chat.completion.chunk",
  "choices": [{
    "index": 0,
    "delta": {
      "cvec_activations": [
        {"token_position": 28, "layer": 12, "magnitude": 2.5},
        // ... one record per (control, AR-step) pair
      ]
    },
    "finish_reason": null
  }]
}
```

`token_position` is the session-local AR token index. `magnitude` is the
scalar passed to the apply kernel (`envelope(t)·polarity` for additive
mode; the target value for project/transport modes).

## Validation harness

Lives at `tools/cvec_validation/`. Two scripts.

### `baseline_logprobs.py` — off-path bit-identity

Captures the top-K logprobs of the first N tokens generated for a fixed
prompt + fixed seed. Two runs of the same code must produce identical
top-K outputs (bit-equal float32). Three modes:

```bash
python3 tools/cvec_validation/baseline_logprobs.py --record               # write baseline.json
python3 tools/cvec_validation/baseline_logprobs.py --verify               # default off-path check
python3 tools/cvec_validation/baseline_logprobs.py --verify-with-capture  # capture flag ON, no controls
```

The `--verify-with-capture` mode is the "instrumented-but-unused"
regression gate: with the capture flag set but no cvecs registered, the
engine's `recordCvecActivation` early-returns and the queue stays empty,
so the output must remain bit-equal to a run with the flag off. Catches
the failure mode where adding the instrumentation accidentally perturbed
the apply path even when nothing was asked of it.

### `on_path_zero_cvec.py` — on-path correctness

Two phases:

1. **Zero-vector cvec applied at peak_magnitude=1.0**: registers a
   `HIDDEN`-length fp16 vector of zeros, applies it via the `controls`
   field. Expects (a) `cvec_activations` records to flow (`apply path
   reached`), and (b) output content to bit-equal baseline (`zero added
   to residual = no change`).

2. **Unit-vector cvec at peak_magnitude=100**: registers a `HIDDEN`-length
   vector that is 1.0 at dimension 0 and 0 elsewhere; applies with a large
   magnitude. Expects (a) `cvec_activations` records with magnitude=100,
   and (b) output content to DIFFER from baseline (`apply kernel actually
   intervened on the residual stream`).

Phase 1 establishes "kernel runs without changing output when contribution
is zero" — the zero-cvec is the strongest possible "applies but produces
no effect" signal, guarding against false-positive interventions where
the apply happens but produces no measurable effect. Phase 2 establishes
"kernel produces a measurable effect when contribution is non-zero" —
guarding against false-negatives where the apply path is a no-op.

## Validation results (artifact)

The most recent run is captured at:
- `docs/media/<date>_cvec_activations_validation_<sha>.txt`
- `docs/media/<date>_cvec_activations_baseline_<sha>.json`

## Implementation layout

- **Engine** (`lm_engine.swift`):
  - `Session.captureCvecActivations: Bool`, `Session.cvecActivationsQueue`
  - `Session.recordCvecActivation()` — early-return when capture is off;
    appends `(token_position, layer, magnitude)` otherwise.
  - `Session.drainCvecActivations()` — pump-side drain.
  - Call site: AR-step inner loop (the `for c in s.activeControls` block
    where `c.magnitudeAt(position:turn:)` is evaluated). The recorder
    sits *after* the magnitude scalar is computed so it's free — no
    extra Metal kernel invocation, no allocation when off.
- **FFI batch** (`ffi_batch.swift`):
  - `CvecActivationRecord` struct (12 bytes wire).
  - Wire format bumped v1 → **v2**; slot size 64 → **72** bytes; the new
    8-byte tail carries `(count, heap_offset)` for the activation array.
  - `StreamUpdateOut.newCvecActivations` populated from `drainCvecActivations`.
  - Flags bit **0x04** enables capture at session construction.
- **FFI Python** (`server/gemma_ffi.py`):
  - `StreamSpec.flags` doc mentions bit 2.
  - `StreamUpdate.new_cvec_activations: list[tuple[int, int, float]]`.
  - Decoder accepts v2, reads the new 8-byte tail.
- **Bridge** (`server/bridge.py`):
  - `POST /v1/resources/register` — register a cvec into the bridge's
    libgemma registry.
  - `POST /v1/chat/completions` request body accepts
    `capture_cvec_activations: bool` and `controls: list`.
  - SSE stream emits the new `delta.cvec_activations` chunks.

## What still needs visualizer-side work

`server/static/steering.html` does not yet consume
`delta.cvec_activations`. The deferred banner on the page now reads
"per-token cvec activations API now live; visualizer rendering still
TODO" with pointers to the test harness so a follow-up session can wire
the heatmap against an already-validated engine surface.
