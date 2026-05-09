# KV-cache correlation diagnosis notes

Date: 2026-05-08

## Diagnosis from cheap audit

The unsteered cache-correlation repro does not exercise cvec state: no
control vectors are attached, so `cvecDigestForPage` returns `0` and the
page hash is token-only by design. The current digest covers the visible
KV-affecting control-vector fields: layer, cvec id, envelope parameters,
mode, target, polarity, transport parameters, start/stop position, and
start/stop turn.

The concrete defect found in the RNG path was seed propagation. The
OpenAI bridge parses `seed`, `server/gemma_ffi.py` serializes it, and
`ffi_batch.swift` decodes it into `DecodedSampling.seed`, but the value
was not copied into `LmEngine.RequestInit` or applied to `Session`.
Therefore every new session kept its randomly initialized
`Session.gpuRngSeed`, even when the client supplied a seed.

That bug makes seeded reproductions and seeded per-trial independence
invalid. It is mechanism 4-class behavior: sampling RNG state was not
actually controlled at the Session/GPU sampler boundary.

## Patch

`Session.applySamplingSeed(_:)` now applies a nonzero client seed to:

- `Session.rng`, for the host-side seedable RNG path.
- `Session.gpuRngSeed`, after SplitMix64 mixing, for the Metal
  `sample_token` Philox path.

`ffi_batch.swift` now forwards `DecodedSampling.seed` on start, continue,
and touch actions. Seed `0` keeps the old behavior and means
"unspecified/random".

## Required validation on a Metal host

The local sandbox used for this pass could build the dylib, but could
not run Metal or localhost HTTP. Run these on the target host:

```bash
CLANG_MODULE_CACHE_PATH=/private/tmp/clang-module-cache \
SWIFT_MODULE_CACHE_PATH=/private/tmp/swift-module-cache \
make libgemma_metal.dylib

python3 server/test_batch_ffi.py
```

Then run the tracked bridge repro:

```bash
python3 tools/st-debug/scripts/cache_correlation_test.py --temperature 0 -k 5
python3 tools/st-debug/scripts/cache_correlation_test.py --temperature 0.4 -k 20
```

Interpretation:

- At `temperature=0`, batch 1 and batch 2 should produce identical first
  prefixes/tool-call decisions. If not, the issue is K/V value
  correctness and the next step is page-by-page K/V diffing.
- At `temperature=0.4`, after the seed fix, unseeded batches should only
  differ by sampling error. Seeded batches should be reproducible across
  repeated runs with the same seed.

If `temperature=0` still diverges, instrument `promoteFinishedPages` and
`adoptSharedPrefixPages` to dump each adopted pair's K/V bytes per layer
and compare them against a fresh prefill of the same prompt. Any nonzero
diff localizes the remaining bug to page contents rather than RNG.
