# Multi-component PCA steering

The standard "pick layer N, mean(pos) − mean(neg), normalize" recipe for
concept vectors treats the residual stream as if the concept lives at
one depth. Empirically it doesn't — contrastive variation spreads across
the stack, with some mid-layers contributing multiple principal
components and many layers contributing essentially none. The
`/v1/control/construct_pca` endpoint and the `/steering` fit panel
surface this structure directly instead of asking the developer-user
to guess.

## Shape of the output

One fit = one pos/neg pair of prose examples → a ranked list of
`(layer, component_index, eigenvalue, unit_direction)` tuples. The
endpoint sorts globally by eigenvalue descending, truncates at the first
component whose cumulative coverage reaches `top_p` (default 0.80), and
registers every kept component as its own cvec under
`<id_prefix>-L<LL>-C<k>`.

Typical fit (5+5 prose pairs, top_p=0.80):

```
captured_fraction:  0.80
components kept:    ~15-25
layers contributing: ~10-15 distinct layers
layers dominant:    ~2-5 mid-stack (often L10-L18)
components/layer:   PC1 dominates; PC2 occasionally clears the threshold
                    for the "strongest" layers; PC3+ rarely survive
```

A component at `(L, 0)` means "the PC1 of the pos-vs-neg delta matrix at
layer L." A component at `(L, 1)` means "the direction of next-largest
variance at layer L, orthogonal to PC1" — often a sub-axis of the
concept (e.g. if the concept is "joy," PC2 might split "calm-happy" from
"excited-happy").

## Apply semantics

Every kept component is attached as an independent `ActiveControl` on
the session. Its `peakMagnitude` = `scale × intensity`, where:

- `scale = eigenvalue / max_eigenvalue` is returned by the endpoint
  (always 1.0 for the strongest component, proportionally less for
  weaker ones) — preserves the relative importance structure the PCA
  inferred.
- `intensity` is the user-facing knob in the UI — a single scalar that
  uniformly scales the whole fit.

Because each ActiveControl carries its own layer, the engine applies
each component at its own residual-stream site. The full intervention is
the sum of all component injections. This is bit-exact reproducible —
`computeCvecDigest` hashes every (layer, cvecId, envelope params)
tuple, so two sessions with the same fit get the same cache key.

## Interaction with the prefix cache

A fit's cvec-digest is the hash of *every* kept component's parameters.
Two sessions run against the same fit (same id_prefix, same intensity,
same envelope) share KV pages at the slide/full pair-promotion level as
normal. Two sessions with *different* fits — even if the prose was
similar — get different digests and correctly miss each other's pages.
The steered K/V is always reproduced against its exact intervention set.

## What NOT to interpret

- **"Layer N is the steering layer."** Even the single strongest
  component's layer is just where *one* direction of variance happens
  to be largest; the full intervention reaches ~70% of its effect from
  other layers' contributions. Don't pin a layer number to a concept.
- **PC2+ as a second concept.** PCs beyond the first are orthogonal
  *within a layer* — they're a decomposition of the same
  pos-vs-neg delta, not independent concepts. If you want to steer
  two different concepts, do two separate fits with two different
  pos/neg sets and attach both.
- **Cumulative fraction as a quality metric.** `captured_fraction ≈
  top_p` always (by construction). What varies is how many components
  it took to get there. Few components = concept is structurally
  localized; many components = concept is diffuse across the stack.
  Both are interesting, just different.

## Debugging a fit

The UI renders a `(NUM_LAYERS × max_components_per_layer)` heatmap
colored by eigenvalue fraction, with yellow outlines on the cells that
made the top-p cut. At a glance:

- Bright cells clustered in the middle = clean fit, concept is
  layer-localized but spread across a few PCs there.
- Diffuse bright cells across many layers = concept is distributed —
  typical for broad stylistic directions (joy/dread).
- Only L0-L4 bright = the fit caught embedding-space variation, not
  semantic content; probably your pos/neg pairs differ in surface
  token distribution more than meaning. Add more prose variety.
- No bright cells anywhere = concept isn't separable with the given
  examples. Add more pairs or sharpen the contrast.

## Speed

~4s for 5+5 pairs on M5 Max: 10 forward passes with
`gCaptureAllLayers` enabled, batched B=4 where possible, plus numpy
SVD per layer (~30 × <5ms). The SVD step is negligible; forward passes
dominate.
