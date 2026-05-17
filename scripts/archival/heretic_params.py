"""Heretic-style parameter primitives — pure functions with no prior
assumptions about "sensible" values. All parameter choices are the
responsibility of a higher-level optimizer (Optuna/TPE harness).

Implements the parameterized-ablation feature set from p-e-w/heretic:

  direction_index  : float | "per_layer"
      Picks the refusal direction to use at every ablated layer.
        integer N → use r̂_N
        fractional d → linearly interpolate r̂_floor(d) and r̂_ceil(d),
                      then renormalize (preserves α=1 ⇒ full ablation
                      of this composite direction).
        "per_layer" → each layer L uses its own r̂_L.

  α(L) tent kernel : (max_weight, max_weight_position, min_weight,
                      min_weight_distance)
      Piecewise-linear "tent" function over layer index L:
        • Peak value `max_weight` at `max_weight_position` (fractional
          — it's a layer position, not an index).
        • Linear decay to `min_weight` at distance `min_weight_distance`
          layers from the peak (both directions).
        • Stays at `min_weight` beyond that distance.
      Evaluated at integer L ∈ [0, num_layers-1].

Separate kernels are maintained for the attention-out component and
the MLP/FFN-out component (heretic allows independent shapes).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct


HIDDEN = 2816
NUM_LAYERS = 30


# ─── F. Direction interpolation ──────────────────────────────────────

def interpolate_direction(
    direction_index: float | str,
    layer_directions: list[list[float]],
) -> list[float]:
    """Produce the effective refusal direction for a given
    direction_index. `layer_directions` is a list of per-layer r̂_L, each
    a HIDDEN-length list[float] (unit-norm). Returns one HIDDEN-length
    list[float] (unit-norm). For "per_layer", raises — that mode means
    the caller should use layer-L's own direction directly, no interp.

    Renormalization after linear combination preserves the
    "α=1 ⇒ fully ablate THIS direction" semantics. Without it, α's
    meaning drifts because ||interpolated|| < 1 for non-antipodal
    neighbor vectors.
    """
    if isinstance(direction_index, str):
        if direction_index == "per_layer":
            raise ValueError(
                "per_layer mode: caller should use layer_directions[L] "
                "directly at each ablated layer, not call interpolate_direction")
        raise ValueError(f"unknown direction_index mode: {direction_index!r}")
    n = len(layer_directions)
    if n == 0:
        raise ValueError("layer_directions is empty")
    d = max(0.0, min(float(n - 1), float(direction_index)))
    lo = int(math.floor(d))
    hi = min(n - 1, lo + 1)
    frac = d - lo
    if frac == 0.0 or lo == hi:
        v = list(layer_directions[lo])
    else:
        a = layer_directions[lo]; b = layer_directions[hi]
        v = [(1.0 - frac) * a[i] + frac * b[i] for i in range(HIDDEN)]
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-9:
        return list(layer_directions[lo])  # degenerate; fall back
    inv = 1.0 / norm
    return [x * inv for x in v]


def fp16_bytes(vec: list[float]) -> bytes:
    """Pack a HIDDEN-length float list into raw fp16 halves."""
    if len(vec) != HIDDEN:
        raise ValueError(f"expected HIDDEN={HIDDEN}, got {len(vec)}")
    return struct.pack(f"<{HIDDEN}e", *vec)


# ─── E. α(L) tent kernel ─────────────────────────────────────────────

@dataclass(frozen=True)
class AlphaKernel:
    """Heretic's parameterization of the per-layer intervention weight."""
    max_weight: float
    max_weight_position: float   # fractional layer index of the peak
    min_weight: float
    min_weight_distance: float   # distance at which α reaches min_weight


def alpha_at_layer(k: AlphaKernel, layer: int) -> float:
    """Piecewise-linear tent: α = max_weight at layer == max_weight_position;
    linearly decays to min_weight at ±min_weight_distance; flat at
    min_weight beyond.

    Matches heretic's diagram: a symmetric tent centered at
    max_weight_position with slopes set so that α(pos ± distance) =
    min_weight. Outside the tent, α is clamped to min_weight (NOT zero —
    that's a deliberate floor, often used to apply a light constant
    suppression across the whole model in addition to the peak)."""
    if k.min_weight_distance <= 0:
        # Degenerate kernel: flat at max_weight everywhere (pick peak value).
        return k.max_weight
    d = abs(float(layer) - k.max_weight_position)
    if d >= k.min_weight_distance:
        return k.min_weight
    # Linear blend between max (at d=0) and min (at d=distance).
    t = d / k.min_weight_distance
    return k.max_weight * (1.0 - t) + k.min_weight * t


def alpha_vector(k: AlphaKernel, num_layers: int = NUM_LAYERS) -> list[float]:
    """[α(L) for L in 0..num_layers-1]."""
    return [alpha_at_layer(k, L) for L in range(num_layers)]


# ─── High-level config → /v1/heretic/configure body ──────────────────

@dataclass(frozen=True)
class HereticConfig:
    direction_index: float | str            # e.g. 10.3 or "per_layer"
    id_prefix: str                          # e.g. "ref1st" → ref1st-LXX-C0
    attn_alpha: AlphaKernel
    ffn_alpha:  AlphaKernel
    # Absolute-α threshold below which we skip emitting an ablation
    # entry for that (layer, component) — saves one kernel dispatch
    # per skipped site.
    min_alpha_emit: float = 1e-3


def build_entries(
    cfg: HereticConfig,
    layer_directions_fp16_bytes: list[bytes] | None = None,
) -> list[dict]:
    """Compose the `entries` body for /v1/heretic/configure.

    If direction_index == "per_layer", we emit a separate cvec_id per
    layer (e.g. "{id_prefix}-L{L:02d}-C0"). Otherwise we compute the
    interpolated direction and send it as cvec_fp16_b64 on every entry
    (same bytes across layers, since heretic applies one direction at
    all layers in index mode). `layer_directions_fp16_bytes` is required
    when direction_index is a float — it's the list of 30 per-layer r̂
    as raw fp16 bytes, which the caller must obtain via
    g.control_get_fp16 for each layer's registered cvec.
    """
    import base64

    entries: list[dict] = []
    attn_alphas = alpha_vector(cfg.attn_alpha)
    ffn_alphas  = alpha_vector(cfg.ffn_alpha)

    interpolated_b64: str | None = None
    if not (isinstance(cfg.direction_index, str)
            and cfg.direction_index == "per_layer"):
        if layer_directions_fp16_bytes is None:
            raise ValueError(
                "direction_index is a float/int but layer_directions_fp16_bytes "
                "was not provided; caller must supply the 30-element list of "
                "registered r̂_L fp16 bytes so we can interpolate.")
        # Decode each to float list, interpolate, pack + b64.
        layer_dirs: list[list[float]] = [
            list(struct.unpack(f"<{HIDDEN}e", b))
            for b in layer_directions_fp16_bytes
        ]
        interp = interpolate_direction(cfg.direction_index, layer_dirs)
        interpolated_b64 = base64.b64encode(fp16_bytes(interp)).decode()

    for L in range(NUM_LAYERS):
        for comp_name, alpha in (("attn_out", attn_alphas[L]),
                                   ("ffn_out",  ffn_alphas[L])):
            if abs(alpha) < cfg.min_alpha_emit:
                continue
            entry: dict = {
                "layer": L, "component": comp_name, "alpha": alpha,
            }
            if interpolated_b64 is not None:
                entry["cvec_fp16_b64"] = interpolated_b64
            else:
                entry["cvec_id"] = f"{cfg.id_prefix}-L{L:02d}-C0"
            entries.append(entry)
    return entries


# ─── Smoke tests (run standalone) ────────────────────────────────────

if __name__ == "__main__":
    # F. Interpolation shape check on synthetic orthogonal directions.
    import random
    random.seed(0)
    d0 = [random.gauss(0, 1) for _ in range(HIDDEN)]
    n0 = math.sqrt(sum(x * x for x in d0)); d0 = [x / n0 for x in d0]
    d1 = [random.gauss(0, 1) for _ in range(HIDDEN)]
    n1 = math.sqrt(sum(x * x for x in d1)); d1 = [x / n1 for x in d1]
    # Interpolate at 0.0 should exactly return d0
    v = interpolate_direction(0.0, [d0, d1])
    err = max(abs(v[i] - d0[i]) for i in range(HIDDEN))
    print(f"F: interp(0.0, [d0, d1]) == d0 within {err:.2e}")
    assert err < 1e-6
    # Interpolate at 1.0 should exactly return d1
    v = interpolate_direction(1.0, [d0, d1])
    err = max(abs(v[i] - d1[i]) for i in range(HIDDEN))
    print(f"F: interp(1.0, [d0, d1]) == d1 within {err:.2e}")
    assert err < 1e-6
    # Interpolate at 0.5 should be normalized (||v||=1) and not equal either
    v = interpolate_direction(0.5, [d0, d1])
    nv = math.sqrt(sum(x * x for x in v))
    assert abs(nv - 1.0) < 1e-6, f"expected unit-norm, got {nv}"
    print(f"F: interp(0.5) is unit-norm (||v||={nv:.6f})")

    # E. Tent at a peak of 1.0 at position 10, decaying to 0 over 4 layers.
    k = AlphaKernel(max_weight=1.0, max_weight_position=10.0,
                     min_weight=0.0, min_weight_distance=4.0)
    alphas = alpha_vector(k, 30)
    print("E: α(L) for (max=1, pos=10, min=0, dist=4):")
    for L in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        print(f"     α({L:>2}) = {alphas[L]:.3f}")
    assert alphas[10] == 1.0
    assert alphas[6] == 0.0 and alphas[14] == 0.0
    assert 0 < alphas[8] < 1.0
    print("E: tent shape is correct (peak at 10, zero outside ±4)")

    # Build-entries smoke: per_layer mode.
    cfg = HereticConfig(
        direction_index="per_layer", id_prefix="ref1st",
        attn_alpha=k, ffn_alpha=k)
    entries = build_entries(cfg)
    print(f"Built {len(entries)} entries for per_layer mode "
          f"(expected ~14 — 7 layers × 2 components inside the ±4 tent)")
    print(f"First entry: {entries[0]}")
