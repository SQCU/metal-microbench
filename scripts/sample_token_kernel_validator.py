#!/usr/bin/env python3
"""Phase-1 validator for the GPU-side `sample_token` kernel.

See docs/dataflow_pipeline_spec.md §2.1 / §3 (P1 gate).

The kernel is a direct port of the CPU `sampleTokenFromLogits`
algorithm (inverse-CDF softmax sampling with temperature + min_p, argmax
at T <= 0). Only deliberate divergence from the CPU reference is the
PRNG: Swift's stdlib RNG is not callable from Metal, so the kernel uses
philox-4x32-10 keyed on (seed, step, slot). Output distribution is
statistically identical; specific draws differ because the PRNG is
different.

Tests:
  1. T = 0 argmax, distinct peaks per slot — bit-exact vs numpy argmax.
  2. T = 0 argmax, ties — tie-break goes to lowest token id.
  3. T = 0 argmax, edge positions (0, VOCAB-1, boundaries of the
     32-wide strided scan) — bit-exact.
  4. T > 0 reproducibility — same (seed, step, slot) → same token.
  5. T > 0 distribution match — draw N samples per (slot, step∈[0..N)),
     compare empirical frequencies against softmax expected. Chi-squared
     goodness-of-fit at α=0.01 over the top-K most-likely tokens.
"""
from __future__ import annotations
import ctypes as C
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DYLIB = REPO / "libgemma_metal.dylib"

B = 4
VOCAB = 262144


def _load() -> C.CDLL:
    lib = C.CDLL(str(DYLIB))
    lib.gemma_test_sample_token.argtypes = [
        C.POINTER(C.c_uint16),  # logits fp16 bits [B*VOCAB]
        C.POINTER(C.c_float),   # logit_bias fp32 [B*VOCAB]
        C.POINTER(C.c_float),   # temperature [B]
        C.POINTER(C.c_float),   # min_p [B]
        C.POINTER(C.c_uint32),  # seed [B]
        C.POINTER(C.c_uint32),  # step [B]
        C.POINTER(C.c_uint32),  # active [B]
        C.POINTER(C.c_uint32),  # out sampled [B]
    ]
    lib.gemma_test_sample_token.restype = C.c_int32
    return lib


def fp16_bits(arr_f32: np.ndarray) -> np.ndarray:
    return arr_f32.astype(np.float16).view(np.uint16)


def _call(lib, *, logits_f32, temp, min_p, seed, step, active, bias=None):
    bits = fp16_bits(logits_f32).reshape(-1)
    if bias is None:
        bias = np.zeros((B, VOCAB), dtype=np.float32)
    bias_flat = np.ascontiguousarray(bias.astype(np.float32).reshape(-1))
    out = np.zeros(B, dtype=np.uint32)
    rc = lib.gemma_test_sample_token(
        bits.ctypes.data_as(C.POINTER(C.c_uint16)),
        bias_flat.ctypes.data_as(C.POINTER(C.c_float)),
        temp.astype(np.float32).ctypes.data_as(C.POINTER(C.c_float)),
        min_p.astype(np.float32).ctypes.data_as(C.POINTER(C.c_float)),
        seed.astype(np.uint32).ctypes.data_as(C.POINTER(C.c_uint32)),
        step.astype(np.uint32).ctypes.data_as(C.POINTER(C.c_uint32)),
        active.astype(np.uint32).ctypes.data_as(C.POINTER(C.c_uint32)),
        out.ctypes.data_as(C.POINTER(C.c_uint32)))
    if rc != 0:
        raise RuntimeError(f"gemma_test_sample_token returned {rc}")
    return out


def _argmax_config(seed_base: int = 0):
    return dict(
        temp=np.zeros(B, dtype=np.float32),
        min_p=np.zeros(B, dtype=np.float32),
        seed=np.full(B, seed_base, dtype=np.uint32),
        step=np.zeros(B, dtype=np.uint32),
        active=np.ones(B, dtype=np.uint32),
    )


def test_argmax_distinct(lib) -> tuple[bool, str]:
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((B, VOCAB)).astype(np.float32) * 0.1
    expected = np.array([1234, 50000, 100000, 200000], dtype=np.uint32)
    for b in range(B):
        logits[b, expected[b]] = 100.0
    out = _call(lib, logits_f32=logits, **_argmax_config())
    ok = np.array_equal(out, expected)
    return ok, f"got {out.tolist()}, expected {expected.tolist()}"


def test_argmax_ties(lib) -> tuple[bool, str]:
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((B, VOCAB)).astype(np.float32) * 0.01
    logits[0, 500] = 10.0;   logits[0, 1500] = 10.0       # tie → 500
    logits[1, 100000] = 5.0; logits[1, 150000] = 5.0      # tie → 100000
    logits[2, 42000] = 20.0
    logits[3, 90] = 30.0
    expected = np.array([500, 100000, 42000, 90], dtype=np.uint32)
    out = _call(lib, logits_f32=logits, **_argmax_config())
    ok = np.array_equal(out, expected)
    return ok, f"got {out.tolist()}, expected {expected.tolist()}"


def test_argmax_edges(lib) -> tuple[bool, str]:
    rng = np.random.default_rng(99)
    logits = rng.standard_normal((B, VOCAB)).astype(np.float32) * 0.001
    logits[0, 0] = 100.0
    logits[1, VOCAB - 1] = 100.0
    logits[2, 31] = 100.0
    logits[3, 32] = 100.0
    expected = np.array([0, VOCAB - 1, 31, 32], dtype=np.uint32)
    out = _call(lib, logits_f32=logits, **_argmax_config())
    ok = np.array_equal(out, expected)
    return ok, f"got {out.tolist()}, expected {expected.tolist()}"


def test_stochastic_reproducibility(lib) -> tuple[bool, str]:
    """Same (seed, step, slot) → same token. Two calls, identical output."""
    rng = np.random.default_rng(2026)
    logits = rng.standard_normal((B, VOCAB)).astype(np.float32) * 2.0
    cfg = dict(
        temp=np.array([1.0, 0.5, 1.0, 0.7], dtype=np.float32),
        min_p=np.zeros(B, dtype=np.float32),
        seed=np.array([11, 22, 33, 44], dtype=np.uint32),
        step=np.array([0, 0, 0, 0], dtype=np.uint32),
        active=np.ones(B, dtype=np.uint32),
    )
    out1 = _call(lib, logits_f32=logits, **cfg)
    out2 = _call(lib, logits_f32=logits, **cfg)
    ok = np.array_equal(out1, out2)
    return ok, f"call1={out1.tolist()} call2={out2.tolist()}"


def test_stochastic_distribution(lib) -> tuple[bool, str]:
    """Slot 0 at T=1.0, 10k draws (step=0..9999, same seed). Compare
    empirical frequency of the top-16 tokens to softmax expected via
    chi-squared. Only slot 0 active; other slots inactive (skipped by
    kernel)."""
    rng = np.random.default_rng(13)
    # Concentrate probability mass on ~100 tokens so chi-squared has
    # real DOF. Base logits are gentle N(0, 0.1); we bump 100 random
    # positions by +4.0 each so their softmax probabilities dominate
    # and each has expected count >> 5 at N=10k.
    logits_slot0 = rng.standard_normal(VOCAB).astype(np.float32) * 0.1
    peaks = rng.choice(VOCAB, size=100, replace=False)
    # Peak bump large enough that peaks dominate the softmax mass — each
    # peak then carries ~1% of total probability, giving N*p ≈ 100 at
    # N=10k (well above the chi-squared validity threshold of 5).
    logits_slot0[peaks] += 10.0
    # Keep other slots' logits undefined — they're inactive. Fill zeros.
    full_logits = np.zeros((B, VOCAB), dtype=np.float32)
    full_logits[0] = logits_slot0

    # Reference softmax probabilities at T=1.0
    # Cast to fp16 and back to match what the GPU kernel sees.
    logits_fp16_as_fp32 = fp16_bits(logits_slot0).view(np.float16).astype(np.float32)
    x = logits_fp16_as_fp32 - logits_fp16_as_fp32.max()
    exp_x = np.exp(x)
    probs = exp_x / exp_x.sum()

    # Draw N samples by varying step.
    N = 10_000
    step_values = np.arange(N, dtype=np.uint32)
    seed_value = np.uint32(0xB0B)

    # Batch the calls — one call per step with B=4 slots but only slot 0
    # active. Accumulate histogram of slot-0 outputs.
    hist = np.zeros(VOCAB, dtype=np.int64)
    for s_chunk in range(0, N, 256):
        s_end = min(s_chunk + 256, N)
        for s in range(s_chunk, s_end):
            cfg = dict(
                temp=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                min_p=np.zeros(B, dtype=np.float32),
                seed=np.array([seed_value, 0, 0, 0], dtype=np.uint32),
                step=np.array([s, 0, 0, 0], dtype=np.uint32),
                active=np.array([1, 0, 0, 0], dtype=np.uint32),
            )
            out = _call(lib, logits_f32=full_logits, **cfg)
            hist[out[0]] += 1
    total = int(hist.sum())
    if total != N:
        return False, f"expected {N} draws for slot 0, got {total} (inactive-slot leakage?)"

    # Chi-squared over top-K tokens. K chosen so expected count >= 5 per
    # cell (standard chi-squared validity rule). Remainder bucketed.
    top_idx = np.argsort(probs)[::-1]
    K_MAX = 256
    cum = 0.0
    K = 0
    for K in range(1, min(K_MAX, VOCAB)):
        cum += probs[top_idx[K - 1]]
        if probs[top_idx[K]] * N < 5.0:
            break
    observed = hist[top_idx[:K]]
    expected = probs[top_idx[:K]] * N
    other_obs = N - int(observed.sum())
    other_exp = N - float(expected.sum())
    if other_exp >= 5.0:
        observed = np.append(observed, other_obs)
        expected = np.append(expected, other_exp)
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = len(observed) - 1
    # Critical value for α=0.01 via scipy if available; else Wilson-Hilferty.
    try:
        from scipy.stats import chi2 as sc_chi2
        crit = float(sc_chi2.ppf(0.99, dof))
        pval = float(1 - sc_chi2.cdf(chi2, dof))
    except ImportError:
        # Wilson-Hilferty approximation: ((chi2/dof)**(1/3) - (1 - 2/(9*dof))) / sqrt(2/(9*dof)) ~ N(0,1)
        crit = dof + 2.326 * np.sqrt(2 * dof) + (2 * 2.326 ** 2 - 2) / 3
        pval = None

    ok = chi2 < crit
    msg = (f"N={N} K={K} chi2={chi2:.2f} critical(α=0.01)={crit:.2f}"
           + (f" p={pval:.4f}" if pval is not None else ""))
    return ok, msg


def test_logit_bias_shifts_argmax(lib) -> tuple[bool, str]:
    """Bias large enough to flip argmax away from the raw logit max."""
    rng = np.random.default_rng(17)
    logits = rng.standard_normal((B, VOCAB)).astype(np.float32) * 0.1
    # Base peaks
    logits[0, 1000] = 5.0; logits[0, 2000] = 1.0
    logits[1, 100] = 4.0;  logits[1, 50000] = 0.5
    logits[2, 42] = 3.0;   logits[2, 99] = 2.0
    logits[3, 30000] = 2.0; logits[3, 31000] = 1.5
    # Bias that pushes the secondary past the primary for slots 0,1.
    # Slots 2,3 get zero bias.
    bias = np.zeros((B, VOCAB), dtype=np.float32)
    bias[0, 2000] = 10.0    # 1.0 + 10.0 > 5.0 → choose 2000
    bias[1, 50000] = 5.0    # 0.5 + 5.0 > 4.0 → choose 50000
    expected = np.array([2000, 50000, 42, 30000], dtype=np.uint32)
    out = _call(lib, logits_f32=logits, bias=bias, **_argmax_config())
    ok = np.array_equal(out, expected)
    return ok, f"got {out.tolist()}, expected {expected.tolist()}"


def test_logit_bias_reshapes_distribution(lib) -> tuple[bool, str]:
    """At T=1.0, a negative bias on a high-probability token should
    reduce its empirical frequency proportionally to exp(bias)."""
    rng = np.random.default_rng(101)
    base_logits = rng.standard_normal(VOCAB).astype(np.float32) * 0.1
    peaks = rng.choice(VOCAB, size=20, replace=False)
    base_logits[peaks] += 8.0   # dominant peaks; each ~equal probability

    full_logits = np.zeros((B, VOCAB), dtype=np.float32)
    full_logits[0] = base_logits

    # Suppress peaks[0] by -5: empirical probability should drop by e^-5 ≈ 0.0067×
    bias_full = np.zeros((B, VOCAB), dtype=np.float32)
    bias_full[0, peaks[0]] = -5.0

    # Ground truth: softmax over (logit + bias) at T=1, quantized through fp16.
    x = fp16_bits(base_logits).view(np.float16).astype(np.float32) + bias_full[0]
    x -= x.max()
    probs = np.exp(x) / np.exp(x).sum()
    p_suppressed = float(probs[peaks[0]])
    p_typical_peak = float(probs[peaks[1]])

    N = 4000
    seed_value = np.uint32(0xDAB5)
    hist = np.zeros(VOCAB, dtype=np.int64)
    for s in range(N):
        cfg = dict(
            temp=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            min_p=np.zeros(B, dtype=np.float32),
            seed=np.array([seed_value, 0, 0, 0], dtype=np.uint32),
            step=np.array([s, 0, 0, 0], dtype=np.uint32),
            active=np.array([1, 0, 0, 0], dtype=np.uint32),
        )
        out = _call(lib, logits_f32=full_logits, bias=bias_full, **cfg)
        hist[out[0]] += 1

    emp_suppressed = hist[peaks[0]] / N
    emp_typical = float(np.mean([hist[p] / N for p in peaks[1:]]))

    # Sanity: suppressed peak must be way below typical peak.
    # With bias=-5, suppressed_prob ≈ e^-5 × typical ≈ 0.007 × typical.
    # Empirical ratio within 10× factor tolerance (sampling noise dominates).
    ok = (emp_suppressed < emp_typical * 0.1) and (p_suppressed < p_typical_peak * 0.1)
    msg = (f"suppressed_peak emp={emp_suppressed:.4f} vs typical emp={emp_typical:.4f}; "
           f"expected ratio ~{p_suppressed / p_typical_peak:.4f}")
    return ok, msg


def main() -> int:
    lib = _load()
    tests = [
        ("T=0 argmax — distinct peaks",      test_argmax_distinct),
        ("T=0 argmax — tie-break to low id", test_argmax_ties),
        ("T=0 argmax — edge positions",      test_argmax_edges),
        ("T>0 — reproducibility",            test_stochastic_reproducibility),
        ("T>0 — distribution match (χ²)",    test_stochastic_distribution),
        ("logit_bias — shifts argmax",       test_logit_bias_shifts_argmax),
        ("logit_bias — reshapes T>0 dist",   test_logit_bias_reshapes_distribution),
    ]
    all_ok = True
    for name, fn in tests:
        try:
            ok, msg = fn(lib)
        except Exception as e:
            ok, msg = False, f"exception: {e}"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        if not ok:
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
