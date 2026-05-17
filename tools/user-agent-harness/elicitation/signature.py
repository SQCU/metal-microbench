"""Layered signature estimator for elicitation Likert data.

Five layers, each sample-size-gated. Higher layers are skipped (return
None) when there aren't enough samples to estimate them honestly — we
never silently degrade to a coarser layer when a finer one is requested
because that masks the input dependency from the consumer.

    Layer 1 — per-axis univariate: mean, std, median   (N >= 2)
    Layer 2 — bivariate Pearson correlation (14×14)    (N >= 6)
    Layer 3 — within-persona pooled covariance         (N >= 30 total,
              + Mahalanobis distance to persona means)   per-persona ≥ 5)
    Layer 4 — PCA on pooled sample matrix              (N >= 2*N_AXES,
                                                         conventionally 50)
    Layer 5 — drift matrix (requested-shift × measured-shift) — populated
              by the workshop loop once it runs; computed from JSONL
              records carrying a `meta.target_shift` field.

These are minimum-N gates, not tight statistical guarantees. The PCA at
N=2*N_AXES is just barely well-conditioned; you'd really want N >= 100
for stable principal directions. We surface the sample count so the
consumer can decide how much to trust the result.
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from axes import AXIS_NAMES, N_AXES


# Sample-size gates.
MIN_N_UNIVARIATE   = 2
MIN_N_CORRELATION  = 6
MIN_N_COV_TOTAL    = 30
MIN_N_COV_PER_PERS = 5
MIN_N_PCA          = 2 * N_AXES   # 28; "minimum sane"; really want 50+


def load_records(path: str | Path) -> list[dict]:
    """Read a JSONL file and return parsed records. Skips empty lines
    and lines that fail to parse (records errors to stderr is the
    caller's job)."""
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


def records_to_matrix(records: list[dict],
                       require_full: bool = True
                       ) -> tuple[np.ndarray, list[str], list[int]]:
    """Convert a list of records into (X, persona_labels, indices) where
    X is an (n_kept × N_AXES) matrix of Likert scores. If require_full,
    drop any record with fewer than N_AXES axes parsed. Otherwise impute
    missing axes by the column mean across the kept records."""
    kept = []
    for r in records:
        likert = r.get("likert") or {}
        if require_full and len(likert) != N_AXES:
            continue
        kept.append(r)

    if not kept:
        return np.zeros((0, N_AXES)), [], []

    if require_full:
        X = np.array([[r["likert"][a] for a in AXIS_NAMES] for r in kept],
                     dtype=float)
    else:
        # Partial-record path: build with NaN and mean-impute by column.
        X = np.full((len(kept), N_AXES), np.nan)
        for i, r in enumerate(kept):
            for j, a in enumerate(AXIS_NAMES):
                v = r.get("likert", {}).get(a)
                if v is not None:
                    X[i, j] = v
        col_means = np.nanmean(X, axis=0)
        # Replace each NaN with its column's mean.
        for j in range(N_AXES):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_means[j]
    return X, [r["persona"] for r in kept], [r.get("sample_index", -1) for r in kept]


# ─── Layer 1 — univariate ────────────────────────────────────────────

@dataclass
class AxisStats:
    n: int
    mean: float
    std: float
    median: float
    p25: float
    p75: float


def univariate(records: list[dict],
                per_persona: bool = True
                ) -> dict[str, dict[str, AxisStats]] | dict[str, AxisStats]:
    """Per-axis statistics. If per_persona, returns {persona: {axis: stats}};
    otherwise returns {axis: stats} over all records."""
    def axis_stats_from_values(vals: list[float]) -> AxisStats | None:
        if len(vals) < MIN_N_UNIVARIATE:
            return None
        a = np.asarray(vals)
        return AxisStats(
            n=int(a.size),
            mean=float(a.mean()),
            std=float(a.std(ddof=0)),
            median=float(np.median(a)),
            p25=float(np.percentile(a, 25)),
            p75=float(np.percentile(a, 75)),
        )

    if per_persona:
        out: dict[str, dict[str, AxisStats]] = defaultdict(dict)
        bucket = defaultdict(lambda: defaultdict(list))
        for r in records:
            for a in AXIS_NAMES:
                v = r.get("likert", {}).get(a)
                if v is not None:
                    bucket[r["persona"]][a].append(v)
        for persona, ax in bucket.items():
            for a, vals in ax.items():
                s = axis_stats_from_values(vals)
                if s is not None:
                    out[persona][a] = s
        return dict(out)
    else:
        bucket = defaultdict(list)
        for r in records:
            for a in AXIS_NAMES:
                v = r.get("likert", {}).get(a)
                if v is not None:
                    bucket[a].append(v)
        return {a: axis_stats_from_values(v) for a, v in bucket.items()
                if axis_stats_from_values(v) is not None}


# ─── Layer 2 — bivariate correlation ─────────────────────────────────

def correlation_matrix(records: list[dict]) -> np.ndarray | None:
    """Pearson correlation across all (axis_i, axis_j) pairs, using only
    records with all N_AXES values present. Returns None if too few
    records to estimate.

    NaN-handling: if any axis has zero variance across the records (a
    column of constants — common at low N when a persona scores the
    same integer on a given axis every sample), `np.corrcoef` returns
    NaN for that row/column. We replace those NaNs with 0 because
    Pearson r is undefined-but-conventionally-zero when one variable
    is constant: there is no linear association you can attribute to
    correlation rather than to the absence of variation.
    """
    X, _, _ = records_to_matrix(records, require_full=True)
    if X.shape[0] < MIN_N_CORRELATION:
        return None
    # numpy's corrcoef wants rows=variables, cols=observations.
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(X.T)
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)


def correlation_top_pairs(corr: np.ndarray, k: int = 8) -> list[tuple[str, str, float]]:
    """Return the top-k (axis_i, axis_j, r) pairs by |r|, excluding the
    diagonal. Useful for spotting axis conflations."""
    out: list[tuple[str, str, float]] = []
    for i in range(N_AXES):
        for j in range(i + 1, N_AXES):
            out.append((AXIS_NAMES[i], AXIS_NAMES[j], float(corr[i, j])))
    out.sort(key=lambda t: -abs(t[2]))
    return out[:k]


# ─── Layer 3 — within-persona covariance + Mahalanobis ──────────────

@dataclass
class CovarianceLayer:
    persona_means: dict[str, np.ndarray]
    pooled_within: np.ndarray
    pooled_within_inv: np.ndarray | None  # None if singular
    n_per_persona: dict[str, int]


def covariance_layer(records: list[dict]) -> CovarianceLayer | None:
    """Pooled within-persona covariance + persona-conditional means.
    Returns None unless gates pass. The pooled covariance is the
    natural noise model for Mahalanobis distance — it treats sample-
    to-sample variation within a persona as the noise floor and asks
    'how far apart are persona means in units of that noise?'"""
    full_records = [r for r in records if r.get("axes_recovered", 0) == N_AXES]
    if len(full_records) < MIN_N_COV_TOTAL:
        return None
    by_persona: dict[str, list[np.ndarray]] = defaultdict(list)
    for r in full_records:
        vec = np.array([r["likert"][a] for a in AXIS_NAMES], dtype=float)
        by_persona[r["persona"]].append(vec)

    eligible = {p: np.stack(vs) for p, vs in by_persona.items()
                if len(vs) >= MIN_N_COV_PER_PERS}
    if not eligible:
        return None

    persona_means = {p: X.mean(axis=0) for p, X in eligible.items()}
    n_per_persona = {p: int(X.shape[0]) for p, X in eligible.items()}

    # Pooled within-class covariance: (N - K) degree-of-freedom denominator,
    # where K is the number of personas and N the total kept samples.
    centered_blocks = []
    for p, X in eligible.items():
        centered_blocks.append(X - persona_means[p])
    C = np.vstack(centered_blocks)
    n_total = C.shape[0]
    K = len(eligible)
    if n_total <= K:
        return None
    pooled = (C.T @ C) / (n_total - K)

    # Add a tiny ridge to keep it invertible. Likert ordinals can produce
    # singular covariances at low N (especially on collapsed axes like
    # `deferential` in our pilot data).
    ridge = 1e-3 * np.trace(pooled) / N_AXES * np.eye(N_AXES)
    pooled_reg = pooled + ridge
    try:
        inv = np.linalg.inv(pooled_reg)
    except np.linalg.LinAlgError:
        inv = None

    return CovarianceLayer(
        persona_means=persona_means,
        pooled_within=pooled,
        pooled_within_inv=inv,
        n_per_persona=n_per_persona,
    )


def mahalanobis(x: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> float:
    """Mahalanobis distance between point x and mean mu under inverse-
    covariance cov_inv. Returns sqrt of the quadratic form."""
    d = x - mu
    q = float(d @ cov_inv @ d)
    return math.sqrt(max(q, 0.0))


def pairwise_persona_distances(cov: CovarianceLayer) -> dict[tuple[str, str], float]:
    """Mahalanobis distance between each pair of persona means under the
    pooled-within-class covariance. This is the canonical 'how distinct
    are these classes' statistic for multivariate data."""
    if cov.pooled_within_inv is None:
        return {}
    personas = sorted(cov.persona_means.keys())
    out = {}
    for i, a in enumerate(personas):
        for b in personas[i + 1:]:
            d = mahalanobis(cov.persona_means[a], cov.persona_means[b],
                            cov.pooled_within_inv)
            out[(a, b)] = d
    return out


# ─── Layer 4 — PCA + effective dimensionality ───────────────────────

@dataclass
class PCALayer:
    components: np.ndarray            # (N_AXES, N_AXES); rows are PCs
    variance_explained: np.ndarray    # (N_AXES,) eigenvalues, decreasing
    cumulative_explained: np.ndarray  # (N_AXES,)
    n_samples: int
    mean_vec: np.ndarray              # what was centered out


def pca(records: list[dict]) -> PCALayer | None:
    """PCA on the pooled sample matrix across all personas. The principal
    directions are the natural axes of variation IN THE OBSERVED DATA —
    they reveal where 14 named axes collapse to a lower-d effective
    subspace. Effective dimensionality is read off the cumulative-
    variance curve."""
    X, _, _ = records_to_matrix(records, require_full=True)
    if X.shape[0] < MIN_N_PCA:
        return None
    n = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    # Sample covariance (N - 1 denominator).
    S = (Xc.T @ Xc) / (n - 1)
    # Eigendecomposition of the symmetric covariance.
    eigvals, eigvecs = np.linalg.eigh(S)
    # eigh returns ascending; flip to descending.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    cum = np.cumsum(eigvals) / max(eigvals.sum(), 1e-12)
    # Components: rows = PCs, columns = axes.
    return PCALayer(
        components=eigvecs.T,
        variance_explained=eigvals,
        cumulative_explained=cum,
        n_samples=n,
        mean_vec=mu,
    )


def effective_dimensionality(pca_layer: PCALayer, threshold: float = 0.90) -> int:
    """Smallest K such that the top-K PCs explain ≥ threshold of the
    total variance. The headline 'is the basis over-parameterized'
    statistic."""
    return int(np.searchsorted(pca_layer.cumulative_explained, threshold) + 1)


def top_loadings(pca_layer: PCALayer, pc_index: int, k: int = 5
                  ) -> list[tuple[str, float]]:
    """For a given principal component, return the top-k axes by
    |loading|. Used to make components human-readable."""
    comp = pca_layer.components[pc_index]
    idxs = np.argsort(-np.abs(comp))[:k]
    return [(AXIS_NAMES[i], float(comp[i])) for i in idxs]


# ─── Combined report builder ─────────────────────────────────────────

@dataclass
class FullSignature:
    n_records: int
    n_full: int
    n_per_persona: dict[str, int]
    univariate_global: dict[str, AxisStats] | None
    univariate_per_persona: dict[str, dict[str, AxisStats]] | None
    correlation: np.ndarray | None
    covariance_layer: CovarianceLayer | None
    pca_layer: PCALayer | None


def full_signature(records: list[dict]) -> FullSignature:
    """Run all layers, gating each by its own sample-size precondition.
    Returns a structured report; consumers decide which layers to use."""
    n_records = len(records)
    n_full = sum(1 for r in records if r.get("axes_recovered", 0) == N_AXES)
    n_pp = defaultdict(int)
    for r in records:
        if r.get("axes_recovered", 0) == N_AXES:
            n_pp[r["persona"]] += 1
    return FullSignature(
        n_records=n_records,
        n_full=n_full,
        n_per_persona=dict(n_pp),
        univariate_global=univariate(records, per_persona=False) if n_full >= MIN_N_UNIVARIATE else None,
        univariate_per_persona=univariate(records, per_persona=True) if n_full >= MIN_N_UNIVARIATE else None,
        correlation=correlation_matrix(records),
        covariance_layer=covariance_layer(records),
        pca_layer=pca(records),
    )
