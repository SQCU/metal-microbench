"""Load an elicitation JSONL store and print the layered signature report.

Usage:
    analyze.py PATH [--persona NAME] [--corr-top K]
    analyze.py PATH --json     # machine-readable single JSON object on stdout
"""

import argparse
import json
import sys

import numpy as np

from axes import AXIS_NAMES, N_AXES
from signature import (
    correlation_top_pairs,
    effective_dimensionality,
    full_signature,
    load_records,
    pairwise_persona_distances,
    top_loadings,
)


def banner(title: str) -> None:
    print()
    print("═" * 72)
    print(f"  {title}")
    print("═" * 72)


def report_univariate(fs):
    if not fs.univariate_per_persona:
        print("  [skipped — too few records for univariate]")
        return
    personas = sorted(fs.univariate_per_persona.keys())
    if not personas:
        print("  [no full-record personas]")
        return
    header = f"  {'axis':<22s}  " + "  ".join(f"{p:<24s}" for p in personas)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for axis in AXIS_NAMES:
        cells = []
        for p in personas:
            s = fs.univariate_per_persona[p].get(axis)
            if s is None:
                cells.append("(none)               ")
            else:
                cells.append(f"med={s.median:.1f} σ={s.std:.2f} n={s.n:<3d}")
        print(f"  {axis:<22s}  " + "  ".join(f"{c:<24s}" for c in cells))


def report_correlation(corr, k=8):
    if corr is None:
        print("  [skipped — too few records for correlation matrix]")
        return
    print("  Top correlated axis pairs (by |r|):")
    for a, b, r in correlation_top_pairs(corr, k=k):
        sign = "+" if r >= 0 else "-"
        print(f"    {a:<22s} ↔ {b:<22s}  r = {sign}{abs(r):.2f}")


def report_covariance(cov):
    if cov is None:
        print("  [skipped — too few records for covariance estimation]")
        print(f"  (need ≥ {30} total records with ≥ {5} per persona)")
        return
    print("  Per-persona records (full only):")
    for p, n in sorted(cov.n_per_persona.items()):
        print(f"    {p:<24s}  n = {n}")
    print()
    # Display per-axis diagonal of pooled within-class covariance:
    # this is the per-axis variance attributable to sample-to-sample
    # noise within a persona — the noise floor for Mahalanobis.
    print("  Pooled within-persona std (the noise floor per axis):")
    for j, axis in enumerate(AXIS_NAMES):
        sigma = float(np.sqrt(max(cov.pooled_within[j, j], 0)))
        print(f"    {axis:<22s}  σ = {sigma:.3f}")
    print()
    print("  Pairwise Mahalanobis distance between persona means:")
    dists = pairwise_persona_distances(cov)
    if not dists:
        print("    [covariance singular, cannot invert]")
        return
    for (a, b), d in sorted(dists.items(), key=lambda kv: -kv[1]):
        # Rule of thumb: Mahalanobis > 3 ≈ very distinct, > 2 ≈ distinct,
        # < 1 ≈ overlapping (under multivariate Gaussian assumption).
        if d > 3.0: tag = "VERY DISTINCT"
        elif d > 2.0: tag = "distinct"
        elif d > 1.0: tag = "separated"
        else: tag = "OVERLAPPING"
        print(f"    {a:<22s} ↔ {b:<22s}  d = {d:.2f}  [{tag}]")


def report_pca(pca_layer, thresholds=(0.80, 0.90, 0.95)):
    if pca_layer is None:
        print("  [skipped — too few records for PCA]")
        print(f"  (need ≥ {2 * N_AXES} full records for minimum sane estimate;")
        print(f"   ≥ 100 recommended for stable principal directions)")
        return
    print(f"  Computed over n = {pca_layer.n_samples} full records.")
    print()
    # Variance explained, per component + cumulative.
    print(f"  {'PC':<5s}  {'eigenvalue':<12s}  {'frac':<8s}  {'cum':<8s}  top loadings")
    for i in range(N_AXES):
        ev = pca_layer.variance_explained[i]
        frac = ev / pca_layer.variance_explained.sum()
        cum = pca_layer.cumulative_explained[i]
        top = top_loadings(pca_layer, i, k=3)
        top_str = ", ".join(f"{name}({load:+.2f})" for name, load in top)
        print(f"  PC{i+1:<3d}  {ev:<12.3f}  {frac:<8.3f}  {cum:<8.3f}  {top_str}")
    print()
    print("  Effective dimensionality (# PCs to reach variance threshold):")
    for thr in thresholds:
        k = effective_dimensionality(pca_layer, threshold=thr)
        ratio = k / N_AXES
        flag = "  ← over-parameterized" if ratio < 0.5 else ""
        print(f"    ≥ {thr:.0%} variance:  {k} of {N_AXES} components  "
              f"({ratio:.0%} of basis){flag}")


def signature_to_json(fs, corr_top=10, pca_thresholds=(0.80, 0.90, 0.95)):
    """Serialize a full_signature() result to a single JSON-ready dict.

    Mirrors the printed report structure (layer-by-layer) so the design-UI
    can render the same content client-side without re-parsing prose. Used
    by the plugin's POST /analyze endpoint and by `analyze.py --json`.
    """
    out = {
        "n_records": int(fs.n_records),
        "n_full": int(fs.n_full),
        "n_per_persona": {p: int(n) for p, n in fs.n_per_persona.items()},
        "layer1_univariate": {},
        "layer2_correlation_top": None,
        "layer3_covariance": None,
        "layer4_pca": None,
    }
    # Layer 1: univariate per-persona per-axis.
    for persona, by_axis in (fs.univariate_per_persona or {}).items():
        out["layer1_univariate"][persona] = {}
        for axis, s in by_axis.items():
            if s is None:
                continue
            out["layer1_univariate"][persona][axis] = {
                "median": float(s.median),
                "std": float(s.std),
                "n": int(s.n),
            }
    # Layer 2: top correlated axis pairs.
    if fs.correlation is not None:
        out["layer2_correlation_top"] = [
            {"a": a, "b": b, "r": float(r)}
            for a, b, r in correlation_top_pairs(fs.correlation, k=corr_top)
        ]
    # Layer 3: covariance + Mahalanobis.
    if fs.covariance_layer is not None:
        cov = fs.covariance_layer
        out["layer3_covariance"] = {
            "n_per_persona": {p: int(n) for p, n in cov.n_per_persona.items()},
            "pooled_within_sigma": {
                axis: float(np.sqrt(max(cov.pooled_within[j, j], 0.0)))
                for j, axis in enumerate(AXIS_NAMES)
            },
            "pairwise_persona_distances": None,
        }
        dists = pairwise_persona_distances(cov)
        if dists:
            tagged = []
            for (a, b), d in sorted(dists.items(), key=lambda kv: -kv[1]):
                if d > 3.0:
                    tag = "VERY DISTINCT"
                elif d > 2.0:
                    tag = "distinct"
                elif d > 1.0:
                    tag = "separated"
                else:
                    tag = "OVERLAPPING"
                tagged.append({"a": a, "b": b, "d": float(d), "tag": tag})
            out["layer3_covariance"]["pairwise_persona_distances"] = tagged
    # Layer 4: PCA.
    if fs.pca_layer is not None:
        pca = fs.pca_layer
        ev_sum = float(pca.variance_explained.sum())
        components = []
        for i in range(N_AXES):
            ev = float(pca.variance_explained[i])
            components.append({
                "index": i + 1,
                "eigenvalue": ev,
                "frac": ev / ev_sum if ev_sum > 0 else 0.0,
                "cum": float(pca.cumulative_explained[i]),
                "top_loadings": [
                    {"axis": name, "load": float(load)}
                    for name, load in top_loadings(pca, i, k=3)
                ],
            })
        out["layer4_pca"] = {
            "n_samples": int(pca.n_samples),
            "components": components,
            "effective_dimensionality": {
                f"{thr:.2f}": int(effective_dimensionality(pca, threshold=thr))
                for thr in pca_thresholds
            },
            "n_axes": N_AXES,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="JSONL judgment store")
    p.add_argument("--persona", help="filter to a single persona")
    p.add_argument("--corr-top", type=int, default=10,
                   help="how many top correlated pairs to show")
    p.add_argument("--json", action="store_true",
                   help="emit a single JSON object on stdout instead of "
                        "the human-readable report")
    args = p.parse_args()

    records = load_records(args.path)
    if args.persona:
        records = [r for r in records if r.get("persona") == args.persona]

    if not records:
        if args.json:
            print(json.dumps({"error": f"no records found at {args.path}",
                              "path": args.path}))
            sys.exit(1)
        print(f"no records found at {args.path}", file=sys.stderr)
        sys.exit(1)

    fs = full_signature(records)

    if args.json:
        # Single JSON object on stdout. The plugin's POST /analyze
        # endpoint reads exactly this and re-exposes it as the HTTP
        # response body, so the design-UI gets the same data the
        # human report renders.
        json.dump({"path": args.path, "persona_filter": args.persona,
                   **signature_to_json(fs, corr_top=args.corr_top)},
                  sys.stdout)
        sys.stdout.write("\n")
        return

    banner(f"ELICITATION SIGNATURE  ·  {args.path}")
    print(f"  total records:      {fs.n_records}")
    print(f"  full (14/14) only:  {fs.n_full}")
    print(f"  per-persona (full): {fs.n_per_persona}")

    banner("LAYER 1 — univariate per-axis per-persona")
    report_univariate(fs)

    banner("LAYER 2 — bivariate Pearson correlation across full records")
    report_correlation(fs.correlation, k=args.corr_top)

    banner("LAYER 3 — pooled within-persona covariance + Mahalanobis distances")
    report_covariance(fs.covariance_layer)

    banner("LAYER 4 — PCA on pooled sample matrix + effective dimensionality")
    report_pca(fs.pca_layer)

    print()


if __name__ == "__main__":
    main()
