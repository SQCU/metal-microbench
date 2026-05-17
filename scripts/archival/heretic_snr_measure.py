#!/usr/bin/env python3
"""Compute representation-space SNR metrics for candidate direction
vectors under the heretic mechanism, per the protocol in
notes/heretic_snr_hypotheses.md.

Metrics computed per-direction per-layer:
  F(r̂, L)          = Fisher discriminant:  (r̂ᵀ(μ+−μ−))² / r̂ᵀ(Σ++Σ−)r̂
  spurious(r̂, L)   = |r̂ᵀ(μ_0 − ½(μ++μ−))|
  nuisance_load    = ||P_{span(nuisance)} r̂||  (if nuisance basis supplied)

Before/after Gram-Schmidt denoising reports so hypothesis G1 (≥1.5× F)
can be falsified cleanly. No engine intervention needed — this is all
numpy math on captured residuals.

Usage:
    python3 notes/heretic_snr_measure.py \
        --positive /tmp/on_policy_seeds.json:positive \
        --negative /tmp/on_policy_seeds.json:negative \
        --control  notes/heretic_snr_control.json \
        --direction-ids ref1st-L08-C0,ref1st-L09-C0,ref1st-L10-C0,ref1st-L11-C0 \
        --nuisance-ids formal-L10-C0,formal-L14-C0      # optional
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import struct
import sys
import urllib.request
import urllib.error

import numpy as np


BASE = "http://127.0.0.1:8000"
HIDDEN = 2816
NUM_LAYERS = 30


# ── Default control set: held-out benign user questions. Residuals here
# should NOT project onto a clean refusal-direction. Kept small so the
# capture pass fits in ~10 s on our server. ────────────────────────────
DEFAULT_CONTROL = [
    "What is the capital of France?",
    "Explain the water cycle.",
    "How do vaccines work?",
    "What causes rain?",
    "Describe the structure of an atom.",
    "What is the difference between a dolphin and a porpoise?",
    "How does a computer's CPU work?",
    "What are the primary colors of light?",
    "What is the Pythagorean theorem?",
    "How does photosynthesis produce oxygen?",
    "Why does ice float on water?",
    "What is DNA made of?",
]


# ── HTTP helpers ─────────────────────────────────────────────────────

def post_json(path: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_json(path: str) -> dict:
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_cvec(cvec_id: str) -> np.ndarray:
    """Return the registered fp16 bytes for a cvec_id as an fp32 array
    of length HIDDEN. Raises KeyError if not registered."""
    try:
        r = get_json(f"/v1/control/vectors/{cvec_id}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise KeyError(f"cvec_id {cvec_id!r} not registered")
        raise
    raw = base64.b64decode(r["fp16_b64"])
    arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    assert arr.shape == (HIDDEN,), f"expected HIDDEN={HIDDEN}, got {arr.shape}"
    return arr


def capture_residuals(seeds: list[str], wrap: bool = True) -> np.ndarray:
    """Capture end-of-prose residuals for each seed at every layer.
    Returns [n_seeds, NUM_LAYERS, HIDDEN] fp32 via the existing
    /v1/control/capture_residuals endpoint."""
    r = post_json("/v1/control/capture_residuals",
                   {"seeds": seeds, "wrap": wrap}, timeout=600.0)
    out = np.zeros((len(seeds), NUM_LAYERS, HIDDEN), dtype=np.float32)
    for i, b64 in enumerate(r["residuals_b64"]):
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        out[i] = arr.reshape(NUM_LAYERS, HIDDEN)
    return out


# ── Prompt-set loaders ───────────────────────────────────────────────

def load_prompts(spec: str) -> list[str]:
    """Accepts:
      "path.json:key"       → json[key] must be list[str]
      "path.json"           → json must be list[str]
      "path.txt"            → one prompt per line
      literal string list with semicolons if no file
    """
    p = pathlib.Path(spec.split(":", 1)[0])
    if not p.exists():
        # Treat spec as a newline-or-semicolon-separated inline list.
        return [s.strip() for s in spec.replace("\n", ";").split(";") if s.strip()]
    if p.suffix == ".json":
        with open(p) as f: d = json.load(f)
        if ":" in spec:
            key = spec.split(":", 1)[1]
            return [str(x) for x in d[key]]
        if isinstance(d, list): return [str(x) for x in d]
        raise ValueError(f"{p}: expected list or ':key'-indexed dict")
    # Text file: one prompt per line.
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


# ── Metrics ──────────────────────────────────────────────────────────

def fisher_ratio(r: np.ndarray, H_plus: np.ndarray,
                  H_minus: np.ndarray) -> float:
    """F(r̂) = (r̂ᵀ(μ+ − μ−))² / (r̂ᵀ(Σ+ + Σ−)r̂), computed via the
    scalar-projection shortcut that avoids materializing Σ.
    H_plus, H_minus: [n_samples, HIDDEN] per-class residual matrices
    at a single layer."""
    r = r / (np.linalg.norm(r) + 1e-12)
    mu_plus  = H_plus.mean(axis=0)
    mu_minus = H_minus.mean(axis=0)
    mu_diff  = float(r @ (mu_plus - mu_minus))
    # r̂ᵀΣ±r̂ = var(projections along r̂) — scalar per class.
    proj_plus  = H_plus  @ r
    proj_minus = H_minus @ r
    within = float(proj_plus.var() + proj_minus.var()) + 1e-12
    return (mu_diff ** 2) / within


def spurious_projection(r: np.ndarray, H_plus: np.ndarray,
                          H_minus: np.ndarray, H_zero: np.ndarray) -> float:
    """|r̂ᵀ(μ_0 − ½(μ+ + μ−))|  — how much the control class's mean
    residual projects onto r̂ relative to the contrastive midpoint."""
    r = r / (np.linalg.norm(r) + 1e-12)
    mid = 0.5 * (H_plus.mean(axis=0) + H_minus.mean(axis=0))
    return abs(float(r @ (H_zero.mean(axis=0) - mid)))


def gram_schmidt_clean(r: np.ndarray, nuisance_basis: np.ndarray) -> np.ndarray:
    """r_clean = r − Σᵢ (nᵢᵀr)·nᵢ, then renormalized. `nuisance_basis`
    is [k, HIDDEN] with rows assumed orthonormal (we re-orthonormalize
    to be safe)."""
    if nuisance_basis.size == 0:
        return r / (np.linalg.norm(r) + 1e-12)
    # Re-orthonormalize the basis to guard against near-collinear inputs.
    Q, _ = np.linalg.qr(nuisance_basis.T)
    N = Q.T  # [rank, HIDDEN]
    coef = N @ r
    r_clean = r - coef @ N
    n = np.linalg.norm(r_clean)
    if n < 1e-9:
        return r / (np.linalg.norm(r) + 1e-12)
    return r_clean / n


def nuisance_load(r: np.ndarray, nuisance_basis: np.ndarray) -> float:
    """||P_{span(N)} r̂|| — fraction of r̂'s energy projecting onto the
    nuisance subspace. 0 = pristine, 1 = entirely nuisance."""
    if nuisance_basis.size == 0:
        return 0.0
    r = r / (np.linalg.norm(r) + 1e-12)
    Q, _ = np.linalg.qr(nuisance_basis.T)
    coef = Q.T @ r
    return float(np.linalg.norm(coef))


# ── Reporting ────────────────────────────────────────────────────────

def per_layer_report(name: str, r_per_layer: list[np.ndarray],
                       H_plus: np.ndarray, H_minus: np.ndarray,
                       H_zero: np.ndarray,
                       nuisance_basis_per_layer: list[np.ndarray] | None = None
                      ) -> dict:
    """Compute F, spurious, nuisance-load for a direction at every layer,
    plus its GS-cleaned variant if a nuisance basis is supplied.
    `r_per_layer[L]` is the direction used at layer L (length HIDDEN)."""
    rows = []
    for L in range(NUM_LAYERS):
        r = r_per_layer[L]
        nb = (nuisance_basis_per_layer[L]
              if nuisance_basis_per_layer is not None
              else np.zeros((0, HIDDEN)))
        F_raw  = fisher_ratio(r, H_plus[:, L], H_minus[:, L])
        sp_raw = spurious_projection(r, H_plus[:, L], H_minus[:, L], H_zero[:, L])
        nl_raw = nuisance_load(r, nb)
        row = {"layer": L, "F_raw": F_raw, "spurious_raw": sp_raw,
               "nuisance_load_raw": nl_raw}
        if nb.size > 0:
            r_clean = gram_schmidt_clean(r, nb)
            row["F_clean"] = fisher_ratio(r_clean, H_plus[:, L], H_minus[:, L])
            row["spurious_clean"] = spurious_projection(
                r_clean, H_plus[:, L], H_minus[:, L], H_zero[:, L])
            row["nuisance_load_clean"] = nuisance_load(r_clean, nb)
        rows.append(row)
    return {"direction": name, "per_layer": rows}


def print_report(report: dict, fmax_only: bool = False) -> None:
    rows = report["per_layer"]
    have_clean = "F_clean" in rows[0]
    best = max(rows, key=lambda r: r["F_raw"])
    print(f"\n=== {report['direction']} ===")
    if have_clean:
        best_clean = max(rows, key=lambda r: r["F_clean"])
        print(f"  Fmax_raw = {best['F_raw']:.3f} @ L{best['layer']}  "
              f"(spurious={best['spurious_raw']:.3f}, "
              f"nuis_load={best['nuisance_load_raw']:.3f})")
        print(f"  Fmax_clean = {best_clean['F_clean']:.3f} @ L{best_clean['layer']}  "
              f"(spurious={best_clean['spurious_clean']:.3f}, "
              f"nuis_load={best_clean['nuisance_load_clean']:.3f})")
        gain = best_clean['F_clean'] / max(best['F_raw'], 1e-9)
        print(f"  gain (Fmax_clean / Fmax_raw) = {gain:.2f}×")
    else:
        print(f"  Fmax_raw = {best['F_raw']:.3f} @ L{best['layer']}  "
              f"(spurious={best['spurious_raw']:.3f})")
    if fmax_only:
        return
    # Per-layer detail.
    print(f"  {'L':>3}  {'F_raw':>8}  {'sp_raw':>8}", end="")
    if have_clean:
        print(f"  {'F_clean':>8}  {'sp_clean':>8}  {'gain':>6}", end="")
    print()
    for row in rows:
        print(f"  {row['layer']:>3}  {row['F_raw']:>8.3f}  "
              f"{row['spurious_raw']:>8.3f}", end="")
        if have_clean:
            g = row['F_clean'] / max(row['F_raw'], 1e-9)
            print(f"  {row['F_clean']:>8.3f}  {row['spurious_clean']:>8.3f}  "
                  f"{g:>5.2f}×", end="")
        print()


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive", default="/tmp/on_policy_seeds.json:positive")
    ap.add_argument("--negative", default="/tmp/on_policy_seeds.json:negative")
    ap.add_argument("--control",  default=None,
                     help="prompt set for class 0 (defaults to DEFAULT_CONTROL)")
    ap.add_argument("--direction-ids", required=True,
                     help="comma-separated list of registered cvec_ids to score "
                          "(each is assumed to be the layer-specific direction for "
                          "its embedded LXX; we use each at its own layer)")
    ap.add_argument("--nuisance-ids", default="",
                     help="comma-separated cvec_ids forming the nuisance basis "
                          "(same convention as direction-ids; used PER LAYER)")
    ap.add_argument("--fmax-only", action="store_true",
                     help="print only the best-layer row for each direction")
    ap.add_argument("--save-report", type=pathlib.Path, default=None)
    args = ap.parse_args()

    plus  = load_prompts(args.positive)
    minus = load_prompts(args.negative)
    zero  = load_prompts(args.control) if args.control else DEFAULT_CONTROL
    print(f"capturing residuals: pos={len(plus)} neg={len(minus)} ctrl={len(zero)} "
          f"prompts (each all-layer, wrapped)")

    H_plus  = capture_residuals(plus)
    H_minus = capture_residuals(minus)
    H_zero  = capture_residuals(zero)
    print(f"  shapes: H+ {H_plus.shape}, H- {H_minus.shape}, H0 {H_zero.shape}")

    # Fetch per-layer direction + nuisance bases.
    def group_by_layer(ids_csv: str) -> list[np.ndarray]:
        """For each layer L (0..NUM_LAYERS-1), collect the fp32 direction(s)
        from ids matching 'L{L:02d}'. Returns list[np.ndarray [k_L, HIDDEN]]."""
        if not ids_csv.strip():
            return [np.zeros((0, HIDDEN), dtype=np.float32)] * NUM_LAYERS
        ids = [s.strip() for s in ids_csv.split(",") if s.strip()]
        per_layer: list[list[np.ndarray]] = [[] for _ in range(NUM_LAYERS)]
        for cid in ids:
            # Parse layer index from "XXXX-LNN-CM" — i.e. a 2-digit
            # number between the two atomic markers "-L" and "-".
            # Replaces re.search(r"-L(\d{2})-", cid) with literal find.
            i = cid.find("-L")
            L = -1
            if i >= 0 and i + 4 < len(cid):
                two = cid[i + 2:i + 4]
                if two.isdigit() and cid[i + 4] == "-":
                    L = int(two)
            if L < 0:
                print(f"  [skip] cvec {cid!r}: can't parse layer from id")
                continue
            per_layer[L].append(fetch_cvec(cid))
        return [np.stack(v) if v else np.zeros((0, HIDDEN), dtype=np.float32)
                for v in per_layer]

    # For directions: expect ONE cvec per layer (the direction to evaluate at that layer).
    dir_per_layer_lists = group_by_layer(args.direction_ids)
    # Collapse to one r̂ per layer (first match). If multiple, only the
    # first is evaluated; user can run the script multiple times for
    # alternate directions.
    direction_per_layer: list[np.ndarray] = []
    n_layers_with_dir = 0
    for L in range(NUM_LAYERS):
        if dir_per_layer_lists[L].size > 0:
            direction_per_layer.append(dir_per_layer_lists[L][0])
            n_layers_with_dir += 1
        else:
            # Fall back to zero vector; F will be 0 at this layer.
            direction_per_layer.append(np.zeros(HIDDEN, dtype=np.float32))
    print(f"  loaded direction at {n_layers_with_dir}/{NUM_LAYERS} layers")

    nuisance_per_layer = group_by_layer(args.nuisance_ids)
    if args.nuisance_ids:
        print(f"  nuisance basis sizes per-layer: "
              f"{[int(n.shape[0]) for n in nuisance_per_layer]}")

    # Score.
    rep = per_layer_report(
        name=args.direction_ids, r_per_layer=direction_per_layer,
        H_plus=H_plus, H_minus=H_minus, H_zero=H_zero,
        nuisance_basis_per_layer=(nuisance_per_layer if args.nuisance_ids else None),
    )
    print_report(rep, fmax_only=args.fmax_only)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\nreport → {args.save_report}")


if __name__ == "__main__":
    main()
