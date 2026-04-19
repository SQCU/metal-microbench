#!/usr/bin/env python
"""
Diff per-layer residual-stream snapshots dumped by the HF oracle
(`extract_lm_logits.py --tag <tag>` → `lm_<tag>_hiddens.npy`) and by the Swift
layer-dump harness (`LM_DUMP_LAYERS=<dir>` → `lm_swift_hiddens.npy`).

Both arrays are fp32 shape [NUM_LAYERS+1, S, HIDDEN]:
  idx 0              = residual after embed_tokens + sqrt(hidden_size) scale
                       (pre decoder_layer[0])
  idx L (L=1..L_MAX) = residual after decoder_layer[L-1]

Reports for each boundary (for slot 0 at each position):
  - mean/max absolute difference
  - cosine similarity
  - relative L2 error
  - oracle and Swift vector norms

Also flags the FIRST boundary where cos-sim drops below a threshold — that's
the leading suspect for the divergence source.

Run:
    .venv/bin/python compare_lm_hiddens.py --tag hello
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hello")
    ap.add_argument("--dir", default=str(Path(__file__).parent))
    ap.add_argument("--position", type=int, default=0,
                    help="Which prompt position to focus detailed stats on; -1 means all.")
    ap.add_argument("--cos-threshold", type=float, default=0.99,
                    help="Cos-sim below this flags a boundary as divergent.")
    args = ap.parse_args()

    d = Path(args.dir)
    oracle = np.load(d / f"lm_{args.tag}_hiddens.npy")        # [L+1, S, H] fp32
    swift = np.load(d / "lm_swift_hiddens.npy")                # [L+1, S, H] fp32
    assert oracle.shape == swift.shape, f"shape mismatch: oracle {oracle.shape} vs swift {swift.shape}"
    L1, S, H = oracle.shape
    print(f"shape: [{L1}, {S}, {H}]  (boundaries, positions, hidden)")
    print(f"oracle dtype: {oracle.dtype}  swift dtype: {swift.dtype}")

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    # Optional: layer-0 intra-layer probes.
    l0_ora_path = d / f"lm_{args.tag}_l0_probes.npy"
    l0_swi_path = d / "lm_swift_l0_probes.npy"
    l0_ora = np.load(l0_ora_path) if l0_ora_path.exists() else None
    l0_swi = np.load(l0_swi_path) if l0_swi_path.exists() else None
    probe_names = ["post_attn+res", "ffw_norm_1 out", "ffw_norm_2 out",
                   "pre_ffw_2 out", "moe_sum raw   "]

    positions = range(S) if args.position < 0 else [args.position]
    for p in positions:
        print(f"\n=== position {p} ===")
        if l0_ora is not None and l0_swi is not None:
            assert l0_ora.shape == l0_swi.shape, f"L0 probe shape mismatch: oracle {l0_ora.shape} vs swift {l0_swi.shape}"
            print("L0 intra-layer probes:")
            print(f"{'probe':<16} {'cos':>8}  {'max|d|':>9}  {'rel-L2':>9}  {'||ora||':>9}  {'||swi||':>9}")
            for i, name in enumerate(probe_names[:l0_ora.shape[0]]):
                o = l0_ora[i, p]; s = l0_swi[i, p]
                d_ = s - o
                c = cos(o, s)
                rl = float(np.linalg.norm(d_) / (np.linalg.norm(o) + 1e-12))
                print(f"{name:<16} {c:8.4f}  {float(np.max(np.abs(d_))):9.3f}  {rl:9.4f}  {float(np.linalg.norm(o)):9.3f}  {float(np.linalg.norm(s)):9.3f}")
            print()

        print(f"{'boundary':<12} {'cos':>8}  {'max|d|':>9}  {'rel-L2':>9}  {'||ora||':>9}  {'||swi||':>9}  {'mean|d|':>9}")
        first_bad: int | None = None
        for L in range(L1):
            o = oracle[L, p]
            s = swift[L, p]
            d = s - o
            c = cos(o, s)
            max_abs = float(np.max(np.abs(d)))
            rel_l2 = float(np.linalg.norm(d) / (np.linalg.norm(o) + 1e-12))
            no = float(np.linalg.norm(o))
            ns = float(np.linalg.norm(s))
            mean_abs = float(np.mean(np.abs(d)))
            label = "embed" if L == 0 else f"layer {L - 1}"
            marker = ""
            if c < args.cos_threshold:
                marker = "  ←"
                if first_bad is None:
                    first_bad = L
            print(f"{label:<12} {c:8.4f}  {max_abs:9.4f}  {rel_l2:9.4f}  {no:9.3f}  {ns:9.3f}  {mean_abs:9.4f}{marker}")
        if first_bad is not None:
            fb_label = "embed" if first_bad == 0 else f"layer {first_bad - 1}"
            print(f"\n  FIRST divergent boundary (cos<{args.cos_threshold}): {fb_label} (idx {first_bad})")
        else:
            print(f"\n  all boundaries have cos>= {args.cos_threshold}")


if __name__ == "__main__":
    main()
