#!/usr/bin/env python
"""Verify flex_attn_full_prefill output against numpy reference.

Geometry: B=1, Q_LEN=8, H_Q=16, H_KV=2, HD=512, PAGE=8.
kv_head for q_head q is q // (H_Q/H_KV) = q // 8.
qk_scale = 1.0 (Gemma-4), causal mask only (full-attn = global, no window).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
Q  = np.load(d / "flex_full_test_Q.npy")
K  = np.load(d / "flex_full_test_K.npy")
V  = np.load(d / "flex_full_test_V.npy")
Ok = np.load(d / "flex_full_test_O_kernel.npy")
print(f"Q {Q.shape}  K {K.shape}  V {V.shape}  O {Ok.shape}")

B, Q_LEN, H_Q, D = Q.shape
H_KV = K.shape[1]
k_len = Q_LEN
Q_PER_KV = H_Q // H_KV

O_ref = np.zeros_like(Q)
for p in range(Q_LEN):
    for q in range(H_Q):
        kh = q // Q_PER_KV
        Qvec = Q[0, p, q].astype(np.float32)
        Kmat = K[:k_len, kh].astype(np.float32)
        Vmat = V[:k_len, kh].astype(np.float32)
        scores = Kmat @ Qvec
        for k in range(k_len):
            if k > p:
                scores[k] = -np.inf
        m = scores.max(); e = np.exp(scores - m); s = e.sum()
        probs = e / s
        O_ref[0, p, q] = probs @ Vmat

diff = Ok - O_ref
print(f"max|diff|: {np.abs(diff).max():.6f}")
print(f"mean|diff|: {np.abs(diff).mean():.6f}")
cos = float(Ok.flatten() @ O_ref.flatten() /
            (np.linalg.norm(Ok) * np.linalg.norm(O_ref) + 1e-12))
print(f"cos-sim  : {cos:.6f}")
fails = 0
for p in range(Q_LEN):
    for q in range(H_Q):
        a = Ok[0, p, q]; b = O_ref[0, p, q]
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if c < 0.999:
            print(f"  FAIL p={p} q={q}: cos={c:.4f}  max|d|={np.abs(a-b).max():.3f}")
            fails += 1
if fails == 0:
    print(f"✓ all {Q_LEN}×{H_Q}={Q_LEN*H_Q} (pos, head) pairs match (cos ≥ 0.999)")
else:
    print(f"✗ {fails} / {Q_LEN*H_Q} pairs diverged")
    sys.exit(1)
