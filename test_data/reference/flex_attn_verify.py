#!/usr/bin/env python
"""
Verify flex_attn_slide_v1_q8 output against a numpy reference attention.

Input shapes (from Swift runFlexAttnSlideV1Test):
  flex_test_Q.npy        [B=1, Q_LEN=8, H_Q=16, D=256]
  flex_test_K.npy        [PAGE=16, H_KV=8, D=256]     (phys page 0)
  flex_test_V.npy        [PAGE=16, H_KV=8, D=256]
  flex_test_O_kernel.npy [B=1, Q_LEN=8, H_Q=16, D=256]  kernel output
Config:
  q_positions = [0, 1, 2, 3, 4, 5, 6, 7]
  k_len = 8   (only positions 0..7 are valid K entries; 8..15 are noise/masked)
  sliding_window disabled (0 → no lower bound)
  qk_scale = 1.0 (Gemma-4 scaling)
  causal mask only: k_pos <= q_pos
  GQA: Q_PER_KV = H_Q/H_KV = 2

Per-head numpy reference:
  For slot 0, q_head q, kv_head kh = q // 2:
    for each q_pos p in 0..Q_LEN-1:
      scores[k] = sum_d Q[p, q, d] * K[k, kh, d]   for k in 0..k_len-1
      scores[k] = -INF for k > p  (causal)
      probs    = softmax(scores)
      O[p, q, :] = sum_k probs[k] * V[k, kh, :]
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

Q  = np.load(d / "flex_test_Q.npy")        # [1, 8, 16, 256]
K  = np.load(d / "flex_test_K.npy")        # [16, 8, 256]  (first 8 rows are valid)
V  = np.load(d / "flex_test_V.npy")
Ok = np.load(d / "flex_test_O_kernel.npy")

B, Q_LEN, H_Q, D = Q.shape
H_KV = K.shape[1]
k_len = 8
assert Q.shape == (1, 8, 16, 256)
print(f"Q {Q.shape}  K {K.shape}  V {V.shape}  O_kernel {Ok.shape}")

# Build reference output.
O_ref = np.zeros_like(Q)
Q_PER_KV = H_Q // H_KV
for p in range(Q_LEN):          # q position
    for q in range(H_Q):        # q head
        kh = q // Q_PER_KV
        # scores [k_len]
        Qvec = Q[0, p, q, :].astype(np.float32)
        Kmat = K[:k_len, kh, :].astype(np.float32)
        Vmat = V[:k_len, kh, :].astype(np.float32)
        scores = Kmat @ Qvec              # [k_len]
        # Causal mask: k > p → -INF
        for k in range(k_len):
            if k > p:
                scores[k] = -np.inf
        # softmax (no sqrt(D) scaling; Gemma-4 qk_scale=1.0).
        m = scores.max()
        e = np.exp(scores - m)
        s = e.sum()
        probs = e / s
        O_ref[0, p, q, :] = probs @ Vmat

# Compare.
diff = Ok - O_ref
print(f"max|diff|: {np.abs(diff).max():.6f}")
print(f"mean|diff|: {np.abs(diff).mean():.6f}")
cos = float((Ok.flatten() @ O_ref.flatten()) /
            (np.linalg.norm(Ok) * np.linalg.norm(O_ref) + 1e-12))
print(f"cos-sim  : {cos:.6f}")
# Per-(pos, head) cos
fails = 0
for p in range(Q_LEN):
    for q in range(H_Q):
        a = Ok[0, p, q, :]; b = O_ref[0, p, q, :]
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if c < 0.999:
            print(f"  FAIL p={p} q={q}: cos={c:.4f}  max|d|={np.abs(a - b).max():.3f}")
            fails += 1
if fails == 0:
    print(f"✓ all {Q_LEN}×{H_Q}={Q_LEN * H_Q} (pos, head) pairs match (cos ≥ 0.999)")
else:
    print(f"✗ {fails} / {Q_LEN * H_Q} pairs diverged")
    sys.exit(1)
