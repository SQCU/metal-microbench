#!/usr/bin/env python3
"""Metal-native Q/K/V capture + selectivity analysis.

Calls the bridge's /v1/control/capture_qkv endpoint (which the server
owns, running in the already-initialized engine process) to get per-
seed per-layer per-head Q/K/V. Then does the offline analysis:
diff-of-means, (L, q_head) top-p truncation, Q·K*/√d selectivity
metric comparing refusal vs compliance query distributions.

Run:
    cd /Users/mdot/metal-microbench
    python3 notes/qkv_capture_test.py
"""
from __future__ import annotations
import base64
import json
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np


SERVER = "http://127.0.0.1:8000"
SEEDS_FILE = "/tmp/on_policy_seeds.json"


def rpc(path, body, timeout=300):
    req = urllib.request.Request(
        SERVER + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def capture_class(seeds: list[str]) -> dict:
    return rpc("/v1/control/capture_qkv", {"seeds": seeds, "wrap": True})


def decode_batch(b64_list: list[str]) -> np.ndarray:
    """Return [N_seeds, total_halves] fp16 -> float32."""
    arrs = [np.frombuffer(base64.b64decode(b), dtype=np.float16).astype(np.float32)
            for b in b64_list]
    return np.stack(arrs, axis=0)


def per_layer_per_head(raw: np.ndarray, shapes: list, stride: int,
                        which: str) -> dict:
    """raw: [N, NUM_LAYERS * stride_halves]. Returns
    {layer: [N, num_heads, HD] fp32}, where num_heads comes from
    shapes[L][0] for Q or shapes[L][1] for K/V."""
    out = {}
    for L, (H, KV_H, HD, _) in enumerate(shapes):
        heads = H if which == "q" else KV_H
        start = L * stride
        per = raw[:, start:start + heads * HD]
        out[L] = per.reshape(-1, heads, HD)
    return out


def main():
    print("=" * 64)
    print("METAL Q/K/V CAPTURE + SELECTIVITY ANALYSIS")
    print("=" * 64)

    with open(SEEDS_FILE) as f:
        d = json.load(f)
    pos, neg = d["positive"], d["negative"]
    N = 6
    pos = pos[:N]; neg = neg[:N]

    t0 = time.time()
    print(f"\n[1] capturing refusal class ({N} seeds via engine)")
    pos_resp = capture_class(pos)
    t1 = time.time()
    print(f"    {t1 - t0:.1f}s  ({N / (t1 - t0):.1f} seeds/s)")
    print(f"[2] capturing compliance class ({N} seeds)")
    neg_resp = capture_class(neg)
    print(f"    {time.time() - t1:.1f}s")

    shapes = [tuple(r) for r in pos_resp["shapes"]]
    q_stride = pos_resp["q_stride_halves"]
    kv_stride = pos_resp["kv_stride_halves"]
    num_layers = len(shapes)
    print(f"\n[3] {num_layers} layers; q_stride={q_stride}  kv_stride={kv_stride}")
    print(f"    first layer shape: H={shapes[0][0]} KV_H={shapes[0][1]} "
          f"HD={shapes[0][2]} full={shapes[0][3]}")

    pos_Q = decode_batch(pos_resp["q_b64"])
    pos_V = decode_batch(pos_resp["v_b64"])
    neg_Q = decode_batch(neg_resp["q_b64"])
    neg_V = decode_batch(neg_resp["v_b64"])

    pos_q_L = per_layer_per_head(pos_Q, shapes, q_stride, "q")
    pos_v_L = per_layer_per_head(pos_V, shapes, kv_stride, "v")
    neg_q_L = per_layer_per_head(neg_Q, shapes, q_stride, "q")
    neg_v_L = per_layer_per_head(neg_V, shapes, kv_stride, "v")

    print("\n[4] per-(layer, q_head) diff-of-means + intervention-mass score")
    scores = []  # (L, q_head, kv_head, q_mag, v_mag, score)
    for L, (H, KV_H, HD, is_full) in enumerate(shapes):
        group = H // KV_H
        q_diff = pos_q_L[L].mean(0) - neg_q_L[L].mean(0)   # [H, HD]
        v_diff = pos_v_L[L].mean(0) - neg_v_L[L].mean(0)   # [KV_H, HD]
        q_mag = np.linalg.norm(q_diff, axis=1)             # [H]
        v_mag = np.linalg.norm(v_diff, axis=1)             # [KV_H]
        for h_q in range(H):
            h_kv = h_q // group
            scores.append((L, h_q, h_kv,
                           float(q_mag[h_q]), float(v_mag[h_kv]),
                           float(q_mag[h_q] * v_mag[h_kv])))

    scores.sort(key=lambda t: -t[5])
    total = sum(t[5] for t in scores)
    print(f"    total mass across all (L, q_head): {total:.1f}")
    print(f"    top 10 by score:")
    for row in scores[:10]:
        L, hq, hkv, qm, vm, sc = row
        is_full = shapes[L][3]
        tag = "FULL" if is_full else "slide"
        print(f"      L{L:02d} qH{hq:02d} kvH{hkv}  q_mag={qm:.2f}  "
              f"v_mag={vm:.2f}  score={sc:.1f}  [{tag}]")

    print("\n[5] top-p truncation over (layer, q_head)")
    for top_p in (0.30, 0.50, 0.80):
        kept = []
        cum = 0.0
        for row in scores:
            if cum / total >= top_p: break
            kept.append(row); cum += row[5]
        layer_hist = defaultdict(int)
        for r in kept: layer_hist[r[0]] += 1
        by_count = sorted(layer_hist.items(), key=lambda x: -x[1])[:8]
        print(f"  top_p={top_p}  k={len(kept)}/{len(scores)}  "
              f"covers {len(layer_hist)}/{num_layers} layers")
        print(f"    most-concentrated layers (L, count): {by_count}")

    print("\n[6] selectivity: Q·K*/√d per class")
    top_p = 0.50
    kept = []
    cum = 0.0
    for row in scores:
        if cum / total >= top_p: break
        kept.append(row); cum += row[5]

    refusal_logits = []
    compliance_logits = []
    for (L, h_q, h_kv, qm, vm, sc) in kept:
        H, KV_H, HD, _ = shapes[L]
        q_diff = pos_q_L[L].mean(0)[h_q] - neg_q_L[L].mean(0)[h_q]
        k_star = q_diff / (np.linalg.norm(q_diff) + 1e-8)
        scale = 1.0 / np.sqrt(HD)
        for n in range(N):
            refusal_logits.append(float(pos_q_L[L][n, h_q] @ k_star * scale))
            compliance_logits.append(float(neg_q_L[L][n, h_q] @ k_star * scale))
    refusal_logits = np.array(refusal_logits)
    compliance_logits = np.array(compliance_logits)
    print(f"    kept {len(kept)} pairs at top_p={top_p}")
    print(f"    refusal-class   Q·K*/√d:  mean={refusal_logits.mean():+.3f}  "
          f"std={refusal_logits.std():.3f}")
    print(f"    compliance-class Q·K*/√d: mean={compliance_logits.mean():+.3f}  "
          f"std={compliance_logits.std():.3f}")
    sep = refusal_logits.mean() - compliance_logits.mean()
    print(f"    separation: {sep:+.3f}")

    # Synthetic-slot attention weight with S real keys at logit ~0.
    S = 50
    wr = np.exp(refusal_logits) / (np.exp(refusal_logits) + S)
    wc = np.exp(compliance_logits) / (np.exp(compliance_logits) + S)
    print(f"\n    simulated synth-slot weight (context S={S}, baseline 0):")
    print(f"    refusal-class  attention weight  mean={wr.mean()*100:.3f}%  max={wr.max()*100:.3f}%")
    print(f"    compliance-class attention weight mean={wc.mean()*100:.3f}%  max={wc.max()*100:.3f}%")
    print(f"    ratio (refusal / compliance): {wr.mean() / (wc.mean() + 1e-12):.2f}×")
    print(f"\n  collateral: constant-bias applies at 100% weight on compliance-class.")
    print(f"              synthetic-KV would apply at {wc.mean()*100:.3f}%.")
    print(f"              ratio: ~{100.0/(wc.mean()*100 + 1e-9):.0f}× less collateral.")


if __name__ == "__main__":
    main()
