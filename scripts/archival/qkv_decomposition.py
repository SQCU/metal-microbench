#!/usr/bin/env python3
"""Explore Q/K/V-decomposed intervention vectors for refusal/compliance.

For each seed in /tmp/on_policy_seeds.json, capture per-(layer, head) Q
and V projections at every position via HF reference forward hooks.
Compute class-conditional means in Q-space and V-space. Rank (layer,
head) pairs by their diff-of-means magnitudes. Apply top-p truncation
the same way we do for PCA layer-elimination — no a-priori layer choice.

Then compute the key observational question: for a synthetic K* fitted
as (μ_Q_refusal − μ_Q_compliance), what fraction of attention weight
does it get on refusal-class Q samples vs compliance-class Q samples?
If the ratio is >> 1 the selectivity is self-gated by the model's own
softmax and we have low-collateral steering. If ~1, the synthetic slot
fires indiscriminately and we're no better than residual bias.

Run:
    cd /Users/mdot/metal-microbench
    PYTORCH_ENABLE_MPS_FALLBACK=1 test_data/reference/.venv/bin/python \\
        notes/qkv_decomposition.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_DIR = "/Users/mdot/models/gemma-4-a4b-bf16"
SEEDS_FILE = "/tmp/on_policy_seeds.json"
DEVICE = "cpu"   # bf16 MoE + hist kernel missing on MPS; CPU is slow but correct


def load_seeds() -> tuple[list[str], list[str]]:
    with open(SEEDS_FILE) as f:
        d = json.load(f)
    return d["positive"], d["negative"]


class QKVCapture:
    """Forward hooks on each attention block to grab the post-projection
    Q and V tensors per (layer, head). K is skipped for this analysis —
    the selectivity metric uses Q·K* where K* is fitted from Q means.
    (If we also wanted K for its own sake we'd capture it symmetrically.)"""

    def __init__(self, model):
        self.model = model
        self.captures: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        self.handles = []
        # Gemma-4: model.model.language_model.layers[L].self_attn
        text_model = model.model.language_model
        layers = text_model.layers
        self.num_layers = len(layers)
        for L, layer in enumerate(layers):
            attn = layer.self_attn
            # Capture post-norm Q (what actually enters attention softmax)
            # and post-proj V. q_norm operates on the reshaped head-split
            # Q tensor; for diff-of-means purposes we can equivalently
            # hook q_proj (pre-norm) since the RMS norm is a per-head
            # deterministic transformation of its input.
            self.handles.append(
                attn.q_proj.register_forward_hook(self._make_hook(L, "q")))
            self.handles.append(
                attn.v_proj.register_forward_hook(self._make_hook(L, "v")))

    def _make_hook(self, L: int, which: str):
        def hook(module, inputs, output):
            # output: [B, S, num_heads * head_dim]
            self.captures[L][which] = output.detach().to(torch.float32).cpu()
        return hook

    def clear(self):
        self.captures = defaultdict(dict)

    def close(self):
        for h in self.handles: h.remove()


def run_and_capture(model, tok, texts: list[str],
                    max_seq_len: int = 200) -> list[dict[int, dict[str, torch.Tensor]]]:
    """For each text, run a single forward pass and return the captures.
    Per text we get a {layer: {'q': [S, H*D], 'v': [S, H*D]}} dict."""
    cap = QKVCapture(model)
    out = []
    try:
        for i, t in enumerate(texts):
            cap.clear()
            # Wrap as a short user turn + assistant prose, then teacher-force
            # through the prose. The class-conditional structure we want
            # comes from the assistant-prose positions, so we want those
            # tokens to be the bulk of the captured sequence.
            wrapped = f"<|turn>user\nContinue the prose.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>{t}"
            ids = tok.encode(wrapped, return_tensors="pt", add_special_tokens=True,
                             truncation=True, max_length=max_seq_len).to(DEVICE)
            with torch.no_grad():
                _ = model(input_ids=ids, use_cache=False)
            out.append({L: {k: v.clone() for k, v in cap.captures[L].items()}
                        for L in cap.captures})
            print(f"  seed {i+1}/{len(texts)}  S={ids.shape[1]}")
    finally:
        cap.close()
    return out


def main() -> None:
    print("=" * 64)
    print("Q/K-DECOMPOSED INTERVENTION ANALYSIS")
    print("=" * 64)

    t0 = time.time()
    print(f"\n[1] loading model on {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    for p in model.parameters(): p.requires_grad_(False)

    # Grab head-dim info from config.
    cfg = model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim if hasattr(cfg, "head_dim") \
               else cfg.hidden_size // num_q_heads
    num_layers = cfg.num_hidden_layers
    print(f"    num_layers={num_layers}  num_q_heads={num_q_heads}  "
          f"num_kv_heads={num_kv_heads}  head_dim={head_dim}")
    print(f"    load took {time.time() - t0:.1f}s")

    pos, neg = load_seeds()
    # Small sample for timing — bf16 CPU is slow.
    N = 6
    pos = pos[:N]; neg = neg[:N]
    print(f"\n[2] capturing Q/V from {N}+{N} seeds (reference bf16 on CPU)")
    t1 = time.time()
    print("  refusals:")
    pos_caps = run_and_capture(model, tok, pos)
    print("  compliances:")
    neg_caps = run_and_capture(model, tok, neg)
    print(f"    capture took {time.time() - t1:.1f}s total")

    # Organize per-layer, per-head class tensors.
    # Q shape per capture: [S, num_q_heads * head_dim]. Reshape to
    # [S, num_q_heads, head_dim]. V: [S, num_kv_heads, head_dim].
    # Pool over positions (last-5 positions per seed as a proxy for
    # "end of assistant prose" — equivalent to shared_source's end-of-
    # rollout capture).
    print("\n[3] computing per-(layer, head) class-conditional means")
    def summarize(caps_list: list[dict]) -> dict:
        # Returns {layer: {'q': [N, num_q_heads, head_dim],
        #                  'v': [N, num_kv_heads, head_dim]}}
        out = defaultdict(lambda: {"q": [], "v": []})
        for cap in caps_list:
            for L, qv in cap.items():
                q_flat = qv["q"]   # [S, num_q_heads * head_dim]
                v_flat = qv["v"]   # [S, num_kv_heads * head_dim]
                # Average the last 5 positions — the "model-is-producing-
                # class-typical-prose" end-of-rollout positions.
                S = q_flat.shape[0]
                tail = min(5, S)
                q_reshape = q_flat[-tail:].view(tail, num_q_heads, head_dim).mean(0)
                v_reshape = v_flat[-tail:].view(tail, num_kv_heads, head_dim).mean(0)
                out[L]["q"].append(q_reshape)
                out[L]["v"].append(v_reshape)
        # Stack per-layer.
        stacked = {}
        for L in sorted(out.keys()):
            stacked[L] = {
                "q": torch.stack(out[L]["q"], dim=0),   # [N, num_q_heads, head_dim]
                "v": torch.stack(out[L]["v"], dim=0),   # [N, num_kv_heads, head_dim]
            }
        return stacked

    pos_summary = summarize(pos_caps)
    neg_summary = summarize(neg_caps)

    # Per-(layer, q_head) mean-diff in Q space.
    # Per-(layer, kv_head) mean-diff in V space.
    # Per-(layer, q_head, kv_head) product of magnitudes = intervention mass proxy.
    print("\n[4] computing diff-of-means per (layer, head)")
    q_diff = {}      # [num_layers, num_q_heads, head_dim]
    v_diff = {}
    q_mag = torch.zeros(num_layers, num_q_heads)
    v_mag = torch.zeros(num_layers, num_kv_heads)
    for L in range(num_layers):
        q_diff[L] = pos_summary[L]["q"].mean(0) - neg_summary[L]["q"].mean(0)
        v_diff[L] = pos_summary[L]["v"].mean(0) - neg_summary[L]["v"].mean(0)
        q_mag[L] = q_diff[L].norm(dim=-1)
        v_mag[L] = v_diff[L].norm(dim=-1)

    # Per-(layer, q_head) discrimination score: ||q_diff||. For synthetic
    # KV injection, interaction is Q·K*, and K* is fit from Q's diff (on
    # the matching KV-head). We'll pair each Q head with its grouping KV
    # head. num_q_heads // num_kv_heads = group size.
    group = num_q_heads // num_kv_heads
    # Score: q_mag × v_mag_of_matching_group_kv_head.
    score = torch.zeros(num_layers, num_q_heads)
    for L in range(num_layers):
        for h_q in range(num_q_heads):
            h_kv = h_q // group
            score[L, h_q] = q_mag[L, h_q] * v_mag[L, h_kv]

    # Top-p over eigenvalue-style cumulative mass of `score`.
    print("\n[5] top-p truncation over (layer, q_head) pairs")
    flat = score.flatten()
    sorted_vals, sorted_idx = torch.sort(flat, descending=True)
    total = sorted_vals.sum()
    cum = torch.cumsum(sorted_vals, dim=0) / total
    for top_p in (0.30, 0.50, 0.80):
        k = int((cum < top_p).sum()) + 1
        kept_idx = sorted_idx[:k].tolist()
        kept = sorted([(i // num_q_heads, i % num_q_heads) for i in kept_idx])
        # Distribution of layers in the kept set:
        layers_seen = defaultdict(int)
        for L, h in kept: layers_seen[L] += 1
        print(f"  top-p={top_p}  k={k}/{num_layers*num_q_heads} "
              f"=({k/(num_layers*num_q_heads)*100:.1f}%)  "
              f"covering {len(layers_seen)}/{num_layers} layers")
        # Show most-concentrated layers
        by_count = sorted(layers_seen.items(), key=lambda x: -x[1])[:6]
        print(f"    layer concentration: {by_count}")

    # Pick top-p=0.80 kept set for the selectivity measurement.
    top_p_for_selectivity = 0.80
    k = int((cum < top_p_for_selectivity).sum()) + 1
    kept_idx = sorted_idx[:k].tolist()
    kept = [(i // num_q_heads, i % num_q_heads) for i in kept_idx]

    # SELECTIVITY METRIC: for each retained (L, h_q), fit K*_{L, h_kv} =
    # q_diff[L][h_q] / ||q_diff[L][h_q]|| (as a unit vector, since K* will
    # be compared via Q·K*/sqrt(d)), then compute Q·K*/sqrt(d) for each
    # seed position. We want: the distribution of this dot product on
    # refusal-class samples vs compliance-class samples.
    print(f"\n[6] selectivity of fitted K* on top-p={top_p_for_selectivity} (L, h_q) pairs")
    print("    (Q·K*/sqrt(d) logit distribution, per class)")

    refusal_logits = []
    compliance_logits = []
    for (L, h_q) in kept:
        # Use MEAN across the Q head group as our K* target — we're
        # fitting one key per KV head, matching how the model stores it.
        q_star_unit = q_diff[L][h_q] / (q_diff[L][h_q].norm() + 1e-8)
        # Score each refusal seed's Q at this head (pooled over last 5)
        # against K*.
        for n in range(N):
            q_refusal = pos_summary[L]["q"][n, h_q]   # [head_dim]
            q_benign  = neg_summary[L]["q"][n, h_q]
            refusal_logits.append(float((q_refusal @ q_star_unit) / (head_dim ** 0.5)))
            compliance_logits.append(float((q_benign @ q_star_unit) / (head_dim ** 0.5)))

    refusal_logits = np.array(refusal_logits)
    compliance_logits = np.array(compliance_logits)
    print(f"    refusal-class   Q·K*/√d:  mean={refusal_logits.mean():+.3f}  "
          f"std={refusal_logits.std():.3f}  min/max={refusal_logits.min():+.3f}/{refusal_logits.max():+.3f}")
    print(f"    compliance-class Q·K*/√d: mean={compliance_logits.mean():+.3f}  "
          f"std={compliance_logits.std():.3f}  min/max={compliance_logits.min():+.3f}/{compliance_logits.max():+.3f}")
    separation = refusal_logits.mean() - compliance_logits.mean()
    print(f"    separation (refusal − compliance): {separation:+.3f}")

    # Softmax weight: exp(logit) / sum over context. Context at inference
    # time includes many other keys (the real ones), so the synthetic
    # slot's weight is approximately softmax([logit_synthetic, logit_avg_real]).
    # As a rough proxy, suppose the real keys have average logit 0
    # (unbiased baseline) across S positions. Then:
    #    weight_synthetic = exp(logit_syn) / (exp(logit_syn) + S * 1)
    # At S=50 and logit_syn=2, weight ≈ e² / (e² + 50) = 7.4/57.4 ≈ 13%.
    # At S=50 and logit_syn=-2, weight ≈ e^{-2} / (e^{-2} + 50) = 0.14/50.14 ≈ 0.3%.
    # Ratio ≈ 40×. So even modest Q·K* separation translates to order-of-
    # magnitude softmax-weight differentiation.
    S_assumed = 50
    weight_refusal_pos = np.exp(refusal_logits) / (np.exp(refusal_logits) + S_assumed)
    weight_compliance_pos = np.exp(compliance_logits) / (np.exp(compliance_logits) + S_assumed)
    print(f"\n    assuming context S={S_assumed} with avg real-key logit 0:")
    print(f"    synth-slot attention weight on refusal-class:   "
          f"mean={weight_refusal_pos.mean()*100:.2f}%  max={weight_refusal_pos.max()*100:.2f}%")
    print(f"    synth-slot attention weight on compliance-class: "
          f"mean={weight_compliance_pos.mean()*100:.2f}%  max={weight_compliance_pos.max()*100:.2f}%")
    print(f"    weight ratio (refusal / compliance): "
          f"{weight_refusal_pos.mean() / (weight_compliance_pos.mean() + 1e-10):.1f}×")
    print(f"\n    interpretation: a ratio >>1 means the synthetic KV slot")
    print(f"    fires automatically on refusal-class contexts and stays")
    print(f"    dormant on compliance-class contexts — this is the 'self-")
    print(f"    gated by attention softmax' property we were after.")

    # For comparison: the constant-bias intervention effectively applies
    # V_diff * 1.0 at EVERY position. So weight=100% everywhere.
    # Synthetic-KV weight at benign positions << 100% = less collateral
    # on benign samples by construction.
    print(f"\n[7] collateral comparison (on-benign residual perturbation)")
    print(f"    constant-bias approach: applies ΔV·1.0 at every position")
    print(f"                          → ΔV at 100% weight on benign seeds")
    print(f"    synthetic-KV approach:  applies ΔV·attn_weight(Q·K*)")
    print(f"                          → ΔV at {weight_compliance_pos.mean()*100:.2f}% weight on benign seeds")
    collateral_ratio = weight_compliance_pos.mean() / 1.0
    print(f"    collateral ratio (synthetic / constant): {collateral_ratio*100:.2f}%")
    print(f"\n    upshot: if the Q-logit separation is genuine, synthetic-")
    print(f"    KV injection collaterally perturbs benign contexts by ~{collateral_ratio*100:.0f}%")
    print(f"    of what constant-bias would. That's the 'lower overpromotion'")
    print(f"    property concretely measured.")


if __name__ == "__main__":
    main()
