#!/usr/bin/env python3
"""One iteration of the gated-intervention optimization, timed.

Loads cached on-policy seeds from /tmp/on_policy_seeds.json (the same
12 refusals / 12 compliances we've been using for the 4-cell matrix),
runs them through the HuggingFace reference impl on MPS, captures
residuals + logits, sets up a tiny gated-intervention module over the
top-K layers, and does ONE forward+backward+optimizer step. Reports
wall clock breakdown.

The point: we want to know whether fitting gated interventions is a
seconds-per-iteration or minutes-per-iteration operation, so we can
decide whether to keep it in PyTorch or bite the bullet and write
Metal autograd.

Run:
    cd /Users/mdot/metal-microbench
    test_data/reference/.venv/bin/python notes/time_gate_iter.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_DIR = "/Users/mdot/models/gemma-4-a4b-bf16"
SEEDS_FILE = "/tmp/on_policy_seeds.json"
DEVICE = "cpu"   # Gemma-4 MoE hits histogram_mps which isn't implemented
NUM_LAYERS = 30
HIDDEN = 2816

# Layers to intervene at (matches the "top 5 layers by eigenvalue"
# pattern we've been using with the diff_of_means fit).
INTERVENE_LAYERS = [9, 10, 11, 12, 13]


def load_seeds() -> tuple[list[str], list[str]]:
    with open(SEEDS_FILE) as f:
        d = json.load(f)
    pos = [str(x) for x in d.get("positive", [])]
    neg = [str(x) for x in d.get("negative", [])]
    return pos, neg


def build_sequences(tok, seeds: list[str], role_for_first: str = "user") -> list[torch.Tensor]:
    """Wrap each seed as a chat-template prompt + seed-as-assistant-response,
    tokenize, return list of 1-D token tensors. The sequences are what we
    teacher-force the reference impl over when capturing residuals +
    logits. Closer to what shared_source does on the server side."""
    # Trivial prompt so the role structure matches the fit's anchor.
    stem = tok.apply_chat_template(
        [{"role": "user", "content": "Continue the response."},
         {"role": "assistant", "content": ""}],
        tokenize=False, add_generation_prompt=False,
    )
    out = []
    for s in seeds:
        full = stem.rstrip() + "\n" + s
        ids = tok.encode(full, return_tensors="pt", add_special_tokens=True).squeeze(0)
        out.append(ids)
    return out


def capture_residuals_and_logits(model, ids_list: list[torch.Tensor]):
    """For each sequence, run forward with output_hidden_states=True and
    return (residual_stack[len, S, layer, hidden], logits[len, S, vocab]).
    We keep bf16 on MPS; cast to fp32 for the optimization."""
    all_hidden = []
    all_logits = []
    for ids in ids_list:
        ids = ids.to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        # hidden_states is a tuple of (NUM_LAYERS+1) tensors, each [1, S, H].
        # Index 0 = post-embed, indices 1..NUM_LAYERS = after each decoder layer.
        hs = torch.stack(out.hidden_states, dim=0).squeeze(1)    # [NUM_LAYERS+1, S, H]
        all_hidden.append(hs.to(torch.float32).cpu())
        all_logits.append(out.logits.squeeze(0).to(torch.float32).cpu())
    return all_hidden, all_logits


class GatedIntervention(nn.Module):
    """Per-layer gated intervention: h_new = h + σ(⟨h,w⟩ − b) · d.
    Parameters are (w_L, b_L, d_L) per intervention layer.
    Directions are unit-normalized to put all magnitude into the gate."""

    def __init__(self, layers: list[int], hidden: int = HIDDEN):
        super().__init__()
        self.layers = layers
        n = len(layers)
        # Initialize as tiny random unit vectors; zero bias means
        # sigmoid fires at 0.5 naturally, which seeds broadly — the
        # fit will drive gates toward 0 on don't-intervene cases.
        self.w = nn.Parameter(torch.randn(n, hidden) * 0.01)
        self.b = nn.Parameter(torch.zeros(n))
        self.d = nn.Parameter(torch.randn(n, hidden) * 0.01)

    def apply_at(self, residual_at_layer: torch.Tensor, L_idx: int) -> torch.Tensor:
        """residual_at_layer: [S, H]. L_idx into self.layers."""
        w = self.w[L_idx]
        d_unit = self.d[L_idx] / (self.d[L_idx].norm() + 1e-8)
        gate = torch.sigmoid((residual_at_layer @ w) - self.b[L_idx])  # [S]
        return residual_at_layer + gate.unsqueeze(-1) * d_unit.unsqueeze(0)


def time_one_iteration():
    print("=" * 64)
    print("GATED INTERVENTION FIT — ONE-ITERATION TIMING")
    print("=" * 64)

    t_load_start = time.time()
    print(f"\n[1] loading tokenizer + bf16 model on {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"    loaded in {time.time() - t_load_start:.1f} s")

    print(f"\n[2] loading cached on-policy seeds from {SEEDS_FILE}")
    pos, neg = load_seeds()
    print(f"    {len(pos)} refusals + {len(neg)} compliances")

    # Keep it small for a per-iter timing: 4 seeds each side.
    N = 4
    pos_subset = pos[:N]
    neg_subset = neg[:N]

    t0 = time.time()
    print(f"\n[3] tokenizing {N}+{N} sequences")
    pos_ids = build_sequences(tok, pos_subset)
    neg_ids = build_sequences(tok, neg_subset)
    print(f"    done in {time.time() - t0:.2f} s; "
          f"seq lens: pos={[x.shape[0] for x in pos_ids]}, "
          f"neg={[x.shape[0] for x in neg_ids]}")

    t0 = time.time()
    print("\n[4] forward pass capture (residuals + logits, no grad)")
    pos_hs, pos_logits = capture_residuals_and_logits(model, pos_ids)
    t_pos = time.time() - t0
    t1 = time.time()
    neg_hs, neg_logits = capture_residuals_and_logits(model, neg_ids)
    t_neg = time.time() - t1
    total_tokens = sum(x.shape[0] for x in pos_ids + neg_ids)
    print(f"    pos forward: {t_pos:.2f} s  ({N} seqs)")
    print(f"    neg forward: {t_neg:.2f} s  ({N} seqs)")
    print(f"    total tokens: {total_tokens} · "
          f"{total_tokens / (t_pos + t_neg):.0f} tok/s forward")

    # Construct a minimal proxy loss: we want to measure wall-clock
    # cost of (forward+backward+step) on the gating module, NOT a real
    # behavioral fit. So we use a mock objective:
    #   L = gating_magnitude²  +  alignment_with_diff_of_means_direction
    # which exercises the gate/direction parameters end-to-end.
    print("\n[5] building gated-intervention module + Adam optimizer")
    gate = GatedIntervention(INTERVENE_LAYERS, hidden=HIDDEN).to(DEVICE).to(torch.float32)
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in gate.parameters())
    print(f"    {n_params} params across {len(INTERVENE_LAYERS)} layers")

    # Precompute diff-of-means direction per layer from the cached
    # residuals — this is the "closed-form warm start" target we'd
    # normally use to initialize d_L.
    # Capture the last-token residual at each intervention layer.
    def last_tok_per_layer(hs_list: list[torch.Tensor]) -> torch.Tensor:
        """hs_list: list of [NUM_LAYERS+1, S, H]. Returns [N, len(layers), H]
        = last-token residual at each intervention layer per sequence."""
        out = []
        for hs in hs_list:
            # hs[L+1] = after decoder_layer[L], shape [S, H]
            per_layer = torch.stack([hs[L + 1, -1] for L in INTERVENE_LAYERS], dim=0)
            out.append(per_layer)
        return torch.stack(out, dim=0)   # [N, len(layers), H]

    pos_lt = last_tok_per_layer(pos_hs).to(DEVICE)   # [N, L, H]
    neg_lt = last_tok_per_layer(neg_hs).to(DEVICE)   # [N, L, H]
    diff_dir = (pos_lt.mean(dim=0) - neg_lt.mean(dim=0))  # [L, H]
    diff_dir = diff_dir / (diff_dir.norm(dim=-1, keepdim=True) + 1e-8)

    print("\n[6] ONE iteration — forward + backward + step (cached activations)")
    t_iter_start = time.time()

    # Forward: compute per-sequence, per-intervention-layer gated
    # activation and its alignment cost against the data.
    total_loss = torch.zeros((), device=DEVICE, requires_grad=False)
    for N_idx, hs in enumerate(pos_hs):
        hs_dev = hs.to(DEVICE)
        # Intervene at each layer: push residual toward neg (compliance)
        # class — this is C-cell semantics. Cost = ‖intervention‖² + KL
        # against a mock logit target (in a real fit this'd be
        # p(y|natural compliance)).
        for i, L in enumerate(INTERVENE_LAYERS):
            h_L = hs_dev[L + 1]   # [S, H]
            h_new = gate.apply_at(h_L, i)
            delta = h_new - h_L
            # Intervention-magnitude cost (OT-style):
            mag_cost = delta.pow(2).sum(dim=-1).mean()
            # Alignment reward (toward diff_of_means direction, pushing
            # to neg side so negative sign):
            align = -((delta * diff_dir[i]).sum(dim=-1).mean())
            total_loss = total_loss + mag_cost + 0.1 * align

    # One-step backward + optimizer.
    t_fwd = time.time() - t_iter_start
    t_bwd_start = time.time()
    total_loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    # Wait for MPS to finish.
    if DEVICE == "mps":
        torch.mps.synchronize()
    t_bwd = time.time() - t_bwd_start

    t_iter = time.time() - t_iter_start
    print(f"    forward (cached activations, gate only):  {t_fwd*1000:.1f} ms")
    print(f"    backward + optimizer step:                {t_bwd*1000:.1f} ms")
    print(f"    ONE iteration total:                      {t_iter*1000:.1f} ms")
    print(f"    loss value: {float(total_loss):.4f}")

    print("\n[7] Caveats")
    print("    - This iteration uses CACHED activations; NO gradient through")
    print("      the model's forward pass. That matches the likely fit path:")
    print("      collect residuals once, optimize the gate over cached data.")
    print("    - A real fit that runs model.forward() WITH intervention each")
    print("      step (to get actual logit KL) would cost one forward pass")
    print(f"      per iteration = ~{(t_pos + t_neg) / (2*N) * 1000:.0f} ms per seq.")
    print("    - If the cached-activation approximation is adequate (it is")
    print("      for finding the linearized intervention), we're at a few")
    print("      ms per iteration, so 10⁴ Adam steps ≈ tens of seconds.")

    print("\n=== summary ===")
    print(f"  model load:            {t_load_start:.0f}s startup")
    print(f"  forward capture (8 seqs): {t_pos + t_neg:.1f} s")
    print(f"  per-iteration fit step: {t_iter*1000:.1f} ms  ({t_fwd*1000:.0f} fwd / {t_bwd*1000:.0f} bwd)")


if __name__ == "__main__":
    time_one_iteration()
