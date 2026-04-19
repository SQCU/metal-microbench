#!/usr/bin/env python
"""
Predict Swift's moe_down_out at layer 0 for a chosen compacted slot (one
expert-token output before the combine step) using HF weights, then diff
against the Swift dump.

This isolates the expert-compute kernels (Q4_K gate_up → gelu*up → Q5_1 down)
from the combine step: a mismatch here means one of those kernels is wrong.

Default: slot 4 = batch 0, expert 52 (top-1 expert at position 0 of
"Hello, my name is").

Run:
    .venv/bin/python compare_moe_slot.py --slot 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = 0.7978845608
    inner = c * (x + 0.044715 * x ** 3)
    inner = np.clip(inner, -20.0, 20.0)
    return 0.5 * x * (1.0 + np.tanh(inner))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=4,
                    help="Compacted MoE slot index to predict/diff.")
    ap.add_argument("--position", type=int, default=0)
    ap.add_argument("--dir", default=str(Path(__file__).parent))
    ap.add_argument("--hf", default="/Users/mdot/models/gemma-4-a4b-bf16")
    args = ap.parse_args()

    d = Path(args.dir)
    # Load Swift dumps.
    moe_down = np.load(d / "lm_swift_l0_moe_down_out.npy")      # [S, TOTAL_SLOTS, HIDDEN]
    gate_up_fused = np.load(d / "lm_swift_l0_gate_up_fused.npy") # [S, TOTAL_SLOTS, 2*MOE_INT]
    gate_proj_sw = np.load(d / "lm_swift_l0_gate_proj.npy")     # [S, TOTAL_SLOTS, MOE_INT]
    slot_tok = np.load(d / "lm_swift_l0_slot_token.npy")         # [S, TOTAL_SLOTS] → batch idx per slot
    group_s  = np.load(d / "lm_swift_l0_group_start.npy")        # [S, E+1]
    hidden_n = np.load(d / "lm_swift_l0_hidden_norm.npy")        # [S, HIDDEN] (router pre-norm input, actually same as pre_ffw_norm_2 here)

    # HF model probes:
    pre_ffw_2_out = np.load(d / "lm_hello_l0_probes.npy")[3]     # [S, HIDDEN] — pre_ffw_2 norm output
    gate_w = np.load(d / "lm_hello_l0_gate_w.npy")               # [S, K]
    exp_ids = np.load(d / "lm_hello_l0_expert_ids.npy")          # [S, K]

    p = args.position
    s = args.slot
    batch = int(slot_tok[p, s])
    # Find which expert this slot belongs to via group_start.
    expert_for_slot = None
    for e in range(128):
        if group_s[p, e] <= s < group_s[p, e + 1]:
            expert_for_slot = e
            break
    assert expert_for_slot is not None, f"slot {s} not in any group"
    print(f"position={p}  slot={s}  →  batch={batch}  expert={expert_for_slot}")
    print(f"  (batch {batch}'s top-k ids at pos {p}: {exp_ids[p].tolist()})")

    print(f"\n  loading HF bf16...")
    m = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    experts = m.model.language_model.layers[0].experts
    gu_w = experts.gate_up_proj[expert_for_slot].detach().float().numpy()   # [2*moe_int, hidden]
    down_w = experts.down_proj[expert_for_slot].detach().float().numpy()    # [hidden, moe_int]
    print(f"  HF gate_up_proj[{expert_for_slot}] shape: {gu_w.shape}")
    print(f"  HF down_proj[{expert_for_slot}]    shape: {down_w.shape}")

    # All batches fed same hidden at pos 0 → pre_ffw_2_out[p] is input for every batch.
    inp = pre_ffw_2_out[p].astype(np.float32)    # [HIDDEN]

    # ---------- Stage 1: gate_up_fused (post-Q4_K) ----------
    gate_up_pred = inp @ gu_w.T                    # [2*moe_int]
    half = gate_up_pred.shape[0] // 2
    gate_pred = gate_up_pred[:half]
    up_pred = gate_up_pred[half:]

    gate_up_act = gate_up_fused[p, s]              # [2*moe_int]
    gate_act = gate_up_act[:half]
    up_act = gate_up_act[half:]
    cos_gu = float(gate_up_pred @ gate_up_act) / (np.linalg.norm(gate_up_pred) * np.linalg.norm(gate_up_act) + 1e-12)
    print(f"\n  [Stage 1] gate_up_fused (post-Q4_K gate_up proj):")
    print(f"    predicted norm={np.linalg.norm(gate_up_pred):.3f}  min={gate_up_pred.min():.3f}  max={gate_up_pred.max():.3f}")
    print(f"    swift     norm={np.linalg.norm(gate_up_act):.3f}  min={gate_up_act.min():.3f}  max={gate_up_act.max():.3f}")
    print(f"    cos={cos_gu:.6f}  max|d|={np.abs(gate_up_pred - gate_up_act).max():.4f}")
    print(f"    gate   cos: {float(gate_pred @ gate_act) / (np.linalg.norm(gate_pred) * np.linalg.norm(gate_act) + 1e-12):.6f}")
    print(f"    up     cos: {float(up_pred @ up_act) / (np.linalg.norm(up_pred) * np.linalg.norm(up_act) + 1e-12):.6f}")

    # ---------- Stage 2: gate_proj (post-gelu*up) ----------
    inner_pred = gelu_tanh(gate_pred) * up_pred    # [moe_int]
    inner_act = gate_proj_sw[p, s]
    cos_gp = float(inner_pred @ inner_act) / (np.linalg.norm(inner_pred) * np.linalg.norm(inner_act) + 1e-12)
    print(f"\n  [Stage 2] gate_proj (post-gelu_tanh*up):")
    print(f"    predicted norm={np.linalg.norm(inner_pred):.3f}  min={inner_pred.min():.3f}  max={inner_pred.max():.3f}")
    print(f"    swift     norm={np.linalg.norm(inner_act):.3f}  min={inner_act.min():.3f}  max={inner_act.max():.3f}")
    print(f"    cos={cos_gp:.6f}")

    # ---------- Stage 3: moe_down_out (post-Q5_1 down) ----------
    predicted = inner_pred @ down_w.T
    actual = moe_down[p, s]
    diff = predicted - actual
    cos = float(predicted @ actual) / (np.linalg.norm(predicted) * np.linalg.norm(actual) + 1e-12)
    print(f"\n  [Stage 3] moe_down_out (post-Q5_1 down proj):")
    print(f"    predicted norm={np.linalg.norm(predicted):.3f}  min={predicted.min():.3f}  max={predicted.max():.3f}")
    print(f"    swift     norm={np.linalg.norm(actual):.3f}  min={actual.min():.3f}  max={actual.max():.3f}")
    print(f"    cos={cos:.6f}  max|d|={np.abs(diff).max():.4f}")


if __name__ == "__main__":
    main()
