#!/usr/bin/env python
"""
Pure-Python validator for layer-0 MoE weights.

For a chosen expert (default 52 — the top-1 expert at position 0 for the
"Hello, my name is" prompt), dequantize its weights from the GGUF on disk
via gguf's built-in dequantizers, and compare to HF's bf16 weights.

If the two match (up to Q4_K/Q5_1 quantization noise, ~1%), the GGUF bytes
contain what we expect and any Swift-side divergence is in the kernel's
byte-access pattern (swizzle) or in the downstream compute. If they don't
match, the quantizer produced unexpected numbers or we're dequantizing the
wrong tensor slice.

Run:
    .venv/bin/python validate_moe_weights.py --expert 52
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from gguf import GGUFReader, quants


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", type=int, default=52, help="Expert index to validate.")
    ap.add_argument("--gguf", default="/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
    ap.add_argument("--hf", default="/Users/mdot/models/gemma-4-a4b-bf16")
    args = ap.parse_args()

    print(f"validating expert {args.expert} at layer 0")
    print(f"  GGUF: {args.gguf}")
    print(f"  HF  : {args.hf}")

    # --- GGUF side: dequantize the two MoE tensors for layer 0 ---
    r = GGUFReader(args.gguf)
    # Shapes per earlier probe:
    #   ffn_gate_up_exps.weight  shape=[2816, 1408, 128]  Q4_K
    #   ffn_down_exps.weight     shape=[704, 2816, 128]   Q5_1
    # Note GGUF axis order is reversed from HF, so byte-flat layout is:
    #   gate_up: [E=128, 2*MOE_INT=1408*2... wait check]
    gguf_tensors: dict[str, np.ndarray] = {}
    for t in r.tensors:
        if t.name in ("blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight"):
            print(f"  {t.name}:  GGUF shape {list(t.shape)}  dtype {t.tensor_type.name}")
            # t.data is raw quantized bytes viewed as uint8. We need a 2D array
            # of shape [num_rows, n_bytes_per_row] to pass to quants.dequantize,
            # then reshape by the per-expert block layout.
            raw = np.asarray(t.data)
            gguf_tensors[t.name] = (raw.copy(), list(t.shape), t.tensor_type)

    # --- HF side: load expert weights directly ---
    from transformers import AutoModelForCausalLM
    print(f"  loading HF model...")
    m = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    experts = m.model.language_model.layers[0].experts
    # experts.gate_up_proj   shape=[E=128, 2*moe_int=1408, hidden=2816]
    # experts.down_proj      shape=[E=128, hidden=2816, moe_int=704]
    print(f"  HF experts.gate_up_proj shape: {list(experts.gate_up_proj.shape)}")
    print(f"  HF experts.down_proj    shape: {list(experts.down_proj.shape)}")

    hf_gate_up = experts.gate_up_proj[args.expert].detach().float().numpy()  # [1408, 2816]
    hf_down    = experts.down_proj[args.expert].detach().float().numpy()      # [2816, 704]

    # --- Dequantize GGUF per-expert slice ---
    # Gate_up: 128 experts × 1408 output rows × 2816 input cols  (per expert).
    # GGUF reports shape=[2816, 1408, 128]; bytes are layed out with axis-0
    # (2816) fastest. Flat offset of (a=in, b=row, c=expert) = a + b*2816 + c*2816*1408.
    # Actually for Q4_K, axis 0 (2816 = D_in) is block-packed — we only see
    # bytes, not individual elements. Each Q4_K block is 256 elements in 144 bytes.
    # 2816 / 256 = 11 blocks per row. Row bytes = 11 * 144 = 1584.
    # Per expert: 1408 rows * 1584 = 2,230,272 bytes.
    # Total: 128 * 2,230,272 = 285,474,816 bytes.
    # Actually total is 507,510,784 (printed by gguf) = 253,755,392 / 1.98... hmm.
    # Let's just compute via quants.dequantize on the full raw blob and reshape.
    print(f"\n--- Q4_K gate_up dequant ---")
    gu_raw, gu_shape, gu_qtype = gguf_tensors["blk.0.ffn_gate_up_exps.weight"]
    print(f"  raw bytes: {gu_raw.shape}  dtype: {gu_raw.dtype}")
    gu_deq = quants.dequantize(gu_raw, gu_qtype)
    print(f"  dequantized shape: {gu_deq.shape}  dtype: {gu_deq.dtype}")
    # gu_deq should have total elements = 2816 * 1408 * 128 = 507510784
    # and reshape to [E=128, 1408, 2816]? GGUF logical shape is [2816, 1408, 128]
    # which means numpy view as (128, 1408, 2816) (axes reversed).
    if gu_deq.ndim == 1:
        gu_deq = gu_deq.reshape(128, 1408, 2816)   # HF-compatible layout
    elif gu_deq.ndim == 3:
        print(f"  (already 3D, shape {gu_deq.shape})")
    print(f"  reshape to HF-compat: {gu_deq.shape}")

    print(f"\n  expert {args.expert}: HF_bf16 vs GGUF_Q4K_dequant")
    g_deq_e = gu_deq[args.expert].astype(np.float32)    # [1408, 2816]
    print(f"    HF  [{args.expert}] shape: {hf_gate_up.shape}  min={hf_gate_up.min():.4f}  max={hf_gate_up.max():.4f}")
    print(f"    GGUF[{args.expert}] shape: {g_deq_e.shape}  min={g_deq_e.min():.4f}  max={g_deq_e.max():.4f}")
    diff = hf_gate_up - g_deq_e
    print(f"    max|diff|: {np.abs(diff).max():.4f}")
    print(f"    mean|diff|: {np.abs(diff).mean():.4f}")
    print(f"    rel-L2   : {np.linalg.norm(diff) / np.linalg.norm(hf_gate_up):.4f}")
    print(f"    cos-sim  : {float((hf_gate_up.flatten() @ g_deq_e.flatten()) / (np.linalg.norm(hf_gate_up) * np.linalg.norm(g_deq_e))):.6f}")

    # Save the GGUF-dequantized expert weights for Swift-side comparison later.
    np.save(f"lm_hello_l0_expert{args.expert}_gate_up_hf.npy", hf_gate_up)
    np.save(f"lm_hello_l0_expert{args.expert}_gate_up_ggufdq.npy", g_deq_e)

    # --- Q5_1 down ---
    print(f"\n--- Q5_1 down dequant ---")
    dw_raw, dw_shape, dw_qtype = gguf_tensors["blk.0.ffn_down_exps.weight"]
    print(f"  raw bytes: {dw_raw.shape}  dtype: {dw_raw.dtype}")
    dw_deq = quants.dequantize(dw_raw, dw_qtype)
    print(f"  dequantized shape: {dw_deq.shape}  dtype: {dw_deq.dtype}")
    if dw_deq.ndim == 1:
        dw_deq = dw_deq.reshape(128, 2816, 704)
    print(f"  reshape to HF-compat: {dw_deq.shape}")

    d_deq_e = dw_deq[args.expert].astype(np.float32)    # [2816, 704]
    print(f"\n  expert {args.expert}: HF_bf16 vs GGUF_Q5_1_dequant")
    print(f"    HF  [{args.expert}] shape: {hf_down.shape}  min={hf_down.min():.4f}  max={hf_down.max():.4f}")
    print(f"    GGUF[{args.expert}] shape: {d_deq_e.shape}  min={d_deq_e.min():.4f}  max={d_deq_e.max():.4f}")
    diff = hf_down - d_deq_e
    print(f"    max|diff|: {np.abs(diff).max():.4f}")
    print(f"    mean|diff|: {np.abs(diff).mean():.4f}")
    print(f"    rel-L2   : {np.linalg.norm(diff) / np.linalg.norm(hf_down):.4f}")
    print(f"    cos-sim  : {float((hf_down.flatten() @ d_deq_e.flatten()) / (np.linalg.norm(hf_down) * np.linalg.norm(d_deq_e))):.6f}")

    np.save(f"lm_hello_l0_expert{args.expert}_down_hf.npy", hf_down)
    np.save(f"lm_hello_l0_expert{args.expert}_down_ggufdq.npy", d_deq_e)


if __name__ == "__main__":
    main()
