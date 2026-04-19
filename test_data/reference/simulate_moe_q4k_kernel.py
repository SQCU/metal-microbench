#!/usr/bin/env python
"""
Replicate Swift's moe_gemv_q4k_v6 kernel's math in numpy for a single expert,
using the swizzled bytes Swift actually reads. Compare to HF's bf16 result.

If this simulation matches HF, the kernel LOGIC is correct and any GPU-side
divergence is a Metal-specific issue (fp16 accumulator, alignment, ...).
If it doesn't match, our kernel has a bug in its formula or indexing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def unpack_q4k_scales(s: np.ndarray, sb: int) -> tuple[int, int]:
    if sb < 4:
        sc = s[sb] & 0x3F
        mn = s[sb + 4] & 0x3F
    else:
        k = sb - 4
        sc = (s[k + 8] & 0x0F) | ((s[k + 0] & 0xC0) >> 2)
        mn = (s[k + 8] >> 4)    | ((s[k + 4] & 0xC0) >> 2)
    return int(sc), int(mn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", type=int, default=52)
    ap.add_argument("--dir", default=str(Path(__file__).parent))
    ap.add_argument("--hf", default="/Users/mdot/models/gemma-4-a4b-bf16")
    args = ap.parse_args()

    D_in = 2816
    D_out = 1408
    BLK = 144
    blkElems = 256
    nbc = D_in // blkElems        # 11
    super_bytes = nbc * 32 * BLK
    expert_bytes = (D_out // 32) * super_bytes

    swiz_path = Path(args.dir) / f"lm_swift_l0_expert{args.expert}_gate_up_swizzled.bin"
    swiz = np.fromfile(swiz_path, dtype=np.uint8)
    print(f"loaded swizzled bytes: {swiz.size} (expected per-expert {expert_bytes})")
    assert swiz.size == expert_bytes

    # The MoE experts receive `hidden_norm` AFTER pre_feedforward_layernorm_2,
    # which is distinct from the router's pre-norm (both live in the same Swift
    # buffer named `hidden_norm` but the MoE call-site's value overwrites it).
    # Probe 3 in lm_hello_l0_probes.npy is pre_ffw_2_out (HF reference).
    hidden_norm = np.load(Path(args.dir) / "lm_hello_l0_probes.npy")[3, 0]   # HF [S=0, HIDDEN]
    print(f"pre_ffw_2_out (HF) shape: {hidden_norm.shape}  min={hidden_norm.min():.4f}  max={hidden_norm.max():.4f}")

    # Simulate the FIXED kernel for every output n in [0, D_out).
    print(f"simulating moe_gemv_q4k_v6 (fixed) for expert {args.expert} (all {D_out} outputs)...")
    sim_out = np.zeros(D_out, dtype=np.float32)
    for n in range(D_out):
        n_block = n // 32
        t = n % 32
        W_super_base = n_block * super_bytes
        acc = 0.0
        for kb in range(nbc):
            blk_off = W_super_base + kb * 32 * BLK + t * BLK
            blk = swiz[blk_off:blk_off + BLK]
            d = float(np.frombuffer(blk[0:2].tobytes(), dtype=np.float16)[0])
            dmin = float(np.frombuffer(blk[2:4].tobytes(), dtype=np.float16)[0])
            scales = blk[4:16]
            qs = blk[16:144]
            for pair in range(4):
                sb_lo = pair * 2
                sb_hi = pair * 2 + 1
                sc_lo, mn_lo = unpack_q4k_scales(scales, sb_lo)
                sc_hi, mn_hi = unpack_q4k_scales(scales, sb_hi)
                dl_lo = d * sc_lo; ml_lo = dmin * mn_lo
                dl_hi = d * sc_hi; ml_hi = dmin * mn_hi
                base_lo = kb * 256 + sb_lo * 32
                base_hi = kb * 256 + sb_hi * 32
                for p in range(32):
                    byte = int(qs[pair * 32 + p])
                    w_lo = dl_lo * (byte & 0xF) - ml_lo
                    w_hi = dl_hi * ((byte >> 4) & 0xF) - ml_hi
                    acc += float(hidden_norm[base_lo + p]) * w_lo \
                         + float(hidden_norm[base_hi + p]) * w_hi
        sim_out[n] = acc
    print(f"sim out (FIXED): min={sim_out.min():.4f}  max={sim_out.max():.4f}  norm={np.linalg.norm(sim_out):.3f}")

    # HF reference.
    print(f"loading HF...")
    m = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    gu_w = m.model.language_model.layers[0].experts.gate_up_proj[args.expert].detach().float().numpy()
    hf_out = hidden_norm.astype(np.float32) @ gu_w.T    # [1408]
    print(f"HF out: min={hf_out.min():.4f}  max={hf_out.max():.4f}  norm={np.linalg.norm(hf_out):.3f}")

    cos = float(sim_out @ hf_out) / (np.linalg.norm(sim_out) * np.linalg.norm(hf_out) + 1e-12)
    diff = sim_out - hf_out
    print(f"sim vs HF: cos={cos:.6f}  max|d|={np.abs(diff).max():.4f}  rel-L2={np.linalg.norm(diff)/np.linalg.norm(hf_out):.4f}")
    print(f"sim[:8]: {sim_out[:8]}")
    print(f"hf [:8]: {hf_out[:8]}")

    # Compare to Swift's actual output at slot 4.
    swift_act = np.load(Path(args.dir) / "lm_swift_l0_gate_up_fused.npy")[0, 4]
    cos_sw = float(sim_out @ swift_act) / (np.linalg.norm(sim_out) * np.linalg.norm(swift_act) + 1e-12)
    diff_sw = sim_out - swift_act
    print(f"\nsim vs Swift-actual: cos={cos_sw:.6f}  max|d|={np.abs(diff_sw).max():.4f}")
    print(f"swift[:8]: {swift_act[:8]}")


if __name__ == "__main__":
    main()
