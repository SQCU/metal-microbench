#!/usr/bin/env python
"""
Dequantize a single Q4_K super-block (256 elements) using the exact formula
from Swift's moe_gemv_q4k_v6 kernel (plus unpack_q4k_scales), and compare to
gguf.quants.dequantize output for the same bytes.

If these differ, our kernel's Q4_K dequant has a bug. If they agree, the bug
is in how we use the dequantized values (indexing into hidden, per-element
mapping to position, etc.).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize


def unpack_q4k_scales(s: np.ndarray, sb: int) -> tuple[int, int]:
    if sb < 4:
        sc = s[sb] & 0x3F
        mn = s[sb + 4] & 0x3F
    else:
        k = sb - 4
        sc = (s[k + 8] & 0x0F) | ((s[k + 0] & 0xC0) >> 2)
        mn = (s[k + 8] >> 4)    | ((s[k + 4] & 0xC0) >> 2)
    return int(sc), int(mn)


def kernel_dequant_block(blk: np.ndarray) -> np.ndarray:
    """Dequant ONE Q4_K super-block (144 bytes) → 256 fp32 values, using our kernel's exact math."""
    assert blk.size == 144
    d = float(np.frombuffer(blk[0:2].tobytes(), dtype=np.float16)[0])
    dmin = float(np.frombuffer(blk[2:4].tobytes(), dtype=np.float16)[0])
    scales = blk[4:16]
    qs = blk[16:144]
    out = np.zeros(256, dtype=np.float32)
    for sb in range(8):
        sc, mn = unpack_q4k_scales(scales, sb)
        dl = d * sc
        ml = dmin * mn
        base = sb * 32
        for p in range(16):
            byte = int(qs[sb * 16 + p])
            out[base + p]      = dl * (byte & 0xF) - ml
            out[base + p + 16] = dl * ((byte >> 4) & 0xF) - ml
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", type=int, default=52)
    ap.add_argument("--dir", default=str(Path(__file__).parent))
    args = ap.parse_args()

    D_in = 2816
    D_out = 1408
    BLK = 144
    nbc = D_in // 256

    # Load per-row (un-swizzled) bytes for expert 52. We need to inverse-swizzle
    # because gguf.quants.dequantize expects per-row layout.
    from compare_swizzled_weights import inverse_swizzle
    swiz = np.fromfile(Path(args.dir) / f"lm_swift_l0_expert{args.expert}_gate_up_swizzled.bin",
                        dtype=np.uint8)
    unswiz = inverse_swizzle(swiz, Dout=D_out, nbc=nbc, blkBytes=BLK)
    # unswiz shape: D_out * nbc * BLK = 1408 * 11 * 144 bytes
    unswiz_rows = unswiz.reshape(D_out, nbc * BLK)

    # Pick row 0, block kb=0 → first 144 bytes.
    blk = unswiz_rows[0, 0:BLK]
    print(f"block md5-ish (first 16 bytes hex): {blk[:16].tobytes().hex()}")

    # Kernel-formula dequant.
    ours = kernel_dequant_block(blk)
    print(f"ours:  min={ours.min():.6f}  max={ours.max():.6f}  first 8: {ours[:8]}")

    # gguf.quants.dequantize of just this block.
    ref = dequantize(blk.reshape(1, BLK), GGMLQuantizationType.Q4_K).flatten()
    print(f"gguf:  min={ref.min():.6f}  max={ref.max():.6f}  first 8: {ref[:8]}")

    diff = ours - ref
    print(f"max|diff|: {np.abs(diff).max():.6f}")
    print(f"mean|diff|: {np.abs(diff).mean():.6f}")
    print(f"cos-sim: {float(ours @ ref) / (np.linalg.norm(ours) * np.linalg.norm(ref) + 1e-12):.6f}")

    # Look for systematic permutations — if ours is a permutation of ref,
    # cos-sim on sorted values would be 1.
    ours_sorted = np.sort(ours)
    ref_sorted = np.sort(ref)
    print(f"sorted cos-sim: {float(ours_sorted @ ref_sorted) / (np.linalg.norm(ours_sorted) * np.linalg.norm(ref_sorted) + 1e-12):.6f}")


if __name__ == "__main__":
    main()
