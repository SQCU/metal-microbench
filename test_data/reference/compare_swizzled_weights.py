#!/usr/bin/env python
"""
Byte-level comparison of Swift's swizzled per-expert MoE weights vs the
Python-simulated swizzle of the same GGUF tensor.

Swift's `loadMoESwizzled` (main.swift) reshuffles the bytes of
`ffn_gate_up_exps.weight` (Q4_K) and `ffn_down_exps.weight` (Q5_1) into a
[nSuper=Dout/32][nbc=Din/blkElems][32 cols][blkBytes] layout. This script:

1. Loads the raw per-expert bytes from the GGUF.
2. Simulates Swift's swizzle in numpy.
3. Loads Swift's actual dumped swizzled bytes.
4. Compares them byte-for-byte.

If they match — Swift's swizzle is correct and any MoE-compute divergence is
downstream (kernel indexing, combine step, etc). If they don't — the swizzle
writes bytes to the wrong destination positions.

Also: inverse-swizzles Swift's bytes back to per-row layout, dequantizes,
and compares to HF's bf16 weights directly — catches any bit corruption.

Run:
    .venv/bin/python compare_swizzled_weights.py --expert 52
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize


def simulate_swizzle(unswizzled: np.ndarray, Dout: int, nbc: int, blkBytes: int) -> np.ndarray:
    """Replicate Swift's loadMoESwizzled for a single expert.
    Input `unswizzled` shape: Dout * nbc * blkBytes uint8 (per-expert byte blob).
    Output: same total size, reordered into [nSuper, nbc, 32, blkBytes]."""
    assert unswizzled.size == Dout * nbc * blkBytes
    assert Dout % 32 == 0, f"Dout={Dout} not divisible by 32"
    nSuper = Dout // 32
    out = np.empty_like(unswizzled)
    # Source: per-row then per-block:  src[(row, kb)] at row*nbc*blkBytes + kb*blkBytes
    # Dest:  [ns, kb, col, byte]    at ns*(nbc*32*blkBytes) + kb*(32*blkBytes) + col*blkBytes
    for ns in range(nSuper):
        for kb in range(nbc):
            for col in range(32):
                row = ns * 32 + col
                src = row * nbc * blkBytes + kb * blkBytes
                dst = ns * (nbc * 32 * blkBytes) + kb * (32 * blkBytes) + col * blkBytes
                out[dst:dst + blkBytes] = unswizzled[src:src + blkBytes]
    return out


def inverse_swizzle(swiz: np.ndarray, Dout: int, nbc: int, blkBytes: int) -> np.ndarray:
    """Reverse of simulate_swizzle: swizzled → per-row layout."""
    assert swiz.size == Dout * nbc * blkBytes
    assert Dout % 32 == 0
    out = np.empty_like(swiz)
    for row in range(Dout):
        ns = row // 32
        col = row % 32
        for kb in range(nbc):
            src = ns * (nbc * 32 * blkBytes) + kb * (32 * blkBytes) + col * blkBytes
            dst = row * nbc * blkBytes + kb * blkBytes
            out[dst:dst + blkBytes] = swiz[src:src + blkBytes]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", type=int, default=52)
    ap.add_argument("--gguf", default="/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
    ap.add_argument("--dir", default=str(Path(__file__).parent))
    args = ap.parse_args()

    d = Path(args.dir)
    r = GGUFReader(args.gguf)

    # Map tensor names to (path component, dtype, Din, Dout, blkElems, blkBytes).
    specs = {
        "gate_up": ("blk.0.ffn_gate_up_exps.weight",
                     GGMLQuantizationType.Q4_K, 2816, 1408, 256, 144),
        "down"   : ("blk.0.ffn_down_exps.weight",
                     GGMLQuantizationType.Q5_1, 704, 2816, 32, 24),
    }

    for label, (name, qtype, Din, Dout, blkElems, blkBytes) in specs.items():
        print(f"\n=== {label} ({name}, {qtype.name}) ===")
        nbc = Din // blkElems
        perExpertBytes = Dout * nbc * blkBytes
        print(f"  Din={Din}, Dout={Dout}, nbc={nbc}, blkBytes={blkBytes}, perExpertBytes={perExpertBytes}")

        # Pull raw bytes from GGUF and slice per-expert.
        tensor = None
        for t in r.tensors:
            if t.name == name:
                tensor = t; break
        assert tensor is not None, f"missing tensor {name}"
        raw = np.asarray(tensor.data)     # shape (E, Dout, nbc*blkBytes) for these tensors
        assert raw.size == 128 * perExpertBytes, \
            f"unexpected raw size {raw.size} vs expected {128 * perExpertBytes}"
        raw_flat = raw.reshape(-1)
        expert_raw = raw_flat[args.expert * perExpertBytes:(args.expert + 1) * perExpertBytes].copy()
        print(f"  GGUF expert[{args.expert}] raw bytes: md5(first 64)={expert_raw[:64].tobytes().hex()[:32]}...")

        # Simulate Swift's swizzle in numpy.
        py_swizzled = simulate_swizzle(expert_raw, Dout, nbc, blkBytes)
        print(f"  python-simulated swizzle: first 64 bytes = {py_swizzled[:64].tobytes().hex()[:32]}...")

        # Load Swift's dump.
        swift_path = d / f"lm_swift_l0_expert{args.expert}_{label}_swizzled.bin"
        assert swift_path.exists(), f"missing {swift_path}"
        swift_bytes = np.fromfile(swift_path, dtype=np.uint8)
        print(f"  swift dumped swizzle:    first 64 bytes = {swift_bytes[:64].tobytes().hex()[:32]}...")
        print(f"  swift file size: {swift_bytes.size}  expected: {perExpertBytes}")

        # Byte comparison.
        if swift_bytes.size != py_swizzled.size:
            print(f"  ❌ SIZE MISMATCH: swift {swift_bytes.size} vs python {py_swizzled.size}")
            continue
        byte_diff = (swift_bytes != py_swizzled).sum()
        if byte_diff == 0:
            print(f"  ✅ bytes match exactly ({perExpertBytes} bytes)")
        else:
            first_diff = int(np.where(swift_bytes != py_swizzled)[0][0])
            print(f"  ❌ BYTES DIFFER: {byte_diff} / {perExpertBytes} bytes mismatch; first at offset {first_diff}")

        # Inverse-swizzle Swift's dump + dequantize + compare to GGUF-dequant reference.
        inv = inverse_swizzle(swift_bytes, Dout, nbc, blkBytes)
        # gguf.quants.dequantize wants bytes reshaped as (num_rows, rowBytes).
        # Per expert: Dout rows × nbc*blkBytes per row.
        inv_rows = inv.reshape(Dout, nbc * blkBytes)
        deq = dequantize(inv_rows, qtype)        # shape (Dout, Din) fp32
        print(f"  inverse-swizzle + dequant shape: {deq.shape}  min={deq.min():.4f}  max={deq.max():.4f}")

        # Compare to GGUF-dequant reference (already saved by validate_moe_weights.py as ggufdq).
        ref_ggufdq_path = d / f"lm_hello_l0_expert{args.expert}_{label}_ggufdq.npy"
        if ref_ggufdq_path.exists():
            ref = np.load(ref_ggufdq_path)
            if ref.shape != deq.shape:
                # Maybe [1408, 2816] vs [2816, 1408] — handle transposes.
                if ref.T.shape == deq.shape:
                    ref = ref.T
                else:
                    print(f"  ⚠️  shape mismatch: ref {ref.shape} vs deq {deq.shape} — skip diff")
                    continue
            diff = deq.astype(np.float32) - ref.astype(np.float32)
            print(f"  vs GGUF-dequant ref: max|d|={np.abs(diff).max():.4f}  mean|d|={np.abs(diff).mean():.4e}  "
                  f"cos-sim={float(deq.flatten() @ ref.flatten()) / (np.linalg.norm(deq) * np.linalg.norm(ref)):.6f}")


if __name__ == "__main__":
    main()
