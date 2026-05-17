"""Self-test for torch_dequant.py.

For each supported format:
  1. Generate a small random fp32 tensor
  2. Quantize via gguf-py reference (NumPy, slow but spec-correct)
  3. Dequant via gguf-py reference → reference output
  4. Dequant via our torch kernel (CPU/CUDA/MPS), pulled back to CPU
  5. Compare element-wise; expect bit-exact equivalence on per-element
     reconstructed values (modulo fp32→fp16 rounding at the cast boundary)

Usage:
  python torch_dequant_selftest.py
  TORCH_DEQUANT_DEVICE=cuda python torch_dequant_selftest.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Local imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_dequant import dequant, DEQUANTERS

# Pull gguf-py for reference quant + dequant.
LLAMA_CPP = Path("/Users/mdot/llama.cpp")
sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
import gguf.quants as gq                                            # type: ignore
from gguf.constants import GGMLQuantizationType                      # type: ignore


GGUF_TYPE = {
    "Q8_0": GGMLQuantizationType.Q8_0,
    "Q4_0": GGMLQuantizationType.Q4_0,
    "Q4_1": GGMLQuantizationType.Q4_1,
    "Q5_0": GGMLQuantizationType.Q5_0,
    "Q5_1": GGMLQuantizationType.Q5_1,
    "Q4_K": GGMLQuantizationType.Q4_K,
    "Q5_K": GGMLQuantizationType.Q5_K,
    "Q6_K": GGMLQuantizationType.Q6_K,
}


def run_one(fmt: str, n_blocks: int, device: str = "cpu") -> dict:
    """Quantize a random fp32 tensor, dequant via gguf-py and via torch_dequant,
    return comparison stats. For K-quants, gguf-py's quantize() isn't
    implemented — fall back to using bytes from a real GGUF tensor."""
    block_size = DEQUANTERS[fmt][2]
    n_elems = n_blocks * block_size

    # gguf-py implements quantize() only for non-K formats. For K-quants we
    # pull real packed bytes from the Q4_K_M GGUF on disk.
    if fmt in {"Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1"}:
        rng = np.random.default_rng(seed=42 + hash(fmt) % 1000)
        src = rng.standard_normal(n_elems).astype(np.float32) * 0.1
        quantized_bytes = gq.quantize(src, GGUF_TYPE[fmt])
    elif fmt in {"Q4_K", "Q5_K", "Q6_K"}:
        quantized_bytes = _real_kquant_bytes(fmt, n_blocks)
        if quantized_bytes is None:
            return {"fmt": fmt, "skipped": True,
                    "reason": f"no real {fmt} tensor found in any GGUF on disk"}
    else:
        raise ValueError(f"unhandled fmt {fmt}")

    ref_fp32 = gq.dequantize(quantized_bytes, GGUF_TYPE[fmt])
    ref_fp16 = ref_fp32.astype(np.float16).astype(np.float32)

    packed = torch.from_numpy(np.ascontiguousarray(quantized_bytes)).to(device)
    out = dequant(packed, fmt)
    out_fp32 = out.to("cpu").to(torch.float32).numpy()

    diff = np.abs(ref_fp16 - out_fp32)
    return {
        "fmt": fmt,
        "n_elems": ref_fp16.size,
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "exact_match": bool(np.array_equal(ref_fp16, out_fp32)),
        "shape_ok": ref_fp16.shape == out_fp32.shape,
    }


def _real_kquant_bytes(fmt: str, n_blocks: int) -> np.ndarray | None:
    """Pull `n_blocks * type_size` bytes from a real GGUF tensor of the
    given K-quant format. Looks at the Q4_K_M shipped GGUF first; for
    Q5_K/Q6_K we'd need a Q5_K_M / Q6_K GGUF (which 03_materialize_grid.sh
    produces). Returns None if no source available."""
    from gguf.gguf_reader import GGUFReader                              # type: ignore
    candidates = [
        Path("/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
        Path("/Users/mdot/models/gemma-4-a4b-quant-search/gemma-4-26B-A4B-it-Q4_K_M.gguf"),
        Path("/Users/mdot/models/gemma-4-a4b-quant-search/gemma-4-26B-A4B-it-Q5_K_M.gguf"),
        Path("/Users/mdot/models/gemma-4-a4b-quant-search/gemma-4-26B-A4B-it-Q6_K.gguf"),
    ]
    block_size, type_size = DEQUANTERS[fmt][2], DEQUANTERS[fmt][1]
    n_bytes = n_blocks * type_size
    for path in candidates:
        if not path.exists():
            continue
        reader = GGUFReader(path)
        for tensor in reader.tensors:
            if tensor.tensor_type.name == fmt:
                # tensor.data is a memory-mapped numpy array of the packed bytes,
                # already shaped as the GGUF spec expects (rows × type_size_bytes).
                # Slice off n_bytes worth from the first row.
                raw = bytes(tensor.data)[:n_bytes]
                if len(raw) == n_bytes:
                    return np.frombuffer(raw, dtype=np.uint8)
    return None


def main():
    device = os.environ.get("TORCH_DEQUANT_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available; falling back to CPU")
        device = "cpu"
    print(f"running self-test on device={device}")
    print(f"{'fmt':<6} {'n_elems':<10} {'max_err':<12} {'mean_err':<12} {'exact?':<8}")
    print("-" * 55)
    n_blocks = {"Q8_0": 32, "Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32,
                "Q4_K": 4, "Q5_K": 4, "Q6_K": 4}
    all_ok = True
    for fmt in DEQUANTERS:
        try:
            result = run_one(fmt, n_blocks[fmt], device=device)
            if result.get("skipped"):
                print(f"{fmt:<6} -          (skipped: {result['reason']})")
                continue
            mark = "✓" if result["exact_match"] else (
                "≈" if result["max_abs_err"] < 1e-3 else "✗")
            print(f"{fmt:<6} {result['n_elems']:<10} "
                  f"{result['max_abs_err']:<12.6e} {result['mean_abs_err']:<12.6e} "
                  f"{mark}")
            if not result["exact_match"] and result["max_abs_err"] > 1e-3:
                all_ok = False
        except Exception as e:
            print(f"{fmt:<6} ERROR: {type(e).__name__}: {e}")
            all_ok = False
    print()
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
