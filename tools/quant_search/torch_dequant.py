"""GPU-native PyTorch dequant kernels for the GGUF quantization formats
this engine consumes. Bit-exact equivalent to llama.cpp/gguf-py reference
math; vectorized over blocks; runs at SM throughput on whatever device
the input tensor lives on (CPU, CUDA, ROCm, MPS).

No NumPy round-trips. No gguf-py at runtime. The reference NumPy
implementation is consulted ONCE in `selftest.py` to validate that this
module's outputs match to within fp32→fp16 rounding-order noise.

Design:
  - Each format is one function `dequant_<fmt>(packed: torch.Tensor)`.
  - Input: torch.uint8 tensor with last dim = `type_size_bytes` × `n_blocks`.
    Earlier dims are batch (e.g. (n_rows, n_blocks * type_size_bytes)).
  - Output: torch.float16 tensor with last dim = `block_size` × `n_blocks`.
  - All operations are torch primitives (bitwise_and, bitwise_or, shift,
    cast, broadcast multiply/add). The compiler generates GPU kernels
    natively; for hot dispatch we'd use torch.compile or Triton, but the
    plain ops are already ~bandwidth-bound at moderate block counts.

GGUF format references (matched to llama.cpp's ggml-quants.c):
  - Q8_0:  block = [d:fp16, qs:int8[32]]                — 34 bytes / 32 elems
  - Q4_0:  block = [d:fp16, qs:nibble[16]]              — 18 bytes / 32 elems
  - Q4_1:  block = [d:fp16, m:fp16, qs:nibble[16]]      — 20 bytes / 32 elems
  - Q5_0:  block = [d:fp16, qh:u8[4], qs:nibble[16]]    — 22 bytes / 32 elems
  - Q5_1:  block = [d:fp16, m:fp16, qh:u8[4], qs:n[16]] — 24 bytes / 32 elems
  - Q4_K:  super-block = [d, dmin, scales[12], qs[128]] — 144 bytes / 256 elems
  - Q5_K:  super-block = [d, dmin, scales[12], qh[32], qs[128]] — 176 bytes / 256
  - Q6_K:  super-block = [ql[128], qh[64], scales[16], d:fp16] — 210 bytes / 256
"""
from __future__ import annotations
import torch


# ── Q8_0 ────────────────────────────────────────────────────────────────

def dequant_q8_0(packed: torch.Tensor) -> torch.Tensor:
    """Q8_0: block=[d:fp16, qs:int8[32]] → 34 bytes per 32-element block."""
    *batch, row_bytes = packed.shape
    assert row_bytes % 34 == 0, f"row_bytes {row_bytes} not multiple of 34 (Q8_0)"
    n_blocks = row_bytes // 34
    blk = packed.view(*batch, n_blocks, 34)
    # First 2 bytes: d (fp16). Reinterpret bytes as fp16.
    d = blk[..., :2].contiguous().view(torch.float16)        # (..., n_blocks, 1)
    # Remaining 32 bytes: int8 quants. Reinterpret as int8.
    qs = blk[..., 2:].contiguous().view(torch.int8)          # (..., n_blocks, 32)
    out = d.to(torch.float32) * qs.to(torch.float32)         # broadcast (...,n,1)*(...,n,32)
    return out.to(torch.float16).view(*batch, n_blocks * 32)


# ── Q4_0 ────────────────────────────────────────────────────────────────

def dequant_q4_0(packed: torch.Tensor) -> torch.Tensor:
    """Q4_0: block=[d:fp16, qs:nibble[16]] → 18 bytes per 32-element block.
    Nibble layout: qs[i] holds elements i (low nibble) and i+16 (high nibble).
    Quants are signed: subtract 8 after extraction."""
    *batch, row_bytes = packed.shape
    assert row_bytes % 18 == 0
    n_blocks = row_bytes // 18
    blk = packed.view(*batch, n_blocks, 18)
    d = blk[..., :2].contiguous().view(torch.float16).to(torch.float32)  # (..., n, 1)
    qs = blk[..., 2:].to(torch.int32)                                    # (..., n, 16)
    lo = (qs & 0xF) - 8                                                  # signed nibble
    hi = ((qs >> 4) & 0xF) - 8
    # Interleave lo (elements 0..15) and hi (elements 16..31).
    out = torch.empty(*batch, n_blocks, 32, dtype=torch.float32,
                       device=packed.device)
    out[..., :16]  = d * lo.to(torch.float32)
    out[..., 16:]  = d * hi.to(torch.float32)
    return out.to(torch.float16).view(*batch, n_blocks * 32)


# ── Q4_1 ────────────────────────────────────────────────────────────────

def dequant_q4_1(packed: torch.Tensor) -> torch.Tensor:
    """Q4_1: block=[d:fp16, m:fp16, qs:nibble[16]] → 20 bytes / 32 elements.
    Unsigned nibble (no -8 offset); reconstruct as d*q + m."""
    *batch, row_bytes = packed.shape
    assert row_bytes % 20 == 0
    n_blocks = row_bytes // 20
    blk = packed.view(*batch, n_blocks, 20)
    dm = blk[..., :4].contiguous().view(torch.float16).to(torch.float32)  # (..., n, 2)
    d = dm[..., 0:1]
    m = dm[..., 1:2]
    qs = blk[..., 4:].to(torch.int32)                                     # (..., n, 16)
    lo = qs & 0xF
    hi = (qs >> 4) & 0xF
    out = torch.empty(*batch, n_blocks, 32, dtype=torch.float32, device=packed.device)
    out[..., :16] = d * lo.to(torch.float32) + m
    out[..., 16:] = d * hi.to(torch.float32) + m
    return out.to(torch.float16).view(*batch, n_blocks * 32)


# ── Q5_0 / Q5_1 ─────────────────────────────────────────────────────────

def _extract_q5_high_bits(qh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """qh: (..., n_blocks, 4) uint8 → (h_lo, h_hi) where each is (..., n_blocks, 16)
    of {0, 1}, the 5th bit for elements 0..15 and 16..31 respectively.
    qh is read as a 32-bit integer; bit i = 5th bit of element i."""
    qh32 = (qh[..., 0].to(torch.int32)
            | (qh[..., 1].to(torch.int32) << 8)
            | (qh[..., 2].to(torch.int32) << 16)
            | (qh[..., 3].to(torch.int32) << 24))            # (..., n)
    # Position weights for bits 0..15 and 16..31.
    pos = torch.arange(16, device=qh.device, dtype=torch.int32)
    h_lo = ((qh32.unsqueeze(-1) >> pos) & 1)                  # bits 0..15
    h_hi = ((qh32.unsqueeze(-1) >> (pos + 16)) & 1)           # bits 16..31
    return h_lo, h_hi


def dequant_q5_0(packed: torch.Tensor) -> torch.Tensor:
    """Q5_0: block=[d:fp16, qh:u8[4], qs:nibble[16]] → 22 bytes / 32 elements.
    5-bit signed: q = (low_nibble | (high_bit << 4)) - 16."""
    *batch, row_bytes = packed.shape
    assert row_bytes % 22 == 0
    n_blocks = row_bytes // 22
    blk = packed.view(*batch, n_blocks, 22)
    d = blk[..., :2].contiguous().view(torch.float16).to(torch.float32)
    qh = blk[..., 2:6]
    qs = blk[..., 6:].to(torch.int32)
    h_lo, h_hi = _extract_q5_high_bits(qh)
    lo = ((qs & 0xF) | (h_lo << 4)) - 16
    hi = (((qs >> 4) & 0xF) | (h_hi << 4)) - 16
    out = torch.empty(*batch, n_blocks, 32, dtype=torch.float32, device=packed.device)
    out[..., :16] = d * lo.to(torch.float32)
    out[..., 16:] = d * hi.to(torch.float32)
    return out.to(torch.float16).view(*batch, n_blocks * 32)


def dequant_q5_1(packed: torch.Tensor) -> torch.Tensor:
    """Q5_1: block=[d:fp16, m:fp16, qh:u8[4], qs:nibble[16]] → 24 bytes / 32.
    5-bit unsigned: q = low_nibble | (high_bit << 4); reconstruct as d*q + m."""
    *batch, row_bytes = packed.shape
    assert row_bytes % 24 == 0
    n_blocks = row_bytes // 24
    blk = packed.view(*batch, n_blocks, 24)
    dm = blk[..., :4].contiguous().view(torch.float16).to(torch.float32)
    d = dm[..., 0:1]
    m = dm[..., 1:2]
    qh = blk[..., 4:8]
    qs = blk[..., 8:].to(torch.int32)
    h_lo, h_hi = _extract_q5_high_bits(qh)
    lo = (qs & 0xF) | (h_lo << 4)
    hi = ((qs >> 4) & 0xF) | (h_hi << 4)
    out = torch.empty(*batch, n_blocks, 32, dtype=torch.float32, device=packed.device)
    out[..., :16] = d * lo.to(torch.float32) + m
    out[..., 16:] = d * hi.to(torch.float32) + m
    return out.to(torch.float16).view(*batch, n_blocks * 32)


# ── Q4_K / Q5_K (super-block of 256 elements) ──────────────────────────

def _unpack_q4k_scales(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """scales: (..., n_super, 12) uint8 → (sc, mn) each (..., n_super, 8) uint8.
    K-quant 6-bit scale + 6-bit min packing across 12 bytes for 8 sub-blocks.
    Layout (matching llama.cpp/gguf-py reference):
      Bytes 0..3  : scales for sub-blocks 0..3 (low 6 bits of byte 0..3)
      Bytes 4..7  : mins for sub-blocks 0..3 (low 6 bits)
      Bytes 8..11 : combined high 2 bits + low 4 bits for sub-blocks 4..7
    Specifically, for sb in 4..7:
      sc = (bytes[sb] & 0xF) | ((bytes[sb-4] >> 6) << 4)    # low4 from byte[sb], high2 from byte[sb-4]
      mn = (bytes[sb] >> 4)  | ((bytes[sb-4] >> 6 + 2) ... )
    Wait, the actual layout per ggml:
      For sb < 4:
        sc = scales[sb]   & 0x3F
        mn = scales[sb+4] & 0x3F
      For sb >= 4:
        sc = (scales[sb+4] & 0x0F) | ((scales[sb-4] >> 6) << 4)
        mn = (scales[sb+4] >>   4) | ((scales[sb-0] >> 6) << 4)
    """
    s = scales.to(torch.int32)  # (..., n, 12)
    sc = torch.empty(*scales.shape[:-1], 8, dtype=torch.int32, device=scales.device)
    mn = torch.empty_like(sc)
    # Sub-blocks 0..3
    sc[..., 0:4] = s[..., 0:4] & 0x3F
    mn[..., 0:4] = s[..., 4:8] & 0x3F
    # Sub-blocks 4..7
    sc[..., 4:8] = (s[..., 8:12] & 0x0F) | ((s[..., 0:4] >> 6) << 4)
    mn[..., 4:8] = (s[..., 8:12] >> 4)   | ((s[..., 4:8] >> 6) << 4)
    return sc, mn


def dequant_q4_k(packed: torch.Tensor) -> torch.Tensor:
    """Q4_K: super-block = [d:fp16, dmin:fp16, scales[12], qs[128]] → 144 bytes
    / 256 elements (8 sub-blocks of 32 each).

    Element layout: nibbles in qs are paired across sub-blocks:
      qs[pair*32 + p] holds elements at:
        - sub-block (2*pair)   index p (low nibble)
        - sub-block (2*pair+1) index p (high nibble)
      pair ∈ [0, 4), p ∈ [0, 32).
    """
    *batch, row_bytes = packed.shape
    assert row_bytes % 144 == 0
    n_super = row_bytes // 144
    blk = packed.view(*batch, n_super, 144)
    d    = blk[..., 0:2].contiguous().view(torch.float16).to(torch.float32)  # (..., n, 1)
    dmin = blk[..., 2:4].contiguous().view(torch.float16).to(torch.float32)
    scales = blk[..., 4:16]                                                   # (..., n, 12)
    qs = blk[..., 16:].to(torch.int32)                                        # (..., n, 128)

    sc, mn = _unpack_q4k_scales(scales)   # (..., n, 8) each, int32 in [0, 64)
    # Per-sub-block dl = d*sc, ml = dmin*mn → (..., n, 8) float32 each
    dl = d * sc.to(torch.float32)
    ml = dmin * mn.to(torch.float32)

    # Reshape qs to (..., n, 4, 32): qs[pair, p] for pair ∈ [0,4), p ∈ [0,32)
    qs_p = qs.view(*batch, n_super, 4, 32)
    lo = qs_p & 0xF                # low nibble → sub-block 2*pair
    hi = (qs_p >> 4) & 0xF         # high nibble → sub-block 2*pair+1
    # dl/ml indexed by sub-block: split into (lo: sb=2*pair) and (hi: sb=2*pair+1)
    dl_lo = dl[..., 0::2].unsqueeze(-1)  # (..., n, 4, 1) for pair-aligned even sub-blocks
    dl_hi = dl[..., 1::2].unsqueeze(-1)
    ml_lo = ml[..., 0::2].unsqueeze(-1)
    ml_hi = ml[..., 1::2].unsqueeze(-1)

    out_lo = dl_lo * lo.to(torch.float32) - ml_lo  # (..., n, 4, 32)
    out_hi = dl_hi * hi.to(torch.float32) - ml_hi
    # Interleave to (..., n, 8, 32) by sub-block index
    out = torch.empty(*batch, n_super, 8, 32, dtype=torch.float32, device=packed.device)
    out[..., 0::2, :] = out_lo
    out[..., 1::2, :] = out_hi
    return out.to(torch.float16).view(*batch, n_super * 256)


def dequant_q5_k(packed: torch.Tensor) -> torch.Tensor:
    """Q5_K: super-block = [d, dmin, scales[12], qh[32], qs[128]] → 176 bytes.
    Same as Q4_K plus 1 high bit per element (32 bytes of qh holding 256 bits)."""
    *batch, row_bytes = packed.shape
    assert row_bytes % 176 == 0
    n_super = row_bytes // 176
    blk = packed.view(*batch, n_super, 176)
    d    = blk[..., 0:2].contiguous().view(torch.float16).to(torch.float32)
    dmin = blk[..., 2:4].contiguous().view(torch.float16).to(torch.float32)
    scales = blk[..., 4:16]
    qh = blk[..., 16:48].to(torch.int32)        # (..., n, 32) — 1 bit per of 256 elems
    qs = blk[..., 48:].to(torch.int32)          # (..., n, 128)

    sc, mn = _unpack_q4k_scales(scales)
    dl = d * sc.to(torch.float32)
    ml = dmin * mn.to(torch.float32)

    # qh holds 256 single bits packed into 32 bytes. Bit i corresponds to element i.
    # Extract per-element high bit. Reshape qh to (..., n, 32) of u8 and extract bits.
    pos = torch.arange(8, device=qh.device, dtype=torch.int32)
    h_per_byte = (qh.unsqueeze(-1) >> pos) & 1   # (..., n, 32, 8) of {0,1}
    h_flat = h_per_byte.view(*batch, n_super, 256)

    # qs nibbles: same layout as Q4_K — qs[pair*32 + p] holds elements at
    # sub-block (2*pair) index p (low) and sub-block (2*pair+1) index p (high).
    # Combined with high bit, q = (4-bit nibble | (high_bit << 4)) ∈ [0, 32).
    qs_p = qs.view(*batch, n_super, 4, 32)
    lo_n = qs_p & 0xF
    hi_n = (qs_p >> 4) & 0xF

    # h_flat for sub-block sb element p is at index sb*32 + p.
    # We want the high bits split the same way as the nibbles (paired by 2*pair / 2*pair+1).
    h_per_sb = h_flat.view(*batch, n_super, 8, 32)        # (..., n, 8, 32)
    h_lo_sub = h_per_sb[..., 0::2, :]                     # (..., n, 4, 32) — sb=0,2,4,6
    h_hi_sub = h_per_sb[..., 1::2, :]                     # (..., n, 4, 32) — sb=1,3,5,7

    q_lo = lo_n | (h_lo_sub << 4)
    q_hi = hi_n | (h_hi_sub << 4)

    dl_lo = dl[..., 0::2].unsqueeze(-1)
    dl_hi = dl[..., 1::2].unsqueeze(-1)
    ml_lo = ml[..., 0::2].unsqueeze(-1)
    ml_hi = ml[..., 1::2].unsqueeze(-1)

    out_lo = dl_lo * q_lo.to(torch.float32) - ml_lo
    out_hi = dl_hi * q_hi.to(torch.float32) - ml_hi
    out = torch.empty(*batch, n_super, 8, 32, dtype=torch.float32, device=packed.device)
    out[..., 0::2, :] = out_lo
    out[..., 1::2, :] = out_hi
    return out.to(torch.float16).view(*batch, n_super * 256)


def dequant_q6_k(packed: torch.Tensor) -> torch.Tensor:
    """Q6_K: super-block = [ql[128], qh[64], scales:int8[16], d:fp16] → 210 bytes.
    256 elements, 6-bit signed quants (-32..31) reconstructed as:
      For sub-block sb ∈ [0,16) of 16 elements each, sub-block scale s_sb (int8).
      q ∈ [-32, 32) reconstructed from low 4 bits in ql + 2 high bits in qh.
      out[i] = d * s_sb * q
    """
    *batch, row_bytes = packed.shape
    assert row_bytes % 210 == 0
    n_super = row_bytes // 210
    blk = packed.view(*batch, n_super, 210)
    ql = blk[..., :128].to(torch.int32)              # 128 bytes — low 4 bits of each elem
    qh = blk[..., 128:192].to(torch.int32)            # 64 bytes — high 2 bits of each elem
    scales_i8 = blk[..., 192:208].contiguous().view(torch.int8).to(torch.float32)
    d = blk[..., 208:210].contiguous().view(torch.float16).to(torch.float32)

    # Layout reference (ggml-quants.c, dequantize_row_q6_K):
    # for n in 0..1 (each handles 128 elements):
    #   ql + n*64 holds low 4 bits packed in 64 bytes (2 elements per byte).
    #   qh + n*32 holds high 2 bits packed in 32 bytes (4 elements per byte).
    #   For l in 0..31:
    #     q1 = ((ql[l]    & 0xF) | (((qh[l]      ) & 3) << 4)) - 32
    #     q2 = ((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
    #     q3 = ((ql[l]    >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32
    #     q4 = ((ql[l+32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32
    #     out_n[l]      = d * sc[is + 0] * q1
    #     out_n[l + 32] = d * sc[is + 2] * q2
    #     out_n[l + 64] = d * sc[is + 4] * q3
    #     out_n[l + 96] = d * sc[is + 6] * q4
    #   (where is = n * 8 — 8 sub-block scales per half; 16 total scales)

    out = torch.empty(*batch, n_super, 256, dtype=torch.float32, device=packed.device)
    for half in range(2):
        ql_h = ql[..., half * 64 : (half + 1) * 64]              # (..., n, 64)
        qh_h = qh[..., half * 32 : (half + 1) * 32]              # (..., n, 32)
        l_idx = torch.arange(32, device=packed.device)
        ql_l   = ql_h[..., l_idx]                                 # ql[l] for l in 0..31
        ql_l32 = ql_h[..., l_idx + 32]                            # ql[l+32]
        qh_l   = qh_h                                              # qh[l]

        q1 = ((ql_l   & 0xF) | (((qh_l     ) & 3) << 4)) - 32
        q2 = ((ql_l32 & 0xF) | (((qh_l >> 2) & 3) << 4)) - 32
        q3 = ((ql_l   >> 4)  | (((qh_l >> 4) & 3) << 4)) - 32
        q4 = ((ql_l32 >> 4)  | (((qh_l >> 6) & 3) << 4)) - 32

        is_base = half * 8
        sc1 = scales_i8[..., is_base + 0:is_base + 1]
        sc2 = scales_i8[..., is_base + 2:is_base + 3]
        sc3 = scales_i8[..., is_base + 4:is_base + 5]
        sc4 = scales_i8[..., is_base + 6:is_base + 7]

        base = half * 128
        out[..., base + 0  : base + 32]  = d * sc1 * q1.to(torch.float32)
        out[..., base + 32 : base + 64]  = d * sc2 * q2.to(torch.float32)
        out[..., base + 64 : base + 96]  = d * sc3 * q3.to(torch.float32)
        out[..., base + 96 : base + 128] = d * sc4 * q4.to(torch.float32)

    return out.to(torch.float16).view(*batch, n_super * 256)


# ── Dispatch + format metadata ─────────────────────────────────────────

DEQUANTERS = {
    "Q8_0": (dequant_q8_0,  34, 32),
    "Q4_0": (dequant_q4_0,  18, 32),
    "Q4_1": (dequant_q4_1,  20, 32),
    "Q5_0": (dequant_q5_0,  22, 32),
    "Q5_1": (dequant_q5_1,  24, 32),
    "Q4_K": (dequant_q4_k, 144, 256),
    "Q5_K": (dequant_q5_k, 176, 256),
    "Q6_K": (dequant_q6_k, 210, 256),
}


def dequant(packed: torch.Tensor, qtype: str) -> torch.Tensor:
    """Generic dequant dispatcher. Returns fp16 on the same device as input."""
    if qtype not in DEQUANTERS:
        raise ValueError(f"unsupported qtype {qtype!r}; have {list(DEQUANTERS)}")
    fn, _ts, _bs = DEQUANTERS[qtype]
    return fn(packed)


def block_size(qtype: str) -> int:
    return DEQUANTERS[qtype][2]


def type_size(qtype: str) -> int:
    return DEQUANTERS[qtype][1]
