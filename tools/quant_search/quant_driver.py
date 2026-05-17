"""Quant driver — produce a GGUF from a base model + per-tensor quant config.

V1: minimal-functional. Uses the gguf-py library to read/write GGUF files.
For per-tensor quantization we shell out to llama.cpp's `llama-quantize`
binary at the per-tensor-class level (which is what llama-quantize already
supports via --token-embedding-type / --output-tensor-type / etc).

The set of valid (tensor_class, format) cells is read from
`<repo>/kernel_capabilities.json` — the SAME file the Metal engine loads
at boot for its dispatcher validation. Adding a new cell here means
adding a corresponding kernel to kernels.swift; both interfaces will see
the change.

A "config" is a dict of:
    {
        "base_kind": "bf16" | "fp16",        # source format
        "tensor_classes": {
            # Each key must be a tensor_class declared in
            # kernel_capabilities.json's "tensor_classes" section.
            # Each value must appear in that class's "allowed" list.
            "attn_q": "Q5_K",
            "ffn_gate_up_exps": "Q4_K",
            ...
            "default": "Q5_K",                 # fallback for unspecified
        },
    }

Status note: full per-tensor-class custom-quant via gguf-py is achievable
but requires a meaningful amount of GGML quantization code to be exposed
to Python. For now this module's `materialize_config` checks whether the
target GGUF already exists in the cache; if not, raises with a
make-llama-quantize-incantation hint. This unblocks running the search
over PRE-BUILT alternative GGUFs immediately.
"""
from __future__ import annotations
import functools
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

# Where pre-built and search-generated GGUFs live.
QUANT_CACHE_DIR = Path(
    os.environ.get("QUANT_CACHE_DIR",
                    "/Users/mdot/models/gemma-4-a4b-quant-search")
)
# llama.cpp binaries (assumed installed via `make llama-quantize` etc).
LLAMA_QUANTIZE = Path(
    os.environ.get("LLAMA_QUANTIZE",
                    "/Users/mdot/llama.cpp/build/bin/llama-quantize")
)

# Repo-root capabilities config — single source of truth for both this
# Python search code and the Metal engine's dispatcher validation.
KERNEL_CAPABILITIES_JSON = Path(
    os.environ.get("KERNEL_CAPABILITIES_JSON",
                    str(Path(__file__).resolve().parents[2] / "kernel_capabilities.json"))
)


@functools.lru_cache(maxsize=1)
def kernel_capabilities() -> dict:
    """Load and cache the engine's kernel-capability matrix.

    Returns the parsed JSON document. Both this Python search code and
    the Metal engine read this file at boot, so adding a (tensor_class,
    format) cell here is the single trigger to expand the search space.
    """
    if not KERNEL_CAPABILITIES_JSON.exists():
        raise FileNotFoundError(
            f"kernel_capabilities.json not found at {KERNEL_CAPABILITIES_JSON}. "
            "This file is the source of truth for kernel coverage; both the "
            "search and the engine read it. Set KERNEL_CAPABILITIES_JSON env "
            "var to point at it, or restore it from version control."
        )
    return json.loads(KERNEL_CAPABILITIES_JSON.read_text())


def supported_formats() -> list[str]:
    """All quantization formats the engine has kernels for."""
    return list(kernel_capabilities()["formats"].keys())


def allowed_formats_for(tensor_class: str) -> list[str]:
    """Which formats the engine can dispatch for this tensor class."""
    tc = kernel_capabilities()["tensor_classes"]
    if tensor_class not in tc:
        raise KeyError(f"Unknown tensor class '{tensor_class}'. Known: {list(tc.keys())}")
    return list(tc[tensor_class]["allowed"])


def llama_quantize_mixes() -> list[dict]:
    """Pre-defined llama-quantize tag presets (Q4_K_M, Q5_K_M, etc.).
    Each entry has {tag, kind, ...} — see kernel_capabilities.json schema.
    """
    return [m for m in kernel_capabilities()["llama_quantize_mixes"]
            if m["kind"] != "uniform_unsupported"]


def config_id(config: dict) -> str:
    """Stable filename slug for a config — short hash + base_kind tag."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]
    return f"{config['base_kind']}_{h}"


def materialize_config(config: dict, base_fp16_gguf: Path) -> Path:
    """Produce a GGUF for `config`, returning the on-disk path.

    Cached: subsequent calls with the same config short-circuit.

    For now: we DON'T do per-tensor quantization here. Instead, the config
    is interpreted as picking ONE OF a small set of pre-built GGUFs that
    match its tensor_classes. If the requested config doesn't match any
    pre-built, we raise. To unblock the search, generate the pre-builds
    once via the helper below.

    Args:
        config: see module docstring
        base_fp16_gguf: path to the source FP16 GGUF (only consulted if
                        we ever expand to in-process quantization)
    """
    QUANT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = QUANT_CACHE_DIR / f"{config_id(config)}.gguf"
    if target.exists():
        return target
    # Try to map the config to a known llama-quantize "type" (e.g. Q4_K_M,
    # Q5_K_M, Q8_0, Q4_0). If all tensor_classes are uniform-ish this works.
    quant_type = _config_to_quant_type(config)
    if quant_type is None:
        raise NotImplementedError(
            f"Config doesn't match a pre-built quant type. "
            f"Per-tensor custom quant requires patching llama-quantize "
            f"(tracked as TODO). Config: {json.dumps(config, indent=2)}"
        )
    if not LLAMA_QUANTIZE.exists():
        raise FileNotFoundError(
            f"llama-quantize binary not found at {LLAMA_QUANTIZE}. "
            f"Build llama.cpp and set LLAMA_QUANTIZE env var."
        )
    if not base_fp16_gguf.exists():
        raise FileNotFoundError(f"base FP16 GGUF not found at {base_fp16_gguf}")
    print(f"[quant_driver] generating {target.name} as {quant_type} from {base_fp16_gguf.name}")
    proc = subprocess.run(
        [str(LLAMA_QUANTIZE), str(base_fp16_gguf), str(target), quant_type],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"llama-quantize failed: {proc.stderr[-500:]}")
    return target


def _config_to_quant_type(config: dict) -> str | None:
    """Map a config to a llama-quantize type string from the
    `llama_quantize_mixes` section of kernel_capabilities.json. Returns
    None if the config doesn't match any preset (in which case the
    materializer would need true per-tensor quantization, currently TODO).

    `token_embd` is excluded from the match because llama-quantize
    enforces its own embed-precision policy (typically promotes to Q8_0
    or Q6_K regardless of the requested tag); the engine's auto-loader
    handles whatever embed dtype actually lands in the GGUF. So matching
    on the bulk classes is enough to pick the correct llama-quantize tag.
    """
    tc = config.get("tensor_classes", {})
    bulk = {k: v for k, v in tc.items() if k != "token_embd"}
    for mix in llama_quantize_mixes():
        if mix["kind"] == "uniform":
            fmt = mix["format"]
            if all(v == fmt for v in bulk.values()):
                return mix["tag"]
        elif mix["kind"] == "k_quant_mix":
            moe_up_ok = tc.get("ffn_gate_up_exps") == mix["moe_up"]
            moe_down_ok = tc.get("ffn_down_exps") == mix["moe_down"]
            # K-quant mixes have a "default" dense format. Dense tensors
            # are accepted as either dense_default OR moe_up (since
            # llama-quantize sometimes emits the moe-up format on dense
            # at certain layers — see Q4_K_M's mixed dense Q8_0/Q5_K).
            allowed_dense = {mix["dense_default"], mix.get("moe_up", "")}
            dense_classes = ("attn_q", "attn_k", "attn_v", "attn_output",
                             "ffn_gate", "ffn_up", "ffn_down", "default")
            dense_ok = all(tc.get(k, mix["dense_default"]) in allowed_dense
                           for k in dense_classes)
            if moe_up_ok and moe_down_ok and dense_ok:
                return mix["tag"]
    return None


def validate_config(config: dict) -> None:
    """Raise ValueError if any tensor_class assignment in `config` lands
    outside the engine's kernel-capability matrix. This catches search
    enumeration bugs early — the engine itself would also fail at GGUF
    load, but enumerating an invalid config wastes a llama-quantize run.
    """
    caps = kernel_capabilities()
    for cls, fmt in config.get("tensor_classes", {}).items():
        if cls == "default":
            continue
        if cls not in caps["tensor_classes"]:
            raise ValueError(
                f"Config assigns format to unknown tensor class '{cls}'. "
                f"Add it to kernel_capabilities.json or remove from config."
            )
        allowed = caps["tensor_classes"][cls]["allowed"]
        if fmt not in allowed:
            raise ValueError(
                f"Config assigns {fmt} to {cls}, but engine has no kernel "
                f"for that combination. Allowed for {cls}: {allowed}. "
                f"Add a kernel + capabilities entry, or pick from allowed."
            )


def initial_search_grid() -> list[dict]:
    """Seed configs for the search. Each entry is engine-servable per the
    capability matrix. Two families:

    1. **Uniform-bulk per format** — every dense+MoE tensor uses one format,
       with token_embd overridden to a format the embed-dequant supports
       (Q8_0 or Q6_K — see kernel_capabilities.json's token_embd "allowed"
       list). For formats that some bulk-class doesn't support (e.g. Q5_K
       isn't in moe_down's allowed list), the uniform-bulk config is
       skipped — covered by the k-quant mixes instead.

    2. **K-quant mixes** — Q4_K_M, Q5_K_M, etc., as declared in the JSON's
       `llama_quantize_mixes` section.

    All entries are validated against the capability matrix before return.
    """
    grid = []
    caps = kernel_capabilities()
    tensor_classes = caps["tensor_classes"]

    # Bulk classes = every tensor class except token_embd (which always
    # gets overridden to an embed-dequant-compatible format).
    bulk_classes = [c for c in tensor_classes if c != "token_embd"]
    embed_allowed = tensor_classes["token_embd"]["allowed"]

    # Uniform-bulk baselines.
    for fmt in supported_formats():
        # Skip if any bulk class can't take this format.
        if any(fmt not in tensor_classes[c]["allowed"] for c in bulk_classes):
            continue
        # Pick an embed format: prefer matching fmt if allowed, else fall
        # back to the canonical Q8_0 (always present in embed_allowed).
        embed_fmt = fmt if fmt in embed_allowed else "Q8_0"
        cfg = {
            "base_kind": "bf16",
            "tag": fmt,
            "tensor_classes": {
                "default": fmt, "token_embd": embed_fmt,
                **{c: fmt for c in bulk_classes},
            },
        }
        grid.append(cfg)

    # K-quant mixes from the JSON's llama_quantize_mixes section.
    for mix in llama_quantize_mixes():
        if mix["kind"] != "k_quant_mix":
            continue
        dense = mix["dense_default"]
        moe_up = mix["moe_up"]
        moe_down = mix["moe_down"]
        cfg = {
            "base_kind": "bf16",
            "tag": mix["tag"],
            "tensor_classes": {
                "default": dense,
                "token_embd": "Q8_0",
                "attn_q": dense, "attn_k": dense, "attn_v": dense, "attn_output": dense,
                "ffn_gate": dense, "ffn_up": dense, "ffn_down": dense,
                "ffn_gate_up_exps": moe_up,
                "ffn_down_exps": moe_down,
            },
        }
        grid.append(cfg)

    for cfg in grid:
        validate_config(cfg)
    return grid
