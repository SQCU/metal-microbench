#!/usr/bin/env python
"""
Reference oracle for Gemma-4-A4B-It vision tower outputs.

Loads ONLY the vision_tower + embed_vision submodules from the HF bf16
checkpoint at /Users/mdot/models/gemma-4-a4b-bf16/, runs every PNG in
/Users/mdot/metal-microbench/test_data/frames/ through the official image
processor, and dumps the projected soft tokens to test_data/reference/.

Item 3 (caption logits) is OPTIONAL and controlled via --with-captions. It
requires loading the full model (51 GB bf16). Default is vision-only.

Run:
    cd /Users/mdot/metal-microbench/test_data/reference
    uv venv --python 3.12                  # one-time
    uv pip install torch torchvision transformers pillow numpy accelerate
    .venv/bin/python extract_vision_reference.py

Or drive it with uv directly:
    uv run --with torch --with transformers --with pillow --with numpy \
        --with accelerate python extract_vision_reference.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
)
from safetensors import safe_open


MODEL_DIR = Path("/Users/mdot/models/gemma-4-a4b-bf16")
FRAMES_DIR = Path("/Users/mdot/metal-microbench/test_data/frames")
OUT_DIR = Path("/Users/mdot/metal-microbench/test_data/reference")
SHARD = MODEL_DIR / "model-00001-of-00002.safetensors"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vision_stack(device: torch.device, dtype: torch.dtype):
    """Instantiate vision_tower + embed_vision and hydrate from the safetensor shard.

    Vision tower is ~0.5 GB of params + small buffers initialized at __init__ time
    (std_bias, std_scale, inv_freq). We instantiate on CPU so buffers get real values,
    then overwrite the trainable params from the shard, then move to `device`.
    """
    cfg = AutoConfig.from_pretrained(MODEL_DIR)

    # Instantiate real modules on CPU (not meta — we need the non-shard buffers to init).
    prev_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        vision_tower = Gemma4VisionModel(cfg.vision_config)
        embed_vision = Gemma4MultimodalEmbedder(cfg.vision_config, cfg.text_config)
    finally:
        torch.set_default_dtype(prev_default)

    # Collect state dicts from shard 1 for just these two submodules.
    vt_prefix = "model.vision_tower."
    ev_prefix = "model.embed_vision."
    vt_state: dict[str, torch.Tensor] = {}
    ev_state: dict[str, torch.Tensor] = {}
    with safe_open(str(SHARD), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(vt_prefix):
                vt_state[key[len(vt_prefix):]] = f.get_tensor(key)
            elif key.startswith(ev_prefix):
                ev_state[key[len(ev_prefix):]] = f.get_tensor(key)

    missing_vt, unexpected_vt = vision_tower.load_state_dict(vt_state, strict=False)
    missing_ev, unexpected_ev = embed_vision.load_state_dict(ev_state, strict=False)
    if missing_vt or unexpected_vt:
        print(f"[vision_tower] missing={len(missing_vt)} unexpected={len(unexpected_vt)}")
        if missing_vt:
            print("  first missing:", missing_vt[:5])
        if unexpected_vt:
            print("  first unexpected:", unexpected_vt[:5])
    if missing_ev or unexpected_ev:
        print(f"[embed_vision] missing={len(missing_ev)} unexpected={len(unexpected_ev)}")
        if missing_ev:
            print("  first missing:", missing_ev[:5])
        if unexpected_ev:
            print("  first unexpected:", unexpected_ev[:5])

    vision_tower = vision_tower.to(device=device, dtype=dtype).eval()
    embed_vision = embed_vision.to(device=device, dtype=dtype).eval()
    return cfg, vision_tower, embed_vision


def process_frame(processor, img_path: Path):
    img = Image.open(img_path).convert("RGB")
    out = processor.image_processor(images=[img], return_tensors="pt")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N frames (for smoke tests).")
    parser.add_argument("--with-captions", action="store_true",
                        help="(Optional) Also produce caption_logits.npy. Loads full model.")
    parser.add_argument("--device", default=None, choices=[None, "mps", "cpu"], nargs="?")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else pick_device()
    dtype = torch.bfloat16
    print(f"Device: {device}  | compute dtype: {dtype}")

    # --- Load processor + vision stack ---
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    print("Loading vision_tower + embed_vision from", SHARD.name, "...")
    t0 = time.time()
    cfg, vision_tower, embed_vision = load_vision_stack(device, dtype)
    print(f"  loaded in {time.time() - t0:.1f}s")

    text_hidden = cfg.text_config.hidden_size
    max_soft = cfg.vision_config.default_output_length
    print(f"Config: text_hidden_size={text_hidden}  vision_soft_tokens(max)={max_soft}")

    # --- Enumerate frames ---
    frames = sorted(FRAMES_DIR.glob("*.png"))
    if args.limit:
        frames = frames[: args.limit]
    n_frames = len(frames)
    if n_frames == 0:
        print("No frames found, bailing.", file=sys.stderr)
        sys.exit(1)
    print(f"Frames: {n_frames}")

    # --- Sanity run on first frame ---
    print("\n=== Sanity run on frame 0 ===")
    sanity_in = process_frame(processor, frames[0])
    px = sanity_in["pixel_values"].to(device)
    pos = sanity_in["image_position_ids"].to(device)
    n_soft_measured = int(sanity_in["num_soft_tokens_per_image"][0].item())
    print(f"pixel_values: shape={tuple(px.shape)} dtype={px.dtype}")
    print(f"image_position_ids: shape={tuple(pos.shape)} dtype={pos.dtype}")
    print(f"num_soft_tokens_per_image (measured): {n_soft_measured}")
    if n_soft_measured != max_soft:
        print(
            f"NOTE: measured soft-token count {n_soft_measured} != config.default_output_length {max_soft}.\n"
            f"      This is expected — the Gemma 4 image processor adapts output length to aspect ratio.\n"
            f"      All 414 frames are 640x598 so every frame yields {n_soft_measured} soft tokens."
        )

    t0 = time.time()
    with torch.inference_mode():
        vt_out = vision_tower(pixel_values=px.to(dtype), pixel_position_ids=pos)
        last_hidden = vt_out.last_hidden_state
        soft = embed_vision(inputs_embeds=last_hidden)
    if device.type == "mps":
        torch.mps.synchronize()
    sanity_dt = time.time() - t0
    print(f"last_hidden_state: shape={tuple(last_hidden.shape)} dtype={last_hidden.dtype}")
    print(f"soft_tokens: shape={tuple(soft.shape)} dtype={soft.dtype}")
    soft_f32 = soft.float().cpu()
    mean = soft_f32.mean().item()
    std = soft_f32.std().item()
    first8 = soft_f32.flatten()[:8].tolist()
    print(f"soft_tokens stats: mean={mean:.5f} std={std:.5f}")
    print(f"first 8 values: {[f'{v:.5f}' for v in first8]}")
    print(f"wall-clock 1 frame: {sanity_dt:.3f}s")

    # The vision_tower/embed_vision output is [num_soft, hidden] with no batch dim
    # (both modules fold the 2D spatial layout into a flat token sequence). We wrap in batch.
    if soft.dim() == 2:
        soft = soft.unsqueeze(0)
    expected_shape = (1, n_soft_measured, text_hidden)
    if tuple(soft.shape) != expected_shape:
        print(
            f"FATAL: soft-token shape {tuple(soft.shape)} != {expected_shape}",
            file=sys.stderr,
        )
        sys.exit(2)
    # Confirm soft-token hidden dim equals text_config.hidden_size (spec sanity check).
    if soft.shape[-1] != cfg.text_config.hidden_size:
        print(
            f"FATAL: soft-token hidden dim {soft.shape[-1]} != text_config.hidden_size {cfg.text_config.hidden_size}",
            file=sys.stderr,
        )
        sys.exit(2)
    if sanity_dt > 60.0:
        print(f"FATAL: >60s per frame ({sanity_dt:.1f}s) — bailing per spec.", file=sys.stderr)
        sys.exit(3)

    est_total = sanity_dt * n_frames
    print(f"Estimated total for {n_frames} frames: {est_total/60:.1f} min")

    # --- Full run ---
    all_soft = np.empty((n_frames, n_soft_measured, text_hidden), dtype=np.float16)
    hashes: dict[str, str] = {}
    order: list[str] = []
    per_frame_soft_count: list[int] = []

    print(f"\n=== Full run ===")
    run_start = time.time()
    for i, fp in enumerate(frames):
        order.append(fp.name)
        pp = process_frame(processor, fp)
        # Hash of processed pixel_values (float32 bytes) for determinism checks.
        px_f32 = pp["pixel_values"].contiguous().float()
        h = hashlib.sha1(px_f32.cpu().numpy().tobytes()).hexdigest()
        hashes[fp.name] = h

        px_d = pp["pixel_values"].to(device=device, dtype=dtype)
        pos_d = pp["image_position_ids"].to(device=device)
        ns = int(pp["num_soft_tokens_per_image"][0].item())
        per_frame_soft_count.append(ns)
        with torch.inference_mode():
            vt_out = vision_tower(pixel_values=px_d, pixel_position_ids=pos_d)
            soft = embed_vision(inputs_embeds=vt_out.last_hidden_state)
        if soft.dim() == 2:
            soft = soft.unsqueeze(0)
        if tuple(soft.shape) != (1, n_soft_measured, text_hidden):
            print(
                f"FATAL: frame {fp.name} produced soft-token shape {tuple(soft.shape)}, "
                f"expected (1, {n_soft_measured}, {text_hidden}). Aborting.",
                file=sys.stderr,
            )
            sys.exit(4)
        all_soft[i] = soft[0].float().cpu().numpy().astype(np.float16)

        if (i + 1) % 25 == 0 or i == n_frames - 1:
            elapsed = time.time() - run_start
            rate = (i + 1) / elapsed
            eta = (n_frames - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_frames}] {elapsed:.1f}s  {rate:.2f} fps  ETA {eta:.1f}s")

    total_dt = time.time() - run_start
    print(f"Full run: {total_dt:.1f}s ({n_frames/total_dt:.2f} fps)")

    # --- Save ---
    soft_path = OUT_DIR / "vision_soft_tokens.npy"
    np.save(soft_path, all_soft)
    print(f"Saved {soft_path.name}: {all_soft.shape} {all_soft.dtype}  "
          f"({soft_path.stat().st_size / 1e6:.2f} MB)")

    order_path = OUT_DIR / "frame_order.txt"
    order_path.write_text("\n".join(order) + "\n")
    print(f"Saved {order_path.name}: {len(order)} entries")

    hashes_path = OUT_DIR / "image_hashes.json"
    with open(hashes_path, "w") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)
    print(f"Saved {hashes_path.name}: {len(hashes)} entries")

    # --- Optional item 3: caption logits ---
    if args.with_captions:
        print("\n=== Captions (item 3) ===")
        caption_run(frames, all_soft, processor, cfg, device, dtype)

    # --- README ---
    write_readme(cfg, n_frames, total_dt, device, n_soft_measured,
                 with_captions=args.with_captions)
    print("\nDone.")


def caption_run(frames, all_soft_np, processor, cfg, device, dtype):
    """Optional: run the full model and dump top-20 logits + top-1 strings for 32 greedy steps."""
    from transformers import Gemma4ForConditionalGeneration

    # Load full model — this will take minutes and consume ~51 GB.
    print("Loading full Gemma4ForConditionalGeneration (this is slow)...")
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    tokenizer = processor.tokenizer
    max_new = 32
    topk = 20
    prompt = "<start_of_image>describe this image:"

    n = len(frames)
    logits_out = np.zeros((n, max_new, topk), dtype=np.float16)
    top1_ids_out = np.full((n, max_new), -1, dtype=np.int32)
    decoded: dict[str, list[str]] = {}

    # Build the prompt embeddings once; reuse for every frame.
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    # We splice in soft tokens after the <start_of_image> marker. Locate it.
    boi = cfg.boi_token_id
    boi_pos = (prompt_ids[0] == boi).nonzero(as_tuple=True)[0]
    if len(boi_pos) == 0:
        print("WARNING: <start_of_image> not in prompt tokens; captions may be wrong.", file=sys.stderr)

    embed_tokens = model.get_input_embeddings()

    for i in range(n):
        # soft_np is fp16; upcast to compute dtype.
        soft = torch.from_numpy(all_soft_np[i]).to(device=device, dtype=dtype).unsqueeze(0)
        # Build embeds: [ ...prompt up to and including boi..., soft_tokens, ...rest of prompt... ]
        if len(boi_pos) > 0:
            cut = int(boi_pos[0].item()) + 1
            pre = embed_tokens(prompt_ids[:, :cut])
            post = embed_tokens(prompt_ids[:, cut:])
            inputs_embeds = torch.cat([pre, soft, post], dim=1)
        else:
            inputs_embeds = torch.cat([embed_tokens(prompt_ids), soft], dim=1)

        generated_ids = []
        top_strs = []
        past = None
        current_embeds = inputs_embeds
        with torch.inference_mode():
            for step in range(max_new):
                out = model(
                    inputs_embeds=current_embeds,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                logits = out.logits[:, -1, :].float()
                topv, topi = torch.topk(logits[0], topk)
                logits_out[i, step, :] = topv.cpu().numpy().astype(np.float16)
                nxt = int(topi[0].item())
                top1_ids_out[i, step] = nxt
                generated_ids.append(nxt)
                top_strs.append(tokenizer.decode([nxt], skip_special_tokens=False))
                if nxt in (cfg.text_config.eos_token_id, 1):
                    break
                current_embeds = embed_tokens(torch.tensor([[nxt]], device=device))
        decoded[frames[i].name] = top_strs
        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  captions [{i+1}/{n}]: {''.join(top_strs)[:80]}")

    np.save(OUT_DIR / "caption_logits.npy", logits_out)
    with open(OUT_DIR / "caption_top1_tokens.json", "w") as f:
        json.dump(decoded, f, indent=2)
    print(f"Saved caption_logits.npy {logits_out.shape} and caption_top1_tokens.json")


def write_readme(cfg, n_frames, total_dt, device, n_soft: int, with_captions: bool):
    import torch as _torch
    import transformers as _tf

    text = f"""# Gemma-4-A4B-It vision reference oracle

Ground-truth outputs of the Gemma 4 vision tower + `embed_vision` projection,
produced for cross-validation of a Metal/Swift reimplementation.

## Model
- Checkpoint: `/Users/mdot/models/gemma-4-a4b-bf16/` (HF bf16)
- Architecture: `Gemma4ForConditionalGeneration`
- Compute dtype: bf16
- Device: `{device}`
- text_config.hidden_size: {cfg.text_config.hidden_size}
- vision_config.default_output_length (max soft tokens / image): {cfg.vision_config.default_output_length}
- vision_config.hidden_size: {cfg.vision_config.hidden_size}
- **Measured soft tokens per 640x598 frame: {n_soft}** (Gemma 4's image processor adapts token
  count to aspect ratio; 280 is a *max*, not a fixed count.)

## Frames
- Source: `/Users/mdot/metal-microbench/test_data/frames/*.png`
- Count: {n_frames} (all 640x598)
- Filename-sorted order, see `frame_order.txt`.

## Files
- `vision_soft_tokens.npy` — shape `[N, {n_soft}, {cfg.text_config.hidden_size}]` fp16. Projected soft tokens from
  `model.embed_vision(model.vision_tower(pixel_values, pixel_position_ids).last_hidden_state)`.
- `frame_order.txt` — one filename per line, matches axis 0 of the npy.
- `image_hashes.json` — SHA-1 of the processor's fp32 `pixel_values` bytes per frame.
  Use this to detect preprocessing non-determinism across runs / machines.
- `caption_logits.npy` / `caption_top1_tokens.json` — {"present" if with_captions else "NOT generated this run (pass --with-captions)"}.

## Versions
- torch: `{_torch.__version__}`
- transformers: `{_tf.__version__}`
- MPS available: `{_torch.backends.mps.is_available()}`

## Wall clock
- Full vision pass for {n_frames} frames: {total_dt:.1f}s ({n_frames/total_dt:.2f} fps) on `{device}`.

## Re-run
```bash
cd /Users/mdot/metal-microbench/test_data/reference
.venv/bin/python extract_vision_reference.py            # vision only
.venv/bin/python extract_vision_reference.py --with-captions  # + item 3 (loads full 51 GB model)
```
Or one-shot with uv:
```bash
uv run --with torch --with transformers --with pillow --with numpy --with accelerate \\
    python /Users/mdot/metal-microbench/test_data/reference/extract_vision_reference.py
```

## Gotchas
- The processor output is pre-patched: `pixel_values` shape is `[1, 2520, 768]` (2520 = 280*9 pre-pool
  patches at 3x3 pooling, 768 = 16*16*3 RGB patch flat). You feed it directly to `Gemma4VisionModel`,
  not a 4D image tensor.
- `Gemma4Processor(images=...)` requires a text arg; call `processor.image_processor(images=...)` directly
  when you only want pixel tensors.
- Only `model-00001-of-00002.safetensors` carries the vision weights — the script loads just what it needs
  and skips the 46 GB text decoder unless `--with-captions` is set.
"""
    (OUT_DIR / "README.md").write_text(text)
    print(f"Saved README.md")


if __name__ == "__main__":
    main()
