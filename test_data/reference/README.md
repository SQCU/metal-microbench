# Gemma-4-A4B-It vision reference oracle

Ground-truth outputs of the Gemma 4 vision tower + `embed_vision` projection,
produced for cross-validation of a Metal/Swift reimplementation.

## Model
- Checkpoint: `/Users/mdot/models/gemma-4-a4b-bf16/` (HF bf16)
- Architecture: `Gemma4ForConditionalGeneration`
- Compute dtype: bf16
- Device: `mps`
- text_config.hidden_size: 2816
- vision_config.default_output_length (max soft tokens / image): 280
- vision_config.hidden_size: 1152
- **Measured soft tokens per 640x598 frame: 272** (Gemma 4's image processor adapts token
  count to aspect ratio; 280 is a *max*, not a fixed count.)

## Frames
- Source: `/Users/mdot/metal-microbench/test_data/frames/*.png`
- Count: 414 (all 640x598)
- Filename-sorted order, see `frame_order.txt`.

## Files
- `vision_soft_tokens.npy` — shape `[N, 272, 2816]` fp16. Projected soft tokens from
  `model.embed_vision(model.vision_tower(pixel_values, pixel_position_ids).last_hidden_state)`.
- `frame_order.txt` — one filename per line, matches axis 0 of the npy.
- `image_hashes.json` — SHA-1 of the processor's fp32 `pixel_values` bytes per frame.
  Use this to detect preprocessing non-determinism across runs / machines.
- `caption_logits.npy` / `caption_top1_tokens.json` — NOT generated this run (pass --with-captions).

## Versions
- torch: `2.11.0`
- transformers: `5.5.4`
- MPS available: `True`

## Wall clock
- Full vision pass for 414 frames: 198.1s (2.09 fps) on `mps`.

## Re-run
```bash
cd /Users/mdot/metal-microbench/test_data/reference
.venv/bin/python extract_vision_reference.py            # vision only
.venv/bin/python extract_vision_reference.py --with-captions  # + item 3 (loads full 51 GB model)
```
Or one-shot with uv:
```bash
uv run --with torch --with transformers --with pillow --with numpy --with accelerate \
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
