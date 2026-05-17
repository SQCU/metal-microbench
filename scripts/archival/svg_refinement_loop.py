#!/usr/bin/env python3
"""Iterative SVG approximation harness — Gemma-4 as a vision-enabled
agent that produces SVG interpretations of a target image, receives
MSE feedback plus a rendering of its previous attempt, and iterates.

Edge-silicon scoped: no backwards pass, no hidden-representation
poking — just Gemma's chat API, PIL rasterization, numpy for MSE,
all external to the model.

Target image sources:
  - `--target <path>`: raster file (PNG/JPEG/etc.)
  - `--donut <seed>`: procedurally generate a torus SDF render (needs
    torch+MPS; uses notes/donut_generator/dataset_torus_native.py)
  - default: tiny synthesized shape (for smoke testing)

Output mode (how Gemma is asked to produce SVG):
  - `svg` (default): emit SVG directly in a ```svg ... ``` code block
  - `python`: emit `def make_svg() -> str: ...` in a ```python ... ```
    code block; we subprocess-exec it (120s timeout) and use the
    returned string. Lets the model compose SVG programmatically.

Feedback shape per iteration:
  - Scalar MSE (previous attempt vs target)
  - Rendering of previous attempt as image_url
  - (optional) Pixel-difference heatmap, also as image_url

Stopping:
  - max_iterations (default 5)
  - MSE threshold for early exit (default 0.003)
  - Plateau detection: no improvement over 2 iterations
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svg_render import render_svg


BASE = os.environ.get("GEMMA_BASE", "http://127.0.0.1:8000")
REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / "notes" / "runs"


# ── HTTP ─────────────────────────────────────────────────────────────

def post(path, body, timeout=300.0):
    req = urllib.request.Request(BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(messages, max_tokens=2048, temperature=0.5, seed=None):
    body = {"messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}
    if seed is not None: body["seed"] = int(seed)
    r = post("/v1/chat/completions", body, timeout=600.0)
    return r["choices"][0]["message"]["content"]


def _count_images(messages) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image_url")
    return n


def chat_stream(messages, max_tokens=2048, temperature=1.0, seed=None,
                 timeout=600.0):
    """Streaming POST to /v1/chat/completions. Returns (text, metrics).

    metrics: {
        ts_submit, ttft_s, decode_s, total_s,
        prompt_tokens, completion_tokens, finish_reason, num_images
    }
    prompt_tokens / completion_tokens come from the final chunk's usage
    (requires server to honor stream_options.include_usage).
    """
    num_images = _count_images(messages)
    body = {
        "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None: body["seed"] = int(seed)
    # STRUCTURED_COT env var: comma-separated phase labels, or "1" for
    # the default GOAL/APPROACH/EDGE preset. Empty/unset → no constraint.
    sc_env = os.environ.get("STRUCTURED_COT", "").strip()
    if sc_env:
        if sc_env == "1" or sc_env.lower() == "true":
            body["structured_cot"] = True
        else:
            body["structured_cot"] = [s.strip() for s in sc_env.split(",") if s.strip()]
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                  "Accept": "text/event-stream"},
        method="POST")

    ts_submit = time.time()
    ts_first = None
    ts_last = None
    chunks: list[str] = []
    usage = None
    finish_reason = None

    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for ch in obj.get("choices", []):
                delta = ch.get("delta") or {}
                content = delta.get("content")
                if content:
                    if ts_first is None:
                        ts_first = time.time()
                    chunks.append(content)
                    ts_last = time.time()
                fr = ch.get("finish_reason")
                if fr:
                    finish_reason = fr
            u = obj.get("usage")
            if u:
                usage = u

    now = time.time()
    if ts_last is None: ts_last = now
    if ts_first is None: ts_first = ts_last
    text = "".join(chunks)
    metrics = {
        "ts_submit": ts_submit,
        "ttft_s": ts_first - ts_submit,
        "decode_s": max(0.0, ts_last - ts_first),
        "total_s": ts_last - ts_submit,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "finish_reason": finish_reason,
        "num_images": num_images,
    }
    return text, metrics


# ── Target image sources ─────────────────────────────────────────────

def load_target_from_path(path: str) -> Image.Image:
    """Raw RGB frame, no harness-side resize. The vision tower's
    aspect-preserving bicubic+antialias is the only resample stage
    the model perceives. The SVG is rasterized at the same native
    resolution for MSE so the rubric matches what the model saw."""
    return Image.open(path).convert("RGB")


def generate_donut_target(seed: int, size: int) -> Image.Image:
    """Procedurally render one donut via the SDF generator."""
    import torch
    sys.path.insert(0, str(REPO / "notes" / "donut_generator"))
    from dataset_torus_native import TorusIterator
    torch.manual_seed(seed)
    it = TorusIterator(device="mps")
    imgs = it.generate_batch(batch_size=1, resolution=size)
    arr = (imgs[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def synth_smoke_target(size: int) -> Image.Image:
    """Tiny deterministic test image for smoke runs."""
    from PIL import ImageDraw
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    # Two overlapping colored shapes.
    d.ellipse((size*0.15, size*0.15, size*0.65, size*0.65), fill=(180, 60, 60))
    d.rectangle((size*0.35, size*0.35, size*0.85, size*0.85), fill=(60, 120, 200))
    return img


# ── Extracting Gemma's output ────────────────────────────────────────

def _strip_channel(t: str) -> str:
    OPEN, CLOSE, TURN = "<|channel>", "<channel|>", "<turn|>"
    out, i = [], 0
    while i < len(t):
        if t.startswith(OPEN, i):
            j = t.find(CLOSE, i + len(OPEN))
            if j < 0:
                out.append(t[i:]); break
            i = j + len(CLOSE)
        elif t.startswith(TURN, i):
            i += len(TURN)
        else:
            out.append(t[i]); i += 1
    return "".join(out)


def _find_svg_block(s: str) -> str | None:
    """Find the first <svg ...>...</svg> block (case-insensitive on the
    tag, case-preserving on the content). The end tag </svg> is atomic
    enough in practice for Gemma's emits that simple substring search
    works; we lowercase a copy for tag-position discovery only.
    """
    lo = s.lower()
    start = lo.find("<svg")
    if start < 0:
        return None
    # Find matching '>' to close the open tag — i.e. the first '>' after
    # `<svg` that isn't inside an attribute-quoted region. Gemma's emits
    # don't put '>' inside quoted attribute values; a plain find is fine.
    open_close = lo.find(">", start + 4)
    if open_close < 0:
        return None
    end = lo.find("</svg>", open_close + 1)
    if end < 0:
        return None
    return s[start:end + len("</svg>")]


def _extract_fenced_block(s: str, langs: tuple[str, ...]) -> str | None:
    """Find the first ```{lang}\\n...``` block whose declared language is
    in `langs` (or empty when `""` is in `langs`). Replaces fence-
    extraction regexes; pure string ops on the '```' boundary.
    """
    parts = s.split("```")
    for idx in range(1, len(parts), 2):
        block = parts[idx]
        nl = block.find("\n")
        if nl < 0:
            continue
        lang = block[:nl].strip().lower()
        if lang in langs:
            return block[nl + 1:]
    return None


def extract_svg_directly(resp: str) -> str | None:
    """Pull the first <svg>...</svg> block from Gemma's response."""
    s = _strip_channel(resp).strip()
    fenced = _extract_fenced_block(s, ("", "svg", "xml", "html"))
    if fenced is not None:
        inner = _find_svg_block(fenced)
        if inner is not None:
            return inner.strip()
    bare = _find_svg_block(s)
    if bare is not None:
        return bare.strip()
    return None


def extract_python_and_run(resp: str, timeout_s: float = 120.0) -> str | None:
    """Pull the first ```python``` block from Gemma's response, exec it
    in a subprocess with a wall-clock timeout, return the function's
    returned string (expected to be SVG). None on any failure."""
    s = _strip_channel(resp).strip()
    code = _extract_fenced_block(s, ("", "python", "py"))
    if code is None:
        return None
    # We expect a `make_svg() -> str` function definition. We append a
    # print(make_svg()) to the subprocess code to capture stdout.
    runner = code + "\n\nimport sys\nsys.stdout.write(make_svg())\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout or None


# ── Rendering + MSE ──────────────────────────────────────────────────

def mse_images(a: Image.Image, b: Image.Image) -> float:
    """RGB pixel-wise MSE in [0,1] units (both images normalized to
    the same size, converted to float32, averaged)."""
    assert a.size == b.size, f"size mismatch: {a.size} vs {b.size}"
    aa = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    bb = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    return float(((aa - bb) ** 2).mean())


def diff_heatmap(a: Image.Image, b: Image.Image) -> Image.Image:
    """Per-pixel absolute-difference heatmap, scaled to [0, 255] for
    visual punch. RGB channels preserved (shows where each channel
    disagrees)."""
    d = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    # Amplify small differences so they're visible.
    arr = np.asarray(d, dtype=np.float32) * 3.0
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))


# ── Base64 image packaging for chat ──────────────────────────────────

def image_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


# ── Prompt construction ──────────────────────────────────────────────

SYSTEM_SVG = (
    "You are an expert at producing SVG approximations of raster images. "
    "Given a reference image, you return a single <svg> document that "
    "approximates the image using simple shapes (rect, circle, ellipse, "
    "polygon, line, path with only M/L/H/V/Z commands). No text, no "
    "images, no gradients, no filters — shapes + solid colors only. "
    "Canvas: the SVG must declare viewBox=\"0 0 {W} {H}\" so it "
    "rasterizes to the reference image's pixel dimensions ({W}×{H}).\n\n"
    "Wrap your SVG in a ```svg code fence. Keep the response concise: "
    "reasoning is optional; the SVG is what matters."
)

SYSTEM_PYTHON = (
    "You are an expert at producing Python code that generates SVG "
    "approximations of raster images. Given a reference image, define "
    "`def make_svg() -> str:` that returns an <svg> document as a "
    "Python string. The SVG uses only simple shapes (rect, circle, "
    "ellipse, polygon, line, path with M/L/H/V/Z). No text, no images, "
    "no gradients. Canvas: viewBox=\"0 0 {W} {H}\" matching the "
    "reference image's {W}×{H} pixel dimensions.\n\n"
    "Wrap your function in a ```python code fence. You may use any "
    "Python stdlib (math, random with a fixed seed, etc.) inside the "
    "function. No external imports beyond stdlib."
)


def first_user_turn(target_img: Image.Image) -> dict:
    return {"role": "user", "content": [
        {"type": "text",
         "text": "Approximate this image as SVG. Use only shape primitives "
                  "(no text, no raster embeds). Match the composition, "
                  "dominant colors, and spatial layout as closely as you can."},
        {"type": "image_url", "image_url": {"url": image_to_data_url(target_img)}},
    ]}


def feedback_user_turn(prev_rendered: Image.Image, mse: float,
                         heatmap: Image.Image | None = None) -> dict:
    text = (f"Your previous attempt scored MSE = {mse:.5f} "
            f"(lower is better, 0 = perfect pixel match). Here is a "
            f"rendering of the SVG your previous attempt produced, so "
            f"you can see where it diverges from the target. ")
    if heatmap is not None:
        text += "A red-ish/saturated pixel-difference heatmap also follows."
    content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_to_data_url(prev_rendered)}},
    ]
    if heatmap is not None:
        content += [
            {"type": "text", "text": "Pixel difference heatmap (amplified 3×):"},
            {"type": "image_url", "image_url": {"url": image_to_data_url(heatmap)}},
        ]
    content.append({
        "type": "text",
        "text": "Now produce a refined SVG. You may completely restructure "
                "the approximation if that's what the MSE suggests."
    })
    return {"role": "user", "content": content}


# ── Loop ─────────────────────────────────────────────────────────────

def run_loop(
    target_img: Image.Image, mode: str,
    max_iterations: int = 5, mse_target: float = 0.003,
    plateau_patience: int = 2, include_heatmap: bool = True,
    out_dir: pathlib.Path | None = None, seed_base: int = 0,
) -> dict:
    """Single rollout. plateau_patience=0 disables early stop (run all
    max_iterations turns regardless of MSE trajectory) — useful for
    long-tail statistical comparison where we want full N-turn
    trajectories from every rollout."""
    assert mode in ("svg", "python")
    out_dir = out_dir or RUNS / f"svg_refine_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_img.save(out_dir / "target.png")
    w, h = target_img.size

    system = SYSTEM_SVG.format(W=w, H=h) if mode == "svg" \
             else SYSTEM_PYTHON.format(W=w, H=h)
    messages: list[dict] = [
        {"role": "system", "content": system},
        first_user_turn(target_img),
    ]

    best = {"mse": float("inf"), "iter": -1, "svg": None}
    plateau = 0
    history = []

    for it in range(max_iterations):
        t0 = time.time()
        resp, req_metrics = chat_stream(
            messages, max_tokens=2048, temperature=1.0, seed=seed_base + it)
        rawpath = out_dir / f"iter_{it:02d}_raw.txt"
        rawpath.write_text(resp)

        # Extract the SVG per the chosen mode.
        if mode == "svg":
            svg_str = extract_svg_directly(resp)
            error_note = None if svg_str else "no <svg> block found in response"
        else:
            svg_str = extract_python_and_run(resp)
            error_note = None if svg_str else \
                ("python subprocess did not return an SVG (either no "
                 "```python``` block, syntax error, timeout, or non-string "
                 "return)")

        if svg_str is None:
            history.append({"iter": it, "error": error_note, "mse": None,
                             **req_metrics})
            print(f"  iter {it}: {error_note}")
            # Give feedback asking for valid output.
            messages.append({"role": "assistant", "content": resp})
            messages.append({"role": "user", "content":
                f"Your response could not be parsed: {error_note}. "
                f"Please emit a single {'SVG' if mode == 'svg' else 'Python function'} "
                f"in a fenced code block."})
            continue

        # Render the SVG.
        try:
            rendered = render_svg(svg_str, width=w, height=h)
        except Exception as e:
            history.append({"iter": it, "error": f"SVG render failed: {e}",
                             "svg": svg_str, "mse": None, **req_metrics})
            print(f"  iter {it}: SVG render failed ({e})")
            messages.append({"role": "assistant", "content": resp})
            messages.append({"role": "user", "content":
                f"Your SVG could not be rasterized: {e}. Check your syntax and try again."})
            continue

        rendered.save(out_dir / f"iter_{it:02d}_rendered.png")
        (out_dir / f"iter_{it:02d}.svg").write_text(svg_str)

        mse = mse_images(target_img, rendered)
        dt = time.time() - t0
        history.append({"iter": it, "mse": mse, "elapsed_s": dt,
                         "svg_chars": len(svg_str), **req_metrics})
        print(f"  iter {it}: MSE = {mse:.5f}   svg_chars={len(svg_str)}   "
              f"({dt:.1f}s)")

        # Track best.
        if mse < best["mse"]:
            plateau = 0
            best = {"mse": mse, "iter": it, "svg": svg_str}
        else:
            plateau += 1

        # Stopping criteria.
        if mse <= mse_target:
            print(f"  reached MSE target ({mse:.5f} ≤ {mse_target})")
            break
        if plateau_patience > 0 and plateau >= plateau_patience:
            print(f"  plateau: no improvement over {plateau} iterations, stopping")
            break
        if it == max_iterations - 1:
            break

        # Feedback turn.
        hm = diff_heatmap(target_img, rendered) if include_heatmap else None
        messages.append({"role": "assistant", "content": resp})
        messages.append(feedback_user_turn(rendered, mse, heatmap=hm))

    # Save best.
    if best["svg"]:
        best_rendered = render_svg(best["svg"], width=w, height=h)
        best_rendered.save(out_dir / "best_rendered.png")
        (out_dir / "best.svg").write_text(best["svg"])

    report = {
        "mode": mode, "width": w, "height": h,
        "max_iterations": max_iterations, "mse_target": mse_target,
        "best_iter": best["iter"], "best_mse": best["mse"],
        "history": history, "out_dir": str(out_dir),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nbest: iter {best['iter']}, MSE = {best['mse']:.5f}")
    print(f"outputs → {out_dir}")
    return report


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--target", type=pathlib.Path,
                     help="path to a target raster image")
    src.add_argument("--donut", type=int, default=None,
                     help="procedurally generate a donut with this seed")
    src.add_argument("--smoke", action="store_true",
                     help="use the built-in tiny synthesized test image")
    ap.add_argument("--mode", choices=["svg", "python"], default="svg")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--plateau-patience", type=int, default=2,
                    help="stop after this many non-improving iters; 0 disables")
    ap.add_argument("--mse-target", type=float, default=0.003)
    ap.add_argument("--no-heatmap", action="store_true")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    ap.add_argument("--seed-base", type=int, default=0,
                    help="seed = seed_base + iter; lets you compare arms with same RNG")
    args = ap.parse_args()

    if args.target:
        target = load_target_from_path(str(args.target))
    elif args.donut is not None:
        target = generate_donut_target(args.donut, args.size)
    else:
        target = synth_smoke_target(args.size)

    report = run_loop(target, args.mode,
                       max_iterations=args.max_iters,
                       mse_target=args.mse_target,
                       plateau_patience=args.plateau_patience,
                       include_heatmap=not args.no_heatmap,
                       out_dir=args.out_dir,
                       seed_base=args.seed_base)
    print(f"\n=== summary ===")
    print(f"mode={report['mode']} size={report['width']}x{report['height']}")
    print(f"best iter={report['best_iter']} MSE={report['best_mse']:.5f}")


if __name__ == "__main__":
    main()
