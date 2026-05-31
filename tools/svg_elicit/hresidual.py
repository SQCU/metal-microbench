#!/usr/bin/env python3
"""Residual + metrics — the SINGLE source for the k-shot harness.

The task is RESIDUAL-driven: we measure the difference between the render and the
reference and drive it down turn-over-turn. So this module makes the residual a
first-class, dumpable object:
  - false_color_residual(ref, render)  : where the render is WRONG vs the reference
  - false_color_delta(prev, render)    : where the model actually CHANGED this turn
  - mse / ssim                         : scalar residual + structural similarity
A nil delta image (delta nearly black everywhere) is the visual tell that the
harness told the model the wrong thing — exactly what we watch for.

Supersedes the scattered copies (svg_refinement_loop.mse_images/diff_heatmap,
elicit.ssim_score, judge_calibrate.ssim_score).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as _ssim


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def mse(ref: Image.Image, img: Image.Image) -> float:
    """Mean squared error over RGB in [0,1]. The residual scalar to drive down."""
    a, b = _arr(ref) / 255.0, _arr(img.resize(ref.size)) / 255.0
    return float(np.mean((a - b) ** 2))


def ssim(ref: Image.Image, img: Image.Image) -> float:
    """Structural similarity in [-1,1] (1 = identical)."""
    a, b = _arr(ref).astype(np.uint8), _arr(img.resize(ref.size)).astype(np.uint8)
    return float(_ssim(a, b, channel_axis=2, data_range=255))


def _magnitude(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel L2 colour distance in [0,1]."""
    return np.sqrt(np.mean((a - b) ** 2, axis=2)) / 255.0


def _hot(mag: np.ndarray) -> Image.Image:
    """Hot colormap: 0=black -> red -> yellow -> white=1. Bright = large error."""
    m = np.clip(mag, 0.0, 1.0)
    r = np.clip(m * 3.0, 0, 1)
    g = np.clip(m * 3.0 - 1.0, 0, 1)
    b = np.clip(m * 3.0 - 2.0, 0, 1)
    rgb = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def false_color_residual(ref: Image.Image, img: Image.Image) -> Image.Image:
    """Where the render is WRONG vs the reference (bright = big error)."""
    a, b = _arr(ref), _arr(img.resize(ref.size))
    return _hot(_magnitude(a, b))


def false_color_delta(prev: Image.Image | None, img: Image.Image) -> Image.Image:
    """Where the render CHANGED since the previous turn (bright = changed a lot).
    All-black => the model produced a near-identical image => nil-change tell."""
    if prev is None:
        return Image.new("RGB", img.size, (0, 0, 0))
    a, b = _arr(prev.resize(img.size)), _arr(img)
    return _hot(_magnitude(a, b))


def residual_stats(ref: Image.Image, img: Image.Image,
                   prev: Image.Image | None = None) -> dict:
    """Scalars for the trajectory: mse/ssim vs reference, and mse vs the previous
    render (the 'did anything actually change' signal — should be LARGE, not ~0)."""
    out = {"mse": round(mse(ref, img), 6), "ssim": round(ssim(ref, img), 4)}
    if prev is not None:
        a, b = _arr(prev.resize(img.size)) / 255.0, _arr(img) / 255.0
        out["mse_vs_prev"] = round(float(np.mean((a - b) ** 2)), 6)
    return out
