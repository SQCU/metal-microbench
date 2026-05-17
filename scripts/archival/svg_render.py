#!/usr/bin/env python3
"""SVG → PIL rasterizer backed by headless Chromium (Playwright).

The rendering environment is the full SVG spec as Chrome/Safari render it:
text with system fonts, <image>, gradients, patterns, filters, CSS,
foreignObject — anything a browser supports the model can use, and we
rasterize it faithfully. Earlier iterations of this module restricted
SVG to a hand-curated shape subset; that was an editorialization that
penalized models for emitting valid SVG (notably <text>, which silently
disappeared with resvg's skip_system_fonts and was explicitly skipped
by the PIL fallback).

Threading: the eval uses a ThreadPoolExecutor; Playwright's sync API
is thread-confined, so each calling thread gets its own playwright
loop + Chromium instance via threading.local. Memory cost is a few
hundred MB per thread; with 8 rollouts that's ~1.5 GB total, which
is fine for an M5 Max.

Usage:
    from svg_render import render_svg
    img = render_svg(svg_text, width=256, height=256)
    img.save("out.png")
"""
from __future__ import annotations

import atexit
import io
import threading

from PIL import Image


_PW_LOCAL = threading.local()


def _get_page():
    """Lazy thread-local Playwright + Chromium + Page. Reused across
    render calls within the same thread; new threads each spin up
    their own browser."""
    if not hasattr(_PW_LOCAL, "page"):
        from playwright.sync_api import sync_playwright
        _PW_LOCAL.pw = sync_playwright().start()
        _PW_LOCAL.browser = _PW_LOCAL.pw.chromium.launch()
        _PW_LOCAL.context = _PW_LOCAL.browser.new_context(
            device_scale_factor=1)
        _PW_LOCAL.page = _PW_LOCAL.context.new_page()
    return _PW_LOCAL.page


def _close_thread_browser():
    for attr in ("page", "context", "browser"):
        obj = getattr(_PW_LOCAL, attr, None)
        if obj is not None:
            try: obj.close()
            except Exception: pass
    pw = getattr(_PW_LOCAL, "pw", None)
    if pw is not None:
        try: pw.stop()
        except Exception: pass
    for attr in ("page", "context", "browser", "pw"):
        if hasattr(_PW_LOCAL, attr):
            delattr(_PW_LOCAL, attr)


atexit.register(_close_thread_browser)


def render_svg(svg_text: str, width: int = 256, height: int = 256,
                background: str = "white") -> Image.Image:
    """Rasterize `svg_text` to a (width, height) RGB PIL image using
    headless Chromium. Background fills any area the SVG doesn't paint
    (default white). The SVG is forced to fill the viewport via CSS;
    its own viewBox controls the internal coordinate scaling."""
    bg_css = background if background else "transparent"
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        f'html,body{{margin:0;padding:0;background:{bg_css};}}'
        f'svg{{display:block;width:{width}px;height:{height}px;}}'
        '</style></head><body>'
        f'{svg_text}'
        '</body></html>'
    )
    page = _get_page()
    page.set_viewport_size({"width": width, "height": height})
    page.set_content(html)
    png_bytes = page.screenshot(type="png", omit_background=False,
                                  full_page=False,
                                  clip={"x": 0, "y": 0,
                                         "width": width, "height": height})
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib
    cases = [
        ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
         '<rect x="10" y="10" width="80" height="80" fill="blue"/></svg>',
         "rect.png"),
        ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
         '<text x="50" y="55" text-anchor="middle" font-size="40" fill="red">Hi</text></svg>',
         "text.png"),
        ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
         '<defs><linearGradient id="g"><stop offset="0%" stop-color="red"/>'
         '<stop offset="100%" stop-color="blue"/></linearGradient></defs>'
         '<rect width="100" height="100" fill="url(#g)"/></svg>',
         "gradient.png"),
    ]
    outdir = pathlib.Path("/tmp/svg_test"); outdir.mkdir(exist_ok=True)
    for svg, name in cases:
        img = render_svg(svg, width=128, height=128)
        img.save(outdir / name)
        print(f"  → {outdir / name}  size={img.size}")
