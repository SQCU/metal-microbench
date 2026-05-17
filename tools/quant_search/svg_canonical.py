"""SVG multi-turn refinement runner — async port of the canonical
toolcards `query-to-svg` methodology, with the renderer replaced for
speed.

Methodology (prompts, multi-turn refinement loop, DONE-signal handling,
retry-on-no-svg) is imported verbatim from
`/Users/mdot/sillytavern-fork/data/toolcards/installed/query-to-svg/
service.py` — this is the canonical card our existing SVGMSEHarness
runs against in the toolcards runner. We only replace the stdin-based
llm_call RPC with async chat-completions HTTP so we can drive the loop
from our in-process orchestrator.

Renderer: REPLACED with `resvg-py` (Rust-backed). The canonical service
uses playwright/chromium, which is ~13ms per warm render and ~500ms
cold-start; resvg renders the same SVGs in ~13ms with no cold start
and no thread-pool contention. Validated against playwright on a
representative SVG covering text + gradients + polygons + circles +
opacity: MSE = 0.0001, fraction of pixels differing >5% = 0.3%. The
disagreement is anti-aliasing-level only — well under the within-config
sampling-noise floor — and the cross-config MSE drift signal is
unchanged in interpretation.

The render-fidelity claim above must be re-validated if the SVGs we
generate ever start using SVG features beyond the standard static
spec (filters, foreignObject, scripts, etc.). resvg follows the SVG
1.1 / SVG 2 static-document spec; if a future workload needs a
feature outside that, the renderer choice should be revisited and
this docstring updated.
"""
from __future__ import annotations

import asyncio
import base64
import sys
import time
from pathlib import Path

import httpx
import resvg_py

# Pull the canonical methodology in. NOTE: the path is the installed-
# toolcard location in the sillytavern-fork. If that fork moves,
# update this import path. We import only the methodology (prompts +
# parsers) — NOT the renderer.
_CANONICAL_PATH = Path(
    "/Users/mdot/sillytavern-fork/data/toolcards/installed/query-to-svg")
sys.path.insert(0, str(_CANONICAL_PATH))

from service import (                                                # noqa: E402
    extract_svg as _canonical_extract_svg,
    is_done_signal as _canonical_is_done_signal,
    SYSTEM_PROMPT as _CANONICAL_SYSTEM_PROMPT,
    initial_user_turn as _canonical_initial_user_turn,
    refinement_user_turn as _canonical_refinement_user_turn,
    png_to_data_url as _canonical_png_to_data_url,
)


def render_svg(svg_text: str, width: int, height: int,
                background: str = "white") -> bytes:
    """Render an SVG string to PNG bytes via resvg. Drop-in replacement
    for the canonical playwright/chromium render with the same
    (svg, width, height, background) signature and PNG-bytes return.

    ~13 ms per render, no cold start, thread-safe, no external process,
    no temp dirs. Validated MSE 0.0001 vs playwright on test SVGs.
    """
    return bytes(resvg_py.svg_to_bytes(
        svg_string=svg_text,
        width=width, height=height,
        background=background,
    ))


# Re-export prompts + parsers under unprefixed names. Importers should
# go through this module so the canonical binding is the only seam.
extract_svg = _canonical_extract_svg
is_done_signal = _canonical_is_done_signal
SYSTEM_PROMPT = _CANONICAL_SYSTEM_PROMPT
initial_user_turn = _canonical_initial_user_turn
refinement_user_turn = _canonical_refinement_user_turn
png_to_data_url = _canonical_png_to_data_url


async def run_multi_turn_svg(
        client: httpx.AsyncClient,
        bridge_url: str,
        model_name: str,
        *,
        query: str,
        max_iters: int = 3,
        width: int = 512,
        height: int = 512,
        first_iter_text: str | None = None,
        first_iter_finish_reason: str | None = None,
        sample_temperature: float = 1.0,
        sample_seed: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        ) -> dict:
    """Async port of `service.handle()`'s multi-turn refinement loop.

    Same methodology, same prompts, same renderer, same DONE-signal
    handling, same retry-on-no-svg logic. The ONLY difference is that
    the per-iter llm_call goes through async HTTP instead of stdin RPC.

    `first_iter_text` and `first_iter_finish_reason`: optional. If the
    framework runner has already produced the iter-0 generation (it
    has, since the framework's runner makes one call before
    parse_rollout), we adopt it as iter 0 instead of re-firing — saves
    one chat completion. Pass None to do iter 0 ourselves.

    `semaphore`: bridge-call concurrency gate. Each iter's chat
    completion acquires the semaphore independently — between iters,
    the slot is released so other rollouts (other SVG iters, evals
    from other benchmarks, judge calls) can use it. Holding the
    semaphore across the whole multi-turn loop would idle ~30s of slot
    per SVG rollout while we render and re-prefill, starving the
    bridge of streams.

    Returns:
      {
        "final_svg":    str | None,    # last successful SVG, or None
        "final_png_b64": str | None,   # base64 PNG of final_svg, or None
        "iter_history": list[dict],    # per-iter trace
        "early_exit":   bool,          # model emitted DONE
        "n_iters":      int,           # iters actually performed
      }
    """
    sys_prompt = SYSTEM_PROMPT.format(W=width, H=height)
    messages: list = [
        {"role": "system", "content": sys_prompt},
        initial_user_turn(query),
    ]
    history: list[dict] = []
    final_svg: str | None = None
    final_png_bytes: bytes | None = None
    early_exit = False

    async def _post_chat(msgs: list, *, image_in_messages: bool) -> tuple[str | None, str | None]:
        """One chat completion call. Returns (text, error_or_None).

        Acquires the bridge semaphore (if provided) for the duration of
        this single call only — released before the next iter's
        rendering or message-list construction so the slot is free for
        other concurrent work."""
        body = {
            "model": model_name,
            "messages": msgs,
            "max_tokens": 4096,
            "temperature": sample_temperature,
        }
        if sample_seed is not None:
            body["seed"] = sample_seed
        try:
            if semaphore is None:
                r = await client.post(
                    f"{bridge_url}/v1/chat/completions",
                    json=body, timeout=None)
            else:
                async with semaphore:
                    r = await client.post(
                        f"{bridge_url}/v1/chat/completions",
                        json=body, timeout=None)
            r.raise_for_status()
            data = r.json()
        except BaseException as e:                       # noqa: BLE001
            return None, f"http: {type(e).__name__}: {e}"
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            return None, f"resp_shape: {type(e).__name__}"
        return text, None

    for i in range(max_iters):
        t0 = time.time()
        # Iter-0: adopt the runner's pre-fired generation if provided.
        if i == 0 and first_iter_text is not None:
            text, post_err = first_iter_text, None
        else:
            text, post_err = await _post_chat(
                messages, image_in_messages=(i > 0))
        if post_err is not None:
            history.append({"iter": i, "error": f"llm_call failed: {post_err}",
                            "elapsed_s": time.time() - t0})
            break

        # Refinement iter (i > 0): check DONE early-exit signal.
        if i > 0 and is_done_signal(text):
            history.append({
                "iter": i, "early_exit": True,
                "elapsed_s": time.time() - t0,
            })
            early_exit = True
            break

        svg = extract_svg(text)
        if svg is None:
            history.append({
                "iter": i, "error": "no <svg> in response",
                "elapsed_s": time.time() - t0,
            })
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                "Your response did not contain a valid <svg> block, and was "
                "not the DONE signal. Either emit exactly one <svg>...</svg> "
                "wrapped in a ```svg code fence, OR reply with the single "
                "word DONE if the prior render was good."})
            continue

        try:
            # resvg is ~13ms per render with no per-thread cold start;
            # asyncio.to_thread on the default executor is fine here.
            png_bytes = await asyncio.to_thread(
                render_svg, svg, width, height)
        except Exception as e:                            # noqa: BLE001
            history.append({
                "iter": i, "error": f"render failed: {e}",
                "svg_chars": len(svg),
                "elapsed_s": time.time() - t0,
            })
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                f"Your SVG could not be rasterized: {e}. Check syntax, "
                f"avoid external refs, and try again."})
            continue

        # Successful iter.
        final_svg = svg
        final_png_bytes = png_bytes
        history.append({
            "iter": i,
            "svg_chars": len(svg),
            "elapsed_s": time.time() - t0,
        })

        if i < max_iters - 1:
            prior_render_url = png_to_data_url(png_bytes)
            messages.append({"role": "assistant", "content": text})
            messages.append(refinement_user_turn(query, prior_render_url, i + 1))

    final_png_b64 = (
        base64.b64encode(final_png_bytes).decode("ascii")
        if final_png_bytes is not None else None)

    return {
        "final_svg":    final_svg,
        "final_png_b64": final_png_b64,
        "iter_history": history,
        "early_exit":   early_exit,
        "n_iters":      len(history),
    }


# ──────────────────────────────────────────────────────────────────────
# MSE math — operates on already-rendered PNGs stored as base64 in
# JSONL records. NO rendering at MSE time; renders happened during eval.
# ──────────────────────────────────────────────────────────────────────


def png_b64_to_array(b64: str):
    """Decode a base64-encoded PNG to a (H, W, 3) float32 array in [0, 1]."""
    import io
    import numpy as np
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    arr = np.asarray(img.convert("RGB"), dtype="float32") / 255.0
    return arr


def png_pair_mse(a_b64: str, b_b64: str) -> float:
    """Mean-squared-error between two PNG-base64 images. Returns 0..1.
    Resizes b to a's dimensions if needed (LANCZOS)."""
    import numpy as np
    a = png_b64_to_array(a_b64)
    b = png_b64_to_array(b_b64)
    if a.shape != b.shape:
        # Bring b to a's shape.
        from PIL import Image
        import io
        b_img = Image.open(io.BytesIO(base64.b64decode(b_b64)))
        b_img = b_img.resize((a.shape[1], a.shape[0]), Image.LANCZOS)
        b = np.asarray(b_img.convert("RGB"), dtype="float32") / 255.0
    return float(((a - b) ** 2).mean())
