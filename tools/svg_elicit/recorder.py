#!/usr/bin/env python3
"""Transcript + artifact recorder for the SVG elicitation/judge harnesses.

Writes one JSONL line per LM call mapping the transcript to EXACT token in/out
indices (including image soft-token spans) and to its rendered/binary-SVG +
parsed-code artifacts — the recording convention the oldest video trials
(svg_concurrent_bench curriculum: per-iter raw/svg/png) get, extended to exact
token-index fidelity.

Token fidelity comes from the bridge itself: requests carry
`return_token_ids: true`, and the non-streaming response returns
  choices[0].token_ids            — raw emitted completion token ids (EXACT)
  choices[0].thinking_token_ids   — raw thinking token ids (if any)
  choices[0].prompt_token_layout  — per-segment text token ids (BOS/turn markers
                                    included; they come from the chat template)
                                    + image segment positions + the exact
                                    image-soft-token total.
(see server/bridge.py return_token_ids handling). So we DON'T re-tokenize or
scrape the bridge log — the engine hands us the assembled token stream.

Image soft-tokens are continuous vision embeddings: they occupy exact positions
(recorded as kind="image_soft" spans with [start,end)) but have no discrete ids.
Per-image count = image_soft_tokens_total / n_images (exact when images share a
size, which they do within a run).

Artifacts (sha-addressed, deduped) under <run>/artifacts/: target/candidate png,
and — when the candidate came from elicit.py — its binary .svg source and parsed
code, linked by relative path.
"""
from __future__ import annotations
import hashlib, io, json, pathlib, time
from PIL import Image


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


class Recorder:
    def __init__(self, run_dir: pathlib.Path):
        self.run_dir = run_dir
        self.art = run_dir / "artifacts"
        self.art.mkdir(parents=True, exist_ok=True)
        self.jsonl = run_dir / "transcripts.jsonl"
        self._seen: set[str] = set()

    def stash_image(self, img: Image.Image, *, source_svg: str | None = None,
                    parsed_code: str | None = None, src_path: str | None = None) -> dict:
        """Save image (+ optional binary svg / parsed code) addressed by sha."""
        b = png_bytes(img)
        h = sha16(b)
        meta = {"image_sha": h, "w": img.size[0], "h": img.size[1], "png": f"artifacts/{h}.png"}
        if h not in self._seen:
            (self.art / f"{h}.png").write_bytes(b)
            if source_svg is not None:
                (self.art / f"{h}.svg").write_text(source_svg)
            if parsed_code is not None:
                (self.art / f"{h}.code.py").write_text(parsed_code)
            self._seen.add(h)
        if (self.art / f"{h}.svg").exists():
            meta["svg"] = f"artifacts/{h}.svg"
        if (self.art / f"{h}.code.py").exists():
            meta["parsed_code"] = f"artifacts/{h}.code.py"
        if src_path:
            meta["origin"] = src_path
        return meta

    @staticmethod
    def _input_spans(layout: dict, image_metas_ordered: list[dict]) -> dict:
        """Build the exact token-in index map from the bridge's prompt_token_layout.
        Text segments carry real token ids (incl. BOS/turn markers); image segments
        become [start,end) soft-token spans tagged with the image sha."""
        spans, idx, img_i = [], 0, 0
        n_img = max(layout.get("n_images", 0), 0)
        soft_total = layout.get("image_soft_tokens_total", 0) or 0
        per_img = soft_total // n_img if n_img else 0
        rem = soft_total - per_img * n_img if n_img else 0
        for seg in layout.get("segments", []):
            if seg.get("kind") == "text":
                ids = seg.get("token_ids", [])
                spans.append({"kind": "text", "n": len(ids), "start": idx,
                              "end": idx + len(ids), "token_ids": ids})
                idx += len(ids)
            else:
                n = per_img + (rem if img_i == n_img - 1 else 0)  # absorb remainder in last
                meta = image_metas_ordered[img_i] if img_i < len(image_metas_ordered) else {}
                spans.append({"kind": "image_soft", "n": n, "start": idx, "end": idx + n,
                              "image_sha": meta.get("image_sha"),
                              "note": "continuous vision soft-tokens; positions exact, no discrete ids"})
                idx += n
                img_i += 1
        return {"spans": spans, "n_tokens_assembled": idx,
                "prompt_tokens_total": layout.get("prompt_tokens_total"),
                "image_soft_tokens_total": soft_total}

    def record(self, *, call_id: str, variant: str, messages: list[dict],
               image_metas_ordered: list[dict], response: dict,
               timing: dict, extra: dict) -> dict:
        choice = (response.get("choices") or [{}])[0]
        layout = choice.get("prompt_token_layout", {}) or {}
        out_ids = choice.get("token_ids", []) or []
        rec = {
            "call_id": call_id, "variant": variant, "ts": time.time(),
            "input": {
                # strings in (compact): role + ordered part kinds + the text itself
                "messages": [{"role": m["role"], "parts": [
                    ({"text": p["text"]} if p.get("type") == "text"
                     else {"image_sha": None})
                    for p in (m["content"] if isinstance(m["content"], list)
                              else [{"type": "text", "text": m["content"]}])]}
                    for m in messages],
                "token_layout": self._input_spans(layout, image_metas_ordered),
                "images": image_metas_ordered,
            },
            "output": {
                "text": choice.get("message", {}).get("content", ""),  # strings out
                "token_ids": out_ids, "n": len(out_ids),               # tokens out (exact)
                "thinking_token_ids": choice.get("thinking_token_ids"),
                "finish_reason": choice.get("finish_reason"),
            },
            "usage": response.get("usage", {}),
            "timing": timing,
            **extra,
        }
        with self.jsonl.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec
