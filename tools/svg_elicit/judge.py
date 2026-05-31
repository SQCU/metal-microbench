#!/usr/bin/env python3
"""Calibrated image VLM-judge: anchored rubric + on-policy k-shot exemplars.

The default judge in elicit.py floored to 1/5 (no scale to anchor to). The
judge_calibrate.py sweep showed: an anchored rubric removes floor-bias, and ~3
on-policy SVG-render exemplars that dissociate SSIM/MSE/semantic push rank-corr
vs ground truth to ~0.87. This module packages that calibrated judge so the
multi-turn elicitation loop can use it for BOTH in-loop feedback and best-of-N
selection. See docs/causal_ophthalmology_harness_design.md §8-9.
"""
from __future__ import annotations
import base64, io, json, pathlib, re
from PIL import Image

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

RUBRIC = ("Score how faithfully the CANDIDATE reproduces the TARGET's subject, "
          "pose, composition, color, and any text, on a 1-5 scale:\n"
          "  5 = near-identical; only minor differences\n"
          "  4 = clearly the same subject/scene; small style or color errors\n"
          "  3 = subject/pose recognizable but notable color/style/detail loss\n"
          "  2 = some shared elements but the subject is hard to recognize\n"
          "  1 = unrelated, blank, or unrecognizable\n"
          "Most reproductions fall in 2-4. Reserve 1 ONLY for unrelated/blank. "
          "Judge SUBJECT RECOGNIZABILITY, not pixel-exactness — a faithful "
          "drawing in the wrong color is still a 3-4, not a 1.")
SYS_RUBRIC = "You are a meticulous, calibrated visual critic. " + RUBRIC
ASK = ('Respond with ONLY JSON: {"score": <int 1-5>, '
       '"missing": ["<qualities in TARGET absent/wrong in CANDIDATE>"], '
       '"biggest_fix": "<the single most important change>"}.')


def _durl(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_exemplars(repo: pathlib.Path, json_path: pathlib.Path) -> list[dict]:
    """Load on-policy exemplars (target/candidate image paths + score + reason)
    into judge-ready data-urls. Returns [] if the path is missing."""
    if not json_path or not pathlib.Path(json_path).exists():
        return []
    out = []
    for e in json.loads(pathlib.Path(json_path).read_text()):
        tgt = Image.open(repo / e["target"]).convert("RGB")
        cand = Image.open(repo / e["candidate"]).convert("RGB")
        out.append({"target_url": _durl(tgt), "cand_url": _durl(cand),
                    "score": e["score"], "reason": e["reason"]})
    return out


def _exemplar_block(exemplars: list[dict]) -> list[dict]:
    content = []
    for i, ex in enumerate(exemplars):
        content += [
            {"type": "text", "text": f"--- Example {i+1} ---\nTARGET:"},
            {"type": "image_url", "image_url": {"url": ex["target_url"]}},
            {"type": "text", "text": "CANDIDATE:"},
            {"type": "image_url", "image_url": {"url": ex["cand_url"]}},
            {"type": "text", "text": "Correct answer: "
                + json.dumps({"score": ex["score"], "reason": ex["reason"]})},
        ]
    return content


def build_messages(exemplars: list[dict], target_img: Image.Image,
                   cand_img: Image.Image) -> list[dict]:
    uc = []
    if exemplars:
        uc.append({"type": "text", "text":
                   "Here are calibrated scoring examples. Study the scale, then "
                   "score the final pair the same way."})
        uc += _exemplar_block(exemplars)
        uc.append({"type": "text", "text": "--- Now score THIS pair ---"})
    uc += [
        {"type": "text", "text": "TARGET:"},
        {"type": "image_url", "image_url": {"url": _durl(target_img)}},
        {"type": "text", "text": "CANDIDATE:"},
        {"type": "image_url", "image_url": {"url": _durl(cand_img)}},
        {"type": "text", "text": ASK},
    ]
    return [{"role": "system", "content": SYS_RUBRIC}, {"role": "user", "content": uc}]


# ---------------------------------------------------------------------------
# Feature-Likert profile judge. A single 1-5 "faithfulness" scalar can't
# resolve semantic differences (a 0.5 mean delta is below its resolution and
# it conflates many axes). Instead: characterize the TARGET and the CANDIDATE
# *independently* on the same k monotonic 1-5 feature scales, then compare the
# two profiles. This (a) gives k× the resolution (mean of k deltas is near-
# continuous), (b) is less circular — the judge DESCRIBES each image rather
# than scoring success — and (c) decomposes "did the meaning survive" into
# interpretable per-feature gaps. Plus one comparative subject_match for raw
# recognizability. Designed for complex video frames, not the amongus canary.
FEATURES = {
    "detail":        "fine visual detail / texture density (1=flat or minimal, 5=intricate)",
    "color_variety": "spread of distinct hues present (1=near-monochrome, 5=many colours)",
    "saturation":    "colour vividness (1=muted/greyish, 5=vivid)",
    "brightness":    "overall luminance (1=dark, 5=bright)",
    "complexity":    "compositional busyness / number of distinct regions (1=simple, 5=busy)",
    "contrast":      "tonal contrast (1=flat, 5=high)",
    "depth":         "sense of foreground/background layering or 3-D structure (1=flat, 5=deep)",
}
FEATURE_SYS = ("You are a meticulous visual analyst. You will see a TARGET image and a CANDIDATE "
               "image. Characterize EACH image independently on a set of 1-5 feature scales — rate "
               "what each image ACTUALLY shows, do not reward or penalize the candidate for being a "
               "reconstruction. The two profiles will be compared feature-by-feature.")


def _feature_ask() -> str:
    lines = "\n".join(f"  {k}: {v}" for k, v in FEATURES.items())
    return ("Rate the TARGET and the CANDIDATE each, independently, on these 1-5 scales:\n"
            f"{lines}\n"
            "Then rate subject_match: does the CANDIDATE depict the same subject/scene/content as "
            "the TARGET? (1=unrelated, 3=partially recognizable, 5=clearly the same).\n"
            'Respond with ONLY JSON: {"target": {<feature>: <int 1-5>, ...}, '
            '"candidate": {<feature>: <int 1-5>, ...}, "subject_match": <int 1-5>}.')


def feature_score(chat, target_img: Image.Image, cand_img: Image.Image) -> dict | None:
    """`chat(messages) -> text`. Returns a feature-profile comparison:
    {target_profile, candidate_profile, feature_deltas, profile_mean_delta,
     subject_match, semantic_distance(0=identical..1=worst)} or None."""
    try:
        msgs = [{"role": "system", "content": FEATURE_SYS},
                {"role": "user", "content": [
                    {"type": "text", "text": "TARGET:"},
                    {"type": "image_url", "image_url": {"url": _durl(target_img)}},
                    {"type": "text", "text": "CANDIDATE:"},
                    {"type": "image_url", "image_url": {"url": _durl(cand_img)}},
                    {"type": "text", "text": _feature_ask()}]}]
        m = _JSON_RE.search(chat(msgs))
        if not m:
            return None
        obj = json.loads(m.group(0))
        tp, cp = obj.get("target", {}), obj.get("candidate", {})
        deltas = {f: abs(float(tp[f]) - float(cp[f]))
                  for f in FEATURES
                  if isinstance(tp.get(f), (int, float)) and isinstance(cp.get(f), (int, float))}
        prof = round(sum(deltas.values()) / len(deltas), 3) if deltas else None
        sm = obj.get("subject_match")
        sm = float(sm) if isinstance(sm, (int, float)) else None
        sem = (round(((prof / 4.0) + (1 - sm / 5.0)) / 2.0, 3)
               if (prof is not None and sm is not None) else None)
        return {"target_profile": tp, "candidate_profile": cp, "feature_deltas": deltas,
                "profile_mean_delta": prof, "subject_match": sm, "semantic_distance": sem}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Joint correspondence judge. A SINGLE comparative call with BOTH images in
# context (TARGET first, then CANDIDATE) that asks "how well does the SECOND
# image reproduce the FIRST" on exactly three GENERAL axes — no named
# structures, no domain content, no hand-coded taxonomy. The three scalars
# feed back to the model in-loop as inter-turn signal alongside the residual.
#   composition   = spatial-layout correspondence (where things sit)
#   forms         = are the first image's distinct shapes/objects present & placed
#   color_texture = do the first image's colour / texture regions correspond
CORR_AXES = {
    "composition":   "spatial-layout correspondence — is the overall arrangement and "
                     "placement of regions in the SECOND image like the FIRST's",
    "forms":         "are the FIRST image's distinct shapes / objects present in the "
                     "SECOND and placed where they belong",
    "color_texture": "do the colour and texture regions of the SECOND image correspond "
                     "to those of the FIRST",
}
CORR_SYS = ("You are a meticulous, calibrated visual comparator. You will see a FIRST image and a "
            "SECOND image. For each axis, judge HOW WELL THE SECOND IMAGE REPRODUCES THE FIRST — a "
            "single comparative judgment per axis, not two independent descriptions. Use the full "
            "1-5 range: 1 = no correspondence, 3 = partial / approximate, 5 = strong correspondence.")


def _corr_ask() -> str:
    lines = "\n".join(f"  {k}: {v}" for k, v in CORR_AXES.items())
    return ("Rate, on a 1-5 scale, how well the SECOND image reproduces the FIRST on each axis "
            "(higher = closer correspondence):\n"
            f"{lines}\n"
            'Respond with ONLY JSON: {"composition": <int 1-5>, "forms": <int 1-5>, '
            '"color_texture": <int 1-5>}.')


def correspondence(chat, target_img: Image.Image, cand_img: Image.Image) -> dict | None:
    """`chat(messages) -> text`. ONE joint/comparative call with BOTH images in
    context (TARGET then CANDIDATE). Returns {composition:int, forms:int,
    color_texture:int} (1-5 each, each a COMPARATIVE judgment of how well the
    candidate reproduces the target on that GENERAL axis) or None on failure."""
    try:
        msgs = [{"role": "system", "content": CORR_SYS},
                {"role": "user", "content": [
                    {"type": "text", "text": "FIRST image:"},
                    {"type": "image_url", "image_url": {"url": _durl(target_img)}},
                    {"type": "text", "text": "SECOND image:"},
                    {"type": "image_url", "image_url": {"url": _durl(cand_img)}},
                    {"type": "text", "text": _corr_ask()}]}]
        m = _JSON_RE.search(chat(msgs))
        if not m:
            return None
        obj = json.loads(m.group(0))
        out = {}
        for ax in CORR_AXES:
            v = obj.get(ax)
            if isinstance(v, (int, float)):
                out[ax] = int(round(float(v)))
        return out or None
    except Exception:
        return None


def score(chat, target_img: Image.Image, cand_img: Image.Image,
          exemplars: list[dict]) -> dict | None:
    """`chat(messages) -> text`. Returns {faithfulness:1-5, missing:[...],
    biggest_fix:str} or None on parse failure."""
    try:
        txt = chat(build_messages(exemplars, target_img, cand_img))
        m = _JSON_RE.search(txt)
        if not m:
            return None
        obj = json.loads(m.group(0))
        s = obj.get("score")
        return {"faithfulness": float(s) if isinstance(s, (int, float)) else None,
                "missing": obj.get("missing") or [],
                "biggest_fix": obj.get("biggest_fix")}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Residual feature vector — a >6-dim comparative judge for the k-shot harness.
# A 3-scalar judge can't tell the model WHAT to fix; this decomposes "how well
# does the SECOND image reproduce the FIRST" into 9 named axes, each 1-5. It is
# BOTH elicitation (the model is told to raise its lowest axes) AND scoring (we
# track all 9 + their mean over turns). One joint/comparative call, both images
# in context (FIRST = reference, SECOND = current render).
RESIDUAL_AXES = {
    "layout":          "overall spatial arrangement of regions matches",
    "large_forms":     "the major shapes / objects are present and placed correctly",
    "fine_detail":     "small details and texture are captured",
    "color_palette":   "the set of colours present matches",
    "color_placement": "colours are in the right places",
    "edges_contours":  "shape boundaries / edges line up",
    "proportions":     "relative sizes and positions are correct",
    "text":            "any legible text is reproduced (presence and accuracy)",
    "completeness":    "nothing major is missing or spuriously added",
}
RESIDUAL_SYS = (
    "You are a meticulous visual comparator. You see a FIRST image (the reference) and a "
    "SECOND image (a reconstruction). For each axis rate, 1-5, HOW WELL THE SECOND REPRODUCES "
    "THE FIRST on that axis — a single comparative judgment. 1 = absent/wrong, 3 = partial, "
    "5 = strong match. Use the full range; be strict so improvement has room to show.")


def _residual_ask() -> str:
    lines = "\n".join(f"  {k}: {v}" for k, v in RESIDUAL_AXES.items())
    keys = ", ".join(f'"{k}": <int 1-5>' for k in RESIDUAL_AXES)
    return ("Rate how well the SECOND image reproduces the FIRST on each axis (1-5):\n"
            f"{lines}\n"
            f'Respond with ONLY JSON: {{{keys}, "worst": "<the single axis most worth fixing next>"}}.')


def feature_residual(chat, ref_img: Image.Image, render_img: Image.Image) -> dict | None:
    """`chat(messages) -> text`. ONE comparative call, both images in context
    (FIRST = reference, SECOND = render). Returns {<9 axes>: int 1-5, worst: str,
    mean: float} or None on failure."""
    try:
        msgs = [{"role": "system", "content": RESIDUAL_SYS},
                {"role": "user", "content": [
                    {"type": "text", "text": "FIRST image (reference):"},
                    {"type": "image_url", "image_url": {"url": _durl(ref_img)}},
                    {"type": "text", "text": "SECOND image (current reconstruction):"},
                    {"type": "image_url", "image_url": {"url": _durl(render_img)}},
                    {"type": "text", "text": _residual_ask()}]}]
        m = _JSON_RE.search(chat(msgs))
        if not m:
            return None
        obj = json.loads(m.group(0))
        out = {}
        for ax in RESIDUAL_AXES:
            v = obj.get(ax)
            if isinstance(v, (int, float)):
                out[ax] = int(round(float(v)))
        if len(out) < 6:                       # require a usable >6-dim vector
            return None
        out["mean"] = round(sum(out.values()) / len(out), 3)
        if isinstance(obj.get("worst"), str):
            out["worst"] = obj["worst"]
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Checklist judge — the FIX for judge collapse. The absolute "how faithful is
# this SVG to a pixel-perfect reference" task floors every reconstruction at the
# same low score (the reference is unreachable by SVG, so 0.62 vs 0.64 faithful
# is below the model's discrimination). Instead: enumerate the reference's
# salient elements ONCE (a fixed per-image checklist), then each turn check how
# many of THOSE elements the render contains. As the writer composes more detail,
# the present-count RISES — a sensitive, non-collapsing signal that directly
# measures "is more of the reference's content reconstructed".
_CHECKLIST_SYS = (
    "You are itemising the visible content of an image so a reconstruction can be checked "
    "against it. List the SALIENT, DISTINCT, individually-checkable elements — objects, "
    "regions, text strings, distinctive colours/patterns — that a faithful reconstruction "
    "must contain. Be concrete and visual (e.g. \"a man's face, upper-right\", \"green panel, "
    "left third\", \"the text '麻雀'\"), not abstract.")


def salient_checklist(chat, ref_img: Image.Image, n: int = 10) -> list[str] | None:
    """`chat(messages) -> text`. Enumerate ~n salient, checkable elements of the
    reference (computed ONCE per image). Returns a list of short element phrases."""
    try:
        msgs = [{"role": "system", "content": _CHECKLIST_SYS},
                {"role": "user", "content": [
                    {"type": "text", "text": "Image:"},
                    {"type": "image_url", "image_url": {"url": _durl(ref_img)}},
                    {"type": "text", "text":
                        f"List the {n} most salient distinct elements this image contains. "
                        f'Respond with ONLY JSON: {{"elements": ["<short phrase>", ...]}}.'}]}]
        m = _JSON_RE.search(chat(msgs))
        if not m:
            return None
        els = json.loads(m.group(0)).get("elements")
        els = [str(e).strip() for e in els if str(e).strip()] if isinstance(els, list) else None
        return els[:n] if els else None
    except Exception:
        return None


_MATCH_SYS = (
    "You check a RECONSTRUCTION against a REFERENCE using a fixed checklist of the reference's "
    "elements. For each checklist item, judge whether the reconstruction contains it: "
    "0 = absent, 1 = attempted but partial / wrong colour / misplaced, 2 = clearly present and "
    "roughly correct. Judge only presence/correspondence of that element, not overall quality.")


def checklist_match(chat, ref_img: Image.Image, render_img: Image.Image,
                    checklist: list[str]) -> dict | None:
    """`chat(messages) -> text`. Score each fixed checklist element 0/1/2 in the
    render. Returns {scores: {item: 0-2}, present_count(weighted sum), total(2*len),
    fraction}. Rises as the writer composes more of the reference's content."""
    try:
        items = "\n".join(f"  {i}: {e}" for i, e in enumerate(checklist))
        keys = ", ".join(f'"{i}": <0-2>' for i in range(len(checklist)))
        msgs = [{"role": "system", "content": _MATCH_SYS},
                {"role": "user", "content": [
                    {"type": "text", "text": "REFERENCE:"},
                    {"type": "image_url", "image_url": {"url": _durl(ref_img)}},
                    {"type": "text", "text": "RECONSTRUCTION:"},
                    {"type": "image_url", "image_url": {"url": _durl(render_img)}},
                    {"type": "text", "text":
                        "Checklist of reference elements:\n" + items + "\n"
                        f'For each, score 0/1/2 in the reconstruction. ONLY JSON: {{{keys}}}.'}]}]
        m = _JSON_RE.search(chat(msgs))
        if not m:
            return None
        obj = json.loads(m.group(0))
        scores = {}
        for i, e in enumerate(checklist):
            v = obj.get(str(i), obj.get(i))
            scores[e] = int(round(float(v))) if isinstance(v, (int, float)) else 0
        total = 2 * len(checklist)
        psum = sum(scores.values())
        return {"scores": scores, "present_count": psum, "total": total,
                "fraction": round(psum / total, 3) if total else None,
                "missing": [e for e, s in scores.items() if s == 0],
                "partial": [e for e, s in scores.items() if s == 1]}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# COMPARATIVE judge designs — the fix for the crushed dynamic range. Absolute
# judges (correspondence/feature_residual) floor every SVG at the same low score
# because the raster reference is unreachable, so the SCALE has no resolution for
# refinement. Comparative designs put the renders side-by-side and force the model
# to DISCRIMINATE, which recovers dynamic range.

def rank_renders(chat, ref_img: Image.Image, renders: list[Image.Image]) -> list[int] | None:
    """Show the reference and ALL N trajectory renders at once; force a DISTINCT
    ordinal rank 1..N (N = most faithful) for each. Spread is guaranteed by
    construction; the per-turn rank trajectory is the (directional) refinement
    signal. Returns a list of N ranks aligned to `renders`."""
    n = len(renders)
    if n < 2:
        return None
    content = [{"type": "text", "text": "REFERENCE image:"},
               {"type": "image_url", "image_url": {"url": _durl(ref_img)}}]
    for i, im in enumerate(renders):
        content += [{"type": "text", "text": f"RECONSTRUCTION {i}:"},
                    {"type": "image_url", "image_url": {"url": _durl(im)}}]
    keys = ", ".join(f'"{i}": <int>' for i in range(n))
    content += [{"type": "text", "text":
                 f"These are {n} attempts to reconstruct the REFERENCE. Rank them by FAITHFULNESS "
                 f"to the reference: assign each a DISTINCT integer from 1 (least faithful) to {n} "
                 f"(most faithful) — use every rank exactly once. ONLY JSON: {{{keys}}}."}]
    try:
        sys = ("You rank reconstructions by how faithfully each reproduces a reference image. "
               "Use the FULL ordinal range; every rank from 1 to N must be used exactly once.")
        m = _JSON_RE.search(chat([{"role": "system", "content": sys},
                                   {"role": "user", "content": content}]))
        if not m:
            return None
        obj = json.loads(m.group(0))
        out = [int(round(float(obj.get(str(i), obj.get(i, 0))))) for i in range(n)]
        return out if any(out) else None
    except Exception:
        return None


def pairwise_improve(chat, ref_img: Image.Image, prev_img: Image.Image,
                     cur_img: Image.Image) -> int | None:
    """Did reconstruction B (current) reproduce the reference BETTER, the same, or
    WORSE than A (previous)? Returns +1 / 0 / -1. Directional per-turn signal; the
    cumulative sum is a non-flat trajectory."""
    content = [{"type": "text", "text": "REFERENCE:"},
               {"type": "image_url", "image_url": {"url": _durl(ref_img)}},
               {"type": "text", "text": "reconstruction A (previous):"},
               {"type": "image_url", "image_url": {"url": _durl(prev_img)}},
               {"type": "text", "text": "reconstruction B (current):"},
               {"type": "image_url", "image_url": {"url": _durl(cur_img)}},
               {"type": "text", "text":
                'Does B reproduce the REFERENCE better, the same, or worse than A overall? '
                'ONLY JSON: {"verdict": "better"|"same"|"worse"}.'}]
    try:
        sys = "You compare two reconstructions of a reference and judge which is more faithful."
        m = _JSON_RE.search(chat([{"role": "system", "content": sys},
                                   {"role": "user", "content": content}]))
        if not m:
            return None
        v = (json.loads(m.group(0)).get("verdict") or "").lower()
        return {"better": 1, "same": 0, "worse": -1}.get(v)
    except Exception:
        return None


def anchored_score(chat, ref_img: Image.Image, render_img: Image.Image,
                   good: Image.Image, bad: Image.Image) -> float | None:
    """Absolute 1-10 score, but the SCALE is anchored by two exemplar renders shown
    in-context (a strong one = 9, a weak one = 2). Anchoring spreads the absolute
    scale that an un-anchored rubric crushes (the 'calibrate the old judge' path)."""
    content = [{"type": "text", "text": "REFERENCE image:"},
               {"type": "image_url", "image_url": {"url": _durl(ref_img)}},
               {"type": "text", "text": "ANCHOR — this reconstruction scores 9/10:"},
               {"type": "image_url", "image_url": {"url": _durl(good)}},
               {"type": "text", "text": "ANCHOR — this reconstruction scores 2/10:"},
               {"type": "image_url", "image_url": {"url": _durl(bad)}},
               {"type": "text", "text": "Now score THIS reconstruction on the same 1-10 scale:"},
               {"type": "image_url", "image_url": {"url": _durl(render_img)}},
               {"type": "text", "text": 'Use the anchors to calibrate. ONLY JSON: {"score": <int 1-10>}.'}]
    try:
        sys = ("You score how faithfully a reconstruction reproduces a reference on a 1-10 scale, "
               "calibrated against two anchor reconstructions whose scores are given. Use the full range.")
        m = _JSON_RE.search(chat([{"role": "system", "content": sys},
                                   {"role": "user", "content": content}]))
        if not m:
            return None
        s = json.loads(m.group(0)).get("score")
        return float(s) if isinstance(s, (int, float)) else None
    except Exception:
        return None
