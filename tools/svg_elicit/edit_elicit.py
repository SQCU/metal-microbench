#!/usr/bin/env python3
"""Source-edit SVG refinement harness — the model EDITS its own drawing PROGRAM.

The string-reassignment harness (repl_elicit) made refinement maximally expensive
(retype the whole SVG to move one feature), so the model settled for a low-LOD
blob and stopped revising. Here the artifact is a small PYTHON PROGRAM that sets
`svg` (so it can script repetitive or procedural structure with loops instead
of hand-placing primitives). The program PERSISTS across turns; the
model does not re-emit it, it EDITS it. Each turn the model sees its line-numbered
source, its render, and the MSE RESIDUAL map, and issues atomic edits to ADD /
REVISE / REDACT features, driving the residual toward zero.

Ablation axis = the EDIT INTERFACE (implementation strength):
  rewrite  : re-emit the FULL revised program in a ```python``` block (baseline —
             same residual+source feedback, but no atomic edits).
  linerange: EDIT replace A-B / insert A / delete A-B  (line-addressed).
  anchored : find/replace a verbatim source snippet (the Edit-tool pattern;
             robust to the model mis-counting lines).

Objective is the MSE residual, made literal. `target` (numpy HxWx3 uint8) is in
the program namespace so the model can MEASURE rather than guess. Outputs under
output_data/.
"""
from __future__ import annotations
import argparse, contextlib, io, json, pathlib, re, sys, time, traceback

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "archival"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np                                           # noqa: E402
from PIL import Image                                         # noqa: E402
from svg_refinement_loop import (                             # noqa: E402
    load_target_from_path, render_svg, mse_images, image_to_data_url, diff_heatmap)
from elicit import call_lm, ssim_score                        # noqa: E402
import judge as _judge                                        # noqa: E402

_PROG_TAG_RE = re.compile(r"<prog>\s*(.*?)</prog>", re.DOTALL | re.IGNORECASE)
_PY_FENCE_RE = re.compile(r"```(?:python|py|python3)?[ \t]*\r?\n(.*?)```", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+.-]*\s*\n?(.*?)```", re.DOTALL)
_SVG_TAG_RE = re.compile(r"<svg[\s>].*?</svg>", re.DOTALL | re.IGNORECASE)
_SVG_ASSIGN_RE = re.compile(r"\bsvg\b\s*=")
_ANCHOR_RE = re.compile(r"<<<OLD\s*\n(.*?)\n<<<NEW\s*\n(.*?)\n<<<END", re.DOTALL)


def _extract_program(text: str) -> str | None:
    """Tolerant, fallback-chained program extraction (ported from repl_elicit's
    reliable parser). Brittle single-delimiter matching was the round-0
    regression — the model only has to drift off the exact ```python fence once
    and the old extractor returned None, after which the harness silently
    substituted a blank `svg=''`. We accept many formats and only give up if
    NOTHING parses."""
    # 1. explicit <prog>...</prog> tag
    m = _PROG_TAG_RE.search(text)
    if m:
        return m.group(1).rstrip()
    # 2. a python-tagged (or bare) fence whose body actually assigns `svg`
    for m in _PY_FENCE_RE.finditer(text):
        if _SVG_ASSIGN_RE.search(m.group(1)) or "get_svg" in m.group(1):
            return m.group(1).rstrip()
    # 3. ANY fenced block that assigns `svg` (covers ```svg-labelled python etc.)
    for m in _ANY_FENCE_RE.finditer(text):
        if _SVG_ASSIGN_RE.search(m.group(1)):
            return m.group(1).rstrip()
    # 4. last resort: a bare hand-written <svg>...</svg> -> wrap as an assignment
    s = _SVG_TAG_RE.search(text)
    if s:
        return "svg = " + repr(s.group(0))
    # 5. any python fence at all beats fabricating a blank
    m = _PY_FENCE_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).rstrip()
    return None


def _numbered(lines: list[str]) -> str:
    return "\n".join(f"{i+1:>3}| {ln}" for i, ln in enumerate(lines))


def _apply_linerange(lines: list[str], text: str) -> tuple[list[str], int, list[str]]:
    """Parse `EDIT replace A-B | insert A | delete A-B` blocks (new code until a
    lone `END`). Apply in descending start-line order so earlier edits don't
    shift later targets."""
    ops, errs = [], []
    toks = text.splitlines()
    i = 0
    hdr = re.compile(r"^\s*EDIT\s+(replace|insert|delete)\s+(\d+)(?:\s*-\s*(\d+))?\s*$", re.I)
    while i < len(toks):
        mh = hdr.match(toks[i])
        if not mh:
            i += 1; continue
        op = mh.group(1).lower(); a = int(mh.group(2)); b = int(mh.group(3) or mh.group(2))
        i += 1
        body = []
        if op != "delete":
            while i < len(toks) and toks[i].strip() != "END":
                body.append(toks[i]); i += 1
        # skip the END line if present
        if i < len(toks) and toks[i].strip() == "END":
            i += 1
        ops.append((op, a, b, body))
    for (op, a, b, body) in sorted(ops, key=lambda o: -o[1]):
        n = len(lines)
        if op == "replace":
            if 1 <= a <= b <= n: lines[a-1:b] = body
            else: errs.append(f"replace {a}-{b} out of range (1..{n})")
        elif op == "insert":
            if 0 <= a <= n: lines[a:a] = body
            else: errs.append(f"insert after {a} out of range (0..{n})")
        elif op == "delete":
            if 1 <= a <= b <= n: del lines[a-1:b]
            else: errs.append(f"delete {a}-{b} out of range (1..{n})")
    return lines, len(ops), errs


def _apply_anchored(lines: list[str], text: str) -> tuple[list[str], int, list[str]]:
    """Find/replace verbatim snippets: <<<OLD ... <<<NEW ... <<<END."""
    src = "\n".join(lines)
    n_applied, errs = 0, []
    for old, new in _ANCHOR_RE.findall(text):
        old, new = old.rstrip("\n"), new.rstrip("\n")
        if old and old in src:
            src = src.replace(old, new, 1); n_applied += 1
        else:
            errs.append(f"anchor not found: {old[:60]!r}")
    return src.split("\n"), n_applied, errs


# ---- line-addressed toolset (no verbatim quoting; cost scales with the change) ----
_T_ADD = re.compile(r"<addlines\s+after\s*=\s*\"?(\d+)\"?\s*>\n?(.*?)</addlines>", re.DOTALL | re.I)
_T_REPL = re.compile(r"<replacelines\s+\"?(\d+)\s*-\s*(\d+)\"?\s*>\n?(.*?)</replacelines>", re.DOTALL | re.I)
_T_DEL = re.compile(r"<deletelines\s+\"?(\d+)\s*-\s*(\d+)\"?\s*/?\s*>", re.I)
_T_GREP = re.compile(r"<grepline>\s*(.*?)\s*</grepline>", re.DOTALL | re.I)


def _apply_tools(lines: list[str], text: str, design: str = "full") -> tuple[list[str], int, list[str], list[str]]:
    """Parse the line-addressed toolset. grepline calls are answered (returned in
    `grep`); add/replace/delete are applied to a COPY, syntax-validated, and only
    COMMITTED if the program still compiles — so a malformed edit never freezes
    the source (the linerange failure mode). design='addonly' exposes only
    addlines+grepline (insert-only is the safest op — it can't orphan a block)."""
    grep: list[str] = []
    for pat in _T_GREP.findall(text):
        hits = [f"{i+1}| {ln}" for i, ln in enumerate(lines) if pat in ln]
        grep.append(f"grepline {pat!r}: " + ("; ".join(hits[:12]) if hits else "(no match)"))
    ops = []
    for a, body in _T_ADD.findall(text):
        ops.append(("insert", int(a), int(a), body.rstrip("\n").split("\n")))
    errs0 = []
    if design == "addonly":
        if _T_REPL.search(text) or _T_DEL.search(text):
            errs0.append("replacelines/deletelines are not available in this mode — express revisions "
                         "as new addlines (and redactions by adding a covering element).")
    else:
        for a, b, body in _T_REPL.findall(text):
            ops.append(("replace", int(a), int(b), body.rstrip("\n").split("\n")))
        for a, b in _T_DEL.findall(text):
            ops.append(("delete", int(a), int(b), []))
    if not ops:
        return lines, 0, errs0, grep
    new = list(lines)
    errs = list(errs0)
    for (op, a, b, body) in sorted(ops, key=lambda o: -o[1]):
        n = len(new)
        if op == "insert" and 0 <= a <= n:
            new[a:a] = body
        elif op == "replace" and 1 <= a <= b <= n:
            new[a-1:b] = body
        elif op == "delete" and 1 <= a <= b <= n:
            del new[a-1:b]
        else:
            errs.append(f"{op} {a}-{b} out of range (1..{n})")
    try:
        compile("\n".join(new), "<prog>", "exec")
    except SyntaxError as e:
        return lines, 0, errs0 + [f"edit REJECTED — would break the program: {e.msg} near line "
                                  f"{e.lineno}. source unchanged; try again."], grep
    return new, len(ops), errs, grep


def _run_program(source: str, tgt_arr: np.ndarray | None) -> tuple[str | None, str]:
    ns: dict = {"np": np}
    if tgt_arr is not None:
        ns["target"] = tgt_arr.copy()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(source, ns)        # noqa: S102 — research harness, model-authored
    except Exception:
        return None, traceback.format_exc(limit=3)
    svg = ns.get("svg")
    return (svg if isinstance(svg, str) else None), ("" if isinstance(svg, str) else "program did not set `svg` to a string")


_SYS = """you're reconstructing a target image as an SVG, but you build it as a small PYTHON PROGRAM that sets the string variable `svg` — so you can SCRIPT repetitive or procedural structure with loops instead of hand-placing every primitive.

you keep ONE program across all turns. you do NOT retype it from scratch — you EDIT it. each turn we show you:
  - your current program SOURCE, line-numbered
  - the image your program currently renders
  - the MSE RESIDUAL map: BRIGHT = where your render is wrong or missing, DARK = matched
  - the current MSE and SSIM

your single goal: drive the MSE RESIDUAL toward zero by editing your program — ADD features where it's bright, REVISE features whose colour/position/size is off, and REDACT features that are wrong or raising error. make small, targeted edits and watch the residual.

`np` (numpy) is available, and the target image is in the namespace as `target`, an (H, W, 3) uint8 RGB array — MEASURE it (sample colours, find region bounds) rather than guessing.
"""

_PROTO = {
    "rewrite": "to update your program, re-emit the FULL revised program in a single ```python``` block.",
    "linerange": ("to update your program, emit one or more edit commands (line numbers refer to the "
                  "source shown):\n  EDIT replace A-B   then the new lines, then a line `END`\n"
                  "  EDIT insert A      then the new lines (inserted AFTER line A), then `END`\n"
                  "  EDIT delete A-B    (no body)\nyou may emit several EDIT blocks in one turn."),
    "anchored": ("to update your program, emit one or more find/replace edit blocks that quote a "
                 "VERBATIM snippet of your current source:\n<<<OLD\n(exact lines to find)\n<<<NEW\n"
                 "(replacement lines)\n<<<END\nthe OLD text must match your source exactly. you may "
                 "emit several blocks in one turn."),
    # line-addressing toolset — you NEVER re-quote existing code; you reference it
    # by line number (cheap — an edit costs only the new content).
    "tools": ("you edit your program with line-addressing tools (line numbers refer to the source "
              "shown — you never re-quote existing code):\n"
              "  <addlines after=N>\n  new lines\n  </addlines>      insert AFTER line N\n"
              "  <replacelines A-B>\n  new lines\n  </replacelines>   replace lines A..B\n"
              "  <deletelines A-B>                                  delete lines A..B\n"
              "  <grepline>text</grepline>                          list source lines containing 'text'\n"
              "emit as many as you like per turn. an edit that would break the program is rejected and "
              "your source is kept unchanged, so edit freely."),
    "tools_verbose": ("you refine your program with a small set of line-addressing tools. you NEVER "
                      "re-type existing code — you point at it by LINE NUMBER (the source we show you is "
                      "numbered), so an edit costs only the new content, never the old.\n\n"
                      "  • ADD a feature:    <addlines after=22>\n                        (new code lines)\n"
                      "                      </addlines>           inserts them AFTER line 22\n"
                      "  • REVISE a feature: <replacelines 14-17>\n                        (new code lines)\n"
                      "                      </replacelines>       swaps lines 14..17 for the new ones\n"
                      "  • REDACT a feature: <deletelines 31-33>   removes lines 31..33\n"
                      "  • CHECK line nums:  <grepline>fill</grepline>   lists every source line containing "
                      "'fill', with its number, so you can target the right lines without guessing.\n\n"
                      "you may issue several tool calls in one turn (e.g. grepline first to locate, then "
                      "replacelines). if an edit would make the program fail to compile we REJECT it and "
                      "keep your source as-is (we tell you why) — so you can edit boldly without fear of "
                      "bricking your program. work the bright areas of the residual: add what's missing, "
                      "fix what's mis-placed, delete what's wrong."),
}


def _tools_desc(design: str, verbose: bool) -> str:
    add = "  <addlines after=N>\n  new lines\n  </addlines>       insert AFTER line N\n"
    repl = "  <replacelines A-B>\n  new lines\n  </replacelines>    replace lines A..B\n"
    dele = "  <deletelines A-B>                                   delete lines A..B\n"
    grep = "  <grepline>text</grepline>                           list source lines containing 'text'\n"
    ops = (add + grep) if design == "addonly" else (add + repl + dele + grep)
    s = ("you edit your program with line-addressing tools (line numbers refer to the numbered source "
         "shown — you NEVER re-quote existing code, you point at it by number):\n" + ops +
         "\nemit as many as you like per turn. an edit that would break the program is rejected and your "
         "source is kept unchanged, so edit boldly.")
    if design == "addonly":
        s += (" THIS MODE IS ADD-ONLY: build the image up by inserting elements; you can't replace or "
              "delete, so layer corrections on top.")
    if verbose:
        s += (" tip: use <grepline> to find the exact line number before you edit, and always work the "
              "BRIGHT regions of the residual — that's where to add what's missing.")
    return s


def run_rollout(target: Image.Image, edit_mode: str, rounds: int, max_tokens: int,
                temperature: float, seed: int, out_dir: pathlib.Path, prefix: str,
                tool_desc: str = "terse", design: str = "full", recovery: bool = False,
                judge: bool = True, turn0: str = "simple") -> dict:
    W, H = target.size
    tgt_arr = np.asarray(target.convert("RGB"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # In-loop joint judge: ONE comparative call per round with BOTH images in
    # context, three GENERAL axes (composition / forms / color_texture). Same
    # model as the harness, different query. ON by default for tools mode — its
    # scalars feed back to the model NEXT turn alongside the residual heatmap.
    judge_on = judge and edit_mode == "tools"

    def _judge_chat(msgs):
        return call_lm(msgs, 512, 0.0, seed)[0]

    proto = _tools_desc(design, tool_desc == "verbose") if edit_mode == "tools" else _PROTO[edit_mode]
    sys_prompt = _SYS + "\n" + proto
    # turn-0 elicitation. "simple" = the original minimal-first-attempt prompt
    # (leaves first-attempt quality on the table). "rich" = a STRONG first attempt
    # using the full feature set, so round 0 is a real one-shot ceiling and
    # refinement has a strong base to improve on (best-known elicitation, ported
    # from elicit.py's SYSTEM_SVG_RICH strategy).
    if turn0 == "rich":
        turn0_text = (
            f"Target image ({W}x{H}). Write your STRONGEST possible reconstruction as your "
            f"INITIAL ```python``` program that sets `svg` — this first attempt should already "
            f"look as much like the target as you can make it; you will then refine it by editing. "
            f"Reproduce EVERYTHING you can see: every distinct region, object, and text string, in "
            f"the right place, size and colour. Use the FULL SVG feature set wherever it improves "
            f"faithfulness — linear/radialGradient for shaded or metallic fills, filters "
            f"(feGaussianBlur) for glow/soft edges, <pattern> for repeated texture, <text> for any "
            f"legible text, clipPath/opacity/transform — and choose the primitive that matches the "
            f"IMAGE STRUCTURE. Do not hold back detail for later; make round 0 your best one-shot.")
    else:
        turn0_text = (f"Target image ({W}x{H}). Write your INITIAL program now — a "
                      f"```python``` block that sets `svg`. Keep it simple; you'll "
                      f"refine it by editing.")
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": turn0_text},
            {"type": "image_url", "image_url": {"url": image_to_data_url(target)}}]},
    ]

    source: list[str] = []
    traj: list[dict] = []
    best = {"source": None, "mse": None, "ssim": None, "render": None, "round": None}
    mse_hist: list[float] = []   # per-round mse (None-free) for delta-vs-prev / vs-round-0

    def _score_and_feedback(rnd: int, edit_note: str, grep: list[str] | None = None) -> dict:
        nonlocal best
        svg, err = _run_program("\n".join(source), tgt_arr)
        # Persist the source EVERY round (even on exec/render error) so a failed
        # round leaves an auditable .py — the old code only wrote it inside the
        # render-success branch, silently dropping render-errored rounds.
        (out_dir / f"{prefix}_r{rnd:02d}.py").write_text("\n".join(source))
        fb: dict = {"round": rnd, "lines": len(source), "edit": edit_note}
        if grep:
            fb["grepline"] = grep
        render = None
        if err:
            fb["program_error"] = err.splitlines()[-1][:200]
        elif svg is None:
            fb["program_error"] = "no `svg` string produced"
        else:
            try:
                render = render_svg(svg, width=W, height=H)
                m = mse_images(target, render); s = ssim_score(target, render)
                fb["mse"], fb["ssim"] = round(m, 5), round(s, 4)
                # MSE-DELTA scalars: vs previous round and vs round 0 (the one-shot
                # ceiling). Negative = improvement (residual driven DOWN).
                if mse_hist:
                    fb["mse_delta_prev"] = round(m - mse_hist[-1], 5)
                    fb["mse_delta_round0"] = round(m - mse_hist[0], 5)
                mse_hist.append(m)
                trow = {"round": rnd, "mse": round(m, 5), "ssim": round(s, 4), "lines": len(source)}
                # In-loop joint judge: 3 general comparative scalars over (target, render).
                if judge_on:
                    jr = None
                    try:
                        jr = _judge.correspondence(_judge_chat, target, render)
                    except Exception:
                        jr = None
                    if jr:
                        fb["composition"] = jr.get("composition")
                        fb["forms"] = jr.get("forms")
                        fb["color_texture"] = jr.get("color_texture")
                        trow["composition"] = jr.get("composition")
                        trow["forms"] = jr.get("forms")
                        trow["color_texture"] = jr.get("color_texture")
                traj.append(trow)
                if best["mse"] is None or m < best["mse"]:
                    best = {"source": "\n".join(source), "mse": m, "ssim": s, "render": render, "round": rnd}
                render.save(out_dir / f"{prefix}_r{rnd:02d}_render.png")  # .py already persisted above
            except Exception:
                fb["render_error"] = traceback.format_exc(limit=2)
        # build the user turn: numbered source + render + residual
        content = [{"type": "text", "text": "result: " + json.dumps(fb)[:1100]
                    + "\n\nyour current program:\n" + _numbered(source)[:4000]}]
        if render is not None:
            judge_note = ""
            if judge_on and "composition" in fb:
                judge_note = ("  a separate comparison rated how well this render reproduces the target "
                              "(1=none .. 5=strong): composition (overall arrangement) "
                              f"{fb.get('composition')}, forms (distinct shapes present & placed) "
                              f"{fb.get('forms')}, color_texture (colour/texture regions) "
                              f"{fb.get('color_texture')} — push every axis toward 5 by composing more "
                              "correct detail.")
            content += [
                {"type": "text", "text": "current render:"},
                {"type": "image_url", "image_url": {"url": image_to_data_url(render)}},
                # The residual heatmap LOCALISES what's missing/wrong, but do NOT
                # chase per-pixel darkness: getting a region's shape, position and
                # colour RIGHT matters more than pixel-exactness (a correct element
                # placed slightly off still reads as faithful). Prioritise the
                # largest missing/wrong STRUCTURES — regions, objects, text — over
                # nudging already-plausible areas. (MSE is position-rigid; it is a
                # localiser here, not the objective.)
                {"type": "text", "text": "residual heatmap (BRIGHT = a region that is missing or wrong) — "
                                         "use it to find the LARGEST missing/incorrect structures and add or "
                                         "fix those (right shape, position, colour, and any legible text); "
                                         "don't micro-optimise already-close areas:" + judge_note},
                {"type": "image_url", "image_url": {"url": image_to_data_url(diff_heatmap(target, render))}},
            ]
        messages.append({"role": "user", "content": content})
        return fb

    t0 = time.time()
    n_edits = n_edit_errs = 0
    parse_retries = 0
    # round 0: initial program. Round 0 IS the one-shot ceiling the experiment
    # measures against, so a parse miss here must NOT be silently scored as a
    # blank canvas (the old `prog or "svg=''"` corrupted every such baseline).
    # Re-prompt ONCE with the exact required format before falling back, and
    # flag a persistent failure as parse_error instead of a fake mse.
    text, _, _ = call_lm(messages, max_tokens, temperature, seed)
    messages.append({"role": "assistant", "content": text})
    prog = _extract_program(text)
    if prog is None:
        parse_retries += 1
        messages.append({"role": "user", "content":
            "Your message contained no extractable program. Reply with ONLY a "
            "```python``` code block that assigns an SVG document string to a "
            "top-level variable named `svg` (e.g. `svg = '''<svg ...>...</svg>'''`)."})
        text, _, _ = call_lm(messages, max_tokens, temperature, seed + 9973)
        messages.append({"role": "assistant", "content": text})
        prog = _extract_program(text)
    parse_failed = prog is None
    source = (prog or "svg = ''").split("\n")
    _score_and_feedback(0, "PARSE FAIL: no program block (blank fallback)"
                        if parse_failed else "initial program")

    for rnd in range(1, rounds + 1):
        text, _, _ = call_lm(messages, max_tokens, temperature, seed + rnd)
        messages.append({"role": "assistant", "content": text})
        if edit_mode == "rewrite":
            prog = _extract_program(text)
            if prog is not None:
                source = prog.split("\n"); n_edits += 1; note = "full rewrite"
            else:
                note = "no program block found"; n_edit_errs += 1
            _score_and_feedback(rnd, note)
            continue
        elif edit_mode == "linerange":
            source, na, errs = _apply_linerange(source, text)
            n_edits += na; n_edit_errs += len(errs)
            note = f"{na} line-edits" + (f"; errors: {errs}" if errs else "")
            _score_and_feedback(rnd, note)
            continue
        elif edit_mode == "tools":
            source, na, errs, grep = _apply_tools(source, text, design)
            n_edits += na; n_edit_errs += len(errs)
            note = f"{na} edits" + (f"; errors: {errs}" if errs else "")
            if recovery and errs:   # error-recovery ablation: re-state the exact syntax
                note += ("  [reminder — to edit, emit e.g.  <addlines after=12>\\n  <code>\\n"
                         "  </addlines>  ; use <grepline>text</grepline> first to confirm the line number]")
            _score_and_feedback(rnd, note, grep=grep)
            continue
        else:  # anchored
            source, na, errs = _apply_anchored(source, text)
            n_edits += na; n_edit_errs += len(errs)
            note = f"{na} anchored-edits" + (f"; errors: {errs}" if errs else "")
        _score_and_feedback(rnd, note)

    wall = time.time() - t0

    # NOTE: the discredited post-hoc independent-ratings feature_score judge that
    # used to run here was removed — it differenced two independent descriptions
    # (degenerates to pixel statistics) and contradicted the spec's JOINT,
    # in-loop correspondence judge, which already ran every round and lives in the
    # trajectory (composition/forms/color_texture). One judge, joint, in-loop.

    target.save(out_dir / f"{prefix}_target.png")
    if best["source"] is not None:
        (out_dir / f"{prefix}_best.py").write_text(best["source"])
        if best["render"] is not None:
            best["render"].save(out_dir / f"{prefix}_best_rendered.png")

    # round 0 IS the one-shot ceiling (free baseline); surface the residual the
    # multi-turn composition is measured against, and the detail-accumulation
    # (source linecount) delta from round 0 to the end.
    r0 = traj[0] if traj else {}
    rN = traj[-1] if traj else {}
    rep = {
        "prefix": prefix, "edit_mode": edit_mode, "size": [W, H], "judge": judge_on,
        "best_mse": best["mse"], "best_ssim": best["ssim"], "best_round": best["round"],
        "oneshot_mse": r0.get("mse"),        # round-0 ceiling (one-shot baseline)
        "oneshot_ssim": r0.get("ssim"),
        "final_mse": rN.get("mse"),
        # how far multi-turn pushed BELOW its own one-shot, and how much detail it accumulated.
        "mse_drop_vs_oneshot": (round(r0["mse"] - best["mse"], 5)
                                if (r0.get("mse") is not None and best["mse"] is not None) else None),
        "lines_round0": r0.get("lines"), "lines_final": rN.get("lines"),
        "detail_accumulation": ((rN.get("lines", 0) - r0.get("lines", 0))
                                if (r0.get("lines") is not None and rN.get("lines") is not None) else None),
        # Joint-correspondence judge at the one-shot (round 0) vs the final round —
        # read from the trajectory (NOT a dangling `jr`, which is None when the
        # last judge call failed or judging is off). Directly answers "did the
        # composition/forms/color_texture correspondence rise over the turns".
        "judge_round0": ({k: r0.get(k) for k in ("composition", "forms", "color_texture")}
                         if judge_on else None),
        "judge_final": ({k: rN.get(k) for k in ("composition", "forms", "color_texture")}
                        if judge_on else None),
        "n_edits_applied": n_edits, "n_edit_errors": n_edit_errs,
        # round-0 one-shot baseline integrity: if parse failed, oneshot_mse is a
        # blank-canvas artifact and must NOT be treated as the model's one-shot.
        "round0_parse_failed": parse_failed, "round0_parse_retries": parse_retries,
        "turn0": turn0,
        "rounds": rounds, "wall_s": round(wall, 1),
        # per-round {round, mse, ssim, composition, forms, color_texture, lines} — incl. round 0.
        "trajectory": traj,
    }
    (out_dir / f"{prefix}_report.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=pathlib.Path, required=True)
    ap.add_argument("--edit-mode", choices=["rewrite", "linerange", "anchored", "tools"], required=True)
    ap.add_argument("--tool-desc", choices=["terse", "verbose"], default="terse",
                    help="(tools mode) description ablation for the line-addressing toolset")
    ap.add_argument("--tool-design", choices=["full", "addonly"], default="full",
                    help="(tools mode) full=add/replace/delete, addonly=insert-only (safest op)")
    ap.add_argument("--error-recovery", action="store_true",
                    help="(tools mode) re-state the edit syntax in feedback whenever an edit fails")
    ap.add_argument("--rounds", type=int, default=12, help="edit rounds after the initial program")
    ap.add_argument("--turn0", choices=["simple", "rich"], default="simple",
                    help="round-0 elicitation: 'rich' asks for the strongest full first attempt "
                         "(real one-shot ceiling) vs 'simple' (minimal, refine-later)")
    ap.add_argument("--no-judge", dest="judge", action="store_false",
                    help="(tools mode) disable the in-loop joint correspondence judge (ON by default)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=pathlib.Path,
                    default=_REPO / "output_data" / "svg_runs" / f"edit_{int(time.time())}")
    args = ap.parse_args()
    target = load_target_from_path(str(args.frame))
    suffix = (f"_{args.tool_design}_{args.tool_desc}{'_rec' if args.error_recovery else ''}"
              if args.edit_mode == "tools" else "")
    prefix = f"{args.frame.stem}_{args.edit_mode}{suffix}"
    rep = run_rollout(target, args.edit_mode, args.rounds, args.max_tokens,
                      args.temperature, args.seed, args.out_root, prefix, tool_desc=args.tool_desc,
                      design=args.tool_design, recovery=args.error_recovery, judge=args.judge,
                      turn0=args.turn0)
    tj = rep["trajectory"]
    print(f"[edit_elicit] mode={args.edit_mode} oneshot_mse={rep['oneshot_mse']} "
          f"best_mse={rep['best_mse']} (r{rep['best_round']}) "
          f"final_mse={rep['final_mse']} drop_vs_oneshot={rep['mse_drop_vs_oneshot']} "
          f"detail+={rep['detail_accumulation']} lines "
          f"edits={rep['n_edits_applied']}/{rep['n_edit_errors']}err -> {args.out_root}")
    if tj:
        print("  mse trajectory:", [t["mse"] for t in tj])
        print("  lines        :", [t["lines"] for t in tj])
        if rep.get("judge"):
            print("  composition  :", [t.get("composition") for t in tj])
            print("  forms        :", [t.get("forms") for t in tj])
            print("  color_texture:", [t.get("color_texture") for t in tj])


if __name__ == "__main__":
    main()
