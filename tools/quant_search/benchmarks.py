"""Concrete Benchmark instances for the multi-benchmark framework.

Design principles (per redesign 2026-05-06)
-------------------------------------------

1. **Pose questions like a normal user would.** No "Hi Gemma-4-a4b",
   no "We're running X evals", no subject tags, no format demands like
   "Answer: X must appear on its own line", no "reply with one letter".
   The model's response distribution conditioning on midwit eval-prompts
   is OOD and produces degenerate behavior — the heterogeneous-quant
   probe under test (unsloth_dyn) was directly observed regurgitating
   our prompt verbatim as hyphenated noun chains. The research goal is
   to measure typical-policy LM behavior in realistic framings, not to
   roleplay as a 2024-era hobbyist eval harness.

2. **Don't cap reasoning.** Generation runs with max_tokens=8192 by
   default; the bridge stops at <end_of_turn> naturally. Truncation at
   max_tokens is itself a measurable signal (`hit_eos: bool` recorded
   in every record). Continuation across truncated rollouts is a v2
   feature; for now we record `parse_status='truncated'` for length-
   limit-cut-off rollouts that aren't looping.

3. **LLM-as-judge for everything semantic.** Single unified
   `framework.judge_rollout` call per rollout returns
   {looping, refused, committed, extracted, equivalent_to_gold}. No
   regex, no normalize-and-substring matching, no character-level
   answer-letter scraping. Equivalence-to-gold is a semantic question
   the language model is good at; making us write a normalizer is
   roleplaying as a 2024 NLP grad student.

4. **Distributional measurements are first-class.** Beyond accuracy
   (least-sensitive coordinate), we measure:
     - status taxonomy distribution (TV cross-config)
     - response length distribution (Wasserstein on chars cross-config)
     - hit-EOS rate (TV cross-config)
     - per-position pos0 logprob distributions (already collected)
   Quant degradation can show up in any of these axes.
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

# Make data_loaders importable when this module is loaded standalone.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from data_loaders import (                                            # noqa: E402
    _load_hellaswag_subset,
    _load_mmlu_subset,
    _load_triviaqa_subset,
    _load_humaneval_subset,
)
from framework import (                                              # noqa: E402
    Benchmark,
    ParseResult,
    hit_eos,
    judge_rollout,
    status_from_judge,
    tv_distance_discrete,
    wasserstein1_1d,
)
from code_runner import extract_python_code, run_with_test            # noqa: E402


BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8001")
# Default OAI `model:` identifier. Reset per-config by the orchestrator
# (`bm.MODEL_NAME = ...` after BridgeContext.__enter__) so each config
# announces the correct identity. The "gemma-4-a4b" placeholder here is
# only used if someone runs benchmarks.py directly without going
# through the orchestrator — it does NOT imply uniform-Q4_K_M anymore;
# uniform quantization is not a valid probe for this project.
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-4-a4b")

# Universal generous default. The bridge stops at <end_of_turn>
# naturally so unused budget costs nothing; the only cost is when the
# model runs over the limit, which is itself a measurable distributional
# signal (hit_eos=False).
DEFAULT_MAX_TOKENS = 8192


# ──────────────────────────────────────────────────────────────────────
# Letter-extraction helper for MC benchmarks (HellaSwag, MMLU)
# ──────────────────────────────────────────────────────────────────────


def _extract_mc_letter(extracted: str | None) -> str | None:
    """The judge's EXTRACTED field for an MC task should be a single
    letter; sometimes it's "A" / "A." / "A) ..." / "the answer is A" /
    "B (because ...)". Pull the first standalone A/B/C/D from it."""
    if not extracted:
        return None
    # Walk character-by-character looking for a letter not followed by
    # an alphanumeric (so we don't grab the 'B' in 'Beethoven').
    s = extracted.strip()
    for i, ch in enumerate(s):
        if ch in "ABCD":
            after = s[i + 1] if i + 1 < len(s) else ""
            before = s[i - 1] if i > 0 else ""
            if not before.isalnum() and not after.isalnum():
                return ch
    # Fall through: literal first character if it's an MC letter.
    if s[:1] in "ABCD":
        return s[:1]
    return None


# ──────────────────────────────────────────────────────────────────────
# 1. HellaSwag — discrete MC (4-way commonsense continuation)
# ──────────────────────────────────────────────────────────────────────


def _hellaswag_load(n: int) -> list[dict]:
    rows = _load_hellaswag_subset(n_samples=n)
    return [
        {
            "item_id": f"hellaswag-{i}",
            "ctx": r["ctx"],
            "endings": r["endings"],
            "label": r["label"],
        }
        for i, r in enumerate(rows)
    ]


def _hellaswag_elicit(problem: dict) -> tuple[list[dict], int]:
    choices = "\n".join(
        f"{lab}. {e}" for lab, e in zip("ABCD", problem["endings"]))
    text = (
        f"Which of these naturally continues this passage?\n\n"
        f"{problem['ctx']}\n\n"
        f"{choices}"
    )
    return [{"role": "user", "content": text}], DEFAULT_MAX_TOKENS


async def _hellaswag_parse(client: httpx.AsyncClient, problem: dict,
                            rollout: str, finish_reason: str | None,
                            sema) -> ParseResult:
    gold_letter = "ABCD"[problem["label"]]
    eos = hit_eos(rollout, finish_reason)
    judge = await judge_rollout(
        client, BRIDGE_URL, MODEL_NAME,
        rollout=rollout,
        ended_naturally=eos,
        task_description=(
            "The model was given a short passage and four candidate "
            "continuations labeled A, B, C, D, and asked which "
            "naturally continues the passage."),
        what_to_extract="the single letter A, B, C, or D the model picked",
        gold_description=f"The correct continuation is letter {gold_letter}.",
        semaphore=sema,
    )
    status = status_from_judge(judge, eos)
    letter = _extract_mc_letter(judge.get("extracted")) if status == "committed" else None
    if status == "committed" and letter is None:
        status = "no_commit"
    correct = (letter == gold_letter) if letter else None
    return ParseResult(
        parse_status=status,
        metric=letter,
        correct=correct,
        judge_meta={"judge": judge, "extracted_letter": letter,
                    "gold_letter": gold_letter},
    )


hellaswag = Benchmark(
    name="hellaswag", metric_type="discrete",
    load_problems=_hellaswag_load, elicit=_hellaswag_elicit,
    parse_rollout=_hellaswag_parse,
    metric_distance=tv_distance_discrete,
)


# ──────────────────────────────────────────────────────────────────────
# 2. Algebra — 2x2 linear systems with integer solutions
# ──────────────────────────────────────────────────────────────────────


def _gen_linear_system(seed: int) -> dict:
    rng = random.Random(seed)
    while True:
        x = rng.randint(-10, 10)
        y = rng.randint(-10, 10)
        a, b = rng.randint(-5, 5), rng.randint(-5, 5)
        c, d = rng.randint(-5, 5), rng.randint(-5, 5)
        det = a * d - b * c
        if det == 0:
            continue
        if 0 in (a, b, c, d):
            continue
        e = a * x + b * y
        f = c * x + d * y
        return {"a": a, "b": b, "c": c, "d": d,
                "e": e, "f": f, "x": x, "y": y}


def _algebra_load(n: int) -> list[dict]:
    return [
        {"item_id": f"algebra-{i}",
         "system": _gen_linear_system(seed=1000 + i)}
        for i in range(n)
    ]


def _algebra_elicit(problem: dict) -> tuple[list[dict], int]:
    s = problem["system"]
    text = (
        f"Solve this system of equations for x and y:\n\n"
        f"  {s['a']}x + {s['b']}y = {s['e']}\n"
        f"  {s['c']}x + {s['d']}y = {s['f']}"
    )
    return [{"role": "user", "content": text}], DEFAULT_MAX_TOKENS


def _parse_int_from_extracted(extracted: str | None) -> float | None:
    """Pull a numeric value out of the judge's EXTRACTED phrase. The
    judge typically returns 'x = -5' or '-5' or 'x is -5'; first signed
    integer/decimal in the string.

    Plain-Python replacement for `re.search(r"-?\\d+(?:\\.\\d+)?", extracted)`:
    scan for the first run of digit characters (with an optional leading
    '-' and optional fractional tail). Identifying the structurally
    better fix is in docs/regex_replacement_plans/judge_numeric_extract.md
    (ask the judge to return JSON {"value": <number>}).
    """
    if not extracted:
        return None
    n = len(extracted)
    i = 0
    while i < n:
        c = extracted[i]
        if c.isdigit():
            start = i - 1 if i > 0 and extracted[i - 1] == "-" else i
            j = i
            while j < n and extracted[j].isdigit():
                j += 1
            if j < n and extracted[j] == ".":
                k = j + 1
                while k < n and extracted[k].isdigit():
                    k += 1
                if k > j + 1:
                    j = k
            try:
                return float(extracted[start:j])
            except ValueError:
                return None
        i += 1
    return None


async def _algebra_parse(client: httpx.AsyncClient, problem: dict,
                          rollout: str, finish_reason: str | None,
                          sema) -> ParseResult:
    gold_x = problem["system"]["x"]
    eos = hit_eos(rollout, finish_reason)
    judge = await judge_rollout(
        client, BRIDGE_URL, MODEL_NAME,
        rollout=rollout,
        ended_naturally=eos,
        task_description=(
            "The model was given a 2x2 system of linear equations and "
            "asked to solve for x and y. The solution is a pair of "
            "integers."),
        what_to_extract=(
            "the integer value of x in the model's solution. Output "
            "just the number, e.g. '-5' or '7'"),
        gold_description=f"The correct value of x is {gold_x}.",
        semaphore=sema,
    )
    status = status_from_judge(judge, eos)
    val = _parse_int_from_extracted(judge.get("extracted")) if status == "committed" else None
    if status == "committed" and val is None:
        status = "no_commit"
    if status == "committed" and val is not None:
        abs_err = abs(val - gold_x)
        return ParseResult(
            parse_status="committed",
            metric=abs_err,
            correct=(abs_err < 0.5),
            judge_meta={"judge": judge, "extracted_val": val, "gold_x": gold_x},
        )
    return ParseResult(
        parse_status=status,
        judge_meta={"judge": judge, "gold_x": gold_x},
    )


algebra = Benchmark(
    name="algebra", metric_type="continuous_scalar",
    load_problems=_algebra_load, elicit=_algebra_elicit,
    parse_rollout=_algebra_parse,
    metric_distance=wasserstein1_1d,
)


# ──────────────────────────────────────────────────────────────────────
# 3. MMLU — discrete MC over diverse academic subjects
# ──────────────────────────────────────────────────────────────────────


def _mmlu_load(n: int) -> list[dict]:
    rows = _load_mmlu_subset(n_samples=n)
    return [
        {"item_id": f"mmlu-{i}",
         "subject": r["subject"],
         "question": r["question"],
         "choices":  list(r["choices"]),
         "label":    r["answer"]}
        for i, r in enumerate(rows)
    ]


def _mmlu_elicit(problem: dict) -> tuple[list[dict], int]:
    choices = "\n".join(
        f"{lab}. {c}" for lab, c in zip("ABCD", problem["choices"]))
    text = f"{problem['question']}\n\n{choices}"
    return [{"role": "user", "content": text}], DEFAULT_MAX_TOKENS


async def _mmlu_parse(client, problem, rollout, finish_reason, sema) -> ParseResult:
    gold_letter = "ABCD"[problem["label"]]
    eos = hit_eos(rollout, finish_reason)
    judge = await judge_rollout(
        client, BRIDGE_URL, MODEL_NAME,
        rollout=rollout,
        ended_naturally=eos,
        task_description=(
            "The model was given a multiple-choice question with four "
            "answer choices labeled A, B, C, D."),
        what_to_extract="the single letter A, B, C, or D the model picked",
        gold_description=f"The correct answer is letter {gold_letter}.",
        semaphore=sema,
    )
    status = status_from_judge(judge, eos)
    letter = _extract_mc_letter(judge.get("extracted")) if status == "committed" else None
    if status == "committed" and letter is None:
        status = "no_commit"
    correct = (letter == gold_letter) if letter else None
    return ParseResult(
        parse_status=status,
        metric=letter,
        correct=correct,
        judge_meta={"judge": judge, "extracted_letter": letter,
                    "gold_letter": gold_letter,
                    "subject": problem["subject"]},
    )


mmlu = Benchmark(
    name="mmlu", metric_type="discrete",
    load_problems=_mmlu_load, elicit=_mmlu_elicit,
    parse_rollout=_mmlu_parse,
    metric_distance=tv_distance_discrete,
)


# ──────────────────────────────────────────────────────────────────────
# 4. TriviaQA — open-ended factual recall
# ──────────────────────────────────────────────────────────────────────


def _triviaqa_load(n: int) -> list[dict]:
    rows = _load_triviaqa_subset(n_samples=n)
    return [
        {"item_id":  f"triviaqa-{i}",
         "question": r["question"],
         "answer":   r["answer"],
         "aliases":  r.get("aliases") or [r["answer"]]}
        for i, r in enumerate(rows)
    ]


def _triviaqa_elicit(problem: dict) -> tuple[list[dict], int]:
    # Just the question. No preamble, no format demands.
    return [{"role": "user", "content": problem["question"]}], DEFAULT_MAX_TOKENS


async def _triviaqa_parse(client, problem, rollout, finish_reason, sema) -> ParseResult:
    aliases = problem["aliases"]
    canonical = problem["answer"]
    eos = hit_eos(rollout, finish_reason)
    gold_desc = (
        f"The canonical answer is '{canonical}'. Acceptable aliases "
        f"include: {', '.join(repr(a) for a in aliases[:6])}. Treat "
        f"semantically equivalent answers (e.g. real name vs stage "
        f"name, common abbreviations, alternate spellings) as "
        f"equivalent — use your judgment."
    )
    judge = await judge_rollout(
        client, BRIDGE_URL, MODEL_NAME,
        rollout=rollout,
        ended_naturally=eos,
        task_description="The model was asked a factual trivia question.",
        what_to_extract="the answer the model gave (a name, phrase, or short fact)",
        gold_description=gold_desc,
        semaphore=sema,
    )
    status = status_from_judge(judge, eos)
    correct = judge.get("equivalent_to_gold") if status == "committed" else None
    # Discrete metric for cross-config TV: equivalence label.
    metric = (
        "equiv" if correct is True
        else "diff" if correct is False
        else None
    ) if status == "committed" else None
    return ParseResult(
        parse_status=status, metric=metric, correct=correct,
        judge_meta={"judge": judge, "canonical_gold": canonical},
    )


triviaqa = Benchmark(
    name="triviaqa", metric_type="discrete",
    load_problems=_triviaqa_load, elicit=_triviaqa_elicit,
    parse_rollout=_triviaqa_parse,
    metric_distance=tv_distance_discrete,
)


# ──────────────────────────────────────────────────────────────────────
# 5. HumanEval — function-completion + subprocess execution as ground
#    truth for `correct`. Judge is still useful for status (looping,
#    refused, truncated) since exec only catches "code passed/failed",
#    not "model gave up before writing code".
# ──────────────────────────────────────────────────────────────────────


def _humaneval_load(n: int) -> list[dict]:
    rows = _load_humaneval_subset(n_samples=n)
    return [
        {"item_id":     r["task_id"],
         "prompt":      r["prompt"],
         "test":        r["test"],
         "entry_point": r["entry_point"]}
        for r in rows
    ]


def _humaneval_elicit(problem: dict) -> tuple[list[dict], int]:
    text = (
        f"Complete this Python function:\n\n"
        f"```python\n{problem['prompt']}```"
    )
    return [{"role": "user", "content": text}], DEFAULT_MAX_TOKENS


async def _humaneval_parse(client, problem, rollout, finish_reason, sema) -> ParseResult:
    """For HumanEval, exec is the ground truth for correctness. We still
    call the judge for the status taxonomy (looping/refused/truncated/
    no_commit). If code extracts and runs, status is 'committed' and
    correct = (tests passed)."""
    eos = hit_eos(rollout, finish_reason)
    code = extract_python_code(rollout)
    judge = await judge_rollout(
        client, BRIDGE_URL, MODEL_NAME,
        rollout=rollout,
        ended_naturally=eos,
        task_description=(
            "The model was given a Python function signature with a "
            "docstring and asked to complete the function body."),
        what_to_extract="the Python function body / implementation",
        gold_description=None,  # exec is the oracle
        semaphore=sema,
    )

    # If we got code, exec it — that's our committed-or-not signal.
    if code and len(code) >= 10:
        if f"def {problem['entry_point']}(" in code:
            full_prompt = ""
        else:
            full_prompt = problem["prompt"]
        result = await asyncio.to_thread(
            run_with_test, full_prompt, code, problem["test"],
            problem["entry_point"], 10.0,
        )
        meta = {
            "judge": judge,
            "passed": result.passed,
            "error_kind": result.error_kind,
            "stderr": result.stderr[:200] if result.stderr else "",
        }
        if result.syntax_error:
            # Couldn't even compile — treat as looping if judge agrees,
            # else no_commit. Don't shoehorn into 'committed'.
            status = "looping" if judge.get("looping") else "no_commit"
            return ParseResult(parse_status=status, judge_meta=meta)
        # Compiled and ran — that's the committed signal.
        return ParseResult(
            parse_status="committed",
            metric=("pass" if result.passed else "fail"),
            correct=result.passed,
            judge_meta=meta,
        )

    # No extractable code — fall back to judge taxonomy.
    return ParseResult(
        parse_status=status_from_judge(judge, eos),
        judge_meta={"judge": judge, "reason": "no_extractable_code"},
    )


humaneval = Benchmark(
    name="humaneval", metric_type="discrete",
    load_problems=_humaneval_load, elicit=_humaneval_elicit,
    parse_rollout=_humaneval_parse,
    metric_distance=tv_distance_discrete,
)


# ──────────────────────────────────────────────────────────────────────
# 6. SVG — multi-turn refinement via the canonical toolcards methodology.
#
#    Methodology: ports `~/sillytavern-fork/data/toolcards/installed/
#    query-to-svg/service.py` via `tools/quant_search/svg_canonical.py`.
#    Per rollout: system + user query → assistant SVG → render PNG →
#    user turn shows model the rendered image asking DONE-or-refine →
#    repeat for up to MAX_ITERS or until model emits DONE. Rendering
#    is INSIDE the rollout (between turns), via the canonical playwright
#    /chromium pipeline.
#
#    Cross-config drift metric: pairwise MSE between final rendered
#    PNGs of two configs on the same prompt. Computed in Phase 3 of the
#    multibench orchestrator from the stored base64 PNGs (no rendering
#    at aggregation time — renders happen during eval).
#
#    The metric we expose at the framework's per-record level
#    (`metric` field) is the number of completed refinement iterations,
#    purely as a behavioral-shape descriptor. The headline behavioral-
#    drift signal lives in the orchestrator's SVG-MSE phase, not here.
# ──────────────────────────────────────────────────────────────────────


_SVG_PROMPTS = [
    "a smiling face", "a tree with leaves and a trunk",
    "a simple house with a door and two windows",
    "a star with five points", "a cat sitting upright", "a sun with rays",
    "a flower with petals around a center", "a sailboat on water",
    "a mountain with a peak and snow", "a fish swimming",
    "a butterfly with symmetric wings", "a clock face showing 3:00",
    "a heart shape", "a car viewed from the side",
    "a snowflake with six arms", "a moon with craters", "a guitar",
    "a coffee cup with steam", "a pencil", "a balloon with a string",
    "a kite flying", "a pizza slice", "an umbrella", "a spaceship rocket",
    "a glasses pair", "a key", "a leaf", "a clover with three leaves",
    "a triangle pyramid", "a hexagonal honeycomb cell",
]


# Canonical render dimensions. The toolcards service.py default is
# 512×512; we match. Override via SVG_RENDER_SIZE env var if needed
# (must be the same for all configs in a comparison).
_SVG_W = int(os.environ.get("SVG_RENDER_SIZE", 512))
_SVG_H = int(os.environ.get("SVG_RENDER_SIZE", 512))
_SVG_MAX_ITERS = int(os.environ.get("SVG_MAX_ITERS", 3))


def _svg_load(n: int) -> list[dict]:
    return [
        {"item_id": f"svg-{i}", "subject": prompt}
        for i, prompt in enumerate(_SVG_PROMPTS[:n])
    ]


def _svg_elicit(problem: dict) -> tuple[list[dict], int]:
    """Iter-0 of the multi-turn refinement loop. Uses the canonical
    SYSTEM_PROMPT + initial_user_turn from svg_canonical (which re-
    exports them from the toolcards service.py)."""
    from svg_canonical import SYSTEM_PROMPT, initial_user_turn
    sys_prompt = SYSTEM_PROMPT.format(W=_SVG_W, H=_SVG_H)
    msgs = [
        {"role": "system", "content": sys_prompt},
        initial_user_turn(problem["subject"]),
    ]
    return msgs, 4096


async def _svg_parse(client, problem, rollout, finish_reason, sema) -> ParseResult:
    """Multi-turn refinement parse. Adopts the runner's iter-0 generation
    (`rollout`) and drives iters 1..MAX_ITERS-1 internally via the
    canonical svg_canonical.run_multi_turn_svg.

    Returns a ParseResult whose judge_meta carries the final SVG +
    final rendered PNG as base64. The orchestrator's Phase 3 reads
    these and computes cross-config MSE distributions.
    """
    from svg_canonical import run_multi_turn_svg
    result = await run_multi_turn_svg(
        client, BRIDGE_URL, MODEL_NAME,
        query=problem["subject"],
        max_iters=_SVG_MAX_ITERS,
        width=_SVG_W, height=_SVG_H,
        first_iter_text=rollout,
        first_iter_finish_reason=finish_reason,
        sample_temperature=1.0,
        semaphore=sema,
    )

    if result["final_png_b64"] is None:
        # No successful render across any iter — model never produced
        # a valid SVG. Status: looping if the iter trace looks
        # degenerate, else no_commit.
        had_render_failures = any(
            "render failed" in str(h.get("error", "")) for h in result["iter_history"])
        had_no_svg = any(
            h.get("error") == "no <svg> in response" for h in result["iter_history"])
        status = "looping" if had_render_failures else "no_commit"
        return ParseResult(
            parse_status=status,
            judge_meta={
                "iter_history": result["iter_history"],
                "n_iters": result["n_iters"],
                "early_exit": result["early_exit"],
                "render_failed": had_render_failures,
                "no_svg_in_any_iter": had_no_svg and not had_render_failures,
            },
        )

    # Committed: final SVG rendered successfully. Per-record metric is
    # the number of refinement iterations completed (a behavioral-shape
    # signal — the headline drift signal is the cross-config PNG MSE
    # computed in Phase 3, not this scalar).
    return ParseResult(
        parse_status="committed",
        metric=float(result["n_iters"]),
        correct=None,                                 # no oracle
        judge_meta={
            "iter_history": result["iter_history"],
            "n_iters": result["n_iters"],
            "early_exit": result["early_exit"],
            "final_svg": result["final_svg"],
            "final_png_b64": result["final_png_b64"],
            "render_w": _SVG_W,
            "render_h": _SVG_H,
        },
    )


svg = Benchmark(
    name="svg", metric_type="continuous_scalar",
    load_problems=_svg_load, elicit=_svg_elicit,
    parse_rollout=_svg_parse,
    metric_distance=wasserstein1_1d,
)


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────


BENCHMARKS: dict[str, Benchmark] = {
    "hellaswag": hellaswag,
    "algebra":   algebra,
    "mmlu":      mmlu,
    "triviaqa":  triviaqa,
    "humaneval": humaneval,
    "svg":       svg,
}
