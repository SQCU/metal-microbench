#!/usr/bin/env python3
"""'Python without for loops' feature elicitation & judge calibration.

Dual-direction task family — gives us a clean (non-refusal, non-safety)
substrate for testing heretic-mechanism positive/negative feature
elicitation with a deterministic judge candidate AND an LLM-as-judge
candidate we can calibrate head-to-head before any feature-vector work.

Layout:
  TASKS           — 12 programming problems where both loop-style and
                    iterator-style solutions are natural in training data.
  STYLE_PROMPTS   — per-style framing that elicits a specific solution
                    style from the model without asking for it by name.
  classify_ast()  — deterministic: walks the AST, returns LOOPY /
                    ITERATOR / INVALID.
  classify_llm()  — LLM-as-judge: gets code + ast.dump + asks Gemma for
                    a binary verdict with a free-form reasoning budget.
  calibrate()     — generate N samples, apply both judges, report
                    agreement matrix + disagreement traces.
  generate_class()— on-policy rollout harness, collects K rollouts per
                    task per style via /v1/chat/completions.

Run:
    python3 notes/pyloops.py --stage calibrate
    python3 notes/pyloops.py --stage collect --n-rollouts 4
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import json
import pathlib
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / "notes" / "runs"


# ── Task set ──────────────────────────────────────────────────────────
# 12 problems where both loop-style and iterator-style solutions are
# idiomatic in training data. Small enough to reason about individually;
# broad enough to reveal whether the feature direction generalizes.
TASKS = [
    "Write a Python function that returns the sum of all even numbers in a given list.",
    "Write a Python function that returns the longest word from a sentence string.",
    "Write a Python function that reads a CSV file and returns the mean of each numeric column.",
    "Write a Python function that flattens an arbitrarily-nested list of lists into a single flat list.",
    "Write a Python function that returns the set of elements common to two given lists.",
    "Write a Python function that returns a dict mapping each word to its frequency in a given text.",
    "Write a Python function that yields the first n Fibonacci numbers.",
    "Write a Python function that checks whether a given string is a palindrome.",
    "Write a Python function that groups items from a list of dicts by a specified key.",
    "Write a Python function that returns the running total of a sequence of numbers.",
    "Write a Python function that returns the rolling average over a window of size k on a sequence.",
    "Write a Python function that solves FizzBuzz for the first n integers.",
]

STYLE_PROMPTS = {
    # Positive class: iterator/generator/comprehension/functional style.
    # Note: we deliberately don't say "no for loops" — we describe the
    # positive target style so the elicitation is a content signal, not
    # an adversarial prompt. This matches the spirit of on-policy
    # generation better.
    "iterator": (
        "Write an idiomatic modern Python solution using list/dict/set "
        "comprehensions, generator expressions, itertools, map/filter, "
        "or the iterator protocol. Prefer functional composition; avoid "
        "statement-level iteration control. Give just the function, no "
        "explanation."),
    "loop": (
        "Write a straightforward Python solution using explicit for or "
        "while loops to iterate over inputs. Stick to standard imperative "
        "Python. Give just the function, no explanation."),
}


# ── HTTP plumbing ─────────────────────────────────────────────────────

def post(path: str, body: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(user_content: str, seed: int | None = None, max_tokens: int = 400,
          temperature: float = 1.0) -> str:
    body = {"messages": [{"role": "user", "content": user_content}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    if seed is not None: body["seed"] = int(seed)
    return post("/v1/chat/completions", body, timeout=300.0)[
        "choices"][0]["message"]["content"]


def chat_batch(jobs: list[dict], workers: int = 4) -> list[str]:
    out: list[str | None] = [None] * len(jobs)
    with cf.ThreadPoolExecutor(max_workers=workers) as exe:
        futs = {exe.submit(chat, **j): i for i, j in enumerate(jobs)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            try: out[i] = fut.result()
            except Exception as e: out[i] = f"<<ERROR: {e}>>"
    return out  # type: ignore


# ── Extracting Python code from model output ─────────────────────────
# Gemma wraps its responses in <|channel>thought<channel|>…<turn|> and
# often puts code in fenced ```python blocks. Pull the largest fenced
# block if one exists, else strip channel markers and try the rest.

def _find_python_fences(resp: str) -> list[str]:
    """Pull every fenced code block whose opening line is ``` or
    ```python (optionally with trailing whitespace). Pure string ops:
    walk on '```' boundaries, treat blocks alternately as code/non-code,
    accept only those whose first line is empty or 'python'. Replaces
    re.compile(r"```(?:python)?\\s*\\n(.*?)```", re.DOTALL).findall.
    """
    parts = resp.split("```")
    # Pairs: outside, inside, outside, inside, ...
    out: list[str] = []
    for idx in range(1, len(parts), 2):
        block = parts[idx]
        nl = block.find("\n")
        if nl < 0:
            # No newline after fence opener — not a multi-line block.
            continue
        lang_line = block[:nl].strip()
        if lang_line and lang_line.lower() != "python":
            # Some other language → skip per the original regex's
            # (?:python)? group (which only matched `python` or empty).
            continue
        out.append(block[nl + 1:])
    return out


def _strip_channel(resp: str) -> str:
    OPEN, CLOSE, TURN = "<|channel>", "<channel|>", "<turn|>"
    out, i = [], 0
    while i < len(resp):
        if resp.startswith(OPEN, i):
            j = resp.find(CLOSE, i + len(OPEN))
            if j < 0:
                out.append(resp[i:]); break
            i = j + len(CLOSE)
        elif resp.startswith(TURN, i):
            i += len(TURN)
        else:
            out.append(resp[i]); i += 1
    return "".join(out)


def extract_code(resp: str) -> str:
    fences = _find_python_fences(resp)
    if fences:
        return max(fences, key=len).strip()
    return _strip_channel(resp).strip()


def _is_word_boundary(text: str, idx: int) -> bool:
    """True iff `idx` is at a Python `\\b` word boundary against `text`.
    A boundary exists where exactly one of (text[idx-1], text[idx]) is a
    word char (alnum or underscore).
    """
    def is_word(c: str | None) -> bool:
        return c is not None and (c.isalnum() or c == "_")
    left = text[idx - 1] if idx > 0 else None
    right = text[idx] if idx < len(text) else None
    return is_word(left) != is_word(right)


def _last_verdict_word(text: str, candidates: tuple[str, ...]) -> str | None:
    """Return the candidate that appears last in `text` as a whole word
    (i.e. surrounded by `\\b` boundaries). Replaces
    `re.finditer(r"\\b(LOOPY|ITERATOR)\\b", text)` + take last group.
    """
    best_pos = -1
    best = None
    for cand in candidates:
        start = 0
        while True:
            idx = text.find(cand, start)
            if idx < 0:
                break
            end = idx + len(cand)
            if _is_word_boundary(text, idx) and _is_word_boundary(text, end):
                if idx > best_pos:
                    best_pos = idx
                    best = cand
            start = idx + 1
    return best


# ── Judge 1: AST walk ─────────────────────────────────────────────────

def classify_ast(code: str) -> dict:
    """Return {'verdict': 'LOOPY'|'ITERATOR'|'INVALID',
              'has_for': bool, 'has_while': bool,
              'has_iterator_feature': bool, 'error': str|None}.

    A code snippet is LOOPY iff it contains any `For` or `While`
    statement nodes in its AST. Comprehensions contain `for` in
    source but desugar to `ListComp`/`DictComp`/`SetComp`/
    `GeneratorExp` nodes, not `For` statements — so they correctly
    classify as ITERATOR. Explicit `yield`, `iter()`/`next()` calls,
    and comprehension expressions count positively as iterator usage.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"verdict": "INVALID", "has_for": False, "has_while": False,
                "has_iterator_feature": False, "error": str(e)}
    has_for = has_while = has_iterator = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For): has_for = True
        elif isinstance(node, ast.While): has_while = True
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                  ast.GeneratorExp, ast.Yield, ast.YieldFrom)):
            has_iterator = True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"iter", "next", "map", "filter",
                                  "reduce", "sum", "any", "all", "zip",
                                  "enumerate"}:
                has_iterator = True
    verdict = "LOOPY" if (has_for or has_while) else "ITERATOR"
    return {"verdict": verdict, "has_for": has_for, "has_while": has_while,
            "has_iterator_feature": has_iterator, "error": None}


# ── Judge 2: Gemma-as-judge on code + AST dump ────────────────────────

_JUDGE_INSTR = (
    "You are a Python style classifier. Given a function and its AST "
    "dump, decide whether it uses explicit iteration statements (for / "
    "while loops) or whether it relies on iterator-style constructs "
    "(comprehensions, generator expressions, map/filter/reduce, "
    "itertools, yield, the iterator protocol).\n\n"
    "Output freeform reasoning if helpful. Your final word must be one "
    "of LOOPY or ITERATOR — that word will be parsed as your verdict.\n\n"
    "Rules:\n"
    "  * A for-statement or while-statement counts as LOOPY. Comprehensions "
    "like `[x for x in xs]` are ITERATOR, because they desugar to ListComp "
    "(not For) at the AST level.\n"
    "  * `yield` and `yield from` count as ITERATOR.\n"
    "  * If the code has BOTH a for-loop AND a comprehension, classify by "
    "whichever appears to be doing the load-bearing iteration for the "
    "function's main computation.")


def classify_llm(code: str, ast_dump: str | None = None,
                  max_tokens: int = 512) -> dict:
    """Ask Gemma. Give it both the source and the AST dump so it can
    reason over structure, not just surface tokens."""
    if ast_dump is None:
        try: ast_dump = ast.dump(ast.parse(code), indent=2)
        except Exception as e: ast_dump = f"<parse error: {e}>"
    user = (f"{_JUDGE_INSTR}\n\n"
            f"CODE:\n```python\n{code}\n```\n\n"
            f"AST DUMP:\n```\n{ast_dump}\n```\n\n"
            f"Classification:")
    out = chat(user, seed=1, temperature=0.0, max_tokens=max_tokens)
    # Take LAST uppercase LOOPY/ITERATOR in the response.
    last = _last_verdict_word(out.upper(), ("LOOPY", "ITERATOR"))
    return {"verdict": last, "raw": out}


def classify_llm_batch(code_samples: list[str]) -> list[dict]:
    """Batched (ThreadPool) LLM-judge calls. Uses B=4 engine batching."""
    jobs = []
    for code in code_samples:
        try: dump = ast.dump(ast.parse(code), indent=2)
        except Exception as e: dump = f"<parse error: {e}>"
        user = (f"{_JUDGE_INSTR}\n\n"
                f"CODE:\n```python\n{code}\n```\n\n"
                f"AST DUMP:\n```\n{dump}\n```\n\n"
                f"Classification:")
        jobs.append({"user_content": user, "temperature": 0.0,
                      "max_tokens": 512, "seed": 1})
    raws = chat_batch(jobs)
    out = []
    for raw in raws:
        last = _last_verdict_word(raw.upper(), ("LOOPY", "ITERATOR"))
        out.append({"verdict": last, "raw": raw})
    return out


# ── Stage 1: calibrate the two judges against each other ─────────────

def calibrate(n_rollouts: int = 4) -> dict:
    """Generate n_rollouts × 12 tasks × 2 styles = 96 samples.
    Apply both judges. Report:
      - Agreement rate
      - Confusion matrix LLM vs AST
      - Trace of disagreements for hand-review.
    Since generate_class() prompts explicitly for each style, we have
    a soft prior on expected verdict — but the whole point of calibration
    is to NOT rely on that prior."""
    print(f"[calibrate] generating {n_rollouts} × {len(TASKS)} × 2 = "
          f"{n_rollouts*len(TASKS)*2} samples")
    samples: list[dict] = []
    t0 = time.time()
    for style, instr in STYLE_PROMPTS.items():
        jobs = []
        for ti, task in enumerate(TASKS):
            for r in range(n_rollouts):
                seed = 100_000 + (0 if style == "iterator" else 10_000) + ti * 10 + r
                jobs.append({"user_content": f"{instr}\n\n{task}",
                              "temperature": 1.0, "seed": seed, "max_tokens": 400})
        raws = []
        for start in range(0, len(jobs), 4):
            raws.extend(chat_batch(jobs[start:start + 4]))
        for j, raw in zip(jobs, raws):
            code = extract_code(raw)
            samples.append({"style": style, "prompt": j["user_content"][:120],
                             "seed": j["seed"], "raw": raw, "code": code})
    print(f"[calibrate] generation: {time.time()-t0:.1f}s")

    # Run AST judge on all samples (fast, local, no HTTP).
    t1 = time.time()
    for s in samples:
        s["ast_verdict"] = classify_ast(s["code"])
    print(f"[calibrate] AST judge: {time.time()-t1:.2f}s")

    # Run LLM judge on samples with valid ASTs (LLM can't usefully
    # classify syntax-invalid code for our purposes).
    valid = [s for s in samples if s["ast_verdict"]["verdict"] != "INVALID"]
    t2 = time.time()
    codes = [s["code"] for s in valid]
    llm_results = []
    for start in range(0, len(codes), 4):
        llm_results.extend(classify_llm_batch(codes[start:start + 4]))
    for s, r in zip(valid, llm_results):
        s["llm_verdict"] = r
    print(f"[calibrate] LLM judge: {time.time()-t2:.1f}s on {len(valid)} valid samples")

    # Tally.
    n_invalid = sum(1 for s in samples if s["ast_verdict"]["verdict"] == "INVALID")
    n_valid = len(valid)
    agree = 0
    disagree = []
    llm_stuck = 0
    confusion = {"LOOPY_LOOPY": 0, "LOOPY_ITERATOR": 0,
                  "ITERATOR_LOOPY": 0, "ITERATOR_ITERATOR": 0}
    for s in valid:
        a = s["ast_verdict"]["verdict"]
        l = s["llm_verdict"]["verdict"]
        if l is None: llm_stuck += 1; continue
        key = f"{a}_{l}"
        confusion[key] = confusion.get(key, 0) + 1
        if a == l: agree += 1
        else: disagree.append(s)

    report = {
        "n_samples_total": len(samples),
        "n_invalid": n_invalid,
        "n_valid": n_valid,
        "llm_stuck": llm_stuck,
        "agreement_rate": agree / max(1, n_valid - llm_stuck),
        "confusion": confusion,
        "style_x_ast": {},
        "style_x_llm": {},
    }
    for s in samples:
        sk = s["style"]
        av = s["ast_verdict"]["verdict"]
        report["style_x_ast"].setdefault(sk, {}).setdefault(av, 0)
        report["style_x_ast"][sk][av] += 1
        lv = s.get("llm_verdict", {}).get("verdict")
        if lv:
            report["style_x_llm"].setdefault(sk, {}).setdefault(lv, 0)
            report["style_x_llm"][sk][lv] += 1
    report["disagreement_traces"] = [
        {"style": s["style"], "code": s["code"][:300],
          "ast": s["ast_verdict"]["verdict"],
          "llm": s["llm_verdict"]["verdict"],
          "llm_raw_tail": s["llm_verdict"]["raw"][-200:]}
        for s in disagree[:20]
    ]

    # Pretty-print.
    print("\n=== calibration report ===")
    print(f"  total samples:         {report['n_samples_total']}")
    print(f"  invalid (AST parse):   {n_invalid}")
    print(f"  valid samples:         {n_valid}")
    print(f"  LLM-stuck (no verdict): {llm_stuck}")
    print(f"  agreement rate:        {report['agreement_rate']:.2%}")
    print(f"  confusion (AST_LLM):   {confusion}")
    print(f"  style x AST verdict:   {report['style_x_ast']}")
    print(f"  style x LLM verdict:   {report['style_x_llm']}")
    if disagree:
        print(f"\n  first disagreement:")
        d = disagree[0]
        print(f"    style={d['style']}  ast={d['ast_verdict']['verdict']}  "
              f"llm={d['llm_verdict']['verdict']}")
        print(f"    code: {d['code'][:200]!r}")
        print(f"    llm tail: {d['llm_verdict']['raw'][-200:]!r}")
    return report


# ── Stage 2: on-policy data collection (for future fit) ──────────────

def collect(n_rollouts: int = 4, out_path: pathlib.Path | None = None) -> dict:
    """Generate labeled on-policy samples for fitting a direction.
    Output is saved as JSON: {positive: [codes], negative: [codes]}
    mirroring the /tmp/on_policy_seeds.json shape."""
    positive, negative = [], []
    for style, instr in STYLE_PROMPTS.items():
        jobs = []
        for ti, task in enumerate(TASKS):
            for r in range(n_rollouts):
                seed = 200_000 + (0 if style == "iterator" else 10_000) + ti * 10 + r
                jobs.append({"user_content": f"{instr}\n\n{task}",
                              "temperature": 1.0, "seed": seed, "max_tokens": 400})
        raws = []
        for start in range(0, len(jobs), 4):
            raws.extend(chat_batch(jobs[start:start + 4]))
        for raw in raws:
            code = extract_code(raw)
            (positive if style == "iterator" else negative).append(code)
    data = {"positive": positive, "negative": negative,
            "note": ("positive = iterator-style, negative = loop-style. "
                     "Seeds fit via construct_pca/rollout for an anti-for direction.")}
    if out_path is None:
        out_path = RUNS / "pyloops_seeds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[collect] wrote {len(positive)}+{len(negative)} samples → {out_path}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["calibrate", "collect"], default="calibrate")
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--save", type=pathlib.Path, default=None)
    args = ap.parse_args()
    if args.stage == "calibrate":
        rep = calibrate(n_rollouts=args.n_rollouts)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            with open(args.save, "w") as f: json.dump(rep, f, indent=2)
            print(f"report → {args.save}")
    elif args.stage == "collect":
        collect(n_rollouts=args.n_rollouts, out_path=args.save)


if __name__ == "__main__":
    main()
