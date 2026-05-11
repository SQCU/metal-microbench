#!/usr/bin/env python3
"""HellaSwag + KL divergence study, multi-strategy, high-concurrency client.

Runs four DISTINCT but equally honest prompt strategies on N HellaSwag
items, captures per-strategy generation + first-token logprobs, and
post-processes per-strategy accuracy + top-K KL between two model configs
(typically fp16 baseline vs a quantized config).

Why four strategies and not one: there is no single canonical way to
elicit a benchmark response from a SOTA instruct model. The model knows
what HellaSwag is, knows it's being evaluated, and was trained with
specific delimiter conventions. Pretending the model is a "continuation
engine" (or any other lie about its identity) puts the prompt
out-of-distribution and degrades the measurement. The interesting
methodological variance is across honest-but-different canonical
framings, all of which acknowledge the eval context.

Strategies (all multiple-choice except #4):
  1. model_aware_mc      — conversational preamble naming the model and
                           the benchmark, then MC choices.
  2. sparse_mc           — minimal "HellaSwag item. Reply A/B/C/D." +
                           lettered choices.
  3. academic_lm_eval    — verbose lm-eval-harness-style framing with
                           explicit "Context:" / "Choices:" / "Answer:"
                           delimiters.
  4. freeform_eval_aware — no MC; ask the model to continue naturally,
                           score against candidates by character LCP.

Uses httpx.AsyncClient (true async HTTP, like commercial-API SDKs) so
concurrency-induced HTTP hangs don't truncate the run.

Usage:
    BRIDGE_URL=http://127.0.0.1:8001 \\
    OUTPUT=/tmp/hellaswag_<config>.jsonl \\
    N_ITEMS=100 CONCURRENCY=32 MODEL_NAME=<as reported by /health> \\
    ./server/.venv/bin/python tools/quant_search/scripts/09_hellaswag_kl_study.py

Then post-process after running both configs:
    FP16_PATH=/tmp/hellaswag_fp16.jsonl \\
    QUANT_PATH=/tmp/hellaswag_quant.jsonl \\
    ./server/.venv/bin/python tools/quant_search/scripts/09_hellaswag_kl_study.py \\
        --post-process

Per-record JSONL schema:
  {
    "strategy":      str,
    "item_idx":      int,
    "ctx":           str,
    "endings":       [str, str, str, str],
    "label":         int,                    # gold candidate index
    "generation":    str,                    # model's response
    "pos0_logprobs": [{token, logprob}, ...] # top-K at first generated pos
    "elapsed_s":     float,
  }
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "quant_search"))

from data_loaders import _load_hellaswag_subset  # noqa: E402


BRIDGE_URL    = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8001")
N_ITEMS       = int(os.environ.get("N_ITEMS", 100))
CONCURRENCY   = int(os.environ.get("CONCURRENCY", 32))
TOP_LOGPROBS  = int(os.environ.get("TOP_LOGPROBS", 20))
OUTPUT_PATH   = Path(os.environ.get("OUTPUT", "/tmp/hellaswag_study.jsonl"))
MODEL_NAME    = os.environ.get("MODEL_NAME", "gemma-4-a4b")
# Let responses conclude naturally — the bridge now stops at the
# chat-template's <end_of_turn> (token 106) for every chat completion,
# so we don't truncate mid-thought just because we picked a low cap.
# 256 is comfortable: most natural responses end at <end_of_turn>
# well before then, and the longest verbose academic replies fit.
MC_MAX_TOKENS    = int(os.environ.get("MC_MAX_TOKENS", 256))
FREE_MAX_TOKENS  = int(os.environ.get("FREE_MAX_TOKENS", 256))
K_SAMPLES        = int(os.environ.get("K_SAMPLES", 1))     # multi-sample at T>0
SAMPLE_TEMPERATURE = float(os.environ.get("SAMPLE_TEMPERATURE", 0.0))


# ──────────────────────────────────────────────────────────────────────
# The four strategies — all honest, all eval-aware, none lying.
# ──────────────────────────────────────────────────────────────────────


_DEFAULT_STRATEGIES = ["model_aware_mc", "sparse_mc", "academic_lm_eval",
                       "freeform_eval_aware"]
STRATEGIES = (os.environ.get("STRATEGIES", "").split(",")
              if os.environ.get("STRATEGIES") else _DEFAULT_STRATEGIES)
STRATEGIES = [s.strip() for s in STRATEGIES if s.strip()]

LABELS = ["A", "B", "C", "D"]


def build_messages(strategy: str, item: dict) -> list[dict]:
    ctx = item["ctx"]
    endings = item["endings"]
    if strategy == "model_aware_mc":
        choices = "\n".join(f"{LABELS[i]}) {e}" for i, e in enumerate(endings))
        return [{
            "role": "user",
            "content": (
                f"Hi Gemma-4-a4b. We're running HellaSwag evals. "
                f"Kindly respond to the problem format below to the best "
                f"of your ability.\n\n"
                f"Context: {ctx}\n\n"
                f"Continuations:\n{choices}\n\n"
                f"Reply with one letter: A, B, C, or D."
            ),
        }]
    if strategy == "sparse_mc":
        choices = "\n".join(f"{LABELS[i]}) {e}" for i, e in enumerate(endings))
        return [{
            "role": "user",
            "content": (
                f"HellaSwag item. Reply with A, B, C, or D — nothing else.\n\n"
                f"{ctx}\n\n"
                f"{choices}"
            ),
        }]
    if strategy == "academic_lm_eval":
        choices = "\n".join(f"({LABELS[i]}) {e}" for i, e in enumerate(endings))
        return [{
            "role": "user",
            "content": (
                f"Below is a HellaSwag dataset item. The task is to "
                f"identify which of the four continuations most naturally "
                f"completes the given context.\n\n"
                f"Context: {ctx}\n\n"
                f"Choices:\n{choices}\n\n"
                f"Answer:"
            ),
        }]
    if strategy == "freeform_eval_aware":
        return [{
            "role": "user",
            "content": (
                f"HellaSwag continuation eval. The text below is the "
                f"BEGINNING of a sentence. Output the most natural "
                f"continuation of that sentence — just the continuation "
                f"text, nothing else.\n\n"
                f"{ctx}"
            ),
        }]
    raise ValueError(f"unknown strategy {strategy!r}")


def max_tokens_for_strategy(strategy: str) -> int:
    return MC_MAX_TOKENS if strategy != "freeform_eval_aware" else FREE_MAX_TOKENS


# ──────────────────────────────────────────────────────────────────────
# Per-strategy scoring
# ──────────────────────────────────────────────────────────────────────


def _ascii_alpha_words(s: str) -> list[str]:
    """Return runs of [A-Za-z]+ as separate tokens. Plain-Python
    replacement for `re.compile(r"[A-Za-z]+").findall(s)`.
    """
    words: list[str] = []
    buf: list[str] = []
    for c in s:
        if "A" <= c <= "Z" or "a" <= c <= "z":
            buf.append(c)
        elif buf:
            words.append("".join(buf))
            buf = []
    if buf:
        words.append("".join(buf))
    return words


# ──────────────────────────────────────────────────────────────────────
# LLM-as-parser primitive — a language model is a more reliable parser
# of natural-language responses than any regex. Earlier versions of
# this code used regex stacks to pull A/B/C/D out of model responses,
# which fundamentally cannot distinguish "the answer is A" from
# "Although the context..." (both contain a leading capital A). For
# verbose responses that include reasoning AND a hedge AND an answer,
# regex bucketing loses information; the LLM judge sees the structure
# and answers the literal question we want to answer ("what letter, if
# any, did the model commit to?").
#
# This is the generic primitive that will also serve SVG-MSE-style
# judging, code-eval extraction, JSON-schema validation, etc. The
# `options` arg is the menu of valid extractable answers; the
# `refusal_label` is what the parser outputs when the model didn't
# commit. Returns (extracted, status) where status ∈
# {ok, refusal, unparseable}.
# ──────────────────────────────────────────────────────────────────────


# Extract the model's actual first-turn output by trimming everything
# from the first `<turn|>` (rendered text of token 106 = <end_of_turn>)
# onward. With the bridge's chat-template-aware stop fix, the engine
# terminates at the first <end_of_turn> emission — but the rendered
# text of that final emitted token still appears in `content` as the
# literal '<turn|>'. Cutting at the first `<turn|>` gives us the
# clean, single-turn assistant output that's safe to embed in any
# downstream prompt without OOD risk. (Without the bridge fix, the
# response also contained POST-first-turn auxiliary scaffolding —
# secondary turn markers, thought channels, EOS — which is no longer
# present, but trimming is still required to remove the closing
# `<turn|>` itself.)
def _first_turn(text: str) -> str:
    idx = text.find("<turn|>")
    return (text[:idx] if idx >= 0 else text).strip()


# Output schema for LLM-as-parser. Single-token tokens to avoid
# tokenization issues (multi-token labels like "REFUSE" can be split
# arbitrarily and the parser may emit only the first piece).
#   A/B/C/D — model committed to that option
#   R       — refusal: coherent decline / hedge ("context is fragmented")
#   I       — incoherent: token loops / degenerate / unrelated
#   U       — unclear: cannot categorize
# The R-vs-I distinction matters because quantization can cause
# DIFFERENT failure modes than refusal — heterogeneous-quant configs
# may produce more I-class outputs (precision loss → loops) while
# keeping R rate similar to fp16.
PARSER_LABEL_MAP = {
    "A": ("A", "ok"),
    "B": ("B", "ok"),
    "C": ("C", "ok"),
    "D": ("D", "ok"),
    "R": (None, "refusal"),
    "I": (None, "incoherent"),
    "U": (None, "unclear"),
}


async def llm_judge_mc(client: httpx.AsyncClient, response: str,
                        semaphore: asyncio.Semaphore | None) -> tuple[str | None, str]:
    """LLM-as-parser using multi-turn chat structure: present the
    parsing task in turn 1 (user), the model's stripped prior response
    in turn 2 (assistant — canonical role for "what was previously
    said"), and the parsing question in turn 3 (user). This avoids
    inlining chat-template artifacts inside a user-message body.
    Returns (letter or None, status ∈ {ok, refusal, incoherent,
    unclear, parser_error, unparseable})."""
    cleaned = _first_turn(response)
    parser_messages = [
        {"role": "user", "content": (
            "I'm running a HellaSwag eval. I'll show you a model's "
            "response and ask you to identify what it committed to."
        )},
        {"role": "assistant", "content": (
            "Got it. Show me the response."
        )},
        {"role": "user", "content": (
            f"Here is the model's response (verbatim, with chat-template "
            f"markers removed):\n\n{cleaned}\n\n"
            f"What did the model commit to? Output exactly ONE letter:\n"
            f"  A — committed to option A\n"
            f"  B — committed to option B\n"
            f"  C — committed to option C\n"
            f"  D — committed to option D\n"
            f"  R — refused or hedged ('I cannot tell', 'context unclear')\n"
            f"  I — incoherent / token loops / unrelated to the task\n"
            f"  U — none of the above (cannot categorize)"
        )},
    ]
    body = {
        "model": MODEL_NAME,
        "messages": parser_messages,
        "max_tokens": 4,
        "temperature": 0.7,
    }
    # Caller may have already acquired the semaphore (single-permit
    # pattern in fire_one); skip re-acquire when given None.
    async def _do_call():
        try:
            r = await client.post(f"{BRIDGE_URL}/v1/chat/completions",
                                   json=body, timeout=None)
            r.raise_for_status()
            return r.json(), None
        except BaseException as e:                   # noqa: BLE001
            return None, e
    if semaphore is None:
        data, err = await _do_call()
    else:
        async with semaphore:
            data, err = await _do_call()
    if err is not None:
        return None, "parser_error"
    try:
        text = (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return None, "parser_error"
    for ch in text:
        if ch in PARSER_LABEL_MAP:
            return PARSER_LABEL_MAP[ch]
    return None, "unparseable"


async def llm_judge_freeform(client: httpx.AsyncClient, response: str,
                              endings: list[str],
                              semaphore: asyncio.Semaphore | None
                              ) -> tuple[int | None, str]:
    """Multi-turn LLM-as-parser for free-form continuations: identify
    which candidate the model's continuation most closely matches.
    Returns (idx 0..3 or None, status)."""
    cleaned = _first_turn(response)
    options_block = "\n".join(
        f"  {LABELS[i]}) {e}" for i, e in enumerate(endings))
    parser_messages = [
        {"role": "user", "content": (
            "I'm running a HellaSwag eval. A model was given a sentence "
            "context and asked to continue it freely. I'll show you the "
            "model's continuation along with four candidate continuations, "
            "and ask which candidate it most resembles."
        )},
        {"role": "assistant", "content": (
            "Got it. Show me the continuation and the candidates."
        )},
        {"role": "user", "content": (
            f"Model's continuation (verbatim, chat markers removed):\n\n"
            f"{cleaned}\n\n"
            f"Candidates:\n{options_block}\n\n"
            f"Which candidate does the continuation most closely match? "
            f"Output exactly ONE letter:\n"
            f"  A / B / C / D — matches that candidate best\n"
            f"  R — model declined / hedged\n"
            f"  I — incoherent / loops / unrelated to all candidates\n"
            f"  U — cannot categorize"
        )},
    ]
    body = {
        "model": MODEL_NAME,
        "messages": parser_messages,
        "max_tokens": 4,
        "temperature": 0.7,
    }
    async def _do_call():
        try:
            r = await client.post(f"{BRIDGE_URL}/v1/chat/completions",
                                   json=body, timeout=None)
            r.raise_for_status()
            return r.json(), None
        except BaseException as e:                   # noqa: BLE001
            return None, e
    if semaphore is None:
        data, err = await _do_call()
    else:
        async with semaphore:
            data, err = await _do_call()
    if err is not None:
        return None, "parser_error"
    try:
        text = (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return None, "parser_error"
    for ch in text:
        if ch in PARSER_LABEL_MAP:
            letter, status = PARSER_LABEL_MAP[ch]
            if letter is None:
                return None, status
            return LABELS.index(letter), "ok"
    return None, "unparseable"


def _word_set(s: str) -> set[str]:
    """Lowercased word tokens, stripped of punctuation."""
    return {w.lower() for w in _ascii_alpha_words(s)}


def _looks_like_token_loop(generation: str, min_tokens: int = 4) -> bool:
    """Heuristic for degenerate fixed-point outputs (e.g. 'de-escalating
    de-escalating de-escalating ...'). If half or more of the generation's
    word tokens are the same single word, mark as unscoreable."""
    tokens = _ascii_alpha_words(generation.lower())
    if len(tokens) < min_tokens:
        return False
    from collections import Counter
    most_common, count = Counter(tokens).most_common(1)[0]
    return count / len(tokens) >= 0.5


def score_prediction(strategy: str, generation: str,
                     endings: list[str]) -> int | None:
    """Return predicted candidate index 0..3, or None if unparseable.

    MC strategies: extract first A/B/C/D letter from response. Works for
    both bare letters ("D<turn|>") and structured replies ("The correct
    answer is **(D)**"); the regex finds the first uppercase A-D char
    regardless of surrounding punctuation.

    Freeform strategy: word-set overlap between the model's continuation
    and each candidate. HellaSwag candidates often share LATER phrases
    (e.g. gold ending "stands and lifts the weight over her head" vs
    model continuation "lifts the weight up" — overlap on {lifts, the,
    weight}). Character-level prefix matching falsely scores these as
    no-match because the first words differ. Word-set overlap captures
    the actual semantic alignment.
    """
    if strategy == "freeform_eval_aware":
        if _looks_like_token_loop(generation):
            return None
        gen_words = _word_set(generation)
        if len(gen_words) < 2:
            return None
        scores = [len(gen_words & _word_set(e)) for e in endings]
        if max(scores) == 0:
            return None
        return max(range(len(scores)), key=lambda i: (scores[i], -i))
    # MC strategies: charitable letter extraction.
    letter, _ = extract_mc_letter(generation)
    if letter is None:
        return None
    return LABELS.index(letter)


# ──────────────────────────────────────────────────────────────────────
# HTTP — httpx.AsyncClient
# ──────────────────────────────────────────────────────────────────────


async def fire_one(client: httpx.AsyncClient, strategy: str, idx: int,
                    item: dict, semaphore: asyncio.Semaphore,
                    sample_idx: int = 0) -> dict:
    """Two phases: (1) get the eval rollout from the bridge, (2) ask
    the bridge to parse its own rollout via the LLM-as-judge primitive.
    Both phases share the same semaphore so total inflight stays bounded.

    `sample_idx` distinguishes K samples per (item, strategy) at T>0;
    the seed is mixed in so each sample is a deterministic reproducible
    draw from the per-config sampling distribution.
    """
    eval_body = {
        "model": MODEL_NAME,
        "messages": build_messages(strategy, item),
        "max_tokens": max_tokens_for_strategy(strategy),
        "temperature": SAMPLE_TEMPERATURE,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
    }
    if SAMPLE_TEMPERATURE > 0.0:
        # Per-sample seed so K samples per item produce different draws
        # but each draw is reproducible across config / re-runs.
        eval_body["seed"] = idx * 10_000 + sample_idx
    base = {
        "strategy": strategy,
        "item_idx": idx,
        "sample_idx": sample_idx,
        "ctx": item["ctx"],
        "endings": item["endings"],
        "label": item["label"],
    }
    t0 = time.time()
    # SINGLE semaphore acquisition for both phases. Earlier versions
    # acquired+released for Phase 1 then re-acquired for Phase 2,
    # which created a FIFO-queue starvation pattern: when 240 tasks
    # all queue for Phase 1 then all queue for Phase 2 in order, the
    # tail tasks wait for every Phase 1 to finish first, and any
    # permit leak (e.g., a code path that returns without releasing)
    # accumulates and can stall the tail. Holding one permit for the
    # whole task eliminates the re-acquire pattern entirely.
    async with semaphore:
        try:
            resp = await client.post(
                f"{BRIDGE_URL}/v1/chat/completions",
                json=eval_body,
                timeout=None,
            )
            resp.raise_for_status()
            data = resp.json()
        except BaseException as e:                   # noqa: BLE001
            # Catch BaseException (not Exception) so CancelledError
            # also produces a record instead of silently dropping.
            return {**base, "error": f"eval: {e!r}",
                    "elapsed_s": time.time() - t0}
        try:
            choice = data["choices"][0]
            generation = choice["message"]["content"] or ""
            pos0 = (
                choice["logprobs"]["content"][0]["top_logprobs"]
                if choice.get("logprobs") and choice["logprobs"].get("content")
                else []
            )
        except (KeyError, IndexError, TypeError) as e:
            return {**base, "error": f"parse_resp: {e!r}",
                    "raw": data, "elapsed_s": time.time() - t0}

        # Phase 2: LLM-as-judge — bridge parses its own rollout.
        # Now passes None as the semaphore (already held); the
        # judge functions will skip their internal acquire when
        # given None.
        try:
            if strategy == "freeform_eval_aware":
                pred_idx, parse_status = await llm_judge_freeform(
                    client, generation, item["endings"], None)
            else:
                letter, parse_status = await llm_judge_mc(
                    client, generation, None)
                pred_idx = LABELS.index(letter) if letter else None
        except BaseException as e:                   # noqa: BLE001
            return {**base, "generation": generation, "pos0_logprobs": pos0,
                    "error": f"judge: {e!r}",
                    "elapsed_s": time.time() - t0}

    return {
        **base,
        "generation": generation,
        "pos0_logprobs": pos0,
        "pred_idx": pred_idx,
        "parse_status": parse_status,
        "elapsed_s": time.time() - t0,
    }


async def run_collection(out_path: Path) -> None:
    print(f"[study] loading {N_ITEMS} HellaSwag items", flush=True)
    rows = _load_hellaswag_subset(n_samples=N_ITEMS)
    print(f"[study] loaded {len(rows)} items", flush=True)
    print(f"[study] strategies: {STRATEGIES}", flush=True)
    print(f"[study] firing at concurrency={CONCURRENCY} → {BRIDGE_URL}",
          flush=True)
    print(f"[study] output → {out_path}", flush=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_calls = len(STRATEGIES) * len(rows) * K_SAMPLES

    print(f"[study] K_SAMPLES={K_SAMPLES} temperature={SAMPLE_TEMPERATURE}",
          flush=True)

    # max_keepalive_connections=0 disables connection pooling. Each
    # request opens a fresh TCP connection and closes it after the
    # response is read. Slightly slower (~5ms/request TCP handshake
    # overhead, negligible vs the 100s+ of ms the engine spends on
    # prefill+AR), but eliminates the half-closed-pool corruption that
    # was hanging the harness's last 16-of-400 requests in earlier runs.
    # The model_aware_mc strategy (longest prompt → slowest prefill →
    # finishes last) was the consistent victim of that corruption.
    limits = httpx.Limits(max_connections=CONCURRENCY * 2,
                          max_keepalive_connections=0)
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        tasks = []
        for strategy in STRATEGIES:
            for i, row in enumerate(rows):
                for k in range(K_SAMPLES):
                    tasks.append(asyncio.create_task(
                        fire_one(client, strategy, i, row, semaphore,
                                 sample_idx=k)
                    ))

        # Replaced asyncio.as_completed with explicit asyncio.wait +
        # periodic stall diagnostics. The previous as_completed pattern
        # would silently hang waiting for tasks that had finished server-
        # side but whose responses were stuck in the client's HTTP layer
        # (or some similar unrecovered state). With asyncio.wait we get
        # control back periodically and can: (a) log what's actually
        # pending when the run stalls, (b) extract `task.exception()` to
        # see what BaseException dropped each task without my try/except
        # noticing, and (c) decide whether to cancel-and-record or keep
        # waiting. timeout=10 below is a *diagnostic* check, not a
        # request timeout — actual HTTP calls remain server-paced
        # (timeout=None on httpx.AsyncClient.post).
        completed = 0
        wall_t0 = time.time()
        pending: set[asyncio.Task] = set(tasks)
        last_progress_t = wall_t0
        with out_path.open("w") as f:
            while pending:
                done, pending = await asyncio.wait(
                    pending, timeout=10.0,
                    return_when=asyncio.FIRST_COMPLETED)
                if done:
                    last_progress_t = time.time()
                for task in done:
                    try:
                        rec = task.result()
                    except BaseException as e:       # noqa: BLE001
                        rec = {"error": f"task_baseexc: {e!r}"}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    completed += 1
                # Periodic progress + stall diagnostics
                el = time.time() - wall_t0
                idle = time.time() - last_progress_t
                if not done:
                    # No tasks finished in the last 10s — log pending state
                    print(f"[study] STALL: {completed}/{total_calls} done, "
                          f"{len(pending)} pending, idle={idle:.1f}s "
                          f"el={el:.1f}s", flush=True)
                    # Sample a pending task's state
                    sample = next(iter(pending), None)
                    if sample is not None:
                        print(f"[study] sample pending task: {sample!r} "
                              f"done={sample.done()} cancelled={sample.cancelled()}",
                              flush=True)
                    # If stalled too long with no bridge work, give up
                    # and cancel pending so the harness exits with what
                    # we have. 60s of bridge-idle stall = real hang.
                    if idle >= 60:
                        print(f"[study] giving up: {len(pending)} tasks stuck. "
                              f"cancelling pending and recording errors.",
                              flush=True)
                        for t in pending:
                            t.cancel()
                        # Drain cancellations
                        cancelled, _ = await asyncio.wait(
                            pending, timeout=5.0)
                        for t in cancelled:
                            try:
                                rec = t.result()
                            except BaseException as e:   # noqa: BLE001
                                rec = {"error": f"force_cancelled: {e!r}"}
                            f.write(json.dumps(rec) + "\n")
                            f.flush()
                            completed += 1
                        pending = set()
                        break
                elif completed % 25 == 0 or completed == total_calls:
                    print(f"[study] {completed}/{total_calls} @ "
                          f"{completed/max(el,0.001):.1f} items/s "
                          f"({el:.1f}s elapsed, {len(pending)} pending)",
                          flush=True)

    print(f"[study] done. wrote {completed} records.", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Post-processing — per-strategy accuracy + KL between configs
# ──────────────────────────────────────────────────────────────────────


def per_strategy_accuracy(records: list[dict]) -> dict:
    """Per-strategy accuracy stats, computed from the LLM-as-parser's
    `pred_idx` and `parse_status` fields stored on each record.

    Reports separately:
      - committals: items the parser saw an answer in
      - refusals: items where the model declined to commit (parse_status
        ∈ {refusal, unrelated})
      - parser_errors: HTTP/parse layer failed
      - accuracy: correct / committals (HellaSwag standard — no abstain)
      - accuracy_with_refusals_wrong: correct / (committals + refusals)
      - se: binomial SE of accuracy over committals
    """
    out: dict[str, dict] = {}
    for s in STRATEGIES:
        rs = [r for r in records if r.get("strategy") == s and "generation" in r]
        correct = 0
        committed = 0
        refusals = 0           # coherent decline
        incoherent = 0         # degenerate / loops / unrelated
        parser_errors = 0      # parser hit HTTP error or unparseable
        unclear = 0            # parser said "U"
        for r in rs:
            status = r.get("parse_status", "unknown")
            pred = r.get("pred_idx")
            if status == "ok" and pred is not None:
                committed += 1
                if pred == r["label"]:
                    correct += 1
            elif status == "refusal":
                refusals += 1
            elif status == "incoherent":
                incoherent += 1
            elif status == "unclear":
                unclear += 1
            else:
                parser_errors += 1
        n_total = len(rs)
        # Accuracy among COMMITTED responses only.
        p = correct / max(committed, 1)
        se = math.sqrt(p * (1 - p) / max(committed, 1))
        # Strict accuracy: count refusal+incoherent+unclear as wrong.
        n_strict = committed + refusals + incoherent + unclear
        p_strict = correct / max(n_strict, 1)
        se_strict = math.sqrt(p_strict * (1 - p_strict) / max(n_strict, 1))
        out[s] = {
            "accuracy": p,
            "se": se,
            "accuracy_strict": p_strict,
            "se_strict": se_strict,
            "correct": correct,
            "committed": committed,
            "refusals": refusals,
            "incoherent": incoherent,
            "unclear": unclear,
            "parser_errors": parser_errors,
            "n_records": n_total,
        }
    return out


def topk_kl_one(p_dist: list[dict], q_dist: list[dict]) -> float | None:
    if not p_dist or not q_dist:
        return None
    p_lp = {t["token"]: t["logprob"] for t in p_dist}
    q_lp = {t["token"]: t["logprob"] for t in q_dist}
    p_min = min(p_lp.values()); q_min = min(q_lp.values())
    union = set(p_lp) | set(q_lp)
    p_unnorm = {t: math.exp(p_lp.get(t, p_min - 2.0)) for t in union}
    q_unnorm = {t: math.exp(q_lp.get(t, q_min - 2.0)) for t in union}
    pz = sum(p_unnorm.values()); qz = sum(q_unnorm.values())
    p = {t: v/pz for t, v in p_unnorm.items()}
    q = {t: v/qz for t, v in q_unnorm.items()}
    kl = 0.0
    for t in union:
        if p[t] > 0 and q[t] > 0:
            kl += p[t] * math.log(p[t] / q[t])
    return kl


def topk_top1_match(p_dist, q_dist) -> bool | None:
    if not p_dist or not q_dist:
        return None
    p_top1 = max(p_dist, key=lambda t: t["logprob"])["token"]
    q_top1 = max(q_dist, key=lambda t: t["logprob"])["token"]
    return p_top1 == q_top1


# ──────────────────────────────────────────────────────────────────────
# Two distinct deltas measure quantization-induced shift differently:
#
#   1. DISTRIBUTIONAL (pos0 top-K KL): how much the model's underlying
#      next-token logprobs at position 0 shift between configs. Captures
#      the *latent* effect on logits even where it doesn't reach the
#      sampled output.
#
#   2. BEHAVIORAL (per-item TV on empirical category distribution from
#      K sampled rollouts): how much the model's *visible behavior* on
#      a given item shifts. Sampling can amplify or smooth out the
#      latent logit shift.
#
# Their ratio matters: a config with high distributional KL but low
# behavioral TV is shifting logits in regions that don't affect the
# sampled output — the quantization is "absorbed by the sampler". A
# config with low distributional KL but high behavioral TV has
# sampling that amplifies small distribution shifts. Both are real
# signals about quantization quality.
# ──────────────────────────────────────────────────────────────────────


CATEGORIES = ["A", "B", "C", "D", "R", "I", "U"]


def category_of(record: dict) -> str:
    """Map a record's parse result to one of the behavioral categories.
    A/B/C/D = committed to that letter; R = refusal; I = incoherent;
    U = unclear / parser failed / no useful classification."""
    status = record.get("parse_status")
    pred = record.get("pred_idx")
    if status == "ok" and pred is not None and 0 <= pred < 4:
        return LABELS[pred]
    if status == "refusal":
        return "R"
    if status == "incoherent":
        return "I"
    return "U"


def empirical_distribution(records: list[dict]) -> dict[str, float]:
    """Empirical distribution over CATEGORIES from a list of K rollouts."""
    if not records:
        return {c: 0.0 for c in CATEGORIES}
    counts = {c: 0 for c in CATEGORIES}
    for r in records:
        counts[category_of(r)] += 1
    n = len(records)
    return {c: counts[c] / n for c in CATEGORIES}


def tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance between two distributions over CATEGORIES.
    Bounded in [0, 1]; symmetric; finite even with zero-mass support
    (unlike KL). Best metric for comparing empirical sample distributions
    where K is small."""
    return 0.5 * sum(abs(p[c] - q[c]) for c in CATEGORIES)


def per_strategy_behavioral(fp16_recs: list[dict],
                             quant_recs: list[dict]) -> dict:
    """Per-strategy behavioral-distribution stats: per-item TV distance
    between fp16's K-sample empirical distribution and quant's K-sample
    empirical distribution, plus per-(item × sample_idx) paired
    category-agreement at matching seeds.

    Returns {strategy: {n_items, tv_mean, tv_median, tv_p90, tv_max,
                        paired_agreement, paired_total,
                        category_marginal_fp16, category_marginal_quant}}.
    """
    out: dict[str, dict] = {}
    for s in STRATEGIES:
        f_by_item: dict[int, list[dict]] = {}
        for r in fp16_recs:
            if r.get("strategy") != s:
                continue
            f_by_item.setdefault(r["item_idx"], []).append(r)
        q_by_item: dict[int, list[dict]] = {}
        for r in quant_recs:
            if r.get("strategy") != s:
                continue
            q_by_item.setdefault(r["item_idx"], []).append(r)
        common = sorted(set(f_by_item) & set(q_by_item))
        tvs: list[float] = []
        paired_agree = 0
        paired_total = 0
        # Marginal category counts (across all items, for sanity)
        f_marg = {c: 0 for c in CATEGORIES}
        q_marg = {c: 0 for c in CATEGORIES}
        for i in common:
            f_recs = f_by_item[i]
            q_recs = q_by_item[i]
            f_dist = empirical_distribution(f_recs)
            q_dist = empirical_distribution(q_recs)
            tvs.append(tv_distance(f_dist, q_dist))
            for r in f_recs: f_marg[category_of(r)] += 1
            for r in q_recs: q_marg[category_of(r)] += 1
            # Paired-sample agreement: matching sample_idx (same seed) on
            # the same item across configs. Tells us "given identical
            # sampling-rng inputs, do the two models commit to the same
            # category?" — orthogonal signal from per-item TV.
            f_by_k = {r.get("sample_idx", 0): r for r in f_recs}
            q_by_k = {r.get("sample_idx", 0): r for r in q_recs}
            for k in set(f_by_k) & set(q_by_k):
                paired_total += 1
                if category_of(f_by_k[k]) == category_of(q_by_k[k]):
                    paired_agree += 1
        tvs.sort()
        n = len(tvs)
        f_marg_n = sum(f_marg.values()) or 1
        q_marg_n = sum(q_marg.values()) or 1
        out[s] = {
            "n_items": n,
            "tv_mean":   sum(tvs) / n if tvs else float("nan"),
            "tv_median": tvs[n // 2] if tvs else float("nan"),
            "tv_p90":    tvs[int(n * 0.9)] if n >= 10 else float("nan"),
            "tv_max":    tvs[-1] if tvs else float("nan"),
            "paired_agreement": paired_agree / max(paired_total, 1),
            "paired_total":     paired_total,
            "category_marginal_fp16":  {c: f_marg[c] / f_marg_n for c in CATEGORIES},
            "category_marginal_quant": {c: q_marg[c] / q_marg_n for c in CATEGORIES},
        }
    return out


def per_strategy_kl(fp16_recs: list[dict],
                     quant_recs: list[dict]) -> dict:
    """Return {strategy: {n, mean, median, p90, p99, max, top1_agree}}."""
    out: dict[str, dict] = {}
    for s in STRATEGIES:
        f_by_idx = {r["item_idx"]: r for r in fp16_recs
                    if r.get("strategy") == s and "pos0_logprobs" in r}
        q_by_idx = {r["item_idx"]: r for r in quant_recs
                    if r.get("strategy") == s and "pos0_logprobs" in r}
        common = sorted(set(f_by_idx) & set(q_by_idx))
        kls = []
        agree = 0
        n_scored = 0
        for i in common:
            kl = topk_kl_one(f_by_idx[i]["pos0_logprobs"],
                              q_by_idx[i]["pos0_logprobs"])
            if kl is not None:
                kls.append(kl)
                m = topk_top1_match(f_by_idx[i]["pos0_logprobs"],
                                     q_by_idx[i]["pos0_logprobs"])
                if m: agree += 1
                n_scored += 1
        kls.sort()
        out[s] = {
            "n_items": n_scored,
            "mean":    sum(kls)/len(kls) if kls else float("nan"),
            "median":  kls[len(kls)//2] if kls else float("nan"),
            "p90":     kls[int(len(kls)*0.9)] if len(kls) >= 10 else float("nan"),
            "p99":     kls[int(len(kls)*0.99)] if len(kls) >= 100 else
                       (kls[-1] if kls else float("nan")),
            "max":     kls[-1] if kls else float("nan"),
            "top1_agreement": agree / max(n_scored, 1),
        }
    return out


def post_process() -> None:
    fp16_path = Path(os.environ.get("FP16_PATH",
                                       "/tmp/hellaswag_fp16.jsonl"))
    quant_path = Path(os.environ.get("QUANT_PATH",
                                       "/tmp/hellaswag_quant.jsonl"))
    if not fp16_path.exists() or not quant_path.exists():
        print(f"need both files: fp16={fp16_path} quant={quant_path}",
              file=sys.stderr)
        sys.exit(1)
    fp16_recs = [json.loads(l) for l in fp16_path.open()]
    quant_recs = [json.loads(l) for l in quant_path.open()]

    fp16_acc = per_strategy_accuracy(fp16_recs)
    behavioral = per_strategy_behavioral(fp16_recs, quant_recs)
    quant_acc = per_strategy_accuracy(quant_recs)
    kl_per_strat = per_strategy_kl(fp16_recs, quant_recs)

    print()
    print("=" * 88)
    print("HellaSwag multi-strategy study  (top_logprobs=20)")
    print("=" * 88)
    print()
    print(f"{'strategy':<22} {'fp16 acc±SE':>14} {'quant acc±SE':>14} "
          f"{'Δ acc (pp)':>12} {'Δ se':>8}")
    print("-" * 80)
    for s in STRATEGIES:
        f = fp16_acc[s]; q = quant_acc[s]
        delta = (q["accuracy"] - f["accuracy"]) * 100
        delta_se = math.sqrt(f["se"] ** 2 + q["se"] ** 2) * 100
        print(f"{s:<22} "
              f"{f['accuracy']*100:>7.1f}±{f['se']*100:>4.1f}  "
              f"{q['accuracy']*100:>7.1f}±{q['se']*100:>4.1f}  "
              f"{delta:>+9.2f}  ±{delta_se:>5.2f}")
    print()
    print("Response-class breakdown (of n_records each):")
    print(f"{'strategy':<22} {'config':>6} {'commit':>7} "
          f"{'refusal':>8} {'incoher':>8} {'unclear':>8} {'pErr':>5}")
    print("-" * 80)
    for s in STRATEGIES:
        for label, accs in [("fp16", fp16_acc), ("quant", quant_acc)]:
            r = accs[s]
            print(f"{s:<22} {label:>6} "
                  f"{r['committed']:>7} {r['refusals']:>8} "
                  f"{r['incoherent']:>8} {r['unclear']:>8} "
                  f"{r['parser_errors']:>5}")
    print()
    print(f"{'strategy':<22} {'fp16 strict':>13} {'quant strict':>13}  "
          f"(refusals/incoherent/unclear all counted as wrong)")
    print("-" * 80)
    for s in STRATEGIES:
        f = fp16_acc[s]; q = quant_acc[s]
        print(f"{s:<22} "
              f"{f['accuracy_strict']*100:>7.1f}±{f['se_strict']*100:>4.1f}  "
              f"{q['accuracy_strict']*100:>7.1f}±{q['se_strict']*100:>4.1f}")
    print()
    print("Refusal vs incoherent are TRACKED SEPARATELY: refusal is a")
    print("coherent decline (model says 'context is fragmented, can't tell');")
    print("incoherent is a quality-failure (token loops, unrelated output).")
    print("Quantization may shift the I rate without shifting R, indicating")
    print("a precision-related quality cost distinct from model uncertainty.")
    print()
    print("DISTRIBUTIONAL delta (top-K KL on pos0 logprobs — latent shift):")
    print(f"{'strategy':<22} {'mean_kl':>9} {'median':>9} "
          f"{'p90':>8} {'max':>8} {'top1_agree':>11} {'n':>5}")
    print("-" * 88)
    for s in STRATEGIES:
        k = kl_per_strat[s]
        print(f"{s:<22} {k['mean']:>9.4f} {k['median']:>9.4f} "
              f"{k['p90']:>8.4f} {k['max']:>8.4f} "
              f"{k['top1_agreement']*100:>10.1f}% {k['n_items']:>5}")
    print()
    print("BEHAVIORAL delta (per-item TV on K-sample empirical distribution — visible shift):")
    print(f"{'strategy':<22} {'tv_mean':>9} {'tv_median':>10} "
          f"{'tv_p90':>8} {'tv_max':>8} {'paired_agree':>14} {'n':>5}")
    print("-" * 88)
    for s in STRATEGIES:
        b = behavioral[s]
        agr_str = (f"{b['paired_agreement']*100:.1f}% "
                   f"({b['paired_total']})")
        print(f"{s:<22} {b['tv_mean']:>9.4f} {b['tv_median']:>10.4f} "
              f"{b['tv_p90']:>8.4f} {b['tv_max']:>8.4f} "
              f"{agr_str:>14} {b['n_items']:>5}")
    print()
    print("Marginal category distributions (across all items × samples):")
    print(f"{'strategy':<22} {'config':<6} " +
          " ".join(f"{c:>7}" for c in CATEGORIES))
    print("-" * 88)
    for s in STRATEGIES:
        b = behavioral[s]
        for label, marg in [("fp16", b['category_marginal_fp16']),
                              ("quant", b['category_marginal_quant'])]:
            row = " ".join(f"{marg[c]*100:>6.1f}%" for c in CATEGORIES)
            print(f"{s:<22} {label:<6} {row}")
    print()
    print("Two deltas to compare:")
    print("  - DISTRIBUTIONAL (pos0 KL): how much the underlying logprobs shifted")
    print("  - BEHAVIORAL (sample-empirical TV): how much sampled rollouts shifted")
    print("  Paired-agreement is at-matching-seed sample agreement: fp16 sample k vs")
    print("  quant sample k on the same item — a different cut from per-item TV.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-process", action="store_true")
    args = parser.parse_args()
    if args.post_process:
        post_process()
        return 0
    asyncio.run(run_collection(OUTPUT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
