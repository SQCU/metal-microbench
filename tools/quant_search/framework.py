"""Multi-benchmark quant-behavior framework.

Each benchmark instantiates the same five primitives:

  - load_problems(n) → list of problem dicts
  - elicit(problem) → (messages, max_tokens) for the bridge
  - parse_rollout(client, problem, rollout_text, semaphore?) → ParseResult
  - metric_distance(samples_a, samples_b) → scalar (cross-config metric distance)
  - metric_type: "discrete" | "continuous_scalar" | "continuous_vector"

Universal trichotomy at parse time
==================================

Every benchmark's parser must classify the rollout into one of:

  REFUSAL    — model coherently declined ("I cannot solve this", hedge)
  INCOHERENT — output is degenerate / unparseable / off-task
  OK         — parseable, on-task; carries a benchmark-specific metric

REFUSAL and INCOHERENT rates per config are universal cross-benchmark
signals. The OK metric is benchmark-specific; metric distributions are
compared across configs by the benchmark's `metric_distance` (TV for
discrete, 1-Wasserstein or KS for continuous, etc.).

Cross-config Pareto axes
========================

Per (benchmark, config) we compute:

  refusal_rate, incoherent_rate, ok_rate                (universal)
  pos0_kl_distribution_summary                          (where logprobs captured)
  per_item_status_tv                                    (universal)
  metric_distance vs reference (e.g. fp16)              (benchmark-specific)
  paired_sample_metric_match (where applicable)         (universal-where-makes-sense)

A quant config's "quality vector" is the concatenation of these per
benchmark. Pareto frontier in that vector space (vs the bandwidth axis)
is the quant-search artifact.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import httpx


# ──────────────────────────────────────────────────────────────────────
# Parse result and benchmark protocol
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ParseResult:
    """Result of parsing a single rollout.

    Status taxonomy
    ---------------
    `parse_status` is one of:

      committed    — judge says the model committed to an answer
      refused      — judge says the model coherently declined
      looping      — judge says the rollout is degenerate / off-task
      truncated    — generation hit max_tokens without natural EOS,
                      and judge thinks it's NOT looping. A continuation
                      candidate (continuation harness is a v2; for now
                      we just record the status so the cross-config
                      taxonomy distinguishes "ran out of room" from
                      "broke down").
      no_commit    — natural EOS but model didn't commit to an answer
                      (e.g. long CoT that trails off without a final
                      letter). This IS a legit signal: some configs
                      will produce more of these than others.
      judge_error  — the judge call itself failed (HTTP / parse). Logged
                      so we don't silently lose records, but excluded
                      from cross-config metric distance computations.

    `metric`, `correct`, `judge_meta` are populated only for committed
    records (the latter two are also populated where relevant on other
    statuses for diagnostics).
    """
    parse_status: str
    metric: Any = None
    correct: bool | None = None
    judge_meta: dict | None = None


@dataclass
class Benchmark:
    """A benchmark definition. Concrete benchmarks are instances of this
    dataclass with their hook callables wired in."""
    name: str
    metric_type: str            # "discrete" | "continuous_scalar" | "continuous_vector"
    load_problems: Callable[[int], list[dict]]
    elicit: Callable[[dict], tuple[list[dict], int]]
    parse_rollout: Callable[
        ["httpx.AsyncClient", dict, str, "str | None",
         "asyncio.Semaphore | None"],
        Awaitable[ParseResult],
    ]
    # Distance between two empirical metric distributions (for OK
    # records only). Implementations: TV for discrete; 1-Wasserstein or
    # KS for continuous_scalar; sliced-Wasserstein or per-axis-stack
    # for continuous_vector.
    metric_distance: Callable[[list[Any], list[Any]], float]


# ──────────────────────────────────────────────────────────────────────
# Helpers: distribution distances
# ──────────────────────────────────────────────────────────────────────


def tv_distance_discrete(a: list[Any], b: list[Any]) -> float:
    """Total variation distance between empirical distributions of
    discrete-valued samples. Bounded in [0, 1]."""
    if not a and not b:
        return 0.0
    keys = set(a) | set(b)
    pa = {k: a.count(k) / max(len(a), 1) for k in keys}
    pb = {k: b.count(k) / max(len(b), 1) for k in keys}
    return 0.5 * sum(abs(pa[k] - pb[k]) for k in keys)


def wasserstein1_1d(a: list[float], b: list[float]) -> float:
    """1-Wasserstein (earth mover's) distance between two 1D empirical
    distributions. Computed as the L1 distance between the empirical
    CDFs. Standard non-parametric distance metric for continuous data."""
    if not a or not b:
        return float("nan")
    aa = sorted(a)
    bb = sorted(b)
    # Merge-style: integrate the absolute difference of empirical CDFs
    # along the union of sample points.
    pts = sorted(set(aa) | set(bb))
    if len(pts) < 2:
        return 0.0
    def ecdf(xs: list[float], v: float) -> float:
        # fraction of xs ≤ v
        lo, hi = 0, len(xs)
        while lo < hi:
            mid = (lo + hi) // 2
            if xs[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(xs)
    total = 0.0
    for i in range(len(pts) - 1):
        v = pts[i]
        w = pts[i + 1]
        total += abs(ecdf(aa, v) - ecdf(bb, v)) * (w - v)
    return total


def ks_statistic_1d(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic: sup |F_a(x) − F_b(x)|.
    Bounded in [0, 1]; complementary to Wasserstein (sup vs integral)."""
    if not a or not b:
        return float("nan")
    aa = sorted(a)
    bb = sorted(b)
    pts = sorted(set(aa) | set(bb))
    def ecdf(xs: list[float], v: float) -> float:
        lo, hi = 0, len(xs)
        while lo < hi:
            mid = (lo + hi) // 2
            if xs[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(xs)
    return max(abs(ecdf(aa, v) - ecdf(bb, v)) for v in pts)


# ──────────────────────────────────────────────────────────────────────
# Generic LLM-as-judge primitives — composable building blocks for
# benchmark parsers. Both strip the chat-template's first turn from the
# rollout (the bridge fix made first-turn extraction the canonical way
# to get a clean assistant turn) before embedding in the judge prompt.
# ──────────────────────────────────────────────────────────────────────


def first_turn(text: str) -> str:
    """Extract content up to the first chat-template turn marker."""
    idx = text.find("<turn|>")
    return (text[:idx] if idx >= 0 else text).strip()


async def _post_with_retry(client: httpx.AsyncClient, url: str,
                            body: dict, n_tries: int = 2,
                            ) -> tuple[dict | None, BaseException | None]:
    """POST a chat-completion body with up to n_tries attempts. The
    bridge occasionally errors on a parser call under concurrent load
    (the eval calls saturate the engine while the parser's small
    follow-up gets queued and sometimes drops). Parser calls are
    short-prompt-short-response and cheap to retry, so a single retry
    on transient errors recovers cleanly without hiding real issues.

    Returns (json data, None) on success; (None, last_exception) on
    persistent failure across all attempts.
    """
    last_err: BaseException | None = None
    for attempt in range(n_tries):
        try:
            r = await client.post(url, json=body, timeout=None)
            r.raise_for_status()
            return r.json(), None
        except BaseException as e:                    # noqa: BLE001
            last_err = e
            if attempt + 1 < n_tries:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
    return None, last_err


def hit_eos(rollout: str, finish_reason: str | None = None) -> bool:
    """Did the rollout end with a natural turn-stop?

    The bridge's stop_sequence on token 106 (<end_of_turn>) terminates
    generation when the model emits that token; the bridge sets
    `finish_reason="stop"` and may or may not include the literal
    `<turn|>` text in the content depending on engine version.

    Authoritative signal: finish_reason. Fallback: presence of the
    `<turn|>` literal in the text (older engine behaviour). When neither
    is available we conservatively report False.
    """
    if finish_reason == "stop":
        return True
    if finish_reason == "length":
        return False
    # No finish_reason — try the text heuristic.
    return "<turn|>" in rollout


async def judge_rollout(
        client: httpx.AsyncClient,
        bridge_url: str,
        model_name: str,
        *,
        rollout: str,
        ended_naturally: bool,
        task_description: str,
        what_to_extract: str,
        gold_description: str | None,
        semaphore: asyncio.Semaphore | None = None,
        ) -> dict:
    """Unified LLM-as-judge call for rollout analysis.

    Asks the on-machine LLM five questions about a rollout in a single
    structured response:

      - is the rollout obviously looping / degenerate / off-task?
      - did the model coherently refuse?
      - did the model commit to a final answer?
      - if so, what did it commit to (verbatim short phrase)?
      - if a gold reference is provided and the model committed,
        is the committed answer semantically equivalent to gold?

    Returns:
      {
        "looping": bool,
        "refused": bool,
        "committed": bool,
        "extracted": str | None,           # None if not committed
        "equivalent_to_gold": bool | None, # None if no gold or not committed
        "raw": str,                        # raw judge response (for debug)
        "judge_status": "ok" | "judge_error:<type>",
      }

    The classifications above are mostly orthogonal but "committed"
    and "refused" are mutually exclusive in practice; "looping" usually
    implies neither but the judge can mark them independently.

    Cost: one chat-completion call per rollout. Generous max_tokens
    so the judge has room to write the structured response without
    truncation.
    """
    cleaned = first_turn(rollout)
    eos_note = ("The response ended naturally (the model finished its turn)."
                if ended_naturally else
                "The response was cut off by a length limit (the model "
                "did not finish its turn).")
    gold_block = (
        f"REFERENCE_CORRECT_ANSWER: {gold_description}\n"
        if gold_description else
        "REFERENCE_CORRECT_ANSWER: (none provided)\n")

    parser_messages = [
        {"role": "user", "content":
            "I'll show you a model's response to a task and ask you to "
            "analyze it. Format your reply exactly as I describe — one "
            "labeled line per question, nothing else."},
        {"role": "assistant", "content":
            "Understood. Show me the task and response."},
        {"role": "user", "content": (
            f"TASK: {task_description}\n"
            f"WHAT_TO_EXTRACT: {what_to_extract}\n"
            f"{gold_block}"
            f"END_STATE: {eos_note}\n\n"
            f"MODEL_RESPONSE (verbatim, chat-template markers stripped):\n"
            f"────────\n{cleaned}\n────────\n\n"
            f"Answer each question on its own labeled line. No extra "
            f"commentary, no preamble, no markdown:\n\n"
            f"LOOPING: yes/no  (response is degenerate: token loops, "
            f"hyphen-chains, the same word repeating, gibberish, or "
            f"clearly stuck. Note: parallel structure like "
            f"'A is wrong... B is wrong... C is wrong... D is right' "
            f"is NORMAL reasoning, not looping. Long chain-of-thought "
            f"is also fine. Mark looping only when the model is "
            f"obviously off the rails.)\n"
            f"REFUSED: yes/no  (model coherently declined to answer)\n"
            f"COMMITTED: yes/no  (model gave a final answer to "
            f"WHAT_TO_EXTRACT, whether or not correct)\n"
            f"EXTRACTED: <the answer the model committed to, brief "
            f"verbatim phrase, or 'none' if not committed>\n"
            f"EQUIVALENCE_REASONING: <one short sentence reasoning about "
            f"whether EXTRACTED and the REFERENCE_CORRECT_ANSWER refer to "
            f"the same thing. Stage names ≡ real names (e.g. 'David "
            f"Seville' ≡ 'Ross Bagdasarian'). Common abbreviations and "
            f"alternate spellings ≡ canonical forms (e.g. 'Sunset Blvd' "
            f"≡ 'Sunset Boulevard'). Different facts ≢ each other "
            f"(e.g. 'Asquith' ≢ 'Campbell-Bannerman' even though both "
            f"are British PMs). 'na' if not applicable.>\n"
            f"EQUIVALENT_TO_GOLD: yes/no/uncertain/na  (must be "
            f"consistent with EQUIVALENCE_REASONING; 'na' if no gold "
            f"or COMMITTED=no)"
        )},
    ]
    body = {
        "model": model_name,
        "messages": parser_messages,
        "max_tokens": 512,         # room for reasoning + answer
        "temperature": 0.0,
    }
    if semaphore is None:
        data, err = await _post_with_retry(
            client, f"{bridge_url}/v1/chat/completions", body)
    else:
        async with semaphore:
            data, err = await _post_with_retry(
                client, f"{bridge_url}/v1/chat/completions", body)

    fail = {
        "looping": False, "refused": False, "committed": False,
        "extracted": None, "equivalent_to_gold": None,
        "raw": "", "judge_status": "judge_error:none",
    }
    if err is not None:
        return {**fail, "judge_status": f"judge_error:{type(err).__name__}"}
    try:
        text = (data["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError) as e:
        return {**fail, "judge_status": f"judge_error:resp_shape:{type(e).__name__}"}

    raw_text = first_turn(text).strip()
    parsed = _parse_judge_lines(raw_text)
    parsed["raw"] = raw_text
    parsed["judge_status"] = "ok"
    return parsed


def _parse_judge_lines(text: str) -> dict:
    """Parse the structured 5-line judge response. Tolerant: missing
    or malformed lines fall back to safe defaults rather than rejecting
    the whole response."""
    out: dict = {
        "looping": False, "refused": False, "committed": False,
        "extracted": None, "equivalent_to_gold": None,
    }
    by_label: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        label, _, val = line.partition(":")
        by_label[label.strip().upper()] = val.strip()

    def _yn(label: str) -> bool:
        v = by_label.get(label, "").lower()
        return v.startswith("y")

    out["looping"]   = _yn("LOOPING")
    out["refused"]   = _yn("REFUSED")
    out["committed"] = _yn("COMMITTED")

    extracted = by_label.get("EXTRACTED", "").strip()
    if extracted and extracted.lower() not in ("none", "n/a", "na", ""):
        out["extracted"] = extracted

    eq = by_label.get("EQUIVALENT_TO_GOLD", "").strip().lower()
    if eq.startswith("y"):
        out["equivalent_to_gold"] = True
    elif eq.startswith("n") and not eq.startswith("na"):
        out["equivalent_to_gold"] = False
    # 'uncertain', 'na', 'n/a', missing → leave as None
    return out


def status_from_judge(judge: dict, ended_naturally: bool) -> str:
    """Universal status taxonomy from the judge result + EOS signal.

    Precedence (commitment dominates; looping/refusal only matter when
    the model didn't actually answer):

      1. judge_error           — judge call failed
      2. committed             — judge says model committed to an answer.
                                  This dominates the looping flag,
                                  because a committed answer is by
                                  definition NOT degenerate output —
                                  the judge is over-triggering on
                                  parallel reasoning structure
                                  (A wrong / B wrong / C wrong / D right)
                                  regardless of how the prompt is
                                  worded. Empirically observed on
                                  hellaswag/MMLU rollouts. Looping is
                                  meant to gate continuation decisions
                                  for non-committal output.
      3. refused               — coherent decline
      4. looping               — degenerate, no commitment
      5. truncated             — no commitment, no natural EOS
                                  (continuation candidate)
      6. no_commit             — natural EOS but no answer
    """
    if judge.get("judge_status", "ok") != "ok":
        return "judge_error"
    if judge["committed"]:
        return "committed"
    if judge["refused"]:
        return "refused"
    if judge["looping"]:
        return "looping"
    if not ended_naturally:
        return "truncated"
    return "no_commit"


# ──────────────────────────────────────────────────────────────────────
# Per-benchmark runner — orchestrates K-sample collection across N
# items for a given benchmark + config.
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Per-config knobs the runner respects.

    Note on timeouts: there is intentionally no per-task wall-clock
    budget. A task's wall-clock time from submit-to-response is the
    sum of HTTP setup + bridge-side queue wait + bridge-side decoding,
    and from the client we cannot separate "task spent 100s queueing
    behind other in-flight streams" from "task spent 100s actively
    decoding". A per-task budget would penalize tasks for queue
    contention they didn't cause and would require messy state-
    accounting we don't want.

    The bridge's chat-template-aware stop_sequence on <end_of_turn>
    already bounds every individual rollout naturally. If a whole
    study runs unreasonably long, that's an orchestrator-level
    decision, not a per-task one.
    """
    bridge_url: str
    model_name: str
    n_items: int
    k_samples: int
    sample_temperature: float
    concurrency: int
    top_logprobs: int = 20


async def _fire_one_rollout(
        benchmark: Benchmark,
        prob: dict,
        sample_idx: int,
        client: httpx.AsyncClient,
        config: RunConfig,
        sema: asyncio.Semaphore,
        ) -> dict:
    """Single (benchmark, item, sample) rollout.

    The semaphore is held across the WHOLE rollout (eval + judge, or
    eval + multi-turn refinement) — NOT released between phases.

    Why slot-held-for-whole-rollout instead of per-call gating: under
    a fair-FIFO asyncio semaphore with hundreds of waiting rollouts in
    a unified pool, a per-call release lets ALL waiting rollouts'
    iter-0 calls jump ahead of an in-progress rollout's iter-1 call.
    Multi-turn rollouts get starved indefinitely (observed: 900s wall
    with 0 completions because rollouts past iter-0 were perpetually
    stuck behind 228 queued iter-0 calls). Holding the slot through
    the whole rollout has a small wave-mode-synchronization cost at
    the bridge but bounded forward progress per slot.
    """
    messages, max_tok = benchmark.elicit(prob)
    eval_body = {
        "model": config.model_name,
        "messages": messages,
        "max_tokens": max_tok,
        "temperature": config.sample_temperature,
        "logprobs": True,
        "top_logprobs": config.top_logprobs,
    }
    if config.sample_temperature > 0.0:
        import hashlib
        seed = int(hashlib.sha256(
            f"{prob['item_id']}|{sample_idx}".encode()).hexdigest()[:8],
            16)
        eval_body["seed"] = seed % (2**31)
    base = {
        "benchmark": benchmark.name,
        "item_id": prob["item_id"],
        "sample_idx": sample_idx,
    }
    t0 = time.time()
    # 2026-05-07: NO REMOTE LOCKS — semaphore acquired per HTTP submit,
    # NOT held across the full eval+judge rollout. The previous design
    # made one rollout's local Python lifetime gate other rollouts'
    # ability to even submit eval calls to the engine — a 'remote lock'
    # that entangled bench-side scheduling with engine work order. With
    # per-submit acquisition, the bench's concurrency knob bounds
    # in-flight HTTP calls only; the engine sees a stream of
    # independent requests and scheduled them however it pleases.
    async with sema:
        try:
            resp = await client.post(
                f"{config.bridge_url}/v1/chat/completions",
                json=eval_body, timeout=None)
            resp.raise_for_status()
            data = resp.json()
        except BaseException as e:                   # noqa: BLE001
            return {**base, "error": f"eval: {e!r}",
                    "elapsed_s": time.time() - t0}
    try:
        choice = data["choices"][0]
        generation = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason")
        pos0 = (
            choice["logprobs"]["content"][0]["top_logprobs"]
            if choice.get("logprobs") and
               choice["logprobs"].get("content")
            else []
        )
    except (KeyError, IndexError, TypeError) as e:
        return {**base, "error": f"resp_parse: {e!r}",
                "raw": data, "elapsed_s": time.time() - t0}
    try:
        # parse_rollout makes its OWN bridge calls (judge, multi-turn
        # refinement). Each of those should re-acquire `sema` per
        # submit — they are independent HTTP requests from the
        # engine's perspective, and bounding bench-side concurrency at
        # the HTTP-submit boundary is the only legal scheduling
        # constraint we get to impose.
        pr = await benchmark.parse_rollout(
            client, prob, generation, finish_reason, sema)
    except BaseException as e:                   # noqa: BLE001
        return {**base, "generation": generation,
                "pos0_logprobs": pos0,
                "finish_reason": finish_reason,
                "error": f"judge: {e!r}",
                "elapsed_s": time.time() - t0}
    return {
        **base,
        "generation": generation,
        "pos0_logprobs": pos0,
        "finish_reason": finish_reason,
        "parse_status": pr.parse_status,
        "metric": pr.metric,
        "correct": pr.correct,
        "judge_meta": pr.judge_meta,
        "hit_eos": hit_eos(generation, finish_reason),
        "gen_chars": len(generation),
        "elapsed_s": time.time() - t0,
    }


async def run_benchmarks_pooled(
        benchmarks: dict[str, Benchmark],
        config: RunConfig,
        out_paths: dict[str, Path],
        ) -> None:
    """Run multiple benchmarks under a SINGLE shared semaphore + worker
    pool. All `(benchmark, item, sample)` tickets compete for the same
    pool of `config.concurrency` slots; tail-end stragglers of one
    benchmark overlap with the next benchmark's head, so the bridge
    stays at full concurrent-stream count throughout.

    Records are still written per-benchmark (one jsonl per benchmark),
    keyed by the `out_paths` dict — backwards-compatible with the
    aggregator and the existing per-benchmark file layout.
    """
    # Load problems for each benchmark up front.
    problems_by_bench: dict[str, list[dict]] = {}
    for name, b in benchmarks.items():
        print(f"[pool] loading {config.n_items} problems for {name}",
              flush=True)
        problems_by_bench[name] = b.load_problems(config.n_items)

    # Build the unified ticket list. Order: ROUND-ROBIN across
    # benchmarks. Each benchmark contributes one ticket in turn until
    # all are exhausted.
    #
    # Why round-robin instead of longest-job-first: with the slot held
    # for the whole rollout (anti-starvation, see _fire_one_rollout),
    # placing all SVG tickets at the front of the queue makes SVG
    # monopolize all 12 slots for the entire SVG cohort drain (~hours)
    # before any single-turn benchmark gets a slot. A killed-mid-run
    # probe then has SVG-only data and zero of the other 5 benchmarks
    # — useless for cross-benchmark comparison. Round-robin gives
    # every benchmark a proportional slice of completed records at
    # any cancellation point.
    #
    # The per-(benchmark) order is sample-major within problem-major
    # so K samples of the same problem cluster together (the bridge's
    # prefix-cache appreciates the locality).
    per_bench_queues: dict[str, list[tuple[str, dict, int]]] = {}
    for name, b in benchmarks.items():
        q: list[tuple[str, dict, int]] = []
        for prob in problems_by_bench[name]:
            for k in range(config.k_samples):
                q.append((name, prob, k))
        per_bench_queues[name] = q

    tickets: list[tuple[str, dict, int]] = []
    queues = list(per_bench_queues.values())
    while any(queues):
        for q in queues:
            if q:
                tickets.append(q.pop(0))

    total = len(tickets)
    print(f"[pool] total tickets: {total} "
          f"({len(benchmarks)} benchmarks × {config.n_items} items × "
          f"{config.k_samples} samples), concurrency={config.concurrency}, "
          f"bridge_url={config.bridge_url}", flush=True)

    sema = asyncio.Semaphore(config.concurrency)
    for p in out_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    file_handles = {name: out_paths[name].open("w")
                    for name in benchmarks}

    limits = httpx.Limits(max_connections=config.concurrency * 4,
                          max_keepalive_connections=0)
    completed_by_bench: dict[str, int] = {n: 0 for n in benchmarks}
    wall_t0 = time.time()
    last_progress_t = wall_t0
    completed = 0

    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        async def _run_ticket(name: str, prob: dict, k: int) -> tuple[str, dict]:
            rec = await _fire_one_rollout(
                benchmarks[name], prob, k, client, config, sema)
            return name, rec

        tasks = [
            asyncio.create_task(_run_ticket(name, prob, k))
            for (name, prob, k) in tickets
        ]
        pending: set[asyncio.Task] = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, timeout=10.0,
                    return_when=asyncio.FIRST_COMPLETED)
                if done:
                    last_progress_t = time.time()
                for task in done:
                    try:
                        bench_name, rec = task.result()
                    except BaseException as e:       # noqa: BLE001
                        bench_name = "?"
                        rec = {"error": f"task_baseexc: {e!r}"}
                    fh = file_handles.get(bench_name)
                    if fh is not None:
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        completed_by_bench[bench_name] = (
                            completed_by_bench.get(bench_name, 0) + 1)
                    completed += 1
                el = time.time() - wall_t0
                idle = time.time() - last_progress_t
                if not done:
                    print(f"[pool] stall: {completed}/{total} done, "
                          f"{len(pending)} pending, idle={idle:.1f}s "
                          f"el={el:.1f}s "
                          f"per_bench={completed_by_bench}", flush=True)
                elif completed % 50 == 0 or completed == total:
                    print(f"[pool] {completed}/{total} @ "
                          f"{completed/max(el,0.001):.1f} items/s "
                          f"({el:.1f}s, {len(pending)} pending) "
                          f"per_bench={completed_by_bench}",
                          flush=True)
        finally:
            for fh in file_handles.values():
                fh.close()

    print(f"[pool] done. wrote {completed}/{total} records across "
          f"{len(benchmarks)} benchmarks in {time.time()-wall_t0:.1f}s",
          flush=True)


async def run_benchmark(
        benchmark: Benchmark,
        config: RunConfig,
        out_path: Path,
        ) -> None:
    """Run a single benchmark. Thin wrapper over run_benchmarks_pooled
    for backwards compatibility / single-benchmark debugging."""
    await run_benchmarks_pooled(
        {benchmark.name: benchmark}, config,
        {benchmark.name: out_path},
    )


# ──────────────────────────────────────────────────────────────────────
# Cross-config aggregation
# ──────────────────────────────────────────────────────────────────────


@dataclass
class CrossConfigSummary:
    """Per-(benchmark) cross-config comparison stats. The reference
    config is typically fp16; the comparison config is whichever quant
    config is being measured.

    Distributional axes (any one of these can carry quant-divergence
    signal independently — accuracy is the LEAST sensitive coordinate):

      - status_tv: marginal TV on the new status taxonomy
        (committed/refused/looping/truncated/no_commit/judge_error)
      - per_item_status_tv: same TV at per-item resolution (averaged)
      - ok_metric_distance: distance between metric distributions on
        committed records only
      - hit_eos_rate_diff: |ref hit-EOS-rate − cmp hit-EOS-rate|. A
        config that runs longer rollouts on the same problems is
        distributionally different even if it ends up answering the
        same things.
      - gen_chars_wasserstein: 1-W on response-length distributions
        (chars). Captures verbosity shifts that the status taxonomy
        misses.
      - accuracy: only meaningful where `correct` is well-defined
        (committed + gold equivalence). Reported but not the headline.
    """
    benchmark: str
    ref_label: str
    cmp_label: str
    n_problems: int
    # New status taxonomy marginals
    ref_marginal: dict[str, float]
    cmp_marginal: dict[str, float]
    status_tv: float
    # Per-item TV on the status distribution from the K samples.
    per_item_status_tv_mean: float
    per_item_status_tv_median: float
    per_item_status_tv_max: float
    # Metric distance over committed records only.
    ok_metric_distance: float
    ok_n_ref: int
    ok_n_cmp: int
    # Length-distribution Wasserstein on generation char counts (all records).
    gen_chars_wasserstein: float
    ref_gen_chars_median: float
    cmp_gen_chars_median: float
    # EOS rates (fraction of records that ended naturally with <turn|>).
    ref_hit_eos_rate: float
    cmp_hit_eos_rate: float
    hit_eos_rate_diff: float
    # Discrete correctness rates if benchmark provides `correct`.
    ref_accuracy: float | None
    cmp_accuracy: float | None
    # Paired-sample agreement (matching item × sample_idx).
    paired_status_agreement: float
    paired_metric_agreement: float | None
    paired_total: int


_STATUS_BUCKETS = (
    "committed", "refused", "looping", "truncated", "no_commit",
    "judge_error", "other",
)


def _empirical_marginal(records: list[dict]) -> dict[str, float]:
    """Marginal distribution over the status taxonomy."""
    bucket = {k: 0 for k in _STATUS_BUCKETS}
    for r in records:
        s = r.get("parse_status")
        if s in bucket:
            bucket[s] += 1
        else:
            bucket["other"] += 1
    n = max(sum(bucket.values()), 1)
    return {k: v / n for k, v in bucket.items()}


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def aggregate_cross_config(
        ref_records: list[dict],
        cmp_records: list[dict],
        benchmark: Benchmark,
        ref_label: str = "fp16",
        cmp_label: str = "quant",
        ) -> CrossConfigSummary:
    """Compute the cross-config stats for one benchmark."""
    ref_marg = _empirical_marginal(ref_records)
    cmp_marg = _empirical_marginal(cmp_records)
    keys = set(ref_marg) | set(cmp_marg)
    status_tv = 0.5 * sum(abs(ref_marg.get(k, 0) - cmp_marg.get(k, 0))
                            for k in keys)

    # Per-item status TV
    ref_by_item: dict[Any, list[dict]] = {}
    for r in ref_records:
        ref_by_item.setdefault(r.get("item_id"), []).append(r)
    cmp_by_item: dict[Any, list[dict]] = {}
    for r in cmp_records:
        cmp_by_item.setdefault(r.get("item_id"), []).append(r)
    common = sorted(set(ref_by_item) & set(cmp_by_item),
                     key=lambda k: str(k))
    per_item_tvs: list[float] = []
    for it in common:
        a_statuses = [r.get("parse_status", "other") for r in ref_by_item[it]]
        b_statuses = [r.get("parse_status", "other") for r in cmp_by_item[it]]
        per_item_tvs.append(tv_distance_discrete(a_statuses, b_statuses))
    per_item_tvs.sort()
    n = len(per_item_tvs) or 1
    pi_mean = sum(per_item_tvs) / n if per_item_tvs else float("nan")
    pi_median = per_item_tvs[n // 2] if per_item_tvs else float("nan")
    pi_max = per_item_tvs[-1] if per_item_tvs else float("nan")

    # Committed-record metric distance. For continuous-scalar metrics
    # we use Wasserstein (the benchmark's metric_distance). For discrete
    # we use TV.
    ref_ok = [r["metric"] for r in ref_records
                if r.get("parse_status") == "committed"
                    and r.get("metric") is not None]
    cmp_ok = [r["metric"] for r in cmp_records
                if r.get("parse_status") == "committed"
                    and r.get("metric") is not None]
    if ref_ok and cmp_ok:
        ok_metric_dist = benchmark.metric_distance(ref_ok, cmp_ok)
    else:
        ok_metric_dist = float("nan")

    # Length-distribution distance (all records, regardless of status).
    # Verbosity shift is a real distributional signal that's invisible
    # at the status-taxonomy resolution.
    ref_lens = [r["gen_chars"] for r in ref_records
                  if r.get("gen_chars") is not None]
    cmp_lens = [r["gen_chars"] for r in cmp_records
                  if r.get("gen_chars") is not None]
    if ref_lens and cmp_lens:
        gen_chars_w = wasserstein1_1d(
            [float(x) for x in ref_lens],
            [float(x) for x in cmp_lens])
    else:
        gen_chars_w = float("nan")

    # Hit-EOS rate: fraction of records that ended naturally.
    def _eos_rate(records: list[dict]) -> float:
        with_eos = [r for r in records if "hit_eos" in r]
        if not with_eos:
            return float("nan")
        return sum(1 for r in with_eos if r["hit_eos"]) / len(with_eos)

    ref_eos = _eos_rate(ref_records)
    cmp_eos = _eos_rate(cmp_records)
    eos_diff = (abs(ref_eos - cmp_eos)
                 if (ref_eos == ref_eos and cmp_eos == cmp_eos) else float("nan"))

    # Accuracy: among committed records with `correct` set
    def _acc(records: list[dict]) -> float | None:
        ok_with_corr = [r for r in records
                          if r.get("parse_status") == "committed"
                              and r.get("correct") is not None]
        if not ok_with_corr:
            return None
        return sum(1 for r in ok_with_corr if r["correct"]) / len(ok_with_corr)

    ref_acc = _acc(ref_records)
    cmp_acc = _acc(cmp_records)

    # Paired-sample agreement (matching item × sample_idx)
    paired_status_agree = 0
    paired_metric_agree = 0
    paired_metric_n = 0
    paired_total = 0
    for it in common:
        f_by_k = {r.get("sample_idx"): r for r in ref_by_item[it]}
        q_by_k = {r.get("sample_idx"): r for r in cmp_by_item[it]}
        for k in set(f_by_k) & set(q_by_k):
            paired_total += 1
            f = f_by_k[k]; q = q_by_k[k]
            if f.get("parse_status") == q.get("parse_status"):
                paired_status_agree += 1
            # Discrete metric agreement (committed pairs, discrete-metric)
            if (f.get("parse_status") == "committed"
                    and q.get("parse_status") == "committed"
                    and benchmark.metric_type == "discrete"):
                paired_metric_n += 1
                if f.get("metric") == q.get("metric"):
                    paired_metric_agree += 1
    paired_status_agreement = (
        paired_status_agree / paired_total if paired_total else float("nan"))
    paired_metric_agreement = (
        paired_metric_agree / paired_metric_n if paired_metric_n else None)

    return CrossConfigSummary(
        benchmark=benchmark.name,
        ref_label=ref_label,
        cmp_label=cmp_label,
        n_problems=len(common),
        ref_marginal=ref_marg, cmp_marginal=cmp_marg,
        status_tv=status_tv,
        per_item_status_tv_mean=pi_mean,
        per_item_status_tv_median=pi_median,
        per_item_status_tv_max=pi_max,
        ok_metric_distance=ok_metric_dist,
        ok_n_ref=len(ref_ok), ok_n_cmp=len(cmp_ok),
        gen_chars_wasserstein=gen_chars_w,
        ref_gen_chars_median=_median([float(x) for x in ref_lens]),
        cmp_gen_chars_median=_median([float(x) for x in cmp_lens]),
        ref_hit_eos_rate=ref_eos, cmp_hit_eos_rate=cmp_eos,
        hit_eos_rate_diff=eos_diff,
        ref_accuracy=ref_acc, cmp_accuracy=cmp_acc,
        paired_status_agreement=paired_status_agreement,
        paired_metric_agreement=paired_metric_agreement,
        paired_total=paired_total,
    )


def render_summary(summary: CrossConfigSummary) -> str:
    """Pretty-print a single cross-config summary."""
    s = summary
    lines = []
    lines.append(f"=== {s.benchmark}  ({s.ref_label} vs {s.cmp_label}) ===")
    lines.append(f"  n_problems: {s.n_problems}")
    lines.append(f"  status marginals (fraction of records):")
    statuses = [k for k in _STATUS_BUCKETS
                  if s.ref_marginal.get(k, 0) > 0 or s.cmp_marginal.get(k, 0) > 0]
    if statuses:
        header = "    " + "  ".join(f"{k:>11}" for k in statuses)
        lines.append(header)
        lines.append("    " + "  ".join(f"{s.ref_marginal.get(k, 0)*100:>10.1f}%"
                                           for k in statuses) + f"  ← {s.ref_label}")
        lines.append("    " + "  ".join(f"{s.cmp_marginal.get(k, 0)*100:>10.1f}%"
                                           for k in statuses) + f"  ← {s.cmp_label}")
    lines.append(f"  status_tv (marginal):       {s.status_tv:.4f}")
    lines.append(f"  per-item status TV:         "
                 f"mean={s.per_item_status_tv_mean:.4f}  "
                 f"median={s.per_item_status_tv_median:.4f}  "
                 f"max={s.per_item_status_tv_max:.4f}")
    lines.append(f"  committed-metric distance:  {s.ok_metric_distance:.4f}  "
                 f"(n_ref={s.ok_n_ref}, n_cmp={s.ok_n_cmp})")
    lines.append(f"  gen_chars Wasserstein:      {s.gen_chars_wasserstein:.1f}  "
                 f"(median: {s.ref_label}={s.ref_gen_chars_median:.0f}  "
                 f"{s.cmp_label}={s.cmp_gen_chars_median:.0f})")
    lines.append(f"  hit-EOS rate:               "
                 f"{s.ref_label}={s.ref_hit_eos_rate*100:.1f}%  "
                 f"{s.cmp_label}={s.cmp_hit_eos_rate*100:.1f}%  "
                 f"(diff {s.hit_eos_rate_diff:.4f})")
    if s.ref_accuracy is not None:
        lines.append(f"  accuracy (committed, vs gold):  "
                     f"{s.ref_label}={s.ref_accuracy*100:.1f}%  "
                     f"{s.cmp_label}={s.cmp_accuracy*100:.1f}%")
    lines.append(f"  paired-sample status agreement: "
                 f"{s.paired_status_agreement*100:.1f}% "
                 f"({s.paired_total} pairs)")
    if s.paired_metric_agreement is not None:
        lines.append(f"  paired-sample metric agreement: "
                     f"{s.paired_metric_agreement*100:.1f}%")
    return "\n".join(lines)
