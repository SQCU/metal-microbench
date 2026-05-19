"""Harnesses — quant-search benchmarks ported to the workload model.

A harness is a benchmark. Each iterates its full real dataset, emits
one work ticket per benchmark task, scores responses, and reports
metrics. None of them know about budgets, none truncate. Pull tickets
from a harness until it stops emitting and you have run the entire
benchmark.

EVERY harness talks to the model through the SAME public OAI-shape
chat API every production client uses:
  - /v1/chat/completions  — chat-shape (model sees role-wrapped
                             messages, generates a reply)

The supported harness battery is exactly {MMLUHarness, GSM8KHarness,
SVGMSEHarness, TokSHarness}. All four are chat-shape; harnesses send
messages directly and NEVER read engine internals.

Per-harness contract (from workload.py):
    async def run(
        inbox:  Queue[CompletedTicket | None],
        outbox: Queue[WorkTicket],
        stop:   asyncio.Event,
    ) -> dict
"""
from __future__ import annotations

import asyncio
import os
import statistics
import uuid
from pathlib import Path

from workload import CompletedTicket, Harness, HarnessBudget, WorkTicket


# Where the standalone toolcards runner (04_toolcards_runner.mjs) listens.
TOOLCARDS_URL = os.environ.get("TOOLCARDS_URL", "http://127.0.0.1:8002")


# ──────────────────────────────────────────────────────────────────────
# MMLU harness — full test split, /v1/chat/completions
# ──────────────────────────────────────────────────────────────────────


def _format_mmlu_prompt(row: dict) -> str:
    q = row["question"]
    c = row["choices"]
    return (
        f"Question: {q}\n"
        f"A) {c[0]}\n"
        f"B) {c[1]}\n"
        f"C) {c[2]}\n"
        f"D) {c[3]}\n"
        f"Answer:"
    )


def _score_mmlu_reply(
    row: dict, completed: CompletedTicket
) -> tuple[float, bool] | None:
    if completed.response is None:
        return None
    try:
        choices = completed.response["choices"]
        content = choices[0]["logprobs"]["content"]
        top = {item["token"].strip().upper(): item["logprob"]
               for item in content[0]["top_logprobs"]}
    except (KeyError, IndexError, TypeError):
        return None
    letters = ["A", "B", "C", "D"]
    lps = [top.get(L, -20.0) for L in letters]
    for i, L in enumerate(letters):
        alt = top.get(" " + L, top.get(L.lower(), None))
        if alt is not None:
            lps[i] = max(lps[i], alt)
    ans_idx = int(row["answer"])
    correct_lp = lps[ans_idx]
    max_other = max(lp for i, lp in enumerate(lps) if i != ans_idx)
    return (correct_lp - max_other, correct_lp >= max(lps))


class MMLUHarness:
    name = "mmlu"

    def __init__(self, top_logprobs: int = 10) -> None:
        self.top_logprobs = top_logprobs
        self.margins: list[float] = []
        self.correct: int = 0
        self.by_subject: dict[str, list[float]] = {}
        self.n_skipped: int = 0
        self.n_emitted: int = 0

    async def run(self, inbox, outbox, stop):
        from data_loaders import _load_mmlu_subset
        rows = await asyncio.to_thread(_load_mmlu_subset, n_samples=20_000)
        pending: dict[str, dict] = {}
        for row in rows:
            if stop.is_set():
                break
            t = WorkTicket(
                issuer=self.name,
                endpoint="/v1/chat/completions",
                body={
                    "messages": [{"role": "user",
                                   "content": _format_mmlu_prompt(row)}],
                    "max_tokens": 1,
                    "temperature": 0.7,
                    "logprobs": True,
                    "top_logprobs": self.top_logprobs,
                },
                meta={"subject": row["subject"]},
            )
            pending[t.ticket_id] = row
            await outbox.put(t)
            self.n_emitted += 1
            while True:
                try:
                    completed = inbox.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if completed is None:
                    return self._finalize()
                self._consume(completed, pending)
        await outbox.put(None)
        while True:
            completed = await inbox.get()
            if completed is None:
                return self._finalize()
            self._consume(completed, pending)

    def _consume(self, completed: CompletedTicket,
                  pending: dict[str, dict]) -> None:
        row = pending.pop(completed.ticket.ticket_id, None)
        if row is None:
            self.n_skipped += 1
            return
        scored = _score_mmlu_reply(row, completed)
        if scored is None:
            self.n_skipped += 1
            return
        margin, was_correct = scored
        self.margins.append(margin)
        if was_correct:
            self.correct += 1
        self.by_subject.setdefault(row["subject"], []).append(margin)

    def _finalize(self) -> dict:
        if not self.margins:
            return {"mmlu_argmax_acc": 0.0, "n_evaluated": 0,
                     "n_emitted": self.n_emitted}
        return {
            "mmlu_argmax_acc": self.correct / len(self.margins),
            "mmlu_margin_mean": statistics.mean(self.margins),
            "mmlu_margin_by_subject": {
                s: statistics.mean(v) for s, v in self.by_subject.items()
            },
            "n_evaluated": len(self.margins),
            "n_skipped": self.n_skipped,
            "n_emitted": self.n_emitted,
        }


# ──────────────────────────────────────────────────────────────────────
# GSM8K harness — full test split, /v1/chat/completions multi-turn few-shot
# ──────────────────────────────────────────────────────────────────────


# GSM8K extraction — see docs/regex_replacement_plans/
# gsm8k_predicted_extraction.md for the structurally better fix
# (ask the LLM judge to extract the final numeric answer). The
# patterns below are pure-Python finite-state scanners equivalent
# to the prior regex pile; they have no regex dependency.

def _scan_gsm8k_number(text: str, pos: int) -> tuple[int, str] | None:
    """At `text[pos]`, try to match the GSM8K number shape
        -? \\d [\\d,]* \\.? \\d*
    i.e. an optional minus, a leading digit, more digits-and-commas,
    optional dot, optional trailing digits. Returns (end_pos, value)
    or None if no match. `value` has commas stripped.
    """
    n = len(text)
    i = pos
    if i >= n:
        return None
    sign_len = 0
    if text[i] == "-":
        sign_len = 1
        i += 1
        if i >= n or not text[i].isdigit():
            return None
    if not text[i].isdigit():
        return None
    j = i + 1
    while j < n and (text[j].isdigit() or text[j] == ","):
        j += 1
    # Drop a trailing comma (numbers like "1,234,").
    while j > i and text[j - 1] == ",":
        j -= 1
    if j < n and text[j] == ".":
        k = j + 1
        while k < n and text[k].isdigit():
            k += 1
        if k > j + 1:
            j = k
    value = text[pos:j].replace(",", "")
    if not value or value == "-":
        return None
    return j, value


def _gsm8k_find_after(text: str, marker: str, *,
                       case_insensitive: bool = False,
                       require_dollar_prefix: bool = False,
                       skip_extra: tuple[str, ...] = ()) -> str | None:
    """Find `marker` in `text`, skip optional whitespace (and an optional
    '$' if `require_dollar_prefix` permits one), then try to parse a
    GSM8K-shaped number. `skip_extra` is a tuple of literal sub-tokens
    that may appear after the marker before the number (case-folded if
    `case_insensitive`). Returns the parsed number or None.
    """
    haystack = text.lower() if case_insensitive else text
    needle = marker.lower() if case_insensitive else marker
    pos = 0
    while True:
        idx = haystack.find(needle, pos)
        if idx < 0:
            return None
        i = idx + len(needle)
        # Skip any whitespace.
        while i < len(text) and text[i] in " \t\n":
            i += 1
        # Skip allowed extras (e.g. "is", ":") followed by whitespace.
        for extra in skip_extra:
            ex = extra.lower() if case_insensitive else extra
            chunk = haystack[i:i + len(ex)]
            if chunk == ex:
                i += len(ex)
                while i < len(text) and text[i] in " \t\n":
                    i += 1
                break
        if require_dollar_prefix and i < len(text) and text[i] == "$":
            i += 1
        parsed = _scan_gsm8k_number(text, i)
        if parsed is not None:
            return parsed[1]
        pos = idx + 1


def _gsm8k_find_boxed(text: str) -> str | None:
    """Match `\\boxed{<number>}`."""
    needle = r"\boxed{"
    pos = 0
    while True:
        idx = text.find(needle, pos)
        if idx < 0:
            return None
        inner = idx + len(needle)
        parsed = _scan_gsm8k_number(text, inner)
        if parsed is not None and parsed[0] < len(text) and text[parsed[0]] == "}":
            return parsed[1]
        pos = idx + 1


def _gsm8k_eq_at_eol(text: str) -> str | None:
    """Match the `=<ws>$?<number><ws>.?<ws>$` line-tail pattern, scanning
    every line independently. Original regex: re.MULTILINE.
    """
    for line in text.splitlines():
        pos = 0
        # The regex used .search, then anchored end with `\s*\.?\s*$`.
        # We walk every '=' occurrence in the line and check the tail.
        while True:
            idx = line.find("=", pos)
            if idx < 0:
                break
            i = idx + 1
            while i < len(line) and line[i] in " \t":
                i += 1
            if i < len(line) and line[i] == "$":
                i += 1
            parsed = _scan_gsm8k_number(line, i)
            if parsed is not None:
                tail = line[parsed[0]:]
                stripped = tail.strip().rstrip(".")
                if stripped == "":
                    return parsed[1]
            pos = idx + 1
    return None


def _gsm8k_find_all_numbers(text: str) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        # Match starts at digit or '-<digit>'.
        if text[i].isdigit() or (text[i] == "-" and i + 1 < n and text[i + 1].isdigit()):
            parsed = _scan_gsm8k_number(text, i)
            if parsed is not None:
                out.append(parsed[1])
                i = parsed[0]
                continue
        i += 1
    return out

_GSM8K_FEW_SHOT_DIALOG: list[tuple[str, str]] = [
    ("Janet's ducks lay 16 eggs per day. She eats three for breakfast "
     "every morning and bakes muffins for her friends every day with "
     "four. She sells the remainder at the farmers' market daily for "
     "$2 per fresh duck egg. How much in dollars does she make every "
     "day at the farmers' market?",
     "Janet eats 3 eggs and bakes with 4 eggs, so she has 16 - 3 - 4 "
     "= 9 eggs left to sell. At $2 per egg, she makes 9 * 2 = $18.\n"
     "#### 18"),
    ("A robe takes 2 bolts of blue fiber and half that much white "
     "fiber. How many bolts in total does it take?",
     "Half of 2 is 1, so it needs 1 bolt of white fiber. Total: "
     "2 + 1 = 3 bolts.\n#### 3"),
    ("James writes a 3-page letter to 2 different friends twice a "
     "week. How many pages does he write a year?",
     "Each week he writes 3 pages * 2 friends * 2 times = 12 pages. "
     "A year has 52 weeks, so 12 * 52 = 624 pages.\n#### 624"),
    ("Mark has a garden with flowers. He planted plants of three "
     "different colors in it. Ten of them are yellow, and there are "
     "80% more of those in purple. There are only 25% as many green "
     "flowers as there are yellow and purple flowers. How many "
     "flowers does Mark have in his garden?",
     "Yellow: 10. Purple: 10 + 80% of 10 = 10 + 8 = 18. Yellow + "
     "purple = 28. Green: 25% of 28 = 7. Total: 10 + 18 + 7 = 35.\n"
     "#### 35"),
]


def _gsm8k_extract_gold(answer_text: str) -> str | None:
    # GSM8K reference answers end with `#### <number>` on the last line.
    return _gsm8k_find_after(answer_text, "####")


def _gsm8k_extract_predicted(text: str) -> str | None:
    # Try each extraction strategy in turn, mirroring the prior list of
    # six compiled regex patterns. See
    # docs/regex_replacement_plans/gsm8k_predicted_extraction.md for the
    # structurally better fix (LLM judge).
    candidates = [
        _gsm8k_find_boxed(text),
        _gsm8k_find_after(text, "####"),
        _gsm8k_find_after(text, "final answer",
                            case_insensitive=True,
                            require_dollar_prefix=True,
                            skip_extra=("is", ":")),
        _gsm8k_find_after(text, "the answer is",
                            case_insensitive=True,
                            require_dollar_prefix=True),
        _gsm8k_find_after(text, "answer:",
                            case_insensitive=True,
                            require_dollar_prefix=True),
        _gsm8k_eq_at_eol(text),
    ]
    for c in candidates:
        if c is not None:
            return c
    nums = _gsm8k_find_all_numbers(text)
    if nums:
        return nums[-1]
    return None


def _gsm8k_numeric_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def _gsm8k_build_messages(question: str) -> list[dict]:
    msgs: list[dict] = []
    for q, a in _GSM8K_FEW_SHOT_DIALOG:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": question})
    return msgs


class GSM8KHarness:
    name = "gsm8k"

    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens
        self.correct: int = 0
        self.evaluated: int = 0
        self.n_emitted: int = 0
        self.n_unparsed: int = 0
        self._pending: dict[str, str] = {}

    async def run(self, inbox, outbox, stop):
        from data_loaders import _load_gsm8k_subset
        rows = await asyncio.to_thread(_load_gsm8k_subset, n_samples=20_000)
        for row in rows:
            if stop.is_set():
                break
            gold = _gsm8k_extract_gold(row["answer"])
            if gold is None:
                continue
            t = WorkTicket(
                issuer=self.name,
                endpoint="/v1/chat/completions",
                body={
                    "messages": _gsm8k_build_messages(row["question"]),
                    "max_tokens": self.max_tokens,
                    "temperature": 0.7,
                },
            )
            self._pending[t.ticket_id] = gold
            await outbox.put(t)
            self.n_emitted += 1
            while True:
                try:
                    completed = inbox.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if completed is None:
                    return self._finalize()
                self._consume(completed)
        await outbox.put(None)
        while True:
            completed = await inbox.get()
            if completed is None:
                return self._finalize()
            self._consume(completed)

    def _consume(self, completed: CompletedTicket) -> None:
        gold = self._pending.pop(completed.ticket.ticket_id, None)
        if gold is None or completed.response is None:
            return
        try:
            text = completed.response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            self.n_unparsed += 1
            self.evaluated += 1
            return
        pred = _gsm8k_extract_predicted(text)
        self.evaluated += 1
        if pred is None:
            self.n_unparsed += 1
            return
        if _gsm8k_numeric_equal(pred, gold):
            self.correct += 1

    def _finalize(self) -> dict:
        if self.evaluated == 0:
            return {"gsm8k_acc": 0.0, "n_evaluated": 0,
                     "n_emitted": self.n_emitted}
        return {
            "gsm8k_acc": self.correct / self.evaluated,
            "n_evaluated": self.evaluated,
            "n_correct": self.correct,
            "n_unparsed": self.n_unparsed,
            "unparsed_rate": self.n_unparsed / self.evaluated,
            "n_emitted": self.n_emitted,
        }


# ──────────────────────────────────────────────────────────────────────
# Tok/s harness — multi-stream throughput probe via /v1/chat/completions
# ──────────────────────────────────────────────────────────────────────


class TokSHarness:
    """Aggregate throughput probe at multiple concurrency levels.

    Fires N concurrent chat completions of fixed-length output, measures
    aggregate tokens/sec from `usage.completion_tokens` summed across
    completions divided by wall time. Sweeps across configured activeB
    values; reports per-activeB tok/s.
    """

    name = "tok_s"

    def __init__(
        self,
        active_b_values: list[int] | None = None,
        per_stream_max_tokens: int = 100,
        n_trials: int = 3,
        warmup: bool = True,
    ) -> None:
        self.active_b_values = active_b_values or [1, 4, 8]
        self.per_stream_max_tokens = per_stream_max_tokens
        self.n_trials = n_trials
        self.warmup = warmup
        self.results: dict[int, dict] = {}
        self.n_emitted = 0
        # Track tickets in-flight per (activeB, trial) batch.
        self._batches: dict[str, dict] = {}
        # Active batch state.
        self._current_batch_id: str | None = None
        self._current_pending: int = 0
        self._current_started_at: float = 0.0
        self._current_completion_tokens: int = 0
        self._current_results: list[dict] = []

    async def run(self, inbox, outbox, stop):
        import time
        # Optional warmup at the largest activeB.
        if self.warmup and not stop.is_set():
            await self._run_one_batch(
                max(self.active_b_values), inbox, outbox, stop, is_warmup=True)
        for nb in self.active_b_values:
            if stop.is_set():
                break
            samples: list[float] = []
            for trial in range(self.n_trials):
                if stop.is_set():
                    break
                tps = await self._run_one_batch(nb, inbox, outbox, stop)
                if tps is not None:
                    samples.append(tps)
            if samples:
                self.results[nb] = {
                    "tok_s_median": statistics.median(samples),
                    "tok_s_trials": samples,
                    "tok_s_stdev": (statistics.stdev(samples)
                                     if len(samples) > 1 else 0.0),
                }
        await outbox.put(None)
        while True:
            completed = await inbox.get()
            if completed is None:
                return self._finalize()
            # Late stragglers — ignore.

    async def _run_one_batch(
        self, n: int, inbox, outbox, stop, is_warmup: bool = False
    ) -> float | None:
        import time
        prompt = "Tell me a short story about a dragon and a knight."
        ticket_ids: list[str] = []
        t0 = time.time()
        for _ in range(n):
            t = WorkTicket(
                issuer=self.name,
                endpoint="/v1/chat/completions",
                body={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.per_stream_max_tokens,
                    "temperature": 0.7,
                },
            )
            ticket_ids.append(t.ticket_id)
            await outbox.put(t)
            self.n_emitted += 1
        completion_tokens_total = 0
        outstanding = set(ticket_ids)
        while outstanding and not stop.is_set():
            completed = await inbox.get()
            if completed is None:
                return None
            tid = completed.ticket.ticket_id
            if tid not in outstanding:
                continue
            outstanding.discard(tid)
            completion_tokens_total += completed.completion_tokens
        elapsed = time.time() - t0
        if elapsed <= 0:
            return None
        if is_warmup:
            return None
        return completion_tokens_total / elapsed

    def _finalize(self) -> dict:
        return {
            "tok_s_by_active_b": self.results,
            "n_emitted": self.n_emitted,
        }


# ──────────────────────────────────────────────────────────────────────
# SVG-MSE harness — multi-turn vision refinement via toolcards runner
# ──────────────────────────────────────────────────────────────────────


_SVG_DEFAULT_PROMPTS: list[str] = [
    "a smiley face",
    "a red circle on white background",
    "three concentric squares",
    "a simple house with a triangular roof",
    "a yellow sun with rays",
]


def _svg_slug(prompt: str) -> str:
    return "_".join(prompt.lower().split())[:40]


class _SentinelReceived(Exception):
    pass


class SVGMSEHarness:
    """Multi-turn image-→-SVG-→-MSE refinement via the toolcards runner.

    For each prompt:
      1. POST start_invoke to the toolcards runner — it does the
         multi-turn refinement loop internally, calling our bridge's
         /v1/chat/completions for each turn.
      2. Poll /sessions until our session_id has a result.
      3. Decode the rendered PNG and compute pixel MSE vs the saved
         reference image.

    All bridge work goes through /v1/chat/completions (toolcards →
    bridge). Harness is a client of the toolcards runner, which is a
    client of the bridge — all standard public APIs.
    """

    name = "svg_mse"

    def __init__(
        self,
        prompts: list[str] | None = None,
        max_iters: int = 3,
        width: int = 256,
        height: int = 256,
        refs_dir: Path | None = None,
        poll_interval_s: float = 2.0,
        per_prompt_deadline_s: float = 600.0,
    ) -> None:
        self.prompts = prompts if prompts is not None else _SVG_DEFAULT_PROMPTS
        self.max_iters = max_iters
        self.width = width
        self.height = height
        if refs_dir is None:
            from data_loaders import REPO_ROOT
            refs_dir = REPO_ROOT / "test_data" / "svg_quant_refs"
        self.refs_dir = refs_dir
        self.poll_interval_s = poll_interval_s
        self.per_prompt_deadline_s = per_prompt_deadline_s
        self.per_prompt: list[dict] = []
        self.n_emitted = 0
        self.bridge_url = os.environ.get(
            "QUANT_BRIDGE_URL", "http://127.0.0.1:8001")
        # Default model name: neutral identifier, not a uniform-quant
        # label. Override via QUANT_MODEL_NAME to match the active
        # config (e.g. "gemma-4-a4b-fp16", "gemma-4-a4b-unsloth_dyn").
        self.model_name = os.environ.get(
            "QUANT_MODEL_NAME", "gemma-4-a4b")

    async def run(self, inbox, outbox, stop):
        for prompt in self.prompts:
            if stop.is_set():
                break
            try:
                row = await self._eval_one_prompt(prompt, inbox, outbox, stop)
            except _SentinelReceived:
                return self._finalize()
            self.per_prompt.append(row)
        await outbox.put(None)
        while True:
            completed = await inbox.get()
            if completed is None:
                return self._finalize()

    async def _eval_one_prompt(
        self, prompt: str, inbox, outbox, stop,
    ) -> dict:
        chat_id = f"qsearch_{uuid.uuid4().hex[:12]}"
        slug = _svg_slug(prompt)
        ref_path = self.refs_dir / f"{slug}.png"
        start_ticket = WorkTicket(
            issuer=self.name,
            endpoint="/api/plugins/toolcards/start_invoke/query-to-svg/generate",
            body={
                "args": {
                    "query": prompt,
                    "max_iters": self.max_iters,
                    "width": self.width,
                    "height": self.height,
                },
                "profile": {
                    "chat_completion_source": "custom",
                    "custom_url": self.bridge_url,
                    "openai_model": self.model_name,
                    "temperature": 0.7,
                },
                "chat_id": chat_id,
            },
            base_url=TOOLCARDS_URL,
        )
        await outbox.put(start_ticket)
        self.n_emitted += 1
        completed = await self._await_my_reply(inbox, start_ticket.ticket_id)
        if completed is None:
            return {"prompt": prompt, "mse": None,
                     "error": "start_invoke failed"}
        resp = completed.response or {}
        session_id = resp.get("session_id")
        if not session_id:
            return {"prompt": prompt, "mse": None,
                     "error": f"no session_id: {resp}"}

        import time
        deadline = time.time() + self.per_prompt_deadline_s
        png_url: str | None = None
        tool_error: str | None = None
        while time.time() < deadline:
            if stop.is_set():
                return {"prompt": prompt, "mse": None,
                         "error": "stop event during polling"}
            poll_ticket = WorkTicket(
                issuer=self.name,
                endpoint=f"/api/plugins/toolcards/sessions?chat_id={chat_id}",
                body={},
                base_url=TOOLCARDS_URL,
            )
            await outbox.put(poll_ticket)
            self.n_emitted += 1
            completed = await self._await_my_reply(inbox, poll_ticket.ticket_id)
            if completed is None:
                return {"prompt": prompt, "mse": None,
                         "error": "poll failed"}
            results = (completed.response or {}).get("results") or []
            for r in results:
                if r.get("session_id") != session_id:
                    continue
                if not r.get("ok"):
                    tool_error = r.get("error") or "tool reported failure"
                    break
                payload = r.get("result") or {}
                if isinstance(payload.get("rendered_png_url"), str):
                    png_url = payload["rendered_png_url"]
                    break
                for part in payload.get("embed", []):
                    if part.get("type") == "image_url":
                        png_url = (part.get("image_url") or {}).get("url")
                        if png_url:
                            break
                if png_url is None and tool_error is None:
                    tool_error = "no rendered_png_url in result"
                break
            if png_url is not None or tool_error is not None:
                break
            await asyncio.sleep(self.poll_interval_s)
        if png_url is None:
            return {"prompt": prompt, "mse": None,
                     "error": tool_error or "deadline exceeded"}
        if not ref_path.exists():
            return {"prompt": prompt, "mse": None,
                     "error": f"no_ref: {ref_path}"}
        mse = await asyncio.to_thread(_svg_compute_mse, png_url, ref_path)
        return {"prompt": prompt, "mse": mse,
                 "iters_run": self.max_iters}

    async def _await_my_reply(self, inbox, ticket_id):
        while True:
            item = await inbox.get()
            if item is None:
                raise _SentinelReceived()
            if item.ticket.ticket_id == ticket_id:
                return item

    def _finalize(self) -> dict:
        valid = [r for r in self.per_prompt if r.get("mse") is not None]
        if not valid:
            return {"svg_mse_mean": float("inf"),
                     "n_evaluated": 0,
                     "n_total": len(self.per_prompt),
                     "per_prompt": self.per_prompt,
                     "n_emitted": self.n_emitted}
        return {
            "svg_mse_mean": statistics.mean(r["mse"] for r in valid),
            "n_evaluated": len(valid),
            "n_total": len(self.per_prompt),
            "per_prompt": self.per_prompt,
            "n_emitted": self.n_emitted,
        }


def _svg_compute_mse(png_data_url: str, ref_path: Path) -> float:
    import base64, io
    import numpy as np
    from PIL import Image
    if not png_data_url.startswith("data:image/"):
        raise ValueError("rendered PNG is not a data URL")
    b64 = png_data_url.split(",", 1)[1]
    gen = Image.open(io.BytesIO(base64.b64decode(b64)))
    ref = Image.open(ref_path)
    if gen.size != ref.size:
        gen = gen.resize(ref.size, Image.LANCZOS)
    gen_arr = np.asarray(gen.convert("RGB"), dtype=np.float32) / 255.0
    ref_arr = np.asarray(ref.convert("RGB"), dtype=np.float32) / 255.0
    return float(((gen_arr - ref_arr) ** 2).mean())
