"""Workload orchestrator for the quant search.

Architecture:
  - A Harness is a benchmark — MMLU, GSM8K, tokens-per-second probe,
    multi-turn SVG MSE. It iterates the FULL real dataset, emitting
    one WorkTicket per benchmark task. Harnesses do not know about
    budgets, do not size themselves, do not truncate. If you keep
    pulling tickets from a harness, eventually you have run the entire
    benchmark.
  - An Orchestrator is the external governor. It owns one pump-worker
    pool of FIXED size (matching the engine's B=8 kernel zoo cell), one
    merger that round-robins per-harness outboxes into the dispatch
    queue, and per-harness counters that sum prompt+completion tokens
    from each `usage` field. When a harness exceeds its configured
    token budget, the orchestrator stops dispatching that harness's
    tickets and signals the harness to stop emitting + drain.

Hard rules:
  - The pump never calls harness code synchronously.
  - No timeouts on bridge HTTP — server-paced.
  - Harnesses run on their own coroutines; CPU-heavy work goes to_thread
    so a slow harness can't stall the wire or other harnesses.
  - Pump pool size is hardcoded at construction. The engine's kernel zoo
    compiles a separate cell per `B_TILE ∈ {1, 2, 4, 8}`; ramping the
    pool from 1 measures the wrong cell. See
    docs/quant_search_motivation.md.
  - Each harness's finalized result is delivered via `harness_finalized_cb`
    immediately when its `run()` coroutine returns, so long-run drivers
    can persist incrementally and survive mid-run kills.

Per-harness contract:
    async def run(
        inbox:  Queue[CompletedTicket | None],   # None sentinel = done
        outbox: Queue[WorkTicket],
        stop:   asyncio.Event,                    # set by orchestrator
                                                  # when budget exceeded
    ) -> dict
"""
from __future__ import annotations

import asyncio
import json
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol


DEFAULT_BRIDGE_URL = "http://127.0.0.1:8001"

# Pump-pool size is hardcoded to match the engine's B=8 kernel zoo cell.
# An adaptive saturation probe used to ramp this from 1 upward; that was
# methodologically wrong because the engine compiles separate kernel cells
# for B_TILE ∈ {1, 2, 4, 8}, and ramping from 1 measures the B=1 cell for
# most of any wall-clock window. Production runs at activeB ≈ 8, so we
# pin the pool there and never ramp. See docs/quant_search_motivation.md.
DEFAULT_TARGET_INFLIGHT = 8
DEFAULT_WINDOW_LOG_S = 6.0
DEFAULT_OUTBOX_MAXSIZE = 256   # backpressure when orchestrator pauses pulling


# ──────────────────────────────────────────────────────────────────────
# Ticket data model
# ──────────────────────────────────────────────────────────────────────


@dataclass
class WorkTicket:
    """One unit of work issued by a harness.

    `base_url` overrides the orchestrator's default bridge URL — used
    by harnesses that talk to a sibling service (e.g. SVG-MSE → the
    toolcards runner on a separate port). When None, the orchestrator's
    `bridge_url` is used. The pump worker is otherwise indifferent to
    which service the ticket targets; tickets are just (URL, endpoint,
    body) tuples it executes and routes the reply for.
    """
    issuer: str
    endpoint: str
    body: dict
    ticket_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    meta: dict = field(default_factory=dict)
    base_url: str | None = None


@dataclass
class CompletedTicket:
    ticket: WorkTicket
    response: dict | None       # parsed JSON; None on transport error
    error: str | None
    elapsed_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


# ──────────────────────────────────────────────────────────────────────
# Harness contract
# ──────────────────────────────────────────────────────────────────────


@dataclass
class HarnessBudget:
    """Per-harness token cap. The orchestrator stops dispatching tickets
    for a harness once these are reached. Values are interpreted as
    expectations — actual consumption may slightly overshoot because
    in-flight tickets at the moment we cross the budget still pay out.
    """
    prefill_tokens: int      # cumulative prompt_tokens from usage
    ar_tokens: int           # cumulative completion_tokens from usage


class Harness(Protocol):
    name: str

    async def run(
        self,
        inbox: "asyncio.Queue[CompletedTicket | None]",
        outbox: "asyncio.Queue[WorkTicket]",
        stop: asyncio.Event,
    ) -> dict:
        """Iterate the FULL benchmark. Emit one ticket per task to outbox.
        Consume replies from inbox; finalize when inbox yields None
        sentinel (orchestrator says: no more replies coming for you).
        Check `stop` between emissions for cooperative early-stop.
        Heavy CPU goes to_thread."""
        ...


# ──────────────────────────────────────────────────────────────────────
# Bridge client (no timeouts)
# ──────────────────────────────────────────────────────────────────────


def _post_blocking(bridge_url: str, endpoint: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{bridge_url}{endpoint}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:           # no timeout
        return json.loads(r.read())


def _get_blocking(bridge_url: str, endpoint: str) -> dict:
    """GET version of `_post_blocking` for endpoints that take query
    strings (e.g. toolcards `/sessions?chat_id=...`)."""
    req = urllib.request.Request(f"{bridge_url}{endpoint}", method="GET")
    with urllib.request.urlopen(req) as r:           # no timeout
        return json.loads(r.read())


async def _send_one(
    bridge_url: str, ticket: WorkTicket
) -> CompletedTicket:
    t0 = time.time()
    target_url = ticket.base_url or bridge_url
    # GET vs POST: tickets with empty `body` and a query-string in
    # `endpoint` are GET — toolcard runner's /sessions?chat_id=... is
    # the canonical case. Sniff to keep the harness API simple.
    try:
        if ticket.body == {} and "?" in ticket.endpoint:
            resp = await asyncio.to_thread(_get_blocking,
                                            target_url, ticket.endpoint)
        else:
            resp = await asyncio.to_thread(
                _post_blocking, target_url, ticket.endpoint, ticket.body)
    except Exception as e:                       # noqa: BLE001
        return CompletedTicket(
            ticket=ticket, response=None, error=str(e),
            elapsed_s=time.time() - t0,
        )
    elapsed = time.time() - t0
    usage = (resp.get("usage") or {}) if isinstance(resp, dict) else {}
    return CompletedTicket(
        ticket=ticket, response=resp, error=None, elapsed_s=elapsed,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        cache_hits=int(usage.get("cache_hits", 0)),
        cache_misses=int(usage.get("cache_misses", 0)),
    )


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _HarnessSlot:
    """Per-harness orchestrator state — counters, queues, signals.

    Budget accounting uses *committed* estimates, not actuals. When a
    ticket is about to be dispatched, the merger increments
    committed_prompt / committed_ar by an upper-bound estimate of its
    cost; the budget governor compares committed to budget and stops
    dispatch once exhausted. This prevents arbitrarily large overshoot
    when many tickets are in flight before any reply lands. The actual
    prompt/completion counts (from `usage`) are also tracked for the
    final consumption report and the saturation probe — they're
    descriptive, not enforcing.
    """
    name: str
    budget: HarnessBudget
    inbox: "asyncio.Queue[CompletedTicket | None]" = field(
        default_factory=asyncio.Queue)
    outbox: "asyncio.Queue[WorkTicket | None]" = field(
        default_factory=lambda: asyncio.Queue(
            maxsize=DEFAULT_OUTBOX_MAXSIZE))
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    # Committed-at-dispatch (governor-side):
    committed_prompt: int = 0
    committed_ar: int = 0
    # Actual-at-reply (descriptive):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    dispatched: int = 0
    completed: int = 0
    sentinel_sent: bool = False
    emit_done: bool = False         # harness pushed `None` to its outbox
    n_replies_dropped: int = 0      # tickets dropped post-stop


def _estimate_ticket_cost(ticket: WorkTicket) -> tuple[int, int]:
    """Upper-bound estimate of (prompt_tokens, completion_tokens) for
    one ticket. Used at dispatch time so the budget governor doesn't
    have to wait for the reply to know what was promised. The actual
    cost from `usage` is tracked separately for reporting.
    """
    body = ticket.body
    endpoint = ticket.endpoint
    if endpoint == "/v1/chat/completions":
        msgs = body.get("messages") or []
        prompt_chars = 0
        for m in msgs:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str):
                prompt_chars += len(c)
            elif isinstance(c, list):
                for part in c:
                    text = part.get("text") if isinstance(part, dict) else None
                    if isinstance(text, str):
                        prompt_chars += len(text)
        if not msgs:
            prompt_text = body.get("prompt") or ""
            if isinstance(prompt_text, str):
                prompt_chars = len(prompt_text)
        est_prompt = max(1, prompt_chars // 4)
        est_prompt += 16    # chat-template overhead (markers, BOS, etc)
        est_ar = int(body.get("max_tokens", 0))
        return (est_prompt, est_ar)
    if endpoint.startswith("/api/plugins/toolcards/start_invoke"):
        # SVG-MSE harness fires this at the toolcards runner; the runner
        # internally generates many bridge calls on our behalf. Charge a
        # per-invocation estimate so the budget governor can stop the
        # harness once it has consumed its allotted slice. The numbers
        # below are rough — they reflect typical multi-iter SVG generation
        # (max_iters × per-turn prefill/AR). Refine empirically.
        args = (body.get("args") or {}) if isinstance(body, dict) else {}
        max_iters = int(args.get("max_iters", 3))
        per_turn_prefill = 1500    # vision encode + prior turns
        per_turn_ar = 1500         # SVG output is verbose
        return (max_iters * per_turn_prefill, max_iters * per_turn_ar)
    if endpoint.startswith("/api/plugins/toolcards/sessions"):
        # Polling endpoint — no bridge work.
        return (0, 0)
    # Unknown endpoint: don't gate, but also don't undercount.
    return (0, 0)


@dataclass
class _LogWindow:
    """Rolling window of throughput observations, reset every
    `window_log_s`. Pure diagnostic — does not influence dispatch."""
    started_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    completed: int = 0

    def tokens_per_s(self) -> float:
        elapsed = max(time.time() - self.started_s, 0.001)
        return (self.prompt_tokens + self.completion_tokens) / elapsed


def _consumption_dict(slot: "_HarnessSlot") -> dict:
    """Extract a serializable consumption record from a harness slot.
    Used both by the per-harness completion callback and by the final
    bundle returned from `Orchestrator.run`."""
    return {
        # Committed = what the governor counted at dispatch.
        "committed_prompt": slot.committed_prompt,
        "committed_ar": slot.committed_ar,
        # Actual = what the bridge reported via usage.
        "prompt_tokens": slot.prompt_tokens,
        "completion_tokens": slot.completion_tokens,
        "cache_hits": slot.cache_hits,
        "cache_misses": slot.cache_misses,
        "dispatched": slot.dispatched,
        "completed": slot.completed,
        "n_replies_dropped": slot.n_replies_dropped,
        "stopped_by_budget": slot.stop.is_set(),
        "budget_prefill": slot.budget.prefill_tokens,
        "budget_ar": slot.budget.ar_tokens,
    }


class Orchestrator:
    def __init__(
        self,
        bridge_url: str = DEFAULT_BRIDGE_URL,
        target_inflight: int = DEFAULT_TARGET_INFLIGHT,
        window_log_s: float = DEFAULT_WINDOW_LOG_S,
        progress_cb: Callable[[str], None] | None = None,
        harness_finalized_cb: Callable[
            [str, "dict | BaseException", dict], None] | None = None,
    ) -> None:
        """
        target_inflight
            Size of the pump-worker pool. Hardcoded default of 8 matches
            the engine's B=8 kernel zoo cell. Do not ramp adaptively; see
            module docstring + docs/quant_search_motivation.md.
        window_log_s
            Cadence of diagnostic throughput windows. Pure logging — does
            not influence dispatch.
        progress_cb
            Per-line text callback for `[orch]` progress lines.
        harness_finalized_cb
            Called with `(harness_name, result_or_exception, consumption_dict)`
            as soon as each harness's `run()` coroutine returns. Used by
            long-run drivers to persist results incrementally so kills
            mid-run don't lose finalized harness results.
        """
        self.bridge_url = bridge_url
        self.target_inflight = max(target_inflight, 1)
        self.window_log_s = window_log_s
        self.progress_cb = progress_cb or (lambda _msg: None)
        self.harness_finalized_cb = harness_finalized_cb

        self._window: _LogWindow = _LogWindow(started_s=time.time())
        # (inflight, tok/s) per logging window — diagnostic record of
        # throughput stability across the run. Inflight is constant by
        # design; the column survives for backwards compatibility with
        # post-processing that expected `concurrency_history`.
        self.history: list[tuple[int, float]] = []

    def _log(self, msg: str) -> None:
        self.progress_cb(msg)

    async def run(
        self,
        harnesses: list[Harness],
        budgets: dict[str, HarnessBudget],
    ) -> dict[str, dict]:
        for h in harnesses:
            if h.name not in budgets:
                raise ValueError(f"missing budget for harness {h.name!r}")

        slots: dict[str, _HarnessSlot] = {
            h.name: _HarnessSlot(name=h.name, budget=budgets[h.name])
            for h in harnesses
        }

        # Spawn harness coroutines. Each is its own task; replies route
        # through its inbox queue. The pump never calls into harness code.
        for h in harnesses:
            s = slots[h.name]
            s.task = asyncio.create_task(
                h.run(s.inbox, s.outbox, s.stop),
                name=f"harness-{h.name}",
            )

        # Dispatch queue feeds the pump worker pool. Items are
        # (ticket, slot) — slot lets the pump update counters and route
        # the reply back to the right inbox.
        dispatch_q: "asyncio.Queue[tuple[WorkTicket, _HarnessSlot]]" = (
            asyncio.Queue(maxsize=DEFAULT_OUTBOX_MAXSIZE))

        pump_tasks: list[asyncio.Task[None]] = []
        run_evt = asyncio.Event()           # cleared when ALL harnesses done

        async def _pump_worker(worker_id: int) -> None:
            while not run_evt.is_set():
                try:
                    ticket, slot = await asyncio.wait_for(
                        dispatch_q.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                completed = await _send_one(self.bridge_url, ticket)
                slot.prompt_tokens += completed.prompt_tokens
                slot.completion_tokens += completed.completion_tokens
                slot.cache_hits += completed.cache_hits
                slot.cache_misses += completed.cache_misses
                slot.completed += 1
                # Probe-window counters (across all harnesses).
                self._window.prompt_tokens += completed.prompt_tokens
                self._window.completion_tokens += completed.completion_tokens
                self._window.completed += 1
                # Route reply unless stop is already past — a reply that
                # comes back after stop fired is still legitimate (we paid
                # for it before stop), so deliver it.
                await slot.inbox.put(completed)
                dispatch_q.task_done()

        # Pool size is fixed at construction time. No adaptive ramping —
        # the engine's kernel zoo specializes per `B_TILE`, and ramping
        # would measure the wrong cell. See module docstring.
        for i in range(self.target_inflight):
            pump_tasks.append(
                asyncio.create_task(_pump_worker(i), name=f"pump-{i}"))

        # ── Merger: pulls from each non-stopped harness outbox, into
        #            dispatch_q. Sets stop event when token budget hit.
        async def _merger() -> None:
            while not run_evt.is_set():
                progressed = False
                for slot in slots.values():
                    # Budget-governor check on COMMITTED (estimated)
                    # tokens. Catches budget exhaustion at dispatch time
                    # rather than waiting for replies, which prevents
                    # large overshoots from in-flight tickets.
                    if (
                        not slot.stop.is_set()
                        and (slot.committed_prompt >= slot.budget.prefill_tokens
                             or slot.committed_ar >= slot.budget.ar_tokens)
                    ):
                        slot.stop.set()
                        self._log(
                            f"[orch] budget reached for {slot.name}: "
                            f"committed prefill={slot.committed_prompt:,}/"
                            f"{slot.budget.prefill_tokens:,}  "
                            f"committed ar={slot.committed_ar:,}/"
                            f"{slot.budget.ar_tokens:,} → stop"
                        )
                    # Drain residual tickets from a stopped harness's
                    # outbox so the harness can unblock from a full
                    # outbox post-stop and proceed to its drain phase.
                    if slot.stop.is_set():
                        try:
                            while True:
                                stale = slot.outbox.get_nowait()
                                slot.outbox.task_done()
                                if stale is None:
                                    slot.emit_done = True
                                else:
                                    slot.n_replies_dropped += 1
                        except asyncio.QueueEmpty:
                            pass
                        continue
                    # Pull from this slot's outbox if available.
                    try:
                        ticket = slot.outbox.get_nowait()
                    except asyncio.QueueEmpty:
                        continue
                    if ticket is None:
                        # Harness signaled end-of-emission. Mark and
                        # let the sentinel-sender finalize once all
                        # in-flight replies have routed.
                        slot.emit_done = True
                        slot.outbox.task_done()
                        progressed = True
                        continue
                    # Estimate cost and gate on budget BEFORE dispatch.
                    est_p, est_a = _estimate_ticket_cost(ticket)
                    if (
                        slot.committed_prompt + est_p > slot.budget.prefill_tokens
                        or slot.committed_ar + est_a > slot.budget.ar_tokens
                    ):
                        # This ticket would overshoot. Don't dispatch.
                        # Stop will fire at the top of the next loop pass
                        # (committed already past or about to be past
                        # budget for this slot).
                        slot.stop.set()
                        slot.n_replies_dropped += 1
                        slot.outbox.task_done()
                        progressed = True
                        continue
                    slot.committed_prompt += est_p
                    slot.committed_ar += est_a
                    slot.dispatched += 1
                    await dispatch_q.put((ticket, slot))
                    slot.outbox.task_done()
                    progressed = True
                if not progressed:
                    await asyncio.sleep(0.01)

        merger_task = asyncio.create_task(_merger(), name="merger")

        # ── Sentinel sender: when a harness has emitted everything it
        #   will emit (its task is done OR stop is set) AND every
        #   dispatched ticket has been replied to, push None into its
        #   inbox so its drain loop exits cleanly.
        async def _sentinel_sender() -> None:
            while not run_evt.is_set():
                for slot in slots.values():
                    if slot.sentinel_sent:
                        continue
                    harness_emit_done = (
                        slot.emit_done or slot.stop.is_set()
                        or (slot.task is not None and slot.task.done())
                    )
                    if harness_emit_done and slot.completed >= slot.dispatched:
                        # Make extra-sure the merger finished any straggler
                        # outbox pulls before we send the sentinel.
                        try:
                            slot.outbox.get_nowait()
                            slot.n_replies_dropped += 1
                            slot.outbox.task_done()
                        except asyncio.QueueEmpty:
                            pass
                        await slot.inbox.put(None)
                        slot.sentinel_sent = True
                        self._log(
                            f"[orch] sentinel → {slot.name} "
                            f"(emitted={slot.dispatched + slot.n_replies_dropped}, "
                            f"dispatched={slot.dispatched}, "
                            f"completed={slot.completed})"
                        )
                # Done when every harness's task has returned.
                if all(slot.task is not None and slot.task.done()
                       for slot in slots.values()) and all(
                           slot.sentinel_sent for slot in slots.values()):
                    return
                await asyncio.sleep(0.05)

        sentinel_task = asyncio.create_task(
            _sentinel_sender(), name="sentinel")

        # ── Window logger: every window_log_s, emit a throughput line
        #   for visibility and append to history. Pure diagnostic — pool
        #   size is fixed; this loop never spawns or cancels workers.
        async def _window_logger() -> None:
            while not run_evt.is_set():
                await asyncio.sleep(self.window_log_s)
                if run_evt.is_set():
                    break
                tps = self._window.tokens_per_s()
                self.history.append((len(pump_tasks), tps))
                self._log(
                    f"[orch] window @ inflight={len(pump_tasks)}: "
                    f"{tps:.1f} tok/s "
                    f"({self._window.completed} tickets, "
                    f"{self._window.prompt_tokens + self._window.completion_tokens} tokens)"
                )
                self._window = _LogWindow(started_s=time.time())

        probe_task = asyncio.create_task(_window_logger(), name="window_log")

        # Await harness completions in finishing order; fire the
        # finalized-callback (if any) per completion so durable
        # persistence can ride harness boundaries rather than the whole
        # config eval. This matters for multi-hour multi-config runs:
        # losing a finalized harness's result because the run was killed
        # mid-config is unacceptable.
        task_to_name = {slot.task: name for name, slot in slots.items()}
        pending_tasks: set[asyncio.Task] = {
            slot.task for slot in slots.values() if slot.task is not None
        }
        results_by_name: dict[str, "dict | BaseException"] = {}
        while pending_tasks:
            done, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                name = task_to_name[t]
                try:
                    res: "dict | BaseException" = t.result()
                except BaseException as e:                  # noqa: BLE001
                    res = e
                results_by_name[name] = res
                if self.harness_finalized_cb is not None:
                    consumption = _consumption_dict(slots[name])
                    try:
                        self.harness_finalized_cb(name, res, consumption)
                    except Exception as cb_err:              # noqa: BLE001
                        self._log(
                            f"[orch] harness_finalized_cb({name}) "
                            f"raised: {cb_err!r}"
                        )

        # Tell the bookkeeping coroutines to wind down.
        run_evt.set()
        await dispatch_q.join()
        for t in pump_tasks:
            t.cancel()
        merger_task.cancel()
        sentinel_task.cancel()
        probe_task.cancel()
        for t in pump_tasks:
            try: await t
            except asyncio.CancelledError: pass
        for t in (merger_task, sentinel_task, probe_task):
            try: await t
            except asyncio.CancelledError: pass

        # Bundle final results + per-harness consumption.
        finals: dict[str, dict] = {}
        for name in slots.keys():
            slot = slots[name]
            res = results_by_name.get(name)
            consumption = _consumption_dict(slot)
            if isinstance(res, BaseException):
                finals[name] = {
                    "error": repr(res),
                    "consumption": consumption,
                }
            else:
                finals[name] = {
                    "metrics": res,
                    "consumption": consumption,
                }
        return finals
