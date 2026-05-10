"""Bridge-budget probe runner for the quant search.

A search "run" evaluates one or more configs. Each config gets one
"eval", and each eval issues a fixed number of "probes". The shape of
an eval is captured by:

    BudgetSpec(prefill_tokens_per_eval, ar_tokens_per_eval, n_probes)

These are EXPECTATIONS, not hard mid-task cutoffs — we size each probe
so the aggregate lands near the budget rather than chopping a probe in
half. Real consumption is read from the bridge's `usage` field on each
/v1/chat/completions response — the single source of truth shared with
all other harnesses in this module. No engine side-channels.

Per-probe accounting (from `usage`):
  prompt_tokens     — input length (incl. cache_hits)
  completion_tokens — AR tokens generated this probe
  cache_hits        — prefix tokens served from KV pages (no work paid)
  cache_misses      — prefix tokens this probe had to prefill itself.
                       This is the BILLED prefill — the bandwidth+ALU
                       cost the budget actually allocates against.

A 100k shared-prefix demo with N probes: probe 0 pays cache_misses≈100k,
probes 1..N-1 hit the engine prefix cache and pay cache_misses≈|suffix|.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# Defaults match harnesses.py — same bridge surface.
DEFAULT_BRIDGE_URL = "http://127.0.0.1:8001"


@dataclass
class BudgetSpec:
    """How big one config-eval should be (per-probe expectations)."""
    prefill_tokens_per_eval: int  # expected prompt_tokens per probe
    ar_tokens_per_eval: int       # expected completion_tokens per probe
    n_probes: int                 # how many probes per eval

    @property
    def expected_total_prompt_tokens(self) -> int:
        return self.prefill_tokens_per_eval * self.n_probes

    @property
    def expected_total_completion_tokens(self) -> int:
        return self.ar_tokens_per_eval * self.n_probes


@dataclass
class ProbeResult:
    probe_idx: int
    prompt_tokens: int
    completion_tokens: int
    cache_hits: int
    cache_misses: int          # = billed prefill
    elapsed_s: float
    response_text: str
    finish_reason: str

    @property
    def billed_prefill(self) -> int:
        return self.cache_misses


@dataclass
class BudgetReport:
    spec: BudgetSpec
    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(p.prompt_tokens for p in self.probes)

    @property
    def total_completion_tokens(self) -> int:
        return sum(p.completion_tokens for p in self.probes)

    @property
    def total_cache_hits(self) -> int:
        return sum(p.cache_hits for p in self.probes)

    @property
    def total_billed_prefill(self) -> int:
        return sum(p.billed_prefill for p in self.probes)

    @property
    def total_elapsed_s(self) -> float:
        return sum(p.elapsed_s for p in self.probes)

    def summary(self) -> str:
        s = self.spec
        lines = []
        lines.append("=" * 72)
        lines.append(f"BudgetReport — spec: {s.n_probes} probes, "
                     f"prefill≈{s.prefill_tokens_per_eval:,}/probe, "
                     f"AR≈{s.ar_tokens_per_eval:,}/probe")
        lines.append("-" * 72)
        lines.append(
            f"{'probe':>5} {'prompt':>10} {'completion':>10} "
            f"{'hits':>10} {'misses':>10} {'elapsed':>8}"
        )
        for p in self.probes:
            lines.append(
                f"{p.probe_idx:>5} {p.prompt_tokens:>10,} "
                f"{p.completion_tokens:>10,} {p.cache_hits:>10,} "
                f"{p.cache_misses:>10,} {p.elapsed_s:>7.1f}s"
            )
        lines.append("-" * 72)
        lines.append(
            f"{'TOTAL':>5} {self.total_prompt_tokens:>10,} "
            f"{self.total_completion_tokens:>10,} {self.total_cache_hits:>10,} "
            f"{self.total_billed_prefill:>10,} {self.total_elapsed_s:>7.1f}s"
        )
        lines.append("")
        lines.append(
            f"Expected vs actual:"
            f"  prefill {self.total_prompt_tokens:,}/"
            f"{s.expected_total_prompt_tokens:,}"
            f"  AR {self.total_completion_tokens:,}/"
            f"{s.expected_total_completion_tokens:,}"
        )
        lines.append(
            f"Billed-prefill ratio: "
            f"{self.total_billed_prefill / max(self.total_prompt_tokens, 1):.3f} "
            f"(1.0 = no prefix sharing; <1.0 = engine prefix cache helped)"
        )
        return "\n".join(lines)


def _post_chat_completions(
    bridge_url: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 1.0  # 0.0 is forbidden,
    timeout: int = 600,
) -> dict[str, Any]:
    """One /v1/chat/completions roundtrip; returns parsed JSON verbatim."""
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{bridge_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_budget_probes(
    spec: BudgetSpec,
    prefix_messages: list[dict],
    probe_suffixes: list[str],
    bridge_url: str = DEFAULT_BRIDGE_URL,
    print_progress: bool = True,
) -> BudgetReport:
    """Issue `spec.n_probes` chat completions; account from `usage` only.

    Each probe sends `prefix_messages + [{role: user, content: suffix_i}]`
    with `max_tokens = spec.ar_tokens_per_eval`. Designed for a shared-
    prefix demo: identical `prefix_messages` across probes lets the
    engine's content-hash KV cache serve probes 2..N's prefix from
    pages, leaving cache_misses ≈ |suffix| per probe after the first.
    """
    if len(probe_suffixes) != spec.n_probes:
        raise ValueError(
            f"len(probe_suffixes)={len(probe_suffixes)} != "
            f"spec.n_probes={spec.n_probes}"
        )

    report = BudgetReport(spec=spec)
    for i, suffix in enumerate(probe_suffixes):
        msgs = list(prefix_messages) + [{"role": "user", "content": suffix}]
        t0 = time.time()
        resp = _post_chat_completions(
            bridge_url, msgs, max_tokens=spec.ar_tokens_per_eval)
        elapsed = time.time() - t0

        usage = resp.get("usage") or {}
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""

        result = ProbeResult(
            probe_idx=i,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cache_hits=int(usage.get("cache_hits", 0)),
            cache_misses=int(usage.get("cache_misses", 0)),
            elapsed_s=elapsed,
            response_text=text,
            finish_reason=str(choice.get("finish_reason") or ""),
        )
        report.probes.append(result)
        if print_progress:
            print(
                f"[probe {i}] prompt={result.prompt_tokens:,} "
                f"completion={result.completion_tokens} "
                f"hits={result.cache_hits:,} misses={result.cache_misses:,} "
                f"in {elapsed:.1f}s",
                flush=True,
            )

    return report
