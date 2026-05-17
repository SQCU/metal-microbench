"""Dataset loaders for the quant-search harness battery.

Reconstructed from `oracles.cpython-312.pyc` after the original `oracles.py`
was deleted as part of the 2026-05-05 quant-search cleanup. The originals
were lazy-imported by `MMLUHarness`, `GSM8KHarness`, and `SVGMSEHarness`.

Each loader downloads the HuggingFace parquet shard once, parses it with
pyarrow, caches a JSONL on disk, and returns rows as plain dicts. The
`datasets` library is deliberately NOT a runtime dependency — only
`pyarrow` is needed to parse the initial parquet, and even that is on the
cold path.

`REPO_ROOT` is the metal-microbench repo root, three parents up from this
file (tools/quant_search/data_loaders.py → tools/quant_search → tools →
repo). `DATA_CACHE` defaults to `<repo>/tools/quant_search/data_cache` and
can be overridden by the `QUANT_DATA_CACHE` env var.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CACHE = Path(os.environ.get(
    "QUANT_DATA_CACHE",
    str(REPO_ROOT / "tools" / "quant_search" / "data_cache"),
))


def _load_mmlu_subset(n_samples: int = 20_000) -> list[dict]:
    """Return n_samples MMLU questions, each as {question, choices, answer, subject}.

    Direct download of the HF Hub parquet, parsed with pyarrow (only
    quality dependency — no `datasets` library). Caches as JSONL after
    first parse so subsequent calls don't even touch parquet.
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_CACHE / "mmlu_test.jsonl"

    if not jsonl_path.exists():
        parquet_url = ("https://huggingface.co/datasets/cais/mmlu/resolve/"
                       "main/all/test-00000-of-00001.parquet")
        parquet_path = DATA_CACHE / "mmlu_test.parquet"
        urllib.request.urlretrieve(parquet_url, str(parquet_path))

        import pyarrow.parquet as parquet
        table = parquet.read_table(str(parquet_path))
        with open(jsonl_path, "w") as f:
            for batch in table.to_batches():
                d = batch.to_pydict()
                n = len(d["question"])
                for i in range(n):
                    row = {
                        "question": d["question"][i],
                        "choices": list(d["choices"][i]),
                        "answer": int(d["answer"][i]),
                        "subject": d["subject"][i],
                    }
                    f.write(json.dumps(row) + "\n")
        parquet_path.unlink()

    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n_samples:
                break
    return rows


def _load_hellaswag_subset(n_samples: int = 20_000) -> list[dict]:
    """Return n_samples HellaSwag validation items, each as
    {ctx, endings, label}. `endings` is a 4-string list; `label` is the
    int index 0..3 of the gold continuation.

    Uses the validation split (test labels are hidden). Direct HF Hub
    parquet → JSONL cache, no `datasets` library at runtime.
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_CACHE / "hellaswag_validation.jsonl"

    if not jsonl_path.exists():
        parquet_url = ("https://huggingface.co/datasets/Rowan/hellaswag/"
                       "resolve/main/data/validation-00000-of-00001.parquet")
        parquet_path = DATA_CACHE / "hellaswag_validation.parquet"
        urllib.request.urlretrieve(parquet_url, str(parquet_path))

        import pyarrow.parquet as parquet
        table = parquet.read_table(str(parquet_path))
        with open(jsonl_path, "w") as f:
            for batch in table.to_batches():
                d = batch.to_pydict()
                n = len(d["ctx"])
                for i in range(n):
                    row = {
                        "ctx": d["ctx"][i],
                        "endings": list(d["endings"][i]),
                        "label": int(d["label"][i]),
                    }
                    f.write(json.dumps(row) + "\n")
        parquet_path.unlink()

    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n_samples:
                break
    return rows


def _load_triviaqa_subset(n_samples: int = 20_000) -> list[dict]:
    """Return n_samples TriviaQA validation items, each as
    {question, answer, aliases}. `answer` is the canonical answer
    string; `aliases` is a list of acceptable variant strings (gold
    answer set). Direct HF-Hub parquet → JSONL cache.

    Uses the rc.nocontext config (no Wikipedia paragraph), since we
    want to test factual recall not reading-comprehension.
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_CACHE / "triviaqa_rc_nocontext_validation.jsonl"

    if not jsonl_path.exists():
        parquet_url = (
            "https://huggingface.co/datasets/mandarjoshi/trivia_qa/"
            "resolve/main/rc.nocontext/validation-00000-of-00001.parquet"
        )
        parquet_path = DATA_CACHE / "triviaqa_validation.parquet"
        urllib.request.urlretrieve(parquet_url, str(parquet_path))

        import pyarrow.parquet as parquet
        table = parquet.read_table(str(parquet_path))
        with open(jsonl_path, "w") as f:
            for batch in table.to_batches():
                d = batch.to_pydict()
                n = len(d["question"])
                for i in range(n):
                    ans = d["answer"][i]
                    # `answer` is a struct: {value, aliases, ...}
                    canonical = (ans.get("value") if isinstance(ans, dict)
                                 else None) or ""
                    aliases_raw = (ans.get("aliases") if isinstance(ans, dict)
                                    else None) or []
                    aliases = [a for a in aliases_raw if a]
                    if not canonical and aliases:
                        canonical = aliases[0]
                    if not canonical:
                        continue        # malformed row, skip
                    row = {
                        "question": d["question"][i],
                        "answer":   canonical,
                        "aliases":  aliases,
                    }
                    f.write(json.dumps(row) + "\n")
        parquet_path.unlink()

    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n_samples:
                break
    return rows


def _load_humaneval_subset(n_samples: int = 20_000) -> list[dict]:
    """Return n_samples HumanEval problems, each as
    {task_id, prompt, canonical_solution, test, entry_point}.

    `prompt` is the function signature + docstring. `test` is a Python
    string containing `check(candidate)` that asserts correctness. The
    benchmark's runner concatenates (prompt + model_completion + test +
    'check({entry_point})') and execs in subprocess.
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_CACHE / "humaneval_test.jsonl"

    if not jsonl_path.exists():
        # Two URL forms exist for openai_humaneval; try them in order.
        candidate_urls = [
            ("https://huggingface.co/datasets/openai/openai_humaneval/"
             "resolve/main/openai_humaneval/test-00000-of-00001.parquet"),
            ("https://huggingface.co/datasets/openai_humaneval/"
             "resolve/main/openai_humaneval/test-00000-of-00001.parquet"),
        ]
        parquet_path = DATA_CACHE / "humaneval_test.parquet"
        last_err: Exception | None = None
        for url in candidate_urls:
            try:
                urllib.request.urlretrieve(url, str(parquet_path))
                last_err = None
                break
            except Exception as e:                   # noqa: BLE001
                last_err = e
        if last_err is not None:
            raise last_err

        import pyarrow.parquet as parquet
        table = parquet.read_table(str(parquet_path))
        with open(jsonl_path, "w") as f:
            for batch in table.to_batches():
                d = batch.to_pydict()
                n = len(d["task_id"])
                for i in range(n):
                    row = {
                        "task_id":            d["task_id"][i],
                        "prompt":             d["prompt"][i],
                        "canonical_solution": d.get("canonical_solution", [""] * n)[i],
                        "test":               d["test"][i],
                        "entry_point":        d["entry_point"][i],
                    }
                    f.write(json.dumps(row) + "\n")
        parquet_path.unlink()

    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n_samples:
                break
    return rows


def _load_gsm8k_subset(n_samples: int = 20_000) -> list[dict]:
    """Return n_samples GSM8K test items, each as {question, answer}.

    `answer` is the original chain-of-thought + `#### N` final answer;
    the harness extracts the gold numeric answer from the trailing
    `#### N` marker. Direct HF-Hub parquet → JSONL cache, no `datasets`
    library at runtime.
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_CACHE / "gsm8k_test.jsonl"

    if not jsonl_path.exists():
        parquet_url = ("https://huggingface.co/datasets/openai/gsm8k/resolve/"
                       "main/main/test-00000-of-00001.parquet")
        parquet_path = DATA_CACHE / "gsm8k_test.parquet"
        urllib.request.urlretrieve(parquet_url, str(parquet_path))

        import pyarrow.parquet as parquet
        table = parquet.read_table(str(parquet_path))
        with open(jsonl_path, "w") as f:
            for batch in table.to_batches():
                d = batch.to_pydict()
                n = len(d["question"])
                for i in range(n):
                    row = {
                        "question": d["question"][i],
                        "answer": d["answer"][i],
                    }
                    f.write(json.dumps(row) + "\n")
        parquet_path.unlink()

    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n_samples:
                break
    return rows
