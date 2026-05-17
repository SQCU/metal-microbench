"""Subprocess runner for HumanEval-style code-execution benchmarks.

The model produces a Python function body; we concatenate it with the
problem's test code and exec the whole thing in a subprocess with a
timeout. The exit code + stdout/stderr tell us pass/fail.

This is NOT a security sandbox — model-generated code can read the
filesystem and make network calls if it wants. For a benchmarking
context where we control what we feed the model and accept that it
might write to /tmp, this is acceptable. For untrusted inputs, run
under nsjail / a container instead.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _first_python_fenced_block(rollout: str) -> str | None:
    """Find the first ```{lang}\\n...``` block whose language tag is
    empty, 'python', or 'py' (case-insensitive). Plain-string
    replacement for the regex
        re.compile(r"```(?:python|py)?\\s*\\n(.*?)\\n```",
                    re.DOTALL | re.IGNORECASE).search
    Walks '```' boundaries: parts alternate non-code/code/non-code/code.
    """
    parts = rollout.split("```")
    for idx in range(1, len(parts), 2):
        block = parts[idx]
        nl = block.find("\n")
        if nl < 0:
            continue
        lang = block[:nl].strip().lower()
        if lang in ("", "python", "py"):
            inside = block[nl + 1:]
            # Drop a trailing newline matching the prior regex's `\n````
            # constraint (the regex required `\n` immediately before the
            # closing fence; here we get whatever comes before the next
            # '```' split-point, which already excludes the closing
            # fence — strip the trailing newline if present).
            if inside.endswith("\n"):
                inside = inside[:-1]
            return inside
    return None


def extract_python_code(rollout: str) -> str:
    """Pull the python code out of a model response. Prefers fenced
    blocks; falls back to the raw text if no fence found."""
    fenced = _first_python_fenced_block(rollout)
    if fenced is not None:
        return fenced.strip()
    # Strip common chat-template artifacts so the bare-text path doesn't
    # try to exec turn markers.
    text = rollout
    for marker in ["<turn|>", "<eos>", "<channel|>", "<|channel>"]:
        text = text.replace(marker, "")
    return text.strip()


@dataclass
class ExecResult:
    passed: bool
    timed_out: bool
    syntax_error: bool
    error_kind: str | None       # "syntax" / "timeout" / "runtime" / None
    stdout: str
    stderr: str


def run_with_test(prompt: str, completion: str, test: str,
                   entry_point: str, timeout_s: float = 10.0
                   ) -> ExecResult:
    """Run `prompt + completion + test + check(entry_point)` in a
    subprocess. Pass/fail is determined by exit code; non-zero → fail.

    HumanEval's test code defines a function `check(candidate)` that
    raises AssertionError on failure. We append `check(entry_point)`
    to invoke it.
    """
    # Compose the script. Catch syntax errors at compile-time so we
    # can distinguish "model wrote junk" from "model wrote correct-
    # syntax but wrong-logic code".
    script_pieces = [prompt, "\n", completion, "\n\n", test, "\n",
                      f"check({entry_point})\n"]
    script = "".join(script_pieces)

    # Pre-compile to detect syntax error before subprocess spawn.
    try:
        compile(script, "<harness>", "exec")
    except SyntaxError as se:
        return ExecResult(passed=False, timed_out=False, syntax_error=True,
                           error_kind="syntax", stdout="",
                           stderr=f"SyntaxError: {se}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        try:
            proc = subprocess.run(
                ["python3", path],
                capture_output=True, text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(passed=False, timed_out=True, syntax_error=False,
                               error_kind="timeout", stdout="", stderr="")
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass

    if proc.returncode == 0:
        return ExecResult(passed=True, timed_out=False, syntax_error=False,
                           error_kind=None,
                           stdout=proc.stdout, stderr=proc.stderr)
    return ExecResult(passed=False, timed_out=False, syntax_error=False,
                       error_kind="runtime",
                       stdout=proc.stdout, stderr=proc.stderr[:500])
