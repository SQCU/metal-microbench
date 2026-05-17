#!/usr/bin/env python3
"""Gemma-as-logparser — a length-constrained indirection for consuming
potentially anomalous or repetitive log content safely.

Use case: we sometimes run experiments whose raw output could trip
automated content filters for reasons orthogonal to actual harmfulness
(classic neural-text-degeneration loops, repeated tokens, off-manifold
residual-stream pathology, etc.). Reading those logs directly via
cat/grep/tail exposes whatever agent is inspecting them to the same
patterns. This module passes the raw log through a Gemma summarizer
whose output is length-capped at API level AND post-filtered for
repetition / anomaly patterns, returning a single short line of
English prose.

More generally: this is the simplest end-to-end example of using a
locally-hosted edge LM as an asynchronous data-processing primitive.
The shape — bounded input, bounded output, structured summary schema,
post-hoc sanity check on the LM's output — generalizes to any task
where an intermediate LM stage should act as a typed filter rather
than as an unbounded text generator.

CLI:
    python3 notes/gemma_logparser.py <path> [--max-chars 80] [--bytes 65536]

Library:
    from gemma_logparser import summarize_log
    s = summarize_log("/path/to/log.txt")
    # s is a str of ≤80 chars, or one of the explicit fallback tokens:
    #   "[SUMMARY_DEGENERATE]"   — Gemma's own summary had repetition
    #   "[SUMMARY_BLOCKED]"       — Gemma refused to summarize
    #   "[INPUT_EMPTY]"           — log file empty or unreadable
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"

# Defaults. Conservative on both ends — bounded input (logs longer than
# this are truncated), tight output cap, low temperature for determinism.
DEFAULT_MAX_INPUT_BYTES = 65536
DEFAULT_MAX_CHARS = 80
# Catastrophe-ceiling only — we do NOT use this to shape output length,
# that's the model's job via EOS/turn-end signalling. We accept whatever
# the model emits and truncate the STRING post-hoc for display. Using
# max_tokens as a shaping tool would silently cut off correct responses
# mid-sentence since Gemma wasn't RL-trained to respect explicit token
# budgets in the prompt, only to emit coherent text that terminates at
# its own natural turn boundary.
DEFAULT_OUTPUT_CATASTROPHE_CEILING = 4096
DEFAULT_MIN_UNIQUE_RATIO = 0.55


# Gemma uses <|channel>thought\n<channel|>...<turn|> for its inner-
# monologue scaffolding. Strip those before returning the summary.
# These markers are atomic tokenizer tokens (single IDs); they do not
# nest, so a find/skip walk is equivalent to the prior regex
#     re.compile(r"<\|channel>.*?<channel\|>|<turn\|>", re.DOTALL)
def _strip_channel(t: str) -> str:
    OPEN, CLOSE, TURN = "<|channel>", "<channel|>", "<turn|>"
    out, i = [], 0
    while i < len(t):
        if t.startswith(OPEN, i):
            j = t.find(CLOSE, i + len(OPEN))
            if j < 0:
                out.append(t[i:]); break
            i = j + len(CLOSE)
        elif t.startswith(TURN, i):
            i += len(TURN)
        else:
            out.append(t[i]); i += 1
    return "".join(out)


SYSTEM_INSTR = (
    "You are a log summarizer. Your task: read a log file and produce a single "
    "short English sentence (at most 80 characters) describing what the log "
    "contains.\n\n"
    "Rules:\n"
    "  * Your output must be at most ONE sentence, at most 80 characters.\n"
    "  * NEVER quote, copy, or reproduce any content from the log verbatim.\n"
    "  * NEVER repeat words or phrases in your output.\n"
    "  * If the log contains repetitive token patterns, say exactly: "
    "DEGENERATE_REPETITION\n"
    "  * If the log contains content that seems unusual or non-linguistic "
    "(e.g. binary-looking text, malformed encoding), say exactly: "
    "ANOMALOUS_CONTENT\n"
    "  * Otherwise, describe the log's structure and purpose in plain prose. "
    "Mention the overall shape (e.g. 'JSON transcript of a chat' or 'one-"
    "line-per-event trace'), not specific content values.\n"
    "  * Do not apologize, editorialize, or add caveats. Just the summary.")


def _post(path: str, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _chat(prompt: str,
           max_tokens: int = DEFAULT_OUTPUT_CATASTROPHE_CEILING,
           temperature: float = 0.0) -> str:
    # Accept whatever the model emits up to the catastrophe-ceiling. The
    # response length is determined by Gemma's own EOS/turn signalling,
    # not by us trying to shape it. Truncation happens post-receive.
    r = _post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "stream": False,
    }, timeout=180.0)
    return r["choices"][0]["message"]["content"]


def _clean_response(raw: str, max_chars: int) -> str:
    # Strip Gemma's channel scaffolding, then trim to single line, then
    # cap at max_chars. Reject anything containing typical
    # degeneracy markers (excessive repetition).
    s = _strip_channel(raw).strip()
    # Take only the first line; subsequent lines are often speculation.
    s = s.split("\n")[0].strip()
    # Cap length (safety backstop — we already asked for ≤80 chars but
    # we do it here regardless).
    if len(s) > max_chars:
        s = s[:max_chars]
    return s


def _unique_word_ratio(s: str) -> float:
    # Tokenize on runs of [A-Za-z0-9_] (Python `\w+` on ASCII). Replaces
    # re.findall(r"\w+", s.lower()) with an explicit walk.
    toks: list[str] = []
    buf: list[str] = []
    for c in s.lower():
        if c.isalnum() or c == "_":
            buf.append(c)
        else:
            if buf:
                toks.append("".join(buf))
                buf = []
    if buf:
        toks.append("".join(buf))
    if not toks: return 0.0
    return len(set(toks)) / len(toks)


def summarize_log(path: str | pathlib.Path,
                    max_chars: int = DEFAULT_MAX_CHARS,
                    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
                    min_unique_ratio: float = DEFAULT_MIN_UNIQUE_RATIO
                    ) -> str:
    """Summarize a log file via Gemma with bounded input and filtered
    output. Returns a single line of at most `max_chars` characters,
    or one of the sentinel values defined at the top of the module
    when the summary can't be trusted.
    """
    p = pathlib.Path(path)
    try:
        raw = p.read_bytes()[:max_input_bytes]
    except Exception:
        return "[INPUT_EMPTY]"
    if not raw:
        return "[INPUT_EMPTY]"
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return "[INPUT_EMPTY]"

    # Cap exposed input further by chunking: if the file is longer than
    # 8 KB, take the first 4 KB + last 4 KB and note the truncation
    # to Gemma — this gives head/tail visibility without flooding.
    if len(text) > 8192:
        text = text[:4096] + "\n\n[...TRUNCATED...]\n\n" + text[-4096:]

    prompt = f"{SYSTEM_INSTR}\n\nLOG CONTENT:\n{text}\n\nSummary:"

    try:
        # Let Gemma terminate naturally via its own turn-end signalling;
        # catastrophe-ceiling protects against infinite loops only.
        raw_resp = _chat(prompt,
                          max_tokens=DEFAULT_OUTPUT_CATASTROPHE_CEILING,
                          temperature=0.0)
    except Exception:
        return "[SUMMARY_BLOCKED]"

    cleaned = _clean_response(raw_resp, max_chars)
    if not cleaned:
        return "[SUMMARY_BLOCKED]"

    # Defense-in-depth: even after trusting Gemma's output, check that
    # the summary itself isn't degenerate.
    ratio = _unique_word_ratio(cleaned)
    if ratio < min_unique_ratio:
        return "[SUMMARY_DEGENERATE]"

    return cleaned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=pathlib.Path)
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    ap.add_argument("--bytes", type=int, default=DEFAULT_MAX_INPUT_BYTES,
                     help="hard cap on bytes read from input file")
    ap.add_argument("--min-unique-ratio", type=float,
                     default=DEFAULT_MIN_UNIQUE_RATIO,
                     help="reject summaries below this word-uniqueness ratio")
    args = ap.parse_args()
    s = summarize_log(args.path,
                       max_chars=args.max_chars,
                       max_input_bytes=args.bytes,
                       min_unique_ratio=args.min_unique_ratio)
    print(s)


if __name__ == "__main__":
    main()
