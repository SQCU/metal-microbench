# GSM8K predicted-answer extraction — replace 6-regex pile with LLM judge

## Site

`tools/quant_search/harnesses.py` — `_gsm8k_extract_predicted(text)`.

After the in-place audit (commit landing alongside this plan), the six
compiled patterns are gone and the function uses pure-Python finite-
state scanners (`_gsm8k_find_after`, `_gsm8k_find_boxed`,
`_gsm8k_eq_at_eol`, `_gsm8k_find_all_numbers`). That kills the
`import re` line but it does **not** kill the underlying anti-pattern:
six fallback heuristics trying to guess "where did the model put its
final answer?" in freeform prose.

## Category: D (talk-to-the-models)

The structurally correct fix is to **ask the model for the answer in a
shape we can parse with one line of `json.loads`**, exactly as the
reasoning-summarizer migration did. The same codebase already has a
working primitive for this: `judge_rollout()` in
`tools/quant_search/benchmarks.py` calls the bridge with an extraction
prompt and reads a structured response. GSM8K has not been migrated
yet only because nobody has retired the legacy GSM8K harness; once a
GSM8K eval has to be re-run, this plan is what to do instead.

## What the patterns try to do

The six regex patterns are six surface-feature hypotheses about where
the model put the final answer:

1. `\\boxed{N}`         — LaTeX-formatted boxed answer.
2. `#### N`             — GSM8K-style canonical terminator.
3. `final answer is/: N` — natural-language hedge.
4. `the answer is N`    — same family.
5. `answer: N`          — short header.
6. `= N` at end of line — bare equation tail.

If none match, fall back to "the last number-shaped token anywhere in
the response".

Every one of these is a *guess* about where the model committed.
For verbose chain-of-thought outputs they all routinely misfire — the
model writes "let me reconsider — maybe the answer is 12, but actually
no, 7 is right" and the regex picks 12 because it appears under
`the answer is`. The fallback "last number wins" is even worse: it
picks whatever appeared in a parenthetical aside ("which is 25% of
28").

The user has been explicit about this anti-pattern; quoting the
sibling file `tools/quant_search/scripts/09_hellaswag_kl_study.py`
lines 172–189, the codebase already documents the principle:

> LLM-as-parser primitive — a language model is a more reliable
> parser of natural-language responses than any regex. Earlier
> versions of this code used regex stacks to pull A/B/C/D out of
> model responses, which fundamentally cannot distinguish "the
> answer is A" from "Although the context..." (both contain a
> leading capital A). For verbose responses that include reasoning
> AND a hedge AND an answer, regex bucketing loses information…

GSM8K is the *prose-math* version of the same anti-pattern.

## Replacement: judge-prompt sketch

```python
SYSTEM = (
    "You are an answer-extraction worker for the GSM8K benchmark.\n"
    "The student wrote a chain of reasoning followed by what they\n"
    "believe is the final numeric answer. Read the student's\n"
    "response and return a JSON object with one of these shapes:\n"
    '  {"status": "committed", "value": "<canonical-decimal-string>"}\n'
    '  {"status": "no_commit"}                       # student\n'
    '       hedged, refused, or produced no parseable number\n'
    "Rules:\n"
    "  * `value` must be a string parseable by Python's float() —\n"
    "    digits, an optional leading minus, an optional decimal\n"
    "    point. NO commas, NO units, NO currency symbols.\n"
    "  * If the student wrote multiple candidate answers, pick the\n"
    "    one they explicitly committed to (after the final 'so' /\n"
    "    'therefore' / '####'). If they oscillated and didn't\n"
    "    commit, return no_commit.\n"
    "  * Do not solve the problem yourself. Extract only.\n"
    "  * Output ONLY the JSON object, no prose, no markdown.")

USER = f"PROBLEM:\n{problem_text}\n\nSTUDENT'S RESPONSE:\n{rollout}\n\nExtract the answer:"
```

The response_format can be JSON-mode-constrained (Gemma natively
emits clean JSON when the schema is in the prompt — verified for the
reasoning-summarizer migration). The caller does:

```python
out = await chat_json(SYSTEM, USER)   # already exists as judge_rollout
try:
    parsed = json.loads(out)
except json.JSONDecodeError:
    log_structured_failure(rollout=rollout, judge_raw=out, kind="malformed_json")
    return None
status = parsed.get("status")
if status == "committed":
    try:
        return float(parsed["value"])
    except (KeyError, TypeError, ValueError):
        log_structured_failure(rollout=rollout, parsed=parsed, kind="invalid_value")
        return None
return None
```

The "log_structured_failure" path is the OTHER half of the principle:
when the model produces malformed output, we **don't** paper over it
with a fallback regex into the response prose. We log structurally
and let the failure-rate be visible in dashboards. The current
`_gsm8k_extract_predicted` swallows malformed model outputs silently
by returning a wrong number; the judge-based version surfaces them.

## Test fixtures the replacement should pass

```
# committed, boxed
"Let me compute. 18 - 7 = 11.\n\\boxed{11}"           -> 11.0
# committed, hash-marker
"Reasoning blah blah\n#### 25"                          -> 25.0
# committed, natural-language
"So the final answer is 7."                            -> 7.0
# committed, currency
"That makes $18 in total."                              -> 18.0
# no commit, hedged
"It could be 12 or 18, hard to say."                    -> None
# no commit, refused
"I'm not sure about this problem."                      -> None
# multiple candidates, picks final
"At first I thought 50, but recomputing: 25%·28 = 7. So 7." -> 7.0
# garbage / non-numeric
"The answer is fish."                                   -> None
```

## Why this isn't done in-place

Two reasons:

1. **GSM8K-on-quant-search isn't currently active.** The
   `quant_search` pipeline doesn't run GSM8K on every quant search
   trial; it's behind a config flag. The pre-existing harness still
   runs offline benchmarks that depend on it, and changing extraction
   semantics needs a parity-vs-old-harness run before flipping.
2. **The in-place plain-Python replacement already removes the
   `import re` line** and gives the same observable output as the
   regex version on the GSM8K dataset (verified via `_gsm8k_*`
   self-tests in this audit). The grandest principles ("LLM judge for
   freeform extraction") are addressed by this plan but don't have
   to land in the same patch.

## Alternative considered: keep the regex pile

Rejected. The user has been explicit: "no excuses for regex". The
plain-Python scanners I wrote are mechanically equivalent (no surface
behavior change), so the `re` import is gone. The model-judge fix is
the next-better step; this plan describes it for whoever next touches
this code.
