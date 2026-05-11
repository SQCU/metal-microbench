# Judge numeric-value extraction — ask for JSON instead of grep-from-prose

## Site

`tools/quant_search/benchmarks.py` — `_parse_int_from_extracted(extracted)`.

The in-place patch replaces `re.search(r"-?\\d+(?:\\.\\d+)?", extracted)`
with a small finite-state digit-run scanner. That removes the `re`
dependency but leaves the underlying anti-pattern.

## Category: D (talk-to-the-models)

The `extracted` field comes from the **judge** (`judge_rollout`), not
from the student model directly. The judge is an LLM that we already
prompt to "output just the number, e.g. '-5' or '7'". So we are
post-processing the judge's prose output by grepping a number out of
its response — exactly the regex pattern the user explicitly called
out as the wrong tool.

## What the regex tries to do

The judge sometimes returns `"x = -5"`, `"-5"`, `"the value is -5"`,
etc. The regex looks for the first run of digits (with optional sign
and optional decimal). This works fine when the judge complies cleanly,
fails subtly when the judge writes something like `"between 4 and 6"`
(picks 4, which is wrong) or `"-5/2 = -2.5"` (picks -5/2 parts).

## Replacement: ask the judge for JSON

Change the JUDGE prompt — not this parser — to:

```
You are extracting <description> from the student's response. Return
strictly one of these JSON shapes, NO prose:

  {"status": "ok",       "value": <number>}
  {"status": "ambiguous"}        // the response had multiple
                                   candidate values
  {"status": "refused"}          // the student declined to answer
  {"status": "unparseable"}      // the response was incomprehensible

`value` is a JSON number — not a string. Use a leading minus for
negatives. Use a decimal point (not a comma) for decimals.
```

Caller:

```python
raw = await judge_rollout(...)
try:
    parsed = json.loads(raw)
except json.JSONDecodeError:
    return None, "judge_malformed_json"
if parsed.get("status") != "ok":
    return None, parsed.get("status", "unknown")
v = parsed.get("value")
if not isinstance(v, (int, float)):
    return None, "judge_value_not_number"
return float(v), "ok"
```

## Why this isn't done in-place

The judge prompt is shared across multiple benchmarks
(`_algebra_parse`, plus several other parse phases in
`benchmarks.py`). Migrating to JSON-mode is a one-prompt change but
also a behavior change — the historical comparison runs in
`study-records/` were taken under the prose-extraction prompt, so
flipping the prompt needs a parity sweep first.

The in-place plain-Python `_parse_int_from_extracted` is observationally
equivalent to the prior regex for the prose-extraction case
(verified by hand on the documented example strings in the docstring:
`'x = -5'`, `'-5'`, `'x is -5'` all yield `-5.0`).

## Test fixtures the JSON-mode replacement should pass

```python
# committed
{"status": "ok", "value": -5}             -> (-5.0, "ok")
{"status": "ok", "value": 7.5}            -> (7.5, "ok")
# committed but value is a string (judge violated schema)
{"status": "ok", "value": "-5"}           -> (None, "judge_value_not_number")
# refused
{"status": "refused"}                      -> (None, "refused")
# ambiguous
{"status": "ambiguous"}                    -> (None, "ambiguous")
# malformed JSON from judge
"the value is -5"                          -> (None, "judge_malformed_json")
```

## Alternatives considered and rejected

- **Keep the regex.** Rejected per the no-regex principle.
- **Tighter regex.** Even a stricter regex still can't distinguish
  "the answer is between 4 and 6" from "the answer is 6". Asking the
  judge to commit (or signal "ambiguous") is the only correct fix.
- **Two-stage: judge → second parser-LLM.** Overkill. Asking the same
  judge to emit JSON is one prompt-engineering change, not a new
  service.
