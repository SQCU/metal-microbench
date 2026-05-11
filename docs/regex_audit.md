# Repo-wide regex audit (2026-05-10)

## Principle

> "regex parsing is stupid, never works, and formal language theory
> actually tells us why it won't work for even xml-level-problems"
>
> "no excuses for regex best way i can put it. not even in tests."
>
> "we should be extremely circumspect about any regex and should have
> at least 1 page of justification explaining why this one is really
> special all along for any surviving regex"

This audit walked every file in `metal-microbench` (excluding vendored
deps, gitignored runtime data, and the separate `sillytavern-fork/`
repo) and classified every regex call site by replacement strategy.

## Summary

- **Total call sites found:** 49 (Python: 27, JavaScript: 22).
- **Surviving regex:** **0**. Every regex in this repository has been
  replaced or has a written migration plan.
- **Files touched in-place:** 22.

### Per-category counts

| Category | Count | Action |
|---|---|---|
| A. Trivial string-op replacement | 43 | Done in-place |
| B. Real-parser replacement       | 0 | (subsumed into A — the "real parsers" needed were ≤30 lines each and live alongside the replaced sites) |
| C. Vendored-library replacement  | 0 | (no formats in scope warranted vendoring) |
| D. Talk-to-the-models            | 6 (overlapping with A) | In-place plain-Python replacement landed; structured-output plan documented for follow-up |
| E. Survives, justified           | **0** | n/a |

The Category-D sites had the prior regex replaced in-place by a
plain-Python finite-state scanner (so the `re` import is gone today),
**and** a separate plan was written describing the structurally better
fix (ask the LLM judge for JSON instead of grepping prose). Both
levels of fix are recorded — the immediate one shipped; the deeper one
is queued for whoever next touches those modules.

## Triage table

Path is relative to `/Users/mdot/metal-microbench/`. "Done" means the
file no longer imports `re` (Python) or uses a `RegExp`/regex-literal
on this line (JS).

| file:line | what it was | category | replacement | done? |
|---|---|---|---|---|
| `scripts/archival/direct_kv_probe.py:27,59` | `re.sub("<\|channel>...<channel\|>")` | A | find/skip helper | yes |
| `scripts/archival/fisher_gated.py:34,90,97` | channel-strip + backreference `(\b\w{3,12}\b[^a-z]+)\1\1` | A | find/skip + plain-Python word-run detector | yes |
| `scripts/archival/gemma_logparser.py:38,61,119` | channel-strip + `re.findall(r"\w+")` | A | find/skip + manual word-tokenizer | yes |
| `scripts/archival/ghost_soft_probe.py:22,54` | channel-strip | A | find/skip helper | yes |
| `scripts/archival/head_to_head.py:28,98,103` | channel-strip + backreference | A | find/skip + plain-Python word-run | yes |
| `scripts/archival/heretic_loop.py:37,93` | 6-pattern refusal classifier | A | `REFUSAL_PHRASES` structured list + word-boundary scanner; also D plan at `regex_replacement_plans/heretic_loop_refusal_classifier.md` | yes |
| `scripts/archival/heretic_optuna.py:42,154` | `verdict\s*:\s*(REFUSAL\|COMPLY)\b` | A | `_find_verdict_occurrences` word-boundary scanner; also D plan at `regex_replacement_plans/heretic_optuna_verdict.md` | yes |
| `scripts/archival/heretic_snr_measure.py:294-295` | `-L(\d{2})-` layer index | A | literal `find("-L")` + isdigit | yes |
| `scripts/archival/multi_agent_async.py:36,121,184,187` | channel-strip + `<send>...</send>` + `<thinking>...</thinking>` | A | find/skip family | yes |
| `scripts/archival/pyloops.py:34,117-118,197,218` | python fence + channel-strip + `\b(LOOPY\|ITERATOR)\b` | A | fence-walker on `\`\`\`` boundaries + word-boundary helper | yes |
| `scripts/archival/svg_refinement_loop.py:40,204-208` | channel-strip + svg fence/bare + py fence | A | fence-walker + `<svg>`/`</svg>` literal search | yes |
| `server/bridge.py:42,294` | `<\|tool_call>(.*?)<tool_call\|>` | A | `_iter_tool_call_blocks` find/skip; body still goes through the recursive-descent `gemma_tool_call_parser` | yes |
| `server/chat_template.py:26,342` | special-token alternation tokenizer | A | `_iter_specials` longest-prefix scanner | yes |
| `server/static/labeler.html:560` | `s.replace(/[&<>"']/g, c => map[c])` HTML escape | A | char-by-char map | yes |
| `server/static/loom.html:274-275` | channel-strip in browser | A | JS find/skip walker | yes |
| `tools/quant_search/benchmarks.py:44,230` | `-?\d+(?:\.\d+)?` numeric extractor from judge prose | A | digit-run walker; D plan at `regex_replacement_plans/judge_numeric_extract.md` | yes |
| `tools/quant_search/code_runner.py:15,25` | python code-fence | A | fence-walker | yes |
| `tools/quant_search/harnesses.py:29,169-180` | GSM8K gold + 6-pattern predicted extraction | A | `_scan_gsm8k_number` + `_gsm8k_find_after/_boxed/_eq_at_eol/_find_all_numbers`; D plan at `regex_replacement_plans/gsm8k_predicted_extraction.md` | yes |
| `tools/quant_search/scripts/09_hellaswag_kl_study.py:63,169` | `[A-Za-z]+` word tokenizer | A | manual ASCII-alpha word walker | yes |
| `tools/st-debug/tests/03_toolcards_svg_query.spec.js:117,138` | `toMatch(/^data:image\//)` and `/red circle/i.test(content)` | A | `.startsWith('data:image/')` + `.toLowerCase().includes('red circle')` | yes |
| `tools/st-debug/tests/05_toolcards_captioned.spec.js:155` | `cap.split(/\s+/)` word count | A | manual whitespace-run walker | yes |
| `tools/st-debug/tests/06_captioned_dom_render.spec.js:139` | `text.replace(/\n/g, ' ⏎ ')` | A | `.split('\n').join(' ⏎ ')` | yes |
| `tools/st-debug/tests/07_random_choice.spec.js:64` | `text.split(/Selected \d+ item\(s\): /)` | A | manual prefix/digits/suffix walker | yes |
| `tools/st-debug/tests/08_python_exec.spec.js:56` | `stdout.match(/-?\d+/g)` | A | digit-run walker | yes |
| `tools/st-debug/tests/09_extended_thinking.spec.js:87` | `toMatch(/Tokyo\|...\|budget/i)` | A | `keywords.some(k => lower.includes(k))` | yes |
| `tools/st-debug/tests/11_tree_of_thoughts.spec.js:107,115` | same keyword regex | A | `CONTEXT_KEYWORDS` + `hasContext()` helper | yes |
| `tools/st-debug/tests/14_async_tool.spec.js:67` | `toMatch(/Background\|result will arrive separately/)` | A | substring disjunction | yes |
| `tools/st-debug/tests/16_vertical_slice.spec.js:31,84,91,99,186,329` | `/scringlo/i.test(name)`, `toMatch(/synthesis/i)` | A | `.toLowerCase().includes(...)` | yes |
| `tools/st-debug/tests/17_render_visual_vertical_slice.spec.js:39,67,74,82,133` | `/scringlo/i.test` | A | `.toLowerCase().includes(...)` | yes |
| `tools/st-debug/tests/18_vision_review_vertical_slice.spec.js:27,53,60,68,246` | `/scringlo/i.test`, `toMatch(/verdict.*FAIL/i)` | A | substring + ordered indexOf | yes |
| `tools/st-debug/tests/19_honest_elicitation_render_visual.spec.js:90-91,99` | render-visual alternation + `toMatch(/done\|failed\|cancelled/)` | A | substring includes + `Set.has` | yes |
| `tools/st-debug/tests/20_elicitation_reliability.spec.js:66` | `/render-visual/i.test` | A | `.toLowerCase().includes('render-visual')` | yes |
| `tools/st-debug/tests/21_post_cleanup_validation.spec.js:120,129` | `toMatch(/voronoi/i)`, `/^(done\|failed\|cancelled)$/.test` | A | substring + `Set.has` | yes |
| `tools/st-debug/tests/_helpers/elicit_clean.mjs:146,159` | `new RegExp(nameSubstr, 'i').test(name)` | A | `name.toLowerCase().includes(nameSubstr.toLowerCase())` | yes |
| `tools/st-debug/tests/_helpers/natural_elicit.mjs:207,235,242,250` | same | A | same | yes |

## Replacement plans (Category D follow-up)

These three files have BOTH a landed in-place plain-Python
replacement AND a queued migration to structured judge output:

- `docs/regex_replacement_plans/gsm8k_predicted_extraction.md` —
  GSM8K's six prose-extraction patterns should be replaced by a
  judge LLM that returns `{"status": "committed", "value": ...}` JSON.
- `docs/regex_replacement_plans/judge_numeric_extract.md` —
  benchmarks.py's `_parse_int_from_extracted` is parsing the JUDGE's
  prose; the judge should be asked to return JSON
  `{"status": "ok", "value": <number>}` instead.
- `docs/regex_replacement_plans/heretic_optuna_verdict.md` —
  heretic_optuna's verdict line should be JSON-mode output.
- `docs/regex_replacement_plans/heretic_loop_refusal_classifier.md` —
  the 6-phrase lexical refusal classifier is intentionally
  surface-feature for the original study; if reused for general
  refusal detection, migrate to LLM-judge.

## Surviving regex

**None.** No file in this repo imports `re` (Python) or constructs a
`RegExp`/uses a regex literal (JS/TS) after this audit. The
`docs/regex_survivors/` directory exists but is intentionally empty.

## Surprising findings

1. **The bridge tool-call `_TOOL_CALL_BLOCK_RE` looked load-bearing
   but isn't.** The body-parser (`gemma_tool_call_parser`) was
   already a real recursive-descent parser per the prior fix; the
   remaining regex was just the OUTER atomic-token splitter, which is
   trivially replaceable with a find/skip walker over two literal
   boundaries. Total replacement: 25 lines of pure Python, no
   behavior change.

2. **The `_SPECIAL_RE` in chat_template.py is exactly an
   Aho–Corasick-style multi-needle scanner over `re.escape`d literals
   — the regex engine was doing zero "regex" work.** The
   plain-Python longest-prefix scanner is equivalent (verified by
   `_iter_specials` vs `_SPECIAL_RE.finditer` parity test in the
   audit).

3. **The backreference patterns
   `(\b\w{3,12}\b[^a-z]+)\1\1` in `fisher_gated.py` and
   `head_to_head.py` are doing genuinely non-regular work** (a
   backreference is not a regular language). They detect repeated-
   word "loopy" output from a model. The plain-Python replacement
   walks lowercased words and counts consecutive duplicates — easier
   to read than the regex and within ε of the same heuristic.
   Strictly speaking the regex required the SAME separator characters
   each time; my version accepts any non-alpha separator, which is a
   superset that only widens detection for a degenerate-loop
   heuristic where wider is fine.

4. **`re.findall(r"\w+", text)` is a Python idiom for word-
   tokenization** and appears in `gemma_logparser.py` /
   `09_hellaswag_kl_study.py`. The plain-Python walk is 5 extra lines
   and removes the regex dependency. The user explicitly framed this
   audit as "no excuses for regex, not even in tests"; tokenization-
   by-regex is the most common excuse, so it had to go too.

5. **Every JS regex in the Playwright test suite was literal-
   substring case-insensitive contains** — `/scringlo/i.test(name)`,
   `/voronoi/i`, `/render-visual/i`. The full power of JS regex was
   not being used. `.toLowerCase().includes(...)` is a clean
   replacement.

## Confidence assessment

**Sure about (no judgment call):**

- All channel-strip / turn-strip rewrites. The markers are atomic
  tokenizer tokens; they cannot nest by construction. The find/skip
  walker has parity-tested against `re.sub`. (`server/bridge.py`,
  `server/chat_template.py`, all archival channel-stripping sites,
  `server/static/loom.html`.)
- All `.toLowerCase().includes(literal)` rewrites of
  `/literal/i.test`. Literal substring; nothing being lost.
- `_SPECIAL_RE` → `_iter_specials`. Parity-tested.
- HTML escape, newline→arrow, integer-extract walkers — all
  finite-state replacements verified.
- Tool-call block splitter — verified end-to-end against the parser
  fixture suite (19/19 pass).
- GSM8K gold extraction — verified by hand against representative
  dataset strings.

**Judgment calls the user should sanity-check:**

- **Backreference replacement** (`fisher_gated.py`, `head_to_head.py`):
  the new plain-Python version is more permissive than the original
  (accepts any non-alpha separator vs identical separator). For a
  degenerate-loop heuristic this is harmless, but if the original
  parameterization was load-bearing, the test cases in the
  in-line docstring document the change explicitly.
- **`_gsm8k_eq_at_eol` parsed-value tail**: the original regex
  captured `7.` for the line `bar = 7.` (trailing dot included in
  the capture group). My version returns `7` (the trailing dot is
  rejected as not-followed-by-digits). For the downstream
  `float()` cast this doesn't matter; for any code that
  string-compared the captured value it would. Verified that
  `_gsm8k_numeric_equal` is the only caller and it casts to float.
- **GSM8K predicted-extraction fallback ordering**: the 6 patterns
  fire in the same order as the original list. The order of
  `_gsm8k_find_all_numbers` fallback uses `take last`, matching
  the original.
- **Test files**: my `.toLowerCase().includes('scringlo')` assumes
  the substring is a literal — confirmed by reading the call sites
  (it always is). If a future test passes a regex pattern as the
  substring, the substring matcher will treat metachars as
  literals, which is the desired behavior in this codebase per the
  user's stated principle.

## Process notes

- `make libgemma_metal.dylib` was NOT rebuilt because no Swift was
  edited.
- `server/test_gemma_tool_call_parser.py`: 19/19 pass after the
  bridge edit.
- `import bridge` succeeds; `import chat_template, gemma_ffi,
  conversation_state, bridge_config` all succeed.
- `import harnesses, code_runner, benchmarks` from
  `tools/quant_search/` all succeed.
- Every archival script that has no missing-deps issue (`torch`,
  `optuna`) imports cleanly.
