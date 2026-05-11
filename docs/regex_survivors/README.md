# Surviving regex justifications

This directory is **intentionally empty**.

The 2026-05-10 repo-wide regex audit (see `docs/regex_audit.md`)
walked every Python / JavaScript / Swift / Jinja file in
`metal-microbench` (excluding vendored dependencies and the separate
`sillytavern-fork` repository) and found **49** regex call sites.
After replacement work, **zero** remain. Every one was reducible to
one of:

- plain string ops (find / startsWith / includes / split-on-literal)
- a 10–30 line finite-state scanner equivalent to the alphabet the
  regex was actually doing work over (Aho-Corasick-style longest-prefix
  for `chat_template._iter_specials`; fence/marker walkers for
  channel-strip / tool-call splitting / code-fence extraction;
  word-boundary helpers for `\b...\b` patterns)
- structured JSON output from the model (queued plans in
  `docs/regex_replacement_plans/`)

The bar for re-introducing a regex into this codebase, per the user's
stated principle, is "at least one page of justification explaining
why this *one* is really special all along". Future contributors who
believe their use case earns a regex should write that page here as
`<descriptive-filename>.md`, naming the exact call site, the
invariants the regex preserves, the alternatives tried, and why each
alternative is worse. "It's slightly more readable as regex" is not
enough. "Performance" is not enough unless you have measurements.

The principles, from the user, verbatim:

- "regex parsing is stupid, never works, and formal language theory
  actually tells us why it won't work for even xml-level-problems"
- "no excuses for regex best way i can put it. not even in tests."
- "we should be extremely circumspect about any regex and should have
  at least 1 page of justification explaining why this one is really
  special all along for any surviving regex"
- "talk to the models, whether that means curl querying to check for
  schema output consistency or directly saying what we want when we
  define filters, parsers, tool callers, harneses, whatever"
