# UI spec — context-driven suggester refactor

**Status:** approved for implementation
**Owner of acceptance:** end-to-end Playwright pixel-space verification against
`tools/st-debug` instance, not unit tests, not "looks right by inspection."
**Surface:** `sillytavern-fork/plugins/user-personas/static/suggester.html`
**Backend (already implemented, do not modify):** `POST /api/plugins/user-personas/yapper-seed`
**Blocking dependency:** none. Doc B (experiment editor) consumes the
Synthesize-CTA query param this doc emits; ship in parallel.

---

## Why this refactor

The current `suggester.html` is bio-list-driven: it shows every biography in
the corpus, and the operator clicks "Suggest" per bio to invoke `/poll`. That
inverts the user-stated mental model.

The user's mental model — articulated repeatedly across this session and
locked in by the quote *"the chat user agent suggestion interface which
surfaces user agents based on a guessed similarity by a llm-as-judge-call …
why this is a complete data view is left as an exercise to reader blah blah
blah"* — is:

> Given the current chat context, the suggester ranks all `(bio, agent)`
> compositions in the corpus by L2 distance from a target signature
> extracted from that chat context. Top-K and Side-K render with their
> distances, the `_meta` strip exposes the ranker's working state, a
> "+ More" button extends the rendered set deeper into the ranking, and a
> "Synthesize for this context" CTA appears when nothing in the corpus is
> close enough.

The backend (`/yapper-seed`) already implements this completely. The
frontend just needs to actually consume it.

## Endpoint contract (reference — do not change)

`POST /api/plugins/user-personas/yapper-seed`

Request body:
```jsonc
{
  "chat_context_summary": "string — recent turns / situation prose",
  "counterparty_card_id": "the-rock.png",   // optional, biases target sig
  "K_top": 3,                                // optional, default 3
  "K_side": 3,                               // optional, default 3
  "axes": [...]                              // optional, override card axes
}
```

Response body (relevant fields):
```jsonc
{
  "top":  [{ bio_id, agent_id, why, distance, persona, agent }, ...],
  "side": [{ bio_id, agent_id, why, distance, persona, agent }, ...],
  "_meta": {
    "K_top": 3, "K_side": 3,
    "target_signature": { "<axis_id>": <int 1-5>, ... },
    "target_completed_axes": <int>,
    "candidates_considered": <int>,
    "bios_total": <int>, "agents_total": <int>,
    "pending_synthesis": [<bio_id>, ...],
    "pending_count": <int>
  }
}
```

`top` = nearest by L2 (vibes-with). `side` = farthest of the remainder with
distinct bios from top (productive friction). Each candidate carries a
`distance` field (float).

## Required FE behavior

### 1. Replace bio-list-driven flow with context-driven flow

The right panel changes from "Biographies" to **"Ranked compositions"**.

- The chat-context scratchpad stays where it is on the left.
- Add a **"Rank for this context"** button next to "Add turn" / "Clear".
  Clicking it POSTs `/yapper-seed` with:
  - `chat_context_summary` = the textual rendering of `history[]` (the
    same JSON the page already maintains for the scratchpad — concatenate
    role-prefixed turns, newline-separated). If the user has typed
    nothing, send the empty-history string and let the backend handle it.
  - `counterparty_card_id` = ST's currently selected character avatar
    filename (read from parent window's ST state via the same idiom
    `designer.html` uses; if unavailable, omit).
  - `K_top: 3, K_side: 3` for the initial request.
- Render `top` and `side` arrays into the right panel. Each row:
  - bio name + agent name (link to their cards if available)
  - distance pill (e.g. `L2=1.23`) — colour: green ≤1, amber 1–2, red >2
  - "why" line (already returned by backend)
  - **"Suggest"** button per row — clicking it calls the existing `/poll`
    endpoint with that bio_id + agent_id and appends the generated
    candidate to the left-side feed (this preserves the existing
    candidate-feed rendering, just driven from ranked rows instead of
    bio-list rows).
- The "raw biography" path (no agent) is gone. Per the ontological-closure
  decision in this session, every bio in the corpus has agents because every
  bio came from a fixed-point experiment. If `top[i].agent_id` is missing,
  that's a corpus-integrity bug, not a UI case — render it with a red badge
  saying "BIO WITHOUT AGENT — corpus bug" and continue.

### 2. `_meta` strip panel

A horizontal strip above the ranked list:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ target signature: astrology_sagittarian=4 · curious=5 · disclosive=2 ... │
│ considered 12 / 12 compositions (3 bios × 4 agents) · K_top=3 K_side=3   │
│ pending synthesis: rpg-rogue-cancer · rpg-monk-virgo  [open in FP tab →] │
└──────────────────────────────────────────────────────────────────────────┘
```

Fields rendered (all from `_meta`):
- `target_signature` as `axis_id=value` pills, axes sorted by axis_id
  alphabetically; render N/A for any axis the judge couldn't score
- `target_completed_axes` / total-axis count if available
- `candidates_considered` / `bios_total × agents_total` decomposition
- `K_top`, `K_side` (echo of request)
- `pending_synthesis`: rendered as clickable bio_ids; clicking opens the
  FP tab via the existing drawer-button selector (`#user-fixed-point-button`)
  and passes `?prefill_bio=<bio_id>` (the experiment editor in Doc B reads
  this query param)
- `pending_count` numeric badge

When the strip would be empty (no rank request yet), show a placeholder:
`Click "Rank for this context" to query the suggester.`

### 3. "+ More" K-paging button

Below the ranked list, a `+ More` button. Clicking it:
- Increments local `K_top` and `K_side` by their original initial values
  (`K_top += 3`, `K_side += 3`)
- Re-POSTs `/yapper-seed` with the bumped values
- Replaces the rendered top/side lists with the bigger response (the
  backend re-runs the ranker so the same prefix of rows reappears at the
  top; do NOT try to dedupe client-side, just re-render)
- Updates the `_meta` strip with the new K values

The button is disabled while a request is in flight. If a request returns
the same number of rows as the previous request (i.e. K hit the corpus
ceiling), disable the button permanently and show: `(no more compositions)`.

### 4. "Synthesize for this context" CTA

When `top[0].distance > SYNTHESIZE_THRESHOLD` (configurable constant in the
HTML, default `2.0`), render an additional CTA row above the ranked list:

```
⚠ Nothing in the corpus is close to this context (best L2=2.4).
   [ Synthesize an experiment targeting this signature → ]
```

The button opens the FP tab via `#user-fixed-point-button` and passes the
`target_signature` from `_meta` as a query param. The experiment editor in
Doc B reads `?target_bio_signature=<urlencoded JSON of axis→value>` and
pre-populates the New Experiment form's `bios[0].target_bio` with that
signature.

**This CTA is the ONLY path to "make me an agent for this kind of context"
that's permitted to exist.** Per the deletion of `synthesize_agents_for.mjs`,
`explore_corpus.mjs`, `cluster_disambiguator.mjs`, and `designCheapAgent` in
this session, no single-iteration synthesis is allowed. The CTA opens the
fixed-point editor; it does not call a one-shot synthesis endpoint, because
no such endpoint exists and never will.

### 5. Delete the "show overlay only" checkbox, the "agent" dropdowns, and
    the right-panel bio list

These were artefacts of the bio-list-driven design. Drop them. The right
panel is now exclusively the ranked-compositions list + _meta strip + +More
+ Synthesize CTA.

### 6. Preserve the chat scratchpad and candidates feed unchanged

The left panel (history scratchpad + per-suggestion candidates feed +
EOS-ALARM rendering) stays as-is. The only change is that the "Suggest"
button moves from per-bio rows on the right to per-ranked-row buttons on
the right, calling the same `/poll` with the ranked row's bio_id+agent_id.

## Acceptance: Playwright spec

Path: `metal-microbench/tools/st-debug/tests/41_context_suggester.spec.js`

Must validate (with explicit assertions, not just "page loads"):

1. **Open suggester drawer** — `loadAndConnect(page)` then click the
   existing suggester header button. Iframe loads.

2. **Initial state** — _meta strip shows the placeholder text, ranked list
   is empty.

3. **Add a turn + rank** — fill the scratchpad with one assistant turn
   ("rolls a die and grins"), click "Rank for this context". Wait for
   the ranked list to populate.

4. **Verify ranked rendering** — at least one row in the top list, each
   row has a bio name, agent name, distance pill, "why" text, and a
   working Suggest button. Distance pill colours match the spec
   (programmatically check class names, not pixel colours).

5. **Verify _meta strip** — target_signature pills are present and
   non-empty, `candidates_considered` matches `bios_total × agents_total`
   shown elsewhere on page.

6. **+More appends/extends** — capture rendered row count, click +More,
   wait for re-render, assert row count grew. If the corpus has fewer
   total compositions than +More's bumped K, assert the button becomes
   disabled with the "no more compositions" label.

7. **Synthesize CTA** — fill scratchpad with prose deliberately off the
   corpus (e.g. "discusses early-modern Spanish viticulture in iambic
   pentameter"), rank, assert that if `top[0].distance > 2.0`, the
   Synthesize CTA renders. Clicking it opens the FP-tab drawer and
   navigates the FP iframe to a URL containing `target_bio_signature=`.
   (Don't assert the editor pre-population — that's Doc B's spec.)

8. **End-to-end suggest** — click Suggest on the top row, wait for a
   candidate to appear in the left-side feed, assert it has bio+agent
   badges + non-empty text.

Run command: `cd /Users/mdot/metal-microbench/tools/st-debug &&
npx playwright test 41_context_suggester.spec.js`

The spec must PASS end-to-end against a live `st-debug` instance with the
current corpus state (2 bios × 2 agent_targets seeded by `lock_in_tetrad`).
If the corpus is too small to exercise +More's K-ceiling path, seed an
additional minimal `experiment` or document the gap and assert the
disabled-state path explicitly.

## Out of scope for this doc

- Modifying `/yapper-seed` (already implemented correctly)
- Modifying `/poll`
- The FP-tab experiment editor (Doc B)
- The bridge stream-lifecycle safety triple (already shipped)
- Persistence of ranking history across page reloads (caller can re-rank)

## File touch list (predicted)

- `sillytavern-fork/plugins/user-personas/static/suggester.html` — rewrite
  right panel + add _meta strip + +More + CTA
- `metal-microbench/tools/st-debug/tests/41_context_suggester.spec.js` — new
