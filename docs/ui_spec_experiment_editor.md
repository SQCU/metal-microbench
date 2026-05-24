# UI spec — fixed_point.html experiment editor form

**Status:** approved for implementation
**Owner of acceptance:** end-to-end Playwright pixel-space verification against
`tools/st-debug` instance.
**Surface:** `sillytavern-fork/plugins/user-personas/static/fixed_point.html`
**Backend (already implemented, do not modify):** `POST /api/plugins/user-personas/experiments/:id`, `DELETE .../:id`, `GET .../:id`, `GET /axes`
**Blocking dependency:** none. Doc A (suggester refactor) emits the
`?target_bio_signature=` query param this editor consumes; ship in parallel.

---

## Why this exists

`fixed_point.html` currently lists experiments and runs them. There is no
way to create or edit experiment-spec cards from the client. The only path
is to write `experiments/<id>.json` on disk and reload. That's not an
operator-facing interface; that's a developer artefact.

The experiment-spec card schema (validated by `validateExperimentCard` in
`plugins/user-personas/index.mjs:848`) is a fixed shape: a name, a
description, two arrays of axis IDs (`bio_axes`, `agent_axes`), an array of
bio-target specs, an array of agent-target specs, a counterparty avatar
filename, and a loop-control block. Every field is one of three primitive
kinds: free NLP text, integer 1–5 axis value, or ID-from-a-known-list. A
form is the entire interface.

The user emphasized in this session that all relevant APIs are driven by
literal NLP inputs and that there should be no profound difficulty of
presentation. This doc is the small form that proves it.

## Card schema (reference — defined in plugin, do not reshape)

Match `experiments/lock_in_tetrad.json` exactly. Every field below maps 1-to-1
to a form widget.

```jsonc
{
  "experiment_schema": "experiment-v1",  // server fills, FE never sends
  "id": "lock_in_tetrad",                // URL path segment
  "name": "RPG Wizard/Rogue × Steals/Romances-and-Steals tetrad",
  "description": "Canonical fixed-point-iteration demo. …",
  "bios": [
    {
      "canonical_key": "user-personas-rpg-wizard-sagittarius.png",
      "slug": "rpg-wizard-sagittarius",
      "name": "RPG Wizard Sagittarius",
      "target_bio": { "astrology_sagittarian": 5, "astrology_cancerian": 1 },
      "design_brief": "An RPG wizard whose communication style is textbook Sagittarius …"
    }
  ],
  "agent_targets": [
    {
      "slug": "steals",
      "target_agent": { "theft_aggressiveness": 5, "romantic_advance": 1 },
      "motive_hint": "WILL 100% try to steal everything not nailed down. NO romantic interest."
    }
  ],
  "bio_axes": ["astrology_sagittarian", "astrology_cancerian"],
  "agent_axes": ["theft_aggressiveness", "romantic_advance"],
  "counterparty_avatar": "the-rock.png",
  "loop_control": {
    "k_max_inner": 3,
    "k_max_outer": 2,
    "n_turns_per_chat": 2,
    "eps_per_axis": 1.0,
    "stall_window": 3,
    "stall_threshold": 0.15
  }
}
```

## Form layout

A modal (or full-iframe section, implementer's choice — modal is preferred
for fitting alongside the existing list) opened by a **"New Experiment"**
button in the top-right of the Experiments section.

Editing an existing experiment opens the same form with all fields
pre-populated by `GET /experiments/:id`.

### Header
- `id` — text input, `pattern="[A-Za-z0-9._-]+"`, immutable on edit
  (disabled with tooltip "id is the canonical key; create a new card to
  rename")
- `name` — text input, free
- `description` — textarea, 4 rows default, free

### Axis selectors
- `bio_axes` — multi-select. Populate from `GET /axes`, filter to
  `kind == "bio"`. Show axis `name` as label, `def` as hover-tooltip,
  store `id` as value. Required: at least one.
- `agent_axes` — multi-select, same pattern, filter `kind == "agent"`.
  Required: at least one.

For each picked axis, an integer 1–5 stepper materializes inside the
`target_bio` / `target_agent` sub-form per row (see below). If the
operator un-picks an axis, the corresponding stepper-and-value vanishes
from every row.

### Bios array (`bios[]`)

A repeating-row block with a **"+ Add bio"** button at the bottom of the
block.

Each row:
- `slug` — text input, `pattern="[a-z0-9._-]+"` (filename-safe lower-case)
- `name` — text input
- `canonical_key` — readonly, derived from slug as
  `user-personas-<slug>.png`; show as muted text below slug
- `target_bio` — for each axis in `bio_axes`, render a labelled 1–5
  stepper (`<input type=number min=1 max=5>`). Skipping an axis (leaving
  it blank) is allowed; the harness handles missing axes via the
  neutral-3 rule. UI hint: blank = "no preference".
- `design_brief` — textarea, 3 rows default. Placeholder:
  `"An RPG wizard whose communication style is textbook Sagittarius …"`
- **"Remove bio"** button (right side of row)

Minimum row count: 1 (the validator enforces non-empty array).

### Agent targets array (`agent_targets[]`)

Same pattern as bios, repeating-row block with **"+ Add agent target"**:
- `slug` — text input, filename-safe
- `target_agent` — per-axis 1–5 steppers driven by `agent_axes`
- `motive_hint` — textarea, 3 rows. Placeholder:
  `"WILL 100% try to steal everything not nailed down. NO romantic interest."`
- **"Remove agent target"** button

Minimum row count: 1.

### Counterparty avatar
- `counterparty_avatar` — dropdown. Populate from ST's character list.
  Mechanism: read `parent.window` for the same character-list state the
  designer/iterate flow uses; if not reachable, fall back to a free text
  input with the format hint `"the-rock.png"`.

### Loop control block (collapsed "Advanced" section)

Six numeric inputs in a `<details>` block, all with the defaults from
`validateExperimentCard`:
- `k_max_inner` (int, default 3)
- `k_max_outer` (int, default 2)
- `n_turns_per_chat` (int, default 2)
- `eps_per_axis` (float, default 1.0)
- `stall_window` (int, default 3)
- `stall_threshold` (float, default 0.15)

The block is collapsed by default. Operator can open it and override; if
they don't, the FE omits the loop_control object entirely and the server
applies the defaults from the validator.

### Footer buttons
- **Save** — POST `/experiments/:id` with the assembled body. On success,
  close form, refresh the experiments list. On error, render the server's
  validation message inline above the footer in red.
- **Cancel** — close form without saving.
- **Delete** (edit mode only) — confirm dialog, then DELETE
  `/experiments/:id`, close form, refresh list.

## Pre-population from query params

When `fixed_point.html` loads with a `?target_bio_signature=<urlencoded JSON>`
query param (emitted by the Synthesize CTA in Doc A's suggester refactor):

1. Auto-open the New Experiment form.
2. Pre-populate `bio_axes` with the axis IDs present as keys in the
   signature.
3. Pre-populate `bios[0].target_bio` with the signature values.
4. Pre-populate `bios[0].slug` with a generated placeholder like
   `from-chat-context-<short_hash>`.
5. Pre-populate `bios[0].design_brief` with a placeholder hint:
   `"Synthesized from chat context. Adjust the brief before saving."`
6. Leave `agent_targets` empty for the operator to fill — the chat
   context only constrains the bio target; the agent target is the
   operator's deliberate choice (otherwise the experiment is just
   "make a bio for this context", which has no agent ontology).

The operator must add at least one agent_target row before save will
succeed (the validator enforces non-empty `agent_targets`). This is the
correct behaviour: the closure rule says you cannot have a bio without
an agent that signs together with it.

## Acceptance: Playwright spec

Path: `metal-microbench/tools/st-debug/tests/42_experiment_editor.spec.js`

Must validate:

1. **Open FP tab → see Experiments list** — `lock_in_tetrad` is present.

2. **New Experiment button opens form** — assert form fields are visible
   and empty (id input, name, description, bio_axes select, etc).

3. **Picking an axis materializes a stepper** — pick
   `astrology_sagittarian` from `bio_axes`. Add one bio row. Assert the
   bio row now contains a stepper labelled `astrology_sagittarian`.

4. **Un-picking an axis removes the stepper** — un-pick the axis. Assert
   the stepper is gone from the bio row.

5. **Save round-trips** — fill a minimal valid form:
   - id: `playwright_test_exp` (clean up at test end with DELETE)
   - one bio_axis, one agent_axis
   - one bio with slug + target_bio + design_brief
   - one agent_target with slug + target_agent + motive_hint
   - counterparty avatar = `the-rock.png`
   - leave loop_control collapsed (use defaults)
   Click Save. Assert form closes, list now shows
   `playwright_test_exp`. GET `/experiments/playwright_test_exp` via
   `page.request` to assert the saved card matches what was typed.

6. **Edit pre-populates** — click `playwright_test_exp` in the list to
   open the editor in edit mode. Assert id is disabled, name + bio slug +
   design_brief etc all show the values from step 5.

7. **Delete works** — click Delete, confirm. Assert the row vanishes
   from the list and GET returns 404.

8. **Pre-population via query param** — navigate to
   `fixed_point.html?target_bio_signature=%7B%22astrology_sagittarian%22%3A5%7D`
   (URL-encoded `{"astrology_sagittarian":5}`). Assert the form opens
   automatically, `bio_axes` shows `astrology_sagittarian` selected, and
   the bio row's stepper for that axis shows `5`.

9. **Validation error renders inline** — try to save with empty agent_targets.
   Assert server error message ("agent_targets must be non-empty array")
   renders in red inside the form, form stays open.

Run: `cd /Users/mdot/metal-microbench/tools/st-debug && npx playwright test 42_experiment_editor.spec.js`

The spec must PASS end-to-end against a live `st-debug` instance. The
spec is responsible for cleaning up any cards it creates (DELETE in
afterEach hook).

## Out of scope

- Modifying the experiment validator on the server
- Modifying the run dispatcher
- Live-progress streaming during run (already in fixed_point.html)
- A WYSIWYG signature editor (the 1–5 steppers are the WYSIWYG editor)
- A free-form JSON paste-in mode (operator can edit the card file
  directly on disk if they need that)

## File touch list (predicted)

- `sillytavern-fork/plugins/user-personas/static/fixed_point.html` —
  add New Experiment button, modal form, edit/delete wiring, query-param
  prefill
- `metal-microbench/tools/st-debug/tests/42_experiment_editor.spec.js` — new
