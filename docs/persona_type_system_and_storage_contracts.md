# Persona / Bio / Agent Type System and Storage Contracts

**Date produced:** 2026-05-24
**Branch context:** `sillytavern-fork` toolcards branch, HEAD `de653b622`
**Sources:** docs/*.md modified 2026-05-17–2026-05-24, implementation files
listed inline.

---

## 1. The Type Vocabulary

There are **two types**. They share nothing: not a directory, not a schema,
not an API endpoint, not an injection mode.

### Type 1: bio (= persona = card = user persona)

These are the same thing under three names. The operator is explicit:

> "bio==persona==card in this case btw. they aren't 3 different or 2 different
> things."
>
> — session `247a1b45`, JSONL line 50218, 2026-05-24T07:25:44Z
> (`docs/bio_agent_type_factorization_errata.md` §1.1)

A bio is the **static identity prefix** injected at the head of context:
voice, register, background. It provides the invariant scaffold that lets a
counterparty parse the semantic meaning of the user's first turn in a fresh
environment. It is never modified by agent synthesis. Formally:

> "biography — system-prompt prefix information that lets a chat counterparty
> parse the semantic meaning of the user's first few messages in a fresh
> environment. Static identity flavor: voice, register, background."
>
> — `docs/user_agent_factorization_spec.md`; also quoted verbatim in
> `docs/bio_agent_type_factorization_errata.md` §1.8

**Disk location:** `<dataRoot>/<user>/User Avatars/<canonical_key>.png`

**File format:** PNG with a `tEXt` chunk keyed `ccv3` or `chara`, base64-encoded
chara_card_v3 JSON. Schema field: `extensions.card_schema = 'bio-v2'` (constant
`BIO_SCHEMA_CURRENT` at `plugins/user-personas/index.mjs:180`).

**canonical_key** matches `[A-Za-z0-9._-]+\.png` and is simultaneously the
filename, the in-memory ID, and the value `designed_for_bio_id` foreign keys
point to. (`plugins/user-personas/PERSONA_API.md`)

### Type 2: agent (= user-agent)

Agents are the **dispositional postfix overlay** — a depth-1 author's-note
appended near the end of context. The MTG analogy from the canonical spec:
"biography ≈ card art + flavor text + creature type line; user-agent ≈ the
mechanics box."

There can be 0..N agents per bio, up to ~128,000:

> "there are to be any number of 'user agents' for any one user bio, up to,
> say, 128,000 different agents, one might say, per 'user bio'."
>
> — session `247a1b45`, JSONL line 50218 (`docs/bio_agent_type_factorization_errata.md` §2.6)

Agents do not replace the bio prefix. Writing agent content into a bio record
is a type error:

> "this is a conceptual error [...] it would be just as much a type error to
> overwrite an assistant card with a chat turn or a base64 dump of a png."
>
> — session `247a1b45`, JSONL line 50218 (`docs/bio_agent_type_factorization_errata.md` §1.1)

**Disk location:** `plugins/user-personas/agents/<id>.json`

**File format:** JSON. Schema field: `agent_schema = 'agent-v1'`
(`plugins/user-personas/index.mjs:181`). Required fields include `agent_text`,
`injection_mode` (must be `'authors_note'`), `injection_depth` (default 1),
`designed_for_bio_id`.

---

## 2. Storage Contracts Table

| Artifact | Canonical path | File format | Writer | Reader | Git-tracked? | settings.json involved? |
|---|---|---|---|---|---|---|
| Bio (user persona / card) | `<dataRoot>/<user>/User Avatars/<key>.png` | PNG + chara_card_v3 tEXt, schema `bio-v2` | plugin `POST /personas/:id`; harness `saveBio()` | `loadPlayers()` via `readCharaCardPng()`; `buildPersonaSnapshotFromFiles()` for FE projection | No — runtime only in `_data/` | **NO** |
| Agent | `plugins/user-personas/agents/<id>.json` | JSON, schema `agent-v1` | plugin `POST /agents/:id`; harness `saveAgent()` | `loadAgents()` | No — runtime only | NO |
| Axis card | `plugins/user-personas/axes/<id>.json` | JSON, schema `axis-v1` | plugin `POST /axes/:id`; harness axis registry | `loadAxes()` | No — runtime only | NO |
| Experiment card | `plugins/user-personas/experiments/<id>.json` | JSON, schema `experiment-v1` | plugin `POST /experiments/:id`; harness `fetchExperiment()` write path | `loadExperiments()` | No — runtime only | NO |
| Run log / trajectory | `plugins/user-personas/data/runs/<run_id>.log` + per-bio JSON outputs | NDJSON log + per-bio plain JSON | harness on run completion | plugin `GET /runs/:id` | No — runtime only | NO |

Source: `docs/multi_user_agent_chat_interface_spec.md` §8 "Storage canonical
contracts"; `plugins/user-personas/PERSONA_API.md`.

**Key invariant:** one canonical store per concept. The
`_mirrorPersonaToSettingsJson` anti-pattern is permanently forbidden.
(`docs/multi_user_agent_chat_interface_spec.md` principle P-CANONICAL-NOT-MIRRORED)

---

## 3. The settings.json Prohibition

### Why it was banned

`settings.json` is a single JSON blob serialized atomically. Concurrent persona
writes both deserialize the pre-state, mutate one key, serialize, write — last
writer wins and the other write is silently lost. This is the **race condition
that destroyed the May-2026 bio corpus** (31 agents lost their bios, documented
in `docs/settings_json_excision_audit.md` §2 Site 2).

The operator mandate, from `docs/bio_agent_type_factorization_errata.md` §4.1
(session `247a1b45`, JSONL line 50218):

> "settings.json is deprecated due to concurrency and race condition issues;
> our sillytavern fork was to remove absolutely all calls that used settings.json
> as a store of user bios and replace them with a file based model of users
> equivalent to the assistant card file based model sillytavern already uses.
> no fallbacks. no failovers! *no fallbacks*."

The `PERSONA_API.md` contract
(`plugins/user-personas/PERSONA_API.md`, line 51–55), added in commit
`844ee3cdf` 2026-05-19:

> "After this contract lands, NO code in the runtime reads or writes
> `settings.json.power_user.{persona_descriptions, personas, default_persona,
> character_persona_overrides}`. The keys exist in settings.json only because
> upstream's schema declares them; they remain empty objects/null forever."

### The tripwire

`src/persona-file-store.js:163` — `assertSettingsJsonHasNoPersonaData(settingsPath)`:

Called from `plugins/user-personas/index.mjs:1784` inside `init()`, **before
any state is loaded**. If `settings.json` contains non-empty
`power_user.persona_descriptions`, `.personas`, `.default_persona`, or
`.character_persona_overrides`, it throws:

```
═══════════════════════════════════════════════════════════════════════════
  FATAL: settings.json contamination detected — persona data present
═══════════════════════════════════════════════════════════════════════════

  File:       <settingsPath>
  Violations:
    - power_user.persona_descriptions has N entries
    [... etc ...]

  PERSONA DATA MUST NEVER LIVE IN settings.json (race-condition prone).
  Bios are PNG cards in <User Avatars>/ with chara_card_v3 tEXt chunks.
  Agents are PNG cards in plugins/user-personas/agents/. NEVER bios.

  See:
    docs/bio_agent_type_factorization_errata.md
    docs/settings_json_excision_audit.md
    plugins/user-personas/PERSONA_API.md

  This server will not start until you run the one-time port:
    python3 plugins/user-personas/scripts/port_settings_personas_to_files.py
```

The server refuses to start entirely — not a warning, a hard throw.

---

## 4. The Three Defense Layers

### Layer 1 — Runtime tripwire

**File:** `src/persona-file-store.js:163` (`assertSettingsJsonHasNoPersonaData`)
**Caller:** `plugins/user-personas/index.mjs:1784` (inside `init()`)
**Failure mode:** Fatal throw — server refuses to start. Error message quoted
in full in §3 above.
**When it fires:** At plugin load time, before any bio or agent state is loaded.

### Layer 2 — Source-code lint

**File:** `plugins/user-personas/scripts/lint_settings_persona_access.mjs`
**What it does:** Walks `plugins/user-personas/` and `src/`, scans all
`.mjs/.js/.ts/.tsx/.py/.html` files for references to forbidden patterns:

```js
const FORBIDDEN = [
    /power_user\.persona_descriptions/,
    /power_user\.personas\b/,
    /power_user\.default_persona/,
    /power_user\.character_persona_overrides/,
    /persona_descriptions\s*[:=]/,
];
```

Allow-listed files: `src/persona-file-store.js`, `src/endpoints/settings.js`,
`scripts/port_settings_personas_to_files.py`, and the lint script itself
(plus `.md`/`.json` files which are not scanned).

**Failure mode:** Exit code 1, with per-violation output:

```
[lint-settings-persona-access] N violation(s):

  <file>:<line>  matched=/<pattern>/
    <line text>

Persona data lives in PNG cards (<User Avatars>/<key>.png with chara_card_v3
tEXt chunks), NEVER in settings.json. See:
  docs/bio_agent_type_factorization_errata.md
  docs/settings_json_excision_audit.md
  plugins/user-personas/PERSONA_API.md

Allowed sites: src/persona-file-store.js (the projection helpers),
src/endpoints/settings.js (the /save and /get chokepoints), and the
one-shot port script. Any other reference is a regression.
```

Added in sillytavern-fork commit `ca98c7473` ("Sub-14: lint forbidding
plugin-source settings.json persona refs").

### Layer 3 — Filesystem type system (structural layout)

**Files:** PNG layout enforced by `loadPlayers()` at
`plugins/user-personas/index.mjs:447`; agent layout enforced by `loadAgents()`;
axis layout by `loadAxes()`; etc.

The directory structure IS the type system: a file in `User Avatars/` is a bio
slot; a file in `agents/` is an agent slot. They cannot overlap. There is no
API call that writes an agent to `User Avatars/` or a bio to `agents/`. The
`AGENT_INJECTION_MODES_ALLOWED` set (line 182) enforces that agents are always
postfix, rejecting at load time any agent card with a non-`authors_note`
`injection_mode` (lines 954–955).

---

## 5. The Vanilla-Avatar Boundary

A PNG in `<User Avatars>/` without a chara_card_v3 `tEXt` chunk is called a
"vanilla avatar." The operator contract is that vanilla PNGs must not exist
in the bio directory at all. The comment in `loadPlayers()` at
`plugins/user-personas/index.mjs:475–479` states:

> "Every PNG in this directory MUST be a bio card. Plain images without
> chara_card_v3 tEXt chunks should not exist here — they get deleted at
> source, not filtered here."

Prior behavior was to silently skip or fallback to settings.json. **As of
toolcards branch HEAD (`de653b622`, commit message: "delete vanilla PNGs from
bio dir; remove the patch-over filter code")**, `loadPlayers()` now FATAL-errors
with:

```
[user-personas] FATAL: <canonical_key> in User Avatars/ has no chara_card_v3 tEXt chunk: <message>
[user-personas]   Every PNG in <User Avatars>/ must be a bio card. Delete the offending file or port it through the bio designer.
```

and throws: `Error: bio directory contains non-bio PNG: <canonical_key>`
(`plugins/user-personas/index.mjs:487–489`)

This halts `loadPlayers()`, which is called during `init()`, causing the plugin
to fail to load entirely.

**Operator-decreed contract:** vanilla PNGs should not exist in the bio
directory. The right resolution is **deleting them at source** (or porting
them through the bio designer if they represent bios that were never enriched),
not adding a filter. Commit `de653b622` made this irreversible at the code level.

---

## 6. Chokepoints in Upstream ST Code

The two upstream endpoints in `src/endpoints/settings.js` were patched to
enforce the ban structurally. Both are documented with `// CHOKEPOINT:` comments
at their patch sites (lines ~210–264).

### POST /settings/save (line 210)

```js
// CHOKEPOINT: persona data MUST NEVER be written to settings.json.
// Bios live as PNG cards in <User Avatars>/ with chara_card_v3 tEXt
// chunks (see src/persona-file-store.js). Any persona fields that
// arrive here are stripped silently — FE writes to power_user.*
// persona keys are no-ops on the persistence layer. The plugin's
// POST /personas endpoint is the explicit write path for bios.
// Race conditions on settings.json caused the May-2026 bio corpus
// loss; this strip is the structural fix. See
// docs/bio_agent_type_factorization_errata.md.
if (request.body && request.body.power_user) {
    stripPersonaKeysFromPowerUser(request.body.power_user);
}
```

`stripPersonaKeysFromPowerUser` (defined at `src/persona-file-store.js:137`)
zeroes `persona_descriptions`, `personas`, `character_persona_overrides` to
`{}` and `default_persona` to `null` in-place before the JSON is written to
disk. The keys remain as empty objects in the file (for schema compat) but
carry no data.

### POST /settings/get (line 235)

```js
// CHOKEPOINT: persona data lives in PNG cards under <User Avatars>/,
// NOT in settings.json. We project the file-based store into the
// legacy power_user.* shape so FE code that reads persona_descriptions
// / personas keeps working — but the source of truth is files.
// See src/persona-file-store.js + docs/bio_agent_type_factorization_errata.md.
try {
    const snapshot = buildPersonaSnapshotFromFiles(request.user.directories.root);
    const parsed = JSON.parse(settings);
    ...
    parsed.power_user.persona_descriptions = snapshot.persona_descriptions;
    parsed.power_user.personas = snapshot.personas;
    ...
    settings = JSON.stringify(parsed);
} catch (e) {
    console.warn('[persona-file-store] failed to inject file-derived persona snapshot:', e.message);
    // settings continues as the raw string — FE will see the (post-strip)
    // empty persona keys from disk. No fallback to settings.json values.
}
```

`buildPersonaSnapshotFromFiles` (defined at `src/persona-file-store.js:88`) reads
`User Avatars/`, parses `tEXt` chunks from every `.png`, and projects them into
the legacy `power_user.persona_descriptions / personas` shape. PNGs without a
`tEXt` chunk are silently excluded (`if (!card) continue; // vanilla avatar, not
a bio`, line 112). The FE receives a freshly computed snapshot from files on
every `/settings/get` call.

**Net effect:** the FE's `power_user.*` persona keys are always in sync with
the file store. No stale settings.json data reaches the client. No persona data
ever persists in settings.json on disk.

---

## 7. How the FE Persona Panel Discovers Bios

The FE uses two independent channels that must agree:

1. `POST /api/avatars/get` — returns a list of PNG filenames from `User Avatars/`
   (all PNGs, including vanilla ones).
2. `POST /settings/get` — returns the settings blob with `power_user.personas`
   and `power_user.persona_descriptions` projected from the file store
   (bio-only PNGs with valid tEXt).

`public/scripts/personas.js:221–227` — `addMissingPersonas`:

```js
function addMissingPersonas(avatarsList) {
    for (const persona of avatarsList) {
        if (!power_user.personas[persona]) {
            initPersona(persona, '[Unnamed Persona]', '', '');
        }
    }
}
```

Called at `personas.js:252` after `getUserAvatars()` receives the avatar list
from `/api/avatars/get`. If any filename appears in the avatars list but not in
`power_user.personas` (the projected bio store), the FE fabricates an
`[Unnamed Persona]` placeholder with empty description.

**The disagreement scenario:** a vanilla PNG (no `tEXt` chunk) lands in
`User Avatars/`. `/api/avatars/get` returns its filename. `/settings/get` does
NOT include it in `power_user.personas` (because `buildPersonaSnapshotFromFiles`
skips no-tEXt PNGs). `addMissingPersonas` detects the miss and fabricates
`[Unnamed Persona]`. The persona drawer then shows a ghost entry with no bio.

This is why the vanilla-avatar boundary (§5) is enforced as FATAL rather than
a filter: even a silently-skipped vanilla PNG creates a visible FE artifact
(`[Unnamed Persona]`) that confuses operators and hides data-quality problems.

---

## Known Gaps

### Gap 1 — `settings_json_excision_audit.md` violations still live in index.mjs

`docs/settings_json_excision_audit.md` (2026-05-24) identifies three live
settings.json contact sites in `plugins/user-personas/index.mjs` (lines 475–537,
877–915, 2701–2718) as of the audit's writing. The audit doc was authored as a
work-order for excision. The HEAD commit `de653b622` message refers to "Sub-14"
work. Verify whether those three sites have been excised in `de653b622` or
remain pending — the audit doc's "remaining violations" table may be stale
relative to `de653b622`'s changes.

The current `loadPlayers()` (lines 447–506, read above) shows NO settings.json
reads — it reads only from PNG tEXt. This indicates Site 1 was excised.
Whether Sites 2 and 3 (`_writePlayerCardPng` and `POST /personas/:id` fallback)
were excised was not directly verified in this pass; run the lint to confirm:

```bash
node /Users/mdot/sillytavern-fork/plugins/user-personas/scripts/lint_settings_persona_access.mjs
```

### Gap 2 — `docs/multi_user_agent_chat_interface_spec.md` still lists settings.json in bio storage table

The storage table at `docs/multi_user_agent_chat_interface_spec.md:369` still
reads:

> `<dataRoot>/<user>/User Avatars/<key>.png` (avatar) + `settings.json →
> power_user.persona_descriptions[<key>]` (bio text)

This was the pre-excision state. The canonical storage contract is now
PNG-only; the settings.json column in that table is stale and should be
removed from the doc.

### Gap 3 — P-CANONICAL-NOT-MIRRORED in spec still says "one settings.json store"

`docs/multi_user_agent_chat_interface_spec.md:75` lists principle
P-CANONICAL-NOT-MIRRORED as: "Bios = ST personas (one settings.json store)".
This contradicts the excision mandate. The principle body correctly says "Mirror
writes forbidden" and names `_mirrorPersonaToSettingsJson` as banned — but the
parenthetical `(one settings.json store)` should read `(one PNG card store)`.

### Gap 4 — agent.json vs agent.png inconsistency

`docs/bio_agent_type_factorization_errata.md` §3.2 quotes the spec as storing
agents at `plugins/user-personas/agents/<id>.png` (chara_card_v3 PNG).
`docs/user_agent_factorization_spec.md` and harness code reference
`agents/<id>.json`. The current `loadAgents()` reads `.json` files.
The PNG format appears in an older spec layer (commit `844ee3cdf`); the current
implementation uses `.json`. Future agents may need a migration note if the
format ever converges on PNG for consistency with bios.
