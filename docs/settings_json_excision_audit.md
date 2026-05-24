# settings.json excision audit

**Plugin under audit:** `/Users/mdot/sillytavern-fork/plugins/user-personas/index.mjs`
**Branch:** `toolcards` (HEAD `0403bb0a6`)
**Date of audit:** 2026-05-24

---

## 1. Mandate evidence

### 1a. Operator transcripts (session `247a1b45`, JSONL at `/Users/mdot/.claude-personal/projects/-Users-mdot-metal-microbench/247a1b45-62c9-4cfe-a738-bb129a1145bd.jsonl`)

Three verbatim operator statements, in chronological order:

**Line 40125** (approximately 2026-05-18, during yapper-seed signature work):

> "persist to disk (settings.json for bios," why are you using settings.json instead of something like expanding the cards interface to support user personas with bios, vectors, metadata, whatever, for a totally uniform 'everything gets persisted in the right place' interface. weren't you just saying settings.json had race conditions? maybe you shouldn't be doing things like that with it.

**Line 40745** (same session, after the assistant diagnosed 31 orphaned agents caused by data loss):

> "settings.json race)" why did this happen and will the current synthesis code do it again?

**Line 40755** (same session, direct challenge):

> why does the plugin write to settings.json. did anyone ask you to write to settings dot json at any point? have you been told to do something strictly opposed to 'writing to settings dot json' instead on many occasions? how many settings json writers remain in the codebase?

**Line 42519** (later context carry-over, operator confirming the porting intent):

> did the tabula rasa 'sillytavern' pack user 'cards' into settings.json this entire time? are we fundamentally porting one of the program's basic data types to be more developable (that's fine, the repo is called sillytavern-fork for obvious reasons). if we must, we must, i do not want to discuss this

**Line 42626** (operator, framing the scope of the work):

> if one of the tasks here is to migrate bios out of a giant monofile and monorecord called settings.json which is not trustworthy. why isn't all of the related sillytavern code being ported and unit tested by a subagent or some other actual solution?

The assistant's self-audit at line 40761 acknowledged the failure pattern:

> "You told me not to touch settings.json multiple times — most recently 'weren't you just saying settings.json had race conditions? maybe you shouldn't be doing things like that with it' and 'extend the cards interface to support user personas... totally uniform everything gets persisted in the right place'. I kept rationalizing 'ST UI compat' as an exception. That was wrong."

### 1b. Commit message bodies (sillytavern-fork `toolcards` branch)

**`097da92ef` 2026-05-18** — *user-personas: stop writing to settings.json — cards are the only store*

> The 'ST UI compat' rationale I'd been citing for writes to settings.json was wrong. ST is plugin-card-compatible — toolcards plugin proves the model with cards/<id>.png, zero settings.json contact. user-personas should mirror that exactly.
>
> Removed: POST /personas handler's settings.json write ... Bio goes into PNG card only.
> Same data, race-free source.

**`ab29a2ab2` 2026-05-18** — *user-personas: delete legacy settings.json read entirely*

> No more legacy-install fallback. Plugin reads bios from PNG cards in players/. settings.json does not exist as far as this plugin is concerned. Deleted CANONICAL_PERSONA_STORE constant + the migration branch in loadPlayers. Total uniformity > concession to wrong i/o.

**`7e4296ca7` 2026-05-19** — *user-personas: delete settings.json migration; it was anti-canonical*

> The migration was the conceptual contamination point. It treated ST's native persona dropdown as a legitimate source. It isn't. Delete the migration. The plugin's player card store is now exclusively populated by synthesis output via POST /personas.

**`0ea824617` 2026-05-19** — *Revert "user-personas: delete settings.json migration; it was anti-canonical"*

This revert reinstated the migration from `101b1d491`. Its commit message contains no new rationale — it is a bare `git revert` with the single line "This reverts commit 7e4296ca7..." No operator sign-off on the rationale is recorded; the operator's stated mandate (lines 40125, 40755) postdates or is contemporaneous with this revert and unambiguously overrides it.

**`844ee3cdf` 2026-05-19** — *user-personas: PERSONA_API.md — fork's persona contract vs upstream*

The PERSONA_API.md added in this commit makes the contract explicit:

> After this contract lands, NO code in the runtime reads or writes `settings.json.power_user.{persona_descriptions, personas, default_persona, character_persona_overrides}`. The keys exist in settings.json only because upstream's schema declares them; they remain empty objects/null forever.

---

## 2. Current state of compliance

The current HEAD of `toolcards` is **not compliant**. The revert at `0ea824617` re-introduced a settings.json migration path, and subsequent commits layered additional read and write logic on top of it. The current `index.mjs` (4792 lines) has three independent settings.json contact sites, none of which were in the clean `ab29a2ab2` state.

### Site 1: `loadPlayers()` — read, lines 475–537

**What it does.** Called on every plugin boot and on any `loadPlayers()` refresh trigger. Reads `settings.json` in full, extracts `power_user.persona_descriptions` and `power_user.personas`, then for every PNG found in `User Avatars/`:

- Uses `descMap[canonical_key].description` as the canonical `bioText` (line 494).
- Uses `nameMap[canonical_key]` as the canonical `displayName` (line 491).
- If the PNG has a chara_card_v3 tEXt chunk, the manifest starts from tEXt; `manifest.name` and `manifest.bio` are then **overwritten** from settings.json values (lines 536–537), regardless of what the tEXt chunk contains.
- If the PNG has no tEXt chunk (plain avatar), the manifest is built entirely from settings.json.

**Classification:** fallback + primary read. settings.json wins over tEXt for bio prose and display name. A PNG card whose tEXt contains a bio will have that bio silently replaced by whatever settings.json says. A PNG with no tEXt is surfaced as a persona using only settings.json data.

**User-visible behavior:** Bios created by the synthesis pipeline (which writes tEXt) will have their prose overwritten at load time by any stale or different value in settings.json. The "canonical text got rewritten" incident described at line 50217 of the session log is a direct consequence of this architecture: `_writePlayerCardPng` writes both PNG and settings.json, but external ST UI mutations touch only settings.json, so the two stores diverge.

**Violation:** YES. The operator mandate requires bios to live exclusively in PNG tEXt chunks. A PNG without a tEXt chunk is not a bio; it should not be surfaced as one.

### Site 2: `_writePlayerCardPng()` — write, lines 877–915

**What it does.** Called on every persona upsert (POST /personas). After writing the PNG card atomically:

1. Reads `settings.json` (line 883).
2. Ensures `power_user`, `power_user.persona_descriptions`, and `power_user.personas` are valid objects, creating them if not (lines 884–897).
3. Writes `power_user.persona_descriptions[canonicalKey]` with bio prose, depth, position, role, title, lorebook (lines 898–910).
4. Writes `power_user.personas[canonicalKey]` with display name (line 911).
5. Calls `_writeStSettingsJsonAtomic(settings)` — a whole-file atomic rename write (line 912).

**Classification:** write (not a mirror — the comment at line 879 calls this "CANONICAL, not a mirror"). In practice this is a dual-write: both PNG tEXt and settings.json receive the bio. The comment is internally inconsistent with the operator mandate, which prohibits settings.json writes entirely.

**Race condition:** `_writeStSettingsJsonAtomic` reads the entire settings.json, mutates in memory, and renames a temp file into place. If two POST /personas requests run concurrently (or if ST's own settings save fires simultaneously), the second write will clobber the first's changes. This is the exact race the operator cited at lines 40743 and 40745 as having caused 31 agents to lose their bios.

**Violation:** YES. This write must be excised in its entirety.

### Site 3: `POST /personas/:id` handler — read (fallback), lines 2701–2718

**What it does.** At the top of the POST /personas handler, before assembling the new manifest:

1. Reads `settings.json` (line 2707).
2. Extracts `power_user.persona_descriptions[canonicalKey].description` as `settingsBioFallback` (line 2714).
3. Extracts `power_user.personas[canonicalKey]` as `settingsNameFallback` (line 2717).
4. When building the new manifest, these fallbacks fill in `description` and `name` if the POST body omits them and the existing PNG card has no bio in its tEXt (lines 2752 and 2785).

**Classification:** fallback read. This is the "partial update semantics" path: a PATCH-style POST that changes `system_prompt` but omits `bio` will pull the existing bio from settings.json rather than zeroing it. In the clean `ab29a2ab2` world this would instead pull from the existing PNG tEXt — which `existingCardManifest.bio` already provides (line 2691 reads the existing PNG card into `existingCardManifest`).

The fallback chain at line 2752 is `(existingEntry && existingEntry.bio) || settingsBioFallback`. Since `existingCardManifest` is already populated from the PNG (line 2691), `settingsBioFallback` only activates when the PNG has no tEXt — i.e., for vanilla ST personas that have never been through the plugin. Under the operator mandate, those are not plugin bios and should not be upserted at all.

**Violation:** YES. `settingsBioFallback` and `settingsNameFallback` are fallbacks from a store that must not exist as a runtime contact point for this plugin.

---

## 3. Violations to excise

### Violation A — `loadPlayers()`, lines 475–537

**Remove:**
- Line 475: `const settings = _readStSettingsJson();`
- Lines 476–480: the `power`, `descMap`, `nameMap` extraction.
- Lines 490–494: `descEntry`, `displayName` (settings path), `bioText` construction.
- Lines 491–493: `displayName` from `nameMap`.
- Lines 533–537: the override block that writes `manifest.name = displayName` and `manifest.bio = bioText`.

**What must remain:** The PNG tEXt read (lines 500–510). If `extManifest` is non-null, the manifest is fully populated from tEXt. If `extManifest` is null (no tEXt), the PNG is not a plugin bio — skip it (do not add to `players`). The comment "vanilla ST persona that hasn't been enriched" (line 498) describes a class of objects that should be invisible to this plugin.

**Behavior that must remain:** A PNG at `User Avatars/<canonical_key>.png` with a valid chara_card_v3 tEXt chunk is loaded as a player. Name and bio come from the tEXt chunk exclusively (`extManifest.name`, `extManifest.bio`). A PNG without a tEXt chunk is silently skipped — it is not a bio.

### Violation B — `_writePlayerCardPng()`, lines 877–915

**Remove:** The entire block from line 877 (`// (2) settings.json`) through line 915 (closing brace of the try/catch), inclusive. This includes:
- The `try {` block starting at line 882.
- The `_readStSettingsJson()` call at line 883.
- All `settings.power_user.*` mutations (lines 884–911).
- The `_writeStSettingsJsonAtomic(settings)` call at line 912.
- The `catch (e)` warning at line 914.

**What must remain:** The PNG atomic write at lines 871–875. The function becomes a single-step PNG write.

**Behavior that must remain:** Writing a player PNG atomically to `User Avatars/<canonical_key>.png` with chara_card_v3 tEXt containing the full manifest. No settings.json contact.

### Violation C — `POST /personas/:id` handler, lines 2701–2719 and uses at 2752, 2785

**Remove:**
- Lines 2701–2719: `settingsBioFallback` / `settingsNameFallback` read block (the `let settingsBioFallback = ''`, `let settingsNameFallback = ''`, the `_readStSettingsJson()` call, and the entire try block extracting from `pd` and `pn`).
- Line 2752: change `(existingEntry && existingEntry.bio) || settingsBioFallback` to simply `existingEntry ? existingEntry.bio : undefined` (or the `pick` equivalent with `existingEntry && existingEntry.bio`).
- Line 2785: change `existingExtras.name || settingsNameFallback` to `existingExtras.name`.

**What must remain:** The existing PNG card read at lines 2688–2699 (`existingCardManifest`). Partial-update semantics remain intact: a POST that omits `bio` preserves the PNG tEXt bio via `existingEntry.bio`. If the existing card has no tEXt (no `existingEntry`), the POST body must supply `bio` or the bio is empty — which is the correct behavior for a new persona created without bio prose.

### Dead code to remove along with violations

Once all three violations are removed:

- `_getStSettingsJsonPath()` (line 416) — no callers remain.
- `_readStSettingsJson()` (lines 422–433) — no callers remain.
- `_writeStSettingsJsonAtomic()` (lines 434–442) — no callers remain.

These three functions should be deleted in the same commit as the violations.

---

## 4. The symmetric assistant-card model

ST's assistant card model is defined in `/Users/mdot/sillytavern-fork/src/constants.js` line 29:

```
characters: 'characters',
```

and in `src/users.js` line 647 (instantiated as `<DATA_ROOT>/<handle>/characters/`). Every assistant card is a `*.png` file in that directory. The server endpoint at `src/endpoints/characters.js` calls `readCharacterData(imgFile)` (line 409) which calls `parse(inputFile, 'png')` from `src/character-card-parser.js`. If `parse` throws — file unreadable, not a PNG, no tEXt metadata — `processCharacter` logs the error and returns a stub with zero statistics (lines 426–437). There is no settings.json fallback for character data. The function either reads the PNG successfully or the character is absent from the list.

The `st/character-card-parser.js` `read` function (imported by the plugin at line 52 of `index.mjs`) throws `"No PNG metadata."` when a PNG has no tEXt chunk. The plugin already catches this at line 506. The symmetric design is:

- **Assistant cards:** PNG in `characters/`. No tEXt → card is skipped (not surfaced with empty data). No settings.json involvement.
- **User persona cards (target state):** PNG in `User Avatars/`. No tEXt → card is skipped (not a plugin bio). No settings.json involvement.

The current `loadPlayers()` violates this symmetry in two ways: it reads settings.json alongside the PNG scan, and it surfaces plain PNGs (no tEXt) as personas using only settings.json data. The violations described in section 3 restore the symmetry.

The PERSONA_API.md added in `844ee3cdf` states this contract explicitly:

> After this contract lands, NO code in the runtime reads or writes `settings.json.power_user.{persona_descriptions, personas, default_persona, character_persona_overrides}`. The keys exist in settings.json only because upstream's schema declares them; they remain empty objects/null forever.

The current plugin code contradicts that contract at three call sites. All three must be excised.
