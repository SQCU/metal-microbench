// Spec T3 — creating a persona via ST's native Persona Management UI
// dispatches K=2 user-agents synthesis through the plugin.
//
// Surface chain under audit:
//   browser → ST UI (Persona Management drawer → "Create" button →
//             name popup → description textarea)
//   → personas.js emits PERSONAS_UPDATED {action:'create', key}
//   → user-personas FE extension listener receives event
//   → POST /api/plugins/user-personas/synthesize-agents-for-persona/:key
//   → plugin builds one-bio K=2 experiment card + spawns
//     tools/user-agent-harness/lock_in_iterative.mjs
//   → harness writes K=2 agent PNG cards into AGENTS_DIR
//   → plugin's loadAgents() picks them up on child exit (or on the
//     next /agents GET if we reload manually).
//
// What this spec ACTUALLY validates (cheapest signal at each step):
//   1. The "Create" affordance is reachable from the drawer.
//   2. PERSONAS_UPDATED with action='create' fires on the eventSource
//      AFTER the description is typed (the empty→non-empty promotion
//      in personas.js). We hook the event by patching eventSource.emit
//      from inside page.evaluate before the click, so we see exactly
//      what the listener sees.
//   3. The plugin's POST endpoint is hit. We intercept it with
//      page.route to capture the request body + status.
//   4. The plugin returns ok=true with a run_id. (We do NOT wait for
//      the actual K=2 agent PNGs to land — that takes minutes of
//      bridge time and would make the spec flaky / expensive. The
//      synthesis dispatch is the unit of behaviour this spec proves.
//      A follow-up spec can poll /agents until provenance matches if
//      we want to validate end-to-end completion.)
//
// Failure modes this spec catches:
//   - event_types.PERSONAS_UPDATED not defined (regression in events.js)
//   - personas.js no longer emits on create / description-input
//   - FE extension listener not attached (regression in boot())
//   - Plugin endpoint missing or returns 4xx/5xx
//   - Persona key lookup fails (player not in canonical store)

import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';

const PLUGIN_BASE = '/api/plugins/user-personas';
const USER_AVATARS_DIR = '/Users/mdot/metal-microbench/tools/st-debug/_data/default-user/User Avatars';
const SETTINGS_JSON_PATH = '/Users/mdot/metal-microbench/tools/st-debug/_data/default-user/settings.json';
// Two experiment dirs to sweep — st-debug has its own clone of
// sillytavern-fork (see tools/st-debug/CLAUDE.md), but historically
// dispatches sometimes landed in the root clone's dir too. We sweep
// both defensively so harness residue can't accumulate either side.
const EXPERIMENT_DIRS = [
    '/Users/mdot/sillytavern-fork/plugins/user-personas/experiments',
    '/Users/mdot/metal-microbench/tools/st-debug/sillytavern-fork/plugins/user-personas/experiments',
];
// Persona-key patterns that identify residue from this spec's
// dispatches (the test-created persona + every harness-synthesized
// bio). The spec_t4_dispatch_probe variant appears when T4 ran
// against the same dataRoot earlier; T3 owns the synth lifecycle so
// it's safe to sweep that here too.
const RESIDUE_KEY_PATTERNS = [
    /t3synth/,
    /^user-personas-oo-synth-/,
    /^user-personas-oo-spec_t4_dispatch_probe-/,
];
// Hoisted out of afterEach so both afterEach AND afterAll can use it
// (and there's only one definition). isResidueKey returns true for
// any persona key that came from a T3/T4 test run.
const isResidueKey = (k) => RESIDUE_KEY_PATTERNS.some(re => re.test(k));
const TEST_DESCRIPTION = 'A wandering tinker who haggles politely and ' +
    'always seems to know exactly which tools you need before you do. ' +
    'Carries a small notebook covered in margin doodles of gears.';
// 100 < length < 300; lets the synthesis prompt have real material to work with.

test.describe('persona create triggers synthesis', () => {
    test.setTimeout(2 * 60 * 1000);

    // Tracks the persona avatar key created in the test body so that
    // afterEach can clean it up even if the test fails mid-flight. The
    // synth dispatch spawns a node-harness child that, on outer-loop
    // completion, POSTs a fresh /personas/:key write — which recreates
    // the avatar PNG after we've deleted it. Reaping the harness
    // children before the fs sweep is the only way to leave the corpus
    // clean (see afterEach). Reset per test by beforeEach.
    let createdAvatarKey = null;

    test.beforeEach(() => {
        createdAvatarKey = null;
    });

    test.afterEach(async ({ page }) => {
        // Cleanup MUST run even if the test failed. Three phases:
        //   (i)   reap harness child processes so they can't recreate
        //         the avatar PNG mid-cleanup,
        //   (ii)  try ST's native delete UI (#persona_delete_button
        //         after re-selecting our persona) — clears settings.json
        //         too via /api/avatars/delete,
        //   (iii) filesystem sweep against the st-debug dataRoot as the
        //         safety net (UI path silently no-ops if the page is
        //         broken; the file may also reappear if the harness
        //         beat the pkill).
        // We deliberately do NOT use the plugin endpoint for cleanup —
        // it would couple this spec to plugin internals that should be
        // incidental.
        //
        // (i) Reap harness child processes. The harness
        // pipeline (outer_outer.mjs → lock_in_iterative.mjs →
        // axis_splitter.mjs) writes back to /personas/:key on
        // outer-loop completion, which RE-CREATES the avatar PNG we're
        // about to delete (observed empirically: file reappears ~30s
        // after deletion). We pkill broadly across the entire
        // user-agent-harness toolchain rather than just the top-level
        // lock_in_iterative.mjs, because:
        //   (a) outer_outer.mjs spawns lock_in_iterative.mjs as its
        //       child; killing only the child leaves the parent alive
        //       and it just respawns.
        //   (b) page load can re-fire PERSONAS_UPDATED('create') for
        //       every orphan `user-personas-oo-synth-*` PNG already on
        //       disk (via ST's `addMissingPersonas → initPersona`),
        //       spawning additional children we never see the POST
        //       response for. Killing all matching children is the
        //       only way to leave the corpus clean.
        try {
            // SIGKILL (not the default SIGTERM) so the children can't
            // race a final fs write before exiting.
            execSync(`pkill -KILL -f "user-agent-harness/"`, { stdio: 'ignore' });
        } catch {
            // No matching processes — that's fine.
        }
        // Give the kernel a brief moment to flush any in-flight writes
        // from the now-dead children before we do the fs sweep.
        await new Promise(r => setTimeout(r, 250));

        if (!createdAvatarKey) return;
        const keyToDelete = createdAvatarKey;
        createdAvatarKey = null;

        // Try ST's native UI delete first — this clears the settings.json
        // entry + invokes the /api/avatars/delete endpoint to unlink the
        // PNG. We then ALWAYS run a filesystem sweep as a safety net in
        // case the UI path silently fails (observed empirically on some
        // viewport / page-state combos: the avatar tile disappears from
        // the DOM because power_user.personas was mutated, but the
        // backing PNG remains on disk). The fs sweep is idempotent — it
        // only unlinks if the file is still there.
        try {
            const avatar = page.locator(
                `#user_avatar_block .avatar-container[data-avatar-id="${keyToDelete}"]`,
            );
            if (await avatar.count() > 0) {
                await avatar.click({ timeout: 5_000 });
                const delBtn = page.locator('#persona_delete_button');
                await delBtn.click({ timeout: 5_000 });
                // Confirm the popup. Popup.show.confirm renders a dialog
                // with .popup-button-ok (yes/confirm).
                const confirmBtn = page.locator('dialog[open] .popup-button-ok').first();
                await confirmBtn.click({ timeout: 5_000 });
                // Give the network round-trip a moment to land before
                // we check the filesystem.
                await page.waitForTimeout(500);
            }
        } catch (e) {
            console.warn(`  afterEach: UI delete path failed: ${e?.message ?? e}`);
        }

        // Filesystem sweep. The point of the spec's cleanup contract is
        // that the User Avatars dir returns to baseline. Anything still
        // on disk now is residue regardless of how it got there.
        try {
            const target = path.join(USER_AVATARS_DIR, keyToDelete);
            if (fs.existsSync(target)) {
                fs.unlinkSync(target);
                console.log(`  afterEach: fs sweep removed ${keyToDelete}`);
            }
        } catch (e) {
            console.warn(`  afterEach: fs sweep failed: ${e?.message ?? e}`);
        }

        // Also sweep any synth-derived orphan PNGs the harness may have
        // written before we killed it (or in previous test runs that
        // never cleaned up). The harness writes these as
        // `user-personas-oo-synth-<keyslug>-<hash>-pass<N>.png`. They
        // pollute the corpus and re-trigger PERSONAS_UPDATED('create')
        // on every subsequent page load via `addMissingPersonas`.
        try {
            for (const name of fs.readdirSync(USER_AVATARS_DIR)) {
                if (name.startsWith('user-personas-oo-synth-') && name.endsWith('.png')) {
                    fs.unlinkSync(path.join(USER_AVATARS_DIR, name));
                    console.log(`  afterEach: swept orphan ${name}`);
                }
                // Also sweep T4 dispatch-probe orphan PNGs if any
                // landed in this dataRoot (T3 owns the synth lifecycle).
                if (name.startsWith('user-personas-oo-spec_t4_dispatch_probe-') && name.endsWith('.png')) {
                    fs.unlinkSync(path.join(USER_AVATARS_DIR, name));
                    console.log(`  afterEach: swept T4-probe orphan ${name}`);
                }
            }
        } catch (e) {
            console.warn(`  afterEach: orphan sweep failed: ${e?.message ?? e}`);
        }

        // (iv) settings.json sweep. The UI delete path above clears
        // ONLY the test-created persona's entry; the harness-spawned
        // user-personas-oo-synth-* bios (and any T4 dispatch-probe
        // bios that came along for the ride) leave their own entries
        // in power_user.persona_descriptions + power_user.personas
        // whenever ST's addMissingPersonas → initPersona pipeline
        // discovers their PNGs on disk. We sweep every key matching
        // the residue patterns, atomically rewrite settings.json (tmp
        // + rename so a crash mid-write can't corrupt the file), and
        // let ST re-read on its next access.
        try {
            const raw = fs.readFileSync(SETTINGS_JSON_PATH, 'utf8');
            const settings = JSON.parse(raw);
            const pu = settings.power_user ?? {};
            const pds = pu.persona_descriptions ?? {};
            const personas = pu.personas ?? {};
            const removedKeys = [];
            for (const k of Object.keys(pds)) {
                if (isResidueKey(k)) {
                    delete pds[k];
                    removedKeys.push(k);
                }
            }
            for (const k of Object.keys(personas)) {
                if (isResidueKey(k)) {
                    delete personas[k];
                    if (!removedKeys.includes(k)) removedKeys.push(k);
                }
            }
            if (removedKeys.length > 0) {
                const tmp = `${SETTINGS_JSON_PATH}.t3-cleanup-${process.pid}-${Date.now()}.tmp`;
                fs.writeFileSync(tmp, JSON.stringify(settings, null, 4));
                fs.renameSync(tmp, SETTINGS_JSON_PATH);
                console.log(`  afterEach: settings.json swept ${removedKeys.length} residue key(s): ${removedKeys.join(', ')}`);
            }
        } catch (e) {
            console.warn(`  afterEach: settings.json sweep failed: ${e?.message ?? e}`);
        }

        // (v) experiment-card sweep. The plugin's
        // synthesize-agents-for-persona endpoint writes one
        // `experiments/synth-<persona>-<hex>.json` per dispatch into
        // the plugin source dir. We sweep both the root and the
        // st-debug-clone dirs (see CLAUDE.md — st-debug has its own
        // clone, but residue has historically landed in both). The
        // `synth-` prefix is unique to harness dispatches; T4's
        // spec-owned fixture is `spec_t4_dispatch_probe.json` (no
        // `synth-` prefix) and we explicitly never touch it.
        for (const dir of EXPERIMENT_DIRS) {
            try {
                if (!fs.existsSync(dir)) continue;
                for (const name of fs.readdirSync(dir)) {
                    if (name.startsWith('synth-') && name.endsWith('.json')) {
                        fs.unlinkSync(path.join(dir, name));
                        console.log(`  afterEach: swept experiment ${name} from ${dir}`);
                    }
                }
            } catch (e) {
                console.warn(`  afterEach: experiment sweep failed for ${dir}: ${e?.message ?? e}`);
            }
        }
    });

    // afterAll: catch the post-afterEach respawn. Observed empirically:
    // ST's debounced settings.json save lands AFTER the test ends, the
    // PERSONAS_UPDATED('create') event fires for the persona, the FE
    // hook POSTs to /synthesize-agents-for-persona, the plugin spawns a
    // FRESH outer_outer.mjs detached child whose lifecycle is no longer
    // inside any test's afterEach scope. Across 3 viewports running this
    // spec sequentially, this can leave 3+ harness processes plus their
    // residue files alive. The suite-end sweep below catches them.
    //
    // This is a defense-in-depth backstop, not a substitute for afterEach;
    // afterEach catches the per-test happy path, afterAll catches the
    // post-test-respawn race.
    test.afterAll(async () => {
        // Loop: sleep + pkill until two consecutive passes find no
        // running harness processes. This catches the timing where ST's
        // debounced settings.json save fires PERSONAS_UPDATED ~hundreds
        // of ms after the test ended, spawning a fresh outer_outer.mjs
        // that's still in the process-tree handoff when the first pkill
        // ran (so it didn't match). Bounded at 15 iterations × 1s =
        // 15s max, well within the suite-end window.
        const harnessRunning = () => {
            try {
                // pgrep returns 0 if matches found, 1 if none.
                execSync(`pgrep -f "user-agent-harness/"`, { stdio: 'ignore' });
                return true;
            } catch {
                return false;
            }
        };
        let stableZeroPasses = 0;
        for (let i = 0; i < 15 && stableZeroPasses < 2; i++) {
            await new Promise(r => setTimeout(r, 1000));
            try {
                execSync(`pkill -KILL -f "user-agent-harness/"`, { stdio: 'ignore' });
            } catch { /* no matches OK */ }
            if (!harnessRunning()) {
                stableZeroPasses++;
            } else {
                stableZeroPasses = 0;
                console.log(`  afterAll: pkill iter ${i + 1}: respawned harness procs found, retrying`);
            }
        }

        // Re-do the full residue sweep (User Avatars/ + settings.json +
        // experiments/). This is idempotent — only removes residue that
        // matches the patterns; never touches keep-set files.
        const AVATARS_DIR = '/Users/mdot/metal-microbench/tools/st-debug/_data/default-user/User Avatars';
        try {
            if (fs.existsSync(AVATARS_DIR)) {
                for (const fn of fs.readdirSync(AVATARS_DIR)) {
                    if (fn.includes('t3synth') ||
                        fn.startsWith('user-personas-oo-synth-') ||
                        fn.startsWith('user-personas-oo-spec_t4_dispatch_probe-')) {
                        fs.unlinkSync(path.join(AVATARS_DIR, fn));
                        console.log(`  afterAll: swept avatar ${fn}`);
                    }
                }
            }
        } catch (e) {
            console.warn(`  afterAll: avatar sweep failed: ${e?.message ?? e}`);
        }

        try {
            const raw = fs.readFileSync(SETTINGS_JSON_PATH, 'utf8');
            const settings = JSON.parse(raw);
            const pds = settings?.power_user?.persona_descriptions || {};
            const personas = settings?.power_user?.personas || {};
            const swept = [];
            for (const k of Object.keys(pds)) {
                if (isResidueKey(k)) { delete pds[k]; swept.push(k); }
            }
            for (const k of Object.keys(personas)) {
                if (isResidueKey(k)) { delete personas[k]; if (!swept.includes(k)) swept.push(k); }
            }
            if (swept.length > 0) {
                const tmp = `${SETTINGS_JSON_PATH}.t3-afterAll-${process.pid}-${Date.now()}.tmp`;
                fs.writeFileSync(tmp, JSON.stringify(settings, null, 4));
                fs.renameSync(tmp, SETTINGS_JSON_PATH);
                console.log(`  afterAll: settings.json swept ${swept.length} residue key(s): ${swept.join(', ')}`);
            }
        } catch (e) {
            console.warn(`  afterAll: settings.json sweep failed: ${e?.message ?? e}`);
        }

        for (const dir of EXPERIMENT_DIRS) {
            try {
                if (!fs.existsSync(dir)) continue;
                for (const name of fs.readdirSync(dir)) {
                    if (name.startsWith('synth-') && name.endsWith('.json')) {
                        fs.unlinkSync(path.join(dir, name));
                        console.log(`  afterAll: swept experiment ${name} from ${dir}`);
                    }
                }
            } catch (e) {
                console.warn(`  afterAll: experiment sweep failed for ${dir}: ${e?.message ?? e}`);
            }
        }
    });

    test('create → PERSONAS_UPDATED → plugin synth POST', async ({ page }) => {
        // ── (1) Intercept the plugin POST so we can assert it landed. ──
        // Pass-through (continue) so the actual synth dispatch still
        // happens server-side; we only observe. We track ALL synth
        // POSTs (since orphan synth-derived PNGs on disk from earlier
        // test runs can trigger their own PERSONAS_UPDATED('create')
        // emits via ST's addMissingPersonas → initPersona pipeline on
        // page load), but the assertions below filter to the POST that
        // targets THIS test's avatar key. Capture an array, not a
        // singleton, to be robust against the orphan-driven noise.
        const capturedAll = [];
        await page.route('**/api/plugins/user-personas/synthesize-agents-for-persona/*', async (route) => {
            const req = route.request();
            const rec = { method: req.method(), url: req.url(), body: req.postData(), status: null, responseJson: null };
            const resp = await route.fetch();
            rec.status = resp.status();
            try { rec.responseJson = await resp.json(); }
            catch { rec.responseJson = null; }
            capturedAll.push(rec);
            await route.fulfill({ response: resp });
        });

        // ── (2) Open ST. (Connecting to the API isn't required for the
        //        persona-create flow — it's a settings-only mutation. We
        //        still wait for the preloader to clear and for the
        //        SillyTavern global to exist.) ──
        await page.goto('/');
        await page.waitForFunction(
            'document.getElementById("preloader") === null',
            { timeout: 60_000 });
        await page.waitForFunction(
            () => typeof window.SillyTavern?.getContext === 'function',
            { timeout: 30_000 });

        // Confirm the event type was added to events.js. This is the
        // structural precondition for the rest of the test.
        const eventTypeName = await page.evaluate(async () => {
            const mod = await import('/scripts/events.js');
            return mod.event_types?.PERSONAS_UPDATED ?? null;
        });
        expect(eventTypeName, 'event_types.PERSONAS_UPDATED defined in events.js')
            .toBe('personas_updated');

        // ── (3) Hook the eventSource so we can observe the emit. We
        //        wrap eventSource.emit and stash every PERSONAS_UPDATED
        //        payload on window.__personasUpdatedLog. This is
        //        OBSERVATION ONLY — we don't suppress or modify the
        //        event, so the real listener still runs. ──
        await page.evaluate(async () => {
            const mod = await import('/scripts/events.js');
            window.__personasUpdatedLog = [];
            const originalEmit = mod.eventSource.emit.bind(mod.eventSource);
            mod.eventSource.emit = function (eventType, ...args) {
                if (eventType === mod.event_types.PERSONAS_UPDATED) {
                    window.__personasUpdatedLog.push({
                        ts: Date.now(),
                        payload: args[0],
                    });
                }
                return originalEmit(eventType, ...args);
            };
        });

        // ── (4) Open the Persona Management drawer. ──
        const drawerToggle = page.locator('#persona-management-button .drawer-toggle');
        await expect(drawerToggle, 'Persona Management drawer present').toBeVisible({ timeout: 15_000 });
        await drawerToggle.click();
        // Drawer content becomes visible. The "Create" button is
        // #create_dummy_persona inside it.
        const createBtn = page.locator('#create_dummy_persona');
        await expect(createBtn, 'Create button visible after drawer open').toBeVisible({ timeout: 10_000 });

        // ── (5) Click "Create" and fill the name popup. Names must be
        //        unique across runs — embed a timestamp. ──
        const uniqueName = `t3-synth-${Date.now()}`;
        await createBtn.click();
        // ST's Popup-class input renders as an <input> inside an open
        // <dialog>. Type the name and confirm.
        const popupInput = page.locator('dialog[open] input[type="text"], dialog[open] textarea').first();
        await expect(popupInput, 'name popup input visible').toBeVisible({ timeout: 10_000 });
        await popupInput.fill(uniqueName);
        await page.locator('dialog[open] .popup-button-ok').first().click();

        // ── (6) Wait for the persona to land in settings (the new avatar
        //        block appears in #user_avatar_block). The avatarId
        //        format is `${Date.now()}-${name-stripped}.png`. ──
        const expectedKeyContains = uniqueName.replace(/-/g, '');
        // The avatar tile uses data-avatar-id="<full filename>".
        const newAvatar = page.locator(`#user_avatar_block .avatar-container[data-avatar-id*="${expectedKeyContains}"]`);
        await expect(newAvatar, 'new persona avatar rendered in drawer').toBeVisible({ timeout: 15_000 });
        const avatarKey = await newAvatar.getAttribute('data-avatar-id');
        expect(avatarKey, 'extracted avatar key').toMatch(/\.png$/);
        // Record the key so afterEach can clean it up even on failure.
        createdAvatarKey = avatarKey;

        // ── (7) Click the new avatar to select it (so #persona_description
        //        binds to it), then type the description into the
        //        textarea. The empty→non-empty transition is what
        //        triggers the second PERSONAS_UPDATED {action:'create'}
        //        emit in personas.js. ──
        await newAvatar.click();
        // Wait for the selection to bind: #persona_description should
        // become enabled and the textarea text should clear.
        const descTextarea = page.locator('#persona_description');
        await expect(descTextarea, 'description textarea visible').toBeVisible({ timeout: 10_000 });
        // Programmatically fill + trigger 'input' so onPersonaDescriptionInput
        // runs (ST listens with $.on('input', ...)). We use .fill() which
        // dispatches the right input event in Playwright.
        await descTextarea.fill(TEST_DESCRIPTION);

        // ── (8) Wait for PERSONAS_UPDATED to be observed AND for the
        //        plugin POST to land. The first emit fires on initPersona
        //        (action='create', empty bio). The second emit fires on
        //        the description input's empty→non-empty transition
        //        (action='create' again — promotes from empty). The FE
        //        listener dedupes within 5s, so only one POST should
        //        actually fire. ──
        await page.waitForFunction(
            () => Array.isArray(window.__personasUpdatedLog)
                && window.__personasUpdatedLog.length >= 1,
            { timeout: 30_000 });
        const log = await page.evaluate(() => window.__personasUpdatedLog);
        console.log(`  PERSONAS_UPDATED events observed:`, JSON.stringify(log));
        const createEvents = log.filter(e => e.payload?.action === 'create');
        expect(createEvents.length, 'at least one create-action event emitted').toBeGreaterThanOrEqual(1);
        const matchedKey = createEvents.find(e => e.payload?.key === avatarKey);
        expect(matchedKey, `create event for key=${avatarKey} observed`).toBeTruthy();

        // ── (9) Wait for the plugin synth POST targeting OUR avatar key.
        //        The listener fires fire-and-forget; route.fetch()
        //        resolves once the plugin returns. We use a key-matching
        //        predicate (raw, encoded, or decoded-path forms) so the
        //        check is robust against URL normalisation differences
        //        across viewports / browser contexts. ──
        const urlTargetsKey = (url, key) => {
            let decoded = url;
            try { decoded = decodeURIComponent(url); } catch { /* leave as-is */ }
            return url.includes(key)
                || url.includes(encodeURIComponent(key))
                || decoded.includes(key);
        };
        let captured = null;
        await expect.poll(() => {
            captured = capturedAll.find(c => urlTargetsKey(c.url, avatarKey));
            return captured ? 'found' : 'not-yet';
        }, {
            message: `plugin synthesize-agents-for-persona endpoint was POSTed for ${avatarKey}`,
            timeout: 30_000,
            intervals: [500, 1000, 2000],
        }).toBe('found');

        expect(captured.method, 'captured request method').toBe('POST');
        expect(captured.status, 'plugin returned 2xx').toBeLessThan(300);
        expect(captured.responseJson, 'plugin returned JSON body').not.toBeNull();
        expect(captured.responseJson.ok, 'plugin response.ok === true').toBe(true);
        expect(captured.responseJson.run_id, 'plugin returned run_id').toBeTruthy();
        expect(captured.responseJson.experiment_id, 'plugin returned experiment_id')
            .toMatch(/^synth-/);
        expect(captured.responseJson.persona_key, 'plugin echoed persona_key')
            .toBe(avatarKey);

        // ── (10) Soft signal — query /agents to record what's currently
        //         visible. We don't ASSERT K=2 agents present (the
        //         synthesis itself takes minutes; the dispatch is what
        //         this spec proves). But we log the current state so
        //         we can correlate with a longer-running follow-up. ──
        const agentsResp = await page.evaluate(async () => {
            const r = await fetch('/api/plugins/user-personas/agents');
            if (!r.ok) return { error: `HTTP ${r.status}` };
            return await r.json();
        });
        console.log(`  /agents snapshot at end of test: count=${agentsResp.count ?? 'unknown'}`);

        // ── (11) Cleanup of the created persona happens in afterEach
        //         so it runs even if any of the above assertions fail.
        //         The synth run-id artefacts (agent PNGs / experiment
        //         card) outside _data/default-user/User Avatars/ may
        //         still land asynchronously; those live under the
        //         plugin's AGENTS_DIR and are out of scope for this
        //         spec's cleanup. ──
    });
});
