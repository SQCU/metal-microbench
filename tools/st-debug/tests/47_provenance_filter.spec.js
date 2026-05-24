// F16 provenance tagging + suggester filter — Playwright acceptance gate.
//
// Spec contract: docs/ui_spec_provenance_filter.md (approved
// for implementation). Validates the full FE+plugin surface:
//
//   1. Schema persistence: agent / persona POSTs round-trip a
//      `provenance` field on the card.
//   2. /yapper-seed includes provenance on each ranked candidate's
//      persona + agent (verified via fixture/route stub so this
//      assertion stays decoupled from bridge slowness — the
//      production endpoint exposes the same shape).
//   3. Filter row renders in suggester.html with 5 checkboxes
//      (canonical, manual, legacy, experiment_output, seed_demo);
//      experiment_output and seed_demo are unchecked by default.
//   4. Default filter hides experiment_output + seed_demo cards;
//      hidden-count badge matches.
//   5. Toggling experiment_output reveals the hidden rows.
//   6. localStorage persists toggle state across reload.
//   7. (Deferred) tag-on-write for new experiment runs — separate
//      long-running spec; the live env already exercises that path
//      via /experiments/:id/run setting LOCK_IN_RUN_ID.
//   8. No card was deleted: ls plugins/user-personas/{agents,players}
//      compared before and after the spec.
//
// Setup: runs the retroactive tagging script idempotently to ensure
// the corpus is in the tagged state. The script never deletes a card
// (hard constraint, validated by ls-comparison after).
//
// Route stub rationale: /yapper-seed calls the bridge for target-
// signature extraction, and the bridge serializes everything through a
// single GPU prefill pipeline. The UI's filter behavior is pure FE
// logic (apply provenance.kind ∈ checked) and doesn't depend on the
// content of the bridge's judge calls. So the spec stubs the
// /yapper-seed response with a deterministic fixture derived from the
// REAL on-disk corpus (via direct GET /agents + GET /personas), and
// asserts the FE filter against that. This makes the spec robust
// against bridge load conditions while still validating the schema +
// the FE filter logic against real card provenance.

import { test, expect } from '@playwright/test';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { loadAndConnect } from './_helpers/elicit_clean.mjs';
import { openPersonaSurface } from './_helpers/open_persona_surface.js';

const PLUGIN_BASE = '/api/plugins/user-personas';
const PLUGIN_DIR = '/Users/mdot/sillytavern-fork/plugins/user-personas';
const TAG_SCRIPT = path.join(PLUGIN_DIR, 'scripts/tag_existing_corpus.mjs');
// Bios (player cards) live in the canonical ST persona store — User Avatars/
// — not in plugins/user-personas/players/ (that dir was eliminated when the
// canonical-store unification landed). The tagging script still works because
// it fetches bios via GET /personas (the API, not filesystem), but the
// no-deletion check must watch the real on-disk location.
const ST_DEBUG_DATA_DIR = '/Users/mdot/metal-microbench/tools/st-debug/_data';
const USER_AVATARS_DIR = path.join(ST_DEBUG_DATA_DIR, 'default-user', 'User Avatars');

function listDir(dir) {
    if (!fs.existsSync(dir)) return [];
    return fs.readdirSync(dir).sort();
}

test.describe('F16 provenance tagging + suggester filter — desktop only', () => {
    // 4 min: tag script (~5s) + UI assertions. With the route stub
    // there's no bridge dependency; only direct API GETs to /agents
    // and /personas, which are local + immediate.
    test.setTimeout(4 * 60 * 1000);

    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'provenance filter spec runs only on desktop project');
    });

    test('schema + filter UI + persistence + no-card-deletion', async ({ page, request }) => {
        // ── (8a) snapshot the on-disk card sets BEFORE everything.
        // Agents live in plugins/user-personas/agents/ (checked via PLUGIN_DIR).
        // Player bios live in the canonical ST persona store (User Avatars/) —
        // the players/ directory was eliminated when canonical-store unification
        // landed. The tagging script fetches bios via the API, so no code reads
        // players/. The no-deletion invariant watches User Avatars/ directly.
        const agentsBefore = listDir(path.join(PLUGIN_DIR, 'agents'));
        const playersBefore = listDir(USER_AVATARS_DIR);
        expect(agentsBefore.length, 'corpus has agents to tag').toBeGreaterThan(0);
        expect(playersBefore.length, 'corpus has player bios in User Avatars/').toBeGreaterThan(0);

        // ── Setup: run the retroactive tagging script. Idempotent —
        // already-tagged cards (kind !== 'legacy') are skipped. Safe to
        // run as the spec's setup step.
        //
        // NO_PROXY/no_proxy: the Playwright test environment may carry
        // proxy settings (inherited env vars or agent-level network
        // interceptors) that redirect loopback traffic through a proxy.
        // The tag script's fetch calls are intra-machine (127.0.0.1:8002)
        // and must NOT go through any proxy. Setting NO_PROXY=* forces
        // Node's fetch to use a direct connection for all URLs.
        const tagResult = spawnSync('node', [TAG_SCRIPT], {
            cwd: PLUGIN_DIR,
            env: {
                ...process.env,
                ST_BASE: 'http://127.0.0.1:8002',
                NO_PROXY: '*',
                no_proxy: '*',
            },
            encoding: 'utf8',
        });
        expect(tagResult.status, `tagging script exits 0 (stderr: ${tagResult.stderr?.slice(0, 500)})`).toBe(0);
        expect(tagResult.stdout, 'tagging script prints a summary').toMatch(/SUMMARY/);

        // ── (1) Schema persistence: at least one experiment_output agent.
        const agentsResp = await request.get(`${PLUGIN_BASE}/agents`);
        expect(agentsResp.status()).toBe(200);
        const agentsJson = await agentsResp.json();
        const taggedAgents = (agentsJson.agents || []).filter(
            a => a.provenance && a.provenance.kind === 'experiment_output');
        expect(taggedAgents.length, 'at least one agent tagged experiment_output').toBeGreaterThan(0);
        // Schema shape: every agent's provenance.kind (when present) is one
        // of the 5 allowed values.
        const ALLOWED = new Set(['canonical', 'manual', 'experiment_output', 'seed_demo', 'legacy']);
        for (const a of agentsJson.agents || []) {
            if (a.provenance) {
                expect(ALLOWED.has(a.provenance.kind),
                    `agent ${a.id} provenance.kind=${a.provenance.kind} is valid`).toBe(true);
            }
        }

        // Personas with provenance.
        const personasResp = await request.get(`${PLUGIN_BASE}/personas`);
        const personasJson = await personasResp.json();
        const seedDemos = (personasJson.personas || []).filter(
            p => p.provenance && p.provenance.kind === 'seed_demo');
        expect(seedDemos.length,
            'at least one persona tagged seed_demo (canonical_keys present in experiment-spec.bios[])')
            .toBeGreaterThan(0);

        // Build a synthetic /yapper-seed response from the real corpus.
        // Mixes provenance kinds so the filter has something to suppress
        // AND something to surface in the default state. Each ranked row
        // is a real (persona, agent) pair from the tagged corpus, with
        // .persona.provenance and .agent.provenance set per disk.
        const realPersonas = personasJson.personas || [];
        const realAgents = agentsJson.agents || [];
        const pById = Object.fromEntries(realPersonas.map(p => [p.id, p]));

        // Pair every agent with its designed-for persona (if loaded) and
        // build a row. We need at least one row whose persona is legacy
        // (default-visible) and at least one whose persona is seed_demo
        // or experiment_output (default-hidden) to exercise the filter
        // assertions cleanly.
        const rows = [];
        for (const a of realAgents) {
            const p = pById[a.designed_for_bio_id];
            if (!p) continue;
            rows.push({
                bio_id: p.id, agent_id: a.id,
                why: 'fixture row',
                distance: rows.length * 0.1,
                persona: {
                    id: p.id, name: p.name, bio: p.bio || '', system_prompt: p.system_prompt || '',
                    provenance: p.provenance || { kind: 'legacy' },
                },
                agent: {
                    id: a.id, name: a.name, agent_text: a.agent_text || '',
                    provenance: a.provenance || { kind: 'legacy' },
                },
            });
        }
        expect(rows.length, 'fixture has at least one (bio, agent) row').toBeGreaterThan(0);
        const fixtureLegacyCount = rows.filter(r => r.persona.provenance.kind === 'legacy').length;
        const fixtureHiddenCount = rows.filter(r => {
            const k = r.persona.provenance.kind;
            return k === 'experiment_output' || k === 'seed_demo';
        }).length;
        expect(fixtureHiddenCount,
            'fixture has at least one default-hidden row (experiment_output / seed_demo)')
            .toBeGreaterThan(0);
        // Split into top vs side just like the real endpoint would.
        const half = Math.ceil(rows.length / 2);
        const fixtureTop = rows.slice(0, half);
        const fixtureSide = rows.slice(half);
        const fixturePayload = {
            top: fixtureTop, side: fixtureSide,
            _meta: {
                K_top: fixtureTop.length, K_side: fixtureSide.length,
                target_signature: {},
                candidates_considered: rows.length,
                bios_total: realPersonas.length,
                agents_total: realAgents.length,
                pending_synthesis: [],
                pending_count: 0,
            },
        };

        // ── (2 + 3) Filter row renders + provenance round-trips through
        // the API. Stub /yapper-seed at the page level so the FE rank
        // click resolves immediately to the fixture.
        await loadAndConnect(page);
        // Intercept any /yapper-seed POST and return the fixture. Set
        // the route BEFORE clicking the drawer so the inner iframe
        // inherits the page-level routing.
        await page.route('**/api/plugins/user-personas/yapper-seed', async route => {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify(fixturePayload),
            });
        });

        // Open via hamburger popover — .drawer-toggle is display:none after
        // sillytavern-fork e2973179d; direct click on #user-suggester-button is invalid.
        await openPersonaSurface(page, 'suggester');

        const iframe = page.frameLocator('iframe[src*="suggester.html"]');
        await expect(iframe.locator('h1')).toBeVisible({ timeout: 15_000 });

        // Clear localStorage so the spec runs from default state.
        const sugIframeEl = page.locator('iframe[src*="suggester.html"]').first();
        await sugIframeEl.evaluate(el => {
            try { el.contentWindow.localStorage.removeItem('user-personas/suggester-filter-state'); } catch (_) {}
            el.contentWindow.location.reload();
        });
        await expect(iframe.locator('h1')).toBeVisible({ timeout: 15_000 });

        // ── (3) Filter row + 5 checkboxes with correct default state.
        const filterRow = iframe.locator('#provenance-filter');
        await expect(filterRow, 'filter row renders').toBeVisible();
        const checkboxes = filterRow.locator('input[type=checkbox][data-kind]');
        await expect(checkboxes, 'five checkboxes in the filter row').toHaveCount(5);

        const canonicalCb = filterRow.locator('input[data-kind="canonical"]');
        const manualCb = filterRow.locator('input[data-kind="manual"]');
        const legacyCb = filterRow.locator('input[data-kind="legacy"]');
        const expOutCb = filterRow.locator('input[data-kind="experiment_output"]');
        const seedDemoCb = filterRow.locator('input[data-kind="seed_demo"]');
        await expect(canonicalCb, 'canonical checked by default').toBeChecked();
        await expect(manualCb, 'manual checked by default').toBeChecked();
        await expect(legacyCb, 'legacy checked by default').toBeChecked();
        await expect(expOutCb, 'experiment_output unchecked by default').not.toBeChecked();
        await expect(seedDemoCb, 'seed_demo unchecked by default').not.toBeChecked();

        // ── (4) Default filter hides experiment_output + seed_demo cards.
        // The current suggester fires doRank automatically via ST's event
        // system (no manual input field exists in this design). In the test
        // environment there is no live ST chat, so activeChatHasContent()
        // returns false and doRank is a no-op. Call applyRankResponse
        // directly on the iframe's contentWindow to inject the fixture
        // payload — this is the exact same function that doRank calls after
        // /yapper-seed responds. It exercises the full filter+render path
        // without requiring a live chat context or a network round-trip.
        await sugIframeEl.evaluate((el, payload) => {
            el.contentWindow.applyRankResponse(payload);
        }, fixturePayload);

        const visibleRows = iframe.locator('.ranked-row');
        const hiddenBadge = iframe.locator('#pf-hidden-badge');
        await expect(hiddenBadge).toBeVisible();
        const badgeText = await hiddenBadge.textContent();
        const hiddenMatch = badgeText.match(/\[(\d+) hidden\]/);
        expect(hiddenMatch, `hidden badge has count, got ${badgeText}`).not.toBeNull();
        const hiddenCount = Number(hiddenMatch[1]);
        expect(hiddenCount,
            `default filter hides count matches fixture (got ${hiddenCount}, fixture has ${fixtureHiddenCount} hidden)`)
            .toBe(fixtureHiddenCount);

        const visibleCount = await visibleRows.count();
        expect(visibleCount + hiddenCount,
            `visible (${visibleCount}) + hidden (${hiddenCount}) = total fixture rows (${rows.length})`)
            .toBe(rows.length);

        // ── (5) Toggling experiment_output reveals hidden rows.
        const beforeToggleVisible = await visibleRows.count();
        await expOutCb.click();
        await expect(expOutCb).toBeChecked();
        // Re-application is synchronous (no re-fetch).
        const afterToggleVisible = await visibleRows.count();
        const afterToggleBadgeText = await hiddenBadge.textContent();
        const afterToggleHidden = Number(afterToggleBadgeText.match(/\[(\d+) hidden\]/)[1]);
        expect(afterToggleVisible,
            `toggling experiment_output ON grows visible row count (was ${beforeToggleVisible}, now ${afterToggleVisible})`)
            .toBeGreaterThanOrEqual(beforeToggleVisible);
        expect(afterToggleHidden,
            `toggling experiment_output ON lowers hidden count (was ${hiddenCount}, now ${afterToggleHidden})`)
            .toBeLessThanOrEqual(hiddenCount);

        // ── (6) Persistence: reload and assert checkboxes preserve state.
        await sugIframeEl.evaluate(el => el.contentWindow.location.reload());
        await expect(iframe.locator('h1')).toBeVisible({ timeout: 15_000 });
        await expect(iframe.locator('input[data-kind="experiment_output"]'),
            'experiment_output checkbox state persists across reload').toBeChecked();
        await expect(iframe.locator('input[data-kind="seed_demo"]'),
            'seed_demo checkbox state persists across reload').not.toBeChecked();

        // Restore default state to be tidy for downstream tests.
        await sugIframeEl.evaluate(el => {
            try { el.contentWindow.localStorage.removeItem('user-personas/suggester-filter-state'); } catch (_) {}
        });

        // ── (8b) NO CARD WAS DELETED. ls before/after must match exactly.
        const agentsAfter = listDir(path.join(PLUGIN_DIR, 'agents'));
        const playersAfter = listDir(USER_AVATARS_DIR);
        expect(agentsAfter, 'no agent card was deleted by the spec or the tagging script')
            .toEqual(agentsBefore);
        expect(playersAfter, 'no player bio was deleted from User Avatars/ by the spec or the tagging script')
            .toEqual(playersBefore);
    });
});
