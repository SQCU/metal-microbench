// ============================================================================
// 98 — + More CEILING-DISABLE: PIXEL-SPACE acceptance (Phase-1 #3).
// ============================================================================
//
// A spec existing is necessary, NOT sufficient: this drives the LIVE st-debug
// SillyTavern (port 8002) in real Chromium and asserts the RENDERED
// #user_personas_augment_btn (the in-chat panel's "+ More" button) actually
// reaches [disabled] when the corpus is exhausted relative to the loadout, and
// re-enables after ↻ Re-suggest. It captures the disabled button as a PNG
// artifact (the rendered proof, not a source regex).
//
// The ceiling SIGNAL is the documented backend behavior: POST /yapper-seed
// returns a 200 with { top: [], side: [] } (never an error) when
// exclude_bio_ids covers every rankable bio. Here that signal is synthesized by
// route interception so the test is deterministic and bridge-free: the SECOND
// /yapper-seed (the + More call, which carries exclude_bio_ids) returns empty.
//
// COMMIT-AS-TEST-BOUNDARY: st-debug clones from the LOCAL root fork; the FE edit
// to index.js + suggester_core.js must be committed in root and pulled into the
// clone, then st-debug restarted, BEFORE this spec sees the disable behavior.
//
// HOW TO RUN:
//   cd tools/st-debug/tests
//   npx playwright test 98_more_ceiling_disable.spec.js

import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

const ST_URL = process.env.ST_URL || 'http://127.0.0.1:8002';
const PLUGIN_BASE = '/api/plugins/user-personas';
const BASIC_AUTH = { username: 'sussy', password: 'amongus' };
const ARTIFACT_DIR = path.join(process.cwd(), 'test-results', 'more_ceiling');

const AUGMENT_BTN = '#user_personas_augment_btn';
const RESUGGEST_BTN = '#user_personas_resuggest_btn';

test.use({ httpCredentials: BASIC_AUTH, trace: 'off', video: 'off' });

async function requireStOnly(request) {
    const st = await request.get(ST_URL).catch(() => null);
    if (!st || ![200, 401].includes(st.status())) {
        test.skip(true, `st-debug not reachable at ${ST_URL}; run tools/st-debug/scripts/run.sh first`);
    }
}

function rankRow(id, i) {
    return {
        bio_id: `${id}.png`,
        agent_id: `${id}-agent`,
        distance: 0.1 + i / 10,
        why: `ceiling fixture ${id}`,
        persona: { id: `${id}.png`, name: `Persona ${id}`, bio: `${id} bio`, provenance: { kind: 'matrix' } },
        agent: { id: `${id}-agent`, name: `Agent ${id}`, designed_for_bio_id: `${id}.png`, provenance: { kind: 'matrix' } },
    };
}

// /yapper-seed that EXHAUSTS on + More. The first render (no exclude_bio_ids)
// returns a 2-pick loadout; the + More call (carries the loadout bios as
// exclude_bio_ids) returns the documented empty ceiling { top: [], side: [] }.
async function installCeilingEndpoints(page) {
    let rankCalls = 0;
    await page.route(`**${PLUGIN_BASE}/yapper-seed`, async route => {
        rankCalls += 1;
        const body = route.request().postDataJSON() || {};
        const excluded = Array.isArray(body.exclude_bio_ids) ? body.exclude_bio_ids : [];
        const exhausted = excluded.length > 0; // the + More probe
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(exhausted
                ? { top: [], side: [], _meta: { K_top: 2, K_side: 2, candidates_considered: 0, bios_total: 2, agents_total: 2, pending_synthesis: [], pending_count: 0 } }
                : { top: [rankRow('alpha', 0), rankRow('beta', 1)], side: [], _meta: { K_top: 2, K_side: 0, candidates_considered: 2, bios_total: 2, agents_total: 2, pending_synthesis: [], pending_count: 0 } }),
        });
    });
    await page.route(`**${PLUGIN_BASE}/poll`, async route => {
        const body = route.request().postDataJSON();
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                applied_overlay: { source: 'agent', agent_id: `${body.persona_id}-agent`, name: `Overlay for ${body.persona_id}`, depth: 1, text_chars: 40 },
                candidates: [{ text: `Candidate for ${body.persona_id}`, truncated: false }],
            }),
        });
    });
    await page.route(`**${PLUGIN_BASE}/dispatch-missing-agent-synth`, route => route.fulfill({
        status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true, dispatched: 0, in_flight: 0 }),
    }));
    return { get rankCalls() { return rankCalls; } };
}

async function loadClientOnly(page) {
    await page.goto(ST_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction('document.getElementById("preloader") === null', { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function', { timeout: 30_000 });
}

async function openUserAgentPanel(page) {
    const button = page.locator('#user_personas_btn');
    await expect(button).toBeVisible({ timeout: 30_000 });
    await button.click();
    const panel = page.locator('#user_personas_panel');
    await expect(panel).toBeVisible({ timeout: 15_000 });
    return panel;
}

test.describe('+ More ceiling-disable (pixel-space, live st-debug)', () => {
    test.setTimeout(120_000);

    test.beforeEach(async ({ request }) => {
        await requireStOnly(request);
    });

    test('+ More reaches [disabled] at the corpus ceiling, re-enables on Re-suggest', async ({ page }) => {
        const endpoints = await installCeilingEndpoints(page);
        await loadClientOnly(page);
        const panel = await openUserAgentPanel(page);

        // Loadout renders → + More is actionable.
        await expect(panel.locator('.user-personas-card').first()).toBeVisible({ timeout: 15_000 });
        const augment = panel.locator(AUGMENT_BTN);
        await expect(augment, 'pre-ceiling: + More enabled').toBeEnabled({ timeout: 15_000 });
        const callsBeforeMore = endpoints.rankCalls;

        // Click + More. The interception returns the documented empty ceiling.
        await augment.click();
        await expect.poll(() => endpoints.rankCalls, { timeout: 10_000, intervals: [100, 250, 500] })
            .toBeGreaterThan(callsBeforeMore);

        // AC2 — the RENDERED button reaches [disabled] (the pixel-space proof).
        await expect(augment, 'AC2: + More disabled at corpus ceiling').toBeDisabled({ timeout: 10_000 });
        await expect(augment, 'AC2: rendered "all personas shown" affordance').toContainText(/all personas shown/i);
        await expect(augment, 'AC2: ceiling marker class').toHaveClass(/is-ceiling/);
        // Confirm it is genuinely non-clickable (the disabled attribute is on the DOM).
        expect(await augment.getAttribute('disabled'), 'AC2: disabled attribute present on the DOM node').not.toBeNull();

        // Capture the disabled button as a PNG artifact (rendered proof).
        fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
        await augment.screenshot({ path: path.join(ARTIFACT_DIR, 'augment_btn_disabled.png') }).catch(() => {});

        // AC3 — ↻ Re-suggest replaces the loadout → the ceiling resets.
        await panel.locator(RESUGGEST_BTN).click();
        await expect(augment, 'AC3: Re-suggest re-enables + More').toBeEnabled({ timeout: 10_000 });
        await expect(augment, 'AC3: label restored to + More').toContainText('+ More');
        await expect(augment, 'AC3: ceiling marker class removed').not.toHaveClass(/is-ceiling/);
        await augment.screenshot({ path: path.join(ARTIFACT_DIR, 'augment_btn_reenabled.png') }).catch(() => {});
    });
});
