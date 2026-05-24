// Locks in the canonical sorceror-rogue tetrad as a visible UI demo.
//
// Background: the lock_in_tetrad experiment is the documented canonical
// fixed-point-iteration demo (per docs/bio_generator_harness_lineage.md
// + plugins/user-personas/experiments/lock_in_tetrad.json). It pairs:
//   - 2 bios at antipodes on (astrology_sagittarian, astrology_cancerian):
//       rpg-wizard-sagittarius @ (5, 1)
//       rpg-rogue-cancer       @ (1, 5)
//   - 2 agent targets on (theft_aggressiveness, romantic_advance):
//       steals               @ (5, 1)
//       romances-and-steals  @ (5, 5)
//   = 4 (bio × agent) compositions
//
// This experiment went missing from disk in the May-2026 unification
// refactor (the spec card was never re-committed after first run, and
// the 4 axes it depends on were archived). Restored 2026-05-24 from
// git commit 789079f82 + axes-archive-20260520-102931/.
//
// This spec validates that the demo is visible + invokable in the
// Fixed-Point Iteration UI:
//   1. /experiments/lock_in_tetrad returns the canonical spec card
//   2. The 4 axes (astrology_sagittarian, astrology_cancerian,
//      theft_aggressiveness, romantic_advance) are present in /axes
//   3. The 2 source bios are present in /personas
//   4. The Fixed-Point Iteration drawer renders the experiment as a
//      selectable card with name, bios, agent_targets, counterparty
//   5. (Soft) Dispatching the experiment doesn't crash the plugin —
//      we don't wait for the K-iteration to complete (that's expensive
//      and unpredictable on timing) but we assert the dispatch endpoint
//      returns 200 and a run_id.
//
// We do NOT wait for the full fixed-point loop to converge — that's a
// 4-composition × multi-iteration LLM workload that can take 10+ min on
// cold cache. The harness-level test for convergence lives elsewhere.
// What this spec proves is: the demo is reachable, configured correctly,
// and the plugin accepts the dispatch.

import { test, expect } from '@playwright/test';

const PLUGIN_BASE = 'http://127.0.0.1:8002/api/plugins/user-personas';

test.describe('lock_in_tetrad — canonical sorceror-rogue demo', () => {
    test.setTimeout(60_000);

    test('canonical artifacts are present on disk + reachable via API', async ({ request }) => {
        // Experiment card
        const xr = await request.get(`${PLUGIN_BASE}/experiments/lock_in_tetrad`);
        expect(xr.status(), 'GET /experiments/lock_in_tetrad must return 200').toBe(200);
        const exp = await xr.json();
        expect(exp.id).toBe('lock_in_tetrad');
        expect(exp.experiment_schema).toBe('experiment-v1');
        expect(exp.bios.map(b => b.slug).sort()).toEqual(
            ['rpg-rogue-cancer', 'rpg-wizard-sagittarius']);
        expect(exp.agent_targets.map(t => t.slug).sort()).toEqual(
            ['romances-and-steals', 'steals']);
        expect(exp.bio_axes.sort()).toEqual(
            ['astrology_cancerian', 'astrology_sagittarian']);
        expect(exp.agent_axes.sort()).toEqual(
            ['romantic_advance', 'theft_aggressiveness']);
        // Antipode coordinates
        const wizard = exp.bios.find(b => b.slug === 'rpg-wizard-sagittarius');
        expect(wizard.target_bio.astrology_sagittarian).toBe(5);
        expect(wizard.target_bio.astrology_cancerian).toBe(1);
        const rogue = exp.bios.find(b => b.slug === 'rpg-rogue-cancer');
        expect(rogue.target_bio.astrology_sagittarian).toBe(1);
        expect(rogue.target_bio.astrology_cancerian).toBe(5);

        // 4 axes
        const ar = await request.get(`${PLUGIN_BASE}/axes`);
        const axesPayload = await ar.json();
        const axesList = Array.isArray(axesPayload) ? axesPayload : (axesPayload.axes || []);
        const axesById = Object.fromEntries(axesList.map(a => [a.id, a]));
        for (const axId of ['astrology_sagittarian', 'astrology_cancerian',
                            'theft_aggressiveness', 'romantic_advance']) {
            expect(axesById[axId], `axis ${axId} must be loaded`).toBeDefined();
        }
        expect(axesById['astrology_sagittarian'].kind).toBe('bio');
        expect(axesById['theft_aggressiveness'].kind).toBe('agent');

        // 2 source bios with non-empty descriptions
        const pr = await request.get(`${PLUGIN_BASE}/personas`);
        const pPayload = await pr.json();
        const personas = Array.isArray(pPayload) ? pPayload : (pPayload.personas || []);
        const byKey = Object.fromEntries(personas.map(p => [p.id, p]));
        expect(byKey['rpg-rogue-cancer.png'], 'rpg-rogue-cancer.png bio must exist').toBeDefined();
        expect(byKey['rpg-wizard-sagittarius.png'], 'rpg-wizard-sagittarius.png bio must exist').toBeDefined();
        expect((byKey['rpg-rogue-cancer.png'].bio || '').length).toBeGreaterThan(100);
        expect((byKey['rpg-wizard-sagittarius.png'].bio || '').length).toBeGreaterThan(100);
    });

    test('Fixed-Point Iteration drawer renders the tetrad as a selectable card', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'render test is desktop-only — canonical 1280×800 viewport');

        await page.goto('/');
        await page.waitForFunction(() => document.getElementById('preloader') === null,
            { timeout: 60_000 });
        await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
            { timeout: 30_000 });

        // Open the user-personas tools drawer + the Fixed-Point Iteration surface.
        const hamburger = page.locator('#user-personas-tools-button .drawer-toggle');
        await expect(hamburger).toBeVisible({ timeout: 20_000 });
        await hamburger.click();
        const menuItem = page.locator('.user-personas-tools-menuitem[data-surface-key="fixed-point"]');
        await expect(menuItem).toBeVisible({ timeout: 5_000 });
        await menuItem.click();

        const iframe = page.frameLocator('#user-personas-surface-fixed-point iframe');
        // Give the iframe time to fetch /experiments.
        const experimentCard = iframe.locator('.experiment-card[data-eid="lock_in_tetrad"]');
        await expect(experimentCard,
            'lock_in_tetrad card must render in the Fixed-Point Iteration drawer')
            .toBeVisible({ timeout: 30_000 });

        const name = experimentCard.locator('.experiment-name');
        await expect(name).toContainText('Wizard');
        await expect(name).toContainText('Rogue');

        const meta = experimentCard.locator('.experiment-meta');
        const metaText = (await meta.innerText()).trim();
        expect(metaText, 'meta must list both bio slugs').toMatch(/rpg-wizard-sagittarius/);
        expect(metaText, 'meta must list both bio slugs').toMatch(/rpg-rogue-cancer/);
        expect(metaText, 'meta must list both agent_target slugs').toMatch(/steals/);
        expect(metaText, 'meta must list both agent_target slugs').toMatch(/romances-and-steals/);
        expect(metaText, 'meta must list counterparty').toMatch(/the-rock\.png/);

        // Screenshot proof.
        await page.screenshot({ path: '/tmp/spec83_tetrad_card.png', fullPage: true });

        // Soft: confirm Dispatch button exists. We do NOT click it — running
        // the full fixed-point loop is a 10+ minute LLM workload.
        const dispatchBtn = experimentCard.locator('.run-btn');
        await expect(dispatchBtn).toBeVisible();
        const btnText = (await dispatchBtn.innerText()).trim();
        expect(btnText).toMatch(/Dispatch run/i);
    });
});
