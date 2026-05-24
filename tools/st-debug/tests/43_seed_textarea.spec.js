// Verbatim-seed textarea — end-to-end spec.
//
// Validates the new Seed input tab inside fixed_point.html per
// docs/ui_spec_seed_textarea.md §"Acceptance: Playwright spec".
//
// The seed-input surface eliminates the pre-paraphrase requirement:
// operator types verbatim bio + motive seeds, the FE pairs them on
// operator-picked axes, materializes an experiment-spec card with
// `design_brief` / `motive_hint` set to the verbatim seed strings,
// and dispatches the same /experiments/:id/run path the structured
// editor uses.
//
// Smoke-validates dispatch — does NOT wait for the loop to converge.

import { test, expect } from '@playwright/test';
import { loadAndConnect } from './_helpers/elicit_clean.mjs';

const PLUGIN_BASE = '/api/plugins/user-personas';
const TEST_EXP_ID = 'playwright_seed_test';

const SEED_TEXT = `bios:
  rpg wizard but he a sagittarius
  rpg rogue but he a cancer

motives:
  will 100% steal all of your stuff
  will try to kiss you but also will 100% steal all of your stuff`;

test.describe('seed-input textarea — desktop only', () => {
    test.setTimeout(2 * 60 * 1000);

    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'seed-input spec runs only on desktop project');
    });

    test.afterEach(async ({ request }) => {
        // Always clean up — the seed-input surface materializes
        // experiment cards on disk; we must not leave them behind.
        await request.delete(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`).catch(() => {});
    });

    async function openSeedInputTab(page) {
        await loadAndConnect(page);
        await page.locator('#user-fixed-point-button').click();
        const iframe = page.frameLocator('iframe[src*="fixed_point.html"]');
        await expect(iframe.locator('h1').first()).toBeVisible({ timeout: 15_000 });
        // Wait for the experiments list to settle before switching tabs.
        await expect(iframe.locator('#experiments-status')).toContainText(/loaded/i, { timeout: 10_000 });
        await iframe.locator('#tab-seed').click();
        await expect(iframe.locator('#view-seed')).toBeVisible({ timeout: 5_000 });
        return iframe;
    }

    test('seed textarea parses verbatim seeds, materializes card with verbatim design_brief, dispatches /run', async ({ page, request }) => {
        // Pre-clean — guard against a prior failed run leaving the card.
        await request.delete(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`).catch(() => {});

        const iframe = await openSeedInputTab(page);

        // (1) Surface elements visible.
        await expect(iframe.locator('#seed-textarea')).toBeVisible();
        await expect(iframe.locator('#parse-seeds-btn')).toBeVisible();
        await expect(iframe.locator('#seed-bio-axes')).toBeVisible();
        await expect(iframe.locator('#seed-agent-axes')).toBeVisible();

        // (2) Type the lock_in_tetrad verbatim seeds, click Parse.
        await iframe.locator('#seed-textarea').fill(SEED_TEXT);
        await iframe.locator('#parse-seeds-btn').click();

        // 2 bio chips, with verbatim text.
        const bioChips = iframe.locator('#bio-chips .seed-chip[data-kind="bio"]');
        await expect(bioChips).toHaveCount(2);
        await expect(bioChips.nth(0)).toHaveText('rpg wizard but he a sagittarius');
        await expect(bioChips.nth(1)).toHaveText('rpg rogue but he a cancer');
        // 2 motive chips.
        const motiveChips = iframe.locator('#motive-chips .seed-chip[data-kind="motive"]');
        await expect(motiveChips).toHaveCount(2);
        await expect(motiveChips.nth(0)).toHaveText('will 100% steal all of your stuff');
        // Count summary — N × M compositions.
        await expect(iframe.locator('#seed-count-summary')).toContainText(/4 compositions/);

        // (3) Pick the axes.
        await iframe.locator('#seed-bio-axes').selectOption(['astrology_sagittarian', 'astrology_cancerian']);
        await iframe.locator('#seed-agent-axes').selectOption(['theft_aggressiveness', 'romantic_advance']);

        // Preview grid populates with auto-assigned targets.
        const previewBioCells = iframe.locator('#preview-bios .preview-cell');
        await expect(previewBioCells).toHaveCount(2);
        const previewMotiveCells = iframe.locator('#preview-motives .preview-cell');
        await expect(previewMotiveCells).toHaveCount(2);
        // Each preview cell renders a target_bio / target_agent line.
        await expect(previewBioCells.nth(0).locator('.target-sig')).toContainText(/ast_sag|ast_can/);

        // (4) Name the experiment and materialize-and-run.
        await iframe.locator('#seed-exp-id').fill(TEST_EXP_ID);
        await expect(iframe.locator('#materialize-btn')).toBeEnabled();
        await iframe.locator('#materialize-btn').click();

        // (5) Server-side: card exists, schema-valid, verbatim seeds preserved.
        // Poll because the POST happens client-side; give it time to complete.
        await expect.poll(async () => {
            const r = await request.get(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`);
            return r.status();
        }, { timeout: 15_000, intervals: [500, 1000, 2000] }).toBe(200);
        const getResp = await request.get(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`);
        const card = await getResp.json();

        expect(card.id).toBe(TEST_EXP_ID);
        expect(card.experiment_schema).toBe('experiment-v1');
        expect(card.bios).toHaveLength(2);
        expect(card.agent_targets).toHaveLength(2);

        // Verbatim seed preservation (no paraphrase) — the whole point.
        const designBriefs = card.bios.map(b => b.design_brief).sort();
        expect(designBriefs).toEqual([
            'rpg rogue but he a cancer',
            'rpg wizard but he a sagittarius',
        ]);
        const motiveHints = card.agent_targets.map(t => t.motive_hint).sort();
        expect(motiveHints).toEqual([
            'will 100% steal all of your stuff',
            'will try to kiss you but also will 100% steal all of your stuff',
        ]);

        // Bio targets: keys are the picked axes, values are 1 or 5 only.
        for (const bio of card.bios) {
            const keys = Object.keys(bio.target_bio).sort();
            expect(keys).toEqual(['astrology_cancerian', 'astrology_sagittarian']);
            for (const v of Object.values(bio.target_bio)) {
                expect([1, 5]).toContain(v);
            }
        }
        // And the two bios occupy opposite corners (differ on every axis).
        const b0 = card.bios[0].target_bio;
        const b1 = card.bios[1].target_bio;
        for (const ax of ['astrology_sagittarian', 'astrology_cancerian']) {
            expect(b0[ax]).not.toBe(b1[ax]);
        }

        // Agent targets: same shape — keys are the picked agent axes,
        // values in {1,5}, the two motives occupy opposite corners.
        for (const t of card.agent_targets) {
            const keys = Object.keys(t.target_agent).sort();
            expect(keys).toEqual(['romantic_advance', 'theft_aggressiveness']);
            for (const v of Object.values(t.target_agent)) {
                expect([1, 5]).toContain(v);
            }
        }
        const a0 = card.agent_targets[0].target_agent;
        const a1 = card.agent_targets[1].target_agent;
        for (const ax of ['theft_aggressiveness', 'romantic_advance']) {
            expect(a0[ax]).not.toBe(a1[ax]);
        }

        // counterparty_avatar defaulted to the-rock.png.
        expect(card.counterparty_avatar).toBe('the-rock.png');

        // (6) Run progress UI activates. Materialize switches to the
        // Experiments tab and shows the running banner with the run_id.
        await expect(iframe.locator('#tab-experiments')).toHaveClass(/active/, { timeout: 5_000 });
        await expect(iframe.locator('#view-experiments')).toBeVisible();
        await expect(iframe.locator('#run-banner')).toBeVisible();
        // run_id should appear in the banner and reference the experiment id.
        await expect(iframe.locator('#run-banner')).toContainText(new RegExp(TEST_EXP_ID));
    });

    test('warning path: too many bios for picked axes disables materialize', async ({ page, request }) => {
        await request.delete(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`).catch(() => {});
        const iframe = await openSeedInputTab(page);

        // 5 bios, 2 bio_axes ⇒ N=5 > 2^K_b=4 — should warn + disable.
        const fiveBios = `bios:
  alpha wizard
  beta wizard
  gamma wizard
  delta wizard
  epsilon wizard

motives:
  steals stuff
  romances you`;
        await iframe.locator('#seed-textarea').fill(fiveBios);
        await iframe.locator('#parse-seeds-btn').click();
        await expect(iframe.locator('#bio-chips .seed-chip[data-kind="bio"]')).toHaveCount(5);

        await iframe.locator('#seed-bio-axes').selectOption(['astrology_sagittarian', 'astrology_cancerian']);
        await iframe.locator('#seed-agent-axes').selectOption(['theft_aggressiveness', 'romantic_advance']);

        // Warning rendered, materialize disabled.
        await expect(iframe.locator('#preview-warning')).toContainText(/5 bios.*K_b=2.*4 corners/);
        await iframe.locator('#seed-exp-id').fill(TEST_EXP_ID);
        await expect(iframe.locator('#materialize-btn')).toBeDisabled();

        // Adding a third bio_axis (8 corners ≥ 5 bios) clears the warning + re-enables.
        // `performative` is bio-kind; available as a 3rd bio axis.
        await iframe.locator('#seed-bio-axes').selectOption([
            'astrology_sagittarian', 'astrology_cancerian', 'performative',
        ]);
        await expect(iframe.locator('#preview-warning')).not.toContainText(/K_b=2/);
        await expect(iframe.locator('#materialize-btn')).toBeEnabled();
    });

    test('parser without headers: surfaces a "need section headers" warning', async ({ page }) => {
        const iframe = await openSeedInputTab(page);

        await iframe.locator('#seed-textarea').fill('rpg wizard but he a sagittarius\nrpg rogue but he a cancer\nwill 100% steal all of your stuff');
        await iframe.locator('#parse-seeds-btn').click();

        // No headers → all lines fall into warnings; bios + motives = 0.
        await expect(iframe.locator('#seed-warnings')).toBeVisible();
        await expect(iframe.locator('#seed-warnings')).toContainText(/before any "bios:" or "motives:" header/);
        // Zero chips → no compositions possible.
        await expect(iframe.locator('#bio-chips .seed-chip[data-kind="bio"]')).toHaveCount(0);
        await expect(iframe.locator('#motive-chips .seed-chip[data-kind="motive"]')).toHaveCount(0);
    });
});
