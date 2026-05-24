// D10 iteration-timeline view — Playwright acceptance gate.
//
// Validates the Trajectory subsection inside fixed_point.html that
// renders the per-bio trace from lock_in_iterative.mjs's on-disk
// output JSONs. Acceptance items 1–9 per
// docs/ui_spec_iteration_timeline.md §"Acceptance: Playwright spec".
//
// Uses the pre-existing lock_in_tetrad result files on disk
// (left over from prior session runs); if those aren't present, the
// spec skips with an annotation explaining the seeding requirement.

import fs from 'node:fs';
import path from 'node:path';
import { test, expect } from '@playwright/test';
import { loadAndConnect } from './_helpers/elicit_clean.mjs';

const PLUGIN_BASE = '/api/plugins/user-personas';
const EXPERIMENT_ID = 'lock_in_tetrad';
const RESULTS_DIR = '/Users/mdot/metal-microbench/data/lock_in_iterative/lock_in_tetrad';

function listSeededBioSlugs() {
    if (!fs.existsSync(RESULTS_DIR)) return [];
    return fs.readdirSync(RESULTS_DIR)
        .filter(f => f.endsWith('.json'))
        .map(f => f.replace(/\.json$/, ''))
        .sort();
}

test.describe('D10 iteration timeline — desktop only', () => {
    test.setTimeout(2 * 60 * 1000);

    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'iteration-timeline spec runs only on desktop project');
        const seeded = listSeededBioSlugs();
        test.skip(seeded.length < 1,
            `no lock_in_tetrad results on disk at ${RESULTS_DIR} — `
            + `run the lock_in_iterative harness first to seed`);
    });

    async function openFixedPointTab(page) {
        await loadAndConnect(page);
        await page.locator('#user-fixed-point-button').click();
        const iframe = page.frameLocator('iframe[src*="fixed_point.html"]');
        await expect(iframe.locator('h1').first()).toBeVisible({ timeout: 15_000 });
        await expect(iframe.locator('#experiments-status')).toContainText(/loaded/i, { timeout: 10_000 });
        return iframe;
    }

    test('endpoint contract + FE Trajectory view renders the full iteration trace', async ({ page, request }) => {
        // ── (1) Endpoint contract: list endpoint returns both bio_slugs ──
        const listResp = await request.get(
            `${PLUGIN_BASE}/experiments/${EXPERIMENT_ID}/results`);
        expect(listResp.status()).toBe(200);
        const listJson = await listResp.json();
        expect(listJson.experiment_id).toBe(EXPERIMENT_ID);
        expect(Array.isArray(listJson.results)).toBe(true);
        const slugs = listJson.results.map(r => r.bio_slug).sort();
        const seeded = listSeededBioSlugs();
        expect(slugs).toEqual(seeded);
        // The fixture has at least the wizard bio.
        expect(slugs).toContain('rpg-wizard-sagittarius');
        for (const r of listJson.results) {
            expect(r.size_bytes).toBeGreaterThan(0);
            expect(r.mtime).toMatch(/^\d{4}-\d{2}-\d{2}T/);
        }

        // ── (2) Per-bio endpoint returns expected top-level keys ─────────
        const wizResp = await request.get(
            `${PLUGIN_BASE}/experiments/${EXPERIMENT_ID}/results/rpg-wizard-sagittarius`);
        expect(wizResp.status()).toBe(200);
        const wizJson = await wizResp.json();
        expect(Object.keys(wizJson)).toEqual(
            expect.arrayContaining(['bio', 'agent_targets', 'result', 'elapsed_ms_total']));
        expect(wizJson.bio.slug).toBe('rpg-wizard-sagittarius');
        expect(wizJson.result.attempts).toBeInstanceOf(Array);
        expect(wizJson.result.attempts.length).toBeGreaterThanOrEqual(1);

        // ── (3) 404 path: nonexistent bio ────────────────────────────────
        const notFound = await request.get(
            `${PLUGIN_BASE}/experiments/${EXPERIMENT_ID}/results/nonexistent-bio`);
        expect(notFound.status()).toBe(404);
        // Body parses as JSON with an error key (not a 500 HTML page).
        const notFoundJson = await notFound.json();
        expect(notFoundJson.error).toMatch(/not found/i);

        // ── (4) FE: clicking the lock_in_tetrad row opens Trajectory ─────
        const iframe = await openFixedPointTab(page);
        await expect(iframe.locator(`.experiment-card[data-eid="${EXPERIMENT_ID}"]`))
            .toBeVisible({ timeout: 10_000 });

        // Click the row body (away from buttons / edit-trigger title).
        // The card has cursor: pointer; clicking the description text
        // dispatches the row handler without triggering the edit-trigger.
        await iframe.locator(`.experiment-card[data-eid="${EXPERIMENT_ID}"] .experiment-desc`).first().click();
        // Trajectory view becomes visible; experiments view hides.
        await expect(iframe.locator('#view-trajectory')).toBeVisible({ timeout: 10_000 });
        await expect(iframe.locator('#view-experiments')).toBeHidden();
        // Breadcrumb shows the experiment id.
        await expect(iframe.locator('#trajectory-exp-id')).toContainText(EXPERIMENT_ID);
        // Both (all seeded) bio buttons rendered.
        const bioButtons = iframe.locator('#trajectory-bio-buttons .traj-bio-btn');
        await expect(bioButtons).toHaveCount(seeded.length);
        for (const slug of seeded) {
            await expect(iframe.locator(`#trajectory-bio-buttons .traj-bio-btn[data-bio-slug="${slug}"]`))
                .toBeVisible();
        }

        // ── (5) Bio header: target_bio pills, stop_reason badge, elapsed ──
        // The first bio is auto-loaded on open. To make assertions
        // deterministic against the fixture, explicitly click the wizard.
        await iframe.locator('.traj-bio-btn[data-bio-slug="rpg-wizard-sagittarius"]').click();
        const wizPane = iframe.locator('.traj-bio-pane[data-bio-pane="rpg-wizard-sagittarius"]');
        await expect(wizPane).toBeVisible();
        // target_bio pills are present + match the JSON's target_bio entries.
        const targetEntries = Object.entries(wizJson.bio.target_bio || {});
        expect(targetEntries.length).toBeGreaterThanOrEqual(1);
        for (const [axis, _val] of targetEntries) {
            await expect(wizPane.locator(`.target-pill[data-axis="${axis}"]`).first()).toBeVisible();
        }
        // stop_reason badge present with the exact text from the result.
        const stopReason = wizJson.result.stop_reason || 'unknown';
        const stopBadge = wizPane.locator(`.stop-badge[data-stop-reason="${stopReason}"]`).first();
        await expect(stopBadge).toBeVisible();
        await expect(stopBadge).toContainText(stopReason);
        // Elapsed time pill present.
        await expect(wizPane.locator('.elapsed-pill[data-elapsed-ms]').first()).toBeVisible();

        // ── (6) Outer accordion: at least one renders, with bio prose +
        //        measured signature pills + max_off shown.
        const outerAccordions = wizPane.locator('details.traj-outer');
        const accordionCount = await outerAccordions.count();
        expect(accordionCount).toBeGreaterThanOrEqual(1);
        // Best-iter accordion is open by default; assert at least one
        // [open] accordion exists and its bio_prose is visible.
        const openAccordion = wizPane.locator('details.traj-outer[open]').first();
        await expect(openAccordion).toBeVisible();
        await expect(openAccordion.locator('pre.bio-prose')).toBeVisible();
        const proseText = (await openAccordion.locator('pre.bio-prose').textContent()) || '';
        expect(proseText.length).toBeGreaterThan(20);
        // measured signature pills present.
        await expect(openAccordion.locator('.measured-line .axis-pill').first()).toBeVisible();
        // max_off visible in the summary.
        await expect(openAccordion.locator('summary .max-off')).toBeVisible();

        // ── (7) Inner blocks per agent_target with at least one attempt row.
        const innerBlocks = openAccordion.locator('.traj-inner-block');
        const innerCount = await innerBlocks.count();
        expect(innerCount).toBeGreaterThanOrEqual(1);
        // Pick the first inner block; assert ≥1 attempt row.
        const firstInner = innerBlocks.first();
        await expect(firstInner.locator('.inner-header .slug')).toBeVisible();
        const attemptRows = firstInner.locator('.traj-attempt-row');
        expect(await attemptRows.count()).toBeGreaterThanOrEqual(1);
        // Attempt rows must show converged ✓ / missed ✗.
        const firstAttempt = attemptRows.first();
        const firstStatus = await firstAttempt.locator('.att-status').textContent();
        expect(firstStatus).toMatch(/✓|✗/);

        // ── (8) Chat preview: clicking [Show chat turns] reveals at least one bubble.
        const chatToggle = firstAttempt.locator('button[data-toggle="chat"]');
        await expect(chatToggle).toBeVisible();
        // Hidden before clicking.
        await expect(firstAttempt.locator('.traj-chat-wrap')).toBeHidden();
        await chatToggle.click();
        await expect(firstAttempt.locator('.traj-chat-wrap')).toBeVisible();
        const bubbles = firstAttempt.locator('.traj-chat-wrap .chat-msg');
        expect(await bubbles.count()).toBeGreaterThanOrEqual(1);
        // The toggle's label flipped.
        await expect(chatToggle).toContainText(/Hide chat turns/i);

        // ── (9) Back button returns to the Experiments list.
        await iframe.locator('#trajectory-back-btn').click();
        await expect(iframe.locator('#view-trajectory')).toBeHidden({ timeout: 5_000 });
        await expect(iframe.locator('#view-experiments')).toBeVisible();
        // Experiment row is still rendered after returning.
        await expect(iframe.locator(`.experiment-card[data-eid="${EXPERIMENT_ID}"]`))
            .toBeVisible();
    });
});
