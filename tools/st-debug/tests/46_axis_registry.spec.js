// B4 axis registry + lineage view — Playwright acceptance gate.
//
// Validates the axes.html drawer that the operator uses to inspect,
// add, edit, and delete the axis cards backing /signature-extract.
// Acceptance items 1–9 per docs/ui_spec_axis_registry.md
// §"Acceptance: Playwright spec".

import { test, expect } from '@playwright/test';
import { loadAndConnect } from './_helpers/elicit_clean.mjs';

const PLUGIN_BASE = '/api/plugins/user-personas';
const TEST_AXIS_ID = 'playwright_test_axis_46';

test.describe('B4 axis registry — desktop only', () => {
    test.setTimeout(2 * 60 * 1000);

    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'axis registry spec runs only on desktop project');
    });

    // Always clean up the test axis, even if the test failed mid-way.
    test.afterEach(async ({ request }) => {
        await request.delete(`${PLUGIN_BASE}/axes/${TEST_AXIS_ID}`).catch(() => {});
    });

    async function openAxesTab(page) {
        await loadAndConnect(page);
        // (1) Drawer button installs and is visible.
        await expect(page.locator('#user-axes-button')).toBeVisible({ timeout: 15_000 });
        await page.locator('#user-axes-button').click();
        // (2) Iframe loads with axes.html.
        const iframeEl = page.locator('iframe[src*="axes.html"]').first();
        await expect(iframeEl).toBeVisible({ timeout: 15_000 });
        const iframe = page.frameLocator('iframe[src*="axes.html"]');
        await expect(iframe.locator('h1').first()).toBeVisible({ timeout: 15_000 });
        // Status flips from "Loading…" to "N axes loaded".
        await expect(iframe.locator('#status')).toContainText(/loaded/i, { timeout: 15_000 });
        return iframe;
    }

    test('drawer + root render + add/edit/delete + orphan section', async ({ page, request }) => {
        // Pre-clean: in case a prior failed run left the test card around.
        await request.delete(`${PLUGIN_BASE}/axes/${TEST_AXIS_ID}`).catch(() => {});

        // Capture the registry state before to verify root rendering + lineage.
        const beforeResp = await request.get(`${PLUGIN_BASE}/axes`);
        expect(beforeResp.status()).toBe(200);
        const beforeJson = await beforeResp.json();
        const beforeAxes = beforeJson.axes;
        expect(beforeAxes.length).toBeGreaterThan(0);

        const iframe = await openAxesTab(page);

        // (3) Every axis from GET /axes is present in the page; roots
        //     (derived_from null) render at the top level (depth=0).
        for (const a of beforeAxes) {
            const card = iframe.locator(`.axis-card[data-axis-id="${a.id}"]`);
            await expect(card).toBeVisible();
            const expectedDepth = a.derived_from ? '1' : '0';
            await expect(card).toHaveAttribute('data-depth', expectedDepth);
        }

        // (4) Per-axis row: id (monospace), kind badge, def text, scored-on counts.
        //     Pick a stable existing axis we know about.
        const probe = beforeAxes.find(a => a.id === 'astrology_sagittarian') || beforeAxes[0];
        const probeCard = iframe.locator(`.axis-card[data-axis-id="${probe.id}"]`);
        await expect(probeCard.locator('.axis-id')).toHaveText(probe.id);
        await expect(probeCard.locator(`.kind-badge[data-kind="${probe.kind}"]`)).toBeVisible();
        await expect(probeCard.locator('[data-role="def"]')).toContainText(probe.def.slice(0, 12));
        await expect(probeCard.locator('[data-role="scored-on"]'))
            .toContainText(/scored on: \d+ bios, \d+ agents/);

        // (5) Add axis flow.
        //
        // P-EMPTY-FORM (UX-T1, 2026-05-21, spec 78): the bare "+ Add
        // axis" form was removed from axes.html because it was the
        // canonical JSON-fields-as-strings anti-pattern. New axes now
        // come from the `axis_splitter` CLI (corpus-driven). To keep
        // this spec exercising the underlying POST /axes endpoint
        // (which IS still supported — the registry API is intact), we
        // POST directly here instead of driving the deleted form. The
        // axis-card render assertion below still proves the surface
        // reflects the new card.
        await expect(iframe.locator('#add-axis-btn'),
            'P-EMPTY-FORM (spec 78): "+ Add axis" button must be absent'
        ).toHaveCount(0);
        await expect(iframe.locator('#add-form'),
            'P-EMPTY-FORM (spec 78): bare add-axis form must be absent'
        ).toHaveCount(0);
        const addResp = await request.post(`${PLUGIN_BASE}/axes/${TEST_AXIS_ID}`, {
            data: {
                name: 'PW test',
                def: '1: foo · 5: bar',
                kind: 'bio',
                scale_min: 1,
                scale_max: 5,
                derived_from: null,
            },
        });
        expect(addResp.status(), 'POST /axes/<id> creates the test axis').toBe(200);
        // Trigger a refresh so the new card renders.
        await iframe.locator('#refresh-btn').click();
        const newCard = iframe.locator(`.axis-card[data-axis-id="${TEST_AXIS_ID}"]`);
        await expect(newCard).toBeVisible({ timeout: 10_000 });
        // Server reflects it.
        const getResp = await request.get(`${PLUGIN_BASE}/axes/${TEST_AXIS_ID}`);
        expect(getResp.status()).toBe(200);
        const saved = await getResp.json();
        expect(saved.id).toBe(TEST_AXIS_ID);
        expect(saved.name).toBe('PW test');
        expect(saved.def).toBe('1: foo · 5: bar');
        expect(saved.kind).toBe('bio');
        expect(saved.scale_min).toBe(1);
        expect(saved.scale_max).toBe(5);
        expect(saved.derived_from).toBeNull();

        // (6) Edit flow — change the def.
        await newCard.locator('button[data-action="edit"]').click();
        const editForm = newCard.locator('[data-role="edit-form"]');
        await expect(editForm).toBeVisible();
        const defField = editForm.locator('textarea');
        await defField.fill('1: changed · 5: changed');
        await editForm.locator('button[data-role="edit-save"]').click();
        // After save, the form closes (slot cleared) and the new def renders.
        // We re-select the card since renderTree rebuilds it.
        await expect(iframe.locator(`.axis-card[data-axis-id="${TEST_AXIS_ID}"] [data-role="def"]`))
            .toContainText('1: changed · 5: changed', { timeout: 10_000 });
        const editedResp = await request.get(`${PLUGIN_BASE}/axes/${TEST_AXIS_ID}`);
        expect(editedResp.status()).toBe(200);
        const edited = await editedResp.json();
        expect(edited.def).toBe('1: changed · 5: changed');

        // (7) Delete flow with orphan warning.
        // Our test axis has 0 references, so the warning text shows the
        // "safe to delete" variant. Click delete → confirm.
        await iframe.locator(`.axis-card[data-axis-id="${TEST_AXIS_ID}"] button[data-action="delete"]`).click();
        await expect(iframe.locator('#confirm-dialog')).toBeVisible();
        await expect(iframe.locator('#confirm-title')).toContainText(TEST_AXIS_ID);
        // The warning area is visible for either case (orphans>0 vs safe-to-delete).
        await expect(iframe.locator('#confirm-warning')).toBeVisible();
        await iframe.locator('#confirm-delete').click();
        // Dialog closes; card vanishes.
        await expect(iframe.locator('#confirm-dialog')).toBeHidden({ timeout: 10_000 });
        await expect(iframe.locator(`.axis-card[data-axis-id="${TEST_AXIS_ID}"]`)).toHaveCount(0, { timeout: 10_000 });
        const after404 = await request.get(`${PLUGIN_BASE}/axes/${TEST_AXIS_ID}`);
        expect(after404.status()).toBe(404);

        // (8) Derived-axis lineage: if any axis has derived_from, assert it
        //     renders indented (data-depth="1") under its parent. If none,
        //     annotate as skipped — derived axes come from axis_splitter.mjs.
        const derivedAxes = beforeAxes.filter(a => a.derived_from);
        if (derivedAxes.length === 0) {
            test.info().annotations.push({
                type: 'skip',
                description: 'no derived axes in registry; lineage assertion path '
                    + 'not exercised (derived axes come from axis_splitter.mjs CLI runs)',
            });
        } else {
            for (const d of derivedAxes) {
                const dCard = iframe.locator(`.axis-card[data-axis-id="${d.id}"]`);
                await expect(dCard).toBeVisible();
                await expect(dCard).toHaveAttribute('data-depth', '1');
                await expect(dCard).toHaveClass(/derived/);
                // derived_from is an axis-v1 object { parent, sibling, … };
                // the .lineage-note renders the parent id string.
                const parentId = typeof d.derived_from === 'object'
                    ? d.derived_from.parent
                    : String(d.derived_from);
                await expect(dCard.locator('.lineage-note'))
                    .toContainText(`derived from ${parentId}`);
            }
        }

        // (9) Orphaned signatures section: the section renders. If the corpus
        //     has orphaned axis IDs (ids in signatures but not in the registry)
        //     they render as .orphan-row elements; if there are none the
        //     empty-state message appears. Both cases are valid.
        const orphansList = iframe.locator('#orphans-list');
        await expect(orphansList, 'orphans-list container renders').toBeVisible();
        const orphanRows = orphansList.locator('.orphan-row');
        const orphanEmpty = orphansList.locator('[data-role="orphans-empty"]');
        const rowCount = await orphanRows.count();
        if (rowCount > 0) {
            // Corpus has orphaned references; each row must have the
            // orphan-id span and orphan-meta span.
            await expect(orphanRows.first().locator('.orphan-id'),
                'orphan row has orphan-id span').toBeVisible();
            await expect(orphanRows.first().locator('.orphan-meta'),
                'orphan row has orphan-meta span').toBeVisible();
        } else {
            // No orphans: empty-state message must be present.
            await expect(orphanEmpty, 'empty-state renders when no orphans').toBeVisible();
            await expect(orphanEmpty).toContainText(/none/i);
        }
    });
});
