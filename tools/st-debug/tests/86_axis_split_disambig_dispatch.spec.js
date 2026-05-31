// Spec 86 — Axis splitter + cluster disambiguator UI dispatch
//
// Sub-30 deliverable. Source spec: docs/feature_factorization_design.md
// (axis splitter §5 + cluster disambiguator §6).
//
// What this spec validates:
//
//   A. Plugin endpoints exist:
//      - POST /axes/:id/split          → 200 {run_id} (or 422 no-traj)
//      - GET  /axes/:id/split/runs     → 200 {runs:[...]}
//      - POST /clusters/:id/disambiguate → 200 {run_id}
//      - GET  /clusters/:id/disambiguate/runs → 200 {runs:[...]}
//      - GET  /clusters               → 200 {clusters:[...]}
//      - GET  /experiments/runs/:id   → 200 {run, log, results}
//        (status/log-tail already exists from experiment dispatch)
//
//   B. corpus.html UI:
//      - Each axis card in the registry has a "Split" button
//        (button[data-action="split"])
//      - Clicking "Split" on rpg_class opens an inline split panel
//        (.split-panel[data-role="split-panel"]) with a
//        "Dispatch split" button
//      - The panel has a visible status element and a log element
//
//   C. Cluster disambiguator surface:
//      - "Cluster disambiguator" section heading is visible
//      - Cluster cards render for each spec in data/clusters/
//      - Each cluster card has a "Disambiguate" button
//        (button[data-action="disambig"])
//      - Clicking "Disambiguate" opens an inline disambig panel
//        (.disambig-panel) with "Dispatch disambiguator" button
//
//   D. POST /axes/:id/split on rpg_class returns either:
//      - 200 { run_id } if a trajectory file referencing rpg_class exists
//      - 422 { error, hint } if no trajectory found (legitimate empty state)
//      Either response is acceptable; the spec validates the INTERFACE
//      contract, not the harness outcome.
//
//   E. POST /clusters/sag_substantive/disambiguate returns 200 { run_id }.
//      (sag_substantive.json exists in data/clusters/)
//
// SPEC FAILS ON CURRENT MAIN: these endpoints and UI affordances do not
// exist before this patch lands. The split/disambig routes in index.mjs
// and the corpus.html button + panel wiring are added by Sub-30.

import { test, expect } from '@playwright/test';

const PLUGIN_BASE = 'http://127.0.0.1:8002/api/plugins/user-personas';
const CORPUS_URL = 'http://127.0.0.1:8002/api/plugins/user-personas/static/corpus.html';

// ── A. Plugin endpoint existence ──────────────────────────────────────
test.describe('A — Plugin endpoints exist (Sub-30)', () => {
    test('GET /clusters returns {clusters:[...]}', async ({ request }) => {
        const resp = await request.get(`${PLUGIN_BASE}/clusters`);
        expect(resp.ok(), `GET /clusters returned ${resp.status()}`).toBe(true);
        const body = await resp.json();
        expect(body).toHaveProperty('clusters');
        expect(Array.isArray(body.clusters)).toBe(true);
    });

    test('GET /axes/:id/split/runs returns {runs:[...]}', async ({ request }) => {
        const resp = await request.get(`${PLUGIN_BASE}/axes/rpg_class/split/runs`);
        expect(resp.ok(), `GET /axes/rpg_class/split/runs returned ${resp.status()}`).toBe(true);
        const body = await resp.json();
        expect(body).toHaveProperty('runs');
        expect(Array.isArray(body.runs)).toBe(true);
    });

    test('GET /clusters/:id/disambiguate/runs returns {runs:[...]}', async ({ request }) => {
        const resp = await request.get(
            `${PLUGIN_BASE}/clusters/sag_substantive/disambiguate/runs`);
        // 200 with empty runs list OR 200 with existing runs. Either is correct.
        expect(resp.ok(), `GET /clusters/sag_substantive/disambiguate/runs returned ${resp.status()}`).toBe(true);
        const body = await resp.json();
        expect(body).toHaveProperty('runs');
        expect(Array.isArray(body.runs)).toBe(true);
    });

    test('POST /axes/:id/split returns run_id or 422 with hint (no trajectory)', async ({ request }) => {
        const resp = await request.post(`${PLUGIN_BASE}/axes/rpg_class/split`, {
            data: {},
        });
        // Two acceptable states:
        //   200 { ok, run_id } — a trajectory referencing rpg_class was found
        //   422 { error, hint } — no trajectory yet (legitimate before any run)
        expect([200, 422], `expected 200 or 422, got ${resp.status()}`).toContain(resp.status());
        const body = await resp.json();
        if (resp.status() === 200) {
            expect(body).toHaveProperty('run_id');
            expect(typeof body.run_id).toBe('string');
            expect(body.run_id.length).toBeGreaterThan(0);
        } else {
            // 422 must carry an error message and a hint for the operator.
            expect(body).toHaveProperty('error');
            expect(body).toHaveProperty('hint');
        }
    });

    test('POST /clusters/:id/disambiguate returns run_id', async ({ request }) => {
        const resp = await request.post(
            `${PLUGIN_BASE}/clusters/sag_substantive/disambiguate`, {
                data: {},
            });
        expect(resp.ok(), `POST /clusters/sag_substantive/disambiguate returned ${resp.status()}`).toBe(true);
        const body = await resp.json();
        expect(body).toHaveProperty('run_id');
        expect(typeof body.run_id).toBe('string');
        expect(body.run_id.length).toBeGreaterThan(0);
        // run_id must start with 'disambig-'
        expect(body.run_id).toMatch(/^disambig-/);

        // Verify the run is retrievable via GET /experiments/runs/:run_id
        const statusResp = await request.get(
            `${PLUGIN_BASE}/experiments/runs/${body.run_id}`);
        expect(statusResp.ok(),
            `GET /experiments/runs/${body.run_id} returned ${statusResp.status()}`).toBe(true);
        const statusBody = await statusResp.json();
        expect(statusBody).toHaveProperty('run');
        expect(['running', 'done', 'failed']).toContain(statusBody.run.status);
    });

    test('POST /clusters/nonexistent/disambiguate returns 404', async ({ request }) => {
        const resp = await request.post(
            `${PLUGIN_BASE}/clusters/this-cluster-does-not-exist/disambiguate`, {
                data: {},
            });
        expect(resp.status()).toBe(404);
        const body = await resp.json();
        expect(body).toHaveProperty('error');
    });
});

// ── B. corpus.html Split button UI ────────────────────────────────────
test.describe('B — Split button present on axis cards (Sub-30)', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(CORPUS_URL);
        // Wait for the axes registry to finish loading.
        await expect(page.locator('#axes-status')).not.toContainText('Loading', { timeout: 15_000 });
    });

    test('Each axis card has a Split button', async ({ page }) => {
        const axisCards = page.locator('.axis-card');
        const count = await axisCards.count();
        expect(count, 'at least 1 axis card must be present').toBeGreaterThan(0);

        // Every axis card must have exactly one Split button.
        for (let i = 0; i < count; i++) {
            const card = axisCards.nth(i);
            const splitBtn = card.locator('button[data-action="split"]');
            await expect(splitBtn,
                `axis card ${i} must have a Split button`).toHaveCount(1);
            await expect(splitBtn).toBeVisible();
            await expect(splitBtn).toContainText(/split/i);
        }
    });

    test('Clicking Split on rpg_class opens inline split panel', async ({ page }) => {
        const rpgCard = page.locator('.axis-card[data-axis-id="rpg_class"]');
        await expect(rpgCard).toBeVisible({ timeout: 10_000 });

        // Panel must not exist yet.
        await expect(rpgCard.locator('[data-role="split-panel"]')).toHaveCount(0);

        // Click the Split button.
        await rpgCard.locator('button[data-action="split"]').click();

        // Split panel must appear.
        const panel = rpgCard.locator('[data-role="split-panel"]');
        await expect(panel).toBeVisible({ timeout: 5_000 });

        // Panel must contain a status line and a "Dispatch split" button.
        await expect(panel.locator('[data-role="split-status"]')).toBeVisible();
        await expect(panel.locator('button[data-action="split-dispatch"]')).toBeVisible();
        await expect(panel.locator('button[data-action="split-dispatch"]'))
            .toContainText(/dispatch split/i);
    });

    test('Clicking Split again closes the panel (toggle)', async ({ page }) => {
        const rpgCard = page.locator('.axis-card[data-axis-id="rpg_class"]');
        await expect(rpgCard).toBeVisible({ timeout: 10_000 });

        // Open.
        await rpgCard.locator('button[data-action="split"]').click();
        await expect(rpgCard.locator('[data-role="split-panel"]')).toBeVisible({ timeout: 3_000 });

        // Close via second click on the Split button.
        await rpgCard.locator('button[data-action="split"]').click();
        await expect(rpgCard.locator('[data-role="split-panel"]')).toHaveCount(0);
    });

    test('"Dispatch split" click fires POST /axes/:id/split', async ({ page }) => {
        const rpgCard = page.locator('.axis-card[data-axis-id="rpg_class"]');
        await expect(rpgCard).toBeVisible({ timeout: 10_000 });
        await rpgCard.locator('button[data-action="split"]').click();
        const panel = rpgCard.locator('[data-role="split-panel"]');
        await expect(panel).toBeVisible({ timeout: 3_000 });

        // Intercept the POST.
        let splitPostFired = false;
        page.on('request', (req) => {
            if (/\/axes\/[^/]+\/split$/.test(req.url()) && req.method() === 'POST') {
                splitPostFired = true;
            }
        });

        await panel.locator('button[data-action="split-dispatch"]').click();

        // Status should change from idle to either 'Dispatching…' or
        // show a run_id (if dispatch happened synchronously).
        await expect.poll(() => splitPostFired, {
            message: 'a POST /axes/<id>/split request must fire when Dispatch split is clicked',
            timeout: 8_000,
        }).toBe(true);

        // Status element should change (no longer the initial idle text).
        const statusEl = panel.locator('[data-role="split-status"]');
        await expect(statusEl).not.toContainText(
            'Dispatch axis_splitter against', { timeout: 8_000 });
    });
});

// ── C. corpus.html Cluster disambiguator section ──────────────────────
test.describe('C — Cluster disambiguator section (Sub-30)', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(CORPUS_URL);
        // Wait for the clusters section to finish loading.
        await expect(page.locator('#clusters-status')).not.toContainText(
            'Loading', { timeout: 15_000 });
    });

    test('Cluster disambiguator heading is visible', async ({ page }) => {
        await expect(page.locator('h2:has-text("Cluster disambiguator")')).toBeVisible();
    });

    test('sag_substantive cluster card renders with Disambiguate button', async ({ page }) => {
        const clusterCard = page.locator('.cluster-card[data-cluster-id="sag_substantive"]');
        await expect(clusterCard).toBeVisible({ timeout: 10_000 });

        // Cluster id is shown.
        await expect(clusterCard.locator('.cluster-id')).toContainText('sag_substantive');

        // Disambiguate button present.
        const btn = clusterCard.locator('button[data-action="disambig"]');
        await expect(btn).toBeVisible();
        await expect(btn).toContainText(/disambiguate/i);
    });

    test('Clicking Disambiguate opens inline panel', async ({ page }) => {
        const clusterCard = page.locator('.cluster-card[data-cluster-id="sag_substantive"]');
        await expect(clusterCard).toBeVisible({ timeout: 10_000 });

        // Panel must not exist yet.
        await expect(clusterCard.locator('[data-role="disambig-panel"]')).toHaveCount(0);

        await clusterCard.locator('button[data-action="disambig"]').click();

        const panel = clusterCard.locator('[data-role="disambig-panel"]');
        await expect(panel).toBeVisible({ timeout: 5_000 });

        await expect(panel.locator('[data-role="disambig-status"]')).toBeVisible();
        await expect(panel.locator('button[data-action="disambig-dispatch"]')).toBeVisible();
        await expect(panel.locator('button[data-action="disambig-dispatch"]'))
            .toContainText(/dispatch disambiguator/i);
    });

    test('"Dispatch disambiguator" click fires POST /clusters/:id/disambiguate', async ({ page }) => {
        const clusterCard = page.locator('.cluster-card[data-cluster-id="sag_substantive"]');
        await expect(clusterCard).toBeVisible({ timeout: 10_000 });
        await clusterCard.locator('button[data-action="disambig"]').click();
        const panel = clusterCard.locator('[data-role="disambig-panel"]');
        await expect(panel).toBeVisible({ timeout: 3_000 });

        let disambigPostFired = false;
        page.on('request', (req) => {
            if (/\/clusters\/[^/]+\/disambiguate$/.test(req.url()) && req.method() === 'POST') {
                disambigPostFired = true;
            }
        });

        await panel.locator('button[data-action="disambig-dispatch"]').click();

        await expect.poll(() => disambigPostFired, {
            message: 'a POST /clusters/<id>/disambiguate request must fire when Dispatch disambiguator is clicked',
            timeout: 8_000,
        }).toBe(true);

        // Status should transition away from the idle description.
        const statusEl = panel.locator('[data-role="disambig-status"]');
        await expect(statusEl).not.toContainText(
            'Dispatch cluster_disambiguator.mjs against this cluster', { timeout: 8_000 });
    });

    test('Disambiguate run has retrievable log via /experiments/runs/:run_id', async ({ request }) => {
        // Dispatch via API.
        const dispResp = await request.post(
            `${PLUGIN_BASE}/clusters/sag_substantive/disambiguate`, { data: {} });
        expect(dispResp.ok()).toBe(true);
        const { run_id } = await dispResp.json();
        expect(run_id).toBeTruthy();

        // Immediately poll status — run may still be in flight, that's fine.
        const statusResp = await request.get(`${PLUGIN_BASE}/experiments/runs/${run_id}`);
        expect(statusResp.ok()).toBe(true);
        const statusBody = await statusResp.json();
        expect(statusBody).toHaveProperty('run');
        expect(statusBody).toHaveProperty('log');
        // Log may be empty string or a short header line — both are valid.
        expect(typeof statusBody.log).toBe('string');
        // run record must have a kind field set by the plugin.
        expect(statusBody.run.kind).toBe('cluster_disambig');
        expect(statusBody.run.cluster_id).toBe('sag_substantive');
    });
});
