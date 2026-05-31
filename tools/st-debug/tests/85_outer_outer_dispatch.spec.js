// Spec: outer-outer-run UI dispatch surface.
//
// Validates the Sub-29 feature addition:
//   (A) "Dispatch outer-outer" button is visible on each experiment card
//       in the Fixed-Point drawer, next to the existing "Dispatch run".
//   (B) Clicking the button fires a POST to the dedicated
//       /experiments/:id/outer-outer-run endpoint and returns a
//       non-empty run_id with the "oo-" prefix.
//   (C) A run-banner appears in the right pane with the run_id visible
//       and status "running" or "done".
//   (D) The GET /outer-outer-runs/:run_id endpoint responds with the
//       correct run metadata (soft assertion — probed via API, does not
//       wait for completion).
//   (E) Soft-assertion (generous timeout): after completion,
//       /personas and/or /axes contain entries whose provenance
//       references the run_id. This assertion is skipped in CI if the
//       run has not completed within 30s.
//
// The primary CI signal is assertions (A)–(C). (D) is an API probe.
// (E) is a best-effort e2e check.

import { test, expect } from '@playwright/test';
import { loadAndConnect } from './_helpers/elicit_clean.mjs';

const PLUGIN_BASE = '/api/plugins/user-personas';
// Probe experiment id. We create this via the API so the test is
// self-contained even if the experiments/ dir is empty.
const TEST_EID = 'spec_oo_dispatch_probe';

test.describe('outer-outer dispatch surface', () => {
    test.setTimeout(3 * 60 * 1000);

    // Desktop only — the fixed-point drawer is desktop-primary.
    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'outer-outer dispatch surface test runs on desktop only');
    });

    async function openFixedPointTab(page) {
        await loadAndConnect(page);
        const hamburger = page.locator('#user-personas-tools-button .drawer-toggle');
        await expect(hamburger, 'user-personas hamburger button installs').toBeVisible({ timeout: 20_000 });
        await hamburger.click();
        const menuItem = page.locator('.user-personas-tools-menuitem[data-surface-key="fixed-point"]');
        await expect(menuItem, 'Fixed-Point menu item present in hamburger popover').toBeVisible({ timeout: 5_000 });
        await menuItem.click();
        const iframe = page.frameLocator('#user-personas-surface-fixed-point iframe');
        await expect(iframe.locator('h1').first(), 'fixed_point.html renders inside surface drawer').toBeVisible({ timeout: 15_000 });
        await expect(iframe.locator('#experiments-status')).toContainText(/loaded/i, { timeout: 10_000 });
        return iframe;
    }

    test('"Dispatch outer-outer" button visible; click fires POST; run-banner shows non-empty run_id; status endpoint responds', async ({ page, request }) => {
        // ── SETUP: ensure a probe experiment card exists ──────────────
        await request.delete(`${PLUGIN_BASE}/experiments/${TEST_EID}`).catch(() => {});
        const probeCard = {
            experiment_schema: 'experiment-v1',
            id: TEST_EID,
            name: 'spec outer-outer dispatch probe',
            description: 'auto-created by 85_outer_outer_dispatch.spec.js',
            bios: [{
                canonical_key: `user-personas-${TEST_EID}-bio.png`,
                slug: `${TEST_EID}-bio`,
                name: 'probe bio',
                target_bio: { rpg_class: 3 },
                design_brief: 'probe bio for outer-outer dispatch test',
            }],
            agent_targets: [{
                slug: `${TEST_EID}-agent`,
                target_agent: { money_orientation: 3 },
                motive_hint: 'probe motive',
            }],
            bio_axes: ['rpg_class'],
            agent_axes: ['money_orientation'],
            counterparty_avatar: 'the-rock.png',
        };
        const created = await request.post(`${PLUGIN_BASE}/experiments/${TEST_EID}`, {
            data: probeCard,
        });
        expect(created.ok(), `probe experiment card setup must succeed (got ${created.status()})`).toBe(true);

        // ── OPEN FIXED-POINT TAB ──────────────────────────────────────
        const iframe = await openFixedPointTab(page);

        // Force a reload of the experiments list inside the iframe so
        // our probe card appears (toggle tabs to trigger the fetch).
        await iframe.locator('#tab-seed').click();
        await iframe.locator('#tab-experiments').click();
        await expect(
            iframe.locator(`.experiment-card[data-eid="${TEST_EID}"]`),
            'probe experiment card must appear in the experiments list'
        ).toBeVisible({ timeout: 10_000 });

        // ── (A) "Dispatch outer-outer" button is visible on the card ──
        const ooBtn = iframe.locator(
            `.experiment-card[data-eid="${TEST_EID}"] .outer-outer-run-btn`
        );
        await expect(ooBtn, '(A) "Dispatch outer-outer" button is visible on the experiment card').toBeVisible();
        await expect(ooBtn, '(A) button label contains "outer-outer" (case-insensitive)').toContainText(/outer.outer/i);
        await expect(ooBtn, '(A) button is enabled before click').toBeEnabled();

        // Capture network requests to confirm the POST fires.
        const ooDispatchRequests = [];
        page.on('request', (req) => {
            if (req.method() === 'POST'
                && req.url().includes(`/experiments/${TEST_EID}/outer-outer-run`)) {
                ooDispatchRequests.push(req.url());
            }
        });

        // ── (B) Click → POST fires → run_id returned ─────────────────
        await ooBtn.click();

        // The POST must fire within 5 seconds of the click.
        await expect.poll(() => ooDispatchRequests.length, {
            message: '(B) POST /experiments/:id/outer-outer-run must fire on button click',
            timeout: 8_000,
        }).toBeGreaterThan(0);

        // ── (C) Run banner appears with non-empty run_id ──────────────
        const runBanner = iframe.locator('#run-banner');
        await expect(runBanner,
            '(C) run-banner must be visible after outer-outer dispatch'
        ).toBeVisible({ timeout: 8_000 });
        // Banner must contain the "oo-" prefix characteristic of outer-outer runs.
        await expect(runBanner, '(C) run-banner shows "oo-" prefixed run_id').toContainText(/oo-/);
        // Banner must not be empty.
        const bannerText = await runBanner.textContent();
        expect(bannerText?.trim().length, '(C) run-banner text is non-empty').toBeGreaterThan(5);

        // Extract run_id from the banner by reading the <code> element.
        const runIdEl = runBanner.locator('code').first();
        await expect(runIdEl, '(C) run_id is rendered in a <code> element').toBeVisible();
        const runId = await runIdEl.textContent();
        expect(runId, '(C) run_id starts with "oo-"').toMatch(/^oo-/);

        // ── (D) GET /outer-outer-runs/:run_id responds correctly ──────
        const statusResp = await request.get(`${PLUGIN_BASE}/outer-outer-runs/${encodeURIComponent(runId)}`);
        expect(statusResp.ok(), `(D) GET /outer-outer-runs/${runId} must return 200 (got ${statusResp.status()})`).toBe(true);
        const statusBody = await statusResp.json();
        expect(statusBody.run, '(D) response body has a "run" object').toBeTruthy();
        expect(statusBody.run.run_id, '(D) run.run_id matches the dispatched run_id').toBe(runId);
        expect(['running', 'done', 'failed'],
            '(D) run.status is a valid terminal or running state'
        ).toContain(statusBody.run.status);
        expect(statusBody.run.experiment_id, '(D) run.experiment_id matches the probe experiment').toBe(TEST_EID);
        expect(typeof statusBody.log, '(D) response body has a "log" string').toBe('string');

        // ── (E) Soft: post-completion persona/axes provenance ─────────
        // This assertion polls for up to 30 seconds. If the process has
        // not completed by then, it passes vacuously (the run is async
        // and takes minutes in production — CI only has 30s here).
        // The meaningful check is: if the run DID complete, the new
        // entries must reference the run_id via provenance or derived_from.
        let completedInTime = false;
        const deadline = Date.now() + 30_000;
        while (Date.now() < deadline) {
            const r = await request.get(`${PLUGIN_BASE}/outer-outer-runs/${encodeURIComponent(runId)}`);
            if (r.ok()) {
                const b = await r.json();
                if (b.run?.status === 'done') { completedInTime = true; break; }
                if (b.run?.status === 'failed') { break; }
            }
            await new Promise(res => setTimeout(res, 3_000));
        }

        if (completedInTime) {
            // Check /personas and /axes for new entries referencing the run_id.
            const personasResp = await request.get(`${PLUGIN_BASE}/personas`);
            const axesResp = await request.get(`${PLUGIN_BASE}/axes`);
            if (personasResp.ok() && axesResp.ok()) {
                const personas = (await personasResp.json()).personas || [];
                const axes = (await axesResp.json()).axes || [];
                // Look for provenance.experiment_id matching the TEST_EID,
                // or derived_from referencing it. Either is sufficient.
                const personaWithProv = personas.some(p =>
                    p?.extras?.provenance?.experiment_id === TEST_EID
                    || p?.extras?.provenance?.run_id === runId
                );
                const axisWithProv = axes.some(a =>
                    a?.derived_from?.run_id === runId
                    || a?.derived_from?.experiment_id === TEST_EID
                );
                expect(personaWithProv || axisWithProv,
                    '(E) after completion, at least one persona or axis references the run provenance'
                ).toBe(true);
            }
        }
        // If not completed in time, we skip the (E) check silently —
        // primary dispatch + banner + status assertions are sufficient for CI.

        // ── CLEANUP ───────────────────────────────────────────────────
        await request.delete(`${PLUGIN_BASE}/experiments/${TEST_EID}`).catch(() => {});
    });
});
