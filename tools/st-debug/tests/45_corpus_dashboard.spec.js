// A1 corpus effective-dim dashboard — full e2e spec.
//
// Validates the dashboard surface per docs/ui_spec_effdim_dashboard.md:
//   1. Drawer button installs (#user-corpus-button) in the top row.
//   2. Clicking opens drawer hosting corpus_dashboard.html in iframe.
//   3. PR number renders with current populated corpus, > 0 and < axis count.
//   4. Bio/agent/composition counts match /personas, /agents.
//   5. Per-axis bar chart renders, sorted descending by variance.
//   6. At least one dormant axis (variance=0) renders with the dormant marker.
//   7. Refresh button triggers a re-fetch.
//   8. Refresh appends a snapshot; subsequent history shows it.
//   9. Empty-state renders cleanly when corpus has 0 compositions.

import { test, expect } from '@playwright/test';
import { loadAndConnect } from './_helpers/elicit_clean.mjs';
import { openPersonaSurface } from './_helpers/open_persona_surface.js';

const PLUGIN_BASE = '/api/plugins/user-personas';

test.describe('corpus dashboard — desktop only', () => {
    test.setTimeout(2 * 60 * 1000);

    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'dashboard spec runs only on desktop project');
    });

    async function openCorpusTab(page) {
        await loadAndConnect(page);
        // Open via hamburger popover — .drawer-toggle is display:none after
        // sillytavern-fork e2973179d; direct click on wrapper is invalid.
        await openPersonaSurface(page, 'corpus');
        const iframe = page.frameLocator('iframe[src*="corpus.html"]');
        await expect(iframe.locator('h1').first(),
            'corpus.html renders inside the drawer iframe').toBeVisible({ timeout: 15_000 });
        // Wait for the PR tile to leave the loading sentinel "—".
        await expect.poll(async () => {
            return await iframe.locator('#pr-value').textContent();
        }, { timeout: 20_000 }).not.toBe('—');
        return iframe;
    }

    test('drawer installs, PR renders, counts match, bar chart + dormant + refresh + snapshot + empty state', async ({ page, request }) => {
        // Pull live truth so we can assert the dashboard agrees.
        const [agentsResp, personasResp, axesResp] = await Promise.all([
            request.get(`${PLUGIN_BASE}/agents`).then(r => r.json()),
            request.get(`${PLUGIN_BASE}/personas`).then(r => r.json()),
            request.get(`${PLUGIN_BASE}/axes`).then(r => r.json()),
        ]);
        const agents = agentsResp.agents || [];
        const personas = personasResp.personas || [];
        const axes = axesResp.axes || [];
        const compositionAgents = agents.filter(
            a => a.signature && typeof a.signature === 'object' && Object.keys(a.signature).length > 0);

        expect(compositionAgents.length, 'corpus must have >=2 compositions for PR to compute').toBeGreaterThanOrEqual(2);
        expect(axes.length, 'axis registry must be populated').toBeGreaterThan(0);

        // Capture network calls to verify refresh re-fetches.
        const networkCalls = [];
        page.on('request', (req) => {
            const u = req.url();
            if (u.includes('/api/plugins/user-personas/')) {
                networkCalls.push({
                    method: req.method(),
                    endpoint: u.replace(/^https?:\/\/[^/]+/, ''),
                    at: Date.now(),
                });
            }
        });

        // (1) + (2): drawer + iframe.
        const iframe = await openCorpusTab(page);

        // (3) PR number renders with the current corpus; > 0, < axis count.
        const prTxt = (await iframe.locator('#pr-value').textContent()).trim();
        const prVal = Number(prTxt);
        expect(Number.isFinite(prVal), `PR value should parse as number, got '${prTxt}'`).toBe(true);
        expect(prVal).toBeGreaterThan(0);
        expect(prVal).toBeLessThan(axes.length);

        // (4) Counts match /personas, /agents, compositions.
        await expect(iframe.locator('#bios-value')).toHaveText(String(personas.length));
        await expect(iframe.locator('#agents-value')).toHaveText(String(agents.length));
        await expect(iframe.locator('#compositions-value')).toHaveText(String(compositionAgents.length));

        // (5) Per-axis bar chart: at least one non-dormant row, sorted descending by variance.
        const axisRows = iframe.locator('#axis-table .axis-row');
        const rowCount = await axisRows.count();
        expect(rowCount, 'axis chart should have one row per axis').toBeGreaterThan(0);

        const variances = [];
        for (let i = 0; i < rowCount; i++) {
            const v = await axisRows.nth(i).getAttribute('data-variance');
            variances.push(Number(v));
        }
        // Sorted descending.
        for (let i = 1; i < variances.length; i++) {
            expect(variances[i - 1], `row ${i - 1} variance should be >= row ${i} variance (descending sort)`).toBeGreaterThanOrEqual(variances[i]);
        }
        // At least one nonzero bar.
        expect(variances.some(v => v > 0), 'at least one axis should have nonzero variance').toBe(true);

        // (6) At least one dormant axis (variance=0) with dormant class.
        const dormantRows = iframe.locator('#axis-table .axis-row.dormant');
        const dormantCount = await dormantRows.count();
        expect(dormantCount, 'at least one dormant axis should render').toBeGreaterThan(0);
        // First dormant row carries the "dormant" tag pill (programmatic class).
        await expect(dormantRows.first().locator('.dormant-tag').first()).toBeVisible();
        // And its bar carries the variance-dormant gray class.
        await expect(dormantRows.first().locator('.axis-bar.variance-dormant')).toHaveCount(1);
        // Top row (largest variance) carries the variance-hot green class.
        await expect(axisRows.first().locator('.axis-bar.variance-hot')).toHaveCount(1);

        // (7) Refresh button triggers a re-fetch of /agents and /personas.
        const refreshT0 = Date.now();
        networkCalls.length = 0; // clear baseline noise
        await iframe.locator('#refresh-btn').click();
        // Wait for both /agents and /personas to be re-requested after click.
        await expect.poll(() => {
            const after = networkCalls.filter(c => c.at >= refreshT0);
            const sawAgents = after.some(c => c.method === 'GET' && c.endpoint.endsWith('/agents'));
            const sawPersonas = after.some(c => c.method === 'GET' && c.endpoint.endsWith('/personas'));
            return sawAgents && sawPersonas;
        }, { timeout: 10_000 }).toBe(true);

        // (8) Refresh also appends a snapshot. Wait for history-list to
        // contain at least one row with the "← current" marker.
        await expect(iframe.locator('#history-list .history-row.current')).toBeVisible({ timeout: 10_000 });
        // Server-side check: the snapshots file now contains at least one row.
        const snapResp = await request.get(`${PLUGIN_BASE}/corpus-snapshot`);
        expect(snapResp.status()).toBe(200);
        const snapData = await snapResp.json();
        expect(Array.isArray(snapData.snapshots)).toBe(true);
        expect(snapData.snapshots.length).toBeGreaterThanOrEqual(1);
        const latest = snapData.snapshots[snapData.snapshots.length - 1];
        expect(latest.pr).toBeCloseTo(prVal, 1);
        expect(latest.n_compositions).toBe(compositionAgents.length);

        // (9) Empty-state: navigate the iframe to ?empty=1, dashboard
        // should still render (with an empty-state message) and not crash.
        const iframeEl = page.locator('iframe[src*="corpus.html"]').first();
        await iframeEl.evaluate((el) => {
            el.src = '/api/plugins/user-personas/static/corpus.html?empty=1';
        });
        await expect(iframe.locator('h1').first()).toBeVisible({ timeout: 15_000 });
        await expect(iframe.locator('#axis-empty-state')).toBeVisible({ timeout: 10_000 });
        // PR tile shows the no-data sentinel.
        await expect(iframe.locator('#pr-value')).toHaveText('—');
        // Compositions count = 0 in empty mode.
        await expect(iframe.locator('#compositions-value')).toHaveText('0');
    });
});
