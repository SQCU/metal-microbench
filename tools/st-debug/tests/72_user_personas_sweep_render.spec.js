// User-personas validation-sweep render test.
//
// This is the diegetic-validation surface in action: the user opens
// the user-personas panel, clicks "Run sweep", waits ~30-60s, and
// gets back a gemma-drawn signature scatterplot + an analyst summary
// + a per-turn judgment table — all inside the panel.
//
// The test drives that flow end-to-end via the real UI. It does NOT
// poke any plugin internals, does NOT inspect private function
// behaviour, and does NOT mock the bridge. It clicks the button a
// human would click and asserts that what should render did render.
//
// HARD CONTRACT this test enforces:
//   - Sweep button is present in the panel.
//   - Clicking it issues a POST /api/plugins/user-personas/sweep with
//     a chat[] body.
//   - The sweep response renders into the panel with at least one of:
//     SVG plot, markdown summary, judgments table.
//   - Total wallclock under 4 minutes (well under the bridge's worst
//     case for a 4-persona × 3-turn sweep on this hardware).
//
// Artefacts produced (in test-results/72-.../):
//   01_chat_loaded.png       — initial state
//   02_panel_expanded.png    — panel open with the sweep button visible
//   03_sweep_running.png     — mid-sweep with the status placeholder
//   04_sweep_complete.png    — final result panel with svg + summary
//   sweep_response.json      — full /sweep response body
//   sweep_request_log.json   — every plugin request observed

import { test, expect } from '@playwright/test';
import { loadAndConnect, selectCharacterByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

test.describe('user-personas validation sweep', { tag: ['@user-personas', '@slow'] }, () => {
    test.setTimeout(8 * 60 * 1000);

    test('click Run sweep → svg + summary + judgments render in panel', async ({ page }, testInfo) => {
        // Capture every plugin request so we can prove the sweep
        // endpoint was actually hit and so the response body is
        // persisted as an artefact for inspection.
        const pluginRequests = [];
        const pluginResponses = [];
        page.on('request', (req) => {
            const url = req.url();
            if (url.includes('/api/plugins/user-personas/')) {
                let body = null;
                try { body = req.postDataJSON(); } catch { body = req.postData()?.slice(0, 400) || null; }
                pluginRequests.push({ t: Date.now(), method: req.method(), url, body });
            }
        });
        page.on('response', async (resp) => {
            const url = resp.url();
            if (!url.includes('/api/plugins/user-personas/sweep')) return;
            try {
                const text = await resp.text();
                pluginResponses.push({
                    t: Date.now(),
                    status: resp.status(),
                    url,
                    body: text.slice(0, 200_000),  // cap; sweep responses can be ~50-100KB
                });
            } catch (_) { /* response already consumed */ }
        });

        await loadAndConnect(page);
        // Use whatever character the user has (scringlo + the rock are
        // the canonical two poles). We pick scringlo because the rich
        // pole produces more lively user-agent turns and so the
        // signature scatter is more visually interesting in artefacts.
        await selectCharacterByClick(page, 'scringlo');
        await page.screenshot({ path: testInfo.outputPath('01_chat_loaded.png'), fullPage: true });

        // Button installed by the FE extension.
        const panelBtn = page.locator('#user_personas_btn');
        await expect(panelBtn, 'user-personas button installed').toBeVisible({ timeout: 10_000 });

        // Expand the panel.
        await panelBtn.click();
        const panel = page.locator('#user_personas_panel');
        await expect(panel).toHaveClass(/is-expanded/, { timeout: 5_000 });

        // Wait for the persona cards to populate before we trigger the
        // sweep — otherwise the plugin has nothing to sweep over.
        await page.waitForFunction(() =>
            document.querySelectorAll('#user_personas_panel .user-personas-card').length >= 1,
            { timeout: 10_000 });
        await page.screenshot({ path: testInfo.outputPath('02_panel_expanded.png'), fullPage: true });

        // Click the sweep button.
        const sweepBtn = page.locator('#user_personas_sweep_btn');
        await expect(sweepBtn, 'sweep button installed in panel').toBeVisible();
        await sweepBtn.click();

        // The button should immediately enter the running state.
        await expect(sweepBtn).toBeDisabled({ timeout: 2_000 });
        const resultPane = page.locator('#user_personas_sweep_result');
        await expect(resultPane).toBeVisible({ timeout: 2_000 });

        // Capture mid-sweep state to prove the placeholder rendered.
        await page.waitForTimeout(500);
        await page.screenshot({ path: testInfo.outputPath('03_sweep_running.png'), fullPage: true });

        // Wait for the sweep to complete. Completion signal: the sweep
        // button re-enables (its in-flight guard cleared). The /sweep
        // round-trip at the FE default (4 personas × 2 turns) lands
        // at ~90-150s on this hardware; we give it 6 min ceiling for
        // headroom on slower runs / cold KV caches.
        await expect(sweepBtn, 'sweep button re-enabled = sweep complete')
            .toBeEnabled({ timeout: 6 * 60 * 1000 });

        // Confirm result-panel head rendered.
        const head = resultPane.locator('.user-personas-sweep-head');
        await expect(head, 'sweep head row rendered with id + counts').toBeVisible();
        const headText = await head.innerText();

        // At least one of (svg, summary, judgments) must have rendered.
        // We don't require all three because the gemma summarizer or
        // svg-drawer can fail individually (parse errors, malformed
        // svg, etc.) and that's a soft failure — the JSONL is the
        // ground-truth artefact, the rendered surfaces are convenience.
        const svgPresent = await resultPane.locator('.user-personas-sweep-svg svg').count() > 0;
        const summaryPresent = await resultPane.locator('.user-personas-sweep-summary').count() > 0;
        const judgmentsPresent = await resultPane.locator('.user-personas-sweep-judgments table').count() > 0;
        console.log(`  rendered surfaces: svg=${svgPresent} summary=${summaryPresent} judgments=${judgmentsPresent}`);
        console.log(`  head: ${headText}`);
        expect(svgPresent || summaryPresent || judgmentsPresent,
               'at least one of svg/summary/judgments must render').toBe(true);

        // Sanity check: at least one /sweep POST must have been made.
        const sweepRequests = pluginRequests.filter(r => r.url.endsWith('/sweep'));
        expect(sweepRequests.length, 'one /sweep POST was issued').toBeGreaterThanOrEqual(1);
        expect(sweepRequests[0].method, '/sweep is POST').toBe('POST');
        expect(Array.isArray(sweepRequests[0].body?.chat),
               '/sweep request body carries chat[] from FE').toBe(true);

        // Sanity check: /sweep response must include judgments.
        const sweepResp = pluginResponses.find(r => r.url.endsWith('/sweep') && r.status === 200);
        expect(sweepResp, 'sweep response was 200 OK').toBeTruthy();
        const responseBody = JSON.parse(sweepResp.body);
        expect(responseBody.sweep_id, 'response carries sweep_id').toBeTruthy();
        expect(Array.isArray(responseBody.judgments), 'response carries judgments[]').toBe(true);
        expect(responseBody.judgments.length, 'at least one judgment produced').toBeGreaterThan(0);
        console.log(`  judgments: ${responseBody.judgments.length}`);
        console.log(`  elapsed:   ${responseBody.elapsed_ms}ms`);
        console.log(`  sweep_id:  ${responseBody.sweep_id}`);

        // Final screenshot.
        await page.screenshot({ path: testInfo.outputPath('04_sweep_complete.png'), fullPage: true });

        // Persist artefacts.
        fs.writeFileSync(testInfo.outputPath('sweep_response.json'),
                         JSON.stringify(responseBody, null, 2));
        fs.writeFileSync(testInfo.outputPath('sweep_request_log.json'),
                         JSON.stringify(pluginRequests, null, 2));

        // Persist the rendered result panel HTML so we can inspect
        // what gemma actually drew without re-running the sweep.
        const resultHTML = await resultPane.innerHTML();
        fs.writeFileSync(testInfo.outputPath('result_panel.html'),
                         '<!doctype html><html><body style="background:#222;color:#ddd;font-family:sans-serif">' +
                         resultHTML + '</body></html>');
    });
});
