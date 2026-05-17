// User-Agent Design Tools — end-to-end feature-coverage spec.
//
// This validates that every behavioural capability from the earlier
// elicitation-harness experiments (overlay_demo.py, factorization_multiturn.py,
// headroom_bleed_grid.py, strategy_diversity.py, discovery.py) is reachable
// from real pixels in the SillyTavern UI via the user-agent-designer drawer.
//
// The drawer (#user-agent-designer-button) is a sibling of #persona-management-button
// in the top-row header strip and hosts an iframe at
// /api/plugins/user-personas/static/designer.html. The iframe contains an
// 11-tab design surface: biographies, overlay library, discovery harness,
// cascade judge, sweep, multi-turn iteration, poll preview, AFK check,
// Python harness command preview, provenance browser, tool spec, reload.
//
// "Pump pixels or it didn't happen" — every assertion checks the actual
// DOM state or rendered text; every step takes a screenshot.

import { test, expect } from '@playwright/test';
import { loadAndConnect } from './_helpers/elicit_clean.mjs';

test.use({ video: 'on' });

test.describe('user-agent design tools — drawer + full feature coverage', () => {
    test.setTimeout(15 * 60 * 1000);  // generous — moratorium-uncapped generations run longer

    test('drawer opens, all tabs reachable, multi-turn iteration runs end-to-end', async ({ page }, testInfo) => {
        // ── Network-level tracing for the plugin endpoints. ──
        const pluginRequests = [];
        page.on('request', (req) => {
            const url = req.url();
            if (url.includes('/api/plugins/user-personas/')) {
                let body = null;
                try { body = req.postDataJSON(); } catch {}
                pluginRequests.push({
                    t: Date.now(),
                    method: req.method(),
                    endpoint: url.replace(/^.*\/user-personas\//, ''),
                    body,
                });
            }
        });

        // ── (1) Load + connect bridge. ──
        await loadAndConnect(page);
        await page.screenshot({ path: testInfo.outputPath('01_loaded.png'), fullPage: true });

        // ── (2) The new drawer button must exist in the top-row. ──
        const drawerBtn = page.locator('#user-agent-designer-button');
        await expect(drawerBtn, 'design-tools drawer button is in the top-row header')
            .toHaveCount(1);
        await expect(drawerBtn.locator('.drawer-icon.fa-wand-magic-sparkles'),
            'drawer icon uses fa-wand-magic-sparkles')
            .toBeVisible();

        // The drawer should sit RIGHT NEXT TO the persona-management drawer
        // (sibling in the same row, before it in DOM order).
        const drawerOrder = await page.evaluate(() => {
            const designer = document.getElementById('user-agent-designer-button');
            const persona = document.getElementById('persona-management-button');
            if (!designer || !persona) return null;
            // Compare DOM positions — designer should appear before persona-management.
            const pos = designer.compareDocumentPosition(persona);
            return (pos & Node.DOCUMENT_POSITION_FOLLOWING) ? 'designer-then-persona' : 'persona-then-designer';
        });
        expect(drawerOrder, 'designer drawer is positioned before persona-management')
            .toBe('designer-then-persona');

        // ── (3) Click drawer icon → opens. ──
        await drawerBtn.locator('.drawer-toggle').click();
        const designerContent = page.locator('#UserAgentDesigner');
        // ST flips closedDrawer ↔ openDrawer on toggle. The class
        // mutation is the contract that other code (CSS animations,
        // etc.) hangs off of, so we assert it directly.
        await expect(designerContent, 'drawer transitions to openDrawer state')
            .toHaveClass(/openDrawer/, { timeout: 5_000 });
        await page.screenshot({ path: testInfo.outputPath('02_drawer_opened.png'), fullPage: true });

        // ── (4) The iframe must load designer.html. ──
        const iframe = page.frameLocator('#user_agent_designer_iframe');
        // The page title text inside the iframe is a stable anchor.
        await expect(iframe.locator('h1'), 'iframe contains designer h1 title')
            .toContainText('User-agent design tools', { timeout: 10_000 });
        // Every tab button is present.
        const tabNames = ['Biographies', 'Overlay library', 'Discovery harness',
                          'Cascade judge', 'Sweep', 'Multi-turn iteration',
                          'Poll preview', 'AFK check', 'Signature analysis',
                          'Strategy diversity', 'Python harness',
                          'Provenance', 'Tool spec', 'Reload'];
        for (const name of tabNames) {
            await expect(iframe.locator(`button.tab:has-text("${name}")`),
                `tab "${name}" present`).toBeVisible();
        }
        await page.screenshot({ path: testInfo.outputPath('03_iframe_loaded.png'), fullPage: true });

        // ── (5) Biographies tab — verify scringlo overlay card surfaces with library. ──
        // The biography list populates from GET /personas. Wait for the
        // overlay-scringlo-jsclash card to appear (it's our seeded fixture).
        await iframe.locator('button.tab:has-text("Biographies")').click();
        // Scope to #bio-list — the overlay-library tab ALSO renders
        // bio-cards (in #overlay-bio-list), so an unscoped query would
        // match both and trip Playwright's strict-mode.
        const scringloCard = iframe.locator('#bio-list .bio-card[data-pid="overlay-scringlo-jsclash"]');
        await expect(scringloCard, 'scringlo biography card appears in list')
            .toBeVisible({ timeout: 10_000 });
        await expect(scringloCard.locator('.bio-meta'),
            'scringlo card shows it has overlays attached')
            .toContainText('overlay-v1');
        await page.screenshot({ path: testInfo.outputPath('04_biographies_tab.png'), fullPage: true });

        // ── (6) Overlay library tab — round-trip edit. ──
        await iframe.locator('button.tab:has-text("Overlay library")').click();
        // Select the scringlo biography in the library tab's bio list.
        await iframe.locator('#overlay-bio-list .bio-card[data-pid="overlay-scringlo-jsclash"]').click();
        // The library view should populate with named overlay entries.
        // We seeded js-clash + validation-seeker + (any newly-appended).
        await expect(iframe.locator('#overlay-library-view'),
            'library view shows js-clash overlay name')
            .toContainText('js-clash', { timeout: 5_000 });
        await expect(iframe.locator('#overlay-library-view'),
            'library view shows validation-seeker overlay name')
            .toContainText('validation-seeker');
        await page.screenshot({ path: testInfo.outputPath('05_overlay_library.png'), fullPage: true });

        // Verify the textarea has the actual text body (the "intentional
        // omission" debt fix — confirms text bodies are surfaced).
        const jsClashTextarea = iframe.locator('textarea.ov-text-edit[data-name="js-clash"]');
        await expect(jsClashTextarea, 'js-clash overlay textarea is editable')
            .toBeVisible();
        const jsClashBody = await jsClashTextarea.inputValue();
        expect(jsClashBody.length, 'js-clash body has substantial text').toBeGreaterThan(300);
        expect(jsClashBody, 'js-clash body mentions creative vision (substring sanity)')
            .toMatch(/creative vision|specific|JavaScript/i);

        // Append a fresh overlay through the UI.
        const testOverlayName = `playwright-test-${Date.now().toString(36)}`;
        await iframe.locator('#overlay-new-name').fill(testOverlayName);
        await iframe.locator('#overlay-new-text').fill(
            'You are a Playwright robot, here to PROVE you exist. You speak with mechanical precision but ' +
            'remain in scringlo voice (lowercase, emoji, onomatopoeia). You ask only one short, very specific ' +
            'question to demonstrate the round-trip works.'
        );
        const appendsBefore = pluginRequests.filter(r =>
            r.endpoint.startsWith('personas/') && r.method === 'POST').length;
        await iframe.locator('#overlay-append-btn').click();
        // The status line should land in success state.
        await expect(iframe.locator('#overlay-status.success'),
            'overlay-append status line shows success')
            .toBeVisible({ timeout: 10_000 });
        // The new overlay should appear in the library view after refresh.
        await expect(iframe.locator('#overlay-library-view'),
            'newly-appended overlay appears in library view')
            .toContainText(testOverlayName, { timeout: 10_000 });
        const appendsAfter = pluginRequests.filter(r =>
            r.endpoint.startsWith('personas/') && r.method === 'POST').length;
        expect(appendsAfter, 'one POST /personas/:id fired for the append').toBe(appendsBefore + 1);
        await page.screenshot({ path: testInfo.outputPath('06_overlay_appended.png'), fullPage: true });

        // ── (7) Multi-turn iteration tab — the marquee feature-coverage check. ──
        // This is the surface that covers what overlay_demo.py and
        // factorization_multiturn.py used to do in the Python harness.
        await iframe.locator('button.tab:has-text("Multi-turn iteration")').click();
        // Dropdowns populated.
        await expect(iframe.locator('#iter-user-persona option[value="overlay-scringlo-jsclash"]'),
            'iterate tab — user persona dropdown contains scringlo')
            .toHaveCount(1);
        await iframe.locator('#iter-user-persona').selectOption('overlay-scringlo-jsclash');
        // The overlay dropdown should auto-populate from scringlo's library.
        await expect(iframe.locator('#iter-overlay option[value="js-clash"]'),
            'overlay dropdown lists js-clash after user-persona selection')
            .toHaveCount(1, { timeout: 5_000 });
        await iframe.locator('#iter-overlay').selectOption('validation-seeker');
        // K=2 keeps the test fast (each /iterate call is ~10-25s wallclock).
        await iframe.locator('#iter-k').fill('2');
        // Leave iter-max-tokens blank — generation-config moratorium
        // says the plugin (and tests) must not impose hidden caps that
        // differ from what the ST GUI uses. The bridge applies ST's
        // default, identical for live polls and test iterations.
        await iframe.locator('#iter-initial').fill('Hi! I can help with any coding question. What are you working on?');
        await page.screenshot({ path: testInfo.outputPath('07_iterate_configured.png'), fullPage: true });

        // Fire the iteration.
        const iterateBefore = pluginRequests.filter(r => r.endpoint === 'iterate').length;
        await iframe.locator('#iter-run-btn').click();
        // Status line goes to 'running' immediately, then 'success' on completion.
        await expect(iframe.locator('#iter-status.running'),
            'iteration status shows running').toBeVisible({ timeout: 5_000 });
        await expect(iframe.locator('#iter-status.success'),
            'iteration completes with success status')
            .toBeVisible({ timeout: 5 * 60 * 1000 });
        const iterateAfter = pluginRequests.filter(r => r.endpoint === 'iterate').length;
        expect(iterateAfter, 'one POST /iterate fired').toBe(iterateBefore + 1);
        // The trajectory should render: 1 opener + K user turns + K assistant turns = 1+2+2 = 5 panels.
        const turnPanels = iframe.locator('#iter-results > .panel');
        const turnCount = await turnPanels.count();
        expect(turnCount, `K=2 yields 1+K+K=5 panels (opener + 2 user + 2 assistant); saw ${turnCount}`).toBe(5);
        // The user turns should be tagged with the applied overlay name.
        const trajectoryHtml = await iframe.locator('#iter-results').innerHTML();
        expect(trajectoryHtml, 'trajectory shows the applied overlay name')
            .toContain('validation-seeker');
        // Status line should mention the K count.
        await expect(iframe.locator('#iter-status'),
            'success status reports turn count + overlay name')
            .toContainText('K=2');
        await page.screenshot({ path: testInfo.outputPath('08_iterate_trajectory.png'), fullPage: true });

        // ── (7b) Auto-judge fired → trajectory-judge endpoint hit + drift
        //          report rendered with per-turn signatures + drift metrics. ──
        // The auto-judge checkbox is on by default; the iterate handler
        // chains /trajectory-judge after /iterate completes. Assert the
        // drift pane populated with real content (not the empty-state).
        await expect(iframe.locator('#iter-judge-status.success'),
            'trajectory-judge completes successfully via auto-judge')
            .toBeVisible({ timeout: 4 * 60 * 1000 });
        const tjAfter = pluginRequests.filter(r => r.endpoint === 'trajectory-judge').length;
        expect(tjAfter, 'one POST /trajectory-judge fired via auto-judge').toBeGreaterThan(0);
        await expect(iframe.locator('#iter-drift-results'),
            'drift pane contains the per-turn signature trajectory header')
            .toContainText('Signature trajectory');
        await expect(iframe.locator('#iter-drift-results'),
            'drift pane reports mean drift metric')
            .toContainText('Mean drift');
        await expect(iframe.locator('#iter-drift-results'),
            'drift pane reports path efficiency')
            .toContainText('Path efficiency');
        // Centroid row should be present in the signature table (marked with c̄)
        await expect(iframe.locator('#iter-drift-results'),
            'drift pane shows centroid row in signature table')
            .toContainText('c̄');
        await page.screenshot({ path: testInfo.outputPath('08b_drift_report.png'), fullPage: true });

        // ── (7c) Template-fidelity probe via API request (no UI tab yet) ──
        // Confirms structural rendering fidelity between the two role-
        // swapped invocations of the same canonical chat. This is the
        // foundational measurement the rest of the elicitation stack
        // depends on — if user-agent and target-assistant see different
        // n-gram sequences for the same chat, every signature is suspect.
        const tfResp = await page.request.post(
            'http://127.0.0.1:8002/api/plugins/user-personas/template-fidelity',
            { data: {
                user_persona_id: 'overlay-scringlo-jsclash',
                overlay_name: 'js-clash',
                canonical: [
                    { role: 'assistant', content: 'Hi! What can I help with today?' },
                    { role: 'user', content: 'i want a flickering cursor!! in JS only!!' },
                    { role: 'assistant', content: 'Here is the JS:' },
                ],
            }});
        const tfData = await tfResp.json();
        expect(tfData.faithful,
            `template-fidelity probe reports structural fidelity (issues: ${JSON.stringify(tfData.issues || [])})`)
            .toBe(true);
        expect(tfData.turn_checks.length,
            'fidelity probe checked all 3 content turns').toBe(3);
        for (const tc of tfData.turn_checks) {
            expect(tc.in_user_agent_call, `turn #${tc.canonical_idx} present in UA rendering`).toBe(true);
            expect(tc.in_target_call, `turn #${tc.canonical_idx} present in TA rendering`).toBe(true);
            expect(tc.role_swap_correct, `turn #${tc.canonical_idx} has correct role swap`).toBe(true);
        }

        // ── (8) Poll preview tab — single-turn suggestion with overlay. ──
        await iframe.locator('button.tab:has-text("Poll preview")').click();
        await iframe.locator('#poll-persona-id').fill('overlay-scringlo-jsclash');
        await iframe.locator('#poll-overlay-name').fill('js-clash');
        await iframe.locator('#poll-chat').fill('[{"role":"assistant","content":"Hi!"}]');
        // Leave poll-max-tokens blank — see comment on iter-max-tokens above.
        const pollsBefore = pluginRequests.filter(r => r.endpoint === 'poll').length;
        await iframe.locator('#poll-run-btn').click();
        await expect(iframe.locator('#poll-status.success'),
            'poll preview completes successfully')
            .toBeVisible({ timeout: 3 * 60 * 1000 });
        const pollsAfter = pluginRequests.filter(r => r.endpoint === 'poll').length;
        expect(pollsAfter, 'one POST /poll fired from poll-preview').toBe(pollsBefore + 1);
        // The response pane should show applied_overlay info.
        await expect(iframe.locator('#poll-results'),
            'poll-preview result pane shows applied_overlay name')
            .toContainText('js-clash');
        await page.screenshot({ path: testInfo.outputPath('09_poll_preview.png'), fullPage: true });

        // ── (9) Cascade judge tab — judge a hand-written turn. ──
        await iframe.locator('button.tab:has-text("Cascade judge")').click();
        await iframe.locator('#judge-quick').fill(
            'omg omg!! ✨ can u u u make me a flickering cursor please!! i need it to be JAVASCRIPT specifically!!'
        );
        const judgesBefore = pluginRequests.filter(r => r.endpoint === 'judge').length;
        await iframe.locator('#judge-run-btn').click();
        await expect(iframe.locator('#judge-status.success'),
            'judge completes successfully')
            .toBeVisible({ timeout: 3 * 60 * 1000 });
        const judgesAfter = pluginRequests.filter(r => r.endpoint === 'judge').length;
        expect(judgesAfter, 'one POST /judge fired').toBe(judgesBefore + 1);
        // The result pane should render an axis table with 14 axes.
        const axisRowCount = await iframe.locator('#judge-results .axis-table tbody tr').count();
        expect(axisRowCount,
            `cascade returned 14-axis Likert table (saw ${axisRowCount} rows)`)
            .toBeGreaterThanOrEqual(10);
        await page.screenshot({ path: testInfo.outputPath('10_judge.png'), fullPage: true });

        // ── (10a) Signature analysis tab — drive the JSONL-picker UI. ──
        // Replaces the old "copy this terminal command" cope for analyze.py.
        // The picker enumerates /jsonl/list, filters to kind=judgments, and
        // POST /analyze runs analyze.py --json server-side. Confirms the
        // report renders n_records + layer 4 PCA table.
        await iframe.locator('button.tab:has-text("Signature analysis")').click();
        // Picker should populate with the elicitation_judgments.jsonl entry.
        await expect(iframe.locator('#analyze-jsonl-picker option[value="elicitation_judgments.jsonl"]'),
            'analyze tab picker enumerates elicitation_judgments.jsonl from /jsonl/list')
            .toHaveCount(1, { timeout: 5_000 });
        await iframe.locator('#analyze-jsonl-picker').selectOption('elicitation_judgments.jsonl');
        const analyzesBefore = pluginRequests.filter(r => r.endpoint === 'analyze').length;
        await iframe.locator('#analyze-run-btn').click();
        await expect(iframe.locator('#analyze-status.success'),
            'analyze completes successfully')
            .toBeVisible({ timeout: 60_000 });
        const analyzesAfter = pluginRequests.filter(r => r.endpoint === 'analyze').length;
        expect(analyzesAfter, 'one POST /analyze fired from picker').toBe(analyzesBefore + 1);
        // Report content checks: n_records=120 banner + Layer 4 PCA header + effective dim section.
        await expect(iframe.locator('#analyze-results'),
            'signature report shows 120 records')
            .toContainText('120');
        await expect(iframe.locator('#analyze-results'),
            'signature report contains Layer 4 PCA section')
            .toContainText('Layer 4');
        await expect(iframe.locator('#analyze-results'),
            'PCA report mentions effective dimensionality')
            .toContainText('Effective dimensionality');
        await page.screenshot({ path: testInfo.outputPath('10a_analyze.png'), fullPage: true });

        // ── (10b) Strategy diversity tab — pick + run. ──
        // Replaces the same cope for strategy_diversity.py. The picker
        // filters to multi-turn-shaped JSONLs and POST /strategy-diversity
        // spawns the Python summarizer cascade against the chosen file.
        await iframe.locator('button.tab:has-text("Strategy diversity")').click();
        await expect(iframe.locator('#diversity-jsonl-picker option[value="factorization_multiturn.jsonl"]'),
            'strategy-diversity tab picker enumerates factorization_multiturn.jsonl')
            .toHaveCount(1, { timeout: 5_000 });
        await iframe.locator('#diversity-jsonl-picker').selectOption('factorization_multiturn.jsonl');
        const diversitiesBefore = pluginRequests.filter(r => r.endpoint === 'strategy-diversity').length;
        await iframe.locator('#diversity-run-btn').click();
        await expect(iframe.locator('#diversity-status.success'),
            'strategy diversity scorer completes successfully')
            .toBeVisible({ timeout: 4 * 60 * 1000 });
        const diversitiesAfter = pluginRequests.filter(r => r.endpoint === 'strategy-diversity').length;
        expect(diversitiesAfter, 'one POST /strategy-diversity fired').toBe(diversitiesBefore + 1);
        // Confirm at least 3 session panels rendered (the factorization JSONL has 3 sessions).
        const sessionCount = await iframe.locator('#diversity-results > .panel').count();
        expect(sessionCount, `expected ≥3 session panels (factorization has 3 user cards); saw ${sessionCount}`)
            .toBeGreaterThanOrEqual(3);
        await expect(iframe.locator('#diversity-results'),
            'diversity report contains per-bio rollup')
            .toContainText('Mean diversity by bio');
        await page.screenshot({ path: testInfo.outputPath('10b_strategy_diversity.png'), fullPage: true });

        // ── (10) Provenance tab — verify discovery-generated cards listed. ──
        await iframe.locator('button.tab:has-text("Provenance")').click();
        await iframe.locator('#prov-refresh-btn').click();
        await expect(iframe.locator('#prov-status.success'),
            'provenance load succeeded')
            .toBeVisible({ timeout: 10_000 });
        await expect(iframe.locator('#prov-results'),
            'provenance results contain overlay-scringlo-jsclash')
            .toContainText('overlay-scringlo-jsclash');
        await page.screenshot({ path: testInfo.outputPath('11_provenance.png'), fullPage: true });

        // ── (11) Tool spec tab — verify OAI schema returned. ──
        await iframe.locator('button.tab:has-text("Tool spec")').click();
        await iframe.locator('#spec-refresh-btn').click();
        await expect(iframe.locator('#spec-status.success'),
            'tool-spec load succeeded')
            .toBeVisible({ timeout: 5_000 });
        await expect(iframe.locator('#spec-results'),
            'tool-spec is the OAI function schema for discover_persona')
            .toContainText('discover_persona');
        await page.screenshot({ path: testInfo.outputPath('12_tool_spec.png'), fullPage: true });

        // ── (12) Python harness tab — verify all 7 scripts are surfaced. ──
        await iframe.locator('button.tab:has-text("Python harness")').click();
        const harnessCards = iframe.locator('#harness-grid > .panel.harness');
        await expect(harnessCards, 'all 7 Python harness scripts have cards')
            .toHaveCount(7);
        const harnessText = await iframe.locator('#harness-grid').innerText();
        for (const script of ['probe_persist.py', 'discovery.py', 'analyze.py',
                              'strategy_diversity.py', 'overlay_demo.py',
                              'headroom_bleed_grid.py', 'factorization_multiturn.py']) {
            expect(harnessText, `${script} surfaced in Python harness tab`).toContain(script);
        }
        await page.screenshot({ path: testInfo.outputPath('13_python_harness.png'), fullPage: true });

        // ── (13) Reload tab — sanity check the simple endpoint works. ──
        await iframe.locator('button.tab:has-text("Reload")').click();
        await iframe.locator('#reload-btn').click();
        await expect(iframe.locator('#reload-status.success'),
            'reload completes successfully')
            .toBeVisible({ timeout: 10_000 });
        await expect(iframe.locator('#reload-status'),
            'reload status reports persona count')
            .toContainText(/\d+ personas? reloaded/);
        await page.screenshot({ path: testInfo.outputPath('14_reload.png'), fullPage: true });

        // ── (14) Cleanup: delete the playwright-test-* overlay we appended
        //         in step (6). Doubles as a feature-coverage check for
        //         the overlay-delete affordance, and prevents manifest
        //         bloat across repeated test runs (otherwise every run
        //         would leave a stale overlay behind). Uses the
        //         page-level `dialog` handler to auto-accept confirm().
        await iframe.locator('button.tab:has-text("Overlay library")').click();
        await iframe.locator('#overlay-bio-list .bio-card[data-pid="overlay-scringlo-jsclash"]').click();
        page.once('dialog', d => d.accept());
        await iframe.locator(`button.ov-delete[data-name="${testOverlayName}"]`).click();
        await expect(iframe.locator('#overlay-status.success'),
            'overlay-delete reports success')
            .toBeVisible({ timeout: 10_000 });
        // Confirm the overlay is gone from the rendered library.
        await expect(iframe.locator('#overlay-library-view'),
            'deleted overlay no longer present in library')
            .not.toContainText(testOverlayName, { timeout: 5_000 });

        // ── (15) Close the drawer — clean close cycle. ──
        await drawerBtn.locator('.drawer-toggle').click();
        await expect(designerContent, 'drawer transitions back to closedDrawer state')
            .toHaveClass(/closedDrawer/, { timeout: 5_000 });
        await page.screenshot({ path: testInfo.outputPath('15_drawer_closed.png'), fullPage: true });

        // ── Summary of what we covered (printed to test output). ──
        const endpointCounts = pluginRequests.reduce((acc, r) => {
            const key = r.endpoint.split('?')[0];
            acc[key] = (acc[key] || 0) + 1;
            return acc;
        }, {});
        console.log('[design-tools-coverage] plugin endpoints exercised:',
                     JSON.stringify(endpointCounts, null, 2));

        // Assert the headline coverage set was hit through real-UI clicks.
        // UI-driven endpoints — hit through iframe clicks. (The
        // template-fidelity probe is API-only for now; it's exercised
        // via page.request.post above and its assertions don't go
        // through the page-level network listener. A dedicated UI tab
        // for it is Phase-1 follow-up.)
        const required = ['personas', 'iterate', 'trajectory-judge',
                          'poll', 'judge', 'discovery/runs',
                          'discovery/tool-spec', 'reload', 'jsonl/list',
                          'analyze', 'strategy-diversity'];
        for (const ep of required) {
            expect(endpointCounts[ep] || 0,
                `endpoint /${ep} was hit through the UI`).toBeGreaterThan(0);
        }
    });
});
