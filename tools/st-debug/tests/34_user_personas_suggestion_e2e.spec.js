// User-personas suggestion mode, end-to-end.
//
// Drives the real chat UI: scringlo character, open the user-personas
// modal, select wry-skeptic (and one or two others), click "Suggest
// replies", wait for candidates to render, click one, assert it lands
// in #send_textarea.
//
// Tests every layer:
//   - ST FE extension auto-loads (button appears in send-form area)
//   - GET /api/plugins/user-personas/personas returns 4+ entries
//   - Modal renders persona pills
//   - POST /api/plugins/user-personas/poll runs against the bridge
//   - Candidates appear in the modal under each active persona
//   - Click-to-insert puts the chosen candidate into #send_textarea
//
// HARD CONTRACT this test enforces: the suggestion-mode flow produces
// real persona-conditioned candidates and inserts them into the actual
// chat input. Anything less than that is a broken integration.

import { test, expect } from '@playwright/test';
import { loadAndConnect, selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

test.describe('user-personas suggestion mode', () => {
    test.setTimeout(8 * 60 * 1000);

    test('select wry-skeptic, click Suggest, candidates appear, click candidate, lands in #send_textarea', async ({ page }, testInfo) => {
        // Capture HTTP requests so we can prove /poll was actually called
        const pluginRequests = [];
        page.on('request', (req) => {
            const url = req.url();
            if (url.includes('/api/plugins/user-personas/')) {
                let body = null;
                try { body = req.postDataJSON(); } catch { body = req.postData()?.slice(0, 400) || null; }
                pluginRequests.push({ t: Date.now(), method: req.method(), url, body });
            }
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        await page.screenshot({ path: testInfo.outputPath('01_chat_loaded.png'), fullPage: true });

        // The button is installed by the extension on boot. Wait for it.
        const btn = page.locator('#user_personas_btn');
        await expect(btn, 'user-personas button installed by FE extension').toBeVisible({ timeout: 10_000 });

        await btn.click();
        const modal = page.locator('#user_personas_modal');
        await expect(modal, 'modal opens on button click').toBeVisible();
        await page.screenshot({ path: testInfo.outputPath('02_modal_open.png'), fullPage: true });

        // Pills populate after fetch.
        await page.waitForFunction(() => {
            const list = document.getElementById('user_personas_pill_list');
            return list && list.querySelectorAll('.user-personas-persona-pill').length > 0;
        }, { timeout: 10_000 });
        const pillCount = await page.evaluate(() =>
            document.querySelectorAll('#user_personas_pill_list .user-personas-persona-pill').length);
        // Pill count tracks the canonical persona store; no hardcoded subset.
        expect(pillCount, 'at least the canonical user-persona set present').toBeGreaterThanOrEqual(4);

        // Stress: rapidly click multiple pills to trigger overlapping
        // refreshes. Without the in-flight serialization, this race-
        // duplicates the pill list (first-cut bug). Assert no duplicates.
        await page.evaluate(() => {
            const pills = document.querySelectorAll('#user_personas_pill_list .user-personas-persona-pill');
            for (const p of pills) p.click();
        });
        await page.waitForTimeout(800);
        const pillCountAfterStress = await page.evaluate(() =>
            document.querySelectorAll('#user_personas_pill_list .user-personas-persona-pill').length);
        expect(pillCountAfterStress,
            'pill count is stable after rapid click stress (no race duplication)').toBe(pillCount);

        // Activate ONLY wry-skeptic (deactivate others if any are active).
        await page.evaluate(() => {
            const pills = document.querySelectorAll('#user_personas_pill_list .user-personas-persona-pill');
            // Toggle off any active that isn't wry-skeptic, then activate wry-skeptic
            for (const p of pills) {
                const name = p.textContent.trim();
                const wantActive = name.toLowerCase().includes('wry') || name.toLowerCase().includes('skeptic');
                const isActive = p.classList.contains('is-active');
                if (wantActive !== isActive) p.click();
            }
        });
        await page.waitForTimeout(300);
        await page.screenshot({ path: testInfo.outputPath('03_wry_skeptic_selected.png'), fullPage: true });

        // Set N=2 candidates (for speed; still enough to prove diversity)
        await page.evaluate(() => {
            const input = document.getElementById('user_personas_n_input');
            input.value = '2';
        });

        // Click Suggest replies. The button issues a POST /poll under the hood.
        await page.locator('#user_personas_suggest_btn').click();

        // Wait for candidates to render (drop the loading placeholder).
        await page.waitForFunction(() => {
            const root = document.getElementById('user_personas_candidates_root');
            if (!root) return false;
            const realCandidates = root.querySelectorAll('.user-personas-candidate:not(.is-loading)');
            return realCandidates.length > 0;
        }, { timeout: 60_000 });
        await page.waitForTimeout(500);
        await page.screenshot({ path: testInfo.outputPath('04_candidates_rendered.png'), fullPage: true });

        // Read the candidates that rendered
        const candidates = await page.evaluate(() =>
            Array.from(document.querySelectorAll('#user_personas_candidates_root .user-personas-candidate:not(.is-loading)'))
                .map(c => c.innerText.trim()));
        console.log(`[candidates] ${candidates.length}:`);
        for (const c of candidates) console.log(`  - ${c.slice(0, 140)}`);
        expect(candidates.length, 'at least 1 candidate rendered').toBeGreaterThan(0);

        // Click the first candidate
        const firstCandidate = candidates[0];
        await page.locator('#user_personas_candidates_root .user-personas-candidate:not(.is-loading)').first().click();

        // Modal should close
        await page.waitForTimeout(500);
        const modalOpen = await page.evaluate(() => {
            const m = document.getElementById('user_personas_modal');
            return m && m.open;
        });
        expect(modalOpen, 'modal closes after candidate click').toBe(false);

        // #send_textarea should now contain the candidate text
        const textareaValue = await page.evaluate(() => {
            const ta = document.getElementById('send_textarea');
            return ta ? ta.value : null;
        });
        await page.screenshot({ path: testInfo.outputPath('05_candidate_in_textarea.png'), fullPage: true });
        expect(textareaValue, 'candidate text inserted into #send_textarea').toBe(firstCandidate);

        // Forensic evidence dump
        fs.writeFileSync(testInfo.outputPath('plugin_requests.json'),
            JSON.stringify(pluginRequests, null, 2));
        fs.writeFileSync(testInfo.outputPath('candidates.txt'),
            candidates.join('\n────\n'));

        // INVARIANT: /poll was called with the wry-skeptic persona
        const pollReq = pluginRequests.find(r =>
            r.method === 'POST' && r.url.endsWith('/poll'));
        expect(pollReq, '/poll request was issued').toBeTruthy();
        expect(pollReq.body?.persona_id, 'poll was for wry-skeptic').toBe('wry-skeptic');
        expect(Array.isArray(pollReq.body?.chat), 'chat array sent to poll').toBe(true);

        console.log('[invariant] /poll request body:', JSON.stringify({
            persona_id: pollReq.body.persona_id,
            n_candidates: pollReq.body.n_candidates,
            chat_length: pollReq.body.chat?.length,
        }));
    });
});
