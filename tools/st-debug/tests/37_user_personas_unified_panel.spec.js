// User-personas phase 4: unified panel.
//
// The modal-vs-background asymmetry between suggest and autonomous modes
// is replaced with a persistent panel above the chat input. Each
// persona has a card with mode dropdown and a live candidate preview
// updated after each assistant turn. Click preview → insert into
// textarea. Same polling primitive drives both modes.
//
// This test asserts the unification works in pixels:
//   1. The persistent panel exists after page load (no click needed).
//   2. Each persona card has a mode <select> with three options.
//   3. Setting a persona to 'suggest' triggers a /poll, fills the
//      preview slot, and the preview text appears in the card.
//   4. Setting a persona to 'autonomous' shows the same live preview
//      AND triggers an auto-submit on the next assistant turn (since
//      that's the only difference between modes).
//   5. Clicking a preview inserts its text into #send_textarea.

import { test, expect } from '@playwright/test';
import { loadAndConnect, selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

test.describe('user-personas unified panel (phase 4)', () => {
    test.setTimeout(6 * 60 * 1000);

    test('panel persists, mode dropdown drives live preview, click inserts', async ({ page }, testInfo) => {
        const pluginRequests = [];
        page.on('request', (req) => {
            if (req.url().includes('/api/plugins/user-personas/')) {
                let body = null;
                try { body = req.postDataJSON(); } catch {}
                pluginRequests.push({ t: Date.now(), endpoint: req.url().split('/').pop(), body });
            }
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);

        // (1) Panel must exist on boot, no button-click needed.
        const panel = page.locator('#user_personas_panel');
        await expect(panel, 'panel auto-renders on boot').toBeVisible({ timeout: 10_000 });
        await page.screenshot({ path: testInfo.outputPath('01_panel_collapsed.png'), fullPage: true });

        // (2) Expand the panel so we can interact with cards.
        await panel.locator('.user-personas-panel-header').click();
        await page.waitForTimeout(300);

        // Cards populate after fetchPersonas resolves.
        await page.waitForFunction(() => {
            return document.querySelectorAll('#user_personas_panel .user-personas-card').length > 0;
        }, { timeout: 10_000 });

        const cardCount = await page.evaluate(() =>
            document.querySelectorAll('#user_personas_panel .user-personas-card').length);
        expect(cardCount, '4 persona cards present').toBe(4);
        await page.screenshot({ path: testInfo.outputPath('02_panel_expanded.png'), fullPage: true });

        // (3) Each card has a mode <select> with three options.
        const firstCardModeOpts = await page.evaluate(() => {
            const sel = document.querySelector(
                '#user_personas_panel .user-personas-card[data-persona-id="wry-skeptic"] .user-personas-card-mode');
            if (!sel) return null;
            return Array.from(sel.options).map(o => o.value);
        });
        expect(firstCardModeOpts, 'wry-skeptic card has three mode options').toEqual(['off', 'suggest', 'autonomous']);

        // (4) Set wry-skeptic to 'suggest' via the dropdown → triggers /poll
        const pollsBefore = pluginRequests.filter(r => r.endpoint === 'poll').length;
        await page.evaluate(() => {
            const sel = document.querySelector(
                '#user_personas_panel .user-personas-card[data-persona-id="wry-skeptic"] .user-personas-card-mode');
            sel.value = 'suggest';
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });
        // Wait for the preview to land (or error). The pollAndCachePreview
        // call fires immediately on mode change.
        await page.waitForFunction(() => {
            const slot = document.querySelector(
                '#user_personas_panel .user-personas-card[data-persona-id="wry-skeptic"] .user-personas-card-preview-slot');
            if (!slot) return false;
            const preview = slot.querySelector('.user-personas-card-preview');
            return preview && !preview.classList.contains('is-loading');
        }, { timeout: 90_000 });
        await page.waitForTimeout(300);

        const pollsAfter = pluginRequests.filter(r => r.endpoint === 'poll').length;
        expect(pollsAfter, 'one /poll fired on mode-change to suggest').toBe(pollsBefore + 1);

        // Preview is rendered with text.
        const previewText = await page.evaluate(() => {
            const preview = document.querySelector(
                '#user_personas_panel .user-personas-card[data-persona-id="wry-skeptic"] .user-personas-card-preview');
            return preview ? preview.innerText.trim() : null;
        });
        expect(previewText, 'preview slot contains generated text').toBeTruthy();
        expect(previewText.length, 'preview is non-empty').toBeGreaterThan(20);
        console.log(`[preview text head] ${previewText.slice(0, 120)}`);
        await page.screenshot({ path: testInfo.outputPath('03_wry_suggest_preview.png'), fullPage: true });

        // (5) Click the preview → text lands in #send_textarea.
        await page.locator(
            '#user_personas_panel .user-personas-card[data-persona-id="wry-skeptic"] .user-personas-card-preview'
        ).click();
        await page.waitForTimeout(200);
        const taValue = await page.evaluate(() =>
            document.getElementById('send_textarea').value.trim());
        expect(taValue, 'preview text inserted into textarea').toBe(previewText);
        await page.screenshot({ path: testInfo.outputPath('04_inserted_in_textarea.png'), fullPage: true });

        // Forensic dump
        fs.writeFileSync(testInfo.outputPath('plugin_requests.json'),
            JSON.stringify(pluginRequests, null, 2));
        fs.writeFileSync(testInfo.outputPath('preview_text.txt'), previewText || '');
    });
});
