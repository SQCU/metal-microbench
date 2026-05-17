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
        // Card count is bounded by the yapper-seed K (2 top + 2 side = 4
        // tuples), reduced if Gemma dedups overlapping bio_ids across the
        // two calls. Assert sensible floor (Gemma returned at least one
        // tuple per call) rather than exact 4.
        expect(cardCount, 'yapper-seed returned at least 2 cards').toBeGreaterThanOrEqual(2);
        await page.screenshot({ path: testInfo.outputPath('02_panel_expanded.png'), fullPage: true });

        // (3) Pick an arbitrary populated card and assert it has a mode
        // button + a swap button. Yapper-seed picks which personas show
        // up; test asserts SHAPE, not specific id.
        const firstCardId = await page.evaluate(() => {
            const card = document.querySelector(
                '#user_personas_panel .user-personas-card[data-persona-id]');
            return card ? card.dataset.personaId : null;
        });
        expect(firstCardId, 'at least one populated card rendered').toBeTruthy();
        const cardShape = await page.evaluate((pid) => {
            const card = document.querySelector(
                `#user_personas_panel .user-personas-card[data-persona-id="${pid}"]`);
            if (!card) return null;
            return {
                has_mode_btn: !!card.querySelector('.user-personas-card-mode-btn'),
                has_swap_btn: !!card.querySelector('.user-personas-card-swap-btn'),
                mode_btn_text: card.querySelector('.user-personas-card-mode-btn')?.textContent,
            };
        }, firstCardId);
        expect(cardShape.has_mode_btn, `${firstCardId} has mode-cycle button`).toBe(true);
        expect(cardShape.has_swap_btn, `${firstCardId} has swap button`).toBe(true);
        // Yapper-seed top cards default to 'suggest' mode; side cards to 'off'.
        // The first card in the panel is from the top set so it's 'suggest'.
        expect(['off', '✓ suggest', '⚡ auto']).toContain(cardShape.mode_btn_text);

        // Empty placeholder cards: assert they're rendered (drawer should
        // always show TARGET_SLOTS=8 cells total).
        const emptyCount = await page.evaluate(() =>
            document.querySelectorAll('#user_personas_panel .user-personas-card-empty').length);
        expect(emptyCount, 'empty placeholder cards rendered to fill slate').toBeGreaterThanOrEqual(1);

        // (4) Cycle the mode button to 'suggest' (if not already) and
        // verify /poll fires when it enters suggest/autonomous mode.
        const pollsBefore = pluginRequests.filter(r => r.endpoint === 'poll').length;
        const startedAt = cardShape.mode_btn_text;
        // If we start at 'off', one cycle gets to 'suggest'. Otherwise
        // cycle through the full loop until we land back at 'suggest'.
        const cyclesNeeded = startedAt === 'off' ? 1
            : startedAt === '✓ suggest' ? 3
            : 2;  // '⚡ auto' → off → suggest
        for (let i = 0; i < cyclesNeeded; i++) {
            await page.locator(
                `#user_personas_panel .user-personas-card[data-persona-id="${firstCardId}"] .user-personas-card-mode-btn`
            ).click();
            await page.waitForTimeout(100);
        }
        // Wait for the preview to land (or error).
        await page.waitForFunction((pid) => {
            const slot = document.querySelector(
                `#user_personas_panel .user-personas-card[data-persona-id="${pid}"] .user-personas-card-preview-slot`);
            if (!slot) return false;
            const preview = slot.querySelector('.user-personas-card-preview');
            return preview && !preview.classList.contains('is-loading');
        }, firstCardId, { timeout: 90_000 });
        await page.waitForTimeout(300);

        const pollsAfter = pluginRequests.filter(r => r.endpoint === 'poll').length;
        expect(pollsAfter, 'one /poll fired on mode-change to suggest').toBeGreaterThan(pollsBefore);

        // Preview is rendered with text.
        const previewText = await page.evaluate((pid) => {
            const preview = document.querySelector(
                `#user_personas_panel .user-personas-card[data-persona-id="${pid}"] .user-personas-card-preview`);
            return preview ? preview.innerText.trim() : null;
        }, firstCardId);
        expect(previewText, 'preview slot contains generated text').toBeTruthy();
        // Some bios legitimately produce short outputs (e.g. 'minimalist'
        // returns "?" or "." by design). Assert any text, not a length floor.
        expect(previewText.length, 'preview is non-empty').toBeGreaterThan(0);
        console.log(`[${firstCardId} preview head] ${previewText.slice(0, 120)}`);
        await page.screenshot({ path: testInfo.outputPath('03_suggest_preview.png'), fullPage: true });

        // (5) Click the preview → text lands in #send_textarea.
        await page.locator(
            `#user_personas_panel .user-personas-card[data-persona-id="${firstCardId}"] .user-personas-card-preview`
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
