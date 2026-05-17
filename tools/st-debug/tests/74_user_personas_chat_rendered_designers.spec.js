// User-personas: chat-rendered designer entry points (replaces deleted designer.html).
//
// The Bio Designer and Agent Designer are SillyTavern characters whose
// chat sessions ARE the design surface. This test asserts the entry
// points wire correctly:
//
//   (1) The yapper-drawer empty-card picker has a "design new bio with
//       Gemma…" option.
//   (2) Clicking it switches to the "User-persona Bio Designer" character.
//   (3) A fresh chat opens, the designer's first_mes greeting renders
//       in #chat as turn 1 (assistant side).
//   (4) Sending a user message yields an assistant reply from the
//       designer in #chat (proving the multiturn elicitation surface
//       actually works through ST's native pipeline).
//
// What this test deliberately does NOT cover:
//   - The full multiturn elicitation to a finalize marker. That
//     requires Gemma to actually decide the bio is ready, which is a
//     multi-turn conversation. A separate longer-form test covers
//     finalize parsing + canonical-store write.
//   - The Agent Designer entry point in the ⚙ panel. Similar shape;
//     covered by a sibling spec once this one stabilizes.

import { test, expect } from '@playwright/test';
import { loadAndConnect, selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

test.describe('user-personas chat-rendered designers (replaces designer.html)', () => {
    test.setTimeout(8 * 60 * 1000);

    test('empty card picker → "design new bio" → Bio Designer chat opens + assistant replies', async ({ page }, testInfo) => {
        // Capture plugin + chat-completion traffic for audit
        const pluginRequests = [];
        page.on('request', (req) => {
            const url = req.url();
            if (url.includes('/api/plugins/user-personas/') ||
                url.includes('/v1/chat/completions') ||
                url.includes('/api/characters/') ||
                url.includes('/api/chats/')) {
                let body = null;
                try { body = req.postDataJSON(); } catch {}
                pluginRequests.push({ t: Date.now(), endpoint: req.url().split('?')[0].split('/').slice(-2).join('/'), body });
            }
        });

        await loadAndConnect(page);
        // Start with a regular character + chat so we have a baseline to
        // switch FROM. scringlo is canonical and always present.
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);

        // (1) Panel + empty card picker exist on boot.
        const panel = page.locator('#user_personas_panel');
        await expect(panel, 'panel auto-renders on boot').toBeVisible({ timeout: 10_000 });
        // Expand if collapsed.
        await panel.locator('.user-personas-panel-header').click();
        await page.waitForTimeout(300);

        // Wait for any cards (populated or empty) to render.
        await page.waitForFunction(() => {
            return document.querySelectorAll('#user_personas_panel .user-personas-card').length > 0;
        }, { timeout: 30_000 });

        // Find the empty placeholder card via its UNIQUE add-button
        // (`.user-personas-card-empty-add`). The class
        // `.user-personas-card-empty` is overloaded — it's also applied
        // to the inner empty preview slot inside OFF-mode populated
        // cards by renderPreviewInto, so first() on that class is
        // ambiguous. The add button only exists on actual placeholder
        // cards. We click the button directly and resolve the parent
        // card via XPath/closest if needed.
        await page.waitForFunction(() => {
            return document.querySelectorAll('#user_personas_panel .user-personas-card-empty-add').length > 0;
        }, { timeout: 10_000 });

        const emptyAddBtn = page.locator('#user_personas_panel .user-personas-card-empty-add').first();
        await expect(emptyAddBtn, '+ add-yapper button on at least one empty card').toBeVisible();
        await page.screenshot({ path: testInfo.outputPath('01_panel_with_empty_cards.png'), fullPage: true });

        // Resolve the parent card for picker scoping (the picker is a
        // sibling of the add button inside the same card).
        const emptyCard = emptyAddBtn.locator('xpath=ancestor::div[contains(@class, "user-personas-card-empty")][1]');
        await emptyAddBtn.click();
        const picker = emptyCard.locator('.user-personas-card-empty-picker');
        await expect(picker, 'picker becomes visible after + click').toBeVisible();

        // (1) Verify the new "design new bio with Gemma…" option exists.
        const designNewBioBtn = picker.locator('[data-choice="design-new-bio"]');
        await expect(designNewBioBtn, '"design new bio" option present in picker').toBeVisible();
        await page.screenshot({ path: testInfo.outputPath('02_picker_open_with_design_option.png'), fullPage: true });

        // (2) Click it. This should switch to the Bio Designer character
        // and open a fresh chat with the designer's first_mes greeting.
        await designNewBioBtn.click();

        // Wait for character switch + new chat. The character header at
        // the top of #chat will update to "User-persona Bio Designer".
        await page.waitForFunction(() => {
            // ST renders the active character's name in a few places;
            // most reliable is the header text or the chat title attr.
            // We check window's selected character via JS.
            const ctx = window.SillyTavern?.getContext?.();
            if (!ctx) return false;
            const chid = ctx.characterId;
            if (chid === undefined || chid === null) return false;
            const char = ctx.characters?.[chid];
            return char && char.name === 'User-persona Bio Designer';
        }, { timeout: 30_000 });
        await page.screenshot({ path: testInfo.outputPath('03_switched_to_bio_designer.png'), fullPage: true });

        // (3) Designer's first_mes greeting renders in #chat as the
        // first assistant message.
        await page.waitForFunction(() => {
            const ctx = window.SillyTavern?.getContext?.();
            return ctx?.chat?.length >= 1 && ctx.chat[0] && !ctx.chat[0].is_user;
        }, { timeout: 30_000 });

        const greetingPresent = await page.evaluate(() => {
            const ctx = window.SillyTavern.getContext();
            const first = ctx.chat[0];
            return {
                is_user: first.is_user,
                mes_head: (first.mes || '').slice(0, 200),
                name: first.name,
            };
        });
        expect(greetingPresent.is_user, 'first turn is assistant (greeting)').toBe(false);
        expect(greetingPresent.mes_head.length, 'greeting is non-empty').toBeGreaterThan(20);
        console.log(`[greeting] ${greetingPresent.mes_head}`);

        // (4) Send a real user message and wait for an assistant reply.
        // We use the Bio Designer's first prompt as our user input —
        // "anxious novice asking too many questions" is one of the
        // examples in its first_mes.
        const ta = page.locator('#send_textarea');
        await ta.fill('anxious novice asking too many questions');
        await page.locator('#send_but').click();

        // Wait for the assistant reply to LAND COMPLETELY. ST creates
        // chat[2] early in the stream (often with just 1-3 chars) and
        // mutates .mes as tokens arrive. A naive `chat[2] !== undefined`
        // check fires on the first chunk, not the final reply. We
        // instead wait until the reply text is meaningfully long AND
        // hasn't grown for a beat — proxies for "streaming complete".
        await page.waitForFunction(() => {
            const ctx = window.SillyTavern?.getContext?.();
            if (!ctx?.chat?.length || ctx.chat.length < 3) return false;
            const m = ctx.chat[2];
            if (!m || m.is_user) return false;
            const text = m.mes || '';
            // Stash the seen length to detect monotonic growth pause.
            const w = window;
            w.__lastReplyLen = w.__lastReplyLen || { len: -1, stable_for: 0 };
            if (text.length === w.__lastReplyLen.len && text.length > 30) {
                w.__lastReplyLen.stable_for += 1;
                // Settled for ~3 polls (~1.5s at 500ms internal polling).
                if (w.__lastReplyLen.stable_for >= 3) return true;
            } else if (text.length !== w.__lastReplyLen.len) {
                w.__lastReplyLen = { len: text.length, stable_for: 0 };
            }
            return false;
        }, { timeout: 180_000, polling: 500 });

        const replyText = await page.evaluate(() => {
            const ctx = window.SillyTavern.getContext();
            return ctx.chat[2].mes || '';
        });
        expect(replyText.length, 'designer reply is meaningfully long').toBeGreaterThan(30);
        console.log(`[designer reply] ${replyText.slice(0, 300)}`);
        await page.screenshot({ path: testInfo.outputPath('04_designer_replied.png'), fullPage: true });

        // Forensic dump
        fs.writeFileSync(testInfo.outputPath('plugin_requests.json'),
            JSON.stringify(pluginRequests, null, 2));
        fs.writeFileSync(testInfo.outputPath('greeting.txt'), greetingPresent.mes_head);
        fs.writeFileSync(testInfo.outputPath('reply.txt'), replyText);

        // Confirm: the chat session that just happened persists as a
        // regular ST chat-history file. Future selectCharacter switches
        // back to Bio Designer + opening this chat will show the
        // greeting + user msg + reply we just rendered — fully
        // auditable, no separate UI surface.
        const chatLength = await page.evaluate(() => window.SillyTavern.getContext().chat.length);
        expect(chatLength, 'chat session has greeting + user + reply').toBeGreaterThanOrEqual(3);
    });
});
