// User-personas truncation + kick-on-toggle invariants.
//
// Two distinct fixes validated:
//
// (1) Candidates must not end mid-sentence. The original /poll
//     default of max_tokens=300 truncated verbose personas
//     (Polite Naturalist, in particular) mid-thought. The plugin
//     now defaults to 800 and personas can override via
//     manifest.generation.max_tokens. The /poll response now
//     surfaces finish_reason per candidate so the FE can flag
//     truncation visually.
//
// (2) Switching a persona to autonomous mode now fires an
//     immediate auto-tick if there's already an assistant turn
//     to react to. Previously, autonomous only fired on the
//     NEXT MESSAGE_RECEIVED, which doesn't happen for a freshly-
//     loaded chat (the first_mes is loaded from disk, not
//     generated). This was the "I toggled autonomous and
//     nothing happened" UX failure.
//
// All assertions are on rendered DOM / chat[] / HTTP state.

import { test, expect } from '@playwright/test';
import { loadAndConnect, selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

test.describe('user-personas: truncation + kick-on-toggle', () => {
    test.setTimeout(8 * 60 * 1000);

    test('polite-naturalist preview ends with terminator (no truncation)', async ({ page }, testInfo) => {
        const pollResponses = [];
        page.on('response', async (resp) => {
            if (!resp.url().endsWith('/api/plugins/user-personas/poll')) return;
            try {
                const j = await resp.json();
                pollResponses.push({ t: Date.now(), body: j });
            } catch {}
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);

        // Expand panel and set polite-naturalist to suggest mode.
        await page.locator('#user_personas_btn').click();
        await page.waitForFunction(() => document.querySelectorAll(
            '#user_personas_panel .user-personas-card').length > 0,
            { timeout: 10_000 });
        await page.evaluate(() => {
            const sel = document.querySelector(
                '#user_personas_panel .user-personas-card[data-persona-id="polite-naturalist"] .user-personas-card-mode');
            sel.value = 'suggest';
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });
        // Wait for the preview to land.
        await page.waitForFunction(() => {
            const preview = document.querySelector(
                '#user_personas_panel .user-personas-card[data-persona-id="polite-naturalist"] .user-personas-card-preview');
            return preview && !preview.classList.contains('is-loading');
        }, { timeout: 90_000 });
        await page.waitForTimeout(300);
        await page.screenshot({ path: testInfo.outputPath('01_polite_preview.png'), fullPage: true });

        const previewText = await page.evaluate(() => {
            const preview = document.querySelector(
                '#user_personas_panel .user-personas-card[data-persona-id="polite-naturalist"] .user-personas-card-preview');
            // strip the trailing "truncated" warning if present
            const clone = preview.cloneNode(true);
            for (const child of Array.from(clone.children)) child.remove();
            return clone.innerText.trim();
        });

        // The plugin response carried finish_reason; verify it was 'stop'
        // (not 'length' which means truncation).
        const resp = pollResponses[pollResponses.length - 1];
        expect(resp, 'at least one /poll response captured').toBeTruthy();
        const candidate = resp.body?.candidates?.[0];
        expect(candidate, 'candidate present in response').toBeTruthy();
        console.log(`[poll] max_tokens_used=${resp.body?.max_tokens_used}`);
        console.log(`[poll] finish_reason=${candidate.finish_reason} truncated=${candidate.truncated}`);
        console.log(`[poll] text length: ${candidate.text.length}`);
        console.log(`[poll] tail: ...${candidate.text.slice(-80)}`);
        expect(candidate.finish_reason, 'polite-naturalist finished naturally (not by token-limit)').toBe('stop');
        expect(candidate.truncated, 'candidate is not marked truncated').toBe(false);

        // Belt and suspenders: the visible preview text ends with a
        // sentence-terminating character. (.?!"')…
        const terminators = '.?!"\'‘’“”…)';
        const lastChar = previewText[previewText.length - 1];
        expect(terminators.includes(lastChar),
            `preview ends with a sentence terminator (got '${lastChar}'); full tail: "${previewText.slice(-80)}"`)
            .toBe(true);

        fs.writeFileSync(testInfo.outputPath('preview.txt'), previewText);
    });

    test('toggle persona to autonomous after chat already has an assistant turn → immediate auto-tick', async ({ page }, testInfo) => {
        // Capture poll requests so we can prove a tick fired.
        const pollRequests = [];
        page.on('request', (req) => {
            if (req.url().endsWith('/api/plugins/user-personas/poll')) {
                let body = null;
                try { body = req.postDataJSON(); } catch {}
                pollRequests.push({ t: Date.now(), body });
            }
        });
        const browserConsole = [];
        page.on('console', (msg) => {
            const t = msg.text();
            if (t.includes('[user-personas]')) {
                browserConsole.push({ t: Date.now(), text: t });
                console.log(`[browser] ${t}`);
            }
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);

        // Send ONE manual user message so chat has an actual assistant
        // turn (scringlo's response).
        const baselineChatLen = await page.evaluate(() => SillyTavern.getContext().chat.length);
        await page.locator('#send_textarea').fill('introduce yourself briefly.');
        await page.locator('#send_but').click();
        // Wait for scringlo to respond.
        const responseEnd = Date.now() + 90_000;
        while (Date.now() < responseEnd) {
            const generating = await page.evaluate(() => {
                const s = document.querySelector('#mes_stop');
                return s && s.offsetParent !== null;
            });
            const chatLen = await page.evaluate(() => SillyTavern.getContext().chat.length);
            if (!generating && chatLen >= baselineChatLen + 2) break;
            await page.waitForTimeout(500);
        }
        await page.waitForTimeout(1500);

        // Now toggle wry-skeptic to autonomous. Without the kick-on-toggle
        // fix, nothing would happen until the user manually sent another
        // message. WITH the fix, a tick should fire immediately.
        await page.locator('#user_personas_btn').click();
        await page.waitForFunction(() => document.querySelectorAll(
            '#user_personas_panel .user-personas-card').length > 0,
            { timeout: 10_000 });
        const pollsBefore = pollRequests.length;
        const chatLenBefore = await page.evaluate(() => SillyTavern.getContext().chat.length);
        await page.evaluate(() => {
            const sel = document.querySelector(
                '#user_personas_panel .user-personas-card[data-persona-id="wry-skeptic"] .user-personas-card-mode');
            sel.value = 'autonomous';
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });

        // Wait for the auto-tick to fire — should fire within ~60s
        // (poll + send + scringlo response). We measure by chat[]
        // growth past the current length.
        const tickEnd = Date.now() + 3 * 60_000;
        while (Date.now() < tickEnd) {
            const chatLen = await page.evaluate(() => SillyTavern.getContext().chat.length);
            if (chatLen >= chatLenBefore + 2) break;
            await page.waitForTimeout(500);
        }
        await page.screenshot({ path: testInfo.outputPath('02_after_kick.png'), fullPage: true });

        const chatLenAfter = await page.evaluate(() => SillyTavern.getContext().chat.length);
        const pollsAfter = pollRequests.length;
        console.log(`[chat] grew from ${chatLenBefore} to ${chatLenAfter}`);
        console.log(`[poll] count: before=${pollsBefore} after=${pollsAfter}`);

        fs.writeFileSync(testInfo.outputPath('poll_requests.json'),
            JSON.stringify(pollRequests, null, 2));
        fs.writeFileSync(testInfo.outputPath('browser_console.json'),
            JSON.stringify(browserConsole, null, 2));

        // ASSERTIONS:
        // (1) Toggling autonomous fired at least one /poll without a
        //     manual user-send first.
        expect(pollsAfter, 'polls fired after toggling to autonomous (kick worked)')
            .toBeGreaterThan(pollsBefore);
        // (2) Chat grew — auto-tick actually submitted.
        expect(chatLenAfter, 'chat[] grew via autonomous tick (no manual send between toggle and growth)')
            .toBeGreaterThanOrEqual(chatLenBefore + 2);
    });
});
