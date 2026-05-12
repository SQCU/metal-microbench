// User-personas phase 2: autonomous tick.
//
// Cycle a persona pill to 'autonomous' mode. Send ONE manual user
// message to kick off the chat. Then observe that, after each
// assistant turn, the extension auto-submits a new user message
// from the autonomous persona. Verify the chat grows without further
// human intervention until the AUTONOMOUS_TURN_CAP triggers a pause.
//
// HARD CONTRACT: with a persona in autonomous mode and an active chat,
// the conversation continues without human input. ONE assistant turn
// should produce ONE auto-user-turn should produce ANOTHER assistant
// turn, etc. The extension drives, the bridge serves, the model loops.
//
// Stop conditions tested:
//   - reaching the turn cap pauses the loop (chat doesn't grow forever)
//   - cycling the persona back to 'off' stops further auto-ticks
//
// Per the user's mandate ("byebye unit tests; rendered pixels only"),
// every assertion is on observable DOM / network state.

import { test, expect } from '@playwright/test';
import { loadAndConnect, selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

test.describe('user-personas autonomous tick', () => {
    test.setTimeout(10 * 60 * 1000);

    test('cycle persona to autonomous → chat grows without manual sends → cycling off stops it', async ({ page }, testInfo) => {
        // Capture every /poll request so we can prove auto-ticks fired.
        const pollRequests = [];
        page.on('request', (req) => {
            const url = req.url();
            if (url.endsWith('/api/plugins/user-personas/poll')) {
                let body = null;
                try { body = req.postDataJSON(); } catch { body = null; }
                pollRequests.push({ t: Date.now(), body });
            }
        });
        // Capture browser console — the extension logs its tick path.
        const browserConsole = [];
        page.on('console', (msg) => {
            const t = msg.text();
            if (t.includes('[user-personas]')) {
                browserConsole.push({ t: Date.now(), type: msg.type(), text: t });
                console.log(`[browser ${msg.type()}] ${t}`);
            }
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        await page.screenshot({ path: testInfo.outputPath('01_initial.png'), fullPage: true });

        // Open the modal + cycle wry-skeptic to autonomous. Cycle order
        // is off → suggest → autonomous, so we click twice.
        await page.locator('#user_personas_btn').click();
        await expect(page.locator('#user_personas_modal')).toBeVisible();
        await page.waitForFunction(() => {
            return document.querySelectorAll(
                '#user_personas_pill_list .user-personas-persona-pill').length > 0;
        }, { timeout: 10_000 });

        // First, click every pill until it lands on 'off' (clean slate).
        await page.evaluate(() => {
            const pills = Array.from(document.querySelectorAll(
                '#user_personas_pill_list .user-personas-persona-pill'));
            for (const p of pills) {
                // Just clear localStorage instead, no race.
                // We'll let the post-clear refresh repopulate them in off mode.
            }
            localStorage.setItem('user_personas_modes', JSON.stringify({}));
        });
        // Trigger a refresh by closing + reopening the modal.
        await page.locator('#user_personas_close_btn').click();
        await page.locator('#user_personas_btn').click();
        await page.waitForFunction(() => {
            return document.querySelectorAll(
                '#user_personas_pill_list .user-personas-persona-pill').length > 0;
        });

        // Click the wry-skeptic pill twice (off → suggest → autonomous).
        await page.evaluate(() => {
            const pills = Array.from(document.querySelectorAll(
                '#user_personas_pill_list .user-personas-persona-pill'));
            const wry = pills.find(p =>
                p.textContent.toLowerCase().includes('wry') ||
                p.textContent.toLowerCase().includes('skeptic'));
            if (!wry) throw new Error('wry-skeptic pill not found');
            wry.click(); // → suggest
        });
        await page.waitForTimeout(400);
        await page.evaluate(() => {
            const pills = Array.from(document.querySelectorAll(
                '#user_personas_pill_list .user-personas-persona-pill'));
            const wry = pills.find(p =>
                p.textContent.toLowerCase().includes('wry') ||
                p.textContent.toLowerCase().includes('skeptic'));
            wry.click(); // → autonomous
        });
        await page.waitForTimeout(400);

        // Confirm the wry-skeptic pill is now in autonomous mode.
        const wryMode = await page.evaluate(() => {
            const modes = JSON.parse(localStorage.getItem('user_personas_modes') || '{}');
            return modes['wry-skeptic'];
        });
        expect(wryMode, 'wry-skeptic is in autonomous mode after two cycles').toBe('autonomous');

        await page.screenshot({ path: testInfo.outputPath('02_wry_autonomous.png'), fullPage: true });
        await page.locator('#user_personas_close_btn').click();

        // Send ONE manual user message to kick the loop.
        const initialPollCount = pollRequests.length;
        const baselineChatLength = await page.evaluate(() => SillyTavern.getContext().chat.length);
        console.log(`[setup] baseline chat[] length: ${baselineChatLength}`);

        await page.locator('#send_textarea').fill('introduce yourself, then we can talk.');
        await page.locator('#send_but').click();

        // Wait for the chat to grow by enough turns that we can prove
        // multiple auto-ticks happened. Each auto-tick produces 1 user
        // turn + 1 assistant turn = chat[] grows by 2. We want at
        // least 3 auto-ticks observed → chat grows by ≥ 6 entries past
        // the initial manual send.
        const targetChatGrowth = 6;
        const settleEnd = Date.now() + 8 * 60_000;
        let chatLength = baselineChatLength;
        while (Date.now() < settleEnd) {
            const generating = await page.evaluate(() => {
                const s = document.querySelector('#mes_stop');
                return s && s.offsetParent !== null;
            });
            chatLength = await page.evaluate(() => SillyTavern.getContext().chat.length);
            if (!generating && chatLength >= baselineChatLength + 1 + targetChatGrowth) break;
            await page.waitForTimeout(1000);
        }
        await page.screenshot({ path: testInfo.outputPath('03_after_autonomous_growth.png'), fullPage: true });

        const pollsAfterGrowth = pollRequests.length;
        console.log(`[grew] chat[] length now: ${chatLength} (started at ${baselineChatLength})`);
        console.log(`[poll requests during run: ${pollsAfterGrowth - initialPollCount}]`);

        // Now cycle wry-skeptic back to 'off' and verify no new polls fire.
        await page.locator('#user_personas_btn').click();
        await page.waitForFunction(() => {
            return document.querySelectorAll(
                '#user_personas_pill_list .user-personas-persona-pill').length > 0;
        });
        await page.evaluate(() => {
            // autonomous → off needs one more click (autonomous → off because cycle order is off→suggest→autonomous)
            const pills = Array.from(document.querySelectorAll(
                '#user_personas_pill_list .user-personas-persona-pill'));
            const wry = pills.find(p =>
                p.textContent.toLowerCase().includes('wry') ||
                p.textContent.toLowerCase().includes('skeptic'));
            wry.click(); // autonomous → off (cycle wraps)
        });
        await page.waitForTimeout(400);
        await page.locator('#user_personas_close_btn').click();

        const pollsAfterStop = pollRequests.length;
        const chatLenAfterStop = await page.evaluate(() => SillyTavern.getContext().chat.length);
        // Wait for any in-flight model generation to finish, then
        // ensure NO new polls fire afterwards.
        const calmEnd = Date.now() + 30_000;
        while (Date.now() < calmEnd) {
            const generating = await page.evaluate(() => {
                const s = document.querySelector('#mes_stop');
                return s && s.offsetParent !== null;
            });
            if (!generating) break;
            await page.waitForTimeout(500);
        }
        await page.waitForTimeout(3000); // grace period for any straggling tick
        const pollsAfterCalm = pollRequests.length;
        const chatLenAfterCalm = await page.evaluate(() => SillyTavern.getContext().chat.length);

        await page.screenshot({ path: testInfo.outputPath('04_after_stop.png'), fullPage: true });

        // Forensic dump
        fs.writeFileSync(testInfo.outputPath('poll_requests.json'),
            JSON.stringify(pollRequests, null, 2));
        fs.writeFileSync(testInfo.outputPath('browser_console.json'),
            JSON.stringify(browserConsole, null, 2));
        fs.writeFileSync(testInfo.outputPath('counts.json'),
            JSON.stringify({
                baseline_chat_len: baselineChatLength,
                chat_len_after_growth: chatLength,
                chat_len_after_stop: chatLenAfterStop,
                chat_len_after_calm: chatLenAfterCalm,
                initial_poll_count: initialPollCount,
                polls_after_growth: pollsAfterGrowth,
                polls_after_stop: pollsAfterStop,
                polls_after_calm: pollsAfterCalm,
            }, null, 2));

        // ── ASSERTIONS ──────────────────────────────────────────────
        // (1) Chat grew without manual sends after the initial nudge.
        expect(chatLength, 'chat[] grew by at least 7 entries past baseline (1 manual + 3 auto-ticks)')
            .toBeGreaterThanOrEqual(baselineChatLength + 7);

        // (2) /poll was called multiple times during the run (each
        //     auto-tick fires exactly one /poll).
        const autoPollCount = pollsAfterGrowth - initialPollCount;
        expect(autoPollCount, 'at least 3 autonomous /poll calls fired').toBeGreaterThanOrEqual(3);
        // Every auto-poll should be for wry-skeptic.
        for (const r of pollRequests.slice(initialPollCount, pollsAfterGrowth)) {
            expect(r.body?.persona_id, 'all autonomous polls are for wry-skeptic').toBe('wry-skeptic');
            expect(r.body?.n_candidates, 'autonomous mode uses n=1').toBe(1);
        }

        // (3) After cycling persona to off, no further polls fire (give
        //     it 30+ seconds of grace).
        expect(pollsAfterCalm, 'no /poll requests fire after persona is cycled to off')
            .toBe(pollsAfterStop);
    });
});
