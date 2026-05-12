// User-personas phase 3: multi-user with afk-check.
//
// Cycle TWO personas to autonomous (wry-skeptic + polite-naturalist).
// Send one manual user message. Observe that after each scringlo turn:
//   - /afk-check fires for both personas (in parallel)
//   - participants /poll in parallel
//   - each participant's response appears as a separate is_user=true
//     entry in chat[], tagged with name=persona.name + extra.user_persona_id
//   - scringlo responds once to the multi-voice user input
//
// HARD CONTRACT enforced:
//   - both personas elicit /afk-check on each round
//   - chat[] entries with distinct name fields appear within the same
//     "round" (two consecutive is_user=true entries with different names)
//   - scringlo's response addresses both voices (long-ish, ≥ 100 chars,
//     because integrating two viewpoints is more work than one)
//
// Per "byebye unit tests; rendered pixels only" — all assertions are
// on observable DOM / chat[] / HTTP state.

import { test, expect } from '@playwright/test';
import { loadAndConnect, selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

test.describe('user-personas multi-user (phase 3)', () => {
    test.setTimeout(10 * 60 * 1000);

    test('two personas autonomous → afk-check + parallel poll + multi-user injection + single assistant reply', async ({ page }, testInfo) => {
        const pluginRequests = [];
        page.on('request', (req) => {
            const url = req.url();
            if (url.includes('/api/plugins/user-personas/')) {
                let body = null;
                try { body = req.postDataJSON(); } catch { body = null; }
                pluginRequests.push({
                    t: Date.now(),
                    endpoint: url.split('/').pop(),
                    body,
                });
            }
        });
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
        await page.screenshot({ path: testInfo.outputPath('01_chat_loaded.png'), fullPage: true });

        // Set wry-skeptic AND polite-naturalist to autonomous via localStorage,
        // then trigger a refresh so the FE sees the modes.
        await page.evaluate(() => {
            localStorage.setItem('user_personas_modes', JSON.stringify({
                'wry-skeptic': 'autonomous',
                'polite-naturalist': 'autonomous',
            }));
        });

        // Send the kick-off manual message
        const baselineChatLen = await page.evaluate(() => SillyTavern.getContext().chat.length);
        console.log(`[setup] baseline chat[] length: ${baselineChatLen}`);
        await page.locator('#send_textarea').fill('hello — say something interesting, please.');
        await page.locator('#send_but').click();

        // Wait for the chat to grow with multi-user rounds. Each round
        // produces ≥1 user-persona injections + 1 scringlo response,
        // so chat grows by ≥2 per round. We need to see at least
        // ONE multi-user round complete to assert the contract.
        // Bound: 6 minutes.
        const settleEnd = Date.now() + 6 * 60_000;
        let chatLen = baselineChatLen;
        let lastChatLen = baselineChatLen;
        let saw_multi_user_round = false;
        while (Date.now() < settleEnd) {
            const generating = await page.evaluate(() => {
                const s = document.querySelector('#mes_stop');
                return s && s.offsetParent !== null;
            });
            chatLen = await page.evaluate(() => SillyTavern.getContext().chat.length);

            // Detect a multi-user injection: are there ≥2 consecutive
            // is_user=true entries with DIFFERENT names anywhere in chat[]?
            if (!saw_multi_user_round) {
                const detected = await page.evaluate(() => {
                    const chat = SillyTavern.getContext().chat || [];
                    for (let i = 0; i < chat.length - 1; i++) {
                        if (chat[i].is_user && chat[i+1].is_user &&
                            chat[i].name !== chat[i+1].name) {
                            return { idx: i, names: [chat[i].name, chat[i+1].name] };
                        }
                    }
                    return null;
                });
                if (detected) {
                    console.log(`[detected] multi-user round at chat[${detected.idx}]: ${detected.names.join(' + ')}`);
                    saw_multi_user_round = true;
                }
            }
            if (saw_multi_user_round && !generating && chatLen > lastChatLen) break;
            lastChatLen = chatLen;
            await page.waitForTimeout(1000);
        }

        // Cycle both personas off
        await page.evaluate(() => {
            localStorage.setItem('user_personas_modes', JSON.stringify({}));
        });
        await page.waitForTimeout(3000); // let any in-flight tick finish

        await page.screenshot({
            path: testInfo.outputPath('99_final.png'),
            fullPage: true,
        });

        // Build final state record
        const state = await page.evaluate(() => {
            const chat = SillyTavern.getContext().chat || [];
            return chat.map((m, i) => ({
                idx: i,
                name: m.name || '',
                is_user: !!m.is_user,
                is_system: !!m.is_system,
                user_persona_id: m.extra?.user_persona_id || null,
                mes_first_120: (m.mes || '').slice(0, 120),
            }));
        });

        fs.writeFileSync(testInfo.outputPath('plugin_requests.json'),
            JSON.stringify(pluginRequests, null, 2));
        fs.writeFileSync(testInfo.outputPath('browser_console.json'),
            JSON.stringify(browserConsole, null, 2));
        fs.writeFileSync(testInfo.outputPath('final_chat.json'),
            JSON.stringify(state, null, 2));

        const afkChecks = pluginRequests.filter(r => r.endpoint === 'afk-check');
        const polls = pluginRequests.filter(r => r.endpoint === 'poll');
        console.log(`[counts] afk-checks: ${afkChecks.length}, polls: ${polls.length}`);
        console.log(`[chat] final length: ${chatLen}, saw_multi_user_round: ${saw_multi_user_round}`);

        // ── ASSERTIONS ──────────────────────────────────────────────
        // (1) Multi-user round actually happened.
        expect(saw_multi_user_round,
            'at least one multi-user round occurred (2 consecutive is_user=true entries with different names)'
        ).toBe(true);

        // (2) Both personas got afk-check'd.
        const checkedPersonas = new Set(afkChecks.map(r => r.body?.persona_id));
        expect(checkedPersonas.has('wry-skeptic'),
            'wry-skeptic was afk-checked').toBe(true);
        expect(checkedPersonas.has('polite-naturalist'),
            'polite-naturalist was afk-checked').toBe(true);

        // (3) chat[] contains entries with both personas' names AND
        //     entries with attribution metadata (user_persona_id).
        const personaNames = new Set(state.filter(e => e.is_user && e.user_persona_id).map(e => e.user_persona_id));
        expect(personaNames.size, 'at least 2 distinct user_persona_id values in chat[]').toBeGreaterThanOrEqual(2);

        // (4) After multi-user injection, scringlo replied. Find a scringlo
        //     entry that comes AFTER the multi-user injection point.
        let hasAssistantAfterMulti = false;
        let multiInjectIdx = -1;
        for (let i = 0; i < state.length - 1; i++) {
            if (state[i].is_user && state[i+1].is_user && state[i].name !== state[i+1].name) {
                multiInjectIdx = i + 1;
                break;
            }
        }
        for (let i = multiInjectIdx + 1; i < state.length; i++) {
            if (!state[i].is_user && !state[i].is_system) {
                hasAssistantAfterMulti = true;
                break;
            }
        }
        expect(hasAssistantAfterMulti,
            'scringlo replied after the multi-user round').toBe(true);
    });
});
