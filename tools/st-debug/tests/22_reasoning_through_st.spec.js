// End-to-end demonstration that the SillyTavern fork correctly
// REQUESTS reasoning_effort and SURFACES the bridge's
// reasoning_content as a rendered "thoughts" panel on the assistant
// turn.
//
// Drives ST as the client (no simulated/inline-HTML test client). Two
// trials in one ST session:
//   trial A: reasoning_effort = "auto"   → ST omits the param; bridge
//            returns content only; ST renders no thoughts panel.
//   trial B: reasoning_effort = "high"  → ST sends the param via its
//            own settings/UI; bridge returns reasoning_content +
//            content; ST renders the collapsible thoughts section
//            showing the trace, plus the regular assistant prose.
//
// Asserts deterministically:
//   - trial A: chat-completions request body has NO reasoning_effort
//   - trial A: chat[last].extra.reasoning is empty
//   - trial B: request body's reasoning_effort === "high"
//   - trial B: chat[last].extra.reasoning is non-empty
//
// Records video + screenshots of the actual ST UI showing the
// thoughts panel populated. Artifacts copy into docs/media at the end.

import { test, expect } from '@playwright/test';
import {
    loadAndConnect, sendAndObserve,
    selectCharacterByClick, freshChatByClick,
} from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

async function setReasoningEffort(page, value) {
    // Drive ST's actual UI control: the openai_reasoning_effort
    // dropdown. This is the same path a human selecting from the
    // dropdown uses; ST's change handler updates oai_settings and
    // saves. No internal-API poking — just clicking the documented
    // settings surface.
    return await page.evaluate(async (v) => {
        const sel = document.getElementById('openai_reasoning_effort');
        if (!sel) return { ok: false, reason: 'no #openai_reasoning_effort element' };
        sel.value = v;
        // ST listens for 'input' on the dropdown (openai.js line 6876),
        // not 'change'. Trigger via jQuery so any select2 wrappers also
        // get notified.
        const win = /** @type {any} */ (window);
        if (win.jQuery) {
            win.jQuery(sel).trigger('input');
        } else {
            sel.dispatchEvent(new Event('input', { bubbles: true }));
        }
        await new Promise(r => setTimeout(r, 200));
        // Verify by re-rendering ST's getReasoningEffort path.
        const ctx = win.SillyTavern?.getContext?.();
        return {
            ok: true,
            sel_value: sel.value,
            // ctx exposes oai_settings via the public API.
            oai_value: ctx?.chatCompletionSettings?.reasoning_effort
                       ?? ctx?.oai_settings?.reasoning_effort
                       ?? null,
        };
    }, value);
}

test.describe('SillyTavern fork: reasoning_effort request → reasoning_content render', () => {
    test.setTimeout(10 * 60 * 1000);

    test('trial A (auto): no reasoning_effort sent. trial B (high): reasoning_effort=high sent, thoughts rendered.', async ({ page }, testInfo) => {
        const captured = [];
        page.on('request', (req) => {
            if (req.url().includes('/api/backends/chat-completions/generate')) {
                let body = null;
                try { body = req.postDataJSON(); } catch (_) {}
                captured.push({ body, ts: Date.now() });
            }
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        await page.screenshot({ path: testInfo.outputPath('00_ready.png') });

        // ── Trial A: reasoning_effort = "auto" (ST omits the param) ─
        const aSet = await setReasoningEffort(page, 'auto');
        console.log('setReasoningEffort(auto) →', JSON.stringify(aSet));
        const aResult = await sendAndObserve(
            page,
            'in 2 short sentences: are you scringlo?',
            { timeoutMs: 4 * 60 * 1000 });
        const aReasoning = await page.evaluate(() => {
            const ctx = window.SillyTavern.getContext();
            const last = (ctx.chat || [])[ctx.chat.length - 1] || {};
            return {
                mes: (last.mes || '').slice(0, 300),
                reasoning: last.extra?.reasoning || '',
                reasoning_duration: last.extra?.reasoning_duration ?? null,
            };
        });
        await page.screenshot({
            path: testInfo.outputPath('01_trial_a_auto.png'),
            fullPage: true,
        });
        const aReq = captured[captured.length - 1] || { body: {} };

        // ── Trial B: reasoning_effort = "high" (ST should send it) ──
        const bSet = await setReasoningEffort(page, 'high');
        console.log('setReasoningEffort(high) →', JSON.stringify(bSet));
        const bResult = await sendAndObserve(
            page,
            'in your own voice and 2-3 sentences, walk me through your reasoning about whether you can draw a perfect circle without tools.',
            { timeoutMs: 6 * 60 * 1000 });
        // Open any rendered reasoning collapsible so the screenshot
        // shows the trace contents, not just the toggle.
        await page.evaluate(() => {
            for (const d of document.querySelectorAll(
                '#chat .mes details, #chat .mes .mes_reasoning_details')) {
                d.setAttribute('open', '');
            }
        });
        await page.waitForTimeout(500);
        const bReasoning = await page.evaluate(() => {
            const ctx = window.SillyTavern.getContext();
            const last = (ctx.chat || [])[ctx.chat.length - 1] || {};
            return {
                mes: (last.mes || '').slice(0, 300),
                reasoning: last.extra?.reasoning || '',
                reasoning_duration: last.extra?.reasoning_duration ?? null,
            };
        });
        await page.screenshot({
            path: testInfo.outputPath('02_trial_b_high.png'),
            fullPage: true,
        });
        const bReq = captured[captured.length - 1] || { body: {} };

        // ── Logs ────────────────────────────────────────────────────
        console.log('=== TRIAL A (reasoning_effort=auto) ===');
        console.log('request.reasoning_effort =', aReq.body?.reasoning_effort);
        console.log('chat.extra.reasoning len =', aReasoning.reasoning.length);
        console.log('chat.mes head            =', aReasoning.mes);
        console.log('=== TRIAL B (reasoning_effort=high) ===');
        console.log('request.reasoning_effort =', bReq.body?.reasoning_effort);
        console.log('chat.extra.reasoning len =', bReasoning.reasoning.length);
        console.log('chat.extra.reasoning head=', bReasoning.reasoning.slice(0, 240));
        console.log('chat.mes head            =', bReasoning.mes);

        fs.writeFileSync(testInfo.outputPath('reasoning_through_st.json'),
            JSON.stringify({
                trial_a: {
                    request_body_reasoning_effort: aReq.body?.reasoning_effort ?? null,
                    chat_extra_reasoning: aReasoning.reasoning,
                    chat_mes: aReasoning.mes,
                    finish_state: aResult.finishState,
                },
                trial_b: {
                    request_body_reasoning_effort: bReq.body?.reasoning_effort ?? null,
                    chat_extra_reasoning: bReasoning.reasoning,
                    chat_mes: bReasoning.mes,
                    finish_state: bResult.finishState,
                },
            }, null, 2));

        // ── Deterministic assertions ───────────────────────────────
        // Trial A: ST omits reasoning_effort when it's "auto"
        // (openai.js#getReasoningEffort returns undefined for auto).
        expect(aReq.body?.reasoning_effort,
            'auto → no reasoning_effort sent to backend').toBeFalsy();
        expect(aReasoning.reasoning.length,
            'auto → no reasoning rendered on assistant turn').toBe(0);

        // Trial B: ST sends reasoning_effort=high through to the
        // bridge AND renders the streamed reasoning_content as
        // chat[last].extra.reasoning.
        expect(bReq.body?.reasoning_effort,
            'high → reasoning_effort=high in request body').toBe('high');
        expect(bReasoning.reasoning.length,
            'high → reasoning_content rendered on assistant turn').toBeGreaterThan(0);
    });
});
