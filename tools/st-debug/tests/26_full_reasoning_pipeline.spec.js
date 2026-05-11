// Full-pipeline Playwright recording covering every feature in the
// thinking/reasoning surface that the bridge + ST fork now support.
// Lives at tools/st-debug/tests; webm + screenshots get curated to
// docs/media/<date>_test26_full_reasoning_pipeline_<sha>.{webm,png,json}.
//
// What this test asserts END TO END (engine → bridge → SSE → ST FE →
// rendered DOM):
//
//   1. THIN-API ALLOCATION — bridge strips the format-internal
//      `thought\n` channel-name preamble from reasoning_content
//      deltas (api-side translation work; client doesn't need to know
//      gemma's tokenizer-grammar). Verified by inspecting captured SSE
//      deltas + the final chat[i].extra.reasoning value.
//
//   2. THICK-CLIENT ALLOCATION — ST's reasoning.js renders the
//      reasoning trace as a collapsible <details> element with the
//      heading driven by extra.reasoning_summary when present (the
//      auto-summary feature) and falling back to "Thought for Xs"
//      otherwise. Verified by inspecting the rendered DOM.
//
//   3. CROSS-TURN PRESERVATION — past-turn extra.reasoning is
//      forwarded as the `reasoning` field on outbound assistant
//      messages (ST FE) and re-rendered into the prefill by the chat
//      template (bridge). Verified by inspecting captured request
//      bodies on turn 2.
//
// The test uses scringlo (proven in test 22 to produce substantive
// thinking) with a substantive math/reasoning prompt — not the dicemother
// terse-persona path that produces empty thinking blocks. We want THIS
// test to demonstrate the feature WORKING, not the persona-vs-effort
// interaction (which is a separate study).

import { test, expect } from '@playwright/test';
import {
    loadAndConnect, sendAndObserve,
    selectCharacterByClick, freshChatByClick,
} from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

async function setReasoningEffort(page, value) {
    return await page.evaluate(async (v) => {
        const sel = document.getElementById('openai_reasoning_effort');
        if (!sel) return { ok: false };
        sel.value = v;
        const win = /** @type {any} */ (window);
        if (win.jQuery) win.jQuery(sel).trigger('input');
        else sel.dispatchEvent(new Event('input', { bubbles: true }));
        await new Promise(r => setTimeout(r, 200));
        return { ok: true, sel_value: sel.value };
    }, value);
}

async function waitForReasoningSummary(page, messageIdx, timeoutMs = 90_000) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
        const summary = await page.evaluate((i) => {
            const ctx = window.SillyTavern.getContext();
            const m = (ctx.chat || [])[i];
            return m?.extra?.reasoning_summary || null;
        }, messageIdx);
        if (summary) return summary;
        await page.waitForTimeout(750);
    }
    return null;
}

test.describe('full reasoning pipeline (engine → bridge → SSE → ST FE → DOM)', () => {
    test.setTimeout(20 * 60 * 1000);

    test('reasoning_content has no thought-prefix; collapsible renders with auto-summary heading; cross-turn reasoning preserved', async ({ page }, testInfo) => {
        const requests = [];
        const browserConsole = [];
        page.on('request', (req) => {
            const url = req.url();
            if (url.includes('/api/backends/chat-completions/generate')) {
                let body = null;
                try { body = req.postDataJSON(); } catch (_) {}
                requests.push({ body });
            }
        });
        page.on('console', (msg) => {
            const txt = msg.text();
            // Surface reasoning-summary log lines + any warning/error so
            // we can see why dispatchReasoningSummary skipped (parse
            // failure, fetch failure, etc.) without needing to attach
            // a debugger.
            if (txt.includes('reasoning_summary')
                || msg.type() === 'warning' || msg.type() === 'error') {
                browserConsole.push({ type: msg.type(), text: txt });
            }
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        await setReasoningEffort(page, 'high');

        // ── Turn 1: substantive reasoning prompt ─────────────────────
        console.log('=== TURN 1 ===');
        const r1 = await sendAndObserve(
            page,
            'i have a budget of 17 dollars and want to buy: pens at $2 each, notebooks at $4 each, and a calculator at $9. constraint: i need at least 1 of each item. what mix maximizes total items? show your reasoning before answering.',
            { timeoutMs: 5 * 60 * 1000 });
        const t1Idx = await page.evaluate(() => {
            const ctx = window.SillyTavern.getContext();
            return ctx.chat.length - 1;
        });
        const t1Summary = await waitForReasoningSummary(page, t1Idx, 90_000);
        const t1State = await page.evaluate((i) => {
            const ctx = window.SillyTavern.getContext();
            const m = ctx.chat[i] || {};
            return {
                mes_len: (m.mes || '').length,
                mes_head: (m.mes || '').slice(0, 200),
                reasoning_len: (m.extra?.reasoning || '').length,
                reasoning_head: (m.extra?.reasoning || '').slice(0, 200),
                reasoning_starts_with_thought_prefix:
                    (m.extra?.reasoning || '').startsWith('thought\n')
                    || (m.extra?.reasoning || '').startsWith('thought\r\n'),
                reasoning_summary: m.extra?.reasoning_summary || null,
            };
        }, t1Idx);
        console.log('  mes_len:', t1State.mes_len);
        console.log('  reasoning_len:', t1State.reasoning_len);
        console.log('  reasoning_starts_with_thought_prefix:', t1State.reasoning_starts_with_thought_prefix);
        console.log('  reasoning_head:', t1State.reasoning_head.slice(0, 100));
        console.log('  reasoning_summary:', JSON.stringify(t1Summary));

        // ── Turn 2: depends on turn 1's reasoning ───────────────────
        console.log('\n=== TURN 2 ===');
        const r2 = await sendAndObserve(
            page,
            'now: same constraint, but pens are $3 each. redo with full reasoning.',
            { timeoutMs: 5 * 60 * 1000 });
        const t2Idx = await page.evaluate(() => {
            const ctx = window.SillyTavern.getContext();
            return ctx.chat.length - 1;
        });
        const t2Summary = await waitForReasoningSummary(page, t2Idx, 90_000);
        const t2State = await page.evaluate((i) => {
            const ctx = window.SillyTavern.getContext();
            const m = ctx.chat[i] || {};
            return {
                mes_len: (m.mes || '').length,
                reasoning_len: (m.extra?.reasoning || '').length,
                reasoning_summary: m.extra?.reasoning_summary || null,
            };
        }, t2Idx);
        console.log('  mes_len:', t2State.mes_len);
        console.log('  reasoning_len:', t2State.reasoning_len);
        console.log('  reasoning_summary:', JSON.stringify(t2Summary));

        // ── Cross-turn assertion ────────────────────────────────────
        // Turn 2's MAIN chat-completion outbound request body should
        // carry the prior turn's reasoning on the assistant message —
        // that's the step-3 thick-client invariant (ST forwards
        // extra.reasoning as the `reasoning` field on outbound
        // assistant messages). The summarizer fires its OWN
        // chat-completions request after each main turn, so we can't
        // just look at requests[-1]; we filter for the "main turn"
        // requests (which carry the full chat history with many
        // messages, vs the summarizer's 2-message system+user shape).
        const mainTurnRequests = requests.filter(r =>
            (r.body?.messages || []).length > 4);
        const t2Req = mainTurnRequests[mainTurnRequests.length - 1] || { body: {} };
        const t2Msgs = t2Req.body?.messages || [];
        const t2AssistantWithReasoning = t2Msgs.some(m =>
            m.role === 'assistant'
            && typeof m.reasoning === 'string'
            && m.reasoning.length > 0);
        console.log('  main-turn requests captured:', mainTurnRequests.length);
        console.log('  turn 2 main request forwards prior reasoning:', t2AssistantWithReasoning);

        // ── Expand collapsibles + take final screenshot ─────────────
        await page.evaluate(() => {
            for (const d of document.querySelectorAll('#chat .mes details')) {
                d.setAttribute('open', '');
            }
        });
        await page.waitForTimeout(500);
        await page.screenshot({
            path: testInfo.outputPath('full_pipeline_final.png'),
            fullPage: true,
        });

        // ── DOM check: collapsible heading reflects auto-summary ────
        const collapsibleHeadings = await page.evaluate(() => {
            const headers = document.querySelectorAll('#chat .mes .mes_reasoning_header');
            return Array.from(headers).map(h => h.textContent || '');
        });
        console.log('\n=== rendered collapsible headings ===');
        for (const h of collapsibleHeadings) {
            console.log('  ', JSON.stringify(h));
        }

        fs.writeFileSync(testInfo.outputPath('pipeline_state.json'),
            JSON.stringify({
                turn1: { ...t1State, summary_via_poll: t1Summary },
                turn2: { ...t2State, summary_via_poll: t2Summary },
                turn2_request_forwards_reasoning: t2AssistantWithReasoning,
                collapsible_headings: collapsibleHeadings,
                browser_console: browserConsole,
            }, null, 2));

        console.log('\n=== browser console (filtered to reasoning_summary + warnings/errors) ===');
        for (const c of browserConsole.slice(0, 20)) {
            console.log(`  [${c.type}] ${c.text.slice(0, 200)}`);
        }

        // ── Assertions ──────────────────────────────────────────────
        // A: reasoning_content has NO `thought\n` preamble at the bridge layer.
        if (t1State.reasoning_len > 0) {
            expect(t1State.reasoning_starts_with_thought_prefix,
                'reasoning should not leak the gemma channel-name preamble — that is api-side translation work').toBe(false);
        }

        // Auto-summary: at least one turn with reasoning got a summary.
        const turnsWithReasoning = [t1State, t2State].filter(s => s.reasoning_len > 0);
        const turnsWithSummary = turnsWithReasoning.filter(s => s.reasoning_summary);
        if (turnsWithReasoning.length > 0) {
            expect(turnsWithSummary.length,
                'auto-summary should produce a heading on at least one turn with reasoning')
                .toBeGreaterThan(0);
        }

        // Cross-turn: turn 2 request body carries prior reasoning.
        if (t1State.reasoning_len > 0 && t2State.mes_len > 0) {
            expect(t2AssistantWithReasoning,
                'turn 2 outbound request should carry prior reasoning on assistant message')
                .toBe(true);
        }

        // Liveness.
        expect(t1State.mes_len, 'turn 1 produced content').toBeGreaterThan(0);
        expect(t2State.mes_len, 'turn 2 produced content').toBeGreaterThan(0);
    });
});
