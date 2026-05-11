// Demonstrate that reasoning_effort actually GATES reasoning emission.
//
// Two chat sessions, single Playwright video, identical user prompt.
// Only the reasoning_effort setting differs. Assertions are
// proportionate: chat A must produce thinking tokens (extra.reasoning
// populated + mes_reasoning_details visible in DOM + SSE has
// reasoning_content deltas); chat B must produce ZERO of those.
//
// Why "auto" disables: ST's getReasoningEffort() in public/scripts/
// openai.js#2546 returns `undefined` when settings.reasoning_effort
// === 'auto' for the CUSTOM source, which causes the field to be
// omitted from the chat-completions request body. The bridge's
// server/bridge.py line 948-949 then sees `re_raw = None` and sets
// `enable_thinking = False`. Engine: thought-channel tokens are not
// routed to thinkingQueue. SSE: no reasoning_content deltas.
//
// This is the cheapest way to disable thinking through the UI today;
// there's no explicit "off" entry in reasoning_effort_types (which is
// arguably a UX gap but not a feature gap — the binary CAN be set).

import { test, expect } from '@playwright/test';
import { loadAndConnect,
    selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

// Set the reasoning_effort dropdown to a given value via the actual UI control.
async function setReasoningEffortUI(page, value) {
    return await page.evaluate(async (v) => {
        const sel = document.getElementById('openai_reasoning_effort');
        if (!sel) throw new Error('#openai_reasoning_effort not found');
        sel.value = v;
        const win = /** @type {any} */ (window);
        if (win.jQuery) win.jQuery(sel).trigger('input');
        else sel.dispatchEvent(new Event('input', { bubbles: true }));
        await new Promise(r => setTimeout(r, 200));
        return sel.value;
    }, value);
}

// Send a prompt that mildly invites thinking but doesn't require it.
// We want the model to think when allowed and not think when blocked.
const USER_PROMPT = 'what is 23 plus 17?';

async function sendAndSettle(page, prompt, timeoutMs = 90_000) {
    await page.locator('#send_textarea').fill(prompt);
    await page.locator('#send_but').click();
    const settleEnd = Date.now() + timeoutMs;
    while (Date.now() < settleEnd) {
        const generating = await page.evaluate(() => {
            const s = document.querySelector('#mes_stop');
            return s && s.offsetParent !== null;
        });
        if (!generating) break;
        await page.waitForTimeout(400);
    }
    await page.waitForTimeout(1500);
}

// Probe the chat state for everything reasoning-related: chat[] data,
// rendered DOM, request bodies, SSE chunks.
async function probeReasoning(page) {
    // Open all collapsibles so DOM probe sees their content.
    await page.evaluate(() => {
        for (const d of document.querySelectorAll('#chat details')) d.setAttribute('open', '');
    });
    await page.waitForTimeout(300);
    return await page.evaluate(() => {
        const ctx = window.SillyTavern.getContext();
        const chat = ctx.chat || [];
        const entries = chat.map((m, i) => ({
            idx: i, is_user: !!m.is_user, is_system: !!m.is_system,
            mes_first_200: (m.mes || '').slice(0, 200),
            has_extra_reasoning: typeof m.extra?.reasoning === 'string' && m.extra.reasoning.length > 0,
            extra_reasoning_first_300: typeof m.extra?.reasoning === 'string'
                ? m.extra.reasoning.slice(0, 300) : null,
            extra_reasoning_duration_ms: m.extra?.reasoning_duration ?? null,
        }));
        const reasoningDom = Array.from(document.querySelectorAll('#chat details.mes_reasoning_details'))
            .map(d => ({
                visible: d.offsetHeight > 0 && getComputedStyle(d).display !== 'none',
                summary: (d.querySelector(':scope > summary')?.innerText || '').trim(),
                body_first_200: (() => {
                    const body = Array.from(d.childNodes).filter(n => n.nodeType === 1 && n.nodeName !== 'SUMMARY');
                    return body.map(n => n.innerText || n.textContent || '').join(' ').slice(0, 200);
                })(),
            }));
        return { entries, reasoningDom };
    });
}

test.describe('reasoning_effort gates thinking emission', () => {
    test.setTimeout(8 * 60 * 1000);

    test('chat A (high) produces thinking, chat B (auto) produces none — single recording', async ({ page }, testInfo) => {
        // Capture every SSE chunk + chat-completions request body for forensic record.
        const sseChunks = [];
        const requestBodies = [];
        page.on('request', (req) => {
            const url = req.url();
            if (!url.includes('/chat-completions/generate') &&
                !url.includes('/v1/chat/completions')) return;
            let body = null;
            try { body = req.postDataJSON(); } catch { body = req.postData()?.slice(0, 2000); }
            requestBodies.push({ t: Date.now(), url, body });
        });
        page.on('response', async (resp) => {
            const ct = resp.headers()['content-type'] || '';
            if (!ct.includes('text/event-stream')) return;
            try {
                const text = await resp.text();
                for (const rec of text.split('\n\n')) {
                    if (rec.trim()) sseChunks.push({ t: Date.now(), raw: rec });
                }
            } catch { /* response body unavailable */ }
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');

        // ── CHAT A: reasoning_effort=high ────────────────────────────
        await freshChatByClick(page);
        const sseBeforeA = sseChunks.length;
        const reqBeforeA = requestBodies.length;
        const effortA = await setReasoningEffortUI(page, 'high');
        expect(effortA, 'reasoning_effort set to high').toBe('high');
        await page.screenshot({ path: testInfo.outputPath('A_01_set_high.png'), fullPage: true });
        await sendAndSettle(page, USER_PROMPT);
        const stateA = await probeReasoning(page);
        await page.screenshot({ path: testInfo.outputPath('A_99_chat_with_thinking.png'), fullPage: true });

        const sseA = sseChunks.slice(sseBeforeA).map(c => c.raw).join('\n');
        const reqsA = requestBodies.slice(reqBeforeA);
        const sentReasoningEffortA = reqsA.find(r => r.body && r.body.reasoning_effort)?.body?.reasoning_effort;
        const sawReasoningContentDeltaA = sseA.includes('"reasoning_content"');
        const reasoningChatEntriesA = stateA.entries.filter(e => !e.is_user && !e.is_system && e.has_extra_reasoning);
        const visibleReasoningDomA = stateA.reasoningDom.filter(d => d.visible);

        console.log('=== CHAT A (reasoning_effort=high) ===');
        console.log(`  sent in request body: reasoning_effort=${sentReasoningEffortA}`);
        console.log(`  SSE contained reasoning_content delta: ${sawReasoningContentDeltaA}`);
        console.log(`  chat[] entries with extra.reasoning: ${reasoningChatEntriesA.length}`);
        console.log(`  visible mes_reasoning_details DOM nodes: ${visibleReasoningDomA.length}`);
        for (const d of visibleReasoningDomA) {
            console.log(`    - summary=${d.summary.slice(0, 80)} body=${d.body_first_200.slice(0, 80)}`);
        }

        // ── CHAT B: reasoning_effort=auto (sends NOTHING to bridge → enable_thinking=False) ──
        await freshChatByClick(page);
        const sseBeforeB = sseChunks.length;
        const reqBeforeB = requestBodies.length;
        const effortB = await setReasoningEffortUI(page, 'auto');
        expect(effortB, 'reasoning_effort set to auto').toBe('auto');
        await page.screenshot({ path: testInfo.outputPath('B_01_set_auto.png'), fullPage: true });
        await sendAndSettle(page, USER_PROMPT);
        const stateB = await probeReasoning(page);
        await page.screenshot({ path: testInfo.outputPath('B_99_chat_without_thinking.png'), fullPage: true });

        const sseB = sseChunks.slice(sseBeforeB).map(c => c.raw).join('\n');
        const reqsB = requestBodies.slice(reqBeforeB);
        const sentReasoningEffortB = reqsB.find(r => r.body && 'reasoning_effort' in r.body)?.body?.reasoning_effort;
        const sawReasoningContentDeltaB = sseB.includes('"reasoning_content"');
        const reasoningChatEntriesB = stateB.entries.filter(e => !e.is_user && !e.is_system && e.has_extra_reasoning);
        const visibleReasoningDomB = stateB.reasoningDom.filter(d => d.visible);

        console.log('=== CHAT B (reasoning_effort=auto) ===');
        console.log(`  sent in request body: reasoning_effort=${sentReasoningEffortB} (undefined means field omitted)`);
        console.log(`  SSE contained reasoning_content delta: ${sawReasoningContentDeltaB}`);
        console.log(`  chat[] entries with extra.reasoning: ${reasoningChatEntriesB.length}`);
        console.log(`  visible mes_reasoning_details DOM nodes: ${visibleReasoningDomB.length}`);

        fs.writeFileSync(testInfo.outputPath('chat_A_state.json'),
            JSON.stringify({ sentReasoningEffort: sentReasoningEffortA, sawReasoningContentDelta: sawReasoningContentDeltaA, ...stateA }, null, 2));
        fs.writeFileSync(testInfo.outputPath('chat_B_state.json'),
            JSON.stringify({ sentReasoningEffort: sentReasoningEffortB, sawReasoningContentDelta: sawReasoningContentDeltaB, ...stateB }, null, 2));
        fs.writeFileSync(testInfo.outputPath('sse_chunks.jsonl'),
            sseChunks.map(c => JSON.stringify(c)).join('\n'));
        fs.writeFileSync(testInfo.outputPath('request_bodies.jsonl'),
            requestBodies.map(c => JSON.stringify(c)).join('\n'));

        // ── ASSERTIONS ────────────────────────────────────────────────
        // Chat A: thinking ON
        expect(sentReasoningEffortA, 'chat A request body carried reasoning_effort=high').toBe('high');
        expect(sawReasoningContentDeltaA, 'chat A SSE included reasoning_content deltas').toBe(true);
        expect(reasoningChatEntriesA.length, 'chat A has assistant turn(s) with extra.reasoning populated').toBeGreaterThan(0);
        expect(visibleReasoningDomA.length, 'chat A renders at least one mes_reasoning_details collapsible').toBeGreaterThan(0);

        // Chat B: thinking OFF
        expect(sentReasoningEffortB, 'chat B request body omitted reasoning_effort (auto → undefined)').toBeUndefined();
        expect(sawReasoningContentDeltaB, 'chat B SSE has NO reasoning_content deltas').toBe(false);
        expect(reasoningChatEntriesB.length, 'chat B has ZERO assistant turns with extra.reasoning').toBe(0);
        expect(visibleReasoningDomB.length, 'chat B has ZERO mes_reasoning_details collapsibles visible').toBe(0);
    });
});
