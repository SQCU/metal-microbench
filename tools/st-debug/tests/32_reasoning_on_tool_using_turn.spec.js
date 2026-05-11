// Probe: does the model's native reasoning trace render on an
// assistant turn that ALSO emits a tool_call?
//
// Why this matters: there are two reasoning mechanisms in this fork:
//   (a) native — model emits <|channel>thought\n...<channel|>, bridge
//       captures, surfaces via OAI o1-shape delta.reasoning_content,
//       ST stores on chat[N].extra.reasoning and renders as a
//       collapsible <details class="mes_reasoning_details">.
//   (b) tool-wrapped — a toolcard (e.g. extended-thinking, now deleted)
//       makes a sub-LLM call and structures the result with
//       SUMMARY:/REASONING: markers, exposing the reasoning via the
//       tool result.
//
// We deleted (b) because it overlaps semantically with (a) and was
// using regex. This test verifies (a) actually works on the path
// that matters: a tool-using assistant turn at reasoning_effort=high
// must have BOTH the thinking trace AND the tool-call residue
// visible. Otherwise we lost a feature.

import { test, expect } from '@playwright/test';
import { loadAndConnect,
    selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

test.describe('native reasoning on tool-using turn', () => {
    test.setTimeout(8 * 60 * 1000);

    test('reasoning_effort=high + tool elicitation: both thinking AND tool residue render', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);

        // Set reasoning_effort=high via the UI dropdown.
        await page.evaluate(async () => {
            const sel = document.getElementById('openai_reasoning_effort');
            if (!sel) throw new Error('#openai_reasoning_effort not found');
            sel.value = 'high';
            const win = /** @type {any} */ (window);
            if (win.jQuery) win.jQuery(sel).trigger('input');
            else sel.dispatchEvent(new Event('input', { bubbles: true }));
            await new Promise(r => setTimeout(r, 200));
        });
        await page.screenshot({ path: testInfo.outputPath('01_effort_high.png'), fullPage: true });

        // Capture every SSE chunk for forensic record.
        const sseChunks = [];
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

        // Same elicitation pattern as test 31. Two-step: natural prompt
        // then nudge if the first try doesn't elicit. We need the
        // model to BOTH think AND call the tool.
        for (let attempt = 0; attempt < 2; attempt++) {
            const prompt = attempt === 0
                ? 'hey scringlo, what should your reasoning effort levels be?'
                : 'use the persona-effort-schema tool to elicit your effort levels.';
            await page.locator('#send_textarea').fill(prompt);
            await page.locator('#send_but').click();

            const settleEnd = Date.now() + 90_000;
            while (Date.now() < settleEnd) {
                const generating = await page.evaluate(() => {
                    const s = document.querySelector('#mes_stop');
                    return s && s.offsetParent !== null;
                });
                if (!generating) break;
                await page.waitForTimeout(500);
            }
            await page.waitForTimeout(2000);

            const sseJoined = sseChunks.map(c => c.raw).join('\n');
            const sawToolCall = sseJoined.includes('"tool_calls"') ||
                                sseJoined.includes('<|tool_call>');
            if (sawToolCall) break;
        }

        // Open all collapsibles for final screenshot + DOM probe.
        await page.evaluate(() => {
            for (const d of document.querySelectorAll('#chat details')) d.setAttribute('open', '');
        });
        await page.waitForTimeout(500);
        await page.screenshot({ path: testInfo.outputPath('99_final.png'), fullPage: true });

        // Probe chat[] + the DOM graph for reasoning content.
        const state = await page.evaluate(() => {
            const ctx = window.SillyTavern.getContext();
            const chat = ctx.chat || [];
            const entries = chat.map((m, i) => ({
                idx: i, is_user: !!m.is_user, is_system: !!m.is_system,
                name: m.name || '',
                mes_first_300: (m.mes || '').slice(0, 300),
                has_extra_reasoning: typeof m.extra?.reasoning === 'string' && m.extra.reasoning.length > 0,
                extra_reasoning_first_300: typeof m.extra?.reasoning === 'string' ? m.extra.reasoning.slice(0, 300) : null,
                extra_reasoning_duration: m.extra?.reasoning_duration ?? null,
                tool_progress_count: (m.extra?.tool_progress || []).length,
                tool_invocations_count: (m.extra?.tool_invocations || []).length,
            }));

            // Reasoning DOM presence: <details class="mes_reasoning_details">
            const reasoningDom = Array.from(document.querySelectorAll('#chat details.mes_reasoning_details'))
                .map(d => {
                    const style = getComputedStyle(d);
                    return {
                        visible: d.offsetHeight > 0 && style.display !== 'none',
                        offset_height: d.offsetHeight,
                        summary_text: (d.querySelector(':scope > summary')?.innerText || '').trim(),
                        body_text_first_300: (() => {
                            const body = Array.from(d.childNodes).filter(n => n.nodeType === 1 && n.nodeName !== 'SUMMARY');
                            return body.map(n => n.innerText || n.textContent || '').join(' ').slice(0, 300);
                        })(),
                    };
                });

            // Tool-call residue check (same as test 31)
            const toolResidueDom = Array.from(document.querySelectorAll(
                '#chat details.custom-tool-progress-collapsible, ' +
                '#chat details.custom-tool-invocations-collapsible, ' +
                '#chat details.custom-tool-call-inline'
            )).map(d => ({
                visible: d.offsetHeight > 0,
                classes: Array.from(d.classList),
                text_first_120: (d.innerText || '').slice(0, 120),
            }));

            return { entries, reasoningDom, toolResidueDom };
        });

        fs.writeFileSync(testInfo.outputPath('final_state.json'),
            JSON.stringify(state, null, 2));
        fs.writeFileSync(testInfo.outputPath('sse_chunks.jsonl'),
            sseChunks.map(c => JSON.stringify(c)).join('\n'));

        const reasoningChatEntries = state.entries.filter(e => !e.is_user && !e.is_system && e.has_extra_reasoning);
        const visibleReasoningDom = state.reasoningDom.filter(d => d.visible);
        const visibleToolResidueDom = state.toolResidueDom.filter(d => d.visible);

        console.log(`[Q1] chat[] entries with extra.reasoning: ${reasoningChatEntries.length}`);
        for (const e of reasoningChatEntries) {
            console.log(`  - chat[${e.idx}] reasoning duration=${e.extra_reasoning_duration}ms first 120: ${e.extra_reasoning_first_300.slice(0, 120)}`);
        }
        console.log(`[Q1] visible mes_reasoning_details DOM nodes: ${visibleReasoningDom.length}`);
        for (const d of visibleReasoningDom) {
            console.log(`  - summary=${d.summary_text} body first 80: ${d.body_text_first_300.slice(0, 80)}`);
        }
        console.log(`[Q1] visible tool-residue DOM nodes: ${visibleToolResidueDom.length}`);
        for (const d of visibleToolResidueDom) {
            console.log(`  - classes=${d.classes} text first 80: ${d.text_first_120.slice(0, 80)}`);
        }

        // ── Q1 ASSERTIONS ────────────────────────────────────────────
        // (1) At least one assistant turn must have extra.reasoning
        //     populated. reasoning_effort=high → engine emits thinking
        //     channel → bridge captures → ST stores. If this fails:
        //     reasoning is broken end-to-end, NOT just for tool turns.
        expect(reasoningChatEntries.length,
            'at least one assistant turn must have extra.reasoning populated (reasoning_effort=high)'
        ).toBeGreaterThan(0);

        // (2) At least one mes_reasoning_details collapsible must be
        //     visibly rendered. Storing reasoning in extra without
        //     rendering it is the same silent-loss pattern we've been
        //     killing all session.
        expect(visibleReasoningDom.length,
            'at least one mes_reasoning_details collapsible visibly rendered in DOM'
        ).toBeGreaterThan(0);

        // (3) Tool-call residue must also be visible. We're testing
        //     that reasoning AND tools coexist on the same surface;
        //     if tool residue is missing this turn just devolved to
        //     a non-tool reasoning chat which isn't the test we wrote.
        expect(visibleToolResidueDom.length,
            'tool-residue collapsibles visible (we asked for a tool-using turn)'
        ).toBeGreaterThan(0);
    });
});
