// Proof-of-tool test: persona-effort-schema, natural elicitation, end-to-end.
//
// API CONTRACTS this test enforces (no model-talk; all about the tool):
//
//   1. Every tool invocation returns a result. ok:true with content or
//      ok:false with error — never null silently.
//   2. Every tool call has a descriptive summary visible in the
//      rendered chat surface.
//   3. Tools ship only after Playwright e2e proof of working end-to-end
//      natural invocation. No proof = not shipped.
//
// The bug pattern this test is designed to catch: a tool's invocation
// path appears to "work" in contrived test scenarios (direct
// ToolManager.invokeFunctionTool, or natural-language prompts that
// spoonfeed every arg verbatim) while ALWAYS failing in real
// natural-prompt usage. A test that vacuously passes when no
// invocation occurs is not a test — it's an obstacle to diagnosis.
//
// HOW THIS TEST IS DESIGNED TO FAIL CLEANLY:
//   - Sends a natural-language prompt that should elicit the tool.
//   - Retries up to MAX_RETRIES with follow-up nudges if the first
//     attempt doesn't elicit. Each retry is a new chat turn.
//   - On the FIRST attempt that elicits a tool_call (SSE has
//     tool_calls or gemma markers), asserts the result appears in
//     the rendered DOM via graph traversal.
//   - If MAX_RETRIES exhausted without elicitation: TEST FAILS
//     hard, NOT vacuously. The contract is "this tool works
//     end-to-end via natural prompt" — if it never elicits, that
//     contract is broken regardless of root cause.

import { test, expect } from '@playwright/test';
import { loadAndConnect,
    selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

const MAX_RETRIES = 4;
const PROMPTS = [
    'hey scringlo, what should your reasoning effort levels be?',
    'use the persona-effort-schema tool to elicit your effort levels.',
    'call the persona-effort-schema__elicit tool with no arguments. Just invoke it.',
    'Please invoke persona-effort-schema__elicit{} — no args needed, the tool reads context server-side.',
];

test.describe('persona-effort-schema: natural elicitation proof', () => {
    test.setTimeout(10 * 60 * 1000);

    test('elicit via natural prompt + assert rendered result residue', async ({ page }, testInfo) => {
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

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        await page.screenshot({ path: testInfo.outputPath('01_fresh.png'), fullPage: true });

        const attempts = [];
        let elicitedAttempt = -1;

        for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
            const prompt = PROMPTS[Math.min(attempt, PROMPTS.length - 1)];
            const sseBefore = sseChunks.length;
            const tSent = Date.now();
            await page.locator('#send_textarea').fill(prompt);
            await page.locator('#send_but').click();

            // Wait for generation to settle (no #mes_stop visible).
            const settleEnd = Date.now() + 90_000;
            while (Date.now() < settleEnd) {
                const generating = await page.evaluate(() => {
                    const s = document.querySelector('#mes_stop');
                    return s && s.offsetParent !== null;
                });
                if (!generating) break;
                await page.waitForTimeout(500);
            }
            // Wait a beat for any toolcards plugin tool_progress to land.
            await page.waitForTimeout(2000);

            const sseSliceJoined = sseChunks.slice(sseBefore).map(c => c.raw).join('\n');
            const sawOaiToolCalls = sseSliceJoined.includes('"tool_calls"');
            const sawGemmaMarker = sseSliceJoined.includes('<|tool_call>');
            const elicited = sawOaiToolCalls || sawGemmaMarker;

            await page.screenshot({
                path: testInfo.outputPath(`10_attempt${attempt}_${elicited ? 'elicited' : 'no-elicit'}.png`),
                fullPage: true,
            });

            attempts.push({
                attempt, prompt, elicited, sawOaiToolCalls, sawGemmaMarker,
                sse_chunks_this_turn: sseChunks.length - sseBefore,
                elapsed_ms: Date.now() - tSent,
            });
            console.log(`[attempt ${attempt}] prompt=${JSON.stringify(prompt.slice(0, 60))}`);
            console.log(`  elicited=${elicited} oai=${sawOaiToolCalls} gemma=${sawGemmaMarker} chunks=${sseChunks.length - sseBefore}`);

            if (elicited) {
                elicitedAttempt = attempt;
                break;
            }
        }

        // Open all collapsibles for final state capture.
        await page.evaluate(() => {
            for (const d of document.querySelectorAll('#chat details')) d.setAttribute('open', '');
        });
        await page.waitForTimeout(500);
        await page.screenshot({ path: testInfo.outputPath('99_final.png'), fullPage: true });

        // Rip the DOM graph + dump chat state.
        const state = await page.evaluate(() => {
            const ctx = window.SillyTavern.getContext();
            const chat = ctx.chat || [];
            const root = document.getElementById('chat');
            const nodes = [];
            const idForNode = new Map();
            let nextId = 0;
            function visit(el, parentId) {
                if (el.nodeType !== 1) return;
                const id = nextId++;
                idForNode.set(el, id);
                const style = getComputedStyle(el);
                const h = el.offsetHeight;
                const visible = h > 0 && style.display !== 'none' && style.visibility !== 'hidden';
                for (const c of el.children) visit(c, id);
                nodes.push({
                    id, parent_id: parentId,
                    tag: el.tagName.toLowerCase(),
                    classes: Array.from(el.classList),
                    text_first_500: (el.innerText || el.textContent || '').slice(0, 500),
                    visible, offset_height: h, display: style.display,
                });
            }
            if (root) visit(root, null);
            return {
                chat_entries: chat.map((m, i) => ({
                    idx: i, is_user: !!m.is_user, is_system: !!m.is_system,
                    name: m.name || '',
                    mes_first_500: (m.mes || '').slice(0, 500),
                    tool_progress_labels: (m.extra?.tool_progress || []).map(e =>
                        `${e.label} • ${e.status}` + (e.summary ? ' • SUMMARY:' + e.summary.slice(0, 200) : '')),
                    tool_invocations_names: (m.extra?.tool_invocations || []).map(i =>
                        `${i.displayName || i.name}: ${String(i.result || '').slice(0, 200)}`),
                })),
                dom_graph_nodes: nodes,
            };
        });

        const byId = new Map(state.dom_graph_nodes.map(n => [n.id, n]));
        function ancestorsVisible(n) {
            let c = n;
            while (c) {
                if (!c.visible) return false;
                if (c.parent_id == null) return true;
                c = byId.get(c.parent_id);
            }
            return true;
        }
        const toolResidueNodes = state.dom_graph_nodes.filter(n =>
            n.classes.some(c => c.includes('tool-progress-collapsible') ||
                                 c.includes('tool-invocations-collapsible') ||
                                 c.includes('tool-call-inline')) &&
            n.visible && ancestorsVisible(n)
        );
        const toolProgressWithSummary = state.chat_entries.flatMap(e =>
            (e.tool_progress_labels || []).filter(l => l.includes('SUMMARY:')));

        fs.writeFileSync(testInfo.outputPath('attempts.json'),
            JSON.stringify(attempts, null, 2));
        fs.writeFileSync(testInfo.outputPath('final_state.json'),
            JSON.stringify(state, null, 2));
        fs.writeFileSync(testInfo.outputPath('sse_chunks.jsonl'),
            sseChunks.map(c => JSON.stringify(c)).join('\n'));

        console.log(`[final] elicited_attempt=${elicitedAttempt}`);
        console.log(`[final] tool residue nodes (visible+ancestor-visible): ${toolResidueNodes.length}`);
        console.log(`[final] tool_progress entries with non-empty summary: ${toolProgressWithSummary.length}`);
        for (const r of toolResidueNodes) {
            console.log(`  - class=${r.classes} text=${r.text_first_500.slice(0, 100)}`);
        }

        // ── HARD CONTRACT ASSERTIONS ──────────────────────────────────
        //
        // (1) Elicitation: the tool MUST be invoked within MAX_RETRIES
        //     attempts via natural prompting. Vacuous-pass-on-no-elicit
        //     is not allowed; this is a proof-of-tool test, not a
        //     proof-of-elicitation-rate test.
        expect(elicitedAttempt,
            `persona-effort-schema was not elicited via natural prompting in ${MAX_RETRIES} attempts. ` +
            `Attempts: ${JSON.stringify(attempts.map(a => ({ attempt: a.attempt, elicited: a.elicited })))}`
        ).toBeGreaterThanOrEqual(0);

        // (2) Residue: a tool-residue node must be visible in the DOM.
        expect(toolResidueNodes.length,
            `Elicited on attempt ${elicitedAttempt} but ZERO tool-residue nodes are visible in the chat DOM. ` +
            `Contract: every tool call has a descriptive summary visible in the rendered chat surface.`
        ).toBeGreaterThan(0);

        // (3) Summary: at least one tool_progress entry must have a
        //     non-empty summary describing what the tool did.
        expect(toolProgressWithSummary.length,
            `Elicited and rendered, but no tool_progress entry has a non-empty summary describing what happened. ` +
            `Contract: every tool call has a descriptive summary.`
        ).toBeGreaterThan(0);
    });
});
