// Total residue invariant.
//
// HARD CONTRACT (forensic / multi-agent RL):
//   Every tool invocation that reaches the FE plugin leaves a
//   permanent rendered residue in the chat DOM. There is no shape of
//   backend output — empty object, null, empty string, empty array,
//   failure, cancellation, network giveup, 404-then-reconcile — that
//   produces silent loss.
//
//   Forensic infrastructure: every action an agent took must produce
//   a record. "The tool produced nothing" is itself information; it
//   must render as such, not be silently swallowed.
//
// This file uses the test-invariant-coverage toolcard, which ships a
// tool for each pathological result shape. Each test invokes via
// ctx.ToolManager.invokeFunctionTool (the REAL FE pipeline) and
// asserts the residue appears in the rendered DOM.
//
// Also asserts STRUCTURAL TERMINATION: every invocation finishes
// within bounds (no "polling forever" — the FE poll loop is bounded
// by 5x ~22s retry budget OR the card's idle_timeout_s, whichever
// fires first; for this test card idle_timeout_s=30s).

import { test, expect } from '@playwright/test';
import { loadAndConnect,
    selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

async function invokeAndProbe(page, qualifiedName, params) {
    return await page.evaluate(async ({ qualifiedName, params }) => {
        const ctx = window.SillyTavern.getContext();
        // Ensure an assistant turn exists for tool_progress attachment.
        let scaffoldIdx = -1;
        for (let i = ctx.chat.length - 1; i >= 0; i--) {
            if (!ctx.chat[i].is_user && !ctx.chat[i].is_system) {
                scaffoldIdx = i; break;
            }
        }
        if (scaffoldIdx < 0) {
            const fake = {
                name: ctx.characters?.[ctx.characterId]?.name || 'Assistant',
                is_user: false, is_system: false,
                send_date: new Date().toLocaleString(),
                mes: '(scaffold)', extra: {},
            };
            ctx.chat.push(fake);
            ctx.addOneMessage(fake);
            scaffoldIdx = ctx.chat.length - 1;
            await new Promise(r => setTimeout(r, 100));
        }

        const t0 = performance.now();
        let backendResult, backendThrew = null;
        try {
            backendResult = await ctx.ToolManager.invokeFunctionTool(
                qualifiedName, JSON.stringify(params));
        } catch (e) {
            backendThrew = String(e?.message || e);
        }
        const elapsed = performance.now() - t0;

        const startedFloor = Date.now() - (elapsed + 2000);
        const matchedEntries = [];
        for (let i = 0; i < ctx.chat.length; i++) {
            const msg = ctx.chat[i];
            for (const e of (msg.extra?.tool_progress || [])) {
                if (typeof e.started_at === 'number' && e.started_at >= startedFloor) {
                    matchedEntries.push({ messageIdx: i, entry: e });
                }
            }
        }
        for (const d of document.querySelectorAll('details.custom-tool-progress-collapsible')) {
            d.setAttribute('open', '');
        }
        const messageIndices = [...new Set(matchedEntries.map(m => m.messageIdx))];
        const domByMessage = messageIndices.map(idx => {
            const bubbles = document.querySelectorAll('#chat .mes');
            const bubble = bubbles[idx];
            if (!bubble) return { idx, bodies: [] };
            return {
                idx,
                bodies: Array.from(bubble.querySelectorAll('details.custom-tool-progress-collapsible')).map(d => {
                    const sum = d.querySelector(':scope > summary');
                    const body = Array.from(d.childNodes).filter(n => n.nodeType === 1 && n.nodeName !== 'SUMMARY');
                    return {
                        headerText: (sum?.innerText || '').trim(),
                        bodyText: body.map(n => n.innerText || n.textContent || '').join(' ').trim(),
                    };
                }),
            };
        });
        const allBodyText = domByMessage.flatMap(d => d.bodies.map(b => b.bodyText)).join('\n---\n');
        const terminalStatuses = matchedEntries.map(m => m.entry.status);

        return {
            qualifiedName, params,
            elapsed_ms: Math.round(elapsed),
            backend_result_string: backendResult,
            backend_threw: backendThrew,
            terminal_statuses: terminalStatuses,
            tool_progress_entries: matchedEntries.map(m => ({
                label: m.entry.label,
                status: m.entry.status,
                duration_ms: m.entry.duration_ms,
                summary_first_300: typeof m.entry.summary === 'string' ? m.entry.summary.slice(0, 300) : null,
                error: typeof m.entry.error === 'string' ? m.entry.error : null,
            })),
            dom_concat_body_text: allBodyText,
        };
    }, { qualifiedName, params });
}

function assertResidueInvariants(obs, label, expectedSubstrings) {
    // INVARIANT 1: termination — every entry reached a terminal status.
    expect(obs.terminal_statuses.length, `${label}: at least one entry exists`).toBeGreaterThan(0);
    for (const s of obs.terminal_statuses) {
        expect(['done', 'failed', 'cancelled'],
            `${label}: status is terminal (got "${s}")`).toContain(s);
    }
    // INVARIANT 2: residue — DOM body is non-empty.
    expect(obs.dom_concat_body_text.length,
        `${label}: DOM body has non-empty content`).toBeGreaterThan(0);
    // INVARIANT 3: per-test expected substrings appear in the residue.
    for (const sub of expectedSubstrings) {
        expect(obs.dom_concat_body_text,
            `${label}: residue contains "${sub}"`).toContain(sub);
    }
}

test.describe('HARD CONTRACT: every tool invocation leaves visible residue', () => {
    test.setTimeout(8 * 60 * 1000);

    test('shape: ok with {} (empty object) → "(tool returned empty object)" residue', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        const obs = await invokeAndProbe(page, 'test-invariant-coverage__empty_object', {});
        await page.waitForTimeout(200);
        await page.screenshot({ path: testInfo.outputPath('empty_object.png'), fullPage: true });
        fs.writeFileSync(testInfo.outputPath('empty_object.json'), JSON.stringify(obs, null, 2));
        console.log('[empty_object]', JSON.stringify(obs.terminal_statuses), obs.dom_concat_body_text.slice(0, 200));
        assertResidueInvariants(obs, 'empty_object', ['tool returned empty object']);
        // Termination bound: this is instant (no bridge call). <2s tops.
        expect(obs.elapsed_ms).toBeLessThan(5000);
    });

    test('shape: ok with null → "(tool returned null)" residue', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        const obs = await invokeAndProbe(page, 'test-invariant-coverage__null_result', {});
        await page.waitForTimeout(200);
        await page.screenshot({ path: testInfo.outputPath('null_result.png'), fullPage: true });
        fs.writeFileSync(testInfo.outputPath('null_result.json'), JSON.stringify(obs, null, 2));
        console.log('[null_result]', obs.dom_concat_body_text.slice(0, 200));
        assertResidueInvariants(obs, 'null_result', ['tool returned null']);
        expect(obs.elapsed_ms).toBeLessThan(5000);
    });

    test('shape: ok with "" (empty string) → "(tool returned empty string)" residue', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        const obs = await invokeAndProbe(page, 'test-invariant-coverage__empty_string', {});
        await page.waitForTimeout(200);
        await page.screenshot({ path: testInfo.outputPath('empty_string.png'), fullPage: true });
        fs.writeFileSync(testInfo.outputPath('empty_string.json'), JSON.stringify(obs, null, 2));
        console.log('[empty_string]', obs.dom_concat_body_text.slice(0, 200));
        assertResidueInvariants(obs, 'empty_string', ['tool returned empty string']);
        expect(obs.elapsed_ms).toBeLessThan(5000);
    });

    test('shape: ok with [] (empty array) → "(tool returned empty array)" residue', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        const obs = await invokeAndProbe(page, 'test-invariant-coverage__empty_array', {});
        await page.waitForTimeout(200);
        await page.screenshot({ path: testInfo.outputPath('empty_array.png'), fullPage: true });
        fs.writeFileSync(testInfo.outputPath('empty_array.json'), JSON.stringify(obs, null, 2));
        console.log('[empty_array]', obs.dom_concat_body_text.slice(0, 200));
        assertResidueInvariants(obs, 'empty_array', ['tool returned empty array']);
        expect(obs.elapsed_ms).toBeLessThan(5000);
    });

    test('shape: ok:false with error → error string in residue', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        const obs = await invokeAndProbe(page, 'test-invariant-coverage__always_fail', {});
        await page.waitForTimeout(200);
        await page.screenshot({ path: testInfo.outputPath('always_fail.png'), fullPage: true });
        fs.writeFileSync(testInfo.outputPath('always_fail.json'), JSON.stringify(obs, null, 2));
        console.log('[always_fail]', obs.dom_concat_body_text.slice(0, 200));
        // For failure path, status must be 'failed' specifically.
        expect(obs.terminal_statuses).toContain('failed');
        assertResidueInvariants(obs, 'always_fail',
            ['intentional failure for test 28 invariant coverage']);
    });

    test('slow_then_succeed: terminates within bound, residue contains success string', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        const t0 = Date.now();
        const obs = await invokeAndProbe(page, 'test-invariant-coverage__slow_then_succeed', { sleep_s: 2 });
        const wallElapsed = Date.now() - t0;
        await page.waitForTimeout(200);
        await page.screenshot({ path: testInfo.outputPath('slow_succeed.png'), fullPage: true });
        fs.writeFileSync(testInfo.outputPath('slow_succeed.json'), JSON.stringify(obs, null, 2));
        console.log('[slow_succeed]', wallElapsed, 'ms wall;', obs.dom_concat_body_text.slice(0, 200));
        // Termination bound: 2s sleep + overhead, < 10s comfortably.
        expect(wallElapsed).toBeLessThan(10_000);
        assertResidueInvariants(obs, 'slow_succeed', ['slept 2.0s']);
    });

    test('cancellation mid-flight: terminates immediately, residue contains cancellation reason', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);

        // Kick off a 30s sleep, then call cancel after 500ms. The
        // entry must reach 'failed' status with 'cancelled by user'
        // as the error, well under the 30s tool duration.
        const obs = await page.evaluate(async () => {
            const ctx = window.SillyTavern.getContext();
            // Scaffold turn
            const fake = {
                name: ctx.characters?.[ctx.characterId]?.name || 'Assistant',
                is_user: false, is_system: false,
                send_date: new Date().toLocaleString(),
                mes: '(scaffold for cancel test)', extra: {},
            };
            ctx.chat.push(fake);
            ctx.addOneMessage(fake);

            const t0 = performance.now();
            // Kick off the invocation without awaiting.
            const invokePromise = ctx.ToolManager.invokeFunctionTool(
                'test-invariant-coverage__slow_then_succeed',
                JSON.stringify({ sleep_s: 30 }),
            );

            // Find the session_id from the live tool_progress entry —
            // the FE plugin attaches it to the entry on start.
            let sessionId = null;
            for (let i = 0; i < 30; i++) {
                await new Promise(r => setTimeout(r, 100));
                for (let j = ctx.chat.length - 1; j >= 0; j--) {
                    const entries = ctx.chat[j].extra?.tool_progress || [];
                    for (const e of entries) {
                        if ((e.label || '').includes('Sleeps') && e.session_id) {
                            sessionId = e.session_id; break;
                        }
                    }
                    if (sessionId) break;
                }
                if (sessionId) break;
            }
            if (!sessionId) return { error: 'session_id never appeared on tool_progress entry' };

            // Cancel.
            await fetch(`/api/plugins/toolcards/cancel/${sessionId}`, { method: 'POST' });

            // Await the invocation to finish (cancellation propagates).
            const backendResult = await invokePromise.catch(e => String(e?.message || e));
            const elapsed = performance.now() - t0;

            // Find the terminal entry.
            const finals = [];
            for (let j = 0; j < ctx.chat.length; j++) {
                for (const e of (ctx.chat[j].extra?.tool_progress || [])) {
                    if (e.session_id === sessionId || (e.label || '').includes('Sleeps')) {
                        finals.push(e);
                    }
                }
            }
            for (const d of document.querySelectorAll('details.custom-tool-progress-collapsible')) {
                d.setAttribute('open', '');
            }
            const bubbles = document.querySelectorAll('#chat .mes');
            const lastBubble = bubbles[bubbles.length - 1];
            const bodies = lastBubble
                ? Array.from(lastBubble.querySelectorAll('details.custom-tool-progress-collapsible')).map(d => {
                    const sum = d.querySelector(':scope > summary');
                    const body = Array.from(d.childNodes).filter(n => n.nodeType === 1 && n.nodeName !== 'SUMMARY');
                    return {
                        headerText: (sum?.innerText || '').trim(),
                        bodyText: body.map(n => n.innerText || n.textContent || '').join(' ').trim(),
                    };
                })
                : [];
            return {
                elapsed_ms: Math.round(elapsed),
                sessionId,
                backend_result_string: backendResult,
                final_entries: finals.map(e => ({
                    status: e.status, error: e.error, duration_ms: e.duration_ms,
                })),
                dom_bodies: bodies,
            };
        });

        await page.waitForTimeout(200);
        await page.screenshot({ path: testInfo.outputPath('cancellation.png'), fullPage: true });
        fs.writeFileSync(testInfo.outputPath('cancellation.json'), JSON.stringify(obs, null, 2));
        console.log('[cancel]', JSON.stringify(obs, null, 2).slice(0, 400));

        // INVARIANTS:
        // 1. Cancellation terminates in well under tool's 30s sleep.
        expect(obs.elapsed_ms, 'cancellation propagates fast').toBeLessThan(15_000);
        // 2. The entry reached a terminal failure state.
        const terminal = obs.final_entries.some(e =>
            e.status === 'failed' || e.status === 'cancelled');
        expect(terminal, 'entry reached terminal status after cancel').toBe(true);
        // 3. The error string mentions cancellation.
        const cancelledIn = obs.final_entries.some(e =>
            (e.error || '').toLowerCase().includes('cancel'));
        expect(cancelledIn, 'cancellation reason stored on entry').toBe(true);
        // 4. The DOM body shows the cancellation residue.
        const domHasCancel = obs.dom_bodies.some(b =>
            b.bodyText.toLowerCase().includes('cancel'));
        expect(domHasCancel, 'cancellation residue visible in DOM').toBe(true);
    });
});
