import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import { loadHarness, naturalElicit } from './_helpers/natural_elicit.mjs';

// Honest end-to-end test for the render-visual card.
//
// Differs from test 17 in the ONE place that matters: the model
// decides whether to call the tool. No ToolManager.invokeFunctionTool
// shim, no synthetic ctx.chat.push, no window.__* result stash.
//
// User asks scringlo "draw me a voronoi diagram with 12 seeds." Test
// observes:
//   - did the model emit a render-visual__render tool call? (with
//     what args?)
//   - did the tool actually fire and return a result?
//   - did the assistant's wrapping turn make sense?
//
// The test does NOT assert the tool MUST fire — that's the point of
// the elicitation reliability harness. This test asserts that WHEN
// the tool fires, the data path is sound. If the model declines to
// emit the tool_call, we capture that as a different finishState
// and the assertion is just "we got a sensible response."

test.use({ video: 'on' });

test.describe('honest render-visual elicitation — Scringlo + voronoi request', () => {
    test.setTimeout(300_000);

    test('user asks for voronoi; model decides whether to invoke render-visual', async ({ page }, testInfo) => {
        // Bare-minimum setup: load page, connect API. No programmatic
        // /newchat, no programmatic selectCharacterById. We drive ST's
        // actual UI to keep its prompt-building flow undisturbed.
        // Whichever character is loaded by default is what we test
        // against; if it's Seraphina rather than Scringlo, the
        // elicitation tests still work because the tool descriptions
        // do the heavy lifting (and the question of "does
        // persona affect elicitation" is a separate study).
        await loadHarness(page);

        await page.screenshot({
            path: testInfo.outputPath('checkpoint_1_chat_ready.png'),
        });

        const record = await naturalElicit(
            page,
            "draw me a voronoi diagram!! 12 seeds scattered, each one's territory in a different color ✨",
            { timeoutMs: 240_000 },
        );

        await page.evaluate(() => {
            for (const d of document.querySelectorAll(
                '#chat .mes details.custom-tool-progress-collapsible')) {
                d.setAttribute('open', '');
            }
        });
        await page.waitForTimeout(500);
        await page.screenshot({
            path: testInfo.outputPath('checkpoint_2_after_response.png'),
        });

        // Save the elicitation record as a curated artifact.
        fs.writeFileSync(
            testInfo.outputPath('elicitation_record.json'),
            JSON.stringify(record, null, 2),
        );

        console.log(`[honest] finishState=${record.finishState} ` +
            `elapsedMs=${Math.round(record.elapsedMs)} ` +
            `toolCallsEmitted=${record.toolCallsEmitted.length} ` +
            `toolProgress=${record.toolProgress.length}`);
        for (const tp of record.toolProgress) {
            const summarySnippet = (tp.summary || '').slice(0, 100);
            console.log(`  tool_progress: ${tp.label} — status=${tp.status} summary=${summarySnippet}`);
        }
        for (const tc of record.toolCallsEmitted) {
            console.log(`  tool_invocation: ${tc.name} args=${tc.parameters_raw}`);
        }

        // Assertions:
        // 1. The model produced SOMETHING (text, tool_call, or both)
        expect(['completed', 'tool_handled'], 'finishState is sensible')
            .toContain(record.finishState);

        // 2. If a tool was handled, it was render-visual specifically
        //    (not some other random tool). If multiple tools were
        //    handled, render-visual must be among them.
        if (record.finishState === 'tool_handled') {
            const labels = record.toolProgress.map(tp => tp.label || '').join(' | ');
            const names = record.toolCallsEmitted.map(tc => tc.name || '').join(' | ');
            const labelsLo = labels.toLowerCase();
            const namesLo = names.toLowerCase();
            const wantedFired =
                labelsLo.includes('render visual') ||
                labelsLo.includes('render-visual') ||
                namesLo.includes('render-visual');
            expect(wantedFired,
                `render-visual fired (saw labels=${labels} names=${names})`)
                .toBeTruthy();

            // Tool reached terminal state
            const TERMINAL = new Set(['done', 'failed', 'cancelled']);
            for (const tp of record.toolProgress) {
                expect(TERMINAL.has(tp.status),
                    `tool ${tp.label} reached terminal status (got ${tp.status})`)
                    .toBe(true);
            }
        }

        // 3. Bounded wall time even when the model declines to use
        //    a tool. A pure-text response shouldn't take >60s.
        if (record.finishState === 'completed') {
            expect(record.elapsedMs, 'pure-text response under 60s')
                .toBeLessThan(60_000);
        }

        // 4. We don't assert specifically that the tool MUST be
        //    called — that's elicitation reliability work, run as
        //    a separate study (test 20).
    });
});
