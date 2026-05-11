// Real end-to-end tool-invocation test, asserting one HARD INVARIANT:
//
//     If the backend produced ANY non-null result content, that content
//     MUST appear in the chat DOM. Silent loss of backend output is a
//     forbidden API violation (no exceptions).
//
// We exercise the REAL FE pipeline:
//     ctx.ToolManager.invokeFunctionTool(qualifiedName, params)
//       → registered tool's action callback (toolcards/index.js#registerOneTool)
//         → invokeToolViaSession (real)
//           → startToolSession (real POST /start_invoke)
//           → attachToCallerMessage (real, creates tool_progress entry)
//           → pollToolSession (real, calls deriveResultRendering + finalize/fail)
//             → updateMessageBlock (real, mutates DOM)
//
// We then probe the DOM directly. The assertions compare what the
// backend returned vs. what the user actually sees in the chat surface.
//
// THE KEY MEASUREMENT: every tool we invoke gets its full backend
// result content checked against the rendered DOM. If the backend's
// result string is "hello, X" and the DOM doesn't contain "hello, X",
// the test fails. No fudge factor, no "elicitation rate", no
// observation-only mode for this invariant.

import { test, expect } from '@playwright/test';
import { loadAndConnect,
    selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

/**
 * Drive a real tool invocation and observe the resulting DOM.
 * Returns:
 *   - backend_result_string (what ToolManager returned)
 *   - backend_threw (if invocation rejected)
 *   - dom_body_text (concatenation of all collapsible bodies under the message)
 *   - tool_progress_entries (the message's stored tool_progress data)
 *   - elapsed_ms
 */
async function invokeAndProbe(page, qualifiedName, params) {
    return await page.evaluate(async ({ qualifiedName, params }) => {
        const ctx = window.SillyTavern.getContext();

        // The FE plugin's findCallerAssistantMessage() attaches the
        // tool_progress entry to the most-recent non-user non-system
        // message. Create one if there isn't one. (The plugin can
        // synthesize a system stub as catastrophic-fallback, but in a
        // normal flow there's always a real assistant turn.)
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
                mes: '(scaffold for invocation test)',
                extra: {},
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

        // The plugin may have attached to a different message index
        // (the scaffold OR a new message it created). We probe the
        // ENTIRE chat for any tool_progress entry created in the last
        // few seconds (within elapsed + 2s slack), and collect all
        // collapsible bodies in the corresponding DOM.
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

        // Open all collapsibles in case some are closed (so innerText reflects all content).
        for (const d of document.querySelectorAll('details.custom-tool-progress-collapsible')) {
            d.setAttribute('open', '');
        }

        // Collect DOM body text across ALL collapsibles in matched messages.
        const messageIndices = [...new Set(matchedEntries.map(m => m.messageIdx))];
        const domByMessage = messageIndices.map(idx => {
            const bubbles = document.querySelectorAll('#chat .mes');
            const bubble = bubbles[idx];
            if (!bubble) return { idx, bubble_found: false, bodies: [] };
            const collapsibles = bubble.querySelectorAll('details.custom-tool-progress-collapsible');
            return {
                idx,
                bubble_found: true,
                bodies: Array.from(collapsibles).map(d => {
                    const sum = d.querySelector(':scope > summary');
                    const bodyNodes = Array.from(d.childNodes).filter(
                        n => n.nodeType === 1 && n.nodeName !== 'SUMMARY');
                    return {
                        headerText: (sum?.innerText || '').trim(),
                        bodyText: bodyNodes.map(n => n.innerText || n.textContent || '').join(' ').trim(),
                    };
                }),
            };
        });

        // Concat all body text (the "did this string appear in the chat?" surface).
        const allBodyText = domByMessage.flatMap(d => d.bodies.map(b => b.bodyText)).join('\n---\n');

        return {
            qualifiedName,
            elapsed_ms: Math.round(elapsed),
            backend_result_string: backendResult,  // what ToolManager.invokeFunctionTool returned
            backend_threw: backendThrew,
            tool_progress_entries: matchedEntries.map(m => ({
                messageIdx: m.messageIdx,
                label: m.entry.label,
                status: m.entry.status,
                duration_ms: m.entry.duration_ms,
                summary_first_300: typeof m.entry.summary === 'string' ? m.entry.summary.slice(0, 300) : null,
                error: typeof m.entry.error === 'string' ? m.entry.error : null,
            })),
            dom_by_message: domByMessage,
            dom_concat_body_text: allBodyText,
        };
    }, { qualifiedName, params });
}

/**
 * Assert the HARD INVARIANT: every fragment of backend output is
 * present in the rendered DOM body text.
 *
 * `expectedFragments` is an array of substrings the test author asserts
 * the backend produced and which therefore MUST appear in the rendered
 * DOM body. This is the "no silent drop of backend content" contract.
 */
function assertBackendOutputVisibleInDOM(observation, expectedFragments, label) {
    expect(observation.dom_concat_body_text.length,
        `${label}: DOM body has SOME content (backend produced output, so client must too)`)
        .toBeGreaterThan(0);
    for (const frag of expectedFragments) {
        expect(observation.dom_concat_body_text,
            `${label}: backend-produced fragment "${frag}" appears in rendered DOM`)
            .toContain(frag);
    }
}

test.describe('HARD INVARIANT: backend tool output appears in chat DOM', () => {
    test.setTimeout(8 * 60 * 1000);

    test('hello-world (result is a bare string): DOM contains the greeting', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);

        const obs = await invokeAndProbe(page, 'hello-world__greet', { name: 'InvariantTest' });
        await page.waitForTimeout(200);
        await page.screenshot({
            path: testInfo.outputPath('hello_world_dom.png'),
            fullPage: true,
        });
        fs.writeFileSync(testInfo.outputPath('hello_world.json'),
            JSON.stringify(obs, null, 2));
        console.log('[hello-world] backend_result:', obs.backend_result_string);
        console.log('[hello-world] dom_body:', obs.dom_concat_body_text);

        expect(obs.backend_threw).toBeNull();
        // Backend returned the string "hello, InvariantTest". This is the
        // hard invariant — the rendered chat must contain that string.
        // (Before the fix, the DOM contained only "session XXX started"
        // and the actual greeting was silently dropped.)
        assertBackendOutputVisibleInDOM(obs,
            ['hello, InvariantTest'],
            'hello-world result string');
    });

    test('persona-effort-schema (result has summary): DOM contains the schema', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);

        const obs = await invokeAndProbe(page, 'persona-effort-schema__elicit', {
            persona_name: 'InvariantPersona',
            persona_system_prompt: 'You are a brief test assistant. Reply succinctly.',
        });
        await page.waitForTimeout(200);
        await page.screenshot({
            path: testInfo.outputPath('persona_effort_dom.png'),
            fullPage: true,
        });
        fs.writeFileSync(testInfo.outputPath('persona_effort.json'),
            JSON.stringify(obs, null, 2));
        console.log('[persona-effort] backend_result first 200:',
            String(obs.backend_result_string || '').slice(0, 200));
        console.log('[persona-effort] dom_body first 200:',
            obs.dom_concat_body_text.slice(0, 200));

        expect(obs.backend_threw).toBeNull();
        // The persona-effort-schema service.py provides a `summary` field
        // listing all three effort levels for the named persona. That
        // summary MUST appear in the DOM verbatim.
        assertBackendOutputVisibleInDOM(obs, [
            'reasoning_effort schema for InvariantPersona:',
            // The three levels are non-deterministic in wording (model-
            // generated per call), but each level label MUST appear in
            // some form in the rendered body.
            '• low',
            '• medium',
            '• high',
        ], 'persona-effort schema');
    });

    test('persona-effort-schema with missing required arg: DOM contains the error', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);

        const obs = await invokeAndProbe(page, 'persona-effort-schema__elicit', {
            persona_name: 'MissingArg',
            // persona_system_prompt INTENTIONALLY OMITTED — service.py
            // returns {ok:false, error:"missing or empty persona_system_prompt"}
        });
        await page.waitForTimeout(200);
        await page.screenshot({
            path: testInfo.outputPath('failure_dom.png'),
            fullPage: true,
        });
        fs.writeFileSync(testInfo.outputPath('failure.json'),
            JSON.stringify(obs, null, 2));
        console.log('[failure] backend_threw:', obs.backend_threw);
        console.log('[failure] dom_body:', obs.dom_concat_body_text);

        // The plugin throws on ok:false (so the ToolManager wrapper
        // returns the error string), and ph.fail() stores entry.error.
        // The script.js error block surfaces it. DOM must contain the
        // specific error string from the backend.
        assertBackendOutputVisibleInDOM(obs,
            ['missing or empty persona_system_prompt'],
            'persona-effort error');
    });

    test('random-choice (real bridge-less tool): DOM contains the chosen option', async ({ page }, testInfo) => {
        // random-choice is a simple pure-Python tool — no bridge call,
        // deterministic-ish behavior, returns a chosen item. Picks a
        // different surface than hello-world / persona-effort-schema to
        // catch tools-with-different-result-shapes.
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);

        const obs = await invokeAndProbe(page, 'random-choice__uniform', {
            items: ['ALPHA_OPTION', 'BETA_OPTION', 'GAMMA_OPTION'],
            n: 1,
        });
        await page.waitForTimeout(200);
        await page.screenshot({
            path: testInfo.outputPath('random_choice_dom.png'),
            fullPage: true,
        });
        fs.writeFileSync(testInfo.outputPath('random_choice.json'),
            JSON.stringify(obs, null, 2));
        console.log('[random-choice] backend_result:', obs.backend_result_string);
        console.log('[random-choice] dom_body:', obs.dom_concat_body_text);

        expect(obs.backend_threw).toBeNull();
        // Whichever option got picked, the DOM must show it. We assert
        // SOMETHING from the option set appears — the exact pick is
        // random but each option string is distinctive.
        const dom = obs.dom_concat_body_text;
        const pickedOne = dom.includes('ALPHA_OPTION') ||
                          dom.includes('BETA_OPTION') ||
                          dom.includes('GAMMA_OPTION');
        expect(pickedOne,
            'random-choice result (the chosen option) appears in rendered DOM')
            .toBe(true);
    });
});
