// Reproducible repro of the streamed-tool-call marker-leak bug.
//
// SYMPTOM (from a user report 2026-05-10):
// When `stream: true` is set on a `/v1/chat/completions` request and the
// model emits `<|tool_call>call:NAME{args}<tool_call|>` markers in its
// generated text, the bridge streams those raw bytes as `delta.content`
// (they're just generated tokens). At end-of-generation the bridge ALSO
// emits a structured `delta.tool_calls` chunk. ST's frontend accumulates
// both: it saves the marker text as the `mes.content` of one assistant
// turn AND creates a second assistant turn carrying the structured
// `tool_calls`. On render: the first turn displays as markdown (Python
// comments like `# Define budgets` become H1 headings); the second
// renders as the normal tool-progress collapsible. The chat ends up
// with a duplicate + visible engine scaffolding the user shouldn't see.
//
// CAUSE: the bridge's streaming path was simplified earlier today (in
// the cleanup that removed the rolling-buffer `<|tool_call` lookahead).
// The new path always streams content live and emits tool_calls at end.
// That's structurally correct for OAI-compat clients that strip
// markers from `content` when `tool_calls` arrives — ST's FE does not
// currently do that strip during streaming.
//
// PERSONA: dicemother (a TTRPG GM persona seeded by bootstrap.sh). She
// has a diegetic reason to call `python-exec` for encounter-table
// rolls, which reliably reproduces the multi-thousand-character
// tool_call argument that demonstrates the bug at full force.
//
// This spec does NOT assert the bug is fixed — it captures the
// observable state of the chat after a streamed tool_call response so
// a follow-up session has a stable Playwright artifact + chat.json + a
// rendered video of the failure to reason from.

import { test, expect } from '@playwright/test';
import {
    loadAndConnect, sendAndObserve,
    selectCharacterByClick, freshChatByClick,
} from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

test.describe('streamed tool_call marker leak — dicemother encounter', () => {
    test.setTimeout(10 * 60 * 1000);

    test('induce + capture: <|tool_call> markers leak into chat content', async ({ page }, testInfo) => {
        // Capture chat-completions request bodies + every SSE delta the
        // FE receives, so the artifact can be replayed offline.
        const networkRecords = [];
        page.on('request', (req) => {
            const url = req.url();
            if (url.includes('/api/backends/chat-completions/generate')) {
                let body = null;
                try { body = req.postDataJSON(); } catch (_) {}
                networkRecords.push({ kind: 'request', url, body });
            }
        });
        page.on('response', async (resp) => {
            const url = resp.url();
            if (url.includes('/api/backends/chat-completions/generate')) {
                // Don't try to read the streamed body here — Playwright's
                // Response.body() will block until the stream completes
                // and would deadlock the test. We just record the URL +
                // headers; the FE's accumulated chat[] gives us the
                // full picture after.
                try {
                    networkRecords.push({
                        kind: 'response_headers', url,
                        status: resp.status(),
                    });
                } catch (_) {}
            }
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'dicemother');
        await freshChatByClick(page);

        await page.screenshot({ path: testInfo.outputPath('00_dicemother_ready.png') });

        // The prompt frames an explicit need for the encounter table
        // tool call. dicemother's persona is structured so a TTRPG-shaped
        // input reliably triggers python-exec for encounter rolls.
        const r = await sendAndObserve(
            page,
            "i kick the door open and step into the next room. roll up what's there — give me an encounter, loot, and atmosphere. use the python tool to do the actual rolling so it isn't faked.",
            { timeoutMs: 5 * 60 * 1000 },
        );

        // Expand any tool-progress collapsibles so the screenshot
        // captures the bug surface in full.
        await page.evaluate(() => {
            for (const d of document.querySelectorAll(
                '#chat .mes details')) {
                d.setAttribute('open', '');
            }
        });
        await page.waitForTimeout(500);
        await page.screenshot({
            path: testInfo.outputPath('01_after_dicemother_response.png'),
            fullPage: true,
        });

        // Pull the saved chat from the FE side. The bug shows up as
        // either:
        //   (a) an assistant turn whose `.mes` contains literal
        //       `<|tool_call>call:` substrings (markers stream-leaked
        //       into content), OR
        //   (b) two adjacent assistant turns: one with the marker-text
        //       content and a second with `extra.tool_invocations`.
        const chatSnapshot = await page.evaluate(() => {
            const ctx = window.SillyTavern.getContext();
            return (ctx.chat || []).map((m, i) => ({
                idx: i,
                is_user: !!m.is_user,
                is_system: !!m.is_system,
                name: m.name,
                mes_head: (m.mes || '').slice(0, 400),
                mes_full_len: (m.mes || '').length,
                contains_tool_call_marker: (m.mes || '').includes('<|tool_call>'),
                contains_atomic_quote_marker: (m.mes || '').includes('<|"|>'),
                tool_invocations: (m.extra?.tool_invocations || []).map(t => ({
                    name: t.displayName || t.name,
                    parameters_keys: t.parameters
                        ? Object.keys(JSON.parse(t.parameters || '{}'))
                        : [],
                })),
                tool_progress: (m.extra?.tool_progress || []).map(t => ({
                    label: t.label, status: t.status,
                })),
            }));
        });

        // Save the captured artifact JSON for downstream debugging.
        fs.writeFileSync(testInfo.outputPath('chat_snapshot.json'),
            JSON.stringify({
                send_result: {
                    finishState: r.finishState,
                    elapsedMs: Math.round(r.elapsedMs),
                    toolInvocations: r.toolInvocations,
                    toolProgress: r.toolProgress.map(t => ({
                        label: t.label, status: t.status,
                    })),
                },
                chat: chatSnapshot,
                network: networkRecords,
            }, null, 2));

        // Log a compact summary to stdout so a CI / quick-skim view
        // sees the bug-presence verdict immediately.
        const leakingTurns = chatSnapshot.filter(m => m.contains_tool_call_marker);
        const toolFiredTurns = chatSnapshot.filter(m => m.tool_progress.length > 0);
        console.log('=== streamed-tool-call marker leak repro ===');
        console.log(`chat turns: ${chatSnapshot.length}`);
        console.log(`turns containing <|tool_call> in .mes: ${leakingTurns.length}`);
        console.log(`turns containing <|"|> atomic-quote in .mes: ` +
            chatSnapshot.filter(m => m.contains_atomic_quote_marker).length);
        console.log(`turns with extra.tool_progress[]: ${toolFiredTurns.length}`);
        for (const t of chatSnapshot) {
            const tags = [];
            if (t.is_user) tags.push('user');
            if (t.is_system) tags.push('system');
            if (t.contains_tool_call_marker) tags.push('🩸<|tool_call>');
            if (t.tool_progress.length) tags.push('✓tool_progress');
            console.log(`  [${t.idx}] ${tags.join(' ')} (${t.mes_full_len} chars) — ${t.name}: ${t.mes_head.slice(0, 80)}`);
        }

        // Regression assertions. The marker-strip fix landed in
        // sillytavern-fork/public/scripts/openai.js (stripModelToolCall-
        // Sentinels in the streaming-delta accumulator) 2026-05-10; this
        // spec now gates on the leak being absent.
        expect(['completed', 'tool_handled']).toContain(r.finishState);
        expect(leakingTurns.length,
            'no chat turn should carry literal <|tool_call> marker text — ' +
            'the FE strip in openai.js should have removed them').toBe(0);
        expect(chatSnapshot.filter(m => m.contains_atomic_quote_marker).length,
            'no chat turn should carry literal <|"|> atomic-quote bytes').toBe(0);
        expect(toolFiredTurns.length,
            'the python-exec tool should still have fired through the ' +
            'structured tool_calls path').toBeGreaterThan(0);
    });
});
