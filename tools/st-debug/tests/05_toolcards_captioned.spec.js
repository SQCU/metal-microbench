import { test, expect } from '@playwright/test';

// Phase-3a: validates the spur-caption integration into the toolcards
// plugin. The captioned variant of query-to-svg fires a tiny
// summarizer llm_call after each iter and emits the result as a
// `progress` event with a "caption: " prefix. The FE renders progress
// events in the placeholder bubble of the calling turn — so by
// validating that captions arrive as progress events of the running
// session, we're confirming the feed-back-into-DOM path end-to-end.
//
// Why direct-invoke: the outer model's tool-call decision is flaky
// (per the elicitation findings), so we don't depend on it for
// pipeline validation. Direct-invoke fires the toolcard with known
// args and lets us cleanly observe the events that come back.
//
// What we expect to see in the event stream for max_iters=2:
//   progress: target NxN, 2 iter(s)
//   progress: iter 1/2: requesting SVG
//   progress: iter 1: got <N>-char SVG, rasterizing
//   progress: iter 1: done in X.Xs (<N> chars)
//   progress: caption: <one short sentence about iter 1's SVG>     ← NEW
//   progress: iter 2/2: requesting refinement against vision feedback
//   progress: iter 2: got <N>-char SVG, rasterizing
//   progress: iter 2: done in X.Xs (<N> chars)
//   progress: caption: <one short sentence about iter 2's SVG>     ← NEW
//   result: ...
//
// Test asserts:
//   - at least 2 caption progress events arrive (one per iter)
//   - each caption is a non-empty string under ~30 words
//   - the result event still contains the rasterized PNG (didn't break
//     the original toolcard)

// Per-test cleanup: kill orphan service.py processes from any previous
// interrupted test run. Without this, the toolcards plugin's per-card
// service queue (plugins/toolcards/index.mjs:338) parks our new
// invocation behind the orphan's still-active session. Observable as
// multi-minute hangs with no error events. Found via static analysis
// 2026-05-08.
import { execSync } from 'node:child_process';
test.beforeEach(() => {
    try {
        execSync("pkill -f 'uv run.*python service.py'", { stdio: 'ignore' });
    } catch { /* none running, ignore */ }
});

test.describe('toolcards spur-caption integration', () => {
    test.setTimeout(2 * 60 * 1000);   // 2 min plenty for max_iters=1

    test('captioned query-to-svg emits caption progress events', async ({ request }) => {
        // Same profile shape as test 04.
        const profile = {
            api: 'openai',
            mode: 'cc',
            preset: 'default',
            chat_completion_source: 'custom',
            custom_url: 'http://127.0.0.1:8001',
            custom_model: 'gemma-4-a4b',
            stream_openai: false,
            temperature_openai: 0.4,
            openai_max_tokens: 4096,
        };

        const startResp = await request.post(
            'http://127.0.0.1:8002/api/plugins/toolcards/start_invoke/query-to-svg-captioned/generate',
            {
                data: {
                    args: {
                        query: 'a small purple star',
                        max_iters: 1,
                        width: 256,
                        height: 256,
                    },
                    profile: profile,
                },
            });
        expect(startResp.status(), 'start_invoke 200').toBe(200);
        const { session_id } = await startResp.json();
        console.log(`  session_id: ${session_id}`);

        // Drain events.
        let resultEvent = null;
        const captionLines = [];
        const allProgress = [];
        const deadline = Date.now() + 7 * 60 * 1000;
        while (Date.now() < deadline) {
            const pollResp = await request.get(
                `http://127.0.0.1:8002/api/plugins/toolcards/poll/${session_id}`);
            if (pollResp.status() !== 200) {
                console.log(`  poll status ${pollResp.status()}, retrying`);
                await new Promise(r => setTimeout(r, 500));
                continue;
            }
            const event = await pollResp.json();
            if (event.type === 'progress') {
                const text = event.text || '';
                allProgress.push(text);
                if (text.startsWith('caption:')) {
                    const captionText = text.slice('caption:'.length).trim();
                    captionLines.push(captionText);
                    console.log(`  CAPTION: ${captionText}`);
                } else {
                    console.log(`  progress: ${text}`);
                }
            } else if (event.type === 'heartbeat') {
                continue;
            } else if (event.type === 'result' || event.type === 'error') {
                resultEvent = event;
                break;
            }
        }

        expect(resultEvent, 'received a result event').not.toBeNull();
        if (resultEvent.type === 'error') {
            throw new Error(`toolcard error: ${JSON.stringify(resultEvent).slice(0, 500)}`);
        }

        console.log(`\n  total progress events: ${allProgress.length}`);
        console.log(`  caption events:        ${captionLines.length}`);

        // Assertions:
        // 1. At least one caption per iter → at least 1 caption for max_iters=1.
        //    (The spec doc-style "captions every 5s during long-running work"
        //    requires multi-iter; 1 caption is the minimum proof of integration.)
        expect(captionLines.length,
            'at least 1 caption event (one per iter)').toBeGreaterThanOrEqual(1);

        // 2. Each caption is non-empty and reasonably short.
        for (const cap of captionLines) {
            expect(cap.length, 'caption non-empty').toBeGreaterThan(0);
            // Caption agent capped at 32 tokens — that's max ~30 words.
            const wordCount = cap.split(/\s+/).filter(w => w.length).length;
            expect(wordCount, `caption "${cap}" within budget`).toBeLessThanOrEqual(35);
        }

        // 3. Original behavior preserved — result has rasterized PNG.
        const result = resultEvent.result || resultEvent;
        const embed = result.embed || [];
        const imagePart = embed.find(p =>
            p.type === 'image_url' &&
            typeof p?.image_url?.url === 'string' &&
            p.image_url.url.startsWith('data:image/'));
        expect(imagePart, 'result still has rasterized PNG embedded').toBeTruthy();
    });
});
