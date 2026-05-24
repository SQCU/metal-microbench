// END-TO-END CACHE VALIDATION FROM REAL ST UI
//
// The task #215 thread: the operator reported that ST sends 2491 prompt
// tokens with `tools=17, messages=5, prompt_tokens=2491, cache_hits=0,
// cache_misses=2491` on EVERY request — every byte cold-prefilled, zero
// reuse. With the radix-trie prefix cache landed, this MUST change: a
// second identical user-action should hit the bulk of the prior turn's
// prefix.
//
// What this spec does:
//   1. Captures the bridge log offset (so we can read only NEW lines)
//   2. Opens ST, connects API
//   3. Sends "uhhh hi" via the real chat textarea
//   4. Waits for the BRIDGE itself to report active_streams=0 (the
//      definitive "this send is done" signal — far more reliable than
//      polling the ST DOM for assistant-text-length stability, which
//      false-fires on the "..." placeholder)
//   5. Reads bridge log lines for send-1
//   6. Sends "uhhh hi" AGAIN
//   7. Waits for active_streams=0 again
//   8. Reads bridge log lines for send-2
//   9. Reports prompt-prefix hashes + cache_hits/misses for both
//  10. Asserts: send-2 cache_hits ≥ 80% of send-1 prompt_tokens

import { test, expect } from '@playwright/test';
import { writeFileSync, statSync, openSync, readSync, closeSync } from 'node:fs';

const BRIDGE_URL = 'http://127.0.0.1:8001';
const BRIDGE_LOG = '/tmp/bridge_serve.log';

function readLogTail(fromByte) {
    const sz = statSync(BRIDGE_LOG).size;
    if (sz <= fromByte) return '';
    const len = sz - fromByte;
    const buf = Buffer.alloc(len);
    const fd = openSync(BRIDGE_LOG, 'r');
    readSync(fd, buf, 0, len, fromByte);
    closeSync(fd);
    return buf.toString('utf-8');
}

function parseBridgeEvents(logText) {
    const events = [];
    const lines = logText.split('\n');
    let current = null;
    for (const ln of lines) {
        const m1 = ln.match(/^\[bridge\] chat_completions: tools=(\d+), tool_choice=\S+, messages=(\d+), stream=(True|False)/);
        if (m1) {
            if (current) events.push(current);
            current = { tools: +m1[1], messages: +m1[2], stream: m1[3] === 'True' };
            continue;
        }
        const m2 = ln.match(/^\[bridge DEBUG\] prompt prefix hash \(first 512 toks\): (\w+) \| total_tokens=(\d+) \| first 20 toks: (\[[^\]]+\])/);
        if (m2 && current) {
            current.promptHash = m2[1];
            current.totalTokens = +m2[2];
            current.firstToks = m2[3];
            continue;
        }
        const m3 = ln.match(/^\[bridge\] usage: prompt_tokens=(\d+), completion_tokens=(\d+), cache_hits=(\d+), cache_misses=(\d+),/);
        if (m3 && current) {
            current.promptTokens = +m3[1];
            current.completionTokens = +m3[2];
            current.cacheHits = +m3[3];
            current.cacheMisses = +m3[4];
            events.push(current);
            current = null;
        }
    }
    if (current) events.push(current);
    return events;
}

// Poll bridge until active_streams == 0 AND we've seen at least one
// usage: line for this send window since `windowOffset`. Times out at
// `timeoutMs`. Returns ms taken to settle.
async function waitForBridgeQuiescent(page, windowOffset, timeoutMs = 180_000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
        const h = await page.request.get(`${BRIDGE_URL}/health`);
        const j = await h.json();
        const tail = readLogTail(windowOffset);
        const hasUsage = /\[bridge\] usage:/.test(tail);
        if (j.active_streams === 0 && hasUsage) {
            return Date.now() - start;
        }
        await page.waitForTimeout(500);
    }
    throw new Error(`bridge did not quiesce within ${timeoutMs}ms`);
}

test.describe('ST cache validation (post-radix-trie)', () => {
    test.setTimeout(600_000);

    test('two identical user-actions; second must hit bulk of first prefix', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'cache validation is desktop-only');

        await page.goto('/');
        await page.waitForFunction(() => document.getElementById('preloader') === null,
            { timeout: 60_000 });
        await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
            { timeout: 30_000 });

        await page.locator('#API-status-top').click();
        await expect(page.locator('#api_button_openai')).toBeVisible();
        await page.locator('#api_button_openai').click();
        await page.waitForFunction(() => {
            const ctx = window.SillyTavern?.getContext?.();
            return ctx?.onlineStatus === 'Valid';
        }, { timeout: 30_000 });

        // Let plugin first-launch chatter quiet down.
        await page.waitForTimeout(3500);
        // Wait for bridge to be idle in case plugin probes are in flight.
        let idle = false;
        for (let i = 0; i < 60; i++) {
            const h = await page.request.get(`${BRIDGE_URL}/health`);
            const j = await h.json();
            if (j.active_streams === 0) { idle = true; break; }
            await page.waitForTimeout(500);
        }
        expect(idle, 'bridge should be idle before send-1').toBe(true);

        // === SEND 1 ===
        const send1WindowStart = statSync(BRIDGE_LOG).size;
        const textarea = page.locator('#send_textarea');
        await textarea.click();
        await textarea.fill('uhhh hi');
        const send1T0 = Date.now();
        await page.locator('#send_but').click();

        const send1SettleMs = await waitForBridgeQuiescent(page, send1WindowStart, 180_000);
        console.log(`  send-1 bridge-quiescent in ${send1SettleMs}ms`);

        // Now wait for ST DOM to catch up (the SSE stream finished;
        // ST's renderer needs a beat to commit the final assistant text).
        await page.waitForTimeout(1500);

        const send1Events = parseBridgeEvents(readLogTail(send1WindowStart));
        const send1Wallclock = (Date.now() - send1T0) / 1000;

        // === SEND 2 ===
        const send2WindowStart = statSync(BRIDGE_LOG).size;
        // Re-locate textarea — ST may rerender after stream end.
        await page.locator('#send_textarea').waitFor({ state: 'visible', timeout: 30_000 });
        await page.locator('#send_textarea').click();
        await page.locator('#send_textarea').fill('uhhh hi');
        const send2T0 = Date.now();
        // Wait for #send_but to be enabled (not in stop mode); ST swaps
        // classes between fa-paper-plane (send) and fa-stop (stop). We
        // just need it to be visible AND clickable. waitForBridgeQuiescent
        // above already established the bridge is idle so the click should
        // succeed immediately.
        const sendBut = page.locator('#send_but');
        await sendBut.waitFor({ state: 'visible', timeout: 30_000 });
        await sendBut.click();

        const send2SettleMs = await waitForBridgeQuiescent(page, send2WindowStart, 180_000);
        console.log(`  send-2 bridge-quiescent in ${send2SettleMs}ms`);

        await page.waitForTimeout(1500);

        const send2Events = parseBridgeEvents(readLogTail(send2WindowStart));
        const send2Wallclock = (Date.now() - send2T0) / 1000;

        // === REPORT ===
        const findMain = (events) => events.length ? events.reduce(
            (m, e) => (e.promptTokens || 0) > (m.promptTokens || 0) ? e : m, events[0]) : null;
        const main1 = findMain(send1Events);
        const main2 = findMain(send2Events);

        const report = {
            send1: {
                wallclock_s: send1Wallclock,
                bridge_settle_ms: send1SettleMs,
                events_count: send1Events.length,
                main_event: main1,
                all_events: send1Events,
            },
            send2: {
                wallclock_s: send2Wallclock,
                bridge_settle_ms: send2SettleMs,
                events_count: send2Events.length,
                main_event: main2,
                all_events: send2Events,
            },
            cache_uplift: main1 && main2 ? {
                send1_prompt_tokens: main1.promptTokens,
                send2_prompt_tokens: main2.promptTokens,
                send2_cache_hits: main2.cacheHits,
                send2_cache_misses: main2.cacheMisses,
                pct_of_send2_prompt_hit: main2.cacheHits / Math.max(1, main2.promptTokens),
                pct_of_send1_prompt_reused: main2.cacheHits / Math.max(1, main1.promptTokens),
                send1_hash: main1.promptHash,
                send2_hash: main2.promptHash,
                send1_first20: main1.firstToks,
                send2_first20: main2.firstToks,
            } : null,
        };
        const reportPath = '/tmp/st_cache_validation_report.json';
        writeFileSync(reportPath, JSON.stringify(report, null, 2));
        console.log('\n  REPORT (' + reportPath + '):');
        console.log(JSON.stringify(report, null, 2).split('\n').map(s => '    ' + s).join('\n'));

        expect(main1, 'send-1 main event present').not.toBeNull();
        expect(main2, 'send-2 main event present').not.toBeNull();
        // The actual cache validation.
        expect(main2.cacheHits / Math.max(1, main1.promptTokens),
            `send-2 must reuse >= 80% of send-1's prefix (got ${main2.cacheHits}/${main1.promptTokens} = ${(main2.cacheHits / main1.promptTokens * 100).toFixed(1)}%)`)
            .toBeGreaterThan(0.80);
    });
});
