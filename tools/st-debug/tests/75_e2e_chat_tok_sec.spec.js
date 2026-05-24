// End-to-end chat tok/sec through the real ST-fork client.
//
// Operator reports 0.1 tok/sec on the scringloscrambler card in the
// actual ST UI. Direct curl to the bridge measures 29 tok/sec. The
// gap MUST be in the ST forwarding layer OR the user-personas plugin's
// parallel bridge calls (yapper-seed, bridge-status polling, etc).
//
// This spec uses ST's real UI: types into the chat textarea, hits
// send, watches the assistant message render. Times wallclock from
// click-to-render. Also intercepts the ST→bridge POST to measure the
// raw forward-time independently of UI overhead.

import { test, expect } from '@playwright/test';
import { writeFileSync } from 'node:fs';

test.describe('e2e chat tok/sec (P-CHEAP-BATCHED-AR-VISIBLE)', () => {
    test.setTimeout(300_000);

    test('send a message + measure end-to-end + count bridge competition', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'measurement spec is desktop-only');

        // Capture ALL outbound bridge calls during the test so we
        // can count concurrent load (yapper-seed, bridge-status,
        // synth dispatches, etc).
        const bridgeCalls = [];
        page.on('request', (req) => {
            const url = req.url();
            // Plugin endpoints + ST's chat-completions forward.
            if (url.includes('/api/plugins/user-personas/') ||
                url.includes('/api/backends/chat-completions/') ||
                url.includes(':8001/')) {
                bridgeCalls.push({
                    ts: Date.now(),
                    method: req.method(),
                    url: url.replace(/^https?:\/\/[^/]+/, ''),
                });
            }
        });

        await page.goto('/');
        await page.waitForFunction(() => document.getElementById('preloader') === null, { timeout: 60_000 });
        await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function', { timeout: 30_000 });

        // Connect to API (the standard ST setup dance).
        await page.locator('#API-status-top').click();
        await expect(page.locator('#api_button_openai')).toBeVisible();
        await page.locator('#api_button_openai').click();
        await page.waitForFunction(() => {
            const ctx = window.SillyTavern?.getContext?.();
            return ctx?.onlineStatus === 'Valid';
        }, { timeout: 30_000 });

        // Settle: wait for any first-launch background activity to quiet down.
        // We want a CLEAN measurement window.
        await page.waitForTimeout(3000);

        // Snapshot bridge state immediately before send.
        const bridgeBefore = await page.request.get('http://127.0.0.1:8001/health');
        const bridgeBeforeJson = await bridgeBefore.json();

        // Snapshot bridge-call count before send (everything captured
        // since page-load up to now is background; we'll measure the
        // delta during the send).
        const bridgeCallsBeforeSend = bridgeCalls.length;

        // Send a message via the actual ST UI.
        const textarea = page.locator('#send_textarea');
        await textarea.click();
        await textarea.fill('uhhh hi');
        const sendT0 = Date.now();
        await page.locator('#send_but').click();

        // Wait for assistant turn to render with non-empty text.
        const messages = page.locator('#chat .mes:not(.smallSysMes)');
        await expect(messages).toHaveCount(2, { timeout: 240_000 });
        const lastText = messages.last().locator('.mes_text');
        await expect(lastText).not.toBeEmpty({ timeout: 120_000 });

        // Wait for streaming to finalize (text stops growing for ~1s).
        let lastLen = 0; let stableTicks = 0;
        for (let i = 0; i < 60; i++) {
            await page.waitForTimeout(500);
            const len = (await lastText.innerText()).length;
            if (len === lastLen) {
                stableTicks++;
                if (stableTicks >= 3) break;
            } else {
                stableTicks = 0;
                lastLen = len;
            }
        }
        const sendT1 = Date.now();
        const wallclock = (sendT1 - sendT0) / 1000;

        // Snapshot bridge state after.
        const bridgeAfter = await page.request.get('http://127.0.0.1:8001/health');
        const bridgeAfterJson = await bridgeAfter.json();
        const tokensEmitted = bridgeAfterJson.total_tokens_emitted - bridgeBeforeJson.total_tokens_emitted;
        const stepsRun = bridgeAfterJson.total_steps - bridgeBeforeJson.total_steps;

        const respText = await lastText.innerText();
        const respLen = respText.length;
        const approxRespTokens = Math.max(1, Math.round(respLen / 4));  // rough estimate

        // Count bridge calls that fired DURING the send window.
        const callsDuringSend = bridgeCalls.slice(bridgeCallsBeforeSend);
        const callsByEndpoint = {};
        for (const c of callsDuringSend) {
            const ep = c.url.replace(/\?.*$/, '').replace(/\/[0-9]+/, '/<id>');
            callsByEndpoint[ep] = (callsByEndpoint[ep] || 0) + 1;
        }

        // Write a verbose report so the operator can eyeball it.
        const report = {
            send_wallclock_seconds: wallclock,
            response_chars: respLen,
            response_first_100: respText.slice(0, 100),
            engine_tokens_emitted_during_test: tokensEmitted,
            engine_steps_during_test: stepsRun,
            measured_tok_per_sec_from_engine: tokensEmitted / wallclock,
            estimated_tok_per_sec_from_response_length: approxRespTokens / wallclock,
            bridge_active_streams_after: bridgeAfterJson.active_streams,
            bridge_calls_during_send_total: callsDuringSend.length,
            bridge_calls_during_send_by_endpoint: callsByEndpoint,
            verdict: tokensEmitted / wallclock < 5
                ? 'REGRESSED (< 5 tok/sec at engine; investigate ST forwarding + plugin contention)'
                : 'OK',
        };
        const reportPath = '/tmp/e2e_chat_tok_sec_report.json';
        writeFileSync(reportPath, JSON.stringify(report, null, 2));
        console.log(`\n  REPORT (${reportPath}):`);
        console.log(JSON.stringify(report, null, 2).split('\n').map(s => '    ' + s).join('\n'));

        // Hard assertion: the engine MUST achieve > 5 tok/sec
        // measured from total_tokens_emitted delta. Below that is
        // a regression — the operator's thesis demands cheap batched
        // AR decoding visible at user-facing speed.
        expect(report.measured_tok_per_sec_from_engine,
            'engine tok/sec (from /health delta) must be > 5; ' +
            'measured ' + report.measured_tok_per_sec_from_engine.toFixed(2))
            .toBeGreaterThan(5);
    });
});
