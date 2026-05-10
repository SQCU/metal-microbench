import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

// End-to-end validation that streaming chat-completions reach the
// browser DOM progressively, not as a single end-of-response dump.
// The plumbing has multiple silent-failure modes:
//   - Bridge-side: SSE could be misconfigured and return one chunk
//   - ST-side: streaming option could be disabled in oai_settings
//   - ST renderer: streaming chunks could be coalesced before render
//   - Network buffering: proxy could batch SSE frames
// All four failure modes look the same to a non-streaming client
// (response just arrives, eventually). This test catches them by
// observing the assistant bubble's textContent grow incrementally
// AND by checking time-to-first-DOM-update + largest-jump-between-
// mutations.
//
// Captures video on success too (test.use video=on) so we have
// real-pixel evidence of the streaming feel.
//
// We DON'T rely on the bridge's finish_reason here (that's a
// separate API-level concern, fixed in bridge.py:948); we look at
// what hits the chat bubble's DOM in real time.

// Always-on video so we have real-pixel evidence of streaming feel
// even when the test passes (default video='retain-on-failure'
// drops them on success).
test.use({ video: 'on' });

async function waitForReady(page) {
    await page.goto('/');
    await page.waitForFunction(
        'document.getElementById("preloader") === null',
        { timeout: 60_000 });
    // Default character must be selected so the chat surface is live.
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return Array.isArray(ctx?.characters) && ctx.characters.length > 0;
    }, { timeout: 60_000 });
    await page.evaluate(async () => {
        const ctx = window.SillyTavern.getContext();
        if (ctx.characters.length && String(ctx.characterId) !== '0') {
            await ctx.selectCharacterById(0);
        }
    });
    // Bootstrap-patched settings.json sets chat_completion_source=custom
    // + custom_url, but ST still needs an explicit "connect" tap before
    // the textarea unlocks. Pattern from test 06.
    await page.locator('#API-status-top').click();
    await expect(page.locator('#api_button_openai')).toBeVisible();
    await page.locator('#api_button_openai').click();
    await expect(page.locator('#send_textarea')).toHaveAttribute(
        'placeholder', 'Type a message, or /? for help', { timeout: 30_000 });
}

async function installDomTrace(page, minMesid) {
    await page.evaluate((floor) => {
        window.__streamingTrace = [];
        window.__traceObserver?.disconnect?.();
        if (window.__tracePoll) clearInterval(window.__tracePoll);

        // Track ONLY the assistant bubble whose mesid is strictly
        // greater than `floor`. This is the bubble being created for
        // THIS test's prompt — pre-existing assistant bubbles from
        // previous tests have mesid <= floor and are ignored.
        const chat = document.getElementById('chat');
        const tStart = performance.now();
        let lastLen = 0;
        let lastMesid = null;

        function findAssistantBubble() {
            const all = chat.querySelectorAll('.mes');
            for (let i = all.length - 1; i >= 0; i--) {
                const m = all[i];
                if (m.getAttribute('is_user') === 'true') continue;
                if (m.getAttribute('is_system') === 'true') continue;
                const mesid = parseInt(m.getAttribute('mesid'), 10);
                if (Number.isFinite(mesid) && mesid > floor) return m;
            }
            return null;
        }

        function record(reason) {
            const bubble = findAssistantBubble();
            if (!bubble) return;
            const mesid = bubble.getAttribute('mesid');
            if (mesid !== lastMesid) {
                lastMesid = mesid;
                lastLen = 0;
            }
            const text = bubble.querySelector('.mes_text')?.textContent || '';
            const len = text.length;
            if (len <= lastLen) return;
            window.__streamingTrace.push({
                tMs: performance.now() - tStart,
                mesid,
                len,
                jump: len - lastLen,
                reason,
            });
            lastLen = len;
        }

        const mo = new MutationObserver(() => record('mutation'));
        mo.observe(chat, {
            childList: true,
            subtree: true,
            characterData: true,
        });
        const poll = setInterval(() => record('poll'), 50);
        window.__traceObserver = mo;
        window.__tracePoll = poll;
        window.__traceTStart = tStart;
        window.__traceFloor = floor;
    }, minMesid);
}

test.describe('streaming reaches DOM progressively', () => {
    test.setTimeout(180_000);

    test('long-response stream renders incrementally with low TTFT', async ({ page }, testInfo) => {
        await waitForReady(page);

        // Capture the highest existing mesid BEFORE sending so the
        // trace observer ignores stale bubbles from prior runs.
        const baselineMesid = await page.evaluate(() => {
            const all = document.querySelectorAll('#chat .mes');
            let max = -1;
            for (const m of all) {
                const id = parseInt(m.getAttribute('mesid'), 10);
                if (Number.isFinite(id) && id > max) max = id;
            }
            return max;
        });
        await installDomTrace(page, baselineMesid);

        const prompt = 'Tell me a vivid 250-word story about a tiny lighthouse keeper who befriends a passing albatross. Use multiple paragraphs.';
        await page.locator('#send_textarea').fill(prompt);
        const sendT0 = await page.evaluate(() => performance.now());
        await page.locator('#send_but').click();

        // Wait until the NEW assistant bubble (mesid > baseline) has
        // at least 500 chars, then a grace period to capture tail
        // mutations. Match by mesid > baseline so we never trip on
        // a stale bubble from a previous test run.
        await page.waitForFunction((floor) => {
            const all = document.querySelectorAll('#chat .mes');
            for (let i = all.length - 1; i >= 0; i--) {
                const m = all[i];
                if (m.getAttribute('is_user') === 'true') continue;
                if (m.getAttribute('is_system') === 'true') continue;
                const id = parseInt(m.getAttribute('mesid'), 10);
                if (!(Number.isFinite(id) && id > floor)) continue;
                const txt = m.querySelector('.mes_text')?.textContent || '';
                return txt.length >= 500;
            }
            return false;
        }, baselineMesid, { timeout: 60_000 });
        await page.waitForTimeout(2000);

        // Pull out the trace.
        const trace = await page.evaluate(() => window.__streamingTrace);
        const sendT1 = await page.evaluate(() => performance.now());

        // Save trace as a sibling artifact to the video.
        const tracePath = testInfo.outputPath('streaming_trace.json');
        fs.mkdirSync(path.dirname(tracePath), { recursive: true });
        fs.writeFileSync(tracePath,
            JSON.stringify({ sendT0, sendT1, prompt, trace }, null, 2));

        // Diagnostic dump (visible in test output).
        const finalLen = trace.length ? trace[trace.length - 1].len : 0;
        const ttfu = trace.length ? trace[0].tMs : null;     // time to first DOM update (ms)
        const numUpdates = trace.length;
        const maxJump = trace.reduce((m, e) => Math.max(m, e.jump), 0);
        const avgJump = numUpdates > 0 ? finalLen / numUpdates : 0;
        console.log(`[streaming] final_chars=${finalLen}, num_updates=${numUpdates}, ttfu=${ttfu}ms, max_jump=${maxJump}, avg_jump=${avgJump.toFixed(1)}`);

        // Assertions — these catch the failure modes the user
        // explicitly cares about (streaming silently disabled, single
        // big buffered dump). Thresholds tuned for headless Playwright
        // browsers where MutationObserver is throttled to ~1Hz; real
        // user browsers see 30Hz mutations. The test is detecting
        // PRESENCE of streaming, not measuring its smoothness.

        // 1. Stream actually delivered content.
        expect(finalLen, 'response actually arrived').toBeGreaterThan(50);

        // 2. Time-to-first-DOM-update. Bridge prefill + first chunk
        //    should land in <4s even on cold caches.
        expect(ttfu, 'time-to-first-DOM-update under 4s').toBeLessThan(4000);

        // 3. Incremental updates — if streaming is silently disabled,
        //    we'd see exactly 1-2 mutations (placeholder + final dump).
        //    In headless we get ~5-10 mutations per ~10s response;
        //    require at least 4 distinct growths to catch the
        //    "single buffered dump" regression cleanly.
        expect(numUpdates, 'response arrives in 4+ DOM updates').toBeGreaterThanOrEqual(4);

        // 4. No single jump should comprise more than half the final
        //    response. If the FE/bridge buffers everything until end,
        //    one mutation would jump from 0 to final_len. With real
        //    streaming, even at 1Hz observer cadence each mutation
        //    only captures a slice of the ongoing token flow.
        expect(maxJump, `no single batched dump (max_jump=${maxJump} of ${finalLen})`)
            .toBeLessThan(finalLen * 0.5);
    });
});
