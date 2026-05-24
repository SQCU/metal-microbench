import { test, expect } from '@playwright/test';

// Cold-prefill tolerance validation.
//
// Before the fix, ST's /generate handler aborted the upstream fetch the
// moment the inbound (browser) socket closed for any reason. Combined
// with multi-second cold-prefill TTFB on local LLM backends, this
// produced silent failures: bridge logged `disconnect before first SSE
// chunk`, browser saw no reply.
//
// After the fix, inbound socket close BEFORE the first response byte
// goes back does NOT abort the upstream — bridges with prefix caches
// (the local LLM bridge under test) get to complete prefill and warm
// the cache, and the SSE proceeds as normal when it does begin.
//
// This test:
//   1) Sends the "uhhh hi" message through the real ST UI.
//   2) Asserts an assistant message appears with non-empty text within
//      60s. ("…" is the welcome placeholder; we explicitly reject it.)
//   3) Reports the wall-clock TTR for the assistant message.
test.describe('cold prefill tolerance', () => {
    test('uhhh hi → assistant reply renders within 60s', async ({ page }) => {
        page.on('console', (msg) => {
            const t = msg.text();
            if (t.includes('[FE TRACE]') || t.includes('signal aborted') ||
                t.includes('Stream stats') || t.includes('Generation')) {
                console.log(`[browser ${msg.type()}] ${t}`);
            }
        });
        page.on('pageerror', (err) => console.log(`[pageerror] ${err.message}`));

        await page.goto('/');
        await page.waitForFunction(
            'document.getElementById("preloader") === null',
            { timeout: 60_000 });
        await expect(page.locator('#chat_completion_source')).toHaveValue('custom');
        await page.locator('#API-status-top').click();
        await expect(page.locator('#api_button_openai')).toBeVisible();
        await page.locator('#api_button_openai').click();
        await expect(page.locator('#send_textarea')).toHaveAttribute(
            'placeholder', 'Type a message, or /? for help', { timeout: 30_000 });

        const messages = page.locator('#chat .mes:not(.smallSysMes)');
        const beforeSendCount = await messages.count();

        await page.locator('#send_textarea').click();
        await page.locator('#send_textarea').fill('uhhh hi');

        const startMs = Date.now();
        await page.locator('#send_but').click();

        // Wait for user-msg count bump.
        await page.waitForFunction(
            (prev) => document.querySelectorAll('#chat .mes:not(.smallSysMes)').length > prev,
            beforeSendCount,
            { timeout: 30_000 });
        const afterUserCount = await messages.count();

        // Wait for assistant reply (last message: not user, not system,
        // non-empty trimmed text, not the placeholder).
        await page.waitForFunction(
            (afterUser) => {
                const ctx = window.SillyTavern?.getContext?.();
                if (!ctx?.chat) return false;
                const last = ctx.chat[ctx.chat.length - 1];
                if (!last) return false;
                if (last.is_user || last.is_system) return false;
                const text = (last.mes || '').trim();
                if (text.length === 0) return false;
                if (text === '…' || text === '...') return false;
                return ctx.chat.length >= afterUser;
            },
            afterUserCount,
            { timeout: 60_000, polling: 500 });

        const elapsedMs = Date.now() - startMs;
        const lastText = await page.evaluate(() => {
            const ctx = window.SillyTavern?.getContext?.();
            return ctx?.chat?.[ctx.chat.length - 1]?.mes || '';
        });
        console.log(`assistant reply arrived in ${(elapsedMs / 1000).toFixed(2)}s`);
        console.log(`assistant text (${lastText.length} chars):`,
            JSON.stringify(lastText.slice(0, 300)));
        expect(lastText.length).toBeGreaterThan(0);
    });
});
