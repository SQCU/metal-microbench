import { test, expect } from '@playwright/test';

// FIRST e2e test — proves the basic loop works end-to-end:
//   browser → ST UI (:8002) → ST backend → bridge (:8001) → engine
//   → bridge → ST backend → ST UI → DOM rendering
//
// Validates BOTH surfaces:
//   - Network: the request body ST forwards to its own backend
//   - DOM: what content ends up rendered for the user
//
// Phase-2 work (separate spec files): tool-call DOM rendering,
// SVG-from-query workflow, long-sequential-call telemetry.

test.describe('basic chat loop', () => {
    test('settings landed + send/receive/render works', async ({ page }) => {
        // ── intercept network call so we can validate the request shape. ──
        const captured = { request: null, responseStatus: null };
        await page.route('**/api/backends/chat-completions/generate', async (route) => {
            const req = route.request();
            captured.request = {
                method: req.method(),
                postData: req.postData(),
            };
            const resp = await route.fetch();
            captured.responseStatus = resp.status();
            await route.fulfill({ response: resp });
        });

        // ── (1) Visit + wait for the preloader to clear. ──
        await page.goto('/');
        await page.waitForFunction(
            'document.getElementById("preloader") === null',
            { timeout: 60_000 });

        // ── (2) Bootstrap-patch landed: select element + URL input show
        //        the values we wrote to settings.json. (Reading the select
        //        is stable across versions; module-scope `oai_settings`
        //        const isn't accessible from Playwright's evaluate context.)
        await expect(page.locator('#chat_completion_source'))
            .toHaveValue('custom');
        await expect(page.locator('#custom_api_url_text'))
            .toHaveValue('http://127.0.0.1:8001');

        // ── (2.5) Connect to the API. After bootstrap, settings ARE loaded
        //         but ST hasn't validated the connection yet — textarea
        //         placeholder shows "Not connected to API!" and the
        //         #API-status-top icon is red. Two clicks needed:
        //           1. open the API connections panel (#API-status-top)
        //           2. click Connect inside it (#api_button_openai)
        //         Then wait for the placeholder to flip.
        await page.locator('#API-status-top').click();
        await expect(page.locator('#api_button_openai')).toBeVisible();
        await page.locator('#api_button_openai').click();
        await expect(page.locator('#send_textarea')).toHaveAttribute(
            'placeholder', 'Type a message, or /? for help', { timeout: 30_000 });

        // ── (3) Type and send. ──
        const textarea = page.locator('#send_textarea');
        await textarea.click();
        await textarea.fill('reply with just "ack"');

        const messages = page.locator('#chat .mes:not(.smallSysMes)');

        await page.locator('#send_but').click();

        // ── (4) Wait until: (a) at least one user-msg + one assistant-msg
        //        rendered, and (b) the LAST one has non-empty text. The
        //        welcome assistant message can be replaced rather than
        //        appended-to depending on the chat state, so we don't
        //        rely on count delta — we look for a USER msg followed
        //        by an ASSISTANT msg with text.
        await expect(messages).toHaveCount(2, { timeout: 90_000 });
        const lastText = messages.last().locator('.mes_text');
        await expect(lastText).not.toBeEmpty({ timeout: 60_000 });
        // Brief settle-time for streaming tail tokens.
        await page.waitForTimeout(800);

        // ── (5) Read the assistant's last message text. ──
        const assistantText = await messages.last()
            .locator('.mes_text').innerText();

        console.log(`  assistant text (${assistantText.length} chars):`,
            JSON.stringify(assistantText.slice(0, 300)));

        expect(assistantText.length, 'assistant produced non-empty response')
            .toBeGreaterThan(0);

        // ── (6) Validate the network round trip happened with the right shape. ──
        expect(captured.request, 'ST proxied a chat-completions request')
            .not.toBeNull();
        expect(captured.responseStatus, 'ST backend returned 2xx')
            .toBeLessThan(300);

        const sentBody = JSON.parse(captured.request.postData);
        expect(sentBody.chat_completion_source).toBe('custom');
        expect(sentBody.custom_url).toBe('http://127.0.0.1:8001');
        expect(Array.isArray(sentBody.messages)).toBe(true);

        // ── (7) Scaffolding-cleanliness check on the DOM. ──
        // Earlier regression: `<|channel>thought\n<channel|>` echo at
        // turn-start, trailing `<turn|>`, and raw `<|tool_call>` markers.
        // The bridge has output-side strippers; this confirms they
        // actually reach the DOM.
        expect(assistantText, 'no <|channel> bleed in DOM')
            .not.toContain('<|channel');
        expect(assistantText, 'no trailing <turn|> in DOM')
            .not.toContain('<turn|>');
        expect(assistantText, 'no raw <|tool_call> markers in DOM')
            .not.toContain('<|tool_call');
    });
});
