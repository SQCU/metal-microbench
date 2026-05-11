import { test, expect } from '@playwright/test';

// Phase-3b: validates that captions actually reach the BROWSER DOM,
// not just the toolcards plugin's event stream (test 05 covers that
// half). This is the end-to-end UX validation: a real browser, a real
// chat session, the captioned toolcard fired, the caption text
// rendered in the placeholder bubble of the calling assistant turn.
//
// Strategy: open ST as a real browser, drive a chat, fire the
// captioned tool. We use a workaround for the model's flaky tool-call
// elicitation: the test sends the user's message THEN hits the plugin
// directly to start_invoke against the same session. The plugin's
// session attachment to the calling message is what makes the
// captions surface in the DOM.
//
// Actually that's complicated. Simpler approach: the toolcards FE
// extension will surface progress events for ANY ongoing session
// regardless of tool-call origin. But the FE needs a "caller_message"
// to attach to. So we use the FE's own visibilitychange-reconcile
// path: start a session via the plugin API while ST is loaded, the
// FE's polling picks it up.
//
// For a focused test that just validates "caption text reaches the
// DOM", we use a SIMPLER approach: the plugin's event stream IS
// what feeds the DOM. We've validated that captions arrive as
// progress events (test 05). The FE renderer for tool_progress
// is well-tested by ST itself. So this test instead validates
// that the FE can be SHOWN a caption progress entry and renders
// it correctly.

// 2026-05-08: marked .skip pending the natural-flow path. The
// externally-injected /start_invoke we use here doesn't trigger the
// FE toolcards extension's polling-and-attaching pipeline — that
// pipeline only kicks in when the FE itself initiated the invocation
// in response to a model-emitted tool_call. Two follow-up paths to
// make this test runnable:
//   (a) drive through the UI: prompt the model with a directive
//       message that elicits a tool_call, retrying until success.
//       Sampling-flaky per the elicitation findings; needs retry
//       logic + bounded budget.
//   (b) extend the toolcards FE extension with a debug entry point
//       that lets external clients attach a session to a specific
//       caller_message. Requires modifying the SillyTavern source.
// Until either lands, test 05 (event-stream-level validation) is the
// canonical integration check.
test.describe.skip('captioned toolcard DOM render', () => {
    test.setTimeout(8 * 60 * 1000);

    test('caption progress events from a running session render in DOM',
        async ({ page, request }) => {
            await page.goto('/');
            await page.waitForFunction(
                'document.getElementById("preloader") === null',
                { timeout: 60_000 });
            await page.locator('#API-status-top').click();
            await expect(page.locator('#api_button_openai')).toBeVisible();
            await page.locator('#api_button_openai').click();
            await expect(page.locator('#send_textarea')).toHaveAttribute(
                'placeholder', 'Type a message, or /? for help', { timeout: 30_000 });

            // Send a regular chat message first to create a caller turn
            // that the toolcards FE can attach progress events to.
            const textarea = page.locator('#send_textarea');
            await textarea.click();
            await textarea.fill('quick chat — say "ready" and only "ready"');
            const messages = page.locator('#chat .mes:not(.smallSysMes)');
            await page.locator('#send_but').click();
            await expect(messages).toHaveCount(2, { timeout: 60_000 });
            await expect(messages.last().locator('.mes_text')).not.toBeEmpty(
                { timeout: 60_000 });

            // Now: get the chat_id + the assistant message's id so we
            // can attach the toolcard session to the right caller.
            const callerCtx = await page.evaluate(() => {
                // ST's chat array — global module var; expose via UI hooks.
                const chat = window.chat || [];
                const callerIdx = chat.length - 1;   // last message
                const chatId = window.getCurrentChatId
                    ? window.getCurrentChatId()
                    : (window.chat_metadata?.chat_id || null);
                return { callerIdx, chatId };
            });
            console.log(`  caller context: ${JSON.stringify(callerCtx)}`);

            // Read oai_settings to build a profile that mirrors the
            // FE's actual configuration — that's what the plugin
            // expects in /start_invoke.
            const profile = await page.evaluate(() => {
                const s = window.oai_settings || {};
                return {
                    api: 'openai',
                    mode: 'cc',
                    chat_completion_source: s.chat_completion_source || 'custom',
                    custom_url: s.custom_url || 'http://127.0.0.1:8001',
                    custom_model: s.custom_model || 'gemma-4-a4b',
                    stream_openai: s.stream_openai || false,
                    temperature_openai: s.temperature_openai || 0.4,
                    openai_max_tokens: s.openai_max_tokens || 4096,
                };
            });

            // Kick off the captioned toolcard.
            const startResp = await request.post(
                'http://127.0.0.1:8002/api/plugins/toolcards/start_invoke/query-to-svg-captioned/generate',
                {
                    data: {
                        args: {
                            query: 'a tiny blue triangle',
                            max_iters: 1,
                            width: 256,
                            height: 256,
                        },
                        profile: profile,
                        chat_id: callerCtx.chatId,
                        caller_message_id: callerCtx.callerIdx,
                    },
                });
            expect(startResp.status(), 'start_invoke 200').toBe(200);
            const { session_id } = await startResp.json();
            console.log(`  session_id: ${session_id}`);

            // The FE's toolcards extension long-polls /poll/<session_id>
            // and writes progress events into msg.extra.tool_progress[].
            // ST's messageFormatting renders that into a status bubble
            // attached to the caller turn. We wait for the rendered
            // bubble to contain a "caption:" line.
            //
            // Selector is best-effort — the toolcards extension renders
            // tool_progress entries within the .mes_text or a sibling
            // container. We grep for the literal "caption:" text within
            // the assistant message div.
            await expect.poll(async () => {
                const messageText = await messages.last().innerText();
                return messageText.includes('caption:') ? messageText : null;
            }, { timeout: 7 * 60 * 1000, intervals: [1000, 2000] }).not.toBeNull();

            const finalText = await messages.last().innerText();
            console.log(`  caller message DOM text (${finalText.length} chars):`);
            console.log(`  ${finalText.slice(0, 600).split('\n').join(' ⏎ ')}`);

            expect(finalText, 'caption appears in the assistant turn DOM')
                .toContain('caption:');

            // Take a screenshot for visual confirmation.
            await page.screenshot({
                path: 'test-results/captioned-dom-render.png',
                fullPage: true,
            });
        });
});
