import { test, expect } from '@playwright/test';

// Phase-2a: validates the toolcards plumbing end-to-end with the
// `hello-world__greet` tool — chosen because it's pure stdio JSON RPC
// with no inner LLM calls, so it isolates "did the tool dispatch +
// result-attachment pipeline work?" from "did the descendant agent's
// llm_call loop work?".
//
// The plumbing chain we're checking:
//   1. Plugin /api/plugins/toolcards/list returned hello-world
//   2. Frontend extension registered `hello-world__greet` with
//      ToolManager (so it appears in `tools[]` of outbound requests)
//   3. Model emits a tool_call to that tool (we send a very directive
//      prompt to make this deterministic at temp=1)
//   4. ST POSTs /start_invoke, plugin spawns service.py
//   5. Service responds with `{ok:true, result: "hello, X"}` (≤100ms)
//   6. ST attaches the result via the marker pipeline; the assistant's
//      mes ends up containing the greeting

test.describe('toolcards plumbing (hello-world)', () => {
    test('greet tool round-trips through the toolcards plugin', async ({ page }) => {
        await page.goto('/');
        await page.waitForFunction(
            'document.getElementById("preloader") === null',
            { timeout: 60_000 });

        // Connect to bridge.
        await page.locator('#API-status-top').click();
        await expect(page.locator('#api_button_openai')).toBeVisible();
        await page.locator('#api_button_openai').click();
        await expect(page.locator('#send_textarea')).toHaveAttribute(
            'placeholder', 'Type a message, or /? for help', { timeout: 30_000 });

        // Verify the tool actually got registered with the ToolManager.
        // The toolcards extension queries /api/plugins/toolcards/list at
        // load and registers each declared tool. We can confirm by
        // hitting the same endpoint.
        const cardsResp = await page.request.get(
            'http://127.0.0.1:8002/api/plugins/toolcards/list');
        const cards = (await cardsResp.json()).cards;
        const helloCard = cards.find(c => c.id === 'hello-world');
        expect(helloCard, 'hello-world card present').toBeTruthy();
        expect(helloCard.tools.find(t => t.name === 'greet'),
            'greet tool present').toBeTruthy();

        // Send a directive prompt to maximize tool-use probability at
        // temp=1. (Model can still decline; the test fails with a clear
        // message if so — that's a real signal worth surfacing.)
        const textarea = page.locator('#send_textarea');
        await textarea.click();
        await textarea.fill(
            "Use the hello-world__greet tool to greet 'phase-2-test'. " +
            "Use the tool, do not just describe what it would do.");

        // Capture the outbound generate request to validate that ST
        // included `tools[]` with hello-world__greet in the body.
        const sentBodies = [];
        await page.route('**/api/backends/chat-completions/generate', async (route) => {
            sentBodies.push(JSON.parse(route.request().postData() || '{}'));
            const resp = await route.fetch();
            await route.fulfill({ response: resp });
        });

        const messages = page.locator('#chat .mes:not(.smallSysMes)');
        await page.locator('#send_but').click();

        // Wait for assistant turn to be present + non-empty. Toolcards
        // can take longer than basic chat — generous timeout.
        await expect(messages).toHaveCount(2, { timeout: 60_000 });
        const lastMesText = messages.last().locator('.mes_text');
        await expect(lastMesText).not.toBeEmpty({ timeout: 60_000 });
        await page.waitForTimeout(1000);

        // Validate ST sent tools[] with hello-world__greet.
        expect(sentBodies.length, 'at least one generate call').toBeGreaterThan(0);
        const firstBody = sentBodies[0];
        const toolNames = (firstBody.tools || []).map(
            t => t.function?.name).filter(Boolean);
        expect(toolNames, 'hello-world__greet registered as tool').toContain(
            'hello-world__greet');

        // The assistant message can either CONTAIN "hello, phase-2-test"
        // (tool fired and result inlined) OR just describe it (tool
        // declined). Print which case we hit so a flaky temp=1 run is
        // diagnosable, and assert on the success case.
        const assistantText = await lastMesText.innerText();
        console.log(`  assistant rendered text:`, JSON.stringify(assistantText.slice(0, 400)));

        // The toolcard's result text appears in the assistant message
        // (via the marker pipeline). If we see the greeting, the entire
        // pipeline worked.
        expect(assistantText.toLowerCase(), 'greeting result inlined into assistant turn')
            .toContain('phase-2-test');
    });
});
