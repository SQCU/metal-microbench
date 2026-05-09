import { test, expect } from '@playwright/test';

async function loadSillyTavern(page) {
    await page.goto('/');
    await page.waitForFunction(
        'document.getElementById("preloader") === null',
        { timeout: 60_000 },
    );

    // We don't need a specific persona for this test — just any
    // character so the chat UI is fully initialized. The bootstrap
    // ensures Seraphina is the auto-loaded default; bootstrap-dropped
    // .json files (Scringlo, Debug) require API-based import which
    // isn't worth the test complexity for cancel-button validation.
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return Array.isArray(ctx?.characters) && ctx.characters.length > 0;
    }, { timeout: 60_000 });

    await page.evaluate(async () => {
        const ctx = window.SillyTavern.getContext();
        if (ctx.characters.length === 0) return;
        const idx = 0;
        if (String(ctx.characterId) !== String(idx)) {
            await ctx.selectCharacterById(idx);
        }
    });

    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return ctx?.characters?.[ctx.characterId]?.name;
    }, { timeout: 30_000 });

    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return typeof window.toolcardsCancelSession === 'function' &&
            ctx?.ToolManager?.tools?.some(t =>
                t?.toFunctionOpenAI?.()?.function?.name === 'async-lookup__lookup');
    }, { timeout: 60_000 });
}

async function startFrontendAsyncLookup(page) {
    return await page.evaluate(() => {
        const ctx = window.SillyTavern.getContext();
        const message = {
            name: ctx.name2 || 'Scringlo',
            is_system: false,
            is_user: false,
            send_date: new Date().toLocaleString(),
            mes: 'Starting a cancellable lookup test.',
            extra: {},
        };
        ctx.chat.push(message);
        const messageId = ctx.chat.length - 1;
        ctx.addOneMessage(message);

        window.__toolcardsCancelButtonResult = ctx.ToolManager
            .invokeFunctionTool('async-lookup__lookup', {
                topic: 'the test cancel target',
            })
            .catch(error => String(error?.message || error));

        return messageId;
    });
}

test.describe('toolcards cancel button', () => {
    test.setTimeout(90_000);

    test('running tool_progress entry renders cancel link and cancels the session', async ({ page }) => {
        await loadSillyTavern(page);
        const messageId = await startFrontendAsyncLookup(page);

        const progress = page.locator(
            `#chat .mes[mesid="${messageId}"] details.custom-tool-progress-collapsible`,
        );
        await expect(progress, 'running tool_progress collapsible appears')
            .toBeVisible({ timeout: 15_000 });
        await expect(progress, 'running progress details starts open')
            .toHaveAttribute('open', '', { timeout: 5_000 });
        await expect(progress.locator('summary'), 'summary shows running status')
            .toContainText('running');

        const cancel = progress.locator('.custom-tool-progress-cancel');
        await expect(cancel, 'cancel link appears for running session')
            .toBeVisible({ timeout: 5_000 });

        const cancelResponse = page.waitForResponse(response =>
            response.url().includes('/api/plugins/toolcards/cancel/') &&
            response.request().method() === 'POST',
        );
        await cancel.click();
        expect((await cancelResponse).status(), 'cancel endpoint response').toBe(200);

        await expect(progress.locator('summary'), 'cancelled entry is marked failed')
            .toContainText('failed', { timeout: 3_000 });
        await expect(cancel, 'cancel link disappears after terminal status')
            .toHaveCount(0, { timeout: 3_000 });

        const entry = await page.evaluate((idx) => {
            const ctx = window.SillyTavern.getContext();
            const progressEntries = ctx.chat[idx]?.extra?.tool_progress || [];
            return progressEntries[progressEntries.length - 1] || null;
        }, messageId);
        expect(entry?.status, 'stored progress entry status').toBe('failed');
        expect(entry?.session_id, 'terminal progress entry clears session_id')
            .toBeNull();
        expect(entry?.error, 'failure reason comes from cancel path')
            .toContain('cancelled by user');
    });
});
