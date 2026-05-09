import { test, expect } from '@playwright/test';

async function loadSillyTavern(page) {
    await page.goto('/');
    await page.waitForFunction(
        'document.getElementById("preloader") === null',
        { timeout: 60_000 },
    );

    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return Array.isArray(ctx?.characters) && ctx.characters.length > 0;
    }, { timeout: 60_000 });

    await page.evaluate(async () => {
        const ctx = window.SillyTavern.getContext();
        if (ctx.characters.length === 0) return;
        if (String(ctx.characterId) !== '0') {
            await ctx.selectCharacterById(0);
        }
    });

    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return ctx?.characters?.[ctx.characterId]?.name;
    }, { timeout: 30_000 });

    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return ctx?.ToolManager?.tools?.some(t =>
            t?.toFunctionOpenAI?.()?.function?.name === 'async-lookup__lookup');
    }, { timeout: 60_000 });
}

test.describe('async tool frontend fire-and-forget', () => {
    test.setTimeout(120_000);

    test('async manifest tools return quickly and inject their later result', async ({ page }) => {
        await loadSillyTavern(page);

        const startedAt = Date.now();
        const { messageId, result } = await page.evaluate(async () => {
            const ctx = window.SillyTavern.getContext();
            const topic = 'something slow for the async action callback';
            const message = {
                name: ctx.name2 || 'Assistant',
                is_system: false,
                is_user: false,
                send_date: new Date().toLocaleString(),
                mes: 'Starting a background lookup test.',
                extra: {},
            };
            ctx.chat.push(message);
            const messageId = ctx.chat.length - 1;
            ctx.addOneMessage(message);

            const result = await ctx.ToolManager.invokeFunctionTool('async-lookup__lookup', {
                topic,
            });
            return { messageId, result };
        });
        const elapsedMs = Date.now() - startedAt;

        expect(elapsedMs, 'function tool action resolves with placeholder instead of slow result')
            .toBeLessThan(2000);
        expect(String(result), 'placeholder text explains that the result is separate')
            .toMatch(/Background|result will arrive separately/);

        const asyncMessage = await page.waitForFunction((idx) => {
            const ctx = window.SillyTavern?.getContext?.();
            return ctx?.chat?.slice(idx + 1).find(m => m?.extra?.tool_async_result === true) || null;
        }, messageId, { timeout: 30_000 });
        const systemMessage = await asyncMessage.jsonValue();

        expect(systemMessage?.is_system, 'async result is injected as a system message').toBe(true);
        expect(systemMessage?.extra?.summary, 'async result stores a non-empty summary')
            .toEqual(expect.any(String));
        expect(systemMessage.extra.summary.trim().length, 'summary is non-empty')
            .toBeGreaterThan(0);

        const entry = await page.waitForFunction((idx) => {
            const ctx = window.SillyTavern?.getContext?.();
            const entries = ctx?.chat?.[idx]?.extra?.tool_progress || [];
            const latest = entries[entries.length - 1];
            return latest?.status === 'done' ? latest : null;
        }, messageId, { timeout: 5_000 });
        const progressEntry = await entry.jsonValue();

        expect(progressEntry?.status, 'original caller progress reaches done').toBe('done');
        expect(progressEntry?.done, 'original caller progress is terminal').toBe(true);
        expect(progressEntry?.session_id, 'terminal progress entry clears session_id').toBeNull();
    });
});
