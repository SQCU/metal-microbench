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
            t?.toFunctionOpenAI?.()?.function?.name === 'tree-of-thoughts__explore');
    }, { timeout: 60_000 });
}

async function startFrontendTreeOfThoughts(page) {
    return await page.evaluate(() => {
        const ctx = window.SillyTavern.getContext();
        const message = {
            name: ctx.name2 || 'Assistant',
            is_system: false,
            is_user: false,
            send_date: new Date().toLocaleString(),
            mes: 'Exploring the prompt across several branches.',
            extra: {},
        };
        ctx.chat.push(message);
        const messageId = ctx.chat.length - 1;
        ctx.addOneMessage(message);

        window.__toolcardsBranchStreamingResult = ctx.ToolManager
            .invokeFunctionTool('tree-of-thoughts__explore', {
                question: 'What is a sensible plan for evaluating a risky product idea before committing a full build?',
                branches: ['practical', 'creative', 'skeptical'],
            })
            .catch(error => String(error?.message || error));

        return messageId;
    });
}

test.describe('tree-of-thoughts branch streaming', () => {
    test.setTimeout(90_000);

    test('renders branch_progress updates before final synthesis completes', async ({ page }) => {
        const branchPollEvents = [];
        page.on('response', async response => {
            if (!response.url().includes('/api/plugins/toolcards/poll/')) return;
            try {
                const event = await response.json();
                if (event?.type === 'branch_progress') branchPollEvents.push(event);
            } catch (_) {
                // Ignore non-JSON and already-consumed response bodies.
            }
        });

        await loadSillyTavern(page);
        const messageId = await startFrontendTreeOfThoughts(page);

        const progress = page.locator(
            `#chat .mes[mesid="${messageId}"] details.custom-tool-progress-collapsible`,
        );
        await expect(progress, 'running tool_progress entry appears')
            .toBeVisible({ timeout: 15_000 });

        const branches = progress.locator('.custom-tool-progress-branch');
        await expect(branches, 'all branch slots render from started events')
            .toHaveCount(3, { timeout: 20_000 });

        await page.waitForFunction((idx) => {
            const ctx = window.SillyTavern?.getContext?.();
            const entries = ctx?.chat?.[idx]?.extra?.tool_progress || [];
            const entry = entries[entries.length - 1];
            return Array.isArray(entry?.branches) &&
                entry.branches.some(branch => branch?.status === 'complete') &&
                entry.status !== 'done';
        }, messageId, { timeout: 75_000 });

        await expect(
            progress.locator('.custom-tool-progress-branch[data-status="complete"]'),
            'all branches eventually complete',
        ).toHaveCount(3, { timeout: 90_000 });

        const summaries = await branches.evaluateAll(nodes =>
            nodes.map(node => node.querySelector('details summary')?.textContent?.trim() || ''),
        );
        expect(summaries, 'all complete branches render a non-empty summary')
            .toHaveLength(3);
        for (const summary of summaries) {
            expect(summary.length, `summary "${summary}" is non-empty`).toBeGreaterThan(0);
        }

        await expect(progress.locator('summary').first(), 'final tool_progress entry reaches done')
            .toContainText('done', { timeout: 90_000 });

        const result = await page.evaluate(async () => window.__toolcardsBranchStreamingResult);
        expect(String(result), 'final tool result includes synthesis payload').toContain('synthesis');
        expect(
            branchPollEvents.filter(event => event.status === 'complete').length,
            'poll stream included per-branch completion events',
        ).toBeGreaterThanOrEqual(3);
    });
});
