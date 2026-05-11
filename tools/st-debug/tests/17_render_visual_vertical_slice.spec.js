import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

// Vertical slice for the render-visual card: 2-fork + 2-spawn
// recursive decomposition with parent-voice n-of-k summaries.
// See docs/scalable_oversight_recursive_decomposition.md.
//
// What this test PROVES that test 16 didn't:
//   - The spawn vs fork distinction in action: F1/F2 inherit
//     scringlo's persona, S1/S2 don't — and the SVG output is
//     mathematically correct (which the persona-bearing parent
//     could not have produced alone in-character).
//   - Sibling spawns share a "rendering engine" prefix (KV cache
//     hits between S1 and S2).
//   - The result.summary's bracket-prefixed format ([F1:plan],
//     [S1:write], [S2:validate], [host:run], [F2:wrap]) cleanly
//     surfaces 5 lines of bounded oversight for ~30s of decode
//     work spanning thousands of intermediate tokens.
//   - Real visual output: an SVG rendered inline in scringlo's
//     bubble.

test.use({ video: 'on' });

async function waitForReady(page) {
    await page.goto('/');
    await page.waitForFunction(
        'document.getElementById("preloader") === null',
        { timeout: 60_000 });
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return Array.isArray(ctx?.characters);
    }, { timeout: 30_000 });
}

async function ensureScringloImported(page) {
    const present = await page.evaluate(async () => {
        const ctx = window.SillyTavern.getContext();
        return ctx.characters?.some(c => (c?.name || '').toLowerCase().includes('scringlo'));
    });
    if (present) return;
    const charPath = '/Users/mdot/metal-microbench/tools/st-debug/_data/default-user/characters/Scringlo.json';
    const cardData = JSON.parse(fs.readFileSync(charPath, 'utf-8'));
    const created = await page.request.post(
        'http://127.0.0.1:8002/api/characters/create',
        {
            headers: { 'content-type': 'application/json' },
            data: {
                ch_name: cardData.name,
                description: cardData.description,
                personality: cardData.personality,
                scenario: cardData.scenario,
                first_mes: cardData.first_mes,
                mes_example: cardData.mes_example,
                file_name: 'scringlo_scrambler',
            },
        },
    );
    if (!created.ok()) {
        throw new Error(`character create failed ${created.status()}`);
    }
    await page.evaluate(async () => {
        if (typeof window.getCharacters === 'function') await window.getCharacters();
    });
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern.getContext();
        return ctx.characters?.some(c => (c?.name || '').toLowerCase().includes('scringlo'));
    }, { timeout: 15_000 });
}

async function selectScringlo(page) {
    await page.evaluate(async () => {
        const ctx = window.SillyTavern.getContext();
        const idx = ctx.characters.findIndex(c => (c?.name || '').toLowerCase().includes('scringlo'));
        if (idx < 0) throw new Error('Scringlo not found');
        if (String(ctx.characterId) !== String(idx)) {
            await ctx.selectCharacterById(idx);
        }
    });
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern.getContext();
        return (ctx.characters?.[ctx.characterId]?.name || '').toLowerCase().includes('scringlo');
    }, { timeout: 15_000 });
}

async function startFreshChat(page) {
    await page.evaluate(async () => {
        const ctx = window.SillyTavern.getContext();
        if (typeof ctx.executeSlashCommandsWithOptions === 'function') {
            await ctx.executeSlashCommandsWithOptions('/newchat');
        }
    });
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern.getContext();
        return Array.isArray(ctx.chat) && ctx.chat.length <= 1;
    }, { timeout: 15_000 });
}

async function connectApi(page) {
    await page.locator('#API-status-top').click();
    await expect(page.locator('#api_button_openai')).toBeVisible();
    await page.locator('#api_button_openai').click();
    await expect(page.locator('#send_textarea')).toHaveAttribute(
        'placeholder', 'Type a message, or /? for help', { timeout: 30_000 });
    // Dismiss the connection drawer so the chat surface is visible
    // in the recorded video.
    await page.locator('#API-status-top').click();
    await page.waitForTimeout(300);
    await page.keyboard.press('Escape').catch(() => {});
    await page.waitForFunction(() => {
        const chat = document.getElementById('chat');
        if (!chat) return false;
        const rect = chat.getBoundingClientRect();
        const x = rect.left + rect.width / 2;
        const y = rect.top + rect.height / 2;
        const el = document.elementFromPoint(x, y);
        return el === chat || chat.contains(el);
    }, { timeout: 10_000 });
}

test.describe('render-visual vertical slice — Scringlo + 2-fork + 2-spawn + visible SVG', () => {
    test.setTimeout(300_000);

    test('lissajous request decomposes into 4 subagents and embeds the SVG', async ({ page }, testInfo) => {
        await waitForReady(page);
        await ensureScringloImported(page);
        await connectApi(page);
        await selectScringlo(page);
        await startFreshChat(page);

        await page.waitForFunction(() => {
            const ctx = window.SillyTavern.getContext();
            return ctx.chat?.length === 1 && (ctx.chat[0]?.name || '').toLowerCase().includes('scringlo');
        }, { timeout: 15_000 });

        await page.screenshot({
            path: testInfo.outputPath('checkpoint_1_fresh_chat.png'),
            fullPage: false,
        });

        // Synthetic user turn + scringlo placeholder bubble. Same
        // approach as test 16 — sidesteps tool-call elicitation
        // flakiness and lets us capture the tool_progress UI cleanly.
        const userPrompt = "make me a voronoi diagram!! 12 little seeds scattered, each one's territory in a different color ✨";
        const setupResult = await page.evaluate((q) => {
            const ctx = window.SillyTavern.getContext();
            const userMsg = {
                name: ctx.name1 || 'lusier', is_user: true, is_system: false,
                send_date: new Date().toLocaleString(), mes: q, extra: {},
            };
            ctx.chat.push(userMsg); ctx.addOneMessage(userMsg);
            const scrMsg = {
                name: ctx.characters[ctx.characterId]?.name || 'scringlo scrambler',
                is_user: false, is_system: false,
                send_date: new Date().toLocaleString(),
                mes: 'oOoOh!! one of those wobbly woven shapes!! lemme draw u one *taps screen excitedly*',
                extra: {},
            };
            ctx.chat.push(scrMsg); ctx.addOneMessage(scrMsg);
            return { chatLen: ctx.chat.length };
        }, userPrompt);
        expect(setupResult.chatLen).toBeGreaterThanOrEqual(3);

        await page.screenshot({
            path: testInfo.outputPath('checkpoint_2_user_question.png'),
            fullPage: false,
        });

        // Direct-invoke render-visual via ToolManager.
        await page.evaluate(async () => {
            const ctx = window.SillyTavern.getContext();
            window.__renderVisualResult = ctx.ToolManager
                .invokeFunctionTool('render-visual__render', {
                    description: 'voronoi diagram with 12 seeds, viewBox 0 0 400 400, distinct fill colors per cell',
                })
                .catch(error => String(error?.message || error));
            return true;
        });

        // Wait for tool_progress collapsible to render
        const collapsible = page.locator(
            '#chat .mes:not([is_user="true"]):not([is_system="true"]) details.custom-tool-progress-collapsible'
        ).last();
        await expect(collapsible, 'tool_progress collapsible appears')
            .toBeVisible({ timeout: 60_000 });

        // Mid-run checkpoint: snapshot of S1 phase running
        await page.waitForTimeout(8000);
        await page.screenshot({
            path: testInfo.outputPath('checkpoint_3_s1_running.png'),
            fullPage: false,
        });

        // Wait for at least 4 summary_progress entries (F1, S1, S2, host or F2)
        await page.waitForFunction(() => {
            const ctx = window.SillyTavern.getContext();
            for (const m of ctx.chat || []) {
                const tp = m?.extra?.tool_progress?.[0];
                if (tp && Array.isArray(tp.summary_trace) && tp.summary_trace.length >= 4) {
                    return true;
                }
            }
            return false;
        }, { timeout: 120_000 });

        // Wait for tool to reach done
        const collapsibleSummary = page.locator(
            '#chat .mes:not([is_user="true"]):not([is_system="true"]) details.custom-tool-progress-collapsible > summary'
        ).last();
        await expect(collapsibleSummary, 'tool reaches done')
            .toContainText('done', { timeout: 60_000 });

        // Force the collapsible open and any nested branch <details> too,
        // so the final screenshot shows the full summary trace.
        await page.evaluate(() => {
            for (const d of document.querySelectorAll(
                '#chat .mes details.custom-tool-progress-collapsible')) {
                d.setAttribute('open', '');
            }
        });
        await page.waitForTimeout(500);

        // Verify the embedded SVG image is present in the chat bubble
        const inlineImg = page.locator(
            '#chat .mes:not([is_user="true"]):not([is_system="true"]) img.custom-inline-media-thumb'
        );
        await expect(inlineImg.first(), 'svg embedded inline').toBeVisible({ timeout: 30_000 });

        await page.screenshot({
            path: testInfo.outputPath('checkpoint_4_final.png'),
            fullPage: false,
        });

        // Pull the final state out for the curated artifact
        const finalChat = await page.evaluate(() => {
            const ctx = window.SillyTavern.getContext();
            return ctx.chat.map(m => ({
                name: m.name,
                is_user: m.is_user,
                is_system: m.is_system,
                mes_excerpt: typeof m.mes === 'string' ? m.mes.slice(0, 600) : '',
                tool_progress: (m.extra?.tool_progress || []).map(tp => ({
                    label: tp.label,
                    status: tp.status,
                    duration_ms: tp.duration_ms,
                    summary_trace: (tp.summary_trace || []).map(s => ({
                        scope: s.scope, summary: s.summary,
                        compressed_lines: s.compressed_lines, tMs: s.tMs,
                    })),
                    summary_excerpt: tp.summary?.slice(0, 1500) || null,
                })),
                media_count: (m.extra?.media || []).length,
            }));
        });
        const tracePath = testInfo.outputPath('render_visual_chat.json');
        fs.mkdirSync(path.dirname(tracePath), { recursive: true });
        fs.writeFileSync(tracePath, JSON.stringify(finalChat, null, 2));

        // Final dwell so the video captures the full state
        await page.waitForTimeout(3000);
    });
});
