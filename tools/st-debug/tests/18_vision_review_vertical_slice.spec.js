import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

// Vertical slice for the vision-review card. Closes the visual-
// oversight gap flagged in test 17: validator-level invariants
// (S2's checklist) faithfully surfaced descendant errors but
// didn't catch "is the visual the shape the user asked for?"
//
// Setup: scringlo is asked to look at her own prior demo recording
// and verify whether it actually shows the claimed feature. She
// invokes vision-review on the curated test 17 failure video.
// The card fires N parallel multimodal LLM calls (one per
// extracted frame), each with the shared "visual transcript
// reviewer" prefix, then a synthesis call to score the claim.
//
// Expected outcome: PASS=false. The video shows a tessellation,
// not a Lissajous curve. Vision-review correctly flags this in
// scringlo's chat as a system message + collapsible with per-
// frame descriptions.

test.use({ video: 'on' });

async function ensureScringloImported(page) {
    const present = await page.evaluate(async () => {
        const ctx = window.SillyTavern.getContext();
        return ctx.characters?.some(c => /scringlo/i.test(c?.name || ''));
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
    if (!created.ok()) throw new Error(`character create failed ${created.status()}`);
    await page.evaluate(async () => {
        if (typeof window.getCharacters === 'function') await window.getCharacters();
    });
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern.getContext();
        return ctx.characters?.some(c => /scringlo/i.test(c?.name || ''));
    }, { timeout: 15_000 });
}

async function selectScringlo(page) {
    await page.evaluate(async () => {
        const ctx = window.SillyTavern.getContext();
        const idx = ctx.characters.findIndex(c => /scringlo/i.test(c?.name || ''));
        if (idx < 0) throw new Error('Scringlo not found');
        if (String(ctx.characterId) !== String(idx)) {
            await ctx.selectCharacterById(idx);
        }
    });
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern.getContext();
        return /scringlo/i.test(ctx.characters?.[ctx.characterId]?.name || '');
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

test.describe('vision-review vertical slice — scringlo audits her own test 17 recording', () => {
    test.setTimeout(300_000);

    test('vision-review correctly identifies that test 17 video does NOT show a Lissajous curve', async ({ page }, testInfo) => {
        await page.goto('/');
        await page.waitForFunction(
            'document.getElementById("preloader") === null',
            { timeout: 60_000 });
        await page.waitForFunction(() => Array.isArray(
            window.SillyTavern?.getContext?.()?.characters), { timeout: 30_000 });

        await ensureScringloImported(page);
        await connectApi(page);
        await selectScringlo(page);
        await startFreshChat(page);

        await page.waitForFunction(() => {
            const ctx = window.SillyTavern.getContext();
            return ctx.chat?.length === 1;
        }, { timeout: 15_000 });

        await page.screenshot({
            path: testInfo.outputPath('checkpoint_1_fresh_chat.png'),
            fullPage: false,
        });

        // Synthetic user message asking scringlo to audit her prior demo
        const userPrompt = "hey can you look back at the recording from our lissajous demo earlier? did you actually draw a real lissajous curve, or do you think u made a mistake somewhere?";
        const setupResult = await page.evaluate((q) => {
            const ctx = window.SillyTavern.getContext();
            ctx.chat.push({ name: ctx.name1 || 'lusier', is_user: true, is_system: false,
                send_date: new Date().toLocaleString(), mes: q, extra: {} });
            ctx.addOneMessage(ctx.chat[ctx.chat.length - 1]);
            ctx.chat.push({
                name: ctx.characters[ctx.characterId]?.name || 'scringlo scrambler',
                is_user: false, is_system: false,
                send_date: new Date().toLocaleString(),
                mes: 'oOoOh!! good idea!! lemme go look at the recording with my fresh eyes... *peers at the screen* 🔍',
                extra: {} });
            ctx.addOneMessage(ctx.chat[ctx.chat.length - 1]);
            return ctx.chat.length;
        }, userPrompt);
        expect(setupResult).toBeGreaterThanOrEqual(3);

        await page.screenshot({
            path: testInfo.outputPath('checkpoint_2_user_question.png'),
            fullPage: false,
        });

        // Direct-invoke vision-review pointed at the curated test 17 failure video
        await page.evaluate(async () => {
            const ctx = window.SillyTavern.getContext();
            window.__visionReviewResult = ctx.ToolManager
                .invokeFunctionTool('vision-review__review', {
                    video_path: '/Users/mdot/metal-microbench/docs/media/2026-05-09_test17_lissajous_failure_surfaced_9b26e22.webm',
                    claim: 'this recording shows scringlo using a tool to draw a Lissajous curve, and the final embedded image is a Lissajous 3:5 frequency-ratio woven curve',
                    num_frames: 5,
                })
                .catch(error => String(error?.message || error));
            return true;
        });

        const collapsible = page.locator(
            '#chat .mes:not([is_user="true"]):not([is_system="true"]) details.custom-tool-progress-collapsible'
        ).last();
        await expect(collapsible, 'tool_progress collapsible appears')
            .toBeVisible({ timeout: 60_000 });

        // Mid-run screenshot during per-frame review
        await page.waitForTimeout(8000);
        await page.screenshot({
            path: testInfo.outputPath('checkpoint_3_mid_review.png'),
            fullPage: false,
        });

        // Wait for at least 5 summary_progress entries (5 frames + verdict)
        await page.waitForFunction(() => {
            const ctx = window.SillyTavern.getContext();
            for (const m of ctx.chat || []) {
                const tp = m?.extra?.tool_progress?.[0];
                if (tp && Array.isArray(tp.summary_trace) && tp.summary_trace.length >= 5) {
                    return true;
                }
            }
            return false;
        }, { timeout: 180_000 });

        // Wait for tool to reach done
        const collapsibleSummary = page.locator(
            '#chat .mes:not([is_user="true"]):not([is_system="true"]) details.custom-tool-progress-collapsible > summary'
        ).last();
        await expect(collapsibleSummary, 'tool reaches done')
            .toContainText('done', { timeout: 60_000 });

        // Force the collapsible open so summary trace is visible in the final screenshot
        await page.evaluate(() => {
            for (const d of document.querySelectorAll(
                '#chat .mes details.custom-tool-progress-collapsible')) {
                d.setAttribute('open', '');
            }
        });
        await page.waitForTimeout(500);

        await page.screenshot({
            path: testInfo.outputPath('checkpoint_4_final.png'),
            fullPage: false,
        });

        // Pull final state for the curated artifact
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
                        scope: s.scope,
                        summary: s.summary,
                        compressed_lines: s.compressed_lines,
                        tMs: s.tMs,
                    })),
                    summary_excerpt: tp.summary?.slice(0, 1500) || null,
                })),
            }));
        });
        const tracePath = testInfo.outputPath('vision_review_chat.json');
        fs.writeFileSync(tracePath, JSON.stringify(finalChat, null, 2));

        // Pull the actual review result and assert it correctly reports FAIL
        const reviewResult = await page.evaluate(async () => {
            return await window.__visionReviewResult;
        });
        // The result is a stringified summary (the action callback returns
        // result.summary). Just check the verdict line is FAIL.
        expect(typeof reviewResult, 'review returned a string').toBe('string');
        expect(reviewResult, 'verdict reported as FAIL')
            .toMatch(/verdict.*FAIL/i);

        await page.waitForTimeout(3000);
    });
});
