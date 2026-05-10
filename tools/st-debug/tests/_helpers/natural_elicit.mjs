// Honest end-to-end elicitation: user types text into the chat
// textarea, clicks send, the model decides whether/how to call a
// tool. No direct-invoke shims, no synthetic ctx.chat.push, no
// window.__* result-stash. Just text into LM in harness, then
// observe what the harness produced.
//
// Returns a structured record of what actually happened, including:
//   - assistantText: the model's natural-language response (may be
//     empty if the response was a pure tool_call)
//   - toolCallsEmitted: array of {qualified_name, args} the model
//     actually emitted, in order
//   - toolResults: per tool_call, the action callback's return value
//     (typically the tool_progress entry's summary string)
//   - elapsedMs: time from send-click to terminal state
//   - finishState: 'completed' | 'tool_handled' | 'no_response' |
//     'timeout'
//
// The harness is used both by:
//   - vertical slice tests (assert specific tools fired)
//   - elicitation reliability studies (run N trials, count rate)

import { expect } from '@playwright/test';

/**
 * Send a user message through the actual chat UI and wait for the
 * model's response to complete (including any tool calls it chose
 * to make). Does NOT directly invoke any tool — the model's
 * decision is what we measure.
 *
 * @param {import('@playwright/test').Page} page
 * @param {string} userPrompt - message text
 * @param {object} [opts]
 * @param {number} [opts.timeoutMs=120000] - overall budget
 * @returns {Promise<object>} elicitation record
 */
export async function naturalElicit(page, userPrompt, opts = {}) {
    const timeoutMs = opts.timeoutMs ?? 120_000;

    // Capture chat length BEFORE send so we can identify the new
    // assistant turn(s) that arrive after.
    const baselineLen = await page.evaluate(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return ctx?.chat?.length ?? 0;
    });

    // Hook tool-invocation events. ST's eventSource emits
    // event_types.MESSAGE_RECEIVED + others. We tap the FE
    // toolcards extension's ToolManager invocations by snapshotting
    // tool_invocations records on the assistant turn after settle.
    await page.evaluate(() => {
        window.__elicitProbe = {
            startedAt: performance.now(),
            sendButClicks: 0,
            toolCallsObserved: [],
            errors: [],
        };
    });

    const sendT0 = await page.evaluate(() => performance.now());
    await page.locator('#send_textarea').fill(userPrompt);
    await page.locator('#send_but').click();

    // Wait for the response to settle. We watch for the send button
    // to re-enable (signals generation finished) AND any pending
    // tool_progress entries to reach a terminal state.
    const settleResult = await page.waitForFunction((args) => {
        const ctx = window.SillyTavern?.getContext?.();
        if (!ctx) return false;
        const chat = ctx.chat || [];
        // The last assistant message is the one we want
        const lastIdx = chat.length - 1;
        if (lastIdx < args.baseline) return false;
        const last = chat[lastIdx];
        if (!last || last.is_user) return false;

        // Generation must have finished (not still streaming)
        const stop = document.querySelector('#mes_stop');
        const generating = stop && stop.offsetParent !== null;
        if (generating) return false;

        // If the assistant turn has tool_progress entries, every one
        // must be in a terminal state (done/failed/cancelled)
        const tp = (last.extra?.tool_progress || []);
        for (const entry of tp) {
            if (entry.status !== 'done' &&
                entry.status !== 'failed' &&
                entry.status !== 'cancelled') {
                return false;
            }
        }

        return {
            lastIdx,
            mes: typeof last.mes === 'string' ? last.mes : '',
            tool_progress: tp.map(e => ({
                label: e.label,
                status: e.status,
                duration_ms: e.duration_ms,
                summary: e.summary || null,
                summary_trace: (e.summary_trace || []).map(s => ({
                    scope: s.scope, summary: s.summary,
                })),
            })),
            tool_invocations: (last.extra?.tool_invocations || []).map(i => ({
                name: i.displayName || i.name,
                parameters_raw: i.parameters,
                result_raw: i.result,
            })),
        };
    }, { baseline: baselineLen }, { timeout: timeoutMs });

    const settle = await settleResult.jsonValue();
    const elapsedMs = (await page.evaluate(() => performance.now())) - sendT0;

    // Distill into the elicitation record
    let finishState;
    if (settle.tool_invocations.length > 0 || settle.tool_progress.length > 0) {
        finishState = 'tool_handled';
    } else if (settle.mes && settle.mes.trim().length > 0) {
        finishState = 'completed';
    } else {
        finishState = 'no_response';
    }

    return {
        userPrompt,
        baselineLen,
        finishState,
        elapsedMs,
        assistantText: settle.mes,
        assistantTurnIdx: settle.lastIdx,
        toolCallsEmitted: settle.tool_invocations,
        toolProgress: settle.tool_progress,
    };
}

/**
 * Run an honest end-to-end test. The harness does the BARE MINIMUM
 * setup needed to reach a usable chat surface, then drives the
 * actual UI (textarea + send button) like a real user.
 *
 * Notably we do NOT do:
 *   - programmatic /newchat (ctx.executeSlashCommandsWithOptions
 *     puts ST into a state where the user-typed message gets
 *     dropped from the next chat-completion request — found
 *     2026-05-10 by capturing the actual fetch body and noticing
 *     `messages` had no user role at all)
 *   - programmatic character selection (ctx.selectCharacterById
 *     similarly mutates ST internal state at a level that bypasses
 *     the UI's own prompt-builder triggers)
 *
 * If you need a specific character or a fresh chat, drive those
 * via the actual UI elements (click the character avatar, click
 * the "new chat" button) so ST's normal flow runs.
 */
export async function honestEnd2End(page, userPrompt, opts = {}) {
    await loadHarness(page);
    if (opts.personaSubstr) {
        await ensurePersona(page, opts.personaSubstr);
        // We can import the character so it's available, but we
        // don't programmatically select it via internal APIs.
        // The default character is whichever ST has loaded.
    }
    return await naturalElicit(page, userPrompt, opts);
}

// ─────────────────────────────────────────────────────────────────
// Standard harness setup, no shims, no direct invokes.
// ─────────────────────────────────────────────────────────────────

import fs from 'node:fs';

export async function loadHarness(page) {
    await page.goto('/');
    await page.waitForFunction(
        'document.getElementById("preloader") === null',
        { timeout: 60_000 });
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return Array.isArray(ctx?.characters);
    }, { timeout: 30_000 });

    // Connect to API + dismiss the connection drawer so the chat
    // surface is visible in any recorded video.
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

export async function ensurePersona(page, substr) {
    const present = await page.evaluate((s) => {
        const ctx = window.SillyTavern.getContext();
        return ctx.characters?.some(c => new RegExp(s, 'i').test(c?.name || ''));
    }, substr);
    if (present) return;

    // Bootstrap drops Scringlo.json. We import via the API.
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
    await page.waitForFunction((s) => {
        const ctx = window.SillyTavern.getContext();
        return ctx.characters?.some(c => new RegExp(s, 'i').test(c?.name || ''));
    }, substr, { timeout: 15_000 });
}

export async function selectPersona(page, substr) {
    await page.evaluate(async (s) => {
        const ctx = window.SillyTavern.getContext();
        const idx = ctx.characters.findIndex(c => new RegExp(s, 'i').test(c?.name || ''));
        if (idx < 0) throw new Error(`persona matching /${s}/i not found`);
        if (String(ctx.characterId) !== String(idx)) {
            await ctx.selectCharacterById(idx);
        }
    }, substr);
    await page.waitForFunction((s) => {
        const ctx = window.SillyTavern.getContext();
        return new RegExp(s, 'i').test(ctx.characters?.[ctx.characterId]?.name || '');
    }, substr, { timeout: 15_000 });
}

export async function freshChat(page) {
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
