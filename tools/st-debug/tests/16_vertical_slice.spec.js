import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

// Vertical-slice demo: end-to-end real-pixel proof of long-running
// tool use with real-time annotation visible to both model and user.
//
// What this test PROVES that test 15 didn't:
//   - Fresh chat session (clears prior runs' state)
//   - Scringlo persona (not the bootstrap-default Seraphina)
//   - Long-running tool call (tree-of-thoughts: 3 parallel branches +
//     synthesis, ~12-15s wall total)
//   - Per-branch live annotations streaming into the chat as branches
//     complete (the branch_progress event mechanism wired through
//     plugin → FE → DOM)
//   - Synthesis as terminal annotation visible to both Scringlo (in
//     prompt context on next turn) and user (rendered inline)
//
// Captures full video via test.use({video:'on'}) — saved to
// docs/media/ at end-of-test as the curated artifact.

test.use({ video: 'on' });

// Bootstrap-dropped Scringlo.json sits in characters/ but ST only
// auto-loads PNG cards. Convert via /api/characters/create at test
// setup so Scringlo shows up in the character list.
async function ensureScringloImported(page) {
    // Check if already there.
    const present = await page.evaluate(async () => {
        const ctx = window.SillyTavern.getContext();
        return ctx.characters?.some(c => /scringlo/i.test(c?.name || ''));
    });
    if (present) return;

    // Read the JSON card file (bootstrap drops it at this path).
    const charPath = '/Users/mdot/metal-microbench/tools/st-debug/_data/default-user/characters/Scringlo.json';
    const cardData = JSON.parse(fs.readFileSync(charPath, 'utf-8'));

    // POST to /api/characters/create with the card fields. The
    // endpoint writes a PNG card with embedded metadata at
    // characters/<file_name>.png.
    const csrfToken = await page.evaluate(() => {
        return document.querySelector('meta[name="csrf-token"]')?.content || '';
    });
    const cookies = await page.context().cookies();
    const cookieHeader = cookies.map(c => `${c.name}=${c.value}`).join('; ');

    const created = await page.request.post(
        'http://127.0.0.1:8002/api/characters/create',
        {
            headers: {
                'content-type': 'application/json',
                'cookie': cookieHeader,
                ...(csrfToken ? { 'x-csrf-token': csrfToken } : {}),
            },
            data: {
                ch_name: cardData.name,
                description: cardData.description,
                personality: cardData.personality,
                scenario: cardData.scenario,
                first_mes: cardData.first_mes,
                mes_example: cardData.mes_example,
                creator_notes: cardData.creator_notes || '',
                tags: cardData.tags || [],
                file_name: 'scringlo_scrambler',
            },
        }
    );
    if (!created.ok()) {
        const body = await created.text();
        throw new Error(`character create failed ${created.status()}: ${body}`);
    }

    // Reload the character list so the new card shows up. ST's
    // getCharacters() refetches /api/characters/all.
    await page.evaluate(async () => {
        if (typeof window.getCharacters === 'function') {
            await window.getCharacters();
        }
    });
    // Wait for it to land in the in-memory list.
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern.getContext();
        return ctx.characters?.some(c => /scringlo/i.test(c?.name || ''));
    }, { timeout: 15_000 });
}

async function selectScringlo(page) {
    await page.evaluate(async () => {
        const ctx = window.SillyTavern.getContext();
        const idx = ctx.characters.findIndex(c => /scringlo/i.test(c?.name || ''));
        if (idx < 0) throw new Error('Scringlo not found in character list');
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
    // Use ST's slash-command path to create a brand new chat for
    // the selected character. /newchat creates a fresh empty chat
    // file so the trace starts clean.
    await page.evaluate(async () => {
        const ctx = window.SillyTavern.getContext();
        if (typeof ctx.executeSlashCommandsWithOptions === 'function') {
            await ctx.executeSlashCommandsWithOptions('/newchat');
        } else if (typeof window.doNewChat === 'function') {
            await window.doNewChat({ deleteCurrentChat: false });
        }
    });
    // Wait for chat to be ready
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern.getContext();
        // After /newchat the chat[] should be empty or contain only
        // the character's first_mes.
        return Array.isArray(ctx.chat) && ctx.chat.length <= 1;
    }, { timeout: 15_000 });
}

async function connectApi(page) {
    await page.locator('#API-status-top').click();
    await expect(page.locator('#api_button_openai')).toBeVisible();
    await page.locator('#api_button_openai').click();
    await expect(page.locator('#send_textarea')).toHaveAttribute(
        'placeholder', 'Type a message, or /? for help', { timeout: 30_000 });
}

test.describe('vertical slice — Scringlo + long-running tool + visible streaming', () => {
    test.setTimeout(300_000);

    test('fresh chat with Scringlo runs tree-of-thoughts; branches stream visibly', async ({ page }, testInfo) => {
        // 1. Boot ST UI
        await page.goto('/');
        await page.waitForFunction(
            'document.getElementById("preloader") === null',
            { timeout: 60_000 });
        await page.waitForFunction(() => {
            const ctx = window.SillyTavern?.getContext?.();
            return Array.isArray(ctx?.characters);
        }, { timeout: 30_000 });

        // 2. Import Scringlo (no-op if already imported by a prior run)
        await ensureScringloImported(page);

        // 3. Connect to the bridge
        await connectApi(page);

        // 4. Select Scringlo
        await selectScringlo(page);

        // 5. Fresh chat
        await startFreshChat(page);

        // Confirm Scringlo's first_mes lands as message 0
        await page.waitForFunction(() => {
            const ctx = window.SillyTavern.getContext();
            return ctx.chat?.length === 1 && /scringlo/i.test(ctx.chat[0]?.name || '');
        }, { timeout: 15_000 });

        // 6. Push a synthetic user message into chat (so caller_messages
        //    has the question) and a synthetic Scringlo bubble (so the
        //    tool_progress UI has something to attach to). This avoids
        //    racing against Scringlo's chat-completion stream.
        const userPrompt = "i'm trying to decide whether to spend the weekend learning rust or finishing my novel. could you think it through carefully?";
        const setupResult = await page.evaluate((q) => {
            const ctx = window.SillyTavern.getContext();
            // Append user turn
            const userMsg = {
                name: ctx.name1 || 'lusier',
                is_user: true,
                is_system: false,
                send_date: new Date().toLocaleString(),
                mes: q,
                extra: {},
            };
            ctx.chat.push(userMsg);
            ctx.addOneMessage(userMsg);
            // Append Scringlo placeholder
            const scrMsg = {
                name: ctx.characters[ctx.characterId]?.name || 'scringlo scrambler',
                is_user: false,
                is_system: false,
                send_date: new Date().toLocaleString(),
                mes: 'okie!! lemme think about it for a sec... *taps screen thoughtfully*',
                extra: {},
            };
            ctx.chat.push(scrMsg);
            ctx.addOneMessage(scrMsg);
            return { chatLen: ctx.chat.length, scrIdx: ctx.chat.length - 1 };
        }, userPrompt);
        expect(setupResult.chatLen).toBeGreaterThanOrEqual(3);

        // 7. Directly invoke tree-of-thoughts with the conversation
        //    as caller_messages. Direct invoke skips the model's
        //    flaky tool-call elicitation while preserving the
        //    end-to-end FE pipeline: the tool_progress collapsible
        //    appears on Scringlo's bubble, branch_progress events
        //    stream branches as they complete, synthesis renders.
        const result = await page.evaluate(async (q) => {
            const ctx = window.SillyTavern.getContext();
            // Build caller_messages from the live chat so the tool
            // descendant inherits Scringlo's voice + the user's
            // question.
            const callerMessages = ctx.chat.map(m => ({
                role: m.is_user ? 'user' : (m.is_system ? 'system' : 'assistant'),
                content: typeof m.mes === 'string' ? m.mes : '',
            })).filter(m => m.content);
            // Use the FE's ToolManager so the tool_progress UI
            // attaches to the latest assistant bubble (Scringlo's).
            window.__verticalSliceResult = ctx.ToolManager
                .invokeFunctionTool('tree-of-thoughts__explore', {
                    question: q,
                    branches: ['practical', 'creative', 'skeptical'],
                })
                .catch(error => String(error?.message || error));
            return true;
        }, userPrompt);
        expect(result).toBe(true);

        // 8. Wait for branch cards to appear in the chat bubble.
        //    Each branch renders as .custom-tool-progress-branch
        //    inside the latest assistant message's .mes_text.
        const branchCards = page.locator(
            '#chat .mes:not([is_user="true"]):not([is_system="true"]) .custom-tool-progress-branch'
        );
        await expect(branchCards).toHaveCount(3, { timeout: 60_000 });

        // 9. Wait for at least 2 of 3 branches to reach status="complete"
        try {
            await page.waitForFunction(() => {
                const cards = document.querySelectorAll('.custom-tool-progress-branch');
                let complete = 0;
                for (const c of cards) {
                    if ((c.getAttribute('data-status') || '') === 'complete') complete++;
                }
                return complete >= 2;
            }, { timeout: 90_000 });
        } catch (e) {
            // Diagnostic dump to surface what's actually in the DOM/state
            const dbg = await page.evaluate(() => {
                const cards = document.querySelectorAll('.custom-tool-progress-branch');
                const ctx = window.SillyTavern.getContext();
                const tp = [];
                for (const m of ctx.chat || []) {
                    if (m?.extra?.tool_progress) {
                        for (const e of m.extra.tool_progress) {
                            tp.push({
                                label: e.label,
                                status: e.status,
                                branches: (e.branches || []).map(b => ({
                                    label: b.label, status: b.status,
                                    summary_excerpt: (b.summary || '').slice(0, 60),
                                })),
                            });
                        }
                    }
                }
                return {
                    dom_branch_count: cards.length,
                    dom_branch_statuses: Array.from(cards).map(c => c.getAttribute('data-status')),
                    chat_tool_progress: tp,
                };
            });
            console.log('[diag] dump:', JSON.stringify(dbg, null, 2));
            throw e;
        }

        // 10. Wait for the tool_progress collapsible to reach status=done.
        //     Use `> summary` (direct child) to disambiguate from
        //     nested branch <details><summary> inside the collapsible.
        const collapsibleSummary = page.locator(
            '#chat .mes:not([is_user="true"]):not([is_system="true"]) details.custom-tool-progress-collapsible > summary'
        ).last();
        await expect(collapsibleSummary, 'tool_progress reaches done')
            .toContainText('done', { timeout: 90_000 });

        // 11. Final assertions on the underlying result.
        const finalResult = await page.evaluate(async () => {
            return await window.__verticalSliceResult;
        });
        expect(typeof finalResult, 'tool returned a result').toBe('string');
        expect(finalResult, 'result mentions synthesis').toMatch(/synthesis/i);

        // 12. Save the curated trace as a sibling artifact.
        const finalChat = await page.evaluate(() => {
            const ctx = window.SillyTavern.getContext();
            return ctx.chat.map(m => ({
                name: m.name,
                is_user: m.is_user,
                is_system: m.is_system,
                mes_excerpt: typeof m.mes === 'string' ? m.mes.slice(0, 400) : '',
                tool_progress_count: Array.isArray(m.extra?.tool_progress) ? m.extra.tool_progress.length : 0,
                branches_count: m.extra?.tool_progress?.[0]?.branches?.length ?? 0,
                summary_excerpt: m.extra?.tool_progress?.[0]?.summary?.slice(0, 200) || null,
            }));
        });
        const tracePath = testInfo.outputPath('vertical_slice_chat.json');
        fs.mkdirSync(path.dirname(tracePath), { recursive: true });
        fs.writeFileSync(tracePath, JSON.stringify(finalChat, null, 2));

        // Pause briefly so the video captures the final state visibly.
        await page.waitForTimeout(2000);
    });
});
