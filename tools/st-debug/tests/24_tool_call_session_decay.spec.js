// Stress repro: "empty responses after ~20-30 tool-call sessions on the
// same persistent bridge process." User report 2026-05-10. Symptom is
// that after many tool-call-heavy chat turns in a row on one ST chat,
// the model starts producing turns with empty content (and ST renders
// them as blank assistant bubbles).
//
// We drive dicemother through N successive "next encounter" prompts on
// ONE chat (same bridge process, same ST session), capture every
// assistant turn, and watch for:
//   - empty .mes (the smoking gun)
//   - growing length of content+reasoning in history (the proposed
//     mechanism: accumulating <|channel>thought\n blocks push the model
//     into thinking-only emissions)
//   - tool_call render inconsistency (some turns render with progress
//     collapsibles + structured tool_invocations, others don't)
//
// Saves a structured per-turn record so the failure pattern is
// observable post-hoc without re-running.

import { test, expect } from '@playwright/test';
import {
    loadAndConnect, sendAndObserve,
    selectCharacterByClick, freshChatByClick,
} from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

const N_TURNS = 30;
const PER_TURN_PROMPTS = [
    'i kick the next door open and step in. roll the encounter.',
    'i creep down the corridor. anything in here?',
    'i poke at the rubble. anything underneath? roll.',
    'i listen at the next door. then i push through. roll.',
    'i sprint to the far side of the cavern. what greets me?',
    'i strike a match and look around. roll what i see.',
    'i descend the spiral staircase one floor. roll.',
    'i open the chest at the foot of the stairs. roll loot.',
];

test.describe('many-tool-call session decay', () => {
    test.setTimeout(40 * 60 * 1000);

    test('30 successive encounter rolls on one chat — watch for empty .mes / inconsistent render', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        await selectCharacterByClick(page, 'dicemother');
        await freshChatByClick(page);

        const turnRecords = [];
        for (let i = 0; i < N_TURNS; i++) {
            const prompt = PER_TURN_PROMPTS[i % PER_TURN_PROMPTS.length];
            console.log(`--- turn ${i + 1}/${N_TURNS}: ${JSON.stringify(prompt)} ---`);
            const t0 = Date.now();
            let record = null;
            try {
                record = await sendAndObserve(page, prompt, {
                    timeoutMs: 4 * 60 * 1000,
                });
            } catch (e) {
                record = { error: String(e?.message || e).slice(0, 300) };
            }
            const elapsed_s = ((Date.now() - t0) / 1000).toFixed(1);
            // Pull what landed on the chat array for THIS turn.
            const lastTurn = await page.evaluate(() => {
                const ctx = window.SillyTavern.getContext();
                const chat = ctx.chat || [];
                const m = chat[chat.length - 1] || {};
                return {
                    role: m.is_user ? 'user' : (m.is_system ? 'system' : 'assistant'),
                    mes_full_len: (m.mes || '').length,
                    mes_head: (m.mes || '').slice(0, 200),
                    has_tool_invocations: ((m.extra?.tool_invocations || []).length > 0),
                    tool_invocations: (m.extra?.tool_invocations || []).map(t => ({
                        name: t.displayName || t.name,
                    })),
                    has_tool_progress: ((m.extra?.tool_progress || []).length > 0),
                    tool_progress: (m.extra?.tool_progress || []).map(t => ({
                        label: t.label, status: t.status,
                        summary_head: (t.summary || '').slice(0, 80),
                    })),
                    has_reasoning: !!m.extra?.reasoning,
                    reasoning_len: (m.extra?.reasoning || '').length,
                };
            });
            // Also pull the total chat[] state so we can chart history growth.
            const chatStats = await page.evaluate(() => {
                const ctx = window.SillyTavern.getContext();
                const chat = ctx.chat || [];
                return {
                    n_turns: chat.length,
                    total_mes_chars: chat.reduce(
                        (acc, m) => acc + (m.mes || '').length, 0),
                    total_reasoning_chars: chat.reduce(
                        (acc, m) => acc + (m.extra?.reasoning || '').length, 0),
                    n_turns_with_tool_invocations: chat.filter(
                        m => (m.extra?.tool_invocations || []).length > 0).length,
                };
            });

            turnRecords.push({
                i: i + 1,
                prompt,
                elapsed_s,
                finishState: record?.finishState ?? null,
                error: record?.error ?? null,
                last_turn: lastTurn,
                chat_stats: chatStats,
            });
            console.log(
                `  ${elapsed_s}s · ` +
                `mes=${lastTurn.mes_full_len}ch · ` +
                `tool_inv=${lastTurn.has_tool_invocations ? '✓' : '×'} · ` +
                `tool_prog=${lastTurn.has_tool_progress ? '✓' : '×'} · ` +
                `reasoning=${lastTurn.reasoning_len}ch · ` +
                `total_history=${chatStats.total_mes_chars}+${chatStats.total_reasoning_chars}ch`);

            // Bail out early once we've collected enough evidence of the
            // failure (3 empty turns is the user's "repeated empty responses").
            const recentEmpties = turnRecords.slice(-5).filter(
                t => t.last_turn.mes_full_len === 0 && t.last_turn.role === 'assistant');
            if (recentEmpties.length >= 3) {
                console.log(`>>> 3 of last 5 turns produced empty .mes; bailing early at turn ${i + 1}`);
                break;
            }
        }

        await page.screenshot({
            path: testInfo.outputPath('99_after_stress.png'),
            fullPage: true,
        });
        fs.writeFileSync(testInfo.outputPath('decay_trace.json'),
            JSON.stringify(turnRecords, null, 2));

        // Summary: where did the empty turns appear?
        const emptyTurns = turnRecords.filter(
            t => t.last_turn.role === 'assistant' && t.last_turn.mes_full_len === 0);
        const renderInconsistencies = turnRecords.filter(
            t => t.last_turn.has_tool_progress !== t.last_turn.has_tool_invocations);
        console.log('=== decay trace summary ===');
        console.log(`turns run: ${turnRecords.length}`);
        console.log(`empty .mes turns: ${emptyTurns.length} (at indices ${emptyTurns.map(t => t.i).join(', ')})`);
        console.log(`turns with mismatched tool_inv/tool_progress: ${renderInconsistencies.length}`);
        for (const t of renderInconsistencies.slice(0, 10)) {
            console.log(`  turn ${t.i}: tool_inv=${t.last_turn.has_tool_invocations} tool_prog=${t.last_turn.has_tool_progress}`);
        }

        // Liveness only — we don't fail if the bug shows; we capture it.
        expect(turnRecords.length).toBeGreaterThan(0);
    });
});
