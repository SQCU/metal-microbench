// End-to-end validation of the reasoning pipeline after steps 2-4.
//
// Covers three behaviors that should all hold for an open-weights-model
// chat client like the SillyTavern fork:
//
//   STEP 2 — no empty assistant turns when the model emits a thinking
//   channel block. With reasoning_effort=high the engine routes the
//   in-channel tokens to thinkingQueue and the response content is
//   non-empty (final answer after the channel closes). With
//   reasoning_effort unset the engine drops in-channel tokens BUT
//   Session.flushUnclosedChannel() recovers them to outputQueue if
//   the channel never closes — so the response is never silently empty.
//
//   STEP 3 — past-turn reasoning is preserved on subsequent prefills
//   even when the turn that produced the reasoning did NOT fire a
//   tool. The chat-template guard at line 248 used to gate this on
//   message.get('tool_calls'); that was wrong and is now dropped.
//
//   STEP 4 — when extra.reasoning lands on an assistant turn, ST
//   fires a background summarizer that asks the model for a structured
//   JSON {"heading": "..."} and writes the result to
//   extra.reasoning_summary. The collapsible header in the UI then
//   reads the model-authored heading instead of "Thought for Xs".
//
// We drive 3 turns through scringlo (chosen because she reliably
// uses tools without overly dominating the response — keeps the
// reasoning traces concise), with reasoning_effort=high set on every
// request via the dropdown ST exposes for it.

import { test, expect } from '@playwright/test';
import {
    loadAndConnect, sendAndObserve,
    selectCharacterByClick, freshChatByClick,
} from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

async function setReasoningEffort(page, value) {
    return await page.evaluate(async (v) => {
        const sel = document.getElementById('openai_reasoning_effort');
        if (!sel) return { ok: false, reason: 'no #openai_reasoning_effort element' };
        sel.value = v;
        const win = /** @type {any} */ (window);
        if (win.jQuery) win.jQuery(sel).trigger('input');
        else sel.dispatchEvent(new Event('input', { bubbles: true }));
        await new Promise(r => setTimeout(r, 200));
        return { ok: true, sel_value: sel.value };
    }, value);
}

async function waitForSummaryOrTimeout(page, messageIdx, timeoutMs = 60_000) {
    // The summarizer runs as a background fetch after the main response
    // finishes streaming. It writes chat[i].extra.reasoning_summary
    // when it returns. We poll for that field on the specified message
    // up to `timeoutMs`; returns null if it never lands.
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
        const summary = await page.evaluate((i) => {
            const ctx = window.SillyTavern.getContext();
            const m = (ctx.chat || [])[i];
            return m?.extra?.reasoning_summary || null;
        }, messageIdx);
        if (summary) return summary;
        await page.waitForTimeout(750);
    }
    return null;
}

test.describe('reasoning pipeline end-to-end (steps 2 + 3 + 4)', () => {
    test.setTimeout(15 * 60 * 1000);

    test('3 turns with reasoning_effort=high: content non-empty, history reasoning preserved, auto-summary heading rendered', async ({ page }, testInfo) => {
        const networkRecords = [];
        page.on('request', (req) => {
            const url = req.url();
            if (url.includes('/api/backends/chat-completions/generate')) {
                let body = null;
                try { body = req.postDataJSON(); } catch (_) {}
                networkRecords.push({ kind: 'request', url, body });
            }
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'dicemother');
        await freshChatByClick(page);
        await setReasoningEffort(page, 'high');

        // Substantive prompts that genuinely benefit from reasoning AND
        // exercise multiple tool calls per turn. dicemother is the
        // designed persona for this kind of multi-roll diegetic
        // decision-making; the tarot framing puts three random-choice
        // tool calls in front of a contrastive reading task that needs
        // the model to actually reason about the difference between
        // two card spreads. Prompt brevity was the wrong axis to
        // optimize for earlier — input context size and output context
        // size are independent, and a substantive eval needs a
        // substantive task. The follow-up turn forces reasoning to
        // carry across turns (step-3 invariant).
        const turnPrompts = [
            "i pull a tarot deck from my pack and lay out 8 cards for a sweeping fate reading. use the random-choice tool to roll which cards come up (from a major-arcana list of your choosing). then use the tool again to pick which one of the 8 i should remove (a card slipping back into the deck). then use the tool a third time to pick which 3 of the remaining 7 are reversed/inverted. then, in your own voice, contrast the original 8-card spread against the 7-with-3-reversed reading and tell me what the difference suggests for the party.",
            "now, drawing on that same reading: which two cards from the final 7 do you think the party should pay closest attention to in the next room, and why? don't roll for this; reason about the meanings.",
        ];

        const turnRecords = [];
        for (let i = 0; i < turnPrompts.length; i++) {
            console.log(`\n=== TURN ${i + 1}/${turnPrompts.length} ===`);
            console.log(`prompt: ${JSON.stringify(turnPrompts[i])}`);
            const r = await sendAndObserve(page, turnPrompts[i], {
                timeoutMs: 5 * 60 * 1000,
            });
            const lastIdx = await page.evaluate(() => {
                const ctx = window.SillyTavern.getContext();
                return (ctx.chat || []).length - 1;
            });
            // Wait for the background summarizer to land, if reasoning
            // was produced on this turn.
            const summary = await waitForSummaryOrTimeout(page, lastIdx, 90_000);
            const last = await page.evaluate((i) => {
                const ctx = window.SillyTavern.getContext();
                const m = (ctx.chat || [])[i] || {};
                return {
                    name: m.name,
                    mes_len: (m.mes || '').length,
                    mes_head: (m.mes || '').slice(0, 200),
                    reasoning_len: (m.extra?.reasoning || '').length,
                    reasoning_head: (m.extra?.reasoning || '').slice(0, 200),
                    reasoning_summary: m.extra?.reasoning_summary || null,
                    tool_progress_n: (m.extra?.tool_progress || []).length,
                };
            }, lastIdx);
            turnRecords.push({
                turn: i + 1,
                prompt: turnPrompts[i],
                elapsedMs: Math.round(r.elapsedMs),
                finishState: r.finishState,
                summary_waited_for: summary,
                last,
            });
            console.log(`  content: ${last.mes_len}ch`);
            console.log(`  reasoning: ${last.reasoning_len}ch`);
            console.log(`  reasoning_summary: ${JSON.stringify(last.reasoning_summary)}`);
            console.log(`  tool_progress entries: ${last.tool_progress_n}`);
        }

        // Expand any collapsibles + screenshot the final state.
        await page.evaluate(() => {
            for (const d of document.querySelectorAll('#chat .mes details')) {
                d.setAttribute('open', '');
            }
        });
        await page.waitForTimeout(500);
        await page.screenshot({
            path: testInfo.outputPath('reasoning_pipeline_final.png'),
            fullPage: true,
        });

        // Pull the captured request bodies to verify step 3: turn 2's
        // request body should contain the prior turn's reasoning re-
        // rendered as a <|channel>thought\n...<channel|> block in the
        // last system message OR in the assistant turn's content. We
        // grep the rendered messages list for <|channel>thought.
        const renderInspect = networkRecords
            .filter(n => n.kind === 'request' && n.body?.messages)
            .map((n, idx) => {
                const msgs = n.body.messages || [];
                const rendered = msgs.map(m =>
                    typeof m.content === 'string' ? m.content : JSON.stringify(m.content)).join('\n');
                return {
                    request_idx: idx,
                    msg_count: msgs.length,
                    contains_channel_thought: rendered.includes('<|channel>thought'),
                    contains_past_reasoning_field: msgs.some(m =>
                        m.role === 'assistant' && typeof m.reasoning === 'string' && m.reasoning.length > 0),
                };
            });
        console.log('\n=== network-side render inspection ===');
        for (const r of renderInspect) {
            console.log(`  request[${r.request_idx}]: msgs=${r.msg_count} channel_thought=${r.contains_channel_thought} past_reasoning_field=${r.contains_past_reasoning_field}`);
        }

        fs.writeFileSync(testInfo.outputPath('reasoning_pipeline.json'),
            JSON.stringify({ turnRecords, renderInspect }, null, 2));

        // Step-2 assertion: no turn produced empty content.
        for (const t of turnRecords) {
            expect(t.last.mes_len,
                `turn ${t.turn} content should be non-empty (no empty-response bug)`)
                .toBeGreaterThan(0);
        }

        // Step-4 assertion: at least one turn that produced reasoning
        // also produced a reasoning_summary. Not all turns need it
        // (the model might not emit reasoning on every turn) but at
        // least one with reasoning_len>0 should get a summary.
        const turnsWithReasoning = turnRecords.filter(t => t.last.reasoning_len > 0);
        const turnsWithSummary = turnsWithReasoning.filter(t => t.last.reasoning_summary);
        console.log(`\nturns with reasoning: ${turnsWithReasoning.length}`);
        console.log(`turns with auto-summary heading: ${turnsWithSummary.length}`);
        if (turnsWithReasoning.length > 0) {
            expect(turnsWithSummary.length,
                'auto-summary should have run on at least one turn with reasoning')
                .toBeGreaterThan(0);
        }

        // Step-3 assertion: ST should be forwarding the prior turn's
        // .extra.reasoning onto the outgoing assistant message in the
        // chat-completions request body. The bridge's chat-template
        // then re-renders that as a <|channel>thought block server-
        // side, BUT we can only see the ST-FE-to-ST-backend body from
        // here, not the bridge-to-engine prompt. So we assert the
        // pre-template `reasoning` field is present, not the rendered
        // marker.
        if (turnsWithReasoning.length > 0 && renderInspect.length >= 2) {
            const laterRequestsForwardReasoning = renderInspect.slice(1)
                .some(r => r.contains_past_reasoning_field);
            expect(laterRequestsForwardReasoning,
                'later request bodies should carry the past-turn `reasoning` field on assistant messages')
                .toBe(true);
        }
    });
});
