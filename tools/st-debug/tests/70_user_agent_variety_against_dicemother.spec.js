// User-agent variety against dicemother seed chats.
//
// IDV (implementation/design/validation) warp:
//   IMPL: 3 canonical seed chats committed to
//         sillytavern-fork/default/content/chats/dicemother/. Each
//         seeds a different decision-stake shape (adversarial
//         accusation, offer-with-strings, ambiguous discovery).
//   DESIGN: each iteration hits /api/plugins/user-personas/poll
//         directly with the seed chat for each of the 4 hand-
//         written user-agents, then asks gemma-4 to review the
//         variety in one shot (scalable oversight — model summarizes
//         models). The mobile-viewport rendering bug is a separate
//         concern handled by a downstream test; THIS test does the
//         token-stream / behavior-variety half.
//   VALIDATION: rendered artifacts are the raw captures.json,
//         variety_review.txt, and variety_review.json per iteration.
//
// Wallclock budget per iteration cycle: ≤ 3 minutes hard. Each
// iteration = one seed × 4 agent /poll calls (n=1) + 1 reviewer call.

import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

const ST_URL = 'http://127.0.0.1:8002';
const BRIDGE_URL = 'http://127.0.0.1:8001';
const FORK_ROOT = '/Users/mdot/sillytavern-fork';
const SEEDS = ['accusation', 'geas', 'cistern'];

// The four hand-written user-agents seeded by the user-personas
// extension on first boot. IDs from /api/plugins/user-personas/personas
// at the current commit.
const USER_AGENTS = [
    { id: 'gushing-fan',          name: 'Gushing Fan' },
    { id: 'polite-naturalist',    name: 'Polite Naturalist' },
    { id: 'pushy-completionist',  name: 'Pushy Completionist' },
    { id: 'wry-skeptic',          name: 'Wry Skeptic' },
];

const TEST_BUDGET = 3 * 60 * 1000;
const POLL_BUDGET = 60 * 1000;
const REVIEWER_BUDGET = 90 * 1000;

/**
 * Parse a seed JSONL into ST's chat[] shape (skipping the metadata
 * header line).
 */
function loadSeedChatArray(seed_id) {
    const srcPath = path.join(
        FORK_ROOT, 'default/content/chats/dicemother', `seed_${seed_id}.jsonl`);
    const lines = fs.readFileSync(srcPath, 'utf8').trim().split('\n');
    // First line is metadata; the rest are message rows.
    return lines.slice(1).map(l => JSON.parse(l));
}

test.describe('user-agent variety against dicemother seeds', { tag: ['@slow', '@user-personas'] }, () => {
    test.setTimeout(TEST_BUDGET);

    for (const seed of SEEDS) {
        test(`seed=${seed}: 4 agents × /poll(n=1) + scalable-oversight review`, async ({ request }, testInfo) => {
            const t0 = Date.now();
            const chat = loadSeedChatArray(seed);
            console.log(`  seed=${seed}: ${chat.length} pre-rolled turns`);

            const captures = [];
            for (const agent of USER_AGENTS) {
                const cT0 = Date.now();
                let resp;
                try {
                    resp = await request.post(
                        `${ST_URL}/api/plugins/user-personas/poll`,
                        {
                            data: {
                                persona_id: agent.id,
                                chat,
                                n_candidates: 1,
                            },
                            timeout: POLL_BUDGET,
                        });
                } catch (e) {
                    captures.push({ agent: agent.name, id: agent.id, text: `(error: ${e.message})`, ms: Date.now() - cT0 });
                    console.warn(`  ⚠ ${agent.name}: poll threw ${e.message}`);
                    continue;
                }
                if (!resp.ok()) {
                    const body = await resp.text().catch(() => '');
                    captures.push({ agent: agent.name, id: agent.id, text: `(HTTP ${resp.status()}: ${body.slice(0, 200)})`, ms: Date.now() - cT0 });
                    console.warn(`  ⚠ ${agent.name}: poll ${resp.status()}`);
                    continue;
                }
                const body = await resp.json();
                const cand = body?.candidates?.[0] || {};
                const text = cand.text?.trim() || '(empty)';
                const ms = Date.now() - cT0;
                // /poll's response includes truncated:bool + finish_reason
                // — these are the canonical signal for the truncation bug
                // independent of the reviewer model's text-level guess.
                captures.push({
                    agent: agent.name, id: agent.id, text, ms,
                    truncated: !!cand.truncated,
                    finish_reason: cand.finish_reason || null,
                    max_tokens_used: body.max_tokens_used || null,
                });
                console.log(`  ✓ ${agent.name}: ${text.length} chars in ${(ms / 1000).toFixed(1)}s ` +
                    `(truncated=${!!cand.truncated} finish=${cand.finish_reason})`);
            }

            fs.writeFileSync(testInfo.outputPath('captures.json'),
                JSON.stringify({ seed, captures }, null, 2));

            // Scalable-oversight reviewer call.
            const reviewerPrompt = [
                `Four hand-written user-agents (Gushing Fan, Polite Naturalist, Pushy Completionist, Wry Skeptic) were each shown the same starting chat with dicemother (a tabletop-RPG dungeon-master character) — seed scenario "${seed}". Each agent produced one candidate user turn. Review them for behavioral variety.`,
                '',
                'Output exactly one JSON object with these keys:',
                '  - agent_signatures: object mapping each agent name to a one-sentence behavioral signature observed in this candidate.',
                '  - spread: one sentence — did the four candidates converge or spread out into distinct strategic positions?',
                '  - truncation_flags: array of agent names whose output appears to be cut off mid-sentence or mid-thought (NOT a deliberate trailing "...").',
                '  - cumulative_variety: one of "high" | "medium" | "low".',
                '  - notes: one paragraph (3-5 sentences) of prose summary.',
                '',
                'Captured candidates:',
                '',
                ...captures.map(c => `[agent=${c.agent}]\n${c.text}\n`),
                '',
                'Respond with exactly one JSON object — no preamble, no markdown fence.',
            ].join('\n');

            const reviewerT0 = Date.now();
            const reviewResp = await request.post(`${BRIDGE_URL}/v1/chat/completions`, {
                data: {
                    model: 'gemma-4-a4b',
                    messages: [
                        { role: 'system', content: 'You output exactly one JSON object — no preamble, no markdown fence.' },
                        { role: 'user', content: reviewerPrompt },
                    ],
                    stream: false,
                    temperature: 0.3,
                    max_tokens: 1024,
                    seed: 9921,
                },
                timeout: REVIEWER_BUDGET,
            });
            expect(reviewResp.ok(), `reviewer ${reviewResp.status()}`).toBe(true);
            const reviewBody = await reviewResp.json();
            const reviewRaw = reviewBody?.choices?.[0]?.message?.content || '';
            const reviewerMs = Date.now() - reviewerT0;

            fs.writeFileSync(testInfo.outputPath('variety_review.txt'), reviewRaw);
            try {
                const o = reviewRaw.indexOf('{');
                const c = reviewRaw.lastIndexOf('}');
                if (o >= 0 && c > o) {
                    const parsed = JSON.parse(reviewRaw.slice(o, c + 1));
                    fs.writeFileSync(testInfo.outputPath('variety_review.json'),
                        JSON.stringify(parsed, null, 2));
                }
            } catch (_) { /* raw txt is canonical */ }

            console.log(`  reviewer landed in ${(reviewerMs / 1000).toFixed(1)}s, total ${((Date.now() - t0) / 1000).toFixed(1)}s`);
            console.log(reviewRaw.slice(0, 1000));
        });
    }
});
