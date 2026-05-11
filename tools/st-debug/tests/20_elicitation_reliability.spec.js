import { test, expect } from '@playwright/test';
import { loadAndConnect, sendAndObserve } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

// Empirical elicitation reliability — uses ONLY the actual ST UI
// (no ctx.* internals, no programmatic /newchat, no
// selectCharacterById). Runs N user-message → observe-response
// trials and reports the rate at which the model emits a tool_call
// for the rendered-visual class of prompts.
//
// This is a measurement, not a pass/fail of model capability. The
// rate is what it is; if it's low, that's prompt-engineering work,
// not a test failure. We assert only that the harness itself is
// alive: every trial got SOME response within timeout.

test.describe('elicitation reliability — render-visual class', () => {
    test.setTimeout(20 * 60 * 1000);  // 20min budget for N trials

    test('N trials of "draw me a voronoi" via real UI, count tool emissions', async ({ page }, testInfo) => {
        await loadAndConnect(page);

        const N = 8;  // small but enough for binomial
        const prompts = [
            'draw me a voronoi diagram with 12 seeds, distinct fill colors per cell',
            'show me a 3:5 lissajous curve',
            'render a sunflower-seed phyllotaxis pattern with 200 dots',
            'visualize a damped harmonic oscillator trajectory',
            'draw a fractal tree branching at the golden ratio',
            'plot 8 random circles within a unit square',
            'sketch a torus seen from 30 degrees above the equator',
            'show what wave interference from two nearby sources looks like',
        ];
        const trials = [];
        for (let i = 0; i < N; i++) {
            const prompt = prompts[i % prompts.length];
            console.log(`[trial ${i}] sending: "${prompt}"`);
            const t0 = Date.now();
            try {
                const record = await sendAndObserve(page, prompt, { timeoutMs: 90_000 });
                trials.push({
                    trial: i,
                    prompt,
                    finishState: record.finishState,
                    elapsedMs: Math.round(record.elapsedMs),
                    toolCalls: record.toolInvocations.map(t => t.name),
                    toolProgress: record.toolProgress.map(t => ({
                        label: t.label, status: t.status,
                    })),
                    assistantText: (record.assistantText || '').slice(0, 200),
                });
                console.log(`[trial ${i}] finishState=${record.finishState} ` +
                    `tools=${record.toolInvocations.map(t => t.name).join(',') || '(none)'} ` +
                    `t=${Math.round(record.elapsedMs)}ms`);
            } catch (e) {
                trials.push({
                    trial: i,
                    prompt,
                    error: String(e?.message || e).slice(0, 300),
                    elapsedMs: Date.now() - t0,
                });
                console.log(`[trial ${i}] ERROR: ${e?.message}`);
            }
        }

        // Statistics
        const toolFires = trials.filter(t => {
            const haystack = (
                (t.toolCalls || []).join(' ') + ' ' +
                (t.toolProgress || []).map(p => p.label).join(' ')
            ).toLowerCase();
            return haystack.includes('render-visual');
        }).length;
        const anyToolFires = trials.filter(t => (t.toolCalls?.length || t.toolProgress?.length || 0) > 0).length;
        const completed = trials.filter(t => t.finishState === 'completed' || t.finishState === 'tool_handled').length;
        const errored = trials.filter(t => t.error).length;

        const summary = {
            total_trials: N,
            errored,
            completed,
            tool_fires_render_visual: toolFires,
            tool_fires_any: anyToolFires,
            render_visual_rate: toolFires / N,
            any_tool_rate: anyToolFires / N,
        };
        console.log('=== elicitation summary ===');
        console.log(JSON.stringify(summary, null, 2));

        fs.writeFileSync(
            testInfo.outputPath('elicitation_trials.json'),
            JSON.stringify({ summary, trials }, null, 2),
        );

        // Harness liveness assertions — no "must elicit" claims
        expect(errored, 'no harness errors across trials').toBeLessThan(N);
        expect(completed, 'every trial got SOME response').toBe(N);
    });
});
