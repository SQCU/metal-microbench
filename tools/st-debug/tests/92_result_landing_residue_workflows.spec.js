// Worker E - focused long-result landing coverage.
//
// These specs exercise the UI residue that appears after long-running
// persona/axis workflows complete, but use deterministic Playwright API
// stubs instead of waiting for lock_in_iterative / axis_splitter /
// cluster_disambiguator model runs.
//
// Source contracts:
// - docs/multi_user_agent_chat_interface_spec.md:
//   Coordinate Picker / Designer / Fixed-point / Axes registry
// - docs/_grounding_ui_specs_summary.md:
//   Iteration Timeline + Feature Factorization
// - tests 71, 85, 86 for existing surface selectors and dispatch shape.

import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';

const ST_URL = 'http://127.0.0.1:8002';
const PLUGIN_BASE = '/api/plugins/user-personas';
const CORPUS_URL = `${ST_URL}${PLUGIN_BASE}/static/corpus.html`;
const DESIGNER_URL = `${ST_URL}${PLUGIN_BASE}/static/designer.html`;
const FIXED_POINT_URL = `${ST_URL}${PLUGIN_BASE}/static/fixed_point.html`;
const STATIC_ROOT = new URL('../sillytavern-fork/plugins/user-personas/static/', import.meta.url);
const STATIC_BOOTSTRAP = `
<script>
window.csrfFetch = window.csrfFetch || ((url, opts) => window.fetch(url, opts));
</script>
`;

const CANDIDATE_ID = 'user-personas-result-landing-candidate.png';
const CANDIDATE_NAME = 'Result Landing Candidate';

const rootAxes = [
    {
        id: 'astrology_sagittarian',
        name: 'Sagittarian orientation',
        kind: 'bio',
        scale_min: 1,
        scale_max: 5,
        def: '1: grounded literalism / 3: mixed / 5: expansive Sagittarian mythmaking',
    },
    {
        id: 'intellectual_application',
        name: 'Intellectual application',
        kind: 'bio',
        scale_min: 1,
        scale_max: 5,
        def: '1: practical only / 3: balanced / 5: abstract systems-first reasoning',
    },
    {
        id: 'rpg_class',
        name: 'RPG class',
        kind: 'bio',
        scale_min: 1,
        scale_max: 5,
        def: '1: rogue / 3: ranger / 5: wizard',
    },
    {
        id: 'money_orientation',
        name: 'Money orientation',
        kind: 'agent',
        scale_min: 1,
        scale_max: 5,
        def: '1: acquisition / 3: balanced / 5: influence',
    },
];

function candidatePersona() {
    return {
        id: CANDIDATE_ID,
        canonical_key: CANDIDATE_ID,
        name: CANDIDATE_NAME,
        description: 'A synthesized candidate bio that has already landed from the experiment output.',
        bio: 'A synthesized candidate bio that has already landed from the experiment output.',
        signature: {
            astrology_sagittarian: 3,
            intellectual_application: 4,
        },
        target_bio: {
            astrology_sagittarian: 3,
            intellectual_application: 4,
        },
        provenance: {
            kind: 'experiment_output',
            experiment_id: 'coord-result-landing',
            run_id: 'coord-result-run-1',
        },
    };
}

function agentsForCandidate() {
    return [1, 3, 5].map((money, i) => ({
        id: `agent-result-landing-${money}.png`,
        name: `Candidate Agent ${money}`,
        designed_for_bio_id: CANDIDATE_ID,
        created_at: new Date(Date.now() + i * 1000).toISOString(),
        agent_text: `Agent ${money} for the result landing candidate.`,
        injection_mode: 'author_note',
        signature: { money_orientation: money },
        provenance: { kind: 'experiment_output', run_id: 'agent-synth-result-run' },
    }));
}

function baseState() {
    return {
        axes: [...rootAxes],
        personas: [{
            id: 'seed-bio.png',
            canonical_key: 'seed-bio.png',
            name: 'Seed Bio',
            description: 'Seed bio fixture.',
            bio: 'Seed bio fixture.',
            signature: { rpg_class: 3 },
            provenance: { kind: 'seed_demo' },
        }],
        agents: [],
        clusters: [{
            id: 'sag_substantive',
            label: 'Sagittarian substantive cluster',
            bio_count: 4,
            nominal_tight_axis: 'astrology_sagittarian',
        }],
        synthAgentDispatched: false,
        splitPolled: 0,
        disambigPolled: 0,
    };
}

async function installPluginApiStubs(page, state = baseState()) {
    await page.route(`**${PLUGIN_BASE}/**`, async (route) => {
        const req = route.request();
        const url = new URL(req.url());
        const path = url.pathname.slice(PLUGIN_BASE.length);

        if (path.startsWith('/static/')) {
            const fileName = path.split('/').pop();
            const allowed = new Set(['corpus.html', 'designer.html', 'fixed_point.html']);
            if (allowed.has(fileName)) {
                const raw = readFileSync(new URL(fileName, STATIC_ROOT), 'utf8');
                await route.fulfill({
                    status: 200,
                    contentType: 'text/html',
                    body: raw.replace('</head>', `${STATIC_BOOTSTRAP}</head>`),
                });
                return;
            }
            await route.fulfill({ status: 404, body: 'static fixture not stubbed' });
            return;
        }

        const json = async (body, status = 200) => route.fulfill({
            status,
            contentType: 'application/json',
            body: JSON.stringify(body),
        });

        if (path === '/axes' && req.method() === 'GET') {
            await json({ axes: state.axes });
            return;
        }
        if (path === '/personas' && req.method() === 'GET') {
            await json({ personas: state.personas });
            return;
        }
        if (path === '/agents' && req.method() === 'GET') {
            await json({ agents: state.agents });
            return;
        }
        if (path === '/corpus-snapshot' && req.method() === 'GET') {
            await json({ snapshots: [] });
            return;
        }
        if (path === '/corpus-snapshot' && req.method() === 'POST') {
            await json({ ok: true });
            return;
        }
        if (path === '/clusters' && req.method() === 'GET') {
            await json({ clusters: state.clusters });
            return;
        }
        if (path === '/synthesize-bio-from-coordinates' && req.method() === 'POST') {
            const candidate = candidatePersona();
            if (!state.personas.some(p => p.id === candidate.id)) {
                state.personas.push(candidate);
            }
            await json({
                ok: true,
                run_id: 'coord-result-run-1',
                candidate_id: CANDIDATE_ID,
            });
            return;
        }
        if (path === `/synthesize-agents-for-persona/${encodeURIComponent(CANDIDATE_ID)}` && req.method() === 'POST') {
            state.synthAgentDispatched = true;
            state.agents = agentsForCandidate();
            await json({
                ok: true,
                run_id: 'agent-synth-result-run',
                experiment_id: 'agent-synth-for-result-candidate',
            });
            return;
        }
        if (path === `/axes/${encodeURIComponent('rpg_class')}/split` && req.method() === 'POST') {
            await json({ ok: true, run_id: 'split-result-run-1' });
            return;
        }
        if (path === `/clusters/${encodeURIComponent('sag_substantive')}/disambiguate` && req.method() === 'POST') {
            await json({ ok: true, run_id: 'disambig-result-run-1' });
            return;
        }
        if (path === '/experiments/runs/split-result-run-1' && req.method() === 'GET') {
            state.splitPolled += 1;
            if (!state.axes.some(a => a.id === 'rpg_class_spell_vs_steel')) {
                state.axes.push({
                    id: 'rpg_class_spell_vs_steel',
                    name: 'Spell versus steel',
                    kind: 'bio',
                    scale_min: 1,
                    scale_max: 5,
                    def: '1: steel-forward martial framing / 5: spell-forward arcane framing',
                    derived_from: 'rpg_class',
                    provenance: { kind: 'experiment_output', run_id: 'split-result-run-1' },
                });
            }
            await json({
                run: { run_id: 'split-result-run-1', status: 'done', kind: 'axis_split', axis_id: 'rpg_class' },
                log: 'AXIS_SPLIT_DONE parent=rpg_class produced_axis=rpg_class_spell_vs_steel\n',
                results: [],
            });
            return;
        }
        if (path === '/experiments/runs/disambig-result-run-1' && req.method() === 'GET') {
            state.disambigPolled += 1;
            if (!state.axes.some(a => a.id === 'sag_substantive_register')) {
                state.axes.push({
                    id: 'sag_substantive_register',
                    name: 'Sagittarian register',
                    kind: 'bio',
                    scale_min: 1,
                    scale_max: 5,
                    def: '1: paraphrase cluster residue / 5: behaviorally distinct register',
                    derived_from: 'astrology_sagittarian',
                    provenance: { kind: 'experiment_output', run_id: 'disambig-result-run-1' },
                });
            }
            await json({
                run: {
                    run_id: 'disambig-result-run-1',
                    status: 'done',
                    kind: 'cluster_disambig',
                    cluster_id: 'sag_substantive',
                },
                log: [
                    'SPREAD_AXIS_FOUND cluster=sag_substantive',
                    'disambiguated astrology_sagittarian residue',
                    'produced_axis=sag_substantive_register',
                    `produced_persona=${CANDIDATE_ID}`,
                ].join('\n'),
                results: [{ axis_id: 'sag_substantive_register', persona_id: CANDIDATE_ID }],
            });
            return;
        }
        if (path === '/experiments' && req.method() === 'GET') {
            await json({
                experiments: [{
                    id: 'timeline-partial-result',
                    name: 'Timeline partial result',
                    description: 'Fixture experiment for in-progress log landing.',
                    bios: [{
                        slug: 'timeline-bio',
                        name: 'Timeline Bio',
                        canonical_key: 'user-personas-timeline-bio.png',
                        target_bio: { rpg_class: 4 },
                    }],
                    agent_targets: [{ slug: 'timeline-agent', target_agent: { money_orientation: 3 } }],
                    bio_axes: ['rpg_class'],
                    agent_axes: ['money_orientation'],
                    counterparty_avatar: 'the-rock.png',
                    loop_control: { k_max_outer: 3, eps_per_axis: 0.5 },
                }],
            });
            return;
        }
        if (path === '/experiments/timeline-partial-result/run' && req.method() === 'POST') {
            await json({
                ok: true,
                run_id: 'timeline-partial-run-1',
                started_at: '2026-05-25T12:00:00.000Z',
            });
            return;
        }
        if (path === '/experiments/runs/timeline-partial-run-1' && req.method() === 'GET') {
            await json({
                run: {
                    run_id: 'timeline-partial-run-1',
                    experiment_id: 'timeline-partial-result',
                    status: 'running',
                    pid: 4242,
                    started_at: '2026-05-25T12:00:00.000Z',
                    finished_at: null,
                    exit_code: null,
                },
                log: [
                    'OUTER k=0 measuring target_bio rpg_class=4',
                    'INNER timeline-agent k=0 partial candidate landed',
                    'judge progress: max_off_axis=1.25',
                ].join('\n'),
                results: [],
            });
            return;
        }
        if (path === '/compare-agents' && req.method() === 'POST') {
            await json({
                a: { id: 'agent-result-landing-1.png', agent_text: 'Acquisition-forward agent.' },
                b: { id: 'agent-result-landing-3.png', agent_text: 'Balanced agent.' },
                axis_deltas: [{ axis: 'money_orientation', a: 1, b: 3, delta: 2 }],
                text_overlap: { jaccard: 0.42, shared: 8, unique_a: 11, unique_b: 13 },
                llm_summary: 'A is acquisition-forward; B is balanced and less transactional.',
            });
            return;
        }
        if (/^\/agents\/[^/]+$/.test(path) && req.method() === 'POST') {
            await json({ ok: true });
            return;
        }

        await json({ ok: true });
    });
    return state;
}

test.describe('result landing and visible residue workflows', () => {
    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'focused landing workflow specs run once under the desktop project');
    });

    test('coordinate picker completion lands candidate name, target_bio, experiment_output provenance, and enabled Save', async ({ page }) => {
        const state = await installPluginApiStubs(page);
        await page.goto(CORPUS_URL);

        await expect(page.locator('#picker-axis-multiselect')).toBeVisible();
        await page.locator('#picker-axis-multiselect').evaluate(sel => {
            const wanted = new Set(['astrology_sagittarian', 'intellectual_application']);
            for (const opt of sel.options) opt.selected = wanted.has(opt.value);
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });
        await page.locator('input[data-axis-id="astrology_sagittarian"]').fill('3');
        await page.locator('input[data-axis-id="intellectual_application"]').fill('4');

        await page.locator('#picker-synthesize-btn').click();

        const strip = page.locator('#picker-result-strip');
        await expect(strip).toBeVisible({ timeout: 8_000 });
        await expect(page.locator('#picker-result-name')).toHaveText(CANDIDATE_NAME);
        await expect(strip.locator('[data-provenance-kind]')).toHaveAttribute('data-provenance-kind', 'experiment_output');
        await expect(strip.locator('[data-provenance-kind]')).toContainText('experiment_output');
        await expect(page.locator('#picker-preview-text')).toContainText('synthesized candidate bio');

        const saveBtn = page.locator('#picker-save-btn');
        await expect(saveBtn).toBeVisible();
        await expect(saveBtn).toBeEnabled();

        const candidate = state.personas.find(p => p.id === CANDIDATE_ID);
        expect(candidate?.target_bio).toMatchObject({
            astrology_sagittarian: 3,
            intellectual_application: 4,
        });
        expect(candidate?.provenance?.kind).toBe('experiment_output');
    });

    test('candidate bio opens missing-agent auto-synth residue for the new candidate', async ({ page }) => {
        const state = await installPluginApiStubs(page);
        state.personas.push(candidatePersona());

        let synthPostSeen = false;
        page.on('request', (req) => {
            if (req.method() === 'POST' && req.url().includes(`/synthesize-agents-for-persona/${encodeURIComponent(CANDIDATE_ID)}`)) {
                synthPostSeen = true;
            }
        });

        await page.goto(`${DESIGNER_URL}?bio=${encodeURIComponent(CANDIDATE_ID)}`);

        await expect(page.locator('#agent-bio-name')).toHaveText(CANDIDATE_NAME);
        await expect(page.locator('#synth-status-top')).toContainText(/synthesize-agents-for-persona|Synth dispatched|Polling \/agents/i, { timeout: 8_000 });
        await expect.poll(() => synthPostSeen || state.synthAgentDispatched, {
            message: 'designer redirect must dispatch missing-agent synthesis for the landed candidate',
            timeout: 8_000,
        }).toBe(true);
    });

    test('axis split completion refreshes registry with derived child under parent residue', async ({ page }) => {
        await installPluginApiStubs(page);
        await page.goto(CORPUS_URL);

        const parent = page.locator('.axis-card[data-axis-id="rpg_class"]');
        await expect(parent).toBeVisible();
        await parent.locator('button[data-action="split"]').click();
        await parent.locator('button[data-action="split-dispatch"]').click();

        const child = page.locator('.axis-card[data-axis-id="rpg_class_spell_vs_steel"]');
        await expect(child).toBeVisible({ timeout: 8_000 });
        await expect(child).toContainText('derived from rpg_class');
        await expect(page.locator('#axes-status')).toContainText('axes loaded');
    });

    test('cluster disambiguator completion leaves verdict log and produced axis/persona residue', async ({ page }) => {
        await installPluginApiStubs(page);
        await page.goto(CORPUS_URL);

        const cluster = page.locator('.cluster-card[data-cluster-id="sag_substantive"]');
        await expect(cluster).toBeVisible();
        await cluster.locator('button[data-action="disambig"]').click();
        await cluster.locator('button[data-action="disambig-dispatch"]').click();

        const panel = cluster.locator('[data-role="disambig-panel"]');
        await expect(panel.locator('[data-role="disambig-log"]')).toContainText('SPREAD_AXIS_FOUND', { timeout: 8_000 });
        await expect(panel.locator('[data-role="disambig-log"]')).toContainText('sag_substantive_register');
        await expect(panel.locator('[data-role="disambig-log"]')).toContainText(CANDIDATE_ID);
        await expect(panel.locator('[data-role="disambig-status"]')).toContainText(/Disambiguator complete|verdict: SPREAD_AXIS_FOUND/i);
        await expect(page.locator('.axis-card[data-axis-id="sag_substantive_register"]')).toBeVisible();
    });

    test('fixed-point in-progress run renders live log/progress before terminal CHILD EXIT', async ({ page }) => {
        await installPluginApiStubs(page);
        await page.goto(FIXED_POINT_URL);

        await expect(page.locator('#experiments-status')).toContainText('1 experiment loaded');
        await page.locator('.experiment-card[data-eid="timeline-partial-result"] .run-btn').click();

        await expect(page.locator('#run-banner')).toContainText('RUNNING', { timeout: 8_000 });
        await expect(page.locator('#run-banner')).toContainText('timeline-partial-run-1');
        await expect(page.locator('#run-log')).toContainText('OUTER k=0 measuring target_bio');
        await expect(page.locator('#run-log')).toContainText('judge progress: max_off_axis=1.25');
        await expect(page.locator('#run-log')).not.toContainText('CHILD EXIT');
        await expect(page.locator('#run-results')).toContainText('No per-bio result files yet');
    });

    test('compare side-by-side affordance exists and renders a diff for synthesized agents', async ({ page }) => {
        const state = await installPluginApiStubs(page);
        state.personas.push(candidatePersona());

        await page.goto(`${DESIGNER_URL}?bio=${encodeURIComponent(CANDIDATE_ID)}`);

        await expect(page.locator('#compare-btn')).toBeVisible();
        await expect(page.locator('#compare-btn')).toBeEnabled({ timeout: 10_000 });
        await page.locator('#compare-btn').click();

        await expect(page.locator('#compare-result')).toContainText('Comparison: agent-result-landing-1.png vs agent-result-landing-3.png');
        await expect(page.locator('#compare-result .compare-grid')).toBeVisible();
        await expect(page.locator('#compare-result')).toContainText('LLM summary');
    });

    // TODO(D10/export-compare affordance): the grounding summary asks for
    // markdown/CSV export buttons alongside compare-side-by-side when
    // implemented. Current static surfaces expose Compare A vs B, but no
    // markdown or CSV export controls were found in user-personas static pages.
    test.fixme('markdown and CSV export affordances exist for landed long-result artifacts', async ({ page }) => {
        await installPluginApiStubs(page);
        await page.goto(FIXED_POINT_URL);
        await expect(page.getByRole('button', { name: /export markdown/i })).toBeVisible();
        await expect(page.getByRole('button', { name: /export csv/i })).toBeVisible();
    });
});
