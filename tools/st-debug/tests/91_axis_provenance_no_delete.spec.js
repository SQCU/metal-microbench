// Axis registry + provenance no-delete invariants.
//
// Focused fixture-backed coverage for:
// - docs/_grounding_ui_specs_summary.md section 1 Axis Registry + Lineage View
// - docs/_grounding_ui_specs_summary.md section 6 Provenance Tagging + View Filter
// - docs/multi_user_agent_chat_interface_spec.md:
//   P-CANONICAL-NOT-MIRRORED and P-VISIBLE-RESIDUE
//
// The route stubs are deliberate: these assertions need fixed lineage,
// orphan, and provenance shapes, but must not mutate the live corpus.

import { test, expect } from '@playwright/test';
import fs from 'node:fs';

const PLUGIN_BASE = '/api/plugins/user-personas';
const STATIC_AXES = `${PLUGIN_BASE}/static/axes.html`;
const STATIC_SUGGESTER = `${PLUGIN_BASE}/static/suggester.html`;
const STATIC_DIR = '/Users/mdot/metal-microbench/tools/st-debug/sillytavern-fork/plugins/user-personas/static';

function axisFixtures() {
    return {
        axes: [
            {
                id: 'temperament',
                name: 'Temperament',
                def: '1: cool headed / 5: volatile',
                kind: 'bio',
                scale_min: 1,
                scale_max: 5,
                derived_from: null,
            },
            {
                id: 'temperament_heat',
                name: 'Temperament heat',
                def: '1: contained / 5: incandescent',
                kind: 'bio',
                scale_min: 1,
                scale_max: 5,
                derived_from: {
                    parent: 'temperament',
                    hypothesis: 'split heat from general temperament',
                },
            },
            {
                id: 'dialogue_control',
                name: 'Dialogue control',
                def: '1: follows / 5: steers',
                kind: 'agent',
                scale_min: 1,
                scale_max: 5,
                derived_from: null,
            },
        ],
        personas: [
            {
                id: 'manual-operator.png',
                name: 'Manual Operator',
                signature: { temperament: 4, temperament_heat: 5, orphan_axis: 2 },
            },
            {
                id: 'legacy-observer.png',
                name: 'Legacy Observer',
                signature: { temperament: 2, orphan_axis: 3 },
            },
        ],
        agents: [
            {
                id: 'agent-a.png',
                name: 'Agent A',
                signature: { temperament: 5, dialogue_control: 4, orphan_axis: 4 },
            },
        ],
    };
}

function provenanceFixtures() {
    const personas = [
        { id: 'canonical-bio.png', name: 'Canonical Bio', provenance: { kind: 'canonical' }, signature: { temperament: 3 } },
        { id: 'manual-bio.png', name: 'Manual Bio', provenance: { kind: 'manual' }, signature: { temperament: 3 } },
        { id: 'legacy-bio.png', name: 'Legacy Bio', signature: { temperament: 3 } },
        {
            id: 'experiment-bio.png',
            name: 'Experiment Bio',
            provenance: { kind: 'experiment_output', experiment_id: 'fixture-exp', run_id: 'run-91' },
            signature: { temperament: 4 },
        },
        { id: 'seed-bio.png', name: 'Seed Bio', provenance: { kind: 'seed_demo', seed_phrase: 'fixture seed' }, signature: { temperament: 2 } },
    ];
    const agents = personas.map((p, i) => ({
        id: `agent-${i}.png`,
        name: `Agent ${i}`,
        designed_for_bio_id: p.id,
        provenance: p.provenance || { kind: 'legacy' },
        signature: { temperament: 3 },
    }));
    const rows = personas.map((p, i) => ({
        bio_id: p.id,
        agent_id: agents[i].id,
        why: `${p.name} fixture row`,
        distance: i * 0.25,
        persona: { id: p.id, name: p.name, provenance: p.provenance || { kind: 'legacy' } },
        agent: { id: agents[i].id, name: agents[i].name, provenance: agents[i].provenance },
    }));
    return {
        personas,
        agents,
        yapperSeed: {
            top: rows.slice(0, 3),
            side: rows.slice(3),
            _meta: {
                K_top: 3,
                K_side: 2,
                target_signature: { temperament: 3 },
                target_completed_axes: 1,
                candidates_considered: rows.length,
                bios_total: personas.length,
                agents_total: agents.length,
                pending_synthesis: [],
                pending_count: 0,
            },
        },
    };
}

async function installStaticPageGlobals(page) {
    await page.addInitScript(() => {
        window.csrfFetch = (url, opts) => window.fetch(url, opts);
        window.localStorage.removeItem('user-personas/suggester-filter-state');
    });
}

async function fulfillJson(route, body, status = 200) {
    await route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(body),
    });
}

async function routeStaticHtml(page, urlPath, fileName) {
    await page.route(`**${urlPath}`, async route => {
        await route.fulfill({
            status: 200,
            contentType: 'text/html',
            body: fs.readFileSync(`${STATIC_DIR}/${fileName}`, 'utf8'),
        });
    });
}

test.describe('axis provenance no-delete invariants - desktop only', () => {
    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'focused fixture spec runs only on desktop project');
    });

    test('Axis Registry: lineage roots, immutable edit fields, confirmed delete warning, and orphan signatures', async ({ page }) => {
        await installStaticPageGlobals(page);
        await routeStaticHtml(page, STATIC_AXES, 'axes.html');

        const fixture = axisFixtures();
        let deleteCalls = 0;
        const postedBodies = [];

        await page.route(`**${PLUGIN_BASE}/axes`, async route => {
            if (route.request().method() === 'GET') {
                await fulfillJson(route, { axes: fixture.axes });
                return;
            }
            await fulfillJson(route, { error: 'unexpected method' }, 405);
        });
        await page.route(`**${PLUGIN_BASE}/personas`, async route => {
            await fulfillJson(route, { personas: fixture.personas });
        });
        await page.route(`**${PLUGIN_BASE}/agents`, async route => {
            await fulfillJson(route, { agents: fixture.agents });
        });
        await page.route(`**${PLUGIN_BASE}/axes/**`, async route => {
            const url = new URL(route.request().url());
            const axisId = decodeURIComponent(url.pathname.split('/').pop());
            if (route.request().method() === 'POST') {
                const body = route.request().postDataJSON();
                postedBodies.push(body);
                const existing = fixture.axes.find(a => a.id === axisId);
                if (existing && (body.kind !== existing.kind || body.scale_min !== existing.scale_min || body.scale_max !== existing.scale_max)) {
                    await fulfillJson(route, { error: 'kind/scale immutable' }, 400);
                    return;
                }
                Object.assign(existing, {
                    name: body.name,
                    def: body.def,
                    derived_from: body.derived_from ?? null,
                });
                await fulfillJson(route, existing);
                return;
            }
            if (route.request().method() === 'DELETE') {
                deleteCalls += 1;
                const ix = fixture.axes.findIndex(a => a.id === axisId);
                if (ix >= 0) fixture.axes.splice(ix, 1);
                await fulfillJson(route, { ok: true, orphaned_signatures: { bios: 2, agents: 1 } });
                return;
            }
            await fulfillJson(route, { error: 'unexpected method' }, 405);
        });

        await page.goto(STATIC_AXES);
        await expect(page.locator('#status')).toContainText('3 axes loaded');

        const root = page.locator('.axis-card[data-axis-id="temperament"]');
        const derived = page.locator('.axis-card[data-axis-id="temperament_heat"]');
        await expect(root).toBeVisible();
        await expect(root).toHaveAttribute('data-depth', '0');
        await expect(derived).toBeVisible();
        await expect(derived).toHaveClass(/derived/);
        await expect(derived).toHaveAttribute('data-depth', '1');
        await expect(derived.locator('.axis-tree-prefix')).not.toHaveText('');
        await expect(derived.locator('.lineage-note')).toContainText('derived from temperament');

        await root.locator('button[data-action="edit"]').click();
        const editForm = root.locator('[data-role="edit-form"]');
        await expect(editForm).toBeVisible();
        await expect(editForm).toContainText('kind / scale');
        await expect(editForm).toContainText('bio, 1-5 (immutable)');
        await expect(editForm.locator('select, input[name="kind"], input[name="scale_min"], input[name="scale_max"]'))
            .toHaveCount(0);

        await editForm.locator('input[type="text"]').fill('Temperament Edited');
        await editForm.locator('textarea').fill('1: newly calm / 5: newly intense');
        await editForm.locator('button[data-role="edit-save"]').click();
        await expect(page.locator('.axis-card[data-axis-id="temperament"] [data-role="def"]'))
            .toContainText('1: newly calm / 5: newly intense');
        expect(postedBodies[0]).toMatchObject({
            name: 'Temperament Edited',
            def: '1: newly calm / 5: newly intense',
            kind: 'bio',
            scale_min: 1,
            scale_max: 5,
        });

        const tamperStatus = await page.evaluate(async () => {
            const r = await window.csrfFetch('/api/plugins/user-personas/axes/temperament', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: 'Tampered',
                    def: 'tampered',
                    kind: 'agent',
                    scale_min: 0,
                    scale_max: 10,
                }),
            });
            return r.status;
        });
        expect(tamperStatus, 'mocked/probe edit rejects kind/scale tampering').toBe(400);

        await page.locator('.axis-card[data-axis-id="temperament"] button[data-action="delete"]').click();
        await expect(page.locator('#confirm-dialog')).toBeVisible();
        await expect(page.locator('#confirm-warning'))
            .toContainText('Deleting this axis will leave 2 bios + 1 agents with dangling references');
        expect(deleteCalls, 'delete is not issued before operator confirmation').toBe(0);
        await page.locator('#confirm-cancel').click();
        await expect(page.locator('#confirm-dialog')).toBeHidden();
        expect(deleteCalls, 'cancel keeps delete unissued').toBe(0);

        await page.locator('.axis-card[data-axis-id="temperament"] button[data-action="delete"]').click();
        await page.locator('#confirm-delete').click();
        await expect(page.locator('#confirm-dialog')).toBeHidden();
        expect(deleteCalls, 'delete is issued only after confirm').toBe(1);

        const orphanRow = page.locator('#orphans-list .orphan-row[data-orphan-id="orphan_axis"]');
        await expect(orphanRow).toBeVisible();
        await expect(orphanRow.locator('.orphan-id')).toHaveText('orphan_axis');
        await expect(orphanRow.locator('.orphan-meta'))
            .toContainText('referenced by 2 bios + 1 agents (no registry entry)');
    });

    test('Provenance filter: default visibility, client-only toggles, full endpoints, and no auto-delete counts', async ({ page }) => {
        await installStaticPageGlobals(page);
        await routeStaticHtml(page, STATIC_SUGGESTER, 'suggester.html');

        const fixture = provenanceFixtures();
        const requestCounts = { personas: 0, agents: 0, yapperSeed: 0 };

        await page.route(`**${PLUGIN_BASE}/personas`, async route => {
            requestCounts.personas += 1;
            await fulfillJson(route, { personas: fixture.personas });
        });
        await page.route(`**${PLUGIN_BASE}/agents`, async route => {
            requestCounts.agents += 1;
            await fulfillJson(route, { agents: fixture.agents });
        });
        await page.route(`**${PLUGIN_BASE}/yapper-seed`, async route => {
            requestCounts.yapperSeed += 1;
            await fulfillJson(route, fixture.yapperSeed);
        });
        await page.route(`**${PLUGIN_BASE}/poll`, async route => {
            await fulfillJson(route, { text: 'fixture poll', applied_overlay: null });
        });
        await page.route(`**${PLUGIN_BASE}/bridge-status`, async route => {
            await fulfillJson(route, { ok: true, bridge_ok: true });
        });

        await page.goto(STATIC_SUGGESTER);
        await expect(page.locator('#provenance-filter')).toBeVisible();

        const endpointCountsBefore = await page.evaluate(async () => {
            const [personas, agents, ranked] = await Promise.all([
                fetch('/api/plugins/user-personas/personas').then(r => r.json()),
                fetch('/api/plugins/user-personas/agents').then(r => r.json()),
                fetch('/api/plugins/user-personas/yapper-seed', { method: 'POST' }).then(r => r.json()),
            ]);
            return {
                personas: personas.personas.length,
                agents: agents.agents.length,
                ranked: (ranked.top || []).length + (ranked.side || []).length,
            };
        });
        expect(endpointCountsBefore).toEqual({ personas: 5, agents: 5, ranked: 5 });

        await page.evaluate(payload => {
            window.applyRankResponse(payload);
        }, fixture.yapperSeed);

        const filter = page.locator('#provenance-filter');
        await expect(filter.locator('input[data-kind="canonical"]')).toBeChecked();
        await expect(filter.locator('input[data-kind="manual"]')).toBeChecked();
        await expect(filter.locator('input[data-kind="legacy"]')).toBeChecked();
        await expect(filter.locator('input[data-kind="experiment_output"]')).not.toBeChecked();
        await expect(filter.locator('input[data-kind="seed_demo"]')).not.toBeChecked();

        await expect(page.locator('#pf-hidden-badge')).toHaveText('[2 hidden]');
        await expect(page.locator('.ranked-row')).toHaveCount(3);
        await expect(page.locator('.ranked-row[data-bio-id="canonical-bio.png"]')).toBeVisible();
        await expect(page.locator('.ranked-row[data-bio-id="manual-bio.png"]')).toBeVisible();
        await expect(page.locator('.ranked-row[data-bio-id="legacy-bio.png"]')).toBeVisible();
        await expect(page.locator('.ranked-row[data-bio-id="experiment-bio.png"]')).toHaveCount(0);
        await expect(page.locator('.ranked-row[data-bio-id="seed-bio.png"]')).toHaveCount(0);

        const routeCountsAfterRender = { ...requestCounts };
        await filter.locator('input[data-kind="experiment_output"]').click();
        await expect(page.locator('#pf-hidden-badge')).toHaveText('[1 hidden]');
        await expect(page.locator('.ranked-row')).toHaveCount(4);
        await expect(page.locator('.ranked-row[data-bio-id="experiment-bio.png"]')).toBeVisible();
        expect(requestCounts, 'provenance toggle re-filters client-side without API calls')
            .toEqual(routeCountsAfterRender);

        await filter.locator('input[data-kind="seed_demo"]').click();
        await expect(page.locator('#pf-hidden-badge')).toHaveText('[0 hidden]');
        await expect(page.locator('.ranked-row')).toHaveCount(5);
        expect(requestCounts, 'second toggle also stays client-side')
            .toEqual(routeCountsAfterRender);

        await page.reload();
        await expect(page.locator('#provenance-filter')).toBeVisible();
        const endpointCountsAfter = await page.evaluate(async () => {
            const [personas, agents, ranked] = await Promise.all([
                fetch('/api/plugins/user-personas/personas').then(r => r.json()),
                fetch('/api/plugins/user-personas/agents').then(r => r.json()),
                fetch('/api/plugins/user-personas/yapper-seed', { method: 'POST' }).then(r => r.json()),
            ]);
            return {
                personas: personas.personas.length,
                agents: agents.agents.length,
                ranked: (ranked.top || []).length + (ranked.side || []).length,
            };
        });
        expect(endpointCountsAfter, 'reload after filter toggles returns all records; no auto-delete/filtering server-side')
            .toEqual(endpointCountsBefore);
    });
});
