// Suggester surface must NOT tell the operator to do agent-work.
//
// Operator observed (screenshot 2026-05-24): the suggester showed a
// red banner saying "Bridge reachable but /agents is empty. Restart ST
// (./scripts/run.sh) to trigger first-launch K=2 auto-synth ... or use
// the Designer to synthesize for a specific bio." That copy violated:
//   - P-NO-EMPTY-FIRST-PAINT (surfaces a failure mode that shouldn't
//     exist if the plugin is healthy)
//   - bios_must_have_agents memory ("No escape hatches.")
//   - "agent owns mandatory behaviors" — telling the user to restart
//     ST is the agent procrastinating
//
// Fix landed in sillytavern-fork@761fb546a: plugin gains
// POST /dispatch-missing-agent-synth endpoint; suggester banner JS
// auto-fires it on detect (bridge_up + agents_empty + bios>0) and
// shows a "Synthesizing K=2 agents for N bios..." STATUS instead of
// asking the user to do anything.
//
// This spec locks the behavior:
//   1. No "Restart ST" text in any user-visible banner
//   2. No "use the Designer to synthesize" escape-hatch text
//   3. Auto-dispatch endpoint reachable + returns dispatched/in_flight
//   4. Banner copy uses "Synthesizing" / "Dispatching" status framing,
//      not "Restart ST" imperative framing

import { test, expect } from '@playwright/test';

const PLUGIN_BASE = 'http://127.0.0.1:8002/api/plugins/user-personas';

test.describe('suggester surface — no defeatist banner', () => {
    test.setTimeout(60_000);

    test('plugin dispatch endpoint exists + returns sane shape', async ({ request }) => {
        const r = await request.post(`${PLUGIN_BASE}/dispatch-missing-agent-synth`, {
            data: { reason: 'spec-80-smoke' },
        });
        expect(r.ok()).toBe(true);
        const body = await r.json();
        expect(body.ok).toBe(true);
        // Either dispatched > 0 (no agents yet), or dispatched == 0
        // (all bios already have agents OR in-flight). Both are sane.
        expect(typeof body.dispatched).toBe('number');
        expect(Array.isArray(body.bios)).toBe(true);
        expect(Array.isArray(body.experimentIds)).toBe(true);
        expect(typeof body.bios_in_corpus).toBe('number');
        expect(typeof body.in_flight).toBe('number');
    });

    test('suggester.html has no "Restart ST" text in user-visible banners', async ({ request }) => {
        const r = await request.get(`${PLUGIN_BASE}/static/suggester.html`);
        expect(r.ok()).toBe(true);
        const html = await r.text();
        // Strip JS comments + multi-line strings from the search; we
        // only care about TEXT that ends up rendered. Heuristic:
        // user-visible banner text lives inside template-literal
        // backticks that get assigned to `bridgeBanner.innerHTML`.
        const innerHTMLBlocks = [];
        const re = /bridgeBanner\.innerHTML\s*=\s*`([^`]+)`/g;
        let m;
        while ((m = re.exec(html)) !== null) innerHTMLBlocks.push(m[1]);
        // At minimum 2 banner-copy blocks (bridge-down + synthesizing).
        expect(innerHTMLBlocks.length).toBeGreaterThanOrEqual(2);
        for (const block of innerHTMLBlocks) {
            expect(block, `defeatist 'Restart ST' text in banner copy: ${block.slice(0, 80)}...`)
                .not.toMatch(/Restart ST/i);
            expect(block, `escape-hatch 'use the Designer' text in banner copy: ${block.slice(0, 80)}...`)
                .not.toMatch(/use the (?:<strong>)?Designer/i);
            expect(block, `manual-shell-cmd '\\.\\/scripts\\/run\\.sh' in banner copy: ${block.slice(0, 80)}...`)
                .not.toMatch(/\.\/scripts\/run\.sh/);
        }
    });

    test('suggester.html banner copy uses status framing, not imperative', async ({ request }) => {
        const r = await request.get(`${PLUGIN_BASE}/static/suggester.html`);
        const html = await r.text();
        // The agents-empty banner must contain Synthesizing OR Dispatching
        // (status framing).
        expect(html).toMatch(/Synthesizing K=2|Dispatching K=2 agent synthesis/);
        // The auto-dispatch fetch must be wired in pollBridgeAndAgents.
        expect(html).toMatch(/dispatch-missing-agent-synth/);
    });
});
