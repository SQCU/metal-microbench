// Spec T6 — factorization stays on-spec.
//
// The operator's directive (2026-05-20): "if any factorization happened
// automatically, shouldn't it have happened upon sorceror/rogue or along
// thiefromance <-> romance? or along other star signs? this to me is
// evidence that the harnesses have been written but tests and validation
// have not been performed."
//
// Acceptance criterion: when the system factorizes an axis (splits a
// root into children), the children's derived_from.parent MUST point at
// a spec'd parent axis (rpg_class, star_sign, money_orientation), not
// at an unrelated dimension.
//
// What this spec asserts that would FAIL against a broken implementation:
//
//   1. STATIC INVARIANT — every non-root axis in the live registry has
//      derived_from.parent referring to a spec'd root or a (recursively)
//      spec-derived ancestor. There should be NO axis whose parent
//      cannot be traced back to the spec via derived_from links.
//
//   2. POSITIVE CASE — creating a derived axis via POST /axes/:id with
//      derived_from.parent='rpg_class' succeeds and the returned axis
//      record carries that genealogy. (Confirms the API+storage round-
//      trip preserves derived_from.)
//
//   3. NEGATIVE CASE — creating a derived axis pointing at a NON-EXISTENT
//      parent ('extractive_utility' — a residue-axis name from earlier
//      runs that has since been purged) must either:
//        a) be rejected by the API (preferred), OR
//        b) be flagged via an orphan/dangling-genealogy field that the
//           operator can query to triage.
//      Silent acceptance of a dangling parent is what got the operator
//      to 20+ axes in the first place. The plugin must surface this.
//
//   4. ROOT AXES INVARIANT — the 3 spec'd roots (rpg_class, star_sign,
//      money_orientation) have derived_from === null. They are the
//      ground truth from which all factorization must descend.

import { test, expect } from '@playwright/test';

const SPEC_ROOTS = ['rpg_class', 'star_sign', 'money_orientation'];
const PLUGIN_BASE = 'http://127.0.0.1:8002/api/plugins/user-personas';

test.describe('factorization stays on-spec (T6)', () => {
    test.setTimeout(60_000);

    // Track derived axes we create so we can clean up.
    let createdAxisIds = [];

    test.afterEach(async ({ request }) => {
        for (const id of createdAxisIds) {
            await request.delete(`${PLUGIN_BASE}/axes/${id}`).catch(() => {});
        }
        createdAxisIds = [];
    });

    test('static invariant — every axis in the registry has a spec-traceable ancestry', async ({ request }) => {
        const resp = await request.get(`${PLUGIN_BASE}/axes`);
        expect(resp.ok(), '/axes responds 2xx').toBeTruthy();
        const body = await resp.json();
        const axes = body.axes || [];

        // Build id → record map.
        const byId = Object.fromEntries(axes.map(a => [a.id, a]));

        // For each axis, walk derived_from.parent up to a root.
        // Acceptable: chain terminates at one of SPEC_ROOTS.
        // Unacceptable: chain ends at an axis with derived_from=null
        // whose id is NOT in SPEC_ROOTS (= a rogue root), OR the chain
        // walks into an axis-id that's not in the registry (dangling
        // parent), OR a cycle.
        const offenders = [];
        const cycleDetectors = [];
        for (const a of axes) {
            const trail = [a.id];
            let cur = a;
            const seen = new Set([a.id]);
            while (cur.derived_from && cur.derived_from.parent) {
                const parentId = cur.derived_from.parent;
                if (seen.has(parentId)) {
                    cycleDetectors.push({ axis: a.id, cycle_at: parentId, trail });
                    break;
                }
                seen.add(parentId);
                trail.push(parentId);
                const parent = byId[parentId];
                if (!parent) {
                    offenders.push({
                        axis: a.id,
                        reason: 'dangling-parent',
                        parent: parentId,
                        trail,
                    });
                    break;
                }
                cur = parent;
            }
            // Chain terminated. Root must be in SPEC_ROOTS.
            if (cur.derived_from === null || cur.derived_from === undefined) {
                if (!SPEC_ROOTS.includes(cur.id)) {
                    offenders.push({
                        axis: a.id,
                        reason: 'rogue-root',
                        root: cur.id,
                        trail,
                    });
                }
            }
        }

        if (cycleDetectors.length > 0) {
            console.error('genealogy cycles detected:', JSON.stringify(cycleDetectors, null, 2));
        }
        expect(cycleDetectors,
            'no genealogy cycles in axis registry').toEqual([]);

        if (offenders.length > 0) {
            console.error('off-spec axes:', JSON.stringify(offenders, null, 2));
        }
        expect(offenders,
            'every axis traces back to a spec root via derived_from')
            .toEqual([]);
    });

    test('spec roots are roots (derived_from === null) and only the spec roots are roots', async ({ request }) => {
        const resp = await request.get(`${PLUGIN_BASE}/axes`);
        const body = await resp.json();
        const axes = body.axes || [];

        const roots = axes.filter(a => !a.derived_from || a.derived_from === null);
        const rootIds = new Set(roots.map(a => a.id));

        // All spec roots present as roots.
        for (const id of SPEC_ROOTS) {
            expect(rootIds.has(id),
                `${id} present in registry as a root (derived_from === null)`)
                .toBe(true);
        }

        // No EXTRA roots beyond the spec set.
        const extraRoots = [...rootIds].filter(id => !SPEC_ROOTS.includes(id));
        if (extraRoots.length > 0) {
            console.error('extra roots:', extraRoots);
        }
        expect(extraRoots,
            'no extra roots beyond {rpg_class, star_sign, money_orientation}')
            .toEqual([]);
    });

    test('positive case — derived axis pointing at a spec root round-trips correctly', async ({ request }) => {
        const id = 't6_test_derived_axis_' + Date.now();
        createdAxisIds.push(id);

        const body = {
            name: 't6 test derived axis',
            def: '1: low end · 5: high end',
            kind: 'bio',
            scale_min: 1,
            scale_max: 5,
            derived_from: {
                parent: 'rpg_class',
                hypothesis: 'T6 positive case probe — split rpg_class on test dimension',
                sibling: null,
            },
        };

        const createResp = await request.post(`${PLUGIN_BASE}/axes/${id}`, { data: body });
        expect(createResp.ok(), `POST /axes/${id} responded 2xx`).toBeTruthy();
        const createBody = await createResp.json();
        expect(createBody.ok).toBe(true);
        expect(createBody.axis.id).toBe(id);
        expect(createBody.axis.derived_from.parent).toBe('rpg_class');

        // Verify it's visible in the listing.
        const listResp = await request.get(`${PLUGIN_BASE}/axes`);
        const list = (await listResp.json()).axes || [];
        const found = list.find(a => a.id === id);
        expect(found, 'derived axis surfaces in /axes listing').toBeTruthy();
        expect(found.derived_from.parent, 'derived_from.parent preserved on read')
            .toBe('rpg_class');
    });

    test('negative case — derived axis pointing at a nonexistent parent is either rejected or flagged', async ({ request }) => {
        const id = 't6_dangling_test_' + Date.now();
        createdAxisIds.push(id);

        const body = {
            name: 't6 dangling axis test',
            def: '1: low · 5: high',
            kind: 'bio',
            scale_min: 1,
            scale_max: 5,
            derived_from: {
                parent: 'extractive_utility',  // residue-axis name, NOT in current registry
                hypothesis: 'T6 negative probe — parent should not exist',
                sibling: null,
            },
        };

        const createResp = await request.post(`${PLUGIN_BASE}/axes/${id}`, { data: body });
        const status = createResp.status();
        const respBody = createResp.ok() ? await createResp.json() : null;

        // The plugin can do this one of two ways:
        //   a) REJECT with 400/422 — preferred ("no dangling parents")
        //   b) ACCEPT but flag via response field / register the orphan
        //      somewhere queryable
        // Silent acceptance (writes the axis with no acknowledgement
        // that its parent doesn't exist) is the failure mode we're
        // catching. ANY surface — error response OR a flagged field —
        // is acceptable; we just need a signal.
        if (!createResp.ok()) {
            // Path (a): rejected. Good. Status should be a client error (4xx).
            expect(status, 'rejection status is a client error (4xx)').toBeGreaterThanOrEqual(400);
            expect(status, 'rejection status is a client error (4xx)').toBeLessThan(500);
            return;
        }

        // Path (b): accepted. Then there must be a signal that the
        // parent is dangling. Acceptable surfaces:
        //   - respBody.dangling_parent === true
        //   - respBody.axis.derived_from.dangling === true
        //   - GET /axes returns the axis with an `orphan` field
        const has_dangling_signal =
            respBody?.dangling_parent === true ||
            respBody?.axis?.derived_from?.dangling === true ||
            respBody?.orphan === true ||
            respBody?.warnings?.some?.(w => /dangling|orphan|missing parent/i.test(w));

        if (!has_dangling_signal) {
            // Also check the GET endpoint for any orphan flag.
            const listResp = await request.get(`${PLUGIN_BASE}/axes`);
            const list = (await listResp.json()).axes || [];
            const found = list.find(a => a.id === id);
            const list_signal =
                found?.derived_from?.dangling === true ||
                found?.orphan === true;
            expect(list_signal,
                'plugin must surface a dangling-parent signal somewhere — silent acceptance of unrooted axes is the failure mode T6 was written to catch')
                .toBe(true);
        }
    });

    test('axis-card schema for spec roots — kind/scale/def all populated', async ({ request }) => {
        const resp = await request.get(`${PLUGIN_BASE}/axes`);
        const body = await resp.json();
        const byId = Object.fromEntries((body.axes || []).map(a => [a.id, a]));

        const expectedKinds = {
            rpg_class: 'bio',
            star_sign: 'bio',
            money_orientation: 'agent',
        };
        for (const [id, kind] of Object.entries(expectedKinds)) {
            const a = byId[id];
            expect(a, `${id} present`).toBeTruthy();
            expect(a.kind, `${id} kind`).toBe(kind);
            expect(a.scale_min, `${id} scale_min`).toBe(1);
            expect(a.scale_max, `${id} scale_max`).toBe(5);
            expect(typeof a.def, `${id} def is string`).toBe('string');
            expect(a.def.length, `${id} def non-trivial`).toBeGreaterThan(30);
            expect(a.derived_from, `${id} is a root (derived_from === null)`).toBeNull();
        }
    });
});
