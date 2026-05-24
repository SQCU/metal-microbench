// Spec T5 — axes registry matches the precollapsed spec.
//
// The user's directive (2026-05-20): the axes registry is the
// precollapsed tiny set of THREE axes:
//   - rpg_class     (bio,   wizard <-> rogue)
//   - star_sign     (bio,   cancer <-> sagittarius)
//   - money_orientation (agent, pure-theft <-> romance-leveraged-theft)
//
// Any other axis present is residue (from background axis_splitter
// runs that auto-factorized during prior synthesis dispatches). The
// principled behavior is operator-triggered factorization only.
//
// This spec locks the registry at the spec'd 3. If a future
// factorization run legitimately splits one of these into children,
// the test will fail with a list of unexpected axes — operator can
// then either promote them to canonical or clean them out.

import { test, expect } from '@playwright/test';

const PRECOLLAPSED_SPEC = new Set(['rpg_class', 'star_sign', 'money_orientation']);

test.describe('axes registry matches precollapsed spec (T5)', () => {
    test.setTimeout(15_000);

    test('GET /axes returns exactly the 3 precollapsed ROOT axes (derived axes OK; T6 validates them)', async ({ page }) => {
        // T5's invariant: the precollapsed STARTING set is 3 root axes.
        // After factorization (T6's machinery) derived axes may exist,
        // each carrying derived_from.parent → one of the 3 roots. T5
        // locks the ROOT inventory; T6 locks the derivation parent
        // attribution. Together they bound the registry to {spec'd roots}
        // ∪ {derived axes with on-spec parents}, no off-spec roots.
        const resp = await page.request.get('http://127.0.0.1:8002/api/plugins/user-personas/axes');
        expect(resp.ok(), 'axes endpoint responds 2xx').toBeTruthy();
        const body = await resp.json();
        const axes = body.axes || [];

        const roots = axes.filter(a => !a.derived_from);
        const derived = axes.filter(a => a.derived_from);

        const rootIds = new Set(roots.map(a => a.id));
        const missing = [...PRECOLLAPSED_SPEC].filter(id => !rootIds.has(id));
        const extraRoots = [...rootIds].filter(id => !PRECOLLAPSED_SPEC.has(id));

        expect(missing,
            `precollapsed parents missing from root set: ${missing.join(', ')}`).toEqual([]);
        expect(extraRoots,
            `unexpected ROOT axes (off-spec): ${extraRoots.join(', ')}`).toEqual([]);

        expect(roots.length, 'exactly 3 root axes (the precollapsed set)').toBe(3);

        // Annotation surfaces derived count so the run log captures the
        // factorization state without log scraping.
        test.info().annotations.push({
            type: 'axis-inventory',
            description: `roots=${roots.length} (precollapsed) derived=${derived.length} total=${axes.length}`,
        });
    });

    test('each spec axis has the expected kind + scale + non-null def', async ({ page }) => {
        const resp = await page.request.get('http://127.0.0.1:8002/api/plugins/user-personas/axes');
        const body = await resp.json();
        const byId = Object.fromEntries(body.axes.map(a => [a.id, a]));

        const expectations = {
            rpg_class:          { kind: 'bio',   minPoles: ['wizard', 'rogue'] },
            star_sign:          { kind: 'bio',   minPoles: ['cancer', 'sagittarius'] },
            money_orientation:  { kind: 'agent', minPoles: ['theft', 'romance'] },
        };

        for (const [id, e] of Object.entries(expectations)) {
            const a = byId[id];
            expect(a, `${id} present`).toBeTruthy();
            expect(a.kind, `${id} kind`).toBe(e.kind);
            expect(a.scale_min, `${id} scale_min`).toBe(1);
            expect(a.scale_max, `${id} scale_max`).toBe(5);
            expect(typeof a.def, `${id} def is string`).toBe('string');
            expect(a.def.length, `${id} def is non-trivial`).toBeGreaterThan(20);

            // Both poles described in the def text (loose match — the
            // def's 1 and 5 anchors should mention the polar concepts).
            const defLower = a.def.toLowerCase();
            for (const pole of e.minPoles) {
                expect(defLower,
                    `${id} def mentions polar concept "${pole}"`).toContain(pole.toLowerCase());
            }

            // Derived_from should be null for root axes (precollapsed = root).
            expect(a.derived_from, `${id} is a root axis (derived_from null)`).toBeNull();
        }
    });
});
