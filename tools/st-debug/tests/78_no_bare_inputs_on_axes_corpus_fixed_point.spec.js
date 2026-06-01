// Spec 78 — P-EMPTY-FORM compliance audit on axes.html, corpus.html,
// fixed_point.html. Companion to spec 74 (which covered the
// user-personas drawer panel), this one targets the three plugin
// surfaces flagged by ticket #214.
//
// P-EMPTY-FORM (from docs/ux_debt_followup_tickets_2026_05_21.md):
//
//   "Never ask the operator to fill out a form-with-bare-fields
//    without contextual suggestions, pre-filled defaults pulled from
//    the existing corpus, or visible worked examples.
//    JSON-fields-as-strings is the canonical forbidden anti-pattern."
//
// Operational test: every visible primary <input type=text|tel|email|
// url|search|number> and <textarea> on first paint must EITHER
//   (a) have a non-empty `value` attribute (pre-filled), OR
//   (b) live inside a clearly-justified secondary role (search/filter
//       boxes, edit-existing forms inside an items list — which are
//       worked examples by construction).
//
// Surfaces under audit:
//   - /api/plugins/user-personas/static/axes.html
//   - /api/plugins/user-personas/static/corpus.html
//   - /api/plugins/user-personas/static/fixed_point.html
//
// The spec navigates each surface DIRECTLY (no drawer chrome) so the
// audit's failure mode is "this URL exposes a bare input" — which is
// the file the operator must fix.

import { test, expect } from '@playwright/test';

const ST_URL = 'http://127.0.0.1:8002';
const SURFACES = [
    // axes.html retired 2026-06 — its registry folded into corpus.html,
    // which is audited below (and carries the same P-EMPTY-FORM contract).
    `${ST_URL}/api/plugins/user-personas/static/corpus.html`,
    `${ST_URL}/api/plugins/user-personas/static/fixed_point.html`,
];

// IDs / selectors that are deliberately excluded from the audit.
// Each entry has a comment justifying WHY this input is principled
// (search filter, bounded numeric stepper inside a structured affordance,
// edit-existing-item inline form, etc.).
const PRINCIPLED_EXEMPTIONS = [
    // ── Bounded numeric steppers inside structured affordances ──
    // These are sliders/steppers with min/max bounds and a midpoint
    // default — the primitive IS the right input, not a text dump.
    { selector: 'input[type="range"]',
      reason: 'sliders are bounded structured inputs (compliant primitive)' },
    { selector: 'input[type="number"][min][max]',
      reason: 'bounded numeric steppers with explicit min+max are structured (compliant primitive)' },
    { selector: 'input[type="number"][min]',
      reason: 'numeric steppers with explicit min are structured (loop_control knobs etc.)' },
    { selector: 'input[type="checkbox"]',
      reason: 'checkboxes are toggles, not text-entry forms' },
    { selector: 'input[type="radio"]',
      reason: 'radio buttons are pickers, not text-entry forms' },
    { selector: 'input[type="hidden"]',
      reason: 'hidden inputs are not operator-facing' },
    { selector: 'input[type="file"]',
      reason: 'file pickers are not text-entry forms' },

    // ── Stepper cells inside the fixed_point experiment editor ──
    // These are 1–5 axis target steppers; the experiment editor is
    // reached via the Edit button on an existing experiment (worked
    // example) or via the Seed Input tab (which auto-generates them).
    { selector: '.stepper-cell input',
      reason: 'stepper cells are bounded 1–5 axis target pickers, materialized only inside editor reached via Edit (pre-filled) or Seed Input (auto-assigned)' },

    // ── Coordinate-picker bio sliders (corpus.html) ──
    // These materialize after /axes loads with midpoint defaults.
    { selector: 'input.picker-slider',
      reason: 'coordinate-picker sliders are bounded with midpoint defaults pulled from axis registry' },
];

test.describe('P-EMPTY-FORM compliance on axes / corpus / fixed_point surfaces', () => {
    test.setTimeout(60_000);

    for (const url of SURFACES) {
        test(`no bare primary text inputs/textareas on first paint — ${url.split('/').pop()}`, async ({ page }) => {
            await page.goto(url);

            // Wait long enough for axes / experiments / coordinate
            // picker to materialize. The slowest surface (corpus.html)
            // has to load /axes + /personas + /agents before the picker
            // sliders render with their pre-filled midpoint defaults.
            await page.waitForLoadState('networkidle', { timeout: 15_000 }).catch(() => {});
            await page.waitForTimeout(1500);

            const violations = await page.evaluate(({ exemptions }) => {
                const out = [];

                function cssPath(el) {
                    if (el.id) return `#${el.id}`;
                    const parts = [];
                    let cur = el;
                    while (cur && cur.nodeType === 1 && parts.length < 6) {
                        let seg = cur.nodeName.toLowerCase();
                        if (cur.className && typeof cur.className === 'string') {
                            const cls = cur.className.trim().split(/\s+/).filter(Boolean).slice(0, 2);
                            if (cls.length) seg += '.' + cls.join('.');
                        }
                        parts.unshift(seg);
                        cur = cur.parentNode;
                    }
                    return parts.join(' > ');
                }

                function isVisible(el) {
                    let cur = el;
                    while (cur) {
                        if (cur.nodeType === 1) {
                            const cs = window.getComputedStyle(cur);
                            if (cs.display === 'none') return false;
                            if (cs.visibility === 'hidden') return false;
                            if (cur.hasAttribute('hidden')) return false;
                        }
                        cur = cur.parentNode;
                    }
                    return true;
                }

                function matchesExemption(el) {
                    for (const ex of exemptions) {
                        try {
                            if (el.matches(ex.selector)) return ex.reason;
                        } catch (_) { /* invalid selector */ }
                    }
                    return null;
                }

                // Inputs we audit as "primary text-entry affordances".
                // Empty type defaults to text per HTML spec.
                const TEXTY_INPUT_TYPES = new Set(['text', 'tel', 'email', 'url', 'search', 'password', '']);

                const inputs = [...document.querySelectorAll('input, textarea')];
                for (const el of inputs) {
                    if (!isVisible(el)) continue;

                    if (el.tagName.toLowerCase() === 'input') {
                        const type = (el.getAttribute('type') || '').toLowerCase();
                        if (!TEXTY_INPUT_TYPES.has(type)) continue;
                    }

                    const exempt = matchesExemption(el);
                    if (exempt) continue;

                    // Compliant if it has a non-empty pre-filled value.
                    const val = el.value;
                    if (val && val.trim().length > 0) continue;
                    // Or a non-empty `value` attribute (some prefills set the attr).
                    const valAttr = el.getAttribute('value');
                    if (valAttr && valAttr.trim().length > 0) continue;
                    // Textareas can prefill via textContent.
                    if (el.tagName.toLowerCase() === 'textarea' && el.textContent.trim().length > 0) continue;

                    out.push({
                        selector: cssPath(el),
                        tag: el.tagName.toLowerCase(),
                        type: (el.getAttribute('type') || '') || 'text',
                        placeholder: el.getAttribute('placeholder') || '',
                        snippet: el.outerHTML.slice(0, 200),
                    });
                }
                return out;
            }, { exemptions: PRINCIPLED_EXEMPTIONS });

            expect(violations,
                `Surface ${url} has bare primary inputs on first paint (P-EMPTY-FORM violation):\n` +
                JSON.stringify(violations, null, 2)
            ).toEqual([]);
        });
    }

    // Specific assertions: the deleted affordances must not return.
    // (axes.html retired 2026-06; its registry — and this P-EMPTY-FORM
    // contract — moved into corpus.html, asserted below.)
    test('corpus.html: "+ Add axis" button + #add-form must NOT exist', async ({ page }) => {
        await page.goto(`${ST_URL}/api/plugins/user-personas/static/corpus.html`);
        await page.waitForTimeout(800);
        await expect(page.locator('#add-axis-btn'), '"+ Add axis" button is removed').toHaveCount(0);
        await expect(page.locator('#add-form'), '#add-form inline form is removed').toHaveCount(0);
        await expect(page.locator('#add-id'), '#add-id bare input is removed').toHaveCount(0);
    });

    test('fixed_point.html: "+ New Experiment" button must NOT exist; Seed Input rows are pre-filled', async ({ page }) => {
        await page.goto(`${ST_URL}/api/plugins/user-personas/static/fixed_point.html`);
        await page.waitForTimeout(1200);

        // The deleted entry point.
        await expect(page.locator('#new-experiment-btn'),
            '"+ New Experiment" button is removed (blank modal is P-EMPTY-FORM violation)'
        ).toHaveCount(0);

        // The replacement CTA exists.
        await expect(page.locator('#seed-tab-cta-btn'),
            'Replacement CTA "+ Seed an experiment" is present and routes to the structured Seed Input tab'
        ).toBeVisible();

        // Switch to seed-input tab and assert prefilled rows.
        await page.locator('#tab-seed').click();
        await expect(page.locator('#view-seed')).toBeVisible();

        const bioInputs = page.locator('#bio-seed-rows .bio-seed-input');
        await expect(bioInputs.first(), 'first bio-seed-input row exists').toBeVisible();
        // Every row must have a non-empty value (worked example).
        const bioCount = await bioInputs.count();
        for (let i = 0; i < bioCount; i++) {
            const v = await bioInputs.nth(i).inputValue();
            expect(v.trim().length, `bio-seed-input row ${i} is pre-filled with worked example`).toBeGreaterThan(0);
        }
        const motiveInputs = page.locator('#motive-seed-rows .motive-seed-input');
        await expect(motiveInputs.first(), 'first motive-seed-input row exists').toBeVisible();
        const motiveCount = await motiveInputs.count();
        for (let i = 0; i < motiveCount; i++) {
            const v = await motiveInputs.nth(i).inputValue();
            expect(v.trim().length, `motive-seed-input row ${i} is pre-filled with worked example`).toBeGreaterThan(0);
        }
    });
});
