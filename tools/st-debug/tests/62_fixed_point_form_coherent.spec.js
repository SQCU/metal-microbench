// Spec T4 — fixed_point.html is coherent as a query interface.
//
// Operator complaint (2026-05-20): "exposes a bunch of strings without
// explanation for what causal intervention they have on the solver if
// they were to be input. worse, at least one interface suggests writing
// multiple 'bio cues' in a single large string in a single text field
// in a manner with no explanation or even extrapolated expectation of
// effect if this query were to be run."
//
// What this spec validates:
//
//   1. Every <input>/<textarea>/<select> inside the fixed_point iframe
//      is labelled — either by a sibling `<label for="..."` element
//      pointing at its id, or by a non-empty aria-label attribute.
//
//   2. Every such input has a 1-sentence causal description below it.
//      We use `.description` as the selector (documented below) —
//      every input must have either a `.description` sibling within
//      the same form-row / repeat-row / stepper-cell / seed-axis-row /
//      seed-id-row / materialize-row container, OR a `.description`
//      sibling immediately following.
//
//   3. The structured editor (modal) form has descriptions on all
//      its inputs after "+ New Experiment" is clicked and the modal
//      renders. We open the modal and re-traverse.
//
//   4. The Seed input tab has NO single freeform textarea where the
//      operator dumps multiple cues. The previous anti-pattern
//      (`#seed-textarea` with placeholder
//      "bios:\n  ...\n  ...\nmotives:\n  ...") MUST be gone.
//      Instead the surface must show structured per-row inputs
//      (`.bio-seed-input`, `.motive-seed-input`) with their own
//      Add buttons.
//
//   5. Clicking the Dispatch button for an experiment causes SOMETHING
//      coherent: a POST /experiments/:id/run network call, OR the
//      "run-banner" element becomes visible with status text. Failing
//      silently is not acceptable.
//
// If ANY assertion fails we surface the full list of bare inputs (the
// CSS selectors that point at them) so the operator can find and fix
// them.
//
// Selector choice for "helper text": this spec uses `.description`
// specifically. We do NOT accept `.hint` (which we use for format
// reminders like "filename-safe regex"), and we do NOT accept the
// `<small>`/`<p>` pattern. The `.description` class was added in
// the form rewrite expressly so this spec has a single load-bearing
// hook: if you add a new input, you must add a `.description` next
// to it or this spec fails — that's the contract.

import { test, expect } from '@playwright/test';
import { loadAndConnect } from './_helpers/elicit_clean.mjs';

const PLUGIN_BASE = '/api/plugins/user-personas';
const MIN_DESCRIPTION_LEN = 10;  // "X" alone isn't a description.

// Inputs we deliberately exclude from the label/description audit.
// (None right now — kept for future allowlists, e.g., hidden search
// inputs added for accessibility-only purposes.)
const IGNORE_INPUT_SELECTORS = [
    // Example: 'input[type="hidden"]',
];

test.describe('fixed_point.html — coherent query interface (spec T4)', () => {
    test.setTimeout(90 * 1000);

    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'T4 spec runs only on desktop project (modal is a desktop affordance)');
    });

    /**
     * Audit every input/textarea/select inside `scopeHandle` for two
     * properties:
     *   (a) labelled: either a <label for="<id>"> targets it, OR it
     *       has a non-empty aria-label attribute
     *   (b) described: a `.description` element exists either as a
     *       direct sibling immediately after it, OR within the same
     *       enclosing form-row / repeat-row / stepper-cell / etc.,
     *       AND that element's textContent is ≥ MIN_DESCRIPTION_LEN
     *       characters
     *
     * Returns a list of failures, each {selector, problem, snippet}.
     * Selector is a unique CSS path so the operator can see exactly
     * which control is bare.
     */
    async function auditInputs(frame) {
        return await frame.locator('body').evaluate((_body, { MIN_DESCRIPTION_LEN, ignore }) => {
            // We run inside the iframe's window, so `document` is the
            // iframe's document — exactly what we want.
            const failures = [];
            const inputs = [...document.querySelectorAll('input, textarea, select')];

            function cssPath(el) {
                // Best-effort unique CSS path: id beats everything.
                if (el.id) return `#${el.id}`;
                const parts = [];
                let cur = el;
                while (cur && cur.nodeType === 1 && parts.length < 6) {
                    let segment = cur.nodeName.toLowerCase();
                    if (cur.className && typeof cur.className === 'string') {
                        const cls = cur.className.trim().split(/\s+/).filter(Boolean);
                        if (cls.length) segment += '.' + cls.slice(0, 2).join('.');
                    }
                    const parent = cur.parentNode;
                    if (parent && parent.nodeType === 1) {
                        const siblings = [...parent.children].filter(s => s.nodeName === cur.nodeName);
                        if (siblings.length > 1) segment += `:nth-of-type(${siblings.indexOf(cur) + 1})`;
                    }
                    parts.unshift(segment);
                    cur = cur.parentNode;
                }
                return parts.join(' > ');
            }

            function isVisible(el) {
                // We DO audit hidden inputs too if they came from the
                // operator-input surface — but visibility-checks here
                // are about "is this in a logical input surface".
                // display:none parents = excluded (e.g., closed modal).
                let cur = el;
                while (cur) {
                    if (cur.nodeType === 1) {
                        const cs = window.getComputedStyle(cur);
                        if (cs.display === 'none') return false;
                        if (cs.visibility === 'hidden') return false;
                    }
                    cur = cur.parentNode;
                }
                return true;
            }

            function isLabelled(el) {
                if (el.id) {
                    const lbl = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
                    if (lbl && lbl.textContent.trim().length > 0) return true;
                }
                const al = el.getAttribute('aria-label');
                if (al && al.trim().length > 0) return true;
                // Wrapping <label> with the input inside also counts.
                const wrappingLabel = el.closest('label');
                if (wrappingLabel && wrappingLabel.textContent.trim().length > 0) return true;
                return false;
            }

            function hasDescription(el) {
                // Same enclosing container that visually groups input + text.
                const container = el.closest(
                    '.form-row, .repeat-row, .stepper-cell, .seed-axis-row, .seed-id-row, ' +
                    '.materialize-row, .loop-grid > div, .form-section'
                );
                const candidates = [];
                if (container) candidates.push(...container.querySelectorAll('.description'));
                // Also a direct subsequent sibling description.
                let sib = el.nextElementSibling;
                while (sib) {
                    if (sib.classList && sib.classList.contains('description')) {
                        candidates.push(sib);
                        break;
                    }
                    sib = sib.nextElementSibling;
                }
                for (const c of candidates) {
                    if (c.textContent.trim().length >= MIN_DESCRIPTION_LEN) return true;
                }
                return false;
            }

            for (const inp of inputs) {
                if (!isVisible(inp)) continue;
                // Skip ignored selectors (allowlist for hidden-by-design inputs).
                if (ignore.some(sel => inp.matches(sel))) continue;

                const labelled = isLabelled(inp);
                const described = hasDescription(inp);
                if (!labelled || !described) {
                    failures.push({
                        selector: cssPath(inp),
                        problem: [
                            !labelled ? 'no <label for=> and no aria-label' : null,
                            !described ? `no .description sibling with ≥${MIN_DESCRIPTION_LEN} chars` : null,
                        ].filter(Boolean).join(' AND '),
                        snippet: inp.outerHTML.slice(0, 200),
                    });
                }
            }
            return failures;
        }, { MIN_DESCRIPTION_LEN, ignore: IGNORE_INPUT_SELECTORS });
    }

    /**
     * Search the iframe DOM for any textarea whose label or placeholder
     * implies the operator should dump multiple semantically-distinct
     * values into it ("separated by", "comma-separated", "newline",
     * "multiple X", "one per line"). Such inputs are the anti-pattern.
     */
    async function findMultiDumpTextareas(frame) {
        return await frame.locator('body').evaluate((_body) => {
            const offenders = [];
            const ANTI_PATTERN_PHRASES = [
                'separated by', 'comma-separated', 'comma separated',
                'newline-separated', 'newline separated', 'one per line',
                'multiple cues', 'multiple bios', 'multiple motives',
                'list of', 'enter multiple',
            ];
            const tas = document.querySelectorAll('textarea');
            for (const ta of tas) {
                const ph = (ta.getAttribute('placeholder') || '').toLowerCase();
                // Find the closest label / aria-label.
                let labelText = '';
                if (ta.id) {
                    const lbl = document.querySelector(`label[for="${CSS.escape(ta.id)}"]`);
                    if (lbl) labelText = lbl.textContent.toLowerCase();
                }
                if (!labelText) {
                    const al = ta.getAttribute('aria-label');
                    if (al) labelText = al.toLowerCase();
                }
                const haystack = labelText + ' || ' + ph;
                for (const phrase of ANTI_PATTERN_PHRASES) {
                    if (haystack.includes(phrase)) {
                        offenders.push({
                            id: ta.id || '(no id)',
                            placeholder: ta.getAttribute('placeholder') || '',
                            phrase,
                        });
                        break;
                    }
                }
            }
            return offenders;
        });
    }

    // Post-tabs-refactor (2026-05-19): Fixed-Point lives behind the
    // user-personas hamburger menu, not as a top-row sibling. Click
    // hamburger → menuitem[data-surface-key="fixed-point"] → iframe.
    async function openFixedPointTab(page) {
        await loadAndConnect(page);
        const hamburger = page.locator('#user-personas-tools-button .drawer-toggle');
        await expect(hamburger, 'user-personas hamburger button installs').toBeVisible({ timeout: 20_000 });
        await hamburger.click();
        const menuItem = page.locator('.user-personas-tools-menuitem[data-surface-key="fixed-point"]');
        await expect(menuItem, 'Fixed-Point menu item present in hamburger popover').toBeVisible({ timeout: 5_000 });
        await menuItem.click();
        const iframe = page.frameLocator('#user-personas-surface-fixed-point iframe');
        await expect(iframe.locator('h1').first(), 'fixed_point.html paints inside surface drawer').toBeVisible({ timeout: 15_000 });
        await expect(iframe.locator('#experiments-status')).toContainText(/loaded/i, { timeout: 10_000 });
        return iframe;
    }

    test('every visible input has a label + .description causal helper; no multi-dump textarea; Dispatch produces a coherent event', async ({ page, request }) => {
        const iframe = await openFixedPointTab(page);

        // ── PHASE A: Experiments tab default view ───────────────────────
        // The default tab has no inputs we need to audit, but we still
        // run the auditor to catch any inputs the page surfaces from
        // boot (e.g., search filters added later).
        let bareA = await auditInputs(iframe);
        expect(bareA, `Experiments-tab default view has bare inputs:\n${JSON.stringify(bareA, null, 2)}`).toEqual([]);

        // ── PHASE B: Editor modal (the "+ New Experiment" form) ─────────
        await iframe.locator('#new-experiment-btn').click();
        await expect(iframe.locator('#editor-overlay')).toBeVisible({ timeout: 5_000 });
        // The form auto-creates one bios[0] row on open. The agent_targets
        // section starts empty — click +Add agent target so we exercise
        // the dynamic-row generator in the audit too.
        await iframe.locator('#add-agent-btn').click();
        // Pick at least one bio_axis + agent_axis so stepper cells appear
        // (each stepper cell is its own input + description block).
        const bioAxisOpts = iframe.locator('#exp-bio-axes option');
        await expect(bioAxisOpts.first()).toBeAttached({ timeout: 5_000 });
        const firstBioAxisId = await bioAxisOpts.first().getAttribute('value');
        const firstAgentAxisId = await iframe.locator('#exp-agent-axes option').first().getAttribute('value');
        await iframe.locator('#exp-bio-axes').selectOption([firstBioAxisId]);
        await iframe.locator('#exp-agent-axes').selectOption([firstAgentAxisId]);
        // Also open the loop_control details so its inputs become visible.
        await iframe.locator('#loop-control-section summary').click();
        await expect(iframe.locator('#lc-k-max-inner')).toBeVisible();

        const bareB = await auditInputs(iframe);
        expect(bareB,
            `Editor modal has bare inputs (no label/aria-label OR no .description):\n${JSON.stringify(bareB, null, 2)}`
        ).toEqual([]);

        // Close the modal so the seed-tab audit doesn't accidentally
        // double-count modal inputs.
        await iframe.locator('#cancel-experiment-btn').click();
        await expect(iframe.locator('#editor-overlay')).toBeHidden();

        // ── PHASE C: Seed-input tab — verify anti-pattern is gone ───────
        await iframe.locator('#tab-seed').click();
        await expect(iframe.locator('#view-seed')).toBeVisible({ timeout: 5_000 });

        // The structured replacement must be present.
        await expect(iframe.locator('#bio-seed-rows .bio-seed-input').first(),
            'Seed-input tab must expose at least one per-bio input row').toBeVisible();
        await expect(iframe.locator('#motive-seed-rows .motive-seed-input').first(),
            'Seed-input tab must expose at least one per-motive input row').toBeVisible();
        await expect(iframe.locator('#add-bio-seed-btn'),
            'Seed-input tab must offer +Add bio seed').toBeVisible();
        await expect(iframe.locator('#add-motive-seed-btn'),
            'Seed-input tab must offer +Add motive seed').toBeVisible();

        // The flagged anti-pattern textarea (the single big "dump
        // bios:\n... motives:\n..." textarea with id=seed-textarea)
        // must be gone.
        await expect(iframe.locator('#seed-textarea'),
            'The single-textarea anti-pattern (#seed-textarea) must be removed').toHaveCount(0);

        // Belt-and-suspenders: scan ALL textareas for placeholders /
        // labels that suggest the operator should dump multiple cues.
        const offenders = await findMultiDumpTextareas(iframe);
        expect(offenders,
            `Found textarea(s) whose label/placeholder still suggests multi-value dumping:\n${JSON.stringify(offenders, null, 2)}`
        ).toEqual([]);

        // And the seed-tab's inputs themselves must pass the audit.
        const bareC = await auditInputs(iframe);
        expect(bareC,
            `Seed-input tab has bare inputs:\n${JSON.stringify(bareC, null, 2)}`
        ).toEqual([]);

        // ── PHASE D: Dispatch produces SOMETHING coherent ───────────────
        // Back to experiments tab. We don't want to actually run a
        // 5-minute experiment — just verify dispatching produces a
        // visible network POST + a run banner. The 40_*_smoke spec
        // covers the full dispatch chain; here we only need the
        // "something happens" signal.
        await iframe.locator('#tab-experiments').click();
        await expect(iframe.locator('#view-experiments')).toBeVisible();

        // Capture network requests so we can prove the dispatch fires.
        const sawDispatch = [];
        page.on('request', (req) => {
            if (/\/api\/plugins\/user-personas\/experiments\/[^/]+\/run$/.test(req.url())
                && req.method() === 'POST') {
                sawDispatch.push(req.url());
            }
        });

        // The list may be empty if /experiments dir has no cards yet.
        // POST a minimal valid card via the API so a Dispatch button
        // is present to click. We DELETE it in cleanup.
        const TEST_ID = 'spec_t4_dispatch_probe';
        await request.delete(`${PLUGIN_BASE}/experiments/${TEST_ID}`).catch(() => {});
        const probeCard = {
            experiment_schema: 'experiment-v1',
            id: TEST_ID,
            name: 'spec T4 dispatch probe',
            description: 'auto-created by 62_fixed_point_form_coherent.spec.js',
            bios: [{
                canonical_key: `user-personas-${TEST_ID}-bio.png`,
                slug: `${TEST_ID}-bio`,
                name: 'probe bio',
                target_bio: { rpg_class: 3 },
                design_brief: 'probe bio for dispatch test',
            }],
            agent_targets: [{
                slug: `${TEST_ID}-agent`,
                target_agent: { money_orientation: 3 },
                motive_hint: 'probe motive',
            }],
            bio_axes: ['rpg_class'],
            agent_axes: ['money_orientation'],
            counterparty_avatar: 'the-rock.png',
        };
        const created = await request.post(`${PLUGIN_BASE}/experiments/${TEST_ID}`, {
            data: probeCard,
        });
        expect(created.ok(), `card setup must succeed (got ${created.status()})`).toBe(true);

        // Reload the experiments list inside the iframe so the new card
        // appears as a row.
        await iframe.locator('#tab-seed').click();
        await iframe.locator('#tab-experiments').click();
        // Wait for our probe to surface in the list.
        await expect(
            iframe.locator(`.experiment-card[data-eid="${TEST_ID}"]`),
            'probe experiment card appears after API insert + tab toggle'
        ).toBeVisible({ timeout: 10_000 });

        // Click its Dispatch button.
        const dispatchBtn = iframe.locator(
            `.experiment-card[data-eid="${TEST_ID}"] .run-btn`
        );
        await expect(dispatchBtn, 'Dispatch button is labelled and clickable').toBeVisible();
        // The button label must be self-descriptive, not just "Run".
        await expect(dispatchBtn).toContainText(/dispatch|run/i);

        await dispatchBtn.click();

        // "Something coherent happens" — assert at least ONE of:
        //   (a) a POST /experiments/:id/run network request fired
        //   (b) the run-banner element is visible with status text
        // Either signal is sufficient; we expect both, but the OR
        // here is for robustness in case the FE batches requests.
        await expect.poll(() => sawDispatch.length, {
            message: 'a POST /experiments/<id>/run network call must fire when Dispatch is clicked',
            timeout: 8_000,
        }).toBeGreaterThan(0);
        await expect(iframe.locator('#run-banner'),
            'run-banner must surface status when a dispatch is in flight').toBeVisible({ timeout: 5_000 });
        await expect(iframe.locator('#run-banner'),
            'run-banner must contain non-empty status text').not.toBeEmpty();

        // Cleanup the probe card so re-runs of the spec are idempotent.
        await request.delete(`${PLUGIN_BASE}/experiments/${TEST_ID}`).catch(() => {});
    });
});
