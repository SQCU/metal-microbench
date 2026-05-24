// Context-driven suggester refactor — Playwright pixel-space verification.
//
// Validates the FE surface that lands per `docs/ui_spec_context_suggester.md`:
//
//   1. Drawer button (#user-suggester-button) installs + iframe loads suggester.html
//   2. Initial state: _meta strip placeholder, ranked list empty
//   3. Add turn + "Rank for this context" → POST /yapper-seed
//   4. Ranked rows render with bio name, agent name, distance pill (class
//      programmatically asserted), why text, working Suggest button
//   5. _meta strip: target_signature pills + candidates_considered consistent
//      with bios_total × agents_total
//   6. + More: row count grows, OR button disables with "no more compositions"
//   7. Synthesize CTA: top[0].distance > 2.0 → CTA renders; clicking opens
//      FP-tab drawer and navigates the FP iframe to URL containing
//      target_bio_signature= (not the editor pre-pop — that's Doc B's spec)
//   8. End-to-end: Suggest on a top row → candidate in left-side feed
//
// CORPUS SEEDING REQUIREMENT
// --------------------------
// This spec assumes the corpus contains at least 1 (bio, agent) composition
// (i.e. agents/ is non-empty). The canonical seed is `lock_in_tetrad`
// (plugins/user-personas/experiments/lock_in_tetrad.json) which produces
// 4 compositions (2 bios × 2 agent_targets) when run to convergence.
// Seeding is performed by running that experiment via the Fixed-Point
// Iteration tab (#user-fixed-point-button → click "Run" on the
// lock_in_tetrad card); each run takes ~5 minutes per bio. DO NOT seed
// via one-shot helper scripts — the only legitimate corpus-growth path
// is the fixed-point experiments themselves.
//
// If the corpus has 0 agents, this spec exercises the empty-corpus
// branch instead (pending_synthesis pills render, ranked list shows
// empty-state). It explicitly skips +More / per-row Suggest / Synthesize
// CTA assertions in that mode and surfaces the seeding requirement as a
// test annotation so the failure-vs-data-state confusion is unambiguous.
//
// If the corpus has compositions but NOT enough to bump K_top + K_side
// beyond the corpus ceiling (e.g. only 1 bio × 1 agent = 1 composition),
// step 6 asserts the +More button DISABLES with the "no more
// compositions" label rather than expecting growth.

import { test, expect } from '@playwright/test';
import { loadAndConnect } from './_helpers/elicit_clean.mjs';
import { openPersonaSurface } from './_helpers/open_persona_surface.js';

const PLUGIN_BASE = '/api/plugins/user-personas';

async function fetchCorpusState(page) {
    // Side-channel snapshot of corpus state via the same endpoint the FE
    // calls. Used to decide which assertion branch to take.
    const data = await page.evaluate(async (base) => {
        const r = await fetch(`${base}/yapper-seed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                chat_context_summary: 'corpus-probe',
                K_top: 3, K_side: 3,
            }),
        });
        if (!r.ok) return { error: `HTTP ${r.status}` };
        return await r.json();
    }, PLUGIN_BASE);
    return data;
}

test.describe('context-driven suggester', () => {
    // 8 minutes: ~4 bridge round-trips at ~30s each (probe, rank, suggest,
    // off-corpus rank) plus page-load + assertion overhead, with headroom
    // for cold prefix-cache state. Earlier the test hit 38s on a warm
    // cache and >180s on cold — the bridge's prefix-maxx serializes the
    // 3 parallel judges of /signature-extract through one prefill, so
    // each rank gates ~25-35s.
    test.setTimeout(8 * 60 * 1000);

    test('drawer + ranking + meta + +More + synthesize CTA + per-row suggest', async ({ page }) => {
        await loadAndConnect(page);

        // ── (1) Open the suggester surface via the hamburger popover.
        // (The per-tab .drawer-toggle is display:none after sillytavern-fork
        // e2973179d; clicking directly on #user-suggester-button or its
        // .drawer-toggle child bypasses the navigability invariant.)
        await openPersonaSurface(page, 'suggester');

        const iframe = page.frameLocator('iframe[src*="suggester.html"]');
        await expect(iframe.locator('h1'), 'suggester.html renders inside iframe').toBeVisible({ timeout: 15_000 });
        await expect(iframe.locator('h1')).toContainText(/Context-driven suggester|suggester/i);

        // ── (2) Initial state: _meta strip placeholder, ranked list empty.
        const metaStrip = iframe.locator('#meta-strip');
        await expect(metaStrip, '_meta strip shows placeholder text initially')
            .toContainText(/Click "Rank for this context"/);
        await expect(metaStrip).toHaveClass(/placeholder/);
        await expect(iframe.locator('#ranked-list .empty-state'),
            'ranked list is in empty state initially').toBeVisible();
        // Bio-list-driven artefacts are gone: no show-overlay checkbox,
        // no overlay-select dropdowns, no bio-card elements.
        await expect(iframe.locator('#show-overlay-only'),
            'show-overlay-only checkbox is gone').toHaveCount(0);
        await expect(iframe.locator('.overlay-select'),
            'overlay-select dropdowns are gone').toHaveCount(0);
        await expect(iframe.locator('.bio-card'),
            'bio-card elements are gone').toHaveCount(0);

        // ── Probe corpus state via /yapper-seed side-channel, then branch.
        const corpusProbe = await fetchCorpusState(page);
        const corpusHasCompositions =
            Array.isArray(corpusProbe?.top) && (corpusProbe.top.length + (corpusProbe.side?.length || 0)) > 0;

        // ── (3) Add a turn + Rank.
        await iframe.locator('#new-content').fill('rolls a die and grins, ready for the next move');
        await iframe.locator('#add-turn-btn').click();
        await expect(iframe.locator('.history-msg').first(),
            'turn appended to history scratchpad').toBeVisible();

        const rankBtn = iframe.locator('#rank-btn');
        await rankBtn.click();
        // Wait for rank request to complete (button label reverts).
        await expect(rankBtn).toBeEnabled({ timeout: 60_000 });

        // ── (5) _meta strip populated.
        await expect(metaStrip, '_meta strip no longer shows placeholder text')
            .not.toContainText(/Click "Rank for this context"/);
        await expect(metaStrip).not.toHaveClass(/placeholder/);
        // K_top / K_side echo present.
        const ktopPill = iframe.locator('#meta-ktop');
        await expect(ktopPill).toBeVisible();
        await expect(ktopPill).toContainText(/K_top=\d+/);
        const ksidePill = iframe.locator('#meta-kside');
        await expect(ksidePill).toBeVisible();
        await expect(ksidePill).toContainText(/K_side=\d+/);

        if (corpusHasCompositions) {
            // candidates_considered = bios_total × agents_total decomposition.
            // Read the data-* attributes off the pill (programmatic, not pixel
            // count) and verify the invariant.
            const considered = iframe.locator('#meta-considered');
            await expect(considered).toBeVisible();
            const counts = await considered.evaluate(el => ({
                considered: +el.dataset.considered,
                bios: +el.dataset.bios,
                agents: +el.dataset.agents,
            }));
            expect(counts.considered, 'candidates_considered is set').toBeGreaterThan(0);
            if (counts.bios > 0 && counts.agents > 0) {
                expect(counts.considered,
                    `candidates_considered (${counts.considered}) ≤ bios×agents (${counts.bios}×${counts.agents}=${counts.bios * counts.agents})`)
                    .toBeLessThanOrEqual(counts.bios * counts.agents);
            }
            // target_signature pills are present and non-empty.
            await expect(iframe.locator('#meta-target-sig .sig-pill').first(),
                'at least one target_signature pill renders').toBeVisible();
            const sigPillCount = await iframe.locator('#meta-target-sig .sig-pill').count();
            expect(sigPillCount, 'target_signature has ≥ 1 axis').toBeGreaterThan(0);

            // ── (4) Verify ranked rendering — at least one top row.
            const topRows = iframe.locator('.ranked-row.top');
            await expect(topRows.first(), 'at least one top row rendered').toBeVisible();
            const firstRow = topRows.first();
            // bio name, distance pill, why text, Suggest button.
            await expect(firstRow.locator('.ranked-name')).toBeVisible();
            const distPill = firstRow.locator('.distance-pill');
            await expect(distPill).toBeVisible();
            await expect(distPill).toContainText(/L2=\d+\.\d+/);
            // Distance pill class is one of the spec'd colour classes.
            const distClass = await distPill.evaluate(el => el.className);
            expect(distClass, 'distance pill has one of {near, mid, far}')
                .toMatch(/(near|mid|far)/);
            await expect(firstRow.locator('.ranked-why')).not.toBeEmpty();
            await expect(firstRow.locator('button.suggest-btn')).toBeEnabled();

            // ── (6) + More: capture row count, click, assert growth or
            //        ceiling-disable.
            const initialRowCount = await iframe.locator('.ranked-row').count();
            const moreBtn = iframe.locator('#more-btn');
            await expect(moreBtn).toBeVisible();
            await expect(moreBtn).toBeEnabled();
            await moreBtn.click();
            // Wait for the re-render to settle: rank button re-enables.
            await expect(rankBtn).toBeEnabled({ timeout: 60_000 });
            const afterRowCount = await iframe.locator('.ranked-row').count();
            const ceilingNote = iframe.locator('#ceiling-note');
            const ceilingHit = await ceilingNote.isVisible();
            if (ceilingHit) {
                // Corpus too small to bump beyond initial K. + More
                // disables permanently with "no more compositions".
                await expect(ceilingNote).toContainText(/no more compositions/i);
                await expect(moreBtn).toBeDisabled();
                expect(afterRowCount,
                    'when ceiling hit, row count must not have grown')
                    .toBeLessThanOrEqual(initialRowCount);
            } else {
                // Row count must have grown.
                expect(afterRowCount,
                    `+More grew row count (was ${initialRowCount}, now ${afterRowCount})`)
                    .toBeGreaterThan(initialRowCount);
            }

            // ── (8) End-to-end suggest from a top row.
            const suggestBtn = topRows.first().locator('button.suggest-btn');
            await suggestBtn.click();
            // Wait for either a candidate to appear or for the button to
            // re-enable (covers both success + error rendering paths). We
            // assert at least one candidate item lands in the feed.
            await expect(iframe.locator('#candidates-feed .candidate').first(),
                'a candidate lands in the left-side feed').toBeVisible({ timeout: 60_000 });
            // Candidate has bio badge + non-empty text.
            const cand = iframe.locator('#candidates-feed .candidate').first();
            await expect(cand.locator('.candidate-bio')).not.toBeEmpty();
            await expect(cand.locator('.candidate-text')).not.toBeEmpty();

            // ── (7) Synthesize CTA: send an off-corpus chat-context, rank
            //        again, then if top[0].distance > 2.0, assert the CTA
            //        renders and the click navigates FP iframe.
            await iframe.locator('#clear-btn').click();
            await iframe.locator('#new-content').fill(
                'discusses early-modern Spanish viticulture in iambic pentameter, ' +
                'with detailed reference to Garcilaso de la Vega and grafting practices');
            await iframe.locator('#add-turn-btn').click();
            await rankBtn.click();
            await expect(rankBtn).toBeEnabled({ timeout: 60_000 });
            // CTA may or may not show — depends on the judge's signature
            // for the off-corpus prose. If it shows, assert the FP-tab
            // navigation works. If not, this branch is data-dependent
            // (judge happened to score the off-prose close to the corpus)
            // and we surface that via test.info().annotations.
            const cta = iframe.locator('#synthesize-cta');
            const ctaVisible = await cta.isVisible();
            if (ctaVisible) {
                await expect(cta).toContainText(/Nothing in the corpus is close/);
                await expect(cta.locator('#cta-best-distance')).toContainText(/L2=\d+\.\d+/);
                await iframe.locator('#synthesize-btn').click();
                // FP-tab drawer opens, iframe src carries target_bio_signature.
                const fpContent = page.locator('#user-fixed-point-button .drawer-content');
                await expect(fpContent).toHaveClass(/openDrawer/, { timeout: 5_000 });
                const fpIframe = page.locator('#user_fixed_point_iframe');
                // Poll until the src reflects the param.
                await expect.poll(
                    async () => await fpIframe.evaluate(el => el.src),
                    { timeout: 10_000, message: 'FP iframe navigates to URL with target_bio_signature= param' }
                ).toMatch(/target_bio_signature=/);
            } else {
                test.info().annotations.push({
                    type: 'synthesize-cta-not-exercised',
                    description: 'off-corpus prose scored close enough to corpus that top[0].distance ≤ 2.0; CTA branch not exercised in this run',
                });
            }
        } else {
            // ── Empty-corpus branch. Document the seeding requirement.
            test.info().annotations.push({
                type: 'corpus-empty',
                description: 'agents/ is empty — run lock_in_tetrad via the FP tab to seed 4 compositions, then this spec exercises full ranking flow',
            });
            // _meta should still surface pending_synthesis pills.
            const pending = iframe.locator('#meta-pending .pending-bio');
            await expect(pending.first(), 'pending_synthesis bios are listed').toBeVisible();
            // Ranked list shows top + side empty-state placeholders.
            await expect(iframe.locator('.ranked-section .empty-state').first()).toBeVisible();
            // The +More row is rendered with the button (visible but
            // there's nothing to expand). The ceiling note may show
            // once +More yields the same empty response.
            const moreBtn = iframe.locator('#more-btn');
            await expect(moreBtn).toBeVisible();
            await moreBtn.click();
            await expect(rankBtn).toBeEnabled({ timeout: 60_000 });
            const ceilingNote = iframe.locator('#ceiling-note');
            // After +More with no growth, ceiling note shows and button disables.
            await expect(ceilingNote).toBeVisible();
            await expect(ceilingNote).toContainText(/no more compositions/i);
            await expect(moreBtn).toBeDisabled();

            // Clicking a pending_synthesis pill opens the FP tab.
            await pending.first().click();
            const fpContent = page.locator('#user-fixed-point-button .drawer-content');
            await expect(fpContent).toHaveClass(/openDrawer/, { timeout: 5_000 });
            const fpIframe = page.locator('#user_fixed_point_iframe');
            await expect.poll(
                async () => await fpIframe.evaluate(el => el.src),
                { timeout: 10_000, message: 'FP iframe navigates to URL with prefill_bio= param' }
            ).toMatch(/prefill_bio=/);
        }
    });
});
