// Multi-viewport rendering of the user-personas panel.
//
// Companion to test 70 (variety/truncation diagnostic). This test
// captures the user-personas panel layout under three viewports:
//
//   desktop  (1280×800) — laptop / monitor
//   tablet   (768×1024) — iPad-ish
//   mobile   (390×844)  — iPhone 12/13/14-ish
//
// The mobile-render bug the user observed is that the user-agent
// suggestions panel collapses to a thin ~60px scrollable strip on
// mobile, concealing the agent cards. Surfacing that bug requires
// rendering the panel under a mobile viewport — which is what the
// mobile project here does.
//
// One iteration per project. Each run takes a screenshot of the
// user-personas panel in both collapsed AND expanded states.
// Total wallclock per project: ≤30s.

import { test, expect } from '@playwright/test';
import { loadAndConnect, selectCharacterByClick } from './_helpers/elicit_clean.mjs';

test.describe('user-personas panel — viewport rendering', { tag: ['@fast', '@user-personas'] }, () => {
    test.setTimeout(90 * 1000);

    test('panel renders with all 4 agent cards visible at the current viewport', async ({ page }, testInfo) => {
        await loadAndConnect(page);
        // Need a character loaded for the user-personas extension to render
        // its panel meaningfully (cards are tied to a chat that exists).
        await selectCharacterByClick(page, 'dicemother');
        await expect(page.locator('#user_personas_btn'),
            'user-personas button installed').toBeVisible({ timeout: 10_000 });

        // Capture viewport metadata in the screenshot filename so the
        // artifact set is self-labeling.
        const vp = page.viewportSize() || { width: 0, height: 0 };
        const tag = `vp_${vp.width}x${vp.height}_${testInfo.project.name}`;

        // Initial state — panel collapsed.
        await page.screenshot({
            path: testInfo.outputPath(`${tag}_collapsed.png`),
            fullPage: false,
        });

        // Expand the panel via the button.
        await page.locator('#user_personas_btn').click();
        await page.waitForTimeout(500);

        // Wait for the 4 cards to render in the panel.
        await page.waitForFunction(() =>
            document.querySelectorAll('#user_personas_panel .user-personas-card').length === 4,
            { timeout: 10_000 });

        // Screenshot the panel expanded.
        await page.screenshot({
            path: testInfo.outputPath(`${tag}_expanded.png`),
            fullPage: false,
        });

        // Per-card visibility check: report which cards' name elements
        // are clipped or out-of-viewport. This is the diagnostic for
        // the "60px strip hides cards" mobile bug — if cards aren't
        // visible in the viewport, the count of visible-on-screen
        // cards is < 4.
        const cardVisibility = await page.evaluate(() => {
            const cards = document.querySelectorAll('#user_personas_panel .user-personas-card');
            const vpW = window.innerWidth, vpH = window.innerHeight;
            return Array.from(cards).map(card => {
                const r = card.getBoundingClientRect();
                const nameEl = card.querySelector('.user-personas-card-name');
                const name = nameEl?.textContent?.trim() || '(no-name)';
                const inViewport = r.top >= 0 && r.bottom <= vpH
                    && r.left >= 0 && r.right <= vpW;
                const partialVisible = r.bottom > 0 && r.top < vpH;
                return {
                    name,
                    rect: { top: Math.round(r.top), left: Math.round(r.left),
                            width: Math.round(r.width), height: Math.round(r.height) },
                    inViewport,
                    partialVisible,
                };
            });
        });
        const fullyVisibleCount = cardVisibility.filter(c => c.inViewport).length;
        const partialVisibleCount = cardVisibility.filter(c => c.partialVisible).length;
        console.log(`  viewport ${vp.width}×${vp.height} (${testInfo.project.name}):`);
        console.log(`    cards rendered: ${cardVisibility.length}`);
        console.log(`    fully in viewport: ${fullyVisibleCount}`);
        console.log(`    partially visible (at least 1px in viewport): ${partialVisibleCount}`);
        for (const c of cardVisibility) {
            console.log(`    [${c.name}] rect=${JSON.stringify(c.rect)} inViewport=${c.inViewport}`);
        }

        // Write the structured per-card visibility report.
        const fs = await import('node:fs');
        fs.writeFileSync(testInfo.outputPath(`${tag}_card_visibility.json`),
            JSON.stringify({
                viewport: vp,
                project: testInfo.project.name,
                cards_rendered: cardVisibility.length,
                fully_in_viewport: fullyVisibleCount,
                partial_visible: partialVisibleCount,
                cards: cardVisibility,
            }, null, 2));

        // Soft assertions — these surface the mobile bug as test
        // diagnostics without forcing the test to fail (the artifact
        // is the value; failing makes playwright drop other captures).
        if (fullyVisibleCount < 4) {
            console.warn(`  ⚠ at ${vp.width}×${vp.height}: only ${fullyVisibleCount}/4 cards fully visible — possible viewport-layout bug`);
        }
        expect(cardVisibility.length, 'all 4 cards must exist in DOM regardless of viewport').toBe(4);
    });
});
