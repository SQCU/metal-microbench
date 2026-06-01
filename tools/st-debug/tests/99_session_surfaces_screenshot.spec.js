// Pixel-evidence capture of every client surface built this session.
//
// The other specs (40-47) assert STRUCTURAL facts via DOM queries —
// strong evidence that the features work, but they don't produce
// visual artifacts. This spec opens each surface and saves a PNG to
// tests/screenshots/, providing the actual pixel proof a human can
// look at and verify the feature renders as designed.
//
// NOT a regression test. Doesn't fail on visual differences. Just
// captures the current state of each surface against the live
// st-debug instance and writes the images to disk.

import { test, expect } from '@playwright/test';
import { loadAndConnect } from './_helpers/elicit_clean.mjs';
import { openPersonaSurface } from './_helpers/open_persona_surface.js';
import * as fs from 'fs';
import * as path from 'path';

const OUT_DIR = '/Users/mdot/metal-microbench/tools/st-debug/tests/screenshots';

test.describe('session surface screenshots', () => {
    test.setTimeout(3 * 60 * 1000);

    test.beforeAll(() => {
        fs.mkdirSync(OUT_DIR, { recursive: true });
    });

    test('capture each surface to PNG', async ({ page }) => {
        await loadAndConnect(page);

        // Allow the extension to install the hamburger button.
        // (#user-fixed-point-button's .drawer-toggle is display:none after
        // sillytavern-fork e2973179d — wait for the hamburger instead.)
        await page.locator('#user-personas-tools-button').waitFor({ state: 'attached', timeout: 15_000 });

        // ── (1) Top bar — hamburger visible; per-tab toggles are display:none.
        await page.screenshot({
            path: path.join(OUT_DIR, '01_toprow_drawer_buttons.png'),
            clip: { x: 0, y: 0, width: 1280, height: 60 },
        });

        // ── (2) Suggester drawer — context-driven ranker, _meta strip,
        //        provenance filter row, ranked list area, +More + Synthesize CTA.
        // Open via hamburger popover — .drawer-toggle is display:none after
        // sillytavern-fork e2973179d; direct .drawer-toggle click is invalid.
        await openPersonaSurface(page, 'suggester');
        const suggIframe = page.frameLocator('iframe[src*="suggester.html"]');
        await expect(suggIframe.locator('h1').first()).toBeVisible({ timeout: 15_000 });
        await page.waitForTimeout(500);
        await page.screenshot({
            path: path.join(OUT_DIR, '02_suggester_drawer_open.png'),
            fullPage: false,
        });
        // No explicit close needed — openPersonaSurface for the next surface
        // calls closeAllOpen() automatically before opening.
        await page.waitForTimeout(300);

        // ── (3) Fixed-point tab — Experiments list (default landing).
        await openPersonaSurface(page, 'fixed-point');
        const fpIframe = page.frameLocator('iframe[src*="fixed_point.html"]');
        await expect(fpIframe.locator('h1, h2').first()).toBeVisible({ timeout: 15_000 });
        await page.waitForTimeout(500);
        await page.screenshot({
            path: path.join(OUT_DIR, '03_fp_experiments_list.png'),
            fullPage: false,
        });

        // ── (4) Fixed-point tab — Seed input mode.
        const seedTab = fpIframe.locator('button:has-text("Seed input")').first();
        if (await seedTab.count() > 0) {
            await seedTab.click();
            await page.waitForTimeout(500);
            // Fill in the lock_in_tetrad verbatim seeds so the screenshot
            // shows the surface in its filled-in state, not blank.
            const textarea = fpIframe.locator('textarea').first();
            await textarea.fill(
                'bios:\n  rpg wizard but he a sagittarius\n  rpg rogue but he a cancer\n\nmotives:\n  will 100% steal all of your stuff\n  will try to kiss you but also will 100% steal all of your stuff'
            );
            await page.waitForTimeout(300);
            await page.screenshot({
                path: path.join(OUT_DIR, '04_fp_seed_input_filled.png'),
                fullPage: false,
            });
            // Try to click Parse so the chips render too.
            const parseBtn = fpIframe.locator('button:has-text("Parse")').first();
            if (await parseBtn.count() > 0) {
                await parseBtn.click();
                await page.waitForTimeout(500);
                await page.screenshot({
                    path: path.join(OUT_DIR, '05_fp_seed_parsed_chips.png'),
                    fullPage: false,
                });
            }
        }

        // ── (5) Fixed-point tab — Trajectory view (click into lock_in_tetrad).
        // Switch back to Experiments tab first.
        const expTab = fpIframe.locator('button:has-text("Experiments")').first();
        if (await expTab.count() > 0) {
            await expTab.click();
            await page.waitForTimeout(300);
        }
        const tetradRow = fpIframe.locator('.experiment-card, .experiment-row')
            .filter({ hasText: /lock_in_tetrad/ }).first();
        if (await tetradRow.count() > 0) {
            await tetradRow.click();
            await page.waitForTimeout(800);
            await page.screenshot({
                path: path.join(OUT_DIR, '06_fp_trajectory_lockin_tetrad.png'),
                fullPage: false,
            });
        }
        // No explicit close — openPersonaSurface for corpus auto-closes FP.
        await page.waitForTimeout(300);

        // ── (6) Corpus dashboard — eff-dim PR + per-axis bar chart.
        // Open via hamburger popover (corpus menu item always present).
        await openPersonaSurface(page, 'corpus');
        const corpIframe = page.frameLocator('iframe[src*="corpus.html"]');
        await expect(corpIframe.locator('h1, h2').first()).toBeVisible({ timeout: 15_000 });
        await page.waitForTimeout(1000);
        await page.screenshot({
            path: path.join(OUT_DIR, '07_corpus_dashboard.png'),
            fullPage: false,
        });
        // ── (7) Axis registry — now the bottom section of corpus.html
        // (the standalone Axes tab was retired in the 2026-06
        // consolidation), captured within the corpus screenshot above.
        await page.waitForTimeout(300);

        // ── (8) Manifest of captured artifacts for the operator to inspect.
        const captured = fs.readdirSync(OUT_DIR).filter(f => f.endsWith('.png')).sort();
        console.log(`[surfaces] captured ${captured.length} screenshots:`);
        for (const c of captured) {
            const stat = fs.statSync(path.join(OUT_DIR, c));
            console.log(`  ${c} ${stat.size} bytes`);
        }
        // No hard fail; this is evidence-capture, not regression.
        expect(captured.length).toBeGreaterThan(0);
    });
});
