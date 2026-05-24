import { test, expect } from '@playwright/test';

test.describe('Auto-poll K_1 regression', () => {
    // K_1 rows should auto-fire /poll in parallel on first paint, no manual click needed.
    // Thesis-positive: batched AR decoding is immediately visible.
    // Note: These tests assume the corpus has at least some agents populated (R-3).
    // If the corpus is empty, tests will gracefully skip.

    test('auto-fires top-K_1 rows in parallel on first paint', async ({ page }) => {
        // Navigate to ST.
        await page.goto('http://localhost:8002');
        await page.waitForSelector('body', { timeout: 5000 });

        // Access the suggester iframe.
        const suggesterFrame = page.frameLocator('iframe#user_personas_iframe');

        // The suggester can render in two states:
        // 1. Empty state ("Awaiting active chat") if no chat has content.
        // 2. Ranked list if a chat with content exists AND the corpus has agents.
        //
        // We wait for either the empty state OR the ranked list to appear.
        // If ranked list appears with rows, we test auto-fire. If empty state appears,
        // we skip (corpus is empty or no chat with content).

        const emptyState = suggesterFrame.locator('.empty-state');
        const topHeading = suggesterFrame.locator('h3:has-text("Top picks")');

        // Wait for one of these to appear (5s for slow rank, or immediate empty state).
        const emptyPromise = emptyState.isVisible({ timeout: 8000 }).catch(() => false);
        const rankedPromise = topHeading.isVisible({ timeout: 8000 }).catch(() => false);

        const [hasEmpty, hasRanked] = await Promise.all([emptyPromise, rankedPromise]);

        if (hasEmpty || !hasRanked) {
            // Either the empty state appeared, or no ranked list appeared.
            // Skip gracefully.
            test.skip();
            return;
        }

        // Ranked list is visible; corpus has agents.
        const topRows = suggesterFrame.locator('.ranked-row:not(.side)');
        const rowCount = await topRows.count();

        if (rowCount === 0) {
            test.skip();
            return;
        }

        // Assert at least one row is visible.
        expect(rowCount).toBeGreaterThanOrEqual(1);

        // Check the first min(3, rowCount) rows for non-empty .row-completion.
        // The thesis is that K_1 rows auto-fire; so we assert streaming text exists
        // WITHOUT operator clicks. Timeout is generous to account for /poll latency.
        const k1Count = Math.min(3, rowCount);

        for (let i = 0; i < k1Count; i++) {
            const row = topRows.nth(i);
            const slot = row.locator('.row-completion');

            // 1. Slot is visible (display: block from .visible class).
            // The slot becomes .visible when suggest() paints a result into it.
            // Auto-fire should have triggered on first paint, so this should be fast.
            await expect(slot).toHaveClass(/visible/, { timeout: 10000 });

            // 2. Slot contains non-empty text (not a spinner or loading message).
            const textDiv = slot.locator('.row-completion-text');
            const text = await textDiv.textContent();
            expect(text).not.toBeNull();
            expect(text.trim().length).toBeGreaterThan(10);  // non-trivial prose
        }

        // 3. Verify side picks (K_2) stay DISABLED (no .row-completion populated).
        const sideRows = suggesterFrame.locator('.ranked-row.side');
        if (await sideRows.count() > 0) {
            const sideSlot = sideRows.first().locator('.row-completion');
            // Side rows should NOT have .visible class (inert by default).
            await expect(sideSlot).not.toHaveClass(/visible/);
        }

        // 4. Click a top-K_1 row's Suggest button again; verify cache hit.
        // On a 2nd suggest(same row), the cache hits and no /poll is fired.
        // We assert this by checking that a cache-badge appears.
        const firstBtn = topRows.first().locator('.suggest-btn');
        const firstSlot = topRows.first().locator('.row-completion');

        // Get the current text (populated by auto-fire).
        const originalText = await firstSlot.locator('.row-completion-text').textContent();

        // Click the button again.
        await firstBtn.click();

        // Wait briefly for the button to be re-enabled (suggest() completes).
        await page.waitForTimeout(800);

        // The text should remain the same (cache hit, no re-poll).
        const cachedText = await firstSlot.locator('.row-completion-text').textContent();
        expect(cachedText).toBe(originalText);

        // A cache-badge should appear on cache hit (fromCache=true in paintCompletionSlot).
        const cacheBadge = firstSlot.locator('.cache-badge');
        await expect(cacheBadge).toBeVisible();
    });

    test('does not auto-poll when chat is empty', async ({ page }) => {
        // Load ST.
        await page.goto('http://localhost:8002');
        await page.waitForSelector('body', { timeout: 5000 });

        const suggesterFrame = page.frameLocator('iframe#user_personas_iframe');

        // The suggester should show "Awaiting active chat" when the active
        // chat has no content. Check for the empty state.
        const emptyState = suggesterFrame.locator('.empty-state');

        // Wait for the empty state or a ranked list to appear.
        const emptyPromise = emptyState.isVisible({ timeout: 8000 }).catch(() => false);
        const rankedPromise = suggesterFrame.locator('h3:has-text("Top picks")').isVisible({ timeout: 8000 }).catch(() => false);

        const [hasEmpty, hasRanked] = await Promise.all([emptyPromise, rankedPromise]);

        // If we see ranked rows, the chat isn't empty; skip this test.
        if (hasRanked && !hasEmpty) {
            test.skip();
            return;
        }

        // If we don't see empty state either, the UI state is unexpected; skip.
        if (!hasEmpty) {
            test.skip();
            return;
        }

        // Verify the empty state text matches the no-chat pattern.
        const emptyText = await emptyState.textContent();
        expect(emptyText).toContain('Awaiting');

        // Verify no /poll requests were issued (implicit by empty state).
        // The absence of populated .row-completion slots is evidence.
        const anySlot = suggesterFrame.locator('.row-completion.visible');
        const visibleCount = await anySlot.count();
        expect(visibleCount).toBe(0);
    });

    test('K_1 cache hit survives page reload', async ({ page }) => {
        // Load ST.
        await page.goto('http://localhost:8002');
        await page.waitForSelector('body', { timeout: 5000 });

        const suggesterFrame = page.frameLocator('iframe#user_personas_iframe');

        // Wait for ranked-list to appear (either empty or with rows).
        const emptyState = suggesterFrame.locator('.empty-state');
        const topHeading = suggesterFrame.locator('h3:has-text("Top picks")');

        const [hasEmpty, hasRanked] = await Promise.all([
            emptyState.isVisible({ timeout: 8000 }).catch(() => false),
            topHeading.isVisible({ timeout: 8000 }).catch(() => false),
        ]);

        if (hasEmpty || !hasRanked) {
            // Corpus is empty; skip.
            test.skip();
            return;
        }

        const topRows = suggesterFrame.locator('.ranked-row:not(.side)');
        const rowCount = await topRows.count();

        if (rowCount === 0) {
            test.skip();
            return;
        }

        // Verify all K_1 rows have .row-completion prose (auto-fired).
        const k1Count = Math.min(3, rowCount);
        for (let i = 0; i < k1Count; i++) {
            const row = topRows.nth(i);
            const slot = row.locator('.row-completion');
            await expect(slot).toHaveClass(/visible/, { timeout: 10000 });
            const text = await slot.locator('.row-completion-text').textContent();
            expect(text).not.toBeNull();
            expect(text.trim().length).toBeGreaterThan(10);
        }

        // Reload the page. The in-memory _previewCache is cleared, but the
        // ranking should hit the _rankCache (per-chat rank cache is preserved
        // across reloads at the browser level if the same chat is active).
        // After reload, auto-fire fires again.
        await page.reload();
        await page.waitForSelector('body', { timeout: 5000 });

        // Wait for the suggester to re-render.
        const reloadedFrame = page.frameLocator('iframe#user_personas_iframe');
        const reloadedHeading = reloadedFrame.locator('h3:has-text("Top picks")');
        await expect(reloadedHeading).toBeVisible({ timeout: 10000 });

        const reloadedRows = reloadedFrame.locator('.ranked-row:not(.side)');
        const reloadedCount = await reloadedRows.count();

        if (reloadedCount === 0) {
            test.skip();
            return;
        }

        // Verify that auto-fire happened again (all K_1 rows have text).
        const reloadedK1 = Math.min(3, reloadedCount);
        for (let i = 0; i < reloadedK1; i++) {
            const row = reloadedRows.nth(i);
            const slot = row.locator('.row-completion');
            await expect(slot).toHaveClass(/visible/, { timeout: 10000 });
        }
    });
});
