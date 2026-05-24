import { test, expect } from '@playwright/test';

test.describe('R-5: SPLIT-DEMO-PRESTAGED', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('http://127.0.0.1:8002/api/plugins/user-personas/static/corpus.html');
    });

    test('(a) Corpus tab opens and renders axes registry', async ({ page }) => {
        // Corpus page loads without error
        await expect(page.locator('h1')).toContainText('Corpus');
        await expect(page.locator('.orphans-section h3')).toContainText('Orphaned signatures');
    });

    test('(b) Registry shows >= 1 derived axis with derived_from.parent set', async ({ page }) => {
        // The axes registry renders the tree
        const axisCards = page.locator('.axis-card');
        const cardCount = await axisCards.count();
        expect(cardCount).toBeGreaterThanOrEqual(4); // 3 roots + at least 1 derived

        // At least one card has the .derived class (border-left: #aa6fd7)
        const derivedCards = page.locator('.axis-card.derived');
        const derivedCount = await derivedCards.count();
        expect(derivedCount).toBeGreaterThan(0);

        // The specific card for rpg_class_combat_intensity exists and is derived
        const combatCard = page.locator('.axis-card[data-axis-id="rpg_class_combat_intensity"]');
        await expect(combatCard).toBeVisible();
        await expect(combatCard).toHaveClass(/derived/);
    });

    test('(c) Derived axis shows lineage expandable with hypothesis text', async ({ page }) => {
        // Find the rpg_class_combat_intensity card
        const card = page.locator('.axis-card[data-axis-id="rpg_class_combat_intensity"]');
        await expect(card).toBeVisible();

        // Lineage button exists
        const lineageBtn = card.locator('button:has-text("show lineage")');
        await expect(lineageBtn).toBeVisible();

        // Click the button to expand
        await lineageBtn.click();

        // Hypothesis text appears (initially hidden, now visible)
        const lineageText = card.locator('.lineage-text');
        await expect(lineageText).toBeVisible();

        // Hypothesis content is non-empty and contains recognizable phrase
        const hypothesisDiv = card.locator('.lineage-text .hypothesis');
        await expect(hypothesisDiv).toBeVisible();

        // Contexts content is non-empty
        const contextsDiv = card.locator('.lineage-text .contexts');
        await expect(contextsDiv).toBeVisible();

        // Verify the content contains expected phrases from the hypothesis/context
        const lineageTextContent = await card.locator('.lineage-text').textContent();
        expect(lineageTextContent).toContain('entangle');
        expect(lineageTextContent).toContain('preparation');
        expect(lineageTextContent).toContain('improvisation');
    });

    test('(d) Orphaned sibling is listed under "Orphaned signatures"', async ({ page }) => {
        // Navigate to orphans section
        const orphansSection = page.locator('.orphans-section');
        await expect(orphansSection).toBeVisible();

        // rpg_class_resource_management is listed as orphaned
        const orphansHeading = page.locator('.orphans-section h3');
        await expect(orphansHeading).toContainText('Orphaned signatures');

        // Find the orphan entry
        const orphanList = page.locator('#orphans-list');
        const orphanRow = orphanList.locator('.orphan-row:has-text("rpg_class_resource_management")');
        await expect(orphanRow).toBeVisible();

        // Verify it's marked as orphaned (referenced but missing)
        const orphanMeta = orphanRow.locator('.orphan-meta');
        const metaText = await orphanMeta.textContent();
        expect(metaText).toContain('referenced by');
        expect(metaText).toContain('no registry entry');
    });
});
