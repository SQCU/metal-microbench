import { test, expect } from '@playwright/test';

test.describe('Lineage badges (R-4)', () => {
    test('suggester.html has .lineage-badge styles and helper', async ({ page }) => {
        // Load the suggester HTML directly (no ST required, no bridge required).
        await page.goto('http://localhost:8002/api/plugins/user-personas/static/suggester.html');

        // Check that the CSS classes exist in the <style> block.
        const styleContent = await page.locator('style').first().textContent();
        expect(styleContent).toContain('.lineage-badge');
        expect(styleContent).toContain('.lineage-badge.root');
        expect(styleContent).toContain('.lineage-badge.derived');
        expect(styleContent).toContain('background: #1f2a30');  // root color
        expect(styleContent).toContain('color: #87af87');       // root text color
        expect(styleContent).toContain('background: #2a1f30');  // derived color
        expect(styleContent).toContain('color: #c8a8d8');       // derived text color
    });

    test('renderLineageBadge helper exists and works', async ({ page }) => {
        // Load the suggester directly.
        await page.goto('http://localhost:8002/api/plugins/user-personas/static/suggester.html');

        // Test the renderLineageBadge function via page evaluation.
        const result = await page.evaluate(() => {
            // Mock row with no derived_from (root case).
            const rootRow = { persona: { derived_from: null }, agent: { derived_from: null } };
            const rootBadge = window.renderLineageBadge(rootRow);

            // Mock row with bio-level derived_from.
            const derivedRow = {
                persona: { derived_from: 'parent_bio_id' },
                agent: { derived_from: null }
            };
            const derivedBadge = window.renderLineageBadge(derivedRow);

            return {
                rootBadge,
                derivedBadge,
                rootMatches: rootBadge.includes('root'),
                derivedMatches: derivedBadge.includes('parent_bio_id'),
            };
        });

        expect(result.rootMatches).toBe(true);
        expect(result.derivedMatches).toBe(true);
        expect(result.rootBadge).toContain('lineage-badge root');
        expect(result.derivedBadge).toContain('lineage-badge derived');
    });

    test('renderRankedRow includes lineage badge in template', async ({ page }) => {
        // Load the suggester directly.
        await page.goto('http://localhost:8002/api/plugins/user-personas/static/suggester.html');

        // Test renderRankedRow generates a row with a lineage badge.
        const result = await page.evaluate(() => {
            const mockRow = {
                bio_id: 'test-bio',
                agent_id: 'test-agent',
                persona: { name: 'Test Bio', derived_from: null },
                agent: { name: 'Test Agent', derived_from: null },
                distance: 1.5,
                why: 'nearest tuple',
            };
            const html = window.renderRankedRow(mockRow, 'top');
            return {
                html,
                hasBadge: html.includes('lineage-badge'),
                hasRoot: html.includes('lineage-badge root'),
            };
        });

        expect(result.hasBadge).toBe(true);
        expect(result.hasRoot).toBe(true);
        expect(result.html).toContain('ranked-head');
    });

    test('API /personas endpoint includes derived_from field', async ({ page }) => {
        // Fetch the /personas endpoint to verify derived_from is exposed.
        const response = await page.goto('http://localhost:8002/api/plugins/user-personas/personas');
        const json = await response.json();

        expect(json).toHaveProperty('personas');
        const personas = json.personas;
        expect(Array.isArray(personas)).toBe(true);

        // Check that at least one persona has the derived_from field (even if null).
        if (personas.length > 0) {
            const firstPersona = personas[0];
            expect(firstPersona).toHaveProperty('derived_from');
            // derived_from can be null or a string/object — just verify the key exists.
        }
    });
});
