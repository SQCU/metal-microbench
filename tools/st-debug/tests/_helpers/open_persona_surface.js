/**
 * Open a user-personas surface via the canonical hamburger popover flow.
 *
 * USE THIS instead of `page.locator('#user-X-button').click()` — the per-tab
 * wrappers have zero visible area when closed (they exist in DOM but their
 * .drawer-toggle is display:none). Clicking them directly bypasses the
 * navigability invariant ("single interface button for all user-agent /
 * persona design features") and would let the visible-top-bar-buttons
 * regression sneak back in.
 *
 * @param {import('@playwright/test').Page} page
 * @param {'suggester'|'corpus'|'fixed-point'|'axes'|'designer'} surfaceKey
 */
export async function openPersonaSurface(page, surfaceKey) {
    // 1. Open the hamburger popover.
    await page.locator('#user-personas-tools-button').click();
    // 2. Wait for the popover to render (it picks up class .openDrawer when open).
    await page.locator('#UserPersonasToolsMenu.openDrawer').waitFor({ state: 'visible', timeout: 10_000 });
    // 3. Click the menu item for the surface (data-surface-key attribute set in
    //    index.js around line 1033).
    await page.locator(`.user-personas-tools-menuitem[data-surface-key="${surfaceKey}"]`).click();
    // 4. Wait for the surface wrapper's drawer-content to enter openDrawer state.
    //    The wrapper IDs follow the pattern `user-${key}-button` for all keys
    //    EXCEPT 'designer' which (per the tabs table) does NOT have a buttonId
    //    and uses the fallback ID `user-personas-surface-designer`. Build the
    //    wrapper selector accordingly.
    const wrapperId = ['suggester', 'corpus', 'fixed-point', 'axes'].includes(surfaceKey)
        ? `user-${surfaceKey}-button`
        : `user-personas-surface-${surfaceKey}`;
    await page.locator(`#${wrapperId} > .drawer-content.openDrawer`).waitFor({ state: 'attached', timeout: 10_000 });
    return page.locator(`#${wrapperId}`);
}
