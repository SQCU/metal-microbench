// Minimal honest harness — uses ONLY the actual ST UI surface,
// no ctx.* internals, no programmatic /newchat, no
// selectCharacterById, no chat.push. The full surface used:
//
//   page.goto('/')                                  # load
//   page.locator('#API-status-top').click()         # open API drawer
//   page.locator('#api_button_openai').click()      # connect
//   page.locator('#send_textarea').fill(...)        # type
//   page.locator('#send_but').click()               # send
//
// Optional UI-driven helpers further below for character switch
// and fresh chat (via the actual buttons in the ST sidebar/UI).
//
// Returns an elicitation record by polling chat[]+DOM through
// SillyTavern.getContext for OBSERVATION ONLY (this is what the
// FE itself exposes to extensions; it's not poking internals to
// MUTATE state).

import { expect } from '@playwright/test';

export async function loadAndConnect(page) {
    await page.goto('/');
    await page.waitForFunction(
        'document.getElementById("preloader") === null',
        { timeout: 60_000 });
    // Wait for the SillyTavern global to exist (used for read-only
    // chat[] / characterId observation, not for state mutation).
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
        { timeout: 30_000 });

    // Click the API status indicator to open the connection drawer.
    await page.locator('#API-status-top').click();
    await expect(page.locator('#api_button_openai')).toBeVisible();
    await page.locator('#api_button_openai').click();
    // Wait for the API connection to actually become Valid — the textarea
    // placeholder transitions earlier than the connection completes, so
    // waiting on placeholder alone races. Verified 2026-05-17 by
    // browser_evaluate: online_status reads "no_connection" right when
    // the placeholder turns to "Type a message…", then "Valid" ~2s
    // later after the bridge handshake. Tests that send a message
    // before this point hit Generate's silent !hasBackendConnection
    // bail and the chat sits forever waiting for a reply that never
    // gets requested.
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return ctx?.onlineStatus === 'Valid';
    }, { timeout: 30_000 });

    // Toggle the drawer closed so it doesn't cover the chat in any
    // recorded video. (Click the same status indicator again.)
    await page.locator('#API-status-top').click();
    await page.waitForTimeout(300);
    await page.keyboard.press('Escape').catch(() => {});

    // Verify the chat container is hit-testable (no overlay).
    await page.waitForFunction(() => {
        const chat = document.getElementById('chat');
        if (!chat) return false;
        const r = chat.getBoundingClientRect();
        const x = r.left + r.width / 2;
        const y = r.top + r.height / 2;
        const el = document.elementFromPoint(x, y);
        return el === chat || chat.contains(el);
    }, { timeout: 10_000 });
}

/**
 * Send a user message via the actual textarea + send button.
 * Wait for the model's response (text + any tool calls) to settle.
 * Returns a structured elicitation record describing what happened.
 *
 * The wait condition only OBSERVES chat[]/DOM state — it doesn't
 * mutate. The chat[] array is what every ST extension reads.
 */
export async function sendAndObserve(page, userPrompt, opts = {}) {
    const timeoutMs = opts.timeoutMs ?? 120_000;

    const baselineLen = await page.evaluate(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return ctx?.chat?.length ?? 0;
    });

    const sendT0 = await page.evaluate(() => performance.now());
    await page.locator('#send_textarea').fill(userPrompt);
    await page.locator('#send_but').click();

    const settle = await page.waitForFunction((floor) => {
        const ctx = window.SillyTavern?.getContext?.();
        if (!ctx) return false;
        const chat = ctx.chat || [];
        if (chat.length <= floor) return false;
        const last = chat[chat.length - 1];
        if (!last || last.is_user) return false;
        // Skip ST's system placeholder turns ("no previous messages",
        // "[chat closed]", etc.) — they're scaffolding, not the model
        // response we're measuring.
        if (last.is_system) return false;
        // Generation must have finished
        const stop = document.querySelector('#mes_stop');
        if (stop && stop.offsetParent !== null) return false;
        // tool_progress entries terminal
        const tp = (last.extra?.tool_progress || []);
        for (const entry of tp) {
            if (!['done', 'failed', 'cancelled'].includes(entry.status)) return false;
        }
        return {
            mes: typeof last.mes === 'string' ? last.mes : '',
            tool_progress: tp.map(e => ({
                label: e.label,
                status: e.status,
                duration_ms: e.duration_ms,
                summary: e.summary || null,
            })),
            tool_invocations: (last.extra?.tool_invocations || []).map(i => ({
                name: i.displayName || i.name,
                parameters_raw: i.parameters,
            })),
        };
    }, baselineLen, { timeout: timeoutMs });

    const result = await settle.jsonValue();
    const elapsedMs = (await page.evaluate(() => performance.now())) - sendT0;

    let finishState;
    if (result.tool_invocations.length > 0 || result.tool_progress.length > 0) {
        finishState = 'tool_handled';
    } else if (result.mes && result.mes.trim().length > 0) {
        finishState = 'completed';
    } else {
        finishState = 'no_response';
    }
    return {
        userPrompt,
        finishState,
        elapsedMs,
        assistantText: result.mes,
        toolInvocations: result.tool_invocations,
        toolProgress: result.tool_progress,
    };
}

// ─────────────────────────────────────────────────────────────────
// UI-driven character switch + fresh chat. These click actual
// buttons in ST's interface rather than calling internal APIs.
// ─────────────────────────────────────────────────────────────────

/**
 * Click the character with the given name in ST's sidebar.
 * The sidebar has a row for each loaded character; the row's avatar
 * is clickable and triggers ST's character-selection flow naturally.
 */
export async function selectCharacterByClick(page, nameSubstr) {
    // Open the character list panel (right sidebar in ST).
    await page.locator('#rightNavDrawerIcon').click();
    await page.waitForTimeout(300);
    // Find the row whose avatar's title contains the name (case-insensitive).
    // All callers pass literal strings (e.g. 'Scringlo'); replacing the
    // RegExp with case-folded includes() is exactly equivalent for
    // literal needles and removes any chance of regex-metachar surprise.
    const needle = nameSubstr.toLowerCase();
    const rows = page.locator('#rm_print_characters_block .character_select');
    const count = await rows.count();
    for (let i = 0; i < count; i++) {
        const row = rows.nth(i);
        const txt = (await row.innerText()).trim();
        if (txt.toLowerCase().includes(needle)) {
            // Scroll into view before clicking — the character list may be
            // partially off-screen (e.g., list is long and dicemother is below
            // the viewport fold). Without this, Playwright retries the click
            // for the full actionTimeout (120s default), consuming the test budget.
            await row.scrollIntoViewIfNeeded().catch(() => {});
            await row.click({ timeout: 10_000 });
            // Close the drawer so it doesn't cover the chat in video
            await page.locator('#rightNavDrawerIcon').click().catch(() => {});
            await page.waitForFunction((name) => {
                const ctx = window.SillyTavern?.getContext?.();
                const c = ctx?.characters?.[ctx.characterId];
                return !!c && (c.name || '').toLowerCase().includes(name.toLowerCase());
            }, nameSubstr, { timeout: 15_000 });
            return;
        }
    }
    throw new Error(`character with name containing '${nameSubstr}' (case-insensitive) not found in sidebar (${count} rows)`);
}

/**
 * Trigger a fresh chat via the actual UI button. ST's "Start new chat"
 * lives in the options menu, opened by #options_button (the hamburger
 * to the left of the textarea).
 *
 * Polls until chat[] has dropped to <= 1 entry (just the first_mes for
 * the active character) or until timeout.
 */
export async function freshChatByClick(page) {
    // Capture chatId before so we can wait for it to change.
    const beforeId = await page.evaluate(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return ctx?.chatId ?? null;
    });
    await page.locator('#options_button').click();
    await page.waitForTimeout(200);
    const startNew = page.locator('#option_start_new_chat');
    await expect(startNew).toBeVisible({ timeout: 5000 });
    await startNew.click();
    // ST shows a "Start new chat?" Popup-class confirmation. The OK
    // button has class .popup-button-ok on the open dialog.
    const okBtn = page.locator('dialog[open] .popup-button-ok').first();
    await expect(okBtn).toBeVisible({ timeout: 5_000 });
    await okBtn.click();
    // Wait for the chatId to change (new chat loaded) AND visible
    // chat to drop to exactly 1 .mes (the first_mes only).
    await page.waitForFunction((before) => {
        const ctx = window.SillyTavern?.getContext?.();
        if (!ctx) return false;
        const id = ctx.chatId ?? null;
        if (before != null && id === before) return false;
        const visible = document.querySelectorAll('#chat .mes').length;
        return visible === 1;
    }, beforeId, { timeout: 30_000 });
    // Make sure options menu + any open dialog is closed. Press Escape
    // and verify the textarea is hit-testable so subsequent fill+click
    // lands on the actual user-input surface (not on a stale overlay).
    await page.keyboard.press('Escape').catch(() => {});
    await page.waitForFunction(() => {
        const opts = document.getElementById('options');
        if (opts && opts.offsetParent !== null) return false;
        const dlg = document.querySelector('dialog[open]');
        if (dlg) return false;
        const ta = document.getElementById('send_textarea');
        if (!ta) return false;
        const r = ta.getBoundingClientRect();
        const x = r.left + r.width / 2;
        const y = r.top + r.height / 2;
        const el = document.elementFromPoint(x, y);
        return el === ta || (ta.contains(el) || el?.contains?.(ta));
    }, { timeout: 5_000 });
}
