// ============================================================================
// SERVER-FREE RUNTIME-CLIENT HARNESS -- the in-chat PANEL surface (index.js).
// ============================================================================
//
// Sibling of suggester_runtime_client.spec.js, for the OTHER user-agent
// surface: the persistent in-chat panel implemented by the real FE extension
// `public/scripts/extensions/user-personas/index.js`. This is the surface most
// directly tied to the operator's phrasing "multiple-users-per-assistant-turn":
// top-K user agents in 'suggest' mode each polled for a user-sided turn,
// immediately, on tab-in -- and RE-polled after each assistant turn.
//
// index.js is a real ES module:
//     import { eventSource, event_types, getRequestHeaders, is_send_press,
//              sendTextareaMessage, Generate, addOneMessage, chat as stChatArray,
//              chat_metadata, saveChatConditional } from '../../../script.js';
//
// To run the REAL module server-free we stub ONLY THE HOST: a fake `script.js`
// that exports those exact symbols, plus `window.SillyTavern`. We do NOT touch
// index.js -- its real bytes are served verbatim and executed in real Chromium.
// Every backend endpoint is route-synthesized (no server).
//
// HOW TO RUN:
//   cd tools/st-debug/tests
//   npx playwright test --config runtime-client/playwright.runtime.config.js panel_runtime_client.spec.js
// ============================================================================

import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

const EXT_DIR = process.env.PANEL_EXT_DIR
    || '/Users/mdot/sillytavern-fork/public/scripts/extensions/user-personas';
const INDEX_JS = path.join(EXT_DIR, 'index.js');

const ORIGIN = 'http://localhost:7332';
// index.js imports `../../../script.js`; served at the path below that resolves
// to /public/script.js.
const INDEX_URL_PATH = '/public/scripts/extensions/user-personas/index.js';
const DOC_URL = `${ORIGIN}/harness.html`;

const POLL_MARKER = 'PANEL_RUNTIME_POLL_PROSE';

const DOC_HTML = `<!doctype html><html><head><meta charset="utf-8"><title>panel harness</title></head>
<body>
  <div id="top-settings-holder"></div>
  <div id="persona-management-button"></div>
  <div id="chat"></div>
  <form id="form_sheld">
    <div id="leftSendForm"></div>
    <textarea id="send_textarea"></textarea>
    <div id="send_but"></div>
  </form>
  <script type="module" src="${INDEX_URL_PATH}"></script>
</body></html>`;

// Minimal host module standing in for ST's public/script.js. Exports EXACTLY
// the symbols index.js imports -- nothing reimplemented from the feature.
// eventSource/event_types are real (the feature drives them); the mutation
// helpers are inert (autonomous-mode mutation isn't exercised here).
const STUB_SCRIPT_JS = `
const _handlers = {};
export const eventSource = {
  on: (ev, cb) => { (_handlers[ev] = _handlers[ev] || []).push(cb); },
  emit: (ev, ...a) => { (_handlers[ev] || []).forEach(cb => cb(...a)); },
};
export const event_types = {
  CHAT_CHANGED: 'chat_changed',
  MESSAGE_RECEIVED: 'message_received',
  MESSAGE_SENT: 'message_sent',
  MESSAGE_SWIPED: 'message_swiped',
  PERSONAS_UPDATED: 'personas_updated',
  CHAT_LOADED: 'chat_loaded',
};
export function getRequestHeaders() { return { 'Content-Type': 'application/json' }; }
export const is_send_press = false;
export async function sendTextareaMessage() {}
export async function Generate() {}
export async function addOneMessage() {}
export const chat = (window.__HARNESS_CHAT__ || []);
export const chat_metadata = {};
export async function saveChatConditional() {}
`;

function rankRow(id, i) {
    return {
        bio_id: `${id}.png`,
        agent_id: `${id}-agent`,
        distance: 0.2 + i * 0.1,
        why: `panel fixture ${id}`,
        persona: { id: `${id}.png`, name: `Persona ${id}`, bio: `${id} bio`, provenance: { kind: 'runtime-client' } },
        agent: { id: `${id}-agent`, name: `Agent ${id}`, designed_for_bio_id: `${id}.png`, provenance: { kind: 'runtime-client' } },
    };
}
function yapperBody() {
    return {
        top: [rankRow('alpha', 0), rankRow('beta', 1)],
        side: [rankRow('gamma', 2)],
        _meta: { K_top: 2, K_side: 1, target_signature: { tone: 3 }, candidates_considered: 3, bios_total: 3, agents_total: 3, pending_synthesis: [], pending_count: 0 },
    };
}
function emptyYapperBody() {
    return { top: [], side: [], _meta: { K_top: 2, K_side: 1, candidates_considered: 0, bios_total: 3, agents_total: 0, pending_synthesis: [{ bio_id: 'alpha.png', name: 'Persona alpha' }], pending_count: 1 } };
}

const POPULATED_CHAT = [
    { is_user: false, name: 'dicemother', mes: 'The inn yard is muddy and quiet. A sealed satchel rests on an overturned cask.' },
    { is_user: true, name: 'scringlo scrambler', mes: 'I set the satchel down carefully and check whether anyone followed me into the yard.' },
    { is_user: false, name: 'dicemother', mes: 'No one follows. Something inside the satchel shifts against the leather.' },
];

// The ONE Suggester core, imported by the real index.js as `./suggester_core.js`.
// Both surfaces share it; without serving the real bytes here the static import
// would resolve to the catch-all's empty 200 and index.js would throw at boot.
const SUGGESTER_CORE_JS = path.join(EXT_DIR, 'suggester_core.js');

async function installHarness(page, { chat, calls, yapper = yapperBody }) {
    if (!fs.existsSync(INDEX_JS)) throw new Error(`real index.js not found: ${INDEX_JS} (set PANEL_EXT_DIR)`);
    if (!fs.existsSync(SUGGESTER_CORE_JS)) throw new Error(`real suggester_core.js not found: ${SUGGESTER_CORE_JS} (set PANEL_EXT_DIR)`);
    const indexSrc = fs.readFileSync(INDEX_JS, 'utf8');
    const coreSrc = fs.readFileSync(SUGGESTER_CORE_JS, 'utf8');

    // Catch-all FIRST (lowest priority). Playwright matches the most-recently-
    // registered handler first, so specific routes below win. (A trailing
    // catch-all instead shadows the real index.js/script.js and breaks module
    // resolution -- learned the hard way.)
    await page.route('**/*', route => route.fulfill({ status: 200, contentType: 'text/plain', body: '' }));

    await page.route('**/harness.html', route => route.fulfill({ status: 200, contentType: 'text/html', body: DOC_HTML }));
    await page.route('**/extensions/user-personas/index.js', route =>
        route.fulfill({ status: 200, contentType: 'application/javascript', body: indexSrc }));
    // index.js imports `./suggester_core.js` (the ONE core). Serve the real
    // bytes with a JS MIME type so the module import resolves.
    await page.route('**/extensions/user-personas/suggester_core.js', route =>
        route.fulfill({ status: 200, contentType: 'application/javascript', body: coreSrc }));
    // The real index.js does `import ... from '../../../script.js'`, which from
    // /public/scripts/extensions/user-personas/index.js resolves to
    // /public/script.js. Serve the host stub there.
    await page.route('**/script.js', route =>
        route.fulfill({ status: 200, contentType: 'application/javascript', body: STUB_SCRIPT_JS }));

    await page.route('**/api/plugins/user-personas/yapper-seed', async route => {
        calls.push({ kind: 'yapper-seed', t: Date.now() - calls._t0 });
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(yapper()) });
    });
    await page.route('**/api/plugins/user-personas/poll', async route => {
        const b = route.request().postDataJSON() || {};
        calls.push({ kind: 'poll', t: Date.now() - calls._t0, bio: b.persona_id });
        await new Promise(res => setTimeout(res, 150));
        await route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({
                applied_overlay: { source: 'agent', agent_id: `${b.persona_id}-agent`, name: `Agent for ${b.persona_id}`, depth: 1, text_chars: POLL_MARKER.length },
                candidates: [{ text: `${POLL_MARKER}: drafted next user turn for ${b.persona_id}.`, truncated: false }],
            }),
        });
    });
    await page.route('**/api/plugins/user-personas/personas', route =>
        route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ personas: [
            { id: 'alpha.png', name: 'Persona alpha' }, { id: 'beta.png', name: 'Persona beta' }, { id: 'gamma.png', name: 'Persona gamma' },
        ] }) }));

    await page.addInitScript((chatArg) => {
        // Reset persisted persona modes so prior runs don't leak suggest-mode
        // state into a fresh boot (index.js stores modes in localStorage).
        try { localStorage.removeItem('user_personas_modes'); } catch (_) {}
        window.__HARNESS_CHAT__ = chatArg;
        const eventTypes = { CHAT_CHANGED: 'chat_changed', MESSAGE_RECEIVED: 'message_received', MESSAGE_SENT: 'message_sent' };
        const ctx = { characterId: 0, chatId: 'panel-chat', characters: [{ avatar: 'dicemother.png', name: 'dicemother' }], chat: chatArg };
        window.SillyTavern = { getContext: () => ctx };
    }, chat);
}

test.describe('user-agent PANEL (index.js) -- REAL runtime client, server-free', () => {
    test('boot/tab-in -> generate -> IMMEDIATE top-K suggest-mode polls -> prose in cards (zero clicks)', async ({ page }) => {
        const calls = [];
        const errors = [];
        page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
        page.on('pageerror', e => errors.push(`pageerror: ${e.message}`));

        await installHarness(page, { chat: POPULATED_CHAT, calls });
        calls._t0 = Date.now();
        await page.goto(DOC_URL, { waitUntil: 'domcontentloaded' });

        // The real module must boot and inject the panel without any action.
        await expect(page.locator('#user_personas_panel'), 'panel injected by real index.js boot()').toHaveCount(1, { timeout: 10_000 });

        // LINK 1: tab-in/boot -> /yapper-seed (generate), no click.
        await expect.poll(() => calls.filter(c => c.kind === 'yapper-seed').length,
            { timeout: 10_000, message: 'LINK-1 BROKEN: panel boot did not trigger /yapper-seed (generation)' }).toBeGreaterThan(0);

        // LINK 2: top-K suggest-mode polled IMMEDIATELY, no click.
        await expect.poll(() => calls.filter(c => c.kind === 'poll').length,
            { timeout: 10_000, message: 'LINK-2 BROKEN: top-K cards did not auto-fire /poll (immediate suggest-mode) without a click' }).toBeGreaterThanOrEqual(2);

        const polledBios = [...new Set(calls.filter(c => c.kind === 'poll').map(c => c.bio))].sort();
        expect(polledBios, 'LINK-2b: both top-K bios auto-polled in parallel').toEqual(['alpha.png', 'beta.png']);

        // LINK 3: prose lands in the card preview WITHOUT a click. The panel may
        // be collapsed (is-collapsed) on first paint, so we assert content
        // (textContent), not visibility.
        await expect(page.locator('#user_personas_cards_container'),
            'LINK-3 BROKEN: suggested user-turn prose never rendered into a card preview without a click')
            .toContainText(POLL_MARKER, { timeout: 10_000 });

        console.log('[panel-runtime-client] EMPIRICAL TIMELINE:', JSON.stringify(calls.map(c => ({ ...c }))));
        if (errors.length) console.log('[panel-runtime-client] PAGE ERRORS:', JSON.stringify(errors));
    });

    test('multiple-users-per-assistant-turn: a new assistant turn re-polls every top-K user agent (zero clicks)', async ({ page }) => {
        const calls = [];
        await installHarness(page, { chat: POPULATED_CHAT, calls });
        calls._t0 = Date.now();
        await page.goto(DOC_URL, { waitUntil: 'domcontentloaded' });

        // Wait for first-paint cascade to settle (top-K polled once).
        await expect.poll(() => [...new Set(calls.filter(c => c.kind === 'poll').map(c => c.bio))].length,
            { timeout: 10_000 }).toBeGreaterThanOrEqual(2);
        const pollsAfterFirstPaint = calls.filter(c => c.kind === 'poll').length;

        // Simulate the assistant yielding a new turn: append an assistant
        // message and emit MESSAGE_RECEIVED through the REAL eventSource the
        // module subscribed to. onAssistantMessageReceived must force-re-poll
        // every active (top-K) user agent -- the "multiple-users-per-assistant-
        // turn" behavior.
        await page.evaluate(async () => {
            const ctx = window.SillyTavern.getContext();
            ctx.chat.push({ is_user: false, name: 'dicemother', mes: 'A hooded night-jay blinks up at you, one leg banded silver.' });
            const mod = await import('/public/script.js');
            mod.eventSource.emit('message_received', ctx.chat.length - 1, 'normal');
        });

        await expect.poll(() => calls.filter(c => c.kind === 'poll').length,
            { timeout: 10_000, message: 'PER-TURN RE-POLL BROKEN: a new assistant turn did not re-poll the active top-K user agents' })
            .toBeGreaterThan(pollsAfterFirstPaint);

        console.log('[panel-runtime-client] PER-TURN TIMELINE:', JSON.stringify(calls.map(c => ({ ...c }))));
    });

    test('empty corpus -> awaiting-synthesis state, NO auto /poll (data-driven empty, not a poll-chain break)', async ({ page }) => {
        // This isolates the most common "looks broken / never fills" experience
        // from a genuine poll-chain regression. When /yapper-seed returns no
        // (bio x agent) picks (fresh install: /agents empty), the spec-correct
        // behavior is: render the awaiting-synthesis empty state and fire NO
        // /poll (there is nothing to poll). If this test "fails" by seeing a
        // poll, the empty-state guard regressed; if the panel goes blank with
        // no awaiting-synthesis copy, P-NO-EMPTY-FIRST-PAINT regressed.
        const calls = [];
        await installHarness(page, { chat: POPULATED_CHAT, calls, yapper: emptyYapperBody });
        calls._t0 = Date.now();
        await page.goto(DOC_URL, { waitUntil: 'domcontentloaded' });

        await expect.poll(() => calls.filter(c => c.kind === 'yapper-seed').length, { timeout: 10_000 }).toBeGreaterThan(0);
        await page.waitForTimeout(1500);

        const polls = calls.filter(c => c.kind === 'poll');
        expect(polls.length, `empty corpus must NOT auto-fire /poll (got ${polls.length})`).toBe(0);

        // P-NO-EMPTY-FIRST-PAINT: the panel must show a meaningful state, not
        // a blank container.
        await expect(page.locator('#user_personas_cards_container'),
            'empty corpus must render an awaiting-synthesis explanation, not a blank panel')
            .toContainText(/compositions to rank|awaiting synthesis|Synthesis runs automatically/i, { timeout: 10_000 });

        console.log('[panel-runtime-client] EMPTY-CORPUS TIMELINE:', JSON.stringify(calls.map(c => ({ ...c }))));
    });
});
