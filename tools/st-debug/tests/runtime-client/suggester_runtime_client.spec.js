// ============================================================================
// SERVER-FREE RUNTIME-CLIENT VALIDATION HARNESS
// for the user-agent / suggester "tab-in -> generate -> poll -> suggest" chain.
// (iframe surface: plugins/user-personas/static/suggester.html)
// ============================================================================
//
// WHY THIS EXISTS
// ---------------
// The operator's spec for this feature:
//   "The instant a chat is tabbed into, a back-end dynamically generates a
//    'user agent' for that chat. A poll fetches the most appropriate user
//    agent for the chat context IMMEDIATELY. Each user agent in 'suggest'
//    mode is polled for a user-sided turn IMMEDIATELY (no pause / delay),
//    producing a diegetic in-interface string indicating the user-agent's
//    response is being polled."
//
// The pre-existing specs that *claim* to validate this are simulacra:
//   - 66_auto_poll_k1.spec.js targets `iframe#user_personas_iframe` -- an id
//     that DOES NOT EXIST in the shipped DOM (the real wrapper id is
//     `user-suggester-button`; the iframe is matched by src*="suggester.html")
//     and calls test.skip() on EVERY failure branch => it asserts nothing.
//   - 80_suggester_no_defeatist_banner.spec.js does request.get()+regex over
//     the HTML *source*; it never boots a browser, never runs the JS.
//   - 94/95 DO drive a browser but page.route()-mock BOTH /yapper-seed AND
//     /poll, so they only prove "the panel renders given canned fixtures",
//     not that the real event -> generate -> poll -> render chain fires.
//
// WHAT THIS DOES DIFFERENTLY (NOT A SIMULACRUM)
// ---------------------------------------------
// It loads the EXACT shipped bytes of suggester.html + csrf-fetch.js off disk
// and executes them in real Chromium via Playwright. It stubs ONLY THE HOST
// (the SillyTavern shell `window.parent.SillyTavern.getContext()`) and the
// backend HTTP endpoints (synthesized by route interception -- no server). The
// feature code itself is never reimplemented or mocked. It then drives the
// empirical user-sided behavior: a chat is "tabbed in", and we assert -- with
// ZERO operator clicks -- that the REAL client (a) generates via /yapper-seed,
// (b) auto-fires /poll for the top-K rows immediately, (c) renders the diegetic
// row, and (d) lands the suggested prose. A failure pinpoints the broken LINK.
//
// HOW TO RUN (no ST server, no bridge required):
//   cd tools/st-debug/tests
//   npx playwright test --config runtime-client/playwright.runtime.config.js
//
// Point at a different client source with:
//   SUGGESTER_DIR=/path/to/plugins/user-personas/static npx playwright test ...
// ============================================================================

import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

// Canonical client source = the root checkout (where edits land).
const SUGGESTER_DIR = process.env.SUGGESTER_DIR
    || '/Users/mdot/sillytavern-fork/plugins/user-personas/static';
const SUGGESTER_HTML = path.join(SUGGESTER_DIR, 'suggester.html');
const CSRF_JS = path.join(SUGGESTER_DIR, 'csrf-fetch.js');
// The ONE Suggester core (suggester_core.js). The iframe is now a
// `<script type="module">` that `import './suggester_core.js'`s this exact
// file; in production the plugin streams it at /static/suggester_core.js (it
// physically lives in the FE extension dir, NOT the static dir). The harness
// must fulfil that module request with the real bytes — otherwise the catch-all
// returns an empty 200, the static import fails, and the WHOLE module never
// boots (no /yapper-seed, no rows). Overridable for an alternate checkout.
const SUGGESTER_CORE_JS = process.env.SUGGESTER_CORE_JS
    || '/Users/mdot/sillytavern-fork/public/scripts/extensions/user-personas/suggester_core.js';

// Fabricated origin. Nothing listens here -- every request is route-fulfilled
// before the network layer, so no server is contacted.
const ORIGIN = 'http://localhost:7331';
const DOC_URL = `${ORIGIN}/api/plugins/user-personas/static/suggester.html`;

const POLL_MARKER = 'RUNTIME_CLIENT_POLL_PROSE';

function rankRow(id, i) {
    return {
        bio_id: `${id}.png`,
        agent_id: `${id}-agent`,
        distance: 0.2 + i * 0.1,
        why: `runtime-client fixture ${id}`,
        persona: { id: `${id}.png`, name: `Persona ${id}`, bio: `${id} bio`, provenance: { kind: 'runtime-client' } },
        agent: { id: `${id}-agent`, name: `Agent ${id}`, designed_for_bio_id: `${id}.png`, provenance: { kind: 'runtime-client' } },
    };
}

function yapperBody() {
    return {
        top: [rankRow('alpha', 0), rankRow('beta', 1)],
        side: [rankRow('gamma', 2)],
        _meta: {
            K_top: 2, K_side: 1,
            target_signature: { tone: 3 },
            target_completed_axes: 1,
            candidates_considered: 3, bios_total: 3, agents_total: 3,
            pending_synthesis: [], pending_count: 0,
        },
    };
}

function loadClientSource() {
    if (!fs.existsSync(SUGGESTER_HTML)) {
        throw new Error(`real client source not found: ${SUGGESTER_HTML} (set SUGGESTER_DIR)`);
    }
    if (!fs.existsSync(SUGGESTER_CORE_JS)) {
        throw new Error(`Suggester core not found: ${SUGGESTER_CORE_JS} (set SUGGESTER_CORE_JS). ` +
            `The iframe imports it as an ES module; without the real bytes the static import fails and nothing boots.`);
    }
    const html = fs.readFileSync(SUGGESTER_HTML, 'utf8');
    const csrf = fs.existsSync(CSRF_JS) ? fs.readFileSync(CSRF_JS, 'utf8') : '';
    const core = fs.readFileSync(SUGGESTER_CORE_JS, 'utf8');
    return { html, csrf, core };
}

async function installHarness(page, { chat, calls }) {
    const { html, csrf, core } = loadClientSource();

    // Catch-all FIRST (lowest priority): keeps stray requests (favicon, etc.)
    // from erroring. Specific routes registered after this win.
    await page.route('**/*', route =>
        route.fulfill({ status: 200, contentType: 'text/plain', body: '' }));

    // Serve the REAL document + its REAL csrf-fetch.js (unmodified bytes).
    await page.route('**/suggester.html', route =>
        route.fulfill({ status: 200, contentType: 'text/html', body: html }));
    await page.route('**/csrf-fetch.js', route =>
        route.fulfill({ status: 200, contentType: 'application/javascript', body: csrf }));
    // Serve the REAL suggester_core.js to the iframe's `<script type="module">`
    // static import. The catch-all above would otherwise return an empty 200,
    // the import would resolve to a module with no exports, and the whole
    // module body would throw on the first `INITIAL_K_TOP` reference. This
    // route is what lets the iframe share the ONE core (no inline mirror).
    // MUST carry a JS MIME type — browsers reject module imports served as
    // text/plain (strict MIME checking for module scripts).
    await page.route('**/suggester_core.js', route =>
        route.fulfill({ status: 200, contentType: 'application/javascript', body: core }));
    // csrf-fetch.js does a GET /csrf-token before the first unsafe request.
    await page.route('**/csrf-token', route =>
        route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ token: 'disabled' }) }));

    // Backend endpoints -- synthesized, NOT served by any process.
    await page.route('**/api/plugins/user-personas/yapper-seed', async route => {
        calls.push({ kind: 'yapper-seed', t: Date.now() - calls._t0 });
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(yapperBody()) });
    });
    await page.route('**/api/plugins/user-personas/poll', async route => {
        const b = route.request().postDataJSON() || {};
        calls.push({ kind: 'poll', t: Date.now() - calls._t0, bio: b.persona_id, agent: b.agent_id });
        // Small delay so the diegetic "polling" state is observable BEFORE the
        // prose lands -- lets the harness witness the in-flight indicator too.
        await new Promise(res => setTimeout(res, 250));
        await route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({
                applied_overlay: {
                    source: 'agent',
                    agent_id: b.agent_id || `${b.persona_id}-agent`,
                    name: `Agent for ${b.persona_id}`,
                    depth: 1, text_chars: POLL_MARKER.length,
                },
                candidates: [{ text: `${POLL_MARKER}: drafted next user turn for ${b.persona_id}.`, truncated: false }],
            }),
        });
    });
    await page.route('**/api/plugins/user-personas/bridge-status', route =>
        route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ reachable: true }) }));
    await page.route('**/api/plugins/user-personas/agents', route =>
        route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({ agents: [
                { id: 'alpha-agent', designed_for_bio_id: 'alpha.png' },
                { id: 'beta-agent', designed_for_bio_id: 'beta.png' },
                { id: 'gamma-agent', designed_for_bio_id: 'gamma.png' },
            ] }),
        }));
    await page.route('**/api/plugins/user-personas/personas', route =>
        route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({ personas: [
                { id: 'alpha.png', name: 'Persona alpha' },
                { id: 'beta.png', name: 'Persona beta' },
                { id: 'gamma.png', name: 'Persona gamma' },
            ] }),
        }));
    await page.route('**/api/plugins/user-personas/dispatch-missing-agent-synth', route =>
        route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({ ok: true, dispatched: 0, in_flight: 0, bios: [], experimentIds: [], bios_in_corpus: 3 }),
        }));

    // Inject the ST host. This is the ONLY thing we stub besides the backend:
    // the shell the iframe reads via window.parent.SillyTavern.getContext().
    // The harness loads suggester.html at the TOP level, so window.parent ===
    // window; window.SillyTavern below is what the iframe code resolves.
    await page.addInitScript((chatArg) => {
        const eventTypes = {
            CHAT_CHANGED: 'chat_changed',
            MESSAGE_RECEIVED: 'message_received',
            MESSAGE_SENT: 'message_sent',
            CHAT_LOADED: 'chat_loaded',
        };
        const handlers = {};
        const eventSource = {
            on: (ev, cb) => { (handlers[ev] = handlers[ev] || []).push(cb); },
            emit: (ev, ...a) => (handlers[ev] || []).forEach(cb => cb(...a)),
        };
        const ctx = {
            eventSource,
            eventTypes,              // st-context.js:135 alias
            event_types: eventTypes, // st-context.js:218 alias
            characterId: 0,
            chatId: 'runtime-client-chat',
            characters: [{ avatar: 'dicemother.png', name: 'dicemother' }],
            chat: chatArg,
        };
        window.SillyTavern = { getContext: () => ctx };
    }, chat);
}

const POPULATED_CHAT = [
    { is_user: false, name: 'dicemother', mes: 'The inn yard is muddy and quiet. A sealed satchel rests on an overturned cask.' },
    { is_user: true, name: 'scringlo scrambler', mes: 'I set the satchel down carefully and check whether anyone followed me into the yard.' },
    { is_user: false, name: 'dicemother', mes: 'No one follows. Something inside the satchel shifts against the leather.' },
];

test.describe('user-agent suggester (iframe) -- REAL runtime client, server-free', () => {
    test('tab-in -> generate -> IMMEDIATE suggest-mode poll -> diegetic prose (zero operator clicks)', async ({ page }) => {
        const calls = [];
        const consoleErrors = [];
        page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });
        page.on('pageerror', e => consoleErrors.push(`pageerror: ${e.message}`));

        await installHarness(page, { chat: POPULATED_CHAT, calls });

        calls._t0 = Date.now();
        await page.goto(DOC_URL, { waitUntil: 'domcontentloaded' });

        // LINK 1: tab-in -> backend generates a user agent for this chat.
        await expect.poll(
            () => calls.filter(c => c.kind === 'yapper-seed').length,
            { timeout: 10_000, message: 'LINK-1 BROKEN: tab-in did not trigger /yapper-seed (user-agent generation) in the real client' },
        ).toBeGreaterThan(0);

        // LINK 2: each top-K user agent in suggest-mode is polled IMMEDIATELY.
        await expect.poll(
            () => calls.filter(c => c.kind === 'poll').length,
            { timeout: 10_000, message: 'LINK-2 BROKEN: top-K rows did not auto-fire /poll (immediate suggest-mode user-turn poll) without a click' },
        ).toBeGreaterThan(0);

        // LINK 2b: "immediately" -- the first /poll lands promptly after tab-in.
        const firstPoll = calls.find(c => c.kind === 'poll');
        expect(firstPoll.t, `LINK-2b TIMING: first /poll must fire promptly after tab-in (was ${firstPoll.t}ms)`).toBeLessThan(6000);

        // LINK 2c: it is the top picks (K_1) that auto-fire, in parallel.
        const polledBios = calls.filter(c => c.kind === 'poll').map(c => c.bio).sort();
        expect(polledBios, 'LINK-2c: both top-K bios auto-polled (parallel suggest-mode)').toEqual(['alpha.png', 'beta.png']);

        // LINK 3: a diegetic ranked row rendered from the generation result.
        const realRows = page.locator('.ranked-row:not([data-bio-id^="__skeleton"])');
        await expect(realRows.first(), 'LINK-3: ranked rows rendered from /yapper-seed result').toBeVisible({ timeout: 10_000 });

        // LINK 4: the suggested user-turn prose lands in the row WITHOUT a click
        // -- the user-observable signal that the agent's response was polled.
        await expect(
            page.locator('.row-completion-text').filter({ hasText: POLL_MARKER }).first(),
            'LINK-4 BROKEN: suggested user-turn prose never rendered into the row-completion slot without an operator click',
        ).toBeVisible({ timeout: 10_000 });

        const timeline = calls.map(c => ({ ...c }));
        console.log('[runtime-client] EMPIRICAL TIMELINE:', JSON.stringify(timeline));
        if (consoleErrors.length) console.log('[runtime-client] PAGE ERRORS:', JSON.stringify(consoleErrors));

        // This test issues ZERO page.click()/fill() -- every request above was
        // produced by the real client autonomously on tab-in. That IS the
        // "immediate, no operator action" contract, measured empirically.
    });

    // Secondary: the no-chat-no-event invariant. suggester.html:671 defines
    // activeChatHasContent() with the comment "if this is false the suggester
    // does nothing -- no spinner, no request, no endpoint call", and the spec
    // (multi_user_agent_chat_interface_spec.md:198) says "If the chat is empty,
    // no auto-poll fires." This measures whether that is actually enforced on
    // the real auto-fire path.
    test('empty chat -> no auto-fired /poll (no-chat-no-event invariant)', async ({ page }) => {
        const calls = [];
        await installHarness(page, { chat: [], calls });
        calls._t0 = Date.now();
        await page.goto(DOC_URL, { waitUntil: 'domcontentloaded' });

        // Give the client ample time to (mis)fire.
        await page.waitForTimeout(3000);

        const polls = calls.filter(c => c.kind === 'poll');
        const yappers = calls.filter(c => c.kind === 'yapper-seed');
        console.log('[runtime-client] EMPTY-CHAT TIMELINE:', JSON.stringify(calls.map(c => ({ ...c }))));
        expect(
            polls.length,
            `NO-CHAT-NO-EVENT VIOLATION: empty chat auto-fired ${polls.length} /poll call(s) ` +
            `(and ${yappers.length} /yapper-seed). suggester.html:671 activeChatHasContent() claims ` +
            `"no request, no endpoint call" on an empty chat, but the rank/auto-fire path does not enforce it.`,
        ).toBe(0);
    });
});
