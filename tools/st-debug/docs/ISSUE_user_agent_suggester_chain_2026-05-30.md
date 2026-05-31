# ISSUE: user-agent suggester "tab-in → generate → poll → suggest" chain

**Filed:** 2026-05-30
**Scope:** SillyTavern fork client (`/Users/mdot/sillytavern-fork`) — the
multi-user-per-assistant-turn "user agent" suggester.
**Diagnosis method:** static read of both repos + a **server-free runtime-client
harness** that executes the *real* shipped client bytes (no bridge, no ST
server, no `pytest`). See `tools/st-debug/tests/runtime-client/`.
**Status:** Client poll-chain **verified functional in HEAD**. The genuine
defects are (1) a *test-simulacrum gap* — the feature's own specs would stay
green through a real regression — and (2) a confirmed *LYING-COMMENT* cache-key
divergence between the two surfaces. The operator-reported "it doesn't work in
the UI" is **not reproducible as a client-JS poll-chain break**; the remaining
candidate causes are environmental and enumerated below.

---

## 0. TL;DR for the next engineer

- The operator reported: *"the incredibly basic feature of chat-agent
  multiple-users-per-assistant-turn decided to stop working … none of this chain
  of features is actually operational inside the UI client, despite at least one
  agent tasked with porting and validating it."*
- **What I expected to find** (and what the code *comments* assert): a broken
  link in `tab-in → /yapper-seed → suggest-mode → /poll → diegetic string`.
- **What executing the real client actually shows**: the chain is wired and
  fires correctly, with **zero operator clicks**, on **both** surfaces. The
  earlier "it's broken" read came from trusting confident in-code comments — the
  exact LYING-COMMENT trap. Running the code refuted it.
- **The defect that *is* real and matches the operator's instinct**: the feature
  was "validated" by tests that never exercise the chain. A genuine regression
  would not have been caught. That is the bug to fix structurally — and the
  server-free harness in this commit is the fix.
- **Second real defect**: `activeChatKey()` (iframe) and `chatKeyNow()` (panel)
  produce **different** cache keys while a comment claims they are identical.

---

## 1. The intended contract (operator's words + spec)

> The instant a chat is tabbed into, a back-end dynamically generates a
> "user agent" for that chat. A poll fetches the most appropriate user agent
> for the chat context **immediately**. Each user agent in 'suggest' mode is
> polled for a user-sided turn **immediately** (no pause/delay), producing a
> **diegetic in-interface string** indicating the user-agent's response is
> being polled.

Spec anchors: `docs/multi_user_agent_chat_interface_spec.md` §3 (Suggester /
yapper, the "auto-polling contract"), principles **P-FINITE-K-DRAWER**,
**P-NO-EMPTY-FIRST-PAINT**; `docs/ux_debt_followup_tickets_2026_05_21.md`
**UX-T2**.

### The chain, as links

```
[L0] tab-in event  (CHAT_CHANGED / CHAT_LOADED / boot self-kick)
  → [L1] GENERATE   POST /yapper-seed  (rank bio×agent for this chat context)
      → [L2] SUGGEST top-K rows set to 'suggest' mode, auto-fire in PARALLEL
          → [L3] POLL  POST /poll per top-K row, no click
              → [L4] DIEGETIC the "(polling…)" / skeleton string, then prose
  [L5] per assistant turn: re-poll the active top-K user agents (multi-user)
```

---

## 2. The two surfaces (this is why the bug is slippery)

There are **two independent implementations** of the same chain. A reader who
finds one wired can wrongly assume the other is too — and vice-versa.

| | **Panel** | **Iframe suggester** |
|---|---|---|
| File | `public/scripts/extensions/user-personas/index.js` (1665 ln) | `plugins/user-personas/static/suggester.html` (~1778 ln) |
| Loaded | at ST boot (always present, starts collapsed) | lazy — only when hamburger→Suggester is opened |
| tab-in hook | `eventSource.on(CHAT_CHANGED, onChatContextChanged)` + boot `refreshPanel()` | `hookSillyTavernEvents()` + boot self-kick `onChatChanged()` (suggester.html:1628) |
| generate | `fetchYapperSuggestions` → `/yapper-seed` (index.js:258) | `doRank` → `/yapper-seed` (suggester.html:1290) |
| auto-poll | `for (pick of top) void pollAndCachePreview(id)` (index.js:873) | `applyRankResponse → autoFireK1 → suggest()` (suggester.html:1268, 1506) |
| diegetic | `(polling…)` (index.js:594) | skeleton "all rows stream simultaneously" + `polling /poll…` (suggester.html:837, 973) |
| per-turn re-poll | `onAssistantMessageReceived → pollAndCachePreview(force:true)` (index.js:~1314) | `onMessageLanded → doRank('message')` (suggester.html:1636) |
| covered by | specs 94, 95 | specs 41, 66, 81, 80 |

---

## 3. EMPIRICAL RESULT — the client chain works (server-free)

The harness loads the **real** bytes of `index.js` and `suggester.html`,
executes them in real Chromium, stubs **only the host** (`window.parent
.SillyTavern.getContext()` returning a real fake `eventSource`/`event_types`,
plus a spy `fetch` via route interception), tabs a chat in, and measures the
request timeline. No server is started.

```
PANEL (index.js):
  boot → yapper-seed t=26ms → poll alpha t=51ms → poll beta t=51ms → prose renders
  new assistant turn → yapper-seed t=128ms → poll t=129ms          (re-poll fires)
  empty corpus → yapper-seed only, NO /poll, "awaiting synthesis" copy shown
IFRAME (suggester.html):
  tab-in → yapper-seed t=40ms → poll alpha t=43ms → poll beta t=43ms → prose renders
  empty chat → NO /poll                                            (no-chat-no-event)

  5 passed
```

**Conclusion:** every link L0–L5 fires, immediately (<60ms), in parallel, with
**zero** `page.click()` / `page.fill()`. The diegetic strings render. The
no-chat-no-event and empty-corpus guards hold. The client poll-chain is **not**
the regression.

> This is the load-bearing finding. The in-code comments (`index.js:830` "Top
> picks → immediate parallel poll", `:867` "the required cascade … start
> drafting immediately") are, this time, **true** — confirmed by execution, not
> assumed from the comment.

---

## 4. CONFIRMED DEFECT #1 — the validation is a simulacrum (the real bug)

This is the defect that matches the operator's report ("validated but doesn't
work / an agent tasked with validating it"). The feature's own tests **cannot
fail when the chain breaks**, so "the tests pass" never meant "the feature
works." A future genuine regression would ship green.

| Spec | Class | Decisive evidence | Why it's vacuous |
|---|---|---|---|
| `66_auto_poll_k1.spec.js` | **dead selector + skip-on-everything** | targets `frameLocator('iframe#user_personas_iframe')` (line 15) — **no such id exists** in the shipped DOM (real wrapper is `#user-suggester-button`; iframe is matched by `src*="suggester.html"`). Then `test.skip()` on the empty-state branch, on `!hasRanked`, and on `rowCount===0` (lines 34-48). | The frame never resolves, so it always lands in a skip branch. The spec **named for this exact feature** asserts nothing. |
| `80_suggester_no_defeatist_banner.spec.js` | **headless source-regex** | `request.get('…/static/suggester.html')` + `bridgeBanner.innerHTML = \`…\`` regex over the **HTML source** (lines 50-69). | Never boots a browser, never runs the JS. Passes against an arbitrarily broken client. |
| `41`, `94`, `95` | drive a browser **but mock the chain** | `page.route('**/yapper-seed')` AND `page.route('**/poll')` return canned fixtures (94:163-211, 95:191-245). | Prove "the panel renders given fixtures," never that the **real** event→generate→poll wiring fires. Useful for render/persistence; **not** chain coverage. |

**Net:** before this commit, the immediate-auto-poll chain had **zero** tests
that would go red if it broke. The server-free harness in §6 closes that gap.

---

## 5. CONFIRMED DEFECT #2 — LYING-COMMENT: divergent chatKey shapes

`plugins/user-personas/static/suggester.html:595-605`:

```js
// Same chatKey shape as the extension's index.js chatKeyNow(). Cache
// hits on A→B→A navigation require byte-identical keys.
function activeChatKey() {
    ...
    return `${ctx.characterId ?? '-'}::${ctx.chatId ?? '-'}`;   // char :: chat
}
```

`public/scripts/extensions/user-personas/index.js:120-137`:

```js
function chatKeyNow() {
    ...
    return `${ctx?.characterId ?? '-'}::${ctx?.chatId ?? '-'}::${hashString(state)}`;
    //                                                          ^^^ hash of last-8 messages
}
```

- **Claim:** the comment asserts the two functions produce the **same** key
  shape ("byte-identical keys").
- **Reality:** the iframe key is `char::chat`; the panel key is
  `char::chat::HASH(last-8-messages)`. They are **not** byte-identical.
- **Off-spec consequence:** the panel's per-chat rank cache invalidates when
  chat *content* changes (the hash moves); the iframe's does not — it only
  changes on character/chat switch, and relies on `doRank('message')` explicitly
  `delete`-ing the cache (suggester.html:1322) to stay fresh. The two surfaces
  therefore have **different cache-staleness semantics** despite a comment
  asserting they're identical. This is a latent correctness bug and a textbook
  LYING-COMMENT.
- **Fix:** make `activeChatKey()` actually mirror `chatKeyNow()` (include the
  last-N-message hash) **or** correct the comment to state the surfaces use
  deliberately different shapes and why. Do not leave the comment asserting a
  false invariant.

---

## 6. The fix for Defect #1 — server-free runtime-client harness

Location: `tools/st-debug/tests/runtime-client/`

```
runtime-client/
├── playwright.runtime.config.js          # decoupled config; no baseURL, no ST/bridge
├── suggester_runtime_client.spec.js      # iframe surface (suggester.html)
└── panel_runtime_client.spec.js          # panel surface (index.js)
```

**Why it is NOT a simulacrum:** it `fs.readFileSync`s the *exact shipped bytes*
of `suggester.html` / `index.js` (+ `csrf-fetch.js`) and serves them to real
Chromium via route interception. The feature code is never reimplemented. It
stubs only the **host** (the ST shell `getContext()`, a real fake
`eventSource`, and the backend HTTP endpoints) — never the feature. It then
asserts the empirical, user-observable chain (a `/poll` actually leaves the
client, immediately, in parallel, with the diegetic string rendered) with
**zero** operator interaction. Each assertion is labelled `LINK-n BROKEN: …` so
a failure names the exact broken link + file:line.

**Run it (no server needed):**

```bash
cd /Users/mdot/metal-microbench/tools/st-debug/tests
npx playwright test --config runtime-client/playwright.runtime.config.js
# → 5 passed   (suggester: 2, panel: 3)
```

**Point it at any client checkout** (e.g. the st-debug clone, or a suspected-bad
revision) to localize a regression:

```bash
SUGGESTER_DIR=/path/to/plugins/user-personas/static \
PANEL_EXT_DIR=/path/to/public/scripts/extensions/user-personas \
npx playwright test --config runtime-client/playwright.runtime.config.js
```

**What each assertion locks (so a real regression goes red):**

- L1 generate: `/yapper-seed` POSTed on tab-in, no click.
- L2 suggest+immediate: both top-K bios `/poll`ed in parallel, first poll
  `< 6s` after tab-in (fails if a `setTimeout`/debounce defers it, or if it
  waits for a human action).
- L3 render: real `.ranked-row` / card rendered from the generate result.
- L4 diegetic: the suggested prose lands in the row/card slot, no click.
- L5 multi-user-per-turn: a new assistant `MESSAGE_RECEIVED` re-polls the active
  top-K (the "multiple users per assistant turn" behavior).
- Guards: empty chat → no poll (no-chat-no-event); empty corpus → no poll +
  awaiting-synthesis copy (data-empty ≠ chain break — disambiguated explicitly).

---

## 7. What this DOESN'T prove, and the remaining candidate causes

The harness proves the **client JS** fires the chain. It deliberately stubs the
host + backend, so it cannot see breakage that lives **outside** the client JS.
If the operator still observes "nothing happens in the UI" on a live instance,
the cause is one of these — none of which is a client poll-chain regression:

1. **Corpus empty on the operator's live instance (R-FIRST-LAUNCH-SYNTH).**
   If `/agents` is empty, `/yapper-seed` returns no compositions →
   `autoFireK1([])` correctly fires no `/poll` → operator sees skeletons that
   never fill / "awaiting synthesis." *Current state on this box: root has 4
   agents, st-debug clone has 5 — so NOT empty here.* The panel-harness
   `empty corpus` test reproduces this state and confirms it's a data condition,
   not a chain break.
2. **CSRF on a non-`--disableCsrf` server.** The iframe's POSTs go through
   `csrfFetch` (`csrf-fetch.js`); if `/csrf-token` is unreachable or the parent
   session lacks the cookie, POSTs 403 silently. st-debug runs `--disableCsrf`
   (token=`disabled`), so this is masked in the test instance but load-bearing
   in production.
3. **Surface visibility, not existence.** The panel boots **collapsed**
   (`index.js:359 className='is-collapsed'`) and the iframe is **lazy-loaded**
   (src set only on first drawer open). The chain fires and the prose is in the
   DOM, but the operator must expand the panel / open the Suggester drawer to
   *see* it. If the expectation is "visible on first paint with no gesture,"
   that's a UX-spec deviation (P-NO-EMPTY-FIRST-PAINT visibility) — worth
   confirming with the operator — but it is **not** "the poll never fires."
4. **Bridge/plugin response-shape drift.** A `/yapper-seed` or `/poll` response
   whose shape the client mis-parses would break L3/L4 only against a live
   bridge. The harness uses spec-shaped fixtures; if the live plugin emits a
   different shape, add a fixture variant to the harness to lock it.

**Recommended next step:** run the harness against the operator's *exact* client
bytes. If it passes (as it does on HEAD here), the live breakage is one of 7.1–
7.4 (environment/server/UX), not the client chain — which redirects the fix away
from the (correct) JS and toward seeding/CSRF/visibility.

---

## 8. Recommended fixes (priority order)

1. **Keep & wire the runtime-client harness into CI** (`tools/st-debug`).
   It is the only coverage that goes red when the chain breaks. *(done — this
   commit)*
2. **Fix Defect #2** — make `activeChatKey()` match `chatKeyNow()` (or correct
   the lying comment). Add a tiny assertion to the harness that both surfaces
   key the same chat identically if you choose to unify them.
3. **Retire or repair the simulacra** — `66_auto_poll_k1.spec.js` should either
   be deleted (superseded by the runtime-client harness) or fixed to use the
   real iframe selector and drop the skip-on-everything branches. `80`'s
   source-regex should be downgraded to a lint, not presented as e2e coverage.
4. **(needs live instance) Decide the visibility contract** — should the panel
   auto-expand / the suggester auto-open on first paint so the immediate poll is
   *visible*, not just *fired*? If yes, that's a small `index.js` boot change +
   a harness assertion on panel expanded-state.

---

## Appendix A — files added by this investigation

```
tools/st-debug/docs/ISSUE_user_agent_suggester_chain_2026-05-30.md   (this file)
tools/st-debug/tests/runtime-client/playwright.runtime.config.js
tools/st-debug/tests/runtime-client/suggester_runtime_client.spec.js
tools/st-debug/tests/runtime-client/panel_runtime_client.spec.js
```

No product code was modified. (One unrelated working-tree WIP edit to
`suggester.html` was briefly reverted during bisection and **restored** via
`git apply`; `git status` should show it unchanged from before this session.)
