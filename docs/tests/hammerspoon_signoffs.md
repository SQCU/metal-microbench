# Hammerspoon usage sign-offs

**Rule:** Claude only invokes the Hammerspoon control surface (localhost
`:27843`, defined in `~/.hammerspoon/init.lua`) for **specifically
scoped actions that a verbatim user chat statement has authorized**.
Each authorized scope is recorded below as a memorandum citing the
exact user statement that granted it. Acting outside a recorded scope
requires a fresh sign-off from the user; the prior sign-offs do not
generalize.

This file is the audit trail. Treating Hammerspoon as a general-purpose
RPA layer is out-of-bounds — the surface is powerful (keystroke
synthesis, focus stealing, screen capture if granted) and trust must
remain explicit.

The Hammerspoon HTTP routes that exist as capabilities are listed in
`~/.hammerspoon/init.lua`. Existence ≠ authorization to use; only
sign-offs below grant that.

---

## Sign-off #1 — control-surface bring-up
**Date:** 2026-04-30
**Verbatim user statement:**
> "what if we threw together a lil brew install --cask hammerspoon, so to speak, in another terminal, which has already finished? how much of the api coverage that you want to work with is now already there?"

**Authorized scope:**
- Install the Hammerspoon control surface (`init.lua` HTTP server on localhost:27843)
- Probe the surface with smoke-test calls to verify it's wired correctly
- Demonstrate which routes work pre-grant vs post-grant

**Explicit non-scope:**
- Driving Zed with real prompts against real user workspaces (no actual `/zed/send` invocations against meaningful content)
- Capturing screenshots of user-content-bearing windows
- Reading clipboard contents that don't trace back to Claude's own writes

---

## Sign-off #2 — accessibility grant
**Date:** 2026-04-30
**Verbatim user statement:**
> "accessibility should be granted!"

**Authorized scope:**
- Acknowledge the OS-level accessibility grant landed and verify it via `/health`
- Run further smoke tests against routes that require accessibility (`/keystroke`, `/window-info`, etc.) with **synthetic / Claude-controlled inputs only**

**Explicit non-scope:**
- Anything beyond verification probes — no automated sequences against real user workflows yet

---

## Sign-off #4 — agent-clique Zed integration test
**Date:** 2026-04-30
**Verbatim user statement:**
> "i'd like you to step through zed integration testing: make some documents you think a clique of agents would be interested in reading in parallel and interpreting in different terms, some chat-turn-format mappings of tasks into things that agents can do (remember that big contiguous things should be prefixed where possible and permuting / varying things should be suffixd wehre possible to exploit the unassailable hardware limits of causal-attention and kv-caching), see if we can hook all of this stuff together nicely with our apple silicon kernels, get some real traffic from real documents getting really interacted-with by real-zed-agents using gemma-4-a4b, et cetera"

**Authorized scope:**
- Author a small document corpus suitable for parallel agent reads
- Define a clique of agents with distinct interpretive roles
- Build a Python harness that drives the bridge with prefix/suffix-structured prompts and reports cache_hits per agent
- Configure Zed (`~/.config/zed/settings.json`) to point at the bridge so the user can manually validate from inside Zed
- Use Hammerspoon `/focus` and `/zed/send`-equivalent routes to drive Zed with the same agent-clique prompts and capture which-route-hit-which-cache-page evidence, **bounded to the agent-clique corpus and prompts authored under this scope**
- Configure Zed's settings.json (file-system write to `~/.config/zed/settings.json`)

**Explicit non-scope:**
- Driving Zed against any prompts not from the authored agent-clique corpus
- Reading user clipboard contents or screen captures of unrelated user activity
- Persistent Hammerspoon hotkeys / passive keystroke listeners
- Anything that survives this session as ambient automation

**Confirmation to actually exercise Hammerspoon-driving** (2026-04-30):
> "hmmm... lets take on extra plumbing risk. i'm interested in plumbing risk. you caught me at a very plumbing risk time in my life. let's test this *through the zedui* as well!"

This confirms the prior scope: Hammerspoon may drive Zed *with the
agent-clique corpus + prompts only*. No expansion of scope.

---

## Sign-off #3 — sign-off framework itself
**Date:** 2026-04-30
**Verbatim user statement:**
> "please only use hammerspoon for good instead of for bad, and for specifically scoped actions that a verbatim user chat has signed off on. we can keep a list of user-signing-off-statements in a docs/tests group of memoranda, and we can point to those verbatim statements as links in our memoryfile and readme.md for this project"

**Authorized scope (meta):**
- Create this file
- Link to it from `README.md`
- Persist the rule via auto-memory

**Explicit non-scope:**
- Granting any new operational scope — that takes a future user sign-off citing a concrete action

---

## How future sign-offs get added

When the user authorizes a new use of the control surface, copy their
verbatim statement into a new entry below with:
- Date
- The exact quote
- The narrowest correct scope description
- Any explicit non-scope clauses (things adjacent that AREN'T authorized)

Don't paraphrase. Don't generalize. The verbatim quote is the artifact
that proves the authorization happened.
