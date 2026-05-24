# Project archeology: multi-user chat-agent suggestion interface

**Codebase examination began:** 2026-05-21T21:26:54Z

**Scope:** 211 MB Claude-Code session transcript (1094 user messages, 19266 assistant turns)
spanning 2026-05-05 through 2026-05-21, cross-referenced against
**57 sillytavern-fork commits** (zyzzyva-authored) and **210 metal-microbench commits**.

**Why this document exists:** the operator has flagged that the project contains
more accepted-to-spec features and design principles than recent regression
cycles have preserved in working code. This archeology surfaces every version
of the multi-user-chat-agent-suggestion interface that has been discussed
(positively or negatively) and indexes them against the commit sequences that
produced them. Use this when you (the assistant) feel like you're
greenfield-designing — you almost certainly aren't.

**Counts:** 277 indexed events (149 transcript design events + 128 commits).

---


## Interface vocabulary (the things that keep changing)

- **toolcard / tool-call card** — the visible-residue contract: every tool invocation leaves a card in chat DOM
- **forked agent / descendant summarization** — long tool runs are captioned by tiny parallel summarizer agents
- **user-personas plugin** — multi-bio + multi-agent suggestion + autonomous-tick orchestration
- **suggester / yapper** — the panel that ranks (bio × agent) compositions against chat-context signature
- **user-agent drawer** — finite-K (k_1 active + k_2 disabled) picker UI; NEVER full corpus, NEVER empty
- **designer (bio / agent)** — auto-chained from selection; agentless bios redirect here
- **fixed-point iteration** — synthesis-loop UI bound to experiment cards
- **axes registry / axis-as-card** — dynamic plugin-queried registry; cards in `plugins/user-personas/axes/`
- **agent-as-card** — chara_card_v3 PNGs with embedded JSON; canonical agent storage
- **bridge stream lifecycle** — disconnect polling + bounded response_q overflow cancel + forward-progress deadline
- **invertChatForPersona** — KV-share refactor; shared-prefix bio-aware chat inversion


## Principles repeatedly asserted by the operator (load-bearing)

1. **Finite K-affinity picker** — drawer surfaces k_1 immediately-polling + k_2 suggested-disabled, never the full corpus, never empty.
2. **Per-chat caching, no DoS patterns** — /poll completes + caches, never aborts on navigate. A→B→A is O(1).
3. **Ontological closure** — bios without agents are unusable; usable bios must have agents (selection IS design).
4. **Vectors persist; no runtime recomputation** — /yapper-seed uses stored signatures, not cold extraction.
5. **API client = GUI client contracts** — every endpoint exercisable via GUI affordance for e2e validation.
6. **Visible residue contract** — every tool invocation leaves a card in DOM; silent deletion forbidden.
7. **End-to-end validation only** — Playwright pixel-space tests; unit tests not valid evidence.
8. **No empty/JSON-field forms** — interfaces always carry contextual suggestions + examples.
9. **Chat is the central interface** — drawers cover the chat ONLY when the operator explicitly chose that surface.
10. **Subagent/LLM-driven sorting** — discretion via API calls is the default; deterministic ordering is the fallback.


## Critical commits (the load-bearing 15)

Read these in order to understand the project's design history. Each is a turning point.

| Date | Repo | Hash | Significance |
|------|------|------|--------------|
| 2026-05-09 | sillytavern-fork | `ec16007a7` | **Toolcards genesis**: server-authoritative plugin + FE wiring. The visible-residue contract begins here. |
| 2026-05-11 00:25 | sillytavern-fork | `6b5f74d9c` | Tool calls render as inline collapsibles, not deleted. Visible-residue enforced. |
| 2026-05-11 16:07 | sillytavern-fork | `c94f950a6` | **User-personas Phase 1**: server plugin + FE extension; suggestion mode. |
| 2026-05-11 18:21 | sillytavern-fork | `abdd6c4af` | User-personas Phase 3: multi-user dialogue (N personas, 1 assistant). |
| 2026-05-11 18:34 | sillytavern-fork | `d1ba7fd6c` | User-personas Phase 4: unified panel + chara-card-style manifest. |
| 2026-05-17 18:10 | sillytavern-fork | `317ebf5cd` | **Agentless bios are broken state**: scene-coupling requires agents. |
| 2026-05-17 18:15 | sillytavern-fork | `cbc5d45e6` | **Selection IS design**: bio→agent auto-chain; agentless-bio redirect. |
| 2026-05-17 19:08 | sillytavern-fork | `4019f9aa1b6` | Agents persisted as PNG chara_card_v3 cards. |
| 2026-05-18 23:10 | metal-microbench | `87254e8` | Bridge stream-lifecycle safety triple lands. |
| 2026-05-19 00:30 | sillytavern-fork | `9ced4d081` | **Axes as cards**: dynamic plugin-queried registry. |
| 2026-05-19 01:27 | sillytavern-fork | `497f4416d` | Fixed-point iteration tab (synthesis-loop UI). |
| 2026-05-17 23:48 | metal-microbench | `4dece95` | Context-suggester Phase A: k-active + k-disabled picks. |
| 2026-05-18 (multiple) | metal-microbench | various | Stream-lifecycle hardening + native-thread FFI. |
| 2026-05-19 01:47 | metal-microbench | `63d4291` | st-debug smoke spec for Fixed-Point Iteration tab. |
| 2026-05-19 01:49 | metal-microbench | `0f6214e` | st-debug delete obsolete persona-library specs. |


---

## Chronological log

Format: `**HH:MM UTC** [src • kind/category • interface] — summary. _quote/subject_`

Kinds: `spec` (design requirement), `positive` (delivered), `negative` (broken/regressed),
`regression` (deleted accepted work), `restoration` (reinstated deleted work).
Commit categories: `genesis`, `feature`, `refactor`, `regression`, `restoration`, `cleanup`,
`validation`, `infrastructure`.


### Period: 2026-05-05 → 2026-05-11

**Toolcards Genesis.** Forked-agent plugin with descendant summarization gets architected and shipped. Voronoi-diagram demo validates end-to-end via Playwright; unit tests rejected as evidence.


#### 2026-05-08

- **20:24** [transcript[i=7478] • spec • forked agent UI & descendant summarization] — present tool-calling descendant agent work in chat interface. _"diegetic presentation of the tool calling descendant agent stuff"_
- **20:24** [transcript[i=7478] • spec • summarization UI] — forked summarizers caption long-running tool calls compactly. _"forking off tiny 'please summarize' forkagents which can autoregressively decode"_
- **22:54** [metal@e7c844e • validation] — KV-cache trial-correlation confirmed empirically. _KV-cache trial-correlation confirmed empirically + spur-caption subagent primitive_
- **23:06** [transcript[i=7770] • spec • toolcard plugin] — integrate incremental summarization into toolcards plugin. _"can we integrate the incremental summarization primitive into the toolcards"_

#### 2026-05-09

- **00:01** [metal@69c5a81 • feature] — spur-caption integration into toolcards. _spur-caption integration into toolcards + sync-block bug found via static analysis_
- **00:13** [metal@54a64bb • validation] — fix toolcards plugin FIFO sync-block bug. _fix toolcards plugin FIFO sync-block bug (pair-programmed with codex)_
- **00:28** [metal@d938913 • validation] — fix KV-cache trial correlation client seed dropped. _fix KV-cache trial correlation: client seed was silently dropped (codex diagnosis)_
- **00:38** [transcript[i=8112] • spec • tool card examples] — example tool cards demonstrating forked agent patterns. _"come up with a few example tool 'cards' with codex as a pair"_
- **00:38** [transcript[i=8112] • spec • default persona card] — scringlo persona with distinct non-baseline voice. _"do we have a default 'card' (scrongle scrimble) which elicits"_
- **00:38** [transcript[i=8112] • spec • forked agent patterns] — demonstrate context-copying vs fresh-context forked agents. _"demonstrate persona-non-violating use of forked agents (context of the"_
- **00:38** [transcript[i=8112] • spec • async forked agents] — asynchronous forked agents with deferred interrupting results. _"asynchronous forked agents which will drop an awaited response"_
- **00:48** [metal@4a91e7c • infrastructure] — toolchain consolidation docs index isolation. _toolchain consolidation: docs index + isolation fix + scringlo persona + random-choice card_
- **01:30** [metal@8d5aa18 • feature] — toolcards shapes A/B/C example cards. _toolcards: shapes A/B/C example cards (python_exec, extended_thinking, async_lookup)_
- **06:32** [metal@a0438f7 • feature] — toolcards tree-of-thoughts shape-D card. _toolcards: tree-of-thoughts shape-D card (context-copying async, parallel branches)_
- **07:04** [metal@fd191b9 • feature] — toolcards Phase 3a cancel UI button. _toolcards Phase 3a: cancel UI button on running tool_progress entries_
- **07:12** [metal@2a274e0 • feature] — toolcards Phase 3b per-branch streaming. _toolcards Phase 3b: per-branch streaming for shape D_
- **07:22** [metal@8c24ab0 • feature] — toolcards Phase 3c fire-and-forget async. _toolcards Phase 3c: true fire-and-forget for async-flagged cards_
- **20:32** [sillytavern@ec16007a7 • genesis] — toolcards genesis server-authoritative forked-agent. _toolcards: server-authoritative forked-agent plugin + FE wiring (Phases 1-3)_
- **20:34** [metal@28f6d08 • cleanup] — remove tools/st-debug/patches directory. _remove tools/st-debug/patches/ — replaced by direct tracking in sillytavern-fork_
- **20:48** [transcript[i=8890] • spec • summarization feature validation] — video proof of sentence summarization in real ST interface. _"hammerspoon mediated screenshots of a real browser with real DOM"_

#### 2026-05-10

- **01:32** [transcript[i=8909] • spec • tree of thought + summarization] — tree-of-thought with branch_progress summary events. _"branch_progress events for one tool (tree-of-thoughts) that surface"_
- **01:54** [metal@50b19ba • feature] — bridge report finish_reason=length budget hit. _bridge: report finish_reason=length when budget hit, not "stop"_
- **02:43** [sillytavern@77d91e04e • infrastructure] — propagate Content-Type and SSE-friendly headers. _util: propagate Content-Type and SSE-friendly headers in forwardFetchResponse_
- **02:44** [metal@b5047c2 • feature] — bridge stream tokens unconditionally fix length. _bridge: stream tokens unconditionally, fix length finish_reason, e2e DOM streaming test_
- **02:49** [transcript[i=9594] • regression • tool call summarization] — summarization features not yet integrated in actual video demo. _"demonstrates none of the long running tool calling behaviors with embedded"_
- **02:52** [metal@14d4a56 • infrastructure] — bridge stop strings tool_choice 501 unsupported. _bridge: stop strings wired through, tool_choice 501 for unsupported modes, strip stale comments_
- **03:19** [metal@75771f7 • validation] — test 16 vertical slice tree-of-thoughts visible. _test 16: vertical slice — Scringlo + tree-of-thoughts + visible branch streaming, video_
- **03:35** [metal@2a69255 • validation] — test 16 dismiss API drawer post-connect. _test 16: dismiss API drawer post-connect, add visible-state checkpoints_
- **03:44** [transcript[i=9868] • spec • scalable oversight UI design] — compact n<<k summaries for recursive tool call chains. _"for k lines of code or text, we see n<<k lines of summary"_
- **03:44** [transcript[i=9868] • spec • tree of thought collapsible] — tree-of-thought visual component with summary lines in div. _"tree of thought collapsible; now we need some design work to ensure"_
- **03:55** [metal@74a879c • feature] — n-of-k scalable oversight design tree-of-thoughts. _n-of-k scalable oversight: design + tree-of-thoughts emits parent-voice summaries_
- **04:00** [metal@1e50758 • cleanup] — docs/media rename to date_test_study_sha. _docs/media: rename to date_test_study_sha convention; recover overwritten history_
- **04:07** [metal@0353f80 • feature] — docs bounded-depth recursive decomposition. _docs: bounded-depth recursive decomposition design (next demo card)_
- **04:16** [transcript[i=10067] • spec • forked vs spawned agents] — clarify: forks inherit context, spawns get assigned prefix+work. _"forked agents inherit everything, spawned agents are given a specific"_
- **04:18** [metal@9b26e22 • cleanup] — docs correct spawn-vs-fork semantics. _docs: correct spawn-vs-fork semantics + scope demo to 2-fork+2-spawn_
- **04:46** [metal@9d8bb36 • validation] — test 17 render-visual card 2-fork+2-spawn. _test 17: render-visual card — 2-fork + 2-spawn vertical slice + visible SVG_
- **04:55** [transcript[i=10210] • positive • voronoi demo validation] — successful end-to-end demo of tool calling with voronoi SVG. _"demo shown in 2026-05-09_test17_voronoi_success_9b26e22 is very confirmatory"_
- **05:31** [metal@fc764d8 • validation] — test 18 vision-review card audit test 17. _test 18: vision-review card + scringlo audits her own test 17 recording_
- **05:43** [metal@af389a3 • restoration] — vision-review switch back to parallel multimodal. _vision-review: switch back to parallel multimodal (initial diagnosis was wrong)_
- **07:05** [metal@32610d4 • feature] — vision fail honest on 0-soft-tokens instead. _vision: fail honest on 0-soft-tokens instead of silently delivering zeros_
- **07:29** [metal@0920f76 • cleanup] — ban temperature=0.0 across codebase. _ban temperature=0.0 across the codebase_
- **07:46** [metal@d032a8e • feature] — elicitation harness strip internal API helpers. _elicitation harness: strip internal-API helpers; add Scringlo persona prompt fixes_
- **08:02** [transcript[i=11299] • negative • cheating test helpers] — reject direct-invoke shims; test must go through actual interface. _"remove direct-invoke shims which doesn't work through the actual interface"_
- **08:35** [transcript[i=11655] • spec • toolcard plugin repository] — commit all toolcards as code in sillytavern-fork, not scripts. _"put every single toolcard into the root sillytavern fork as committed"_
- **23:43** [transcript[i=12628] • negative • internal ST simulation] — reject reimplementing ST in test harness; use real ST API. _"inline simulations of sillytavern in particular is rotten"_

### Period: 2026-05-11 → 2026-05-15

**Early UI Build.** Tool-card output contract failures lead to playwright-only validation mandate. user-personas Phase-1 plugin lands (suggestion mode); roster-management questions emerge for >20 personas.


#### 2026-05-11

- **01:27** [transcript[i=13027] • positive • tool calls across card types] — successful mixed tool calls: python-exec, async-lookup, etc.. _"role: 'assistant', tool_calls: [ { id: 'call_9962941ace304a07'"_
- **03:31** [transcript[i=13500] • negative • regex-based validation] — trust summarizer to work; log errors instead of regex checks. _"why don't we trust the summarizer to actually do their job, and log"_
- **05:26** [transcript[i=13923] • spec • tool-card] — tool cards and client edits must live in sillytavern. _"put all da tool cards and all da client edits to da sillytavern"_
- **05:31** [sillytavern@92262229d • feature] — gemma-4 compat chat-template-aware client surface. _gemma-4 compat: chat-template-aware client surface + toolcards + example personas_
- **05:33** [metal@e0a6f7a • infrastructure] — session 2026-05-10 thinking pipeline tool-call. _session 2026-05-10: thinking pipeline + tool-call DSL parser + regex deprecation + static viz revive_
- **05:35** [transcript[i=14005] • spec • tool-card] — tool-call rendering and parsing architecture spec. _"tool call marker leak rendered as text in ST chat"_
- **05:36** [metal@4c0c859 • cleanup] — cleanup drop install_*_toolcard.sh scripts. _cleanup: drop install_*_toolcard.sh shell scripts + dead session artifacts_
- **05:56** [sillytavern@767dced3d • feature] — toolcards personas align upstream seed convention. _toolcards + personas: align with upstream seed convention; ship persona-effort-schema_
- **05:58** [sillytavern@3d8ff226f • cleanup] — fixup actually commit seed-mechanism wiring. _fixup: actually commit the seed-mechanism wiring_
- **05:59** [metal@a54672d • cleanup] — bootstrap drop toolcards example-characters copy. _bootstrap: drop toolcards + example-characters copy loops; rely on ST upstream seed_
- **06:26** [sillytavern@bb719a090 • feature] — toolcards rendering tool_progress visible body. _toolcards rendering: every tool_progress entry now has a visible body_
- **06:27** [metal@196538b • validation] — test 27 tool-output-rendering invariants visible. _test 27: tool-output-rendering invariants — every entry has a visible body_
- **06:39** [sillytavern@b4c7fa54b • feature] — toolcards HARD INVARIANT backend output never. _toolcards: HARD INVARIANT — backend tool output is never silently dropped_
- **06:40** [metal@3b32618 • validation] — test 28 HARD INVARIANT backend tool result DOM. _test 28: HARD INVARIANT — every backend tool result appears in the chat DOM_
- **06:50** [sillytavern@6dae853b3 • feature] — toolcards every invocation leaves visible residue. _toolcards: every invocation leaves visible residue (no shape escapes the contract)_
- **06:50** [metal@2abfb7b • validation] — test 29 HARD CONTRACT tool invocation residue. _test 29: HARD CONTRACT — every tool invocation leaves visible residue + bounded termination_
- **07:19** [sillytavern@551159379 • cleanup] — script.js fix three kludges hid tool-call residue. _script.js: fix three layered kludges that hid tool-call residue from the chat surface_
- **07:19** [metal@ad0f692 • validation] — test 30 real streaming pipeline DOM graph. _test 30: real streaming pipeline via DOM graph traversal (replaces tests 28+29)_
- **07:25** [sillytavern@6b5f74d9c • refactor] — tool calls render as inline collapsibles reasoning. _REFACTOR: tool calls render as inline collapsibles (reasoning-trace pattern), not deleted_
- **07:25** [metal@892e8a0 • cleanup] — docs artifacts inline-collapsible tool-call refactor. _docs: artifacts for the inline-collapsible tool-call refactor (fork commit 6b5f74d)_
- **07:33** [transcript[i=15159] • negative • tool-card] — tool call card fails to render summary. _"tool call with no effect and no result and no summary"_
- **07:36** [transcript[i=15176] • negative • tool-card] — tool parse errors indicate untested tooling. _"there should be no tool parse errors because we"_
- **07:42** [transcript[i=15231] • spec • tool-card] — demand for playwright end-to-end validation. _"playwright rendering of the code 1: always yield output div"_
- **07:45** [sillytavern@c275bb50e • feature] — persona-effort-schema v0.2.0 tool reads caller. _persona-effort-schema v0.2.0: tool reads persona prompt from caller_messages, zero required args_
- **07:45** [metal@49e263c • validation] — test 31 persona-effort-schema natural-elicitation. _test 31: persona-effort-schema natural-elicitation Playwright proof; bridge diagnostic safety net_
- **07:57** [sillytavern@6ec7f24b5 • cleanup] — toolcards delete seed mechanism cards live plugin. _toolcards: delete the seed mechanism; cards live one place inside the plugin_
- **07:58** [metal@44f93df • cleanup] — bootstrap update warnings toolcards plugin. _bootstrap: update warnings to reflect that toolcards now live inside the plugin_
- **08:00** [transcript[i=15393] • spec • designer] — thinking traces in client and reasoning effort. _""_
- **08:04** [sillytavern@d5c6070f9 • cleanup] — toolcards delete extended-thinking semantic overlap. _toolcards: delete extended-thinking (regex + semantic overlap with native reasoning)_
- **08:05** [sillytavern@b71c06b01 • cleanup] — toolcards plugin don't rewrite manifest boot. _toolcards plugin: don't rewrite the manifest on boot (was dirtying tracked source)_
- **08:06** [metal@633baa5 • validation] — test 32 native reasoning tool-using turn Q1. _test 32: native reasoning renders on tool-using turn (Q1 evidence + extended-thinking deletion valid_
- **08:16** [sillytavern@474b8f455 • refactor] — toolcards kill materializeCard bake/unbake. _toolcards: kill materializeCard + the entire bake/unbake indirection_
- **08:28** [sillytavern@aff88ef23 • feature] — csrf quiet 403 client auto-recovery stale token. _csrf: quiet 403 + client auto-recovery on stale token_
- **17:26** [metal@717b777 • validation] — test 33 reasoning_effort gates thinking high. _test 33: reasoning_effort gates thinking — high vs auto, one recording_
- **22:42** [sillytavern@5f108c2cc • genesis] — user-personas 4 maximally-different player personas. _user-personas: 4 maximally-different player personas for diversity testing_
- **22:43** [metal@f198082 • validation] — user-agent diversity harness first-cut report. _user-agent diversity harness + first-cut report_
- **22:57** [metal@d5c0316 • infrastructure] — user-agent harness gitignored outputs gemma-as-judge. _user-agent harness: gitignored outputs + gemma-as-judge diversity (drop sentence embeddings)_
- **23:03** [metal@92bd37c • infrastructure] — user-agent harness vectorized scheduler 2.4x. _user-agent harness: vectorized scheduler (2.4x speedup, properly saturates bridge batching)_
- **23:07** [sillytavern@c94f950a6 • genesis] — user-personas phase 1 server plugin FE suggestion. _user-personas: phase 1 — server plugin + FE extension for suggestion mode_
- **23:10** [sillytavern@e74024ff3 • feature] — user-personas FE serialize refreshPillList prevent. _user-personas FE: serialize refreshPillList to prevent race-duplication_
- **23:10** [metal@0491d5b • validation] — test 34 user-personas phase 1 suggestion mode. _test 34: user-personas phase 1 (suggestion mode) e2e proof through ST UI_

#### 2026-05-12

- **00:16** [sillytavern@a92316452 • feature] — user-personas phase 2 autonomous tick personas. _user-personas phase 2: autonomous tick — personas yap on their own_
- **00:16** [metal@6624fb7 • validation] — test 35 user-personas phase 2 autonomous-tick. _test 35: user-personas phase 2 autonomous-tick e2e proof_
- **01:21** [sillytavern@abdd6c4af • feature] — user-personas phase 3 multi-user dialogue N. _user-personas phase 3: multi-user dialogue — N personas, 1 assistant_
- **01:21** [metal@1bd94f9 • validation] — test 36 user-personas phase 3 multi-user. _test 36: user-personas phase 3 multi-user e2e proof_
- **01:34** [sillytavern@d1ba7fd6c • feature] — user-personas phase 4 unified panel chara-card. _user-personas phase 4: unified panel + chara-card-style manifest enrichment_
- **01:35** [metal@89524d4 • validation] — test 37 user-personas unified-panel e2e proof. _test 37: user-personas unified-panel e2e proof_
- **02:23** [sillytavern@0c4caaae4 • feature] — user-personas fix generation truncation immediate. _user-personas: fix generation truncation + immediate kick on autonomous toggle_
- **02:24** [metal@5f6e53b • validation] — test 38 user-personas truncation kick-on-toggle. _test 38: user-personas truncation + kick-on-toggle invariants_
- **02:31** [transcript[i=16612] • spec • chat-suggestion] — roster and composition UI affordances needed. _"what user interface affordances do we need"_
- **02:36** [transcript[i=16622] • spec • tool-card] — 2-layer collapsible tool summary UI. _"tool call divs present compressed summary above full details"_
- **02:36** [transcript[i=16623] • spec • chat-suggestion] — card-motive coupling should be composeable. _"card-motive coupling being swappable or composeable"_
- **05:24** [transcript[i=17429] • spec • user-agent] — unify user agents with user personas in ST. _"toggle is mode: manual <-> suggest <-> autonomous"_
- **18:44** [transcript[i=18525] • spec • designer] — composition interface must be CRUD-complete. _"composition interface should be CRUD-complete"_
- **19:31** [transcript[i=18663] • spec • designer] — use ST top-tab style for persistent editor. _"using the design standard for top tab entries"_

#### 2026-05-13

- **00:19** [transcript[i=19501] • negative • designer] — motive editor misdesigned and unmapped. _"drawn on right instead of dropdown; no sliders"_
- **06:11** [sillytavern@aa2f27e42 • feature] — the-rock minimal-interlocutor probe character card. _the-rock: minimal-interlocutor probe character card_
- **20:20** [sillytavern@e03b5296f • feature] — dicemother seed chats 3 canonical in-medias-res. _dicemother seed chats: 3 canonical in-medias-res openings_
- **21:03** [sillytavern@c7c6ebac1 • feature] — user-personas panel CSS layout fix viewport. _user-personas panel: CSS layout fix — claim defined viewport slice_
- **21:22** [sillytavern@6c8595452 • validation] — user-personas plugin fix three leakage bugs. _user-personas plugin: fix three compounding leakage bugs in /poll_
- **21:45** [sillytavern@b10eed6a7 • restoration] — user-personas re-add inline creation bulk-mode. _user-personas: re-add inline persona creation + bulk-mode + truncation surface_
- **22:39** [sillytavern@067ee11bd • feature] — user-personas remove plugin-side max_tokens cap. _user-personas: remove plugin-side max_tokens cap_
- **23:14** [sillytavern@e3b08039a • feature] — user-personas revert sampling distortions rep_penalty. _user-personas: revert sampling distortions (rep_penalty out, top_p=1.0)_
- **23:51** [sillytavern@4f1f7c53e • validation] — user-personas diegetic /sweep validation suite. _user-personas: diegetic /sweep validation suite + render test_

#### 2026-05-14

- **00:23** [metal@0c66635 • feature] — bridge admission backpressure sustained-load. _bridge: admission backpressure for sustained-load stability_
- **00:45** [sillytavern@33e8039a • validation] — user-personas prefix-cache priming engine stats. _user-personas: prefix-cache priming + engine stats + opt-in shared-prefix_
- **00:45** [sillytavern@e716d638b • validation] — user-personas shared_prefix slower collapses. _user-personas: shared_prefix is slower AND collapses variety (don't use)_
- **04:34** [sillytavern@99b2b8766 • infrastructure] — user-personas on-policy judge summarizer svg-drawer. _user-personas: on-policy judge / summarizer / svg-drawer harnesses_

### Period: 2026-05-15 → 2026-05-18

**Heavy Design Discourse.** Agent-card is the centerpiece (30 spec discussions). Persona/bio designer regresses repeatedly under bidirectional-edit pressure; fixed-point overlay stabilizes via bounded iteration.


#### 2026-05-15

- **01:42** [transcript[i=27808] • spec • discovery] — author-note injection for overlay management. _"user-agent elicitation as authors note preserves"_
- **18:18** [transcript[i=28302] • spec • chat-suggestion] — keystone move: convert oracles to sidebar suggester. _"throw together keystone move and see how much"_
- **18:43** [transcript[i=28478] • negative • chat-suggestion] — sidebar missing design tools and API surfaces. _"sidebar interface exists but doesnt handle design"_
- **21:17** [transcript[i=29284] • restoration • tool-call-card] — okay, so the current design pass was motivated by reaching client parity. _"okay, so the current design pass was motivated by reaching client parity"_
- **21:29** [transcript[i=29295] • spec • agent-card] — " Why I suspect it: We sample at t=0. _"" Why I suspect it: We sample at t=0"_
- **21:45** [transcript[i=29495] • spec • agent-card] — so, from here, can we start testing our user-agent-elicitation questions in terms. _"so, from here, can we start testing our user-agent-elicitation questions in t..."_
- **23:27** [transcript[i=30092] • spec • agent-card] — " The DESIGNER under clean feedback points at specific tokens it emitted. _""   The DESIGNER under clean feedback points at specific tokens it emitted in..."_
- **23:57** [transcript[i=30318] • spec • agent-card] — [Image #2] now we have some cleanup to do in terms of. _"[Image #2] now we have some cleanup to do in terms of removing the huge bank ..."_

#### 2026-05-16

- **02:49** [transcript[i=30943] • spec • agent-card] — lets relax the idea of a fully automatic deletion linter and settle. _"lets relax the idea of a fully automatic deletion linter and settle for 'lint..."_
- **03:59** [transcript[i=30953] • spec • agent-card] — the goal here is not to re-describe the existing featureset (much of. _"the goal here is not to re-describe the existing featureset (much of what you..."_
- **04:11** [transcript[i=31018] • negative • drawer] — This session is being continued from a previous conversation that ran out. _"This session is being continued from a previous conversation that ran out of ..."_
- **04:31** [transcript[i=31484] • positive • agent-card] — sure, lets see if spawn in flow gets user agents whose elicited. _"sure, lets see if spawn in flow gets user agents whose elicited behavior has ..."_
- **05:06** [transcript[i=31682] • spec • agent-card] — high variance within a user agent is actually okay as long as. _"high variance within a user agent is actually okay as long as the different a..."_
- **05:41** [transcript[i=31895] • spec • agent-card] — "references specific JS APIs (requestAnimationFrame, Canvas). _""references specific JS APIs   (requestAnimationFrame, Canvas)"_
- **05:47** [transcript[i=31919] • spec • agent-card] — " A. _""  A"_

#### 2026-05-17

- **02:20** [transcript[i=32511] • spec • agent-card] — okay! some things to think about for now: " The judge is. _"okay! some things to think about for now: "  The judge is half-blind in a spe..."_
- **03:57** [transcript[i=33339] • spec • suggestion-panel] — <task-notification> <task-id>a263928caa5bef43b</task-id> <tool-use-id>toolu_01GxXciBg4ivBFLij8So7TkW</tool-use-id> <output-file>/private/tmp/claude-501/-Users-mdot-metal-microbench/247a1b45-62c9-4cfe-a738-bb129a1145bd/tasks/a263928caa5bef43b. _"<task-notification> <task-id>a263928caa5bef43b</task-id> <tool-use-id>toolu_0..."_
- **04:01** [transcript[i=33362] • negative • agent-card] — This session is being continued from a previous conversation that ran out. _"This session is being continued from a previous conversation that ran out of ..."_
- **04:22** [metal@f8ac83a • infrastructure] — user-agent harness prefix-maxx prompt ordering. _user-agent harness: prefix-maxx prompt ordering + merged judge in drift_compare_
- **04:22** [sillytavern@5c5ea29b4 • feature] — user-personas prefix-maxx prompt ordering lint. _user-personas: prefix-maxx prompt ordering + lint phase 4_
- **04:25** [sillytavern@7a887ffbe • feature] — user-personas bio-v2 agent-v1 migration agents. _user-personas: bio-v2 / agent-v1 migration + agents/ inventory + UI refit_
- **04:26** [metal@3057a3c • infrastructure] — working-tree snapshot user-agent harness corpus. _working-tree snapshot: user-agent harness corpus + quant search + docs + bridge updates_
- **05:46** [transcript[i=34147] • positive • tool-call-card] — "no progress signal" is nothing being written to output files or intermediates. _""no progress signal" is nothing being written to output files or intermediate..."_
- **05:54** [transcript[i=34189] • negative • agent-card] — no we have several things that would work well as probes but. _"no we have several things that would work well as probes but right now you're..."_
- **06:08** [transcript[i=34210] • negative • agent-card] — " - the-rock — deflective, gravelly. _"" - the-rock — deflective, gravelly"_
- **06:16** [transcript[i=34257] • spec • agent-card] — delete wry skeptic to prove you understand what we're talking about. _"delete wry skeptic to prove you understand what we're talking about"_
- **06:38** [transcript[i=34348] • negative • designer] — ""drive any arbitrary persona on any arbitrary complementary system prompt"," 'taboo' the. _"""drive any arbitrary persona on any arbitrary   complementary system prompt"..."_
- **06:44** [transcript[i=34356] • spec • designer] — "the operator should be able to write a new agent for an. _""the operator should be able to write a new agent for an existing biography, ..."_
- **07:09** [transcript[i=34373] • spec • designer] — i'd like you to take a position of greater autonomy over launching. _"i'd like you to take a position of greater autonomy over launching experiment..."_
- **07:22** [transcript[i=34455] • spec • agent-card] — remember the factorization goal described earlier, and how we specifically wanted user. _"remember the factorization goal described earlier, and how we specifically wa..."_
- **08:04** [transcript[i=34503] • positive • agent-card] — " - You provide bios. _"" - You provide bios"_
- **08:18** [transcript[i=34512] • restoration • agent-card] — " - A distance metric in some feature space + a max-min-distance. _""  - A distance metric in some feature space + a max-min-distance objective"_
- **08:45** [transcript[i=34609] • spec • agent-card] — remember that the design of the user agent model was based upon. _"remember that the design of the user agent model was based upon generating tr..."_
- **08:47** [transcript[i=34618] • spec • agent-card] — sure go 4 it. _"sure go 4 it"_
- **09:20** [transcript[i=34717] • spec • agent-card] — why are weven talking about approaches other than finishing the complete port. _"why are weven talking about approaches other than finishing the complete port..."_
- **09:53** [transcript[i=34771] • positive • agent-card] — how was feature dimension expansion scoped in the user agent split to. _"how was feature dimension expansion scoped in the user agent split to handle ..."_
- **09:56** [transcript[i=34788] • spec • agent-card] — why are there any unported features frm the user agent side? i. _"why are there any unported features frm the user agent side? i have specifica..."_
- **10:15** [transcript[i=35014] • negative • suggestion-panel] — This session is being continued from a previous conversation that ran out. _"This session is being continued from a previous conversation that ran out of ..."_
- **16:18** [transcript[i=35100] • spec • agent-card] — claude. _"claude"_
- **16:38** [transcript[i=35155] • spec • agent-card] — why is there any talk of 'bootstrapping from four real canonical user. _"why is there any talk of 'bootstrapping from four real canonical user personas'"_
- **16:54** [transcript[i=35163] • negative • designer] — " 3. _"" 3"_
- **18:40** [transcript[i=35392] • spec • agent-card] — we don't really need to have exactly 1 bio per clusterable clique. _"we don't really need to have exactly 1 bio per clusterable clique (we can act..."_
- **19:20** [transcript[i=35539] • spec • tool-call-card] — "because the system prompt differs. _""because the system prompt differs"_
- **20:33** [transcript[i=35881] • spec • agent-card] — what is the trope density axis and where did it actualyl come. _"what is the trope density axis and where did it actualyl come from in dialogu..."_
- **20:37** [transcript[i=35889] • spec • agent-card] — "That's a one-line addition to LIKERT_AXES in axes. _""That's a one-line   addition to LIKERT_AXES in axes"_
- **20:44** [transcript[i=35916] • spec • suggestion-panel] — if these things belong in the client what refactoring, moduleification, deduplication of. _"if these things belong in the client what refactoring, moduleification, dedup..."_
- **21:08** [transcript[i=36141] • spec • agent-card] — the current problen now is that the client interface for using user-personas. _"the current problen now is that the client interface for using user-personas ..."_
- **21:31** [transcript[i=36156] • spec • agent-card] — "The factorization you specified — bio × agent — is not present. _""The factorization you specified — bio × agent — is not present in the    cha..."_
- **21:43** [transcript[i=36166] • spec • agent-card] — almost on board with oyu here. _"almost on board with oyu here"_
- **21:55** [transcript[i=36190] • spec • agent-card] — how do we handle selecting user bio & user agent combos from. _"how do we handle selecting user bio & user agent combos from intractably larg..."_
- **21:58** [transcript[i=36199] • spec • agent-card] — "places it in one of ≈20 behavioral clusters" hmmm no i do'nt. _""places it in one of ≈20 behavioral clusters" hmmm no i do'nt think we cna re..."_
- **22:00** [transcript[i=36207] • spec • agent-card] — did clusters ever exist for the user agent design and testing harness. _"did clusters ever exist for the user agent design and testing harness or can ..."_
- **22:02** [transcript[i=36247] • spec • agent-card] — so what is the thing that the k-selecting agent actually does to. _"so what is the thing that the k-selecting agent actually does to retrieve car..."_
- **23:19** [transcript[i=36649] • positive • agent-card] — [Image #5] point 1: mysterious red box with nothing in it and. _"[Image #5] point 1: mysterious red box with nothing in it and a 'x' button do..."_
- **23:29** [transcript[i=36748] • negative • drawer] — This session is being continued from a previous conversation that ran out. _"This session is being continued from a previous conversation that ran out of ..."_
- **23:41** [transcript[i=37071] • restoration • designer] — # /loop — schedule a recurring or self-paced prompt Parse the input. _"# /loop — schedule a recurring or self-paced prompt  Parse the input below in..."_
- **23:57** [sillytavern@445ccdd69 • refactor] — user-personas delete designer.html chat-rendered. _user-personas: delete designer.html + chat-rendered designers replace it_
- **23:57** [metal@4132df2 • validation] — st-debug tests replace test 73 designer.html. _st-debug tests: replace test 73 (designer.html coverage) with test 74 (chat-rendered designer)_

### Period: 2026-05-18 → 2026-05-21

**Regression Cycle.** Operator names features stripped from accepted-to-spec work: k_1/k_2 polling deleted, 22 axes appear vs 4-axis spec, synthesized bios invisible in ST persona UI. Demands archeology.


#### 2026-05-18

- **01:06** [transcript[i=37234] • negative • agent-card] — some really basic issues reviewing the interface rn: 1: [Image #6] we. _"some really basic issues reviewing the interface rn: 1: [Image #6] we see tha..."_
- **01:10** [sillytavern@317ebf5cd • feature] — user-personas agentless bios broken state. _user-personas: agentless bios are broken state + plays-nicely triplet_
- **01:10** [metal@0c22aed • infrastructure] — axes.py plays-nicely-with-others triplet ludic. _axes.py: plays-nicely-with-others triplet (ludic / multipolarity / awareness)_
- **01:13** [transcript[i=37334] • negative • designer] — "as │ │ ignores rock │ broken state with inline "design agent". _""as    │   │ ignores rock          │ broken state with inline "design agent" ..."_
- **01:15** [sillytavern@cbc5d45e6 • feature] — user-personas selection IS design auto-chain. _user-personas: selection IS design — auto-chain bio→agent + agentless-bio redirect_
- **01:41** [transcript[i=37662] • spec • designer] — " 6 + "description": "You are the Agent Designer for the user-personas. _""      6 +        "description": "You are the Agent Designer for the user-per..."_
- **02:08** [metal@2c4e7a2 • infrastructure] — st-debug loadAndConnect waits online_status Valid. _st-debug: loadAndConnect waits for online_status='Valid' (was racing)_
- **02:08** [sillytavern@4019f9aa1b6 • feature] — user-personas agents persisted as PNG cards. _user-personas: agents persisted as PNG cards (chara_card_v3)_
- **02:08** [sillytavern@64a267dec • refactor] — user-personas drop source_card_id middleman FK. _user-personas: drop source_card_id middleman; harness sets agent FK_
- **02:11** [transcript[i=37944] • negative • agent-card] — okay so does the problem you were trying to address. _"okay so does the problem you were trying to address"_
- **02:15** [transcript[i=37953] • spec • suggestion-panel] — " Capability: Off-manifold discovery pipeline" okay so it's simply time to implement. _""  Capability: Off-manifold discovery pipeline" okay so it's simply time to i..."_
- **02:19** [transcript[i=37996] • spec • tool-call-card] — why are you writing this as a character card or something instead. _"why are you writing this as a character card or something instead of literall..."_
- **02:21** [sillytavern@90ef67a38 • feature] — user-personas /signature-extract endpoint batch-parallel. _user-personas: /signature-extract endpoint (batch-parallel judges, prefix-shared)_
- **02:35** [sillytavern@4382a0f2d • feature] — user-personas agents carry composition signatures. _user-personas: agents carry composition signatures + backfill script_
- **02:35** [transcript[i=38118] • positive • suggestion-panel] — what is yapper seed and why do we need it. _"what is yapper seed and why do we need it"_
- **03:10** [transcript[i=38313] • negative • agent-card] — well can we get rid of every single interface whic his not. _"well can we get rid of every single interface whic his not compliant with the..."_
- **03:12** [transcript[i=38328] • negative • designer] — " 2. _""  2"_
- **03:15** [transcript[i=38352] • negative • designer] — <task-notification> <task-id>aec396d00013d3144</task-id> <tool-use-id>toolu_01MQyL7nq6k1QMYerf7q3246</tool-use-id> <output-file>/private/tmp/claude-501/-Users-mdot-metal-microbench/247a1b45-62c9-4cfe-a738-bb129a1145bd/tasks/aec396d00013d3144. _"<task-notification> <task-id>aec396d00013d3144</task-id> <tool-use-id>toolu_0..."_
- **03:21** [sillytavern@891e968c0 • regression] — user-personas revert chat-rendered designer restore. _user-personas: revert chat-rendered designer characters; restore designer.html mechanism_
- **03:21** [metal@7a826b7 • regression] — st-debug tests revert test 74 chat-rendered restore. _st-debug tests: revert test 74 (chat-rendered designers) + restore test 73 (designer.html)_
- **03:22** [transcript[i=38456] • negative • agent-card] — what are we *doing* right now. _"what are we *doing* right now"_
- **03:27** [transcript[i=38463] • negative • designer] — " was the answer; that part landed correctly and is still in. _""  was the answer; that part landed correctly and is still in axes"_
- **03:30** [transcript[i=38471] • spec • agent-card] — can we figure out what tests were being run instead of the. _"can we figure out what tests were being run instead of the actual validation ..."_
- **03:37** [transcript[i=38510] • spec • agent-card] — " - Bio authoring: ST's mainline user-persona UI. _""  - Bio authoring: ST's mainline user-persona UI"_
- **03:38** [transcript[i=38519] • spec • designer] — " Because deletion is easy and composition is hard. _""  Because deletion is easy and composition is hard"_
- **03:43** [transcript[i=38527] • restoration • designer] — "Iterates on synthetic single turns against an imagined counterparty within the design. _""Iterates on synthetic single turns against an imagined   counterparty within..."_
- **03:50** [transcript[i=38542] • restoration • fixed-point] — if it's easier to understand this as linear algebra can we specify. _"if it's easier to understand this as linear algebra can we specify it in term..."_
- **03:59** [transcript[i=38649] • restoration • designer] — " - Composition in chat: the rogue actually tries to steal the. _""  - Composition in chat: the rogue actually tries to steal the rock, then tr..."_
- **04:00** [transcript[i=38657] • restoration • fixed-point] — can we expand to only this exact scope (demonstrate we're getting the. _"can we expand to only this exact scope (demonstrate we're getting the user bi..."_
- **04:08** [transcript[i=38779] • negative • designer] — This session is being continued from a previous conversation that ran out. _"This session is being continued from a previous conversation that ran out of ..."_
- **04:21** [transcript[i=38828] • restoration • designer] — huh, holding for direction? " Fixed-point evidence per bio (data/lock_in_iterative/): rpg-wizard-sagittarius. _"huh, holding for direction? "  Fixed-point evidence per bio (data/lock_in_ite..."_
- **04:40** [metal@99584a5 • cleanup] — docs feature_factorization_design.md linal. _docs: feature_factorization_design.md (linal + pseudohaskell + ball-and-stick)_
- **04:46** [metal@67e07bf • infrastructure] — harness axis registry axis splitter velocity. _harness: axis registry + axis splitter + velocity-stall + first split run_
- **05:13** [transcript[i=39030] • restoration • fixed-point] — every bio should always have an agent elicitation string, even if it's. _"every bio should always have an agent elicitation string, even if it's a bad ..."_
- **05:15** [metal@671746c • cleanup] — docs cluster disambiguator spec item 6. _docs: cluster disambiguator spec (item 6) + design clarifications_
- **05:32** [metal@a665cfa • infrastructure] — harness cluster disambiguator item 6 demo. _harness: cluster disambiguator (item 6) + first two demo runs_
- **05:56** [metal@8eaba6c • infrastructure] — harness outer-outer eff-dim objective item 3. _harness: outer-outer w/ eff-dim objective (item 3) + harness_lib_
- **06:09** [metal@275a22d • validation] — harness judge prompt A/B V6 promotion run3. _harness: judge prompt A/B + V6 promotion + run3 with strict judge_
- **06:43** [transcript[i=33] • spec • user-agent-drawer] — k_1/k_2 agent suggestions, finite picker. _"k_1 user agents as candidates...k_2 user agents as suggested-but-disabled"_
- **06:48** [metal@4dece95 • feature] — harness context-suggester Phase A k-active+k-disabled. _harness: context-suggester Phase A — k-active+k-disabled persona picks_
- **06:51** [transcript[i=34] • spec • feature-vector-suggester] — No explicit scalar feature list; axes inferred dynamically. _"the suggester could look at the first turn chat context...pick something very vi"_
- **06:52** [transcript[i=35] • regression • user-agent-drawer] — Full-list fallback deleted per design. _"no, we won't ever be wanting this ever at any point. ever. k-many bios...it's a "_
- **06:55** [metal@05a5fe4 • feature] — harness context_suggester uniform-prior Norm. _harness: context_suggester uniform-prior + doc on Norm dominance_
- **07:02** [sillytavern@45dcc69bc31 • feature] — yapper-seed yeet signedness gate oracle-on-demand. _yapper-seed: yeet signedness gate — oracle-on-demand, no type-validation_
- **07:04** [transcript[i=39] • regression • drawer] — List ontology change: finite selected vs full hidden. _"you are *making a different ontological list* of *only things that were selected"_
- **07:07** [transcript[i=40] • spec • drawer] — k_1+k_2 defaults; never empty, never overstuffed. _"filled by k_1+k_2 many selections by default...we are asking for the synthesis"_
- **07:17** [sillytavern@439cb0be1 • feature] — user-personas sidebar loadout ontology not filter. _user-personas sidebar: loadout ontology, not filter_
- **07:19** [transcript[i=42] • regression • yapper-seed-cold-path] — Post-hoc feature extraction forbidden. _"this is not appropriate...we ALREADY EXTRACTED FEATURE VECTORS FOR EVERY SINGLE "_
- **07:23** [transcript[i=43] • spec • feature-vector-persistence] — Vectors must persist; no runtime recomputation. _"things that should be persisted not being persisted...which is the highest prior"_
- **07:24** [transcript[i=45] • regression • bio-signature] — Skipping vs running harness for unsigned bios. _"didn't the user JUST say that selectively running the exact llm as judge harness"_
- **07:26** [transcript[i=47] • regression • persistence-layer] — settings.json race conditions; expand cards interface. _"why are you using settings.json instead of something like expanding the cards in"_
- **07:34** [transcript[i=50] • regression • bio-creation] — Bios born without signatures; must fix at creation. _"the bios are born wrong by construction and we absolutely must fix this at highe"_
- **07:37** [transcript[i=52] • regression • api-duplication] — Two APIs doing same thing; delete one and callers. _"delete one of them and delete callers which depend on it until the environment i"_
- **07:42** [transcript[i=53] • regression • dead-code] — Deleted /compare /designer /sweep endpoints completely. _"deletecompare delete designer drawer tabs delete run sweep...comprehensively del"_
- **08:10** [sillytavern@74e6bbb9d • cleanup] — user-personas aggressive purge canonical-18-axis. _user-personas: aggressive purge of canonical-18-axis dead code_
- **08:10** [metal@1785b6e • feature] — harness born-signed bios sign_unsigned legacy. _harness: born-signed bios + sign_unsigned for legacy; delete obsolete tests_

#### 2026-05-19

- **04:06** [transcript[i=56] • spec • bio-synthesis] — Legit synthesis path for user-added bios. _"there is an extremely simple ontological closure here...agents have no source wh"_
- **04:48** [sillytavern@097da92ef • feature] — user-personas stop writing settings.json cards only. _user-personas: stop writing to settings.json — cards are the only store_
- **04:51** [sillytavern@ab29a2ab2 • cleanup] — user-personas delete legacy settings.json read. _user-personas: delete legacy settings.json read entirely_
- **06:10** [metal@87254e8 • feature] — fix bridge stream-lifecycle safety canonical. _fix: bridge stream-lifecycle safety + canonical partial-resume path_
- **06:20** [sillytavern@e4915ee86 • cleanup] — lint recursive walk LINT-OK-MAX-TOKENS escape. _lint: recursive walk + LINT-OK-MAX-TOKENS escape_
- **06:20** [metal@c71c9f3 • feature] — harness drop max_tokens caps bridgeCall call-sites. _harness: drop max_tokens caps from bridgeCall call-sites in harness_lib_
- **06:31** [metal@ec1dbe4 • cleanup] — harness delete retired experiments truncation. _harness: delete retired experiments + truncation A/B + scringlo scripts_
- **06:31** [sillytavern@9750f29e0 • cleanup] — lint remove max_tokens escape hatch docstring. _lint: remove max_tokens escape hatch + docstring/string-literal awareness_
- **06:42** [metal@7658a88 • regression] — Revert harness delete retired experiments. _Revert "harness: delete retired experiments + truncation A/B + scringlo scripts"_
- **07:02** [metal@49ccc8f • cleanup] — harness dedup shared helpers single source truth. _harness: dedup shared helpers — single source of truth in harness_lib_
- **07:12** [transcript[i=96] • spec • suggester-ranking] — LLM-driven K-nearest-neighbor agent selection. _"automatic suggestion of k-most-relevant user agents from a llm-as-judge query"_
- **07:15** [transcript[i=97] • regression • rubric-storage] — Implicit rubrics; should be persisted on axis registry. _"why is there still random data corruption nonsense in here...placeholder rubrics"_
- **07:17** [transcript[i=98] • regression • feature-dimensions] — Implicit axis definitions; should be dedup'd and explicit. _"if we can literally make arbitrary data stores...why are feature dimension defin"_
- **07:30** [sillytavern@9ced4d081 • feature] — user-personas axes as cards missing storage. _user-personas: axes as cards — the missing storage paradigm_
- **07:30** [metal@0b472c3 • infrastructure] — harness read axes from plugin /axes delete. _harness: read axes from plugin /axes; delete axis_registry.mjs_
- **07:42** [sillytavern@789079f82 • feature] — user-personas experiments as cards same paradigm. _user-personas: experiments as cards — same paradigm as axes_
- **07:42** [metal@cdfa02b • infrastructure] — harness lock_in_iterative loads experiment spec. _harness: lock_in_iterative loads experiment spec from card, drops caps_
- **07:57** [sillytavern@101b1d491 • feature] — user-personas migrate ST persona_descriptions player. _user-personas: migrate ST persona_descriptions to player cards at boot_
- **08:01** [sillytavern@7e4296ca7 • cleanup] — user-personas delete settings.json migration anti-canonical. _user-personas: delete settings.json migration; it was anti-canonical_
- **08:03** [sillytavern@0ea824617 • regression] — Revert user-personas delete settings.json migration. _Revert "user-personas: delete settings.json migration; it was anti-canonical"_
- **08:15** [transcript[i=113] • spec • chat-suggestion-interface] — Central missing data flow: synthesis->cards->suggester->user. _"it's the chat user agent suggestion interface which surfaces user agents based o"_
- **08:17** [transcript[i=114] • spec • suggester-paging] — +More button for K-paging through ranked results. _"a button to append another k1_k2 items by searching again was emphatically descr"_
- **08:27** [sillytavern@497f4416d • feature] — user-personas fixed-point iteration tab synthesis. _user-personas: fixed-point iteration tab (synthesis-loop UI)_
- **08:27** [transcript[i=116] • positive • fixed-point-tab] — Fixed-point iteration UI delivered; endpoints wired. _"installFixedPointDrawer()...three sections...Experiments list, Run progress...Va"_
- **08:36** [sillytavern@844ee3cdf • cleanup] — user-personas PERSONA_API.md fork persona contract. _user-personas: PERSONA_API.md — fork's persona contract vs upstream_
- **08:47** [metal@63d4291 • validation] — st-debug smoke spec Fixed-Point Iteration tab. _st-debug: smoke spec for the Fixed-Point Iteration tab_
- **09:02** [transcript[i=125] • positive • experiment-editor] — Structured editor for experiment specs; axis multi-selects. _"axis multi-selects driving materialized 1-5 steppers per row...repeating bios/ag"_
- **09:20** [transcript[i=126] • positive • context-driven-suggester] — Suggester ranked by similarity; K-paging support. _"context-driven /yapper-seed flow...+More K-paging with corpus-ceiling disable"_
- **17:23** [transcript[i=137] • spec • harness-resilience] — K-shot iteration tolerance for stochastic API failures. _"how should any part of our client programming...behave under this sort of error?"_
- **18:23** [transcript[i=152] • spec • output-format-tolerance] — YAML/JSON/TOML interchangeable for harness robustness. _"can we refactor all json out schema code...to be retemplateable to use any of {y"_
- **19:33** [transcript[i=170] • spec • seed-contrast-spec] — Client interface for feature-contrast seed input. _"Make N bios that contrast pairwise on (these axes) within (this thematic frame)"_
- **21:33** [transcript[i=179] • positive • seed-textarea] — Verbatim seed textarea + materialization path. _"contrast-spec textarea...can ship now, accepting verbatim seeds...directly to th"_
- **21:45** [transcript[i=180] • positive • iteration-timeline] — Full trajectory view of outer/inner loops + convergence. _"per-bio header...max_off sparkline...outer accordions...agent_text expand toggle"_
- **22:40** [transcript[i=184] • positive • axis-registry-view] — Tree view + add/edit/delete axes; orphan detection. _"full V1 axis-registry UI...tree with ASCII prefixes...per-row id/kind/scale/def."_
- **22:43** [transcript[i=183] • positive • corpus-dashboard] — Eff-dim PR visualization; per-axis variance bars. _"PR=12.68...summary tiles, per-axis variance bar chart...saturation history, refr"_

#### 2026-05-20

- **01:31** [transcript[i=189] • positive • provenance-tagging] — Provenance metadata + suggester filter on card kind. _"tag_existing_corpus.mjs...classifies (filename *-iter<N> experiment_output)...fi"_
- **01:50** [transcript[i=192] • regression • axis-splitter] — Feature splitting not invoked in GUI client path. _"the feature splitting was specified as an implicit and mandatory feature...if it"_
- **02:31** [transcript[i=207] • positive • outer-outer-run] — Multi-pass synthesis with entanglement detection + derivation. _"ask for 2 agents...ask for 2 more after the first warp...eventually end up with "_
- **03:27** [transcript[i=223] • regression • feature-affordances] — Feature discovery not rendered in client UI. _"new feature dimension and the effect it has on the effective dimensionality...ca"_
- **04:39** [transcript[i=230] • regression • suggester-polling] — Per-agent polling removed; breaks k_1 feature spec. _"the yapper suggestion is supposed to take a fixed time...choose agents who start"_
- **04:51** [transcript[i=233] • regression • cache-pattern] — Abort-on-nav DoS pattern forbidden; use persisted cache. _"don't use any design patterns which would constitute a DoS pattern if you applie"_
- **05:56** [transcript[i=234] • regression • ui-eviction] — New drawers evict character/person affordances; CSS broken. _"current css literally evicts the person and character interface affordances...ma"_
- **05:56** [transcript[i=234] • regression • persona-integration] — Synthesized bios not visible in ST native persona UI. _"none of the personas made by the synthesis tool are accessible as personas in th"_
- **17:22** [transcript[i=245] • regression • interactive-features] — Resynth, suggest, poll affordances don't work. _"clicking, triggering, interacting with the resynth interfaces does nothing"_
- **17:22** [transcript[i=245] • regression • axis-count] — 22 axes vs spec 4; no factorization evidence. _"there's like 20 feature dims. we only specified like maybe 4 feature dims explic"_
- **17:22** [transcript[i=245] • regression • end-to-end-testing] — Tests written but validation not performed. _"harnesses have been written but tests and validation have not been performed, no"_
- **17:26** [transcript[i=246] • spec • axis-reset] — Delete all 22 axes; regenerate minimal 4-axis set. _"deleting all of the axes (yes, all of them) and regenerating the precollapsed ti"_

#### 2026-05-21

- **21:06** [transcript[i=269] • regression • json-form-builder] — Empty form asking for manual JSON field entry forbidden. _"there is a diegetic 'add user agent: please fill out a json's fields with a form"_
- **21:06** [transcript[i=269] • regression • suggester-first-paint] — No k_1/k_2 streaming suggestions on startup. _"there are no k_1 and k_2 high affinity user agents for talking to 'the rock' wit"_
- **21:06** [transcript[i=269] • regression • factorization-affordance] — No UI to demonstrate axis split or choose synthetic bio. _"the interface doesn't *demonstrate* feature dimension splitting...without these "_


---

## How to update this archeology

When the operator next reports "you forgot a feature" or "you regressed accepted work",
add a new period narrative + appended chronological events. The archeology is the
source of truth for what was previously agreed; consult it before designing.

This document was assembled by five parallel haiku-tier subagents on
2026-05-21T21:26:54Z, scanning the full Claude-Code session transcript and both repos'
git logs. Methodology: extract user-text messages only (1094 turns), chunk
into 4 time-ordered slices (~25 MB each in raw transcript bytes), scan
each for interface-related mentions with structured-JSON output, then
merge against `git log --author=zyzzyva` for sillytavern-fork (57 commits)
and full `git log` for metal-microbench (210 commits).
