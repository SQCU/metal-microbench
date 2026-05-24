# AGENTS.md — metal-microbench

This file is the entry point for assistants working on this repo. Before
designing anything in the multi-user-chat-agent-suggestion space, read
the archeology — you (the assistant) have likely already designed and
shipped this feature before.

## Required reading before designing UI

[**docs/multi_user_agent_chat_interface_spec.md**](docs/multi_user_agent_chat_interface_spec.md)
— Restorative specification. Component-by-component contract + 11 load-bearing
principles + thesis-statement + paired-agent acceptance protocol. Read this
BEFORE proposing any change to the chat interface.

[**docs/project_archeology.md**](docs/project_archeology.md) — Indexed
timeline of every chat-suggestion interface version. Cross-references
commits to operator-stated specs.

[**docs/ux_debt_followup_tickets_2026_05_21.md**](docs/ux_debt_followup_tickets_2026_05_21.md)
— Followup ticket backlog and the load-bearing UX principles operators
have restated multiple times. Forbidden anti-patterns are listed there.

## Critical files

- `server/bridge.py` — FastAPI bridge to Metal engine.
- `tools/st-debug/sillytavern-fork/` — Gitignored ST clone for tests.
- `tools/st-debug/CLAUDE.md` — st-debug sync workflow.
- `tools/user-agent-harness/` — Synthesis harness (lock_in_iterative, outer_outer, axis_splitter).

## Anti-pattern reminders

- Never write to mirrors. One canonical store per concept.
- Never propose commits as a synchronization point. They are not.
- Never ship a UI with empty forms / "click here to begin" placeholders.
- Never displace the chat for navigation gestures.
