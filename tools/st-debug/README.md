# st-debug — claude-controllable SillyTavern integration harness

Isolated, state-free SillyTavern-fork instance for autonomous integration
testing of the bridge ↔ ST ↔ browser stack. **NEVER reads or writes the
user's main `~/sillytavern-fork` data dir** — its dataRoot lives at
`tools/st-debug/_data/` (gitignored) and is wiped on demand.

## Why this exists

The interactions between our bridge (port 8001), SillyTavern's frontend
proxy (port 8002 here), and the browser-side rendering of streamed
responses + tool-call DOM mutation form a 3-process pipeline where any
contract violation (mis-shaped tool_calls, scaffolding bleed, streaming
delta ordering, MCP plugin behavior, etc.) can ONLY be observed by
running an actual end-to-end test through the actual ST UI.

Bulk-collecting "real user sessions" was rejected because:
- Test tempo can't scale (needs human-in-the-loop)
- No LLM-driven user side for parallelism / distributional statistics
- Hides whole classes of integration regressions
- Long sequential calls (e.g., SVG-refinement workflows) have no
  affordance indicating runtime depth or ongoing contents

This harness solves all four. It's Claude-controllable: scripts run via
`uv run` / `npx playwright test`, no UI clicking required.

## Layout

```
tools/st-debug/
├── README.md             this file
├── scripts/
│   ├── bootstrap.sh      first-time setup: seed _data/ + patch settings
│   ├── run.sh            launch ST against bridge (idempotent)
│   ├── reset.sh          wipe _data/ → fresh state
│   └── api_probe.py      curl-driven probe (hits ST's HTTP backend)
├── tests/
│   ├── package.json      playwright test deps
│   ├── playwright.config.js
│   └── *.spec.js         e2e tests (browser DOM + API behavior)
└── _data/                ST's --dataRoot — gitignored, regenerable
    └── default-user/
        ├── settings.json    patched to talk to bridge :8001
        ├── chats/           wiped at reset
        └── ...
```

## Workflow

```bash
# One-time bootstrap: seed _data/, patch settings.json, leaves ST stopped.
./scripts/bootstrap.sh

# Launch ST against our bridge (assumes bridge on :8001 is up).
./scripts/run.sh
# UI: http://127.0.0.1:8002

# Run the e2e tests (browser-side + API-side combined).
cd tests && npm install && npx playwright test

# Reset state between scenarios.
./scripts/reset.sh
```

## Ports

| Service            | Port | Notes                              |
|--------------------|------|------------------------------------|
| Our bridge         | 8001 | (canonical, from server/config.toml) |
| ST debug instance  | 8002 | `--port 8002` to ST                |
| ST main install    | 8000 | (untouched by this harness)        |

## Phase-1 scope (today)

- Working state-free ST instance pointing at our bridge ✓
- One e2e test: send a message, assert response renders in DOM ✓
- One API probe: hit ST's `/api/backends/chat-completions/generate`, capture
  the request that gets forwarded to the bridge, validate shape ✓

## Phase-2 scope (next session)

- Tool registration: spin up MCP plugins for `query-to-svg__generate`
  etc. so the test exercises tool calls (the SVG-from-query workflow).
- Distributional statistics: a Gemma-as-user driver that fires N varied
  prompts in parallel, collects success/failure rates per response shape.
- Long-sequential-call telemetry: surface "this stream has been
  generating for 90s and has emitted 2 tool calls so far" affordances.
