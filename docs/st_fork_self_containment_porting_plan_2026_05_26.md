# ST-Fork Self-Containment Porting Plan

Date: 2026-05-26

## Mandate

`sillytavern-fork` is the canonical owner for user-personas client code,
server-plugin code, plugin metadata, plugin configuration, generated plugin
artifacts, and plugin-owned harness entrypoints. The plugin must not infer
inputs or outputs from a developer workstation path, from the
`metal-microbench` checkout, or from default localhost ports.

`metal-microbench` may provision external services for validation, such as the
bridge process and the four-clone ST matrix, but the behavior under test must
come from cloned `sillytavern-fork` source and ST configuration.

## Coverage Groups

### 1. Runtime Path Ownership

Acceptance criteria:

- `plugins/user-personas/index.mjs` resolves spawned harness scripts from
  `plugins/user-personas/harness/`.
- Plugin output defaults resolve under `plugins/user-personas/data/`.
- ST data defaults resolve from ST's configured data root, not from
  `tools/st-debug` or another checkout.
- `node plugins/user-personas/scripts/lint_no_host_paths.mjs` passes inside a
  cloned ST checkout.

Implemented guard:

- `tools/st-debug/tests/97_st_fork_user_personas_self_contained.spec.js`
  executes the st-fork-owned host-path lint inside a matrix clone.

### 2. Harness Portability

Acceptance criteria:

- `lock_in_iterative.mjs`, `outer_outer.mjs`, `axis_splitter.mjs`, and
  `cluster_disambiguator.mjs` exist under `plugins/user-personas/harness/`.
- Harness defaults write plugin data under the current clone, not the root
  checkout and not `metal-microbench/data`.
- Harness HTTP URLs come from the plugin-launched environment:
  `ST_URL`, `PLUGIN_URL`, and `BRIDGE_URL`.
- Matrix clones include newly added plugin harness files during validation.

Implemented guard:

- `tools/st-debug/scripts/multi_st_matrix.mjs` copies untracked plugin files
  from the root st-fork working tree into each clone before launch.

### 3. Endpoint and UI Behavior

Acceptance criteria:

- `/api/plugins/user-personas/yapper-seed` produces real top-k rows through
  the cloned plugin without Playwright route stubs.
- Browser UI never renders an empty or apologetic stalled state for top-k
  chat-agent recommendation.
- Swipe-specific recommendation state persists through browser refresh and ST
  server restart.

Implemented guards:

- `tools/st-debug/tests/96_real_yapper_seed_plugin_endpoint.spec.js`
- `tools/st-debug/tests/95_user_agent_recommendation_persistence_swipes.spec.js`
- `tools/st-debug/tests/94_user_agent_panel_never_empty_during_client_turns.spec.js`

### 4. Remaining Porting Work

Acceptance criteria:

- ST-fork owns any future Playwright helpers needed by the user-personas
  extension, or the metal harness only invokes helpers shipped in the cloned
  ST tree.
- One-shot migration and seed scripts use ST config/data roots and plugin-local
  assets only.
- Plugin startup lints run against st-fork-owned code only; they do not scan
  sibling repos to prove correctness.

Current status:

- Runtime host-path defaults are removed from st-fork user-personas plugin code.
- The fixed-point harness entrypoints are ported into st-fork.
- Matrix validation still lives in `metal-microbench/tools/st-debug` because it
  provisions bridge and multi-server infrastructure. Its next cleanup target is
  to delegate all plugin-specific fixtures/helpers to files shipped in
  `sillytavern-fork`.
