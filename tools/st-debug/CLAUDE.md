# st-debug — agent-oriented operating notes

This directory hosts an **isolated** SillyTavern instance used for
automated playwright + curl integration testing of the bridge ↔ ST ↔
browser pipeline. The isolation is real: st-debug has its own clone of
sillytavern-fork at `tools/st-debug/sillytavern-fork/` (gitignored). It
does **not** share source files with the user's main checkout at
`/Users/mdot/sillytavern-fork`.

---

## Layout

```
tools/st-debug/
├── CLAUDE.md             this file
├── README.md             human-oriented overview
├── .gitignore            excludes _data/ + sillytavern-fork/
├── scripts/
│   ├── bootstrap.sh      one-time _data/ seed + settings.json patch
│   ├── run.sh            launch ST on port 8002 against the bridge
│   ├── reset.sh          wipe _data/ → fresh state
│   └── api_probe.py      curl-driven probe
├── tests/
│   ├── package.json      playwright test deps
│   ├── playwright.config.js
│   └── *.spec.js         e2e tests
├── _data/                ST --dataRoot (gitignored, regenerable)
└── sillytavern-fork/     ← own clone (gitignored). origin = local root
```

`sillytavern-fork/` is a **separate git repo**. Its `origin` points at
`/Users/mdot/sillytavern-fork` (local path, no network needed).

---

## Why the source is not shared

Earlier setup: `run.sh` did `cd /Users/mdot/sillytavern-fork && node
server.js --dataRoot _data/`. That meant both root sillytavern-fork
(if it were ever launched directly) and st-debug ran the **same**
plugin code from the same physical path. The plugin's `PLUGIN_DIR`
resolved identically in both instances, so any plugin code that derived
"where do I write?" from filesystem geometry silently wrote to root's
data dir regardless of which instance loaded it. Discovered 2026-05-19
when persona-mirror writes landed in root's `settings.json` instead of
st-debug's `_data/default-user/settings.json`.

The fix: st-debug owns its own clone. The plugin still uses
`process.argv` parsing of `--dataRoot` to find the right settings.json
(belt-and-suspenders), but the source-isolation itself is structural
now, not heuristic.

---

## Sync workflow (the canonical loop)

When you edit a plugin / FE / server file in root sillytavern-fork and
want st-debug to see it:

```bash
# 1. Commit the change in root.
cd /Users/mdot/sillytavern-fork
git status                  # confirm what you're committing
git add plugins/user-personas/index.mjs  # specific files, not -A
git commit -m "<msg>"

# 2. Pull into the st-debug clone.
cd /Users/mdot/metal-microbench/tools/st-debug/sillytavern-fork
git pull                    # pulls from local /Users/mdot/sillytavern-fork

# 3. Restart st-debug to load the new code (ST doesn't hot-reload plugins).
cd /Users/mdot/metal-microbench/tools/st-debug
pkill -f 'node server.js.*--port 8002' || true
sleep 1
./scripts/run.sh --bg

# 4. (optional) Confirm st-debug is back up.
curl -fsS http://127.0.0.1:8002/ > /dev/null && echo "up"
```

**Do NOT** edit files inside `tools/st-debug/sillytavern-fork/`
directly. Edit in root → commit → pull. The clone is a read-only-ish
mirror; ad-hoc edits there will be silently overwritten on next pull
(or worse, create a divergent state that's confusing to debug).

If you've already edited in the clone by accident, port the change to
root and follow the workflow:

```bash
# Diff what diverged in the clone:
cd /Users/mdot/metal-microbench/tools/st-debug/sillytavern-fork
git diff > /tmp/clone-wip.patch

# Reset the clone to clean state:
git checkout .

# Apply the patch in root, commit, pull back:
cd /Users/mdot/sillytavern-fork
git apply /tmp/clone-wip.patch
# (edit further as needed, then commit + pull as above)
```

---

## Initial bootstrap (rare — only on fresh checkout)

If the clone directory doesn't exist (e.g., fresh machine, or you
deleted it):

```bash
cd /Users/mdot/metal-microbench/tools/st-debug

# 1. Clone the source (hardlinks .git/objects to save disk).
git clone /Users/mdot/sillytavern-fork ./sillytavern-fork

# 2. Sync any uncommitted WIP from root (one-time; future syncs use git pull).
rsync -a /Users/mdot/sillytavern-fork/plugins/user-personas/ \
        ./sillytavern-fork/plugins/user-personas/
rsync -a /Users/mdot/sillytavern-fork/public/scripts/extensions/user-personas/ \
        ./sillytavern-fork/public/scripts/extensions/user-personas/
rsync -a /Users/mdot/sillytavern-fork/public/index.html \
        ./sillytavern-fork/public/index.html

# 3. Copy node_modules (~318M; not in git).
cp -aR /Users/mdot/sillytavern-fork/node_modules ./sillytavern-fork/node_modules

# 4. Seed the _data/ dir if it doesn't exist.
./scripts/bootstrap.sh

# 5. Launch.
./scripts/run.sh --bg
```

---

## Running the tests

```bash
# Make sure ST is up and the bridge is up.
curl -fsS http://127.0.0.1:8001/health > /dev/null && echo "bridge ok"
curl -fsS http://127.0.0.1:8002/       > /dev/null && echo "ST ok"

# Run all specs.
cd /Users/mdot/metal-microbench/tools/st-debug/tests
npx playwright test

# Run a single spec.
npx playwright test 50_post_factorization_affordances.spec.js

# Headed (visible browser) for debugging UI issues.
npx playwright test 50_post_factorization_affordances.spec.js --headed
```

---

## Critical invariants

1. **Never** `cd /Users/mdot/sillytavern-fork && node server.js
   --dataRoot _data/`. That re-introduces the shared-source bug.
   Always use `./scripts/run.sh` (which `cd`s into the clone).
2. **Never** commit `sillytavern-fork/` into metal-microbench. It's
   gitignored; keep it that way.
3. **Never** write to `/Users/mdot/sillytavern-fork/data/` from st-debug
   code paths. If you see this happen, it means the plugin's
   `_stSettingsJsonPath()` / equivalent path-resolver lost track of
   `--dataRoot`. Fix the resolver, don't paper over it.
4. **Killing the ST process**: `pkill -f 'node server.js.*--port 8002'`
   targets only the st-debug instance. Don't `pkill node`.

---

## Ports

| Service           | Port |
|-------------------|------|
| Our bridge        | 8001 |
| st-debug ST       | 8002 |
| Root sillytavern  | (off by default; would conflict on 8000) |

---

## Memory references

See also (in this project's MEMORY.md):
- `st_instance_separation.md` — the original isolation rule, now
  structurally enforced by this separate clone.
- `canonical_store_for_personas.md` — settings.json is the canonical
  ST UI persona store; plugin mirrors into it on a narrow path.
- `helpers_cannot_inherit_low_level_optimizations.md` — measurement
  goes through plugin endpoints, not internal helpers.
