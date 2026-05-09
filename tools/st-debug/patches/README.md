# Plugin patches applied via codex pair-programming

These are SNAPSHOTS of files in the user's `~/sillytavern-fork/` install
that were modified to fix bugs found via static analysis. The originals
are untracked in the user's fork (working tree changes only); the
snapshots here let us:
  - reproduce the patches from this repo if the user resets their fork
  - track the patch contents alongside the metal-microbench tests
    that exercise them

## Files

### `toolcards-index.mjs.codex-patched`

Replacement for `~/sillytavern-fork/plugins/toolcards/index.mjs`.
Refactors the FIFO sync-block session model into per-session service
processes with a warm pool. See
`docs/toolcards_fifo_session_finding.md` for the analysis and design.

Verified by `tools/st-debug/tests/05_toolcards_captioned.spec.js`:
  - existing single-session test passes in ~8s
  - new concurrency test passes in 18ms (two same-card sessions emit
    progress events without blocking each other)

To apply to a fresh sillytavern-fork checkout:
    cp tools/st-debug/patches/toolcards-index.mjs.codex-patched \
       ~/sillytavern-fork/plugins/toolcards/index.mjs
