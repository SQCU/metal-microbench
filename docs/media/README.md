# Curated test artifacts

Filenames here are **descriptive, dated, and pinned to the commit
that produced them**. Don't overwrite — each curated artifact is a
record of what the harness produced at a specific commit, and is
referenced by docs and commit messages by exact filename.

## Naming convention

```
YYYY-MM-DD_<test-id>_<study-tag>_<commit-short-sha>.<ext>
```

- `YYYY-MM-DD`: capture date.
- `<test-id>`: which test produced it, e.g. `test16`, `test15`.
- `<study-tag>`: short hyphenated description of WHAT the artifact
  shows. Use specific feature/bug names — `drawer_overlay_BUG`,
  `branches_visible`, `n_of_k_summary_trace`. The tag is the way a
  reader figures out which artifact to look at without opening it.
- `<commit-short-sha>`: 7-char commit hash from
  `git rev-parse --short HEAD` AT THE TIME OF CAPTURE. Pins the
  artifact to the source state that produced it.

## Why

Earlier I was overwriting `vertical_slice_2026-05-09.webm` across
three distinct captures: the original (drawer-overlay bug, useless),
the post-drawer-fix capture (branches visible), and the n-of-k
summary-trace capture. All three commits referenced "the
vertical_slice_2026-05-09.webm" but each was actually a different
file. That's misleading — the design history needs each capture
to remain accessible. Renamed retroactively (see
`2026-05-09_test16_*_<sha>.webm` siblings).

## Curation flow

After a test run produces useful artifacts in
`tools/st-debug/tests/test-results/<test-name>/`:

```bash
# Determine descriptive tag + current commit
TAG="branches_visible"
SHA=$(git rev-parse --short HEAD)
DATE=$(date +%Y-%m-%d)

cp tools/st-debug/tests/test-results/16_vertical_slice-vertical-*/video.webm \
   docs/media/${DATE}_test16_${TAG}_${SHA}.webm

git add docs/media/${DATE}_test16_${TAG}_${SHA}.webm
git commit -m "..."
```

Companion helper: `docs/media/curate.sh` (TODO).

## Index of current artifacts

| File | What it shows |
|------|---------------|
| `2026-05-09_test16_drawer_overlay_BUG_75771f7.webm` | The original test 16 video — API connection drawer covered the entire interface for the full 25s recording. Test passed because DOM-query assertions don't care about visibility. **Negative example**: a test that "passes" but produces useless evidence. |
| `2026-05-09_test16_branches_visible_2a69255.webm` | After the connectApi() drawer-dismiss fix. Shows fresh chat with Scringlo, user question, tree-of-thoughts collapsible expanding with 3 branch cards transitioning running → complete → done. |
| `2026-05-09_test16_n_of_k_summary_trace_74a879c.webm` | After the n-of-k summary_progress feature landed. Same flow as v2 plus the new pink-bordered summary trace block with parent-voice scringlo summaries (4 lines compressing 24+ raw branch reasoning lines). |
| `2026-05-09_test16_checkpoint4_branches_visible_2a69255.png` | Final-frame screenshot from v2: collapsible at "done (18.8s)", branches collapsed. |
| `2026-05-09_test16_checkpoint4_summary_trace_74a879c.png` | Final-frame screenshot from v3: collapsible forced open, summary trace visible with 💭-prefixed lines and (K→1) compression annotations. |
| `2026-05-09_test16_chat_*.json` | The chat[] state dump matching each captured video — names, message excerpts, tool_progress + summary_trace contents. |
| `streaming_e2e_2026-05-09.webm` | Test 15: SSE streaming reaches DOM progressively. (Single capture; not yet renamed to convention. TODO.) |
| `streaming_e2e_2026-05-09.trace.json` | DOM mutation trace for test 15. |
