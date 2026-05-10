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

### test 17 — render-visual (2-fork + 2-spawn recursive decomposition)

| File | What it shows |
|------|---------------|
| `2026-05-09_test17_lissajous_failure_surfaced_9b26e22.webm` | User asked for a Lissajous 3:5 curve; gemma-4 (in S1) misinterpreted and produced Voronoi-like tessellation code. S2 validation reported 2 of 5 invariants failed. F2 (scringlo wrap-up) honestly told the user "my teeny robot friend got all confused and started making a mosaic instead of a wobbly loop! it's a bit of a math-oopsie." Real-pixel evidence of failure-surfacing-as-affordance: the harness exposed the descendant's mistake in scringlo's voice. |
| `2026-05-09_test17_voronoi_success_9b26e22.webm` | User asked for a Voronoi diagram with 12 seeds; gemma-4 produced correct tessellation code. S2 reported 4/5 PASS / 1 FAIL on heuristic invariants. F2 wrapped up with caveats ("had a teeny tiny tummy ache with some of the math-y bits at first, but it managed to wiggle through"). The validator's heuristics aren't strong enough to catch all failures; F2 honestly conveys the partial-pass status. |
| `2026-05-09_test17_lissajous_failure_surfaced_chat_*.json` + `_checkpoint4_*.png` | Final-state captures from the failure run, showing the embedded SVG mosaic and the summary_trace lines surfaced through scringlo's voice. |
| `2026-05-09_test17_voronoi_success_chat_*.json` + `_checkpoint4_*.png` | Final-state captures from the success run — recognizable Voronoi tessellation embedded inline. |

### Open oversight gap from test 17

Both runs surface VALIDATOR-LEVEL findings (S2's invariant
checklist) faithfully, but neither catches "is the visual the
shape the user asked for?" The validator scores well-formedness
+ syntactic invariants of the python output; it does NOT
re-examine the rendered SVG's visual correctness. F2 paraphrases
the validator report without itself looking at the picture.

This is a real and useful learning from the demo: n-of-k
oversight catches what its validators are designed to catch.
For visual-output tasks, a vision-enabled validator stage is
needed (image_url passed back, "does this look like a Lissajous
curve?"). Or stronger numerical invariants (e.g., for Lissajous
3:5: count distinct self-intersections of the path; should be 4).
Logged as future work; documented here as a feature of the
honest demo, not a defect in the harness pattern.

### test 18 — vision-review (closes the visual oversight gap from test 17)

| File | What it shows |
|------|---------------|
| `2026-05-09_test18_vision_review_audit_9d8bb36.webm` | Scringlo asked "did you actually draw a real lissajous curve?" Invokes vision-review on the curated test 17 failure video. Per-frame multimodal review: frame 4 description is "a tessellated composition of various polygons in colors such as green, purple, blue, tan, brown, and pink." Final verdict: **FAIL** — "no visual evidence of a tool being used to draw, nor is there a Lissajous curve." Closes the test 17 oversight gap with real-pixel evidence: model-on-model audit of recorded transcripts works end-to-end. |
| `2026-05-09_test18_vision_review_audit_checkpoint4_9d8bb36.png` | Final-state screenshot showing scringlo's bubble with the Vision Review collapsible expanded, displaying 5 💭-prefixed per-frame descriptions in the summary trace + verdict. |
| `2026-05-09_test18_vision_review_audit_chat_9d8bb36.json` | Full chat[] state with all 5 frame descriptions + the verdict reasoning. Wall time: 11.4s for the whole audit (5 sequential multimodal vision calls + 1 verdict synthesis). |

### Findings from test 18

- **Model-on-model audit works**: gemma-4 in the reviewer role correctly identified the test 17 video's actual content (tessellation, not Lissajous curve) and reported FAIL with reasoning grounded in the per-frame descriptions.
- **Per-frame review is sequential, not parallel**: parallel multimodal calls produced `<unused6226>` placeholder-token outputs at the bridge (logged for future bridge debugging). Sequential is fine — 5 frames take ~11s with prefix-cache hits on the shared REVIEWER_PREFIX.
- **The visual oversight gap from test 17 is closed**: a downstream invocation of vision-review on the recorded webm catches "shape doesn't match request" failures that test 17's invariant-level S2 didn't.
- **This card composes**: any video-producing toolcard (render-visual, future iter-svg-refine, etc.) can be followed by vision-review on its captured output to close the visual-correctness loop.

### Bridge bug logged for future investigation

Parallel multimodal calls (multiple `image_url` content parts dispatched
concurrently via parallel_llm_call) produce `<unused6226>` placeholder-
token outputs. Single-frame multimodal works correctly. Workaround in
`vision-review/service.py`: sequential per-frame loop. Suspected cause:
vision encoder state corruption across concurrent stream submissions
in the bridge; see `/Users/mdot/metal-microbench/server/bridge.py`
vision integration. Logged here, not yet fixed.
