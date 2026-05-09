# toolcards plugin: FIFO session serialization is fragile under client churn

**Status**: Architectural finding from static analysis of
`~/sillytavern-fork/plugins/toolcards/index.mjs`, 2026-05-08. The
captioned-toolcard test (st-debug `05_toolcards_captioned.spec.js`)
exposed the symptom — multi-minute hangs with no error events whenever a
prior test run was interrupted. Root cause identified by walking
startSession's state flow rather than running more dynamic tests.

## The bug

```javascript
// plugins/toolcards/index.mjs, ~line 338
async startSession(toolName, args, opts = {}) {
    await this._spawnIfNeeded();
    const session = { ... };
    if (this.activeSession) {
        this.invokeQueue.push({ session });   // ← FIFO sync block
    } else {
        this._beginSession(session);
    }
    return session;
}
```

There is **one service process per card**. New sessions queue behind
any active session. Sessions only "finish" via `finishSession` which
is called from the `result` event handler (line ~727) — never from a
timeout, never from a heartbeat, never from a client disconnect.

## How orphans accumulate

The pathology that made the captioned-toolcard test hang:

1. Test fires `start_invoke` against `query-to-svg-captioned`. Plugin
   spawns the service, marks an active session, sends the invocation
   request.
2. Service starts iter 1, fires `llm_call` for SVG generation. Plugin
   dispatches to bridge, gets response, writes `llm_response` to
   service stdin. Service unblocks, processes the SVG.
3. Test gets killed (Ctrl-C, timeout, browser navigation, network
   blip). Plugin still has the session marked active. Service is
   still alive holding the active-session reference.
4. Service either keeps running its iter loop (will eventually emit
   `result` and finish — could take 30s to several minutes for
   multi-iter), OR is mid-`sys.stdin.readline()` waiting for an
   `llm_response` that no one will trigger.
5. Next `start_invoke` arrives. Plugin sees `this.activeSession`
   non-null, pushes the new session onto `invokeQueue`. **The new
   session waits for the orphan to emit `result`.**
6. If the orphan was waiting on a never-arriving stdin response, it
   waits forever; the queue is blocked forever; the new test sees
   no events and hangs to its 7-minute timeout.

There's an idle timer (`_bumpIdle`, 120s default) that kills the
service if no llm_call/result/sendLlmResponse activity happens. But:
- It's bumped on `_beginSession`, on result events, and on
  sendLlmResponse — so a service waiting on stdin keeps the timer
  alive as long as SOME llm_call recently completed.
- It KILLS the process on idle, which forces a SIGTERM. `_reset`
  fires, releasing the active session. But this is a 2-minute
  worst-case wait per orphan.
- Multiple stacked orphans + queued sessions = compounding delay.

## Why this is the wrong architecture

The user's framing (2026-05-08): *FIFO sync blocks are almost never
appropriate, and async programming should tolerate large numbers of
clients which might be slightly partitioned for a little while
without entering soft-locked states.*

The plugin's design assumes:
- One client at a time per card (FE-driven, sequential tool calls)
- Clean termination always (FE notifies on disconnect)
- Bounded service runtime (idle timer is the only safety net)

In practice:
- Test harnesses, parallel agent clients, and HTTP clients across
  network blips all break "one at a time."
- Disconnects don't always notify (Ctrl-C of a curl, browser tab
  close mid-poll, network partition).
- Long-running cards (multi-iter SVG refinement, document analysis)
  can run minutes — the idle timer doesn't reflect "this client is
  alive and wants to wait" vs "this orphan is dead."

The result is a system that becomes brittle under any kind of
concurrency or partial-failure, manifesting as silent hangs that
look like correctness bugs.

## Three plausible fixes (in increasing reorganization)

### A. Per-session service process (drop the queue)

Every `startSession` spawns a NEW service process. The existing
"one process, queued sessions" pattern is replaced with "one process
per active session." Services are lightweight (Python + stdlib for
trivial cards, Python + heavy deps like playwright for SVG cards) so
spawn cost matters but not prohibitively — the heavy deps are loaded
once via `uv`'s cache.

Cleanup: process exits when its session emits `result` or hits idle
timeout. Plugin's `getOrCreateService` becomes per-session, not
per-card.

Pros: simplest mental model. No queue, no cross-session interaction.
Cons: spawn cost per invocation (1-3s for query-to-svg with
playwright). Mitigations: warm-pool of pre-spawned services per
card (k=2 ready, refilled on use).

### B. Heartbeat-driven session expiry

Each active session has a heartbeat — bumped by every event from the
service AND by every poll/cancel call from the client. If no
heartbeat in N seconds (e.g., 60), the session is force-finished
(synthetic `error` event), service is reset, queue moves on.

Pros: keeps the per-card-process model. Bounds blast radius to N
seconds per orphan.
Cons: still serializes across clients. Long-running legitimate
sessions need to either heartbeat from the service side or set a
generous N.

### C. Multiplexed services (proper concurrency)

Services declare in the manifest whether they support multiplexing
(`runtime.multiplex: true`). Multiplexed services get ONE process
that handles N concurrent sessions distinguished by event `id`.
Non-multiplexed services use option A.

The query-to-svg service.py would need rework to be multiplex-safe
(track per-session state, route llm_response back to the correct
in-flight call). Maybe `id` already does this and just needs a
session_id added.

Pros: proper async concurrency, no extra processes.
Cons: most invasive; service.py reworks; manifest schema change.

## Recommendation

**A is the right answer** — per-session service processes, no shared
queue. (B is an anti-pattern in this context; see below.)

The user's correction (2026-05-08): *2 min synchronized blocking is
way too long; a 10 min concurrent async timeout is relatively modest.*
This rules out option B. The reason heartbeat-expiry doesn't fix the
real problem:

- Heartbeat-expiry shortens the SYNC blocking window (30s, 60s, 120s
  pick your number) — but it's still SYNC. Every client behind the
  orphan waits the full window.
- A 10-minute concurrent async timeout is fine because OTHER CLIENTS
  AREN'T BLOCKED by it. Each session has its own bounded path; they
  run in parallel. Long-tail latency on one session doesn't affect
  any other.
- The right axis isn't "make the sync timeout shorter," it's "stop
  having a sync timeout in the first place."

So **A is the architectural fix**. Each session gets its own service
process (or its own slot in a multiplexed service that genuinely
supports concurrency). The "2 minutes idle = kill" timer can go to
10 minutes or longer per-session and the system stays responsive
because no other client is gated on it.

The warm-pool optimization (k=2 ready services per card, refilled on
use) keeps spawn cost per invocation low — relevant for query-to-svg
which needs playwright loaded (~1-2s cold, ~50ms warm).

C (multiplexed services) is the architecturally clean answer for
LIGHTWEIGHT cards (hello-world, anything stdio-pure), and probably
the long-term destination, but requires coordinated plugin + service
+ card-schema changes. A is achievable as a single plugin-side
refactor.

## Workaround currently shipped

`tools/st-debug/scripts/run.sh` and the `beforeEach` in the
captioned-toolcard test both `pkill -f 'uv run.*python service.py'`
to kill orphans. This is a brutal band-aid that loses any in-progress
work, but it makes the test reliable. Verified test 05 wall-time
dropped from 7+ minute hangs to **8.3s clean pass** after applying.

The pkill is local to the debug instance — does not affect the user's
main install at `~/sillytavern-fork/data/toolcards/`.

## Cross-references

- `tools/st-debug/scripts/install_captioned_toolcard.sh` — installs the
  captioned variant whose multi-llm-call pattern surfaced the bug
- `tools/st-debug/tests/05_toolcards_captioned.spec.js` — has the
  beforeEach orphan-kill, plus comments referencing this finding
- `tools/st-debug/scripts/run.sh` — kills orphans on every server
  start, comments reference this finding
- `docs/descendant_agent_ux_spec.md` — broader context for the toolcard
  UX work that motivated the captioned variant in the first place
