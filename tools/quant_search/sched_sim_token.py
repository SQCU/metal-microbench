#!/usr/bin/env python3
"""Empirical simulation: where does the CURRENT scheduling implementation
actually have a throughput advantage over the NULL control flow ('always run
a kernel that emits tokens')?

Models the three decisions codex+claude jointly identified as token-blocking:

  D1: Throughput-prefer-AR defer (lm_engine.swift:2420)
       CURRENT — if nAR >= 2 and one slot is priming, run AR (defer prefill)
       NULL    — if any slot is priming, run prefill immediately

  D2: Shared-prefix follower deferral (ffi_batch.swift:386, 930)
       CURRENT — followers (sharing leader's first-page hash) sit in a
                 deferred queue until leader is past prefill, then adopt
                 leader's KV pages for free
       NULL    — followers admit immediately as separate sessions, prefill
                 their own KV pages independently (no sharing)

  D3: gemma_poll early-return on first token (ffi_batch.swift:1049)
       CURRENT — engine returns to Python on any update; ~T_python overhead
                 between productive CBs
       NULL    — engine keeps driving CBs until intake is empty AND no work,
                 OR hits the timeout; no host gap between CBs

This simulator runs all three. For each config, both CURRENT and NULL paths
execute against the same arrival schedule, and aggregate tok/s is reported.

The model is discrete-CB:
  - AR CB:           wall = T_AR (85 ms), emits 1 token per slot in .generating
  - Prefill CB:      wall = T_prefill (~T_AR for short qLen), advances priming
                     slots by Q_TILE tokens, emits 0 tokens, silences non-target
                     slots in single-slot prefill mode
  - Idle CB:         wall = T_AR, emits 0 tokens (no work)
  - Each productive CB in CURRENT incurs T_python overhead before next CB

Output: aggregate tokens/sec over a fixed simulated wall window for each
config. Comparisons are CURRENT/NULL ratios.
"""
from __future__ import annotations
import argparse
import dataclasses
import enum
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

# ─── Engine constants (matched to bootstrap.swift / lm_engine.swift) ──────────
B = 8                     # number of slots
T_AR_MS = 85.0            # observed AR-tick wall in production (ms)
T_PREFILL_MS = 85.0       # prefill ~= AR for typical short qLen
Q_TILE = 128              # prefill chunk size (tokens advanced per prefill CB)
T_PYTHON_GAP_MS = 8.0     # observed host gap on early-return (5-15 ms typical)
SIM_WALL_MS = 60_000.0    # simulate 60 wall-seconds per config


class State(enum.Enum):
    IDLE = 0       # no session here
    PRIMING = 1    # session present, prefill not done
    GENERATING = 2 # session present, decoding tokens
    DONE = 3       # session finished, slot will free at next admission


@dataclass
class Session:
    """One inference request."""
    sid: int
    prompt_len: int
    completion_len: int
    arrival_ms: float
    shared_prefix_hash: Optional[int] = None  # None = no shared prefix
    # runtime state ↓
    state: State = State.IDLE
    prefilled: int = 0          # how much of the prompt is in KV cache
    decoded: int = 0            # how many completion tokens emitted
    is_follower: bool = False   # set when sharing prefix with another session
    leader_sid: Optional[int] = None


@dataclass
class Result:
    tokens_emitted: int = 0
    cb_count: int = 0
    ar_cbs: int = 0
    prefill_cbs: int = 0
    idle_cbs: int = 0
    sim_wall_ms: float = 0.0

    @property
    def aggregate_tps(self) -> float:
        return 1000.0 * self.tokens_emitted / max(self.sim_wall_ms, 1.0)


# ─── Workload generation ──────────────────────────────────────────────────────
def make_arrivals(rng: random.Random, n: int, lam_per_sec: float,
                  prompt_len_mean: int, completion_len_mean: int,
                  p_shared: float, share_group_size: int = 4) -> list[Session]:
    """Poisson-arrival sessions with exponential prompt/completion lengths.
    Shared-prefix sessions arrive as bursts of `share_group_size` with the
    same hash."""
    sessions: list[Session] = []
    t_ms = 0.0
    sid = 0
    next_share_hash = 1
    pending_share_count = 0
    pending_share_hash = None
    while len(sessions) < n:
        # Inter-arrival: exponential with mean 1/lam
        if lam_per_sec > 0:
            dt_ms = rng.expovariate(lam_per_sec / 1000.0)
        else:
            dt_ms = 1e9
        t_ms += dt_ms
        prompt = max(1, int(rng.expovariate(1.0 / prompt_len_mean)))
        completion = max(1, int(rng.expovariate(1.0 / completion_len_mean)))
        # Decide shared-prefix membership
        if pending_share_count > 0:
            sh = pending_share_hash
            pending_share_count -= 1
        elif rng.random() < p_shared:
            sh = next_share_hash
            next_share_hash += 1
            pending_share_hash = sh
            pending_share_count = share_group_size - 1
        else:
            sh = None
        sessions.append(Session(sid=sid, prompt_len=prompt, completion_len=completion,
                                 arrival_ms=t_ms, shared_prefix_hash=sh))
        sid += 1
    return sessions


# ─── Engine simulator ─────────────────────────────────────────────────────────
@dataclass
class Engine:
    # policy flags
    use_throughput_prefer_ar: bool   # D1: True = CURRENT, False = NULL
    use_follower_deferral: bool      # D2: True = CURRENT, False = NULL
    use_poll_early_return: bool      # D3: True = CURRENT, False = NULL
    # state ↓
    slots: list[Optional[Session]] = field(default_factory=lambda: [None]*B)
    deferred: list[Session] = field(default_factory=list)  # follower queue (D2)
    pending_intake: list[Session] = field(default_factory=list)
    leader_active_hash: dict[int, int] = field(default_factory=dict)  # hash → leader_sid
    leader_done_hash: set[int] = field(default_factory=set)
    t_ms: float = 0.0
    result: Result = field(default_factory=Result)

    # ── admission ────────────────────────────────────────────────────────────
    def _admit(self) -> None:
        # Free DONE slots
        for i, s in enumerate(self.slots):
            if s is not None and s.state == State.DONE:
                self.slots[i] = None
        # Try to admit pending
        new_pending = []
        for s in self.pending_intake:
            if s.shared_prefix_hash is not None and self.use_follower_deferral:
                # Follower deferral logic
                h = s.shared_prefix_hash
                if h in self.leader_active_hash:
                    # Already a leader: defer
                    s.is_follower = True
                    s.leader_sid = self.leader_active_hash[h]
                    self.deferred.append(s)
                    continue
                else:
                    # Become leader
                    self.leader_active_hash[h] = s.sid
            placed = False
            for i in range(B):
                if self.slots[i] is None:
                    self.slots[i] = s
                    s.state = State.PRIMING
                    s.prefilled = 0
                    placed = True
                    break
            if not placed:
                new_pending.append(s)
        self.pending_intake = new_pending
        # Retry deferred followers whose leader has primed (D2 CURRENT path)
        if self.use_follower_deferral:
            still_deferred = []
            for s in self.deferred:
                h = s.shared_prefix_hash
                if h in self.leader_done_hash:
                    # Find leader session, copy prefilled = leader.prompt_len (free!)
                    leader = self._find_session(s.leader_sid)
                    if leader is None:
                        # leader gone — admit normally
                        for i in range(B):
                            if self.slots[i] is None:
                                self.slots[i] = s
                                s.state = State.PRIMING
                                s.prefilled = 0
                                break
                        else:
                            still_deferred.append(s)
                        continue
                    placed = False
                    for i in range(B):
                        if self.slots[i] is None:
                            self.slots[i] = s
                            # KEY: follower adopts leader's prefill state.
                            s.prefilled = leader.prompt_len
                            if s.prefilled >= s.prompt_len:
                                s.state = State.GENERATING
                            else:
                                s.state = State.PRIMING
                            placed = True
                            break
                    if not placed:
                        still_deferred.append(s)
                else:
                    still_deferred.append(s)
            self.deferred = still_deferred

    def _find_session(self, sid: Optional[int]) -> Optional[Session]:
        if sid is None: return None
        for s in self.slots:
            if s is not None and s.sid == sid: return s
        return None

    # ── path selection ──────────────────────────────────────────────────────
    def _pick_path(self) -> str:
        """Decide what kind of CB to run next. Returns one of:
        'idle', 'ar', 'prefill_single', 'prefill_multi'."""
        active = [s for s in self.slots if s is not None
                  and s.state in (State.PRIMING, State.GENERATING)]
        if not active:
            return 'idle'
        n_priming = sum(1 for s in active if s.state == State.PRIMING)
        n_generating = sum(1 for s in active if s.state == State.GENERATING)

        # CURRENT: Throughput-prefer-AR — if nAR>=2 and exactly 1 priming,
        # defer prefill and run AR.
        if self.use_throughput_prefer_ar and n_generating >= 2 and n_priming == 1:
            return 'ar'

        # If some are priming, prefer prefill (multi if >=2 priming)
        if n_priming >= 2:
            return 'prefill_multi'
        if n_priming == 1:
            # NULL or CURRENT-without-deferral both prefill here
            if n_generating == 0:
                return 'prefill_single'  # nothing to silence
            # NULL routes always to prefill_single; CURRENT may have already
            # picked AR above
            return 'prefill_single'
        # All generating
        return 'ar'

    # ── CB execution ─────────────────────────────────────────────────────────
    def _run_ar_cb(self) -> int:
        """Run one AR CB. Returns tokens emitted across all generating slots."""
        emitted = 0
        for s in self.slots:
            if s is None or s.state != State.GENERATING:
                continue
            s.decoded += 1
            emitted += 1
            if s.decoded >= s.completion_len:
                s.state = State.DONE
                # Mark leader as past-priming so followers can apply
                if s.shared_prefix_hash is not None:
                    self.leader_done_hash.add(s.shared_prefix_hash)
                    self.leader_active_hash.pop(s.shared_prefix_hash, None)
        return emitted

    def _run_prefill_cb(self, multi: bool) -> int:
        """Run one prefill CB. Advances priming slots by Q_TILE. Multi advances
        all priming slots; single advances one (silencing others, no AR emit)."""
        if multi:
            for s in self.slots:
                if s is None or s.state != State.PRIMING:
                    continue
                s.prefilled = min(s.prefilled + Q_TILE, s.prompt_len)
                if s.prefilled >= s.prompt_len:
                    s.state = State.GENERATING
                    if s.shared_prefix_hash is not None:
                        self.leader_done_hash.add(s.shared_prefix_hash)
        else:
            # single-slot prefill: pick first priming slot
            for s in self.slots:
                if s is None or s.state != State.PRIMING:
                    continue
                s.prefilled = min(s.prefilled + Q_TILE, s.prompt_len)
                if s.prefilled >= s.prompt_len:
                    s.state = State.GENERATING
                    if s.shared_prefix_hash is not None:
                        self.leader_done_hash.add(s.shared_prefix_hash)
                break  # only one slot advances
        return 0

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self, arrivals: list[Session], sim_wall_ms: float) -> Result:
        arrivals = sorted(arrivals, key=lambda s: s.arrival_ms)
        arr_idx = 0
        last_productive = False
        while self.t_ms < sim_wall_ms:
            # Drain arrivals up to current time
            while arr_idx < len(arrivals) and arrivals[arr_idx].arrival_ms <= self.t_ms:
                self.pending_intake.append(arrivals[arr_idx])
                arr_idx += 1
            self._admit()

            path = self._pick_path()
            if path == 'idle':
                # No work — advance by T_AR (or wait for next arrival)
                if arr_idx < len(arrivals):
                    next_arr = arrivals[arr_idx].arrival_ms
                    self.t_ms = max(self.t_ms + 1.0, next_arr)
                else:
                    self.t_ms += T_AR_MS
                self.result.cb_count += 1
                self.result.idle_cbs += 1
                last_productive = False
                continue

            # CURRENT D3: incur python gap before this CB if last was productive
            if self.use_poll_early_return and last_productive:
                self.t_ms += T_PYTHON_GAP_MS

            if path == 'ar':
                tok = self._run_ar_cb()
                self.t_ms += T_AR_MS
                self.result.ar_cbs += 1
                self.result.tokens_emitted += tok
                last_productive = (tok > 0)
            elif path == 'prefill_single':
                self._run_prefill_cb(multi=False)
                self.t_ms += T_PREFILL_MS
                self.result.prefill_cbs += 1
                last_productive = True   # productive in the sense work happened
            elif path == 'prefill_multi':
                self._run_prefill_cb(multi=True)
                self.t_ms += T_PREFILL_MS
                self.result.prefill_cbs += 1
                last_productive = True
            self.result.cb_count += 1

        self.result.sim_wall_ms = self.t_ms
        return self.result


# ─── Sweep ────────────────────────────────────────────────────────────────────
def run_one(arrivals: list[Session], current_d1: bool, current_d2: bool,
             current_d3: bool, sim_wall_ms: float = SIM_WALL_MS) -> Result:
    # Deep copy arrivals so each run starts fresh
    fresh = [dataclasses.replace(s) for s in arrivals]
    eng = Engine(use_throughput_prefer_ar=current_d1,
                  use_follower_deferral=current_d2,
                  use_poll_early_return=current_d3)
    return eng.run(fresh, sim_wall_ms)


def sweep():
    rng = random.Random(42)
    print(f"# Workload throughput characterization, sim_wall={SIM_WALL_MS/1000:.0f}s, "
          f"B={B}, T_AR={T_AR_MS}ms, T_python_gap={T_PYTHON_GAP_MS}ms")
    print()

    # Three scans, one per decision, isolating its effect.
    scans = [
        ("D1: Throughput-prefer-AR  (defer prefill when nAR>=2 and 1 priming)",
         "lam_per_sec",
         [(0.5, "sparse"), (1.0, "moderate"), (2.0, "saturating"), (5.0, "bursty")],
         dict(prompt_len_mean=128, completion_len_mean=200, p_shared=0.0),
         lambda lam, kw: ((True, False, False), (False, False, False))),
        ("D2: Shared-prefix follower deferral",
         "p_shared",
         [(0.0, "no shared"), (0.25, "25%"), (0.5, "50%"), (1.0, "all groups of 4")],
         dict(lam_per_sec=2.0, prompt_len_mean=512, completion_len_mean=200),
         lambda p, kw: ((False, True, False), (False, False, False))),
        ("D3: gemma_poll early-return (T_python gap per productive CB)",
         "lam_per_sec",
         [(0.5, "sparse"), (1.0, "moderate"), (2.0, "saturating"), (5.0, "bursty")],
         dict(prompt_len_mean=128, completion_len_mean=200, p_shared=0.0),
         lambda lam, kw: ((False, False, True), (False, False, False))),
    ]

    for title, sweep_var, sweep_vals, fixed, policy_fn in scans:
        print(f"## {title}")
        print(f"   fixed: {', '.join(f'{k}={v}' for k,v in fixed.items())}")
        print()
        print(f"  {sweep_var:>16}  {'label':>20}  {'CURRENT':>12}  {'NULL':>12}  "
              f"{'delta':>10}  {'%':>8}")
        print(f"  {'-'*16}  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*8}")
        for val, label in sweep_vals:
            kw = dict(fixed)
            kw[sweep_var] = val
            arrivals = make_arrivals(random.Random(42), n=2000, **kw)
            cur_pol, null_pol = policy_fn(val, kw)
            cur = run_one(arrivals, *cur_pol)
            nul = run_one(arrivals, *null_pol)
            delta = cur.aggregate_tps - nul.aggregate_tps
            pct = 100.0 * delta / max(nul.aggregate_tps, 0.01)
            mark = "  ✓" if cur.aggregate_tps > nul.aggregate_tps else (
                "" if abs(pct) < 0.5 else "  ✗")
            print(f"  {val:16}  {label:>20}  {cur.aggregate_tps:>12.1f}  "
                  f"{nul.aggregate_tps:>12.1f}  {delta:>+10.1f}  {pct:>+7.1f}%{mark}")
        print()

    print("## D2 break-even surface (p_shared × prompt_len, lam=2.0, comp=200)")
    print("   Where does follower-deferral start to win, and by how much?")
    print()
    print(f"  {'p_shared':>9}  {'prompt=128':>12}  {'prompt=512':>12}  "
          f"{'prompt=1024':>13}  {'prompt=4096':>13}")
    print(f"  {'-'*9}  {'-'*12}  {'-'*12}  {'-'*13}  {'-'*13}")
    for ps in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        row_vals = []
        for pl in (128, 512, 1024, 4096):
            arrivals = make_arrivals(random.Random(42), n=2000,
                                      lam_per_sec=2.0, prompt_len_mean=pl,
                                      completion_len_mean=200, p_shared=ps)
            cur = run_one(arrivals, False, True, False)   # D2 only
            nul = run_one(arrivals, False, False, False)
            pct = 100.0 * (cur.aggregate_tps - nul.aggregate_tps) / max(nul.aggregate_tps, 0.01)
            mark = "+" if pct > 0 else ""
            row_vals.append(f"{mark}{pct:>+5.1f}%")
        print(f"  {ps:>9.2f}  {row_vals[0]:>12}  {row_vals[1]:>12}  "
              f"{row_vals[2]:>13}  {row_vals[3]:>13}")
    print()

    print("## Combined (all 3 CURRENT vs all 3 NULL)")
    print()
    print(f"  {'lam':>5}  {'p_shared':>9}  {'prompt':>7}  {'comp':>5}  "
          f"{'CURRENT':>10}  {'NULL':>10}  {'delta':>8}  {'%':>7}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*7}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*7}")
    grid = [
        # (lam_per_sec, p_shared, prompt, completion)
        (0.5, 0.0, 128, 200),
        (1.0, 0.0, 128, 200),
        (2.0, 0.0, 128, 200),
        (5.0, 0.0, 128, 200),
        (2.0, 0.5, 128, 200),
        (2.0, 1.0, 128, 200),
        (5.0, 0.5, 128, 200),
        (5.0, 1.0, 128, 200),
        (1.0, 0.0, 512, 200),
        (2.0, 0.0, 512, 200),
        (1.0, 0.0, 1024, 200),
        (2.0, 0.0, 128, 50),
        (2.0, 0.0, 128, 500),
    ]
    for lam, ps, pl, cl in grid:
        arrivals = make_arrivals(random.Random(42), n=2000,
                                  lam_per_sec=lam, prompt_len_mean=pl,
                                  completion_len_mean=cl, p_shared=ps)
        cur = run_one(arrivals, True, True, True)
        nul = run_one(arrivals, False, False, False)
        delta = cur.aggregate_tps - nul.aggregate_tps
        pct = 100.0 * delta / max(nul.aggregate_tps, 0.01)
        mark = "  ✓" if cur.aggregate_tps > nul.aggregate_tps else (
            "" if abs(pct) < 0.5 else "  ✗")
        print(f"  {lam:>5.1f}  {ps:>9.2f}  {pl:>7d}  {cl:>5d}  "
              f"{cur.aggregate_tps:>10.1f}  {nul.aggregate_tps:>10.1f}  "
              f"{delta:>+8.1f}  {pct:>+6.1f}%{mark}")
    print()


if __name__ == "__main__":
    sweep()
