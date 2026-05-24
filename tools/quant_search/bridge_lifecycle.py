"""Bridge process lifecycle — verified kill + launch + health-wait.

Why this exists
===============
The naive pattern ``subprocess.run(["pkill", "-f", "server/serve.py"])``
followed by ``time.sleep(2)`` and a fresh launch silently fails in practice:

* ``pkill`` defaults to SIGTERM, which Python's signal handling can defer
  indefinitely when the bridge is in async-IO or holding the GIL through
  C extensions. Uvicorn additionally intercepts SIGTERM for "graceful
  shutdown", which tries to finish in-flight requests before exiting; if
  those requests are stuck, shutdown hangs and the process never dies.
* No verification step means the launcher proceeds whether the prior
  bridge died or not. Across a workday, orphan bridges accumulate.
  Each one keeps a Metal command queue + compiled PSO cache resident on
  the GPU, even when idle — and they compete with the "current" bridge
  for scheduler headroom on the same Apple Silicon GPU.
* In one observed instance, three orphan bridges from a single workday
  caused a 31× slowdown of a small benchmark study (200s vs 6.4s for
  the same 40-call workload). Killing the orphans restored throughput.

This module replaces that pattern with: SIGKILL (kernel-enforced,
synchronous) → verified absence via pgrep → port-availability check →
fresh launch → /health poll. Eval scripts use ``BridgeContext`` as a
context manager and never call shell commands.

Usage
=====

    from bridge_lifecycle import BridgeContext

    with BridgeContext("/path/to/model.gguf") as bridge:
        # bridge.url is "http://127.0.0.1:8001"
        # bridge.pid is the bridge process pid
        # ... fire requests ...
    # On exit: bridge is SIGKILL'd, verified dead, port reclaimed.
    # Subsequent BridgeContext entries will not find orphans.

For finer control:

    kill_all_bridges()             # SIGKILL all + verify
    proc = launch_bridge(gguf, port=8001)  # kill+launch+health
    # ... use ...
    kill_all_bridges()             # cleanup
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BRIDGE_PROCESS_PATTERN = "server/serve.py"
DEFAULT_PORT = 8001
DEFAULT_HEALTH_TIMEOUT_S = 120.0
DEFAULT_KILL_TIMEOUT_S = 5.0


def find_bridge_pids() -> list[int]:
    """Return PIDs of all running bridge processes (matched by argv pattern)."""
    out = subprocess.run(
        ["pgrep", "-f", BRIDGE_PROCESS_PATTERN],
        capture_output=True, text=True, check=False,
    )
    pids: list[int] = []
    for tok in out.stdout.strip().split():
        try:
            pids.append(int(tok))
        except ValueError:
            pass
    # Filter ourselves out (we're a Python process that COULD match by
    # cwd / module path arguments).
    me = os.getpid()
    return [p for p in pids if p != me]


def kill_all_bridges(timeout_s: float = DEFAULT_KILL_TIMEOUT_S) -> None:
    """SIGKILL every bridge process and verify all are gone.

    SIGKILL is kernel-enforced; the process cannot catch it, defer it,
    or "gracefully" handle it. After signaling, we poll pgrep until
    no bridge PIDs remain or until the timeout expires.

    Raises RuntimeError if any bridge survives after `timeout_s`.
    """
    pids = find_bridge_pids()
    if not pids:
        return
    print(f"[bridge_lifecycle] SIGKILL {len(pids)} bridge(s): {pids}",
          flush=True, file=sys.stderr)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass    # already dead
        except PermissionError:
            print(f"[bridge_lifecycle] permission denied killing pid={pid}",
                  flush=True, file=sys.stderr)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        remaining = find_bridge_pids()
        if not remaining:
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"bridge process(es) survived SIGKILL: {find_bridge_pids()}"
    )


def is_port_free(port: int, host: str = "0.0.0.0") -> bool:
    """True if `port` is available for bind on `host`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def wait_for_port_free(port: int, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_port_free(port):
            return
        time.sleep(0.25)
    raise RuntimeError(f"port {port} still bound after {timeout_s}s "
                       f"(orphan bridge or other listener?)")


def wait_for_health(url: str, timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S
                     ) -> dict:
    """Poll {url}/health until status=='ready' or timeout. Returns the
    final health JSON."""
    deadline = time.time() + timeout_s
    last_err: BaseException | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                d = json.loads(r.read())
                if d.get("status") == "ready":
                    return d
        except BaseException as e:                    # noqa: BLE001
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(
        f"bridge /health did not report 'ready' within {timeout_s}s "
        f"(last_err={last_err!r})"
    )


def _repo_root() -> Path:
    """Repo root: parent.parent.parent of this file's location."""
    return Path(__file__).resolve().parents[2]


def launch_bridge(gguf_path: str | Path,
                   port: int = DEFAULT_PORT,
                   log_path: str | Path | None = None,
                   health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
                   extra_env: dict[str, str] | None = None,
                   ) -> subprocess.Popen:
    """Verified-clean launch: SIGKILL existing → wait port-free → spawn
    bridge → wait for /health ready → return Popen handle.

    Logging strategy (post-2026-05-06 macOS-watchdog-crash autopsy):

    - Logs land in `<repo>/bridge-logs/`, NOT `/tmp/`. macOS wipes
      `/tmp/` on reboot, so any kernel-watchdog-flavored crash
      destroys the only forensic record. Repo-relative survives.
    - One file per launch, named `<stem>_<UTC-timestamp>.log` so we
      keep history across launches instead of overwriting.
    - `<repo>/bridge-logs/latest_<stem>.log` is a symlink kept fresh
      so common tooling can `tail -f` without needing to know the
      timestamp.
    - Lines are timestamp-prefixed by a passthrough thread reading
      the bridge's stdout. The bridge itself doesn't need to know it
      is being logged — its existing `print(..., flush=True)` calls
      flow through unmodified.
    - File is opened with line buffering AND every line is flushed
      immediately. A hard kill (SIGKILL, panic, watchdog reset)
      preserves everything up to the most recent newline.
    - PYTHONUNBUFFERED=1 forces the bridge's own stdio to emit lines
      as soon as it generates them — no Python-side buffering hiding
      pre-crash state.
    """
    gguf_path = Path(gguf_path).resolve()
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")
    kill_all_bridges()
    wait_for_port_free(port)

    repo_root = _repo_root()
    serve_py = repo_root / "server" / "serve.py"
    venv_py = repo_root / "server" / ".venv" / "bin" / "python"
    if not serve_py.exists():
        raise FileNotFoundError(f"serve.py not found at {serve_py}")
    if not venv_py.exists():
        raise FileNotFoundError(f"venv python not found at {venv_py}")

    if log_path is None:
        logs_dir = repo_root / "bridge-logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        log_path = logs_dir / f"{gguf_path.stem}_{ts}.log"
        # Refresh the convenience symlink. The lifecycle of the
        # symlink is independent of the file it points at; if the
        # next bridge dies the symlink still references this older
        # log until the next successful launch.
        latest = logs_dir / f"latest_{gguf_path.stem}.log"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(log_path.name)
        except OSError:
            pass
    log_path = Path(log_path)

    env = {**os.environ,
           "GEMMA_GGUF": str(gguf_path),
           "GEMMA_PORT": str(port),
           # Force the bridge's own stdio to flush per line — without
           # this, FastAPI/uvicorn buffer prints in 4KB chunks and the
           # last several seconds of bridge state can be lost on a
           # hard crash.
           "PYTHONUNBUFFERED": "1",
           # Enable engine's per-100-AR-step profile prints
           # (lm_engine.swift:1834). Adds [PROF] lines to bridge log
           # with wall/gpu/cpu/handler/finalize/prep ms breakdowns
           # plus sched(ar/sM/sS/tM/tS) category counters. Costless
           # — gated by env var, only fires every 100 AR steps.
           "LM_PROF": "1"}
    # Metal validation (`MTL_DEBUG_LAYER`) — used during the
    # 2026-05-06 debug session to surface a real threadgroup-memory
    # overflow in flex_attn_slide_v1_q8 (the kernel was using 50 KB
    # against a 32 KB hardware limit). The kernel has since been
    # refactored to one-q-head-per-TG geometry (kernels.swift:9941)
    # which fits in 12.6 KB. To re-enable validation for further
    # debugging, set MTL_DEBUG_LAYER=1 in extra_env.
    # Metal validation (`MTL_DEBUG_LAYER`, `MTL_SHADER_VALIDATION`)
    # was used during the 2026-05-06 debug session to surface a
    # threadgroup-memory overflow in flex_attn_slide_v1_q8 (uses
    # 50,560 bytes per Metal's accounting; 32 KB hardware limit).
    # That overflow is silently tolerated in production runs without
    # validation. Validation is OFF by default — set
    # MTL_DEBUG_LAYER=1 in the environment to re-enable for further
    # debugging. Do NOT enable in production: validation hard-asserts
    # on the kernel's overflow and kills the bridge process.
    if extra_env:
        env.update(extra_env)

    # Open log file in line-buffered mode (buffering=1). Each line
    # gets flushed to disk immediately on newline.
    log_fh = log_path.open("w", buffering=1, encoding="utf-8")

    # Spawn a passthrough thread that timestamps every line from the
    # bridge's stdout (which we capture via a pipe) and writes it to
    # the log file. This way we get ~ms-precision timestamps without
    # modifying any of the bridge's print sites.
    proc = subprocess.Popen(
        [str(venv_py), "-u", str(serve_py)],   # -u: unbuffered Python
        env=env,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,                              # line-buffered pipe
        text=True,                              # decoded str, not bytes
    )

    import threading
    from datetime import datetime, timezone

    def _pump():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                # Strip trailing newline; we add our own with the prefix.
                line = line.rstrip("\n")
                # LINT-OK-PREFIX-SAFE: timestamp for log-file line prefix, not prompt content.
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
                log_fh.write(f"[{ts}Z] {line}\n")
                log_fh.flush()                  # explicit flush per line
        except Exception as e:                  # noqa: BLE001
            try:
                log_fh.write(f"[pump-error] {type(e).__name__}: {e}\n")
                log_fh.flush()
            except Exception:
                pass
        finally:
            try: log_fh.close()
            except Exception: pass

    pump_t = threading.Thread(target=_pump, name="bridge-log-pump",
                               daemon=True)
    pump_t.start()

    print(f"[bridge_lifecycle] launched pid={proc.pid} "
          f"gguf={gguf_path.name} port={port} log={log_path}",
          flush=True, file=sys.stderr)

    try:
        wait_for_health(f"http://127.0.0.1:{port}", health_timeout_s)
    except BaseException:
        # Bridge failed to come up — kill it so we don't leave an orphan.
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise
    return proc


class BridgeContext:
    """Context manager for clean bridge lifecycle in eval scripts.

    Always SIGKILLs on exit (success OR exception), verifies the kill
    completed, and frees the port. Use this in any eval script that
    needs to swap models or run against a single config — never reach
    for shell pkill again.

    Example::

        with BridgeContext("/path/to/fp16.gguf") as bridge:
            for item in dataset:
                resp = httpx.post(bridge.url + "/v1/chat/completions", ...)
        # bridge is gone here; orphan-free.

        with BridgeContext("/path/to/heterogeneous-quant.gguf") as bridge:
            ...
    """

    def __init__(self,
                 gguf_path: str | Path,
                 port: int = DEFAULT_PORT,
                 log_path: str | Path | None = None,
                 health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S):
        self.gguf_path = Path(gguf_path)
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.health_timeout_s = health_timeout_s
        self.proc: subprocess.Popen | None = None

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def __enter__(self) -> "BridgeContext":
        self.proc = launch_bridge(
            self.gguf_path, port=self.port,
            log_path=self.log_path,
            health_timeout_s=self.health_timeout_s,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        kill_all_bridges()
        return False    # do not suppress exceptions


# ──────────────────────────────────────────────────────────────────────
# CLI: useful for shell-level orchestration without writing Python.
# ──────────────────────────────────────────────────────────────────────


def _cli() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Bridge process lifecycle helper (kill / launch / "
                    "verify). Replaces ad-hoc pkill+sleep+launch.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_kill = sub.add_parser("kill",
                             help="SIGKILL all bridge processes and verify.")
    p_kill.add_argument("--timeout", type=float,
                         default=DEFAULT_KILL_TIMEOUT_S)
    p_launch = sub.add_parser("launch",
                               help="Verified-clean launch of a bridge.")
    p_launch.add_argument("--gguf", required=True,
                           help="path to the GGUF to load")
    p_launch.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_launch.add_argument("--log", default=None,
                           help="bridge stdout/stderr log path")
    p_launch.add_argument("--health-timeout", type=float,
                           default=DEFAULT_HEALTH_TIMEOUT_S)
    p_launch.add_argument("--detach", action="store_true",
                           help="exit immediately after /health is ready; "
                                "leave the bridge running. (Default: hold "
                                "the foreground until SIGINT, then kill.)")
    p_status = sub.add_parser("status",
                               help="Print PIDs of running bridges + "
                                    "/health if any.")
    p_status.add_argument("--port", type=int, default=DEFAULT_PORT)

    args = parser.parse_args()

    if args.cmd == "kill":
        kill_all_bridges(timeout_s=args.timeout)
        print("ok", flush=True)
        return 0

    if args.cmd == "status":
        pids = find_bridge_pids()
        print(f"bridge_pids: {pids}")
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                d = json.loads(r.read())
            print(f"health: {d}")
        except BaseException as e:                    # noqa: BLE001
            print(f"health: unreachable ({e!r})")
        return 0

    if args.cmd == "launch":
        proc = launch_bridge(
            args.gguf, port=args.port, log_path=args.log,
            health_timeout_s=args.health_timeout,
        )
        print(f"pid={proc.pid} url=http://127.0.0.1:{args.port}",
              flush=True)
        if args.detach:
            return 0
        # Foreground hold: wait until SIGINT, then kill.
        try:
            print("Foreground hold; press Ctrl-C to stop.", flush=True)
            proc.wait()
        except KeyboardInterrupt:
            pass
        finally:
            kill_all_bridges()
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
