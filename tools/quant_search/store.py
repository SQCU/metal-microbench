"""Pareto store for quantization-search results.

SQLite-backed because the result set is small enough that we don't need
parquet, and SQLite gives us free indexed lookup, transactional writes,
and round-trip verification via `sqlite3` CLI for inspection.

Schema:
    configs(config_hash TEXT PRIMARY KEY, config_json TEXT,
             kernel_version TEXT, created_at REAL)
    metrics(config_hash TEXT, metric TEXT, activeB INTEGER,
            value REAL, raw_json TEXT, measured_at REAL,
            PRIMARY KEY(config_hash, metric, activeB))

Configs are immutable; metrics accumulate (multiple harnesses can score
the same config independently). The (config_hash, metric, activeB) primary
key on metrics means re-runs of the same harness on the same config update
in place — protecting against accidental duplicate evals.

Pareto pruning is computed on demand from the metrics table; we don't
maintain a "frontier" view because the frontier shifts as new configs
land and as kernels change (which invalidates tok/s rows for older
kernel_versions).
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS configs (
    config_hash    TEXT PRIMARY KEY,
    config_json    TEXT NOT NULL,
    kernel_version TEXT NOT NULL,
    created_at     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    config_hash  TEXT NOT NULL,
    metric       TEXT NOT NULL,
    activeB      INTEGER NOT NULL,
    value        REAL NOT NULL,
    raw_json     TEXT,
    measured_at  REAL NOT NULL,
    PRIMARY KEY(config_hash, metric, activeB),
    FOREIGN KEY(config_hash) REFERENCES configs(config_hash)
);
CREATE INDEX IF NOT EXISTS idx_metrics_metric ON metrics(metric, activeB);
"""


def hash_config(config: dict) -> str:
    """Stable hash of a quantization config dict. Sort keys so reordering
    doesn't produce a new hash. SHA-1 truncated to 16 hex chars (collision
    probability negligible at our scale)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def upsert_config(self, config: dict, kernel_version: str) -> str:
        """Insert config if new; return its hash. No-op if already present.
        kernel_version is a free-form tag (e.g., "v11+v11" for the current
        production state); used to know when tok/s metrics are stale."""
        h = hash_config(config)
        with self._tx() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO configs (config_hash, config_json, kernel_version, created_at) "
                "VALUES (?, ?, ?, ?)",
                (h, json.dumps(config, sort_keys=True), kernel_version, time.time()),
            )
        return h

    def record_metric(
        self,
        config_hash: str,
        metric: str,
        activeB: int,
        value: float,
        raw: dict | None = None,
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO metrics (config_hash, metric, activeB, value, raw_json, measured_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (config_hash, metric, activeB, value,
                 json.dumps(raw) if raw else None, time.time()),
            )

    def get_metric(self, config_hash: str, metric: str, activeB: int) -> float | None:
        cur = self._conn.execute(
            "SELECT value FROM metrics WHERE config_hash=? AND metric=? AND activeB=?",
            (config_hash, metric, activeB),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def all_metrics_for(self, config_hash: str) -> dict[tuple[str, int], float]:
        cur = self._conn.execute(
            "SELECT metric, activeB, value FROM metrics WHERE config_hash=?",
            (config_hash,),
        )
        return {(m, b): v for m, b, v in cur.fetchall()}

    def configs_with_kernel(self, kernel_version: str) -> list[tuple[str, dict]]:
        cur = self._conn.execute(
            "SELECT config_hash, config_json FROM configs WHERE kernel_version=?",
            (kernel_version,),
        )
        return [(h, json.loads(j)) for h, j in cur.fetchall()]

    def pareto_frontier(
        self,
        kernel_version: str,
        x_metric: str = "kl_mean",
        y_metric: str = "tok_s",
        x_activeB: int = 0,
        y_activeB: int = 8,
    ) -> list[dict]:
        """Compute the Pareto frontier over (x_metric, y_metric) where
        we want LOW x and HIGH y (e.g., low KL, high tok/s).

        Returns a list of dicts with {config_hash, config, x, y}, sorted
        by x ascending (frontier order). Configs lacking either metric
        are skipped silently.
        """
        cur = self._conn.execute(
            """
            SELECT c.config_hash, c.config_json, mx.value, my.value
            FROM configs c
            JOIN metrics mx ON mx.config_hash=c.config_hash AND mx.metric=? AND mx.activeB=?
            JOIN metrics my ON my.config_hash=c.config_hash AND my.metric=? AND my.activeB=?
            WHERE c.kernel_version=?
            """,
            (x_metric, x_activeB, y_metric, y_activeB, kernel_version),
        )
        points = [
            {"config_hash": h, "config": json.loads(j), "x": x, "y": y}
            for h, j, x, y in cur.fetchall()
        ]
        # Pareto: keep p iff there's no other p' with x' <= x AND y' >= y AND
        # at least one strict. Sort by x asc, then sweep keeping running-max y.
        points.sort(key=lambda p: (p["x"], -p["y"]))
        frontier = []
        best_y = float("-inf")
        for p in points:
            if p["y"] > best_y:
                frontier.append(p)
                best_y = p["y"]
        return frontier
