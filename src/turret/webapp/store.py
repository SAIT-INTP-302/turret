"""Tiny SQLite event log for the dashboard.

Kept dependency-free (stdlib sqlite3 only) so it runs on the Pi with nothing
extra to install beyond Flask for the server itself.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_PATH = Path("turret_events.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,        -- "sighting" | "fired"
    cx INTEGER,
    cy INTEGER,
    area REAL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts DESC);
"""


class EventStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log(
        self,
        kind: str,
        *,
        cx: int | None = None,
        cy: int | None = None,
        area: float | None = None,
        note: str | None = None,
        ts: float | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (ts, kind, cx, cy, area, note) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts if ts is not None else time.time(), kind, cx, cy, area, note),
            )

    def recent(self, limit: int = 100, since: float | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if since is not None:
                rows = conn.execute(
                    "SELECT * FROM events WHERE ts > ? ORDER BY ts DESC LIMIT ?",
                    (since, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) AS n FROM events GROUP BY kind"
            ).fetchall()
            return {r["kind"]: r["n"] for r in rows}
