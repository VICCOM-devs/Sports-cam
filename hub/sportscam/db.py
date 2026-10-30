"""SQLite storage. Small, synchronous, one connection per call – plenty for one rink."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
  id TEXT PRIMARY KEY,
  code TEXT UNIQUE NOT NULL,
  sport TEXT NOT NULL,
  home TEXT NOT NULL,
  away TEXT NOT NULL,
  home_color TEXT NOT NULL,
  away_color TEXT NOT NULL,
  status TEXT NOT NULL,            -- lobby | live | ended
  period INTEGER NOT NULL DEFAULT 1,
  home_score INTEGER NOT NULL DEFAULT 0,
  away_score INTEGER NOT NULL DEFAULT 0,
  clock_running INTEGER NOT NULL DEFAULT 0,
  clock_base_ms INTEGER NOT NULL DEFAULT 0,   -- elapsed ms when clock last stopped
  clock_started_at REAL,                      -- wall time when clock last started
  created_at REAL NOT NULL,
  started_at REAL,
  ended_at REAL
);
CREATE TABLE IF NOT EXISTS tokens (
  token TEXT PRIMARY KEY,
  match_id TEXT NOT NULL,
  role TEXT NOT NULL,              -- admin | scorekeeper | camera | viewer
  label TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS cameras (
  id TEXT PRIMARY KEY,
  match_id TEXT NOT NULL,
  label TEXT NOT NULL,
  position TEXT,
  kind TEXT NOT NULL DEFAULT 'phone',   -- phone | rink | capture
  status TEXT NOT NULL DEFAULT 'joined', -- joined | live | offline
  battery REAL,
  signal TEXT,
  joined_at REAL NOT NULL,
  last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id TEXT NOT NULL,
  t_ms INTEGER NOT NULL,           -- game clock ms
  wall REAL NOT NULL,              -- wall time, for lining up recordings
  kind TEXT NOT NULL,              -- goal | shot | save | hit | penalty | highlight | period | score
  team TEXT,                       -- home | away | null
  payload TEXT,                    -- json
  by_token TEXT
);
CREATE INDEX IF NOT EXISTS events_match ON events(match_id, id);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path, timeout=5)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # -- matches -------------------------------------------------------------
    def insert_match(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self.conn() as c:
            c.execute(f"INSERT INTO matches ({cols}) VALUES ({qs})", list(row.values()))

    def match_by_code(self, code: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM matches WHERE code=?", (code,)).fetchone()
            return dict(r) if r else None

    def match(self, match_id: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM matches WHERE id=?", (match_id,)).fetchone()
            return dict(r) if r else None

    def update_match(self, match_id: str, **fields: Any) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE matches SET {sets} WHERE id=?", [*fields.values(), match_id])

    def recent_matches(self, limit: int = 20) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM matches ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # -- tokens --------------------------------------------------------------
    def insert_token(self, token: str, match_id: str, role: str, label: str | None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO tokens (token, match_id, role, label, created_at) VALUES (?,?,?,?,?)",
                (token, match_id, role, label, time.time()),
            )

    def token(self, token: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM tokens WHERE token=?", (token,)).fetchone()
            return dict(r) if r else None

    # -- cameras -------------------------------------------------------------
    def upsert_camera(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "id")
        with self.conn() as c:
            c.execute(
                f"INSERT INTO cameras ({cols}) VALUES ({qs}) ON CONFLICT(id) DO UPDATE SET {updates}",
                list(row.values()),
            )

    def cameras(self, match_id: str) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM cameras WHERE match_id=? ORDER BY joined_at", (match_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def camera(self, cam_id: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
            return dict(r) if r else None

    def update_camera(self, cam_id: str, **fields: Any) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE cameras SET {sets} WHERE id=?", [*fields.values(), cam_id])

    # -- events --------------------------------------------------------------
    def insert_event(
        self, match_id: str, t_ms: int, kind: str, team: str | None, payload: dict | None, by: str | None
    ) -> dict:
        wall = time.time()
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO events (match_id, t_ms, wall, kind, team, payload, by_token) VALUES (?,?,?,?,?,?,?)",
                (match_id, t_ms, wall, kind, team, json.dumps(payload or {}), by),
            )
            return {
                "id": cur.lastrowid, "match_id": match_id, "t_ms": t_ms, "wall": wall,
                "kind": kind, "team": team, "payload": payload or {},
            }

    def events(self, match_id: str) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM events WHERE match_id=? ORDER BY id", (match_id,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["payload"] = json.loads(d.pop("payload") or "{}")
                d.pop("by_token", None)
                out.append(d)
            return out
