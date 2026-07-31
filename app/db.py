"""SQLite persistence for batches and per-file results. Stdlib only."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

DB_PATH = Path(os.environ.get("DATA_DIR", "data")) / "autoace.db"
_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
  id TEXT PRIMARY KEY,
  name TEXT,
  created_at REAL,
  status TEXT,              -- validating | processing | done
  total INTEGER DEFAULT 0,
  warnings TEXT DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id TEXT,
  filename TEXT,
  path TEXT,
  status TEXT,              -- queued | processing | done | failed
  error TEXT,
  result_json TEXT,
  detail_json TEXT,         -- features / llm / trace / overlap / ser
  expected_json TEXT,
  duration_s REAL,
  wall_s REAL,
  cost_usd REAL
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    return con


_con = None


def con() -> sqlite3.Connection:
    global _con
    if _con is None:
        _con = connect()
    return _con


def create_batch(name: str, warnings: list[str]) -> str:
    bid = uuid.uuid4().hex[:12]
    with _lock:
        con().execute(
            "INSERT INTO batches (id, name, created_at, status, warnings) VALUES (?,?,?,?,?)",
            (bid, name, time.time(), "processing", json.dumps(warnings)))
        con().commit()
    return bid


def add_file(batch_id: str, filename: str, path: str, expected_json: str | None,
             status: str = "queued", error: str | None = None) -> int:
    with _lock:
        cur = con().execute(
            "INSERT INTO files (batch_id, filename, path, status, error, expected_json) "
            "VALUES (?,?,?,?,?,?)",
            (batch_id, filename, path, status, error, expected_json))
        con().execute("UPDATE batches SET total = total + 1 WHERE id = ?", (batch_id,))
        con().commit()
        return cur.lastrowid


def set_file_status(file_id: int, status: str, **fields) -> None:
    cols = ", ".join(f"{k} = ?" for k in fields)
    sql = f"UPDATE files SET status = ?{', ' + cols if cols else ''} WHERE id = ?"
    with _lock:
        con().execute(sql, (status, *fields.values(), file_id))
        con().commit()


def finish_batch_if_done(batch_id: str) -> None:
    with _lock:
        row = con().execute(
            "SELECT COUNT(*) AS pending FROM files WHERE batch_id = ? "
            "AND status IN ('queued', 'processing')", (batch_id,)).fetchone()
        if row["pending"] == 0:
            con().execute("UPDATE batches SET status = 'done' WHERE id = ?", (batch_id,))
        con().commit()


def batch_summary(batch_id: str) -> dict | None:
    b = con().execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    if not b:
        return None
    counts = {r["status"]: r["n"] for r in con().execute(
        "SELECT status, COUNT(*) AS n FROM files WHERE batch_id = ? GROUP BY status",
        (batch_id,))}
    return {**dict(b), "warnings": json.loads(b["warnings"]), "counts": counts}


def batch_files(batch_id: str) -> list[dict]:
    return [dict(r) for r in con().execute(
        "SELECT id, filename, status, error, result_json, expected_json, duration_s, "
        "wall_s, cost_usd FROM files WHERE batch_id = ? ORDER BY filename", (batch_id,))]


def file_detail(file_id: int) -> dict | None:
    r = con().execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    return dict(r) if r else None


def delete_batch(batch_id: str) -> bool:
    """Drop a batch and its file rows. Returns False if it never existed."""
    with _lock:
        cur = con().execute("SELECT 1 FROM batches WHERE id = ?", (batch_id,))
        if cur.fetchone() is None:
            return False
        con().execute("DELETE FROM files WHERE batch_id = ?", (batch_id,))
        con().execute("DELETE FROM batches WHERE id = ?", (batch_id,))
        con().commit()
        return True


def list_batches(limit: int = 50) -> list[dict]:
    out = []
    for b in con().execute(
            "SELECT id FROM batches ORDER BY created_at DESC LIMIT ?", (limit,)):
        out.append(batch_summary(b["id"]))
    return out
