"""Durable cache and audit log of raw model calls.

Design (docs/decisions.md S1 change 4): the *raw* response is the cached artefact.
Parsing, verification, ID repair and consensus are deterministic functions of it, so they
can be improved and re-run without spending API quota. The cache key covers everything
that can change the answer (model, endpoint, request parameters, full prompt text), so a
prompt or episode edit invalidates exactly the affected calls and nothing else.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def call_key(stage: str, model: str, base_url: str, params: dict, prompt: str) -> str:
    material = json.dumps({"stage": stage, "model": model, "base_url": base_url, "params": params,
                           "prompt": prompt}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class CallCache:
    """Thread-safe SQLite store of raw model responses, keyed by full request content."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS llm_calls (
              cache_key TEXT PRIMARY KEY,
              stage TEXT NOT NULL, series_id TEXT NOT NULL, episode_id TEXT NOT NULL,
              model TEXT NOT NULL, params_json TEXT NOT NULL, prompt_sha TEXT NOT NULL,
              raw TEXT NOT NULL, usage_json TEXT NOT NULL, finish_reason TEXT NOT NULL,
              latency_s REAL NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS llm_calls_episode ON llm_calls(series_id, episode_id, stage);
            """)
            self._db.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM llm_calls WHERE cache_key=?", (key,)).fetchone()
        if row is None:
            return None
        return {"raw": row["raw"], "usage": json.loads(row["usage_json"]), "finish_reason": row["finish_reason"],
                "latency_s": row["latency_s"], "model": row["model"], "created_at": row["created_at"]}

    def put(self, key: str, *, stage: str, series_id: str, episode_id: str, model: str, params: dict,
            prompt: str, raw: str, usage: dict, finish_reason: str, latency_s: float) -> None:
        with self._lock:
            self._db.execute(
                """INSERT OR REPLACE INTO llm_calls(cache_key, stage, series_id, episode_id, model, params_json,
                   prompt_sha, raw, usage_json, finish_reason, latency_s) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (key, stage, series_id, episode_id, model, json.dumps(params, sort_keys=True),
                 hashlib.sha256(prompt.encode("utf-8")).hexdigest(), raw, json.dumps(usage), finish_reason, latency_s))
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
