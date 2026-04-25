"""
db_cache.py — Persistent SQLite-backed JSON cache with TTL.

Used to cache expensive API responses (valuation, screener, insider) so we
don't slam yfinance / EDGAR / FRED on every page load. Survives restarts
(unlike the in-memory cache.py).

Storage: a single SQLite file at $TRADING_CACHE_DB or ./cache/api_cache.db
Key:     SHA-256 of "<namespace>|<ticker>|<canonical-json-of-params>"
Value:   JSON-serialized response payload
TTL:     per-call, stored as expires_at epoch

Public API:
    get(namespace, ticker, params)        -> dict | None
    set(namespace, ticker, params, value, ttl_seconds) -> None
    invalidate(namespace, ticker)         -> int  (rows deleted)
    purge_expired()                       -> int  (rows deleted)
    stats()                               -> dict
    clear()                               -> None
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ─── Storage location ─────────────────────────────────────────────────────
_DEFAULT_PATH = Path(__file__).parent / "cache" / "api_cache.db"
_DB_PATH = Path(os.environ.get("TRADING_CACHE_DB", _DEFAULT_PATH))
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None


def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(
            _DB_PATH,
            check_same_thread=False,
            isolation_level=None,   # autocommit
            timeout=10.0,
        )
        _CONN.execute("PRAGMA journal_mode = WAL;")
        _CONN.execute("PRAGMA synchronous = NORMAL;")
        _CONN.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                key         TEXT PRIMARY KEY,
                namespace   TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                value_json  TEXT NOT NULL,
                created_at  INTEGER NOT NULL,
                expires_at  INTEGER NOT NULL
            )
        """)
        _CONN.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_cache_expires
            ON api_cache(expires_at)
        """)
        _CONN.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_cache_ns_ticker
            ON api_cache(namespace, ticker)
        """)
    return _CONN


def _make_key(namespace: str, ticker: str, params: dict | None) -> str:
    canon = json.dumps(params or {}, sort_keys=True, default=str)
    raw = f"{namespace}|{ticker.upper()}|{canon}".encode()
    return hashlib.sha256(raw).hexdigest()


# ─── Public API ───────────────────────────────────────────────────────────

def get(namespace: str, ticker: str, params: dict | None = None) -> Any | None:
    """Return cached value, or None if missing/expired."""
    key = _make_key(namespace, ticker, params)
    now = int(time.time())
    with _LOCK:
        try:
            row = _conn().execute(
                "SELECT value_json, expires_at FROM api_cache WHERE key = ?",
                (key,),
            ).fetchone()
        except Exception as e:
            log.warning("db_cache get failed: %s", e)
            return None
    if not row:
        return None
    value_json, expires_at = row
    if expires_at <= now:
        return None
    try:
        return json.loads(value_json)
    except Exception:
        return None


def set(namespace: str, ticker: str, params: dict | None,
        value: Any, ttl_seconds: int) -> None:
    """Store a value with TTL. Silently no-ops on serialization failure."""
    try:
        value_json = json.dumps(value, default=str)
    except Exception as e:
        log.warning("db_cache set: value not JSON-serializable: %s", e)
        return
    key = _make_key(namespace, ticker, params)
    now = int(time.time())
    expires = now + int(ttl_seconds)
    with _LOCK:
        try:
            _conn().execute("""
                INSERT INTO api_cache
                    (key, namespace, ticker, value_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
            """, (key, namespace, ticker.upper(), value_json, now, expires))
        except Exception as e:
            log.warning("db_cache set failed: %s", e)


def invalidate(namespace: str | None = None,
               ticker: str | None = None) -> int:
    """Delete entries matching namespace/ticker. Both optional."""
    sql = "DELETE FROM api_cache WHERE 1=1"
    args: list = []
    if namespace:
        sql += " AND namespace = ?"; args.append(namespace)
    if ticker:
        sql += " AND ticker = ?";    args.append(ticker.upper())
    with _LOCK:
        cur = _conn().execute(sql, args)
        return cur.rowcount or 0


def purge_expired() -> int:
    """Delete expired rows. Safe to call periodically."""
    now = int(time.time())
    with _LOCK:
        cur = _conn().execute("DELETE FROM api_cache WHERE expires_at <= ?", (now,))
        return cur.rowcount or 0


def stats() -> dict:
    now = int(time.time())
    with _LOCK:
        c = _conn()
        total = c.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
        live  = c.execute(
            "SELECT COUNT(*) FROM api_cache WHERE expires_at > ?", (now,)
        ).fetchone()[0]
        by_ns = c.execute("""
            SELECT namespace, COUNT(*) FROM api_cache
            WHERE expires_at > ? GROUP BY namespace
        """, (now,)).fetchall()
    return {
        "db_path": str(_DB_PATH),
        "rows_total": total,
        "rows_live": live,
        "rows_expired": total - live,
        "by_namespace": dict(by_ns),
    }


def clear() -> None:
    """Wipe the cache table entirely."""
    with _LOCK:
        _conn().execute("DELETE FROM api_cache")
