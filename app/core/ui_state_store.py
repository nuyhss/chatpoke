"""
Persistent chat UI state storage backed by SQLite.

This replaces per-user JSON files with a small transactional store while
keeping a one-time migration path from the legacy files.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from app.config import DATA_DIR, UI_STATE_DB_PATH

logger = logging.getLogger("tilon.ui_state_store")

_db_lock = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent() -> None:
    UI_STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _legacy_state_path(user_id: Optional[str]) -> Path:
    safe = str(user_id or "").strip()
    if safe:
        return DATA_DIR / "ui_state" / f"chats_{Path(safe).name}.json"
    return DATA_DIR / "ui_state" / "chats_default.json"


def _connect() -> sqlite3.Connection:
    _ensure_parent()
    conn = sqlite3.connect(UI_STATE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ui_chat_state (
            user_id TEXT PRIMARY KEY,
            chats_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def _normalize_user_key(user_id: Optional[str]) -> str:
    value = str(user_id or "").strip()
    return value or "__default__"


def _read_legacy_chats(user_id: Optional[str]) -> Dict[str, Any]:
    path = _legacy_state_path(user_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read legacy UI chat state (%s): %s", path, exc)
        return {}

    chats = payload.get("chats") if isinstance(payload, dict) else {}
    return chats if isinstance(chats, dict) else {}


def _delete_legacy_file(user_id: Optional[str]) -> None:
    path = _legacy_state_path(user_id)
    if not path.exists():
        return
    try:
        path.unlink()
    except Exception as exc:
        logger.warning("Failed to remove legacy UI chat state (%s): %s", path, exc)


def get_chats(user_id: Optional[str] = None) -> Dict[str, Any]:
    user_key = _normalize_user_key(user_id)

    with _db_lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT chats_json FROM ui_chat_state WHERE user_id = ?",
                (user_key,),
            ).fetchone()
            if row:
                payload = json.loads(row["chats_json"])
                return payload if isinstance(payload, dict) else {}

            legacy_chats = _read_legacy_chats(user_id)
            if legacy_chats:
                encoded = json.dumps(legacy_chats, ensure_ascii=False)
                now = _utc_now()
                conn.execute(
                    """
                    INSERT INTO ui_chat_state (user_id, chats_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        chats_json = excluded.chats_json,
                        updated_at = excluded.updated_at
                    """,
                    (user_key, encoded, now, now),
                )
                conn.commit()
                _delete_legacy_file(user_id)
                return legacy_chats
            return {}
        finally:
            conn.close()


def save_chats(chats: Dict[str, Any], user_id: Optional[str] = None) -> None:
    user_key = _normalize_user_key(user_id)
    encoded = json.dumps(chats, ensure_ascii=False)
    now = _utc_now()

    with _db_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO ui_chat_state (user_id, chats_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    chats_json = excluded.chats_json,
                    updated_at = excluded.updated_at
                """,
                (user_key, encoded, now, now),
            )
            conn.commit()
        finally:
            conn.close()
