import base64
import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from app.config import DATA_DIR

AUTH_DB_PATH = DATA_DIR / "auth.db"
PBKDF2_ITERATIONS = 200_000
SESSION_TTL_DAYS = 7

ROLE_SUPER_ADMIN = "adminD"
ROLE_ADMIN = "admin"
ROLE_USER = "user"
ADMIN_ROLES = {ROLE_SUPER_ADMIN, ROLE_ADMIN}
VALID_ROLES = {ROLE_SUPER_ADMIN, ROLE_ADMIN, ROLE_USER}
VALID_DEPARTMENTS = {"RD", "MD", "OD"}
_BOOTSTRAP_LOCK = Lock()

DEFAULT_USERS = (
    {"username": "admin", "password": "admin123", "role": ROLE_ADMIN, "department": "ALL"},
    {"username": "rd", "password": "123qwe", "role": ROLE_USER, "department": "RD"},
    {"username": "md", "password": "123qwe", "role": ROLE_USER, "department": "MD"},
    {"username": "od", "password": "123qwe", "role": ROLE_USER, "department": "OD"},
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_role(role: Optional[str]) -> str:
    normalized = (role or "").strip()
    lowered = normalized.lower()

    if normalized in VALID_ROLES:
        return normalized
    if lowered == "admind":
        return ROLE_SUPER_ADMIN
    if lowered == "admin":
        return ROLE_ADMIN
    if lowered == "user":
        return ROLE_USER
    if normalized == "관리자":
        return ROLE_ADMIN
    if normalized == "직원":
        return ROLE_USER
    return ROLE_USER


def is_admin_role(role: Optional[str]) -> bool:
    return _normalize_role(role) in ADMIN_ROLES


def _normalize_department(department: Optional[str], role: Optional[str]) -> str:
    normalized_role = _normalize_role(role)
    normalized_department = (department or "").strip().upper()
    if normalized_role in ADMIN_ROLES:
        return "ALL"
    if normalized_department not in VALID_DEPARTMENTS:
        return "RD"
    return normalized_department


def _serialize_user(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "role": _normalize_role(row["role"]),
        "department": _normalize_department(row["department"], row["role"]),
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    raw_password = password or ""
    raw_salt = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        raw_password.encode("utf-8"),
        raw_salt,
        PBKDF2_ITERATIONS,
    )
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.b64encode(raw_salt).decode("ascii"),
        base64.b64encode(derived).decode("ascii"),
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt_b64, digest_b64 = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
    except Exception:
        return False

    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        (password or "").encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate, expected)


def _hash_session_token(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def _bootstrap(conn: sqlite3.Connection) -> None:
    now = _utcnow().isoformat()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            department TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )

    conn.execute("UPDATE users SET role = ? WHERE role = ?", (ROLE_ADMIN, "관리자"))
    conn.execute("UPDATE users SET role = ? WHERE role = ?", (ROLE_USER, "직원"))
    conn.execute(
        "UPDATE users SET role = ?, department = 'ALL' WHERE lower(role) = 'admind'",
        (ROLE_SUPER_ADMIN,),
    )
    conn.execute(
        "UPDATE users SET role = ?, department = 'ALL' WHERE lower(role) = 'admin'",
        (ROLE_ADMIN,),
    )
    conn.execute(
        "UPDATE users SET role = ? WHERE lower(role) = 'user'",
        (ROLE_USER,),
    )
    conn.execute(
        "UPDATE users SET department = 'ALL' WHERE role IN (?, ?)",
        (ROLE_SUPER_ADMIN, ROLE_ADMIN),
    )
    conn.execute(
        "UPDATE users SET department = 'RD' WHERE role = ? AND department NOT IN ('RD', 'MD', 'OD')",
        (ROLE_USER,),
    )

    for user in DEFAULT_USERS:
        conn.execute(
            """
            INSERT OR IGNORE INTO users (username, password_hash, role, department, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            (
                user["username"],
                hash_password(user["password"]),
                user["role"],
                user["department"],
                now,
                now,
            ),
        )

    conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    with _BOOTSTRAP_LOCK:
        _bootstrap(conn)
        conn.commit()
    return conn


def init_auth_store() -> None:
    with _connect() as conn:
        conn.commit()


def list_users() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, username, role, department, is_active, created_at, updated_at
            FROM users
            ORDER BY
                CASE
                    WHEN role = 'adminD' THEN 0
                    WHEN role = 'admin' THEN 1
                    ELSE 2
                END,
                username
            """
        ).fetchall()
    return [_serialize_user(row) for row in rows]


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    normalized = (username or "").strip()
    if not normalized:
        return None
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, username, role, department, is_active, created_at, updated_at
            FROM users
            WHERE username = ?
            """,
            (normalized,),
        ).fetchone()
    return _serialize_user(row) if row else None


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    normalized = (username or "").strip()
    if not normalized:
        return None
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (normalized,)).fetchone()
    if not row or not row["is_active"]:
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return _serialize_user(row)


def create_session(username: str, ttl_days: int = SESSION_TTL_DAYS) -> str:
    user = get_user_by_username(username)
    if not user:
        raise ValueError("User not found.")

    raw_token = secrets.token_urlsafe(32)
    now = _utcnow()
    expires_at = now + timedelta(days=ttl_days)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, user_id, created_at, last_seen_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                _hash_session_token(raw_token),
                user["id"],
                now.isoformat(),
                now.isoformat(),
                expires_at.isoformat(),
            ),
        )
        conn.commit()
    return raw_token


def get_user_by_session_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None

    token_hash = _hash_session_token(token)
    now = _utcnow().isoformat()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                users.id,
                users.username,
                users.role,
                users.department,
                users.is_active,
                users.created_at,
                users.updated_at
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.session_id = ?
            """,
            (token_hash,),
        ).fetchone()

        if not row:
            return None

        session_row = conn.execute(
            "SELECT expires_at FROM sessions WHERE session_id = ?",
            (token_hash,),
        ).fetchone()
        if not session_row or session_row["expires_at"] <= now:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (token_hash,))
            conn.commit()
            return None

        if not row["is_active"]:
            return None

        conn.execute(
            "UPDATE sessions SET last_seen_at = ? WHERE session_id = ?",
            (now, token_hash),
        )
        conn.commit()

    return _serialize_user(row)


def delete_session(token: str) -> None:
    if not token:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (_hash_session_token(token),))
        conn.commit()


def create_user(username: str, password: str, role: str, department: Optional[str]) -> Dict[str, Any]:
    normalized_username = (username or "").strip()
    if not normalized_username:
        raise ValueError("아이디를 입력해주세요.")
    if len(normalized_username) < 2:
        raise ValueError("아이디는 2자 이상이어야 합니다.")
    if not password or len(password) < 4:
        raise ValueError("비밀번호는 4자 이상이어야 합니다.")

    normalized_role = _normalize_role(role)
    if normalized_role not in VALID_ROLES:
        raise ValueError("올바른 권한이 아닙니다.")
    normalized_department = _normalize_department(department, normalized_role)
    now = _utcnow().isoformat()

    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (normalized_username,),
        ).fetchone()
        if existing:
            raise ValueError("이미 존재하는 계정입니다.")

        conn.execute(
            """
            INSERT INTO users (username, password_hash, role, department, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            (
                normalized_username,
                hash_password(password),
                normalized_role,
                normalized_department,
                now,
                now,
            ),
        )
        conn.commit()

    created = get_user_by_username(normalized_username)
    if not created:
        raise ValueError("계정 생성에 실패했습니다.")
    return created


def _privileged_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM users WHERE role IN (?, ?) AND is_active = 1",
        (ROLE_SUPER_ADMIN, ROLE_ADMIN),
    ).fetchone()
    return int(row["cnt"]) if row else 0


def delete_user(username: str) -> None:
    normalized = (username or "").strip()
    if not normalized:
        raise ValueError("삭제할 계정이 없습니다.")

    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (normalized,)).fetchone()
        if not row:
            raise ValueError("계정을 찾을 수 없습니다.")
        if _normalize_role(row["role"]) in ADMIN_ROLES and _privileged_count(conn) <= 1:
            raise ValueError("마지막 관리자 계정은 삭제할 수 없습니다.")

        conn.execute("DELETE FROM users WHERE username = ?", (normalized,))
        conn.commit()


def reset_password(username: str) -> str:
    normalized = (username or "").strip()
    if not normalized:
        raise ValueError("계정을 찾을 수 없습니다.")

    user = get_user_by_username(normalized)
    if not user:
        raise ValueError("계정을 찾을 수 없습니다.")

    password = "admin123" if user["role"] in ADMIN_ROLES and normalized == "admin" else "123qwe"
    update_password(normalized, password)
    return password


def update_password(username: str, new_password: str) -> None:
    normalized = (username or "").strip()
    if not normalized:
        raise ValueError("계정을 찾을 수 없습니다.")
    if not new_password or len(new_password) < 4:
        raise ValueError("비밀번호는 4자 이상이어야 합니다.")

    now = _utcnow().isoformat()
    with _connect() as conn:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (normalized,)).fetchone()
        if not row:
            raise ValueError("계정을 찾을 수 없습니다.")

        conn.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE username = ?",
            (hash_password(new_password), now, normalized),
        )
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
        conn.commit()


def change_password(username: str, current_password: str, new_password: str) -> None:
    normalized = (username or "").strip()
    if not normalized:
        raise ValueError("계정을 찾을 수 없습니다.")
    if not new_password or len(new_password) < 4:
        raise ValueError("비밀번호는 4자 이상이어야 합니다.")

    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (normalized,)).fetchone()
        if not row:
            raise ValueError("계정을 찾을 수 없습니다.")
        if not verify_password(current_password, row["password_hash"]):
            raise ValueError("현재 비밀번호가 올바르지 않습니다.")

    update_password(normalized, new_password)
