import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from app.config import PENDING_UPLOADS_DIR, UPLOAD_REQUESTS_PATH

_store_lock = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent() -> None:
    UPLOAD_REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _load_store() -> Dict[str, Any]:
    _ensure_parent()
    if not UPLOAD_REQUESTS_PATH.exists():
        return {"requests": []}

    try:
        return json.loads(UPLOAD_REQUESTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"requests": []}


def _save_store(data: Dict[str, Any]) -> None:
    _ensure_parent()
    UPLOAD_REQUESTS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_visible_departments(values: Optional[List[str]]) -> List[str]:
    cleaned = []
    seen = set()
    for value in values or []:
        raw = str(value or "").strip().upper()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        cleaned.append(raw)
    return cleaned


def create_upload_request(
    *,
    filename: str,
    source_path: Path,
    uploader_id: str,
    department: str,
    visibility: str = "public",
    visible_departments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    request_id = f"req_{uuid.uuid4().hex}"
    now = _utc_now()
    visible_departments = _normalize_visible_departments(visible_departments)
    normalized_visibility = "private" if str(visibility or "").lower() == "private" else "public"

    entry = {
        "request_id": request_id,
        "source": filename,
        "source_path": str(source_path),
        "uploader_id": uploader_id,
        "department": str(department or "").strip().upper() or "미지정",
        "visibility": normalized_visibility,
        "visible_departments": visible_departments if normalized_visibility == "private" else [],
        "access_level": "선택 부서 공개" if normalized_visibility == "private" else "전체 공개",
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "uploaded_at": now,
    }

    with _store_lock:
        store = _load_store()
        store.setdefault("requests", []).append(entry)
        _save_store(store)
    return entry


def list_upload_requests(
    *,
    uploader_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with _store_lock:
        store = _load_store()

    requests = list(store.get("requests", []))
    if uploader_id:
        requests = [item for item in requests if item.get("uploader_id") == uploader_id]
    if status:
        requests = [item for item in requests if item.get("status") == status]
    return requests


def get_upload_request(request_id: str) -> Optional[Dict[str, Any]]:
    with _store_lock:
        store = _load_store()
    return next((item for item in store.get("requests", []) if item.get("request_id") == request_id), None)


def update_upload_request(request_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    safe_updates = {key: value for key, value in (updates or {}).items() if key and value is not None}
    if not safe_updates:
        return get_upload_request(request_id)

    with _store_lock:
        store = _load_store()
        entries = store.get("requests", [])
        target = next((item for item in entries if item.get("request_id") == request_id), None)
        if not target:
            return None
        target.update(safe_updates)
        target["updated_at"] = _utc_now()
        _save_store(store)
        return dict(target)
