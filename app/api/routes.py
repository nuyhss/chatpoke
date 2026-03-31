"""
Core API routes — /chat, /ingest, /health, /docs-list, etc.

IMPROVEMENTS over original:
- Routes separated from business logic
- Response models for type safety
- Better error messages
- Health check includes more diagnostics
"""

import json
import logging
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body, Depends
from starlette.concurrency import run_in_threadpool

from app.api.deps import AuthUser, get_current_user, require_admin
from app.config import (
    OLLAMA_MODEL,
    AVAILABLE_MODELS,
    DATA_DIR,
    LIBRARY_DIR,
    UPLOADS_DIR,
    CHROMA_DIR,
    ENABLE_OCR,
    VISION_MODEL,
    WHISPER_MODEL,
    DUCKDUCKGO_REGION,
)
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    CountKeywordRequest,
    WebSearchRequest,
    SourceInfo,
)
from app.core.llm import check_ollama_health
from app.core.stt import transcribe_audio_bytes
from app.core.vision import analyze_image_bytes
from app.core.web_search import search_web
from app.core.document_registry import (
    clear_document_registry,
    get_document,
    list_documents,
    remove_documents,
    update_document,
)
from app.core.watcher import suppress_watcher_for
from app.core.vectorstore import (
    get_vectorstore,
    get_collection_stats,
    get_all_metadata,
    delete_documents,
    reset as reset_vectorstore,
)
from app.chat.handlers import handle_chat
from app.pipeline.ingest import ingest_folder, ingest_single_file
from app.pipeline.parser import extract_full_text

logger = logging.getLogger("tilon.api")

router = APIRouter()
_ui_state_lock = Lock()


def _normalize_user_id(user_id: Optional[str]) -> Optional[str]:
    if user_id is None:
        return None
    raw = str(user_id).strip()
    if not raw:
        return None

    safe = Path(raw).name.strip()
    if not safe or safe in {".", ".."}:
        return None
    return safe


def _ui_chats_state_path(user_id: Optional[str]) -> Path:
    normalized_user_id = _normalize_user_id(user_id)
    state_dir = DATA_DIR / "ui_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    filename = f"chats_{normalized_user_id}.json" if normalized_user_id else "chats_default.json"
    return state_dir / filename


def _resolve_upload_target(filename: str, user_id: Optional[str]) -> Tuple[Path, Optional[str], str]:
    safe_filename = Path(filename or "").name
    if not safe_filename:
        raise HTTPException(status_code=400, detail="유효한 파일명이 필요합니다.")

    normalized_user_id = _normalize_user_id(user_id)
    target_dir = UPLOADS_DIR / normalized_user_id if normalized_user_id else UPLOADS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    return target_dir / safe_filename, normalized_user_id, safe_filename


def _coerce_department(value: Optional[str], fallback: Optional[str]) -> str:
    raw = (value or fallback or "").strip()
    return raw or "미지정"


def _normalize_department(value: Optional[str]) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    safe = Path(raw).name.strip()
    if not safe or safe in {".", ".."}:
        return None
    return safe.upper()


def _coerce_access_level(value: Optional[str]) -> str:
    raw = (value or "").strip()
    return raw or "전체 공개"


def _normalize_visibility(value: Optional[str], fallback_department: Optional[str] = None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"public", "private"}:
        return raw
    normalized_department = _normalize_department(fallback_department)
    return "private" if normalized_department and normalized_department != "ALL" else "public"


def _normalize_visible_departments(
    value: Optional[str],
    fallback_department: Optional[str] = None,
    visibility: Optional[str] = None,
) -> List[str]:
    if isinstance(value, str):
        items = [item.strip().upper() for item in value.split(",")]
    else:
        items = []
    normalized = [
        item for item in items
        if item and item not in {".", "..", "ALL"}
    ]
    if normalized:
        return normalized

    normalized_department = _normalize_department(fallback_department)
    if visibility == "private" and normalized_department and normalized_department != "ALL":
        return [normalized_department]
    return []


def _normalize_scope_rules(value: Optional[str]) -> List[dict]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        raw_rules = json.loads(value)
    except Exception:
        return []
    if not isinstance(raw_rules, list):
        return []

    normalized_rules = []
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        visibility = _normalize_visibility(item.get("visibility"), item.get("department"))
        department = str(item.get("department") or "").strip()
        if visibility == "public":
            department = "ALL"
        else:
            department = _normalize_department(department) or ""
            if not department or department == "ALL":
                continue
        rank = str(item.get("rank") or "").strip() or "전체"
        normalized_rules.append(
            {
                "visibility": visibility,
                "department": department,
                "rank": rank,
            }
        )
    return normalized_rules


def _build_admin_document_payload(entry: dict) -> dict:
    raw_source_path = str(entry.get("source_path") or "").strip()
    source_path = Path(raw_source_path).expanduser() if raw_source_path else None
    exists = bool(source_path and source_path.exists() and source_path.is_file())
    stat = source_path.stat() if exists and source_path else None
    raw_status = str(entry.get("status") or "").strip().lower()
    is_approved = raw_status == "approved"
    if is_approved:
        status = "승인 완료"
        status_category = "approved"
    elif raw_status in {"ingested", "pending"}:
        status = "승인 대기"
        status_category = "pending"
    else:
        status = entry.get("status") or "대기"
        status_category = "pending"

    visibility = _normalize_visibility(entry.get("visibility"), entry.get("department"))
    visible_departments = _normalize_visible_departments(
        entry.get("visible_departments"),
        fallback_department=entry.get("department"),
        visibility=visibility,
    )

    return {
        "doc_id": entry.get("doc_id"),
        "source": entry.get("source"),
        "source_type": entry.get("source_type"),
        "source_path": str(source_path) if source_path else "",
        "file_exists": exists,
        "file_size_bytes": stat.st_size if stat else None,
        "uploaded_at": entry.get("uploaded_at") or entry.get("created_at"),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "approved_at": entry.get("approved_at"),
        "approved_by": entry.get("approved_by"),
        "page_total": entry.get("page_total"),
        "chunk_count": entry.get("chunk_count", 0),
        "status": status,
        "status_category": status_category,
        "status_raw": entry.get("status"),
        "owner_id": entry.get("owner_id"),
        "uploader_id": entry.get("uploader_id"),
        "department": _coerce_department(entry.get("department"), entry.get("owner_id")),
        "access_level": _coerce_access_level(entry.get("access_level")),
        "department_only": bool(entry.get("department_only")),
        "visibility": visibility,
        "visible_departments": visible_departments,
        "scope_rules": entry.get("scope_rules") or [],
        "languages": entry.get("languages") or [],
        "extractors_used": entry.get("extractors_used") or [],
        "input_type": entry.get("input_type"),
    }


def _delete_managed_document(
    source: Optional[str] = None,
    doc_id: Optional[str] = None,
    source_type: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> dict:
    if not source and not doc_id:
        raise HTTPException(status_code=400, detail="source 또는 doc_id 중 하나는 필요합니다.")

    normalized_owner_id = _normalize_user_id(owner_id)
    source_candidates: List[Optional[str]] = []
    if source:
        source_candidates.append(source)
        safe_source = Path(source).name
        if safe_source and safe_source != source:
            source_candidates.append(safe_source)

    attempts: List[tuple[Optional[str], Optional[str], Optional[str]]] = []
    seen = set()

    def add_attempt(s: Optional[str], d: Optional[str], st: Optional[str]):
        if not s and not d and not st:
            return
        key = (s or "", d or "", st or "")
        if key in seen:
            return
        seen.add(key)
        attempts.append((s, d, st))

    source_types: List[Optional[str]] = [source_type] if source_type else ["upload", "library", None]

    for s in source_candidates or [None]:
        for st in source_types:
            add_attempt(s, doc_id, st)

    if doc_id:
        for st in source_types:
            add_attempt(None, doc_id, st)

    deleted_chunks = 0
    removed_registry = 0
    exact_paths: List[Path] = []
    if doc_id:
        registry_doc = get_document(doc_id)
        registry_source_path = str((registry_doc or {}).get("source_path") or "").strip()
        if registry_source_path:
            exact_paths.append(Path(registry_source_path))
    for s, d, st in attempts:
        scoped_owner_id = normalized_owner_id if st == "upload" else None
        deleted_chunks += delete_documents(source=s, doc_id=d, source_type=st, owner_id=scoped_owner_id)
        removed_registry += remove_documents(source=s, doc_id=d, source_type=st, owner_id=scoped_owner_id)

    file_deleted = False
    for exact_path in exact_paths:
        if exact_path.exists() and exact_path.is_file():
            exact_path.unlink()
            file_deleted = True

    candidate_dirs: List[Path] = [LIBRARY_DIR]
    if normalized_owner_id:
        candidate_dirs.append(UPLOADS_DIR / normalized_owner_id)
    candidate_dirs.append(UPLOADS_DIR)

    for s in source_candidates:
        if not s:
            continue
        safe_name = Path(s).name
        for base_dir in candidate_dirs:
            file_path = base_dir / safe_name
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                file_deleted = True

    if deleted_chunks == 0 and removed_registry == 0 and not file_deleted:
        raise HTTPException(status_code=404, detail="삭제할 문서를 찾지 못했습니다.")

    return {
        "message": "문서가 삭제되었습니다.",
        "deleted_chunks": deleted_chunks,
        "removed_registry": removed_registry,
        "file_deleted": file_deleted,
        "source": source,
        "doc_id": doc_id,
        "source_type": source_type,
        "attempts": len(attempts),
        "user_id": normalized_owner_id,
    }


# ── Root ───────────────────────────────────────────────────────────────

@router.get("/")
def root():
    return {
        "message": "Tilon AI Chatbot API is running",
        "version": "7.0.0",
        "model": OLLAMA_MODEL,
        "data_dir": str(DATA_DIR),
        "library_dir": str(LIBRARY_DIR),
        "uploads_dir": str(UPLOADS_DIR),
        "chroma_dir": str(CHROMA_DIR),
        "ocr_enabled": ENABLE_OCR,
    }


# ── Health ─────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    try:
        ollama_status = check_ollama_health()
        stats = get_collection_stats()

        return {
            "status": "ok",
            "ollama": ollama_status["status"],
            "model": OLLAMA_MODEL,
            "available_models": AVAILABLE_MODELS,
            "documents_in_vectorstore": stats["total_chunks"],
            "ocr_enabled": ENABLE_OCR,
            "vision_model": VISION_MODEL,
            "stt_model": WHISPER_MODEL,
            "web_search_provider": "tavily_or_duckduckgo",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


# ── Available Models ──────────────────────────────────────────────────

@router.get("/models")
def list_models():
    """Return available Ollama models for the UI model selector."""
    return {
        "default": OLLAMA_MODEL,
        "available": AVAILABLE_MODELS,
    }


# ── UI State (Chat History Persistence) ───────────────────────────────

@router.get("/ui-state/chats")
def get_ui_state_chats(current_user: AuthUser = Depends(get_current_user)):
    state_path = _ui_chats_state_path(current_user.username)

    if not state_path.exists():
        return {"chats": {}}

    with _ui_state_lock:
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read UI chat state (%s): %s", state_path, e)
            return {"chats": {}}

    chats = payload.get("chats") if isinstance(payload, dict) else {}
    if not isinstance(chats, dict):
        chats = {}

    return {"chats": chats}


@router.put("/ui-state/chats")
def put_ui_state_chats(
    payload: dict = Body(...),
    current_user: AuthUser = Depends(get_current_user),
):
    chats = payload.get("chats") if isinstance(payload, dict) else None
    if not isinstance(chats, dict):
        raise HTTPException(status_code=400, detail="'chats' must be an object.")

    state_path = _ui_chats_state_path(current_user.username)
    doc = {"chats": chats}

    try:
        encoded = json.dumps(doc, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid chats payload: {e}")

    if len(encoded.encode("utf-8")) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Chat state payload too large.")

    with _ui_state_lock:
        try:
            state_path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.exception("Failed to write UI chat state (%s)", state_path)
            raise HTTPException(status_code=500, detail=f"Failed to save UI chat state: {e}")

    return {"saved": True, "count": len(chats)}


# ── Chat ───────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, current_user: AuthUser = Depends(get_current_user)):
    """
    Main chat endpoint. No hardcoded modes — always searches for context,
    LLM decides how to respond. Works like a normal chatbot.
    """
    try:
        result = handle_chat(
            user_message=req.message,
            history=req.history,
            model=req.model or OLLAMA_MODEL,
            active_source=req.active_source,
            active_doc_id=req.active_doc_id,
            active_source_type=req.active_source_type,
            system_prompt=req.system_prompt,
            web_search_enabled=req.web_search_enabled,
            user_id=current_user.username,
            user_role=current_user.role,
            department=current_user.department,
        )

        return ChatResponse(
            model=req.model or OLLAMA_MODEL,
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result.get("sources", [])],
            mode=result.get("mode", "general"),
            active_source=result.get("active_source", req.active_source),
            active_doc_id=result.get("active_doc_id", req.active_doc_id),
            done=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


# ── Chat with File Upload (NEW — the missing piece) ──────────────────

@router.post("/chat-with-file")
async def chat_with_file(
    file: UploadFile = File(...),
    message: str = Form(default="이 문서의 내용을 요약해줘"),
    model: str = Form(default=None),
    web_search_enabled: bool = Form(default=True),
    current_user: AuthUser = Depends(require_admin),
):
    """
    Upload a file AND ask a question about it in one request.
    The file is saved, parsed, chunked, stored, then the question is answered.

    Usage:
      curl -X POST http://localhost:8000/chat-with-file \
        -F "file=@document.pdf" \
        -F "message=이 문서의 주요 내용은?"
    """
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    save_path, normalized_user_id, safe_filename = _resolve_upload_target(file.filename, current_user.username)
    ext = Path(safe_filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}",
        )

    # Step 1: Save the file to user-scoped uploads

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        suppress_watcher_for(save_path)
        logger.info("Saved uploaded file: %s (%d bytes, user=%s)", safe_filename, len(content), normalized_user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Step 2: Ingest into ChromaDB
    try:
        ingest_result = await run_in_threadpool(ingest_single_file, save_path, normalized_user_id)
    except Exception as e:
        logger.exception("chat-with-file ingest failed")
        raise HTTPException(status_code=500, detail=f"File ingest failed: {e}")

    if ingest_result.get("count", 0) == 0:
        return {
            "model": OLLAMA_MODEL,
            "answer": f"파일 '{file.filename}'에서 텍스트를 추출하지 못했습니다. "
                      "스캔된 이미지 PDF일 수 있습니다. OCR 설정을 확인해주세요.",
            "sources": [],
            "mode": "document_qa",
            "ingest": ingest_result,
            "done": True,
        }

    # Step 3: Answer using the unified handler, scoped to this file
    selected_model = model or OLLAMA_MODEL

    try:
        result = handle_chat(
            user_message=message,
            model=selected_model,
            active_source=safe_filename,
            active_doc_id=ingest_result.get("doc_id"),
            web_search_enabled=web_search_enabled,
            user_id=normalized_user_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("chat-with-file answer generation failed")
        raise HTTPException(status_code=500, detail=f"chat-with-file failed: {e}")

    return {
        "model": selected_model,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "mode": result.get("mode", "document_qa"),
        "active_source": safe_filename,
        "active_doc_id": ingest_result.get("doc_id"),
        "ingest": ingest_result,
        "done": True,
    }

# ── Ingest ─────────────────────────────────────────────────────────────

@router.post("/ingest")
def ingest(req: IngestRequest, _: AuthUser = Depends(require_admin)):
    folder = Path(req.folder_path) if req.folder_path else LIBRARY_DIR

    try:
        result = ingest_folder(folder)
        return result
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


# ── Upload (NEW) ──────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
MAX_MULTI_UPLOAD_FILES = 90


@router.post("/admin/upload")
async def upload_department_document(
    file: UploadFile = File(...),
    department: str = Form(...),
    visibility: Optional[str] = Form(default=None),
    access_level: Optional[str] = Form(default=None),
    job_level: Optional[str] = Form(default=None),
    department_only: bool = Form(default=False),
    visible_departments: Optional[str] = Form(default=None),
    scope_rules: Optional[str] = Form(default=None),
    current_user: AuthUser = Depends(require_admin),
):
    normalized_department = _normalize_department(department)
    if not normalized_department:
        raise HTTPException(status_code=400, detail="유효한 부서 코드가 필요합니다.")

    safe_filename = Path(file.filename or "").name
    if not safe_filename:
        raise HTTPException(status_code=400, detail="유효한 파일명이 필요합니다.")

    ext = Path(safe_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    target_dir = LIBRARY_DIR / normalized_department
    target_dir.mkdir(parents=True, exist_ok=True)
    save_path = target_dir / safe_filename
    normalized_admin_id = current_user.username
    normalized_scope_rules = _normalize_scope_rules(scope_rules)
    if normalized_scope_rules:
        normalized_visibility = normalized_scope_rules[0]["visibility"]
        normalized_visible_departments = sorted(
            {
                rule["department"]
                for rule in normalized_scope_rules
                if rule.get("visibility") == "private" and rule.get("department")
            }
        )
        if normalized_visibility == "public":
            normalized_department = "ALL"
        elif len(normalized_visible_departments) == 1:
            normalized_department = normalized_visible_departments[0]
        else:
            normalized_department = "MULTI"
        rank_values = []
        for rule in normalized_scope_rules:
            rank = str(rule.get("rank") or "").strip() or "전체"
            if rank not in rank_values:
                rank_values.append(rank)
        normalized_access_level = " · ".join(rank_values) if rank_values else _coerce_access_level(access_level or job_level)
    else:
        normalized_visibility = _normalize_visibility(visibility, normalized_department)
        normalized_visible_departments = _normalize_visible_departments(
            visible_departments,
            fallback_department=normalized_department,
            visibility=normalized_visibility,
        )
        normalized_access_level = _coerce_access_level(access_level or job_level)
    normalized_department_only = bool(department_only or normalized_visibility == "private")

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        suppress_watcher_for(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        result = await run_in_threadpool(
            ingest_single_file,
            save_path,
            normalized_admin_id,
            normalized_department,
        )
        if result.get("count", 0) == 0:
            raise HTTPException(
                status_code=422,
                detail=result.get("message", "Could not extract text from file."),
            )

        if result.get("doc_id"):
            update_document(
                result["doc_id"],
                {
                    "department": normalized_department,
                    "uploader_id": normalized_admin_id,
                    "visibility": normalized_visibility,
                    "visible_departments": normalized_visible_departments,
                    "scope_rules": normalized_scope_rules,
                    "access_level": normalized_access_level,
                    "department_only": normalized_department_only,
                },
            )

        return {
            "message": result["message"],
            "filename": safe_filename,
            "doc_id": result.get("doc_id"),
            "department": normalized_department,
            "visibility": normalized_visibility,
            "visible_departments": normalized_visible_departments,
            "scope_rules": normalized_scope_rules,
            "access_level": normalized_access_level,
            "department_only": normalized_department_only,
            "chunks_stored": result.get("count", 0),
            "source_type": result.get("source_type"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("admin upload failed")
        raise HTTPException(status_code=500, detail=f"admin upload failed: {e}")

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    department: Optional[str] = Form(default=None),
    access_level: Optional[str] = Form(default=None),
    department_only: bool = Form(default=False),
    current_user: AuthUser = Depends(require_admin),
):
    """
    Upload a file, parse it, chunk it, and store it in the vectorstore.

    This is the MISSING PIECE from the original code:
    The chat UI lets users attach files, but the backend had no way
    to receive and process them. Now it does.

    Usage:
        curl -X POST http://localhost:8000/upload -F "file=@document.pdf"
    """
    # Validate file type
    save_path, normalized_user_id, safe_filename = _resolve_upload_target(file.filename, current_user.username)
    ext = Path(safe_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Save to user-scoped uploads directory

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        suppress_watcher_for(save_path)
        logger.info("Saved uploaded file: %s (%d bytes, user=%s)", safe_filename, len(content), normalized_user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Parse, chunk, and store
    try:
        result = await run_in_threadpool(ingest_single_file, save_path, normalized_user_id)

        if result["count"] == 0:
            raise HTTPException(
                status_code=422,
                detail=result.get("message", "Could not extract text from file."),
            )

        if result.get("doc_id"):
            update_document(
                result["doc_id"],
                {
                    "department": _coerce_department(department, normalized_user_id),
                    "access_level": _coerce_access_level(access_level),
                    "department_only": bool(department_only),
                },
            )

        return {
            "message": result["message"],
            "filename": safe_filename,
            "chunks_stored": result["count"],
            "doc_id": result.get("doc_id"),
            "source_type": result.get("source_type"),
            "department": _coerce_department(department, normalized_user_id),
            "access_level": _coerce_access_level(access_level),
            "department_only": bool(department_only),
            "timings": result.get("timings"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload processing failed")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {e}")


@router.post("/upload-multiple")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    department: Optional[str] = Form(default=None),
    access_level: Optional[str] = Form(default=None),
    department_only: bool = Form(default=False),
    current_user: AuthUser = Depends(require_admin),
):
    """Upload and ingest multiple files at once."""
    if len(files) > MAX_MULTI_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"한 번에 최대 {MAX_MULTI_UPLOAD_FILES}개 파일만 업로드할 수 있습니다.",
        )

    results = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({"file": file.filename, "status": "skipped", "reason": f"Unsupported: {ext}"})
            continue

        try:
            save_path, normalized_user_id, safe_filename = _resolve_upload_target(file.filename, current_user.username)
            with open(save_path, "wb") as f:
                content = await file.read()
                f.write(content)
            suppress_watcher_for(save_path)

            result = await run_in_threadpool(ingest_single_file, save_path, normalized_user_id)
            if result.get("doc_id"):
                update_document(
                    result["doc_id"],
                    {
                        "department": _coerce_department(department, normalized_user_id),
                        "access_level": _coerce_access_level(access_level),
                        "department_only": bool(department_only),
                    },
                )
            results.append({
                "file": safe_filename,
                "status": "success" if result["count"] > 0 else "failed",
                "chunks": result["count"],
                "message": result["message"],
                "doc_id": result.get("doc_id"),
                "source_type": result.get("source_type"),
                "department": _coerce_department(department, normalized_user_id),
                "access_level": _coerce_access_level(access_level),
                "department_only": bool(department_only),
            })
        except Exception as e:
            results.append({"file": file.filename, "status": "error", "reason": str(e)})

    total_chunks = sum(r.get("chunks", 0) for r in results)
    return {
        "message": f"Processed {len(results)} files, {total_chunks} total chunks stored.",
        "results": results,
    }


# ── Reset DB ───────────────────────────────────────────────────────────

@router.delete("/reset-db")
def reset_db(_: AuthUser = Depends(require_admin)):
    try:
        reset_vectorstore()
        clear_document_registry()
        return {"message": "벡터 DB 초기화 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reset-db failed: {e}")


# ── Document List ──────────────────────────────────────────────────────

@router.get("/admin/documents")
def admin_documents(_: AuthUser = Depends(require_admin)):
    try:
        documents = [_build_admin_document_payload(entry) for entry in list_documents()]
        documents.sort(
            key=lambda doc: (
                str(doc.get("uploaded_at") or ""),
                str(doc.get("updated_at") or ""),
                str(doc.get("source") or ""),
            ),
            reverse=True,
        )

        departments = {}
        access_levels = {}
        alerts = []

        for doc in documents:
            department = doc["department"]
            access_level = doc["access_level"]
            departments[department] = departments.get(department, 0) + 1
            access_levels[access_level] = access_levels.get(access_level, 0) + 1

            if not doc["file_exists"]:
                alerts.append({
                    "level": "warning",
                    "message": f"{doc['source']} 파일 경로를 찾지 못했습니다.",
                    "doc_id": doc.get("doc_id"),
                })

        return {
            "summary": {
                "indexing_status": "실행 중",
                "total_documents": len(documents),
                "pending_count": sum(1 for doc in documents if doc.get("status_category") != "approved"),
                "queue_count": 0,
            },
            "filters": {
                "departments": [{"name": name, "count": count} for name, count in sorted(departments.items())],
                "access_levels": [{"name": name, "count": count} for name, count in sorted(access_levels.items())],
            },
            "documents": documents,
            "alerts": alerts[:8],
        }
    except Exception as e:
        logger.exception("admin/documents failed")
        raise HTTPException(status_code=500, detail=f"admin/documents failed: {e}")


@router.post("/admin/documents/approve")
def approve_admin_documents(
    payload: dict = Body(...),
    current_user: AuthUser = Depends(require_admin),
):
    doc_ids = payload.get("doc_ids") if isinstance(payload, dict) else None
    if not isinstance(doc_ids, list):
        raise HTTPException(status_code=400, detail="'doc_ids' must be an array.")

    normalized_doc_ids = [
        str(doc_id).strip()
        for doc_id in doc_ids
        if str(doc_id).strip()
    ]
    if not normalized_doc_ids:
        raise HTTPException(status_code=400, detail="승인할 문서를 하나 이상 선택해 주세요.")

    approved_count = 0
    missing_doc_ids = []

    for doc_id in normalized_doc_ids:
        updated = update_document(
            doc_id,
            {
                "status": "approved",
            },
        )
        if updated is None:
            missing_doc_ids.append(doc_id)
            continue
        update_document(
            doc_id,
            {
                "status": "approved",
                "approved_at": updated.get("updated_at"),
                "approved_by": current_user.username,
            },
        )
        approved_count += 1

    if approved_count == 0:
        raise HTTPException(status_code=404, detail="선택한 문서를 찾지 못했습니다.")

    return {
        "approved_count": approved_count,
        "approved_doc_ids": [doc_id for doc_id in normalized_doc_ids if doc_id not in set(missing_doc_ids)],
        "missing_doc_ids": missing_doc_ids,
        "approved_by": current_user.username,
    }


@router.get("/docs-list")
def docs_list(current_user: AuthUser = Depends(require_admin)):
    try:
        normalized_user_id = current_user.username
        metadata_list = get_all_metadata(owner_id=normalized_user_id)

        unique_docs = {}
        for meta in metadata_list:
            if not meta:
                continue
            key = (
                meta.get("doc_id"),
                meta.get("source"),
                meta.get("page"),
                meta.get("chunk_index"),
            )
            unique_docs[key] = {
                "doc_id": meta.get("doc_id"),
                "source": meta.get("source"),
                "source_type": meta.get("source_type"),
                "page_total": meta.get("page_total"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "source_path": meta.get("source_path"),
                "extraction_method": meta.get("extraction_method"),
            }

        return {
            "count": len(unique_docs),
            "documents": list(unique_docs.values()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"docs-list failed: {e}")


@router.delete("/upload-document")
def delete_upload_document(
    source: Optional[str] = None,
    doc_id: Optional[str] = None,
    current_user: AuthUser = Depends(require_admin),
):
    """Delete one uploaded document from vectorstore/registry and remove local upload file."""
    try:
        result = _delete_managed_document(
            source=source,
            doc_id=doc_id,
            source_type="upload",
            owner_id=current_user.username,
        )
        result["message"] = "업로드 파일이 삭제되었습니다."
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload-document delete failed: {e}")


@router.delete("/admin/document")
def delete_admin_document(
    source: Optional[str] = None,
    doc_id: Optional[str] = None,
    source_type: Optional[str] = None,
    _: AuthUser = Depends(require_admin),
):
    try:
        return _delete_managed_document(source=source, doc_id=doc_id, source_type=source_type)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"admin/document delete failed: {e}")


# ── Keyword Count ──────────────────────────────────────────────────────

@router.post("/count-keyword")
def count_keyword(req: CountKeywordRequest, _: AuthUser = Depends(get_current_user)):
    try:
        candidate_paths = [
            UPLOADS_DIR / req.filename,
            LIBRARY_DIR / req.filename,
            DATA_DIR / req.filename,
        ]
        target_path = next((path for path in candidate_paths if path.exists()), None)

        if target_path is None:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

        text = extract_full_text(str(target_path))
        if not text:
            return {
                "filename": req.filename,
                "keyword": req.keyword,
                "count": 0,
                "message": "추출된 텍스트가 없습니다.",
            }

        count = text.lower().count(req.keyword.lower())

        return {
            "filename": req.filename,
            "keyword": req.keyword,
            "count": count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"count-keyword failed: {e}")


@router.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    prompt: str = Form(default="이 이미지를 한국어로 자세히 설명해주세요."),
    model: Optional[str] = Form(default=None),
    _: AuthUser = Depends(get_current_user),
):
    """Analyze a single image directly with the vision model."""
    try:
        content = await file.read()
        reply = analyze_image_bytes(content, file.filename or "", prompt=prompt, model=model)
        return {
            "file": file.filename,
            "reply": reply,
            "model": model or VISION_MODEL,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload-image failed: {e}")


@router.post("/stt")
async def speech_to_text(file: UploadFile = File(...), _: AuthUser = Depends(get_current_user)):
    """Transcribe one uploaded audio file with Whisper."""
    try:
        content = await file.read()
        text = transcribe_audio_bytes(content, file.filename or "")
        return {
            "file": file.filename,
            "text": text,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"stt failed: {e}")


@router.post("/chat-audio")
async def chat_audio(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None),
    active_source: Optional[str] = Form(default=None),
    active_doc_id: Optional[str] = Form(default=None),
    active_source_type: Optional[str] = Form(default=None),
    system_prompt: Optional[str] = Form(default=None),
    web_search_enabled: bool = Form(default=True),
    current_user: AuthUser = Depends(get_current_user),
):
    """Transcribe audio and pass the recognized text through the normal chat flow."""
    try:
        content = await file.read()
        recognized_text = transcribe_audio_bytes(content, file.filename or "")
        result = handle_chat(
            user_message=recognized_text,
            model=model or OLLAMA_MODEL,
            active_source=active_source,
            active_doc_id=active_doc_id,
            active_source_type=active_source_type,
            system_prompt=system_prompt,
            web_search_enabled=web_search_enabled,
            user_id=current_user.username,
            user_role=current_user.role,
            department=current_user.department,
        )
        return {
            "recognized_text": recognized_text,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "mode": result.get("mode", "general"),
            "active_source": result.get("active_source", active_source),
            "active_doc_id": result.get("active_doc_id", active_doc_id),
            "model": model or OLLAMA_MODEL,
            "done": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chat-audio failed: {e}")


@router.post("/web-search")
def web_search(req: WebSearchRequest):
    """Structured web search endpoint with Tavily or DuckDuckGo fallback."""
    try:
        results = search_web(
            req.query,
            max_results=req.max_results,
            region=req.region or DUCKDUCKGO_REGION,
        )
        return {
            "query": req.query,
            "count": len(results),
            "options": {
                "region": req.region,
                "max_results": req.max_results,
            },
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"web-search failed: {e}")
