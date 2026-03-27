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

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from starlette.concurrency import run_in_threadpool

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
from app.core.document_registry import clear_document_registry, remove_documents
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
def get_ui_state_chats(user_id: Optional[str] = None):
    state_path = _ui_chats_state_path(user_id)

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
    user_id: Optional[str] = None,
):
    chats = payload.get("chats") if isinstance(payload, dict) else None
    if not isinstance(chats, dict):
        raise HTTPException(status_code=400, detail="'chats' must be an object.")

    state_path = _ui_chats_state_path(user_id)
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
def chat(req: ChatRequest):
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
            user_id=req.user_id,
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
    user_id: Optional[str] = Form(default=None),
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
    save_path, normalized_user_id, safe_filename = _resolve_upload_target(file.filename, user_id)
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
def ingest(req: IngestRequest):
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

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: Optional[str] = Form(default=None)):
    """
    Upload a file, parse it, chunk it, and store it in the vectorstore.

    This is the MISSING PIECE from the original code:
    The chat UI lets users attach files, but the backend had no way
    to receive and process them. Now it does.

    Usage:
        curl -X POST http://localhost:8000/upload -F "file=@document.pdf"
    """
    # Validate file type
    save_path, normalized_user_id, safe_filename = _resolve_upload_target(file.filename, user_id)
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

        return {
            "message": result["message"],
            "filename": safe_filename,
            "chunks_stored": result["count"],
            "doc_id": result.get("doc_id"),
            "source_type": result.get("source_type"),
            "timings": result.get("timings"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload processing failed")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {e}")


@router.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...), user_id: Optional[str] = Form(default=None)):
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
            save_path, normalized_user_id, safe_filename = _resolve_upload_target(file.filename, user_id)
            with open(save_path, "wb") as f:
                content = await file.read()
                f.write(content)
            suppress_watcher_for(save_path)

            result = await run_in_threadpool(ingest_single_file, save_path, normalized_user_id)
            results.append({
                "file": safe_filename,
                "status": "success" if result["count"] > 0 else "failed",
                "chunks": result["count"],
                "message": result["message"],
                "doc_id": result.get("doc_id"),
                "source_type": result.get("source_type"),
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
def reset_db():
    try:
        reset_vectorstore()
        clear_document_registry()
        return {"message": "벡터 DB 초기화 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reset-db failed: {e}")


# ── Document List ──────────────────────────────────────────────────────

@router.get("/docs-list")
def docs_list(user_id: Optional[str] = None):
    try:
        normalized_user_id = _normalize_user_id(user_id)
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
    user_id: Optional[str] = None,
):
    """Delete one uploaded document from vectorstore/registry and remove local upload file."""
    if not source and not doc_id:
        raise HTTPException(status_code=400, detail="source 또는 doc_id 중 하나는 필요합니다.")

    try:
        normalized_user_id = _normalize_user_id(user_id)
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

        for s in source_candidates:
            add_attempt(s, doc_id, "upload")
        for s in source_candidates:
            add_attempt(s, doc_id, None)

        if doc_id:
            add_attempt(None, doc_id, "upload")
            add_attempt(None, doc_id, None)

        for s in source_candidates:
            add_attempt(s, None, "upload")
            add_attempt(s, None, None)

        deleted_chunks = 0
        removed_registry = 0
        for s, d, st in attempts:
            deleted_chunks += delete_documents(source=s, doc_id=d, source_type=st, owner_id=normalized_user_id)
            removed_registry += remove_documents(source=s, doc_id=d, source_type=st, owner_id=normalized_user_id)

        file_deleted = False
        candidate_dirs: List[Path] = []
        if normalized_user_id:
            candidate_dirs.append(UPLOADS_DIR / normalized_user_id)
        candidate_dirs.append(UPLOADS_DIR)

        for s in source_candidates:
            if not s:
                continue
            safe_name = Path(s).name
            for base_dir in candidate_dirs:
                upload_path = base_dir / safe_name
                if upload_path.exists() and upload_path.is_file():
                    upload_path.unlink()
                    file_deleted = True

        if deleted_chunks == 0 and removed_registry == 0 and not file_deleted:
            raise HTTPException(status_code=404, detail="삭제할 업로드 문서를 찾지 못했습니다.")

        return {
            "message": "업로드 파일이 삭제되었습니다.",
            "deleted_chunks": deleted_chunks,
            "removed_registry": removed_registry,
            "file_deleted": file_deleted,
            "source": source,
            "doc_id": doc_id,
            "attempts": len(attempts),
            "user_id": normalized_user_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload-document delete failed: {e}")


# ── Keyword Count ──────────────────────────────────────────────────────

@router.post("/count-keyword")
def count_keyword(req: CountKeywordRequest):
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
async def speech_to_text(file: UploadFile = File(...)):
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
