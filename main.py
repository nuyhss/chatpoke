"""
Application entry point.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

IMPROVEMENTS over original:
- Uses modern lifespan instead of deprecated @app.on_event("startup")
- Structured logging on startup
- Routes cleanly mounted from separate modules
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import (
    setup_logging,
    logger,
    AUTO_INGEST_ON_STARTUP,
    DATA_DIR,
    LIBRARY_DIR,
    UPLOADS_DIR,
    TEMP_DIR,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    RERANKER_ENABLED,
    RERANKER_MODEL,
)
from app.core.vectorstore import get_vectorstore
from app.core.watcher import start_watcher, stop_watcher
from app.pipeline.ingest import ingest_folder
from app.api.routes import router as main_router
from app.api.openai_compat import router as openai_router
from app.api.upload_ui import router as ui_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("Tilon AI Chatbot API starting up...")
    logger.info("  LLM model     : %s", OLLAMA_MODEL)
    logger.info("  Embedding     : %s", EMBEDDING_MODEL)
    logger.info("  Reranker      : %s", RERANKER_MODEL if RERANKER_ENABLED else "disabled")
    logger.info("  Data dir      : %s", DATA_DIR)
    logger.info("  Library dir   : %s", LIBRARY_DIR)
    logger.info("  Uploads dir   : %s", UPLOADS_DIR)
    logger.info("  Temp dir      : %s", TEMP_DIR)
    logger.info("  File watcher  : active (auto-ingests new files in data/library/)")
    logger.info("  Chat UI       : http://localhost:8000/ui")
    logger.info("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Keep vectorstore initialization lazy so the UI can start even when
    # embedding models are not available yet or the machine is offline.
    logger.info("Vectorstore initialization deferred until first retrieval request.")

    # Auto-ingest if configured
    if AUTO_INGEST_ON_STARTUP:
        try:
            result = ingest_folder(LIBRARY_DIR)
            logger.info("Auto-ingest result: %s", result.get("message"))
        except Exception as e:
            logger.warning("Auto-ingest skipped: %s", e)

    # Start file watcher (auto-ingests new files dropped into data/library/)
    start_watcher()

    yield  # App is running

    stop_watcher()
    logger.info("Tilon AI Chatbot API shutting down.")


app = FastAPI(
    title="Tilon AI Chatbot API",
    version="7.1.0",
    description="RAG-based AI Chatbot with PDF parsing, vector retrieval, and QLoRA fine-tuning support.",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount routers
app.include_router(main_router)
app.include_router(openai_router)
app.include_router(ui_router)
