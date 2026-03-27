"""
Admin document management UI.
Access at: http://localhost:8000/admin
"""

import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

logger = logging.getLogger("tilon.admin_ui")
router = APIRouter(tags=["Admin UI"])
ADMIN_UI_PATH = Path(__file__).resolve().parents[2] / "static" / "admin.html"


@router.get("/admin", response_class=HTMLResponse)
def admin_ui():
    if ADMIN_UI_PATH.exists():
        return FileResponse(ADMIN_UI_PATH)

    logger.warning("admin UI not found at %s", ADMIN_UI_PATH)
    return HTMLResponse("<h1>admin.html not found</h1>", status_code=500)
