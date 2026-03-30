"""
Files UI.
Access at: http://localhost:8000/files
"""

import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

logger = logging.getLogger("tilon.files_ui")
router = APIRouter(tags=["Files UI"])

FILES_UI_PATH = Path(__file__).resolve().parents[2] / "static" / "files.html"
FILES_LOGIN_UI_PATH = Path(__file__).resolve().parents[2] / "static" / "files_login.html"


def html_file_response(path: Path, not_found_message: str) -> HTMLResponse:
    if not path.exists():
        logger.warning(not_found_message, path)
        return HTMLResponse(f"<h1>{path.name} not found</h1>", status_code=500)

    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/files/login", response_class=HTMLResponse)
def files_login_ui():
    return html_file_response(FILES_LOGIN_UI_PATH, "static files login UI not found at %s")


@router.get("/files", response_class=HTMLResponse)
def files_ui():
    return html_file_response(FILES_UI_PATH, "static files UI not found at %s")


@router.get("/admin", include_in_schema=False)
def legacy_admin_redirect():
    return RedirectResponse(url="/files", status_code=307)
