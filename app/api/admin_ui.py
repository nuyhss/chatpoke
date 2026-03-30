import logging
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, RedirectResponse

from app.api.deps import AuthUser, get_optional_user
from app.core.auth_store import is_admin_role

logger = logging.getLogger("tilon.files_ui")
router = APIRouter(tags=["Files UI"])

FILES_LOGIN_UI_PATH = Path(__file__).resolve().parents[2] / "static" / "files_login.html"
FILES_USER_UI_PATH = Path(__file__).resolve().parents[2] / "static" / "files_user.html"
FILES_ADMIN_UI_PATH = Path(__file__).resolve().parents[2] / "static" / "files_admin.html"


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


@router.get("/files", include_in_schema=False)
def files_root(user: AuthUser | None = Depends(get_optional_user)):
    if not user:
        return RedirectResponse(url="/files/login", status_code=307)
    if is_admin_role(user.role):
        return RedirectResponse(url="/files/admin", status_code=307)
    return RedirectResponse(url="/files/user", status_code=307)


@router.get("/files/user", response_class=HTMLResponse)
def files_user_ui():
    return html_file_response(FILES_USER_UI_PATH, "static files user UI not found at %s")


@router.get("/files/admin", response_class=HTMLResponse)
def files_admin_ui():
    return html_file_response(FILES_ADMIN_UI_PATH, "static files admin UI not found at %s")


@router.get("/admin", include_in_schema=False)
def legacy_admin_redirect():
    return RedirectResponse(url="/files/admin", status_code=307)
