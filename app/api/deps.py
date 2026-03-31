from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.responses import Response

from app.core.auth_store import get_user_by_session_token, is_admin_role

CHAT_AUTH_CONTEXT = "chat"
FILES_USER_AUTH_CONTEXT = "files_user"
FILES_ADMIN_AUTH_CONTEXT = "files_admin"

AUTH_COOKIE_BY_CONTEXT = {
    CHAT_AUTH_CONTEXT: "chat_session",
    FILES_USER_AUTH_CONTEXT: "files_user_session",
    FILES_ADMIN_AUTH_CONTEXT: "files_admin_session",
}

SESSION_COOKIE_NAME = AUTH_COOKIE_BY_CONTEXT[CHAT_AUTH_CONTEXT]
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 7


@dataclass
class AuthUser:
    id: int
    username: str
    role: str
    department: str
    is_active: bool


def get_cookie_name_for_context(context: str) -> str:
    return AUTH_COOKIE_BY_CONTEXT.get(context, AUTH_COOKIE_BY_CONTEXT[CHAT_AUTH_CONTEXT])


def _infer_auth_context(request: Request) -> str:
    explicit = (
        request.headers.get("X-Auth-Context")
        or request.query_params.get("auth_context")
        or ""
    ).strip().lower()
    if explicit in AUTH_COOKIE_BY_CONTEXT:
        return explicit

    path = request.url.path or ""
    if path.startswith("/chat") or path.startswith("/ui-state") or path.startswith("/upload") or path.startswith("/docs-list"):
        return CHAT_AUTH_CONTEXT
    if path.startswith("/files/admin") or path.startswith("/admin"):
        return FILES_ADMIN_AUTH_CONTEXT
    if path.startswith("/files/user") or path.startswith("/files/requests"):
        return FILES_USER_AUTH_CONTEXT
    if path.startswith("/auth"):
        for context in (FILES_ADMIN_AUTH_CONTEXT, FILES_USER_AUTH_CONTEXT, CHAT_AUTH_CONTEXT):
            cookie_name = get_cookie_name_for_context(context)
            if request.cookies.get(cookie_name):
                return context
    return CHAT_AUTH_CONTEXT


def get_user_from_request_context(request: Request, context: Optional[str] = None) -> Optional[AuthUser]:
    header_token = (request.headers.get("X-Session-Token") or "").strip()
    if header_token:
        token = header_token
    else:
        resolved_context = context or _infer_auth_context(request)
        cookie_name = get_cookie_name_for_context(resolved_context)
        token = request.cookies.get(cookie_name, "")
    user = get_user_by_session_token(token)
    if not user:
        return None
    return AuthUser(
        id=user["id"],
        username=user["username"],
        role=user["role"],
        department=user["department"],
        is_active=user["is_active"],
    )


def set_session_cookie(response: Response, token: str, context: str = CHAT_AUTH_CONTEXT) -> None:
    cookie_name = get_cookie_name_for_context(context)
    response.set_cookie(
        cookie_name,
        token,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


def clear_session_cookie(response: Response, context: str = CHAT_AUTH_CONTEXT) -> None:
    cookie_name = get_cookie_name_for_context(context)
    response.delete_cookie(cookie_name, path="/", samesite="lax")


def get_optional_user(request: Request) -> Optional[AuthUser]:
    return get_user_from_request_context(request)


def get_current_user(user: Optional[AuthUser] = Depends(get_optional_user)) -> AuthUser:
    if not user:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return user


def require_admin(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if not is_admin_role(user.role):
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    return user
