from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.responses import Response

from app.core.auth_store import get_user_by_session_token, is_admin_role

SESSION_COOKIE_NAME = "tilon_session"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 7


@dataclass
class AuthUser:
    id: int
    username: str
    role: str
    department: str
    is_active: bool


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/", samesite="lax")


def get_optional_user(request: Request) -> Optional[AuthUser]:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    user = get_user_by_session_token(token or "")
    if not user:
        return None
    return AuthUser(
        id=user["id"],
        username=user["username"],
        role=user["role"],
        department=user["department"],
        is_active=user["is_active"],
    )


def get_current_user(user: Optional[AuthUser] = Depends(get_optional_user)) -> AuthUser:
    if not user:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return user


def require_admin(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if not is_admin_role(user.role):
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    return user
