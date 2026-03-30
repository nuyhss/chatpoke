from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from app.api.deps import (
    AuthUser,
    clear_session_cookie,
    get_current_user,
    get_optional_user,
    require_admin,
    set_session_cookie,
)
from app.core.auth_store import (
    ROLE_USER,
    ADMIN_ROLES,
    authenticate_user,
    change_password,
    create_session,
    create_user,
    delete_session,
    delete_user,
    list_users,
    reset_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str
    login_type: Optional[str] = "employee"


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str
    department: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/login")
def login(payload: LoginRequest, response: Response):
    user = authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")

    login_type = (payload.login_type or "employee").strip().lower()
    if login_type == "admin" and user["role"] not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="관리자 로그인에는 adminD 또는 admin 권한이 필요합니다.")
    if login_type == "employee" and user["role"] != ROLE_USER:
        raise HTTPException(status_code=403, detail="직원 로그인에는 user 권한 계정이 필요합니다.")

    token = create_session(user["username"])
    set_session_cookie(response, token)
    return {
        "user": {
            "id": user["username"],
            "username": user["username"],
            "role": user["role"],
            "department": user["department"],
        }
    }


@router.post("/logout")
def logout(request: Request, response: Response, user: Optional[AuthUser] = Depends(get_optional_user)):
    if user:
        delete_session(request.cookies.get("tilon_session", ""))
    clear_session_cookie(response)
    return {"ok": True}


@router.get("/me")
def me(user: AuthUser = Depends(get_current_user)):
    return {
        "user": {
            "id": user.username,
            "username": user.username,
            "role": user.role,
            "department": user.department,
        }
    }


@router.get("/users")
def get_users(_: AuthUser = Depends(require_admin)):
    return {"users": list_users()}


@router.post("/users")
def create_auth_user(payload: CreateUserRequest, _: AuthUser = Depends(require_admin)):
    try:
        user = create_user(
            username=payload.username,
            password=payload.password,
            role=payload.role,
            department=payload.department,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "user": {
            "id": user["username"],
            "username": user["username"],
            "role": user["role"],
            "department": user["department"],
        }
    }


@router.delete("/users/{username}")
def delete_auth_user(username: str, user: AuthUser = Depends(require_admin)):
    if username == user.username:
        raise HTTPException(status_code=400, detail="현재 로그인한 관리자 계정은 삭제할 수 없습니다.")
    try:
        delete_user(username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": True}


@router.post("/users/{username}/reset-password")
def reset_auth_user_password(username: str, _: AuthUser = Depends(require_admin)):
    try:
        new_password = reset_password(username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"reset": True, "password": new_password}


@router.post("/change-password")
def change_my_password(payload: ChangePasswordRequest, user: AuthUser = Depends(get_current_user)):
    try:
        change_password(user.username, payload.current_password, payload.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"changed": True}
