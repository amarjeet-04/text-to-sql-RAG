from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List

from backend.services.session import verify_user, register_user, list_users, delete_user, store
from backend.routes.deps import get_session

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str
    password: str
    name: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "Analyst"
    name: str


class UserInfo(BaseModel):
    username: str
    role: str
    name: str


class LoginResponse(BaseModel):
    token: str
    user: UserInfo


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    user = verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    session = store.create(user)
    return LoginResponse(
        token=session.token,
        user=UserInfo(**user),
    )


@router.post("/signup", response_model=LoginResponse)
def signup(req: SignupRequest):
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if not req.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    try:
        user = register_user(
            username=req.username.strip(),
            password=req.password,
            role="Analyst",
            name=req.name.strip() or req.username.strip(),
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    session = store.create(user)
    return LoginResponse(
        token=session.token,
        user=UserInfo(**user),
    )


@router.post("/logout")
def logout(token: str):
    store.delete(token)
    return {"success": True}


# ── Admin-only user management ────────────────────────────────────────────────

def _require_admin(session=Depends(get_session)):
    if session.user.get("role") != "Admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return session


@router.get("/users", response_model=List[UserInfo])
def get_users(session=Depends(_require_admin)):
    return list_users()


@router.post("/users", response_model=UserInfo)
def create_user(req: CreateUserRequest, session=Depends(_require_admin)):
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if req.role not in ("Admin", "Analyst"):
        raise HTTPException(status_code=400, detail="Role must be Admin or Analyst")
    try:
        user = register_user(
            username=req.username.strip(),
            password=req.password,
            role=req.role,
            name=req.name.strip() or req.username.strip(),
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return UserInfo(**user)


@router.delete("/users/{username}")
def remove_user(username: str, session=Depends(_require_admin)):
    if username == session.user.get("username"):
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    try:
        delete_user(username)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True}
