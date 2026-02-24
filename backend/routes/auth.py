from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.session import verify_user, store

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


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


@router.post("/logout")
def logout(token: str):
    store.delete(token)
    return {"success": True}
