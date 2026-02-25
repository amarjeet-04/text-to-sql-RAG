"""### MINIMAL CORE — Auth dependency."""
from fastapi import Header, HTTPException
from backend.services.session import store, Session
from backend.services.runtime import set_session_id


def get_session(authorization: str = Header(...)) -> Session:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header")
    token = authorization[7:]
    session = store.get(token)
    if session is None:
        raise HTTPException(401, "Invalid or expired session")
    set_session_id(token)
    return session
