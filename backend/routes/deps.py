from fastapi import Header, HTTPException

from backend.services.session import store, Session


def get_session(authorization: str = Header(...)) -> Session:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[7:]
    session = store.get(token)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return session
