from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from backend.services.session import store, Session
from backend.services.sql_engine import initialize_connection
from backend.routes.deps import get_session

router = APIRouter(prefix="/api/db", tags=["database"])


class ConnectRequest(BaseModel):
    host: str
    port: str
    username: str
    password: str
    database: str
    llm_provider: str = "OpenAI"
    api_key: str = ""
    model: str = "gpt-4"
    temperature: float = 0.0
    query_timeout: int = 30
    view_support: bool = True


class ConnectResponse(BaseModel):
    success: bool
    message: str
    tables_count: int = 0
    views_count: int = 0


class StatusResponse(BaseModel):
    connected: bool
    tables_count: int = 0
    views_count: int = 0


@router.post("/connect", response_model=ConnectResponse)
def connect_db(req: ConnectRequest, session: Session = Depends(get_session)):
    try:
        db, db_config, sql_chain, llm, reasoning_llm, rag, embedder, message, tables_count, views_count, cached_schema_text = initialize_connection(
            host=req.host,
            port=req.port,
            db_username=req.username,
            db_password=req.password,
            database=req.database,
            llm_provider=req.llm_provider,
            api_key=req.api_key,
            model=req.model,
            temperature=req.temperature,
            query_timeout=req.query_timeout,
            view_support=req.view_support,
        )

        session.db = db
        session.db_config = db_config
        session.sql_chain = sql_chain
        session.llm = llm
        session.reasoning_llm = reasoning_llm
        session.rag_engine = rag
        session.embedder = embedder
        session.cached_schema_text = cached_schema_text
        session.connected = True
        session.query_cache = []
        session.chat_history = []
        session.conversation_context = ""

        return ConnectResponse(success=True, message=message, tables_count=tables_count, views_count=views_count)
    except Exception as e:
        return ConnectResponse(success=False, message=f"Error: {str(e)}")


@router.get("/status", response_model=StatusResponse)
def db_status(session: Session = Depends(get_session)):
    if not session.connected or session.db is None:
        return StatusResponse(connected=False)

    # Fast status check — use cached table names instead of slow introspection
    try:
        table_names = list(session.db.get_usable_table_names())
        return StatusResponse(
            connected=True,
            tables_count=len(table_names),
            views_count=0,
        )
    except Exception:
        return StatusResponse(connected=False)
