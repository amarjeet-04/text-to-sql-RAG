"""### MINIMAL CORE — Database connect / status routes."""
import logging
import os
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.services.session import Session, SessionState, QueryCacheStore
from backend.services.sql_engine import (
    initialize_connection,
    normalize_sql_dialect,
    invalidate_runtime_caches,
)
from backend.services.runtime import log_event
from backend.routes.deps import get_session

router = APIRouter(prefix="/api/db", tags=["database"])
logger = logging.getLogger("database_route")


def _resolve_provider_api_key(llm_provider: str, api_key: str, model: str) -> str:
    provider = (llm_provider or "").strip().lower()
    key = (api_key or "").strip()
    model_low = (model or "").strip().lower()
    deepseek_env = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    openai_env = (os.getenv("OPENAI_API_KEY") or "").strip()

    # Provider-aware defaults/fallbacks.
    if provider == "deepseek" or model_low.startswith("deepseek"):
        if key.startswith("sk-proj-") and deepseek_env:
            logger.warning("deepseek provider received OpenAI project key; using DEEPSEEK_API_KEY fallback")
            return deepseek_env
        return key or deepseek_env
    if provider == "openai":
        return key or openai_env
    return key or deepseek_env or openai_env


class ConnectRequest(BaseModel):
    host:          str
    port:          str
    username:      str
    password:      str
    database:      str
    llm_provider:  str   = "OpenAI"
    api_key:       str   = ""
    model:         str   = "gpt-4"
    temperature:   float = 0.0
    query_timeout: int   = 30
    view_support:  bool  = True
    sql_dialect:   str   = "sqlserver"
    enable_nolock: bool  = True


class ConnectResponse(BaseModel):
    success:      bool
    message:      str
    tables_count: int = 0
    views_count:  int = 0


class StatusResponse(BaseModel):
    connected:    bool
    tables_count: int = 0
    views_count:  int = 0


@router.post("/connect", response_model=ConnectResponse)
def connect_db(req: ConnectRequest, session: Session = Depends(get_session)):
    try:
        resolved_api_key = _resolve_provider_api_key(req.llm_provider, req.api_key, req.model)
        if not resolved_api_key:
            return ConnectResponse(
                success=False,
                message=(
                    "Error: missing API key. Provide api_key in request or set "
                    "DEEPSEEK_API_KEY / OPENAI_API_KEY in environment."
                ),
            )
        old_db = session.db
        if old_db is not None:
            try:
                old_engine = getattr(old_db, "_engine", None)
                if old_engine is not None:
                    old_engine.dispose()
            except Exception:
                logger.warning("failed_to_dispose_old_engine", exc_info=True)

        invalidate_runtime_caches(reason="db_reconnect")
        (db, db_config, sql_chain, llm, reasoning_llm,
         rag, embedder, message, tables_count, views_count,
         cached_schema_text) = initialize_connection(
            host=req.host, port=req.port,
            db_username=req.username, db_password=req.password,
            database=req.database,
            llm_provider=req.llm_provider, api_key=resolved_api_key,
            model=req.model, temperature=req.temperature,
            query_timeout=req.query_timeout, view_support=req.view_support,
        )
        with session.lock:
            session.db                 = db
            session.db_config          = db_config
            session.sql_chain          = sql_chain
            session.llm                = llm
            session.reasoning_llm      = reasoning_llm
            session.rag_engine         = rag
            session.embedder           = embedder
            session.cached_schema_text = cached_schema_text
            session.connected          = True
            session.sql_dialect        = normalize_sql_dialect(req.sql_dialect)
            session.enable_nolock      = bool(req.enable_nolock)
            # Reset conversation on new connection
            session.chat_history        = []
            session.conversation_turns  = []
            session.session_state       = SessionState(dialect=session.sql_dialect)
            session.query_cache         = QueryCacheStore()

        if rag is not None and hasattr(rag, "clear_retrieval_cache"):
            try:
                rag.clear_retrieval_cache()
            except Exception:
                logger.warning("failed_to_clear_rag_cache", exc_info=True)

        log_event(
            logger,
            logging.INFO,
            "db_connect_success",
            host=req.host,
            port=req.port,
            database=req.database,
            tables_count=tables_count,
            views_count=views_count,
        )
        return ConnectResponse(success=True, message=message,
                               tables_count=tables_count, views_count=views_count)
    except Exception as exc:
        log_event(logger, logging.ERROR, "db_connect_error", error=str(exc))
        return ConnectResponse(success=False, message=f"Error: {exc}")


@router.get("/status", response_model=StatusResponse)
def db_status(session: Session = Depends(get_session)):
    if not session.connected or session.db is None:
        return StatusResponse(connected=False)
    try:
        names = list(session.db.get_usable_table_names())
        return StatusResponse(connected=True, tables_count=len(names), views_count=0)
    except Exception:
        return StatusResponse(connected=False)
