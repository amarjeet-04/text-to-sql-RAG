"""### MINIMAL CORE — Chat routes (no pandas in the request hot path)."""
import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from contextvars import copy_context
from datetime import datetime
from typing import Optional, List, Any, Dict

from openai import AsyncOpenAI
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.session import (
    ConversationTurn,
    FOLLOWUP_WINDOW_TURNS,
    QueryCacheStore,
    Session,
    SessionState,
    append_with_sliding_window,
)
from backend.services.sql_engine import (
    handle_query as legacy_handle_query,
    generate_nl_response,
    detect_intent_simple,
    invalidate_runtime_caches,
    StepTimer,
)
from backend.services.runtime import (
    run_with_timeout,
    log_event,
    get_request_id,
    get_foreground_executor,
)
from backend.routes.deps import get_session

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger("chat_route")
NL_RESPONSE_TTL_SECONDS = max(30, int(os.getenv("NL_RESPONSE_TTL_SECONDS", "180")))
NL_RESPONSE_MAX_ENTRIES = max(64, int(os.getenv("NL_RESPONSE_MAX_ENTRIES", "512")))
NL_STREAM_WAIT_SECONDS = max(2.0, float(os.getenv("NL_STREAM_WAIT_SECONDS", "15")))
_NL_RESPONSES: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_NL_RESPONSE_TASKS: Dict[str, asyncio.Task] = {}
_NL_LOCK = threading.RLock()


def _normalize_stage_timing(
    timing: Optional[Dict[str, Any]],
    *,
    total_ms: Optional[float] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {stage: 0.0 for stage in StepTimer.STAGES}
    extras: Dict[str, float] = {}

    raw = timing if isinstance(timing, dict) else {}
    for stage, value in raw.items():
        if not isinstance(stage, str):
            continue
        try:
            parsed = round(float(value), 2)
        except Exception:
            continue
        if stage in out:
            out[stage] = parsed
        else:
            extras[stage] = parsed

    if total_ms is not None:
        try:
            out["total"] = round(float(total_ms), 2)
        except Exception:
            pass

    out.update(extras)
    return out


def _nl_cache_key(session_token: str, request_id: str) -> str:
    return f"{session_token}:{request_id}"


def _cleanup_nl_cache_locked(now_ts: float) -> None:
    expired_keys = [k for k, v in _NL_RESPONSES.items() if float(v.get("expires_at", 0.0)) <= now_ts]
    for key in expired_keys:
        _NL_RESPONSES.pop(key, None)
        task = _NL_RESPONSE_TASKS.pop(key, None)
        if task is not None and not task.done():
            task.cancel()
    while len(_NL_RESPONSES) > NL_RESPONSE_MAX_ENTRIES:
        key, _ = _NL_RESPONSES.popitem(last=False)
        task = _NL_RESPONSE_TASKS.pop(key, None)
        if task is not None and not task.done():
            task.cancel()


def _set_nl_state(
    *,
    session_token: str,
    request_id: str,
    status: str,
    question: Optional[str] = None,
    results: Optional[List[Any]] = None,
    nl_answer: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    now_ts = time.time()
    key = _nl_cache_key(session_token, request_id)
    with _NL_LOCK:
        _cleanup_nl_cache_locked(now_ts)
        previous = _NL_RESPONSES.get(key) or {}
        payload = {
            "session_token": session_token,
            "request_id": request_id,
            "status": status,
            "question": question if question is not None else previous.get("question"),
            "results": results if results is not None else previous.get("results"),
            "nl_answer": nl_answer,
            "error": error,
            "updated_at": now_ts,
            "expires_at": now_ts + NL_RESPONSE_TTL_SECONDS,
        }
        _NL_RESPONSES[key] = payload
        _NL_RESPONSES.move_to_end(key)
        return dict(payload)


def _get_nl_state(*, session_token: str, request_id: str) -> Optional[Dict[str, Any]]:
    now_ts = time.time()
    key = _nl_cache_key(session_token, request_id)
    with _NL_LOCK:
        _cleanup_nl_cache_locked(now_ts)
        item = _NL_RESPONSES.get(key)
        if item is None:
            return None
        _NL_RESPONSES.move_to_end(key)
        return dict(item)


def _clear_nl_state_for_session(session_token: str) -> None:
    with _NL_LOCK:
        keys = [k for k, v in _NL_RESPONSES.items() if v.get("session_token") == session_token]
        for key in keys:
            _NL_RESPONSES.pop(key, None)
            task = _NL_RESPONSE_TASKS.pop(key, None)
            if task is not None and not task.done():
                task.cancel()


def _extract_secret(value: Any) -> str:
    if value is None:
        return ""
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        try:
            return str(getter() or "").strip()
        except Exception:
            return ""
    return str(value).strip()


def _format_results_for_nl(results: Optional[List[Any]]) -> str:
    if not results or not isinstance(results, list):
        return "No results found."
    first = results[0]
    if not isinstance(first, dict):
        return "No results found."
    keys = list(first.keys())
    if not keys:
        return "No results found."
    rows = results[:15]
    header = " | ".join(str(k) for k in keys)
    sep = "-" * min(len(header), 120)
    row_lines = []
    for row in rows:
        if isinstance(row, dict):
            row_lines.append(" | ".join(str(row.get(k, "")) for k in keys))
    return "\n".join([header, sep] + row_lines) if row_lines else "No results found."


def _build_nl_messages(question: str, results: Optional[List[Any]]) -> List[Dict[str, str]]:
    results_str = _format_results_for_nl(results)
    return [
        {
            "role": "system",
            "content": (
                "You are a data analyst assistant. Summarize SQL results accurately and concisely. "
                "Only use provided results, and avoid speculation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Results:\n{results_str}\n\n"
                "Provide a concise business summary and mention notable values."
            ),
        },
    ]


def _get_openai_client(llm: Any) -> Optional[AsyncOpenAI]:
    api_key = _extract_secret(getattr(llm, "openai_api_key", None))
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    api_base = str(getattr(llm, "openai_api_base", "") or "").rstrip("/") or None
    return AsyncOpenAI(api_key=api_key, base_url=api_base)


async def _openai_nl_token_stream(question: str, results: Optional[List[Any]], llm: Any):
    client = _get_openai_client(llm)
    if client is None:
        return
    model_name = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "gpt-4o-mini")
    async with client:
        stream = await client.chat.completions.create(
            model=model_name,
            messages=_build_nl_messages(question, results),
            temperature=0.0,
            stream=True,
        )
        async for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if token:
                yield token


async def _generate_nl_answer_async(question: str, results: Optional[List[Any]], llm: Any) -> str:
    if llm is None:
        return ""
    client = _get_openai_client(llm)
    if client is not None:
        try:
            model_name = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "gpt-4o-mini")
            async with client:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=_build_nl_messages(question, results),
                    temperature=0.0,
                    stream=False,
                )
            return str(resp.choices[0].message.content or "").strip()
        except Exception:
            logger.warning("openai_async_nl_generation_failed", exc_info=True)
    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(
        get_foreground_executor(), generate_nl_response, question, results, llm
    )
    return (answer or "").strip()


def _update_chat_history_nl(session: Session, request_id: str, answer: str) -> None:
    with session.lock:
        for item in reversed(session.chat_history):
            if str(item.get("request_id")) == str(request_id):
                item["nl_answer"] = answer
                break
        else:
            if session.chat_history:
                session.chat_history[-1]["nl_answer"] = answer
        # Also backfill nl_answer into the matching ConversationTurn so follow-up
        # queries receive the previous answer in their prompt context.
        if session.conversation_turns:
            session.conversation_turns[-1].nl_answer = answer


def _schedule_nl_background(session: Session, request_id: str, question: str, results: Optional[List[Any]]) -> None:
    if session.llm is None or not isinstance(results, list) or not results:
        return
    key = _nl_cache_key(session.token, request_id)
    with _NL_LOCK:
        _cleanup_nl_cache_locked(time.time())
        existing = _NL_RESPONSES.get(key)
        if existing and existing.get("status") == "ready":
            return
        existing_task = _NL_RESPONSE_TASKS.get(key)
        if existing_task is not None and not existing_task.done():
            return
        _set_nl_state(
            session_token=session.token,
            request_id=request_id,
            status="pending",
            question=question,
            results=results,
            nl_answer=None,
            error=None,
        )

    async def _run() -> None:
        try:
            answer = await _generate_nl_answer_async(question, results, session.llm)
            answer = answer or "No summary available."
            _set_nl_state(
                session_token=session.token,
                request_id=request_id,
                status="ready",
                question=question,
                results=results,
                nl_answer=answer,
                error=None,
            )
            _update_chat_history_nl(session, request_id, answer)
        except Exception as exc:
            _set_nl_state(
                session_token=session.token,
                request_id=request_id,
                status="error",
                question=question,
                results=results,
                nl_answer=None,
                error=str(exc),
            )
            logger.warning("background_nl_generation_failed", exc_info=True)

    task = asyncio.create_task(_run())
    with _NL_LOCK:
        _NL_RESPONSE_TASKS[key] = task

    def _done(_task: asyncio.Task) -> None:
        with _NL_LOCK:
            _NL_RESPONSE_TASKS.pop(key, None)

    task.add_done_callback(_done)


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


def _stream_tokens_from_text(text: str) -> List[str]:
    if not text:
        return []
    pieces = text.split(" ")
    if len(pieces) <= 1:
        return [text]
    return [f"{piece} " if idx < len(pieces) - 1 else piece for idx, piece in enumerate(pieces)]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    intent:           Optional[str]        = None
    nl_answer:        Optional[str]        = None
    sql:              Optional[str]        = None
    results:          Optional[List[Any]]  = None
    row_count:        int                  = 0
    from_cache:       bool                 = False
    error:            Optional[str]        = None
    nl_pending:       bool                 = False
    fallback_used:    bool                 = False
    timing:           Optional[Dict[str, float]] = None
    request_id:       Optional[str]        = None
    query_complexity: Optional[str]        = None  # "deterministic" | "simple_llm" | "complex_llm"


class NLRequest(BaseModel):
    question: str
    results:  List[Any]
    request_id: Optional[str] = None


class NLResponse(BaseModel):
    nl_answer: str


class NLStatusResponse(BaseModel):
    request_id: str
    status: str
    nl_answer: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# POST /api/chat/query
# ---------------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request, session: Session = Depends(get_session)):
    if not session.connected or session.db is None:
        raise HTTPException(400, "Not connected to database. Please connect first.")
    started = time.perf_counter()
    request_id = get_request_id()
    log_event(
        logger,
        logging.INFO,
        "chat_query_start",
        path=str(request.url.path),
        question_len=len(req.question or ""),
    )

    try:
        def _query_worker() -> Dict[str, Any]:
            return legacy_handle_query(
                question=req.question,
                db=session.db,
                db_config=session.db_config,
                sql_chain=session.sql_chain,
                llm=session.llm,
                rag_engine=session.rag_engine,
                embedder=session.embedder,
                chat_history=session.chat_history,
                query_cache=session.query_cache,
                cached_schema_text=session.cached_schema_text,
                session_state=session.session_state,
                conversation_turns=session.conversation_turns,
                reasoning_llm=session.reasoning_llm,
                sql_dialect=session.sql_dialect,
                enable_nolock=session.enable_nolock,
            )

        loop = asyncio.get_running_loop()
        ctx = copy_context()
        result = await loop.run_in_executor(
            get_foreground_executor(),
            lambda: ctx.run(_query_worker),
        )

        # ### STRUCTURED MEMORY — update SessionState from result
        with session.lock:
            us = result.get("updated_state")
            if us:
                ss = session.session_state
                # us may be a SessionState object (sql_engine_u) or a dict (sql_engine legacy)
                _get = us.get if isinstance(us, dict) else lambda k, d=None: getattr(us, k, d)
                ss.last_sql         = _get("last_sql",         ss.last_sql)
                ss.last_table       = _get("last_table",       ss.last_table)
                ss.last_date_col    = _get("last_date_col",    ss.last_date_col)
                ss.last_time_window = _get("last_time_window", ss.last_time_window)
                ss.last_dimensions  = _get("last_dimensions",  ss.last_dimensions)
                ss.last_metrics     = _get("last_metrics",     ss.last_metrics)
                ss.last_filters     = _get("last_filters",     ss.last_filters)

            # Append structured turn
            turn = result.get("conversation_turn")
            if turn is not None:
                append_with_sliding_window(
                    session.conversation_turns,
                    turn,
                    max_items=FOLLOWUP_WINDOW_TURNS,
                )

            # ### MINIMAL CORE — chat history: raw dicts only, no pandas DataFrame
            append_with_sliding_window(session.chat_history, {
                "question":   req.question,
                "sql":        result.get("sql"),
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "from_cache": result.get("from_cache", False),
                "nl_answer":  result.get("nl_answer"),
                "intent":     result.get("intent"),
                "error":      result.get("error"),
                "request_id": request_id,
                # result_rows stored as list-of-dicts for follow-up value detection (no pandas)
                "result_rows": result.get("results") or [],
            }, max_items=FOLLOWUP_WINDOW_TURNS)

        has_rows = bool(isinstance(result.get("results"), list) and result.get("results"))
        if bool(result.get("nl_pending")) and has_rows:
            _schedule_nl_background(
                session=session,
                request_id=request_id,
                question=req.question,
                results=result.get("results"),
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            logger,
            logging.INFO,
            "chat_query_complete",
            elapsed_ms=elapsed_ms,
            from_cache=bool(result.get("from_cache", False)),
            row_count=int(result.get("row_count", 0) or 0),
            has_error=bool(result.get("error")),
        )
        raw_timing = result.get("timing")
        has_total = isinstance(raw_timing, dict) and "total" in raw_timing
        normalized_timing = _normalize_stage_timing(
            raw_timing,
            total_ms=None if has_total else float(elapsed_ms),
        )
        return QueryResponse(
            intent            = result.get("intent"),
            nl_answer         = result.get("nl_answer"),
            sql               = result.get("sql"),
            results           = result.get("results"),
            row_count         = result.get("row_count", 0),
            from_cache        = result.get("from_cache", False),
            error             = result.get("error"),
            nl_pending        = result.get("nl_pending", False),
            fallback_used     = result.get("fallback_used", False),
            timing            = normalized_timing,
            request_id        = request_id,
            query_complexity  = result.get("query_complexity"),
        )
    except Exception:
        logger.exception("chat_query_unhandled_error")
        error_timing = _normalize_stage_timing(
            {"total": round((time.perf_counter() - started) * 1000.0, 2)}
        )
        return QueryResponse(
            intent="DATA_QUERY",
            nl_answer=None,
            sql=None,
            results=None,
            row_count=0,
            from_cache=False,
            error="Internal error while processing query.",
            nl_pending=False,
            fallback_used=False,
            timing=error_timing,
            request_id=request_id,
        )


# ---------------------------------------------------------------------------
# GET /api/chat/nl-response  (one-shot status poll)
# ---------------------------------------------------------------------------
@router.get("/nl-response", response_model=NLStatusResponse)
async def get_nl_response(request_id: str, session: Session = Depends(get_session)):
    if not session.connected:
        raise HTTPException(400, "Not connected.")
    state = _get_nl_state(session_token=session.token, request_id=request_id)
    if state is None:
        return NLStatusResponse(request_id=request_id, status="not_found", nl_answer=None, error=None)
    return NLStatusResponse(
        request_id=request_id,
        status=str(state.get("status") or "pending"),
        nl_answer=state.get("nl_answer"),
        error=state.get("error"),
    )


# ---------------------------------------------------------------------------
# POST /api/chat/nl-response  (compatibility/manual regeneration path)
# ---------------------------------------------------------------------------
@router.post("/nl-response", response_model=NLResponse)
async def nl_response(req: NLRequest, session: Session = Depends(get_session)):
    if not session.connected or session.llm is None:
        raise HTTPException(400, "Not connected.")

    if req.request_id:
        cached = _get_nl_state(session_token=session.token, request_id=req.request_id)
        if cached and cached.get("status") == "ready" and cached.get("nl_answer"):
            answer = str(cached.get("nl_answer"))
            _update_chat_history_nl(session, req.request_id, answer)
            return NLResponse(nl_answer=answer)

    answer = await _generate_nl_answer_async(req.question, req.results, session.llm)
    answer = answer or "No summary available."
    if req.request_id:
        _set_nl_state(
            session_token=session.token,
            request_id=req.request_id,
            status="ready",
            question=req.question,
            results=req.results,
            nl_answer=answer,
            error=None,
        )
        _update_chat_history_nl(session, req.request_id, answer)
    elif session.chat_history:
        with session.lock:
            if session.chat_history:
                session.chat_history[-1]["nl_answer"] = answer
    return NLResponse(nl_answer=answer)


# ---------------------------------------------------------------------------
# GET /api/chat/nl-stream  (SSE token stream)
# ---------------------------------------------------------------------------
@router.get("/nl-stream")
async def nl_stream(request: Request, request_id: str, session: Session = Depends(get_session)):
    if not session.connected or session.llm is None:
        raise HTTPException(400, "Not connected.")

    async def _event_generator():
        key = _nl_cache_key(session.token, request_id)
        state = _get_nl_state(session_token=session.token, request_id=request_id)
        if state is None:
            yield _sse_event("error", {"request_id": request_id, "error": "nl_state_not_found"})
            return

        if state.get("status") == "ready" and state.get("nl_answer"):
            answer_text = str(state.get("nl_answer") or "")
            for token in _stream_tokens_from_text(answer_text):
                if await request.is_disconnected():
                    return
                yield _sse_event("token", {"request_id": request_id, "token": token})
            yield _sse_event("done", {"request_id": request_id, "nl_answer": answer_text})
            return

        if state.get("status") == "error":
            yield _sse_event("error", {"request_id": request_id, "error": state.get("error") or "nl_generation_failed"})
            return

        question = str(state.get("question") or "")
        results = state.get("results") if isinstance(state.get("results"), list) else []

        # Cancel any pending background task — we'll stream live instead
        with _NL_LOCK:
            running = _NL_RESPONSE_TASKS.pop(key, None)
        if running is not None and not running.done():
            running.cancel()

        try:
            parts: List[str] = []
            async for token in _openai_nl_token_stream(question, results, session.llm):
                if await request.is_disconnected():
                    return
                parts.append(token)
                yield _sse_event("token", {"request_id": request_id, "token": token})
            answer_text = "".join(parts).strip() or "No summary available."
            _set_nl_state(
                session_token=session.token,
                request_id=request_id,
                status="ready",
                question=question,
                results=results,
                nl_answer=answer_text,
                error=None,
            )
            _update_chat_history_nl(session, request_id, answer_text)
            yield _sse_event("done", {"request_id": request_id, "nl_answer": answer_text})
        except Exception:
            logger.warning("nl_stream_live_stream_failed", exc_info=True)
            yield _sse_event("error", {"request_id": request_id, "error": "nl_stream_failed"})

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# POST /api/chat/clear
# ---------------------------------------------------------------------------
@router.post("/clear")
def clear_chat(session: Session = Depends(get_session)):
    with session.lock:
        session.chat_history       = []
        session.conversation_turns = []
        session.session_state      = SessionState(dialect=session.sql_dialect)
    _clear_nl_state_for_session(session.token)
    return {"success": True}


@router.post("/clear-cache")
def clear_cache(session: Session = Depends(get_session)):
    if session.user.get("role") != "Admin":
        raise HTTPException(403, "Only admins can clear cache")
    with session.lock:
        if hasattr(session.query_cache, "clear"):
            session.query_cache.clear()
        session.query_cache = QueryCacheStore()
    _clear_nl_state_for_session(session.token)
    invalidate_runtime_caches(reason="manual_clear_cache")
    return {"success": True}
