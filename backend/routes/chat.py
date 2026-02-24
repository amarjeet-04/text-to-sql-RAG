"""### MINIMAL CORE — Chat routes (no pandas in the request hot path)."""
import asyncio
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextvars import copy_context
from datetime import datetime
from typing import Optional, List, Any, Dict

import httpx
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.session import Session, SessionState, ConversationTurn, QueryCacheStore
from backend.services.sql_engine import (
    handle_query as legacy_handle_query,
    generate_nl_response,
    detect_intent_simple,
    build_domain_digest,
    _read_stored_procedure_file,
    invalidate_runtime_caches,
)
from backend.services.guarded_text2sql_pipeline import handle_query as guarded_handle_query
from backend.services.runtime import (
    run_with_timeout,
    log_event,
    get_request_id,
    get_foreground_executor,
)
from backend.routes.deps import get_session
from app.db_utils import execute_query_safe, validate_sql_dry_run

router = APIRouter(prefix="/api/chat", tags=["chat"])
USE_GUARDED_PIPELINE = os.getenv("USE_GUARDED_PIPELINE", "true").lower() in {"1", "true", "yes", "on"}
logger = logging.getLogger("chat_route")
GUARDED_PIPELINE_BUDGET_S = max(5.0, float(os.getenv("GUARDED_PIPELINE_BUDGET_S", "20")))
GUARDED_SCHEMA_CHAR_BUDGET = max(4000, int(os.getenv("GUARDED_SCHEMA_CHAR_BUDGET", "8000")))
NL_RESPONSE_TTL_SECONDS = max(30, int(os.getenv("NL_RESPONSE_TTL_SECONDS", "180")))
NL_RESPONSE_MAX_ENTRIES = max(64, int(os.getenv("NL_RESPONSE_MAX_ENTRIES", "512")))
NL_STREAM_WAIT_SECONDS = max(2.0, float(os.getenv("NL_STREAM_WAIT_SECONDS", "15")))
_NL_RESPONSES: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_NL_RESPONSE_TASKS: Dict[str, asyncio.Task] = {}
_NL_LOCK = threading.RLock()


class _LLMAdapter:
    """Adapter to satisfy llm.chat(messages, timeout_s) expected by guarded pipeline."""

    def __init__(self, llm):
        self._llm = llm

    def chat(self, messages: List[Dict[str, str]], timeout_s: float) -> str:
        prompt = "\n\n".join(str(m.get("content", "")) for m in messages or [])
        started = time.perf_counter()

        def _invoke():
            resp = self._llm.invoke(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)

        try:
            out = run_with_timeout(_invoke, timeout_s=max(0.1, float(timeout_s)))
            log_event(
                logger,
                logging.INFO,
                "guarded_llm_call_ok",
                timeout_s=float(timeout_s),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                prompt_chars=len(prompt),
            )
            return out
        except FuturesTimeoutError as exc:
            log_event(
                logger,
                logging.WARNING,
                "guarded_llm_call_timeout",
                timeout_s=float(timeout_s),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                prompt_chars=len(prompt),
            )
            raise TimeoutError(f"llm_timeout_{timeout_s}s") from exc


def _compact_schema_for_guarded(schema_text: str, table_hints: List[str]) -> str:
    text = schema_text or ""
    if len(text) <= GUARDED_SCHEMA_CHAR_BUDGET:
        return text
    hints = {h.lower().strip() for h in table_hints if isinstance(h, str) and h.strip()}
    if not hints:
        return text[:GUARDED_SCHEMA_CHAR_BUDGET]

    # Keep CREATE TABLE blocks matching hints first.
    blocks = re.split(r"(?i)(?=CREATE\s+TABLE\s+)", text)
    selected: List[str] = []
    for block in blocks:
        low = block.lower()
        if any(h in low for h in hints):
            selected.append(block)
        if sum(len(x) for x in selected) >= GUARDED_SCHEMA_CHAR_BUDGET:
            break
    if not selected:
        return text[:GUARDED_SCHEMA_CHAR_BUDGET]
    compact = "\n\n".join(selected)
    return compact[:GUARDED_SCHEMA_CHAR_BUDGET]


def _rows_from_df(df) -> List[Dict[str, Any]]:
    if df is None:
        return []
    try:
        return df.to_dict(orient="records")
    except Exception:
        return []


def _run_guarded_data_query(req: "QueryRequest", session: Session) -> Dict[str, Any]:
    question = req.question
    schema_compact = session.cached_schema_text or (session.db.get_table_info() if session.db is not None else "")
    rules_compact = build_domain_digest(_read_stored_procedure_file())

    retrieved_tables_hint: List[str] = []
    if session.rag_engine is not None:
        try:
            rag_ctx = session.rag_engine.retrieve(question, top_k=2, fast_mode=True, skip_for_followup=False, intent_key="chat")
            retrieved_tables_hint = list(rag_ctx.get("tables", []) or [])[:6]
        except TypeError:
            try:
                rag_ctx = session.rag_engine.retrieve(question, top_k=2)
                retrieved_tables_hint = list(rag_ctx.get("tables", []) or [])[:6]
            except Exception:
                retrieved_tables_hint = []
        except Exception:
            retrieved_tables_hint = []

    schema_compact = _compact_schema_for_guarded(schema_compact, retrieved_tables_hint)
    log_event(
        logger,
        logging.INFO,
        "guarded_prompt_payload",
        schema_chars=len(schema_compact or ""),
        rules_chars=len(rules_compact or ""),
        hint_tables=len(retrieved_tables_hint),
    )

    llm_adapter = _LLMAdapter(session.llm)
    timeout_seconds = max(5, int(getattr(session.db_config, "query_timeout", 30)))
    # Cap the guarded pipeline's DB execution at 20 s so a slow complex SQL
    # fails fast and the outer exception handler falls back to the legacy
    # pipeline (which generates simpler, faster SQL from its own cache/LLM).
    guarded_db_timeout = min(timeout_seconds, 20)

    def _db_execute(sql: str):
        df, err = execute_query_safe(session.db, sql, timeout_seconds=guarded_db_timeout, max_rows=1000)
        if err:
            raise RuntimeError(err)
        return _rows_from_df(df)

    def _db_dry_run(sql: str):
        ok, err = validate_sql_dry_run(session.db, sql)
        return bool(ok), (err or "")

    guarded_result = guarded_handle_query(
        question=question,
        schema_compact=schema_compact,
        rules_compact=rules_compact,
        retrieved_tables_hint=retrieved_tables_hint,
        db_execute=_db_execute,
        db_dry_run=_db_dry_run,
        llm=llm_adapter,
        dialect="mssql",
        current_date=datetime.now().strftime("%Y-%m-%d"),
        budget_seconds=GUARDED_PIPELINE_BUDGET_S,
    )

    status = guarded_result.get("status")
    sql = guarded_result.get("sql")
    rows = guarded_result.get("rows")
    elapsed_ms = int(guarded_result.get("elapsed_ms") or 0)
    reason = guarded_result.get("reason") or ""
    intent_json = guarded_result.get("intent_json") or {}

    # If the guarded pipeline hit the time budget but produced a valid SQL,
    # try to execute it directly here rather than falling back to the slow legacy path.
    if status == "timeout" and sql and not rows:
        try:
            timeout_seconds = max(5, int(getattr(session.db_config, "query_timeout", 30)))
            df, exec_err = execute_query_safe(session.db, sql, timeout_seconds=timeout_seconds, max_rows=1000)
            if not exec_err and df is not None:
                rows = _rows_from_df(df)
                status = "ok"    # promote to success
                reason = ""
                log_event(logger, logging.INFO, "guarded_budget_exceeded_but_executed",
                          rows=len(rows), sql_chars=len(sql))
        except Exception:
            pass  # fall through to raise below

    # If guarded pipeline timed out (LLM call timeout, budget exceeded, or DB
    # execution timeout on a complex guarded SQL), raise so the outer try/except
    # in query() falls back to the legacy pipeline which generates simpler SQL.
    if status == "timeout" or (
        status == "error" and (
            "timeout" in reason.lower()
            or "query timed out" in reason.lower()
            or "query_timeout" in reason.lower()
        )
    ):
        raise TimeoutError(f"guarded_pipeline_timeout: {reason}")

    # Update minimal structured memory for follow-ups.
    if sql:
        session.session_state.last_sql = sql
        session.session_state.last_table = (intent_json.get("tables_used") or [None])[0]
        tw = intent_json.get("time_window") or {}
        session.session_state.last_time_window = {
            "start": tw.get("start"),
            "end": tw.get("end_exclusive"),
        }
        session.session_state.last_dimensions = [str(x) for x in (intent_json.get("group_by") or [])][:6]
        session.session_state.last_metrics = [k for k, v in (intent_json.get("metrics") or {}).items() if v][:6]
        session.session_state.last_filters = ["booking_status_exclusions"] if (intent_json.get("filters") or {}).get("booking_status_exclusions") else []

    if status == "ok":
        out_rows = rows if isinstance(rows, list) else []
        turn = ConversationTurn(
            question=question,
            sql=sql,
            topic="guarded:data",
            columns=list(out_rows[0].keys()) if out_rows else [],
            row_count=len(out_rows),
            status="ok",
        )
        return {
            "intent": "DATA_QUERY",
            "nl_answer": None,
            "sql": sql,
            "results": out_rows,
            "row_count": len(out_rows),
            "from_cache": False,
            "error": None,
            "updated_state": session.session_state.to_dict(),
            "nl_pending": len(out_rows) > 0,
            "conversation_turn": turn,
            "fallback_used": False,
            "timing": {"total": float(elapsed_ms)},
        }

    # rejected/timeout/error
    return {
        "intent": "DATA_QUERY",
        "nl_answer": reason if isinstance(reason, str) and reason else ("Query rejected by guarded pipeline." if status == "rejected" else None),
        "sql": sql,
        "results": [] if status != "error" else None,
        "row_count": 0,
        "from_cache": False,
        "error": reason if status in {"rejected", "error"} else None,
        "updated_state": session.session_state.to_dict(),
        "nl_pending": False,
        "fallback_used": bool(status == "timeout" and sql),
        "timing": {"total": float(elapsed_ms)},
    }


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


def _is_deepseek_llm(llm: Any) -> bool:
    model_name = str(getattr(llm, "model_name", "") or getattr(llm, "model", "")).lower()
    api_base = str(getattr(llm, "openai_api_base", "")).lower()
    return "deepseek" in model_name or "deepseek" in api_base


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


async def _deepseek_nl_answer_async(question: str, results: Optional[List[Any]], llm: Any) -> str:
    model_name = str(getattr(llm, "model_name", "") or "deepseek-chat")
    api_base = str(getattr(llm, "openai_api_base", "") or "https://api.deepseek.com").rstrip("/")
    api_key = _extract_secret(getattr(llm, "openai_api_key", None))
    if not api_key:
        return ""
    timeout = httpx.Timeout(connect=5.0, read=45.0, write=20.0, pool=10.0)
    payload = {
        "model": model_name,
        "messages": _build_nl_messages(question, results),
        "temperature": 0.0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
        resp = await client.post(f"{api_base}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        body = resp.json()
    try:
        return str(body["choices"][0]["message"]["content"]).strip()
    except Exception:
        return ""


async def _deepseek_nl_token_stream(question: str, results: Optional[List[Any]], llm: Any):
    model_name = str(getattr(llm, "model_name", "") or "deepseek-chat")
    api_base = str(getattr(llm, "openai_api_base", "") or "https://api.deepseek.com").rstrip("/")
    api_key = _extract_secret(getattr(llm, "openai_api_key", None))
    if not api_key:
        return
    timeout = httpx.Timeout(connect=5.0, read=60.0, write=20.0, pool=10.0)
    payload = {
        "model": model_name,
        "messages": _build_nl_messages(question, results),
        "temperature": 0.0,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
        async with client.stream("POST", f"{api_base}/chat/completions", headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                payload_line = line[5:].strip()
                if payload_line == "[DONE]":
                    break
                try:
                    parsed = json.loads(payload_line)
                except Exception:
                    continue
                choices = parsed.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                token = delta.get("content")
                if token:
                    yield str(token)


async def _generate_nl_answer_async(question: str, results: Optional[List[Any]], llm: Any) -> str:
    if llm is None:
        return ""
    if _is_deepseek_llm(llm):
        try:
            answer = await _deepseek_nl_answer_async(question, results, llm)
            if answer:
                return answer
        except Exception:
            logger.warning("deepseek_async_nl_generation_failed", exc_info=True)
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
                return
        if session.chat_history:
            session.chat_history[-1]["nl_answer"] = answer


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
    intent:       Optional[str]        = None
    nl_answer:    Optional[str]        = None
    sql:          Optional[str]        = None
    results:      Optional[List[Any]]  = None
    row_count:    int                  = 0
    from_cache:   bool                 = False
    error:        Optional[str]        = None
    nl_pending:   bool                 = False
    fallback_used: bool                = False
    timing:       Optional[Dict[str, float]] = None
    request_id:   Optional[str]        = None


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
        guarded_enabled=USE_GUARDED_PIPELINE,
        question_len=len(req.question or ""),
    )

    intent_hint = detect_intent_simple(req.question) or ""
    use_guarded = USE_GUARDED_PIPELINE and intent_hint not in {"GREETING", "THANKS", "HELP", "FAREWELL", "OUT_OF_SCOPE", "CLARIFICATION_NEEDED"}

    try:
        def _run_legacy() -> Dict[str, Any]:
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

        def _query_worker() -> Dict[str, Any]:
            if use_guarded:
                try:
                    return _run_guarded_data_query(req, session)
                except Exception:
                    logger.warning("guarded_pipeline_failed_falling_back", exc_info=True)
                    return _run_legacy()
            return _run_legacy()

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
                ss.last_sql         = us.get("last_sql",         ss.last_sql)
                ss.last_table       = us.get("last_table",       ss.last_table)
                ss.last_date_col    = us.get("last_date_col",    ss.last_date_col)
                ss.last_time_window = us.get("last_time_window", ss.last_time_window)
                ss.last_dimensions  = us.get("last_dimensions",  ss.last_dimensions)
                ss.last_metrics     = us.get("last_metrics",     ss.last_metrics)
                ss.last_filters     = us.get("last_filters",     ss.last_filters)

            # Append structured turn
            turn = result.get("conversation_turn")
            if turn is not None:
                session.conversation_turns.append(turn)

            # ### MINIMAL CORE — chat history: raw dicts only, no pandas DataFrame
            session.chat_history.append({
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
            })

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
        return QueryResponse(
            intent        = result.get("intent"),
            nl_answer     = result.get("nl_answer"),
            sql           = result.get("sql"),
            results       = result.get("results"),
            row_count     = result.get("row_count", 0),
            from_cache    = result.get("from_cache", False),
            error         = result.get("error"),
            nl_pending    = result.get("nl_pending", False),
            fallback_used = result.get("fallback_used", False),
            timing        = result.get("timing"),
            request_id    = request_id,
        )
    except Exception:
        logger.exception("chat_query_unhandled_error")
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
            timing={"total": round((time.perf_counter() - started) * 1000.0, 2)},
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
        if _is_deepseek_llm(session.llm):
            with _NL_LOCK:
                running = _NL_RESPONSE_TASKS.pop(key, None)
            if running is not None and not running.done():
                running.cancel()
            try:
                parts: List[str] = []
                async for token in _deepseek_nl_token_stream(question, results, session.llm):
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
                return
            except Exception:
                logger.warning("nl_stream_deepseek_live_stream_failed", exc_info=True)

        deadline = time.monotonic() + NL_STREAM_WAIT_SECONDS
        while time.monotonic() < deadline:
            if await request.is_disconnected():
                return
            state = _get_nl_state(session_token=session.token, request_id=request_id)
            if state is None:
                yield _sse_event("error", {"request_id": request_id, "error": "nl_state_not_found"})
                return
            status = str(state.get("status") or "pending")
            if status == "ready":
                answer_text = str(state.get("nl_answer") or "")
                for token in _stream_tokens_from_text(answer_text):
                    if await request.is_disconnected():
                        return
                    yield _sse_event("token", {"request_id": request_id, "token": token})
                yield _sse_event("done", {"request_id": request_id, "nl_answer": answer_text})
                return
            if status == "error":
                yield _sse_event("error", {"request_id": request_id, "error": state.get("error") or "nl_generation_failed"})
                return
            await asyncio.sleep(0.12)

        yield _sse_event("error", {"request_id": request_id, "error": "nl_stream_timeout"})

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
