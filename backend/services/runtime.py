"""
Runtime utilities:
- shared thread pool with safe shutdown
- request/session context for structured logs
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional

_LOGGER = logging.getLogger("runtime")

_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")
_SESSION_ID: ContextVar[str] = ContextVar("session_id", default="-")

_BG_EXECUTOR: Optional[ThreadPoolExecutor] = None
_FG_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()
_PENDING_LOCK = threading.Lock()
_PENDING_FUTURES: set[Future] = set()
_MAX_WORKERS = max(4, int(os.getenv("APP_THREADPOOL_MAX_WORKERS", "8")))
_FOREGROUND_WORKERS = max(2, int(os.getenv("APP_FOREGROUND_MAX_WORKERS", "4")))


def get_request_id() -> str:
    return _REQUEST_ID.get() or "-"


def get_session_id() -> str:
    return _SESSION_ID.get() or "-"


def set_request_id(request_id: Optional[str]) -> str:
    rid = (request_id or "").strip() or str(uuid.uuid4())
    _REQUEST_ID.set(rid)
    return rid


def set_session_id(session_id: Optional[str]) -> str:
    sid = (session_id or "").strip() or "-"
    _SESSION_ID.set(sid)
    return sid


def clear_context() -> None:
    _REQUEST_ID.set("-")
    _SESSION_ID.set("-")


def structured_fields(**extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "request_id": get_request_id(),
        "session_id": get_session_id(),
    }
    payload.update(extra)
    return payload


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    payload = structured_fields(event=event, **fields)
    logger.log(level, json.dumps(payload, default=str, ensure_ascii=True))


def _get_bg_executor() -> ThreadPoolExecutor:
    global _BG_EXECUTOR
    if _BG_EXECUTOR is not None:
        return _BG_EXECUTOR
    with _EXECUTOR_LOCK:
        if _BG_EXECUTOR is None:
            _BG_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="text2sql-bg")
        return _BG_EXECUTOR


def _get_fg_executor() -> ThreadPoolExecutor:
    global _FG_EXECUTOR
    if _FG_EXECUTOR is not None:
        return _FG_EXECUTOR
    with _EXECUTOR_LOCK:
        if _FG_EXECUTOR is None:
            _FG_EXECUTOR = ThreadPoolExecutor(max_workers=_FOREGROUND_WORKERS, thread_name_prefix="text2sql-fg")
        return _FG_EXECUTOR


def get_foreground_executor() -> ThreadPoolExecutor:
    return _get_fg_executor()


def submit_background_task(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    future = _get_bg_executor().submit(fn, *args, **kwargs)
    with _PENDING_LOCK:
        _PENDING_FUTURES.add(future)

    def _done(fut: Future) -> None:
        with _PENDING_LOCK:
            _PENDING_FUTURES.discard(fut)

    future.add_done_callback(_done)
    return future


def run_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Any:
    future = _get_fg_executor().submit(fn)
    with _PENDING_LOCK:
        _PENDING_FUTURES.add(future)

    def _done(fut: Future) -> None:
        with _PENDING_LOCK:
            _PENDING_FUTURES.discard(fut)

    future.add_done_callback(_done)
    try:
        return future.result(timeout=max(0.05, float(timeout_s)))
    except FuturesTimeoutError:
        future.cancel()
        raise


def shutdown_shared_executor(wait: bool = False) -> None:
    global _BG_EXECUTOR, _FG_EXECUTOR
    with _EXECUTOR_LOCK:
        if _BG_EXECUTOR is None and _FG_EXECUTOR is None:
            return
        with _PENDING_LOCK:
            pending = list(_PENDING_FUTURES)
            _PENDING_FUTURES.clear()
        for fut in pending:
            fut.cancel()
        if _BG_EXECUTOR is not None:
            _BG_EXECUTOR.shutdown(wait=wait, cancel_futures=True)
            _BG_EXECUTOR = None
        if _FG_EXECUTOR is not None:
            _FG_EXECUTOR.shutdown(wait=wait, cancel_futures=True)
            _FG_EXECUTOR = None
        _LOGGER.info("shared_executor_shutdown")
