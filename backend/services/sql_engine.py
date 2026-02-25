"""
SQL Engine - Core logic extracted from streamlit_app.py.
Handles intent detection, SQL generation, validation, caching, and execution.
"""
import re
import json
import os
import sys
import time
import socket
import logging
import threading
import hashlib
from concurrent.futures import TimeoutError as FuturesTimeoutError
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import OrderedDict
from contextlib import closing

import numpy as np
import pandas as pd
import sqlalchemy
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Ensure app/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.db_utils import DatabaseConfig, create_database_with_views, execute_query_safe, get_views_and_tables, validate_sql_dry_run
from app.rag.rag_engine import RAGEngine
from backend.services.session import ConversationTurn, SessionState, serialize_conversation_turns
from backend.services.runtime import run_with_timeout, submit_background_task, log_event

logger = logging.getLogger("sql_engine")

# --- Constants ---
FOLLOW_UP_WORDS = {"that", "those", "it", "them", "this", "more", "instead", "above"}
DATABASE_ENTITY_WORDS = {
    "country", "city", "supplier", "agent", "product", "hotel", "chain", "booking",
    "nationality", "customer", "region", "sales", "revenue", "profit", "date", "amount",
    "checkin", "checkout", "details", "information", "data", "records", "rows",
    "filter", "show", "get", "list",
}
MAX_CACHE_SIZE = max(50, int(os.getenv("QUERY_CACHE_MAX_SIZE", os.getenv("QUERY_RESULT_CACHE_MAX_ENTRIES", "300"))))
BLOCKED_KEYWORDS = {"drop", "delete", "truncate", "insert", "update", "alter", "create", "exec", "execute", "grant", "revoke"}
DEFAULT_SQL_DIALECT = "sqlserver"
PROMPT_BUDGET_CHARS = 20000
LLM_SQL_TIMEOUT_MS = int(os.getenv("LLM_SQL_TIMEOUT_MS", "8000"))
ENABLE_LLM_SLA = os.getenv("ENABLE_LLM_SLA", "true").lower() in {"1", "true", "yes", "on"}
SCHEMA_CACHE_TTL_SECONDS = max(300, int(os.getenv("SCHEMA_CACHE_TTL_SECONDS", "21600")))
SCHEMA_CACHE_MAX_ENTRIES = max(8, int(os.getenv("SCHEMA_CACHE_MAX_ENTRIES", "32")))
QUERY_RESULT_CACHE_TTL_SECONDS = max(30, int(os.getenv("QUERY_CACHE_TTL_SECONDS", os.getenv("QUERY_RESULT_CACHE_TTL_SECONDS", "300"))))
QUERY_RESULT_RELATIVE_TTL_SECONDS = max(15, int(os.getenv("QUERY_RESULT_RELATIVE_TTL_SECONDS", "60")))
GLOBAL_QUERY_CACHE_MAX_ENTRIES = max(50, int(os.getenv("QUERY_CACHE_MAX_SIZE", os.getenv("QUERY_RESULT_CACHE_MAX_ENTRIES", "300"))))
QUERY_CACHE_ENABLE_SEMANTIC = os.getenv("QUERY_CACHE_ENABLE_SEMANTIC", "false").lower() in {"1", "true", "yes", "on"}
QUERY_CACHE_SEMANTIC_THRESHOLD = float(os.getenv("QUERY_CACHE_SEMANTIC_THRESHOLD", "0.97"))
ENABLE_QUERY_FRESHNESS_MARKER = os.getenv("QUERY_CACHE_ENABLE_FRESHNESS_MARKER", "true").lower() in {"1", "true", "yes", "on"}

# Broad / vague business-overview detection.
# This is a deliberate "de-clevering" guardrail to prevent the LLM from
# inventing complex UNION ALL multi-view queries for questions that should
# return a simple KPI snapshot.
BUSINESS_OVERVIEW_PHRASES = {
    "how my business looks",
    "how my business looks like",
    "how does my business look",
    "how does my business look like",
    "business looks like",
    "business overview",
    "overall overview",
    "overall performance",
    "business performance",
    "company performance",
    "overall summary",
    "summary of business",
    "kpi",
    "kpis",
    "dashboard",
    "health of business",
}

BUSINESS_OVERVIEW_NEGATORS = {
    # If any of these appear, it's likely *not* a single-row overview.
    "trend",
    "monthly",
    "weekly",
    "daily",
    "by agent",
    "by country",
    "by city",
    "by supplier",
    "by hotel",
    "by chain",
    "agent wise",
    "country wise",
    "city wise",
    "supplier wise",
    "top ",
    "bottom ",
    "best ",
    "worst ",
    "compare",
    "vs",
    "versus",
    "breakdown",
    "split",
}

# Keyword-to-dimension hints for top/bottom-N routing.
TOPN_DIMENSION_HINTS = [
    ({"chain"}, ["chain", "hotelchain"]),
    ({"hotel", "hotels", "resort", "resorts"}, ["hotelname", "productname", "hotel"]),
    ({"product", "products"}, ["productname", "hotelname", "product"]),
    ({"agent", "agents"}, ["agentname", "agentcode"]),
    ({"supplier", "suppliers"}, ["suppliername"]),
    ({"city", "cities"}, ["city"]),
    ({"country", "countries"}, ["country", "agentcountry"]),
    ({"nationality", "clientnationality"}, ["clientnationality", "nationality"]),
    ({"booking", "bookings"}, ["bookingid", "booking_id", "bookingno", "voucherid", "confirmationno"]),
]

TOPN_DEFAULT_DIMENSION_CANDIDATES = [
    "productname",
    "hotelname",
    "bookingid",
    "booking_id",
    "agentname",
    "suppliername",
    "country",
    "city",
]

RANKING_ENTITY_TOKENS = [
    ({"supplier", "suppliers"}, {"suppliername", "supplierid", "employeeid"}),
    ({"agent", "agents"}, {"agentname", "agentid", "agentcode"}),
    ({"country", "countries"}, {"country", "countryid", "productcountryid", "agentcountry"}),
    ({"city", "cities"}, {"city", "cityid", "productcityid", "agentcity"}),
    ({"hotel", "hotels", "product", "products", "resort", "resorts"}, {"hotelname", "productname", "productid", "hotelid"}),
    ({"chain"}, {"chain", "hotelchain"}),
    ({"nationality", "clientnationality"}, {"clientnationality", "nationality"}),
]

_SCHEMA_PROFILE_CACHE: Dict[int, Dict[str, Any]] = {}
_TABLE_EXISTS_CACHE: Dict[Tuple[int, str], bool] = {}
_SCHEMA_RUNTIME_LOCK = threading.RLock()


# --- Global Schema Cache (cross-session) ---
# Avoids re-loading schema + rebuilding FAISS index when reconnecting to the same DB.
_GLOBAL_SCHEMA_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_GLOBAL_SCHEMA_CACHE_TTL = SCHEMA_CACHE_TTL_SECONDS
_GLOBAL_SCHEMA_CACHE_LOCK = threading.RLock()


def _get_global_schema_cache(host: str, port: str, database: str) -> Optional[Dict[str, Any]]:
    """Return cached schema data if fresh, else None."""
    key = (host.lower(), str(port), database.lower())
    with _GLOBAL_SCHEMA_CACHE_LOCK:
        cached = _GLOBAL_SCHEMA_CACHE.get(key)
        if cached and (time.time() - cached["timestamp"]) < _GLOBAL_SCHEMA_CACHE_TTL:
            logger.info(f"Global schema cache HIT for {host}:{port}/{database}")
            return cached
        if cached:
            _GLOBAL_SCHEMA_CACHE.pop(key, None)
    return None


def _set_global_schema_cache(host: str, port: str, database: str, schema_text: str, rag_engine, embedder):
    """Store schema data in global cache."""
    key = (host.lower(), str(port), database.lower())
    with _GLOBAL_SCHEMA_CACHE_LOCK:
        _GLOBAL_SCHEMA_CACHE[key] = {
            "schema_text": schema_text,
            "rag_engine": rag_engine,
            "embedder": embedder,
            "timestamp": time.time(),
        }
        if len(_GLOBAL_SCHEMA_CACHE) > SCHEMA_CACHE_MAX_ENTRIES:
            oldest_key = min(_GLOBAL_SCHEMA_CACHE, key=lambda k: _GLOBAL_SCHEMA_CACHE[k]["timestamp"])
            _GLOBAL_SCHEMA_CACHE.pop(oldest_key, None)
    logger.info(f"Global schema cache SET for {host}:{port}/{database}")


# --- Global Query Cache (cross-session, with TTL) ---
# Serves repeated questions without LLM or DB calls.

class GlobalQueryCache:
    """Thread-safe query cache.

    Fast path is exact-match lookup in O(1) by normalized question + DB identity.
    Semantic fallback is optional and disabled by default.
    """

    def __init__(
        self,
        max_size: int = GLOBAL_QUERY_CACHE_MAX_ENTRIES,
        ttl_seconds: int = QUERY_RESULT_CACHE_TTL_SECONDS,
        enable_semantic: bool = QUERY_CACHE_ENABLE_SEMANTIC,
        semantic_threshold: float = QUERY_CACHE_SEMANTIC_THRESHOLD,
    ):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.enable_semantic = enable_semantic
        self.semantic_threshold = semantic_threshold
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.RLock()

    def _db_identity(self, db: Optional[SQLDatabase]) -> str:
        if db is None:
            return "unknown"
        try:
            url = db._engine.url
            return f"{(url.host or '').lower()}:{url.port or ''}/{(url.database or '').lower()}"
        except Exception:
            return "unknown"

    def _normalize_question(self, question: str) -> str:
        return " ".join((question or "").strip().lower().split())

    def _build_cache_key(self, db_id: str, question: str) -> str:
        payload = f"{db_id}|{self._normalize_question(question)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _is_expired(self, entry: Dict[str, Any], now_ts: float) -> bool:
        ttl = int(entry.get("ttl_seconds") or self.ttl)
        ts = float(entry.get("timestamp") or 0.0)
        return ts <= 0.0 or (now_ts - ts) > ttl

    def _freshness_matches(self, entry: Dict[str, Any], db: Optional[SQLDatabase]) -> bool:
        expected = entry.get("freshness_marker")
        if not expected or not ENABLE_QUERY_FRESHNESS_MARKER or db is None:
            return True
        try:
            actual = _compute_query_freshness_marker(db, entry.get("sql") or "")
            return actual == expected
        except Exception:
            return True

    def _evict_if_needed(self, now_ts: float) -> None:
        # O(1) LRU eviction; also drop expired entries encountered at head.
        while self._cache:
            first_key = next(iter(self._cache))
            first_entry = self._cache[first_key]
            if self._is_expired(first_entry, now_ts):
                self._cache.pop(first_key, None)
                continue
            break
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def find(self, question: str, embedder, db: Optional[SQLDatabase] = None) -> Tuple[Optional[str], Optional[Any]]:
        """Lookup cached result.

        Exact lookup first (O(1)); semantic fallback only on exact miss and when enabled.
        """
        if is_time_sensitive(question=question, sql=""):
            return None, None
        db_id = self._db_identity(db)
        cache_key = self._build_cache_key(db_id, question)
        now_ts = time.time()

        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is not None:
                if self._is_expired(entry, now_ts) or not self._freshness_matches(entry, db):
                    self._cache.pop(cache_key, None)
                else:
                    self._cache.move_to_end(cache_key)
                    log_event(logger, logging.INFO, "global_cache_hit", mode="exact")
                    return entry.get("sql"), entry.get("df")

        if not self.enable_semantic or embedder is None:
            return None, None

        # Optional slow fallback path.
        try:
            query_emb = np.array(embedder.embed_query(question), dtype=np.float32)
        except Exception:
            return None, None

        entities = extract_key_values(question)
        best_score = -1.0
        best_key = None
        best_entry = None
        with self._lock:
            for key, entry in self._cache.items():
                if entry.get("db_id") != db_id:
                    continue
                if self._is_expired(entry, now_ts):
                    continue
                emb = entry.get("embedding")
                if emb is None:
                    continue
                cached_entities = set(entry.get("entities") or [])
                if entities and cached_entities and entities != cached_entities:
                    continue
                score = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-9))
                if score > best_score:
                    best_score = score
                    best_key = key
                    best_entry = entry
            if best_entry is not None and best_score >= self.semantic_threshold and self._freshness_matches(best_entry, db):
                self._cache.move_to_end(best_key)
                log_event(logger, logging.INFO, "global_cache_hit", mode="semantic", score=round(best_score, 4))
                return best_entry.get("sql"), best_entry.get("df")
        return None, None

    def add(self, question: str, sql: str, df, embedder, db: Optional[SQLDatabase] = None):
        """Add exact-match cache entry.

        For time-sensitive prompts, bypass caching to avoid stale answers.
        """
        if is_time_sensitive(question=question, sql=sql):
            return
        db_id = self._db_identity(db)
        cache_key = self._build_cache_key(db_id, question)
        now_ts = time.time()

        embedding = None
        if self.enable_semantic and embedder is not None:
            try:
                embedding = np.array(embedder.embed_query(question), dtype=np.float32)
            except Exception:
                embedding = None

        entry = {
            "question": question,
            "sql": sql,
            "df": df,
            "embedding": embedding,
            "timestamp": now_ts,
            "ttl_seconds": self.ttl,
            "db_id": db_id,
            "entities": list(extract_key_values(question)),
            "freshness_marker": _compute_query_freshness_marker(db, sql) if ENABLE_QUERY_FRESHNESS_MARKER else None,
        }
        with self._lock:
            self._cache[cache_key] = entry
            self._cache.move_to_end(cache_key)
            self._evict_if_needed(now_ts)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


_GLOBAL_QUERY_CACHE = GlobalQueryCache()


def invalidate_runtime_caches(reason: str = "manual") -> None:
    """Invalidation hook for reconnect/schema refresh/admin cache clear."""
    try:
        _GLOBAL_QUERY_CACHE.clear()
    except Exception:
        logger.warning("failed_to_clear_global_query_cache", exc_info=True)
    try:
        if hasattr(RAGEngine, "clear_retrieval_cache"):
            RAGEngine.clear_retrieval_cache()
    except Exception:
        logger.warning("failed_to_clear_rag_retrieval_cache", exc_info=True)
    with _SCHEMA_RUNTIME_LOCK:
        _SCHEMA_PROFILE_CACHE.clear()
        _TABLE_EXISTS_CACHE.clear()
    with _GLOBAL_SCHEMA_CACHE_LOCK:
        _GLOBAL_SCHEMA_CACHE.clear()
    log_event(logger, logging.INFO, "runtime_cache_invalidated", reason=reason)


### STEP TIMER
class StepTimer:
    """Lightweight step-level timer for request lifecycle instrumentation."""

    STAGES = [
        "start",
        "intent_detection",
        "schema_loading",
        "stored_procedure_guidance",
        "cache_lookup",
        "rag_retrieval",
        "sql_generation",
        "sql_validation",
        "guardrails_applied",
        "db_execution",
        "results_formatting",
        "total",
    ]

    def __init__(self):
        self._marks: Dict[str, float] = {}

    def mark(self, stage: str) -> None:
        self._marks[stage] = time.perf_counter()

    def summary(self) -> Dict[str, float]:
        now = time.perf_counter()
        marks = dict(self._marks)
        if "start" not in marks:
            marks["start"] = now
        if "total" not in marks:
            marks["total"] = now

        out: Dict[str, float] = {"start": 0.0}
        prev_stage = "start"
        prev_ts = marks["start"]
        for stage in self.STAGES[1:]:
            ts = marks.get(stage)
            if ts is None:
                out[stage] = 0.0
                continue
            out[stage] = round((ts - prev_ts) * 1000.0, 2)
            prev_ts = ts
            prev_stage = stage

        # Ensure total reflects start -> total even if intermediates are missing.
        out["total"] = round((marks["total"] - marks["start"]) * 1000.0, 2)
        return out


def normalize_sql_dialect(dialect: Optional[str]) -> str:
    d = (dialect or "").strip().lower()
    if d in {"postgresql", "postgres", "psql"}:
        return "postgres"
    if d in {"sqlserver", "mssql", "sql server"}:
        return "sqlserver"
    return DEFAULT_SQL_DIALECT


def _dialect_label(dialect: str) -> str:
    d = normalize_sql_dialect(dialect)
    return "Postgres" if d == "postgres" else "SQL Server"


def _detect_sql_dialect_from_db(db: Optional[SQLDatabase]) -> str:
    try:
        name = (db._engine.dialect.name or "").lower() if db is not None else ""
    except Exception:
        name = ""
    if "postgres" in name:
        return "postgres"
    return "sqlserver"


def _get_ist_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Kolkata"))


def _get_ist_date_ranges(now_ist: Optional[datetime] = None) -> Dict[str, str]:
    now = now_ist or _get_ist_now()
    today = now.date()
    tomorrow = today + timedelta(days=1)
    yesterday = today - timedelta(days=1)
    this_week_start = today - timedelta(days=today.weekday())  # Monday
    next_week_start = this_week_start + timedelta(days=7)
    last_week_start = this_week_start - timedelta(days=7)
    month_start = today.replace(day=1)
    if month_start.month == 12:
        next_month_start = month_start.replace(year=month_start.year + 1, month=1, day=1)
    else:
        next_month_start = month_start.replace(month=month_start.month + 1, day=1)
    this_year_start = today.replace(month=1, day=1)
    next_year_start = this_year_start.replace(year=this_year_start.year + 1)
    return {
        "today_start": today.isoformat(),
        "today_end": tomorrow.isoformat(),
        "yesterday_start": yesterday.isoformat(),
        "yesterday_end": today.isoformat(),
        "this_week_start": this_week_start.isoformat(),
        "this_week_end": next_week_start.isoformat(),
        "last_week_start": last_week_start.isoformat(),
        "last_week_end": this_week_start.isoformat(),
        "this_month_start": month_start.isoformat(),
        "this_month_end": next_month_start.isoformat(),
        "this_year_start": this_year_start.isoformat(),
        "this_year_end": next_year_start.isoformat(),
        "now_ist": now.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _build_relative_date_reference(now_ist: Optional[datetime] = None) -> str:
    r = _get_ist_date_ranges(now_ist)
    return (
        f"now_ist={r['now_ist']}; "
        f"today=[{r['today_start']},{r['today_end']}); "
        f"yesterday=[{r['yesterday_start']},{r['yesterday_end']}); "
        f"this_week=[{r['this_week_start']},{r['this_week_end']}); "
        f"last_week=[{r['last_week_start']},{r['last_week_end']}); "
        f"this_month=[{r['this_month_start']},{r['this_month_end']}); "
        f"this_year=[{r['this_year_start']},{r['this_year_end']})"
    )


GREETING_PATTERNS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", "greetings"}
THANKS_PATTERNS = {"thank", "thanks", "thx", "appreciate", "grateful"}
HELP_PATTERNS = {"help", "what can you do", "how do you work", "capabilities", "what are you"}
FAREWELL_PATTERNS = {"bye", "goodbye", "see you", "later", "take care"}

INTENT_DETECTION_TEMPLATE = ChatPromptTemplate.from_template(
    """Classify intent: GREETING|THANKS|HELP|FAREWELL|OUT_OF_SCOPE|DATA_QUERY

Tables: {schema_summary}
Context: {conversation_context}
Message: {message}

Rules:
- DATA_QUERY: show/find/get/list/count data, filter, sort, follow-ups about data
- OUT_OF_SCOPE: predictions, ML, unrelated topics (weather, jokes)
- Default to DATA_QUERY if mentions: booking, supplier, country, agent, product

Output ONLY the label:"""
)

CONVERSATIONAL_TEMPLATE = ChatPromptTemplate.from_template(
    """Database assistant. Tables: {tables}
Message: {message} | Intent: {intent}

Reply briefly (1-2 sentences):
- GREETING: Welcome, offer to help query data
- THANKS: You're welcome
- HELP: Explain you query databases, give 1 example
- FAREWELL: Goodbye
- OUT_OF_SCOPE: Can't do that, suggest what you CAN do

Response:"""
)

NL_RESPONSE_TEMPLATE = ChatPromptTemplate.from_template(
    """You are a data analyst. Answer the user's question using the query results below.

Question: {question}

Query Results:
{results}

Instructions:
- Give a direct, specific answer using the ACTUAL numbers from the results.
- Mention the top 3-5 entries by name and their values if it's a ranking query.
- If it's a total/summary, state the exact figure.
- Keep it concise (2-4 sentences). No SQL, no technical jargon.
- Use the column names from the results to describe what the numbers represent.

Answer:"""
)

SQL_TEMPLATE = """You are an elite SQL performance engineer and Text-to-SQL agent.

GOAL:
Given the user's question + provided database schema context, produce ONE optimized SQL query for large datasets.

DIALECT:
- Active dialect: {dialect_label}
- Use correct syntax for the active dialect (TOP for SQL Server, LIMIT for Postgres).

OUTPUT FORMAT (strict):
1) SQL (single query) in one code block.
2) Optional "Notes:" with at most 3 bullets (assumptions only).
3) No other text.

SCHEMA:
{full_schema}

DOMAIN BUSINESS RULES:
{stored_procedure_guidance}

CONVERSATION CONTEXT:
{context}

ASIA/KOLKATA REFERENCE RANGES:
{relative_date_reference}

RETRIEVED TABLE HINTS (must use at least one of these tables/views):
{retrieved_tables_hint}

USER QUESTION:
{question}

MANDATORY RULES:
- Output ONLY a SELECT query (no DDL/DML, no EXEC).
- Use ONLY tables/columns that appear in SCHEMA or retrieved table hints; never invent names.
- If you cannot answer using the provided schema, return a minimal SELECT that explains the limitation in Notes.
- SARGable date filters: col >= start AND col < end (no YEAR(), CAST(date as date), etc).
- No SELECT *.
- Do not use NOLOCK unless enabled: {enable_nolock}

QUALITY CHECK (silent):
- SQL matches dialect
- GROUP BY correctness
- Date boundary correctness
- Minimal joins, filter early
- Avoid unnecessary DISTINCT

{few_shot_examples}

Return the final SQL."""

RETRY_PROMPT_TEMPLATE = """You are an elite SQL performance engineer and Text-to-SQL agent.

Fix and return ONE optimized query that matches the user's intent.

Active dialect: {dialect_label}
Session NOLOCK enabled: {enable_nolock}
Asia/Kolkata date reference: {relative_date_reference}

SCHEMA:
{full_schema}

DOMAIN BUSINESS RULES:
{stored_procedure_guidance}

QUESTION:
{question}

OUTPUT FORMAT (strict):
1) SQL (single query) in one code block.
2) Optional "Notes:" with at most 3 bullets (assumptions only).
3) No other text.

MANDATORY:
- SARGable date predicates using range filters only.
- For text filters: `=` first, `LIKE 'x%'` for prefix, `LIKE '%x%'` only if user asked contains.
- Avoid FORMAT() unless explicitly requested; return MonthStart date for monthly buckets.
- TOP for SQL Server, LIMIT for Postgres.
- No SELECT *.
- Do not use NOLOCK unless enabled.
- Use only schema columns/tables; do not invent.

Return the final SQL."""

RANKING_RESHAPE_PROMPT_TEMPLATE = """You must correct this ranking query while preserving intent.

Active dialect: {dialect_label}
Session NOLOCK enabled: {enable_nolock}
Asia/Kolkata date reference: {relative_date_reference}

QUESTION:
{question}

SCHEMA:
{full_schema}

STORED PROCEDURE LOGIC:
{stored_procedure_guidance}

CURRENT SQL:
{current_sql}

ISSUE:
{violation_reason}

RULES:
- Return ONE query only.
- Keep only user-requested metric(s).
- Ranking must aggregate at entity level first, then apply TOP/LIMIT + ORDER BY.
- Do not group by date unless the question explicitly asks date-wise ranking.
- SARGable date filters only; resolve relative dates using Asia/Kolkata.
- Do not use NOLOCK unless enabled.
- Use only schema-provided objects.

OUTPUT FORMAT (strict):
1) SQL (single query) in one code block.
2) Optional "Notes:" with at most 3 bullets (assumptions only).
3) No other text.

Return the final SQL."""

SQL_VALIDATOR_TEMPLATE = """You are a SQL validator. Output ONLY JSON.

Inputs:
- QUESTION
- SQL
- DIALECT
- SCHEMA
- REQUIRED_TABLE_HINTS (optional)

Return JSON:
{{
  "ok_to_execute": true/false,
  "failure_type": "syntax"|"schema"|"intent_mismatch"|"performance_risk"|"safety",
  "reasons": ["max 4 short reasons"],
  "fixed_sql": "single SELECT" | null
}}

Rules:
- Must be exactly one SELECT.
- No tables/columns outside SCHEMA.
- Must match QUESTION intent (metric + grouping + date window).
- Flag performance_risk if: missing obvious filter on large table, SELECT *, unbounded result.
- If you can confidently fix it, provide fixed_sql; else null.

DIALECT: {dialect}
SCHEMA: {schema_compact}
REQUIRED_TABLE_HINTS: {top_tables}
QUESTION: {question}
SQL: {sql}
"""


# --- Raw ODBC connection helper ---

def _raw_odbc_connect(host: str, port, username: str, password: str, database: str, timeout: int = 15):
    """Open a raw pyodbc connection using ODBC Driver 18 for SQL Server."""
    import pyodbc
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={host},{port};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"LoginTimeout=5;"
        f"QueryTimeout={timeout};"
        f"TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str, timeout=5)


def _raw_conn_from_engine(db: SQLDatabase, timeout: int = 15):
    """Extract connection params from a SQLAlchemy engine and open a raw pyodbc connection."""
    from urllib.parse import parse_qs, unquote
    url = db._engine.url
    # For pyodbc engines the params are in the query string
    odbc_connect = url.query.get("odbc_connect")
    if odbc_connect:
        import pyodbc
        return pyodbc.connect(unquote(odbc_connect), timeout=5)
    # Fallback: extract individual params
    return _raw_odbc_connect(
        host=url.host or "", port=url.port or 1433,
        username=url.username or "", password=url.password or "",
        database=url.database or "", timeout=timeout,
    )


# --- Schema loading ---

def load_schema_from_sqldb(rag_engine: RAGEngine, db: SQLDatabase):
    """Load schema into RAG engine and return schema text for LLM prompts.

    Uses a single bulk INFORMATION_SCHEMA query instead of per-table
    inspector calls. On remote databases this is ~100x faster
    (0.3s vs 48s for 9 tables).
    Returns the schema text string.
    """
    from collections import defaultdict

    engine = db._engine
    table_names = list(db.get_usable_table_names())

    if not table_names:
        return ""

    # Single bulk query for all columns across all tables
    table_cols_map = defaultdict(list)
    try:
        with closing(_raw_conn_from_engine(db)) as conn:
            with closing(conn.cursor()) as cursor:
                placeholders = ",".join([f"'{t}'" for t in table_names])
                cursor.execute(
                    f"SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE "
                    f"FROM INFORMATION_SCHEMA.COLUMNS "
                    f"WHERE TABLE_NAME IN ({placeholders}) "
                    f"ORDER BY TABLE_NAME, ORDINAL_POSITION"
                )
                for row in cursor.fetchall():
                    table_cols_map[row[0]].append({
                        "name": row[1],
                        "type": row[2].upper(),
                        "nullable": row[3] == "YES",
                    })
    except Exception:
        # Fallback to per-table inspector if bulk query fails
        inspector = sqlalchemy.inspect(engine)
        for table_name in table_names:
            try:
                for col in inspector.get_columns(table_name):
                    table_cols_map[table_name].append({
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                    })
            except Exception:
                continue

    schema_parts = []
    for table_name in table_names:
        cols_raw = table_cols_map.get(table_name, [])
        if not cols_raw:
            continue

        columns = []
        col_lines = []
        for col in cols_raw:
            columns.append({
                "name": col["name"],
                "type": col["type"],
                "description": (
                    f"Column {col['name']} of type {col['type']}"
                    + (" (nullable)" if col["nullable"] else " (NOT NULL)")
                ),
            })
            nullable_str = "NULL" if col["nullable"] else "NOT NULL"
            col_lines.append(f"  {col['name']} {col['type']} {nullable_str}")

        description = f"Table '{table_name}' with {len(columns)} columns"
        rag_engine.add_table(name=table_name, description=description, columns=columns)

        schema_parts.append(
            f"CREATE TABLE {table_name} (\n" + ",\n".join(col_lines) + "\n)"
        )

    return "\n\n".join(schema_parts)


# --- Stored procedure knowledge ---

STORED_PROCEDURE_FILE = Path(__file__).parent.parent.parent / "stored_procedure.txt"
DEFAULT_STATUS_EXCLUSIONS = ("Cancelled", "Not Confirmed", "On Request")

# CHANGED: mtime-aware cache for stored procedure raw text + derived guidance.
# Auto-refreshes when STORED_PROCEDURE_FILE changes on disk.
_STORED_PROCEDURE_CACHE: Dict[str, Any] = {
    "path": str(STORED_PROCEDURE_FILE),
    "mtime_ns": None,
    "raw_text": "",
    "raw_hash": None,
    "guidance": None,
    "domain_digest": None,
    "domain_digest_hash": None,
}
_STORED_PROCEDURE_CACHE_LOCK = threading.RLock()


def _compute_text_hash(text: str) -> int:
    return hash(text or "")


def _read_stored_procedure_file(path: Path = STORED_PROCEDURE_FILE) -> str:
    cache = _STORED_PROCEDURE_CACHE
    with _STORED_PROCEDURE_CACHE_LOCK:
        cache["path"] = str(path)
    try:
        stat = path.stat()
        mtime_ns = int(stat.st_mtime_ns)
    except Exception:
        # File missing/unreadable -> keep runtime safe.
        with _STORED_PROCEDURE_CACHE_LOCK:
            cache.update({
                "mtime_ns": None,
                "raw_text": "",
                "raw_hash": _compute_text_hash(""),
                "guidance": None,
            })
        return ""

    with _STORED_PROCEDURE_CACHE_LOCK:
        if cache.get("mtime_ns") == mtime_ns and cache.get("raw_text") is not None:
            return cache.get("raw_text") or ""

    try:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raw_text = ""

    with _STORED_PROCEDURE_CACHE_LOCK:
        cache.update({
            "mtime_ns": mtime_ns,
            "raw_text": raw_text,
            "raw_hash": _compute_text_hash(raw_text),
            "guidance": None,  # Invalidate derived guidance when file changes.
            "domain_digest": None,
            "domain_digest_hash": None,
        })
    return raw_text


def _parse_stored_procedure_sections(raw_text: str) -> List[Dict[str, str]]:
    """Parse stored_procedure.txt into numbered sections. (Unchanged from original.)"""
    if not raw_text:
        return []
    pattern = re.compile(r"(?m)^\s*(\d+)\.\s*([^\n:]+):")
    matches = list(pattern.finditer(raw_text))
    sections: List[Dict[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        title = " ".join(match.group(2).strip().split())
        body = raw_text[start:end].strip()
        if body:
            sections.append({"index": match.group(1), "title": title, "sql": body})
    return sections

def _extract_domain_rules(sections: List[Dict[str, str]]) -> Dict:
    """Analyze all stored procedure SQL and extract structured business rules."""
    all_sql = "\n".join(s["sql"] for s in sections)
    rules = {
        "metrics": {},
        "status_exclusions": [],
        "chain_case_sql": None,
    }

    # Metric formulas
    if re.search(r"sum\s*\(\s*agentbuyingprice\s*\)", all_sql, re.IGNORECASE):
        rules["metrics"]["revenue"] = ("SUM(AgentBuyingPrice)", "total_sales")
    if re.search(r"sum\s*\(\s*agentbuyingprice\s*-\s*companybuyingprice\s*\)", all_sql, re.IGNORECASE):
        rules["metrics"]["profit"] = ("SUM(AgentBuyingPrice - CompanyBuyingPrice)", "total_profit")
    if re.search(r"sum\s*\(\s*companybuyingprice\s*\)", all_sql, re.IGNORECASE):
        rules["metrics"]["cost"] = ("SUM(CompanyBuyingPrice)", "total_cost")
    else:
        rules["metrics"]["cost"] = ("SUM(CompanyBuyingPrice)", "total_cost")
    booking_match = re.search(r"count\s*\(\s*distinct\s+(\w+)\s*\)", all_sql, re.IGNORECASE)
    if booking_match:
        rules["metrics"]["bookings"] = (f"COUNT(DISTINCT {booking_match.group(1)})", "total_booking")

    # Status exclusion values
    status_match = re.search(r"bookingstatus\s+not\s+in\s*\(([^)]+)\)", all_sql, re.IGNORECASE)
    if status_match:
        rules["status_exclusions"] = [v.strip().strip("'\"") for v in status_match.group(1).split(",")]

    # Hotel chain CASE WHEN
    chain_match = re.search(
        r"(case\s+when\s+AM\.chain.*?end)\s+as\s+chain",
        all_sql, re.IGNORECASE | re.DOTALL,
    )
    if chain_match:
        rules["chain_case_sql"] = chain_match.group(1).strip()

    return rules

def _extract_dimension_info(section: Dict[str, str]) -> Dict:
    """Extract base table, joins, and dimension columns from a single section."""
    sql = section["sql"]
    info = {"title": section["title"], "base_table": None, "joins": [], "dimension_cols": []}

    from_match = re.search(r"\bfrom\s+(?:\[?dbo\]?\.)?(\[?\w+\]?)\s+", sql, re.IGNORECASE)
    if from_match:
        info["base_table"] = from_match.group(1).replace("[", "").replace("]", "")

    join_pattern = re.compile(
        r"(?:left\s+)?join\s+(?:\[?[^\]]*\]?\.)*\[?(\w+)\]?"
        r"(?:\s+\w+)?(?:\s+with\s*\([^)]*\))?\s+on\s+([^\n]+)",
        re.IGNORECASE,
    )
    for m in join_pattern.finditer(sql):
        info["joins"].append({
            "table": m.group(1),
            "condition": m.group(2).strip().split("\n")[0].strip(),
        })

    group_match = re.search(r"group\s+by\s+(.+?)(?:\)|$)", sql, re.IGNORECASE | re.DOTALL)
    if group_match:
        for part in group_match.group(1).split(","):
            part = part.strip()
            if "cast(" in part.lower() or not part:
                continue
            col = part.split(".")[-1].replace("[", "").replace("]", "").strip()
            if col:
                info["dimension_cols"].append(col)

    return info

def build_domain_digest(raw_text: str) -> str:
    """Create compact stored-procedure guidance with SARGable date rules."""
    cache = _STORED_PROCEDURE_CACHE
    raw_hash = _compute_text_hash(raw_text)
    with _STORED_PROCEDURE_CACHE_LOCK:
        if cache.get("domain_digest") is not None and cache.get("domain_digest_hash") == raw_hash:
            return cache["domain_digest"]

    sql_text = raw_text or ""
    lower_text = sql_text.lower()

    status_values = set(DEFAULT_STATUS_EXCLUSIONS)
    for m in re.finditer(r"(?is)\bbookingstatus\s+not\s+in\s*\(([^)]+)\)", sql_text):
        for item in m.group(1).split(","):
            val = item.strip().strip("'\"")
            if val:
                status_values.add(val)

    metric_lines = [
        "• Revenue: SUM(AgentBuyingPrice) AS total_sales",
        "• Cost: SUM(CompanyBuyingPrice) AS total_cost",
        "• Profit: SUM(AgentBuyingPrice - CompanyBuyingPrice) AS total_profit",
        "• Bookings: COUNT(DISTINCT PNRNo) AS total_booking",
    ]
    if "sum(total_sales)" in lower_text:
        metric_lines.append("• Revenue (view): SUM(total_sales) AS total_sales")
    if "sum(total_profit)" in lower_text:
        metric_lines.append("• Profit (view): SUM(total_profit) AS total_profit")
    if "sum(total_booking)" in lower_text:
        metric_lines.append("• Bookings (view): SUM(total_booking) AS total_booking")

    digest_lines: List[str] = []
    digest_lines.append("=== DOMAIN DIGEST (SARGABLE) ===")
    digest_lines.append("STATUS EXCLUSIONS:")
    digest_lines.append("• BookingStatus NOT IN (" + ", ".join(f"'{v}'" for v in sorted(status_values)) + ")")
    digest_lines.append("")
    digest_lines.append("METRIC FORMULAS:")
    digest_lines.extend(metric_lines)
    digest_lines.append("")
    digest_lines.append("DATE COLUMNS:")
    digest_lines.append("• CreatedDate -> booking creation date (default)")
    digest_lines.append("• CheckInDate -> travel/check-in date")
    digest_lines.append("• CheckOutDate -> travel/check-out date")
    digest_lines.append("• *_level_view columns booking_date/checkin_date/checkout_date follow same semantics")
    digest_lines.append("")
    digest_lines.append("SARGABLE DATE FILTERS (MANDATORY):")
    digest_lines.append("• NEVER use CAST(date_col AS DATE), YEAR(date_col), MONTH(date_col) in WHERE")
    digest_lines.append("• Always use: date_col >= <start_date> AND date_col < <end_date>")
    digest_lines.append("• Current month:")
    digest_lines.append("  date_col >= DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)")
    digest_lines.append("  AND date_col < DATEADD(MONTH, 1, DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1))")
    digest_lines.append("• Current year:")
    digest_lines.append("  date_col >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1)")
    digest_lines.append("  AND date_col < DATEFROMPARTS(YEAR(GETDATE()) + 1, 1, 1)")
    digest_lines.append("")
    digest_lines.append("CANONICAL JOINS:")
    digest_lines.append("• BookingData.AgentId = AgentMaster_V1.AgentId")
    digest_lines.append("• BookingData.AgentId = AgentTypeMapping.AgentId")
    digest_lines.append("• BookingData.SupplierId = suppliermaster_Report.EmployeeId")
    digest_lines.append("• BookingData.ProductCountryid = Master_Country.CountryID")
    digest_lines.append("• BookingData.ProductCityId = Master_City.CityId")
    digest_lines.append("• BookingData.ProductId = Hotelchain.HotelId")
    digest_lines.append("• Chain normalization: CASE WHEN Chain IS NULL OR Chain IN ('', 'Others') THEN HotelName ELSE Chain END")
    digest_lines.append("")
    digest_lines.append("PERFORMANCE RULES:")
    digest_lines.append("• Avoid SELECT * and DISTINCT unless required.")
    digest_lines.append("• For monthly trend use MonthStart = DATEFROMPARTS(YEAR(date_col), MONTH(date_col), 1)")
    digest_lines.append("• For top-N aggregate first, then apply TOP/LIMIT and ORDER BY.")

    digest = "\n".join(digest_lines[:120])
    with _STORED_PROCEDURE_CACHE_LOCK:
        cache["domain_digest"] = digest
        cache["domain_digest_hash"] = raw_hash
        cache["guidance"] = digest
        cache["raw_text"] = sql_text
    return digest


def _add_stored_procedure_knowledge_to_rag(rag_engine: Optional[RAGEngine], raw_text: str):
    if rag_engine is None:
        return
    sections = _parse_stored_procedure_sections(raw_text)
    if not sections:
        return

    # ── Core business rules ──
    rag_engine.add_rule(
        "stored_procedure_business_rules",
        "Revenue=SUM(AgentBuyingPrice), Cost=SUM(CompanyBuyingPrice), "
        "Profit=SUM(AgentBuyingPrice-CompanyBuyingPrice), "
        "Bookings=COUNT(DISTINCT PNRNo), exclude BookingStatus in (Cancelled, Not Confirmed, On Request) when available.",
        "BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')",
    )

    # ── Extended metric rules from Excel dashboard formulas ──
    rag_engine.add_rule(
        "extended_metric_formulas",
        "Room Nights: SUM(DATEDIFF(DAY,CheckInDate,CheckOutDate)). "
        "Avg Booking Window / Lead Time: AVG(DATEDIFF(DAY,CreatedDate,CheckInDate)). "
        "Avg Booking Value: SUM(AgentBuyingPrice)/NULLIF(COUNT(DISTINCT PNRNo),0). "
        "Last Minute Bookings (checkin within 1 day of booking): COUNT(DISTINCT CASE WHEN DATEDIFF(DAY,CreatedDate,CheckInDate)<=1 THEN PNRNo END). "
        "Cancelled Bookings: COUNT(DISTINCT CASE WHEN BookingStatus='Cancelled' THEN PNRNo END) — do NOT apply status exclusion filter. "
        "Non-Refundable Bookings: COUNT(DISTINCT CASE WHEN DATEDIFF(DAY,CreatedDate,CancellationDeadLine)<=0 THEN PNRNo END). "
        "Pax Adults: SUM(NoofAdult). Pax Children: SUM(NoofChild).",
        None,
    )

    # ── Column annotation rule ──
    rag_engine.add_rule(
        "bookingdata_column_usage",
        "USED columns in BookingData: PNRNo, AgentBuyingPrice, CompanyBuyingPrice, BookingStatus, "
        "CreatedDate, CheckInDate, CheckOutDate, CancellationDeadLine, ProductName, ProductId, "
        "ServiceName, AgentId, SupplierId, ProductCountryid, ProductCityId, ClientNatinality (note spelling), "
        "NoofAdult, NoofChild. "
        "NOT USED (never reference): AgentSellingPrice, SubAgentSellingPrice, PaymentType, IsPackage, "
        "CurrencyId, RateOfexchange, SellingRateOfexchange, SupplierRateOfexchange, SupplierCurrencyId, "
        "RoomTypeName, Provider, OfferCode, OfferDescription, LoyaltyPoints, OTHPromoCode, "
        "IsXMLSupplierBooking, PackageId, PackageName, EntryDate, BranchId, SubUserId, CreditcardCharges, "
        "AgentReferenceNo, IsSameCurrency.",
        None,
    )

    # ── Join key rules ──
    rag_engine.add_rule(
        "table_join_keys",
        "AgentMaster_V1: BD.AgentId = A.AgentId — gives AgentCode, AgentName, AgentCountry, AgentCity. "
        "AgentTypeMapping: BD.AgentId = AM.AgentId — gives AgentType (API/B2B/We Groups); "
        "agentcode='WEGR' maps to 'We Groups'. "
        "suppliermaster_Report: BD.SupplierId = D.EmployeeId — gives suppliername. "
        "Master_Country: BD.ProductCountryid = MCO.CountryID — gives Country. "
        "Master_City: BD.ProductCityId = MCI.CityId — gives City. "
        "Hotelchain: BD.ProductId = HC.HotelId — gives HotelName, Country, City, Star, Chain.",
        None,
    )

    # ── RAG examples for stored procedure sections ──
    for sec in sections:
        title = sec["title"]
        sql_text = sec["sql"].strip()
        if not sql_text:
            continue
        question = f"show {title} performance"
        variations = [title, f"{title} summary", f"{title} metrics"]
        # Richer variations for agent type — kept tightly scoped to
        # avoid bleeding into hotel/growth/YoY queries via semantic similarity
        if "agent type" in title.lower():
            variations += [
                "split of bookings by agent type",
                "bookings by agent type",
                "agent type breakdown",
                "B2B API We Groups split",
                "agent category distribution",
            ]
        rag_engine.add_example(question=question, sql=sql_text, variations=variations)

    # ── Synthetic examples for YoY / growth queries ──
    _YOY_HOTEL_SQL = (
        "with main as (select distinct productname, "
        "SUM(CASE WHEN cast(CreatedDate as date) >= DATEFROMPARTS(YEAR(GETDATE()),1,1) "
        "AND cast(CreatedDate as date) <= GETDATE() "
        "AND bookingstatus not in ('Cancelled','Not Confirmed','On Request') THEN agentbuyingprice END) AS YTD_Sales, "
        "SUM(CASE WHEN cast(CreatedDate as date) >= DATEFROMPARTS(YEAR(GETDATE())-1,1,1) "
        "AND cast(CreatedDate as date) <= DATEADD(YEAR,-1,GETDATE()) "
        "AND bookingstatus not in ('Cancelled','Not Confirmed','On Request') THEN agentbuyingprice END) AS PYTD_Sales "
        "from dbo.BookingData BD with (nolock) "
        "where cast(CreatedDate as date) >= DATEFROMPARTS(YEAR(GETDATE())-1,1,1) "
        "group by productname) "
        "SELECT TOP 15 productname, YTD_Sales, PYTD_Sales, "
        "(YTD_Sales - PYTD_Sales) AS Growth_Amount, "
        "CASE WHEN YTD_Sales = 0 THEN NULL "
        "ELSE (YTD_Sales - PYTD_Sales) * 100.0 / NULLIF(PYTD_Sales, 0) END AS Growth_Percentage "
        "FROM main WHERE YTD_Sales IS NOT NULL ORDER BY Growth_Percentage DESC"
    )
    rag_engine.add_example(
        question="show top 15 hotels showing growth in sales compared to last year and their growth percentage",
        sql=_YOY_HOTEL_SQL,
        variations=[
            "hotel sales growth",
            "hotels with growth compared to last year",
            "top hotels by growth percentage",
            "hotel YTD vs PYTD",
            "hotel year over year growth",
            "which hotels grew the most",
            "hotel sales increase vs last year",
            "growth percentage by hotel",
            "hotel growth ranking",
        ],
    )
    rag_engine.add_example(
        question="show top 15 suppliers showing growth in sales compared to last year",
        sql=(
            "with AM as (select distinct employeeid, suppliername from [MIS_Report_Data].[dbo].[suppliermaster_Report] with (nolock)), "
            "main as (select distinct AM.suppliername, "
            "SUM(CASE WHEN cast(BD.CreatedDate as date) >= DATEFROMPARTS(YEAR(GETDATE()),1,1) "
            "AND cast(BD.CreatedDate as date) <= GETDATE() "
            "AND BD.bookingstatus not in ('Cancelled','Not Confirmed','On Request') THEN BD.agentbuyingprice END) AS YTD_Sales, "
            "SUM(CASE WHEN cast(BD.CreatedDate as date) >= DATEFROMPARTS(YEAR(GETDATE())-1,1,1) "
            "AND cast(BD.CreatedDate as date) <= DATEADD(YEAR,-1,GETDATE()) "
            "AND BD.bookingstatus not in ('Cancelled','Not Confirmed','On Request') THEN BD.agentbuyingprice END) AS PYTD_Sales "
            "from dbo.BookingData BD with (nolock) "
            "left join AM on AM.employeeid = BD.supplierid "
            "where cast(BD.CreatedDate as date) >= DATEFROMPARTS(YEAR(GETDATE())-1,1,1) "
            "group by AM.suppliername) "
            "SELECT TOP 15 suppliername, YTD_Sales, PYTD_Sales, "
            "(YTD_Sales - PYTD_Sales) AS Growth_Amount, "
            "CASE WHEN YTD_Sales = 0 THEN NULL "
            "ELSE (YTD_Sales - PYTD_Sales) * 100.0 / NULLIF(PYTD_Sales, 0) END AS Growth_Percentage "
            "FROM main WHERE YTD_Sales IS NOT NULL ORDER BY Growth_Percentage DESC"
        ),
        variations=[
            "supplier sales growth",
            "suppliers with growth compared to last year",
            "top suppliers by growth percentage",
            "supplier YTD vs PYTD",
            "supplier year over year growth",
        ],
    )

    # ── Synthetic examples for extended metrics ──
    rag_engine.add_example(
        question="total room nights this month",
        sql=(
            "SELECT SUM(DATEDIFF(DAY, CheckInDate, CheckOutDate)) AS total_room_nights "
            "FROM dbo.BookingData WITH (NOLOCK) "
            "WHERE YEAR(CreatedDate) = YEAR(GETDATE()) AND MONTH(CreatedDate) = MONTH(GETDATE()) "
            "AND BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')"
        ),
        variations=["room nights", "total nights", "night count", "how many room nights"],
    )
    rag_engine.add_example(
        question="average booking window this year",
        sql=(
            "SELECT AVG(DATEDIFF(DAY, CreatedDate, CheckInDate)) AS avg_booking_window "
            "FROM dbo.BookingData WITH (NOLOCK) "
            "WHERE YEAR(CreatedDate) = YEAR(GETDATE()) "
            "AND BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')"
        ),
        variations=["booking window", "lead time", "average lead time", "booking lead time", "avg booking window"],
    )
    rag_engine.add_example(
        question="average booking value by agent",
        sql=(
            "SELECT A.AgentName, "
            "SUM(BD.AgentBuyingPrice) / NULLIF(COUNT(DISTINCT BD.PNRNo), 0) AS avg_booking_value "
            "FROM dbo.BookingData BD WITH (NOLOCK) "
            "JOIN AgentMaster_V1 A ON A.AgentId = BD.AgentId "
            "WHERE YEAR(BD.CreatedDate) = YEAR(GETDATE()) "
            "AND BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') "
            "GROUP BY A.AgentName ORDER BY avg_booking_value DESC"
        ),
        variations=["average booking value", "avg value per booking", "booking value by agent"],
    )
    rag_engine.add_example(
        question="last minute bookings this month",
        sql=(
            "SELECT COUNT(DISTINCT CASE WHEN DATEDIFF(DAY, CreatedDate, CheckInDate) <= 1 THEN PNRNo END) AS last_minute_bookings "
            "FROM dbo.BookingData WITH (NOLOCK) "
            "WHERE YEAR(CreatedDate) = YEAR(GETDATE()) AND MONTH(CreatedDate) = MONTH(GETDATE())"
        ),
        variations=["last minute bookings", "last minute", "same day bookings", "bookings within 1 day"],
    )
    rag_engine.add_example(
        question="total cancelled bookings this year",
        sql=(
            "SELECT COUNT(DISTINCT CASE WHEN BookingStatus = 'Cancelled' THEN PNRNo END) AS cancelled_bookings "
            "FROM dbo.BookingData WITH (NOLOCK) "
            "WHERE YEAR(CreatedDate) = YEAR(GETDATE())"
        ),
        variations=["cancelled bookings", "cancellations", "how many cancelled", "number of cancellations"],
    )
    rag_engine.add_example(
        question="total non refundable bookings",
        sql=(
            "SELECT COUNT(DISTINCT CASE WHEN DATEDIFF(DAY, CreatedDate, CancellationDeadLine) <= 0 THEN PNRNo END) AS non_refundable_bookings "
            "FROM dbo.BookingData WITH (NOLOCK) "
            "WHERE YEAR(CreatedDate) = YEAR(GETDATE()) "
            "AND BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')"
        ),
        variations=["non refundable", "non-refundable bookings", "non refundable count"],
    )


def _extract_table_hints_from_stored_procedure_text(raw_text: str) -> List[str]:
    if not raw_text:
        return []

    hints = set()
    patterns = [
        re.compile(r"(?i)\bfrom\s+([^\s,;]+)"),
        re.compile(r"(?i)\bjoin\s+([^\s,;]+)"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(raw_text):
            token = match.group(1).strip()
            token = token.replace("[", "").replace("]", "")
            token = token.split(".")[-1]
            token = token.strip()
            if token:
                hints.add(token)
    return sorted(hints)


# --- Runtime schema helpers ---

def _normalize_identifier(name: str) -> str:
    token = str(name or "").strip().strip(";")
    token = token.replace("[", "").replace("]", "").replace("`", "").replace('"', "")
    parts = [p for p in token.split(".") if p]
    return parts[-1].lower() if parts else token.lower()


def _extract_from_table_name(sql: str) -> Optional[str]:
    if not sql:
        return None
    match = re.search(r"\bFROM\s+([^\s,;]+)", sql, re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).strip()
    if token.startswith("("):
        return None
    return _normalize_identifier(token)


def _is_text_type(type_name: str) -> bool:
    t = (type_name or "").lower()
    return any(k in t for k in ("char", "text", "string", "nchar", "nvarchar", "varchar"))


def _get_schema_profile(db: SQLDatabase) -> Dict[str, Any]:
    if db is None or getattr(db, "_engine", None) is None:
        return {"tables": [], "lookup": {}}

    cache_key = id(db._engine)
    with _SCHEMA_RUNTIME_LOCK:
        cached = _SCHEMA_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        usable_tables = list(db.get_usable_table_names())
    except Exception:
        profile = {"tables": [], "lookup": {}}
        with _SCHEMA_RUNTIME_LOCK:
            _SCHEMA_PROFILE_CACHE[cache_key] = profile
        return profile

    # Use bulk INFORMATION_SCHEMA query instead of per-table inspector calls.
    # On remote DBs this is ~100x faster (0.3s vs 30s+).
    from collections import defaultdict
    table_cols_raw: Dict[str, list] = defaultdict(list)

    try:
        with closing(_raw_conn_from_engine(db)) as conn:
            with closing(conn.cursor()) as cursor:
                placeholders = ",".join([f"'{t}'" for t in usable_tables])
                cursor.execute(
                    f"SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
                    f"FROM INFORMATION_SCHEMA.COLUMNS "
                    f"WHERE TABLE_NAME IN ({placeholders}) "
                    f"ORDER BY TABLE_NAME, ORDINAL_POSITION"
                )
                for row in cursor.fetchall():
                    table_cols_raw[row[0]].append({"name": row[1], "type": row[2]})
    except Exception:
        # Fallback to per-table inspector
        inspector = sqlalchemy.inspect(db._engine)
        for table_name in usable_tables:
            try:
                for col in inspector.get_columns(table_name):
                    table_cols_raw[table_name].append({
                        "name": col["name"],
                        "type": str(col.get("type", "")),
                    })
            except Exception:
                continue

    tables = []
    lookup: Dict[str, Dict[str, Any]] = {}

    for table_name in usable_tables:
        cols_raw = table_cols_raw.get(table_name, [])
        if not cols_raw:
            continue

        columns = []
        columns_map = {}
        column_types = {}
        text_columns = []

        for col in cols_raw:
            col_name = col["name"]
            columns.append(col_name)
            lower = col_name.lower()
            columns_map[lower] = col_name
            type_name = col["type"]
            column_types[lower] = type_name
            if _is_text_type(type_name):
                text_columns.append(col_name)

        entry = {
            "table": table_name,
            "table_lower": table_name.lower(),
            "table_normalized": _normalize_identifier(table_name),
            "columns": columns,
            "columns_map": columns_map,
            "column_types": column_types,
            "text_columns": text_columns,
        }
        tables.append(entry)
        lookup[entry["table_lower"]] = entry
        lookup.setdefault(entry["table_normalized"], entry)

    profile = {"tables": tables, "lookup": lookup}
    with _SCHEMA_RUNTIME_LOCK:
        _SCHEMA_PROFILE_CACHE[cache_key] = profile
    return profile


def _get_table_entry_from_sql(sql: str, db: SQLDatabase) -> Optional[Dict[str, Any]]:
    table_normalized = _extract_from_table_name(sql)
    if not table_normalized:
        return None
    profile = _get_schema_profile(db)
    return profile["lookup"].get(table_normalized)


def _table_exists_fast(db: SQLDatabase, table_name: str) -> bool:
    if db is None or getattr(db, "_engine", None) is None or not table_name:
        return False
    key = (id(db._engine), table_name.lower())
    with _SCHEMA_RUNTIME_LOCK:
        cached = _TABLE_EXISTS_CACHE.get(key)
    if cached is not None:
        return cached

    profile = _get_schema_profile(db)
    for entry in profile.get("tables", []):
        if entry.get("table_lower") == table_name.lower() or entry.get("table_normalized") == table_name.lower():
            with _SCHEMA_RUNTIME_LOCK:
                _TABLE_EXISTS_CACHE[key] = True
            return True

    exists = False
    try:
        with closing(_raw_conn_from_engine(db)) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute(
                    "SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?",
                    (table_name,),
                )
                exists = cursor.fetchone() is not None
    except Exception:
        exists = False

    with _SCHEMA_RUNTIME_LOCK:
        _TABLE_EXISTS_CACHE[key] = exists
    return exists


def _pick_existing_column(table_entry: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    cols = table_entry.get("columns_map", {})
    for cand in candidates:
        actual = cols.get(cand.lower())
        if actual:
            return actual
    return None


def _detect_date_column(table_entry: Dict[str, Any]) -> Optional[str]:
    return _pick_existing_column(
        table_entry,
        [
            "booking_date",
            "bookingdate",
            "createddate",
            "created_date",
            "checkin_date",
            "checkindate",
            "checkout_date",
            "checkoutdate",
            "date",
        ],
    )


def _metric_exprs_for_table(table_entry: Dict[str, Any]) -> Dict[str, Any]:
    revenue_col = _pick_existing_column(
        table_entry,
        ["total_sales", "agentbuyingprice", "totalamount", "revenue", "saleamount", "amount"],
    )
    profit_col = _pick_existing_column(table_entry, ["total_profit", "profit"])
    cost_col = _pick_existing_column(
        table_entry,
        ["companybuyingprice", "buyingprice", "costprice", "total_cost", "costamount"],
    )
    bookings_sum_col = _pick_existing_column(
        table_entry,
        ["total_booking", "totalbookings", "totalbooking", "bookingcount", "bookings"],
    )
    booking_id_col = _pick_existing_column(
        table_entry,
        ["pnrno", "bookingid", "booking_id", "bookingno", "reservationid", "voucherid", "id"],
    )

    if revenue_col:
        revenue_expr = f"SUM(COALESCE({_quote_sql_identifier(revenue_col)}, 0))"
    else:
        revenue_expr = "0"

    if profit_col:
        profit_expr = f"SUM(COALESCE({_quote_sql_identifier(profit_col)}, 0))"
    elif revenue_col and cost_col:
        profit_expr = (
            f"SUM(COALESCE({_quote_sql_identifier(revenue_col)}, 0) - "
            f"COALESCE({_quote_sql_identifier(cost_col)}, 0))"
        )
    else:
        profit_expr = "0"

    if cost_col:
        cost_expr = f"SUM(COALESCE({_quote_sql_identifier(cost_col)}, 0))"
    else:
        cost_expr = "0"

    if bookings_sum_col:
        bookings_expr = f"SUM(COALESCE({_quote_sql_identifier(bookings_sum_col)}, 0))"
    elif booking_id_col:
        bookings_expr = f"COUNT(DISTINCT {_quote_sql_identifier(booking_id_col)})"
    else:
        bookings_expr = "COUNT(*)"

    return {
        "revenue_expr": revenue_expr,
        "profit_expr": profit_expr,
        "cost_expr": cost_expr,
        "bookings_expr": bookings_expr,
        "has_revenue": revenue_expr != "0",
        "has_profit": profit_expr != "0",
        "has_cost": cost_expr != "0",
        "has_group_metric": bookings_expr != "0" or revenue_expr != "0" or cost_expr != "0",
    }


def _detect_status_column(table_entry: Dict[str, Any]) -> Optional[str]:
    return _pick_existing_column(table_entry, ["bookingstatus", "status", "booking_status"])


def _status_filter_expr_for_table(table_entry: Dict[str, Any]) -> Optional[str]:
    status_col = _detect_status_column(table_entry)
    if not status_col:
        return None
    status_col_quoted = _quote_sql_identifier(status_col)
    values = ", ".join([f"'{v}'" for v in DEFAULT_STATUS_EXCLUSIONS])
    return f"{status_col_quoted} NOT IN ({values})"


def _infer_text_columns(table_entry: Dict[str, Any]) -> List[str]:
    cols = table_entry.get("columns", [])
    names = []
    seen = set()
    keyword_hints = ("name", "city", "country", "chain", "nationality", "type", "product", "hotel", "agent", "supplier")

    for col in table_entry.get("text_columns", []):
        low = col.lower()
        if low not in seen:
            names.append(col)
            seen.add(low)

    for col in cols:
        low = col.lower()
        if low in seen:
            continue
        if any(k in low for k in keyword_hints):
            names.append(col)
            seen.add(low)

    return names


# --- Embedding helpers ---

def compute_embedding(text: str, embedder) -> Optional[np.ndarray]:
    if embedder is None:
        return None
    return np.array(embedder.embed_query(text), dtype=np.float32)


# --- Follow-up detection ---

def is_followup_question(question: str) -> bool:
    words = set(question.lower().split())
    return len(words) < 6 and bool(words & FOLLOW_UP_WORDS)


def is_sort_followup(question: str) -> bool:
    q_lower = question.lower()
    # NOTE:
    # "by" is intentionally excluded; it appears in many fresh queries like
    # "top 10 bookings by revenue" and should not force follow-up sort mode.
    sort_words = {"sort", "order", "arrange", "rank", "ascending", "descending", "asc", "desc", "differently", "reverse"}
    words = set(q_lower.split())
    has_sort = bool(words & sort_words)
    has_followup = bool(words & FOLLOW_UP_WORDS)
    is_short_sort = has_sort and len(words) <= 6
    if _parse_topn_request(question):
        return False
    return has_sort and (has_followup or is_short_sort)


def _question_mentions_result_value(question_lower: str, previous_result_df) -> bool:
    """Check if the question mentions any value from the previous result.
    Checks both: full value in question, AND question words in values."""
    if previous_result_df is None or len(previous_result_df) == 0:
        return False
    # Extract meaningful words from question (skip short/common words)
    skip_words = {"what", "about", "how", "show", "the", "for", "and", "but", "with", "from", "only", "just", "that", "this", "those", "them", "it", "me", "can", "you", "is", "are", "was", "were", "do", "does", "did"}
    question_words = [w.strip("?.,!") for w in question_lower.split() if len(w.strip("?.,!")) > 2 and w.strip("?.,!") not in skip_words]

    # CHANGED: cap scan effort to keep follow-up detection fast on hot path.
    scanned_cols = 0
    for col in previous_result_df.columns:
        if scanned_cols >= 3:
            break
        try:
            series = previous_result_df[col].dropna()
            if series.empty:
                continue
            sample_vals = series.astype(str).head(300)
            # Prefer text-like columns for value mention matching.
            if not any(c.isalpha() for c in " ".join(sample_vals.head(20).tolist())):
                continue

            unique_values = sample_vals.str.lower().drop_duplicates().head(50).tolist()
            scanned_cols += 1
            for val in unique_values:
                if len(val) > 2:
                    # Check if full value appears in question
                    if val in question_lower:
                        return True
                    # Check if any question word appears in a result value
                    for word in question_words:
                        if len(word) > 2 and word in val:
                            return True
        except Exception:
            continue
    return False


def is_filter_modification_followup(question: str, previous_result_df=None) -> bool:
    q_lower = question.lower()
    words = q_lower.split()

    if previous_result_df is None or len(previous_result_df) == 0:
        return False

    new_query_indicators = [
        "agent wise", "agent-wise", "agentwise",
        "country wise", "country-wise", "countrywise",
        "product wise", "product-wise", "productwise",
        "monthly", "trend", "total revenue", "performance",
        "top ", "best ", "worst ", "all agents", "all bookings",
        "how many", "what is the", "show all", "list all",
    ]
    if any(indicator in q_lower for indicator in new_query_indicators):
        return False

    same_patterns = ["same", "similar", "but with", "but for", "instead of", "with cancelled", "with confirmed", "with pending"]
    if any(p in q_lower for p in same_patterns) and len(words) < 15:
        return True

    only_patterns = ["only", "just", "specifically", "particular"]
    has_only = any(p in q_lower for p in only_patterns)
    if has_only and len(words) <= 8:
        return True

    # "what about X?" / "how about X?" / "and X?" patterns
    about_patterns = ["what about", "how about", "and what about"]
    if any(p in q_lower for p in about_patterns) and len(words) <= 8:
        if _question_mentions_result_value(q_lower, previous_result_df):
            return True

    if len(words) <= 5 and any(p in q_lower for p in ["for ", "of ", "from ", "about "]):
        if _question_mentions_result_value(q_lower, previous_result_df):
            return True
        return False

    if _question_mentions_result_value(q_lower, previous_result_df):
        return True

    return False


# --- SQL modification ---

def modify_sql_for_filter(original_sql: str, filter_request: str, llm, db: Optional[SQLDatabase] = None) -> str:
    table_name = _extract_from_table_name(original_sql) or ""
    available_cols = ""
    if db is not None and table_name:
        table_entry = _get_schema_profile(db)["lookup"].get(table_name)
        if table_entry:
            available_cols = ", ".join(table_entry.get("columns", []))

    column_hint = ""
    if available_cols:
        column_hint = f"\nIMPORTANT: The table '{table_name}' only has these columns: {available_cols}\nDo NOT use columns that are not in this list."

    prompt = f"""You are a SQL expert. The user previously ran this query:

Previous SQL: {original_sql}

Now the user wants to modify it with this request: "{filter_request}"

Modify the SQL to satisfy the user's request. You MUST keep the SAME table/view and the SAME SELECT columns. Only add or change WHERE filters.

Rules:
- Output ONLY the modified SQL, nothing else
- KEEP the same FROM table - do NOT switch to a different table/view
- KEEP the same SELECT columns, GROUP BY, ORDER BY structure
- Only ADD or MODIFY the WHERE clause to filter results
- Text matching priority:
  1) exact match with '=' by default
  2) prefix match with LIKE 'value%' if user intent suggests starts-with/prefix
  3) contains match with LIKE '%value%' ONLY if user explicitly says contains/substring
- Keep date filters SARGable:
  - Use range predicates: date_col >= 'YYYY-MM-DD' AND date_col < 'YYYY-MM-DD'
  - Do NOT use YEAR(date_col), MONTH(date_col), or CAST(date_col AS DATE) in WHERE
- ONLY use columns that exist in the table being queried{column_hint}

Modified SQL:"""

    try:
        response = llm.invoke(prompt)
        modified_sql = response.content.strip() if hasattr(response, "content") else str(response).strip()
        modified_sql = _clean_sql_response(modified_sql)
        if modified_sql.upper().startswith("SELECT"):
            return modified_sql
        return original_sql
    except Exception:
        return original_sql


def modify_sql_for_sort(original_sql: str, sort_request: str, llm) -> str:
    sql_no_order = re.sub(r"\s+ORDER\s+BY\s+[^;]+", "", original_sql, flags=re.IGNORECASE)
    req_lower = sort_request.lower()

    if "differently" in req_lower or "reverse" in req_lower:
        new_order = "ORDER BY total_sales ASC" if "desc" in original_sql.lower() else "ORDER BY total_sales DESC"
    elif "name" in req_lower or "product" in req_lower or "hotel" in req_lower:
        new_order = "ORDER BY productname ASC"
    elif "date" in req_lower or "recent" in req_lower:
        new_order = "ORDER BY booking_date DESC"
    elif "price" in req_lower or "value" in req_lower or "sales" in req_lower or "revenue" in req_lower:
        new_order = "ORDER BY total_sales DESC"
    elif "profit" in req_lower:
        new_order = "ORDER BY total_profit DESC"
    elif "desc" in req_lower or "high" in req_lower:
        new_order = "ORDER BY total_sales DESC"
    elif "asc" in req_lower or "low" in req_lower:
        new_order = "ORDER BY total_sales ASC"
    else:
        prompt = f"""SQL: {original_sql}
User wants: {sort_request}
Output ONLY the ORDER BY clause (e.g., "ORDER BY column DESC"). No other text:"""
        try:
            resp = llm.invoke(prompt)
            clause = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            new_order = clause if clause.upper().startswith("ORDER BY") else "ORDER BY total_sales DESC"
        except Exception:
            new_order = "ORDER BY total_sales DESC"

    return f"{sql_no_order.rstrip().rstrip(';')} {new_order}"


def _is_bare_topn_followup(question: str) -> bool:
    """True for short contextual prompts like: 'give me top 10'."""
    req = _parse_topn_request(question)
    if req is None:
        return False
    q = " ".join((question or "").lower().strip().split())
    words = set(re.findall(r"[a-z_]+", q))
    # If user explicitly provides dimension/metric/date intent, this is not a bare follow-up.
    explicit_terms = {
        "revenue", "sales", "profit", "cost", "expense", "spend", "booking", "bookings",
        "supplier", "suppliers", "agent", "agents", "country", "countries", "city", "cities",
        "hotel", "hotels", "product", "products", "chain", "travel", "checkin", "checkout",
        "today", "yesterday", "month", "year", "week", "date",
    }
    if words & explicit_terms:
        return False
    if " by " in f" {q} ":
        return False
    return len(words) <= 6


def _replace_outer_top_clause(sql: str, limit: int) -> str:
    sql_clean = (sql or "").rstrip().rstrip(";")
    if not sql_clean:
        return sql_clean
    matches = list(re.finditer(r"(?i)\bSELECT\s+(DISTINCT\s+)?(?:TOP\s+\d+\s+)?", sql_clean))
    if not matches:
        return sql_clean
    m = matches[-1]
    distinct = m.group(1) or ""
    replacement = f"SELECT {distinct}TOP {limit} "
    return sql_clean[:m.start()] + replacement + sql_clean[m.end():]


def _set_order_direction(sql: str, direction: str) -> str:
    sql_clean = (sql or "").rstrip().rstrip(";")
    match = re.search(r"\bORDER\s+BY\b", sql_clean, flags=re.IGNORECASE)
    if not match:
        return sql_clean
    head = sql_clean[:match.start()]
    order_clause = sql_clean[match.start():]
    if direction == "bottom":
        if re.search(r"\bDESC\b", order_clause, flags=re.IGNORECASE):
            order_clause = re.sub(r"\bDESC\b", "ASC", order_clause, count=1, flags=re.IGNORECASE)
        elif not re.search(r"\bASC\b", order_clause, flags=re.IGNORECASE):
            order_clause = re.sub(r"(?i)(\bORDER\s+BY\s+[^,]+)", r"\1 ASC", order_clause, count=1)
    else:
        if re.search(r"\bASC\b", order_clause, flags=re.IGNORECASE) and not re.search(r"\bDESC\b", order_clause, flags=re.IGNORECASE):
            order_clause = re.sub(r"\bASC\b", "DESC", order_clause, count=1, flags=re.IGNORECASE)
        elif not re.search(r"\bASC\b|\bDESC\b", order_clause, flags=re.IGNORECASE):
            order_clause = re.sub(r"(?i)(\bORDER\s+BY\s+[^,]+)", r"\1 DESC", order_clause, count=1)
    return f"{head}{order_clause}"


def modify_sql_for_topn_followup(previous_sql: str, question: str) -> Optional[str]:
    """Reuse previous query context and only change TOP N / order direction."""
    req = _parse_topn_request(question)
    if req is None or not previous_sql:
        return None
    if not re.search(r"\bSELECT\b", previous_sql, flags=re.IGNORECASE):
        return None

    limit = max(1, min(int(req.get("limit", 10)), 100))
    rewritten = _replace_outer_top_clause(previous_sql, limit)
    rewritten = _set_order_direction(rewritten, req.get("direction", "top"))
    return rewritten


# --- Intent detection ---

def detect_intent_simple(message: str) -> Optional[str]:
    msg_lower = " ".join((message or "").lower().strip().split())
    words = set(re.findall(r"[a-z_]+", msg_lower))

    # Treat broad business-health prompts as data queries.
    if is_business_overview_question(msg_lower):
        return "DATA_QUERY"

    data_keywords = {
        "booking", "bookings", "supplier", "suppliers", "agent", "agents",
        "product", "products", "service", "services", "order", "orders",
        "customer", "customers", "sales", "revenue", "profit", "cost",
        "expense", "spend", "budget", "country", "countries", "city",
        "cities", "region", "regions", "show", "list", "find", "get",
        "count", "select", "where", "hotel", "hotels", "resort", "resorts",
        "all", "travel", "checkin", "checkout", "yesterday", "today",
        "month", "year", "date",
    }
    if len(words & data_keywords) >= 2:
        return "DATA_QUERY"

    ranking_words = {"top", "bottom", "highest", "maximum", "max", "lowest", "minimum", "most", "least"}
    entity_words = {"supplier", "suppliers", "agent", "agents", "hotel", "hotels", "product", "products", "country", "countries", "city", "cities"}
    metric_words = {"booking", "bookings", "revenue", "sales", "profit", "cost", "expense", "spend"}
    if words & ranking_words and (words & metric_words or words & entity_words):
        return "DATA_QUERY"

    if {"which", "who"} & words and (words & entity_words or words & metric_words):
        return "DATA_QUERY"

    if any(g in msg_lower for g in GREETING_PATTERNS) and len(words) <= 4:
        return "GREETING"
    if any(t in msg_lower for t in THANKS_PATTERNS):
        return "THANKS"
    if any(f in msg_lower for f in FAREWELL_PATTERNS):
        return "FAREWELL"
    if any(h in msg_lower for h in HELP_PATTERNS):
        return "HELP"

    return None


def detect_intent_llm(message: str, schema_summary: str, llm, conversation_context: str = "") -> str:
    try:
        context_snippet = conversation_context[-800:] if conversation_context else "No prior conversation."
        chain = INTENT_DETECTION_TEMPLATE | llm | StrOutputParser()
        result = chain.invoke({
            "message": message,
            "schema_summary": schema_summary,
            "conversation_context": context_snippet,
        })
        intent = result.strip().upper().replace(" ", "_")
        valid_intents = {"GREETING", "THANKS", "HELP", "FAREWELL", "OUT_OF_SCOPE", "CLARIFICATION_NEEDED", "DATA_QUERY"}
        return intent if intent in valid_intents else "DATA_QUERY"
    except Exception:
        logger.warning("LLM intent detection failed, defaulting to DATA_QUERY", exc_info=True)
        return "DATA_QUERY"


def is_contextual_data_query(question: str) -> bool:
    msg_lower = question.lower()
    words = set(msg_lower.split())
    has_entity_word = bool(words & DATABASE_ENTITY_WORDS)
    action_words = {"show", "give", "get", "list", "find", "filter", "details", "about", "for", "from", "with"}
    has_action_word = bool(words & action_words)
    if has_entity_word and has_action_word:
        return True
    if ("about" in msg_lower or "for" in msg_lower or "from" in msg_lower or "with" in msg_lower) and has_entity_word:
        return True
    return False


def _is_short_contextual_followup(question: str) -> bool:
    """
    Detect short follow-up phrases that are clearly data queries when there's prior context.
    Avoids LLM intent call for patterns like "what about India?", "only last week",
    "filter by UAE", "and Dubai?", etc.
    """
    q = question.strip().lower().rstrip("?. ")
    words = q.split()
    # Very short questions (≤ 5 words) starting with follow-up markers
    FOLLOWUP_STARTERS = (
        "what about", "how about", "and ", "only for", "just ", "only ",
        "filter by", "filter for", "show only", "but ", "what if",
        "drill down", "break down", "break it", "what's for", "what is for",
    )
    if any(q.startswith(p) for p in FOLLOWUP_STARTERS) and len(words) <= 7:
        return True
    # Short bare questions: "top 10?", "last week?", "only india" etc.
    if len(words) <= 4 and not any(g in q for g in ("hi", "hello", "hey", "bye", "thank")):
        return True
    return False


def has_recent_data_query(chat_history: List[Dict]) -> bool:
    if not chat_history:
        return False
    for chat in reversed(chat_history[-3:]):
        if chat.get("sql") and chat.get("intent") != "CONVERSATION":
            return True
    return False


# --- Response generation ---

_CONVERSATIONAL_RESPONSES = {
    "GREETING": [
        "Hi! I'm ready to help you explore your data. You can ask things like \"top 10 agents by revenue this month\" or \"total bookings last week\".",
        "Hello! Ask me anything about your bookings, agents, suppliers, countries, or hotels.",
        "Hey there! What data would you like to explore today?",
    ],
    "THANKS": [
        "You're welcome! Let me know if you have more questions.",
        "Happy to help! Feel free to ask anything else.",
        "Anytime! What else would you like to explore?",
    ],
    "FAREWELL": [
        "Goodbye! Come back anytime to explore your data.",
        "See you! Your data will be here when you return.",
    ],
    "HELP": [
        "I can query your booking database. Try asking:\n• \"Top 15 agents by cost for travel date last week\"\n• \"Revenue by country this month\"\n• \"Show bookings for India\"\n• \"YoY growth by hotel\"\n• \"Total bookings yesterday\"",
    ],
    "OUT_OF_SCOPE": [
        "I can only help with database queries about bookings, agents, suppliers, hotels, and related data. Try asking something like \"top agents by revenue this month\".",
    ],
}


def generate_conversational_response(message: str, intent: str, tables: str, history: str, llm) -> str:
    """Return an instant template response — no LLM call needed for conversational replies."""
    import random
    responses = _CONVERSATIONAL_RESPONSES.get(intent)
    if responses:
        return random.choice(responses)
    # Fallback only for CLARIFICATION_NEEDED — still needs LLM
    try:
        chain = CONVERSATIONAL_TEMPLATE | llm | StrOutputParser()
        return chain.invoke({"message": message, "intent": intent, "tables": tables, "history": history})
    except Exception:
        return "I'm here to help you explore your database. What would you like to know?"


def generate_nl_response(question: str, results, llm) -> str:
    """Generate NL answer. Accepts list[dict] (new) or pd.DataFrame (legacy)."""
    if llm is None:
        return ""
    try:
        if results is None:
            results_str = "No results found."
        elif hasattr(results, "head"):
            # Legacy DataFrame path
            pd.set_option("display.max_colwidth", None)
            results_str = results.head(15).to_string(index=False)
        elif isinstance(results, list) and results:
            rows = results[:15]
            keys = list(rows[0].keys()) if rows else []
            if keys:
                header = " | ".join(str(k) for k in keys)
                sep = "-" * min(len(header), 120)
                row_strs = [" | ".join(str(r.get(k, "")) for k in keys) for r in rows]
                results_str = "\n".join([header, sep] + row_strs)
            else:
                results_str = "No results found."
        else:
            results_str = "No results found."
        chain = NL_RESPONSE_TEMPLATE | llm | StrOutputParser()
        return chain.invoke({"question": question, "results": results_str})
    except Exception:
        return ""


def _round_cell_value(value: Any, places: int = 2) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, Decimal):
            q = Decimal("1").scaleb(-places)  # e.g. places=2 -> Decimal('0.01')
            return float(value.quantize(q, rounding=ROUND_HALF_UP))
        if isinstance(value, (float, np.floating)):
            if pd.isna(value):
                return value
            return round(float(value), places)
    except Exception:
        return value
    return value


def _results_to_records(df: Optional[pd.DataFrame], places: int = 2) -> List[Dict[str, Any]]:
    """Serialize DataFrame rows for API response with numeric values rounded to N decimals."""
    if df is None or len(df) == 0:
        return []
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_float_dtype(series):
            out[col] = series.round(places)
        elif pd.api.types.is_object_dtype(series):
            out[col] = series.map(lambda v: _round_cell_value(v, places))
    return out.to_dict(orient="records")


# --- SQL validation & fixing ---

def validate_sql(sql: str) -> Tuple[bool, Optional[str]]:
    if not sql or not sql.strip():
        return False, "The model returned an empty response. Please try rephrasing your question."
    normalized = sql.strip().lower()
    if not normalized.startswith(("select", "with")):
        return False, sql.strip()
    first_word = normalized.split()[0]
    if first_word in BLOCKED_KEYWORDS:
        return False, "Only SELECT queries are allowed."
    for keyword in BLOCKED_KEYWORDS:
        if f" {keyword} " in f" {normalized} ":
            return False, f"Query contains blocked keyword: {keyword.upper()}"
    return True, None


def _default_validator_result(
    ok_to_execute: bool = True,
    failure_type: str = "syntax",
    reasons: Optional[List[str]] = None,
    fixed_sql: Optional[str] = None,
    needs_retry: bool = False,
) -> Dict[str, Any]:
    out = {
        "ok_to_execute": bool(ok_to_execute),
        "failure_type": failure_type if failure_type in {"syntax", "schema", "safety", "intent_mismatch", "performance_risk"} else "syntax",
        "reasons": (reasons or [])[:5],
        "fixed_sql": fixed_sql if fixed_sql else None,
        "needs_retry": bool(needs_retry),
    }
    return out


def _parse_validator_output(raw_text: str) -> Dict[str, Any]:
    txt = (raw_text or "").strip()
    if not txt:
        return _default_validator_result()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", txt, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        txt = fence_match.group(1).strip()

    obj = None
    try:
        obj = json.loads(txt)
    except Exception:
        first = txt.find("{")
        last = txt.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                obj = json.loads(txt[first:last + 1])
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        return _default_validator_result(
            ok_to_execute=False,
            failure_type="syntax",
            reasons=["validator_output_not_json"],
            fixed_sql=None,
            needs_retry=False,
        )

    return _default_validator_result(
        ok_to_execute=bool(obj.get("ok_to_execute", False)),
        failure_type=str(obj.get("failure_type", "syntax")),
        reasons=[str(r) for r in (obj.get("reasons") or [])][:4],
        fixed_sql=obj.get("fixed_sql"),
        needs_retry=bool(obj.get("needs_retry", not bool(obj.get("ok_to_execute", False)))),
    )


def run_sql_intent_validator(
    *,
    llm,
    question: str,
    sql: str,
    dialect_label: str,
    full_schema: str,
    stored_procedure_guidance: str = "",
    required_table_hints: str = "",
    retrieved_context: str = "",
    timeout_ms: int = 1800,
) -> Dict[str, Any]:
    """Validate SQL against syntax/schema/intent/performance via strict JSON prompt."""
    if llm is None:
        valid, msg = validate_sql(sql)
        if valid:
            return _default_validator_result(ok_to_execute=True, failure_type="syntax", reasons=[], fixed_sql=None, needs_retry=False)
        return _default_validator_result(ok_to_execute=False, failure_type="syntax", reasons=[msg or "invalid_sql"], fixed_sql=None, needs_retry=True)

    prompt = SQL_VALIDATOR_TEMPLATE.format(
        dialect=dialect_label,
        schema_compact=(full_schema or ""),
        top_tables=(required_table_hints or "none"),
        question=question or "",
        sql=sql or "",
    )
    try:
        resp = _invoke_with_timeout(lambda: llm.invoke(prompt), timeout_ms=timeout_ms)
        raw = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        return _parse_validator_output(raw)
    except Exception as exc:
        logger.warning("SQL intent validator failed, continuing with local validation: %s", exc)
        valid, msg = validate_sql(sql)
        if valid:
            return _default_validator_result(ok_to_execute=True, failure_type="syntax", reasons=[], fixed_sql=None, needs_retry=False)
        return _default_validator_result(ok_to_execute=False, failure_type="syntax", reasons=[msg or "invalid_sql"], fixed_sql=None, needs_retry=True)


def fix_common_sql_errors(sql: str, dialect: str = DEFAULT_SQL_DIALECT) -> str:
    if not sql:
        return sql
    d = normalize_sql_dialect(dialect)
    if d == "sqlserver":
        # Convert PostgreSQL LIMIT N to SQL Server TOP N.
        limit_match = re.search(r"\bLIMIT\s+(\d+)\s*;?\s*$", sql, flags=re.IGNORECASE)
        if limit_match and not re.search(r"\bSELECT\s+TOP\s+\d+\b", sql, flags=re.IGNORECASE):
            limit_val = limit_match.group(1)
            sql = sql[:limit_match.start()].rstrip().rstrip(";")
            sql = re.sub(
                r"(?i)^SELECT\s+(DISTINCT\s+)?",
                lambda m: f"SELECT {m.group(1) or ''}TOP {limit_val} ",
                sql,
                count=1,
            )

        # Normalize date/time functions to SQL Server syntax.
        sql = re.sub(r"\bCURRENT_DATE\b", "CAST(GETDATE() AS DATE)", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\bNOW\(\)", "GETDATE()", sql, flags=re.IGNORECASE)
        sql = re.sub(r"EXTRACT\(\s*YEAR\s+FROM\s+([^)]+)\)", r"YEAR(\1)", sql, flags=re.IGNORECASE)
        sql = re.sub(r"EXTRACT\(\s*MONTH\s+FROM\s+([^)]+)\)", r"MONTH(\1)", sql, flags=re.IGNORECASE)
        sql = re.sub(
            r"date_trunc\(\s*'month'\s*,\s*([^)]+)\)",
            r"DATEFROMPARTS(YEAR(\1), MONTH(\1), 1)",
            sql,
            flags=re.IGNORECASE,
        )
        sql = re.sub(
            r"(GETDATE\(\)|CAST\(GETDATE\(\)\s+AS\s+DATE\))\s*-\s*INTERVAL\s*'(\d+)\s+days'",
            r"DATEADD(DAY, -\2, \1)",
            sql,
            flags=re.IGNORECASE,
        )
        sql = re.sub(
            r"(GETDATE\(\)|CAST\(GETDATE\(\)\s+AS\s+DATE\))\s*-\s*INTERVAL\s*'(\d+)\s+months?'",
            r"DATEADD(MONTH, -\2, \1)",
            sql,
            flags=re.IGNORECASE,
        )

        # SQL Server does not support ILIKE.
        sql = re.sub(r"\bILIKE\b", "LIKE", sql, flags=re.IGNORECASE)
    else:
        # Convert SQL Server TOP N to PostgreSQL LIMIT N.
        top_match = re.search(r"(?is)^\s*SELECT\s+(DISTINCT\s+)?TOP\s+(\d+)\s+", sql)
        if top_match and not re.search(r"\bLIMIT\s+\d+\b", sql, flags=re.IGNORECASE):
            limit_val = top_match.group(2)
            sql = re.sub(
                r"(?is)^(\s*SELECT\s+)(DISTINCT\s+)?TOP\s+\d+\s+",
                lambda m: f"{m.group(1)}{m.group(2) or ''}",
                sql,
                count=1,
            ).rstrip().rstrip(";")
            sql = f"{sql} LIMIT {limit_val}"
    return sql


def _as_sql_date_expr(expr: str) -> str:
    expr_clean = " ".join((expr or "").strip().split())
    if not expr_clean:
        return "CAST(GETDATE() AS DATE)"
    if re.match(r"(?i)^cast\(.+as\s+date\)$", expr_clean):
        return expr_clean
    if re.match(r"(?i)^getdate\(\)$", expr_clean):
        return "CAST(GETDATE() AS DATE)"
    if re.match(r"(?i)^'[^']+'$", expr_clean):
        return f"CAST({expr_clean} AS DATE)"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", expr_clean):
        return f"CAST('{expr_clean}' AS DATE)"
    return f"CAST({expr_clean} AS DATE)"


def _rewrite_cast_date_predicates_sargable(sql: str) -> str:
    col_pat = r"(?:\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*))?"
    rhs_pat = (
        r"(?:CAST\s*\(\s*GETDATE\(\)\s+AS\s+DATE\s*\)|GETDATE\(\)|DATEADD\s*\([^)]*\)|"
        r"DATEFROMPARTS\s*\([^)]*\)|'[^']+'|\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)?)"
    )

    def repl_eq(match: re.Match) -> str:
        col = match.group("col")
        rhs = _as_sql_date_expr(match.group("rhs"))
        return f"{col} >= {rhs} AND {col} < DATEADD(DAY, 1, {rhs})"

    def repl_ge(match: re.Match) -> str:
        col = match.group("col")
        rhs = _as_sql_date_expr(match.group("rhs"))
        return f"{col} >= {rhs}"

    def repl_gt(match: re.Match) -> str:
        col = match.group("col")
        rhs = _as_sql_date_expr(match.group("rhs"))
        return f"{col} >= DATEADD(DAY, 1, {rhs})"

    def repl_le(match: re.Match) -> str:
        col = match.group("col")
        rhs = _as_sql_date_expr(match.group("rhs"))
        return f"{col} < DATEADD(DAY, 1, {rhs})"

    def repl_lt(match: re.Match) -> str:
        col = match.group("col")
        rhs = _as_sql_date_expr(match.group("rhs"))
        return f"{col} < {rhs}"

    sql = re.sub(
        rf"CAST\(\s*(?P<col>{col_pat})\s+AS\s+DATE\s*\)\s*=\s*(?P<rhs>{rhs_pat})",
        repl_eq,
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        rf"CAST\(\s*(?P<col>{col_pat})\s+AS\s+DATE\s*\)\s*>=\s*(?P<rhs>{rhs_pat})",
        repl_ge,
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        rf"CAST\(\s*(?P<col>{col_pat})\s+AS\s+DATE\s*\)\s*>\s*(?P<rhs>{rhs_pat})",
        repl_gt,
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        rf"CAST\(\s*(?P<col>{col_pat})\s+AS\s+DATE\s*\)\s*<=\s*(?P<rhs>{rhs_pat})",
        repl_le,
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        rf"CAST\(\s*(?P<col>{col_pat})\s+AS\s+DATE\s*\)\s*<\s*(?P<rhs>{rhs_pat})",
        repl_lt,
        sql,
        flags=re.IGNORECASE,
    )
    return sql


def _rewrite_year_month_predicates_sargable(sql: str) -> str:
    col_pat = r"(?:\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*))?"
    rhs_pat = r"(?:GETDATE\(\)|DATEADD\s*\([^)]*\)|CAST\s*\(\s*GETDATE\(\)\s+AS\s+DATE\s*\)|'[^']+'|\d{4}-\d{2}-\d{2})"

    def token_key(token: str) -> str:
        return re.sub(r"\s+", "", (token or "")).lower()

    def repl_month_year(match: re.Match) -> str:
        col1 = match.group("col1")
        col2 = match.group("col2")
        rhs1 = match.group("rhs1")
        rhs2 = match.group("rhs2")
        if token_key(col1) != token_key(col2) or token_key(rhs1) != token_key(rhs2):
            return match.group(0)
        rhs = rhs1.strip()
        month_start = f"DATEFROMPARTS(YEAR({rhs}), MONTH({rhs}), 1)"
        return f"{col1} >= {month_start} AND {col1} < DATEADD(MONTH, 1, {month_start})"

    sql = re.sub(
        rf"YEAR\(\s*(?P<col1>{col_pat})\s*\)\s*=\s*YEAR\(\s*(?P<rhs1>{rhs_pat})\s*\)\s*"
        rf"AND\s*MONTH\(\s*(?P<col2>{col_pat})\s*\)\s*=\s*MONTH\(\s*(?P<rhs2>{rhs_pat})\s*\)",
        repl_month_year,
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        rf"MONTH\(\s*(?P<col1>{col_pat})\s*\)\s*=\s*MONTH\(\s*(?P<rhs1>{rhs_pat})\s*\)\s*"
        rf"AND\s*YEAR\(\s*(?P<col2>{col_pat})\s*\)\s*=\s*YEAR\(\s*(?P<rhs2>{rhs_pat})\s*\)",
        repl_month_year,
        sql,
        flags=re.IGNORECASE,
    )

    def repl_year_func(match: re.Match) -> str:
        col = match.group("col")
        rhs = match.group("rhs").strip()
        return f"{col} >= DATEFROMPARTS(YEAR({rhs}), 1, 1) AND {col} < DATEFROMPARTS(YEAR({rhs}) + 1, 1, 1)"

    sql = re.sub(
        rf"YEAR\(\s*(?P<col>{col_pat})\s*\)\s*=\s*YEAR\(\s*(?P<rhs>{rhs_pat})\s*\)",
        repl_year_func,
        sql,
        flags=re.IGNORECASE,
    )

    def repl_year_lit(match: re.Match) -> str:
        col = match.group("col")
        year_val = int(match.group("year"))
        return f"{col} >= '{year_val:04d}-01-01' AND {col} < '{year_val + 1:04d}-01-01'"

    sql = re.sub(
        rf"YEAR\(\s*(?P<col>{col_pat})\s*\)\s*=\s*(?P<year>(?:19|20)\d{{2}})",
        repl_year_lit,
        sql,
        flags=re.IGNORECASE,
    )
    return sql


def _rewrite_literal_year_month_pairs_sargable(sql: str) -> str:
    col_pat = r"(?:\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*))?"

    def repl(match: re.Match) -> str:
        col = match.group("col")
        year_val = int(match.group("year"))
        month_val = int(match.group("month"))
        if month_val < 1 or month_val > 12:
            return match.group(0)
        start = f"DATEFROMPARTS({year_val}, {month_val}, 1)"
        return f"{col} >= {start} AND {col} < DATEADD(MONTH, 1, {start})"

    sql = re.sub(
        rf"YEAR\(\s*(?P<col>{col_pat})\s*\)\s*=\s*(?P<year>(?:19|20)\d{{2}})\s*"
        rf"AND\s*MONTH\(\s*(?P=col)\s*\)\s*=\s*(?P<month>(?:0?[1-9]|1[0-2]))",
        repl,
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        rf"MONTH\(\s*(?P<col>{col_pat})\s*\)\s*=\s*(?P<month>(?:0?[1-9]|1[0-2]))\s*"
        rf"AND\s*YEAR\(\s*(?P=col)\s*\)\s*=\s*(?P<year>(?:19|20)\d{{2}})",
        repl,
        sql,
        flags=re.IGNORECASE,
    )
    return sql


# Keywords that can appear after a table name and must NOT be mistaken for aliases.
_NOLOCK_STOP_WORDS = frozenset({
    "with", "on", "where", "group", "order", "having",
    "inner", "left", "right", "full", "cross", "outer",
    "select", "union", "set", "into", "join", "from",
    "as", "and", "or", "not", "in", "is", "null",
})


def _inject_nolock_hints(sql: str) -> str:
    """Add WITH (NOLOCK) to every table reference that does not already have it.

    Handles:
      FROM dbo.BookingData          → FROM dbo.BookingData WITH (NOLOCK)
      FROM dbo.BookingData BD       → FROM dbo.BookingData BD WITH (NOLOCK)
      JOIN AM on ...                → JOIN AM WITH (NOLOCK) on ...
      FROM dbo.BookingData with (nolock) → unchanged (no duplicate)
    """
    if not sql:
        return sql

    stop_alt = "|".join(sorted(_NOLOCK_STOP_WORDS, key=len, reverse=True))
    # alias_part: optional whitespace + one word that is NOT a stop-word keyword.
    # NOTE: no outer negative lookahead here — we use the replacement function
    # to check the original string, avoiding the regex backtracking bug where
    # the engine retries without the alias and the lookahead incorrectly passes.
    alias_part = rf"(?:\s+(?!(?:{stop_alt})\b)[A-Za-z_]\w*)?"

    pattern = re.compile(
        r"(?i)"
        r"(\b(?:FROM|JOIN)\s+)"              # FROM or JOIN   (group 1)
        r"((?:\[?\w+\]?\.)?(?:\[?\w+\]?))"  # [schema.]table (group 2)
        rf"({alias_part})",                  # optional alias (group 3)
    )

    def _repl(m: re.Match) -> str:
        # Check the ORIGINAL string for WITH ( immediately after this match.
        # This avoids the backtracking issue with a negative lookahead on an
        # optional group: we let the engine match greedily, then decide here.
        remaining = sql[m.end():]
        if re.match(r"\s*WITH\s*\(", remaining, re.IGNORECASE):
            return m.group(0)  # already has a hint — leave untouched
        return f"{m.group(1)}{m.group(2)}{m.group(3)} WITH (NOLOCK)"

    return pattern.sub(_repl, sql)


def _strip_nolock_hints(sql: str) -> str:
    if not sql:
        return sql
    return re.sub(r"\s+WITH\s*\(\s*NOLOCK\s*\)", "", sql, flags=re.IGNORECASE)


def optimize_sql_for_performance(
    sql: str,
    dialect: str = DEFAULT_SQL_DIALECT,
    enable_nolock: bool = False,
) -> str:
    """Apply safe SQL rewrites that keep intent but improve execution speed."""
    if not sql:
        return sql
    d = normalize_sql_dialect(dialect)
    optimized = sql
    if d == "sqlserver":
        optimized = _rewrite_cast_date_predicates_sargable(optimized)
        optimized = _rewrite_year_month_predicates_sargable(optimized)
    optimized = _inject_nolock_hints(optimized) if enable_nolock else _strip_nolock_hints(optimized)
    return optimized


def _drop_distinct_when_grouped(sql: str) -> str:
    if not sql:
        return sql
    if not re.search(r"\bGROUP\s+BY\b", sql, flags=re.IGNORECASE):
        return sql
    return re.sub(r"(?i)\bSELECT\s+DISTINCT\b", "SELECT", sql, count=1)


def _query_has_aggregation(sql: str) -> bool:
    sql_lower = (sql or "").lower()
    return any(tok in sql_lower for tok in (" sum(", " count(", " avg(", " min(", " max(", " group by "))


def _question_requests_all_time(question: str) -> bool:
    q = (question or "").lower()
    all_time_tokens = {
        "all time", "all data", "overall", "historical", "history", "lifetime", "since inception", "ever", "from beginning",
    }
    return any(t in q for t in all_time_tokens)


def _question_has_explicit_date_scope(question: str) -> bool:
    q = (question or "").lower()
    if re.search(r"\b(19|20)\d{2}\b", q):
        return True
    date_tokens = {
        "today", "yesterday", "week", "month", "year", "quarter", "date", "ytd", "pytd",
        "last ", "this ", "between", "from ", "to ", "since ",
    }
    return any(t in q for t in date_tokens)


def _sql_has_date_filter_for_col(sql: str, date_col: str) -> bool:
    if not sql or not date_col:
        return False
    sql_lower = sql.lower()
    col_lower = date_col.lower()
    if " where " not in f" {sql_lower} ":
        return False
    return col_lower in sql_lower


def apply_query_performance_guardrails(
    sql: str,
    question: str,
    db: Optional[SQLDatabase] = None,
    dialect: str = DEFAULT_SQL_DIALECT,
) -> str:
    """Apply conservative, execution-focused SQL guardrails."""
    if not sql:
        return sql
    guarded = _drop_distinct_when_grouped(sql)
    if db is None:
        return guarded
    if normalize_sql_dialect(dialect) != "sqlserver":
        return guarded

    # For heavy BookingData aggregates without explicit time scope, constrain to current year.
    # This avoids full-table scans on large transactional history.
    if _query_has_aggregation(guarded) and not _question_requests_all_time(question) and not _question_has_explicit_date_scope(question):
        table_entry = _get_table_entry_from_sql(guarded, db)
        if table_entry and table_entry.get("table_normalized") == "bookingdata":
            created_col = _pick_existing_column(table_entry, ["createddate", "created_date", "bookingdate", "booking_date"])
            if created_col and not _sql_has_date_filter_for_col(guarded, created_col):
                created_col_quoted = _quote_sql_identifier(created_col)
                current_year_cond = (
                    f"{created_col_quoted} >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1) "
                    f"AND {created_col_quoted} < DATEFROMPARTS(YEAR(GETDATE()) + 1, 1, 1)"
                )
                guarded = _inject_condition_into_sql(guarded, current_year_cond)
    return guarded


def _inject_condition_into_sql(sql: str, condition_sql: str) -> str:
    sql_clean = sql.rstrip().rstrip(";")
    match = re.search(r"\b(GROUP\s+BY|ORDER\s+BY|HAVING)\b", sql_clean, re.IGNORECASE)
    if re.search(r"\bWHERE\b", sql_clean, re.IGNORECASE):
        if match:
            return sql_clean[:match.start()].rstrip() + f" AND {condition_sql} " + sql_clean[match.start():]
        return sql_clean + f" AND {condition_sql}"
    if match:
        return sql_clean[:match.start()].rstrip() + f" WHERE {condition_sql} " + sql_clean[match.start():]
    return sql_clean + f" WHERE {condition_sql}"


_LAST_PERF_GUARDRAILS_APPLIED: List[str] = []


def _set_last_perf_guardrails(applied: List[str]) -> None:
    global _LAST_PERF_GUARDRAILS_APPLIED
    _LAST_PERF_GUARDRAILS_APPLIED = list(applied)


def _get_last_perf_guardrails() -> List[str]:
    return list(_LAST_PERF_GUARDRAILS_APPLIED)


def _sql_requests_all_time(sql: str) -> bool:
    s = (sql or "").lower()
    return any(tok in s for tok in ("all time", "all_time", "overall", "lifetime", "since inception"))


def _targets_large_fact_like_source(sql: str) -> bool:
    s = (sql or "").lower()
    patterns = (
        "bookingdata",
        "bookingtablequery",
        "country_level_view",
        "agent_level_view",
    )
    return any(p in s for p in patterns)


def _sql_has_any_date_filter(sql: str) -> bool:
    if not sql:
        return False
    where_match = re.search(
        r"(?is)\bWHERE\b\s*(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|$)",
        sql,
    )
    if not where_match:
        return False
    where_sql = where_match.group(1)
    date_cols = (
        "createddate", "checkindate", "checkoutdate",
        "bookingdate", "booking_date", "date",
    )
    for col in date_cols:
        col_pat = rf"(?:\b\w+\.)?\[?{re.escape(col)}\]?"
        if re.search(rf"(?i){col_pat}\s*(=|>=|>|<=|<|between)", where_sql):
            return True
        if re.search(rf"(?i)\bcast\s*\(\s*[^)]*{col_pat}[^)]*\)\s+as\s+date\s*\)", where_sql):
            return True
        if re.search(rf"(?i)\b(year|month)\s*\(\s*[^)]*{col_pat}[^)]*\)", where_sql):
            return True
    return False


def _rewrite_leading_wildcard_like(sql: str, question: str = "") -> str:
    if not sql:
        return sql
    q = (question or "").lower()
    if any(tok in q for tok in ("contains", "contain", "substring", "anywhere", "includes")):
        return sql

    pattern = re.compile(
        r"(?i)(?P<col>(?:\[[^\]]+\]|\w+)(?:\.(?:\[[^\]]+\]|\w+))?)\s+LIKE\s+N?'%(?P<val>[^%_']+)%'"
    )

    def repl(match: re.Match) -> str:
        col = match.group("col")
        val = match.group("val").strip()
        if not val:
            return match.group(0)
        # Prefer exact match for identifier-like tokens; otherwise prefix for index-friendliness.
        if re.fullmatch(r"[A-Za-z0-9_.\-]+", val):
            return f"{col} = '{val}'"
        return f"{col} LIKE '{val}%'"

    return pattern.sub(repl, sql)


def _outer_select_has_top_or_limit(sql: str) -> bool:
    if not sql:
        return True
    sql_clean = sql.strip()
    if re.search(r"(?i)\bLIMIT\s+\d+\b", sql_clean):
        return True
    select_matches = list(re.finditer(r"(?i)\bSELECT\b", sql_clean))
    if not select_matches:
        return True
    outer = sql_clean[select_matches[-1].start():]
    return re.match(r"(?is)\s*SELECT\s+(DISTINCT\s+)?TOP\s*\(?\s*\d+\s*\)?\s+", outer) is not None


def _add_outer_top_200(sql: str) -> str:
    if not sql:
        return sql
    if _outer_select_has_top_or_limit(sql):
        return sql
    sql_clean = sql.rstrip().rstrip(";")
    matches = list(re.finditer(r"(?i)\bSELECT\s+(DISTINCT\s+)?", sql_clean))
    if not matches:
        return sql
    m = matches[-1]
    distinct = m.group(1) or ""
    replacement = f"SELECT {distinct}TOP (200) "
    return sql_clean[:m.start()] + replacement + sql_clean[m.end():]


def _detect_preferred_date_col(sql: str, db: Optional[SQLDatabase] = None) -> Optional[str]:
    candidates = ["CreatedDate", "BookingDate", "booking_date", "CheckInDate", "CheckOutDate"]
    for col in candidates:
        if re.search(rf"(?i)\b{re.escape(col)}\b", sql or ""):
            return col

    table_entry = _get_table_entry_from_sql(sql, db) if db is not None else None
    if table_entry:
        picked = _pick_existing_column(table_entry, [c.lower() for c in candidates])
        if picked:
            return picked
        return _detect_date_column(table_entry)

    if _targets_large_fact_like_source(sql):
        return "CreatedDate"
    return None


def _add_default_order_by_date_desc(sql: str, date_col: Optional[str]) -> str:
    if not sql or not date_col:
        return sql
    if re.search(r"(?i)\bORDER\s+BY\b", sql):
        return sql
    return f"{sql.rstrip().rstrip(';')} ORDER BY {date_col} DESC"


def _has_top_level_group_by(sql: str) -> bool:
    return re.search(r"(?i)\bGROUP\s+BY\b", sql or "") is not None


def _has_aggregate_functions(sql: str) -> bool:
    return bool(re.search(r"(?i)\b(sum|count|avg|min|max)\s*\(", sql or ""))


def _is_single_row_aggregate_query(sql: str) -> bool:
    """True for aggregate KPI-style queries: aggregates present, no GROUP BY."""
    if not sql:
        return False
    return _has_aggregate_functions(sql) and not _has_top_level_group_by(sql)


def _strip_trailing_order_by(sql: str) -> str:
    """Remove trailing ORDER BY clause from outer query if present."""
    if not sql:
        return sql
    sql_clean = sql.rstrip().rstrip(";")
    stripped = re.sub(r"(?is)\s+ORDER\s+BY\s+[^;]+$", "", sql_clean).strip()
    return stripped if stripped else sql


def _extract_tables_for_diagnostics(sql: str) -> List[str]:
    if not sql:
        return []
    tables = []
    for m in re.finditer(r"(?i)\b(?:FROM|JOIN)\s+([^\s,;]+)", sql):
        token = _normalize_identifier(m.group(1))
        if token and token not in tables:
            tables.append(token)
    return tables


def _has_non_sargable_date_funcs(sql: str) -> bool:
    if not sql:
        return False
    where_match = re.search(
        r"(?is)\bWHERE\b\s*(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|$)",
        sql,
    )
    scope = where_match.group(1) if where_match else sql
    return bool(
        re.search(r"(?i)\bCAST\s*\([^)]*\bAS\s+DATE\s*\)", scope)
        or re.search(r"(?i)\bYEAR\s*\(", scope)
        or re.search(r"(?i)\bMONTH\s*\(", scope)
    )


### PERFORMANCE GUARDRAILS
def _has_bounded_date_range(sql: str, date_col: str) -> bool:
    if not sql:
        return False
    col = re.escape((date_col or "CreatedDate").strip("[]"))
    lower = re.search(rf"(?i)\b(?:BD\.)?\[?{col}\]?\s*(>=|>)", sql) is not None
    upper = re.search(rf"(?i)\b(?:BD\.)?\[?{col}\]?\s*(<|<=)", sql) is not None
    between = re.search(rf"(?i)\b(?:BD\.)?\[?{col}\]?\s+BETWEEN\b", sql) is not None
    return (lower and upper) or between


def _ensure_bookingdata_bounded_date_range(sql: str, question: str = "") -> str:
    if not sql:
        return sql
    if not re.search(r"(?i)\bbookingdata\b", sql):
        return sql
    if _question_requests_all_time(question) or _sql_requests_all_time(sql):
        return sql

    date_col = _detect_preferred_date_col(sql) or "CreatedDate"
    if _has_bounded_date_range(sql, date_col):
        return sql

    month_start = "DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)"
    col = re.escape(date_col.strip("[]"))
    lower_exists = re.search(rf"(?i)\b(?:BD\.)?\[?{col}\]?\s*(>=|>)", sql) is not None
    upper_exists = re.search(rf"(?i)\b(?:BD\.)?\[?{col}\]?\s*(<|<=)", sql) is not None
    conditions = []
    if not lower_exists:
        conditions.append(f"{date_col} >= {month_start}")
    if not upper_exists:
        conditions.append(f"{date_col} < DATEADD(MONTH, 1, {month_start})")
    if not conditions:
        return sql
    return _inject_condition_into_sql(sql, " AND ".join(conditions))


def _rewrite_format_to_monthstart(sql: str, question: str = "") -> str:
    if not sql:
        return sql
    q = (question or "").lower()
    if any(tok in q for tok in ("format", "formatted", "as text", "string month")):
        return sql

    pattern = re.compile(r"(?i)FORMAT\s*\(\s*(?P<expr>[^,\)]+?)\s*,\s*'[^']+'\s*\)")

    def repl(match: re.Match) -> str:
        expr = match.group("expr").strip()
        return f"DATEFROMPARTS(YEAR({expr}), MONTH({expr}), 1)"

    rewritten = pattern.sub(repl, sql)
    rewritten = re.sub(
        r"(?i)AS\s+\[?(month|monthname|month_key)\]?",
        "AS [MonthStart]",
        rewritten,
    )
    return rewritten


def _remove_distinct_from_agentmaster(sql: str) -> str:
    if not sql:
        return sql
    # Trim DISTINCT in AgentMaster-only lookup subqueries/CTEs where it is usually redundant.
    return re.sub(
        r"(?is)(SELECT)\s+DISTINCT(\s+.*?\bFROM\s+(?:\[?dbo\]?\.)?\[?AgentMaster_V1\]?\b)",
        r"\1\2",
        sql,
    )


### PERFORMANCE GUARDRAILS
def performance_guardrails(
    sql: str,
    schema_text: str = "",
    question: str = "",
    state: Optional[Dict[str, Any]] = None,
    db: Optional[SQLDatabase] = None,
    dialect: str = DEFAULT_SQL_DIALECT,
    enable_nolock: bool = False,
) -> str:
    """Deterministic SQL performance guardrails executed before query execution."""
    if not sql:
        _set_last_perf_guardrails([])
        return sql

    # Backward compatibility: older call sites may pass question as 2nd positional arg.
    if (
        not question
        and schema_text
        and len(schema_text) < 300
        and not re.search(r"(?i)\b(create\s+table|select\s+|from\s+)\b", schema_text)
    ):
        question = schema_text
        schema_text = ""

    applied: List[str] = []
    guarded = sql
    d = normalize_sql_dialect(dialect)

    # Rule 2: rewrite non-SARGable date predicates.
    if d == "sqlserver":
        before = guarded
        guarded = _rewrite_cast_date_predicates_sargable(guarded)
        guarded = _rewrite_literal_year_month_pairs_sargable(guarded)
        guarded = _rewrite_year_month_predicates_sargable(guarded)
        if guarded != before:
            applied.append("rewrite_non_sargable_date_predicates")

        before = guarded
        guarded = _rewrite_format_to_monthstart(guarded, question=question)
        if guarded != before:
            applied.append("rewrite_format_to_monthstart")

    # Rule 3: replace leading-wildcard LIKE by exact/prefix unless user asked contains.
    before = guarded
    guarded = _rewrite_leading_wildcard_like(guarded, question=question)
    if guarded != before:
        applied.append("rewrite_leading_wildcard_like")

    # Preserve existing safe rewrites (NOLOCK policy + sargability helper consistency).
    before = guarded
    guarded = optimize_sql_for_performance(guarded, dialect=d, enable_nolock=enable_nolock)
    if guarded != before:
        applied.append("optimize_sql_for_performance")

    before = guarded
    guarded = _remove_distinct_from_agentmaster(guarded)
    if guarded != before:
        applied.append("remove_distinct_agentmaster")

    # Rule 1: default date window for large fact-like sources when missing date filter.
    if (
        d == "sqlserver"
        and (_targets_large_fact_like_source(guarded) or _targets_large_fact_like_source(schema_text))
        and not _sql_has_any_date_filter(guarded)
        and not _has_bounded_date_range(guarded, _detect_preferred_date_col(guarded) or "CreatedDate")
        and not _question_requests_all_time(question)
        and not _sql_requests_all_time(guarded)
    ):
        month_start = "DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)"
        default_window = (
            f"CreatedDate >= {month_start} "
            f"AND CreatedDate < DATEADD(MONTH, 1, {month_start})"
        )
        before = guarded
        guarded = _inject_condition_into_sql(guarded, default_window)
        if guarded != before:
            applied.append("inject_default_current_month_window")

    # BookingData hard guardrail: require bounded SARGable CreatedDate range.
    if d == "sqlserver":
        before = guarded
        guarded = _ensure_bookingdata_bounded_date_range(guarded, question=question)
        if guarded != before:
            applied.append("ensure_bookingdata_bounded_createddate_range")

    # Single-row aggregate safety: ORDER BY on non-grouped columns is invalid on SQL Server.
    if d == "sqlserver" and _is_single_row_aggregate_query(guarded):
        before = guarded
        guarded = _strip_trailing_order_by(guarded)
        if guarded != before:
            applied.append("strip_order_by_single_row_aggregate")

    # Rule 4: cap detailed result sets.
    if d == "sqlserver":
        has_group_by = re.search(r"(?i)\bGROUP\s+BY\b", guarded) is not None
        has_aggregation = _has_aggregate_functions(guarded)
        # Apply TOP/ORDER cap only for true detail-row queries.
        # Aggregate-only queries (SUM/COUNT without GROUP BY) must stay single-row
        # and cannot ORDER BY non-grouped columns.
        if not has_group_by and not has_aggregation and not _outer_select_has_top_or_limit(guarded):
            before = guarded
            guarded = _add_outer_top_200(guarded)
            date_col = (
                (state or {}).get("last_date_col")
                or _detect_preferred_date_col(guarded, db=db)
            )
            guarded = _add_default_order_by_date_desc(guarded, date_col=date_col)
            if guarded != before:
                applied.append("detail_cap_top_200_and_order_by_date")

    # Keep existing conservative guardrails.
    before = guarded
    guarded = apply_query_performance_guardrails(guarded, question=question, db=db, dialect=d)
    if guarded != before:
        applied.append("apply_query_performance_guardrails")

    _set_last_perf_guardrails(applied)
    return guarded


# Known ID columns that should never appear as output — map to their name counterparts.
_ID_COLUMN_MAP: Dict[str, str] = {
    "agentid":          "AgentName",
    "productid":        "ProductName",
    "hotelid":          "HotelName",
    "countryid":        "Country",
    "productcountryid": "Country",
    "cityid":           "City",
    "productcityid":    "City",
    "supplierid":       "suppliername",
    "employeeid":       "suppliername",
    "subagentid":       "AgentName",
    "masterid":         "ProductName",
    "branchid":         "BranchId",   # no friendly name — flag only
}


def _detect_raw_id_columns_in_select(sql: str) -> List[str]:
    """Return a list of raw ID column names found in the SELECT clause output."""
    if not sql:
        return []
    sql_upper = sql.upper()
    # Find the first SELECT (skip CTEs)
    last_select = sql_upper.rfind("SELECT")
    if last_select == -1:
        return []
    # Extract SELECT list — up to first FROM
    select_segment = sql[last_select:]
    from_match = re.search(r"\bFROM\b", select_segment, re.IGNORECASE)
    if from_match:
        select_segment = select_segment[:from_match.start()]

    found = []
    for id_col in _ID_COLUMN_MAP:
        # Match the bare column name as a selected column (not inside a function or WHERE)
        # Pattern: id_col appearing as SELECT output — either aliased or bare
        if re.search(rf"(?<![.\w]){re.escape(id_col)}(?!\s*=|\s*<|\s*>|\s*!=|\s*IN\b|\s*NOT\b|\s*IS\b|\s*LIKE\b|\s*\()", select_segment, re.IGNORECASE):
            found.append(id_col)
    return found


# Join templates for deterministic ID-column fix (avoids LLM retry).
_DETERMINISTIC_ID_FIX_JOINS: Dict[str, Dict[str, str]] = {
    "agentid":          {"name_col": "AgentName",      "join": "LEFT JOIN dbo.AgentMaster_V1 _IDJOIN ON _IDJOIN.AgentId = {alias}.AgentId",       "alias_col": "AgentId"},
    "productid":        {"name_col": "ProductName",     "join": None,  "alias_col": None},  # direct on BookingData
    "hotelid":          {"name_col": "HotelName",       "join": "LEFT JOIN dbo.Hotelchain _IDJOIN ON _IDJOIN.HotelId = {alias}.ProductId",         "alias_col": "ProductId"},
    "countryid":        {"name_col": "Country",         "join": "LEFT JOIN dbo.Master_Country _IDJOIN ON _IDJOIN.CountryID = {alias}.ProductCountryid", "alias_col": "ProductCountryid"},
    "productcountryid": {"name_col": "Country",         "join": "LEFT JOIN dbo.Master_Country _IDJOIN ON _IDJOIN.CountryID = {alias}.ProductCountryid", "alias_col": "ProductCountryid"},
    "cityid":           {"name_col": "City",            "join": "LEFT JOIN dbo.Master_City _IDJOIN ON _IDJOIN.CityId = {alias}.ProductCityId",      "alias_col": "ProductCityId"},
    "productcityid":    {"name_col": "City",            "join": "LEFT JOIN dbo.Master_City _IDJOIN ON _IDJOIN.CityId = {alias}.ProductCityId",      "alias_col": "ProductCityId"},
    "supplierid":       {"name_col": "suppliername",    "join": "LEFT JOIN dbo.suppliermaster_Report _IDJOIN ON _IDJOIN.EmployeeId = {alias}.SupplierId", "alias_col": "SupplierId"},
    "employeeid":       {"name_col": "suppliername",    "join": "LEFT JOIN dbo.suppliermaster_Report _IDJOIN ON _IDJOIN.EmployeeId = {alias}.SupplierId", "alias_col": "SupplierId"},
}


def _try_deterministic_id_fix(sql: str, bad_ids: List[str]) -> Optional[str]:
    """Try to replace raw ID columns with name columns deterministically.

    Returns the fixed SQL, or None if the fix is not possible
    (in which case the caller should fall back to the LLM).
    """
    if not bad_ids or not sql:
        return None

    fixed = sql
    joins_to_add = []
    join_counter = 0

    # Detect main table alias (e.g. "BD" from "FROM dbo.BookingData BD")
    alias_match = re.search(
        r"\bFROM\s+(?:\[?dbo\]?\.)?(?:\[?\w+\]?)\s+(\w+)\s",
        sql, re.IGNORECASE,
    )
    main_alias = alias_match.group(1) if alias_match else ""

    for id_col_lower in bad_ids:
        fix_info = _DETERMINISTIC_ID_FIX_JOINS.get(id_col_lower)
        if not fix_info:
            return None  # Unknown ID column — let LLM handle

        name_col = fix_info["name_col"]
        join_template = fix_info["join"]

        # Replace the ID column in SELECT with the name column
        # Pattern: match the ID col in SELECT clause (with optional alias prefix)
        id_pattern = rf"(?<![.\w])(?:\w+\.)?{re.escape(id_col_lower)}(?!\s*=|\s*<|\s*>)"
        select_from = re.search(r"\bFROM\b", fixed, re.IGNORECASE)
        if not select_from:
            return None

        select_part = fixed[:select_from.start()]
        rest_part = fixed[select_from.start():]

        if join_template is None:
            # Direct column on main table (e.g. ProductName)
            replacement = f"{main_alias}.{name_col}" if main_alias else name_col
        else:
            join_counter += 1
            join_alias = f"_IDJ{join_counter}"
            actual_join = join_template.replace("_IDJOIN", join_alias)
            if main_alias:
                actual_join = actual_join.replace("{alias}", main_alias)
            else:
                actual_join = actual_join.replace("{alias}.", "")
            joins_to_add.append(actual_join)
            replacement = f"{join_alias}.{name_col}"

        select_part = re.sub(id_pattern, replacement, select_part, count=1, flags=re.IGNORECASE)

        # Also fix GROUP BY if it references the ID column
        rest_part = re.sub(
            rf"(?<![.\w])(?:\w+\.)?{re.escape(id_col_lower)}(?!\s*=|\s*<|\s*>)",
            replacement, rest_part, flags=re.IGNORECASE,
        )
        fixed = select_part + rest_part

    # Inject the JOINs before the WHERE clause
    if joins_to_add:
        join_str = "\n".join(joins_to_add)
        where_match = re.search(r"\bWHERE\b", fixed, re.IGNORECASE)
        group_match = re.search(r"\bGROUP\s+BY\b", fixed, re.IGNORECASE)
        order_match = re.search(r"\bORDER\s+BY\b", fixed, re.IGNORECASE)

        insert_pos = None
        for m in [where_match, group_match, order_match]:
            if m:
                insert_pos = m.start()
                break
        if insert_pos is None:
            fixed = fixed.rstrip().rstrip(";") + "\n" + join_str
        else:
            fixed = fixed[:insert_pos] + join_str + "\n" + fixed[insert_pos:]

    # Validate the fix didn't break anything
    valid, _ = validate_sql(fixed)
    return fixed if valid else None


def apply_stored_procedure_guardrails(sql: str, db: Optional[SQLDatabase] = None) -> str:
    """Apply safe domain defaults derived from stored procedure logic.

    Adds BookingStatus exclusion when:
    - queried table has a status column,
    - query does not already filter by status.
    """
    if not sql or db is None:
        return sql
    if re.search(r"\bbookingstatus\b", sql, re.IGNORECASE):
        return sql

    table_entry = _get_table_entry_from_sql(sql, db)
    if not table_entry:
        return sql
    status_filter = _status_filter_expr_for_table(table_entry)
    if not status_filter:
        return sql
    return _inject_condition_into_sql(sql, status_filter)


def expand_fuzzy_search(sql: str, db: Optional[SQLDatabase] = None) -> str:
    sql_lower = sql.lower()
    if "ilike" not in sql_lower and "like" not in sql_lower:
        return sql

    table_name = _extract_from_table_name(sql)
    if not table_name:
        return sql

    valid_columns = None
    if db is not None:
        table_entry = _get_schema_profile(db)["lookup"].get(table_name)
        if table_entry:
            valid_columns = _infer_text_columns(table_entry)
    if not valid_columns:
        return sql
    valid_column_lowers = {c.lower() for c in valid_columns}

    pattern = r"(\w+)\s+(?:I?LIKE)\s+'%([^%]+)%'"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    if matches:
        terms = {}
        for col, term in matches:
            term_lower = term.lower()
            if term_lower not in terms:
                terms[term_lower] = []
            terms[term_lower].append(col.lower())

        for term, cols in terms.items():
            if len(cols) == 1 and cols[0] in valid_column_lowers:
                single_pattern = rf"(\w+)\s+(?:I?LIKE)\s+'%{re.escape(term)}%'"
                conditions = [f"{col} LIKE '%{term}%'" for col in valid_columns]
                replacement = "(" + " OR ".join(conditions) + ")"
                sql = re.sub(single_pattern, replacement, sql, count=1, flags=re.IGNORECASE)
    return sql


def _clean_sql_response(sql: str) -> str:
    if not sql:
        return ""
    cleaned = sql.strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1].strip()

    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    cleaned = re.split(r"(?im)^\s*Notes\s*:", cleaned, maxsplit=1)[0].strip()

    start_match = re.search(r"(?is)\b(with|select)\b", cleaned)
    if start_match:
        cleaned = cleaned[start_match.start():].strip()

    cleaned = cleaned.replace("```", "").strip()
    return cleaned


# --- "De-clevering" guardrails ---

def _estimate_query_complexity(question: str) -> str:
    """Classify query complexity for routing and timeout decisions.

    Returns:
        'deterministic' - handled by hard-coded SQL builders (no LLM needed)
        'simple_llm'    - straightforward query, skip full retry pipeline on success
        'complex_llm'   - needs full retry + validation pipeline
    """
    q = (question or "").lower()
    words = set(re.findall(r"[a-z_]+", q))

    # Already handled deterministically
    if is_business_overview_question(q):
        return "deterministic"
    if _parse_topn_request(question) is not None:
        return "deterministic"

    # Complex indicators: need full LLM pipeline with retries
    complex_indicators = {
        "growth", "compared", "versus", "trend", "yoy", "ytd", "pytd",
        "correlation", "year over year", "month over month", "increase",
        "decrease", "change", "previous year", "last year vs",
    }
    if any(t in q for t in complex_indicators):
        return "complex_llm"

    # Multi-dimension or join-heavy
    dimension_words = {"agent", "supplier", "country", "city", "hotel", "chain", "nationality"}
    dim_count = len(words & dimension_words)
    if dim_count >= 2:
        return "complex_llm"

    # Simple: single-dimension aggregation, short questions
    simple_patterns = [
        r"total\s+(sales|revenue|profit|bookings|cost)",
        r"(how many|count)\s+(bookings|agents|suppliers|hotels|countries|cities)",
    ]
    if any(re.search(p, q) for p in simple_patterns) and len(q.split()) <= 10:
        return "simple_llm"

    # Default: simple enough unless proven otherwise
    return "simple_llm" if len(q.split()) <= 12 else "complex_llm"


def is_business_overview_question(question: str) -> bool:
    """Return True if the user is asking for a broad business snapshot.

    This is intentionally heuristic (not LLM-based) to keep the system
    predictable and avoid multi-view UNION queries for vague prompts.
    """
    if not question:
        return False
    q = " ".join(question.lower().strip().split())
    if re.search(r"\b(top|bottom)\b", q):
        return False
    if any(phrase in q for phrase in BUSINESS_OVERVIEW_PHRASES):
        # If the question already asks for a breakdown/trend/comparison, it's not a single-row overview.
        if any(neg in q for neg in BUSINESS_OVERVIEW_NEGATORS):
            return False
        return True

    # Catch short vague prompts like: "business overview", "overall performance", "how are we doing"
    # while avoiding queries that mention a specific dimension.
    short_vague = (
        len(q.split()) <= 6
        and any(w in q for w in ["overview", "performance", "summary", "kpi", "kpis", "business", "how are we doing", "how we are doing"])
    )
    if short_vague and not any(neg in q for neg in BUSINESS_OVERVIEW_NEGATORS):
        # If it explicitly mentions a dimension + "by"/"wise", treat as grouped query, not overview.
        if (" by " in q or "wise" in q) and any(dim in q for dim in ["agent", "country", "city", "supplier", "hotel", "chain", "nationality"]):
            return False
        return True

    return False


def _quote_sql_identifier(name: str) -> str:
    parts = [p for p in str(name).split(".") if p]
    return ".".join(f"[{p}]" for p in parts)


def _resolve_business_overview_source(db: SQLDatabase) -> Optional[Dict[str, str]]:
    """Find the best available table/view to compute business overview KPIs.

    This avoids hardcoding `country_level_view` for databases where that view
    does not exist (common in MSSQL environments).
    """
    if db is None:
        return None

    try:
        usable = list(db.get_usable_table_names())
    except Exception:
        return None

    if not usable:
        return None

    lower_to_actual = {t.lower(): t for t in usable}

    # Priority order: canonical aggregated view first, then common MSSQL tables.
    candidates = [
        {
            "table": "country_level_view",
            "required_cols": {"booking_date", "total_sales", "total_profit", "total_booking"},
            "date_col": "booking_date",
            "revenue_expr": "SUM(COALESCE(total_sales, 0))",
            "profit_expr": "SUM(COALESCE(total_profit, 0))",
            "bookings_expr": "SUM(COALESCE(total_booking, 0))",
            "status_filter_expr": None,
        },
        {
            "table": "BookingData",
            "required_cols": {"createddate", "agentbuyingprice", "companybuyingprice"},
            "date_col": "CreatedDate",
            "revenue_expr": "SUM(COALESCE(AgentBuyingPrice, 0))",
            "profit_expr": "SUM(COALESCE(AgentBuyingPrice, 0) - COALESCE(CompanyBuyingPrice, 0))",
            "bookings_expr": "COUNT(DISTINCT PNRNo)",
            "status_filter_expr": "[BookingStatus] NOT IN ('Cancelled', 'Not Confirmed', 'On Request')",
        },
        {
            "table": "Agent_SOA_Report",
            "required_cols": {"bookingdate", "totalamount"},
            "date_col": "BookingDate",
            "revenue_expr": "SUM(COALESCE(TotalAmount, 0))",
            "profit_expr": "0",
            "bookings_expr": "COUNT(DISTINCT BookingID)",
            "status_filter_expr": None,
        },
        {
            "table": "Supplier_SOA_Report",
            "required_cols": {"bookingdate", "totalamount"},
            "date_col": "BookingDate",
            "revenue_expr": "SUM(COALESCE(TotalAmount, 0))",
            "profit_expr": "0",
            "bookings_expr": "COUNT(DISTINCT BookingId)",
            "status_filter_expr": None,
        },
        {
            "table": "GetTotalBookingCount",
            "required_cols": {"bookingdate", "totalbooking"},
            "date_col": "BookingDate",
            "revenue_expr": "0",
            "profit_expr": "0",
            "bookings_expr": "SUM(COALESCE(TotalBooking, 0))",
            "status_filter_expr": None,
        },
    ]

    # Use the bulk-loaded schema profile (fast) instead of per-table inspector calls.
    profile = _get_schema_profile(db)

    for cand in candidates:
        actual = lower_to_actual.get(cand["table"].lower())
        if not actual:
            continue
        entry = profile["lookup"].get(actual.lower())
        if not entry:
            continue
        cols = set(entry["columns_map"].keys())
        if cand["required_cols"].issubset(cols):
            resolved = dict(cand)
            resolved["table_actual"] = actual
            return resolved

    # Generic fallback: pick the best available table with a date column and usable metrics.
    best = None
    best_score = -1
    for table_entry in profile.get("tables", []):
        date_col = _detect_date_column(table_entry)
        if not date_col:
            continue
        metric_exprs = _metric_exprs_for_table(table_entry)
        if not metric_exprs["has_group_metric"]:
            continue

        score = 0
        if metric_exprs["has_revenue"]:
            score += 10
        if metric_exprs["has_profit"]:
            score += 4
        if "booking" in table_entry["table_lower"]:
            score += 6
        if "report" in table_entry["table_lower"]:
            score += 2

        if score > best_score:
            best_score = score
            best = {
                "table_actual": table_entry["table"],
                "date_col": date_col,
                "revenue_expr": metric_exprs["revenue_expr"],
                "profit_expr": metric_exprs["profit_expr"],
                "bookings_expr": metric_exprs["bookings_expr"],
                "status_filter_expr": _status_filter_expr_for_table(table_entry),
            }

    if best:
        return best

    # Hard fallback: if BookingData exists but schema reflection is sparse,
    # still force a deterministic overview query on BookingData.
    if _table_exists_fast(db, "BookingData"):
        return {
            "table_actual": "BookingData",
            "date_col": "CreatedDate",
            "revenue_expr": "SUM(COALESCE(AgentBuyingPrice, 0))",
            "profit_expr": "SUM(COALESCE(AgentBuyingPrice, 0) - COALESCE(CompanyBuyingPrice, 0))",
            "bookings_expr": "COUNT(DISTINCT PNRNo)",
            "status_filter_expr": "[BookingStatus] NOT IN ('Cancelled', 'Not Confirmed', 'On Request')",
        }

    return None


def build_business_overview_sql_from_source(source: Dict[str, str], date_scope: str = "latest_year") -> str:
    """Build single-row KPI overview SQL using a resolved source table/view."""
    table_name = _quote_sql_identifier(source["table_actual"])
    date_col = _quote_sql_identifier(source["date_col"])
    revenue_expr = source["revenue_expr"]
    profit_expr = source["profit_expr"]
    bookings_expr = source["bookings_expr"]
    status_filter_expr = source.get("status_filter_expr")
    r = _get_ist_date_ranges()
    this_month_start = datetime.fromisoformat(r["this_month_start"]).date()
    this_year_start = datetime.fromisoformat(r["this_year_start"]).date()

    if date_scope == "this_month":
        date_filter = f"{date_col} >= '{r['this_month_start']}' AND {date_col} < '{r['this_month_end']}'"
    elif date_scope == "last_month":
        last_month_end = this_month_start
        last_month_start = (last_month_end.replace(day=1) - timedelta(days=1)).replace(day=1)
        date_filter = f"{date_col} >= '{last_month_start.isoformat()}' AND {date_col} < '{last_month_end.isoformat()}'"
    elif date_scope == "latest_year":
        date_filter = f"{date_col} >= '{r['this_year_start']}' AND {date_col} < '{r['this_year_end']}'"
    elif date_scope == "last_year":
        last_year_start = this_year_start.replace(year=this_year_start.year - 1)
        date_filter = f"{date_col} >= '{last_year_start.isoformat()}' AND {date_col} < '{this_year_start.isoformat()}'"
    else:
        date_filter = f"{date_col} >= '{r['this_year_start']}' AND {date_col} < '{r['this_year_end']}'"

    where_clauses = [date_filter]
    if status_filter_expr:
        where_clauses.append(status_filter_expr)

    return (
        "SELECT\n"
        f"  COALESCE({revenue_expr}, 0) AS revenue,\n"
        f"  COALESCE({profit_expr}, 0) AS profit,\n"
        f"  COALESCE({bookings_expr}, 0) AS bookings,\n"
        "  CASE\n"
        f"    WHEN COALESCE({bookings_expr}, 0) = 0 THEN 0\n"
        f"    ELSE COALESCE({revenue_expr}, 0) / NULLIF({bookings_expr}, 0)\n"
        "  END AS avg_booking_value\n"
        f"FROM {table_name}\n"
        f"WHERE {' AND '.join(where_clauses)};"
    )


def _parse_topn_request(question: str) -> Optional[Dict[str, Any]]:
    """Extract top/bottom-N intent details from question.

    Returns None (skips guardrail) when the question contains extra filters
    like 'status is confirmed', 'where country = X', etc.  These are too
    complex for the deterministic SQL builder — let the LLM handle them.
    """
    if not question:
        return None
    q = " ".join(question.lower().strip().split())
    ranking_tokens = re.findall(r"[a-z]+", q)
    ranking_words = set(ranking_tokens)
    has_top_bottom = bool(re.search(r"\b(top|bottom)\b", q))
    has_max_style = bool(ranking_words & {"highest", "maximum", "max", "most"})
    has_min_style = bool(ranking_words & {"lowest", "minimum", "min", "least"})
    if not (has_top_bottom or has_max_style or has_min_style):
        return None

    # If the question has value-based filters OR requires multi-period / growth
    # calculations, let the LLM generate the SQL — the deterministic builder
    # cannot produce YoY CTEs, growth %, or comparison queries.
    filter_indicators = [
        "where", "status", "confirmed", "cancelled", "canceled", "pending",
        "failed", "refund", "equal", "greater", "less than", "between",
        "not ", "exclude", "only ", "filter",
        # YoY / growth / comparison — needs CTE with two date ranges
        "growth", "compared to", "vs last", "versus last", "year over year",
        "yoy", "ytd", "pytd", "last year", "previous year", "prior year",
        "increase", "decrease", "change", "trend",
    ]
    if any(indicator in q for indicator in filter_indicators):
        return None

    match = re.search(r"\b(top|bottom)\s+(\d+)\b", q)
    if match:
        direction = match.group(1)
        limit = int(match.group(2))
    elif has_top_bottom:
        direction = "bottom" if "bottom" in q else "top"
        limit = 10
    elif has_min_style:
        direction = "bottom"
        limit = 1
    else:
        direction = "top"
        limit = 1

    # If limit is still the default (1), try to extract N from natural language:
    # "give me 10", "show me 5", "show 10", "get 20", "find top 10", "I want 15"
    # Also catches a plain number appended to the question: "...? give me 10"
    if limit == 1:
        explicit_n = re.search(
            r"\b(?:give\s+me|show\s+(?:me\s+)?|get\s+|find\s+|want\s+|need\s+|list\s+(?:me\s+)?|fetch\s+)(\d+)\b",
            q,
        )
        if explicit_n:
            limit = max(1, min(int(explicit_n.group(1)), 100))
        else:
            # Fallback: any bare integer in the question that is not a year (1900-2100)
            all_nums = [
                int(n) for n in re.findall(r"\b(\d+)\b", q)
                if not (1900 <= int(n) <= 2100)
            ]
            if all_nums:
                limit = max(1, min(max(all_nums), 100))

    if any(t in q for t in ["cost wise", "cost", "expense", "spend", "buying cost"]):
        metric_alias = "cost"
    elif "profit" in q:
        metric_alias = "profit"
    elif "booking" in q:
        metric_alias = "bookings"
    else:
        # Default metric for ranking questions
        metric_alias = "revenue"

    if "by revenue" in q or "by sales" in q:
        metric_alias = "revenue"
    elif "by cost" in q or "by expense" in q or "by spend" in q:
        metric_alias = "cost"
    elif "by profit" in q:
        metric_alias = "profit"
    elif "by booking" in q or "by bookings" in q:
        metric_alias = "bookings"

    return {"direction": direction, "limit": limit, "metric_alias": metric_alias, "normalized_question": q}


def _is_ranking_question(question: str) -> bool:
    if not question:
        return False
    q = question.lower()
    ranking_terms = {"top", "bottom", "highest", "maximum", "max", "lowest", "minimum", "most", "least"}
    metric_terms = {"booking", "bookings", "revenue", "sales", "profit", "cost", "expense", "spend"}
    words = set(re.findall(r"[a-z_]+", q))
    return bool(words & ranking_terms) and bool(words & metric_terms)


def _ranking_requests_checkin_checkout(question: str) -> bool:
    q = (question or "").lower()
    checkin_terms = {"checkin", "check-in", "check in", "checkout", "check-out", "check out", "stay date", "stay-date"}
    return any(t in q for t in checkin_terms)


def _ranking_is_single_day(question: str) -> bool:
    q = (question or "").lower()
    if any(t in q for t in ["today", "yesterday", "on ", "for date", "specific date"]):
        return True
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", q):
        return True
    return False


def _extract_group_by_clause(sql: str) -> str:
    if not sql:
        return ""
    match = re.search(
        r"\bGROUP\s+BY\s+(.*?)(?:\bORDER\s+BY\b|\bHAVING\b|;|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    return " ".join(match.group(1).split()).strip()


def _ranking_expected_entity_tokens(question: str) -> set:
    words = set(re.findall(r"[a-z_]+", (question or "").lower()))
    expected = set()
    for keywords, tokens in RANKING_ENTITY_TOKENS:
        if words & keywords:
            expected.update(tokens)
    return expected


def validate_ranking_sql_shape(question: str, sql: str) -> Tuple[bool, str]:
    """Validate that ranking SQL is aggregated at entity level (not split by extra dates)."""
    if not _is_ranking_question(question):
        return True, ""
    clause = _extract_group_by_clause(sql)
    if not clause:
        return True, ""

    clause_lower = clause.lower()
    asks_stay_dates = _ranking_requests_checkin_checkout(question)
    is_single_day = _ranking_is_single_day(question)

    if not asks_stay_dates and any(t in clause_lower for t in ["checkin", "checkout"]):
        return False, "Ranking query groups by check-in/check-out even though the question did not ask stay-date analysis."

    if is_single_day and any(t in clause_lower for t in ["booking_date", "bookingdate", "createddate", "created_date"]):
        return False, "Single-day ranking should filter date, not group by booking/created date."

    expected_tokens = _ranking_expected_entity_tokens(question)
    if expected_tokens and not any(tok in clause_lower for tok in expected_tokens):
        return False, "Ranking query GROUP BY does not include the requested business entity."

    # Heuristic: ranking should usually have compact grouping dimensions.
    if clause.count(",") >= 3:
        return False, "Ranking query has too many GROUP BY dimensions and may be over-segmented."

    return True, ""


def _retry_ranking_shape_if_needed(
    question: str,
    sql: str,
    llm,
    full_schema: str,
    stored_procedure_guidance: str,
    sql_dialect: str = DEFAULT_SQL_DIALECT,
    enable_nolock: bool = False,
) -> str:
    is_ok, reason = validate_ranking_sql_shape(question, sql)
    if is_ok:
        return sql
    if llm is None:
        return sql

    prompt = RANKING_RESHAPE_PROMPT_TEMPLATE.format(
        question=question,
        full_schema=full_schema or "",
        stored_procedure_guidance=stored_procedure_guidance or "",
        current_sql=sql,
        violation_reason=reason,
        dialect_label=_dialect_label(sql_dialect),
        enable_nolock=str(bool(enable_nolock)).lower(),
        relative_date_reference=_build_relative_date_reference(),
    )
    try:
        resp = llm.invoke(prompt)
        candidate = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        candidate = _clean_sql_response(candidate)
        candidate = fix_common_sql_errors(candidate, dialect=sql_dialect)
        valid, _ = validate_sql(candidate)
        if not valid:
            return sql
        ok2, _ = validate_ranking_sql_shape(question, candidate)
        return candidate if ok2 else sql
    except Exception:
        return sql


def _topn_dimension_candidates(question_lower: str) -> Tuple[List[str], int]:
    """Return (ordered_candidates, primary_count).

    primary_count is the number of candidates derived from the user's
    question keywords (e.g. "agent" → agentname, agentcode).  The rest
    are generic defaults.  Callers can use primary_count to decide
    whether the chosen dimension actually matches the user's intent.
    """
    words = set(re.findall(r"[a-z_]+", question_lower))
    ordered = []
    seen = set()

    for keyword_set, candidates in TOPN_DIMENSION_HINTS:
        if words & keyword_set:
            for col in candidates:
                low = col.lower()
                if low not in seen:
                    ordered.append(low)
                    seen.add(low)

    primary_count = len(ordered)

    for col in TOPN_DEFAULT_DIMENSION_CANDIDATES:
        low = col.lower()
        if low not in seen:
            ordered.append(low)
            seen.add(low)

    return ordered, primary_count


def _resolve_topn_source(question_lower: str, db: SQLDatabase) -> Optional[Dict[str, Any]]:
    profile = _get_schema_profile(db)
    if not profile["tables"]:
        return None

    words = set(re.findall(r"[a-z_]+", question_lower))
    dimension_candidates, primary_count = _topn_dimension_candidates(question_lower)
    best = None
    best_score = -1
    best_is_primary = False

    for table_entry in profile["tables"]:
        metric_exprs = _metric_exprs_for_table(table_entry)
        if not metric_exprs["has_group_metric"]:
            continue

        date_col = _detect_date_column(table_entry)
        table_lower = table_entry["table_lower"]

        for idx, dim_col_lower in enumerate(dimension_candidates):
            actual_dim = table_entry["columns_map"].get(dim_col_lower)
            if not actual_dim:
                continue

            is_primary = idx < primary_count

            score = max(1, 100 - idx)
            if "booking" in words and "booking" in table_lower:
                score += 12
            if "hotel" in words and ("hotel" in table_lower or dim_col_lower.startswith("hotel")):
                score += 8
            if "agent" in words and "agent" in table_lower:
                score += 8
            if "supplier" in words and "supplier" in table_lower:
                score += 8
            if metric_exprs["has_revenue"]:
                score += 10
            if metric_exprs["has_profit"]:
                score += 4
            if date_col:
                score += 3

            if score > best_score:
                best_score = score
                best_is_primary = is_primary
                best = {
                    "table": table_entry["table"],
                    "dimension_col": actual_dim,
                    "date_col": date_col,
                    "columns_map": dict(table_entry.get("columns_map", {})),
                    "status_filter_expr": _status_filter_expr_for_table(table_entry),
                    **metric_exprs,
                }

    # If the user asked for a specific dimension (e.g. "agents") but no table
    # has that column, best_is_primary will be False — the guardrail picked a
    # wrong default like "productname".  Return None so the LLM can generate
    # the correct JOIN query instead.
    if primary_count > 0 and not best_is_primary:
        return None

    return best


# Dimensions that live in a lookup table — must JOIN to get the human-readable name.
# Each entry: id_col (on BookingData), name_col, join_table, join_alias, join_key
_TOPN_JOIN_DIMENSIONS: Dict[str, Dict[str, str]] = {
    "supplier": {
        "id_col":      "SupplierId",
        "name_col":    "suppliername",
        "join_table":  "dbo.suppliermaster_Report",
        "join_alias":  "DIM",
        "join_key":    "DIM.EmployeeId = BD.SupplierId",
    },
    "agent": {
        "id_col":      "AgentId",
        "name_col":    "AgentName",
        "join_table":  "dbo.AgentMaster_V1",
        "join_alias":  "DIM",
        "join_key":    "DIM.AgentId = BD.AgentId",
    },
    "country": {
        "id_col":      "ProductCountryid",
        "name_col":    "Country",
        "join_table":  "dbo.Master_Country",
        "join_alias":  "DIM",
        "join_key":    "DIM.CountryID = BD.ProductCountryid",
    },
    "city": {
        "id_col":      "ProductCityId",
        "name_col":    "City",
        "join_table":  "dbo.Master_City",
        "join_alias":  "DIM",
        "join_key":    "DIM.CityId = BD.ProductCityId",
    },
}


def _resolve_topn_known_source(question_lower: str, db: SQLDatabase) -> Optional[Dict[str, Any]]:
    """Fallback source when schema reflection is unavailable/empty.

    Uses dbo.BookingData so top-N queries stay deterministic without LLM.
    Dimensions like supplier/agent/country/city resolve to name columns via JOIN.
    """
    preferred_tables = ["BookingData", "bookingdata"]
    table_name = None
    for candidate in preferred_tables:
        if _table_exists_fast(db, candidate):
            table_name = candidate
            break
    if not table_name:
        return None

    words = set(re.findall(r"[a-z_]+", question_lower))

    # Detect which JOIN dimension applies (in priority order)
    join_info: Optional[Dict[str, str]] = None
    dim: str = "ProductName"          # default — direct column on BookingData
    if words & {"supplier", "suppliers"}:
        join_info = _TOPN_JOIN_DIMENSIONS["supplier"]
        dim = join_info["name_col"]
    elif words & {"agent", "agents"}:
        join_info = _TOPN_JOIN_DIMENSIONS["agent"]
        dim = join_info["name_col"]
    elif words & {"country", "countries"}:
        join_info = _TOPN_JOIN_DIMENSIONS["country"]
        dim = join_info["name_col"]
    elif words & {"city", "cities"}:
        join_info = _TOPN_JOIN_DIMENSIONS["city"]
        dim = join_info["name_col"]
    elif words & {"chain", "chains"}:
        dim = "ProductName"   # chains live in Hotelchain; let LLM handle the join
    # else: ProductName (hotel/product) — direct column, no join needed

    base = {
        "table": table_name,
        "dimension_col": dim,
        "date_col": "CreatedDate",
        "columns_map": {
            "createddate": "CreatedDate",
            "checkindate": "CheckInDate",
            "checkoutdate": "CheckOutDate",
            "bookingstatus": "BookingStatus",
        },
        "status_filter_expr": "BD.[BookingStatus] NOT IN ('Cancelled', 'Not Confirmed', 'On Request')",
        "revenue_expr": "SUM(COALESCE(BD.[AgentBuyingPrice], 0))",
        "profit_expr": "SUM(COALESCE(BD.[AgentBuyingPrice], 0) - COALESCE(BD.[CompanyBuyingPrice], 0))",
        "cost_expr": "SUM(COALESCE(BD.[CompanyBuyingPrice], 0))",
        "bookings_expr": "COUNT(DISTINCT BD.[PNRNo])",
        "has_revenue": True,
        "has_profit": True,
        "has_cost": True,
        "has_group_metric": True,
    }
    if join_info:
        base["join_info"] = join_info
    return base


def _build_date_where_clauses(q: str, date_col_expr: str) -> List[str]:
    """Build SARGable date filter clauses from natural language question.

    Supports: today, yesterday, this/last week, this/last month, this/last year,
    last N days, exact dates (YYYY-MM-DD).
    """
    clauses: List[str] = []
    if not date_col_expr:
        return clauses
    r = _get_ist_date_ranges()

    if "this month" in q:
        clauses.append(f"{date_col_expr} >= '{r['this_month_start']}'")
        clauses.append(f"{date_col_expr} < '{r['this_month_end']}'")
    elif "last month" in q:
        this_month_start = datetime.fromisoformat(r["this_month_start"]).date()
        last_month_end = this_month_start
        last_month_start = (last_month_end.replace(day=1) - timedelta(days=1)).replace(day=1)
        clauses.append(f"{date_col_expr} >= '{last_month_start.isoformat()}'")
        clauses.append(f"{date_col_expr} < '{last_month_end.isoformat()}'")
    elif "last week" in q:
        clauses.append(f"{date_col_expr} >= '{r['last_week_start']}'")
        clauses.append(f"{date_col_expr} < '{r['last_week_end']}'")
    elif "this week" in q:
        clauses.append(f"{date_col_expr} >= '{r['this_week_start']}'")
        clauses.append(f"{date_col_expr} < '{r['this_week_end']}'")
    elif "yesterday" in q:
        clauses.append(f"{date_col_expr} >= '{r['yesterday_start']}'")
        clauses.append(f"{date_col_expr} < '{r['yesterday_end']}'")
    elif "today" in q:
        clauses.append(f"{date_col_expr} >= '{r['today_start']}'")
        clauses.append(f"{date_col_expr} < '{r['today_end']}'")
    elif "this year" in q:
        clauses.append(f"{date_col_expr} >= '{r['this_year_start']}'")
        clauses.append(f"{date_col_expr} < '{r['this_year_end']}'")
    elif "last year" in q:
        this_year_start = datetime.fromisoformat(r["this_year_start"]).date()
        last_year_start = this_year_start.replace(year=this_year_start.year - 1)
        clauses.append(f"{date_col_expr} >= '{last_year_start.isoformat()}'")
        clauses.append(f"{date_col_expr} < '{this_year_start.isoformat()}'")
    else:
        exact_date = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
        if exact_date:
            exact_start = datetime.fromisoformat(exact_date.group(1)).date()
            exact_end = exact_start + timedelta(days=1)
            clauses.append(f"{date_col_expr} >= '{exact_start.isoformat()}'")
            clauses.append(f"{date_col_expr} < '{exact_end.isoformat()}'")
        else:
            last_n_days = re.search(r"last\s+(\d+)\s+days", q)
            if last_n_days:
                n = max(1, min(int(last_n_days.group(1)), 365))
                today_start = datetime.fromisoformat(r["today_start"]).date()
                start_date = today_start - timedelta(days=n)
                end_date = today_start + timedelta(days=1)
                clauses.append(f"{date_col_expr} >= '{start_date.isoformat()}'")
                clauses.append(f"{date_col_expr} < '{end_date.isoformat()}'")
    return clauses


def _resolve_date_col_expr(q: str, source: Dict[str, Any]) -> str:
    """Resolve the correct date column based on user intent.

    travel date / check-in / stay → CheckInDate
    checkout / departure          → CheckOutDate
    booking date / default        → CreatedDate
    """
    col_map_lower = source.get("columns_map", {})
    has_checkin = "checkindate" in col_map_lower or "checkin_date" in col_map_lower
    has_checkout = "checkoutdate" in col_map_lower or "checkout_date" in col_map_lower

    travel_date_terms = ["travel date", "travel", "check-in", "checkin", "stay date", "stay"]
    checkout_terms = ["checkout", "check-out", "departure", "travel end"]

    if any(t in q for t in travel_date_terms) and has_checkin:
        return _quote_sql_identifier(
            col_map_lower.get("checkindate") or col_map_lower.get("checkin_date") or "CheckInDate"
        )
    if any(t in q for t in checkout_terms) and has_checkout:
        return _quote_sql_identifier(
            col_map_lower.get("checkoutdate") or col_map_lower.get("checkout_date") or "CheckOutDate"
        )
    date_col = source.get("date_col")
    return _quote_sql_identifier(date_col) if date_col else "[CreatedDate]"


def _build_metric_select(metric_alias: str, source: Dict[str, Any], bd_prefix: str = "BD.") -> Tuple[str, str]:
    """Return (select_expr, user_friendly_alias) for the requested metric ONLY.

    Follows the principle: only return what the user asked for.
    """
    # User-friendly aliases for clean result column names
    if metric_alias == "cost":
        return f"SUM({bd_prefix}[CompanyBuyingPrice])", "Total Cost"
    elif metric_alias == "revenue":
        return f"SUM({bd_prefix}[AgentBuyingPrice])", "Total Revenue"
    elif metric_alias == "profit":
        return f"SUM({bd_prefix}[AgentBuyingPrice] - {bd_prefix}[CompanyBuyingPrice])", "Total Profit"
    elif metric_alias == "bookings":
        return f"COUNT(DISTINCT {bd_prefix}[PNRNo])", "Total Bookings"
    elif metric_alias == "avg_booking_value":
        return (
            f"SUM({bd_prefix}[AgentBuyingPrice]) / NULLIF(COUNT(DISTINCT {bd_prefix}[PNRNo]), 0)",
            "Avg Booking Value",
        )
    # Fallback: revenue
    return f"SUM({bd_prefix}[AgentBuyingPrice])", "Total Revenue"


# CTE definitions for dimension lookup tables (deduplication via DISTINCT)
_CTE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "agent": {
        "cte_name": "AM",
        "cte_sql": "SELECT AgentId, AgentCode, AgentName, AgentCountry, AgentCity FROM [dbo].[AgentMaster_V1] WITH (NOLOCK)",
        "join_on": "AM.AgentId = BD.AgentId",
        "name_col": "AgentName",
        "friendly_name": "Agent Name",
    },
    "supplier": {
        "cte_name": "SM",
        "cte_sql": "SELECT DISTINCT EmployeeId, SupplierName FROM [dbo].[suppliermaster_Report] WITH (NOLOCK)",
        "join_on": "SM.EmployeeId = BD.SupplierId",
        "name_col": "SupplierName",
        "friendly_name": "Supplier Name",
    },
    "country": {
        "cte_name": "MC",
        "cte_sql": "SELECT DISTINCT CountryID, Country FROM [dbo].[Master_Country] WITH (NOLOCK)",
        "join_on": "MC.CountryID = BD.ProductCountryid",
        "name_col": "Country",
        "friendly_name": "Country",
    },
    "city": {
        "cte_name": "MCI",
        "cte_sql": "SELECT DISTINCT CityId, City FROM [dbo].[Master_City] WITH (NOLOCK)",
        "join_on": "MCI.CityId = BD.ProductCityId",
        "name_col": "City",
        "friendly_name": "City",
    },
}


def _detect_dimension(q: str) -> str:
    """Detect the dimension from the normalized question text."""
    words = set(re.findall(r"[a-z_]+", q))
    if words & {"supplier", "suppliers"}:
        return "supplier"
    elif words & {"agent", "agents"}:
        return "agent"
    elif words & {"country", "countries"}:
        return "country"
    elif words & {"city", "cities"}:
        return "city"
    elif words & {"chain", "chains"}:
        return "chain"
    elif words & {"nationality"}:
        return "nationality"
    return "hotel"  # default: ProductName on BookingData


def build_topn_fallback_sql(question: str, db: SQLDatabase) -> Optional[str]:
    """Build deterministic, optimized SQL for top/bottom-N ranking queries.

    Uses CTE pattern with DISTINCT for lookup table dedup (matching stored_procedure.txt).
    Only returns the metric the user asked for — no over-fetching.
    Uses NOT EXISTS for status filter on large datasets.
    """
    req = _parse_topn_request(question)
    if req is None:
        return None

    q = req["normalized_question"]
    source = _resolve_topn_source(q, db)
    if not source:
        source = _resolve_topn_known_source(q, db)
    if not source:
        return None

    metric_alias = req["metric_alias"]
    direction = "DESC" if req["direction"] == "top" else "ASC"
    limit = max(1, min(int(req["limit"]), 100))

    # Resolve which date column to use based on user intent
    date_col_expr = _resolve_date_col_expr(q, source)

    # Build date WHERE clauses (handles last week, this week, etc.)
    date_clauses = _build_date_where_clauses(q, f"BD.{date_col_expr.strip('[]')}")

    # Status filter using NOT EXISTS pattern is not needed here;
    # the simple NOT IN on a small constant list is fine and clearer.
    # Use the BookingStatus filter.
    where_clauses = [
        "BD.[BookingStatus] NOT IN ('Cancelled', 'Not Confirmed', 'On Request')"
    ]
    where_clauses.extend(date_clauses)
    where_str = "WHERE " + "\n  AND ".join(where_clauses)

    # Get ONLY the metric the user asked for
    metric_expr, metric_friendly = _build_metric_select(metric_alias, source)

    # Detect dimension
    dimension = _detect_dimension(q)
    cte_def = _CTE_DEFINITIONS.get(dimension)

    if cte_def:
        # ── CTE-based query (agent / supplier / country / city) ──
        # Step 1: CTE deduplicates the lookup table (AgentMaster_V1 has duplicates)
        # Step 2: Main CTE aggregates with JOIN
        # Step 3: Final SELECT does TOP N + ORDER BY
        cte_name = cte_def["cte_name"]
        cte_sql = cte_def["cte_sql"]
        join_on = cte_def["join_on"]
        name_col = cte_def["name_col"]
        friendly_name = cte_def["friendly_name"]

        return (
            f"WITH {cte_name} AS (\n"
            f"  {cte_sql}\n"
            f"),\n"
            f"main AS (\n"
            f"  SELECT {cte_name}.{name_col} AS [{friendly_name}],\n"
            f"    {metric_expr} AS [{metric_friendly}]\n"
            f"  FROM dbo.BookingData BD WITH (NOLOCK)\n"
            f"  LEFT JOIN {cte_name}\n"
            f"    ON {join_on}\n"
            f"  {where_str}\n"
            f"  GROUP BY {cte_name}.{name_col}\n"
            f")\n"
            f"SELECT TOP {limit} [{friendly_name}], [{metric_friendly}]\n"
            f"FROM main\n"
            f"ORDER BY [{metric_friendly}] {direction};"
        )

    # ── Simple query (hotel / product / nationality — direct column on BookingData) ──
    if dimension == "nationality":
        dim_col = "BD.[ClientNatinality]"
        friendly_name = "Nationality"
    elif dimension == "chain":
        dim_col = "BD.[ProductName]"
        friendly_name = "Hotel Name"
    else:
        dim_col = "BD.[ProductName]"
        friendly_name = "Hotel Name"

    return (
        f"WITH main AS (\n"
        f"  SELECT {dim_col} AS [{friendly_name}],\n"
        f"    {metric_expr} AS [{metric_friendly}]\n"
        f"  FROM dbo.BookingData BD WITH (NOLOCK)\n"
        f"  {where_str}\n"
        f"  GROUP BY {dim_col}\n"
        f")\n"
        f"SELECT TOP {limit} [{friendly_name}], [{metric_friendly}]\n"
        f"FROM main\n"
        f"ORDER BY [{metric_friendly}] {direction};"
    )


def should_replace_with_topn_fallback(question: str, sql: str) -> bool:
    """Detect malformed top/bottom-N SQL (single-row aggregate, missing ranking shape)."""
    if not sql:
        return False
    req = _parse_topn_request(question)
    if req is None:
        return False

    sql_lower = " ".join(sql.lower().split())
    has_group_by = " group by " in f" {sql_lower} "
    has_limit = " top " in f" {sql_lower} " or " limit " in f" {sql_lower} "
    has_aggregate = any(fn in sql_lower for fn in ["sum(", "avg(", "count(", "min(", "max("])

    # For ranking queries we need grouped, limited output.
    if not has_group_by:
        return True
    if not has_limit:
        return True
    if has_aggregate and has_group_by:
        return False
    return False


# --- Caching ---

def extract_key_values(text: str) -> set:
    values = set()
    values.update(re.findall(r"\b(20\d{2})\b", text))
    values.update(re.findall(r"\b(\d+)\b", text))
    values.update(re.findall(r"'([^']+)'", text))
    values.update(re.findall(r'"([^"]+)"', text))
    return values


def _snapshot_query_cache(query_cache: Any) -> List[Any]:
    if query_cache is None:
        return []
    if hasattr(query_cache, "snapshot"):
        try:
            return list(query_cache.snapshot())
        except Exception:
            return []
    if isinstance(query_cache, list):
        return list(query_cache)
    return []


def _append_query_cache(query_cache: Any, item: Any) -> None:
    if query_cache is None:
        return
    if hasattr(query_cache, "put"):
        key = item.get("cache_key") if isinstance(item, dict) else None
        if key:
            query_cache.put(key, item)
            return
    if hasattr(query_cache, "append"):
        query_cache.append(item)
        return
    if isinstance(query_cache, list):
        if len(query_cache) >= MAX_CACHE_SIZE:
            query_cache.pop(0)
        query_cache.append(item)


def _query_cache_db_identity(db: Optional[SQLDatabase]) -> str:
    if db is None:
        return "unknown"
    try:
        url = db._engine.url
        return f"{(url.host or '').lower()}:{url.port or ''}/{(url.database or '').lower()}"
    except Exception:
        return "unknown"


def _normalize_cache_question(question: str) -> str:
    return " ".join((question or "").strip().lower().split())


def _make_query_cache_key(question: str, db: Optional[SQLDatabase]) -> str:
    payload = f"{_query_cache_db_identity(db)}|{_normalize_cache_question(question)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_entry_expired(entry: Dict[str, Any], now_ts: float) -> bool:
    ts = float(entry.get("timestamp") or 0.0)
    ttl_seconds = int(entry.get("ttl_seconds") or QUERY_RESULT_CACHE_TTL_SECONDS)
    return ts <= 0.0 or (now_ts - ts) > ttl_seconds


def is_time_sensitive(question: str = "", sql: str = "") -> bool:
    """Detect time-sensitive prompts/SQL that should not use long-lived result cache."""
    text = f"{question} {sql}".lower()
    sensitive_tokens = (
        "today",
        "yesterday",
        "now",
        "current",
        "this month",
        "this year",
        "current_date",
        "getdate",
        "now()",
    )
    return any(tok in text for tok in sensitive_tokens)


def _cache_ttl_seconds(question: str, sql: str) -> int:
    return QUERY_RESULT_RELATIVE_TTL_SECONDS if is_time_sensitive(question=question, sql=sql) else QUERY_RESULT_CACHE_TTL_SECONDS


def _safe_sql_identifier(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "", str(name or ""))


def _compute_query_freshness_marker(db: Optional[SQLDatabase], sql: str) -> Optional[str]:
    """Best-effort freshness marker for cache invalidation.

    Uses indexed date columns (MAX(date_col)) from tables referenced by the SQL.
    """
    if db is None or not ENABLE_QUERY_FRESHNESS_MARKER:
        return None
    tables = _extract_tables_for_diagnostics(sql)[:2]
    if not tables:
        return None

    markers: List[str] = []
    try:
        profile = _get_schema_profile(db)
        lookup = profile.get("lookup", {})
        with db._engine.connect() as conn:
            for table_name in tables:
                table_entry = lookup.get(_normalize_identifier(table_name))
                if not table_entry:
                    table_entry = lookup.get(table_name.lower())
                if not table_entry:
                    continue
                date_col = _pick_existing_column(
                    table_entry,
                    ["CreatedDate", "BookingDate", "booking_date", "UpdatedDate", "ModifiedDate", "LastUpdated", "CheckInDate"],
                )
                if not date_col:
                    continue
                table_actual = _safe_sql_identifier(table_entry.get("table") or table_name)
                col_actual = _safe_sql_identifier(date_col)
                if not table_actual or not col_actual:
                    continue
                marker_sql = f"SELECT CONVERT(varchar(33), MAX([{col_actual}]), 126) AS marker FROM [{table_actual}]"
                marker = conn.execute(sqlalchemy.text(marker_sql)).scalar()
                markers.append(f"{table_actual}:{marker or 'null'}")
    except Exception:
        return None
    return "|".join(markers) if markers else None


def find_cached_result(
    question: str,
    query_cache: Any,
    embedder,
    threshold: float = QUERY_CACHE_SEMANTIC_THRESHOLD,
    db: Optional[SQLDatabase] = None,
):
    if is_time_sensitive(question=question, sql=""):
        return None, None
    entries = _snapshot_query_cache(query_cache)
    if not entries:
        return None, None
    if is_followup_question(question):
        return None, None

    cache_key = _make_query_cache_key(question, db)
    now_ts = time.time()
    db_id = _query_cache_db_identity(db)

    # Fast exact cache lookup first.
    if hasattr(query_cache, "get"):
        try:
            exact_entry = query_cache.get(cache_key)
        except Exception:
            exact_entry = None
        if isinstance(exact_entry, dict):
            if _cache_entry_expired(exact_entry, now_ts):
                try:
                    query_cache.delete(cache_key)
                except Exception:
                    pass
            elif exact_entry.get("db_id") == db_id and (
                not exact_entry.get("freshness_marker")
                or _compute_query_freshness_marker(db, exact_entry.get("sql") or "") == exact_entry.get("freshness_marker")
            ):
                return exact_entry.get("sql"), exact_entry.get("df")

    # Backward-compatible exact scan for list-style caches.
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("cache_key") != cache_key:
            continue
        if entry.get("db_id") != db_id:
            continue
        if _cache_entry_expired(entry, now_ts):
            continue
        marker = entry.get("freshness_marker")
        if marker and _compute_query_freshness_marker(db, entry.get("sql") or "") != marker:
            continue
        return entry.get("sql"), entry.get("df")

    if not QUERY_CACHE_ENABLE_SEMANTIC or embedder is None:
        return None, None

    query_emb = compute_embedding(question, embedder)
    if query_emb is None:
        return None, None

    current_values = extract_key_values(question)
    best_similarity = -1.0
    best_match = None
    best_cached_question = None

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cached_emb = entry.get("embedding")
        cached_question = entry.get("question", "")
        cached_sql = entry.get("sql")
        cached_df = entry.get("df")
        entry_entities = set(entry.get("entities") or [])
        entry_marker = entry.get("freshness_marker")
        entry_db_id = entry.get("db_id") or "unknown"
        if cached_emb is None or cached_sql is None:
            continue
        if entry_db_id != db_id:
            continue
        if _cache_entry_expired(entry, now_ts):
            continue
        if entry_entities and current_values and entry_entities != current_values:
            continue
        if entry_marker and _compute_query_freshness_marker(db, cached_sql) != entry_marker:
            continue
        similarity = float(
            np.dot(query_emb, cached_emb)
            / (np.linalg.norm(query_emb) * np.linalg.norm(cached_emb) + 1e-8)
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = (cached_sql, cached_df)
            best_cached_question = cached_question

    if best_similarity >= threshold and best_match is not None:
        cached_values = extract_key_values(best_cached_question or "")
        if current_values or cached_values:
            if current_values != cached_values:
                return None, None
        return best_match

    return None, None


def cache_query_result(question: str, sql: str, result_df, query_cache: Any, embedder, db: Optional[SQLDatabase] = None):
    if query_cache is None:
        return
    if is_time_sensitive(question=question, sql=sql):
        return
    cache_key = _make_query_cache_key(question, db)
    ttl_seconds = _cache_ttl_seconds(question, sql)
    db_id = _query_cache_db_identity(db)
    freshness_marker = _compute_query_freshness_marker(db, sql)

    # Asynchronous cache write via shared executor to avoid blocking response path.
    def _do_cache():
        emb = None
        if QUERY_CACHE_ENABLE_SEMANTIC and embedder is not None:
            emb = compute_embedding(question, embedder)
        entry = {
            "cache_key": cache_key,
            "embedding": emb,
            "question": question,
            "sql": sql,
            "df": result_df,
            "timestamp": time.time(),
            "ttl_seconds": ttl_seconds,
            "entities": list(extract_key_values(question)),
            "db_id": db_id,
            "freshness_marker": freshness_marker,
        }
        _append_query_cache(query_cache, entry)

    submit_background_task(_do_cache)


# --- Connection setup ---

# Core business tables to include when no pre-aggregated views are found.
# These are the key tables for booking/sales queries in the travel domain.
CORE_BUSINESS_TABLES = {
    "BookingData", "HotelData", "AgentMaster", "Master_City",
    "Master_Country", "HotelChain", "Master_HotelData",
    "SupplierMappingTable", "Master_ServiceType",
    "suppliermaster_Report",
}


def _discover_relevant_tables(
    host: str, port: int, username: str, password: str, database: str
) -> Optional[List[str]]:
    """Fast table discovery using a single INFORMATION_SCHEMA query.

    Avoids introspecting all 100+ tables by finding only:
    1. Pre-aggregated views (*_level_view pattern) if they exist, OR
    2. Core business tables needed for booking/sales queries.

    Uses raw pyodbc for speed (~2s total vs ~180s for full introspection).
    """
    try:
        with closing(_raw_odbc_connect(host, port, username, password, database, timeout=10)) as conn:
            with closing(conn.cursor()) as cursor:
                # Check for pre-aggregated views first (ideal for text-to-SQL)
                cursor.execute(
                    "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.VIEWS "
                    "WHERE TABLE_NAME LIKE '%\\_level\\_view' ESCAPE '\\' "
                    "ORDER BY TABLE_NAME"
                )
                level_views = [r[0] for r in cursor.fetchall()]

                if level_views:
                    return level_views

                # No level views — find which core business tables actually exist
                cursor.execute(
                    "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                    "WHERE TABLE_TYPE IN ('BASE TABLE', 'VIEW') "
                    "ORDER BY TABLE_NAME"
                )
                all_tables = {r[0] for r in cursor.fetchall()}

        stored_proc_hints = set(_extract_table_hints_from_stored_procedure_text(_read_stored_procedure_file()))
        target_tables = set(CORE_BUSINESS_TABLES) | stored_proc_hints
        found = [t for t in target_tables if t in all_tables]
        return found if found else None

    except Exception:
        return None


def initialize_connection(
    host: str,
    port: str,
    db_username: str,
    db_password: str,
    database: str,
    llm_provider: str,
    api_key: str,
    model: str,
    temperature: float,
    query_timeout: int,
    view_support: bool,
):
    """Initialize DB connection, RAG engine, and LLM. Returns (db, db_config, sql_chain, llm, rag, embedder, message, tables_count, views_count, cached_schema_text)."""
    if not host or not port or not db_username or not database:
        raise ValueError("Missing required DB settings: host, port, username, and database are required.")
    if "@" in host or "://" in host:
        raise ValueError(
            "Invalid host value. Provide host only (e.g., 95.168.168.71), not credentials or full URI."
        )
    if not str(port).isdigit():
        raise ValueError(f"Invalid port value: {port}")
    port_num = int(port)
    # Fast network reachability check to avoid long connection hangs.
    try:
        with socket.create_connection((host, port_num), timeout=3):
            pass
    except OSError as e:
        raise ValueError(f"Cannot reach database server at {host}:{port_num} (3s timeout): {e}")

    enable_rag_on_connect = os.getenv("ENABLE_RAG_ON_CONNECT", "true").lower() in {"1", "true", "yes", "on"}
    fetch_schema_counts_on_connect = os.getenv("FETCH_SCHEMA_COUNTS_ON_CONNECT", "false").lower() in {"1", "true", "yes", "on"}
    stored_proc_raw_text = _read_stored_procedure_file()
    stored_proc_sections_count = len(_parse_stored_procedure_sections(stored_proc_raw_text))

    # Fast discovery: query INFORMATION_SCHEMA to find relevant tables/views
    # instead of letting SQLAlchemy introspect ALL 100+ tables (which takes minutes).
    relevant_tables = _discover_relevant_tables(host, port_num, db_username, db_password, database)

    db_config = DatabaseConfig(
        host=host,
        port=str(port_num),
        username=db_username,
        password=db_password,
        database=database,
        query_timeout=query_timeout,
        connect_timeout=5,
        view_support=view_support,
        sample_rows_in_table_info=0,
        lazy_table_reflection=True,
        include_tables=relevant_tables if relevant_tables else None,
    )

    db = create_database_with_views(db_config)

    # Fast connectivity probe for clearer failures.
    _, ping_error = execute_query_safe(db, "SELECT 1 AS ok;", timeout_seconds=min(query_timeout, 10), max_rows=1)
    if ping_error:
        raise RuntimeError(ping_error)

    rag = None
    embedder = None
    cached_schema_text = ""
    rag_status = "RAG disabled for fast connect"

    # Check global schema cache first — avoids re-loading schema + rebuilding FAISS index
    global_cache_hit = _get_global_schema_cache(host, str(port_num), database) if enable_rag_on_connect else None
    if global_cache_hit:
        rag = global_cache_hit["rag_engine"]
        embedder = global_cache_hit["embedder"]
        cached_schema_text = global_cache_hit["schema_text"]
        rag_backend = getattr(rag, "vector_backend", "faiss") if rag else "none"
        rag_status = f"RAG enabled ({rag_backend}, cached)"
    elif enable_rag_on_connect:
        try:
            rag = RAGEngine()
            # load_schema_from_sqldb uses fast inspector (no sample rows) and
            # returns CREATE TABLE text suitable for LLM prompts.
            cached_schema_text = load_schema_from_sqldb(rag, db)
            _add_stored_procedure_knowledge_to_rag(rag, stored_proc_raw_text)
            rag.build_index()
            embedder = rag.embedder
            rag_backend = getattr(rag, "vector_backend", "faiss")
            rag_status = f"RAG enabled ({rag_backend})"
            # Store in global cache for future sessions
            _set_global_schema_cache(host, str(port_num), database, cached_schema_text, rag, embedder)
        except Exception:
            logger.warning("RAG initialization failed, continuing without embeddings", exc_info=True)
            rag = None
            embedder = None
            rag_status = "RAG unavailable (continuing without embeddings)"

    # If RAG was skipped, fall back to bulk INFORMATION_SCHEMA schema text.
    if not cached_schema_text:
        try:
            from collections import defaultdict
            tbl_names = list(db.get_usable_table_names())
            tbl_cols = defaultdict(list)
            with closing(_raw_odbc_connect(host, port_num, db_username, db_password, database, timeout=15)) as conn:
                with closing(conn.cursor()) as cursor:
                    placeholders = ",".join([f"'{t}'" for t in tbl_names])
                    cursor.execute(
                        f"SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
                        f"FROM INFORMATION_SCHEMA.COLUMNS "
                        f"WHERE TABLE_NAME IN ({placeholders}) "
                        f"ORDER BY TABLE_NAME, ORDINAL_POSITION"
                    )
                    for r in cursor.fetchall():
                        tbl_cols[r[0]].append(f"  {r[1]} {r[2].upper()}")
            parts = [f"CREATE TABLE {t} (\n" + ",\n".join(tbl_cols[t]) + "\n)" for t in tbl_names if tbl_cols.get(t)]
            cached_schema_text = "\n\n".join(parts)
        except Exception:
            cached_schema_text = ""

    prompt = ChatPromptTemplate.from_template(SQL_TEMPLATE)

    def get_full_schema(inputs):
        override = inputs.get("full_schema_override")
        return override if override is not None else cached_schema_text

    def get_stored_procedure_guidance(inputs):
        override = inputs.get("stored_procedure_guidance_override")
        if override is not None:
            return override
        return build_domain_digest(_read_stored_procedure_file())

    api_base = None
    if llm_provider.lower() == "deepseek":
        api_base = "https://api.deepseek.com"

    if api_base:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=api_base,
        )
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model=model, temperature=temperature)

    # Create reasoning LLM — used as the PRIMARY SQL generator for all LLM queries.
    # When provider is DeepSeek, always spin up deepseek-reasoner regardless of which
    # chat model the user picked (deepseek-chat, deepseek-coder, etc.).
    # If the user already picked "deepseek-reasoner" as the model, reuse llm directly.
    reasoning_llm = None
    if llm_provider.lower() == "deepseek":
        if "reasoner" in model.lower() or "r1" in model.lower():
            # User already chose the reasoning model — reuse it
            reasoning_llm = llm
            logger.info(f"Using {model} as reasoning model (user-selected)")
        else:
            try:
                reasoning_llm = ChatOpenAI(
                    model="deepseek-reasoner",
                    temperature=0,
                    openai_api_key=api_key,
                    openai_api_base="https://api.deepseek.com",
                    max_tokens=4096,
                )
                logger.info("DeepSeek R1 reasoning model initialized — will handle ALL SQL generation")
            except Exception as e:
                logger.warning(f"Could not initialize deepseek-reasoner, falling back to {model}: {e}")

    def get_few_shot_examples(inputs):
        return inputs.get("few_shot_examples", "")

    def get_retrieved_tables_hint(inputs):
        return inputs.get("retrieved_tables_hint", "none")

    # sql_chain is kept as fallback when no reasoning model is available
    sql_chain = (
        RunnablePassthrough.assign(
            full_schema=get_full_schema,
            stored_procedure_guidance=get_stored_procedure_guidance,
            few_shot_examples=get_few_shot_examples,
            retrieved_tables_hint=get_retrieved_tables_hint,
        )
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    # Optional: schema inspection can be slow on some remote DBs, so keep connect fast by default.
    tables_count = 0
    views_count = 0
    if fetch_schema_counts_on_connect:
        schema_info = get_views_and_tables(db)
        tables_count = len(schema_info.get("tables", []))
        views_count = len(schema_info.get("views", []))

    reasoning_status = f"reasoning=deepseek-reasoner (ALL queries)" if reasoning_llm else "reasoning=none"
    message = (
        f"Connected! Using {llm_provider} ({model}). "
        f"Schema counts on connect: tables={tables_count}, views={views_count}. "
        f"Timeout: {query_timeout}s. {rag_status}. "
        f"Stored procedure guides: {stored_proc_sections_count}. {reasoning_status}."
    )
    return db, db_config, sql_chain, llm, reasoning_llm, rag, embedder, message, tables_count, views_count, cached_schema_text


# --- Conversation context helpers ---

_CONTEXT_TURN_SEP = "---TURN---"
_MAX_CONTEXT_TURNS = 5  # keep last N Q&A turns visible to the LLM


def _extract_query_topic(question: str, sql: str) -> str:
    """Return a short label describing the dimension + metric being analyzed."""
    q = question.lower()
    # Dimension
    if "hotel" in q or "product" in q:
        dim = "hotel"
    elif "supplier" in q:
        dim = "supplier"
    elif "agent type" in q or "b2b" in q or "api" in q or "we group" in q:
        dim = "agent_type"
    elif "agent" in q:
        dim = "agent"
    elif "country" in q:
        dim = "country"
    elif "city" in q:
        dim = "city"
    elif "nationality" in q or "natinality" in q:
        dim = "nationality"
    elif "chain" in q:
        dim = "chain"
    else:
        dim = "overall"
    # Metric
    if "growth" in q or "yoy" in q or "compared" in q:
        metric = "growth"
    elif "profit" in q:
        metric = "profit"
    elif "booking" in q:
        metric = "bookings"
    elif "room night" in q:
        metric = "room_nights"
    elif "cost" in q or "expense" in q:
        metric = "cost"
    else:
        metric = "sales"
    return f"{dim}:{metric}"



def _trim_conversation_context(context: str, max_turns: int = _MAX_CONTEXT_TURNS) -> str:
    """Keep only the most recent `max_turns` turns from the context string."""
    if not context:
        return context
    # Split on the separator we now use; fallback to double-newline for old format
    if _CONTEXT_TURN_SEP in context:
        parts = [p for p in context.split(_CONTEXT_TURN_SEP) if p.strip()]
        if len(parts) > max_turns:
            parts = parts[-max_turns:]
        return _CONTEXT_TURN_SEP.join(parts)
    else:
        # Legacy format: split on double newline
        turns = [t for t in context.split("\n\n") if t.strip()]
        if len(turns) > max_turns:
            turns = turns[-max_turns:]
        return "\n\n".join(turns) + "\n\n"



def _new_session_query_state(dialect: str) -> Dict[str, Any]:
    return SessionState(dialect=normalize_sql_dialect(dialect)).to_dict()



def _extract_sql_where_filters(sql: Optional[str]) -> List[str]:
    if not sql:
        return []
    where_match = re.search(
        r"(?is)\bWHERE\b\s*(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|$)",
        sql,
    )
    if not where_match:
        return []
    where_clause = " ".join(where_match.group(1).split())
    parts = [p.strip() for p in re.split(r"(?i)\s+AND\s+", where_clause) if p.strip()]
    return parts[:8]


def _extract_sql_dimensions(sql: Optional[str]) -> List[str]:
    if not sql:
        return []
    group_match = re.search(
        r"(?is)\bGROUP\s+BY\b\s*(.+?)(?:\bORDER\s+BY\b|\bHAVING\b|$)",
        sql,
    )
    if not group_match:
        return []
    dims = []
    for raw in group_match.group(1).split(","):
        val = " ".join(raw.strip().split())
        if val:
            dims.append(val)
    return dims[:6]


def _extract_sql_metrics(sql: Optional[str]) -> List[str]:
    if not sql:
        return []
    metrics: List[str] = []
    alias_matches = re.findall(r"(?i)\bAS\s+\[?([A-Za-z][A-Za-z0-9_ ]*)\]?", sql)
    for alias in alias_matches:
        alias_clean = " ".join(alias.strip().split())
        if any(k in alias_clean.lower() for k in ("revenue", "sales", "cost", "profit", "booking", "avg", "count")):
            metrics.append(alias_clean)
    for func in re.findall(r"(?i)\b(sum|count|avg|min|max)\s*\(", sql):
        metrics.append(func.upper())
    seen = set()
    deduped = []
    for m in metrics:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped[:6]


def _extract_sql_time_window(sql: Optional[str]) -> Dict[str, Optional[str]]:
    if not sql:
        return {"start": None, "end": None}

    # Prefer explicit paired range on the same column: col >= 'YYYY-MM-DD' AND col < 'YYYY-MM-DD'
    pair_pattern = re.compile(
        r"(?is)([A-Za-z_][A-Za-z0-9_\.\[\]]*)\s*>=\s*'(\d{4}-\d{2}-\d{2})'\s+AND\s+\1\s*<\s*'(\d{4}-\d{2}-\d{2})'"
    )
    pair_match = pair_pattern.search(sql)
    if pair_match:
        return {"start": pair_match.group(2), "end": pair_match.group(3)}

    starts = re.findall(r"(?is)\b[A-Za-z_][A-Za-z0-9_\.\[\]]*\s*>=\s*'(\d{4}-\d{2}-\d{2})'", sql)
    ends = re.findall(r"(?is)\b[A-Za-z_][A-Za-z0-9_\.\[\]]*\s*<\s*'(\d{4}-\d{2}-\d{2})'", sql)
    return {
        "start": starts[0] if starts else None,
        "end": ends[0] if ends else None,
    }


def _extract_question_time_window(question: Optional[str]) -> Dict[str, Optional[str]]:
    q = (question or "").lower()
    r = _get_ist_date_ranges()
    if "today" in q:
        return {"start": r["today_start"], "end": r["today_end"]}
    if "yesterday" in q:
        return {"start": r["yesterday_start"], "end": r["yesterday_end"]}
    if "this week" in q:
        return {"start": r["this_week_start"], "end": r["this_week_end"]}
    if "last week" in q:
        return {"start": r["last_week_start"], "end": r["last_week_end"]}
    if "this month" in q:
        return {"start": r["this_month_start"], "end": r["this_month_end"]}
    if "this year" in q:
        return {"start": r["this_year_start"], "end": r["this_year_end"]}
    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    if date_match:
        start = datetime.fromisoformat(date_match.group(1)).date()
        end = start + timedelta(days=1)
        return {"start": start.isoformat(), "end": end.isoformat()}
    return {"start": None, "end": None}


def _update_session_query_state(
    state: Dict[str, Any],
    sql: Optional[str],
    dialect: str,
    question: Optional[str] = None,
) -> None:
    if not state or not sql:
        return
    state["dialect"] = normalize_sql_dialect(dialect)
    state["last_sql"] = sql
    state["last_table"] = _extract_from_table_name(sql)
    state["last_date_col"] = _detect_preferred_date_col(sql)
    sql_window = _extract_sql_time_window(sql)
    if not (sql_window.get("start") and sql_window.get("end")):
        q_window = _extract_question_time_window(question)
        if q_window.get("start") and q_window.get("end"):
            sql_window = q_window
    state["last_time_window"] = sql_window
    state["last_dimensions"] = _extract_sql_dimensions(sql)
    state["last_metrics"] = _extract_sql_metrics(sql)
    state["last_filters"] = _extract_sql_where_filters(sql)


def _build_prompt_context_from_state(
    state: Dict[str, Any],
    conversation_context: str,
    conversation_turns: Optional[List[ConversationTurn]],
) -> str:
    # CHANGED: prompts now prioritize structured memory over long raw strings.
    recent_text = ""
    if conversation_context and conversation_context.strip():
        trimmed = _trim_conversation_context(conversation_context, max_turns=2)
        recent_text = trimmed[-700:]
    elif conversation_turns:
        recent_text = serialize_conversation_turns(conversation_turns, max_turns=2)[-700:]

    structured_json = json.dumps(state, ensure_ascii=True)
    recent_block = recent_text if recent_text else "none"
    return f"STRUCTURED_STATE: {structured_json}\nRECENT_CONVERSATION: {recent_block}"


### PROMPT BUDGET
def _compact_state_summary(state: Dict[str, Any]) -> str:
    """Very small structured memory block for prompt compression mode."""
    if not state:
        return "STRUCTURED_STATE: {}"
    compact = {
        "dialect": state.get("dialect"),
        "last_table": state.get("last_table"),
        "last_date_col": state.get("last_date_col"),
        "last_time_window": state.get("last_time_window"),
        "last_dimensions": (state.get("last_dimensions") or [])[:4],
        "last_metrics": (state.get("last_metrics") or [])[:4],
        "last_filters": (state.get("last_filters") or [])[:4],
    }
    return f"STRUCTURED_STATE: {json.dumps(compact, ensure_ascii=True)}"


def _extract_key_guidance_rules(stored_guidance: str) -> str:
    """Keep only core rules: metric formulas + status exclusions + date mapping."""
    if not stored_guidance:
        return "No guidance available."

    lines = [ln.strip() for ln in stored_guidance.splitlines() if ln.strip()]
    kept: List[str] = []
    section = ""
    for line in lines:
        if line.startswith("==="):
            section = line.lower()
            continue
        if "metric definitions" in section or "metric formulas" in section:
            if line.startswith("•"):
                kept.append(line)
            continue
        if "mandatory filters" in section or "status exclusions" in section:
            if "bookingstatus" in line.lower() or "cancelled" in line.lower():
                kept.append(line)
            continue
        if "date column usage" in section or "date columns" in section or "sargable date filters" in section:
            low = line.lower()
            if any(tok in low for tok in ("createddate", "checkindate", "checkoutdate", "date_col >=", "date_col <")):
                kept.append(line)
            continue
        if "canonical joins" in section:
            if line.startswith("•"):
                kept.append(line)
            continue

    if not kept:
        return "\n".join(lines[:60]) if lines else "No guidance available."
    header = [
        "=== METRIC + STATUS + DATE RULES (compressed) ===",
    ]
    return "\n".join(header + kept[:25])


def _score_table_relevance(
    question_tokens: set,
    table_entry: Dict[str, Any],
    preferred_tables: Optional[set] = None,
) -> int:
    score = 0
    table_name = table_entry.get("table_lower", "")
    table_words = set(re.findall(r"[a-z0-9]+", table_name.replace("_", " ")))
    score += 15 * len(question_tokens & table_words)

    col_map = table_entry.get("columns_map", {})
    col_hits = 0
    for col_lower in col_map.keys():
        col_words = set(re.findall(r"[a-z0-9]+", col_lower.replace("_", " ")))
        if question_tokens & col_words:
            col_hits += 1
    score += min(col_hits, 8) * 2

    if "booking" in question_tokens and "booking" in table_name:
        score += 8
    if "agent" in question_tokens and "agent" in table_name:
        score += 6
    if "supplier" in question_tokens and "supplier" in table_name:
        score += 6
    if "country" in question_tokens and "country" in table_name:
        score += 5
    if "city" in question_tokens and "city" in table_name:
        score += 5
    if preferred_tables and table_name in preferred_tables:
        score += 20
    if preferred_tables and table_entry.get("table", "").lower() in preferred_tables:
        score += 20
    return score


def _render_schema_snippet(table_entry: Dict[str, Any], selected_cols: List[str]) -> str:
    table_name = table_entry.get("table", "")
    col_types = table_entry.get("column_types", {})
    rendered = []
    for col in selected_cols:
        ctype = str(col_types.get(col.lower(), "VARCHAR")).upper()
        rendered.append(f"  {col} {ctype}")
    return f"CREATE TABLE {table_name} (\n" + ",\n".join(rendered) + "\n)"


### RELEVANT SCHEMA ONLY
def _build_relevant_schema_only(question: str, db: Optional[SQLDatabase], fallback_schema_text: str = "") -> str:
    """Compress schema to only likely-needed tables/columns for this question."""
    if db is None:
        return (fallback_schema_text or "")[:6000]

    profile = _get_schema_profile(db)
    tables = profile.get("tables", [])
    if not tables:
        return (fallback_schema_text or "")[:6000]

    q_tokens = set(re.findall(r"[a-z0-9_]+", (question or "").lower()))
    preferred_tables: set = set()
    state_last_table = (getattr(db, "_state_last_table_hint", None) or "").strip().lower() if db is not None else ""
    if state_last_table:
        preferred_tables.add(_normalize_identifier(state_last_table))
        preferred_tables.add(state_last_table)
    rag_hints = getattr(db, "_rag_table_hints", None) if db is not None else None
    if isinstance(rag_hints, list):
        for hint in rag_hints:
            hint_low = str(hint or "").strip().lower()
            if hint_low:
                preferred_tables.add(_normalize_identifier(hint_low))
                preferred_tables.add(hint_low)

    ranked = []
    for entry in tables:
        score = _score_table_relevance(q_tokens, entry, preferred_tables=preferred_tables)
        if score > 0:
            ranked.append((score, entry))
    ranked.sort(key=lambda item: item[0], reverse=True)

    if not ranked:
        # deterministic fallback: keep BookingData first when present
        ranked = [
            (1, entry) for entry in tables
            if entry.get("table_lower") in {"bookingdata", "bookingtablequery", "country_level_view", "agent_level_view"}
        ]
        if not ranked:
            ranked = [(1, tables[0])]

    must_have_cols = {
        "createddate", "bookingdate", "booking_date", "checkindate", "checkoutdate",
        "bookingstatus", "pnrno", "agentid", "supplierid", "productname",
        "agentbuyingprice", "companybuyingprice", "productcountryid", "productcityid",
    }
    snippets = []
    for _, entry in ranked[:4]:
        cols = entry.get("columns", [])
        if not cols:
            continue
        selected: List[str] = []
        for col in cols:
            col_lower = col.lower()
            col_tokens = set(re.findall(r"[a-z0-9_]+", col_lower.replace("_", " ")))
            if q_tokens & col_tokens:
                selected.append(col)
            elif col_lower in must_have_cols:
                selected.append(col)
            if len(selected) >= 20:
                break
        if not selected:
            selected = cols[:12]
        snippets.append(_render_schema_snippet(entry, selected))

    return "\n\n".join(snippets) if snippets else (fallback_schema_text or "")[:6000]


def _build_prompt_payload(
    question: str,
    db: Optional[SQLDatabase],
    schema_text: str,
    stored_guidance: str,
    conversation_context: str,
    prompt_context: str,
    few_shot_examples: str,
    session_state: Dict[str, Any],
    rag_table_hints: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Apply prompt budget controls and return effective prompt payload blocks."""
    schema_len = len(schema_text or "")
    guidance_len = len(stored_guidance or "")
    context_len = len(conversation_context or "")
    few_shot_len = len(few_shot_examples or "")
    total_prompt_chars = schema_len + guidance_len + context_len + few_shot_len

    logger.info(
        "Prompt length chars schema_text=%s stored_procedure_guidance=%s conversation_context=%s few_shot_examples=%s combined=%s",
        schema_len,
        guidance_len,
        context_len,
        few_shot_len,
        total_prompt_chars,
    )

    if db is not None:
        # transient hints for relevant-schema builder; keeps function signature simple.
        setattr(db, "_state_last_table_hint", session_state.get("last_table"))
        setattr(db, "_rag_table_hints", rag_table_hints or [])

    payload = {
        "schema_text": schema_text or "",
        "stored_guidance": stored_guidance or "",
        "prompt_context": prompt_context or "",
        "few_shot_examples": few_shot_examples or "",
    }

    if total_prompt_chars <= PROMPT_BUDGET_CHARS:
        return payload

    # Budget overflow: compress all high-cardinality blocks.
    payload["schema_text"] = _build_relevant_schema_only(question, db, fallback_schema_text=schema_text)
    payload["stored_guidance"] = _extract_key_guidance_rules(stored_guidance)
    payload["prompt_context"] = _compact_state_summary(session_state)

    new_combined = (
        len(payload["schema_text"])
        + len(payload["stored_guidance"])
        + len(payload["prompt_context"])
        + len(payload["few_shot_examples"])
    )
    logger.info(
        "Prompt budget compression applied schema=%s guidance=%s context=%s few_shot=%s combined=%s limit=%s",
        len(payload["schema_text"]),
        len(payload["stored_guidance"]),
        len(payload["prompt_context"]),
        len(payload["few_shot_examples"]),
        new_combined,
        PROMPT_BUDGET_CHARS,
    )
    return payload


def _infer_metric_alias_from_question(question: str) -> str:
    q = (question or "").lower()
    if any(tok in q for tok in ("cost", "expense", "spend")):
        return "cost"
    if "profit" in q:
        return "profit"
    if "booking" in q:
        return "bookings"
    return "revenue"


def _is_monthly_trend_question(question: str) -> bool:
    q = (question or "").lower()
    has_monthly = any(tok in q for tok in ("monthly", "monthwise", "month wise", "by month", "month on month"))
    has_trend = any(tok in q for tok in ("trend", "over time", "timeline", "month"))
    has_dimension = any(tok in q for tok in ("agent", "supplier", "country", "city", "hotel", "chain", "category"))
    return (has_monthly or has_trend) and has_dimension


def build_monthly_trend_fallback_sql(question: str, db: Optional[SQLDatabase]) -> Optional[str]:
    """Deterministic monthly trend SQL (MonthStart + category) for timeout fallback."""
    if db is None or not _is_monthly_trend_question(question):
        return None
    if not _table_exists_fast(db, "BookingData"):
        return None

    q = " ".join((question or "").lower().split())
    metric_alias = _infer_metric_alias_from_question(q)
    if metric_alias == "cost":
        metric_expr, metric_name = "SUM(BD.[CompanyBuyingPrice])", "Total Cost"
    elif metric_alias == "profit":
        metric_expr, metric_name = "SUM(BD.[AgentBuyingPrice] - BD.[CompanyBuyingPrice])", "Total Profit"
    elif metric_alias == "bookings":
        metric_expr, metric_name = "COUNT(DISTINCT BD.[PNRNo])", "Total Bookings"
    else:
        metric_expr, metric_name = "SUM(BD.[AgentBuyingPrice])", "Total Revenue"

    joins = ""
    category_expr = "BD.[ProductName]"
    if any(tok in q for tok in ("agent", "agents")):
        joins = "LEFT JOIN dbo.AgentMaster_V1 AM ON AM.AgentId = BD.AgentId"
        category_expr = "AM.AgentName"
    elif any(tok in q for tok in ("supplier", "suppliers")):
        joins = "LEFT JOIN dbo.suppliermaster_Report SM ON SM.EmployeeId = BD.SupplierId"
        category_expr = "SM.SupplierName"
    elif any(tok in q for tok in ("country", "countries")):
        joins = "LEFT JOIN dbo.Master_Country MC ON MC.CountryID = BD.ProductCountryid"
        category_expr = "MC.Country"
    elif any(tok in q for tok in ("city", "cities")):
        joins = "LEFT JOIN dbo.Master_City MCI ON MCI.CityId = BD.ProductCityId"
        category_expr = "MCI.City"
    elif any(tok in q for tok in ("chain", "chains")):
        joins = "LEFT JOIN dbo.Hotelchain HC ON HC.HotelId = BD.ProductId"
        category_expr = "HC.Chain"

    date_clauses = _build_date_where_clauses(q, "BD.[CreatedDate]")
    if not date_clauses and not _question_requests_all_time(question):
        r = _get_ist_date_ranges()
        date_clauses = [
            f"BD.[CreatedDate] >= '{r['this_year_start']}'",
            f"BD.[CreatedDate] < '{r['this_year_end']}'",
        ]

    where_parts = ["BD.[BookingStatus] NOT IN ('Cancelled', 'Not Confirmed', 'On Request')"] + date_clauses
    where_sql = " AND ".join(where_parts) if where_parts else "1=1"
    month_start_expr = "DATEFROMPARTS(YEAR(BD.[CreatedDate]), MONTH(BD.[CreatedDate]), 1)"

    return (
        "SELECT\n"
        f"  {month_start_expr} AS [MonthStart],\n"
        f"  {category_expr} AS [Category],\n"
        f"  {metric_expr} AS [{metric_name}]\n"
        "FROM dbo.BookingData BD\n"
        f"{joins + chr(10) if joins else ''}"
        f"WHERE {where_sql}\n"
        f"GROUP BY {month_start_expr}, {category_expr}\n"
        "ORDER BY [MonthStart] DESC, [Category] ASC;"
    )


def _is_supplier_price_comparison_question(question: str) -> bool:
    q = " ".join((question or "").lower().split())
    has_supplier = any(tok in q for tok in ("supplier", "suppliers"))
    has_agent_price = any(tok in q for tok in ("agent buying", "agent buying price", "agent price"))
    has_company_price = any(tok in q for tok in ("company buying", "company buying price", "company price"))
    has_compare = any(tok in q for tok in ("compare", "vs", "versus", "difference", "variance"))
    has_percent = any(tok in q for tok in ("percent", "percentage", "%"))
    return has_supplier and has_agent_price and has_company_price and (has_compare or has_percent)


def build_supplier_price_comparison_fallback_sql(question: str, db: Optional[SQLDatabase]) -> Optional[str]:
    """Deterministic supplier-level comparison SQL for agent/company buying price queries."""
    if db is None or not _is_supplier_price_comparison_question(question):
        return None
    if not _table_exists_fast(db, "BookingData"):
        return None

    q = " ".join((question or "").lower().split())
    date_clauses = _build_date_where_clauses(q, "BD.[CreatedDate]")
    if not date_clauses and not _question_requests_all_time(question):
        r = _get_ist_date_ranges()
        date_clauses = [
            f"BD.[CreatedDate] >= '{r['this_year_start']}'",
            f"BD.[CreatedDate] < '{r['this_year_end']}'",
        ]
    where_parts = ["BD.[BookingStatus] NOT IN ('Cancelled', 'Not Confirmed', 'On Request')"] + date_clauses
    where_sql = " AND ".join(where_parts) if where_parts else "1=1"

    if _table_exists_fast(db, "suppliermaster_Report"):
        return (
            "WITH SupplierMap AS (\n"
            "  SELECT DISTINCT [EmployeeId], [SupplierName]\n"
            "  FROM dbo.[suppliermaster_Report] WITH (NOLOCK)\n"
            ")\n"
            "SELECT TOP (100)\n"
            "  COALESCE(SM.[SupplierName], CONCAT('Supplier-', CAST(BD.[SupplierId] AS VARCHAR(50)))) AS [Supplier],\n"
            "  SUM(COALESCE(BD.[AgentBuyingPrice], 0)) AS [Agent Buying Price],\n"
            "  SUM(COALESCE(BD.[CompanyBuyingPrice], 0)) AS [Company Buying Price],\n"
            "  SUM(COALESCE(BD.[AgentBuyingPrice], 0) - COALESCE(BD.[CompanyBuyingPrice], 0)) AS [Variance],\n"
            "  CASE\n"
            "    WHEN SUM(COALESCE(BD.[CompanyBuyingPrice], 0)) = 0 THEN NULL\n"
            "    ELSE (\n"
            "      SUM(COALESCE(BD.[AgentBuyingPrice], 0) - COALESCE(BD.[CompanyBuyingPrice], 0)) * 100.0\n"
            "      / NULLIF(SUM(COALESCE(BD.[CompanyBuyingPrice], 0)), 0)\n"
            "    )\n"
            "  END AS [Variance Percentage]\n"
            "FROM dbo.[BookingData] BD\n"
            "LEFT JOIN SupplierMap SM ON SM.[EmployeeId] = BD.[SupplierId]\n"
            f"WHERE {where_sql}\n"
            "GROUP BY COALESCE(SM.[SupplierName], CONCAT('Supplier-', CAST(BD.[SupplierId] AS VARCHAR(50))))\n"
            "HAVING SUM(COALESCE(BD.[AgentBuyingPrice], 0)) <> 0 OR SUM(COALESCE(BD.[CompanyBuyingPrice], 0)) <> 0\n"
            "ORDER BY [Variance] DESC;"
        )

    return (
        "SELECT TOP (100)\n"
        "  CONCAT('Supplier-', CAST(BD.[SupplierId] AS VARCHAR(50))) AS [Supplier],\n"
        "  SUM(COALESCE(BD.[AgentBuyingPrice], 0)) AS [Agent Buying Price],\n"
        "  SUM(COALESCE(BD.[CompanyBuyingPrice], 0)) AS [Company Buying Price],\n"
        "  SUM(COALESCE(BD.[AgentBuyingPrice], 0) - COALESCE(BD.[CompanyBuyingPrice], 0)) AS [Variance],\n"
        "  CASE\n"
        "    WHEN SUM(COALESCE(BD.[CompanyBuyingPrice], 0)) = 0 THEN NULL\n"
        "    ELSE (\n"
        "      SUM(COALESCE(BD.[AgentBuyingPrice], 0) - COALESCE(BD.[CompanyBuyingPrice], 0)) * 100.0\n"
        "      / NULLIF(SUM(COALESCE(BD.[CompanyBuyingPrice], 0)), 0)\n"
        "    )\n"
        "  END AS [Variance Percentage]\n"
        "FROM dbo.[BookingData] BD\n"
        f"WHERE {where_sql}\n"
        "GROUP BY BD.[SupplierId]\n"
        "HAVING SUM(COALESCE(BD.[AgentBuyingPrice], 0)) <> 0 OR SUM(COALESCE(BD.[CompanyBuyingPrice], 0)) <> 0\n"
        "ORDER BY [Variance] DESC;"
    )


def build_ytd_pytd_growth_fallback_sql(question: str, db: Optional[SQLDatabase]) -> Optional[str]:
    """Deterministic YTD vs PYTD growth SQL template."""
    if db is None or not _table_exists_fast(db, "BookingData"):
        return None
    q = (question or "").lower()
    if not any(tok in q for tok in ("ytd", "pytd", "growth", "compared to last year", "year over year", "yoy")):
        return None

    if any(tok in q for tok in ("supplier", "suppliers")):
        dim_expr, dim_alias, join_sql = "SM.SupplierName", "Category", "LEFT JOIN dbo.suppliermaster_Report SM ON SM.EmployeeId = BD.SupplierId"
    elif any(tok in q for tok in ("agent", "agents")):
        dim_expr, dim_alias, join_sql = "AM.AgentName", "Category", "LEFT JOIN dbo.AgentMaster_V1 AM ON AM.AgentId = BD.AgentId"
    elif any(tok in q for tok in ("country", "countries")):
        dim_expr, dim_alias, join_sql = "MC.Country", "Category", "LEFT JOIN dbo.Master_Country MC ON MC.CountryID = BD.ProductCountryid"
    elif any(tok in q for tok in ("city", "cities")):
        dim_expr, dim_alias, join_sql = "MCI.City", "Category", "LEFT JOIN dbo.Master_City MCI ON MCI.CityId = BD.ProductCityId"
    else:
        dim_expr, dim_alias, join_sql = "BD.ProductName", "Category", ""

    return (
        "WITH base AS (\n"
        f"  SELECT {dim_expr} AS [{dim_alias}],\n"
        "    SUM(CASE\n"
        "      WHEN BD.CreatedDate >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1)\n"
        "       AND BD.CreatedDate < DATEADD(DAY, 1, CAST(GETDATE() AS DATE))\n"
        "      THEN BD.AgentBuyingPrice ELSE 0 END) AS YTD_Sales,\n"
        "    SUM(CASE\n"
        "      WHEN BD.CreatedDate >= DATEFROMPARTS(YEAR(GETDATE()) - 1, 1, 1)\n"
        "       AND BD.CreatedDate < DATEADD(DAY, 1, DATEADD(YEAR, -1, CAST(GETDATE() AS DATE)))\n"
        "      THEN BD.AgentBuyingPrice ELSE 0 END) AS PYTD_Sales\n"
        "  FROM dbo.BookingData BD\n"
        f"{join_sql + chr(10) if join_sql else ''}"
        "  WHERE BD.CreatedDate >= DATEFROMPARTS(YEAR(GETDATE()) - 1, 1, 1)\n"
        "    AND BD.CreatedDate < DATEFROMPARTS(YEAR(GETDATE()) + 1, 1, 1)\n"
        "    AND BD.BookingStatus NOT IN ('Cancelled', 'Not Confirmed', 'On Request')\n"
        f"  GROUP BY {dim_expr}\n"
        ")\n"
        "SELECT TOP (50)\n"
        f"  [{dim_alias}],\n"
        "  YTD_Sales,\n"
        "  PYTD_Sales,\n"
        "  (YTD_Sales - PYTD_Sales) AS Growth_Amount,\n"
        "  CASE WHEN PYTD_Sales = 0 THEN NULL\n"
        "       ELSE (YTD_Sales - PYTD_Sales) * 100.0 / NULLIF(PYTD_Sales, 0)\n"
        "  END AS Growth_Percentage\n"
        "FROM base\n"
        "WHERE YTD_Sales IS NOT NULL\n"
        "ORDER BY Growth_Percentage DESC;"
    )


### LLM SLA + FALLBACK
def _invoke_with_timeout(fn, timeout_ms: int):
    return run_with_timeout(fn, timeout_s=max(1, timeout_ms) / 1000.0)


def _deterministic_timeout_fallback_sql(question: str, db: Optional[SQLDatabase]) -> Tuple[Optional[str], Optional[str]]:
    supplier_compare_sql = build_supplier_price_comparison_fallback_sql(question, db)
    if supplier_compare_sql:
        return supplier_compare_sql, "supplier_price_comparison"
    if is_business_overview_question(question):
        source = _resolve_business_overview_source(db) if db is not None else None
        if source is not None:
            return build_business_overview_sql_from_source(source, date_scope="this_month"), "business_overview"
    yoy_sql = build_ytd_pytd_growth_fallback_sql(question, db)
    if yoy_sql:
        return yoy_sql, "ytd_pytd_growth"
    monthly_sql = build_monthly_trend_fallback_sql(question, db)
    if monthly_sql:
        return monthly_sql, "monthly_trend"
    topn_sql = build_topn_fallback_sql(question, db) if db is not None else None
    if topn_sql:
        return topn_sql, "topn"
    return None, None


# --- Main query handler ---

def handle_query(
    question: str,
    db: SQLDatabase,
    db_config,
    sql_chain,
    llm,
    rag_engine,
    embedder,
    chat_history: List[Dict],
    query_cache: Any,
    cached_schema_text: str = "",
    session_state: Optional[SessionState] = None,
    conversation_turns: Optional[List] = None,
    reasoning_llm=None,
    sql_dialect: str = DEFAULT_SQL_DIALECT,
    enable_nolock: bool = False,
) -> Dict[str, Any]:
    """
    Process a user question end-to-end.
    Returns dict with: intent, nl_answer, sql, results, row_count, from_cache, error, updated_state, conversation_turn
    """
    t_start = time.perf_counter()
    timer = StepTimer()
    timer.mark("start")
    active_dialect = normalize_sql_dialect(sql_dialect or _detect_sql_dialect_from_db(db))
    dialect_label = _dialect_label(active_dialect)
    relative_date_reference = _build_relative_date_reference()
    nolock_setting = str(bool(enable_nolock)).lower()
    log_event(
        logger,
        logging.INFO,
        "handle_query_start",
        dialect=active_dialect,
        question_len=len(question or ""),
    )
    # ### STRUCTURED MEMORY — resolve state dict and build conversation context
    if session_state is not None:
        _sq = session_state.to_dict()
        _struct_summary = session_state.compact_summary() if session_state.has_context() else ""
        _turns_text = serialize_conversation_turns(conversation_turns or [], max_turns=3)
        _turns_clean = _turns_text if _turns_text != "No prior conversation." else ""
        if _struct_summary and _turns_clean:
            conversation_context = f"{_struct_summary}\n{_turns_clean}".strip()
        else:
            conversation_context = _struct_summary or _turns_clean
    else:
        _sq = _new_session_query_state(active_dialect)
        _turns_text = serialize_conversation_turns(conversation_turns or [], max_turns=3)
        conversation_context = _turns_text if _turns_text != "No prior conversation." else ""
    # Rebind session_state to dict so existing internal code is unchanged
    session_state = _sq
    full_schema_text = ""
    stored_procedure_guidance = "No stored procedure guidance available."
    fallback_used_request = False

    def _elapsed_seconds() -> float:
        return time.perf_counter() - t_start

    def _finalize_result(payload: Dict[str, Any]) -> Dict[str, Any]:
        timer.mark("total")
        payload.setdefault("fallback_used", fallback_used_request)
        payload["timing"] = timer.summary()
        log_event(
            logger,
            logging.INFO,
            "handle_query_complete",
            total_ms=payload["timing"].get("total", 0.0),
            from_cache=bool(payload.get("from_cache", False)),
            has_error=bool(payload.get("error")),
        )
        if payload["timing"].get("total", 0.0) > 5000:
            logger.warning("SLOW QUERY: %s", payload["timing"])
        return payload

    # Step 1: Detect intent
    t_intent = time.perf_counter()
    intent = detect_intent_simple(question)

    if intent is None:
        if has_recent_data_query(chat_history) and (
            is_contextual_data_query(question) or _is_short_contextual_followup(question)
        ):
            # Short follow-ups in a data session ("what about India?", "only last week") are
            # almost always data queries — skip LLM intent detection for speed.
            intent = "DATA_QUERY"
        else:
            schema_summary = ", ".join(db.get_usable_table_names()) if db else ""
            intent = detect_intent_llm(question, schema_summary, llm, conversation_context)
    logger.info(f"Intent detected: {intent} ({(time.perf_counter()-t_intent)*1000:.0f}ms) | Q: {question[:80]}")
    timer.mark("intent_detection")

    # Step 2: Handle non-data-query intents
    if intent in {"GREETING", "THANKS", "HELP", "FAREWELL", "OUT_OF_SCOPE", "CLARIFICATION_NEEDED"}:
        tables = ", ".join(db.get_usable_table_names()) if db else "none"
        history = conversation_context[-500:] if conversation_context else "none"
        response = generate_conversational_response(question, intent, tables, history, llm)
        return _finalize_result({
            "intent": "CONVERSATION",
            "nl_answer": response,
            "sql": None,
            "results": None,
            "row_count": 0,
            "from_cache": False,
            "error": None,
            "updated_state": session_state,
        })

    full_schema_text = cached_schema_text or (db.get_table_info() if db else "")
    timer.mark("schema_loading")
    stored_procedure_guidance = build_domain_digest(_read_stored_procedure_file())
    timer.mark("stored_procedure_guidance")

    # Step 3: Check for sort/filter follow-up
    previous_sql = _get_last_sql(chat_history) or session_state.get("last_sql")
    previous_result_df = _get_last_result_df(chat_history)

    cleaned_query = None
    is_valid = False
    llm_retry_budget = 1

    # Step 3a: "De-clevering" fast-path for vague business overview questions.
    # Only applies when this is NOT a follow-up and there is no previous SQL context.
    contextual_topn_sql = None
    if previous_sql and _is_bare_topn_followup(question):
        contextual_topn_sql = modify_sql_for_topn_followup(previous_sql, question)

    topn_fallback_sql = contextual_topn_sql or build_topn_fallback_sql(question, db)
    is_overview_request = previous_sql is None and not is_followup_question(question) and is_business_overview_question(question)
    if is_overview_request:
        q_lower = question.lower()
        if "last month" in q_lower:
            date_scope = "last_month"
        elif "this month" in q_lower:
            date_scope = "this_month"
        elif "last year" in q_lower:
            date_scope = "last_year"
        else:
            # Performance-first default for vague overviews: current month snapshot.
            # This avoids expensive full-year scans on large BookingData tables.
            date_scope = "this_month"
        overview_source = _resolve_business_overview_source(db)
        if overview_source is None:
            # Deterministic fallback for this project: use BookingData KPIs.
            overview_source = {
                "table_actual": "BookingData",
                "date_col": "CreatedDate",
                "revenue_expr": "SUM(COALESCE(AgentBuyingPrice, 0))",
                "profit_expr": "SUM(COALESCE(AgentBuyingPrice, 0) - COALESCE(CompanyBuyingPrice, 0))",
                "bookings_expr": "COUNT(DISTINCT PNRNo)",
                "status_filter_expr": "[BookingStatus] NOT IN ('Cancelled', 'Not Confirmed', 'On Request')",
            }
        if overview_source is not None:
            cleaned_query = build_business_overview_sql_from_source(overview_source, date_scope=date_scope)
            is_valid = True
            # Skip the rest of SQL generation logic and execute below.
            is_sort_request = False
            is_filter_mod = False
        else:
            # No suitable source found - fall back to normal SQL generation.
            is_sort_request = False
            is_filter_mod = False
    elif topn_fallback_sql is not None:
        # Force deterministic grouped ranking for top/bottom-N prompts,
        # even when there's previous SQL in history.
        # For bare follow-ups like "give me top 10", this reuses previous SQL context.
        cleaned_query = topn_fallback_sql
        is_valid = True
        is_sort_request = False
        is_filter_mod = False
    else:
        is_sort_request = previous_sql and is_sort_followup(question)
        is_filter_mod = previous_sql and is_filter_modification_followup(question, previous_result_df)

    is_followup_filter = False
    if is_sort_request:
        cleaned_query = modify_sql_for_sort(previous_sql, question, llm)
        is_valid = True
    elif is_filter_mod:
        cleaned_query = modify_sql_for_filter(previous_sql, question, llm, db=db)
        is_valid = True
        is_followup_filter = True
    elif cleaned_query is None:
        ### FAST PATH: skip cache lookups for follow-up/time-sensitive prompts.
        is_followup = is_followup_question(question)
        cache_bypassed = is_followup or is_time_sensitive(question=question, sql="")
        if cache_bypassed:
            cached_sql, cached_df = None, None
            log_event(
                logger,
                logging.INFO,
                "query_cache_bypass",
                reason="followup_or_time_sensitive",
                followup=bool(is_followup),
                time_sensitive=bool(is_time_sensitive(question=question, sql="")),
            )
        else:
            # Check global cache first, then session cache
            cached_sql, cached_df = _GLOBAL_QUERY_CACHE.find(question, embedder, db=db)
            if cached_sql is None:
                cached_sql, cached_df = find_cached_result(question, query_cache, embedder, db=db)
        timer.mark("cache_lookup")
        if cached_sql is not None:
            log_event(logger, logging.INFO, "query_cache_hit", scope="global_or_session")
            _update_session_query_state(session_state, cached_sql, active_dialect, question=question)
            results = _results_to_records(cached_df, places=2)
            timer.mark("results_formatting")
            return _finalize_result({
                "intent": "DATA_QUERY",
                "nl_answer": None,
                "sql": cached_sql,
                "results": results,
                "row_count": len(results),
                "from_cache": True,
                "error": None,
                "updated_state": session_state,
                    "nl_pending": True,
                })
        else:
            log_event(logger, logging.INFO, "query_cache_miss", scope="global_and_session")

        # CHANGED: structured state is primary prompt memory; raw text is secondary.
        prompt_context = _build_prompt_context_from_state(session_state, conversation_context, conversation_turns)

        t_sql_gen = time.perf_counter()
        # Retrieve few-shot examples from RAG for better SQL generation
        few_shot_str = ""
        rag_table_hints: List[str] = []
        rag_context: Dict[str, Any] = {}
        q_word_count = len(re.findall(r"[A-Za-z0-9_]+", question or ""))
        skip_rag_retrieval = bool(
            is_sort_request
            or is_filter_mod
            or _is_short_contextual_followup(question)
            or _is_bare_topn_followup(question)
            or is_followup_question(question)
            or q_word_count < 7
        )
        if rag_engine and not skip_rag_retrieval:
            try:
                rag_context = rag_engine.retrieve(
                    question,
                    top_k=2,
                    fast_mode=True,
                    skip_for_followup=False,
                    intent_key=_extract_query_topic(question, previous_sql or ""),
                )
                examples = rag_context.get("examples", [])[:3]
                rag_table_hints = list(rag_context.get("tables", []) or [])
                if examples:
                    parts = ["EXAMPLE QUERIES (follow these patterns closely):"]
                    for ex in examples:
                        q = ex.get("question", "")
                        s = ex.get("sql", "")
                        if q and s:
                            parts.append(f"Q: {q}\nSQL: {s}")
                    few_shot_str = "\n\n".join(parts)
                    logger.info(f"Injected {len(examples)} few-shot examples from RAG")
            except Exception:
                logger.warning("RAG retrieval failed for few-shot examples", exc_info=True)
        elif rag_engine and skip_rag_retrieval:
            logger.info("Skipped RAG retrieval for follow-up/short-context query")
        timer.mark("rag_retrieval")

        prompt_payload = _build_prompt_payload(
            question=question,
            db=db,
            schema_text=full_schema_text,
            stored_guidance=stored_procedure_guidance,
            conversation_context=conversation_context,
            prompt_context=prompt_context,
            few_shot_examples=few_shot_str,
            session_state=session_state,
            rag_table_hints=rag_table_hints,
        )
        effective_schema_text = prompt_payload["schema_text"]
        effective_guidance = prompt_payload["stored_guidance"]
        effective_prompt_context = prompt_payload["prompt_context"]
        effective_few_shot = prompt_payload["few_shot_examples"]
        retrieved_tables_hint = ", ".join(rag_table_hints[:6]) if rag_table_hints else (session_state.get("last_table") or "none")

        # ── Complexity-based LLM routing ──
        # deterministic → already handled above (no LLM call at all)
        # simple_llm    → fast deepseek-chat (target <3s)
        # complex_llm   → deepseek-reasoner when available (accepts longer wait for quality)
        # follow-ups    → fast chat model (just modifying existing SQL)
        pre_complexity = _estimate_query_complexity(question)
        use_reasoning = (
            reasoning_llm is not None
            and pre_complexity == "complex_llm"
            and not is_sort_request
            and not is_followup_filter
        )
        if ENABLE_LLM_SLA:
            # SLA mode prioritizes bounded latency; reasoner is intentionally disabled.
            use_reasoning = False
        if use_reasoning and _elapsed_seconds() > 2.0:
            # CHANGED: latency budget guard (<5s target) — avoid slow reasoner on late path.
            logger.info("Skipping reasoning_llm due to latency budget (>2s elapsed)")
            use_reasoning = False

        try:
            fallback_used = False
            fallback_reason = None
            validation_msg = None
            resp_text = ""
            validator_needs_retry = True

            def _try_deterministic_timeout_fallback(reason: str) -> bool:
                nonlocal cleaned_query, is_valid, validation_msg, fallback_used, fallback_reason, fallback_used_request
                fallback_sql, resolved_reason = _deterministic_timeout_fallback_sql(question, db)
                if not fallback_sql:
                    return False
                cleaned_query = fix_common_sql_errors(fallback_sql, dialect=active_dialect)
                is_valid, validation_msg = validate_sql(cleaned_query)
                fallback_used = True
                fallback_used_request = True
                fallback_reason = resolved_reason or reason
                logger.warning(
                    "fallback_used=true reason=%s stage=sql_generation",
                    fallback_reason,
                )
                return True

            if use_reasoning:
                # Build flat prompt string — reasoner works best with a single user message
                reasoning_prompt = SQL_TEMPLATE.format(
                    full_schema=effective_schema_text,
                    stored_procedure_guidance=effective_guidance,
                    context=effective_prompt_context,
                    question=question,
                    retrieved_tables_hint=retrieved_tables_hint,
                    few_shot_examples=effective_few_shot,
                    dialect_label=dialect_label,
                    relative_date_reference=relative_date_reference,
                    enable_nolock=nolock_setting,
                )
                try:
                    reasoning_resp = _invoke_with_timeout(
                        lambda: reasoning_llm.invoke(reasoning_prompt),
                        timeout_ms=LLM_SQL_TIMEOUT_MS,
                    )
                    raw_text = reasoning_resp.content.strip() if hasattr(reasoning_resp, "content") else str(reasoning_resp).strip()
                    # Strip <think>...</think> reasoning block that DeepSeek R1 prepends
                    resp_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                except FuturesTimeoutError:
                    if not _try_deterministic_timeout_fallback("reasoning_timeout"):
                        # Hard timeout fallback for non-deterministic intents: try fast chat model.
                        logger.warning("reasoning_llm timeout; trying fast_llm path")
                        use_reasoning = False
            else:
                # Fast path: use standard chat model via pre-built chain
                pass

            if not fallback_used and not use_reasoning:
                try:
                    resp_text = _invoke_with_timeout(
                        lambda: sql_chain.invoke({
                            "question": question,
                            "context": effective_prompt_context,
                            "few_shot_examples": effective_few_shot,
                            "retrieved_tables_hint": retrieved_tables_hint,
                            "dialect_label": dialect_label,
                            "relative_date_reference": relative_date_reference,
                            "enable_nolock": nolock_setting,
                            "full_schema_override": effective_schema_text,
                            "stored_procedure_guidance_override": effective_guidance,
                        }),
                        timeout_ms=LLM_SQL_TIMEOUT_MS,
                    )
                except FuturesTimeoutError:
                    if not _try_deterministic_timeout_fallback("fast_llm_timeout"):
                        raise

            generation_ms = (time.perf_counter() - t_sql_gen) * 1000
            if not fallback_used and generation_ms > LLM_SQL_TIMEOUT_MS:
                if _try_deterministic_timeout_fallback("slow_generation"):
                    generation_ms = (time.perf_counter() - t_sql_gen) * 1000

            if not fallback_used:
                cleaned_query = _clean_sql_response(resp_text.strip())
                cleaned_query = fix_common_sql_errors(cleaned_query, dialect=active_dialect)
                is_valid, validation_msg = validate_sql(cleaned_query)

                if is_valid:
                    validator_context = {
                        "tables": rag_context.get("tables", []),
                        "examples": (rag_context.get("examples", []) or [])[:2],
                        "rules": (rag_context.get("rules", []) or [])[:3],
                    }
                    validator_result = run_sql_intent_validator(
                        llm=llm,
                        question=question,
                        sql=cleaned_query,
                        dialect_label=dialect_label,
                        full_schema=effective_schema_text,
                        stored_procedure_guidance=effective_guidance,
                        required_table_hints=retrieved_tables_hint,
                        retrieved_context=json.dumps(validator_context, ensure_ascii=True),
                        timeout_ms=min(1500, LLM_SQL_TIMEOUT_MS),
                    )
                    if not validator_result.get("ok_to_execute", False):
                        validator_needs_retry = bool(validator_result.get("needs_retry", True))
                        reasons = validator_result.get("reasons") or []
                        validation_msg = "; ".join(reasons) if reasons else "validator_failed"
                        fixed_sql = validator_result.get("fixed_sql")
                        if isinstance(fixed_sql, str) and fixed_sql.strip():
                            candidate_sql = fix_common_sql_errors(_clean_sql_response(fixed_sql), dialect=active_dialect)
                            fixed_valid, _ = validate_sql(candidate_sql)
                            if fixed_valid:
                                cleaned_query = candidate_sql
                                is_valid = True
                                logger.info(
                                    "validator_applied_fixed_sql failure_type=%s",
                                    validator_result.get("failure_type"),
                                )
                            else:
                                is_valid = False
                        else:
                            is_valid = False

            if use_reasoning and not fallback_used:
                logger.info(f"DeepSeek reasoner generated SQL ({generation_ms:.0f}ms), complexity={pre_complexity}")
            elif not fallback_used:
                logger.info(f"Fast chat model generated SQL ({generation_ms:.0f}ms), complexity={pre_complexity}")
            log_event(
                logger,
                logging.INFO,
                "llm_sql_generation",
                duration_ms=round(generation_ms, 2),
                reasoning=bool(use_reasoning),
                fallback_used=bool(fallback_used),
            )
            timer.mark("sql_generation")

            logger.info(
                "SQL generated (%sms), valid=%s, reasoning=%s, fallback_used=%s",
                round((time.perf_counter() - t_sql_gen) * 1000, 0),
                is_valid,
                use_reasoning,
                fallback_used,
            )
            timer.mark("sql_validation")

            if not is_valid and validator_needs_retry and llm_retry_budget > 0:
                # Retry with stricter prompt — prefer reasoning model for retry too
                llm_retry_budget -= 1
                retry_prompt = RETRY_PROMPT_TEMPLATE.format(
                    question=question,
                    full_schema=effective_schema_text,
                    stored_procedure_guidance=effective_guidance,
                    dialect_label=dialect_label,
                    relative_date_reference=relative_date_reference,
                    enable_nolock=nolock_setting,
                )
                try:
                    active_llm = reasoning_llm if (reasoning_llm is not None and _elapsed_seconds() <= 2.0) else llm
                    retry_resp = _invoke_with_timeout(
                        lambda: active_llm.invoke(retry_prompt),
                        timeout_ms=LLM_SQL_TIMEOUT_MS,
                    )
                    retry_raw = retry_resp.content.strip() if hasattr(retry_resp, "content") else str(retry_resp).strip()
                    retry_raw = re.sub(r"<think>.*?</think>", "", retry_raw, flags=re.DOTALL).strip()
                    retry_sql = _clean_sql_response(retry_raw)
                    retry_sql = fix_common_sql_errors(retry_sql, dialect=active_dialect)
                    is_valid_retry, _ = validate_sql(retry_sql)
                    if is_valid_retry:
                        cleaned_query = retry_sql
                        is_valid = True
                        timer.mark("sql_validation")
                except Exception:
                    logger.warning("SQL retry with stricter prompt failed", exc_info=True)

            # Top-N guardrail: if model returned a single aggregate row, rewrite as grouped ranking SQL.
            if is_valid and should_replace_with_topn_fallback(question, cleaned_query):
                fallback_sql = build_topn_fallback_sql(question, db)
                if fallback_sql:
                    cleaned_query = fallback_sql
                    is_valid = True

            # ID-column guardrail: if SELECT contains raw ID columns,
            # first try deterministic fix (no LLM), then fall back to LLM.
            if is_valid:
                bad_ids = _detect_raw_id_columns_in_select(cleaned_query)
                if bad_ids:
                    # Fast path: deterministic regex-based fix
                    deterministic_fix = _try_deterministic_id_fix(cleaned_query, bad_ids)
                    if deterministic_fix:
                        logger.info("ID-column fix applied deterministically (no LLM call)")
                        cleaned_query = deterministic_fix
                    elif llm_retry_budget > 0:
                        # Slow path: LLM retry
                        llm_retry_budget -= 1
                        logger.info("ID-column fix: deterministic fix failed, falling back to LLM")
                        id_fix_pairs = ", ".join(
                            f"{c} → {_ID_COLUMN_MAP.get(c.lower(), 'name column')}"
                            for c in bad_ids
                        )
                        id_fix_prompt = (
                            f"Return one corrected {_dialect_label(active_dialect)} SELECT query.\n"
                            f"The previous query incorrectly selected raw ID columns: {id_fix_pairs}.\n"
                            f"Replace each ID column with its human-readable name column, adding a JOIN if needed.\n\n"
                            f"SCHEMA:\n{effective_schema_text}\n\n"
                            f"DOMAIN BUSINESS RULES:\n{effective_guidance}\n\n"
                            f"CURRENT SQL (fix it):\n{cleaned_query}\n\n"
                            f"Output format: one SQL code block, optional Notes section.\n"
                            f"FIXED SQL:"
                        )
                        try:
                            fix_resp = _invoke_with_timeout(
                                lambda: llm.invoke(id_fix_prompt),
                                timeout_ms=LLM_SQL_TIMEOUT_MS,
                            )
                            fix_sql = fix_resp.content.strip() if hasattr(fix_resp, "content") else str(fix_resp).strip()
                            fix_sql = _clean_sql_response(fix_sql)
                            fix_sql = fix_common_sql_errors(fix_sql, dialect=active_dialect)
                            fix_valid, _ = validate_sql(fix_sql)
                            if fix_valid:
                                cleaned_query = fix_sql
                                timer.mark("sql_validation")
                        except Exception:
                            logger.warning("ID-column LLM fix failed, keeping original", exc_info=True)

            if not is_valid:
                return _finalize_result({
                    "intent": "DATA_QUERY",
                    "nl_answer": validation_msg,
                    "sql": cleaned_query,
                    "results": None,
                    "row_count": 0,
                    "from_cache": False,
                    "error": "Could not generate valid SQL",
                    "updated_state": session_state,
                })
        except Exception as e:
            err_text = str(e).strip()
            if not err_text and isinstance(e, FuturesTimeoutError):
                err_text = f"LLM request timed out after {LLM_SQL_TIMEOUT_MS} ms"
            if not err_text:
                err_text = e.__class__.__name__
            return _finalize_result({
                "intent": "DATA_QUERY",
                "nl_answer": None,
                "sql": None,
                "results": None,
                "row_count": 0,
                "from_cache": False,
                "error": f"Error generating SQL query: {err_text}",
                "updated_state": session_state,
            })

    if "cache_lookup" not in timer._marks:
        timer.mark("cache_lookup")
    if "rag_retrieval" not in timer._marks:
        timer.mark("rag_retrieval")
    if cleaned_query and is_valid and "sql_generation" not in timer._marks:
        timer.mark("sql_generation")
    if cleaned_query and is_valid and "sql_validation" not in timer._marks:
        timer.mark("sql_validation")

    # Step 4: Execute query
    if cleaned_query and is_valid:
        guardrails_applied: List[str] = []
        cleaned_query = fix_common_sql_errors(cleaned_query, dialect=active_dialect)
        cleaned_query = expand_fuzzy_search(cleaned_query, db=db)
        cleaned_query = apply_stored_procedure_guardrails(cleaned_query, db=db)
        cleaned_query = _retry_ranking_shape_if_needed(
            question=question,
            sql=cleaned_query,
            llm=None,  # Keep retries bounded; rely on deterministic reshapes on hot path.
            full_schema=full_schema_text,
            stored_procedure_guidance=stored_procedure_guidance,
            sql_dialect=active_dialect,
            enable_nolock=enable_nolock,
        )
        cleaned_query = performance_guardrails(
            cleaned_query,
            schema_text=full_schema_text,
            question=question,
            state=session_state,
            db=db,
            dialect=active_dialect,
            enable_nolock=enable_nolock,
        )
        guardrails_applied = _get_last_perf_guardrails()
        timer.mark("guardrails_applied")

        # Adjust timeout based on query complexity
        base_timeout = getattr(db_config, "query_timeout", 30)
        complexity = _estimate_query_complexity(question)
        if complexity == "complex_llm":
            timeout_seconds = max(base_timeout, 60)
        elif complexity == "simple_llm":
            timeout_seconds = min(base_timeout, 20)
        else:
            timeout_seconds = base_timeout
        logger.info(f"Query complexity: {complexity}, timeout: {timeout_seconds}s")

        # Dry-run validation: catch column/table errors before actual execution
        dry_ok, dry_err = validate_sql_dry_run(db, cleaned_query)
        if not dry_ok:
            logger.warning(f"SQL dry-run failed: {dry_err}")
            cleaned_query = fix_common_sql_errors(cleaned_query, dialect=active_dialect)
            dry_ok_after_fix, _ = validate_sql_dry_run(db, cleaned_query)
            if dry_ok_after_fix:
                logger.info("Deterministic SQL fix resolved dry-run error without LLM retry")
            elif llm_retry_budget > 0:
                llm_retry_budget -= 1
                # Retry with error context
                retry_prompt = (
                    f"Return one corrected {_dialect_label(active_dialect)} SELECT query.\n"
                    f"The SQL below has this error: {dry_err}\n"
                    f"Fix the error while preserving the query intent.\n\n"
                    f"SCHEMA:\n{full_schema_text or ''}\n\n"
                    f"BROKEN SQL:\n{cleaned_query}\n\n"
                    f"Output format: one SQL code block, optional Notes section.\n"
                    f"FIXED SQL:"
                )
                try:
                    active_llm = reasoning_llm if (reasoning_llm is not None and _elapsed_seconds() <= 2.0) else llm
                    fix_resp = _invoke_with_timeout(
                        lambda: active_llm.invoke(retry_prompt),
                        timeout_ms=LLM_SQL_TIMEOUT_MS,
                    )
                    fix_raw = fix_resp.content.strip() if hasattr(fix_resp, "content") else str(fix_resp).strip()
                    fix_raw = re.sub(r"<think>.*?</think>", "", fix_raw, flags=re.DOTALL).strip()
                    fix_sql = _clean_sql_response(fix_raw)
                    fix_sql = fix_common_sql_errors(fix_sql, dialect=active_dialect)
                    fix_valid, _ = validate_sql(fix_sql)
                    if fix_valid:
                        cleaned_query = fix_sql
                        cleaned_query = performance_guardrails(
                            cleaned_query,
                            schema_text=full_schema_text,
                            question=question,
                            state=session_state,
                            db=db,
                            dialect=active_dialect,
                            enable_nolock=enable_nolock,
                        )
                        guardrails_applied = _get_last_perf_guardrails()
                        timer.mark("guardrails_applied")
                        logger.info("SQL fixed after dry-run failure")
                except Exception:
                    logger.warning("LLM fix after dry-run failed", exc_info=True)

        t_exec = time.perf_counter()
        df, error = execute_query_safe(db, cleaned_query, timeout_seconds=timeout_seconds, max_rows=1000)
        exec_ms = (time.perf_counter() - t_exec) * 1000
        timer.mark("db_execution")
        total_ms = (time.perf_counter() - t_start) * 1000
        row_count = len(df) if df is not None else 0
        logger.info(f"Query executed ({exec_ms:.0f}ms), rows={row_count}, error={error is not None}, total={total_ms:.0f}ms")
        logger.info("DB execution time ms=%s rows=%s", round(exec_ms, 2), row_count)
        log_event(
            logger,
            logging.INFO,
            "db_execution",
            duration_ms=round(exec_ms, 2),
            row_count=row_count,
            has_error=bool(error),
        )
        if exec_ms > 5000:
            logger.warning(
                "Slow query (>5s). guardrails_applied=%s final_sql=%s",
                guardrails_applied,
                cleaned_query,
            )
        if exec_ms > 3000:
            slow_tables = _extract_tables_for_diagnostics(cleaned_query)
            has_date_filter = _sql_has_any_date_filter(cleaned_query)
            has_wildcard = bool(re.search(r"(?i)\bLIKE\s+N?'%[^']*'", cleaned_query))
            has_non_sargable = _has_non_sargable_date_funcs(cleaned_query)
            logger.warning(
                "SLOW_SQL tables=%s date_filter=%s wildcard=%s non_sargable=%s sql=%s",
                slow_tables,
                has_date_filter,
                has_wildcard,
                has_non_sargable,
                cleaned_query,
            )

        if error:
            # If follow-up filter caused an error or wrong columns, retry with cross-view search
            if is_followup_filter:
                retry_result = _retry_followup_across_views(
                    question,
                    previous_sql,
                    db,
                    db_config,
                    llm,
                    query_cache,
                    embedder,
                    conversation_context,
                    timeout_seconds,
                    sql_dialect=active_dialect,
                    enable_nolock=enable_nolock,
                )
                if retry_result is not None:
                    _update_session_query_state(
                        session_state,
                        retry_result.get("sql"),
                        active_dialect,
                        question=question,
                    )
                    retry_result["updated_state"] = session_state
                    return _finalize_result(retry_result)

            return _finalize_result({
                "intent": "DATA_QUERY",
                "nl_answer": None,
                "sql": cleaned_query,
                "results": None,
                "row_count": 0,
                "from_cache": False,
                "error": error,
                "updated_state": session_state,
            })

        if df is not None and len(df) > 0:
            _update_session_query_state(session_state, cleaned_query, active_dialect, question=question)
            cache_query_result(question, cleaned_query, df, query_cache, embedder, db=db)
            _GLOBAL_QUERY_CACHE.add(question, cleaned_query, df, embedder, db=db)
            results = _results_to_records(df, places=2)
            timer.mark("results_formatting")
            turn = ConversationTurn(
                question=question, sql=cleaned_query,
                topic=_extract_query_topic(question, cleaned_query),
                columns=list(df.columns), row_count=len(df), status="ok",
            )
            return _finalize_result({
                "intent": "DATA_QUERY",
                "nl_answer": None,
                "sql": cleaned_query,
                "results": results,
                "row_count": len(results),
                "from_cache": False,
                "error": None,
                "updated_state": session_state,
                "nl_pending": True,
                "conversation_turn": turn,
            })
        elif (df is None or len(df) == 0) and is_followup_filter and error is None:
            # Follow-up filter returned no results — the value likely belongs
            # to a different view. Retry with cross-view search.
            retry_result = _retry_followup_across_views(
                question,
                previous_sql,
                db,
                db_config,
                llm,
                query_cache,
                embedder,
                conversation_context,
                timeout_seconds,
                sql_dialect=active_dialect,
                enable_nolock=enable_nolock,
            )
            if retry_result is not None:
                _update_session_query_state(
                    session_state,
                    retry_result.get("sql"),
                    active_dialect,
                    question=question,
                )
                return _finalize_result(retry_result)
            # Both attempts returned nothing
            _update_session_query_state(session_state, cleaned_query, active_dialect, question=question)
            cache_query_result(question, cleaned_query, None, query_cache, embedder, db=db)
            timer.mark("results_formatting")
            return _finalize_result({
                "intent": "DATA_QUERY",
                "nl_answer": "No results found for your query.",
                "sql": cleaned_query,
                "results": [],
                "row_count": 0,
                "from_cache": False,
                "error": None,
                "updated_state": session_state,
            })
        else:
            _update_session_query_state(session_state, cleaned_query, active_dialect, question=question)
            cache_query_result(question, cleaned_query, None, query_cache, embedder, db=db)
            timer.mark("results_formatting")
            return _finalize_result({
                "intent": "DATA_QUERY",
                "nl_answer": "No results found for your query.",
                "sql": cleaned_query,
                "results": [],
                "row_count": 0,
                "from_cache": False,
                "error": None,
                "updated_state": session_state,
            })

    # Should not reach here
    return _finalize_result({
        "intent": "DATA_QUERY",
        "nl_answer": None,
        "sql": None,
        "results": None,
        "row_count": 0,
        "from_cache": False,
        "error": "Unexpected error processing query",
        "updated_state": session_state,
    })


def _get_last_sql(chat_history: List[Dict]) -> Optional[str]:
    for chat in reversed(chat_history):
        if chat.get("sql") and chat["sql"].strip().upper().startswith("SELECT"):
            return chat["sql"]
    return None


def _get_last_result_df(chat_history: List[Dict]):
    for chat in reversed(chat_history):
        if chat.get("result_df") is not None:
            return chat["result_df"]
        # New format: result_rows is a list[dict] (no pandas in chat hot path)
        rows = chat.get("result_rows")
        if rows:
            try:
                return pd.DataFrame(rows)
            except Exception:
                pass
    return None


def _get_previous_topic(chat_history: List[Dict]) -> str:
    """Get the previous question for context in retry."""
    for chat in reversed(chat_history):
        if chat.get("question"):
            return chat["question"]
    return "bookings"


def _extract_search_term(question: str) -> Optional[str]:
    """Extract the likely search term from a follow-up question like 'what about barcelona?'"""
    q_lower = question.lower().strip().rstrip("?").rstrip(".")
    # Remove common follow-up prefixes
    for prefix in ["what about", "how about", "and what about", "show me", "show", "and", "what of", "how is"]:
        if q_lower.startswith(prefix):
            q_lower = q_lower[len(prefix):].strip()
            break
    # Remove trailing words like "doing", "performing"
    for suffix in ["doing", "performing", "looking"]:
        if q_lower.endswith(suffix):
            q_lower = q_lower[:-(len(suffix))].strip()
    return q_lower if q_lower else None


def _build_retry_search_config(db: SQLDatabase) -> List[Dict[str, Any]]:
    profile = _get_schema_profile(db)
    configs = []

    for table_entry in profile["tables"]:
        search_cols = _infer_text_columns(table_entry)
        if not search_cols:
            continue

        metric_exprs = _metric_exprs_for_table(table_entry)
        if not metric_exprs["has_group_metric"]:
            continue

        group_cols = search_cols[:2]
        group_cols_quoted = [_quote_sql_identifier(col) for col in group_cols]
        select_parts = group_cols_quoted + [
            f"COALESCE({metric_exprs['bookings_expr']}, 0) AS bookings",
            f"COALESCE({metric_exprs['revenue_expr']}, 0) AS revenue",
            f"COALESCE({metric_exprs['profit_expr']}, 0) AS profit",
        ]

        configs.append(
            {
                "table": table_entry["table"],
                "table_normalized": table_entry["table_normalized"],
                "search_cols": search_cols[:3],
                "group_cols": group_cols_quoted,
                "select_clause": ", ".join(select_parts),
            }
        )

    return configs


def _retry_followup_across_views(
    question: str,
    previous_sql: str,
    db,
    db_config,
    llm,
    query_cache: Any,
    embedder,
    conversation_context: str,
    timeout_seconds: int,
    sql_dialect: str = DEFAULT_SQL_DIALECT,
    enable_nolock: bool = False,
) -> Optional[Dict[str, Any]]:
    """When a follow-up filter returns no results or errors, try searching across all views."""
    search_term = _extract_search_term(question)
    if not search_term or len(search_term) < 2:
        return None

    # Detect the original view so we skip it (already tried)
    original_view = _extract_from_table_name(previous_sql or "") or ""
    search_term_sql = search_term.replace("'", "''")

    for cfg in _build_retry_search_config(db):
        if cfg["table_normalized"] == original_view:
            continue  # Already tried this view
        try:
            where_parts = [f"{_quote_sql_identifier(col)} LIKE '%{search_term_sql}%'" for col in cfg["search_cols"]]
            where_clause = " OR ".join(where_parts)
            table_name = _quote_sql_identifier(cfg["table"])
            group_by = ", ".join(cfg["group_cols"])
            retry_sql = (
                f"SELECT TOP 20 {cfg['select_clause']} "
                f"FROM {table_name} "
                f"WHERE {where_clause} "
                f"GROUP BY {group_by} "
                f"ORDER BY revenue DESC"
            )
            retry_sql = fix_common_sql_errors(retry_sql, dialect=sql_dialect)
            retry_sql = performance_guardrails(
                retry_sql,
                schema_text="",
                question=question,
                db=db,
                dialect=sql_dialect,
                enable_nolock=enable_nolock,
            )
            retry_df, retry_error = execute_query_safe(db, retry_sql, timeout_seconds=timeout_seconds, max_rows=1000)
            if retry_df is not None and len(retry_df) > 0:
                cache_query_result(question, retry_sql, retry_df, query_cache, embedder, db=db)
                results = _results_to_records(retry_df, places=2)
                return {
                    "intent": "DATA_QUERY",
                    "nl_answer": None,
                    "sql": retry_sql,
                    "results": results,
                    "row_count": len(results),
                    "from_cache": False,
                    "error": None,
                    "updated_state": session_state,
                    "nl_pending": True,
                }
        except Exception:
            continue

    return None


def _run_guardrail_self_tests() -> None:
    """Unit-like assertions for SQL Server guardrail behavior."""
    # 1) Aggregate KPI query must not keep/add ORDER BY.
    kpi_sql = (
        "SELECT SUM(AgentBuyingPrice) AS revenue "
        "FROM BookingData "
        "WHERE CreatedDate >= '2026-02-01' AND CreatedDate < '2026-03-01' "
        "ORDER BY CreatedDate DESC"
    )
    kpi_out = performance_guardrails(kpi_sql, question="overall revenue", dialect="sqlserver")
    assert "ORDER BY" not in kpi_out.upper(), "Aggregate KPI query should not contain ORDER BY"
    assert "TOP (200)" not in kpi_out.upper(), "Aggregate KPI query should not add TOP (200)"

    # 2) Detail list query should add TOP + ORDER BY.
    detail_sql = "SELECT ProductName, CreatedDate FROM BookingData WHERE BookingStatus = 'Confirmed'"
    detail_out = performance_guardrails(detail_sql, question="list bookings", dialect="sqlserver")
    assert "TOP (200)" in detail_out.upper(), "Detail query should include TOP (200)"
    assert re.search(r"(?i)\bORDER\s+BY\s+CreatedDate\s+DESC\b", detail_out), "Detail query should include ORDER BY CreatedDate DESC"

    # 3) Existing date filter should not inject another default month range.
    existing_date_sql = (
        "SELECT ProductName FROM BookingData "
        "WHERE CreatedDate >= '2026-02-01' AND CreatedDate < '2026-03-01'"
    )
    existing_out = performance_guardrails(existing_date_sql, question="bookings in feb 2026", dialect="sqlserver")
    assert "DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)" not in existing_out, "Should not inject duplicate default date window"


if __name__ == "__main__":
    _run_guardrail_self_tests()
    print("Guardrail self-tests passed")
