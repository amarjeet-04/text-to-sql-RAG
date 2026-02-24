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
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import OrderedDict

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
from backend.services.session import ConversationTurn, serialize_conversation_turns

logger = logging.getLogger("sql_engine")

# --- Constants ---
FOLLOW_UP_WORDS = {"that", "those", "it", "them", "this", "more", "instead", "above"}
DATABASE_ENTITY_WORDS = {
    "country", "city", "supplier", "agent", "product", "hotel", "chain", "booking",
    "nationality", "customer", "region", "sales", "revenue", "profit", "date", "amount",
    "checkin", "checkout", "details", "information", "data", "records", "rows",
    "filter", "show", "get", "list",
}
MAX_CACHE_SIZE = 50
BLOCKED_KEYWORDS = {"drop", "delete", "truncate", "insert", "update", "alter", "create", "exec", "execute", "grant", "revoke"}
DEFAULT_SQL_DIALECT = "sqlserver"

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


# --- Global Schema Cache (cross-session) ---
# Avoids re-loading schema + rebuilding FAISS index when reconnecting to the same DB.
_GLOBAL_SCHEMA_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_GLOBAL_SCHEMA_CACHE_TTL = 3600  # 1 hour


def _get_global_schema_cache(host: str, port: str, database: str) -> Optional[Dict[str, Any]]:
    """Return cached schema data if fresh, else None."""
    key = (host.lower(), str(port), database.lower())
    cached = _GLOBAL_SCHEMA_CACHE.get(key)
    if cached and (time.time() - cached["timestamp"]) < _GLOBAL_SCHEMA_CACHE_TTL:
        logger.info(f"Global schema cache HIT for {host}:{port}/{database}")
        return cached
    return None


def _set_global_schema_cache(host: str, port: str, database: str, schema_text: str, rag_engine, embedder):
    """Store schema data in global cache."""
    key = (host.lower(), str(port), database.lower())
    _GLOBAL_SCHEMA_CACHE[key] = {
        "schema_text": schema_text,
        "rag_engine": rag_engine,
        "embedder": embedder,
        "timestamp": time.time(),
    }
    logger.info(f"Global schema cache SET for {host}:{port}/{database}")


# --- Global Query Cache (cross-session, with TTL) ---
# Serves repeated questions without LLM or DB calls.

class GlobalQueryCache:
    """Thread-safe, TTL-based query result cache shared across all sessions."""

    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict = OrderedDict()  # key: question_hash -> {sql, df, embedding, timestamp}

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, v in self._cache.items() if now - v["timestamp"] > self.ttl]
        for k in expired:
            del self._cache[k]

    def find(self, question: str, embedder) -> Tuple[Optional[str], Optional[Any]]:
        """Find a cached result by semantic similarity. Returns (sql, DataFrame) or (None, None)."""
        self._evict_expired()
        if not self._cache or embedder is None:
            return None, None

        try:
            q_embedding = np.array(embedder.embed_query(question), dtype=np.float32)
        except Exception:
            return None, None

        best_score = -1.0
        best_entry = None
        for entry in self._cache.values():
            cached_emb = entry.get("embedding")
            if cached_emb is None:
                continue
            score = float(np.dot(q_embedding, cached_emb) / (np.linalg.norm(q_embedding) * np.linalg.norm(cached_emb) + 1e-9))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= 0.95:
            logger.info(f"Global query cache HIT (score={best_score:.3f})")
            return best_entry["sql"], best_entry.get("df")
        return None, None

    def add(self, question: str, sql: str, df, embedder):
        """Add a query result to the global cache."""
        if embedder is None:
            return
        try:
            embedding = np.array(embedder.embed_query(question), dtype=np.float32)
        except Exception:
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        key = hash(question)
        self._cache[key] = {
            "question": question,
            "sql": sql,
            "df": df,
            "embedding": embedding,
            "timestamp": time.time(),
        }


_GLOBAL_QUERY_CACHE = GlobalQueryCache()


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
- Default to SQL Server unless the session explicitly sets Postgres.
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

ASIA/KOLKATA REFERENCE RANGES (use these to resolve relative dates to explicit ranges):
{relative_date_reference}

USER QUESTION:
{question}

OPTIMIZATION RULES (must follow):
- Filter early and project only required columns (no SELECT *).
- Write SARGable predicates: never apply functions/casts to indexed columns in WHERE.
- Use range predicates: col >= start AND col < end.
- Prefer EXISTS over IN for subquery membership checks.
- Avoid unnecessary subqueries/CTEs unless they reduce repeated work.
- Avoid DISTINCT unless required.
- Join order: start from the most selective filtered table, then join outward.
- Use GROUP BY only for necessary columns; aggregate after filtering.
- Prefer index-friendly comparisons and equality joins on keys.
- String filters: use `=` by default; use `LIKE 'value%'` for prefix intent; use `LIKE '%value%'` only when user explicitly asks for contains/substring.
- Do not use NOLOCK unless this session explicitly enables it.
- Session NOLOCK enabled: {enable_nolock}

CONVERSATION MEMORY (must follow):
- If the user references "same", "above", "that query", reuse the last query intent/tables/filters and apply only the delta.
- Resolve relative dates (today/this month/yesterday/last week) into explicit date ranges using Asia/Kolkata.
- Use only provided schema context; never invent table/column names.
- If the exact mapping is missing, choose the closest known mapping and state it in Notes.

QUALITY CHECK (silent):
- Predicates are SARGable
- No redundant scans
- Correct GROUP BY
- Correct date range boundaries
- SQL matches the active dialect

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
        conn = _raw_conn_from_engine(db)
        cursor = conn.cursor()
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
        conn.close()
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
}


def _compute_text_hash(text: str) -> int:
    return hash(text or "")


def _read_stored_procedure_file(path: Path = STORED_PROCEDURE_FILE) -> str:
    cache = _STORED_PROCEDURE_CACHE
    cache["path"] = str(path)
    try:
        stat = path.stat()
        mtime_ns = int(stat.st_mtime_ns)
    except Exception:
        # File missing/unreadable -> keep runtime safe.
        cache.update({
            "mtime_ns": None,
            "raw_text": "",
            "raw_hash": _compute_text_hash(""),
            "guidance": None,
        })
        return ""

    if cache.get("mtime_ns") == mtime_ns and cache.get("raw_text") is not None:
        return cache.get("raw_text") or ""

    try:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raw_text = ""

    cache.update({
        "mtime_ns": mtime_ns,
        "raw_text": raw_text,
        "raw_hash": _compute_text_hash(raw_text),
        "guidance": None,  # Invalidate derived guidance when file changes.
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

def build_stored_procedure_guidance(raw_text: str) -> str:
    """
    Build comprehensive domain guidance from stored_procedure.txt.

    This replaces _build_stored_procedure_guidance(). It dynamically extracts
    ALL business rules so nothing is hardcoded in the prompt. The LLM gets:
      - Exact metric formulas
      - Mandatory status exclusions
      - Date column mapping
      - Per-dimension join patterns and key columns
      - Hotel chain normalization logic
      - Common query patterns

    CHANGED: Guidance cache is tied to file/content fingerprint so it refreshes
    automatically when stored_procedure.txt changes.
    """
    cache = _STORED_PROCEDURE_CACHE
    raw_hash = _compute_text_hash(raw_text)
    if cache.get("guidance") is not None and cache.get("raw_hash") == raw_hash:
        return cache["guidance"]
    cache["raw_hash"] = raw_hash

    sections = _parse_stored_procedure_sections(raw_text)
    if not sections:
        cache["guidance"] = "No stored procedure guidance available."
        cache["raw_text"] = raw_text or ""
        return "No stored procedure guidance available."

    rules = _extract_domain_rules(sections)
    lines = []

    # ── Metric Definitions ──
    lines.append("=== METRIC DEFINITIONS (use these exact formulas on base tables) ===")
    for label, key in [("Revenue/Sales", "revenue"), ("Cost/Expense", "cost"),
                       ("Profit/Margin", "profit"), ("Booking Count", "bookings")]:
        if key in rules["metrics"]:
            formula, alias = rules["metrics"][key]
            lines.append(f"• {label}: {formula} AS {alias}")
    lines.append("• On pre-aggregated *_level_view tables, use SUM(total_sales)/SUM(total_profit)/SUM(total_booking) instead.")
    lines.append("")

    # ── Extended Metric Definitions (from Excel dashboard formulas) ──
    lines.append("=== EXTENDED METRIC DEFINITIONS (additional dashboard formulas) ===")
    lines.append("• Total Room Nights:         SUM(DATEDIFF(DAY, CheckInDate, CheckOutDate)) AS total_room_nights")
    lines.append("• Avg Booking Window:        AVG(DATEDIFF(DAY, CreatedDate, CheckInDate)) AS avg_booking_window")
    lines.append("• Avg Booking Value:         SUM(AgentBuyingPrice) / NULLIF(COUNT(DISTINCT PNRNo), 0) AS avg_booking_value")
    lines.append("• Last Minute Bookings:      COUNT(DISTINCT CASE WHEN DATEDIFF(DAY, CreatedDate, CheckInDate) <= 1 THEN PNRNo END) AS last_minute_bookings")
    lines.append("• Cancelled Bookings:        COUNT(DISTINCT CASE WHEN BookingStatus = 'Cancelled' THEN PNRNo END) AS cancelled_bookings")
    lines.append("• Non-Refundable Bookings:   COUNT(DISTINCT CASE WHEN DATEDIFF(DAY, CreatedDate, CancellationDeadLine) <= 0 THEN PNRNo END) AS non_refundable_bookings")
    lines.append("• Pax Count (Adults):        SUM(NoofAdult) AS total_adults")
    lines.append("• Pax Count (Children):      SUM(NoofChild) AS total_children")
    lines.append("NOTE: Room nights and booking window use raw date columns, NOT pre-aggregated views.")
    lines.append("")

    # ── Mandatory Filters ──
    lines.append("=== MANDATORY FILTERS ===")
    if rules["status_exclusions"]:
        vals = ", ".join(f"'{v}'" for v in rules["status_exclusions"])
        lines.append(f"• ALWAYS apply: BookingStatus NOT IN ({vals})")
        lines.append("  Required whenever BookingStatus column exists in the queried table.")
    lines.append("• Exception: Cancelled Bookings metric intentionally includes Cancelled status — omit the filter for that metric only.")
    lines.append("")

    # ── Date Columns ──
    lines.append("=== DATE COLUMN USAGE ===")
    lines.append("• Keep date filters SARGable (fast on indexed columns):")
    lines.append("  - NEVER use CAST(date_col AS DATE), YEAR(date_col), MONTH(date_col) in WHERE")
    lines.append("  - Use ranges: date_col >= <start_date> AND date_col < <end_date>")
    lines.append("  - CreatedDate          → booking date (when booked) — DEFAULT for date filters")
    lines.append("  - CheckInDate          → check-in / travel / stay start date")
    lines.append("  - CheckOutDate         → check-out / departure date")
    lines.append("  - CancellationDeadLine → used only for non-refundable booking calculation")
    lines.append("• Intent mapping:")
    lines.append('  - "booking date" / "created date" / "booked on"  → CreatedDate with date range filter')
    lines.append('  - "travel date" / "check-in" / "stay date"       → CheckInDate with date range filter')
    lines.append('  - "checkout" / "departure"                        → CheckOutDate with date range filter')
    lines.append("• Week date filter (this week / current week):")
    lines.append("    CreatedDate >= DATEADD(DAY, -(DATEPART(WEEKDAY, CAST(GETDATE() AS DATE))-1), CAST(GETDATE() AS DATE))")
    lines.append("    AND CreatedDate < DATEADD(DAY, 7-(DATEPART(WEEKDAY, CAST(GETDATE() AS DATE))-1), CAST(GETDATE() AS DATE))")
    lines.append("• Month date filter (this month):")
    lines.append("    CreatedDate >= DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)")
    lines.append("    AND CreatedDate < DATEADD(MONTH, 1, DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1))")
    lines.append("• Last week date filter (Monday-to-Sunday of previous week):")
    lines.append("    date_col >= DATEADD(DAY, -(DATEPART(WEEKDAY, CAST(GETDATE() AS DATE)) - 1) - 7, CAST(GETDATE() AS DATE))")
    lines.append("    AND date_col < DATEADD(DAY, -(DATEPART(WEEKDAY, CAST(GETDATE() AS DATE)) - 1), CAST(GETDATE() AS DATE))")
    lines.append("• This week date filter (Monday of current week to Sunday):")
    lines.append("    date_col >= DATEADD(DAY, -(DATEPART(WEEKDAY, CAST(GETDATE() AS DATE)) - 1), CAST(GETDATE() AS DATE))")
    lines.append("    AND date_col < DATEADD(DAY, 7 - (DATEPART(WEEKDAY, CAST(GETDATE() AS DATE)) - 1), CAST(GETDATE() AS DATE))")
    lines.append("")

    # ── SQL Optimization Rules ──
    lines.append("=== SQL OPTIMIZATION (MANDATORY for large datasets) ===")
    lines.append("• Use CTE with DISTINCT for ALL lookup table joins — these tables have duplicates:")
    lines.append("  AgentMaster_V1, suppliermaster_Report, Master_Country, Master_City, Hotelchain")
    lines.append("  Pattern: WITH AM AS (SELECT DISTINCT col1,col2 FROM table)")
    lines.append("• For ranking (TOP N): aggregate in a CTE, then TOP N + ORDER BY in outer SELECT.")
    lines.append("• Put the largest table (BookingData) as driving table in FROM, LEFT JOIN smaller lookup tables.")
    lines.append("• ONLY return the metric the user asked for. 'cost wise' → ONLY SUM(CompanyBuyingPrice).")
    lines.append("• Result column aliases MUST be user-friendly: 'Agent Name', 'Total Cost', 'Total Revenue'.")
    lines.append("")

    # ── BookingData Column Reference ──
    lines.append("=== dbo.BookingData — COLUMN REFERENCE ===")
    lines.append("USED columns (reference these in queries):")
    lines.append("  PNRNo              — unique booking identifier; use COUNT(DISTINCT PNRNo) for booking count")
    lines.append("  AgentBuyingPrice   — revenue/sales amount: SUM(AgentBuyingPrice)")
    lines.append("  CompanyBuyingPrice — cost amount: SUM(CompanyBuyingPrice)")
    lines.append("  BookingStatus      — filter: NOT IN ('Cancelled','Not Confirmed','On Request')")
    lines.append("  CreatedDate        — booking creation date (DEFAULT date column)")
    lines.append("  CheckInDate        — hotel check-in date")
    lines.append("  CheckOutDate       — hotel check-out date")
    lines.append("  CancellationDeadLine — used for non-refundable bookings calculation")
    lines.append("  ProductName        — hotel name (direct on BookingData)")
    lines.append("  ProductId          — joins to Hotelchain.HotelId for chain queries")
    lines.append("  ServiceName        — service type (Hotel, Meal, Transfer, etc.)")
    lines.append("  AgentId            — FK → AgentMaster_V1.AgentId AND AgentTypeMapping.AgentId")
    lines.append("  SupplierId         — FK → suppliermaster_Report.EmployeeId")
    lines.append("  ProductCountryid   — FK → Master_Country.CountryID")
    lines.append("  ProductCityId      — FK → Master_City.CityId")
    lines.append("  ClientNatinality   — client nationality (direct; note spelling: 'Natinality')")
    lines.append("  NoofAdult          — number of adults (SUM for pax count)")
    lines.append("  NoofChild          — number of children (SUM for pax count)")
    lines.append("NOT USED (never reference these): MasterId, DetailId, SubUserId, CreditcardCharges,")
    lines.append("  PaymentType, IsPackage, AgentReferenceNo, CurrencyId, AgentSellingPrice,")
    lines.append("  IsSameCurrency, SubAgentSellingPrice, RateOfexchange, SellingRateOfexchange,")
    lines.append("  SupplierRateOfexchange, SupplierCurrencyId, RoomTypeName, Provider, OfferCode,")
    lines.append("  OfferDescription, LoyaltyPoints, OTHPromoCode, IsXMLSupplierBooking, PackageId,")
    lines.append("  PackageName, EntryDate, BranchId")
    lines.append("")

    # ── Lookup Table Schemas ──
    lines.append("=== LOOKUP TABLE SCHEMAS ===")
    lines.append("• AgentMaster_V1 — join on: BD.AgentId = A.AgentId")
    lines.append("  Columns: AgentId, AgentCode, AgentName, AgentCountry, AgentCity")
    lines.append("")
    lines.append("• AgentTypeMapping — join on: BD.AgentId = AM.AgentId")
    lines.append("  Columns: AgentId, AgentName, AgentCode, AgentType (values: API / B2B / We Groups)")
    lines.append("  Special: agentcode='WEGR' must be mapped to 'We Groups' via CASE WHEN")
    lines.append("")
    lines.append("• suppliermaster_Report — join on: BD.SupplierId = D.EmployeeId")
    lines.append("  Key column: suppliername")
    lines.append("")
    lines.append("• Master_Country — join on: BD.ProductCountryid = MCO.CountryID")
    lines.append("  Key column: Country")
    lines.append("")
    lines.append("• Master_City — join on: BD.ProductCityId = MCI.CityId")
    lines.append("  Key column: City")
    lines.append("")
    lines.append("• Hotelchain — join on: BD.ProductId = HC.HotelId")
    lines.append("  Columns: HotelId, HotelName, Country, City, Star, Chain")
    lines.append("  Use Chain column with normalization CASE WHEN (see HOTEL CHAIN NORMALIZATION section)")
    lines.append("")

    # ── Dimension Query Patterns ──
    lines.append("=== DIMENSION QUERIES (follow these join patterns) ===")
    for section in sections:
        info = _extract_dimension_info(section)
        lines.append(f"\n• {info['title']}:")
        if info["base_table"]:
            lines.append(f"  Base: {info['base_table']}")
        for j in info["joins"]:
            lines.append(f"  JOIN {j['table']} ON {j['condition']}")
        if info["dimension_cols"]:
            lines.append(f"  Group by: {', '.join(info['dimension_cols'])}")
    lines.append("")

    # ── Hotel Chain Normalization ──
    if rules["chain_case_sql"]:
        lines.append("=== HOTEL CHAIN NORMALIZATION (apply when querying chains) ===")
        lines.append("Normalize chain names using this CASE WHEN logic:")
        # Include full CASE WHEN but cap at reasonable length
        snippet = rules["chain_case_sql"][:700]
        lines.append(snippet)
        lines.append("")

    # ── Dimension Reference Table ──
    lines.append("=== QUICK REFERENCE: DIMENSION → SOURCE ===")
    ref_rows = [
        ("Supplier",           "suppliername",       "suppliermaster_Report", "employeeid = supplierid"),
        ("Agent",              "agentname, agentcode", "agent_level_view (preferred) or BookingData", "—"),
        ("Agent Type / Category", "agenttype (with CASE WHEN for WEGR→'We Groups')", "AgentTypeMapping + AgentMaster_V1 via CTE", "AgentTypeMapping.agentid = BD.agentid; AgentMaster_V1.agentid = BD.agentid"),
        ("Country",            "Country",            "Master_Country",        "CountryID = ProductCountryid"),
        ("City",               "City",               "Master_City",           "CityId = ProductCityId"),
        ("Client Nationality", "ClientNatinality",   "direct on BookingData", "—"),
        ("Hotel / Product",    "productname",        "direct on BookingData", "—"),
        ("Hotel Chain",        "chain (CASE WHEN)",  "Hotelchain",            "hotelid = ProductId"),
    ]
    for dim, cols, source, join_key in ref_rows:
        lines.append(f"  {dim}: columns=[{cols}], source={source}, join={join_key}")
    lines.append("")

    # ── ID → Name column mapping ──
    lines.append("=== NEVER SELECT THESE ID COLUMNS — USE NAME COLUMNS INSTEAD ===")
    lines.append("If you see an ID column in SELECT, replace with its name column (join if needed):")
    id_name_rows = [
        ("AgentId",          "AgentName",      "AgentMaster_V1",          "AgentMaster_V1.AgentId = BD.AgentId"),
        ("ProductId",        "ProductName",    "dbo.BookingData (direct)", "BD.ProductName is already on BookingData"),
        ("HotelId",          "HotelName",      "Hotelchain",              "Hotelchain.HotelId = BD.ProductId"),
        ("CountryId / ProductCountryid", "Country", "Master_Country",      "Master_Country.CountryID = BD.ProductCountryid"),
        ("CityId / ProductCityId",       "City",    "Master_City",         "Master_City.CityId = BD.ProductCityId"),
        ("SupplierId / EmployeeId",      "suppliername", "suppliermaster_Report", "suppliermaster_Report.EmployeeId = BD.SupplierId"),
    ]
    for id_col, name_col, table, join_hint in id_name_rows:
        lines.append(f"  {id_col} → use {name_col} from {table}  [{join_hint}]")
    lines.append("")

    # ── Agent Type CTE Pattern ──
    lines.append("=== AGENT TYPE QUERY PATTERN ===")
    lines.append("ONLY use this pattern when the question explicitly asks for 'agent type', 'agent category', or 'B2B/API/We Groups breakdown'. Do NOT use for hotel, supplier, country, city, growth, YoY, or any other dimension.")
    lines.append("""WITH AM AS (
    SELECT DISTINCT agentid, agentname, agenttype FROM AgentTypeMapping
),
AMV AS (
    SELECT DISTINCT AgentId, AgentCode, AgentName FROM AgentMaster_V1
),
ag_type AS (
    SELECT
        CASE WHEN AMV.agentcode = 'WEGR' THEN 'We Groups' ELSE AM.agenttype END AS agenttype,
        COUNT(DISTINCT BD.PNRNo) AS total_bookings
    FROM dbo.BookingData BD
    LEFT JOIN AM ON AM.agentid = BD.agentid
    LEFT JOIN AMV ON AMV.agentid = BD.agentid
    WHERE <date_filter>
      AND BookingStatus NOT IN ('Cancelled', 'Not Confirmed', 'On Request')
    GROUP BY AMV.agentcode, AM.agenttype
)
SELECT agenttype, SUM(total_bookings) AS total_bookings
FROM ag_type
GROUP BY agenttype
ORDER BY total_bookings DESC""")
    lines.append("Replace <date_filter> with the appropriate SARGable CreatedDate range for the requested period.")
    lines.append("For 'this week': CreatedDate >= DATEADD(DAY, -(DATEPART(WEEKDAY, CAST(GETDATE() AS DATE))-1), CAST(GETDATE() AS DATE))")
    lines.append("                 AND CreatedDate < DATEADD(DAY, 7-(DATEPART(WEEKDAY, CAST(GETDATE() AS DATE))-1), CAST(GETDATE() AS DATE))")
    lines.append("")

    # ── YoY / Growth Pattern ──
    lines.append("=== YoY / GROWTH QUERY PATTERN (use for: growth, compared to last year, YTD vs PYTD, increase/decrease) ===")
    lines.append("IMPORTANT: Use SUM(CASE WHEN ... THEN metric END) inside a single CTE — do NOT use FULL OUTER JOIN or two separate year filters.")
    lines.append("YTD  = CreatedDate >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1) AND CreatedDate < DATEADD(DAY, 1, CAST(GETDATE() AS DATE))")
    lines.append("PYTD = CreatedDate >= DATEFROMPARTS(YEAR(GETDATE())-1, 1, 1) AND CreatedDate < DATEADD(DAY, 1, DATEADD(YEAR, -1, CAST(GETDATE() AS DATE)))")
    lines.append("Outer WHERE (SARGABLE — never wrap CreatedDate in CAST for filtering): CreatedDate >= DATEFROMPARTS(YEAR(GETDATE())-1,1,1) AND CreatedDate < DATEFROMPARTS(YEAR(GETDATE())+1,1,1)")
    lines.append("Growth_Percentage = (YTD_Sales - PYTD_Sales) * 100.0 / NULLIF(PYTD_Sales, 0)")
    lines.append("Filter final SELECT: WHERE YTD_Sales IS NOT NULL  [only show hotels/entities with current-year sales]")
    lines.append("ORDER BY Growth_Percentage DESC for highest growth; ASC for declining.")
    lines.append("Apply same BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') inside each CASE WHEN block.")
    lines.append("Adapt dimension column per entity: productname (hotel), suppliername (supplier), Country (country), City (city), AgentName (agent).")
    lines.append("")

    # ── AGENT RANKING CTE PATTERN (reference for all ranking queries) ──
    lines.append("=== AGENT RANKING CTE PATTERN (use for ALL top/bottom N agent queries) ===")
    lines.append("""WITH AM AS (
  SELECT DISTINCT AgentId, AgentCode, AgentName, AgentCountry, AgentCity
  FROM [dbo].[AgentMaster_V1]
),
main AS (
  SELECT DISTINCT AM.AgentName AS [Agent Name],
    SUM(BD.CompanyBuyingPrice) AS [Total Cost]
  FROM dbo.BookingData BD
  LEFT JOIN AM ON AM.AgentId = BD.AgentId
  WHERE BD.CheckInDate >= <travel_date_start>
    AND BD.CheckInDate < <travel_date_end>
    AND BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')
  GROUP BY AM.AgentName
)
SELECT TOP 15 [Agent Name], [Total Cost]
FROM main
ORDER BY [Total Cost] DESC""")
    lines.append("ADAPT: replace metric (Total Cost→Total Revenue using SUM(AgentBuyingPrice)), date column, and TOP N as needed.")
    lines.append("ADAPT: replace CheckInDate with CreatedDate if user says 'booking date' instead of 'travel date'.")
    lines.append("")

    # ── Common Patterns ──
    lines.append("=== COMMON QUERY PATTERNS ===")
    lines.append("• Ranking with CTE: WITH dim AS (...DISTINCT...), main AS (SELECT ... GROUP BY ...) SELECT TOP N ... FROM main ORDER BY ...")
    lines.append("• ONLY return the metric the user asked for. 'cost wise' = ONLY SUM(CompanyBuyingPrice). Do NOT add revenue/profit/bookings.")
    lines.append("• Time filter: WHERE CreatedDate >= 'YYYY-01-01'")
    lines.append("• Single-day: WHERE CreatedDate >= 'YYYY-MM-DD' AND CreatedDate < DATEADD(DAY, 1, 'YYYY-MM-DD') — filter only, do NOT group by date")
    lines.append("• Current year: WHERE CreatedDate >= DATEFROMPARTS(YEAR(GETDATE()),1,1) AND CreatedDate < DATEFROMPARTS(YEAR(GETDATE())+1,1,1)")

    cache["guidance"] = "\n".join(lines)
    cache["raw_text"] = raw_text or ""
    return cache["guidance"]


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
    cached = _SCHEMA_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        usable_tables = list(db.get_usable_table_names())
    except Exception:
        profile = {"tables": [], "lookup": {}}
        _SCHEMA_PROFILE_CACHE[cache_key] = profile
        return profile

    # Use bulk INFORMATION_SCHEMA query instead of per-table inspector calls.
    # On remote DBs this is ~100x faster (0.3s vs 30s+).
    from collections import defaultdict
    table_cols_raw: Dict[str, list] = defaultdict(list)

    try:
        conn = _raw_conn_from_engine(db)
        cursor = conn.cursor()
        placeholders = ",".join([f"'{t}'" for t in usable_tables])
        cursor.execute(
            f"SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
            f"FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_NAME IN ({placeholders}) "
            f"ORDER BY TABLE_NAME, ORDINAL_POSITION"
        )
        for row in cursor.fetchall():
            table_cols_raw[row[0]].append({"name": row[1], "type": row[2]})
        conn.close()
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
    cached = _TABLE_EXISTS_CACHE.get(key)
    if cached is not None:
        return cached

    profile = _get_schema_profile(db)
    for entry in profile.get("tables", []):
        if entry.get("table_lower") == table_name.lower() or entry.get("table_normalized") == table_name.lower():
            _TABLE_EXISTS_CACHE[key] = True
            return True

    exists = False
    try:
        conn = _raw_conn_from_engine(db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?",
            (table_name,),
        )
        exists = cursor.fetchone() is not None
        conn.close()
    except Exception:
        exists = False

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


def generate_nl_response(question: str, df: Optional[pd.DataFrame], llm) -> str:
    if llm is None:
        return ""
    try:
        if df is not None and len(df) > 0:
            # Pass up to 15 rows so the LLM can mention top entries by name
            pd.set_option("display.max_colwidth", None)
            results_str = df.head(15).to_string(index=False)
        else:
            results_str = "No results found."
        chain = NL_RESPONSE_TEMPLATE | llm | StrOutputParser()
        return chain.invoke({"question": question, "results": results_str})
    except Exception:
        return ""


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
        "all time", "overall", "historical", "history", "lifetime", "since inception", "ever", "from beginning",
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
        "cte_sql": "SELECT DISTINCT AgentId, AgentCode, AgentName, AgentCountry, AgentCity FROM [dbo].[AgentMaster_V1] WITH (NOLOCK)",
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
            f"  SELECT DISTINCT {cte_name}.{name_col} AS [{friendly_name}],\n"
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
        f"  SELECT DISTINCT {dim_col} AS [{friendly_name}],\n"
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


def find_cached_result(question: str, query_cache: List, embedder, threshold: float = 0.95):
    if not query_cache or embedder is None:
        return None, None
    if is_followup_question(question):
        return None, None

    query_emb = compute_embedding(question, embedder)
    if query_emb is None:
        return None, None

    current_values = extract_key_values(question)
    best_similarity = -1.0
    best_match = None
    best_cached_question = None

    for cached_emb, cached_question, cached_sql, cached_df in query_cache:
        similarity = float(
            np.dot(query_emb, cached_emb)
            / (np.linalg.norm(query_emb) * np.linalg.norm(cached_emb) + 1e-8)
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = (cached_sql, cached_df)
            best_cached_question = cached_question

    if best_similarity >= threshold and best_match is not None:
        cached_values = extract_key_values(best_cached_question)
        if current_values or cached_values:
            if current_values != cached_values:
                return None, None
        return best_match

    return None, None


def cache_query_result(question: str, sql: str, result_df, query_cache: List, embedder):
    # Run embedding + cache insertion in a daemon thread so the caller
    # (handle_query) can return results to the user immediately without
    # waiting ~100-300 ms for the sentence-transformer encode call.
    import threading

    def _do_cache():
        emb = compute_embedding(question, embedder)
        if emb is not None:
            if len(query_cache) >= MAX_CACHE_SIZE:
                query_cache.pop(0)
            query_cache.append((emb, question, sql, result_df))

    t = threading.Thread(target=_do_cache, daemon=True)
    t.start()


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
        conn = _raw_odbc_connect(host, port, username, password, database, timeout=10)
        cursor = conn.cursor()

        # Check for pre-aggregated views first (ideal for text-to-SQL)
        cursor.execute(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.VIEWS "
            "WHERE TABLE_NAME LIKE '%\\_level\\_view' ESCAPE '\\' "
            "ORDER BY TABLE_NAME"
        )
        level_views = [r[0] for r in cursor.fetchall()]

        if level_views:
            conn.close()
            return level_views

        # No level views — find which core business tables actually exist
        cursor.execute(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_TYPE IN ('BASE TABLE', 'VIEW') "
            "ORDER BY TABLE_NAME"
        )
        all_tables = {r[0] for r in cursor.fetchall()}
        conn.close()

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
            conn = _raw_odbc_connect(host, port_num, db_username, db_password, database, timeout=15)
            cursor = conn.cursor()
            tbl_names = list(db.get_usable_table_names())
            placeholders = ",".join([f"'{t}'" for t in tbl_names])
            cursor.execute(
                f"SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
                f"FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_NAME IN ({placeholders}) "
                f"ORDER BY TABLE_NAME, ORDINAL_POSITION"
            )
            tbl_cols = defaultdict(list)
            for r in cursor.fetchall():
                tbl_cols[r[0]].append(f"  {r[1]} {r[2].upper()}")
            conn.close()
            parts = [f"CREATE TABLE {t} (\n" + ",\n".join(tbl_cols[t]) + "\n)" for t in tbl_names if tbl_cols.get(t)]
            cached_schema_text = "\n\n".join(parts)
        except Exception:
            cached_schema_text = ""

    prompt = ChatPromptTemplate.from_template(SQL_TEMPLATE)

    def get_full_schema(_):
        return cached_schema_text

    def get_stored_procedure_guidance(_):
        return build_stored_procedure_guidance(_read_stored_procedure_file())

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

    # sql_chain is kept as fallback when no reasoning model is available
    sql_chain = (
        RunnablePassthrough.assign(
            full_schema=get_full_schema,
            stored_procedure_guidance=get_stored_procedure_guidance,
            few_shot_examples=get_few_shot_examples,
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


def _build_context_entry(question: str, sql: Optional[str], row_count: int,
                         columns: List[str], status: str = "ok") -> str:
    """Build a single context turn string that is informative for the LLM."""
    topic = _extract_query_topic(question, sql or "")
    col_str = ", ".join(columns[:8]) + ("..." if len(columns) > 8 else "")
    sql_snippet = ""
    if sql:
        # Include up to 600 chars so the LLM sees the table, joins AND WHERE filters
        # (300 was too short — date filters in CTE queries appear after char 350)
        sql_snippet = f"\n  SQL: {sql[:600].strip()}{'...' if len(sql) > 600 else ''}"
    if status == "ok" and row_count > 0:
        result_str = f"{row_count} rows, columns=[{col_str}]"
    elif status == "no_rows":
        result_str = "no rows returned"
    elif status == "error":
        result_str = "query failed"
    elif status == "cache":
        result_str = "from cache"
    else:
        result_str = status
    return f"{_CONTEXT_TURN_SEP}\nTopic: {topic}\nQ: {question}{sql_snippet}\nResult: {result_str}\n"


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


_SESSION_QUERY_STATE: Dict[int, Dict[str, Any]] = {}


def _new_session_query_state(dialect: str) -> Dict[str, Any]:
    return {
        # CHANGED: lightweight structured memory for follow-up quality.
        "dialect": normalize_sql_dialect(dialect),
        "last_sql": None,
        "last_table": None,
        "last_time_window": {"start": None, "end": None},
        "last_dimensions": [],
        "last_metrics": [],
        "last_filters": [],
    }


def _get_session_query_state(chat_history: List[Dict], dialect: str) -> Dict[str, Any]:
    key = id(chat_history)
    state = _SESSION_QUERY_STATE.get(key)
    if state is None:
        state = _new_session_query_state(dialect)
        _SESSION_QUERY_STATE[key] = state
    else:
        state["dialect"] = normalize_sql_dialect(dialect)
    return state


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
    conversation_context: str,
    query_cache: List,
    cached_schema_text: str = "",
    conversation_turns: Optional[List] = None,
    reasoning_llm=None,
    sql_dialect: str = DEFAULT_SQL_DIALECT,
    enable_nolock: bool = False,
) -> Dict[str, Any]:
    """
    Process a user question end-to-end.
    Returns dict with: intent, nl_answer, sql, results, row_count, from_cache, error, updated_context, conversation_turn
    """
    t_start = time.perf_counter()
    active_dialect = normalize_sql_dialect(sql_dialect or _detect_sql_dialect_from_db(db))
    dialect_label = _dialect_label(active_dialect)
    relative_date_reference = _build_relative_date_reference()
    nolock_setting = str(bool(enable_nolock)).lower()
    session_state = _get_session_query_state(chat_history, active_dialect)

    def _elapsed_seconds() -> float:
        return time.perf_counter() - t_start

    # Step 1: Detect intent
    t_intent = time.perf_counter()
    intent = detect_intent_simple(question)
    stored_procedure_guidance = build_stored_procedure_guidance(_read_stored_procedure_file())

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

    # Step 2: Handle non-data-query intents
    if intent in {"GREETING", "THANKS", "HELP", "FAREWELL", "OUT_OF_SCOPE", "CLARIFICATION_NEEDED"}:
        tables = ", ".join(db.get_usable_table_names()) if db else "none"
        history = conversation_context[-500:] if conversation_context else "none"
        response = generate_conversational_response(question, intent, tables, history, llm)
        entry = f"{_CONTEXT_TURN_SEP}\nTopic: conversation\nQ: {question}\nA: {response[:200]}\n"
        new_context = _trim_conversation_context(conversation_context + entry)
        return {
            "intent": "CONVERSATION",
            "nl_answer": response,
            "sql": None,
            "results": None,
            "row_count": 0,
            "from_cache": False,
            "error": None,
            "updated_context": new_context,
        }

    # Step 3: Check for sort/filter follow-up
    previous_sql = _get_last_sql(chat_history) or session_state.get("last_sql")
    previous_result_df = _get_last_result_df(chat_history)

    cleaned_query = None
    is_valid = False

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
        # Check global cache first, then session cache
        cached_sql, cached_df = _GLOBAL_QUERY_CACHE.find(question, embedder)
        if cached_sql is None:
            cached_sql, cached_df = find_cached_result(question, query_cache, embedder)
        if cached_sql is not None:
            _update_session_query_state(session_state, cached_sql, active_dialect, question=question)
            results = cached_df.to_dict(orient="records") if cached_df is not None and len(cached_df) > 0 else []
            entry = _build_context_entry(question, cached_sql, len(results),
                                         list(cached_df.columns) if cached_df is not None else [],
                                         status="cache")
            new_context = _trim_conversation_context(conversation_context + entry)
            return {
                "intent": "DATA_QUERY",
                "nl_answer": None,
                "sql": cached_sql,
                "results": results,
                "row_count": len(results),
                "from_cache": True,
                "error": None,
                "updated_context": new_context,
                "nl_pending": True,
            }

        # CHANGED: structured state is primary prompt memory; raw text is secondary.
        prompt_context = _build_prompt_context_from_state(session_state, conversation_context, conversation_turns)

        t_sql_gen = time.perf_counter()
        # Retrieve few-shot examples from RAG for better SQL generation
        few_shot_str = ""
        skip_rag_retrieval = bool(
            is_sort_request
            or is_filter_mod
            or _is_short_contextual_followup(question)
            or _is_bare_topn_followup(question)
            or is_followup_question(question)
        )
        if rag_engine and not skip_rag_retrieval:
            try:
                rag_context = rag_engine.retrieve(question, top_k=4)
                examples = rag_context.get("examples", [])[:3]
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
        if use_reasoning and _elapsed_seconds() > 2.0:
            # CHANGED: latency budget guard (<5s target) — avoid slow reasoner on late path.
            logger.info("Skipping reasoning_llm due to latency budget (>2s elapsed)")
            use_reasoning = False

        try:
            if use_reasoning:
                # Build flat prompt string — reasoner works best with a single user message
                reasoning_prompt = SQL_TEMPLATE.format(
                    full_schema=cached_schema_text or (db.get_table_info() if db else ""),
                    stored_procedure_guidance=stored_procedure_guidance,
                    context=prompt_context,
                    question=question,
                    few_shot_examples=few_shot_str,
                    dialect_label=dialect_label,
                    relative_date_reference=relative_date_reference,
                    enable_nolock=nolock_setting,
                )
                reasoning_resp = reasoning_llm.invoke(reasoning_prompt)
                raw_text = reasoning_resp.content.strip() if hasattr(reasoning_resp, "content") else str(reasoning_resp).strip()
                # Strip <think>...</think> reasoning block that DeepSeek R1 prepends
                resp_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                logger.info(f"DeepSeek reasoner generated SQL ({(time.perf_counter()-t_sql_gen)*1000:.0f}ms), complexity={pre_complexity}")
            else:
                # Fast path: use standard chat model via pre-built chain
                resp_text = sql_chain.invoke({
                    "question": question,
                    "context": prompt_context,
                    "few_shot_examples": few_shot_str,
                    "dialect_label": dialect_label,
                    "relative_date_reference": relative_date_reference,
                    "enable_nolock": nolock_setting,
                })
                logger.info(f"Fast chat model generated SQL ({(time.perf_counter()-t_sql_gen)*1000:.0f}ms), complexity={pre_complexity}")

            cleaned_query = _clean_sql_response(resp_text.strip())
            cleaned_query = fix_common_sql_errors(cleaned_query, dialect=active_dialect)
            is_valid, validation_msg = validate_sql(cleaned_query)
            logger.info(f"SQL generated ({(time.perf_counter()-t_sql_gen)*1000:.0f}ms), valid={is_valid}, reasoning={use_reasoning}")

            if not is_valid:
                # Retry with stricter prompt — prefer reasoning model for retry too
                retry_prompt = RETRY_PROMPT_TEMPLATE.format(
                    question=question,
                    full_schema=cached_schema_text or (db.get_table_info() if db else ""),
                    stored_procedure_guidance=stored_procedure_guidance,
                    dialect_label=dialect_label,
                    relative_date_reference=relative_date_reference,
                    enable_nolock=nolock_setting,
                )
                try:
                    active_llm = reasoning_llm if (reasoning_llm is not None and _elapsed_seconds() <= 2.0) else llm
                    retry_resp = active_llm.invoke(retry_prompt)
                    retry_raw = retry_resp.content.strip() if hasattr(retry_resp, "content") else str(retry_resp).strip()
                    retry_raw = re.sub(r"<think>.*?</think>", "", retry_raw, flags=re.DOTALL).strip()
                    retry_sql = _clean_sql_response(retry_raw)
                    retry_sql = fix_common_sql_errors(retry_sql, dialect=active_dialect)
                    is_valid_retry, _ = validate_sql(retry_sql)
                    if is_valid_retry:
                        cleaned_query = retry_sql
                        is_valid = True
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
                    else:
                        # Slow path: LLM retry
                        logger.info("ID-column fix: deterministic fix failed, falling back to LLM")
                        id_fix_pairs = ", ".join(
                            f"{c} → {_ID_COLUMN_MAP.get(c.lower(), 'name column')}"
                            for c in bad_ids
                        )
                        id_fix_prompt = (
                            f"Return one corrected {_dialect_label(active_dialect)} SELECT query.\n"
                            f"The previous query incorrectly selected raw ID columns: {id_fix_pairs}.\n"
                            f"Replace each ID column with its human-readable name column, adding a JOIN if needed.\n\n"
                            f"SCHEMA:\n{cached_schema_text or (db.get_table_info() if db else '')}\n\n"
                            f"DOMAIN BUSINESS RULES:\n{stored_procedure_guidance}\n\n"
                            f"CURRENT SQL (fix it):\n{cleaned_query}\n\n"
                            f"Output format: one SQL code block, optional Notes section.\n"
                            f"FIXED SQL:"
                        )
                        try:
                            fix_resp = llm.invoke(id_fix_prompt)
                            fix_sql = fix_resp.content.strip() if hasattr(fix_resp, "content") else str(fix_resp).strip()
                            fix_sql = _clean_sql_response(fix_sql)
                            fix_sql = fix_common_sql_errors(fix_sql, dialect=active_dialect)
                            fix_valid, _ = validate_sql(fix_sql)
                            if fix_valid:
                                cleaned_query = fix_sql
                        except Exception:
                            logger.warning("ID-column LLM fix failed, keeping original", exc_info=True)

            if not is_valid:
                entry = _build_context_entry(question, cleaned_query, 0, [], status="error")
                new_context = _trim_conversation_context(conversation_context + entry)
                return {
                    "intent": "DATA_QUERY",
                    "nl_answer": validation_msg,
                    "sql": cleaned_query,
                    "results": None,
                    "row_count": 0,
                    "from_cache": False,
                    "error": "Could not generate valid SQL",
                    "updated_context": new_context,
                }
        except Exception as e:
            entry = _build_context_entry(question, None, 0, [], status="error")
            new_context = _trim_conversation_context(conversation_context + entry)
            return {
                "intent": "DATA_QUERY",
                "nl_answer": None,
                "sql": None,
                "results": None,
                "row_count": 0,
                "from_cache": False,
                "error": f"Error generating SQL query: {str(e)}",
                "updated_context": new_context,
            }

    # Step 4: Execute query
    if cleaned_query and is_valid:
        cleaned_query = fix_common_sql_errors(cleaned_query, dialect=active_dialect)
        cleaned_query = expand_fuzzy_search(cleaned_query, db=db)
        cleaned_query = apply_stored_procedure_guardrails(cleaned_query, db=db)
        cleaned_query = _retry_ranking_shape_if_needed(
            question=question,
            sql=cleaned_query,
            llm=llm,
            full_schema=cached_schema_text,
            stored_procedure_guidance=stored_procedure_guidance,
            sql_dialect=active_dialect,
            enable_nolock=enable_nolock,
        )
        cleaned_query = optimize_sql_for_performance(
            cleaned_query,
            dialect=active_dialect,
            enable_nolock=enable_nolock,
        )
        cleaned_query = apply_query_performance_guardrails(
            cleaned_query,
            question=question,
            db=db,
            dialect=active_dialect,
        )

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
            # Retry with error context
            retry_prompt = (
                f"Return one corrected {_dialect_label(active_dialect)} SELECT query.\n"
                f"The SQL below has this error: {dry_err}\n"
                f"Fix the error while preserving the query intent.\n\n"
                f"SCHEMA:\n{cached_schema_text or ''}\n\n"
                f"BROKEN SQL:\n{cleaned_query}\n\n"
                f"Output format: one SQL code block, optional Notes section.\n"
                f"FIXED SQL:"
            )
            try:
                active_llm = reasoning_llm if (reasoning_llm is not None and _elapsed_seconds() <= 2.0) else llm
                fix_resp = active_llm.invoke(retry_prompt)
                fix_raw = fix_resp.content.strip() if hasattr(fix_resp, "content") else str(fix_resp).strip()
                fix_raw = re.sub(r"<think>.*?</think>", "", fix_raw, flags=re.DOTALL).strip()
                fix_sql = _clean_sql_response(fix_raw)
                fix_sql = fix_common_sql_errors(fix_sql, dialect=active_dialect)
                fix_valid, _ = validate_sql(fix_sql)
                if fix_valid:
                    cleaned_query = fix_sql
                    cleaned_query = optimize_sql_for_performance(
                        cleaned_query,
                        dialect=active_dialect,
                        enable_nolock=enable_nolock,
                    )
                    logger.info("SQL fixed after dry-run failure")
            except Exception:
                logger.warning("LLM fix after dry-run failed", exc_info=True)

        t_exec = time.perf_counter()
        df, error = execute_query_safe(db, cleaned_query, timeout_seconds=timeout_seconds, max_rows=1000)
        exec_ms = (time.perf_counter() - t_exec) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000
        row_count = len(df) if df is not None else 0
        logger.info(f"Query executed ({exec_ms:.0f}ms), rows={row_count}, error={error is not None}, total={total_ms:.0f}ms")

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
                    return retry_result

            entry = _build_context_entry(question, cleaned_query, 0, [], status="error")
            new_context = _trim_conversation_context(conversation_context + entry)
            return {
                "intent": "DATA_QUERY",
                "nl_answer": None,
                "sql": cleaned_query,
                "results": None,
                "row_count": 0,
                "from_cache": False,
                "error": error,
                "updated_context": new_context,
            }

        if df is not None and len(df) > 0:
            _update_session_query_state(session_state, cleaned_query, active_dialect, question=question)
            cache_query_result(question, cleaned_query, df, query_cache, embedder)
            _GLOBAL_QUERY_CACHE.add(question, cleaned_query, df, embedder)
            results = df.to_dict(orient="records")
            entry = _build_context_entry(question, cleaned_query, len(df), list(df.columns))
            new_context = _trim_conversation_context(conversation_context + entry)
            turn = ConversationTurn(
                question=question, sql=cleaned_query,
                topic=_extract_query_topic(question, cleaned_query),
                columns=list(df.columns), row_count=len(df), status="ok",
            )
            return {
                "intent": "DATA_QUERY",
                "nl_answer": None,
                "sql": cleaned_query,
                "results": results,
                "row_count": len(results),
                "from_cache": False,
                "error": None,
                "updated_context": new_context,
                "nl_pending": True,
                "conversation_turn": turn,
            }
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
                return retry_result
            # Both attempts returned nothing
            _update_session_query_state(session_state, cleaned_query, active_dialect, question=question)
            cache_query_result(question, cleaned_query, None, query_cache, embedder)
            entry = _build_context_entry(question, cleaned_query, 0, [], status="no_rows")
            new_context = _trim_conversation_context(conversation_context + entry)
            return {
                "intent": "DATA_QUERY",
                "nl_answer": "No results found for your query.",
                "sql": cleaned_query,
                "results": [],
                "row_count": 0,
                "from_cache": False,
                "error": None,
                "updated_context": new_context,
            }
        else:
            _update_session_query_state(session_state, cleaned_query, active_dialect, question=question)
            cache_query_result(question, cleaned_query, None, query_cache, embedder)
            entry = _build_context_entry(question, cleaned_query, 0, [], status="no_rows")
            new_context = _trim_conversation_context(conversation_context + entry)
            return {
                "intent": "DATA_QUERY",
                "nl_answer": "No results found for your query.",
                "sql": cleaned_query,
                "results": [],
                "row_count": 0,
                "from_cache": False,
                "error": None,
                "updated_context": new_context,
            }

    # Should not reach here
    return {
        "intent": "DATA_QUERY",
        "nl_answer": None,
        "sql": None,
        "results": None,
        "row_count": 0,
        "from_cache": False,
        "error": "Unexpected error processing query",
        "updated_context": conversation_context,
    }


def _get_last_sql(chat_history: List[Dict]) -> Optional[str]:
    for chat in reversed(chat_history):
        if chat.get("sql") and chat["sql"].strip().upper().startswith("SELECT"):
            return chat["sql"]
    return None


def _get_last_result_df(chat_history: List[Dict]):
    for chat in reversed(chat_history):
        if chat.get("result_df") is not None:
            return chat["result_df"]
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
    query_cache: List,
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
            retry_sql = optimize_sql_for_performance(
                retry_sql,
                dialect=sql_dialect,
                enable_nolock=enable_nolock,
            )
            retry_sql = apply_query_performance_guardrails(
                retry_sql,
                question=question,
                db=db,
                dialect=sql_dialect,
            )
            retry_df, retry_error = execute_query_safe(db, retry_sql, timeout_seconds=timeout_seconds, max_rows=1000)
            if retry_df is not None and len(retry_df) > 0:
                cache_query_result(question, retry_sql, retry_df, query_cache, embedder)
                results = retry_df.to_dict(orient="records")
                new_context = conversation_context + f"Q: {question}\nSQL: {retry_sql}\nResult: {len(retry_df)} rows returned with columns {list(retry_df.columns)}\n\n"
                return {
                    "intent": "DATA_QUERY",
                    "nl_answer": None,
                    "sql": retry_sql,
                    "results": results,
                    "row_count": len(results),
                    "from_cache": False,
                    "error": None,
                    "updated_context": new_context,
                    "nl_pending": True,
                }
        except Exception:
            continue

    return None
