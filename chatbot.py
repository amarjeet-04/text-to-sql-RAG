"""
chatbot.py
──────────
Streamlit chatbot using the same pipeline as streamlit_test.py / rag_pipeline_trace.py:
  - FIXED_SCHEMA + MINIMAL_RULES + DIRECT_SQL_TEMPLATE (hardcoded, clean)
  - RAG for few-shot examples
  - Direct LLM call (no handle_query wrapper, no validator, no guardrails)
  - Conversation history via conversation_turns (last 3 Q+SQL pairs in prompt)

Run:
    streamlit run chatbot.py
"""

import os
import re
import sys
import time
import hashlib
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import OrderedDict

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

st.set_page_config(page_title="Text-to-SQL Chatbot", page_icon="🤖", layout="wide")


# ── users store (role-based) ───────────────────────────────────────────────────
# In production replace with a real DB. Passwords stored as sha256 hashes.
# Default users: admin/admin123, analyst/analyst123
_USERS: Dict[str, Dict] = {
    "admin":   {"password_hash": hashlib.sha256(b"admin123").hexdigest(),   "role": "admin"},
    "analyst": {"password_hash": hashlib.sha256(b"analyst123").hexdigest(), "role": "analyst"},
}

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _verify_user(username: str, password: str) -> Optional[Dict]:
    user = _USERS.get(username)
    if user and user["password_hash"] == _hash_pw(password):
        return user
    return None

def _signup_user(username: str, password: str, role: str) -> bool:
    if username in _USERS or not username.strip():
        return False
    _USERS[username] = {"password_hash": _hash_pw(password), "role": role}
    return True


# ── query result cache (exact-match LRU + TTL) ────────────────────────────────
_QUERY_CACHE: "OrderedDict[str, Dict]" = OrderedDict()
_CACHE_TTL_S  = 300   # 5 minutes
_CACHE_MAX    = 200

def _cache_key(question: str) -> str:
    return question.strip().lower()

def _cache_get(question: str) -> Optional[Dict]:
    key = _cache_key(question)
    entry = _QUERY_CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL_S:
        _QUERY_CACHE.move_to_end(key)
        return entry["value"]
    if entry:
        _QUERY_CACHE.pop(key, None)
    return None

def _cache_put(question: str, value: Dict) -> None:
    key = _cache_key(question)
    _QUERY_CACHE[key] = {"ts": time.time(), "value": value}
    _QUERY_CACHE.move_to_end(key)
    while len(_QUERY_CACHE) > _CACHE_MAX:
        _QUERY_CACHE.popitem(last=False)


# ── LLM singleton cache (avoid re-init on every request) ─────────────────────
_LLM_CACHE: Dict[str, Any] = {}
_LLM_CACHE_LOCK = threading.Lock()

def _get_llm(model: str, api_key: str, timeout: int = 30):
    key = f"{model}:{api_key}"
    with _LLM_CACHE_LOCK:
        if key not in _LLM_CACHE:
            from langchain_openai import ChatOpenAI
            _LLM_CACHE[key] = ChatOpenAI(
                model=model, temperature=0.0,
                openai_api_key=api_key, request_timeout=timeout,
            )
        return _LLM_CACHE[key]


# ── schema ─────────────────────────────────────────────────────────────────────

FIXED_SCHEMA = (
    "-- NOTE: abbreviated for trace — production uses full INFORMATION_SCHEMA dump.\n"
    "TABLE: BookingData  (PNRNo, AgentId, SupplierId, ProductCountryid,\n"
    "                     BookingStatus, AgentBuyingPrice, CompanyBuyingPrice, CreatedDate, CheckInDate)\n"
    "TABLE: AgentMaster_V1  (AgentId, AgentName, AgentType)\n"
    "TABLE: suppliermaster_Report  (EmployeeId, SupplierName)\n"
    "TABLE: Master_Country  (CountryID, Country)\n"
    "TABLE: Master_City     (CityID, City, CountryID)\n"
    "TABLE: Hotelchain      (HotelId, HotelName, Country, City, Star, Chain)\n"
    "Table: AgentTypeMapping (AgentId, agenttype)\n"

)


# ── business rules ─────────────────────────────────────────────────────────────

MINIMAL_RULES = (
    "### BASE TABLE RULE\n"
    "• Always use BookingData as the base table.\n"
    "• All queries must start FROM BookingData BD.\n"
    "• Dimension tables must be LEFT JOINed to BookingData.\n"
    "• Never start FROM AgentMaster_V1, Master_Country, Master_City, or Supplier tables.\n"
    "\n"
    "Example pattern:\n"
    "FROM BookingData BD\n"
    "LEFT JOIN AgentMaster_V1 AM ON BD.AgentId = AM.AgentId\n"
    "LEFT JOIN Master_Country MC ON BD.ProductCountryid = MC.CountryID\n"
    "LEFT JOIN Master_City MCI ON BD.ProductCityId = MCI.CityId\n"
    "LEFT JOIN suppliermaster_Report SM ON BD.SupplierId = SM.EmployeeId\n"
    "\n"
    "### GLOBAL FILTERS\n"
    "• Default KPI filter:\n"
    "  BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')\n"
    "• Exception:\n"
    "  Total Cancelled Bookings uses BookingStatus = 'Cancelled'\n"
    "\n"
    "### DATE FILTER RULES\n"
    "• Never use CAST(date_col AS DATE) in WHERE.\n"
    "• Always use SARGable ranges:\n"
    "  BD.CreatedDate >= 'YYYY-MM-DD'\n"
    "  AND BD.CreatedDate < 'YYYY-MM-DD'\n"
    "\n"
    "### DUPLICATE PREVENTION\n"
    "• Lookup tables may contain duplicates.\n"
    "• Use DISTINCT mapping if needed:\n"
    "  WITH AM AS (\n"
    "    SELECT DISTINCT AgentId, AgentName\n"
    "    FROM AgentMaster_V1\n"
    "  )\n"
    "\n"
    "### AGENT DIMENSION RULES\n"
    "• Always join AgentMaster from BookingData:\n"
    "  LEFT JOIN dbo.AgentMaster_V1 A ON BD.AgentId = A.AgentId\n"
    "• AgentName = A.AgentName\n"
    "• AgentCode = A.AgentCode\n"
    "• AgentType:\n"
    "  - Prefer AgentTypeMapping if available:\n"
    "    LEFT JOIN dbo.AgentTypeMapping ATM ON ATM.AgentId = BD.AgentId\n"
    "    AgentType = ATM.agenttype\n"
    "  - Special override: if A.AgentCode = 'WEGR' then AgentType = 'We Groups'\n"
    "  - Else AgentType = COALESCE(ATM.agenttype,'Unknown')\n"
    "• Never select BD.AgentId in final output.\n"
    "\n"
    "### HOTEL DIMENSION RULES\n"
    "• DO NOT use BD.HotelChainId, HC.HotelChainId, HotelChainName, HotelChainId.\n"
    "• The ONLY allowed hotel join is:\n"
    "• Always join Hotelchain from BookingData:\n"
    "  LEFT JOIN dbo.Hotelchain HC ON BD.ProductId = HC.HotelId\n"
    "• Columns: HC.HotelName, HC.Country, HC.City, HC.Star\n"
    "• Plain 'country' means destination country from Master_Country, not HC.Country.\n"
    "• If the question asks for both chain and country, join both:\n"
    "  LEFT JOIN dbo.Hotelchain HC ON BD.ProductId = HC.HotelId\n"
    "  LEFT JOIN dbo.Master_Country MC ON BD.ProductCountryid = MC.CountryID\n"
    "  Use HC only for Chain and MC.Country for Country.\n"
    "• Chain bucket:\n"
    "  CASE\n"
    "    WHEN HC.chain IS NULL THEN 'No Chain Hotels'\n"
    "    WHEN HC.chain IN ('COURTYARD BY MARRIOTT','Marriott - Renaissance Hotels','Marriott','Marriott International') THEN 'Marriot Group'\n"
    "    WHEN HC.chain IN ('Hyatt Hotels Corporation','Hyatt Hotels') THEN 'Hyatt Group'\n"
    "    WHEN HC.chain IN ('Accor - Thalassa','Novotel Accor Hotels','Accor') THEN 'Accor Group'\n"
    "    WHEN HC.chain IN ('Best Western Hotels','Best Western Hotels & Resorts','Best Western') THEN 'Best Western Group'\n"
    "    WHEN HC.chain IN ('Kempinski Hotels & Resorts','Kempinski') THEN 'Kempinski Group'\n"
    "    WHEN HC.chain IN ('Four Seasons','Four Seasons Real Estate','Four Seasons Hotels and Resorts') THEN 'Four Seasons Group'\n"
    "    WHEN HC.chain IN ('Shangri-La Hotels and Resorts','Shangri-La Hotels - Traders Hotels') THEN 'Shangri-La Group'\n"
    "    ELSE HC.chain\n"
    "  END AS Chain\n"
    "\n"

    "### METRIC DEFINITIONS\n"
    "• Total Sales = SUM(COALESCE(BD.AgentBuyingPrice,0))\n"
    "• Total Cost  = SUM(COALESCE(BD.CompanyBuyingPrice,0))\n"
    "• Total Profit = SUM(COALESCE(BD.AgentBuyingPrice,0) - COALESCE(BD.CompanyBuyingPrice,0))\n"
    "• Total Booking Count = COUNT(DISTINCT BD.PNRNo)\n"
    "\n"
    "• Avg Booking Value =\n"
    "  SUM(COALESCE(BD.AgentBuyingPrice,0))\n"
    "  / NULLIF(COUNT(DISTINCT BD.PNRNo),0)\n"
    "\n"
    "• Total Room Nights =\n"
    "  SUM(DATEDIFF(DAY, BD.CheckInDate, BD.CheckOutDate))\n"
    "\n"
    "• Avg Booking Window =\n"
    "  AVG(CAST(DATEDIFF(DAY, BD.CreatedDate, BD.CheckInDate) AS FLOAT))\n"
    "\n"
    "• Total Last Minute Bookings =\n"
    "  COUNT(DISTINCT CASE\n"
    "    WHEN DATEDIFF(DAY, BD.CreatedDate, BD.CheckInDate) <= 1\n"
    "    THEN BD.PNRNo\n"
    "  END)\n"
    "\n"
    "• Total Cancelled Bookings =\n"
    "  COUNT(DISTINCT CASE\n"
    "    WHEN BD.BookingStatus = 'Cancelled'\n"
    "    THEN BD.PNRNo\n"
    "  END)\n"
    "\n"
    "• Total Non Refundable Bookings =\n"
    "  COUNT(DISTINCT CASE\n"
    "    WHEN DATEDIFF(DAY, BD.CreatedDate, BD.CancellationDate) <= 0\n"
    "    THEN BD.PNRNo\n"
    "  END)\n"
    "--------------------------------------------------------------\n"
    "### TEXT FILTER RULES (HARD)\n"
    "• For name-based filters (AgentName, SupplierName, HotelName, ProductName, Country, City):\n"
    "  - Default MUST be PREFIX match (index-friendly): col LIKE 'term%'\n"
    "  - DO NOT use leading wildcard by default: col LIKE '%term%'\n"
    "  - Use '%term%' ONLY if user explicitly asks: contains / match anywhere / partial anywhere / includes / similar\n"
    "• If term contains a single quote, escape it (SQL): replace ' with ''\n"
    "• Optional (only if your DB collation is case-sensitive): LOWER(col) LIKE 'term%'\n"
    "\n"
)



# ── prompt template ────────────────────────────────────────────────────────────

DIRECT_SQL_TEMPLATE = """\
You are a Text-to-SQL expert. Output ONLY a single SQL query inside a ```sql``` block. No explanation.

SCHEMA:
{full_schema}

BUSINESS RULES:
{stored_procedure_guidance}

EXAMPLES:
{few_shot_examples}

DIALECT : {dialect_label}
NOLOCK  : {enable_nolock}
TODAY   : {relative_date_reference}

HISTORY:
{context}

↑ If this question is a follow-up (references "that", "those", "it", "more", or lacks an explicit date/table):
  - Inherit the time window from the previous query — do NOT default to the current month/year.
  - Reuse the same primary table and date column unless explicitly changed.
  - Add the new filter or dimension on top of the inherited context.
If the question is fully self-contained, ignore history and build fresh.

RELEVANT TABLES: {retrieved_tables_hint}

Hard rules:
- One SELECT only. No DDL/DML.
- Alias every aggregated column with a clear business name.
- Use TOP N (SQL Server) or LIMIT N (PostgreSQL) for rankings.
- Apply WITH (NOLOCK) only when NOLOCK=True.
- Filter: BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request').
- Date filters: use >= / < boundaries, never YEAR() / MONTH() / CAST in WHERE.
- Lookup tables have duplicates — wrap in CTE with SELECT DISTINCT before joining.

QUESTION: {question}

```sql"""


# ── lazy-cached resources ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG engine…")
def _load_rag_engine():
    from app.rag.rag_engine import RAGEngine
    engine = RAGEngine(model_name="sentence-transformers/all-MiniLM-L6-v2", enable_embeddings=True)
    engine.load_default_schema()
    return engine


def _get_db_connection(host, port, username, password, database, api_key, model, query_timeout, view_support):
    from backend.services.sql_engine import initialize_connection
    (db, db_config, sql_chain, llm, reasoning_llm,
     rag, embedder, message, tables_count, views_count,
     cached_schema_text) = initialize_connection(
        host=host, port=str(port),
        db_username=username, db_password=password,
        database=database, llm_provider="openai",
        api_key=api_key, model=model,
        temperature=0.0, query_timeout=query_timeout,
        view_support=view_support,
    )
    return db, db_config, llm, message, tables_count, views_count


# ── pipeline ───────────────────────────────────────────────────────────────────

def _run_pipeline(
    question: str,
    llm,
    db,
    db_config,
    rag_engine,
    conversation_turns: List[Dict],
    api_key: str,
    sql_dialect: str = "sqlserver",
) -> Dict[str, Any]:
    """
    Same pipeline as streamlit_test.py / rag_pipeline_trace.py:
      Step 0: classify complexity → DIRECT (simple_llm) or COT (complex_llm)
      Step 1: RAG retrieve few-shot examples
      Step 2: build prompt with FIXED_SCHEMA + business rules
      Step 3: call LLM directly
      Step 4: extract + validate SQL
      Step 5: execute against DB
    """
    from backend.services.sql_engine import (
        _estimate_query_complexity,
        _build_relative_date_reference,
        _dialect_label,
        _clean_sql_response,
        validate_sql,
        SQL_TEMPLATE,
    )
    from app.db_utils import execute_query_safe
    import pandas as pd

    timings: Dict[str, float] = {}
    t_total = time.perf_counter()

    # Step 0: classify
    t0 = time.perf_counter()
    complexity = _estimate_query_complexity(question)
    use_direct = (complexity == "simple_llm")
    dialect    = sql_dialect
    dial_label = _dialect_label(dialect)
    date_ref   = _build_relative_date_reference()
    nolock_str = str(dialect == "sqlserver")   # "True" / "False"
    timings["routing"] = (time.perf_counter() - t0) * 1000

    # Step 1: RAG
    t1 = time.perf_counter()
    rag_context: Dict[str, Any] = {"examples": [], "tables": [], "rules": []}
    if rag_engine is not None:
        try:
            rag_context = rag_engine.retrieve(
                query=question, top_k=3, fast_mode=True, skip_for_followup=False
            )
        except Exception:
            pass
    timings["rag_retrieval"] = (time.perf_counter() - t1) * 1000

    # Step 2: build prompt
    t2 = time.perf_counter()

    examples = rag_context.get("examples") or []
    if examples:
        parts = ["EXAMPLE QUERIES (follow these patterns closely):"]
        for ex in examples:
            q_ = ex.get("question", "")
            s_ = ex.get("sql", "")
            if q_ and s_:
                parts.append(f"Q: {q_}\nSQL: {s_}")
        few_shot_str = "\n\n".join(parts)
    else:
        few_shot_str = ""

    raw_tables = rag_context.get("tables") or []
    rag_hints  = [t["table"] if isinstance(t, dict) else str(t) for t in raw_tables if t]
    retrieved_tables_hint = ", ".join(rag_hints[:6]) if rag_hints else "none"

    # conversation context — last 4 turns with Q + SQL + result snippet
    if conversation_turns:
        ctx_lines = []
        for turn in conversation_turns[-4:]:
            q_prev = turn.get("question", "")
            s_prev = turn.get("sql", "")
            r_prev = turn.get("result_summary", "")
            if q_prev:
                ctx_lines.append(f"Q: {q_prev}")
            if s_prev:
                ctx_lines.append(f"SQL: {s_prev}")
            if r_prev:
                ctx_lines.append(f"RESULTS:\n{r_prev}")
        context_str = "\n".join(ctx_lines) if ctx_lines else "(no prior conversation)"
    else:
        context_str = "(no prior conversation)"

    # business rules — RAG rules when available, else MINIMAL_RULES fallback
    rag_rules = rag_context.get("rules") or []
    # stored_guidance = (
    #     "\n".join(r.get("content", r.get("name", "")) for r in rag_rules)
    #     if rag_rules else MINIMAL_RULES
    # )
    stored_guidance = MINIMAL_RULES
    if rag_rules:
        extra = "\n".join(r.get("content", r.get("name", "")) for r in rag_rules)
        stored_guidance = f"{MINIMAL_RULES}\n### ADDITIONAL RETRIEVED RULES\n{extra}"

    prompt_vars = dict(
        dialect_label             = dial_label,
        enable_nolock             = nolock_str,
        relative_date_reference   = date_ref,
        full_schema               = FIXED_SCHEMA,
        stored_procedure_guidance = stored_guidance,
        context                   = context_str,
        retrieved_tables_hint     = retrieved_tables_hint,
        few_shot_examples         = few_shot_str,
        question                  = question,
    )

    template     = DIRECT_SQL_TEMPLATE if use_direct else SQL_TEMPLATE
    timings["prompt_build"] = (time.perf_counter() - t2) * 1000

    # Step 3: LLM call
    t3 = time.perf_counter()
    raw_response = ""
    llm_error    = None
    try:
        if use_direct:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            simple_model = os.getenv("LLM_MODEL_SIMPLE", "gpt-4o-mini")
            simple_llm   = _get_llm(simple_model, api_key, timeout=30)
            chain = (
                ChatPromptTemplate.from_template(DIRECT_SQL_TEMPLATE)
                | simple_llm
                | StrOutputParser()
            )
            raw_response = chain.invoke(prompt_vars)
        else:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            chain = (
                ChatPromptTemplate.from_template(SQL_TEMPLATE)
                | llm
                | StrOutputParser()
            )
            raw_response = chain.invoke(prompt_vars)
    except Exception as exc:
        llm_error = str(exc)
    timings["llm_call"] = (time.perf_counter() - t3) * 1000

    # Step 4: extract + validate SQL
    t4 = time.perf_counter()
    cleaned_sql = ""
    is_valid    = False
    if raw_response and not llm_error:
        raw_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
        cleaned_sql  = _clean_sql_response(raw_response)
        is_valid, _  = validate_sql(cleaned_sql)
    timings["sql_extraction"] = (time.perf_counter() - t4) * 1000

    # Step 5: execute
    t5 = time.perf_counter()
    result_df  = pd.DataFrame()
    exec_error = None
    row_count  = 0
    if is_valid and cleaned_sql and db is not None:
        timeout = getattr(db_config, "query_timeout", 30)
        df, exec_error = execute_query_safe(db, cleaned_sql, timeout_seconds=timeout, max_rows=500)
        if df is not None:
            result_df = df
            row_count = len(df)
    timings["db_execution"] = (time.perf_counter() - t5) * 1000
    timings["total"]        = (time.perf_counter() - t_total) * 1000

    error_msg = llm_error or (str(exec_error) if exec_error else "")
    if not is_valid and not llm_error:
        error_msg = "LLM did not produce valid SQL."

    return {
        "sql":            cleaned_sql,
        "results_df":     result_df,
        "row_count":      row_count,
        "error":          error_msg,
        "complexity":     complexity,
        "timings":        timings,
        "few_shot_count": len(examples),
        "rag_tables":     retrieved_tables_hint,
    }


# ── session state ──────────────────────────────────────────────────────────────

if "chat_history"       not in st.session_state: st.session_state.chat_history       = []
if "connected"          not in st.session_state: st.session_state.connected          = False
if "pipeline"           not in st.session_state: st.session_state.pipeline           = {}
if "conversation_turns" not in st.session_state: st.session_state.conversation_turns = []
if "auth_user"          not in st.session_state: st.session_state.auth_user          = None
if "auth_role"          not in st.session_state: st.session_state.auth_role          = None
if "auth_tab"           not in st.session_state: st.session_state.auth_tab           = "login"
if "show_settings"      not in st.session_state: st.session_state.show_settings      = False
if "show_users"         not in st.session_state: st.session_state.show_users         = False
# persistent connection settings (so they survive panel close/open)
if "cfg_api_key"        not in st.session_state: st.session_state.cfg_api_key        = os.getenv("OPENAI_API_KEY", "")
if "cfg_model"          not in st.session_state: st.session_state.cfg_model          = os.getenv("LLM_MODEL", "gpt-4o")
if "cfg_host"           not in st.session_state: st.session_state.cfg_host           = os.getenv("DB_HOST", "localhost")
if "cfg_port"           not in st.session_state: st.session_state.cfg_port           = os.getenv("DB_PORT", "1433")
if "cfg_username"       not in st.session_state: st.session_state.cfg_username       = os.getenv("DB_USER", "sa")
if "cfg_password"       not in st.session_state: st.session_state.cfg_password       = os.getenv("DB_PASSWORD", "")
if "cfg_database"       not in st.session_state: st.session_state.cfg_database       = os.getenv("DB_NAME", os.getenv("DB_DATABASE", ""))
if "cfg_timeout"        not in st.session_state: st.session_state.cfg_timeout        = 30
if "cfg_view_support"   not in st.session_state: st.session_state.cfg_view_support   = True


# ── auth gate ──────────────────────────────────────────────────────────────────

if st.session_state.auth_user is None:
    # hide sidebar on login page too
    st.markdown(
        """<style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        </style>""",
        unsafe_allow_html=True,
    )

    # centered narrow card layout
    _, card_col, _ = st.columns([1, 1.4, 1])
    with card_col:
        _logo_path = ROOT / "frontend" / "image (1).png"
        if _logo_path.exists():
            import base64
            with open(str(_logo_path), "rb") as _f:
                _logo_b64 = base64.b64encode(_f.read()).decode()
            st.markdown(
                f"<div style='display:flex; justify-content:center; margin-bottom:12px;'>"
                f"<img src='data:image/png;base64,{_logo_b64}' "
                f"style='width:100px; height:100px; border-radius:50%; object-fit:cover; "
                f"box-shadow:0 4px 16px rgba(0,0,0,0.18);'/>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<h2 style='text-align:center; margin-top:4px; margin-bottom:4px;'>Within Earth Chatbot</h2>"
            "<p style='text-align:center; color:#888; margin-bottom:20px;'>Sign in to continue</p>",
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(["Sign In", "Sign Up"])

        with tab_login:
            login_user = st.text_input("Username", key="login_user", placeholder="Enter username")
            login_pw   = st.text_input("Password", type="password", key="login_pw", placeholder="Enter password")
            if st.button("Sign In", type="primary", use_container_width=True, key="btn_login"):
                user = _verify_user(login_user, login_pw)
                if user:
                    st.session_state.auth_user = login_user
                    st.session_state.auth_role = user["role"]
                    # ── auto-connect on sign-in ──────────────────────────────
                    with st.spinner("Connecting to database…"):
                        try:
                            db, db_config, llm, _msg, tc, vc = _get_db_connection(
                                st.session_state.cfg_host,
                                st.session_state.cfg_port,
                                st.session_state.cfg_username,
                                st.session_state.cfg_password,
                                st.session_state.cfg_database,
                                st.session_state.cfg_api_key,
                                st.session_state.cfg_model,
                                st.session_state.cfg_timeout,
                                st.session_state.cfg_view_support,
                            )
                            rag_engine = _load_rag_engine()
                            st.session_state.pipeline = dict(
                                db=db, db_config=db_config, llm=llm,
                                rag_engine=rag_engine, api_key=st.session_state.cfg_api_key,
                            )
                            st.session_state.connected = True
                        except Exception:
                            # auto-connect failed — user can connect manually via Settings
                            st.session_state.connected = False
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_signup:
            new_user = st.text_input("Username", key="signup_user", placeholder="Choose a username")
            new_pw   = st.text_input("Password", type="password", key="signup_pw", placeholder="Min. 6 characters")
            new_pw2  = st.text_input("Confirm Password", type="password", key="signup_pw2", placeholder="Repeat password")
            new_role = st.selectbox("Role", ["analyst", "admin"], key="signup_role")
            if st.button("Sign Up", type="primary", use_container_width=True, key="btn_signup"):
                if not new_user.strip():
                    st.error("Username cannot be empty.")
                elif new_pw != new_pw2:
                    st.error("Passwords do not match.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif not _signup_user(new_user, new_pw, new_role):
                    st.error("Username already taken.")
                else:
                    st.success(f"Account created! Sign in as **{new_user}**.")
    st.stop()


# ── hide default sidebar & hamburger ──────────────────────────────────────────
st.markdown(
    """<style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    </style>""",
    unsafe_allow_html=True,
)

# ── header bar ────────────────────────────────────────────────────────────────
is_admin = st.session_state.auth_role == "admin"

h_left, h_mid, h_right = st.columns([5, 1, 4])

with h_left:
    status_icon = "🟢" if st.session_state.connected else "🔴"
    _logo_path = ROOT / "frontend" / "image (1).png"
    if _logo_path.exists():
        import base64 as _b64
        with open(str(_logo_path), "rb") as _f:
            _hdr_b64 = _b64.b64encode(_f.read()).decode()
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:12px;'>"
            f"<img src='data:image/png;base64,{_hdr_b64}' "
            f"style='width:48px; height:48px; border-radius:50%; object-fit:cover; "
            f"box-shadow:0 2px 8px rgba(0,0,0,0.15); flex-shrink:0;'/>"
            f"<span style='font-size:1.5rem; font-weight:700; line-height:1.2;'>"
            f"Within Earth Chatbot &nbsp;{status_icon}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"## Within Earth Chatbot &nbsp; {status_icon}")

with h_mid:
    st.write("")  # vertical spacer

with h_right:
    btn_cols = st.columns([1, 1, 1, 1] if is_admin else [1, 1])
    col_idx = 0

    # Settings button (admin only)
    if is_admin:
        if btn_cols[col_idx].button(
            "⚙️ Settings",
            use_container_width=True,
            type="primary" if st.session_state.show_settings else "secondary",
        ):
            st.session_state.show_settings = not st.session_state.show_settings
            st.session_state.show_users = False
        col_idx += 1

        # Users button (admin only)
        if btn_cols[col_idx].button(
            "👥 Users",
            use_container_width=True,
            type="primary" if st.session_state.show_users else "secondary",
        ):
            st.session_state.show_users = not st.session_state.show_users
            st.session_state.show_settings = False
        col_idx += 1

    # Clear chat
    if btn_cols[col_idx].button("🗑 Clear", use_container_width=True):
        st.session_state.chat_history       = []
        st.session_state.conversation_turns = []
        st.rerun()
    col_idx += 1

    # Sign Out
    if btn_cols[col_idx].button("Sign Out", use_container_width=True):
        for k in ["auth_user", "auth_role", "chat_history", "connected",
                  "pipeline", "conversation_turns", "show_settings", "show_users"]:
            if k in ("auth_user", "auth_role"):
                st.session_state[k] = None
            elif k in ("chat_history", "conversation_turns"):
                st.session_state[k] = []
            elif k == "connected":
                st.session_state[k] = False
            elif k in ("show_settings", "show_users"):
                st.session_state[k] = False
            else:
                st.session_state[k] = {}
        st.rerun()

st.divider()

# ── Settings panel ────────────────────────────────────────────────────────────
if st.session_state.show_settings and is_admin:
    with st.container(border=True):
        st.subheader("⚙️ Connection Settings")
        s_col1, s_col2 = st.columns(2)

        with s_col1:
            st.markdown("**LLM**")
            st.session_state.cfg_api_key = st.text_input(
                "OpenAI API Key", value=st.session_state.cfg_api_key, type="password", key="s_api_key"
            )
            st.session_state.cfg_model = st.text_input(
                "Model (complex)", value=st.session_state.cfg_model, key="s_model"
            )

        with s_col2:
            st.markdown("**Database**")
            st.session_state.cfg_host     = st.text_input("Host",     value=st.session_state.cfg_host,     key="s_host")
            st.session_state.cfg_port     = st.text_input("Port",     value=st.session_state.cfg_port,     key="s_port")
            st.session_state.cfg_username = st.text_input("Username", value=st.session_state.cfg_username, key="s_dbuser")
            st.session_state.cfg_password = st.text_input("Password", value=st.session_state.cfg_password, type="password", key="s_dbpw")
            st.session_state.cfg_database = st.text_input("Database", value=st.session_state.cfg_database, key="s_dbname")

        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns([2, 2, 1, 1])
        st.session_state.cfg_timeout      = adv_col1.number_input("Query timeout (s)", min_value=5, max_value=120, value=st.session_state.cfg_timeout, key="s_timeout")
        st.session_state.cfg_view_support = adv_col2.checkbox("Enable view support", value=st.session_state.cfg_view_support, key="s_views")

        btn_c1, btn_c2, _ = st.columns([1, 1, 4])
        connect_btn    = btn_c1.button("🔌 Connect",    use_container_width=True, type="primary", key="s_connect")
        disconnect_btn = btn_c2.button("⛔ Disconnect", use_container_width=True,                 key="s_disconnect")

        if connect_btn:
            with st.spinner("Connecting…"):
                try:
                    db, db_config, llm, msg, tc, vc = _get_db_connection(
                        st.session_state.cfg_host,
                        st.session_state.cfg_port,
                        st.session_state.cfg_username,
                        st.session_state.cfg_password,
                        st.session_state.cfg_database,
                        st.session_state.cfg_api_key,
                        st.session_state.cfg_model,
                        st.session_state.cfg_timeout,
                        st.session_state.cfg_view_support,
                    )
                    rag_engine = _load_rag_engine()
                    st.session_state.pipeline = dict(
                        db=db, db_config=db_config, llm=llm,
                        rag_engine=rag_engine, api_key=st.session_state.cfg_api_key,
                    )
                    st.session_state.connected          = True
                    st.session_state.chat_history       = []
                    st.session_state.conversation_turns = []
                    st.session_state.show_settings      = False
                    st.success(f"✅ Connected — {tc} tables, {vc} views")
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ {exc}")
                    st.session_state.connected = False

        if disconnect_btn:
            st.session_state.connected          = False
            st.session_state.pipeline           = {}
            st.session_state.chat_history       = []
            st.session_state.conversation_turns = []
            st.session_state.show_settings      = False
            st.info("Disconnected.")
            st.rerun()

# ── Users panel (admin only) ──────────────────────────────────────────────────
if st.session_state.show_users and is_admin:
    with st.container(border=True):
        st.subheader("👥 User Management")

        # user list table
        import pandas as pd
        users_list = [
            {"Username": u, "Role": v["role"]}
            for u, v in _USERS.items()
        ]
        st.dataframe(pd.DataFrame(users_list), use_container_width=True, hide_index=True)

        st.markdown("**Add new user**")
        u_col1, u_col2, u_col3, u_col4, u_col5 = st.columns([2, 2, 2, 1, 1])
        new_uname = u_col1.text_input("Username", key="nu_uname", label_visibility="collapsed", placeholder="Username")
        new_upw   = u_col2.text_input("Password", key="nu_pw",    label_visibility="collapsed", placeholder="Password", type="password")
        new_urole = u_col3.selectbox("Role", ["analyst", "admin"], key="nu_role", label_visibility="collapsed")
        add_btn   = u_col4.button("➕ Add",    use_container_width=True, type="primary", key="nu_add")
        del_uname = u_col5.text_input("Delete username", key="del_uname", label_visibility="collapsed", placeholder="Delete user")

        del_col1, del_col2, _ = st.columns([1, 1, 4])
        del_btn = del_col1.button("🗑 Delete", use_container_width=True, key="nu_del")

        if add_btn:
            if not new_uname.strip():
                st.error("Username cannot be empty.")
            elif len(new_upw) < 6:
                st.error("Password must be at least 6 characters.")
            elif not _signup_user(new_uname.strip(), new_upw, new_urole):
                st.error(f"Username **{new_uname}** already exists.")
            else:
                st.success(f"✅ User **{new_uname}** ({new_urole}) created.")
                st.rerun()

        if del_btn:
            target = del_uname.strip()
            if not target:
                st.error("Enter a username to delete.")
            elif target == st.session_state.auth_user:
                st.error("Cannot delete your own account.")
            elif target not in _USERS:
                st.error(f"User **{target}** not found.")
            else:
                del _USERS[target]
                st.success(f"✅ User **{target}** deleted.")
                st.rerun()

# ── main area ──────────────────────────────────────────────────────────────────

st.caption(f"👤 {st.session_state.auth_user} · {st.session_state.auth_role} · Clean schema · Minimal rules · Direct LLM call · Conversation history via last 3 turns.")

# ── suggested questions (shown only when chat is empty and connected) ──────────
_SUGGESTIONS = [
    "What are the top 10 agents by total sales this year?",
    "Show total bookings and revenue by country for 2024",
    "Which suppliers have the highest profit margin?",
    "What is the monthly booking trend for the last 6 months?",
    "List top 5 hotels by number of bookings",
    "What is the cancellation rate by agent type?",
]

if st.session_state.connected and not st.session_state.chat_history:
    st.markdown("#### Suggested questions")
    sg_col1, sg_col2 = st.columns(2)
    for i, suggestion in enumerate(_SUGGESTIONS):
        col = sg_col1 if i % 2 == 0 else sg_col2
        if col.button(suggestion, key=f"sg_{i}", use_container_width=True):
            st.session_state._auto_question = suggestion
            st.rerun()

# render existing chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sql"):
            with st.expander("Generated SQL", expanded=False):
                st.code(msg["sql"], language="sql")
        if msg.get("data") is not None:
            st.dataframe(msg["data"], use_container_width=True, hide_index=True)
        if msg.get("error"):
            st.error(msg["error"])
        if msg.get("timing"):
            st.caption(f"⏱ {msg['timing']}")

# chat input
_typed = st.chat_input(
    "Ask a question about your data…",
    disabled=not st.session_state.connected,
)

# pick up auto-question from suggestion button click
_auto = None
if hasattr(st.session_state, "_auto_question") and st.session_state._auto_question:
    _auto = st.session_state._auto_question
    st.session_state._auto_question = ""
question = _typed or _auto

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("_Thinking…_")

        t0 = time.perf_counter()
        try:
            import pandas as pd

            # ── cache check ────────────────────────────────────────────
            cached = _cache_get(question)
            if cached is not None:
                status.empty()
                sql        = cached["sql"]
                result_df  = cached["result_df"]
                row_count  = cached["row_count"]
                reply      = cached["reply"]
                complexity = cached["complexity"]
                total_ms   = cached["total_ms"]
                timings    = cached["timings"]
                st.caption("⚡ Served from cache")
                st.write(reply)
                if sql:
                    with st.expander("Generated SQL", expanded=False):
                        st.code(sql, language="sql")
                if result_df is not None and not result_df.empty:
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                complexity_labels = {
                    "deterministic": "⚡ Instant (deterministic)",
                    "simple_llm":    "🟢 Fast (gpt-4o-mini)",
                    "complex_llm":   "🟡 Complex (gpt-4o)",
                }
                timing_str = f"{total_ms:.0f} ms · {complexity_labels.get(complexity, complexity)} · ⚡ cached"
                st.caption(f"⏱ {timing_str}")
                st.session_state.conversation_turns.append({
                    "question":       question,
                    "sql":            sql or "",
                    "result_summary": result_df.head(10).to_string(index=False) if result_df is not None and not result_df.empty else "",
                })
                st.session_state.chat_history.append({
                    "role": "assistant", "content": reply, "sql": sql,
                    "data": result_df if (result_df is not None and not result_df.empty) else None,
                    "error": "", "timing": timing_str,
                })
                st.stop()

            p = st.session_state.pipeline
            result = _run_pipeline(
                question           = question,
                llm                = p["llm"],
                db                 = p["db"],
                db_config          = p["db_config"],
                rag_engine         = p.get("rag_engine"),
                conversation_turns = st.session_state.conversation_turns,
                api_key            = p.get("api_key", st.session_state.cfg_api_key),
            )
            status.empty()

            sql        = result["sql"]
            result_df  = result["results_df"]
            row_count  = result["row_count"]
            error      = result["error"]
            complexity = result["complexity"]
            timings    = result["timings"]
            total_ms   = timings.get("total", (time.perf_counter() - t0) * 1000)

            # ── NL summary — run in thread so it doesn't block render ──
            def _generate_summary(q, df, key):
                try:
                    from langchain_core.prompts import ChatPromptTemplate
                    from langchain_core.output_parsers import StrOutputParser
                    _llm = _get_llm("gpt-4o-mini", key, timeout=15)
                    prompt = ChatPromptTemplate.from_template(
                        "You are a data analyst. Given the question and query results below, "
                        "write a concise 1-3 sentence summary of the key findings. "
                        "Use numbers and business terms. No markdown headers.\n\n"
                        "Question: {question}\n\nResults (top rows):\n{data}\n\nSummary:"
                    )
                    chain = prompt | _llm | StrOutputParser()
                    return chain.invoke({"question": q, "data": df.head(10).to_string(index=False)})
                except Exception:
                    return None

            if error:
                reply = f"⚠️ {error}"
                st.write(reply)
                summary_future = None
            elif result_df is not None and not result_df.empty:
                _eff_key = p.get("api_key") or st.session_state.cfg_api_key
                summary_future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
                    _generate_summary, question, result_df, _eff_key
                )
                reply = f"Query executed — {row_count} rows returned."
            else:
                reply = f"Query executed — {row_count} rows returned."
                summary_future = None

            # resolve summary (by now the thread has likely finished)
            if summary_future is not None:
                try:
                    nl = summary_future.result(timeout=12)
                    if nl:
                        reply = nl
                except Exception:
                    pass
            st.write(reply)

            if sql:
                with st.expander("Generated SQL", expanded=False):
                    st.code(sql, language="sql")

            if result_df is not None and not result_df.empty:
                st.dataframe(result_df, use_container_width=True, hide_index=True)

            if error:
                st.error(error)

            # timing caption
            complexity_labels = {
                "deterministic": "⚡ Instant (deterministic)",
                "simple_llm":    "🟢 Fast (gpt-4o-mini)",
                "complex_llm":   "🟡 Complex (gpt-4o)",
            }
            timing_str = f"{total_ms:.0f} ms · {complexity_labels.get(complexity, complexity)}"
            st.caption(f"⏱ {timing_str}")

            # stage timing table
            pipeline_rows = [
                ("routing  (classify complexity)",      timings.get("routing",        0.0)),
                ("RAG  embed + retrieve few-shots",     timings.get("rag_retrieval",  0.0)),
                ("prompt  build + format",              timings.get("prompt_build",   0.0)),
                ("LLM  call (gpt-4o-mini or gpt-4o)",  timings.get("llm_call",       0.0)),
                ("SQL  extract + validate",             timings.get("sql_extraction", 0.0)),
                ("DB  execute_query_safe",              timings.get("db_execution",   0.0)),
            ]
            rows_sum    = sum(ms for _, ms in pipeline_rows)
            unaccounted = round(total_ms - rows_sum, 1)

            rows = []
            for label, ms_val in pipeline_rows:
                bar = "█" * min(30, int(ms_val / 200))
                rows.append({"Step": label, "ms": f"{ms_val:.0f}", "bar": bar})
            rows.append({"Step": "─" * 38,           "ms": "─────",           "bar": ""})
            rows.append({"Step": "sum of rows",       "ms": f"{rows_sum:.0f}", "bar": ""})
            if abs(unaccounted) > 1:
                rows.append({"Step": "⚠ unaccounted", "ms": f"{unaccounted:.0f}", "bar": "← overhead"})
            rows.append({"Step": "TOTAL  (wall clock)", "ms": f"{total_ms:.0f}",
                         "bar": "█" * min(30, int(total_ms / 200))})

            rag_info = f"  ·  {result['few_shot_count']} RAG example(s)  ·  tables hint: {result['rag_tables']}"
            with st.expander(f"⏱ Step timings{rag_info}", expanded=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── store in cache (skip if error) ─────────────────────────
            if not error and result_df is not None:
                _cache_put(question, {
                    "sql":        sql,
                    "result_df":  result_df,
                    "row_count":  row_count,
                    "reply":      reply,
                    "complexity": complexity,
                    "total_ms":   total_ms,
                    "timings":    timings,
                })

            # save turn for follow-up conversation context
            if question:
                result_summary = ""
                if result_df is not None and not result_df.empty:
                    result_summary = result_df.head(10).to_string(index=False)
                st.session_state.conversation_turns.append({
                    "question":       question,
                    "sql":            sql or "",
                    "result_summary": result_summary,
                })

            # persist in chat_history
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": reply,
                "sql":     sql,
                "data":    result_df if (result_df is not None and not result_df.empty) else None,
                "error":   error,
                "timing":  timing_str,
            })

        except Exception as exc:
            status.empty()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            err_msg    = str(exc)
            st.error(f"Pipeline error: {err_msg}")
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": f"❌ Error: {err_msg}",
                "sql":     "",
                "data":    None,
                "error":   err_msg,
                "timing":  f"{elapsed_ms:.0f} ms",
            })

if not st.session_state.connected:
    st.info("Click **⚙️ Settings** above to enter connection details and connect to the database.")
