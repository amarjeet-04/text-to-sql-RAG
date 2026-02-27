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
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

st.set_page_config(page_title="Text-to-SQL Chatbot", page_icon="🤖", layout="wide")


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
"### QUERY SHAPE RULES\n"
    "• GROUP BY only dimension columns.\n"
    "• If user asks TOP/BEST: use TOP(N) + ORDER BY metric DESC.\n"
    "• If user asks OVERALL KPI: do NOT add ORDER BY.\n"
    "• Detail list queries must use TOP(200).\n"
    "• Prefer aggregated queries over row-level queries.\n"
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
                query=question, top_k=4, fast_mode=False, skip_for_followup=False
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

    # conversation context — last 3 Q+SQL turns
    if conversation_turns:
        ctx_lines = []
        for turn in conversation_turns[-3:]:
            q_prev = turn.get("question", "")
            s_prev = turn.get("sql", "")
            if q_prev:
                ctx_lines.append(f"Q: {q_prev}")
            if s_prev:
                ctx_lines.append(f"SQL: {s_prev}")
        context_str = "\n".join(ctx_lines) if ctx_lines else "(no prior conversation)"
    else:
        context_str = "(no prior conversation)"

    # business rules — RAG rules when available, else MINIMAL_RULES fallback
    rag_rules = rag_context.get("rules") or []
    stored_guidance = (
        "\n".join(r.get("content", r.get("name", "")) for r in rag_rules)
        if rag_rules else MINIMAL_RULES
    )

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
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            simple_model = os.getenv("LLM_MODEL_SIMPLE", "gpt-4o-mini")
            simple_llm   = ChatOpenAI(
                model=simple_model, temperature=0,
                openai_api_key=api_key, request_timeout=30,
            )
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


# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Connection")

    st.subheader("LLM")
    api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    model   = st.text_input("Model (complex)", value=os.getenv("LLM_MODEL", "gpt-4o"))

    st.subheader("Database")
    host     = st.text_input("Host",     value=os.getenv("DB_HOST",     "localhost"))
    port     = st.text_input("Port",     value=os.getenv("DB_PORT",     "1433"))
    username = st.text_input("Username", value=os.getenv("DB_USER",     "sa"))
    password = st.text_input("Password", value=os.getenv("DB_PASSWORD", ""), type="password")
    database = st.text_input("Database", value=os.getenv("DB_NAME", os.getenv("DB_DATABASE", "")))

    st.divider()
    query_timeout = st.number_input("Query timeout (s)", min_value=5, max_value=120, value=30)
    view_support  = st.checkbox("Enable view support", value=True)

    col1, col2 = st.columns(2)
    connect_btn    = col1.button("Connect",    use_container_width=True, type="primary")
    disconnect_btn = col2.button("Disconnect", use_container_width=True)

    if connect_btn:
        with st.spinner("Connecting…"):
            try:
                db, db_config, llm, msg, tc, vc = _get_db_connection(
                    host, port, username, password, database,
                    api_key, model, query_timeout, view_support,
                )
                rag_engine = _load_rag_engine()
                st.session_state.pipeline = dict(
                    db=db, db_config=db_config, llm=llm,
                    rag_engine=rag_engine, api_key=api_key,
                )
                st.session_state.connected          = True
                st.session_state.chat_history       = []
                st.session_state.conversation_turns = []
                st.success(f"✅ Connected — {tc} tables, {vc} views")
            except Exception as exc:
                st.error(f"❌ {exc}")
                st.session_state.connected = False

    if disconnect_btn:
        st.session_state.connected          = False
        st.session_state.pipeline           = {}
        st.session_state.chat_history       = []
        st.session_state.conversation_turns = []
        st.info("Disconnected.")

    st.divider()
    if st.session_state.connected:
        st.success("🟢 Connected")
    else:
        st.warning("🔴 Not connected")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history       = []
        st.session_state.conversation_turns = []
        st.rerun()


# ── main area ──────────────────────────────────────────────────────────────────

st.title("🤖 Text-to-SQL Chatbot")
st.caption("Clean schema · Minimal rules · Direct LLM call · Conversation history via last 3 turns.")

# render existing chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sql"):
            with st.expander("Generated SQL", expanded=False):
                st.code(msg["sql"], language="sql")
        if msg.get("data") is not None:
            st.dataframe(msg["data"], use_container_width=True)
        if msg.get("error"):
            st.error(msg["error"])
        if msg.get("timing"):
            st.caption(f"⏱ {msg['timing']}")

# chat input
question = st.chat_input(
    "Ask a question about your data…",
    disabled=not st.session_state.connected,
)

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

            p = st.session_state.pipeline
            result = _run_pipeline(
                question           = question,
                llm                = p["llm"],
                db                 = p["db"],
                db_config          = p["db_config"],
                rag_engine         = p.get("rag_engine"),
                conversation_turns = st.session_state.conversation_turns,
                api_key            = p.get("api_key", api_key),
            )
            status.empty()

            sql        = result["sql"]
            result_df  = result["results_df"]
            row_count  = result["row_count"]
            error      = result["error"]
            complexity = result["complexity"]
            timings    = result["timings"]
            total_ms   = timings.get("total", (time.perf_counter() - t0) * 1000)

            # reply
            if error:
                reply = f"⚠️ {error}"
            else:
                reply = f"Query executed — **{row_count} rows** returned."

            st.markdown(reply)

            if sql:
                with st.expander("Generated SQL", expanded=False):
                    st.code(sql, language="sql")

            if result_df is not None and not result_df.empty:
                st.dataframe(result_df, use_container_width=True)

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

            # save turn for follow-up conversation context
            # only append if we have at least a question (skip noisy empty-SQL turns)
            if question:
                st.session_state.conversation_turns.append({
                    "question": question,
                    "sql":      sql or "",   # keep even if empty — context_str skips blank SQL
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
    st.info("👈 Fill in connection details in the sidebar and click **Connect** to start.")
