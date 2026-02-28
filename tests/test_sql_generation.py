"""
Standalone SQL generation test — no DB execution, no LLM retry pipeline.
Calls the same prompt + model as the main application and logs the cleaned SQL.

Usage:
  venv/bin/python tests/test_sql_generation.py
  venv/bin/python tests/test_sql_generation.py "your custom question here"
"""
import os
import sys
import time
import logging
import re
from pathlib import Path

# ── make project root importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.services.sql_engine import (
    DIRECT_SQL_TEMPLATE,
    _clean_sql_response,
    fix_common_sql_errors,
    validate_sql,
    _get_full_stored_procedure_guidance,
    _estimate_query_complexity,
    _get_ist_date_ranges,
)

# ── logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("sql_gen_test")

# ── config from .env ───────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_SIMPLE  = os.getenv("LLM_MODEL_SIMPLE", "gpt-4o-mini")
MODEL_COMPLEX = os.getenv("LLM_MODEL", "gpt-4o")
NOLOCK        = True   # same as main app default

# ── minimal schema (same tables as SCHEMA_TABLES in .env) ─────────────────────
SCHEMA_TEXT = """
Table: BookingData
Columns: PNRNo, AgentId, SupplierId, ProductId, ProductCountryid, ProductCityId,
         AgentBuyingPrice, CompanyBuyingPrice, BookingStatus, CreatedDate,
         CheckInDate, CheckOutDate, ClientNatinality

Table: AgentMaster_V1
Columns: AgentId, AgentName, AgentCode

Table: AgentTypeMapping
Columns: AgentId, AgentType

Table: suppliermaster_Report
Columns: EmployeeId, SupplierName

Table: Hotelchain
Columns: HotelId, HotelName, Country, City, Star, chain

Table: Master_Country
Columns: CountryID, Country

Table: Master_City
Columns: CityId, City
""".strip()


def run_test(question: str):
    logger.info("=" * 70)
    logger.info("QUESTION : %s", question)

    complexity = _estimate_query_complexity(question)

    # Local override: questions with temporal scope words ("this year", "this month",
    # "overview", "business", "ytd") need gpt-4o to honour relative_date_reference.
    _TEMPORAL_OVERRIDE_WORDS = {
        "this year", "this month", "ytd", "overview", "business look",
        "how does", "year to date",
    }
    q_lower = question.lower()
    if complexity == "simple_llm" and any(w in q_lower for w in _TEMPORAL_OVERRIDE_WORDS):
        complexity = "complex_llm"
        logger.info("COMPLEXITY: %s  (overridden → complex_llm for temporal/overview scope)", complexity)
    else:
        logger.info("COMPLEXITY: %s", complexity)

    # pick model + token cap — mirrors handle_query exactly
    if complexity == "simple_llm":
        model      = MODEL_SIMPLE
        max_tokens = 800
    else:
        model      = MODEL_COMPLEX
        max_tokens = 1500

    logger.info("MODEL     : %s  max_tokens=%s", model, max_tokens)

    # build the same vars as handle_query
    date_ranges = _get_ist_date_ranges()
    _today       = date_ranges.get('today_ist', 'unknown')
    _month_start = date_ranges.get('this_month_start', '')
    _month_end   = date_ranges.get('this_month_end', '')
    _year_start  = date_ranges.get('this_year_start', '')
    _year_end    = date_ranges.get('this_year_end', '')
    relative_date_reference = (
        f"IMPORTANT — use ONLY these dates, do NOT invent dates from training data:\n"
        f"  Today (IST)  : {_today}\n"
        f"  This month   : {_month_start} → {_month_end}\n"
        f"  This year    : {_year_start} → {_year_end}\n"
        f"When the user says 'this year', filter CreatedDate >= '{_year_start}' AND CreatedDate < '{_year_end}'."
    )

    guidance  = _get_full_stored_procedure_guidance()
    nolock_setting = "true" if NOLOCK else "false"

    prompt_vars = {
        "question":                 question,
        "context":                  "",          # no chat history in standalone test
        "few_shot_examples":        "",
        "retrieved_tables_hint":    "BookingData, AgentMaster_V1, AgentTypeMapping, suppliermaster_Report, Hotelchain, Master_Country, Master_City",
        "dialect_label":            "SQL Server (T-SQL)",
        "relative_date_reference":  relative_date_reference,
        "enable_nolock":            nolock_setting,
        "full_schema":              SCHEMA_TEXT,
        "stored_procedure_guidance": guidance,
    }

    # build chain — same as main app
    llm = ChatOpenAI(
        model=model,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.0,
    )
    chain = (
        ChatPromptTemplate.from_template(DIRECT_SQL_TEMPLATE)
        | llm.bind(stop=["\nSQLResult:"], max_tokens=max_tokens)
        | StrOutputParser()
    )

    t0 = time.perf_counter()
    raw = chain.invoke(prompt_vars)
    gen_ms = round((time.perf_counter() - t0) * 1000, 1)

    # clean — same as main app (lines 6133–6135)
    cleaned = _clean_sql_response(raw.strip())
    cleaned = fix_common_sql_errors(cleaned, dialect="sqlserver")
    is_valid, validation_msg = validate_sql(cleaned)

    logger.info("GEN TIME  : %s ms", gen_ms)
    logger.info("IS VALID  : %s  (%s)", is_valid, validation_msg or "ok")
    logger.info("-" * 70)
    logger.info("CLEANED SQL:\n%s", cleaned)
    logger.info("=" * 70)

    return cleaned, is_valid


# ── default questions to test ──────────────────────────────────────────────────
DEFAULT_QUESTIONS = [
    "Which nationalities had the biggest increase in bookings this year?",
    
]

if __name__ == "__main__":
    questions = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUESTIONS
    for q in questions:
        run_test(q)
        time.sleep(0.5)   # avoid rate-limit on rapid sequential calls
