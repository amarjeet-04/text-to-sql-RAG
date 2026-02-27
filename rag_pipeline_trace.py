"""
rag_pipeline_trace.py
─────────────────────
Step-by-step trace of the Text-to-SQL RAG pipeline with AUTO routing.

Covers only the RAG pipeline steps (Steps 0–5):
  Step 0 — Auto-routing (classify question → pick model + prompt)
  Step 1 — Vector store bootstrap (FAISS + HuggingFace embeddings)
  Step 2 — Query embedding + schema retrieval
  Step 3 — Prompt construction
  Step 4 — LLM call
  Step 5 — SQL extraction + validation

Each step is timestamped. The generated SQL query is printed at the end.
DB execution and result formatting are intentionally excluded.

Usage
  python rag_pipeline_trace.py
  python rag_pipeline_trace.py -q "Top 10 agents by revenue in 2024"
  python rag_pipeline_trace.py -q "YoY revenue growth by agent 2023 vs 2024"
  python rag_pipeline_trace.py -q "Compare supplier profit last year vs this year"
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from app.rag.rag_engine import RAGEngine

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── colours ──────────────────────────────────────────────────────────────────
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
DIM     = "\033[2m"
RESET   = "\033[0m"

# ── routing config per complexity tier ───────────────────────────────────────
ROUTING: Dict[str, Dict] = {
    "deterministic": {
        "model":       None,
        "prompt":      None,
        "color":       BLUE,
        "description": "Hard-coded SQL builder — no LLM call needed",
        "expected_ms": "~0 ms",
    },
    "simple_llm": {
        "model":       "gpt-4o-mini",
        "prompt":      "direct",
        "color":       GREEN,
        "description": "gpt-4o-mini  +  short direct-answer prompt  (SQL-only output)",
        "expected_ms": "~2–4 s",
    },
    "complex_llm": {
        "model":       None,          # reads LLM_MODEL from .env (gpt-4o)
        "prompt":      "cot",
        "color":       YELLOW,
        "description": "gpt-4o  +  full chain-of-thought prompt  (6 reasoning steps)",
        "expected_ms": "~8–10 s",
    },
}

# ── direct (short) prompt ─────────────────────────────────────────────────────
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


# ── print helpers ─────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def _banner(step: int, title: str, color: str = CYAN) -> None:
    print(f"\n{BOLD}{color}{'─'*64}{RESET}")
    print(f"{BOLD}{color}  STEP {step}: {title}{RESET}")
    print(f"{BOLD}{color}{'─'*64}{RESET}")

def _ok(label: str, value: Any = None, elapsed_ms: Optional[float] = None) -> None:
    ts    = _ts()
    suf   = f"  {DIM}(+{elapsed_ms:.0f} ms){RESET}" if elapsed_ms is not None else ""
    v_str = (str(value)[:117] + "...") if value and len(str(value)) > 120 else str(value) if value else None
    if v_str is None:
        print(f"  {GREEN}✓{RESET}  [{ts}] {label}{suf}")
    else:
        print(f"  {GREEN}✓{RESET}  [{ts}] {BOLD}{label}{RESET}: {v_str}{suf}")

def _info(label: str, value: Any = None) -> None:
    ts    = _ts()
    v_str = (str(value)[:197] + "...") if value and len(str(value)) > 200 else str(value) if value else None
    if v_str is None:
        print(f"  {YELLOW}ℹ{RESET}  [{ts}] {label}")
    else:
        print(f"  {YELLOW}ℹ{RESET}  [{ts}] {BOLD}{label}{RESET}: {v_str}")

def _route(label: str, value: Any = None, color: str = MAGENTA) -> None:
    ts    = _ts()
    v_str = f": {value}" if value is not None else ""
    print(f"  {color}▶{RESET}  [{ts}] {BOLD}{color}{label}{v_str}{RESET}")

def _err(label: str, exc: Exception) -> None:
    ts = _ts()
    print(f"  {RED}✗{RESET}  [{ts}] {BOLD}{label}{RESET}: {exc}")

def _section(title: str) -> None:
    pad = max(0, 50 - len(title))
    print(f"\n  {DIM}── {title} {'─'*pad}{RESET}")

def _preview(text: str, max_lines: int = 6) -> None:
    lines = text.strip().splitlines()
    for line in lines[:max_lines]:
        print(f"    {DIM}{line}{RESET}")
    if len(lines) > max_lines:
        print(f"    {DIM}... ({len(lines) - max_lines} more lines){RESET}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 0 — Auto-routing
# ═══════════════════════════════════════════════════════════════════════════

def step0_route(question: str) -> Dict:
    """Classify the question and return the routing config."""
    _banner(0, "AUTO-ROUTING  — classify question → pick model + prompt", color=MAGENTA)
    t0 = time.perf_counter()

    from backend.services.sql_engine import _estimate_query_complexity

    _info("Input question", question)

    _section("Routing rules (from _estimate_query_complexity)")
    _info("deterministic", "business overview phrases  OR  top-N pattern (e.g. 'top 10 X by Y')")
    _info("complex_llm  ", "contains: growth/yoy/ytd/pytd/trend/compared/versus/increase/decrease/change/previous year")
    _info("             ", "OR question mentions 2+ dimensions (agent+country, hotel+nationality, …)")
    _info("             ", "OR question word count > 12")
    _info("simple_llm   ", "everything else — single dimension, single metric, short question")

    q = question.lower()
    words_list = q.split()
    words_set  = set(re.findall(r"[a-z_]+", q))

    OVERVIEW_PHRASES = {"business overview", "executive summary", "performance summary",
                        "how is business", "how does my business"}
    TOPN_PATTERN     = re.compile(r"\b(top|bottom)\s+\d+\b", re.I)
    complex_keywords = {"growth","compared","versus","trend","yoy","ytd","pytd",
                        "correlation","increase","decrease","change","previous"}
    dimension_words  = {"agent","supplier","country","city","hotel","chain","nationality"}
    matched_complex  = complex_keywords & words_set
    matched_dims     = dimension_words & words_set
    is_topn          = bool(TOPN_PATTERN.search(question))
    is_overview      = any(ph in q for ph in OVERVIEW_PHRASES)

    _section("Classifying …")
    complexity = _estimate_query_complexity(question)
    route_cfg  = ROUTING[complexity]
    color      = route_cfg["color"]

    _section("Evidence for this classification")
    _info("Word count",            len(words_list))
    _info("Top-N pattern match",   f"YES — '{TOPN_PATTERN.search(question).group()}'" if is_topn else "no")
    _info("Overview phrase match", "YES" if is_overview else "no")
    if matched_complex:
        _info("Complex keywords found", ", ".join(sorted(matched_complex)))
    else:
        _info("Complex keywords found", "none")
    if matched_dims:
        _info("Dimension words found",  f"{sorted(matched_dims)}  ({len(matched_dims)} dimension{'s' if len(matched_dims)>1 else ''})")
    else:
        _info("Dimension words found",  "none")

    _section("Rule that triggered this tier")
    if complexity == "deterministic":
        if is_overview:
            _info("Trigger", "overview phrase matched → deterministic business overview SQL")
        elif is_topn:
            _info("Trigger", "top-N pattern matched → deterministic top-N SQL builder")
        else:
            _info("Trigger", "other deterministic pattern")
    elif complexity == "complex_llm":
        if len(words_list) > 12:
            _info("Trigger", f"word count {len(words_list)} > 12 → complex_llm")
        elif matched_complex:
            _info("Trigger", f"complex keywords {sorted(matched_complex)} → complex_llm")
        elif len(matched_dims) >= 2:
            _info("Trigger", f"2+ dimensions {sorted(matched_dims)} → complex_llm")
    else:
        _info("Trigger", "no complex/deterministic signals → simple_llm (default)")

    _section("Routing decision")
    _route("Complexity tier", complexity.upper(), color=color)
    _route("→ Model",         route_cfg["model"] or os.getenv("LLM_MODEL", "gpt-4o"), color=color)
    _route("→ Prompt",        "direct (SQL-only, ~100-200 output tokens)" if route_cfg["prompt"] == "direct"
                              else "chain-of-thought (6 reasoning steps, ~1500-2000 output tokens)" if route_cfg["prompt"] == "cot"
                              else "NONE — deterministic SQL builder", color=color)
    _route("→ Expected time", route_cfg["expected_ms"], color=color)
    _route("→ Reason",        route_cfg["description"], color=color)

    elapsed = (time.perf_counter() - t0) * 1000
    _ok("STEP 0 COMPLETE — routing decided", elapsed_ms=elapsed)
    return {"complexity": complexity, "route": route_cfg, "step_ms": elapsed}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Vector store bootstrap
# ═══════════════════════════════════════════════════════════════════════════

def step1_build_vector_store() -> "RAGEngine":
    _banner(1, "VECTOR STORE BOOTSTRAP  (FAISS + HuggingFace embeddings)")
    t0 = time.perf_counter()

    _info("Loading RAGEngine …")
    from app.rag.rag_engine import RAGEngine
    _ok("RAGEngine imported", elapsed_ms=(time.perf_counter() - t0) * 1000)

    _section("Initialising embedder")
    t1 = time.perf_counter()
    engine = RAGEngine(model_name="sentence-transformers/all-MiniLM-L6-v2", enable_embeddings=True)
    embedder_ok = engine.embedder is not None
    _ok("Embedder ready" if embedder_ok else "Embedder unavailable (keyword fallback)",
        elapsed_ms=(time.perf_counter() - t1) * 1000)
    if embedder_ok:
        _info("Embedding model", engine.model_name)

    _section("Populating schema documents")
    t2 = time.perf_counter()
    engine.load_default_schema()
    _ok("Schema docs loaded", f"{len(engine.documents)} documents",
        elapsed_ms=(time.perf_counter() - t2) * 1000)
    _ok("FAISS index built" if engine.vector_store else "FAISS index NOT built (keyword fallback)")

    _section("Document breakdown")
    counts: Dict[str, int] = {}
    for d in engine.documents:
        counts[d.doc_type] = counts.get(d.doc_type, 0) + 1
    for dtype, n in sorted(counts.items()):
        _info(f"  doc_type={dtype}", f"{n} docs")
    _info("Cache TTL", f"{engine.CACHE_TTL}s  (max {engine.CACHE_MAX} entries)")

    elapsed = (time.perf_counter() - t0) * 1000
    _ok("STEP 1 COMPLETE — vector store ready", elapsed_ms=elapsed)
    return engine, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Query embedding + retrieval
# ═══════════════════════════════════════════════════════════════════════════

def step2_embed_and_retrieve(engine: "RAGEngine", question: str) -> Dict:
    _banner(2, "QUERY EMBEDDING + SCHEMA RETRIEVAL FROM VECTOR STORE")
    t0 = time.perf_counter()
    _info("Input question", question)

    _section("Embedding the query")
    t1 = time.perf_counter()
    if engine.embedder:
        import numpy as np
        vec = np.array(engine.embedder.embed_query(question), dtype="float32")
        _ok("Query embedded", f"dim={len(vec)}", elapsed_ms=(time.perf_counter() - t1) * 1000)
        _info("First 8 dims", "  ".join(f"{v:+.4f}" for v in vec[:8]))
    else:
        _info("No embedder — keyword fallback")

    _section("Follow-up / skip detection")
    _info("Is follow-up / too short?", engine._is_followup_or_simple(question))
    eff_k = engine._effective_k(question, top_k=4)
    _info("Effective top_k",           eff_k)
    _info("RAG cache key (MD5)",        engine._cache_key(question, eff_k))

    _section("FAISS similarity search")
    t2 = time.perf_counter()
    rag_context = engine.retrieve(query=question, top_k=4, fast_mode=False, skip_for_followup=False)
    _ok("Retrieval complete", elapsed_ms=(time.perf_counter() - t2) * 1000)

    examples = rag_context.get("examples", [])
    rules    = rag_context.get("rules",    [])
    tables   = rag_context.get("tables",   [])
    _info("Retrieved examples", len(examples))
    for i, ex in enumerate(examples, 1):
        _info(f"  example[{i}].question", ex.get("question", ""))
    _info("Retrieved rules", len(rules))
    for i, r in enumerate(rules, 1):
        _info(f"  rule[{i}]", r.get("name", r.get("content", "")[:60]))
    _info("Retrieved tables", len(tables))

    elapsed = (time.perf_counter() - t0) * 1000
    _ok("STEP 2 COMPLETE — context retrieved from FAISS", elapsed_ms=elapsed)
    return rag_context, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Prompt construction
# ═══════════════════════════════════════════════════════════════════════════

def step3_build_prompt(question: str, rag_context: Dict, route_cfg: Dict) -> Dict:
    _banner(3, "CONTEXT ASSEMBLY + PROMPT CONSTRUCTION")
    t0 = time.perf_counter()

    use_direct = route_cfg["prompt"] == "direct"

    from backend.services.sql_engine import (
        SQL_TEMPLATE,
        _build_relative_date_reference,
        _dialect_label,
        DEFAULT_SQL_DIALECT,
    )

    _section("Template selected by router")
    color = route_cfg["color"]
    if route_cfg["prompt"] is None:
        _route("Template", "NONE — deterministic path, skipping LLM", color=color)
    elif use_direct:
        _route("Template", "DIRECT  — SQL-only, no chain-of-thought", color=color)
        _route("Output tokens", "~100–200  (saves ~3–5 s vs chain-of-thought)", color=color)
        _route("Prompt caching", "static sections first → OpenAI will cache the prefix across calls", color=color)
    else:
        _route("Template", "CHAIN-OF-THOUGHT  — 6-step reasoning required for this complexity", color=color)
        _route("Output tokens", "~1500–2000  (LLM must reason through date math + multi-join logic)", color=color)

    _section("Building few-shot block from retrieved examples")
    examples = rag_context.get("examples", [])
    few_shot_str = ""
    if examples:
        parts = ["EXAMPLE QUERIES (follow these patterns closely):"]
        for ex in examples:
            q_ = ex.get("question", ""); s_ = ex.get("sql", "")
            if q_ and s_:
                parts.append(f"Q: {q_}\nSQL: {s_}")
        few_shot_str = "\n\n".join(parts)
        _info("Few-shot block", f"{len(examples)} examples, {len(few_shot_str)} chars")
    else:
        _info("No few-shot examples retrieved")

    _section("RAG table hints")
    raw_tables = rag_context.get("tables", []) or []
    rag_hints  = [t["table"] if isinstance(t, dict) else str(t) for t in raw_tables if t]
    retrieved_tables_hint = ", ".join(rag_hints[:6]) if rag_hints else "none"
    _info("Table hints", retrieved_tables_hint)

    _section("Schema text")
    schema_text = (
        "-- NOTE: abbreviated for trace — production uses full INFORMATION_SCHEMA dump.\n"
        "TABLE: BookingData  (PNRNo, AgentId, SupplierId, ProductCountryid,\n"
        "                     BookingStatus, AgentBuyingPrice, CompanyBuyingPrice, CreatedDate, CheckInDate)\n"
        "TABLE: AgentMaster_V1  (AgentId, AgentName, AgentType)\n"
        "TABLE: suppliermaster_Report  (EmployeeId, SupplierName)\n"
        "TABLE: Master_Country  (CountryID, Country)\n"
        "TABLE: Master_City     (CityID, City, CountryID)\n"
    )
    _info("Schema chars", len(schema_text))
    _preview(schema_text)

    _section("Business rules")
    rules = rag_context.get("rules", [])
    stored_guidance = "\n".join(r.get("content", r.get("name", "")) for r in rules) if rules else (
        "• BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')\n"
        "• Never use CAST(date_col AS DATE) in WHERE; use >= / < boundaries.\n"
        "• Lookup tables have duplicates — wrap in CTE with SELECT DISTINCT."
    )
    _info("Rules chars", len(stored_guidance))
    _preview(stored_guidance)

    _section("Date reference + dialect")
    dialect    = DEFAULT_SQL_DIALECT
    date_ref   = _build_relative_date_reference()
    dial_label = _dialect_label(dialect)
    _info("Dialect",    dial_label)
    _info("TODAY (IST)", date_ref)
    _info("Conversation context", "(no prior conversation)")

    _section("Assembling final prompt")
    prompt_vars = dict(
        dialect_label             = dial_label,
        enable_nolock             = str(dialect == "sqlserver"),
        relative_date_reference   = date_ref,
        full_schema               = schema_text,
        stored_procedure_guidance = stored_guidance,
        context                   = "(no prior conversation)",
        retrieved_tables_hint     = retrieved_tables_hint,
        few_shot_examples         = few_shot_str,
        question                  = question,
    )

    template      = DIRECT_SQL_TEMPLATE if use_direct else SQL_TEMPLATE
    template_name = "DIRECT" if use_direct else "COT"
    final_prompt  = template.format(**prompt_vars)

    _info("Template", template_name)
    _ok("Prompt assembled", f"{len(final_prompt)} chars  (~{len(final_prompt)//4} input tokens)",
        elapsed_ms=(time.perf_counter() - t0) * 1000)

    _section("Prompt preview (first 30 lines)")
    _preview(final_prompt, max_lines=30)

    elapsed = (time.perf_counter() - t0) * 1000
    return {"prompt_vars": prompt_vars, "final_prompt_text": final_prompt,
            "template": template, "use_direct": use_direct, "step_ms": elapsed}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — LLM call
# ═══════════════════════════════════════════════════════════════════════════

def step4_call_llm(prompt_vars: Dict, route_cfg: Dict, template: str,
                   complexity: str) -> str:
    _banner(4, "LLM CALL")
    t0 = time.perf_counter()

    if complexity == "deterministic":
        _route("Deterministic path — no LLM call needed for this question",
               color=route_cfg["color"])
        _info("Production would call _deterministic_timeout_fallback_sql() or build_topn_fallback_sql()")
        _ok("STEP 4 SKIPPED — deterministic SQL will be used", elapsed_ms=0)
        return "__DETERMINISTIC__", 0.0

    api_key   = os.getenv("OPENAI_API_KEY", "")
    env_model = os.getenv("LLM_MODEL", "gpt-4o")
    llm_model = route_cfg["model"] or env_model
    timeout_s = int(os.getenv("LLM_SQL_TIMEOUT_MS", "15000")) / 1000

    color = route_cfg["color"]
    _route("Model chosen by router",   llm_model, color=color)
    _route("Prompt type",              "direct (SQL-only)" if route_cfg["prompt"] == "direct"
                                       else "chain-of-thought (6 steps)", color=color)
    _route("Expected latency",         route_cfg["expected_ms"], color=color)
    _info("Timeout", f"{timeout_s:.1f}s")
    _info("API key present", bool(api_key and api_key != "your-openai-api-key-here"))

    if not api_key or api_key == "your-openai-api-key-here":
        _info("OPENAI_API_KEY not set — using mock response (set key in .env for real call)")
        mock_sql = (
            "WITH SM AS (\n    SELECT DISTINCT EmployeeId, SupplierName\n"
            "    FROM dbo.suppliermaster_Report WITH (NOLOCK)\n)\n"
            "SELECT TOP 5\n    SM.SupplierName AS [Supplier Name],\n"
            "    COUNT(DISTINCT BD.PNRNo) AS [Total Bookings]\n"
            "FROM dbo.BookingData BD WITH (NOLOCK)\n"
            "LEFT JOIN SM ON SM.EmployeeId = BD.SupplierId\n"
            "WHERE BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')\n"
            "  AND BD.CreatedDate >= '2024-01-01'\n  AND BD.CreatedDate < '2025-01-01'\n"
            "GROUP BY SM.SupplierName\nORDER BY [Total Bookings] DESC"
        )
        raw = f"```sql\n{mock_sql}\n```"
        elapsed = (time.perf_counter() - t0) * 1000
        _ok("Mock LLM response generated", elapsed_ms=elapsed)
        _preview(raw, max_lines=20)
        return raw, elapsed

    raw_response = ""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        _section("Building LangChain chain")
        llm   = ChatOpenAI(model=llm_model, temperature=0,
                           request_timeout=timeout_s, openai_api_key=api_key)
        chain = ChatPromptTemplate.from_template(template) | llm | StrOutputParser()
        _ok("Chain constructed", elapsed_ms=(time.perf_counter() - t0) * 1000)

        _section("Invoking LLM …")
        t1 = time.perf_counter()
        raw_response = chain.invoke(prompt_vars)
        llm_ms = (time.perf_counter() - t1) * 1000

        _ok("LLM responded", f"{len(raw_response)} chars", elapsed_ms=llm_ms)

        est_out = len(raw_response) // 4
        if route_cfg["prompt"] == "direct":
            _route(f"Output tokens used: ~{est_out}  (chain-of-thought would have been ~1500–2000)",
                   color=color)
        else:
            _route(f"Output tokens used: ~{est_out}  (includes full reasoning steps)",
                   color=color)

        _section("Raw LLM output preview")
        _preview(raw_response, max_lines=30)

    except Exception as exc:
        _err("LLM call failed", exc)

    elapsed = (time.perf_counter() - t0) * 1000
    _ok("STEP 4 COMPLETE", elapsed_ms=elapsed)
    return raw_response, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — SQL extraction + validation
# ═══════════════════════════════════════════════════════════════════════════

def step5_extract_sql(raw_response: str, complexity: str) -> str:
    _banner(5, "SQL EXTRACTION + VALIDATION")
    t0 = time.perf_counter()

    if complexity == "deterministic":
        _info("Deterministic path — SQL built by hard-coded builder, no LLM SQL to extract")
        _ok("STEP 5 SKIPPED — deterministic SQL built at query time", elapsed_ms=0)
        return "(deterministic — sql built at query time)", 0.0

    from backend.services.sql_engine import _clean_sql_response, validate_sql, BLOCKED_KEYWORDS

    _section("Cleaning raw LLM response")
    t1 = time.perf_counter()
    cleaned_sql = _clean_sql_response(raw_response)
    _ok("SQL extracted from markdown fence", elapsed_ms=(time.perf_counter() - t1) * 1000)

    _section("Extracted SQL")
    if cleaned_sql:
        _preview(cleaned_sql, max_lines=25)
    else:
        _info("Empty SQL — extraction returned nothing")

    _section("Validating SQL")
    t2 = time.perf_counter()
    is_valid, err_msg = validate_sql(cleaned_sql)
    if is_valid:
        _ok("SQL passed validation", elapsed_ms=(time.perf_counter() - t2) * 1000)
    else:
        _err(f"SQL failed validation: {err_msg}", Exception(err_msg))

    _section("Safety check — blocked keywords")
    blocked = [kw for kw in BLOCKED_KEYWORDS if re.search(rf"\b{kw}\b", cleaned_sql.lower())]
    if blocked:
        _err("Blocked keywords found", Exception(str(blocked)))
    else:
        _ok("No blocked keywords (DROP/DELETE/INSERT/UPDATE/…)")

    elapsed = (time.perf_counter() - t0) * 1000
    _ok("STEP 5 COMPLETE — final SQL ready", elapsed_ms=elapsed)
    return cleaned_sql, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Trace the Text-to-SQL RAG pipeline (Steps 0–5, no DB execution).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python rag_pipeline_trace.py -q "Top 10 agents by revenue in 2024"\n'
            '      → simple_llm  : gpt-4o-mini + direct prompt  (~2–4 s)\n\n'
            '  python rag_pipeline_trace.py -q "YoY revenue growth by agent 2023 vs 2024"\n'
            '      → complex_llm : gpt-4o + chain-of-thought    (~8–10 s)\n\n'
            '  python rag_pipeline_trace.py -q "How does my business look"\n'
            '      → deterministic: hard-coded SQL builder       (~0 ms)\n'
        ),
    )
    parser.add_argument("--question", "-q",
        default="Show me monthly revenue and profit for each supplier over the last 6 months",
        help="Natural-language question to trace through the pipeline.")
    args     = parser.parse_args()
    question = args.question.strip()

    print(f"\n{BOLD}{'═'*64}{RESET}")
    print(f"{BOLD}  TEXT-TO-SQL RAG PIPELINE TRACE  (Steps 0–5){RESET}")
    print(f"{BOLD}{'═'*64}{RESET}")
    print(f"  Question : {BOLD}{question}{RESET}")
    print(f"  Started  : {_ts()}")
    print(f"{BOLD}{'═'*64}{RESET}")

    t_pipeline = time.perf_counter()

    routing    = step0_route(question)
    complexity = routing["complexity"]
    route_cfg  = routing["route"]
    color      = route_cfg["color"]
    ms0        = routing["step_ms"]

    engine,      ms1 = step1_build_vector_store()
    rag_context, ms2 = step2_embed_and_retrieve(engine, question)
    prompt_res       = step3_build_prompt(question, rag_context, route_cfg)
    ms3              = prompt_res["step_ms"]
    raw_resp,    ms4 = step4_call_llm(prompt_res["prompt_vars"], route_cfg,
                                      prompt_res["template"], complexity)
    final_sql,   ms5 = step5_extract_sql(raw_resp, complexity)

    total_ms = (time.perf_counter() - t_pipeline) * 1000

    # ── generated SQL ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}{GREEN}{'═'*64}{RESET}")
    print(f"{BOLD}{GREEN}  GENERATED SQL QUERY{RESET}")
    print(f"{BOLD}{GREEN}{'═'*64}{RESET}")
    if final_sql and not final_sql.startswith("("):
        for line in final_sql.strip().splitlines():
            print(f"  {BOLD}{line}{RESET}")
    else:
        print(f"  {DIM}{final_sql}{RESET}")
    print(f"{BOLD}{GREEN}{'═'*64}{RESET}")

    # ── pipeline summary ──────────────────────────────────────────────────────
    model_used = route_cfg["model"] or os.getenv("LLM_MODEL", "gpt-4o")
    print(f"\n{BOLD}{CYAN}{'═'*64}{RESET}")
    print(f"{BOLD}{CYAN}  PIPELINE SUMMARY{RESET}")
    print(f"{BOLD}{CYAN}{'═'*64}{RESET}")
    print(f"  Question        : {question}")
    print(f"  Complexity tier : {BOLD}{color}{complexity.upper()}{RESET}")
    print(f"  Model used      : {BOLD}{model_used}{RESET}")
    print(f"  Prompt type     : {'direct (SQL-only)' if route_cfg['prompt'] == 'direct' else 'chain-of-thought (6 steps)' if route_cfg['prompt'] == 'cot' else 'none (deterministic)'}")
    print(f"  Docs in index   : {len(engine.documents)}")
    print(f"  Examples found  : {len(rag_context.get('examples', []))}")
    print(f"  Rules found     : {len(rag_context.get('rules', []))}")
    print(f"  Prompt chars    : {len(prompt_res['final_prompt_text'])}")
    print(f"  SQL lines       : {len(final_sql.splitlines()) if not final_sql.startswith('(') else 'N/A (deterministic)'}")
    print(f"{BOLD}{CYAN}{'─'*64}{RESET}")
    print(f"{BOLD}{CYAN}  STAGE TIMINGS{RESET}")
    print(f"{BOLD}{CYAN}{'─'*64}{RESET}")
    print(f"  Step 0  auto-routing      :  {ms0:>8.0f} ms")
    print(f"  Step 1  vector store boot :  {ms1:>8.0f} ms")
    print(f"  Step 2  RAG retrieval     :  {ms2:>8.0f} ms")
    print(f"  Step 3  prompt assembly   :  {ms3:>8.0f} ms")
    print(f"  Step 4  LLM call          :  {ms4:>8.0f} ms")
    print(f"  Step 5  SQL extraction    :  {ms5:>8.0f} ms")
    print(f"{BOLD}{CYAN}{'─'*64}{RESET}")
    print(f"  {BOLD}Total wall time           :  {total_ms:>8.0f} ms{RESET}")
    print(f"  Expected (live)           :  {route_cfg['expected_ms']}")
    print(f"{BOLD}{CYAN}{'═'*64}{RESET}\n")


if __name__ == "__main__":
    main()
