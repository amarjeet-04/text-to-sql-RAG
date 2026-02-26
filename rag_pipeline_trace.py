"""
rag_pipeline_trace.py
─────────────────────
Step-by-step trace of the Text-to-SQL RAG pipeline with AUTO routing.

The script reads your question, classifies its complexity using the same
_estimate_query_complexity() function the production server uses, then
automatically picks the right model + prompt — no --mode flag needed.

Routing table
  deterministic  →  no LLM at all  (hard-coded SQL builders)    ~0 ms
  simple_llm     →  gpt-4o-mini  + short direct-answer prompt   ~2–4 s
  complex_llm    →  gpt-4o       + full chain-of-thought prompt  ~8–10 s

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

# ── colours ─────────────────────────────────────────────────────────────────
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
DIM     = "\033[2m"
RESET   = "\033[0m"

# ── routing config per complexity tier ──────────────────────────────────────
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

# ── direct (short) prompt ────────────────────────────────────────────────────
# Static sections (schema, rules, examples) first → maximises OpenAI prefix cache.
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


# ── print helpers ────────────────────────────────────────────────────────────

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
# STEP 0 — Auto-routing: classify question → choose model + prompt
# ═══════════════════════════════════════════════════════════════════════════

def step0_route(question: str) -> Dict:
    """Classify the question and return the routing config."""
    _banner(0, "AUTO-ROUTING  — classify question → pick model + prompt", color=MAGENTA)
    t0 = time.perf_counter()

    from backend.services.sql_engine import _estimate_query_complexity

    _info("Input question", question)

    # ── show the rules the classifier uses ──────────────────────────────────
    _section("Routing rules (from _estimate_query_complexity)")
    _info("deterministic", "business overview phrases  OR  top-N pattern (e.g. 'top 10 X by Y')")
    _info("complex_llm  ", "contains: growth/yoy/ytd/pytd/trend/compared/versus/increase/decrease/change/previous year")
    _info("             ", "OR question mentions 2+ dimensions (agent+country, hotel+nationality, …)")
    _info("             ", "OR question word count > 12")
    _info("simple_llm   ", "everything else — single dimension, single metric, short question")

    # ── pre-check deterministic triggers so we can explain them ─────────────
    q = question.lower()
    words_list = q.split()
    words_set  = set(re.findall(r"[a-z_]+", q))

    # mirrors logic inside _estimate_query_complexity
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

    # ── classify ────────────────────────────────────────────────────────────
    _section("Classifying …")
    complexity = _estimate_query_complexity(question)
    route_cfg  = ROUTING[complexity]
    color      = route_cfg["color"]

    _section("Evidence for this classification")
    _info("Word count",            len(words_list))
    _info("Top-N pattern match",   f"YES — '{TOPN_PATTERN.search(question).group()}'" if is_topn else "no")
    _info("Overview phrase match", f"YES" if is_overview else "no")
    if matched_complex:
        _info("Complex keywords found", ", ".join(sorted(matched_complex)))
    else:
        _info("Complex keywords found", "none")
    if matched_dims:
        _info("Dimension words found",  f"{sorted(matched_dims)}  ({len(matched_dims)} dimension{'s' if len(matched_dims)>1 else ''})")
    else:
        _info("Dimension words found",  "none")

    # explain which rule fired
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

    elapsed = (time.perf_counter()-t0)*1000
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
    _ok("RAGEngine imported", elapsed_ms=(time.perf_counter()-t0)*1000)

    _section("Initialising embedder")
    t1 = time.perf_counter()
    engine = RAGEngine(model_name="sentence-transformers/all-MiniLM-L6-v2", enable_embeddings=True)
    embedder_ok = engine.embedder is not None
    _ok("Embedder ready" if embedder_ok else "Embedder unavailable (keyword fallback)",
        elapsed_ms=(time.perf_counter()-t1)*1000)
    if embedder_ok:
        _info("Embedding model", engine.model_name)

    _section("Populating schema documents")
    t2 = time.perf_counter()
    engine.load_default_schema()
    _ok("Schema docs loaded", f"{len(engine.documents)} documents",
        elapsed_ms=(time.perf_counter()-t2)*1000)
    _ok("FAISS index built" if engine.vector_store else "FAISS index NOT built (keyword fallback)")

    _section("Document breakdown")
    counts: Dict[str, int] = {}
    for d in engine.documents:
        counts[d.doc_type] = counts.get(d.doc_type, 0) + 1
    for dtype, n in sorted(counts.items()):
        _info(f"  doc_type={dtype}", f"{n} docs")
    _info("Cache TTL", f"{engine.CACHE_TTL}s  (max {engine.CACHE_MAX} entries)")

    elapsed = (time.perf_counter()-t0)*1000
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
        _ok("Query embedded", f"dim={len(vec)}", elapsed_ms=(time.perf_counter()-t1)*1000)
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
    _ok("Retrieval complete", elapsed_ms=(time.perf_counter()-t2)*1000)

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

    elapsed = (time.perf_counter()-t0)*1000
    _ok("STEP 2 COMPLETE — context retrieved from FAISS", elapsed_ms=elapsed)
    return rag_context, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Prompt construction (template chosen by router)
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

    # ── show which template was chosen and why ───────────────────────────────
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

    # ── few-shot examples ───────────────────────────────────────────────────
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

    # ── RAG table hints ─────────────────────────────────────────────────────
    _section("RAG table hints")
    raw_tables = rag_context.get("tables", []) or []
    rag_hints  = [t["table"] if isinstance(t, dict) else str(t) for t in raw_tables if t]
    retrieved_tables_hint = ", ".join(rag_hints[:6]) if rag_hints else "none"
    _info("Table hints", retrieved_tables_hint)

    # ── schema (abbreviated for trace) ──────────────────────────────────────
    _section("Schema text")
    schema_text = (
        "-- NOTE: abbreviated for trace — production uses full INFORMATION_SCHEMA dump.\n"
        "TABLE: BookingData  (PNRNo, AgentId, SupplierId, ProductCountryid,\n"
        "                     BookingStatus, AgentBuyingPrice, CreatedDate, CheckInDate)\n"
        "TABLE: AgentMaster_V1  (AgentId, AgentName, AgentType)\n"
        "TABLE: suppliermaster_Report  (EmployeeId, SupplierName)\n"
        "TABLE: Master_Country  (CountryID, Country)\n"
        "TABLE: Master_City     (CityID, City, CountryID)\n"
    )
    _info("Schema chars", len(schema_text))
    _preview(schema_text)

    # ── business rules ───────────────────────────────────────────────────────
    _section("Business rules")
    rules = rag_context.get("rules", [])
    stored_guidance = "\n".join(r.get("content", r.get("name", "")) for r in rules) if rules else (
        "• BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')\n"
        "• Never use CAST(date_col AS DATE) in WHERE; use >= / < boundaries.\n"
        "• Lookup tables have duplicates — wrap in CTE with SELECT DISTINCT."
    )
    _info("Rules chars", len(stored_guidance))
    _preview(stored_guidance)

    # ── date + dialect ───────────────────────────────────────────────────────
    _section("Date reference + dialect")
    dialect    = DEFAULT_SQL_DIALECT
    date_ref   = _build_relative_date_reference()
    dial_label = _dialect_label(dialect)
    _info("Dialect",    dial_label)
    _info("TODAY (IST)", date_ref)
    _info("Conversation context", "(no prior conversation)")

    # ── assemble ─────────────────────────────────────────────────────────────
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
        elapsed_ms=(time.perf_counter()-t0)*1000)

    _section("Prompt preview (first 30 lines)")
    _preview(final_prompt, max_lines=30)

    elapsed = (time.perf_counter()-t0)*1000
    return {"prompt_vars": prompt_vars, "final_prompt_text": final_prompt,
            "template": template, "use_direct": use_direct, "step_ms": elapsed}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — LLM interaction
# ═══════════════════════════════════════════════════════════════════════════

def step4_call_llm(prompt_vars: Dict, route_cfg: Dict, template: str,
                   complexity: str) -> str:
    _banner(4, "LLM INTERACTION")
    t0 = time.perf_counter()

    # ── deterministic short-circuit ─────────────────────────────────────────
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
        elapsed = (time.perf_counter()-t0)*1000
        _ok("Mock LLM response generated", elapsed_ms=elapsed)
        _preview(raw, max_lines=20)
        return raw, elapsed

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        _section("Building LangChain chain")
        llm   = ChatOpenAI(model=llm_model, temperature=0,
                           request_timeout=timeout_s, openai_api_key=api_key)
        chain = ChatPromptTemplate.from_template(template) | llm | StrOutputParser()
        _ok("Chain constructed", elapsed_ms=(time.perf_counter()-t0)*1000)

        _section("Invoking LLM …")
        t1 = time.perf_counter()
        raw_response = chain.invoke(prompt_vars)
        llm_ms = (time.perf_counter()-t1)*1000

        _ok("LLM responded", f"{len(raw_response)} chars", elapsed_ms=llm_ms)

        # show how many output tokens were used vs the other path
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
        raw_response = ""

    elapsed = (time.perf_counter()-t0)*1000
    _ok("STEP 4 COMPLETE", elapsed_ms=elapsed)
    return raw_response, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — SQL extraction + validation
# ═══════════════════════════════════════════════════════════════════════════

def step5_extract_sql(raw_response: str, complexity: str) -> str:
    _banner(5, "SQL EXTRACTION + VALIDATION")
    t0 = time.perf_counter()

    if complexity == "deterministic":
        _info("Deterministic path — SQL will be built in Step 6 (no LLM SQL to extract here)")
        _ok("STEP 5 SKIPPED — deterministic SQL built at execution time", elapsed_ms=0)
        return "(deterministic — sql built in step6)", 0.0

    from backend.services.sql_engine import _clean_sql_response, validate_sql, BLOCKED_KEYWORDS

    _section("Cleaning raw LLM response")
    t1 = time.perf_counter()
    cleaned_sql = _clean_sql_response(raw_response)
    _ok("SQL extracted from markdown fence", elapsed_ms=(time.perf_counter()-t1)*1000)

    _section("Extracted SQL")
    _preview(cleaned_sql, max_lines=25) if cleaned_sql else _info("Empty SQL — extraction returned nothing")

    _section("Validating SQL")
    t2 = time.perf_counter()
    is_valid, err_msg = validate_sql(cleaned_sql)
    if is_valid:
        _ok("SQL passed validation", elapsed_ms=(time.perf_counter()-t2)*1000)
    else:
        _err(f"SQL failed validation: {err_msg}", Exception(err_msg))

    _section("Safety check — blocked keywords")
    blocked = [kw for kw in BLOCKED_KEYWORDS if re.search(rf"\b{kw}\b", cleaned_sql.lower())]
    if blocked:
        _err("Blocked keywords found", Exception(str(blocked)))
    else:
        _ok("No blocked keywords (DROP/DELETE/INSERT/UPDATE/…)")

    elapsed = (time.perf_counter()-t0)*1000
    _ok("STEP 5 COMPLETE — final SQL ready", elapsed_ms=elapsed)

    print(f"\n{BOLD}{GREEN}{'═'*64}{RESET}")
    print(f"{BOLD}{GREEN}  FINAL SQL QUERY{RESET}")
    print(f"{BOLD}{GREEN}{'═'*64}{RESET}")
    print(f"{BOLD}{cleaned_sql}{RESET}")
    print(f"{BOLD}{GREEN}{'═'*64}{RESET}\n")
    return cleaned_sql, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — DB execution
# ═══════════════════════════════════════════════════════════════════════════

def step6_db_execution(sql: str, complexity: str, question: str = "") -> Dict:
    """Connect to the real DB and execute the SQL, measuring wall-clock time.

    For the deterministic tier, builds the SQL via _deterministic_timeout_fallback_sql()
    before executing — so all three tiers always hit the DB.
    """
    _banner(6, "DATABASE EXECUTION")
    t0 = time.perf_counter()

    result: Dict = {
        "rows": [],
        "columns": [],
        "row_count": 0,
        "error": None,
        "skipped": False,
        "exec_ms": 0.0,
        "sql_used": sql,
    }

    # ── build DB connection from .env ────────────────────────────────────────
    _section("Reading DB config from .env")
    host     = os.getenv("DB_HOST", "")
    port     = os.getenv("DB_PORT", "1433")
    user     = os.getenv("DB_USER", "")
    password = os.getenv("DB_PASSWORD", "")
    database = os.getenv("DB_NAME", "")

    if not host or not user:
        _info("DB_HOST / DB_USER not set in .env — skipping real execution")
        result["skipped"] = True
        _ok("STEP 6 SKIPPED — no DB config", elapsed_ms=0)
        return result

    _info("Host",     f"{host}:{port}")
    _info("Database", database)
    _info("User",     user)

    try:
        from urllib.parse import quote_plus
        from sqlalchemy import create_engine, text as sa_text
        import pandas as pd

        _section("Connecting to database (lightweight engine)")
        t_conn = time.perf_counter()
        odbc_params = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={host},{port};"
            f"DATABASE={database};"
            f"UID={user};"
            f"PWD={password};"
            f"LoginTimeout=10;"
            f"QueryTimeout=30;"
            f"TrustServerCertificate=yes;"
        )
        conn_uri = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_params)}"
        sa_engine = create_engine(conn_uri, pool_pre_ping=False)

        with sa_engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        conn_ms = (time.perf_counter() - t_conn) * 1000
        _ok("Connected", f"{host}:{port}/{database}", elapsed_ms=conn_ms)

        # ── deterministic: build SQL now that we have a live connection ───────
        if complexity == "deterministic":
            _section("Building deterministic SQL (using live DB connection)")
            from backend.services.sql_engine import _deterministic_timeout_fallback_sql
            from app.db_utils import DatabaseConfig, create_database_with_views
            try:
                cfg = DatabaseConfig(
                    host=host, port=port,
                    username=user, password=password,
                    database=database,
                    include_tables=os.getenv("SCHEMA_TABLES", "").split(",") or None,
                )
                db_obj = create_database_with_views(cfg)
                det_sql, det_kind = _deterministic_timeout_fallback_sql(question, db_obj)
            except Exception as e:
                _info(f"create_database_with_views failed ({e}) — trying without db_obj")
                det_sql, det_kind = _deterministic_timeout_fallback_sql(question, None)

            if det_sql:
                _ok(f"Deterministic SQL built", f"kind={det_kind}")
                _preview(det_sql, max_lines=20)
                sql = det_sql
                result["sql_used"] = sql
            else:
                _info("No deterministic SQL returned for this question — skipping execution")
                result["skipped"] = True
                _ok("STEP 6 SKIPPED — deterministic builder returned nothing", elapsed_ms=0)
                return result

        # ── guard: still no valid SQL ─────────────────────────────────────────
        if not sql or sql.startswith("("):
            _info("No valid SQL available — skipping DB execution")
            result["skipped"] = True
            _ok("STEP 6 SKIPPED — no SQL", elapsed_ms=0)
            return result

        # ── execute ───────────────────────────────────────────────────────────
        _section("Executing SQL")
        _preview(sql, max_lines=20)
        t_exec = time.perf_counter()
        with sa_engine.connect() as conn:
            df = pd.read_sql(sa_text(sql), conn)
        exec_ms = (time.perf_counter() - t_exec) * 1000
        result["exec_ms"] = exec_ms

        if len(df) > 500:
            df = df.iloc[:500]

        if len(df) == 0:
            _info("Query returned 0 rows")
            result["row_count"] = 0
        else:
            result["rows"]      = df.to_dict(orient="records")
            result["columns"]   = list(df.columns)
            result["row_count"] = len(df)
            _ok(f"Query executed successfully — {len(df)} row(s) returned", elapsed_ms=exec_ms)
            _info("Columns", ", ".join(df.columns.tolist()))

    except Exception as exc:
        err_str = str(exc)
        if "10060" in err_str or "08S01" in err_str or "TCP Provider" in err_str:
            _err("DB not reachable from this machine (TCP timeout)", Exception(
                "The database is likely only accessible from the production server. "
                "Run rag_pipeline_trace.py on the server, or use a VPN/SSH tunnel."))
        else:
            _err("DB connection / execution error", exc)
        result["error"] = err_str

    _ok("STEP 6 COMPLETE", elapsed_ms=(time.perf_counter() - t0) * 1000)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Result formatting
# ═══════════════════════════════════════════════════════════════════════════

def step7_result_formatting(db_result: Dict) -> Dict:
    """Format raw DB rows into display-ready records with timing."""
    _banner(7, "RESULT FORMATTING")
    t0 = time.perf_counter()

    formatted: Dict = {
        "records": [],
        "row_count": 0,
        "columns": [],
        "format_ms": 0.0,
    }

    if db_result.get("skipped") or db_result.get("error"):
        reason = "DB step was skipped" if db_result.get("skipped") else f"DB error: {db_result.get('error')}"
        _info(f"Nothing to format — {reason}")
        _ok("STEP 7 SKIPPED", elapsed_ms=0)
        return formatted

    rows    = db_result.get("rows", [])
    columns = db_result.get("columns", [])

    if not rows:
        _info("0 rows returned — nothing to format")
        _ok("STEP 7 COMPLETE — empty result set", elapsed_ms=(time.perf_counter() - t0) * 1000)
        return formatted

    # ── formatting ───────────────────────────────────────────────────────────
    _section("Rounding numeric values (2 decimal places)")
    t_fmt = time.perf_counter()

    records = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if isinstance(v, float):
                clean[k] = round(v, 2)
            else:
                clean[k] = v
        records.append(clean)

    fmt_ms = (time.perf_counter() - t_fmt) * 1000
    formatted["records"]   = records
    formatted["row_count"] = len(records)
    formatted["columns"]   = columns
    formatted["format_ms"] = fmt_ms

    _ok(f"Formatted {len(records)} record(s)", elapsed_ms=fmt_ms)

    # ── print table ──────────────────────────────────────────────────────────
    _section("Result preview (first 10 rows)")
    col_widths = {c: max(len(str(c)), max((len(str(r.get(c, ""))) for r in records[:20]), default=0))
                  for c in columns}
    header = "  " + "  ".join(str(c).ljust(col_widths[c]) for c in columns)
    sep    = "  " + "  ".join("─" * col_widths[c] for c in columns)
    print(f"    {DIM}{header}{RESET}")
    print(f"    {DIM}{sep}{RESET}")
    for row in records[:10]:
        line = "  " + "  ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in columns)
        print(f"    {line}")
    if len(records) > 10:
        print(f"    {DIM}  … {len(records) - 10} more rows{RESET}")

    _ok("STEP 7 COMPLETE — results ready for frontend", elapsed_ms=(time.perf_counter() - t0) * 1000)
    return formatted


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Trace the Text-to-SQL RAG pipeline with automatic complexity routing.",
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
        default="how our business is performing",
        help="Natural-language question to trace through the pipeline.")
    args     = parser.parse_args()
    question = args.question.strip()

    print(f"\n{BOLD}{'═'*64}{RESET}")
    print(f"{BOLD}  TEXT-TO-SQL RAG PIPELINE — AUTO-ROUTED TRACE{RESET}")
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
    db_result        = step6_db_execution(final_sql, complexity, question=question)
    ms6              = db_result.get("exec_ms", 0)
    fmt_result       = step7_result_formatting(db_result)
    ms7              = fmt_result.get("format_ms", 0)

    total_ms = (time.perf_counter() - t_pipeline) * 1000

    # ── timing breakdown ─────────────────────────────────────────────────────
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
    print(f"  Rules found     : {len(rag_context.get('rules',    []))}")
    print(f"  Prompt chars    : {len(prompt_res['final_prompt_text'])}")
    print(f"  SQL lines       : {len(final_sql.splitlines())}")
    print(f"  Rows returned   : {fmt_result['row_count']}")
    print(f"{BOLD}{CYAN}{'─'*64}{RESET}")
    print(f"{BOLD}{CYAN}  STAGE TIMINGS{RESET}")
    print(f"{BOLD}{CYAN}{'─'*64}{RESET}")
    print(f"  Step 0  auto-routing      :  {ms0:.0f} ms")
    print(f"  Step 1  vector store boot :  {ms1:.0f} ms")
    print(f"  Step 2  RAG retrieval     :  {ms2:.0f} ms")
    print(f"  Step 3  prompt assembly   :  {ms3:.0f} ms")
    print(f"  Step 4  LLM call          :  {ms4:.0f} ms")
    print(f"  Step 5  SQL extraction    :  {ms5:.0f} ms")
    print(f"  Step 6  DB execution      :  {ms6:.0f} ms")
    print(f"  Step 7  result formatting :  {ms7:.1f} ms")
    print(f"{BOLD}{CYAN}{'─'*64}{RESET}")
    print(f"  {BOLD}Total wall time   :  {total_ms:.0f} ms{RESET}")
    print(f"  Expected (live)   :  {route_cfg['expected_ms']}")
    if db_result.get("error") and "10060" not in db_result["error"] and "08S01" not in db_result["error"]:
        print(f"  {RED}DB error          :  {db_result['error'][:80]}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*64}{RESET}\n")

    # ── query results ─────────────────────────────────────────────────────────
    records = fmt_result.get("records", [])
    columns = fmt_result.get("columns", [])
    if records and columns:
        print(f"{BOLD}{GREEN}{'═'*64}{RESET}")
        print(f"{BOLD}{GREEN}  QUERY RESULTS  ({len(records)} row{'s' if len(records) != 1 else ''}){RESET}")
        print(f"{BOLD}{GREEN}{'═'*64}{RESET}")
        col_widths = {
            c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in records))
            for c in columns
        }
        header = "  ".join(str(c).ljust(col_widths[c]) for c in columns)
        sep    = "  ".join("─" * col_widths[c] for c in columns)
        print(f"  {BOLD}{header}{RESET}")
        print(f"  {DIM}{sep}{RESET}")
        for row in records:
            line = "  ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in columns)
            print(f"  {line}")
        if len(records) > 500:
            print(f"  {DIM}  … (capped at 500 rows){RESET}")
        print(f"{BOLD}{GREEN}{'═'*64}{RESET}\n")
    elif db_result.get("skipped"):
        print(f"{BOLD}{BLUE}  [No DB results — deterministic or no DB config]{RESET}\n")
    elif db_result.get("error"):
        print(f"{BOLD}{RED}  [DB error — see above]{RESET}\n")
    else:
        print(f"{BOLD}{YELLOW}  [Query returned 0 rows]{RESET}\n")


if __name__ == "__main__":
    main()
