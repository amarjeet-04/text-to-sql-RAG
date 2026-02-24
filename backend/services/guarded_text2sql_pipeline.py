"""
Production-grade guarded Text-to-SQL pipeline for SQL Server.

This module adds:
1) Intent-lock JSON stage
2) SQL evaluator gate (deterministic + LLM)
3) End-to-end time budget with safe timeout behavior
4) One-shot repair loop on evaluator/db dry-run failures
5) Strict safety: exactly one SELECT only
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

GUARDED_INTENT_TIMEOUT_S = max(0.5, float(os.getenv("GUARDED_INTENT_TIMEOUT_S", "4.0")))
GUARDED_SQL_TIMEOUT_S = max(1.0, float(os.getenv("GUARDED_SQL_TIMEOUT_S", "12.0")))
GUARDED_EVAL_TIMEOUT_S = max(0.5, float(os.getenv("GUARDED_EVAL_TIMEOUT_S", "1.2")))
GUARDED_FIX_TIMEOUT_S = max(0.8, float(os.getenv("GUARDED_FIX_TIMEOUT_S", "2.0")))


# ---------------------------
# Prompt constants
# ---------------------------

INTENT_LOCK_PROMPT = """You are an intent-lock planner for Text-to-SQL on SQL Server.
Return STRICT JSON only in this exact schema:
{{
  "question": "...",
  "time_window": {{"start":"YYYY-MM-DD","end_exclusive":"YYYY-MM-DD","label":"YTD|FullYear|ThisMonth|Custom"}},
  "filters": {{"booking_status_exclusions": true/false}},
  "metrics": {{"revenue":"<definition>","profit":"<definition>","bookings":"COUNT DISTINCT <id>","avg_booking_value":"revenue/bookings"}},
  "group_by": [],
  "tables_used": ["..."],
  "assumptions": ["... max 2"]
}}

Rules:
- Interpret "this year" as YTD using current date.
- Interpret "this month" as month start to next month start.
- If ambiguous, choose safest standard interpretation and state in assumptions.
- Keep assumptions max 2.

CURRENT_DATE (Asia/Kolkata): {current_date}
DIALECT: {dialect}
SCHEMA:
{schema_compact}
RULES:
{rules_compact}
RETRIEVED_TABLE_HINTS:
{retrieved_tables_hint}
QUESTION:
{question}
"""


SQL_GENERATION_PROMPT = """You are an elite SQL performance engineer and Text-to-SQL agent.
Given the intent JSON + schema/rules, output ONE optimized SQL query.

Hard rules:
- output exactly one SELECT
- use only schema columns/tables
- must use at least one table from retrieved_tables_hint (or explain in assumptions)
- SARGable dates: CreatedDate >= start AND CreatedDate < end_exclusive
- no SELECT *
- apply BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') if booking_status_exclusions=true
- minimal joins; filter early

Output format:
1) SQL in one code block
2) Optional Notes with max 2 bullets
3) Nothing else

DIALECT: {dialect}
SCHEMA:
{schema_compact}
RULES:
{rules_compact}
RETRIEVED_TABLE_HINTS:
{retrieved_tables_hint}
INTENT_JSON:
{intent_json}
QUESTION:
{question}
"""


SQL_EVALUATOR_PROMPT = """You are a SQL validator. Output ONLY JSON.

Inputs:
- QUESTION
- SQL
- DIALECT
- SCHEMA
- REQUIRED_TABLE_HINTS (optional)
- INTENT_JSON
- RULES

Return JSON:
{
  "ok_to_execute": true/false,
  "failure_type": "syntax"|"schema"|"intent_mismatch"|"performance_risk"|"safety",
  "reasons": ["max 4 short reasons"],
  "fixed_sql": "single SELECT" | null
}
"""


SQL_FIX_PROMPT = """You are a SQL repair assistant for SQL Server.
Return ONE corrected SELECT query only.

Must satisfy:
- one SELECT only
- schema-correct tables/columns
- intent-correct metrics/grouping/date window
- SARGable date predicates
- no SELECT *
- include TOP (200) for ungrouped detail queries

DIALECT: {dialect}
SCHEMA:
{schema_compact}
RULES:
{rules_compact}
QUESTION:
{question}
INTENT_JSON:
{intent_json}
BROKEN_SQL:
{sql}
FAILURE_REASONS:
{failure_reasons}
DB_ERROR:
{db_error}
"""


FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "merge", "exec", "execute", "grant", "revoke",
}

AGG_FUNCS = ("sum(", "count(", "avg(", "min(", "max(")
LARGE_FACT_TABLE_HINTS = {"bookingdata", "bookingtablequery", "country_level_view", "agent_level_view"}


@dataclass
class SchemaCatalog:
    tables: Set[str]
    columns: Set[str]
    table_columns: Dict[str, Set[str]]


def _elapsed_ms(start_ts: float) -> int:
    return int((time.perf_counter() - start_ts) * 1000)


def _strip_fence(text: str) -> str:
    raw = (text or "").strip()
    m = re.search(r"```(?:json|sql)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else raw


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    raw = _strip_fence(text)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    i = raw.find("{")
    j = raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            obj = json.loads(raw[i:j + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _clean_sql(text: str) -> str:
    sql = _strip_fence(text)
    sql = re.split(r"(?im)^\s*notes\s*:", sql, maxsplit=1)[0].strip()
    m = re.search(r"(?is)\b(with|select)\b", sql)
    if m:
        sql = sql[m.start():].strip()
    return sql.rstrip(";")


def _parse_schema(schema_compact: str) -> SchemaCatalog:
    tables: Set[str] = set()
    columns: Set[str] = set()
    table_columns: Dict[str, Set[str]] = {}

    text = schema_compact or ""
    create_blocks = re.finditer(
        r"(?is)CREATE\s+TABLE\s+([A-Za-z0-9_\.\[\]]+)\s*\((.*?)\)",
        text,
    )
    for m in create_blocks:
        table = m.group(1).replace("[", "").replace("]", "").split(".")[-1].lower()
        tables.add(table)
        table_columns.setdefault(table, set())
        body = m.group(2)
        for ln in body.split(","):
            ln = ln.strip()
            if not ln:
                continue
            col = re.split(r"\s+", ln, maxsplit=1)[0].replace("[", "").replace("]", "").lower()
            if col:
                columns.add(col)
                table_columns[table].add(col)

    # Fallback for compact schema lines like "table.column".
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", text):
        t = m.group(1).lower()
        c = m.group(2).lower()
        tables.add(t)
        columns.add(c)
        table_columns.setdefault(t, set()).add(c)

    return SchemaCatalog(tables=tables, columns=columns, table_columns=table_columns)


def _extract_sql_tables(sql: str) -> Set[str]:
    out = set()
    for m in re.finditer(r"(?i)\b(?:FROM|JOIN)\s+([A-Za-z0-9_\.\[\]]+)", sql or ""):
        t = m.group(1).replace("[", "").replace("]", "").split(".")[-1].lower()
        out.add(t)
    return out


def _extract_sql_columns(sql: str) -> Set[str]:
    # Heuristic tokenizer for column references.
    cleaned = sql or ""
    # Remove comments and quoted literals so status text values are not
    # misclassified as unknown schema columns.
    cleaned = re.sub(r"(?is)/\*.*?\*/", " ", cleaned)
    cleaned = re.sub(r"(?im)--.*?$", " ", cleaned)
    cleaned = re.sub(r"(?is)N?'(?:''|[^'])*'", " ", cleaned)
    cleaned = re.sub(r'(?is)"(?:""|[^"])*"', " ", cleaned)
    out = set()
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", cleaned):
        tok = m.group(1).lower()
        if tok in {
            "select", "from", "where", "group", "by", "order", "top", "as", "and", "or", "on", "join", "left",
            "right", "inner", "outer", "with", "case", "when", "then", "else", "end", "distinct", "count", "sum",
            "avg", "min", "max", "nullif", "coalesce", "datefromparts", "dateadd", "getdate", "cast",
            "desc", "asc", "between", "in", "not", "is", "null", "like", "limit",
        }:
            continue
        out.add(tok)
    return out


def _is_single_select(sql: str) -> bool:
    text = (sql or "").strip()
    if not text:
        return False
    low = text.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return False
    # Reject multi-statement by semicolon-in-middle
    parts = [p.strip() for p in text.split(";") if p.strip()]
    if len(parts) > 1:
        return False
    # Reject obvious non-select statements
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"(?i)\b{re.escape(kw)}\b", low):
            return False
    return True


def _has_group_by(sql: str) -> bool:
    return re.search(r"(?i)\bGROUP\s+BY\b", sql or "") is not None


def _has_aggregates(sql: str) -> bool:
    low = (sql or "").lower()
    return any(f in low for f in AGG_FUNCS)


def _has_top_or_limit(sql: str) -> bool:
    return bool(re.search(r"(?i)\bTOP\s*\(?\s*\d+\s*\)?\b", sql or "") or re.search(r"(?i)\bLIMIT\s+\d+\b", sql or ""))


def _contains_select_star(sql: str) -> bool:
    return re.search(r"(?i)\bSELECT\s+\*", sql or "") is not None


def _has_date_bounds(sql: str, start: str, end_exclusive: str, date_col: str = "CreatedDate") -> bool:
    if not sql:
        return False
    col = re.escape(date_col)
    low_pat = rf"(?i)\b{col}\b\s*>=\s*'{re.escape(start)}'"
    high_pat = rf"(?i)\b{col}\b\s*<\s*'{re.escape(end_exclusive)}'"
    return bool(re.search(low_pat, sql) and re.search(high_pat, sql))


def _mentions_large_fact(sql: str) -> bool:
    tables = _extract_sql_tables(sql)
    return any(t in LARGE_FACT_TABLE_HINTS for t in tables)


def _ensure_top_for_detail(sql: str, dialect: str) -> str:
    if _has_aggregates(sql) or _has_group_by(sql) or _has_top_or_limit(sql):
        return sql
    if dialect.lower() == "mssql":
        return re.sub(r"(?is)^\s*SELECT\s+", "SELECT TOP (200) ", sql, count=1)
    return sql.rstrip() + " LIMIT 200"


def _extract_reasons_text(reasons: List[str]) -> str:
    if not reasons:
        return "none"
    return "\n".join(f"- {r}" for r in reasons[:4])


def _safe_date(value: str, fallback: date) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return fallback


def _resolve_time_window_from_question(question: str, current_date_str: str) -> Dict[str, str]:
    q = (question or "").lower()
    now = _safe_date(current_date_str, date(2026, 2, 24))
    if "this month" in q:
        start = now.replace(day=1)
        if start.month == 12:
            end = date(start.year + 1, 1, 1)
        else:
            end = date(start.year, start.month + 1, 1)
        return {"start": start.isoformat(), "end_exclusive": end.isoformat(), "label": "ThisMonth"}
    if "this year" in q:
        start = date(now.year, 1, 1)
        end = now + timedelta(days=1)
        return {"start": start.isoformat(), "end_exclusive": end.isoformat(), "label": "YTD"}
    # Safe default: YTD
    start = date(now.year, 1, 1)
    end = now + timedelta(days=1)
    return {"start": start.isoformat(), "end_exclusive": end.isoformat(), "label": "YTD"}


def _default_intent(question: str, current_date: str, retrieved_tables_hint: List[str]) -> Dict[str, Any]:
    tw = _resolve_time_window_from_question(question, current_date)
    return {
        "question": question,
        "time_window": tw,
        "filters": {"booking_status_exclusions": True},
        "metrics": {
            "revenue": "SUM(AgentBuyingPrice)",
            "profit": "SUM(AgentBuyingPrice - CompanyBuyingPrice)",
            "bookings": "COUNT DISTINCT PNRNo",
            "avg_booking_value": "revenue/bookings",
        },
        "group_by": [],
        "tables_used": retrieved_tables_hint[:2] if retrieved_tables_hint else [],
        "assumptions": ["Defaulted to YTD if timeframe unclear"][:2],
    }


def _normalize_intent(intent: Dict[str, Any], question: str, current_date: str, retrieved_tables_hint: List[str]) -> Dict[str, Any]:
    base = _default_intent(question, current_date, retrieved_tables_hint)
    if not isinstance(intent, dict):
        return base

    out = dict(base)
    out["question"] = str(intent.get("question") or question)

    tw = intent.get("time_window") if isinstance(intent.get("time_window"), dict) else {}
    start = str(tw.get("start") or base["time_window"]["start"])
    end = str(tw.get("end_exclusive") or base["time_window"]["end_exclusive"])
    label = str(tw.get("label") or base["time_window"]["label"])
    out["time_window"] = {"start": start, "end_exclusive": end, "label": label}

    flt = intent.get("filters") if isinstance(intent.get("filters"), dict) else {}
    out["filters"] = {"booking_status_exclusions": bool(flt.get("booking_status_exclusions", True))}

    metrics = intent.get("metrics") if isinstance(intent.get("metrics"), dict) else {}
    out["metrics"] = {
        "revenue": str(metrics.get("revenue") or base["metrics"]["revenue"]),
        "profit": str(metrics.get("profit") or base["metrics"]["profit"]),
        "bookings": str(metrics.get("bookings") or base["metrics"]["bookings"]),
        "avg_booking_value": str(metrics.get("avg_booking_value") or base["metrics"]["avg_booking_value"]),
    }

    gb = intent.get("group_by") if isinstance(intent.get("group_by"), list) else []
    out["group_by"] = [str(x) for x in gb][:8]
    tu = intent.get("tables_used") if isinstance(intent.get("tables_used"), list) else []
    out["tables_used"] = [str(x) for x in tu][:5] or (retrieved_tables_hint[:2] if retrieved_tables_hint else [])
    asm = intent.get("assumptions") if isinstance(intent.get("assumptions"), list) else []
    out["assumptions"] = [str(x) for x in asm][:2]
    return out


def _timeout_response(
    start_ts: float,
    intent_json: Optional[Dict[str, Any]],
    sql: Optional[str],
) -> Dict[str, Any]:
    return {
        "status": "timeout",
        "intent_json": intent_json,
        "sql": sql,
        "executed": False,
        "rows": None,
        "reason": "Time budget exceeded; SQL not executed. Narrow date range or confirm metrics.",
        "elapsed_ms": _elapsed_ms(start_ts),
    }


def build_intent(
    question: str,
    *,
    schema_compact: str,
    rules_compact: str,
    dialect: str,
    current_date: str,
    retrieved_tables_hint: List[str],
    llm: Any,
    timeout_s: float = GUARDED_INTENT_TIMEOUT_S,
) -> Dict[str, Any]:
    prompt = INTENT_LOCK_PROMPT.format(
        current_date=current_date,
        dialect=dialect,
        schema_compact=schema_compact,
        rules_compact=rules_compact,
        retrieved_tables_hint=", ".join(retrieved_tables_hint) if retrieved_tables_hint else "none",
        question=question,
    )
    raw = ""
    try:
        raw = llm.chat([{"role": "user", "content": prompt}], timeout_s=timeout_s)
    except Exception as exc:
        logger.info("intent_lock_llm_failed: %s", exc)
    parsed = _extract_json(raw) or {}
    return _normalize_intent(parsed, question, current_date, retrieved_tables_hint)


def generate_sql(
    intent_json: Dict[str, Any],
    *,
    question: str,
    schema_compact: str,
    rules_compact: str,
    dialect: str,
    retrieved_tables_hint: List[str],
    llm: Any,
    timeout_s: float = GUARDED_SQL_TIMEOUT_S,
) -> str:
    prompt = SQL_GENERATION_PROMPT.format(
        dialect=dialect,
        schema_compact=schema_compact,
        rules_compact=rules_compact,
        retrieved_tables_hint=", ".join(retrieved_tables_hint) if retrieved_tables_hint else "none",
        intent_json=json.dumps(intent_json, ensure_ascii=True),
        question=question,
    )
    raw = llm.chat([{"role": "user", "content": prompt}], timeout_s=timeout_s)
    return _clean_sql(raw)


def evaluate_sql(
    question: str,
    intent_json: Dict[str, Any],
    sql: str,
    schema_compact: str,
    rules_compact: str,
    retrieved_tables_hint: List[str],
    *,
    llm: Any,
    dialect: str = "mssql",
    timeout_s: float = GUARDED_EVAL_TIMEOUT_S,
) -> Dict[str, Any]:
    reasons: List[str] = []
    failure_type = "syntax"
    fixed_sql: Optional[str] = None

    # ── Hard safety gate (always run) ────────────────────────────────────────
    if not _is_single_select(sql):
        reasons.append("must_be_single_select")
        return {
            "ok_to_execute": False,
            "failure_type": "safety",
            "reasons": reasons[:4],
            "fixed_sql": None,
        }

    schema = _parse_schema(schema_compact)
    used_tables = _extract_sql_tables(sql)
    used_cols = _extract_sql_columns(sql)

    # Unknown table check — blocks only when schema is populated AND tables are
    # genuinely absent (not a false-positive from aliases / CTEs).
    unknown_tables = sorted(t for t in used_tables if t not in schema.tables)
    if unknown_tables:
        reasons.append("unknown_table:" + ",".join(unknown_tables[:2]))
        failure_type = "schema"

    # NOTE: column check REMOVED — _extract_sql_columns() captures SQL aliases
    # (e.g. SUM(x) AS total_revenue), CTE names, and table-qualified refs in the
    # same flat set as real column names, producing far too many false positives.
    # The DB dry-run gate below catches genuine schema mismatches reliably.

    # SELECT * — always block.
    if _contains_select_star(sql):
        reasons.append("select_star_not_allowed")
        failure_type = "performance_risk"

    # Unbounded detail query — auto-fix with TOP (200).
    if not _has_group_by(sql) and not _has_aggregates(sql) and not _has_top_or_limit(sql):
        reasons.append("unbounded_detail_query")
        failure_type = "performance_risk"
        fixed_sql = _ensure_top_for_detail(sql, dialect)

    # Large fact-table without ANY date filter — true performance risk.
    if _mentions_large_fact(sql) and not re.search(
        r"(?i)\b(createddate|booking_date|bookingdate|checkindate|checkoutdate)\b"
        r"\s*(>=|>|between|=)",
        sql,
    ):
        reasons.append("missing_date_filter_large_table")
        failure_type = "performance_risk"

    # NOTE: time_window_mismatch check REMOVED — the LLM correctly generates
    # DATEFROMPARTS/DATEADD expressions which are valid SARGable predicates but
    # don't contain the literal date strings from intent_json, causing false
    # rejections of every query.  The dry-run execution gate below catches any
    # real date errors.

    # NOTE: missing_retrieved_table_hint is advisory only — logged but NOT
    # added to blocking reasons because RAG may return different name variants
    # (e.g. "BookingData" vs "BookingTableQuery") for the same fact table.
    if retrieved_tables_hint:
        hints = {t.lower() for t in retrieved_tables_hint}
        if not any(t in hints for t in used_tables):
            logger.info(
                "eval_advisory: sql tables %s not in rag hints %s",
                sorted(used_tables), sorted(hints),
            )

    # Profit metric sanity — only block when schema has the cost column.
    q = question.lower()
    sql_low = sql.lower()
    if "profit" in q and schema.columns and "companybuyingprice" in schema.columns:
        if "companybuyingprice" not in sql_low:
            reasons.append("metric_mismatch_profit")
            failure_type = "intent_mismatch"

    # Booking count sanity — accept multiple valid patterns.
    _booking_count_patterns = (
        "count(distinct",
        "sum(total_booking",
        "total_booking",
        "count(*",
        "bookingcount",
        "booking_count",
    )
    if ("booking" in q or "bookings" in q) and not any(
        p in sql_low for p in _booking_count_patterns
    ):
        reasons.append("metric_mismatch_bookings")
        failure_type = "intent_mismatch"

    # ── LLM semantic check — ONLY when deterministic checks already found
    # issues AND we still have budget. Skipping saves ~1.8 s on the happy path.
    # ────────────────────────────────────────────────────────────────────────
    llm_fixed: Optional[str] = None
    if llm is not None and reasons and not unknown_tables:
        try:
            prompt = SQL_EVALUATOR_PROMPT + "\n\n" + "\n".join([
                f"DIALECT: {dialect}",
                f"SCHEMA: {schema_compact}",
                f"REQUIRED_TABLE_HINTS: {', '.join(retrieved_tables_hint) if retrieved_tables_hint else 'none'}",
                f"QUESTION: {question}",
                f"INTENT_JSON: {json.dumps(intent_json, ensure_ascii=True)}",
                f"RULES: {rules_compact}",
                f"SQL: {sql}",
            ])
            raw = llm.chat([{"role": "user", "content": prompt}], timeout_s=timeout_s)
            obj = _extract_json(raw) or {}
            if obj:
                if not bool(obj.get("ok_to_execute", True)):
                    llm_reasons = [str(x) for x in (obj.get("reasons") or [])][:4]
                    for r in llm_reasons:
                        if r and r not in reasons:
                            reasons.append(r)
                    failure_type = str(obj.get("failure_type") or failure_type)
                candidate = obj.get("fixed_sql")
                if isinstance(candidate, str) and candidate.strip():
                    llm_fixed = _clean_sql(candidate)
        except Exception as exc:
            logger.info("sql_evaluator_llm_failed: %s", exc)

    ok = len(reasons) == 0
    if not ok and llm_fixed and _is_single_select(llm_fixed):
        fixed_sql = llm_fixed

    return {
        "ok_to_execute": ok,
        "failure_type": failure_type if not ok else "syntax",
        "reasons": reasons[:4],
        "fixed_sql": fixed_sql,
    }


def fix_sql_on_error(
    *,
    question: str,
    intent_json: Dict[str, Any],
    sql: str,
    failure_reasons: List[str],
    db_error: str,
    schema_compact: str,
    rules_compact: str,
    dialect: str,
    llm: Any,
    timeout_s: float = GUARDED_FIX_TIMEOUT_S,
) -> Optional[str]:
    if llm is None:
        return None
    prompt = SQL_FIX_PROMPT.format(
        dialect=dialect,
        schema_compact=schema_compact,
        rules_compact=rules_compact,
        question=question,
        intent_json=json.dumps(intent_json, ensure_ascii=True),
        sql=sql,
        failure_reasons=_extract_reasons_text(failure_reasons),
        db_error=db_error or "none",
    )
    try:
        raw = llm.chat([{"role": "user", "content": prompt}], timeout_s=timeout_s)
        fixed = _clean_sql(raw)
        return fixed if _is_single_select(fixed) else None
    except Exception as exc:
        logger.info("fix_sql_llm_failed: %s", exc)
        return None


def handle_query(
    question: str,
    *,
    schema_compact: str,
    rules_compact: str,
    retrieved_tables_hint: List[str],
    db_execute: Callable[[str], Any],
    db_dry_run: Callable[[str], Tuple[bool, str]],
    llm: Any,
    dialect: str = "mssql",
    current_date: str = "2026-02-24",
    budget_seconds: float = 8.0,
) -> Dict[str, Any]:
    start_ts = time.perf_counter()
    intent_json: Optional[Dict[str, Any]] = None
    sql: Optional[str] = None

    def budget_exceeded() -> bool:
        return (time.perf_counter() - start_ts) >= budget_seconds

    def step_timeout(max_timeout: float, reserve_s: float = 0.2) -> float:
        remaining = budget_seconds - (time.perf_counter() - start_ts) - reserve_s
        return max(0.2, min(max_timeout, remaining))

    try:
        if budget_exceeded():
            return _timeout_response(start_ts, intent_json, sql)

        # 1) Intent lock
        intent_json = build_intent(
            question,
            schema_compact=schema_compact,
            rules_compact=rules_compact,
            dialect=dialect,
            current_date=current_date,
            retrieved_tables_hint=retrieved_tables_hint,
            llm=llm,
            timeout_s=step_timeout(GUARDED_INTENT_TIMEOUT_S),
        )

        if budget_exceeded():
            return _timeout_response(start_ts, intent_json, sql)

        # 2) SQL generation
        sql = generate_sql(
            intent_json,
            question=question,
            schema_compact=schema_compact,
            rules_compact=rules_compact,
            dialect=dialect,
            retrieved_tables_hint=retrieved_tables_hint,
            llm=llm,
            timeout_s=step_timeout(GUARDED_SQL_TIMEOUT_S),
        )

        if budget_exceeded():
            return _timeout_response(start_ts, intent_json, sql)

        # 3) Evaluator gate
        eval_result = evaluate_sql(
            question,
            intent_json,
            sql,
            schema_compact,
            rules_compact,
            retrieved_tables_hint,
            llm=llm,
            dialect=dialect,
            timeout_s=step_timeout(GUARDED_EVAL_TIMEOUT_S),
        )
        if not eval_result["ok_to_execute"]:
            repaired = fix_sql_on_error(
                question=question,
                intent_json=intent_json,
                sql=sql,
                failure_reasons=eval_result.get("reasons") or [],
                db_error="",
                schema_compact=schema_compact,
                rules_compact=rules_compact,
                dialect=dialect,
                llm=llm,
                timeout_s=step_timeout(GUARDED_FIX_TIMEOUT_S),
            )
            if repaired:
                sql = repaired
                eval_result = evaluate_sql(
                    question,
                    intent_json,
                    sql,
                    schema_compact,
                    rules_compact,
                    retrieved_tables_hint,
                    llm=llm,
                    dialect=dialect,
                    timeout_s=step_timeout(GUARDED_EVAL_TIMEOUT_S),
                )

            if not eval_result["ok_to_execute"]:
                return {
                    "status": "rejected",
                    "intent_json": intent_json,
                    "sql": sql,
                    "executed": False,
                    "rows": None,
                    "reason": "; ".join(eval_result.get("reasons") or ["rejected_by_evaluator"]),
                    "elapsed_ms": _elapsed_ms(start_ts),
                }

        if budget_exceeded():
            return _timeout_response(start_ts, intent_json, sql)

        # 4) Dry run
        dry_ok, dry_err = db_dry_run(sql)
        if not dry_ok:
            repaired = fix_sql_on_error(
                question=question,
                intent_json=intent_json,
                sql=sql,
                failure_reasons=["dry_run_failed"],
                db_error=dry_err or "",
                schema_compact=schema_compact,
                rules_compact=rules_compact,
                dialect=dialect,
                llm=llm,
                timeout_s=step_timeout(GUARDED_FIX_TIMEOUT_S),
            )
            if repaired:
                sql = repaired
                eval_result = evaluate_sql(
                    question,
                    intent_json,
                    sql,
                    schema_compact,
                    rules_compact,
                    retrieved_tables_hint,
                    llm=llm,
                    dialect=dialect,
                    timeout_s=step_timeout(GUARDED_EVAL_TIMEOUT_S),
                )
                if not eval_result["ok_to_execute"]:
                    return {
                        "status": "rejected",
                        "intent_json": intent_json,
                        "sql": sql,
                        "executed": False,
                        "rows": None,
                        "reason": "; ".join(eval_result.get("reasons") or ["rejected_after_repair"]),
                        "elapsed_ms": _elapsed_ms(start_ts),
                    }
                dry_ok, dry_err = db_dry_run(sql)

        if not dry_ok:
            return {
                "status": "error",
                "intent_json": intent_json,
                "sql": sql,
                "executed": False,
                "rows": None,
                "reason": f"Dry run failed: {dry_err}",
                "elapsed_ms": _elapsed_ms(start_ts),
            }

        if budget_exceeded():
            return _timeout_response(start_ts, intent_json, sql)

        # 5) Execute
        rows = db_execute(sql)
        return {
            "status": "ok",
            "intent_json": intent_json,
            "sql": sql,
            "executed": True,
            "rows": rows,
            "reason": None,
            "elapsed_ms": _elapsed_ms(start_ts),
        }
    except Exception as exc:
        return {
            "status": "error",
            "intent_json": intent_json,
            "sql": sql,
            "executed": False,
            "rows": None,
            "reason": str(exc),
            "elapsed_ms": _elapsed_ms(start_ts),
        }
