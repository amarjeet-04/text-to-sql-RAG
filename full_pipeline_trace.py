"""
full_pipeline_trace.py
──────────────────────
Runs 3 example questions through the COMPLETE end-to-end pipeline.
Every step prints its own [HH:MM:SS.mmm] timestamp.

Pipeline per question:
  RAG  Step 0  Auto-routing          → complexity tier
  RAG  Step 1  Vector store boot     → RAGEngine  (cached after Q1)
  RAG  Step 2  Query embed + FAISS   → rag_context
  RAG  Step 3  Prompt assembly       → final_prompt
  RAG  Step 4  LLM call              → raw_response
  RAG  Step 5  SQL extract + validate → final_sql
                       ↓ final_sql
  DB   Stage 1  execute_query_safe()  → df
  DB   Stage 2  _results_to_records() → records
  DB   Stage 3  display

After all 3 questions a cross-question summary table is printed.

Usage
  python full_pipeline_trace.py                      # 3 built-in examples
  python full_pipeline_trace.py -q "your question"   # 1 custom question
  python full_pipeline_trace.py --no-display         # skip result rows
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

# ── re-use all steps from rag_pipeline_trace ─────────────────────────────────
from rag_pipeline_trace import (
    step0_route,
    step1_build_vector_store,
    step2_embed_and_retrieve,
    step3_build_prompt,
    step4_call_llm,
    step5_extract_sql,
    BOLD, CYAN, GREEN, YELLOW, RED, MAGENTA, BLUE, DIM, RESET,
    _ts, _banner, _ok, _info, _err,
)

# ── DB stages ─────────────────────────────────────────────────────────────────
from app.db_utils import DatabaseConfig, create_engine_with_timeout, execute_query_safe
from backend.services.sql_engine import _results_to_records
from langchain_community.utilities import SQLDatabase

# ── Module-level RAGEngine cache (mirrors _GLOBAL_SCHEMA_CACHE in sql_engine) ─
_CACHED_RAG_ENGINE = None
_CACHED_RAG_MS     = 0.0

# ── 3 representative example questions ────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "Agent-wise profit for Q1 2024.",                          # simple_llm
    "YoY revenue growth by agent: 2023 vs 2024.",              # complex_llm
    "Revenue by agent type in 2023.",                       # deterministic
]


# ──────────────────────────────────────────────────────────────────────────────
# Timestamp container — one per step per question
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class StepTiming:
    name: str
    ts_start: str  = ""   # wall-clock string at entry
    ts_end:   str  = ""   # wall-clock string at exit
    ms:       float = 0.0

@dataclass
class QuestionRun:
    question:   str
    complexity: str = ""
    steps:      List[StepTiming] = field(default_factory=list)
    total_ms:   float = 0.0
    final_sql:  str   = ""
    row_count:  int   = 0
    db_error:   Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Display helper (Stage 3)
# ──────────────────────────────────────────────────────────────────────────────
def _display_records(records: List[Dict[str, Any]], max_display: int = 10) -> None:
    if not records:
        print("  (no rows)")
        return
    cols = list(records[0].keys())
    widths = {c: max(len(str(c)), max((len(str(r.get(c, ""))) for r in records[:max_display]), default=0))
              for c in cols}
    sep    = "  " + "-+-".join("-" * widths[c] for c in cols)
    header = "  " + " | ".join(str(c).ljust(widths[c]) for c in cols)
    print(sep); print(header); print(sep)
    for row in records[:max_display]:
        print("  " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))
    print(sep)
    if len(records) > max_display:
        print(f"  … {len(records) - max_display} more rows not shown")


# ──────────────────────────────────────────────────────────────────────────────
# Run DB stages and return timing dict
# ──────────────────────────────────────────────────────────────────────────────
def _run_db_stages(db: SQLDatabase, sql: str,
                   timeout_seconds: int, max_rows: int,
                   show_results: bool) -> Dict:
    out = dict(db_exec_ms=0.0, formatting_ms=0.0, display_ms=0.0,
               total_ms=0.0, row_count=0, error=None,
               ts_s1_start="", ts_s1_end="",
               ts_s2_start="", ts_s2_end="",
               ts_s3_start="", ts_s3_end="")

    t0 = time.perf_counter()

    # Stage 1 — execute_query_safe
    out["ts_s1_start"] = _ts()
    print(f"  {GREEN}→{RESET}  [{out['ts_s1_start']}]  Stage 1  execute_query_safe() …")
    df, error = execute_query_safe(db, sql, timeout_seconds=timeout_seconds, max_rows=max_rows)
    out["db_exec_ms"]  = round((time.perf_counter() - t0) * 1_000, 2)
    out["ts_s1_end"]   = _ts()

    if error:
        _err("Stage 1 FAILED", Exception(error))
        out["error"]    = error
        out["total_ms"] = out["db_exec_ms"]
        return out

    out["row_count"] = len(df) if df is not None else 0
    print(f"  {GREEN}✓{RESET}  [{out['ts_s1_end']}]  Stage 1  done  →  "
          f"{out['db_exec_ms']} ms  |  {out['row_count']} rows × "
          f"{len(df.columns) if df is not None else 0} cols")
    print(f"  {DIM}             ↓ df → Stage 2{RESET}")

    # Stage 2 — _results_to_records
    out["ts_s2_start"] = _ts()
    print(f"  {GREEN}→{RESET}  [{out['ts_s2_start']}]  Stage 2  _results_to_records(df) …")
    t1 = time.perf_counter()
    records: List[Dict[str, Any]] = _results_to_records(df, places=2)
    out["formatting_ms"] = round((time.perf_counter() - t1) * 1_000, 2)
    out["ts_s2_end"]     = _ts()
    print(f"  {GREEN}✓{RESET}  [{out['ts_s2_end']}]  Stage 2  done  →  "
          f"{out['formatting_ms']} ms  |  {len(records)} records serialized")
    print(f"  {DIM}             ↓ records → Stage 3{RESET}")

    # Stage 3 — display
    out["ts_s3_start"] = _ts()
    print(f"  {GREEN}→{RESET}  [{out['ts_s3_start']}]  Stage 3  display …\n")
    t2 = time.perf_counter()
    if show_results:
        _display_records(records, max_display=10)
    out["display_ms"] = round((time.perf_counter() - t2) * 1_000, 2)
    out["ts_s3_end"]  = _ts()
    print(f"\n  {GREEN}✓{RESET}  [{out['ts_s3_end']}]  Stage 3  done  →  {out['display_ms']} ms")

    out["total_ms"] = round((time.perf_counter() - t0) * 1_000, 2)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Run one question through the full pipeline and return QuestionRun
# ──────────────────────────────────────────────────────────────────────────────
def run_question(question: str, q_no: int, db: Optional[SQLDatabase],
                 timeout_seconds: int, max_rows: int,
                 show_results: bool) -> QuestionRun:

    global _CACHED_RAG_ENGINE, _CACHED_RAG_MS

    run = QuestionRun(question=question)
    t_wall = time.perf_counter()

    print(f"\n{BOLD}{'█'*64}{RESET}")
    print(f"{BOLD}  QUESTION {q_no}:  {question}{RESET}")
    print(f"  Started : {_ts()}")
    print(f"{BOLD}{'█'*64}{RESET}")

    # helper to record a step
    def _record(name: str, ts_start: str, ts_end: str, ms: float):
        run.steps.append(StepTiming(name=name, ts_start=ts_start, ts_end=ts_end, ms=ms))

    # ── Step 0: routing ───────────────────────────────────────────────────────
    ts0s = _ts()
    routing    = step0_route(question)
    ts0e       = _ts()
    complexity = routing["complexity"]
    route_cfg  = routing["route"]
    run.complexity = complexity
    _record("Step 0  auto-routing     ", ts0s, ts0e, routing["step_ms"])

    # ── Step 1: vector store (cached after first question) ────────────────────
    ts1s = _ts()
    if _CACHED_RAG_ENGINE is None:
        _CACHED_RAG_ENGINE, _CACHED_RAG_MS = step1_build_vector_store()
        ms1        = _CACHED_RAG_MS
        cache_note = "built now"
    else:
        ms1        = 0.0
        cache_note = f"cache hit  (saved {_CACHED_RAG_MS:.0f} ms)"
        _banner(1, "VECTOR STORE — CACHE HIT (skipped rebuild)", color=BLUE)
        _ok("RAGEngine reused from memory", cache_note, elapsed_ms=0)
    ts1e = _ts()
    _record(f"Step 1  vector store     ", ts1s, ts1e, ms1)
    engine_rag = _CACHED_RAG_ENGINE

    # ── Step 2: RAG retrieval ─────────────────────────────────────────────────
    ts2s = _ts()
    rag_context, ms2 = step2_embed_and_retrieve(engine_rag, question)
    ts2e = _ts()
    _record("Step 2  RAG retrieval    ", ts2s, ts2e, ms2)

    # ── Step 3: prompt assembly ───────────────────────────────────────────────
    ts3s = _ts()
    prompt_res = step3_build_prompt(question, rag_context, route_cfg)
    ts3e = _ts()
    _record("Step 3  prompt assembly  ", ts3s, ts3e, prompt_res["step_ms"])

    # ── Step 4: LLM call ──────────────────────────────────────────────────────
    ts4s = _ts()
    raw_resp, ms4 = step4_call_llm(prompt_res["prompt_vars"], route_cfg,
                                   prompt_res["template"], complexity)
    ts4e = _ts()
    _record("Step 4  LLM call         ", ts4s, ts4e, ms4)

    # ── Step 5: SQL extraction ────────────────────────────────────────────────
    ts5s = _ts()
    final_sql, ms5 = step5_extract_sql(raw_resp, complexity)
    ts5e = _ts()
    _record("Step 5  SQL extraction   ", ts5s, ts5e, ms5)
    run.final_sql = final_sql

    # ── Handoff ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{MAGENTA}{'─'*64}{RESET}")
    print(f"{BOLD}{MAGENTA}  HANDOFF: final_sql → execute_query_safe(){RESET}")
    print(f"{BOLD}{MAGENTA}{'─'*64}{RESET}")
    if final_sql and not final_sql.startswith("("):
        for line in final_sql.strip().splitlines():
            print(f"    {BOLD}{line}{RESET}")
    else:
        print(f"  {DIM}{final_sql}{RESET}")
    print(f"{BOLD}{MAGENTA}{'─'*64}{RESET}")

    # ── DB stages ─────────────────────────────────────────────────────────────
    if db and final_sql and not final_sql.startswith("("):
        _banner(6, f"DB STAGES  (Q{q_no})", color=GREEN)
        db_r = _run_db_stages(db, final_sql, timeout_seconds, max_rows, show_results)
        run.row_count = db_r["row_count"]
        run.db_error  = db_r.get("error")
        _record("Stage 1  execute_query_safe ", db_r["ts_s1_start"], db_r["ts_s1_end"], db_r["db_exec_ms"])
        _record("Stage 2  _results_to_records", db_r["ts_s2_start"], db_r["ts_s2_end"], db_r["formatting_ms"])
        _record("Stage 3  display            ", db_r["ts_s3_start"], db_r["ts_s3_end"], db_r["display_ms"])
    else:
        _info("DB skipped — no SQL or no DB connection")
        run.db_error = "skipped"

    run.total_ms = round((time.perf_counter() - t_wall) * 1_000, 2)

    # ── Per-question step summary ─────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}  ┌─ Q{q_no} STEP TIMINGS ({'─'*44}){RESET}")
    for st in run.steps:
        bar_len = min(30, int(st.ms / 100))
        bar     = "█" * bar_len
        note    = f"  {DIM}(cache hit){RESET}" if "vector store" in st.name and st.ms == 0 else ""
        print(f"{BOLD}{CYAN}  │{RESET}  [{st.ts_start}] → [{st.ts_end}]  "
              f"{st.name}  {st.ms:>7.0f} ms  {YELLOW}{bar}{RESET}{note}")
    print(f"{BOLD}{CYAN}  └─ Total: {run.total_ms:.0f} ms   "
          f"complexity={run.complexity.upper()}   rows={run.row_count}{RESET}")

    return run


# ──────────────────────────────────────────────────────────────────────────────
# Cross-question comparison table
# ──────────────────────────────────────────────────────────────────────────────
def print_comparison(runs: List[QuestionRun]) -> None:
    print(f"\n{BOLD}{'═'*72}{RESET}")
    print(f"{BOLD}  CROSS-QUESTION COMPARISON  ({len(runs)} questions){RESET}")
    print(f"{BOLD}{'═'*72}{RESET}")

    # header
    col_w = 28
    q_w   = 10
    print(f"  {'STEP':<{col_w}}", end="")
    for i, r in enumerate(runs, 1):
        label = f"Q{i} ({r.complexity[:6]})"
        print(f"  {label:>{q_w}}", end="")
    print()
    print(f"  {'─'*col_w}", end="")
    for _ in runs:
        print(f"  {'─'*q_w}", end="")
    print()

    # collect all unique step names in order
    all_steps = []
    seen = set()
    for r in runs:
        for s in r.steps:
            if s.name not in seen:
                all_steps.append(s.name)
                seen.add(s.name)

    for step_name in all_steps:
        print(f"  {step_name:<{col_w}}", end="")
        for r in runs:
            match = next((s for s in r.steps if s.name == step_name), None)
            if match is None:
                print(f"  {'—':>{q_w}}", end="")
            elif match.ms == 0 and "vector" in step_name:
                print(f"  {'0 (cached)':>{q_w}}", end="")
            else:
                print(f"  {match.ms:>{q_w}.0f}", end="")
        print()

    # totals
    print(f"  {'─'*col_w}", end="")
    for _ in runs:
        print(f"  {'─'*q_w}", end="")
    print()
    print(f"  {'TOTAL ms':<{col_w}}", end="")
    for r in runs:
        print(f"  {r.total_ms:>{q_w}.0f}", end="")
    print()
    print(f"  {'Rows returned':<{col_w}}", end="")
    for r in runs:
        val = str(r.row_count) if not r.db_error else "—"
        print(f"  {val:>{q_w}}", end="")
    print()
    print(f"{BOLD}{'═'*72}{RESET}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3 example questions through the full pipeline with timestamps"
    )
    parser.add_argument("-q", "--question", default=None,
        help="Single custom question (skips the 3 built-in examples)")
    parser.add_argument("--host",     default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--port",     default=os.getenv("DB_PORT", "1433"))
    parser.add_argument("--user",     default=os.getenv("DB_USER", "sa"))
    parser.add_argument("--password", default=os.getenv("DB_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("DB_NAME",
                                       os.getenv("DB_DATABASE", "master")))
    parser.add_argument("--query-timeout", type=int, default=30)
    parser.add_argument("--max-rows",      type=int, default=500)
    parser.add_argument("--no-display", action="store_true",
        help="Skip printing result rows (timing only)")
    args = parser.parse_args()

    questions = [args.question.strip()] if args.question else EXAMPLE_QUESTIONS

    # ── build DB connection once, reused for all questions ────────────────────
    db = None
    config = DatabaseConfig(
        host=args.host, port=args.port,
        username=args.user, password=args.password,
        database=args.database,
        connect_timeout=10, query_timeout=args.query_timeout,
        view_support=True, lazy_table_reflection=True,
    )
    try:
        db_engine = create_engine_with_timeout(config)
        db        = SQLDatabase(engine=db_engine, lazy_table_reflection=True)
        print(f"\n  {GREEN}✓{RESET}  DB connected: {args.host}:{args.port} / {args.database}")
    except Exception as exc:
        print(f"\n  {YELLOW}⚠{RESET}  DB unavailable — DB stages will be skipped: {exc}")

    print(f"\n{BOLD}{'═'*64}{RESET}")
    print(f"{BOLD}  FULL PIPELINE TRACE  —  {len(questions)} question(s){RESET}")
    print(f"{BOLD}{'═'*64}{RESET}")
    print(f"  Pipeline : RAG Steps 0–5  →  DB Stages 1–3")
    print(f"  Started  : {_ts()}")
    print(f"{BOLD}{'═'*64}{RESET}")

    # ── run each question ─────────────────────────────────────────────────────
    runs: List[QuestionRun] = []
    for i, q in enumerate(questions, 1):
        r = run_question(
            question=q, q_no=i, db=db,
            timeout_seconds=args.query_timeout,
            max_rows=args.max_rows,
            show_results=not args.no_display,
        )
        runs.append(r)

    # ── cross-question comparison ─────────────────────────────────────────────
    if len(runs) > 1:
        print_comparison(runs)

    if db:
        db_engine.dispose()


if __name__ == "__main__":
    main()
