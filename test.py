import argparse
import os
import random
import re
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from backend.services.sql_engine import (
    LLM_SQL_TIMEOUT_MS,
    SQL_TEMPLATE,
    _build_relative_date_reference,
    _clean_sql_response,
    _deterministic_timeout_fallback_sql,
    _dialect_label,
    _estimate_query_complexity,
    _read_stored_procedure_file,
    _invoke_with_timeout,
    build_domain_digest,
    fix_common_sql_errors,
    handle_query,
    initialize_connection,
    normalize_sql_dialect,
)

# Edit this question directly for quick local tests.
TEST_QUESTION = "Average booking value by country this year."

QUESTION_POOL = [
    "How does my business look in 2023?",
    "Show top 10 agents by total revenue in 2023.",
    "Show bottom 5 agents by total bookings in 2023.",
    "Revenue, cost, and profit by supplier for 2023.",
    "Top 10 suppliers by profit this year.",
    "Total bookings by country in 2023.",
    "Top 10 cities by bookings in 2023.",
    "Revenue by agent type in 2023.",
    "Average booking value by agent in 2023.",
    "Total bookings per month in 2023.",
    "Total revenue per month in 2023.",
    "Profit by hotel chain in 2023.",
    "Top 15 hotels by booking count in 2023.",
    "Supplier-wise bookings for January 2024.",
    "Agent-wise profit for Q1 2024.",
    "Country-wise revenue and bookings for 2024.",
    "Compare total revenue and total cost in 2023.",
    "Show bookings with check-in date in December 2023 by city.",
    "Which agent country generated the highest revenue in 2023?",
    "Top 10 agent regions by bookings this year.",
    "Show total bookings by supplier this month.",
    "Top 5 countries by revenue in 2024.",
    "Monthly profit trend for 2024.",
    "Top 10 agents by bookings this month.",
    "List revenue and bookings by city for 2023.",
    "Which hotel chain has the highest bookings in 2023?",
    "Show top 20 suppliers by revenue in 2024.",
    "Average booking value by country this year.",
    "Total profit by agent country in 2023.",
    "Show month-wise bookings for 2024.",
]


def _is_true(value: str, default: bool = False) -> bool:
    raw = (value or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_required(*names: str) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip().strip('"')
        if value:
            return value
    raise RuntimeError(f"Missing required env var. Expected one of: {', '.join(names)}")


def _build_few_shot_from_rag(question: str, rag_engine: Any) -> tuple[str, str]:
    if rag_engine is None:
        return "", "none"
    try:
        rag_ctx: Dict[str, Any] = rag_engine.retrieve(
            question,
            top_k=2,
            fast_mode=True,
            skip_for_followup=False,
            intent_key="sql_only",
        )
    except TypeError:
        rag_ctx = rag_engine.retrieve(question, top_k=2)
    except Exception:
        return "", "none"

    examples: List[Dict[str, Any]] = list(rag_ctx.get("examples", []) or [])[:3]
    raw_hints = list(rag_ctx.get("tables", []) or [])
    table_hints: List[str] = []
    for hint in raw_hints:
        if isinstance(hint, str):
            name = hint.strip()
        elif isinstance(hint, dict):
            name = str(hint.get("table") or hint.get("name") or "").strip()
        else:
            name = ""
        if name:
            table_hints.append(name)
    retrieved_tables_hint = ", ".join(table_hints[:6]) if table_hints else "none"

    if not examples:
        return "", retrieved_tables_hint

    parts = ["EXAMPLE QUERIES (follow these patterns closely):"]
    for ex in examples:
        q = str(ex.get("question") or "").strip()
        s = str(ex.get("sql") or "").strip()
        if q and s:
            parts.append(f"Q: {q}\nSQL: {s}")
    return "\n\n".join(parts), retrieved_tables_hint


def _build_runtime() -> Dict[str, Any]:
    host = _env_required("DB_HOST")
    port = _env_required("DB_PORT")
    db_user = _env_required("DB_USER", "DB_USERNAME")
    db_password = _env_required("DB_PASSWORD")
    db_name = _env_required("DB_NAME")

    llm_provider = (os.getenv("LLM_PROVIDER") or "openai").strip()
    model = (os.getenv("LLM_MODEL") or "gpt-4o").strip()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    query_timeout = int(os.getenv("DB_QUERY_TIMEOUT", "60"))
    view_support = _is_true(os.getenv("DB_VIEW_SUPPORT"), default=True)

    if llm_provider.lower() == "deepseek" or model.lower().startswith("deepseek"):
        api_key = _env_required("DEEPSEEK_API_KEY", "OPENAI_API_KEY")
    else:
        api_key = _env_required("OPENAI_API_KEY")

    (
        db,
        db_config,
        sql_chain,
        llm,
        reasoning_llm,
        rag,
        embedder,
        message,
        tables_count,
        views_count,
        cached_schema_text,
    ) = initialize_connection(
        host=host,
        port=port,
        db_username=db_user,
        db_password=db_password,
        database=db_name,
        llm_provider=llm_provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        query_timeout=query_timeout,
        view_support=view_support,
    )

    return {
        "db": db,
        "db_config": db_config,
        "sql_chain": sql_chain,
        "llm": llm,
        "reasoning_llm": reasoning_llm,
        "rag": rag,
        "embedder": embedder,
        "llm_provider": llm_provider,
        "llm_model": model,
        "connection_message": message,
        "tables_count": tables_count,
        "views_count": views_count,
        "rag_on_connect": _is_true(os.getenv("ENABLE_RAG_ON_CONNECT"), default=True),
        "cached_schema_text": cached_schema_text or "",
        "active_dialect": normalize_sql_dialect(os.getenv("SQL_DIALECT", "sqlserver")),
        "dialect_label": _dialect_label(normalize_sql_dialect(os.getenv("SQL_DIALECT", "sqlserver"))),
        "relative_date_reference": _build_relative_date_reference(),
        "enable_nolock": _is_true(os.getenv("ENABLE_NOLOCK"), default=False),
        "nolock_setting": str(_is_true(os.getenv("ENABLE_NOLOCK"), default=False)).lower(),
        "stored_procedure_guidance": build_domain_digest(_read_stored_procedure_file()),
        "use_reasoning": _is_true(os.getenv("SQL_ONLY_USE_REASONING"), default=False) and reasoning_llm is not None,
    }


def _dispose_runtime(runtime: Dict[str, Any]) -> None:
    db = runtime.get("db")
    if db is None:
        return
    try:
        engine = getattr(db, "_engine", None)
        if engine is not None:
            engine.dispose()
    except Exception:
        pass


def _generate_sql_with_runtime(question: str, runtime: Dict[str, Any]) -> str:
    few_shot_examples, retrieved_tables_hint = _build_few_shot_from_rag(question, runtime.get("rag"))
    complexity = _estimate_query_complexity(question)
    if os.getenv("SQL_ONLY_TIMEOUT_MS"):
        timeout_ms = int(os.getenv("SQL_ONLY_TIMEOUT_MS", str(LLM_SQL_TIMEOUT_MS)))
    elif complexity == "complex_llm":
        timeout_ms = max(24000, int(LLM_SQL_TIMEOUT_MS) * 3)
    elif complexity == "simple_llm":
        timeout_ms = max(10000, int(LLM_SQL_TIMEOUT_MS))
    else:
        timeout_ms = max(16000, int(LLM_SQL_TIMEOUT_MS) * 2)

    def _fallback_sql_on_timeout() -> Optional[str]:
        fallback_sql, _reason = _deterministic_timeout_fallback_sql(question, runtime.get("db"))
        if not fallback_sql:
            return None
        return fix_common_sql_errors(fallback_sql, dialect=runtime["active_dialect"])

    if runtime.get("use_reasoning"):
        prompt = SQL_TEMPLATE.format(
            full_schema=runtime["cached_schema_text"],
            stored_procedure_guidance=runtime["stored_procedure_guidance"],
            context="",
            question=question,
            retrieved_tables_hint=retrieved_tables_hint,
            few_shot_examples=few_shot_examples,
            dialect_label=runtime["dialect_label"],
            relative_date_reference=runtime["relative_date_reference"],
            enable_nolock=runtime["nolock_setting"],
        )
        try:
            resp = _invoke_with_timeout(lambda: runtime["reasoning_llm"].invoke(prompt), timeout_ms=timeout_ms)
        except FuturesTimeoutError:
            fallback = _fallback_sql_on_timeout()
            if fallback:
                return fallback
            raise
        raw = resp.content if hasattr(resp, "content") else str(resp)
        raw = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.DOTALL).strip()
    else:
        try:
            raw = _invoke_with_timeout(
                lambda: runtime["sql_chain"].invoke(
                    {
                        "question": question,
                        "context": "",
                        "few_shot_examples": few_shot_examples,
                        "retrieved_tables_hint": retrieved_tables_hint,
                        "dialect_label": runtime["dialect_label"],
                        "relative_date_reference": runtime["relative_date_reference"],
                        "enable_nolock": runtime["nolock_setting"],
                        "full_schema_override": runtime["cached_schema_text"],
                        "stored_procedure_guidance_override": runtime["stored_procedure_guidance"],
                    }
                ),
                timeout_ms=timeout_ms,
            )
        except FuturesTimeoutError:
            fallback = _fallback_sql_on_timeout()
            if fallback:
                return fallback
            # One retry with a larger timeout before surfacing timeout.
            raw = _invoke_with_timeout(
                lambda: runtime["sql_chain"].invoke(
                    {
                        "question": question,
                        "context": "",
                        "few_shot_examples": few_shot_examples,
                        "retrieved_tables_hint": retrieved_tables_hint,
                        "dialect_label": runtime["dialect_label"],
                        "relative_date_reference": runtime["relative_date_reference"],
                        "enable_nolock": runtime["nolock_setting"],
                        "full_schema_override": runtime["cached_schema_text"],
                        "stored_procedure_guidance_override": runtime["stored_procedure_guidance"],
                    }
                ),
                timeout_ms=max(timeout_ms, 45000),
            )

    cleaned = _clean_sql_response(str(raw).strip())
    cleaned = fix_common_sql_errors(cleaned, dialect=runtime["active_dialect"])
    if not cleaned.strip().lower().startswith(("select", "with")):
        fallback = _fallback_sql_on_timeout()
        if fallback:
            return fallback
    return cleaned


def generate_sql_only(question: str) -> str:
    runtime = _build_runtime()
    try:
        return _generate_sql_with_runtime(question, runtime)
    finally:
        _dispose_runtime(runtime)


def run_handle_query_once(question: str, runtime: Dict[str, Any]) -> Dict[str, Any]:
    return handle_query(
        question=question,
        db=runtime["db"],
        db_config=runtime["db_config"],
        sql_chain=runtime["sql_chain"],
        llm=runtime["llm"],
        rag_engine=runtime["rag"],
        embedder=runtime["embedder"],
        chat_history=[],
        query_cache=[],
        cached_schema_text=runtime["cached_schema_text"],
        session_state=None,
        conversation_turns=[],
        reasoning_llm=runtime["reasoning_llm"],
        sql_dialect=runtime["active_dialect"],
        enable_nolock=runtime["enable_nolock"],
    )


def _print_handle_query_timing(timing: Dict[str, Any]) -> None:
    if not timing:
        return
    print("[handle_query_timing]", flush=True)
    for stage, ms in timing.items():
        print(f"  {stage}: {ms} ms", flush=True)
    print(f"[end_to_end_ms] {timing.get('total', 0.0)}", flush=True)


def _print_timing(enabled: bool, stage: str, started_at: float) -> None:
    if not enabled:
        return
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    print(f"[timing] {stage}: {elapsed_ms:.1f} ms", flush=True)


def run_batch_random_20(
    output_file: Path,
    seed: Optional[int] = None,
    show_timing: bool = True,
    mode: str = "handle_query",
) -> None:
    if not QUESTION_POOL:
        raise RuntimeError("QUESTION_POOL is empty.")

    count = min(20, len(QUESTION_POOL))
    rng = random.Random(seed)
    questions = rng.sample(QUESTION_POOL, count)

    setup_start = time.perf_counter()
    runtime = _build_runtime()
    _print_timing(show_timing, "build_runtime", setup_start)
    if show_timing:
        print(
            f"[config] provider={runtime['llm_provider']} model={runtime['llm_model']} "
            f"rag_on_connect={runtime['rag_on_connect']} tables={runtime['tables_count']} views={runtime['views_count']}",
            flush=True,
        )
    try:
        chunks: List[str] = []
        query_cache: List[Dict[str, Any]] = []
        for idx, q in enumerate(questions, start=1):
            print(f"[{idx}/{count}] Generating SQL...", flush=True)
            q_start = time.perf_counter()
            try:
                if mode == "handle_query":
                    result = handle_query(
                        question=q,
                        db=runtime["db"],
                        db_config=runtime["db_config"],
                        sql_chain=runtime["sql_chain"],
                        llm=runtime["llm"],
                        rag_engine=runtime["rag"],
                        embedder=runtime["embedder"],
                        chat_history=[],
                        query_cache=query_cache,
                        cached_schema_text=runtime["cached_schema_text"],
                        session_state=None,
                        conversation_turns=[],
                        reasoning_llm=runtime["reasoning_llm"],
                        sql_dialect=runtime["active_dialect"],
                        enable_nolock=runtime["enable_nolock"],
                    )
                    sql = result.get("sql") or "-- NO SQL GENERATED"
                    meta_lines = [
                        f"INTENT: {result.get('intent')}",
                        f"ROW_COUNT: {result.get('row_count', 0)}",
                        f"FROM_CACHE: {result.get('from_cache', False)}",
                        f"ERROR: {result.get('error') or 'None'}",
                        f"TOTAL_MS: {result.get('timing', {}).get('total', 0.0)}",
                    ]
                else:
                    sql = _generate_sql_with_runtime(q, runtime)
                    meta_lines = []
            except FuturesTimeoutError:
                sql = "-- ERROR: SQL generation timed out"
                meta_lines = []
            except Exception as exc:
                sql = f"-- ERROR: {exc}"
                meta_lines = []
            _print_timing(show_timing, f"q{idx}_generate_sql", q_start)
            section = [f"### {idx}.", f"QUESTION: {q}"] + meta_lines + ["SQL:", sql]
            chunks.append("\n".join(section))
        output_file.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
        print(f"Saved batch SQL output: {output_file}")
        print(f"Questions tested: {count}")
    finally:
        try:
            _dispose_runtime(runtime)
        except Exception:
            pass


def main() -> None:
    process_start = time.perf_counter()
    load_dotenv(Path(__file__).resolve().parent / ".env")

    parser = argparse.ArgumentParser(description="Test SQL generation (sql_only) or full pipeline (handle_query).")
    parser.add_argument("--question", type=str, default="", help="Question for SQL generation.")
    parser.add_argument("--batch20", action="store_true", help="Run 20 random questions and save all generated SQL.")
    parser.add_argument("--out", type=str, default="sql_generation_results.txt", help="Output file for --batch20 mode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible --batch20 order.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["handle_query", "sql_only"],
        default="handle_query",
        help="Execution mode: full end-to-end `handle_query` or SQL-only generation.",
    )
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG during connection init for faster SQL-only tests.")
    parser.add_argument("--show-timing", action="store_true", help="Print stage timings.")
    args = parser.parse_args()
    show_timing = args.show_timing or _is_true(os.getenv("TEST_SHOW_TIMING"), default=False)

    if args.no_rag:
        os.environ["ENABLE_RAG_ON_CONNECT"] = "false"
    if show_timing:
        _print_timing(True, "startup_env_load", process_start)

    if args.batch20:
        output_path = Path(args.out)
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent / output_path
        run_batch_random_20(output_file=output_path, seed=args.seed, show_timing=show_timing, mode=args.mode)
        _print_timing(show_timing, "total", process_start)
        return

    question = (args.question or TEST_QUESTION).strip()
    if not question:
        raise RuntimeError("Set TEST_QUESTION in test.py or pass --question.")

    runtime_start = time.perf_counter()
    runtime = _build_runtime()
    _print_timing(show_timing, "build_runtime", runtime_start)
    if show_timing:
        print(
            f"[config] provider={runtime['llm_provider']} model={runtime['llm_model']} "
            f"rag_on_connect={runtime['rag_on_connect']} tables={runtime['tables_count']} views={runtime['views_count']}",
            flush=True,
        )
        if runtime.get("connection_message"):
            print(f"[connect] {runtime['connection_message']}", flush=True)

    generate_start = time.perf_counter()
    try:
        if args.mode == "handle_query":
            result = run_handle_query_once(question, runtime)
            sql = result.get("sql") or ""
            print(
                f"[handle_query] intent={result.get('intent')} row_count={result.get('row_count', 0)} "
                f"from_cache={result.get('from_cache', False)} error={result.get('error') or 'None'}",
                flush=True,
            )
            if show_timing:
                _print_handle_query_timing(result.get("timing", {}))
            if result.get("nl_answer"):
                print(f"[nl_answer] {result['nl_answer']}", flush=True)
            if sql:
                print(sql)
        else:
            sql = _generate_sql_with_runtime(question, runtime)
            print(sql)
    finally:
        _dispose_runtime(runtime)
    _print_timing(show_timing, "generate_sql", generate_start)
    _print_timing(show_timing, "total", process_start)


if __name__ == "__main__":
    main()
