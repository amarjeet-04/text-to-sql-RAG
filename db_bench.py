"""
db_bench.py
───────────
Runs the exact 3-stage pipeline the real app uses, once per --runs:

  Stage 1 — execute_query_safe()        app/db_utils.py:240
             returns df (DataFrame)
                ↓ df passed to ↓
  Stage 2 — _results_to_records(df)     backend/services/sql_engine.py:2261
             returns records (list[dict])
                ↓ records passed to ↓
  Stage 3 — display table               terminal

Each stage prints its own [HH:MM:SS.mmm] timestamp.
After all runs a summary table shows per-stage timings.

Usage
  python db_bench.py --host myserver --port 1433 --user sa --password secret --database mydb
  python db_bench.py --runs 3
  python db_bench.py --sql "SELECT TOP 100 * FROM dbo.Bookings"
  python db_bench.py --sql-file my_query.sql --no-display
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

# ── Stage 1: DB execution ─────────────────────────────────────────────────────
from app.db_utils import DatabaseConfig, create_engine_with_timeout, execute_query_safe
from langchain_community.utilities import SQLDatabase

# ── Stage 2: result formatting (imported directly from sql_engine) ────────────
from backend.services.sql_engine import _results_to_records

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("db_bench")


# ──────────────────────────────────────────────────────────────────────────────
# Default SQL
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SQL = """
WITH RevenueByCountry AS (
    SELECT 
        c.Country,
        SUM(b.AgentBuyingPrice) AS TotalRevenue,
        YEAR(b.CreatedDate) AS RevenueYear
    FROM BookingData b WITH (NOLOCK)
    JOIN suppliermaster_Report s ON b.SupplierId = s.EmployeeId
    JOIN Master_Country c ON s.EmployeeId = c.CountryID
    WHERE b.CreatedDate >= '2025-01-01' AND b.CreatedDate < '2027-01-01'
    AND b.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')
    GROUP BY c.Country, YEAR(b.CreatedDate)
)
SELECT 
    Country,
    SUM(CASE WHEN RevenueYear = 2025 THEN TotalRevenue ELSE 0 END) AS Revenue2025,
    SUM(CASE WHEN RevenueYear = 2026 THEN TotalRevenue ELSE 0 END) AS Revenue2026
FROM RevenueByCountry
GROUP BY Country;
"""


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    run_no: int
    success: bool
    # per-stage ms
    db_exec_ms: float = 0.0
    formatting_ms: float = 0.0
    display_ms: float = 0.0
    total_ms: float = 0.0
    row_count: int = 0
    error: Optional[str] = None
    # timestamps (wall-clock strings)
    ts_start: str = ""
    ts_after_exec: str = ""
    ts_after_format: str = ""
    ts_after_display: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Display helper — pretty-print records as a table (Stage 3)
# ──────────────────────────────────────────────────────────────────────────────
def _display_records(records: List[Dict[str, Any]], max_display: int = 20) -> None:
    if not records:
        print("  (no rows)")
        return
    cols = list(records[0].keys())
    # column widths
    widths = {c: max(len(str(c)), max((len(str(r.get(c, ""))) for r in records[:max_display]), default=0))
              for c in cols}
    sep = "  " + "-+-".join("-" * widths[c] for c in cols)
    header = "  " + " | ".join(str(c).ljust(widths[c]) for c in cols)
    print(sep)
    print(header)
    print(sep)
    for row in records[:max_display]:
        print("  " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))
    print(sep)
    if len(records) > max_display:
        print(f"  … {len(records) - max_display} more rows not shown")


# ──────────────────────────────────────────────────────────────────────────────
# Single run: df → records → display, each stage timestamped
# ──────────────────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def run_once(db: SQLDatabase, sql: str, run_no: int,
             timeout_seconds: int, max_rows: int,
             show_results: bool = True) -> RunResult:

    r = RunResult(run_no=run_no, success=False)
    print(f"\n  ┌── Run {run_no} {'─' * 48}")

    # ── Stage 1: execute_query_safe() → df ───────────────────────────────────
    r.ts_start = _ts()
    t0 = time.perf_counter()
    print(f"  │  [{r.ts_start}]  Stage 1  execute_query_safe() …")

    df, error = execute_query_safe(db, sql,
                                   timeout_seconds=timeout_seconds,
                                   max_rows=max_rows)

    r.db_exec_ms = round((time.perf_counter() - t0) * 1_000, 2)
    r.ts_after_exec = _ts()

    if error:
        print(f"  │  [{r.ts_after_exec}]  Stage 1  FAILED in {r.db_exec_ms} ms")
        print(f"  │             error: {error}")
        print(f"  └{'─' * 54}")
        r.error = error
        return r

    r.row_count = len(df) if df is not None else 0
    print(f"  │  [{r.ts_after_exec}]  Stage 1  done  →  {r.db_exec_ms} ms  "
          f"| df: {r.row_count} rows × {len(df.columns) if df is not None else 0} cols")
    print(f"  │             ↓ df passed to Stage 2")

    # ── Stage 2: _results_to_records(df) → records ───────────────────────────
    t1 = time.perf_counter()
    print(f"  │  [{r.ts_after_exec}]  Stage 2  _results_to_records(df) …")

    records: List[Dict[str, Any]] = _results_to_records(df, places=2)

    r.formatting_ms = round((time.perf_counter() - t1) * 1_000, 2)
    r.ts_after_format = _ts()
    print(f"  │  [{r.ts_after_format}]  Stage 2  done  →  {r.formatting_ms} ms  "
          f"| {len(records)} records serialized (Decimal→float, rounded)")
    print(f"  │             ↓ records passed to Stage 3")

    # ── Stage 3: display(records) ─────────────────────────────────────────────
    t2 = time.perf_counter()
    print(f"  │  [{r.ts_after_format}]  Stage 3  display …")

    if show_results:
        print(f"  │")
        _display_records(records, max_display=10)
        print(f"  │")

    r.display_ms = round((time.perf_counter() - t2) * 1_000, 2)
    r.ts_after_display = _ts()
    r.total_ms = round((time.perf_counter() - t0) * 1_000, 2)
    print(f"  │  [{r.ts_after_display}]  Stage 3  done  →  {r.display_ms} ms")
    print(f"  └── total: {r.total_ms} ms  "
          f"(exec={r.db_exec_ms}  format={r.formatting_ms}  display={r.display_ms})")

    r.success = True
    return r


# ──────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────────────
def _pct(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


def _fmt_stats(vals: List[float]) -> str:
    return (f"min={min(vals):.1f}  p50={_pct(vals,50):.1f}  "
            f"p95={_pct(vals,95):.1f}  max={max(vals):.1f}  "
            f"mean={statistics.mean(vals):.1f}  (ms)")


def print_summary(results: List[RunResult]) -> None:
    ok   = [r for r in results if r.success]
    fail = [r for r in results if not r.success]

    print("\n" + "═" * 72)
    print(f"  BENCHMARK SUMMARY   runs={len(results)}  ok={len(ok)}  failed={len(fail)}")
    print("═" * 72)

    if not ok:
        print("  No successful runs — check credentials / SQL / server.")
        for r in fail:
            print(f"  run={r.run_no}  error={r.error}")
        print()
        return

    print(f"  {'1. execute_query_safe()':<30}  {_fmt_stats([r.db_exec_ms    for r in ok])}")
    print(f"  {'2. _results_to_records()':<30}  {_fmt_stats([r.formatting_ms for r in ok])}")
    print(f"  {'3. display':<30}  {_fmt_stats([r.display_ms    for r in ok])}")
    print(f"  {'TOTAL':<30}  {_fmt_stats([r.total_ms      for r in ok])}")
    print(f"  Rows/run : {ok[0].row_count}")

    if fail:
        print("\n  Failed runs:")
        for r in fail:
            print(f"    run={r.run_no}  error={r.error}")

    print()
    print(f"  {'RUN':>4}  {'DB EXEC ms':>11}  {'FORMAT ms':>10}  {'DISPLAY ms':>11}  {'TOTAL ms':>9}  {'ROWS':>6}  STATUS")
    print("  " + "-" * 62)
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"  {r.run_no:>4}  {r.db_exec_ms:>11.1f}  {r.formatting_ms:>10.1f}  "
              f"{r.display_ms:>11.1f}  {r.total_ms:>9.1f}  {r.row_count:>6}  {status}")
    print("═" * 72 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark execute_query_safe → _results_to_records → display"
    )
    p.add_argument("--host",     default=os.getenv("DB_HOST", "localhost"))
    p.add_argument("--port",     default=os.getenv("DB_PORT", "1433"))
    p.add_argument("--user",     default=os.getenv("DB_USER", "sa"))
    p.add_argument("--password", default=os.getenv("DB_PASSWORD", ""))
    p.add_argument("--database", default=os.getenv("DB_NAME",
                                  os.getenv("DB_DATABASE", "master")))
    p.add_argument("--connect-timeout", type=int, default=10)
    p.add_argument("--query-timeout",   type=int, default=30)
    p.add_argument("--max-rows",        type=int, default=500)
    p.add_argument("--runs",            type=int, default=1)
    p.add_argument("--sql",      default=None)
    p.add_argument("--sql-file", default=None)
    p.add_argument("--no-display", action="store_true",
                   help="Skip printing result rows (timing only)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    sql = (Path(args.sql_file).read_text().strip() if args.sql_file
           else args.sql.strip() if args.sql
           else DEFAULT_SQL)

    print()
    print("┌─ DB Bench ─────────────────────────────────────────────────────┐")
    print(f"│  host     : {args.host}:{args.port}")
    print(f"│  database : {args.database}")
    print(f"│  runs     : {args.runs}   max_rows={args.max_rows}   timeout={args.query_timeout}s")
    print(f"│  stages   : execute_query_safe → _results_to_records → display")
    sql_preview = sql.strip()[:72].replace("\n", " ") + ("…" if len(sql.strip()) > 72 else "")
    print(f"│  SQL      : {sql_preview}")
    print("└────────────────────────────────────────────────────────────────┘")

    # ── build engine + SQLDatabase ────────────────────────────────────────────
    config = DatabaseConfig(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        database=args.database,
        connect_timeout=args.connect_timeout,
        query_timeout=args.query_timeout,
        view_support=True,
        lazy_table_reflection=True,
    )

    log.info("Connecting …")
    try:
        engine = create_engine_with_timeout(config)
        db = SQLDatabase(engine=engine, lazy_table_reflection=True)
        log.info("Connected")
    except Exception as exc:
        log.error("Connection failed: %s", exc)
        sys.exit(1)

    # ── runs ──────────────────────────────────────────────────────────────────
    results: List[RunResult] = []
    for i in range(1, args.runs + 1):
        r = run_once(db, sql, run_no=i,
                     timeout_seconds=args.query_timeout,
                     max_rows=args.max_rows,
                     show_results=not args.no_display)
        results.append(r)

    print_summary(results)
    engine.dispose()


if __name__ == "__main__":
    main()
