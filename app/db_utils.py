"""
Database Utilities - Timeout, View Support, and Safe Query Execution
"""
import re
import signal
import os
import time
import logging
from typing import Optional, List, Tuple, Any, Dict
from functools import wraps
from dataclasses import dataclass
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import pandas as pd
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from langchain_community.utilities import SQLDatabase

# Import shared runtime utilities for thread-safe execution
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.services.runtime import run_with_timeout, log_event

# Import caching utilities
from functools import lru_cache
import threading

logger = logging.getLogger("db_utils")


# Global engine cache with TTL
_ENGINE_CACHE: Dict[str, Engine] = {}
_ENGINE_CACHE_LOCK = threading.RLock()
_ENGINE_CACHE_TTL = 3600  # 1 hour TTL


def _get_cached_engine(connection_uri: str) -> Optional[Engine]:
    """Get cached engine by connection URI."""
    with _ENGINE_CACHE_LOCK:
        engine = _ENGINE_CACHE.get(connection_uri)
        if engine:
            # Check if engine is still alive
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return engine
            except Exception:
                # Engine is dead, remove from cache
                _ENGINE_CACHE.pop(connection_uri, None)
    return None


def _set_cached_engine(connection_uri: str, engine: Engine) -> None:
    """Cache engine by connection URI."""
    with _ENGINE_CACHE_LOCK:
        _ENGINE_CACHE[connection_uri] = engine
        # Limit cache size to prevent memory issues
        if len(_ENGINE_CACHE) > 100:
            # Remove oldest entries
            keys_to_remove = list(_ENGINE_CACHE.keys())[:-50]
            for key in keys_to_remove:
                _ENGINE_CACHE.pop(key, None)


def _set_cached_engine(connection_uri: str, engine: Engine) -> None:
    """Cache engine by connection URI."""
    with _ENGINE_CACHE_LOCK:
        _ENGINE_CACHE[connection_uri] = engine
        # Limit cache size to prevent memory issues
        if len(_ENGINE_CACHE) > 100:
            # Remove oldest entries
            keys_to_remove = list(_ENGINE_CACHE.keys())[:-50]
            for key in keys_to_remove:
                _ENGINE_CACHE.pop(key, None)


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    port: str
    username: str
    password: str
    database: str

    # Timeout settings (in seconds)
    connect_timeout: int = 10
    query_timeout: int = 30

    # View support
    view_support: bool = True
    include_tables: Optional[List[str]] = None  # If set, only these tables/views
    ignore_tables: Optional[List[str]] = None   # Tables to exclude

    # Performance settings
    sample_rows_in_table_info: int = 1
    lazy_table_reflection: bool = False

    @property
    def connection_uri(self) -> str:
        """Build Microsoft SQL Server connection URI using ODBC Driver 18."""
        from urllib.parse import quote_plus
        host = (self.host or "").strip()
        port = (self.port or "").strip()
        database = (self.database or "").strip()
        username = self.username or ""
        password = self.password or ""
        login_timeout = int(self.connect_timeout)
        query_timeout = int(self.query_timeout)
        odbc_params = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={host},{port};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"LoginTimeout={login_timeout};"
            f"QueryTimeout={query_timeout};"
            f"TrustServerCertificate=yes;"
        )
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_params)}"


class QueryTimeoutError(Exception):
    """Raised when a query exceeds the timeout limit"""
    pass


class QueryExecutionError(Exception):
    """Raised when query execution fails"""
    pass


def create_engine_with_timeout(config: DatabaseConfig) -> Engine:
    """
    Create SQLAlchemy engine with connection and query timeouts.
    Uses caching by connection URI to avoid recreating engines.

    Uses ODBC Driver 18 for SQL Server (faster than pymssql).
    Timeouts are set in the ODBC connection string via connection_uri.
    """
    # Check cache first
    cached_engine = _get_cached_engine(config.connection_uri)
    if cached_engine:
        log_event(logger, logging.DEBUG, "engine_cache_hit", connection_uri_hash=hash(config.connection_uri))
        return cached_engine
    
    # Create new engine
    engine = create_engine(
        config.connection_uri,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=300,    # Recycle connections every 5 minutes (remote DBs drop idle connections)
        pool_size=5,
        max_overflow=10,
        # SQL Server specific optimizations
        isolation_level="READ UNCOMMITTED",  # Reduce locking contention
        echo=False,  # Disable SQL logging for performance
        echo_pool=False,  # Disable pool logging for performance
    )

    # Apply session-level settings on EVERY new connection from the pool,
    # including LangChain's internal INFORMATION_SCHEMA introspection queries.
    @event.listens_for(engine, "connect")
    def _set_session_options(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("SET DEADLOCK_PRIORITY LOW")
            cursor.execute("SET LOCK_TIMEOUT 8000")   # 8 s — covers slow metadata queries
        finally:
            cursor.close()

    # Warm up the connection pool — first query is faster
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        # Cache the engine after successful warmup
        _set_cached_engine(config.connection_uri, engine)
        log_event(logger, logging.INFO, "engine_created_and_cached", connection_uri_hash=hash(config.connection_uri))
    except Exception as e:
        # Pool warming is best-effort; actual errors surface on real queries
        logger.debug(f"Engine warmup failed: {e}")

    return engine


def create_database_with_views(config: DatabaseConfig) -> SQLDatabase:
    """
    Create LangChain SQLDatabase with view support and timeout configuration.

    Benefits of using views:
    - Simpler schema for LLM to understand
    - Pre-joined data reduces query complexity
    - Security: hide internal columns
    - Business-friendly naming
    """
    engine = create_engine_with_timeout(config)

    db = SQLDatabase(
        engine=engine,
        schema=None,  # Use default schema
        view_support=config.view_support,
        include_tables=config.include_tables,
        ignore_tables=config.ignore_tables,
        sample_rows_in_table_info=config.sample_rows_in_table_info,
        lazy_table_reflection=config.lazy_table_reflection,
    )

    return db


@contextmanager
def query_timeout_context(seconds: int):
    """
    Context manager for query timeout using shared runtime executor.

    Usage:
        with query_timeout_context(30):
            result = execute_query(...)
    """
    class TimeoutContext:
        def run(self, func, *args, **kwargs):
            # Use shared runtime executor instead of creating new ThreadPoolExecutor
            from backend.services.runtime import run_with_timeout
            try:
                return run_with_timeout(func, timeout_seconds=seconds, *args, **kwargs)
            except FuturesTimeoutError as exc:
                raise QueryTimeoutError(f"Query exceeded {seconds} second timeout") from exc

    yield TimeoutContext()


def _is_transient_operational_error(error_text: str) -> bool:
    text_low = (error_text or "").lower()
    transient_signals = (
        "timeout",
        "timed out",
        "could not connect",
        "connection reset",
        "connection is busy",
        "deadlock victim",
        "transport-level error",
        "communication link failure",
    )
    return any(sig in text_low for sig in transient_signals)


def execute_query_safe(
    db: SQLDatabase,
    query: str,
    timeout_seconds: int = 30,
    max_rows: int = 1000
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute a SQL query with timeout protection and row limiting.
    
    Uses shared runtime executor instead of per-call ThreadPoolExecutor.
    
    SQL Server optimizations:
    - SET DEADLOCK_PRIORITY LOW
    - SET LOCK_TIMEOUT
    - SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED (if consistent with NOLOCK)

    Args:
        db: LangChain SQLDatabase instance
        query: SQL query string
        timeout_seconds: Maximum execution time
        max_rows: Maximum rows to return (prevents memory issues)

    Returns:
        Tuple of (DataFrame or None, error message or None)
    """
    import sqlalchemy

    query_stripped = query.rstrip().rstrip(';')
    query_upper = query_stripped.upper()

    engine = db._engine
    dialect = (getattr(getattr(engine, "dialect", None), "name", "") or "").lower()
    if "mssql" in dialect:
        # Add TOP N row cap only when the query has no row limit already.
        # We check both the outer SELECT and — for CTE/WITH queries — the final
        # SELECT that follows the last closing parenthesis, so we never inject
        # TOP N inside a CTE body where it would be a syntax error.
        has_top = "TOP " in query_upper
        last_select_pos = query_upper.rfind("SELECT")
        is_cte = query_upper.lstrip().startswith("WITH")
        if not has_top:
            if not is_cte and query_upper.lstrip().startswith("SELECT"):
                if not ("COUNT(*)" in query_upper and "GROUP BY" not in query_upper):
                    query_stripped = re.sub(
                        r"(?i)^SELECT\b",
                        f"SELECT TOP {max_rows}",
                        query_stripped,
                        count=1,
                    )
            elif is_cte and last_select_pos != -1:
                before = query_stripped[:last_select_pos]
                after = query_stripped[last_select_pos:]
                after = re.sub(r"(?i)^SELECT\b", f"SELECT TOP {max_rows}", after, count=1)
                query_stripped = before + after
    else:
        # Non-SQL Server fallback for tests/local tooling.
        has_limit = re.search(r"(?i)\bLIMIT\s+\d+\b", query_stripped) is not None
        if not has_limit and not re.search(r"(?i)\bCOUNT\s*\(\s*\*\s*\)", query_stripped):
            query_stripped = query_stripped.rstrip() + f" LIMIT {max_rows}"

    max_retries = max(0, int(os.getenv("DB_TRANSIENT_RETRIES", "1")))
    base_backoff = max(0.05, float(os.getenv("DB_TRANSIENT_RETRY_BACKOFF_SECONDS", "0.2")))

    attempt = 0
    while True:
        try:
            def _execute_query():
                with engine.connect() as conn:
                    # SQL Server optimizations for faster execution
                    dialect = (getattr(getattr(engine, "dialect", None), "name", "") or "").lower()
                    if "mssql" in dialect:
                        # Reduce deadlock likelihood and improve concurrency
                        conn.execute(text("SET DEADLOCK_PRIORITY LOW"))
                        # Set lock timeout to prevent long waits
                        lock_timeout_ms = int(timeout_seconds * 1000)
                        conn.execute(text(f"SET LOCK_TIMEOUT {lock_timeout_ms}"))
                        # Use read uncommitted if consistent with NOLOCK usage
                        # This matches the existing NOLOCK pattern in the codebase
                        conn.execute(text("SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED"))
                    
                    result_proxy = conn.execute(text(query_stripped))
                    columns = list(result_proxy.keys())
                    # fetchmany in chunks avoids materialising the full result set
                    # in one blocking call — the driver returns control sooner for
                    # large result sets and keeps peak memory lower.
                    chunk_size = 500
                    rows: list = []
                    while True:
                        chunk = result_proxy.fetchmany(chunk_size)
                        if not chunk:
                            break
                        rows.extend(chunk)
                        if len(rows) >= max_rows:
                            rows = rows[:max_rows]
                            break

                if rows:
                    df = pd.DataFrame(rows, columns=columns)
                    return df, None
                return pd.DataFrame(), None
            
            # Use shared runtime executor for timeout protection
            result = run_with_timeout(_execute_query, timeout_seconds)
            return result
            
        except FuturesTimeoutError:
            return None, f"Query timed out after {timeout_seconds} seconds. Try a more specific query."
        except sqlalchemy.exc.OperationalError as e:
            error_str = str(e)
            if attempt < max_retries and _is_transient_operational_error(error_str):
                delay = round(base_backoff * (2 ** attempt), 3)
                logger.warning(
                    "transient_db_error retrying attempt=%s max=%s delay=%ss error=%s",
                    attempt + 1,
                    max_retries,
                    delay,
                    error_str[:180],
                )
                time.sleep(delay)
                attempt += 1
                continue
            if "timeout" in error_str.lower() or "cancelled" in error_str.lower():
                return None, f"Query timed out after {timeout_seconds} seconds. Try a more specific query."
            return None, f"Database error: {error_str}"
        except Exception as e:
            return None, f"Query execution failed: {str(e)}"


def validate_sql_dry_run(db: SQLDatabase, query: str) -> Tuple[bool, Optional[str]]:
    """Validate SQL without executing it using SET NOEXEC ON.

    This catches column-not-found, table-not-found, and syntax errors
    without actually running the query (no I/O, no locks, fast).
    Returns (True, None) on success or (False, error_message) on failure.
    """
    import sqlalchemy

    try:
        engine = db._engine
        dialect = (getattr(getattr(engine, "dialect", None), "name", "") or "").lower()
        if "mssql" not in dialect:
            # NOEXEC is SQL Server specific; don't block execution for other dialects.
            return True, None
        with engine.connect() as conn:
            conn.execute(text("SET NOEXEC ON"))
            try:
                conn.execute(text(query.rstrip().rstrip(";")))
                return True, None
            except sqlalchemy.exc.OperationalError as e:
                return False, str(e).split("\n")[0]
            except Exception as e:
                return False, str(e).split("\n")[0]
            finally:
                try:
                    conn.execute(text("SET NOEXEC OFF"))
                except Exception:
                    pass
                try:
                    conn.rollback()
                except Exception:
                    pass
    except Exception as e:
        # Connection-level error — skip dry-run, let real execution handle it
        return True, None


def get_views_and_tables(db: SQLDatabase) -> dict:
    """
    Get list of available tables and views in the database.

    Returns:
        {
            "tables": ["table1", "table2"],
            "views": ["view1", "view2"],
            "all": ["table1", "table2", "view1", "view2"]
        }
    """
    import sqlalchemy

    inspector = sqlalchemy.inspect(db._engine)

    tables = inspector.get_table_names()
    views = inspector.get_view_names()

    return {
        "tables": tables,
        "views": views,
        "all": list(db.get_usable_table_names())
    }


def recommend_views_for_text_to_sql(db: SQLDatabase) -> List[str]:
    """
    Analyze schema and recommend views that would simplify text-to-SQL.

    Returns list of SQL statements to create recommended views.
    """
    import sqlalchemy

    inspector = sqlalchemy.inspect(db._engine)
    recommendations = []

    tables = inspector.get_table_names()

    # Find tables with foreign key relationships
    join_candidates = {}
    for table in tables:
        fks = inspector.get_foreign_keys(table)
        for fk in fks:
            referred_table = fk["referred_table"]
            key = tuple(sorted([table, referred_table]))
            if key not in join_candidates:
                join_candidates[key] = {
                    "tables": [table, referred_table],
                    "joins": []
                }
            join_candidates[key]["joins"].append({
                "from_table": table,
                "from_cols": fk["constrained_columns"],
                "to_table": referred_table,
                "to_cols": fk["referred_columns"]
            })

    # Generate view recommendations
    for key, info in join_candidates.items():
        table1, table2 = info["tables"]
        join_info = info["joins"][0]

        view_name = f"vw_{table1}_{table2}"
        recommendation = f"""
-- Recommended view: {view_name}
-- Simplifies queries joining {table1} and {table2}
CREATE OR REPLACE VIEW {view_name} AS
SELECT
    t1.*,
    t2.*
FROM {join_info['from_table']} t1
JOIN {join_info['to_table']} t2
    ON t1.{join_info['from_cols'][0]} = t2.{join_info['to_cols'][0]};
"""
        recommendations.append(recommendation)

    return recommendations


# Convenience function for quick setup
def quick_connect(
    host: str,
    port: str,
    username: str,
    password: str,
    database: str,
    query_timeout: int = 30,
    view_support: bool = True
) -> SQLDatabase:
    """
    Quick connection setup with sensible defaults.

    Usage:
        db = quick_connect("localhost", "5432", "user", "pass", "mydb")
    """
    config = DatabaseConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        query_timeout=query_timeout,
        view_support=view_support
    )

    return create_database_with_views(config)
