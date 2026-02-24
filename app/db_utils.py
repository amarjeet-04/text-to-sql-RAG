"""
Database Utilities - Timeout, View Support, and Safe Query Execution
"""
import re
import signal
import threading
from typing import Optional, List, Tuple, Any
from functools import wraps
from dataclasses import dataclass
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from langchain_community.utilities import SQLDatabase


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

    Uses ODBC Driver 18 for SQL Server (faster than pymssql).
    Timeouts are set in the ODBC connection string via connection_uri.
    """
    engine = create_engine(
        config.connection_uri,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=300,    # Recycle connections every 5 minutes (remote DBs drop idle connections)
        pool_size=5,
        max_overflow=10,
    )

    # Warm up the connection pool — first query is faster
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pass  # Pool warming is best-effort; actual errors surface on real queries

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
    Context manager for query timeout using threading (cross-platform).

    Usage:
        with query_timeout_context(30):
            result = execute_query(...)
    """
    result = [None]
    exception = [None]
    completed = threading.Event()

    def target(func, args, kwargs):
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
        finally:
            completed.set()

    class TimeoutContext:
        def run(self, func, *args, **kwargs):
            thread = threading.Thread(target=target, args=(func, args, kwargs))
            thread.daemon = True
            thread.start()

            if not completed.wait(timeout=seconds):
                raise QueryTimeoutError(f"Query exceeded {seconds} second timeout")

            if exception[0]:
                raise exception[0]

            return result[0]

    yield TimeoutContext()


def execute_query_safe(
    db: SQLDatabase,
    query: str,
    timeout_seconds: int = 30,
    max_rows: int = 1000
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute a SQL query with timeout protection and row limiting.

    Args:
        db: LangChain SQLDatabase instance
        query: SQL query string
        timeout_seconds: Maximum execution time
        max_rows: Maximum rows to return (prevents memory issues)

    Returns:
        Tuple of (DataFrame or None, error message or None)
    """
    import sqlalchemy

    # Add TOP N row cap only when the query has no row limit already.
    # We check both the outer SELECT and — for CTE/WITH queries — the final
    # SELECT that follows the last closing parenthesis, so we never inject
    # TOP N inside a CTE body where it would be a syntax error.
    query_stripped = query.rstrip().rstrip(';')
    query_upper = query_stripped.upper()
    has_top = "TOP " in query_upper
    # Locate the final SELECT statement (after any CTE block)
    last_select_pos = query_upper.rfind("SELECT")
    is_cte = query_upper.lstrip().startswith("WITH")
    if not has_top:
        if not is_cte and query_upper.lstrip().startswith("SELECT"):
            # Simple SELECT — inject directly after SELECT keyword
            if not ("COUNT(*)" in query_upper and "GROUP BY" not in query_upper):
                query_stripped = re.sub(
                    r'(?i)^SELECT\b',
                    f'SELECT TOP {max_rows}',
                    query_stripped,
                    count=1,
                )
        elif is_cte and last_select_pos != -1:
            # CTE query — inject TOP N into the final outer SELECT only
            before = query_stripped[:last_select_pos]
            after = query_stripped[last_select_pos:]
            after = re.sub(r'(?i)^SELECT\b', f'SELECT TOP {max_rows}', after, count=1)
            query_stripped = before + after

    try:
        engine = db._engine

        with engine.connect() as conn:
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
        else:
            return pd.DataFrame(), None

    except sqlalchemy.exc.OperationalError as e:
        error_str = str(e)
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
