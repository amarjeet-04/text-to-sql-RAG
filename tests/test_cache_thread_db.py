import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase

from app.db_utils import execute_query_safe
from backend.services.sql_engine import GlobalQueryCache, is_time_sensitive


class _CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_query(self, query: str):
        self.calls += 1
        q = (query or "").lower()
        return [1.0, 0.0] if "revenue" in q else [0.0, 1.0]


def test_exact_cache_hit_is_fast_and_does_not_embed():
    cache = GlobalQueryCache(max_size=16, ttl_seconds=300, enable_semantic=False)
    embedder = _CountingEmbedder()

    cache.add("show revenue", "SELECT 1 AS revenue", [{"revenue": 1}], embedder, db=None)
    assert embedder.calls == 0

    sql, rows = cache.find("show revenue", embedder, db=None)
    assert sql == "SELECT 1 AS revenue"
    assert rows == [{"revenue": 1}]
    # Exact hit path should not call embedder when semantic is disabled.
    assert embedder.calls == 0


def test_cache_ttl_expiry():
    cache = GlobalQueryCache(max_size=8, ttl_seconds=1, enable_semantic=False)
    embedder = _CountingEmbedder()
    cache.add("show revenue", "SELECT 1 AS revenue", [{"revenue": 1}], embedder, db=None)

    sql, _ = cache.find("show revenue", embedder, db=None)
    assert sql is not None

    time.sleep(1.15)
    sql_after, rows_after = cache.find("show revenue", embedder, db=None)
    assert sql_after is None
    assert rows_after is None


def test_global_cache_concurrent_access_thread_safe():
    cache = GlobalQueryCache(max_size=256, ttl_seconds=120, enable_semantic=False)
    embedder = _CountingEmbedder()

    def _task(i: int):
        q = f"revenue agent {i % 20}"
        cache.add(q, f"SELECT {i} AS v", [{"v": i}], embedder, db=None)
        cache.find(q, embedder, db=None)

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(_task, i) for i in range(400)]
        for f in futures:
            f.result()

    # If locks are missing this test intermittently crashes; reaching here is success.
    assert True


def test_time_sensitive_questions_bypass_cache():
    cache = GlobalQueryCache(max_size=16, ttl_seconds=300, enable_semantic=False)
    embedder = _CountingEmbedder()

    assert is_time_sensitive(question="show revenue today", sql="")
    assert is_time_sensitive(question="business this month", sql="")
    assert is_time_sensitive(question="", sql="SELECT GETDATE()")

    cache.add("show revenue today", "SELECT 1", [{"v": 1}], embedder, db=None)
    sql, rows = cache.find("show revenue today", embedder, db=None)
    assert sql is None
    assert rows is None


def test_db_exception_does_not_poison_followup_query():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE bookingdata (id INTEGER, amount REAL)"))
        conn.execute(text("INSERT INTO bookingdata (id, amount) VALUES (1, 10.5), (2, 20.0)"))
    db = SQLDatabase(engine=engine)

    # First query fails.
    df_bad, err_bad = execute_query_safe(db, "SELECT invalid_col FROM bookingdata", timeout_seconds=5, max_rows=100)
    assert df_bad is None
    assert err_bad is not None

    # Next query should still succeed (no poisoned transaction state).
    df_ok, err_ok = execute_query_safe(db, "SELECT id, amount FROM bookingdata ORDER BY id", timeout_seconds=5, max_rows=100)
    assert err_ok is None
    assert df_ok is not None
    assert len(df_ok) == 2
