from backend.services import sql_engine


class _DummyDB:
    _engine = object()


def test_supplier_price_comparison_fallback_sql_is_generated(monkeypatch):
    monkeypatch.setattr(
        sql_engine,
        "_table_exists_fast",
        lambda _db, table: table in {"BookingData", "suppliermaster_Report"},
    )
    q = "compare the company buying price vs agent buying price for each supplier - show the variance and percentage difference"
    sql = sql_engine.build_supplier_price_comparison_fallback_sql(q, _DummyDB())
    assert sql is not None
    assert "FROM dbo.[BookingData] BD" in sql
    assert "LEFT JOIN SupplierMap SM" in sql
    assert "Variance Percentage" in sql
    assert "GROUP BY" in sql


def test_deterministic_timeout_fallback_prefers_supplier_price_comparison(monkeypatch):
    monkeypatch.setattr(
        sql_engine,
        "_table_exists_fast",
        lambda _db, table: table in {"BookingData", "suppliermaster_Report"},
    )
    q = "compare the company buying price vs agent buying price for each supplier - show the variance and percentage difference"
    sql, reason = sql_engine._deterministic_timeout_fallback_sql(q, _DummyDB())
    assert sql is not None
    assert reason == "supplier_price_comparison"
