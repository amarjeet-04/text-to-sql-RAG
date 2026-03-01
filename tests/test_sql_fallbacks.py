import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.services import sql_engine


class _DummyDB:
    _engine = object()


class _DummyLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)

        class _Resp:
            def __init__(self, content: str):
                self.content = content

        return _Resp(self.response_text)


def test_extract_followup_entity_candidate_handles_detail_prompts():
    assert sql_engine._extract_followup_entity_candidate("give me more info about webbeds") == "webbeds"
    assert sql_engine._extract_followup_entity_candidate("what is the cancellation rate for Expedia Rapid") == "Expedia Rapid"


def test_modify_sql_for_filter_includes_stored_procedure_guidance():
    llm = _DummyLLM("SELECT * FROM dbo.BookingData WHERE AgentName LIKE '%webbeds%'")
    original_sql = "SELECT AgentName, SUM(AgentBuyingPrice) AS total_sales FROM dbo.BookingData GROUP BY AgentName"

    sql_engine.modify_sql_for_filter(
        original_sql,
        "show me only webbeds",
        llm,
        stored_guidance="ALWAYS use BookingData as base table",
    )

    assert llm.prompts
    assert "DOMAIN BUSINESS RULES:" in llm.prompts[0]
    assert "ALWAYS use BookingData as base table" in llm.prompts[0]


def test_modify_sql_for_sort_includes_stored_procedure_guidance():
    llm = _DummyLLM("ORDER BY total_sales DESC")

    sql_engine.modify_sql_for_sort(
        "SELECT AgentName, SUM(AgentBuyingPrice) AS total_sales FROM dbo.BookingData GROUP BY AgentName",
        "sort it in a custom way",
        llm,
        stored_guidance="Revenue means SUM(AgentBuyingPrice)",
    )

    assert llm.prompts
    assert "DOMAIN BUSINESS RULES:" in llm.prompts[0]
    assert "Revenue means SUM(AgentBuyingPrice)" in llm.prompts[0]


def test_modify_sql_for_filter_allows_dimension_override_for_followup():
    llm = _DummyLLM("SELECT SupplierName, SUM(AgentBuyingPrice) AS total_sales FROM dbo.BookingData GROUP BY SupplierName")

    sql_engine.modify_sql_for_filter(
        "SELECT AgentName, SUM(AgentBuyingPrice) AS total_sales FROM dbo.BookingData GROUP BY AgentName",
        "give me more info about webbeds",
        llm,
        stored_guidance="Use suppliermaster_Report for supplier names",
        followup_override={
            "candidate": "webbeds",
            "dimension": "supplier",
            "name_col": "SupplierName",
            "table": "suppliermaster_Report",
        },
    )

    assert llm.prompts
    prompt = llm.prompts[0]
    assert "IMPORTANT FOLLOW-UP OVERRIDE:" in prompt
    assert "webbeds" in prompt
    assert "suppliermaster_Report.SupplierName" in prompt
    assert "you MAY change SELECT columns, GROUP BY, and JOINs" in prompt


def test_metric_intent_validator_rejects_room_nights_for_profit_question():
    sql = """
    SELECT
      SUM(DATEDIFF(DAY, BD.CheckInDate, BD.CheckOutDate)) AS TotalRoomNights
    FROM BookingData BD WITH (NOLOCK)
    LEFT JOIN dbo.AgentMaster_V1 A WITH (NOLOCK) ON BD.AgentId = A.AgentId
    WHERE BD.BookingStatus NOT IN ('Cancelled', 'Not Confirmed', 'On Request')
      AND BD.CreatedDate >= '2024-01-01'
      AND BD.CreatedDate < '2025-01-01'
    GROUP BY A.AgentName
    ORDER BY SUM(COALESCE(BD.AgentBuyingPrice,0) - COALESCE(BD.CompanyBuyingPrice,0)) DESC
    """

    result = sql_engine._validate_metric_intent_match(
        "Which travel agent generated the most profit in 2024?",
        sql,
    )

    assert result["ok_to_execute"] is False
    assert result["failure_type"] == "intent_mismatch"
    assert "question expects profit" in result["reasons"][0]
    assert "room_nights" in result["reasons"][0]


def test_validate_sql_candidate_accepts_profit_projection_for_profit_question():
    sql = """
    WITH AM AS (
      SELECT AgentId, AgentName FROM dbo.AgentMaster_V1 WITH (NOLOCK)
    )
    SELECT TOP 1 [Agent Name], [Total Profit]
    FROM (
      SELECT
        AM.AgentName AS [Agent Name],
        SUM(BD.AgentBuyingPrice - BD.CompanyBuyingPrice) AS [Total Profit]
      FROM dbo.BookingData BD WITH (NOLOCK)
      LEFT JOIN AM ON AM.AgentId = BD.AgentId
      WHERE BD.BookingStatus NOT IN ('Cancelled', 'Not Confirmed', 'On Request')
        AND BD.CheckInDate >= '2024-01-01'
        AND BD.CheckInDate < '2025-01-01'
      GROUP BY AM.AgentName
    ) main
    ORDER BY [Total Profit] DESC;
    """

    is_valid, message = sql_engine._validate_sql_candidate(
        "Which travel agent generated the most profit in 2024?",
        sql,
    )

    assert is_valid is True
    assert message is None or isinstance(message, str)


def test_performance_guardrails_rewrite_chain_country_to_destination_country():
    sql = """
    WITH HC AS (
      SELECT HotelId, HotelName, Country,
        CASE
          WHEN chain IS NULL THEN 'No Chain Hotels'
          ELSE chain
        END AS Chain
      FROM HotelChain
    )
    SELECT HC.Chain, HC.Country, SUM(COALESCE(BD.AgentBuyingPrice,0)) AS TotalSales
    FROM BookingData BD
    LEFT JOIN HC ON BD.ProductId = HC.HotelId
    WHERE BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')
      AND BD.CreatedDate >= '2025-12-01'
      AND BD.CreatedDate < '2026-03-01'
    GROUP BY HC.Chain, HC.Country
    """

    guarded = sql_engine.performance_guardrails(
        sql,
        question="Compare revenue by hotel chain and country for last 3 months",
        dialect="sqlserver",
    )

    assert "LEFT JOIN dbo.Master_Country MC_DEST ON MC_DEST.CountryID = BD.ProductCountryid" in guarded
    assert "SELECT HC.Chain, MC_DEST.Country, SUM(COALESCE(BD.AgentBuyingPrice,0)) AS TotalSales" in guarded
    assert "GROUP BY HC.Chain, MC_DEST.Country" in guarded
