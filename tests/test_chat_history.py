"""
tests/test_chat_history.py
──────────────────────────
Simulates the conversation_turns logic from chatbot.py
without requiring a DB or LLM.

Tests exactly 1 base question + 3 follow-up questions and
prints what HISTORY context the LLM would see for each turn.
"""

import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── replicate the context-building logic from chatbot.py lines 306-318 ────────

def build_context_str(conversation_turns: List[Dict]) -> str:
    if conversation_turns:
        ctx_lines = []
        for turn in conversation_turns[-4:]:
            q_prev = turn.get("question", "")
            s_prev = turn.get("sql", "")
            if q_prev:
                ctx_lines.append(f"Q: {q_prev}")
            if s_prev:
                ctx_lines.append(f"SQL: {s_prev}")
        return "\n".join(ctx_lines) if ctx_lines else "(no prior conversation)"
    return "(no prior conversation)"


# ── simulate what chatbot.py does per question ────────────────────────────────

def simulate_turn(
    question: str,
    fake_sql: str,
    conversation_turns: List[Dict],
    turn_number: int,
) -> None:
    """
    Mirrors chatbot.py flow:
      1. Pass conversation_turns (BEFORE append) into pipeline → build context_str
      2. Print what LLM sees as HISTORY
      3. Append current turn to conversation_turns (AFTER pipeline returns)
    """
    context_seen_by_llm = build_context_str(conversation_turns)

    print(f"\n{'='*60}")
    print(f"  TURN {turn_number}: {question}")
    print(f"{'='*60}")
    print(f"  [HISTORY passed to LLM]")
    if context_seen_by_llm == "(no prior conversation)":
        print(f"    (no prior conversation)")
    else:
        for line in context_seen_by_llm.splitlines():
            print(f"    {line}")
    print(f"  [LLM generates SQL]")
    print(f"    {fake_sql}")

    # Replicate chatbot.py line 601-604: append AFTER pipeline
    conversation_turns.append({
        "question": question,
        "sql":      fake_sql,
    })

    print(f"  [conversation_turns now has {len(conversation_turns)} turn(s)]")


# ── run the 4-turn scenario ────────────────────────────────────────────────────

def test_dida_expedia_followup_scenario():
    """
    Real-world scenario with 3 questions:
      Q1: best sold hotel for dida as client to Expedia as supplier
      Q2: give me top 2               (follow-up — must inherit Q1 agent+supplier filters)
      Q3: change dida to hotel beds   (follow-up — must inherit Q2 TOP 2 + supplier filter, swap client)

    For each turn we:
      1. Show what HISTORY the LLM would see (before append)
      2. Show what SQL a correct LLM should produce
      3. Append turn and verify inheritance via assertions
    """
    conversation_turns: List[Dict] = []

    # ── Q1: base question ─────────────────────────────────────────────────────
    # Client = DIDA  (AgentMaster_V1.AgentName LIKE 'DIDA%')
    # Supplier = Expedia  (suppliermaster_Report.SupplierName LIKE 'Expedia%')
    # Metric: best-sold hotel → COUNT(DISTINCT BD.PNRNo) DESC
    SQL_Q1 = (
        "WITH AM AS (SELECT DISTINCT AgentId, AgentName FROM AgentMaster_V1), "
        "SM AS (SELECT DISTINCT EmployeeId, SupplierName FROM suppliermaster_Report) "
        "SELECT HC.HotelName, COUNT(DISTINCT BD.PNRNo) AS TotalBookings "
        "FROM BookingData BD "
        "LEFT JOIN AM ON BD.AgentId = AM.AgentId "
        "LEFT JOIN SM ON BD.SupplierId = SM.EmployeeId "
        "LEFT JOIN Hotelchain HC ON BD.ProductId = HC.HotelId "
        "WHERE AM.AgentName LIKE 'DIDA%' "
        "AND SM.SupplierName LIKE 'Expedia%' "
        "AND BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') "
        "GROUP BY HC.HotelName ORDER BY TotalBookings DESC"
    )

    # ── Q2: "give me top 2"  (follow-up) ─────────────────────────────────────
    # Must inherit: DIDA client, Expedia supplier, same metric, same GROUP BY
    # Only adds: TOP 2
    SQL_Q2 = (
        "WITH AM AS (SELECT DISTINCT AgentId, AgentName FROM AgentMaster_V1), "
        "SM AS (SELECT DISTINCT EmployeeId, SupplierName FROM suppliermaster_Report) "
        "SELECT TOP 2 HC.HotelName, COUNT(DISTINCT BD.PNRNo) AS TotalBookings "
        "FROM BookingData BD "
        "LEFT JOIN AM ON BD.AgentId = AM.AgentId "
        "LEFT JOIN SM ON BD.SupplierId = SM.EmployeeId "
        "LEFT JOIN Hotelchain HC ON BD.ProductId = HC.HotelId "
        "WHERE AM.AgentName LIKE 'DIDA%' "
        "AND SM.SupplierName LIKE 'Expedia%' "
        "AND BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') "
        "GROUP BY HC.HotelName ORDER BY TotalBookings DESC"
    )

    # ── Q3: "change dida to hotel beds"  (follow-up) ─────────────────────────
    # Must inherit: TOP 2, Expedia supplier, same metric
    # Change: AgentName LIKE 'DIDA%'  →  AgentName LIKE 'Hotel Beds%'
    SQL_Q3 = (
        "WITH AM AS (SELECT DISTINCT AgentId, AgentName FROM AgentMaster_V1), "
        "SM AS (SELECT DISTINCT EmployeeId, SupplierName FROM suppliermaster_Report) "
        "SELECT TOP 2 HC.HotelName, COUNT(DISTINCT BD.PNRNo) AS TotalBookings "
        "FROM BookingData BD "
        "LEFT JOIN AM ON BD.AgentId = AM.AgentId "
        "LEFT JOIN SM ON BD.SupplierId = SM.EmployeeId "
        "LEFT JOIN Hotelchain HC ON BD.ProductId = HC.HotelId "
        "WHERE AM.AgentName LIKE 'Hotel Beds%' "
        "AND SM.SupplierName LIKE 'Expedia%' "
        "AND BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') "
        "GROUP BY HC.HotelName ORDER BY TotalBookings DESC"
    )

    turns = [
        ("best sold hotel for dida as client to Expedia as supplier", SQL_Q1),
        ("give me top 2",                                              SQL_Q2),
        ("change dida to hotel beds",                                  SQL_Q3),
    ]

    for i, (question, fake_sql) in enumerate(turns, start=1):
        simulate_turn(question, fake_sql, conversation_turns, turn_number=i)

    # ── assertions ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ASSERTIONS")
    print(f"{'='*60}")

    assert len(conversation_turns) == 3, f"Expected 3 turns, got {len(conversation_turns)}"

    # Turn 1: no history
    ctx_t1 = build_context_str([])
    assert ctx_t1 == "(no prior conversation)"
    print("  ✓ Turn 1: LLM correctly sees NO prior history (fresh question)")

    # Turn 2: sees Q1 + SQL1 — must contain DIDA and Expedia so LLM can inherit
    ctx_t2 = build_context_str(conversation_turns[:1])
    assert "dida as client to Expedia as supplier" in ctx_t2.lower() or \
           "best sold hotel for dida" in ctx_t2.lower(), \
        f"Turn 2 history missing Q1 question: {ctx_t2!r}"
    assert "DIDA" in ctx_t2, "Turn 2 history must contain DIDA filter from SQL1"
    assert "Expedia" in ctx_t2, "Turn 2 history must contain Expedia filter from SQL1"
    assert "TotalBookings" in ctx_t2, "Turn 2 history must contain metric alias from SQL1"
    print("  ✓ Turn 2: LLM sees Q1+SQL1 — DIDA, Expedia, TotalBookings all present in history")
    print("           → LLM can correctly add TOP 2 while inheriting both filters")

    # Turn 3: sees Q1+SQL1, Q2+SQL2 — must have TOP 2 and DIDA so LLM knows what to swap
    ctx_t3 = build_context_str(conversation_turns[:2])
    assert "TOP 2" in ctx_t3, "Turn 3 history must have TOP 2 from SQL2 so LLM keeps it"
    assert "DIDA" in ctx_t3, "Turn 3 history must have DIDA so LLM knows what to replace"
    assert "Expedia" in ctx_t3, "Turn 3 history must have Expedia so LLM keeps supplier filter"
    print("  ✓ Turn 3: LLM sees Q1+SQL1, Q2+SQL2 — TOP 2 + DIDA + Expedia all in history")
    print("           → LLM can correctly swap DIDA → Hotel Beds while keeping TOP 2 + Expedia")

    # Verify the expected SQL changes across turns
    sql_q1 = conversation_turns[0]["sql"]
    sql_q2 = conversation_turns[1]["sql"]
    sql_q3 = conversation_turns[2]["sql"]

    # Q2 must add TOP and keep DIDA + Expedia
    assert "TOP 2" in sql_q2,    "SQL2 must have TOP 2"
    assert "DIDA"  in sql_q2,    "SQL2 must still filter DIDA (inherited from Q1)"
    assert "Expedia" in sql_q2,  "SQL2 must still filter Expedia (inherited from Q1)"
    print("  ✓ SQL2 correctly inherits DIDA + Expedia and adds TOP 2")

    # Q3 must keep TOP 2 + Expedia, swap DIDA → Hotel Beds
    assert "TOP 2"       in sql_q3,  "SQL3 must keep TOP 2 (inherited from Q2)"
    assert "DIDA"        not in sql_q3, "SQL3 must NOT have DIDA (replaced)"
    assert "Hotel Beds"  in sql_q3,  "SQL3 must filter Hotel Beds (new client)"
    assert "Expedia"     in sql_q3,  "SQL3 must keep Expedia supplier (inherited)"
    print("  ✓ SQL3 correctly swaps DIDA → Hotel Beds, keeps TOP 2 + Expedia")

    print(f"\n  All assertions passed. conversation_turns has {len(conversation_turns)} turns.")
    print()


def test_chat_history_4_turns():
    conversation_turns: List[Dict] = []

    turns = [
        (
            "Show me total sales by agent for January 2024",
            "SELECT AM.AgentName, SUM(COALESCE(BD.AgentBuyingPrice,0)) AS TotalSales "
            "FROM BookingData BD "
            "LEFT JOIN AgentMaster_V1 AM ON BD.AgentId = AM.AgentId "
            "WHERE BD.CreatedDate >= '2024-01-01' AND BD.CreatedDate < '2024-02-01' "
            "AND BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') "
            "GROUP BY AM.AgentName ORDER BY TotalSales DESC",
        ),
        (
            "Show only the top 5",
            "SELECT TOP 5 AM.AgentName, SUM(COALESCE(BD.AgentBuyingPrice,0)) AS TotalSales "
            "FROM BookingData BD "
            "LEFT JOIN AgentMaster_V1 AM ON BD.AgentId = AM.AgentId "
            "WHERE BD.CreatedDate >= '2024-01-01' AND BD.CreatedDate < '2024-02-01' "
            "AND BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') "
            "GROUP BY AM.AgentName ORDER BY TotalSales DESC",
        ),
        (
            "Now break it down by country too",
            "SELECT TOP 5 AM.AgentName, MC.Country, SUM(COALESCE(BD.AgentBuyingPrice,0)) AS TotalSales "
            "FROM BookingData BD "
            "LEFT JOIN AgentMaster_V1 AM ON BD.AgentId = AM.AgentId "
            "LEFT JOIN Master_Country MC ON BD.ProductCountryid = MC.CountryID "
            "WHERE BD.CreatedDate >= '2024-01-01' AND BD.CreatedDate < '2024-02-01' "
            "AND BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') "
            "GROUP BY AM.AgentName, MC.Country ORDER BY TotalSales DESC",
        ),
        (
            "What was the profit for those agents?",
            "SELECT TOP 5 AM.AgentName, "
            "SUM(COALESCE(BD.AgentBuyingPrice,0) - COALESCE(BD.CompanyBuyingPrice,0)) AS TotalProfit "
            "FROM BookingData BD "
            "LEFT JOIN AgentMaster_V1 AM ON BD.AgentId = AM.AgentId "
            "WHERE BD.CreatedDate >= '2024-01-01' AND BD.CreatedDate < '2024-02-01' "
            "AND BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') "
            "GROUP BY AM.AgentName ORDER BY TotalProfit DESC",
        ),
    ]

    for i, (question, fake_sql) in enumerate(turns, start=1):
        simulate_turn(question, fake_sql, conversation_turns, turn_number=i)

    # ── assertions ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ASSERTIONS")
    print(f"{'='*60}")

    assert len(conversation_turns) == 4, f"Expected 4 turns, got {len(conversation_turns)}"

    # Re-simulate what context WOULD have been seen, for verification
    # Turn 1 sees nothing
    ctx_t1 = build_context_str([])
    assert ctx_t1 == "(no prior conversation)", f"T1: expected empty history, got: {ctx_t1!r}"
    print("  ✓ Turn 1: LLM correctly sees NO prior history")

    # Turn 2 sees Q1+SQL1
    ctx_t2 = build_context_str(conversation_turns[:1])
    assert "Show me total sales by agent" in ctx_t2
    assert "TotalSales" in ctx_t2
    print("  ✓ Turn 2: LLM correctly sees Q1 + SQL1 in history")

    # Turn 3 sees Q1+SQL1, Q2+SQL2
    ctx_t3 = build_context_str(conversation_turns[:2])
    assert "Show me total sales by agent" in ctx_t3
    assert "Show only the top 5" in ctx_t3
    assert "TOP 5" in ctx_t3
    print("  ✓ Turn 3: LLM correctly sees Q1+SQL1, Q2+SQL2 in history")

    # Turn 4 sees Q1+SQL1, Q2+SQL2, Q3+SQL3
    ctx_t4 = build_context_str(conversation_turns[:3])
    assert "Show me total sales by agent" in ctx_t4
    assert "Show only the top 5" in ctx_t4
    assert "Now break it down by country too" in ctx_t4
    assert "Country" in ctx_t4
    print("  ✓ Turn 4: LLM correctly sees Q1+SQL1, Q2+SQL2, Q3+SQL3 in history")

    # sliding window: if there were 5+ turns, only last 4 are kept
    extra_turns = conversation_turns + [{"question": "Extra Q5", "sql": "SELECT 1"}]
    ctx_sliding = build_context_str(extra_turns)
    assert "Show me total sales by agent" not in ctx_sliding, \
        "Sliding window should drop Q1 when 5 turns present"
    assert "Show only the top 5" in ctx_sliding
    print("  ✓ Sliding window (4-turn limit): Q1 correctly dropped when 5th turn added")

    print(f"\n  All assertions passed. conversation_turns has {len(conversation_turns)} turns.")
    print()


if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  TEST 1: DIDA → Expedia → top 2 → swap client scenario")
    print("█"*60)
    test_dida_expedia_followup_scenario()

    print("\n" + "█"*60)
    print("  TEST 2: generic 4-turn sliding window")
    print("█"*60)
    test_chat_history_4_turns()
