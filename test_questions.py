"""
Test questions for the Text-to-SQL pipeline.
Covers simple_llm and complex_llm routing categories.

Routing rules (from _estimate_query_complexity):
  simple_llm  — single-dimension aggregation, ≤12 words, no complex keywords
  complex_llm — growth/trend/YoY/comparison, or 2+ dimension words, or >12 words
"""

TEST_QUESTIONS = [
    # ─── simple_llm (12 questions) ───────────────────────────────────────────
    # Single-dimension, short, no comparison keywords
    {
        "question": "What is total revenue this month?",
        "expected_complexity": "simple_llm",
        "notes": "Single metric, current month, ≤6 words",
    },
    {
        "question": "How many bookings were made today?",
        "expected_complexity": "simple_llm",
        "notes": "Count + single time filter",
    },
    {
        "question": "What is total profit this month?",
        "expected_complexity": "simple_llm",
        "notes": "Single metric, no dimension breakdown",
    },
    {
        "question": "How many agents are active?",
        "expected_complexity": "simple_llm",
        "notes": "Count on single dimension",
    },
    {
        "question": "Show me total bookings by country this month",
        "expected_complexity": "simple_llm",
        "notes": "Single dimension (country), simple aggregation",
    },
    {
        "question": "What is the average booking value this month?",
        "expected_complexity": "simple_llm",
        "notes": "Single metric, no joins required",
    },
    {
        "question": "How many hotels do we have?",
        "expected_complexity": "simple_llm",
        "notes": "Simple count, single table",
    },
    {
        "question": "Show total revenue by agent this month",
        "expected_complexity": "simple_llm",
        "notes": "Single dimension (agent), ≤8 words",
    },
    {
        "question": "What is total sales for last month?",
        "expected_complexity": "simple_llm",
        "notes": "Single metric, last month filter",
    },
    {
        "question": "How many bookings were cancelled this month?",
        "expected_complexity": "simple_llm",
        "notes": "Count with status filter, single dimension",
    },
    {
        "question": "Show me revenue by country for last month",
        "expected_complexity": "simple_llm",
        "notes": "Single dimension (country), simple aggregation, no TopN",
    },
    {
        "question": "What is the total cost this month?",
        "expected_complexity": "simple_llm",
        "notes": "Single metric, no breakdown",
    },

    # ─── complex_llm (13 questions) ──────────────────────────────────────────
    # Growth/trend/YoY keywords, or 2+ dimension words, or >12 words
    {
        "question": "What is the revenue growth compared to last month?",
        "expected_complexity": "complex_llm",
        "notes": "Keyword: 'growth', 'compared' → complex",
    },
    {
        "question": "Show me year over year revenue trend by country",
        "expected_complexity": "complex_llm",
        "notes": "Keyword: 'year over year' → complex",
    },
    {
        "question": "How did bookings change from last year to this year?",
        "expected_complexity": "complex_llm",
        "notes": "Keyword: 'change', 'last year' → complex",
    },
    {
        "question": "What is the YTD revenue versus the previous year?",
        "expected_complexity": "complex_llm",
        "notes": "Keyword: 'ytd', 'versus' → complex",
    },
    {
        "question": "Show revenue and profit trend month over month for 2025",
        "expected_complexity": "complex_llm",
        "notes": "Keyword: 'trend', 'month over month' → complex",
    },
    {
        "question": "Show me revenue and profit by agent and country this month",
        "expected_complexity": "complex_llm",
        "notes": "2 dimensions: agent + country → complex",
    },
    {
        "question": "Show me bookings by agent and supplier for this month",
        "expected_complexity": "complex_llm",
        "notes": "2 dimensions: agent + supplier → complex",
    },
    {
        "question": "Compare revenue by hotel chain and country for last 3 months",
        "expected_complexity": "complex_llm",
        "notes": "Keyword: 'compared'; 2 dims: chain + country → complex",
    },
    {
        "question": "What is the correlation between booking volume and profit by city?",
        "expected_complexity": "complex_llm",
        "notes": "Keyword: 'correlation' → complex",
    },
    {
        "question": "Show me revenue breakdown by agent, country, and hotel for the last 3 months",
        "expected_complexity": "complex_llm",
        "notes": ">12 words, multi-dim: agent + country + hotel → complex",
    },
    {
        "question": "Which nationalities had the biggest increase in bookings this year?",
        "expected_complexity": "complex_llm",
        "notes": "Keyword: 'increase' → complex",
    },
    {
        "question": "Show me monthly revenue and profit for each supplier over the last 6 months",
        "expected_complexity": "complex_llm",
        "notes": ">12 words, time series grouping → complex",
    },
    {
        "question": "What percentage of total revenue came from each country and city combination last quarter?",
        "expected_complexity": "complex_llm",
        "notes": ">12 words, 2 dims: country + city, percentage calc → complex",
    },
]


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from backend.services.sql_engine import _estimate_query_complexity

    simple_count  = sum(1 for q in TEST_QUESTIONS if q["expected_complexity"] == "simple_llm")
    complex_count = sum(1 for q in TEST_QUESTIONS if q["expected_complexity"] == "complex_llm")
    print(f"Total: {len(TEST_QUESTIONS)} questions  |  simple_llm: {simple_count}  |  complex_llm: {complex_count}\n")
    print(f"{'#':<4} {'Complexity':<14} {'Match':<7} Question")
    print("-" * 90)

    mismatches = 0
    for i, q in enumerate(TEST_QUESTIONS, 1):
        actual = _estimate_query_complexity(q["question"])
        expected = q["expected_complexity"]
        match = "✓" if actual == expected else "✗"
        if actual != expected:
            mismatches += 1
        print(f"{i:<4} {actual:<14} {match:<7} {q['question']}")

    print("-" * 90)
    if mismatches == 0:
        print("All questions routed as expected.")
    else:
        print(f"{mismatches} question(s) routed differently than expected.")
