"""
Evaluation Page - Test SQL generation accuracy with ground truth data
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation import (
    SQLEvaluator,
    EvaluationCase,
    load_evaluation_cases,
    create_sample_evaluation_cases
)
from app.db_utils import execute_query_safe

st.set_page_config(
    page_title="SQL Evaluation",
    page_icon="📊",
    layout="wide"
)

# ============================================
# AUTHENTICATION CHECK (shared with main app)
# ============================================

# Check if user is authenticated
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login first from the main page.")
    st.info("Go to the main page (Text to SQL RAG) to sign in.")
    st.stop()

# Check if user is Admin (evaluation is admin-only)
user = st.session_state.get("user", {})
if user.get("role") != "Admin":
    st.error("Access Denied: This page is only available to Admin users.")
    st.info(f"You are logged in as: {user.get('name', 'Unknown')} ({user.get('role', 'Unknown')})")
    st.stop()

# Show user info
st.sidebar.markdown(f"**Logged in as:** {user.get('name', 'Unknown')}")
st.sidebar.markdown(f"**Role:** {user.get('role', 'Unknown')}")
st.sidebar.divider()

st.title("📊 SQL Generation Evaluation")
st.markdown("Evaluate the accuracy of generated SQL queries against ground truth data")

# Check if connected
if 'db' not in st.session_state or st.session_state.db is None:
    st.warning("Please connect to the database first from the main page.")
    st.stop()

if 'sql_chain' not in st.session_state or st.session_state.sql_chain is None:
    st.warning("Please initialize the SQL chain from the main page.")
    st.stop()


# Initialize evaluation state
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'evaluation_cases' not in st.session_state:
    st.session_state.evaluation_cases = []


def generate_sql(question: str) -> str:
    """Generate SQL using the configured chain"""
    try:
        resp = st.session_state.sql_chain.invoke({
            "question": question,
            "context": ""
        })
        cleaned = resp.strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]
        if cleaned.startswith("```sql"):
            cleaned = cleaned[6:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


def execute_query(query: str):
    """Execute query and return results"""
    timeout = getattr(st.session_state.get("db_config"), "query_timeout", 30)
    return execute_query_safe(st.session_state.db, query, timeout_seconds=timeout)


# Sidebar options
with st.sidebar:
    st.header("Evaluation Options")

    data_source = st.radio(
        "Test Cases Source",
        ["Load from file", "Use sample cases", "Add custom case"]
    )

    if data_source == "Load from file":
        uploaded_file = st.file_uploader(
            "Upload evaluation cases (JSON)",
            type=["json"],
            key="eval_file_uploader"
        )

        # Process uploaded file
        if uploaded_file is not None:
            if st.button("📂 Load uploaded file"):
                try:
                    # Read the uploaded file content
                    file_content = uploaded_file.read().decode("utf-8")
                    data = json.loads(file_content)

                    # Parse cases - support both {cases: [...]} and [...] formats
                    cases_data = data.get("cases", data) if isinstance(data, dict) else data

                    loaded_cases = [
                        EvaluationCase(
                            question=case["question"],
                            ground_truth_sql=case["ground_truth_sql"],
                            category=case.get("category", "general"),
                            difficulty=case.get("difficulty", "medium"),
                            description=case.get("description", ""),
                        )
                        for case in cases_data
                    ]

                    st.session_state.evaluation_cases = loaded_cases
                    st.success(f"Loaded {len(loaded_cases)} cases from uploaded file")
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {e}")
                except KeyError as e:
                    st.error(f"Missing required field: {e}")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        st.divider()

        # Check for default file
        default_file = Path(__file__).parent.parent / "evaluation_cases.json"
        if default_file.exists():
            if st.button("📁 Load default cases (evaluation_cases.json)"):
                try:
                    st.session_state.evaluation_cases = load_evaluation_cases(str(default_file))
                    st.success(f"Loaded {len(st.session_state.evaluation_cases)} cases")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading: {e}")

    elif data_source == "Use sample cases":
        if st.button("Load sample cases"):
            st.session_state.evaluation_cases = create_sample_evaluation_cases()
            st.success(f"Loaded {len(st.session_state.evaluation_cases)} sample cases")

    st.divider()
    st.metric("Loaded Cases", len(st.session_state.evaluation_cases))


# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Test Cases")

    if data_source == "Add custom case":
        with st.form("add_case"):
            question = st.text_input("Question")
            ground_truth_sql = st.text_area("Ground Truth SQL")
            category = st.selectbox(
                "Category",
                ["aggregation", "filter", "date", "join", "text_search", "general"]
            )
            difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])

            if st.form_submit_button("Add Case"):
                if question and ground_truth_sql:
                    new_case = EvaluationCase(
                        question=question,
                        ground_truth_sql=ground_truth_sql,
                        category=category,
                        difficulty=difficulty
                    )
                    st.session_state.evaluation_cases.append(new_case)
                    st.success("Case added!")
                    st.rerun()

    # Display loaded cases
    if st.session_state.evaluation_cases:
        cases_df = pd.DataFrame([
            {
                "Question": c.question[:60] + "..." if len(c.question) > 60 else c.question,
                "Category": c.category,
                "Difficulty": c.difficulty,
            }
            for c in st.session_state.evaluation_cases
        ])
        st.dataframe(cases_df, use_container_width=True)

        # Run evaluation button
        if st.button("🚀 Run Evaluation", type="primary"):
            evaluator = SQLEvaluator(st.session_state.db, execute_fn=execute_query)

            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []
            total = len(st.session_state.evaluation_cases)

            for i, case in enumerate(st.session_state.evaluation_cases):
                status_text.text(f"Evaluating: {case.question[:50]}...")

                generated_sql = generate_sql(case.question)
                result = evaluator.evaluate_case(case, generated_sql)
                results.append(result)

                progress_bar.progress((i + 1) / total)

            evaluator.results = results
            st.session_state.evaluation_results = evaluator.get_summary()
            st.session_state.evaluation_detailed = results

            status_text.text("Evaluation complete!")
            st.success(f"Evaluated {total} cases")
            st.rerun()

with col2:
    st.subheader("Quick Actions")

    if st.session_state.evaluation_cases:
        if st.button("Clear all cases"):
            st.session_state.evaluation_cases = []
            st.session_state.evaluation_results = None
            st.rerun()

        if st.session_state.evaluation_results:
            # Export results
            results_json = json.dumps({
                "timestamp": datetime.now().isoformat(),
                "summary": st.session_state.evaluation_results
            }, indent=2, default=str)

            st.download_button(
                "📥 Download Results",
                results_json,
                file_name=f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Results section
if st.session_state.evaluation_results:
    st.divider()
    st.subheader("Evaluation Results")

    summary = st.session_state.evaluation_results

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall Score",
            f"{summary['average_score']:.1%}",
            help="Weighted average of all metrics"
        )

    with col2:
        st.metric(
            "Execution Success",
            f"{summary['execution_success_rate']:.1%}",
            help="Queries that executed without errors"
        )

    with col3:
        st.metric(
            "Result Match",
            f"{summary['result_match_rate']:.1%}",
            help="Results matching ground truth"
        )

    with col4:
        st.metric(
            "Avg Execution Time",
            f"{summary['avg_execution_time_ms']:.0f}ms"
        )

    # RAGAS Metrics Section
    if summary.get("ragas_metrics"):
        st.divider()
        st.markdown("### 🎯 RAGAS Metrics")
        st.caption("RAGAS-style metrics adapted for Text-to-SQL evaluation")

        ragas = summary["ragas_metrics"]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "RAGAS Score",
                f"{ragas.get('avg_ragas_score', 0):.1%}",
                help="Overall RAGAS composite score"
            )
        with col2:
            st.metric(
                "Faithfulness",
                f"{ragas.get('avg_faithfulness', 0):.1%}",
                help="Does SQL match question intent?"
            )
        with col3:
            st.metric(
                "Answer Correctness",
                f"{ragas.get('avg_answer_correctness', 0):.1%}",
                help="Do results match ground truth?"
            )
        with col4:
            st.metric(
                "SQL Validity",
                f"{ragas.get('avg_sql_validity', 0):.1%}",
                help="Syntactically correct SQL?"
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Context Relevance",
                f"{ragas.get('avg_context_relevance', 0):.1%}",
                help="Is schema used correctly?"
            )
        with col2:
            st.metric(
                "Execution Accuracy",
                f"{ragas.get('avg_execution_accuracy', 0):.1%}",
                help="Query runs without error?"
            )
        with col3:
            st.metric(
                "Result Similarity",
                f"{ragas.get('avg_result_similarity', 0):.1%}",
                help="Partial match score"
            )

    st.divider()

    # Detailed metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**By Category**")
        if summary.get("by_category"):
            cat_df = pd.DataFrame([
                {
                    "Category": cat,
                    "Count": data["count"],
                    "Success Rate": f"{data['execution_success_rate']:.1%}",
                    "Match Rate": f"{data['result_match_rate']:.1%}",
                    "Score": f"{data['average_score']:.1%}"
                }
                for cat, data in summary["by_category"].items()
            ])
            st.dataframe(cat_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**By Difficulty**")
        if summary.get("by_difficulty"):
            diff_df = pd.DataFrame([
                {
                    "Difficulty": diff,
                    "Count": data["count"],
                    "Success Rate": f"{data['execution_success_rate']:.1%}",
                    "Match Rate": f"{data['result_match_rate']:.1%}",
                    "Score": f"{data['average_score']:.1%}"
                }
                for diff, data in summary["by_difficulty"].items()
            ])
            st.dataframe(diff_df, use_container_width=True, hide_index=True)

    # Failed cases
    if summary.get("failed_cases"):
        st.markdown("**Failed Cases**")
        with st.expander(f"Show {len(summary['failed_cases'])} failed cases"):
            for i, failed in enumerate(summary["failed_cases"]):
                st.markdown(f"**{i+1}. {failed['question']}**")
                st.code(failed.get("generated_sql", "N/A"), language="sql")
                if failed.get("error"):
                    st.error(failed["error"])
                st.divider()

    # Detailed results
    if hasattr(st.session_state, 'evaluation_detailed'):
        st.markdown("**Detailed Results**")

        detailed_df = pd.DataFrame([
            {
                "Question": r.question[:50] + "..." if len(r.question) > 50 else r.question,
                "Category": r.category,
                "Exec ✓": "✅" if r.execution_success else "❌",
                "Match ✓": "✅" if r.result_match else "❌",
                "Score": f"{r.score():.1%}",
                "RAGAS": f"{r.ragas_metrics.overall_score():.1%}" if r.ragas_metrics else "N/A",
                "Time (ms)": f"{r.execution_time_ms:.0f}",
            }
            for r in st.session_state.evaluation_detailed
        ])

        st.dataframe(detailed_df, use_container_width=True, hide_index=True)

        # Compare specific result
        st.markdown("**🔍 Compare Query Results**")
        selected_idx = st.selectbox(
            "Select a case to compare",
            range(len(st.session_state.evaluation_detailed)),
            format_func=lambda i: st.session_state.evaluation_detailed[i].question[:60]
        )

        if selected_idx is not None:
            result = st.session_state.evaluation_detailed[selected_idx]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Ground Truth SQL**")
                st.code(result.ground_truth_sql, language="sql")

            with col2:
                st.markdown("**Generated SQL**")
                st.code(result.generated_sql, language="sql")

            if result.error_message:
                st.error(f"Error: {result.error_message}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Ground Truth Result**")
                if result.ground_truth_result is not None:
                    st.dataframe(result.ground_truth_result.head(10), use_container_width=True)
                else:
                    st.info("No result")

            with col2:
                st.markdown("**Generated Result**")
                if result.generated_result is not None:
                    st.dataframe(result.generated_result.head(10), use_container_width=True)
                else:
                    st.info("No result")

            # Show RAGAS metrics for this case
            if result.ragas_metrics:
                st.markdown("**🎯 RAGAS Metrics for this case**")
                ragas = result.ragas_metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RAGAS Score", f"{ragas.overall_score():.1%}")
                with col2:
                    st.metric("Faithfulness", f"{ragas.faithfulness:.1%}")
                with col3:
                    st.metric("Answer Correctness", f"{ragas.answer_correctness:.1%}")
                with col4:
                    st.metric("SQL Validity", f"{ragas.sql_validity:.1%}")
