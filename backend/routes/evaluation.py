"""
Evaluation endpoints - Test SQL generation accuracy with ground truth data.
Admin-only access.
"""
import json
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel

from backend.services.session import Session
from backend.services.sql_engine import _clean_sql_response, fix_common_sql_errors
from backend.routes.deps import get_session
from app.evaluation import (
    SQLEvaluator,
    EvaluationCase,
    load_evaluation_cases,
    create_sample_evaluation_cases,
)
from app.db_utils import execute_query_safe

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

# Project root for finding default evaluation files
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _require_admin(session: Session):
    if session.user.get("role") != "Admin":
        raise HTTPException(status_code=403, detail="Admin access required for evaluation")


def _require_connected(session: Session):
    if not session.connected or session.db is None:
        raise HTTPException(status_code=400, detail="Not connected to database")


# --- Models ---

class EvalCaseInput(BaseModel):
    question: str
    ground_truth_sql: str
    category: str = "general"
    difficulty: str = "medium"
    description: str = ""


class RunEvalRequest(BaseModel):
    cases: List[EvalCaseInput]


class EvalResultItem(BaseModel):
    question: str
    category: str
    difficulty: str
    ground_truth_sql: str
    generated_sql: str
    execution_success: bool
    result_match: bool
    column_match: bool
    row_count_match: bool
    score: float
    ragas_score: Optional[float] = None
    ragas_faithfulness: Optional[float] = None
    ragas_answer_correctness: Optional[float] = None
    ragas_sql_validity: Optional[float] = None
    ragas_context_relevance: Optional[float] = None
    ragas_execution_accuracy: Optional[float] = None
    ragas_result_similarity: Optional[float] = None
    execution_time_ms: float = 0.0
    error_message: str = ""


class EvalSummary(BaseModel):
    total_cases: int
    average_score: float
    execution_success_rate: float
    result_match_rate: float
    avg_execution_time_ms: float
    by_category: Dict[str, Any] = {}
    by_difficulty: Dict[str, Any] = {}
    ragas_metrics: Optional[Dict[str, float]] = None
    failed_cases: List[Dict[str, Any]] = []


class RunEvalResponse(BaseModel):
    results: List[EvalResultItem]
    summary: EvalSummary


# --- Endpoints ---

@router.get("/files")
def list_eval_files(session: Session = Depends(get_session)):
    """List available evaluation JSON files in the project."""
    _require_admin(session)

    files = []
    for pattern in ["evaluation_cases*.json", "text2sql_ground_truth*.json"]:
        for f in PROJECT_ROOT.glob(pattern):
            files.append({"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)})
    return {"files": files}


@router.get("/load/{filename}")
def load_eval_file(filename: str, session: Session = Depends(get_session)):
    """Load evaluation cases from a project file."""
    _require_admin(session)

    filepath = PROJECT_ROOT / filename
    if not filepath.exists() or not filepath.suffix == ".json":
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    cases = load_evaluation_cases(str(filepath))
    return {
        "cases": [
            {
                "question": c.question,
                "ground_truth_sql": c.ground_truth_sql,
                "category": c.category,
                "difficulty": c.difficulty,
                "description": c.description,
            }
            for c in cases
        ],
        "count": len(cases),
    }


@router.get("/sample-cases")
def get_sample_cases(session: Session = Depends(get_session)):
    """Get built-in sample evaluation cases."""
    _require_admin(session)

    cases = create_sample_evaluation_cases()
    return {
        "cases": [
            {
                "question": c.question,
                "ground_truth_sql": c.ground_truth_sql,
                "category": c.category,
                "difficulty": c.difficulty,
                "description": c.description,
            }
            for c in cases
        ],
        "count": len(cases),
    }


@router.post("/upload")
async def upload_eval_file(file: UploadFile = File(...), session: Session = Depends(get_session)):
    """Upload a JSON file with evaluation cases."""
    _require_admin(session)

    content = await file.read()
    try:
        data = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    cases_data = data.get("cases", data) if isinstance(data, dict) else data
    if not isinstance(cases_data, list):
        raise HTTPException(status_code=400, detail="Expected a list of cases")

    cases = []
    for case in cases_data:
        if "question" not in case or "ground_truth_sql" not in case:
            raise HTTPException(status_code=400, detail="Each case needs 'question' and 'ground_truth_sql'")
        cases.append({
            "question": case["question"],
            "ground_truth_sql": case["ground_truth_sql"],
            "category": case.get("category", "general"),
            "difficulty": case.get("difficulty", "medium"),
            "description": case.get("description", ""),
        })

    return {"cases": cases, "count": len(cases)}


@router.post("/run", response_model=RunEvalResponse)
def run_evaluation(req: RunEvalRequest, session: Session = Depends(get_session)):
    """Run evaluation on provided test cases."""
    _require_admin(session)
    _require_connected(session)

    if not req.cases:
        raise HTTPException(status_code=400, detail="No test cases provided")

    def execute_query(query: str):
        timeout = getattr(session.db_config, "query_timeout", 30)
        return execute_query_safe(session.db, query, timeout_seconds=timeout)

    evaluator = SQLEvaluator(session.db, execute_fn=execute_query)

    results = []
    for case_input in req.cases:
        eval_case = EvaluationCase(
            question=case_input.question,
            ground_truth_sql=case_input.ground_truth_sql,
            category=case_input.category,
            difficulty=case_input.difficulty,
            description=case_input.description,
        )

        # Generate SQL
        generated_sql = _generate_sql(session, case_input.question)

        # Evaluate
        result = evaluator.evaluate_case(eval_case, generated_sql)
        results.append(result)

    evaluator.results = results
    summary_data = evaluator.get_summary()

    # Convert to response models
    result_items = []
    for r in results:
        item = EvalResultItem(
            question=r.question,
            category=r.category,
            difficulty=r.difficulty,
            ground_truth_sql=r.ground_truth_sql,
            generated_sql=r.generated_sql,
            execution_success=r.execution_success,
            result_match=r.result_match,
            column_match=r.column_match,
            row_count_match=r.row_count_match,
            score=r.score(),
            execution_time_ms=r.execution_time_ms,
            error_message=r.error_message,
        )
        if r.ragas_metrics:
            item.ragas_score = r.ragas_metrics.overall_score()
            item.ragas_faithfulness = r.ragas_metrics.faithfulness
            item.ragas_answer_correctness = r.ragas_metrics.answer_correctness
            item.ragas_sql_validity = r.ragas_metrics.sql_validity
            item.ragas_context_relevance = r.ragas_metrics.context_relevance
            item.ragas_execution_accuracy = r.ragas_metrics.execution_accuracy
            item.ragas_result_similarity = r.ragas_metrics.result_similarity
        result_items.append(item)

    summary = EvalSummary(
        total_cases=summary_data.get("total_cases", len(results)),
        average_score=summary_data.get("average_score", 0),
        execution_success_rate=summary_data.get("execution_success_rate", 0),
        result_match_rate=summary_data.get("result_match_rate", 0),
        avg_execution_time_ms=summary_data.get("avg_execution_time_ms", 0),
        by_category=summary_data.get("by_category", {}),
        by_difficulty=summary_data.get("by_difficulty", {}),
        ragas_metrics=summary_data.get("ragas_metrics"),
        failed_cases=summary_data.get("failed_cases", []),
    )

    return RunEvalResponse(results=result_items, summary=summary)


def _generate_sql(session: Session, question: str) -> str:
    """Generate SQL using the session's configured chain."""
    try:
        resp = session.sql_chain.invoke({"question": question, "context": ""})
        cleaned = _clean_sql_response(resp.strip())
        cleaned = fix_common_sql_errors(cleaned)
        return cleaned
    except Exception as e:
        return f"ERROR: {str(e)}"
