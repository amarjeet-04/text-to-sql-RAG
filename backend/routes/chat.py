from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime

import pandas as pd

from backend.services.session import Session
from backend.services.sql_engine import handle_query, generate_nl_response
from backend.routes.deps import get_session

router = APIRouter(prefix="/api/chat", tags=["chat"])


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    intent: Optional[str] = None
    nl_answer: Optional[str] = None
    sql: Optional[str] = None
    results: Optional[List[Any]] = None
    row_count: int = 0
    from_cache: bool = False
    error: Optional[str] = None
    nl_pending: bool = False


class NLRequest(BaseModel):
    question: str
    results: List[Any]


class NLResponse(BaseModel):
    nl_answer: str


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, session: Session = Depends(get_session)):
    if not session.connected or session.db is None:
        raise HTTPException(status_code=400, detail="Not connected to database. Please connect first.")

    result = handle_query(
        question=req.question,
        db=session.db,
        db_config=session.db_config,
        sql_chain=session.sql_chain,
        llm=session.llm,
        rag_engine=session.rag_engine,
        embedder=session.embedder,
        chat_history=session.chat_history,
        conversation_context=session.conversation_context,
        query_cache=session.query_cache,
        cached_schema_text=session.cached_schema_text,
        conversation_turns=session.conversation_turns,
        reasoning_llm=session.reasoning_llm,
    )

    # Update session state
    session.conversation_context = result.get("updated_context", session.conversation_context)
    # Store structured conversation turn if returned
    turn = result.get("conversation_turn")
    if turn is not None:
        session.conversation_turns.append(turn)

    # Build chat history entry
    chat_entry = {
        "question": req.question,
        "sql": result.get("sql"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "from_cache": result.get("from_cache", False),
        "nl_answer": result.get("nl_answer"),
        "intent": result.get("intent"),
        "error": result.get("error"),
    }

    # Store result_df for follow-up detection
    if result.get("results") and len(result["results"]) > 0:
        chat_entry["result_df"] = pd.DataFrame(result["results"])
    else:
        chat_entry["result_df"] = None

    session.chat_history.append(chat_entry)

    return QueryResponse(
        intent=result.get("intent"),
        nl_answer=result.get("nl_answer"),
        sql=result.get("sql"),
        results=result.get("results"),
        row_count=result.get("row_count", 0),
        from_cache=result.get("from_cache", False),
        error=result.get("error"),
        nl_pending=result.get("nl_pending", False),
    )


@router.post("/nl-response", response_model=NLResponse)
async def nl_response(req: NLRequest, session: Session = Depends(get_session)):
    """Generate NL summary for already-returned results.

    Called by the frontend AFTER displaying the data table, so the user
    sees results instantly and the NL summary streams in afterwards.
    Runs the LLM call in a thread executor to avoid blocking the event loop.
    """
    import asyncio

    if not session.connected or session.llm is None:
        raise HTTPException(status_code=400, detail="Not connected.")

    df = pd.DataFrame(req.results) if req.results else None
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None, generate_nl_response, req.question, df, session.llm
    )

    # Update the last chat history entry with the NL answer
    if session.chat_history:
        session.chat_history[-1]["nl_answer"] = answer

    return NLResponse(nl_answer=answer or "No summary available.")


@router.post("/clear")
def clear_chat(session: Session = Depends(get_session)):
    session.chat_history = []
    session.conversation_context = ""
    session.conversation_turns = []
    return {"success": True}


@router.post("/clear-cache")
def clear_cache(session: Session = Depends(get_session)):
    if session.user.get("role") != "Admin":
        raise HTTPException(status_code=403, detail="Only admins can clear cache")
    session.query_cache = []
    return {"success": True}
