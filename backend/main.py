"""
FastAPI backend for Text-to-SQL RAG chatbot.
Run with: uvicorn backend.main:app --reload --port 8000
"""
import sys
import os
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.auth import router as auth_router
from backend.routes.database import router as db_router
from backend.routes.chat import router as chat_router
from backend.routes.evaluation import router as eval_router
from backend.services.runtime import set_request_id, clear_context, shutdown_shared_executor

app = FastAPI(title="Within Earth Chatbot API", version="1.0.0")
logger = logging.getLogger("backend")

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@app.on_event("startup")
def preload_embedder():
    """Pre-load the sentence-transformers model at startup so /connect is fast."""
    should_preload = os.getenv("PRELOAD_EMBEDDER", "false").lower() in {"1", "true", "yes", "on"}
    if not should_preload:
        return
    from app.rag.rag_engine import RAGEngine
    try:
        RAGEngine()
    except Exception:
        # Keep API startup resilient if embedding dependencies are unavailable.
        pass


@app.on_event("shutdown")
def shutdown_workers():
    shutdown_shared_executor(wait=False)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = set_request_id(request.headers.get("x-request-id") or str(uuid.uuid4()))
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed request_id=%s path=%s", request_id, request.url.path)
        raise
    finally:
        clear_context()
    response.headers["x-request-id"] = request_id
    return response

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
        "http://localhost:5175", "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(db_router)
app.include_router(chat_router)
app.include_router(eval_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
