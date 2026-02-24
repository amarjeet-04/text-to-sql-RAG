"""
In-memory session management for the FastAPI backend.
Each session stores user info, DB connection, LLM, RAG engine, chat history, and query cache.
"""
import uuid
import hashlib
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


USERS_DB = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "Admin",
        "name": "Administrator",
    },
    "analyst": {
        "password": hashlib.sha256("analyst123".encode()).hexdigest(),
        "role": "Analyst",
        "name": "User",
    },
}


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_user(username: str, password: str) -> Optional[Dict[str, str]]:
    if username in USERS_DB:
        if USERS_DB[username]["password"] == hash_password(password):
            return {
                "username": username,
                "role": USERS_DB[username]["role"],
                "name": USERS_DB[username]["name"],
            }
    return None


@dataclass
class ConversationTurn:
    """Structured record of a single conversation turn for better context tracking."""
    question: str
    sql: Optional[str] = None
    topic: str = "unknown"  # e.g. "agent:revenue", "country:bookings"
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    status: str = "ok"  # "ok", "error", "no_rows", "cache", "conversation"
    timestamp: float = field(default_factory=time.time)


def serialize_conversation_turns(turns: List['ConversationTurn'], max_turns: int = 5) -> str:
    """Serialize recent conversation turns into a structured context string for the LLM.

    Includes the previous SQL so the LLM can build follow-up queries correctly
    (e.g. "filter by UAE", "show only last week", "sort by cost").
    """
    if not turns:
        return "No prior conversation."
    recent = turns[-max_turns:]
    parts = []
    for t in recent:
        cols_str = ", ".join(t.columns[:8]) if t.columns else "N/A"
        # Include SQL snippet so the LLM sees what was previously queried
        sql_line = ""
        if t.sql:
            sql_snippet = t.sql[:500].strip()
            if len(t.sql) > 500:
                sql_snippet += "..."
            sql_line = f"\nSQL: {sql_snippet}"
        parts.append(
            f"Topic: {t.topic}\n"
            f"Q: {t.question}{sql_line}\n"
            f"Columns: {cols_str}\n"
            f"Rows: {t.row_count} | Status: {t.status}"
        )
    return "\n---\n".join(parts)


@dataclass
class Session:
    token: str
    user: Dict[str, str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Database state
    db: Any = None
    db_config: Any = None
    sql_chain: Any = None
    llm: Any = None
    reasoning_llm: Any = None  # DeepSeek reasoner for complex queries
    rag_engine: Any = None
    embedder: Any = None
    cached_schema_text: str = ""
    connected: bool = False
    sql_dialect: str = "sqlserver"
    enable_nolock: bool = False

    # Chat state
    chat_history: List[Dict] = field(default_factory=list)
    conversation_context: str = ""
    conversation_turns: List[ConversationTurn] = field(default_factory=list)
    query_cache: List = field(default_factory=list)


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create(self, user: Dict[str, str]) -> Session:
        token = str(uuid.uuid4())
        session = Session(token=token, user=user)
        self._sessions[token] = session
        return session

    def get(self, token: str) -> Optional[Session]:
        return self._sessions.get(token)

    def delete(self, token: str):
        self._sessions.pop(token, None)


# Global session store
store = SessionStore()
