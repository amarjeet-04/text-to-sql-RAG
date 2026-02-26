"""
### MINIMAL CORE
In-memory session management.
Holds DB connection, LLM handles, structured conversation state, and query cache.
"""
import os
import uuid
import hashlib
import time
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
_users_lock = threading.RLock()

USERS_DB: Dict[str, Dict[str, str]] = {
    "admin":   {"password": hashlib.sha256(b"admin123").hexdigest(),   "role": "Admin",   "name": "Administrator"},
    "analyst": {"password": hashlib.sha256(b"analyst123").hexdigest(), "role": "Analyst", "name": "User"},
}


def verify_user(username: str, password: str) -> Optional[Dict[str, str]]:
    with _users_lock:
        row = USERS_DB.get(username)
    if row and row["password"] == hashlib.sha256(password.encode()).hexdigest():
        return {"username": username, "role": row["role"], "name": row["name"]}
    return None


def register_user(username: str, password: str, role: str = "Analyst", name: str = "") -> Dict[str, str]:
    """Register a new user. Raises ValueError if username is already taken."""
    with _users_lock:
        if username in USERS_DB:
            raise ValueError("Username already taken")
        USERS_DB[username] = {
            "password": hashlib.sha256(password.encode()).hexdigest(),
            "role": role,
            "name": name or username,
        }
    return {"username": username, "role": role, "name": name or username}


def list_users() -> List[Dict[str, str]]:
    """Return all users (without passwords)."""
    with _users_lock:
        return [
            {"username": u, "role": d["role"], "name": d["name"]}
            for u, d in USERS_DB.items()
        ]


def delete_user(username: str) -> None:
    """Delete a user. Raises ValueError if user not found or is the last admin."""
    with _users_lock:
        if username not in USERS_DB:
            raise ValueError("User not found")
        # Prevent deleting the last Admin
        admins = [u for u, d in USERS_DB.items() if d["role"] == "Admin"]
        if USERS_DB[username]["role"] == "Admin" and len(admins) <= 1:
            raise ValueError("Cannot delete the last admin account")
        del USERS_DB[username]


# ---------------------------------------------------------------------------
# ### STRUCTURED MEMORY — ConversationTurn
# Compact per-turn record. SQL included so follow-up queries can reuse it.
# ---------------------------------------------------------------------------
@dataclass
class ConversationTurn:
    question:  str
    sql:       Optional[str] = None
    nl_answer: Optional[str] = None
    topic:     str           = "unknown"
    columns:   List[str]     = field(default_factory=list)
    row_count: int           = 0
    status:    str           = "ok"       # ok | error | no_rows | cache
    timestamp: float         = field(default_factory=time.time)


def serialize_conversation_turns(turns: List[ConversationTurn], max_turns: int = 3) -> str:
    """Compact context string passed into LLM prompt.
    Includes previous Q, answer, and SQL so the model has full context for follow-ups.
    """
    if not turns:
        return "No prior conversation."
    parts = []
    for t in turns[-max_turns:]:
        sql_line = ""
        if t.sql:
            snippet = t.sql[:500]
            sql_line = f"\nSQL: {snippet}{'...' if len(t.sql) > 500 else ''}"
        answer_line = ""
        if t.nl_answer:
            snippet = t.nl_answer[:300]
            answer_line = f"\nAnswer: {snippet}{'...' if len(t.nl_answer) > 300 else ''}"
        cols = ", ".join(t.columns[:8]) or "N/A"
        parts.append(
            f"Topic: {t.topic}\nQ: {t.question}{sql_line}{answer_line}\n"
            f"Columns: {cols}\nRows: {t.row_count} | Status: {t.status}"
        )
    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# ### STRUCTURED MEMORY — SessionState
# Replaces the old free-text conversation_context string.
# Extracted from each successful SQL result; used for follow-up context.
# ---------------------------------------------------------------------------
@dataclass
class SessionState:
    """Compact structured memory used for follow-up SQL handling."""
    dialect:          str                   = "sqlserver"
    last_sql:         Optional[str]         = None
    last_table:       Optional[str]         = None
    last_date_col:    Optional[str]         = None
    last_time_window: Dict[str, Optional[str]] = field(default_factory=lambda: {"start": None, "end": None})
    last_dimensions:  List[str]             = field(default_factory=list)
    last_metrics:     List[str]             = field(default_factory=list)
    last_filters:     List[str]             = field(default_factory=list)

    def compact_summary(self) -> str:
        """One-liner context for the LLM prompt; keeps prompt budget low."""
        tw = self.last_time_window or {}
        return (
            f"table={self.last_table or 'none'}; "
            f"date_col={self.last_date_col or 'none'}; "
            f"time_window={tw}; "
            f"dims={self.last_dimensions[:4]}; "
            f"metrics={self.last_metrics[:4]}; "
            f"filters={self.last_filters[:3]}"
        )

    def has_context(self) -> bool:
        return bool(self.last_sql)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dialect": self.dialect, "last_sql": self.last_sql,
            "last_table": self.last_table, "last_date_col": self.last_date_col,
            "last_time_window": dict(self.last_time_window or {}),
            "last_dimensions": list(self.last_dimensions),
            "last_metrics": list(self.last_metrics),
            "last_filters": list(self.last_filters),
        }


class QueryCacheStore:
    """Thread-safe per-session query cache storage."""

    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max(10, int(max_size or os.getenv("QUERY_CACHE_MAX_SIZE", os.getenv("QUERY_RESULT_CACHE_MAX_ENTRIES", "300"))))
        self._entries: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.RLock()

    def snapshot(self) -> List[Any]:
        with self._lock:
            return list(self._entries.values())

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def append(self, item: Any) -> None:
        key = str(item.get("cache_key")) if isinstance(item, dict) and item.get("cache_key") else str(uuid.uuid4())
        self.put(key, item)

    def put(self, key: str, item: Any) -> None:
        with self._lock:
            self._entries[key] = item
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._entries.get(key)
            if item is not None:
                self._entries.move_to_end(key)
            return item

    def delete(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------
@dataclass
class Session:
    token:   str
    user:    Dict[str, str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # DB / LLM handles
    db:               Any = None
    db_config:        Any = None
    sql_chain:        Any = None
    llm:              Any = None
    reasoning_llm:    Any = None
    rag_engine:       Any = None
    embedder:         Any = None
    cached_schema_text: str = ""
    connected:        bool  = False
    sql_dialect:      str   = "sqlserver"
    enable_nolock:    bool  = False

    # Conversation state (replaces free-text conversation_context)
    chat_history:      List[Dict]          = field(default_factory=list)
    conversation_turns: List[ConversationTurn] = field(default_factory=list)
    session_state:     SessionState        = field(default_factory=SessionState)
    query_cache:       QueryCacheStore     = field(default_factory=QueryCacheStore)
    lock:              Any                 = field(default_factory=threading.RLock, repr=False)


# ---------------------------------------------------------------------------
# Session store (in-memory, single-process)
# ---------------------------------------------------------------------------
class SessionStore:
    def __init__(self):
        self._store: Dict[str, Session] = {}
        self._lock = threading.RLock()

    def create(self, user: Dict[str, str]) -> Session:
        token = str(uuid.uuid4())
        s = Session(token=token, user=user)
        with self._lock:
            self._store[token] = s
        return s

    def get(self, token: str) -> Optional[Session]:
        with self._lock:
            return self._store.get(token)

    def delete(self, token: str):
        with self._lock:
            self._store.pop(token, None)


store = SessionStore()
