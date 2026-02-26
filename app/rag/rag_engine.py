"""
### FAST RAG
FAISS-only in-memory RAG engine with TTL-cached retrieval results.
Pinecone support removed — FAISS is sufficient for the dataset size.

Optimisations vs. previous version:
  - Retrieval results cached 10 min by (normalised-question, top_k)
  - Follow-up / sort / filter queries skip retrieval entirely
  - Complex queries get top_k=4; simple ones get top_k=2
  - Class-level embedder singleton (loads once per process)
"""
import re, os, hashlib, time, threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import OrderedDict

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import logging
from backend.services.runtime import log_event
logger = logging.getLogger("rag_engine")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class SchemaDoc:
    doc_id:   str
    doc_type: str          # table | example | rule | synonym
    content:  str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# RAGEngine — FAISS backend only
# ---------------------------------------------------------------------------
class RAGEngine:
    """
    FAISS in-memory RAG engine.

    Retrieve flow:
      1. Skip entirely for follow-ups / sort / filter (no value added)
      2. Cap top_k=2 for simple queries (saves embedding time)
      3. Hit TTL cache (10 min) by normalised question key
      4. FAISS similarity search → keyword fallback if FAISS unavailable
    """

    # ### FAST RAG — class-level singleton embedder (loaded once per process)
    _cached_embedder:    Any = None
    _cached_model_name:  Optional[str] = None
    # retrieval result cache: key → (timestamp, result_dict)
    _retrieve_cache: "OrderedDict[str, Tuple[float, Dict]]" = OrderedDict()
    _cache_lock = threading.RLock()
    CACHE_TTL: int = int(os.getenv("RAG_CACHE_TTL_SECONDS", "900"))   # 15 min default
    CACHE_MAX: int = int(os.getenv("RAG_CACHE_MAX_ENTRIES", "256"))

    # Keywords that signal a complex query needing more examples
    _COMPLEX_SIGNALS = frozenset({
        "yoy", "ytd", "pytd", "growth", "trend", "compare", "versus",
        "vs", "monthly", "quarterly", "yearly", "breakdown", "split",
    })

    # Prefixes / patterns that identify follow-up / filter / sort queries
    _FOLLOWUP_PREFIXES = (
        "what about", "how about", "only for", "filter by", "just ",
        "and ", "show only", "same but", "now show", "drill down",
    )

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_embeddings: bool = True,
    ):
        self.documents:    List[SchemaDoc] = []
        self.vector_store: Optional[FAISS] = None
        self.model_name                    = model_name
        self.embedder:     Any             = None

        if not enable_embeddings:
            return

        # Reuse class-level cached embedder (saves ~2-3 s startup)
        if RAGEngine._cached_embedder and RAGEngine._cached_model_name == model_name:
            self.embedder = RAGEngine._cached_embedder
        else:
            try:
                self.embedder = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
                )
                RAGEngine._cached_embedder   = self.embedder
                RAGEngine._cached_model_name = model_name
                logger.info("Embedder loaded: %s", model_name)
            except Exception:
                logger.warning("Embedder unavailable — keyword fallback only", exc_info=True)
                self.embedder = None

    # ------------------------------------------------------------------
    # Document ingestion helpers
    # ------------------------------------------------------------------
    def _doc_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _add_doc(self, doc: SchemaDoc):
        self.documents.append(doc)

    def add_example(self, question: str, sql: str,
                    variations: Optional[List[str]] = None):
        for q in [question] + (variations or []):
            self._add_doc(SchemaDoc(
                doc_id   = self._doc_id(q + sql),
                doc_type = "example",
                content  = f"Q: {q}\nSQL: {sql}",
                metadata = {"question": q, "sql": sql, "type": "example"},
            ))

    def add_rule(self, name: str, description: str,
                 sql_hint: Optional[str] = None):
        content = f"RULE: {name}\n{description}"
        if sql_hint:
            content += f"\nSQL hint: {sql_hint}"
        self._add_doc(SchemaDoc(self._doc_id(content), "rule", content,
                                {"name": name, "type": "rule"}))

    def add_table(self, name: str, description: str, columns: List[Dict],
                  aliases: Optional[List[str]] = None):
        col_str = ", ".join(c.get("name", "") for c in columns[:20])
        content = f"TABLE: {name}\n{description}\nColumns: {col_str}"
        if aliases:
            content += f"\nAliases: {', '.join(aliases)}"
        self._add_doc(SchemaDoc(self._doc_id(name + content), "table", content,
                                {"table": name, "type": "table"}))

    def add_synonym(self, term: str, means: str, context: Optional[str] = None):
        content = f"SYNONYM: {term} → {means}"
        if context:
            content += f" ({context})"
        self._add_doc(SchemaDoc(self._doc_id(content), "synonym", content,
                                {"term": term, "type": "synonym"}))

    # ------------------------------------------------------------------
    # Index build
    # ------------------------------------------------------------------
    def build_index(self):
        if not self.embedder or not self.documents:
            return
        lc_docs = [Document(page_content=d.content, metadata=d.metadata)
                   for d in self.documents]
        try:
            self.vector_store = FAISS.from_documents(lc_docs, self.embedder)
            logger.info("FAISS index built: %d docs", len(lc_docs))
        except Exception:
            logger.warning("FAISS index build failed", exc_info=True)
            self.vector_store = None

    # ------------------------------------------------------------------
    # ### FAST RAG — retrieval skip detection
    # ------------------------------------------------------------------
    def _is_followup_or_simple(self, query: str) -> bool:
        """Return True if the query is a follow-up / filter / sort request
        that gains nothing from RAG retrieval."""
        q = query.lower().strip()
        if any(q.startswith(p) for p in self._FOLLOWUP_PREFIXES):
            return True
        words = q.split()
        # Very short questions are almost always contextual follow-ups
        return len(words) <= 4

    def _effective_k(self, query: str, top_k: int) -> int:
        """Reduce top_k for simple queries to cut embedding time."""
        words = set(re.findall(r"[a-z]+", query.lower()))
        return top_k if (words & self._COMPLEX_SIGNALS) else min(top_k, 2)

    def _cache_key(self, query: str, top_k: int, intent_key: Optional[str] = None) -> str:
        norm = re.sub(r"\s+", " ", query.lower().strip())
        intent = (intent_key or "").strip().lower()
        return hashlib.md5(f"{intent}:{norm}:{top_k}".encode()).hexdigest()

    # ------------------------------------------------------------------
    # Public retrieve API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query:             str,
        top_k:             int  = 4,
        fast_mode:         bool = False,
        skip_for_followup: bool = False,
        intent_key:        Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return {examples, rules, tables} for the given query.

        Args:
            skip_for_followup: when True, short/follow-up queries return empty immediately.
            fast_mode:         ignored (kept for API compat).
        """
        # ### FAST RAG — skip retrieval for follow-ups
        if skip_for_followup and self._is_followup_or_simple(query):
            return {"examples": [], "rules": [], "tables": [], "skipped": True}

        k = self._effective_k(query, top_k)
        cache_key = self._cache_key(query, k, intent_key=intent_key)

        # ### FAST RAG — TTL cache hit (minimal lock time)
        now = time.time()
        
        # Quick cache check with minimal lock time
        with self._cache_lock:
            cached = self._retrieve_cache.get(cache_key)
            if cached and (now - cached[0]) < self.CACHE_TTL:
                self._retrieve_cache.move_to_end(cache_key)
                log_event(logger, logging.INFO, "rag_cache_hit", cache_key=cache_key, top_k=k)
                return cached[1]
            if cached:
                self._retrieve_cache.pop(cache_key, None)

        # Do the expensive retrieval work outside the lock
        result = self._do_retrieve(query, k)

        # Store in cache, evict oldest if over limit (minimal lock time)
        with self._cache_lock:
            self._retrieve_cache[cache_key] = (now, result)
            self._retrieve_cache.move_to_end(cache_key)
            while len(self._retrieve_cache) > self.CACHE_MAX:
                self._retrieve_cache.popitem(last=False)
        log_event(logger, logging.INFO, "rag_cache_miss", cache_key=cache_key, top_k=k)

        return result

    @classmethod
    def clear_retrieval_cache(cls) -> None:
        with cls._cache_lock:
            cls._retrieve_cache.clear()

    # ------------------------------------------------------------------
    # Internal retrieval
    # ------------------------------------------------------------------
    def _do_retrieve(self, query: str, top_k: int) -> Dict[str, Any]:
        if self.vector_store and self.embedder:
            try:
                docs = self.vector_store.similarity_search(query, k=top_k * 2)
                seen: set = set()
                examples, rules, tables = [], [], []
                for doc in docs:
                    key = doc.page_content[:80]
                    if key in seen:
                        continue
                    seen.add(key)
                    t = doc.metadata.get("type", "")
                    if t == "example" and len(examples) < top_k:
                        examples.append({
                            "question": doc.metadata.get("question", ""),
                            "sql":      doc.metadata.get("sql", ""),
                        })
                    elif t == "rule" and len(rules) < 2:
                        rules.append({"name": doc.metadata.get("name", ""),
                                      "content": doc.page_content})
                    elif t == "table" and len(tables) < 2:
                        tables.append({"table": doc.metadata.get("table", ""),
                                       "content": doc.page_content})
                return {"examples": examples, "rules": rules, "tables": tables}
            except Exception:
                logger.warning("FAISS search failed, using keyword fallback", exc_info=True)

        return self._keyword_retrieve(query, top_k)

    def _keyword_retrieve(self, query: str, top_k: int) -> Dict[str, Any]:
        q_words = set(re.findall(r"[a-z]+", query.lower()))
        scored = sorted(
            [(len(q_words & set(re.findall(r"[a-z]+", d.content.lower()))), d)
             for d in self.documents],
            key=lambda x: -x[0],
        )
        examples = [
            {"question": d.metadata.get("question", ""),
             "sql":      d.metadata.get("sql", "")}
            for sc, d in scored
            if sc > 0 and d.doc_type == "example"
        ][:top_k]
        return {"examples": examples, "rules": [], "tables": []}

    # ------------------------------------------------------------------
    # Default schema / example population
    # Called by initialize_connection after RAGEngine is created.
    # ------------------------------------------------------------------
    def load_default_schema(self):
        """Load hard-coded domain examples so RAG has content on first boot."""
        self.add_example(
            "show top 10 agents by revenue",
            "WITH AM AS (SELECT DISTINCT AgentId, AgentName FROM dbo.AgentMaster_V1 WITH (NOLOCK)) "
            "SELECT TOP 10 AM.AgentName AS [Agent Name], SUM(BD.AgentBuyingPrice) AS [Total Revenue] "
            "FROM dbo.BookingData BD WITH (NOLOCK) LEFT JOIN AM ON AM.AgentId = BD.AgentId "
            "WHERE BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') "
            "GROUP BY AM.AgentName ORDER BY [Total Revenue] DESC",
        )
        self.add_example(
            "total bookings by country this month",
            "WITH MC AS (SELECT DISTINCT CountryID, Country FROM dbo.Master_Country WITH (NOLOCK)) "
            "SELECT MC.Country AS [Country], COUNT(DISTINCT BD.PNRNo) AS [Total Bookings] "
            "FROM dbo.BookingData BD WITH (NOLOCK) LEFT JOIN MC ON MC.CountryID = BD.ProductCountryid "
            "WHERE BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') "
            "AND BD.CreatedDate >= DATEFROMPARTS(YEAR(GETDATE()),MONTH(GETDATE()),1) "
            "AND BD.CreatedDate < DATEADD(MONTH,1,DATEFROMPARTS(YEAR(GETDATE()),MONTH(GETDATE()),1)) "
            "GROUP BY MC.Country ORDER BY [Total Bookings] DESC",
        )
        self.add_example(
            "supplier with highest bookings last week",
            "WITH SM AS (SELECT DISTINCT EmployeeId, SupplierName FROM dbo.suppliermaster_Report WITH (NOLOCK)) "
            "SELECT TOP 1 SM.SupplierName AS [Supplier Name], COUNT(DISTINCT BD.PNRNo) AS [Total Bookings] "
            "FROM dbo.BookingData BD WITH (NOLOCK) LEFT JOIN SM ON SM.EmployeeId = BD.SupplierId "
            "WHERE BD.BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request') "
            "AND BD.CreatedDate >= DATEADD(DAY,-(DATEPART(WEEKDAY,CAST(GETDATE() AS DATE))-1)-7,CAST(GETDATE() AS DATE)) "
            "AND BD.CreatedDate < DATEADD(DAY,-(DATEPART(WEEKDAY,CAST(GETDATE() AS DATE))-1),CAST(GETDATE() AS DATE)) "
            "GROUP BY SM.SupplierName ORDER BY [Total Bookings] DESC",
        )
        self.add_rule(
            "Status exclusion",
            "Always filter: BookingStatus NOT IN ('Cancelled','Not Confirmed','On Request')",
        )
        self.add_rule(
            "SARGable dates",
            "Never use CAST(date_col AS DATE) in WHERE. Use date_col >= start AND date_col < end.",
        )
        self.add_rule(
            "CTE dedup",
            "Lookup tables (AgentMaster_V1, suppliermaster_Report, Master_Country, Master_City) "
            "have duplicates. Always wrap in a CTE with SELECT DISTINCT before joining.",
        )
        self.build_index()
