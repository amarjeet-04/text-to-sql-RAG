"""
RAG Engine - Retrieves relevant schema context for SQL generation
Supports FAISS (default) and Pinecone for vector search.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re
import os
import hashlib
import time

# LangChain imports (required)
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    Pinecone = None
    ServerlessSpec = None




@dataclass
class SchemaDoc:
    """Document in knowledge base"""
    doc_id: str
    doc_type: str  # table, column, example, rule, synonym
    content: str
    metadata: Dict[str, Any]


class RAGEngine:
    """
    RAG Engine for schema context retrieval.
    Optimized for FAISS in-memory retrieval and Pinecone remote retrieval.

    Stores and retrieves:
    - Table definitions
    - Column descriptions
    - Example queries (question → SQL)
    - Business rules
    - Synonyms/aliases
    """

    # Class-level cache for embedder (expensive to load)
    _cached_embedder = None
    _cached_model_name = None

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_embeddings: bool = True,
    ):
        # Stored docs for keyword fallback and context construction
        self.documents: List[SchemaDoc] = []
        # Vector stores for semantic retrieval
        self.vector_store = None
        self.pinecone_index = None
        self.model_name = model_name

        requested_backend = os.getenv("VECTOR_BACKEND", "faiss").strip().lower()
        self.vector_backend = requested_backend if requested_backend in {"faiss", "pinecone"} else "faiss"
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        self.pinecone_index_name = os.getenv("PINECONE_INDEX", "text2sql-schema").strip() or "text2sql-schema"
        self.pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "default").strip() or "default"
        self.pinecone_metric = os.getenv("PINECONE_METRIC", "cosine").strip().lower() or "cosine"
        self.pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws").strip() or "aws"
        self.pinecone_region = os.getenv("PINECONE_REGION", "us-east-1").strip() or "us-east-1"

        if self.vector_backend == "pinecone":
            if not self.pinecone_api_key or Pinecone is None:
                # Fall back safely if Pinecone is not configured/installed.
                self.vector_backend = "faiss"

        self.embedder = None

        if not enable_embeddings:
            return

        # Reuse cached embedder if model is the same (saves ~2-3 seconds)
        if RAGEngine._cached_embedder is not None and RAGEngine._cached_model_name == model_name:
            self.embedder = RAGEngine._cached_embedder
        else:
            try:
                self.embedder = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
                )
                # Cache for future use
                RAGEngine._cached_embedder = self.embedder
                RAGEngine._cached_model_name = model_name
            except Exception:
                # Graceful fallback: keep app usable even if embedding deps/model are unavailable.
                self.embedder = None

    def _hash_doc_id(self, doc: SchemaDoc) -> str:
        key = f"{doc.doc_id}|{doc.doc_type}|{doc.content}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
                continue
            if isinstance(value, list):
                items = []
                for item in value:
                    if isinstance(item, (str, int, float, bool)):
                        items.append(item)
                    elif item is not None:
                        items.append(str(item))
                cleaned[key] = items
                continue
            cleaned[key] = str(value)
        return cleaned

    def _resolve_embedding_dim(self) -> int:
        if self.embedder is None:
            return 0
        vec = self.embedder.embed_query("dimension_probe")
        return len(vec)

    def _pinecone_index_exists(self, pc: Any, index_name: str) -> bool:
        try:
            result = pc.list_indexes()
            if hasattr(result, "names"):
                return index_name in result.names()
            if isinstance(result, dict):
                if "indexes" in result:
                    return any(i.get("name") == index_name for i in result.get("indexes", []))
                return index_name in result
            return any(getattr(i, "name", None) == index_name for i in result)
        except Exception:
            return False

    def _ensure_pinecone_index(self):
        if self.pinecone_index is not None:
            return
        if Pinecone is None or not self.pinecone_api_key:
            raise RuntimeError("Pinecone is not available. Install dependency and set PINECONE_API_KEY.")

        pc = Pinecone(api_key=self.pinecone_api_key)
        dim = self._resolve_embedding_dim()
        if dim <= 0:
            raise RuntimeError("Unable to resolve embedding dimension for Pinecone index.")

        if not self._pinecone_index_exists(pc, self.pinecone_index_name):
            if ServerlessSpec is None:
                raise RuntimeError("Pinecone ServerlessSpec is unavailable in current pinecone package.")
            pc.create_index(
                name=self.pinecone_index_name,
                dimension=dim,
                metric=self.pinecone_metric,
                spec=ServerlessSpec(cloud=self.pinecone_cloud, region=self.pinecone_region),
            )
            for _ in range(30):
                try:
                    desc = pc.describe_index(self.pinecone_index_name)
                    ready = False
                    if hasattr(desc, "status"):
                        status = getattr(desc, "status")
                        if isinstance(status, dict):
                            ready = bool(status.get("ready", False))
                        else:
                            ready = bool(getattr(status, "ready", False))
                    elif isinstance(desc, dict):
                        ready = bool(desc.get("status", {}).get("ready", False))
                    if ready:
                        break
                except Exception:
                    pass
                time.sleep(1)

        self.pinecone_index = pc.Index(self.pinecone_index_name)
    
    def add_table(self, name: str, description: str, columns: List[Dict], aliases: List[str] = None):
        """Add table with columns and optional aliases"""
        # Main table doc
        col_text = "\n".join([
            f"  • {c['name']} ({c.get('type','')}) - {c.get('description','')}"
            for c in columns
        ])
        
        content = f"Table: {name}\nDescription: {description}\nColumns:\n{col_text}"
        
        self._add_doc(SchemaDoc(
            doc_id=f"table_{name}",
            doc_type="table",
            content=content,
            metadata={"table": name, "columns": [c["name"] for c in columns]}
        ))
        
        # Add column docs
        for col in columns:
            self._add_doc(SchemaDoc(
                doc_id=f"col_{name}_{col['name']}",
                doc_type="column",
                content=f"{col['name']} in {name}: {col.get('description', '')} (Type: {col.get('type', '')})",
                metadata={"table": name, "column": col["name"], "type": col.get("type")}
            ))
        
        # Add aliases
        if aliases:
            for alias in aliases:
                self._add_doc(SchemaDoc(
                    doc_id=f"alias_{name}_{alias}",
                    doc_type="synonym",
                    content=f"'{alias}' refers to {name} table. When user mentions {alias}, use {name}.",
                    metadata={"table": name, "alias": alias}
                ))
    
    def add_relationship(self, from_table: str, from_col: str, to_table: str, to_col: str):
        """Add table relationship for JOINs"""
        content = f"JOIN: {from_table}.{from_col} → {to_table}.{to_col}\n"
        content += f"To connect {from_table} with {to_table}, use: "
        content += f"{from_table} JOIN {to_table} ON {from_table}.{from_col} = {to_table}.{to_col}"
        
        self._add_doc(SchemaDoc(
            doc_id=f"rel_{from_table}_{to_table}",
            doc_type="relationship",
            content=content,
            metadata={"from": from_table, "to": to_table}
        ))
    
    def add_example(self, question: str, sql: str, variations: List[str] = None):
        """Add example query with variations"""
        content = f"Question: {question}\nSQL: {sql}"
        
        self._add_doc(SchemaDoc(
            doc_id=f"ex_{hash(question) % 10000}",
            doc_type="example",
            content=content,
            metadata={"question": question, "sql": sql}
        ))
        
        # Add variations as separate examples
        if variations:
            for var in variations:
                self._add_doc(SchemaDoc(
                    doc_id=f"ex_{hash(var) % 10000}",
                    doc_type="example",
                    content=f"Question: {var}\nSQL: {sql}",
                    metadata={"question": var, "sql": sql, "is_variation": True}
                ))
    
    def add_rule(self, name: str, description: str, sql_hint: str = None):
        """Add business rule"""
        content = f"Rule - {name}: {description}"
        if sql_hint:
            content += f"\nSQL: {sql_hint}"
        
        self._add_doc(SchemaDoc(
            doc_id=f"rule_{name}",
            doc_type="rule",
            content=content,
            metadata={"rule": name, "sql": sql_hint}
        ))
    
    def add_synonym(self, term: str, means: str, context: str = None):
        """Add term synonym/mapping"""
        content = f"'{term}' means '{means}'"
        if context:
            content += f" in the context of {context}"
        
        self._add_doc(SchemaDoc(
            doc_id=f"syn_{term}",
            doc_type="synonym",
            content=content,
            metadata={"term": term, "means": means}
        ))
    
    def _add_doc(self, doc: SchemaDoc):
        """Add document to pending list (will be indexed in batch later)"""
        self.documents.append(doc)

    def build_index(self):
        """Build vector index from all documents in one batch."""
        if not self.documents or self.embedder is None:
            return

        if self.vector_backend == "pinecone":
            try:
                self._ensure_pinecone_index()
                self.pinecone_index.delete(delete_all=True, namespace=self.pinecone_namespace)

                batch_size = 64
                for i in range(0, len(self.documents), batch_size):
                    chunk = self.documents[i:i + batch_size]
                    texts = [doc.content for doc in chunk]
                    embeddings = self.embedder.embed_documents(texts)
                    vectors = []
                    for doc, emb in zip(chunk, embeddings):
                        metadata = self._sanitize_metadata({"doc_id": doc.doc_id, "doc_type": doc.doc_type, **doc.metadata})
                        metadata["_content"] = doc.content
                        vectors.append({
                            "id": self._hash_doc_id(doc),
                            "values": emb,
                            "metadata": metadata,
                        })
                    self.pinecone_index.upsert(vectors=vectors, namespace=self.pinecone_namespace)
                return
            except Exception:
                # Keep app usable if Pinecone calls fail.
                self.vector_backend = "faiss"
                self.pinecone_index = None

        lc_docs = [
            Document(
                page_content=doc.content,
                metadata={"doc_id": doc.doc_id, "doc_type": doc.doc_type, **doc.metadata}
            )
            for doc in self.documents
        ]
        self.vector_store = FAISS.from_documents(lc_docs, self.embedder)
    
    def retrieve(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.
        
        Returns:
            {
                "schema_context": str,
                "tables": List[str],
                "examples": List[Dict],
                "rules": List[str],
            }
        """
        if not self.documents:
            return {"schema_context": "", "tables": [], "examples": [], "rules": []}
        
        # Get relevant docs
        if self.vector_store is not None or (self.vector_backend == "pinecone" and self.pinecone_index is not None):
            docs = self._retrieve_by_embedding(query, top_k)
        else:
            docs = self._retrieve_by_keyword(query, top_k)
        
        # Organize results
        tables = set()
        examples = []
        rules = []
        schema_parts = []
        
        for doc in docs:
            if doc.doc_type == "table":
                tables.add(doc.metadata.get("table"))
                schema_parts.append(doc.content)
            elif doc.doc_type == "column":
                tables.add(doc.metadata.get("table"))
                schema_parts.append(doc.content)
            elif doc.doc_type == "relationship":
                schema_parts.append(doc.content)
            elif doc.doc_type == "example":
                examples.append({
                    "question": doc.metadata.get("question"),
                    "sql": doc.metadata.get("sql")
                })
            elif doc.doc_type == "rule":
                rules.append(doc.content)
            elif doc.doc_type == "synonym":
                schema_parts.append(doc.content)

        # Fallback: ensure we always return some schema context
        if not schema_parts:
            for doc in self.documents:
                if doc.doc_type == "table":
                    tables.add(doc.metadata.get("table"))
                    schema_parts.append(doc.content)
                if len(schema_parts) >= 2:
                    break

        return {
            "schema_context": "\n\n".join(schema_parts),
            "tables": list(tables),
            "examples": examples[:3],
            "rules": rules
        }
    
    def _retrieve_by_embedding(self, query: str, top_k: int) -> List[SchemaDoc]:
        """Semantic similarity retrieval using Pinecone or FAISS."""
        if self.vector_backend == "pinecone" and self.pinecone_index is not None and self.embedder is not None:
            try:
                query_vec = self.embedder.embed_query(query)
                response = self.pinecone_index.query(
                    vector=query_vec,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=self.pinecone_namespace,
                )
                matches = getattr(response, "matches", None)
                if matches is None and isinstance(response, dict):
                    matches = response.get("matches", [])
                mapped = []
                for match in matches or []:
                    metadata = getattr(match, "metadata", None)
                    if metadata is None and isinstance(match, dict):
                        metadata = match.get("metadata", {}) or {}
                    metadata = dict(metadata or {})
                    content = metadata.pop("_content", "")
                    mapped.append(
                        SchemaDoc(
                            doc_id=str(metadata.pop("doc_id", "")),
                            doc_type=str(metadata.pop("doc_type", "table")),
                            content=content,
                            metadata=metadata,
                        )
                    )
                if mapped:
                    return mapped
            except Exception:
                pass

        if self.vector_store is None:
            return []
        results: List[Document] = self.vector_store.similarity_search(query, k=top_k)

        # Convert LC Documents back to SchemaDoc-like objects for downstream logic
        mapped: List[SchemaDoc] = []
        for doc in results:
            meta = doc.metadata or {}
            mapped.append(
                SchemaDoc(
                    doc_id=meta.get("doc_id", ""),
                    doc_type=meta.get("doc_type", "table"),
                    content=doc.page_content,
                    metadata={k: v for k, v in meta.items() if k not in ["doc_id", "doc_type"]}
                )
            )
        return mapped
    
    def _retrieve_by_keyword(self, query: str, top_k: int) -> List[SchemaDoc]:
        """Keyword-based fallback retrieval"""
        query_words = set(query.lower().split())
        
        scores = []
        for doc in self.documents:
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            scores.append((overlap, doc))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scores[:top_k] if score > 0]
    
    def load_default_schema(self):
        """Load travel booking aggregated views schema"""

        # AGENT LEVEL VIEW
        self.add_table(
            name="agent_level_view",
            description="Aggregated booking data at the agent level with sales, profit, and booking counts per agent per date",
            columns=[
                {"name": "agentid", "type": "BIGINT", "description": "Agent unique identifier"},
                {"name": "agentcode", "type": "VARCHAR(10)", "description": "Short agent code"},
                {"name": "agentname", "type": "VARCHAR(255)", "description": "Agent or agency name"},
                {"name": "agenttype", "type": "VARCHAR(100)", "description": "Type of agent"},
                {"name": "agentcountry", "type": "VARCHAR(100)", "description": "Country where agent is based"},
                {"name": "agentcity", "type": "VARCHAR(100)", "description": "City where agent is based"},
                {"name": "booking_date", "type": "DATE", "description": "Date when booking was made"},
                {"name": "checkin_date", "type": "DATE", "description": "Guest check-in date"},
                {"name": "checkout_date", "type": "DATE", "description": "Guest check-out date"},
                {"name": "total_sales", "type": "NUMERIC(14,2)", "description": "Total sales amount (revenue)"},
                {"name": "total_profit", "type": "NUMERIC(14,2)", "description": "Total profit amount"},
                {"name": "total_booking", "type": "INTEGER", "description": "Total number of bookings"},
            ],
            aliases=["agents", "agent bookings", "agent performance", "agent sales"]
        )

        # SUPPLIER LEVEL VIEW
        self.add_table(
            name="supplier_level_view",
            description="Aggregated booking data at the supplier level with sales, profit, and booking counts per supplier per date",
            columns=[
                {"name": "supplierid", "type": "BIGINT", "description": "Supplier unique identifier"},
                {"name": "employeeid", "type": "BIGINT", "description": "Employee identifier linked to supplier"},
                {"name": "suppliername", "type": "VARCHAR(255)", "description": "Supplier name"},
                {"name": "booking_date", "type": "DATE", "description": "Date when booking was made"},
                {"name": "checkin_date", "type": "DATE", "description": "Guest check-in date"},
                {"name": "checkout_date", "type": "DATE", "description": "Guest check-out date"},
                {"name": "total_sales", "type": "NUMERIC(14,2)", "description": "Total sales amount (revenue)"},
                {"name": "total_profit", "type": "NUMERIC(14,2)", "description": "Total profit amount"},
                {"name": "total_booking", "type": "INTEGER", "description": "Total number of bookings"},
            ],
            aliases=["suppliers", "supplier bookings", "supplier performance"]
        )

        # COUNTRY LEVEL VIEW
        self.add_table(
            name="country_level_view",
            description="Aggregated booking data at the destination country level with sales, profit, and booking counts per country per date",
            columns=[
                {"name": "productcountryid", "type": "BIGINT", "description": "Product country identifier"},
                {"name": "countryid", "type": "BIGINT", "description": "Country identifier"},
                {"name": "country", "type": "VARCHAR(150)", "description": "Destination country name"},
                {"name": "booking_date", "type": "DATE", "description": "Date when booking was made"},
                {"name": "checkin_date", "type": "DATE", "description": "Guest check-in date"},
                {"name": "checkout_date", "type": "DATE", "description": "Guest check-out date"},
                {"name": "total_sales", "type": "NUMERIC(14,2)", "description": "Total sales amount (revenue)"},
                {"name": "total_profit", "type": "NUMERIC(14,2)", "description": "Total profit amount"},
                {"name": "total_booking", "type": "INTEGER", "description": "Total number of bookings"},
            ],
            aliases=["countries", "country bookings", "destination countries"]
        )

        # CITY LEVEL VIEW
        self.add_table(
            name="city_level_view",
            description="Aggregated booking data at the destination city level with sales, profit, and booking counts per city per date",
            columns=[
                {"name": "productcityid", "type": "BIGINT", "description": "Product city identifier"},
                {"name": "cityid", "type": "BIGINT", "description": "City identifier"},
                {"name": "city", "type": "VARCHAR(150)", "description": "Destination city name"},
                {"name": "booking_date", "type": "DATE", "description": "Date when booking was made"},
                {"name": "checkin_date", "type": "DATE", "description": "Guest check-in date"},
                {"name": "checkout_date", "type": "DATE", "description": "Guest check-out date"},
                {"name": "total_sales", "type": "NUMERIC(14,2)", "description": "Total sales amount (revenue)"},
                {"name": "total_profit", "type": "NUMERIC(14,2)", "description": "Total profit amount"},
                {"name": "total_booking", "type": "INTEGER", "description": "Total number of bookings"},
            ],
            aliases=["cities", "city bookings", "destination cities"]
        )

        # CLIENT NATIONALITY LEVEL VIEW
        self.add_table(
            name="client_nationality_level_view",
            description="Aggregated booking data at the client nationality level with sales, profit, and booking counts per nationality per date",
            columns=[
                {"name": "clientnationality", "type": "VARCHAR(150)", "description": "Client/guest nationality"},
                {"name": "booking_date", "type": "DATE", "description": "Date when booking was made"},
                {"name": "checkin_date", "type": "DATE", "description": "Guest check-in date"},
                {"name": "checkout_date", "type": "DATE", "description": "Guest check-out date"},
                {"name": "total_sales", "type": "NUMERIC(14,2)", "description": "Total sales amount (revenue)"},
                {"name": "total_profit", "type": "NUMERIC(14,2)", "description": "Total profit amount"},
                {"name": "total_booking", "type": "INTEGER", "description": "Total number of bookings"},
            ],
            aliases=["nationalities", "nationality bookings", "guest nationalities", "client nationalities"]
        )

        # HOTEL LEVEL VIEW
        self.add_table(
            name="hotel_level_view",
            description="Aggregated booking data at the hotel/product level with sales, profit, and booking counts per hotel per date",
            columns=[
                {"name": "productname", "type": "VARCHAR(255)", "description": "Hotel or product name"},
                {"name": "booking_date", "type": "DATE", "description": "Date when booking was made"},
                {"name": "checkin_date", "type": "DATE", "description": "Guest check-in date"},
                {"name": "checkout_date", "type": "DATE", "description": "Guest check-out date"},
                {"name": "total_sales", "type": "NUMERIC(14,2)", "description": "Total sales amount (revenue)"},
                {"name": "total_profit", "type": "NUMERIC(14,2)", "description": "Total profit amount"},
                {"name": "total_booking", "type": "INTEGER", "description": "Total number of bookings"},
            ],
            aliases=["hotels", "hotel bookings", "products", "hotel performance"]
        )

        # HOTEL CHAIN LEVEL VIEW
        self.add_table(
            name="hotel_chain_level_view",
            description="Aggregated booking data at the hotel chain level with sales, profit, and booking counts per chain per date",
            columns=[
                {"name": "productid", "type": "BIGINT", "description": "Product identifier"},
                {"name": "hotelid", "type": "BIGINT", "description": "Hotel identifier"},
                {"name": "hotelname", "type": "VARCHAR(255)", "description": "Hotel name"},
                {"name": "chain", "type": "VARCHAR(150)", "description": "Hotel chain name (e.g., Marriott, Hilton)"},
                {"name": "booking_date", "type": "DATE", "description": "Date when booking was made"},
                {"name": "checkin_date", "type": "DATE", "description": "Guest check-in date"},
                {"name": "checkout_date", "type": "DATE", "description": "Guest check-out date"},
                {"name": "total_sales", "type": "NUMERIC(14,2)", "description": "Total sales amount (revenue)"},
                {"name": "total_profit", "type": "NUMERIC(14,2)", "description": "Total profit amount"},
                {"name": "total_booking", "type": "INTEGER", "description": "Total number of bookings"},
            ],
            aliases=["hotel chains", "chain bookings", "chain performance"]
        )

        # SYNONYMS
        self.add_synonym("revenue", "total_sales", "SUM(total_sales) in any view")
        self.add_synonym("sales", "total_sales", "SUM(total_sales) in any view")
        self.add_synonym("income", "total_sales", "SUM(total_sales) in any view")
        self.add_synonym("profit", "total_profit", "SUM(total_profit) in any view")
        self.add_synonym("bookings", "total_booking", "SUM(total_booking) in any view")
        self.add_synonym("nationality", "clientnationality", "client_nationality_level_view table")
        self.add_synonym("country", "country", "country_level_view table for destinations")
        self.add_synonym("city", "city", "city_level_view table for destinations")
        self.add_synonym("hotel", "productname", "hotel_level_view table")
        self.add_synonym("chain", "chain", "hotel_chain_level_view table")

        # BUSINESS RULES
        self.add_rule(
            "aggregated_views",
            "All tables are pre-aggregated views. Use SUM() for total_sales, total_profit, total_booking when grouping by dimensions. Each row already represents an aggregate per dimension per date.",
            "SUM(total_sales) AS revenue, SUM(total_profit) AS profit, SUM(total_booking) AS bookings"
        )
        self.add_rule(
            "date_columns",
            "Each view has three date columns: booking_date (when booked), checkin_date (guest arrival), checkout_date (guest departure). Use the appropriate one based on the user's question.",
            "WHERE booking_date >= CURRENT_DATE - INTERVAL '7 days'"
        )
        self.add_rule(
            "choose_correct_view",
            "Choose the view that matches the user's question: agent_level_view for agent queries, supplier_level_view for supplier queries, country_level_view for country/destination queries, city_level_view for city queries, client_nationality_level_view for nationality queries, hotel_level_view for hotel queries, hotel_chain_level_view for chain queries.",
            None
        )

        # EXAMPLE QUERIES
        self.add_example(
            "Show top 5 agents by revenue",
            "SELECT agentname, SUM(total_sales) AS revenue FROM agent_level_view GROUP BY agentname ORDER BY revenue DESC LIMIT 5",
            variations=[
                "top agents",
                "best performing agents",
                "which agents have highest revenue",
                "agent performance",
            ]
        )

        self.add_example(
            "Revenue by country this month",
            "SELECT country, SUM(total_sales) AS revenue, SUM(total_booking) AS bookings FROM country_level_view WHERE date_trunc('month', booking_date) = date_trunc('month', CURRENT_DATE) GROUP BY country ORDER BY revenue DESC",
            variations=[
                "breakdown by country",
                "revenue per country",
                "bookings by country",
            ]
        )

        self.add_example(
            "Top hotels by bookings last month",
            "SELECT productname, SUM(total_booking) AS bookings, SUM(total_sales) AS revenue FROM hotel_level_view WHERE date_trunc('month', booking_date) = date_trunc('month', CURRENT_DATE - INTERVAL '1 month') GROUP BY productname ORDER BY bookings DESC LIMIT 10",
            variations=[
                "best hotels last month",
                "most booked hotels",
                "hotel performance last month",
            ]
        )

        self.add_example(
            "Bookings by client nationality this year",
            "SELECT clientnationality, SUM(total_booking) AS bookings, SUM(total_sales) AS revenue FROM client_nationality_level_view WHERE EXTRACT(YEAR FROM booking_date) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY clientnationality ORDER BY bookings DESC LIMIT 20",
            variations=[
                "nationality breakdown",
                "guest nationalities",
                "bookings by nationality",
            ]
        )

        self.add_example(
            "Top cities by revenue",
            "SELECT city, SUM(total_sales) AS revenue, SUM(total_profit) AS profit FROM city_level_view GROUP BY city ORDER BY revenue DESC LIMIT 10",
            variations=[
                "best cities",
                "top destinations",
                "city performance",
            ]
        )


# Optional singleton (lazy) for scripts that want a process-global instance.
rag_engine = None


def get_rag_engine() -> RAGEngine:
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine
