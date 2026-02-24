"""
Script to inspect embeddings stored in the RAG engine.
Shows documents and their vector representations.
"""

import numpy as np
from app.rag.rag_engine import RAGEngine

def main():
    print("=" * 60)
    print("RAG Engine Embedding Inspector")
    print("=" * 60)

    # Create RAG engine and load default schema
    print("\n[1] Initializing RAG Engine...")
    rag = RAGEngine()

    print("[2] Loading default schema (aggregated booking views)...")
    rag.load_default_schema()

    print(f"[3] Total documents added: {len(rag.documents)}")

    print("\n[4] Building FAISS index...")
    rag.build_index()

    # Show document details
    print("\n" + "=" * 60)
    print("DOCUMENTS IN KNOWLEDGE BASE")
    print("=" * 60)

    for i, doc in enumerate(rag.documents):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {doc.doc_id}")
        print(f"Type: {doc.doc_type}")
        print(f"Content: {doc.content[:200]}..." if len(doc.content) > 200 else f"Content: {doc.content}")
        print(f"Metadata: {doc.metadata}")

    # Show embeddings
    print("\n" + "=" * 60)
    print("EMBEDDINGS")
    print("=" * 60)

    # Get embeddings for each document
    print(f"\nEmbedding model: {rag.model_name}")
    print(f"Embedding dimension: {len(rag.embedder.embed_query('test'))}")

    print("\n--- Sample Embeddings (first 10 dimensions) ---")
    for i, doc in enumerate(rag.documents[:5]):  # Show first 5 documents
        embedding = rag.embedder.embed_query(doc.content)
        embedding_preview = np.array(embedding[:10]).round(4)
        print(f"\nDoc {i+1} ({doc.doc_type}): {doc.doc_id}")
        print(f"  Embedding[0:10]: {embedding_preview}")
        print(f"  Norm: {np.linalg.norm(embedding):.4f}")

    # Show FAISS index info
    print("\n" + "=" * 60)
    print("FAISS INDEX INFO")
    print("=" * 60)

    if rag.vector_store:
        index = rag.vector_store.index
        print(f"Index type: {type(index).__name__}")
        print(f"Total vectors: {index.ntotal}")
        print(f"Dimension: {index.d}")

        # Test retrieval
        print("\n--- Test Retrieval ---")
        test_queries = [
            "Top agents by revenue",
            "Bookings by country this month",
            "Hotel chain performance"
        ]

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = rag.retrieve(query, top_k=3)
            print(f"  Retrieved tables: {results['tables']}")
            print(f"  Schema context length: {len(results['schema_context'])} chars")
            if results['examples']:
                print(f"  Example queries found: {len(results['examples'])}")
    else:
        print("Vector store not initialized!")

    # Show similarity between documents
    print("\n" + "=" * 60)
    print("DOCUMENT SIMILARITY MATRIX (first 5 docs)")
    print("=" * 60)

    docs_to_compare = rag.documents[:5]
    embeddings = [np.array(rag.embedder.embed_query(doc.content)) for doc in docs_to_compare]

    print("\nCosine similarity between documents:")
    print(f"{'':20}", end="")
    for i in range(len(docs_to_compare)):
        print(f"Doc{i+1:3}", end="  ")
    print()

    for i, emb_i in enumerate(embeddings):
        print(f"Doc{i+1} ({docs_to_compare[i].doc_type[:8]:8})", end=" ")
        for j, emb_j in enumerate(embeddings):
            similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
            print(f"{similarity:.3f}", end="  ")
        print()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
