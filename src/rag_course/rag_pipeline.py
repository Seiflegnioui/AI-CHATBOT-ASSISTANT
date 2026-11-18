"""
rag_pipeline.py — The complete RAG pipeline tied together.

This is the "brain" of your chatbot. It connects all the pieces:
  1. document_loader.py  → loads and chunks your .md files
  2. embeddings.py       → converts text to vectors
  3. vector_store.py     → stores vectors in ChromaDB and searches them
  4. llm.py              → sends question + context to Gemini for an answer

Usage:
    from rag_course.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    answer = rag.ask("What are Seif's main projects?")
    print(answer)
"""

from .document_loader import load_documents, split_into_chunks
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .llm import ask as llm_ask
from . import config


class RAGPipeline:
    """
    The complete RAG pipeline in one class.

    When you create it, it:
      1. Loads the embedding model
      2. Connects to ChromaDB
      3. If the DB is empty, loads documents and indexes them

    Then you just call .ask(question) and it handles everything.
    """

    def __init__(self):
        print("=" * 50)
        print("Initializing RAG Pipeline...")
        print("=" * 50)

        # Step 1: Create the embedding model (shared between indexing and querying)
        self.embedder = EmbeddingModel()

        # Step 2: Connect to the vector store
        self.vector_store = VectorStore(embedding_model=self.embedder)

        # Step 3: If the vector store is empty, load and index documents
        if self.vector_store.collection.count() == 0:
            print("\n📄 Vector store is empty. Indexing documents...")
            self._index_documents()
        else:
            count = self.vector_store.collection.count()
            print(f"\n📦 Vector store already has {count} chunks. Ready to go!")

        print("\n" + "=" * 50)
        print("✅ RAG Pipeline is ready!")
        print("=" * 50)

    def _index_documents(self):
        """Load markdown files, split into chunks, and store in ChromaDB."""
        # Load all markdown files
        documents = load_documents()

        # Split them into chunks by markdown headers
        chunks = split_into_chunks(documents)

        # Embed and store in ChromaDB
        self.vector_store.add_chunks(chunks)

    def ask(self, question: str) -> dict:
        """
        Answer a question using RAG.

        The full flow:
          1. Search ChromaDB for relevant chunks
          2. Send question + chunks to Gemini
          3. Return the answer + sources

        Args:
            question: The recruiter's question

        Returns:
            Dict with:
              - "answer":  The generated answer
              - "sources": List of retrieved chunks (for transparency)
        """
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(
            query=question,
            top_k=config.TOP_K,
        )

        # Step 2: Generate answer with Gemini
        answer = llm_ask(question, retrieved_chunks)

        # Step 3: Return answer + sources for transparency
        sources = []
        for chunk in retrieved_chunks:
            source_path = chunk["metadata"].get("source", "unknown")
            source_name = source_path.split("\\")[-1].split("/")[-1]
            sources.append({
                "file": source_name,
                "section": chunk["metadata"].get("h1", ""),
                "subsection": chunk["metadata"].get("h2", ""),
                "relevance_distance": round(chunk["distance"], 4),
                "preview": chunk["text"][:150] + "...",
            })

        return {
            "answer": answer,
            "sources": sources,
        }

    def rebuild_index(self):
        """
        Re-index all documents. Use this when you update your .md files.

        Call this after editing any file in the Data/ folder.
        """
        documents = load_documents()
        chunks = split_into_chunks(documents)
        self.vector_store.rebuild(chunks)
        print("✅ Index rebuilt successfully!")


# ──────────────────────────────────────────────
# Quick test: run this file directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGPipeline()

    test_questions = [
        "What are Seif's main projects?",
        "What technologies does Seif know?",
        "What is Seif's education background?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ Question: {q}")
        print("=" * 60)
        result = rag.ask(q)
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n📎 Sources used:")
        for s in result["sources"]:
            print(f"   - {s['file']} → {s['section']} / {s['subsection']}")
