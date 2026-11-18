"""
vector_store.py — Step 3: Store and search vectors with ChromaDB.

RAG Pipeline Flow:
  [Chunks + Vectors] → ChromaDB → query("question") → [Top K relevant chunks]

What is a vector store?
  Think of it like a database, but instead of searching by exact text match
  (WHERE name = 'Seif'), you search by MEANING. You give it a question,
  it converts it to a vector, and finds the stored chunks whose vectors
  are closest — meaning most semantically similar.

Why ChromaDB?
  - Simple API (just add + query)
  - Stores data on disk (persists between runs)
  - No external server needed (unlike Pinecone, Weaviate, etc.)
  - Perfect for a portfolio chatbot with ~50 chunks
"""

import chromadb

from . import config
from .embeddings import EmbeddingModel


class VectorStore:
    """
    Manages the ChromaDB vector database.

    Usage:
        store = VectorStore()
        store.add_chunks(chunks)              # One-time: index your documents
        results = store.search("question")    # Every query: find relevant chunks
    """

    def __init__(self, embedding_model: EmbeddingModel | None = None):
        # Create or load the embedding model
        self.embedder = embedding_model or EmbeddingModel()

        # Create a persistent ChromaDB client (saves to disk)
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

        # Get or create our collection
        # A collection is like a table — it holds all our chunks and their vectors
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"description": "Seif's portfolio knowledge base"},
        )

        print(f"✅ ChromaDB ready! Collection '{config.COLLECTION_NAME}' "
              f"has {self.collection.count()} chunks stored.")

    def add_chunks(self, chunks):
        """
        Embed and store document chunks in ChromaDB.

        Each chunk gets:
          - An ID (so ChromaDB can track it)
          - The text content (what gets returned in search results)
          - The embedding vector (what gets compared during search)
          - Metadata (source file, headers — useful for showing context)

        Args:
            chunks: List of LangChain Document objects from the splitter.
        """
        if self.collection.count() > 0:
            print(f"⚠️  Collection already has {self.collection.count()} chunks. "
                  "Skipping to avoid duplicates. Use rebuild() to re-index.")
            return

        # Extract text from each chunk
        texts = [chunk.page_content for chunk in chunks]

        # Generate embeddings for all chunks at once (faster than one-by-one)
        vectors = self.embedder.embed(texts)

        # Prepare the data for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Clean metadata: ChromaDB only accepts str, int, float, bool values
        metadatas = []
        for chunk in chunks:
            clean_meta = {}
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_meta[key] = value
                else:
                    clean_meta[key] = str(value)
            metadatas.append(clean_meta)

        # Add everything to ChromaDB
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=vectors.tolist(),  # ChromaDB wants plain lists, not numpy
            metadatas=metadatas,
        )

        print(f"✅ Stored {len(chunks)} chunks in ChromaDB!")

    def search(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        """
        Find the most relevant chunks for a question.

        How it works:
          1. Convert the question to a vector
          2. Find the top_k stored vectors closest to it
          3. Return those chunks with their text and metadata

        Args:
            query:  The recruiter's question, e.g. "What is Seif's experience?"
            top_k:  How many chunks to return (default: 5)

        Returns:
            List of dicts, each with:
              - "text": the chunk content
              - "metadata": source file, headers, etc.
              - "distance": how far the vectors are (lower = more relevant)
        """
        # Embed the question
        query_vector = self.embedder.embed([query])

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=top_k,
        )

        # Format the results into a clean list
        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return formatted

    def rebuild(self, chunks):
        """
        Delete everything and re-index from scratch.
        Useful when you update your markdown files.
        """
        # Delete the existing collection
        self.client.delete_collection(config.COLLECTION_NAME)

        # Recreate it
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"description": "Seif's portfolio knowledge base"},
        )

        print("🔄 Collection cleared. Re-indexing...")
        self.add_chunks(chunks)


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from .document_loader import load_documents, split_into_chunks

    # Load and chunk documents
    docs = load_documents()
    chunks = split_into_chunks(docs)

    # Store them
    store = VectorStore()
    store.add_chunks(chunks)

    # Search!
    results = store.search("What technologies does Seif know?")
    print("\n🔍 Search results for 'What technologies does Seif know?':")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (distance: {r['distance']:.4f}) ---")
        print(f"Source: {r['metadata'].get('source', 'unknown')}")
        print(f"Text: {r['text'][:200]}...")
