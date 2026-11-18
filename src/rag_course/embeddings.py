"""
embeddings.py — Step 2: Turn text into vectors (numbers).

RAG Pipeline Flow:
  [Chunks] → EmbeddingModel.embed() → [Vectors (arrays of 384 numbers)]

What is an embedding?
  It's a way to represent text as a list of numbers so that
  similar texts have similar numbers. For example:
    "Python programming" → [0.12, -0.45, 0.78, ...]
    "coding in Python"   → [0.11, -0.44, 0.77, ...]  ← very similar!
    "banana smoothie"    → [0.95, 0.23, -0.56, ...]  ← very different

  We use these vectors to find which chunks are most relevant
  to a recruiter's question.

Why sentence-transformers?
  It runs 100% locally — no API key, no internet, no cost.
  The "all-MiniLM-L6-v2" model is small (80MB) but surprisingly good.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config


class EmbeddingModel:
    """Wraps a sentence-transformer model to generate text embeddings."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Download (first time only) and load the embedding model."""
        print(f"⏳ Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        dim = self.model.get_embedding_dimension()
        print(f"✅ Embedding model loaded! Each text becomes a {dim}-dim vector.")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Convert a list of texts into embeddings.

        Args:
            texts: List of strings to embed.
                   e.g. ["What is your experience?", "Tell me about MLOps"]

        Returns:
            numpy array of shape (len(texts), 384)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Something went wrong in __init__.")

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,  # Only show bar for many texts
        )

    @property
    def dimension(self) -> int:
        """How many numbers each embedding has (384 for MiniLM)."""
        return self.model.get_embedding_dimension()


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    embedder = EmbeddingModel()

    test_texts = [
        "What projects has Seif worked on?",
        "Tell me about the MLOps churn platform",
        "What is a banana?",
    ]

    vectors = embedder.embed(test_texts)
    print(f"\nEmbedded {len(test_texts)} texts → shape: {vectors.shape}")

    # Show that similar questions get similar vectors
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    print(f"\nSimilarity (projects ↔ MLOps):   {cosine_sim(vectors[0], vectors[1]):.3f}")
    print(f"Similarity (projects ↔ banana):  {cosine_sim(vectors[0], vectors[2]):.3f}")
    print("^ The first pair should be much higher — that's how retrieval works!")
