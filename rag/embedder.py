"""
embedder.py
───────────
Wraps sentence-transformers to produce L2-normalised dense vectors.

Why sentence-transformers?
  They produce semantically meaningful embeddings — texts with similar
  *meaning* are close in vector space, even if they share no words.
  This is vastly superior to bag-of-words / TF-IDF for retrieval.

Model default: all-MiniLM-L6-v2
  • 384-dimensional embeddings
  • ~22 MB download, runs comfortably on CPU
  • Excellent quality/speed trade-off for English text
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    """
    Thin wrapper around a SentenceTransformer model.

    All vectors are L2-normalised so that cosine similarity equals
    the dot product — this is required by FAISS IndexFlatIP.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[Embedder] Loading model '{model_name}' ...")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Ready. Embedding dimension: {self.dim}")

    # ── public ────────────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Returns
        -------
        np.ndarray of shape (N, dim), dtype float32, L2-normalised.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list")

        vectors = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # ← L2 normalise
        ).astype(np.float32)

        return vectors

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience method for a single string. Returns shape (1, dim)."""
        return self.embed([text])


# ── convenience API ──────────────────────────────────────────────────────────

_DEFAULT_EMBEDDER = None

def _get_embedder():
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = Embedder()
    return _DEFAULT_EMBEDDER

def embed_query(text: str) -> np.ndarray:
    """Embed a single query string using the default model."""
    # Return as 1D array for easier use in simple tests
    return _get_embedder().embed_one(text)[0]

def embed_chunks(texts: List[str]) -> np.ndarray:
    """Embed a list of text chunks using the default model."""
    return _get_embedder().embed(texts)
