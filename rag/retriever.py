"""
retriever.py
────────────
Takes a user query string, embeds it, and returns the top-k most
relevant chunks from the VectorStore.

This is the "R" in RAG.
"""

from __future__ import annotations
from typing import List, Tuple

from .document_loader import Chunk
from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    """
    Retrieves the most semantically similar chunks for a given query.

    Parameters
    ----------
    embedder     : shared Embedder instance (avoid loading model twice)
    vector_store : populated VectorStore
    top_k        : number of chunks to return per query
    min_score    : discard chunks with similarity below this threshold
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 3,
        min_score: float = 0.2,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str) -> List[Tuple[Chunk, float]]:
        """
        Embed the query and return the top-k chunks.

        Returns
        -------
        List of (Chunk, cosine_score) sorted by descending score,
        filtered by min_score.
        """
        query_vec = self.embedder.embed_one(query)   # shape (1, dim)
        results = self.vector_store.search(query_vec, k=self.top_k)
        return [(chunk, score) for chunk, score in results if score >= self.min_score]

    def retrieve_texts(self, query: str) -> List[str]:
        """Convenience method — returns just the chunk texts."""
        return [chunk.text for chunk, _ in self.retrieve(query)]
