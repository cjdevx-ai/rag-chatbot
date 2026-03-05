"""
vector_store.py
───────────────
FAISS-backed vector store that maps embedding indices → Chunk objects.

Why FAISS?
  Facebook AI Similarity Search is the industry standard for fast
  approximate nearest-neighbour search. IndexFlatIP performs exact
  inner-product (= cosine, given L2-normalised vectors) search.
  For larger corpora you'd swap to IndexIVFFlat or HNSW.

Persistence:
  The FAISS index is saved to disk as `index.faiss` and chunk metadata
  as `chunks.npy`. On next run, load() skips re-embedding.
"""

from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .document_loader import Chunk


class VectorStore:
    """
    Stores dense vectors in a FAISS index and chunk objects in a parallel list.

    Usage
    -----
    store = VectorStore(dim=384)
    store.add(chunks, embeddings)         # index everything
    results = store.search(query_vec, k=3)
    store.save("./index")                 # persist to disk
    store = VectorStore.load("./index")   # restore
    """

    def __init__(self, dim: int):
        self.dim = dim
        # IndexFlatIP: exact inner-product search (cosine on L2-norm vectors)
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: List[Chunk] = []

    # ── building ──────────────────────────────────────────────────────────────

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their precomputed embeddings to the store.

        Parameters
        ----------
        chunks     : parallel list of Chunk objects
        embeddings : float32 array of shape (N, dim)
        """
        assert len(chunks) == len(embeddings), "chunks and embeddings must have the same length"
        assert embeddings.dtype == np.float32, "embeddings must be float32"

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        print(f"[VectorStore] Indexed {len(chunks)} chunks. Total: {self._index.ntotal}")

    # ── querying ──────────────────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Find the k most similar chunks to *query_vec*.

        Returns list of (Chunk, score) sorted by descending score.
        Score is inner-product (≈ cosine similarity), range [-1, 1].
        """
        if self._index.ntotal == 0:
            raise RuntimeError("VectorStore is empty. Call add() first.")

        # Ensure query_vec is 2D for FAISS (n_queries, dim)
        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_vec.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS uses -1 for "not found"
                continue
            results.append((self._chunks[idx], float(score)))

        return results   # already sorted descending by FAISS

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        print(f"[VectorStore] Saved to {directory}/")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        store = cls.__new__(cls)
        store.dim = index.d
        store._index = index
        store._chunks = chunks
        print(f"[VectorStore] Loaded {index.ntotal} vectors from {directory}/")
        return store

    @property
    def total(self) -> int:
        return self._index.ntotal
