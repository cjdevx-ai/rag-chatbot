"""
pipeline.py
───────────
Assembles the full RAG pipeline:

  DocumentLoader → Embedder → VectorStore → Retriever → Generator

Also manages the chat history for multi-turn conversations and
provides a simple index caching mechanism (skip re-embedding on
subsequent runs if docs haven't changed).
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

from .document_loader import Chunk, DocumentLoader
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator


class RAGPipeline:
    """
    High-level interface for the entire RAG system.

    Parameters
    ----------
    docs_dir    : directory containing knowledge-base documents
    index_dir   : where to cache the FAISS index (None = no caching)
    embed_model : sentence-transformers model name
    top_k       : number of chunks to retrieve per query
    chunk_size  : characters per chunk
    chunk_overlap: overlap between chunks
    api_key     : Anthropic API key
    """

    def __init__(
        self,
        docs_dir: str,
        index_dir: str | None = ".rag_cache",
        embed_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        api_key: str | None = None,
    ):
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir) if index_dir else None
        self.top_k = top_k
        self.history: List[dict] = []

        # ── Step 1: load & chunk documents ───────────────────────────────────
        loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks: List[Chunk] = loader.load_directory(docs_dir)
        print(f"[Pipeline] Loaded {len(self.chunks)} chunks from {docs_dir}")

        # ── Step 2: create embedder ───────────────────────────────────────────
        self.embedder = Embedder(embed_model)

        # ── Step 3: build or load vector store ───────────────────────────────
        if self.index_dir and (self.index_dir / "index.faiss").exists():
            print("[Pipeline] Found existing index — loading from cache.")
            self.vector_store = VectorStore.load(self.index_dir)
        else:
            print("[Pipeline] Building index (embedding all chunks) ...")
            texts = [c.text for c in self.chunks]
            embeddings = self.embedder.embed(texts)
            self.vector_store = VectorStore(dim=self.embedder.dim)
            self.vector_store.add(self.chunks, embeddings)
            if self.index_dir:
                self.vector_store.save(self.index_dir)

        # ── Step 4: retriever & generator ────────────────────────────────────
        self.retriever = Retriever(self.embedder, self.vector_store, top_k=top_k)
        self.generator = Generator(api_key=api_key)

    # ── public ────────────────────────────────────────────────────────────────

    def query(self, user_input: str) -> Tuple[str, List[Tuple[Chunk, float]]]:
        """
        Run one RAG turn.

        Parameters
        ----------
        user_input : the user's question

        Returns
        -------
        (answer_text, retrieved_chunks_with_scores)
        """
        # Step 5: retrieve relevant chunks
        retrieved = self.retriever.retrieve(user_input)

        # Step 6: generate grounded answer
        answer = self.generator.generate(user_input, retrieved, self.history)

        # Step 7: update chat history (keep last 10 turns to avoid token overflow)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return answer, retrieved

    def reset_history(self) -> None:
        """Clear conversation history (start a new session)."""
        self.history = []
        print("[Pipeline] Conversation history cleared.")
