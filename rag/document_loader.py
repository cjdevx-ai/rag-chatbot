"""
document_loader.py
──────────────────
Loads plain-text documents from a directory and splits them into
overlapping chunks suitable for embedding.

Why chunking?
  LLMs have limited context windows and embedding models work best on
  short, focused passages (~100-500 tokens). Chunking with overlap
  ensures sentences that span a boundary appear in at least one chunk.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    """A single text chunk with its provenance metadata."""
    text: str
    source: str          # filename
    chunk_id: int        # position within the source document
    char_start: int      # character offset in original doc
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(source={self.source!r}, id={self.chunk_id}, text={preview!r}...)"


class DocumentLoader:
    """
    Loads .txt and .md files from a directory and returns a flat list of Chunks.

    Parameters
    ----------
    chunk_size    : max characters per chunk
    chunk_overlap : characters shared between consecutive chunks
    """

    SUPPORTED = {".txt", ".md"}

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── public ────────────────────────────────────────────────────────────────

    def load_directory(self, path: str | Path) -> List[Chunk]:
        """Recursively load all supported files under *path*."""
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        chunks: List[Chunk] = []
        for filepath in sorted(directory.rglob("*")):
            if filepath.suffix.lower() in self.SUPPORTED:
                chunks.extend(self._load_file(filepath))

        if not chunks:
            raise ValueError(f"No supported files ({self.SUPPORTED}) found in {directory}")

        return chunks

    def load_file(self, path: str | Path) -> List[Chunk]:
        return self._load_file(Path(path))

    # ── private ───────────────────────────────────────────────────────────────

    def _load_file(self, filepath: Path) -> List[Chunk]:
        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return self._split(text, source=filepath.name)

    def _split(self, text: str, source: str) -> List[Chunk]:
        """
        Sliding-window character splitter.
        """
        chunks = []
        start = 0
        chunk_id = 0
        text_len = len(text)

        while start < text_len:
            # Determine initial end point
            end = min(start + self.chunk_size, text_len)
            
            # Extract basic snippet
            snippet = text[start:end]

            # Try to find a logical break point (sentence boundary) 
            # only if we aren't already at the end of the text.
            if end < text_len:
                # Look for a period followed by space/newline in the last half of the chunk
                last_period = snippet.rfind(". ")
                if last_period == -1:
                    last_period = snippet.rfind(".\n")
                
                # If we found a break point, adjust the snippet to end there
                if last_period > self.chunk_size // 2:
                    snippet = snippet[: last_period + 1]

            # Store the chunk
            actual_snippet = snippet.strip()
            if actual_snippet:
                chunks.append(
                    Chunk(
                        text=actual_snippet,
                        source=source,
                        chunk_id=chunk_id,
                        char_start=start,
                    )
                )

            # --- Critical Loop Protection & Movement ---
            # We must move the cursor forward. 
            # In a normal scenario, we move by (len(snippet) - overlap).
            move_by = len(snippet) - self.chunk_overlap
            
            # If the snippet is too small for overlap, we MUST still move forward 
            # by at least 1 or the full snippet length to avoid a hang.
            if move_by <= 0:
                start += len(snippet) if len(snippet) > 0 else 1
            else:
                start += move_by
            
            chunk_id += 1

        return chunks


def load_documents(path: str | Path, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Chunk]:
    """Helper function to quickly load a directory of documents."""
    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return loader.load_directory(path)
