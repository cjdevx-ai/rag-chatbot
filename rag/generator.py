"""
generator.py
────────────
Calls the Google Gemini API with a system prompt that includes
the retrieved context chunks.

This is the "G" in RAG — Augmented Generation.
"""

from __future__ import annotations
from typing import List, Tuple
import google.generativeai as genai

from .document_loader import Chunk


# ── prompt templates ──────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are a knowledgeable assistant with access to a curated knowledge base.

RETRIEVED CONTEXT
─────────────────
{context}

INSTRUCTIONS
────────────
• Ground your answers in the retrieved context above whenever possible.
• If the context fully answers the question, use it and cite the source filename.
• If the context is partially relevant, use what applies and note any gaps.
• If the context is irrelevant or missing, answer from your general knowledge
  and clearly say "This is not from the knowledge base."
• Be concise, accurate, and well-structured.
• Never fabricate facts.
"""

NO_CONTEXT_SYSTEM = """\
You are a knowledgeable assistant. No relevant documents were found in the
knowledge base for this query. Answer using your general knowledge and
clearly note that you are not drawing from retrieved documents.
"""


class Generator:
    """
    Wraps the Google Gemini client to generate RAG-augmented responses.

    Parameters
    ----------
    api_key   : Gemini API key (or set GEMINI_API_KEY env var)
    model     : Gemini model string
    max_tokens: maximum tokens in the response
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-flash-latest",
        max_tokens: int = 1024,
    ):
        if api_key:
            genai.configure(api_key=api_key)
        
        self.model_name = model
        self.generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": 0.1,  # Low temperature for more grounded answers
        }
        self.model = genai.GenerativeModel(model_name=model)

    # ── public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        retrieved: List[Tuple[Chunk, float]],
        history: List[dict],
    ) -> str:
        """
        Generate a response.
        """
        system_instruction = self._build_system(retrieved)
        
        # Re-initialize model with the dynamic system instruction for this turn
        # Gemini allows setting system_instruction at model level.
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction
        )

        # Convert history roles to Gemini format ("assistant" -> "model")
        gemini_history = []
        for turn in history:
            role = "model" if turn["role"] == "assistant" else turn["role"]
            gemini_history.append({"role": role, "parts": [turn["content"]]})

        # Start a chat session with history
        chat = model.start_chat(history=gemini_history)
        
        try:
            response = chat.send_message(query, generation_config=self.generation_config)
            return response.text
        except Exception as e:
            return f"Error during generation: {str(e)}"

    # ── private ───────────────────────────────────────────────────────────────

    def _build_system(self, retrieved: List[Tuple[Chunk, float]]) -> str:
        if not retrieved:
            return NO_CONTEXT_SYSTEM

        context_blocks = []
        for i, (chunk, score) in enumerate(retrieved, 1):
            block = (
                f"[{i}] Source: {chunk.source}  "
                f"(chunk {chunk.chunk_id}, similarity {score:.3f})\n"
                f"{chunk.text}"
            )
            context_blocks.append(block)

        context = "\n\n".join(context_blocks)
        return SYSTEM_TEMPLATE.format(context=context)
