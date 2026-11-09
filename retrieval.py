"""Similarity retrieval with an optional relevance reranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document
from langchain_chroma import Chroma


class RetrievalError(RuntimeError):
    """Raised when a vector retrieval operation fails."""


@dataclass(frozen=True)
class RetrievedChunk:
    document: Document
    score: float


Reranker = Callable[[list[RetrievedChunk], str], list[RetrievedChunk]]


class Retriever:
    """Retrieve candidates, optionally rerank them, and return the best top-k."""

    def __init__(
        self,
        vector_store: Chroma,
        *,
        top_k: int = 3,
        candidate_k: int | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if candidate_k is not None and candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        self.vector_store = vector_store
        self.top_k = top_k
        self.candidate_k = candidate_k if candidate_k is not None else max(top_k, 10 if reranker else top_k)
        self.reranker = reranker

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        try:
            matches = self.vector_store.similarity_search_with_relevance_scores(query, k=self.candidate_k)
            results = [RetrievedChunk(document=doc, score=float(score)) for doc, score in matches]
            if self.reranker is not None:
                results = self.reranker(results, query)
            return results[: self.top_k]
        except Exception as exc:
            raise RetrievalError(f"Similarity search failed: {exc}") from exc
