"""Optional cross-encoder relevance reranking for retrieved passages.

The model is loaded only when the reranker is instantiated. The default model
is a compact MS MARCO model; the first use may download its weights.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from retrieval import RetrievedChunk


class RerankingError(RuntimeError):
    """Raised when the cross-encoder cannot score candidate passages."""


class CrossEncoderReranker:
    """Rescore (query, passage) pairs and return candidates best-first."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        try:
            self.model = CrossEncoder(
                model_name,
                device=device,
                max_length=max_length,
            )
        except Exception as exc:
            raise RerankingError(f"Unable to load cross-encoder '{model_name}': {exc}") from exc
        self.batch_size = batch_size

    def __call__(self, candidates: list[RetrievedChunk], query: str) -> list[RetrievedChunk]:
        if not candidates:
            return []
        try:
            pairs = [(query, item.document.page_content) for item in candidates]
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            if len(scores) != len(candidates):
                raise RerankingError("Cross-encoder returned a different number of scores than candidates.")
            rescored = [
                RetrievedChunk(document=item.document, score=float(score))
                for item, score in zip(candidates, scores, strict=True)
            ]
            return sorted(rescored, key=lambda item: item.score, reverse=True)
        except RerankingError:
            raise
        except Exception as exc:
            raise RerankingError(f"Cross-encoder scoring failed: {exc}") from exc
