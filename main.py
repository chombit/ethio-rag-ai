"""End-to-end RAG demo: index sample text, retrieve, and answer a query."""

from __future__ import annotations

import os
import sys

from generation import GroundedGenerator
from ingestion import DocumentIndexer
from retrieval import Retriever
from reranker import CrossEncoderReranker


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: set OPENAI_API_KEY before running this example.", file=sys.stderr)
        return 2

    sample = (
        "Northstar Coffee was founded in 2018 in Portland. Its flagship product is the "
        "Aurora blend, a medium-roast coffee sourced from Guatemala and Colombia. "
        "The company offers free shipping on orders over $40."
    )
    question = "When was Northstar Coffee founded, and where?"

    try:
        indexer = DocumentIndexer()
        indexed = indexer.index_text(sample, source="northstar_sample.txt")
        reranker = CrossEncoderReranker()
        retriever = Retriever(
            indexer.vector_store,
            top_k=3,
            candidate_k=10,
            reranker=reranker,
        )
        context = retriever.retrieve(question)
        answer = GroundedGenerator().answer(question, context)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"RAG pipeline failed: {exc}", file=sys.stderr)
        return 1

    print(f"Indexed {indexed} chunk(s).")
    print(f"Question: {question}\nAnswer: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
