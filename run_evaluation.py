"""Run the RAG benchmark against a JSONL labeled-query dataset.

Example: python run_evaluation.py eval_dataset.jsonl --output evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from evaluation import EvaluationExample, FaithfulnessJudge, RAGEvaluator
from generation import GroundedGenerator
from ingestion import DocumentIndexer
from reranker import CrossEncoderReranker
from retrieval import Retriever


def load_dataset(path: Path) -> list[EvaluationExample]:
    """Load JSON Lines with query and relevant_sources fields."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    examples: list[EvaluationExample] = []
    with path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row: dict[str, Any] = json.loads(line)
                query = row["query"]
                relevant_sources = row["relevant_sources"]
                if not isinstance(query, str) or not isinstance(relevant_sources, list):
                    raise ValueError("'query' must be a string and 'relevant_sources' must be a list")
                if any(not isinstance(source, str) for source in relevant_sources):
                    raise ValueError("Every relevant_sources entry must be a filename string")
                examples.append(
                    EvaluationExample(
                        query=query,
                        relevant_sources=relevant_sources,
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid dataset row at {path}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"No evaluation examples found in {path}")
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer faithfulness.")
    parser.add_argument("dataset", type=Path, help="JSONL file with query and relevant_sources fields")
    parser.add_argument("--k", type=int, default=3, help="Cutoff for retrieval metrics (default: 3)")
    parser.add_argument("--store", type=Path, default=Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "rag_documents"))
    parser.add_argument("--output", type=Path, help="Optional path to save the JSON report")
    parser.add_argument("--no-rerank", action="store_true", help="Disable sentence-transformers cross-encoder reranking")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY must be set")
    if args.k < 1:
        parser.error("--k must be at least 1")

    try:
        examples = load_dataset(args.dataset)
        indexer = DocumentIndexer(
            persist_directory=args.store,
            collection_name=args.collection,
        )
        reranker = None if args.no_rerank else CrossEncoderReranker()
        retriever = Retriever(
            indexer.vector_store,
            top_k=args.k,
            candidate_k=max(args.k * 3, args.k),
            reranker=reranker,
        )
        generator = GroundedGenerator(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"))
        judge = FaithfulnessJudge(model=os.getenv("RAG_EVAL_JUDGE_MODEL", "gpt-4o"))
        report = RAGEvaluator(retriever, generator, judge, k=args.k).evaluate(examples)
        encoded = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
            print(f"Saved evaluation report to {args.output}")
        print(encoded)
        return 0
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
