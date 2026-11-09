"""Document loading, chunking, embedding, and Chroma indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


class IngestionError(RuntimeError):
    """Raised when a document cannot be loaded, chunked, or indexed."""


def _token_length(text: str) -> int:
    """Count tokens with the encoding used by OpenAI's embedding models."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as exc:
        raise IngestionError(f"Unable to tokenize text: {exc}") from exc


class DocumentIndexer:
    """Create and populate a Chroma vector store.

    Set ``persist_directory`` to enable persistent local storage; leave it as
    ``None`` for an in-memory store suitable for local testing.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        persist_directory: str | Path | None = None,
        collection_name: str = "rag_documents",
    ) -> None:
        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small", api_key=api_key
            )
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=(str(persist_directory) if persist_directory else None),
            )
        except Exception as exc:
            raise IngestionError(f"Could not initialize embedding/vector store: {exc}") from exc

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=_token_length,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

    def _index_documents(self, documents: Iterable[Document]) -> int:
        docs = list(documents)
        if not docs:
            raise IngestionError("No document content was supplied for indexing.")

        try:
            chunks = self.splitter.split_documents(docs)
            if not chunks:
                raise IngestionError("The supplied documents produced no text chunks.")

            # Stable, explicit metadata fields are useful for citations and tracing.
            for index, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = str(index)
                chunk.metadata.setdefault("source", "unknown")
            self.vector_store.add_documents(chunks)
            return len(chunks)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Failed to split, embed, or index documents: {exc}") from exc

    def index_directory(self, directory: str | Path) -> int:
        """Index all readable .txt and .pdf documents below ``directory``."""
        root = Path(directory).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Document directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Expected a directory, got: {root}")

        files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".txt", ".pdf"})
        if not files:
            raise FileNotFoundError(f"No .txt or .pdf documents found in {root}")

        documents: list[Document] = []
        errors: list[str] = []
        for path in files:
            try:
                loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path), encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = path.name
                documents.extend(loaded)
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        if not documents:
            detail = "; ".join(errors)
            raise IngestionError(f"No documents could be loaded. {detail}")
        count = self._index_documents(documents)
        if errors:
            # Successfully indexed readable files, while making skipped files visible.
            print("Warning: some files were skipped: " + "; ".join(errors))
        return count

    def index_text(self, text: str, *, source: str = "sample.txt") -> int:
        """Index a supplied string, useful for examples and tests."""
        if not text or not text.strip():
            raise IngestionError("Cannot index an empty text string.")
        return self._index_documents([Document(page_content=text, metadata={"source": source})])
