"""FastAPI wrapper for the modular RAG pipeline.

Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from uuid import uuid4

from conversation_memory import ConversationMemoryError, SQLiteConversationMemory
from generation import GenerationError, GroundedGenerator
from ingestion import DocumentIndexer, IngestionError
from retrieval import RetrievalError, Retriever
from reranker import CrossEncoderReranker


MAX_UPLOAD_BYTES = int(os.getenv("RAG_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
ALLOWED_SUFFIXES = {".txt", ".pdf"}


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)


class SourceChunk(BaseModel):
    source: str
    chunk_id: str
    score: float
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    conversation_id: str


class UploadResponse(BaseModel):
    indexed_files: list[str]
    chunk_count: int


def _build_services() -> tuple[DocumentIndexer, Retriever, GroundedGenerator, SQLiteConversationMemory]:
    """Initialize clients once for the lifetime of the API process."""
    persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
    collection = os.getenv("CHROMA_COLLECTION", "rag_documents")
    top_k = int(os.getenv("RAG_TOP_K", "3"))
    candidate_k = int(os.getenv("RAG_CANDIDATE_K", "10"))
    indexer = DocumentIndexer(
        persist_directory=persist_dir,
        collection_name=collection,
    )

    reranker_enabled = os.getenv("RAG_RERANK_ENABLED", "true").lower() in {"1", "true", "yes"}
    reranker = CrossEncoderReranker() if reranker_enabled else None
    retriever = Retriever(
        indexer.vector_store,
        top_k=top_k,
        candidate_k=max(candidate_k, top_k),
        reranker=reranker,
    )
    generator = GroundedGenerator(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"))
    memory = SQLiteConversationMemory(
        database_path=os.getenv("CONVERSATION_DB_PATH", "./conversation_memory.sqlite3"),
        max_messages=int(os.getenv("CONVERSATION_MAX_MESSAGES", "12")),
    )
    return indexer, retriever, generator, memory


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        (app.state.indexer, app.state.retriever, app.state.generator,
         app.state.conversation_memory) = await run_in_threadpool(_build_services)
    except Exception as exc:
        raise RuntimeError(f"Unable to initialize RAG service: {exc}") from exc
    yield


app = FastAPI(
    title="RAG API",
    description="Document-grounded multi-turn question answering with uploadable TXT and PDF sources.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def health() -> dict[str, str]:
    """Simple liveness endpoint; does not make an external API call."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest, request: Request) -> QueryResponse:
    """Answer a question using retrieved document context."""
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=422, detail="query must not be blank")

    try:
        conversation_id = payload.conversation_id or str(uuid4())
        memory: SQLiteConversationMemory = request.app.state.conversation_memory
        history = await run_in_threadpool(memory.get_history, conversation_id)
        generator: GroundedGenerator = request.app.state.generator
        search_query = await run_in_threadpool(generator.contextualize_query, query_text, history)
        retrieved = await run_in_threadpool(request.app.state.retriever.retrieve, search_query)
        answer = await run_in_threadpool(generator.answer, query_text, retrieved, history)
        await run_in_threadpool(memory.add_exchange, conversation_id, query_text, answer)
        sources = [
            SourceChunk(
                source=str(item.document.metadata.get("source", "unknown")),
                chunk_id=str(item.document.metadata.get("chunk_id", "unknown")),
                score=item.score,
                content=item.document.page_content,
            )
            for item in retrieved
        ]
        return QueryResponse(answer=answer, sources=sources, conversation_id=conversation_id)
    except RetrievalError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ConversationMemoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected RAG query failure") from exc


@app.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_documents(
    request: Request,
    files: Annotated[list[UploadFile], File(description="One or more .txt or .pdf documents")],
) -> UploadResponse:
    """Ingest uploaded documents into the service's shared vector collection."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    accepted: list[tuple[UploadFile, str]] = []
    seen_names: set[str] = set()
    for upload in files:
        original_name = Path(upload.filename or "").name
        suffix = Path(original_name).suffix.lower()
        if not original_name or suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"Unsupported file type for '{original_name or 'unnamed file'}'; only .txt and .pdf are accepted")
        if original_name in seen_names:
            raise HTTPException(status_code=400, detail=f"Duplicate filename in upload batch: '{original_name}'")
        seen_names.add(original_name)
        accepted.append((upload, original_name))

    try:
        with tempfile.TemporaryDirectory(prefix="rag-upload-") as temp_dir:
            upload_dir = Path(temp_dir)
            for upload, original_name in accepted:
                contents = await upload.read(MAX_UPLOAD_BYTES + 1)
                if len(contents) > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"'{original_name}' exceeds the {MAX_UPLOAD_BYTES}-byte upload limit",
                    )
                if not contents:
                    raise HTTPException(status_code=400, detail=f"'{original_name}' is empty")
                # Path(...).name above strips client-supplied directory traversal.
                (upload_dir / original_name).write_bytes(contents)

            try:
                count = await run_in_threadpool(request.app.state.indexer.index_directory, upload_dir)
            except (IngestionError, OSError) as exc:
                raise HTTPException(status_code=422, detail=f"Could not index uploaded files: {exc}") from exc

            # Report client-visible names (not temporary UUID-prefixed filenames).
            return UploadResponse(indexed_files=[name for _, name in accepted], chunk_count=count)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected document upload failure") from exc
    finally:
        for upload, _ in accepted:
            await upload.close()
