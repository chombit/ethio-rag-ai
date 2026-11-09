# Modular LangChain RAG example

A small retrieval-augmented generation implementation using LangChain, OpenAI `text-embedding-3-small`, ChromaDB, a sentence-transformers cross-encoder, and `gpt-4o`. Chunk size and overlap are 500 and 50 **tokens**, respectively, measured with `tiktoken`'s `cl100k_base` encoding. Chroma is in-memory by default; specify a persistence directory to retain an index on disk.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python main.py
```

The demo indexes an in-memory sample string, retrieves up to 10 candidates, reranks them, and passes the best three chunks to the answer model. The default cross-encoder weights are downloaded on first use. No model/API calls are made if the API key is absent.

## Cross-encoder reranking

`Retriever` requests an expanded candidate pool (10 by default when a reranker is configured), then applies a reranker and returns only `top_k` results. The provided `CrossEncoderReranker` uses `cross-encoder/ms-marco-MiniLM-L-6-v2`; change `model_name` to select another compatible sentence-transformers cross-encoder. Cross-encoder scores replace the retrieval scores in the returned `RetrievedChunk` objects, so interpret them as reranking scores, not calibrated probabilities.

## Index a directory

`DocumentIndexer.index_directory(path)` reads `.txt` and `.pdf` files recursively, adding source filename and chunk ID metadata. For example:

```python
from ingestion import DocumentIndexer
from retrieval import Retriever
from reranker import CrossEncoderReranker
from generation import GroundedGenerator

indexer = DocumentIndexer(persist_directory="./chroma_data")
count = indexer.index_directory("./documents")
reranker = CrossEncoderReranker()
retrieved = Retriever(
    indexer.vector_store, top_k=3, candidate_k=10, reranker=reranker
).retrieve("What is in the documents?")
print(f"Indexed {count} chunks")
print(GroundedGenerator().answer("What is in the documents?", retrieved))
```

## FastAPI service

Start the service after installing requirements and setting `OPENAI_API_KEY`:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Upload one or more `.txt`/`.pdf` files (20 MiB maximum per file by default), then query the shared index:

```bash
curl -F "files=@./documents/handbook.pdf" http://localhost:8000/upload
curl -X POST http://localhost:8000/query \\
  -H 'Content-Type: application/json' \\
  -d '{"query":"What does the handbook say about leave?"}'
```

`POST /upload` returns indexed filenames and the number of chunks added. `POST /query` returns the grounded answer, retrieved source chunks/scores, and a `conversation_id`. Send that ID with later `/query` requests to continue the same conversation; omit it to start a conversation. `GET /healthz` is a liveness check. Configuration uses `CHROMA_PERSIST_DIRECTORY` (default `./chroma_data`), `CHROMA_COLLECTION`, `OPENAI_CHAT_MODEL`, `RAG_TOP_K`, `RAG_CANDIDATE_K`, `RAG_RERANK_ENABLED` (default `true`), `RAG_MAX_UPLOAD_BYTES`, `CONVERSATION_DB_PATH` (default `./conversation_memory.sqlite3`), and `CONVERSATION_MAX_MESSAGES` (default `12`). Recent history is stored in SQLite and survives service restarts. History is used only to resolve references and is explicitly not treated as evidence; answers remain grounded in retrieved document chunks. Uploaded files are stored only in a temporary directory; only their parsed content and metadata are retained in Chroma. The API process must have write access to its Chroma and SQLite paths.

## Streamlit frontend

With the API running in one terminal, launch the client in another:

```bash
streamlit run streamlit_app.py
```

Open the displayed Streamlit URL, set the FastAPI base URL in the sidebar if it is not `http://localhost:8000`, and use the tabs to upload `.txt`/`.pdf` files or ask questions. The UI keeps the server-issued conversation ID and renders the turn history; use **New conversation** to begin a separate chat. It displays answers, source chunks, and retrieval scores. It is a thin HTTP client and does not initialize another model or vector store.

## Evaluation

Prepare a JSONL benchmark whose `relevant_sources` entries exactly match source filenames stored in Chroma (e.g., `company_overview.pdf`). See [eval_dataset.example.jsonl](eval_dataset.example.jsonl). Index the benchmark documents into the same persistent collection, then run:

```bash
python run_evaluation.py eval_dataset.jsonl --k 3 --output evaluation_results.json
```

The runner reports `precision_at_k`, source-level `recall_at_k`, `mean_reciprocal_rank`, and LLM-judged `faithfulness` (fraction of extracted factual answer claims supported by retrieved context), plus per-example answers and unsupported claims. Retrieval metrics use filename relevance labels; every retrieved chunk whose source filename is labeled relevant counts as relevant. Faithfulness is RAGAS-style rather than the RAGAS library itself: it uses a structured-output LLM judge and is an estimate, so calibrate it against human-reviewed examples. `--no-rerank` disables the cross-encoder; `RAG_EVAL_JUDGE_MODEL` selects the judge model (default `gpt-4o`). Metrics assume the benchmark is representative and compare runs with the same corpus, labels, and `k`.

## Modules

- `ingestion.py`: text/PDF loading, token-aware recursive splitting, embeddings, and Chroma indexing.
- `retrieval.py`: expanded candidate retrieval, top-k selection, and reranker callback support.
- `reranker.py`: sentence-transformers cross-encoder for query-passage relevance scoring.
- `generation.py`: strict grounding prompt and insufficient-context fallback.
- `api.py`: FastAPI query, upload, and liveness endpoints.
- `streamlit_app.py`: browser-based upload and query client for the API.
- `conversation_memory.py`: bounded SQLite conversation store.
- `evaluation.py`: precision/recall/MRR and structured faithfulness evaluation.
- `run_evaluation.py`: JSONL benchmark command-line runner.
- `main.py`: sample-string end-to-end example.

API credentials are read from `OPENAI_API_KEY` by LangChain's OpenAI integrations; never commit secrets. In-memory stores are lost when the process exits. Persistent stores reopen from the same directory and collection name.
