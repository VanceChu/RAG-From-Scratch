# RAG From Scratch

[中文 README](README.zh.md)

## What this repository is
A small, runnable RAG (Retrieval-Augmented Generation) demo.
It turns documents into a searchable knowledge base and answers questions with citations.

## What it does
- Ingests PDF/DOCX/Markdown/text/image files
- Parses PDFs with DeepDoc (layout, tables, figures, positions)
- Chunks content with a token-aware splitter and keeps positions
- Creates embeddings using Volcengine/OpenAI-compatible APIs or SentenceTransformers
- Stores and searches vectors in Milvus (HNSW, inner product)
- Optionally builds a local BM25 index for lexical hybrid retrieval
- Reranks with a cross-encoder by default (disable with `--no-rerank`)
- Generates answers with evidence snippets

## How it works (high level)
1) Parse documents into structured sections (text + positions + images)
2) Split into token-sized chunks and persist image crops
3) Store vectors in Milvus
4) Embed the query and retrieve top matches
5) Rerank results (enabled by default)
6) Build a prompt and generate an answer

## Quickstart
1) Install dependencies
```bash
conda run -n llm python -m pip install -r requirements.txt
```

2) Set your API keys (auto-loads `.env`)
```bash
cat <<'EOF' > .env
OPENAI_API_KEY=...
VOLC_API_KEY=...
VOLC_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
EOF
```

3) Ingest documents
```bash
conda run -n llm python scripts/ingest.py --paths data/raw/docs --index-type FLAT
```
Re-running ingest without `--reset` performs incremental updates and persists state under `data/index/ingest_state`.

4) Ask questions (interactive by default)
```bash
conda run -n llm python scripts/ask.py --index-type FLAT
```
To ask a single question:
```bash
conda run -n llm python scripts/ask.py --query "your question" --index-type FLAT
```
Reranking is enabled by default; add `--no-rerank` to disable it.
Prefer guided setup? Use the interactive wizard flags below.

## Smoke Test (Verified)
This workflow was verified using the multimodal Ark endpoint ID for text-only embeddings.
Put a small PDF under `data/raw/test/`, then run:

```bash
conda run -n llm python scripts/ingest.py \
  --paths data/raw/test \
  --collection smoke_test \
  --collection-raw \
  --index-type FLAT \
  --enable-bm25 \
  --embedding-provider volcengine \
  --embedding-model ep-20260126203123-rhjcv \
  --embedding-endpoint embeddings/multimodal \
  --embedding-dim 2048 \
  --reset

conda run -n llm python scripts/ask.py \
  --query "What is the relative strength index?" \
  --collection smoke_test \
  --collection-raw \
  --index-type FLAT \
  --enable-bm25 \
  --embedding-provider volcengine \
  --embedding-model ep-20260126203123-rhjcv \
  --embedding-endpoint embeddings/multimodal \
  --embedding-dim 2048
```

Logs are written to `data/index/logs/` when you redirect output.

## Configuration
Environment variables are loaded from `.env` at project root (if present).

- `RAG_COLLECTION` (default: `rag_chunks`, treated as a base name)
- `MILVUS_URI` (default: `data/index/milvus.db`)
- `RAG_INDEX_TYPE` (default: `FLAT`, Lite supports `FLAT`, `IVF_FLAT`, `AUTOINDEX`)
- `RAG_INDEX_NLIST` (default: `128`, IVF only)
- `RAG_INDEX_M` (default: `8`, HNSW only)
- `RAG_INDEX_EF_CONSTRUCTION` (default: `64`, HNSW only)
- `RAG_EMBEDDING_PROVIDER` (default: `volcengine`)
- `RAG_EMBEDDING_MODEL` (default: `ep-20260126203123-rhjcv`)
- `RAG_EMBEDDING_API_KEY` (default: empty, for OpenAI-compatible providers)
- `RAG_EMBEDDING_BASE_URL` (default: empty, for OpenAI-compatible providers)
- `RAG_EMBEDDING_ENDPOINT` (default: `embeddings/multimodal`)
- `VOLC_API_KEY` (default: empty, provider-level key for Volcengine)
- `VOLC_API_BASE_URL` (default: `https://ark.cn-beijing.volces.com/api/v3`)
- `RAG_EMBEDDING_DIM` (default: `2048`)
- `RAG_SPARSE_PROVIDER` (default: `api`)
- `RAG_SPARSE_API_URL` (default: `https://api.siliconflow.cn/v1/embeddings`)
- `RAG_SPARSE_API_KEY` (default: empty)
- `RAG_RERANK_MODEL` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `RAG_OPENAI_MODEL` (default: `gpt-5.1`)
- `RAG_RERANK_ENABLED` (default: `true`)
- `RAG_HISTORY_TURNS` (default: `3`)
- `RAG_STREAM` (default: `true`)
- `RAG_INTERACTIVE` (default: `false`)
- `RAG_CHUNK_SIZE` (default: `800`, tokens)
- `RAG_CHUNK_OVERLAP` (default: `120`, tokens)
- `RAG_TOP_K` (default: `5`)
- `RAG_SEARCH_K` (default: `20`)
- `RAG_RERANK_TOP_K` (default: `5`)
- `RAG_BATCH_SIZE` (default: `64`)
- `RAG_ENABLE_BM25` (default: `false`)
- `RAG_ENABLE_SPARSE` (default: `false`)
- `RAG_FUSION` (default: `weighted`)
- `RAG_RRF_K` (default: `60`)
- `RAG_HYBRID_ALPHA` (default: `0.5`)
- `RAG_COLLECTION_RAW` (default: `false`)
- `RAG_RESET` (default: `false`)
- `RAG_STATE_DIR` (default: `data/index/ingest_state`)
- `RAG_IMAGE_DIR` (default: `data/index/chunk_images`, images stored under `<base>/<collection>/`)
- `RAG_BM25_DIR` (default: `data/index/bm25`)
- `LANGFUSE_ENABLED` (default: `false`)
- `LANGFUSE_PUBLIC_KEY` (default: empty)
- `LANGFUSE_SECRET_KEY` (default: empty)
- `LANGFUSE_HOST` (default: `https://cloud.langfuse.com`)
- `RAG_EVAL_OPENAI_MODEL` (default: `gpt-5.1`)
- `RAG_EVAL_OPENAI_API_KEY` (default: empty)
- `RAG_EVAL_OPENAI_BASE_URL` (default: empty)
- `RAG_EVAL_SAMPLE_RATE` (default: `0.0`)

Chunk images are automatically isolated per collection under `RAG_IMAGE_DIR/<collection>/`.
When the embedding provider/model differs from the defaults, the CLI automatically
derives a collection name from the base to avoid mixing embeddings. Use
`--collection-raw` to opt out.

Recommended local layout:
- `data/raw/`: source documents you ingest
- `data/index/`: local indexes and artifacts (Milvus Lite DB, ingest state, chunk images, ragflow models)

Milvus Lite uses a limited set of index types; the default is `FLAT` for local
`data/index/milvus.db`. If you switch to a Milvus server and want HNSW:
```bash
export RAG_INDEX_TYPE=HNSW
```

Use OpenAI embeddings (no local model download):
```bash
export RAG_EMBEDDING_PROVIDER=openai
export RAG_EMBEDDING_MODEL=text-embedding-3-small
```

Use Volcengine Ark embeddings (OpenAI-compatible):
```bash
export RAG_EMBEDDING_PROVIDER=volcengine
export RAG_EMBEDDING_MODEL=doubao-embedding-large-text-250515
export RAG_EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export RAG_EMBEDDING_API_KEY=...
export RAG_EMBEDDING_DIM=4096
```
You can also set `VOLC_API_KEY` / `VOLC_API_BASE_URL` at the provider level. When
`RAG_EMBEDDING_PROVIDER=volcengine`, the embedding client falls back to these
values if `RAG_EMBEDDING_API_KEY` / `RAG_EMBEDDING_BASE_URL` are unset.
For Ark multimodal endpoints (endpoint IDs like `ep-...`), set:
```bash
export RAG_EMBEDDING_ENDPOINT=embeddings/multimodal
```

## CLI Reference
### Wizard
```bash
conda run -n llm python scripts/ingest.py --wizard
conda run -n llm python scripts/ask.py --wizard
```

### Ingest
```bash
conda run -n llm python scripts/ingest.py --paths <files_or_dirs> [--reset]
```
Key flags:
- `--paths`: files or directories to ingest
- `--chunk-size`, `--overlap`: chunking behavior (tokens)
- `--index-type`: HNSW, IVF_FLAT, FLAT, AUTOINDEX
- `--index-nlist`: IVF_FLAT only
- `--index-m`: HNSW only
- `--index-ef-construction`: HNSW only
- `--embedding-provider`: `sentence-transformers`, `openai`, or `volcengine`
- `--embedding-model`: embedding model name
- `--embedding-base-url`: base URL for OpenAI-compatible providers
- `--embedding-endpoint`: embedding endpoint path (e.g. `embeddings/multimodal`)
- `--embedding-dim`: required for unknown OpenAI models
- `--milvus-uri`: Milvus settings
- `--collection`: base collection name (auto-suffixed for non-default embedding models)
- `--collection-raw`: disable model-based collection suffix
- `--enable-bm25`: build a local BM25 index for lexical hybrid retrieval
- `--enable-sparse`: generate sparse vectors via API for hybrid search
- `--reset`: drop collection and clear ingest state before ingest

By default, ingest is incremental: unchanged documents are skipped, and changed documents are refreshed.
If you want BM25 hybrid retrieval, run ingest with `--enable-bm25` to build the local index.

### Ask
```bash
conda run -n llm python scripts/ask.py [--query "..."] [--no-rerank]
```
Key flags:
- `--query`: question text (optional; interactive mode starts if omitted)
- `--search-k`: retrieve this many chunks before rerank
- `--top-k`: number of chunks used in the final context
- `--rerank`/`--no-rerank`: enable or disable cross-encoder reranking (default: enabled)
- `--stream`, `--no-stream`: stream tokens during generation (default: stream)
- `--interactive`: force interactive mode even when `--query` is provided
- `--history-turns`: number of recent turns included in the prompt
- `--index-type`: HNSW, IVF_FLAT, FLAT, AUTOINDEX
- `--index-nlist`: IVF_FLAT only
- `--index-m`: HNSW only
- `--index-ef-construction`: HNSW only
- `--embedding-provider`: `sentence-transformers`, `openai`, or `volcengine`
- `--embedding-model`: embedding model name

## LangFuse + RAGAS (Observability + Evaluation)
### 1) Enable LangFuse tracing
Set the following in `.env`:
```bash
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```
Start the API server and make a request. The response includes `trace_id` which is also shown in the UI:
```bash
conda run -n llm python scripts/api.py
```

### 2) Online evaluation (RAGAS scores in API responses)
You can request a single evaluation or enable sampling:
```json
{
  "query": "什么是RAG?",
  "enable_evaluation": true
}
```
Or sampling:
```json
{
  "query": "什么是RAG?",
  "eval_sample_rate": 0.1
}
```
Scores are returned in `evaluation` and also written into LangFuse as scores + trace metadata.

### 3) Batch evaluation CLI (RAGAS)
Prepare a dataset (JSON or JSONL) and run:
```bash
conda run -n llm python scripts/evaluate.py \
  --dataset data/eval/sample_dataset.jsonl \
  --output data/eval/results.json \
  --output-csv data/eval/results.csv \
  --output-html data/eval/report.html \
  --upload-langfuse
```
The HTML report provides a quick visual summary, while CSV/JSON are easy to analyze.
- `--embedding-base-url`: base URL for OpenAI-compatible providers
- `--embedding-endpoint`: embedding endpoint path (e.g. `embeddings/multimodal`)
- `--embedding-dim`: required for unknown OpenAI models
- `--collection`: base collection name (auto-suffixed for non-default embedding models)
- `--collection-raw`: disable model-based collection suffix
- `--openai-model`: OpenAI chat model name
- `--enable-bm25`: enable BM25 hybrid retrieval (uses local BM25 index)
- `--enable-sparse`: enable sparse embeddings for hybrid search
- `--fusion`: fusion strategy when BM25 is enabled: `weighted`, `rrf`, or `dense`
- `--rrf-k`: RRF k parameter (only for `--fusion rrf`)
- `--hybrid-alpha`: dense weight for hybrid search (0.0 = BM25/sparse, 1.0 = dense)

When BM25 is enabled, the pipeline can fuse dense + BM25 results using
`--fusion weighted` (score normalization + `--hybrid-alpha`) or `--fusion rrf`
(rank fusion with `--rrf-k`). Use `--fusion dense` to ignore BM25 even if enabled.
If you pass both `--enable-bm25` and `--enable-sparse`, BM25 takes precedence.

## API Server (FastAPI)
Run the lightweight API for the frontend:

```bash
conda run -n llm python scripts/api.py
# or: conda run -n llm uvicorn scripts.api:app --reload
```

Endpoints:
- `POST /ingest` (multipart form-data): files + collection config
- `POST /query` (JSON): query + collection config

## Lightweight Frontend (React + TypeScript)
The lightweight UI lives under `frontend/` and provides a chat workspace plus file upload.
It runs in mock mode by default.

```bash
cd frontend
npm install
npm run dev
```

Environment variables (optional):
- `VITE_API_BASE_URL` (default: `http://localhost:8000`)
- `VITE_USE_MOCK` (default: `true`; set to `false` to call the API)

## Milvus Modes
- Default uses Milvus Lite via `MILVUS_URI=data/index/milvus.db`.
- For a server, set `MILVUS_URI=http://localhost:19530` and start Milvus separately.

## Project Structure (File by File)
- `.gitignore`: ignores caches, local data, and Milvus DB files.
- `.env`: local environment variables (not committed).
- `requirements.txt`: Python dependencies.
- `data/raw/`: source documents for ingestion (ignored by git).
- `data/index/`: local indexes/artifacts (Milvus Lite, ingest state, chunk images, ragflow models; ignored by git).
- `data/index/bm25/`: local BM25 index per collection (ignored by git).
- `rag_core/__init__.py`: package exports.
- `rag_core/config.py`: default configuration + env overrides.
- `rag_core/ragflow_pipeline.py`: RagFlow-based parse + split pipeline.
- `rag_core/vendor/ragflow_slim/`: vendored RagFlow/DeepDoc code (slim subset).
- `rag_core/parsers/`: legacy Markdown parsers (unused by default).
- `rag_core/chunking.py`: legacy heading-based chunking (unused by default).
- `rag_core/embeddings.py`: embedding model wrapper.
- `rag_core/vector_store.py`: Milvus collection + insert/search.
- `rag_core/ingest_state.py`: persistent ingest state for incremental updates.
- `rag_core/ingest.py`: ingestion pipeline (parse -> chunk -> embed -> insert).
- `rag_core/retriever.py`: query embedding + vector search.
- `rag_core/rerank.py`: cross-encoder reranker.
- `rag_core/llm.py`: OpenAI chat wrapper.
- `rag_core/answer.py`: prompt/context builder + LLM call.
- `scripts/ingest.py`: CLI for ingestion.
- `scripts/ask.py`: CLI for Q&A (interactive loop + streaming answers + evidence).

## Notes
- First run will download OCR/layout models used by DeepDoc (can be large).
- The CLI prints evidence with `source/page/section` so answers are verifiable.
