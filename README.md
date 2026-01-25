# RAG From Scratch

[中文 README](README.zh.md)

## What this repository is
A small, runnable RAG (Retrieval-Augmented Generation) demo.
It turns documents into a searchable knowledge base and answers questions with citations.

## What it does
- Ingests PDF/DOCX/Markdown/text/image files
- Parses PDFs with DeepDoc (layout, tables, figures, positions)
- Chunks content with a token-aware splitter and keeps positions
- Creates embeddings using SentenceTransformers
- Stores and searches vectors in Milvus (HNSW, inner product)
- Optionally reranks with a cross-encoder
- Generates answers with evidence snippets

## How it works (high level)
1) Parse documents into structured sections (text + positions + images)
2) Split into token-sized chunks and persist image crops
3) Store vectors in Milvus
4) Embed the query and retrieve top matches
5) (Optional) rerank results
6) Build a prompt and generate an answer

## Quickstart
1) Install dependencies
```bash
conda run -n llm python -m pip install -r requirements.txt
```

2) Set your OpenAI key (auto-loads `.env`)
```bash
echo "OPENAI_API_KEY=..." > .env
```

3) Ingest documents
```bash
conda run -n llm python scripts/ingest.py --paths data/docs --index-type FLAT
```
Re-running ingest without `--reset` performs incremental updates and persists state under `data/ingest_state`.

4) Ask questions (interactive by default)
```bash
conda run -n llm python scripts/ask.py --index-type FLAT --rerank
```
To ask a single question:
```bash
conda run -n llm python scripts/ask.py --query "your question" --index-type FLAT --rerank
```

## Configuration
Environment variables are loaded from `.env` at project root (if present).

- `RAG_COLLECTION` (default: `rag_chunks`)
- `MILVUS_URI` (default: `data/milvus.db`)
- `RAG_INDEX_TYPE` (default: `HNSW`, Lite supports `FLAT`, `IVF_FLAT`, `AUTOINDEX`)
- `RAG_INDEX_NLIST` (default: `128`, IVF only)
- `RAG_INDEX_M` (default: `8`, HNSW only)
- `RAG_INDEX_EF_CONSTRUCTION` (default: `64`, HNSW only)
- `RAG_EMBEDDING_PROVIDER` (default: `sentence-transformers`)
- `RAG_EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `RAG_EMBEDDING_DIM` (default: `0`, required for unknown OpenAI models)
- `RAG_RERANK_MODEL` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `RAG_OPENAI_MODEL` (default: `gpt-5.1`)
- `RAG_CHUNK_SIZE` (default: `800`, tokens)
- `RAG_CHUNK_OVERLAP` (default: `120`, tokens)
- `RAG_TOP_K` (default: `5`)
- `RAG_SEARCH_K` (default: `20`)
- `RAG_RERANK_TOP_K` (default: `5`)
- `RAG_BATCH_SIZE` (default: `64`)
- `RAG_STATE_DIR` (default: `data/ingest_state`)
- `RAG_IMAGE_DIR` (default: `data/chunk_images`, images stored under `<base>/<collection>/`)

Chunk images are automatically isolated per collection under `RAG_IMAGE_DIR/<collection>/`.

Milvus Lite uses a limited set of index types. For local `data/milvus.db`, set:
```bash
export RAG_INDEX_TYPE=FLAT
```

Use OpenAI embeddings (no local model download):
```bash
export RAG_EMBEDDING_PROVIDER=openai
export RAG_EMBEDDING_MODEL=text-embedding-3-small
```

## CLI Reference
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
- `--embedding-provider`: `sentence-transformers` or `openai`
- `--embedding-model`: embedding model name
- `--embedding-dim`: required for unknown OpenAI models
- `--milvus-uri`, `--collection`: Milvus settings
- `--reset`: drop collection and clear ingest state before ingest

By default, ingest is incremental: unchanged documents are skipped, and changed documents are refreshed.

### Ask
```bash
conda run -n llm python scripts/ask.py [--query "..."] [--rerank]
```
Key flags:
- `--query`: question text (optional; interactive mode starts if omitted)
- `--search-k`: retrieve this many chunks before rerank
- `--top-k`: number of chunks used in the final context
- `--rerank`: enable cross-encoder reranking
- `--stream`, `--no-stream`: stream tokens during generation (default: stream)
- `--interactive`: force interactive mode even when `--query` is provided
- `--history-turns`: number of recent turns included in the prompt
- `--index-type`: HNSW, IVF_FLAT, FLAT, AUTOINDEX
- `--index-nlist`: IVF_FLAT only
- `--index-m`: HNSW only
- `--index-ef-construction`: HNSW only
- `--embedding-provider`: `sentence-transformers` or `openai`
- `--embedding-model`: embedding model name
- `--embedding-dim`: required for unknown OpenAI models
- `--openai-model`: OpenAI chat model name

## Milvus Modes
- Default uses Milvus Lite via `MILVUS_URI=data/milvus.db`.
- For a server, set `MILVUS_URI=http://localhost:19530` and start Milvus separately.

## Project Structure (File by File)
- `.gitignore`: ignores caches, local data, and Milvus DB files.
- `.env`: local environment variables (not committed).
- `requirements.txt`: Python dependencies.
- `data/`: local data folder (ignored by git, except `.gitkeep`).
- `rag_core/__init__.py`: package exports.
- `rag_core/config.py`: default configuration + env overrides.
- `rag_core/ragflow_pipeline.py`: RagFlow-based parse + split pipeline.
- `rag_core/vendor/ragflow/`: vendored RagFlow/DeepDoc code.
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
