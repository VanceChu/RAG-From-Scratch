# RAG Demo (Milvus + SentenceTransformers + OpenAI)

[中文 README](README.zh.md)

## Overview
This is a minimal end-to-end RAG demo that follows your architecture diagram:
- Stage 1 (Ingestion): parse documents -> chunk -> embed -> store in Milvus
- Stage 2 (Q&A): query embed -> search -> optional rerank -> prompt LLM -> answer + citations

## Architecture (Mapping to Diagram)
- Document parsing: `rag/parsers/*` converts PDF/DOCX/Markdown/text to Markdown.
- Chunking: `rag/chunking.py` splits Markdown by headings with overlap.
- Embedding: `rag/embeddings.py` generates vectors with SentenceTransformers.
- Vector store: `rag/vector_store.py` writes/reads Milvus (HNSW, inner product).
- Retrieval: `rag/retriever.py` embeds query then searches Milvus.
- Rerank (optional): `rag/rerank.py` uses a cross-encoder.
- Answer: `rag/answer.py` builds context and calls the LLM.
- CLI entrypoints: `scripts/ingest.py` and `scripts/ask.py`.

## Project Layout (File by File)
- `.gitignore`: ignores caches, virtual envs, local data, Milvus DB.
- `requirements.txt`: Python dependencies.
- `data/`: local data folder (ignored by git, except `.gitkeep`).
- `rag/__init__.py`: package exports.
- `rag/config.py`: default configuration + env overrides.
- `rag/parsers/__init__.py`: chooses parser based on file extension.
- `rag/parsers/pdf_parser.py`: PDF -> Markdown.
- `rag/parsers/docx_parser.py`: DOCX -> Markdown (headings/lists/paragraphs).
- `rag/parsers/md_parser.py`: Markdown/text passthrough.
- `rag/chunking.py`: heading-aware chunking + overlap.
- `rag/embeddings.py`: embedding model wrapper.
- `rag/vector_store.py`: Milvus collection + insert/search.
- `rag/ingest.py`: ingestion pipeline (parse -> chunk -> embed -> insert).
- `rag/retriever.py`: query embedding + vector search.
- `rag/rerank.py`: cross-encoder reranker.
- `rag/llm.py`: OpenAI Chat wrapper.
- `rag/answer.py`: prompt/context builder + LLM call.
- `scripts/ingest.py`: CLI for ingestion.
- `scripts/ask.py`: CLI for Q&A (answer + evidence).

## Quickstart (Conda)
> You said to use the `llm` conda environment for commands.

1) Install deps
```bash
conda run -n llm python -m pip install -r requirements.txt
```

2) Set your OpenAI key (auto-loads `.env`)
```bash
# Option A: .env
echo 'OPENAI_API_KEY=...' > .env

# Option B: shell export
export OPENAI_API_KEY="..."
```

3) Ingest documents
```bash
conda run -n llm python scripts/ingest.py --paths data/docs
```

4) Ask a question
```bash
conda run -n llm python scripts/ask.py --query "your question" --search-k 20 --top-k 5 --rerank
```

## Configuration (Env Vars)
- `RAG_COLLECTION` (default: `rag_chunks`)
- `MILVUS_URI` (default: `data/milvus.db`)
- `RAG_EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `RAG_RERANK_MODEL` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `RAG_OPENAI_MODEL` (default: `gpt-4o-mini`)
- `RAG_CHUNK_SIZE` (default: `800`)
- `RAG_CHUNK_OVERLAP` (default: `120`)
- `RAG_TOP_K` (default: `5`)
- `RAG_SEARCH_K` (default: `20`)
- `RAG_RERANK_TOP_K` (default: `5`)
- `RAG_BATCH_SIZE` (default: `64`)

## Notes
- `MILVUS_URI=data/milvus.db` uses Milvus Lite (local embedded DB via pymilvus).
- If you use a Milvus server, set `MILVUS_URI` to `http://localhost:19530` and start Milvus separately.
- The CLI prints evidence with `source/page/section` to make the answer verifiable.
