# Repository Guidelines

## Project Structure & Module Organization
- `rag_core/`: core library code (parsing, chunking, embeddings, retrieval, rerank, LLM, answer assembly).
- `rag_core/parsers/`: format-specific document parsers.
- `scripts/`: CLI entrypoints (`ingest.py`, `ask.py`).
- `data/`: local documents and Milvus Lite database; ignored by git.
- `requirements.txt`: Python dependencies; `.env` holds local config.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` installs runtime dependencies.
- `echo "OPENAI_API_KEY=..." > .env` sets required secrets (auto-loaded).
- `python scripts/ingest.py --paths data/docs` ingests files into Milvus Lite.
- `python scripts/ask.py --query "..." --search-k 20 --top-k 5 --rerank` queries and prints answers with evidence.
- Optional: `python scripts/ingest.py --paths data/docs --reset` to rebuild the collection.
- Optional: `MILVUS_URI=http://localhost:19530` to use a Milvus server instead of Lite.

## Coding Style & Naming Conventions
- Python with 4-space indentation and PEP 8 style.
- Use type hints as in existing modules.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants (see `rag_core/config.py`).
- CLI flags are kebab-case and defined via `argparse` in `scripts/`.

## Testing Guidelines
- No automated tests are present.
- For changes, run a manual smoke test: ingest a small set of docs and verify `ask.py` returns evidence snippets.
- If you add tests, place them in `tests/` and use `pytest` with `test_*.py` filenames.

## Commit & Pull Request Guidelines
- Commit messages must follow Conventional Commits (e.g., `feat: add rerank toggle`, `fix: handle empty docs`).
- Existing history predates this rule; new commits should comply strictly.
- Keep subjects short and imperative; keep commits focused by area (ingest, retrieval, rerank, docs).
- PRs should describe the dataset used, any config changes (do not commit `.env`), and include the exact CLI commands to reproduce results.
- If retrieval logic or prompts change, update `README.md` accordingly.

## Configuration & Security Notes
- Secrets live in `.env` and must not be committed.
- Model downloads and Milvus data live under `data/`; keep them local.
