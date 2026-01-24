# RAG Demo（Milvus + SentenceTransformers + OpenAI）

[English README](README.md)

## 概览
这是一个最小可用的端到端 RAG Demo，流程与你的架构图一致：
- 阶段一（入库）：解析文档 -> 切分 chunk -> 向量化 -> 写入 Milvus
- 阶段二（问答）：查询向量化 -> 检索 -> 可选重排 -> 调用 LLM -> 返回答案 + 证据

## 架构说明（对应你的图）
- 文档解析：`rag/parsers/*` 将 PDF/DOCX/Markdown/文本统一为 Markdown。
- 切分：`rag/chunking.py` 按 Markdown 标题切分并设置 overlap。
- 向量化：`rag/embeddings.py` 使用 SentenceTransformers 生成向量。
- 向量库：`rag/vector_store.py` 负责 Milvus 写入与检索（HNSW + 内积）。
- 检索：`rag/retriever.py` 对 query 做向量化并检索。
- 重排（可选）：`rag/rerank.py` 用 cross-encoder 重新排序。
- 生成：`rag/answer.py` 拼接上下文并调用 LLM。
- CLI 入口：`scripts/ingest.py` 和 `scripts/ask.py`。

## 项目结构（逐文件说明）
- `.gitignore`：忽略缓存、虚拟环境、本地数据、Milvus 数据库。
- `requirements.txt`：Python 依赖。
- `data/`：本地数据目录（被 git 忽略，保留 `.gitkeep`）。
- `rag/__init__.py`：包导出。
- `rag/config.py`：默认配置 + 环境变量覆盖。
- `rag/parsers/__init__.py`：按后缀选择解析器。
- `rag/parsers/pdf_parser.py`：PDF -> Markdown。
- `rag/parsers/docx_parser.py`：DOCX -> Markdown（标题/列表/正文）。
- `rag/parsers/md_parser.py`：Markdown/文本直读。
- `rag/chunking.py`：标题感知切分 + overlap。
- `rag/embeddings.py`：向量模型封装。
- `rag/vector_store.py`：Milvus 集合管理 + 插入/检索。
- `rag/ingest.py`：入库流水线（解析 -> 切分 -> 向量 -> 写入）。
- `rag/retriever.py`：查询向量化 + 检索。
- `rag/rerank.py`：cross-encoder 重排。
- `rag/llm.py`：OpenAI Chat 封装。
- `rag/answer.py`：上下文拼接 + LLM 调用。
- `scripts/ingest.py`：入库 CLI。
- `scripts/ask.py`：问答 CLI（答案 + 证据）。

## 快速开始（Conda）
> 你要求终端命令使用 `llm` 环境。

1) 安装依赖
```bash
conda run -n llm python -m pip install -r requirements.txt
```

2) 设置 OpenAI Key（会自动加载 `.env`）
```bash
# 方式一：.env
echo 'OPENAI_API_KEY=...' > .env

# 方式二：环境变量
export OPENAI_API_KEY="..."
```

3) 入库
```bash
conda run -n llm python scripts/ingest.py --paths data/docs
```

4) 问答
```bash
conda run -n llm python scripts/ask.py --query "你的问题" --search-k 20 --top-k 5 --rerank
```

## 配置项（环境变量）
- `RAG_COLLECTION`（默认：`rag_chunks`）
- `MILVUS_URI`（默认：`data/milvus.db`）
- `RAG_EMBEDDING_MODEL`（默认：`sentence-transformers/all-MiniLM-L6-v2`）
- `RAG_RERANK_MODEL`（默认：`cross-encoder/ms-marco-MiniLM-L-6-v2`）
- `RAG_OPENAI_MODEL`（默认：`gpt-4o-mini`）
- `RAG_CHUNK_SIZE`（默认：`800`）
- `RAG_CHUNK_OVERLAP`（默认：`120`）
- `RAG_TOP_K`（默认：`5`）
- `RAG_SEARCH_K`（默认：`20`）
- `RAG_RERANK_TOP_K`（默认：`5`）
- `RAG_BATCH_SIZE`（默认：`64`）

## 说明
- `MILVUS_URI=data/milvus.db` 会使用 Milvus Lite（pymilvus 内置本地库）。
- 如使用 Milvus 服务端，请将 `MILVUS_URI` 设为 `http://localhost:19530` 并自行启动服务。
- CLI 会输出包含 `source/page/section` 的证据片段，便于验证答案来源。
