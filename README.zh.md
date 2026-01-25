# RAG From Scratch（从零搭建 RAG）

[English README](README.md)

## 这是什么
一个小而完整的 RAG（检索增强生成）Demo。
它会把你的文档变成可检索的知识库，并在回答问题时给出引用证据。

## 能做什么
- 支持 PDF/DOCX/Markdown/文本/图片入库
- PDF 使用 DeepDoc 解析（布局/表格/图片/位置）
- 使用 token 级切分并保留位置
- 使用 SentenceTransformers 生成向量
- 使用 Milvus 存储与检索（HNSW + 内积）
- 可选 cross-encoder 重排
- 输出答案与证据片段

## 工作流程（概览）
1) 解析文档为结构化片段（文本 + 位置 + 图片）
2) 进行 token 切分并保存图片裁剪
3) 向量写入 Milvus
4) 查询向量化并检索
5)（可选）重排结果
6) 拼接上下文并生成回答

## 快速开始
1) 安装依赖
```bash
python -m pip install -r requirements.txt
```

2) 设置 OpenAI Key（自动读取 `.env`）
```bash
echo "OPENAI_API_KEY=..." > .env
```

3) 入库
```bash
python scripts/ingest.py --paths data/docs
```

4) 问答
```bash
python scripts/ask.py --query "你的问题" --search-k 20 --top-k 5 --rerank
```

## 配置
环境变量会自动从项目根目录 `.env` 读取。

- `RAG_COLLECTION`（默认：`rag_chunks`）
- `MILVUS_URI`（默认：`data/milvus.db`）
- `RAG_INDEX_TYPE`（默认：`HNSW`，Lite 支持 `FLAT`、`IVF_FLAT`、`AUTOINDEX`）
- `RAG_INDEX_NLIST`（默认：`128`，仅 IVF）
- `RAG_INDEX_M`（默认：`8`，仅 HNSW）
- `RAG_INDEX_EF_CONSTRUCTION`（默认：`64`，仅 HNSW）
- `RAG_EMBEDDING_PROVIDER`（默认：`sentence-transformers`）
- `RAG_EMBEDDING_MODEL`（默认：`sentence-transformers/all-MiniLM-L6-v2`）
- `RAG_EMBEDDING_DIM`（默认：`0`，OpenAI 未知模型需要设置）
- `RAG_RERANK_MODEL`（默认：`cross-encoder/ms-marco-MiniLM-L-6-v2`）
- `RAG_OPENAI_MODEL`（默认：`gpt-4o-mini`）
- `RAG_CHUNK_SIZE`（默认：`800`，token）
- `RAG_CHUNK_OVERLAP`（默认：`120`，token）
- `RAG_TOP_K`（默认：`5`）
- `RAG_SEARCH_K`（默认：`20`）
- `RAG_RERANK_TOP_K`（默认：`5`）
- `RAG_BATCH_SIZE`（默认：`64`）
- `RAG_IMAGE_DIR`（默认：`data/chunk_images`）

Milvus Lite 只支持部分索引类型，本地 `data/milvus.db` 建议设置：
```bash
export RAG_INDEX_TYPE=FLAT
```

使用 OpenAI 向量化（避免本地模型下载）：
```bash
export RAG_EMBEDDING_PROVIDER=openai
export RAG_EMBEDDING_MODEL=text-embedding-3-small
```

## CLI 入口
### 入库
```bash
python scripts/ingest.py --paths <files_or_dirs> [--reset]
```
主要参数：
- `--paths`：待入库的文件或目录
- `--chunk-size`、`--overlap`：切分配置（token）
- `--index-type`：HNSW、IVF_FLAT、FLAT、AUTOINDEX
- `--index-nlist`：仅 IVF
- `--index-m`：仅 HNSW
- `--index-ef-construction`：仅 HNSW
- `--embedding-provider`：`sentence-transformers` 或 `openai`
- `--embedding-model`：向量模型
- `--embedding-dim`：OpenAI 未知模型需要设置
- `--milvus-uri`、`--collection`：Milvus 配置
- `--reset`：入库前删除集合

### 问答
```bash
python scripts/ask.py --query "..." [--rerank]
```
主要参数：
- `--query`：问题文本
- `--search-k`：检索候选数量
- `--top-k`：进入上下文的最终 chunk 数
- `--rerank`：是否重排
- `--index-type`：HNSW、IVF_FLAT、FLAT、AUTOINDEX
- `--index-nlist`：仅 IVF
- `--index-m`：仅 HNSW
- `--index-ef-construction`：仅 HNSW
- `--embedding-provider`：`sentence-transformers` 或 `openai`
- `--embedding-model`：向量模型
- `--embedding-dim`：OpenAI 未知模型需要设置
- `--openai-model`：OpenAI 模型

## Milvus 运行模式
- 默认使用 Milvus Lite（`MILVUS_URI=data/milvus.db`）。
- 如使用 Milvus 服务端，请设置 `MILVUS_URI=http://localhost:19530` 并自行启动服务。

## 项目结构（逐文件）
- `.gitignore`：忽略缓存、本地数据、Milvus DB。
- `.env`：本地环境变量（不提交）。
- `requirements.txt`：Python 依赖。
- `data/`：本地数据目录（git 忽略，保留 `.gitkeep`）。
- `rag_core/__init__.py`：包导出。
- `rag_core/config.py`：默认配置 + 环境变量覆盖。
- `rag_core/ragflow_pipeline.py`：基于 RagFlow 的解析 + 切分。
- `rag_core/vendor/ragflow/`：vendored RagFlow/DeepDoc 源码。
- `rag_core/parsers/`：旧 Markdown 解析（默认不使用）。
- `rag_core/chunking.py`：旧标题切分（默认不使用）。
- `rag_core/embeddings.py`：向量模型封装。
- `rag_core/vector_store.py`：Milvus 集合管理 + 插入/检索。
- `rag_core/ingest.py`：入库流水线（解析 -> 切分 -> 向量 -> 写入）。
- `rag_core/retriever.py`：查询向量化 + 检索。
- `rag_core/rerank.py`：cross-encoder 重排。
- `rag_core/llm.py`：OpenAI Chat 封装。
- `rag_core/answer.py`：上下文拼接 + LLM 调用。
- `scripts/ingest.py`：入库 CLI。
- `scripts/ask.py`：问答 CLI（答案 + 证据）。

## 说明
- 首次运行会下载 DeepDoc 的 OCR/布局模型，体积较大。
- CLI 会输出 `source/page/section` 证据片段，便于验证答案来源。
