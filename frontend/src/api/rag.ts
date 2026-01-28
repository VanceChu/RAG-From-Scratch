import { http } from "./http";
import type { Settings } from "../types/settings";

type QueryResponse = {
  answer?: string;
  citations?: string[];
  trace_id?: string;
  evaluation?: Record<string, number>;
};

type QueryResult = {
  answer: string;
  citations: string[];
  traceId?: string;
  evaluation?: Record<string, number>;
};

function buildQueryPayload(query: string, settings: Settings): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    query,
    collection: settings.collection,
    collection_raw: settings.collectionRaw,
    milvus_uri: settings.milvusUri,
    embedding_provider: settings.embeddingProvider,
    embedding_model: settings.embeddingModel,
    embedding_dim: settings.embeddingDim,
    index_type: settings.indexType,
    index_nlist: settings.indexNlist,
    index_m: settings.indexM,
    index_ef_construction: settings.indexEfConstruction,
    search_k: settings.searchK,
    top_k: settings.topK,
    rerank: settings.rerank,
    rerank_model: settings.rerankModel,
    rerank_top_k: settings.rerankTopK,
    enable_sparse: settings.enableSparse,
    enable_bm25: settings.enableBm25,
    fusion: settings.fusion,
    rrf_k: settings.rrfK,
    hybrid_alpha: settings.hybridAlpha,
    stream: settings.stream,
    interactive: settings.interactive,
    openai_model: settings.openaiModel,
    history_turns: settings.historyTurns,
    enable_evaluation: settings.enableEvaluation,
    eval_sample_rate: settings.evalSampleRate,
  };

  if (settings.embeddingBaseUrl.trim()) {
    payload.embedding_base_url = settings.embeddingBaseUrl.trim();
  }
  if (settings.embeddingEndpoint.trim()) {
    payload.embedding_endpoint = settings.embeddingEndpoint.trim();
  }

  return payload;
}

function buildIngestFormData(files: FileList, settings: Settings, fallbackCollection: string) {
  const form = new FormData();
  Array.from(files).forEach((file) => form.append("files", file));

  form.append("collection", settings.collection || fallbackCollection);
  form.append("collection_raw", settings.collectionRaw ? "true" : "false");
  form.append("milvus_uri", settings.milvusUri);
  form.append("embedding_provider", settings.embeddingProvider);
  form.append("embedding_model", settings.embeddingModel);
  if (settings.embeddingBaseUrl.trim()) {
    form.append("embedding_base_url", settings.embeddingBaseUrl.trim());
  }
  if (settings.embeddingEndpoint.trim()) {
    form.append("embedding_endpoint", settings.embeddingEndpoint.trim());
  }
  form.append("embedding_dim", String(settings.embeddingDim));
  form.append("index_type", settings.indexType);
  form.append("index_nlist", String(settings.indexNlist));
  form.append("index_m", String(settings.indexM));
  form.append("index_ef_construction", String(settings.indexEfConstruction));
  form.append("chunk_size", String(settings.chunkSize));
  form.append("overlap", String(settings.overlap));
  form.append("batch_size", String(settings.batchSize));
  form.append("enable_sparse", settings.enableSparse ? "true" : "false");
  form.append("enable_bm25", settings.enableBm25 ? "true" : "false");
  form.append("reset", settings.reset ? "true" : "false");

  return form;
}

export async function queryRag(
  query: string,
  settings: Settings,
  apiBaseUrl?: string,
): Promise<QueryResult> {
  const payload = buildQueryPayload(query, settings);
  const response = await http.post<QueryResponse>("/query", payload, {
    baseURL: apiBaseUrl || undefined,
  });
  const data = response.data || {};
  return {
    answer: data.answer || "No answer returned.",
    citations: data.citations || [],
    traceId: data.trace_id,
    evaluation: data.evaluation,
  };
}

export async function ingestFiles(
  files: FileList,
  settings: Settings,
  fallbackCollection: string,
  apiBaseUrl?: string,
): Promise<void> {
  const form = buildIngestFormData(files, settings, fallbackCollection);
  await http.post("/ingest", form, {
    baseURL: apiBaseUrl || undefined,
  });
}
