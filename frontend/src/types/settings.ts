export type Settings = {
  collection: string;
  collectionRaw: boolean;
  milvusUri: string;
  embeddingProvider: string;
  embeddingModel: string;
  embeddingBaseUrl: string;
  embeddingEndpoint: string;
  embeddingDim: number;
  indexType: string;
  indexNlist: number;
  indexM: number;
  indexEfConstruction: number;
  searchK: number;
  topK: number;
  enableBm25: boolean;
  enableSparse: boolean;
  fusion: string;
  hybridAlpha: number;
  rrfK: number;
  rerank: boolean;
  rerankModel: string;
  rerankTopK: number;
  openaiModel: string;
  historyTurns: number;
  stream: boolean;
  interactive: boolean;
  chunkSize: number;
  overlap: number;
  batchSize: number;
  reset: boolean;
  enableEvaluation: boolean;
  evalSampleRate: number;
};

export const DEFAULT_SETTINGS: Settings = {
  collection: "rag_chunks",
  collectionRaw: false,
  milvusUri: "data/index/milvus.db",
  embeddingProvider: "volcengine",
  embeddingModel: "ep-20260126203123-rhjcv",
  embeddingBaseUrl: "",
  embeddingEndpoint: "embeddings/multimodal",
  embeddingDim: 2048,
  indexType: "FLAT",
  indexNlist: 128,
  indexM: 8,
  indexEfConstruction: 64,
  searchK: 20,
  topK: 5,
  enableBm25: false,
  enableSparse: false,
  fusion: "weighted",
  hybridAlpha: 0.5,
  rrfK: 60,
  rerank: true,
  rerankModel: "cross-encoder/ms-marco-MiniLM-L-6-v2",
  rerankTopK: 5,
  openaiModel: "gpt-5.1",
  historyTurns: 3,
  stream: false,
  interactive: false,
  chunkSize: 800,
  overlap: 120,
  batchSize: 64,
  reset: false,
  enableEvaluation: false,
  evalSampleRate: 0,
};

export const PROVIDER_OPTIONS = [
  "volcengine",
  "openai",
  "openai-compatible",
  "openai-embeddings",
  "sentence-transformers",
  "ark",
];

export const INDEX_OPTIONS = ["FLAT", "IVF_FLAT", "HNSW", "AUTOINDEX"];
export const FUSION_OPTIONS = ["weighted", "rrf", "dense"];
