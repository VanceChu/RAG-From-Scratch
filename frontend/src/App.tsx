import { useMemo, useState } from "react";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: string[];
  timestamp: string;
};

type KnowledgeBase = {
  id: string;
  name: string;
  documents: number;
  updatedAt: string;
};

type Settings = {
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
};

type InfoTipProps = {
  text: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const USE_MOCK = import.meta.env.VITE_USE_MOCK !== "false";

const INITIAL_KBS: KnowledgeBase[] = [
  {
    id: "rag_chunks",
    name: "Default Workspace",
    documents: 1,
    updatedAt: "Today"
  },
  {
    id: "strategy_lab",
    name: "Strategy Lab",
    documents: 14,
    updatedAt: "Yesterday"
  }
];

const DEFAULT_SETTINGS: Settings = {
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
  reset: false
};

const PROVIDER_OPTIONS = [
  "volcengine",
  "openai",
  "openai-compatible",
  "openai-embeddings",
  "sentence-transformers",
  "ark"
];

const INDEX_OPTIONS = ["FLAT", "IVF_FLAT", "HNSW", "AUTOINDEX"];
const FUSION_OPTIONS = ["weighted", "rrf", "dense"];

const INITIAL_MESSAGES: Message[] = [
  {
    id: "m-1",
    role: "assistant",
    content:
      "Ask a question or upload a file. I will answer with citations once the API is connected.",
    citations: ["Getting started"],
    timestamp: "Just now"
  }
];

function formatNow(): string {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function InfoTip({ text }: InfoTipProps) {
  return (
    <span
      className="info-tip"
      tabIndex={0}
      aria-label={text}
      onClick={(event) => event.stopPropagation()}
      onMouseDown={(event) => event.stopPropagation()}
    >
      <span className="info-icon">i</span>
      <span className="tooltip">{text}</span>
    </span>
  );
}

function FieldLabel({ text, tip }: { text: string; tip: string }) {
  return (
    <div className="field-label">
      <span>{text}</span>
      <InfoTip text={tip} />
    </div>
  );
}

export default function App() {
  const [knowledgeBases] = useState<KnowledgeBase[]>(INITIAL_KBS);
  const [selectedKb, setSelectedKb] = useState<KnowledgeBase>(INITIAL_KBS[0]);
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [apiBaseUrl, setApiBaseUrl] = useState(API_BASE);
  const [useMock, setUseMock] = useState(USE_MOCK);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);

  const chatTitle = useMemo(() => {
    return selectedKb ? `Conversation · ${selectedKb.name}` : "Conversation";
  }, [selectedKb]);

  const canSend = input.trim().length > 0 && !isSending;

  const updateSetting = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleResetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
    setApiBaseUrl(API_BASE);
    setUseMock(USE_MOCK);
  };

  const handleSend = async () => {
    if (!canSend) return;

    const trimmed = input.trim();
    const userMessage: Message = {
      id: `m-${Date.now()}`,
      role: "user",
      content: trimmed,
      timestamp: formatNow()
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsSending(true);

    try {
      const response = useMock
        ? await mockAnswer(trimmed)
        : await fetchAnswer(trimmed, settings, apiBaseUrl);

      const assistantMessage: Message = {
        id: `m-${Date.now()}-a`,
        role: "assistant",
        content: response.answer,
        citations: response.citations,
        timestamp: formatNow()
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const assistantMessage: Message = {
        id: `m-${Date.now()}-err`,
        role: "assistant",
        content:
          "I could not reach the backend. Start the API server or set VITE_USE_MOCK=false.",
        citations: ["Connection error"],
        timestamp: formatNow()
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } finally {
      setIsSending(false);
    }
  };

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setUploadStatus("Uploading files...");

    try {
      if (useMock) {
        await new Promise((resolve) => setTimeout(resolve, 1200));
      } else {
        const form = buildIngestFormData(files, settings, selectedKb.id);
        const res = await fetch(`${apiBaseUrl}/ingest`, {
          method: "POST",
          body: form
        });

        if (!res.ok) {
          throw new Error("Upload failed");
        }
      }
      setUploadStatus(`Uploaded ${files.length} file(s) successfully.`);
    } catch (error) {
      setUploadStatus("Upload failed. Please retry.");
    }
  };

  const handleClear = () => {
    setMessages(INITIAL_MESSAGES);
  };

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <span className="wordmark">OpenAI RAG</span>
          <span className="brand-sub">Research Console</span>
        </div>
        <nav className="nav">
          <button type="button" className="nav-link">
            Workspace
          </button>
          <button type="button" className="nav-link">
            Ingest
          </button>
          <button type="button" className="nav-link">
            Ask
          </button>
        </nav>
        <div className="status">
          <span className="status-dot" />
          {useMock ? "Mock mode" : "Live API"}
        </div>
      </header>

      <main className="content">
        <section className="hero fade-in">
          <p className="eyebrow">RAG Workspace</p>
          <h1>Grounded answers for your knowledge base.</h1>
          <p className="lead">
            Upload documents, retrieve context, and respond with traceable evidence.
          </p>
          <div className="hero-meta">
            <div>
              <span className="meta-label">Active collection</span>
              <strong>{selectedKb.id}</strong>
            </div>
            <div>
              <span className="meta-label">Mode</span>
              <strong>{useMock ? "Mock" : "Live API"}</strong>
            </div>
          </div>
        </section>

        <section className="grid">
          <section className="panel conversation fade-in delay-1">
            <div className="panel-header">
              <div>
                <p className="title">{chatTitle}</p>
                <p className="subtitle">
                  Evidence-backed answers from {selectedKb.name}.
                </p>
              </div>
              <button type="button" className="btn ghost" onClick={handleClear}>
                Clear
              </button>
            </div>

            <div className="chat-body">
              {messages.map((message) => (
                <article
                  key={message.id}
                  className={message.role === "user" ? "bubble user" : "bubble"}
                >
                  <div className="bubble-meta">
                    <span>{message.role === "user" ? "You" : "Assistant"}</span>
                    <span>{message.timestamp}</span>
                  </div>
                  <p>{message.content}</p>
                  {message.citations && (
                    <div className="citations">
                      {message.citations.map((cite) => (
                        <span key={cite} className="chip">
                          {cite}
                        </span>
                      ))}
                    </div>
                  )}
                </article>
              ))}
              {isSending && (
                <article className="bubble typing">
                  <div className="bubble-meta">
                    <span>Assistant</span>
                    <span>Thinking</span>
                  </div>
                  <p>Retrieving evidence and drafting the response.</p>
                </article>
              )}
            </div>

            <div className="chat-input">
              <textarea
                placeholder="Ask a question about this knowledge base..."
                value={input}
                onChange={(event) => setInput(event.target.value)}
                rows={3}
              />
              <div className="actions">
                <div className="helper">
                  <span>Shift+Enter for newline</span>
                  <span>Collection: {selectedKb.id}</span>
                </div>
                <button type="button" className="btn primary" onClick={handleSend} disabled={!canSend}>
                  {isSending ? "Sending..." : "Send"}
                </button>
              </div>
            </div>
          </section>

          <aside className="column">
            <section className="panel fade-in delay-2">
              <div className="panel-header">
                <h2>Knowledge Bases</h2>
                <button type="button" className="btn ghost">
                  New
                </button>
              </div>
              <div className="kb-list">
                {knowledgeBases.map((kb) => (
                  <button
                    key={kb.id}
                    type="button"
                    className={kb.id === selectedKb.id ? "kb-item active" : "kb-item"}
                    onClick={() => setSelectedKb(kb)}
                  >
                    <div>
                      <p className="kb-name">{kb.name}</p>
                      <p className="kb-meta">
                        {kb.documents} docs · {kb.updatedAt}
                      </p>
                    </div>
                    <span className="kb-tag">{kb.id}</span>
                  </button>
                ))}
              </div>
            </section>

            <section className="panel fade-in delay-3">
              <h2>Upload Documents</h2>
              <p className="muted">
                PDFs, DOCX, MD, TXT, PNG. The pipeline will parse and index them.
              </p>
              <label className="upload-area">
                <input
                  type="file"
                  multiple
                  onChange={(event) => handleUpload(event.target.files)}
                />
                <span>Drop files here or click to upload</span>
              </label>
              {uploadStatus && <p className="status-text">{uploadStatus}</p>}
            </section>

            <section className="panel fade-in delay-3">
              <div className="panel-header">
                <h2>Controls</h2>
                <button type="button" className="btn ghost" onClick={handleResetSettings}>
                  Reset
                </button>
              </div>

              <div className="settings">
                <details className="accordion" open>
                  <summary>Connection</summary>
                  <div className="form-grid">
                    <label className="field">
                      <FieldLabel
                        text="API base URL"
                        tip="Definition: Base URL of the FastAPI server.\nImpact: All /query and /ingest calls go here; wrong URL = failed requests.\nTypical: http://localhost:8000.\nUnits: URL."
                      />
                      <input
                        className="input"
                        value={apiBaseUrl}
                        onChange={(event) => setApiBaseUrl(event.target.value)}
                        placeholder="http://localhost:8000"
                      />
                    </label>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={useMock}
                        onChange={(event) => setUseMock(event.target.checked)}
                      />
                      <span className="toggle-text">Use mock answers</span>
                      <InfoTip text="Definition: Local mock mode.\nImpact: No backend calls; no cost; for UI testing.\nTypical: off in production.\nUnits: boolean." />
                    </label>
                  </div>
                </details>

                <details className="accordion" open>
                  <summary>Collection & Storage</summary>
                  <div className="form-grid">
                    <label className="field">
                      <FieldLabel
                        text="Collection name"
                        tip="Definition: Base collection name for vector + BM25 storage.\nImpact: Controls where data is stored; dimension mismatch will error.\nTypical: rag_chunks (auto-suffix when collection_raw=false).\nUnits: string."
                      />
                      <input
                        className="input"
                        value={settings.collection}
                        onChange={(event) => updateSetting("collection", event.target.value)}
                      />
                    </label>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={settings.collectionRaw}
                        onChange={(event) => updateSetting("collectionRaw", event.target.checked)}
                      />
                      <span className="toggle-text">Use collection name as-is</span>
                      <InfoTip text="Definition: Use collection name exactly as typed.\nImpact: Avoids auto-suffix; risk dimension mismatch when switching models.\nTypical: off.\nUnits: boolean." />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="Milvus URI"
                        tip="Definition: Milvus Lite DB path or Milvus server URI.\nImpact: Local path = file DB; remote URI = networked server (latency/cost).\nTypical: data/index/milvus.db or http://localhost:19530.\nUnits: path/URL."
                      />
                      <input
                        className="input"
                        value={settings.milvusUri}
                        onChange={(event) => updateSetting("milvusUri", event.target.value)}
                      />
                    </label>
                  </div>
                </details>

                <details className="accordion">
                  <summary>Embeddings</summary>
                  <div className="form-grid">
                    <label className="field">
                      <FieldLabel
                        text="Provider"
                        tip="Definition: Embedding backend/provider.\nImpact: Must match API keys; affects quality, latency, and cost.\nTypical: volcengine, openai, sentence-transformers.\nUnits: string."
                      />
                      <input
                        className="input"
                        list="provider-options"
                        value={settings.embeddingProvider}
                        onChange={(event) => updateSetting("embeddingProvider", event.target.value)}
                      />
                    </label>
                    <datalist id="provider-options">
                      {PROVIDER_OPTIONS.map((option) => (
                        <option key={option} value={option} />
                      ))}
                    </datalist>
                    <label className="field">
                      <FieldLabel
                        text="Model"
                        tip="Definition: Embedding model name or endpoint ID.\nImpact: Determines vector quality and dimension; may change cost.\nTypical: ep-xxxx, text-embedding-3-small, bge-m3.\nUnits: string."
                      />
                      <input
                        className="input"
                        value={settings.embeddingModel}
                        onChange={(event) => updateSetting("embeddingModel", event.target.value)}
                      />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="Base URL"
                        tip="Definition: Base URL for OpenAI-compatible embeddings API.\nImpact: Overrides provider default; wrong URL fails; may route to self-hosted.\nTypical: blank (use provider default).\nUnits: URL."
                      />
                      <input
                        className="input"
                        value={settings.embeddingBaseUrl}
                        onChange={(event) => updateSetting("embeddingBaseUrl", event.target.value)}
                        placeholder="Leave blank for default"
                      />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="Endpoint path"
                        tip="Definition: Endpoint path appended to base URL.\nImpact: Must match provider API route; wrong path fails.\nTypical: embeddings/multimodal.\nUnits: URL path."
                      />
                      <input
                        className="input"
                        value={settings.embeddingEndpoint}
                        onChange={(event) => updateSetting("embeddingEndpoint", event.target.value)}
                      />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="Embedding dimension"
                        tip="Definition: Embedding vector dimension.\nImpact: Must match model output and existing collection; higher dims increase storage/latency.\nTypical: 1536/3072 (OpenAI), 2048 (Doubao).\nUnits: dimensions."
                      />
                      <input
                        className="input"
                        type="number"
                        value={settings.embeddingDim}
                        onChange={(event) =>
                          updateSetting("embeddingDim", Number(event.target.value))
                        }
                      />
                    </label>
                  </div>
                </details>

                <details className="accordion">
                  <summary>Index</summary>
                  <div className="form-grid">
                    <label className="field">
                      <FieldLabel
                        text="Index type"
                        tip="Definition: Milvus index algorithm.\nImpact: FLAT = exact/slow; IVF/HNSW = faster/approx; affects build time/memory.\nTypical: FLAT (Lite), IVF_FLAT for larger data.\nUnits: enum."
                      />
                      <select
                        className="input"
                        value={settings.indexType}
                        onChange={(event) => updateSetting("indexType", event.target.value)}
                      >
                        {INDEX_OPTIONS.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="nlist"
                        tip="Definition: IVF cluster count (nlist).\nImpact: Higher improves recall but increases memory/build time; too high can hurt speed.\nTypical: 64-1024 depending on scale.\nUnits: clusters."
                      />
                      <input
                        className="input"
                        type="number"
                        value={settings.indexNlist}
                        onChange={(event) => updateSetting("indexNlist", Number(event.target.value))}
                      />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="M"
                        tip="Definition: HNSW graph degree (M).\nImpact: Higher improves recall but increases memory/build time.\nTypical: 8-32.\nUnits: connections."
                      />
                      <input
                        className="input"
                        type="number"
                        value={settings.indexM}
                        onChange={(event) => updateSetting("indexM", Number(event.target.value))}
                      />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="efConstruction"
                        tip="Definition: HNSW efConstruction.\nImpact: Higher improves recall but slows indexing and increases memory.\nTypical: 64-200.\nUnits: steps."
                      />
                      <input
                        className="input"
                        type="number"
                        value={settings.indexEfConstruction}
                        onChange={(event) =>
                          updateSetting("indexEfConstruction", Number(event.target.value))
                        }
                      />
                    </label>
                  </div>
                </details>

                <details className="accordion">
                  <summary>Retrieval</summary>
                  <div className="form-grid">
                    <label className="field">
                      <FieldLabel
                        text="Search k"
                        tip="Definition: Candidate pool size before fusion/rerank.\nImpact: Higher recall but more latency and cost.\nTypical: 20-100.\nUnits: chunks."
                      />
                      <input
                        className="input"
                        type="number"
                        value={settings.searchK}
                        onChange={(event) => updateSetting("searchK", Number(event.target.value))}
                      />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="Top k"
                        tip="Definition: Final context size sent to the LLM.\nImpact: More context increases tokens/cost; too low risks missing evidence.\nTypical: 3-8.\nUnits: chunks."
                      />
                      <input
                        className="input"
                        type="number"
                        value={settings.topK}
                        onChange={(event) => updateSetting("topK", Number(event.target.value))}
                      />
                    </label>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={settings.enableBm25}
                        onChange={(event) => updateSetting("enableBm25", event.target.checked)}
                      />
                      <span className="toggle-text">Enable BM25</span>
                      <InfoTip text="Definition: Enable BM25 lexical retrieval.\nImpact: Improves keyword recall; adds BM25 index build/storage.\nTypical: on for mixed queries.\nUnits: boolean." />
                    </label>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={settings.enableSparse}
                        onChange={(event) => updateSetting("enableSparse", event.target.checked)}
                      />
                      <span className="toggle-text">Enable sparse embeddings</span>
                      <InfoTip text="Definition: Enable sparse vector retrieval.\nImpact: Good for keyword-like queries; may add API cost; ignored if BM25 on.\nTypical: off unless sparse API is configured.\nUnits: boolean." />
                    </label>
                    {settings.enableBm25 && (
                      <label className="field">
                        <FieldLabel
                          text="Fusion"
                          tip="Definition: Hybrid fusion strategy.\nImpact: weighted blends scores, rrf blends ranks, dense ignores BM25.\nTypical: weighted or rrf.\nUnits: enum."
                        />
                        <select
                          className="input"
                          value={settings.fusion}
                          onChange={(event) => updateSetting("fusion", event.target.value)}
                        >
                          {FUSION_OPTIONS.map((option) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    {settings.enableBm25 && settings.fusion === "weighted" && (
                      <label className="field">
                        <FieldLabel
                          text={`Hybrid alpha (${settings.hybridAlpha.toFixed(2)})`}
                          tip="Definition: Dense weight in weighted fusion.\nImpact: 0=BM25 only, 1=dense only; balances lexical vs semantic.\nTypical: 0.3-0.7.\nUnits: ratio (0-1)."
                        />
                        <input
                          className="range"
                          type="range"
                          min={0}
                          max={1}
                          step={0.01}
                          value={settings.hybridAlpha}
                          onChange={(event) =>
                            updateSetting("hybridAlpha", Number(event.target.value))
                          }
                        />
                      </label>
                    )}
                    {settings.enableBm25 && settings.fusion === "rrf" && (
                      <label className="field">
                        <FieldLabel
                          text="RRF k"
                          tip="Definition: Reciprocal Rank Fusion constant (k).\nImpact: Larger values smooth rank gaps; smaller favors top ranks.\nTypical: 60.\nUnits: rank constant."
                        />
                        <input
                          className="input"
                          type="number"
                          value={settings.rrfK}
                          onChange={(event) => updateSetting("rrfK", Number(event.target.value))}
                        />
                      </label>
                    )}
                    {settings.enableBm25 && settings.enableSparse && (
                      <p className="hint">BM25 takes precedence over sparse embeddings.</p>
                    )}
                  </div>
                </details>

                <details className="accordion">
                  <summary>Rerank</summary>
                  <div className="form-grid">
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={settings.rerank}
                        onChange={(event) => updateSetting("rerank", event.target.checked)}
                      />
                      <span className="toggle-text">Enable rerank</span>
                      <InfoTip text="Definition: Enable reranking.\nImpact: Higher precision but extra latency/cost per query.\nTypical: on for QA.\nUnits: boolean." />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="Rerank model"
                        tip="Definition: Rerank cross-encoder model.\nImpact: Improves precision but increases latency/cost.\nTypical: cross-encoder/ms-marco-MiniLM-L-6-v2.\nUnits: string."
                      />
                      <input
                        className="input"
                        value={settings.rerankModel}
                        onChange={(event) => updateSetting("rerankModel", event.target.value)}
                      />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="Rerank top k"
                        tip="Definition: Post-rerank keep count.\nImpact: Smaller = shorter prompt; too small risks missing evidence.\nTypical: 3-8.\nUnits: chunks."
                      />
                      <input
                        className="input"
                        type="number"
                        value={settings.rerankTopK}
                        onChange={(event) => updateSetting("rerankTopK", Number(event.target.value))}
                      />
                    </label>
                  </div>
                </details>

                <details className="accordion">
                  <summary>Generation</summary>
                  <div className="form-grid">
                    <label className="field">
                      <FieldLabel
                        text="OpenAI model"
                        tip="Definition: Chat model used for answer generation.\nImpact: Quality/latency/cost depend on model.\nTypical: gpt-5.1, gpt-4o.\nUnits: string."
                      />
                      <input
                        className="input"
                        value={settings.openaiModel}
                        onChange={(event) => updateSetting("openaiModel", event.target.value)}
                      />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="History turns"
                        tip="Definition: Conversation history turns included.\nImpact: More context but higher token cost.\nTypical: 0-5.\nUnits: turns."
                      />
                      <input
                        className="input"
                        type="number"
                        value={settings.historyTurns}
                        onChange={(event) => updateSetting("historyTurns", Number(event.target.value))}
                      />
                    </label>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={settings.stream}
                        onChange={(event) => updateSetting("stream", event.target.checked)}
                      />
                      <span className="toggle-text">Stream tokens (API)</span>
                      <InfoTip text="Definition: Request token streaming from API.\nImpact: Faster perceived response; UI still renders final output.\nTypical: off in UI.\nUnits: boolean." />
                    </label>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={settings.interactive}
                        onChange={(event) => updateSetting("interactive", event.target.checked)}
                      />
                      <span className="toggle-text">Interactive mode</span>
                      <InfoTip text="Definition: Interactive mode flag.\nImpact: No effect in API UI flow.\nTypical: off.\nUnits: boolean." />
                    </label>
                    <p className="hint">
                      Streaming is enabled server-side; UI still renders the final answer.
                    </p>
                  </div>
                </details>

                <details className="accordion">
                  <summary>Ingest</summary>
                  <div className="form-grid">
                    <label className="field">
                      <FieldLabel
                        text="Chunk size"
                        tip="Definition: Target chunk length for splitting.\nImpact: Larger chunks reduce count but increase context size.\nTypical: 500-1000.\nUnits: characters."
                      />
                      <input
                        className="input"
                        type="number"
                        value={settings.chunkSize}
                        onChange={(event) => updateSetting("chunkSize", Number(event.target.value))}
                      />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="Overlap"
                        tip="Definition: Overlap between chunks.\nImpact: Improves continuity but increases storage.\nTypical: 50-200.\nUnits: characters."
                      />
                      <input
                        className="input"
                        type="number"
                        value={settings.overlap}
                        onChange={(event) => updateSetting("overlap", Number(event.target.value))}
                      />
                    </label>
                    <label className="field">
                      <FieldLabel
                        text="Batch size"
                        tip="Definition: Embedding batch size during ingest.\nImpact: Larger batches improve throughput but use more memory.\nTypical: 16-128.\nUnits: texts per batch."
                      />
                      <input
                        className="input"
                        type="number"
                        value={settings.batchSize}
                        onChange={(event) => updateSetting("batchSize", Number(event.target.value))}
                      />
                    </label>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={settings.reset}
                        onChange={(event) => updateSetting("reset", event.target.checked)}
                      />
                      <span className="toggle-text">Reset before ingest</span>
                      <InfoTip text="Definition: Reset collection before ingest.\nImpact: Destructive; clears existing vectors and BM25 index.\nTypical: on for rebuilds only.\nUnits: boolean." />
                    </label>
                  </div>
                </details>
              </div>
            </section>
          </aside>
        </section>
      </main>
    </div>
  );
}

async function fetchAnswer(query: string, settings: Settings, apiBaseUrl: string) {
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
    history_turns: settings.historyTurns
  };

  if (settings.embeddingBaseUrl.trim()) {
    payload.embedding_base_url = settings.embeddingBaseUrl.trim();
  }
  if (settings.embeddingEndpoint.trim()) {
    payload.embedding_endpoint = settings.embeddingEndpoint.trim();
  }

  const response = await fetch(`${apiBaseUrl}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error("Query failed");
  }

  const data = (await response.json()) as {
    answer?: string;
    citations?: string[];
  };

  return {
    answer: data.answer || "No answer returned.",
    citations: data.citations || []
  };
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

async function mockAnswer(query: string) {
  await new Promise((resolve) => setTimeout(resolve, 800));
  return {
    answer: `Mock answer: I searched the index for "${query}" and found 5 relevant chunks. Connect the API to get real citations.`,
    citations: ["chunk-01", "chunk-04", "chunk-07"]
  };
}
