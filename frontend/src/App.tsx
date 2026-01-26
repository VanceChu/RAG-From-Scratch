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

export default function App() {
  const [knowledgeBases] = useState<KnowledgeBase[]>(INITIAL_KBS);
  const [selectedKb, setSelectedKb] = useState<KnowledgeBase>(INITIAL_KBS[0]);
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const chatTitle = useMemo(() => {
    return selectedKb ? `Conversation · ${selectedKb.name}` : "Conversation";
  }, [selectedKb]);

  const canSend = input.trim().length > 0 && !isSending;

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
      const response = USE_MOCK
        ? await mockAnswer(trimmed)
        : await fetchAnswer(trimmed, selectedKb.id);

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
      if (USE_MOCK) {
        await new Promise((resolve) => setTimeout(resolve, 1200));
      } else {
        const form = new FormData();
        Array.from(files).forEach((file) => form.append("files", file));
        form.append("collection", selectedKb.id);

        const res = await fetch(`${API_BASE}/ingest`, {
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
          {USE_MOCK ? "Mock mode" : "Live API"}
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
              <strong>{USE_MOCK ? "Mock" : "Live API"}</strong>
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
              <h2>Retrieval</h2>
              <div className="stack">
                <div>
                  <p className="label">Fusion</p>
                  <p className="value">Weighted (alpha 0.5)</p>
                </div>
                <div>
                  <p className="label">Rerank</p>
                  <p className="value">Cross-encoder enabled</p>
                </div>
                <div>
                  <p className="label">Index</p>
                  <p className="value">Milvus Lite · FLAT</p>
                </div>
              </div>
            </section>
          </aside>
        </section>
      </main>
    </div>
  );
}

async function fetchAnswer(query: string, collection: string) {
  const response = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query, collection })
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

async function mockAnswer(query: string) {
  await new Promise((resolve) => setTimeout(resolve, 800));
  return {
    answer: `Mock answer: I searched the index for "${query}" and found 5 relevant chunks. Connect the API to get real citations.`,
    citations: ["chunk-01", "chunk-04", "chunk-07"]
  };
}
