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
      "Hi! Drop a file or ask a question. I will answer with evidence snippets once the backend is wired.",
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
    return selectedKb ? `Chat · ${selectedKb.name}` : "Chat";
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
          "I could not reach the backend. Please check the API server or set VITE_USE_MOCK=false once it is ready.",
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

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <p className="brand">RAG Light Console</p>
          <h1>Knowledge you can trace.</h1>
        </div>
        <div className="status-pill">
          <span className={USE_MOCK ? "dot warning" : "dot"} />
          {USE_MOCK ? "Mock mode" : "Live API"}
        </div>
      </header>

      <aside className="sidebar">
        <section className="card">
          <div className="card-header">
            <h2>Knowledge Bases</h2>
            <button className="secondary">New</button>
          </div>
          <div className="kb-list">
            {knowledgeBases.map((kb) => (
              <button
                key={kb.id}
                className={kb.id === selectedKb.id ? "kb active" : "kb"}
                onClick={() => setSelectedKb(kb)}
              >
                <div>
                  <p className="kb-name">{kb.name}</p>
                  <p className="kb-meta">
                    {kb.documents} docs · {kb.updatedAt}
                  </p>
                </div>
                <span className="pill">{kb.id}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="card upload-card">
          <h2>Upload Documents</h2>
          <p className="muted">
            PDFs, DOCX, MD, TXT, PNG. They will be parsed and indexed.
          </p>
          <label className="upload-zone">
            <input
              type="file"
              multiple
              onChange={(event) => handleUpload(event.target.files)}
            />
            <span>Drag files here or click to upload</span>
          </label>
          {uploadStatus && <p className="status">{uploadStatus}</p>}
        </section>

        <section className="card">
          <h2>Quality Controls</h2>
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

      <main className="chat">
        <div className="chat-header">
          <div>
            <p className="title">{chatTitle}</p>
            <p className="subtitle">Evidence-backed answers from {selectedKb.name}.</p>
          </div>
          <button className="secondary">Clear</button>
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
                <span>Thinking…</span>
              </div>
              <p>Retrieving evidence and drafting the response.</p>
            </article>
          )}
        </div>

        <div className="chat-input">
          <textarea
            placeholder="Ask a question about this knowledge base…"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            rows={3}
          />
          <div className="actions">
            <div className="helper">
              <span>Shift+Enter for newline</span>
              <span>Collection: {selectedKb.id}</span>
            </div>
            <button className="primary" onClick={handleSend} disabled={!canSend}>
              {isSending ? "Sending…" : "Send"}
            </button>
          </div>
        </div>
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
