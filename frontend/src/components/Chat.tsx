"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

interface ChatProps {
  onStateUpdate: (state: any) => void;
  sessionId: string | null;
  setSessionId: (id: string) => void;
}

export default function Chat({ onStateUpdate, sessionId, setSessionId }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Welcome to **AutoML Studio** 🚀\n\nUpload a CSV dataset below to start the automated analytics pipeline. I'll guide you through every step."
    }
  ]);
  const [inputVal, setInputVal] = useState("");
  const [loading, setLoading] = useState(false);
  const [awaitingApproval, setAwaitingApproval] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];

    setLoading(true);
    setMessages(prev => [...prev, { role: "user", content: `📄 Uploaded **${file.name}**` }]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setSessionId(data.session_id);
      onStateUpdate(data);

      setMessages(prev => [...prev, { role: "system", content: "Dataset uploaded successfully. Running ingestion agent..." }]);

      await sendMessage("Start analysis", "approve", data.session_id);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: "system", content: "⚠️ Could not connect to backend. Is it running on port 8000?" }]);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async (message: string, action: string | null = null, explicitSessionId: string | null = null) => {
    const sId = explicitSessionId || sessionId;
    if (!sId) return;

    if (!action) {
      setMessages(prev => [...prev, { role: "user", content: message }]);
    }

    setLoading(true);
    setAwaitingApproval(false);

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sId, message, action })
      });

      const data = await res.json();
      setMessages(prev => [...prev, { role: "assistant", content: data.reply }]);
      onStateUpdate(data);

      if (data.status === "awaiting_human") {
        setAwaitingApproval(true);
      }
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: "system", content: "⚠️ Error communicating with backend." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputVal.trim()) return;
    sendMessage(inputVal);
    setInputVal("");
  };

  return (
    <>
      {/* Header */}
      <div className="chat-header">
        <div className="chat-header-icon">🤖</div>
        <div>
          <h2>AutoML Studio</h2>
          <span className="chat-header-sub">Multi-Agent Analytics Pipeline</span>
        </div>
      </div>

      {/* Messages */}
      <div className="messages-area">
        {messages.map((msg, i) => (
          <div key={i} className={`msg msg-${msg.role} animate-fade-in-up`}>
            <div className="msg-bubble">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}

        {loading && (
          <div className="msg msg-assistant">
            <div className="typing-indicator">
              <span className="typing-dot"></span>
              <span className="typing-dot"></span>
              <span className="typing-dot"></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Controls */}
      <div className="chat-controls">
        {!sessionId ? (
          <label className="upload-zone">
            <input type="file" accept=".csv" style={{ display: "none" }} onChange={handleFileUpload} />
            <div className="upload-zone-icon">📊</div>
            <div className="upload-zone-label">Upload CSV Dataset</div>
            <div className="upload-zone-sub">Drag & drop or click to browse</div>
          </label>
        ) : (
          <>
            {awaitingApproval && (
              <div className="approval-row">
                <button className="btn btn-approve" onClick={() => sendMessage("Approved", "approve")}>
                  ✓ Approve & Continue
                </button>
              </div>
            )}
            <form onSubmit={handleSend} className="chat-input-row">
              <input
                className="chat-input"
                type="text"
                value={inputVal}
                onChange={(e) => setInputVal(e.target.value)}
                placeholder="Ask a question or give instructions..."
                disabled={loading}
              />
              <button type="submit" className="btn btn-send" disabled={loading || !inputVal.trim()}>
                Send
              </button>
            </form>
          </>
        )}
      </div>
    </>
  );
}
