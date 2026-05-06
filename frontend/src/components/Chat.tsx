"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { PIPELINE_STAGES, type StageId } from "../lib/studio";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

interface DatasetMeta {
  name: string;
  rows: number;
  columns: number;
}

interface ChatProps {
  currentStage: StageId;
  onStateUpdate: (state: Record<string, unknown>) => void;
  pipelineStatus: string;
  sessionId: string | null;
  setSessionId: (id: string | null) => void;
}

function getStageLabel(stageId: StageId): string {
  return PIPELINE_STAGES.find((stage) => stage.id === stageId)?.label ?? "Notebook";
}

export default function Chat({
  currentStage,
  onStateUpdate,
  pipelineStatus,
  sessionId,
  setSessionId,
}: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Upload a CSV and I’ll orchestrate the full AutoML pipeline while the notebook on the right updates stage by stage.",
    },
    {
      role: "system",
      content:
        "Notebook preview is interactive even before a dataset is attached, so you can inspect every stage layout right away.",
    },
  ]);
  const [inputVal, setInputVal] = useState("");
  const [loading, setLoading] = useState(false);
  const [awaitingApproval, setAwaitingApproval] = useState(false);
  const [datasetMeta, setDatasetMeta] = useState<DatasetMeta | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const stageLabel = getStageLabel(currentStage);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [loading, messages]);

  async function sendMessage(
    message: string,
    action: string | null = null,
    explicitSessionId: string | null = null,
  ) {
    const resolvedSessionId = explicitSessionId ?? sessionId;
    if (!resolvedSessionId) {
      return;
    }

    if (!action) {
      setMessages((previous) => [...previous, { role: "user", content: message }]);
    }

    setLoading(true);
    setAwaitingApproval(false);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          message,
          session_id: resolvedSessionId,
        }),
      });

      const data = (await response.json()) as Record<string, unknown>;

      if (!response.ok) {
        throw new Error(String(data.detail ?? "Unable to reach the AutoML backend."));
      }

      if (typeof data.reply === "string") {
        setMessages((previous) => [
          ...previous,
          { role: "assistant", content: data.reply as string },
        ]);
      }

      onStateUpdate(data);

      if (data.status === "awaiting_human") {
        setAwaitingApproval(true);
      }
    } catch (error) {
      const messageText =
        error instanceof Error
          ? error.message
          : "Could not communicate with the backend service.";

      setMessages((previous) => [
        ...previous,
        { role: "system", content: `⚠️ ${messageText}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function handleFileUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setLoading(true);
    setMessages((previous) => [
      ...previous,
      { role: "user", content: `Attached dataset **${file.name}**` },
    ]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      const data = (await response.json()) as Record<string, unknown>;

      if (!response.ok) {
        throw new Error(String(data.detail ?? "Dataset upload failed."));
      }

      const nextSessionId = typeof data.session_id === "string" ? data.session_id : null;
      setSessionId(nextSessionId);
      onStateUpdate(data);
      setDatasetMeta({
        name: file.name,
        rows: Number(data.rows ?? 0),
        columns: Array.isArray(data.columns) ? data.columns.length : 0,
      });
      setMessages((previous) => [
        ...previous,
        {
          role: "system",
          content:
            "Dataset connected. I’m running the ingestion notebook now and will pause wherever human approval is needed.",
        },
      ]);

      await sendMessage("Start analysis", "approve", nextSessionId);
    } catch (error) {
      const messageText =
        error instanceof Error
          ? error.message
          : "Could not connect to the backend service.";

      setMessages((previous) => [
        ...previous,
        { role: "system", content: `⚠️ ${messageText}` },
      ]);
    } finally {
      event.target.value = "";
      setLoading(false);
    }
  }

  function handleSend(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!inputVal.trim() || !sessionId) {
      return;
    }

    sendMessage(inputVal);
    setInputVal("");
  }

  return (
    <>
      <div className="chat-panel-header">
        <div className="assistant-summary">
          <span className="assistant-dot" />
          <div>
            <p className="assistant-chip-label">AutoML Copilot</p>
            <h2>Focused chat workspace</h2>
          </div>
        </div>
      </div>

      <div className="chat-context-bar">
        <span className="context-chip">
          {datasetMeta
            ? `${datasetMeta.name} · ${datasetMeta.rows.toLocaleString()} rows · ${datasetMeta.columns} cols`
            : "No dataset attached"}
        </span>
        <span className="context-chip">Stage: {stageLabel}</span>
        <span className="context-chip">
          Status: {pipelineStatus === "awaiting_human" ? "Awaiting approval" : pipelineStatus}
        </span>
      </div>

      <div className="messages-area">
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className={`message-row message-${message.role}`}>
            <div className="message-avatar">
              {message.role === "assistant" ? "AI" : message.role === "user" ? "You" : "Sys"}
            </div>
            <div className="message-stack">
              <span className="message-role">
                {message.role === "assistant"
                  ? "AutoML Copilot"
                  : message.role === "user"
                    ? "You"
                    : "System"}
              </span>
              <div className="message-bubble">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="message-row message-assistant">
            <div className="message-avatar">AI</div>
            <div className="typing-indicator" aria-label="Assistant is typing">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="chat-controls">
        {awaitingApproval && (
          <button
            type="button"
            className="action-button action-button-primary"
            onClick={() => sendMessage("Approved", "approve")}
          >
            Approve and continue
          </button>
        )}

        <form className="composer" onSubmit={handleSend}>
          <button
            type="button"
            className="action-button action-button-secondary"
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
          >
            Attach CSV
          </button>

          <input
            className="composer-input"
            type="text"
            value={inputVal}
            onChange={(event) => setInputVal(event.target.value)}
            placeholder={
              sessionId
                ? "Ask for changes, approvals, or follow-up analysis..."
                : "Attach a CSV to start a live conversation with the pipeline"
            }
            disabled={!sessionId || loading}
          />

          <button
            type="submit"
            className="action-button action-button-primary"
            disabled={loading || !sessionId || !inputVal.trim()}
          >
            Send
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            hidden
            onChange={handleFileUpload}
          />
        </form>

        <p className="composer-footnote">
          The copilot stays synced with the notebook stages, so every approval updates the workflow on
          the right.
        </p>
      </div>
    </>
  );
}
