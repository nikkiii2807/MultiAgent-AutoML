"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  PIPELINE_STAGES,
  type ChatMessage,
  type DatasetMeta,
  type StageId,
} from "../lib/studio";

const API_BASE_URL = "http://localhost:8000";

interface ChatProps {
  currentStage: StageId;
  datasetMeta: DatasetMeta | null;
  messages: ChatMessage[];
  onStateUpdate: (state: Record<string, unknown>) => void;
  pipelineStatus: string;
  sessionId: string | null;
}

function getStageLabel(stageId: StageId): string {
  return PIPELINE_STAGES.find((stage) => stage.id === stageId)?.label ?? "Notebook";
}

const PREVIEW_MESSAGES: ChatMessage[] = [
  {
    role: "assistant",
    content:
      "Upload a CSV and I’ll orchestrate the full AutoML pipeline while the notebook on the right updates stage by stage.",
  },
  {
    role: "system",
    content:
      "Open History from the left edge anytime to reopen a saved session and continue where you left off.",
  },
];

export default function Chat({
  currentStage,
  datasetMeta,
  messages,
  onStateUpdate,
  pipelineStatus,
  sessionId,
}: ChatProps) {
  const [inputVal, setInputVal] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const stageLabel = getStageLabel(currentStage);
  const awaitingApproval = Boolean(sessionId) && pipelineStatus === "awaiting_human";
  const displayedMessages = useMemo(
    () => (sessionId || messages.length > 0 ? messages : PREVIEW_MESSAGES),
    [messages, sessionId],
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [displayedMessages, loading]);

  async function sendMessage(
    message: string,
    action: string | null = null,
    explicitSessionId: string | null = null,
  ) {
    const resolvedSessionId = explicitSessionId ?? sessionId;
    if (!resolvedSessionId) {
      return;
    }

    setLoading(true);
    setErrorMessage(null);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
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

      onStateUpdate(data);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Could not communicate with the backend service.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleFileUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    const allowedExtensions = [".csv", ".xlsx"];
    const fileExtension = "." + file.name.split(".").pop()?.toLowerCase();
    if (!allowedExtensions.includes(fileExtension)) {
      setErrorMessage("Unsupported file type. Please upload a .csv or .xlsx file.");
      event.target.value = "";
      return;
    }

    setLoading(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = (await response.json()) as Record<string, unknown>;

      if (!response.ok) {
        throw new Error(String(data.detail ?? "Dataset upload failed."));
      }

      onStateUpdate(data);
      const nextSessionId = typeof data.session_id === "string" ? data.session_id : null;
      if (nextSessionId) {
        await sendMessage("Start analysis", "approve", nextSessionId);
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Could not connect to the backend service.");
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

    void sendMessage(inputVal);
    setInputVal("");
  }

  return (
    <div className="chat-workspace">
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
        {displayedMessages.map((message, index) => (
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
              <div className="message-bubble readable-content">
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
        {errorMessage && <p className="chat-error-banner">⚠ {errorMessage}</p>}

        {awaitingApproval && (
          <button
            type="button"
            className="action-button action-button-primary"
            onClick={() => void sendMessage("Approved", "approve")}
          >
            Approve and continue
          </button>
        )}

        <form className="composer" onSubmit={handleSend}>
          <div className="composer-attach-row">
            <button
              type="button"
              className="action-button action-button-secondary composer-attach-button"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
            >
              Attach CSV / .xlsx
            </button>
          </div>

          <div className="composer-row">
            <input
              className="composer-input"
              type="text"
              value={inputVal}
              onChange={(event) => setInputVal(event.target.value)}
              placeholder={
                sessionId
                  ? "Ask for changes, approvals, or follow-up analysis..."
                  : "Attach a CSV to start or reopen a live session"
              }
              disabled={!sessionId || loading}
            />

            <button
              type="submit"
              className="action-button action-button-primary composer-send-button"
              disabled={loading || !sessionId || !inputVal.trim()}
            >
              Send
            </button>
          </div>
        </form>

        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx"
          className="composer-file-input"
          onChange={handleFileUpload}
        />
      </div>
    </div>
  );
}
