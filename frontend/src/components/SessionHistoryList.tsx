"use client";

import { PIPELINE_STAGES, type SessionSummary } from "../lib/studio";

interface SessionHistoryListProps {
  activeSessionId: string | null;
  isLoading?: boolean;
  onNewSession: () => void;
  onSelectSession: (sessionId: string) => void;
  sessions: SessionSummary[];
}

function formatUpdatedAt(updatedAt: string): string {
  if (!updatedAt) {
    return "Saved";
  }

  const parsed = new Date(updatedAt);
  return Number.isNaN(parsed.getTime()) ? "Saved" : parsed.toLocaleString();
}

export default function SessionHistoryList({
  activeSessionId,
  isLoading = false,
  onNewSession,
  onSelectSession,
  sessions,
}: SessionHistoryListProps) {
  return (
    <>
      <div className="history-drawer-header">
        <div>
          <p className="assistant-chip-label">History</p>
          <h2 id="history-drawer-title">Saved sessions</h2>
        </div>
        <button type="button" className="chrome-button" onClick={onNewSession} disabled={isLoading}>
          New session
        </button>
      </div>

      <div className="history-drawer-list" role="list">
        {sessions.length === 0 ? (
          <p className="history-empty-state">
            Upload a dataset to create your first persistent session.
          </p>
        ) : (
          sessions.map((session) => {
            const isActive = session.id === activeSessionId;
            return (
              <button
                key={session.id}
                type="button"
                role="listitem"
                className={`history-card${isActive ? " is-active" : ""}`}
                disabled={isLoading}
                onClick={() => onSelectSession(session.id)}
              >
                <div className="history-card-top">
                  <strong>{session.title}</strong>
                  <span>
                    {PIPELINE_STAGES.find((stage) => stage.id === session.current_step)?.label ??
                      session.current_step}
                  </span>
                </div>
                <p className="history-card-file">{session.filename}</p>
                <p className="history-card-preview">
                  {session.preview || "Saved workflow with persisted artifacts."}
                </p>
                <div className="history-card-meta">
                  <span>{session.status}</span>
                  <span>{formatUpdatedAt(session.updated_at)}</span>
                </div>
              </button>
            );
          })
        )}
      </div>
    </>
  );
}
