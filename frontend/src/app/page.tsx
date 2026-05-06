"use client";

import { useCallback, useState } from "react";
import Chat from "../components/Chat";
import Visualizer from "../components/Visualizer";
import {
  PIPELINE_STAGES,
  getProducerStage,
  getStageStatusLabel,
  getStageVisualStatus,
  isStageId,
  type StageId,
  type StepResult,
  type StepResults,
} from "../lib/studio";

function buildStepResult(data: Record<string, unknown>): StepResult {
  return {
    analysis: typeof data.reply === "string" ? data.reply : "No analysis available.",
    data_preview: typeof data.data_preview === "string" ? data.data_preview : null,
    metrics:
      typeof data.metrics === "object" && data.metrics !== null
        ? (data.metrics as Record<string, string | number>)
        : {},
    charts: Array.isArray(data.charts) ? (data.charts as StepResult["charts"]) : [],
    updatedAt: new Date().toISOString(),
  };
}

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<StageId>("ingestion");
  const [activeTab, setActiveTab] = useState<StageId>("ingestion");
  const [isNotebookCollapsed, setIsNotebookCollapsed] = useState(false);
  const [stepResults, setStepResults] = useState<StepResults>({});
  const [status, setStatus] = useState("idle");

  const handleStateUpdate = useCallback(
    (data: Record<string, unknown>) => {
      if (typeof data.session_id === "string") {
        const nextSessionId = data.session_id;
        setSessionId((existingId) => existingId ?? nextSessionId);
      }

      if (typeof data.status === "string") {
        setStatus(data.status);
      }

      const nextStageValue = String(data.state ?? "");

      if (!isStageId(nextStageValue)) {
        return;
      }

      const nextStage = nextStageValue;
      const stageStayedOpen = nextStage === currentStage;

      setCurrentStage(nextStage);

      if (!data.reply) {
        setActiveTab(nextStage);
        return;
      }

      const result = buildStepResult(data);

      if (nextStage === "evaluation" && data.status === "completed") {
        setStepResults((previous) => ({
          ...previous,
          modeling: previous.modeling ?? result,
          evaluation: result,
        }));
        setActiveTab("evaluation");
        return;
      }

      const resultStage = stageStayedOpen ? nextStage : getProducerStage(nextStage);

      setStepResults((previous) => ({
        ...previous,
        [resultStage]: result,
      }));
      setActiveTab(resultStage);
    },
    [currentStage],
  );

  const completedStages = PIPELINE_STAGES.filter(
    (stage) => getStageVisualStatus(stage.id, currentStage, status, stepResults) === "completed",
  ).length;

  return (
    <div className="studio-shell">
      <header className="studio-navbar">
        <div className="studio-navbar-top">
          <div className="studio-brand">
            <div className="studio-brand-mark">A</div>
            <div>
              <p className="studio-eyebrow">AutoML Workspace</p>
              <h1>AutoML Studio</h1>
            </div>
          </div>

          <div className="studio-navbar-meta">
            <div className="meta-copy">
              <span>{sessionId ? "Live session" : "Preview mode"}</span>
              <span>
                {completedStages}/{PIPELINE_STAGES.length} complete
              </span>
            </div>
            <button
              type="button"
              className="chrome-button"
              onClick={() => setIsNotebookCollapsed((currentValue) => !currentValue)}
            >
              {isNotebookCollapsed ? "Show Notebook" : "Focus Mode"}
            </button>
          </div>
        </div>

        <nav className="studio-tabs" aria-label="AutoML workflow stages" role="tablist">
          {PIPELINE_STAGES.map((stage) => {
            const stageStatus = getStageVisualStatus(stage.id, currentStage, status, stepResults);
            const isActive = activeTab === stage.id;

            return (
              <button
                key={stage.id}
                type="button"
                className={`studio-tab${isActive ? " active" : ""}`}
                role="tab"
                aria-selected={isActive}
                onClick={() => setActiveTab(stage.id)}
                title={`${stage.label} · ${getStageStatusLabel(stageStatus)}`}
              >
                <span className={`stage-status-dot is-${stageStatus}`} aria-hidden="true" />
                <span className="studio-tab-title">{stage.label}</span>
              </button>
            );
          })}
        </nav>
      </header>

      <main className={`studio-main${isNotebookCollapsed ? " is-focus-mode" : ""}`}>
        <section className="chat-panel">
          <Chat
            currentStage={currentStage}
            onStateUpdate={handleStateUpdate}
            pipelineStatus={status}
            sessionId={sessionId}
            setSessionId={setSessionId}
          />
        </section>

        <section className={`workspace${isNotebookCollapsed ? " is-collapsed" : ""}`}>
          <Visualizer
            activeTab={activeTab}
            currentStage={currentStage}
            hasSession={Boolean(sessionId)}
            isCollapsed={isNotebookCollapsed}
            onToggleNotebook={() => setIsNotebookCollapsed((currentValue) => !currentValue)}
            status={status}
            stepResults={stepResults}
          />
        </section>

        {isNotebookCollapsed && (
          <button
            type="button"
            className="restore-panel-button"
            onClick={() => setIsNotebookCollapsed(false)}
          >
            Show Notebook
          </button>
        )}
      </main>
    </div>
  );
}
