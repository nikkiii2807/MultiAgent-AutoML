"use client";

import { useCallback, useEffect, useState } from "react";
import Chat from "../components/Chat";
import Visualizer from "../components/Visualizer";
import {
  PIPELINE_ORDER,
  PIPELINE_STAGES,
  getProducerStage,
  getStageStatusLabel,
  getStageVisualStatus,
  isStageId,
  type ChatMessage,
  type DatasetMeta,
  type SessionSummary,
  type StageId,
  type StepResult,
  type StepResults,
} from "../lib/studio";

const API_BASE_URL = "http://localhost:8000";

function buildStepResult(data: Record<string, unknown>): StepResult {
  return {
    analysis:
      typeof data.analysis === "string"
        ? data.analysis
        : typeof data.reply === "string"
          ? data.reply
          : "No analysis available.",
    data_preview:
      typeof data.data_preview === "string"
        ? data.data_preview
        : typeof data.dataPreview === "string"
          ? data.dataPreview
          : null,
    metrics:
      typeof data.metrics === "object" && data.metrics !== null
        ? (data.metrics as Record<string, string | number>)
        : {},
    charts: Array.isArray(data.charts) ? (data.charts as StepResult["charts"]) : [],
    updatedAt:
      typeof data.updated_at === "string"
        ? data.updated_at
        : typeof data.updatedAt === "string"
          ? data.updatedAt
          : new Date().toISOString(),
  };
}

function buildStepResultsMap(rawStepResults: unknown): StepResults {
  if (!rawStepResults || typeof rawStepResults !== "object") {
    return {};
  }

  const parsedEntries = Object.entries(rawStepResults).flatMap(([stageId, value]) => {
    if (!isStageId(stageId) || !value || typeof value !== "object") {
      return [];
    }

    return [[stageId, buildStepResult(value as Record<string, unknown>)] as const];
  });

  return parsedEntries.length > 0 ? Object.fromEntries(parsedEntries) : {};
}

function buildMessages(rawMessages: unknown): ChatMessage[] {
  if (!Array.isArray(rawMessages)) {
    return [];
  }

  return rawMessages.flatMap((message) => {
    if (!message || typeof message !== "object") {
      return [];
    }

    const role = String((message as Record<string, unknown>).role ?? "");
    const normalizedRole = role === "assistant" || role === "user" || role === "system" ? role : "system";
    const content = (message as Record<string, unknown>).content;
    if (typeof content !== "string") {
      return [];
    }

    return [
      {
        role: normalizedRole,
        content,
        action:
          typeof (message as Record<string, unknown>).action === "string"
            ? String((message as Record<string, unknown>).action)
            : null,
        created_at:
          typeof (message as Record<string, unknown>).created_at === "string"
            ? String((message as Record<string, unknown>).created_at)
            : undefined,
      },
    ];
  });
}

function buildDatasetMeta(data: Record<string, unknown>): DatasetMeta | null {
  const datasetInfo =
    typeof data.dataset_info === "object" && data.dataset_info !== null
      ? (data.dataset_info as Record<string, unknown>)
      : null;

  const filename =
    typeof data.filename === "string"
      ? data.filename
      : typeof data.name === "string"
        ? data.name
        : null;

  if (!datasetInfo && !filename) {
    return null;
  }

  return {
    name: filename ?? "Saved dataset",
    rows: Number(datasetInfo?.rows ?? data.rows ?? 0),
    columns: Number(datasetInfo?.cols ?? (Array.isArray(data.columns) ? data.columns.length : 0)),
  };
}

function resolveActiveTab(
  currentStage: StageId,
  stepResults: StepResults,
  explicitResultStep?: string,
): StageId {
  if (explicitResultStep && isStageId(explicitResultStep)) {
    return explicitResultStep;
  }

  if (stepResults[currentStage]) {
    return currentStage;
  }

  const producerStage = getProducerStage(currentStage);
  if (stepResults[producerStage]) {
    return producerStage;
  }

  for (let index = PIPELINE_ORDER.length - 1; index >= 0; index -= 1) {
    const stageId = PIPELINE_ORDER[index];
    if (stepResults[stageId]) {
      return stageId;
    }
  }

  return currentStage;
}

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<StageId>("ingestion");
  const [activeTab, setActiveTab] = useState<StageId>("ingestion");
  const [isNotebookCollapsed, setIsNotebookCollapsed] = useState(false);
  const [stepResults, setStepResults] = useState<StepResults>({});
  const [status, setStatus] = useState("idle");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [datasetMeta, setDatasetMeta] = useState<DatasetMeta | null>(null);
  const [sessionHistory, setSessionHistory] = useState<SessionSummary[]>([]);

  const fetchSessionHistory = useCallback(async (): Promise<SessionSummary[]> => {
    const response = await fetch(`${API_BASE_URL}/sessions`);
    const data = (await response.json()) as unknown;
    if (!response.ok || !Array.isArray(data)) {
      throw new Error("Could not load saved sessions.");
    }

    const parsedSessions = data.flatMap((session) => {
      if (!session || typeof session !== "object") {
        return [];
      }

      const currentStep = String((session as Record<string, unknown>).current_step ?? "");
      if (!isStageId(currentStep)) {
        return [];
      }

      return [
        {
          id: String((session as Record<string, unknown>).id ?? ""),
          title: String((session as Record<string, unknown>).title ?? "Saved session"),
          filename: String((session as Record<string, unknown>).filename ?? "dataset.csv"),
          current_step: currentStep,
          status: String((session as Record<string, unknown>).status ?? "idle"),
          updated_at: String((session as Record<string, unknown>).updated_at ?? ""),
          preview:
            typeof (session as Record<string, unknown>).preview === "string"
              ? String((session as Record<string, unknown>).preview)
              : "",
        },
      ];
    });

    return parsedSessions;
  }, []);

  const applySessionPayload = useCallback((data: Record<string, unknown>) => {
    if (typeof data.session_id === "string") {
      setSessionId(data.session_id);
    }

    if (typeof data.status === "string") {
      setStatus(data.status);
    }

    const nextStageValue =
      typeof data.current_step === "string" ? data.current_step : String(data.state ?? "");
    const nextStage = isStageId(nextStageValue) ? nextStageValue : "ingestion";
    setCurrentStage(nextStage);

    const backendStepResults = buildStepResultsMap(data.step_results);
    setStepResults(backendStepResults);
    setMessages(buildMessages(data.messages));
    setDatasetMeta(buildDatasetMeta(data));
    setActiveTab(resolveActiveTab(nextStage, backendStepResults, String(data.result_step ?? "")));
  }, []);

  const handleStateUpdate = useCallback(
    (data: Record<string, unknown>) => {
      applySessionPayload(data);
      void fetchSessionHistory().then(setSessionHistory).catch(() => undefined);
    },
    [applySessionPayload, fetchSessionHistory],
  );

  const handleLoadSession = useCallback(
    async (nextSessionId: string) => {
      const response = await fetch(`${API_BASE_URL}/sessions/${nextSessionId}`);
      const data = (await response.json()) as Record<string, unknown>;

      if (!response.ok) {
        throw new Error(String(data.detail ?? "Unable to load the selected session."));
      }

      applySessionPayload(data);
      void fetchSessionHistory().then(setSessionHistory).catch(() => undefined);
    },
    [applySessionPayload, fetchSessionHistory],
  );

  const handleNewSession = useCallback(() => {
    setSessionId(null);
    setCurrentStage("ingestion");
    setActiveTab("ingestion");
    setStepResults({});
    setStatus("idle");
    setMessages([]);
    setDatasetMeta(null);
  }, []);

  useEffect(() => {
    void fetchSessionHistory().then(setSessionHistory).catch(() => undefined);
  }, [fetchSessionHistory]);

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
            datasetMeta={datasetMeta}
            messages={messages}
            onLoadSession={handleLoadSession}
            onNewSession={handleNewSession}
            onStateUpdate={handleStateUpdate}
            pipelineStatus={status}
            sessionHistory={sessionHistory}
            sessionId={sessionId}
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
