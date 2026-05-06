export type StageId =
  | "ingestion"
  | "preprocessing"
  | "eda"
  | "standardization"
  | "modeling"
  | "evaluation";

export interface ChartBarConfig {
  key: string;
  color: string | null;
  label: string;
}

export type ChartDatum = Record<string, string | number | null>;

export interface ChartConfig {
  id: string;
  title: string;
  type: "bar" | "pie";
  data: ChartDatum[];
  xKey?: string;
  bars?: ChartBarConfig[];
}

export interface StepResult {
  analysis: string;
  data_preview: string | null;
  metrics: Record<string, string | number>;
  charts: ChartConfig[];
  updatedAt?: string;
}

export type StepResults = Partial<Record<StageId, StepResult>>;

export interface StageDefinition {
  id: StageId;
  label: string;
  icon: string;
  notebookLabel: string;
  prompt: string;
  helper: string;
}

export type StageVisualStatus = "completed" | "running" | "error" | "queued";

export const PIPELINE_STAGES: StageDefinition[] = [
  {
    id: "ingestion",
    label: "Ingestion",
    icon: "01",
    notebookLabel: "01_data_ingestion.ipynb",
    prompt: "Connect the raw dataset, infer schema, and surface data quality flags before automation begins.",
    helper: "Schema checks, row profiling, and upload validation.",
  },
  {
    id: "preprocessing",
    label: "Preprocessing",
    icon: "02",
    notebookLabel: "02_preprocessing.ipynb",
    prompt: "Clean nulls, encode categories, and prepare a modeling-ready dataframe.",
    helper: "Imputation, encoding, and type corrections.",
  },
  {
    id: "eda",
    label: "EDA",
    icon: "03",
    notebookLabel: "03_exploration.ipynb",
    prompt: "Inspect distribution shifts, correlations, and class balance with notebook-style outputs.",
    helper: "Distributions, correlation scan, and early risk detection.",
  },
  {
    id: "standardization",
    label: "Feature Engineering",
    icon: "04",
    notebookLabel: "04_feature_engineering.ipynb",
    prompt: "Scale, transform, and synthesize higher-signal features before model search.",
    helper: "Scaling, interactions, and feature selection.",
  },
  {
    id: "modeling",
    label: "Modeling",
    icon: "05",
    notebookLabel: "05_modeling.ipynb",
    prompt: "Benchmark candidate models and surface the strongest leaderboard entries.",
    helper: "AutoML search, leaderboard ranking, and best-model selection.",
  },
  {
    id: "evaluation",
    label: "Evaluation",
    icon: "06",
    notebookLabel: "06_evaluation.ipynb",
    prompt: "Explain performance, compare tradeoffs, and make the final deployment recommendation.",
    helper: "Metrics review, prediction breakdown, and launch readiness.",
  },
];

export const PIPELINE_ORDER = PIPELINE_STAGES.map((stage) => stage.id);

export function isStageId(value: string): value is StageId {
  return PIPELINE_ORDER.includes(value as StageId);
}

export function getProducerStage(stage: StageId): StageId {
  const index = PIPELINE_ORDER.indexOf(stage);
  return index <= 0 ? stage : PIPELINE_ORDER[index - 1];
}

export function getStageVisualStatus(
  stageId: StageId,
  currentStage: StageId,
  pipelineStatus: string,
  stepResults: StepResults,
): StageVisualStatus {
  const stageIndex = PIPELINE_ORDER.indexOf(stageId);
  const currentIndex = PIPELINE_ORDER.indexOf(currentStage);

  if (pipelineStatus === "error" && stageId === currentStage) {
    return "error";
  }

  if (pipelineStatus === "completed" && stageId === "evaluation") {
    return "completed";
  }

  if (stageIndex < currentIndex) {
    return "completed";
  }

  if (stageIndex === currentIndex) {
    return pipelineStatus === "completed" ? "completed" : "running";
  }

  if (stepResults[stageId]) {
    return "completed";
  }

  return "queued";
}

export function getStageStatusLabel(status: StageVisualStatus): string {
  switch (status) {
    case "completed":
      return "✔ Completed";
    case "running":
      return "⏳ Running";
    case "error":
      return "⚠ Error";
    default:
      return "○ Queued";
  }
}
