import type { ChartConfig, StageId } from "../studio";

export interface NotebookCellDefinition {
  id: string;
  code: string;
  label: string;
  language: string;
  note: string;
  output: "analysis" | "analytics" | "table";
  title: string;
}

export interface PreviewStageDefinition {
  analysis: string;
  cells: NotebookCellDefinition[];
  dataPreview: string;
  insight: string;
  metrics: Record<string, string | number>;
  stageSummary: string;
  charts: ChartConfig[];
}

export type PreviewNotebookRegistry = Record<StageId, PreviewStageDefinition>;
