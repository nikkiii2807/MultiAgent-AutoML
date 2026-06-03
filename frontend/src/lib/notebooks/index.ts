import type { PreviewNotebookRegistry } from "./types";
import { edaNotebook } from "./steps/eda";
import { evaluationNotebook } from "./steps/evaluation";
import { ingestionNotebook } from "./steps/ingestion";
import { modelingNotebook } from "./steps/modeling";
import { preprocessingNotebook } from "./steps/preprocessing";
import { standardizationNotebook } from "./steps/standardization";

export const PREVIEW_NOTEBOOKS: PreviewNotebookRegistry = {
  ingestion: ingestionNotebook,
  preprocessing: preprocessingNotebook,
  eda: edaNotebook,
  standardization: standardizationNotebook,
  modeling: modelingNotebook,
  evaluation: evaluationNotebook,
};

export type { NotebookCellDefinition, PreviewStageDefinition } from "./types";
