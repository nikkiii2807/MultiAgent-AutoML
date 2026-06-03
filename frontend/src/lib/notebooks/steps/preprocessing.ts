import type { PreviewStageDefinition } from "../types";
import { SAMPLE_SOURCE_ROWS } from "../shared";

export const preprocessingNotebook: PreviewStageDefinition = {
  stageSummary:
    "Preprocessing converts raw inputs into a dependable training frame while keeping the transformations readable in notebook cells.",
  insight:
    "A notebook workflow is useful here because we can show exactly what was encoded, dropped, and imputed instead of hiding it in a black-box pipeline.",
  analysis: `## Preprocessing actions

This stage rewrites the raw frame into a model-friendly table while keeping the transformation recipe visible.

1. Impute sparse numeric gaps with median values.
2. Normalize contract and payment categories.
3. Drop low-signal identifiers after preserving them for traceability.
4. Persist the cleaned dataframe for EDA and model search.

The result is a cleaner handoff into exploration without losing the lineage of each feature operation.`,
  metrics: {
    nulls_resolved: 217,
    encoded_features: 11,
    rows_retained: "99.4%",
    ready_columns: 31,
  },
  charts: [
    {
      id: "preprocessing-actions",
      title: "Transformation Coverage",
      type: "bar",
      data: [
        { name: "Imputed", value: 217 },
        { name: "Encoded", value: 88 },
        { name: "Dropped", value: 3 },
        { name: "Validated", value: 31 },
      ],
      xKey: "name",
      bars: [{ key: "value", color: null, label: "Columns" }],
    },
    {
      id: "preprocessing-balance",
      title: "Class Balance After Cleaning",
      type: "pie",
      data: [
        { name: "Retained", value: 12767 },
        { name: "Filtered", value: 75 },
      ],
    },
  ],
  dataPreview: SAMPLE_SOURCE_ROWS,
  cells: [
    {
      id: "cleaning-plan",
      label: "In [4]",
      language: "python",
      note: "Imputation, encoding, and row-level cleaning strategy.",
      output: "analysis",
      title: "Author preprocessing recipe",
      code: `cleaned = (\n    dataset\n    .impute(strategy="median")\n    .encode(categories="target")\n    .drop(columns=["customer_id"])\n)`,
    },
    {
      id: "quality-checks",
      label: "In [5]",
      language: "python",
      note: "Surface the operations that changed the dataset.",
      output: "analytics",
      title: "Validate cleanup outcomes",
      code: `cleaned.report_changes()\ncleaned.class_balance(target="churn")`,
    },
    {
      id: "clean-preview",
      label: "In [6]",
      language: "python",
      note: "Compare the cleaned dataframe before moving into EDA.",
      output: "table",
      title: "Preview cleaned frame",
      code: `cleaned.head(4)\ncleaned.schema()`,
    },
  ],
};
