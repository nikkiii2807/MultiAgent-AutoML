import type { PreviewStageDefinition } from "../types";
import { SAMPLE_SOURCE_ROWS } from "../shared";

export const ingestionNotebook: PreviewStageDefinition = {
  stageSummary:
    "The ingestion notebook validates the CSV, inspects the schema, and establishes the baseline dataset fingerprint.",
  insight:
    "The studio surfaces schema issues before any model work begins, so the workflow feels deliberate instead of opaque.",
  analysis: `## Dataset intake summary

The uploaded dataset is profiled immediately after connection so the rest of the pipeline inherits a stable schema contract.

- **12,842 rows** and **26 columns** detected
- **Target candidate:** \`churn\`
- **Quality flags:** sparse null pockets in \`monthly_charge\` and \`support_calls\`
- **Notebook behavior:** ingestion writes a structured artifact that downstream cells can reference

This stage behaves like the first notebook section in VS Code: code up top, rich outputs underneath, and human sign-off before automation moves forward.`,
  metrics: {
    dataset_rows: 12842,
    target_signal: "Churn",
    columns_detected: 26,
    missing_fields: 217,
  },
  charts: [
    {
      id: "ingestion-missing",
      title: "Missing Values by Column",
      type: "bar",
      data: [
        { name: "monthly_charge", value: 91 },
        { name: "support_calls", value: 64 },
        { name: "payment_type", value: 38 },
        { name: "zip_region", value: 24 },
      ],
      xKey: "name",
      bars: [{ key: "value", color: null, label: "Missing" }],
    },
    {
      id: "ingestion-dtypes",
      title: "Column Data Types",
      type: "pie",
      data: [
        { name: "numeric", value: 14 },
        { name: "categorical", value: 9 },
        { name: "boolean", value: 3 },
      ],
    },
  ],
  dataPreview: SAMPLE_SOURCE_ROWS,
  cells: [
    {
      id: "connect",
      label: "In [1]",
      language: "python",
      note: "Dataset connection and schema registration.",
      output: "analysis",
      title: "Load source data",
      code: `from automl_studio import Studio\n\nstudio = Studio(project="customer-retention")\ndataset = studio.connect_csv("customer_churn.csv")\nprofile = dataset.profile(target="churn")`,
    },
    {
      id: "inspect",
      label: "In [2]",
      language: "python",
      note: "Notebook diagnostics and structured profiling outputs.",
      output: "analytics",
      title: "Inspect schema health",
      code: `profile.summary()\nprofile.missing_values(top_k=4)\nprofile.type_map()`,
    },
    {
      id: "preview",
      label: "In [3]",
      language: "python",
      note: "Preview the sampled records that anchor the notebook.",
      output: "table",
      title: "Preview source rows",
      code: `dataset.head(4)\ndataset.sample(seed=42)`,
    },
  ],
};
