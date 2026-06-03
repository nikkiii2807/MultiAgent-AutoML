import type { PreviewStageDefinition } from "../types";
import { SAMPLE_SOURCE_ROWS } from "../shared";

export const edaNotebook: PreviewStageDefinition = {
  stageSummary:
    "EDA turns the notebook into an interactive findings board with correlations, distributions, and class patterns lined up under each code cell.",
  insight:
    "This is where the split layout shines: the assistant can narrate the findings while the notebook keeps the quantitative evidence pinned beside it.",
  analysis: `## Exploratory insights

The EDA notebook focuses on patterns that will change modeling decisions, not on exhaustive chart spam.

- \`monthly_charge\` and \`tenure_months\` show the strongest predictive separation.
- Short-tenure monthly contracts carry the highest churn concentration.
- Support-call frequency increases sharply in the churn-positive segment.

Each output appears as an expandable notebook section so the interface keeps the rhythm of a real VS Code \`.ipynb\` workflow.`,
  metrics: {
    strongest_correlation: 0.78,
    drift_alerts: 2,
    minority_class: "38.4%",
    top_signal: "Monthly contract",
  },
  charts: [
    {
      id: "eda-correlation",
      title: "Correlation Matrix",
      type: "bar",
      data: [
        { name: "monthly_charge", churn: 0.78, tenure: -0.62, support_calls: 0.51 },
        { name: "tenure_months", churn: -0.69, tenure: 1, support_calls: -0.28 },
        { name: "support_calls", churn: 0.44, tenure: -0.28, support_calls: 1 },
      ],
      xKey: "name",
      bars: [
        { key: "churn", color: null, label: "vs churn" },
        { key: "tenure", color: null, label: "vs tenure" },
        { key: "support_calls", color: null, label: "vs support calls" },
      ],
    },
    {
      id: "eda-distribution",
      title: "Churn Distribution",
      type: "pie",
      data: [
        { name: "Stayed", value: 7912 },
        { name: "Churned", value: 4930 },
      ],
    },
  ],
  dataPreview: SAMPLE_SOURCE_ROWS,
  cells: [
    {
      id: "profiling",
      label: "In [7]",
      language: "python",
      note: "Generate the curated exploratory notebook section.",
      output: "analysis",
      title: "Run targeted exploration",
      code: `eda = cleaned.explore(\n    target="churn",\n    compare=["monthly_charge", "tenure_months", "support_calls"],\n)\neda.summary()`,
    },
    {
      id: "visuals",
      label: "In [8]",
      language: "python",
      note: "Pin correlations and class balance directly below the code cell.",
      output: "analytics",
      title: "Render exploratory outputs",
      code: `eda.correlation_matrix()\neda.distribution("churn")\neda.segment("contract")`,
    },
    {
      id: "eda-preview",
      label: "In [9]",
      language: "python",
      note: "Preview the rows driving the strongest exploratory signal.",
      output: "table",
      title: "Inspect high-signal samples",
      code: `cleaned.sort_values("monthly_charge", ascending=False).head(4)`,
    },
  ],
};
