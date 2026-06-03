import type { PreviewStageDefinition } from "../types";
import { SAMPLE_FEATURE_ROWS } from "../shared";

export const evaluationNotebook: PreviewStageDefinition = {
  stageSummary:
    "Evaluation closes the notebook with the final performance story, production recommendation, and decision-ready evidence.",
  insight:
    "Accuracy, F1, and the prediction mix are surfaced as the headline outputs so the last notebook section feels immediately actionable.",
  analysis: `## Final evaluation

The best model is ready for review with strong balance between precision and recall.

- **Accuracy:** 94.2%
- **F1 score:** 91.8%
- **Recommended model:** Gradient Boosting
- **Operational note:** prediction latency remains inside the target window for interactive scoring

This closing notebook section is intentionally executive-friendly while still feeling like a real analyst workflow.`,
  metrics: {
    accuracy: 0.942,
    f1_score: 0.918,
    roc_auc: 0.955,
    inference_latency_ms: 43,
  },
  charts: [
    {
      id: "evaluation-comparison",
      title: "Model Performance Comparison",
      type: "bar",
      data: [
        { name: "Gradient Boost", accuracy: 0.942, f1: 0.918, precision: 0.921 },
        { name: "Random Forest", accuracy: 0.934, f1: 0.907, precision: 0.914 },
        { name: "Logistic Reg.", accuracy: 0.881, f1: 0.851, precision: 0.842 },
      ],
      xKey: "name",
      bars: [
        { key: "accuracy", color: null, label: "Accuracy" },
        { key: "f1", color: null, label: "F1 score" },
        { key: "precision", color: null, label: "Precision" },
      ],
    },
    {
      id: "evaluation-feature-importance",
      title: "Feature Importance",
      type: "bar",
      data: [
        { name: "tenure_interaction", importance: 0.39 },
        { name: "monthly_charge_z", importance: 0.31 },
        { name: "support_calls_z", importance: 0.18 },
        { name: "contract_monthly", importance: 0.12 },
      ],
      xKey: "name",
      bars: [{ key: "importance", color: null, label: "Importance" }],
    },
    {
      id: "evaluation-prediction-breakdown",
      title: "Prediction Breakdown",
      type: "pie",
      data: [
        { name: "True Positive", value: 45 },
        { name: "True Negative", value: 42 },
        { name: "False Positive", value: 5 },
        { name: "False Negative", value: 8 },
      ],
    },
  ],
  dataPreview: SAMPLE_FEATURE_ROWS,
  cells: [
    {
      id: "evaluation-story",
      label: "In [16]",
      language: "python",
      note: "Compose the final model recommendation in notebook form.",
      output: "analysis",
      title: "Summarize evaluation findings",
      code: `report = leaderboard.evaluate(\n    metrics=["accuracy", "f1_score", "roc_auc"],\n    explain=True,\n)\nreport.summary()`,
    },
    {
      id: "evaluation-visuals",
      label: "In [17]",
      language: "python",
      note: "Show the final scorecards, charts, and breakdown outputs.",
      output: "analytics",
      title: "Render final scorecards",
      code: `report.scorecards()\nreport.model_comparison()\nreport.prediction_breakdown()`,
    },
    {
      id: "evaluation-preview",
      label: "In [18]",
      language: "python",
      note: "Pin the prediction inputs used for the final decision trace.",
      output: "table",
      title: "Inspect scored feature rows",
      code: `report.sample_predictions().head(4)`,
    },
  ],
};
