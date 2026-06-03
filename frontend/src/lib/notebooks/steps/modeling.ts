import type { PreviewStageDefinition } from "../types";
import { SAMPLE_FEATURE_ROWS } from "../shared";

export const modelingNotebook: PreviewStageDefinition = {
  stageSummary:
    "The modeling notebook feels like a VS Code experiment board, pairing an AutoML leaderboard with training artifacts inside notebook output cells.",
  insight:
    "Charts follow the same blue-to-pink theme as the active tabs and controls, so the whole studio reads like one coherent product surface.",
  analysis: `## Model search overview

The AutoML search evaluates a short, high-signal leaderboard rather than overwhelming the user with dozens of near-duplicate models.

- Gradient Boosting edges out Random Forest on balanced performance
- Random Forest remains competitive on interpretability
- Logistic Regression provides a lightweight baseline for comparison

This stage is tuned to look like a notebook experiment log, not a generic analytics dashboard.`,
  metrics: {
    accuracy: 0.942,
    f1_score: 0.918,
    best_model: "Gradient Boosting",
    train_runtime_min: 6.8,
  },
  charts: [
    {
      id: "model-comparison",
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
      id: "feature-importance",
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
      id: "prediction-breakdown",
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
      id: "search-space",
      label: "In [13]",
      language: "python",
      note: "Define the notebook cell that launches the AutoML search.",
      output: "analysis",
      title: "Launch model search",
      code: `leaderboard = features.train(\n    models=["gradient_boosting", "random_forest", "logistic_regression"],\n    metric="f1_score",\n    cv=5,\n)`,
    },
    {
      id: "leaderboard",
      label: "In [14]",
      language: "python",
      note: "Render the leaderboard, metrics cards, and feature diagnostics.",
      output: "analytics",
      title: "Inspect leaderboard outputs",
      code: `leaderboard.compare()\nleaderboard.feature_importance()\nleaderboard.prediction_breakdown()`,
    },
    {
      id: "training-preview",
      label: "In [15]",
      language: "python",
      note: "Preview the training matrix and best-model metadata.",
      output: "table",
      title: "Inspect training artifacts",
      code: `leaderboard.best_model.metadata()\nfeatures.head(4)`,
    },
  ],
};
