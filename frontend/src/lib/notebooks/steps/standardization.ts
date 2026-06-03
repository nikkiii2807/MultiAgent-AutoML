import type { PreviewStageDefinition } from "../types";
import { SAMPLE_FEATURE_ROWS } from "../shared";

export const standardizationNotebook: PreviewStageDefinition = {
  stageSummary:
    "Feature engineering combines scaling, feature synthesis, and selection in notebook cells that stay readable to humans.",
  insight:
    "Renaming this stage visually to Feature Engineering makes the workflow clearer for end users, while the backend can still keep its original step identifier.",
  analysis: `## Feature engineering pass

The notebook now standardizes the strongest numeric columns and builds interaction features for the AutoML search.

- Numeric signals are centered and scaled
- Contract behavior and tenure are combined into cross-features
- Low-value columns are filtered before training
- The resulting matrix is easier to compare across candidate models

The interface keeps these operations explicit so the user can trust the automation instead of guessing what happened.`,
  metrics: {
    scaled_columns: 8,
    engineered_features: 14,
    selected_features: 17,
    feature_store_ready: "Yes",
  },
  charts: [
    {
      id: "feature-scale",
      title: "Before vs After Standardization",
      type: "bar",
      data: [
        { name: "monthly_charge", original_mean: 68.4, scaled_mean: 0.01 },
        { name: "tenure_months", original_mean: 29.8, scaled_mean: 0.0 },
        { name: "support_calls", original_mean: 1.9, scaled_mean: -0.02 },
      ],
      xKey: "name",
      bars: [
        { key: "original_mean", color: null, label: "Original mean" },
        { key: "scaled_mean", color: null, label: "Scaled mean" },
      ],
    },
    {
      id: "feature-groups",
      title: "Engineered Feature Families",
      type: "pie",
      data: [
        { name: "Scaled numeric", value: 8 },
        { name: "Interaction terms", value: 4 },
        { name: "Encoded categories", value: 5 },
      ],
    },
  ],
  dataPreview: SAMPLE_FEATURE_ROWS,
  cells: [
    {
      id: "engineering-plan",
      label: "In [10]",
      language: "python",
      note: "Scale, transform, and synthesize candidate features.",
      output: "analysis",
      title: "Build engineered feature set",
      code: `features = (\n    cleaned\n    .standardize(columns=["monthly_charge", "tenure_months", "support_calls"])\n    .interactions([("monthly_charge", "tenure_months")])\n    .select_top_k(k=17)\n)`,
    },
    {
      id: "engineering-metrics",
      label: "In [11]",
      language: "python",
      note: "Track how the feature matrix changed before modeling.",
      output: "analytics",
      title: "Measure feature impact",
      code: `features.summary()\nfeatures.compare_scale()\nfeatures.feature_groups()`,
    },
    {
      id: "engineering-preview",
      label: "In [12]",
      language: "python",
      note: "Inspect the transformed rows that feed the model search.",
      output: "table",
      title: "Preview engineered matrix",
      code: `features.head(4)\nfeatures.schema()`,
    },
  ],
};
