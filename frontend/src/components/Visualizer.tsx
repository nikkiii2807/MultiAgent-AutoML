"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  PIPELINE_STAGES,
  getStageStatusLabel,
  getStageVisualStatus,
  type ChartConfig,
  type StageId,
  type StepResults,
} from "../lib/studio";

interface VisualizerProps {
  activeTab: StageId;
  currentStage: StageId;
  hasSession: boolean;
  isCollapsed: boolean;
  onToggleNotebook: () => void;
  status: string;
  stepResults: StepResults;
}

interface NotebookCellDefinition {
  id: string;
  code: string;
  label: string;
  language: string;
  note: string;
  output: "analysis" | "analytics" | "table";
  title: string;
}

interface PreviewStageDefinition {
  analysis: string;
  cells: NotebookCellDefinition[];
  dataPreview: string;
  insight: string;
  metrics: Record<string, string | number>;
  stageSummary: string;
  charts: ChartConfig[];
}

const PIE_COLORS = ["#3b82f6", "#60a5fa", "#93c5fd", "#cbd5e1", "#94a3b8", "#64748b"];
const BAR_COLORS = ["#3b82f6", "#60a5fa", "#94a3b8", "#cbd5e1"];

const SAMPLE_SOURCE_ROWS = JSON.stringify([
  {
    churn: "Yes",
    contract: "Monthly",
    customer_id: "C-1024",
    monthly_charge: 79.9,
    support_calls: 3,
    tenure_months: 18,
  },
  {
    churn: "No",
    contract: "Annual",
    customer_id: "C-1098",
    monthly_charge: 54.2,
    support_calls: 1,
    tenure_months: 42,
  },
  {
    churn: "No",
    contract: "Annual",
    customer_id: "C-1137",
    monthly_charge: 61.4,
    support_calls: 0,
    tenure_months: 57,
  },
  {
    churn: "Yes",
    contract: "Monthly",
    customer_id: "C-1184",
    monthly_charge: 98.1,
    support_calls: 4,
    tenure_months: 7,
  },
]);

const SAMPLE_FEATURE_ROWS = JSON.stringify([
  {
    churn_flag: 1,
    contract_monthly: 1,
    monthly_charge_z: 0.92,
    support_calls_z: 0.76,
    tenure_interaction: 1.24,
    tenure_months_z: -0.48,
  },
  {
    churn_flag: 0,
    contract_monthly: 0,
    monthly_charge_z: -0.31,
    support_calls_z: -0.52,
    tenure_interaction: -0.68,
    tenure_months_z: 0.87,
  },
  {
    churn_flag: 0,
    contract_monthly: 0,
    monthly_charge_z: 0.02,
    support_calls_z: -0.95,
    tenure_interaction: 0.75,
    tenure_months_z: 1.12,
  },
  {
    churn_flag: 1,
    contract_monthly: 1,
    monthly_charge_z: 1.38,
    support_calls_z: 1.27,
    tenure_interaction: -1.08,
    tenure_months_z: -1.34,
  },
]);

const PREVIEW_NOTEBOOKS: Record<StageId, PreviewStageDefinition> = {
  ingestion: {
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
  },
  preprocessing: {
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
  },
  eda: {
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
  },
  standardization: {
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
  },
  modeling: {
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
  },
  evaluation: {
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
  },
};

function formatMetricLabel(metricKey: string): string {
  return metricKey
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatMetricValue(metricKey: string, metricValue: string | number): string {
  if (typeof metricValue === "string") {
    return metricValue;
  }

  const loweredKey = metricKey.toLowerCase();

  if (
    loweredKey.includes("accuracy") ||
    loweredKey.includes("f1") ||
    loweredKey.includes("precision") ||
    loweredKey.includes("recall") ||
    loweredKey.includes("auc")
  ) {
    return `${(metricValue * 100).toFixed(1)}%`;
  }

  if (loweredKey.includes("latency") || loweredKey.endsWith("_ms")) {
    return `${metricValue.toFixed(0)} ms`;
  }

  if (Number.isInteger(metricValue)) {
    return metricValue.toLocaleString();
  }

  if (metricValue >= 10) {
    return metricValue.toFixed(1);
  }

  return metricValue.toFixed(2);
}

function parseRows(rawPreview: string | null): Array<Record<string, unknown>> {
  if (!rawPreview) {
    return [];
  }

  try {
    const parsed = JSON.parse(rawPreview);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function renderTableValue(value: unknown): string {
  if (value === null) {
    return "null";
  }

  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }

  return String(value);
}

function ChartCard({ chart }: { chart: ChartConfig }) {
  if (chart.type === "pie") {
    return (
      <article className="chart-card">
        <div className="chart-card-header">
          <div>
            <p className="chart-card-label">Output cell</p>
            <h4>{chart.title}</h4>
          </div>
        </div>

        <div className="chart-card-body">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                animationDuration={700}
                data={chart.data}
                dataKey="value"
                innerRadius={62}
                nameKey="name"
                outerRadius={104}
                stroke="rgba(10, 16, 34, 0.65)"
                strokeWidth={2}
              >
                {chart.data.map((_, index) => (
                  <Cell key={`${chart.id}-slice-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "#ffffff",
                  border: "1px solid #d8dee9",
                  borderRadius: "14px",
                  color: "#0f172a",
                }}
              />
              <Legend wrapperStyle={{ color: "#64748b", fontSize: "12px" }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </article>
    );
  }

  const bars = chart.bars ?? [{ key: "value", color: null, label: "Value" }];

  return (
    <article className="chart-card">
      <div className="chart-card-header">
        <div>
          <p className="chart-card-label">Output cell</p>
          <h4>{chart.title}</h4>
        </div>
      </div>

      <div className="chart-card-body">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chart.data} margin={{ left: -16, right: 8, top: 6 }}>
            <CartesianGrid stroke="#e8edf5" strokeDasharray="3 3" />
            <XAxis
              axisLine={{ stroke: "#d8dee9" }}
              dataKey={chart.xKey ?? "name"}
              tick={{ fill: "#64748b", fontSize: 12 }}
              tickLine={false}
            />
            <YAxis
              axisLine={{ stroke: "#d8dee9" }}
              tick={{ fill: "#64748b", fontSize: 12 }}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "#ffffff",
                border: "1px solid #d8dee9",
                borderRadius: "14px",
                color: "#0f172a",
              }}
              cursor={{ fill: "rgba(59, 130, 246, 0.08)" }}
            />
            <Legend wrapperStyle={{ color: "#64748b", fontSize: "12px" }} />
            {bars.map((bar, index) => (
              <Bar
                key={bar.key}
                animationDuration={700}
                dataKey={bar.key}
                fill={BAR_COLORS[index % BAR_COLORS.length]}
                maxBarSize={44}
                name={bar.label}
                radius={[10, 10, 2, 2]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </article>
  );
}

function AnalysisBlock({
  insight,
  markdown,
}: {
  insight: string;
  markdown: string;
}) {
  return (
    <div className="output-card output-card-rich">
      <div className="output-card-header">
        <span className="output-label">Out[1]</span>
        <span className="output-chip">Narrative output</span>
      </div>
      <div className="analysis-callout">
        <span className="analysis-callout-label">Assistant note</span>
        <p>{insight}</p>
      </div>
      <div className="analysis-markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
      </div>
    </div>
  );
}

function AnalyticsBlock({
  charts,
  metrics,
}: {
  charts: ChartConfig[];
  metrics: Record<string, string | number>;
}) {
  return (
    <div className="output-stack">
      <div className="metrics-grid">
        {Object.entries(metrics).map(([metricKey, metricValue]) => (
          <article key={metricKey} className="metric-card">
            <span className="metric-label">{formatMetricLabel(metricKey)}</span>
            <strong className="metric-value">{formatMetricValue(metricKey, metricValue)}</strong>
          </article>
        ))}
      </div>

      <div className="charts-grid">
        {charts.map((chart) => (
          <ChartCard key={chart.id} chart={chart} />
        ))}
      </div>
    </div>
  );
}

function TableBlock({ rows }: { rows: Array<Record<string, unknown>> }) {
  if (rows.length === 0) {
    return (
      <div className="output-card">
        <div className="output-card-header">
          <span className="output-label">Out[3]</span>
          <span className="output-chip">Table output</span>
        </div>
        <div className="table-empty-state">The stage has not produced a table preview yet.</div>
      </div>
    );
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="output-card">
      <div className="output-card-header">
        <span className="output-label">Out[3]</span>
        <span className="output-chip">Notebook table</span>
      </div>
      <div className="table-wrap">
        <table className="notebook-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={`row-${rowIndex}`}>
                {columns.map((column) => (
                  <td key={`${rowIndex}-${column}`}>{renderTableValue(row[column])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Visualizer({
  activeTab,
  currentStage,
  hasSession,
  isCollapsed,
  onToggleNotebook,
  status,
  stepResults,
}: VisualizerProps) {
  const preview = PREVIEW_NOTEBOOKS[activeTab];
  const liveResult = stepResults[activeTab];
  const stageStatus = getStageVisualStatus(activeTab, currentStage, status, stepResults);
  const activeStage = PIPELINE_STAGES.find((stage) => stage.id === activeTab) ?? PIPELINE_STAGES[0];
  const [expandedCells, setExpandedCells] = useState<Record<string, boolean>>({});

  const mergedMetrics =
    liveResult && Object.keys(liveResult.metrics).length > 0 ? liveResult.metrics : preview.metrics;
  const mergedCharts =
    liveResult && liveResult.charts.length > 0 ? liveResult.charts : preview.charts;
  const mergedAnalysis = liveResult?.analysis ?? preview.analysis;
  const mergedRows = parseRows(liveResult?.data_preview ?? preview.dataPreview);

  function toggleCell(cellKey: string) {
    setExpandedCells((previous) => ({
      ...previous,
      [cellKey]: !(previous[cellKey] ?? true),
    }));
  }

  return (
    <div className={`workspace-surface${isCollapsed ? " is-hidden" : ""}`}>
      <div className="workspace-toolbar">
        <div className="workspace-file-tab">
          <span className="workspace-file-pill">Notebook</span>
          <div>
            <p className="workspace-file-label">{activeStage.label}</p>
            <h2>{activeStage.notebookLabel}</h2>
          </div>
        </div>

        <div className="workspace-toolbar-pills">
          <span className={`workspace-mode-pill${liveResult ? " is-live" : ""}`}>
            {liveResult ? "Live output" : hasSession ? "Preview until this stage runs" : "Preview"}
          </span>
          <span className={`workspace-status-pill is-${stageStatus}`}>
            {getStageStatusLabel(stageStatus)}
          </span>
          <button type="button" className="chrome-button" onClick={onToggleNotebook}>
            Hide Panel
          </button>
        </div>
      </div>

      <div className="workspace-scroll">
        <section className="workspace-summary">
          <div>
            <p className="notebook-hero-eyebrow">{activeStage.helper}</p>
            <h3>{activeStage.prompt}</h3>
          </div>
          <p className="workspace-summary-copy">{preview.stageSummary}</p>
        </section>

        {preview.cells.map((cell) => {
          const cellKey = `${activeTab}-${cell.id}`;
          const isExpanded = expandedCells[cellKey] ?? true;

          return (
            <section key={cellKey} className="notebook-block">
              <button
                type="button"
                className="code-cell-header"
                onClick={() => toggleCell(cellKey)}
              >
                <div className="code-cell-meta">
                  <span className="code-cell-index">{cell.label}</span>
                  <div>
                    <p className="code-cell-title">{cell.title}</p>
                    <span className="code-cell-note">{cell.note}</span>
                  </div>
                </div>

                <div className="code-cell-actions">
                  <span className="code-language-tag">{cell.language}</span>
                  <span className="code-toggle">{isExpanded ? "Collapse" : "Expand"}</span>
                </div>
              </button>

              {isExpanded && (
                <div className="code-cell-body">
                  <pre>
                    <code>{cell.code}</code>
                  </pre>
                </div>
              )}

              <div className="output-cell">
                {cell.output === "analysis" && (
                  <AnalysisBlock insight={preview.insight} markdown={mergedAnalysis} />
                )}
                {cell.output === "analytics" && (
                  <AnalyticsBlock charts={mergedCharts} metrics={mergedMetrics} />
                )}
                {cell.output === "table" && <TableBlock rows={mergedRows} />}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
