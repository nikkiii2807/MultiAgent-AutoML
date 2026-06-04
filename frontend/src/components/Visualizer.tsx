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
import { PREVIEW_NOTEBOOKS } from "../lib/notebooks";

interface VisualizerProps {
  activeTab: StageId;
  currentStage: StageId;
  hasSession: boolean;
  isCollapsed: boolean;
  onToggleNotebook: () => void;
  status: string;
  stepResults: StepResults;
}

const PIE_COLORS = ["#3b82f6", "#60a5fa", "#93c5fd", "#cbd5e1", "#94a3b8", "#64748b"];
const BAR_COLORS = ["#3b82f6", "#60a5fa", "#94a3b8", "#cbd5e1"];

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
      <div className="analysis-markdown readable-content">
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
  const [collapsedCells, setCollapsedCells] = useState<Record<string, boolean>>({});

  const mergedMetrics =
    liveResult && Object.keys(liveResult.metrics).length > 0 ? liveResult.metrics : preview.metrics;
  const mergedCharts =
    liveResult && liveResult.charts.length > 0 ? liveResult.charts : preview.charts;
  const mergedAnalysis = liveResult?.analysis ?? preview.analysis;
  const mergedRows = parseRows(liveResult?.data_preview ?? preview.dataPreview);

  const cellKeys = preview.cells.map((cell) => `${activeTab}-${cell.id}`);
  const expandedCount = cellKeys.filter((cellKey) => !collapsedCells[cellKey]).length;
  const allExpanded = expandedCount === cellKeys.length;

  function toggleCell(cellKey: string) {
    setCollapsedCells((previous) => ({
      ...previous,
      [cellKey]: !(previous[cellKey] ?? false),
    }));
  }

  function setAllCellsExpanded(nextExpanded: boolean) {
    if (nextExpanded) {
      setCollapsedCells({});
      return;
    }

    setCollapsedCells(Object.fromEntries(cellKeys.map((cellKey) => [cellKey, true])));
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
          <button
            type="button"
            className="chrome-button"
            onClick={() => setAllCellsExpanded(!allExpanded)}
          >
            {allExpanded ? "Collapse all cells" : "Expand all cells"}
          </button>
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
          <p className="workspace-summary-copy readable-content">{preview.stageSummary}</p>
        </section>

        {preview.cells.map((cell) => {
          const cellKey = `${activeTab}-${cell.id}`;
          const isExpanded = !(collapsedCells[cellKey] ?? false);

          return (
            <section
              key={cellKey}
              className={`notebook-block${isExpanded ? " is-expanded" : " is-collapsed"}`}
            >
              <button
                type="button"
                className="code-cell-header"
                aria-expanded={isExpanded}
                onClick={() => toggleCell(cellKey)}
              >
                <div className="code-cell-meta">
                  <span className="code-cell-chevron" aria-hidden="true">
                    {isExpanded ? "▾" : "▸"}
                  </span>
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
                <div className="notebook-cell-content">
                  <div className="code-cell-body">
                    <pre>
                      <code>{cell.code}</code>
                    </pre>
                  </div>

                  <div className="output-cell">
                    {cell.output === "analysis" && (
                      <AnalysisBlock insight={preview.insight} markdown={mergedAnalysis} />
                    )}
                    {cell.output === "analytics" && (
                      <AnalyticsBlock charts={mergedCharts} metrics={mergedMetrics} />
                    )}
                    {cell.output === "table" && <TableBlock rows={mergedRows} />}
                  </div>
                </div>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
