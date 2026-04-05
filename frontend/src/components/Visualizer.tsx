"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell
} from "recharts";
import type { StepResult, ChartConfig } from "../app/page";

const STEPS = [
  { id: "ingestion", label: "Ingestion", icon: "📥" },
  { id: "preprocessing", label: "Preprocessing", icon: "🧹" },
  { id: "eda", label: "EDA", icon: "📊" },
  { id: "standardization", label: "Feature Eng.", icon: "⚙️" },
  { id: "modeling", label: "Modeling", icon: "🧠" },
  { id: "evaluation", label: "Evaluation", icon: "🏆" },
];

const PIE_COLORS = ["#7c6cf0", "#34d399", "#60a5fa", "#fbbf24", "#f87171", "#a78bfa", "#fb923c", "#2dd4bf"];
const AUTO_BAR_COLORS = ["#7c6cf0", "#34d399", "#60a5fa", "#fbbf24", "#f87171", "#a78bfa"];

interface VisualizerProps {
  currentStep: string;
  activeTab: string;
  setActiveTab: (tab: string) => void;
  stepResults: Record<string, StepResult>;
  status: string;
}

function ChartCard({ chart }: { chart: ChartConfig }) {
  if (chart.type === "pie") {
    return (
      <div className="card animate-fade-in-up">
        <div className="card-header">
          <h4>{chart.title}</h4>
          <span className="card-badge badge-info">Chart</span>
        </div>
        <div className="card-body" style={{ display: "flex", justifyContent: "center" }}>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={chart.data}
                cx="50%"
                cy="50%"
                outerRadius={100}
                innerRadius={50}
                dataKey="value"
                nameKey="name"
                label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                labelLine={true}
                stroke="rgba(0,0,0,0.3)"
                strokeWidth={1}
              >
                {chart.data.map((_: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "#1a1d26",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                  color: "#eef0f6",
                  fontSize: "13px"
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  }

  // Bar chart
  const bars = chart.bars || [{ key: "value", color: "#7c6cf0", label: "Value" }];

  return (
    <div className="card animate-fade-in-up">
      <div className="card-header">
        <h4>{chart.title}</h4>
        <span className="card-badge badge-info">Chart</span>
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chart.data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey={chart.xKey || "name"}
              tick={{ fill: "#8b90a5", fontSize: 12 }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#8b90a5", fontSize: 12 }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "#1a1d26",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "8px",
                color: "#eef0f6",
                fontSize: "13px"
              }}
              cursor={{ fill: "rgba(124,108,240,0.08)" }}
            />
            <Legend
              wrapperStyle={{ fontSize: "12px", color: "#8b90a5" }}
            />
            {bars.map((bar, idx) => (
              <Bar
                key={bar.key}
                dataKey={bar.key}
                name={bar.label}
                fill={bar.color || AUTO_BAR_COLORS[idx % AUTO_BAR_COLORS.length]}
                radius={[4, 4, 0, 0]}
                maxBarSize={40}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function Visualizer({ currentStep, activeTab, setActiveTab, stepResults, status }: VisualizerProps) {
  const currentStepIdx = STEPS.findIndex(s => s.id === currentStep);
  const activeResult = stepResults[activeTab];

  const hasData = Object.keys(stepResults).length > 0;

  let previewRows: any[] = [];
  if (activeResult?.data_preview) {
    try { previewRows = JSON.parse(activeResult.data_preview); } catch {}
  }

  const metrics = activeResult?.metrics || {};
  const hasMetrics = Object.keys(metrics).length > 0;
  const charts = activeResult?.charts || [];

  if (!hasData) {
    return (
      <>
        <div className="step-tabs">
          {STEPS.map((step, idx) => (
            <button key={step.id} className="step-tab disabled" disabled>
              <span className="step-tab-number">{idx + 1}</span>
              {step.label}
            </button>
          ))}
        </div>
        <div className="workspace-content">
          <div className="empty-state">
            <div className="empty-icon">📈</div>
            <h3>No Data Yet</h3>
            <p>Upload a CSV dataset in the chat panel to begin the automated analytics pipeline.</p>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      {/* Clickable Step Tabs */}
      <div className="step-tabs">
        {STEPS.map((step, idx) => {
          const isCompleted = idx < currentStepIdx;
          const isActive = step.id === activeTab;
          const hasResult = !!stepResults[step.id];
          const isDisabled = !hasResult;

          let className = "step-tab";
          if (isActive) className += " active";
          if (isCompleted && hasResult) className += " completed";
          if (isDisabled) className += " disabled";

          return (
            <button
              key={step.id}
              className={className}
              onClick={() => { if (hasResult) setActiveTab(step.id); }}
              disabled={isDisabled}
              title={hasResult ? `View ${step.label} results` : "Not yet reached"}
            >
              <span className="step-tab-number">
                {isCompleted && hasResult ? "✓" : idx + 1}
              </span>
              {step.icon} {step.label}
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div className="workspace-content">
        {activeResult ? (
          <div className="animate-fade-in-up" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>

            {/* Metrics */}
            {hasMetrics && (
              <div className="metrics-grid">
                {Object.entries(metrics).map(([key, val]) => (
                  <div key={key} className="metric-card">
                    <div className="metric-label">{key.replace(/_/g, " ")}</div>
                    <div className="metric-value">
                      {typeof val === "number" ? (val < 1 && val > 0 ? `${(val * 100).toFixed(1)}%` : val.toFixed(2)) : String(val)}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Charts */}
            {charts.length > 0 && (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(450px, 1fr))", gap: "16px" }}>
                {charts.map((chart) => (
                  <ChartCard key={chart.id} chart={chart} />
                ))}
              </div>
            )}

            {/* Analysis */}
            <div className="card">
              <div className="card-header">
                <h4>
                  {STEPS.find(s => s.id === activeTab)?.icon}{" "}
                  {STEPS.find(s => s.id === activeTab)?.label} Analysis
                </h4>
                <span className={`card-badge ${status === "completed" && activeTab === "evaluation" ? "badge-success" : "badge-info"}`}>
                  {status === "completed" && activeTab === "evaluation" ? "Complete" : "Analysis"}
                </span>
              </div>
              <div className="card-body analysis-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {activeResult.analysis}
                </ReactMarkdown>
              </div>
            </div>

            {/* Data Table */}
            {previewRows.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <h4>📋 Data Preview</h4>
                  <span className="card-badge badge-info">{previewRows.length} rows</span>
                </div>
                <div className="data-table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {Object.keys(previewRows[0]).map(col => (
                          <th key={col}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewRows.map((row: any, i: number) => (
                        <tr key={i}>
                          {Object.values(row).map((val: any, j: number) => (
                            <td key={j}>
                              {val === null ? <span style={{ color: "var(--error)", fontStyle: "italic" }}>null</span> :
                               typeof val === "number" ? val.toFixed(2) : String(val)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="no-data">
            <div className="no-data-icon">🔍</div>
            <p>No results for this step yet.</p>
          </div>
        )}
      </div>
    </>
  );
}
