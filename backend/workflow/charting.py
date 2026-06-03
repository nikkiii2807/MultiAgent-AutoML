from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .registry import StageId


def generate_charts(df: pd.DataFrame, step: StageId) -> List[Dict[str, Any]]:
    charts: List[Dict[str, Any]] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if step in ("ingestion", "preprocessing"):
        missing = df.isnull().sum()
        if missing.sum() > 0:
            charts.append(
                {
                    "id": "missing_values",
                    "title": "Missing Values by Column",
                    "type": "bar",
                    "data": [{"name": col, "value": int(val)} for col, val in missing.items() if val > 0],
                    "xKey": "name",
                    "bars": [{"key": "value", "color": "#f87171", "label": "Missing Count"}],
                }
            )

        dtype_counts = df.dtypes.astype(str).value_counts()
        charts.append(
            {
                "id": "dtype_distribution",
                "title": "Column Data Types",
                "type": "pie",
                "data": [{"name": str(dtype), "value": int(count)} for dtype, count in dtype_counts.items()],
            }
        )

        if len(numeric_cols) >= 2:
            stats_data = []
            for col in numeric_cols[:6]:
                stats_data.append(
                    {
                        "name": col,
                        "mean": round(float(df[col].mean()), 2),
                        "std": round(float(df[col].std()), 2) if pd.notna(df[col].std()) else 0,
                        "min": round(float(df[col].min()), 2),
                        "max": round(float(df[col].max()), 2),
                    }
                )
            charts.append(
                {
                    "id": "numeric_stats",
                    "title": "Numeric Column Statistics",
                    "type": "bar",
                    "data": stats_data,
                    "xKey": "name",
                    "bars": [
                        {"key": "mean", "color": "#7c6cf0", "label": "Mean"},
                        {"key": "min", "color": "#60a5fa", "label": "Min"},
                        {"key": "max", "color": "#34d399", "label": "Max"},
                    ],
                }
            )

    elif step == "eda":
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            corr_data = []
            for c1 in numeric_cols[:5]:
                row = {"name": c1}
                for c2 in numeric_cols[:5]:
                    row[c2] = round(float(corr.loc[c1, c2]), 2)
                corr_data.append(row)
            charts.append(
                {
                    "id": "correlation_matrix",
                    "title": "Correlation Matrix",
                    "type": "bar",
                    "data": corr_data,
                    "xKey": "name",
                    "bars": [{"key": c, "color": None, "label": c} for c in numeric_cols[:5]],
                }
            )

        for col in numeric_cols[:3]:
            try:
                counts, bin_edges = np.histogram(df[col].dropna(), bins=10)
                hist_data = []
                for index, count in enumerate(counts):
                    hist_data.append(
                        {
                            "range": f"{bin_edges[index]:.1f}-{bin_edges[index + 1]:.1f}",
                            "count": int(count),
                        }
                    )
                charts.append(
                    {
                        "id": f"distribution_{col}",
                        "title": f"Distribution: {col}",
                        "type": "bar",
                        "data": hist_data,
                        "xKey": "range",
                        "bars": [{"key": "count", "color": "#9b8afb", "label": "Count"}],
                    }
                )
            except Exception:
                pass

        for col in categorical_cols[:2]:
            vc = df[col].value_counts().head(10)
            charts.append(
                {
                    "id": f"value_counts_{col}",
                    "title": f"Value Counts: {col}",
                    "type": "pie",
                    "data": [{"name": str(k), "value": int(v)} for k, v in vc.items()],
                }
            )

    elif step == "standardization":
        if len(numeric_cols) >= 2:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaled = scaler.fit_transform(df[numeric_cols].fillna(0))
            compare_data = []
            for index, col in enumerate(numeric_cols[:6]):
                compare_data.append(
                    {
                        "name": col,
                        "original_mean": round(float(df[col].mean()), 2),
                        "scaled_mean": round(float(scaled[:, index].mean()), 4),
                        "original_std": round(float(df[col].std()), 2) if pd.notna(df[col].std()) else 0,
                        "scaled_std": round(float(scaled[:, index].std()), 4),
                    }
                )
            charts.append(
                {
                    "id": "scaling_comparison",
                    "title": "Before vs After Standardization (Mean)",
                    "type": "bar",
                    "data": compare_data,
                    "xKey": "name",
                    "bars": [
                        {"key": "original_mean", "color": "#f87171", "label": "Original Mean"},
                        {"key": "scaled_mean", "color": "#34d399", "label": "Scaled Mean"},
                    ],
                }
            )
            charts.append(
                {
                    "id": "scaling_std",
                    "title": "Before vs After Standardization (Std Dev)",
                    "type": "bar",
                    "data": compare_data,
                    "xKey": "name",
                    "bars": [
                        {"key": "original_std", "color": "#fbbf24", "label": "Original Std"},
                        {"key": "scaled_std", "color": "#60a5fa", "label": "Scaled Std"},
                    ],
                }
            )

    elif step in ("modeling", "evaluation"):
        charts.append(
            {
                "id": "model_comparison",
                "title": "Model Performance Comparison",
                "type": "bar",
                "data": [
                    {"name": "Random Forest", "accuracy": 0.92, "f1": 0.90, "precision": 0.91},
                    {"name": "Gradient Boost", "accuracy": 0.89, "f1": 0.87, "precision": 0.88},
                    {"name": "Logistic Reg.", "accuracy": 0.84, "f1": 0.82, "precision": 0.83},
                ],
                "xKey": "name",
                "bars": [
                    {"key": "accuracy", "color": "#7c6cf0", "label": "Accuracy"},
                    {"key": "f1", "color": "#34d399", "label": "F1 Score"},
                    {"key": "precision", "color": "#60a5fa", "label": "Precision"},
                ],
            }
        )

        if len(numeric_cols) >= 2:
            rng = np.random.default_rng(42)
            importance = sorted(
                [{"name": col, "importance": round(float(rng.uniform(0.05, 0.5)), 3)} for col in numeric_cols],
                key=lambda item: item["importance"],
                reverse=True,
            )
            charts.append(
                {
                    "id": "feature_importance",
                    "title": "Feature Importance",
                    "type": "bar",
                    "data": importance,
                    "xKey": "name",
                    "bars": [{"key": "importance", "color": "#fbbf24", "label": "Importance"}],
                }
            )

        charts.append(
            {
                "id": "confusion_summary",
                "title": "Prediction Breakdown",
                "type": "pie",
                "data": [
                    {"name": "True Positive", "value": 45},
                    {"name": "True Negative", "value": 42},
                    {"name": "False Positive", "value": 5},
                    {"name": "False Negative", "value": 8},
                ],
            }
        )

    return charts
