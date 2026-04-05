from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
import os
import uuid
import json


# Initialize FastAPI app
app = FastAPI(title="Multi-Agent AutoML API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_STORE = {}
STATE_STORE = {}


def generate_charts(df: pd.DataFrame, step: str) -> list:
    """Generate chart data based on the current pipeline step."""
    charts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if step in ("ingestion", "preprocessing"):
        # Chart 1: Missing values per column
        missing = df.isnull().sum()
        if missing.sum() > 0:
            charts.append({
                "id": "missing_values",
                "title": "Missing Values by Column",
                "type": "bar",
                "data": [{"name": col, "value": int(val)} for col, val in missing.items() if val > 0],
                "xKey": "name",
                "bars": [{"key": "value", "color": "#f87171", "label": "Missing Count"}]
            })

        # Chart 2: Data types distribution
        dtype_counts = df.dtypes.astype(str).value_counts()
        charts.append({
            "id": "dtype_distribution",
            "title": "Column Data Types",
            "type": "pie",
            "data": [{"name": str(dtype), "value": int(count)} for dtype, count in dtype_counts.items()]
        })

        # Chart 3: Row count per numeric column (as value ranges / histogram-like)
        if len(numeric_cols) >= 2:
            stats_data = []
            for col in numeric_cols[:6]:
                stats_data.append({
                    "name": col,
                    "mean": round(float(df[col].mean()), 2),
                    "std": round(float(df[col].std()), 2) if df[col].std() == df[col].std() else 0,
                    "min": round(float(df[col].min()), 2),
                    "max": round(float(df[col].max()), 2),
                })
            charts.append({
                "id": "numeric_stats",
                "title": "Numeric Column Statistics",
                "type": "bar",
                "data": stats_data,
                "xKey": "name",
                "bars": [
                    {"key": "mean", "color": "#7c6cf0", "label": "Mean"},
                    {"key": "min", "color": "#60a5fa", "label": "Min"},
                    {"key": "max", "color": "#34d399", "label": "Max"},
                ]
            })

    elif step == "eda":
        # Chart 1: Correlation heatmap data (as a grouped bar for simplicity)
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            # Send as list of {col1, col2, value}
            corr_data = []
            for i, c1 in enumerate(numeric_cols[:5]):
                row = {"name": c1}
                for c2 in numeric_cols[:5]:
                    row[c2] = round(float(corr.loc[c1, c2]), 2)
                corr_data.append(row)
            charts.append({
                "id": "correlation_matrix",
                "title": "Correlation Matrix",
                "type": "bar",
                "data": corr_data,
                "xKey": "name",
                "bars": [{"key": c, "color": None, "label": c} for c in numeric_cols[:5]]
            })

        # Chart 2: Distribution of each numeric column (histogram approximation)
        for col in numeric_cols[:3]:
            try:
                counts, bin_edges = np.histogram(df[col].dropna(), bins=10)
                hist_data = []
                for j in range(len(counts)):
                    hist_data.append({
                        "range": f"{bin_edges[j]:.1f}-{bin_edges[j+1]:.1f}",
                        "count": int(counts[j])
                    })
                charts.append({
                    "id": f"distribution_{col}",
                    "title": f"Distribution: {col}",
                    "type": "bar",
                    "data": hist_data,
                    "xKey": "range",
                    "bars": [{"key": "count", "color": "#9b8afb", "label": "Count"}]
                })
            except Exception:
                pass

        # Chart 3: Categorical value counts
        for col in categorical_cols[:2]:
            vc = df[col].value_counts().head(10)
            charts.append({
                "id": f"value_counts_{col}",
                "title": f"Value Counts: {col}",
                "type": "pie",
                "data": [{"name": str(k), "value": int(v)} for k, v in vc.items()]
            })

    elif step == "standardization":
        # Show before/after scaling comparison (simulated)
        if len(numeric_cols) >= 2:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df[numeric_cols].fillna(0))
            compare_data = []
            for i, col in enumerate(numeric_cols[:6]):
                compare_data.append({
                    "name": col,
                    "original_mean": round(float(df[col].mean()), 2),
                    "scaled_mean": round(float(scaled[:, i].mean()), 4),
                    "original_std": round(float(df[col].std()), 2) if df[col].std() == df[col].std() else 0,
                    "scaled_std": round(float(scaled[:, i].std()), 4),
                })
            charts.append({
                "id": "scaling_comparison",
                "title": "Before vs After Standardization (Mean)",
                "type": "bar",
                "data": compare_data,
                "xKey": "name",
                "bars": [
                    {"key": "original_mean", "color": "#f87171", "label": "Original Mean"},
                    {"key": "scaled_mean", "color": "#34d399", "label": "Scaled Mean"},
                ]
            })
            charts.append({
                "id": "scaling_std",
                "title": "Before vs After Standardization (Std Dev)",
                "type": "bar",
                "data": compare_data,
                "xKey": "name",
                "bars": [
                    {"key": "original_std", "color": "#fbbf24", "label": "Original Std"},
                    {"key": "scaled_std", "color": "#60a5fa", "label": "Scaled Std"},
                ]
            })

    elif step in ("modeling", "evaluation"):
        # Simulated model comparison
        charts.append({
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
            ]
        })
        # Feature importance
        if len(numeric_cols) >= 2:
            importance = sorted(
                [{"name": col, "importance": round(np.random.uniform(0.05, 0.5), 3)} for col in numeric_cols],
                key=lambda x: x["importance"],
                reverse=True
            )
            charts.append({
                "id": "feature_importance",
                "title": "Feature Importance",
                "type": "bar",
                "data": importance,
                "xKey": "name",
                "bars": [{"key": "importance", "color": "#fbbf24", "label": "Importance"}]
            })
        # Confusion matrix as a simple bar
        charts.append({
            "id": "confusion_summary",
            "title": "Prediction Breakdown",
            "type": "pie",
            "data": [
                {"name": "True Positive", "value": 45},
                {"name": "True Negative", "value": 42},
                {"name": "False Positive", "value": 5},
                {"name": "False Negative", "value": 8},
            ]
        })

    return charts


@app.get("/")
def read_root():
    return {"message": "Multi-Agent AutoML Backend is running."}

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        session_id = str(uuid.uuid4())
        
        DATA_STORE[session_id] = {
            "raw": df,
            "current": df.copy(),
            "filename": file.filename
        }
        
        STATE_STORE[session_id] = {
            "messages": [],
            "current_step": "ingestion",
            "dataset_info": {
                "rows": df.shape[0],
                "cols": df.shape[1],
                "columns": df.columns.tolist(),
                "head": df.head(5).to_json(orient="records"),
                "summary": df.describe(include='all').to_json()
            },
            "status": "awaiting_human"
        }
        
        return {
            "message": "File uploaded successfully",
            "session_id": session_id,
            "columns": df.columns.tolist(),
            "rows": df.shape[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

class ChatMessage(BaseModel):
    message: str
    session_id: str
    action: Optional[str] = None

@app.post("/chat")
async def chat(data: ChatMessage):
    session_id = data.session_id
    if session_id not in STATE_STORE:
        raise HTTPException(status_code=404, detail="Session not found")
        
    state = STATE_STORE[session_id]
    state["messages"].append({"role": "user", "content": data.message})
    
    from agent import process_agent_workflow
    
    new_state = await process_agent_workflow(session_id, state, data.message, data.action, DATA_STORE[session_id])
    STATE_STORE[session_id] = new_state
    
    latest_msg = new_state["messages"][-1] if new_state["messages"] else {"role": "assistant", "content": "Error"}
    
    current_df = DATA_STORE[session_id]["current"]
    current_step = new_state["current_step"]
    
    # Generate charts for the step that just produced results
    steps_list = ["ingestion", "preprocessing", "eda", "standardization", "modeling", "evaluation"]
    step_idx = steps_list.index(current_step) if current_step in steps_list else 0
    producer_step = steps_list[step_idx - 1] if step_idx > 0 else current_step
    charts = generate_charts(current_df, producer_step)
    
    return {
        "reply": latest_msg["content"],
        "state": new_state["current_step"],
        "status": new_state["status"],
        "data_preview": current_df.head(10).to_json(orient="records") if current_df is not None else None,
        "metrics": new_state.get("metrics", {}),
        "charts": charts
    }

@app.get("/state/{session_id}")
async def get_state(session_id: str):
    if session_id not in STATE_STORE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = STATE_STORE[session_id]
    current_df = DATA_STORE[session_id].get("current")
    
    return {
        "current_step": state.get("current_step"),
        "status": state.get("status"),
        "dataset_info": state.get("dataset_info"),
        "metrics": state.get("metrics", {}),
        "data_preview": current_df.head(10).to_json(orient="records") if current_df is not None else None
    }
