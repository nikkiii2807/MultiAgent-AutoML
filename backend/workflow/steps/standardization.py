from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..helpers import append_message, get_current_df, invalidate_downstream, record_step_result, set_current_df
from ..types import AgentState


def _apply_standardization(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    transformed = df.copy()
    numeric_cols = transformed.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return transformed, {"scaled_columns": 0, "engineered_features": 0, "selected_features": transformed.shape[1]}

    scaler = StandardScaler()
    transformed[numeric_cols] = scaler.fit_transform(transformed[numeric_cols].fillna(0))
    metrics = {
        "scaled_columns": len(numeric_cols),
        "engineered_features": len(numeric_cols),
        "selected_features": transformed.shape[1],
        "feature_store_ready": "Yes",
    }
    return transformed, metrics


def run_standardization(state: AgentState, data_store_ref: Dict[str, Any]) -> AgentState:
    action = state.get("user_action")

    if action == "approve":
        transformed_df, metrics = _apply_standardization(get_current_df(data_store_ref))
        set_current_df(data_store_ref, transformed_df)
        invalidate_downstream(state, data_store_ref, "standardization")
        analysis = (
            "Features are standardized and the transformed dataset has been saved as the feature-engineering "
            "artifact. We are ready for modeling. Based on this tabular data, I recommend testing a Random "
            "Forest Classifier and a Gradient Boosting Classifier. Which model would you prefer, or should "
            "I auto-select the best?"
        )
        record_step_result(state, data_store_ref, "standardization", analysis, transformed_df, metrics)
        append_message(state, analysis)
        state["current_step"] = "modeling"
        state["status"] = "awaiting_human"
        return state

    analysis = f"Understood, I will keep the current feature-engineering stage open while we consider: {state.get('user_message', '')}"
    df = get_current_df(data_store_ref)
    record_step_result(state, data_store_ref, "standardization", analysis, df, {})
    append_message(state, analysis)
    state["status"] = "awaiting_human"
    return state
