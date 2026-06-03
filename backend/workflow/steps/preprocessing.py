from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..helpers import append_message, get_current_df, invalidate_downstream, record_step_result, set_current_df
from ..llm import invoke_llm
from ..types import AgentState


def _apply_preprocessing(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    cleaned = df.copy()
    missing_before = int(cleaned.isna().sum().sum())
    duplicate_rows = int(cleaned.duplicated().sum())
    if duplicate_rows:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for col in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            if cleaned[col].isna().any():
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        else:
            mode = cleaned[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)

    metrics = {
        "nulls_resolved": missing_before,
        "duplicate_rows_removed": duplicate_rows,
        "rows_retained": cleaned.shape[0],
        "ready_columns": cleaned.shape[1],
    }
    return cleaned, metrics


def run_preprocessing(state: AgentState, data_store_ref: Dict[str, Any]) -> AgentState:
    action = state.get("user_action")
    user_message = state.get("user_message", "")

    if action == "approve":
        transformed_df, metrics = _apply_preprocessing(get_current_df(data_store_ref))
        set_current_df(data_store_ref, transformed_df)
        invalidate_downstream(state, data_store_ref, "preprocessing")
        analysis = (
            "Great, preprocessing completed! I handled the missing values, removed duplicate rows, "
            "and preserved a cleaned dataframe artifact for the next stage. Should we move to "
            "Exploratory Data Analysis (EDA) next?"
        )
        record_step_result(state, data_store_ref, "preprocessing", analysis, transformed_df, metrics)
        append_message(state, analysis)
        state["current_step"] = "eda"
        state["status"] = "awaiting_human"
        return state

    prompt = f"User feedback on preprocessing: {user_message}. Provide an updated recommended plan."
    fallback = (
        "## Updated preprocessing plan\n"
        f"- User request: {user_message or 'Refine the current cleaning strategy.'}\n"
        "- Revisit imputation rules and keep the transformations localized to preprocessing.\n"
        "- Downstream EDA and modeling artifacts will remain unchanged until preprocessing is approved.\n"
    )
    analysis = invoke_llm(prompt, fallback)
    df = get_current_df(data_store_ref)
    record_step_result(state, data_store_ref, "preprocessing", analysis, df, {})
    append_message(state, analysis)
    state["status"] = "awaiting_human"
    return state
