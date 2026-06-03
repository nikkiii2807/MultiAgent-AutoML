from __future__ import annotations

from typing import Any, Dict

from ..helpers import append_message, get_current_df, record_step_result
from ..llm import invoke_llm
from ..types import AgentState


def run_ingestion(state: AgentState, data_store_ref: Dict[str, Any]) -> AgentState:
    info = state["dataset_info"]
    prompt = f"""You are an Expert Data Scientist AI assistant. The user has uploaded a dataset. Analyze it and provide a structured report.

Dataset Overview:
- Rows: {info.get('rows')}, Columns: {info.get('cols')}
- Column names: {info.get('columns')}
- Sample data (first 5 rows): {info.get('head')}
- Statistical summary: {info.get('summary')}

Respond in structured Markdown with dataset overview, data quality assessment, and recommended preprocessing steps.
"""
    fallback = (
        "## Dataset Overview\n"
        f"- Rows: **{info.get('rows', 0)}**\n"
        f"- Columns: **{info.get('cols', 0)}**\n"
        f"- Candidate fields: `{', '.join(info.get('columns', [])[:8])}`\n\n"
        "## Data Quality Assessment\n"
        "- Review missing values and column types before transforming the data.\n"
        "- Confirm whether any identifier or target column should be excluded from feature preparation.\n\n"
        "## Recommended Preprocessing Steps\n"
        "1. Impute missing numeric and categorical values.\n"
        "2. Normalize inconsistent data types.\n"
        "3. Remove duplicate rows if present.\n"
        "4. Prepare the cleaned frame for EDA.\n"
    )
    analysis = invoke_llm(prompt, fallback)
    df = get_current_df(data_store_ref)
    record_step_result(state, data_store_ref, "ingestion", analysis, df, {})
    append_message(state, analysis)
    state["current_step"] = "preprocessing"
    state["status"] = "awaiting_human"
    return state
