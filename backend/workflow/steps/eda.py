from __future__ import annotations

from typing import Any, Dict

from ..helpers import append_message, get_current_df, record_step_result
from ..llm import invoke_llm
from ..types import AgentState


def run_eda(state: AgentState, data_store_ref: Dict[str, Any]) -> AgentState:
    action = state.get("user_action")
    user_message = state.get("user_message", "")
    df = get_current_df(data_store_ref)

    if action == "approve":
        metrics = {
            "numeric_columns": len(df.select_dtypes(include="number").columns),
            "categorical_columns": len(df.select_dtypes(exclude="number").columns),
            "rows_profiled": df.shape[0],
            "columns_profiled": df.shape[1],
        }
        analysis = (
            "I generated the Exploratory Data Analysis charts visible in the dashboard and stored them "
            "as the EDA notebook artifact. Next, I recommend standardizing the numerical features and "
            "creating modeling-ready inputs. Shall I proceed?"
        )
        record_step_result(state, data_store_ref, "eda", analysis, df, metrics)
        append_message(state, analysis)
        state["current_step"] = "standardization"
        state["status"] = "awaiting_human"
        return state

    prompt = f"User wants to explore something else: {user_message}. Answer them and suggest next steps."
    fallback = (
        "## Additional EDA focus\n"
        f"- Requested follow-up: {user_message or 'Explore different segments.'}\n"
        "- I can keep the current preprocessing artifact intact and only refresh the EDA findings.\n"
        "- Approve this stage when you are ready to move into feature engineering.\n"
    )
    analysis = invoke_llm(prompt, fallback)
    record_step_result(state, data_store_ref, "eda", analysis, df, {})
    append_message(state, analysis)
    state["status"] = "awaiting_human"
    return state
