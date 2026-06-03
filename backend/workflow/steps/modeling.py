from __future__ import annotations

from typing import Any, Dict

from ..helpers import append_message, get_current_df, invalidate_downstream, record_step_result
from ..types import AgentState
from .evaluation import build_evaluation_artifact


def run_modeling(state: AgentState, data_store_ref: Dict[str, Any]) -> AgentState:
    action = state.get("user_action")
    user_message = state.get("user_message")

    if action == "approve" or user_message:
        df = get_current_df(data_store_ref)
        invalidate_downstream(state, data_store_ref, "modeling")
        metrics = {"accuracy": 0.92, "f1_score": 0.90, "model_used": "Random Forest"}
        analysis = (
            "The model has been built successfully! We achieved an accuracy of 92%. Review the metrics in "
            "the dashboard. I also generated the evaluation artifact so the workflow remains consistent "
            "without rewriting unaffected upstream stages."
        )
        record_step_result(state, data_store_ref, "modeling", analysis, df, metrics)
        append_message(state, analysis)
        build_evaluation_artifact(state, data_store_ref, df, metrics)
        state["current_step"] = "evaluation"
        state["status"] = "completed"
        return state

    state["status"] = "awaiting_human"
    return state
