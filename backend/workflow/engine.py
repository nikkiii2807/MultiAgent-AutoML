from __future__ import annotations

from typing import Any, Dict

from .helpers import ensure_session_artifacts, ensure_state_defaults
from .steps import run_eda, run_evaluation, run_ingestion, run_modeling, run_preprocessing, run_standardization
from .types import AgentState


async def process_agent_workflow(
    session_id: str,
    current_state: Dict[str, Any],
    user_message: str,
    action: str,
    data_store_ref: Dict[str, Any],
) -> Dict[str, Any]:
    agent_state: AgentState = {
        "messages": current_state.get("messages", []),
        "current_step": current_state.get("current_step", "ingestion"),
        "status": current_state.get("status", "working"),
        "dataset_info": current_state.get("dataset_info", {}),
        "user_action": action,
        "user_message": user_message,
        "metrics": current_state.get("metrics", {}),
        "step_results": current_state.get("step_results", {}),
    }
    ensure_state_defaults(agent_state)
    ensure_session_artifacts(data_store_ref)

    step = agent_state["current_step"]
    if step == "ingestion":
        new_state = run_ingestion(agent_state, data_store_ref)
    elif step == "preprocessing":
        new_state = run_preprocessing(agent_state, data_store_ref)
    elif step == "eda":
        new_state = run_eda(agent_state, data_store_ref)
    elif step == "standardization":
        new_state = run_standardization(agent_state, data_store_ref)
    elif step == "modeling":
        new_state = run_modeling(agent_state, data_store_ref)
    else:
        new_state = run_evaluation(agent_state, data_store_ref)

    return dict(new_state)
