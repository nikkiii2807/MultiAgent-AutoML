from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

from .charting import generate_charts
from .registry import STAGE_DEFINITIONS, StageId, get_downstream_steps
from .types import AgentState, StepResult


def ensure_session_artifacts(data_store_ref: Dict[str, Any]) -> None:
    data_store_ref.setdefault("artifacts", {})
    data_store_ref["artifacts"].setdefault("datasets", {})
    data_store_ref["artifacts"].setdefault("step_outputs", {})
    data_store_ref.setdefault("current_dataset_step", "ingestion")


def ensure_state_defaults(state: AgentState) -> AgentState:
    state.setdefault("messages", [])
    state.setdefault("metrics", {})
    state.setdefault("step_results", {})
    return state


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_current_df(data_store_ref: Dict[str, Any]) -> pd.DataFrame:
    return data_store_ref["current"].copy()


def set_current_df(data_store_ref: Dict[str, Any], df: pd.DataFrame) -> None:
    data_store_ref["current"] = df.copy()


def record_step_result(
    state: AgentState,
    data_store_ref: Dict[str, Any],
    step: StageId,
    analysis: str,
    df: pd.DataFrame,
    metrics: Dict[str, Any] | None = None,
) -> StepResult:
    ensure_state_defaults(state)
    ensure_session_artifacts(data_store_ref)
    step_metrics = metrics or {}
    result: StepResult = {
        "analysis": analysis,
        "data_preview": df.head(10).to_json(orient="records") if df is not None else None,
        "metrics": step_metrics,
        "charts": generate_charts(df, step),
        "notebook_label": STAGE_DEFINITIONS[step]["notebook_label"],
        "updated_at": timestamp(),
    }
    state["step_results"][step] = result
    state["metrics"] = step_metrics
    data_store_ref["artifacts"]["step_outputs"][step] = result
    data_store_ref["artifacts"]["datasets"][step] = df.copy()
    data_store_ref["current_dataset_step"] = step
    return result


def invalidate_downstream(state: AgentState, data_store_ref: Dict[str, Any], step: StageId) -> None:
    ensure_state_defaults(state)
    ensure_session_artifacts(data_store_ref)
    for downstream_step in get_downstream_steps(step):
        state["step_results"].pop(downstream_step, None)
        data_store_ref["artifacts"]["step_outputs"].pop(downstream_step, None)
        data_store_ref["artifacts"]["datasets"].pop(downstream_step, None)


def append_message(state: AgentState, content: str) -> None:
    ensure_state_defaults(state)
    state["messages"].append({"role": "assistant", "content": content, "created_at": timestamp()})
