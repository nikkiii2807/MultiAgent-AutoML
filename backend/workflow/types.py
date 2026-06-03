from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from .registry import StageId


class StepResult(TypedDict, total=False):
    analysis: str
    data_preview: Optional[str]
    metrics: Dict[str, Any]
    charts: List[Dict[str, Any]]
    notebook_label: str
    updated_at: str


class AgentState(TypedDict, total=False):
    messages: List[Dict[str, str]]
    current_step: StageId
    status: str
    dataset_info: Dict[str, Any]
    user_action: Optional[str]
    user_message: Optional[str]
    metrics: Dict[str, Any]
    step_results: Dict[StageId, StepResult]
