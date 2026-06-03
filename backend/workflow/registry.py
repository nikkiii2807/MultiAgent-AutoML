from __future__ import annotations

from typing import Dict, List, Literal, TypedDict

StageId = Literal[
    "ingestion",
    "preprocessing",
    "eda",
    "standardization",
    "modeling",
    "evaluation",
]


class StageDefinition(TypedDict):
    id: StageId
    notebook_label: str
    depends_on: List[StageId]


PIPELINE_ORDER: List[StageId] = [
    "ingestion",
    "preprocessing",
    "eda",
    "standardization",
    "modeling",
    "evaluation",
]

STAGE_DEFINITIONS: Dict[StageId, StageDefinition] = {
    "ingestion": {
        "id": "ingestion",
        "notebook_label": "01_data_ingestion.ipynb",
        "depends_on": [],
    },
    "preprocessing": {
        "id": "preprocessing",
        "notebook_label": "02_preprocessing.ipynb",
        "depends_on": ["ingestion"],
    },
    "eda": {
        "id": "eda",
        "notebook_label": "03_exploration.ipynb",
        "depends_on": ["preprocessing"],
    },
    "standardization": {
        "id": "standardization",
        "notebook_label": "04_feature_engineering.ipynb",
        "depends_on": ["eda"],
    },
    "modeling": {
        "id": "modeling",
        "notebook_label": "05_modeling.ipynb",
        "depends_on": ["standardization"],
    },
    "evaluation": {
        "id": "evaluation",
        "notebook_label": "06_evaluation.ipynb",
        "depends_on": ["modeling"],
    },
}


def get_downstream_steps(stage_id: StageId) -> List[StageId]:
    stage_index = PIPELINE_ORDER.index(stage_id)
    return PIPELINE_ORDER[stage_index + 1 :]
