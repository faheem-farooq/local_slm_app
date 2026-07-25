"""Loads and validates the standardized prompt set from data/prompts.yaml."""

from pathlib import Path

import yaml

from local_slm.config import PROMPTS_PATH
from local_slm.schemas import PromptRecord, TaskType


def load_prompts(path: Path = PROMPTS_PATH) -> list[PromptRecord]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a YAML list of prompt records")
    return [PromptRecord.model_validate(entry) for entry in raw]


def filter_by_task(prompts: list[PromptRecord], task_type: TaskType) -> list[PromptRecord]:
    return [p for p in prompts if p.task_type == task_type]
