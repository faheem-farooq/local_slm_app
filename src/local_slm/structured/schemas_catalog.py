"""Pydantic models the model's JSON output is validated against (Phase 2/3 extraction tasks).

Each PromptRecord with task_type=json_extraction names one of these via `schema_name`,
which prompts.py resolves through SCHEMA_REGISTRY back to the class itself.
"""

from pydantic import BaseModel

from local_slm.schemas import PromptRecord, TaskType


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


class EventDetails(BaseModel):
    title: str
    date: str
    location: str


class ProductInfo(BaseModel):
    name: str
    price: float
    in_stock: bool


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "ContactInfo": ContactInfo,
    "EventDetails": EventDetails,
    "ProductInfo": ProductInfo,
}


def resolve_schema_cls(prompt: PromptRecord) -> type[BaseModel] | None:
    """The single place that maps a json_extraction prompt to its target schema.
    Used everywhere a prompt is sent to Ollama, so every code path that generates
    for a json_extraction prompt applies the same `format` JSON-schema constraint
    -- benchmarking, the temperature experiment, and quality scoring all reuse
    this instead of duplicating (and risking drifting) the same lookup."""
    if prompt.task_type == TaskType.JSON_EXTRACTION and prompt.schema_name:
        return SCHEMA_REGISTRY[prompt.schema_name]
    return None
