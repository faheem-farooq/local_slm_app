"""Core Pydantic models shared across the prompt set, benchmarking, and reporting."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    FACTUAL_QA = "factual_qa"
    SUMMARIZATION = "summarization"
    JSON_EXTRACTION = "json_extraction"


class PromptRecord(BaseModel):
    """One entry from data/prompts.yaml, the single standardized prompt set."""

    id: str
    task_type: TaskType
    prompt: str

    # factual_qa
    expected_answer: str | None = None

    # summarization
    expected_keywords: list[str] | None = None

    # json_extraction
    schema_name: str | None = None
    expected_fields: dict | None = None


class GenerationResult(BaseModel):
    """One completed (non-streaming-final) generation from OllamaClient.generate()."""

    model: str
    prompt_id: str
    response_text: str
    temperature: float
    ttft_seconds: float
    total_seconds: float
    prompt_tokens: int
    completion_tokens: int
    load_duration_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        eval_seconds = self.total_seconds - self.load_duration_seconds
        if eval_seconds <= 0 or self.completion_tokens == 0:
            return 0.0
        return self.completion_tokens / eval_seconds


class MetricSummary(BaseModel):
    """Mean + spread for one metric across N trials."""

    mean: float
    stdev: float
    min: float
    max: float
    n: int


class BenchmarkSummary(BaseModel):
    """Aggregated benchmark result for one model across all trials/prompts."""

    model: str
    n_trials: int
    ttft_seconds: MetricSummary
    tokens_per_second: MetricSummary
    total_seconds: MetricSummary
    cold_load_duration_seconds: float = Field(
        description="load_duration from the first trial only; excluded from steady-state means "
        "since it would otherwise skew tokens/sec on a one-time cold start."
    )
