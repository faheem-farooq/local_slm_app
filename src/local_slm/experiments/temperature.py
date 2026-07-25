"""Temperature 0 vs 0.7 experiment: how much does output vary, and how much more
often does JSON-schema validation fail, once sampling is no longer greedy?

For each prompt we run n_repeats generations at each temperature. Diversity is
measured as (# distinct response strings) / n_repeats at that temperature, and
agreement is measured against a single temp-0 reference response (temp 0 is
expected to be near-deterministic, so its first repeat serves as "the answer").
For json_extraction prompts we additionally measure the Pydantic validation
failure rate at each temperature -- this is the concrete number that answers
"does raising temperature break structured output more often?".
"""

from __future__ import annotations

import statistics
from typing import Protocol

from pydantic import BaseModel

from local_slm.config import TEMP_HIGH, TEMP_LOW, TEMPERATURE_EXPERIMENT_REPEATS
from local_slm.schemas import GenerationResult, PromptRecord, TaskType
from local_slm.structured.schemas_catalog import resolve_schema_cls
from local_slm.structured.validator import validate


class GeneratesText(Protocol):
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        format_schema: dict | None = None,
        prompt_id: str = "",
    ) -> GenerationResult: ...


class TemperaturePromptResult(BaseModel):
    prompt_id: str
    task_type: TaskType
    temp0_responses: list[str]
    temp07_responses: list[str]
    temp0_unique_fraction: float
    temp07_unique_fraction: float
    agreement_rate_vs_temp0_reference: float
    temp0_validation_failures: int
    temp07_validation_failures: int
    is_json_task: bool


class TemperatureExperimentSummary(BaseModel):
    model: str
    n_repeats: int
    per_prompt: list[TemperaturePromptResult]
    mean_temp0_unique_fraction: float
    mean_temp07_unique_fraction: float
    mean_agreement_rate: float
    temp0_json_validation_failure_rate: float
    temp07_json_validation_failure_rate: float


def _run_prompt(
    client: GeneratesText, model: str, prompt: PromptRecord, n_repeats: int
) -> TemperaturePromptResult:
    schema_cls = resolve_schema_cls(prompt)
    format_schema = schema_cls.model_json_schema() if schema_cls else None

    temp0_responses = [
        client.generate(
            model=model, prompt=prompt.prompt, temperature=TEMP_LOW, format_schema=format_schema
        ).response_text
        for _ in range(n_repeats)
    ]
    temp07_responses = [
        client.generate(
            model=model, prompt=prompt.prompt, temperature=TEMP_HIGH, format_schema=format_schema
        ).response_text
        for _ in range(n_repeats)
    ]

    if schema_cls:
        temp0_failures = sum(1 for r in temp0_responses if not validate(r, schema_cls).valid)
        temp07_failures = sum(1 for r in temp07_responses if not validate(r, schema_cls).valid)
    else:
        temp0_failures = 0
        temp07_failures = 0

    reference = temp0_responses[0]
    agreement_rate = sum(1 for r in temp07_responses if r == reference) / n_repeats

    return TemperaturePromptResult(
        prompt_id=prompt.id,
        task_type=prompt.task_type,
        temp0_responses=temp0_responses,
        temp07_responses=temp07_responses,
        temp0_unique_fraction=len(set(temp0_responses)) / n_repeats,
        temp07_unique_fraction=len(set(temp07_responses)) / n_repeats,
        agreement_rate_vs_temp0_reference=agreement_rate,
        temp0_validation_failures=temp0_failures,
        temp07_validation_failures=temp07_failures,
        is_json_task=schema_cls is not None,
    )


def run_temperature_experiment(
    client: GeneratesText,
    model: str,
    prompts: list[PromptRecord],
    n_repeats: int = TEMPERATURE_EXPERIMENT_REPEATS,
) -> TemperatureExperimentSummary:
    if not prompts:
        raise ValueError("prompts must be non-empty")

    per_prompt = [_run_prompt(client, model, p, n_repeats) for p in prompts]
    json_results = [r for r in per_prompt if r.is_json_task]

    def failure_rate(attr: str) -> float:
        if not json_results:
            return 0.0
        total_failures = sum(getattr(r, attr) for r in json_results)
        return total_failures / (len(json_results) * n_repeats)

    return TemperatureExperimentSummary(
        model=model,
        n_repeats=n_repeats,
        per_prompt=per_prompt,
        mean_temp0_unique_fraction=statistics.mean(r.temp0_unique_fraction for r in per_prompt),
        mean_temp07_unique_fraction=statistics.mean(r.temp07_unique_fraction for r in per_prompt),
        mean_agreement_rate=statistics.mean(
            r.agreement_rate_vs_temp0_reference for r in per_prompt
        ),
        temp0_json_validation_failure_rate=failure_rate("temp0_validation_failures"),
        temp07_json_validation_failure_rate=failure_rate("temp07_validation_failures"),
    )
