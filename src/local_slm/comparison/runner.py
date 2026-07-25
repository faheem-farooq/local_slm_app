"""Phase 3: benchmark + score all three models on the same standardized prompt set.

Models are always run one at a time, start to finish, never interleaved -- this
matches Ollama's own swap-on-demand behavior and keeps `ollama ps` memory
readings and cold-load timings attributable to a single model.

Quality scoring reuses the first benchmarking trial's response per prompt
(generated at temperature=0.0) rather than issuing a separate generation pass --
the benchmark and quality-scoring passes ask the exact same question, so
duplicating the call would just double runtime for no new information.
"""

from __future__ import annotations

import statistics
from typing import Protocol

from pydantic import BaseModel

from local_slm.benchmarking.harness import GeneratesText, run_benchmark
from local_slm.benchmarking.memory import ModelMemoryInfo, get_loaded_model_memory
from local_slm.config import BENCHMARK_TRIALS
from local_slm.schemas import BenchmarkSummary, GenerationResult, PromptRecord, TaskType
from local_slm.scoring.factual import score_factual
from local_slm.scoring.json_quality import score_json_extraction
from local_slm.scoring.summarization import score_summarization
from local_slm.structured.schemas_catalog import resolve_schema_cls


class MemoryFn(Protocol):
    def __call__(self, model: str) -> ModelMemoryInfo | None: ...


class PromptQualityResult(BaseModel):
    prompt_id: str
    task_type: TaskType
    correct: bool | None = None
    summarization_score: float | None = None
    json_schema_valid: bool | None = None
    json_field_accuracy: float | None = None


class ModelComparisonResult(BaseModel):
    model: str
    benchmark: BenchmarkSummary
    memory: ModelMemoryInfo | None
    factual_accuracy: float
    summarization_mean_score: float
    json_schema_valid_rate: float
    json_field_accuracy: float
    per_prompt_quality: list[PromptQualityResult]


def _first_response_per_prompt(results: list[GenerationResult]) -> dict[str, str]:
    seen: dict[str, str] = {}
    for r in results:
        seen.setdefault(r.prompt_id, r.response_text)
    return seen


def _score_prompt(prompt: PromptRecord, response_text: str) -> PromptQualityResult:
    if prompt.task_type == TaskType.FACTUAL_QA:
        correct = score_factual(response_text, prompt.expected_answer or "")
        return PromptQualityResult(prompt_id=prompt.id, task_type=prompt.task_type, correct=correct)

    if prompt.task_type == TaskType.SUMMARIZATION:
        s = score_summarization(response_text, prompt.expected_keywords or [])
        return PromptQualityResult(
            prompt_id=prompt.id, task_type=prompt.task_type, summarization_score=s.overall_score
        )

    schema_cls = resolve_schema_cls(prompt)
    s = score_json_extraction(response_text, prompt.expected_fields or {}, schema_cls)
    return PromptQualityResult(
        prompt_id=prompt.id,
        task_type=prompt.task_type,
        json_schema_valid=s.schema_valid,
        json_field_accuracy=s.field_accuracy,
    )


def run_comparison_for_model(
    client: GeneratesText,
    model: str,
    prompts: list[PromptRecord],
    n_trials: int = BENCHMARK_TRIALS,
    memory_fn: MemoryFn = get_loaded_model_memory,
) -> ModelComparisonResult:
    benchmark_summary, raw_results = run_benchmark(client, model, prompts, n_trials=n_trials)
    responses = _first_response_per_prompt(raw_results)
    memory = memory_fn(model)

    per_prompt = [_score_prompt(p, responses.get(p.id, "")) for p in prompts]

    factual = [r.correct for r in per_prompt if r.correct is not None]
    summarization = [r.summarization_score for r in per_prompt if r.summarization_score is not None]
    json_valid = [r.json_schema_valid for r in per_prompt if r.json_schema_valid is not None]
    json_acc = [r.json_field_accuracy for r in per_prompt if r.json_field_accuracy is not None]

    return ModelComparisonResult(
        model=model,
        benchmark=benchmark_summary,
        memory=memory,
        factual_accuracy=statistics.mean(factual) if factual else 0.0,
        summarization_mean_score=statistics.mean(summarization) if summarization else 0.0,
        json_schema_valid_rate=statistics.mean(json_valid) if json_valid else 0.0,
        json_field_accuracy=statistics.mean(json_acc) if json_acc else 0.0,
        per_prompt_quality=per_prompt,
    )


def run_full_comparison(
    client: GeneratesText,
    models: list[str],
    prompts: list[PromptRecord],
    n_trials: int = BENCHMARK_TRIALS,
    memory_fn: MemoryFn = get_loaded_model_memory,
) -> list[ModelComparisonResult]:
    if not models:
        raise ValueError("models must be non-empty")
    return [
        run_comparison_for_model(client, model, prompts, n_trials=n_trials, memory_fn=memory_fn)
        for model in models
    ]
