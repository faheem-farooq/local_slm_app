"""Runs N trials per prompt against a model and aggregates timing metrics.

Cold-start load_duration (from the very first generation) is reported separately
rather than folded into the tokens/sec mean, since a one-time model load would
otherwise skew the steady-state throughput number.
"""

from __future__ import annotations

import statistics
from typing import Protocol

from local_slm.config import BENCHMARK_TRIALS
from local_slm.schemas import BenchmarkSummary, GenerationResult, MetricSummary, PromptRecord
from local_slm.structured.schemas_catalog import resolve_schema_cls


class GeneratesText(Protocol):
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        format_schema: dict | None = None,
        prompt_id: str = "",
    ) -> GenerationResult: ...


def collect_results(
    client: GeneratesText,
    model: str,
    prompts: list[PromptRecord],
    n_trials: int = BENCHMARK_TRIALS,
    temperature: float = 0.0,
) -> list[GenerationResult]:
    """Benchmarks generate using the same `format` JSON-schema constraint the app
    applies to json_extraction prompts in production (see resolve_schema_cls) --
    otherwise a model could look artificially unreliable at structured output
    here just because this pass forgot to ask for constrained decoding, which
    would corrupt any quality scoring that reuses these responses (see
    comparison/runner.py)."""
    if not prompts:
        raise ValueError("prompts must be non-empty")
    results: list[GenerationResult] = []
    for prompt in prompts:
        schema_cls = resolve_schema_cls(prompt)
        format_schema = schema_cls.model_json_schema() if schema_cls else None
        for _ in range(n_trials):
            results.append(
                client.generate(
                    model=model,
                    prompt=prompt.prompt,
                    temperature=temperature,
                    format_schema=format_schema,
                    prompt_id=prompt.id,
                )
            )
    return results


def _summarize_metric(values: list[float]) -> MetricSummary:
    return MetricSummary(
        mean=statistics.mean(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        max=max(values),
        n=len(values),
    )


def summarize(model: str, results: list[GenerationResult]) -> BenchmarkSummary:
    if not results:
        raise ValueError("results must be non-empty")
    return BenchmarkSummary(
        model=model,
        n_trials=len(results),
        ttft_seconds=_summarize_metric([r.ttft_seconds for r in results]),
        tokens_per_second=_summarize_metric([r.tokens_per_second for r in results]),
        total_seconds=_summarize_metric([r.total_seconds for r in results]),
        cold_load_duration_seconds=results[0].load_duration_seconds,
    )


def run_benchmark(
    client: GeneratesText,
    model: str,
    prompts: list[PromptRecord],
    n_trials: int = BENCHMARK_TRIALS,
    temperature: float = 0.0,
) -> tuple[BenchmarkSummary, list[GenerationResult]]:
    results = collect_results(client, model, prompts, n_trials, temperature)
    return summarize(model, results), results
