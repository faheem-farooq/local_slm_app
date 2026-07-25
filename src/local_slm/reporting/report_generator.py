"""Pure functions that turn result data into Markdown. No I/O, no model calls --
easy to unit test with small canned inputs (see test_report_generator.py)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from local_slm.comparison.runner import ModelComparisonResult
from local_slm.experiments.temperature import TemperatureExperimentSummary
from local_slm.schemas import BenchmarkSummary, MetricSummary


def format_metric(m: MetricSummary, unit: str = "s", precision: int = 3) -> str:
    return f"{m.mean:.{precision}f} ± {m.stdev:.{precision}f} {unit} (n={m.n})"


def benchmark_summary_table(summaries: list[BenchmarkSummary]) -> str:
    """One Markdown table row per model, comparing TTFT / tokens-per-sec / total latency."""
    lines = [
        "| Model | TTFT (s) | Tokens/sec | Total latency (s) | Cold load (s) | Trials |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.model} | {format_metric(s.ttft_seconds)} "
            f"| {format_metric(s.tokens_per_second, unit='tok/s')} "
            f"| {format_metric(s.total_seconds)} "
            f"| {s.cold_load_duration_seconds:.3f} | {s.n_trials} |"
        )
    return "\n".join(lines)


def temperature_experiment_report(summary: TemperatureExperimentSummary) -> str:
    lines = [
        f"# Temperature experiment: {summary.model} (n_repeats={summary.n_repeats})",
        "",
        "| Metric | temp=0.0 | temp=0.7 |",
        "|---|---|---|",
        f"| Mean response diversity (unique/repeats) | {summary.mean_temp0_unique_fraction:.2f} "
        f"| {summary.mean_temp07_unique_fraction:.2f} |",
        f"| JSON schema validation failure rate | {summary.temp0_json_validation_failure_rate:.2%} "
        f"| {summary.temp07_json_validation_failure_rate:.2%} |",
        "",
        f"Mean agreement rate of temp=0.7 responses vs. the temp=0 reference answer: "
        f"**{summary.mean_agreement_rate:.2%}**",
        "",
        "## Per-prompt detail",
        "",
        "| Prompt | Task | temp0 diversity | temp07 diversity | agreement vs temp0 |",
        "|---|---|---|---|---|",
    ]
    for p in summary.per_prompt:
        lines.append(
            f"| {p.prompt_id} | {p.task_type.value} | {p.temp0_unique_fraction:.2f} "
            f"| {p.temp07_unique_fraction:.2f} | {p.agreement_rate_vs_temp0_reference:.2%} |"
        )
    return "\n".join(lines)


def comparison_report(results: list[ModelComparisonResult]) -> str:
    """The Phase 3 headline table: speed, memory, and per-task-type quality side by side."""
    lines = [
        "# Model comparison",
        "",
        "| Model | Tokens/sec | TTFT (s) | Memory | Factual acc. | Summarization | "
        "JSON valid rate | JSON field acc. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        memory_str = r.memory.size_str if r.memory else "n/a"
        tps = format_metric(r.benchmark.tokens_per_second, unit="tok/s", precision=1)
        ttft = format_metric(r.benchmark.ttft_seconds, precision=2)
        lines.append(
            f"| {r.model} | {tps} | {ttft} | {memory_str} "
            f"| {r.factual_accuracy:.2%} | {r.summarization_mean_score:.2f} "
            f"| {r.json_schema_valid_rate:.2%} | {r.json_field_accuracy:.2%} |"
        )
    lines += [
        "",
        "Quality scoring is heuristic, not human- or LLM-judged: factual accuracy is "
        "exact/substring/fuzzy match against a short reference answer; summarization score is "
        "0.7 x keyword coverage + 0.3 x an ideal-length band; JSON metrics come from Pydantic "
        "schema validation plus per-field comparison against expected values. See README for "
        "details and caveats.",
    ]
    return "\n".join(lines)


def write_json(path: Path, data: BaseModel | list[BaseModel] | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        payload = data.model_dump(mode="json")
    elif isinstance(data, list):
        payload = [d.model_dump(mode="json") if isinstance(d, BaseModel) else d for d in data]
    else:
        payload = data
    path.write_text(json.dumps(payload, indent=2))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
