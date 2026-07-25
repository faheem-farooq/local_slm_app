from local_slm.comparison.runner import ModelComparisonResult
from local_slm.experiments.temperature import TemperatureExperimentSummary, TemperaturePromptResult
from local_slm.reporting.report_generator import (
    benchmark_summary_table,
    comparison_report,
    format_metric,
    temperature_experiment_report,
)
from local_slm.schemas import BenchmarkSummary, MetricSummary, TaskType


def make_summary(model="llama3.2:3b"):
    m = MetricSummary(mean=1.234, stdev=0.056, min=1.0, max=1.5, n=25)
    return BenchmarkSummary(
        model=model,
        n_trials=25,
        ttft_seconds=m,
        tokens_per_second=m,
        total_seconds=m,
        cold_load_duration_seconds=0.5,
    )


def test_format_metric_includes_mean_stdev_and_n():
    m = MetricSummary(mean=1.0, stdev=0.1, min=0.9, max=1.1, n=10)
    formatted = format_metric(m, unit="tok/s")
    assert "1.000" in formatted
    assert "0.100" in formatted
    assert "tok/s" in formatted
    assert "n=10" in formatted


def test_benchmark_summary_table_has_header_and_one_row_per_model():
    table = benchmark_summary_table([make_summary("llama3.2:3b"), make_summary("mistral:7b")])
    assert "| Model | TTFT (s)" in table
    assert "llama3.2:3b" in table
    assert "mistral:7b" in table
    assert table.count("\n") == 3  # header, separator, 2 rows -> 3 newlines


def test_temperature_experiment_report_includes_key_numbers():
    per_prompt = [
        TemperaturePromptResult(
            prompt_id="fact_01",
            task_type=TaskType.FACTUAL_QA,
            temp0_responses=["Paris"] * 4,
            temp07_responses=["Paris"] * 4,
            temp0_unique_fraction=0.25,
            temp07_unique_fraction=0.5,
            agreement_rate_vs_temp0_reference=0.75,
            temp0_validation_failures=0,
            temp07_validation_failures=0,
            is_json_task=False,
        )
    ]
    summary = TemperatureExperimentSummary(
        model="llama3.2:3b",
        n_repeats=4,
        per_prompt=per_prompt,
        mean_temp0_unique_fraction=0.25,
        mean_temp07_unique_fraction=0.5,
        mean_agreement_rate=0.75,
        temp0_json_validation_failure_rate=0.0,
        temp07_json_validation_failure_rate=0.1,
    )
    report = temperature_experiment_report(summary)
    assert "llama3.2:3b" in report
    assert "75.00%" in report
    assert "fact_01" in report


def test_comparison_report_includes_all_models_and_a_caveat_note():
    results = [
        ModelComparisonResult(
            model="llama3.2:3b",
            benchmark=make_summary("llama3.2:3b"),
            memory=None,
            factual_accuracy=0.9,
            summarization_mean_score=0.7,
            json_schema_valid_rate=1.0,
            json_field_accuracy=0.95,
            per_prompt_quality=[],
        ),
        ModelComparisonResult(
            model="mistral:7b",
            benchmark=make_summary("mistral:7b"),
            memory=None,
            factual_accuracy=0.95,
            summarization_mean_score=0.8,
            json_schema_valid_rate=1.0,
            json_field_accuracy=0.97,
            per_prompt_quality=[],
        ),
    ]
    report = comparison_report(results)
    assert "llama3.2:3b" in report
    assert "mistral:7b" in report
    assert "90.00%" in report
    assert "heuristic" in report.lower()
