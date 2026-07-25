"""Typer CLI: benchmark / extract / temp-experiment / compare / report."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from local_slm.benchmarking.harness import run_benchmark
from local_slm.comparison.runner import run_comparison_for_model
from local_slm.config import (
    BENCHMARK_TRIALS,
    COMPARISON_MODELS,
    DEFAULT_MODEL,
    RESULTS_DIR,
    TEMPERATURE_EXPERIMENT_REPEATS,
)
from local_slm.experiments.temperature import run_temperature_experiment
from local_slm.ollama_client import OllamaClient
from local_slm.prompts import load_prompts
from local_slm.reporting.report_generator import (
    benchmark_summary_table,
    comparison_report,
    temperature_experiment_report,
    write_json,
    write_text,
)
from local_slm.schemas import TaskType
from local_slm.structured.retry import ExtractionSuccess, extract_with_retry
from local_slm.structured.schemas_catalog import SCHEMA_REGISTRY

app = typer.Typer(help="Local SLM App: benchmark, validate, and compare small models via Ollama.")
console = Console()


@app.command()
def benchmark(
    model: str = typer.Option(DEFAULT_MODEL, help="Ollama model tag to benchmark."),
    n_trials: int = typer.Option(BENCHMARK_TRIALS, help="Trials per prompt."),
    limit: int = typer.Option(0, help="Only use the first N prompts (0 = all 45)."),
    out_name: str = typer.Option("phase1_benchmarks", help="Base filename under results/."),
) -> None:
    """Run the Phase 1 inference benchmark (TTFT, tokens/sec, latency) for one model."""
    prompts = load_prompts()
    if limit:
        prompts = prompts[:limit]

    console.print(
        f"Benchmarking [bold]{model}[/bold] on {len(prompts)} prompts x {n_trials} trials..."
    )
    with OllamaClient() as client:
        summary, raw_results = run_benchmark(client, model, prompts, n_trials=n_trials)

    table = Table(title=f"Benchmark: {model}")
    table.add_column("Metric")
    table.add_column("Mean ± stdev")
    table.add_row("TTFT (s)", f"{summary.ttft_seconds.mean:.3f} ± {summary.ttft_seconds.stdev:.3f}")
    table.add_row(
        "Tokens/sec",
        f"{summary.tokens_per_second.mean:.2f} ± {summary.tokens_per_second.stdev:.2f}",
    )
    table.add_row(
        "Total latency (s)",
        f"{summary.total_seconds.mean:.3f} ± {summary.total_seconds.stdev:.3f}",
    )
    table.add_row("Cold load (s)", f"{summary.cold_load_duration_seconds:.3f}")
    console.print(table)

    model_slug = model.replace(":", "_").replace("/", "_")
    write_json(RESULTS_DIR / f"{out_name}_{model_slug}.json", summary)
    write_json(RESULTS_DIR / "raw" / f"{out_name}_{model_slug}_raw.json", raw_results)
    write_text(
        RESULTS_DIR / f"{out_name}_{model_slug}.md",
        f"# Phase 1 benchmark: {model}\n\n" + benchmark_summary_table([summary]) + "\n",
    )
    console.print(f"[green]Results written to {RESULTS_DIR}[/green]")


@app.command()
def stream(
    prompt: str = typer.Argument(..., help="Prompt to send."),
    model: str = typer.Option(DEFAULT_MODEL, help="Ollama model tag."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
) -> None:
    """Stream a single generation live, printing tokens/sec as they arrive (demo target)."""
    import time

    console.print(f"[bold]{model}[/bold] (temp={temperature})\n")
    start = time.monotonic()
    n_chars = 0
    text = ""
    with OllamaClient() as client, Live(console=console, refresh_per_second=8) as live:
        for piece in client.stream_generate(model=model, prompt=prompt, temperature=temperature):
            text += piece
            n_chars += len(piece)
            elapsed = max(time.monotonic() - start, 1e-6)
            live.update(f"{text}\n\n[dim]{n_chars / elapsed:.1f} chars/sec[/dim]")


@app.command()
def extract(
    model: str = typer.Option(DEFAULT_MODEL, help="Ollama model tag."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
) -> None:
    """Run the JSON-schema-validate-retry pipeline over the 15 json_extraction prompts."""
    prompts = [p for p in load_prompts() if p.task_type == TaskType.JSON_EXTRACTION]
    console.print(
        f"Running structured extraction on {len(prompts)} prompts with [bold]{model}[/bold]..."
    )

    table = Table(title=f"Structured extraction: {model}")
    table.add_column("Prompt")
    table.add_column("Schema")
    table.add_column("Outcome")

    successes_1st, successes_2nd, failures = 0, 0, 0
    with OllamaClient() as client:
        for p in prompts:
            schema_cls = SCHEMA_REGISTRY[p.schema_name]
            result = extract_with_retry(
                client, model, p.prompt, schema_cls, temperature=temperature
            )
            if isinstance(result, ExtractionSuccess):
                if result.attempts == 1:
                    successes_1st += 1
                    table.add_row(p.id, p.schema_name, "[green]valid (1st try)[/green]")
                else:
                    successes_2nd += 1
                    table.add_row(p.id, p.schema_name, "[yellow]valid (after retry)[/yellow]")
            else:
                failures += 1
                table.add_row(p.id, p.schema_name, "[red]failed (both attempts)[/red]")

    console.print(table)
    console.print(
        f"1st-try valid: {successes_1st}  |  valid after retry: {successes_2nd}  |  "
        f"failed both: {failures}"
    )


@app.command(name="temp-experiment")
def temp_experiment(
    model: str = typer.Option(DEFAULT_MODEL, help="Ollama model tag."),
    n_repeats: int = typer.Option(TEMPERATURE_EXPERIMENT_REPEATS, help="Repeats per temperature."),
    limit: int = typer.Option(0, help="Only use the first N prompts (0 = all 45)."),
) -> None:
    """Run the Phase 2 temperature 0 vs 0.7 determinism experiment."""
    prompts = load_prompts()
    if limit:
        prompts = prompts[:limit]

    console.print(
        f"Running temperature experiment on {len(prompts)} prompts x {n_repeats} repeats "
        f"with [bold]{model}[/bold] (this calls the model {len(prompts) * n_repeats * 2} times)..."
    )
    with OllamaClient() as client:
        summary = run_temperature_experiment(client, model, prompts, n_repeats=n_repeats)

    console.print(
        f"Mean diversity: temp=0 {summary.mean_temp0_unique_fraction:.2f}, "
        f"temp=0.7 {summary.mean_temp07_unique_fraction:.2f}"
    )
    console.print(f"Mean agreement (temp0.7 vs temp0 reference): {summary.mean_agreement_rate:.2%}")
    console.print(
        f"JSON validation failure rate: temp=0 {summary.temp0_json_validation_failure_rate:.2%}, "
        f"temp=0.7 {summary.temp07_json_validation_failure_rate:.2%}"
    )

    model_slug = model.replace(":", "_").replace("/", "_")
    write_json(RESULTS_DIR / f"phase2_temperature_experiment_{model_slug}.json", summary)
    write_text(
        RESULTS_DIR / f"phase2_temperature_experiment_{model_slug}.md",
        temperature_experiment_report(summary),
    )
    console.print(f"[green]Results written to {RESULTS_DIR}[/green]")


def _parse_trials_overrides(overrides: list[str]) -> dict[str, int]:
    """Parses repeated 'model=n' strings, e.g. 'mistral:7b=2', into a dict.
    Uses rpartition on '=' since model tags themselves contain ':'."""
    parsed: dict[str, int] = {}
    for item in overrides:
        model_name, sep, n = item.rpartition("=")
        if not sep:
            raise ValueError(f"Invalid --trials-override '{item}', expected 'model=n'")
        parsed[model_name] = int(n)
    return parsed


@app.command()
def compare(
    models: list[str] = typer.Option(
        COMPARISON_MODELS, help="Models to compare, run sequentially."
    ),
    n_trials: int = typer.Option(BENCHMARK_TRIALS, help="Default trials per prompt per model."),
    trials_override: list[str] = typer.Option(
        [],
        "--trials-override",
        help="Override trials for one model, format 'model=n' (repeatable). Useful for "
        "slower/larger models on constrained hardware, e.g. --trials-override mistral:7b=2",
    ),
    limit: int = typer.Option(0, help="Only use the first N prompts (0 = all 45)."),
) -> None:
    """Run the Phase 3 three-model comparison: speed, memory, and quality on the same set."""
    prompts = load_prompts()
    if limit:
        prompts = prompts[:limit]
    overrides = _parse_trials_overrides(trials_override)

    console.print(
        f"Comparing {models} on {len(prompts)} prompts (default {n_trials} trials, "
        f"overrides={overrides}), models run sequentially, never interleaved..."
    )
    with OllamaClient() as client:
        results = [
            run_comparison_for_model(
                client, model, prompts, n_trials=overrides.get(model, n_trials)
            )
            for model in models
        ]

    table = Table(title="Model comparison")
    table.add_column("Model")
    table.add_column("Tok/s")
    table.add_column("TTFT (s)")
    table.add_column("Memory")
    table.add_column("Factual")
    table.add_column("Summ.")
    table.add_column("JSON valid")
    table.add_column("JSON acc.")
    for r in results:
        memory_str = r.memory.size_str if r.memory else "n/a"
        table.add_row(
            r.model,
            f"{r.benchmark.tokens_per_second.mean:.1f}",
            f"{r.benchmark.ttft_seconds.mean:.2f}",
            memory_str,
            f"{r.factual_accuracy:.0%}",
            f"{r.summarization_mean_score:.2f}",
            f"{r.json_schema_valid_rate:.0%}",
            f"{r.json_field_accuracy:.0%}",
        )
    console.print(table)

    write_json(RESULTS_DIR / "phase3_model_comparison.json", results)
    write_text(RESULTS_DIR / "phase3_model_comparison.md", comparison_report(results))
    console.print(f"[green]Results written to {RESULTS_DIR}[/green]")


if __name__ == "__main__":
    app()
