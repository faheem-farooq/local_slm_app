import pytest

from local_slm.benchmarking.harness import collect_results, run_benchmark, summarize
from local_slm.schemas import GenerationResult, PromptRecord, TaskType


class FakeClient:
    """Returns canned, deterministic GenerationResults so aggregation math is exact."""

    def __init__(self, ttfts, tps_completion_tokens, total_seconds):
        self._ttfts = ttfts
        self._completion_tokens = tps_completion_tokens
        self._total_seconds = total_seconds
        self.calls = 0
        self.format_schemas_seen = []

    def generate(self, model, prompt, temperature=0.0, format_schema=None, prompt_id=""):
        i = self.calls
        self.calls += 1
        self.format_schemas_seen.append(format_schema)
        return GenerationResult(
            model=model,
            prompt_id=prompt_id,
            response_text="x",
            temperature=temperature,
            ttft_seconds=self._ttfts[i],
            total_seconds=self._total_seconds[i],
            prompt_tokens=10,
            completion_tokens=self._completion_tokens[i],
            load_duration_seconds=0.5 if i == 0 else 0.0,
        )


PROMPTS = [
    PromptRecord(id="p1", task_type=TaskType.FACTUAL_QA, prompt="q1", expected_answer="a"),
    PromptRecord(id="p2", task_type=TaskType.FACTUAL_QA, prompt="q2", expected_answer="a"),
]

JSON_PROMPT = PromptRecord(
    id="json_1",
    task_type=TaskType.JSON_EXTRACTION,
    prompt="extract",
    schema_name="ContactInfo",
    expected_fields={"name": "A", "email": "a@b.com", "phone": "1"},
)


def test_collect_results_calls_client_n_trials_times_per_prompt():
    client = FakeClient(ttfts=[0.1] * 4, tps_completion_tokens=[10] * 4, total_seconds=[1.0] * 4)
    results = collect_results(client, "m", PROMPTS, n_trials=2)
    assert client.calls == 4
    assert len(results) == 4
    assert {r.prompt_id for r in results} == {"p1", "p2"}


def test_summarize_computes_mean_and_stdev():
    results = [
        GenerationResult(
            model="m",
            prompt_id="p",
            response_text="x",
            temperature=0.0,
            ttft_seconds=v,
            total_seconds=1.0,
            prompt_tokens=1,
            completion_tokens=10,
            load_duration_seconds=0.0,
        )
        for v in [0.1, 0.2, 0.3]
    ]
    summary = summarize("m", results)
    assert summary.ttft_seconds.mean == pytest.approx(0.2)
    assert summary.ttft_seconds.n == 3
    assert summary.ttft_seconds.stdev > 0


def test_cold_load_duration_taken_from_first_result_only():
    client = FakeClient(ttfts=[0.1] * 4, tps_completion_tokens=[10] * 4, total_seconds=[1.0] * 4)
    summary, results = run_benchmark(client, "m", PROMPTS, n_trials=2)
    assert summary.cold_load_duration_seconds == 0.5
    assert results[1].load_duration_seconds == 0.0


def test_summarize_rejects_empty_results():
    with pytest.raises(ValueError):
        summarize("m", [])


def test_collect_results_rejects_empty_prompts():
    client = FakeClient(ttfts=[], tps_completion_tokens=[], total_seconds=[])
    with pytest.raises(ValueError):
        collect_results(client, "m", [], n_trials=2)


def test_collect_results_applies_format_schema_only_to_json_extraction_prompts():
    from local_slm.structured.schemas_catalog import ContactInfo

    client = FakeClient(ttfts=[0.1] * 4, tps_completion_tokens=[10] * 4, total_seconds=[1.0] * 4)
    collect_results(client, "m", [PROMPTS[0], JSON_PROMPT], n_trials=2)

    # PROMPTS[0] (factual) -> 2 calls with format_schema=None
    assert client.format_schemas_seen[0] is None
    assert client.format_schemas_seen[1] is None
    # JSON_PROMPT -> 2 calls with the ContactInfo JSON schema, matching what
    # extract_with_retry uses in production -- otherwise quality scoring that
    # reuses these responses would be scoring unconstrained generations.
    assert client.format_schemas_seen[2] == ContactInfo.model_json_schema()
    assert client.format_schemas_seen[3] == ContactInfo.model_json_schema()
