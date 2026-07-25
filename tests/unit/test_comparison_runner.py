from local_slm.benchmarking.memory import ModelMemoryInfo
from local_slm.comparison.runner import run_comparison_for_model, run_full_comparison
from local_slm.schemas import GenerationResult, PromptRecord, TaskType

FACTUAL = PromptRecord(
    id="fact_01",
    task_type=TaskType.FACTUAL_QA,
    prompt="capital of France?",
    expected_answer="Paris",
)
SUMM = PromptRecord(
    id="summ_01",
    task_type=TaskType.SUMMARIZATION,
    prompt="summarize",
    expected_keywords=["photosynthesis", "oxygen"],
)
JSON_P = PromptRecord(
    id="json_contact_01",
    task_type=TaskType.JSON_EXTRACTION,
    prompt="extract",
    schema_name="ContactInfo",
    expected_fields={"name": "Sarah Chen", "email": "sarah@example.com", "phone": "555-1234"},
)

RESPONSES = {
    "fact_01": "Paris",
    "summ_01": "This is about photosynthesis and oxygen production in plants today now.",
    "json_contact_01": '{"name": "Sarah Chen", "email": "sarah@example.com", "phone": "555-1234"}',
}


class FakeClient:
    def __init__(self):
        self.calls = 0

    def generate(self, model, prompt, temperature=0.0, format_schema=None, prompt_id=""):
        self.calls += 1
        return GenerationResult(
            model=model,
            prompt_id=prompt_id,
            response_text=RESPONSES[prompt_id],
            temperature=temperature,
            ttft_seconds=0.1,
            total_seconds=0.5,
            prompt_tokens=5,
            completion_tokens=10,
        )


def fake_memory_fn(model: str) -> ModelMemoryInfo | None:
    return ModelMemoryInfo(model=model, size_str="2.0 GB", processor="100% GPU")


def test_run_comparison_for_model_scores_all_task_types():
    client = FakeClient()
    result = run_comparison_for_model(
        client, "llama3.2:3b", [FACTUAL, SUMM, JSON_P], n_trials=2, memory_fn=fake_memory_fn
    )
    assert result.factual_accuracy == 1.0
    assert result.summarization_mean_score > 0.0
    assert result.json_schema_valid_rate == 1.0
    assert result.json_field_accuracy == 1.0
    assert result.memory.size_str == "2.0 GB"
    assert len(result.per_prompt_quality) == 3


def test_run_comparison_calls_client_n_trials_times_per_prompt():
    client = FakeClient()
    run_comparison_for_model(client, "m", [FACTUAL, SUMM], n_trials=3, memory_fn=fake_memory_fn)
    assert client.calls == 6


def test_run_full_comparison_runs_each_model_sequentially_never_interleaved():
    client = FakeClient()
    call_order = []
    original_generate = client.generate

    def tracking_generate(model, *args, **kwargs):
        call_order.append(model)
        return original_generate(model, *args, **kwargs)

    client.generate = tracking_generate
    results = run_full_comparison(
        client, ["model-a", "model-b"], [FACTUAL], n_trials=2, memory_fn=fake_memory_fn
    )
    assert [r.model for r in results] == ["model-a", "model-b"]
    # all of model-a's calls happen before any of model-b's
    first_b_index = call_order.index("model-b")
    assert all(m == "model-a" for m in call_order[:first_b_index])


def test_run_full_comparison_rejects_empty_models():
    import pytest

    with pytest.raises(ValueError):
        run_full_comparison(FakeClient(), [], [FACTUAL], memory_fn=fake_memory_fn)


def test_memory_none_when_lookup_fails():
    client = FakeClient()
    result = run_comparison_for_model(
        client, "m", [FACTUAL], n_trials=1, memory_fn=lambda model: None
    )
    assert result.memory is None
