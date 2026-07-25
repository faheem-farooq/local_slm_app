import pytest

from local_slm.experiments.temperature import run_temperature_experiment
from local_slm.schemas import GenerationResult, PromptRecord, TaskType


class ScriptedClient:
    """Returns responses from a per-temperature script, indexed by call order."""

    def __init__(self, temp0_script: list[str], temp07_script: list[str]):
        self._temp0_script = temp0_script
        self._temp07_script = temp07_script
        self._temp0_i = 0
        self._temp07_i = 0

    def generate(self, model, prompt, temperature=0.0, format_schema=None, prompt_id=""):
        if temperature == 0.0:
            text = self._temp0_script[self._temp0_i]
            self._temp0_i += 1
        else:
            text = self._temp07_script[self._temp07_i]
            self._temp07_i += 1
        return GenerationResult(
            model=model,
            prompt_id=prompt_id,
            response_text=text,
            temperature=temperature,
            ttft_seconds=0.1,
            total_seconds=0.5,
            prompt_tokens=5,
            completion_tokens=5,
        )


FACTUAL_PROMPT = PromptRecord(
    id="fact_01",
    task_type=TaskType.FACTUAL_QA,
    prompt="capital of France?",
    expected_answer="Paris",
)

JSON_PROMPT = PromptRecord(
    id="json_contact_01",
    task_type=TaskType.JSON_EXTRACTION,
    prompt="extract contact",
    schema_name="ContactInfo",
    expected_fields={"name": "A", "email": "a@b.com", "phone": "1"},
)


def test_fully_deterministic_temp0_gives_zero_diversity_and_full_agreement():
    client = ScriptedClient(
        temp0_script=["Paris"] * 4,
        temp07_script=["Paris"] * 4,
    )
    summary = run_temperature_experiment(client, "m", [FACTUAL_PROMPT], n_repeats=4)
    p = summary.per_prompt[0]
    assert p.temp0_unique_fraction == pytest.approx(0.25)  # 1 unique / 4 repeats
    assert p.temp07_unique_fraction == pytest.approx(0.25)
    assert p.agreement_rate_vs_temp0_reference == 1.0


def test_temp07_diversity_and_disagreement_detected():
    client = ScriptedClient(
        temp0_script=["Paris"] * 4,
        temp07_script=["Paris", "Paris, France", "The capital is Paris", "Paris"],
    )
    summary = run_temperature_experiment(client, "m", [FACTUAL_PROMPT], n_repeats=4)
    p = summary.per_prompt[0]
    assert p.temp07_unique_fraction == pytest.approx(0.75)  # 3 unique / 4
    assert p.agreement_rate_vs_temp0_reference == pytest.approx(0.5)  # 2 of 4 match "Paris"


def test_json_validation_failure_rate_measured_only_for_json_tasks():
    valid = '{"name": "A", "email": "a@b.com", "phone": "1"}'
    invalid = "not json"
    client = ScriptedClient(
        temp0_script=[valid, valid, valid, valid],
        temp07_script=[valid, invalid, valid, invalid],
    )
    summary = run_temperature_experiment(client, "m", [JSON_PROMPT], n_repeats=4)
    assert summary.temp0_json_validation_failure_rate == 0.0
    assert summary.temp07_json_validation_failure_rate == pytest.approx(0.5)


def test_non_json_prompts_contribute_zero_to_json_failure_rate():
    client = ScriptedClient(temp0_script=["Paris"] * 4, temp07_script=["Paris"] * 4)
    summary = run_temperature_experiment(client, "m", [FACTUAL_PROMPT], n_repeats=4)
    assert summary.temp0_json_validation_failure_rate == 0.0
    assert summary.temp07_json_validation_failure_rate == 0.0


def test_rejects_empty_prompts():
    with pytest.raises(ValueError):
        run_temperature_experiment(ScriptedClient([], []), "m", [], n_repeats=4)
