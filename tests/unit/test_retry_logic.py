from local_slm.schemas import GenerationResult
from local_slm.structured.retry import ExtractionSuccess, StructuredFailure, extract_with_retry
from local_slm.structured.schemas_catalog import ContactInfo


class FakeClient:
    """Returns canned raw_text responses in sequence, one per generate() call."""

    def __init__(self, raw_texts: list[str]):
        self._raw_texts = raw_texts
        self.calls: list[dict] = []

    def generate(self, model, prompt, temperature=0.0, format_schema=None, prompt_id=""):
        idx = len(self.calls)
        self.calls.append({"model": model, "prompt": prompt, "format_schema": format_schema})
        return GenerationResult(
            model=model,
            prompt_id=prompt_id,
            response_text=self._raw_texts[idx],
            temperature=temperature,
            ttft_seconds=0.1,
            total_seconds=0.5,
            prompt_tokens=10,
            completion_tokens=10,
        )


VALID_CONTACT = '{"name": "Sarah Chen", "email": "sarah@example.com", "phone": "555-1234"}'
INVALID_CONTACT = "not json"


def test_success_on_first_attempt_makes_exactly_one_call():
    client = FakeClient([VALID_CONTACT])
    result = extract_with_retry(client, "m", "extract contact", ContactInfo)
    assert isinstance(result, ExtractionSuccess)
    assert result.attempts == 1
    assert len(client.calls) == 1
    assert client.calls[0]["format_schema"] == ContactInfo.model_json_schema()


def test_success_on_second_attempt_after_first_failure():
    client = FakeClient([INVALID_CONTACT, VALID_CONTACT])
    result = extract_with_retry(client, "m", "extract contact", ContactInfo)
    assert isinstance(result, ExtractionSuccess)
    assert result.attempts == 2
    assert len(client.calls) == 2
    # The reprompt must embed the original prompt, the invalid output, and the schema.
    assert "extract contact" in client.calls[1]["prompt"]
    assert INVALID_CONTACT in client.calls[1]["prompt"]


def test_both_attempts_failing_returns_structured_failure_not_exception():
    client = FakeClient([INVALID_CONTACT, INVALID_CONTACT])
    result = extract_with_retry(client, "m", "extract contact", ContactInfo)
    assert isinstance(result, StructuredFailure)
    assert len(client.calls) == 2  # never loops beyond 2 attempts
    assert result.first_error
    assert result.second_error
    assert result.schema_name == "ContactInfo"


def test_never_makes_a_third_call_even_on_double_failure():
    client = FakeClient([INVALID_CONTACT, INVALID_CONTACT])
    extract_with_retry(client, "m", "extract contact", ContactInfo)
    assert len(client.calls) == 2
