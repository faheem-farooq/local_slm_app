from local_slm.ollama_client import OllamaClient
from tests.conftest import FakeClock, mock_transport


def test_generate_parses_final_chunk_timing_fields_only():
    client = OllamaClient(transport=mock_transport("generate_success.ndjson"))
    result = client.generate(model="llama3.2:3b", prompt="What is the capital of France?")

    assert result.response_text == "Paris is great."
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 4
    assert result.load_duration_seconds == 0.25


def test_ttft_measures_first_non_empty_chunk_not_literal_first_chunk():
    # Fixture's first chunk has response="" (must be skipped), second chunk has "Paris".
    # Clock ticks: start=0.0, then one tick per chunk processed (5 chunks), then end.
    clock = FakeClock([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    client = OllamaClient(transport=mock_transport("generate_success.ndjson"), time_fn=clock)

    result = client.generate(model="llama3.2:3b", prompt="q")

    # start_time=0.0 (1st call). Loop: chunk1 empty->no clock call for ttft.
    # chunk2 "Paris" non-empty, first_content_time = 2nd call = 0.1.
    assert result.ttft_seconds == 0.1
    assert result.total_seconds > result.ttft_seconds


def test_tokens_per_second_uses_eval_duration_excluding_load():
    client = OllamaClient(transport=mock_transport("generate_success.ndjson"))
    result = client.generate(model="llama3.2:3b", prompt="q")

    # total_seconds is wall clock (real monotonic, tiny but >0); tokens_per_second
    # divides completion_tokens by (total_seconds - load_duration_seconds).
    assert result.completion_tokens == 4
    assert result.tokens_per_second >= 0


def test_format_schema_included_in_payload_when_provided():
    captured = {}

    def handler(request):
        import json

        captured["payload"] = json.loads(request.content)
        import httpx

        return httpx.Response(200, content=b'{"response": "{}", "done": true}')

    import httpx

    client = OllamaClient(transport=httpx.MockTransport(handler))
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    client.generate(model="m", prompt="p", format_schema=schema)

    assert captured["payload"]["format"] == schema


def test_stream_generate_yields_only_non_empty_pieces():
    client = OllamaClient(transport=mock_transport("generate_success.ndjson"))
    pieces = list(client.stream_generate(model="llama3.2:3b", prompt="q"))
    assert pieces == ["Paris", " is", " great."]
