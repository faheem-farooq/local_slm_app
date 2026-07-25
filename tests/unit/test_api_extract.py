from fastapi.testclient import TestClient

from local_slm.api.app import app
from local_slm.api.deps import get_client
from local_slm.schemas import GenerationResult


class FakeExtractClient:
    def __init__(self, raw_texts: list[str]):
        self._raw_texts = raw_texts
        self.calls = 0

    def generate(self, model, prompt, temperature=0.0, format_schema=None, prompt_id=""):
        idx = self.calls
        self.calls += 1
        return GenerationResult(
            model=model,
            prompt_id=prompt_id,
            response_text=self._raw_texts[idx],
            temperature=temperature,
            ttft_seconds=0.1,
            total_seconds=0.5,
            prompt_tokens=5,
            completion_tokens=5,
        )


VALID = '{"name": "Sarah Chen", "email": "sarah@example.com", "phone": "555-1234"}'


def _override(client):
    app.dependency_overrides[get_client] = lambda: client


def test_extract_success_returns_parsed_json():
    _override(FakeExtractClient([VALID]))
    try:
        client = TestClient(app)
        resp = client.post(
            "/extract", json={"prompt": "extract", "schema_name": "ContactInfo", "model": "m"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["attempts"] == 1
        assert body["parsed"]["name"] == "Sarah Chen"
    finally:
        app.dependency_overrides.clear()


def test_extract_failure_after_retry_returns_success_false_not_500():
    _override(FakeExtractClient(["not json", "still not json"]))
    try:
        client = TestClient(app)
        resp = client.post(
            "/extract", json={"prompt": "extract", "schema_name": "ContactInfo", "model": "m"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["error"]
    finally:
        app.dependency_overrides.clear()


def test_unknown_schema_name_returns_400():
    _override(FakeExtractClient([VALID]))
    try:
        client = TestClient(app)
        resp = client.post(
            "/extract", json={"prompt": "extract", "schema_name": "NotASchema", "model": "m"}
        )
        assert resp.status_code == 400
    finally:
        app.dependency_overrides.clear()
