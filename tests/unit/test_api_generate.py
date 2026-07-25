from fastapi.testclient import TestClient

from local_slm.api.app import app
from local_slm.api.deps import get_client


class FakeStreamingClient:
    def stream_generate(self, model, prompt, temperature=0.0, format_schema=None):
        yield "Hello"
        yield " world"


def test_health_check():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_generate_streams_concatenated_pieces():
    app.dependency_overrides[get_client] = lambda: FakeStreamingClient()
    try:
        client = TestClient(app)
        resp = client.post("/generate", json={"prompt": "hi", "model": "m", "temperature": 0.0})
        assert resp.status_code == 200
        assert resp.text == "Hello world"
    finally:
        app.dependency_overrides.clear()
