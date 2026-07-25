from pathlib import Path

import httpx
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ollama_responses"


def load_fixture_bytes(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()


def mock_transport(*fixture_names: str) -> httpx.MockTransport:
    """Build a transport that returns fixture_names[i]'s content on the i-th call.
    If more calls happen than fixtures given, the last fixture repeats."""
    bodies = [load_fixture_bytes(name) for name in fixture_names]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        idx = min(calls["n"], len(bodies) - 1)
        calls["n"] += 1
        return httpx.Response(200, content=bodies[idx])

    return httpx.MockTransport(handler)


class FakeClock:
    """Deterministic clock: each call returns the next value, then holds the last."""

    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._i = 0

    def __call__(self) -> float:
        v = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return v


@pytest.fixture
def fixture_bytes():
    return load_fixture_bytes
