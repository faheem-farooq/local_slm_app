"""FastAPI dependency wiring. Tests override get_client to avoid any real network call."""

from local_slm.ollama_client import OllamaClient

_client: OllamaClient | None = None


def get_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
