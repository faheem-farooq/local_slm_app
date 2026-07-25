#!/usr/bin/env python3
"""Manual sanity check against a real, running Ollama instance. Not collected by
pytest (outside tests/) -- CI never depends on Ollama or a pulled model."""

from local_slm.config import DEFAULT_MODEL
from local_slm.ollama_client import OllamaClient

if __name__ == "__main__":
    with OllamaClient() as client:
        result = client.generate(model=DEFAULT_MODEL, prompt="Say hello in one word.")
        print(f"model={result.model}")
        print(f"response={result.response_text!r}")
        print(f"ttft={result.ttft_seconds:.3f}s tokens/sec={result.tokens_per_second:.1f}")
