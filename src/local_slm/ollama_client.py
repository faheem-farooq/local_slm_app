"""Thin synchronous client over Ollama's /api/generate streaming endpoint.

Ollama streams newline-delimited JSON (NDJSON). Every chunk carries a `response`
text fragment; only the final chunk (`done: true`) carries the timing/count
fields (`prompt_eval_count`, `eval_count`, `eval_duration`, `load_duration`, all
in nanoseconds). Reading those fields off any chunk other than the final one is
a real bug (they'd be absent/zero) -- see tests/unit/test_ollama_client.py.

"Time to first token" is measured as time to first *non-empty* content chunk:
Ollama sometimes emits an initial chunk with an empty `response` string, and
counting that as the first token would understate real latency.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from time import monotonic
from typing import Any

import httpx

from local_slm.config import OLLAMA_BASE_URL
from local_slm.schemas import GenerationResult

NANOS_PER_SECOND = 1_000_000_000


class OllamaClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        time_fn: Callable[[], float] = monotonic,
    ) -> None:
        self.base_url = base_url
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._time_fn = time_fn

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _stream_raw(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self._client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                yield json.loads(line)

    def stream_generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        format_schema: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield raw text fragments as they stream in. Used by the API layer for
        live token-by-token display; not used by the benchmarking harness, which
        needs the full GenerationResult with timing."""
        payload = self._build_payload(model, prompt, temperature, format_schema)
        for chunk in self._stream_raw(payload):
            piece = chunk.get("response", "")
            if piece:
                yield piece

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        format_schema: dict[str, Any] | None = None,
        prompt_id: str = "",
    ) -> GenerationResult:
        payload = self._build_payload(model, prompt, temperature, format_schema)

        start_time = self._time_fn()
        first_content_time: float | None = None
        response_text = ""
        final_chunk: dict[str, Any] = {}

        for chunk in self._stream_raw(payload):
            piece = chunk.get("response", "")
            if piece:
                if first_content_time is None:
                    first_content_time = self._time_fn()
                response_text += piece
            if chunk.get("done"):
                final_chunk = chunk

        end_time = self._time_fn()
        if first_content_time is None:
            first_content_time = end_time

        load_duration_seconds = final_chunk.get("load_duration", 0) / NANOS_PER_SECOND

        return GenerationResult(
            model=model,
            prompt_id=prompt_id,
            response_text=response_text,
            temperature=temperature,
            ttft_seconds=first_content_time - start_time,
            total_seconds=end_time - start_time,
            prompt_tokens=final_chunk.get("prompt_eval_count", 0),
            completion_tokens=final_chunk.get("eval_count", 0),
            load_duration_seconds=load_duration_seconds,
        )

    @staticmethod
    def _build_payload(
        model: str,
        prompt: str,
        temperature: float,
        format_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if format_schema is not None:
            payload["format"] = format_schema
        return payload
