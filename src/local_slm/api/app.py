"""FastAPI service layer. Exists to satisfy the "expose it as a service" framing;
it imports the same OllamaClient and extract_with_retry the CLI and benchmarking
harness use, so there's no forked logic between the CLI and API paths.

Benchmarking deliberately does NOT go through this layer -- see comparison/runner.py
and benchmarking/harness.py, which call OllamaClient directly so uvicorn/Starlette
overhead never contaminates the TTFT/tokens-per-sec numbers.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from local_slm.api.deps import get_client
from local_slm.config import DEFAULT_MODEL
from local_slm.ollama_client import OllamaClient
from local_slm.structured.retry import ExtractionSuccess, extract_with_retry
from local_slm.structured.schemas_catalog import SCHEMA_REGISTRY

app = FastAPI(
    title="Local SLM App", description="Ollama-backed generation and structured extraction."
)


class GenerateRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    temperature: float = 0.0


class ExtractRequest(BaseModel):
    prompt: str
    schema_name: str
    model: str = DEFAULT_MODEL
    temperature: float = 0.0


class ExtractResponse(BaseModel):
    success: bool
    attempts: int | None = None
    parsed: dict | None = None
    error: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest, client: OllamaClient = Depends(get_client)) -> StreamingResponse:
    return StreamingResponse(
        client.stream_generate(model=req.model, prompt=req.prompt, temperature=req.temperature),
        media_type="text/plain",
    )


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest, client: OllamaClient = Depends(get_client)) -> ExtractResponse:
    schema_cls = SCHEMA_REGISTRY.get(req.schema_name)
    if schema_cls is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown schema_name '{req.schema_name}'. Options: {list(SCHEMA_REGISTRY)}",
        )

    result = extract_with_retry(
        client, req.model, req.prompt, schema_cls, temperature=req.temperature
    )
    if isinstance(result, ExtractionSuccess):
        return ExtractResponse(success=True, attempts=result.attempts, parsed=result.parsed)
    return ExtractResponse(success=False, error=result.second_error)
