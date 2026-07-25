"""JSON-schema-constrained generation with exactly one reprompt on validation failure.

The retry is structural (two literal call sites), not a max_retries counter that
could accidentally be set wrong: attempt once, validate, and only on failure build
a feedback reprompt (original prompt + the invalid output + the Pydantic error) and
try a second time. If the second attempt also fails validation, this returns a
StructuredFailure -- it never raises, and it never loops beyond two attempts.
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel

from local_slm.schemas import GenerationResult
from local_slm.structured.validator import validate

logger = logging.getLogger(__name__)


class GeneratesText(Protocol):
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        format_schema: dict | None = None,
        prompt_id: str = "",
    ) -> GenerationResult: ...


class ExtractionSuccess(BaseModel):
    parsed: dict
    attempts: int
    raw_text: str


class StructuredFailure(BaseModel):
    prompt: str
    schema_name: str
    first_raw: str
    first_error: str
    second_raw: str
    second_error: str


def _build_reprompt(
    original_prompt: str, invalid_raw: str, error: str, schema_cls: type[BaseModel]
) -> str:
    return (
        f"{original_prompt}\n\n"
        f"Your previous response was not valid JSON matching the required schema.\n"
        f"Previous response: {invalid_raw}\n"
        f"Validation error: {error}\n"
        f"Required JSON schema: {schema_cls.model_json_schema()}\n"
        f"Reply with ONLY corrected JSON matching the schema, no extra text."
    )


def extract_with_retry(
    client: GeneratesText,
    model: str,
    prompt: str,
    schema_cls: type[BaseModel],
    temperature: float = 0.0,
) -> ExtractionSuccess | StructuredFailure:
    schema_json = schema_cls.model_json_schema()

    first = client.generate(
        model=model, prompt=prompt, temperature=temperature, format_schema=schema_json
    )
    first_outcome = validate(first.response_text, schema_cls)
    if first_outcome.valid:
        return ExtractionSuccess(
            parsed=first_outcome.parsed, attempts=1, raw_text=first.response_text
        )

    reprompt = _build_reprompt(prompt, first.response_text, first_outcome.error or "", schema_cls)
    second = client.generate(
        model=model, prompt=reprompt, temperature=temperature, format_schema=schema_json
    )
    second_outcome = validate(second.response_text, schema_cls)
    if second_outcome.valid:
        return ExtractionSuccess(
            parsed=second_outcome.parsed, attempts=2, raw_text=second.response_text
        )

    logger.warning(
        "Structured extraction failed after retry: model=%s schema=%s "
        "first_error=%s second_error=%s",
        model,
        schema_cls.__name__,
        first_outcome.error,
        second_outcome.error,
    )
    return StructuredFailure(
        prompt=prompt,
        schema_name=schema_cls.__name__,
        first_raw=first.response_text,
        first_error=first_outcome.error or "",
        second_raw=second.response_text,
        second_error=second_outcome.error or "",
    )
