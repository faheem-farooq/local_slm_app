"""Validates a model's raw text output against a target Pydantic schema.

Ollama's `format` JSON-schema parameter (grammar-constrained decoding) makes the
raw output syntactically valid JSON in the common case, but it does not guarantee
semantic correctness -- a quantized model can still put a hallucinated string into
a field or mistype a number. Pydantic validation is kept as an independent second
gate rather than treated as redundant with `format`.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError


class ValidationOutcome(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    valid: bool
    parsed: dict | None = None
    error: str | None = None


def validate(raw_text: str, schema_cls: type[BaseModel]) -> ValidationOutcome:
    try:
        instance = schema_cls.model_validate_json(raw_text)
    except ValidationError as e:
        return ValidationOutcome(valid=False, error=str(e))
    except ValueError as e:  # malformed JSON (not even parseable)
        return ValidationOutcome(valid=False, error=f"invalid JSON: {e}")
    return ValidationOutcome(valid=True, parsed=instance.model_dump())
