"""Scores json_extraction responses: schema validity, then per-field accuracy.

Field accuracy is only meaningful once the response parses at all, so a schema
failure scores 0.0 rather than attempting partial credit on unparseable output.
"""

from __future__ import annotations

from pydantic import BaseModel

from local_slm.structured.validator import validate


class JsonQualityScore(BaseModel):
    schema_valid: bool
    correct_fields: int
    total_fields: int
    field_accuracy: float


def _fields_match(actual: object, expected: object) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual == expected
    if isinstance(expected, int | float):
        try:
            return abs(float(actual) - float(expected)) < 0.01
        except (TypeError, ValueError):
            return False
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().lower() == expected.strip().lower()
    return actual == expected


def score_json_extraction(
    raw_text: str, expected_fields: dict, schema_cls: type[BaseModel]
) -> JsonQualityScore:
    outcome = validate(raw_text, schema_cls)
    total_fields = len(expected_fields)

    if not outcome.valid or outcome.parsed is None:
        return JsonQualityScore(
            schema_valid=False, correct_fields=0, total_fields=total_fields, field_accuracy=0.0
        )

    correct = sum(
        1
        for field, expected_value in expected_fields.items()
        if _fields_match(outcome.parsed.get(field), expected_value)
    )
    return JsonQualityScore(
        schema_valid=True,
        correct_fields=correct,
        total_fields=total_fields,
        field_accuracy=correct / total_fields if total_fields else 0.0,
    )
