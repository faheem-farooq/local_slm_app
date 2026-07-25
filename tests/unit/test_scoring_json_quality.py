from local_slm.scoring.json_quality import score_json_extraction
from local_slm.structured.schemas_catalog import ContactInfo, ProductInfo


def test_fully_correct_response_scores_perfect():
    raw = '{"name": "Sarah Chen", "email": "sarah@example.com", "phone": "555-1234"}'
    expected = {"name": "Sarah Chen", "email": "sarah@example.com", "phone": "555-1234"}
    score = score_json_extraction(raw, expected, ContactInfo)
    assert score.schema_valid
    assert score.field_accuracy == 1.0
    assert score.correct_fields == 3


def test_schema_invalid_response_scores_zero():
    score = score_json_extraction(
        "not json", {"name": "A", "email": "a@b.com", "phone": "1"}, ContactInfo
    )
    assert not score.schema_valid
    assert score.field_accuracy == 0.0


def test_partial_field_match_scores_partial_credit():
    raw = '{"name": "Wrong Name", "email": "sarah@example.com", "phone": "555-1234"}'
    expected = {"name": "Sarah Chen", "email": "sarah@example.com", "phone": "555-1234"}
    score = score_json_extraction(raw, expected, ContactInfo)
    assert score.schema_valid
    assert score.correct_fields == 2
    assert score.field_accuracy == 2 / 3


def test_numeric_field_uses_tolerance_not_exact_equality():
    raw = '{"name": "Widget", "price": 19.990001, "in_stock": true}'
    expected = {"name": "Widget", "price": 19.99, "in_stock": True}
    score = score_json_extraction(raw, expected, ProductInfo)
    assert score.field_accuracy == 1.0


def test_case_insensitive_string_match():
    raw = '{"name": "widget", "price": 1.0, "in_stock": true}'
    expected = {"name": "Widget", "price": 1.0, "in_stock": True}
    score = score_json_extraction(raw, expected, ProductInfo)
    assert score.field_accuracy == 1.0
