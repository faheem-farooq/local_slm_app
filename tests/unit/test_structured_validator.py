from local_slm.structured.schemas_catalog import ContactInfo, ProductInfo
from local_slm.structured.validator import validate


def test_valid_json_matching_schema_passes():
    outcome = validate(
        '{"name": "Sarah Chen", "email": "sarah@example.com", "phone": "555-1234"}',
        ContactInfo,
    )
    assert outcome.valid
    assert outcome.parsed["name"] == "Sarah Chen"
    assert outcome.error is None


def test_malformed_json_fails_with_error():
    outcome = validate("not json at all {{{", ContactInfo)
    assert not outcome.valid
    assert outcome.parsed is None
    assert outcome.error is not None


def test_valid_json_missing_required_field_fails():
    outcome = validate('{"name": "Sarah Chen", "email": "sarah@example.com"}', ContactInfo)
    assert not outcome.valid
    assert "phone" in outcome.error


def test_valid_json_wrong_type_fails():
    outcome = validate(
        '{"name": "P", "price": "not-a-number", "in_stock": true}',
        ProductInfo,
    )
    assert not outcome.valid
