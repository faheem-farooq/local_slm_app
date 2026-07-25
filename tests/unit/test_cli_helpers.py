import pytest

from local_slm.cli import _parse_trials_overrides


def test_parses_single_override():
    assert _parse_trials_overrides(["mistral:7b=2"]) == {"mistral:7b": 2}


def test_parses_multiple_overrides():
    result = _parse_trials_overrides(["mistral:7b=2", "phi4-mini:3.8b=3"])
    assert result == {"mistral:7b": 2, "phi4-mini:3.8b": 3}


def test_empty_list_returns_empty_dict():
    assert _parse_trials_overrides([]) == {}


def test_rejects_missing_equals_sign():
    with pytest.raises(ValueError):
        _parse_trials_overrides(["mistral:7b"])
