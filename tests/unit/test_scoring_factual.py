from local_slm.scoring.factual import score_factual


def test_exact_match():
    assert score_factual("Paris", "Paris") is True


def test_case_insensitive_match():
    assert score_factual("paris", "Paris") is True


def test_substring_match_for_verbose_answer():
    assert score_factual("The capital of France is Paris.", "Paris") is True


def test_fuzzy_match_for_minor_misspelling():
    assert score_factual("Neal Armstrong", "Neil Armstrong") is True


def test_wrong_answer_fails():
    assert score_factual("London", "Paris") is False


def test_empty_response_fails():
    assert score_factual("", "Paris") is False
