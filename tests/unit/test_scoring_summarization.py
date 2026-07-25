from local_slm.scoring.summarization import score_summarization


def test_full_keyword_coverage_and_ideal_length_scores_near_one():
    text = " ".join(["word"] * 20) + " photosynthesis chlorophyll oxygen glucose"
    score = score_summarization(text, ["photosynthesis", "chlorophyll", "oxygen", "glucose"])
    assert score.keyword_coverage == 1.0
    assert score.length_score == 1.0
    assert score.overall_score == 1.0


def test_missing_keywords_reduces_coverage():
    score = score_summarization("a short summary", ["photosynthesis", "chlorophyll"])
    assert score.keyword_coverage == 0.0
    assert score.overall_score < 1.0


def test_too_short_response_reduces_length_score():
    score = score_summarization("short", ["short"])
    assert score.keyword_coverage == 1.0
    assert score.length_score < 1.0


def test_too_long_response_reduces_length_score():
    text = " ".join(["word"] * 200) + " keyword"
    score = score_summarization(text, ["keyword"])
    assert score.length_score < 1.0


def test_empty_expected_keywords_gives_zero_coverage_not_error():
    score = score_summarization("anything", [])
    assert score.keyword_coverage == 0.0
