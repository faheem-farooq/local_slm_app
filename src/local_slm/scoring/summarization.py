"""Scores summarization responses with a proxy metric, not a reference-based one.

No hand-written reference summaries exist (out of scope for a solo project), so
quality is approximated as 0.7 * keyword_coverage + 0.3 * length_score, where
length_score peaks for summaries within an ideal word-count band and decays
outside it. This is explicitly a heuristic proxy, not a substitute for human or
LLM-judge evaluation -- documented here and called out plainly in the README.
"""

from __future__ import annotations

from pydantic import BaseModel

IDEAL_MIN_WORDS = 15
IDEAL_MAX_WORDS = 45


class SummarizationScore(BaseModel):
    keyword_coverage: float
    length_score: float
    overall_score: float
    matched_keywords: list[str]
    word_count: int


def _length_score(word_count: int) -> float:
    if word_count == 0:
        return 0.0
    if IDEAL_MIN_WORDS <= word_count <= IDEAL_MAX_WORDS:
        return 1.0
    if word_count < IDEAL_MIN_WORDS:
        return word_count / IDEAL_MIN_WORDS
    overflow = word_count - IDEAL_MAX_WORDS
    return max(0.0, 1.0 - overflow / IDEAL_MAX_WORDS)


def score_summarization(response_text: str, expected_keywords: list[str]) -> SummarizationScore:
    text_norm = response_text.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in text_norm]
    keyword_coverage = len(matched) / len(expected_keywords) if expected_keywords else 0.0
    word_count = len(response_text.split())
    length_score = _length_score(word_count)
    overall = 0.7 * keyword_coverage + 0.3 * length_score

    return SummarizationScore(
        keyword_coverage=keyword_coverage,
        length_score=length_score,
        overall_score=overall,
        matched_keywords=matched,
        word_count=word_count,
    )
