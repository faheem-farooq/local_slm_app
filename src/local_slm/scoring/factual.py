"""Scores factual_qa responses: exact/substring match, falling back to fuzzy match.

The fuzzy threshold (FUZZY_MATCH_THRESHOLD, currently 85) is a judgment call
documented here and in the README, not a rigorously derived cutoff -- it exists
to tolerate minor phrasing differences ("Paris, France" vs "Paris") without
accepting genuinely wrong answers.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from local_slm.config import FUZZY_MATCH_THRESHOLD


def score_factual(response_text: str, expected_answer: str) -> bool:
    response_norm = response_text.strip().lower()
    expected_norm = expected_answer.strip().lower()

    if not response_norm:
        return False
    if response_norm == expected_norm or expected_norm in response_norm:
        return True
    return fuzz.token_sort_ratio(response_norm, expected_norm) >= FUZZY_MATCH_THRESHOLD
