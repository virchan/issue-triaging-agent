from __future__ import annotations

import math

from src.duplicate_detection import (
    MINIMUM_SIMILARITY,
    find_possible_duplicate,
    issue_text,
    similarity_score,
)


def test_similarity_score_identical_text_is_one() -> None:
    text = "PCA raises a ValueError on float32 input"
    assert similarity_score(text, text) == 1.0


def test_similarity_score_completely_different_text_is_low() -> None:
    score = similarity_score(
        "PCA raises a ValueError on float32 input",
        "Documentation typo in the RandomForestClassifier example",
    )
    assert score < 0.3


def test_similarity_score_is_symmetric() -> None:
    a = "Race condition in RecordingCallback during process-based parallelism"
    b = "RecordingCallback has a race condition under process parallelism"
    assert similarity_score(a, b) == similarity_score(b, a)


def test_similarity_score_empty_strings_are_identical_by_convention() -> None:
    """difflib's own documented behavior for two empty sequences - noted
    explicitly so it's a known, tested edge case, not a surprise later."""

    assert similarity_score("", "") == 1.0


def test_issue_text_combines_title_and_body() -> None:
    text = issue_text("A title", "A body")
    assert "A title" in text
    assert "A body" in text


def test_issue_text_handles_a_missing_body() -> None:
    text = issue_text("A title", None)
    assert "A title" in text


def test_find_possible_duplicate_returns_the_best_match() -> None:
    target = [1.0, 0.0]
    candidates = [
        (101, [0.0, 1.0]),  # orthogonal - similarity 0.0
        (102, [1.0, 0.0]),  # identical direction - similarity 1.0
        (103, [0.9, 0.1]),  # close, but not the best
    ]
    assert find_possible_duplicate(target, candidates) == (102, 1.0)


def test_find_possible_duplicate_returns_none_for_an_empty_pool() -> None:
    assert find_possible_duplicate([1.0, 0.0], []) is None


def test_find_possible_duplicate_returns_none_below_the_sanity_floor() -> None:
    # Orthogonal vectors score 0.0, well below MINIMUM_SIMILARITY.
    assert find_possible_duplicate([1.0, 0.0], [(101, [0.0, 1.0])]) is None


def test_find_possible_duplicate_returns_a_match_just_above_the_floor() -> None:
    # A candidate scoring just above MINIMUM_SIMILARITY should count -
    # comparing against a value clearly above it, not the exact boundary
    # itself, since floating-point rounding at an exact threshold is
    # inherently unreliable to test against.
    target = [1.0, 0.0]
    theta = math.acos(MINIMUM_SIMILARITY + 0.05)
    candidate = [math.cos(theta), math.sin(theta)]
    result = find_possible_duplicate(target, [(101, candidate)])
    assert result is not None
    assert result[0] == 101
    assert result[1] > MINIMUM_SIMILARITY
