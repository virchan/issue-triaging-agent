from __future__ import annotations

import math

from src.db import ReviewedJudgment
from src.duplicate_detection import (
    MINIMUM_SIMILARITY,
    find_possible_duplicate,
    find_similar_reviewed_examples,
    issue_text,
    similarity_score,
)
from src.judgment import IssueJudgment


def _reviewed(title: str, correction_text: str | None = None) -> ReviewedJudgment:
    return ReviewedJudgment(
        issue_title=title,
        issue_body=None,
        judgment=IssueJudgment(
            suggested_label="Bug",
            is_spam=False,
            summary="s",
            priority="medium",
            rationale="r",
            confidence=0.8,
        ),
        correction_text=correction_text,
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


def test_find_similar_reviewed_examples_ranks_most_similar_first() -> None:
    target = [1.0, 0.0]
    close = _reviewed("Close match")
    exact = _reviewed("Exact match")
    candidates = [
        (101, [0.9, 0.1], close),
        (102, [1.0, 0.0], exact),
    ]

    result = find_similar_reviewed_examples(target, candidates)

    assert [r.issue_title for r in result] == ["Exact match", "Close match"]
    assert result[0].similarity == 1.0


def test_find_similar_reviewed_examples_excludes_below_the_floor() -> None:
    # Orthogonal vectors score 0.0, well below MINIMUM_SIMILARITY.
    candidates = [(101, [0.0, 1.0], _reviewed("Unrelated"))]
    assert find_similar_reviewed_examples([1.0, 0.0], candidates) == []


def test_find_similar_reviewed_examples_respects_the_limit() -> None:
    candidates = [(100 + i, [1.0, 0.0], _reviewed(f"Match {i}")) for i in range(5)]

    result = find_similar_reviewed_examples([1.0, 0.0], candidates, limit=2)

    assert len(result) == 2


def test_find_similar_reviewed_examples_empty_pool_returns_empty_list() -> None:
    assert find_similar_reviewed_examples([1.0, 0.0], []) == []


def test_find_similar_reviewed_examples_preserves_correction_text() -> None:
    reviewed = _reviewed("Corrected issue", correction_text="should be Documentation")
    result = find_similar_reviewed_examples([1.0, 0.0], [(101, [1.0, 0.0], reviewed)])

    assert result[0].correction_text == "should be Documentation"
    assert result[0].similarity == 1.0
