from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.judgment import IssueJudgment


def _valid_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "suggested_labels": ["Bug"],
        "is_spam": False,
        "summary": "A short summary.",
        "priority": "medium",
        "rationale": "Because of X.",
        "confidence": 0.8,
    }
    kwargs.update(overrides)
    return kwargs


def test_valid_judgment_constructs() -> None:
    judgment = IssueJudgment(**_valid_kwargs())
    assert judgment.suggested_labels == ["Bug"]
    assert judgment.priority == "medium"


def test_suggested_labels_defaults_to_empty_list() -> None:
    kwargs = _valid_kwargs()
    del kwargs["suggested_labels"]
    assert IssueJudgment(**kwargs).suggested_labels == []


def test_is_spam_defaults_to_false() -> None:
    kwargs = _valid_kwargs()
    del kwargs["is_spam"]
    assert IssueJudgment(**kwargs).is_spam is False


@pytest.mark.parametrize("priority", ["urgent", "", "Low", "HIGH"])
def test_invalid_priority_rejected(priority: str) -> None:
    with pytest.raises(ValidationError):
        IssueJudgment(**_valid_kwargs(priority=priority))


@pytest.mark.parametrize("confidence", [-0.1, 1.1, 2.0])
def test_confidence_out_of_range_rejected(confidence: float) -> None:
    with pytest.raises(ValidationError):
        IssueJudgment(**_valid_kwargs(confidence=confidence))


@pytest.mark.parametrize("confidence", [0.0, 1.0, 0.5])
def test_confidence_boundary_values_accepted(confidence: float) -> None:
    assert (
        IssueJudgment(**_valid_kwargs(confidence=confidence)).confidence == confidence
    )


@pytest.mark.parametrize(
    "missing_field", ["summary", "priority", "rationale", "confidence"]
)
def test_missing_required_field_rejected(missing_field: str) -> None:
    kwargs = _valid_kwargs()
    del kwargs[missing_field]
    with pytest.raises(ValidationError):
        IssueJudgment(**kwargs)
