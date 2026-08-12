from __future__ import annotations

from src.eval import EvalOutcome, evaluate_judgment
from src.judgment import IssueJudgment


def _judgment(
    label: str | None = "Bug", is_spam: bool = False, priority: str = "medium"
) -> IssueJudgment:
    return IssueJudgment(
        suggested_label=label,
        is_spam=is_spam,
        summary="Some summary text, worded differently each time.",
        priority=priority,
        rationale="Some rationale, also free text.",
        confidence=0.9,
    )


def test_confirmed_example_matching_new_judgment_passes() -> None:
    outcome = evaluate_judgment(
        golden_label="Bug",
        golden_is_spam=False,
        golden_priority="medium",
        was_corrected=False,
        new_judgment=_judgment(),
    )
    assert outcome == EvalOutcome.PASS


def test_confirmed_example_different_label_is_a_regression() -> None:
    outcome = evaluate_judgment(
        golden_label="Bug",
        golden_is_spam=False,
        golden_priority="medium",
        was_corrected=False,
        new_judgment=_judgment(label="Documentation"),
    )
    assert outcome == EvalOutcome.REGRESSION


def test_confirmed_example_different_priority_is_a_regression() -> None:
    outcome = evaluate_judgment(
        golden_label="Bug",
        golden_is_spam=False,
        golden_priority="medium",
        was_corrected=False,
        new_judgment=_judgment(priority="low"),
    )
    assert outcome == EvalOutcome.REGRESSION


def test_confirmed_example_different_is_spam_is_a_regression() -> None:
    outcome = evaluate_judgment(
        golden_label="Bug",
        golden_is_spam=False,
        golden_priority="medium",
        was_corrected=False,
        new_judgment=_judgment(is_spam=True),
    )
    assert outcome == EvalOutcome.REGRESSION


def test_corrected_example_reproducing_the_same_mistake_is_a_regression() -> None:
    outcome = evaluate_judgment(
        golden_label="Documentation",
        golden_is_spam=False,
        golden_priority="low",
        was_corrected=True,
        new_judgment=_judgment(label="Documentation", priority="low"),
    )
    assert outcome == EvalOutcome.REGRESSION


def test_corrected_example_with_a_different_new_judgment_needs_review() -> None:
    outcome = evaluate_judgment(
        golden_label="Documentation",
        golden_is_spam=False,
        golden_priority="low",
        was_corrected=True,
        new_judgment=_judgment(label="Array API", priority="low"),
    )
    assert outcome == EvalOutcome.NEEDS_REVIEW


def test_free_text_fields_are_never_compared() -> None:
    # summary/rationale differ every time in _judgment() by construction;
    # this just documents that they play no role in the outcome.
    outcome = evaluate_judgment(
        golden_label="Bug",
        golden_is_spam=False,
        golden_priority="medium",
        was_corrected=False,
        new_judgment=_judgment(),
    )
    assert outcome == EvalOutcome.PASS
