from __future__ import annotations

from src.eval import EvalOutcome, evaluate_judgment
from src.judgment import IssueJudgment


def _judgment(
    labels: list[str] | None = None, is_spam: bool = False, priority: str = "medium"
) -> IssueJudgment:
    return IssueJudgment(
        suggested_labels=["Bug"] if labels is None else labels,
        is_spam=is_spam,
        summary="Some summary text, worded differently each time.",
        priority=priority,
        rationale="Some rationale, also free text.",
        confidence=0.9,
    )


def test_matching_new_judgment_passes() -> None:
    outcome = evaluate_judgment(
        golden_labels=["Bug"],
        golden_is_spam=False,
        golden_priority="medium",
        new_judgment=_judgment(),
    )
    assert outcome == EvalOutcome.PASS


def test_different_label_is_a_regression() -> None:
    outcome = evaluate_judgment(
        golden_labels=["Bug"],
        golden_is_spam=False,
        golden_priority="medium",
        new_judgment=_judgment(labels=["Documentation"]),
    )
    assert outcome == EvalOutcome.REGRESSION


def test_missing_one_of_several_golden_labels_is_a_regression() -> None:
    outcome = evaluate_judgment(
        golden_labels=["Documentation", "module:cluster"],
        golden_is_spam=False,
        golden_priority="low",
        new_judgment=_judgment(labels=["Documentation"], priority="low"),
    )
    assert outcome == EvalOutcome.REGRESSION


def test_extra_label_beyond_golden_still_passes() -> None:
    # golden_labels only needs to be a subset of what the new judgment
    # suggests - an additional, unconfirmed label isn't the same kind of
    # failure as dropping a confirmed one.
    outcome = evaluate_judgment(
        golden_labels=["Bug"],
        golden_is_spam=False,
        golden_priority="medium",
        new_judgment=_judgment(labels=["Bug", "module:tree"]),
    )
    assert outcome == EvalOutcome.PASS


def test_different_priority_is_a_regression() -> None:
    outcome = evaluate_judgment(
        golden_labels=["Bug"],
        golden_is_spam=False,
        golden_priority="medium",
        new_judgment=_judgment(priority="low"),
    )
    assert outcome == EvalOutcome.REGRESSION


def test_different_is_spam_is_a_regression() -> None:
    outcome = evaluate_judgment(
        golden_labels=["Bug"],
        golden_is_spam=False,
        golden_priority="medium",
        new_judgment=_judgment(is_spam=True),
    )
    assert outcome == EvalOutcome.REGRESSION


def test_corrected_example_matching_the_confirmed_label_passes() -> None:
    # golden_labels here is the *post-correction* value - update_judgment
    # overwrites judgments in place after every correction, so this is
    # the human-confirmed final state, not "the original mistake." A
    # match against it is a real pass, not a reproduced mistake.
    outcome = evaluate_judgment(
        golden_labels=["Documentation"],
        golden_is_spam=False,
        golden_priority="low",
        new_judgment=_judgment(labels=["Documentation"], priority="low"),
    )
    assert outcome == EvalOutcome.PASS


def test_corrected_example_with_a_different_new_judgment_is_a_regression() -> None:
    outcome = evaluate_judgment(
        golden_labels=["Documentation"],
        golden_is_spam=False,
        golden_priority="low",
        new_judgment=_judgment(labels=["Array API"], priority="low"),
    )
    assert outcome == EvalOutcome.REGRESSION


def test_free_text_fields_are_never_compared() -> None:
    # summary/rationale differ every time in _judgment() by construction;
    # this just documents that they play no role in the outcome.
    outcome = evaluate_judgment(
        golden_labels=["Bug"],
        golden_is_spam=False,
        golden_priority="medium",
        new_judgment=_judgment(),
    )
    assert outcome == EvalOutcome.PASS
