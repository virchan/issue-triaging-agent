from __future__ import annotations

from enum import Enum

from src.judgment import IssueJudgment


class EvalOutcome(str, Enum):
    """Result of comparing a new judgment against one golden example.

    PASS: the new judgment matches the golden example exactly on the
    structured decision fields (suggested_label, is_spam, priority) -
    free-text fields (summary, rationale) are deliberately not compared,
    since differently worded but equally correct phrasing shouldn't count
    as a failure.

    REGRESSION: the new judgment differs from the golden example on any
    structured decision field.

    Confirmed and corrected golden examples are scored by the identical
    rule, deliberately: a correction always triggers a fresh
    judge_with_correction call whose result is written back into
    judgments in place (see src.db.update_judgment) - there is no column
    anywhere that preserves the pre-correction value, only which fields
    changed. golden_label/golden_is_spam/golden_priority are therefore
    always the human-confirmed *final* state for a corrected example,
    exactly as they are for a plain confirmed one - never "the original
    mistake" the correction was written against, so there is nothing
    left to score a corrected example against except the same
    match/no-match rule a confirmed example uses.
    """

    PASS = "pass"
    REGRESSION = "regression"


def evaluate_judgment(
    *,
    golden_label: str | None,
    golden_is_spam: bool,
    golden_priority: str,
    new_judgment: IssueJudgment,
) -> EvalOutcome:
    """Classify a new judgment against one golden example's structured fields."""

    matches = (
        new_judgment.suggested_label == golden_label
        and new_judgment.is_spam == golden_is_spam
        and new_judgment.priority == golden_priority
    )

    return EvalOutcome.PASS if matches else EvalOutcome.REGRESSION
