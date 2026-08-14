from __future__ import annotations

from enum import Enum

from src.judgment import IssueJudgment


class EvalOutcome(str, Enum):
    """Result of comparing a new judgment against one golden example.

    PASS: the golden example was confirmed correct, and the new judgment
    matches it exactly on the structured decision fields (suggested_label,
    is_spam, priority) - free-text fields (summary, rationale) are
    deliberately not compared, since differently worded but equally
    correct phrasing shouldn't count as a failure.

    REGRESSION: either (a) a confirmed example whose new judgment no
    longer matches a previously-confirmed-correct answer, or (b) a
    corrected example whose new judgment reproduces the exact same
    mistake that was corrected. Both are real, automatable failure
    signals - CI should fail on these.

    NEEDS_REVIEW: a corrected example whose new judgment differs from
    the original (wrong) one. Not an automatic pass: the correction was
    free text (e.g. "#34648 should be labelled as \"array API\"."), not
    a structured expected answer, so there's no automated way to confirm
    the new judgment actually satisfies the correction's intent. Flagged
    for human review rather than faked as a pass - see design-plan.md
    §8's call for "an explicit rubric for partial credit vs. failure"
    instead of an exact-match-only comparison that can't represent this
    case honestly.
    """

    PASS = "pass"
    REGRESSION = "regression"
    NEEDS_REVIEW = "needs_review"


def evaluate_judgment(
    *,
    golden_label: str | None,
    golden_is_spam: bool,
    golden_priority: str,
    was_corrected: bool,
    new_judgment: IssueJudgment,
) -> EvalOutcome:
    """Classify a new judgment against one golden example's structured fields."""

    matches_original = (
        new_judgment.suggested_label == golden_label
        and new_judgment.is_spam == golden_is_spam
        and new_judgment.priority == golden_priority
    )

    if not was_corrected:
        return EvalOutcome.PASS if matches_original else EvalOutcome.REGRESSION

    return EvalOutcome.REGRESSION if matches_original else EvalOutcome.NEEDS_REVIEW
