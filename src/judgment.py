from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IssueJudgment(BaseModel):
    """Structured judgment produced by the LLM for a single non-bot issue.

    This is the LLM's output contract only — issue_id/digest_id and other
    persistence concerns are attached separately when storing a judgment
    (see src/db.py), not part of what the model produces.

    Duplicate-candidate detection is intentionally not part of this
    schema - dropped from the MVP.
    """

    suggested_labels: list[str] = Field(
        default_factory=list,
        description=(
            "Topic/category label(s) the issue may warrant, e.g. "
            "['linear model']. Empty if there is no clear match. Most "
            "issues warrant exactly one label - suggest more than one "
            "only when the issue genuinely spans more than one area "
            "(e.g. both a specific module and a cross-cutting concern "
            "like Documentation), not as a hedge."
        ),
    )
    is_spam: bool = Field(
        default=False,
        description="Whether the issue looks like spam or is clearly off-topic.",
    )
    summary: str = Field(
        description="A short (1-2 sentence) summary of what the issue is about.",
    )
    priority: Literal["low", "medium", "high"] = Field(
        description=(
            "Judged importance of the issue, used to rank issues within "
            "the daily digest."
        ),
    )
    rationale: str = Field(
        description="A short explanation of why this judgment was made.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The model's confidence in this judgment, from 0 to 1.",
    )
