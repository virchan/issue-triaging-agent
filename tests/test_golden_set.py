"""Live regression suite against eval/golden_set.json.

Excluded from the default `pytest` run (see pyproject.toml's addopts) -
unlike every other test in this project, these call the real Gemini API
and need GEMINI_API_KEY. Run explicitly with:

    uv run pytest -m golden -v

CI runs this as a separate job from the main credential-free
lint+test gate (see .github/workflows/ci.yml) - a Gemini hiccup here
shouldn't block the fast, always-required gate everything else depends
on. See eval/rubric.md for what PASS/REGRESSION/NEEDS_REVIEW mean.

Calls are spaced out (CALL_DELAY_SECONDS) to stay under the free-tier
per-minute rate limit - firing all examples back-to-back tripped a real
429 RESOURCE_EXHAUSTED after 3 calls during development. A 429 here is
reported as a skip, not a failure: rate limiting is an infrastructure
hiccup, not evidence the model regressed, and conflating the two would
make this suite cry wolf on every quota hiccup.

Each example is judged with real recent_examples/similar_examples
few-shot context built from the *other* examples in the same snapshot
(self-excluded), not bare - production always builds both, and testing
judge() without them would validate a code path nothing in production
actually uses. similar_examples specifically reuses
find_similar_reviewed_examples, the same retrieval-augmented-generation
ranking function the real pipeline calls - this is what actually
exercises the RAG feature here, not a live DB query (CI has no database
access), a static, versioned snapshot of real embeddings instead (see
scripts/export_golden_set.py). An example with no stored embedding
degrades gracefully to no similar_examples for that one call, mirroring
the pipeline's own real degradation when embedding fails.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from src.db import ReviewedJudgment
from src.duplicate_detection import find_similar_reviewed_examples
from src.eval import EvalOutcome, evaluate_judgment
from src.gemini_client import GeminiJudge, GeminiUnavailableError
from src.github_client import GitHubClient
from src.judgment import IssueJudgment

load_dotenv()

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "eval" / "golden_set.json"
GEMINI_MODEL = "gemini-3.5-flash"
CALL_DELAY_SECONDS = 15

# Mirrors src.db.get_recent_reviewed_judgments' own default - the golden
# set is small enough that this rarely truncates anything in practice.
RECENT_EXAMPLES_LIMIT = 10


def _load_golden_set() -> list[dict[str, Any]]:
    return json.loads(GOLDEN_SET_PATH.read_text())


# Loaded once, at collection time - both the parametrization below and each
# test invocation need the full set (to build every other example's
# recent_examples/similar_examples context), and it's small enough that
# re-reading it per test would only add noise, not correctness.
_GOLDEN_SET = _load_golden_set()


def _as_reviewed_judgment(example: dict[str, Any]) -> ReviewedJudgment:
    return ReviewedJudgment(
        issue_title=example["issue_title"],
        issue_body=example["issue_body"],
        judgment=IssueJudgment(**example["judgment"]),
        correction_text=example["correction_text"],
    )


def _recent_examples_for(
    example: dict[str, Any], golden_set: list[dict[str, Any]]
) -> list[ReviewedJudgment]:
    """The other golden examples, most-recent-first by digest date - the
    same recency ordering get_recent_reviewed_judgments uses for real."""

    others = [
        other
        for other in golden_set
        if other["github_number"] != example["github_number"]
    ]
    others.sort(key=lambda other: other["digest_date"], reverse=True)
    return [_as_reviewed_judgment(other) for other in others[:RECENT_EXAMPLES_LIMIT]]


def _similar_examples_for(
    example: dict[str, Any], golden_set: list[dict[str, Any]]
) -> list[ReviewedJudgment]:
    """Real retrieval-augmented context, ranked by find_similar_reviewed_examples
    against the other examples' stored embeddings - empty if this example
    was never embedded, the same graceful degradation the real pipeline
    falls back to when embedding an issue fails."""

    if example["embedding"] is None:
        return []

    candidates = [
        (other["github_number"], other["embedding"], _as_reviewed_judgment(other))
        for other in golden_set
        if other["github_number"] != example["github_number"]
        and other["embedding"] is not None
    ]
    return find_similar_reviewed_examples(example["embedding"], candidates)


@pytest.fixture(scope="module")
def known_labels() -> list[str]:
    """Real, current scikit-learn labels - fetched live (unauthenticated,
    no secret needed) rather than snapshotted at export time, so this
    suite always grounds against what's actually valid right now."""

    with GitHubClient() as client:
        return client.fetch_labels("scikit-learn", "scikit-learn")


@pytest.fixture(scope="module")
def judge() -> GeminiJudge:
    return GeminiJudge(model=GEMINI_MODEL, api_key=os.environ["GEMINI_API_KEY"])


@pytest.mark.golden
@pytest.mark.parametrize(
    "example",
    _GOLDEN_SET,
    ids=lambda example: f"#{example['github_number']}",
)
def test_golden_example_does_not_regress(
    example: dict[str, Any],
    judge: GeminiJudge,
    known_labels: list[str],
) -> None:
    time.sleep(CALL_DELAY_SECONDS)

    try:
        new_judgment = judge.judge(
            title=example["issue_title"],
            body=example["issue_body"],
            known_labels=known_labels,
            recent_examples=_recent_examples_for(example, _GOLDEN_SET),
            similar_examples=_similar_examples_for(example, _GOLDEN_SET),
        )
    except GeminiUnavailableError as error:
        pytest.skip(f"Gemini unavailable for #{example['github_number']}: {error}")

    outcome = evaluate_judgment(
        golden_label=example["judgment"]["suggested_label"],
        golden_is_spam=example["judgment"]["is_spam"],
        golden_priority=example["judgment"]["priority"],
        new_judgment=new_judgment,
    )

    print(f"#{example['github_number']}: {outcome.value}")

    assert outcome != EvalOutcome.REGRESSION, (
        f"Regression on #{example['github_number']}: new judgment gave "
        f"label={new_judgment.suggested_label!r}, "
        f"is_spam={new_judgment.is_spam}, priority={new_judgment.priority!r} "
        f"vs. golden label={example['judgment']['suggested_label']!r}, "
        f"is_spam={example['judgment']['is_spam']}, "
        f"priority={example['judgment']['priority']!r}"
    )
