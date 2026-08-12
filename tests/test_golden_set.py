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
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from src.eval import EvalOutcome, evaluate_judgment
from src.gemini_client import GeminiJudge, GeminiUnavailableError
from src.github_client import GitHubClient

load_dotenv()

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "eval" / "golden_set.json"
GEMINI_MODEL = "gemini-3.5-flash"
CALL_DELAY_SECONDS = 15


def _load_golden_set() -> list[dict[str, Any]]:
    return json.loads(GOLDEN_SET_PATH.read_text())


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
    _load_golden_set(),
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
        )
    except GeminiUnavailableError as error:
        pytest.skip(f"Gemini unavailable for #{example['github_number']}: {error}")

    outcome = evaluate_judgment(
        golden_label=example["judgment"]["suggested_label"],
        golden_is_spam=example["judgment"]["is_spam"],
        golden_priority=example["judgment"]["priority"],
        was_corrected=example["correction_text"] is not None,
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
