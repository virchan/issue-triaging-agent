from __future__ import annotations

from typing import Any

import pytest

from src.db import ReviewedJudgment
from src.gemini_client import (
    GeminiConfigurationError,
    GeminiJudge,
    GeminiResponseError,
    GeminiUnavailableError,
    _format_examples,
)
from src.judgment import IssueJudgment

KNOWN_LABELS = ["Bug", "Documentation", "module:preprocessing"]


@pytest.fixture
def client(mocker: Any) -> Any:
    return mocker.Mock()


@pytest.fixture
def judge(client: Any) -> GeminiJudge:
    return GeminiJudge(model="gemini-3.5-flash", client=client)


def response_text(
    *,
    suggested_label: str | None = "Bug",
    is_spam: bool = False,
    summary: str = "A short summary.",
    priority: str = "medium",
    rationale: str = "Because of X.",
    confidence: float = 0.8,
) -> str:
    return IssueJudgment(
        suggested_label=suggested_label,
        is_spam=is_spam,
        summary=summary,
        priority=priority,
        rationale=rationale,
        confidence=confidence,
    ).model_dump_json()


def test_requires_api_key_without_injected_client() -> None:
    with pytest.raises(GeminiConfigurationError, match="API key is required"):
        GeminiJudge(model="gemini-3.5-flash")


def test_requires_model_name() -> None:
    with pytest.raises(GeminiConfigurationError, match="model name is required"):
        GeminiJudge(model="   ", client=object())


def test_judge_returns_parsed_judgment(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(
        text=response_text(suggested_label="module:preprocessing")
    )

    judgment = judge.judge(
        title="OneHotEncoder crashes",
        body="Steps to reproduce...",
        known_labels=KNOWN_LABELS,
    )

    assert judgment.suggested_label == "module:preprocessing"
    assert judgment.priority == "medium"


def test_judge_sends_labels_and_issue_content_in_prompt(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(text=response_text())

    judge.judge(
        title="Crash on fit()",
        body="full traceback here",
        known_labels=KNOWN_LABELS,
    )

    call = client.models.generate_content.call_args
    prompt = call.kwargs["contents"]
    config = call.kwargs["config"]

    assert "Crash on fit()" in prompt
    assert "full traceback here" in prompt
    assert '"Bug"' in prompt
    assert '"module:preprocessing"' in prompt
    assert call.kwargs["model"] == "gemini-3.5-flash"
    assert config.response_mime_type == "application/json"
    assert config.response_schema is IssueJudgment


def test_judge_rejects_blank_title(client: Any, judge: GeminiJudge) -> None:
    with pytest.raises(ValueError, match="title is required"):
        judge.judge(title="   ", body=None, known_labels=KNOWN_LABELS)

    client.models.generate_content.assert_not_called()


def test_judge_translates_provider_failure(client: Any, judge: GeminiJudge) -> None:
    client.models.generate_content.side_effect = RuntimeError("provider details")

    with pytest.raises(GeminiUnavailableError, match="could not complete"):
        judge.judge(title="Some issue", body=None, known_labels=KNOWN_LABELS)


def test_judge_rejects_empty_response(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(text=None)

    with pytest.raises(GeminiResponseError, match="empty response"):
        judge.judge(title="Some issue", body=None, known_labels=KNOWN_LABELS)


def test_judge_rejects_invalid_json(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(text="not json")

    with pytest.raises(GeminiResponseError, match="invalid response"):
        judge.judge(title="Some issue", body=None, known_labels=KNOWN_LABELS)


def test_judge_rejects_label_outside_known_list(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(
        text=response_text(suggested_label="Totally Invented Label")
    )

    with pytest.raises(GeminiResponseError, match="unknown label"):
        judge.judge(title="Some issue", body=None, known_labels=KNOWN_LABELS)


def test_judge_allows_null_suggested_label(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(
        text=response_text(suggested_label=None)
    )

    judgment = judge.judge(title="Some issue", body=None, known_labels=KNOWN_LABELS)
    assert judgment.suggested_label is None


def test_format_examples_empty_list_returns_empty_string() -> None:
    assert _format_examples([]) == ""


def test_format_examples_shows_correction_and_confirmation() -> None:
    examples = [
        ReviewedJudgment(
            issue_title="Corrected issue",
            issue_body=None,
            judgment=IssueJudgment(
                suggested_label="Bug",
                is_spam=False,
                summary="s",
                priority="medium",
                rationale="r",
                confidence=0.8,
            ),
            correction_text="#1 should be Documentation",
        ),
        ReviewedJudgment(
            issue_title="Confirmed issue",
            issue_body=None,
            judgment=IssueJudgment(
                suggested_label="Documentation",
                is_spam=False,
                summary="s",
                priority="low",
                rationale="r",
                confidence=0.9,
            ),
            correction_text=None,
        ),
    ]

    text = _format_examples(examples)

    assert "Corrected issue" in text
    assert "corrected by the operator: '#1 should be Documentation'" in text
    assert "Confirmed issue" in text
    assert "confirmed correct, no correction needed" in text


def test_judge_includes_recent_examples_in_prompt(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(text=response_text())
    examples = [
        ReviewedJudgment(
            issue_title="A past issue about OPTICS",
            issue_body=None,
            judgment=IssueJudgment(
                suggested_label="Bug",
                is_spam=False,
                summary="s",
                priority="medium",
                rationale="r",
                confidence=0.8,
            ),
            correction_text=None,
        )
    ]

    judge.judge(
        title="New issue",
        body=None,
        known_labels=KNOWN_LABELS,
        recent_examples=examples,
    )

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "A past issue about OPTICS" in prompt
    assert "confirmed correct" in prompt


def test_judge_without_recent_examples_omits_the_section(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(text=response_text())

    judge.judge(title="New issue", body=None, known_labels=KNOWN_LABELS)

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "Recent reviewed judgments" not in prompt
