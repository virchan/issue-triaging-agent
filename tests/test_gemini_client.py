from __future__ import annotations

from typing import Any

import pytest

from src.db import ReviewedJudgment
from src.gemini_client import (
    _JUDGE_INSTRUCTIONS,
    _REJUDGE_INSTRUCTIONS,
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


def test_instructions_load_from_their_template_files() -> None:
    """Regression test for the entry-91 refactor: _JUDGE_INSTRUCTIONS and
    _REJUDGE_INSTRUCTIONS moved from inline string constants to
    src/templates/*-instructions.md.jinja, loaded via render_template.
    Confirms the files actually loaded (not empty, not the wrong file)
    rather than trusting the refactor silently."""

    assert "triage assistant for the scikit-learn" in _JUDGE_INSTRUCTIONS
    assert "suggested_label must be exactly one label" in _JUDGE_INSTRUCTIONS
    assert "revising a previous triage judgment" in _REJUDGE_INSTRUCTIONS
    assert "Produce a revised judgment" in _REJUDGE_INSTRUCTIONS


def test_judge_rejects_blank_title(client: Any, judge: GeminiJudge) -> None:
    with pytest.raises(ValueError, match="title is required"):
        judge.judge(title="   ", body=None, known_labels=KNOWN_LABELS)

    client.models.generate_content.assert_not_called()


def test_judge_translates_provider_failure(client: Any, judge: GeminiJudge) -> None:
    client.models.generate_content.side_effect = RuntimeError("provider details")

    with pytest.raises(GeminiUnavailableError, match="could not complete"):
        judge.judge(title="Some issue", body=None, known_labels=KNOWN_LABELS)


def test_judge_logs_latency_on_success(
    mocker: Any, client: Any, judge: GeminiJudge, caplog: Any
) -> None:
    client.models.generate_content.return_value = mocker.Mock(text=response_text())

    with caplog.at_level("INFO"):
        judge.judge(title="Some issue", body=None, known_labels=KNOWN_LABELS)

    records = [
        r for r in caplog.records if getattr(r, "event", None) == "judgment_latency"
    ]
    assert len(records) == 1
    assert records[0].outcome == "ok"
    assert records[0].model == "gemini-3.5-flash"
    assert records[0].latency_seconds >= 0


def test_judge_logs_latency_on_failure(
    client: Any, judge: GeminiJudge, caplog: Any
) -> None:
    client.models.generate_content.side_effect = RuntimeError("provider details")

    with caplog.at_level("INFO"), pytest.raises(GeminiUnavailableError):
        judge.judge(title="Some issue", body=None, known_labels=KNOWN_LABELS)

    records = [
        r for r in caplog.records if getattr(r, "event", None) == "judgment_latency"
    ]
    assert len(records) == 1
    assert records[0].outcome == "error"


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


def test_judge_rejects_missing_required_field(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    """Valid JSON, but missing a required field (e.g. `summary` dropped
    mid-generation) - a different failure shape than plain-invalid JSON,
    exercising the same ValidationError guardrail."""

    client.models.generate_content.return_value = mocker.Mock(
        text='{"suggested_label": "Bug", "is_spam": false, "priority": "medium", '
        '"rationale": "r", "confidence": 0.8}'
    )

    with pytest.raises(GeminiResponseError, match="invalid response"):
        judge.judge(title="Some issue", body=None, known_labels=KNOWN_LABELS)


def test_judge_rejects_invalid_priority_value(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    """Valid JSON, valid types, but priority outside the low/medium/high
    Literal - the model hallucinating a value outside its own schema."""

    client.models.generate_content.return_value = mocker.Mock(
        text='{"suggested_label": "Bug", "is_spam": false, "summary": "s", '
        '"priority": "urgent", "rationale": "r", "confidence": 0.8}'
    )

    with pytest.raises(GeminiResponseError, match="invalid response"):
        judge.judge(title="Some issue", body=None, known_labels=KNOWN_LABELS)


def test_judge_rejects_confidence_out_of_range(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(
        text='{"suggested_label": "Bug", "is_spam": false, "summary": "s", '
        '"priority": "medium", "rationale": "r", "confidence": 1.5}'
    )

    with pytest.raises(GeminiResponseError, match="invalid response"):
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
    assert "corrected: '#1 should be Documentation'" in text
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


PREVIOUS_JUDGMENT = IssueJudgment(
    suggested_label="module:preprocessing",
    is_spam=False,
    summary="Old summary.",
    priority="medium",
    rationale="Old rationale.",
    confidence=0.8,
)


def test_judge_with_correction_returns_revised_judgment(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(
        text=response_text(suggested_label="Bug", summary="Revised summary.")
    )

    judgment = judge.judge_with_correction(
        title="DictionaryLearning breaks Pipeline.predict",
        body="Steps to reproduce...",
        previous_judgment=PREVIOUS_JUDGMENT,
        correction_text="Should also carry the float32 label.",
        known_labels=KNOWN_LABELS,
    )

    assert judgment.suggested_label == "Bug"
    assert judgment.summary == "Revised summary."


def test_judge_with_correction_sends_previous_judgment_and_correction_in_prompt(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(text=response_text())

    judge.judge_with_correction(
        title="DictionaryLearning breaks Pipeline.predict",
        body="full traceback here",
        previous_judgment=PREVIOUS_JUDGMENT,
        correction_text="Should also carry the float32 label.",
        known_labels=KNOWN_LABELS,
    )

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "DictionaryLearning breaks Pipeline.predict" in prompt
    assert "full traceback here" in prompt
    assert "Should also carry the float32 label." in prompt
    assert '"module:preprocessing"' in prompt  # previous judgment, JSON-dumped
    assert "Old rationale." in prompt


def test_judge_with_correction_rejects_blank_title(
    client: Any, judge: GeminiJudge
) -> None:
    with pytest.raises(ValueError, match="title is required"):
        judge.judge_with_correction(
            title="   ",
            body=None,
            previous_judgment=PREVIOUS_JUDGMENT,
            correction_text="fix this",
            known_labels=KNOWN_LABELS,
        )

    client.models.generate_content.assert_not_called()


def test_judge_with_correction_rejects_blank_correction(
    client: Any, judge: GeminiJudge
) -> None:
    with pytest.raises(ValueError, match="correction is required"):
        judge.judge_with_correction(
            title="Some issue",
            body=None,
            previous_judgment=PREVIOUS_JUDGMENT,
            correction_text="   ",
            known_labels=KNOWN_LABELS,
        )

    client.models.generate_content.assert_not_called()


def test_judge_with_correction_rejects_label_outside_known_list(
    mocker: Any, client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.return_value = mocker.Mock(
        text=response_text(suggested_label="Totally Invented Label")
    )

    with pytest.raises(GeminiResponseError, match="unknown label"):
        judge.judge_with_correction(
            title="Some issue",
            body=None,
            previous_judgment=PREVIOUS_JUDGMENT,
            correction_text="fix this",
            known_labels=KNOWN_LABELS,
        )


def test_judge_with_correction_translates_provider_failure(
    client: Any, judge: GeminiJudge
) -> None:
    client.models.generate_content.side_effect = RuntimeError("provider details")

    with pytest.raises(GeminiUnavailableError, match="could not complete"):
        judge.judge_with_correction(
            title="Some issue",
            body=None,
            previous_judgment=PREVIOUS_JUDGMENT,
            correction_text="fix this",
            known_labels=KNOWN_LABELS,
        )
