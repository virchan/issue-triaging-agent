from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from src.corrections import (
    CaptureResult,
    capture_corrections,
    extract_referenced_issue_number,
    format_acknowledgment,
)
from src.github_client import GitHubComment


def _comment(comment_id: int, body: str) -> GitHubComment:
    return GitHubComment(
        id=comment_id,
        body=body,
        created_at=dt.datetime(2026, 8, 4, 12, 0, 0, tzinfo=dt.UTC),
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("#34649 is not about linear model, it's about SVC", 34649),
        ("Re #100: looks right to me", 100),
        ("no reference here at all", None),
        ("", None),
    ],
)
def test_extract_referenced_issue_number(body: str, expected: int | None) -> None:
    assert extract_referenced_issue_number(body) == expected


def test_format_acknowledgment_with_corrections() -> None:
    text = format_acknowledgment(
        CaptureResult(issue_still_open=False, already_reviewed=False, captured=2)
    )
    assert "Recorded 2 corrections" in text
    assert "reviewed" in text
    assert "Note:" not in text


def test_format_acknowledgment_with_no_corrections() -> None:
    text = format_acknowledgment(
        CaptureResult(issue_still_open=False, already_reviewed=False, captured=0)
    )
    assert "no corrections needed" in text


def test_format_acknowledgment_notes_unattributed_comments() -> None:
    text = format_acknowledgment(
        CaptureResult(
            issue_still_open=False,
            already_reviewed=False,
            captured=1,
            unattributed_comment_ids=[42],
        )
    )
    assert "Note: 1 comment could not be matched" in text


@pytest.fixture
def shadow_client(mocker: Any) -> Any:
    return mocker.Mock()


@pytest.fixture
def connection(mocker: Any) -> Any:
    return mocker.Mock()


def test_capture_corrections_skips_if_already_reviewed(
    mocker: Any, shadow_client: Any, connection: Any
) -> None:
    mocker.patch("src.corrections.is_digest_reviewed", return_value=True)

    result = capture_corrections(
        connection,
        shadow_client,
        digest_id=1,
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        shadow_issue_number=4,
    )

    assert result.already_reviewed is True
    shadow_client.get_issue_state.assert_not_called()
    shadow_client.create_issue_comment.assert_not_called()


def test_capture_corrections_skips_if_issue_still_open(
    mocker: Any, shadow_client: Any, connection: Any
) -> None:
    mocker.patch("src.corrections.is_digest_reviewed", return_value=False)
    shadow_client.get_issue_state.return_value = "open"

    result = capture_corrections(
        connection,
        shadow_client,
        digest_id=1,
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        shadow_issue_number=4,
    )

    assert result.issue_still_open is True
    assert result.captured == 0
    shadow_client.fetch_issue_comments.assert_not_called()
    shadow_client.create_issue_comment.assert_not_called()


def test_capture_corrections_captures_referenced_comments_and_marks_reviewed(
    mocker: Any, shadow_client: Any, connection: Any
) -> None:
    mocker.patch("src.corrections.is_digest_reviewed", return_value=False)
    shadow_client.get_issue_state.return_value = "closed"
    shadow_client.fetch_issue_comments.return_value = [
        _comment(1, "#34649 is not about linear model, it's about SVC"),
        _comment(2, "no reference in this one"),
    ]
    mocker.patch(
        "src.corrections.get_judgment_id_for_issue_number",
        side_effect=lambda conn, digest_id, number: 501 if number == 34649 else None,
    )
    save_correction = mocker.patch("src.corrections.save_correction")
    mark_reviewed = mocker.patch("src.corrections.mark_digest_reviewed")

    result = capture_corrections(
        connection,
        shadow_client,
        digest_id=1,
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        shadow_issue_number=4,
    )

    assert result.captured == 1
    assert result.unattributed_comment_ids == [2]
    save_correction.assert_called_once_with(
        connection,
        501,
        1,
        "#34649 is not about linear model, it's about SVC",
        dt.datetime(2026, 8, 4, 12, 0, 0, tzinfo=dt.UTC),
    )
    mark_reviewed.assert_called_once_with(connection, 1)


def test_capture_corrections_posts_one_acknowledgment_before_marking_reviewed(
    mocker: Any, shadow_client: Any, connection: Any
) -> None:
    mocker.patch("src.corrections.is_digest_reviewed", return_value=False)
    shadow_client.get_issue_state.return_value = "closed"
    shadow_client.fetch_issue_comments.return_value = []
    mark_reviewed = mocker.patch("src.corrections.mark_digest_reviewed")

    manager = mocker.MagicMock()
    manager.attach_mock(shadow_client.create_issue_comment, "create_issue_comment")
    manager.attach_mock(mark_reviewed, "mark_digest_reviewed")

    capture_corrections(
        connection,
        shadow_client,
        digest_id=1,
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        shadow_issue_number=4,
    )

    shadow_client.create_issue_comment.assert_called_once_with(
        "virchan",
        "issue-triaging-agent-digests",
        4,
        "Recorded — no corrections needed for this digest.\n\n"
        "This digest is now marked reviewed.",
    )
    # Posted before the commit, so a failed post never leaves a reviewed
    # digest with no acknowledgment - see corrections.py's comment.
    assert [c[0] for c in manager.mock_calls] == [
        "create_issue_comment",
        "mark_digest_reviewed",
    ]
