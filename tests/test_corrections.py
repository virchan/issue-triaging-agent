from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from src.corrections import (
    CaptureResult,
    capture_corrections,
    extract_corrections_by_issue,
    format_acknowledgment,
)
from src.github_client import GitHubComment


def _comment(comment_id: int, body: str) -> GitHubComment:
    return GitHubComment(
        id=comment_id,
        body=body,
        created_at=dt.datetime(2026, 8, 4, 12, 0, 0, tzinfo=dt.UTC),
    )


def test_extract_corrections_by_issue_single_reference() -> None:
    body = "`scikit-learn/scikit-learn/34649` is not about linear model, it's about SVC"
    assert extract_corrections_by_issue(body) == {34649: body}


def test_extract_corrections_by_issue_no_reference() -> None:
    assert extract_corrections_by_issue("no reference here at all") == {}
    assert extract_corrections_by_issue("") == {}


def test_extract_corrections_by_issue_multiple_lines_multiple_issues() -> None:
    """Regression test: one comment correcting several issues (one bullet
    per issue) must produce one correction per issue, not just the first
    one found or the whole comment glued to a single judgment."""

    body = (
        "* `scikit-learn/scikit-learn/34436` should include the "
        '"Numerical Stability" label.\n'
        "* `scikit-learn/scikit-learn/34618` should not include the "
        '"Performance" label.\n'
        "Otherwise, LGTM."
    )

    result = extract_corrections_by_issue(body)

    assert set(result) == {34436, 34618}
    assert "Numerical Stability" in result[34436]
    assert "Performance" in result[34618]


def test_extract_corrections_by_issue_combines_repeated_references() -> None:
    """Two lines about the same issue in one comment are combined into a
    single correction, not two separate ones (which the DB would reject
    as a duplicate (comment, judgment) pair anyway)."""

    body = (
        "* `scikit-learn/scikit-learn/34618` should not include the "
        '"Performance" label.\n'
        "* `scikit-learn/scikit-learn/34618` should include the "
        '"module:decomposition" label.'
    )

    result = extract_corrections_by_issue(body)

    assert set(result) == {34618}
    assert "Performance" in result[34618]
    assert "module:decomposition" in result[34618]


def test_extract_corrections_by_issue_ignores_a_bare_hash_number() -> None:
    """A bare "#NNN" (e.g. referencing this repo's own issue number, not
    a scikit-learn issue) must not be mistaken for an owner/repo/number
    reference - a real correction was silently dropped this way once,
    when a stray "#13" elsewhere in the comment matched instead of the
    intended reference."""

    body = "I thought I already closed #13 before this issue was created."

    assert extract_corrections_by_issue(body) == {}


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
        _comment(
            1,
            "`scikit-learn/scikit-learn/34649` is not about linear model, "
            "it's about SVC",
        ),
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
        "`scikit-learn/scikit-learn/34649` is not about linear model, it's about SVC",
        dt.datetime(2026, 8, 4, 12, 0, 0, tzinfo=dt.UTC),
    )
    mark_reviewed.assert_called_once_with(connection, 1)


def test_capture_corrections_produces_one_correction_per_referenced_issue(
    mocker: Any, shadow_client: Any, connection: Any
) -> None:
    """Regression test: a single comment correcting several issues (one
    bullet per issue) must save one correction per issue - not one
    correction for the whole comment attributed to just the first."""

    mocker.patch("src.corrections.is_digest_reviewed", return_value=False)
    shadow_client.get_issue_state.return_value = "closed"
    shadow_client.fetch_issue_comments.return_value = [
        _comment(
            1,
            "* `scikit-learn/scikit-learn/34436` should include Numerical "
            "Stability.\n"
            "* `scikit-learn/scikit-learn/34618` should not include "
            "Performance.",
        ),
    ]
    mocker.patch(
        "src.corrections.get_judgment_id_for_issue_number",
        side_effect=lambda conn, digest_id, number: {34436: 501, 34618: 502}.get(
            number
        ),
    )
    save_correction = mocker.patch("src.corrections.save_correction")
    mocker.patch("src.corrections.mark_digest_reviewed")

    result = capture_corrections(
        connection,
        shadow_client,
        digest_id=1,
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        shadow_issue_number=4,
    )

    assert result.captured == 2
    assert result.unattributed_comment_ids == []
    judgment_ids = {call.args[1] for call in save_correction.call_args_list}
    assert judgment_ids == {501, 502}


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
