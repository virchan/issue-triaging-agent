from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from src.corrections import (
    REJUDGE_CAP,
    AppliedCorrection,
    CaptureResult,
    SupersededCorrection,
    capture_corrections,
    extract_corrections_by_issue,
    format_acknowledgment,
)
from src.db import RejudgeContext
from src.gemini_client import GeminiResponseError, GeminiUnavailableError
from src.github_client import GitHubComment
from src.judgment import IssueJudgment

WINDOW_END = dt.datetime(2026, 8, 17, 0, 0, 0, tzinfo=dt.UTC)


def _comment(comment_id: int, body: str) -> GitHubComment:
    return GitHubComment(
        id=comment_id,
        body=body,
        created_at=dt.datetime(2026, 8, 4, 12, 0, 0, tzinfo=dt.UTC),
    )


def _judgment(**overrides: Any) -> IssueJudgment:
    defaults = {
        "suggested_label": "Bug",
        "is_spam": False,
        "summary": "s",
        "priority": "medium",
        "rationale": "r",
        "confidence": 0.8,
    }
    defaults.update(overrides)
    return IssueJudgment(**defaults)


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


def test_format_acknowledgment_lists_applied_corrections() -> None:
    """Regression test for the real confusion on digest #15: a bare count
    ("Recorded 3 corrections") gave no way to verify which corrections
    were captured without waiting for the next digest. The collapsed
    detail section must show every applied correction's text and, when a
    live re-judge ran, the resulting label."""

    text = format_acknowledgment(
        CaptureResult(
            issue_still_open=False,
            already_reviewed=False,
            captured=2,
            applied=[
                AppliedCorrection(
                    34436, "should include Numerical Stability", "Numerical Stability"
                ),
                AppliedCorrection(34618, "line one\nline two", new_label=None),
            ],
        )
    )
    assert "<details>" in text
    assert "Corrections recorded (2)" in text
    assert "`scikit-learn/scikit-learn/34436`" in text
    assert "should include Numerical Stability" in text
    assert "→ label updated to **Numerical Stability**" in text
    assert "`scikit-learn/scikit-learn/34618`" in text
    assert "line one\nline two" in text
    assert "</details>" in text
    # Entries must be separated by a blank line (a divider, here), not run
    # together in one paragraph - GitHub's renderer needs it to tell them
    # apart as distinct blocks inside the collapsed section.
    assert (
        "**Numerical Stability**\n\n---\n\n**`scikit-learn/scikit-learn/34618`**"
        in text
    )


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


def test_format_acknowledgment_notes_superseded_correction() -> None:
    text = format_acknowledgment(
        CaptureResult(
            issue_still_open=False,
            already_reviewed=False,
            captured=0,
            superseded=[SupersededCorrection(34649, 16)],
        )
    )
    assert "`scikit-learn/scikit-learn/34649`" in text
    assert "#16" in text


def test_format_acknowledgment_notes_capped_corrections() -> None:
    text = format_acknowledgment(
        CaptureResult(
            issue_still_open=False, already_reviewed=False, captured=1, capped=2
        )
    )
    assert "2 corrections were recorded but not re-judged" in text


def test_format_acknowledgment_notes_rejudge_failures() -> None:
    text = format_acknowledgment(
        CaptureResult(
            issue_still_open=False,
            already_reviewed=False,
            captured=1,
            rejudge_failures=[(34649, "boom")],
        )
    )
    assert "1 correction" in text
    assert "re-judge call failed" in text


@pytest.fixture
def shadow_client(mocker: Any) -> Any:
    return mocker.Mock()


@pytest.fixture
def gemini_judge(mocker: Any) -> Any:
    judge = mocker.Mock()
    judge.judge_with_correction.return_value = _judgment(summary="revised")
    return judge


@pytest.fixture
def connection(mocker: Any) -> Any:
    return mocker.Mock()


def _run(
    mocker: Any,
    shadow_client: Any,
    gemini_judge: Any,
    connection: Any,
    **overrides: Any,
) -> Any:
    kwargs = {
        "digest_id": 1,
        "digest_window_end": WINDOW_END,
        "shadow_owner": "virchan",
        "shadow_repo": "issue-triaging-agent-digests",
        "shadow_issue_number": 4,
        "known_labels": ["Bug", "module:decomposition"],
    }
    kwargs.update(overrides)
    return capture_corrections(connection, shadow_client, gemini_judge, **kwargs)


def test_capture_corrections_skips_if_already_reviewed(
    mocker: Any, shadow_client: Any, gemini_judge: Any, connection: Any
) -> None:
    mocker.patch("src.corrections.is_digest_reviewed", return_value=True)

    result = _run(mocker, shadow_client, gemini_judge, connection)

    assert result.already_reviewed is True
    shadow_client.get_issue_state.assert_not_called()
    shadow_client.create_issue_comment.assert_not_called()


def test_capture_corrections_skips_if_issue_still_open(
    mocker: Any, shadow_client: Any, gemini_judge: Any, connection: Any
) -> None:
    mocker.patch("src.corrections.is_digest_reviewed", return_value=False)
    shadow_client.get_issue_state.return_value = "open"

    result = _run(mocker, shadow_client, gemini_judge, connection)

    assert result.issue_still_open is True
    assert result.captured == 0
    shadow_client.fetch_issue_comments.assert_not_called()
    shadow_client.create_issue_comment.assert_not_called()


def test_capture_corrections_captures_and_rejudges_a_new_correction(
    mocker: Any, shadow_client: Any, gemini_judge: Any, connection: Any
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
        side_effect=lambda conn, number: 501 if number == 34649 else None,
    )
    mocker.patch(
        "src.corrections.get_authoritative_correction_digest", return_value=None
    )
    mark_superseded = mocker.patch("src.corrections.mark_corrections_superseded")
    save_correction = mocker.patch("src.corrections.save_correction")
    mocker.patch(
        "src.corrections.get_rejudge_context",
        return_value=RejudgeContext(
            title="OneHotEncoder crashes", body="body", judgment=_judgment()
        ),
    )
    update_judgment = mocker.patch("src.corrections.update_judgment")
    mark_reviewed = mocker.patch("src.corrections.mark_digest_reviewed")

    result = _run(mocker, shadow_client, gemini_judge, connection)

    assert result.captured == 1
    assert result.applied == [
        AppliedCorrection(
            34649,
            "`scikit-learn/scikit-learn/34649` is not about linear model, "
            "it's about SVC",
            new_label="Bug",
        )
    ]
    assert result.unattributed_comment_ids == [2]
    assert result.superseded == []
    mark_superseded.assert_called_once_with(connection, 501)
    save_correction.assert_called_once_with(
        connection,
        501,
        1,
        1,
        "`scikit-learn/scikit-learn/34649` is not about linear model, it's about SVC",
        dt.datetime(2026, 8, 4, 12, 0, 0, tzinfo=dt.UTC),
        superseded=False,
    )
    gemini_judge.judge_with_correction.assert_called_once()
    update_judgment.assert_called_once_with(
        connection, 501, _judgment(summary="revised")
    )
    mark_reviewed.assert_called_once_with(connection, 1)


def test_capture_corrections_produces_one_correction_per_referenced_issue(
    mocker: Any, shadow_client: Any, gemini_judge: Any, connection: Any
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
        side_effect=lambda conn, number: {34436: 501, 34618: 502}.get(number),
    )
    mocker.patch(
        "src.corrections.get_authoritative_correction_digest", return_value=None
    )
    mocker.patch("src.corrections.mark_corrections_superseded")
    save_correction = mocker.patch("src.corrections.save_correction")
    mocker.patch(
        "src.corrections.get_rejudge_context",
        return_value=RejudgeContext(title="t", body="b", judgment=_judgment()),
    )
    mocker.patch("src.corrections.update_judgment")
    mocker.patch("src.corrections.mark_digest_reviewed")

    result = _run(mocker, shadow_client, gemini_judge, connection)

    assert result.captured == 2
    assert {a.github_number for a in result.applied} == {34436, 34618}
    assert all(a.new_label == "Bug" for a in result.applied)
    assert result.unattributed_comment_ids == []
    judgment_ids = {call.args[1] for call in save_correction.call_args_list}
    assert judgment_ids == {501, 502}
    assert gemini_judge.judge_with_correction.call_count == 2


def test_capture_corrections_marks_a_stale_thread_correction_superseded(
    mocker: Any, shadow_client: Any, gemini_judge: Any, connection: Any
) -> None:
    """Case II from the design: a correction on an older thread must not
    override the authoritative correction on a more recently created
    thread - it's recorded but marked superseded, not re-judged."""

    mocker.patch("src.corrections.is_digest_reviewed", return_value=False)
    shadow_client.get_issue_state.return_value = "closed"
    shadow_client.fetch_issue_comments.return_value = [
        _comment(1, "`scikit-learn/scikit-learn/34649` needs a different label"),
    ]
    mocker.patch("src.corrections.get_judgment_id_for_issue_number", return_value=501)
    newer_window_end = WINDOW_END + dt.timedelta(days=1)
    mocker.patch(
        "src.corrections.get_authoritative_correction_digest",
        return_value=(16, newer_window_end),
    )
    mark_superseded = mocker.patch("src.corrections.mark_corrections_superseded")
    save_correction = mocker.patch("src.corrections.save_correction")
    mocker.patch("src.corrections.mark_digest_reviewed")

    result = _run(mocker, shadow_client, gemini_judge, connection)

    assert result.captured == 0
    assert result.superseded == [SupersededCorrection(34649, 16)]
    mark_superseded.assert_not_called()
    save_correction.assert_called_once_with(
        connection,
        501,
        1,
        1,
        "`scikit-learn/scikit-learn/34649` needs a different label",
        dt.datetime(2026, 8, 4, 12, 0, 0, tzinfo=dt.UTC),
        superseded=True,
    )
    gemini_judge.judge_with_correction.assert_not_called()


def test_capture_corrections_own_thread_being_the_most_recent_still_applies(
    mocker: Any, shadow_client: Any, gemini_judge: Any, connection: Any
) -> None:
    """Case I: if the existing authoritative correction is on an *older*
    thread than this one, this thread's correction still applies."""

    mocker.patch("src.corrections.is_digest_reviewed", return_value=False)
    shadow_client.get_issue_state.return_value = "closed"
    shadow_client.fetch_issue_comments.return_value = [
        _comment(1, "`scikit-learn/scikit-learn/34649` needs a different label"),
    ]
    mocker.patch("src.corrections.get_judgment_id_for_issue_number", return_value=501)
    older_window_end = WINDOW_END - dt.timedelta(days=1)
    mocker.patch(
        "src.corrections.get_authoritative_correction_digest",
        return_value=(12, older_window_end),
    )
    mocker.patch("src.corrections.mark_corrections_superseded")
    mocker.patch("src.corrections.save_correction")
    mocker.patch(
        "src.corrections.get_rejudge_context",
        return_value=RejudgeContext(title="t", body="b", judgment=_judgment()),
    )
    mocker.patch("src.corrections.update_judgment")
    mocker.patch("src.corrections.mark_digest_reviewed")

    result = _run(mocker, shadow_client, gemini_judge, connection)

    assert result.captured == 1
    assert result.superseded == []
    gemini_judge.judge_with_correction.assert_called_once()


def test_capture_corrections_respects_the_rejudge_cap(
    mocker: Any, shadow_client: Any, gemini_judge: Any, connection: Any
) -> None:
    comments = [
        _comment(i, f"`scikit-learn/scikit-learn/{34600 + i}` needs a fix")
        for i in range(REJUDGE_CAP + 2)
    ]
    mocker.patch("src.corrections.is_digest_reviewed", return_value=False)
    shadow_client.get_issue_state.return_value = "closed"
    shadow_client.fetch_issue_comments.return_value = comments
    mocker.patch(
        "src.corrections.get_judgment_id_for_issue_number",
        side_effect=lambda conn, number: number,
    )
    mocker.patch(
        "src.corrections.get_authoritative_correction_digest", return_value=None
    )
    mocker.patch("src.corrections.mark_corrections_superseded")
    mocker.patch("src.corrections.save_correction")
    mocker.patch(
        "src.corrections.get_rejudge_context",
        return_value=RejudgeContext(title="t", body="b", judgment=_judgment()),
    )
    mocker.patch("src.corrections.update_judgment")
    mocker.patch("src.corrections.mark_digest_reviewed")

    result = _run(mocker, shadow_client, gemini_judge, connection)

    assert result.captured == REJUDGE_CAP + 2
    assert result.capped == 2
    assert gemini_judge.judge_with_correction.call_count == REJUDGE_CAP
    # Every captured correction is still surfaced, even the capped ones -
    # they just carry no new_label since no re-judge ran for them.
    assert len(result.applied) == REJUDGE_CAP + 2
    assert [a.new_label for a in result.applied[:REJUDGE_CAP]] == ["Bug"] * REJUDGE_CAP
    assert [a.new_label for a in result.applied[REJUDGE_CAP:]] == [None, None]


def test_capture_corrections_records_a_rejudge_failure(
    mocker: Any, shadow_client: Any, gemini_judge: Any, connection: Any
) -> None:
    mocker.patch("src.corrections.is_digest_reviewed", return_value=False)
    shadow_client.get_issue_state.return_value = "closed"
    shadow_client.fetch_issue_comments.return_value = [
        _comment(1, "`scikit-learn/scikit-learn/34649` needs a fix"),
    ]
    mocker.patch("src.corrections.get_judgment_id_for_issue_number", return_value=501)
    mocker.patch(
        "src.corrections.get_authoritative_correction_digest", return_value=None
    )
    mocker.patch("src.corrections.mark_corrections_superseded")
    mocker.patch("src.corrections.save_correction")
    mocker.patch(
        "src.corrections.get_rejudge_context",
        return_value=RejudgeContext(title="t", body="b", judgment=_judgment()),
    )
    update_judgment = mocker.patch("src.corrections.update_judgment")
    mocker.patch("src.corrections.mark_digest_reviewed")
    gemini_judge.judge_with_correction.side_effect = GeminiUnavailableError("down")

    result = _run(mocker, shadow_client, gemini_judge, connection)

    assert result.captured == 1
    assert result.rejudge_failures == [(34649, "down")]
    assert result.applied == [
        AppliedCorrection(34649, "`scikit-learn/scikit-learn/34649` needs a fix")
    ]
    update_judgment.assert_not_called()


def test_capture_corrections_translates_response_error_to_failure(
    mocker: Any, shadow_client: Any, gemini_judge: Any, connection: Any
) -> None:
    mocker.patch("src.corrections.is_digest_reviewed", return_value=False)
    shadow_client.get_issue_state.return_value = "closed"
    shadow_client.fetch_issue_comments.return_value = [
        _comment(1, "`scikit-learn/scikit-learn/34649` needs a fix"),
    ]
    mocker.patch("src.corrections.get_judgment_id_for_issue_number", return_value=501)
    mocker.patch(
        "src.corrections.get_authoritative_correction_digest", return_value=None
    )
    mocker.patch("src.corrections.mark_corrections_superseded")
    mocker.patch("src.corrections.save_correction")
    mocker.patch(
        "src.corrections.get_rejudge_context",
        return_value=RejudgeContext(title="t", body="b", judgment=_judgment()),
    )
    mocker.patch("src.corrections.update_judgment")
    mocker.patch("src.corrections.mark_digest_reviewed")
    gemini_judge.judge_with_correction.side_effect = GeminiResponseError("bad")

    result = _run(mocker, shadow_client, gemini_judge, connection)

    assert result.rejudge_failures == [(34649, "bad")]


def test_capture_corrections_posts_one_acknowledgment_before_marking_reviewed(
    mocker: Any, shadow_client: Any, gemini_judge: Any, connection: Any
) -> None:
    mocker.patch("src.corrections.is_digest_reviewed", return_value=False)
    shadow_client.get_issue_state.return_value = "closed"
    shadow_client.fetch_issue_comments.return_value = []
    mark_reviewed = mocker.patch("src.corrections.mark_digest_reviewed")

    manager = mocker.MagicMock()
    manager.attach_mock(shadow_client.create_issue_comment, "create_issue_comment")
    manager.attach_mock(mark_reviewed, "mark_digest_reviewed")

    _run(mocker, shadow_client, gemini_judge, connection)

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
