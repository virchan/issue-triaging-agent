from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from src.db import JudgedIssue
from src.digest import (
    DigestContent,
    build_digest,
    format_digest_body,
    format_digest_title,
    publish_digest,
)
from src.judgment import IssueJudgment


def _judged_issue(
    number: int,
    priority: str = "medium",
    is_spam: bool = False,
    judgment_id: int = 1,
) -> JudgedIssue:
    return JudgedIssue(
        issue_id=number,
        judgment_id=judgment_id,
        github_number=number,
        title=f"Issue {number}",
        html_url=f"https://github.com/scikit-learn/scikit-learn/issues/{number}",
        judgment=IssueJudgment(
            suggested_label="Bug",
            is_spam=is_spam,
            summary="A summary.",
            priority=priority,
            rationale="A rationale.",
            confidence=0.9,
        ),
    )


def test_format_digest_title() -> None:
    assert format_digest_title(dt.date(2026, 8, 4)) == "Triage digest — 2026-08-04"


def test_format_digest_body_handles_empty_day_without_label() -> None:
    body = format_digest_body(dt.date(2026, 8, 4), [])
    assert "No newly created non-bot issue(s) were found for 2026-08-04" in body


def test_format_digest_body_names_the_label_when_given() -> None:
    """Regression test for a real gap found during real operation (see
    LOG.md, daily-log.md): the message previously implied no activity at
    all, when the query is actually scoped to one label."""

    body = format_digest_body(dt.date(2026, 8, 4), [], label="Needs Triage")
    assert 'No newly created issue(s) labelled "Needs Triage" were found' in body
    assert "non-bot" not in body


def test_format_digest_body_summary_line_names_the_label_when_given() -> None:
    body = format_digest_body(
        dt.date(2026, 8, 4), [_judged_issue(1)], label="Needs Triage"
    )
    assert '1 issue(s) labelled "Needs Triage" reviewed for 2026-08-04' in body


def test_format_digest_body_groups_by_priority_present_only() -> None:
    issues = [
        _judged_issue(1, priority="low"),
        _judged_issue(2, priority="high"),
    ]
    body = format_digest_body(dt.date(2026, 8, 4), issues)

    assert "## High priority" in body
    assert "## Low priority" in body
    assert "## Medium priority" not in body
    assert body.index("## High priority") < body.index("## Low priority")


def test_format_digest_body_marks_spam_visibly() -> None:
    body = format_digest_body(dt.date(2026, 8, 4), [_judged_issue(1, is_spam=True)])
    assert "⚠️ possible spam" in body


def test_format_digest_body_never_creates_a_clickable_cross_reference() -> None:
    """Regression test for the real cross-reference incident (see LOG.md).

    A markdown link with the raw scikit-learn URL as its target creates a
    visible GitHub cross-reference on the target issue - confirmed
    empirically. This must never reappear.
    """

    body = format_digest_body(dt.date(2026, 8, 4), [_judged_issue(34649)])

    assert "](https://github.com/scikit-learn" not in body
    assert "`https://github.com/scikit-learn/scikit-learn/issues/34649`" in body
    assert "#34649" in body


@pytest.fixture
def connection(mocker: Any) -> Any:
    return mocker.Mock()


def test_build_digest_aggregates_and_persists(mocker: Any, connection: Any) -> None:
    window_start = dt.datetime(2026, 8, 3, 20, 0, 0, tzinfo=dt.UTC)
    window_end = dt.datetime(2026, 8, 4, 20, 0, 0, tzinfo=dt.UTC)  # 13:00 PDT Aug 4

    issues = [_judged_issue(1, judgment_id=501)]
    mocker.patch("src.digest.get_judged_issues_in_window", return_value=issues)
    create_digest = mocker.patch("src.digest.create_digest", return_value=7)
    link = mocker.patch("src.digest.link_judgments_to_digest")

    content = build_digest(
        connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        window_start=window_start,
        window_end=window_end,
        label="Needs Triage",
    )

    assert content.digest_id == 7
    assert content.issue_count == 1
    assert content.title == "Triage digest — 2026-08-04"
    assert 'labelled "Needs Triage"' in content.body
    create_digest.assert_called_once_with(
        connection, "virchan", "issue-triaging-agent-digests", window_start, window_end
    )
    link.assert_called_once_with(connection, 7, [501])


def test_publish_digest_skips_if_already_published(
    mocker: Any, connection: Any, caplog: Any
) -> None:
    mocker.patch("src.digest.is_digest_published", return_value=True)
    shadow_client = mocker.Mock()
    digest = DigestContent(digest_id=7, title="t", body="b", issue_count=1)

    with caplog.at_level("INFO"):
        result = publish_digest(
            connection,
            shadow_client,
            digest,
            shadow_owner="virchan",
            shadow_repo="issue-triaging-agent-digests",
        )

    assert result is None
    shadow_client.create_issue.assert_not_called()

    records = [
        r for r in caplog.records if getattr(r, "event", None) == "publish_outcome"
    ]
    assert len(records) == 1
    assert records[0].outcome == "already_published"
    assert records[0].digest_id == 7


def test_publish_digest_creates_issue_and_marks_published(
    mocker: Any, connection: Any, caplog: Any
) -> None:
    mocker.patch("src.digest.is_digest_published", return_value=False)
    mark_published = mocker.patch("src.digest.mark_digest_published")
    shadow_client = mocker.Mock()
    shadow_client.create_issue.return_value = (5, "https://example.com/issues/5")
    digest = DigestContent(digest_id=7, title="t", body="b", issue_count=1)

    with caplog.at_level("INFO"):
        result = publish_digest(
            connection,
            shadow_client,
            digest,
            shadow_owner="virchan",
            shadow_repo="issue-triaging-agent-digests",
        )

    assert result == (5, "https://example.com/issues/5")
    shadow_client.create_issue.assert_called_once_with(
        "virchan", "issue-triaging-agent-digests", "t", "b"
    )
    mark_published.assert_called_once_with(connection, 7, 5)

    records = [
        r for r in caplog.records if getattr(r, "event", None) == "publish_outcome"
    ]
    assert len(records) == 1
    assert records[0].outcome == "published"
    assert records[0].issue_number == 5
    assert records[0].issue_count == 1
