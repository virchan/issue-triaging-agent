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
        repo_owner="scikit-learn",
        repo_name="scikit-learn",
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
    """Regression test for a real gap found during real operation: the
    message previously implied no activity at all, when the query is
    actually scoped to one label."""

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
    """Regression test for a real cross-reference incident.

    A markdown link whose target is a real github.com issue/PR URL
    creates a visible GitHub cross-reference on that issue - confirmed
    empirically. The heading reference is a real, clickable link, but its
    target must be the redirect.github.com form (confirmed empirically
    NOT to create a cross-reference - see LOG.md entry 67), never a raw
    github.com URL. Nor may a bare "#NNN" or "owner/repo#NNN" appear
    anywhere. The separate "Link:" field still carries the real URL, but
    backtick-wrapped (inert), so it can't autolink either.
    """

    body = format_digest_body(dt.date(2026, 8, 4), [_judged_issue(34649)])

    assert "](https://github.com/scikit-learn" not in body
    assert (
        "[<code>scikit-learn/34649</code>]"
        "(https://redirect.github.com/scikit-learn/scikit-learn/issues/34649)"
    ) in body
    assert "`https://github.com/scikit-learn/scikit-learn/issues/34649`" in body
    assert "scikit-learn#34649" not in body


def test_format_digest_body_renders_backlog_only_section() -> None:
    """Backlog catch-up: when nothing new was found, older backlog
    issues get reviewed and shown instead - as a clearly distinct
    section, not silently merged with "new"."""

    body = format_digest_body(
        dt.date(2026, 8, 15),
        [],
        label="Needs Triage",
        backlog_issues=[_judged_issue(1)],
    )

    assert 'No newly created issue(s) labelled "Needs Triage" were found' in body
    assert "1 older open issue(s) that still need triaging" in body
    assert "<code>scikit-learn/1</code>" in body


def test_format_digest_body_renders_combined_new_and_backlog_sections() -> None:
    body = format_digest_body(
        dt.date(2026, 8, 15),
        [_judged_issue(1)],
        label="Needs Triage",
        backlog_issues=[_judged_issue(2)],
    )

    assert "<code>scikit-learn/1</code>" in body
    assert "<code>scikit-learn/2</code>" in body
    assert "1 older open issue(s) that still need triaging too" in body


def test_format_digest_body_omits_backlog_section_when_none_given() -> None:
    body = format_digest_body(dt.date(2026, 8, 4), [_judged_issue(1)])
    assert "older open issue(s)" not in body


def test_format_digest_body_omits_wip_reminder_by_default() -> None:
    body = format_digest_body(dt.date(2026, 8, 4), [_judged_issue(1)])
    assert "Still working on" not in body


def test_format_digest_body_includes_wip_reminder_when_given() -> None:
    """A still-open prior digest gets a reminder line pointing back at
    it, using a real clickable "#NNN" - unlike scikit-learn references,
    this points at another issue in the *same* repo, where autolinking
    is desired."""

    body = format_digest_body(
        dt.date(2026, 8, 17), [_judged_issue(1)], wip_digest_issue_number=12
    )

    assert "Still working on #12?" in body


def test_format_digest_body_includes_wip_reminder_even_when_fully_empty() -> None:
    body = format_digest_body(dt.date(2026, 8, 17), [], wip_digest_issue_number=12)

    assert "Still working on #12?" in body
    assert "No newly created" in body


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
    assert content.labels == []


def test_build_digest_passes_through_labels(mocker: Any, connection: Any) -> None:
    window_start = dt.datetime(2026, 8, 3, 20, 0, 0, tzinfo=dt.UTC)
    window_end = dt.datetime(2026, 8, 4, 20, 0, 0, tzinfo=dt.UTC)

    mocker.patch("src.digest.get_judged_issues_in_window", return_value=[])
    mocker.patch("src.digest.create_digest", return_value=7)
    mocker.patch("src.digest.link_judgments_to_digest")

    content = build_digest(
        connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        window_start=window_start,
        window_end=window_end,
        label="Needs Triage",
        labels=["daily digest", "manually-triggered"],
    )

    assert content.labels == ["daily digest", "manually-triggered"]


def test_build_digest_includes_backlog_issues(mocker: Any, connection: Any) -> None:
    window_start = dt.datetime(2026, 8, 14, 0, 0, 0, tzinfo=dt.UTC)
    window_end = dt.datetime(2026, 8, 15, 0, 0, 0, tzinfo=dt.UTC)

    mocker.patch("src.digest.get_judged_issues_in_window", return_value=[])
    backlog_issues = [_judged_issue(99, judgment_id=901)]
    get_by_numbers = mocker.patch(
        "src.digest.get_judged_issues_by_numbers", return_value=backlog_issues
    )
    mocker.patch("src.digest.create_digest", return_value=8)
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
        backlog_issue_numbers=[99],
    )

    assert content.issue_count == 1
    assert "<code>scikit-learn/99</code>" in content.body
    get_by_numbers.assert_called_once_with(
        connection, "scikit-learn", "scikit-learn", [99]
    )
    link.assert_called_once_with(connection, 8, [901])


def test_build_digest_passes_through_wip_reminder(mocker: Any, connection: Any) -> None:
    window_start = dt.datetime(2026, 8, 16, 12, 0, 0, tzinfo=dt.UTC)
    window_end = dt.datetime(2026, 8, 17, 0, 0, 0, tzinfo=dt.UTC)

    mocker.patch("src.digest.get_judged_issues_in_window", return_value=[])
    mocker.patch("src.digest.create_digest", return_value=9)
    mocker.patch("src.digest.link_judgments_to_digest")

    content = build_digest(
        connection,
        source_owner="scikit-learn",
        source_repo="scikit-learn",
        shadow_owner="virchan",
        shadow_repo="issue-triaging-agent-digests",
        window_start=window_start,
        window_end=window_end,
        label="Needs Triage",
        wip_digest_issue_number=12,
    )

    assert "Still working on #12?" in content.body


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
    digest = DigestContent(
        digest_id=7, title="t", body="b", issue_count=1, labels=["daily digest"]
    )

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
        "virchan", "issue-triaging-agent-digests", "t", "b", labels=["daily digest"]
    )
    mark_published.assert_called_once_with(connection, 7, 5)

    records = [
        r for r in caplog.records if getattr(r, "event", None) == "publish_outcome"
    ]
    assert len(records) == 1
    assert records[0].outcome == "published"
    assert records[0].issue_number == 5
    assert records[0].issue_count == 1
