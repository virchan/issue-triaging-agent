from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Any

import psycopg

from src.github_client import GitHubIssue
from src.judgment import IssueJudgment


def connect() -> psycopg.Connection[Any]:
    """Open a connection using DATABASE_URL from the environment."""

    return psycopg.connect(os.environ["DATABASE_URL"])


def save_issue_snapshot(
    connection: psycopg.Connection[Any],
    repo_owner: str,
    repo_name: str,
    issue: GitHubIssue,
    is_bot: bool,
) -> int:
    """Insert an issue snapshot, or return the existing row's id if already stored.

    Idempotent on (repo_owner, repo_name, github_number): re-fetching the
    same issue on a later run (e.g. after a retry) does not create a
    duplicate row or clobber a snapshot a judgment may already reference.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO issues (
                repo_owner, repo_name, github_number, title, body,
                author_login, github_created_at, html_url, is_bot
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (repo_owner, repo_name, github_number) DO NOTHING
            RETURNING id
            """,
            (
                repo_owner,
                repo_name,
                issue.number,
                issue.title,
                issue.body,
                issue.author_login,
                issue.created_at,
                issue.html_url,
                is_bot,
            ),
        )
        row = cursor.fetchone()
        if row is not None:
            return row[0]

        cursor.execute(
            """
            SELECT id FROM issues
            WHERE repo_owner = %s AND repo_name = %s AND github_number = %s
            """,
            (repo_owner, repo_name, issue.number),
        )
        existing = cursor.fetchone()
        assert existing is not None
        return existing[0]


def save_issue_snapshots(
    connection: psycopg.Connection[Any],
    repo_owner: str,
    repo_name: str,
    non_bot_issues: list[GitHubIssue],
    bot_issues: list[GitHubIssue],
) -> dict[int, int]:
    """Store both partitions from bot_filter.partition_bot_issues.

    Bot issues are stored too (tagged is_bot=True), not discarded, so
    there's an audit trail of what the filter excluded and why.

    Returns a mapping of GitHub issue number -> database row id, for all
    stored issues (both partitions), so callers (e.g. the judgment
    pipeline) can look up an issue's id without relying on result
    ordering.
    """

    ids_by_number: dict[int, int] = {
        issue.number: save_issue_snapshot(
            connection, repo_owner, repo_name, issue, is_bot=False
        )
        for issue in non_bot_issues
    }
    ids_by_number.update(
        {
            issue.number: save_issue_snapshot(
                connection, repo_owner, repo_name, issue, is_bot=True
            )
            for issue in bot_issues
        }
    )
    connection.commit()
    return ids_by_number


def has_judgment(connection: psycopg.Connection[Any], issue_id: int) -> bool:
    """Whether an issue already has a stored judgment.

    Checked before calling the LLM (not just before writing to the DB),
    so a retry after a partial failure doesn't waste API calls
    re-judging issues that already succeeded.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM judgments WHERE issue_id = %s",
            (issue_id,),
        )
        return cursor.fetchone() is not None


def save_judgment(
    connection: psycopg.Connection[Any],
    issue_id: int,
    judgment: IssueJudgment,
    digest_id: int | None = None,
) -> int:
    """Insert a judgment, or return the existing row's id if already judged.

    Idempotent on issue_id, mirroring save_issue_snapshot: a retry after
    a partial pipeline failure does not create a duplicate or overwrite
    an existing judgment. Re-judging an issue on purpose is not yet a
    supported operation - no real need for it has come up yet.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO judgments (
                issue_id, digest_id, suggested_label, is_spam,
                summary, priority, rationale, confidence
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (issue_id) DO NOTHING
            RETURNING id
            """,
            (
                issue_id,
                digest_id,
                judgment.suggested_label,
                judgment.is_spam,
                judgment.summary,
                judgment.priority,
                judgment.rationale,
                judgment.confidence,
            ),
        )
        row = cursor.fetchone()
        if row is not None:
            return row[0]

        cursor.execute(
            "SELECT id FROM judgments WHERE issue_id = %s",
            (issue_id,),
        )
        existing = cursor.fetchone()
        assert existing is not None
        return existing[0]


@dataclass
class JudgedIssue:
    """A judged, non-bot issue - the result of joining issues and judgments."""

    issue_id: int
    judgment_id: int
    github_number: int
    title: str
    html_url: str
    judgment: IssueJudgment


def get_judged_issues_in_window(
    connection: psycopg.Connection[Any],
    repo_owner: str,
    repo_name: str,
    window_start: dt.datetime,
    window_end: dt.datetime,
) -> list[JudgedIssue]:
    """Fetch non-bot issues created in [window_start, window_end) that
    already have a judgment."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.id, j.id, i.github_number, i.title, i.html_url,
                   j.suggested_label, j.is_spam, j.summary, j.priority,
                   j.rationale, j.confidence
            FROM issues i
            JOIN judgments j ON j.issue_id = i.id
            WHERE i.repo_owner = %s AND i.repo_name = %s
              AND i.github_created_at >= %s AND i.github_created_at < %s
              AND i.is_bot = FALSE
            ORDER BY i.github_number
            """,
            (repo_owner, repo_name, window_start, window_end),
        )
        rows = cursor.fetchall()

    return [
        JudgedIssue(
            issue_id=row[0],
            judgment_id=row[1],
            github_number=row[2],
            title=row[3],
            html_url=row[4],
            judgment=IssueJudgment(
                suggested_label=row[5],
                is_spam=row[6],
                summary=row[7],
                priority=row[8],
                rationale=row[9],
                confidence=row[10],
            ),
        )
        for row in rows
    ]


def create_digest(
    connection: psycopg.Connection[Any],
    shadow_repo_owner: str,
    shadow_repo_name: str,
    window_start: dt.datetime,
    window_end: dt.datetime,
) -> int:
    """Create a new digest row for [window_start, window_end).

    No uniqueness constraint on the window: each run's window_start is
    derived from the previous digest's window_end (see
    get_latest_digest_window_end), so windows chain together and don't
    naturally collide. A double-invocation just produces a second,
    narrower window rather than a duplicate - has_judgment already
    prevents re-judging any issue that window happens to re-fetch.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO digests (window_start, window_end, shadow_repo_owner, shadow_repo_name)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (window_start, window_end, shadow_repo_owner, shadow_repo_name),
        )
        row = cursor.fetchone()
        assert row is not None
        return row[0]


def get_latest_digest_window_end(
    connection: psycopg.Connection[Any],
) -> dt.datetime | None:
    """The most recent digest's window_end, or None if no digest exists yet.

    The watermark the next poll's window_start is derived from - see
    LOG.md entry 53. None only for the very first run ever, before any
    digest has been created.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT window_end FROM digests ORDER BY window_end DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row[0] if row is not None else None


def link_judgments_to_digest(
    connection: psycopg.Connection[Any],
    digest_id: int,
    judgment_ids: list[int],
) -> None:
    """Attach a set of judgments to a digest."""

    if not judgment_ids:
        return

    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE judgments SET digest_id = %s WHERE id = ANY(%s)",
            (digest_id, judgment_ids),
        )
    connection.commit()


def is_digest_published(connection: psycopg.Connection[Any], digest_id: int) -> bool:
    """Whether a digest has already been posted to the shadow repo.

    Checked before publishing, mirroring has_judgment: a retry after a
    partial failure should not post a duplicate issue.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT shadow_issue_number FROM digests WHERE id = %s",
            (digest_id,),
        )
        row = cursor.fetchone()
        return row is not None and row[0] is not None


def mark_digest_published(
    connection: psycopg.Connection[Any],
    digest_id: int,
    shadow_issue_number: int,
) -> None:
    """Record that a digest was posted, and which shadow-repo issue it became."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE digests
            SET shadow_issue_number = %s, state = 'published', published_at = now()
            WHERE id = %s
            """,
            (shadow_issue_number, digest_id),
        )
    connection.commit()


def is_digest_reviewed(connection: psycopg.Connection[Any], digest_id: int) -> bool:
    """Whether a digest's corrections have already been captured."""

    with connection.cursor() as cursor:
        cursor.execute("SELECT state FROM digests WHERE id = %s", (digest_id,))
        row = cursor.fetchone()
        return row is not None and row[0] == "reviewed"


def mark_digest_reviewed(connection: psycopg.Connection[Any], digest_id: int) -> None:
    """Record that a digest's issue was closed and its corrections captured."""

    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE digests SET state = 'reviewed', closed_at = now() WHERE id = %s",
            (digest_id,),
        )
    connection.commit()


def get_judgment_id_for_issue_number(
    connection: psycopg.Connection[Any],
    digest_id: int,
    github_number: int,
) -> int | None:
    """Look up the judgment, within a specific digest, for a referenced issue number."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT j.id FROM judgments j
            JOIN issues i ON i.id = j.issue_id
            WHERE j.digest_id = %s AND i.github_number = %s
            """,
            (digest_id, github_number),
        )
        row = cursor.fetchone()
        return row[0] if row is not None else None


def save_correction(
    connection: psycopg.Connection[Any],
    judgment_id: int,
    github_comment_id: int,
    comment_body: str,
    github_created_at: dt.datetime,
) -> int:
    """Insert a correction, or return the existing row's id if already captured.

    Idempotent on github_comment_id.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO corrections (
                judgment_id, github_comment_id, comment_body, github_created_at
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (github_comment_id) DO NOTHING
            RETURNING id
            """,
            (judgment_id, github_comment_id, comment_body, github_created_at),
        )
        row = cursor.fetchone()
        if row is not None:
            return row[0]

        cursor.execute(
            "SELECT id FROM corrections WHERE github_comment_id = %s",
            (github_comment_id,),
        )
        existing = cursor.fetchone()
        assert existing is not None
        return existing[0]


@dataclass
class ReviewedJudgment:
    """A past judgment whose digest has been reviewed - either explicitly
    corrected (correction_text set) or implicitly confirmed correct (the
    digest was reviewed and no correction exists for this judgment).
    """

    issue_title: str
    issue_body: str | None
    judgment: IssueJudgment
    correction_text: str | None


def get_recent_reviewed_judgments(
    connection: psycopg.Connection[Any],
    limit: int = 10,
) -> list[ReviewedJudgment]:
    """Fetch the most recent reviewed judgments, for use as few-shot context.

    Includes both corrected and implicitly-confirmed judgments (see
    ReviewedJudgment) - both are real signal about past accuracy, not
    just corrections. Ordered most-recent-first by digest date.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.title, i.body, j.suggested_label, j.is_spam, j.summary,
                   j.priority, j.rationale, j.confidence, c.comment_body
            FROM judgments j
            JOIN issues i ON i.id = j.issue_id
            JOIN digests d ON d.id = j.digest_id
            LEFT JOIN corrections c ON c.judgment_id = j.id
            WHERE d.state = 'reviewed'
            ORDER BY d.window_end DESC, j.id DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    return [
        ReviewedJudgment(
            issue_title=row[0],
            issue_body=row[1],
            judgment=IssueJudgment(
                suggested_label=row[2],
                is_spam=row[3],
                summary=row[4],
                priority=row[5],
                rationale=row[6],
                confidence=row[7],
            ),
            correction_text=row[8],
        )
        for row in rows
    ]


@dataclass
class GoldenExample:
    """One real, reviewed judgment, traceable back to its source issue and
    digest - the unit the golden evaluation set (Step 22) is built from.

    Distinct from ReviewedJudgment (few-shot context, recent-N, no
    traceability fields needed) even though the underlying data
    overlaps - these two serve different purposes and are expected to
    diverge further once Step 23 defines the correctness rubric.
    """

    github_number: int
    issue_title: str
    issue_body: str | None
    judgment: IssueJudgment
    correction_text: str | None
    digest_date: dt.date
    """Pacific calendar date the digest's window ended on - a display
    label derived from window_end, not part of the digest's identity.
    See LOG.md entry 53."""


def get_all_reviewed_judgments(
    connection: psycopg.Connection[Any],
) -> list[GoldenExample]:
    """Fetch every reviewed judgment ever, for exporting the golden
    evaluation set (see scripts/export_golden_set.py). Unlike
    get_recent_reviewed_judgments, this is unbounded - the golden set is
    meant to grow with all real triage history, not just a recent
    window.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.github_number, i.title, i.body, j.suggested_label,
                   j.is_spam, j.summary, j.priority, j.rationale,
                   j.confidence, c.comment_body,
                   (d.window_end AT TIME ZONE 'America/Los_Angeles')::date
            FROM judgments j
            JOIN issues i ON i.id = j.issue_id
            JOIN digests d ON d.id = j.digest_id
            LEFT JOIN corrections c ON c.judgment_id = j.id
            WHERE d.state = 'reviewed'
            ORDER BY d.window_end ASC, j.id ASC
            """
        )
        rows = cursor.fetchall()

    return [
        GoldenExample(
            github_number=row[0],
            issue_title=row[1],
            issue_body=row[2],
            judgment=IssueJudgment(
                suggested_label=row[3],
                is_spam=row[4],
                summary=row[5],
                priority=row[6],
                rationale=row[7],
                confidence=row[8],
            ),
            correction_text=row[9],
            digest_date=row[10],
        )
        for row in rows
    ]


@dataclass
class JudgmentAuditEntry:
    """One judgment's full status, for operational visibility (/judgments)."""

    github_number: int
    title: str
    suggested_label: str | None
    is_spam: bool
    priority: str
    confidence: float
    digest_date: dt.date | None
    """Pacific calendar date the digest's window ended on - a display
    label derived from window_end, not part of the digest's identity.
    See LOG.md entry 53."""
    digest_state: str | None
    correction_text: str | None


def get_judgment_audit_trail(
    connection: psycopg.Connection[Any],
    limit: int = 50,
) -> list[JudgmentAuditEntry]:
    """Fetch recent judgments (all of them, not just reviewed ones) with
    their digest and correction status, most recent first - the read
    audit trail behind the FastAPI service's /judgments endpoint.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.github_number, i.title, j.suggested_label, j.is_spam,
                   j.priority, j.confidence,
                   (d.window_end AT TIME ZONE 'America/Los_Angeles')::date,
                   d.state, c.comment_body
            FROM judgments j
            JOIN issues i ON i.id = j.issue_id
            LEFT JOIN digests d ON d.id = j.digest_id
            LEFT JOIN corrections c ON c.judgment_id = j.id
            ORDER BY j.id DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    return [
        JudgmentAuditEntry(
            github_number=row[0],
            title=row[1],
            suggested_label=row[2],
            is_spam=row[3],
            priority=row[4],
            confidence=row[5],
            digest_date=row[6],
            digest_state=row[7],
            correction_text=row[8],
        )
        for row in rows
    ]


@dataclass
class UnreviewedDigest:
    """A published digest not yet marked reviewed - a candidate to check
    for correction capture (the issue may or may not be closed yet)."""

    digest_id: int
    shadow_owner: str
    shadow_repo: str
    shadow_issue_number: int


def get_unreviewed_digests(
    connection: psycopg.Connection[Any],
) -> list[UnreviewedDigest]:
    """Fetch all published-but-not-yet-reviewed digests, oldest first."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, shadow_repo_owner, shadow_repo_name, shadow_issue_number
            FROM digests
            WHERE state = 'published'
            ORDER BY window_end ASC
            """
        )
        rows = cursor.fetchall()

    return [
        UnreviewedDigest(
            digest_id=row[0],
            shadow_owner=row[1],
            shadow_repo=row[2],
            shadow_issue_number=row[3],
        )
        for row in rows
    ]
