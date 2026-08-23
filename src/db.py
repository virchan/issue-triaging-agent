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


def get_issue_embedding(
    connection: psycopg.Connection[Any], issue_id: int
) -> list[float] | None:
    """The stored embedding for one issue, if it has one - checked
    before computing a new one (see src.duplicate_detection), so an
    issue already covered by the backfill or a prior judgment never
    gets re-embedded (an avoidable API call) just because it's being
    judged for real now."""

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT embedding FROM issue_embeddings WHERE issue_id = %s",
            (issue_id,),
        )
        row = cursor.fetchone()
        return list(row[0]) if row is not None else None


def save_issue_embedding(
    connection: psycopg.Connection[Any],
    issue_id: int,
    model: str,
    embedding: list[float],
) -> None:
    """Store an issue's embedding - upsert on issue_id, so re-running
    the backfill or re-judging is safe to call unconditionally without
    creating duplicate rows."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO issue_embeddings (issue_id, model, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (issue_id) DO UPDATE
            SET model = EXCLUDED.model, embedding = EXCLUDED.embedding,
                created_at = now()
            """,
            (issue_id, model, embedding),
        )


def get_all_issue_embeddings(
    connection: psycopg.Connection[Any],
) -> list[tuple[int, int, list[float]]]:
    """Every stored embedding, as (issue_id, github_number, embedding) -
    the full candidate pool src.duplicate_detection.find_most_similar
    compares a new issue's embedding against. Includes both backfilled
    issues (scripts/backfill_issue_embeddings.py) and previously judged
    ones - issue_embeddings doesn't distinguish the two."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT e.issue_id, i.github_number, e.embedding
            FROM issue_embeddings e
            JOIN issues i ON i.id = e.issue_id
            """
        )
        return [(row[0], row[1], list(row[2])) for row in cursor.fetchall()]


def set_possible_duplicate(
    connection: psycopg.Connection[Any],
    judgment_id: int,
    github_number: int | None,
    similarity: float | None,
) -> None:
    """Record the most similar issue found for a judgment, if any - a
    ranked suggestion (see JudgedIssue.possible_duplicate_number's
    docstring), not a classification. Both arguments None means no
    candidate cleared the loose sanity floor."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE judgments
            SET possible_duplicate_number = %s, possible_duplicate_similarity = %s
            WHERE id = %s
            """,
            (github_number, similarity, judgment_id),
        )


def prune_old_issue_embeddings(
    connection: psycopg.Connection[Any], cutoff: dt.datetime
) -> int:
    """Delete embeddings for issues created before cutoff - keeps
    storage bounded to a rolling window (see scripts/backfill_issue_embeddings.py's
    2-year horizon) rather than growing forever, since new issues get
    embedded every day going forward but old ones stop being realistic
    duplicate candidates. Only the embedding is deleted, never the
    underlying issues/judgments/corrections rows - those remain the
    permanent audit trail regardless of this. Returns the number of rows
    deleted, for logging.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM issue_embeddings
            WHERE issue_id IN (
                SELECT id FROM issues WHERE github_created_at < %s
            )
            """,
            (cutoff,),
        )
        return cursor.rowcount


def get_backfill_state(connection: psycopg.Connection[Any]) -> dt.datetime | None:
    """The end of the last successfully completed backfill window, or
    None if scripts/backfill_issue_embeddings.py has never completed a
    run - the caller uses this to decide between a full BACKFILL_WINDOW
    sweep (never run before) and an incremental one (continue from here)."""

    with connection.cursor() as cursor:
        cursor.execute("SELECT last_window_end FROM backfill_state WHERE id = 1")
        row = cursor.fetchone()
        return row[0] if row is not None else None


def set_backfill_state(
    connection: psycopg.Connection[Any], window_end: dt.datetime
) -> None:
    """Record that the backfill has successfully covered up through
    window_end - upsert on the single fixed row (id = 1). Only called
    once, after every chunk in a run has completed - a crash partway
    through leaves this unset, so the next run re-sweeps from the same
    starting point rather than silently skipping whatever the crashed
    run didn't finish. Safe either way: already-embedded issues are
    skipped on re-processing regardless."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO backfill_state (id, last_window_end)
            VALUES (1, %s)
            ON CONFLICT (id) DO UPDATE SET last_window_end = EXCLUDED.last_window_end
            """,
            (window_end,),
        )
        connection.commit()


@dataclass
class JudgedIssue:
    """A judged, non-bot issue - the result of joining issues and judgments."""

    issue_id: int
    judgment_id: int
    github_number: int
    title: str
    body: str | None
    html_url: str
    repo_owner: str
    repo_name: str
    judgment: IssueJudgment
    possible_duplicate_number: int | None
    """A ranked suggestion (see set_possible_duplicate), not a
    classification - the most similar issue found, if any cleared the
    loose sanity floor. None if no candidate was close enough, or none
    existed yet."""
    possible_duplicate_similarity: float | None


def _row_to_judged_issue(row: tuple[Any, ...]) -> JudgedIssue:
    return JudgedIssue(
        issue_id=row[0],
        judgment_id=row[1],
        github_number=row[2],
        title=row[3],
        body=row[4],
        html_url=row[5],
        repo_owner=row[6],
        repo_name=row[7],
        judgment=IssueJudgment(
            suggested_label=row[8],
            is_spam=row[9],
            summary=row[10],
            priority=row[11],
            rationale=row[12],
            confidence=row[13],
        ),
        possible_duplicate_number=row[14],
        possible_duplicate_similarity=row[15],
    )


_JUDGED_ISSUE_SELECT = """
    SELECT i.id, j.id, i.github_number, i.title, i.body, i.html_url,
           i.repo_owner, i.repo_name,
           j.suggested_label, j.is_spam, j.summary, j.priority,
           j.rationale, j.confidence,
           j.possible_duplicate_number, j.possible_duplicate_similarity
    FROM issues i
    JOIN judgments j ON j.issue_id = i.id
"""


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
            _JUDGED_ISSUE_SELECT
            + """
            WHERE i.repo_owner = %s AND i.repo_name = %s
              AND i.github_created_at >= %s AND i.github_created_at < %s
              AND i.is_bot = FALSE
            ORDER BY i.github_number
            """,
            (repo_owner, repo_name, window_start, window_end),
        )
        rows = cursor.fetchall()

    return [_row_to_judged_issue(row) for row in rows]


def get_judged_issues_by_numbers(
    connection: psycopg.Connection[Any],
    repo_owner: str,
    repo_name: str,
    github_numbers: list[int],
) -> list[JudgedIssue]:
    """Fetch judged issues by explicit github_number, regardless of when
    they were created.

    Used for the digest's backlog section (see src.pipeline.fetch_and_judge_backlog,
    Phase 8 idea A) - backlog issues are older than any time window by
    definition, so get_judged_issues_in_window can never find them.
    """

    if not github_numbers:
        return []

    with connection.cursor() as cursor:
        cursor.execute(
            _JUDGED_ISSUE_SELECT
            + """
            WHERE i.repo_owner = %s AND i.repo_name = %s
              AND i.github_number = ANY(%s)
            ORDER BY i.github_number
            """,
            (repo_owner, repo_name, github_numbers),
        )
        rows = cursor.fetchall()

    return [_row_to_judged_issue(row) for row in rows]


def create_digest(
    connection: psycopg.Connection[Any],
    shadow_repo_owner: str,
    shadow_repo_name: str,
    window_start: dt.datetime,
    window_end: dt.datetime,
) -> int:
    """Create a new digest row for [window_start, window_end).

    No uniqueness constraint on the window: window_start/window_end are a
    fixed lookback from "now", so nothing about the schema requires them
    to be distinct across runs. A double-invocation just produces a
    second, overlapping window rather than a duplicate - has_judgment
    already prevents re-judging any issue that window happens to
    re-fetch.
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
    github_number: int,
) -> int | None:
    """Look up the judgment for a referenced issue number.

    Not scoped to a specific digest: a judgment's digest_id is
    reassigned every time backlog catch-up re-surfaces it into a newer
    digest, so an older, already-closed digest's own link can no longer
    be trusted to find it by the time that digest's comments are
    processed - this looks the judgment up directly by github_number
    instead. Assumes a single source repo, consistent with the rest of
    this codebase.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT j.id FROM judgments j
            JOIN issues i ON i.id = j.issue_id
            WHERE i.github_number = %s
            """,
            (github_number,),
        )
        row = cursor.fetchone()
        return row[0] if row is not None else None


def save_correction(
    connection: psycopg.Connection[Any],
    judgment_id: int,
    digest_id: int,
    github_comment_id: int,
    comment_body: str,
    github_created_at: dt.datetime,
    *,
    superseded: bool = False,
) -> int:
    """Insert a correction, or return the existing row's id if already captured.

    Idempotent on (github_comment_id, judgment_id) - one comment can
    correct several issues at once (one bullet per issue), so a single
    comment legitimately produces several correction rows; only a retry
    of the exact same (comment, judgment) pair is a duplicate.

    digest_id records which thread this correction came from, and
    superseded marks whether it's the currently-authoritative correction
    for judgment_id or one overridden by a newer thread's correction for
    the same issue - see mark_corrections_superseded.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO corrections (
                judgment_id, digest_id, github_comment_id, comment_body,
                github_created_at, superseded
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (github_comment_id, judgment_id) DO NOTHING
            RETURNING id
            """,
            (
                judgment_id,
                digest_id,
                github_comment_id,
                comment_body,
                github_created_at,
                superseded,
            ),
        )
        row = cursor.fetchone()
        if row is not None:
            return row[0]

        cursor.execute(
            "SELECT id FROM corrections WHERE github_comment_id = %s AND judgment_id = %s",
            (github_comment_id, judgment_id),
        )
        existing = cursor.fetchone()
        assert existing is not None
        return existing[0]


def mark_corrections_superseded(
    connection: psycopg.Connection[Any],
    judgment_id: int,
) -> None:
    """Flip any existing authoritative correction(s) for a judgment to
    superseded - called right before a new, more-recently-created
    thread's correction becomes authoritative for the same judgment.

    Deliberately does not commit: this is one step of a single logical
    "a correction became authoritative" event, together with
    save_correction and update_judgment - all three should land or roll
    back together, not risk a crash between them leaving the old
    correction marked superseded with no new one to replace it.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE corrections SET superseded = TRUE WHERE judgment_id = %s AND superseded = FALSE",
            (judgment_id,),
        )


def get_authoritative_correction_digest(
    connection: psycopg.Connection[Any],
    judgment_id: int,
) -> tuple[int, dt.datetime] | None:
    """The (shadow_issue_number, window_end) of the digest holding the
    current authoritative (non-superseded) correction for this judgment,
    if one exists - used to decide whether a new correction is for the
    most-recently-created thread referencing this issue, or a stale one
    that should be marked superseded instead."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT d.shadow_issue_number, d.window_end
            FROM corrections c
            JOIN digests d ON d.id = c.digest_id
            WHERE c.judgment_id = %s AND c.superseded = FALSE
            ORDER BY d.window_end DESC
            LIMIT 1
            """,
            (judgment_id,),
        )
        row = cursor.fetchone()
        return (row[0], row[1]) if row is not None else None


@dataclass
class RejudgeContext:
    """What's needed to re-judge an issue given a correction: its real
    content and its current judgment."""

    title: str
    body: str | None
    judgment: IssueJudgment


def get_rejudge_context(
    connection: psycopg.Connection[Any],
    judgment_id: int,
) -> RejudgeContext | None:
    """Fetch a judgment's underlying issue text and current judgment, for
    passing to a correction-triggered re-judge."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.title, i.body, j.suggested_label, j.is_spam, j.summary,
                   j.priority, j.rationale, j.confidence
            FROM judgments j
            JOIN issues i ON i.id = j.issue_id
            WHERE j.id = %s
            """,
            (judgment_id,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    return RejudgeContext(
        title=row[0],
        body=row[1],
        judgment=IssueJudgment(
            suggested_label=row[2],
            is_spam=row[3],
            summary=row[4],
            priority=row[5],
            rationale=row[6],
            confidence=row[7],
        ),
    )


def update_judgment(
    connection: psycopg.Connection[Any],
    judgment_id: int,
    judgment: IssueJudgment,
) -> None:
    """Overwrite an existing judgment in place - used when a correction
    triggers a re-judge. corrections remains the audit trail of what
    changed and why; this keeps only the judgment's current state, not a
    history of every revision.

    Deliberately does not commit - see mark_corrections_superseded for
    why: this, save_correction, and mark_corrections_superseded are one
    logical event and should land together.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE judgments
            SET suggested_label = %s, is_spam = %s, summary = %s,
                priority = %s, rationale = %s, confidence = %s
            WHERE id = %s
            """,
            (
                judgment.suggested_label,
                judgment.is_spam,
                judgment.summary,
                judgment.priority,
                judgment.rationale,
                judgment.confidence,
                judgment_id,
            ),
        )


# The judgment fields a correction is tracked against for
# get_correction_field_counts. Deliberately excludes rationale (always
# regenerated, so it "changes" on every re-judge trivially) and
# confidence (a continuous float, same problem) - tracking either would
# be noise, not signal about what a human actually corrected.
TRACKED_CORRECTION_FIELDS = ("suggested_label", "is_spam", "priority")


def set_correction_changed_fields(
    connection: psycopg.Connection[Any],
    correction_id: int,
    changed_fields: list[str],
) -> None:
    """Record which of TRACKED_CORRECTION_FIELDS actually differed between
    a judgment's pre- and post-correction values - called once per
    successful re-judge, right after update_judgment, while both the old
    and new values are still in the caller's hands (see src/corrections.py).

    Deliberately does not commit - lands with the same re-judge event as
    update_judgment/mark_corrections_superseded/save_correction.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE corrections SET changed_fields = %s WHERE id = %s",
            (changed_fields, correction_id),
        )


def get_correction_field_counts(
    connection: psycopg.Connection[Any],
    since: dt.datetime | None = None,
) -> dict[str, int]:
    """How many corrections touched each tracked field, across every
    correction that actually got a live re-judge (changed_fields IS NOT
    NULL - capped/failed/superseded corrections never got one, so they
    can't say anything about what would have changed).

    Only reflects data captured after set_correction_changed_fields
    existed - corrections recorded before this column was added have
    changed_fields = NULL and are indistinguishable, in this query, from
    "never re-judged". Not backfilled: the pre-correction judgment values
    those older rows would need are already overwritten in judgments,
    with nothing else recording what they used to be.
    """

    query = """
        SELECT field, COUNT(*)
        FROM corrections, unnest(changed_fields) AS field
        WHERE changed_fields IS NOT NULL
    """
    params: list[Any] = []
    if since is not None:
        query += " AND captured_at >= %s"
        params.append(since)
    query += " GROUP BY field ORDER BY COUNT(*) DESC"

    with connection.cursor() as cursor:
        cursor.execute(query, params)
        return {row[0]: row[1] for row in cursor.fetchall()}


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

    A judgment can have more than one correction row now (superseded
    ones from stale threads - see mark_corrections_superseded), so the
    join is restricted to the non-superseded one to avoid surfacing a
    judgment twice with conflicting-looking corrections, or surfacing a
    stale correction instead of the one that actually shaped the current
    judgment.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.title, i.body, j.suggested_label, j.is_spam, j.summary,
                   j.priority, j.rationale, j.confidence, c.comment_body
            FROM judgments j
            JOIN issues i ON i.id = j.issue_id
            JOIN digests d ON d.id = j.digest_id
            LEFT JOIN corrections c ON c.judgment_id = j.id AND c.superseded = FALSE
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
    digest - the unit the golden evaluation set is built from.

    Distinct from ReviewedJudgment (few-shot context, recent-N, no
    traceability fields needed) even though the underlying data
    overlaps - these two serve different purposes and are expected to
    diverge further as the correctness rubric evolves.
    """

    github_number: int
    issue_title: str
    issue_body: str | None
    judgment: IssueJudgment
    correction_text: str | None
    digest_date: dt.date
    """Pacific calendar date the digest's window ended on - a display
    label derived from window_end, not part of the digest's identity."""


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
    label derived from window_end, not part of the digest's identity."""
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
    for correction capture (the issue may or may not be closed yet), and
    to detect whether the daily job should treat this run as "still
    working on a WIP digest" rather than a fresh poll.
    """

    digest_id: int
    shadow_owner: str
    shadow_repo: str
    shadow_issue_number: int
    window_end: dt.datetime


def get_unreviewed_digests(
    connection: psycopg.Connection[Any],
) -> list[UnreviewedDigest]:
    """Fetch all published-but-not-yet-reviewed digests, oldest first.

    The last element, if any, is the most recently created WIP digest -
    used by run_daily_cycle both to detect that a WIP digest exists at
    all, and as the window_start for "what's new since then" once one
    does.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, shadow_repo_owner, shadow_repo_name, shadow_issue_number,
                   window_end
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
            window_end=row[4],
        )
        for row in rows
    ]
