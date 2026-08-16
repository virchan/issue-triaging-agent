from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import psycopg

from src.db import (
    JudgedIssue,
    create_digest,
    get_judged_issues_by_numbers,
    get_judged_issues_in_window,
    is_digest_published,
    link_judgments_to_digest,
    mark_digest_published,
)
from src.github_client import GitHubClient

LOGGER = logging.getLogger(__name__)

# Display only - see LOG.md entry 53. The digest's actual identity is
# [window_start, window_end); this only decides what calendar date the
# title/body show a human, and can never cause a query to miss data the
# way computing "today" in the wrong timezone did.
OPERATOR_TIMEZONE = ZoneInfo("America/Los_Angeles")

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_PRIORITY_HEADINGS = {
    "high": "High priority",
    "medium": "Medium priority",
    "low": "Low priority",
}


@dataclass
class DigestContent:
    """Aggregated digest content, ready to publish (see Step 17)."""

    digest_id: int
    title: str
    body: str
    issue_count: int


def format_digest_title(date: dt.date) -> str:
    return f"Triage digest — {date.isoformat()}"


def _render_issue_section(issues: list[JudgedIssue]) -> list[str]:
    """Render one priority-grouped block of issues - the shared per-issue
    formatting used for both the "new" and "backlog" sections.

    Issue references are NOT rendered as clickable markdown links to the
    scikit-learn URL, and the URL that is shown is wrapped in backticks
    (inline code). A real markdown link with the raw URL as its target
    creates a visible GitHub cross-reference on the target scikit-learn
    issue - confirmed empirically (see LOG.md) - which would surface
    this project on scikit-learn's side the moment the shadow repo goes
    public. Backtick-wrapped URLs and bare "#NNN" text do not trigger
    this.
    """

    lines: list[str] = []
    ordered = sorted(issues, key=lambda item: _PRIORITY_ORDER[item.judgment.priority])

    current_priority: str | None = None
    for item in ordered:
        judgment = item.judgment
        if judgment.priority != current_priority:
            current_priority = judgment.priority
            lines.append(f"## {_PRIORITY_HEADINGS[current_priority]}")
            lines.append("")

        spam_flag = " ⚠️ possible spam" if judgment.is_spam else ""
        lines.append(f"### #{item.github_number} — {item.title}{spam_flag}")
        lines.append("")
        lines.append(f"- **Link:** `{item.html_url}`")
        lines.append(f"- **Suggested label:** {judgment.suggested_label or '(none)'}")
        lines.append(f"- **Confidence:** {judgment.confidence:.2f}")
        lines.append("")
        lines.append(judgment.summary)
        lines.append("")
        lines.append(f"*Rationale: {judgment.rationale}*")
        lines.append("")

    return lines


def format_digest_body(
    date: dt.date,
    issues: list[JudgedIssue],
    label: str | None = None,
    backlog_issues: list[JudgedIssue] | None = None,
) -> str:
    """Render a day's judged issues into a Markdown digest body.

    `label` (e.g. "Needs Triage") is named explicitly in the empty/summary
    messages when given - "no non-bot issues" previously implied no
    activity at all, when the query is actually scoped to one label. A
    real gap flagged during real operation (see LOG.md, daily-log.md).

    `backlog_issues` (Phase 8 idea A - see LOG.md) are older, already-open
    issues reviewed because nothing new needed triage, rendered as a
    clearly distinct section - never silently merged with `issues`, since
    that would misrepresent old backlog as new activity.
    """

    backlog_issues = backlog_issues or []
    scope = f'issue(s) labelled "{label}"' if label else "non-bot issue(s)"

    if not issues and not backlog_issues:
        return f"No newly created {scope} were found for {date.isoformat()}."

    lines: list[str] = []

    if issues:
        lines.append(f"{len(issues)} {scope} reviewed for {date.isoformat()}.")
        lines.append("")
        lines.extend(_render_issue_section(issues))

    if backlog_issues:
        if issues:
            lines.append("---")
            lines.append("")
            lines.append(
                f"No new {scope} required attention, so "
                f"{len(backlog_issues)} older open issue(s) were reviewed too:"
            )
        else:
            lines.append(
                f"No newly created {scope} were found for {date.isoformat()}, "
                f"so {len(backlog_issues)} older open issue(s) with that label "
                "were reviewed instead:"
            )
        lines.append("")
        lines.extend(_render_issue_section(backlog_issues))

    return "\n".join(lines).strip() + "\n"


def build_digest(
    connection: psycopg.Connection[Any],
    *,
    source_owner: str,
    source_repo: str,
    shadow_owner: str,
    shadow_repo: str,
    window_start: dt.datetime,
    window_end: dt.datetime,
    label: str | None = None,
    backlog_issue_numbers: list[int] | None = None,
) -> DigestContent:
    """Aggregate a window's judged issues into digest content.

    Persists the digest record and links each judgment to it, but does
    not publish anything to GitHub - see publish_digest. The title/body
    show window_end's Pacific calendar date - display only, not the
    digest's identity (see LOG.md entry 53). `label` should be the same
    label the pipeline fetched with (e.g. "Needs Triage"), so the body
    accurately describes what was actually searched for.

    `backlog_issue_numbers` (Phase 8 idea A - see LOG.md) are the
    github_numbers judged by fetch_and_judge_backlog this run, if any -
    fetched here by explicit number rather than by window, since a
    backlog issue is by definition older than window_start.
    """

    display_date = window_end.astimezone(OPERATOR_TIMEZONE).date()

    judged_issues = get_judged_issues_in_window(
        connection, source_owner, source_repo, window_start, window_end
    )
    backlog_issues = get_judged_issues_by_numbers(
        connection, source_owner, source_repo, backlog_issue_numbers or []
    )

    digest_id = create_digest(
        connection, shadow_owner, shadow_repo, window_start, window_end
    )
    all_judgment_ids = [item.judgment_id for item in judged_issues] + [
        item.judgment_id for item in backlog_issues
    ]
    link_judgments_to_digest(connection, digest_id, all_judgment_ids)

    return DigestContent(
        digest_id=digest_id,
        title=format_digest_title(display_date),
        body=format_digest_body(display_date, judged_issues, label, backlog_issues),
        issue_count=len(judged_issues) + len(backlog_issues),
    )


def publish_digest(
    connection: psycopg.Connection[Any],
    shadow_client: GitHubClient,
    digest: DigestContent,
    *,
    shadow_owner: str,
    shadow_repo: str,
) -> tuple[int, str] | None:
    """Post a digest as an issue in the shadow repo.

    Idempotent: returns None without posting anything if this digest was
    already published (checked before calling GitHub, not just before
    writing to the DB - same pattern as the judgment pipeline's
    has_judgment check). shadow_client must be authenticated with
    SHADOW_REPO_TOKEN, not the read-only client used for scikit-learn.

    Publishes even on a zero-issue day (format_digest_body's "no issues"
    message) rather than skipping - a daily record of "the pipeline ran
    and found nothing" is itself evidence of continuous operation,
    distinct from "the pipeline didn't run."
    """

    if is_digest_published(connection, digest.digest_id):
        LOGGER.info(
            f"Digest {digest.digest_id} already published, skipping",
            extra={
                "event": "publish_outcome",
                "digest_id": digest.digest_id,
                "outcome": "already_published",
            },
        )
        return None

    issue_number, html_url = shadow_client.create_issue(
        shadow_owner, shadow_repo, digest.title, digest.body
    )
    mark_digest_published(connection, digest.digest_id, issue_number)

    LOGGER.info(
        f"Digest {digest.digest_id} published as {shadow_owner}/{shadow_repo}#{issue_number}",
        extra={
            "event": "publish_outcome",
            "digest_id": digest.digest_id,
            "outcome": "published",
            "issue_number": issue_number,
            "issue_count": digest.issue_count,
        },
    )

    return issue_number, html_url
