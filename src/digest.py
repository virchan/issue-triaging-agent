from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any

import psycopg

from src.db import (
    JudgedIssue,
    create_digest,
    get_judged_issues_for_date,
    is_digest_published,
    link_judgments_to_digest,
    mark_digest_published,
)
from src.github_client import GitHubClient

LOGGER = logging.getLogger(__name__)

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


def format_digest_body(date: dt.date, issues: list[JudgedIssue]) -> str:
    """Render a day's judged issues into a Markdown digest body.

    Grouped by priority (high, then medium, then low - only priorities
    that actually have issues get a heading), spam-flagged issues are
    marked distinctly rather than silently mixed in.

    Issue references are NOT rendered as clickable markdown links to the
    scikit-learn URL, and the URL that is shown is wrapped in backticks
    (inline code). A real markdown link with the raw URL as its target
    creates a visible GitHub cross-reference on the target scikit-learn
    issue - confirmed empirically (see LOG.md) - which would surface
    this project on scikit-learn's side the moment the shadow repo goes
    public. Backtick-wrapped URLs and bare "#NNN" text do not trigger
    this.
    """

    if not issues:
        return f"No non-bot issues were created on {date.isoformat()}."

    lines = [f"{len(issues)} issue(s) reviewed for {date.isoformat()}.", ""]

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

    return "\n".join(lines).strip() + "\n"


def build_digest(
    connection: psycopg.Connection[Any],
    *,
    source_owner: str,
    source_repo: str,
    shadow_owner: str,
    shadow_repo: str,
    date: dt.date,
) -> DigestContent:
    """Aggregate a day's judged issues into digest content.

    Persists the digest record and links each judgment to it, but does
    not publish anything to GitHub - see publish_digest.
    """

    judged_issues = get_judged_issues_for_date(
        connection, source_owner, source_repo, date
    )
    digest_id = create_digest(connection, shadow_owner, shadow_repo, date)
    link_judgments_to_digest(
        connection, digest_id, [item.judgment_id for item in judged_issues]
    )

    return DigestContent(
        digest_id=digest_id,
        title=format_digest_title(date),
        body=format_digest_body(date, judged_issues),
        issue_count=len(judged_issues),
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
