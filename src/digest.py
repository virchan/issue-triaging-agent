from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
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

# Display only. The digest's actual identity is [window_start,
# window_end); this only decides what calendar date the title/body show
# a human, and can never cause a query to miss data the way computing
# "today" in the wrong timezone did.
OPERATOR_TIMEZONE = ZoneInfo("America/Los_Angeles")

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_PRIORITY_HEADINGS = {
    "high": "High priority",
    "medium": "Medium priority",
    "low": "Low priority",
}


@dataclass
class DigestContent:
    """Aggregated digest content, ready to publish."""

    digest_id: int
    title: str
    body: str
    issue_count: int
    labels: list[str] = field(default_factory=list)


def format_digest_title(date: dt.date) -> str:
    return f"Triage digest — {date.isoformat()}"


def _redirect_url(html_url: str) -> str:
    """Swap a real github.com issue/PR URL for its redirect.github.com
    equivalent - same path (so /issues/ vs /pull/ stays correct without
    us having to know which one this is), different host. Confirmed
    empirically (by the operator, against a real scikit-learn issue) that
    a redirect.github.com link does not create a GitHub cross-reference
    the way a real github.com link does - see
    https://github.com/orgs/community/discussions/23123. Unofficial and
    undocumented by GitHub, so this could in principle stop working; if
    it ever does, the fix is here, not scattered across every call site.
    """

    return html_url.replace("https://github.com/", "https://redirect.github.com/", 1)


def _render_issue_section(issues: list[JudgedIssue]) -> list[str]:
    """Render one priority-grouped block of issues - the shared per-issue
    formatting used for both the "new" and "backlog" sections.

    Issue references are a real, clickable Markdown link - `[<code>repo/
    number</code>](redirect.github.com/...)` - rather than the inert
    backtick-wrapped `owner/repo/number` text used before. A real link
    (or GitHub's own owner/repo#NNN autolink syntax) whose target is a
    real github.com issue/PR URL creates a visible GitHub cross-reference
    on that issue; redirect.github.com does not (see _redirect_url). The
    `<code>` tags, not backticks, are deliberate: inline code inside a
    Markdown link label renders inconsistently, HTML tags don't.
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
        link_text = f"{item.repo_name}/{item.github_number}"
        reference = f"[<code>{link_text}</code>]({_redirect_url(item.html_url)})"
        lines.append(f"### {reference} — {item.title}{spam_flag}")
        lines.append("")
        # Deliberately kept alongside the now-clickable heading above, not
        # redundant with it: the heading depends on redirect.github.com
        # staying up (unofficial, undocumented - see _redirect_url), while
        # this is the canonical github.com URL, correct regardless. Still
        # backtick-wrapped/inert so it can't itself autolink into a
        # cross-reference.
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
    wip_digest_issue_number: int | None = None,
) -> str:
    """Render a day's judged issues into a Markdown digest body.

    `label` (e.g. "Needs Triage") is named explicitly in the empty/summary
    messages when given - "no non-bot issues" previously implied no
    activity at all, when the query is actually scoped to one label. A
    real gap flagged during real operation.

    `backlog_issues` are older, already-open issues reviewed because
    nothing new needed triage, rendered as a clearly distinct section -
    never silently merged with `issues`, since that would misrepresent
    old backlog as new activity.

    `wip_digest_issue_number`, when given, means a still-open,
    not-yet-reviewed digest already exists - a reminder line is
    prepended pointing back at it. A bare "#NNN" is used here
    deliberately (unlike scikit-learn issue references): this points at
    another issue in *this same repo*, where a real clickable
    cross-reference is exactly what's wanted, not something to avoid.
    """

    backlog_issues = backlog_issues or []
    scope = f'issue(s) labelled "{label}"' if label else "non-bot issue(s)"

    lines: list[str] = []
    if wip_digest_issue_number is not None:
        lines.append(
            f"_Still working on #{wip_digest_issue_number}? "
            "The issue(s) below are what's new since it was opened._"
        )
        lines.append("")

    if not issues and not backlog_issues:
        lines.append(f"No newly created {scope} were found for {date.isoformat()}.")
        return "\n".join(lines).strip() + "\n"

    if issues:
        lines.append(f"{len(issues)} {scope} reviewed for {date.isoformat()}.")
        lines.append("")
        lines.extend(_render_issue_section(issues))

    if backlog_issues:
        if issues:
            lines.append("---")
            lines.append("")
            lines.append(
                "That's everything newly created. Because there's nothing "
                f"else new to triage, here are {len(backlog_issues)} older "
                "open issue(s) that still need triaging too:"
            )
        else:
            lines.append(
                f"No newly created {scope} were found for {date.isoformat()}. "
                f"Because there's nothing new to triage, here are "
                f"{len(backlog_issues)} older open issue(s) that still need "
                "triaging:"
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
    labels: list[str] | None = None,
    wip_digest_issue_number: int | None = None,
) -> DigestContent:
    """Aggregate a window's judged issues into digest content.

    Persists the digest record and links each judgment to it, but does
    not publish anything to GitHub - see publish_digest. The title/body
    show window_end's Pacific calendar date - display only, not the
    digest's identity. `label` should be the same label the pipeline
    fetched with (e.g. "Needs Triage"), so the body accurately describes
    what was actually searched for.

    `backlog_issue_numbers` are the github_numbers judged by
    fetch_and_judge_backlog this run, if any - fetched here by explicit
    number rather than by window, since a backlog issue is by definition
    older than window_start.

    `labels` (e.g. ["daily digest"], plus "manually-triggered" when
    applicable) are attached to the GitHub issue publish_digest creates.
    Must already exist in the shadow repo.

    `wip_digest_issue_number` is the shadow-repo issue number of the most
    recent still-open digest, if one exists - passed straight through to
    format_digest_body's reminder line.
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
        body=format_digest_body(
            display_date,
            judged_issues,
            label,
            backlog_issues,
            wip_digest_issue_number=wip_digest_issue_number,
        ),
        issue_count=len(judged_issues) + len(backlog_issues),
        labels=labels or [],
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
        shadow_owner, shadow_repo, digest.title, digest.body, labels=digest.labels
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
