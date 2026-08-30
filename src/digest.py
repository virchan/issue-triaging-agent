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
from src.rendering import render_template

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


def _redirect_issue_url(owner: str, repo: str, number: int) -> str:
    """Build a redirect.github.com URL directly from owner/repo/number -
    used for the possible-duplicate reference, where only a github_number
    is stored (JudgedIssue.possible_duplicate_number), not a full
    html_url the way the issue's own reference has."""

    return f"https://redirect.github.com/{owner}/{repo}/issues/{number}"


def _has_code_block(body: str | None) -> bool:
    """Deterministic, not LLM-judged: does the issue body contain at
    least one fenced code block (a pair of triple-backtick markers)?
    Cheap real signal for whether an issue likely includes a reproducer,
    directly requested by a real maintainer reviewing digests - no
    Gemini call needed for something this mechanical to check.
    """

    return body is not None and body.count("```") >= 2


def _group_issues_by_priority(issues: list[JudgedIssue]) -> list[dict[str, Any]]:
    """Group issues by priority, in _PRIORITY_ORDER order, skipping empty
    groups - the one piece of real logic (grouping/sorting/URL-building,
    not Markdown layout) the digest template still delegates to Python.
    Shared by the "new" and "backlog" sections - see digest.md.jinja's
    render_groups macro, which both call into.

    Each issue's reference is a real, clickable link - `[<code>owner/repo
    #number</code>](redirect.github.com/...)` in the template - GitHub's
    own real autolink syntax as the display text, not the inert
    backtick-wrapped `owner/repo/number` text used before the
    redirect.github.com change. Writing `owner/repo#NNN` as plain text
    targeting a real github.com issue/PR would normally create its own
    GitHub cross-reference independent of any surrounding link - but
    wrapped in `<code>` with a redirect.github.com href, empirically
    confirmed not to (traced a real correction comment using this exact
    form to a target issue's timeline and found no cross-reference from
    it). html_url itself is not passed to the template - only
    redirect_url is rendered, real confirmed to not create a
    cross-reference; keeping a second, redundant "Link:" field showing
    the same URL twice per issue wasn't worth it once that was
    independently confirmed live.

    possible_duplicate (None unless set) is a ranked suggestion, not a
    classification - the most similar other issue found by embedding
    similarity, above a loose sanity floor. Computed
    once at judgment time (src.pipeline._find_and_record_possible_duplicate),
    not here - this only renders whatever was already stored.
    """

    ordered = sorted(issues, key=lambda item: _PRIORITY_ORDER[item.judgment.priority])

    groups: list[dict[str, Any]] = []
    current_priority: str | None = None
    for item in ordered:
        judgment = item.judgment
        if judgment.priority != current_priority:
            current_priority = judgment.priority
            groups.append(
                {"heading": _PRIORITY_HEADINGS[current_priority], "issues": []}
            )

        groups[-1]["issues"].append(
            {
                "reference_text": f"{item.repo_owner}/{item.repo_name}#{item.github_number}",
                "redirect_url": _redirect_url(item.html_url),
                "title": item.title,
                # Plain text, not a boolean the template branches on: an
                # inline {% if %}...{% endif %} right before a blank line
                # interacts badly with the Environment's trim_blocks
                # setting (it eats the newline right after the tag,
                # collapsing the blank line meant to follow it).
                "spam_flag": " ⚠️ possible spam" if judgment.is_spam else "",
                "suggested_labels": judgment.suggested_labels,
                "confidence": judgment.confidence,
                "has_code_block": _has_code_block(item.body),
                "possible_duplicate": (
                    {
                        "reference_text": f"{item.repo_owner}/{item.repo_name}#{item.possible_duplicate_number}",
                        "redirect_url": _redirect_issue_url(
                            item.repo_owner,
                            item.repo_name,
                            item.possible_duplicate_number,
                        ),
                        "similarity": item.possible_duplicate_similarity,
                    }
                    if item.possible_duplicate_number is not None
                    else None
                ),
                "summary": judgment.summary,
                "rationale": judgment.rationale,
            }
        )

    return groups


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

    return render_template(
        "digest.md.jinja",
        date=date.isoformat(),
        scope=scope,
        wip_digest_issue_number=wip_digest_issue_number,
        has_issues=bool(issues),
        has_backlog=bool(backlog_issues),
        issue_count=len(issues),
        backlog_count=len(backlog_issues),
        issue_groups=_group_issues_by_priority(issues),
        backlog_groups=_group_issues_by_priority(backlog_issues),
    )


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
