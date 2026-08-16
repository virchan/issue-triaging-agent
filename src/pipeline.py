from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg

from src.bot_filter import partition_bot_issues
from src.db import (
    ReviewedJudgment,
    get_judged_github_numbers,
    get_recent_reviewed_judgments,
    has_judgment,
    save_issue_snapshots,
    save_judgment,
)
from src.gemini_client import GeminiJudge, GeminiResponseError, GeminiUnavailableError
from src.github_client import GitHubClient, GitHubIssue

LOGGER = logging.getLogger(__name__)

# Backlog catch-up (Phase 8 idea A) caps how many older issues get judged
# on an idle day at once, matching the operator's real stated daily
# review capacity (2-3/day, up to 5) - see LOG.md.
BACKLOG_CAP = 3


@dataclass
class PipelineResult:
    """Summary of one fetch-filter-judge-store run."""

    fetched: int
    bot_excluded: int
    judged: int
    already_judged: int
    failures: list[tuple[int, str]] = field(default_factory=list)


def _judge_and_persist(
    *,
    gemini_judge: GeminiJudge,
    connection: psycopg.Connection[Any],
    issues: list[GitHubIssue],
    ids_by_number: dict[int, int],
    known_labels: list[str],
    recent_examples: list[ReviewedJudgment],
) -> tuple[int, int, list[tuple[int, str]], list[int]]:
    """Judge each not-yet-judged issue and persist the result.

    Shared by fetch_and_judge and fetch_and_judge_backlog - only how the
    issue list is sourced differs between the two. Returns
    (judged_count, already_judged_count, failures, judged_github_numbers) -
    the last one lets a caller identify exactly which issues were newly
    judged this call, needed to build the digest's backlog section.
    """

    failures: list[tuple[int, str]] = []
    judged_count = 0
    already_judged_count = 0
    judged_numbers: list[int] = []

    for issue in issues:
        issue_id = ids_by_number[issue.number]
        if has_judgment(connection, issue_id):
            already_judged_count += 1
            continue

        try:
            judgment = gemini_judge.judge(
                title=issue.title,
                body=issue.body,
                known_labels=known_labels,
                recent_examples=recent_examples,
            )
            save_judgment(connection, issue_id, judgment)
            judged_count += 1
            judged_numbers.append(issue.number)
        except (GeminiUnavailableError, GeminiResponseError) as error:
            LOGGER.warning(f"Judgment failed for issue #{issue.number}: {error}")
            failures.append((issue.number, str(error)))

    return judged_count, already_judged_count, failures, judged_numbers


def fetch_and_judge(
    *,
    github_client: GitHubClient,
    gemini_judge: GeminiJudge,
    connection: psycopg.Connection[Any],
    owner: str,
    repo: str,
    window_start: dt.datetime,
    window_end: dt.datetime,
    label: str | None = None,
) -> PipelineResult:
    """Fetch issues created in [window_start, window_end), filter, judge,
    and persist everything.

    If label is given (e.g. "Needs Triage"), only issues carrying that
    label are fetched in the first place - issues a maintainer has
    already triaged never enter the pipeline at all, rather than being
    fetched and then discarded.

    A single issue's judgment failure is logged and skipped rather than
    aborting the whole run - one bad issue should not block the rest of
    the digest. Digest aggregation and shadow-repo publishing are not
    part of this function - see later steps.

    Recent reviewed judgments (real corrections and confirmations, see
    src.db.get_recent_reviewed_judgments) are fetched once per run and
    passed to every judge() call as few-shot context - not refetched per
    issue, since they don't change within a single run.
    """

    start = time.monotonic()
    issues = github_client.fetch_issues_created_between(
        owner, repo, window_start, window_end, label=label
    )
    non_bot, bot = partition_bot_issues(issues)

    ids_by_number = save_issue_snapshots(connection, owner, repo, non_bot, bot)

    known_labels = github_client.fetch_labels(owner, repo)
    recent_examples = get_recent_reviewed_judgments(connection)

    judged_count, already_judged_count, failures, _ = _judge_and_persist(
        gemini_judge=gemini_judge,
        connection=connection,
        issues=non_bot,
        ids_by_number=ids_by_number,
        known_labels=known_labels,
        recent_examples=recent_examples,
    )

    result = PipelineResult(
        fetched=len(issues),
        bot_excluded=len(bot),
        judged=judged_count,
        already_judged=already_judged_count,
        failures=failures,
    )

    LOGGER.info(
        f"Poll run completed for {owner}/{repo}, "
        f"window {window_start.isoformat()} to {window_end.isoformat()}",
        extra={
            "event": "poll_run",
            "owner": owner,
            "repo": repo,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "duration_seconds": time.monotonic() - start,
            "fetched": result.fetched,
            "bot_excluded": result.bot_excluded,
            "judged": result.judged,
            "already_judged": result.already_judged,
            "failure_count": len(result.failures),
        },
    )

    return result


def fetch_and_judge_backlog(
    *,
    github_client: GitHubClient,
    gemini_judge: GeminiJudge,
    connection: psycopg.Connection[Any],
    owner: str,
    repo: str,
    label: str,
    cap: int = BACKLOG_CAP,
) -> tuple[PipelineResult, list[int]]:
    """When a poll finds nothing new, look for older open issues still
    carrying `label` that have never been judged, oldest first, and
    judge up to `cap` of them (Phase 8 idea A - see LOG.md/daily-log.md).

    Backlog issues fall outside any time window by definition, so they
    can never be produced by fetch_and_judge - this is a genuinely
    separate sourcing path, not a variant of the window-based one.
    Returns the usual PipelineResult, plus the github_numbers actually
    judged this call - the digest needs those explicitly, since
    get_judged_issues_in_window can't find issues outside its window.
    """

    start = time.monotonic()
    already_judged_numbers = get_judged_github_numbers(connection, owner, repo)

    candidates = github_client.fetch_open_issues_with_label(owner, repo, label)
    non_bot, bot = partition_bot_issues(candidates)
    not_yet_judged = [
        issue for issue in non_bot if issue.number not in already_judged_numbers
    ]
    to_judge = not_yet_judged[:cap]

    ids_by_number = save_issue_snapshots(connection, owner, repo, to_judge, bot)

    known_labels = github_client.fetch_labels(owner, repo)
    recent_examples = get_recent_reviewed_judgments(connection)

    judged_count, already_judged_count, failures, judged_numbers = _judge_and_persist(
        gemini_judge=gemini_judge,
        connection=connection,
        issues=to_judge,
        ids_by_number=ids_by_number,
        known_labels=known_labels,
        recent_examples=recent_examples,
    )

    result = PipelineResult(
        fetched=len(to_judge),
        bot_excluded=len(bot),
        judged=judged_count,
        already_judged=already_judged_count,
        failures=failures,
    )

    LOGGER.info(
        f"Backlog catch-up completed for {owner}/{repo}",
        extra={
            "event": "backlog_catchup",
            "owner": owner,
            "repo": repo,
            "label": label,
            "candidates_found": len(candidates),
            "already_judged_skipped": len(non_bot) - len(not_yet_judged),
            "capped_out": len(not_yet_judged) - len(to_judge),
            "duration_seconds": time.monotonic() - start,
            "judged": result.judged,
            "failure_count": len(result.failures),
        },
    )

    return result, judged_numbers
