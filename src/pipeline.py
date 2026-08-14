from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg

from src.bot_filter import partition_bot_issues
from src.db import (
    get_recent_reviewed_judgments,
    has_judgment,
    save_issue_snapshots,
    save_judgment,
)
from src.gemini_client import GeminiJudge, GeminiResponseError, GeminiUnavailableError
from src.github_client import GitHubClient

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of one fetch-filter-judge-store run."""

    fetched: int
    bot_excluded: int
    judged: int
    already_judged: int
    failures: list[tuple[int, str]] = field(default_factory=list)


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

    failures: list[tuple[int, str]] = []
    judged_count = 0
    already_judged_count = 0
    for issue in non_bot:
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
        except (GeminiUnavailableError, GeminiResponseError) as error:
            LOGGER.warning(f"Judgment failed for issue #{issue.number}: {error}")
            failures.append((issue.number, str(error)))

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
