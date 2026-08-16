from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import psycopg

from src.corrections import CaptureResult, capture_corrections
from src.db import get_latest_digest_window_end, get_unreviewed_digests
from src.digest import DigestContent, build_digest, publish_digest
from src.gemini_client import GeminiJudge
from src.github_client import GitHubClient
from src.pipeline import PipelineResult, fetch_and_judge, fetch_and_judge_backlog

# How far back the very first poll ever looks, when there's no previous
# digest's window_end to chain from. Matches the daily cadence; only used
# once, on the first real run.
BOOTSTRAP_WINDOW = dt.timedelta(hours=24)


@dataclass
class DailyCycleResult:
    """Summary of one full state-machine pass: forward (fetch through
    publish) and backward (correction capture on previously-published,
    now-possibly-closed digests)."""

    pipeline: PipelineResult
    digest: DigestContent
    published: tuple[int, str] | None
    reviews: list[CaptureResult] = field(default_factory=list)
    backlog: PipelineResult | None = None
    """Set only when the window found nothing new and a backlog catch-up
    ran (Phase 8 idea A - see LOG.md)."""


def run_daily_cycle(
    *,
    github_client: GitHubClient,
    shadow_client: GitHubClient,
    gemini_judge: GeminiJudge,
    connection: psycopg.Connection[Any],
    source_owner: str,
    source_repo: str,
    shadow_owner: str,
    shadow_repo: str,
    label: str | None,
) -> DailyCycleResult:
    """Run the full state machine: fetched -> filtered/judged -> digested
    -> published -> reviewed/corrected -> closed.

    The poll window is computed here, not passed in by the caller - see
    LOG.md entry 53. window_start is the previous digest's window_end (a
    watermark, not "today"), so a late or missed run automatically
    covers the full gap next time, and callers no longer need to reason
    about timezones or calendar days at all. window_end is always now.

    Forward half (this window's issues through publishing a new digest)
    runs first, then every previously-published digest not yet marked
    reviewed is checked - capture_corrections is a no-op for any whose
    GitHub issue isn't closed yet, so calling it here is always safe,
    not just when a digest is known to be ready.

    If the window finds nothing new at all, a backlog catch-up runs
    instead (Phase 8 idea A - see LOG.md): older, already-open issues
    carrying `label` that have never been judged, so the system doesn't
    stay permanently blind to a pre-existing backlog. Requires a real
    `label` - there's no well-defined "backlog" without one.

    github_client must be read-only (scikit-learn); shadow_client must
    be authenticated with SHADOW_REPO_TOKEN (shadow repo writes).
    """

    window_end = dt.datetime.now(dt.UTC)
    previous_window_end = get_latest_digest_window_end(connection)
    window_start = (
        previous_window_end
        if previous_window_end is not None
        else window_end - BOOTSTRAP_WINDOW
    )

    pipeline_result = fetch_and_judge(
        github_client=github_client,
        gemini_judge=gemini_judge,
        connection=connection,
        owner=source_owner,
        repo=source_repo,
        window_start=window_start,
        window_end=window_end,
        label=label,
    )

    backlog_result: PipelineResult | None = None
    backlog_issue_numbers: list[int] = []
    if pipeline_result.fetched == 0 and label is not None:
        backlog_result, backlog_issue_numbers = fetch_and_judge_backlog(
            github_client=github_client,
            gemini_judge=gemini_judge,
            connection=connection,
            owner=source_owner,
            repo=source_repo,
            label=label,
        )

    digest = build_digest(
        connection,
        source_owner=source_owner,
        source_repo=source_repo,
        shadow_owner=shadow_owner,
        shadow_repo=shadow_repo,
        window_start=window_start,
        window_end=window_end,
        label=label,
        backlog_issue_numbers=backlog_issue_numbers,
    )

    published = publish_digest(
        connection,
        shadow_client,
        digest,
        shadow_owner=shadow_owner,
        shadow_repo=shadow_repo,
    )

    reviews: list[CaptureResult] = []
    for unreviewed in get_unreviewed_digests(connection):
        reviews.append(
            capture_corrections(
                connection,
                shadow_client,
                digest_id=unreviewed.digest_id,
                shadow_owner=unreviewed.shadow_owner,
                shadow_repo=unreviewed.shadow_repo,
                shadow_issue_number=unreviewed.shadow_issue_number,
            )
        )

    return DailyCycleResult(
        pipeline=pipeline_result,
        digest=digest,
        published=published,
        reviews=reviews,
        backlog=backlog_result,
    )
