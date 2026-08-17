from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import psycopg

from src.corrections import CaptureResult, capture_corrections
from src.db import get_unreviewed_digests
from src.digest import DigestContent, build_digest, publish_digest
from src.gemini_client import GeminiJudge
from src.github_client import GitHubClient
from src.pipeline import PipelineResult, fetch_and_judge, fetch_and_judge_backlog

# Every poll looks back this far from "now", regardless of when the
# previous run happened - see LOG.md entry 56. Matches the daily 09:00
# PDT cadence: a shorter window would leave a permanent gap between runs.
WINDOW_DURATION = dt.timedelta(hours=24)

# Attached to every digest issue created - see LOG.md entry 57. Must
# already exist in the shadow repo (created by hand, not by this code).
DIGEST_LABEL = "daily digest"
MANUALLY_TRIGGERED_LABEL = "manually-triggered"


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
    manually_triggered: bool = False,
) -> DailyCycleResult:
    """Run the full state machine: fetched -> filtered/judged -> digested
    -> published -> reviewed/corrected -> closed.

    The poll window is computed here, not passed in by the caller - see
    LOG.md entries 53/56. window_end is always now; window_start is a
    fixed WINDOW_DURATION back from that - not chained from the previous
    digest (see entry 56 for why the watermark-chain design was replaced
    with this simpler fixed lookback). Callers never reason about
    timezones or calendar days at all.

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

    `manually_triggered` (see LOG.md entry 57) is True only when this
    call came from app.py's POST /trigger (the on-demand path), never
    from the scheduled Cloud Run Job - it controls whether the published
    digest issue also gets MANUALLY_TRIGGERED_LABEL alongside
    DIGEST_LABEL.
    """

    window_end = dt.datetime.now(dt.UTC)
    window_start = window_end - WINDOW_DURATION

    digest_labels = [DIGEST_LABEL]
    if manually_triggered:
        digest_labels.append(MANUALLY_TRIGGERED_LABEL)

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
        labels=digest_labels,
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
