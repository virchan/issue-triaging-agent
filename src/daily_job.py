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
from src.pipeline import (
    BACKLOG_CAP,
    PipelineResult,
    fetch_and_judge,
    fetch_and_judge_backlog,
)

# Every poll looks back this far from "now", regardless of when the
# previous run happened - see LOG.md entry 56. Matches the daily 09:00
# PDT cadence: a shorter window would leave a permanent gap between runs.
# Only used when no WIP digest exists (see entry 58) - a WIP digest's own
# window_end is used instead when one does.
WINDOW_DURATION = dt.timedelta(hours=24)

# Attached to every digest issue created - see LOG.md entry 57. Must
# already exist in the shadow repo (created by hand, not by this code).
DIGEST_LABEL = "daily digest"
MANUALLY_TRIGGERED_LABEL = "manually-triggered"
AGENT_TRIGGERED_LABEL = "triggered-by:agent"


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
    LOG.md entries 53/56/58. window_end is always now. window_start
    depends on whether a WIP digest exists - see below.

    Forward half (this window's issues through publishing a new digest)
    runs first, then every previously-published digest not yet marked
    reviewed is checked - capture_corrections is a no-op for any whose
    GitHub issue isn't closed yet, so calling it here is always safe,
    not just when a digest is known to be ready. The same
    get_unreviewed_digests call is reused for both purposes (see entry
    58) - it's fetched once, before this run's own digest exists, so the
    backward pass never tries to check a digest this same call just
    created.

    WIP-digest handling (LOG.md entry 58): if any previously-published
    digest is still unreviewed (its GitHub issue hasn't been closed
    yet), the most recently created one is treated as a WIP digest -
    window_start becomes its window_end (a "what's new since you
    started this" query, capped at BACKLOG_CAP), and backlog catch-up is
    skipped entirely: the point is showing what's new since you opened
    it, not padding with more backlog while you're still mid-review. The
    published digest also gets a reminder line pointing back at it.

    Only when no WIP digest exists does the normal fixed WINDOW_DURATION
    lookback apply, and only then can backlog catch-up run (Phase 8 idea
    A - see LOG.md) if that window finds nothing new: older, still-open
    issues carrying `label`, newest-created first, judged up to
    BACKLOG_CAP regardless of whether each one has been judged before -
    reusing an existing judgment rather than excluding it, so a still-open
    real "Needs Triage" issue never silently disappears from view just
    because it was judged once already (see entry 58). Requires a real
    `label` - there's no well-defined "backlog" without one.

    github_client must be read-only (scikit-learn); shadow_client must
    be authenticated with SHADOW_REPO_TOKEN (shadow repo writes).

    `manually_triggered` (see LOG.md entry 57) is True only when this
    call came from app.py's POST /trigger (the on-demand path), never
    from the scheduled Cloud Run Job - it controls whether the published
    digest gets MANUALLY_TRIGGERED_LABEL or AGENT_TRIGGERED_LABEL
    alongside DIGEST_LABEL.
    """

    unreviewed_digests = get_unreviewed_digests(connection)
    most_recent_wip = unreviewed_digests[-1] if unreviewed_digests else None

    window_end = dt.datetime.now(dt.UTC)
    window_start = (
        most_recent_wip.window_end
        if most_recent_wip is not None
        else window_end - WINDOW_DURATION
    )

    digest_labels = [DIGEST_LABEL]
    digest_labels.append(
        MANUALLY_TRIGGERED_LABEL if manually_triggered else AGENT_TRIGGERED_LABEL
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
        cap=BACKLOG_CAP if most_recent_wip is not None else None,
    )

    backlog_result: PipelineResult | None = None
    backlog_issue_numbers: list[int] = []
    if most_recent_wip is None and pipeline_result.fetched == 0 and label is not None:
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
        wip_digest_issue_number=(
            most_recent_wip.shadow_issue_number if most_recent_wip is not None else None
        ),
    )

    published = publish_digest(
        connection,
        shadow_client,
        digest,
        shadow_owner=shadow_owner,
        shadow_repo=shadow_repo,
    )

    reviews: list[CaptureResult] = []
    for unreviewed in unreviewed_digests:
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
