from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import psycopg

from src.corrections import CaptureResult, capture_corrections
from src.db import UnreviewedDigest, get_unreviewed_digests
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
# previous run happened. Matches the daily 09:00 PDT cadence: a shorter
# window would leave a permanent gap between runs. Only used when no WIP
# digest exists - a WIP digest's own window_end is used instead when one
# does.
WINDOW_DURATION = dt.timedelta(hours=24)

# Attached to every digest issue created. Must already exist in the
# shadow repo (created by hand, not by this code).
DIGEST_LABEL = "daily digest"
MANUALLY_TRIGGERED_LABEL = "manually-triggered"
AGENT_TRIGGERED_LABEL = "triggered-by:agent"
TESTING_LABEL = "testing"

# While the system is still being tested rather than trusted as
# production output, every digest also gets TESTING_LABEL so a reader
# can't mistake in-progress testing activity for finished output. Flip
# to False (a one-line change) once the operator considers it
# production-ready.
STILL_TESTING = True


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
    ran."""


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

    The poll window is computed here, not passed in by the caller.
    window_end is always now. window_start depends on whether a WIP
    digest exists - see below.

    Backward half (correction capture on previously-published digests)
    runs first, not last: running it last would mean a digest the
    operator had already closed before this run even started is still
    treated as "still open" for the WIP check below, since our own DB
    hasn't caught up yet within this same call. capture_corrections is a
    no-op for any whose GitHub issue isn't closed yet, so calling it
    unconditionally here is always safe. get_unreviewed_digests is called
    once, before this run's own digest exists, so the backward pass never
    tries to check a digest this same call just created.

    WIP-digest handling: of the digests that are genuinely still open
    *after* the backward pass just ran (per each capture_corrections
    call's live GitHub check, not the pre-pass DB snapshot), the most
    recently created one is treated as a WIP digest - window_start
    becomes its window_end (a "what's new since you started this" query,
    capped at BACKLOG_CAP), and backlog catch-up is skipped entirely: the
    point is showing what's new since you opened it, not padding with
    more backlog while you're still mid-review. The published digest also
    gets a reminder line pointing back at it.

    Only when no WIP digest exists does the normal fixed WINDOW_DURATION
    lookback apply, and only then can backlog catch-up run if that window
    finds nothing new: older, still-open issues carrying `label`,
    newest-created first, judged up to BACKLOG_CAP regardless of whether
    each one has been judged before - reusing an existing judgment rather
    than excluding it, so a still-open real "Needs Triage" issue never
    silently disappears from view just because it was judged once
    already. Requires a real `label` - there's no well-defined "backlog"
    without one.

    github_client must be read-only (scikit-learn); shadow_client must
    be authenticated with SHADOW_REPO_TOKEN (shadow repo writes).

    `manually_triggered` is True only when this call came from app.py's
    POST /trigger (the on-demand path), never from the scheduled Cloud
    Run Job - it controls whether the published digest gets
    MANUALLY_TRIGGERED_LABEL or AGENT_TRIGGERED_LABEL alongside
    DIGEST_LABEL.
    """

    unreviewed_digests = get_unreviewed_digests(connection)

    reviews: list[CaptureResult] = []
    still_open_digests: list[UnreviewedDigest] = []
    for unreviewed in unreviewed_digests:
        review = capture_corrections(
            connection,
            shadow_client,
            digest_id=unreviewed.digest_id,
            shadow_owner=unreviewed.shadow_owner,
            shadow_repo=unreviewed.shadow_repo,
            shadow_issue_number=unreviewed.shadow_issue_number,
        )
        reviews.append(review)
        if review.issue_still_open:
            still_open_digests.append(unreviewed)

    most_recent_wip = still_open_digests[-1] if still_open_digests else None

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
    if STILL_TESTING:
        digest_labels.append(TESTING_LABEL)

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

    return DailyCycleResult(
        pipeline=pipeline_result,
        digest=digest,
        published=published,
        reviews=reviews,
        backlog=backlog_result,
    )
