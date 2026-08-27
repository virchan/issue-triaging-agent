from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any

import psycopg

from src.corrections import CaptureResult, capture_corrections
from src.db import (
    UnreviewedDigest,
    get_recent_reviewed_judgments,
    get_unreviewed_digests,
    prune_old_issue_embeddings,
)
from src.digest import OPERATOR_TIMEZONE, DigestContent, build_digest, publish_digest
from src.embeddings import IssueEmbedder
from src.gemini_client import GeminiJudge
from src.github_client import GitHubClient
from src.pipeline import (
    BACKLOG_CAP,
    PipelineResult,
    fetch_and_judge,
    fetch_and_judge_backlog,
)

# How far back a stored embedding stays useful as a duplicate-candidate -
# matches scripts/backfill_issue_embeddings.py's own backfill horizon
# (LOG.md entry 75: the operator's real adjudicated examples showed
# duplicates come from scikit-learn's broader history, not just recently
# judged issues - but "broader" still isn't "forever"). Applied as a
# rolling window, not a one-time cutoff, so storage stays bounded
# indefinitely rather than growing forever as new issues get embedded
# each day.
EMBEDDING_RETENTION = dt.timedelta(days=365 * 2)

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

LOGGER = logging.getLogger(__name__)


@dataclass
class DailyCycleResult:
    """Summary of one full state-machine pass: forward (fetch through
    publish) and backward (correction capture on previously-published,
    now-possibly-closed digests)."""

    pipeline: PipelineResult
    digest: DigestContent | None
    """None only when the same-day-duplicate guard skipped publishing
    entirely - see run_daily_cycle."""
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
    issue_embedder: IssueEmbedder,
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
    tries to check a digest this same call just created. A correction can
    now trigger an actual re-judge (see src.corrections), so this pass
    also needs known_labels/recent_examples - fetched once for the whole
    pass, not once per digest.

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

    issue_embedder computes the duplicate-candidate suggestion for each
    freshly judged issue (see src.pipeline._find_and_record_possible_duplicate)
    and, every cycle regardless of what else happened, its stored
    embeddings get pruned to a rolling EMBEDDING_RETENTION window so
    storage doesn't grow forever.

    `manually_triggered` is True only when this call came from app.py's
    POST /trigger (the on-demand path), never from the scheduled Cloud
    Run Job - it controls whether the published digest also gets
    MANUALLY_TRIGGERED_LABEL alongside DIGEST_LABEL (no extra label for
    the normal scheduled path - LOG.md entry 98: "triggered-by:agent"
    and "testing" were both dropped once the project moved past needing
    to flag every digest as in-progress/experimental).

    Returns with digest=None, published=None (no GitHub issue created,
    no new digests row) when a same-day WIP digest already exists and
    this run's window found nothing new - see the same-day duplicate
    guard below.
    """

    unreviewed_digests = get_unreviewed_digests(connection)

    reviews: list[CaptureResult] = []
    still_open_digests: list[UnreviewedDigest] = []
    if unreviewed_digests:
        # Fetched once for the whole backward pass, not once per digest -
        # a correction-triggered re-judge (see src.corrections) needs the
        # same real, current labels and few-shot context every judge()
        # call already uses.
        known_labels = github_client.fetch_labels(source_owner, source_repo)
        recent_examples = get_recent_reviewed_judgments(connection)

        for unreviewed in unreviewed_digests:
            review = capture_corrections(
                connection,
                shadow_client,
                gemini_judge,
                digest_id=unreviewed.digest_id,
                digest_window_end=unreviewed.window_end,
                shadow_owner=unreviewed.shadow_owner,
                shadow_repo=unreviewed.shadow_repo,
                shadow_issue_number=unreviewed.shadow_issue_number,
                known_labels=known_labels,
                recent_examples=recent_examples,
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
    if manually_triggered:
        digest_labels.append(MANUALLY_TRIGGERED_LABEL)

    pipeline_result = fetch_and_judge(
        github_client=github_client,
        gemini_judge=gemini_judge,
        issue_embedder=issue_embedder,
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
            issue_embedder=issue_embedder,
            connection=connection,
            owner=source_owner,
            repo=source_repo,
            label=label,
        )

    # Keeps issue_embeddings storage bounded to a rolling window rather
    # than growing forever - safe to run every cycle regardless of
    # whether anything was actually judged this time.
    prune_old_issue_embeddings(connection, window_end - EMBEDDING_RETENTION)

    # Same-day duplicate guard: a WIP digest already published earlier
    # today, plus this run finding nothing new, means a fresh digest
    # would only repeat the exact same "nothing new since #N" reminder
    # the still-open one already carries - real, not hypothetical (a
    # Cloud Scheduler double-fire produced exactly this on 2026-08-24,
    # two "nothing new" issues 32 seconds apart). Compares calendar dates
    # in OPERATOR_TIMEZONE, matching build_digest's own display_date -
    # a WIP digest from *yesterday* still gets today's normal reminder
    # digest; only a same-day repeat is skipped.
    if (
        most_recent_wip is not None
        and pipeline_result.fetched == 0
        and most_recent_wip.window_end.astimezone(OPERATOR_TIMEZONE).date()
        == window_end.astimezone(OPERATOR_TIMEZONE).date()
    ):
        LOGGER.info(
            f"Skipping digest: {shadow_owner}/{shadow_repo}#"
            f"{most_recent_wip.shadow_issue_number} already covers today "
            "and this run found nothing new",
            extra={
                "event": "daily_cycle_skip",
                "wip_digest_issue_number": most_recent_wip.shadow_issue_number,
                "reason": "duplicate_same_day_no_new_issues",
            },
        )
        return DailyCycleResult(
            pipeline=pipeline_result,
            digest=None,
            published=None,
            reviews=reviews,
            backlog=backlog_result,
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
