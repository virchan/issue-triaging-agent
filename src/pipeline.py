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
    get_all_issue_embeddings,
    get_issue_embedding,
    get_recent_reviewed_judgments,
    get_reviewed_judgments_with_embeddings,
    has_judgment,
    save_issue_embedding,
    save_issue_snapshots,
    save_judgment,
    set_possible_duplicate,
)
from src.duplicate_detection import (
    find_possible_duplicate,
    find_similar_reviewed_examples,
    issue_text,
)
from src.embeddings import EMBEDDING_MODEL, IssueEmbedder
from src.gemini_client import GeminiJudge, GeminiResponseError, GeminiUnavailableError
from src.github_client import GitHubClient, GitHubIssue

LOGGER = logging.getLogger(__name__)

# Backlog catch-up caps how many older issues get judged on an idle day
# at once. Bounded well under the Gemini free-tier's 20 requests/day/
# project/model quota, not tied to the operator's real daily review
# capacity (2-3/day, up to 5) - the operator explicitly chose to review
# more of the backlog per idle day than that pace alone would suggest.
BACKLOG_CAP = 15


@dataclass
class PipelineResult:
    """Summary of one fetch-filter-judge-store run."""

    fetched: int
    bot_excluded: int
    judged: int
    already_judged: int
    failures: list[tuple[int, str]] = field(default_factory=list)


def _embed_issue(
    *,
    connection: psycopg.Connection[Any],
    issue_embedder: IssueEmbedder,
    issue: GitHubIssue,
    issue_id: int,
) -> list[float] | None:
    """Embed issue (reusing a stored embedding if one already exists -
    e.g. from the backfill, or a prior judgment). Computed *before*
    judging - the same embedding is used both to
    retrieve semantically similar past judgments as RAG context for
    judge() and, after judging, to find this issue's own
    possible-duplicate suggestion - one embed call serves both, not two.

    Returns None on a real embedding-service failure - callers must
    degrade gracefully (empty similar_examples, no duplicate lookup),
    never fail the judgment itself over this.
    """

    try:
        embedding = get_issue_embedding(connection, issue_id)
        if embedding is None:
            embedding = issue_embedder.embed(issue_text(issue.title, issue.body))
            save_issue_embedding(connection, issue_id, EMBEDDING_MODEL, embedding)
        return embedding
    except GeminiUnavailableError as error:
        LOGGER.warning(f"Embedding failed for issue #{issue.number}: {error}")
        return None


def _record_possible_duplicate(
    *,
    connection: psycopg.Connection[Any],
    embedding: list[float],
    issue: GitHubIssue,
    judgment_id: int,
) -> None:
    """Find the most similar other issue in the pool and record the
    result on this judgment - a ranked suggestion, not a classification,
    always recording the best match found above
    a loose sanity floor, or explicitly "none found" rather than leaving
    the columns unset and ambiguous."""

    candidates = [
        (github_number, vector)
        for _, github_number, vector in get_all_issue_embeddings(connection)
        if github_number != issue.number
    ]
    match = find_possible_duplicate(embedding, candidates)
    github_number, similarity = match if match is not None else (None, None)
    set_possible_duplicate(connection, judgment_id, github_number, similarity)


def _find_similar_examples(
    *,
    connection: psycopg.Connection[Any],
    embedding: list[float],
    issue: GitHubIssue,
) -> list[ReviewedJudgment]:
    """Retrieval-augmented few-shot context for judge(): past reviewed
    judgments retrieved by embedding similarity to
    this specific issue, not recency - a genuinely different signal
    from get_recent_reviewed_judgments, reusing the same embeddings
    built for duplicate detection."""

    candidates = [
        (github_number, vector, reviewed)
        for github_number, vector, reviewed in get_reviewed_judgments_with_embeddings(
            connection
        )
        if github_number != issue.number
    ]
    return find_similar_reviewed_examples(embedding, candidates)


def _judge_and_persist(
    *,
    gemini_judge: GeminiJudge,
    issue_embedder: IssueEmbedder,
    connection: psycopg.Connection[Any],
    issues: list[GitHubIssue],
    ids_by_number: dict[int, int],
    known_labels: list[str],
    recent_examples: list[ReviewedJudgment],
) -> tuple[int, int, list[tuple[int, str]], list[int], list[int]]:
    """Judge each not-yet-judged issue and persist the result.

    Shared by fetch_and_judge and fetch_and_judge_backlog - only how the
    issue list is sourced differs between the two. Returns
    (judged_count, already_judged_count, failures, judged_github_numbers,
    reused_github_numbers) - a caller needs both number lists to know
    every issue that ended up with a judgment this call, whether freshly
    computed or reused from a prior run. Reusing an existing judgment
    rather than re-calling Gemini for it is deliberate.

    Each issue is embedded *before* judging now, not
    after - the embedding is used to retrieve similar_examples for
    judge()'s RAG context, then reused afterward for the possible-
    duplicate lookup, one embed call serving both. An embedding failure
    degrades gracefully (empty similar_examples, no duplicate lookup)
    without ever blocking the judgment itself.
    """

    failures: list[tuple[int, str]] = []
    judged_count = 0
    already_judged_count = 0
    judged_numbers: list[int] = []
    reused_numbers: list[int] = []

    for issue in issues:
        issue_id = ids_by_number[issue.number]
        if has_judgment(connection, issue_id):
            already_judged_count += 1
            reused_numbers.append(issue.number)
            continue

        embedding = _embed_issue(
            connection=connection,
            issue_embedder=issue_embedder,
            issue=issue,
            issue_id=issue_id,
        )
        similar_examples = (
            _find_similar_examples(
                connection=connection, embedding=embedding, issue=issue
            )
            if embedding is not None
            else []
        )

        try:
            judgment = gemini_judge.judge(
                title=issue.title,
                body=issue.body,
                known_labels=known_labels,
                recent_examples=recent_examples,
                similar_examples=similar_examples,
            )
            judgment_id = save_judgment(connection, issue_id, judgment)
            judged_count += 1
            judged_numbers.append(issue.number)
        except (GeminiUnavailableError, GeminiResponseError) as error:
            LOGGER.warning(f"Judgment failed for issue #{issue.number}: {error}")
            failures.append((issue.number, str(error)))
            continue

        if embedding is not None:
            _record_possible_duplicate(
                connection=connection,
                embedding=embedding,
                issue=issue,
                judgment_id=judgment_id,
            )

    return judged_count, already_judged_count, failures, judged_numbers, reused_numbers


def fetch_and_judge(
    *,
    github_client: GitHubClient,
    gemini_judge: GeminiJudge,
    issue_embedder: IssueEmbedder,
    connection: psycopg.Connection[Any],
    owner: str,
    repo: str,
    window_start: dt.datetime,
    window_end: dt.datetime,
    label: str | None = None,
    cap: int | None = None,
) -> PipelineResult:
    """Fetch issues created in [window_start, window_end), filter, judge,
    and persist everything.

    If label is given (e.g. "Needs Triage"), only issues carrying that
    label are fetched in the first place - issues a maintainer has
    already triaged never enter the pipeline at all, rather than being
    fetched and then discarded.

    `cap`, if given, judges at most that many of the found issues - used
    for the WIP-digest "what's new since digest X" query, bounded like
    backlog catch-up is. None (the default) judges everything found - the
    normal 24h window rarely finds enough issues for this label to need
    bounding.

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
    to_judge = non_bot[:cap] if cap is not None else non_bot

    ids_by_number = save_issue_snapshots(connection, owner, repo, to_judge, bot)

    known_labels = github_client.fetch_labels(owner, repo)
    recent_examples = get_recent_reviewed_judgments(connection)

    judged_count, already_judged_count, failures, _, _ = _judge_and_persist(
        gemini_judge=gemini_judge,
        issue_embedder=issue_embedder,
        connection=connection,
        issues=to_judge,
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
    issue_embedder: IssueEmbedder,
    connection: psycopg.Connection[Any],
    owner: str,
    repo: str,
    label: str,
    cap: int = BACKLOG_CAP,
) -> tuple[PipelineResult, list[int]]:
    """When a poll finds nothing new, look for open issues still carrying
    `label`, newest-created first, and judge up to `cap` of them.

    Candidates are considered regardless of whether they've been judged
    before - an issue already judged in a prior run is *reused* (its
    existing judgment is shown, no new Gemini call) rather than excluded,
    so a still-open real "Needs Triage" issue never silently disappears
    from the digest just because we computed a judgment for it once.

    Backlog issues fall outside any time window by definition, so they
    can never be produced by fetch_and_judge - this is a genuinely
    separate sourcing path, not a variant of the window-based one.
    Returns the usual PipelineResult, plus every github_number that ended
    up with a judgment this call (freshly judged or reused) - the digest
    needs those explicitly, since get_judged_issues_in_window can't find
    issues outside its window.
    """

    start = time.monotonic()
    candidates = github_client.fetch_open_issues_with_label(owner, repo, label)
    non_bot, bot = partition_bot_issues(candidates)
    to_judge = non_bot[:cap]

    ids_by_number = save_issue_snapshots(connection, owner, repo, to_judge, bot)

    known_labels = github_client.fetch_labels(owner, repo)
    recent_examples = get_recent_reviewed_judgments(connection)

    judged_count, already_judged_count, failures, judged_numbers, reused_numbers = (
        _judge_and_persist(
            gemini_judge=gemini_judge,
            issue_embedder=issue_embedder,
            connection=connection,
            issues=to_judge,
            ids_by_number=ids_by_number,
            known_labels=known_labels,
            recent_examples=recent_examples,
        )
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
            "reused_existing_judgment": already_judged_count,
            "capped_out": len(non_bot) - len(to_judge),
            "duration_seconds": time.monotonic() - start,
            "judged": result.judged,
            "failure_count": len(result.failures),
        },
    )

    return result, judged_numbers + reused_numbers
