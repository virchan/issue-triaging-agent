from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import psycopg

from src.db import (
    ReviewedJudgment,
    get_authoritative_correction_digest,
    get_judgment_id_for_issue_number,
    get_rejudge_context,
    is_digest_reviewed,
    mark_corrections_superseded,
    mark_digest_reviewed,
    save_correction,
    update_judgment,
)
from src.gemini_client import GeminiJudge, GeminiResponseError, GeminiUnavailableError
from src.github_client import GitHubClient

LOGGER = logging.getLogger(__name__)

_ISSUE_REFERENCE_PATTERN = re.compile(r"[\w.-]+/[\w.-]+/(\d+)")

# How many corrections get an actual re-judge (a fresh Gemini call) per
# capture_corrections call. Corrections beyond this are still recorded
# as authoritative, just without a live re-judge this round - a rare
# edge case at this operator's real correction volume (2-3/day, up to
# 5), not worth a retry mechanism for.
REJUDGE_CAP = 5


def extract_corrections_by_issue(comment_body: str) -> dict[int, str]:
    """Split a comment into lines and extract every owner/repo/number
    issue reference, one correction per referenced issue.

    A digest comment often corrects several issues at once - one bullet
    per issue - so this returns a dict, not a single number: each line
    naming a specific issue (e.g. "`scikit-learn/scikit-learn/34649` is
    not about linear model, it's about SVC") becomes that issue's
    correction text, matching the reference format the digest itself
    renders. Lines naming the same issue twice within one comment are
    combined into a single correction. A line with no recognizable
    reference isn't attributed to any judgment (judgment_id is required)
    and is dropped rather than silently merged into an unrelated issue's
    correction.
    """

    corrections: dict[int, list[str]] = {}
    for line in comment_body.splitlines():
        match = _ISSUE_REFERENCE_PATTERN.search(line)
        if match is None:
            continue
        github_number = int(match.group(1))
        corrections.setdefault(github_number, []).append(line.strip())

    return {number: "\n".join(lines) for number, lines in corrections.items()}


@dataclass
class SupersededCorrection:
    """A correction that was recognized but not applied, because a more
    recently created thread already holds the authoritative correction
    for that same issue."""

    github_number: int
    authoritative_shadow_issue_number: int


@dataclass
class CaptureResult:
    """Summary of one correction-capture attempt."""

    issue_still_open: bool
    already_reviewed: bool
    captured: int = 0
    unattributed_comment_ids: list[int] = field(default_factory=list)
    superseded: list[SupersededCorrection] = field(default_factory=list)
    capped: int = 0
    """Corrections recorded as authoritative but not re-judged this call
    because REJUDGE_CAP was reached."""
    rejudge_failures: list[tuple[int, str]] = field(default_factory=list)
    """(github_number, error message) for corrections that were captured
    but whose re-judge call failed - the correction stays on record even
    though the live judgment wasn't updated."""


def format_acknowledgment(result: CaptureResult) -> str:
    """Build the single acknowledgment reply posted once review is captured.

    A deliberately small, one-shot response - not an interactive chat.
    The fuller conversational version was scoped out for now.
    """

    lines: list[str] = []
    if result.captured:
        plural = "s" if result.captured != 1 else ""
        lines.append(f"Recorded {result.captured} correction{plural}. Thank you!")
    else:
        lines.append("Recorded — no corrections needed for this digest.")

    for item in result.superseded:
        lines.append(
            f"Note: your note about `scikit-learn/scikit-learn/{item.github_number}` "
            "wasn't applied here - a more recent correction for that issue "
            f"already exists on #{item.authoritative_shadow_issue_number}. "
            "See that thread instead."
        )

    if result.unattributed_comment_ids:
        count = len(result.unattributed_comment_ids)
        plural = "s" if count != 1 else ""
        lines.append(
            f"Note: {count} comment{plural} could not be matched to a specific "
            "issue (no owner/repo/number reference found) and were not recorded."
        )

    if result.capped:
        plural = "s" if result.capped != 1 else ""
        lines.append(
            f"Note: {result.capped} correction{plural} were recorded but not "
            "re-judged this round (today's re-judge limit was reached)."
        )

    if result.rejudge_failures:
        count = len(result.rejudge_failures)
        plural = "s" if count != 1 else ""
        lines.append(
            f"Note: {count} correction{plural} were recorded but the re-judge "
            "call failed - the correction is on file, the judgment itself "
            "wasn't updated."
        )

    lines.append("This digest is now marked reviewed.")
    return "\n\n".join(lines)


def capture_corrections(
    connection: psycopg.Connection[Any],
    shadow_client: GitHubClient,
    gemini_judge: GeminiJudge,
    *,
    digest_id: int,
    digest_window_end: dt.datetime,
    shadow_owner: str,
    shadow_repo: str,
    shadow_issue_number: int,
    known_labels: list[str],
    recent_examples: list[ReviewedJudgment] | None = None,
) -> CaptureResult:
    """Capture corrections from a closed digest issue's comments.

    Idempotent: does nothing if the digest was already marked reviewed,
    or if the issue isn't closed yet (review still in progress).
    Comments without a parseable owner/repo/number reference on any line
    are not stored as corrections (judgment_id is required) but are
    reported back so nothing is silently dropped without a trace. A
    comment naming several issues (one bullet per issue) produces one
    correction per issue, not one correction for the whole comment.

    A referenced issue can already carry a correction from a different,
    more recently created thread (the same real issue re-surfaced via
    backlog catch-up while an older thread referencing it was still
    open). digest_window_end decides precedence: if a correction already
    exists on a thread whose window_end is later than this one's, this
    comment's correction is recorded but marked superseded - not used to
    re-judge, not surfaced as few-shot context - rather than silently
    dropped or allowed to overwrite a newer thread's correction. Otherwise
    this becomes the new authoritative correction: any previous
    authoritative correction for the same judgment is marked superseded,
    and a fresh Gemini call (judge_with_correction) revises the judgment
    in place, up to REJUDGE_CAP re-judges per call.

    Posts a single acknowledgment comment (via shadow_client, so it
    appears as virchan-mirror) once processing is done. This is
    intentionally a one-shot reply, not an interactive back-and-forth.
    """

    if is_digest_reviewed(connection, digest_id):
        return CaptureResult(issue_still_open=False, already_reviewed=True)

    state = shadow_client.get_issue_state(
        shadow_owner, shadow_repo, shadow_issue_number
    )
    if state != "closed":
        return CaptureResult(issue_still_open=True, already_reviewed=False)

    comments = shadow_client.fetch_issue_comments(
        shadow_owner, shadow_repo, shadow_issue_number
    )

    captured = 0
    rejudges = 0
    unattributed: list[int] = []
    superseded: list[SupersededCorrection] = []
    capped = 0
    rejudge_failures: list[tuple[int, str]] = []

    for comment in comments:
        corrections_by_issue = extract_corrections_by_issue(comment.body)
        if not corrections_by_issue:
            unattributed.append(comment.id)
            continue

        for github_number, text in corrections_by_issue.items():
            judgment_id = get_judgment_id_for_issue_number(connection, github_number)
            if judgment_id is None:
                continue

            existing = get_authoritative_correction_digest(connection, judgment_id)
            if existing is not None and existing[1] > digest_window_end:
                authoritative_issue_number, _ = existing
                save_correction(
                    connection,
                    judgment_id,
                    digest_id,
                    comment.id,
                    text,
                    comment.created_at,
                    superseded=True,
                )
                superseded.append(
                    SupersededCorrection(github_number, authoritative_issue_number)
                )
                continue

            mark_corrections_superseded(connection, judgment_id)
            save_correction(
                connection,
                judgment_id,
                digest_id,
                comment.id,
                text,
                comment.created_at,
                superseded=False,
            )
            captured += 1

            if rejudges >= REJUDGE_CAP:
                capped += 1
                continue

            context = get_rejudge_context(connection, judgment_id)
            if context is None:
                continue

            try:
                revised = gemini_judge.judge_with_correction(
                    title=context.title,
                    body=context.body,
                    previous_judgment=context.judgment,
                    correction_text=text,
                    known_labels=known_labels,
                    recent_examples=recent_examples,
                )
                update_judgment(connection, judgment_id, revised)
                rejudges += 1
            except (GeminiUnavailableError, GeminiResponseError) as error:
                LOGGER.warning(f"Re-judge failed for issue #{github_number}: {error}")
                rejudge_failures.append((github_number, str(error)))

    result = CaptureResult(
        issue_still_open=False,
        already_reviewed=False,
        captured=captured,
        unattributed_comment_ids=unattributed,
        superseded=superseded,
        capped=capped,
        rejudge_failures=rejudge_failures,
    )

    # Posted before mark_digest_reviewed's commit below: if posting fails,
    # nothing here has been committed yet, so a retry safely redoes the
    # whole cycle rather than silently never retrying a lost
    # acknowledgment (is_digest_reviewed would otherwise short-circuit
    # future attempts once the state was already committed).
    shadow_client.create_issue_comment(
        shadow_owner,
        shadow_repo,
        shadow_issue_number,
        format_acknowledgment(result),
    )

    mark_digest_reviewed(connection, digest_id)

    return result
