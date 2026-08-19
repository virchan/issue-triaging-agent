from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import psycopg

from src.db import (
    get_judgment_id_for_issue_number,
    is_digest_reviewed,
    mark_digest_reviewed,
    save_correction,
)
from src.github_client import GitHubClient

_ISSUE_REFERENCE_PATTERN = re.compile(r"#(\d+)")


def extract_referenced_issue_number(comment_body: str) -> int | None:
    """Extract the first #NNN issue reference from a comment, if any.

    A digest issue aggregates multiple judged issues into one comment
    thread, so a correction comment needs to say which issue it's about -
    the natural way to do that is referencing it the same way the digest
    itself does (e.g. "#34649 is not about linear model, it's about
    SVC"). A comment with no recognizable reference can't be attributed
    to a specific judgment, since correction.judgment_id is required.
    """

    match = _ISSUE_REFERENCE_PATTERN.search(comment_body)
    return int(match.group(1)) if match else None


@dataclass
class CaptureResult:
    """Summary of one correction-capture attempt."""

    issue_still_open: bool
    already_reviewed: bool
    captured: int = 0
    unattributed_comment_ids: list[int] = field(default_factory=list)


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

    if result.unattributed_comment_ids:
        count = len(result.unattributed_comment_ids)
        plural = "s" if count != 1 else ""
        lines.append(
            f"Note: {count} comment{plural} could not be matched to a specific "
            "issue (no #NNN reference found) and were not recorded."
        )

    lines.append("This digest is now marked reviewed.")
    return "\n\n".join(lines)


def capture_corrections(
    connection: psycopg.Connection[Any],
    shadow_client: GitHubClient,
    *,
    digest_id: int,
    shadow_owner: str,
    shadow_repo: str,
    shadow_issue_number: int,
) -> CaptureResult:
    """Capture corrections from a closed digest issue's comments.

    Idempotent: does nothing if the digest was already marked reviewed,
    or if the issue isn't closed yet (review still in progress).
    Comments without a parseable #NNN reference are not stored as
    corrections (judgment_id is required) but are reported back so
    nothing is silently dropped without a trace.

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
    unattributed: list[int] = []
    for comment in comments:
        issue_number = extract_referenced_issue_number(comment.body)
        judgment_id = (
            get_judgment_id_for_issue_number(connection, digest_id, issue_number)
            if issue_number is not None
            else None
        )

        if judgment_id is None:
            unattributed.append(comment.id)
            continue

        save_correction(
            connection,
            judgment_id,
            comment.id,
            comment.body,
            comment.created_at,
        )
        captured += 1

    result = CaptureResult(
        issue_still_open=False,
        already_reviewed=False,
        captured=captured,
        unattributed_comment_ids=unattributed,
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
