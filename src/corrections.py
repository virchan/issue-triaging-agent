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

_ISSUE_REFERENCE_PATTERN = re.compile(r"[\w.-]+/[\w.-]+/(\d+)")


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
            "issue (no owner/repo/number reference found) and were not recorded."
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
    Comments without a parseable owner/repo/number reference on any line
    are not stored as corrections (judgment_id is required) but are
    reported back so nothing is silently dropped without a trace. A
    comment naming several issues (one bullet per issue) produces one
    correction per issue, not one correction for the whole comment.

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
        corrections_by_issue = extract_corrections_by_issue(comment.body)
        if not corrections_by_issue:
            unattributed.append(comment.id)
            continue

        for github_number, text in corrections_by_issue.items():
            judgment_id = get_judgment_id_for_issue_number(
                connection, digest_id, github_number
            )
            if judgment_id is None:
                continue

            save_correction(
                connection,
                judgment_id,
                comment.id,
                text,
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
