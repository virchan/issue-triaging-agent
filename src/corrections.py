from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import psycopg

from src.db import (
    TRACKED_CORRECTION_FIELDS,
    ReviewedJudgment,
    get_authoritative_correction_digest,
    get_judgment_id_for_issue_number,
    get_rejudge_context,
    is_digest_reviewed,
    mark_corrections_superseded,
    mark_digest_reviewed,
    save_correction,
    set_correction_changed_fields,
    update_judgment,
)
from src.gemini_client import GeminiJudge, GeminiResponseError, GeminiUnavailableError
from src.github_client import GitHubClient
from src.rendering import render_template

LOGGER = logging.getLogger(__name__)

# Known short names the operator might casually use for scikit-learn/
# scikit-learn instead of spelling out the full owner/repo. An allowlist,
# not open-ended fuzzy matching: a pattern loose enough to guess at any
# "word/number" is exactly what caused the entry 63 incident (a stray
# "#13" elsewhere in a comment got misattributed). Anything not on this
# list, or not in one of the recognized forms below, isn't matched - it
# falls through to extract_corrections_by_issue's existing
# unattributed-comment reporting rather than being guessed at. Add more
# aliases here as they come up in real use.
_KNOWN_REPO_ALIASES = ("scikit-learn", "sklearn")

# The one real GitHub org/repo this agent tracks judgments for - the only
# owner/repo#number reference _SOURCE_REPO_HASH_PATTERN below recognizes.
# A reference naming any *other* repo (e.g. a real duplicate that lives in
# uxlfoundation/scikit-learn-intelex, LOG.md entry 85) has no judgment to
# attribute a correction to here, so it's correctly left unmatched -
# same allowlist discipline as _KNOWN_REPO_ALIASES, not a gap to widen.
_SOURCE_REPO_HASH_PATTERN = re.compile(
    r"scikit-learn/scikit-learn#(\d+)", re.IGNORECASE
)

# Tried in order, most specific first - a line is matched by the first
# pattern that hits, not all of them (multiple patterns can, coincidentally,
# both match somewhere in a full "owner/repo/number" reference, since
# "repo/number" is a literal substring of it - trying the specific form
# first keeps which pattern actually matched unambiguous rather than
# relying on that overlap always extracting the same number).
_REFERENCE_PATTERNS = [
    re.compile(
        r"[\w.-]+/[\w.-]+/(\d+)"
    ),  # owner/repo/number, e.g. scikit-learn/scikit-learn/34649
    _SOURCE_REPO_HASH_PATTERN,  # GitHub's real autolink form, e.g. scikit-learn/scikit-learn#34649
    re.compile(
        r"\b(?:"
        + "|".join(re.escape(alias) for alias in _KNOWN_REPO_ALIASES)
        + r")[/#](\d+)\b",
        re.IGNORECASE,
    ),  # known short alias/number or alias#number, e.g. scikit-learn/34649,
    # sklearn/34649, or scikit-learn#34649 (LOG.md entry 94 - real usage
    # showed the operator writing the alias with "#" too, not just "/").
]

# Bare "#NNN" is deliberately NOT matched here, even though it's a
# tempting shorthand: it already means something else in this codebase -
# a real, clickable same-repo reference to another digest thread (see
# digest.py's wip_digest_issue_number reminder line and corrections.py's
# SupersededCorrection message). Recognizing it here too would silently
# reintroduce the exact ambiguity entry 63 fixed, just from the other
# direction. _SOURCE_REPO_HASH_PATTERN above is unambiguous because it
# requires the literal "scikit-learn/scikit-learn" prefix before the "#" -
# a standalone "#NNN" never matches it.

# How many corrections get an actual re-judge (a fresh Gemini call) per
# capture_corrections call. Corrections beyond this are still recorded
# as authoritative, just without a live re-judge this round - a rare
# edge case at this operator's real correction volume (2-3/day, up to
# 5), not worth a retry mechanism for.
REJUDGE_CAP = 5


_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(https?://[^)]*\)")


def _strip_markdown_link_urls(line: str) -> str:
    """Unwrap `[text](url)` down to just `text` before reference-matching.

    LOG.md entry 92: a real correction embedded a markdown link whose
    target was `https://redirect.github.com/uxlfoundation/scikit-learn-intelex/issues/3377`
    - a URL shaped exactly like `owner/repo/issues/number`, which the
    (already loose) owner/repo/number pattern matched, misattributing
    the whole line to a phantom "issue #3377". A URL is never something
    a human typed as a reference - it's incidental to a link's
    structure - so no reference pattern should ever see it. Only used
    for matching; the correction text actually stored is the original,
    unstripped line.
    """

    return _MARKDOWN_LINK_PATTERN.sub(r"\1", line)


def _match_issue_reference(line: str) -> int | None:
    """Try each pattern in _REFERENCE_PATTERNS, in order, returning the
    first match's github_number - None if the line matches none of them.
    """

    line = _strip_markdown_link_urls(line)
    for pattern in _REFERENCE_PATTERNS:
        match = pattern.search(line)
        if match is not None:
            return int(match.group(1))
    return None


def extract_corrections_by_issue(comment_body: str) -> dict[int, str]:
    """Split a comment into lines and extract every recognizable issue
    reference (see _REFERENCE_PATTERNS), one correction per referenced
    issue.

    A digest comment often corrects several issues at once - one bullet
    per issue - so this returns a dict, not a single number: each line
    naming a specific issue (e.g. "`scikit-learn/scikit-learn/34649` is
    not about linear model, it's about SVC", GitHub's own real autolink
    form "scikit-learn/scikit-learn#34649", or the shorter
    "scikit-learn/34649"/"sklearn/34649"/"scikit-learn#34649") becomes
    that issue's correction text. Lines naming the same issue twice
    within one comment are combined into a single correction. A line
    with no recognizable reference isn't attributed to any judgment
    (judgment_id is required)
    and is dropped rather than silently merged into an unrelated issue's
    correction - this includes a line naming an issue in a *different*
    repo (e.g. "the real duplicate is uxlfoundation/scikit-learn-intelex
    #3377"), since there's no judgment here to attribute it to.
    """

    corrections: dict[int, list[str]] = {}
    for line in comment_body.splitlines():
        github_number = _match_issue_reference(line)
        if github_number is None:
            continue
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
class AppliedCorrection:
    """One correction recorded as authoritative this round - surfaced in
    the acknowledgment so the reviewer can verify what was actually
    captured without waiting for the next digest or checking the DB
    directly. A single entry can cover multiple lines of the original
    comment: extract_corrections_by_issue combines every line about the
    same issue into one correction_text before this is built."""

    github_number: int
    correction_text: str
    new_label: str | None = None
    """The revised suggested_label, set only when a live re-judge
    succeeded this round. None when the correction was recorded but not
    (yet) re-judged - see CaptureResult.capped/rejudge_failures for why."""


@dataclass
class CaptureResult:
    """Summary of one correction-capture attempt."""

    issue_still_open: bool
    already_reviewed: bool
    captured: int = 0
    applied: list[AppliedCorrection] = field(default_factory=list)
    """One entry per captured correction (len == captured), in the order
    encountered - what the acknowledgment's collapsed detail section
    lists."""
    unattributed_comment_ids: list[int] = field(default_factory=list)
    unmatched_references: list[int] = field(default_factory=list)
    """github_numbers that parsed as a valid reference (see
    _REFERENCE_PATTERNS) but matched no issue this agent has ever
    judged - distinct from unattributed_comment_ids (no reference found
    at all). Without tracking this separately, a correction referencing
    an unjudged issue (a typo, or a real issue the agent hasn't surfaced
    yet) would be silently dropped with no trace anywhere - see LOG.md
    entry 72."""
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

    applied = [
        {
            "reference": f"scikit-learn/scikit-learn#{item.github_number}",
            "correction_text": item.correction_text,
            "new_label": item.new_label,
        }
        for item in result.applied
    ]
    superseded = [
        {
            "github_number": item.github_number,
            "authoritative_shadow_issue_number": item.authoritative_shadow_issue_number,
        }
        for item in result.superseded
    ]

    return render_template(
        "correction-acknowledgement.md.jinja",
        captured=result.captured,
        applied=applied,
        superseded=superseded,
        unattributed_count=len(result.unattributed_comment_ids),
        unmatched_references=result.unmatched_references,
        capped=result.capped,
        rejudge_failure_count=len(result.rejudge_failures),
    )


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
    or if the issue isn't closed yet (review still in progress). A
    comment naming several issues (one bullet per issue) produces one
    correction per issue, not one correction for the whole comment.

    Two distinct ways a correction can go unrecorded, both reported back
    so nothing is silently dropped without a trace (see LOG.md entry 72):
    a comment with no parseable reference on any line at all
    (unattributed_comment_ids), and a comment with a syntactically valid
    reference that matches no issue this agent has ever judged - a typo,
    or a real issue not yet surfaced (unmatched_references). judgment_id
    is required either way, but these are different failures worth
    telling the operator apart, not one undifferentiated "didn't work."

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
    in place, up to REJUDGE_CAP re-judges per call. Each successful
    re-judge also records which of TRACKED_CORRECTION_FIELDS actually
    changed (see set_correction_changed_fields) - Phase 8 evidence for
    which parts of a judgment corrections tend to be about, computed by
    diffing the judgment's pre- and post-correction values while both are
    still in hand, not inferred from the correction's free text.

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
    applied: list[AppliedCorrection] = []
    unattributed: list[int] = []
    unmatched: list[int] = []
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
                unmatched.append(github_number)
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
            correction_id = save_correction(
                connection,
                judgment_id,
                digest_id,
                comment.id,
                text,
                comment.created_at,
                superseded=False,
            )
            captured += 1
            applied_entry = AppliedCorrection(github_number, text)
            applied.append(applied_entry)

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
                changed_fields = [
                    name
                    for name in TRACKED_CORRECTION_FIELDS
                    if getattr(context.judgment, name) != getattr(revised, name)
                ]
                set_correction_changed_fields(connection, correction_id, changed_fields)
                applied_entry.new_label = revised.suggested_label
                rejudges += 1
            except (GeminiUnavailableError, GeminiResponseError) as error:
                LOGGER.warning(f"Re-judge failed for issue #{github_number}: {error}")
                rejudge_failures.append((github_number, str(error)))

    result = CaptureResult(
        issue_still_open=False,
        already_reviewed=False,
        captured=captured,
        applied=applied,
        unattributed_comment_ids=unattributed,
        unmatched_references=unmatched,
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
